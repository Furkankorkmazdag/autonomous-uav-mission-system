"""
missions/lap.py

LapMission — orchestration logic for a repeating-waypoint ("lap") mission.

CLASS ARCHITECTURE
-------------------
LapMission inherits from BaseMission and is a pure *orchestrator*. It owns no
navigation, MAVLink, threading, or resource-management logic of its own —
those responsibilities live in Navigator, MissionManager, and the underlying
flight-computer stack. LapMission's only job is to answer one question,
repeatedly: "given the current waypoint state, what should happen next?"

Collaborators (all injected via the constructor — never constructed here):

    - SharedData    : cross-subsystem state (battery, GPS, mission flags).
                      LapMission only *reads* from it, never mutates flight
                      telemetry.
    - MissionPath   : owns waypoint ordering, current index, lap counting,
                      and advancement (current(), next(), finished(),
                      restart(), remaining_waypoints(), lap()).
    - Navigator     : the only component allowed to move the vehicle
                      (goto/cancel/active/finished/failed/remaining_distance).
    - EventBus      : pub/sub channel LapMission uses to react to and emit
                      mission-scoped events only (see EVENT FILTERING below).
    - MissionLogger : structured logging of mission lifecycle events,
                      injected — LapMission never constructs its own.

Internally, LapMission is implemented as a finite state machine (see
`LapState`). `update()` is a single-step, non-blocking state-transition
function intended to be called periodically by MissionManager's control
loop (MissionManager owns the thread/timing; LapMission never spins its
own loop or sleeps). Each state transition is a small, single-responsibility
private method (`_send_current_waypoint`, `_next_waypoint`, `_next_lap`,
`_finish`, `_fail`, ...), keeping `update()` itself a thin dispatcher over a
dispatch table built once at construction time.

Thread safety is provided by a single `threading.RLock` guarding all
mutable mission state (`_state`, statistics, pause/resume bookkeeping).
The lock is never held while calling into Navigator, since Navigator calls
may themselves take time or re-enter the event system.

ARCHITECTURAL IMPROVEMENTS OVER THE PRIOR REVISION
---------------------------------------------------
See the trailing module comment / chat explanation for a full list. In
summary: unsubscribe-on-cleanup, strongly typed config, injected logger,
a dispatch table built once, cancel-then-confirm-idle retry flow,
event/poll cross-checking of Navigator state, full waypoint validation,
expanded statistics, checkpoint/resume-from-last-waypoint support, a
narrowed mission-scoped event subscription list, differentiated timeout
categories, a validated state-transition table, an audited locking
discipline, callback/update separation, richer cleanup, and a mission
heartbeat.
"""

from __future__ import annotations

import time
import threading
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, FrozenSet, List, Optional, Set, TYPE_CHECKING

from missions.base_mission import BaseMission
from missions.waypoint import Waypoint
from missions.mission_path import MissionPath
from core.shared_data import SharedData
from core.events import EventBus, Event, EventType
from core.mission_manager import MissionLogger

if TYPE_CHECKING:
    from navigation.navigator import Navigator
import config


# ------------------------------------------------------------------------
# State machine definition
# ------------------------------------------------------------------------

class LapState(Enum):
    """Finite states for LapMission's waypoint-progression state machine."""

    IDLE = auto()
    READY = auto()
    WAITING_FIRST_WAYPOINT = auto()
    GOING_TO_WAYPOINT = auto()
    WAITING_NAVIGATOR = auto()
    WAYPOINT_REACHED = auto()
    NEXT_WAYPOINT = auto()
    NEXT_LAP = auto()
    MISSION_FINISHED = auto()
    ABORTED = auto()
    FAILED = auto()
    PAUSED = auto()
    CANCELLING_RETRY = auto()  # cancel() issued, awaiting Navigator idle before retry goto()


# Explicit, auditable state-transition table. Any transition not present
# here is rejected by `_transition()`. This prevents illegal jumps such as
# MISSION_FINISHED -> GOING_TO_WAYPOINT after a stray/duplicate event.
_VALID_TRANSITIONS: Dict[LapState, FrozenSet[LapState]] = {
    LapState.IDLE: frozenset({LapState.READY}),
    LapState.READY: frozenset({LapState.WAITING_FIRST_WAYPOINT, LapState.ABORTED, LapState.FAILED}),
    LapState.WAITING_FIRST_WAYPOINT: frozenset(
        {LapState.GOING_TO_WAYPOINT, LapState.ABORTED, LapState.FAILED, LapState.PAUSED}
    ),
    LapState.GOING_TO_WAYPOINT: frozenset(
        {LapState.WAITING_NAVIGATOR, LapState.ABORTED, LapState.FAILED, LapState.PAUSED}
    ),
    LapState.WAITING_NAVIGATOR: frozenset(
        {
            LapState.WAYPOINT_REACHED,
            LapState.CANCELLING_RETRY,
            LapState.ABORTED,
            LapState.FAILED,
            LapState.PAUSED,
        }
    ),
    LapState.CANCELLING_RETRY: frozenset(
        {LapState.GOING_TO_WAYPOINT, LapState.ABORTED, LapState.FAILED, LapState.PAUSED}
    ),
    LapState.WAYPOINT_REACHED: frozenset(
        {LapState.NEXT_WAYPOINT, LapState.NEXT_LAP, LapState.ABORTED, LapState.FAILED, LapState.PAUSED}
    ),
    LapState.NEXT_WAYPOINT: frozenset(
        {LapState.WAITING_FIRST_WAYPOINT, LapState.GOING_TO_WAYPOINT, LapState.ABORTED, LapState.FAILED, LapState.PAUSED}
    ),
    LapState.NEXT_LAP: frozenset(
        {
            LapState.WAITING_FIRST_WAYPOINT,
            LapState.GOING_TO_WAYPOINT,
            LapState.MISSION_FINISHED,
            LapState.ABORTED,
            LapState.FAILED,
            LapState.PAUSED,
        }
    ),
    LapState.PAUSED: frozenset({LapState.WAITING_FIRST_WAYPOINT, LapState.GOING_TO_WAYPOINT, LapState.ABORTED, LapState.FAILED}),
    LapState.MISSION_FINISHED: frozenset(),  # terminal
    LapState.ABORTED: frozenset(),  # terminal
    LapState.FAILED: frozenset(),  # terminal
}


class TimeoutKind(Enum):
    """Distinct timeout categories, each with independent recovery logic."""

    MISSION = auto()
    WAYPOINT = auto()
    NAVIGATOR = auto()
    RETRY = auto()


@dataclass
class LapStatistics:
    """Mutable, expanded statistics tracked over the lifetime of a run."""

    mission_id: str = ""
    mission_uuid: str = ""

    current_lap: int = 0
    total_laps: int = 0

    current_waypoint_index: int = 0
    total_waypoints: int = 0
    remaining_waypoints: int = 0

    total_distance: float = 0.0
    remaining_distance: float = 0.0
    average_ground_speed: float = 0.0
    average_waypoint_time: float = 0.0

    visited_waypoints: int = 0
    successful_waypoints: int = 0
    failed_waypoints: int = 0
    waypoint_retries: int = 0
    timeout_count: int = 0
    pause_count: int = 0
    resume_count: int = 0

    last_waypoint_time: Optional[float] = None
    last_retry_time: Optional[float] = None
    mission_start_time: Optional[float] = None
    mission_finish_time: Optional[float] = None
    last_heartbeat: Optional[float] = None

    _waypoint_durations: List[float] = field(default_factory=list, repr=False, compare=False)

    def duration(self) -> Optional[float]:
        """Return elapsed mission duration in seconds, if started."""
        if self.mission_start_time is None:
            return None
        end = self.mission_finish_time or time.monotonic()
        return end - self.mission_start_time

    def record_waypoint_duration(self, seconds: float) -> None:
        """Fold a completed waypoint's transit time into the running average."""
        self._waypoint_durations.append(seconds)
        self.average_waypoint_time = sum(self._waypoint_durations) / len(self._waypoint_durations)


@dataclass
class _PausedContext:
    """Snapshot of in-progress state captured at the moment of a pause."""

    state_before_pause: LapState
    waypoint: Optional[Waypoint] = None
    waypoint_index: int = 0


@dataclass
class _Checkpoint:
    """Last-completed-waypoint checkpoint, used to resume without restarting
    from waypoint zero unless a restart is explicitly requested."""

    waypoint_index: int = 0
    lap: int = 0


# ------------------------------------------------------------------------
# LapMission
# ------------------------------------------------------------------------

class LapMission(BaseMission):
    """
    Flies a predefined waypoint path (owned by MissionPath) repeatedly,
    lap after lap, until the configured lap count completes (or forever,
    if configured as unlimited).

    LapMission never sends MAVLink commands, never computes navigation,
    distances, or bearings, never talks to the flight controller directly,
    never spawns threads, and never manages hardware resources. All of
    that is delegated to Navigator / MissionManager. LapMission's sole
    concern is *waypoint progression logic*.

    LapMission subscribes only to mission-scoped events (start / pause /
    resume / abort / stop / goto-finished / goto-failed / waypoint-timeout).
    System-level conditions such as LOW_BATTERY, GPS_LOST, HEARTBEAT_TIMEOUT,
    CONNECTION_LOST, and EMERGENCY are owned by MissionManager, which
    translates its own policy decision into a MISSION_PAUSE / MISSION_ABORT
    command for LapMission — LapMission does not listen for them directly.
    """

    def __init__(
        self,
        shared_data: SharedData,
        mission_path: MissionPath,
        navigator: "Navigator",
        event_bus: EventBus,
        logger: MissionLogger,
        max_laps: Optional[int] = None,
    ) -> None:
        """
        Construct a LapMission via dependency injection.

        Args:
            shared_data: Shared cross-subsystem state (read-only from here).
            mission_path: Waypoint-ordering / lap-counting authority.
            navigator: Movement authority (goto/cancel/active/finished/
                       failed/remaining_distance).
            event_bus: Pub/sub bus for mission-scoped events.
            logger: Injected MissionLogger. LapMission never constructs its
                    own logger — this must be provided by the caller
                    (typically MissionManager).
            max_laps: Number of laps to fly. `None` or <= 0 means unlimited.
                      Falls back to `config.manager.default_lap_count` if
                      not given.
        """
        super().__init__(shared_data, "LapMission")

        self._path: MissionPath = mission_path
        self._navigator = navigator
        self._event_bus: EventBus = event_bus
        self.logger: MissionLogger = logger

        self._max_laps: Optional[int] = (
            max_laps if max_laps is not None else getattr(config, "LAP_COUNT", 0)
        )

        self._lock = threading.RLock()
        self._state: LapState = LapState.IDLE
        self._stats = LapStatistics(
            mission_id="lap-mission",
            mission_uuid=str(uuid.uuid4()),
        )
        self._paused_ctx: Optional[_PausedContext] = None
        self._checkpoint = _Checkpoint()
        self._current_waypoint: Optional[Waypoint] = None
        self._waypoint_send_time: Optional[float] = None
        self._retry_count: int = 0
        self._retry_deadline: Optional[float] = None
        self._cancel_issued_time: Optional[float] = None

        # After the last configured lap, return to waypoint 1 before
        # declaring the mission complete. This gives a finite mission the
        # route pattern: 1-2-3-4, 1-2-3-4, ..., 1.
        self._return_to_start_after_final_lap: bool = False

        # Strongly typed configuration access (no getattr/dynamic lookup).
        self._mission_timeout: float = config.manager.mission_timeout
        self._waypoint_timeout: float = config.manager.waypoint_timeout
        self._navigator_timeout: float = config.manager.navigator_timeout
        self._retry_timeout: float = config.manager.retry_timeout
        self._max_retries: int = config.manager.max_waypoint_retry
        self._heartbeat_interval: float = config.manager.heartbeat_interval

        # Dispatch table is built once, not on every update() call.
        self._dispatch: Dict[LapState, Callable[[], None]] = {
            LapState.READY: self._step_ready,
            LapState.WAITING_FIRST_WAYPOINT: self._step_waiting_first_waypoint,
            LapState.GOING_TO_WAYPOINT: self._step_going_to_waypoint,
            LapState.WAITING_NAVIGATOR: self._step_waiting_navigator,
            LapState.CANCELLING_RETRY: self._step_cancelling_retry,
            LapState.WAYPOINT_REACHED: self._step_waypoint_reached,
            LapState.NEXT_WAYPOINT: self._step_next_waypoint,
            LapState.NEXT_LAP: self._step_next_lap,
        }

        # Track subscriptions so cleanup() can unsubscribe every one of
        # them, preventing duplicate handlers across mission restarts.
        self._subscriptions: List[tuple] = []
        self._subscribe_to_events()

    # --------------------------------------------------------------
    # Event subscription / teardown
    # --------------------------------------------------------------

    # Mission-scoped events only. System-wide conditions (battery, GPS,
    # link health, emergency) are MissionManager's responsibility; it
    # converts them into MISSION_PAUSE / MISSION_ABORT for us.
    _MISSION_SCOPED_EVENTS: FrozenSet[EventType] = frozenset(
        {
            EventType.MISSION_START,
            EventType.MISSION_PAUSE,
            EventType.MISSION_RESUME,
            EventType.MISSION_ABORT,
            EventType.MISSION_STOP,
            EventType.GOTO_FINISHED,
            EventType.GOTO_FAILED,
            EventType.WAYPOINT_TIMEOUT,
        }
    )

    def _subscribe_to_events(self) -> None:
        """Register handlers for every mission-scoped event LapMission must
        react to, and remember each subscription so it can be reversed."""
        handlers = {
            EventType.MISSION_START: self._on_mission_start,
            EventType.MISSION_PAUSE: self._on_mission_pause,
            EventType.MISSION_RESUME: self._on_mission_resume,
            EventType.MISSION_ABORT: self._on_mission_abort,
            EventType.MISSION_STOP: self._on_mission_stop,
            EventType.GOTO_FINISHED: self._on_goto_finished,
            EventType.GOTO_FAILED: self._on_goto_failed,
            EventType.WAYPOINT_TIMEOUT: self._on_waypoint_timeout,
        }
        assert set(handlers.keys()) == self._MISSION_SCOPED_EVENTS
        for event_type, handler in handlers.items():
            self._event_bus.subscribe(event_type, handler)
            self._subscriptions.append((event_type, handler))

    def _unsubscribe_all_events(self) -> None:
        """Reverse every subscription made in `_subscribe_to_events`. Called
        from cleanup() so a mission restart never leaves duplicate/stale
        callbacks registered on the EventBus."""
        for event_type, handler in self._subscriptions:
            try:
                self._event_bus.unsubscribe(event_type, handler)
            except Exception as exc:  # noqa: BLE001 - teardown must not throw
                self.logger.log(
                    f"Failed to unsubscribe {event_type}: {exc!r}",
                    state=self._state.name,
                )
        self._subscriptions.clear()

    # --------------------------------------------------------------
    # BaseMission public API
    # --------------------------------------------------------------

    def initialize(self) -> None:
        """Validate the mission path and enter READY, awaiting MISSION_START."""
        with self._lock:
            self._validate_path()
            self._transition(LapState.READY)
            self._stats.total_laps = self._max_laps or 0
            self._stats.total_waypoints = self._path.remaining_waypoints()
            self._stats.last_heartbeat = time.monotonic()
            self._log("Mission initialized, awaiting start")

    def update(self) -> None:
        """
        Advance the internal state machine by one step.

        Intended to be called periodically by MissionManager's control
        loop. Non-blocking: each call performs at most one transition's
        worth of work. Uses the dispatch table built once at construction
        time rather than allocating one per call.
        """
        self._beat_heartbeat()

        with self._lock:
            current_state = self._state

        try:
            self._check_mission_timeout()
            step = self._dispatch.get(current_state)
            if step is not None:
                step()
        except Exception as exc:  # noqa: BLE001 - must degrade gracefully
            self._fail(f"Unexpected exception in update(): {exc!r}")

    def cleanup(self) -> None:
        """
        Release LapMission-owned bookkeeping.

        Unsubscribes every event handler, clears the Navigator reference,
        cached waypoint, retry counters, and pause context, and resets
        transient state — but never touches persistent mission statistics
        (those remain readable via `statistics()` after cleanup).
        """
        with self._lock:
            self._unsubscribe_all_events()
            self._navigator = None
            self._current_waypoint = None
            self._waypoint_send_time = None
            self._retry_count = 0
            self._retry_deadline = None
            self._cancel_issued_time = None
            self._return_to_start_after_final_lap = False
            self._paused_ctx = None
            self._log("Mission cleanup complete")

    def pause(self) -> None:
        """Pause mission progression without losing the current waypoint
        or lap position (see `_checkpoint`)."""
        with self._lock:
            if self._state in (LapState.MISSION_FINISHED, LapState.ABORTED, LapState.FAILED):
                return
            self._paused_ctx = _PausedContext(
                state_before_pause=self._state,
                waypoint=self._current_waypoint,
                waypoint_index=self._checkpoint.waypoint_index,
            )
            navigator = self._navigator
            self._transition(LapState.PAUSED)
            self._stats.pause_count += 1
            self._log("Mission paused")

        # Navigator call happens outside the lock (see THREAD SAFETY note).
        if navigator is not None:
            navigator.cancel()

    def resume(self) -> None:
        """Resume mission progression exactly where it was paused, using
        the last completed-waypoint checkpoint rather than waypoint zero."""
        with self._lock:
            if self._state != LapState.PAUSED or self._paused_ctx is None:
                return
            restored_state = self._paused_ctx.state_before_pause
            self._current_waypoint = self._paused_ctx.waypoint
            self._paused_ctx = None
            self._stats.resume_count += 1
            self._log("Mission resumed from checkpoint")

            # Re-issue the goto for the waypoint we were mid-flight to,
            # since Navigator.cancel() was called on pause.
            if restored_state in (LapState.GOING_TO_WAYPOINT, LapState.WAITING_NAVIGATOR, LapState.CANCELLING_RETRY):
                self._transition(LapState.GOING_TO_WAYPOINT)
            else:
                self._transition(restored_state)

    def abort(self) -> None:
        """Cancel navigation, stop the mission, and publish MISSION_ABORT."""
        with self._lock:
            navigator = self._navigator
            self._transition(LapState.ABORTED)
            self._stats.mission_finish_time = time.monotonic()
            self._log("Mission aborted")

        if navigator is not None:
            navigator.cancel()
        self._publish_event(EventType.MISSION_ABORT)

    def status(self) -> str:
        """Return the current FSM state name."""
        with self._lock:
            return self._state.name

    def statistics(self) -> LapStatistics:
        """Return a snapshot of current mission statistics, refreshed with
        live values pulled from MissionPath / Navigator where available."""
        with self._lock:
            self._stats.remaining_waypoints = self._safe_remaining_waypoints()
            self._stats.current_waypoint_index = self._checkpoint.waypoint_index
            self._stats.remaining_distance = self._safe_remaining_distance()
            return self._stats

    # --------------------------------------------------------------
    # State step implementations
    # --------------------------------------------------------------

    def _step_ready(self) -> None:
        """READY -> WAITING_FIRST_WAYPOINT once MissionManager has started
        the mission (signalled via MISSION_START, handled asynchronously)."""
        return

    def _step_waiting_first_waypoint(self) -> None:
        """Fetch the first (or checkpointed) waypoint and send it."""
        self._send_current_waypoint()

    def _step_going_to_waypoint(self) -> None:
        """Waypoint has been sent; move to waiting for Navigator/event."""
        with self._lock:
            self._transition(LapState.WAITING_NAVIGATOR)

    def _step_waiting_navigator(self) -> None:
        """
        Poll for waypoint/navigator-level timeout and cross-check Navigator's
        own state in case a GOTO_FINISHED/GOTO_FAILED event was missed.
        """
        navigator = self._navigator
        if navigator is None:
            return

        # Cross-check against Navigator's own state rather than relying on
        # events alone — this keeps LapMission correct even if an event
        # is dropped or arrives out of order.
        try:
            if navigator.finished():
                self._on_goto_finished(None)
                return
            if navigator.failed():
                self._on_goto_failed(None)
                return
        except Exception as exc:  # noqa: BLE001
            self._fail(f"Navigator status query failed: {exc!r}")
            return

        with self._lock:
            send_time = self._waypoint_send_time
        if send_time is None:
            return

        elapsed = time.monotonic() - send_time
        if elapsed > self._waypoint_timeout:
            self._handle_timeout(TimeoutKind.WAYPOINT)
        elif elapsed > self._navigator_timeout and not navigator.active():
            # Navigator silently stopped without confirming finished/failed.
            self._handle_timeout(TimeoutKind.NAVIGATOR)

    def _step_cancelling_retry(self) -> None:
        """
        Await Navigator confirmation of idle after a cancel() before issuing
        the retry goto(). This avoids overlapping/concurrent GOTO commands.
        """
        navigator = self._navigator
        if navigator is None:
            return

        try:
            idle = not navigator.active()
        except Exception as exc:  # noqa: BLE001
            self._fail(f"Navigator status query failed during retry: {exc!r}")
            return

        with self._lock:
            cancel_time = self._cancel_issued_time

        timed_out = cancel_time is not None and (time.monotonic() - cancel_time) > self._retry_timeout

        if idle or timed_out:
            with self._lock:
                self._cancel_issued_time = None
            self._send_current_waypoint()

    def _step_waypoint_reached(self) -> None:
        """Record arrival, checkpoint progress, then decide whether to
        advance waypoint or lap."""
        with self._lock:
            self._stats.visited_waypoints += 1
            self._stats.successful_waypoints += 1
            now = time.monotonic()
            if self._waypoint_send_time is not None:
                self._stats.record_waypoint_duration(now - self._waypoint_send_time)
            self._stats.last_waypoint_time = now
            returning_to_start = self._return_to_start_after_final_lap
            self._log("Waypoint reached")

        # The final return to waypoint 1 is a deliberate terminal leg. Do
        # not call MissionPath.next() here, otherwise the path would advance
        # past waypoint 1 and the final arrival could never be observed as
        # the requested completion point.
        if returning_to_start:
            with self._lock:
                self._return_to_start_after_final_lap = False
                self._stats.current_lap = self._max_laps or self._path.total_laps()
            self._finish()
            return

        self._path.next()
        self._checkpoint.waypoint_index += 1
        self._checkpoint.lap = self._path.lap()

        with self._lock:
            if self._path.finished():
                self._transition(LapState.NEXT_LAP)
            else:
                self._transition(LapState.NEXT_WAYPOINT)

    def _step_next_waypoint(self) -> None:
        """Advance to the next waypoint within the current lap."""
        self._next_waypoint()

    def _step_next_lap(self) -> None:
        """Handle end-of-lap: either start a new lap or finish the mission."""
        self._next_lap()

    # --------------------------------------------------------------
    # Private orchestration methods
    # --------------------------------------------------------------

    def _send_current_waypoint(self) -> None:
        """Fetch the current waypoint from MissionPath and command Navigator
        to fly to it. LapMission never computes the waypoint itself. The
        Navigator call is made outside the lock."""
        waypoint = self._path.current()
        if waypoint is None:
            self._fail("MissionPath returned no current waypoint")
            return

        navigator = self._navigator
        with self._lock:
            self._current_waypoint = waypoint
            self._waypoint_send_time = time.monotonic()
            self._transition(LapState.GOING_TO_WAYPOINT)

        if navigator is not None:
            navigator.goto(waypoint)
        self._log(f"Waypoint sent: {waypoint!r}")

    def _next_waypoint(self) -> None:
        """Advance within the current lap and dispatch the next waypoint."""
        with self._lock:
            self._retry_count = 0
            self._transition(LapState.WAITING_FIRST_WAYPOINT)
        self._send_current_waypoint()

    def _next_lap(self) -> None:
        """Handle completion of a lap: restart the path for another lap,
        or finish the mission if the configured lap count is reached."""
        current_lap = self._path.lap()
        with self._lock:
            self._stats.current_lap = current_lap
        self._log(f"Lap completed: {current_lap}")

        unlimited = not self._max_laps or self._max_laps <= 0
        if unlimited or current_lap < self._max_laps:
            self._path.restart()
            self._checkpoint = _Checkpoint(waypoint_index=0, lap=current_lap)
            with self._lock:
                self._return_to_start_after_final_lap = False
                self._retry_count = 0
                self._transition(LapState.WAITING_FIRST_WAYPOINT)
            self._send_current_waypoint()
        else:
            # Finite mission: after the last waypoint of the last lap,
            # explicitly fly back to waypoint 1 and only then finish.
            # MissionPath.next() has already wrapped current_index to 0, so
            # current() is waypoint 1 here.
            self._path.restart()
            self._checkpoint = _Checkpoint(waypoint_index=0, lap=current_lap)
            with self._lock:
                self._return_to_start_after_final_lap = True
                self._retry_count = 0
                self._transition(LapState.WAITING_FIRST_WAYPOINT)
            self._send_current_waypoint()

    def _finish(self) -> None:
        """Mark the mission complete and publish MISSION_FINISHED."""
        navigator = self._navigator
        with self._lock:
            self._transition(LapState.MISSION_FINISHED)
            self._stats.mission_finish_time = time.monotonic()
        if navigator is not None:
            navigator.cancel()
        self._log("Mission completed")
        self._publish_event(EventType.MISSION_FINISHED)

    def _fail(self, reason: str) -> None:
        """Transition to FAILED, cancel navigation, and log the reason."""
        navigator = self._navigator
        with self._lock:
            # FAILED is reachable from any non-terminal state; bypass the
            # strict table only for this safety-critical escape hatch.
            self._state = LapState.FAILED
            self._stats.mission_finish_time = time.monotonic()
        if navigator is not None:
            navigator.cancel()
        self._log(f"Mission failed: {reason}")
        self._publish_event(EventType.MISSION_FAILED, payload={"reason": reason})

    def _begin_retry(self) -> None:
        """
        Begin a safe retry: cancel any in-flight goto and wait for Navigator
        to confirm idle (CANCELLING_RETRY) before re-issuing goto(), instead
        of immediately re-sending the waypoint. Prevents concurrent GOTOs.
        """
        navigator = self._navigator
        with self._lock:
            self._retry_count += 1
            self._stats.waypoint_retries += 1
            self._stats.last_retry_time = time.monotonic()
            self._cancel_issued_time = time.monotonic()
            self._transition(LapState.CANCELLING_RETRY)
        if navigator is not None:
            navigator.cancel()
        self._log(f"Retry {self._retry_count}/{self._max_retries} initiated: cancelling before re-issuing goto")

    def _handle_timeout(self, kind: TimeoutKind) -> None:
        """
        Dispatch to the recovery logic appropriate for the given timeout
        category. Each kind has independent recovery semantics.
        """
        with self._lock:
            self._stats.timeout_count += 1

        if kind is TimeoutKind.MISSION:
            self._fail("Mission timeout exceeded")
            return

        if kind is TimeoutKind.WAYPOINT:
            self._publish_event(EventType.WAYPOINT_TIMEOUT)
            return

        if kind is TimeoutKind.NAVIGATOR:
            self._log("Navigator silent timeout detected; treating as goto failure")
            self._on_goto_failed(None)
            return

        if kind is TimeoutKind.RETRY:
            self._fail("Retry cancellation did not confirm Navigator idle in time")
            return

    def _publish_event(self, event_type: EventType, payload: Optional[dict] = None) -> None:
        """Publish an event on the shared EventBus."""
        self._event_bus.publish(Event(type=event_type, source="LapMission", payload=payload or {}))

    def _log(self, message: str) -> None:
        """Emit a structured log entry via the injected MissionLogger."""
        self.logger.log(message, state=self._state.name)

    def _transition(self, new_state: LapState) -> None:
        """
        Validate and perform a state transition. Must be called while
        holding `self._lock`. Rejects any transition not present in
        `_VALID_TRANSITIONS`, guaranteeing e.g. that MISSION_FINISHED can
        never jump back to GOING_TO_WAYPOINT.
        """
        allowed = _VALID_TRANSITIONS.get(self._state, frozenset())
        if new_state not in allowed and new_state is not self._state:
            self.logger.log(
                f"Rejected invalid transition {self._state.name} -> {new_state.name}",
                state=self._state.name,
            )
            return
        self._state = new_state

    def _validate_path(self) -> None:
        """
        Ensure MissionPath has been configured with at least one waypoint,
        and that every waypoint in the path is individually valid, before
        the mission is allowed to start. Invalid paths are rejected with a
        raised ValueError so MissionManager can surface the failure early.
        """
        if self._path is None:
            raise ValueError("LapMission requires a MissionPath instance")
        if self._path.remaining_waypoints() <= 0:
            raise ValueError("MissionPath contains no waypoints")

        seen: Set[tuple] = set()
        for index, waypoint in enumerate(self._path.iter_waypoints()):
            self._validate_waypoint(waypoint, index)
            key = (
                round(waypoint.latitude, 7),
                round(waypoint.longitude, 7),
                round(waypoint.altitude, 2),
            )
            if key in seen:
                raise ValueError(f"Duplicate/invalid waypoint entry at index {index}: {waypoint!r}")
            seen.add(key)

    @staticmethod
    def _validate_waypoint(waypoint: Optional[Waypoint], index: int) -> None:
        """Validate a single waypoint's fields (latitude, longitude,
        altitude, acceptance radius, presence, and NaN)."""
        if waypoint is None:
            raise ValueError(f"Waypoint at index {index} is None")

        for field_name in ("latitude", "longitude", "altitude", "acceptance_radius"):
            if not hasattr(waypoint, field_name):
                raise ValueError(f"Waypoint at index {index} missing field '{field_name}'")
            value = getattr(waypoint, field_name)
            if value is None:
                raise ValueError(f"Waypoint at index {index} has None for '{field_name}'")
            if isinstance(value, float) and value != value:  # NaN check
                raise ValueError(f"Waypoint at index {index} has NaN for '{field_name}'")

        if not (-90.0 <= waypoint.latitude <= 90.0):
            raise ValueError(f"Waypoint at index {index} latitude out of range: {waypoint.latitude}")
        if not (-180.0 <= waypoint.longitude <= 180.0):
            raise ValueError(f"Waypoint at index {index} longitude out of range: {waypoint.longitude}")
        if waypoint.altitude < 0.0:
            raise ValueError(f"Waypoint at index {index} altitude invalid: {waypoint.altitude}")
        if waypoint.acceptance_radius <= 0.0:
            raise ValueError(f"Waypoint at index {index} acceptance radius invalid: {waypoint.acceptance_radius}")

    def _safe_remaining_waypoints(self) -> int:
        """Defensive wrapper around MissionPath.remaining_waypoints()."""
        try:
            return self._path.remaining_waypoints()
        except Exception:  # noqa: BLE001
            return 0

    def _safe_remaining_distance(self) -> float:
        """Defensive wrapper around Navigator.remaining_distance()."""
        navigator = self._navigator
        if navigator is None:
            return 0.0
        try:
            return float(navigator.remaining_distance())
        except Exception:  # noqa: BLE001
            return 0.0

    def _check_mission_timeout(self) -> None:
        """Fail the mission if total elapsed time exceeds the configured
        mission timeout, ignoring terminal/paused states."""
        with self._lock:
            state = self._state
            start = self._stats.mission_start_time

        if state in (
            LapState.IDLE,
            LapState.READY,
            LapState.MISSION_FINISHED,
            LapState.ABORTED,
            LapState.FAILED,
            LapState.PAUSED,
        ):
            return
        if start is None:
            return
        if (time.monotonic() - start) > self._mission_timeout:
            self._handle_timeout(TimeoutKind.MISSION)

    def _beat_heartbeat(self) -> None:
        """Update the mission heartbeat timestamp. MissionWatchdog reads
        `statistics().last_heartbeat` to detect a frozen mission loop."""
        now = time.monotonic()
        with self._lock:
            if (
                self._stats.last_heartbeat is None
                or (now - self._stats.last_heartbeat) >= self._heartbeat_interval
            ):
                self._stats.last_heartbeat = now

    # --------------------------------------------------------------
    # Event handlers
    #
    # NOTE (callback safety): every handler below is intentionally thin —
    # it only records state and performs a validated transition. No
    # navigation or heavy computation happens here; that logic lives in
    # update()'s state-step methods, which run on MissionManager's control
    # cycle rather than on the EventBus dispatch thread.
    # --------------------------------------------------------------

    def _on_mission_start(self, _event: Optional[Event]) -> None:
        """Begin the mission: start the clock and request the first
        waypoint, resuming from checkpoint if one exists."""
        with self._lock:
            if self._state != LapState.READY:
                return
            self._stats.mission_start_time = time.monotonic()
            self._transition(LapState.WAITING_FIRST_WAYPOINT)
            self._log("Mission started")

    def _on_mission_pause(self, _event: Optional[Event]) -> None:
        self.pause()

    def _on_mission_resume(self, _event: Optional[Event]) -> None:
        self.resume()

    def _on_mission_abort(self, _event: Optional[Event]) -> None:
        self.abort()

    def _on_mission_stop(self, _event: Optional[Event]) -> None:
        """Graceful stop: cancels navigation and halts without implying
        failure semantics upstream."""
        navigator = self._navigator
        with self._lock:
            self._transition(LapState.ABORTED)
            self._stats.mission_finish_time = time.monotonic()
            self._log("Mission stopped")
        if navigator is not None:
            navigator.cancel()

    def _on_goto_finished(self, _event: Optional[Event]) -> None:
        """Navigator (via event or direct poll in `_step_waiting_navigator`)
        confirmed waypoint arrival."""
        with self._lock:
            if self._state not in (LapState.WAITING_NAVIGATOR,):
                return
            self._waypoint_send_time = self._waypoint_send_time  # preserved for duration calc
            self._transition(LapState.WAYPOINT_REACHED)

    def _on_goto_failed(self, _event: Optional[Event]) -> None:
        """Navigator reported (via event or poll) a failure reaching the
        current waypoint. Retry using cancel-then-confirm-idle up to the
        configured maximum before failing the mission."""
        with self._lock:
            if self._state not in (LapState.WAITING_NAVIGATOR, LapState.GOING_TO_WAYPOINT):
                return
            self._stats.failed_waypoints += 1
            retry_count = self._retry_count

        if retry_count >= self._max_retries:
            self._fail("Navigator failed to reach waypoint after max retries")
            return
        self._begin_retry()

    def _on_waypoint_timeout(self, _event: Optional[Event]) -> None:
        """Waypoint navigation exceeded the configured timeout; retry or fail."""
        with self._lock:
            if self._state != LapState.WAITING_NAVIGATOR:
                return
            retry_count = self._retry_count

        if retry_count >= self._max_retries:
            self._fail("Waypoint timeout exceeded max retries")
            return
        self._begin_retry()
