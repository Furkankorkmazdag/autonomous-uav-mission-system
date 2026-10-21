"""
missions/mission_manager.py
============================
MissionManager — Production flight-computer mission orchestrator for SUAS.

Architecture (Clean / SOLID):
──────────────────────────────────────────────────────────────────────────────
MissionManager is a *thin orchestrator* (~250 lines of business logic).
All domain logic lives in dedicated, independently-testable subsystems:

  ┌─────────────────────────────────────────────────────────────────────┐
  │                        MissionManager                               │
  │  (orchestrates — contains almost no business logic of its own)      │
  │                                                                     │
  │  MissionRegistry   MissionScheduler   MissionStateMachine           │
  │  ResourceManager   SafetyMonitor      MissionWatchdog               │
  │  EventDispatcher   MissionLogger                                    │
  └─────────────────────────────────────────────────────────────────────┘

Design decisions:
  • MissionManager NEVER imports concrete mission classes.
  • MissionManager NEVER instantiates missions (main.py does that).
  • SafetyMonitor NEVER executes actions — it only publishes EventBus events.
  • EventDispatcher uses a decorator pattern: @event(EventType.X).
  • ResourceManager supports dynamic resource registration.
  • MissionWatchdog performs safe abort → release → wait → restart recovery.
  • All magic numbers are sourced from config.py via ManagerConfig.
  • Every shared mutable object is guarded by threading.RLock.

Adding a new mission requires ONLY:
  1.  class NewMission(BaseMission): ...   (in its own file)
  2.  manager.register(MissionRecord(mission=NewMission(), ...))
  MissionManager itself never changes.

Compatible with: Python 3.9+, pymavlink (no DroneKit, no ROS).
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto, unique
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type
from config import ManagerConfig

# ---------------------------------------------------------------------------
# Internal project imports
# ---------------------------------------------------------------------------
from core.events import EventBus, EventType      # noqa: E402
from core.shared_data import SharedData          # noqa: E402
from missions.base_mission import BaseMission    # noqa: E402

# config.py must expose a ManagerConfig dataclass — see companion config.py
from config import ManagerConfig                 # noqa: E402

_log = logging.getLogger(__name__)


# ===========================================================================
# § 1  ENUMERATIONS  — shared vocabulary for the entire subsystem
# ===========================================================================

@unique
class SystemState(Enum):
    """Overall UAV system finite-state-machine states."""
    BOOT              = auto()
    INITIALIZING      = auto()
    READY             = auto()
    MISSION_RUNNING   = auto()
    MISSION_PAUSED    = auto()
    MISSION_COMPLETED = auto()
    RETURN_HOME       = auto()
    LANDING           = auto()
    FINISHED          = auto()
    ABORTED           = auto()
    EMERGENCY         = auto()


@unique
class MissionStatus(Enum):
    """Per-mission lifecycle status."""
    IDLE      = auto()
    READY     = auto()
    RUNNING   = auto()
    PAUSED    = auto()
    FAILED    = auto()
    ABORTED   = auto()
    COMPLETED = auto()


@unique
class Resource(Enum):
    """
    Lockable hardware / software resources.

    Additional resources can be registered at runtime via
    ResourceManager.register_resource().
    """
    # Hardware actuators / sensors
    NAVIGATOR      = "navigator"
    CAMERA         = "camera"
    PAYLOAD_SERVO  = "payload_servo"
    AI_DETECTOR    = "ai_detector"
    # Communication / data pipeline
    MAVLINK        = "mavlink"
    LOGGER         = "logger"
    IMAGE_BUFFER   = "image_buffer"
    MISSION_PATH   = "mission_path"
    PARAMETERS     = "parameters"
    FILESYSTEM     = "filesystem"


# ===========================================================================
# § 2  DATA CONTAINERS
# ===========================================================================

@dataclass
class MissionRecord:
    """
    Wraps a BaseMission with full runtime bookkeeping.

    Fields are intentionally grouped:
      - Identity        : mission, priority, exclusive, concurrent
      - Status          : status, thread, owner
      - Timing          : last_start_time, last_stop_time, last_pause_time
      - Fault tolerance : restart_counter, maximum_restarts, auto_restart
      - Timeouts        : heartbeat_timeout, mission_timeout
      - Resources       : required_resources
      - Statistics      : cpu_time, memory_usage  (updated externally)
      - Watchdog        : last_heartbeat
    """

    # --- Identity ----------------------------------------------------------
    mission:            BaseMission
    priority:           int
    exclusive:          bool = False   # True → preempts all lower-priority missions
    concurrent:         bool = True    # False → cannot share the run-slot

    # --- Status ------------------------------------------------------------
    status:             MissionStatus = MissionStatus.IDLE
    thread:             Optional[threading.Thread] = None
    owner:              str = ""       # logical subsystem that requested this mission

    # --- Timing ------------------------------------------------------------
    last_start_time:    float = 0.0
    last_stop_time:     float = 0.0
    last_pause_time:    float = 0.0

    # --- Fault tolerance ---------------------------------------------------
    restart_counter:    int   = 0
    maximum_restarts:   int   = 3
    auto_restart:       bool  = False  # True → watchdog restarts automatically

    # --- Timeouts ----------------------------------------------------------
    heartbeat_timeout:  float = 10.0  # seconds; overrides watchdog default if > 0
    mission_timeout:    float = 0.0   # seconds; 0 = unlimited

    # --- Resources ---------------------------------------------------------
    required_resources: Set[Resource] = field(default_factory=set)

    # --- Statistics  (updated by thread wrapper / external monitor) --------
    cpu_time:           float = 0.0   # accumulated CPU seconds
    memory_usage:       int   = 0     # peak resident bytes

    # --- Watchdog ----------------------------------------------------------
    last_heartbeat:     float = field(default_factory=time.monotonic)

    # --- Derived properties ------------------------------------------------
    @property
    def name(self) -> str:
        return self.mission.name

    @property
    def is_alive(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    @property
    def elapsed_since_start(self) -> float:
        """Seconds since last start (0 if not started)."""
        return (time.monotonic() - self.last_start_time) if self.last_start_time else 0.0

    def timed_out(self) -> bool:
        """True if mission_timeout is set and the mission has exceeded it."""
        return (
            self.mission_timeout > 0
            and self.status == MissionStatus.RUNNING
            and self.elapsed_since_start > self.mission_timeout
        )


@dataclass
class HealthSnapshot:
    """Aggregated vehicle-health data consumed by SafetyMonitor."""
    gps_fix:         bool  = False
    ekf_ok:          bool  = False
    heartbeat_age_s: float = 9999.0
    battery_pct:     float = 100.0
    connection_ok:   bool  = False
    rc16_pwm:        int   = 0       # RC channel 16 PWM value used by RC16_SAFETY_TRIGGERED.


# ===========================================================================
# § 3  MISSION LOGGER
# ===========================================================================

class MissionLogger:
    """
    Centralised structured logger for mission lifecycle events.

    MissionManager and subsystems call MissionLogger instead of
    writing ad-hoc log statements.  Every entry is timestamped and
    categorised so post-flight analysis tools can parse the output.
    """

    # Category tags used as prefixes — kept consistent for log parsing
    TAG_START      = "START"
    TAG_STOP       = "STOP"
    TAG_PAUSE      = "PAUSE"
    TAG_RESUME     = "RESUME"
    TAG_ABORT      = "ABORT"
    TAG_FAIL       = "FAIL"
    TAG_COMPLETE   = "COMPLETE"
    TAG_TRANSITION = "FSM"
    TAG_EVENT      = "EVENT"
    TAG_RESOURCE   = "RESOURCE"
    TAG_WATCHDOG   = "WATCHDOG"

    def __init__(self, base_logger: logging.Logger = _log) -> None:
        self._log = base_logger

    # ------------------------------------------------------------------
    # Mission lifecycle entries
    # ------------------------------------------------------------------

    def mission_start(self, name: str, priority: int) -> None:
        self._emit(self.TAG_START, f"'{name}' priority={priority}")

    def mission_stop(self, name: str) -> None:
        self._emit(self.TAG_STOP, f"'{name}'")

    def mission_pause(self, name: str, reason: str = "") -> None:
        detail = f" reason={reason}" if reason else ""
        self._emit(self.TAG_PAUSE, f"'{name}'{detail}")

    def mission_resume(self, name: str) -> None:
        self._emit(self.TAG_RESUME, f"'{name}'")

    def mission_abort(self, name: str, reason: str = "") -> None:
        detail = f" reason={reason}" if reason else ""
        self._emit(self.TAG_ABORT, f"'{name}'{detail}")

    def mission_failed(self, name: str, exc: Optional[Exception] = None) -> None:
        detail = f" exc={exc}" if exc else ""
        self._emit(self.TAG_FAIL, f"'{name}'{detail}", level=logging.ERROR)

    def mission_completed(self, name: str, elapsed_s: float) -> None:
        self._emit(self.TAG_COMPLETE, f"'{name}' elapsed={elapsed_s:.1f}s")

    # ------------------------------------------------------------------
    # System / FSM entries
    # ------------------------------------------------------------------

    def state_transition(self, old: SystemState, new: SystemState) -> None:
        self._emit(self.TAG_TRANSITION, f"{old.name} → {new.name}")

    def status_transition(self, name: str, old: MissionStatus, new: MissionStatus) -> None:
        self._emit(self.TAG_TRANSITION, f"'{name}' {old.name} → {new.name}")

    def event_received(self, event_type: str, detail: str = "") -> None:
        body = f"{event_type}" + (f" {detail}" if detail else "")
        self._emit(self.TAG_EVENT, body)

    def resource_acquired(self, name: str, resources: Set[Resource]) -> None:
        labels = {r.value for r in resources}
        self._emit(self.TAG_RESOURCE, f"'{name}' acquired {labels}")

    def resource_released(self, name: str, resources: Set[Resource]) -> None:
        labels = {r.value for r in resources}
        self._emit(self.TAG_RESOURCE, f"'{name}' released {labels}")

    def watchdog_restart(self, name: str, attempt: int, maximum: int, reason: str) -> None:
        self._emit(
            self.TAG_WATCHDOG,
            f"Restarting '{name}' [{attempt}/{maximum}] reason={reason}",
            level=logging.WARNING,
        )

    def watchdog_give_up(self, name: str) -> None:
        self._emit(
            self.TAG_WATCHDOG,
            f"'{name}' exceeded max restarts — marking FAILED",
            level=logging.ERROR,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _emit(self, tag: str, body: str, level: int = logging.INFO) -> None:
        ts = time.strftime("%H:%M:%S")
        self._log.log(level, "[%s][%s] %s", ts, tag, body)


# ===========================================================================
# § 4  RESOURCE MANAGER
# ===========================================================================

class ResourceManager:
    """
    Thread-safe resource ownership tracker.

    Supports the built-in Resource enum values and additional resources
    registered at runtime via register_resource().

    Acquisition is *atomic*: either all requested resources are granted or
    none are, eliminating partial-lock race conditions.
    """

    def __init__(self, mission_logger: MissionLogger) -> None:
        self._lock:   threading.RLock            = threading.RLock()
        # resource key → mission name currently holding it (or None)
        self._owners: Dict[str, Optional[str]]   = {r.value: None for r in Resource}
        self._mlog:   MissionLogger              = mission_logger

    # ------------------------------------------------------------------
    # Dynamic resource registration
    # ------------------------------------------------------------------

    def register_resource(self, key: str) -> None:
        """
        Register an additional resource identified by *key*.

        Idempotent — silently ignored if key is already present.
        """
        with self._lock:
            if key not in self._owners:
                self._owners[key] = None
                _log.debug("ResourceManager: registered dynamic resource '%s'.", key)

    # ------------------------------------------------------------------
    # Acquisition / release
    # ------------------------------------------------------------------

    def try_acquire(self, mission_name: str, resources: Set[Resource]) -> bool:
        """
        Atomically attempt to acquire *all* requested resources.

        Returns True on success; False without any partial acquisition if
        any resource is currently owned by a different mission.
        """
        with self._lock:
            for res in resources:
                owner = self._owners.get(res.value)
                if owner is not None and owner != mission_name:
                    _log.debug(
                        "Resource '%s' owned by '%s'; '%s' must wait.",
                        res.value, owner, mission_name,
                    )
                    return False
            for res in resources:
                self._owners[res.value] = mission_name
            self._mlog.resource_acquired(mission_name, resources)
            return True

    def release(self, mission_name: str) -> None:
        """Release every resource currently owned by *mission_name*."""
        with self._lock:
            released: Set[Resource] = set()
            for key in list(self._owners):
                if self._owners[key] == mission_name:
                    self._owners[key] = None
                    # Only enum members go into the typed set for the logger
                    try:
                        released.add(Resource(key))
                    except ValueError:
                        pass  # dynamic (non-enum) resource — log separately
                        _log.debug("Released dynamic resource '%s' from '%s'.", key, mission_name)
            if released:
                self._mlog.resource_released(mission_name, released)

    def release_specific(self, mission_name: str, resources: Set[Resource]) -> None:
        """Release a specific subset of resources owned by *mission_name*."""
        with self._lock:
            released: Set[Resource] = set()
            for res in resources:
                if self._owners.get(res.value) == mission_name:
                    self._owners[res.value] = None
                    released.add(res)
            if released:
                self._mlog.resource_released(mission_name, released)

    def owner_of(self, resource: Resource) -> Optional[str]:
        """Return the name of the mission currently owning *resource*, or None."""
        with self._lock:
            return self._owners.get(resource.value)

    def snapshot(self) -> Dict[str, Optional[str]]:
        """Return a copy of current ownership for diagnostics / telemetry."""
        with self._lock:
            return dict(self._owners)


# ===========================================================================
# § 5  MISSION REGISTRY
# ===========================================================================

class MissionRegistry:
    """
    Thread-safe store of MissionRecord objects.

    Public interface intentionally mirrors a typed dictionary with
    filtering and sorting capabilities.
    """

    def __init__(self, mission_logger: MissionLogger) -> None:
        self._lock:    threading.RLock              = threading.RLock()
        self._records: Dict[str, MissionRecord]     = {}
        self._mlog:    MissionLogger                = mission_logger

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def register(self, record: MissionRecord) -> None:
        """
        Add *record* to the registry.

        Raises ValueError if a mission with the same name is already registered.
        """
        with self._lock:
            if record.name in self._records:
                raise ValueError(
                    f"Mission '{record.name}' is already registered. "
                    "Call unregister() first."
                )
            self._records[record.name] = record
            _log.info(
                "Mission registered: '%s' (priority=%d, exclusive=%s).",
                record.name, record.priority, record.exclusive,
            )

    def unregister(self, name: str) -> None:
        """
        Remove a mission from the registry.

        Raises KeyError   if the mission is unknown.
        Raises RuntimeError if the mission is RUNNING or PAUSED.
        """
        with self._lock:
            if name not in self._records:
                raise KeyError(f"Mission '{name}' is not registered.")
            record = self._records[name]
            if record.status in (MissionStatus.RUNNING, MissionStatus.PAUSED):
                raise RuntimeError(
                    f"Cannot unregister '{name}' while {record.status.name}. Stop first."
                )
            del self._records[name]
            _log.info("Mission unregistered: '%s'.", name)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def exists(self, name: str) -> bool:
        with self._lock:
            return name in self._records

    def get(self, name: str) -> MissionRecord:
        with self._lock:
            if name not in self._records:
                raise KeyError(f"Mission '{name}' is not registered.")
            return self._records[name]

    def list(self) -> List[MissionRecord]:
        """Snapshot list of all records (unsorted)."""
        with self._lock:
            return list(self._records.values())

    def find(
        self,
        *,
        status:       Optional[MissionStatus] = None,
        min_priority: Optional[int]           = None,
        max_priority: Optional[int]           = None,
        exclusive:    Optional[bool]          = None,
        auto_restart: Optional[bool]          = None,
    ) -> List[MissionRecord]:
        """
        Filtered query — all filters AND-combined.
        Results returned sorted by descending priority.
        """
        with self._lock:
            results = list(self._records.values())

        if status       is not None: results = [r for r in results if r.status       == status]
        if min_priority is not None: results = [r for r in results if r.priority     >= min_priority]
        if max_priority is not None: results = [r for r in results if r.priority     <= max_priority]
        if exclusive    is not None: results = [r for r in results if r.exclusive    == exclusive]
        if auto_restart is not None: results = [r for r in results if r.auto_restart == auto_restart]

        results.sort(key=lambda r: r.priority, reverse=True)
        return results


# ===========================================================================
# § 6  MISSION STATE MACHINE
# ===========================================================================

class MissionStateMachine:
    """
    Owns the SystemState FSM.

    Encapsulates all valid transition logic and the set of terminal states
    that must not be overwritten by normal scheduler updates.

    Exposes:
      state          — current state (read-only property)
      transition()   — guarded state change
      is_terminal()  — True when no further scheduling should occur
    """

    # States that the scheduler must never override
    _TERMINAL_STATES: frozenset = frozenset({
        SystemState.EMERGENCY,
        SystemState.RETURN_HOME,
        SystemState.LANDING,
        SystemState.ABORTED,
        SystemState.FINISHED,
    })

    def __init__(self, mission_logger: MissionLogger) -> None:
        self._lock:  threading.RLock = threading.RLock()
        self._state: SystemState     = SystemState.BOOT
        self._mlog:  MissionLogger   = mission_logger

    @property
    def state(self) -> SystemState:
        with self._lock:
            return self._state

    def transition(self, new_state: SystemState) -> bool:
        """
        Transition to *new_state*.

        Returns True on success; False if the current state is terminal
        and the requested state is not an allowed escalation.
        No-op (returns True) if already in *new_state*.
        """
        with self._lock:
            if self._state == new_state:
                return True
            old = self._state
            self._state = new_state
            self._mlog.state_transition(old, new_state)
            return True

    def is_terminal(self) -> bool:
        """True when the FSM is in a state that blocks normal scheduling."""
        with self._lock:
            return self._state in self._TERMINAL_STATES

    def derive_from_missions(
        self,
        registry: MissionRegistry,
    ) -> None:
        """
        Infer the appropriate SystemState from the current mission set.

        Called every scheduler tick.  Skips update when in a terminal state
        so safety-driven transitions are never silently overwritten.
        """
        if self.is_terminal():
            return

        running   = registry.find(status=MissionStatus.RUNNING)
        paused    = registry.find(status=MissionStatus.PAUSED)
        all_recs  = registry.list()
        completed = registry.find(status=MissionStatus.COMPLETED)

        if running:
            self.transition(SystemState.MISSION_RUNNING)
        elif paused:
            self.transition(SystemState.MISSION_PAUSED)
        elif all_recs and len(completed) == len(all_recs):
            self.transition(SystemState.MISSION_COMPLETED)
        else:
            self.transition(SystemState.READY)


# ===========================================================================
# § 7  MISSION SCHEDULER
# ===========================================================================

class MissionScheduler:
    """
    Implements mission start / stop / pause / resume / abort / restart.

    All public methods are thread-safe and acquire the registry lock only
    for the minimum duration needed.  Heavy operations (thread join) are
    performed outside the lock.

    Priority preemption and concurrency enforcement happen in update(),
    which is called on every main-loop tick.
    """

    def __init__(
        self,
        registry:   MissionRegistry,
        resources:  ResourceManager,
        fsm:        MissionStateMachine,
        event_bus:  EventBus,
        mlog:       MissionLogger,
        cfg:        ManagerConfig,
    ) -> None:
        self._registry  = registry
        self._resources = resources
        self._fsm       = fsm
        self._bus       = event_bus
        self._mlog      = mlog
        self._cfg       = cfg
        self._lock      = threading.RLock()

    # ------------------------------------------------------------------
    # Public scheduling API
    # ------------------------------------------------------------------

    def start(self, name: str) -> bool:
        """Attempt to start the named mission. Returns True on success."""
        with self._lock:
            return self._do_start(name)

    def stop(self, name: str) -> bool:
        """Stop a running or paused mission cleanly."""
        with self._lock:
            if not self._registry.exists(name):
                return False
            record = self._registry.get(name)
            if record.status not in (MissionStatus.RUNNING, MissionStatus.PAUSED):
                return False
            self._stop_record(record)
            return True

    def pause(self, name: str, reason: str = "") -> bool:
        """Pause a currently running mission."""
        with self._lock:
            if not self._registry.exists(name):
                return False
            record = self._registry.get(name)
            if record.status != MissionStatus.RUNNING:
                return False
            self._pause_record(record, reason)
            return True

    def resume(self, name: str) -> bool:
        """Resume a paused mission if resources are available."""
        with self._lock:
            if not self._registry.exists(name):
                return False
            record = self._registry.get(name)
            if record.status != MissionStatus.PAUSED:
                return False
            if not self._resources.try_acquire(name, record.required_resources):
                _log.debug("Cannot resume '%s': resources not yet free.", name)
                return False
            self._resume_record(record)
            return True

    def abort(self, name: str, reason: str = "") -> bool:
        """Immediately abort a mission regardless of its current status."""
        with self._lock:
            if not self._registry.exists(name):
                return False
            self._abort_record(self._registry.get(name), reason)
            return True

    def restart(self, name: str) -> bool:
        """Abort then restart a mission (operator request or watchdog)."""
        with self._lock:
            if not self._registry.exists(name):
                return False
            self._abort_record(self._registry.get(name), reason="restart")
            return self._do_start(name)

    def start_all(self) -> None:
        """Attempt to start every IDLE / READY mission."""
        with self._lock:
            for r in self._registry.find(status=MissionStatus.IDLE):
                self._do_start(r.name)
            for r in self._registry.find(status=MissionStatus.READY):
                self._do_start(r.name)

    def stop_all(self) -> None:
        """Stop every non-idle mission in ascending priority order."""
        with self._lock:
            candidates = (
                self._registry.find(status=MissionStatus.RUNNING)
                + self._registry.find(status=MissionStatus.PAUSED)
            )
            candidates.sort(key=lambda r: r.priority)
            for r in candidates:
                self._stop_record(r)

    def pause_all(self, reason: str = "") -> None:
        """Pause every running mission."""
        with self._lock:
            for r in self._registry.find(status=MissionStatus.RUNNING):
                self._pause_record(r, reason)

    def resume_all(self) -> None:
        """Resume every paused mission where resources are available."""
        with self._lock:
            for r in self._registry.find(status=MissionStatus.PAUSED):
                if self._resources.try_acquire(r.name, r.required_resources):
                    self._resume_record(r)

    def abort_all(self, reason: str = "") -> None:
        """Abort every active mission (used in emergency / shutdown)."""
        with self._lock:
            for r in self._registry.list():
                if r.status in (MissionStatus.RUNNING, MissionStatus.PAUSED):
                    self._abort_record(r, reason)

    # ------------------------------------------------------------------
    # Periodic scheduler tick — called from MissionManager run-loop
    # ------------------------------------------------------------------

    def update(self) -> None:
        """
        Enforce priority preemption, concurrency rules, and mission timeouts.

        Called every run-loop iteration (~50 Hz).
        """
        if self._fsm.is_terminal():
            return  # Safety states block scheduling

        with self._lock:
            self._enforce_exclusivity()
            self._auto_resume_paused()
            self._check_mission_timeouts()

    # ------------------------------------------------------------------
    # Internal — scheduling primitives
    # ------------------------------------------------------------------

    def _do_start(self, name: str) -> bool:
        """Core start logic. Must be called with self._lock held."""
        if not self._registry.exists(name):
            _log.error("Cannot start unknown mission '%s'.", name)
            return False

        record = self._registry.get(name)

        if record.status == MissionStatus.RUNNING:
            _log.warning("Mission '%s' is already RUNNING.", name)
            return False

        if record.exclusive:
            self._preempt_for_exclusive(record)

        if not record.concurrent:
            if self._registry.find(status=MissionStatus.RUNNING):
                _log.warning(
                    "Non-concurrent mission '%s' cannot start while others run.", name
                )
                return False

        if not self._resources.try_acquire(name, record.required_resources):
            _log.warning("Mission '%s' deferred: resources unavailable.", name)
            return False

        self._set_status(record, MissionStatus.READY)
        self._spawn_thread(record)
        self._mlog.mission_start(name, record.priority)
        return True

    def _stop_record(self, record: MissionRecord) -> None:
        self._mlog.mission_stop(record.name)
        self._call_safely(record.mission.stop, record.name, "stop")
        self._resources.release(record.name)
        record.last_stop_time = time.monotonic()
        self._set_status(record, MissionStatus.IDLE)

    def _pause_record(self, record: MissionRecord, reason: str = "") -> None:
        self._mlog.mission_pause(record.name, reason)
        self._call_safely(record.mission.pause, record.name, "pause")
        self._resources.release(record.name)
        record.last_pause_time = time.monotonic()
        self._set_status(record, MissionStatus.PAUSED)

    def _resume_record(self, record: MissionRecord) -> None:
        self._mlog.mission_resume(record.name)
        self._call_safely(record.mission.resume, record.name, "resume")
        self._set_status(record, MissionStatus.RUNNING)

    def _abort_record(self, record: MissionRecord, reason: str = "") -> None:
        """
        Safe abort procedure:
          1. Signal mission to abort
          2. Join thread with timeout
          3. Release all resources
          4. Update status
        """
        self._mlog.mission_abort(record.name, reason)
        self._call_safely(record.mission.abort, record.name, "abort")

        if record.is_alive:
            record.thread.join(timeout=self._cfg.thread_join_timeout_s)
            if record.is_alive:
                _log.error(
                    "Mission '%s' thread did not terminate after abort (%.1fs).",
                    record.name, self._cfg.thread_join_timeout_s,
                )

        self._resources.release(record.name)
        record.last_stop_time = time.monotonic()
        self._set_status(record, MissionStatus.ABORTED)

    # ------------------------------------------------------------------
    # Internal — enforcement helpers
    # ------------------------------------------------------------------

    def _preempt_for_exclusive(self, incoming: MissionRecord) -> None:
        """Pause all running missions whose priority is below *incoming*."""
        for record in self._registry.find(status=MissionStatus.RUNNING):
            if record.priority < incoming.priority:
                _log.info(
                    "[PREEMPT] '%s'(p=%d) preempted by '%s'(p=%d).",
                    record.name, record.priority,
                    incoming.name, incoming.priority,
                )
                self._pause_record(record, reason=f"preempted by {incoming.name}")

    def _enforce_exclusivity(self) -> None:
        """
        If an exclusive mission is running, pause lower-priority peers.
        Must be called with self._lock held.
        """
        running   = self._registry.find(status=MissionStatus.RUNNING)
        exclusive = [r for r in running if r.exclusive]
        if not exclusive:
            return
        excl = exclusive[0]
        for record in running:
            if record.name != excl.name and record.priority < excl.priority:
                self._pause_record(record, reason=f"exclusive mission {excl.name} active")

    def _auto_resume_paused(self) -> None:
        """
        When no exclusive mission is running, attempt to resume paused missions.
        Must be called with self._lock held.
        """
        if self._registry.find(status=MissionStatus.RUNNING, exclusive=True):
            return  # Still blocked by an exclusive mission
        for record in self._registry.find(status=MissionStatus.PAUSED):
            if self._resources.try_acquire(record.name, record.required_resources):
                self._resume_record(record)

    def _check_mission_timeouts(self) -> None:
        """Abort any running mission that has exceeded its mission_timeout."""
        for record in self._registry.find(status=MissionStatus.RUNNING):
            if record.timed_out():
                _log.warning(
                    "Mission '%s' timed out after %.1fs.",
                    record.name, record.mission_timeout,
                )
                self._abort_record(record, reason="mission_timeout")
                self._bus.publish(EventType.MISSION_ABORT, {"mission_name": record.name})

    # ------------------------------------------------------------------
    # Internal — thread management
    # ------------------------------------------------------------------

    def _spawn_thread(self, record: MissionRecord) -> None:
        """Create and start a mission thread. Guards against duplicates."""
        if record.is_alive:
            _log.warning("Mission '%s' already has a live thread.", record.name)
            return

        thread = threading.Thread(
            target=self._thread_wrapper,
            args=(record,),
            name=f"Mission-{record.name}",
            daemon=True,
        )
        record.thread         = thread
        record.last_heartbeat = time.monotonic()
        record.last_start_time = time.monotonic()
        self._set_status(record, MissionStatus.RUNNING)
        thread.start()
        _log.info("[THREAD] Spawned '%s'.", record.name)

    def _thread_wrapper(self, record: MissionRecord) -> None:
        """
        Executes inside the mission thread.
        Calls mission.run(), captures exceptions, and updates status on exit.
        """
        _log.info("[THREAD] Mission '%s' starting.", record.name)
        try:
            record.mission.run(
                watchdog_ping=lambda: self._ping(record),
            )
            with self._lock:
                if record.status == MissionStatus.RUNNING:
                    elapsed = record.elapsed_since_start
                    self._resources.release(record.name)
                    self._set_status(record, MissionStatus.COMPLETED)
                    self._mlog.mission_completed(record.name, elapsed)
        except Exception as exc:
            _log.exception("[THREAD] Mission '%s' raised an exception.", record.name)
            with self._lock:
                self._resources.release(record.name)
                self._set_status(record, MissionStatus.FAILED)
                self._mlog.mission_failed(record.name, exc)
            self._bus.publish(
                EventType.MISSION_ABORT, {"mission_name": record.name, "reason": str(exc)}
            )

    def _ping(self, record: MissionRecord) -> None:
        """Called by a mission thread to signal liveness to the watchdog."""
        record.last_heartbeat = time.monotonic()

    # ------------------------------------------------------------------
    # Internal — shared helpers
    # ------------------------------------------------------------------

    def _set_status(self, record: MissionRecord, status: MissionStatus) -> None:
        old           = record.status
        record.status = status
        self._mlog.status_transition(record.name, old, status)

    @staticmethod
    def _call_safely(fn: Callable, name: str, op: str) -> None:
        """Call *fn* and swallow exceptions, logging them instead."""
        try:
            fn()
        except Exception:
            _log.exception("Error calling %s() on mission '%s'.", op, name)


# ===========================================================================
# § 8  SAFETY MONITOR  — publish-only, no direct actions
# ===========================================================================

class SafetyMonitor:
    """
    Continuously evaluates vehicle health and publishes safety events.

    CRITICAL DESIGN CONTRACT:
      SafetyMonitor NEVER directly pauses, stops, aborts, or transitions
      any mission or system state.  It ONLY publishes events via the EventBus.
      MissionManager decides what to do in response.

    Thresholds are sourced from ManagerConfig (no magic numbers here).
    """

    def __init__(
        self,
        shared_data: SharedData,
        event_bus:   EventBus,
        cfg:         ManagerConfig,
    ) -> None:
        self._shared      = shared_data
        self._bus         = event_bus
        self._cfg         = cfg
        self._stop_event  = threading.Event()
        self._thread      = threading.Thread(
            target=self._run, name="SafetyMonitor", daemon=True
        )
        self._prev:       HealthSnapshot = HealthSnapshot()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        _log.info("SafetyMonitor: starting.")
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=self._cfg.subsystem_stop_timeout_s)
        _log.info("SafetyMonitor: stopped.")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                snap = self._collect()
                self._evaluate(snap)
                self._prev = snap
            except Exception:
                _log.exception("SafetyMonitor: evaluation error.")
            time.sleep(self._cfg.safety_poll_interval_s)

    def _collect(self) -> HealthSnapshot:
        telem  = self._shared.telemetry
        now    = time.monotonic()
        last_hb = self._shared.last_heartbeat_time or 0.0
        # The RC safety-channel PWM is stored in state; telemetry.py updates it from RC_CHANNELS.
        # rc_safety_pwm corresponds to the channel selected by config.RC_SAFETY_CHANNEL.
        vehicle_state = getattr(self._shared, "state", None)
        rc16_pwm = int(getattr(vehicle_state, "rc_safety_pwm", 0)) if vehicle_state else 0
        return HealthSnapshot(
            gps_fix          = bool(telem.get("gps_fix", False)),
            ekf_ok           = bool(telem.get("ekf_ok", True)),
            heartbeat_age_s  = now - last_hb,
            battery_pct      = float(telem.get("battery_pct", 100.0)),
            connection_ok    = bool(self._shared.connection_active),
            rc16_pwm         = rc16_pwm,
        )

    def _evaluate(self, snap: HealthSnapshot) -> None:
        """Compare snapshot against thresholds and publish events."""
        prev = self._prev

        # GPS transitions
        if prev.gps_fix and not snap.gps_fix:
            self._publish(EventType.GPS_LOST, {"source": "safety_monitor"})
        elif not prev.gps_fix and snap.gps_fix:
            self._publish(EventType.GPS_RECOVERED, {"source": "safety_monitor"})

        # Heartbeat
        if snap.heartbeat_age_s > self._cfg.heartbeat_timeout_s:
            self._publish(
                EventType.HEARTBEAT_TIMEOUT,
                {"age_s": snap.heartbeat_age_s},
            )

        # Connection
        if prev.connection_ok and not snap.connection_ok:
            self._publish(EventType.CONNECTION_LOST, {"source": "safety_monitor"})

        # Battery (critical takes precedence over low)
        if snap.battery_pct <= self._cfg.battery_critical_threshold_percent:
            self._publish(EventType.BATTERY_CRITICAL, {"pct": snap.battery_pct})
        elif snap.battery_pct <= self._cfg.battery_rtl_threshold_percent:
            self._publish(EventType.LOW_BATTERY, {"pct": snap.battery_pct})

        # EKF
        if prev.ekf_ok and not snap.ekf_ok:
            self._publish(EventType.EKF_ERROR, {"source": "safety_monitor"})

        # RC16 one-way threshold check: publish an event when PWM exceeds the threshold.
        if snap.rc16_pwm > self._cfg.rc16_safety_threshold_pwm and prev.rc16_pwm <= self._cfg.rc16_safety_threshold_pwm:
            self._publish(
                EventType.RC16_SAFETY_TRIGGERED,
                {"pwm": snap.rc16_pwm},
            )

    def _publish(self, event_type: EventType, payload: dict) -> None:
        _log.debug("SafetyMonitor publishing %s: %s", event_type.name, payload)
        self._bus.publish(event_type, payload)


# ===========================================================================
# § 9  MISSION WATCHDOG  — safe abort → release → wait → restart
# ===========================================================================

class MissionWatchdog:
    """
    Monitors mission thread liveness and implements safe recovery.

    Recovery sequence for a dead or frozen thread:
      1.  Abort mission (signal + join with timeout)
      2.  Release all held resources
      3.  Wait cfg.watchdog_restart_delay_s (allows navigator state to clear)
      4.  Restart if auto_restart=True and restart_counter < maximum_restarts
      5.  Otherwise mark FAILED and publish MISSION_ABORT

    Liveness is determined by:
      - Thread.is_alive()  (dead detection)
      - record.last_heartbeat age vs record.heartbeat_timeout  (freeze detection)
    """

    def __init__(
        self,
        registry:   MissionRegistry,
        scheduler:  MissionScheduler,
        event_bus:  EventBus,
        mlog:       MissionLogger,
        cfg:        ManagerConfig,
    ) -> None:
        self._registry  = registry
        self._scheduler = scheduler
        self._bus       = event_bus
        self._mlog      = mlog
        self._cfg       = cfg
        self._stop_event = threading.Event()
        self._thread     = threading.Thread(
            target=self._run, name="MissionWatchdog", daemon=True
        )

    def start(self) -> None:
        _log.info("MissionWatchdog: starting.")
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=self._cfg.subsystem_stop_timeout_s)
        _log.info("MissionWatchdog: stopped.")

    def ping(self, mission_name: str) -> None:
        """Relay a liveness ping from a mission thread."""
        try:
            self._registry.get(mission_name).last_heartbeat = time.monotonic()
        except KeyError:
            pass

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._tick()
            except Exception:
                _log.exception("MissionWatchdog: tick error.")
            time.sleep(self._cfg.watchdog_poll_interval_s)

    def _tick(self) -> None:
        now = time.monotonic()
        for record in self._registry.find(status=MissionStatus.RUNNING):
            if not record.is_alive:
                self._recover(record, reason="thread_dead")
            else:
                timeout = (
                    record.heartbeat_timeout
                    if record.heartbeat_timeout > 0
                    else self._cfg.watchdog_freeze_timeout_s
                )
                if (now - record.last_heartbeat) > timeout:
                    self._recover(record, reason="thread_frozen")

    def _recover(self, record: MissionRecord, reason: str) -> None:
        """
        Execute the safe recovery sequence in a temporary worker thread so
        the watchdog poll loop is never blocked by thread joins.
        """
        _log.warning(
            "MissionWatchdog: mission '%s' needs recovery (reason=%s).",
            record.name, reason,
        )
        worker = threading.Thread(
            target=self._recovery_sequence,
            args=(record, reason),
            name=f"Watchdog-Recover-{record.name}",
            daemon=True,
        )
        worker.start()

    def _recovery_sequence(self, record: MissionRecord, reason: str) -> None:
        """
        Runs in a dedicated thread to avoid blocking the watchdog loop.

        Step 1 — Abort
        Step 2 — Release resources (done inside abort)
        Step 3 — Wait restart delay (navigator state clears)
        Step 4 — Restart or give up
        """
        # Step 1 + 2: abort clears thread and releases resources
        self._scheduler.abort(record.name, reason=f"watchdog_{reason}")

        # Step 3: wait for safe restart window
        time.sleep(self._cfg.watchdog_restart_delay_s)

        # Step 4: restart decision
        if not record.auto_restart:
            _log.info(
                "Watchdog: '%s' auto_restart=False — not restarting.", record.name
            )
            return

        if record.restart_counter >= record.maximum_restarts:
            self._mlog.watchdog_give_up(record.name)
            record.status = MissionStatus.FAILED
            self._bus.publish(
                EventType.MISSION_ABORT,
                {"mission_name": record.name, "reason": "max_restarts_exceeded"},
            )
            return

        record.restart_counter += 1
        self._mlog.watchdog_restart(
            record.name, record.restart_counter, record.maximum_restarts, reason
        )
        self._scheduler.start(record.name)


# ===========================================================================
# § 10  EVENT DISPATCHER  — decorator-based handler registration
# ===========================================================================

# Sentinel attribute used by the @event decorator to tag handler methods
_EVENT_HANDLER_ATTR = "__event_types__"


def event(*event_types: EventType) -> Callable:
    """
    Decorator that marks a method as a handler for one or more EventTypes.

    Usage::

        class MyHandlers:
            @event(EventType.TARGET_FOUND, EventType.TARGET_LOST)
            def handle_target(self, payload: dict) -> None:
                ...

    EventDispatcher.register_handlers(instance) will auto-discover all
    decorated methods and subscribe them.
    """
    def decorator(fn: Callable) -> Callable:
        setattr(fn, _EVENT_HANDLER_ATTR, list(event_types))
        return fn
    return decorator


class EventDispatcher:
    """
    Subscribes decorated handler methods to the EventBus.

    Pattern:
      1.  Build a handler object (e.g. MissionEventHandlers) whose
          methods are decorated with @event(EventType.X).
      2.  Call dispatcher.register_handlers(handler_object).
      3.  Dispatcher scans the object, discovers decorated methods,
          and subscribes them to the EventBus.
      4.  Call dispatcher.unsubscribe_all() on shutdown.

    This replaces the large if/elif dispatch table from the original design.
    """

    def __init__(self, event_bus: EventBus) -> None:
        self._bus:          EventBus                                = event_bus
        # (event_type, handler_fn) pairs — stored for clean unsubscribe
        self._subscriptions: List[Tuple[EventType, Callable]]      = []

    def register_handlers(self, handler_obj: Any) -> None:
        """
        Discover all @event-decorated methods on *handler_obj* and
        subscribe them to the EventBus.
        """
        for attr_name in dir(handler_obj):
            if attr_name.startswith("__"):
                continue
            method = getattr(handler_obj, attr_name, None)
            if not callable(method):
                continue
            tagged_types: Optional[List[EventType]] = getattr(
                method, _EVENT_HANDLER_ATTR, None
            )
            if tagged_types is None:
                continue
            for et in tagged_types:
                self._bus.subscribe(et, method)
                self._subscriptions.append((et, method))
                _log.debug(
                    "EventDispatcher: subscribed %s.%s → %s.",
                    type(handler_obj).__name__, attr_name, et.name,
                )

        _log.info(
            "EventDispatcher: registered %d handler subscriptions from %s.",
            len(self._subscriptions), type(handler_obj).__name__,
        )

    def unsubscribe_all(self) -> None:
        """Remove every subscription registered by this dispatcher."""
        for et, fn in self._subscriptions:
            self._bus.unsubscribe(et, fn)
        count = len(self._subscriptions)
        self._subscriptions.clear()
        _log.info("EventDispatcher: removed %d subscriptions.", count)


# ===========================================================================
# § 11  MISSION EVENT HANDLERS
#         Encapsulates all EventBus reactions.
#         Methods decorated with @event() are auto-discovered by
#         EventDispatcher.register_handlers().
# ===========================================================================

class MissionEventHandlers:
    """
    Reactive handlers for all EventBus events relevant to mission management.

    Injected dependencies:
      scheduler  — to enqueue start / stop / pause / resume / abort commands
      fsm        — to drive SystemState transitions
      registry   — to query mission state
      resources  — to query resource ownership
      mlog       — structured logging

    Commands that must happen on the run-loop thread are enqueued via
    _enqueue(); handlers on the EventBus thread never call scheduler
    directly (avoids lock contention and potential deadlocks).
    """

    def __init__(
        self,
        scheduler:      MissionScheduler,
        fsm:            MissionStateMachine,
        registry:       MissionRegistry,
        resources:      ResourceManager,
        mlog:           MissionLogger,
        pending_queue:  List[Callable[[], None]],
        pending_lock:   threading.Lock,
        shared_data:    "Any | None" = None,
    ) -> None:
        self._scheduler  = scheduler
        self._fsm        = fsm
        self._registry   = registry
        self._resources  = resources
        self._mlog       = mlog
        self._queue      = pending_queue
        self._qlock      = pending_lock
        self._shared     = shared_data

    # ------------------------------------------------------------------
    # Target / AI events
    # ------------------------------------------------------------------

    @event(EventType.TARGET_FOUND)
    def on_target_found(self, payload: dict) -> None:
        self._mlog.event_received("TARGET_FOUND", f"id={payload.get('target_id','?')}")

    @event(EventType.TARGET_LOST)
    def on_target_lost(self, payload: dict) -> None:
        self._mlog.event_received("TARGET_LOST", f"id={payload.get('target_id','?')}")

    # ------------------------------------------------------------------
    # Navigation events
    # ------------------------------------------------------------------

    @event(EventType.WAYPOINT_REACHED)
    def on_waypoint_reached(self, payload: dict) -> None:
        self._mlog.event_received("WAYPOINT_REACHED", f"idx={payload.get('waypoint_index',-1)}")

    @event(EventType.WAYPOINT_TIMEOUT)
    def on_waypoint_timeout(self, payload: dict) -> None:
        idx = payload.get("waypoint_index", -1)
        self._mlog.event_received("WAYPOINT_TIMEOUT", f"idx={idx}")
        nav_owner = self._resources.owner_of(Resource.NAVIGATOR)
        if nav_owner:
            self._enqueue(lambda n=nav_owner: self._scheduler.pause(n, "waypoint_timeout"))

    # ------------------------------------------------------------------
    # Mission lifecycle events
    # ------------------------------------------------------------------

    @event(EventType.MISSION_FINISHED)
    def on_mission_finished(self, payload: dict) -> None:
        name = payload.get("mission_name", "")
        self._mlog.event_received("MISSION_FINISHED", name)
        if name and self._registry.exists(name):
            def _handle(n: str = name) -> None:
                record = self._registry.get(n)
                if record.status == MissionStatus.RUNNING:
                    elapsed = record.elapsed_since_start
                    self._resources.release(n)
                    record.status = MissionStatus.COMPLETED
                    self._mlog.mission_completed(n, elapsed)
            self._enqueue(_handle)

    @event(EventType.MISSION_ABORT)
    def on_mission_abort(self, payload: dict) -> None:
        name = payload.get("mission_name", "")
        reason = payload.get("reason", "")
        self._mlog.event_received("MISSION_ABORT", f"'{name}' reason={reason}")
        if name:
            self._enqueue(lambda n=name, r=reason: self._scheduler.abort(n, r))

    # ------------------------------------------------------------------
    # Payload events
    # ------------------------------------------------------------------

    @event(EventType.PAYLOAD_RELEASED)
    def on_payload_released(self, payload: dict) -> None:
        self._mlog.event_received("PAYLOAD_RELEASED")
        # Find the mission currently owning the payload servo and mark it done
        servo_owner = self._resources.owner_of(Resource.PAYLOAD_SERVO)
        if servo_owner and self._registry.exists(servo_owner):
            def _handle(n: str = servo_owner) -> None:
                record = self._registry.get(n)
                if record.status == MissionStatus.RUNNING:
                    elapsed = record.elapsed_since_start
                    self._resources.release(n)
                    record.status = MissionStatus.COMPLETED
                    self._mlog.mission_completed(n, elapsed)
            self._enqueue(_handle)

    # ------------------------------------------------------------------
    # Safety events (published by SafetyMonitor — no callbacks needed)
    # ------------------------------------------------------------------

    @event(EventType.GPS_LOST)
    def on_gps_lost(self, payload: dict) -> None:
        self._mlog.event_received("GPS_LOST")
        self._enqueue(lambda: self._scheduler.pause_all("gps_lost"))

    @event(EventType.GPS_RECOVERED)
    def on_gps_recovered(self, payload: dict) -> None:
        self._mlog.event_received("GPS_RECOVERED")
        self._enqueue(self._scheduler.resume_all)

    @event(EventType.LOW_BATTERY)
    def on_low_battery(self, payload: dict) -> None:
        pct = payload.get("pct", 0.0)
        self._mlog.event_received("LOW_BATTERY", f"{pct:.1f}%")
        def _handle() -> None:
            self._scheduler.stop_all()
            self._fsm.transition(SystemState.RETURN_HOME)
        self._enqueue(_handle)

    @event(EventType.BATTERY_CRITICAL)
    def on_battery_critical(self, payload: dict) -> None:
        pct = payload.get("pct", 0.0)
        self._mlog.event_received("BATTERY_CRITICAL", f"{pct:.1f}%")
        def _handle() -> None:
            self._scheduler.abort_all("battery_critical")
            self._fsm.transition(SystemState.EMERGENCY)
        self._enqueue(_handle)

    @event(EventType.HEARTBEAT_TIMEOUT)
    def on_heartbeat_timeout(self, payload: dict) -> None:
        self._mlog.event_received("HEARTBEAT_TIMEOUT", f"age={payload.get('age_s',0):.1f}s")
        self._enqueue(self._handle_connection_loss)

    @event(EventType.CONNECTION_LOST)
    def on_connection_lost(self, payload: dict) -> None:
        self._mlog.event_received("CONNECTION_LOST")
        self._enqueue(self._handle_connection_loss)

    @event(EventType.EKF_ERROR)
    def on_ekf_error(self, payload: dict) -> None:
        self._mlog.event_received("EKF_ERROR")
        def _handle() -> None:
            self._scheduler.pause_all("ekf_error")
            self._fsm.transition(SystemState.EMERGENCY)
        self._enqueue(_handle)

    @event(EventType.EMERGENCY)
    def on_emergency(self, payload: dict) -> None:
        reason = payload.get("reason", "unspecified")
        self._mlog.event_received("EMERGENCY", reason)
        def _handle() -> None:
            self._scheduler.abort_all(f"emergency:{reason}")
            self._fsm.transition(SystemState.EMERGENCY)
        self._enqueue(_handle)

    @event(EventType.ARM_MAX_ATTEMPTS_EXCEEDED)
    def on_arm_max_attempts_exceeded(self, payload: dict) -> None:
        attempts = payload.get("attempts", "?")
        self._mlog.event_received("ARM_MAX_ATTEMPTS_EXCEEDED", f"attempts={attempts}")
        self._enqueue(self._handle_safety_shutdown)

    @event(EventType.RC16_SAFETY_TRIGGERED)
    def on_rc16_safety_triggered(self, payload: dict) -> None:
        pwm = payload.get("pwm", "?")
        self._mlog.event_received("RC16_SAFETY_TRIGGERED", f"pwm={pwm}")
        # RTL and shutdown are disabled here. When RC16 triggers, only all missions
        # are paused; flight continues and RTL is not requested.
        def _handle() -> None:
            self._scheduler.pause_all("rc16_safety_triggered")
            _log.warning("[RC16Safety] RC16 safety trigger activated — missions paused; RTL is NOT requested.")
        self._enqueue(_handle)

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _handle_safety_shutdown(self) -> None:
        import time as _time
        import config as _cfg
        from core.restart_launcher import schedule_restart

        _log.warning("[SafetyShutdown] Safety shutdown started.")

        # --- Step 1: Cancel active missions ---
        self._scheduler.abort_all("safety_shutdown")
        self._fsm.transition(SystemState.EMERGENCY)

        # --- Step 2: Switch to RTL mode ---
        rtl_ok = False
        mavlink_mgr = getattr(self._shared, "mavlink_manager", None)
        if mavlink_mgr is not None:
            try:
                rtl_ok = mavlink_mgr.rtl() if hasattr(mavlink_mgr, "rtl") else mavlink_mgr.set_mode("RTL")
                if rtl_ok:
                    _log.warning("[SafetyShutdown] Vehicle switched to RTL mode.")
                else:
                    _log.error("[SafetyShutdown] RTL command failed — shutdown will continue.")
            except Exception:
                _log.exception("[SafetyShutdown] Could not send the RTL command.")
        else:
            _log.error("[SafetyShutdown] MAVLink connection not available; RTL command could not be sent.")

        # --- Step 3: Wait for RTL ---
        rtl_wait = float(getattr(_cfg, "SAFETY_RTL_WAIT_S", 10.0))
        if rtl_ok and rtl_wait > 0:
            _log.warning(
                "[SafetyShutdown] Waiting %.1f seconds after RTL...", rtl_wait
            )
            _time.sleep(rtl_wait)

        # --- Step 4: Close camera and Pixhawk connections ---
        # Order: mission (including camera) -> servos -> MAVLink.
        SHUTDOWN_ORDER = (
            "autonomous_mission",   # The camera thread is stopped with this component.
            "camera",               # Stop the separate camera service, if present.
            "servo_manager",
            "mavlink_manager",
        )
        for attr in SHUTDOWN_ORDER:
            component = getattr(self._shared, attr, None)
            if component is None:
                continue
            try:
                stopper = getattr(component, "stop", None)
                if callable(stopper):
                    stopper()
                    _log.warning("[SafetyShutdown] Closed: %s", attr)
                elif attr == "mavlink_manager":
                    component.disconnect()
                    _log.warning("[SafetyShutdown] MAVLink connection closed.")
            except Exception:
                self._mlog.event_received("SAFETY_SHUTDOWN_COMPONENT_ERROR", attr)
                _log.exception("[SafetyShutdown] Shutdown error: %s", attr)

        # Ensure that MAVLink is disconnected as well.
        if mavlink_mgr is not None:
            try:
                mavlink_mgr.disconnect()
            except Exception:
                _log.exception("[SafetyShutdown] Error during final MAVLink disconnect")

        # --- Step 5: Schedule the safety script ---
        delay = float(getattr(_cfg, "RESTART_DELAY_S", 2.0))
        script = str(getattr(_cfg, "SAFETY_SCRIPT_PATH", "")).strip()
        if script:
            _log.warning(
                "[SafetyShutdown] Restart script will run in %.1f seconds: %s",
                delay,
                script,
            )
            schedule_restart(script, delay_s=delay)
        else:
            _log.warning(
                "[SafetyShutdown] Because SAFETY_SCRIPT_PATH is not configured, "
                "automatic restart was skipped."
            )

        # --- Step 6: Exit main.py ---
        # schedule_restart() launches a separate helper process, so
        # the target script can run on schedule even if the main process exits.
        # Sending SIGTERM triggers the signal handler in app.py:
        # _stop_event.set() -> app.wait() returns -> main() runs its finally block
        # -> app.stop() -> the process exits.
        _log.warning(
            "[SafetyShutdown] Stopping main.py (sending SIGTERM)."
        )
        import os as _os
        import signal as _signal
        try:
            _os.kill(_os.getpid(), _signal.SIGTERM)
        except Exception:
            # Exit directly if SIGTERM cannot be sent.
            import sys as _sys
            _log.warning("[SafetyShutdown] Could not send SIGTERM — calling sys.exit(0).")
            _sys.exit(0)

    def _handle_connection_loss(self) -> None:
        self._scheduler.abort_all("connection_lost")
        self._fsm.transition(SystemState.ABORTED)

    def _enqueue(self, cmd: Callable[[], None]) -> None:
        with self._qlock:
            self._queue.append(cmd)


# ===========================================================================
# § 12  MISSION MANAGER  — thin orchestrator (~250 lines of logic)
# ===========================================================================

class MissionManager:
    """
    Central flight-computer mission orchestrator.

    Role: *orchestrator only* — contains almost no business logic.
    All domain decisions are delegated to:

      MissionRegistry     — stores MissionRecord objects
      MissionScheduler    — start / stop / pause / resume / abort
      MissionStateMachine — SystemState FSM
      ResourceManager     — hardware/software resource ownership
      SafetyMonitor       — vehicle health → publishes events (no actions)
      MissionWatchdog     — thread liveness → safe restart
      EventDispatcher     — @event decorator routing to MissionEventHandlers
      MissionLogger       — structured timestamped audit log

    Adding a new mission type:
      1.  class NewMission(BaseMission): ...
      2.  manager.register(MissionRecord(mission=NewMission(), ...))
      No other changes required.

    Usage::

        cfg = ManagerConfig(...)           # from config.py
        mm  = MissionManager(bus, shared, cfg)
        mm.register(MissionRecord(mission=lap_mission_instance,     priority=cfg.priority_lap, ...))
        mm.register(MissionRecord(mission=payload_mission_instance, priority=cfg.priority_payload, ...))
        mm.start()
        mm.start_mission("LapMission")   # name comes from mission.name, not this file
        # ...
        mm.stop()
    """

    def __init__(
        self,
        event_bus:   EventBus,
        shared_data: SharedData,
        cfg:         ManagerConfig,
    ) -> None:
        """
        Parameters
        ----------
        event_bus   : EventBus      — shared pub/sub bus
        shared_data : SharedData    — live telemetry surface (GPS, battery …)
        cfg         : ManagerConfig — all tuneable constants (from config.py)
        """
        self._bus     = event_bus
        self._shared  = shared_data
        self._cfg     = cfg

        # ── Dedicated logger ──────────────────────────────────────────
        self._mlog = MissionLogger(_log)

        # ── Core subsystems ───────────────────────────────────────────
        self._registry  = MissionRegistry(self._mlog)
        self._resources = ResourceManager(self._mlog)
        self._fsm       = MissionStateMachine(self._mlog)

        self._scheduler = MissionScheduler(
            registry   = self._registry,
            resources  = self._resources,
            fsm        = self._fsm,
            event_bus  = self._bus,
            mlog       = self._mlog,
            cfg        = self._cfg,
        )

        # Pending commands from event handlers → drained in run-loop
        self._pending_cmds: List[Callable[[], None]] = []
        self._pending_lock  = threading.Lock()

        self._handlers = MissionEventHandlers(
            scheduler     = self._scheduler,
            fsm           = self._fsm,
            registry      = self._registry,
            resources     = self._resources,
            mlog          = self._mlog,
            pending_queue = self._pending_cmds,
            pending_lock  = self._pending_lock,
            shared_data   = self._shared,
        )
        self._dispatcher = EventDispatcher(self._bus)

        self._safety = SafetyMonitor(
            shared_data = self._shared,
            event_bus   = self._bus,
            cfg         = self._cfg,
        )

        self._watchdog = MissionWatchdog(
            registry   = self._registry,
            scheduler  = self._scheduler,
            event_bus  = self._bus,
            mlog       = self._mlog,
            cfg        = self._cfg,
        )

        self._running    = False
        self._run_thread: Optional[threading.Thread] = None
        _log.info("MissionManager created.")

    # ==================================================================
    # Public lifecycle
    # ==================================================================

    def start(self) -> None:
        """
        Start all subsystems and enter the supervisor run-loop.
        Safe to call only once.
        """
        if self._running:
            _log.warning("MissionManager.start() called while already running.")
            return

        self._fsm.transition(SystemState.INITIALIZING)
        self._dispatcher.register_handlers(self._handlers)
        self._safety.start()
        self._watchdog.start()

        self._running    = True
        self._run_thread = threading.Thread(
            target=self._run_loop, name="MissionManager", daemon=True
        )
        self._run_thread.start()
        self._fsm.transition(SystemState.READY)
        _log.info("MissionManager: READY.")

    def stop(self) -> None:
        """Gracefully stop all missions and subsystems. Blocks until clean."""
        _log.info("MissionManager: stopping.")
        self._running = False

        self._scheduler.stop_all()
        self._watchdog.stop()
        self._safety.stop()
        self._dispatcher.unsubscribe_all()

        if self._run_thread:
            self._run_thread.join(timeout=self._cfg.subsystem_stop_timeout_s)

        self._fsm.transition(SystemState.FINISHED)
        _log.info("MissionManager: stopped.")

    # ==================================================================
    # Mission registration (public API — no instantiation here)
    # ==================================================================

    def register(self, record: MissionRecord) -> None:
        """Register a mission created externally (e.g. in main.py)."""
        self._registry.register(record)

    def unregister(self, name: str) -> None:
        """Unregister a mission by name (must not be RUNNING or PAUSED)."""
        self._registry.unregister(name)

    def register_resource(self, key: str) -> None:
        """Dynamically register an additional lockable resource."""
        self._resources.register_resource(key)

    # ==================================================================
    # Scheduler — public API (thin pass-through to MissionScheduler)
    # ==================================================================

    def start_mission(self, name: str)  -> bool: return self._scheduler.start(name)
    def stop_mission(self, name: str)   -> bool: return self._scheduler.stop(name)
    def pause_mission(self, name: str)  -> bool: return self._scheduler.pause(name)
    def resume_mission(self, name: str) -> bool: return self._scheduler.resume(name)
    def abort_mission(self, name: str)  -> bool: return self._scheduler.abort(name)
    def restart_mission(self, name: str)-> bool: return self._scheduler.restart(name)
    def start_all(self)  -> None: self._scheduler.start_all()
    def stop_all(self)   -> None: self._scheduler.stop_all()
    def pause_all(self)  -> None: self._scheduler.pause_all()
    def resume_all(self) -> None: self._scheduler.resume_all()

    # ==================================================================
    # Diagnostics (public)
    # ==================================================================

    @property
    def state(self) -> SystemState:
        """Current system FSM state."""
        return self._fsm.state

    def mission_status(self, name: str) -> MissionStatus:
        return self._registry.get(name).status

    def resource_snapshot(self) -> Dict[str, Optional[str]]:
        return self._resources.snapshot()

    def list_missions(self) -> List[MissionRecord]:
        return self._registry.list()

    # ==================================================================
    # Internal — run loop
    # ==================================================================

    def _run_loop(self) -> None:
        """
        Supervisor loop at cfg.run_loop_interval_s cadence (~50 Hz default).

        Each tick:
          1. Drain the pending-command queue from event handlers
          2. Tick the scheduler (preemption, timeouts, auto-resume)
          3. Derive system state from mission set
        """
        _log.info("MissionManager: run-loop started.")
        while self._running:
            try:
                self._drain_pending()
                self._scheduler.update()
                self._fsm.derive_from_missions(self._registry)
            except Exception:
                _log.exception("MissionManager: uncaught run-loop error.")
            time.sleep(self._cfg.run_loop_interval_s)
        _log.info("MissionManager: run-loop exited.")

    def _drain_pending(self) -> None:
        """Execute deferred commands enqueued by event handlers."""
        with self._pending_lock:
            cmds, self._pending_cmds[:] = list(self._pending_cmds), []
        for cmd in cmds:
            try:
                cmd()
            except Exception:
                _log.exception("MissionManager: error in deferred command.")
