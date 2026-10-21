"""Runs the mapping mission and follows generated mapping waypoints."""

from __future__ import annotations

import logging
import math
import threading
import time

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Sequence, Tuple

from missions.mission_path import MissionPath
from missions.base_mission import BaseMission
from missions.waypoint import (
    Waypoint,
    WaypointStatus,
    CameraRequest,
)
from mapping.mapping_grid import (
    MappingGrid,
    MappingGridConfig,
)


from core.events import EventBus, Event, EventType
from core.shared_data import SharedData


logger = logging.getLogger(__name__)


Coordinate = Tuple[float, float]


# ============================================================================
# Mapping state
# ============================================================================

class MappingState(Enum):
    """Finite-state machine states for MappingMission."""

    IDLE = auto()
    READY = auto()

    GENERATING_GRID = auto()
    GRID_READY = auto()

    WAITING_FIRST_WAYPOINT = auto()
    GOING_TO_WAYPOINT = auto()
    WAITING_NAVIGATOR = auto()

    WAITING_CAMERA = auto()
    PHOTO_REQUESTED = auto()
    PHOTO_CAPTURED = auto()

    WAYPOINT_REACHED = auto()
    NEXT_WAYPOINT = auto()

    PAUSED = auto()

    FINISHED = auto()
    ABORTED = auto()
    FAILED = auto()


# ============================================================================
# Statistics
# ============================================================================

@dataclass
class MappingStatistics:
    """Runtime statistics for a mapping mission."""

    mission_id: str = "MappingMission"

    total_waypoints: int = 0
    current_waypoint_index: int = 0
    remaining_waypoints: int = 0

    visited_waypoints: int = 0

    successful_waypoints: int = 0
    failed_waypoints: int = 0

    photos_requested: int = 0
    photos_captured: int = 0
    photo_failures: int = 0

    waypoint_retries: int = 0
    timeout_count: int = 0

    pause_count: int = 0
    resume_count: int = 0

    mission_start_time: Optional[float] = None
    mission_finish_time: Optional[float] = None

    last_waypoint_time: Optional[float] = None
    last_photo_time: Optional[float] = None

    total_distance_m: float = 0.0
    remaining_distance_m: float = 0.0

    last_heartbeat: Optional[float] = None

    _waypoint_durations: List[float] = field(
        default_factory=list,
        repr=False,
    )

    average_waypoint_time_s: float = 0.0

    def duration(self) -> Optional[float]:
        """Return mission duration in seconds."""

        if self.mission_start_time is None:
            return None

        end = (
            self.mission_finish_time
            if self.mission_finish_time is not None
            else time.monotonic()
        )

        return end - self.mission_start_time

    def record_waypoint_duration(
        self,
        duration_s: float,
    ) -> None:
        """Record waypoint transit duration."""

        if duration_s < 0:
            return

        self._waypoint_durations.append(duration_s)

        self.average_waypoint_time_s = (
            sum(self._waypoint_durations)
            / len(self._waypoint_durations)
        )


# ============================================================================
# Camera request state
# ============================================================================

@dataclass
class _CameraState:
    """Internal camera request tracking."""

    requested: bool = False
    completed: bool = False
    failed: bool = False

    request_time: Optional[float] = None
    completion_time: Optional[float] = None

    waypoint_index: int = -1


# ============================================================================
# Pause context
# ============================================================================

@dataclass
class _PausedContext:
    """State captured when the mission is paused."""

    state_before_pause: MappingState

    waypoint_index: int

    waypoint: Optional[Waypoint]

    camera_pending: bool


# ============================================================================
# MappingMission
# ============================================================================

class MappingMission(BaseMission):
    """
    Autonomous aerial mapping mission.

    MappingMission uses:

        MappingGrid
            ↓
        Waypoint
            ↓
        MissionPath
            ↓
        Navigator

    Camera interaction:

        MappingMission
            ↓
        CAMERA_CAPTURE_REQUEST
            ↓
        Camera subsystem

    MappingMission never accesses the camera hardware directly.
    """

    def __init__(
        self,
        shared_data: SharedData,
        mission_path: MissionPath,
        navigator: Any,
        event_bus: EventBus,
        mapping_polygon: Sequence[Coordinate],
        grid_config: Optional[MappingGridConfig] = None,
        logger_instance: Optional[logging.Logger] = None,
        start_point: Optional[Coordinate] = None,
        photo_required: bool = True,
    ) -> None:

        super().__init__(shared_data, "MappingMission")

        self.name = "MappingMission"

        self._shared = shared_data
        self._path = mission_path
        self._navigator = navigator
        self._event_bus = event_bus

        self._logger = logger_instance or logger

        self._mapping_polygon = list(mapping_polygon)

        self._grid_config = (
            grid_config
            or MappingGridConfig()
        )

        self._start_point = start_point

        self._photo_required = photo_required

        self._lock = threading.RLock()

        self._state = MappingState.IDLE

        self._statistics = MappingStatistics()

        self._grid: Optional[MappingGrid] = None

        self._current_waypoint: Optional[Waypoint] = None

        self._waypoint_start_time: Optional[float] = None

        self._camera = _CameraState()

        self._paused_context: Optional[
            _PausedContext
        ] = None

        self._waypoint_retry_count = 0

        self._max_waypoint_retries = 3

        self._waypoint_timeout_s = 120.0

        self._photo_timeout_s = 5.0

        self._heartbeat_counter = 0

        self._subscriptions: List[
            Tuple[EventType, Any]
        ] = []

        self._dispatch = {
            MappingState.READY:
                self._step_ready,

            MappingState.GENERATING_GRID:
                self._step_generating_grid,

            MappingState.GRID_READY:
                self._step_grid_ready,

            MappingState.WAITING_FIRST_WAYPOINT:
                self._step_waiting_first_waypoint,

            MappingState.GOING_TO_WAYPOINT:
                self._step_going_to_waypoint,

            MappingState.WAITING_NAVIGATOR:
                self._step_waiting_navigator,

            MappingState.WAITING_CAMERA:
                self._step_waiting_camera,

            MappingState.PHOTO_REQUESTED:
                self._step_photo_requested,

            MappingState.PHOTO_CAPTURED:
                self._step_photo_captured,

            MappingState.WAYPOINT_REACHED:
                self._step_waypoint_reached,

            MappingState.NEXT_WAYPOINT:
                self._step_next_waypoint,
        }

        self._subscribe_to_events()

    # ========================================================================
    # Lifecycle
    # ========================================================================

    def initialize(self) -> None:
        """
        Validate the mapping mission and prepare grid generation.
        """

        with self._lock:

            if self._state not in (
                MappingState.IDLE,
                MappingState.FAILED,
                MappingState.ABORTED,
            ):
                return

            self._validate_polygon()

            self._statistics = MappingStatistics(
                mission_id=self.name,
            )

            self._state = MappingState.READY

            self._log(
                "MappingMission initialized"
            )

    # ------------------------------------------------------------------------

    def update(self) -> None:
        """
        Execute one non-blocking mission update.

        This method is expected to be called by MissionManager's
        supervisor thread.
        """

        self._beat_heartbeat()

        with self._lock:

            state = self._state

        try:

            handler = self._dispatch.get(state)

            if handler is not None:
                handler()

            self._update_remaining_statistics()

        except Exception as exc:

            self._fail(
                f"Unexpected exception in mapping update: {exc!r}"
            )

    # ------------------------------------------------------------------------

    def cleanup(self) -> None:
        """Release transient mission state."""

        with self._lock:

            self._unsubscribe_events()

            self._current_waypoint = None

            self._waypoint_start_time = None

            self._paused_context = None

            self._camera = _CameraState()

            self._log(
                "MappingMission cleanup complete"
            )

    # ========================================================================
    # Pause / Resume / Abort
    # ========================================================================

    def pause(self) -> None:
        """Pause mapping without losing waypoint progress."""

        with self._lock:

            if self._state in (
                MappingState.FINISHED,
                MappingState.ABORTED,
                MappingState.FAILED,
            ):
                return

            self._paused_context = _PausedContext(
                state_before_pause=self._state,
                waypoint_index=(
                    self._statistics.current_waypoint_index
                ),
                waypoint=self._current_waypoint,
                camera_pending=self._camera.requested
                and not self._camera.completed,
            )

            self._state = MappingState.PAUSED

            self._statistics.pause_count += 1

            navigator = self._navigator

        if navigator is not None:

            try:
                navigator.cancel()
            except Exception:

                self._log_exception(
                    "Navigator cancel failed during pause"
                )

        self._log(
            "Mapping mission paused"
        )

    # ------------------------------------------------------------------------

    def resume(self) -> None:
        """Resume mapping from the paused waypoint."""

        with self._lock:

            if self._state != MappingState.PAUSED:
                return

            context = self._paused_context

            if context is None:

                self._state = (
                    MappingState.WAITING_FIRST_WAYPOINT
                )

            else:

                self._current_waypoint = (
                    context.waypoint
                )

                if context.camera_pending:

                    self._state = (
                        MappingState.WAITING_CAMERA
                    )

                else:

                    self._state = (
                        MappingState.GOING_TO_WAYPOINT
                    )

            self._statistics.resume_count += 1

        self._log(
            "Mapping mission resumed"
        )

    # ------------------------------------------------------------------------

    def abort(self) -> None:
        """Abort mapping mission safely."""

        navigator = self._navigator

        with self._lock:

            if self._state in (
                MappingState.FINISHED,
                MappingState.ABORTED,
            ):
                return

            self._state = MappingState.ABORTED

            self._statistics.mission_finish_time = (
                time.monotonic()
            )

        if navigator is not None:

            try:
                navigator.cancel()
            except Exception:

                self._log_exception(
                    "Navigator cancel failed during abort"
                )

        self._publish(
            EventType.MISSION_ABORT,
            {
                "mission_name": self.name,
                "reason": "mapping_abort",
            },
        )

        self._log(
            "Mapping mission aborted"
        )

    # ========================================================================
    # State steps
    # ========================================================================

    def _step_ready(self) -> None:
        """Begin grid generation."""

        with self._lock:

            self._state = (
                MappingState.GENERATING_GRID
            )

    # ------------------------------------------------------------------------

    def _step_generating_grid(self) -> None:
        """Generate the mapping grid."""

        try:

            grid = MappingGrid(
                polygon=self._mapping_polygon,
                config=self._grid_config,
            )

            waypoints = grid.generate_waypoints(
                start_point=self._start_point
            )

            if not waypoints:

                raise RuntimeError(
                    "MappingGrid generated zero waypoints"
                )

            self._grid = grid

            self._load_waypoints(
                waypoints
            )

            with self._lock:

                self._statistics.total_waypoints = (
                    len(waypoints)
                )

                self._statistics.remaining_waypoints = (
                    len(waypoints)
                )

                self._statistics.total_distance_m = (
                    grid.total_distance()
                )

                self._state = (
                    MappingState.GRID_READY
                )

            self._log(
                f"Mapping grid generated: "
                f"{len(waypoints)} waypoints"
            )

        except Exception as exc:

            self._fail(
                f"Grid generation failed: {exc!r}"
            )

    # ------------------------------------------------------------------------

    def _step_grid_ready(self) -> None:
        """Prepare the first waypoint."""

        with self._lock:

            self._state = (
                MappingState.WAITING_FIRST_WAYPOINT
            )

    # ------------------------------------------------------------------------

    def _step_waiting_first_waypoint(self) -> None:
        """Send the first mapping waypoint."""

        self._send_current_waypoint()

    # ------------------------------------------------------------------------

    def _step_going_to_waypoint(self) -> None:
        """Move from command state to navigator monitoring."""

        with self._lock:

            self._state = (
                MappingState.WAITING_NAVIGATOR
            )

    # ------------------------------------------------------------------------

    def _step_waiting_navigator(self) -> None:
        """Monitor Navigator until waypoint arrival."""

        waypoint = self._current_waypoint

        if waypoint is None:

            self._fail(
                "No current waypoint while navigating"
            )

            return

        navigator = self._navigator

        if navigator is None:

            self._fail(
                "Navigator is unavailable"
            )

            return

        try:

            if navigator.finished():

                with self._lock:

                    self._state = (
                        MappingState.WAYPOINT_REACHED
                    )

                return

        except Exception:
            pass

        try:

            if navigator.failed():

                self._handle_navigation_failure(
                    "Navigator reported failure"
                )

                return

        except Exception:
            pass

        with self._lock:

            start_time = (
                self._waypoint_start_time
            )

        if start_time is None:
            return

        elapsed = (
            time.monotonic()
            - start_time
        )

        if elapsed > self._waypoint_timeout_s:

            self._statistics.timeout_count += 1

            self._handle_navigation_failure(
                "Waypoint timeout"
            )

    # ------------------------------------------------------------------------

    def _step_waiting_camera(self) -> None:
        """Wait for camera result."""

        if not self._photo_required:

            with self._lock:

                self._camera.completed = True

                self._state = (
                    MappingState.PHOTO_CAPTURED
                )

            return

        with self._lock:

            request_time = (
                self._camera.request_time
            )

            completed = (
                self._camera.completed
            )

            failed = (
                self._camera.failed
            )

        if completed:

            with self._lock:

                self._state = (
                    MappingState.PHOTO_CAPTURED
                )

            return

        if failed:

            self._handle_photo_failure(
                "Camera reported failure"
            )

            return

        if request_time is None:
            return

        if (
            time.monotonic()
            - request_time
            > self._photo_timeout_s
        ):

            self._handle_photo_failure(
                "Camera capture timeout"
            )

    # ------------------------------------------------------------------------

    def _step_photo_requested(self) -> None:
        """Wait for asynchronous camera response."""

        with self._lock:

            self._state = (
                MappingState.WAITING_CAMERA
            )

    # ------------------------------------------------------------------------

    def _step_photo_captured(self) -> None:
        """Photo completed, waypoint can be finalized."""

        with self._lock:

            self._statistics.photos_captured += 1

            self._statistics.last_photo_time = (
                time.monotonic()
            )

            self._state = (
                MappingState.WAYPOINT_REACHED
            )

    # ------------------------------------------------------------------------

    def _step_waypoint_reached(self) -> None:
        """Record waypoint and move to the next one."""

        waypoint = self._current_waypoint

        now = time.monotonic()

        with self._lock:

            self._statistics.visited_waypoints += 1

            self._statistics.successful_waypoints += 1

            self._statistics.last_waypoint_time = now

            if self._waypoint_start_time is not None:

                self._statistics.record_waypoint_duration(
                    now - self._waypoint_start_time
                )

        if waypoint is not None:

            try:
                waypoint.mark_reached()
            except Exception:
                pass

        try:

            self._path.next()

        except Exception as exc:

            self._fail(
                f"MissionPath.next() failed: {exc!r}"
            )

            return

        try:

            finished = self._path.finished()

        except Exception as exc:

            self._fail(
                f"MissionPath.finished() failed: {exc!r}"
            )

            return

        if finished:

            self._finish()

            return

        with self._lock:

            self._statistics.current_waypoint_index += 1

            self._waypoint_retry_count = 0

            self._camera = _CameraState()

            self._state = (
                MappingState.NEXT_WAYPOINT
            )

    # ------------------------------------------------------------------------

    def _step_next_waypoint(self) -> None:
        """Send the next waypoint."""

        self._send_current_waypoint()

    # ========================================================================
    # Navigation
    # ========================================================================

    def _send_current_waypoint(self) -> None:
        """Send the current MissionPath waypoint to Navigator."""

        try:

            waypoint = self._path.current()

        except Exception as exc:

            self._fail(
                f"MissionPath.current() failed: {exc!r}"
            )

            return

        if waypoint is None:

            self._finish()

            return

        if not waypoint.enabled:

            self._skip_current_waypoint()

            return

        self._current_waypoint = waypoint

        try:
            waypoint.activate()
        except Exception:
            pass

        with self._lock:

            self._waypoint_start_time = (
                time.monotonic()
            )

            self._camera = _CameraState(
                waypoint_index=(
                    self._statistics.current_waypoint_index
                )
            )

            self._state = (
                MappingState.GOING_TO_WAYPOINT
            )

        try:

            self._navigator.goto(
                waypoint
            )

        except Exception as exc:

            self._handle_navigation_failure(
                f"Navigator.goto() failed: {exc!r}"
            )

            return

        self._log(
            "Mapping waypoint sent: "
            f"{waypoint.name or waypoint.id}"
        )

        self._request_camera_if_needed(
            waypoint
        )

    # ------------------------------------------------------------------------

    def _skip_current_waypoint(self) -> None:
        """Skip disabled waypoint."""

        try:

            self._path.next()

        except Exception as exc:

            self._fail(
                f"Failed to skip waypoint: {exc!r}"
            )

            return

        with self._lock:

            self._statistics.current_waypoint_index += 1

        self._send_current_waypoint()

    # ========================================================================
    # Camera
    # ========================================================================

    def _request_camera_if_needed(
        self,
        waypoint: Waypoint,
    ) -> None:
        """
        Publish camera capture request.

        The camera subsystem performs the actual capture.
        """

        if not self._photo_required:

            return

        request = getattr(
            waypoint,
            "camera_request",
            None,
        )

        # MappingGrid should mark mapping waypoints with
        # camera_trigger metadata. If no explicit request exists,
        # MappingMission creates the request itself.

        requested = False

        if request is not None:

            requested = bool(
                getattr(
                    request,
                    "take_photo",
                    False,
                )
            )

        if not requested:

            requested = bool(
                waypoint.waypoint_type.name
                == "MAPPING"
            )

        if not requested:

            return

        with self._lock:

            self._camera.requested = True

            self._camera.request_time = (
                time.monotonic()
            )

            self._statistics.photos_requested += 1

            self._state = (
                MappingState.PHOTO_REQUESTED
            )

        self._publish(
            EventType.CAMERA_CAPTURE_REQUEST,
            {
                "mission_name": self.name,
                "waypoint_id": waypoint.id,
                "waypoint_index": (
                    self._statistics.current_waypoint_index
                ),
                "latitude": waypoint.latitude,
                "longitude": waypoint.longitude,
                "altitude": waypoint.altitude,
            },
        )

        self._log(
            "Camera capture requested"
        )

    # ========================================================================
    # Navigation failure / retry
    # ========================================================================

    def _handle_navigation_failure(
        self,
        reason: str,
    ) -> None:
        """Retry navigation or fail the mission."""

        with self._lock:

            self._statistics.failed_waypoints += 1

            self._waypoint_retry_count += 1

            self._statistics.waypoint_retries += 1

            retry_count = (
                self._waypoint_retry_count
            )

        try:

            self._navigator.cancel()

        except Exception:

            pass

        if retry_count <= self._max_waypoint_retries:

            self._log(
                f"Waypoint retry "
                f"{retry_count}/"
                f"{self._max_waypoint_retries}: "
                f"{reason}"
            )

            with self._lock:

                self._state = (
                    MappingState.NEXT_WAYPOINT
                )

            return

        self._fail(
            f"Navigation failed after "
            f"{self._max_waypoint_retries} retries: "
            f"{reason}"
        )

    # ========================================================================
    # Photo failure
    # ========================================================================

    def _handle_photo_failure(
        self,
        reason: str,
    ) -> None:

        with self._lock:

            self._camera.failed = True

            self._statistics.photo_failures += 1

        self._log(
            f"Photo failure: {reason}"
        )

        # Camera failure should not silently advance.
        # Mapping quality would be compromised.
        self._fail(
            f"Mapping photo failure: {reason}"
        )

    # ========================================================================
    # Finish
    # ========================================================================

    def _finish(self) -> None:
        """Complete mapping mission."""

        with self._lock:

            if self._state in (
                MappingState.FINISHED,
                MappingState.ABORTED,
                MappingState.FAILED,
            ):
                return

            self._state = MappingState.FINISHED

            self._statistics.mission_finish_time = (
                time.monotonic()
            )

        try:

            if self._navigator is not None:

                self._navigator.cancel()

        except Exception:

            self._log_exception(
                "Navigator cancel failed during finish"
            )

        self._log(
            "Mapping mission completed"
        )

        self._publish(
            EventType.MISSION_FINISHED,
            {
                "mission_name": self.name,
                "mission_type": "mapping",
                "waypoints": (
                    self._statistics.total_waypoints
                ),
                "visited_waypoints": (
                    self._statistics.visited_waypoints
                ),
                "photos_captured": (
                    self._statistics.photos_captured
                ),
            },
        )

    # ========================================================================
    # Failure
    # ========================================================================

    def _fail(
        self,
        reason: str,
    ) -> None:
        """Fail mapping mission safely."""

        with self._lock:

            if self._state in (
                MappingState.FINISHED,
                MappingState.ABORTED,
            ):
                return

            self._state = MappingState.FAILED

            self._statistics.mission_finish_time = (
                time.monotonic()
            )

        try:

            if self._navigator is not None:

                self._navigator.cancel()

        except Exception:

            pass

        self._logger.error(
            "[MappingMission] FAILED: %s",
            reason,
        )

        self._publish(
            EventType.MISSION_FAILED,
            {
                "mission_name": self.name,
                "mission_type": "mapping",
                "reason": reason,
            },
        )

    # ========================================================================
    # Grid / MissionPath
    # ========================================================================

    def _load_waypoints(
        self,
        waypoints: Sequence[Waypoint],
    ) -> None:
        """
        Load generated waypoints into MissionPath.

        MissionPath is the authority for waypoint ordering.
        """

        if not waypoints:

            raise ValueError(
                "Cannot load an empty mapping path"
            )

        # MissionPath implementations may expose different
        # loading APIs. Prefer replace/set/load in that order.

        if hasattr(self._path, "replace"):

            self._path.replace(
                list(waypoints)
            )

            return

        if hasattr(self._path, "set_waypoints"):

            self._path.set_waypoints(
                list(waypoints)
            )

            return

        if hasattr(self._path, "load"):

            self._path.load(
                list(waypoints)
            )

            return

        if hasattr(self._path, "clear") and hasattr(
            self._path,
            "append",
        ):

            self._path.clear()

            for waypoint in waypoints:

                self._path.append(
                    waypoint
                )

            return

        raise RuntimeError(
            "MissionPath does not expose a supported "
            "waypoint loading API"
        )

    # ========================================================================
    # Validation
    # ========================================================================

    def _validate_polygon(self) -> None:
        """Validate mapping polygon."""

        if len(self._mapping_polygon) < 3:

            raise ValueError(
                "Mapping polygon requires at least 3 points"
            )

        for index, point in enumerate(
            self._mapping_polygon
        ):

            if len(point) != 2:

                raise ValueError(
                    f"Invalid polygon point at index {index}"
                )

            lat, lon = point

            if not (
                isinstance(lat, (int, float))
                and isinstance(lon, (int, float))
            ):

                raise ValueError(
                    f"Invalid coordinate at index {index}"
                )

            if not (
                math.isfinite(lat)
                and math.isfinite(lon)
            ):

                raise ValueError(
                    f"NaN/Infinity coordinate at index {index}"
                )

            if not -90.0 <= lat <= 90.0:

                raise ValueError(
                    f"Latitude out of range: {lat}"
                )

            if not -180.0 <= lon <= 180.0:

                raise ValueError(
                    f"Longitude out of range: {lon}"
                )

    # ========================================================================
    # Events
    # ========================================================================

    def _subscribe_to_events(self) -> None:
        """Subscribe to events relevant to mapping."""

        handlers = [
            (
                EventType.MISSION_START,
                self._on_mission_start,
            ),
            (
                EventType.MISSION_PAUSE,
                self._on_mission_pause,
            ),
            (
                EventType.MISSION_RESUME,
                self._on_mission_resume,
            ),
            (
                EventType.MISSION_ABORT,
                self._on_mission_abort,
            ),
            (
                EventType.GOTO_FINISHED,
                self._on_goto_finished,
            ),
            (
                EventType.GOTO_FAILED,
                self._on_goto_failed,
            ),
            (
                EventType.CAMERA_CAPTURED,
                self._on_camera_captured,
            ),
            (
                EventType.CAMERA_CAPTURE_FAILED,
                self._on_camera_capture_failed,
            ),
        ]

        for event_type, handler in handlers:

            try:

                self._event_bus.subscribe(
                    event_type,
                    handler,
                )

                self._subscriptions.append(
                    (event_type, handler)
                )

            except Exception:

                self._log_exception(
                    f"Failed subscribing to {event_type}"
                )

    # ------------------------------------------------------------------------

    def _unsubscribe_events(self) -> None:
        """Remove all event subscriptions."""

        for event_type, handler in (
            self._subscriptions
        ):

            try:

                self._event_bus.unsubscribe(
                    event_type,
                    handler,
                )

            except Exception:

                pass

        self._subscriptions.clear()

    # ========================================================================
    # Event callbacks
    # ========================================================================

    def _on_mission_start(
        self,
        event: Any,
    ) -> None:
        """Request mission start."""

        if not self._event_belongs_to_mission(
            event
        ):
            return

        with self._lock:

            if self._state == MappingState.READY:

                self._statistics.mission_start_time = (
                    time.monotonic()
                )

                self._state = (
                    MappingState.GENERATING_GRID
                )

    # ------------------------------------------------------------------------

    def _on_mission_pause(
        self,
        event: Any,
    ) -> None:
        """Request pause without executing it on EventBus thread."""

        if not self._event_belongs_to_mission(
            event
        ):
            return

        # Event callbacks only mark request.
        self._publish_internal_request(
            "pause"
        )

    # ------------------------------------------------------------------------

    def _on_mission_resume(
        self,
        event: Any,
    ) -> None:
        """Request resume."""

        if not self._event_belongs_to_mission(
            event
        ):
            return

        self._publish_internal_request(
            "resume"
        )

    # ------------------------------------------------------------------------

    def _on_mission_abort(
        self,
        event: Any,
    ) -> None:
        """Request abort."""

        if not self._event_belongs_to_mission(
            event
        ):
            return

        self._publish_internal_request(
            "abort"
        )

    # ------------------------------------------------------------------------

    def _on_goto_finished(
        self,
        event: Any,
    ) -> None:
        """Navigator reports waypoint completion."""

        if not self._event_belongs_to_mission(
            event
        ):
            return

        with self._lock:

            if self._state not in (
                MappingState.WAITING_NAVIGATOR,
                MappingState.GOING_TO_WAYPOINT,
            ):
                return

            self._state = (
                MappingState.WAYPOINT_REACHED
            )

    # ------------------------------------------------------------------------

    def _on_goto_failed(
        self,
        event: Any,
    ) -> None:
        """Navigator reports waypoint failure."""

        if not self._event_belongs_to_mission(
            event
        ):
            return

        self._handle_navigation_failure(
            "Navigator emitted GOTO_FAILED"
        )

    # ------------------------------------------------------------------------

    def _on_camera_captured(
        self,
        event: Any,
    ) -> None:
        """Camera confirms successful photo capture."""

        if not self._event_belongs_to_mission(
            event
        ):
            return

        with self._lock:

            self._camera.completed = True

            self._camera.completion_time = (
                time.monotonic()
            )

    # ------------------------------------------------------------------------

    def _on_camera_capture_failed(
        self,
        event: Any,
    ) -> None:
        """Camera reports failed photo capture."""

        if not self._event_belongs_to_mission(
            event
        ):
            return

        with self._lock:

            self._camera.failed = True

    # ========================================================================
    # Event helpers
    # ========================================================================

    def _event_belongs_to_mission(
        self,
        event: Any,
    ) -> bool:
        """
        Determine whether an event belongs to this mission.

        Supports both Event objects and dictionaries.
        """

        if event is None:

            return True

        payload = getattr(
            event,
            "payload",
            event,
        )

        if not isinstance(payload, dict):

            return True

        mission_name = payload.get(
            "mission_name"
        )

        if mission_name is None:

            return True

        return mission_name == self.name

    # ------------------------------------------------------------------------

    def _publish_internal_request(
        self,
        action: str,
    ) -> None:
        """
        Store a lifecycle request.

        The actual lifecycle operation is intentionally not performed
        directly from the EventBus callback.
        """

        with self._lock:

            if action == "pause":

                self._pending_pause = True

            elif action == "resume":

                self._pending_resume = True

            elif action == "abort":

                self._pending_abort = True

    # ========================================================================
    # Heartbeat
    # ========================================================================

    def _beat_heartbeat(self) -> None:
        """Update mission heartbeat."""

        now = time.monotonic()

        with self._lock:

            self._heartbeat_counter += 1

            self._statistics.last_heartbeat = now

            # Process deferred lifecycle requests.

            pending_abort = getattr(
                self,
                "_pending_abort",
                False,
            )

            pending_pause = getattr(
                self,
                "_pending_pause",
                False,
            )

            pending_resume = getattr(
                self,
                "_pending_resume",
                False,
            )

            self._pending_abort = False
            self._pending_pause = False
            self._pending_resume = False

        if pending_abort:

            self.abort()

            return

        if pending_pause:

            self.pause()

            return

        if pending_resume:

            self.resume()

    # ========================================================================
    # Statistics
    # ========================================================================

    def _update_remaining_statistics(self) -> None:
        """Update dynamic progress values."""

        try:

            remaining = (
                self._path.remaining_waypoints()
            )

        except Exception:

            remaining = 0

        with self._lock:

            self._statistics.remaining_waypoints = (
                remaining
            )

            self._statistics.remaining_distance_m = (
                self._safe_remaining_distance()
            )

    # ------------------------------------------------------------------------

    def _safe_remaining_distance(self) -> float:
        """Read Navigator remaining distance safely."""

        navigator = self._navigator

        if navigator is None:

            return 0.0

        try:

            value = navigator.remaining_distance()

            if value is None:

                return 0.0

            value = float(value)

            if not math.isfinite(value):

                return 0.0

            return max(
                value,
                0.0,
            )

        except Exception:

            return 0.0

    # ------------------------------------------------------------------------

    def statistics(self) -> MappingStatistics:
        """Return a snapshot of mission statistics."""

        with self._lock:

            snapshot = MappingStatistics(
                mission_id=self._statistics.mission_id,

                total_waypoints=(
                    self._statistics.total_waypoints
                ),

                current_waypoint_index=(
                    self._statistics.current_waypoint_index
                ),

                remaining_waypoints=(
                    self._statistics.remaining_waypoints
                ),

                visited_waypoints=(
                    self._statistics.visited_waypoints
                ),

                successful_waypoints=(
                    self._statistics.successful_waypoints
                ),

                failed_waypoints=(
                    self._statistics.failed_waypoints
                ),

                photos_requested=(
                    self._statistics.photos_requested
                ),

                photos_captured=(
                    self._statistics.photos_captured
                ),

                photo_failures=(
                    self._statistics.photo_failures
                ),

                waypoint_retries=(
                    self._statistics.waypoint_retries
                ),

                timeout_count=(
                    self._statistics.timeout_count
                ),

                pause_count=(
                    self._statistics.pause_count
                ),

                resume_count=(
                    self._statistics.resume_count
                ),

                mission_start_time=(
                    self._statistics.mission_start_time
                ),

                mission_finish_time=(
                    self._statistics.mission_finish_time
                ),

                last_waypoint_time=(
                    self._statistics.last_waypoint_time
                ),

                last_photo_time=(
                    self._statistics.last_photo_time
                ),

                total_distance_m=(
                    self._statistics.total_distance_m
                ),

                remaining_distance_m=(
                    self._statistics.remaining_distance_m
                ),

                last_heartbeat=(
                    self._statistics.last_heartbeat
                ),

                average_waypoint_time_s=(
                    self._statistics.average_waypoint_time_s
                ),
            )

            return snapshot

    # ========================================================================
    # Status
    # ========================================================================

    def status(self) -> str:
        """Return current mapping mission state."""

        with self._lock:

            return self._state.name

    # ------------------------------------------------------------------------

    def current_waypoint(
        self,
    ) -> Optional[Waypoint]:
        """Return the current waypoint."""

        with self._lock:

            return self._current_waypoint

    # ------------------------------------------------------------------------

    def grid_statistics(self) -> Optional[Any]:
        """Return MappingGrid statistics."""

        grid = self._grid

        if grid is None:

            return None

        try:

            return grid.statistics()

        except Exception:

            return None

    # ========================================================================
    # Event publishing
    # ========================================================================

    def _publish(
        self,
        event_type: EventType,
        payload: Optional[dict] = None,
    ) -> None:
        """Publish an EventBus event."""

        try:

            self._event_bus.publish(
                event_type,
                payload or {},
            )

        except Exception:

            self._log_exception(
                f"Failed publishing {event_type}"
            )

    # ========================================================================
    # Logging
    # ========================================================================

    def _log(
        self,
        message: str,
    ) -> None:
        """Write mapping mission log."""

        self._logger.info(
            "[MappingMission] %s",
            message,
        )

    # ------------------------------------------------------------------------

    def _log_exception(
        self,
        message: str,
    ) -> None:
        """Log an exception with traceback."""

        self._logger.exception(
            "[MappingMission] %s",
            message,
        )

    # ========================================================================
    # Representation
    # ========================================================================

    def __repr__(self) -> str:

        with self._lock:

            return (
                "MappingMission("
                f"state={self._state.name}, "
                f"waypoint="
                f"{self._statistics.current_waypoint_index}/"
                f"{self._statistics.total_waypoints}, "
                f"photos="
                f"{self._statistics.photos_captured}/"
                f"{self._statistics.photos_requested}"
                ")"
            )
