"""Coordinates target detections and detection state during a mission."""

from __future__ import annotations

import logging
import math
import threading
import time

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Sequence

import numpy as np

from perception.yolo import YOLODetector, Detection
from perception.detector import (
    DetectionProcessor,
    DetectionCandidate,
    DetectorConfig,
)
from navigation.tracker import (
    TargetTracker,
    TrackerConfig,
    Track,
    TrackState,
)
from navigation.target import (
    Target,
    TargetConfig,
    TargetSnapshot,
    TargetState,
)

from core.events import EventType


logger = logging.getLogger(__name__)


# ============================================================================
# Mission state
# ============================================================================

class ObjectDetectionState(str, Enum):
    """Object detection mission state."""

    IDLE = "idle"

    INITIALIZING = "initializing"

    SEARCHING = "searching"

    DETECTED = "detected"

    TRACKING = "tracking"

    TARGET_CONFIRMED = "target_confirmed"

    TARGET_LOST = "target_lost"

    PAYLOAD_REQUESTED = "payload_requested"

    PAYLOAD_RELEASED = "payload_released"

    PAUSED = "paused"

    COMPLETED = "completed"

    ABORTED = "aborted"

    FAILED = "failed"


# ============================================================================
# Configuration
# ============================================================================

@dataclass(frozen=True)
class ObjectDetectionConfig:
    """
    Object detection mission configuration.
    """

    target_class: Optional[str] = None

    target_class_id: Optional[int] = None

    minimum_confidence: float = 0.60

    require_confirmed_target: bool = True

    lost_target_timeout_s: float = 3.0

    minimum_target_hits: int = 3

    payload_enabled: bool = True

    payload_cooldown_s: float = 2.0

    require_stable_target: bool = True

    stable_center_threshold: float = 0.15

    stable_time_s: float = 0.5

    maximum_mission_time_s: float = 0.0

    frame_timeout_s: float = 2.0

    stop_after_payload: bool = True

    max_payload_attempts: int = 1

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate configuration."""

        if not 0.0 <= self.minimum_confidence <= 1.0:
            raise ValueError(
                "minimum_confidence must be between 0 and 1"
            )

        if self.target_class is None and (
            self.target_class_id is None
        ):
            raise ValueError(
                "Either target_class or target_class_id "
                "must be specified"
            )

        if self.target_class_id is not None:
            if self.target_class_id < 0:
                raise ValueError(
                    "target_class_id cannot be negative"
                )

        if self.lost_target_timeout_s < 0:
            raise ValueError(
                "lost_target_timeout_s cannot be negative"
            )

        if self.minimum_target_hits <= 0:
            raise ValueError(
                "minimum_target_hits must be positive"
            )

        if self.payload_cooldown_s < 0:
            raise ValueError(
                "payload_cooldown_s cannot be negative"
            )

        if not 0.0 <= self.stable_center_threshold <= 1.0:
            raise ValueError(
                "stable_center_threshold must be between 0 and 1"
            )

        if self.stable_time_s < 0:
            raise ValueError(
                "stable_time_s cannot be negative"
            )

        if self.maximum_mission_time_s < 0:
            raise ValueError(
                "maximum_mission_time_s cannot be negative"
            )

        if self.frame_timeout_s < 0:
            raise ValueError(
                "frame_timeout_s cannot be negative"
            )

        if self.max_payload_attempts <= 0:
            raise ValueError(
                "max_payload_attempts must be positive"
            )


# ============================================================================
# Mission statistics
# ============================================================================

@dataclass
class ObjectDetectionStatistics:
    """Runtime statistics."""

    frames_received: int = 0

    frames_processed: int = 0

    detections_received: int = 0

    candidates_found: int = 0

    targets_detected: int = 0

    targets_confirmed: int = 0

    target_lost_count: int = 0

    payload_requests: int = 0

    payload_releases: int = 0

    payload_failures: int = 0

    mission_start_time: Optional[float] = None

    mission_finish_time: Optional[float] = None

    last_frame_time: Optional[float] = None

    last_detection_time: Optional[float] = None

    last_target_time: Optional[float] = None

    last_payload_time: Optional[float] = None


# ============================================================================
# ObjectDetectionMission
# ============================================================================

class ObjectDetectionMission:

    name = "ObjectDetectionMission"

    def __init__(
        self,
        yolo_detector: YOLODetector,
        event_bus: Any,
        camera: Any = None,
        navigator: Any = None,
        config: Optional[ObjectDetectionConfig] = None,
        detector_config: Optional[DetectorConfig] = None,
        tracker_config: Optional[TrackerConfig] = None,
        target_config: Optional[TargetConfig] = None,
        shared_data: Any = None,
    ) -> None:

        if yolo_detector is None:
            raise ValueError(
                "yolo_detector cannot be None"
            )

        if event_bus is None:
            raise ValueError(
                "event_bus cannot be None"
            )

        self._yolo = yolo_detector

        self._event_bus = event_bus

        self._camera = camera

        self._navigator = navigator

        self._shared = shared_data

        self.config = (
            config
            or ObjectDetectionConfig(
                target_class="target"
            )
        )

        # ---------------------------------------------------------------
        # AI pipeline
        # ---------------------------------------------------------------

        self._processor = DetectionProcessor(
            detector=self._yolo,
            config=(
                detector_config
                or DetectorConfig(
                    confidence_threshold=(
                        self.config.minimum_confidence
                    )
                )
            ),
        )

        self._tracker = TargetTracker(
            config=(
                tracker_config
                or TrackerConfig(
                    min_confirmed_hits=(
                        self.config.minimum_target_hits
                    )
                )
            )
        )

        self._target = Target(
            config=(
                target_config
                or TargetConfig(
                    minimum_confidence=(
                        self.config.minimum_confidence
                    ),
                    minimum_confirmed_hits=(
                        self.config.minimum_target_hits
                    ),
                    maximum_lost_time_s=(
                        self.config.lost_target_timeout_s
                    ),
                    require_confirmed_track=(
                        self.config.require_confirmed_target
                    ),
                )
            )
        )

        # ---------------------------------------------------------------
        # Runtime
        # ---------------------------------------------------------------

        self._lock = threading.RLock()

        self._state = (
            ObjectDetectionState.IDLE
        )

        self._running = False

        self._paused = False

        self._start_time: Optional[float] = None

        self._last_frame_time: Optional[float] = None

        self._target_stable_since: Optional[float] = None

        self._last_payload_time: Optional[float] = None

        self._payload_attempts = 0

        self._pending_payload = False

        self._latest_frame: Optional[np.ndarray] = None

        self._latest_snapshot: Optional[
            TargetSnapshot
        ] = None

        self._statistics = (
            ObjectDetectionStatistics()
        )

        self._initialized = False

        self._finish_requested = False

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def initialize(self) -> None:
        """Initialize YOLO and reset mission state."""

        with self._lock:

            if self._initialized:
                return

            self._state = (
                ObjectDetectionState.INITIALIZING
            )

        try:

            self._yolo.load()

            with self._lock:

                self._tracker.reset()

                self._target.reset()

                self._statistics = (
                    ObjectDetectionStatistics()
                )

                self._initialized = True

                self._state = (
                    ObjectDetectionState.SEARCHING
                )

            logger.info(
                "[ObjectDetectionMission] Initialized"
            )

        except Exception as exc:

            self._fail(
                f"Initialization failed: {exc}"
            )

            raise

    # -------------------------------------------------------------------------

    def start(self) -> None:
        """Start object detection mission."""

        if not self._initialized:
            self.initialize()

        with self._lock:

            if self._running:
                return

            self._running = True

            self._paused = False

            self._finish_requested = False

            self._start_time = (
                time.monotonic()
            )

            self._statistics.mission_start_time = (
                self._start_time
            )

            self._state = (
                ObjectDetectionState.SEARCHING
            )

        self._publish(
            EventType.MISSION_START,
            {
                "mission_name": self.name,
                "mission_type": "object_detection",
            },
        )

        logger.info(
            "[ObjectDetectionMission] Started"
        )

    # -------------------------------------------------------------------------

    def update(self) -> None:
        """Update the current state from the latest input."""

        with self._lock:

            if not self._running:
                return

            if self._paused:
                return

            if self._finish_requested:
                return

        self._check_mission_timeout()

        if self._finish_requested:
            return

        frame = self._get_frame()

        if frame is None:

            self._handle_no_frame()

            return

        self._process_frame(
            frame
        )

    # =========================================================================
    # Frame handling
    # =========================================================================

    def submit_frame(
        self,
        frame: np.ndarray,
    ) -> None:

        if frame is None:
            return

        if not isinstance(
            frame,
            np.ndarray,
        ):
            raise TypeError(
                "frame must be numpy.ndarray"
            )

        if frame.size == 0:
            return

        with self._lock:

            self._latest_frame = frame.copy()

            self._last_frame_time = (
                time.monotonic()
            )

            self._statistics.frames_received += 1

            self._statistics.last_frame_time = (
                self._last_frame_time
            )

    # -------------------------------------------------------------------------

    def _get_frame(self) -> Optional[np.ndarray]:
        """Read a frame from camera or submitted-frame buffer."""

        if self._camera is not None:

            try:

                frame = self._camera.latest_frame()

                if frame is not None:

                    return frame

            except Exception:

                logger.exception(
                    "[ObjectDetectionMission] "
                    "Camera latest_frame failed"
                )

        with self._lock:

            if self._latest_frame is None:
                return None

            return self._latest_frame.copy()

    # =========================================================================
    # Frame processing
    # =========================================================================

    def _process_frame(
        self,
        frame: np.ndarray,
    ) -> None:
        """Run complete AI pipeline for one frame."""

        try:

            detections = (
                self._yolo.detect(
                    frame
                )
            )

            with self._lock:

                self._statistics.frames_processed += 1

                self._statistics.detections_received += (
                    len(detections)
                )

                if detections:

                    self._statistics.last_detection_time = (
                        time.monotonic()
                    )

            candidates = (
                self._processor.process(
                    detections
                )
            )

            target_candidates = (
                self._filter_target_candidates(
                    candidates
                )
            )

            with self._lock:

                self._statistics.candidates_found += (
                    len(target_candidates)
                )

            tracks = (
                self._tracker.update(
                    target_candidates
                )
            )

            target_track = (
                self._select_target_track(
                    tracks
                )
            )

            self._update_target(
                target_track
            )

            self._handle_target_state()

        except Exception as exc:

            logger.exception(
                "[ObjectDetectionMission] "
                "Frame processing failed"
            )

            self._fail(
                f"Frame processing failed: {exc}"
            )

    # =========================================================================
    # Candidate filtering
    # =========================================================================

    def _filter_target_candidates(
        self,
        candidates: Sequence[
            DetectionCandidate
        ],
    ) -> list[DetectionCandidate]:
        """Keep only the configured target class."""

        result = []

        for candidate in candidates:

            if (
                self.config.target_class is not None
                and candidate.class_name
                != self.config.target_class
            ):
                continue

            if (
                self.config.target_class_id is not None
                and candidate.class_id
                != self.config.target_class_id
            ):
                continue

            if (
                candidate.confidence
                < self.config.minimum_confidence
            ):
                continue

            result.append(
                candidate
            )

        return result

    # =========================================================================
    # Track selection
    # =========================================================================

    def _select_target_track(
        self,
        tracks: Sequence[Track],
    ) -> Optional[Track]:
        """Select the most suitable target track."""

        valid_tracks = []

        for track in tracks:

            if track.state == TrackState.DELETED:
                continue

            if (
                self.config.target_class
                is not None
                and track.class_name
                != self.config.target_class
            ):
                continue

            if (
                self.config.target_class_id
                is not None
                and track.class_id
                != self.config.target_class_id
            ):
                continue

            if (
                track.confidence
                < self.config.minimum_confidence
            ):
                continue

            if (
                self.config.require_confirmed_target
                and track.state
                != TrackState.CONFIRMED
            ):
                continue

            valid_tracks.append(
                track
            )

        if not valid_tracks:
            return None

        return max(
            valid_tracks,
            key=lambda track: (
                track.confidence,
                track.hits,
                -track.missed_frames,
            ),
        )

    # =========================================================================
    # Target update
    # =========================================================================

    def _update_target(
        self,
        track: Optional[Track],
    ) -> None:
        """Synchronize Target with tracker."""

        if track is None:

            self._target.update_state(
                None
            )

            with self._lock:

                if (
                    self._state
                    not in (
                        ObjectDetectionState.SEARCHING,
                        ObjectDetectionState.TARGET_LOST,
                    )
                ):

                    self._state = (
                        ObjectDetectionState.TARGET_LOST
                    )

                    self._statistics.target_lost_count += 1

            return

        self._target.update_from_track(
            track
        )

        snapshot = (
            self._target.snapshot()
        )

        with self._lock:

            self._latest_snapshot = (
                snapshot
            )

            self._statistics.targets_detected += 1

            self._statistics.last_target_time = (
                time.monotonic()
            )

            if snapshot.state == (
                TargetState.CONFIRMED
            ):

                self._statistics.targets_confirmed += 1

                self._state = (
                    ObjectDetectionState.TARGET_CONFIRMED
                )

            elif snapshot.state in (
                TargetState.TRACKING,
                TargetState.DETECTED,
            ):

                self._state = (
                    ObjectDetectionState.TRACKING
                )

    # =========================================================================
    # Target state handling
    # =========================================================================

    def _handle_target_state(self) -> None:

        snapshot = (
            self._target.snapshot()
        )

        if not snapshot.valid:

            if snapshot.state == (
                TargetState.LOST
            ):

                with self._lock:

                    self._state = (
                        ObjectDetectionState.TARGET_LOST
                    )

            return

        self._publish(
            EventType.TARGET_DETECTED,
            {
                "mission_name": self.name,
                "target_id": snapshot.target_id,
                "class_id": snapshot.class_id,
                "class_name": snapshot.class_name,
                "confidence": snapshot.confidence,
                "x": snapshot.x,
                "y": snapshot.y,
                "normalized_x": snapshot.normalized_x,
                "normalized_y": snapshot.normalized_y,
            },
        )

        if not self.config.payload_enabled:
            return

        if self._payload_attempts >= (
            self.config.max_payload_attempts
        ):
            return

        if self.config.require_stable_target:

            if not self._is_target_stable(
                snapshot
            ):
                return

        else:

            if snapshot.state != (
                TargetState.CONFIRMED
            ):
                return

        self._request_payload(
            snapshot
        )

    # =========================================================================
    # Target stability
    # =========================================================================

    def _is_target_stable(
        self,
        snapshot: TargetSnapshot,
    ) -> bool:
        """
        Determine whether target has been sufficiently stable
        before requesting payload release.
        """

        if not snapshot.valid:
            self._target_stable_since = None
            return False

        distance = (
            self._target.center_distance()
        )

        if (
            distance
            > self.config.stable_center_threshold
        ):

            self._target_stable_since = None

            return False

        now = time.monotonic()

        if self._target_stable_since is None:

            self._target_stable_since = now

            return False

        return (
            now
            - self._target_stable_since
            >= self.config.stable_time_s
        )

    # =========================================================================
    # Payload
    # =========================================================================

    def _request_payload(
        self,
        snapshot: TargetSnapshot,
    ) -> None:
        """Publish payload release request."""

        now = time.monotonic()

        if self._last_payload_time is not None:

            if (
                now
                - self._last_payload_time
                < self.config.payload_cooldown_s
            ):
                return

        with self._lock:

            if self._payload_attempts >= (
                self.config.max_payload_attempts
            ):
                return

            self._payload_attempts += 1

            self._pending_payload = True

            self._last_payload_time = now

            self._statistics.payload_requests += 1

            self._statistics.last_payload_time = now

            self._state = (
                ObjectDetectionState.PAYLOAD_REQUESTED
            )

        self._publish(
            EventType.PAYLOAD_RELEASE_REQUEST,
            {
                "mission_name": self.name,

                "target_id": snapshot.target_id,

                "class_id": snapshot.class_id,

                "class_name": snapshot.class_name,

                "confidence": snapshot.confidence,

                "x": snapshot.x,

                "y": snapshot.y,

                "normalized_x": (
                    snapshot.normalized_x
                ),

                "normalized_y": (
                    snapshot.normalized_y
                ),
            },
        )

        logger.info(
            "[ObjectDetectionMission] "
            "Payload release requested for target %s",
            snapshot.target_id,
        )

    # =========================================================================
    # Payload result
    # =========================================================================

    def payload_released(
        self,
    ) -> None:
        """
        Called by payload subsystem after successful release.
        """

        with self._lock:

            self._pending_payload = False

            self._statistics.payload_releases += 1

            self._state = (
                ObjectDetectionState.PAYLOAD_RELEASED
            )

        self._publish(
            EventType.PAYLOAD_RELEASED,
            {
                "mission_name": self.name,

                "target_id": (
                    self._target.target_id
                ),
            },
        )

        if self.config.stop_after_payload:

            self.complete()

    # -------------------------------------------------------------------------

    def payload_failed(
        self,
        reason: str = "unknown",
    ) -> None:
        """Called when payload release fails."""

        with self._lock:

            self._pending_payload = False

            self._statistics.payload_failures += 1

            self._state = (
                ObjectDetectionState.TARGET_CONFIRMED
            )

        self._publish(
            EventType.PAYLOAD_RELEASE_FAILED,
            {
                "mission_name": self.name,
                "reason": reason,
                "target_id": (
                    self._target.target_id
                ),
            },
        )

    # =========================================================================
    # Timeout
    # =========================================================================

    def _check_mission_timeout(self) -> None:
        """Check total mission timeout."""

        if (
            self.config.maximum_mission_time_s
            <= 0
        ):
            return

        if self._start_time is None:
            return

        elapsed = (
            time.monotonic()
            - self._start_time
        )

        if (
            elapsed
            >= self.config.maximum_mission_time_s
        ):

            self._fail(
                "Object detection mission timeout"
            )

    # =========================================================================
    # No frame
    # =========================================================================

    def _handle_no_frame(self) -> None:

        if self._last_frame_time is None:
            return

        elapsed = (
            time.monotonic()
            - self._last_frame_time
        )

        if (
            elapsed
            > self.config.frame_timeout_s
        ):

            self._fail(
                "Camera frame timeout"
            )

    # =========================================================================
    # Pause
    # =========================================================================

    def pause(self) -> None:
        """Pause object detection processing."""

        with self._lock:

            if not self._running:
                return

            self._paused = True

            self._state = (
                ObjectDetectionState.PAUSED
            )

        self._publish(
            EventType.MISSION_PAUSE,
            {
                "mission_name": self.name,
            },
        )

    # =========================================================================
    # Resume
    # =========================================================================

    def resume(self) -> None:
        """Resume object detection."""

        with self._lock:

            if not self._running:
                return

            self._paused = False

            if self._target.is_valid():

                self._state = (
                    ObjectDetectionState.TRACKING
                )

            else:

                self._state = (
                    ObjectDetectionState.SEARCHING
                )

        self._publish(
            EventType.MISSION_RESUME,
            {
                "mission_name": self.name,
            },
        )

    # =========================================================================
    # Abort
    # =========================================================================

    def abort(
        self,
        reason: str = "operator_abort",
    ) -> None:
        """Abort mission."""

        with self._lock:

            if not self._running:
                return

            self._running = False

            self._paused = False

            self._state = (
                ObjectDetectionState.ABORTED
            )

            self._statistics.mission_finish_time = (
                time.monotonic()
            )

        self._publish(
            EventType.MISSION_ABORT,
            {
                "mission_name": self.name,
                "reason": reason,
            },
        )

        logger.warning(
            "[ObjectDetectionMission] "
            "Aborted: %s",
            reason,
        )

    # =========================================================================
    # Complete
    # =========================================================================

    def complete(self) -> None:
        """Complete mission successfully."""

        with self._lock:

            if self._state in (
                ObjectDetectionState.COMPLETED,
                ObjectDetectionState.ABORTED,
                ObjectDetectionState.FAILED,
            ):
                return

            self._running = False

            self._paused = False

            self._finish_requested = True

            self._state = (
                ObjectDetectionState.COMPLETED
            )

            self._statistics.mission_finish_time = (
                time.monotonic()
            )

        self._publish(
            EventType.MISSION_FINISHED,
            {
                "mission_name": self.name,
                "mission_type": "object_detection",
                "target_id": (
                    self._target.target_id
                ),
                "payload_released": (
                    self._statistics.payload_releases
                    > 0
                ),
            },
        )

        logger.info(
            "[ObjectDetectionMission] Completed"
        )

    # =========================================================================
    # Failure
    # =========================================================================

    def _fail(
        self,
        reason: str,
    ) -> None:
        """Fail mission."""

        with self._lock:

            if self._state in (
                ObjectDetectionState.COMPLETED,
                ObjectDetectionState.ABORTED,
                ObjectDetectionState.FAILED,
            ):
                return

            self._running = False

            self._finish_requested = True

            self._state = (
                ObjectDetectionState.FAILED
            )

            self._statistics.mission_finish_time = (
                time.monotonic()
            )

        self._publish(
            EventType.MISSION_FAILED,
            {
                "mission_name": self.name,
                "mission_type": "object_detection",
                "reason": reason,
            },
        )

        logger.error(
            "[ObjectDetectionMission] FAILED: %s",
            reason,
        )

    # =========================================================================
    # Cleanup
    # =========================================================================

    def cleanup(self) -> None:
        """Release mission runtime resources."""

        with self._lock:

            self._running = False

            self._paused = False

            self._latest_frame = None

            self._latest_snapshot = None

            self._target_stable_since = None

            self._pending_payload = False

        logger.info(
            "[ObjectDetectionMission] Cleanup complete"
        )

    # =========================================================================
    # Event helper
    # =========================================================================

    def _publish(
        self,
        event_type: EventType,
        payload: dict,
    ) -> None:
        """Publish an EventBus event."""

        try:

            self._event_bus.publish(
                event_type,
                payload,
            )

        except Exception:

            logger.exception(
                "[ObjectDetectionMission] "
                "Failed to publish event %s",
                event_type,
            )

    # =========================================================================
    # Status
    # =========================================================================

    @property
    def state(
        self,
    ) -> ObjectDetectionState:

        with self._lock:
            return self._state

    # -------------------------------------------------------------------------

    @property
    def running(
        self,
    ) -> bool:

        with self._lock:
            return self._running

    # -------------------------------------------------------------------------

    @property
    def paused(
        self,
    ) -> bool:

        with self._lock:
            return self._paused

    # -------------------------------------------------------------------------

    def current_target(
        self,
    ) -> Optional[TargetSnapshot]:

        with self._lock:

            if self._latest_snapshot is None:
                return None

            return self._latest_snapshot

    # -------------------------------------------------------------------------

    def statistics(
        self,
    ) -> ObjectDetectionStatistics:
        """Return statistics snapshot."""

        with self._lock:

            return ObjectDetectionStatistics(
                frames_received=(
                    self._statistics.frames_received
                ),

                frames_processed=(
                    self._statistics.frames_processed
                ),

                detections_received=(
                    self._statistics.detections_received
                ),

                candidates_found=(
                    self._statistics.candidates_found
                ),

                targets_detected=(
                    self._statistics.targets_detected
                ),

                targets_confirmed=(
                    self._statistics.targets_confirmed
                ),

                target_lost_count=(
                    self._statistics.target_lost_count
                ),

                payload_requests=(
                    self._statistics.payload_requests
                ),

                payload_releases=(
                    self._statistics.payload_releases
                ),

                payload_failures=(
                    self._statistics.payload_failures
                ),

                mission_start_time=(
                    self._statistics.mission_start_time
                ),

                mission_finish_time=(
                    self._statistics.mission_finish_time
                ),

                last_frame_time=(
                    self._statistics.last_frame_time
                ),

                last_detection_time=(
                    self._statistics.last_detection_time
                ),

                last_target_time=(
                    self._statistics.last_target_time
                ),

                last_payload_time=(
                    self._statistics.last_payload_time
                ),
            )

    # =========================================================================
    # AI accessors
    # =========================================================================

    @property
    def tracker(self) -> TargetTracker:
        """Return tracker instance."""

        return self._tracker

    # -------------------------------------------------------------------------

    @property
    def detector(self) -> DetectionProcessor:
        """Return detection processor."""

        return self._processor

    # -------------------------------------------------------------------------

    @property
    def target(self) -> Target:
        """Return current Target object."""

        return self._target

    # =========================================================================
    # Representation
    # =========================================================================

    def __repr__(self) -> str:

        snapshot = self.current_target()

        target_text = (
            "none"
            if snapshot is None
            else (
                f"{snapshot.class_name}:"
                f"{snapshot.confidence:.2f}"
            )
        )

        return (
            "ObjectDetectionMission("
            f"state={self.state.value!r}, "
            f"running={self.running}, "
            f"target={target_text}"
            ")"
        )
