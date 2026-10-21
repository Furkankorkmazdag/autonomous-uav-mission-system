"""Mission-level representation of a detected and tracked target."""

from __future__ import annotations

import math
import threading
import time

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

from navigation.tracker import Track, TrackState


# ============================================================================
# Exceptions
# ============================================================================

class TargetError(Exception):
    """Base target exception."""


class TargetConfigurationError(TargetError):
    """Invalid target configuration."""


# ============================================================================
# Target state
# ============================================================================

class TargetState(str, Enum):
    """
    High-level state of the target.
    """

    UNKNOWN = "unknown"

    DETECTED = "detected"

    TRACKING = "tracking"

    CONFIRMED = "confirmed"

    LOST = "lost"

    INVALID = "invalid"


# ============================================================================
# Target configuration
# ============================================================================

@dataclass(frozen=True)
class TargetConfig:
    """
    Target state configuration.
    """

    minimum_confidence: float = 0.50

    maximum_lost_time_s: float = 2.0

    minimum_tracking_time_s: float = 0.20

    minimum_confirmed_hits: int = 3

    history_size: int = 30

    maximum_position_jump_normalized: float = 0.50

    require_confirmed_track: bool = True

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate configuration."""

        if not (
            0.0
            <= self.minimum_confidence
            <= 1.0
        ):
            raise TargetConfigurationError(
                "minimum_confidence must be between 0 and 1"
            )

        if self.maximum_lost_time_s < 0:
            raise TargetConfigurationError(
                "maximum_lost_time_s cannot be negative"
            )

        if self.minimum_tracking_time_s < 0:
            raise TargetConfigurationError(
                "minimum_tracking_time_s cannot be negative"
            )

        if self.minimum_confirmed_hits <= 0:
            raise TargetConfigurationError(
                "minimum_confirmed_hits must be positive"
            )

        if self.history_size <= 0:
            raise TargetConfigurationError(
                "history_size must be positive"
            )

        if not (
            0.0
            < self.maximum_position_jump_normalized
            <= 2.0
        ):
            raise TargetConfigurationError(
                "maximum_position_jump_normalized "
                "must be in (0, 2]"
            )


# ============================================================================
# Target observation
# ============================================================================

@dataclass(frozen=True)
class TargetObservation:
    """
    One historical observation of the target.
    """

    timestamp: float

    x: float

    y: float

    width: float

    height: float

    confidence: float

    track_id: int

    frame_width: int

    frame_height: int

    @property
    def normalized_x(self) -> float:
        """Return normalized X coordinate."""

        if self.frame_width <= 0:
            return 0.0

        return self.x / self.frame_width

    @property
    def normalized_y(self) -> float:
        """Return normalized Y coordinate."""

        if self.frame_height <= 0:
            return 0.0

        return self.y / self.frame_height

    @property
    def center(self) -> Tuple[float, float]:
        """Return pixel center."""

        return self.x, self.y


# ============================================================================
# Target snapshot
# ============================================================================

@dataclass(frozen=True)
class TargetSnapshot:

    target_id: int

    class_id: int

    class_name: str

    state: TargetState

    confidence: float

    x: float

    y: float

    width: float

    height: float

    frame_width: int

    frame_height: int

    normalized_x: float

    normalized_y: float

    age_s: float

    lost_time_s: float

    hit_count: int

    missed_frames: int

    valid: bool

    timestamp: float


# ============================================================================
# Target
# ============================================================================

class Target:

    def __init__(
        self,
        config: Optional[TargetConfig] = None,
    ) -> None:

        self.config = (
            config
            or TargetConfig()
        )

        self._lock = threading.RLock()

        self._target_id: Optional[int] = None

        self._class_id: Optional[int] = None

        self._class_name: Optional[str] = None

        self._state = TargetState.UNKNOWN

        self._confidence = 0.0

        self._x = 0.0
        self._y = 0.0

        self._width = 0.0
        self._height = 0.0

        self._frame_width = 0
        self._frame_height = 0

        self._first_seen: Optional[float] = None

        self._last_seen: Optional[float] = None

        self._last_update: Optional[float] = None

        self._hit_count = 0

        self._missed_frames = 0

        self._history: List[
            TargetObservation
        ] = []

    # =========================================================================
    # Update from tracker
    # =========================================================================

    def update_from_track(
        self,
        track: Track,
    ) -> None:

        if track is None:

            raise ValueError(
                "track cannot be None"
            )

        now = time.monotonic()

        with self._lock:

            # ---------------------------------------------------------------
            # Basic identity
            # ---------------------------------------------------------------

            self._target_id = (
                track.track_id
            )

            self._class_id = (
                track.class_id
            )

            self._class_name = (
                track.class_name
            )

            # ---------------------------------------------------------------
            # Position
            # ---------------------------------------------------------------

            self._x = (
                float(track.x)
            )

            self._y = (
                float(track.y)
            )

            self._width = (
                float(track.width)
            )

            self._height = (
                float(track.height)
            )

            self._frame_width = (
                int(track.frame_width)
            )

            self._frame_height = (
                int(track.frame_height)
            )

            # ---------------------------------------------------------------
            # Confidence
            # ---------------------------------------------------------------

            self._confidence = max(
                0.0,
                min(
                    1.0,
                    float(
                        track.confidence
                    ),
                ),
            )

            # ---------------------------------------------------------------
            # Time
            # ---------------------------------------------------------------

            if self._first_seen is None:

                self._first_seen = now

            self._last_seen = (
                now
            )

            self._last_update = (
                now
            )

            # ---------------------------------------------------------------
            # Tracking statistics
            # ---------------------------------------------------------------

            self._hit_count = (
                track.hits
            )

            self._missed_frames = (
                track.missed_frames
            )

            # ---------------------------------------------------------------
            # State
            # ---------------------------------------------------------------

            self._state = (
                self._state_from_track(
                    track
                )
            )

            # ---------------------------------------------------------------
            # History
            # ---------------------------------------------------------------

            self._append_history(
                timestamp=now
            )

    # =========================================================================
    # State update
    # =========================================================================

    def update_state(
        self,
        track: Optional[Track],
    ) -> None:
        """
        Update state when a new Track is available.

        track=None:
            Target observation is temporarily lost.
        """

        now = time.monotonic()

        with self._lock:

            if track is not None:

                self.update_from_track(
                    track
                )

                return

            # No track.
            if self._last_seen is None:

                self._state = (
                    TargetState.UNKNOWN
                )

                return

            lost_time = (
                now
                - self._last_seen
            )

            self._missed_frames += 1

            if (
                lost_time
                > self.config.maximum_lost_time_s
            ):

                self._state = (
                    TargetState.LOST
                )

            else:

                self._state = (
                    TargetState.LOST
                )

            self._last_update = now

    # =========================================================================
    # State mapping
    # =========================================================================

    def _state_from_track(
        self,
        track: Track,
    ) -> TargetState:
        """
        Convert Tracker TrackState to TargetState.
        """

        if track.state == (
            TrackState.TENTATIVE
        ):

            return TargetState.DETECTED

        if track.state == (
            TrackState.CONFIRMED
        ):

            if (
                track.hits
                >= self.config.minimum_confirmed_hits
            ):

                return TargetState.CONFIRMED

            return TargetState.TRACKING

        if track.state == (
            TrackState.LOST
        ):

            return TargetState.LOST

        if track.state == (
            TrackState.DELETED
        ):

            return TargetState.LOST

        return TargetState.UNKNOWN

    # =========================================================================
    # History
    # =========================================================================

    def _append_history(
        self,
        timestamp: float,
    ) -> None:
        """Append current observation to history."""

        if self._target_id is None:
            return

        observation = TargetObservation(
            timestamp=timestamp,

            x=self._x,
            y=self._y,

            width=self._width,
            height=self._height,

            confidence=self._confidence,

            track_id=self._target_id,

            frame_width=self._frame_width,
            frame_height=self._frame_height,
        )

        self._history.append(
            observation
        )

        if len(self._history) > (
            self.config.history_size
        ):

            del self._history[
                :-
                self.config.history_size
            ]

    # =========================================================================
    # Validity
    # =========================================================================

    def is_valid(self) -> bool:
        """
        Determine whether target is valid for mission processing.
        """

        with self._lock:

            if self._target_id is None:
                return False

            if self._state in (
                TargetState.UNKNOWN,
                TargetState.INVALID,
                TargetState.LOST,
            ):
                return False

            if (
                self._confidence
                < self.config.minimum_confidence
            ):
                return False

            if (
                self.config.require_confirmed_track
                and self._state
                != TargetState.CONFIRMED
            ):
                return False

            if self._frame_width <= 0:
                return False

            if self._frame_height <= 0:
                return False

            if self._width <= 0:
                return False

            if self._height <= 0:
                return False

            return True

    # =========================================================================
    # Position
    # =========================================================================

    def center(
        self,
    ) -> Tuple[float, float]:
        """Return current pixel center."""

        with self._lock:

            return (
                self._x,
                self._y,
            )

    # -------------------------------------------------------------------------

    def normalized_center(
        self,
    ) -> Tuple[float, float]:
        """Return normalized center."""

        with self._lock:

            if (
                self._frame_width <= 0
                or self._frame_height <= 0
            ):

                return (
                    0.0,
                    0.0,
                )

            return (
                self._x
                / self._frame_width,

                self._y
                / self._frame_height,
            )

    # =========================================================================
    # Center error
    # =========================================================================

    def center_error(
        self,
    ) -> Tuple[float, float]:
        """
        Return target error relative to image center.

        Returns:

            error_x
            error_y

        Normalized range is approximately:

            -1.0 ... +1.0
        """

        with self._lock:

            if (
                self._frame_width <= 0
                or self._frame_height <= 0
            ):

                return (
                    0.0,
                    0.0,
                )

            center_x = (
                self._frame_width
                / 2.0
            )

            center_y = (
                self._frame_height
                / 2.0
            )

            error_x = (
                self._x
                - center_x
            ) / center_x

            error_y = (
                self._y
                - center_y
            ) / center_y

            return (
                error_x,
                error_y,
            )

    # =========================================================================
    # Distance from image center
    # =========================================================================

    def center_distance(
        self,
    ) -> float:
        """
        Return normalized Euclidean distance from image center.
        """

        error_x, error_y = (
            self.center_error()
        )

        distance = math.sqrt(
            error_x * error_x
            +
            error_y * error_y
        )

        return min(
            1.0,
            distance
            / math.sqrt(2.0),
        )

    # =========================================================================
    # Bounding box
    # =========================================================================

    def bounding_box(
        self,
    ) -> Tuple[
        float,
        float,
        float,
        float,
    ]:
        """Return x1, y1, x2, y2."""

        with self._lock:

            return (
                self._x
                - self._width / 2.0,

                self._y
                - self._height / 2.0,

                self._x
                + self._width / 2.0,

                self._y
                + self._height / 2.0,
            )

    # =========================================================================
    # Velocity in image plane
    # =========================================================================

    def image_velocity(
        self,
    ) -> Tuple[float, float]:
        """
        Estimate target pixel velocity.

        Returns:

            vx_pixels_per_second
            vy_pixels_per_second

        Estimate is based on the last two observations.
        """

        with self._lock:

            if len(self._history) < 2:

                return (
                    0.0,
                    0.0,
                )

            previous = (
                self._history[-2]
            )

            current = (
                self._history[-1]
            )

            dt = (
                current.timestamp
                - previous.timestamp
            )

            if dt <= 0:

                return (
                    0.0,
                    0.0,
                )

            vx = (
                current.x
                - previous.x
            ) / dt

            vy = (
                current.y
                - previous.y
            ) / dt

            return (
                vx,
                vy,
            )

    # =========================================================================
    # Age / lost time
    # =========================================================================

    def age_s(self) -> float:
        """Return target age."""

        with self._lock:

            if self._first_seen is None:

                return 0.0

            return max(
                0.0,
                time.monotonic()
                - self._first_seen,
            )

    # -------------------------------------------------------------------------

    def lost_time_s(self) -> float:
        """Return time since last observation."""

        with self._lock:

            if self._last_seen is None:

                return float("inf")

            return max(
                0.0,
                time.monotonic()
                - self._last_seen,
            )

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def target_id(
        self,
    ) -> Optional[int]:

        with self._lock:
            return self._target_id

    # -------------------------------------------------------------------------

    @property
    def class_id(
        self,
    ) -> Optional[int]:

        with self._lock:
            return self._class_id

    # -------------------------------------------------------------------------

    @property
    def class_name(
        self,
    ) -> Optional[str]:

        with self._lock:
            return self._class_name

    # -------------------------------------------------------------------------

    @property
    def state(
        self,
    ) -> TargetState:

        with self._lock:
            return self._state

    # -------------------------------------------------------------------------

    @property
    def confidence(
        self,
    ) -> float:

        with self._lock:
            return self._confidence

    # -------------------------------------------------------------------------

    @property
    def width(
        self,
    ) -> float:

        with self._lock:
            return self._width

    # -------------------------------------------------------------------------

    @property
    def height(
        self,
    ) -> float:

        with self._lock:
            return self._height

    # -------------------------------------------------------------------------

    @property
    def frame_size(
        self,
    ) -> Tuple[int, int]:

        with self._lock:

            return (
                self._frame_width,
                self._frame_height,
            )

    # -------------------------------------------------------------------------

    @property
    def hit_count(
        self,
    ) -> int:

        with self._lock:
            return self._hit_count

    # -------------------------------------------------------------------------

    @property
    def missed_frames(
        self,
    ) -> int:

        with self._lock:
            return self._missed_frames

    # =========================================================================
    # History
    # =========================================================================

    def history(
        self,
    ) -> List[TargetObservation]:
        """Return a copy of target history."""

        with self._lock:

            return list(
                self._history
            )

    # =========================================================================
    # Snapshot
    # =========================================================================

    def snapshot(
        self,
    ) -> TargetSnapshot:
        """
        Return immutable snapshot of current target.
        """

        with self._lock:

            normalized_x = (
                self._x
                / self._frame_width
                if self._frame_width > 0
                else 0.0
            )

            normalized_y = (
                self._y
                / self._frame_height
                if self._frame_height > 0
                else 0.0
            )

            now = time.monotonic()

            age = (
                max(
                    0.0,
                    now
                    - self._first_seen,
                )
                if self._first_seen is not None
                else 0.0
            )

            lost_time = (
                max(
                    0.0,
                    now
                    - self._last_seen,
                )
                if self._last_seen is not None
                else float("inf")
            )

            return TargetSnapshot(
                target_id=(
                    self._target_id
                    if self._target_id is not None
                    else -1
                ),

                class_id=(
                    self._class_id
                    if self._class_id is not None
                    else -1
                ),

                class_name=(
                    self._class_name
                    if self._class_name is not None
                    else ""
                ),

                state=self._state,

                confidence=self._confidence,

                x=self._x,
                y=self._y,

                width=self._width,
                height=self._height,

                frame_width=self._frame_width,
                frame_height=self._frame_height,

                normalized_x=normalized_x,
                normalized_y=normalized_y,

                age_s=age,
                lost_time_s=lost_time,

                hit_count=self._hit_count,
                missed_frames=self._missed_frames,

                valid=self.is_valid(),

                timestamp=now,
            )

    # =========================================================================
    # Reset
    # =========================================================================

    def reset(self) -> None:
        """Reset target completely."""

        with self._lock:

            self._target_id = None

            self._class_id = None

            self._class_name = None

            self._state = (
                TargetState.UNKNOWN
            )

            self._confidence = 0.0

            self._x = 0.0
            self._y = 0.0

            self._width = 0.0
            self._height = 0.0

            self._frame_width = 0
            self._frame_height = 0

            self._first_seen = None
            self._last_seen = None
            self._last_update = None

            self._hit_count = 0
            self._missed_frames = 0

            self._history.clear()

    # =========================================================================
    # Representation
    # =========================================================================

    def __repr__(self) -> str:

        snapshot = self.snapshot()

        return (
            "Target("
            f"id={snapshot.target_id}, "
            f"class={snapshot.class_name!r}, "
            f"state={snapshot.state.value!r}, "
            f"confidence="
            f"{snapshot.confidence:.2f}, "
            f"center=("
            f"{snapshot.x:.1f}, "
            f"{snapshot.y:.1f}"
            ")"
            ")"
        )
