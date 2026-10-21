"""Associates detections across frames and maintains stable target track IDs."""

from __future__ import annotations

import math
import threading
import time

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from perception.detector import DetectionCandidate


# ============================================================================
# Exceptions
# ============================================================================

class TrackerError(Exception):
    """Base tracker exception."""


class TrackerConfigurationError(TrackerError):
    """Invalid tracker configuration."""


# ============================================================================
# Track state
# ============================================================================

class TrackState(str, Enum):
    """Lifecycle state of a target track."""

    TENTATIVE = "tentative"

    CONFIRMED = "confirmed"

    LOST = "lost"

    DELETED = "deleted"


# ============================================================================
# Configuration
# ============================================================================

@dataclass(frozen=True)
class TrackerConfig:
    """
    Target tracker configuration.

    max_match_distance:
        Normalized maximum center distance allowed for matching.

    max_size_ratio_change:
        Maximum allowed relative bounding-box size change.

    max_lost_frames:
        Number of frames a target may be absent before deletion.

    max_lost_time_s:
        Maximum wall-clock time a track may remain unseen.

    min_confirmed_hits:
        Number of successful matches required to confirm a track.

    smoothing_alpha:
        Exponential smoothing factor.

        1.0 -> no smoothing
        smaller -> more smoothing

    class_match_required:
        If True, detections with different classes cannot be matched.

    max_tracks:
        Maximum simultaneous tracks.
    """

    max_match_distance: float = 0.20

    max_size_ratio_change: float = 2.5

    max_lost_frames: int = 10

    max_lost_time_s: float = 1.0

    min_confirmed_hits: int = 3

    smoothing_alpha: float = 0.60

    class_match_required: bool = True

    max_tracks: int = 20

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate tracker configuration."""

        if self.max_match_distance <= 0:
            raise TrackerConfigurationError(
                "max_match_distance must be positive"
            )

        if self.max_match_distance > 1.0:
            raise TrackerConfigurationError(
                "max_match_distance cannot exceed 1.0"
            )

        if self.max_size_ratio_change < 1.0:
            raise TrackerConfigurationError(
                "max_size_ratio_change must be >= 1.0"
            )

        if self.max_lost_frames < 0:
            raise TrackerConfigurationError(
                "max_lost_frames cannot be negative"
            )

        if self.max_lost_time_s < 0:
            raise TrackerConfigurationError(
                "max_lost_time_s cannot be negative"
            )

        if self.min_confirmed_hits <= 0:
            raise TrackerConfigurationError(
                "min_confirmed_hits must be positive"
            )

        if not (
            0.0 < self.smoothing_alpha <= 1.0
        ):
            raise TrackerConfigurationError(
                "smoothing_alpha must be in (0, 1]"
            )

        if self.max_tracks <= 0:
            raise TrackerConfigurationError(
                "max_tracks must be positive"
            )


# ============================================================================
# Track point
# ============================================================================

@dataclass
class TrackPoint:
    """
    One historical observation of a target.
    """

    timestamp: float

    x: float

    y: float

    width: float

    height: float

    confidence: float


# ============================================================================
# Track
# ============================================================================

@dataclass
class Track:
    """
    Persistent target track.

    Track ID remains stable while the target is tracked.
    """

    track_id: int

    class_id: int

    class_name: str

    state: TrackState = TrackState.TENTATIVE

    first_seen: float = 0.0

    last_seen: float = 0.0

    hits: int = 0

    missed_frames: int = 0

    total_matches: int = 0

    confidence: float = 0.0

    x: float = 0.0

    y: float = 0.0

    width: float = 0.0

    height: float = 0.0

    frame_width: int = 0

    frame_height: int = 0

    history: List[TrackPoint] = field(
        default_factory=list
    )

    @property
    def center(self) -> Tuple[float, float]:
        """Return current target center."""

        return self.x, self.y

    @property
    def bbox(self) -> Tuple[
        float,
        float,
        float,
        float,
    ]:
        """Return current bounding box."""

        return (
            self.x - self.width / 2.0,
            self.y - self.height / 2.0,
            self.x + self.width / 2.0,
            self.y + self.height / 2.0,
        )

    @property
    def age_s(self) -> float:
        """Return track age."""

        if self.first_seen <= 0:
            return 0.0

        return max(
            0.0,
            time.monotonic()
            - self.first_seen,
        )

    @property
    def time_since_seen_s(self) -> float:
        """Return time since last observation."""

        if self.last_seen <= 0:
            return float("inf")

        return max(
            0.0,
            time.monotonic()
            - self.last_seen,
        )

    @property
    def is_confirmed(self) -> bool:
        """Return whether track is confirmed."""

        return (
            self.state
            == TrackState.CONFIRMED
        )

    @property
    def is_lost(self) -> bool:
        """Return whether track is lost."""

        return (
            self.state
            == TrackState.LOST
        )


# ============================================================================
# Tracker statistics
# ============================================================================

@dataclass
class TrackerStatistics:
    """Runtime tracker statistics."""

    frames_processed: int = 0

    detections_received: int = 0

    tracks_created: int = 0

    tracks_confirmed: int = 0

    tracks_deleted: int = 0

    matches: int = 0

    unmatched_detections: int = 0

    unmatched_tracks: int = 0

    active_tracks: int = 0

    lost_tracks: int = 0

    last_update_timestamp: Optional[float] = None


# ============================================================================
# Tracker
# ============================================================================

class TargetTracker:

    def __init__(
        self,
        config: Optional[TrackerConfig] = None,
    ) -> None:

        self.config = (
            config
            or TrackerConfig()
        )

        self._lock = threading.RLock()

        self._tracks: Dict[
            int,
            Track,
        ] = {}

        self._next_track_id = 1

        self._statistics = (
            TrackerStatistics()
        )

    # =========================================================================
    # Main update
    # =========================================================================

    def update(
        self,
        candidates: Sequence[DetectionCandidate],
    ) -> List[Track]:
        """
        Update tracker with detections from current frame.

        Returns currently active tracks.
        """

        now = time.monotonic()

        with self._lock:

            self._statistics.frames_processed += 1

            self._statistics.detections_received += (
                len(candidates)
            )

            self._statistics.last_update_timestamp = (
                now
            )

            if not candidates:

                self._mark_all_unmatched(
                    now
                )

                self._remove_expired_tracks(
                    now
                )

                return self.active_tracks()

            matched_track_ids = set()

            matched_candidate_indices = set()

            # ---------------------------------------------------------------
            # Build candidate matches
            # ---------------------------------------------------------------

            pairs = []

            for candidate_index, candidate in enumerate(
                candidates
            ):

                for track_id, track in self._tracks.items():

                    if track.state == TrackState.DELETED:
                        continue

                    if (
                        self.config.class_match_required
                        and (
                            candidate.class_id
                            != track.class_id
                        )
                    ):
                        continue

                    distance = (
                        self._candidate_track_distance(
                            candidate,
                            track,
                        )
                    )

                    if distance is None:
                        continue

                    pairs.append(
                        (
                            distance,
                            candidate_index,
                            track_id,
                        )
                    )

            pairs.sort(
                key=lambda item: item[0]
            )

            # ---------------------------------------------------------------
            # Greedy matching
            # ---------------------------------------------------------------

            for (
                distance,
                candidate_index,
                track_id,
            ) in pairs:

                if candidate_index in (
                    matched_candidate_indices
                ):
                    continue

                if track_id in matched_track_ids:
                    continue

                candidate = candidates[
                    candidate_index
                ]

                track = self._tracks.get(
                    track_id
                )

                if track is None:
                    continue

                if (
                    distance
                    > self.config.max_match_distance
                ):
                    continue

                self._update_track(
                    track,
                    candidate,
                    now,
                )

                matched_track_ids.add(
                    track_id
                )

                matched_candidate_indices.add(
                    candidate_index
                )

                self._statistics.matches += 1

            # ---------------------------------------------------------------
            # Create tracks for unmatched detections
            # ---------------------------------------------------------------

            for candidate_index, candidate in enumerate(
                candidates
            ):

                if candidate_index in (
                    matched_candidate_indices
                ):
                    continue

                if len(self._tracks) >= (
                    self.config.max_tracks
                ):
                    break

                track = self._create_track(
                    candidate,
                    now,
                )

                self._tracks[
                    track.track_id
                ] = track

                self._statistics.tracks_created += 1

                matched_candidate_indices.add(
                    candidate_index
                )

            # ---------------------------------------------------------------
            # Mark unmatched tracks
            # ---------------------------------------------------------------

            for track_id, track in list(
                self._tracks.items()
            ):

                if track_id in matched_track_ids:
                    continue

                # Newly created tracks are already current.
                if track.last_seen == now:
                    continue

                track.missed_frames += 1

                if track.state == (
                    TrackState.CONFIRMED
                ):
                    track.state = (
                        TrackState.LOST
                    )

                self._statistics.unmatched_tracks += 1

            self._statistics.unmatched_detections += (
                len(candidates)
                - len(matched_candidate_indices)
            )

            self._remove_expired_tracks(
                now
            )

            self._update_statistics()

            return self.active_tracks()

    # =========================================================================
    # Track matching
    # =========================================================================

    def _candidate_track_distance(
        self,
        candidate: DetectionCandidate,
        track: Track,
    ) -> Optional[float]:
        """
        Calculate normalized matching distance.

        Returns None when candidate cannot match the track.
        """

        detection = candidate.detection

        if (
            detection.frame_width
            != track.frame_width
            or detection.frame_height
            != track.frame_height
        ):

            return None

        center_x, center_y = (
            detection.center
        )

        track_x = track.x
        track_y = track.y

        frame_width = (
            max(
                1,
                detection.frame_width,
            )
        )

        frame_height = (
            max(
                1,
                detection.frame_height,
            )
        )

        dx = (
            center_x - track_x
        ) / frame_width

        dy = (
            center_y - track_y
        ) / frame_height

        center_distance = math.sqrt(
            dx * dx
            + dy * dy
        )

        # ---------------------------------------------------------------
        # Size consistency
        # ---------------------------------------------------------------

        detection_area = (
            detection.bbox.area
        )

        track_area = (
            track.width
            * track.height
        )

        if (
            detection_area <= 0
            or track_area <= 0
        ):
            return None

        ratio = (
            detection_area
            / track_area
        )

        ratio_inverse = (
            1.0 / ratio
        )

        size_ratio = max(
            ratio,
            ratio_inverse,
        )

        if (
            size_ratio
            > self.config.max_size_ratio_change
        ):

            return None

        size_penalty = min(
            1.0,
            abs(
                math.log(
                    max(
                        ratio,
                        1e-9,
                    )
                )
            ),
        )

        return (
            center_distance
            + 0.15 * size_penalty
        )

    # =========================================================================
    # Track creation
    # =========================================================================

    def _create_track(
        self,
        candidate: DetectionCandidate,
        timestamp: float,
    ) -> Track:
        """Create a new track."""

        detection = candidate.detection

        center_x, center_y = (
            detection.center
        )

        track = Track(
            track_id=self._next_track_id,

            class_id=detection.class_id,

            class_name=detection.class_name,

            state=TrackState.TENTATIVE,

            first_seen=timestamp,

            last_seen=timestamp,

            hits=1,

            missed_frames=0,

            total_matches=1,

            confidence=detection.confidence,

            x=center_x,

            y=center_y,

            width=detection.bbox.width,

            height=detection.bbox.height,

            frame_width=detection.frame_width,

            frame_height=detection.frame_height,
        )

        self._next_track_id += 1

        track.history.append(
            TrackPoint(
                timestamp=timestamp,
                x=center_x,
                y=center_y,
                width=detection.bbox.width,
                height=detection.bbox.height,
                confidence=detection.confidence,
            )
        )

        if (
            track.hits
            >= self.config.min_confirmed_hits
        ):

            track.state = (
                TrackState.CONFIRMED
            )

            self._statistics.tracks_confirmed += 1

        return track

    # =========================================================================
    # Track update
    # =========================================================================

    def _update_track(
        self,
        track: Track,
        candidate: DetectionCandidate,
        timestamp: float,
    ) -> None:
        """Update existing track."""

        detection = candidate.detection

        new_x, new_y = (
            detection.center
        )

        alpha = (
            self.config.smoothing_alpha
        )

        track.x = (
            alpha * new_x
            + (1.0 - alpha) * track.x
        )

        track.y = (
            alpha * new_y
            + (1.0 - alpha) * track.y
        )

        track.width = (
            alpha * detection.bbox.width
            + (1.0 - alpha) * track.width
        )

        track.height = (
            alpha * detection.bbox.height
            + (1.0 - alpha) * track.height
        )

        track.confidence = (
            alpha * detection.confidence
            + (1.0 - alpha)
            * track.confidence
        )

        track.last_seen = timestamp

        track.hits += 1

        track.total_matches += 1

        track.missed_frames = 0

        track.frame_width = (
            detection.frame_width
        )

        track.frame_height = (
            detection.frame_height
        )

        track.history.append(
            TrackPoint(
                timestamp=timestamp,
                x=track.x,
                y=track.y,
                width=track.width,
                height=track.height,
                confidence=detection.confidence,
            )
        )

        # Keep memory bounded.
        if len(track.history) > 100:

            del track.history[:-100]

        if (
            track.hits
            >= self.config.min_confirmed_hits
        ):

            if track.state != (
                TrackState.CONFIRMED
            ):

                track.state = (
                    TrackState.CONFIRMED
                )

                self._statistics.tracks_confirmed += 1

        else:

            track.state = (
                TrackState.TENTATIVE
            )

    # =========================================================================
    # Missing tracks
    # =========================================================================

    def _mark_all_unmatched(
        self,
        timestamp: float,
    ) -> None:
        """Mark every track as missed."""

        for track in self._tracks.values():

            track.missed_frames += 1

            if track.state == (
                TrackState.CONFIRMED
            ):

                track.state = (
                    TrackState.LOST
                )

    # =========================================================================
    # Expiration
    # =========================================================================

    def _remove_expired_tracks(
        self,
        timestamp: float,
    ) -> None:
        """Delete tracks that have been lost too long."""

        for track_id, track in list(
            self._tracks.items()
        ):

            frame_expired = (
                track.missed_frames
                > self.config.max_lost_frames
            )

            time_expired = (
                timestamp
                - track.last_seen
                > self.config.max_lost_time_s
            )

            if (
                frame_expired
                or time_expired
            ):

                track.state = (
                    TrackState.DELETED
                )

                del self._tracks[
                    track_id
                ]

                self._statistics.tracks_deleted += 1

    # =========================================================================
    # Queries
    # =========================================================================

    def active_tracks(
        self,
    ) -> List[Track]:
        """Return all non-deleted tracks."""

        with self._lock:

            return [
                track
                for track in self._tracks.values()
                if track.state != TrackState.DELETED
            ]

    # -------------------------------------------------------------------------

    def confirmed_tracks(
        self,
    ) -> List[Track]:
        """Return confirmed tracks."""

        with self._lock:

            return [
                track
                for track in self._tracks.values()
                if track.state
                == TrackState.CONFIRMED
            ]

    # -------------------------------------------------------------------------

    def lost_tracks(
        self,
    ) -> List[Track]:
        """Return currently lost tracks."""

        with self._lock:

            return [
                track
                for track in self._tracks.values()
                if track.state
                == TrackState.LOST
            ]

    # -------------------------------------------------------------------------

    def get_track(
        self,
        track_id: int,
    ) -> Optional[Track]:
        """Return a specific track."""

        with self._lock:

            return self._tracks.get(
                track_id
            )

    # =========================================================================
    # Best track
    # =========================================================================

    def best_track(
        self,
        confirmed_only: bool = True,
    ) -> Optional[Track]:
        """
        Select best current track.

        Selection:

            1. Confirmed state
            2. Confidence
            3. Fewer missed frames
        """

        with self._lock:

            tracks = (
                self.confirmed_tracks()
                if confirmed_only
                else self.active_tracks()
            )

            if not tracks:
                return None

            return max(
                tracks,
                key=lambda track: (
                    track.confidence,
                    -track.missed_frames,
                ),
            )

    # =========================================================================
    # Class filtering
    # =========================================================================

    def tracks_by_class(
        self,
        class_name: str,
    ) -> List[Track]:
        """Return tracks matching class name."""

        with self._lock:

            return [
                track
                for track in self._tracks.values()
                if (
                    track.class_name
                    == class_name
                )
                and
                track.state
                != TrackState.DELETED
            ]

    # -------------------------------------------------------------------------

    def tracks_by_class_id(
        self,
        class_id: int,
    ) -> List[Track]:
        """Return tracks matching class ID."""

        with self._lock:

            return [
                track
                for track in self._tracks.values()
                if (
                    track.class_id
                    == class_id
                )
                and
                track.state
                != TrackState.DELETED
            ]

    # =========================================================================
    # Reset
    # =========================================================================

    def reset(self) -> None:
        """Delete all current tracks."""

        with self._lock:

            self._tracks.clear()

            self._next_track_id = 1

            self._statistics = (
                TrackerStatistics()
            )

    # =========================================================================
    # Statistics
    # =========================================================================

    def statistics(
        self,
    ) -> TrackerStatistics:
        """Return statistics snapshot."""

        with self._lock:

            active = len(
                [
                    track
                    for track in self._tracks.values()
                    if track.state
                    != TrackState.DELETED
                ]
            )

            lost = len(
                [
                    track
                    for track in self._tracks.values()
                    if track.state
                    == TrackState.LOST
                ]
            )

            return TrackerStatistics(
                frames_processed=(
                    self._statistics.frames_processed
                ),

                detections_received=(
                    self._statistics.detections_received
                ),

                tracks_created=(
                    self._statistics.tracks_created
                ),

                tracks_confirmed=(
                    self._statistics.tracks_confirmed
                ),

                tracks_deleted=(
                    self._statistics.tracks_deleted
                ),

                matches=(
                    self._statistics.matches
                ),

                unmatched_detections=(
                    self._statistics
                    .unmatched_detections
                ),

                unmatched_tracks=(
                    self._statistics
                    .unmatched_tracks
                ),

                active_tracks=active,

                lost_tracks=lost,

                last_update_timestamp=(
                    self._statistics
                    .last_update_timestamp
                ),
            )

    # =========================================================================
    # Representation
    # =========================================================================

    def __len__(self) -> int:

        return len(
            self.active_tracks()
        )

    # -------------------------------------------------------------------------

    def __repr__(self) -> str:

        stats = self.statistics()

        return (
            "TargetTracker("
            f"active={stats.active_tracks}, "
            f"lost={stats.lost_tracks}, "
            f"created={stats.tracks_created}, "
            f"deleted={stats.tracks_deleted}"
            ")"
        )
