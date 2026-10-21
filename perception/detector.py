"""Filters raw YOLO detections and selects the most suitable target candidate."""

from __future__ import annotations

import math
import threading

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, List, Optional, Sequence, Tuple

from perception.yolo import (
    BoundingBox,
    Detection,
    YOLODetector,
)


# ============================================================================
# Exceptions
# ============================================================================

class DetectorError(Exception):
    """Base exception for detector subsystem."""


class DetectorConfigurationError(DetectorError):
    """Invalid detector configuration."""


class DetectorInputError(DetectorError):
    """Invalid detector input."""


# ============================================================================
# Target selection strategy
# ============================================================================

class TargetSelectionStrategy(str, Enum):
    """
    Strategy used to select one detection from multiple candidates.
    """

    HIGHEST_CONFIDENCE = "highest_confidence"

    LARGEST_AREA = "largest_area"

    CENTER_CLOSEST = "center_closest"

    CONFIDENCE_AND_CENTER = "confidence_and_center"


# ============================================================================
# Configuration
# ============================================================================

@dataclass(frozen=True)
class DetectorConfig:
    """
    Configuration for DetectionProcessor.
    """

    confidence_threshold: float = 0.50

    min_bbox_width_px: float = 5.0

    min_bbox_height_px: float = 5.0

    min_bbox_area_px: float = 25.0

    max_bbox_area_ratio: float = 1.0

    center_weight: float = 0.30

    confidence_weight: float = 0.70

    selection_strategy: TargetSelectionStrategy = (
        TargetSelectionStrategy.CONFIDENCE_AND_CENTER
    )

    allowed_classes: Optional[Tuple[str, ...]] = None

    allowed_class_ids: Optional[Tuple[int, ...]] = None

    maximum_candidates: int = 50

    require_valid_bbox: bool = True

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate detector configuration."""

        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise DetectorConfigurationError(
                "confidence_threshold must be between 0 and 1"
            )

        if self.min_bbox_width_px < 0:
            raise DetectorConfigurationError(
                "min_bbox_width_px cannot be negative"
            )

        if self.min_bbox_height_px < 0:
            raise DetectorConfigurationError(
                "min_bbox_height_px cannot be negative"
            )

        if self.min_bbox_area_px < 0:
            raise DetectorConfigurationError(
                "min_bbox_area_px cannot be negative"
            )

        if not 0.0 < self.max_bbox_area_ratio <= 1.0:
            raise DetectorConfigurationError(
                "max_bbox_area_ratio must be in (0, 1]"
            )

        if self.center_weight < 0:
            raise DetectorConfigurationError(
                "center_weight cannot be negative"
            )

        if self.confidence_weight < 0:
            raise DetectorConfigurationError(
                "confidence_weight cannot be negative"
            )

        if (
            self.center_weight
            + self.confidence_weight
            <= 0
        ):
            raise DetectorConfigurationError(
                "At least one selection weight must be positive"
            )

        if self.maximum_candidates <= 0:
            raise DetectorConfigurationError(
                "maximum_candidates must be positive"
            )


# ============================================================================
# Candidate
# ============================================================================

@dataclass(frozen=True)
class DetectionCandidate:
    """Detection together with derived target-selection scores."""

    detection: Detection

    confidence_score: float

    center_score: float

    area_score: float

    final_score: float

    center_distance_normalized: float

    accepted: bool = True

    rejection_reason: Optional[str] = None

    @property
    def class_id(self) -> int:
        return self.detection.class_id

    @property
    def class_name(self) -> str:
        return self.detection.class_name

    @property
    def confidence(self) -> float:
        return self.detection.confidence

    @property
    def bbox(self) -> BoundingBox:
        return self.detection.bbox

    @property
    def center(self) -> Tuple[float, float]:
        return self.detection.center


# ============================================================================
# Processing statistics
# ============================================================================

@dataclass
class DetectorStatistics:
    """
    Runtime statistics for DetectionProcessor.
    """

    frames_processed: int = 0

    detections_received: int = 0

    detections_accepted: int = 0

    detections_rejected: int = 0

    candidates_created: int = 0

    targets_selected: int = 0

    no_target_frames: int = 0

    class_rejections: int = 0

    confidence_rejections: int = 0

    bbox_rejections: int = 0

    last_detection_timestamp: Optional[float] = None

    last_target_timestamp: Optional[float] = None


# ============================================================================
# Detection Processor
# ============================================================================

class DetectionProcessor:
    """Filters YOLO detections and ranks candidate targets."""

    def __init__(
        self,
        detector: YOLODetector,
        config: Optional[DetectorConfig] = None,
    ) -> None:

        if detector is None:
            raise DetectorInputError(
                "YOLO detector cannot be None"
            )

        self._detector = detector

        self.config = (
            config
            or DetectorConfig()
        )

        self._lock = threading.RLock()

        self._statistics = DetectorStatistics()

    # =========================================================================
    # Process
    # =========================================================================

    def process(
        self,
        detections: Sequence[Detection],
    ) -> List[DetectionCandidate]:
        """Process the supplied input and return the result."""

        if detections is None:
            raise DetectorInputError(
                "detections cannot be None"
            )

        with self._lock:

            self._statistics.frames_processed += 1

            self._statistics.detections_received += (
                len(detections)
            )

        candidates: List[DetectionCandidate] = []

        for detection in detections:

            candidate = self._evaluate(
                detection
            )

            if candidate.accepted:

                candidates.append(
                    candidate
                )

                with self._lock:

                    self._statistics.detections_accepted += 1

            else:

                with self._lock:

                    self._statistics.detections_rejected += 1

        candidates.sort(
            key=lambda candidate:
                candidate.final_score,
            reverse=True,
        )

        candidates = candidates[
            :self.config.maximum_candidates
        ]

        with self._lock:

            self._statistics.candidates_created += (
                len(candidates)
            )

            if candidates:

                self._statistics.last_detection_timestamp = (
                    detections[0].timestamp
                    if detections
                    else None
                )

            else:

                self._statistics.no_target_frames += 1

        return candidates

    # =========================================================================
    # Evaluate
    # =========================================================================

    def _evaluate(
        self,
        detection: Detection,
    ) -> DetectionCandidate:
        """Evaluate one detection."""

        if detection is None:

            raise DetectorInputError(
                "Detection cannot be None"
            )

        if (
            self.config.allowed_classes is not None
            and detection.class_name
            not in self.config.allowed_classes
        ):

            with self._lock:
                self._statistics.class_rejections += 1

            return self._rejected_candidate(
                detection,
                "class_not_allowed",
            )

        if (
            self.config.allowed_class_ids is not None
            and detection.class_id
            not in self.config.allowed_class_ids
        ):

            with self._lock:
                self._statistics.class_rejections += 1

            return self._rejected_candidate(
                detection,
                "class_id_not_allowed",
            )

        if (
            detection.confidence
            < self.config.confidence_threshold
        ):

            with self._lock:
                self._statistics.confidence_rejections += 1

            return self._rejected_candidate(
                detection,
                "confidence_below_threshold",
            )

        bbox_valid, bbox_reason = (
            self._validate_bbox(
                detection
            )
        )

        if (
            self.config.require_valid_bbox
            and not bbox_valid
        ):

            with self._lock:
                self._statistics.bbox_rejections += 1

            return self._rejected_candidate(
                detection,
                bbox_reason,
            )

        confidence_score = (
            self._confidence_score(
                detection
            )
        )

        center_score = (
            self._center_score(
                detection
            )
        )

        area_score = (
            self._area_score(
                detection
            )
        )

        final_score = (
            self._calculate_final_score(
                confidence_score,
                center_score,
                area_score,
            )
        )

        center_distance = (
            self._center_distance_normalized(
                detection
            )
        )

        return DetectionCandidate(
            detection=detection,
            confidence_score=confidence_score,
            center_score=center_score,
            area_score=area_score,
            final_score=final_score,
            center_distance_normalized=center_distance,
        )

    # =========================================================================
    # Validation
    # =========================================================================

    def _validate_bbox(
        self,
        detection: Detection,
    ) -> Tuple[bool, str]:
        """Validate detection bounding box."""

        bbox = detection.bbox

        if bbox.width < self.config.min_bbox_width_px:

            return (
                False,
                "bbox_width_too_small",
            )

        if bbox.height < self.config.min_bbox_height_px:

            return (
                False,
                "bbox_height_too_small",
            )

        if bbox.area < self.config.min_bbox_area_px:

            return (
                False,
                "bbox_area_too_small",
            )

        frame_area = (
            detection.frame_width
            * detection.frame_height
        )

        if frame_area <= 0:

            return (
                False,
                "invalid_frame_dimensions",
            )

        area_ratio = (
            bbox.area
            / frame_area
        )

        if (
            area_ratio
            > self.config.max_bbox_area_ratio
        ):

            return (
                False,
                "bbox_area_too_large",
            )

        return True, ""

    # =========================================================================
    # Scoring
    # =========================================================================

    @staticmethod
    def _confidence_score(
        detection: Detection,
    ) -> float:
        """Normalize confidence to 0..1."""

        return max(
            0.0,
            min(
                1.0,
                float(
                    detection.confidence
                ),
            ),
        )

    # -------------------------------------------------------------------------

    def _center_score(
        self,
        detection: Detection,
    ) -> float:
        """
        Calculate how close detection is to image center.

        Center:

            1.0

        Image edge:

            approximately 0.0
        """

        distance = (
            self._center_distance_normalized(
                detection
            )
        )

        return max(
            0.0,
            min(
                1.0,
                1.0 - distance,
            ),
        )

    # -------------------------------------------------------------------------

    @staticmethod
    def _center_distance_normalized(
        detection: Detection,
    ) -> float:
        """Calculate normalized distance from image center."""

        frame_width = (
            detection.frame_width
        )

        frame_height = (
            detection.frame_height
        )

        if (
            frame_width <= 0
            or frame_height <= 0
        ):

            return 1.0

        center_x = frame_width / 2.0
        center_y = frame_height / 2.0

        detection_x, detection_y = (
            detection.center
        )

        dx = (
            detection_x - center_x
        ) / center_x

        dy = (
            detection_y - center_y
        ) / center_y

        distance = math.sqrt(
            dx * dx
            + dy * dy
        )

        # Maximum distance from center to image
        # corner is sqrt(2).
        return min(
            1.0,
            distance / math.sqrt(2.0),
        )

    # -------------------------------------------------------------------------

    @staticmethod
    def _area_score(
        detection: Detection,
    ) -> float:

        frame_area = (
            detection.frame_width
            * detection.frame_height
        )

        if frame_area <= 0:
            return 0.0

        ratio = (
            detection.bbox.area
            / frame_area
        )

        # Saturation prevents very large objects from
        # completely dominating selection.
        return min(
            1.0,
            math.sqrt(
                max(
                    0.0,
                    ratio,
                )
            ) * 4.0,
        )

    # -------------------------------------------------------------------------

    def _calculate_final_score(
        self,
        confidence_score: float,
        center_score: float,
        area_score: float,
    ) -> float:
        """Calculate final candidate score."""

        strategy = (
            self.config.selection_strategy
        )

        if strategy == (
            TargetSelectionStrategy.HIGHEST_CONFIDENCE
        ):

            return confidence_score

        if strategy == (
            TargetSelectionStrategy.LARGEST_AREA
        ):

            return area_score

        if strategy == (
            TargetSelectionStrategy.CENTER_CLOSEST
        ):

            return center_score

        total_weight = (
            self.config.confidence_weight
            + self.config.center_weight
        )

        if total_weight <= 0:

            return confidence_score

        return (
            (
                confidence_score
                * self.config.confidence_weight
            )
            +
            (
                center_score
                * self.config.center_weight
            )
        ) / total_weight

    # =========================================================================
    # Target selection
    # =========================================================================

    def select_target(
        self,
        candidates: Sequence[DetectionCandidate],
    ) -> Optional[DetectionCandidate]:

        if not candidates:

            with self._lock:
                self._statistics.no_target_frames += 1

            return None

        valid_candidates = [
            candidate
            for candidate in candidates
            if candidate.accepted
        ]

        if not valid_candidates:

            return None

        target = max(
            valid_candidates,
            key=lambda candidate:
                candidate.final_score,
        )

        with self._lock:

            self._statistics.targets_selected += 1

            self._statistics.last_target_timestamp = (
                target.detection.timestamp
            )

        return target

    # =========================================================================
    # Combined operation
    # =========================================================================

    def process_and_select(
        self,
        detections: Sequence[Detection],
    ) -> Optional[DetectionCandidate]:
        """
        Process detections and immediately select best target.
        """

        candidates = self.process(
            detections
        )

        return self.select_target(
            candidates
        )

    # =========================================================================
    # Class filtering
    # =========================================================================

    def filter_class(
        self,
        candidates: Sequence[DetectionCandidate],
        class_name: str,
    ) -> List[DetectionCandidate]:
        """Return candidates belonging to a class."""

        return [
            candidate
            for candidate in candidates
            if candidate.class_name == class_name
        ]

    # -------------------------------------------------------------------------

    def filter_class_id(
        self,
        candidates: Sequence[DetectionCandidate],
        class_id: int,
    ) -> List[DetectionCandidate]:
        """Return candidates belonging to a class ID."""

        return [
            candidate
            for candidate in candidates
            if candidate.class_id == class_id
        ]

    # =========================================================================
    # Spatial filtering
    # =========================================================================

    def filter_center_region(
        self,
        candidates: Sequence[DetectionCandidate],
        max_distance: float = 0.5,
    ) -> List[DetectionCandidate]:
        """
        Keep candidates within normalized distance from image center.

        max_distance:

            0.0 = exact center

            1.0 = entire image
        """

        if not 0.0 <= max_distance <= 1.0:

            raise ValueError(
                "max_distance must be between 0 and 1"
            )

        return [
            candidate
            for candidate in candidates
            if (
                candidate.center_distance_normalized
                <= max_distance
            )
        ]

    # =========================================================================
    # Statistics
    # =========================================================================

    def statistics(self) -> DetectorStatistics:
        """Return a statistics snapshot."""

        with self._lock:

            return DetectorStatistics(
                frames_processed=(
                    self._statistics.frames_processed
                ),

                detections_received=(
                    self._statistics.detections_received
                ),

                detections_accepted=(
                    self._statistics.detections_accepted
                ),

                detections_rejected=(
                    self._statistics.detections_rejected
                ),

                candidates_created=(
                    self._statistics.candidates_created
                ),

                targets_selected=(
                    self._statistics.targets_selected
                ),

                no_target_frames=(
                    self._statistics.no_target_frames
                ),

                class_rejections=(
                    self._statistics.class_rejections
                ),

                confidence_rejections=(
                    self._statistics.confidence_rejections
                ),

                bbox_rejections=(
                    self._statistics.bbox_rejections
                ),

                last_detection_timestamp=(
                    self._statistics.last_detection_timestamp
                ),

                last_target_timestamp=(
                    self._statistics.last_target_timestamp
                ),
            )

    # =========================================================================
    # Helpers
    # =========================================================================

    @staticmethod
    def _rejected_candidate(
        detection: Detection,
        reason: str,
    ) -> DetectionCandidate:
        """Create rejected candidate."""

        return DetectionCandidate(
            detection=detection,
            confidence_score=0.0,
            center_score=0.0,
            area_score=0.0,
            final_score=0.0,
            center_distance_normalized=1.0,
            accepted=False,
            rejection_reason=reason,
        )

    # =========================================================================
    # Representation
    # =========================================================================

    def __repr__(self) -> str:

        return (
            "DetectionProcessor("
            f"confidence="
            f"{self.config.confidence_threshold:.2f}, "
            f"strategy="
            f"{self.config.selection_strategy.value}"
            ")"
        )
