"""YOLO-based object detection adapters used by the perception subsystem.

Model files are intentionally excluded from source control; configure their path
in config.py and use a PyTorch build compatible with the target device.
"""


import logging
import math
import os
import time
import traceback
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np

from ultralytics import YOLO


logger = logging.getLogger(__name__)


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self):
        if self.x2 < self.x1:
            self.x1, self.x2 = self.x2, self.x1

        if self.y2 < self.y1:
            self.y1, self.y2 = self.y2, self.y1

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2.0


@dataclass
class Detection:
    bbox: BoundingBox
    confidence: float
    class_id: int
    class_name: str
    timestamp: float = 0.0
    inference_id: int = 0

    @property
    def center_x(self) -> float:
        return self.bbox.center_x

    @property
    def center_y(self) -> float:
        return self.bbox.center_y


@dataclass
class YOLOStatistics:
    total_frames: int = 0
    processed_frames: int = 0
    total_detections: int = 0
    mannequin_detections: int = 0
    tent_detections: int = 0
    inference_time_total: float = 0.0
    last_inference_time: float = 0.0

    @property
    def average_inference_time(self) -> float:
        if self.processed_frames <= 0:
            return 0.0

        return self.inference_time_total / self.processed_frames


@dataclass
class YOLOConfig:
    model_path: str

    image_size: int = 1024

    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45

    max_detections: int = 300

    device: Optional[Any] = None

    half_precision: bool = True

    verbose: bool = False

    classes: Optional[List[int]] = None

    # SAHI
    sahi_rows: int = 1
    sahi_cols: int = 1

    # Frame skipping
    frame_skip: int = 1

    # NMS
    use_nms: bool = True


# ============================================================
# YOLO DETECTOR
# ============================================================

class YOLODetector:

    def __init__(self, config: YOLOConfig):

        self.config = config

        # ----------------------------------------------------
        # Force an image size of 1024 for TensorRT / RT-DETR static batch=1.
        # ----------------------------------------------------

        self.config.image_size = 1024

        # ----------------------------------------------------
        # Environment variables
        # ----------------------------------------------------

        self.config.sahi_rows = max(
            1,
            int(os.getenv(
                "YOLO_SAHI_ROWS",
                str(self.config.sahi_rows)
            ))
        )

        self.config.sahi_cols = max(
            1,
            int(os.getenv(
                "YOLO_SAHI_COLS",
                str(self.config.sahi_cols)
            ))
        )

        self.config.frame_skip = max(
            1,
            int(os.getenv(
                "YOLO_FRAME_SKIP",
                str(self.config.frame_skip)
            ))
        )

        # ----------------------------------------------------
        # MODEL
        # ----------------------------------------------------

        self._model = None
        self._initialized = False

        # ----------------------------------------------------
        # CLASS NAMES
        #
        # Class 0: mannequin (model label: "manken")
        # Class 1: canopy (model label: "tente")
        #
        # Ignore invalid class names emitted by the TensorRT engine;
        # do not use generic names such as class0 ... class998.
        # ----------------------------------------------------

        self._class_names = {
            0: "manken",
            1: "tente",
        }

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        self._statistics = YOLOStatistics()

        self._inference_counter = 0

        self._last_detections: List[Detection] = []

        # ----------------------------------------------------
        # Live FPS reporting
        # Print camera and inference FPS to the terminal every second.
        # The status is printed here; per-window counters are stored here.
        # ----------------------------------------------------
        self._fps_report_time = time.perf_counter()
        self._fps_window_frames = 0
        self._fps_window_processed = 0
        self._fps_window_inference_time = 0.0

        # ----------------------------------------------------
        # Load the model.
        # ----------------------------------------------------

        self._load_model()

    # ========================================================
    # LOAD COMPATIBILITY METHOD
    # ========================================================

    def load(self):
        """Load the configured model or resource."""

        if self._model is None:
            self._load_model()

        self._initialized = True

        return self

    # ========================================================
    # MODEL LOAD
    # ========================================================

    def _load_model(self):

        try:

            logger.info(
                "[YOLO] Loading model: %s",
                self.config.model_path
            )

            # ------------------------------------------------
            # Ultralytics YOLO / RT-DETR
            # ------------------------------------------------

            self._model = YOLO(
                self.config.model_path,
                task="detect"
            )

            # ------------------------------------------------
            # Check class names in the engine metadata.
            #
            # Use these names only when the model provides exactly two valid classes.
            # However, if TensorRT'ten class0...class998 gelirse
            # Do not use this fallback.
            # ------------------------------------------------

            try:

                model_names = getattr(
                    self._model,
                    "names",
                    None
                )

                valid_names = None

                if isinstance(model_names, dict):

                    converted = {
                        int(k): str(v)
                        for k, v in model_names.items()
                    }

                    if (
                        len(converted) == 2
                        and 0 in converted
                        and 1 in converted
                        and not all(
                            v.startswith("class")
                            for v in converted.values()
                        )
                    ):
                        valid_names = converted

                elif isinstance(model_names, list):

                    if (
                        len(model_names) == 2
                        and not all(
                            str(v).startswith("class")
                            for v in model_names
                        )
                    ):
                        valid_names = {
                            i: str(v)
                            for i, v in enumerate(model_names)
                        }

                if valid_names is not None:

                    self._class_names = valid_names

                else:

                    self._class_names = {
                        0: "manken",
                        1: "tente",
                    }

            except Exception:

                self._class_names = {
                    0: "manken",
                    1: "tente",
                }

            # ------------------------------------------------
            # Warmup
            # ------------------------------------------------

            self._warmup()

            self._initialized = True

            logger.info(
                "[YOLO] ✓ Model ready | frame_skip=%d | sahi=%dx%d | half=%s",
                self.config.frame_skip,
                self.config.sahi_rows,
                self.config.sahi_cols,
                self.config.half_precision,
            )

        except Exception as exc:

            logger.error(
                "[YOLO] Model loading error: %s",
                exc
            )

            logger.error(
                traceback.format_exc()
            )

            self._model = None
            self._initialized = False

            raise

    # ========================================================
    # WARMUP
    # ========================================================

    def _warmup(self):

        if self._model is None:
            raise RuntimeError(
                "The model has not been loaded before warmup."
            )

        dummy = np.zeros(
            (
                self.config.image_size,
                self.config.image_size,
                3
            ),
            dtype=np.uint8
        )

        start = time.perf_counter()

        self._run_single_inference(dummy)

        elapsed = time.perf_counter() - start

        logger.debug(
            "[YOLO] Warmup complete | %.1f ms",
            elapsed * 1000.0
        )

    # ========================================================
    # NUMPY CONVERSION
    # ========================================================

    @staticmethod
    def _to_numpy(value):

        if value is None:
            return None

        if isinstance(value, np.ndarray):
            return value

        try:

            if hasattr(value, "detach"):
                value = value.detach()

            if hasattr(value, "cpu"):
                value = value.cpu()

            if hasattr(value, "numpy"):
                return value.numpy()

        except Exception:
            pass

        try:
            return np.asarray(value)

        except Exception:
            return None

    # ========================================================
    # SINGLE INFERENCE
    #
    # Important:
    # Because the TensorRT engine uses batch=1, send
    # exactly one frame or slice at a time.
    # ========================================================

    def _run_single_inference(self, frame):

        if self._model is None:
            raise RuntimeError(
                "The YOLO model has not been loaded."
            )

        kwargs = {
            "imgsz": self.config.image_size,
            "conf": self.config.confidence_threshold,
            "iou": self.config.iou_threshold,
            "max_det": self.config.max_detections,
            "verbose": self.config.verbose,
        }

        if self.config.device is not None:
            kwargs["device"] = self.config.device

        if self.config.half_precision:
            kwargs["half"] = True

        if self.config.classes is not None:
            kwargs["classes"] = list(
                self.config.classes
            )

        return self._model.predict(
            source=frame,
            **kwargs
        )

    # ========================================================
    # LIVE FPS REPORT
    # ========================================================

    def _update_fps_report(self, processed: bool = False, inference_time: float = 0.0):
        now = time.perf_counter()

        self._fps_window_frames += 1

        if processed:
            self._fps_window_processed += 1
            self._fps_window_inference_time += max(0.0, inference_time)

        elapsed = now - self._fps_report_time

        if elapsed < 1.0:
            return

        camera_fps = self._fps_window_frames / elapsed
        processed_fps = self._fps_window_processed / elapsed

        if self._fps_window_inference_time > 0.0:
            avg_inference_ms = (
                self._fps_window_inference_time
                / max(1, self._fps_window_processed)
                * 1000.0
            )
            inference_fps = (
                self._fps_window_processed
                / elapsed
            )
        else:
            avg_inference_ms = 0.0
            inference_fps = 0.0

        print(
            f"[PERCEPTION FPS] "
            f"Camera={camera_fps:.1f} FPS | "
            f"Processed={processed_fps:.1f} FPS | "
            f"Inference={inference_fps:.1f} FPS | "
            f"AvgInference={avg_inference_ms:.1f} ms | "
            f"Skip={self.config.frame_skip}",
            flush=True,
        )

        self._fps_report_time = now
        self._fps_window_frames = 0
        self._fps_window_processed = 0
        self._fps_window_inference_time = 0.0

    # ========================================================
    # DETECT
    # ========================================================

    def detect(
        self,
        frame: np.ndarray,
        timestamp: Optional[float] = None
    ) -> List[Detection]:

        if frame is None:
            return []

        if not isinstance(frame, np.ndarray):
            return []

        if frame.size == 0:
            return []

        if self._model is None:
            return []

        self._statistics.total_frames += 1

        # Live camera FPS counter: count each frame before frame skipping.
        # Count every frame received by detect().
        self._update_fps_report(processed=False)

        # ----------------------------------------------------
        # Frame skip
        # ----------------------------------------------------

        if (
            self.config.frame_skip > 1
            and (
                self._statistics.total_frames
                - 1
            ) % self.config.frame_skip != 0
        ):
            return self._last_detections

        self._statistics.processed_frames += 1

        if timestamp is None:
            timestamp = time.time()

        frame_height, frame_width = frame.shape[:2]

        self._inference_counter += 1

        inference_id = self._inference_counter

        start_time = time.perf_counter()

        all_detections: List[Detection] = []

        try:

            # ------------------------------------------------
            # SAHI slices
            # ------------------------------------------------

            slices = self._create_slices(frame)

            # ------------------------------------------------
            # TensorRT batch=1:
            # Run inference separately for each slice.
            # ------------------------------------------------

            for slice_index, (
                slice_img,
                offset_x,
                offset_y
            ) in enumerate(slices):

                try:

                    results = self._run_single_inference(
                        slice_img
                    )

                    detections = (
                        self._parse_results_for_slice(
                            results,
                            slice_img,
                            inference_id,
                            offset_x=offset_x,
                            offset_y=offset_y,
                            timestamp=timestamp,
                            frame_width=frame_width,
                            frame_height=frame_height,
                        )
                    )

                    all_detections.extend(
                        detections
                    )

                except Exception as slice_exc:

                    logger.error(
                        "[YOLO] Inference error for slice %d: %s",
                        slice_index,
                        slice_exc
                    )

                    continue

            # ------------------------------------------------
            # NMS
            # ------------------------------------------------

            if self.config.use_nms:

                all_detections = self._apply_nms(
                    all_detections
                )

            # ------------------------------------------------
            # Statistics
            # ------------------------------------------------

            elapsed = time.perf_counter() - start_time

            self._statistics.last_inference_time = elapsed

            self._statistics.inference_time_total += elapsed

            # Include the processed frame in the live FPS report.
            self._update_fps_report(
                processed=True,
                inference_time=elapsed,
            )

            self._statistics.total_detections += len(
                all_detections
            )

            for detection in all_detections:

                if detection.class_id == 0:
                    self._statistics.mannequin_detections += 1

                elif detection.class_id == 1:
                    self._statistics.tent_detections += 1

            self._last_detections = all_detections

            return all_detections

        except Exception as exc:

            logger.error(
                "[YOLO] Detection error: %s",
                exc
            )

            logger.error(
                traceback.format_exc()
            )

            return self._last_detections

    # ========================================================
    # SAHI SLICE CREATION
    # ========================================================

    def _create_slices(
        self,
        frame: np.ndarray
    ) -> List[Tuple[np.ndarray, int, int]]:

        height, width = frame.shape[:2]

        rows = max(
            1,
            self.config.sahi_rows
        )

        cols = max(
            1,
            self.config.sahi_cols
        )

        # ----------------------------------------------------
        # Normal full-frame
        # ----------------------------------------------------

        if rows == 1 and cols == 1:

            return [
                (
                    frame,
                    0,
                    0
                )
            ]

        # ----------------------------------------------------
        # SAHI
        # ----------------------------------------------------

        slice_width = math.ceil(
            width / cols
        )

        slice_height = math.ceil(
            height / rows
        )

        slices = []

        for row in range(rows):

            for col in range(cols):

                x1 = col * slice_width
                y1 = row * slice_height

                x2 = min(
                    width,
                    x1 + slice_width
                )

                y2 = min(
                    height,
                    y1 + slice_height
                )

                if x2 <= x1 or y2 <= y1:
                    continue

                slice_img = frame[
                    y1:y2,
                    x1:x2
                ]

                slices.append(
                    (
                        slice_img,
                        x1,
                        y1
                    )
                )

        return slices

    # ========================================================
    # PARSE RESULTS
    # ========================================================

    def _parse_results_for_slice(
        self,
        results,
        slice_img: np.ndarray,
        inference_id: int,
        offset_x: int,
        offset_y: int,
        timestamp: float,
        frame_width: int,
        frame_height: int,
    ) -> List[Detection]:

        detections: List[Detection] = []

        if results is None:
            return detections

        # ----------------------------------------------------
        # Results list
        # ----------------------------------------------------

        if not isinstance(results, (list, tuple)):
            results = [results]

        for result in results:

            try:

                boxes = getattr(
                    result,
                    "boxes",
                    None
                )

                if boxes is None:
                    continue

                # ------------------------------------------------
                # Boxes data
                # ------------------------------------------------

                xyxy = self._to_numpy(
                    getattr(
                        boxes,
                        "xyxy",
                        None
                    )
                )

                conf = self._to_numpy(
                    getattr(
                        boxes,
                        "conf",
                        None
                    )
                )

                cls = self._to_numpy(
                    getattr(
                        boxes,
                        "cls",
                        None
                    )
                )

                if xyxy is None:
                    continue

                if conf is None:
                    continue

                if cls is None:
                    continue

                xyxy = np.asarray(
                    xyxy
                )

                conf = np.asarray(
                    conf
                ).reshape(-1)

                cls = np.asarray(
                    cls
                ).reshape(-1)

                if xyxy.ndim != 2:
                    continue

                if xyxy.shape[1] < 4:
                    continue

                count = min(
                    len(xyxy),
                    len(conf),
                    len(cls)
                )

                # ------------------------------------------------
                # Process each bounding box.
                # ------------------------------------------------

                for i in range(count):

                    try:

                        score = float(
                            conf[i]
                        )

                        class_id = int(
                            cls[i]
                        )

                        # ----------------------------------------
                        # Only the two configured model classes.
                        # ----------------------------------------

                        if class_id not in (
                            0,
                            1
                        ):
                            continue

                        # ----------------------------------------
                        # Confidence
                        # ----------------------------------------

                        if (
                            score
                            < self.config.confidence_threshold
                        ):
                            continue

                        x1 = float(
                            xyxy[i][0]
                        )

                        y1 = float(
                            xyxy[i][1]
                        )

                        x2 = float(
                            xyxy[i][2]
                        )

                        y2 = float(
                            xyxy[i][3]
                        )

                        # ----------------------------------------
                        # Check for NaN / Inf values.
                        # ----------------------------------------

                        values = [
                            x1,
                            y1,
                            x2,
                            y2,
                        ]

                        if not all(
                            np.isfinite(v)
                            for v in values
                        ):
                            continue

                        # ----------------------------------------
                        # Correct inverted coordinates.
                        # ----------------------------------------

                        if x2 < x1:
                            x1, x2 = x2, x1

                        if y2 < y1:
                            y1, y2 = y2, y1

                        # ----------------------------------------
                        # Clamp coordinates to slice boundaries.
                        # ----------------------------------------

                        slice_h, slice_w = (
                            slice_img.shape[:2]
                        )

                        x1 = max(
                            0.0,
                            min(
                                x1,
                                float(slice_w)
                            )
                        )

                        x2 = max(
                            0.0,
                            min(
                                x2,
                                float(slice_w)
                            )
                        )

                        y1 = max(
                            0.0,
                            min(
                                y1,
                                float(slice_h)
                            )
                        )

                        y2 = max(
                            0.0,
                            min(
                                y2,
                                float(slice_h)
                            )
                        )

                        # ----------------------------------------
                        # Invalid bounding box.
                        # ----------------------------------------

                        if x2 <= x1:
                            continue

                        if y2 <= y1:
                            continue

                        # ----------------------------------------
                        # Convert to global-frame coordinates.
                        # ----------------------------------------

                        global_x1 = x1 + offset_x
                        global_y1 = y1 + offset_y

                        global_x2 = x2 + offset_x
                        global_y2 = y2 + offset_y

                        # ----------------------------------------
                        # Full frame clamp
                        # ----------------------------------------

                        global_x1 = max(
                            0.0,
                            min(
                                global_x1,
                                float(frame_width)
                            )
                        )

                        global_x2 = max(
                            0.0,
                            min(
                                global_x2,
                                float(frame_width)
                            )
                        )

                        global_y1 = max(
                            0.0,
                            min(
                                global_y1,
                                float(frame_height)
                            )
                        )

                        global_y2 = max(
                            0.0,
                            min(
                                global_y2,
                                float(frame_height)
                            )
                        )

                        if global_x2 <= global_x1:
                            continue

                        if global_y2 <= global_y1:
                            continue

                        # ----------------------------------------
                        # Class name
                        # ----------------------------------------

                        class_name = self._class_names.get(
                            class_id,
                            "bilinmeyen"
                        )

                        # ----------------------------------------
                        # Detection
                        # ----------------------------------------

                        detection = Detection(

                            bbox=BoundingBox(
                                x1=global_x1,
                                y1=global_y1,
                                x2=global_x2,
                                y2=global_y2,
                            ),

                            confidence=score,

                            class_id=class_id,

                            class_name=class_name,

                            timestamp=timestamp,

                            inference_id=inference_id,
                        )

                        detections.append(
                            detection
                        )

                    except Exception:
                        continue

            except Exception as result_exc:

                logger.error(
                    "[YOLO] Result parsing error: %s",
                    result_exc
                )

                continue

        return detections

    # ========================================================
    # NMS
    # ========================================================

    def _apply_nms(
        self,
        detections: List[Detection]
    ) -> List[Detection]:

        if len(detections) <= 1:
            return detections

        final_detections = []

        # ----------------------------------------------------
        # NMS is applied per class.
        # The mannequin and canopy classes do not suppress each other.
        # ----------------------------------------------------

        class_ids = sorted(
            set(
                d.class_id
                for d in detections
            )
        )

        for class_id in class_ids:

            class_detections = [
                d
                for d in detections
                if d.class_id == class_id
            ]

            class_detections.sort(
                key=lambda d: d.confidence,
                reverse=True
            )

            selected = []

            while class_detections:

                best = class_detections.pop(0)

                selected.append(best)

                remaining = []

                for candidate in class_detections:

                    iou = self._calculate_iou(
                        best.bbox,
                        candidate.bbox
                    )

                    if (
                        iou
                        <= self.config.iou_threshold
                    ):
                        remaining.append(
                            candidate
                        )

                class_detections = remaining

            final_detections.extend(
                selected
            )

        # ----------------------------------------------------
        # Sort by confidence.
        # ----------------------------------------------------

        final_detections.sort(
            key=lambda d: d.confidence,
            reverse=True
        )

        return final_detections

    # ========================================================
    # IOU
    # ========================================================

    @staticmethod
    def _calculate_iou(
        box_a: BoundingBox,
        box_b: BoundingBox
    ) -> float:

        x1 = max(
            box_a.x1,
            box_b.x1
        )

        y1 = max(
            box_a.y1,
            box_b.y1
        )

        x2 = min(
            box_a.x2,
            box_b.x2
        )

        y2 = min(
            box_a.y2,
            box_b.y2
        )

        intersection_width = max(
            0.0,
            x2 - x1
        )

        intersection_height = max(
            0.0,
            y2 - y1
        )

        intersection = (
            intersection_width
            * intersection_height
        )

        area_a = box_a.area
        area_b = box_b.area

        union = (
            area_a
            + area_b
            - intersection
        )

        if union <= 0:
            return 0.0

        return intersection / union

    # ========================================================
    # STATISTICS
    # ========================================================

    def get_statistics(self) -> YOLOStatistics:
        return self._statistics

    # ========================================================
    # RESET STATISTICS
    # ========================================================

    def reset_statistics(self):

        self._statistics = YOLOStatistics()

        self._fps_report_time = time.perf_counter()
        self._fps_window_frames = 0
        self._fps_window_processed = 0
        self._fps_window_inference_time = 0.0

    # ========================================================
    # LAST DETECTIONS
    # ========================================================

    def get_last_detections(
        self
    ) -> List[Detection]:

        return self._last_detections

    # ========================================================
    # CLASS NAMES
    # ========================================================

    def get_class_name(
        self,
        class_id: int
    ) -> str:

        return self._class_names.get(
            int(class_id),
            "bilinmeyen"
        )

    # ========================================================
    # CLOSE
    # ========================================================

    def close(self):

        self._model = None

        self._initialized = False

        self._last_detections = []

        logger.info(
            "[YOLO] Model closed."
        )


# ============================================================
# FACTORY
# ============================================================

def create_yolo_detector(
    model_path: str,
    confidence_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    device=None,
    half_precision: bool = True,
    sahi_rows: int = 1,
    sahi_cols: int = 1,
    frame_skip: int = 1,
    classes: Optional[List[int]] = None,
) -> YOLODetector:

    config = YOLOConfig(

        model_path=model_path,

        # TensorRT engine static 1024
        image_size=1024,

        confidence_threshold=confidence_threshold,

        iou_threshold=iou_threshold,

        device=device,

        half_precision=half_precision,

        sahi_rows=sahi_rows,

        sahi_cols=sahi_cols,

        frame_skip=frame_skip,

        classes=classes,

        use_nms=True,
    )

    return YOLODetector(config)

