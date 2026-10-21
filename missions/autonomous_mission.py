"""Coordinates the autonomous mission sequence: takeoff, lap route, mapping, target delivery, return-to-home, and landing."""

from __future__ import annotations

import glob
import logging
import math
import os
import queue
import subprocess
import threading
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from pymavlink import mavutil

from config import (
    TAKEOFF_ALTITUDE_M,
    LAP_COUNT,
    LAP_SPEED_MPS,
    LAP_WAYPOINTS,
    MAPPING_POLYGON,
    MAPPING_ALTITUDE_M,
    MAPPING_HORIZONTAL_FOV_DEG,
    MAPPING_VERTICAL_FOV_DEG,
    MAPPING_FORWARD_OVERLAP_RATIO,
    MAPPING_SIDE_OVERLAP_RATIO,
    MAPPING_SPEED_MPS,
    YOLO_MODEL_PATH,
    YOLO_ENABLED_DURING_MAPPING,
    YOLO_CONFIDENCE_THRESHOLD,
    YOLO_IOU_THRESHOLD,
    YOLO_HALF_PRECISION,
    USE_CUDA,
    CAMERA_SOURCE,
    DISPLAY_CAMERA_PREVIEW,
    GSTREAMER_FRAME_SAVE_DIR,
    GSTREAMER_SAVE_ALL_FRAMES,
    GSTREAMER_JPEG_QUALITY,
    TARGET_CLASS_NAMES,
    TARGET_CLASS_IDS,
    AUTO_ASSIGN_TARGET_CLASSES,
    PAYLOAD_TARGET_CHANNELS,
    PAYLOAD_RESET_PWM,
    PAYLOAD_OPEN_PWM,
    PAYLOAD_SEQUENCE_DELAY_S,
    PAYLOAD_OPEN_HOLD_S,
    PAYLOAD_DELAY_S,
    TARGET_CENTER_TOLERANCE_NORMALIZED,
    OBJECT_DETECTION_INTERVAL_S,
    HOME_ACCEPT_RADIUS_M,
    GUIDED_SPEED_MPS,
    TARGET_CENTERING_MAX_SPEED_MPS,
    TARGET_CENTERING_KP_X,
    TARGET_CENTERING_KP_Y,
    TARGET_CENTERING_DEADBAND,
    RETURN_HOME_ALTITUDE_M,
    TARGET_DELIVERY_CONFIDENCE_THRESHOLD,
    TARGET_CENTERING_TIMEOUT_S,
    CENTERING_STABLE_TIME_S,
    TARGET_REQUIRED_CONSECUTIVE_DETECTIONS,
    YOLO_CENTER_ZOOM,
    YOLO_SAHI_ROWS,
    YOLO_SAHI_COLS,
    YOLO_FRAME_SKIP,
    DELIVERY_APPROACH_ALTITUDE_M,
    DELIVERY_DROP_ALTITUDE_M,
    DELIVERY_ASCEND_ALTITUDE_M,
    DELIVERY_HOLD_S,
    SKIP_UNCONFIRMED_TARGETS,
    FALLBACK_DROP_ENABLED,
    MANNEQUIN_SERVO_CHANNEL_1,
    MANNEQUIN_SERVO_CHANNEL_2,
    CANOPY_SERVO_CHANNEL_1,
    CANOPY_SERVO_CHANNEL_2,
    CENTERING_SERVO_RESET_PWM,
    SIYI_A8_MINI_ENABLED,
    SIYI_A8_MINI_GIMBAL_PITCH_DEG,
    SIYI_A8_MINI_GIMBAL_YAW_DEG,
    SIYI_A8_MINI_GIMBAL_SETTLE_S,
    PHOTO_CAPTURE_ENABLED,
    PHOTO_SAVE_PATH,
    PHOTO_JPEG_QUALITY,
    PHOTO_MIN_SPACING_M,
    PHOTO_MAX_SPACING_M,
    PHOTO_METADATA_ENABLED,
    FLASH_PHOTO_COPY_ENABLED,
    FLASH_PHOTO_COPY_RETRY,
    FLASH_PHOTO_COPY_VERIFY,
    FLASH_PHOTO_COPY_WAIT_S,
    FLASH_COPY_SOURCE_DIR,
    FLASH_COPY_DEST_DIR,
)

from core.events import EventType
from mapping.mapping_grid import MappingGrid, MappingGridConfig
from mavlink.mission_protocol import MissionItem
from navigation.distance import distance
from navigation.target_centering import (
    TargetCenteringConfig,
    TargetCenteringController,
    activate_mannequin_centering_servos,
    activate_canopy_centering_servos,
)
from perception.yolo import YOLOConfig, YOLODetector

logger = logging.getLogger(__name__)

Coordinate = Tuple[float, float]


class AutonomousState(Enum):
    IDLE = auto()
    WAITING_HOME = auto()
    LAP_AUTO = auto()
    MAPPING_AUTO = auto()
    DELIVERY = auto()
    GUIDED_TO_TARGET = auto()
    CENTERING = auto()
    FALLBACK_CENTER = auto()
    RETURN_HOME = auto()
    LANDING = auto()
    FINISHED = auto()
    FAILED = auto()
    STOPPED = auto()


@dataclass
class TargetObservation:
    class_name: str
    class_id: Optional[int]
    best_error: float = 999.0
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    confidence: float = 0.0
    seen: bool = False
    consecutive_detections: int = 0
    confirmed: bool = False


class AutonomousMission:

    name = "autonomous_mission"

    def __init__(self, shared_data: Any, mavlink: Any, servo_manager: Any) -> None:
        self.shared = shared_data
        self.mavlink = mavlink
        self.servo_manager = servo_manager
        self.events = shared_data.events

        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._camera_thread: Optional[threading.Thread] = None

        self._state = AutonomousState.IDLE
        self._phase = "idle"

        self._home: Optional[Coordinate] = None
        self._current_position: Optional[Coordinate] = None
        self._current_altitude = 0.0

        self._lap_first_seq = 0
        self._lap_speed_seq = 0
        self._lap_final_seq = 0
        self._mapping_speed_seq = 0
        self._mapping_start_seq = 0
        self._mapping_first_wp_seq = 0
        self._mapping_final_seq = 0
        self._last_mission_reached_seq: Optional[int] = None
        self._current_mission_seq: Optional[int] = None

        self._mapping_waypoints: List[Tuple[float, float]] = []
        self._mapping_polygon = list(MAPPING_POLYGON)

        self._targets: Dict[str, TargetObservation] = {}
        self._delivery_queue: List[str] = []
        self._delivery_index = 0
        self._current_delivery_target: Optional[str] = None
        self._target_goto_started = 0.0
        self._center_stable_since: Optional[float] = None
        self._last_detection_time = 0.0
        self._latest_detections = []
        self._latest_detection_time = 0.0
        self._last_processed_detection_time = 0.0
        self._pre_delivery_position: Optional[Coordinate] = None

        self._detector: Optional[YOLODetector] = None
        self._camera = None
        self._camera_running = False

        # ── Asenkron YOLO inference ──────────────────────────────────────
        # The camera loop enqueues the frame and immediately continues.
        # _yolo_worker_thread consumes the queue and stores results in _latest_detections.
        # maxsize=1 prevents a backlog; the most recent frame is processed.
        self._yolo_queue: queue.Queue = queue.Queue(maxsize=1)
        self._yolo_thread: Optional[threading.Thread] = None

        self._photo_saver = None
        self._flash_copier = None   # FlashPhotoCopier starts after mapping.

        self._a8mini_recorder = None
        if SIYI_A8_MINI_ENABLED:
            try:
                from perception.a8mini_recorder import A8MiniRecorder
                self._a8mini_recorder = A8MiniRecorder()
                logger.info("[AutonomousMission] A8 Mini recorder ready")
            except Exception:
                logger.exception("[AutonomousMission] Could not load the A8 Mini recorder")

        self._centering = TargetCenteringController(
            mavlink,
            TargetCenteringConfig(
                kp_x=float(TARGET_CENTERING_KP_X),
                kp_y=float(TARGET_CENTERING_KP_Y),
                max_velocity_mps=float(TARGET_CENTERING_MAX_SPEED_MPS),
                deadband=float(TARGET_CENTERING_DEADBAND),
            ),
        )

        self._subscriptions: List[int] = []

        self._build_target_table()
        self._subscribe_mavlink_messages()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> bool:
        if self._thread and self._thread.is_alive():
            return True

        self._stop.clear()
        self._state = AutonomousState.WAITING_HOME
        self._phase = "waiting_home"

        self._thread = threading.Thread(
            target=self._run,
            name="AutonomousMission",
            daemon=True,
        )
        self._thread.start()
        return True

    def stop(self, timeout_s: float = 5.0) -> bool:
        self._stop.set()
        self._stop_camera()
        self._centering.stop()

        if self._photo_saver is not None and self._photo_saver.is_running:
            try:
                self._photo_saver.stop()
            except Exception:
                logger.exception("[AutonomousMission] Could not stop PhotoSaver")

        if self._a8mini_recorder is not None:
            try:
                self._a8mini_recorder.stop_recording()
            except Exception:
                logger.exception("[AutonomousMission] Could not stop A8 Mini recording")

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=max(0.1, timeout_s))

        if self._camera_thread and self._camera_thread.is_alive():
            self._camera_thread.join(timeout=1.0)

        # If copying is active, request a graceful stop and wait briefly.
        if self._flash_copier is not None and self._flash_copier.is_running:
            self._flash_copier.stop()
            self._flash_copier.join(timeout_s=10.0)

        for sid in self._subscriptions:
            try:
                self.mavlink.message_router.unsubscribe(sid)
            except Exception:
                pass
        self._subscriptions.clear()

        self._state = AutonomousState.STOPPED
        return True

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _build_target_table(self) -> None:
        names = list(TARGET_CLASS_NAMES)
        ids = list(TARGET_CLASS_IDS)

        for index, name in enumerate(names):
            class_id = ids[index] if index < len(ids) else None
            self._targets[str(name)] = TargetObservation(
                class_name=str(name),
                class_id=class_id,
            )

    def _subscribe_mavlink_messages(self) -> None:
        router = getattr(self.mavlink, "message_router", None)
        if router is None:
            return

        self._subscriptions.append(
            router.subscribe(
                "GLOBAL_POSITION_INT",
                self._on_global_position,
                owner=self.name,
            )
        )
        self._subscriptions.append(
            router.subscribe(
                "MISSION_ITEM_REACHED",
                self._on_mission_item_reached,
                owner=self.name,
            )
        )
        self._subscriptions.append(
            router.subscribe(
                "MISSION_CURRENT",
                self._on_mission_current,
                owner=self.name,
            )
        )

    # ------------------------------------------------------------------
    # MAVLink callbacks
    # ------------------------------------------------------------------

    def _on_global_position(self, message: Any) -> None:
        try:
            lat_raw = float(message.lat)
            lon_raw = float(message.lon)
            alt_raw = float(message.relative_alt)

            lat = lat_raw / 1e7
            lon = lon_raw / 1e7
            alt = alt_raw / 1000.0

        except Exception:
            logger.exception(
                "[GPS] GLOBAL_POSITION_INT parse edilemedi | message=%r",
                message,
            )
            return

        if abs(lat) < 1e-8 and abs(lon) < 1e-8:
            logger.warning(
                "[GPS] GLOBAL_POSITION_INT received with invalid coordinates: "
                "lat=%s lon=%s",
                lat,
                lon,
            )
            return

        with self._lock:
            self._current_position = (lat, lon)
            self._current_altitude = alt

    def _on_mission_item_reached(self, message: Any) -> None:
        try:
            seq = int(message.seq)
        except Exception:
            return

        with self._lock:
            self._last_mission_reached_seq = seq

    def _on_mission_current(self, message: Any) -> None:
        try:
            seq = int(message.seq)
        except Exception:
            return

        with self._lock:
            self._current_mission_seq = seq

    # ------------------------------------------------------------------
    # Main state machine
    # ------------------------------------------------------------------

    def _run(self) -> None:
        try:
            logger.info(
                "[MISSION] Preparing | takeoff=%.1f m | laps=%d @ %.1f m/s | "
                "Mapping=%.1f m @ %.1f m/s",
                float(TAKEOFF_ALTITUDE_M), int(LAP_COUNT), float(LAP_SPEED_MPS),
                float(MAPPING_ALTITUDE_M), float(MAPPING_SPEED_MPS),
            )
            logger.info(
                "[CONFIG] Camera=%s | targets=%s | detection>=%.0f%% | "
                "drop=%.1f m | ascent=%.1f m | hold=%.1f s",
                str(CAMERA_SOURCE), ", ".join(TARGET_CLASS_NAMES),
                float(TARGET_DELIVERY_CONFIDENCE_THRESHOLD) * 100.0,
                float(DELIVERY_DROP_ALTITUDE_M), float(DELIVERY_ASCEND_ALTITUDE_M),
                float(DELIVERY_HOLD_S),
            )

            # Initialize servo outputs CH5-CH8 to 1000 PWM.
            _servo_init_channels = [
                MANNEQUIN_SERVO_CHANNEL_1,
                MANNEQUIN_SERVO_CHANNEL_2,
                CANOPY_SERVO_CHANNEL_1,
                CANOPY_SERVO_CHANNEL_2,
            ]
            for _ch in _servo_init_channels:
                try:
                    self.mavlink.send_servo_command(_ch, int(CENTERING_SERVO_RESET_PWM))
                    logger.info(
                        "[SERVO] Initializing output: channel %d -> %d PWM",
                        _ch, int(CENTERING_SERVO_RESET_PWM),
                    )
                except Exception as _exc:
                    logger.warning(
                        "[SERVO] Initial output setup failed on channel %d: %s", _ch, _exc
                    )

            self._wait_for_home()
            if self._stop.is_set():
                return

            if not self._start_combined_auto():
                raise RuntimeError("Combined AUTO mission could not be started")

            # Skip the lap phase entirely when LAP_COUNT=0.
            # After TAKEOFF in the AUTO mission, continue directly to mapping
            # waypoints without calling the lap-waiting function.
            if int(LAP_COUNT) > 0:
                self._phase = "lap_auto"
                self._state = AutonomousState.LAP_AUTO
                self._wait_for_lap_completion()
                if self._stop.is_set():
                    return
            else:
                logger.info(
                    "[MISSION] Lap count is 0 → skipping the lap phase; "
                    "proceeding directly to mapping waypoints"
                )

            self._phase = "mapping_auto"
            self._state = AutonomousState.MAPPING_AUTO
            if YOLO_ENABLED_DURING_MAPPING:
                logger.info(
                    "[MISSION] Mapping started | loading YOLO model: %s",
                    YOLO_MODEL_PATH,
                )
                self._start_camera()
            else:
                self._start_camera()
            self._mapping_loop()
            if self._stop.is_set():
                return

            # Stop GPS/EXIF photo capture as soon as mapping finishes.
            # Keep the camera/YOLO pipeline running because target centering
            # may still be needed during delivery.
            self._stop_mapping_photo_capture()

            # After mapping, enter a safe GUIDED hover before payload delivery.
            self._enter_delivery_hover()
            self._prepare_delivery_queue()
            self._delivery_loop()
            if self._stop.is_set():
                return

            with self._lock:
                current_alt = self._current_altitude
                already_finished = self._state == AutonomousState.FINISHED
            if not already_finished and current_alt > 1.0 and not self._stop.is_set():
                self._return_home_and_land()

            if not self._stop.is_set():
                self._state = AutonomousState.FINISHED
                self._phase = "finished"
        except Exception:
            self._state = AutonomousState.FAILED
            self._phase = "failed"
            logger.exception("[AutonomousMission] Mission failed")
            # RTL fallback is disabled; only ARM failure triggers RTL.
            # Other errors do not trigger RTL automatically.

    # ------------------------------------------------------------------
    # Home / lap
    # ------------------------------------------------------------------

    def _wait_for_home(self) -> None:
        deadline = time.monotonic() + 60.0

        while not self._stop.is_set() and time.monotonic() < deadline:
            with self._lock:
                pos = self._current_position

            if pos is not None:
                self._home = pos
                return

            time.sleep(0.1)

        if self._stop.is_set():
            return
        raise RuntimeError("No valid GPS position for home")

    def _build_combined_mission(self) -> List[MissionItem]:
        if not LAP_WAYPOINTS:
            raise RuntimeError("LAP_WAYPOINTS is empty")
        if not self._mapping_waypoints:
            raise RuntimeError("Mapping grid is empty")
        if self._home is None:
            raise RuntimeError("HOME is not available")

        items: List[MissionItem] = []
        seq = 0

        # --------------------------------------------------------------
        # 0) NAV_TAKEOFF — takeoff within the AUTO mission.
        # --------------------------------------------------------------
        items.append(
            MissionItem(
                latitude=float(self._home[0]),
                longitude=float(self._home[1]),
                altitude_m=float(TAKEOFF_ALTITUDE_M),
                seq=seq,
                command=mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
                frame=mavutil.mavlink.MAV_FRAME_GLOBAL_INT,
                current=1,
                autocontinue=1,
                param1=0.0,
                param2=0.0,
                param3=0.0,
                param4=float("nan"),  # yaw: NaN keeps the current heading.
            )
        )
        seq += 1


        items.append(
            MissionItem(
                latitude=float(self._home[0]),
                longitude=float(self._home[1]),
                altitude_m=float(TAKEOFF_ALTITUDE_M),
                seq=seq,
                command=mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
                frame=mavutil.mavlink.MAV_FRAME_GLOBAL_INT,
                current=1,
                autocontinue=1,
                param1=0.0,
                param2=0.0,
                param3=0.0,
                param4=float("nan"),  # yaw: NaN keeps the current heading.
            )
        )
        seq += 1


        # --------------------------------------------------------------
        # 1) LAP SPEED COMMAND
        # --------------------------------------------------------------
        self._lap_speed_seq = seq
        items.append(
            MissionItem(
                latitude=float(self._home[0]),
                longitude=float(self._home[1]),
                altitude_m=float(TAKEOFF_ALTITUDE_M),
                seq=seq,
                command=mavutil.mavlink.MAV_CMD_DO_CHANGE_SPEED,
                frame=mavutil.mavlink.MAV_FRAME_GLOBAL_INT,
                current=0,
                autocontinue=1,
                param1=1.0,
                param2=float(LAP_SPEED_MPS),
                param3=-1.0,
                param4=0.0,
            )
        )
        seq += 1

        # --------------------------------------------------------------
        # 2) LAP
        # --------------------------------------------------------------
        def _distance_sq(lat: float, lon: float) -> float:
            dlat = float(lat) - float(self._home[0])
            dlon = float(lon) - float(self._home[1])
            return dlat * dlat + dlon * dlon

        # Use LAP waypoints in exactly the order specified in config.
        # Do not reorder the route based on proximity to HOME.
        lap_points = list(LAP_WAYPOINTS)

        self._lap_first_seq = seq

        # When LAP_COUNT=0, no lap waypoints are added. After takeoff,
        # the aircraft proceeds directly to the mapping phase.
        if int(LAP_COUNT) > 0:
            first_lap_lat, first_lap_lon = lap_points[0]
            for _lap_index in range(int(LAP_COUNT)):
                # Each lap visits every waypoint...
                for lat, lon in lap_points:
                    items.append(
                        MissionItem(
                            latitude=float(lat),
                            longitude=float(lon),
                            altitude_m=float(TAKEOFF_ALTITUDE_M),
                            seq=seq,
                            command=mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
                            frame=mavutil.mavlink.MAV_FRAME_GLOBAL_INT,
                            current=0,
                            autocontinue=1,
                            param1=0.0,
                        )
                    )
                    seq += 1
                # ...then returns to the first point to complete the lap.
                items.append(
                    MissionItem(
                        latitude=float(first_lap_lat),
                        longitude=float(first_lap_lon),
                        altitude_m=float(TAKEOFF_ALTITUDE_M),
                        seq=seq,
                        command=mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
                        frame=mavutil.mavlink.MAV_FRAME_GLOBAL_INT,
                        current=0,
                        autocontinue=1,
                        param1=0.0,
                    )
                )
                seq += 1

            self._lap_final_seq = seq - 1
        else:
            # Takeoff items use sequence numbers 0 and 1, so when laps are skipped,
            # is treated as a second takeoff; _wait_for_lap_completion()
            # the mission proceeds directly to mapping without an extra wait.
            self._lap_final_seq = seq - 1
            logger.info(
                "[MISSION] Lap count is 0 → skipping lap phase and starting mapping directly"
            )

        self._mapping_speed_seq = seq

        # --------------------------------------------------------------
        # 3) MAPPING SPEED COMMAND
        # --------------------------------------------------------------
        items.append(
            MissionItem(
                latitude=float(self._home[0]),
                longitude=float(self._home[1]),
                altitude_m=float(TAKEOFF_ALTITUDE_M),
                seq=seq,
                command=mavutil.mavlink.MAV_CMD_DO_CHANGE_SPEED,
                frame=mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                current=0,
                autocontinue=1,
                param1=1.0,
                param2=float(MAPPING_SPEED_MPS),
                param3=-1.0,
                param4=0.0,
            )
        )
        seq += 1

        # --------------------------------------------------------------
        # 4) MAPPING ALTITUDE TRANSITION
        # --------------------------------------------------------------
        first_mapping_transition_lat, first_mapping_transition_lon = self._mapping_waypoints[0]
        self._mapping_start_seq = seq
        items.append(
            MissionItem(
                latitude=float(first_mapping_transition_lat),
                longitude=float(first_mapping_transition_lon),
                altitude_m=float(MAPPING_ALTITUDE_M),
                seq=seq,
                command=mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
                frame=mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                current=0,
                autocontinue=1,
            )
        )
        seq += 1
        self._mapping_first_wp_seq = seq

        # --------------------------------------------------------------
        # 5) MAPPING GRID
        # --------------------------------------------------------------
        for lat, lon in self._mapping_waypoints:
            items.append(
                MissionItem(
                    latitude=float(lat),
                    longitude=float(lon),
                    altitude_m=float(MAPPING_ALTITUDE_M),
                    seq=seq,
                    command=mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
                    frame=mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                    current=0,
                    autocontinue=1,
                )
            )
            seq += 1

        self._mapping_final_seq = seq - 1

        logger.info(
            "[MISSION] Mission ready | AUTO takeoff via NAV_TAKEOFF | "
            "laps=%d at %.1f m/s | mapping_waypoints=%d at %.1f m/s",
            int(LAP_COUNT), float(LAP_SPEED_MPS),
            len(self._mapping_waypoints), float(MAPPING_SPEED_MPS),
        )
        return items

    def _prepare_mapping_grid(self) -> None:
        if not self._mapping_polygon:
            raise RuntimeError("MAPPING_POLYGON is empty")

        # Use the first point of the lap route rather than HOME
        # as the reference for nearest-edge detection.
        if LAP_WAYPOINTS:
            start_point = (float(LAP_WAYPOINTS[0][0]), float(LAP_WAYPOINTS[0][1]))
        else:
            start_point = self._current_position

        grid = MappingGrid(
            self._mapping_polygon,
            MappingGridConfig(
                altitude_m=float(MAPPING_ALTITUDE_M),
                hfov_deg=float(MAPPING_HORIZONTAL_FOV_DEG),
                vfov_deg=float(MAPPING_VERTICAL_FOV_DEG),
                forward_overlap=float(MAPPING_FORWARD_OVERLAP_RATIO),
                side_overlap=float(MAPPING_SIDE_OVERLAP_RATIO),
                ground_speed_mps=float(MAPPING_SPEED_MPS),
                camera_trigger=True,
                acceptance_radius_m=2.0,
            ),
        )

        waypoints = grid.generate_waypoints(start_point=start_point)
        self._mapping_waypoints = [
            (float(w.latitude), float(w.longitude))
            for w in waypoints
        ]

        if not self._mapping_waypoints:
            raise RuntimeError("Mapping grid generated no waypoints")

        if PHOTO_CAPTURE_ENABLED:
            try:
                from perception.photo_saver import PhotoSaver, PhotoSaverConfig

                self._photo_saver = PhotoSaver.from_camera_config(
                    altitude_m=float(MAPPING_ALTITUDE_M),
                    vfov_deg=float(MAPPING_VERTICAL_FOV_DEG),
                    forward_overlap=float(MAPPING_FORWARD_OVERLAP_RATIO),
                    mapping_polygon=tuple(
                        (float(lat), float(lon))
                        for lat, lon in self._mapping_polygon
                    ),
                    min_spacing_m=float(PHOTO_MIN_SPACING_M),
                    max_spacing_m=float(PHOTO_MAX_SPACING_M),
                    output_dir=str(PHOTO_SAVE_PATH),
                    jpeg_quality=int(PHOTO_JPEG_QUALITY),
                    write_metadata=bool(PHOTO_METADATA_ENABLED),
                )
                logger.info(
                    "[AutonomousMission] PhotoSaver ready | spacing=%.1f m "
                    "| area check=ENABLED | polygon=%d points "
                    "(MappingGrid spacing=%.1fm)",
                    self._photo_saver.spacing_m,
                    len(self._mapping_polygon),
                    grid.photo_spacing_m,
                )
            except Exception:
                logger.exception("[AutonomousMission] Could not create PhotoSaver")

    def _sleep(self, seconds: float) -> bool:
        deadline = time.monotonic() + seconds
        while not self._stop.is_set() and time.monotonic() < deadline:
            time.sleep(min(0.05, deadline - time.monotonic()))
        return not self._stop.is_set()

    def _start_combined_auto(self) -> bool:
        # ------------------------------------------------------------------
        # 1) Prepare the grid and mission.
        # ------------------------------------------------------------------
        self._prepare_mapping_grid()
        items = self._build_combined_mission()

        # ------------------------------------------------------------------
        # 2) Upload the mission.
        # ------------------------------------------------------------------
        logger.info("[MISSION] Uploading mission to vehicle: %d items", len(items))
        result = self.mavlink.mission_protocol.upload(items)
        if not result.success:
            logger.warning(
                "[AutonomousMission] Mission upload failed: %s; "
                "retrying once...",
                result.message,
            )
            time.sleep(0.5)
            result = self.mavlink.mission_protocol.upload(items)

        if not result.success:
            logger.error(
                "[AutonomousMission] Mission upload failed again on the second attempt: %s",
                result.message,
            )
            return False

        logger.info("[MISSION] Mission uploaded successfully")

        # ------------------------------------------------------------------
        # 3) Set mission current to sequence 0 (first NAV_TAKEOFF).
        # ------------------------------------------------------------------
        set_result = self.mavlink.mission_protocol.set_current(0)
        if hasattr(set_result, "success"):
            if not set_result.success:
                logger.error(
                    "[AutonomousMission] Could not set mission current sequence to 0: %s",
                    getattr(set_result, "message", ""),
                )
                return False
        elif not set_result:
            return False

        # ------------------------------------------------------------------
        # 4) Set WPNAV_SPEED to the lap speed.
        # ------------------------------------------------------------------
        initial_speed_mps = float(LAP_SPEED_MPS) if int(LAP_COUNT) > 0 else float(MAPPING_SPEED_MPS)
        initial_speed_cm_s = initial_speed_mps * 100.0
        if not self.mavlink.set_parameter("WPNAV_SPEED", initial_speed_cm_s):
            logger.error("[AutonomousMission] Could not set WPNAV_SPEED during initialization")
            return False
        # ------------------------------------------------------------------
        # 5) GUIDED moda al
        # ------------------------------------------------------------------
        logger.info("[MISSION] Switching to GUIDED mode (arming preparation)...")
        guided_ok = False
        for attempt in range(1, 4):
            if self.mavlink.set_mode("GUIDED"):
                guided_ok = True
                break
            logger.warning(
                "[AutonomousMission] GUIDED mode attempt %d failed; retrying...",
                attempt,
            )
            time.sleep(1.0)

        if not guided_ok:
            logger.error("[AutonomousMission] Could not switch to GUIDED mode")
            return False

        logger.info("[MISSION] GUIDED mode active")
        self._log_vehicle_state()

        # ArduPilot 4.6.x: wait briefly after GUIDED mode is confirmed.
        if not self._sleep(1.0):
            return False

        # ------------------------------------------------------------------
        # 6) ARM
        # ------------------------------------------------------------------
        logger.info("[MISSION] Arming...")
        if not self.mavlink.arm():
            logger.error("[AutonomousMission] ARM failed")
            return False

        logger.info("[MISSION] Arming confirmed")

        # ArduPilot 4.6.x: wait for motor and state initialization after ARM.
        if not self._sleep(3.0):
            return False

        # ------------------------------------------------------------------
        # 7) Switch to AUTO mode.
        # ------------------------------------------------------------------
        logger.info("[MISSION] Switching to AUTO mode...")

        auto_confirmed = False
        for attempt in range(1, 6):
            if self.mavlink.set_mode("AUTO"):
                auto_confirmed = True
                break
            logger.warning(
                "[AutonomousMission] AUTO mode attempt %d failed; retrying in 1.5 s...",
                attempt,
            )
            if not self._sleep(1.5):
                return False

        if not auto_confirmed:
            logger.error(
                "[AutonomousMission] AUTO mode could not be confirmed after 5 attempts"
            )
            return False

        logger.info("[MISSION] AUTO mode active")

        # ------------------------------------------------------------------
        # 8) MAV_CMD_MISSION_START — required for AUTO takeoff on ArduPilot 4.6.x.
        # ------------------------------------------------------------------
        if not self._sleep(0.5):
            return False

        try:
            MISSION_START_CMD = getattr(
                mavutil.mavlink, "MAV_CMD_MISSION_START", 300
            )

            # MAVLinkManager exposes the connection object through the `master` property.
            mav_master = self.mavlink.master
            if mav_master is None:
                logger.error("[AutonomousMission] MAVLink master connection is None")
                return False

            mav_master.mav.command_long_send(
                self.mavlink.target_system,
                self.mavlink.target_component,
                MISSION_START_CMD,
                0,    # confirmation
                0,    # param1: first_item
                0,    # param2: last_item
                0, 0, 0, 0, 0,
            )
            logger.info(
                "[MISSION] MISSION_START command sent (command=%d)", MISSION_START_CMD
            )

        except Exception:
            logger.exception("[AutonomousMission] Could not send MISSION_START")
            return False

        # ------------------------------------------------------------------
        # 9) Wait for NAV_TAKEOFF to complete.
        # ------------------------------------------------------------------
        target_alt = float(TAKEOFF_ALTITUDE_M)
        climb_deadline = time.monotonic() + 900.0
        while not self._stop.is_set() and time.monotonic() < climb_deadline:
            with self._lock:
                alt = float(self._current_altitude)
            if alt >= 10 * 0.95:
                logger.info(
                    "[MISSION] AUTO takeoff complete: %.1f m | %s",
                    alt, "LAP phase starting" if int(LAP_COUNT) > 0 else "proceeding directly to mapping",
                )
                break
            time.sleep(0.2)
        else:
            if self._stop.is_set():
                return False
            logger.error(
                "[AutonomousMission] NAV_TAKEOFF did not reach the target altitude "
                "(target=%.1f m, timeout=900 s)",
                target_alt,
            )
            return False

        if self._stop.is_set():
            return False

        if int(LAP_COUNT) > 0:
            logger.info(
                "[MISSION] AUTO takeoff and lap phase started: %d laps at %.1f m/s",
                int(LAP_COUNT), float(LAP_SPEED_MPS),
            )
        else:
            logger.info(
                "[MISSION] AUTO takeoff complete → no laps configured; mapping starts next"
            )
        return True

    def _log_vehicle_state(self) -> None:
        pass

    def _wait_for_lap_completion(self) -> None:
        last_reported = None
        last_log_time = time.monotonic()

        while not self._stop.is_set():
            with self._lock:
                reached = self._last_mission_reached_seq
                current = self._current_mission_seq

            progress = reached if reached is not None else current

            now = time.monotonic()
            if progress != last_reported or (now - last_log_time) > 5.0:
                last_reported = progress
                last_log_time = now

            if (reached is not None and reached >= self._lap_final_seq) or (
                current is not None and current >= self._lap_final_seq
            ):
                mapping_speed_cm_s = float(MAPPING_SPEED_MPS) * 100.0
                if not self.mavlink.set_parameter("WPNAV_SPEED", mapping_speed_cm_s):
                    raise RuntimeError("Could not set WPNAV_SPEED for mapping")

                return

            time.sleep(0.1)

    # ------------------------------------------------------------------
    # Mapping
    # ------------------------------------------------------------------

    def _mapping_loop(self) -> None:
        logger.info(
            "[MISSION] Lap complete | Mapping started: %.1f m at %.1f m/s",
            float(MAPPING_ALTITUDE_M), float(MAPPING_SPEED_MPS),
        )

        photo_saver = self._photo_saver
        if photo_saver is not None:
            photo_saver.start()
            logger.info(
                "[MISSION] Photo capture active | spacing=%.1f m | directory=%s",
                photo_saver.spacing_m, PHOTO_SAVE_PATH,
            )

        try:
            while not self._stop.is_set():
                self._detect_mapping_targets()

                if photo_saver is not None and photo_saver.is_running:
                    with self._lock:
                        pos = self._current_position
                        alt = self._current_altitude

                    if pos is not None:
                        frame = self.shared.frame_buffer.get()
                        if frame is not None:
                            photo_saver.update(
                                lat=pos[0],
                                lon=pos[1],
                                alt=alt,
                                frame=frame,
                            )

                with self._lock:
                    seq = self._last_mission_reached_seq
                    current = self._current_mission_seq

                progress = seq if seq is not None else current
                if progress is not None and progress >= self._mapping_final_seq:
                    logger.info("[MISSION] Mapping complete")
                    self._save_target_observations()
                    return

                time.sleep(0.05)

        finally:
            # Always stop EXIF/JPEG capture even if the mapping loop exits unexpectedly.
            # Do not stop the camera thread here;
            # it is still needed for YOLO target centering during delivery.
            self._stop_mapping_photo_capture()

    def _stop_mapping_photo_capture(self) -> None:
        # ── 1) Stop PhotoSaver, if present ────────────────────────────────
        photo_saver = self._photo_saver
        photo_count = 0
        if photo_saver is not None:
            try:
                if photo_saver.is_running:
                    photo_saver.stop()
                    photo_count = photo_saver.photo_count
                    logger.info(
                        "[MISSION] Mapping complete -> photo capture STOPPED | "
                        "total=%d photos | directory=%s",
                        photo_count, str(PHOTO_SAVE_PATH),
                    )
            except Exception:
                logger.exception("[MISSION] Could not stop EXIF photo capture")
        else:
            logger.info("[MISSION] PhotoSaver unavailable (piexif may be missing); removable-storage copy will still run")

        # ── 2) Copy mission data to removable storage ───────────────────
        if not FLASH_PHOTO_COPY_ENABLED:
            return

        try:
            import os as _os

            from perception.flash_copy import FlashPhotoCopier

            source_dir = str(FLASH_COPY_SOURCE_DIR)

            source_empty = not _os.path.isdir(source_dir) or not any(
                True for _root, _dirs, _fnames in _os.walk(source_dir) for _ in _fnames
            )
            if source_empty:
                logger.info(
                    "[FlashCopy] No files to copy (%s is empty) → skipping removable-storage copy",
                    source_dir,
                )
                return

# Store each export in a separate timestamped directory.
# This prevents a later mapping run from overwriting an earlier export.
# The destination is kept under the configured export directory.
            import os as _os
            from datetime import datetime as _datetime
            timestamp = _datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            dest_dir = str(_os.path.join(FLASH_COPY_DEST_DIR, timestamp))
            logger.info(
                "[FlashCopy] Destination directory: %s", dest_dir,
            )

            def _on_copy_done(copied: int, failed: int) -> None:
                if failed == 0:
                    logger.info(
                        "[FlashCopy] ✓ Copied the data/ directory to removable storage | "
                        "total=%d files | %s → %s",
                        copied, source_dir, dest_dir,
                    )
                else:
                    logger.warning(
                        "[FlashCopy] Copy finished, but %d files failed | "
                        "copied=%d | %s → %s",
                        failed, copied, source_dir, dest_dir,
                    )

            self._flash_copier = FlashPhotoCopier(
                source_dir=source_dir,
                dest_dir=dest_dir,
                retry_count=int(FLASH_PHOTO_COPY_RETRY),
                verify=bool(FLASH_PHOTO_COPY_VERIFY),
                on_done=_on_copy_done,
            )
            self._flash_copier.start()
# Wait for the configured interval for removable-storage copying to finish.
# If the wait expires, keep the mission running while copying continues.
            finished = self._flash_copier.join(timeout_s=float(FLASH_PHOTO_COPY_WAIT_S))
            if not finished:
                logger.warning(
                    "[FlashCopy] Wait limit of %.1f s reached; delivery continues while "
                    "copying may finish in the background.", float(FLASH_PHOTO_COPY_WAIT_S)
                )

        except Exception:
            # File-copy errors must not interrupt the autonomous mission.
            logger.exception(
                "[FlashCopy] Could not start removable-storage copy; mission continues"
            )

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def _start_camera(self) -> None:
        if self._camera_running:
            return

        if SIYI_A8_MINI_ENABLED and self._a8mini_recorder is not None:
            logger.info("[AutonomousMission] A8 Mini camera mode enabled")

            try:
                self._a8mini_recorder.set_gimbal_angle(
                    yaw=float(SIYI_A8_MINI_GIMBAL_YAW_DEG),
                    pitch=float(SIYI_A8_MINI_GIMBAL_PITCH_DEG),
                )
                time.sleep(float(SIYI_A8_MINI_GIMBAL_SETTLE_S))
            except Exception:
                logger.exception("[AutonomousMission] Could not set the gimbal angle")

            ok = self._a8mini_recorder.start_recording()
            if not ok:
                logger.error(
                    "[AutonomousMission] Could not start A8 Mini recording; "
                    "switched to Gazebo camera mode"
                )
            else:
                self._camera_running = True
                self._camera_thread = threading.Thread(
                    target=self._a8mini_camera_loop,
                    name="A8MiniCameraLoop",
                    daemon=True,
                )
                self._camera_thread.start()

                # Keep video capture available even when the detector cannot load.
                # Use the configured model and inference-device settings.
                # Inference runs in a worker so camera acquisition stays responsive.
                try:
                    device = "cuda:0" if USE_CUDA else "cpu"
                    if not os.path.isfile(YOLO_MODEL_PATH):
                        raise FileNotFoundError(
                            f"YOLO model not found: {YOLO_MODEL_PATH}"
                        )
                    self._detector = YOLODetector(
                        YOLOConfig(
                            model_path=YOLO_MODEL_PATH,
                            confidence_threshold=float(YOLO_CONFIDENCE_THRESHOLD),
                            iou_threshold=float(YOLO_IOU_THRESHOLD),
                            device=device,
                            half_precision=bool(YOLO_HALF_PRECISION),
                            sahi_rows=int(YOLO_SAHI_ROWS),
                            sahi_cols=int(YOLO_SAHI_COLS),
                            frame_skip=int(YOLO_FRAME_SKIP),
                        )
                    )
                    self._detector.load()
                    logger.info("[AutonomousMission] ✓ YOLO model loaded")
                    # Start the asynchronous inference worker after model loading.
                    self._start_yolo_worker()
                except Exception as exc:
                    self._detector = None
                    logger.warning(
                        "[AutonomousMission] YOLO is unavailable: %s | "
                        "Camera streaming, video recording, and EXIF photo capture will continue.",
                        exc,
                    )
                return

        self._camera_running = True
        self._camera_thread = threading.Thread(
            target=self._camera_loop,
            name="AutonomousCamera",
            daemon=True,
        )
        self._camera_thread.start()

        # YOLO is optional. If the model is missing or invalid, only object detection
        # YOLO is optional; camera streaming and photo capture should continue
        # when its model is missing or cannot be initialized.
        try:
            device = "cuda:0" if USE_CUDA else "cpu"
            if not os.path.isfile(YOLO_MODEL_PATH):
                raise FileNotFoundError(
                    f"YOLO model not found: {YOLO_MODEL_PATH}"
                )
            self._detector = YOLODetector(
                YOLOConfig(
                    model_path=YOLO_MODEL_PATH,
                    confidence_threshold=float(YOLO_CONFIDENCE_THRESHOLD),
                    iou_threshold=float(YOLO_IOU_THRESHOLD),
                    device=device,
                    half_precision=bool(YOLO_HALF_PRECISION),
                    sahi_rows=int(YOLO_SAHI_ROWS),
                    sahi_cols=int(YOLO_SAHI_COLS),
                    frame_skip=int(YOLO_FRAME_SKIP),
                )
            )
            self._detector.load()
            logger.info("[AutonomousMission] ✓ YOLO model loaded")
            # Keep the camera loop alive even when inference is unavailable.
            self._start_yolo_worker()
        except Exception as exc:
            self._detector = None
            logger.warning(
                "[AutonomousMission] YOLO is unavailable: %s | "
                "Camera streaming, video recording, and EXIF photo capture will continue.",
                exc,
            )

    def _camera_loop(self) -> None:
        CAMERA_TOPIC = (
            "/world/iris_runway/model/iris_with_gimbal/model/gimbal/link/"
            "pitch_link/sensor/camera/image/enable_streaming"
        )
        UDP_PORT = 5600
        window_name = "SUAS Camera - Annotated YOLO"
        process = None
        last_inference = 0.0
        last_detections = []
        frame_dir = os.path.abspath(GSTREAMER_FRAME_SAVE_DIR)
        frame_pattern = os.path.join(frame_dir, "frame_%06d.jpg")

        def enable_gazebo_streaming() -> None:
            deadline = time.monotonic() + 30.0
            while time.monotonic() < deadline and self._camera_running and not self._stop.is_set():
                try:
                    result = subprocess.run(
                        ["gz", "topic", "-l"],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.DEVNULL,
                        text=True,
                        timeout=3.0,
                    )
                    if CAMERA_TOPIC in result.stdout:
                        break
                except Exception:
                    pass
                time.sleep(1.0)
            else:
                raise RuntimeError(f"Gazebo camera topic not found: {CAMERA_TOPIC}")

            for _ in range(5):
                subprocess.run(
                    [
                        "gz", "topic", "-t", CAMERA_TOPIC,
                        "-m", "gz.msgs.Boolean", "-p", "data: true",
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=3.0,
                    check=False,
                )
                time.sleep(0.5)

        def latest_frame_file():
            try:
                files = glob.glob(os.path.join(frame_dir, "frame_*.jpg"))
                if not files:
                    return None
                return max(files, key=os.path.getmtime)
            except Exception:
                return None

        try:
            enable_gazebo_streaming()

            os.makedirs(frame_dir, exist_ok=True)
            for old in glob.glob(os.path.join(frame_dir, "frame_*.jpg")):
                try:
                    os.remove(old)
                except OSError:
                    pass

            pipeline = [
                "gst-launch-1.0", "-q",
                "udpsrc", "address=127.0.0.1", f"port={UDP_PORT}",
                "caps=application/x-rtp,media=video,clock-rate=90000,encoding-name=H264,payload=96",
                "!", "rtph264depay",
                "!", "avdec_h264",
                "!", "videoconvert",
                "!", "jpegenc", f"quality={int(GSTREAMER_JPEG_QUALITY)}",
                "!", "multifilesink", f"location={frame_pattern}",
            ]

            if not GSTREAMER_SAVE_ALL_FRAMES:
                pipeline.extend(["max-files=3"])
            else:
                pipeline.extend(["max-files=0"])

            process = subprocess.Popen(
                pipeline,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                bufsize=0,
            )

            def drain_stderr(stream):
                try:
                    while process is not None and process.poll() is None:
                        line = stream.readline()
                        if not line:
                            time.sleep(0.05)
                            continue
                        text = line.decode(errors="replace").strip()
                        if text:
                            logger.warning("[GStreamer] %s", text)
                except Exception:
                    pass

            if process.stderr is not None:
                threading.Thread(
                    target=drain_stderr,
                    args=(process.stderr,),
                    name="GStreamerStderr",
                    daemon=True,
                ).start()

            if int(DISPLAY_CAMERA_PREVIEW) == 1:
                try:
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(window_name, 960, 540)
                except Exception:
                    logger.exception("[AutonomousMission] Could not create the camera window")

            last_file = None
            last_frame_time = 0.0
            shown_placeholder = False

            while self._camera_running and not self._stop.is_set():
                if process.poll() is not None:
                    raise RuntimeError(
                        f"GStreamer camera stream exited (return code={process.returncode})"
                    )

                filename = latest_frame_file()
                if filename and filename != last_file:
                    try:
                        size1 = os.path.getsize(filename)
                        time.sleep(0.003)
                        size2 = os.path.getsize(filename)
                        if size1 != size2 or size2 < 1000:
                            filename = None
                    except OSError:
                        filename = None

                if filename and filename != last_file:
                    frame = cv2.imread(filename, cv2.IMREAD_COLOR)
                    if frame is not None:
                        last_file = filename
                        last_frame_time = time.monotonic()
                        self.shared.frame_buffer.update(frame)
                        display = frame.copy()
                        now = time.monotonic()
                        detector = self._detector

                        # Submit frames to the inference worker without blocking
                        # the camera loop; the bounded queue controls frame backlog.
                        # If full, the oldest queued frame is discarded below.
                        if detector is not None and now - last_inference >= 0.15:
                            try:
                                self._yolo_queue.put_nowait((frame, detector))
                            except queue.Full:
                                try:
                                    self._yolo_queue.get_nowait()
                                except queue.Empty:
                                    pass
                                try:
                                    self._yolo_queue.put_nowait((frame, detector))
                                except queue.Full:
                                    pass
                            last_inference = now
                        # Use the most recently available detections for display.
                        with self._lock:
                            last_detections = list(self._latest_detections)

                        h, w = display.shape[:2]
                        cv2.drawMarker(
                            display, (w // 2, h // 2), (0, 255, 255),
                            cv2.MARKER_CROSS, 30, 2,
                        )

                        for det in last_detections:
                            if float(det.confidence) < float(YOLO_CONFIDENCE_THRESHOLD):
                                continue
                            x1 = max(0, min(w - 1, int(det.bbox.x1)))
                            y1 = max(0, min(h - 1, int(det.bbox.y1)))
                            x2 = max(0, min(w - 1, int(det.bbox.x2)))
                            y2 = max(0, min(h - 1, int(det.bbox.y2)))
                            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            label = f"{det.class_name} {float(det.confidence):.2f}"
                            cv2.putText(
                                display, label, (x1, max(25, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
                                cv2.LINE_AA,
                            )

                        if int(DISPLAY_CAMERA_PREVIEW) == 1:
                            cv2.putText(
                                display,
                                f"YOLO ANNOTATED | Objects: {len(last_detections)}",
                                (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                (0, 255, 255), 2, cv2.LINE_AA,
                            )
                            cv2.imshow(window_name, display)
                            shown_placeholder = False

                if int(DISPLAY_CAMERA_PREVIEW) == 1:
                    if last_frame_time == 0.0 and not shown_placeholder:
                        placeholder = np.zeros((540, 960, 3), dtype=np.uint8)
                        cv2.putText(
                            placeholder, "Waiting for the Gazebo camera stream...",
                            (180, 270), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                            (255, 255, 255), 2, cv2.LINE_AA,
                        )
                        cv2.imshow(window_name, placeholder)
                        shown_placeholder = True

                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        self._camera_running = False
                        break
                time.sleep(0.005)

        except Exception:
            logger.exception("[AutonomousMission] Camera/GStreamer loop failed")
        finally:
            if process is not None:
                try:
                    process.terminate()
                    process.wait(timeout=2.0)
                except Exception:
                    try:
                        process.kill()
                    except Exception:
                        pass
            if int(DISPLAY_CAMERA_PREVIEW) == 1:
                try:
                    cv2.destroyWindow(window_name)
                except Exception:
                    pass

    def _a8mini_camera_loop(self) -> None:
        import config as _cfg
        import gi

        gi.require_version("Gst", "1.0")
        from gi.repository import Gst

        Gst.init(None)

        rtsp_url = _cfg.SIYI_A8_MINI_RTSP_URL
        max_fails = max(3, int(_cfg.SIYI_A8_MINI_MAX_FRAME_FAILURES))
        reconnect_delay = max(0.5, float(_cfg.SIYI_A8_MINI_RECONNECT_DELAY_S))
        max_reboots = int(_cfg.SIYI_A8_MINI_MAX_REBOOT_ATTEMPTS)
        window_name = "SUAS - A8 Mini Camera"

        if int(DISPLAY_CAMERA_PREVIEW) == 1:
            try:
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window_name, 960, 540)
            except Exception:
                pass

        last_detections = []
        last_inference = 0.0
        reboot_attempts = 0
        consecutive_fail = 0

        recorder = self._a8mini_recorder

        # Jetson Orin NX:
        # RTSP(H264) -> RTP depay -> H264 parse -> NVDEC -> NVMM conversion
        # -> BGRx -> leaky queue -> appsink.
        #
        # Downstream leaky queues and a one-frame appsink buffer prevent stale
        # frames from accumulating when inference is slower than video decoding.
        pipeline_desc = (
            f'rtspsrc location="{rtsp_url}" '
            f'protocols=tcp '
            f'latency=200 '
            f'drop-on-latency=true '
            f'tcp-timeout=3000000 '
            f'timeout=3000000 '
            f'! rtph264depay '
            f'! h264parse config-interval=1 '
            f'! nvv4l2decoder enable-max-performance=1 '
            f'! nvvidconv '
            f'! video/x-raw,format=BGRx '
            f'! queue max-size-buffers=1 max-size-bytes=0 max-size-time=0 leaky=downstream '
            f'! appsink name=appsink emit-signals=false sync=false '
            f'max-buffers=1 drop=true'
        )

        def make_pipeline():
            try:
                return Gst.parse_launch(pipeline_desc)
            except Exception:
                logger.exception("[A8Mini GStreamer] Could not create the pipeline")
                return None

        def sample_to_bgr(sample):
            """GStreamer BGRx sample -> OpenCV BGR."""
            buf = sample.get_buffer()
            caps = sample.get_caps()
            if buf is None or caps is None:
                return None

            structure = caps.get_structure(0)
            width = structure.get_value("width")
            height = structure.get_value("height")
            if not width or not height:
                return None

            ok, map_info = buf.map(Gst.MapFlags.READ)
            if not ok:
                return None

            try:
                # BGRx = 4 byte/pixel. map_info.data PyCapsule/bytes olabilir;
                # map_info.data may be bytes or a PyCapsule; frombuffer avoids
                raw = np.frombuffer(map_info.data, dtype=np.uint8)
                expected = int(width) * int(height) * 4
                if raw.size < expected:
                    return None

                bgra = raw[:expected].reshape((int(height), int(width), 4))
                # Return a contiguous BGR frame independent of the mapped buffer.
                return np.ascontiguousarray(bgra[:, :, :3])
            finally:
                buf.unmap(map_info)

        while self._camera_running and not self._stop.is_set():
            pipeline = None
            appsink = None

            try:
                logger.info("[A8Mini GStreamer] Connecting to RTSP: %s", rtsp_url)

                pipeline = make_pipeline()
                if pipeline is None:
                    consecutive_fail += 1
                    time.sleep(reconnect_delay)
                    continue

                appsink = pipeline.get_by_name("appsink")
                if appsink is None:
                    raise RuntimeError("appsink was not found")

                pipeline.set_state(Gst.State.PLAYING)

                # Wait for the pipeline state transition before pulling frames.
                ret, state, pending = pipeline.get_state(3 * Gst.SECOND)
                if ret == Gst.StateChangeReturn.FAILURE:
                    raise RuntimeError("GStreamer could not transition to PLAYING")

                consecutive_fail = 0
                reboot_attempts = 0
                logger.info(
                    "[A8Mini GStreamer] ✓ Jetson NVDEC RTSP connection established"
                )

                bus = pipeline.get_bus()

                while self._camera_running and not self._stop.is_set():
                    # Use a short pull timeout so shutdown remains responsive.
                    sample = appsink.emit(
                        "try-pull-sample",
                        200 * Gst.MSECOND,
                    )

                    if sample is None:
                        # Inspect the bus when no sample arrives to detect stream errors.
                        msg = bus.pop_filtered(
                            Gst.MessageType.ERROR | Gst.MessageType.EOS
                        )

                        if msg is not None:
                            if msg.type == Gst.MessageType.ERROR:
                                err, dbg = msg.parse_error()
                                logger.warning(
                                    "[A8Mini GStreamer] H264/RTSP ERROR: %s | %s",
                                    err.message,
                                    dbg or "",
                                )
                            else:
                                logger.warning(
                                    "[A8Mini GStreamer] RTSP EOS — reconnecting"
                                )

                        consecutive_fail += 1
                        if consecutive_fail >= max_fails:
                            logger.warning(
                                "[A8Mini GStreamer] Frame timeout %d/%d — "
                                "rebuilding pipeline",
                                consecutive_fail,
                                max_fails,
                            )
                            break
                        continue

                    frame = sample_to_bgr(sample)
                    if frame is None:
                        consecutive_fail += 1
                        if consecutive_fail >= max_fails:
                            logger.warning(
                                "[A8Mini GStreamer] Invalid sample %d/%d",
                                consecutive_fail,
                                max_fails,
                            )
                            break
                        continue

                    consecutive_fail = 0

                    # Share the latest frame with downstream perception components.
                    self.shared.frame_buffer.update(frame)

                    now = time.monotonic()
                    detector = self._detector
                    if (
                        detector is not None
                        and now - last_inference >= OBJECT_DETECTION_INTERVAL_S
                    ):
                        # Do not block camera acquisition on slow inference.
                        # A full queue is handled by dropping the oldest frame.
                        try:
                            self._yolo_queue.put_nowait((frame, detector))
                        except queue.Full:
                            try:
                                self._yolo_queue.get_nowait()
                            except queue.Empty:
                                pass
                            try:
                                self._yolo_queue.put_nowait((frame, detector))
                            except queue.Full:
                                pass
                        last_inference = now
                    # Use the most recently available detections for display.
                    with self._lock:
                        last_detections = list(self._latest_detections)

                    if int(DISPLAY_CAMERA_PREVIEW) == 1:
                        display = frame.copy()
                        h, w = display.shape[:2]
                        cv2.drawMarker(
                            display, (w // 2, h // 2), (0, 255, 255),
                            cv2.MARKER_CROSS, 30, 2,
                        )
                        for det in last_detections:
                            if float(det.confidence) < float(YOLO_CONFIDENCE_THRESHOLD):
                                continue
                            x1 = max(0, min(w - 1, int(det.bbox.x1)))
                            y1 = max(0, min(h - 1, int(det.bbox.y1)))
                            x2 = max(0, min(w - 1, int(det.bbox.x2)))
                            y2 = max(0, min(h - 1, int(det.bbox.y2)))
                            cv2.rectangle(
                                display, (x1, y1), (x2, y2),
                                (0, 255, 0), 2
                            )
                            label = f"{det.class_name} {float(det.confidence):.2f}"
                            cv2.putText(
                                display, label, (x1, max(25, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 255, 0), 2, cv2.LINE_AA,
                            )

                        cv2.putText(
                            display,
                            f"A8 Mini NVDEC | Objects: {len(last_detections)}",
                            (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (0, 200, 255), 2, cv2.LINE_AA,
                        )
                        cv2.imshow(window_name, display)

                        key = cv2.waitKey(1) & 0xFF
                        if key in (ord("q"), 27):
                            self._camera_running = False
                            break

            except Exception:
                logger.exception("[A8Mini GStreamer] Camera loop error")
                consecutive_fail += 1

            finally:
                if pipeline is not None:
                    try:
                        pipeline.set_state(Gst.State.NULL)
                        pipeline.get_state(2 * Gst.SECOND)
                    except Exception:
                        pass
                    try:
                        pipeline = None
                    except Exception:
                        pass

            if not self._camera_running or self._stop.is_set():
                break

            # A single dropped frame no longer triggers a soft reboot.
            # The pipeline is restarted cleanly first.
            if consecutive_fail < max_fails:
                time.sleep(reconnect_delay)
                continue

            if reboot_attempts >= max_reboots:
                logger.error(
                    "[A8Mini GStreamer] Maximum reboot count reached (%d).",
                    max_reboots,
                )
                break

            reboot_attempts += 1
            consecutive_fail = 0

            logger.warning(
                "[A8Mini GStreamer] Stream timeout → "
                "soft-rebooting camera (%d/%d)",
                reboot_attempts,
                max_reboots,
            )

            if recorder is not None:
                try:
                    recorder.stop_recording()
                    recorder.soft_reboot()
                    reachable = recorder.wait_until_reachable()
                except Exception:
                    logger.exception(
                        "[A8Mini GStreamer] Soft reboot failed"
                    )
                    reachable = False
            else:
                reachable = False

            if not reachable:
                time.sleep(reconnect_delay)
                continue

            logger.info("[A8Mini GStreamer] Camera stream recovered.")

            if recorder is not None:
                try:
                    recorder.set_gimbal_angle(
                        yaw=float(SIYI_A8_MINI_GIMBAL_YAW_DEG),
                        pitch=float(SIYI_A8_MINI_GIMBAL_PITCH_DEG),
                    )
                    time.sleep(float(SIYI_A8_MINI_GIMBAL_SETTLE_S))
                    recorder.start_recording()
                except Exception:
                    logger.exception(
                        "[A8Mini GStreamer] Could not restart recording"
                    )

        if int(DISPLAY_CAMERA_PREVIEW) == 1:
            try:
                cv2.destroyWindow(window_name)
            except Exception:
                pass

        logger.info(
            "[A8Mini GStreamer] Camera loop ended "
            "(reboot=%d, last error=%d)",
            reboot_attempts,
            consecutive_fail,
        )

    def _stop_camera(self) -> None:
        self._camera_running = False

        # Stop the YOLO worker by enqueueing a sentinel (None) and waiting for it.
        try:
            self._yolo_queue.put_nowait(None)
        except queue.Full:
            pass
        if self._yolo_thread is not None and self._yolo_thread.is_alive():
            self._yolo_thread.join(timeout=2.0)
        self._yolo_thread = None

        with self._lock:
            self._latest_detections = []
            self._latest_detection_time = 0.0
        if self._a8mini_recorder is not None:
            try:
                self._a8mini_recorder.stop_recording()
            except Exception:
                logger.exception("[AutonomousMission] Could not stop A8 Mini recording")

    def _start_yolo_worker(self) -> None:
        if self._yolo_thread is not None and self._yolo_thread.is_alive():
            return
        self._yolo_thread = threading.Thread(
            target=self._yolo_worker_loop,
            name="YOLOWorker",
            daemon=True,
        )
        self._yolo_thread.start()

    def _yolo_worker_loop(self) -> None:
        logger.info("[YOLOWorker] Started")
        while self._camera_running and not self._stop.is_set():
            try:
                # Wait for the blocking queue; exit when the None sentinel is received.
                item = self._yolo_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if item is None:
                break  # Sentinel value used to stop the worker.

            frame, detector = item

            try:
                detections = detector.detect(self._yolo_frame(frame))
                now = time.monotonic()
                with self._lock:
                    self._latest_detections = list(detections)
                    self._latest_detection_time = now
            except Exception:
                logger.exception("[YOLOWorker] Inference error")

        logger.info("[YOLOWorker] Stopped")

    def _detect_mapping_targets(self) -> None:
        now = time.monotonic()
        if now - self._last_detection_time < float(OBJECT_DETECTION_INTERVAL_S):
            return
        self._last_detection_time = now

        if self._detector is None:
            return

        with self._lock:
            detections = list(self._latest_detections)
            detection_time = float(self._latest_detection_time)
            current_position = self._current_position
            last_processed = float(self._last_processed_detection_time)

        # Do not count the same inference result in multiple mapping iterations.
        if detection_time <= 0.0 or detection_time <= last_processed:
            return
        with self._lock:
            self._last_processed_detection_time = detection_time

        if current_position is None:
            return

        seen_this_cycle = set()
        best_by_target = {}
        for detection in detections:
            try:
                confidence = float(detection.confidence)
            except Exception:
                continue
            if confidence < float(TARGET_DELIVERY_CONFIDENCE_THRESHOLD):
                continue
            target = self._target_for_detection(detection)
            if target is None:
                continue
            x, y = detection.normalized_center
            error = math.hypot(x - 0.5, y - 0.5)
            key = target.class_name
            seen_this_cycle.add(key)
            previous = best_by_target.get(key)
            if previous is None or error < previous[0]:
                best_by_target[key] = (error, confidence, target)

        with self._lock:
            for target in self._targets.values():
                if target.class_name not in seen_this_cycle:
                    target.consecutive_detections = 0

            for key, (error, confidence, target) in best_by_target.items():
                target.consecutive_detections += 1
                if target.consecutive_detections >= int(TARGET_REQUIRED_CONSECUTIVE_DETECTIONS):
                    target.confirmed = True
                    target.seen = True
                    if error < target.best_error:
                        # Do not save a target position when the aircraft is outside the mapping area.
                        if not self._point_in_polygon(current_position[0], current_position[1]):
                            logger.debug(
                                "[TARGET] %s detected outside the polygon -> location not saved | "
                                "lat=%.7f lon=%.7f",
                                target.class_name, current_position[0], current_position[1],
                            )
                            continue
                        target.best_error = error
                        target.latitude = current_position[0]
                        target.longitude = current_position[1]
                        target.confidence = confidence
                        logger.info(
                            "[TARGET] %s CONFIRMED | consecutive=%d/%d | lat=%.7f lon=%.7f | confidence=%.1f%% | center_error=%.4f",
                            target.class_name, target.consecutive_detections,
                            int(TARGET_REQUIRED_CONSECUTIVE_DETECTIONS),
                            target.latitude, target.longitude, confidence * 100.0, error,
                        )

    def _target_for_detection(self, detection: Any) -> Optional[TargetObservation]:
        name = str(getattr(detection, "class_name", "")).strip()
        class_id = getattr(detection, "class_id", None)

        for target in self._targets.values():
            if target.class_name.strip().lower() == name.lower():
                return target
            if target.class_id is not None and class_id == target.class_id:
                return target

        if AUTO_ASSIGN_TARGET_CLASSES and name:
            for target in self._targets.values():
                if target.seen and target.class_name.strip().lower() == name.lower():
                    return target
            for target in self._targets.values():
                if target.class_name.lower().startswith("object") and target.class_id is None:
                    logger.warning(
                        "[AutonomousMission] YOLO class %r (id=%r) -> automatically mapped to target %s",
                        name, class_id, target.class_name,
                    )
                    target.class_name = name
                    target.class_id = class_id
                    return target

        return None

    def _save_target_observations(self) -> None:
        try:
            import json
            from pathlib import Path

            path = Path("data/target_positions.json")
            path.parent.mkdir(parents=True, exist_ok=True)

            payload = {}
            for name, target in self._targets.items():
                payload[name] = {
                    "seen": target.seen,
                    "confirmed": target.confirmed,
                    "consecutive_detections": target.consecutive_detections,
                    "latitude": target.latitude,
                    "longitude": target.longitude,
                    "confidence": target.confidence,
                    "center_error": target.best_error if target.seen else None,
                }

            path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        except Exception:
            logger.exception("[AutonomousMission] Could not save target locations")

    # ------------------------------------------------------------------
    # Delivery
    # ------------------------------------------------------------------

    def _prepare_delivery_queue(self) -> None:
        with self._lock:
            current = self._current_position
        confirmed = []
        for name, target in self._targets.items():
            if target.confirmed and target.latitude is not None and target.longitude is not None:
                confirmed.append(str(name))

        if current is not None:
            confirmed.sort(key=lambda n: distance(current[0], current[1],
                                                  float(self._targets[n].latitude),
                                                  float(self._targets[n].longitude)))
        self._delivery_queue = confirmed
        self._delivery_index = 0
        self._delivery_fallback_used = False
        logger.info("[MISSION] Confirmed target queue: %s", " -> ".join(confirmed) if confirmed else "none")
        if current is not None:
            logger.info("[MISSION] Delivery start/hover position: %.7f, %.7f", current[0], current[1])

    def _enter_delivery_hover(self) -> bool:
        with self._lock:
            pos = self._current_position
            alt = max(float(self._current_altitude), float(DELIVERY_APPROACH_ALTITUDE_M))
        if pos is None:
            raise RuntimeError("Current GPS position unavailable for delivery hover")
        if not self.mavlink.set_mode("GUIDED"):
            raise RuntimeError("Could not enter GUIDED hover")
        if not self.mavlink.goto_guided(pos[0], pos[1], alt, speed_mps=min(float(GUIDED_SPEED_MPS), 3.0)):
            raise RuntimeError("Could not command delivery hover")
        self._state = AutonomousState.DELIVERY
        logger.info("[MISSION] Mapping complete -> GUIDED hover | %.7f, %.7f at %.1f m", pos[0], pos[1], alt)
        return True

    def _goto_auto_and_wait(self, lat: float, lon: float, alt: float, timeout_s: float = 180.0) -> None:
        lat = float(lat)
        lon = float(lon)
        alt = float(alt)

        # Explicitly return to GUIDED mode before handling each target.
        # This avoids conflicts with the previous target mission upload in AUTO mode.
        if not self.mavlink.set_mode("GUIDED"):
            raise RuntimeError("Could not enter GUIDED before delivery mission upload")
        logger.info("[MISSION] Preparing delivery target -> GUIDED active")

        # The delivery waypoint is intentionally uploaded twice as sequence items 0 and 1.
        # Both items use the same coordinate. This matches the expected ArduPilot mission flow,
        # where the target waypoint appears in two consecutive items.
        item_seq0 = MissionItem(
            latitude=lat, longitude=lon, altitude_m=MAPPING_ALTITUDE_M,
            seq=0, current=1, autocontinue=1,
            command=mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
            frame=mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
            param1=0.0, param2=2.0, param3=0.0, param4=0.0,
        )
        item_seq1 = MissionItem(
            latitude=lat, longitude=lon, altitude_m=MAPPING_ALTITUDE_M,
            seq=1, current=0, autocontinue=1,
            command=mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
            frame=mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
            param1=0.0, param2=2.0, param3=0.0, param4=0.0,
        )

        result = self.mavlink.mission_protocol.upload([item_seq0, item_seq1])
        if not getattr(result, "success", False):
            raise RuntimeError(
                "Temporary AUTO target mission upload failed: "
                f"{getattr(result, 'message', result)}"
            )
        logger.info(
            "[MISSION] AUTO waypoints uploaded | seq=0 and seq=1 | "
            "%.7f, %.7f @ %.1f m", lat, lon, alt
        )

        result = self.mavlink.mission_protocol.set_current(0)
        if not getattr(result, "success", False):
            raise RuntimeError(
                "Temporary AUTO target set_current failed: "
                f"{getattr(result, 'message', result)}"
            )
        logger.info("[MISSION] AUTO current waypoint set to 0")

        # Use the same ArduPilot 4.6.x sequence as the combined mission:
        # AUTO -> MISSION_START. Verify the AUTO transition more than once.
        auto_ok = False
        for attempt in range(1, 4):
            if self.mavlink.set_mode("AUTO"):
                auto_ok = True
                break
            logger.warning(
                "[MISSION] Could not switch to AUTO for delivery (%d/3); retrying...",
                attempt,
            )
            time.sleep(0.75)

        if not auto_ok:
            raise RuntimeError("Could not enter AUTO for delivery target")

        logger.info("[MISSION] AUTO mode active -> starting delivery waypoint")

        mission_start_cmd = getattr(mavutil.mavlink, "MAV_CMD_MISSION_START", 300)
        if not self.mavlink.command_long(mission_start_cmd, 0, 0, 0, 0, 0, 0, 0):
            raise RuntimeError("Could not send MISSION_START for delivery target")
        logger.info("[MISSION] MISSION_START sent (command=%d)", mission_start_cmd)

        # Verify that AUTO remains active after the command.
        time.sleep(0.5)
        if not self.mavlink.set_mode("AUTO"):
            raise RuntimeError("AUTO dropped after MISSION_START for delivery target")

        logger.info("[MISSION] AUTO target: %.7f, %.7f @ %.1f m", lat, lon, alt)
        deadline = time.monotonic() + float(timeout_s)
        while not self._stop.is_set() and time.monotonic() < deadline:
            with self._lock:
                pos = self._current_position
                cur_alt = float(self._current_altitude)

            if (
                pos is not None
                and distance(pos[0], pos[1], lat, lon) <= float(HOME_ACCEPT_RADIUS_M)            ):
                logger.info(
                    "[MISSION] AUTO target reached | %.7f, %.7f | GPS distance=%.1f m",
                    lat, lon, distance(pos[0], pos[1], lat, lon),
                )
                # After the AUTO mission completes, explicitly switch back to GUIDED
                # mode before issuing target-centering commands.
                if not self.mavlink.set_mode("GUIDED"):
                    raise RuntimeError("Could not return to GUIDED for target centering")
                logger.info("[MISSION] GUIDED active at target -> starting centering")
                return
            time.sleep(0.1)

        raise RuntimeError("AUTO delivery target timeout")

    def _delivery_loop(self) -> None:
        """Deliver confirmed targets and apply the configured missing-target policy."""
        remaining = list(self._delivery_queue)
        while not self._stop.is_set():
            if remaining:
                with self._lock:
                    cur = self._current_position
                if cur is not None:
                    remaining.sort(key=lambda n: distance(cur[0], cur[1],
                                                         float(self._targets[n].latitude),
                                                         float(self._targets[n].longitude)))
                name = remaining.pop(0)
                target = self._targets[name]
                self._current_delivery_target = name
                self._state = AutonomousState.GUIDED_TO_TARGET
                self._goto_auto_and_wait(float(target.latitude), float(target.longitude), float(DELIVERY_APPROACH_ALTITUDE_M))
                if self._stop.is_set(): return
                self._state = AutonomousState.CENTERING
                centered = self._center_specific_target(name, timeout_s=float(TARGET_CENTERING_TIMEOUT_S))
                self._centering.stop()
                with self._lock:
                    drop_pos = self._current_position
                if drop_pos is None:
                    raise RuntimeError(f"No GPS position for target {name}")
                if not centered:
                    logger.warning("[MISSION] Could not center %s -> attempting safe release at the current position", name)
                self._release_payload(name)
                self._wait_after_release()
                continue

            # By default, do not release a payload for targets that were not confirmed during mapping.
            missing = [str(n) for n, target in self._targets.items() if not target.confirmed]
            if missing:
                if SKIP_UNCONFIRMED_TARGETS or not FALLBACK_DROP_ENABLED:
                    logger.warning(
                        "[MISSION] Targets without confirmed detections were skipped; payload fallback is disabled: %s",
                        ", ".join(missing),
                    )
                    break

                # Fallback release is available only when both settings are deliberately enabled:
                # SKIP_UNCONFIRMED_TARGETS=False ve FALLBACK_DROP_ENABLED=True.
                center = self._polygon_center()
                logger.warning(
                    "[MISSION] Targets could not be confirmed; moving to the configured fallback area: %s",
                    ", ".join(missing),
                )
                for name in missing:
                    if self._stop.is_set():
                        return
                    self._current_delivery_target = name
                    self._state = AutonomousState.FALLBACK_CENTER
                    self._goto_auto_and_wait(
                        center[0], center[1], float(DELIVERY_APPROACH_ALTITUDE_M)
                    )
                    if self._stop.is_set():
                        return
                    # Fallback mode does not use visual centering, so it must be enabled explicitly.
                    self._release_payload(name)
                    self._wait_after_release()
                    self._targets[name].confirmed = True
                break
            break

        self._current_delivery_target = None
        logger.info("[MISSION] Deliveries complete | HOME and LAND")

    def _wait_after_release(self) -> None:
        delay = max(0.0, float(PAYLOAD_DELAY_S))
        if delay > 0:
            deadline = time.monotonic() + delay
            while not self._stop.is_set() and time.monotonic() < deadline:
                time.sleep(0.05)

    def _hold_position(self, lat: float, lon: float, alt: float, duration_s: float) -> bool:
        """Refresh GUIDED position commands to hold position at the target."""
        deadline = time.monotonic() + max(0.0, duration_s)
        next_cmd = 0.0
        while not self._stop.is_set() and time.monotonic() < deadline:
            now = time.monotonic()
            if now >= next_cmd:
                if not self.mavlink.goto_guided(
                    lat, lon, alt, speed_mps=min(float(GUIDED_SPEED_MPS), 3.0)
                ):
                    return False
                next_cmd = now + 1.0
            time.sleep(0.1)
        return not self._stop.is_set()

    def _goto_guided_and_wait(self, lat: float, lon: float, alt: float) -> None:
        if not self.mavlink.set_mode("GUIDED"):
            raise RuntimeError("Could not enter GUIDED")

        ALT_TOLERANCE_M = 1.5
        HORIZONTAL_TOLERANCE_M = float(HOME_ACCEPT_RADIUS_M)
        REFRESH_INTERVAL_S = 2.0
        deadline = time.monotonic() + 180.0

        with self._lock:
            current_pos = self._current_position
            current_alt = float(self._current_altitude)

        if current_pos is not None and abs(current_alt - float(alt)) > ALT_TOLERANCE_M:
            hold_lat, hold_lon = current_pos
            logger.info(
                "[MISSION] Altitude change | %.1f m -> %.1f m | vertical movement first",
                current_alt, float(alt),
            )

            if not self.mavlink.goto_guided(
                hold_lat, hold_lon, float(alt),
                speed_mps=min(float(GUIDED_SPEED_MPS), 5.0),
            ):
                raise RuntimeError("GUIDED vertical altitude command failed")

            next_refresh = time.monotonic() + REFRESH_INTERVAL_S
            while not self._stop.is_set() and time.monotonic() < deadline:
                now = time.monotonic()
                if now >= next_refresh:
                    self.mavlink.goto_guided(
                        hold_lat, hold_lon, float(alt),
                        speed_mps=min(float(GUIDED_SPEED_MPS), 5.0),
                    )
                    next_refresh = now + REFRESH_INTERVAL_S

                with self._lock:
                    pos = self._current_position
                    cur_alt = float(self._current_altitude)

                horiz_ok = (
                    pos is None
                    or distance(pos[0], pos[1], hold_lat, hold_lon)
                    <= HORIZONTAL_TOLERANCE_M
                )
                if horiz_ok and abs(cur_alt - float(alt)) <= ALT_TOLERANCE_M:
                    logger.info(
                        "[MISSION] Target altitude reached | %.1f m | beginning horizontal movement",
                        cur_alt,
                    )
                    break
                time.sleep(0.1)
            else:
                if self._stop.is_set():
                    return
                raise RuntimeError("GUIDED vertical altitude transition timeout")

        if not self.mavlink.goto_guided(lat, lon, float(alt), speed_mps=GUIDED_SPEED_MPS):
            raise RuntimeError("GUIDED goto command failed")

        next_refresh = time.monotonic() + REFRESH_INTERVAL_S
        while not self._stop.is_set() and time.monotonic() < deadline:
            now = time.monotonic()
            if now >= next_refresh:
                self.mavlink.goto_guided(lat, lon, float(alt), speed_mps=GUIDED_SPEED_MPS)
                next_refresh = now + REFRESH_INTERVAL_S

            with self._lock:
                pos = self._current_position
                cur_alt = float(self._current_altitude)

            if pos is not None:
                horiz_ok = distance(pos[0], pos[1], lat, lon) <= HORIZONTAL_TOLERANCE_M
                vert_ok = abs(cur_alt - float(alt)) <= ALT_TOLERANCE_M
                if horiz_ok and vert_ok:
                    self._centering.stop()
                    return
            time.sleep(0.1)

        if self._stop.is_set():
            return
        raise RuntimeError("GUIDED goto timeout")

    def _center_specific_target(self, name: str, timeout_s: Optional[float] = None) -> bool:
        duration_s = float(
            TARGET_CENTERING_TIMEOUT_S if timeout_s is None else timeout_s
        )
        duration_s = max(0.0, duration_s)
        start_time = time.monotonic()
        deadline = start_time + duration_s

        stable_since = None
        ever_centered = False
        last_log = 0.0

        logger.info(
            "\n"
            "┌─────────────────────────────────────────────────────┐\n"
            "│  CENTERING STARTED  %-30s  │\n"
            "│  Time: %-5.1f s  |  Threshold: %-4.0f%%  |  Tolerance: %-5.3f  │\n"
            "└─────────────────────────────────────────────────────┘",
            name, duration_s,
            float(TARGET_DELIVERY_CONFIDENCE_THRESHOLD) * 100.0,
            float(TARGET_CENTER_TOLERANCE_NORMALIZED),
        )

        while not self._stop.is_set() and time.monotonic() < deadline:
            now = time.monotonic()
            frame = self.shared.frame_buffer.get()

            if frame is None or self._detector is None:
                self._centering.stop()
                stable_since = None
                time.sleep(0.05)
                continue

            try:
                detections = self._detector.detect(frame)
            except Exception:
                self._centering.stop()
                stable_since = None
                time.sleep(0.05)
                continue

            best = None
            best_error = 999.0
            for detection in detections:
                try:
                    confidence = float(detection.confidence)
                except Exception:
                    continue

                if confidence < float(TARGET_DELIVERY_CONFIDENCE_THRESHOLD):
                    continue
                if not self._matches_target(name, detection):
                    continue

                try:
                    x, y = detection.normalized_center
                    error = math.hypot(x - 0.5, y - 0.5)
                except Exception:
                    continue
                if error < best_error:
                    best = detection
                    best_error = error

            if best is None:
                # If the target is briefly lost, stop issuing velocity commands only;
                # do not reset the overall TARGET_CENTERING_TIMEOUT_S timeout.
                self._centering.stop()
                stable_since = None
                if now - last_log > 2.0:
                    visible = []
                    for detection in detections:
                        try:
                            visible.append(
                                f"{str(getattr(detection, 'class_name', '')).strip() or '?'} "
                                f"{float(getattr(detection, 'confidence', 0.0)) * 100.0:.0f}%"
                            )
                        except Exception:
                            pass
                    logger.warning(
                        "[CENTER] %-8s  TARGET NOT FOUND | seen=[%s] | "
                        "threshold=%.0f%% | remaining=%.1f s",
                        name, ", ".join(visible[:8]) or "none",
                        float(TARGET_DELIVERY_CONFIDENCE_THRESHOLD) * 100.0,
                        max(0.0, deadline - now),
                    )
                    last_log = now
                time.sleep(float(OBJECT_DETECTION_INTERVAL_S))
                continue

            try:
                x, y = best.normalized_center
                ok, vx_cmd, vy_cmd, err_x, err_y = self._centering.update_with_debug(x, y)

                if not ok:
                    logger.warning("[CENTER] %s  velocity command COULD NOT BE SENT", name)

                # ── Readable terminal status line for each frame ────────────────
                # Direction indicators: positive = forward/right, negative = backward/left.
                _vx_arrow = "▲ FORWARD " if vx_cmd > 0.005 else ("▼ BACKWARD " if vx_cmd < -0.005 else "● IDLE   ")
                _vy_arrow = "► RIGHT   " if vy_cmd > 0.005 else ("◄ LEFT   " if vy_cmd < -0.005 else "● IDLE   ")
                _centered_tag = " ✓ CENTERED" if best_error <= float(TARGET_CENTER_TOLERANCE_NORMALIZED) else ""

                if now - last_log >= 0.5:
                    logger.info(
                        "[CENTER] %-8s | "
                        "X-error:%+.3f  Y-error:%+.3f  |  "
                        "vx(%s):%+.2f m/s  vy(%s):%+.2f m/s  |  "
                        "total-error:%.3f  elapsed:%.1f/%.1f s%s",
                        name,
                        err_x, err_y,
                        _vx_arrow.strip(), vx_cmd,
                        _vy_arrow.strip(), vy_cmd,
                        best_error,
                        max(0.0, now - start_time), duration_s,
                        _centered_tag,
                    )
                    last_log = now

                if best_error <= float(TARGET_CENTER_TOLERANCE_NORMALIZED):
                    if stable_since is None:
                        stable_since = now
                    elif now - stable_since >= float(CENTERING_STABLE_TIME_S):
                        logger.info(
                            "\n"
                            "┌──────────────────────────────────────────────┐\n"
                            "│  ✓  CENTER REACHED: %-20s  │\n"
                            "│     stable time=%.1f s -> releasing payload     │\n"
                            "└──────────────────────────────────────────────┘",
                            name, float(CENTERING_STABLE_TIME_S),
                        )
                        ever_centered = True
                        self._centering.stop()
                        break
                else:
                    stable_since = None
            except Exception:
                # Target-centering update failed for this frame.
                # Instead of aborting the entire mission (including RTL fallback), treat the target
                # as temporarily lost, stop movement, and reset the stability counter.
                # Retry within the same TARGET_CENTERING_TIMEOUT_S timeout window.
                logger.exception(
                    "[CENTER] %s  unexpected error during centering update "
                    "-> frame skipped; mission continues",
                    name,
                )
                self._centering.stop()
                stable_since = None

            time.sleep(float(OBJECT_DETECTION_INTERVAL_S))

        self._centering.stop()

        elapsed = min(duration_s, max(0.0, time.monotonic() - start_time))
        if ever_centered:
            logger.info(
                "\n"
                "┌──────────────────────────────────────────────┐\n"
                "│  ✓  CENTERING COMPLETE: %-16s  │\n"
                "│     elapsed=%.1f/%.1f s                          │\n"
                "└──────────────────────────────────────────────┘",
                name, elapsed, duration_s,
            )
            return True

        logger.warning(
            "\n"
            "┌──────────────────────────────────────────────┐\n"
            "│  ✗  CENTERING FAILED: %-16s   │\n"
            "│     %.1f s elapsed without reaching the center         │\n"
            "└──────────────────────────────────────────────┘",
            name, duration_s,
        )
        return False

    def _matches_target(self, name: str, detection: Any) -> bool:
        target = self._targets[name]
        detected_name = str(getattr(detection, "class_name", "")).strip().casefold()
        target_name = str(target.class_name).strip().casefold()

        if detected_name and detected_name == target_name:
            return True

        target_class_id = getattr(target, "class_id", None)
        detected_class_id = getattr(detection, "class_id", None)
        if target_class_id is not None and detected_class_id is not None:
            try:
                return int(detected_class_id) == int(target_class_id)
            except (TypeError, ValueError):
                return detected_class_id == target_class_id

        return False

    def _release_payload(self, target_name: str) -> None:
        mapping = PAYLOAD_TARGET_CHANNELS.get(str(target_name).lower())
        if not mapping:
            logger.error("[SERVO] No channel mapping configured for %s", target_name)
            return

        ch1 = int(mapping["first_servo"])
        ch2 = int(mapping["second_servo"])

        def send(channel: int, pwm: int) -> bool:
            try:
                ok = bool(self.mavlink.send_servo_command(channel, int(pwm)))
            except Exception as exc:
                logger.exception(
                    "[SERVO] Command error | CH%d PWM=%d | %s", channel, pwm, exc
                )
                return False
            if ok:
                logger.info("[SERVO] CH%d -> %d PWM", channel, pwm)
            else:
                logger.error("[SERVO] CH%d -> %d PWM command was rejected", channel, pwm)
            return ok

        logger.info(
            "[SERVO] %s release sequence | CH%d, CH%d | open=%d reset=%d",
            target_name, ch1, ch2, int(PAYLOAD_OPEN_PWM), int(PAYLOAD_RESET_PWM)
        )

        ok1 = send(ch1, int(PAYLOAD_OPEN_PWM))
        if not ok1:
            logger.error("[SERVO] Could not open %s on channel %d", target_name, ch1)

        deadline = time.monotonic() + max(0.0, float(PAYLOAD_SEQUENCE_DELAY_S))
        while not self._stop.is_set() and time.monotonic() < deadline:
            time.sleep(0.02)

        ok2 = send(ch2, int(PAYLOAD_OPEN_PWM))
        if not ok2:
            logger.error("[SERVO] Could not open %s on channel %d", target_name, ch2)

        hold = max(0.0, float(PAYLOAD_OPEN_HOLD_S))
        deadline = time.monotonic() + hold
        while not self._stop.is_set() and time.monotonic() < deadline:
            time.sleep(0.02)

        logger.info(
            "[SERVO] %s release sequence completed | CH%d/CH%d reset to %d PWM",
            target_name, ch1, ch2, int(PAYLOAD_RESET_PWM)
        )

    # ------------------------------------------------------------------
    # Return and land: return-to-home and landing in AUTO mode.
    # ------------------------------------------------------------------

    def _return_home_and_land(self) -> None:
        if self._home is None:
            raise RuntimeError("Home position is unavailable")

        self._state = AutonomousState.RETURN_HOME
        home_lat = float(self._home[0])
        home_lon = float(self._home[1])
        return_alt = float(RETURN_HOME_ALTITUDE_M)

        logger.info(
            "[MISSION] Return-to-home started (AUTO) | "
            "HOME=%.7f,%.7f at %.1f m | LAND at HOME",
            home_lat, home_lon, return_alt,
        )

        # ── 1) Switch to GUIDED mode ────────────────────────────────────
        if not self.mavlink.set_mode("GUIDED"):
            raise RuntimeError("Could not enter GUIDED before return-home mission upload")
        logger.info("[MISSION] RTH: GUIDED active -> uploading mission")

        # ── 2) Mini mission: HOME WP + NAV_LAND ───────────────────────────
        #
        # seq=0: navigate to HOME at RETURN_HOME_ALTITUDE_M.
        # seq=1: land at HOME (altitude_m=0 means ground level).
        #
        # The mission item uses a relative-altitude frame, so its altitude is relative to the home position.
        # This matches the current configuration behaviour.
        rth_items = [
            MissionItem(
                latitude=home_lat,
                longitude=home_lon,
                altitude_m=MAPPING_ALTITUDE_M,
                seq=0,
                command=mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
                frame=mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                current=1,
                autocontinue=1,
                param1=0.0,   # hold time
                param2=float(HOME_ACCEPT_RADIUS_M),  # acceptance radius
                param3=0.0,
                param4=float("nan"),  # yaw: keep the current heading.
            ),
            MissionItem(
                latitude=home_lat,
                longitude=home_lon,
                altitude_m=MAPPING_ALTITUDE_M,
                seq=1,
                command=mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
                frame=mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                current=1,
                autocontinue=1,
                param1=0.0,   # hold time
                param2=float(HOME_ACCEPT_RADIUS_M),  # acceptance radius
                param3=0.0,
                param4=float("nan"),  # yaw: keep the current heading.
            ),
            MissionItem(
                latitude=home_lat,
                longitude=home_lon,
                altitude_m=0.0,
                seq=2,
                command=mavutil.mavlink.MAV_CMD_NAV_LAND,
                frame=mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                current=0,
                autocontinue=1,
                param1=0.0,   # abort altitude (0 disables the limit)
                param2=0.0,   # land mode (0 = normal)
                param3=0.0,
                param4=float("nan"),  # yaw: keep the current heading.
            ),
        ]

        result = self.mavlink.mission_protocol.upload(rth_items)
        if not getattr(result, "success", False):
            logger.warning(
                "[MISSION] RTH mission upload failed: %s; retrying...",
                getattr(result, "message", result),
            )
            if not self._sleep(0.5):
                return
            result = self.mavlink.mission_protocol.upload(rth_items)

        if not getattr(result, "success", False):
            raise RuntimeError(
                "Return-home AUTO mission upload failed: "
                f"{getattr(result, 'message', result)}"
            )
        logger.info("[MISSION] RTH mission uploaded (2 items: WP + LAND)")

        # ── 3) Set current=0 to start at the HOME waypoint ─────────────
        set_result = self.mavlink.mission_protocol.set_current(0)
        if not getattr(set_result, "success", False):
            raise RuntimeError(
                "Return-home set_current(0) failed: "
                f"{getattr(set_result, 'message', set_result)}"
            )
        logger.info("[MISSION] RTH: current mission item set to 0")

        # ── 4) Switch to AUTO mode ─────────────────────────────────────
        auto_ok = False
        for attempt in range(1, 4):
            if self.mavlink.set_mode("AUTO"):
                auto_ok = True
                break
            logger.warning(
                "[MISSION] RTH AUTO transition failed (%d/3); retrying...",
                attempt,
            )
            if not self._sleep(1.0):
                return

        if not auto_ok:
            raise RuntimeError("Could not enter AUTO for return-home mission")
        logger.info("[MISSION] RTH: AUTO mode active")

        # ── 5) MISSION_START ──────────────────────────────────────────────
        if not self._sleep(0.5):
            return

        mission_start_cmd = getattr(mavutil.mavlink, "MAV_CMD_MISSION_START", 300)
        if not self.mavlink.command_long(mission_start_cmd, 0, 0, 0, 0, 0, 0, 0):
            raise RuntimeError("Could not send MISSION_START for return-home")
        logger.info(
            "[MISSION] RTH: MISSION_START sent (command=%d)", mission_start_cmd
        )

        # Verify that AUTO remains active.
        if not self._sleep(0.5):
            return
        if not self.mavlink.set_mode("AUTO"):
            logger.warning("[MISSION] RTH: AUTO mode verification failed; continuing")

        # ── 6) Wait until GPS distance confirms arrival at HOME ────────
        self._state = AutonomousState.RETURN_HOME
        wp_deadline = time.monotonic() + 180.0
        logger.info(
            "[MISSION] RTH: flying to HOME waypoint | %.7f,%.7f at %.1f m",
            home_lat, home_lon, return_alt,
        )

        while not self._stop.is_set() and time.monotonic() < wp_deadline:
            with self._lock:
                pos = self._current_position

            if pos is not None:
                dist = distance(pos[0], pos[1], home_lat, home_lon)
                if dist <= float(HOME_ACCEPT_RADIUS_M) * 2.0:
                    logger.info(
                        "[MISSION] RTH: HOME waypoint reached | "
                        "GPS distance=%.1f m -> waiting for NAV_LAND",
                        dist,
                    )
                    break
            time.sleep(0.2)
        else:
            if self._stop.is_set():
                return
            logger.error(
                "[MISSION] RTH: HOME WP timeout (180 s) — "
                "continuing to wait for NAV_LAND"
            )

        # ── 7) Wait for NAV_LAND to complete (altitude <= 0.5 m) ───────
        self._state = AutonomousState.LANDING
        land_deadline = time.monotonic() + 120.0
        logger.info("[MISSION] RTH: landing in progress (NAV_LAND)...")

        while not self._stop.is_set() and time.monotonic() < land_deadline:
            with self._lock:
                alt = float(self._current_altitude)

            if alt <= 0.5:
                logger.info(
                    "[MISSION] RTH: landing complete | altitude=%.2f m", alt
                )
                break
            time.sleep(0.2)
        else:
            if not self._stop.is_set():
                logger.warning(
                    "[MISSION] RTH: landing timed out (120 s) — attempting DISARM anyway"
                )

        # ── 8) DISARM ─────────────────────────────────────────────────────
        if not self._stop.is_set():
            try:
                self.mavlink.disarm()
                logger.info("[MISSION] RTH: DISARM complete")
            except Exception:
                logger.exception("[MISSION] RTH: DISARM failed")

    def _point_in_polygon(self, lat: float, lon: float) -> bool:
        pts = [(float(a), float(b)) for a, b in self._mapping_polygon]
        n = len(pts)
        if n < 3:
            return False
        inside = False
        x, y = float(lon), float(lat)  # lon=x, lat=y
        j = n - 1
        for i in range(n):
            xi, yi = pts[i][1], pts[i][0]
            xj, yj = pts[j][1], pts[j][0]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside

    def _polygon_center(self) -> Coordinate:
        pts = [(float(a), float(b)) for a, b in self._mapping_polygon]
        if len(pts) < 3:
            return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))
        x = [p[1] for p in pts]  # lon
        y = [p[0] for p in pts]  # lat
        a2 = 0.0; cx = 0.0; cy = 0.0
        for i in range(len(pts)):
            j = (i + 1) % len(pts)
            cross = x[i] * y[j] - x[j] * y[i]
            a2 += cross; cx += (x[i] + x[j]) * cross; cy += (y[i] + y[j]) * cross
        if abs(a2) < 1e-15:
            return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))
        return float(cy / (3.0 * a2)), float(cx / (3.0 * a2))

    def _yolo_frame(self, frame: np.ndarray) -> np.ndarray:
        z = max(1.0, float(YOLO_CENTER_ZOOM))
        if z <= 1.000001:
            return frame
        h, w = frame.shape[:2]
        crop_w = max(1, int(round(w / z)))
        crop_h = max(1, int(round(h / z)))
        x1 = max(0, (w - crop_w) // 2); y1 = max(0, (h - crop_h) // 2)
        return frame[y1:y1+crop_h, x1:x1+crop_w]

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def state(self) -> AutonomousState:
        with self._lock:
            return self._state

    @property
    def phase(self) -> str:
        with self._lock:
            return self._phase
