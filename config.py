"""Central configuration for the SUAS autonomous UAV mission software.

Keep hardware-specific values here. The repository ships with autonomous
mission start disabled, empty routes, and hardware integrations disabled.
Validate every change in SITL before connecting to a real aircraft.
"""

from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# Startup and MAVLink
# ---------------------------------------------------------------------------

AUTO_START_MISSION = False
MAVLINK_CONNECTION = "udpin:127.0.0.1:14550"  # Local ArduPilot SITL endpoint
BAUDRATE = 460800
HEARTBEAT_TIMEOUT = 10.0
RECONNECT_DELAY = 3.0


# ---------------------------------------------------------------------------
# Flight profile and mission routes
# ---------------------------------------------------------------------------

TAKEOFF_ALTITUDE_M = 10.0
GUIDED_SPEED_MPS = 5.0
HOME_ACCEPT_RADIUS_M = 2.0
RETURN_HOME_ALTITUDE_M = TAKEOFF_ALTITUDE_M
MAX_FLIGHT_ALTITUDE_M = 30.0
MISSION_LOOP_RATE_HZ = 10

LAP_COUNT = 1
LAP_SPEED_MPS = 5.0
# Add approved test-area coordinates as (latitude, longitude) pairs.
LAP_WAYPOINTS = []

# Add at least three (latitude, longitude) points inside the approved test area.
MAPPING_POLYGON = []
MAPPING_ALTITUDE_M = 10.0
MAPPING_HORIZONTAL_FOV_DEG = 81.0
MAPPING_VERTICAL_FOV_DEG = 65.0
MAPPING_FORWARD_OVERLAP_RATIO = 0.70
MAPPING_SIDE_OVERLAP_RATIO = 0.65
MAPPING_SPEED_MPS = GUIDED_SPEED_MPS


# ---------------------------------------------------------------------------
# Object detection
# ---------------------------------------------------------------------------

# These labels must match the class names embedded in the trained model.
# The current model uses the labels "manken" and "tente".
TARGET_CLASS_NAMES = ("manken", "tente")
TARGET_CLASS_IDS = (None, None)  # Set IDs only when model class IDs are verified.
AUTO_ASSIGN_TARGET_CLASSES = False

YOLO_MODEL_PATH = "models/yolo_model.pt"
YOLO_ENABLED_DURING_MAPPING = True
YOLO_CONFIDENCE_THRESHOLD = 0.50
YOLO_IOU_THRESHOLD = 0.35
USE_CUDA = False
YOLO_HALF_PRECISION = False
YOLO_CENTER_ZOOM = 1.0
YOLO_SAHI_ROWS = 2
YOLO_SAHI_COLS = 2
YOLO_FRAME_SKIP = 4
TARGET_REQUIRED_CONSECUTIVE_DETECTIONS = 3
TARGET_DELIVERY_CONFIDENCE_THRESHOLD = 0.55
SKIP_UNCONFIRMED_TARGETS = True


# ---------------------------------------------------------------------------
# Target centering
# ---------------------------------------------------------------------------

TARGET_CENTER_TOLERANCE_NORMALIZED = 0.05
OBJECT_DETECTION_INTERVAL_S = 0.08
TARGET_CENTERING_MAX_SPEED_MPS = 1.0
TARGET_CENTERING_KP_X = 4.0
TARGET_CENTERING_KP_Y = 4.0
TARGET_CENTERING_DEADBAND = 0.02
TARGET_CENTERING_TIMEOUT_S = 5.0
CENTERING_STABLE_TIME_S = 1.0

# Pixhawk MAIN OUT channel numbers used by the target-centering mechanism.
MANNEQUIN_SERVO_CHANNEL_1 = 5
MANNEQUIN_SERVO_CHANNEL_2 = 6
CANOPY_SERVO_CHANNEL_1 = 7
CANOPY_SERVO_CHANNEL_2 = 8
CENTERING_SERVO_ACTIVATE_PWM = 2200
CENTERING_SERVO_RESET_PWM = 1000
CENTERING_SERVO_DELAY_S = 1.0
CENTERING_SERVO_HOLD_S = 1.0


# ---------------------------------------------------------------------------
# Payload release
# ---------------------------------------------------------------------------

# PWM values are in microseconds. Verify all channel mappings on the bench.
PAYLOAD_INITIAL_PWM = 1000
PAYLOAD_RESET_PWM = 1000
PAYLOAD_OPEN_PWM = 2200
PAYLOAD_TARGET_CHANNELS = {
    "manken": {"first_servo": 5, "second_servo": 6},
    "tente": {"first_servo": 7, "second_servo": 8},
}
PAYLOAD_SEQUENCE_DELAY_S = 0.0
PAYLOAD_OPEN_HOLD_S = 10.0
PAYLOAD_DELAY_S = 1.0

DELIVERY_APPROACH_ALTITUDE_M = TAKEOFF_ALTITUDE_M
DELIVERY_DROP_ALTITUDE_M = TAKEOFF_ALTITUDE_M
DELIVERY_ASCEND_ALTITUDE_M = TAKEOFF_ALTITUDE_M
DELIVERY_HOLD_S = 0.0
# Disabled by default: do not release payload if a target was not confirmed.
FALLBACK_DROP_ENABLED = False


# ---------------------------------------------------------------------------
# Camera and photo capture
# ---------------------------------------------------------------------------

CAMERA_SOURCE = 0  # 0 selects the default local camera; use an RTSP URL as needed.
DISPLAY_CAMERA_PREVIEW = False

GSTREAMER_FRAME_SAVE_DIR = "data/photos"
GSTREAMER_SAVE_ALL_FRAMES = False
GSTREAMER_JPEG_QUALITY = 85

PHOTO_CAPTURE_ENABLED = False
PHOTO_SAVE_PATH = "data/photos"
PHOTO_JPEG_QUALITY = 92
PHOTO_MIN_SPACING_M = 2.0
PHOTO_MAX_SPACING_M = 50.0
PHOTO_METADATA_ENABLED = True


# ---------------------------------------------------------------------------
# Copy mission data to removable storage
# ---------------------------------------------------------------------------

FLASH_PHOTO_COPY_ENABLED = False
FLASH_COPY_SOURCE_DIR = "data"
FLASH_COPY_DEST_DIR = str(Path(__file__).resolve().parent / "mission_data_export")
FLASH_PHOTO_COPY_RETRY = 3
FLASH_PHOTO_COPY_VERIFY = True
FLASH_PHOTO_COPY_WAIT_S = 120.0


# ---------------------------------------------------------------------------
# Optional SIYI A8 Mini camera integration
# ---------------------------------------------------------------------------

SIYI_A8_MINI_ENABLED = False
SIYI_A8_MINI_CAMERA_IP = "192.168.144.25"
SIYI_A8_MINI_RTSP_URL = (
    f"rtsp://{SIYI_A8_MINI_CAMERA_IP}:8554/main.264"
)
SIYI_A8_MINI_RTSP_LATENCY_MS = 200
SIYI_A8_MINI_UDP_PORT = 37260
SIYI_A8_MINI_GIMBAL_PITCH_DEG = -90.0
SIYI_A8_MINI_GIMBAL_YAW_DEG = 0.0
SIYI_A8_MINI_GIMBAL_SETTLE_S = 2.0
SIYI_A8_MINI_MAX_FRAME_FAILURES = 5
SIYI_A8_MINI_REBOOT_WAIT_S = 15.0
SIYI_A8_MINI_RECONNECT_DELAY_S = 3.0
SIYI_A8_MINI_MAX_REBOOT_ATTEMPTS = 3
SIYI_A8_MINI_PING_TIMEOUT_S = 1.0
SIYI_A8_MINI_RECORD_DIR = "data/recordings"
SIYI_A8_MINI_GSTREAMER_PLUGIN_PATH = (
    "/usr/lib/aarch64-linux-gnu/gstreamer-1.0:"
    "/usr/lib/x86_64-linux-gnu/gstreamer-1.0:"
    "/usr/lib/gstreamer-1.0"
)


# ---------------------------------------------------------------------------
# Safety shutdown and recovery
# ---------------------------------------------------------------------------

ARM_MAX_ATTEMPTS = 5
RC_SAFETY_CHANNEL = 6
RC16_SAFETY_THRESHOLD_PWM = 1800
SAFETY_RTL_WAIT_S = 10.0
RESTART_DELAY_S = 2.0
SAFETY_SCRIPT_PATH = ""


# ---------------------------------------------------------------------------
# Mission manager defaults
# ---------------------------------------------------------------------------

BATTERY_RTL_THRESHOLD_PERCENT = 25.0
BATTERY_CRITICAL_THRESHOLD_PERCENT = 15.0


@dataclass(frozen=True)
class ManagerConfig:
    """Runtime settings for mission scheduling, safety, and watchdogs."""

    priority_lap: int = 10
    priority_payload: int = 20
    default_lap_count: int = LAP_COUNT
    run_loop_interval_s: float = 1.0 / MISSION_LOOP_RATE_HZ
    subsystem_stop_timeout_s: float = 10.0
    thread_join_timeout_s: float = 5.0
    safety_poll_interval_s: float = 0.1
    battery_rtl_threshold_percent: float = BATTERY_RTL_THRESHOLD_PERCENT
    battery_critical_threshold_percent: float = BATTERY_CRITICAL_THRESHOLD_PERCENT
    heartbeat_timeout_s: float = HEARTBEAT_TIMEOUT
    watchdog_freeze_timeout_s: float = 5.0
    watchdog_poll_interval_s: float = 1.0
    watchdog_restart_delay_s: float = 1.0
    arm_max_attempts: int = ARM_MAX_ATTEMPTS
    rc_safety_channel: int = RC_SAFETY_CHANNEL
    rc16_safety_threshold_pwm: int = RC16_SAFETY_THRESHOLD_PWM
    safety_rtl_wait_s: float = SAFETY_RTL_WAIT_S
    safety_script_path: str = SAFETY_SCRIPT_PATH
    restart_delay_s: float = RESTART_DELAY_S
