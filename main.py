"""Entry point for the SUAS autonomous UAV application.

Autonomous startup is disabled by default. When enabled deliberately, the
application validates the mission configuration before importing or starting
hardware-dependent components.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from config import (
    AUTO_START_MISSION,
    LAP_WAYPOINTS,
    MAPPING_ALTITUDE_M,
    MAPPING_POLYGON,
    MAX_FLIGHT_ALTITUDE_M,
    RESTART_DELAY_S,
    SAFETY_SCRIPT_PATH,
    TAKEOFF_ALTITUDE_M,
    TARGET_CLASS_NAMES,
    USE_CUDA,
    YOLO_MODEL_PATH,
)
from core.restart_launcher import (
    disconnect_camera_and_mavlink,
    restart_already_scheduled,
    schedule_restart,
)


_RESTART_SCRIPT_PATH = SAFETY_SCRIPT_PATH.strip()
_restart_request_handled = False

LOG_DIR = Path(__file__).resolve().parent / "data" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / f"{datetime.now():%Y-%m-%d_%H-%M-%S}.log"

_file_handler = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
_file_handler.setLevel(logging.INFO)
_file_handler.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
)
_console_handler = logging.StreamHandler()
_console_handler.setLevel(logging.INFO)
_console_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

logging.basicConfig(
    level=logging.INFO,
    handlers=[_file_handler, _console_handler],
)
logger = logging.getLogger("app")
logging.getLogger("missions.autonomous_mission").setLevel(logging.INFO)


def _print_startup_banner(torch_module) -> None:
    """Print application and compute-device information after validation."""
    print("\n========================================")
    print("        SUAS AUTONOMOUS UAV")
    print("========================================")
    print(f"[INFO] Detailed log: {LOG_FILE.name}")

    gpu_available = torch_module.cuda.is_available()
    if USE_CUDA and gpu_available:
        device_message = (
            "[DEVICE] YOLO inference will use the GPU "
            f"({torch_module.cuda.get_device_name(0)})."
        )
    elif USE_CUDA:
        device_message = (
            "[DEVICE] USE_CUDA is enabled, but CUDA is unavailable; "
            "YOLO inference will use the CPU."
        )
    else:
        device_message = "[DEVICE] YOLO inference will use the CPU."

    print(device_message)
    logger.info(device_message)


def _schedule_configured_restart() -> None:
    """Schedule a restart only when an explicit script path is configured."""
    global _restart_request_handled
    if _restart_request_handled:
        return
    _restart_request_handled = True
    if not _RESTART_SCRIPT_PATH:
        logger.warning(
            "Automatic restart is disabled because SAFETY_SCRIPT_PATH is empty."
        )
        return
    schedule_restart(_RESTART_SCRIPT_PATH, delay_s=RESTART_DELAY_S)


def _validate_startup_configuration() -> None:
    """Reject incomplete or inconsistent settings before opening MAVLink."""
    errors: list[str] = []

    if len(LAP_WAYPOINTS) < 2:
        errors.append("LAP_WAYPOINTS must contain at least two GPS points.")
    if len(MAPPING_POLYGON) < 3:
        errors.append("MAPPING_POLYGON must contain at least three GPS points.")

    coordinates = list(LAP_WAYPOINTS) + list(MAPPING_POLYGON)
    for index, point in enumerate(coordinates, start=1):
        if not isinstance(point, (tuple, list)) or len(point) != 2:
            errors.append(f"GPS point #{index} must be a (latitude, longitude) pair.")
            continue
        try:
            latitude, longitude = float(point[0]), float(point[1])
        except (TypeError, ValueError):
            errors.append(f"GPS point #{index} must contain numeric coordinates.")
            continue
        if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
            errors.append(f"GPS point #{index} is outside the valid coordinate range.")

    if not TARGET_CLASS_NAMES:
        errors.append("TARGET_CLASS_NAMES must not be empty.")
    if TAKEOFF_ALTITUDE_M <= 0 or MAPPING_ALTITUDE_M <= 0:
        errors.append("Takeoff and mapping altitudes must be greater than zero.")
    if max(TAKEOFF_ALTITUDE_M, MAPPING_ALTITUDE_M) > MAX_FLIGHT_ALTITUDE_M:
        errors.append("Takeoff or mapping altitude exceeds MAX_FLIGHT_ALTITUDE_M.")

    model_path = Path(YOLO_MODEL_PATH)
    if not model_path.is_absolute():
        model_path = Path(__file__).resolve().parent / model_path
    if not model_path.is_file():
        errors.append(f"YOLO model file was not found: {model_path}")

    if errors:
        raise ValueError("\n- " + "\n- ".join(errors))


def main() -> int:
    # The default path must exit before importing hardware- or AI-dependent modules.
    if not AUTO_START_MISSION:
        logger.warning(
            "Autonomous startup is disabled. Configure config.py, complete SITL "
            "testing, and set AUTO_START_MISSION=True only when intentionally enabled."
        )
        return 0

    try:
        _validate_startup_configuration()
    except (TypeError, ValueError) as exc:
        logger.error("Invalid startup configuration: %s", exc)
        return 2

    try:
        import torch
        from app import Application
    except ImportError as exc:
        logger.error("A required runtime dependency is missing: %s", exc)
        return 3

    _print_startup_banner(torch)
    app = Application()

    if not app.start():
        logger.error("Application failed to start.")
        disconnect_camera_and_mavlink(app)
        _schedule_configured_restart()
        return 1

    shared = app.shared
    if shared is None or shared.mavlink is None:
        logger.error("MAVLink service is unavailable.")
        disconnect_camera_and_mavlink(app)
        app.stop()
        _schedule_configured_restart()
        return 1

    if getattr(shared, "autonomous_mission", None) is None:
        logger.error("AutonomousMission has not been configured.")
        disconnect_camera_and_mavlink(app)
        app.stop()
        _schedule_configured_restart()
        return 1

    logger.info("SUAS flight computer started in autonomous mode.")
    logger.info("Mission sequence: HOME -> LAP -> MAPPING -> TARGETS -> HOME -> LAND.")

    try:
        while True:
            if app.wait(timeout_s=1.0):
                break
            mission = getattr(app.shared, "autonomous_mission", None)
            mission_thread = getattr(mission, "_thread", None)
            if mission_thread is not None and not mission_thread.is_alive():
                logger.info("Autonomous mission thread has completed.")
                break

    except (KeyboardInterrupt, EOFError):
        logger.info("Manual interruption received.")
        disconnect_camera_and_mavlink(app)
        _schedule_configured_restart()

    except Exception:
        logger.exception("Unexpected application error.")
        disconnect_camera_and_mavlink(app)
        _schedule_configured_restart()

    finally:
        app.stop()
        if not restart_already_scheduled():
            logger.info("Application stopped; closing camera and MAVLink connections.")
            disconnect_camera_and_mavlink(app)
            _schedule_configured_restart()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
