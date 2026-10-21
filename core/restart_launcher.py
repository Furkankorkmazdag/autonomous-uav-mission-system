"""Helpers for safely disconnecting hardware and launching an optional restart process."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
import time

logger = logging.getLogger(__name__)

# Set to True when schedule_restart() is called.
# main.py checks this flag to avoid launching the restart script twice.
_restart_scheduled: bool = False


# ============================================================================
# DISCONNECT CAMERA AND PIXHAWK
# ============================================================================

def disconnect_camera_and_mavlink(app: "Any") -> None:
    shared = getattr(app, "shared", None)
    if shared is None:
        logger.warning("[RestartLauncher] disconnect_camera_and_mavlink: shared=None; skipping.")
        return

    # --- 1. AutonomousMission (also stops its camera thread) ---
    autonomous = getattr(shared, "autonomous_mission", None)
    if autonomous is not None:
        try:
            stopper = getattr(autonomous, "stop", None)
            if callable(stopper):
                stopper()
                logger.info("[RestartLauncher] AutonomousMission stopped (camera closed).")
        except Exception:
            logger.exception("[RestartLauncher] Could not stop AutonomousMission.")

    # --- 2. Separate camera service, if present ---
    camera = getattr(shared, "camera", None)
    if camera is not None:
        try:
            stopper = getattr(camera, "stop", None)
            if callable(stopper):
                stopper()
                logger.info("[RestartLauncher] Camera service stopped.")
        except Exception:
            logger.exception("[RestartLauncher] Could not stop the camera service.")

    # --- 3. MAVLink (flight-controller connection) ---
    mavlink = getattr(shared, "mavlink", None)
    if mavlink is not None:
        try:
            # Call stop first (receive loop and subsystems), then disconnect.
            stopper = getattr(mavlink, "stop", None)
            if callable(stopper):
                stopper()
                logger.info("[RestartLauncher] MAVLink stopped (flight-controller connection closed).")
            else:
                disconnecter = getattr(mavlink, "disconnect", None)
                if callable(disconnecter):
                    disconnecter()
                    logger.info("[RestartLauncher] MAVLink connection closed.")
        except Exception:
            logger.exception("[RestartLauncher] Could not stop MAVLink.")


def restart_already_scheduled() -> bool:
    return _restart_scheduled


# ============================================================================
# DETECT PYTHON EXECUTABLE
# ============================================================================

def _find_python() -> str:

    # 1. The current interpreter is always a valid candidate.
    if sys.executable:
        return sys.executable

    # 2. Use the active virtual environment, if one is configured.
    venv = os.environ.get("VIRTUAL_ENV")
    if venv:
        candidate = os.path.join(venv, "bin", "python3")
        if os.path.isfile(candidate):
            return candidate

    # 3. Fallback: the system python3 executable.
    return "python3"


# ============================================================================
# LAUNCHER
# ============================================================================

def launch_py_a8mini(script_path: str) -> bool:

    if not str(script_path).strip():
        logger.info("[RestartLauncher] Script path is empty; restart is disabled.")
        return False

    expanded = os.path.expanduser(script_path)

    if not os.path.isfile(expanded):
        logger.error(
            "[RestartLauncher] py_a8mini.py not found: %s", expanded
        )
        return False

    python = _find_python()

    try:
        proc = subprocess.Popen(
            [python, expanded],
            # Keep the helper process independent from the main process.
            start_new_session=True,
            # Inherit stdout/stderr; DEVNULL can be used if desired.
            stdin=subprocess.DEVNULL,
        )
        logger.info(
            "[RestartLauncher] py_a8mini.py started (PID=%d): %s",
            proc.pid,
            expanded,
        )
        return True

    except Exception:
        logger.exception(
            "[RestartLauncher] Could not start py_a8mini.py: %s", expanded
        )
        return False


# ============================================================================
# SCHEDULED RESTART
# ============================================================================

def schedule_restart(
    script_path: str,
    delay_s: float = 5.0,
):

    global _restart_scheduled
    if not str(script_path).strip():
        logger.info("[RestartLauncher] Script path is empty; restart is disabled.")
        return None

    _restart_scheduled = True

    expanded = os.path.expanduser(script_path)
    python = _find_python()
    if not os.path.isfile(expanded):
        logger.error("[RestartLauncher] Script not found: %s", expanded)
        return None
    # Use a separate helper process instead of a daemon thread, so the script can run
    # five seconds later even if the main process has exited.
    helper = (
        "import os,time,subprocess,sys; "
        f"time.sleep({max(0.0,float(delay_s))!r}); "
        f"subprocess.Popen([sys.executable, {expanded!r}], start_new_session=True, stdin=subprocess.DEVNULL)"
    )
    try:
        proc = subprocess.Popen(
            [python, "-c", helper],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        logger.info("[RestartLauncher] Independent restart helper started PID=%d | delay=%.1f s", proc.pid, float(delay_s))
        return proc
    except Exception:
        logger.exception("[RestartLauncher] Could not start the restart helper")
        return None
