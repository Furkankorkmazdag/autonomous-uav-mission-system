"""Thread-safe shared state for telemetry, vehicle status, and mission coordination."""

from __future__ import annotations

import threading
import time
from copy import deepcopy
from typing import Any, Dict, Optional


from core.state import VehicleState
from core.events import EventBus

from perception.frame_buffer import FrameBuffer

from mavlink.connection import MAVLinkConnection
from mavlink.commands import Commands


class SharedData:

    def __init__(self):

        # ============================================================
        # LOCK
        # ============================================================

        # Telemetry may be read and written by multiple threads.
        self._telemetry_lock = threading.RLock()

        # ============================================================
        # CORE
        # ============================================================

        self.state = VehicleState()

        self.events = EventBus()

        self.frame_buffer = FrameBuffer()

        # ============================================================
        # TELEMETRY
        # ============================================================

        """
        Central live telemetry interface.

        Updated by MAVLinkManager.

        Example structure:

        telemetry = {
            "gps": {
                "lat": 0.0,
                "lon": 0.0,
                "alt": 0.0,
                "fix_type": 0,
                "satellites": 0,
            },

            "battery": {
                "voltage": 0.0,
                "current": 0.0,
                "remaining": -1,
            },

            "attitude": {
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
            },

            "velocity": {
                "vx": 0.0,
                "vy": 0.0,
                "vz": 0.0,
                "ground_speed": 0.0,
            },

            "heartbeat": {
                "last_time": 0.0,
            },

            "connection": {
                "connected": False,
            },

            "ekf": {
                "healthy": True,
            },
        }
        """

        self.telemetry: Dict[str, Any] = {

            "gps": {
                "lat": None,
                "lon": None,
                "alt": None,
                "fix_type": 0,
                "satellites": 0,
            },

            "battery": {
                "voltage": None,
                "current": None,
                "remaining": None,
            },

            "attitude": {
                "roll": None,
                "pitch": None,
                "yaw": None,
            },

            "velocity": {
                "vx": None,
                "vy": None,
                "vz": None,
                "ground_speed": None,
            },

            "heartbeat": {
                "last_time": None,
            },

            "connection": {
                "connected": False,
            },

            "ekf": {
                "healthy": True,
            },
        }

        # Time of the most recent telemetry update.
        self._last_telemetry_time: Optional[float] = None

        # Used directly by MissionManager:
        # timestamp of the last heartbeat.
        #
        # Initialize this value in __init__.
        # Otherwise, MissionManager may raise an
        # AttributeError before the first heartbeat arrives.
        self.last_heartbeat_time: float = 0.0

        self.connection_active: bool = False

        # ============================================================
        # MAVLINK
        # ============================================================

        """Compatibility adapter for the legacy SharedData telemetry structure."""

        self.connection = MAVLinkConnection()

        self.commands = Commands(
            self.connection
        )

        # ============================================================
        # NAVIGATION
        # ============================================================

        self.navigator = None

        # ============================================================
        # MISSIONS
        # ============================================================

        self.lap = None

        self.mapping = None

        self.object_detection = None

        self.payload = None

        self.mission_manager = None

        # ============================================================
        # AI
        # ============================================================

        self.detector = None

        self.tracker = None

        # ============================================================
        # CAMERA
        # ============================================================

        self.camera = None

    # ================================================================
    # TELEMETRY
    # ================================================================

    def update_telemetry(
        self,
        data: Dict[str, Any],
    ) -> None:

        if not isinstance(data, dict):
            raise TypeError(
                "Telemetry data must be a dictionary"
            )

        with self._telemetry_lock:

            self._deep_update(
                self.telemetry,
                data,
            )

            self._last_telemetry_time = time.time()

    # ================================================================
    # TELEMETRY READ
    # ================================================================

    def get_telemetry(self) -> Dict[str, Any]:

        with self._telemetry_lock:

            return deepcopy(
                self.telemetry
            )

    # ================================================================
    # TELEMETRY AGE
    # ================================================================

    def get_last_telemetry_time(
        self,
    ) -> Optional[float]:

        with self._telemetry_lock:

            return self._last_telemetry_time

    # ================================================================
    # TELEMETRY AGE
    # ================================================================

    def get_telemetry_age(
        self,
    ) -> Optional[float]:

        with self._telemetry_lock:

            if self._last_telemetry_time is None:

                return None

            return (
                time.time()
                - self._last_telemetry_time
            )

    # ================================================================
    # CONNECTION
    # ================================================================

    def set_connection_status(
        self,
        connected: bool,
    ) -> None:

        with self._telemetry_lock:

            connected = bool(connected)

            self.connection_active = connected

            self.telemetry["connection"]["connected"] = connected

            self._last_telemetry_time = time.time()

    # ================================================================
    # HEARTBEAT
    # ================================================================

    def update_heartbeat(
        self,
        timestamp: Optional[float] = None,
    ) -> None:

        if timestamp is None:
            timestamp = time.time()

        with self._telemetry_lock:

            # MissionManager reads this value directly.
            self.last_heartbeat_time = timestamp

            self.connection_active = True

            # Update the telemetry view as well.
            self.telemetry["heartbeat"]["last_time"] = timestamp

            # Connection is active.
            self.telemetry["connection"]["connected"] = True

            # Timestamp for the latest telemetry update.
            self._last_telemetry_time = timestamp

    # ================================================================
    # GPS
    # ================================================================

    def update_gps(
        self,
        *,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        alt: Optional[float] = None,
        fix_type: Optional[int] = None,
        satellites: Optional[int] = None,
    ) -> None:

        data = {
            "gps": {}
        }

        if lat is not None:
            data["gps"]["lat"] = lat

        if lon is not None:
            data["gps"]["lon"] = lon

        if alt is not None:
            data["gps"]["alt"] = alt

        if fix_type is not None:
            data["gps"]["fix_type"] = fix_type

        if satellites is not None:
            data["gps"]["satellites"] = satellites

        self.update_telemetry(
            data
        )

    # ================================================================
    # BATTERY
    # ================================================================

    def update_battery(
        self,
        *,
        voltage: Optional[float] = None,
        current: Optional[float] = None,
        remaining: Optional[float] = None,
    ) -> None:

        data = {
            "battery": {}
        }

        if voltage is not None:
            data["battery"]["voltage"] = voltage

        if current is not None:
            data["battery"]["current"] = current

        if remaining is not None:
            data["battery"]["remaining"] = remaining

        self.update_telemetry(
            data
        )

    # ================================================================
    # ATTITUDE
    # ================================================================

    def update_attitude(
        self,
        *,
        roll: Optional[float] = None,
        pitch: Optional[float] = None,
        yaw: Optional[float] = None,
    ) -> None:

        data = {
            "attitude": {}
        }

        if roll is not None:
            data["attitude"]["roll"] = roll

        if pitch is not None:
            data["attitude"]["pitch"] = pitch

        if yaw is not None:
            data["attitude"]["yaw"] = yaw

        self.update_telemetry(
            data
        )

    # ================================================================
    # VELOCITY
    # ================================================================

    def update_velocity(
        self,
        *,
        vx: Optional[float] = None,
        vy: Optional[float] = None,
        vz: Optional[float] = None,
        ground_speed: Optional[float] = None,
    ) -> None:

        data = {
            "velocity": {}
        }

        if vx is not None:
            data["velocity"]["vx"] = vx

        if vy is not None:
            data["velocity"]["vy"] = vy

        if vz is not None:
            data["velocity"]["vz"] = vz

        if ground_speed is not None:
            data["velocity"][
                "ground_speed"
            ] = ground_speed

        self.update_telemetry(
            data
        )

    # ================================================================
    # EKF
    # ================================================================

    def update_ekf_health(
        self,
        healthy: bool,
    ) -> None:

        self.update_telemetry(
            {
                "ekf": {
                    "healthy": bool(healthy)
                }
            }
        )

    # ================================================================
    # GENERIC DEEP UPDATE
    # ================================================================

    @staticmethod
    def _deep_update(
        target: Dict[str, Any],
        source: Dict[str, Any],
    ) -> None:

        for key, value in source.items():

            if (
                key in target
                and isinstance(
                    target[key],
                    dict,
                )
                and isinstance(
                    value,
                    dict,
                )
            ):

                SharedData._deep_update(
                    target[key],
                    value,
                )

            else:

                target[key] = value

    # ================================================================
    # NAVIGATION INITIALIZATION
    # ================================================================

    def initialize_navigation(
        self,
        navigator,
    ) -> None:

        self.navigator = navigator

    # ================================================================
    # CAMERA INITIALIZATION
    # ================================================================

    def initialize_camera(
        self,
        camera,
    ) -> None:

        self.camera = camera

    # ================================================================
    # MISSION MANAGER INITIALIZATION
    # ================================================================

    def initialize_mission_manager(
        self,
        manager,
    ) -> None:

        self.mission_manager = manager

    # ================================================================
    # LAP INITIALIZATION
    # ================================================================

    def initialize_lap(
        self,
        lap,
    ) -> None:

        self.lap = lap

    # ================================================================
    # MAPPING INITIALIZATION
    # ================================================================

    def initialize_mapping(
        self,
        mapping,
    ) -> None:

        self.mapping = mapping

    # ================================================================
    # OBJECT DETECTION INITIALIZATION
    # ================================================================

    def initialize_object_detection(
        self,
        mission,
    ) -> None:

        self.object_detection = mission

    # ================================================================
    # PAYLOAD INITIALIZATION
    # ================================================================

    def initialize_payload(
        self,
        payload,
    ) -> None:

        self.payload = payload

    # ================================================================
    # DETECTOR INITIALIZATION
    # ================================================================

    def initialize_detector(
        self,
        detector,
    ) -> None:

        self.detector = detector

    # ================================================================
    # TRACKER INITIALIZATION
    # ================================================================

    def initialize_tracker(
        self,
        tracker,
    ) -> None:

        self.tracker = tracker
