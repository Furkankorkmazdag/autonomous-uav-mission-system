"""Data structures for the vehicle, telemetry, and mission state."""

import threading
import time


class VehicleState:

    def __init__(self):

        self.lock = threading.RLock()

        # -------------------------------------------------
        # Connection
        # -------------------------------------------------

        self.connected = False

        self.last_heartbeat = 0

        # -------------------------------------------------
        # GPS
        # -------------------------------------------------

        self.latitude = 0.0

        self.longitude = 0.0

        self.altitude = 0.0

        self.relative_altitude = 0.0

        self.fix_type = 0

        self.satellites = 0

        # -------------------------------------------------
        # Hareket
        # -------------------------------------------------

        self.ground_speed = 0.0

        self.air_speed = 0.0

        self.vertical_speed = 0.0

        self.heading = 0.0

        # -------------------------------------------------
        # Attitude
        # -------------------------------------------------

        self.roll = 0.0

        self.pitch = 0.0

        self.yaw = 0.0

        # -------------------------------------------------
        # Vehicle State
        # -------------------------------------------------

        self.mode = "UNKNOWN"

        self.armed = False

        self.system_status = ""

        # -------------------------------------------------
        # Batarya
        # -------------------------------------------------

        self.voltage = 0.0

        self.current = 0.0

        self.battery_remaining = 0

        # -------------------------------------------------
        # Mission
        # -------------------------------------------------

        self.current_waypoint = 0

        self.total_waypoints = 0

        self.lap = 0

        self.mission_finished = False

        # -------------------------------------------------
        # Camera
        # -------------------------------------------------

        self.frame = None

        self.frame_time = 0

        # -------------------------------------------------
        # AI
        # -------------------------------------------------

        self.target_detected = False

        self.target_class = ""

        self.target_lat = 0.0

        self.target_lon = 0.0

        self.target_confidence = 0.0

        # -------------------------------------------------
        # Payload
        # -------------------------------------------------

        self.payload_dropped = False

        # -------------------------------------------------
        # RC Safety Channel
        # -------------------------------------------------

        # Current PWM value of the channel selected by config.RC_SAFETY_CHANNEL.
        # Updated by the telemetry thread and read by SafetyMonitor.
        # rc16_pwm is the legacy name; both attributes hold the same value.
        self.rc_safety_pwm: int = 0
        self.rc16_pwm: int = 0  # Backward compatibility.

        # -------------------------------------------------
        # System
        # -------------------------------------------------

        self.start_time = time.time()

    # =====================================================
    # GPS
    # =====================================================

    def update_gps(self,
                   lat,
                   lon,
                   alt,
                   rel_alt,
                   fix,
                   sats):

        with self.lock:

            self.latitude = lat

            self.longitude = lon

            self.altitude = alt

            self.relative_altitude = rel_alt

            self.fix_type = fix

            self.satellites = sats

    # =====================================================
    # Attitude
    # =====================================================

    def update_attitude(self,
                        roll,
                        pitch,
                        yaw):

        with self.lock:

            self.roll = roll

            self.pitch = pitch

            self.yaw = yaw

    # =====================================================
    # Speed
    # =====================================================

    def update_speed(self,
                     ground,
                     air,
                     vertical):

        with self.lock:

            self.ground_speed = ground

            self.air_speed = air

            self.vertical_speed = vertical

    # =====================================================
    # Heading
    # =====================================================

    def update_heading(self, heading):

        with self.lock:

            self.heading = heading

    # =====================================================
    # Mode
    # =====================================================

    def update_mode(self,
                    mode,
                    armed,
                    status):

        with self.lock:

            self.mode = mode

            self.armed = armed

            self.system_status = status

    # =====================================================
    # Batarya
    # =====================================================

    def update_battery(self,
                       voltage,
                       current,
                       remaining):

        with self.lock:

            self.voltage = voltage

            self.current = current

            self.battery_remaining = remaining

    # =====================================================
    # Camera
    # =====================================================

    def update_frame(self,
                     frame):

        with self.lock:

            self.frame = frame

            self.frame_time = time.time()

    # =====================================================
    # AI
    # =====================================================

    def update_target(self,
                      detected,
                      cls,
                      lat,
                      lon,
                      conf):

        with self.lock:

            self.target_detected = detected

            self.target_class = cls

            self.target_lat = lat

            self.target_lon = lon

            self.target_confidence = conf

    # =====================================================
    # Mission
    # =====================================================

    def update_waypoint(self,
                        current,
                        total):

        with self.lock:

            self.current_waypoint = current

            self.total_waypoints = total

    # =====================================================
    # Payload
    # =====================================================

    def payload_release(self):

        with self.lock:

            self.payload_dropped = True
