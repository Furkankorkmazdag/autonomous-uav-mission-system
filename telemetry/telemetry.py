"""Reads MAVLink telemetry and updates the shared vehicle state."""

import time
from pymavlink import mavutil


class Telemetry:

    def __init__(self, connection, state):

        self.connection = connection
        self.state = state

        self.running = False

    # --------------------------------------------

    def start(self):

        self.running = True

        master = self.connection.get_master()

        while self.running:

            try:

                msg = master.recv_match(
                    blocking=True,
                    timeout=1
                )

                if msg is None:
                    continue

                msg_type = msg.get_type()

                # ---------------------------------
                # HEARTBEAT
                # ---------------------------------

                if msg_type == "HEARTBEAT":

                    mode = mavutil.mode_string_v10(msg)

                    armed = bool(
                        msg.base_mode &
                        mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
                    )

                    self.state.update_mode(
                        mode,
                        armed,
                        str(msg.system_status)
                    )

                    self.state.connected = True
                    self.state.last_heartbeat = time.time()

                # ---------------------------------
                # GLOBAL POSITION
                # ---------------------------------

                elif msg_type == "GLOBAL_POSITION_INT":

                    self.state.latitude = msg.lat / 1e7

                    self.state.longitude = msg.lon / 1e7

                    self.state.altitude = msg.alt / 1000

                    self.state.relative_altitude = msg.relative_alt / 1000

                    self.state.heading = msg.hdg / 100 if msg.hdg != 65535 else 0

                # ---------------------------------
                # ATTITUDE
                # ---------------------------------

                elif msg_type == "ATTITUDE":

                    self.state.roll = msg.roll

                    self.state.pitch = msg.pitch

                    self.state.yaw = msg.yaw

                # ---------------------------------
                # GPS
                # ---------------------------------

                elif msg_type == "GPS_RAW_INT":

                    self.state.fix_type = msg.fix_type

                    self.state.satellites = msg.satellites_visible

                # ---------------------------------
                # BATTERY
                # ---------------------------------

                elif msg_type == "SYS_STATUS":

                    self.state.voltage = msg.voltage_battery / 1000

                    self.state.current = msg.current_battery / 100

                    self.state.battery_remaining = msg.battery_remaining

                # ---------------------------------
                # SPEED
                # ---------------------------------

                elif msg_type == "VFR_HUD":

                    self.state.air_speed = msg.airspeed

                    self.state.ground_speed = msg.groundspeed

                    self.state.vertical_speed = msg.climb

                # ---------------------------------
                # CURRENT WAYPOINT
                # ---------------------------------

                elif msg_type == "MISSION_CURRENT":

                    self.state.current_waypoint = msg.seq

                # ---------------------------------
                # STATUSTEXT
                # ---------------------------------

                elif msg_type == "STATUSTEXT":

                    print("[PIXHAWK]", msg.text)

                # ---------------------------------
                # RC CHANNELS (RC16 safety)
                # ---------------------------------

                elif msg_type == "RC_CHANNELS":

                    # Read the monitored channel from config.
                    # Read chanN_raw according to config.RC_SAFETY_CHANNEL.
                    # Valid channel range is 1-18; store 0 for an out-of-range value.
                    import config as _rc_cfg
                    _ch = int(getattr(_rc_cfg, "RC_SAFETY_CHANNEL", 6))
                    _attr = f"chan{_ch}_raw" if 1 <= _ch <= 18 else None
                    _pwm = int(getattr(msg, _attr, 0)) if _attr else 0
                    self.state.rc_safety_pwm = _pwm
                    self.state.rc16_pwm = _pwm  # Backward compatibility.

            except Exception as e:

                print("Telemetry Error:", e)

                self.state.connected = False

                time.sleep(1)

    # --------------------------------------------

    def stop(self):

        self.running = False
