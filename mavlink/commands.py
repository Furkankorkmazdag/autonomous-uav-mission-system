"""Definitions and helpers for MAVLink commands."""

import time
from pymavlink import mavutil


class Commands:

    def __init__(self, connection):

        self.connection = connection

    # --------------------------------------------------
    # Master
    # --------------------------------------------------

    def master(self):

        return self.connection.get_master()

    # --------------------------------------------------
    # ARM
    # --------------------------------------------------

    def arm(self):

        master = self.master()

        print("[COMMAND] ARM")

        master.arducopter_arm()

        master.motors_armed_wait()

    # --------------------------------------------------
    # DISARM
    # --------------------------------------------------

    def disarm(self):

        master = self.master()

        print("[COMMAND] DISARM")

        master.arducopter_disarm()

        master.motors_disarmed_wait()

    # --------------------------------------------------
    # MODE
    # --------------------------------------------------

    def set_mode(self, mode):

        master = self.master()

        mode_id = master.mode_mapping()[mode]

        master.mav.set_mode_send(

            master.target_system,

            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,

            mode_id

        )

        print(f"[COMMAND] MODE -> {mode}")

    # --------------------------------------------------
    # TAKEOFF
    # --------------------------------------------------

    def takeoff(self, altitude):

        master = self.master()

        print(f"[COMMAND] TAKEOFF {altitude} m")

        master.mav.command_long_send(

            master.target_system,

            master.target_component,

            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,

            0,

            0,

            0,

            0,

            0,

            0,

            0,

            altitude

        )

    # --------------------------------------------------
    # LAND
    # --------------------------------------------------

    def land(self):

        master = self.master()

        print("[COMMAND] LAND")

        master.mav.command_long_send(

            master.target_system,

            master.target_component,

            mavutil.mavlink.MAV_CMD_NAV_LAND,

            0,

            0,

            0,

            0,

            0,

            0,

            0,

            0

        )

    # --------------------------------------------------
    # RTL
    # --------------------------------------------------

    def rtl(self):

        print("[COMMAND] RTL")

        self.set_mode("RTL")

    # --------------------------------------------------
    # GUIDED
    # --------------------------------------------------

    def guided(self):

        print("[COMMAND] GUIDED")

        self.set_mode("GUIDED")

    # --------------------------------------------------
    # AUTO
    # --------------------------------------------------

    def auto(self):

        print("[COMMAND] AUTO")

        self.set_mode("AUTO")

    # --------------------------------------------------
    # LOITER
    # --------------------------------------------------

    def loiter(self):

        print("[COMMAND] LOITER")

        self.set_mode("LOITER")

    # --------------------------------------------------
    # GOTO
    # --------------------------------------------------

    def goto(self,
             latitude,
             longitude,
             altitude):

        master = self.master()

        master.mav.set_position_target_global_int_send(

            0,

            master.target_system,

            master.target_component,

            mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,

            int(0b110111111000),

            int(latitude * 1e7),

            int(longitude * 1e7),

            altitude,

            0,

            0,

            0,

            0,

            0,

            0,

            0,

            0

        )

    # --------------------------------------------------
    # SPEED
    # --------------------------------------------------

    def set_speed(self, speed):

        master = self.master()

        master.mav.command_long_send(

            master.target_system,

            master.target_component,

            mavutil.mavlink.MAV_CMD_DO_CHANGE_SPEED,

            0,

            1,

            speed,

            -1,

            0,

            0,

            0,

            0

        )

    # --------------------------------------------------
    # YAW
    # --------------------------------------------------

    def set_yaw(self,
                heading,
                clockwise=True):

        master = self.master()

        direction = 1 if clockwise else -1

        master.mav.command_long_send(

            master.target_system,

            master.target_component,

            mavutil.mavlink.MAV_CMD_CONDITION_YAW,

            0,

            heading,

            20,

            direction,

            1,

            0,

            0,

            0

        )

    # --------------------------------------------------
    # SERVO
    # --------------------------------------------------

    def set_servo(self,
                  channel,
                  pwm):

        master = self.master()

        master.mav.command_long_send(

            master.target_system,

            master.target_component,

            mavutil.mavlink.MAV_CMD_DO_SET_SERVO,

            0,

            channel,

            pwm,

            0,

            0,

            0,

            0,

            0

        )

    # --------------------------------------------------
    # DELAY
    # --------------------------------------------------

    @staticmethod
    def sleep(seconds):

        time.sleep(seconds)
