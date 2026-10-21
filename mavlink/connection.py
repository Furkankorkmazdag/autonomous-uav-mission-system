"""Owns the MAVLink connection to the flight controller."""

from pymavlink import mavutil
import threading
import time

import config


class MAVLinkConnection:

    def __init__(self):

        self.master = None

        self.connected = False

        self.lock = threading.Lock()

    # --------------------------------------------------
    # Connect to the Pixhawk.
    # --------------------------------------------------

    def connect(self):

        while True:

            try:

                print(f"\n[MAVLINK] Connecting: {config.MAVLINK_CONNECTION}")

                master = mavutil.mavlink_connection(
                    config.MAVLINK_CONNECTION,
                    baud=config.BAUDRATE
                )

                print("[MAVLINK] Waiting for heartbeat...")

                master.wait_heartbeat(timeout=config.HEARTBEAT_TIMEOUT)

                print(
                    f"[MAVLINK] Connected "
                    f"(SYS={master.target_system} "
                    f"COMP={master.target_component})"
                )

                with self.lock:

                    self.master = master

                    self.connected = True

                return True

            except Exception as e:

                print(f"[MAVLINK] Connection error: {e}")

                self.connected = False

                time.sleep(config.RECONNECT_DELAY)

    # --------------------------------------------------
    # Check the connection.
    # --------------------------------------------------

    def is_connected(self):

        return self.connected

    # --------------------------------------------------
    # Return the master connection object.
    # --------------------------------------------------

    def get_master(self):

        with self.lock:

            return self.master

    # --------------------------------------------------
    # Close the connection.
    # --------------------------------------------------

    def close(self):

        with self.lock:

            if self.master is not None:

                try:

                    self.master.close()

                except:

                    pass

            self.connected = False

    # --------------------------------------------------
    # Check whether the connection has been lost.
    # --------------------------------------------------

    def monitor(self):

        while True:

            try:

                if self.master is None:

                    raise Exception("Master None")

                msg = self.master.recv_match(
                    type="HEARTBEAT",
                    blocking=True,
                    timeout=3
                )

                if msg is None:

                    raise Exception("Heartbeat Timeout")

            except Exception:

                print("[MAVLINK] Connection lost.")

                self.close()

                self.connect()

            time.sleep(1)
