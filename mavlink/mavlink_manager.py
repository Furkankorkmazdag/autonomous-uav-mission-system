"""Central owner of MAVLink I/O, command handling, telemetry routing, and mission protocol services."""

from __future__ import annotations

import logging
import threading
import time

from dataclasses import dataclass
from queue import Queue, Empty
from typing import Any, Callable, Optional

from pymavlink import mavutil

from .message_router import MessageRouter
from .command_router import CommandRouter
from .mission_protocol import MissionProtocol
from .parameter_manager import ParameterManager
from .transaction import TransactionManager


logger = logging.getLogger(__name__)


# ============================================================================
# Connection State
# ============================================================================

@dataclass(frozen=True)
class MAVLinkState:

    connected: bool

    heartbeat_received: bool

    target_system: int

    target_component: int

    last_heartbeat: Optional[float]

    reconnect_count: int

    receive_loop_running: bool


# ============================================================================
# MAVLink Manager
# ============================================================================

class MAVLinkManager:

    name = "MAVLinkManager"

    # ========================================================================
    # Constructor
    # ========================================================================

    def __init__(
        self,
        connection_string: str,
        baudrate: int = 57600,
        *,
        source_system: int = 255,
        source_component: int = 0,
        reconnect_enabled: bool = True,
        reconnect_delay_s: float = 2.0,
        heartbeat_timeout_s: float = 3.0,
        receive_timeout_s: float = 0.05,
        parameter_timeout_s: float = 3.0,
        mission_timeout_s: float = 5.0,
        # Timeout for COMMAND_ACK during ARM and flight-mode changes.
        # Usually 1-2 seconds is sufficient in SITL; allow 5 seconds for a real Pixhawk.
        command_ack_timeout_s: float = 10.0,
        # Optional EventBus used for the ARM_MAX_ATTEMPTS_EXCEEDED event.
        event_bus: object = None,
        # Used to write RC_CHANNELS data to shared_data.state.rc_safety_pwm.
        shared_data: object = None,
    ) -> None:

        if not connection_string:
            raise ValueError("connection_string cannot be empty")

        if baudrate <= 0:
            raise ValueError("baudrate must be greater than zero")

        if reconnect_delay_s <= 0:
            raise ValueError("reconnect_delay_s must be greater than zero")

        if heartbeat_timeout_s <= 0:
            raise ValueError("heartbeat_timeout_s must be greater than zero")

        if receive_timeout_s <= 0:
            raise ValueError("receive_timeout_s must be greater than zero")

        self.connection_string = connection_string
        self.baudrate = int(baudrate)
        self.source_system = int(source_system)
        self.source_component = int(source_component)
        self.reconnect_enabled = bool(reconnect_enabled)
        self.reconnect_delay_s = float(reconnect_delay_s)
        self.heartbeat_timeout_s = float(heartbeat_timeout_s)
        self.receive_timeout_s = float(receive_timeout_s)
        self.command_ack_timeout_s = float(command_ack_timeout_s)

        # ====================================================================
        # Internal state
        # ====================================================================

        self._master: Optional[Any] = None
        self._target_system = 0
        self._target_component = 0
        self._connected = False
        self._heartbeat_received = False
        self._last_heartbeat: Optional[float] = None
        self._reconnect_count = 0
        self._running = False
        self._event_bus = event_bus  # Optional EventBus for ARM failure events.
        self._shared_data = shared_data  # SharedData | None — for RC_CHANNELS

        # Last custom_mode read from HEARTBEAT (ArduPilot mode ID).
        self._last_heartbeat_custom_mode: Optional[int] = None
        self._last_heartbeat_base_mode: Optional[int] = None

        self._receive_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._connection_lock = threading.RLock()
        self._state_lock = threading.RLock()

        # ====================================================================
        # Central subsystems
        # ====================================================================

        self.message_router = MessageRouter()

        self.transaction_manager = TransactionManager(
            default_timeout_s=mission_timeout_s
        )

        self.command_router = CommandRouter(self)

        self.mission_protocol = MissionProtocol(
            self,
            timeout_s=mission_timeout_s,
        )

        self.parameter_manager = ParameterManager(
            self,
            timeout_s=parameter_timeout_s,
        )

        # ====================================================================
        # External callbacks
        # ====================================================================

        self._connection_callbacks: list[Callable[[bool], None]] = []

        # ====================================================================
        # Internal subscriptions
        # ====================================================================

        self._parameter_subscription_id = self.message_router.subscribe(
            "PARAM_VALUE",
            self.parameter_manager.handle_message,
            owner="ParameterManager",
        )

        self._heartbeat_subscription_id = self.message_router.subscribe(
            "HEARTBEAT",
            self._handle_heartbeat,
            owner="MAVLinkManager",
        )

        self._statustext_cache: tuple = ()   # (monotonic_ts, text)
        self._statustext_subscription_id = self.message_router.subscribe(
            "STATUSTEXT",
            self._handle_statustext,
            owner="MAVLinkManager",
        )

        # RC_CHANNELS -> shared_data.state.rc_safety_pwm
        self._rc_channels_subscription_id = self.message_router.subscribe(
            "RC_CHANNELS",
            self._handle_rc_channels,
            owner="MAVLinkManager",
        )

    # ========================================================================
    # Start
    # ========================================================================

    def start(self) -> bool:
        """Start the component or operation."""

        with self._state_lock:
            if self._running:
                return True
            self._running = True
            self._stop_event.clear()

        self.message_router.start()
        self.transaction_manager.start()
        self.command_router.start()
        self.mission_protocol.start()
        self.parameter_manager.start()

        if not self.connect():
            if not self.reconnect_enabled:
                self.stop()
                return False
            logger.warning(
                "[MAVLinkManager] "
                "Initial connection failed. "
                "Reconnect loop will continue."
            )

        self._receive_thread = threading.Thread(
            target=self._receive_loop,
            name="MAVLinkReceiveLoop",
            daemon=True,
        )
        self._receive_thread.start()

        logger.info("[MAVLinkManager] Started")
        return True

    # ========================================================================
    # Stop
    # ========================================================================

    def stop(self) -> None:
        """Stop the component or operation safely."""

        with self._state_lock:
            if not self._running:
                return
            self._running = False

        self._stop_event.set()

        thread = self._receive_thread
        if (
            thread is not None
            and thread.is_alive()
            and thread is not threading.current_thread()
        ):
            thread.join(timeout=3.0)

        self._receive_thread = None

        self.disconnect()

        self.parameter_manager.stop()
        self.mission_protocol.stop()
        self.command_router.stop()
        self.transaction_manager.stop()
        self.message_router.stop()

        logger.info("[MAVLinkManager] Stopped")

    # ========================================================================
    # Connect
    # ========================================================================

    def connect(self) -> bool:

        with self._connection_lock:
            if self._connected:
                return True

            try:
                logger.info(
                    "[MAVLinkManager] Connecting: %s",
                    self.connection_string,
                )

                self._master = mavutil.mavlink_connection(
                    self.connection_string,
                    baud=self.baudrate,
                    source_system=self.source_system,
                    source_component=self.source_component,
                    autoreconnect=False,
                )

                heartbeat = self._master.wait_heartbeat(
                    timeout=self.heartbeat_timeout_s
                )

                if heartbeat is None:
                    logger.warning("[MAVLinkManager] Heartbeat timeout")
                    self._close_master()
                    self._set_connected(False)
                    return False

                self._target_system = int(self._master.target_system)
                self._target_component = int(self._master.target_component)

                with self._state_lock:
                    self._connected = True
                    self._heartbeat_received = True
                    self._last_heartbeat = time.monotonic()

                logger.info(
                    "[MAVLinkManager] Connected system=%d component=%d",
                    self._target_system,
                    self._target_component,
                )

                self._notify_connection(True)
                return True

            except Exception:
                logger.exception("[MAVLinkManager] Connection failed")
                self._close_master()
                self._set_connected(False)
                return False

    # ========================================================================
    # Disconnect
    # ========================================================================

    def disconnect(self) -> None:

        with self._connection_lock:
            was_connected = self._connected
            self._close_master()
            self._set_connected(False)
            self._heartbeat_received = False
            self._last_heartbeat = None

        if was_connected:
            self._notify_connection(False)

        logger.info("[MAVLinkManager] Disconnected")

    # ========================================================================
    # Close Master
    # ========================================================================

    def _close_master(self) -> None:
        master = self._master
        self._master = None

        if master is None:
            return

        try:
            master.close()
        except Exception:
            logger.exception("[MAVLinkManager] Failed to close MAVLink connection")

    # ========================================================================
    # Receive Loop
    # ========================================================================

    def _receive_loop(self) -> None:

        logger.info("[MAVLinkManager] Receive loop started")

        while not self._stop_event.is_set():

            if not self.connected:
                if not self.reconnect_enabled:
                    self._stop_event.wait(self.reconnect_delay_s)
                    continue

                connected = self.connect()
                if not connected:
                    self._stop_event.wait(self.reconnect_delay_s)
                    continue

            try:
                master = self._master
                if master is None:
                    self._set_connected(False)
                    continue

                message = master.recv_match(
                    blocking=True,
                    timeout=self.receive_timeout_s,
                )

                if message is None:
                    self._check_heartbeat()
                    continue

                self.message_router.route(message)
                self._check_heartbeat()

            except Exception:
                logger.exception("[MAVLinkManager] Receive loop error")
                self._handle_connection_failure()

        logger.info("[MAVLinkManager] Receive loop stopped")

    # ========================================================================
    # Handle Connection Failure
    # ========================================================================

    def _handle_connection_failure(self) -> None:
        self.disconnect()

        if not self.reconnect_enabled:
            return

        self._stop_event.wait(self.reconnect_delay_s)

    # ========================================================================
    # Heartbeat
    # ========================================================================

    def _handle_heartbeat(self, message: Any) -> None:

        try:
            self._last_heartbeat = time.monotonic()
            self._heartbeat_received = True

            # Store the active flight mode reported by ArduPilot.
            with self._state_lock:
                self._last_heartbeat_custom_mode = int(
                    getattr(message, "custom_mode", -1)
                )
                self._last_heartbeat_base_mode = int(
                    getattr(message, "base_mode", 0)
                )

            if self._master is not None:
                self._target_system = int(self._master.target_system)
                self._target_component = int(self._master.target_component)

        except Exception:
            logger.exception("[MAVLinkManager] Heartbeat processing failed")

    def _handle_statustext(self, message: Any) -> None:
        try:
            text = str(getattr(message, "text", "")).strip()
            if text:
                self._statustext_cache = (time.monotonic(), text)
                logger.debug("[MAVLinkManager] STATUSTEXT: %s", text)
        except Exception:
            pass

    def _handle_rc_channels(self, message: Any) -> None:
        """
        Read the RC_CHANNELS message and update shared_data.state.rc_safety_pwm.
        SafetyMonitor reads this value to monitor the RC safety channel selected in config.
        """
        import config as _rc_cfg
        try:
            ch = int(getattr(_rc_cfg, "RC_SAFETY_CHANNEL", 6))
            attr = f"chan{ch}_raw" if 1 <= ch <= 18 else None
            pwm = int(getattr(message, attr, 0)) if attr else 0
            sd = self._shared_data
            if sd is not None:
                state = getattr(sd, "state", None)
                if state is not None:
                    state.rc_safety_pwm = pwm
                    state.rc16_pwm = pwm
        except Exception:
            pass

    # ========================================================================
    # Heartbeat Check
    # ========================================================================

    def _check_heartbeat(self) -> None:

        with self._state_lock:
            if not self._connected:
                return
            last = self._last_heartbeat

        if last is None:
            return

        age = time.monotonic() - last

        if age > self.heartbeat_timeout_s:
            logger.warning(
                "[MAVLinkManager] Heartbeat timeout age=%.2fs",
                age,
            )
            self._handle_connection_failure()

    # ========================================================================
    # Connected Setter
    # ========================================================================

    def _set_connected(self, value: bool) -> None:
        with self._state_lock:
            old_value = self._connected
            self._connected = bool(value)

        if old_value != value:
            self._notify_connection(value)

    # ========================================================================
    # Connection Callback
    # ========================================================================

    def add_connection_callback(self, callback: Callable[[bool], None]) -> None:
        if not callable(callback):
            raise TypeError("callback must be callable")

        with self._state_lock:
            if callback not in self._connection_callbacks:
                self._connection_callbacks.append(callback)

    def remove_connection_callback(self, callback: Callable[[bool], None]) -> None:
        with self._state_lock:
            try:
                self._connection_callbacks.remove(callback)
            except ValueError:
                pass

    def _notify_connection(self, connected: bool) -> None:
        with self._state_lock:
            callbacks = list(self._connection_callbacks)

        for callback in callbacks:
            try:
                callback(connected)
            except Exception:
                logger.exception("[MAVLinkManager] Connection callback failed")

    # ========================================================================
    # Master
    # ========================================================================

    @property
    def master(self) -> Any:
        return self._master

    # ========================================================================
    # Connected
    # ========================================================================

    @property
    def connected(self) -> bool:
        with self._state_lock:
            return self._connected

    # ========================================================================
    # Running
    # ========================================================================

    @property
    def running(self) -> bool:
        with self._state_lock:
            return self._running

    # ========================================================================
    # Target System
    # ========================================================================

    @property
    def target_system(self) -> int:
        with self._state_lock:
            return self._target_system

    # ========================================================================
    # Target Component
    # ========================================================================

    @property
    def target_component(self) -> int:
        with self._state_lock:
            return self._target_component

    # ========================================================================
    # Last Heartbeat
    # ========================================================================

    @property
    def last_heartbeat(self) -> Optional[float]:
        with self._state_lock:
            return self._last_heartbeat

    # ========================================================================
    # Heartbeat Age
    # ========================================================================

    @property
    def heartbeat_age(self) -> Optional[float]:
        last = self.last_heartbeat
        if last is None:
            return None
        return max(0.0, time.monotonic() - last)

    # ========================================================================
    # State
    # ========================================================================

    def state(self) -> MAVLinkState:
        """
        MAVLink state snapshot.
        """

        with self._state_lock:
            return MAVLinkState(
                connected=self._connected,
                heartbeat_received=self._heartbeat_received,
                target_system=self._target_system,
                target_component=self._target_component,
                last_heartbeat=self._last_heartbeat,
                reconnect_count=self._reconnect_count,
                receive_loop_running=(
                    self._receive_thread is not None
                    and self._receive_thread.is_alive()
                ),
            )

    # ========================================================================
    # Reconnect
    # ========================================================================

    def reconnect(self) -> bool:
        logger.info("[MAVLinkManager] Manual reconnect requested")
        self._reconnect_count += 1
        self.disconnect()
        return self.connect()

    # ========================================================================
    # Send Raw
    # ========================================================================

    def send_message(self, message: Any) -> bool:

        if not self.connected:
            return False

        master = self._master
        if master is None:
            return False

        try:
            master.mav.send(message)
            return True
        except Exception:
            logger.exception("[MAVLinkManager] Raw message send failed")
            return False

    # ========================================================================
    # _wait_command_ack (PRIVATE YARDIMCI)
    # ========================================================================

    def _wait_command_ack(
        self,
        expected_command: int,
        timeout_s: float,
    ) -> bool:

        router = self.message_router
        incoming: Queue[Any] = Queue()

        def _on_ack(msg: Any) -> None:
            incoming.put(msg)

        sub_id = router.subscribe(
            "COMMAND_ACK",
            _on_ack,
            owner="MAVLinkManager._wait_command_ack",
        )

        try:
            deadline = time.monotonic() + timeout_s

            while time.monotonic() < deadline:
                remaining = max(0.05, deadline - time.monotonic())

                try:
                    msg = incoming.get(timeout=min(0.5, remaining))
                except Empty:
                    continue

                try:
                    ack_command = int(msg.command)
                    ack_result = int(msg.result)
                except Exception:
                    continue

                if ack_command != expected_command:
                    # ACK belongs to another command; ignore it and keep waiting.
                    continue

                accepted = (
                    ack_result == mavutil.mavlink.MAV_RESULT_ACCEPTED
                )

                if accepted:
                    logger.info(
                        "[MAVLinkManager] COMMAND_ACK accepted cmd=%d",
                        expected_command,
                    )
                else:
                    logger.warning(
                        "[MAVLinkManager] COMMAND_ACK NOT accepted "
                        "cmd=%d result=%d",
                        expected_command,
                        ack_result,
                    )

                return accepted

            logger.warning(
                "[MAVLinkManager] COMMAND_ACK timeout cmd=%d after %.1fs",
                expected_command,
                timeout_s,
            )
            return False

        finally:
            try:
                router.unsubscribe(sub_id)
            except Exception:
                pass

    # ========================================================================
    # ARM
    # ========================================================================

    def arm(self, force: bool = False) -> bool:

        if not self.connected:
            return False

        master = self._master
        if master is None:
            return False

        import config as _cfg
        from core.events import EventType

        max_attempts: int = getattr(_cfg, "ARM_MAX_ATTEMPTS", 5)
        param2 = 21196 if force else 0

        for attempt in range(1, max_attempts + 1):
            try:
                # Return immediately if the vehicle is already armed.
                if master.motors_armed():
                    logger.info("[MAVLinkManager] Vehicle is already armed")
                    return True

                # Send the ARM command.
                logger.info(
                    "[MAVLinkManager] ARM attempt %d/%d",
                    attempt, max_attempts,
                )
                master.mav.command_long_send(
                    self.target_system,
                    self.target_component,
                    mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                    0,
                    1,      # param1: 1=ARM
                    param2, # param2: force flag
                    0, 0, 0, 0, 0,
                )

                logger.info("[MAVLinkManager] ARM command sent, waiting for ACK...")

            except Exception:
                logger.exception(
                    "[MAVLinkManager] ARM send failed (attempt %d/%d)",
                    attempt, max_attempts,
                )

            # Wait for the command ACK.
            ack_ok = self._wait_command_ack(
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                timeout_s=self.command_ack_timeout_s,
            )

            # After a successful ACK, verify that the vehicle is actually armed.
            if ack_ok:
                if master.motors_armed():
                    logger.info(
                        "[MAVLinkManager] ARM confirmed (attempt %d/%d)",
                        attempt, max_attempts,
                    )
                    return True

            logger.warning(
                "[MAVLinkManager] ARM failed (attempt %d/%d).",
                attempt, max_attempts,
            )

            if attempt < max_attempts:
                time.sleep(1.0)

        # All attempts were exhausted; publish the event.
        logger.error(
            "[MAVLinkManager] ARM failed after %d attempts. "
            "Publishing ARM_MAX_ATTEMPTS_EXCEEDED.",
            max_attempts,
        )
        if self._event_bus is not None:
            self._event_bus.publish(
                EventType.ARM_MAX_ATTEMPTS_EXCEEDED,
                {"attempts": max_attempts},
                source="MAVLinkManager",
            )
        return False

    # ========================================================================
    # DISARM
    # ========================================================================

    def disarm(self, force: bool = False) -> bool:
        if not self.connected:
            return False

        master = self._master
        if master is None:
            return False

        try:
            param2 = 21196 if force else 0

            master.mav.command_long_send(
                self.target_system,
                self.target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                0,
                0,      # param1: 0=DISARM
                param2,
                0, 0, 0, 0, 0,
            )

            return True

        except Exception:
            logger.exception("[MAVLinkManager] DISARM failed")
            return False

    # ========================================================================
    # Takeoff
    # ========================================================================

    def takeoff(self, altitude_m: float) -> bool:
        if altitude_m <= 0:
            return False

        if not self.connected:
            return False

        master = self._master
        if master is None:
            return False

        _RESULT_NAMES = {
            0: "ACCEPTED",
            1: "TEMPORARILY_REJECTED",
            2: "DENIED",
            3: "UNSUPPORTED",
            4: "TEMPORARILY_REJECTED",
            5: "CANCELLED",
        }

        MAX_ATTEMPTS = 4
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                master.mav.command_long_send(
                    self.target_system,
                    self.target_component,
                    mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
                    0,
                    0, 0, 0, 0, 0, 0,
                    float(altitude_m),
                )
                logger.info(
                    "[MAVLinkManager] TAKEOFF %.2f m sent (attempt %d/%d); waiting for ACK...",
                    float(altitude_m), attempt, MAX_ATTEMPTS,
                )
            except Exception:
                logger.exception("[MAVLinkManager] TAKEOFF send failed")
                return False

            ack_ok = self._wait_command_ack(
                mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
                timeout_s=self.command_ack_timeout_s,
            )

            if ack_ok:
                logger.info(
                    "[MAVLinkManager] TAKEOFF command accepted by ArduPilot "
                    "(attempt %d/%d)",
                    attempt, MAX_ATTEMPTS,
                )
                return True

            # Failure: inspect STATUSTEXT to determine the reason.
            statustext = self._last_statustext()
            if statustext:
                logger.warning(
                    "[MAVLinkManager] TAKEOFF rejected (attempt %d/%d) | "
                    "ArduPilot message: \"%s\"",
                    attempt, MAX_ATTEMPTS, statustext,
                )
            else:
                logger.warning(
                    "[MAVLinkManager] TAKEOFF rejected (attempt %d/%d) | "
                    "Possible causes: vehicle is not armed, GPS fix is unavailable, "
                    "EKF is not ready or a pre-arm check failed.",
                    attempt, MAX_ATTEMPTS,
                )

            if attempt < MAX_ATTEMPTS:
                wait_s = attempt * 0.75
                logger.info(
                    "[MAVLinkManager] Retrying in %.1f s...", wait_s
                )
                time.sleep(wait_s)

        logger.error(
            "[MAVLinkManager] TAKEOFF was not accepted after %d attempts — "
            "check pre-arm status (GPS fix, EKF, battery, and mode).",
            MAX_ATTEMPTS,
        )
        return False

    def _last_statustext(self, max_age_s: float = 3.0) -> str:
        try:
            entry = getattr(self, "_statustext_cache", None)
            if entry is None:
                return ""
            ts, text = entry
            if time.monotonic() - ts <= max_age_s:
                return text
        except Exception:
            pass
        return ""

    # ========================================================================
    # Land
    # ========================================================================

    def land(self) -> bool:
        return self.set_mode("LAND")

    # ========================================================================
    # RTL
    # ========================================================================

    def rtl(self) -> bool:
        return self.set_mode("RTL")

    # ========================================================================
    # Set Mode
    # ========================================================================

    def set_mode(self, mode: str) -> bool:

        if not mode:
            return False

        if not self.connected:
            return False

        master = self._master
        if master is None:
            return False

        try:
            mode_upper = mode.upper()
            mapping = master.mode_mapping()

            if not mapping:
                logger.error("[MAVLinkManager] No mode mapping available")
                return False

            if mode_upper not in mapping:
                logger.error("[MAVLinkManager] Unknown mode: %s", mode_upper)
                return False

            mode_id = mapping[mode_upper]
            master.set_mode(mode_id)

            logger.info(
                "[MAVLinkManager] SET_MODE %s (id=%d) sent, waiting for HEARTBEAT confirmation...",
                mode_upper,
                mode_id,
            )

        except Exception:
            logger.exception("[MAVLinkManager] Set mode failed")
            return False

        # Confirm the mode transition through HEARTBEAT.
        # ArduPilot usually reports the new mode in HEARTBEAT within 1-3 seconds.
        deadline = time.monotonic() + self.command_ack_timeout_s

        while time.monotonic() < deadline:
            time.sleep(0.1)

            with self._state_lock:
                current_custom_mode = self._last_heartbeat_custom_mode

            if current_custom_mode is not None:
                # Check the mode ID.
                # Compare the reported mode ID with mapping[mode_upper].
                try:
                    expected_id = master.mode_mapping().get(mode_upper, -1)
                except Exception:
                    expected_id = -1

                if current_custom_mode == expected_id:
                    logger.info(
                        "[MAVLinkManager] Mode %s confirmed via HEARTBEAT "
                        "(custom_mode=%d)",
                        mode_upper,
                        current_custom_mode,
                    )
                    return True

        logger.warning(
            "[MAVLinkManager] Mode %s NOT confirmed via HEARTBEAT within %.1fs. "
            "Last custom_mode=%s",
            mode_upper,
            self.command_ack_timeout_s,
            self._last_heartbeat_custom_mode,
        )
        return False

    # ========================================================================
    # Servo
    # ========================================================================

    def send_servo_command(self, channel: int, pwm: int) -> bool:
        if channel < 1:
            return False

        if not (500 <= pwm <= 2500):
            return False

        if not self.connected:
            return False

        master = self._master
        if master is None:
            return False

        try:
            master.mav.command_long_send(
                self.target_system,
                self.target_component,
                mavutil.mavlink.MAV_CMD_DO_SET_SERVO,
                0,
                float(channel),
                float(pwm),
                0, 0, 0, 0, 0,
            )
            return True
        except Exception:
            logger.exception("[MAVLinkManager] Servo command failed")
            return False

    # ========================================================================
    # GUIDED global position
    # ========================================================================

    def goto_guided(
        self,
        latitude: float,
        longitude: float,
        altitude_m: float,
        speed_mps: float = 3.0,
    ) -> bool:

        if not self.connected:
            return False

        master = self._master
        if master is None:
            return False

        try:
            master.mav.command_int_send(
                self.target_system,
                self.target_component,
                mavutil.mavlink.MAV_FRAME_GLOBAL_INT,
                mavutil.mavlink.MAV_CMD_DO_REPOSITION,
                0,
                0,
                float(speed_mps),
                0,
                0,
                0,
                int(float(latitude) * 1e7),
                int(float(longitude) * 1e7),
                float(altitude_m),
            )
            return True
        except Exception:
            logger.exception("[MAVLinkManager] GUIDED goto failed")
            return False

    # ========================================================================
    # BODY_NED velocity
    # ========================================================================

    def send_body_velocity(
        self,
        vx: float,
        vy: float,
        vz: float = 0.0,
    ) -> bool:

        if not self.connected:
            return False

        master = self._master
        if master is None:
            return False

        try:
            # Bit 11 CLEAR enables yaw_rate; sending 0.0 holds the heading.
            # Bit 10 SET ignores the yaw angle; no absolute heading is commanded.
            type_mask = 0x07C7

            master.mav.set_position_target_local_ned_send(
                0,
                self.target_system,
                self.target_component,
                mavutil.mavlink.MAV_FRAME_BODY_NED,
                type_mask,
                0.0, 0.0, 0.0,
                float(vx), float(vy), float(vz),
                0.0, 0.0, 0.0,
                0.0,   # yaw (ignored when bit 10 is set)
                0.0,   # yaw_rate = 0 rad/s -> heading kilitli
            )
            return True
        except Exception:
            logger.exception("[MAVLinkManager] BODY_NED velocity command failed")
            return False

    # ========================================================================
    # Set Parameter
    # ========================================================================

    def set_parameter(self, name: str, value: float) -> bool:
        if not self.connected:
            return False

        master = self._master
        if master is None:
            return False

        try:
            master.mav.param_set_send(
                self.target_system,
                self.target_component,
                name.encode("ascii"),
                float(value),
                mavutil.mavlink.MAV_PARAM_TYPE_REAL32,
            )
            return True
        except Exception:
            logger.exception("[MAVLinkManager] Set parameter failed")
            return False

    # ========================================================================
    # Request Parameter
    # ========================================================================

    def request_parameter(self, name: str) -> bool:
        if not name:
            return False

        if not self.connected:
            return False

        master = self._master
        if master is None:
            return False

        try:
            master.mav.param_request_read_send(
                self.target_system,
                self.target_component,
                name.encode("ascii"),
                -1,
            )
            return True
        except Exception:
            logger.exception("[MAVLinkManager] Request parameter failed")
            return False

    # ========================================================================
    # Command Long
    # ========================================================================

    def command_long(self, command: int, *params: float) -> bool:
        """
        Generic MAV_CMD_COMMAND_LONG.

        parameters can contain at most 7 values.
        """

        if not self.connected:
            return False

        if len(params) > 7:
            raise ValueError("COMMAND_LONG accepts maximum 7 parameters")

        master = self._master
        if master is None:
            return False

        values = list(params)
        while len(values) < 7:
            values.append(0.0)

        try:
            master.mav.command_long_send(
                self.target_system,
                self.target_component,
                int(command),
                0,
                float(values[0]),
                float(values[1]),
                float(values[2]),
                float(values[3]),
                float(values[4]),
                float(values[5]),
                float(values[6]),
            )
            return True
        except Exception:
            logger.exception("[MAVLinkManager] COMMAND_LONG failed")
            return False

    # ========================================================================
    # Mission Progress
    # ========================================================================

    def mission_progress(self) -> tuple[int, int]:

        return (0, 0)

    # ========================================================================
    # Representation
    # ========================================================================

    def __repr__(self) -> str:
        return (
            "MAVLinkManager("
            f"connected={self.connected}, "
            f"running={self.running}, "
            f"target_system={self.target_system}, "
            f"target_component={self.target_component}"
            ")"
        )
