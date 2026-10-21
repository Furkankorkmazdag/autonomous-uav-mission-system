"""MAVLink-based servo output control for payload release."""

from __future__ import annotations

import logging
import threading
import time

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

from config import PAYLOAD_INITIAL_PWM


logger = logging.getLogger(__name__)


# ============================================================================
# Servo State
# ============================================================================

class ServoState(str, Enum):
    """Servo runtime state."""

    UNKNOWN = "unknown"

    DISABLED = "disabled"

    READY = "ready"

    MOVING = "moving"

    ERROR = "error"


# ============================================================================
# Servo Configuration
# ============================================================================

@dataclass(frozen=True)
class ServoConfig:

    name: str

    channel: int

    min_pwm: int = 1000

    max_pwm: int = 2000

    neutral_pwm: int = 1500

    min_angle_deg: float = 0.0

    max_angle_deg: float = 180.0

    default_angle_deg: float = 90.0

    command_timeout_s: float = 1.0

    enabled: bool = True

    def __post_init__(self) -> None:

        if not self.name:

            raise ValueError(
                "Servo name cannot be empty"
            )

        if self.channel < 1:

            raise ValueError(
                "Servo channel must be >= 1"
            )

        if (
            self.min_pwm
            >= self.max_pwm
        ):

            raise ValueError(
                "min_pwm must be lower than max_pwm"
            )

        if not (
            self.min_pwm
            <= self.neutral_pwm
            <= self.max_pwm
        ):

            raise ValueError(
                "neutral_pwm is outside PWM limits"
            )

        if (
            self.min_angle_deg
            >= self.max_angle_deg
        ):

            raise ValueError(
                "min_angle_deg must be lower "
                "than max_angle_deg"
            )

        if not (
            self.min_angle_deg
            <= self.default_angle_deg
            <= self.max_angle_deg
        ):

            raise ValueError(
                "default_angle_deg is outside "
                "angle limits"
            )

        if self.command_timeout_s <= 0:

            raise ValueError(
                "command_timeout_s must be greater than zero"
            )


# ============================================================================
# Servo Snapshot
# ============================================================================

@dataclass(frozen=True)
class ServoSnapshot:
    """
    Immutable servo state snapshot.
    """

    name: str

    channel: int

    state: ServoState

    pwm: int

    angle_deg: float

    enabled: bool

    last_command_time: Optional[float]

    last_error: Optional[str]


# ============================================================================
# MAVLink Driver
# ============================================================================

class PixhawkServoDriver:

    def __init__(
        self,
        mavlink: Any,
    ) -> None:

        if mavlink is None:

            raise ValueError(
                "mavlink cannot be None"
            )

        self._mavlink = mavlink

        self._lock = threading.RLock()

    # =========================================================================
    # Send PWM
    # =========================================================================

    def set_pwm(
        self,
        channel: int,
        pwm: int,
    ) -> bool:
        """
        Send servo PWM command to Pixhawk.
        """

        if channel < 1:

            raise ValueError(
                "channel must be >= 1"
            )

        if pwm < 500 or pwm > 2500:

            raise ValueError(
                "PWM outside allowed MAVLink range"
            )

        with self._lock:

            try:

                self._send_servo_command(
                    channel,
                    pwm,
                )

                return True

            except Exception:

                logger.exception(
                    "[PixhawkServoDriver] "
                    "Failed to send servo command "
                    "channel=%d pwm=%d",
                    channel,
                    pwm,
                )

                return False

    # =========================================================================
    # MAVLink command
    # =========================================================================

    def _send_servo_command(
        self,
        channel: int,
        pwm: int,
    ) -> None:

        # ---------------------------------------------------------------------
        # Preferred architecture:
        #
        # MAVLinkManager.send_servo_command(...)
        # ---------------------------------------------------------------------

        method = getattr(
            self._mavlink,
            "send_servo_command",
            None,
        )

        if callable(method):

            result = method(
                channel,
                pwm,
            )

            if result is False:

                raise RuntimeError(
                    "MAVLink servo command was rejected"
                )

            return

        # ---------------------------------------------------------------------
        # Alternative:
        #
        # MAVLinkManager.command_do_set_servo(...)
        # ---------------------------------------------------------------------

        method = getattr(
            self._mavlink,
            "command_do_set_servo",
            None,
        )

        if callable(method):

            result = method(
                channel,
                pwm,
            )

            if result is False:

                raise RuntimeError(
                    "MAVLink servo command was rejected"
                )

            return

        # ---------------------------------------------------------------------
        # Direct pymavlink fallback.
        #
        # Use this section only when the central MAVLink object exposes the
        # underlying master connection object.
        # ---------------------------------------------------------------------

        master = getattr(
            self._mavlink,
            "master",
            None,
        )

        if master is None:

            raise AttributeError(
                "MAVLink interface must provide "
                "send_servo_command(), "
                "command_do_set_servo(), "
                "or master"
            )

        mavlink_module = getattr(
            master,
            "mavlink",
            None,
        )

        if mavlink_module is None:

            raise AttributeError(
                "pymavlink MAVLink module not found"
            )

        master.mav.command_long_send(
            master.target_system,
            master.target_component,

            mavlink_module
            .MAV_CMD_DO_SET_SERVO,

            0,

            channel,
            pwm,

            0,
            0,
            0,
            0,
            0,
        )


# ============================================================================
# Servo Channel
# ============================================================================

class ServoChannel:

    def __init__(
        self,
        config: ServoConfig,
        driver: PixhawkServoDriver,
    ) -> None:

        self.config = config

        self._driver = driver

        self._lock = threading.RLock()

        self._state = (
            ServoState.UNKNOWN
        )

        self._pwm = int(PAYLOAD_INITIAL_PWM)

        self._angle_deg = (
            config.default_angle_deg
        )

        self._last_command_time: Optional[
            float
        ] = None

        self._last_error: Optional[
            str
        ] = None

    # =========================================================================
    # Initialize
    # =========================================================================

    def initialize(self) -> bool:

        with self._lock:

            if not self.config.enabled:

                self._state = (
                    ServoState.DISABLED
                )

                return False

        success = self.set_angle(
            self.config.default_angle_deg
        )

        if success:

            with self._lock:

                self._state = (
                    ServoState.READY
                )

        return success

    # =========================================================================
    # Set PWM
    # =========================================================================

    def set_pwm(
        self,
        pwm: int,
    ) -> bool:

        with self._lock:

            if (
                not self.config.enabled
                or self._state
                == ServoState.DISABLED
            ):

                return False

            safe_pwm = max(
                self.config.min_pwm,
                min(
                    self.config.max_pwm,
                    int(pwm),
                ),
            )

            self._state = (
                ServoState.MOVING
            )

        try:

            success = (
                self._driver.set_pwm(
                    self.config.channel,
                    safe_pwm,
                )
            )

            if not success:

                raise RuntimeError(
                    "Pixhawk rejected servo command"
                )

            angle = (
                self._pwm_to_angle(
                    safe_pwm
                )
            )

            with self._lock:

                self._pwm = safe_pwm

                self._angle_deg = angle

                self._last_command_time = (
                    time.monotonic()
                )

                self._last_error = None

                self._state = (
                    ServoState.READY
                )

            logger.debug(
                "[Servo:%s] "
                "channel=%d pwm=%d",
                self.config.name,
                self.config.channel,
                safe_pwm,
            )

            return True

        except Exception as exc:

            self._set_error(
                exc
            )

            return False

    # =========================================================================
    # Set angle
    # =========================================================================

    def set_angle(
        self,
        angle_deg: float,
    ) -> bool:

        with self._lock:

            if (
                not self.config.enabled
            ):

                return False

            safe_angle = max(
                self.config.min_angle_deg,
                min(
                    self.config.max_angle_deg,
                    float(angle_deg),
                ),
            )

            pwm = (
                self._angle_to_pwm(
                    safe_angle
                )
            )

        return self.set_pwm(
            pwm
        )

    # =========================================================================
    # Angle -> PWM
    # =========================================================================

    def _angle_to_pwm(
        self,
        angle_deg: float,
    ) -> int:
        """
        Linear angle -> PWM conversion.
        """

        angle_range = (
            self.config.max_angle_deg
            - self.config.min_angle_deg
        )

        pwm_range = (
            self.config.max_pwm
            - self.config.min_pwm
        )

        normalized = (
            angle_deg
            - self.config.min_angle_deg
        ) / angle_range

        pwm = (
            self.config.min_pwm
            + normalized
            * pwm_range
        )

        return int(
            round(pwm)
        )

    # =========================================================================
    # PWM -> Angle
    # =========================================================================

    def _pwm_to_angle(
        self,
        pwm: int,
    ) -> float:
        """
        PWM -> angle conversion.
        """

        pwm_range = (
            self.config.max_pwm
            - self.config.min_pwm
        )

        angle_range = (
            self.config.max_angle_deg
            - self.config.min_angle_deg
        )

        normalized = (
            pwm
            - self.config.min_pwm
        ) / pwm_range

        return (
            self.config.min_angle_deg
            + normalized
            * angle_range
        )

    # =========================================================================
    # Neutral
    # =========================================================================

    def neutral(self) -> bool:
        """
        Servo'yu neutral PWM'e getir.
        """

        return self.set_pwm(
            self.config.neutral_pwm
        )

    # =========================================================================
    # Disable
    # =========================================================================

    def disable(self) -> bool:

        with self._lock:

            self._state = (
                ServoState.DISABLED
            )

        return True

    # =========================================================================
    # Error
    # =========================================================================

    def _set_error(
        self,
        exception: Exception,
    ) -> None:

        with self._lock:

            self._state = (
                ServoState.ERROR
            )

            self._last_error = (
                str(exception)
            )

        logger.error(
            "[Servo:%s] "
            "Servo command failed: %s",
            self.config.name,
            exception,
        )

    # =========================================================================
    # Snapshot
    # =========================================================================

    def snapshot(
        self,
    ) -> ServoSnapshot:

        with self._lock:

            return ServoSnapshot(
                name=self.config.name,

                channel=self.config.channel,

                state=self._state,

                pwm=self._pwm,

                angle_deg=self._angle_deg,

                enabled=self.config.enabled,

                last_command_time=(
                    self._last_command_time
                ),

                last_error=(
                    self._last_error
                ),
            )

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def channel(
        self,
    ) -> int:

        return self.config.channel

    @property
    def pwm(
        self,
    ) -> int:

        with self._lock:

            return self._pwm

    @property
    def angle(
        self,
    ) -> float:

        with self._lock:

            return self._angle_deg

    @property
    def state(
        self,
    ) -> ServoState:

        with self._lock:

            return self._state


# ============================================================================
# Servo Manager
# ============================================================================

class ServoManager:

    name = "ServoManager"

    def __init__(
        self,
        mavlink: Any,
    ) -> None:

        self._lock = threading.RLock()

        self._driver = (
            PixhawkServoDriver(
                mavlink
            )
        )

        self._servos: Dict[
            str,
            ServoChannel,
        ] = {}

        self._running = False

    # =========================================================================
    # Start
    # =========================================================================

    def start(self) -> bool:
        """Start the component or operation."""

        with self._lock:

            if self._running:

                return True

            self._running = True

            servos = list(
                self._servos.values()
            )

        success = True

        for servo in servos:

            if not servo.initialize():

                success = False

                logger.error(
                    "[ServoManager] "
                    "Failed to initialize %s",
                    servo.config.name,
                )

        logger.info(
            "[ServoManager] Started"
        )

        return success

    # =========================================================================
    # Stop
    # =========================================================================

    def stop(self) -> None:
        """Stop the component or operation safely."""

        with self._lock:

            if not self._running:

                return

            servos = list(
                self._servos.values()
            )

            self._running = False

        for servo in servos:

            try:

                servo.disable()

            except Exception:

                logger.exception(
                    "[ServoManager] "
                    "Failed to disable %s",
                    servo.config.name,
                )

        logger.info(
            "[ServoManager] Stopped"
        )

    # =========================================================================
    # Register
    # =========================================================================

    def register(
        self,
        config: ServoConfig,
    ) -> ServoChannel:

        with self._lock:

            if config.name in self._servos:

                raise ValueError(
                    f"Servo already registered: "
                    f"{config.name}"
                )

            for existing in (
                self._servos.values()
            ):

                if (
                    existing.config.channel
                    == config.channel
                ):

                    raise ValueError(
                        "Pixhawk servo channel "
                        f"{config.channel} is already "
                        f"assigned to "
                        f"{existing.config.name}"
                    )

            servo = ServoChannel(
                config=config,

                driver=self._driver,
            )

            self._servos[
                config.name
            ] = servo

            running = self._running

        if running:

            servo.initialize()

        logger.info(
            "[ServoManager] "
            "Registered %s -> Pixhawk channel %d",
            config.name,
            config.channel,
        )

        return servo

    # =========================================================================
    # Get
    # =========================================================================

    def get(
        self,
        name: str,
    ) -> Optional[ServoChannel]:
        """
        Servo'yu logical name ile getir.
        """

        with self._lock:

            return self._servos.get(
                name
            )

    # =========================================================================
    # Set angle
    # =========================================================================

    def set_angle(
        self,
        name: str,
        angle_deg: float,
    ) -> bool:

        servo = self.get(
            name
        )

        if servo is None:

            logger.error(
                "[ServoManager] "
                "Unknown servo: %s",
                name,
            )

            return False

        return servo.set_angle(
            angle_deg
        )

    # =========================================================================
    # Set PWM
    # =========================================================================

    def set_pwm(
        self,
        name: str,
        pwm: int,
    ) -> bool:

        servo = self.get(
            name
        )

        if servo is None:

            return False

        return servo.set_pwm(
            pwm
        )

    # =========================================================================
    # Neutral
    # =========================================================================

    def neutral(
        self,
        name: str,
    ) -> bool:
        """
        Servo'yu neutral pozisyona getir.
        """

        servo = self.get(
            name
        )

        if servo is None:

            return False

        return servo.neutral()

    # =========================================================================
    # Initialize one
    # =========================================================================

    def initialize(
        self,
        name: str,
    ) -> bool:
        """
        Initialize a single servo.
        """

        servo = self.get(
            name
        )

        if servo is None:

            return False

        return servo.initialize()

    # =========================================================================
    # All
    # =========================================================================

    def all(
        self,
    ) -> Dict[
        str,
        ServoSnapshot,
    ]:

        with self._lock:

            return {
                name: servo.snapshot()
                for name, servo
                in self._servos.items()
            }

    # =========================================================================
    # Running
    # =========================================================================

    @property
    def running(
        self,
    ) -> bool:

        with self._lock:

            return self._running

    # =========================================================================
    # Representation
    # =========================================================================

    def __repr__(
        self,
    ) -> str:

        with self._lock:

            return (
                "ServoManager("
                f"running={self._running}, "
                f"servos="
                f"{len(self._servos)}"
                ")"
            )


# ============================================================================
# Default SUAS Servo Configuration
# ============================================================================

def create_default_servo_manager(
    mavlink: Any,
) -> ServoManager:
    """SUAS payload servo manager. MAIN OUT 5-8 are all initialized to 1000 PWM."""
    manager = ServoManager(mavlink=mavlink)

    channels = [
        ("payload_servo_mannequin_1", 5),
        ("payload_servo_mannequin_2", 6),
        ("payload_servo_canopy_1", 7),
        ("payload_servo_canopy_2", 8),
    ]

    for name, channel in channels:
        manager.register(
            ServoConfig(
                name=name,
                channel=channel,
                min_pwm=1000,
                max_pwm=2000,
                neutral_pwm=int(PAYLOAD_INITIAL_PWM),
                min_angle_deg=0.0,
                max_angle_deg=180.0,
                default_angle_deg=0.0,
            )
        )

    return manager

