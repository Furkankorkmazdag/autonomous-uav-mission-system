"""Servo-backed implementation of the payload-driver interface."""

from __future__ import annotations

import logging
import threading
import time

from dataclasses import dataclass
from typing import Any, Optional

from payload.payload_driver import (
    PayloadDriver,
    PayloadDriverConfig,
    PayloadDriverError,
    PayloadDriverNotReadyError,
    PayloadDriverReleaseError,
)


logger = logging.getLogger(__name__)


# ============================================================================
# Exceptions
# ============================================================================

class ServoDriverError(
    PayloadDriverError
):
    """Base servo driver exception."""



class ServoDriverNotReadyError(ServoDriverError):
    """Raised when the servo driver is used before initialization."""

class ServoConfigurationError(
    ServoDriverError
):
    """Invalid servo configuration."""


class ServoCommandError(
    ServoDriverError
):
    """Servo command failed."""


# ============================================================================
# Configuration
# ============================================================================

@dataclass(frozen=True)
class ServoDriverConfig(
    PayloadDriverConfig
):
    """
    Servo payload configuration.

    servo_channel:
        MAVLink servo output channel.

    release_pwm:
        PWM value used for payload release.

    reset_pwm:
        PWM value used after release.

    release_duration_s:
        How long the release position is maintained.

    pwm_min / pwm_max:
        Valid PWM range.

    auto_reset:
        Automatically return servo to reset position after release.
    """

    servo_channel: int = 9

    release_pwm: int = 1900

    reset_pwm: int = 1100

    release_duration_s: float = 0.5

    pwm_min: int = 800

    pwm_max: int = 2200

    auto_reset: bool = True

    def __post_init__(self) -> None:

        super().__post_init__()

        self.validate()

    def validate(self) -> None:

        super().validate()

        if self.servo_channel < 1:

            raise ServoConfigurationError(
                "servo_channel must be >= 1"
            )

        if self.release_duration_s < 0:

            raise ServoConfigurationError(
                "release_duration_s cannot be negative"
            )

        if not (
            self.pwm_min
            <= self.release_pwm
            <= self.pwm_max
        ):

            raise ServoConfigurationError(
                "release_pwm is outside PWM range"
            )

        if not (
            self.pwm_min
            <= self.reset_pwm
            <= self.pwm_max
        ):

            raise ServoConfigurationError(
                "reset_pwm is outside PWM range"
            )


# ============================================================================
# Servo status
# ============================================================================

@dataclass(frozen=True)
class ServoStatus:
    """Current servo state."""

    channel: int

    pwm: int

    initialized: bool

    ready: bool

    released: bool

    last_error: Optional[str]


# ============================================================================
# Servo payload driver
# ============================================================================

class ServoPayloadDriver(
    PayloadDriver
):

    def __init__(
        self,
        backend: Any,
        config: Optional[
            ServoDriverConfig
        ] = None,
    ) -> None:

        if backend is None:

            raise ValueError(
                "backend cannot be None"
            )

        self._backend = backend

        self.servo_config = (
            config
            or ServoDriverConfig()
        )

        super().__init__(
            config=self.servo_config
        )

        self._servo_lock = (
            threading.RLock()
        )

        self._current_pwm = (
            self.servo_config.reset_pwm
        )

        self._released = False

    # =========================================================================
    # Initialization
    # =========================================================================

    def _initialize_hardware(
        self,
    ) -> bool:

        with self._servo_lock:

            self._set_pwm(
                self.servo_config.reset_pwm
            )

            self._current_pwm = (
                self.servo_config.reset_pwm
            )

            self._released = False

        logger.info(
            "[ServoPayloadDriver] "
            "Servo initialized "
            "channel=%d pwm=%d",
            self.servo_config.servo_channel,
            self.servo_config.reset_pwm,
        )

        return True

    # =========================================================================
    # Ready
    # =========================================================================

    def _hardware_ready(
        self,
    ) -> bool:
        """
        Check backend availability.
        """

        if self._backend is None:

            return False

        return True

    # =========================================================================
    # Release
    # =========================================================================

    def _release_hardware(
        self,
    ) -> bool:
        """
        Perform actual servo release.
        """

        with self._servo_lock:

            logger.info(
                "[ServoPayloadDriver] "
                "Release started "
                "channel=%d pwm=%d",
                self.servo_config.servo_channel,
                self.servo_config.release_pwm,
            )

            # ---------------------------------------------------------------
            # Release position
            # ---------------------------------------------------------------

            self._set_pwm(
                self.servo_config.release_pwm
            )

            self._current_pwm = (
                self.servo_config.release_pwm
            )

            self._released = True

            # ---------------------------------------------------------------
            # Keep servo in release position
            # ---------------------------------------------------------------

            if (
                self.servo_config
                .release_duration_s
                > 0
            ):

                time.sleep(
                    self.servo_config
                    .release_duration_s
                )

            # ---------------------------------------------------------------
            # Return to reset position
            # ---------------------------------------------------------------

            if self.servo_config.auto_reset:

                self._set_pwm(
                    self.servo_config.reset_pwm
                )

                self._current_pwm = (
                    self.servo_config.reset_pwm
                )

            logger.info(
                "[ServoPayloadDriver] "
                "Release completed"
            )

            return True

    # =========================================================================
    # Reset
    # =========================================================================

    def _reset_hardware(
        self,
    ) -> bool:
        """
        Return servo to safe/reset position.
        """

        with self._servo_lock:

            self._set_pwm(
                self.servo_config.reset_pwm
            )

            self._current_pwm = (
                self.servo_config.reset_pwm
            )

            self._released = False

        logger.info(
            "[ServoPayloadDriver] "
            "Servo reset "
            "channel=%d pwm=%d",
            self.servo_config.servo_channel,
            self.servo_config.reset_pwm,
        )

        return True

    # =========================================================================
    # Stop
    # =========================================================================

    def _stop_hardware(
        self,
    ) -> None:
        """
        Safely return servo to reset position.
        """

        try:

            with self._servo_lock:

                self._set_pwm(
                    self.servo_config.reset_pwm
                )

                self._current_pwm = (
                    self.servo_config.reset_pwm
                )

                self._released = False

        except Exception:

            logger.exception(
                "[ServoPayloadDriver] "
                "Failed to reset servo during stop"
            )

    # =========================================================================
    # PWM command
    # =========================================================================

    def _set_pwm(
        self,
        pwm: int,
    ) -> None:
        """
        Send PWM command to backend.
        """

        if not (
            self.servo_config.pwm_min
            <= pwm
            <= self.servo_config.pwm_max
        ):

            raise ServoCommandError(
                f"PWM {pwm} is outside valid range "
                f"{self.servo_config.pwm_min}-"
                f"{self.servo_config.pwm_max}"
            )

        # ---------------------------------------------------------------
        # Preferred backend API
        # ---------------------------------------------------------------

        method = getattr(
            self._backend,
            "set_servo_pwm",
            None,
        )

        if method is not None:

            try:

                result = method(
                    self.servo_config.servo_channel,
                    pwm,
                )

            except Exception as exc:

                raise ServoCommandError(
                    f"set_servo_pwm failed: {exc}"
                ) from exc

            if result is False:

                raise ServoCommandError(
                    "Backend rejected servo PWM command"
                )

            return

        # ---------------------------------------------------------------
        # Alternative backend API
        # ---------------------------------------------------------------

        method = getattr(
            self._backend,
            "set_servo",
            None,
        )

        if method is not None:

            try:

                result = method(
                    self.servo_config.servo_channel,
                    pwm,
                )

            except Exception as exc:

                raise ServoCommandError(
                    f"set_servo failed: {exc}"
                ) from exc

            if result is False:

                raise ServoCommandError(
                    "Backend rejected servo command"
                )

            return

        raise ServoCommandError(
            "Backend does not implement "
            "set_servo_pwm() or set_servo()"
        )

    # =========================================================================
    # Manual servo control
    # =========================================================================

    def set_pwm(
        self,
        pwm: int,
    ) -> bool:

        with self._servo_lock:

            if not self.initialized:

                raise ServoDriverNotReadyError(
                    "Servo driver is not initialized"
                )

            self._set_pwm(
                pwm
            )

            self._current_pwm = pwm

        return True

    # =========================================================================
    # Status
    # =========================================================================

    def servo_status(
        self,
    ) -> ServoStatus:
        """Return current servo status."""

        with self._servo_lock:

            return ServoStatus(
                channel=(
                    self.servo_config
                    .servo_channel
                ),

                pwm=self._current_pwm,

                initialized=self.initialized,

                ready=self.ready(),

                released=self._released,

                last_error=(
                    self._last_error
                ),
            )

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def servo_channel(
        self,
    ) -> int:

        return (
            self.servo_config
            .servo_channel
        )

    # -------------------------------------------------------------------------

    @property
    def current_pwm(
        self,
    ) -> int:

        with self._servo_lock:

            return self._current_pwm

    # -------------------------------------------------------------------------

    @property
    def released(
        self,
    ) -> bool:

        with self._servo_lock:

            return self._released

    # =========================================================================
    # Representation
    # =========================================================================

    def __repr__(
        self,
    ) -> str:

        status = (
            self.servo_status()
        )

        return (
            "ServoPayloadDriver("
            f"channel={status.channel}, "
            f"pwm={status.pwm}, "
            f"ready={status.ready}, "
            f"released={status.released}"
            ")"
        )
