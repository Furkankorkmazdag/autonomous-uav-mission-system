"""Hardware-facing abstraction for payload initialization, release, reset, and shutdown."""

from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional


logger = logging.getLogger(__name__)


# ============================================================================
# Exceptions
# ============================================================================

class PayloadDriverError(Exception):
    """Base payload driver exception."""


class PayloadDriverNotReadyError(
    PayloadDriverError
):
    """Raised when payload hardware is not ready."""


class PayloadDriverReleaseError(
    PayloadDriverError
):
    """Raised when payload release fails."""


class PayloadDriverInitializationError(
    PayloadDriverError
):
    """Raised when driver initialization fails."""


# ============================================================================
# Driver State
# ============================================================================

class PayloadDriverState(str, Enum):
    """Hardware driver state."""

    CREATED = "created"

    INITIALIZING = "initializing"

    READY = "ready"

    BUSY = "busy"

    RELEASED = "released"

    ERROR = "error"

    STOPPED = "stopped"


# ============================================================================
# Driver Configuration
# ============================================================================

@dataclass(frozen=True)
class PayloadDriverConfig:
    """
    Generic payload driver configuration.

    Hardware-specific parameters should normally live in the
    concrete driver implementation or central configuration.
    """

    enabled: bool = True

    release_timeout_s: float = 5.0

    initialize_timeout_s: float = 5.0

    allow_release_without_ready: bool = False

    reset_after_release: bool = False

    def __post_init__(self) -> None:

        self.validate()

    def validate(self) -> None:
        """Validate driver configuration."""

        if self.release_timeout_s <= 0:

            raise ValueError(
                "release_timeout_s must be greater than zero"
            )

        if self.initialize_timeout_s <= 0:

            raise ValueError(
                "initialize_timeout_s must be greater than zero"
            )


# ============================================================================
# Driver Status
# ============================================================================

@dataclass(frozen=True)
class PayloadDriverStatus:
    """Immutable snapshot of hardware driver state."""

    state: PayloadDriverState

    ready: bool

    enabled: bool

    release_count: int

    last_error: Optional[str]

    initialized: bool


# ============================================================================
# Abstract Driver
# ============================================================================

class PayloadDriver(ABC):

    def __init__(
        self,
        config: Optional[
            PayloadDriverConfig
        ] = None,
    ) -> None:

        self.config = (
            config
            or PayloadDriverConfig()
        )

        self._lock = threading.RLock()

        self._state = (
            PayloadDriverState.CREATED
        )

        self._initialized = False

        self._release_count = 0

        self._last_error: Optional[str] = None

    # =========================================================================
    # Initialization
    # =========================================================================

    def initialize(self) -> bool:
        """
        Initialize hardware driver.

        Returns True if successful.
        """

        with self._lock:

            if not self.config.enabled:

                self._state = (
                    PayloadDriverState.STOPPED
                )

                return False

            if self._initialized:

                return True

            self._state = (
                PayloadDriverState.INITIALIZING
            )

        try:

            success = self._initialize_hardware()

            if not success:

                raise PayloadDriverInitializationError(
                    "Hardware initialization returned False"
                )

            with self._lock:

                self._initialized = True

                self._state = (
                    PayloadDriverState.READY
                )

                self._last_error = None

            logger.info(
                "[PayloadDriver] Initialized"
            )

            return True

        except Exception as exc:

            with self._lock:

                self._initialized = False

                self._state = (
                    PayloadDriverState.ERROR
                )

                self._last_error = str(
                    exc
                )

            logger.exception(
                "[PayloadDriver] "
                "Initialization failed"
            )

            return False

    # =========================================================================
    # Hardware initialization
    # =========================================================================

    @abstractmethod
    def _initialize_hardware(
        self,
    ) -> bool:
        """
        Concrete driver hardware initialization.

        Examples:

            GPIO setup
            Servo initialization
            MAVLink peripheral initialization
        """

        raise NotImplementedError

    # =========================================================================
    # Ready
    # =========================================================================

    def ready(self) -> bool:
        """
        Return whether payload hardware is ready.
        """

        with self._lock:

            if not self.config.enabled:
                return False

            if not self._initialized:
                return False

            if self._state in (
                PayloadDriverState.ERROR,
                PayloadDriverState.STOPPED,
            ):
                return False

        try:

            hardware_ready = (
                self._hardware_ready()
            )

            if not hardware_ready:

                return False

            return True

        except Exception as exc:

            with self._lock:

                self._state = (
                    PayloadDriverState.ERROR
                )

                self._last_error = str(
                    exc
                )

            logger.exception(
                "[PayloadDriver] "
                "Hardware readiness check failed"
            )

            return False

    # =========================================================================
    # Hardware ready
    # =========================================================================

    def _hardware_ready(
        self,
    ) -> bool:
        """
        Hardware-specific readiness check.

        Default implementation considers the driver ready
        when initialized.
        """

        return True

    # =========================================================================
    # Release
    # =========================================================================

    def release(self) -> bool:
        """
        Release payload.

        Returns True when release succeeds.
        """

        with self._lock:

            if not self.config.enabled:

                raise PayloadDriverNotReadyError(
                    "Payload driver is disabled"
                )

            if not self._initialized:

                raise PayloadDriverNotReadyError(
                    "Payload driver is not initialized"
                )

            if (
                not self.config
                .allow_release_without_ready
            ):

                if not self.ready():

                    raise PayloadDriverNotReadyError(
                        "Payload driver is not ready"
                    )

            if self._state == (
                PayloadDriverState.BUSY
            ):

                raise PayloadDriverReleaseError(
                    "Payload driver is already busy"
                )

            self._state = (
                PayloadDriverState.BUSY
            )

        try:

            success = self._release_hardware()

            if not success:

                raise PayloadDriverReleaseError(
                    "Hardware release returned False"
                )

            with self._lock:

                self._release_count += 1

                self._state = (
                    PayloadDriverState.RELEASED
                )

                self._last_error = None

            logger.info(
                "[PayloadDriver] "
                "Payload released"
            )

            if self.config.reset_after_release:

                self.reset()

            return True

        except Exception as exc:

            with self._lock:

                self._state = (
                    PayloadDriverState.ERROR
                )

                self._last_error = str(
                    exc
                )

            if isinstance(
                exc,
                PayloadDriverError,
            ):

                raise

            raise PayloadDriverReleaseError(
                str(exc)
            ) from exc

    # =========================================================================
    # Hardware release
    # =========================================================================

    @abstractmethod
    def _release_hardware(
        self,
    ) -> bool:

        raise NotImplementedError

    # =========================================================================
    # Reset
    # =========================================================================

    def reset(self) -> bool:
        """
        Reset payload hardware.
        """

        with self._lock:

            if not self._initialized:

                return False

        try:

            success = self._reset_hardware()

            if not success:

                raise PayloadDriverError(
                    "Hardware reset returned False"
                )

            with self._lock:

                self._state = (
                    PayloadDriverState.READY
                )

                self._last_error = None

            logger.info(
                "[PayloadDriver] Reset complete"
            )

            return True

        except Exception as exc:

            with self._lock:

                self._state = (
                    PayloadDriverState.ERROR
                )

                self._last_error = str(
                    exc
                )

            logger.exception(
                "[PayloadDriver] "
                "Reset failed"
            )

            return False

    # =========================================================================
    # Hardware reset
    # =========================================================================

    def _reset_hardware(
        self,
    ) -> bool:
        """
        Hardware-specific reset.

        Default implementation does nothing.
        """

        return True

    # =========================================================================
    # Stop
    # =========================================================================

    def stop(self) -> None:
        """
        Stop hardware driver safely.
        """

        try:

            self._stop_hardware()

        except Exception as exc:

            logger.exception(
                "[PayloadDriver] "
                "Hardware stop failed"
            )

            with self._lock:

                self._last_error = str(
                    exc
                )

        finally:

            with self._lock:

                self._initialized = False

                self._state = (
                    PayloadDriverState.STOPPED
                )

    # =========================================================================
    # Hardware stop
    # =========================================================================

    def _stop_hardware(
        self,
    ) -> None:
        """
        Hardware-specific shutdown.

        Default implementation does nothing.
        """

    # =========================================================================
    # Status
    # =========================================================================

    def status(
        self,
    ) -> PayloadDriverStatus:
        """
        Return immutable driver status.
        """

        with self._lock:

            return PayloadDriverStatus(
                state=self._state,

                ready=(
                    self.ready()
                ),

                enabled=(
                    self.config.enabled
                ),

                release_count=(
                    self._release_count
                ),

                last_error=(
                    self._last_error
                ),

                initialized=(
                    self._initialized
                ),
            )

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def state(
        self,
    ) -> PayloadDriverState:

        with self._lock:

            return self._state

    # -------------------------------------------------------------------------

    @property
    def release_count(
        self,
    ) -> int:

        with self._lock:

            return self._release_count

    # -------------------------------------------------------------------------

    @property
    def initialized(
        self,
    ) -> bool:

        with self._lock:

            return self._initialized

    # =========================================================================
    # Context manager
    # =========================================================================

    def __enter__(
        self,
    ) -> "PayloadDriver":

        if not self.initialize():

            raise PayloadDriverInitializationError(
                "Failed to initialize payload driver"
            )

        return self

    # -------------------------------------------------------------------------

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> None:

        self.stop()

    # =========================================================================
    # Representation
    # =========================================================================

    def __repr__(self) -> str:

        status = self.status()

        return (
            "PayloadDriver("
            f"state={status.state.value!r}, "
            f"ready={status.ready}, "
            f"releases={status.release_count}"
            ")"
        )


# ============================================================================
# Null Driver
# ============================================================================

class NullPayloadDriver(
    PayloadDriver
):

    def _initialize_hardware(
        self,
    ) -> bool:

        logger.info(
            "[NullPayloadDriver] "
            "Initialized"
        )

        return True

    def _hardware_ready(
        self,
    ) -> bool:

        return True

    def _release_hardware(
        self,
    ) -> bool:

        logger.info(
            "[NullPayloadDriver] "
            "Simulated payload release"
        )

        return True

    def _reset_hardware(
        self,
    ) -> bool:

        logger.info(
            "[NullPayloadDriver] "
            "Simulated reset"
        )

        return True

    def _stop_hardware(
        self,
    ) -> None:

        logger.info(
            "[NullPayloadDriver] "
            "Stopped"
        )
