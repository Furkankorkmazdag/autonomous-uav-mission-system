"""GPIO abstraction for platform-dependent hardware access."""

from __future__ import annotations

import logging
import threading
import time

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional


logger = logging.getLogger(__name__)


# ============================================================================
# GPIO Direction
# ============================================================================

class GPIODirection(str, Enum):
    """GPIO pin direction."""

    INPUT = "input"

    OUTPUT = "output"


# ============================================================================
# GPIO Pull
# ============================================================================

class GPIOPull(str, Enum):
    """GPIO pull configuration."""

    NONE = "none"

    UP = "up"

    DOWN = "down"


# ============================================================================
# GPIO State
# ============================================================================

class GPIOState(str, Enum):
    """GPIO runtime state."""

    UNKNOWN = "unknown"

    LOW = "low"

    HIGH = "high"

    DISABLED = "disabled"

    ERROR = "error"


# ============================================================================
# GPIO Configuration
# ============================================================================

@dataclass(frozen=True)
class GPIOConfig:

    name: str

    pin: int

    direction: GPIODirection

    pull: GPIOPull = GPIOPull.NONE

    active_high: bool = True

    initial_state: GPIOState = GPIOState.LOW

    enabled: bool = True

    def __post_init__(self) -> None:

        if not self.name:

            raise ValueError(
                "GPIO name cannot be empty"
            )

        if self.pin < 0:

            raise ValueError(
                "GPIO pin cannot be negative"
            )

        if (
            self.direction
            == GPIODirection.INPUT
            and self.initial_state
            != GPIOState.LOW
        ):

            raise ValueError(
                "Input GPIO cannot define "
                "an output initial state"
            )


# ============================================================================
# GPIO Snapshot
# ============================================================================

@dataclass(frozen=True)
class GPIOSnapshot:
    """Immutable GPIO state snapshot."""

    name: str

    pin: int

    direction: GPIODirection

    state: GPIOState

    enabled: bool

    last_update_time: Optional[float]

    last_error: Optional[str]


# ============================================================================
# GPIO Driver Interface
# ============================================================================

class GPIOHardwareDriver:

    def setup(
        self,
        pin: int,
        direction: GPIODirection,
        pull: GPIOPull = GPIOPull.NONE,
    ) -> None:
        """
        Configure GPIO pin.
        """

        raise NotImplementedError

    def write(
        self,
        pin: int,
        value: bool,
    ) -> None:
        """
        Write digital output.
        """

        raise NotImplementedError

    def read(
        self,
        pin: int,
    ) -> bool:
        """
        Read digital input.
        """

        raise NotImplementedError

    def cleanup(
        self,
        pin: int,
    ) -> None:
        """
        Release GPIO pin.
        """

        raise NotImplementedError

    def close(self) -> None:
        """
        Close driver.
        """


# ============================================================================
# Null GPIO Driver
# ============================================================================

class NullGPIOHardwareDriver(
    GPIOHardwareDriver
):

    def __init__(self) -> None:

        self._lock = threading.RLock()

        self._configured: Dict[
            int,
            GPIODirection,
        ] = {}

        self._values: Dict[
            int,
            bool,
        ] = {}

    # =========================================================================
    # Setup
    # =========================================================================

    def setup(
        self,
        pin: int,
        direction: GPIODirection,
        pull: GPIOPull = GPIOPull.NONE,
    ) -> None:

        with self._lock:

            self._configured[
                pin
            ] = direction

            if direction == (
                GPIODirection.OUTPUT
            ):

                self._values[
                    pin
                ] = False

    # =========================================================================
    # Write
    # =========================================================================

    def write(
        self,
        pin: int,
        value: bool,
    ) -> None:

        with self._lock:

            if pin not in self._configured:

                raise RuntimeError(
                    f"GPIO {pin} is not configured"
                )

            if (
                self._configured[pin]
                != GPIODirection.OUTPUT
            ):

                raise RuntimeError(
                    f"GPIO {pin} is not configured "
                    "as output"
                )

            self._values[
                pin
            ] = bool(value)

    # =========================================================================
    # Read
    # =========================================================================

    def read(
        self,
        pin: int,
    ) -> bool:

        with self._lock:

            if pin not in self._configured:

                raise RuntimeError(
                    f"GPIO {pin} is not configured"
                )

            return self._values.get(
                pin,
                False,
            )

    # =========================================================================
    # Cleanup
    # =========================================================================

    def cleanup(
        self,
        pin: int,
    ) -> None:

        with self._lock:

            self._configured.pop(
                pin,
                None,
            )

            self._values.pop(
                pin,
                None,
            )

    # =========================================================================
    # Close
    # =========================================================================

    def close(self) -> None:

        with self._lock:

            self._configured.clear()

            self._values.clear()


# ============================================================================
# GPIO Pin
# ============================================================================

class GPIOPin:

    def __init__(
        self,
        config: GPIOConfig,
        driver: GPIOHardwareDriver,
    ) -> None:

        self.config = config

        self._driver = driver

        self._lock = threading.RLock()

        self._state = (
            GPIOState.UNKNOWN
        )

        self._last_update_time: Optional[
            float
        ] = None

        self._last_error: Optional[
            str
        ] = None

    # =========================================================================
    # Initialize
    # =========================================================================

    def initialize(self) -> bool:
        """
        Configure GPIO.
        """

        if not self.config.enabled:

            with self._lock:

                self._state = (
                    GPIOState.DISABLED
                )

            return False

        try:

            self._driver.setup(
                pin=self.config.pin,

                direction=self.config.direction,

                pull=self.config.pull,
            )

            if (
                self.config.direction
                == GPIODirection.OUTPUT
            ):

                initial_high = (
                    self.config.initial_state
                    == GPIOState.HIGH
                )

                self._driver.write(
                    self.config.pin,
                    self._physical_value(
                        initial_high
                    ),
                )

                logical_state = (
                    GPIOState.HIGH
                    if initial_high
                    else GPIOState.LOW
                )

            else:

                value = (
                    self._driver.read(
                        self.config.pin
                    )
                )

                logical_value = (
                    self._logical_value(
                        value
                    )
                )

                logical_state = (
                    GPIOState.HIGH
                    if logical_value
                    else GPIOState.LOW
                )

            with self._lock:

                self._state = logical_state

                self._last_update_time = (
                    time.monotonic()
                )

                self._last_error = None

            return True

        except Exception as exc:

            self._set_error(
                exc
            )

            return False

    # =========================================================================
    # Write
    # =========================================================================

    def write(
        self,
        value: bool,
    ) -> bool:

        if (
            self.config.direction
            != GPIODirection.OUTPUT
        ):

            raise RuntimeError(
                f"GPIO {self.config.name} "
                "is not an output"
            )

        with self._lock:

            if (
                self._state
                == GPIOState.DISABLED
            ):

                return False

        try:

            physical_value = (
                self._physical_value(
                    bool(value)
                )
            )

            self._driver.write(
                self.config.pin,
                physical_value,
            )

            with self._lock:

                self._state = (
                    GPIOState.HIGH
                    if value
                    else GPIOState.LOW
                )

                self._last_update_time = (
                    time.monotonic()
                )

                self._last_error = None

            return True

        except Exception as exc:

            self._set_error(
                exc
            )

            return False

    # =========================================================================
    # Set HIGH
    # =========================================================================

    def high(self) -> bool:
        """Set logical HIGH."""

        return self.write(True)

    # =========================================================================
    # Set LOW
    # =========================================================================

    def low(self) -> bool:
        """Set logical LOW."""

        return self.write(False)

    # =========================================================================
    # Toggle
    # =========================================================================

    def toggle(self) -> bool:
        """Toggle output state."""

        current = self.read()

        return self.write(
            not current
        )

    # =========================================================================
    # Read
    # =========================================================================

    def read(self) -> bool:

        try:

            physical_value = (
                self._driver.read(
                    self.config.pin
                )
            )

            logical_value = (
                self._logical_value(
                    physical_value
                )
            )

            with self._lock:

                self._state = (
                    GPIOState.HIGH
                    if logical_value
                    else GPIOState.LOW
                )

                self._last_update_time = (
                    time.monotonic()
                )

                self._last_error = None

            return logical_value

        except Exception as exc:

            self._set_error(
                exc
            )

            return False

    # =========================================================================
    # Physical value
    # =========================================================================

    def _physical_value(
        self,
        logical_value: bool,
    ) -> bool:
        """
        Convert logical value to physical GPIO value.
        """

        if self.config.active_high:

            return logical_value

        return not logical_value

    # =========================================================================
    # Logical value
    # =========================================================================

    def _logical_value(
        self,
        physical_value: bool,
    ) -> bool:
        """
        Convert physical GPIO value to logical value.
        """

        if self.config.active_high:

            return physical_value

        return not physical_value

    # =========================================================================
    # Disable
    # =========================================================================

    def disable(self) -> bool:
        """
        Disable logical GPIO.
        """

        try:

            self._driver.cleanup(
                self.config.pin
            )

            with self._lock:

                self._state = (
                    GPIOState.DISABLED
                )

                self._last_update_time = (
                    time.monotonic()
                )

            return True

        except Exception as exc:

            self._set_error(
                exc
            )

            return False

    # =========================================================================
    # Error
    # =========================================================================

    def _set_error(
        self,
        exception: Exception,
    ) -> None:

        with self._lock:

            self._state = (
                GPIOState.ERROR
            )

            self._last_error = (
                str(exception)
            )

        logger.error(
            "[GPIO:%s] Operation failed: %s",
            self.config.name,
            exception,
        )

    # =========================================================================
    # Snapshot
    # =========================================================================

    def snapshot(
        self,
    ) -> GPIOSnapshot:

        with self._lock:

            return GPIOSnapshot(
                name=self.config.name,

                pin=self.config.pin,

                direction=self.config.direction,

                state=self._state,

                enabled=self.config.enabled,

                last_update_time=(
                    self._last_update_time
                ),

                last_error=(
                    self._last_error
                ),
            )

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def state(
        self,
    ) -> GPIOState:

        with self._lock:

            return self._state

    @property
    def pin(
        self,
    ) -> int:

        return self.config.pin


# ============================================================================
# GPIO Manager
# ============================================================================

class GPIOManager:

    name = "GPIOManager"

    def __init__(
        self,
        driver: Optional[
            GPIOHardwareDriver
        ] = None,
    ) -> None:

        self._lock = threading.RLock()

        self._driver = (
            driver
            or NullGPIOHardwareDriver()
        )

        self._pins: Dict[
            str,
            GPIOPin,
        ] = {}

        self._running = False

    # =========================================================================
    # Start
    # =========================================================================

    def start(self) -> bool:
        """
        Initialize all registered GPIO pins.
        """

        with self._lock:

            if self._running:

                return True

            self._running = True

            pins = list(
                self._pins.values()
            )

        success = True

        for pin in pins:

            if not pin.initialize():

                success = False

                logger.error(
                    "[GPIOManager] "
                    "Failed to initialize %s",
                    pin.config.name,
                )

        logger.info(
            "[GPIOManager] Started"
        )

        return success

    # =========================================================================
    # Stop
    # =========================================================================

    def stop(self) -> None:
        """
        Cleanup all GPIO pins.
        """

        with self._lock:

            if not self._running:

                return

            pins = list(
                self._pins.values()
            )

            self._running = False

        for pin in pins:

            try:

                pin.disable()

            except Exception:

                logger.exception(
                    "[GPIOManager] "
                    "Failed to cleanup %s",
                    pin.config.name,
                )

        try:

            self._driver.close()

        except Exception:

            logger.exception(
                "[GPIOManager] "
                "Failed to close GPIO driver"
            )

        logger.info(
            "[GPIOManager] Stopped"
        )

    # =========================================================================
    # Register
    # =========================================================================

    def register(
        self,
        config: GPIOConfig,
    ) -> GPIOPin:
        """
        Register a GPIO pin.
        """

        with self._lock:

            if config.name in self._pins:

                raise ValueError(
                    f"GPIO already registered: "
                    f"{config.name}"
                )

            for existing in (
                self._pins.values()
            ):

                if (
                    existing.config.pin
                    == config.pin
                ):

                    raise ValueError(
                        f"GPIO pin {config.pin} "
                        "is already registered as "
                        f"{existing.config.name}"
                    )

            gpio = GPIOPin(
                config=config,

                driver=self._driver,
            )

            self._pins[
                config.name
            ] = gpio

            running = self._running

        if running:

            gpio.initialize()

        logger.info(
            "[GPIOManager] "
            "Registered %s -> GPIO %d",
            config.name,
            config.pin,
        )

        return gpio

    # =========================================================================
    # Get
    # =========================================================================

    def get(
        self,
        name: str,
    ) -> Optional[GPIOPin]:
        """Get GPIO by logical name."""

        with self._lock:

            return self._pins.get(
                name
            )

    # =========================================================================
    # Write
    # =========================================================================

    def write(
        self,
        name: str,
        value: bool,
    ) -> bool:
        """Write GPIO."""

        gpio = self.get(
            name
        )

        if gpio is None:

            logger.error(
                "[GPIOManager] "
                "Unknown GPIO: %s",
                name,
            )

            return False

        return gpio.write(
            value
        )

    # =========================================================================
    # HIGH
    # =========================================================================

    def high(
        self,
        name: str,
    ) -> bool:
        """Set GPIO HIGH."""

        return self.write(
            name,
            True,
        )

    # =========================================================================
    # LOW
    # =========================================================================

    def low(
        self,
        name: str,
    ) -> bool:
        """Set GPIO LOW."""

        return self.write(
            name,
            False,
        )

    # =========================================================================
    # Toggle
    # =========================================================================

    def toggle(
        self,
        name: str,
    ) -> bool:
        """Toggle GPIO."""

        gpio = self.get(
            name
        )

        if gpio is None:

            return False

        return gpio.toggle()

    # =========================================================================
    # Read
    # =========================================================================

    def read(
        self,
        name: str,
    ) -> bool:
        """Read GPIO."""

        gpio = self.get(
            name
        )

        if gpio is None:

            logger.error(
                "[GPIOManager] "
                "Unknown GPIO: %s",
                name,
            )

            return False

        return gpio.read()

    # =========================================================================
    # Disable
    # =========================================================================

    def disable(
        self,
        name: str,
    ) -> bool:
        """Disable GPIO."""

        gpio = self.get(
            name
        )

        if gpio is None:

            return False

        return gpio.disable()

    # =========================================================================
    # Snapshot
    # =========================================================================

    def all(
        self,
    ) -> Dict[
        str,
        GPIOSnapshot,
    ]:
        """Return all GPIO snapshots."""

        with self._lock:

            return {
                name: gpio.snapshot()
                for name, gpio
                in self._pins.items()
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
                "GPIOManager("
                f"running={self._running}, "
                f"pins="
                f"{len(self._pins)}"
                ")"
            )
