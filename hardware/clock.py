"""Clock abstraction used by hardware-dependent code and tests."""

from __future__ import annotations

import logging
import threading
import time

from dataclasses import dataclass
from typing import Optional


logger = logging.getLogger(__name__)


# ============================================================================
# Clock Interface
# ============================================================================

class Clock:

    def monotonic(self) -> float:

        raise NotImplementedError

    def time(self) -> float:
        """
        Unix epoch time in seconds.
        """

        raise NotImplementedError

    def sleep(
        self,
        seconds: float,
    ) -> None:
        """
        Sleep for specified seconds.
        """

        raise NotImplementedError


# ============================================================================
# System Clock
# ============================================================================

class SystemClock(Clock):

    def monotonic(self) -> float:
        return time.monotonic()

    def time(self) -> float:
        return time.time()

    def sleep(
        self,
        seconds: float,
    ) -> None:

        if seconds < 0:

            raise ValueError(
                "seconds cannot be negative"
            )

        time.sleep(
            seconds
        )


# ============================================================================
# Simulation Clock
# ============================================================================

@dataclass
class SimulationClock(Clock):

    initial_time: float = 0.0

    def __post_init__(self) -> None:

        self._lock = threading.RLock()

        self._monotonic_time = (
            float(self.initial_time)
        )

        self._epoch_time = (
            float(self.initial_time)
        )

        self._condition = (
            threading.Condition(
                self._lock
            )
        )

    # =========================================================================
    # Monotonic
    # =========================================================================

    def monotonic(self) -> float:

        with self._lock:

            return self._monotonic_time

    # =========================================================================
    # Epoch
    # =========================================================================

    def time(self) -> float:

        with self._lock:

            return self._epoch_time

    # =========================================================================
    # Sleep
    # =========================================================================

    def sleep(
        self,
        seconds: float,
    ) -> None:

        if seconds < 0:

            raise ValueError(
                "seconds cannot be negative"
            )

        if seconds == 0:

            return

        with self._condition:

            target = (
                self._monotonic_time
                + seconds
            )

            while (
                self._monotonic_time
                < target
            ):

                self._condition.wait()

    # =========================================================================
    # Advance
    # =========================================================================

    def advance(
        self,
        seconds: float,
    ) -> None:

        if seconds < 0:

            raise ValueError(
                "seconds cannot be negative"
            )

        with self._condition:

            self._monotonic_time += (
                seconds
            )

            self._epoch_time += (
                seconds
            )

            self._condition.notify_all()

    # =========================================================================
    # Set
    # =========================================================================

    def set_time(
        self,
        value: float,
    ) -> None:

        with self._condition:

            value = float(value)

            if (
                value
                < self._monotonic_time
            ):

                raise ValueError(
                    "Simulation monotonic time "
                    "cannot move backwards"
                )

            delta = (
                value
                - self._monotonic_time
            )

            self._monotonic_time = value

            self._epoch_time += delta

            self._condition.notify_all()


# ============================================================================
# Clock Provider
# ============================================================================

class ClockProvider:

    _lock = threading.RLock()

    _clock: Clock = SystemClock()

    # =========================================================================
    # Get
    # =========================================================================

    @classmethod
    def get(
        cls,
    ) -> Clock:

        with cls._lock:

            return cls._clock

    # =========================================================================
    # Set
    # =========================================================================

    @classmethod
    def set_clock(
        cls,
        clock: Clock,
    ) -> None:

        if clock is None:

            raise ValueError(
                "clock cannot be None"
            )

        with cls._lock:

            cls._clock = clock

        logger.info(
            "[ClockProvider] "
            "Clock changed to %s",
            type(clock).__name__,
        )

    # =========================================================================
    # Reset
    # =========================================================================

    @classmethod
    def reset(
        cls,
    ) -> None:

        with cls._lock:

            cls._clock = SystemClock()

        logger.info(
            "[ClockProvider] "
            "Clock reset to SystemClock"
        )


# ============================================================================
# Convenience Functions
# ============================================================================

def monotonic() -> float:
    """
    Return current monotonic time.
    """

    return ClockProvider.get().monotonic()


def now() -> float:
    """
    Return current Unix epoch time.
    """

    return ClockProvider.get().time()


def sleep(
    seconds: float,
) -> None:
    """
    Sleep using current clock.
    """

    ClockProvider.get().sleep(
        seconds
    )


# ============================================================================
# Deadline
# ============================================================================

@dataclass(frozen=True)
class Deadline:

    timestamp: float

    # =========================================================================
    # Create
    # =========================================================================

    @classmethod
    def after(
        cls,
        seconds: float,
        clock: Optional[
            Clock
        ] = None,
    ) -> "Deadline":

        if seconds < 0:

            raise ValueError(
                "seconds cannot be negative"
            )

        active_clock = (
            clock
            or ClockProvider.get()
        )

        return cls(
            timestamp=(
                active_clock.monotonic()
                + seconds
            )
        )

    # =========================================================================
    # Expired
    # =========================================================================

    def expired(
        self,
        clock: Optional[
            Clock
        ] = None,
    ) -> bool:

        active_clock = (
            clock
            or ClockProvider.get()
        )

        return (
            active_clock.monotonic()
            >= self.timestamp
        )

    # =========================================================================
    # Remaining
    # =========================================================================

    def remaining(
        self,
        clock: Optional[
            Clock
        ] = None,
    ) -> float:

        active_clock = (
            clock
            or ClockProvider.get()
        )

        return max(
            0.0,
            self.timestamp
            - active_clock.monotonic(),
        )


# ============================================================================
# Rate Limiter
# ============================================================================

class RateLimiter:

    def __init__(
        self,
        interval_s: float,
        clock: Optional[
            Clock
        ] = None,
    ) -> None:

        if interval_s < 0:

            raise ValueError(
                "interval_s cannot be negative"
            )

        self.interval_s = (
            float(interval_s)
        )

        self._clock = (
            clock
            or ClockProvider.get()
        )

        self._lock = threading.RLock()

        self._last_time: Optional[
            float
        ] = None

    # =========================================================================
    # Ready
    # =========================================================================

    def ready(self) -> bool:

        with self._lock:

            now_value = (
                self._clock.monotonic()
            )

            if self._last_time is None:

                return True

            return (
                now_value
                - self._last_time
                >= self.interval_s
            )

    # =========================================================================
    # Trigger
    # =========================================================================

    def trigger(self) -> bool:

        with self._lock:

            if not self.ready():

                return False

            self._last_time = (
                self._clock.monotonic()
            )

            return True

    # =========================================================================
    # Reset
    # =========================================================================

    def reset(self) -> None:

        with self._lock:

            self._last_time = None


# ============================================================================
# Stopwatch
# ============================================================================

class Stopwatch:

    def __init__(
        self,
        clock: Optional[
            Clock
        ] = None,
    ) -> None:

        self._clock = (
            clock
            or ClockProvider.get()
        )

        self._lock = threading.RLock()

        self._start_time: Optional[
            float
        ] = None

        self._stop_time: Optional[
            float
        ] = None

    # =========================================================================
    # Start
    # =========================================================================

    def start(self) -> None:

        with self._lock:

            self._start_time = (
                self._clock.monotonic()
            )

            self._stop_time = None

    # =========================================================================
    # Stop
    # =========================================================================

    def stop(self) -> float:

        with self._lock:

            if self._start_time is None:

                raise RuntimeError(
                    "Stopwatch has not been started"
                )

            self._stop_time = (
                self._clock.monotonic()
            )

            return (
                self._stop_time
                - self._start_time
            )

    # =========================================================================
    # Elapsed
    # =========================================================================

    def elapsed(self) -> float:

        with self._lock:

            if self._start_time is None:

                return 0.0

            end_time = (
                self._stop_time
                if self._stop_time is not None
                else self._clock.monotonic()
            )

            return max(
                0.0,
                end_time
                - self._start_time,
            )

    # =========================================================================
    # Running
    # =========================================================================

    @property
    def running(self) -> bool:

        with self._lock:

            return (
                self._start_time is not None
                and self._stop_time is None
            )

    # =========================================================================
    # Reset
    # =========================================================================

    def reset(self) -> None:

        with self._lock:

            self._start_time = None

            self._stop_time = None
