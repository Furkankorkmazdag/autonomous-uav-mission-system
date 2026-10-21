"""Base lifecycle component for long-running SUAS software services."""

from __future__ import annotations

import logging
import threading
import time

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional


logger = logging.getLogger(__name__)


# ============================================================================
# Component State
# ============================================================================

class ComponentState(str, Enum):
    """Common lifecycle states."""

    CREATED = "created"

    STARTING = "starting"

    RUNNING = "running"

    STOPPING = "stopping"

    STOPPED = "stopped"

    FAILED = "failed"


# ============================================================================
# Component Statistics
# ============================================================================

@dataclass
class ComponentStatistics:
    """Runtime statistics of a component."""

    start_count: int = 0

    stop_count: int = 0

    failure_count: int = 0

    last_start_time: Optional[float] = None

    last_stop_time: Optional[float] = None

    last_failure_time: Optional[float] = None

    last_error: Optional[str] = None


# ============================================================================
# Component
# ============================================================================

class Component:
    """Base class for thread-safe component lifecycle management."""

    name = "Component"

    # =========================================================================
    # Constructor
    # =========================================================================

    def __init__(
        self,
        name: Optional[str] = None,
    ) -> None:

        self.name = (
            name
            or self.__class__.__name__
        )

        self._lock = threading.RLock()

        self._state = (
            ComponentState.CREATED
        )

        self._thread: Optional[
            threading.Thread
        ] = None

        self._stop_event = (
            threading.Event()
        )

        self._started_event = (
            threading.Event()
        )

        self._failure_event = (
            threading.Event()
        )

        self._statistics = (
            ComponentStatistics()
        )

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def start(self) -> bool:
        """
        Start component.

        Lifecycle:

            CREATED / STOPPED
                    ↓
                STARTING
                    ↓
                 RUNNING
        """

        with self._lock:

            if self._state == (
                ComponentState.RUNNING
            ):

                return True

            if self._state in (
                ComponentState.STARTING,
                ComponentState.STOPPING,
            ):

                return False

            self._state = (
                ComponentState.STARTING
            )

            self._stop_event.clear()

            self._started_event.clear()

            self._failure_event.clear()

        try:

            self._on_start()

            with self._lock:

                self._state = (
                    ComponentState.RUNNING
                )

                self._statistics.start_count += 1

                self._statistics.last_start_time = (
                    time.monotonic()
                )

                self._statistics.last_error = None

            self._started_event.set()

            # ---------------------------------------------------------------
            # Background worker
            # ---------------------------------------------------------------

            if self._has_background_loop():

                self._thread = (
                    threading.Thread(
                        target=self._thread_entry,
                        name=self.name,
                        daemon=True,
                    )
                )

                self._thread.start()

            logger.info(
                "[%s] Started",
                self.name,
            )

            return True

        except Exception as exc:

            self._handle_failure(
                exc
            )

            return False

    # =========================================================================
    # Stop
    # =========================================================================

    def stop(
        self,
        timeout_s: float = 5.0,
    ) -> bool:
        """
        Stop component gracefully.
        """

        if timeout_s < 0:

            raise ValueError(
                "timeout_s cannot be negative"
            )

        with self._lock:

            if self._state == (
                ComponentState.STOPPED
            ):

                return True

            if self._state == (
                ComponentState.CREATED
            ):

                self._state = (
                    ComponentState.STOPPED
                )

                return True

            self._state = (
                ComponentState.STOPPING
            )

            self._stop_event.set()

            thread = self._thread

        success = True

        try:

            self._on_stop()

        except Exception as exc:

            success = False

            logger.exception(
                "[%s] Stop callback failed",
                self.name,
            )

            with self._lock:

                self._statistics.last_error = (
                    str(exc)
                )

        # ---------------------------------------------------------------------
        # Wait for background worker
        # ---------------------------------------------------------------------

        if (
            thread is not None
            and thread.is_alive()
            and thread
            is not threading.current_thread()
        ):

            thread.join(
                timeout=timeout_s
            )

            if thread.is_alive():

                success = False

                logger.error(
                    "[%s] "
                    "Worker thread did not stop "
                    "within %.2fs",
                    self.name,
                    timeout_s,
                )

        with self._lock:

            self._state = (
                ComponentState.STOPPED
            )

            self._statistics.stop_count += 1

            self._statistics.last_stop_time = (
                time.monotonic()
            )

        logger.info(
            "[%s] Stopped",
            self.name,
        )

        return success

    # =========================================================================
    # Restart
    # =========================================================================

    def restart(
        self,
        timeout_s: float = 5.0,
    ) -> bool:
        """
        Stop and start component again.
        """

        logger.warning(
            "[%s] Restart requested",
            self.name,
        )

        self.stop(
            timeout_s=timeout_s
        )

        return self.start()

    # =========================================================================
    # Thread entry
    # =========================================================================

    def _thread_entry(self) -> None:

        try:

            self._run()

        except Exception as exc:

            logger.exception(
                "[%s] Background worker failed",
                self.name,
            )

            self._handle_failure(
                exc
            )

    # =========================================================================
    # Background loop
    # =========================================================================

    def _run(self) -> None:

        while not self._stop_event.wait(
            1.0
        ):

            pass

    # =========================================================================
    # Start callback
    # =========================================================================

    def _on_start(self) -> None:
        pass

    # =========================================================================
    # Stop callback
    # =========================================================================

    def _on_stop(self) -> None:
        pass

    # =========================================================================
    # Failure
    # =========================================================================

    def _handle_failure(
        self,
        exception: Exception,
    ) -> None:
        """
        Put component into FAILED state.
        """

        now = time.monotonic()

        with self._lock:

            self._state = (
                ComponentState.FAILED
            )

            self._failure_event.set()

            self._stop_event.set()

            self._statistics.failure_count += 1

            self._statistics.last_failure_time = (
                now
            )

            self._statistics.last_error = (
                str(exception)
            )

        logger.error(
            "[%s] FAILED: %s",
            self.name,
            exception,
        )

    # =========================================================================
    # Wait until started
    # =========================================================================

    def wait_until_started(
        self,
        timeout_s: Optional[float] = None,
    ) -> bool:
        """
        Wait until component reaches RUNNING state.
        """

        if not self._started_event.wait(
            timeout=timeout_s
        ):

            return False

        return self.is_running()

    # =========================================================================
    # Wait until failure
    # =========================================================================

    def wait_until_failed(
        self,
        timeout_s: Optional[float] = None,
    ) -> bool:
        """
        Wait until component fails.
        """

        return self._failure_event.wait(
            timeout=timeout_s
        )

    # =========================================================================
    # Background loop detection
    # =========================================================================

    def _has_background_loop(
        self,
    ) -> bool:

        return True

    # =========================================================================
    # State
    # =========================================================================

    def is_running(self) -> bool:
        """Return True when component is running."""

        with self._lock:

            return (
                self._state
                == ComponentState.RUNNING
            )

    # -------------------------------------------------------------------------

    def is_stopped(self) -> bool:
        """Return True when component is stopped."""

        with self._lock:

            return (
                self._state
                == ComponentState.STOPPED
            )

    # -------------------------------------------------------------------------

    def is_failed(self) -> bool:
        """Return True when component has failed."""

        with self._lock:

            return (
                self._state
                == ComponentState.FAILED
            )

    # =========================================================================
    # Health
    # =========================================================================

    def health(self) -> str:
        """
        Return simple health state.

        HealthMonitor may use this method.
        """

        with self._lock:

            if self._state == (
                ComponentState.RUNNING
            ):

                return "healthy"

            if self._state == (
                ComponentState.FAILED
            ):

                return "failed"

            if self._state == (
                ComponentState.STOPPED
            ):

                return "stopped"

            return "unknown"

    # =========================================================================
    # State property
    # =========================================================================

    @property
    def state(
        self,
    ) -> ComponentState:

        with self._lock:

            return self._state

    # =========================================================================
    # Running property
    # =========================================================================

    @property
    def running(
        self,
    ) -> bool:

        return self.is_running()

    # =========================================================================
    # Stop requested
    # =========================================================================

    @property
    def stop_requested(
        self,
    ) -> bool:

        return self._stop_event.is_set()

    # =========================================================================
    # Statistics
    # =========================================================================

    def statistics(
        self,
    ) -> ComponentStatistics:
        """Return immutable-like statistics snapshot."""

        with self._lock:

            return ComponentStatistics(
                start_count=(
                    self._statistics.start_count
                ),

                stop_count=(
                    self._statistics.stop_count
                ),

                failure_count=(
                    self._statistics.failure_count
                ),

                last_start_time=(
                    self._statistics
                    .last_start_time
                ),

                last_stop_time=(
                    self._statistics
                    .last_stop_time
                ),

                last_failure_time=(
                    self._statistics
                    .last_failure_time
                ),

                last_error=(
                    self._statistics
                    .last_error
                ),
            )

    # =========================================================================
    # Thread
    # =========================================================================

    @property
    def thread(
        self,
    ) -> Optional[
        threading.Thread
    ]:

        with self._lock:

            return self._thread

    # =========================================================================
    # Representation
    # =========================================================================

    def __repr__(
        self,
    ) -> str:

        return (
            f"{self.__class__.__name__}("
            f"name={self.name!r}, "
            f"state={self.state.value!r}"
            ")"
        )
