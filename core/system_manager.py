"""Coordinates application-level system startup, health, and shutdown."""

from __future__ import annotations

import logging
import threading
import time

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional


logger = logging.getLogger(__name__)


# ============================================================================
# System state
# ============================================================================

class SystemState(str, Enum):
    """Overall system lifecycle state."""

    CREATED = "created"

    INITIALIZING = "initializing"

    READY = "ready"

    RUNNING = "running"

    PAUSED = "paused"

    STOPPING = "stopping"

    STOPPED = "stopped"

    ERROR = "error"


# ============================================================================
# Configuration
# ============================================================================

@dataclass(frozen=True)
class SystemManagerConfig:
    """
    System manager configuration.
    """

    startup_timeout_s: float = 10.0

    shutdown_timeout_s: float = 10.0

    require_safety_monitor: bool = True

    require_mission_manager: bool = True

    require_scheduler: bool = False

    require_watchdog: bool = False

    auto_start_scheduler: bool = True

    auto_start_watchdog: bool = True

    auto_start_safety: bool = True

    def __post_init__(self) -> None:

        if self.startup_timeout_s <= 0:

            raise ValueError(
                "startup_timeout_s must be greater than zero"
            )

        if self.shutdown_timeout_s <= 0:

            raise ValueError(
                "shutdown_timeout_s must be greater than zero"
            )


# ============================================================================
# System statistics
# ============================================================================

@dataclass
class SystemStatistics:
    """Runtime system statistics."""

    start_count: int = 0

    stop_count: int = 0

    failed_start_count: int = 0

    last_start_time: Optional[float] = None

    last_stop_time: Optional[float] = None

    last_error: Optional[str] = None


# ============================================================================
# System Manager
# ============================================================================

class SystemManager:

    name = "SystemManager"

    def __init__(
        self,
        mission_manager: Any = None,
        safety_monitor: Any = None,
        watchdog: Any = None,
        scheduler: Any = None,
        shared_data: Any = None,
        config: Optional[
            SystemManagerConfig
        ] = None,
    ) -> None:

        self._mission_manager = (
            mission_manager
        )

        self._safety_monitor = (
            safety_monitor
        )

        self._watchdog = (
            watchdog
        )

        self._scheduler = (
            scheduler
        )

        self._shared = (
            shared_data
        )

        self.config = (
            config
            or SystemManagerConfig()
        )

        self._lock = threading.RLock()

        self._state = (
            SystemState.CREATED
        )

        self._running = False

        self._statistics = (
            SystemStatistics()
        )

    # =========================================================================
    # Start
    # =========================================================================

    def start(self) -> bool:
        """
        Start complete system.

        Startup order:

            1. MissionManager
            2. SafetyMonitor
            3. MissionWatchdog
            4. MissionScheduler
        """

        with self._lock:

            if self._state in (
                SystemState.RUNNING,
                SystemState.READY,
            ):

                return True

            if self._state == (
                SystemState.INITIALIZING
            ):

                return False

            self._state = (
                SystemState.INITIALIZING
            )

        started = []

        try:

            # ---------------------------------------------------------------
            # MissionManager
            # ---------------------------------------------------------------

            if (
                self._mission_manager
                is not None
            ):

                self._start_component(
                    self._mission_manager,
                    "MissionManager",
                )

                started.append(
                    self._mission_manager
                )

            elif (
                self.config
                .require_mission_manager
            ):

                raise RuntimeError(
                    "MissionManager is required "
                    "but was not provided"
                )

            # ---------------------------------------------------------------
            # SafetyMonitor
            # ---------------------------------------------------------------

            if (
                self._safety_monitor
                is not None
                and self.config.auto_start_safety
            ):

                self._start_component(
                    self._safety_monitor,
                    "SafetyMonitor",
                )

                started.append(
                    self._safety_monitor
                )

            elif (
                self.config
                .require_safety_monitor
            ):

                raise RuntimeError(
                    "SafetyMonitor is required "
                    "but was not provided"
                )

            # ---------------------------------------------------------------
            # Watchdog
            # ---------------------------------------------------------------

            if (
                self._watchdog
                is not None
                and self.config.auto_start_watchdog
            ):

                self._start_component(
                    self._watchdog,
                    "MissionWatchdog",
                )

                started.append(
                    self._watchdog
                )

            elif (
                self.config
                .require_watchdog
            ):

                raise RuntimeError(
                    "MissionWatchdog is required "
                    "but was not provided"
                )

            # ---------------------------------------------------------------
            # Scheduler
            # ---------------------------------------------------------------

            if (
                self._scheduler
                is not None
                and self.config.auto_start_scheduler
            ):

                self._start_component(
                    self._scheduler,
                    "MissionScheduler",
                )

                started.append(
                    self._scheduler
                )

            elif (
                self.config
                .require_scheduler
            ):

                raise RuntimeError(
                    "MissionScheduler is required "
                    "but was not provided"
                )

            # ---------------------------------------------------------------
            # Ready
            # ---------------------------------------------------------------

            with self._lock:

                self._running = True

                self._state = (
                    SystemState.RUNNING
                )

                self._statistics.start_count += 1

                self._statistics.last_start_time = (
                    time.monotonic()
                )

                self._statistics.last_error = None

            logger.info(
                "[SystemManager] "
                "System started successfully"
            )

            return True

        except Exception as exc:

            logger.exception(
                "[SystemManager] "
                "System startup failed"
            )

            with self._lock:

                self._running = False

                self._state = (
                    SystemState.ERROR
                )

                self._statistics.failed_start_count += 1

                self._statistics.last_error = (
                    str(exc)
                )

            # ---------------------------------------------------------------
            # Roll back already started components.
            # ---------------------------------------------------------------

            self._stop_components_reverse(
                started
            )

            return False

    # =========================================================================
    # Component start
    # =========================================================================

    def _start_component(
        self,
        component: Any,
        name: str,
    ) -> None:
        """
        Start one component.

        Component must expose:

            start()
        """

        start_method = getattr(
            component,
            "start",
            None,
        )

        if not callable(
            start_method
        ):

            raise RuntimeError(
                f"{name} does not implement start()"
            )

        logger.info(
            "[SystemManager] "
            "Starting %s",
            name,
        )

        result = start_method()

        if result is False:

            raise RuntimeError(
                f"{name}.start() returned False"
            )

    # =========================================================================
    # Stop
    # =========================================================================

    def stop(self) -> bool:
        """
        Stop complete system.

        Shutdown order is reverse of startup:

            Scheduler
            Watchdog
            SafetyMonitor
            MissionManager
        """

        with self._lock:

            if self._state == (
                SystemState.STOPPED
            ):

                return True

            self._state = (
                SystemState.STOPPING
            )

            self._running = False

        components = [
            self._scheduler,
            self._watchdog,
            self._safety_monitor,
            self._mission_manager,
        ]

        success = True

        for component in components:

            if component is None:
                continue

            try:

                self._stop_component(
                    component
                )

            except Exception:

                success = False

                logger.exception(
                    "[SystemManager] "
                    "Component shutdown failed"
                )

        with self._lock:

            self._state = (
                SystemState.STOPPED
            )

            self._statistics.stop_count += 1

            self._statistics.last_stop_time = (
                time.monotonic()
            )

        logger.info(
            "[SystemManager] "
            "System stopped"
        )

        return success

    # =========================================================================
    # Component stop
    # =========================================================================

    def _stop_component(
        self,
        component: Any,
    ) -> None:
        """Stop one component."""

        stop_method = getattr(
            component,
            "stop",
            None,
        )

        if not callable(
            stop_method
        ):

            return

        name = getattr(
            component,
            "name",
            component.__class__.__name__,
        )

        logger.info(
            "[SystemManager] "
            "Stopping %s",
            name,
        )

        stop_method()

    # =========================================================================
    # Reverse stop
    # =========================================================================

    def _stop_components_reverse(
        self,
        components: list[Any],
    ) -> None:
        """Stop components in reverse startup order."""

        for component in reversed(
            components
        ):

            try:

                self._stop_component(
                    component
                )

            except Exception:

                logger.exception(
                    "[SystemManager] "
                    "Rollback shutdown failed"
                )

    # =========================================================================
    # Pause
    # =========================================================================

    def pause(self) -> bool:
        """
        Pause schedulable system activity.

        Safety monitoring remains active.
        """

        with self._lock:

            if not self._running:
                return False

            self._state = (
                SystemState.PAUSED
            )

        if self._scheduler is not None:

            pause_method = getattr(
                self._scheduler,
                "pause",
                None,
            )

            if callable(
                pause_method
            ):

                pause_method()

        if self._mission_manager is not None:

            pause_all = getattr(
                self._mission_manager,
                "pause_all",
                None,
            )

            if callable(
                pause_all
            ):

                pause_all()

        logger.info(
            "[SystemManager] "
            "System paused"
        )

        return True

    # =========================================================================
    # Resume
    # =========================================================================

    def resume(self) -> bool:
        """Resume system activity."""

        with self._lock:

            if not self._running:
                return False

            self._state = (
                SystemState.RUNNING
            )

        if self._mission_manager is not None:

            resume_all = getattr(
                self._mission_manager,
                "resume_all",
                None,
            )

            if callable(
                resume_all
            ):

                resume_all()

        if self._scheduler is not None:

            resume_method = getattr(
                self._scheduler,
                "resume",
                None,
            )

            if callable(
                resume_method
            ):

                resume_method()

        logger.info(
            "[SystemManager] "
            "System resumed"
        )

        return True

    # =========================================================================
    # Emergency stop
    # =========================================================================

    def emergency_stop(
        self,
        reason: str = "emergency_stop",
    ) -> None:

        logger.critical(
            "[SystemManager] "
            "EMERGENCY STOP: %s",
            reason,
        )

        if self._scheduler is not None:

            try:

                active_requests = getattr(
                    self._scheduler,
                    "active_requests",
                    None,
                )

                if callable(
                    active_requests
                ):

                    for request in (
                        active_requests()
                    ):

                        try:

                            self._scheduler.cancel(
                                request.request_id
                            )

                        except Exception:

                            logger.exception(
                                "[SystemManager] "
                                "Failed to cancel emergency "
                                "mission"
                            )

            except Exception:

                logger.exception(
                    "[SystemManager] "
                    "Emergency scheduler handling failed"
                )

        if self._mission_manager is not None:

            try:

                stop = getattr(
                    self._mission_manager,
                    "stop",
                    None,
                )

                if callable(stop):

                    stop()

            except Exception:

                logger.exception(
                    "[SystemManager] "
                    "Emergency MissionManager stop failed"
                )

    # =========================================================================
    # Health
    # =========================================================================

    def health(self) -> dict:
        """
        Return high-level system health information.
        """

        with self._lock:

            state = self._state

        safety_state = None

        if self._safety_monitor is not None:

            try:

                snapshot = (
                    self._safety_monitor
                    .snapshot()
                )

                safety_state = (
                    snapshot.state.value
                )

            except Exception:

                safety_state = "unknown"

        scheduler_state = None

        if self._scheduler is not None:

            try:

                scheduler_state = (
                    self._scheduler
                    .state.value
                )

            except Exception:

                scheduler_state = "unknown"

        watchdog_running = False

        if self._watchdog is not None:

            try:

                watchdog_running = bool(
                    self._watchdog
                    .is_running()
                )

            except Exception:

                watchdog_running = False

        return {
            "system_state": state.value,

            "running": (
                state
                == SystemState.RUNNING
            ),

            "safety_state": safety_state,

            "scheduler_state": scheduler_state,

            "watchdog_running": (
                watchdog_running
            ),
        }

    # =========================================================================
    # Statistics
    # =========================================================================

    def statistics(
        self,
    ) -> SystemStatistics:
        """Return system statistics snapshot."""

        with self._lock:

            return SystemStatistics(
                start_count=(
                    self._statistics.start_count
                ),

                stop_count=(
                    self._statistics.stop_count
                ),

                failed_start_count=(
                    self._statistics
                    .failed_start_count
                ),

                last_start_time=(
                    self._statistics
                    .last_start_time
                ),

                last_stop_time=(
                    self._statistics
                    .last_stop_time
                ),

                last_error=(
                    self._statistics
                    .last_error
                ),
            )

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def state(
        self,
    ) -> SystemState:

        with self._lock:
            return self._state

    # -------------------------------------------------------------------------

    @property
    def running(
        self,
    ) -> bool:

        with self._lock:
            return self._running

    # =========================================================================
    # Component access
    # =========================================================================

    @property
    def mission_manager(self) -> Any:

        return self._mission_manager

    # -------------------------------------------------------------------------

    @property
    def scheduler(self) -> Any:

        return self._scheduler

    # -------------------------------------------------------------------------

    @property
    def watchdog(self) -> Any:

        return self._watchdog

    # -------------------------------------------------------------------------

    @property
    def safety_monitor(self) -> Any:

        return self._safety_monitor

    # =========================================================================
    # Representation
    # =========================================================================

    def __repr__(self) -> str:

        return (
            "SystemManager("
            f"state={self.state.value!r}, "
            f"running={self.running}"
            ")"
        )
