"""Application composition root for the SUAS flight computer. Creates shared services and wires mission components together."""

from __future__ import annotations

import logging
import signal
import threading
from dataclasses import dataclass
from typing import Any, Optional

from config import ManagerConfig
from core.events import EventBus
from core.shared_data import SharedData


logger = logging.getLogger(__name__)


# ============================================================================
# Application State
# ============================================================================

class ApplicationState:
    CREATED = "created"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


# ============================================================================
# Shared Context
# ============================================================================

@dataclass
class SharedContext:
    """Container for application-wide shared services and state."""

    # Central configuration
    config: ManagerConfig

    # EventBus
    events: EventBus

    # Shared live flight data.
    shared_data: SharedData

    # Services
    mavlink: Any = None
    servo_manager: Any = None
    gpio_manager: Any = None
    mission_manager: Any = None
    health_monitor: Any = None
    autonomous_mission: Any = None

    # Lifecycle
    lifecycle: Any = None


# ============================================================================
# Application Configuration
# ============================================================================

@dataclass(frozen=True)
class ApplicationConfig:
    """Configuration for application-level services."""

    shutdown_timeout_s: float = 10.0

    install_signal_handlers: bool = True

    enable_gpio: bool = True

    enable_servos: bool = True

    enable_health_monitor: bool = True

    def __post_init__(self) -> None:

        if self.shutdown_timeout_s <= 0:
            raise ValueError(
                "shutdown_timeout_s must be greater than zero"
            )


# ============================================================================
# Application
# ============================================================================

class Application:
    """Builds, starts, and stops the application dependency graph."""

    name = "SUASApplication"

    def __init__(
        self,
        config: Optional[ApplicationConfig] = None,
        *,
        manager_config: Optional[ManagerConfig] = None,
    ) -> None:

        self.config = (
            config
            or ApplicationConfig()
        )

        self.manager_config = (
            manager_config
            or ManagerConfig()
        )

        self._lock = threading.RLock()

        self._stop_event = threading.Event()

        self._state = (
            ApplicationState.CREATED
        )

        self.shared: Optional[
            SharedContext
        ] = None

        self._signal_handlers_installed = False

    # =========================================================================
    # Build
    # =========================================================================

    def build(self) -> SharedContext:

        with self._lock:

            if self.shared is not None:
                return self.shared

        logger.info(
            "[Application] Building system"
        )

        # ---------------------------------------------------------------------
        # EventBus
        # ---------------------------------------------------------------------

        events = EventBus()

        # ---------------------------------------------------------------------
        # SharedData
        # ---------------------------------------------------------------------

        shared_data = SharedData()

        # Application and SharedData must use the same EventBus instance.
        shared_data.events = events

        # ---------------------------------------------------------------------
        # SharedContext
        # ---------------------------------------------------------------------

        shared = SharedContext(
            config=self.manager_config,
            events=events,
            shared_data=shared_data,
        )

        # ---------------------------------------------------------------------
        # MAVLink
        # ---------------------------------------------------------------------

        shared.mavlink = (
            self._create_mavlink()
        )

        # ---------------------------------------------------------------------
        # Servo Manager
        # ---------------------------------------------------------------------

        if (
            self.config.enable_servos
            and shared.mavlink is not None
        ):

            from payload.servo import (
                create_default_servo_manager,
            )

            shared.servo_manager = (
                create_default_servo_manager(
                    shared.mavlink
                )
            )

        # ---------------------------------------------------------------------
        # GPIO Manager
        # ---------------------------------------------------------------------

        if self.config.enable_gpio:

            from hardware.gpio import (
                GPIOManager,
            )

            shared.gpio_manager = (
                GPIOManager()
            )

        # ---------------------------------------------------------------------
        # Mission Manager
        # ---------------------------------------------------------------------

        from core.mission_manager import (
            MissionManager,
        )

        shared.mission_manager = (
            MissionManager(
                event_bus=shared.events,
                shared_data=shared.shared_data,
                cfg=shared.config,
            )
        )

        # ---------------------------------------------------------------------
        # Autonomous Mission
        # ---------------------------------------------------------------------

        from missions.autonomous_mission import AutonomousMission

        shared.autonomous_mission = AutonomousMission(
            shared_data=shared.shared_data,
            mavlink=shared.mavlink,
            servo_manager=shared.servo_manager,
        )

        # Shared context so a safety shutdown can reach all hardware owners.
        shared.shared_data.mavlink_manager = shared.mavlink
        shared.shared_data.servo_manager = shared.servo_manager
        shared.shared_data.autonomous_mission = shared.autonomous_mission

        # ---------------------------------------------------------------------
        # Health Monitor
        # ---------------------------------------------------------------------

        if self.config.enable_health_monitor:

            try:

                from core.health_monitor import (
                    HealthMonitor,
                )

                shared.health_monitor = (
                    HealthMonitor(
                        shared
                    )
                )

            except (
                ImportError,
                TypeError,
            ):

                logger.warning(
                    "[Application] "
                    "HealthMonitor could not be created. "
                    "It will remain disabled."
                )

                shared.health_monitor = None

        # ---------------------------------------------------------------------
        # Lifecycle
        # ---------------------------------------------------------------------

        from core.lifecycle import (
            LifecycleManager,
        )

        lifecycle = LifecycleManager()

        shared.lifecycle = lifecycle

        # ---------------------------------------------------------------------
        # Register components
        # ---------------------------------------------------------------------

        self._register_lifecycle_components(
            lifecycle,
            shared,
        )

        with self._lock:

            self.shared = shared

        logger.info(
            "[Application] "
            "System build completed"
        )

        return shared

    # =========================================================================
    # MAVLink creation
    # =========================================================================

    def _create_mavlink(self) -> Any:

        try:

            from mavlink.mavlink_manager import (
                MAVLinkManager,
            )

        except ImportError:

            logger.exception(
                "[Application] "
                "MAVLinkManager import failed"
            )

            return None

        # ---------------------------------------------------------------------
        # Use the current names from config.py.
        #
        # Raise a clear error if these settings are missing from config.py.
        # ---------------------------------------------------------------------

        required = (
            "MAVLINK_CONNECTION",
            "BAUDRATE",
            "RECONNECT_DELAY",
            "HEARTBEAT_TIMEOUT",
        )

        missing = [
            name
            for name in required
            if not hasattr(
                __import__("config"),
                name,
            )
        ]

        if missing:

            raise RuntimeError(
                "Missing MAVLink configuration values "
                "in config.py: "
                + ", ".join(missing)
            )

        cfg_module = __import__(
            "config"
        )

        try:

            return MAVLinkManager(
                connection_string=(
                    cfg_module.MAVLINK_CONNECTION
                ),
                baudrate=(
                    cfg_module.BAUDRATE
                ),
                reconnect_delay_s=(
                    cfg_module.RECONNECT_DELAY
                ),
                heartbeat_timeout_s=(
                    cfg_module.HEARTBEAT_TIMEOUT
                ),
                # Inject the EventBus for the ARM_MAX_ATTEMPTS_EXCEEDED event.
                event_bus=self._shared.events if hasattr(self, "_shared") else None,
                shared_data=self._shared.shared_data if hasattr(self, "_shared") else None,
            )

        except Exception:

            logger.exception(
                "[Application] "
                "MAVLinkManager creation failed"
            )

            raise

    # =========================================================================
    # Lifecycle registration
    # =========================================================================

    def _register_lifecycle_components(
        self,
        lifecycle: Any,
        shared: SharedContext,
    ) -> None:

        # ---------------------------------------------------------------------
        # MAVLink
        # ---------------------------------------------------------------------

        if shared.mavlink is not None:

            lifecycle.register(
                name="mavlink",
                component=shared.mavlink,
                required=True,
            )

        # ---------------------------------------------------------------------
        # GPIO
        # ---------------------------------------------------------------------

        if shared.gpio_manager is not None:

            lifecycle.register(
                name="gpio",
                component=shared.gpio_manager,
                required=False,
            )

        # ---------------------------------------------------------------------
        # Servo
        # ---------------------------------------------------------------------

        if shared.servo_manager is not None:

            dependencies = []

            if shared.mavlink is not None:
                dependencies.append(
                    "mavlink"
                )

            lifecycle.register(
                name="servo",
                component=shared.servo_manager,
                dependencies=dependencies,
                required=False,
            )

        # ---------------------------------------------------------------------
        # Mission Manager
        #
        # MissionManager kendi scheduler/safety/watchdog
        # Manages the subsystems.
        # ---------------------------------------------------------------------

        mission_dependencies = []

        if shared.mavlink is not None:
            mission_dependencies.append(
                "mavlink"
            )

        lifecycle.register(
            name="mission_manager",
            component=shared.mission_manager,
            dependencies=mission_dependencies,
            required=True,
        )

        # ---------------------------------------------------------------------
        # Autonomous Mission
        # ---------------------------------------------------------------------

        if shared.autonomous_mission is not None:

            dependencies = []

            if shared.mavlink is not None:
                dependencies.append("mavlink")

            if shared.mission_manager is not None:
                dependencies.append("mission_manager")

            if shared.servo_manager is not None:
                dependencies.append("servo")

            lifecycle.register(
                name="autonomous_mission",
                component=shared.autonomous_mission,
                dependencies=dependencies,
                required=True,
            )

        # ---------------------------------------------------------------------
        # Health Monitor
        # ---------------------------------------------------------------------

        if shared.health_monitor is not None:

            lifecycle.register(
                name="health",
                component=shared.health_monitor,
                dependencies=(
                    ["mavlink"]
                    if shared.mavlink is not None
                    else []
                ),
                required=False,
            )

    # =========================================================================
    # Start
    # =========================================================================

    def start(self) -> bool:
        """Start the component or operation."""

        with self._lock:

            if (
                self._state
                == ApplicationState.RUNNING
            ):
                return True

            self._state = (
                ApplicationState.STARTING
            )

        logger.info(
            "[Application] Starting"
        )

        try:

            shared = self.build()

            if self.config.install_signal_handlers:

                self._install_signal_handlers()

            # ---------------------------------------------------------------
            # Lifecycle
            # ---------------------------------------------------------------

            success = (
                shared.lifecycle.start_all()
            )

            if not success:

                raise RuntimeError(
                    "Lifecycle startup failed"
                )

            with self._lock:

                self._state = (
                    ApplicationState.RUNNING
                )

            logger.info(
                "[Application] "
                "Started successfully"
            )

            return True

        except Exception as exc:

            logger.exception(
                "[Application] "
                "Startup failed: %s",
                exc,
            )

            with self._lock:

                self._state = (
                    ApplicationState.FAILED
                )

            self.stop()

            return False

    # =========================================================================
    # Stop
    # =========================================================================

    def stop(self) -> bool:
        """Stop the component or operation safely."""

        with self._lock:

            if (
                self._state
                == ApplicationState.STOPPED
            ):
                return True

            self._state = (
                ApplicationState.STOPPING
            )

        logger.info(
            "[Application] Stopping"
        )

        success = True

        shared = self.shared

        if shared is None:

            with self._lock:

                self._state = (
                    ApplicationState.STOPPED
                )

            self._stop_event.set()

            return True

        # ---------------------------------------------------------------------
        # Lifecycle shutdown
        #
        # Lifecycle shutdown must follow reverse dependency order.
        # This allows MissionManager to stop its subsystems first.
        # ---------------------------------------------------------------------

        try:

            if shared.lifecycle is not None:

                result = (
                    shared.lifecycle.stop_all()
                )

                if result is False:

                    success = False

        except Exception:

            success = False

            logger.exception(
                "[Application] "
                "Lifecycle shutdown failed"
            )

        self._stop_event.set()

        with self._lock:

            self._state = (
                ApplicationState.STOPPED
            )

        logger.info(
            "[Application] Stopped"
        )

        return success

    # =========================================================================
    # Restart
    # =========================================================================

    def restart(self) -> bool:

        logger.warning(
            "[Application] Restart requested"
        )

        self.stop()

        self._stop_event.clear()

        with self._lock:

            self.shared = None

        return self.start()

    # =========================================================================
    # Wait
    # =========================================================================

    def wait(
        self,
        timeout_s: Optional[float] = None,
    ) -> bool:

        return self._stop_event.wait(
            timeout=timeout_s
        )

    # =========================================================================
    # Signal handlers
    # =========================================================================

    def _install_signal_handlers(
        self,
    ) -> None:

        if self._signal_handlers_installed:

            return

        try:

            signal.signal(
                signal.SIGINT,
                self._signal_handler,
            )

            signal.signal(
                signal.SIGTERM,
                self._signal_handler,
            )

            self._signal_handlers_installed = (
                True
            )

        except ValueError:

            logger.debug(
                "[Application] "
                "Signal handlers can only be "
                "installed from the main thread"
            )

    # =========================================================================
    # Signal handler
    # =========================================================================

    def _signal_handler(
        self,
        signum: int,
        frame: Any,
    ) -> None:

        logger.warning(
            "[Application] "
            "Received signal %s",
            signum,
        )

        threading.Thread(
            target=self.stop,
            name="ApplicationShutdown",
            daemon=True,
        ).start()

    # =========================================================================
    # State
    # =========================================================================

    @property
    def state(self) -> str:

        with self._lock:

            return self._state

    # =========================================================================
    # Running
    # =========================================================================

    @property
    def running(self) -> bool:

        with self._lock:

            return (
                self._state
                == ApplicationState.RUNNING
            )

    # =========================================================================
    # Failed
    # =========================================================================

    @property
    def failed(self) -> bool:

        with self._lock:

            return (
                self._state
                == ApplicationState.FAILED
            )

    # =========================================================================
    # Representation
    # =========================================================================

    def __repr__(self) -> str:

        with self._lock:

            return (
                "Application("
                f"state={self._state!r}"
                ")"
            )


# ============================================================================
# Application Factory
# ============================================================================

def create_application(
    config: Optional[
        ApplicationConfig
    ] = None,
    *,
    manager_config: Optional[
        ManagerConfig
    ] = None,
) -> Application:

    return Application(
        config=config,
        manager_config=manager_config,
    )


# ============================================================================
# Main
# ============================================================================

def main() -> int:

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s "
            "[%(levelname)s] "
            "%(name)s: "
            "%(message)s"
        ),
    )

    logger.info(
        "========================================"
    )

    logger.info(
        "SUAS Flight Computer"
    )

    logger.info(
        "Starting application"
    )

    logger.info(
        "========================================"
    )

    app = create_application()

    if not app.start():

        logger.critical(
            "Application failed to start"
        )

        return 1

    try:

        app.wait()

    except KeyboardInterrupt:

        logger.info(
            "Keyboard interrupt received"
        )

    finally:

        app.stop()

    return 0


# ============================================================================
# Python entry point
# ============================================================================

if __name__ == "__main__":

    raise SystemExit(
        main()
    )
