"""Coordinates dependency-aware startup and reverse-order shutdown of components."""

from __future__ import annotations

import inspect
import logging
import threading
import time

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional


logger = logging.getLogger(__name__)


# ============================================================================
# Lifecycle State
# ============================================================================

class LifecycleState(str, Enum):
    """Lifecycle manager state."""

    CREATED = "created"

    STARTING = "starting"

    RUNNING = "running"

    STOPPING = "stopping"

    STOPPED = "stopped"

    FAILED = "failed"


# ============================================================================
# Lifecycle Component
# ============================================================================

@dataclass
class LifecycleComponent:

    name: str

    component: Any

    dependencies: List[str] = field(
        default_factory=list
    )

    required: bool = True

    started: bool = False


# ============================================================================
# Configuration
# ============================================================================

@dataclass(frozen=True)
class LifecycleConfig:
    """
    Lifecycle configuration.
    """

    startup_timeout_s: float = 10.0

    shutdown_timeout_s: float = 10.0

    continue_on_optional_failure: bool = True

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
# Lifecycle Manager
# ============================================================================

class LifecycleManager:

    name = "LifecycleManager"

    def __init__(
        self,
        config: Optional[
            LifecycleConfig
        ] = None,
    ) -> None:

        self.config = (
            config
            or LifecycleConfig()
        )

        self._lock = threading.RLock()

        self._components: Dict[
            str,
            LifecycleComponent,
        ] = {}

        self._start_order: List[
            str
        ] = []

        self._state = (
            LifecycleState.CREATED
        )

        self._running = False

    # =========================================================================
    # Register
    # =========================================================================

    def register(
        self,
        name: str,
        component: Any,
        dependencies: Optional[
            Iterable[str]
        ] = None,
        required: bool = True,
    ) -> None:
        """
        Register lifecycle component.
        """

        if not name:

            raise ValueError(
                "Component name cannot be empty"
            )

        if component is None:

            raise ValueError(
                "component cannot be None"
            )

        with self._lock:

            if name in self._components:

                raise ValueError(
                    f"Component already registered: "
                    f"{name}"
                )

            dependency_list = list(
                dependencies
                or []
            )

            if name in dependency_list:

                raise ValueError(
                    f"Component cannot depend on itself: "
                    f"{name}"
                )

            self._components[
                name
            ] = LifecycleComponent(
                name=name,

                component=component,

                dependencies=dependency_list,

                required=required,
            )

        logger.info(
            "[LifecycleManager] "
            "Registered component: %s",
            name,
        )

    # =========================================================================
    # Unregister
    # =========================================================================

    def unregister(
        self,
        name: str,
    ) -> bool:
        """
        Remove component from lifecycle manager.

        Running component unregister edilemez.
        """

        with self._lock:

            item = self._components.get(
                name
            )

            if item is None:

                return False

            if item.started:

                return False

            # Check whether another component depends on it.

            for other in (
                self._components.values()
            ):

                if name in other.dependencies:

                    return False

            self._components.pop(
                name,
                None,
            )

        return True

    # =========================================================================
    # Dependency validation
    # =========================================================================

    def validate_dependencies(
        self,
    ) -> None:

        with self._lock:

            for item in (
                self._components.values()
            ):

                for dependency in (
                    item.dependencies
                ):

                    if dependency not in (
                        self._components
                    ):

                        raise ValueError(
                            f"Unknown dependency "
                            f"{dependency!r} for "
                            f"{item.name!r}"
                        )

            self._topological_order()

    # =========================================================================
    # Topological sort
    # =========================================================================

    def _topological_order(
        self,
    ) -> List[str]:
        """
        Calculate dependency-safe startup order.
        """

        visited: set[str] = set()

        visiting: set[str] = set()

        result: List[str] = []

        def visit(
            name: str,
        ) -> None:

            if name in visited:

                return

            if name in visiting:

                raise ValueError(
                    "Lifecycle dependency cycle detected "
                    f"at {name!r}"
                )

            visiting.add(
                name
            )

            component = (
                self._components[name]
            )

            for dependency in (
                component.dependencies
            ):

                visit(
                    dependency
                )

            visiting.remove(
                name
            )

            visited.add(
                name
            )

            result.append(
                name
            )

        for name in (
            self._components
        ):

            visit(
                name
            )

        return result

    # =========================================================================
    # Start all
    # =========================================================================

    def start_all(self) -> bool:

        with self._lock:

            if self._state == (
                LifecycleState.RUNNING
            ):

                return True

            self._state = (
                LifecycleState.STARTING
            )

            self._start_order.clear()

            try:

                order = (
                    self._topological_order()
                )

            except Exception:

                self._state = (
                    LifecycleState.FAILED
                )

                raise

        started: List[
            str
        ] = []

        start_time = (
            time.monotonic()
        )

        try:

            for name in order:

                elapsed = (
                    time.monotonic()
                    - start_time
                )

                if (
                    elapsed
                    > self.config.startup_timeout_s
                ):

                    raise TimeoutError(
                        "Lifecycle startup timeout"
                    )

                item = self._components[
                    name
                ]

                success = (
                    self._start_component(
                        item
                    )
                )

                if success:

                    started.append(
                        name
                    )

                    with self._lock:

                        item.started = True

                        self._start_order.append(
                            name
                        )

                    continue

                # -------------------------------------------------------------
                # Component failed.
                # -------------------------------------------------------------

                if item.required:

                    raise RuntimeError(
                        f"Required component failed "
                        f"to start: {name}"
                    )

                if (
                    not self.config
                    .continue_on_optional_failure
                ):

                    raise RuntimeError(
                        f"Optional component failed "
                        f"to start: {name}"
                    )

                logger.warning(
                    "[LifecycleManager] "
                    "Optional component failed: %s",
                    name,
                )

            with self._lock:

                self._running = True

                self._state = (
                    LifecycleState.RUNNING
                )

            logger.info(
                "[LifecycleManager] "
                "All components started"
            )

            return True

        except Exception as exc:

            logger.exception(
                "[LifecycleManager] "
                "Startup failed"
            )

            self._rollback(
                started
            )

            with self._lock:

                self._running = False

                self._state = (
                    LifecycleState.FAILED
                )

            raise exc

    # =========================================================================
    # Start component
    # =========================================================================

    def _start_component(
        self,
        item: LifecycleComponent,
    ) -> bool:
        """
        Start one component.
        """

        component = item.component

        start_method = getattr(
            component,
            "start",
            None,
        )

        if not callable(
            start_method
        ):

            raise TypeError(
                f"Component {item.name!r} "
                f"does not implement start()"
            )

        logger.info(
            "[LifecycleManager] "
            "Starting %s",
            item.name,
        )

        result = start_method()

        if result is False:

            return False

        return True

    # =========================================================================
    # Rollback
    # =========================================================================

    def _rollback(
        self,
        started: List[str],
    ) -> None:
        """
        Stop already started components in reverse order.
        """

        logger.warning(
            "[LifecycleManager] "
            "Rolling back startup"
        )

        for name in reversed(
            started
        ):

            item = self._components.get(
                name
            )

            if item is None:

                continue

            try:

                self._stop_component(
                    item
                )

            except Exception:

                logger.exception(
                    "[LifecycleManager] "
                    "Rollback failed for %s",
                    name,
                )

            finally:

                with self._lock:

                    item.started = False

    # =========================================================================
    # Stop all
    # =========================================================================

    def stop_all(self) -> bool:
        """
        Stop all components in reverse dependency order.
        """

        with self._lock:

            if self._state == (
                LifecycleState.STOPPED
            ):

                return True

            self._state = (
                LifecycleState.STOPPING
            )

            order = list(
                self._start_order
            )

        success = True

        stop_start = (
            time.monotonic()
        )

        for name in reversed(
            order
        ):

            elapsed = (
                time.monotonic()
                - stop_start
            )

            if (
                elapsed
                > self.config.shutdown_timeout_s
            ):

                logger.error(
                    "[LifecycleManager] "
                    "Shutdown timeout"
                )

                success = False

                break

            item = self._components.get(
                name
            )

            if item is None:

                continue

            try:

                result = (
                    self._stop_component(
                        item
                    )
                )

                if not result:

                    success = False

            except Exception:

                success = False

                logger.exception(
                    "[LifecycleManager] "
                    "Failed to stop %s",
                    name,
                )

            finally:

                with self._lock:

                    item.started = False

        with self._lock:

            self._running = False

            self._state = (
                LifecycleState.STOPPED
            )

            self._start_order.clear()

        logger.info(
            "[LifecycleManager] "
            "All components stopped"
        )

        return success

    # =========================================================================
    # Stop component
    # =========================================================================

    def _stop_component(
        self,
        item: LifecycleComponent,
    ) -> bool:
        """
        Stop one component.
        """

        component = item.component

        stop_method = getattr(
            component,
            "stop",
            None,
        )

        if not callable(
            stop_method
        ):

            logger.warning(
                "[LifecycleManager] "
                "%s has no stop() method",
                item.name,
            )

            return True

        logger.info(
            "[LifecycleManager] "
            "Stopping %s",
            item.name,
        )

        # Components may expose different stop() method signatures.
        #
        # Some components accept:
        #     stop()
        #
        # others accept:
        #     stop(timeout_s=...)
        #
        # Do not force timeout_s onto every component.
        # Do not pass this argument blindly. First check whether the method
        # supports it.

        try:
            signature = inspect.signature(
                stop_method
            )

            parameters = signature.parameters

            accepts_timeout = (
                "timeout_s" in parameters
                or any(
                    parameter.kind
                    == inspect.Parameter.VAR_KEYWORD
                    for parameter in parameters.values()
                )
            )

            if accepts_timeout:

                result = stop_method(
                    timeout_s=(
                        self.config
                        .shutdown_timeout_s
                    )
                )

            else:

                result = stop_method()

        except (TypeError, ValueError):
            # A signature may not be available for some callable types.
            # In this case, retry stop() without arguments.
            result = stop_method()

        if result is False:

            return False

        return True

    # =========================================================================
    # Restart
    # =========================================================================

    def restart(self) -> bool:
        """
        Restart complete lifecycle.
        """

        self.stop_all()

        return self.start_all()

    # =========================================================================
    # Component state
    # =========================================================================

    def is_started(
        self,
        name: str,
    ) -> bool:
        """Check whether component has started."""

        with self._lock:

            item = self._components.get(
                name
            )

            if item is None:

                return False

            return item.started

    # =========================================================================
    # Get component
    # =========================================================================

    def get(
        self,
        name: str,
    ) -> Optional[Any]:
        """Return registered component."""

        with self._lock:

            item = self._components.get(
                name
            )

            if item is None:

                return None

            return item.component

    # =========================================================================
    # Component information
    # =========================================================================

    def components(
        self,
    ) -> Dict[
        str,
        LifecycleComponent,
    ]:
        """
        Return lifecycle component snapshots.
        """

        with self._lock:

            return {
                name: LifecycleComponent(
                    name=item.name,

                    component=item.component,

                    dependencies=list(
                        item.dependencies
                    ),

                    required=item.required,

                    started=item.started,
                )

                for name, item
                in self._components.items()
            }

    # =========================================================================
    # Start order
    # =========================================================================

    def start_order(
        self,
    ) -> List[str]:
        """
        Return calculated startup order.
        """

        with self._lock:

            return self._topological_order()

    # =========================================================================
    # State
    # =========================================================================

    @property
    def state(
        self,
    ) -> LifecycleState:

        with self._lock:

            return self._state

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
                "LifecycleManager("
                f"state="
                f"{self._state.value!r}, "
                f"running="
                f"{self._running}, "
                f"components="
                f"{len(self._components)}"
                ")"
            )
