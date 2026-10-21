"""Monitors the runtime health and liveness of mission subsystems."""

from __future__ import annotations

import logging
import threading
import time

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional


logger = logging.getLogger(__name__)


# ============================================================================
# Health state
# ============================================================================

class HealthState(str, Enum):
    """Overall subsystem health state."""

    UNKNOWN = "unknown"

    HEALTHY = "healthy"

    DEGRADED = "degraded"

    FAILED = "failed"

    STOPPED = "stopped"


# ============================================================================
# Component health
# ============================================================================

@dataclass
class ComponentHealth:
    """
    Runtime health information for one component.
    """

    name: str

    state: HealthState = (
        HealthState.UNKNOWN
    )

    last_heartbeat: Optional[float] = None

    last_check: Optional[float] = None

    failure_count: int = 0

    message: str = ""

    enabled: bool = True


# ============================================================================
# Health snapshot
# ============================================================================

@dataclass(frozen=True)
class HealthSnapshot:
    """
    Immutable health snapshot.
    """

    overall_state: HealthState

    healthy_count: int

    degraded_count: int

    failed_count: int

    stopped_count: int

    unknown_count: int

    timestamp: float


# ============================================================================
# Configuration
# ============================================================================

@dataclass(frozen=True)
class HealthMonitorConfig:
    """
    Health monitor configuration.
    """

    poll_interval_s: float = 1.0

    heartbeat_timeout_s: float = 5.0

    degraded_timeout_s: float = 2.0

    failure_threshold: int = 3

    event_repeat_interval_s: float = 5.0

    def __post_init__(self) -> None:

        if self.poll_interval_s <= 0:

            raise ValueError(
                "poll_interval_s must be greater than zero"
            )

        if self.heartbeat_timeout_s <= 0:

            raise ValueError(
                "heartbeat_timeout_s must be greater than zero"
            )

        if self.degraded_timeout_s <= 0:

            raise ValueError(
                "degraded_timeout_s must be greater than zero"
            )

        if self.degraded_timeout_s >= (
            self.heartbeat_timeout_s
        ):

            raise ValueError(
                "degraded_timeout_s must be lower "
                "than heartbeat_timeout_s"
            )

        if self.failure_threshold <= 0:

            raise ValueError(
                "failure_threshold must be positive"
            )

        if self.event_repeat_interval_s < 0:

            raise ValueError(
                "event_repeat_interval_s cannot be negative"
            )


# ============================================================================
# Health statistics
# ============================================================================

@dataclass
class HealthStatistics:
    """Runtime health monitor statistics."""

    checks: int = 0

    failures: int = 0

    degraded_events: int = 0

    recovery_events: int = 0

    last_failure_time: Optional[float] = None

    last_recovery_time: Optional[float] = None


# ============================================================================
# Health Monitor
# ============================================================================

class HealthMonitor:

    name = "HealthMonitor"

    def __init__(
        self,
        event_bus: Any = None,
        config: Optional[
            HealthMonitorConfig
        ] = None,
    ) -> None:

        self._event_bus = event_bus

        self.config = (
            config
            or HealthMonitorConfig()
        )

        self._lock = threading.RLock()

        self._components: Dict[
            str,
            Any,
        ] = {}

        self._health: Dict[
            str,
            ComponentHealth,
        ] = {}

        self._last_publish: Dict[
            str,
            float,
        ] = {}

        self._running = False

        self._thread: Optional[
            threading.Thread
        ] = None

        self._stop_event = (
            threading.Event()
        )

        self._statistics = (
            HealthStatistics()
        )

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def start(self) -> None:
        """Start health monitoring."""

        with self._lock:

            if self._running:

                return

            self._running = True

            self._stop_event.clear()

            self._thread = threading.Thread(
                target=self._run,
                name="HealthMonitor",
                daemon=True,
            )

            self._thread.start()

        logger.info(
            "[HealthMonitor] Started"
        )

    # -------------------------------------------------------------------------

    def stop(self) -> None:
        """Stop health monitoring."""

        with self._lock:

            if not self._running:

                return

            self._running = False

            self._stop_event.set()

            thread = self._thread

        if thread is not None:

            thread.join(
                timeout=3.0
            )

        with self._lock:

            for health in (
                self._health.values()
            ):

                health.state = (
                    HealthState.STOPPED
                )

        logger.info(
            "[HealthMonitor] Stopped"
        )

    # =========================================================================
    # Register
    # =========================================================================

    def register(
        self,
        name: str,
        component: Any,
    ) -> None:
        """
        Register a subsystem.

        Component None olamaz.
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

            self._components[
                name
            ] = component

            self._health[
                name
            ] = ComponentHealth(
                name=name,
                last_heartbeat=time.monotonic(),
            )

        logger.info(
            "[HealthMonitor] "
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
        """Unregister a component."""

        with self._lock:

            if name not in self._components:

                return False

            self._components.pop(
                name,
                None,
            )

            self._health.pop(
                name,
                None,
            )

            self._last_publish.pop(
                name,
                None,
            )

        return True

    # =========================================================================
    # Heartbeat
    # =========================================================================

    def heartbeat(
        self,
        name: str,
    ) -> None:
        """
        Update component heartbeat.
        """

        now = time.monotonic()

        with self._lock:

            health = self._health.get(
                name
            )

            if health is None:

                health = ComponentHealth(
                    name=name
                )

                self._health[
                    name
                ] = health

            health.last_heartbeat = now

            if health.state in (
                HealthState.DEGRADED,
                HealthState.FAILED,
            ):

                previous = health.state

                health.state = (
                    HealthState.HEALTHY
                )

                health.message = ""

                self._statistics.recovery_events += 1

                self._statistics.last_recovery_time = now

                logger.info(
                    "[HealthMonitor] "
                    "Component recovered: %s "
                    "(previous=%s)",
                    name,
                    previous.value,
                )

    # =========================================================================
    # Main loop
    # =========================================================================

    def _run(self) -> None:
        """Health monitor loop."""

        while not self._stop_event.wait(
            self.config.poll_interval_s
        ):

            try:

                self.check()

            except Exception:

                logger.exception(
                    "[HealthMonitor] "
                    "Health check failed"
                )

    # =========================================================================
    # Check
    # =========================================================================

    def check(self) -> HealthSnapshot:
        """
        Execute one health check cycle.
        """

        now = time.monotonic()

        with self._lock:

            self._statistics.checks += 1

            names = list(
                self._components.keys()
            )

        for name in names:

            self._check_component(
                name,
                now,
            )

        snapshot = (
            self.snapshot()
        )

        return snapshot

    # =========================================================================
    # Component check
    # =========================================================================

    def _check_component(
        self,
        name: str,
        now: float,
    ) -> None:
        """Check one component."""

        with self._lock:

            component = (
                self._components.get(
                    name
                )
            )

            health = (
                self._health.get(
                    name
                )
            )

        if component is None or health is None:

            return

        # ---------------------------------------------------------------
        # Explicit health API
        # ---------------------------------------------------------------

        explicit_state = (
            self._read_explicit_health(
                component
            )
        )

        if explicit_state is not None:

            self._apply_state(
                name=name,
                state=explicit_state,
                now=now,
            )

            return

        # ---------------------------------------------------------------
        # Running state
        # ---------------------------------------------------------------

        running = (
            self._read_running_state(
                component
            )
        )

        if running is False:

            self._apply_state(
                name=name,
                state=HealthState.STOPPED,
                now=now,
                message="Component is not running",
            )

            return

        # ---------------------------------------------------------------
        # Heartbeat
        # ---------------------------------------------------------------

        with self._lock:

            last_heartbeat = (
                health.last_heartbeat
            )

        if last_heartbeat is None:

            self._apply_state(
                name=name,
                state=HealthState.UNKNOWN,
                now=now,
                message="No heartbeat received",
            )

            return

        age = (
            now
            - last_heartbeat
        )

        if (
            age
            > self.config.heartbeat_timeout_s
        ):

            self._apply_state(
                name=name,
                state=HealthState.FAILED,
                now=now,
                message=(
                    f"Heartbeat timeout: "
                    f"{age:.2f}s"
                ),
            )

            return

        if (
            age
            > self.config.degraded_timeout_s
        ):

            self._apply_state(
                name=name,
                state=HealthState.DEGRADED,
                now=now,
                message=(
                    f"Heartbeat delayed: "
                    f"{age:.2f}s"
                ),
            )

            return

        self._apply_state(
            name=name,
            state=HealthState.HEALTHY,
            now=now,
        )

    # =========================================================================
    # Explicit health
    # =========================================================================

    def _read_explicit_health(
        self,
        component: Any,
    ) -> Optional[HealthState]:
        """
        Read health from component if it exposes a health API.
        """

        try:

            method = getattr(
                component,
                "health",
                None,
            )

            if callable(method):

                result = method()

                if result is None:

                    return None

                if isinstance(
                    result,
                    HealthState,
                ):

                    return result

                if isinstance(
                    result,
                    str,
                ):

                    try:

                        return HealthState(
                            result.lower()
                        )

                    except ValueError:

                        return None

        except Exception:

            logger.exception(
                "[HealthMonitor] "
                "Failed to read explicit health"
            )

        return None

    # =========================================================================
    # Running state
    # =========================================================================

    def _read_running_state(
        self,
        component: Any,
    ) -> Optional[bool]:
        """
        Read component running state.
        """

        try:

            method = getattr(
                component,
                "is_running",
                None,
            )

            if callable(method):

                return bool(
                    method()
                )

            value = getattr(
                component,
                "running",
                None,
            )

            if value is not None:

                return bool(value)

        except Exception:

            logger.exception(
                "[HealthMonitor] "
                "Failed to read running state"
            )

        return None

    # =========================================================================
    # Apply state
    # =========================================================================

    def _apply_state(
        self,
        name: str,
        state: HealthState,
        now: float,
        message: str = "",
    ) -> None:
        """Apply new component health state."""

        with self._lock:

            health = self._health.get(
                name
            )

            if health is None:

                return

            previous = health.state

            if state == HealthState.FAILED:

                health.failure_count += 1

                self._statistics.failures += 1

                self._statistics.last_failure_time = now

            elif state == HealthState.DEGRADED:

                self._statistics.degraded_events += 1

            if (
                previous != state
            ):

                health.state = state

                health.message = message

                self._publish_state_change(
                    health,
                    previous,
                    now,
                )

            else:

                health.state = state

                health.message = message

            health.last_check = now

    # =========================================================================
    # State change event
    # =========================================================================

    def _publish_state_change(
        self,
        health: ComponentHealth,
        previous: HealthState,
        now: float,
    ) -> None:
        """Publish health state change."""

        if self._event_bus is None:

            return

        key = health.name

        last_publish = (
            self._last_publish.get(
                key
            )
        )

        if (
            last_publish is not None
            and (
                now
                - last_publish
                < self.config
                .event_repeat_interval_s
            )
        ):

            return

        self._last_publish[
            key
        ] = now

        payload = {
            "component": health.name,

            "previous_state": (
                previous.value
            ),

            "state": (
                health.state.value
            ),

            "message": health.message,

            "timestamp": now,

            "source": self.name,
        }

        try:

            from core.events import EventType

            event_type = getattr(
                EventType,
                "SYSTEM_HEALTH_CHANGED",
                None,
            )

            if event_type is None:

                logger.warning(
                    "[HealthMonitor] "
                    "EventType.SYSTEM_HEALTH_CHANGED "
                    "does not exist"
                )

                return

            self._event_bus.publish(
                event_type,
                payload,
            )

        except Exception:

            logger.exception(
                "[HealthMonitor] "
                "Failed to publish health event"
            )

    # =========================================================================
    # Component status
    # =========================================================================

    def get(
        self,
        name: str,
    ) -> Optional[ComponentHealth]:
        """Return component health snapshot."""

        with self._lock:

            health = self._health.get(
                name
            )

            if health is None:

                return None

            return ComponentHealth(
                name=health.name,

                state=health.state,

                last_heartbeat=(
                    health.last_heartbeat
                ),

                last_check=(
                    health.last_check
                ),

                failure_count=(
                    health.failure_count
                ),

                message=health.message,

                enabled=health.enabled,
            )

    # =========================================================================
    # All component status
    # =========================================================================

    def all(
        self,
    ) -> Dict[
        str,
        ComponentHealth,
    ]:
        """Return health snapshots for all components."""

        with self._lock:

            return {
                name: ComponentHealth(
                    name=health.name,

                    state=health.state,

                    last_heartbeat=(
                        health.last_heartbeat
                    ),

                    last_check=(
                        health.last_check
                    ),

                    failure_count=(
                        health.failure_count
                    ),

                    message=health.message,

                    enabled=health.enabled,
                )

                for name, health
                in self._health.items()
            }

    # =========================================================================
    # Snapshot
    # =========================================================================

    def snapshot(
        self,
    ) -> HealthSnapshot:
        """Return overall health snapshot."""

        with self._lock:

            states = [
                health.state
                for health
                in self._health.values()
            ]

            healthy_count = states.count(
                HealthState.HEALTHY
            )

            degraded_count = states.count(
                HealthState.DEGRADED
            )

            failed_count = states.count(
                HealthState.FAILED
            )

            stopped_count = states.count(
                HealthState.STOPPED
            )

            unknown_count = states.count(
                HealthState.UNKNOWN
            )

            # ---------------------------------------------------------------
            # Overall state
            # ---------------------------------------------------------------

            if failed_count > 0:

                overall = (
                    HealthState.FAILED
                )

            elif degraded_count > 0:

                overall = (
                    HealthState.DEGRADED
                )

            elif unknown_count > 0:

                overall = (
                    HealthState.UNKNOWN
                )

            elif (
                stopped_count > 0
                and healthy_count == 0
            ):

                overall = (
                    HealthState.STOPPED
                )

            else:

                overall = (
                    HealthState.HEALTHY
                )

            return HealthSnapshot(
                overall_state=overall,

                healthy_count=healthy_count,

                degraded_count=degraded_count,

                failed_count=failed_count,

                stopped_count=stopped_count,

                unknown_count=unknown_count,

                timestamp=time.monotonic(),
            )

    # =========================================================================
    # Statistics
    # =========================================================================

    def statistics(
        self,
    ) -> HealthStatistics:
        """Return health statistics."""

        with self._lock:

            return HealthStatistics(
                checks=(
                    self._statistics.checks
                ),

                failures=(
                    self._statistics.failures
                ),

                degraded_events=(
                    self._statistics
                    .degraded_events
                ),

                recovery_events=(
                    self._statistics
                    .recovery_events
                ),

                last_failure_time=(
                    self._statistics
                    .last_failure_time
                ),

                last_recovery_time=(
                    self._statistics
                    .last_recovery_time
                ),
            )

    # =========================================================================
    # Properties
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

        snapshot = self.snapshot()

        return (
            "HealthMonitor("
            f"state="
            f"{snapshot.overall_state.value!r}, "
            f"components="
            f"{snapshot.healthy_count + snapshot.degraded_count + snapshot.failed_count + snapshot.stopped_count + snapshot.unknown_count}"
            ")"
        )
