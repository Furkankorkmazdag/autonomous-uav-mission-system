"""Monitors safety conditions and publishes events when a configured threshold is crossed."""

from __future__ import annotations

import logging
import threading
import time

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional


logger = logging.getLogger(__name__)


# ============================================================================
# Safety State
# ============================================================================

class SafetyState(str, Enum):
    """Overall safety state."""

    UNKNOWN = "unknown"

    SAFE = "safe"

    WARNING = "warning"

    CRITICAL = "critical"

    EMERGENCY = "emergency"


# ============================================================================
# Safety Event
# ============================================================================

class SafetyEvent(str, Enum):
    """Safety conditions detected by SafetyMonitor."""

    HEARTBEAT_TIMEOUT = "heartbeat_timeout"

    LOW_BATTERY = "low_battery"

    BATTERY_CRITICAL = "battery_critical"

    GPS_LOST = "gps_lost"

    GPS_DEGRADED = "gps_degraded"

    TELEMETRY_LOST = "telemetry_lost"

    SYSTEM_ERROR = "system_error"

    SAFETY_RECOVERED = "safety_recovered"


# ============================================================================
# Configuration
# ============================================================================

@dataclass(frozen=True)
class SafetyMonitorConfig:

    poll_interval_s: float = 0.5

    heartbeat_timeout_s: float = 3.0

    battery_rtl_pct: float = 20.0

    battery_critical_pct: float = 10.0

    gps_min_satellites: int = 6

    gps_required: bool = True

    telemetry_timeout_s: float = 3.0

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

        if not (
            0.0
            <= self.battery_critical_pct
            <= 100.0
        ):
            raise ValueError(
                "battery_critical_pct must be between 0 and 100"
            )

        if not (
            0.0
            <= self.battery_rtl_pct
            <= 100.0
        ):
            raise ValueError(
                "battery_rtl_pct must be between 0 and 100"
            )

        if (
            self.battery_critical_pct
            >= self.battery_rtl_pct
        ):
            raise ValueError(
                "battery_critical_pct must be lower "
                "than battery_rtl_pct"
            )

        if self.gps_min_satellites < 0:
            raise ValueError(
                "gps_min_satellites cannot be negative"
            )

        if self.telemetry_timeout_s <= 0:
            raise ValueError(
                "telemetry_timeout_s must be greater than zero"
            )

        if self.event_repeat_interval_s < 0:
            raise ValueError(
                "event_repeat_interval_s cannot be negative"
            )


# ============================================================================
# Safety Snapshot
# ============================================================================

@dataclass(frozen=True)
class SafetySnapshot:
    """Immutable safety status snapshot."""

    state: SafetyState

    heartbeat_ok: bool

    battery_ok: bool

    battery_percent: Optional[float]

    gps_ok: bool

    satellite_count: Optional[int]

    telemetry_ok: bool

    active_events: tuple[str, ...]

    timestamp: float


# ============================================================================
# Statistics
# ============================================================================

@dataclass
class SafetyStatistics:
    """Runtime safety statistics."""

    checks: int = 0

    heartbeat_failures: int = 0

    low_battery_events: int = 0

    critical_battery_events: int = 0

    gps_failures: int = 0

    telemetry_failures: int = 0

    recovered_events: int = 0

    last_event_time: Optional[float] = None


# ============================================================================
# Safety Monitor
# ============================================================================

class SafetyMonitor:

    name = "SafetyMonitor"

    def __init__(
        self,
        shared_data: Any,
        event_bus: Any,
        config: Optional[
            SafetyMonitorConfig
        ] = None,
    ) -> None:

        if shared_data is None:
            raise ValueError(
                "shared_data cannot be None"
            )

        if event_bus is None:
            raise ValueError(
                "event_bus cannot be None"
            )

        self._shared = shared_data

        self._event_bus = event_bus

        self.config = (
            config
            or SafetyMonitorConfig()
        )

        self._lock = threading.RLock()

        self._running = False

        self._thread: Optional[
            threading.Thread
        ] = None

        self._stop_event = (
            threading.Event()
        )

        self._state = SafetyState.UNKNOWN

        self._active_events: Dict[
            SafetyEvent,
            float,
        ] = {}

        self._last_event_publish: Dict[
            SafetyEvent,
            float,
        ] = {}

        self._heartbeat_ok = False

        self._battery_ok = True

        self._battery_percent: Optional[
            float
        ] = None

        self._gps_ok = True

        self._satellite_count: Optional[
            int
        ] = None

        self._telemetry_ok = True

        self._statistics = (
            SafetyStatistics()
        )

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def start(self) -> None:
        """Start safety monitoring thread."""

        with self._lock:

            if self._running:
                return

            self._running = True

            self._stop_event.clear()

            self._thread = threading.Thread(
                target=self._run,
                name="SafetyMonitor",
                daemon=True,
            )

            self._thread.start()

        logger.info(
            "[SafetyMonitor] Started"
        )

    # -------------------------------------------------------------------------

    def stop(self) -> None:
        """Stop safety monitoring."""

        with self._lock:

            if not self._running:
                return

            self._running = False

            self._stop_event.set()

            thread = self._thread

        if thread is not None:

            thread.join(
                timeout=5.0
            )

        logger.info(
            "[SafetyMonitor] Stopped"
        )

    # =========================================================================
    # Main loop
    # =========================================================================

    def _run(self) -> None:
        """Safety monitoring loop."""

        while not self._stop_event.wait(
            self.config.poll_interval_s
        ):

            try:

                self.check()

            except Exception as exc:

                logger.exception(
                    "[SafetyMonitor] "
                    "Safety check failed"
                )

                self._activate_event(
                    SafetyEvent.SYSTEM_ERROR,
                    {
                        "reason": str(exc),
                    },
                )

    # =========================================================================
    # Main safety check
    # =========================================================================

    def check(self) -> SafetySnapshot:

        now = time.monotonic()

        with self._lock:

            self._statistics.checks += 1

        self._check_heartbeat(
            now
        )

        self._check_battery(
            now
        )

        self._check_gps(
            now
        )

        self._check_telemetry(
            now
        )

        self._update_overall_state()

        self._check_recovery()

        return self.snapshot()

    # =========================================================================
    # Heartbeat
    # =========================================================================

    def _check_heartbeat(
        self,
        now: float,
    ) -> None:
        """Check MAVLink heartbeat."""

        age = self._get_heartbeat_age()

        if age is None:

            with self._lock:

                self._heartbeat_ok = False

            self._activate_event(
                SafetyEvent.HEARTBEAT_TIMEOUT,
                {
                    "heartbeat_age_s": None,
                },
            )

            return

        heartbeat_ok = (
            age
            <= self.config.heartbeat_timeout_s
        )

        with self._lock:

            self._heartbeat_ok = (
                heartbeat_ok
            )

        if not heartbeat_ok:

            with self._lock:

                self._statistics.heartbeat_failures += 1

            self._activate_event(
                SafetyEvent.HEARTBEAT_TIMEOUT,
                {
                    "heartbeat_age_s": age,
                    "timeout_s": (
                        self.config
                        .heartbeat_timeout_s
                    ),
                },
            )

        else:

            self._clear_event(
                SafetyEvent.HEARTBEAT_TIMEOUT
            )

    # =========================================================================
    # Battery
    # =========================================================================

    def _check_battery(
        self,
        now: float,
    ) -> None:
        """Check battery percentage."""

        battery = (
            self._get_battery_percent()
        )

        with self._lock:

            self._battery_percent = battery

        if battery is None:

            with self._lock:

                self._battery_ok = True

            return

        if (
            battery
            <= self.config.battery_critical_pct
        ):

            with self._lock:

                self._battery_ok = False

                self._statistics.critical_battery_events += 1

            self._activate_event(
                SafetyEvent.BATTERY_CRITICAL,
                {
                    "battery_percent": battery,

                    "threshold": (
                        self.config
                        .battery_critical_pct
                    ),
                },
            )

            return

        if (
            battery
            <= self.config.battery_rtl_pct
        ):

            with self._lock:

                self._battery_ok = False

                self._statistics.low_battery_events += 1

            self._activate_event(
                SafetyEvent.LOW_BATTERY,
                {
                    "battery_percent": battery,

                    "threshold": (
                        self.config
                        .battery_rtl_pct
                    ),
                },
            )

            self._clear_event(
                SafetyEvent.BATTERY_CRITICAL
            )

            return

        with self._lock:

            self._battery_ok = True

        self._clear_event(
            SafetyEvent.LOW_BATTERY
        )

        self._clear_event(
            SafetyEvent.BATTERY_CRITICAL
        )

    # =========================================================================
    # GPS
    # =========================================================================

    def _check_gps(
        self,
        now: float,
    ) -> None:
        """Check GPS state."""

        if not self.config.gps_required:

            with self._lock:

                self._gps_ok = True

            return

        satellites = (
            self._get_satellite_count()
        )

        with self._lock:

            self._satellite_count = satellites

        if satellites is None:

            with self._lock:

                self._gps_ok = False

            self._activate_event(
                SafetyEvent.GPS_LOST,
                {
                    "satellite_count": None,
                },
            )

            return

        if (
            satellites
            < self.config.gps_min_satellites
        ):

            with self._lock:

                self._gps_ok = False

                self._statistics.gps_failures += 1

            self._activate_event(
                SafetyEvent.GPS_DEGRADED,
                {
                    "satellite_count": satellites,

                    "minimum": (
                        self.config
                        .gps_min_satellites
                    ),
                },
            )

            return

        with self._lock:

            self._gps_ok = True

        self._clear_event(
            SafetyEvent.GPS_LOST
        )

        self._clear_event(
            SafetyEvent.GPS_DEGRADED
        )

    # =========================================================================
    # Telemetry
    # =========================================================================

    def _check_telemetry(
        self,
        now: float,
    ) -> None:
        """Check telemetry freshness."""

        age = (
            self._get_telemetry_age()
        )

        if age is None:

            with self._lock:

                self._telemetry_ok = False

            self._activate_event(
                SafetyEvent.TELEMETRY_LOST,
                {
                    "telemetry_age_s": None,
                },
            )

            return

        telemetry_ok = (
            age
            <= self.config.telemetry_timeout_s
        )

        with self._lock:

            self._telemetry_ok = (
                telemetry_ok
            )

        if not telemetry_ok:

            with self._lock:

                self._statistics.telemetry_failures += 1

            self._activate_event(
                SafetyEvent.TELEMETRY_LOST,
                {
                    "telemetry_age_s": age,

                    "timeout_s": (
                        self.config
                        .telemetry_timeout_s
                    ),
                },
            )

        else:

            self._clear_event(
                SafetyEvent.TELEMETRY_LOST
            )

    # =========================================================================
    # Overall state
    # =========================================================================

    def _update_overall_state(self) -> None:
        """Calculate overall safety state."""

        with self._lock:

            active = set(
                self._active_events.keys()
            )

            if (
                SafetyEvent.BATTERY_CRITICAL
                in active
            ):

                self._state = (
                    SafetyState.EMERGENCY
                )

                return

            if (
                SafetyEvent.HEARTBEAT_TIMEOUT
                in active
            ):

                self._state = (
                    SafetyState.CRITICAL
                )

                return

            if (
                SafetyEvent.TELEMETRY_LOST
                in active
            ):

                self._state = (
                    SafetyState.CRITICAL
                )

                return

            if (
                SafetyEvent.GPS_LOST
                in active
            ):

                self._state = (
                    SafetyState.CRITICAL
                )

                return

            if active:

                self._state = (
                    SafetyState.WARNING
                )

                return

            self._state = (
                SafetyState.SAFE
            )

    # =========================================================================
    # Event activation
    # =========================================================================

    def _activate_event(
        self,
        event: SafetyEvent,
        payload: dict,
    ) -> None:
        """Activate safety event."""

        now = time.monotonic()

        with self._lock:

            was_active = (
                event
                in self._active_events
            )

            self._active_events[
                event
            ] = now

            last_publish = (
                self._last_event_publish.get(
                    event
                )
            )

            should_publish = (
                not was_active
                or last_publish is None
                or (
                    now
                    - last_publish
                    >= self.config
                    .event_repeat_interval_s
                )
            )

            if should_publish:

                self._last_event_publish[
                    event
                ] = now

                self._statistics.last_event_time = now

        if not should_publish:
            return

        event_payload = dict(
            payload
        )

        event_payload.update(
            {
                "safety_event": event.value,

                "timestamp": now,

                "source": self.name,
            }
        )

        self._publish(
            event,
            event_payload,
        )

        logger.warning(
            "[SafetyMonitor] "
            "Safety event: %s | %s",
            event.value,
            payload,
        )

    # =========================================================================
    # Event clearing
    # =========================================================================

    def _clear_event(
        self,
        event: SafetyEvent,
    ) -> None:
        """Clear active safety event."""

        with self._lock:

            was_active = (
                event
                in self._active_events
            )

            if not was_active:
                return

            self._active_events.pop(
                event,
                None,
            )

            self._statistics.recovered_events += 1

        self._publish(
            SafetyEvent.SAFETY_RECOVERED,
            {
                "recovered_event": event.value,

                "timestamp": time.monotonic(),

                "source": self.name,
            },
        )

    # =========================================================================
    # Recovery
    # =========================================================================

    def _check_recovery(self) -> None:
        """
        Publish recovery when all safety conditions are normal.

        State transition itself is handled by _update_overall_state().
        """

        with self._lock:

            if self._active_events:
                return

            if self._state != SafetyState.SAFE:
                return

        # No active problems remain.

    # =========================================================================
    # Event publishing
    # =========================================================================

    def _publish(
        self,
        event: SafetyEvent,
        payload: dict,
    ) -> None:
        """Publish safety event to EventBus."""

        try:

            # Use the event with the matching EventType name when available.
            from core.events import EventType

            event_name = event.name

            event_type = getattr(
                EventType,
                event_name,
                None,
            )

            if event_type is None:

                logger.warning(
                    "[SafetyMonitor] "
                    "EventType.%s does not exist",
                    event_name,
                )

                return

            self._event_bus.publish(
                event_type,
                payload,
            )

        except Exception:

            logger.exception(
                "[SafetyMonitor] "
                "Failed to publish event %s",
                event.value,
            )

    # =========================================================================
    # Shared data helpers
    # =========================================================================

    def _get_heartbeat_age(
        self,
    ) -> Optional[float]:
        """
        Read heartbeat age from shared data.

        Supported APIs:

            get_heartbeat_age()

            heartbeat_age

        If unavailable, return None.
        """

        try:

            method = getattr(
                self._shared,
                "get_heartbeat_age",
                None,
            )

            if callable(method):

                value = method()

                if value is None:
                    return None

                return float(value)

            value = getattr(
                self._shared,
                "heartbeat_age",
                None,
            )

            if value is not None:

                return float(value)

        except Exception:

            logger.exception(
                "[SafetyMonitor] "
                "Failed to read heartbeat age"
            )

        return None

    # -------------------------------------------------------------------------

    def _get_battery_percent(
        self,
    ) -> Optional[float]:
        """Read battery percentage."""

        try:

            method = getattr(
                self._shared,
                "get_battery_percent",
                None,
            )

            if callable(method):

                value = method()

                if value is None:
                    return None

                return float(value)

            value = getattr(
                self._shared,
                "battery_percent",
                None,
            )

            if value is not None:

                return float(value)

        except Exception:

            logger.exception(
                "[SafetyMonitor] "
                "Failed to read battery percentage"
            )

        return None

    # -------------------------------------------------------------------------

    def _get_satellite_count(
        self,
    ) -> Optional[int]:
        """Read GPS satellite count."""

        try:

            method = getattr(
                self._shared,
                "get_satellite_count",
                None,
            )

            if callable(method):

                value = method()

                if value is None:
                    return None

                return int(value)

            value = getattr(
                self._shared,
                "satellite_count",
                None,
            )

            if value is not None:

                return int(value)

        except Exception:

            logger.exception(
                "[SafetyMonitor] "
                "Failed to read satellite count"
            )

        return None

    # -------------------------------------------------------------------------

    def _get_telemetry_age(
        self,
    ) -> Optional[float]:
        """Read telemetry age."""

        try:

            method = getattr(
                self._shared,
                "get_telemetry_age",
                None,
            )

            if callable(method):

                value = method()

                if value is None:
                    return None

                return float(value)

            value = getattr(
                self._shared,
                "telemetry_age",
                None,
            )

            if value is not None:

                return float(value)

        except Exception:

            logger.exception(
                "[SafetyMonitor] "
                "Failed to read telemetry age"
            )

        return None

    # =========================================================================
    # Manual state
    # =========================================================================

    def force_event(
        self,
        event: SafetyEvent,
        payload: Optional[dict] = None,
    ) -> None:

        self._activate_event(
            event,
            payload or {},
        )

        self._update_overall_state()

    # =========================================================================
    # Clear all
    # =========================================================================

    def clear_events(self) -> None:
        """Clear all currently active safety events."""

        with self._lock:

            events = list(
                self._active_events.keys()
            )

        for event in events:

            self._clear_event(
                event
            )

        self._update_overall_state()

    # =========================================================================
    # Snapshot
    # =========================================================================

    def snapshot(
        self,
    ) -> SafetySnapshot:
        """Return immutable safety snapshot."""

        with self._lock:

            return SafetySnapshot(
                state=self._state,

                heartbeat_ok=(
                    self._heartbeat_ok
                ),

                battery_ok=(
                    self._battery_ok
                ),

                battery_percent=(
                    self._battery_percent
                ),

                gps_ok=(
                    self._gps_ok
                ),

                satellite_count=(
                    self._satellite_count
                ),

                telemetry_ok=(
                    self._telemetry_ok
                ),

                active_events=tuple(
                    event.value
                    for event
                    in self._active_events
                ),

                timestamp=time.monotonic(),
            )

    # =========================================================================
    # Statistics
    # =========================================================================

    def statistics(
        self,
    ) -> SafetyStatistics:
        """Return statistics snapshot."""

        with self._lock:

            return SafetyStatistics(
                checks=(
                    self._statistics.checks
                ),

                heartbeat_failures=(
                    self._statistics
                    .heartbeat_failures
                ),

                low_battery_events=(
                    self._statistics
                    .low_battery_events
                ),

                critical_battery_events=(
                    self._statistics
                    .critical_battery_events
                ),

                gps_failures=(
                    self._statistics
                    .gps_failures
                ),

                telemetry_failures=(
                    self._statistics
                    .telemetry_failures
                ),

                recovered_events=(
                    self._statistics
                    .recovered_events
                ),

                last_event_time=(
                    self._statistics
                    .last_event_time
                ),
            )

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def state(
        self,
    ) -> SafetyState:

        with self._lock:
            return self._state

    # -------------------------------------------------------------------------

    @property
    def running(
        self,
    ) -> bool:

        with self._lock:
            return self._running

    # -------------------------------------------------------------------------

    @property
    def active_events(
        self,
    ) -> tuple[str, ...]:

        with self._lock:

            return tuple(
                event.value
                for event
                in self._active_events
            )

    # =========================================================================
    # Representation
    # =========================================================================

    def __repr__(self) -> str:

        snapshot = self.snapshot()

        return (
            "SafetyMonitor("
            f"state={snapshot.state.value!r}, "
            f"running={self.running}, "
            f"events={snapshot.active_events}"
            ")"
        )
