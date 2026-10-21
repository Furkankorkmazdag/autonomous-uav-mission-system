"""Detects stalled components and coordinates configured recovery actions."""

from __future__ import annotations

import logging
import threading
import time

from dataclasses import dataclass
from typing import Any, Dict, Optional


logger = logging.getLogger(__name__)


# ============================================================================
# Watchdog configuration
# ============================================================================

@dataclass(frozen=True)
class WatchdogConfig:

    poll_interval_s: float = 1.0

    freeze_timeout_s: float = 10.0

    maximum_restarts: int = 3

    restart_delay_s: float = 1.0

    def __post_init__(self) -> None:

        if self.poll_interval_s <= 0:
            raise ValueError(
                "poll_interval_s must be greater than zero"
            )

        if self.freeze_timeout_s <= 0:
            raise ValueError(
                "freeze_timeout_s must be greater than zero"
            )

        if self.maximum_restarts < 0:
            raise ValueError(
                "maximum_restarts cannot be negative"
            )

        if self.restart_delay_s < 0:
            raise ValueError(
                "restart_delay_s cannot be negative"
            )


# ============================================================================
# Mission watchdog record
# ============================================================================

@dataclass
class WatchdogRecord:
    """Runtime information for one monitored mission."""

    mission_name: str

    last_heartbeat: float

    restart_count: int = 0

    frozen_count: int = 0

    enabled: bool = True

    last_error: Optional[str] = None


# ============================================================================
# Mission Watchdog
# ============================================================================

class MissionWatchdog:

    def __init__(
        self,
        mission_manager: Any,
        config: Optional[
            WatchdogConfig
        ] = None,
    ) -> None:

        if mission_manager is None:
            raise ValueError(
                "mission_manager cannot be None"
            )

        self._mission_manager = (
            mission_manager
        )

        self.config = (
            config
            or WatchdogConfig()
        )

        self._lock = threading.RLock()

        self._records: Dict[
            str,
            WatchdogRecord,
        ] = {}

        self._running = False

        self._thread: Optional[
            threading.Thread
        ] = None

        self._stop_event = (
            threading.Event()
        )

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def start(self) -> None:
        """Start watchdog thread."""

        with self._lock:

            if self._running:
                return

            self._running = True

            self._stop_event.clear()

            self._thread = threading.Thread(
                target=self._run,
                name="MissionWatchdog",
                daemon=True,
            )

            self._thread.start()

        logger.info(
            "[MissionWatchdog] Started"
        )

    # -------------------------------------------------------------------------

    def stop(self) -> None:
        """Stop watchdog thread."""

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

        logger.info(
            "[MissionWatchdog] Stopped"
        )

    # =========================================================================
    # Register
    # =========================================================================

    def register(
        self,
        mission_name: str,
    ) -> None:
        """Register mission for watchdog monitoring."""

        now = time.monotonic()

        with self._lock:

            self._records[
                mission_name
            ] = WatchdogRecord(
                mission_name=mission_name,
                last_heartbeat=now,
            )

    # =========================================================================
    # Unregister
    # =========================================================================

    def unregister(
        self,
        mission_name: str,
    ) -> None:
        """Remove mission from watchdog."""

        with self._lock:

            self._records.pop(
                mission_name,
                None,
            )

    # =========================================================================
    # Heartbeat
    # =========================================================================

    def heartbeat(
        self,
        mission_name: str,
    ) -> None:

        now = time.monotonic()

        with self._lock:

            record = self._records.get(
                mission_name
            )

            if record is None:

                self._records[
                    mission_name
                ] = WatchdogRecord(
                    mission_name=mission_name,
                    last_heartbeat=now,
                )

                return

            record.last_heartbeat = now

            record.last_error = None

    # =========================================================================
    # Main loop
    # =========================================================================

    def _run(self) -> None:
        """Watchdog monitoring loop."""

        while not self._stop_event.wait(
            self.config.poll_interval_s
        ):

            try:

                self._check_missions()

            except Exception:

                logger.exception(
                    "[MissionWatchdog] "
                    "Watchdog cycle failed"
                )

    # =========================================================================
    # Mission checks
    # =========================================================================

    def _check_missions(self) -> None:
        """Check all registered missions."""

        now = time.monotonic()

        with self._lock:

            records = list(
                self._records.values()
            )

        for record in records:

            if not record.enabled:
                continue

            age = (
                now
                - record.last_heartbeat
            )

            if (
                age
                <= self.config.freeze_timeout_s
            ):
                continue

            self._handle_frozen_mission(
                record,
                age,
            )

    # =========================================================================
    # Frozen mission
    # =========================================================================

    def _handle_frozen_mission(
        self,
        record: WatchdogRecord,
        heartbeat_age: float,
    ) -> None:

        with self._lock:

            current = self._records.get(
                record.mission_name
            )

            if current is None:
                return

            current.frozen_count += 1

            if (
                current.restart_count
                >= self.config.maximum_restarts
            ):

                current.last_error = (
                    "Maximum watchdog restarts exceeded"
                )

                logger.error(
                    "[MissionWatchdog] "
                    "Mission %s exceeded restart limit",
                    record.mission_name,
                )

                self._abort_mission(
                    record.mission_name,
                    reason=(
                        "watchdog_restart_limit"
                    ),
                )

                return

            current.restart_count += 1

            restart_number = (
                current.restart_count
            )

        logger.warning(
            "[MissionWatchdog] "
            "Mission %s appears frozen "
            "(heartbeat age %.2fs), "
            "restart %d/%d",
            record.mission_name,
            heartbeat_age,
            restart_number,
            self.config.maximum_restarts,
        )

        self._restart_mission(
            record.mission_name
        )

    # =========================================================================
    # Restart
    # =========================================================================

    def _restart_mission(
        self,
        mission_name: str,
    ) -> None:
        """Restart mission through MissionManager."""

        try:

            self._mission_manager.abort_mission(
                mission_name
            )

            if (
                self.config.restart_delay_s
                > 0
            ):

                time.sleep(
                    self.config.restart_delay_s
                )

            self._mission_manager.start_mission(
                mission_name
            )

            self.heartbeat(
                mission_name
            )

        except Exception as exc:

            with self._lock:

                record = self._records.get(
                    mission_name
                )

                if record is not None:

                    record.last_error = str(
                        exc
                    )

            logger.exception(
                "[MissionWatchdog] "
                "Failed to restart mission %s",
                mission_name,
            )

    # =========================================================================
    # Abort
    # =========================================================================

    def _abort_mission(
        self,
        mission_name: str,
        reason: str,
    ) -> None:
        """Abort mission through MissionManager."""

        try:

            self._mission_manager.abort_mission(
                mission_name
            )

            self._publish_failure(
                mission_name,
                reason,
            )

        except Exception as exc:

            logger.exception(
                "[MissionWatchdog] "
                "Failed to abort mission %s: %s",
                mission_name,
                exc,
            )

    # =========================================================================
    # Event publishing
    # =========================================================================

    def _publish_failure(
        self,
        mission_name: str,
        reason: str,
    ) -> None:
        """
        Publish watchdog failure event when MissionManager
        exposes an event bus.
        """

        event_bus = getattr(
            self._mission_manager,
            "events",
            None,
        )

        if event_bus is None:
            return

        try:

            from core.events import EventType

            event_bus.publish(
                EventType.MISSION_FAILED,
                {
                    "mission_name": mission_name,
                    "reason": reason,
                    "source": "watchdog",
                },
            )

        except Exception:

            logger.exception(
                "[MissionWatchdog] "
                "Failed to publish failure event"
            )

    # =========================================================================
    # Status
    # =========================================================================

    def is_running(self) -> bool:

        with self._lock:
            return self._running

    # -------------------------------------------------------------------------

    def get_record(
        self,
        mission_name: str,
    ) -> Optional[WatchdogRecord]:

        with self._lock:

            return self._records.get(
                mission_name
            )

    # -------------------------------------------------------------------------

    def records(
        self,
    ) -> Dict[
        str,
        WatchdogRecord,
    ]:

        with self._lock:

            return dict(
                self._records
            )

    # =========================================================================
    # Reset restart counter
    # =========================================================================

    def reset_restart_count(
        self,
        mission_name: str,
    ) -> None:
        """Reset watchdog restart counter."""

        with self._lock:

            record = self._records.get(
                mission_name
            )

            if record is None:
                return

            record.restart_count = 0

            record.frozen_count = 0

            record.last_error = None

            record.last_heartbeat = (
                time.monotonic()
            )

    # =========================================================================
    # Representation
    # =========================================================================

    def __repr__(self) -> str:

        with self._lock:

            return (
                "MissionWatchdog("
                f"running={self._running}, "
                f"missions={len(self._records)}"
                ")"
            )
