"""High-level controller for AUTO missions and GUIDED target-centering control."""

from __future__ import annotations

import logging
import threading
from typing import Any, Optional, Sequence, Tuple

from pymavlink import mavutil

from core.events import EventType
from mavlink.mission_protocol import MissionItem
from navigation.target_centering import (
    TargetCenteringConfig,
    TargetCenteringController,
)

logger = logging.getLogger(__name__)


class FlightController:
    """Controls AUTO mission execution and GUIDED target-centering behaviour."""

    def __init__(
        self,
        mavlink: Any,
        event_bus: Any,
        waypoints: Sequence[Tuple[float, float]],
        altitude_m: float = 30.0,
        laps: int = 1,
        centering_config: Optional[TargetCenteringConfig] = None,
    ) -> None:
        if mavlink is None:
            raise ValueError("mavlink cannot be None")
        if event_bus is None:
            raise ValueError("event_bus cannot be None")
        if not waypoints:
            raise ValueError("waypoints cannot be empty")
        if altitude_m <= 0:
            raise ValueError("altitude_m must be positive")

        self.mavlink = mavlink
        self.event_bus = event_bus
        self.waypoints = list(waypoints)
        self.altitude_m = float(altitude_m)
        self.laps = max(1, int(laps))

        self.centering = TargetCenteringController(
            mavlink,
            centering_config,
        )

        self._lock = threading.RLock()
        self._mission_uploaded = False
        self._guided = False
        self._last_target_seen = False

        self._subscriptions = []
        self._subscribe()

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------

    def _subscribe(self) -> None:
        self.event_bus.subscribe(
            EventType.TARGET_DETECTED,
            self._on_target_detected,
        )
        self._subscriptions.append(
            (EventType.TARGET_DETECTED, self._on_target_detected)
        )

        self.event_bus.subscribe(
            EventType.TARGET_LOST,
            self._on_target_lost,
        )
        self._subscriptions.append(
            (EventType.TARGET_LOST, self._on_target_lost)
        )

    def close(self) -> None:
        for event_type, callback in self._subscriptions:
            self.event_bus.unsubscribe(event_type, callback)
        self._subscriptions.clear()
        self.centering.stop()

    def _on_target_detected(self, event) -> None:
        data = event.data or {}

        try:
            x = float(data.get("normalized_x", 0.5))
            y = float(data.get("normalized_y", 0.5))
        except (TypeError, ValueError):
            return

        with self._lock:
            if not self._guided:
                if not self.mavlink.set_mode("GUIDED"):
                    logger.error("[FlightController] GUIDED mode could not be selected")
                    return
                self._guided = True
                logger.info("[FlightController] AUTO -> GUIDED: target detected")

        self.centering.update(x, y)
        self._last_target_seen = True

    def _on_target_lost(self, event) -> None:
        self.centering.stop()

        with self._lock:
            if self._guided:
                if self.mavlink.set_mode("AUTO"):
                    logger.info("[FlightController] GUIDED -> AUTO: target lost")
                    self._guided = False

    # ------------------------------------------------------------------
    # AUTO mission
    # ------------------------------------------------------------------

    def build_mission_items(self) -> list[MissionItem]:
        lat0, lon0 = self.waypoints[0]

        items = [
            MissionItem(
                latitude=float(lat0),
                longitude=float(lon0),
                altitude_m=self.altitude_m,
                seq=0,
                command=mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
                frame=mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
            )
        ]

        seq = 1
        for _ in range(self.laps):
            for lat, lon in self.waypoints:
                items.append(
                    MissionItem(
                        latitude=float(lat),
                        longitude=float(lon),
                        altitude_m=self.altitude_m,
                        seq=seq,
                        command=mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
                        frame=mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                    )
                )
                seq += 1

        return items

    def upload_auto_mission(self) -> bool:
        if not self.mavlink.connected:
            logger.error("[FlightController] MAVLink not connected")
            return False

        result = self.mavlink.mission_protocol.upload(
            self.build_mission_items()
        )

        if not result.success:
            logger.error(
                "[FlightController] Mission upload failed: %s",
                result.message,
            )
            return False

        current = self.mavlink.mission_protocol.set_current(0)
        if not current.success:
            logger.warning(
                "[FlightController] Could not set mission current: %s",
                current.message,
            )

        self._mission_uploaded = True
        logger.info(
            "[FlightController] AUTO mission uploaded: %d items",
            result.count,
        )
        return True

    def start_auto(self) -> bool:
        if not self._mission_uploaded and not self.upload_auto_mission():
            return False

        if not self.mavlink.arm():
            logger.error("[FlightController] ARM failed")
            return False

        if not self.mavlink.set_mode("AUTO"):
            logger.error("[FlightController] AUTO mode could not be selected")
            return False

        with self._lock:
            self._guided = False

        logger.info("[FlightController] AUTO mission started")
        return True

    def guided(self) -> bool:
        ok = self.mavlink.set_mode("GUIDED")
        if ok:
            with self._lock:
                self._guided = True
        return ok

    def return_to_auto(self) -> bool:
        self.centering.stop()
        ok = self.mavlink.set_mode("AUTO")
        if ok:
            with self._lock:
                self._guided = False
        return ok

    def rtl(self) -> bool:
        self.centering.stop()
        with self._lock:
            self._guided = False
        return self.mavlink.rtl()

    def status(self) -> dict:
        return {
            "connected": bool(self.mavlink.connected),
            "mode_guided": self._guided,
            "mission_uploaded": self._mission_uploaded,
            "heartbeat_age": self.mavlink.heartbeat_age,
        }
