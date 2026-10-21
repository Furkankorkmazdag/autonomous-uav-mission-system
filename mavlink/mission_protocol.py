"""Implements MAVLink mission upload, download, and current-item operations."""

from __future__ import annotations

import logging
import threading
import time

from queue import Queue, Empty

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence


try:
    from pymavlink import mavutil
except ImportError as exc:
    raise ImportError(
        "pymavlink is required. "
        "Install it with: pip install pymavlink"
    ) from exc


logger = logging.getLogger(__name__)


# ============================================================================
# Mission Item
# ============================================================================

@dataclass(frozen=True)
class MissionItem:
    """
    MAVLink mission item.

    latitude:
        Latitude in degrees.

    longitude:
        Longitude in degrees.

    altitude_m:
        Altitude in meters.

    seq:
        Mission sequence number.

    command:
        MAVLink command ID.

    frame:
        MAVLink coordinate frame.

    current:
        Current mission item flag.

    autocontinue:
        Autocontinue flag.

    param1..param4:
        MAVLink command parameters.
    """

    latitude: float

    longitude: float

    altitude_m: float

    seq: int = 0

    command: int = (
        mavutil.mavlink.MAV_CMD_NAV_WAYPOINT
    )

    frame: int = (
        mavutil.mavlink.MAV_FRAME_GLOBAL_INT
    )

    current: int = 0

    autocontinue: int = 1

    param1: float = 0.0

    param2: float = 0.0

    param3: float = 0.0

    param4: float = 0.0


# ============================================================================
# Mission Result
# ============================================================================

@dataclass(frozen=True)
class MissionResult:
    """
    Mission protocol operation result.
    """

    success: bool

    message: str = ""

    count: int = 0

    items: tuple[MissionItem, ...] = ()


# ============================================================================
# Mission Protocol
# ============================================================================

class MissionProtocol:

    name = "MissionProtocol"

    def __init__(
        self,
        mavlink: Any,
        timeout_s: float = 5.0,
    ) -> None:

        if mavlink is None:

            raise ValueError(
                "mavlink cannot be None"
            )

        if timeout_s <= 0:

            raise ValueError(
                "timeout_s must be greater than zero"
            )

        self._mavlink = mavlink

        self.timeout_s = (
            float(timeout_s)
        )

        self._lock = threading.RLock()

        self._running = False

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def start(self) -> bool:
        """
        Activate the mission protocol.
        """

        with self._lock:

            if self._running:

                return True

            self._running = True

        logger.info(
            "[MissionProtocol] Started"
        )

        return True

    # =========================================================================
    # Stop
    # =========================================================================

    def stop(self) -> None:
        """
        Stop the mission protocol.
        """

        with self._lock:

            self._running = False

        logger.info(
            "[MissionProtocol] Stopped"
        )

    # =========================================================================
    # Master
    # =========================================================================

    def _master(self) -> Any:
        """
        Return the central MAVLink master connection.
        """

        master = getattr(
            self._mavlink,
            "master",
            None,
        )

        if master is None:

            raise RuntimeError(
                "MAVLink connection is not available"
            )

        return master

    # =========================================================================
    # Target system
    # =========================================================================

    @property
    def target_system(self) -> int:

        return int(
            getattr(
                self._mavlink,
                "target_system",
                1,
            )
        )

    # =========================================================================
    # Target component
    # =========================================================================

    @property
    def target_component(self) -> int:

        return int(
            getattr(
                self._mavlink,
                "target_component",
                1,
            )
        )

    # =========================================================================
    # Running
    # =========================================================================

    @property
    def running(self) -> bool:

        with self._lock:

            return self._running

    # =========================================================================
    # Connected
    # =========================================================================

    @property
    def connected(self) -> bool:

        return bool(
            getattr(
                self._mavlink,
                "connected",
                False,
            )
        )

    # =========================================================================
    # Clear Mission
    # =========================================================================

    def clear(
        self,
    ) -> MissionResult:

        if not self.connected:

            return MissionResult(
                success=False,

                message=(
                    "MAVLink is not connected"
                ),
            )

        with self._lock:

            try:

                master = self._master()

                master.mav.mission_clear_all_send(
                    self.target_system,
                    self.target_component,

                    mavutil.mavlink
                    .MAV_MISSION_TYPE_MISSION,
                )

                logger.info(
                    "[MissionProtocol] "
                    "Mission clear command sent"
                )

                return MissionResult(
                    success=True,

                    message=(
                        "Mission clear command sent"
                    ),
                )

            except Exception as exc:

                logger.exception(
                    "[MissionProtocol] "
                    "Mission clear failed"
                )

                return MissionResult(
                    success=False,

                    message=str(exc),
                )

    # =========================================================================
    # Upload
    # =========================================================================

    def upload(
        self,
        items: Sequence[MissionItem],
        timeout_s: Optional[float] = None,
    ) -> MissionResult:

        if not self.connected:

            return MissionResult(
                success=False,

                message=(
                    "MAVLink is not connected"
                ),
            )

        if not items:

            return MissionResult(
                success=False,

                message=(
                    "Mission cannot be empty"
                ),
            )

        timeout = (
            timeout_s
            if timeout_s is not None
            else self.timeout_s
        )

        if timeout <= 0:

            return MissionResult(
                success=False,

                message=(
                    "timeout_s must be greater than zero"
                ),
            )

        normalized = (
            self._normalize_items(
                items
            )
        )

        with self._lock:

            router = getattr(
                self._mavlink,
                "message_router",
                None,
            )

            if router is None:
                return MissionResult(
                    success=False,
                    message=(
                        "MAVLink MessageRouter is not available"
                    ),
                    count=len(normalized),
                )

            master = self._master()

            # IMPORTANT:
            # MAVLinkManager has the single receive loop. Therefore this
            # method must NOT call master.recv_match(). Mission requests and
            # ACKs are delivered by MessageRouter.
            incoming: Queue[Any] = Queue()

            def _on_mission_message(message: Any) -> None:
                incoming.put(message)

            subscription_ids = []

            try:
                subscription_ids = router.subscribe_many(
                    [
                        "MISSION_REQUEST_INT",
                        "MISSION_REQUEST",
                        "MISSION_ACK",
                    ],
                    _on_mission_message,
                    owner="MissionProtocol.upload",
                )

                # -------------------------------------------------------------
                # Start from a clean mission transfer state. ArduPilot cancels
                # an in-progress upload from the same sender when a new
                # MISSION_COUNT arrives, but explicitly clearing first avoids
                # stale/partial mission state between retries.
                # -------------------------------------------------------------
                logger.info("[MissionProtocol] Clearing existing mission before upload")
                master.mav.mission_clear_all_send(
                    self.target_system,
                    self.target_component,
                    mavutil.mavlink.MAV_MISSION_TYPE_MISSION,
                )

                clear_deadline = time.monotonic() + min(2.0, timeout)
                while time.monotonic() < clear_deadline:
                    try:
                        clear_msg = incoming.get(timeout=0.1)
                    except Empty:
                        continue
                    if clear_msg.get_type() == "MISSION_ACK":
                        clear_type = int(getattr(clear_msg, "type", -1))
                        logger.info(
                            "[MissionProtocol] Mission clear ACK=%d",
                            clear_type,
                        )
                        break

                # -------------------------------------------------------------
                # Mission count
                # -------------------------------------------------------------

                logger.info(
                    "[MissionProtocol] Sending MISSION_COUNT=%d",
                    len(normalized),
                )
                master.mav.mission_count_send(
                    self.target_system,
                    self.target_component,
                    len(normalized),
                    mavutil.mavlink.MAV_MISSION_TYPE_MISSION,
                )

                deadline = (
                    time.monotonic()
                    + timeout
                )

                while time.monotonic() < deadline:

                    remaining = max(
                        0.05,
                        min(
                            0.5,
                            deadline - time.monotonic(),
                        ),
                    )

                    try:
                        message = incoming.get(
                            timeout=remaining
                        )
                    except Empty:
                        continue

                    message_type = message.get_type()

                    # ---------------------------------------------------------
                    # ACK
                    # ---------------------------------------------------------

                    if message_type == "MISSION_ACK":

                        if (
                            int(message.type)
                            == mavutil.mavlink.MAV_MISSION_ACCEPTED
                        ):

                            logger.info(
                                "[MissionProtocol] "
                                "Mission upload accepted"
                            )

                            return MissionResult(
                                success=True,
                                message="Mission upload accepted",
                                count=len(normalized),
                                items=tuple(normalized),
                            )

                        logger.error(
                            "[MissionProtocol] Mission rejected ACK=%s (%s); "
                            "last_requested_seq=%s",
                            message.type,
                            {
                                13: "MAV_MISSION_INVALID_SEQUENCE",
                                12: "MAV_MISSION_INVALID_PARAM7",
                                11: "MAV_MISSION_INVALID_PARAM6_Y",
                                10: "MAV_MISSION_INVALID_PARAM5_X",
                                5: "MAV_MISSION_INVALID",
                                3: "MAV_MISSION_UNSUPPORTED",
                            }.get(int(message.type), "unknown"),
                            locals().get("seq", None),
                        )

                        return MissionResult(
                            success=False,
                            message=(
                                "Mission rejected "
                                f"ACK={message.type}"
                            ),
                            count=len(normalized),
                        )

                    # ---------------------------------------------------------
                    # Mission request
                    # ---------------------------------------------------------

                    try:
                        seq = int(message.seq)
                    except (AttributeError, TypeError, ValueError):
                        logger.error(
                            "[MissionProtocol] "
                            "Mission request has invalid sequence"
                        )
                        return MissionResult(
                            success=False,
                            message="Invalid mission request sequence",
                            count=len(normalized),
                        )

                    if not 0 <= seq < len(normalized):

                        logger.error(
                            "[MissionProtocol] "
                            "Invalid mission request seq=%d",
                            seq,
                        )

                        return MissionResult(
                            success=False,
                            message=(
                                f"Invalid mission request "
                                f"sequence={seq}"
                            ),
                            count=len(normalized),
                        )

                    item = normalized[seq]

                    logger.info(
                        "[MissionProtocol] MISSION_REQUEST %s seq=%d/%d -> "
                        "cmd=%d frame=%d current=%d lat=%.7f lon=%.7f alt=%.2f",
                        message_type,
                        seq,
                        len(normalized) - 1,
                        item.command,
                        item.frame,
                        item.current,
                        item.latitude,
                        item.longitude,
                        item.altitude_m,
                    )

                    self._send_item(
                        master,
                        item,
                    )

                logger.error(
                    "[MissionProtocol] "
                    "Mission upload timeout after %.2fs",
                    timeout,
                )

                return MissionResult(
                    success=False,
                    message="Mission upload timeout",
                    count=len(normalized),
                )

            except Exception as exc:

                logger.exception(
                    "[MissionProtocol] "
                    "Mission upload failed"
                )

                return MissionResult(
                    success=False,
                    message=str(exc),
                    count=len(normalized),
                )

            finally:

                for subscription_id in subscription_ids:

                    try:
                        router.unsubscribe(
                            subscription_id
                        )
                    except Exception:
                        logger.exception(
                            "[MissionProtocol] "
                            "Failed to remove temporary upload subscription"
                        )

    # =========================================================================
    # Send Item
    # =========================================================================

    def _send_item(
        self,
        master: Any,
        item: MissionItem,
    ) -> None:

        master.mav.mission_item_int_send(
            self.target_system,
            self.target_component,

            item.seq,

            item.frame,

            item.command,

            item.current,

            item.autocontinue,

            item.param1,
            item.param2,
            item.param3,
            item.param4,

            int(
                round(
                    item.latitude
                    * 1e7
                )
            ),

            int(
                round(
                    item.longitude
                    * 1e7
                )
            ),

            float(
                item.altitude_m
            ),

            mavutil.mavlink
            .MAV_MISSION_TYPE_MISSION,
        )

        logger.debug(
            "[MissionProtocol] "
            "Sent mission item seq=%d "
            "lat=%.7f lon=%.7f alt=%.2f",
            item.seq,
            item.latitude,
            item.longitude,
            item.altitude_m,
        )

    # =========================================================================
    # Normalize
    # =========================================================================

    def _normalize_items(
        self,
        items: Sequence[MissionItem],
    ) -> List[MissionItem]:

        result = []

        for index, item in enumerate(
            items
        ):

            result.append(
                MissionItem(
                    latitude=item.latitude,

                    longitude=item.longitude,

                    altitude_m=item.altitude_m,

                    seq=index,

                    command=item.command,

                    frame=item.frame,

                    current=(
                        1
                        if index == 0
                        and item.current
                        else 0
                    ),

                    autocontinue=item.autocontinue,

                    param1=item.param1,

                    param2=item.param2,

                    param3=item.param3,

                    param4=item.param4,
                )
            )

        if result:

            # If no mission item is marked as current,
            # the first item becomes the current item.

            if not any(
                item.current
                for item in result
            ):

                first = result[0]

                result[0] = MissionItem(
                    latitude=first.latitude,

                    longitude=first.longitude,

                    altitude_m=first.altitude_m,

                    seq=first.seq,

                    command=first.command,

                    frame=first.frame,

                    current=1,

                    autocontinue=first.autocontinue,

                    param1=first.param1,

                    param2=first.param2,

                    param3=first.param3,

                    param4=first.param4,
                )

        return result

    # =========================================================================
    # Download
    # =========================================================================

    def download(
        self,
        timeout_s: Optional[float] = None,
    ) -> MissionResult:

        if not self.connected:

            return MissionResult(
                success=False,

                message=(
                    "MAVLink is not connected"
                ),
            )

        timeout = (
            timeout_s
            if timeout_s is not None
            else self.timeout_s
        )

        with self._lock:

            try:

                master = self._master()

                # -------------------------------------------------------------
                # Request mission list
                # -------------------------------------------------------------

                master.mav.mission_request_list_send(
                    self.target_system,
                    self.target_component,

                    mavutil.mavlink
                    .MAV_MISSION_TYPE_MISSION,
                )

                deadline = (
                    time.monotonic()
                    + timeout
                )

                count = None

                while (
                    time.monotonic()
                    < deadline
                ):

                    message = (
                        master.recv_match(
                            type="MISSION_COUNT",

                            blocking=True,

                            timeout=0.5,
                        )
                    )

                    if message is None:

                        continue

                    count = int(
                        message.count
                    )

                    break

                if count is None:

                    return MissionResult(
                        success=False,

                        message=(
                            "MISSION_COUNT timeout"
                        ),
                    )

                if count == 0:

                    master.mav.mission_ack_send(
                        self.target_system,
                        self.target_component,

                        mavutil.mavlink
                        .MAV_MISSION_ACCEPTED,

                        mavutil.mavlink
                        .MAV_MISSION_TYPE_MISSION,
                    )

                    return MissionResult(
                        success=True,

                        message=(
                            "Mission is empty"
                        ),

                        count=0,
                    )

                # -------------------------------------------------------------
                # Request each item
                # -------------------------------------------------------------

                result = []

                for seq in range(
                    count
                ):

                    master.mav.mission_request_int_send(
                        self.target_system,
                        self.target_component,

                        seq,

                        mavutil.mavlink
                        .MAV_MISSION_TYPE_MISSION,
                    )

                    item_message = (
                        self._wait_for_item(
                            master,
                            seq,
                            timeout,
                        )
                    )

                    if item_message is None:

                        return MissionResult(
                            success=False,

                            message=(
                                "Mission item timeout "
                                f"seq={seq}"
                            ),

                            count=count,

                            items=tuple(
                                result
                            ),
                        )

                    item = (
                        self._message_to_item(
                            item_message
                        )
                    )

                    result.append(
                        item
                    )

                # -------------------------------------------------------------
                # ACK
                # -------------------------------------------------------------

                master.mav.mission_ack_send(
                    self.target_system,
                    self.target_component,

                    mavutil.mavlink
                    .MAV_MISSION_ACCEPTED,

                    mavutil.mavlink
                    .MAV_MISSION_TYPE_MISSION,
                )

                logger.info(
                    "[MissionProtocol] "
                    "Downloaded %d mission items",
                    len(result),
                )

                return MissionResult(
                    success=True,

                    message=(
                        "Mission download successful"
                    ),

                    count=len(result),

                    items=tuple(
                        result
                    ),
                )

            except Exception as exc:

                logger.exception(
                    "[MissionProtocol] "
                    "Mission download failed"
                )

                return MissionResult(
                    success=False,

                    message=str(exc),
                )

    # =========================================================================
    # Wait for Item
    # =========================================================================

    def _wait_for_item(
        self,
        master: Any,
        expected_seq: int,
        timeout_s: float,
    ) -> Optional[Any]:

        deadline = (
            time.monotonic()
            + timeout_s
        )

        while (
            time.monotonic()
            < deadline
        ):

            message = (
                master.recv_match(
                    type=[
                        "MISSION_ITEM_INT",
                        "MISSION_ITEM",
                    ],

                    blocking=True,

                    timeout=0.5,
                )
            )

            if message is None:

                continue

            seq = int(
                message.seq
            )

            if seq != expected_seq:

                logger.warning(
                    "[MissionProtocol] "
                    "Received unexpected "
                    "mission seq=%d expected=%d",
                    seq,
                    expected_seq,
                )

                continue

            return message

        return None

    # =========================================================================
    # Message -> Mission Item
    # =========================================================================

    def _message_to_item(
        self,
        message: Any,
    ) -> MissionItem:

        message_type = (
            message.get_type()
        )

        if message_type == "MISSION_ITEM_INT":

            latitude = (
                message.x
                / 1e7
            )

            longitude = (
                message.y
                / 1e7
            )

        else:

            latitude = float(
                message.x
            )

            longitude = float(
                message.y
            )

        return MissionItem(
            latitude=latitude,

            longitude=longitude,

            altitude_m=float(
                message.z
            ),

            seq=int(
                message.seq
            ),

            command=int(
                message.command
            ),

            frame=int(
                message.frame
            ),

            current=int(
                message.current
            ),

            autocontinue=int(
                message.autocontinue
            ),

            param1=float(
                message.param1
            ),

            param2=float(
                message.param2
            ),

            param3=float(
                message.param3
            ),

            param4=float(
                message.param4
            ),
        )

    # =========================================================================
    # Set Current
    # =========================================================================

    def set_current(
        self,
        sequence: int,
    ) -> MissionResult:

        if sequence < 0:

            return MissionResult(
                success=False,

                message=(
                    "Sequence cannot be negative"
                ),
            )

        if not self.connected:

            return MissionResult(
                success=False,

                message=(
                    "MAVLink is not connected"
                ),
            )

        with self._lock:

            try:

                master = self._master()

                master.mav.mission_set_current_send(
                    self.target_system,
                    self.target_component,

                    sequence,
                )

                logger.info(
                    "[MissionProtocol] "
                    "Current mission set to %d",
                    sequence,
                )

                return MissionResult(
                    success=True,

                    message=(
                        f"Current mission set to {sequence}"
                    ),
                )

            except Exception as exc:

                logger.exception(
                    "[MissionProtocol] "
                    "Set current failed"
                )

                return MissionResult(
                    success=False,

                    message=str(exc),
                )

    # =========================================================================
    # Read Current
    # =========================================================================

    def current(
        self,
    ) -> Optional[int]:

        method = getattr(
            self._mavlink,
            "mission_progress",
            None,
        )

        if not callable(method):

            return None

        try:

            current, _ = method()

            return int(
                current
            )

        except Exception:

            logger.exception(
                "[MissionProtocol] "
                "Failed to read current mission"
            )

            return None

    # =========================================================================
    # Count
    # =========================================================================

    def count(
        self,
    ) -> Optional[int]:

        method = getattr(
            self._mavlink,
            "mission_progress",
            None,
        )

        if not callable(method):

            return None

        try:

            _, count = method()

            return int(
                count
            )

        except Exception:

            logger.exception(
                "[MissionProtocol] "
                "Failed to read mission count"
            )

            return None

    # =========================================================================
    # Representation
    # =========================================================================

    def __repr__(
        self,
    ) -> str:

        with self._lock:

            return (
                "MissionProtocol("
                f"running={self._running}, "
                f"connected={self.connected}"
                ")"
            )
