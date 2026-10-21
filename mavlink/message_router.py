"""Routes received MAVLink messages to registered subscribers."""

from __future__ import annotations

import logging
import threading
import time

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, DefaultDict, Dict, List, Optional


logger = logging.getLogger(__name__)


# ============================================================================
# Types
# ============================================================================

MessageCallback = Callable[[Any], None]


# ============================================================================
# Message Metadata
# ============================================================================

@dataclass(frozen=True)
class MessageMetadata:

    message_type: str

    received_at: float

    sequence: int


# ============================================================================
# Router Statistics
# ============================================================================

@dataclass
class RouterStatistics:
    """
    Message router istatistikleri.
    """

    total_messages: int = 0

    routed_messages: int = 0

    dropped_messages: int = 0

    callback_errors: int = 0

    unknown_messages: int = 0


# ============================================================================
# Subscription
# ============================================================================

@dataclass(frozen=True)
class Subscription:
    """Associates an event type with a callback."""

    subscription_id: int

    message_type: str

    callback: MessageCallback

    owner: Optional[str] = None


# ============================================================================
# Message Router
# ============================================================================

class MessageRouter:

    name = "MessageRouter"

    # Wildcard subscription.
    # "*" subscribes to all message types.
    WILDCARD = "*"

    def __init__(
        self,
        max_history: int = 100,
    ) -> None:

        if max_history < 0:

            raise ValueError(
                "max_history cannot be negative"
            )

        self.max_history = (
            int(max_history)
        )

        self._lock = (
            threading.RLock()
        )

        self._running = False

        self._next_subscription_id = 1

        self._subscriptions: DefaultDict[
            str,
            List[Subscription],
        ] = defaultdict(list)

        self._subscriptions_by_id: Dict[
            int,
            Subscription,
        ] = {}

        self._statistics = (
            RouterStatistics()
        )

        self._last_message_time: Dict[
            str,
            float,
        ] = {}

        self._message_count: DefaultDict[
            str,
            int,
        ] = defaultdict(int)

        self._sequence = 0

        self._history: List[
            MessageMetadata
        ] = []

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def start(self) -> bool:
        """Start the component or operation."""

        with self._lock:

            if self._running:

                return True

            self._running = True

        logger.info(
            "[MessageRouter] Started"
        )

        return True

    # =========================================================================
    # Stop
    # =========================================================================

    def stop(self) -> None:
        """Stop the component or operation safely."""

        with self._lock:

            self._running = False

        logger.info(
            "[MessageRouter] Stopped"
        )

    # =========================================================================
    # Subscribe
    # =========================================================================

    def subscribe(
        self,
        message_type: str,
        callback: MessageCallback,
        *,
        owner: Optional[str] = None,
    ) -> int:
        """Register a callback for an event type."""

        if not message_type:

            raise ValueError(
                "message_type cannot be empty"
            )

        if not callable(callback):

            raise TypeError(
                "callback must be callable"
            )

        normalized_type = (
            message_type.upper()
        )

        with self._lock:

            subscription_id = (
                self._next_subscription_id
            )

            self._next_subscription_id += 1

            subscription = Subscription(
                subscription_id=(
                    subscription_id
                ),

                message_type=(
                    normalized_type
                ),

                callback=callback,

                owner=owner,
            )

            self._subscriptions[
                normalized_type
            ].append(
                subscription
            )

            self._subscriptions_by_id[
                subscription_id
            ] = subscription

        logger.debug(
            "[MessageRouter] "
            "Subscription added "
            "id=%d type=%s owner=%s",
            subscription_id,
            normalized_type,
            owner,
        )

        return subscription_id

    # =========================================================================
    # Subscribe Many
    # =========================================================================

    def subscribe_many(
        self,
        message_types: List[str],
        callback: MessageCallback,
        *,
        owner: Optional[str] = None,
    ) -> List[int]:
        """
        Register one callback for multiple message types.
        """

        if not message_types:

            return []

        subscription_ids = []

        for message_type in message_types:

            subscription_ids.append(
                self.subscribe(
                    message_type,
                    callback,
                    owner=owner,
                )
            )

        return subscription_ids

    # =========================================================================
    # Unsubscribe
    # =========================================================================

    def unsubscribe(
        self,
        subscription_id: int,
    ) -> bool:
        """Remove a callback subscription."""

        with self._lock:

            subscription = (
                self._subscriptions_by_id.pop(
                    subscription_id,
                    None,
                )
            )

            if subscription is None:

                return False

            subscriptions = (
                self._subscriptions.get(
                    subscription.message_type,
                    [],
                )
            )

            try:

                subscriptions.remove(
                    subscription
                )

            except ValueError:

                pass

            if not subscriptions:

                self._subscriptions.pop(
                    subscription.message_type,
                    None,
                )

        logger.debug(
            "[MessageRouter] "
            "Subscription removed id=%d",
            subscription_id,
        )

        return True

    # =========================================================================
    # Unsubscribe Owner
    # =========================================================================

    def unsubscribe_owner(
        self,
        owner: str,
    ) -> int:

        if not owner:

            return 0

        with self._lock:

            subscription_ids = [
                subscription.subscription_id
                for subscription
                in self._subscriptions_by_id.values()
                if subscription.owner == owner
            ]

        removed = 0

        for subscription_id in subscription_ids:

            if self.unsubscribe(
                subscription_id
            ):

                removed += 1

        return removed

    # =========================================================================
    # Unsubscribe All
    # =========================================================================

    def clear_subscriptions(self) -> None:

        with self._lock:

            self._subscriptions.clear()

            self._subscriptions_by_id.clear()

        logger.info(
            "[MessageRouter] "
            "All subscriptions cleared"
        )

    # =========================================================================
    # Route
    # =========================================================================

    def route(
        self,
        message: Any,
    ) -> int:

        if message is None:

            with self._lock:

                self._statistics.dropped_messages += 1

            return 0

        # ---------------------------------------------------------------------
        # Message type
        # ---------------------------------------------------------------------

        try:

            message_type = (
                message.get_type()
            )

        except Exception:

            logger.exception(
                "[MessageRouter] "
                "Unable to determine message type"
            )

            with self._lock:

                self._statistics.dropped_messages += 1

            return 0

        if not message_type:

            with self._lock:

                self._statistics.dropped_messages += 1

            return 0

        message_type = (
            message_type.upper()
        )

        now = time.monotonic()

        # ---------------------------------------------------------------------
        # Update metadata
        # ---------------------------------------------------------------------

        with self._lock:

            self._sequence += 1

            sequence = self._sequence

            self._statistics.total_messages += 1

            self._last_message_time[
                message_type
            ] = now

            self._message_count[
                message_type
            ] += 1

            if self.max_history > 0:

                self._history.append(
                    MessageMetadata(
                        message_type=(
                            message_type
                        ),

                        received_at=now,

                        sequence=sequence,
                    )
                )

                if len(
                    self._history
                ) > self.max_history:

                    self._history.pop(
                        0
                    )

            callbacks = list(
                self._subscriptions.get(
                    message_type,
                    [],
                )
            )

            wildcard_callbacks = list(
                self._subscriptions.get(
                    self.WILDCARD,
                    [],
                )
            )

        # ---------------------------------------------------------------------
        # No subscriber
        # ---------------------------------------------------------------------

        if not callbacks and not wildcard_callbacks:

            with self._lock:

                self._statistics.unknown_messages += 1

            return 0

        # ---------------------------------------------------------------------
        # Dispatch
        # ---------------------------------------------------------------------

        dispatched = 0

        all_subscriptions = (
            callbacks
            + wildcard_callbacks
        )

        for subscription in all_subscriptions:

            try:

                subscription.callback(
                    message
                )

                dispatched += 1

            except Exception:

                with self._lock:

                    self._statistics.callback_errors += 1

                logger.exception(
                    "[MessageRouter] "
                    "Subscriber callback failed "
                    "type=%s owner=%s id=%d",
                    message_type,
                    subscription.owner,
                    subscription.subscription_id,
                )

        with self._lock:

            self._statistics.routed_messages += (
                dispatched
            )

        return dispatched

    # =========================================================================
    # Publish Alias
    # =========================================================================

    def publish(
        self,
        message: Any,
    ) -> int:
        """Publish an event to the event bus."""

        return self.route(
            message
        )

    # =========================================================================
    # Message Count
    # =========================================================================

    def message_count(
        self,
        message_type: str,
    ) -> int:

        if not message_type:

            return 0

        with self._lock:

            return self._message_count.get(
                message_type.upper(),
                0,
            )

    # =========================================================================
    # Last Message Time
    # =========================================================================

    def last_message_time(
        self,
        message_type: str,
    ) -> Optional[float]:

        if not message_type:

            return None

        with self._lock:

            return self._last_message_time.get(
                message_type.upper()
            )

    # =========================================================================
    # Message Age
    # =========================================================================

    def message_age(
        self,
        message_type: str,
    ) -> Optional[float]:

        timestamp = (
            self.last_message_time(
                message_type
            )
        )

        if timestamp is None:

            return None

        return max(
            0.0,
            time.monotonic()
            - timestamp,
        )

    # =========================================================================
    # Has Received
    # =========================================================================

    def has_received(
        self,
        message_type: str,
    ) -> bool:

        return (
            self.message_count(
                message_type
            )
            > 0
        )

    # =========================================================================
    # Statistics
    # =========================================================================

    def statistics(
        self,
    ) -> RouterStatistics:
        """
        Router statistics snapshot.
        """

        with self._lock:

            return RouterStatistics(
                total_messages=(
                    self._statistics
                    .total_messages
                ),

                routed_messages=(
                    self._statistics
                    .routed_messages
                ),

                dropped_messages=(
                    self._statistics
                    .dropped_messages
                ),

                callback_errors=(
                    self._statistics
                    .callback_errors
                ),

                unknown_messages=(
                    self._statistics
                    .unknown_messages
                ),
            )

    # =========================================================================
    # Subscription Count
    # =========================================================================

    def subscription_count(
        self,
        message_type: Optional[str] = None,
    ) -> int:

        with self._lock:

            if message_type is None:

                return len(
                    self._subscriptions_by_id
                )

            return len(
                self._subscriptions.get(
                    message_type.upper(),
                    [],
                )
            )

    # =========================================================================
    # History
    # =========================================================================

    def history(
        self,
    ) -> List[MessageMetadata]:
        """Return recorded event history."""

        with self._lock:

            return list(
                self._history
            )

    # =========================================================================
    # Reset Statistics
    # =========================================================================

    def reset_statistics(
        self,
    ) -> None:

        with self._lock:

            self._statistics = (
                RouterStatistics()
            )

            self._message_count.clear()

            self._last_message_time.clear()

        logger.debug(
            "[MessageRouter] "
            "Statistics reset"
        )

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
    # Subscription IDs
    # =========================================================================

    def subscription_ids(
        self,
    ) -> List[int]:

        with self._lock:

            return list(
                self._subscriptions_by_id.keys()
            )

    # =========================================================================
    # Representation
    # =========================================================================

    def __repr__(
        self,
    ) -> str:

        with self._lock:

            return (
                "MessageRouter("
                f"running={self._running}, "
                f"subscriptions="
                f"{len(self._subscriptions_by_id)}, "
                f"messages="
                f"{self._statistics.total_messages}"
                ")"
            )
