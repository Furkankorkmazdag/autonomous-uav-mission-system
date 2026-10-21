"""Thread-safe dispatcher that routes published events to registered subscribers."""

from __future__ import annotations

import logging
import threading
import time

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Callable, Deque, Dict, List, Optional


logger = logging.getLogger(__name__)


# ============================================================================
# Types
# ============================================================================

EventCallback = Callable[
    [Any, Any],
    None,
]


# ============================================================================
# Event Record
# ============================================================================

@dataclass(frozen=True)
class EventRecord:

    event_type: Any

    payload: Any

    timestamp: float


# ============================================================================
# Subscription
# ============================================================================

@dataclass(frozen=True)
class Subscription:
    """
    Event subscription bilgisi.
    """

    subscription_id: int

    event_type: Any

    callback: EventCallback


# ============================================================================
# Configuration
# ============================================================================

@dataclass(frozen=True)
class EventDispatcherConfig:
    """
    Event dispatcher configuration.
    """

    history_size: int = 1000

    isolate_callback_errors: bool = True

    wildcard_enabled: bool = True

    def __post_init__(self) -> None:

        if self.history_size < 0:

            raise ValueError(
                "history_size cannot be negative"
            )


# ============================================================================
# Event Dispatcher
# ============================================================================

class EventDispatcher:

    name = "EventDispatcher"

    WILDCARD = "*"

    def __init__(
        self,
        config: Optional[
            EventDispatcherConfig
        ] = None,
    ) -> None:

        self.config = (
            config
            or EventDispatcherConfig()
        )

        self._lock = threading.RLock()

        self._subscriptions: Dict[
            Any,
            Dict[
                int,
                Subscription,
            ],
        ] = defaultdict(dict)

        self._subscription_index: Dict[
            int,
            Subscription,
        ] = {}

        self._next_subscription_id = 1

        self._history: Deque[
            EventRecord
        ] = deque(
            maxlen=self.config.history_size
            if self.config.history_size > 0
            else None
        )

        self._published_count = 0

        self._failed_callbacks = 0

        self._running = False

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def start(self) -> None:
        """Start dispatcher."""

        with self._lock:

            self._running = True

        logger.info(
            "[EventDispatcher] Started"
        )

    # -------------------------------------------------------------------------

    def stop(self) -> None:
        """Stop dispatcher."""

        with self._lock:

            self._running = False

        logger.info(
            "[EventDispatcher] Stopped"
        )

    # =========================================================================
    # Subscribe
    # =========================================================================

    def subscribe(
        self,
        event_type: Any,
        callback: EventCallback,
    ) -> int:
        """
        Subscribe callback to event type.

        Returns subscription ID.
        """

        if event_type is None:

            raise ValueError(
                "event_type cannot be None"
            )

        if not callable(callback):

            raise TypeError(
                "callback must be callable"
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

                event_type=event_type,

                callback=callback,
            )

            self._subscriptions[
                event_type
            ][
                subscription_id
            ] = subscription

            self._subscription_index[
                subscription_id
            ] = subscription

        logger.debug(
            "[EventDispatcher] "
            "Subscribed %s to %r",
            subscription_id,
            event_type,
        )

        return subscription_id

    # =========================================================================
    # Subscribe wildcard
    # =========================================================================

    def subscribe_all(
        self,
        callback: EventCallback,
    ) -> int:
        """
        Subscribe callback to every event.
        """

        if not self.config.wildcard_enabled:

            raise RuntimeError(
                "Wildcard subscriptions are disabled"
            )

        return self.subscribe(
            self.WILDCARD,
            callback,
        )

    # =========================================================================
    # Unsubscribe
    # =========================================================================

    def unsubscribe(
        self,
        subscription_id: int,
    ) -> bool:
        """
        Remove subscription by ID.
        """

        with self._lock:

            subscription = (
                self._subscription_index.pop(
                    subscription_id,
                    None,
                )
            )

            if subscription is None:

                return False

            subscribers = (
                self._subscriptions.get(
                    subscription.event_type
                )
            )

            if subscribers is not None:

                subscribers.pop(
                    subscription_id,
                    None,
                )

                if not subscribers:

                    self._subscriptions.pop(
                        subscription.event_type,
                        None,
                    )

        return True

    # =========================================================================
    # Unsubscribe callback
    # =========================================================================

    def unsubscribe_callback(
        self,
        event_type: Any,
        callback: EventCallback,
    ) -> int:
        """
        Remove all subscriptions matching event_type + callback.

        Returns number of removed subscriptions.
        """

        removed = 0

        with self._lock:

            subscribers = (
                self._subscriptions.get(
                    event_type,
                    {},
                )
            )

            matching = [
                subscription_id
                for subscription_id, subscription
                in subscribers.items()
                if subscription.callback
                is callback
            ]

            for subscription_id in matching:

                self._subscription_index.pop(
                    subscription_id,
                    None,
                )

                subscribers.pop(
                    subscription_id,
                    None,
                )

                removed += 1

            if not subscribers:

                self._subscriptions.pop(
                    event_type,
                    None,
                )

        return removed

    # =========================================================================
    # Publish
    # =========================================================================

    def publish(
        self,
        event_type: Any,
        payload: Any = None,
    ) -> int:
        """
        Publish an event.

        Returns number of callbacks invoked.
        """

        if event_type is None:

            raise ValueError(
                "event_type cannot be None"
            )

        timestamp = time.monotonic()

        with self._lock:

            self._published_count += 1

            if self.config.history_size > 0:

                self._history.append(
                    EventRecord(
                        event_type=event_type,

                        payload=payload,

                        timestamp=timestamp,
                    )
                )

            callbacks = list(
                self._subscriptions.get(
                    event_type,
                    {},
                ).values()
            )

            if (
                self.config.wildcard_enabled
            ):

                callbacks.extend(
                    self._subscriptions.get(
                        self.WILDCARD,
                        {},
                    ).values()
                )

        invoked = 0

        for subscription in callbacks:

            try:

                subscription.callback(
                    event_type,
                    payload,
                )

                invoked += 1

            except Exception:

                with self._lock:

                    self._failed_callbacks += 1

                logger.exception(
                    "[EventDispatcher] "
                    "Callback failed for event %r",
                    event_type,
                )

                if not self.config.isolate_callback_errors:

                    raise

        return invoked

    # =========================================================================
    # Async publish
    # =========================================================================

    def publish_async(
        self,
        event_type: Any,
        payload: Any = None,
    ) -> threading.Thread:

        thread = threading.Thread(
            target=self.publish,
            args=(
                event_type,
                payload,
            ),
            name="EventDispatch",
            daemon=True,
        )

        thread.start()

        return thread

    # =========================================================================
    # Wait for event
    # =========================================================================

    def wait_for(
        self,
        event_type: Any,
        timeout_s: Optional[float] = None,
    ) -> Optional[EventRecord]:
        """Wait for the requested event or condition."""

        condition = threading.Condition()

        result: List[
            EventRecord
        ] = []

        def callback(
            received_type: Any,
            payload: Any,
        ) -> None:

            with condition:

                result.append(
                    EventRecord(
                        event_type=(
                            received_type
                        ),

                        payload=payload,

                        timestamp=(
                            time.monotonic()
                        ),
                    )
                )

                condition.notify()

        subscription_id = self.subscribe(
            event_type,
            callback,
        )

        try:

            with condition:

                if result:

                    return result[0]

                condition.wait(
                    timeout=timeout_s
                )

                if result:

                    return result[0]

                return None

        finally:

            self.unsubscribe(
                subscription_id
            )

    # =========================================================================
    # History
    # =========================================================================

    def history(
        self,
        event_type: Any = None,
    ) -> List[EventRecord]:
        """Return recorded event history."""

        with self._lock:

            records = list(
                self._history
            )

        if event_type is None:

            return records

        return [
            record
            for record in records
            if record.event_type
            == event_type
        ]

    # =========================================================================
    # Clear history
    # =========================================================================

    def clear_history(self) -> None:
        """Clear event history."""

        with self._lock:

            self._history.clear()

    # =========================================================================
    # Subscription inspection
    # =========================================================================

    def subscription_count(
        self,
        event_type: Any = None,
    ) -> int:
        """
        Return number of subscriptions.

        event_type None:
            all subscriptions.
        """

        with self._lock:

            if event_type is None:

                return len(
                    self._subscription_index
                )

            return len(
                self._subscriptions.get(
                    event_type,
                    {},
                )
            )

    # =========================================================================
    # Subscribers
    # =========================================================================

    def subscribers(
        self,
        event_type: Any,
    ) -> List[Subscription]:
        """
        Return subscriptions for event.
        """

        with self._lock:

            result = list(
                self._subscriptions.get(
                    event_type,
                    {},
                ).values()
            )

            if self.config.wildcard_enabled:

                result.extend(
                    self._subscriptions.get(
                        self.WILDCARD,
                        {},
                    ).values()
                )

            return result

    # =========================================================================
    # Statistics
    # =========================================================================

    def statistics(self) -> dict:
        """Return dispatcher statistics."""

        with self._lock:

            return {
                "running": (
                    self._running
                ),

                "published_count": (
                    self._published_count
                ),

                "failed_callbacks": (
                    self._failed_callbacks
                ),

                "subscription_count": (
                    len(
                        self._subscription_index
                    )
                ),

                "history_size": (
                    len(self._history)
                ),
            }

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
    # Clear subscriptions
    # =========================================================================

    def clear_subscriptions(self) -> None:
        """
        Remove all subscriptions.
        """

        with self._lock:

            self._subscriptions.clear()

            self._subscription_index.clear()

    # =========================================================================
    # Representation
    # =========================================================================

    def __repr__(
        self,
    ) -> str:

        with self._lock:

            return (
                "EventDispatcher("
                f"running={self._running}, "
                f"subscriptions="
                f"{len(self._subscription_index)}, "
                f"history="
                f"{len(self._history)}"
                ")"
            )
