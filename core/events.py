"""Event types and a thread-safe event bus for communication between mission subsystems."""

from __future__ import annotations

from enum import Enum, auto
from queue import Queue, Empty
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
import threading
import time


# ============================================================
# EVENT TYPES
# ============================================================

class EventType(Enum):
    """Event identifiers used for inter-subsystem communication."""

    # --------------------------------------------------------
    # Mission
    # --------------------------------------------------------

    MISSION_START = auto()
    MISSION_STOP = auto()
    MISSION_PAUSE = auto()
    MISSION_RESUME = auto()
    MISSION_FINISHED = auto()
    MISSION_ABORT = auto()
    MISSION_FAILED = auto()

    # --------------------------------------------------------
    # Navigation
    # --------------------------------------------------------

    WAYPOINT_REACHED = auto()
    WAYPOINT_TIMEOUT = auto()

    GOTO_FINISHED = auto()
    GOTO_FAILED = auto()

    # --------------------------------------------------------
    # AI / Object Detection
    # --------------------------------------------------------

    TARGET_FOUND = auto()
    TARGET_LOST = auto()
    TARGET_LOCKED = auto()
    TARGET_DETECTED = auto()

    # --------------------------------------------------------
    # Mapping / Camera
    # --------------------------------------------------------

    PHOTO_CAPTURED = auto()
    MAP_FINISHED = auto()

    CAMERA_CAPTURE_REQUEST = auto()
    CAMERA_CAPTURED = auto()
    CAMERA_CAPTURE_FAILED = auto()

    # --------------------------------------------------------
    # Payload
    # --------------------------------------------------------

    PAYLOAD_RELEASED = auto()
    PAYLOAD_RELEASE_REQUEST = auto()
    PAYLOAD_RELEASE_FAILED = auto()

    # --------------------------------------------------------
    # Safety
    # --------------------------------------------------------

    LOW_BATTERY = auto()
    GPS_LOST = auto()
    CONNECTION_LOST = auto()
    EMERGENCY = auto()
    BATTERY_CRITICAL = auto()
    GPS_RECOVERED = auto()
    HEARTBEAT_TIMEOUT = auto()
    EKF_ERROR = auto()
    SYSTEM_HEALTH_CHANGED = auto()

    # --------------------------------------------------------
    # System Shutdown
    # --------------------------------------------------------

    # Shut down the system after five failed ARM attempts.
    ARM_MAX_ATTEMPTS_EXCEEDED = auto()

    # Trigger an emergency shutdown when RC16 PWM exceeds 1800.
    RC16_SAFETY_TRIGGERED = auto()


# ============================================================
# EVENT
# ============================================================

@dataclass()
class Event:
    """Data envelope passed through the event bus."""

    event: EventType

    data: Dict[str, Any] = field(default_factory=dict)

    timestamp: float = field(default_factory=time.time)

    source: Optional[str] = None


# ============================================================
# SUBSCRIPTION
# ============================================================

@dataclass()
class Subscription:
    """
    Event subscription bilgisi.
    """

    event_type: EventType
    callback: Callable[[Event], None]


# ============================================================
# EVENT BUS
# ============================================================

class EventBus:
    """Thread-safe event queue and publisher/subscriber interface."""

    def __init__(self) -> None:

        # ----------------------------------------------------
        # Ana event queue
        # ----------------------------------------------------

        self.queue: Queue[Event] = Queue()

        # ----------------------------------------------------
        # Subscriber listesi
        #
        # EventType
        #       |
        #       +--> callback
        #       +--> callback
        #       +--> callback
        # ----------------------------------------------------

        self._subscribers: Dict[
            EventType,
            List[Callable[[Event], None]]
        ] = {}

        # ----------------------------------------------------
        # Lock protecting the subscriber dictionary.
        # ----------------------------------------------------

        self._lock = threading.RLock()

    # ========================================================
    # PUBLISH
    # ========================================================

    def publish(
        self,
        event: EventType,
        data: Optional[Dict[str, Any]] = None,
        source: Optional[str] = None,
    ) -> Event:
        """Publish an event to the event bus."""

        if data is None:
            data = {}

        event_obj = Event(
            event=event,
            data=data,
            source=source,
        )

        # Add the event to the central queue.
        self.queue.put(event_obj)

        return event_obj

    # ========================================================
    # SUBSCRIBE
    # ========================================================

    def subscribe(
        self,
        event_type: EventType,
        callback: Callable[[Event], None],
    ) -> None:
        """Register a callback for an event type."""

        if not isinstance(event_type, EventType):
            raise TypeError(
                "event_type must be an EventType"
            )

        if not callable(callback):
            raise TypeError(
                "callback must be callable"
            )

        with self._lock:

            subscribers = self._subscribers.setdefault(
                event_type,
                []
            )

            # The same callback must not be registered twice.
            # register edilmesini engelle.
            if callback not in subscribers:

                subscribers.append(callback)

    # ========================================================
    # UNSUBSCRIBE
    # ========================================================

    def unsubscribe(
        self,
        event_type: EventType,
        callback: Callable[[Event], None],
    ) -> None:
        """Remove a callback subscription."""

        with self._lock:

            subscribers = self._subscribers.get(
                event_type
            )

            if not subscribers:
                return

            try:
                subscribers.remove(callback)

            except ValueError:
                return

            # Remove the entry from the dictionary if the list becomes empty.
            if not subscribers:

                self._subscribers.pop(
                    event_type,
                    None
                )

    # ========================================================
    # GET
    # ========================================================

    def get(
        self,
        timeout: float = 0.1,
    ) -> Optional[Event]:

        try:

            return self.queue.get(
                timeout=timeout
            )

        except Empty:

            return None

    # ========================================================
    # DISPATCH
    # ========================================================

    def dispatch(
        self,
        event: Event,
    ) -> None:
        """Dispatch an event to matching subscribers."""

        with self._lock:

            callbacks = list(
                self._subscribers.get(
                    event.event,
                    []
                )
            )

        for callback in callbacks:

            try:

                callback(event)

            except Exception:

                # A subscriber failure
                # must not affect other subscribers.
                #
                # Exceptions are not propagated from this point.
                # Use the project logger for this error in production.

                import logging

                logging.getLogger(
                    __name__
                ).exception(
                    "Event handler failed: %s",
                    event.event.name,
                )

    # ========================================================
    # PROCESS ONE
    # ========================================================

    def process_one(
        self,
        timeout: float = 0.1,
    ) -> bool:

        event = self.get(
            timeout=timeout
        )

        if event is None:

            return False

        self.dispatch(
            event
        )

        return True

    # ========================================================
    # PROCESS ALL
    # ========================================================

    def process_all(self) -> int:

        processed = 0

        while True:

            try:

                event = self.queue.get_nowait()

            except Empty:

                break

            self.dispatch(
                event
            )

            processed += 1

        return processed

    # ========================================================
    # EMPTY
    # ========================================================

    def empty(self) -> bool:

        return self.queue.empty()

    # ========================================================
    # CLEAR
    # ========================================================

    def clear(self) -> None:

        while True:

            try:

                self.queue.get_nowait()

            except Empty:

                break

    # ========================================================
    # CLEAR SUBSCRIBERS
    # ========================================================

    def clear_subscribers(self) -> None:

        with self._lock:

            self._subscribers.clear()

    # ========================================================
    # SUBSCRIBER COUNT
    # ========================================================

    def subscriber_count(
        self,
        event_type: Optional[EventType] = None,
    ) -> int:

        with self._lock:

            if event_type is not None:

                return len(
                    self._subscribers.get(
                        event_type,
                        []
                    )
                )

            return sum(
                len(callbacks)
                for callbacks
                in self._subscribers.values()
            )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    bus = EventBus()

    # --------------------------------------------------------
    # Handler
    # --------------------------------------------------------

    def on_waypoint_reached(event: Event):

        print(
            "[EVENT]",
            event.event.name,
            event.data,
            event.source,
            event.timestamp,
        )

    # --------------------------------------------------------
    # Subscribe
    # --------------------------------------------------------

    bus.subscribe(
        EventType.WAYPOINT_REACHED,
        on_waypoint_reached,
    )

    # --------------------------------------------------------
    # Publish
    # --------------------------------------------------------

    bus.publish(
        EventType.WAYPOINT_REACHED,
        {
            "waypoint_index": 5,
            "latitude": 37.123,
            "longitude": 32.456,
        },
        source="Navigator",
    )

    # --------------------------------------------------------
    # Process
    # --------------------------------------------------------

    bus.process_all()
