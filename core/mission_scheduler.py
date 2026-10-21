"""Schedules missions based on priority, status, and required resources."""

from __future__ import annotations

import heapq
import logging
import threading
import time

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set


logger = logging.getLogger(__name__)


# ============================================================================
# Scheduler state
# ============================================================================

class SchedulerState(str, Enum):
    """Mission scheduler lifecycle state."""

    STOPPED = "stopped"

    RUNNING = "running"

    PAUSED = "paused"

    ERROR = "error"


# ============================================================================
# Mission request state
# ============================================================================

class MissionRequestState(str, Enum):
    """State of a scheduled mission request."""

    QUEUED = "queued"

    STARTING = "starting"

    RUNNING = "running"

    COMPLETED = "completed"

    CANCELLED = "cancelled"

    FAILED = "failed"

    PREEMPTED = "preempted"


# ============================================================================
# Mission request
# ============================================================================

@dataclass
class MissionRequest:
    """
    One mission scheduling request.

    Higher priority missions are scheduled before lower priority missions.
    """

    request_id: int

    mission_name: str

    priority: int

    exclusive: bool = False

    resources: Set[str] = field(
        default_factory=set
    )

    created_at: float = field(
        default_factory=time.monotonic
    )

    state: MissionRequestState = (
        MissionRequestState.QUEUED
    )

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    sequence: int = 0


# ============================================================================
# Scheduler configuration
# ============================================================================

@dataclass(frozen=True)
class MissionSchedulerConfig:
    """
    Mission scheduler configuration.
    """

    loop_interval_s: float = 0.10

    maximum_queue_size: int = 50

    allow_preemption: bool = True

    preemption_priority_difference: int = 20

    auto_start: bool = True

    def __post_init__(self) -> None:

        if self.loop_interval_s <= 0:

            raise ValueError(
                "loop_interval_s must be greater than zero"
            )

        if self.maximum_queue_size <= 0:

            raise ValueError(
                "maximum_queue_size must be positive"
            )

        if self.preemption_priority_difference < 0:

            raise ValueError(
                "preemption_priority_difference "
                "cannot be negative"
            )


# ============================================================================
# Scheduler statistics
# ============================================================================

@dataclass
class MissionSchedulerStatistics:
    """Runtime scheduler statistics."""

    requests_received: int = 0

    requests_started: int = 0

    requests_completed: int = 0

    requests_cancelled: int = 0

    requests_failed: int = 0

    preemptions: int = 0

    rejected_requests: int = 0

    last_start_time: Optional[float] = None

    last_completion_time: Optional[float] = None


# ============================================================================
# Mission Scheduler
# ============================================================================

class MissionScheduler:

    name = "MissionScheduler"

    def __init__(
        self,
        mission_manager: Any,
        config: Optional[
            MissionSchedulerConfig
        ] = None,
        shared_data: Any = None,
    ) -> None:

        if mission_manager is None:

            raise ValueError(
                "mission_manager cannot be None"
            )

        self._mission_manager = (
            mission_manager
        )

        self._shared = shared_data

        self.config = (
            config
            or MissionSchedulerConfig()
        )

        self._lock = threading.RLock()

        self._condition = (
            threading.Condition(
                self._lock
            )
        )

        self._state = (
            SchedulerState.STOPPED
        )

        self._running = False

        self._stop_event = (
            threading.Event()
        )

        self._thread: Optional[
            threading.Thread
        ] = None

        self._next_request_id = 1

        self._sequence = 0

        self._queue: List[
            tuple
        ] = []

        self._requests: Dict[
            int,
            MissionRequest,
        ] = {}

        self._active_requests: Dict[
            str,
            MissionRequest,
        ] = {}

        self._statistics = (
            MissionSchedulerStatistics()
        )

        # Mission resource locks.
        self._resource_owners: Dict[
            str,
            str,
        ] = {}

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def start(self) -> None:
        """Start scheduler."""

        with self._condition:

            if self._running:
                return

            self._running = True

            self._state = (
                SchedulerState.RUNNING
            )

            self._stop_event.clear()

            self._thread = threading.Thread(
                target=self._run,
                name="MissionScheduler",
                daemon=True,
            )

            self._thread.start()

            self._condition.notify_all()

        logger.info(
            "[MissionScheduler] Started"
        )

    # -------------------------------------------------------------------------

    def stop(self) -> None:
        """Stop scheduler."""

        with self._condition:

            if not self._running:
                return

            self._running = False

            self._state = (
                SchedulerState.STOPPED
            )

            self._stop_event.set()

            self._condition.notify_all()

            thread = self._thread

        if thread is not None:

            thread.join(
                timeout=3.0
            )

        logger.info(
            "[MissionScheduler] Stopped"
        )

    # =========================================================================
    # Schedule
    # =========================================================================

    def schedule(
        self,
        mission_name: str,
        priority: int,
        exclusive: bool = False,
        resources: Optional[
            Set[str]
        ] = None,
        metadata: Optional[
            Dict[str, Any]
        ] = None,
    ) -> Optional[int]:
        """
        Add a mission to scheduler queue.

        Returns request ID.
        """

        with self._condition:

            if len(self._queue) >= (
                self.config.maximum_queue_size
            ):

                self._statistics.rejected_requests += 1

                logger.warning(
                    "[MissionScheduler] "
                    "Queue is full"
                )

                return None

            if not self._mission_exists(
                mission_name
            ):

                self._statistics.rejected_requests += 1

                logger.error(
                    "[MissionScheduler] "
                    "Mission does not exist: %s",
                    mission_name,
                )

                return None

            request_id = (
                self._next_request_id
            )

            self._next_request_id += 1

            sequence = self._sequence

            self._sequence += 1

            request = MissionRequest(
                request_id=request_id,

                mission_name=mission_name,

                priority=priority,

                exclusive=exclusive,

                resources=set(
                    resources
                    or set()
                ),

                metadata=dict(
                    metadata
                    or {}
                ),

                sequence=sequence,
            )

            self._requests[
                request_id
            ] = request

            # heapq is min-heap.
            # Negative priority gives highest priority first.
            heapq.heappush(
                self._queue,
                (
                    -priority,
                    sequence,
                    request_id,
                ),
            )

            self._statistics.requests_received += 1

            self._condition.notify_all()

        logger.info(
            "[MissionScheduler] "
            "Mission queued: %s "
            "(priority=%d, request=%d)",
            mission_name,
            priority,
            request_id,
        )

        return request_id

    # =========================================================================
    # Convenience scheduling
    # =========================================================================

    def schedule_emergency(
        self,
        mission_name: str,
        resources: Optional[
            Set[str]
        ] = None,
        metadata: Optional[
            Dict[str, Any]
        ] = None,
    ) -> Optional[int]:
        """Schedule highest-priority emergency mission."""

        return self.schedule(
            mission_name=mission_name,

            priority=100,

            exclusive=True,

            resources=resources,

            metadata=metadata,
        )

    # -------------------------------------------------------------------------

    def schedule_payload(
        self,
        mission_name: str,
        resources: Optional[
            Set[str]
        ] = None,
        metadata: Optional[
            Dict[str, Any]
        ] = None,
    ) -> Optional[int]:
        """Schedule payload mission."""

        return self.schedule(
            mission_name=mission_name,

            priority=90,

            exclusive=True,

            resources=resources,

            metadata=metadata,
        )

    # -------------------------------------------------------------------------

    def schedule_object_detection(
        self,
        mission_name: str,
        resources: Optional[
            Set[str]
        ] = None,
        metadata: Optional[
            Dict[str, Any]
        ] = None,
    ) -> Optional[int]:
        """Schedule object detection mission."""

        return self.schedule(
            mission_name=mission_name,

            priority=80,

            exclusive=False,

            resources=resources,

            metadata=metadata,
        )

    # -------------------------------------------------------------------------

    def schedule_mapping(
        self,
        mission_name: str,
        resources: Optional[
            Set[str]
        ] = None,
        metadata: Optional[
            Dict[str, Any]
        ] = None,
    ) -> Optional[int]:
        """Schedule mapping mission."""

        return self.schedule(
            mission_name=mission_name,

            priority=40,

            exclusive=False,

            resources=resources,

            metadata=metadata,
        )

    # -------------------------------------------------------------------------

    def schedule_lap(
        self,
        mission_name: str,
        resources: Optional[
            Set[str]
        ] = None,
        metadata: Optional[
            Dict[str, Any]
        ] = None,
    ) -> Optional[int]:
        """Schedule lap mission."""

        return self.schedule(
            mission_name=mission_name,

            priority=20,

            exclusive=False,

            resources=resources,

            metadata=metadata,
        )

    # =========================================================================
    # Main loop
    # =========================================================================

    def _run(self) -> None:
        """Scheduler worker loop."""

        while not self._stop_event.wait(
            self.config.loop_interval_s
        ):

            try:

                self._process_queue()

                self._update_active_requests()

            except Exception:

                logger.exception(
                    "[MissionScheduler] "
                    "Scheduler cycle failed"
                )

                with self._lock:

                    self._state = (
                        SchedulerState.ERROR
                    )

    # =========================================================================
    # Queue processing
    # =========================================================================

    def _process_queue(self) -> None:
        """Try to start queued missions."""

        while True:

            with self._lock:

                if not self._queue:
                    return

                request = (
                    self._peek_next_request()
                )

                if request is None:
                    return

                if not self._can_start(
                    request
                ):

                    # Highest priority request currently
                    # cannot start. Do not skip it blindly.
                    return

                self._pop_request()

                request.state = (
                    MissionRequestState.STARTING
                )

            if self._start_request(
                request
            ):

                with self._lock:

                    request.state = (
                        MissionRequestState.RUNNING
                    )

                    self._active_requests[
                        request.mission_name
                    ] = request

                    self._statistics.requests_started += 1

                    self._statistics.last_start_time = (
                        time.monotonic()
                    )

                    self._acquire_resources(
                        request
                    )

            else:

                with self._lock:

                    request.state = (
                        MissionRequestState.FAILED
                    )

                    self._statistics.requests_failed += 1

    # =========================================================================
    # Peek
    # =========================================================================

    def _peek_next_request(
        self,
    ) -> Optional[MissionRequest]:

        if not self._queue:
            return None

        _, _, request_id = (
            self._queue[0]
        )

        return self._requests.get(
            request_id
        )

    # =========================================================================
    # Pop
    # =========================================================================

    def _pop_request(self) -> Optional[
        MissionRequest
    ]:

        if not self._queue:
            return None

        _, _, request_id = (
            heapq.heappop(
                self._queue
            )
        )

        return self._requests.get(
            request_id
        )

    # =========================================================================
    # Start request
    # =========================================================================

    def _start_request(
        self,
        request: MissionRequest,
    ) -> bool:
        """Start mission through MissionManager."""

        try:

            success = (
                self._mission_manager
                .start_mission(
                    request.mission_name
                )
            )

            if not success:

                logger.error(
                    "[MissionScheduler] "
                    "MissionManager rejected mission %s",
                    request.mission_name,
                )

                return False

            logger.info(
                "[MissionScheduler] "
                "Started mission %s",
                request.mission_name,
            )

            return True

        except Exception:

            logger.exception(
                "[MissionScheduler] "
                "Failed to start mission %s",
                request.mission_name,
            )

            return False

    # =========================================================================
    # Can start
    # =========================================================================

    def _can_start(
        self,
        request: MissionRequest,
    ) -> bool:
        """
        Determine whether mission can start.

        Rules:

        1. Exclusive mission cannot run with another mission.
        2. Existing exclusive mission blocks all other missions.
        3. Resource conflicts block mission.
        """

        active = list(
            self._active_requests.values()
        )

        if not active:
            return True

        # ---------------------------------------------------------------
        # Existing exclusive mission
        # ---------------------------------------------------------------

        for active_request in active:

            if active_request.exclusive:

                return self._try_preempt(
                    request,
                    active_request,
                )

        # ---------------------------------------------------------------
        # New exclusive mission
        # ---------------------------------------------------------------

        if request.exclusive:

            for active_request in active:

                if (
                    request.priority
                    >= active_request.priority
                    + self.config
                    .preemption_priority_difference
                    and self.config
                    .allow_preemption
                ):

                    self._preempt(
                        active_request
                    )

                    continue

                return False

            return True

        # ---------------------------------------------------------------
        # Resource conflicts
        # ---------------------------------------------------------------

        for resource in request.resources:

            owner = (
                self._resource_owners.get(
                    resource
                )
            )

            if owner is None:
                continue

            active_request = (
                self._active_requests.get(
                    owner
                )
            )

            if active_request is None:
                continue

            if (
                request.priority
                >= active_request.priority
                + self.config
                .preemption_priority_difference
                and self.config
                .allow_preemption
            ):

                self._preempt(
                    active_request
                )

            else:

                return False

        return True

    # =========================================================================
    # Exclusive preemption
    # =========================================================================

    def _try_preempt(
        self,
        request: MissionRequest,
        active_request: MissionRequest,
    ) -> bool:
        """Determine whether active exclusive mission can be preempted."""

        if not self.config.allow_preemption:
            return False

        if (
            request.priority
            >= active_request.priority
            + self.config.preemption_priority_difference
        ):

            self._preempt(
                active_request
            )

            return True

        return False

    # =========================================================================
    # Preempt
    # =========================================================================

    def _preempt(
        self,
        request: MissionRequest,
    ) -> None:
        """Preempt an active mission."""

        logger.warning(
            "[MissionScheduler] "
            "Preempting mission %s",
            request.mission_name,
        )

        try:

            self._mission_manager.abort_mission(
                request.mission_name
            )

        except Exception:

            logger.exception(
                "[MissionScheduler] "
                "Failed to preempt mission %s",
                request.mission_name,
            )

        with self._lock:

            request.state = (
                MissionRequestState.PREEMPTED
            )

            self._release_resources(
                request
            )

            self._active_requests.pop(
                request.mission_name,
                None,
            )

            self._statistics.preemptions += 1

    # =========================================================================
    # Active request update
    # =========================================================================

    def _update_active_requests(
        self,
    ) -> None:
        """Detect missions that have completed externally."""

        with self._lock:

            active = list(
                self._active_requests.values()
            )

        for request in active:

            if self._mission_is_running(
                request.mission_name
            ):

                continue

            with self._lock:

                self._active_requests.pop(
                    request.mission_name,
                    None,
                )

                self._release_resources(
                    request
                )

                if request.state == (
                    MissionRequestState.RUNNING
                ):

                    request.state = (
                        MissionRequestState.COMPLETED
                    )

                    self._statistics.requests_completed += 1

                    self._statistics.last_completion_time = (
                        time.monotonic()
                    )

            logger.info(
                "[MissionScheduler] "
                "Mission completed: %s",
                request.mission_name,
            )

    # =========================================================================
    # Mission state
    # =========================================================================

    def _mission_is_running(
        self,
        mission_name: str,
    ) -> bool:
        """Check MissionManager mission state."""

        try:

            mission = (
                self._mission_manager
                .get(
                    mission_name
                )
            )

            if mission is None:
                return False

            return bool(
                mission.is_running()
            )

        except Exception:

            logger.exception(
                "[MissionScheduler] "
                "Failed to query mission state"
            )

            return False

    # =========================================================================
    # Resources
    # =========================================================================

    def _acquire_resources(
        self,
        request: MissionRequest,
    ) -> None:
        """Acquire mission resources."""

        for resource in request.resources:

            self._resource_owners[
                resource
            ] = request.mission_name

    # -------------------------------------------------------------------------

    def _release_resources(
        self,
        request: MissionRequest,
    ) -> None:
        """Release mission resources."""

        for resource in list(
            request.resources
        ):

            owner = (
                self._resource_owners.get(
                    resource
                )
            )

            if owner == request.mission_name:

                self._resource_owners.pop(
                    resource,
                    None,
                )

    # =========================================================================
    # Cancel
    # =========================================================================

    def cancel(
        self,
        request_id: int,
    ) -> bool:
        """
        Cancel queued or active request.
        """

        with self._lock:

            request = self._requests.get(
                request_id
            )

            if request is None:
                return False

            if request.state == (
                MissionRequestState.QUEUED
            ):

                request.state = (
                    MissionRequestState.CANCELLED
                )

                self._remove_from_queue(
                    request_id
                )

                self._statistics.requests_cancelled += 1

                return True

            if request.state == (
                MissionRequestState.RUNNING
            ):

                mission_name = (
                    request.mission_name
                )

            else:

                return False

        try:

            self._mission_manager.stop_mission(
                mission_name
            )

        except Exception:

            logger.exception(
                "[MissionScheduler] "
                "Failed to stop cancelled mission"
            )

            return False

        with self._lock:

            request.state = (
                MissionRequestState.CANCELLED
            )

            self._active_requests.pop(
                mission_name,
                None,
            )

            self._release_resources(
                request
            )

            self._statistics.requests_cancelled += 1

        return True

    # =========================================================================
    # Remove queue entry
    # =========================================================================

    def _remove_from_queue(
        self,
        request_id: int,
    ) -> None:
        """Remove request from heap queue."""

        self._queue = [
            item
            for item in self._queue
            if item[2] != request_id
        ]

        heapq.heapify(
            self._queue
        )

    # =========================================================================
    # Pause
    # =========================================================================

    def pause(self) -> None:
        """Pause scheduling."""

        with self._lock:

            self._state = (
                SchedulerState.PAUSED
            )

    # -------------------------------------------------------------------------

    def resume(self) -> None:
        """Resume scheduling."""

        with self._condition:

            self._state = (
                SchedulerState.RUNNING
            )

            self._condition.notify_all()

    # =========================================================================
    # Queue inspection
    # =========================================================================

    def queued_requests(
        self,
    ) -> List[MissionRequest]:
        """Return queued requests ordered by priority."""

        with self._lock:

            requests = []

            for _, _, request_id in (
                self._queue
            ):

                request = self._requests.get(
                    request_id
                )

                if request is not None:

                    requests.append(
                        request
                    )

            requests.sort(
                key=lambda item: (
                    -item.priority,
                    item.sequence,
                )
            )

            return requests

    # -------------------------------------------------------------------------

    def active_requests(
        self,
    ) -> List[MissionRequest]:
        """Return active mission requests."""

        with self._lock:

            return list(
                self._active_requests.values()
            )

    # -------------------------------------------------------------------------

    def get_request(
        self,
        request_id: int,
    ) -> Optional[MissionRequest]:

        with self._lock:

            return self._requests.get(
                request_id
            )

    # =========================================================================
    # Mission existence
    # =========================================================================

    def _mission_exists(
        self,
        mission_name: str,
    ) -> bool:
        """Check mission registration."""

        try:

            mission = (
                self._mission_manager
                .get(
                    mission_name
                )
            )

            return mission is not None

        except Exception:

            return False

    # =========================================================================
    # Statistics
    # =========================================================================

    def statistics(
        self,
    ) -> MissionSchedulerStatistics:
        """Return statistics snapshot."""

        with self._lock:

            return MissionSchedulerStatistics(
                requests_received=(
                    self._statistics
                    .requests_received
                ),

                requests_started=(
                    self._statistics
                    .requests_started
                ),

                requests_completed=(
                    self._statistics
                    .requests_completed
                ),

                requests_cancelled=(
                    self._statistics
                    .requests_cancelled
                ),

                requests_failed=(
                    self._statistics
                    .requests_failed
                ),

                preemptions=(
                    self._statistics
                    .preemptions
                ),

                rejected_requests=(
                    self._statistics
                    .rejected_requests
                ),

                last_start_time=(
                    self._statistics
                    .last_start_time
                ),

                last_completion_time=(
                    self._statistics
                    .last_completion_time
                ),
            )

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def state(
        self,
    ) -> SchedulerState:

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
    def queue_size(
        self,
    ) -> int:

        with self._lock:
            return len(self._queue)

    # =========================================================================
    # Representation
    # =========================================================================

    def __repr__(self) -> str:

        return (
            "MissionScheduler("
            f"state={self.state.value!r}, "
            f"running={self.running}, "
            f"queue={self.queue_size}, "
            f"active="
            f"{len(self.active_requests())}"
            ")"
        )
