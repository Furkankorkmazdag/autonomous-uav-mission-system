"""Coordinates payload-release requests and prevents duplicate release actions."""

from __future__ import annotations

import logging
import threading
import time

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from core.events import EventType


logger = logging.getLogger(__name__)


# ============================================================================
# Payload state
# ============================================================================

class PayloadState(str, Enum):
    """Payload lifecycle state."""

    UNKNOWN = "unknown"

    READY = "ready"

    BUSY = "busy"

    RELEASE_REQUESTED = "release_requested"

    RELEASING = "releasing"

    RELEASED = "released"

    FAILED = "failed"

    DISABLED = "disabled"


# ============================================================================
# Configuration
# ============================================================================

@dataclass(frozen=True)
class PayloadConfig:
    """
    Payload manager configuration.
    """

    enabled: bool = True

    release_cooldown_s: float = 2.0

    release_timeout_s: float = 5.0

    maximum_release_attempts: int = 1

    require_driver_ready: bool = True

    allow_repeat_release: bool = False

    auto_reset_after_failure: bool = True

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate configuration."""

        if self.release_cooldown_s < 0:
            raise ValueError(
                "release_cooldown_s cannot be negative"
            )

        if self.release_timeout_s <= 0:
            raise ValueError(
                "release_timeout_s must be greater than zero"
            )

        if self.maximum_release_attempts <= 0:
            raise ValueError(
                "maximum_release_attempts must be positive"
            )


# ============================================================================
# Payload result
# ============================================================================

@dataclass(frozen=True)
class PayloadReleaseResult:
    """
    Result of a payload release operation.
    """

    success: bool

    timestamp: float

    attempt: int

    reason: str = ""

    target_id: Optional[int] = None


# ============================================================================
# Statistics
# ============================================================================

@dataclass
class PayloadStatistics:
    """Runtime payload statistics."""

    requests: int = 0

    successful_releases: int = 0

    failed_releases: int = 0

    rejected_requests: int = 0

    release_attempts: int = 0

    last_request_time: Optional[float] = None

    last_release_time: Optional[float] = None

    last_failure_time: Optional[float] = None


# ============================================================================
# Payload Manager
# ============================================================================

class PayloadManager:
    """
    Central payload management layer.

    Payload hardware implementation is intentionally injected as
    ``driver``.

    Driver may implement:

        ready()
        release()
        reset()

    The manager does not assume a specific hardware implementation.
    """

    name = "PayloadManager"

    def __init__(
        self,
        event_bus: Any,
        driver: Any = None,
        config: Optional[PayloadConfig] = None,
        shared_data: Any = None,
    ) -> None:

        if event_bus is None:
            raise ValueError(
                "event_bus cannot be None"
            )

        self._event_bus = event_bus

        self._driver = driver

        self._shared = shared_data

        self.config = (
            config
            or PayloadConfig()
        )

        self._lock = threading.RLock()

        self._state = (
            PayloadState.READY
            if self.config.enabled
            else PayloadState.DISABLED
        )

        self._running = False

        self._release_in_progress = False

        self._release_attempts = 0

        self._last_release_time: Optional[
            float
        ] = None

        self._last_request_time: Optional[
            float
        ] = None

        self._current_target_id: Optional[
            int
        ] = None

        self._statistics = (
            PayloadStatistics()
        )

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def start(self) -> None:
        """Start payload manager."""

        with self._lock:

            if not self.config.enabled:

                self._state = (
                    PayloadState.DISABLED
                )

                return

            self._running = True

            self._state = (
                PayloadState.READY
            )

        logger.info(
            "[PayloadManager] Started"
        )

    # -------------------------------------------------------------------------

    def stop(self) -> None:
        """Stop payload manager."""

        with self._lock:

            self._running = False

            self._release_in_progress = False

            if self.config.enabled:

                self._state = (
                    PayloadState.READY
                )

        logger.info(
            "[PayloadManager] Stopped"
        )

    # =========================================================================
    # Request
    # =========================================================================

    def request_release(
        self,
        target_id: Optional[int] = None,
        metadata: Optional[dict] = None,
    ) -> bool:
        """
        Request a payload release.

        Returns True if request was accepted.
        """

        now = time.monotonic()

        with self._lock:

            self._statistics.requests += 1

            self._statistics.last_request_time = now

            if not self.config.enabled:

                self._statistics.rejected_requests += 1

                return False

            if not self._running:

                self._statistics.rejected_requests += 1

                logger.warning(
                    "[PayloadManager] "
                    "Release rejected: manager not running"
                )

                return False

            if self._release_in_progress:

                self._statistics.rejected_requests += 1

                return False

            if (
                not self.config.allow_repeat_release
                and self._release_attempts
                >= self.config.maximum_release_attempts
            ):

                self._statistics.rejected_requests += 1

                logger.warning(
                    "[PayloadManager] "
                    "Maximum release attempts reached"
                )

                return False

            if (
                self._last_release_time is not None
                and (
                    now
                    - self._last_release_time
                    < self.config.release_cooldown_s
                )
            ):

                self._statistics.rejected_requests += 1

                return False

            self._current_target_id = target_id

            self._state = (
                PayloadState.RELEASE_REQUESTED
            )

        if self.config.require_driver_ready:

            if not self._driver_ready():

                with self._lock:

                    self._statistics.rejected_requests += 1

                    self._state = (
                        PayloadState.FAILED
                    )

                self._publish(
                    EventType.PAYLOAD_RELEASE_FAILED,
                    {
                        "reason": "driver_not_ready",

                        "target_id": target_id,
                    },
                )

                return False

        self._publish(
            EventType.PAYLOAD_RELEASE_REQUEST,
            {
                "target_id": target_id,

                "metadata": (
                    metadata
                    or {}
                ),
            },
        )

        self._perform_release(
            target_id=target_id
        )

        return True

    # =========================================================================
    # Release
    # =========================================================================

    def _perform_release(
        self,
        target_id: Optional[int],
    ) -> None:
        """Perform release using injected payload driver."""

        with self._lock:

            self._release_in_progress = True

            self._release_attempts += 1

            self._statistics.release_attempts += 1

            self._state = (
                PayloadState.RELEASING
            )

            attempt = (
                self._release_attempts
            )

        try:

            if self._driver is None:

                raise RuntimeError(
                    "Payload driver is not configured"
                )

            release_method = getattr(
                self._driver,
                "release",
                None,
            )

            if release_method is None:

                raise RuntimeError(
                    "Payload driver does not implement release()"
                )

            result = release_method()

            # ---------------------------------------------------------------
            # Driver result handling
            # ---------------------------------------------------------------

            success = (
                True
                if result is None
                else bool(result)
            )

            if not success:

                raise RuntimeError(
                    "Payload driver returned failure"
                )

            self._handle_release_success(
                target_id=target_id,
                attempt=attempt,
            )

        except Exception as exc:

            self._handle_release_failure(
                target_id=target_id,
                attempt=attempt,
                reason=str(exc),
            )

    # =========================================================================
    # Success
    # =========================================================================

    def _handle_release_success(
        self,
        target_id: Optional[int],
        attempt: int,
    ) -> None:

        now = time.monotonic()

        with self._lock:

            self._release_in_progress = False

            self._last_release_time = now

            self._statistics.successful_releases += 1

            self._statistics.last_release_time = now

            self._state = (
                PayloadState.RELEASED
            )

        result = PayloadReleaseResult(
            success=True,

            timestamp=now,

            attempt=attempt,

            reason="released",

            target_id=target_id,
        )

        self._publish(
            EventType.PAYLOAD_RELEASED,
            {
                "target_id": target_id,

                "attempt": attempt,

                "timestamp": now,

                "success": True,

                "reason": result.reason,
            },
        )

        logger.info(
            "[PayloadManager] "
            "Payload released successfully "
            "(target=%s, attempt=%d)",
            target_id,
            attempt,
        )

    # =========================================================================
    # Failure
    # =========================================================================

    def _handle_release_failure(
        self,
        target_id: Optional[int],
        attempt: int,
        reason: str,
    ) -> None:

        now = time.monotonic()

        with self._lock:

            self._release_in_progress = False

            self._statistics.failed_releases += 1

            self._statistics.last_failure_time = now

            self._state = (
                PayloadState.FAILED
            )

        self._publish(
            EventType.PAYLOAD_RELEASE_FAILED,
            {
                "target_id": target_id,

                "attempt": attempt,

                "timestamp": now,

                "success": False,

                "reason": reason,
            },
        )

        logger.error(
            "[PayloadManager] "
            "Payload release failed: %s",
            reason,
        )

        if self.config.auto_reset_after_failure:

            self.reset()

    # =========================================================================
    # Driver readiness
    # =========================================================================

    def _driver_ready(self) -> bool:
        """Check whether payload driver is ready."""

        if self._driver is None:

            return False

        ready_method = getattr(
            self._driver,
            "ready",
            None,
        )

        if ready_method is None:

            # If driver does not expose readiness,
            # assume it is available.
            return True

        try:

            return bool(
                ready_method()
            )

        except Exception:

            logger.exception(
                "[PayloadManager] "
                "Driver ready() failed"
            )

            return False

    # =========================================================================
    # Reset
    # =========================================================================

    def reset(self) -> bool:
        """
        Reset payload hardware if driver supports it.
        """

        if self._driver is None:

            with self._lock:

                if self.config.enabled:

                    self._state = (
                        PayloadState.READY
                    )

            return False

        reset_method = getattr(
            self._driver,
            "reset",
            None,
        )

        if reset_method is None:

            with self._lock:

                self._state = (
                    PayloadState.READY
                    if self.config.enabled
                    else PayloadState.DISABLED
                )

            return False

        try:

            result = reset_method()

            with self._lock:

                self._state = (
                    PayloadState.READY
                    if self.config.enabled
                    else PayloadState.DISABLED
                )

            return (
                True
                if result is None
                else bool(result)
            )

        except Exception:

            logger.exception(
                "[PayloadManager] "
                "Driver reset failed"
            )

            return False

    # =========================================================================
    # Event handling
    # =========================================================================

    def handle_event(
        self,
        event_type: EventType,
        payload: Optional[dict] = None,
    ) -> None:

        payload = payload or {}

        if event_type == (
            EventType.PAYLOAD_RELEASE_REQUEST
        ):

            target_id = payload.get(
                "target_id"
            )

            metadata = dict(
                payload
            )

            metadata.pop(
                "target_id",
                None,
            )

            self.request_release(
                target_id=target_id,
                metadata=metadata,
            )

    # =========================================================================
    # Status
    # =========================================================================

    @property
    def state(
        self,
    ) -> PayloadState:

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
    def release_in_progress(
        self,
    ) -> bool:

        with self._lock:
            return self._release_in_progress

    # -------------------------------------------------------------------------

    @property
    def release_attempts(
        self,
    ) -> int:

        with self._lock:
            return self._release_attempts

    # =========================================================================
    # Statistics
    # =========================================================================

    def statistics(
        self,
    ) -> PayloadStatistics:
        """Return immutable-style statistics snapshot."""

        with self._lock:

            return PayloadStatistics(
                requests=(
                    self._statistics.requests
                ),

                successful_releases=(
                    self._statistics
                    .successful_releases
                ),

                failed_releases=(
                    self._statistics
                    .failed_releases
                ),

                rejected_requests=(
                    self._statistics
                    .rejected_requests
                ),

                release_attempts=(
                    self._statistics
                    .release_attempts
                ),

                last_request_time=(
                    self._statistics
                    .last_request_time
                ),

                last_release_time=(
                    self._statistics
                    .last_release_time
                ),

                last_failure_time=(
                    self._statistics
                    .last_failure_time
                ),
            )

    # =========================================================================
    # Event publishing
    # =========================================================================

    def _publish(
        self,
        event_type: EventType,
        payload: dict,
    ) -> None:
        """Publish EventBus event safely."""

        try:

            self._event_bus.publish(
                event_type,
                payload,
            )

        except Exception:

            logger.exception(
                "[PayloadManager] "
                "Failed to publish event %s",
                event_type,
            )

    # =========================================================================
    # Representation
    # =========================================================================

    def __repr__(self) -> str:

        return (
            "PayloadManager("
            f"state={self.state.value!r}, "
            f"running={self.running}, "
            f"attempts={self.release_attempts}"
            ")"
        )
