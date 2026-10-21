"""Reads, caches, and updates ArduPilot parameters."""

from __future__ import annotations

import logging
import threading
import time

from dataclasses import dataclass
from typing import Any, Dict, Optional


logger = logging.getLogger(__name__)


# ============================================================================
# Parameter Value
# ============================================================================

@dataclass(frozen=True)
class ParameterValue:

    name: str

    value: float

    param_type: Optional[int] = None

    received_at: float = 0.0


# ============================================================================
# Parameter Result
# ============================================================================

@dataclass(frozen=True)
class ParameterResult:

    success: bool

    name: str

    value: Optional[float] = None

    message: str = ""

    parameter: Optional[
        ParameterValue
    ] = None


# ============================================================================
# Parameter Manager
# ============================================================================

class ParameterManager:

    name = "ParameterManager"

    def __init__(
        self,
        mavlink: Any,
        timeout_s: float = 3.0,
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

        self.timeout_s = float(
            timeout_s
        )

        self._lock = (
            threading.RLock()
        )

        self._running = False

        self._cache: Dict[
            str,
            ParameterValue,
        ] = {}

        self._pending: Dict[
            str,
            threading.Event,
        ] = {}

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def start(self) -> bool:

        with self._lock:

            if self._running:

                return True

            self._running = True

        logger.info(
            "[ParameterManager] Started"
        )

        return True

    # =========================================================================
    # Stop
    # =========================================================================

    def stop(self) -> None:

        with self._lock:

            self._running = False

            pending = list(
                self._pending.values()
            )

            self._pending.clear()

        for event in pending:

            event.set()

        logger.info(
            "[ParameterManager] Stopped"
        )

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
    # Running
    # =========================================================================

    @property
    def running(self) -> bool:

        with self._lock:

            return self._running

    # =========================================================================
    # Normalize Name
    # =========================================================================

    @staticmethod
    def _normalize_name(
        name: str,
    ) -> str:

        if not isinstance(
            name,
            str,
        ):

            raise TypeError(
                "Parameter name must be a string"
            )

        name = name.strip().upper()

        if not name:

            raise ValueError(
                "Parameter name cannot be empty"
            )

        return name

    # =========================================================================
    # Set
    # =========================================================================

    def set(
        self,
        name: str,
        value: float,
    ) -> ParameterResult:

        name = self._normalize_name(
            name
        )

        try:

            value = float(
                value
            )

        except (
            TypeError,
            ValueError,
        ):

            return ParameterResult(
                success=False,

                name=name,

                message=(
                    "Parameter value must be numeric"
                ),
            )

        if not self.connected:

            return ParameterResult(
                success=False,

                name=name,

                value=value,

                message=(
                    "MAVLink is not connected"
                ),
            )

        try:

            success = (
                self._mavlink
                .set_parameter(
                    name,
                    value,
                )
            )

            if not success:

                return ParameterResult(
                    success=False,

                    name=name,

                    value=value,

                    message=(
                        "Parameter set command failed"
                    ),
                )

            logger.info(
                "[ParameterManager] "
                "Parameter set requested "
                "%s=%s",
                name,
                value,
            )

            return ParameterResult(
                success=True,

                name=name,

                value=value,

                message=(
                    "Parameter set command sent"
                ),
            )

        except Exception as exc:

            logger.exception(
                "[ParameterManager] "
                "Parameter set failed"
            )

            return ParameterResult(
                success=False,

                name=name,

                value=value,

                message=str(exc),
            )

    # =========================================================================
    # Request
    # =========================================================================

    def request(
        self,
        name: str,
    ) -> ParameterResult:
        """
        Pixhawk'tan tek parameter iste.
        """

        name = self._normalize_name(
            name
        )

        if not self.connected:

            return ParameterResult(
                success=False,

                name=name,

                message=(
                    "MAVLink is not connected"
                ),
            )

        try:

            success = (
                self._mavlink
                .request_parameter(
                    name
                )
            )

            if not success:

                return ParameterResult(
                    success=False,

                    name=name,

                    message=(
                        "Parameter request failed"
                    ),
                )

            logger.debug(
                "[ParameterManager] "
                "Parameter request sent: %s",
                name,
            )

            return ParameterResult(
                success=True,

                name=name,

                message=(
                    "Parameter request sent"
                ),
            )

        except Exception as exc:

            logger.exception(
                "[ParameterManager] "
                "Parameter request failed"
            )

            return ParameterResult(
                success=False,

                name=name,

                message=str(exc),
            )

    # =========================================================================
    # Get Cached
    # =========================================================================

    def get(
        self,
        name: str,
    ) -> Optional[
        ParameterValue
    ]:

        name = self._normalize_name(
            name
        )

        with self._lock:

            return self._cache.get(
                name
            )

    # =========================================================================
    # Get Value
    # =========================================================================

    def get_value(
        self,
        name: str,
    ) -> Optional[float]:

        parameter = self.get(
            name
        )

        if parameter is None:

            return None

        return parameter.value

    # =========================================================================
    # Update Cache
    # =========================================================================

    def update_cache(
        self,
        name: str,
        value: float,
        param_type: Optional[int] = None,
    ) -> ParameterValue:

        name = self._normalize_name(
            name
        )

        parameter = ParameterValue(
            name=name,

            value=float(
                value
            ),

            param_type=param_type,

            received_at=time.monotonic(),
        )

        with self._lock:

            self._cache[
                name
            ] = parameter

            pending_event = (
                self._pending.pop(
                    name,
                    None,
                )
            )

        if pending_event is not None:

            pending_event.set()

        logger.debug(
            "[ParameterManager] "
            "Cache updated %s=%s",
            name,
            value,
        )

        return parameter

    # =========================================================================
    # Request And Wait
    # =========================================================================

    def get_remote(
        self,
        name: str,
        timeout_s: Optional[
            float
        ] = None,
        *,
        use_cache: bool = True,
    ) -> ParameterResult:

        name = self._normalize_name(
            name
        )

        timeout = (
            timeout_s
            if timeout_s is not None
            else self.timeout_s
        )

        if timeout <= 0:

            return ParameterResult(
                success=False,

                name=name,

                message=(
                    "timeout_s must be greater than zero"
                ),
            )

        if use_cache:

            cached = self.get(
                name
            )

            if cached is not None:

                return ParameterResult(
                    success=True,

                    name=name,

                    value=cached.value,

                    message="Value returned from cache",

                    parameter=cached,
                )

        if not self.connected:

            return ParameterResult(
                success=False,

                name=name,

                message=(
                    "MAVLink is not connected"
                ),
            )

        event = threading.Event()

        with self._lock:

            self._pending[
                name
            ] = event

            # Remove the old cached value.
            self._cache.pop(
                name,
                None,
            )

        try:

            result = self.request(
                name
            )

            if not result.success:

                return result

            if not event.wait(
                timeout
            ):

                return ParameterResult(
                    success=False,

                    name=name,

                    message=(
                        "Parameter response timeout"
                    ),
                )

            parameter = self.get(
                name
            )

            if parameter is None:

                return ParameterResult(
                    success=False,

                    name=name,

                    message=(
                        "Parameter response received "
                        "but cache is empty"
                    ),
                )

            return ParameterResult(
                success=True,

                name=name,

                value=parameter.value,

                message=(
                    "Parameter received"
                ),

                parameter=parameter,
            )

        finally:

            with self._lock:

                self._pending.pop(
                    name,
                    None,
                )

    # =========================================================================
    # Handle PARAM_VALUE
    # =========================================================================

    def handle_message(
        self,
        message: Any,
    ) -> bool:

        if message is None:

            return False

        try:

            if (
                message.get_type()
                != "PARAM_VALUE"
            ):

                return False

            raw_name = (
                message.param_id
            )

            if isinstance(
                raw_name,
                bytes,
            ):

                name = (
                    raw_name
                    .decode(
                        "ascii",
                        errors="ignore",
                    )
                    .rstrip("\x00")
                )

            else:

                name = str(
                    raw_name
                ).rstrip(
                    "\x00"
                )

            self.update_cache(
                name=name,

                value=float(
                    message.param_value
                ),

                param_type=getattr(
                    message,
                    "param_type",
                    None,
                ),
            )

            return True

        except Exception:

            logger.exception(
                "[ParameterManager] "
                "Failed to process PARAM_VALUE"
            )

            return False

    # =========================================================================
    # Clear Cache
    # =========================================================================

    def clear_cache(
        self,
    ) -> None:

        with self._lock:

            self._cache.clear()

        logger.debug(
            "[ParameterManager] "
            "Cache cleared"
        )

    # =========================================================================
    # Cache Size
    # =========================================================================

    @property
    def cache_size(
        self,
    ) -> int:

        with self._lock:

            return len(
                self._cache
            )

    # =========================================================================
    # Cached Names
    # =========================================================================

    def cached_names(
        self,
    ) -> list[str]:

        with self._lock:

            return list(
                self._cache.keys()
            )

    # =========================================================================
    # Snapshot
    # =========================================================================

    def snapshot(
        self,
    ) -> Dict[
        str,
        ParameterValue,
    ]:

        with self._lock:

            return dict(
                self._cache
            )

    # =========================================================================
    # Wait Until Value
    # =========================================================================

    def wait_for_value(
        self,
        name: str,
        expected_value: float,
        timeout_s: Optional[
            float
        ] = None,
        tolerance: float = 0.0,
    ) -> bool:

        name = self._normalize_name(
            name
        )

        timeout = (
            timeout_s
            if timeout_s is not None
            else self.timeout_s
        )

        if tolerance < 0:

            raise ValueError(
                "tolerance cannot be negative"
            )

        deadline = (
            time.monotonic()
            + timeout
        )

        while (
            time.monotonic()
            < deadline
        ):

            parameter = self.get(
                name
            )

            if parameter is not None:

                if (
                    abs(
                        parameter.value
                        - expected_value
                    )
                    <= tolerance
                ):

                    return True

            remaining = max(
                0.05,
                deadline
                - time.monotonic(),
            )

            self.request(
                name
            )

            time.sleep(
                min(
                    0.2,
                    remaining,
                )
            )

        return False

    # =========================================================================
    # Representation
    # =========================================================================

    def __repr__(
        self,
    ) -> str:

        with self._lock:

            return (
                "ParameterManager("
                f"running={self._running}, "
                f"connected={self.connected}, "
                f"cache_size={len(self._cache)}"
                ")"
            )
