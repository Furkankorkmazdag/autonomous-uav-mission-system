"""Serializes MAVLink transactions to prevent concurrent protocol conflicts."""

from __future__ import annotations

import logging
import threading
import time

from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Generator, Optional


logger = logging.getLogger(__name__)


# ============================================================================
# Transaction State
# ============================================================================

class TransactionState(str, Enum):
    """
    Transaction lifecycle states.
    """

    CREATED = "created"

    RUNNING = "running"

    SUCCESS = "success"

    FAILED = "failed"

    TIMEOUT = "timeout"

    CANCELLED = "cancelled"


# ============================================================================
# Transaction Result
# ============================================================================

@dataclass(frozen=True)
class TransactionResult:
    """
    Transaction sonucu.
    """

    success: bool

    state: TransactionState

    transaction_id: int

    operation: str

    message: str = ""

    value: Any = None

    duration_s: float = 0.0


# ============================================================================
# Transaction
# ============================================================================

class Transaction:

    def __init__(
        self,
        transaction_id: int,
        operation: str,
        timeout_s: float,
    ) -> None:

        if not operation:

            raise ValueError(
                "operation cannot be empty"
            )

        if timeout_s <= 0:

            raise ValueError(
                "timeout_s must be greater than zero"
            )

        self.transaction_id = (
            transaction_id
        )

        self.operation = operation

        self.timeout_s = (
            float(timeout_s)
        )

        self.state = (
            TransactionState.CREATED
        )

        self.created_at = (
            time.monotonic()
        )

        self.started_at: Optional[
            float
        ] = None

        self.finished_at: Optional[
            float
        ] = None

        self._cancel_event = (
            threading.Event()
        )

        self._lock = (
            threading.RLock()
        )

    # =========================================================================
    # Start
    # =========================================================================

    def start(self) -> None:

        with self._lock:

            if (
                self.state
                != TransactionState.CREATED
            ):

                raise RuntimeError(
                    "Transaction has already started"
                )

            self.started_at = (
                time.monotonic()
            )

            self.state = (
                TransactionState.RUNNING
            )

    # =========================================================================
    # Finish
    # =========================================================================

    def finish(
        self,
        success: bool,
        *,
        value: Any = None,
        message: str = "",
    ) -> TransactionResult:

        with self._lock:

            if self.finished_at is None:

                self.finished_at = (
                    time.monotonic()
                )

            if success:

                self.state = (
                    TransactionState.SUCCESS
                )

            else:

                self.state = (
                    TransactionState.FAILED
                )

            return self.result(
                value=value,
                message=message,
            )

    # =========================================================================
    # Timeout
    # =========================================================================

    def timeout(
        self,
        message: str = "",
    ) -> TransactionResult:
        """
        Transaction timeout.
        """

        with self._lock:

            self.finished_at = (
                time.monotonic()
            )

            self.state = (
                TransactionState.TIMEOUT
            )

            return self.result(
                message=(
                    message
                    or "Transaction timeout"
                )
            )

    # =========================================================================
    # Cancel
    # =========================================================================

    def cancel(
        self,
        message: str = "",
    ) -> TransactionResult:

        with self._lock:

            self._cancel_event.set()

            self.finished_at = (
                time.monotonic()
            )

            self.state = (
                TransactionState.CANCELLED
            )

            return self.result(
                message=(
                    message
                    or "Transaction cancelled"
                )
            )

    # =========================================================================
    # Cancelled
    # =========================================================================

    @property
    def cancelled(
        self,
    ) -> bool:

        return self._cancel_event.is_set()

    # =========================================================================
    # Deadline
    # =========================================================================

    @property
    def deadline(
        self,
    ) -> Optional[float]:

        if self.started_at is None:

            return None

        return (
            self.started_at
            + self.timeout_s
        )

    # =========================================================================
    # Remaining
    # =========================================================================

    @property
    def remaining_s(
        self,
    ) -> float:

        deadline = self.deadline

        if deadline is None:

            return self.timeout_s

        return max(
            0.0,
            deadline
            - time.monotonic(),
        )

    # =========================================================================
    # Expired
    # =========================================================================

    @property
    def expired(
        self,
    ) -> bool:

        return (
            self.remaining_s
            <= 0.0
        )

    # =========================================================================
    # Result
    # =========================================================================

    def result(
        self,
        *,
        value: Any = None,
        message: str = "",
    ) -> TransactionResult:

        with self._lock:

            if (
                self.started_at is None
            ):

                duration = 0.0

            else:

                end = (
                    self.finished_at
                    or time.monotonic()
                )

                duration = max(
                    0.0,
                    end
                    - self.started_at,
                )

            return TransactionResult(
                success=(
                    self.state
                    == TransactionState.SUCCESS
                ),

                state=self.state,

                transaction_id=(
                    self.transaction_id
                ),

                operation=self.operation,

                message=message,

                value=value,

                duration_s=duration,
            )


# ============================================================================
# Transaction Manager
# ============================================================================

class TransactionManager:

    name = "TransactionManager"

    def __init__(
        self,
        default_timeout_s: float = 5.0,
    ) -> None:

        if default_timeout_s <= 0:

            raise ValueError(
                "default_timeout_s must be greater than zero"
            )

        self.default_timeout_s = (
            float(default_timeout_s)
        )

        self._lock = (
            threading.RLock()
        )

        self._transaction_lock = (
            threading.Lock()
        )

        self._id_lock = (
            threading.Lock()
        )

        self._next_id = 1

        self._active: Optional[
            Transaction
        ] = None

        self._running = False

    # =========================================================================
    # Start
    # =========================================================================

    def start(self) -> bool:

        with self._lock:

            if self._running:

                return True

            self._running = True

        logger.info(
            "[TransactionManager] Started"
        )

        return True

    # =========================================================================
    # Stop
    # =========================================================================

    def stop(self) -> None:

        with self._lock:

            self._running = False

            active = self._active

        if active is not None:

            active.cancel(
                "Transaction manager stopped"
            )

        logger.info(
            "[TransactionManager] Stopped"
        )

    # =========================================================================
    # ID
    # =========================================================================

    def _allocate_id(
        self,
    ) -> int:

        with self._id_lock:

            value = self._next_id

            self._next_id += 1

            if self._next_id > 2_147_483_647:

                self._next_id = 1

            return value

    # =========================================================================
    # Create
    # =========================================================================

    def create(
        self,
        operation: str,
        timeout_s: Optional[
            float
        ] = None,
    ) -> Transaction:

        timeout = (
            timeout_s
            if timeout_s is not None
            else self.default_timeout_s
        )

        return Transaction(
            transaction_id=(
                self._allocate_id()
            ),

            operation=operation,

            timeout_s=timeout,
        )

    # =========================================================================
    # Acquire
    # =========================================================================

    def acquire(
        self,
        operation: str,
        timeout_s: Optional[
            float
        ] = None,
    ) -> Optional[Transaction]:

        if not self._running:

            logger.warning(
                "[TransactionManager] "
                "Manager is not running"
            )

            return None

        transaction = self.create(
            operation,
            timeout_s,
        )

        acquired = (
            self._transaction_lock.acquire(
                timeout=transaction.timeout_s
            )
        )

        if not acquired:

            logger.warning(
                "[TransactionManager] "
                "Transaction lock timeout "
                "operation=%s",
                operation,
            )

            return None

        with self._lock:

            self._active = transaction

        transaction.start()

        logger.debug(
            "[TransactionManager] "
            "Transaction started "
            "id=%d operation=%s",
            transaction.transaction_id,
            operation,
        )

        return transaction

    # =========================================================================
    # Release
    # =========================================================================

    def release(
        self,
        transaction: Transaction,
    ) -> None:

        with self._lock:

            if (
                self._active
                is transaction
            ):

                self._active = None

        try:

            self._transaction_lock.release()

        except RuntimeError:

            logger.warning(
                "[TransactionManager] "
                "Transaction lock already released"
            )

        logger.debug(
            "[TransactionManager] "
            "Transaction released "
            "id=%d operation=%s",
            transaction.transaction_id,
            transaction.operation,
        )

    # =========================================================================
    # Context Manager
    # =========================================================================

    @contextmanager
    def transaction(
        self,
        operation: str,
        timeout_s: Optional[
            float
        ] = None,
    ) -> Generator[
        Optional[Transaction],
        None,
        None,
    ]:

        transaction = self.acquire(
            operation,
            timeout_s,
        )

        try:

            yield transaction

        finally:

            if transaction is not None:

                if transaction.state == (
                    TransactionState.RUNNING
                ):

                    transaction.finish(
                        False,
                        message=(
                            "Transaction exited "
                            "without explicit result"
                        ),
                    )

                self.release(
                    transaction
                )

    # =========================================================================
    # Execute
    # =========================================================================

    def execute(
        self,
        operation: str,
        callback: Callable[
            [Transaction],
            Any,
        ],
        timeout_s: Optional[
            float
        ] = None,
    ) -> TransactionResult:

        if not callable(callback):

            raise TypeError(
                "callback must be callable"
            )

        transaction = self.acquire(
            operation,
            timeout_s,
        )

        if transaction is None:

            return TransactionResult(
                success=False,

                state=(
                    TransactionState.TIMEOUT
                ),

                transaction_id=0,

                operation=operation,

                message=(
                    "Unable to acquire "
                    "transaction lock"
                ),
            )

        try:

            if transaction.cancelled:

                return transaction.cancel(
                    "Transaction cancelled"
                )

            if transaction.expired:

                return transaction.timeout()

            try:

                value = callback(
                    transaction
                )

                if transaction.cancelled:

                    return transaction.cancel()

                if transaction.expired:

                    return transaction.timeout()

                return transaction.finish(
                    True,
                    value=value,
                )

            except TimeoutError as exc:

                return transaction.timeout(
                    str(exc)
                )

            except Exception as exc:

                logger.exception(
                    "[TransactionManager] "
                    "Transaction failed "
                    "operation=%s",
                    operation,
                )

                return transaction.finish(
                    False,
                    message=str(exc),
                )

        finally:

            self.release(
                transaction
            )

    # =========================================================================
    # Active Transaction
    # =========================================================================

    def active_transaction(
        self,
    ) -> Optional[Transaction]:

        with self._lock:

            return self._active

    # =========================================================================
    # Busy
    # =========================================================================

    @property
    def busy(
        self,
    ) -> bool:

        with self._lock:

            return self._active is not None

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
    # Representation
    # =========================================================================

    def __repr__(
        self,
    ) -> str:

        with self._lock:

            active = self._active

            if active is None:

                active_name = None

            else:

                active_name = (
                    active.operation
                )

            return (
                "TransactionManager("
                f"running={self._running}, "
                f"busy={self.busy}, "
                f"active={active_name!r}"
                ")"
            )
