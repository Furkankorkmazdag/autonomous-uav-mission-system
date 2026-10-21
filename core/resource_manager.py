"""Coordinates exclusive access to shared hardware and software resources."""

from __future__ import annotations

import logging
import threading
import time

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, Set


logger = logging.getLogger(__name__)


# ============================================================================
# Resource State
# ============================================================================

class ResourceState(str, Enum):
    """Resource lifecycle state."""

    AVAILABLE = "available"

    RESERVED = "reserved"

    BUSY = "busy"

    DISABLED = "disabled"


# ============================================================================
# Resource Definition
# ============================================================================

@dataclass
class Resource:

    name: str

    exclusive: bool = True

    state: ResourceState = (
        ResourceState.AVAILABLE
    )

    owner: Optional[str] = None

    users: Set[str] = field(
        default_factory=set
    )


# ============================================================================
# Resource Lease
# ============================================================================

@dataclass(frozen=True)
class ResourceLease:

    resource_name: str

    owner: str

    acquired_at: float


# ============================================================================
# Resource Manager
# ============================================================================

class ResourceManager:

    name = "ResourceManager"

    def __init__(self) -> None:

        self._lock = threading.RLock()

        self._condition = (
            threading.Condition(
                self._lock
            )
        )

        self._resources: Dict[
            str,
            Resource,
        ] = {}

        self._leases: Dict[
            tuple[str, str],
            ResourceLease,
        ] = {}

        self._running = False

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def start(self) -> None:
        """Start resource manager."""

        with self._lock:

            self._running = True

        logger.info(
            "[ResourceManager] Started"
        )

    # -------------------------------------------------------------------------

    def stop(self) -> None:
        """Stop the component or operation safely."""

        with self._condition:

            self._running = False

            for resource in (
                self._resources.values()
            ):

                resource.owner = None

                resource.users.clear()

                if (
                    resource.state
                    != ResourceState.DISABLED
                ):

                    resource.state = (
                        ResourceState.AVAILABLE
                    )

            self._leases.clear()

            self._condition.notify_all()

        logger.info(
            "[ResourceManager] Stopped"
        )

    # =========================================================================
    # Register
    # =========================================================================

    def register(
        self,
        name: str,
        exclusive: bool = True,
    ) -> None:
        """
        Register a new resource.
        """

        if not name:
            raise ValueError(
                "Resource name cannot be empty"
            )

        with self._condition:

            if name in self._resources:

                raise ValueError(
                    f"Resource already registered: {name}"
                )

            self._resources[name] = Resource(
                name=name,

                exclusive=exclusive,
            )

            self._condition.notify_all()

        logger.info(
            "[ResourceManager] "
            "Registered resource: %s "
            "(exclusive=%s)",
            name,
            exclusive,
        )

    # =========================================================================
    # Unregister
    # =========================================================================

    def unregister(
        self,
        name: str,
    ) -> bool:
        """
        Remove a resource.

        Occupied resources cannot be removed.
        """

        with self._condition:

            resource = self._resources.get(
                name
            )

            if resource is None:
                return False

            if resource.owner is not None:

                return False

            if resource.users:

                return False

            del self._resources[name]

            self._condition.notify_all()

        return True

    # =========================================================================
    # Enable / Disable
    # =========================================================================

    def disable(
        self,
        name: str,
    ) -> bool:
        """Disable a resource."""

        with self._condition:

            resource = self._resources.get(
                name
            )

            if resource is None:
                return False

            if resource.owner is not None:
                return False

            if resource.users:
                return False

            resource.state = (
                ResourceState.DISABLED
            )

            self._condition.notify_all()

        logger.warning(
            "[ResourceManager] "
            "Resource disabled: %s",
            name,
        )

        return True

    # -------------------------------------------------------------------------

    def enable(
        self,
        name: str,
    ) -> bool:
        """Enable a disabled resource."""

        with self._condition:

            resource = self._resources.get(
                name
            )

            if resource is None:
                return False

            resource.state = (
                ResourceState.AVAILABLE
            )

            self._condition.notify_all()

        logger.info(
            "[ResourceManager] "
            "Resource enabled: %s",
            name,
        )

        return True

    # =========================================================================
    # Acquire
    # =========================================================================

    def acquire(
        self,
        resource_name: str,
        owner: str,
        timeout_s: float = 0.0,
    ) -> Optional[ResourceLease]:
        """
        Acquire a resource.

        timeout_s = 0:
            Immediate acquisition.

        timeout_s > 0:
            Wait until resource becomes available or timeout occurs.
        """

        if not owner:

            raise ValueError(
                "owner cannot be empty"
            )

        if timeout_s < 0:

            raise ValueError(
                "timeout_s cannot be negative"
            )

        deadline = (
            time.monotonic()
            + timeout_s
        )

        with self._condition:

            while True:

                resource = (
                    self._resources.get(
                        resource_name
                    )
                )

                if resource is None:

                    raise KeyError(
                        f"Unknown resource: "
                        f"{resource_name}"
                    )

                if (
                    resource.state
                    == ResourceState.DISABLED
                ):

                    return None

                # ------------------------------------------------------------
                # Already owned by same owner
                # ------------------------------------------------------------

                if (
                    resource.owner
                    == owner
                ):

                    lease = (
                        self._leases.get(
                            (
                                resource_name,
                                owner,
                            )
                        )
                    )

                    if lease is not None:

                        return lease

                if owner in resource.users:

                    lease = (
                        self._leases.get(
                            (
                                resource_name,
                                owner,
                            )
                        )
                    )

                    if lease is not None:

                        return lease

                # ------------------------------------------------------------
                # Exclusive resource
                # ------------------------------------------------------------

                if resource.exclusive:

                    available = (
                        resource.owner
                        is None
                        and not resource.users
                    )

                    if available:

                        lease = (
                            self._grant(
                                resource,
                                owner,
                            )
                        )

                        return lease

                # ------------------------------------------------------------
                # Shared resource
                # ------------------------------------------------------------

                else:

                    if (
                        resource.owner is None
                    ):

                        lease = (
                            self._grant(
                                resource,
                                owner,
                            )
                        )

                        return lease

                # ------------------------------------------------------------
                # Immediate mode
                # ------------------------------------------------------------

                if timeout_s == 0:

                    return None

                remaining = (
                    deadline
                    - time.monotonic()
                )

                if remaining <= 0:

                    return None

                self._condition.wait(
                    timeout=remaining
                )

    # =========================================================================
    # Grant
    # =========================================================================

    def _grant(
        self,
        resource: Resource,
        owner: str,
    ) -> ResourceLease:
        """Grant resource to owner."""

        now = time.monotonic()

        lease = ResourceLease(
            resource_name=resource.name,

            owner=owner,

            acquired_at=now,
        )

        if resource.exclusive:

            resource.owner = owner

            resource.state = (
                ResourceState.BUSY
            )

        else:

            resource.users.add(
                owner
            )

            resource.state = (
                ResourceState.BUSY
            )

        self._leases[
            (
                resource.name,
                owner,
            )
        ] = lease

        logger.debug(
            "[ResourceManager] "
            "Resource acquired: %s by %s",
            resource.name,
            owner,
        )

        return lease

    # =========================================================================
    # Release
    # =========================================================================

    def release(
        self,
        resource_name: str,
        owner: str,
    ) -> bool:
        """
        Release resource owned by owner.
        """

        with self._condition:

            resource = self._resources.get(
                resource_name
            )

            if resource is None:
                return False

            key = (
                resource_name,
                owner,
            )

            if key not in self._leases:

                return False

            if resource.exclusive:

                if resource.owner != owner:

                    return False

                resource.owner = None

            else:

                if owner not in resource.users:

                    return False

                resource.users.discard(
                    owner
                )

            self._leases.pop(
                key,
                None,
            )

            if (
                resource.owner is None
                and not resource.users
            ):

                resource.state = (
                    ResourceState.AVAILABLE
                )

            self._condition.notify_all()

        logger.debug(
            "[ResourceManager] "
            "Resource released: %s by %s",
            resource_name,
            owner,
        )

        return True

    # =========================================================================
    # Release all
    # =========================================================================

    def release_all(
        self,
        owner: str,
    ) -> int:
        """
        Release every resource owned by a mission.
        """

        with self._condition:

            resources = [
                name
                for (
                    name,
                    current_owner,
                ) in [
                    (
                        resource_name,
                        lease.owner,
                    )
                    for (
                        resource_name,
                        current_owner,
                    ),
                    lease in self._leases.items()
                ]
                if current_owner == owner
            ]

        released = 0

        for resource_name in resources:

            if self.release(
                resource_name,
                owner,
            ):

                released += 1

        return released

    # =========================================================================
    # Is available
    # =========================================================================

    def is_available(
        self,
        resource_name: str,
    ) -> bool:
        """Check whether resource is available."""

        with self._lock:

            resource = self._resources.get(
                resource_name
            )

            if resource is None:
                return False

            return (
                resource.state
                == ResourceState.AVAILABLE
            )

    # =========================================================================
    # Is owned
    # =========================================================================

    def is_owned_by(
        self,
        resource_name: str,
        owner: str,
    ) -> bool:
        """Check whether owner currently owns resource."""

        with self._lock:

            resource = self._resources.get(
                resource_name
            )

            if resource is None:
                return False

            if resource.exclusive:

                return (
                    resource.owner
                    == owner
                )

            return (
                owner
                in resource.users
            )

    # =========================================================================
    # Owner
    # =========================================================================

    def owner(
        self,
        resource_name: str,
    ) -> Optional[str]:
        """Return owner of exclusive resource."""

        with self._lock:

            resource = self._resources.get(
                resource_name
            )

            if resource is None:
                return None

            return resource.owner

    # =========================================================================
    # Users
    # =========================================================================

    def users(
        self,
        resource_name: str,
    ) -> Set[str]:
        """Return current users of resource."""

        with self._lock:

            resource = self._resources.get(
                resource_name
            )

            if resource is None:
                return set()

            users = set(
                resource.users
            )

            if resource.owner is not None:

                users.add(
                    resource.owner
                )

            return users

    # =========================================================================
    # Resource status
    # =========================================================================

    def status(
        self,
        resource_name: str,
    ) -> Optional[Resource]:
        """Return resource snapshot."""

        with self._lock:

            resource = self._resources.get(
                resource_name
            )

            if resource is None:
                return None

            return Resource(
                name=resource.name,

                exclusive=resource.exclusive,

                state=resource.state,

                owner=resource.owner,

                users=set(
                    resource.users
                ),
            )

    # =========================================================================
    # All resources
    # =========================================================================

    def resources(
        self,
    ) -> Dict[
        str,
        Resource,
    ]:
        """Return snapshot of all resources."""

        with self._lock:

            return {
                name: Resource(
                    name=resource.name,

                    exclusive=resource.exclusive,

                    state=resource.state,

                    owner=resource.owner,

                    users=set(
                        resource.users
                    ),
                )

                for name, resource
                in self._resources.items()
            }

    # =========================================================================
    # Mission resource acquisition
    # =========================================================================

    def acquire_many(
        self,
        resource_names: Set[str],
        owner: str,
        timeout_s: float = 0.0,
    ) -> Optional[Set[ResourceLease]]:

        acquired: Set[
            ResourceLease
        ] = set()

        start = time.monotonic()

        for resource_name in sorted(
            resource_names
        ):

            remaining = max(
                0.0,
                timeout_s
                - (
                    time.monotonic()
                    - start
                ),
            )

            lease = self.acquire(
                resource_name=resource_name,

                owner=owner,

                timeout_s=remaining,
            )

            if lease is None:

                for acquired_lease in acquired:

                    self.release(
                        acquired_lease.resource_name,
                        owner,
                    )

                return None

            acquired.add(
                lease
            )

        return acquired

    # =========================================================================
    # Mission resource release
    # =========================================================================

    def release_many(
        self,
        resource_names: Set[str],
        owner: str,
    ) -> int:
        """Release multiple resources."""

        released = 0

        for resource_name in resource_names:

            if self.release(
                resource_name,
                owner,
            ):

                released += 1

        return released

    # =========================================================================
    # Resource registration helpers
    # =========================================================================

    def register_default_resources(
        self,
    ) -> None:

        defaults = {
            "navigator": True,

            "camera": True,

            "gps": False,

            "payload": True,

            "mapping": True,

            "object_detector": True,
        }

        for name, exclusive in (
            defaults.items()
        ):

            with self._lock:

                exists = (
                    name
                    in self._resources
                )

            if not exists:

                self.register(
                    name=name,
                    exclusive=exclusive,
                )

    # =========================================================================
    # Lifecycle state
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

            return (
                "ResourceManager("
                f"running={self._running}, "
                f"resources="
                f"{len(self._resources)}"
                ")"
            )
