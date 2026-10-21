"""
navigation/waypoint.py

Waypoint — a pure, thread-safe data model representing a single navigation
target.

ARCHITECTURE
------------
Waypoint is a leaf data object at the bottom of the navigation stack. It
has no knowledge of, and no dependency on, MissionManager, Navigator,
MissionPath, or MAVLink connections. It never opens a socket, never spawns
a thread, and never issues a flight command — it only *describes* a target
and tracks its own lifecycle metadata (status, retries, visit timestamps).

Two identifiers are kept deliberately distinct:

    - `id` (a UUID string): the waypoint's permanent identity. Never
      changes for the lifetime of the object. Used for `__eq__`/`__hash__`
      so Waypoints are safe to put in `set`/`dict` even while MissionPath
      reorders them.
    - `mission_index`: the waypoint's current sequential position within
      a MissionPath. This is mutable — MissionPath may renumber it during
      a reorder — and carries no identity weight.

Mutable runtime fields (status, retry counter, visited/timestamps, enabled
flag) are guarded by an internal `threading.RLock`, because Waypoint
instances are shared and concurrently touched by MissionManager, Navigator,
MissionPath, MappingMission, and ObjectDetectionMission. All mutation goes
through methods (`mark_reached()`, `enable()`, ...) — internal mutable
state is never handed out directly; accessors return copies/snapshots.

Distance and bearing are computed locally with the WGS84 / Haversine
approximation. Waypoint never asks Navigator to do this for it — Navigator
depends on Waypoint, never the other way around.

Camera and payload fields (`CameraRequest`, `PayloadRequest`) are metadata
only: Waypoint records that a photo/video/AI-trigger or a payload release/
arm/verify was *requested* at this waypoint. It never talks to a camera or
a payload controller — MissionManager reads these flags and decides
whether, and how, to act on them.

INTERACTION SUMMARY (see end-of-file docstring for detail)
------------------------------------------------------------
    MissionPath    -> owns ordering; reassigns `mission_index`; reads
                       `status`/`enabled` to decide skip/include.
    Navigator      -> reads lat/lon/altitude/heading/ground_speed/
                       acceptance_radius to fly to the waypoint; never
                       mutates Waypoint directly.
    LapMission     -> calls `mark_reached()`/`mark_failed()`/`reset()` as
                       the FSM progresses; reads `checkpoint` metadata for
                       resume support.
    MissionManager -> reads camera/payload request metadata to decide
                       whether to actually trigger those subsystems; reads
                       `to_mission_item()` for MAVLink upload construction.
"""

from __future__ import annotations

import json
import math
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, Optional

# WGS84 mean earth radius (meters), consistent with standard Haversine use
# in aerospace/GIS applications.
_EARTH_RADIUS_M = 6371000.0


# ------------------------------------------------------------------------
# Enums
# ------------------------------------------------------------------------

class WaypointType(Enum):
    """Semantic role of a waypoint within a mission."""

    NORMAL = "NORMAL"
    TAKEOFF = "TAKEOFF"
    LAND = "LAND"
    RTL = "RTL"
    LOITER = "LOITER"
    PAYLOAD = "PAYLOAD"
    MAPPING = "MAPPING"
    SEARCH = "SEARCH"
    CUSTOM = "CUSTOM"


class WaypointStatus(Enum):
    """Lifecycle status of a waypoint during mission execution."""

    READY = "READY"
    ACTIVE = "ACTIVE"
    REACHED = "REACHED"
    SKIPPED = "SKIPPED"
    FAILED = "FAILED"
    DISABLED = "DISABLED"


# ------------------------------------------------------------------------
# Metadata-only request models (camera / payload)
# ------------------------------------------------------------------------

@dataclass(frozen=True)
class CameraRequest:
    """
    Camera-related metadata requested at this waypoint.

    Waypoint never controls the camera; this is a description of intent
    for MissionManager (or a dedicated camera-control component) to act on.
    """

    take_photo: bool = False
    start_video: bool = False
    stop_video: bool = False
    trigger_ai_detection: bool = False

    def is_empty(self) -> bool:
        """True if no camera action was requested."""
        return not (self.take_photo or self.start_video or self.stop_video or self.trigger_ai_detection)


@dataclass(frozen=True)
class PayloadRequest:
    """
    Payload-related metadata requested at this waypoint.

    Waypoint never actuates a payload; MissionManager decides whether and
    how to act on these flags.
    """

    release: bool = False
    arm: bool = False
    verify: bool = False

    def is_empty(self) -> bool:
        """True if no payload action was requested."""
        return not (self.release or self.arm or self.verify)


@dataclass(frozen=True)
class CheckpointInfo:
    """
    Persisted resume information for this waypoint, written when the
    waypoint is reached or otherwise finalized, and read back after a
    mission restart so execution can continue correctly.
    """

    arrival_time: Optional[float] = None
    departure_time: Optional[float] = None
    retry_count_at_checkpoint: int = 0
    mission_lap: int = 0
    is_checkpoint: bool = False


# ------------------------------------------------------------------------
# Waypoint
# ------------------------------------------------------------------------

@dataclass(eq=False)
class Waypoint:
    """
    A single navigation target, immutable in its core geospatial identity
    and mutable only in tightly controlled lifecycle fields (all guarded
    by an internal lock).

    Construction validates every geospatial/kinematic field eagerly, so an
    invalid Waypoint can never enter a MissionPath in the first place.
    """

    # --- identity -----------------------------------------------------
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    mission_index: int = 0
    name: str = ""

    # --- geospatial -----------------------------------------------------
    latitude: float = 0.0
    longitude: float = 0.0
    altitude: float = 0.0
    heading: Optional[float] = None
    ground_speed: Optional[float] = None

    # --- tolerances -----------------------------------------------------
    acceptance_radius: float = 2.0
    altitude_tolerance: float = 1.0
    hold_time: float = 0.0

    # --- classification -----------------------------------------------------
    priority: int = 0
    waypoint_type: WaypointType = WaypointType.NORMAL
    status: WaypointStatus = WaypointStatus.READY

    # --- mission bookkeeping -----------------------------------------------------
    lap_number: int = 0
    retry_counter: int = 0
    max_retries: int = 3
    enabled: bool = True
    visited: bool = False

    # --- optional action metadata (camera / payload) -------------------
    camera_request: CameraRequest = field(default_factory=CameraRequest)
    payload_request: PayloadRequest = field(default_factory=PayloadRequest)
    checkpoint: CheckpointInfo = field(default_factory=CheckpointInfo)

    # --- future-compatible extension metadata --------------------------
    # Reserved keys used elsewhere in the codebase (gimbal orientation,
    # camera zoom, target identifier, formation-flight offsets, no-fly
    # metadata, terrain altitude, ...) live here so the dataclass schema
    # never has to change to support them.
    metadata: Dict[str, Any] = field(default_factory=dict)

    # --- timestamps -----------------------------------------------------
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    visited_at: Optional[float] = None

    # --- internal, non-serialized -----------------------------------------------------
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False, compare=False)

    # --------------------------------------------------------------
    # Validation
    # --------------------------------------------------------------

    def __post_init__(self) -> None:
        """Eagerly validate every field so an invalid Waypoint can never
        be constructed."""
        self._validate()

    def _validate(self) -> None:
        """Run every field-level validation check, raising a descriptive
        ValueError on the first violation found."""
        self._validate_number("latitude", self.latitude)
        self._validate_number("longitude", self.longitude)
        self._validate_number("altitude", self.altitude)
        self._validate_number("acceptance_radius", self.acceptance_radius)
        self._validate_number("altitude_tolerance", self.altitude_tolerance)

        if self.heading is not None:
            self._validate_number("heading", self.heading)
        if self.ground_speed is not None:
            self._validate_number("ground_speed", self.ground_speed)

        if not (-90.0 <= self.latitude <= 90.0):
            raise ValueError(f"Waypoint '{self.name or self.id}': latitude out of range: {self.latitude}")
        if not (-180.0 <= self.longitude <= 180.0):
            raise ValueError(f"Waypoint '{self.name or self.id}': longitude out of range: {self.longitude}")
        if self.altitude < 0.0:
            raise ValueError(f"Waypoint '{self.name or self.id}': altitude cannot be negative: {self.altitude}")
        if self.acceptance_radius <= 0.0:
            raise ValueError(
                f"Waypoint '{self.name or self.id}': acceptance_radius must be positive: {self.acceptance_radius}"
            )
        if self.altitude_tolerance < 0.0:
            raise ValueError(
                f"Waypoint '{self.name or self.id}': altitude_tolerance cannot be negative: {self.altitude_tolerance}"
            )
        if self.heading is not None and not (0.0 <= self.heading < 360.0):
            raise ValueError(f"Waypoint '{self.name or self.id}': heading out of range [0, 360): {self.heading}")
        if self.ground_speed is not None and self.ground_speed < 0.0:
            raise ValueError(f"Waypoint '{self.name or self.id}': ground_speed cannot be negative: {self.ground_speed}")
        if self.hold_time < 0.0:
            raise ValueError(f"Waypoint '{self.name or self.id}': hold_time cannot be negative: {self.hold_time}")
        if self.max_retries < 0:
            raise ValueError(f"Waypoint '{self.name or self.id}': max_retries cannot be negative: {self.max_retries}")

    @staticmethod
    def _validate_number(field_name: str, value: Any) -> None:
        """Reject None, NaN, and Infinity for a numeric field."""
        if value is None:
            raise ValueError(f"Waypoint field '{field_name}' cannot be None")
        if not isinstance(value, (int, float)):
            raise ValueError(f"Waypoint field '{field_name}' must be numeric, got {type(value).__name__}")
        if isinstance(value, float):
            if math.isnan(value):
                raise ValueError(f"Waypoint field '{field_name}' cannot be NaN")
            if math.isinf(value):
                raise ValueError(f"Waypoint field '{field_name}' cannot be Infinity")

    # --------------------------------------------------------------
    # Geometry (WGS84 / Haversine — never delegated to Navigator)
    # --------------------------------------------------------------

    def distance_to(self, other: "Waypoint") -> float:
        """
        Great-circle distance to another waypoint, in meters, using the
        Haversine formula against a WGS84 mean earth radius. Purely a
        function of both waypoints' lat/lon — no Navigator involved.
        """
        return self._haversine(self.latitude, self.longitude, other.latitude, other.longitude)

    def bearing_to(self, other: "Waypoint") -> float:
        """
        Initial great-circle bearing to another waypoint, in degrees
        [0, 360), measured clockwise from true north.
        """
        lat1 = math.radians(self.latitude)
        lat2 = math.radians(other.latitude)
        delta_lon = math.radians(other.longitude - self.longitude)

        x = math.sin(delta_lon) * math.cos(lat2)
        y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(delta_lon)
        bearing = math.degrees(math.atan2(x, y))
        return (bearing + 360.0) % 360.0

    def remaining_distance(self, current_position: "Waypoint") -> float:
        """
        Distance from an arbitrary current position (itself expressed as a
        Waypoint-shaped lat/lon/alt) to this waypoint. Convenience wrapper
        around `distance_to` for callers that hold a live position sample
        rather than another persisted Waypoint.
        """
        return self._haversine(current_position.latitude, current_position.longitude, self.latitude, self.longitude)

    @staticmethod
    def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Haversine great-circle distance in meters between two lat/lon
        pairs (WGS84 mean radius approximation)."""
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        d_phi = math.radians(lat2 - lat1)
        d_lambda = math.radians(lon2 - lon1)

        a = math.sin(d_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2.0) ** 2
        c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
        return _EARTH_RADIUS_M * c

    # --------------------------------------------------------------
    # Lifecycle mutation (thread-safe)
    # --------------------------------------------------------------

    def mark_reached(self) -> None:
        """Mark this waypoint REACHED, recording visit/arrival timestamps
        and checkpoint information for resume support."""
        with self._lock:
            now = time.time()
            self.status = WaypointStatus.REACHED
            self.visited = True
            self.visited_at = now
            self.checkpoint = CheckpointInfo(
                arrival_time=now,
                departure_time=self.checkpoint.departure_time,
                retry_count_at_checkpoint=self.retry_counter,
                mission_lap=self.lap_number,
                is_checkpoint=True,
            )
            self._touch()

    def mark_failed(self) -> None:
        """Mark this waypoint FAILED and increment its retry counter."""
        with self._lock:
            self.status = WaypointStatus.FAILED
            self.retry_counter += 1
            self._touch()

    def mark_skipped(self) -> None:
        """Mark this waypoint SKIPPED (e.g. disabled or bypassed by mission logic)."""
        with self._lock:
            self.status = WaypointStatus.SKIPPED
            self._touch()

    def mark_departed(self) -> None:
        """Record the moment the vehicle departs this waypoint, completing
        the arrival/departure checkpoint pair used for resume support."""
        with self._lock:
            now = time.time()
            self.checkpoint = CheckpointInfo(
                arrival_time=self.checkpoint.arrival_time,
                departure_time=now,
                retry_count_at_checkpoint=self.checkpoint.retry_count_at_checkpoint,
                mission_lap=self.checkpoint.mission_lap,
                is_checkpoint=self.checkpoint.is_checkpoint,
            )
            self._touch()

    def enable(self) -> None:
        """Enable this waypoint for navigation. If it was DISABLED, resets
        status to READY."""
        with self._lock:
            self.enabled = True
            if self.status == WaypointStatus.DISABLED:
                self.status = WaypointStatus.READY
            self._touch()

    def disable(self) -> None:
        """Disable this waypoint; MissionPath is expected to skip it."""
        with self._lock:
            self.enabled = False
            self.status = WaypointStatus.DISABLED
            self._touch()

    def activate(self) -> None:
        """Mark this waypoint ACTIVE (currently being navigated to)."""
        with self._lock:
            self.status = WaypointStatus.ACTIVE
            self._touch()

    def reset(self) -> None:
        """
        Reset all runtime/lifecycle state back to its initial values,
        preserving identity, geospatial data, tolerances, and metadata.
        Used e.g. by LapMission when restarting a lap.
        """
        with self._lock:
            self.status = WaypointStatus.READY
            self.retry_counter = 0
            self.visited = False
            self.visited_at = None
            self.checkpoint = CheckpointInfo()
            self._touch()

    def _touch(self) -> None:
        """Update `updated_at`. Must be called while holding `self._lock`."""
        self.updated_at = time.time()

    # --------------------------------------------------------------
    # Copy / clone
    # --------------------------------------------------------------

    def copy(self) -> "Waypoint":
        """
        Return a deep, independent copy of this waypoint that preserves
        the same identity (`id`). Useful for snapshotting current state
        without sharing the lock or mutable containers.
        """
        with self._lock:
            return Waypoint(
                id=self.id,
                mission_index=self.mission_index,
                name=self.name,
                latitude=self.latitude,
                longitude=self.longitude,
                altitude=self.altitude,
                heading=self.heading,
                ground_speed=self.ground_speed,
                acceptance_radius=self.acceptance_radius,
                altitude_tolerance=self.altitude_tolerance,
                hold_time=self.hold_time,
                priority=self.priority,
                waypoint_type=self.waypoint_type,
                status=self.status,
                lap_number=self.lap_number,
                retry_counter=self.retry_counter,
                max_retries=self.max_retries,
                enabled=self.enabled,
                visited=self.visited,
                camera_request=self.camera_request,
                payload_request=self.payload_request,
                checkpoint=self.checkpoint,
                metadata=dict(self.metadata),
                created_at=self.created_at,
                updated_at=self.updated_at,
                visited_at=self.visited_at,
            )

    def clone(self) -> "Waypoint":
        """
        Return a copy of this waypoint with a *new* identity (`id`) — used
        when duplicating a waypoint template into a distinct mission
        target rather than snapshotting the same one.
        """
        clone = self.copy()
        clone.id = str(uuid.uuid4())
        clone.created_at = time.time()
        clone.updated_at = clone.created_at
        return clone

    # --------------------------------------------------------------
    # Serialization
    # --------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this waypoint to a plain, JSON-safe dictionary. Suitable
        for mission persistence (save/load) and inter-process transport.
        """
        with self._lock:
            return {
                "id": self.id,
                "mission_index": self.mission_index,
                "name": self.name,
                "latitude": self.latitude,
                "longitude": self.longitude,
                "altitude": self.altitude,
                "heading": self.heading,
                "ground_speed": self.ground_speed,
                "acceptance_radius": self.acceptance_radius,
                "altitude_tolerance": self.altitude_tolerance,
                "hold_time": self.hold_time,
                "priority": self.priority,
                "waypoint_type": self.waypoint_type.value,
                "status": self.status.value,
                "lap_number": self.lap_number,
                "retry_counter": self.retry_counter,
                "max_retries": self.max_retries,
                "enabled": self.enabled,
                "visited": self.visited,
                "camera_request": asdict(self.camera_request),
                "payload_request": asdict(self.payload_request),
                "checkpoint": asdict(self.checkpoint),
                "metadata": dict(self.metadata),
                "created_at": self.created_at,
                "updated_at": self.updated_at,
                "visited_at": self.visited_at,
            }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Waypoint":
        """Reconstruct a Waypoint from a dictionary produced by `to_dict()`
        (or an equivalently shaped persisted mission definition)."""
        payload = dict(data)  # defensive shallow copy

        waypoint_type = payload.get("waypoint_type", WaypointType.NORMAL.value)
        status = payload.get("status", WaypointStatus.READY.value)
        camera = payload.get("camera_request") or {}
        payload_req = payload.get("payload_request") or {}
        checkpoint = payload.get("checkpoint") or {}

        return cls(
            id=payload.get("id", str(uuid.uuid4())),
            mission_index=payload.get("mission_index", 0),
            name=payload.get("name", ""),
            latitude=payload["latitude"],
            longitude=payload["longitude"],
            altitude=payload.get("altitude", 0.0),
            heading=payload.get("heading"),
            ground_speed=payload.get("ground_speed"),
            acceptance_radius=payload.get("acceptance_radius", 2.0),
            altitude_tolerance=payload.get("altitude_tolerance", 1.0),
            hold_time=payload.get("hold_time", 0.0),
            priority=payload.get("priority", 0),
            waypoint_type=WaypointType(waypoint_type),
            status=WaypointStatus(status),
            lap_number=payload.get("lap_number", 0),
            retry_counter=payload.get("retry_counter", 0),
            max_retries=payload.get("max_retries", 3),
            enabled=payload.get("enabled", True),
            visited=payload.get("visited", False),
            camera_request=CameraRequest(**camera) if camera else CameraRequest(),
            payload_request=PayloadRequest(**payload_req) if payload_req else PayloadRequest(),
            checkpoint=CheckpointInfo(**checkpoint) if checkpoint else CheckpointInfo(),
            metadata=dict(payload.get("metadata", {})),
            created_at=payload.get("created_at", time.time()),
            updated_at=payload.get("updated_at", time.time()),
            visited_at=payload.get("visited_at"),
        )

    def to_json(self) -> str:
        """Serialize this waypoint to a JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, data: str) -> "Waypoint":
        """Reconstruct a Waypoint from a JSON string produced by `to_json()`."""
        return cls.from_dict(json.loads(data))

    # --------------------------------------------------------------
    # MAVLink support (data construction only — never sends anything)
    # --------------------------------------------------------------

    def to_mission_item(self, sequence: Optional[int] = None) -> Dict[str, Any]:
        """
        Build a plain-data MAVLink mission-item-shaped dictionary describing
        this waypoint (command, frame, params, coordinates). This is only a
        data object — it is handed to `mavlink/commands.py` by callers that
        actually talk to Pixhawk; Waypoint itself never opens a connection
        or sends a message.
        """
        command_by_type = {
            WaypointType.NORMAL: "MAV_CMD_NAV_WAYPOINT",
            WaypointType.TAKEOFF: "MAV_CMD_NAV_TAKEOFF",
            WaypointType.LAND: "MAV_CMD_NAV_LAND",
            WaypointType.RTL: "MAV_CMD_NAV_RETURN_TO_LAUNCH",
            WaypointType.LOITER: "MAV_CMD_NAV_LOITER_TIME",
            WaypointType.PAYLOAD: "MAV_CMD_NAV_WAYPOINT",
            WaypointType.MAPPING: "MAV_CMD_NAV_WAYPOINT",
            WaypointType.SEARCH: "MAV_CMD_NAV_WAYPOINT",
            WaypointType.CUSTOM: "MAV_CMD_NAV_WAYPOINT",
        }

        with self._lock:
            return {
                "seq": sequence if sequence is not None else self.mission_index,
                "frame": "MAV_FRAME_GLOBAL_RELATIVE_ALT",
                "command": command_by_type.get(self.waypoint_type, "MAV_CMD_NAV_WAYPOINT"),
                "param1": self.hold_time,
                "param2": self.acceptance_radius,
                "param3": 0.0,
                "param4": self.heading if self.heading is not None else float("nan"),
                "x": self.latitude,
                "y": self.longitude,
                "z": self.altitude,
                "autocontinue": True,
                "current": False,
                "waypoint_id": self.id,
            }

    # --------------------------------------------------------------
    # Equality / hashing (identity == UUID, never geospatial values)
    # --------------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Waypoint):
            return NotImplemented
        return self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)

    # --------------------------------------------------------------
    # Human-readable representations
    # --------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"Waypoint(id={self.id!r}, name={self.name!r}, "
            f"lat={self.latitude:.6f}, lon={self.longitude:.6f}, alt={self.altitude:.1f}, "
            f"type={self.waypoint_type.value}, status={self.status.value}, "
            f"index={self.mission_index}, lap={self.lap_number})"
        )

    def __str__(self) -> str:
        label = self.name or self.id[:8]
        return (
            f"Waypoint '{label}' [{self.waypoint_type.value}] "
            f"@ ({self.latitude:.6f}, {self.longitude:.6f}, {self.altitude:.1f}m) "
            f"— {self.status.value}"
            + (" (visited)" if self.visited else "")
        )


"""
INTERACTION DETAIL
-------------------

MissionPath:
    Holds an ordered collection of Waypoint instances. It is the sole
    authority on ordering and reassigns `mission_index` whenever the path
    is reordered — Waypoint's own `id` never changes, so MissionPath can
    freely resequence without breaking equality/hashing elsewhere (e.g. in
    a `set` used for duplicate detection). MissionPath reads `enabled`/
    `status` to decide whether a waypoint should be skipped, and calls
    `reset()` when restarting a lap.

Navigator:
    Reads `latitude`/`longitude`/`altitude`/`heading`/`ground_speed`/
    `acceptance_radius`/`altitude_tolerance` to fly toward the waypoint and
    to decide when it has arrived. Navigator never mutates a Waypoint
    directly — it reports outcomes (finished/failed) upward via events, and
    it is LapMission/MissionManager that calls `mark_reached()` /
    `mark_failed()` in response.

LapMission:
    Calls `activate()` when a goto is issued, `mark_reached()` when
    Navigator confirms arrival, `mark_failed()` on a goto failure, and
    `reset()` when a lap restarts. It reads `checkpoint` (arrival/departure
    time, retry count, lap) to support resuming a mission from the last
    completed waypoint rather than from index zero after an interruption.

MissionManager:
    Reads `camera_request`/`payload_request` metadata at each waypoint to
    decide whether to actually trigger a photo/video/AI-detection or a
    payload release/arm/verify action — Waypoint only stores the *intent*,
    never performs the action. MissionManager also uses `to_mission_item()`
    when constructing MAVLink mission-upload payloads, and `to_dict()` /
    `from_dict()` / `to_json()` / `from_json()` for saving and reloading a
    mission definition across restarts.
"""
