"""Generates an area-coverage waypoint grid from a GPS polygon and camera footprint settings."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from missions.waypoint import (
    Waypoint,
    WaypointType,
    CameraRequest,
)


Coordinate = Tuple[float, float]

EARTH_RADIUS_M = 6371000.0


# ============================================================================
# Coordinate geometry and mapping-grid generation.
# ============================================================================

def _point_segment_distance(px, py, x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    if dx == 0 and dy == 0:
        return math.hypot(px - x1, py - y1)
    t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return math.hypot(px - proj_x, py - proj_y)


def generate_single_grid(polygon, alt, hfov, vfov, overlap, sidelap,
                          start_lat=None, start_lon=None):
    # --- Ground footprint calculation ---
    footprint_w = 2 * alt * math.tan(math.radians(hfov) / 2)  # Horizontal field-of-view footprint.
    footprint_h = 2 * alt * math.tan(math.radians(vfov) / 2)  # Vertical field-of-view footprint.
    strip_dist_m = footprint_w * (1 - sidelap)                # Distance between scan lines (meters).

    # --- Convert the polygon to a local ENU plane in meters ---
    avg_lat = sum(p[0] for p in polygon) / len(polygon)
    lat_per_meter = 1.0 / 111139.0
    lon_per_meter = 1.0 / (111320.0 * math.cos(math.radians(avg_lat)))

    origin_lat, origin_lon = polygon[0]

    def to_xy(lat, lon):
        x = (lon - origin_lon) / lon_per_meter  # east (meters)
        y = (lat - origin_lat) / lat_per_meter  # Northing (meters).
        return (x, y)

    def to_latlon(x, y):
        lat = origin_lat + y * lat_per_meter
        lon = origin_lon + x * lon_per_meter
        return (lat, lon)

    poly_xy = [to_xy(lat, lon) for lat, lon in polygon]
    n = len(poly_xy)

    # --- Find the polygon edge nearest to the start/home position ---
    if start_lat is not None and start_lon is not None:
        start_x, start_y = to_xy(start_lat, start_lon)
    else:
        # Use the polygon centroid when no start position is provided.
        start_x = sum(p[0] for p in poly_xy) / n
        start_y = sum(p[1] for p in poly_xy) / n

    nearest_edge_idx = 0
    min_edge_dist = float("inf")
    for i in range(n):
        x1, y1 = poly_xy[i]
        x2, y2 = poly_xy[(i + 1) % n]
        d = _point_segment_distance(start_x, start_y, x1, y1, x2, y2)
        if d < min_edge_dist:
            min_edge_dist = d
            nearest_edge_idx = i

    ex1, ey1 = poly_xy[nearest_edge_idx]
    ex2, ey2 = poly_xy[(nearest_edge_idx + 1) % n]
    scan_angle = math.atan2(ey2 - ey1, ex2 - ex1)

    print(f"[GRID] Preparing mapping area...")
    print(f"[GRID] Suitable scan direction selected.")

    cos_r, sin_r = math.cos(-scan_angle), math.sin(-scan_angle)
    cos_i, sin_i = math.cos(scan_angle), math.sin(scan_angle)

    def rotate(x, y):
        return (x * cos_r - y * sin_r, x * sin_r + y * cos_r)

    def rotate_back(x, y):
        return (x * cos_i - y * sin_i, x * sin_i + y * cos_i)

    # Rotate the polygon so the nearest edge is parallel to the x-axis.
    poly_rot = [rotate(x, y) for x, y in poly_xy]

    ys = [p[1] for p in poly_rot]
    min_y, max_y = min(ys), max(ys)

    # Use the rotated y-position of the nearest edge to choose which boundary
    # (min_y or max_y) to start from.
    near_edge_y = rotate(ex1, ey1)[1]
    # Start from the nearest edge.
    start_from_min = (near_edge_y - min_y) <= (max_y - near_edge_y)

    # --- Generate scan lines ---
    # Include the polygon boundaries in the scan.
    # Place the first and last scan lines directly on the nearest and opposite boundaries.
    lines_y = []
    y = min_y
    while y <= max_y + 1e-6:
        lines_y.append(y)
        y += strip_dist_m

    if abs(lines_y[-1] - max_y) > 0.2:
        lines_y.append(max_y)

    if len(lines_y) > 2:
        lines_y = lines_y[1:-1]
    elif len(lines_y) == 2:
        mid = (lines_y[0] + lines_y[1]) / 2.0
        lines_y = [mid]

    # Order scan lines from the nearest edge to the far edge.
    if not start_from_min:
        lines_y = lines_y[::-1]

    print(f"[GRID] Camera field of view: {footprint_w:.1f} x {footprint_h:.1f} m")
    print(f"[GRID] Distance between flight lines: {strip_dist_m:.1f} m")
    print("[GRID] Generating flight route...")

    # --- Helper: find x-coordinates where polygon edges intersect the given y
    # coordinate in the rotated plane. ---
    def _row_intersections(curr_y, eps=1e-6):
        intersections = []
        for i in range(n):
            x1, y1 = poly_rot[i]
            x2, y2 = poly_rot[(i + 1) % n]

            if abs(y1 - y2) < eps:
                if abs(curr_y - y1) < 0.05:
                    intersections.extend([x1, x2])
                continue

            ymin = min(y1, y2) - eps
            ymax = max(y1, y2) + eps
            if ymin <= curr_y <= ymax:
                t = (curr_y - y1) / (y2 - y1)
                if -eps <= t <= 1 + eps:
                    x = x1 + t * (x2 - x1)
                    intersections.append(x)

        if len(intersections) < 2:
            return None

        intersections = sorted(intersections)

        # Remove duplicate intersections at the same vertex.
        uniq = []
        for x in intersections:
            if not uniq or abs(x - uniq[-1]) > 0.05:
                uniq.append(x)

        if len(uniq) < 2:
            # This line touches the polygon at only one vertex,
            # so there is no usable width. Return None and let the caller shift
            # retry the scan line.
            return None

        return uniq[0], uniq[-1]

    generated_wps_rot = []
    direction = 1   # Direction: +1 left-to-right, -1 right-to-left in the rotated frame.

    mid_y = (min_y + max_y) / 2.0
    nudge_step = 0.05          # Shift distance in meters for each retry.
    max_nudge = strip_dist_m   # Never shift by more than one scan-line spacing.

    # Prevent the first and last scan lines from overlapping their neighbors.
    edge_offset = strip_dist_m * 0.5
    if len(lines_y) >= 2:
        if start_from_min:
            lines_y[0] += edge_offset
            lines_y[-1] -= edge_offset
        else:
            lines_y[0] -= edge_offset
            lines_y[-1] += edge_offset

    for nominal_y in lines_y:
        # Shift inward: move down when near max_y, or up when near min_y.
        # Always shift toward the polygon interior.
        nudge_dir = -1.0 if nominal_y >= mid_y else 1.0

        curr_y = nominal_y
        result = _row_intersections(curr_y)

        total_nudge = 0.0
        while result is None and total_nudge < max_nudge:
            total_nudge += nudge_step
            curr_y = nominal_y + nudge_dir * total_nudge
            result = _row_intersections(curr_y)

        if result is None:
            print(f"[GRID][WARNING] y={nominal_y:.2f} has no valid intersections; skipping the scan line.")
            continue

        x_start, x_end = result

        if total_nudge > 0:
            print(f"[GRID] y={nominal_y:.2f} intersects only one polygon vertex, shifted inward by "
                  f"{total_nudge:.2f} m (the far boundary is still covered).")

        if direction == 1:
            generated_wps_rot.append((x_start, curr_y))
            generated_wps_rot.append((x_end, curr_y))
        else:
            generated_wps_rot.append((x_end, curr_y))
            generated_wps_rot.append((x_start, curr_y))

        direction *= -1

    # Rotate points back, then convert them to latitude/longitude.
    generated_wps = []
    for x, y in generated_wps_rot:
        xg, yg = rotate_back(x, y)
        generated_wps.append(to_latlon(xg, yg))

    print(f"[GRID] Total {len(generated_wps)} flight waypoints ready.")
    return generated_wps, footprint_h * (1 - overlap)  # (waypoints, step_distance)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class MappingGridConfig:

    altitude_m: float = 25.0

    hfov_deg: float = 68.75
    vfov_deg: float = 54.3

    forward_overlap: float = 0.85
    side_overlap: float = 0.80

    ground_speed_mps: float = 10.0

    acceptance_radius_m: float = 2.0

    altitude_tolerance_m: float = 1.0

    camera_trigger: bool = True

    close_path: bool = False

    # Avoid generating unnecessary waypoints for very small grids.
    minimum_spacing_m: float = 1.0

    # Tolerance used to prevent duplicate waypoints.
    duplicate_tolerance_m: float = 0.5


# ============================================================================
# MappingGrid
# ============================================================================

class MappingGrid:

    def __init__(
        self,
        polygon: Sequence[Coordinate],
        config: Optional[MappingGridConfig] = None,
    ) -> None:

        self.config = config or MappingGridConfig()

        self.polygon: List[Coordinate] = [
            (float(lat), float(lon))
            for lat, lon in polygon
        ]

        self._validate_polygon()

        self._reference_lat = (
            sum(lat for lat, _ in self.polygon)
            / len(self.polygon)
        )

        self._reference_lon = (
            sum(lon for _, lon in self.polygon)
            / len(self.polygon)
        )

        self._waypoints: List[Waypoint] = []

        self._total_distance_m = 0.0

        self._photo_spacing_m = 0.0
        self._line_spacing_m = 0.0

    # =========================================================================
    # Public API
    # =========================================================================

    def generate_waypoints(
        self,
        start_point: Optional[Coordinate] = None,
    ) -> List[Waypoint]:
        """Use the user-provided ideal grid algorithm unchanged for geometry."""

        start_lat = None
        start_lon = None
        if start_point is not None:
            start_lat, start_lon = start_point

        geo_route, step_m = generate_single_grid(
            polygon=self.polygon,
            alt=self.config.altitude_m,
            hfov=self.config.hfov_deg,
            vfov=self.config.vfov_deg,
            overlap=self.config.forward_overlap,
            sidelap=self.config.side_overlap,
            start_lat=start_lat,
            start_lon=start_lon,
        )

        if not geo_route:
            raise RuntimeError("Mapping grid generated zero points")

        self._photo_spacing_m = float(step_m)
        self._line_spacing_m = float(
            2 * self.config.altitude_m
            * math.tan(math.radians(self.config.hfov_deg) / 2)
            * (1.0 - self.config.side_overlap)
        )
        self._waypoints = self._create_waypoints(geo_route)
        self._total_distance_m = self._calculate_route_distance(geo_route)
        return list(self._waypoints)

    # =========================================================================
    # Camera geometry
    # =========================================================================

    def _calculate_camera_spacing(self) -> None:
        """
        Calculate photographic spacing from camera FOV and overlap.

        Ground footprint:

            width  = 2 * altitude * tan(HFOV / 2)
            height = 2 * altitude * tan(VFOV / 2)

        Photo spacing:

            forward_spacing =
                ground_height * (1 - forward_overlap)

            side_spacing =
                ground_width * (1 - side_overlap)
        """

        altitude = self.config.altitude_m

        hfov = math.radians(
            self.config.hfov_deg
        )

        vfov = math.radians(
            self.config.vfov_deg
        )

        ground_width = (
            2.0
            * altitude
            * math.tan(hfov / 2.0)
        )

        ground_height = (
            2.0
            * altitude
            * math.tan(vfov / 2.0)
        )

        self._photo_spacing_m = (
            ground_height
            * (1.0 - self.config.forward_overlap)
        )

        self._line_spacing_m = (
            ground_width
            * (1.0 - self.config.side_overlap)
        )

        self._photo_spacing_m = max(
            self._photo_spacing_m,
            self.config.minimum_spacing_m,
        )

        self._line_spacing_m = max(
            self._line_spacing_m,
            self.config.minimum_spacing_m,
        )

    # =========================================================================
    # Flight lines
    # =========================================================================

    def _generate_flight_lines(
        self,
        local_polygon: Sequence[Coordinate],
        min_x: float,
        max_x: float,
        min_y: float,
        max_y: float,
    ) -> List[List[Coordinate]]:

        lines: List[List[Coordinate]] = []

        y = min_y

        while y <= max_y + 1e-6:

            intersections = (
                self._horizontal_intersections(
                    local_polygon,
                    y,
                )
            )

            if len(intersections) >= 2:

                intersections.sort()

                # Build segments from intersection pairs.
                for i in range(
                    0,
                    len(intersections) - 1,
                    2,
                ):

                    x1 = intersections[i]
                    x2 = intersections[i + 1]

                    if x2 <= x1:
                        continue

                    lines.append(
                        [
                            (x1, y),
                            (x2, y),
                        ]
                    )

            y += self._line_spacing_m

        return lines

    # =========================================================================
    # Polygon intersection
    # =========================================================================

    def _horizontal_intersections(
        self,
        polygon: Sequence[Coordinate],
        y: float,
    ) -> List[float]:
        """
        Calculate polygon intersections with horizontal line y.
        """

        intersections: List[float] = []

        count = len(polygon)

        for i in range(count):

            x1, y1 = polygon[i]

            x2, y2 = polygon[
                (i + 1) % count
            ]

            # Horizontal edge.
            if abs(y1 - y2) < 1e-9:
                continue

            # Ray crossing rule.
            if (
                (y1 <= y < y2)
                or
                (y2 <= y < y1)
            ):

                ratio = (
                    (y - y1)
                    / (y2 - y1)
                )

                x = (
                    x1
                    + ratio * (x2 - x1)
                )

                intersections.append(x)

        return intersections

    # =========================================================================
    # Zig-zag route
    # =========================================================================

    def _build_zigzag_route(
        self,
        flight_lines: Sequence[
            Sequence[Coordinate]
        ],
    ) -> List[Coordinate]:

        route: List[Coordinate] = []

        for index, line in enumerate(
            flight_lines
        ):

            if len(line) < 2:
                continue

            start = line[0]
            end = line[-1]

            # Generate photo points along the line.
            points = self._sample_line(
                start,
                end,
                self._photo_spacing_m,
            )

            if index % 2 == 1:

                points.reverse()

            for point in points:

                if self._is_duplicate_point(
                    point,
                    route,
                ):
                    continue

                route.append(point)

        return route

    # =========================================================================
    # Line sampling
    # =========================================================================

    def _sample_line(
        self,
        start: Coordinate,
        end: Coordinate,
        spacing_m: float,
    ) -> List[Coordinate]:
        """Generate equally spaced points along a line."""

        x1, y1 = start
        x2, y2 = end

        dx = x2 - x1
        dy = y2 - y1

        distance = math.hypot(
            dx,
            dy,
        )

        if distance <= 0.0:

            return [start]

        count = max(
            1,
            int(
                math.floor(
                    distance / spacing_m
                )
            ),
        )

        points: List[Coordinate] = []

        for i in range(count + 1):

            ratio = (
                i / count
                if count > 0
                else 0.0
            )

            x = (
                x1
                + dx * ratio
            )

            y = (
                y1
                + dy * ratio
            )

            points.append(
                (x, y)
            )

        return points

    # =========================================================================
    # Start-point route rotation
    # =========================================================================

    def _rotate_route_to_start_point(
        self,
        route: Sequence[Coordinate],
        start_point: Coordinate,
    ) -> List[Coordinate]:

        if not route:
            return []

        start_local = self._geo_to_local(
            start_point[0],
            start_point[1],
        )

        nearest_index = min(
            range(len(route)),
            key=lambda i: self._distance_2d(
                route[i],
                start_local,
            ),
        )

        return (
            list(route[nearest_index:])
            + list(route[:nearest_index])
        )

    # =========================================================================
    # Waypoint creation
    # =========================================================================

    def _create_waypoints(
        self,
        coordinates: Sequence[Coordinate],
    ) -> List[Waypoint]:
        """Convert geographic coordinates to Waypoint objects."""

        waypoints: List[Waypoint] = []

        for index, (lat, lon) in enumerate(
            coordinates
        ):

            camera_request = CameraRequest(
                take_photo=(
                    self.config.camera_trigger
                )
            )

            waypoint = Waypoint(
                mission_index=index,

                name=f"MAPPING_{index + 1:04d}",

                latitude=lat,
                longitude=lon,

                altitude=self.config.altitude_m,

                ground_speed=(
                    self.config.ground_speed_mps
                ),

                acceptance_radius=(
                    self.config.acceptance_radius_m
                ),

                altitude_tolerance=(
                    self.config.altitude_tolerance_m
                ),

                waypoint_type=(
                    WaypointType.MAPPING
                ),

                camera_request=(
                    camera_request
                ),
            )

            waypoints.append(
                waypoint
            )

        return waypoints

    # =========================================================================
    # Distance
    # =========================================================================

    def total_distance(self) -> float:
        """Return total route distance in meters."""

        return self._total_distance_m

    # -------------------------------------------------------------------------

    def _calculate_route_distance(
        self,
        route: Sequence[Coordinate],
    ) -> float:
        """Calculate total distance between consecutive points."""

        if len(route) < 2:
            return 0.0

        total = 0.0

        for i in range(
            len(route) - 1
        ):

            total += self._distance_2d(
                route[i],
                route[i + 1],
            )

        if self.config.close_path:

            total += self._distance_2d(
                route[-1],
                route[0],
            )

        return total

    # =========================================================================
    # Coordinate conversion
    # =========================================================================

    def _geo_to_local(
        self,
        lat: float,
        lon: float,
    ) -> Coordinate:
        """
        Convert latitude/longitude to local meter coordinates.

        x = East/West
        y = North/South
        """

        lat_rad = math.radians(
            self._reference_lat
        )

        x = (
            math.radians(
                lon - self._reference_lon
            )
            * EARTH_RADIUS_M
            * math.cos(lat_rad)
        )

        y = (
            math.radians(
                lat - self._reference_lat
            )
            * EARTH_RADIUS_M
        )

        return x, y

    # -------------------------------------------------------------------------

    def _local_to_geo(
        self,
        x: float,
        y: float,
    ) -> Coordinate:
        """Convert local meter coordinates to latitude/longitude."""

        lat = (
            self._reference_lat
            + math.degrees(
                y / EARTH_RADIUS_M
            )
        )

        lat_rad = math.radians(
            self._reference_lat
        )

        if abs(math.cos(lat_rad)) < 1e-12:

            lon = self._reference_lon

        else:

            lon = (
                self._reference_lon
                + math.degrees(
                    x
                    / (
                        EARTH_RADIUS_M
                        * math.cos(lat_rad)
                    )
                )
            )

        return lat, lon

    # =========================================================================
    # Geometry helpers
    # =========================================================================

    @staticmethod
    def _distance_2d(
        a: Coordinate,
        b: Coordinate,
    ) -> float:
        """Euclidean distance in local meter coordinates."""

        return math.hypot(
            b[0] - a[0],
            b[1] - a[1],
        )

    # -------------------------------------------------------------------------

    def _is_duplicate_point(
        self,
        point: Coordinate,
        existing: Sequence[Coordinate],
    ) -> bool:
        """Check whether point already exists."""

        tolerance = (
            self.config.duplicate_tolerance_m
        )

        for item in existing:

            if (
                self._distance_2d(
                    point,
                    item,
                )
                <= tolerance
            ):
                return True

        return False

    # =========================================================================
    # Polygon validation
    # =========================================================================

    def _validate_polygon(self) -> None:
        """Validate input polygon."""

        if len(self.polygon) < 3:

            raise ValueError(
                "Mapping polygon requires "
                "at least 3 coordinates"
            )

        for index, (
            lat,
            lon,
        ) in enumerate(
            self.polygon
        ):

            if not math.isfinite(lat):

                raise ValueError(
                    f"Invalid latitude at "
                    f"index {index}: {lat}"
                )

            if not math.isfinite(lon):

                raise ValueError(
                    f"Invalid longitude at "
                    f"index {index}: {lon}"
                )

            if not (
                -90.0
                <= lat
                <= 90.0
            ):

                raise ValueError(
                    f"Latitude out of range "
                    f"at index {index}: {lat}"
                )

            if not (
                -180.0
                <= lon
                <= 180.0
            ):

                raise ValueError(
                    f"Longitude out of range "
                    f"at index {index}: {lon}"
                )

    # =========================================================================
    # Public information
    # =========================================================================

    @property
    def photo_spacing_m(self) -> float:
        """Distance between photographs."""

        return self._photo_spacing_m

    @property
    def line_spacing_m(self) -> float:
        """Distance between flight lines."""

        return self._line_spacing_m

    @property
    def waypoints(self) -> List[Waypoint]:
        """Return generated waypoint list."""

        return list(self._waypoints)

    def waypoint_count(self) -> int:
        """Return number of generated waypoints."""

        return len(self._waypoints)

    # =========================================================================
    # Statistics
    # =========================================================================

    def statistics(self) -> dict:
        """Return grid statistics."""

        return {
            "waypoint_count": len(
                self._waypoints
            ),

            "total_distance_m": (
                self._total_distance_m
            ),

            "photo_spacing_m": (
                self._photo_spacing_m
            ),

            "line_spacing_m": (
                self._line_spacing_m
            ),

            "altitude_m": (
                self.config.altitude_m
            ),

            "hfov_deg": (
                self.config.hfov_deg
            ),

            "vfov_deg": (
                self.config.vfov_deg
            ),

            "forward_overlap": (
                self.config.forward_overlap
            ),

            "side_overlap": (
                self.config.side_overlap
            ),
        }

    # =========================================================================
    # Representation
    # =========================================================================

    def __repr__(self) -> str:

        return (
            "MappingGrid("
            f"polygon_points={len(self.polygon)}, "
            f"waypoints={len(self._waypoints)}, "
            f"distance="
            f"{self._total_distance_m:.1f}m, "
            f"photo_spacing="
            f"{self._photo_spacing_m:.1f}m, "
            f"line_spacing="
            f"{self._line_spacing_m:.1f}m"
            ")"
        )
