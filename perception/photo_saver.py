"""Distance-based photo capture with GPS EXIF metadata during mapping."""

from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional, Tuple

import cv2
import numpy as np

try:
    import piexif
except ImportError as exc:
    raise ImportError(
        "piexif is not installed. Install it with: python3 -m pip install piexif"
    ) from exc

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration.
# ============================================================================

@dataclass
class PhotoSaverConfig:
    """Configuration for distance-based image capture and metadata output."""

    # Calculated ideal photo spacing from the camera footprint, in meters.
    # Calculated automatically by from_camera_config().
    photo_spacing_m: float = 5.0

    # Safety bounds prevent photos from being captured too frequently or too rarely.
    min_spacing_m: float = 2.0
    max_spacing_m: float = 50.0

    # Output directory.
    output_dir: str = "data/photos"

    # Mapping polygon as (latitude, longitude) points.
    # If None, no polygon-boundary check is performed.
    mapping_polygon: Optional[Tuple[Tuple[float, float], ...]] = None

    # JPEG kalitesi (0-100).
    jpeg_quality: int = 92

    # Write JSON metadata alongside each photo.
    write_metadata: bool = True


# ============================================================================
# Photo saver.
# ============================================================================

def _rational(value: float, scale: int = 1000000) -> tuple[int, int]:
    """Convert a decimal value to a rational pair for EXIF."""
    value = max(0.0, float(value))
    return int(round(value * scale)), scale


def _decimal_to_dms(value: float) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    """Convert a decimal GPS coordinate to EXIF DMS format."""
    value = abs(float(value))

    degrees = int(value)
    minutes_float = (value - degrees) * 60.0
    minutes = int(minutes_float)
    seconds = (minutes_float - minutes) * 60.0

    return (
        (degrees, 1),
        (minutes, 1),
        _rational(seconds, 10000),
    )


def _write_exif_jpeg(
    path: str,
    frame: Any,
    lat: float,
    lon: float,
    alt: float,
    jpeg_quality: int,
) -> None:
    """Encode a JPEG and write GPS EXIF metadata."""
    ok, encoded = cv2.imencode(
        ".jpg",
        frame,
        [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)],
    )
    if not ok:
        raise RuntimeError("JPEG encoding failed.")

    exif = {
        "0th": {
            piexif.ImageIFD.Make: b"SIYI",
            piexif.ImageIFD.Model: b"A8 Mini",
            piexif.ImageIFD.Software: b"Jetson A8Mini EXIF",
        },
        "Exif": {
            piexif.ExifIFD.DateTimeOriginal:
                datetime.now().strftime("%Y:%m:%d %H:%M:%S").encode(),
        },
        "GPS": {},
        "1st": {},
        "thumbnail": None,
    }

    gps = exif["GPS"]

    # Latitude
    gps[piexif.GPSIFD.GPSLatitudeRef] = b"N" if lat >= 0 else b"S"
    gps[piexif.GPSIFD.GPSLatitude] = _decimal_to_dms(lat)

    # Longitude
    gps[piexif.GPSIFD.GPSLongitudeRef] = b"E" if lon >= 0 else b"W"
    gps[piexif.GPSIFD.GPSLongitude] = _decimal_to_dms(lon)

    # Altitude
    gps[piexif.GPSIFD.GPSAltitudeRef] = 0 if alt >= 0 else 1
    gps[piexif.GPSIFD.GPSAltitude] = _rational(abs(alt), 100)

    # GPS UTC date/time
    now = datetime.utcnow()
    gps[piexif.GPSIFD.GPSDateStamp] = now.strftime("%Y:%m:%d").encode()
    gps[piexif.GPSIFD.GPSTimeStamp] = (
        (now.hour, 1),
        (now.minute, 1),
        _rational(now.second + now.microsecond / 1e6, 1000),
    )

    exif_bytes = piexif.dump(exif)
    jpeg_data = encoded.tobytes()

    # Important: the third argument is the output file path.
    piexif.insert(exif_bytes, jpeg_data, str(path))


def _verify_exif(path: str) -> bool:
    """Verify GPS fields in saved image metadata."""
    exif = piexif.load(str(path))
    gps_ifd = exif.get("GPS", {})

    return (
        piexif.GPSIFD.GPSLatitude in gps_ifd
        and piexif.GPSIFD.GPSLongitude in gps_ifd
        and piexif.GPSIFD.GPSAltitude in gps_ifd
    )


class PhotoSaver:
    """Saves mapping photos at a configured ground-distance interval and writes GPS EXIF metadata."""

    def __init__(self, cfg: PhotoSaverConfig) -> None:
        self._cfg = cfg
        self._lock = threading.Lock()

        # The effective photo spacing is cfg.photo_spacing_m clamped to the safety bounds.
        self._spacing_m = max(
            cfg.min_spacing_m,
            min(cfg.photo_spacing_m, cfg.max_spacing_m),
        )

        self._running = False
        self._output_dir: str = cfg.output_dir

        # GPS position where the previous photo was captured.
        self._last_photo_pos: Optional[Tuple[float, float]] = None

        # Statistics.
        self._photo_count = 0
        self._total_distance_m = 0.0

        logger.info(
            "[PhotoSaver] Started | spacing=%.1f m (limits: %.1f–%.1f m) | directory=%s",
            self._spacing_m, cfg.min_spacing_m, cfg.max_spacing_m, cfg.output_dir,
        )

    # -------------------------------------------------------------------------
    # Factory: derive settings from camera parameters.
    # -------------------------------------------------------------------------

    @classmethod
    def from_camera_config(
        cls,
        altitude_m: float,
        vfov_deg: float,
        forward_overlap: float,
        *,
        mapping_polygon: Optional[Tuple[Tuple[float, float], ...]] = None,
        min_spacing_m: float = 2.0,
        max_spacing_m: float = 50.0,
        output_dir: str = "data/photos",
        jpeg_quality: int = 92,
        write_metadata: bool = True,
    ) -> "PhotoSaver":
        """Build photo-capture settings from camera and overlap parameters."""
        ground_height = 2.0 * altitude_m * math.tan(math.radians(vfov_deg / 2.0))
        raw_spacing = ground_height * (1.0 - forward_overlap)
        spacing = max(min_spacing_m, min(raw_spacing, max_spacing_m))

        logger.info(
            "[PhotoSaver] Camera geometry | altitude=%.1fm vfov=%.1f° overlap=%.0f%% "
            "→ ground_h=%.2fm raw_spacing=%.2fm effective_spacing=%.2fm",
            altitude_m, vfov_deg, forward_overlap * 100,
            ground_height, raw_spacing, spacing,
        )

        cfg = PhotoSaverConfig(
            photo_spacing_m=spacing,
            min_spacing_m=min_spacing_m,
            max_spacing_m=max_spacing_m,
            output_dir=output_dir,
            mapping_polygon=(
                tuple((float(lat), float(lon)) for lat, lon in mapping_polygon)
                if mapping_polygon is not None else None
            ),
            jpeg_quality=jpeg_quality,
            write_metadata=write_metadata,
        )
        return cls(cfg)

    # -------------------------------------------------------------------------
    # Lifecycle.
    # -------------------------------------------------------------------------

    def start(self, output_dir: Optional[str] = None) -> None:
        """Start the component or operation."""
        with self._lock:
            if output_dir:
                self._output_dir = output_dir
            os.makedirs(self._output_dir, exist_ok=True)
            self._running = True
            self._last_photo_pos = None
            self._photo_count = 0
            self._total_distance_m = 0.0
        logger.info("[PhotoSaver] Capture enabled → %s", self._output_dir)

    def stop(self) -> None:
        """Stop the photo saver and log summary statistics."""
        with self._lock:
            self._running = False
        logger.info(
            "[PhotoSaver] Stopped | %d photos captured | %.1f m flown",
            self._photo_count, self._total_distance_m,
        )

    # -------------------------------------------------------------------------
    # Mapping-area validation.
    # -------------------------------------------------------------------------

    @staticmethod
    def _point_on_segment(
        lat: float,
        lon: float,
        a_lat: float,
        a_lon: float,
        b_lat: float,
        b_lon: float,
        eps: float = 1e-10,
    ) -> bool:
        cross = (lat - a_lat) * (b_lon - a_lon) - (lon - a_lon) * (b_lat - a_lat)
        if abs(cross) > eps:
            return False

        return (
            min(a_lat, b_lat) - eps <= lat <= max(a_lat, b_lat) + eps
            and min(a_lon, b_lon) - eps <= lon <= max(a_lon, b_lon) + eps
        )

    @classmethod
    def _is_inside_polygon(
        cls,
        lat: float,
        lon: float,
        polygon: Tuple[Tuple[float, float], ...],
    ) -> bool:
        """Return whether a GPS point lies inside the polygon, including the boundary."""
        if len(polygon) < 3:
            return False

        inside = False
        j = len(polygon) - 1

        for i in range(len(polygon)):
            yi, xi = polygon[i]      # latitude, longitude
            yj, xj = polygon[j]

            # Treat points on the polygon boundary as inside.
            if cls._point_on_segment(lat, lon, yi, xi, yj, xj):
                return True

            intersects = ((yi > lat) != (yj > lat)) and (
                lon < (xj - xi) * (lat - yi) / (yj - yi) + xi
            )

            if intersects:
                inside = not inside

            j = i

        return inside

    def _is_mapping_area_valid(self, lat: float, lon: float) -> bool:
        """Check whether the current GPS position lies within the mapping area."""
        polygon = self._cfg.mapping_polygon

        if polygon is None:
            return True

        return self._is_inside_polygon(float(lat), float(lon), polygon)

    # -------------------------------------------------------------------------
    # Ana metod
    # -------------------------------------------------------------------------

    def update(
        self,
        lat: float,
        lon: float,
        alt: float,
        frame: Any,           # numpy ndarray (BGR)
    ) -> bool:
        """Update the current state from the latest input."""
        if frame is None or not isinstance(frame, np.ndarray):
            return False

        with self._lock:
            if not self._running:
                return False

            # Capture photos only inside the mapping polygon.
            # When outside the area, do not update distance calculations or the last-photo position;
            # this prevents outside flight distance from affecting the in-area
            # photo interval.
            if not self._is_mapping_area_valid(lat, lon):
                logger.debug(
                    "[PhotoSaver] Outside mapping area → image skipped | lat=%.7f lon=%.7f",
                    lat, lon,
                )
                return False

            current_pos = (lat, lon)

            # Capture a photo immediately at the first valid position.
            if self._last_photo_pos is None:
                return self._save(frame, lat, lon, alt, dist_m=0.0)

            # Check the travelled distance.
            dist_m = _haversine(
                self._last_photo_pos[0], self._last_photo_pos[1],
                lat, lon,
            )
            self._total_distance_m += dist_m

            if dist_m < self._spacing_m:
                return False

            return self._save(frame, lat, lon, alt, dist_m=dist_m)

    # -------------------------------------------------------------------------
    # Inside the mapping polygon: save the photo.
    # -------------------------------------------------------------------------

    def _save(
        self,
        frame: Any,
        lat: float,
        lon: float,
        alt: float,
        dist_m: float,
    ) -> bool:
        """Save an image and its optional metadata, then update counters."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]   # ms hassasiyeti
        index = self._photo_count + 1
        basename = f"map_{index:04d}_{ts}"
        jpg_path = os.path.join(self._output_dir, f"{basename}.jpg")

        # JPEG with GPS EXIF metadata.
        # Uses the same approach as the working implementation in a8mini_exif_test.py.
        # piexif.insert(exif_bytes, jpeg_data, output_path)
        try:
            _write_exif_jpeg(
                jpg_path,
                frame,
                lat,
                lon,
                alt,
                self._cfg.jpeg_quality,
            )
        except Exception as exc:
            logger.error("[PhotoSaver] JPEG/EXIF write error: %s", exc)
            return False

        if not os.path.isfile(jpg_path):
            logger.error("[PhotoSaver] Image file was not created: %s", jpg_path)
            return False

        # Validate the actual GPS EXIF metadata.
        try:
            if not _verify_exif(jpg_path):
                logger.error(
                    "[PhotoSaver] Image was created, but GPS EXIF data is missing: %s",
                    jpg_path,
                )
                return False
        except Exception as exc:
            logger.error(
                "[PhotoSaver] EXIF validation failed: %s | %s",
                jpg_path,
                exc,
            )
            return False

        # Metadata JSON
        if self._cfg.write_metadata:
            meta = {
                "index": index,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "latitude": lat,
                "longitude": lon,
                "altitude_m": alt,
                "dist_from_prev_m": round(dist_m, 2),
                "photo_spacing_m": round(self._spacing_m, 2),
                "file": os.path.basename(jpg_path),
            }
            json_path = os.path.join(self._output_dir, f"{basename}.json")
            try:
                with open(json_path, "w", encoding="utf-8") as fh:
                    json.dump(meta, fh, indent=2, ensure_ascii=False)
            except OSError as exc:
                logger.warning("[PhotoSaver] Could not write metadata: %s", exc)

        self._photo_count += 1
        self._last_photo_pos = (lat, lon)

        # The photo and EXIF metadata are saved successfully. Per-photo GPS/EXIF
        # details are not printed to the terminal. Errors are still logged above.
        
        return True

    # -------------------------------------------------------------------------
    # Read-only properties.
    # -------------------------------------------------------------------------

    @property
    def photo_count(self) -> int:
        return self._photo_count

    @property
    def spacing_m(self) -> float:
        return self._spacing_m

    @property
    def is_running(self) -> bool:
        return self._running


# ============================================================================
# Helper: Haversine distance.
# ============================================================================

def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6378137.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2.0 * R * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
