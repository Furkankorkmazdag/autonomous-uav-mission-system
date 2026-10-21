"""GPS coordinate validation and coordinate-conversion helpers."""

import math


EARTH_RADIUS = 6378137.0


# ----------------------------------------------------------
# Derece -> Radyan
# ----------------------------------------------------------

def deg2rad(degree):

    return math.radians(degree)


# ----------------------------------------------------------
# Radyan -> Derece
# ----------------------------------------------------------

def rad2deg(radian):

    return math.degrees(radian)


# ----------------------------------------------------------
# Is the GPS position valid?
# ----------------------------------------------------------

def valid_gps(lat, lon):

    if lat == 0 and lon == 0:
        return False

    if lat < -90 or lat > 90:
        return False

    if lon < -180 or lon > 180:
        return False

    return True


# ----------------------------------------------------------
# Derece -> MAVLink Integer
# ----------------------------------------------------------

def to_int(lat, lon):

    return int(lat * 1e7), int(lon * 1e7)


# ----------------------------------------------------------
# MAVLink Integer -> Derece
# ----------------------------------------------------------

def from_int(lat, lon):

    return lat / 1e7, lon / 1e7


# ----------------------------------------------------------
# Compute a new GPS position.
# ----------------------------------------------------------

def offset_coordinate(latitude,
                      longitude,
                      north,
                      east):

    d_lat = north / EARTH_RADIUS

    d_lon = east / (
        EARTH_RADIUS *
        math.cos(math.radians(latitude))
    )

    new_lat = latitude + math.degrees(d_lat)

    new_lon = longitude + math.degrees(d_lon)

    return new_lat, new_lon


# ----------------------------------------------------------
# Compute a GPS point from a heading.
# ----------------------------------------------------------

def project_coordinate(latitude,
                       longitude,
                       heading,
                       distance):

    heading = math.radians(heading)

    north = distance * math.cos(heading)

    east = distance * math.sin(heading)

    return offset_coordinate(
        latitude,
        longitude,
        north,
        east
    )


# ----------------------------------------------------------
# Format a GPS coordinate for display.
# ----------------------------------------------------------

def format_gps(lat,
               lon,
               precision=7):

    return (
        f"{lat:.{precision}f}, "
        f"{lon:.{precision}f}"
    )
