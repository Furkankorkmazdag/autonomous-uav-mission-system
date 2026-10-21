"""Geographic distance, bearing, and waypoint-reach calculations."""

import math

EARTH_RADIUS = 6378137.0


# ---------------------------------------------------------
# Haversine
# ---------------------------------------------------------

def distance(lat1, lon1, lat2, lon2):

    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)

    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        math.sin(dlat / 2) ** 2
        +
        math.cos(lat1)
        * math.cos(lat2)
        * math.sin(dlon / 2) ** 2
    )

    c = 2 * math.atan2(
        math.sqrt(a),
        math.sqrt(1 - a)
    )

    return EARTH_RADIUS * c


# ---------------------------------------------------------
# Bearing
# ---------------------------------------------------------

def bearing(lat1, lon1, lat2, lon2):

    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)

    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    dlon = lon2 - lon1

    x = math.sin(dlon) * math.cos(lat2)

    y = (
        math.cos(lat1) * math.sin(lat2)
        -
        math.sin(lat1)
        * math.cos(lat2)
        * math.cos(dlon)
    )

    angle = math.degrees(math.atan2(x, y))

    return (angle + 360) % 360


# ---------------------------------------------------------
# North/east displacement.
# ---------------------------------------------------------

def north_east_offset(lat1, lon1, lat2, lon2):

    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)

    north = d_lat * EARTH_RADIUS

    east = (
        d_lon
        * EARTH_RADIUS
        * math.cos(math.radians(lat1))
    )

    return north, east


# ---------------------------------------------------------
# Has the waypoint been reached?
# ---------------------------------------------------------

def reached(lat1,
            lon1,
            lat2,
            lon2,
            radius):

    return distance(
        lat1,
        lon1,
        lat2,
        lon2
    ) <= radius


# ---------------------------------------------------------
# 3D Distance
# ---------------------------------------------------------

def distance3d(lat1,
               lon1,
               alt1,
               lat2,
               lon2,
               alt2):

    horizontal = distance(
        lat1,
        lon1,
        lat2,
        lon2
    )

    vertical = alt2 - alt1

    return math.sqrt(
        horizontal ** 2
        +
        vertical ** 2
    )


# ---------------------------------------------------------
# Angular difference.
# ---------------------------------------------------------

def heading_error(current,
                  target):

    error = (target - current + 180) % 360 - 180

    return error


# ---------------------------------------------------------
# Are the points close?
# ---------------------------------------------------------

def is_close(value,
             target,
             tolerance):

    return abs(value - target) <= tolerance
