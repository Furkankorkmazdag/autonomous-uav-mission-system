"""Tracks mission waypoints and traversal progress."""

from missions.waypoint import Waypoint


class MissionPath:

    def __init__(self,
                 waypoints=None,
                 lap_count=1):

        self.waypoints = []

        if waypoints:

            self.waypoints.extend(waypoints)

        self.lap_count = max(1, lap_count)

        self.current_index = 0

        self.current_lap = 1

    # ==================================================
    # Waypoint Handling
    # ==================================================

    def add(self, waypoint: Waypoint):

        self.waypoints.append(waypoint)

    def insert(self,
               index,
               waypoint):

        self.waypoints.insert(index, waypoint)

    def remove(self, index):

        self.waypoints.pop(index)

        if self.current_index >= len(self.waypoints):

            self.current_index = max(
                0,
                len(self.waypoints)-1
            )

    def clear(self):

        self.waypoints.clear()

        self.current_index = 0

        self.current_lap = 1

    # ==================================================
    # Bilgiler
    # ==================================================

    def size(self):

        return len(self.waypoints)

    def empty(self):

        return len(self.waypoints) == 0

    def current(self):

        if self.empty():

            return None

        return self.waypoints[self.current_index]

    def previous(self):

        if self.current_index == 0:

            return None

        return self.waypoints[self.current_index-1]

    def next(self):


        if self.empty():

            return False

        self.current_index += 1

        if self.current_index < len(self.waypoints):

            return True

        self.current_index = 0

        self.current_lap += 1

        if self.current_lap > self.lap_count:

            return False

        return True

    # ==================================================
    # Status
    # ==================================================

    def finished(self):

        return self.current_lap > self.lap_count

    def restart(self):

        self.current_index = 0

        self.current_lap = 1

    # ==================================================
    # Getter
    # ==================================================

    def index(self):

        return self.current_index

    def lap(self):

        return self.current_lap

    def total_laps(self):

        return self.lap_count

    def remaining_waypoints(self):

        if self.empty():

            return 0

        return len(self.waypoints)-self.current_index-1

    def total_waypoints(self):

        return len(self.waypoints)

    # ==================================================
    # Setter
    # ==================================================

    def set_laps(self,
                 lap_count):

        self.lap_count = max(1, lap_count)

    def set_current(self,
                    index):

        if index < 0:

            return

        if index >= len(self.waypoints):

            return

        self.current_index = index

    # ==================================================
    # Iterator
    # ==================================================

    def __iter__(self):

        return iter(self.waypoints)

    def __len__(self):

        return len(self.waypoints)

    def __getitem__(self,
                    item):

        return self.waypoints[item]

    def __str__(self):

        return (
            f"MissionPath("
            f"waypoints={len(self.waypoints)}, "
            f"lap={self.current_lap}/{self.lap_count}, "
            f"current={self.current_index}"
            f")"
        )
