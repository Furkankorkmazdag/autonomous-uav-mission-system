"""Shared navigation interface used by mission implementations."""

import time
import threading

from navigation.distance import distance
from core.events import EventType


class Navigator:

    def __init__(self, shared):

        self.shared = shared

        self.state = shared.state

        self.commands = shared.commands

        self.events = shared.events

        self.lock = threading.RLock()

        self.running = False

        self.active = False

        self.target = None

        self.accept_radius = 2.0

        self.last_command = 0

        self.command_interval = 2

        self.timeout = 60

        self.start_time = 0

    # ==================================================

    def start(self):

        self.running = True

    # ==================================================

    def stop(self):

        self.running = False

        self.active = False

    # ==================================================

    def goto(self,
             latitude,
             longitude,
             altitude,
             accept_radius=None):

        with self.lock:

            self.target = {

                "lat": latitude,

                "lon": longitude,

                "alt": altitude

            }

            if accept_radius is not None:

                self.accept_radius = accept_radius

            self.active = True

            self.start_time = time.time()

            self.send_goto()

    # ==================================================

    def send_goto(self):

        if self.target is None:

            return

        self.commands.goto(

            self.target["lat"],

            self.target["lon"],

            self.target["alt"]

        )

        self.last_command = time.time()

    # ==================================================

    def remaining_distance(self):

        if self.target is None:

            return None

        return distance(

            self.state.latitude,

            self.state.longitude,

            self.target["lat"],

            self.target["lon"]

        )

    # ==================================================

    def reached(self):

        d = self.remaining_distance()

        if d is None:

            return False

        return d <= self.accept_radius

    # ==================================================

    def refresh(self):

        if not self.active:

            return

        if time.time() - self.last_command > self.command_interval:

            self.send_goto()


    # ==================================================

    def timeout_check(self):

        if not self.active:

            return False

        if time.time() - self.start_time > self.timeout:

            return True

        return False

    # ==================================================

    def cancel(self):

        self.active = False

        self.target = None

    # ==================================================

    def update(self):

        if not self.active:

            return

        self.refresh()

        if self.reached():

            self.active = False

            self.events.publish(

                EventType.GOTO_FINISHED

            )

            return

        if self.timeout_check():

            self.active = False

            self.events.publish(

                EventType.GOTO_FAILED

            )

    # ==================================================

    def run(self):

        while self.running:

            try:

                self.update()

            except Exception as e:

                print("[Navigator]", e)

            time.sleep(0.05)
