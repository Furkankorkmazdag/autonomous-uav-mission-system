"""Base interface and lifecycle hooks shared by mission implementations."""

from abc import ABC, abstractmethod
from enum import Enum
import threading
import time


class MissionStatus(Enum):

    IDLE = 0

    READY = 1

    RUNNING = 2

    PAUSED = 3

    COMPLETED = 4

    FAILED = 5

    ABORTED = 6


class BaseMission(ABC):

    def __init__(self,
                 shared,
                 name):

        self.shared = shared

        self.name = name

        self.status = MissionStatus.IDLE

        self.running = False

        self.lock = threading.RLock()

        self.start_time = None

        self.end_time = None

    # ==================================================

    def start(self):

        with self.lock:

            if self.running:

                return

            self.running = True

            self.status = MissionStatus.RUNNING

            self.start_time = time.time()

            print(f"[MISSION] {self.name} START")

    # ==================================================

    def stop(self):

        with self.lock:

            self.running = False

            self.status = MissionStatus.COMPLETED

            self.end_time = time.time()

            print(f"[MISSION] {self.name} STOP")

    # ==================================================

    def pause(self):

        with self.lock:

            if not self.running:

                return

            self.status = MissionStatus.PAUSED

            print(f"[MISSION] {self.name} PAUSE")

    # ==================================================

    def resume(self):

        with self.lock:

            if not self.running:

                return

            self.status = MissionStatus.RUNNING

            print(f"[MISSION] {self.name} RESUME")

    # ==================================================

    def abort(self):

        with self.lock:

            self.running = False

            self.status = MissionStatus.ABORTED

            self.end_time = time.time()

            print(f"[MISSION] {self.name} ABORT")

    # ==================================================

    def fail(self):

        with self.lock:

            self.running = False

            self.status = MissionStatus.FAILED

            self.end_time = time.time()

            print(f"[MISSION] {self.name} FAILED")

    # ==================================================

    def is_running(self):

        return self.running

    # ==================================================

    def is_paused(self):

        return self.status == MissionStatus.PAUSED

    # ==================================================

    def is_finished(self):

        return self.status in (

            MissionStatus.COMPLETED,

            MissionStatus.FAILED,

            MissionStatus.ABORTED

        )

    # ==================================================

    def duration(self):

        if self.start_time is None:

            return 0

        if self.end_time is None:

            return time.time() - self.start_time

        return self.end_time - self.start_time

    # ==================================================

    @abstractmethod
    def initialize(self):
        pass

    # ==================================================

    @abstractmethod
    def update(self):
        """Update the current state from the latest input."""
        pass

    # ==================================================

    @abstractmethod
    def cleanup(self):
        pass

    # ==================================================

    def run(self):

        self.initialize()

        while self.running:

            if self.status == MissionStatus.PAUSED:

                time.sleep(0.05)

                continue

            try:

                self.update()

            except Exception as e:

                print(f"[{self.name}] {e}")

                self.fail()

                break

            time.sleep(0.02)

        self.cleanup()
