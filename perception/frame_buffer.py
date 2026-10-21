"""Thread-safe buffer that retains the latest camera frame."""

import threading
import time


class FrameBuffer:

    def __init__(self):

        self._frame = None

        self._timestamp = 0

        self._frame_id = 0

        self._lock = threading.Lock()

    # --------------------------------------------------
    # Store a new frame.
    # --------------------------------------------------

    def update(self, frame):

        with self._lock:

            # The GStreamer frame is already an independent NumPy buffer.
            # Avoid a second full-frame copy to reduce Jetson CPU and memory usage.
            self._frame = frame

            self._timestamp = time.time()

            self._frame_id += 1

    # --------------------------------------------------
    # Read the latest frame.
    # --------------------------------------------------

    def get(self):

        with self._lock:

            if self._frame is None:
                return None

            return self._frame.copy()

    # --------------------------------------------------
    # Timestamp
    # --------------------------------------------------

    def timestamp(self):

        with self._lock:

            return self._timestamp

    # --------------------------------------------------
    # Frame index.
    # --------------------------------------------------

    def frame_id(self):

        with self._lock:

            return self._frame_id

    # --------------------------------------------------
    # Has a new frame arrived?
    # --------------------------------------------------

    def has_frame(self):

        with self._lock:

            return self._frame is not None

    # --------------------------------------------------
    # Buffer temizle
    # --------------------------------------------------

    def clear(self):

        with self._lock:

            self._frame = None

            self._timestamp = 0

            self._frame_id = 0

    # --------------------------------------------------
    # Bilgi
    # --------------------------------------------------

    def info(self):

        with self._lock:

            return {
                "frame_id": self._frame_id,
                "timestamp": self._timestamp,
                "has_frame": self._frame is not None
            }
