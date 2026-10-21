"""SIYI A8 Mini camera stream and gimbal communication helpers.

Camera networking and GStreamer access are optional hardware integrations.
Configure device addresses in config.py before enabling this module.
"""

import sys
import socket
import struct
import time
import threading
import subprocess
import gi

gi.require_version('Gst', '1.0')
gi.require_version('GLib', '2.0')
from gi.repository import Gst, GLib

RTSP_URL = "rtsp://192.168.144.25:8554/main.264"
CAMERA_IP = "192.168.144.25"
SIYI_PORT  = 37260
TARGET_PITCH = -90
TARGET_YAW   = 0

# ============================================================
# CRC16-CCITT
# ============================================================
def crc16(data):
    crc = 0x0000
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc

# ============================================================
# PACKET CREATION
# ============================================================
def build_packet(cmd_id, data=b'', seq=0, need_ack=True):
    ctrl = 0x01 if need_ack else 0x00
    pkt = bytearray()
    pkt += bytes([0x55, 0x66, ctrl])
    pkt += struct.pack("<H", len(data))
    pkt += struct.pack("<H", seq)
    pkt += bytes([cmd_id])
    pkt += data
    pkt += struct.pack("<H", crc16(pkt))
    return bytes(pkt)

# ============================================================
# CAMERA ACK WATCHDOG
# Send periodic packets to the camera. If no ACK is received, mark it as
# unresponsive and notify CameraStream.
# ============================================================
class CameraWatchdog:
    """Monitors camera responsiveness and requests stream recovery."""

    CHECK_INTERVAL  = 2.0   # Polling interval in seconds.
    MISS_THRESHOLD  = 3     # Number of consecutive missed replies before marking the camera offline.
    SOCK_TIMEOUT    = 1.0   # Timeout for an individual request, in seconds.
    RESTART_COOLDOWN = 5.0  # Wait interval after restarting the camera.

    def __init__(self, stop_event: threading.Event):
        self._stop        = stop_event
        self._miss_count  = 0
        self._alive       = True          # Readable by external components.
        self._restart_cb  = None          # Set by CameraStream.
        self._lock        = threading.Lock()
        self._last_restart = 0.0

    # ── Public interface ─────────────────────────────────────
    @property
    def alive(self):
        with self._lock:
            return self._alive

    def set_restart_callback(self, cb):
        self._restart_cb = cb

    # ── Internal helpers ────────────────────────────────────
    def _send_heartbeat(self, sock) -> bool:
        packet = build_packet(0x0D, b'', need_ack=False)
        try:
            sock.sendto(packet, (CAMERA_IP, SIYI_PORT))
            resp, _ = sock.recvfrom(1024)
            return len(resp) > 0
        except socket.timeout:
            return False
        except Exception as e:
            print(f"[Watchdog] Heartbeat error: {e}")
            return False

    def _ping_icmp(self) -> bool:
        try:
            result = subprocess.run(
                ["ping", "-c", "1", "-W", "1", CAMERA_IP],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=3
            )
            return result.returncode == 0
        except Exception:
            return False

    # ── Main loop ────────────────────────────────────────────
    def run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(self.SOCK_TIMEOUT)
        print("[Watchdog] Started.")

        while not self._stop.is_set():
            responded = self._send_heartbeat(sock)

            with self._lock:
                if responded:
                    if not self._alive:
                        print("[Watchdog] ✓ Camera recovered.")
                    self._alive      = True
                    self._miss_count = 0
                else:
                    self._miss_count += 1
                    print(f"[Watchdog] No response received ({self._miss_count}/{self.MISS_THRESHOLD})")

                    if self._miss_count >= self.MISS_THRESHOLD:
                        # Verify connectivity using ICMP as a fallback.
                        if not self._ping_icmp():
                            if self._alive:
                                print("[Watchdog] ✗ Camera is unreachable! Restarting the stream...")
                            self._alive = False
                            now = time.time()
                            # Cooldown to avoid repeated restarts.
                            if (self._restart_cb is not None and
                                    now - self._last_restart > self.RESTART_COOLDOWN):
                                self._last_restart = now
                                # Call the callback on a separate thread so the watchdog loop is not blocked.
                                threading.Thread(
                                    target=self._restart_cb,
                                    daemon=True
                                ).start()
                        else:
                            # ICMP succeeded but UDP did not respond: the camera is reachable, but the SIYI service may be unavailable.
                            print("[Watchdog] Ping succeeds but UDP is unavailable — the SIYI service is not responding.")

            self._stop.wait(self.CHECK_INTERVAL)

        sock.close()
        print("[Watchdog] Stopped.")


# ============================================================
# SET GIMBAL ANGLE — fast, non-blocking operation.
# ============================================================
def set_gimbal_angle(yaw, pitch, sock=None, max_retry=3, timeout=0.4):
    data   = struct.pack("<hh", int(yaw * 10), int(pitch * 10))
    packet = build_packet(0x0E, data)

    own_sock = sock is None
    if own_sock:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(timeout)

    try:
        for attempt in range(1, max_retry + 1):
            sock.sendto(packet, (CAMERA_IP, SIYI_PORT))
            try:
                sock.recvfrom(1024)
                print(f"[Gimbal] ACK ✓  Yaw={yaw}° Pitch={pitch}° ({attempt}. attempt)")
                return True
            except socket.timeout:
                pass
        print(f"[Gimbal] ACK was not received ({max_retry} attempts); continuing anyway.")
        return False
    finally:
        if own_sock:
            sock.close()

# ============================================================
# READ GIMBAL ANGLE (0x0D)
# ============================================================
def read_gimbal_attitude(sock):
    packet = build_packet(0x0D, b'', need_ack=False)
    try:
        sock.sendto(packet, (CAMERA_IP, SIYI_PORT))
        resp, _ = sock.recvfrom(1024)
        if len(resp) >= 14:
            pitch_raw = struct.unpack_from("<h", resp, 10)[0]
            yaw_raw   = struct.unpack_from("<h", resp,  8)[0]
            return pitch_raw / 10.0, yaw_raw / 10.0
    except socket.timeout:
        pass
    except Exception as e:
        print(f"[Guardian] Read error: {e}")
    return None, None

# ============================================================
# PITCH KORUMA — arka plan thread
# ============================================================
_stop_event = threading.Event()

def pitch_guardian_thread(watchdog: CameraWatchdog):
    TOLERANCE      = 5.0
    CHECK_INTERVAL = 1.5
    FORCE_INTERVAL = 8.0
    last_force     = 0.0

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(0.8)
    print("[Guardian] Started.")

    while not _stop_event.is_set():
        # Do not send gimbal commands while the camera is offline.
        if not watchdog.alive:
            _stop_event.wait(CHECK_INTERVAL)
            continue

        try:
            pitch, _ = read_gimbal_attitude(sock)
            now = time.time()

            if pitch is not None:
                if abs(pitch - TARGET_PITCH) > TOLERANCE:
                    print(f"[Guardian] Gimbal deviation! {pitch}°; correcting...")
                    set_gimbal_angle(TARGET_YAW, TARGET_PITCH, sock=sock,
                                     max_retry=3, timeout=0.4)
            else:
                if now - last_force > FORCE_INTERVAL:
                    print("[Guardian] Sending periodic gimbal command...")
                    set_gimbal_angle(TARGET_YAW, TARGET_PITCH, sock=sock,
                                     max_retry=2, timeout=0.4)
                    last_force = now

        except Exception as e:
            print(f"[Guardian] Error: {e}")
            try: sock.close()
            except: pass
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.settimeout(0.8)

        _stop_event.wait(CHECK_INTERVAL)

    sock.close()
    print("[Guardian] Stopped.")

# ============================================================
# CAMERA STREAM — threaded GStreamer pipeline with reconnection.
# ============================================================
class CameraStream:
    PIPELINE = (
        f'rtspsrc location="{RTSP_URL}" '
        f'protocols=tcp '
        f'latency=100 '
        f'drop-on-latency=true '
        f'retry=0 '
        f'tcp-timeout=2000000 '
        f'timeout=3000000 '
        f'! rtph264depay ! h264parse ! '
        f'nvv4l2decoder ! nvvidconv ! video/x-raw, format=BGRx ! '
        f'videoconvert ! video/x-raw, format=BGR ! '
        f'appsink sync=false drop=true max-buffers=1'
    )

    # Use the CAP_GSTREAMER flag when opening the camera.
    # cap = cv2.VideoCapture(PIPELINE, cv2.CAP_GSTREAMER)

    # Use the CAP_GSTREAMER flag when opening the camera.
    # cap = cv2.VideoCapture(PIPELINE, cv2.CAP_GSTREAMER)

    def __init__(self, watchdog: CameraWatchdog):
        self.running    = True
        self._watchdog  = watchdog
        self._lock      = threading.Lock()
        self._pipeline  = None
        self._thread    = None
        self._error_ev  = threading.Event()
        self._restart_requested = threading.Event()   # signal from the watchdog

        # Called by the watchdog when it detects a lost camera connection.
        watchdog.set_restart_callback(self._on_watchdog_restart)

    # ── Watchdog callback ────────────────────────────────────
    def _on_watchdog_restart(self):
        print("[Stream] Watchdog triggered; stopping the pipeline.")
        self._restart_requested.set()
        self._force_stop()

    # ── Pipeline thread ──────────────────────────────────────
    def _stream_thread(self):
        self._error_ev.clear()
        loop = GLib.MainLoop()

        pipeline = Gst.parse_launch(self.PIPELINE)
        with self._lock:
            self._pipeline = pipeline

        bus = pipeline.get_bus()
        bus.add_signal_watch()

        def on_msg(bus, msg):
            t = msg.type
            if t == Gst.MessageType.ERROR:
                err, dbg = msg.parse_error()
                print(f"[Stream] ERROR: {err.message}")
                self._error_ev.set()
                loop.quit()
            elif t == Gst.MessageType.EOS:
                print("[Stream] EOS.")
                self._error_ev.set()
                loop.quit()
            elif t == Gst.MessageType.WARNING:
                w, _ = msg.parse_warning()
                print(f"[Stream] Warning: {w.message}")

        bus.connect("message", on_msg)

        ret = pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            print("[Stream] Pipeline could not be started.")
            self._error_ev.set()
        else:
            print("[Stream] ▶ Stream started.")
            try:
                loop.run()
            except Exception as e:
                print(f"[Stream] Loop error: {e}")
                self._error_ev.set()

        pipeline.set_state(Gst.State.NULL)
        bus.remove_signal_watch()
        with self._lock:
            self._pipeline = None
        print("[Stream] Pipeline stopped.")

    # ── Helpers ─────────────────────────────────────────────
    def _start_once(self):
        t = threading.Thread(target=self._stream_thread, daemon=True)
        self._thread = t
        t.start()

    def _wait_for_thread(self, deadline_s=5.0):
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=deadline_s)
            if self._thread.is_alive():
                print("[Stream] Warning: thread is still running; continuing.")

    def _force_stop(self):
        with self._lock:
            pl = self._pipeline
        if pl:
            try:
                pl.set_state(Gst.State.NULL)
            except Exception:
                pass

    # ── Wait for the camera to recover. ───────────────────────
    def _wait_for_camera(self, poll_interval=2.0, max_wait=120.0):
        print("[Stream] Waiting for camera...")
        start = time.time()
        while not _stop_event.is_set():
            if self._watchdog.alive:
                print("[Stream] ✓ Camera recovered; reconnecting.")
                return True
            elapsed = time.time() - start
            if elapsed > max_wait:
                print(f"[Stream] {max_wait} seconds elapsed without a camera frame; retrying anyway.")
                return False
            print(f"[Stream] Waiting for camera... ({elapsed:.0f}s)")
            time.sleep(poll_interval)
        return False

    # ── Main loop ────────────────────────────────────────────
    def run(self):
        RECONNECT_DELAY = 0.5

        print("[Stream] Press Ctrl+C to exit.")

        while self.running:
            # Wait for the camera to recover if it is offline.
            if not self._watchdog.alive:
                self._wait_for_camera()
                if not self.running:
                    break

            self._restart_requested.clear()

            # Configure the gimbal.
            print("[Stream] Setting gimbal pitch to -90°...")
            set_gimbal_angle(TARGET_YAW, TARGET_PITCH, max_retry=3, timeout=0.4)

            # Connect.
            self._start_once()

            # Wait for an error, EOS, or a watchdog restart request.
            while self.running:
                if self._error_ev.wait(timeout=0.2):
                    break                           # GStreamer error / EOS.
                if self._restart_requested.is_set():
                    print("[Stream] Restarting at the watchdog request...")
                    break

            if not self.running:
                break

            print(f"[Stream] Connection lost. {RECONNECT_DELAY}s before retrying...")
            self._force_stop()
            self._wait_for_thread(deadline_s=3.0)
            time.sleep(RECONNECT_DELAY)

        # Exit.
        self._force_stop()
        self._wait_for_thread(deadline_s=3.0)
        print("[Stream] Closed.")


# ============================================================
# MAIN
# ============================================================
def main():
    Gst.init(None)

    print("=" * 50)
    print(f"Target: Yaw={TARGET_YAW}°  Pitch={TARGET_PITCH}°")
    print("=" * 50)

    # Initial gimbal setup.
    set_gimbal_angle(TARGET_YAW, TARGET_PITCH, max_retry=5, timeout=0.5)
    time.sleep(1)

    # Start the watchdog.
    watchdog = CameraWatchdog(_stop_event)
    watchdog_thread = threading.Thread(target=watchdog.run, daemon=True)
    watchdog_thread.start()

    # Pitch koruma thread
    guardian = threading.Thread(
        target=pitch_guardian_thread,
        args=(watchdog,),
        daemon=True
    )
    guardian.start()

    # Camera stream (blocking; press Ctrl+C to exit).
    stream = CameraStream(watchdog)
    try:
        stream.run()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        stream.running = False
        _stop_event.set()
        guardian.join(timeout=3)
        watchdog_thread.join(timeout=3)

if __name__ == "__main__":
    main()
