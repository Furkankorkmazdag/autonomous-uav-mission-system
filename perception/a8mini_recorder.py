"""Optional SIYI A8 Mini RTSP recording and gimbal-control integration."""

from __future__ import annotations

import logging
import os
import signal
import socket
import struct
import subprocess
import threading
import time
from datetime import datetime
from typing import Optional

import config

logger = logging.getLogger(__name__)


class A8MiniRecorder:

    def __init__(self) -> None:
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._record_file: Optional[str] = None

    # =========================================================================
    # Gimbal
    # =========================================================================

    @staticmethod
    def _crc16(data: bytes) -> int:
        crc = 0x0000
        for byte in data:
            crc ^= byte << 8
            for _ in range(8):
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
        return crc

    def set_gimbal_angle(
        self,
        yaw: float = 0.0,
        pitch: Optional[float] = None,
        max_retry: int = 5,
    ) -> bool:
        if pitch is None:
            pitch = float(config.SIYI_A8_MINI_GIMBAL_PITCH_DEG)

        cmd_id = 0x0E
        data = struct.pack("<hh", int(yaw * 10), int(pitch * 10))

        packet = bytearray()
        packet += bytes([0x55, 0x66, 0x01])
        packet += struct.pack("<H", len(data))
        packet += struct.pack("<H", 0)
        packet += bytes([cmd_id])
        packet += data
        packet += struct.pack("<H", self._crc16(packet))

        ip = config.SIYI_A8_MINI_CAMERA_IP
        port = config.SIYI_A8_MINI_UDP_PORT

        for attempt in range(1, max_retry + 1):
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.settimeout(2)
            try:
                sock.sendto(packet, (ip, port))
                logger.debug(
                    "[A8Mini Gimbal] [%d/%d] Yaw=%.1f° Pitch=%.1f°",
                    attempt, max_retry, yaw, pitch,
                )
                try:
                    sock.recvfrom(1024)
                    logger.debug("[A8Mini Gimbal] ACK received")
                    return True
                except socket.timeout:
                    logger.debug("[A8Mini Gimbal] No ACK received; retrying...")
                    time.sleep(1)
            finally:
                sock.close()

        logger.warning("[A8Mini Gimbal] ACK was not received (max_retry=%d)", max_retry)
        return False

    # =========================================================================
    # GStreamer binary arama
    # =========================================================================

    @staticmethod
    def _find_gstreamer() -> Optional[str]:
        candidates = [
            "/usr/bin/gst-launch-1.0",
            "/usr/local/bin/gst-launch-1.0",
        ]
        for path in candidates:
            if os.path.isfile(path) and os.access(path, os.X_OK):
                if "/snap/" not in os.path.realpath(path):
                    return path
        return None

    # =========================================================================
    # Recording
    # =========================================================================

    def start_recording(self) -> bool:
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                logger.info("[A8Mini] Recording is already active")
                return True

            gst = self._find_gstreamer()
            if gst is None:
                logger.error(
                    "[A8Mini] GStreamer was not found on this system. "
                    "sudo apt install gstreamer1.0-tools gstreamer1.0-plugins-good "
                    "gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav"
                )
                return False

            record_dir = os.path.abspath(config.SIYI_A8_MINI_RECORD_DIR)
            os.makedirs(record_dir, exist_ok=True)

            ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            outfile = os.path.join(record_dir, f"a8mini_{ts}.mkv")

            rtsp_url = config.SIYI_A8_MINI_RTSP_URL

            gst_cmd = [
                gst, "-e",
                "rtspsrc", f"location={rtsp_url}",
                "protocols=tcp",
                f"latency={config.SIYI_A8_MINI_RTSP_LATENCY_MS}",
                "drop-on-latency=true",
                "!",
                "rtph264depay",
                "!",
                "h264parse", "config-interval=-1",
                "!",
                "matroskamux", "streamable=true",
                "!",
                "filesink", f"location={outfile}", "sync=false",
            ]

            # Use a clean environment to avoid Snap/LD library conflicts.
            env = {
                "PATH": "/usr/bin:/usr/local/bin:/bin:/usr/sbin:/sbin",
                "HOME": os.path.expanduser("~"),
                "LANG": os.environ.get("LANG", "C.UTF-8"),
                "LC_ALL": os.environ.get("LC_ALL", ""),
                "GST_PLUGIN_PATH": config.SIYI_A8_MINI_GSTREAMER_PLUGIN_PATH,
                "GST_PLUGIN_SYSTEM_PATH": config.SIYI_A8_MINI_GSTREAMER_PLUGIN_PATH,
                "GST_REGISTRY": os.path.join(
                    os.path.expanduser("~"), ".cache", "gstreamer-1.0", "registry-a8mini.bin"
                ),
            }

            try:
                os.makedirs(
                    os.path.join(os.path.expanduser("~"), ".cache", "gstreamer-1.0"),
                    exist_ok=True,
                )
            except Exception:
                pass

            logger.info("[A8Mini] Starting recording → %s", outfile)

            try:
                proc = subprocess.Popen(
                    gst_cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    env=env,
                    start_new_session=True,
                )
            except Exception as exc:
                logger.error("[A8Mini] Could not start GStreamer: %s", exc)
                return False

            # stderr logger thread
            threading.Thread(
                target=self._drain_stderr,
                args=(proc,),
                daemon=True,
            ).start()

            # Wait three seconds and check whether the process exited immediately.
            time.sleep(3)
            if proc.poll() is not None:
                logger.error(
                    "[A8Mini] GStreamer exited unexpectedly (return code=%d)", proc.returncode
                )
                try:
                    if os.path.exists(outfile) and os.path.getsize(outfile) < 4096:
                        os.remove(outfile)
                except Exception:
                    pass
                return False

            self._proc = proc
            self._record_file = outfile
            logger.info("[A8Mini] ✓ Recording active | PID=%d | %s", proc.pid, outfile)
            return True

    def stop_recording(self) -> None:
        with self._lock:
            if self._proc is None:
                return

            if self._proc.poll() is not None:
                self._proc = None
                self._record_file = None
                return

            proc = self._proc
            outfile = self._record_file

            logger.info("[A8Mini] Stopping recording...")
            try:
                proc.send_signal(signal.SIGINT)
                proc.wait(timeout=15)
                logger.info("[A8Mini] ✓ GStreamer exited cleanly")
            except subprocess.TimeoutExpired:
                logger.warning("[A8Mini] Did not exit within 15 s → terminating")
                try:
                    proc.terminate()
                    proc.wait(timeout=5)
                except Exception:
                    proc.kill()
            except Exception as exc:
                logger.error("[A8Mini] Shutdown error: %s", exc)
                try:
                    proc.kill()
                except Exception:
                    pass
            finally:
                self._proc = None
                self._record_file = None

        if outfile and os.path.isfile(outfile):
            size = os.path.getsize(outfile)
            if size > 4096:
                logger.info(
                    "[A8Mini] ✓ MKV file ready: %s (%.2f MB)",
                    outfile, size / (1024 * 1024),
                )
            else:
                logger.warning("[A8Mini] MKV file is empty or too small: %s", outfile)

    # =========================================================================
    # Soft Reboot
    # =========================================================================

    def soft_reboot(self) -> bool:
        cmd_id  = 0x80
        data    = bytes([0x01])          # reboot payload
        packet  = bytearray()
        packet += bytes([0x55, 0x66, 0x01])
        packet += struct.pack("<H", len(data))
        packet += struct.pack("<H", 0)
        packet += bytes([cmd_id])
        packet += data
        packet += struct.pack("<H", self._crc16(packet))

        ip   = config.SIYI_A8_MINI_CAMERA_IP
        port = config.SIYI_A8_MINI_UDP_PORT

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(2)
        try:
            sock.sendto(packet, (ip, port))
            logger.warning(
                "[A8Mini] Soft-reboot command sent → %s:%d", ip, port
            )
            try:
                sock.recvfrom(1024)
                logger.debug("[A8Mini] Reboot ACK received")
            except socket.timeout:
                # The camera may not return an ACK while rebooting; this is expected.
                logger.debug("[A8Mini] No reboot ACK received (the camera may be restarting)")
            return True
        except Exception as exc:
            logger.error("[A8Mini] Could not send the soft-reboot command: %s", exc)
            return False
        finally:
            sock.close()

    def wait_until_reachable(
        self,
        timeout_s: float | None = None,
        interval_s: float = 2.0,
    ) -> bool:
        if timeout_s is None:
            timeout_s = float(config.SIYI_A8_MINI_REBOOT_WAIT_S)

        ip       = config.SIYI_A8_MINI_CAMERA_IP
        deadline = time.time() + timeout_s

        logger.info(
            "[A8Mini] Waiting for camera (%s) — timeout %.0f s...", ip, timeout_s
        )

        while time.time() < deadline:
            try:
                result = subprocess.run(
                    ["ping", "-c", "1", "-W", str(int(config.SIYI_A8_MINI_PING_TIMEOUT_S)), ip],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=float(config.SIYI_A8_MINI_PING_TIMEOUT_S) + 1,
                )
                if result.returncode == 0:
                    logger.info("[A8Mini] ✓ Camera is reachable: %s", ip)
                    return True
            except Exception:
                pass
            time.sleep(interval_s)

        logger.error(
            "[A8Mini] Camera did not become reachable within %.0f s: %s", timeout_s, ip
        )
        return False

    def is_recording(self) -> bool:
        """Return whether recording is active."""
        with self._lock:
            return self._proc is not None and self._proc.poll() is None

    # =========================================================================
    # Internal
    # =========================================================================

    @staticmethod
    def _drain_stderr(proc: subprocess.Popen) -> None:
        try:
            for raw in proc.stderr:
                line = raw.decode(errors="replace").rstrip()
                if line:
                    logger.debug("[A8Mini GST] %s", line)
        except Exception:
            pass
