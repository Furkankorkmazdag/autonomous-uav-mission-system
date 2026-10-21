"""Copies mission output data to removable storage without blocking the flight workflow."""

from __future__ import annotations

import logging
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Callable, List, Optional, Tuple

logger = logging.getLogger(__name__)


def find_mounted_flash(label: str) -> Optional[Path]:
    """Find a mounted, writable removable-storage volume."""
    if not label:
        return None

    candidates: List[Path] = []

    for base in (Path("/media"), Path("/run/media")):
        try:
            if base.is_dir():
                for user_dir in base.iterdir():
                    candidates.append(user_dir / label)
        except OSError:
            pass

    candidates.append(Path("/media") / label)

    for candidate in candidates:
        try:
            if candidate.is_dir() and os.access(str(candidate), os.W_OK):
                return candidate
        except OSError:
            continue

    return None


class FlashPhotoCopier:
    """Copies the mission data directory to removable storage in a background thread."""

    def __init__(
        self,
        source_dir: str,
        dest_dir: str,
        retry_count: int = 3,
        verify: bool = True,
        on_done: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        self._source = Path(source_dir)
        self._dest = Path(dest_dir)
        self._retry = max(1, int(retry_count))
        self._verify = bool(verify)
        self._on_done = on_done

        self._thread: Optional[threading.Thread] = None
        self._stop_flag = threading.Event()

        # Statistics are readable from the worker thread and external callers.
        self._copied = 0
        self._failed = 0
        self._skipped = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> bool:
        """Start the component or operation."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("[FlashCopy] Already running; ignoring the new request")
            return True

        self._stop_flag.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="FlashPhotoCopier",
            daemon=True,   # This worker does not keep the main process alive after shutdown.
        )
        self._thread.start()
        logger.info(
            "[FlashCopy] Copy started | source=%s → destination=%s",
            self._source, self._dest,
        )
        return True

    def stop(self) -> None:
        """Stop the component or operation safely."""
        self._stop_flag.set()

    def join(self, timeout_s: float = 0.0) -> bool:
        if self._thread is None or not self._thread.is_alive():
            return True
        if timeout_s > 0:
            self._thread.join(timeout=timeout_s)
        return not self._thread.is_alive()

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def stats(self) -> dict:
        return {
            "copied": self._copied,
            "failed": self._failed,
            "skipped": self._skipped,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run(self) -> None:
        try:
            self._copied = 0
            self._failed = 0
            self._skipped = 0

            # ── 1) Check removable-storage availability ───────────────
            if not self._is_flash_accessible():
                logger.warning(
                    "[FlashCopy] Removable storage is unavailable: %s | "
                    "Files remain on local storage; mission continues.",
                    self._dest,
                )
                return

            # ── 2) Create the destination directory ──────────────────
            try:
                self._dest.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                logger.error(
                    "[FlashCopy] Could not create destination directory: %s | %s | "
                    "Skipping removable-storage copy.",
                    self._dest, exc,
                )
                return

            # ── 3) Collect source files, including subdirectories ────
            files = self._collect_files()
            if not files:
                logger.info(
                    "[FlashCopy] No files found to copy: %s", self._source
                )
                return

            logger.info(
                "[FlashCopy] Copying %d files (including subdirectories) | %s → %s",
                len(files), self._source, self._dest,
            )

            # ── 4) Copy files ─────────────────────────────────────────
            for src_path, rel_path in files:
                if self._stop_flag.is_set():
                    logger.info(
                        "[FlashCopy] Stop signal received | "
                        "copied=%d skipped=%d failed=%d",
                        self._copied, self._skipped, self._failed,
                    )
                    break
                dest_path = self._dest / rel_path
                self._copy_file(src_path, dest_path)

            # ── 5) Summary ───────────────────────────────────────────
            logger.info(
                "[FlashCopy] Completed | copied=%d skipped=%d failed=%d",
                self._copied, self._skipped, self._failed,
            )

            if self._on_done is not None:
                try:
                    self._on_done(self._copied, self._failed)
                except Exception:
                    logger.exception("[FlashCopy] Error in on_done callback")

        except Exception:
            logger.exception(
                "[FlashCopy] Unexpected error — mission continues"
            )

    def _is_flash_accessible(self) -> bool:
        try:
            # Find the nearest existing parent directory.
            check = self._dest
            while not check.exists():
                parent = check.parent
                if parent == check:
                    # No existing parent directory was found, even at the filesystem root.
                    return False
                check = parent
            return os.access(str(check), os.W_OK)
        except Exception as exc:
            logger.debug("[FlashCopy] Access-check error: %s", exc)
            return False

    def _collect_files(self) -> List[Tuple[Path, Path]]:
        if not self._source.exists():
            logger.warning("[FlashCopy] Source directory not found: %s", self._source)
            return []

        files: List[Tuple[Path, Path]] = []
        try:
            for item in sorted(self._source.rglob("*")):
                if item.is_file():
                    rel = item.relative_to(self._source)
                    files.append((item, rel))
        except OSError as exc:
            logger.error(
                "[FlashCopy] Could not list source directory: %s | %s",
                self._source, exc,
            )
        return files

    def _copy_file(self, src: Path, dest: Path) -> None:
        # ── Create the destination subdirectory if needed ────────
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.error(
                "[FlashCopy] Could not create destination subdirectory: %s | %s",
                dest.parent, exc,
            )
            self._failed += 1
            return

        # ── Check for destination conflicts ──────────────────────
        if dest.exists():
            try:
                if src.stat().st_size == dest.stat().st_size:
                    logger.debug("[FlashCopy] Skipped (already exists): %s", dest.name)
                    self._skipped += 1
                    return
                logger.warning(
                    "[FlashCopy] File size mismatch; retrying copy: %s "
                    "(source=%d B, destination=%d B)",
                    dest.name, src.stat().st_size, dest.stat().st_size,
                )
            except OSError:
                pass  # Try the copy even if file metadata cannot be read.

        # ── Copy with retries ─────────────────────────────────────────
        for attempt in range(1, self._retry + 1):
            try:
                # copy2 preserves file contents and timestamps.
                # EXIF metadata remains intact because it is stored inside the image file.
                shutil.copy2(str(src), str(dest))

                if self._verify:
                    src_size = src.stat().st_size
                    dst_size = dest.stat().st_size
                    if src_size != dst_size:
                        raise OSError(
                            f"Verification failed: "
                            f"kaynak={src_size} B destination={dst_size} B"
                        )

                logger.debug("[FlashCopy] ✓ %s", dest.name)
                self._copied += 1
                return

            except Exception as exc:
                logger.warning(
                    "[FlashCopy] Copy failed (attempt %d/%d): %s | %s",
                    attempt, self._retry, dest.name, exc,
                )
                if attempt < self._retry:
                    time.sleep(0.3 * attempt)  # Exponential backoff (0.3 / 0.6 / 0.9 s)
                else:
                    logger.error(
                        "[FlashCopy] Could not copy %s after %d attempts; skipping.",
                        src.name, self._retry,
                    )
                    self._failed += 1
                    # Remove an incomplete or corrupt destination copy if necessary.
                    try:
                        if dest.exists():
                            dest.unlink()
                    except OSError:
                        pass
