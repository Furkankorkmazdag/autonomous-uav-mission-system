"""Computes BODY_NED velocity commands from image-center error and controls target-centering servos."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

from config import (
    MANNEQUIN_SERVO_CHANNEL_1,
    MANNEQUIN_SERVO_CHANNEL_2,
    CANOPY_SERVO_CHANNEL_1,
    CANOPY_SERVO_CHANNEL_2,
    CENTERING_SERVO_ACTIVATE_PWM,
    CENTERING_SERVO_RESET_PWM,
    CENTERING_SERVO_DELAY_S,
    CENTERING_SERVO_HOLD_S,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TargetCenteringConfig:
    kp_x: float = 2.5
    kp_y: float = 2.5
    max_velocity_mps: float = 3.0
    deadband: float = 0.05

    # Set this to -1 if required by the physical camera mounting orientation.
    image_x_to_body_y_sign: float = 1.0
    image_y_to_body_x_sign: float = -1.0  # Forward/backward direction is inverted; corrected with -1.

    def __post_init__(self) -> None:
        if self.kp_x < 0 or self.kp_y < 0:
            raise ValueError("P gains cannot be negative")
        if self.max_velocity_mps <= 0:
            raise ValueError("max_velocity_mps must be positive")
        if not 0 <= self.deadband < 1:
            raise ValueError("deadband must be in [0, 1)")


class TargetCenteringController:
    """Computes target-centering velocity commands and controls the associated servo sequence."""

    def __init__(self, mavlink, config: TargetCenteringConfig | None = None):
        self._mavlink = mavlink
        self.config = config or TargetCenteringConfig()

    @staticmethod
    def _deadband(error: float, deadband: float) -> float:
        if abs(error) <= deadband:
            return 0.0
        if error > 0:
            return error - deadband
        return error + deadband

    def compute(self, normalized_x: float, normalized_y: float) -> tuple[float, float, float]:
        """Compute BODY_NED velocity commands from normalized image coordinates.

For a downward-facing camera, image x controls lateral motion and image y controls forward/backward motion. Verify the sign configuration against the physical camera mount and confirm that yaw-rate remains zero during centering."""
        x = max(0.0, min(1.0, float(normalized_x)))
        y = max(0.0, min(1.0, float(normalized_y)))

        error_x = self._deadband(x - 0.5, self.config.deadband)
        error_y = self._deadband(y - 0.5, self.config.deadband)

        vy = (
            self.config.kp_x
            * error_x
            * self.config.image_x_to_body_y_sign
        )
        vx = (
            self.config.kp_y
            * error_y
            * self.config.image_y_to_body_x_sign
        )

        limit = self.config.max_velocity_mps
        vx = max(-limit, min(limit, vx))
        vy = max(-limit, min(limit, vy))

        return vx, vy, 0.0

    def compute_with_debug(
        self, normalized_x: float, normalized_y: float
    ) -> tuple[float, float, float, float, float]:
        x = max(0.0, min(1.0, float(normalized_x)))
        y = max(0.0, min(1.0, float(normalized_y)))
        error_x = self._deadband(x - 0.5, self.config.deadband)
        error_y = self._deadband(y - 0.5, self.config.deadband)
        vy = self.config.kp_x * error_x * self.config.image_x_to_body_y_sign
        vx = self.config.kp_y * error_y * self.config.image_y_to_body_x_sign
        limit = self.config.max_velocity_mps
        vx = max(-limit, min(limit, vx))
        vy = max(-limit, min(limit, vy))
        return vx, vy, 0.0, error_x, error_y

    def update(self, normalized_x: float, normalized_y: float) -> bool:
        """Update the current state from the latest input."""
        vx, vy, vz = self.compute(normalized_x, normalized_y)
        return bool(self._mavlink.send_body_velocity(vx, vy, vz))

    def update_with_debug(
        self, normalized_x: float, normalized_y: float
    ) -> tuple[bool, float, float, float, float]:
        vx, vy, vz, ex, ey = self.compute_with_debug(normalized_x, normalized_y)
        ok = bool(self._mavlink.send_body_velocity(vx, vy, vz))
        return ok, vx, vy, ex, ey

    def stop(self) -> bool:
        """Stop the component or operation safely."""
        return bool(self._mavlink.send_body_velocity(0.0, 0.0, 0.0))


# ============================================================================
# Servo-trigger routines used during target centering.
# ============================================================================

def activate_mannequin_centering_servos(mavlink: object, stop_event: threading.Event) -> None:
    try:
        if stop_event.is_set():
            return

        # Step 1: set channel 5 to 1900 PWM.
        logger.info(
            "[CenteringServos] Mannequin: ch%d -> %d PWM",
            MANNEQUIN_SERVO_CHANNEL_1, CENTERING_SERVO_ACTIVATE_PWM,
        )
        mavlink.send_servo_command(MANNEQUIN_SERVO_CHANNEL_1, CENTERING_SERVO_ACTIVATE_PWM)

        # Step 2: wait for the configured delay.
        deadline = time.monotonic() + float(CENTERING_SERVO_DELAY_S)
        while time.monotonic() < deadline:
            if stop_event.is_set():
                return
            time.sleep(0.02)

        # Step 3: set channel 6 to 1900 PWM.
        logger.info(
            "[CenteringServos] Mannequin: ch%d -> %d PWM",
            MANNEQUIN_SERVO_CHANNEL_2, CENTERING_SERVO_ACTIVATE_PWM,
        )
        mavlink.send_servo_command(MANNEQUIN_SERVO_CHANNEL_2, CENTERING_SERVO_ACTIVATE_PWM)

        # Step 4: wait for the configured hold interval.
        deadline = time.monotonic() + float(CENTERING_SERVO_HOLD_S)
        while time.monotonic() < deadline:
            if stop_event.is_set():
                break
            time.sleep(0.02)

        # Step 5: reset channel 6 to 1000 PWM.
        logger.info(
            "[CenteringServos] Mannequin: channel %d -> %d PWM (reset)",
            MANNEQUIN_SERVO_CHANNEL_2, CENTERING_SERVO_RESET_PWM,
        )
        mavlink.send_servo_command(MANNEQUIN_SERVO_CHANNEL_2, CENTERING_SERVO_RESET_PWM)

        logger.info("[CenteringServos] Mannequin servo sequence completed.")

    except Exception:
        logger.exception("[CenteringServos] Error during mannequin servo sequence")


def activate_canopy_centering_servos(mavlink: object, stop_event: threading.Event) -> None:
    try:
        if stop_event.is_set():
            return

        # Step 1: set channel 7 to 1900 PWM.
        logger.info(
            "[CenteringServos] Canopy: ch%d -> %d PWM",
            CANOPY_SERVO_CHANNEL_1, CENTERING_SERVO_ACTIVATE_PWM,
        )
        mavlink.send_servo_command(CANOPY_SERVO_CHANNEL_1, CENTERING_SERVO_ACTIVATE_PWM)

        # Step 2: wait for the configured delay.
        deadline = time.monotonic() + float(CENTERING_SERVO_DELAY_S)
        while time.monotonic() < deadline:
            if stop_event.is_set():
                return
            time.sleep(0.02)

        # Step 3: set channel 8 to 1900 PWM.
        logger.info(
            "[CenteringServos] Canopy: ch%d -> %d PWM",
            CANOPY_SERVO_CHANNEL_2, CENTERING_SERVO_ACTIVATE_PWM,
        )
        mavlink.send_servo_command(CANOPY_SERVO_CHANNEL_2, CENTERING_SERVO_ACTIVATE_PWM)

        # Step 4: wait for the configured hold interval.
        deadline = time.monotonic() + float(CENTERING_SERVO_HOLD_S)
        while time.monotonic() < deadline:
            if stop_event.is_set():
                break
            time.sleep(0.02)

        # Step 5: reset channel 8 to 1000 PWM.
        logger.info(
            "[CenteringServos] Canopy: channel %d -> %d PWM (reset)",
            CANOPY_SERVO_CHANNEL_2, CENTERING_SERVO_RESET_PWM,
        )
        mavlink.send_servo_command(CANOPY_SERVO_CHANNEL_2, CENTERING_SERVO_RESET_PWM)

        logger.info("[CenteringServos] Canopy servo sequence completed.")

    except Exception:
        logger.exception("[CenteringServos] Error during canopy servo sequence")
