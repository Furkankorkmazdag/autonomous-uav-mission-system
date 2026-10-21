"""Routes MAVLink commands through the shared flight-controller connection."""

from __future__ import annotations

import logging
import threading

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional


logger = logging.getLogger(__name__)


# ============================================================================
# Command Result
# ============================================================================

class CommandResult(str, Enum):
    """Command execution result."""

    SUCCESS = "success"

    FAILED = "failed"

    REJECTED = "rejected"

    NOT_CONNECTED = "not_connected"

    INVALID = "invalid"


# ============================================================================
# Command Response
# ============================================================================

@dataclass(frozen=True)
class CommandResponse:

    result: CommandResult

    command: str

    message: str = ""

    value: Optional[Any] = None

    def success(self) -> bool:

        return (
            self.result
            == CommandResult.SUCCESS
        )


# ============================================================================
# Command Router
# ============================================================================

class CommandRouter:

    name = "CommandRouter"

    def __init__(
        self,
        mavlink: Any,
    ) -> None:

        if mavlink is None:

            raise ValueError(
                "mavlink cannot be None"
            )

        self._mavlink = mavlink

        self._lock = threading.RLock()

        self._running = False

    # =========================================================================
    # Start
    # =========================================================================

    def start(self) -> bool:
        """Start the component or operation."""

        with self._lock:

            if self._running:

                return True

            self._running = True

        logger.info(
            "[CommandRouter] Started"
        )

        return True

    # =========================================================================
    # Stop
    # =========================================================================

    def stop(self) -> None:
        """Stop the component or operation safely."""

        with self._lock:

            self._running = False

        logger.info(
            "[CommandRouter] Stopped"
        )

    # =========================================================================
    # Availability
    # =========================================================================

    def _available(
        self,
    ) -> bool:

        with self._lock:

            if not self._running:

                return False

        return bool(
            getattr(
                self._mavlink,
                "connected",
                False,
            )
        )

    # =========================================================================
    # ARM
    # =========================================================================

    def arm(
        self,
        force: bool = False,
    ) -> CommandResponse:
        """
        Vehicle arm command.
        """

        if not self._available():

            return CommandResponse(
                result=CommandResult.NOT_CONNECTED,

                command="ARM",

                message=(
                    "MAVLink is not connected"
                ),
            )

        try:

            success = self._mavlink.arm(
                force=force
            )

            if success:

                logger.info(
                    "[CommandRouter] ARM sent"
                )

                return CommandResponse(
                    result=CommandResult.SUCCESS,

                    command="ARM",

                    message="ARM command sent",
                )

            return CommandResponse(
                result=CommandResult.REJECTED,

                command="ARM",

                message="ARM command rejected",
            )

        except Exception as exc:

            logger.exception(
                "[CommandRouter] ARM failed"
            )

            return CommandResponse(
                result=CommandResult.FAILED,

                command="ARM",

                message=str(exc),
            )

    # =========================================================================
    # DISARM
    # =========================================================================

    def disarm(
        self,
        force: bool = False,
    ) -> CommandResponse:
        """
        Vehicle disarm command.
        """

        if not self._available():

            return CommandResponse(
                result=CommandResult.NOT_CONNECTED,

                command="DISARM",

                message=(
                    "MAVLink is not connected"
                ),
            )

        try:

            success = self._mavlink.disarm(
                force=force
            )

            if success:

                logger.info(
                    "[CommandRouter] DISARM sent"
                )

                return CommandResponse(
                    result=CommandResult.SUCCESS,

                    command="DISARM",

                    message="DISARM command sent",
                )

            return CommandResponse(
                result=CommandResult.REJECTED,

                command="DISARM",

                message="DISARM command rejected",
            )

        except Exception as exc:

            logger.exception(
                "[CommandRouter] DISARM failed"
            )

            return CommandResponse(
                result=CommandResult.FAILED,

                command="DISARM",

                message=str(exc),
            )

    # =========================================================================
    # TAKEOFF
    # =========================================================================

    def takeoff(
        self,
        altitude_m: float,
    ) -> CommandResponse:
        """
        Takeoff command.
        """

        if altitude_m <= 0:

            return CommandResponse(
                result=CommandResult.INVALID,

                command="TAKEOFF",

                message=(
                    "Altitude must be greater than zero"
                ),
            )

        if not self._available():

            return CommandResponse(
                result=CommandResult.NOT_CONNECTED,

                command="TAKEOFF",

                message=(
                    "MAVLink is not connected"
                ),
            )

        try:

            success = (
                self._mavlink.takeoff(
                    altitude_m
                )
            )

            if success:

                logger.info(
                    "[CommandRouter] "
                    "TAKEOFF %.2f m sent",
                    altitude_m,
                )

                return CommandResponse(
                    result=CommandResult.SUCCESS,

                    command="TAKEOFF",

                    message="TAKEOFF command sent",

                    value=altitude_m,
                )

            return CommandResponse(
                result=CommandResult.REJECTED,

                command="TAKEOFF",

                message="TAKEOFF command rejected",
            )

        except Exception as exc:

            logger.exception(
                "[CommandRouter] "
                "TAKEOFF failed"
            )

            return CommandResponse(
                result=CommandResult.FAILED,

                command="TAKEOFF",

                message=str(exc),
            )

    # =========================================================================
    # LAND
    # =========================================================================

    def land(
        self,
    ) -> CommandResponse:
        """
        Land command.
        """

        if not self._available():

            return CommandResponse(
                result=CommandResult.NOT_CONNECTED,

                command="LAND",

                message=(
                    "MAVLink is not connected"
                ),
            )

        try:

            success = (
                self._mavlink.land()
            )

            if success:

                logger.warning(
                    "[CommandRouter] "
                    "LAND sent"
                )

                return CommandResponse(
                    result=CommandResult.SUCCESS,

                    command="LAND",

                    message="LAND command sent",
                )

            return CommandResponse(
                result=CommandResult.REJECTED,

                command="LAND",

                message="LAND command rejected",
            )

        except Exception as exc:

            logger.exception(
                "[CommandRouter] "
                "LAND failed"
            )

            return CommandResponse(
                result=CommandResult.FAILED,

                command="LAND",

                message=str(exc),
            )

    # =========================================================================
    # RTL
    # =========================================================================

    def rtl(
        self,
    ) -> CommandResponse:
        """
        Return To Launch.
        """

        if not self._available():

            return CommandResponse(
                result=CommandResult.NOT_CONNECTED,

                command="RTL",

                message=(
                    "MAVLink is not connected"
                ),
            )

        try:

            success = (
                self._mavlink.rtl()
            )

            if success:

                logger.warning(
                    "[CommandRouter] "
                    "RTL sent"
                )

                return CommandResponse(
                    result=CommandResult.SUCCESS,

                    command="RTL",

                    message="RTL command sent",
                )

            return CommandResponse(
                result=CommandResult.REJECTED,

                command="RTL",

                message="RTL command rejected",
            )

        except Exception as exc:

            logger.exception(
                "[CommandRouter] "
                "RTL failed"
            )

            return CommandResponse(
                result=CommandResult.FAILED,

                command="RTL",

                message=str(exc),
            )

    # =========================================================================
    # GUIDED
    # =========================================================================

    def guided(
        self,
    ) -> CommandResponse:

        return self.set_mode(
            "GUIDED"
        )

    # =========================================================================
    # SET MODE
    # =========================================================================

    def set_mode(
        self,
        mode: str,
    ) -> CommandResponse:

        if not mode:

            return CommandResponse(
                result=CommandResult.INVALID,

                command="SET_MODE",

                message=(
                    "Mode cannot be empty"
                ),
            )

        if not self._available():

            return CommandResponse(
                result=CommandResult.NOT_CONNECTED,

                command="SET_MODE",

                message=(
                    "MAVLink is not connected"
                ),
            )

        mode = mode.upper()

        try:

            success = (
                self._mavlink.set_mode(
                    mode
                )
            )

            if success:

                logger.info(
                    "[CommandRouter] "
                    "Mode %s sent",
                    mode,
                )

                return CommandResponse(
                    result=CommandResult.SUCCESS,

                    command="SET_MODE",

                    message=(
                        f"Mode {mode} command sent"
                    ),

                    value=mode,
                )

            return CommandResponse(
                result=CommandResult.REJECTED,

                command="SET_MODE",

                message=(
                    f"Mode {mode} rejected"
                ),
            )

        except Exception as exc:

            logger.exception(
                "[CommandRouter] "
                "SET_MODE failed"
            )

            return CommandResponse(
                result=CommandResult.FAILED,

                command="SET_MODE",

                message=str(exc),
            )

    # =========================================================================
    # SERVO
    # =========================================================================

    def servo(
        self,
        channel: int,
        pwm: int,
    ) -> CommandResponse:
        """
        Pixhawk servo output command.

        Senin sisteminde:

            channel 7 -> MAIN OUT 7
            channel 8 -> MAIN OUT 8
        """

        if channel < 1:

            return CommandResponse(
                result=CommandResult.INVALID,

                command="SERVO",

                message=(
                    "Channel must be >= 1"
                ),
            )

        if not (
            500
            <= pwm
            <= 2500
        ):

            return CommandResponse(
                result=CommandResult.INVALID,

                command="SERVO",

                message=(
                    "PWM must be between "
                    "500 and 2500"
                ),
            )

        if not self._available():

            return CommandResponse(
                result=CommandResult.NOT_CONNECTED,

                command="SERVO",

                message=(
                    "MAVLink is not connected"
                ),
            )

        try:

            success = (
                self._mavlink
                .send_servo_command(
                    channel,
                    pwm,
                )
            )

            if success:

                logger.info(
                    "[CommandRouter] "
                    "SERVO channel=%d pwm=%d",
                    channel,
                    pwm,
                )

                return CommandResponse(
                    result=CommandResult.SUCCESS,

                    command="SERVO",

                    message=(
                        "Servo command sent"
                    ),

                    value={
                        "channel": channel,
                        "pwm": pwm,
                    },
                )

            return CommandResponse(
                result=CommandResult.REJECTED,

                command="SERVO",

                message=(
                    "Servo command rejected"
                ),
            )

        except Exception as exc:

            logger.exception(
                "[CommandRouter] "
                "SERVO failed"
            )

            return CommandResponse(
                result=CommandResult.FAILED,

                command="SERVO",

                message=str(exc),
            )

    # =========================================================================
    # SERVO 7
    # =========================================================================

    def servo_7(
        self,
        pwm: int,
    ) -> CommandResponse:
        """
        MAIN OUT 7 servo command.
        """

        return self.servo(
            channel=7,
            pwm=pwm,
        )

    # =========================================================================
    # SERVO 8
    # =========================================================================

    def servo_8(
        self,
        pwm: int,
    ) -> CommandResponse:
        """
        MAIN OUT 8 servo command.
        """

        return self.servo(
            channel=8,
            pwm=pwm,
        )

    # =========================================================================
    # PARAMETER
    # =========================================================================

    def set_parameter(
        self,
        name: str,
        value: float,
    ) -> CommandResponse:

        if not name:

            return CommandResponse(
                result=CommandResult.INVALID,

                command="SET_PARAMETER",

                message=(
                    "Parameter name cannot be empty"
                ),
            )

        if not self._available():

            return CommandResponse(
                result=CommandResult.NOT_CONNECTED,

                command="SET_PARAMETER",

                message=(
                    "MAVLink is not connected"
                ),
            )

        try:

            success = (
                self._mavlink
                .set_parameter(
                    name,
                    value,
                )
            )

            if success:

                logger.info(
                    "[CommandRouter] "
                    "Parameter %s=%s",
                    name,
                    value,
                )

                return CommandResponse(
                    result=CommandResult.SUCCESS,

                    command="SET_PARAMETER",

                    message=(
                        f"Parameter {name} sent"
                    ),

                    value=value,
                )

            return CommandResponse(
                result=CommandResult.REJECTED,

                command="SET_PARAMETER",

                message=(
                    f"Parameter {name} rejected"
                ),
            )

        except Exception as exc:

            logger.exception(
                "[CommandRouter] "
                "SET_PARAMETER failed"
            )

            return CommandResponse(
                result=CommandResult.FAILED,

                command="SET_PARAMETER",

                message=str(exc),
            )

    # =========================================================================
    # Generic Command
    # =========================================================================

    def command_long(
        self,
        command: int,
        *params: float,
    ) -> CommandResponse:
        """
        Generic MAVLink COMMAND_LONG.
        """

        if not self._available():

            return CommandResponse(
                result=CommandResult.NOT_CONNECTED,

                command="COMMAND_LONG",

                message=(
                    "MAVLink is not connected"
                ),
            )

        try:

            success = (
                self._mavlink.command_long(
                    command,
                    *params,
                )
            )

            if success:

                return CommandResponse(
                    result=CommandResult.SUCCESS,

                    command="COMMAND_LONG",

                    message=(
                        "COMMAND_LONG sent"
                    ),

                    value=command,
                )

            return CommandResponse(
                result=CommandResult.REJECTED,

                command="COMMAND_LONG",

                message=(
                    "COMMAND_LONG rejected"
                ),

                value=command,
            )

        except Exception as exc:

            logger.exception(
                "[CommandRouter] "
                "COMMAND_LONG failed"
            )

            return CommandResponse(
                result=CommandResult.FAILED,

                command="COMMAND_LONG",

                message=str(exc),

                value=command,
            )

    # =========================================================================
    # Connection
    # =========================================================================

    @property
    def connected(
        self,
    ) -> bool:
        """
        MAVLink connection status.
        """

        return bool(
            getattr(
                self._mavlink,
                "connected",
                False,
            )
        )

    # =========================================================================
    # Running
    # =========================================================================

    @property
    def running(
        self,
    ) -> bool:

        with self._lock:

            return self._running

    # =========================================================================
    # Representation
    # =========================================================================

    def __repr__(
        self,
    ) -> str:

        with self._lock:

            return (
                "CommandRouter("
                f"running={self._running}, "
                f"connected={self.connected}"
                ")"
            )
