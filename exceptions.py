"""Custom exceptions used by the SUAS mission software."""

from __future__ import annotations

from typing import Any, Optional


# ============================================================================
# Base Exception
# ============================================================================

class SUASError(Exception):

    def __init__(
        self,
        message: str = "",
        *,
        code: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> None:

        super().__init__(
            message
        )

        self.message = message

        self.code = code

        self.details = (
            details
            or {}
        )

    def __str__(self) -> str:

        if self.code:

            return (
                f"[{self.code}] "
                f"{self.message}"
            )

        return self.message


# ============================================================================
# Configuration Errors
# ============================================================================

class ConfigurationError(SUASError):

    pass


class InvalidConfigurationError(
    ConfigurationError
):

    pass


class MissingConfigurationError(
    ConfigurationError
):

    pass


# ============================================================================
# Lifecycle Errors
# ============================================================================

class LifecycleError(SUASError):

    pass


class ComponentStartError(
    LifecycleError
):

    pass


class ComponentStopError(
    LifecycleError
):

    pass


class ComponentAlreadyRunningError(
    LifecycleError
):

    pass


class ComponentNotRunningError(
    LifecycleError
):

    pass


# ============================================================================
# Communication Errors
# ============================================================================

class CommunicationError(SUASError):

    pass


class ConnectionError(
    CommunicationError
):

    pass


class ConnectionTimeoutError(
    ConnectionError
):

    pass


class MAVLinkError(
    CommunicationError
):

    pass


class MAVLinkCommandError(
    MAVLinkError
):

    pass


class MAVLinkTimeoutError(
    MAVLinkError
):
    """
    Timed out while waiting for a MAVLink response.
    """

    pass


class HeartbeatTimeoutError(
    MAVLinkError
):
    """
    MAVLink heartbeat timeout.
    """

    pass


# ============================================================================
# Mission Errors
# ============================================================================

class MissionError(SUASError):

    pass


class MissionNotFoundError(
    MissionError
):

    pass


class MissionStateError(
    MissionError
):

    pass


class MissionStartError(
    MissionError
):

    pass


class MissionStopError(
    MissionError
):

    pass


class MissionPauseError(
    MissionError
):
    """
    Mission pause edilemedi.
    """

    pass


class MissionResumeError(
    MissionError
):
    """
    Mission resume edilemedi.
    """

    pass


class MissionTimeoutError(
    MissionError
):

    pass


class MissionAbortedError(
    MissionError
):
    """
    The mission was aborted.
    """

    pass


class MissionConfigurationError(
    MissionError
):

    pass


# ============================================================================
# Resource Errors
# ============================================================================

class ResourceError(SUASError):

    pass


class ResourceNotFoundError(
    ResourceError
):

    pass


class ResourceBusyError(
    ResourceError
):

    pass


class ResourceAcquireTimeoutError(
    ResourceError
):
    """
    Timed out while waiting to acquire a resource.
    """

    pass


class ResourceOwnershipError(
    ResourceError
):

    pass


class ResourceDisabledError(
    ResourceError
):

    pass


# ============================================================================
# Safety Errors
# ============================================================================

class SafetyError(SUASError):

    pass


class BatteryLowError(
    SafetyError
):

    pass


class BatteryCriticalError(
    SafetyError
):

    pass


class GPSDegradedError(
    SafetyError
):

    pass


class GPSLostError(
    SafetyError
):
    """
    GPS signal was lost.
    """

    pass


class TelemetryLostError(
    SafetyError
):

    pass


class SafetyViolationError(
    SafetyError
):
    """
    A general safety violation occurred.
    """

    pass


# ============================================================================
# Payload Errors
# ============================================================================

class PayloadError(SUASError):

    pass


class PayloadNotReadyError(
    PayloadError
):

    pass


class PayloadReleaseError(
    PayloadError
):

    pass


class PayloadConfigurationError(
    PayloadError
):

    pass


class ServoError(
    PayloadError
):

    pass


class ServoTimeoutError(
    ServoError
):
    """
    Servo command timeout.
    """

    pass


# ============================================================================
# Hardware Errors
# ============================================================================

class HardwareError(SUASError):

    pass


class SensorError(
    HardwareError
):

    pass


class CameraError(
    HardwareError
):

    pass


class GPIOError(
    HardwareError
):

    pass


class MotorError(
    HardwareError
):

    pass


# ============================================================================
# Navigation Errors
# ============================================================================

class NavigationError(SUASError):

    pass


class NavigationNotReadyError(
    NavigationError
):

    pass


class NavigationTargetError(
    NavigationError
):

    pass


class NavigationTimeoutError(
    NavigationError
):
    """
    Navigation operation timeout.
    """

    pass


# ============================================================================
# Mapping Errors
# ============================================================================

class MappingError(SUASError):

    pass


class MappingConfigurationError(
    MappingError
):

    pass


class MappingAreaError(
    MappingError
):

    pass


class MappingGridError(
    MappingError
):

    pass


# ============================================================================
# Object Detection Errors
# ============================================================================

class ObjectDetectionError(
    SUASError
):

    pass


class ModelLoadError(
    ObjectDetectionError
):

    pass


class DetectionError(
    ObjectDetectionError
):

    pass


class InferenceTimeoutError(
    ObjectDetectionError
):
    """
    AI inference timeout.
    """

    pass


# ============================================================================
# Event Errors
# ============================================================================

class EventError(SUASError):

    pass


class EventSubscriptionError(
    EventError
):

    pass


class EventDispatchError(
    EventError
):

    pass


# ============================================================================
# Scheduler Errors
# ============================================================================

class SchedulerError(SUASError):

    pass


class SchedulerQueueFullError(
    SchedulerError
):
    """
    Scheduler queue dolu.
    """

    pass


class SchedulerConflictError(
    SchedulerError
):
    """
    Scheduler mission conflict.
    """

    pass


class SchedulerPreemptionError(
    SchedulerError
):

    pass


# ============================================================================
# Watchdog Errors
# ============================================================================

class WatchdogError(SUASError):

    pass


class MissionFreezeError(
    WatchdogError
):

    pass


class MissionRestartError(
    WatchdogError
):

    pass


# ============================================================================
# Helper functions
# ============================================================================

def is_suas_error(
    exception: BaseException,
) -> bool:
    """
    Check whether exception belongs to SUAS hierarchy.
    """

    return isinstance(
        exception,
        SUASError,
    )


def get_error_code(
    exception: BaseException,
) -> Optional[str]:
    """
    Return SUAS error code if available.
    """

    if isinstance(
        exception,
        SUASError,
    ):

        return exception.code

    return None


def get_error_details(
    exception: BaseException,
) -> dict[str, Any]:
    """
    Return structured error details.
    """

    if isinstance(
        exception,
        SUASError,
    ):

        return dict(
            exception.details
        )

    return {}


def format_error(
    exception: BaseException,
) -> str:
    """
    Convert exception to structured human-readable text.
    """

    if isinstance(
        exception,
        SUASError,
    ):

        if exception.code:

            return (
                f"[{exception.code}] "
                f"{exception.message}"
            )

        return exception.message

    return str(
        exception
    )
