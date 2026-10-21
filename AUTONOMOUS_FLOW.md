# Autonomous SUAS Mission Flow

## Mission sequence

1. The application starts and connects to the flight controller when explicitly enabled.
2. The current GPS position is recorded as HOME.
3. The lap route is uploaded as a MAVLink AUTO mission.
4. The takeoff item uses the captured HOME coordinate.
5. The configured number of laps is flown through `LAP_WAYPOINTS`.
6. After the final lap item is reached, a new AUTO mission containing the mapping grid is uploaded.
7. The grid is generated inside `MAPPING_POLYGON`. At the transition, the first grid point is selected based on proximity to the aircraft's current position.
8. During mapping, YOLO detects the configured `TARGET_CLASS_NAMES`. For each target, the system stores the aircraft GPS position associated with the smallest observed image-center error.
9. When mapping is complete, target observations are saved to `data/target_positions.json`.
10. Each selected target is handled in sequence:
    - The aircraft navigates to the saved GPS position in GUIDED mode.
    - A P controller uses BODY_NED velocity commands to center the target in the image.
    - The associated payload servos are actuated.
    - An unconfirmed target is skipped by default. A polygon-center fallback is only permitted when `SKIP_UNCONFIRMED_TARGETS = False` and `FALLBACK_DROP_ENABLED = True`.
11. After target handling is complete, the aircraft returns to the captured HOME coordinate.
12. LAND mode is selected, and the application waits for the relative altitude to reach the ground threshold.

## Important settings

Configure these values in `config.py`:

- `LAP_COUNT`
- `LAP_WAYPOINTS`
- `MAPPING_POLYGON`
- `MAPPING_ALTITUDE_M`
- `TARGET_CLASS_NAMES`
- `TARGET_CLASS_IDS`, when verified class IDs are preferred
- `PAYLOAD_TARGET_CHANNELS`
- `YOLO_MODEL_PATH`
- `CAMERA_SOURCE`
- target-centering P-controller parameters

`LAP_WAYPOINTS` and `MAPPING_POLYGON` are separate so the lap and mapping areas can be configured independently.

## Target selection

For a detection normalized to image coordinates `(x, y)`, the center error is calculated as:

```text
center_error = sqrt((x - 0.5)^2 + (y - 0.5)^2)
```

A smaller error replaces the previously stored GPS position. The selected position therefore corresponds to the best observed image centering during the mapping pass.

## Payload mapping

`PAYLOAD_TARGET_CHANNELS` maps model class labels to Pixhawk output channels. Confirm every channel and PWM value against the actual hardware before connecting servos. Unconfirmed targets are skipped by default; fallback release is intentionally opt-in.

## Safety: test in SITL first

`AUTO_START_MISSION` is `False` by default. The entry point exits before opening a flight-controller connection until autonomous execution is explicitly enabled. Validate all configuration and mission transitions in ArduPilot SITL first.

## Route configuration

No site or competition coordinates are included in this repository. Before testing, configure:

- `LAP_WAYPOINTS`: the lap route as `(latitude, longitude)` pairs.
- `MAPPING_POLYGON`: at least three points defining the approved mapping area.
- `SKIP_UNCONFIRMED_TARGETS` and `FALLBACK_DROP_ENABLED`: determine whether unconfirmed targets are skipped or handled using the optional polygon-center fallback.

Use only an authorized test area, and validate the route in SITL before any outdoor flight.

### Return-to-start behaviour

For a finite lap route, the configured laps are followed by an explicit return to the first lap waypoint. The mission completes only after that point is reached. For example, a three-lap route with four points follows the route three times, then returns to its first point.
