# SUAS Autonomous UAV Mission Software

A modular Python application for autonomous UAV mission execution using MAVLink and ArduPilot. The project combines route following, grid-based area mapping, YOLO object detection, target-position tracking, payload-servo control, telemetry, and application lifecycle/safety support.

> **Flight safety:** This repository is not a flight-ready configuration. Autonomous startup is disabled, route coordinates are empty, camera integrations are disabled, and the default MAVLink endpoint targets local SITL. Configure and validate all settings in simulation before connecting real hardware. Operate only in an authorized test area and follow the applicable flight-safety procedures.

## Features

- **MAVLink integration:** telemetry, vehicle status, mission upload, and flight-mode commands.
- **Mission orchestration:** lap routes, mapping grids, target visits, return-to-home, and landing.
- **Mapping and perception:** grid generation, YOLO-based detections, image capture, and GPS metadata.
- **Target centering:** image-center error converted to BODY_NED velocity commands.
- **Payload control:** target-to-servo channel mapping, PWM configuration, and sequenced release.
- **Runtime reliability:** lifecycle management, health monitoring, resource management, watchdogs, and shutdown helpers.
- **Optional SIYI camera support:** RTSP recording and gimbal control through GStreamer.

## Architecture

```text
main.py
  └── app.py                     Application composition and lifecycle
      ├── core/                  Events, mission manager, scheduler, safety, watchdog
      ├── mavlink/               Connection, commands, message routing, mission protocol
      ├── missions/              Autonomous mission, lap, mapping, waypoint, detection
      ├── mapping/               Mapping grid and route generation
      ├── navigation/            GPS, distance, target tracking and centering
      ├── perception/            Camera, YOLO, frames, metadata and photo capture
      ├── payload/               Servo driver and payload manager
      ├── telemetry/             Vehicle telemetry
      └── hardware/              Platform-specific hardware abstractions
```

## Requirements

- Linux/Ubuntu is recommended.
- Python 3.10 or later.
- An ArduPilot SITL environment for simulation testing.
- PyTorch and Ultralytics for YOLO inference. CUDA/TensorRT setup depends on the GPU and JetPack/runtime versions.
- System GStreamer packages for the optional SIYI A8 Mini integration.

## Installation

Replace `USERNAME` and `REPOSITORY` with your GitHub account and repository name:

```bash
git clone https://github.com/USERNAME/REPOSITORY.git
cd REPOSITORY
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

On Jetson/CUDA systems, install a PyTorch/TorchVision build that matches the device and JetPack release **before** installing the remaining requirements:

```bash
pip install -r requirements.txt
```

For optional GStreamer integration on Ubuntu, system packages may be required. Package names can vary by Ubuntu release:

```bash
sudo apt install python3-gi gir1.2-gstreamer-1.0 gir1.2-glib-2.0 \
  gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
  gstreamer1.0-plugins-bad
```

If a virtual environment needs the system `gi` bindings, create it with `--system-site-packages` or configure a compatible Python environment deliberately.

## Configuration

Mission and hardware settings are centralized in `config.py`. The most important settings are:

| Setting | Purpose |
|---|---|
| `AUTO_START_MISSION` | Must remain `False` until configuration and SITL checks pass. |
| `MAVLINK_CONNECTION` | MAVLink endpoint. The default is a local UDP endpoint for SITL. |
| `LAP_WAYPOINTS` | Lap route as `(latitude, longitude)` pairs; empty by default. |
| `MAPPING_POLYGON` | Mapping boundary as at least three GPS points; empty by default. |
| `TAKEOFF_ALTITUDE_M`, `MAPPING_ALTITUDE_M`, `MAX_FLIGHT_ALTITUDE_M` | Flight-altitude settings in meters. |
| `YOLO_MODEL_PATH` | Path to the trained model under `models/`. Model weights are not included. |
| `SIYI_A8_MINI_ENABLED`, `CAMERA_SOURCE` | Optional camera configuration. Integrations are disabled by default. |
| `PAYLOAD_TARGET_CHANNELS` | Mapping from model class labels to Pixhawk servo output channels. |
| `SKIP_UNCONFIRMED_TARGETS`, `FALLBACK_DROP_ENABLED` | Controls behaviour when a target is not confirmed. Fallback release is disabled by default. |
| `SAFETY_SCRIPT_PATH` | Optional restart script path. Empty means no automatic restart script is launched. |

The current model labels are `manken` and `tente`. Keep `TARGET_CLASS_NAMES` aligned with the class labels produced by your trained model; changing those labels without updating the model can break target matching.

No site or competition coordinates are included in this public repository. Add only coordinates for an authorized test area. Verify latitude/longitude order and validate the route in SITL before switching to a real serial connection. Bench-test servo channels and PWM values. Also test return-to-home, landing, and link-loss behaviour in simulation.

## Run

The default configuration does not start a mission:

```bash
python main.py
```

The application logs a warning and exits before importing hardware-dependent components. To opt in to autonomous execution, complete the route, model, altitude, camera, and payload settings in `config.py`, pass static validation, and test the mission in ArduPilot SITL. Only then should `AUTO_START_MISSION` be deliberately set to `True`.

## Offline YOLO video test

Use `tools/test_yolo_video.py` to inspect detections in a video without opening a MAVLink connection or controlling a vehicle:

```bash
python tools/test_yolo_video.py \
  --model models/yolo_model.pt \
  --video /path/to/video.mp4 \
  --zoom 1.0 \
  --frame-skip 1 \
  --output output/annotated.mp4
```

## Documentation

- [`AUTONOMOUS_FLOW.md`](AUTONOMOUS_FLOW.md): mission sequence and target selection.
- [`MISSION_SPEEDS_AND_TARGETS.md`](MISSION_SPEEDS_AND_TARGETS.md): speed settings and target handling.
- [`models/README.md`](models/README.md): model-weight management.

## Checks

Run the syntax check and dependency-free startup tests from the repository root:

```bash
python -m compileall -q .
python -m unittest discover -s tests -v
```

These checks do not validate hardware-in-the-loop behaviour or a real flight. Test mission logic in SITL, then validate it in a controlled and authorized field environment using appropriate safety procedures.

## Models, outputs, and licensing

- Model weights (`.pt`, `.engine`, `.onnx`), videos, photos, logs, and runtime outputs are excluded by `.gitignore`. Store large files in a suitable location with appropriate access control.
- TensorRT `.engine` files are generally tied to the target GPU, TensorRT, and CUDA environment; do not assume they are portable between devices.
- No license is included. Before publishing the project as open source, review the copyright and third-party dependency terms and add an appropriate license.
