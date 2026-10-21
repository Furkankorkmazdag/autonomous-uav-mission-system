# Model Files

Place authorized YOLO model weights in this directory (for example, `yolo_model.pt`) and set `YOLO_MODEL_PATH` in `config.py`. Model weights and compiled TensorRT engines are excluded from Git because they may be large and platform-specific. Build TensorRT `.engine` files for the target Jetson/GPU, TensorRT, and CUDA environment.
