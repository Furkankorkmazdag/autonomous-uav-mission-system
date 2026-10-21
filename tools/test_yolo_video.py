#!/usr/bin/env python3
"""Run a YOLO model against a local video and display annotated frames.

Example:
    python tools/test_yolo_video.py --model models/yolo_model.pt --video /path/to/video.mp4

This is an offline diagnostic utility; it does not connect to MAVLink or control a vehicle.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    """Parse command-line options for a local YOLO video test."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path, help="Path to a YOLO model file")
    parser.add_argument("--video", required=True, type=Path, help="Path to an input video")
    parser.add_argument("--zoom", type=float, default=1.0, help="Digital center-crop zoom (>= 1.0)")
    parser.add_argument("--frame-skip", type=int, default=1, help="Process every Nth frame (>= 1)")
    parser.add_argument("--output", type=Path, help="Optional path for an annotated output video")
    return parser.parse_args()


def center_zoom(frame, factor: float):
    """Return a center-cropped frame resized to the original dimensions."""
    if factor <= 1.0:
        return frame
    height, width = frame.shape[:2]
    crop_width = max(1, int(width / factor))
    crop_height = max(1, int(height / factor))
    x0 = (width - crop_width) // 2
    y0 = (height - crop_height) // 2
    crop = frame[y0:y0 + crop_height, x0:x0 + crop_width]
    return cv2.resize(crop, (width, height), interpolation=cv2.INTER_LINEAR)


def main() -> int:
    """Validate inputs and run offline inference until the video ends or Q is pressed."""
    args = parse_args()
    if args.zoom < 1.0:
        raise SystemExit("--zoom must be greater than or equal to 1.0.")
    if args.frame_skip < 1:
        raise SystemExit("--frame-skip must be at least 1.")
    if not args.model.is_file():
        raise SystemExit(f"Model file not found: {args.model}")
    if not args.video.is_file():
        raise SystemExit(f"Video file not found: {args.video}")

    model = YOLO(str(args.model))
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise SystemExit(f"Could not open video: {args.video}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = None
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        codec = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(args.output), codec, fps / args.frame_skip, (width, height))
        if not writer.isOpened():
            capture.release()
            raise SystemExit(f"Could not create output video: {args.output}")

    frame_index = 0
    processed_frames = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_index += 1
            if (frame_index - 1) % args.frame_skip:
                continue

            inference_frame = center_zoom(frame, args.zoom)
            result = model.predict(inference_frame, verbose=False)[0]
            annotated = result.plot()
            processed_frames += 1

            if writer is not None:
                writer.write(annotated)
            cv2.imshow("YOLO Video Test - press Q to exit", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    print(f"Processed frames: {processed_frames}")
    if args.output:
        print(f"Output video: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
