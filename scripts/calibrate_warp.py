#!/usr/bin/env python3
"""Calibrate the bird's-eye warp and lane width on a straight.

Drag the sliders until both markings are vertical and parallel in the warped
pane; ``s`` prints a YAML block for your config, ``q`` quits.

python scripts/calibrate_warp.py --config configs/vehicle.yaml
python scripts/calibrate_warp.py --video docs/clips/straight.mp4
"""

from __future__ import annotations

import argparse
import sys

import cv2
import numpy as np

from bfmc.camera import create_source
from bfmc.config import AppConfig
from bfmc.perception import PerspectiveWarp, create_detector

WINDOW = "calibrate warp  [s] print yaml  [q] quit"

# name -> (config attribute, slider maximum), values held as percentages.
SLIDERS = {
    "horizon %": ("horizon_y_frac", 100),
    "top left %": ("top_left_x_frac", 100),
    "top right %": ("top_right_x_frac", 100),
    "bottom left %": ("bottom_left_x_frac", 100),
    "bottom right %": ("bottom_right_x_frac", 100),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-c", "--config", help="YAML config to start from")
    parser.add_argument("--video", help="calibrate against a clip instead of the camera")
    return parser


def parse_args(argv=None):
    return build_parser().parse_args(argv)


def as_yaml(warp_config, lane_width_px):
    lines = ["warp:"]
    for attribute, _max in SLIDERS.values():
        lines.append(f"  {attribute}: {getattr(warp_config, attribute):.2f}")
    lines += ["", "tracking:"]
    if lane_width_px is None:
        lines.append("  # lane_width_px: both lanes were never detected together")
    else:
        lines.append(f"  lane_width_px: {lane_width_px}")
        lines += ["", "metric:",
                  "  # measure these on the real track; they set every",
                  "  # physical quantity the controller uses",
                  "  lane_width_m: 0.35",
                  "  view_depth_m: 1.20"]
    return "\n".join(lines)


def main(argv=None) -> int:
    args = parse_args(argv)

    config = AppConfig.load(args.config)
    if args.video:
        config.camera.backend = "opencv"
        config.camera.source = args.video
        config.camera.loop_video = True

    source = create_source(config.camera)
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    for label, (attribute, maximum) in SLIDERS.items():
        initial = int(getattr(config.warp, attribute) * 100)
        cv2.createTrackbar(label, WINDOW, initial, maximum, lambda _value: None)

    lane_width_px = None
    try:
        while True:
            frame = source.read()
            if frame is None:
                break
            height, width = frame.shape[:2]

            for label, (attribute, _max) in SLIDERS.items():
                setattr(config.warp, attribute,
                        cv2.getTrackbarPos(label, WINDOW) / 100.0)

            warp = PerspectiveWarp(config.warp, width, height)
            # Rebuilt every frame, so no stale fit survives a warp change.
            detection = create_detector(warp, config.detection).detect(frame)

            preview = frame.copy()
            cv2.polylines(preview, [warp.src_points.astype(np.int32)], True, (0, 255, 255), 2)

            warped = detection.warped
            if detection.left is not None and detection.right is not None:
                lane_width_px = int(round(
                    detection.right.x_at(height) - detection.left.x_at(height)))
                for lane in (detection.left, detection.right):
                    cv2.polylines(warped, [lane.points()], False, (0, 255, 0), 3)
                label = f"lane width: {lane_width_px} px"
                color = (0, 255, 0)
            else:
                label = "lane width: both lanes not detected"
                color = (0, 165, 255)

            cv2.putText(warped, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.imshow(WINDOW, np.hstack((preview, warped)))

            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                print(as_yaml(config.warp, lane_width_px))
    except KeyboardInterrupt:
        pass
    finally:
        source.close()
        cv2.destroyAllWindows()

    print(as_yaml(config.warp, lane_width_px))
    return 0


if __name__ == "__main__":
    sys.exit(main())
