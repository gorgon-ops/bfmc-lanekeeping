"""Bird's-eye warp and two detectors: sliding-window (follows curves) and Hough (straight-line
baseline).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import cv2
import numpy as np

from .config import DetectionConfig, HoughConfig, WarpConfig
from .fitting import split_by_slope, weighted_average_lane
from .geometry import LanePolynomial
from .polyfit import SlidingWindowFinder
from .thresholding import lane_mask

__all__ = ["PerspectiveWarp", "LaneDetection", "Detector",
           "SlidingWindowDetector", "HoughDetector", "create_detector"]


class PerspectiveWarp:
    def __init__(self, config: WarpConfig, width: int, height: int) -> None:
        self.width = width
        self.height = height
        horizon = int(height * config.horizon_y_frac)
        src = np.float32([
            [int(width * config.top_left_x_frac), horizon],
            [int(width * config.top_right_x_frac), horizon],
            [int(width * config.bottom_right_x_frac), height],
            [int(width * config.bottom_left_x_frac), height],
        ])
        dst = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
        self.matrix = cv2.getPerspectiveTransform(src, dst)
        self.inverse_matrix = cv2.getPerspectiveTransform(dst, src)
        self.src_points = src

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        return cv2.warpPerspective(frame, self.matrix, (self.width, self.height))

    def unwarp_points(self, points: np.ndarray) -> np.ndarray:
        reshaped = np.asarray(points, np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(reshaped, self.inverse_matrix).reshape(-1, 2)


@dataclass(frozen=True)
class LaneDetection:
    warped: np.ndarray
    mask: np.ndarray
    left: LanePolynomial | None = None
    right: LanePolynomial | None = None
    left_pixels: np.ndarray = field(default_factory=lambda: np.empty((0, 2), int))
    right_pixels: np.ndarray = field(default_factory=lambda: np.empty((0, 2), int))
    tracked: bool = False


class Detector(Protocol):
    warp: PerspectiveWarp

    def detect(self, frame: np.ndarray) -> LaneDetection:
        ...

    def reset(self) -> None:
        ...


class SlidingWindowDetector:
    def __init__(self, warp: PerspectiveWarp, config: DetectionConfig) -> None:
        self.warp = warp
        self.config = config
        self.finder = SlidingWindowFinder(**vars(config.window))
        self.reset()

    def reset(self) -> None:
        self._prior_left: LanePolynomial | None = None
        self._prior_right: LanePolynomial | None = None

    def detect(self, frame: np.ndarray) -> LaneDetection:
        warped = self.warp(frame)
        mask = lane_mask(warped, **vars(self.config.mask))

        result = self.finder.find(mask, self._prior_left, self._prior_right)
        self._prior_left, self._prior_right = result.left, result.right

        return LaneDetection(
            warped=warped,
            mask=mask,
            left=result.left,
            right=result.right,
            left_pixels=result.left_pixels,
            right_pixels=result.right_pixels,
            tracked=result.tracked,
        )


class HoughDetector:
    def __init__(self, warp: PerspectiveWarp, config: HoughConfig) -> None:
        self.warp = warp
        self.config = config

    def reset(self) -> None:
        pass

    def detect(self, frame: np.ndarray) -> LaneDetection:
        cfg = self.config
        warped = self.warp(frame)
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (cfg.blur_kernel | 1, cfg.blur_kernel | 1), 0)
        edges = cv2.Canny(blurred, cfg.canny_low, cfg.canny_high)

        segments = cv2.HoughLinesP(
            edges, 1, np.pi / 180, cfg.threshold,
            minLineLength=cfg.min_line_length, maxLineGap=cfg.max_line_gap,
        )
        # Newer OpenCV returns (N, 4) rather than (N, 1, 4).
        flat = segments.reshape(-1, 4) if segments is not None else []
        raw = [tuple(int(v) for v in s) for s in flat]

        left_segments, right_segments = split_by_slope(raw, cfg.min_slope)
        height = warped.shape[0]

        def fit(group):
            line = weighted_average_lane(
                group, height, cfg.min_segment_length, cfg.fit_min_slope, cfg.lane_top_frac
            )
            if line is None:
                return None
            # More supporting segments, more trust.
            confidence = float(np.clip(len(group) / 6.0, 0.2, 1.0))
            return LanePolynomial.from_line(line, confidence)

        return LaneDetection(warped=warped, mask=edges,
                             left=fit(left_segments), right=fit(right_segments))


def create_detector(warp: PerspectiveWarp, config: DetectionConfig) -> Detector:
    if config.method == "sliding_window":
        return SlidingWindowDetector(warp, config)
    if config.method == "hough":
        return HoughDetector(warp, config.hough)
    raise ValueError(f"unknown detection method: {config.method!r}")
