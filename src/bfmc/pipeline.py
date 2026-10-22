from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .camera import FrameSource, create_source
from .config import AppConfig
from .control import DriveCommand, SteeringController
from .overlay import compose, draw_birdseye, draw_hud, draw_lane_region
from .perception import LaneDetection, PerspectiveWarp, create_detector
from .telemetry import NullSpeedSensor, SpeedSensor
from .tracking import LaneSnapshot, LaneTracker

__all__ = ["FrameResult", "LaneKeepingPipeline"]

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class FrameResult:
    frame: np.ndarray
    detection: LaneDetection
    snapshot: LaneSnapshot
    command: DriveCommand
    speed_ms: float
    #: Seconds spent in :meth:`LaneKeepingPipeline.process`.
    latency_s: float = 0.0

    def summary(self) -> str:
        return (
            f"[{str(self.snapshot.state):10s}] "
            f"steer {self.command.steering_deg:+6.1f} deg  "
            f"v {self.speed_ms:4.2f} -> {self.command.target_speed_ms:4.2f} m/s  "
            f"off {self.snapshot.lateral_offset_m * 100:+5.1f} cm  "
            f"{self.snapshot.road_shape:11s} "
            f"conf {self.snapshot.confidence:.2f}  "
            f"{self.latency_s * 1000:4.1f} ms"
        )

    def as_row(self) -> dict:
        return {
            "state": str(self.snapshot.state),
            "steering_deg": round(self.command.steering_deg, 3),
            "target_speed_ms": round(self.command.target_speed_ms, 3),
            "speed_ms": round(self.speed_ms, 3),
            "lateral_offset_m": round(self.snapshot.lateral_offset_m, 4),
            "heading_rad": round(self.snapshot.heading_rad, 4),
            "curvature": round(self.snapshot.curvature, 4),
            "curve_radius_m": round(self.snapshot.curve_radius_m, 3),
            "min_curve_radius_m": round(self.snapshot.min_curve_radius_m, 3),
            "visible_distance_m": round(self.snapshot.visible_distance_m, 3),
            "lane_width_px": round(self.snapshot.lane_width_px, 1),
            "confidence": round(self.snapshot.confidence, 3),
            "feedforward_deg": round(self.command.feedforward_deg, 3),
            "feedback_deg": round(self.command.feedback_deg, 3),
            "lookahead_m": round(self.command.lookahead_m, 3),
            "latency_ms": round(self.latency_s * 1000, 2),
        }


class LaneKeepingPipeline:
    """Use as a context manager so the camera and serial thread are always released."""

    def __init__(self, config: AppConfig, source: FrameSource | None = None) -> None:
        self.config = config
        self.source = source if source is not None else create_source(config.camera)

        self.width, self.height = config.camera.width, config.camera.height
        self.vehicle_x = self.width // 2
        self._build_geometry(self.width, self.height)

        if config.encoder.enabled:
            self.speed_sensor = SpeedSensor(
                config.encoder.port,
                config.encoder.baud_rate,
                config.encoder.wheel_circumference_m,
            )
        else:
            self.speed_sensor = NullSpeedSensor()

        self._geometry_locked = False
        self._latencies: deque = deque(maxlen=30)

    def _build_geometry(self, width: int, height: int) -> None:
        config = self.config
        self.width, self.height = width, height
        self.vehicle_x = width // 2
        self.scale = config.metric_scale(height)

        self.warp = PerspectiveWarp(config.warp, width, height)
        self.detector = create_detector(self.warp, config.detection)
        self.tracker = LaneTracker(
            memory_frames=config.tracking.memory_frames,
            lane_width_px=config.tracking.lane_width_px,
            straight_radius_m=config.tracking.straight_radius_m,
            width_tolerance=config.tracking.width_tolerance,
            scale=self.scale,
        )
        self.controller = SteeringController(
            scale=self.scale, **vars(config.control), **vars(config.vehicle)
        )

    def __enter__(self) -> LaneKeepingPipeline:
        self.speed_sensor.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        self.speed_sensor.stop()
        self.source.close()

    def reset(self) -> None:
        self.detector.reset()
        self.tracker.reset()
        self.controller.reset()

    @property
    def fps(self) -> float:
        if not self._latencies:
            return 0.0
        mean = sum(self._latencies) / len(self._latencies)
        return 1.0 / mean if mean > 0 else 0.0

    def process(self, frame: np.ndarray) -> FrameResult:
        self._sync_geometry(frame)
        started = time.perf_counter()

        detection = self.detector.detect(frame)
        snapshot = self.tracker.update(
            detection.left, detection.right, self.height, self.vehicle_x
        )
        speed_ms = self.speed_sensor.speed_ms
        command = self.controller.update(snapshot, self.height, self.vehicle_x, speed_ms)

        latency = time.perf_counter() - started
        self._latencies.append(latency)
        return FrameResult(frame, detection, snapshot, command, speed_ms, latency)

    def render(self, result: FrameResult, show_mask: bool = False) -> np.ndarray:
        camera = result.frame.copy()
        draw_lane_region(camera, self.warp, result.snapshot)
        draw_hud(camera, result.snapshot, result.command, result.speed_ms,
                 self.fps, self.controller.max_steer_deg)

        birdseye = draw_birdseye(
            result.detection.warped.copy(), result.snapshot, result.command,
            self.vehicle_x, result.detection.left_pixels, result.detection.right_pixels,
        )
        panes = [camera, birdseye]
        if show_mask:
            panes.append(result.detection.mask)
        return compose(*panes)

    def run(
        self,
        on_result: Callable[[FrameResult], None] | None = None,
        show_mask: bool = False,
    ) -> int:
        """Until the source runs dry, ``q``, or Ctrl-C. Returns the number of frames processed."""
        import cv2

        frames = 0
        try:
            while True:
                frame = self.source.read()
                if frame is None:
                    log.info("Frame source exhausted")
                    break

                result = self.process(frame)
                frames += 1
                if on_result is not None:
                    on_result(result)

                if self.config.display:
                    cv2.imshow("bfmc lane keeping", self.render(result, show_mask))
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                elif on_result is None:
                    log.info("%s", result.summary())

                if self.config.max_frames and frames >= self.config.max_frames:
                    break
        except KeyboardInterrupt:
            log.info("Interrupted")
        finally:
            if self.config.display:
                cv2.destroyAllWindows()
        return frames

    def _sync_geometry(self, frame: np.ndarray) -> None:
        """The camera may not honour the requested size, and a mismatched warp silently ruins
        everything downstream.
        """
        if self._geometry_locked:
            return
        height, width = frame.shape[:2]
        if (width, height) != (self.width, self.height):
            log.warning("Camera returned %dx%d, not %dx%d; rebuilding geometry",
                        width, height, self.width, self.height)
            self._build_geometry(width, height)
        self._geometry_locked = True
