"""Closed-loop track simulator: render the camera view, run the pipeline, move a bicycle model.

The view is drawn in bird's-eye space and pushed through the inverse warp, so it
is consistent with what the pipeline will undo. Cross-track error is exact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import AppConfig
from .geometry import MetricScale
from .perception import PerspectiveWarp

__all__ = ["Track", "oval_track", "figure_eight_track", "CarState", "TrackSimulator"]

#: Metres of arc length between samples.
SAMPLE_STEP_M = 0.02


@dataclass(frozen=True)
class Track:
    points: np.ndarray
    #: Radians.
    headings: np.ndarray
    lane_width_m: float = 0.35
    closed: bool = False

    @property
    def length_m(self) -> float:
        return len(self.points) * SAMPLE_STEP_M

    @classmethod
    def from_profile(
        cls,
        segments: list[tuple[float, float]],
        lane_width_m: float = 0.35,
        closed: bool = False,
        start_heading: float = math.pi / 2,
    ) -> Track:
        """Segments are ``(length_m, curvature_1_per_m)``; positive curvature bends left."""
        xs, ys, headings = [0.0], [0.0], [start_heading]
        x, y, heading = 0.0, 0.0, start_heading

        for length_m, curvature in segments:
            steps = max(1, int(round(length_m / SAMPLE_STEP_M)))
            for _ in range(steps):
                heading += curvature * SAMPLE_STEP_M
                x += math.cos(heading) * SAMPLE_STEP_M
                y += math.sin(heading) * SAMPLE_STEP_M
                xs.append(x)
                ys.append(y)
                headings.append(heading)

        return cls(
            points=np.stack([xs, ys], axis=1),
            headings=np.asarray(headings),
            lane_width_m=lane_width_m,
            closed=closed,
        )

    def nearest(self, position: np.ndarray) -> tuple[int, float]:
        """Signed offset is positive when the car is right of the centreline."""
        deltas = self.points - np.asarray(position, float)
        index = int(np.argmin((deltas ** 2).sum(axis=1)))

        heading = self.headings[index]
        # Right of the path is the heading rotated by -90 degrees.
        right = np.array([math.sin(heading), -math.cos(heading)])
        return index, float(-deltas[index] @ right)

    def boundary_points(self, side: int) -> np.ndarray:
        normals = np.stack([np.sin(self.headings), -np.cos(self.headings)], axis=1)
        return self.points + normals * (side * self.lane_width_m / 2.0)


def oval_track(straight_m: float = 3.0, radius_m: float = 1.2,
               lane_width_m: float = 0.35) -> Track:
    curvature = 1.0 / radius_m
    half_circle = math.pi * radius_m
    return Track.from_profile(
        [(straight_m, 0.0), (half_circle, curvature),
         (straight_m, 0.0), (half_circle, curvature)],
        lane_width_m=lane_width_m, closed=True,
    )


def figure_eight_track(straight_m: float = 1.5, radius_m: float = 1.0,
                       lane_width_m: float = 0.35) -> Track:
    """Turns both ways, which an asymmetric bug will fail."""
    curvature = 1.0 / radius_m
    arc = math.pi * 1.5 * radius_m
    return Track.from_profile(
        [(straight_m, 0.0), (arc, curvature),
         (straight_m, 0.0), (arc, -curvature)],
        lane_width_m=lane_width_m,
    )


@dataclass
class CarState:
    x: float = 0.0
    y: float = 0.0
    #: Radians from the world x axis.
    yaw: float = math.pi / 2
    speed_ms: float = 0.0
    odometer_m: float = 0.0

    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y])


@dataclass
class TrackSimulator:
    track: Track
    config: AppConfig
    car: CarState = field(default_factory=CarState)

    #: Noise and brightness drift separate a detector that works from one that only
    #: works on a clean synthetic image.
    marking_width_px: int = 8
    dashed_center: bool = True
    noise_sigma: float = 6.0
    brightness: int = 40
    road_gray: int = 55
    marking_gray: int = 235
    render_depth_m: float = 2.5
    seed: int = 0

    def __post_init__(self) -> None:
        self.width = self.config.camera.width
        self.height = self.config.camera.height
        self.warp = PerspectiveWarp(self.config.warp, self.width, self.height)
        self.scale: MetricScale = self.config.metric_scale(self.height)
        self.vehicle_x = self.width // 2
        self._rng = np.random.default_rng(self.seed)
        self.place_on_track(0)

    def place_on_track(self, index: int = 0, lateral_offset_m: float = 0.0,
                       heading_error_rad: float = 0.0) -> None:
        heading = float(self.track.headings[index])
        right = np.array([math.sin(heading), -math.cos(heading)])
        position = self.track.points[index] + right * lateral_offset_m
        self.car = CarState(
            x=float(position[0]), y=float(position[1]),
            yaw=heading + heading_error_rad,
        )

    def step(self, steering_deg: float, target_speed_ms: float, dt: float = 0.05) -> None:
        wheelbase = self.config.vehicle.wheelbase_m
        limit = self.config.vehicle.max_steer_deg
        steering = math.radians(float(np.clip(steering_deg, -limit, limit)))

        # First-order lag: an instant speed change would hide longitudinal tuning
        # problems.
        self.car.speed_ms += (target_speed_ms - self.car.speed_ms) * min(1.0, dt * 4.0)

        distance = self.car.speed_ms * dt
        self.car.x += math.cos(self.car.yaw) * distance
        self.car.y += math.sin(self.car.yaw) * distance
        # Positive steering is clockwise, so it decreases yaw.
        self.car.yaw -= distance / wheelbase * math.tan(steering)
        self.car.odometer_m += distance

    @property
    def cross_track_error_m(self) -> float:
        return self.track.nearest(self.car.position)[1]

    @property
    def on_track(self) -> bool:
        return abs(self.cross_track_error_m) <= self.track.lane_width_m

    @property
    def at_end_of_track(self) -> bool:
        """Running out of an open course is a finished run, not a failure."""
        if self.track.closed:
            return False
        index, _ = self.track.nearest(self.car.position)
        return index >= len(self.track.points) - int(self.render_depth_m / SAMPLE_STEP_M)

    def render(self) -> np.ndarray:
        birdseye = np.full((self.height, self.width, 3), self.road_gray, np.uint8)

        for side in (-1, 1):
            for run in self._to_birdseye(self.track.boundary_points(side)):
                self._draw_polyline(birdseye, run, dashed=False)

        if self.dashed_center:
            for run in self._to_birdseye(self.track.points):
                self._draw_polyline(birdseye, run, dashed=True)

        camera = cv2.warpPerspective(
            birdseye, self.warp.inverse_matrix, (self.width, self.height)
        )
        return self._add_realism(camera)

    def _to_birdseye(self, world_points: np.ndarray) -> list:
        """Points ahead and within the frame sideways (a closed circuit often has track
        straight ahead but metres off to the side), split into contiguous runs so no
        polyline bridges a gap with a marking that is not there.
        """
        delta = world_points - self.car.position
        forward = np.array([math.cos(self.car.yaw), math.sin(self.car.yaw)])
        right = np.array([math.sin(self.car.yaw), -math.cos(self.car.yaw)])

        ahead = delta @ forward
        lateral = delta @ right

        # Plus a margin so a marking leaving the frame is drawn up to the edge.
        half_width_m = self.vehicle_x * self.scale.m_per_px_x * 1.5
        visible = (
            (ahead > 0.0) & (ahead < self.render_depth_m)
            & (np.abs(lateral) < half_width_m)
        )
        indices = np.nonzero(visible)[0]
        if indices.size == 0:
            return []

        xs = self.vehicle_x + lateral[indices] / self.scale.m_per_px_x
        ys = self.height - ahead[indices] / self.scale.m_per_px_y
        pixels = np.stack([xs, ys], axis=1).astype(np.int32)

        breaks = np.nonzero(np.diff(indices) != 1)[0] + 1
        return [run for run in np.split(pixels, breaks) if len(run) >= 2]

    def _draw_polyline(self, canvas: np.ndarray, pixels: np.ndarray, dashed: bool) -> None:
        if len(pixels) < 2:
            return
        color = (self.marking_gray,) * 3
        if not dashed:
            cv2.polylines(canvas, [pixels], False, color, self.marking_width_px)
            return

        # In samples, which are a fixed arc length apart.
        dash = max(2, int(0.12 / SAMPLE_STEP_M))
        for start in range(0, len(pixels) - 1, dash * 2):
            segment = pixels[start:start + dash]
            if len(segment) >= 2:
                cv2.polylines(canvas, [segment], False, color, self.marking_width_px)

    def _add_realism(self, image: np.ndarray) -> np.ndarray:
        result = image.astype(np.int16)
        if self.brightness:
            # Slow drift along the lap, so a run passes through good and bad light.
            phase = math.sin(self.car.odometer_m * 0.6)
            result += int(self.brightness * phase)
        if self.noise_sigma:
            result += self._rng.normal(0.0, self.noise_sigma, result.shape).astype(np.int16)
        return np.clip(result, 0, 255).astype(np.uint8)
