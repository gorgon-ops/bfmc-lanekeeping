"""Lane memory, single-boundary recovery and a width plausibility check.

Two fits implausibly close together usually means both searches locked onto one
marking, which is worse than seeing nothing; the weaker one is rebuilt from the
stronger.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .geometry import LanePolynomial, MetricScale, lane_center
from .states import DrivingState

__all__ = ["LaneSnapshot", "LaneTracker"]


@dataclass(frozen=True)
class LaneSnapshot:
    left: LanePolynomial | None
    right: LanePolynomial | None
    center: LanePolynomial | None
    state: DrivingState = DrivingState.STOP

    left_stale: bool = False
    right_stale: bool = False
    left_inferred: bool = False
    right_inferred: bool = False
    left_age: int = 0
    right_age: int = 0

    #: 1/m, positive for a right-hand bend.
    curvature: float = 0.0
    #: Metres; infinite on a straight.
    curve_radius_m: float = math.inf
    #: Tightest radius in view, so the car slows before the corner.
    min_curve_radius_m: float = math.inf
    road_shape: str = "STRAIGHT"

    #: Metres, positive when the car is right of centre.
    lateral_offset_m: float = 0.0
    #: Radians, positive when the road runs off to the right.
    heading_rad: float = 0.0

    #: Metres the centreline fit reaches: the sensing horizon, which shrinks in
    #: corners and pulls the speed limit down with it.
    visible_distance_m: float = 0.0
    lane_width_px: float = 0.0
    #: Of the weaker boundary, 0 to 1.
    confidence: float = 0.0
    width_rejected: bool = False

    @property
    def has_both(self) -> bool:
        return self.left is not None and self.right is not None


class LaneTracker:
    def __init__(
        self,
        memory_frames: int = 15,
        lane_width_px: float = 320.0,
        straight_radius_m: float = 6.0,
        width_tolerance: float = 0.35,
        scale: MetricScale | None = None,
    ) -> None:
        self.memory_frames = memory_frames
        self.lane_width_px = lane_width_px
        self.straight_radius_m = straight_radius_m
        self.width_tolerance = width_tolerance
        self.scale = scale if scale is not None else MetricScale()
        self.reset()

    def reset(self) -> None:
        self._mem_left: LanePolynomial | None = None
        self._mem_right: LanePolynomial | None = None
        self._left_age = 0
        self._right_age = 0

    def update(
        self,
        left: LanePolynomial | None,
        right: LanePolynomial | None,
        height: int,
        vehicle_x: float,
    ) -> LaneSnapshot:
        left, right, width_rejected = self._check_width(left, right, height)

        if left is not None:
            self._mem_left, self._left_age = left, 0
        else:
            self._left_age += 1

        if right is not None:
            self._mem_right, self._right_age = right, 0
        else:
            self._right_age += 1

        active_left = left if left is not None else self._remembered(
            self._mem_left, self._left_age)
        active_right = right if right is not None else self._remembered(
            self._mem_right, self._right_age)

        left_stale = left is None and active_left is not None
        right_stale = right is None and active_right is not None

        left_inferred = right_inferred = False
        if active_left is None and active_right is not None:
            # Only the right boundary is known; the left runs parallel to it.
            active_left = active_right.shifted(-self.lane_width_px)
            left_inferred = True
        elif active_right is None and active_left is not None:
            active_right = active_left.shifted(self.lane_width_px)
            right_inferred = True

        state = self._classify(active_left, active_right, left_inferred,
                               right_inferred, left_stale, right_stale)

        if active_left is None or active_right is None:
            return LaneSnapshot(
                left=active_left, right=active_right, center=None, state=state,
                left_stale=left_stale, right_stale=right_stale,
                left_age=self._left_age, right_age=self._right_age,
                width_rejected=width_rejected,
            )

        center = lane_center(active_left, active_right)
        return LaneSnapshot(
            left=active_left,
            right=active_right,
            center=center,
            state=state,
            left_stale=left_stale,
            right_stale=right_stale,
            left_inferred=left_inferred,
            right_inferred=right_inferred,
            left_age=self._left_age,
            right_age=self._right_age,
            width_rejected=width_rejected,
            **self._describe(center, active_left, active_right, height, vehicle_x),
        )

    def _remembered(self, lane, age):
        return lane if age <= self.memory_frames else None

    def _check_width(self, left, right, height):
        if left is None or right is None:
            return left, right, False

        width = right.x_at(height) - left.x_at(height)
        error = abs(width - self.lane_width_px) / self.lane_width_px
        if width > 0 and error <= self.width_tolerance:
            return left, right, False

        # Keep the better-supported fit and rebuild the other from it.
        if left.confidence >= right.confidence:
            return left, None, True
        return None, right, True

    def _describe(self, center, left, right, height, vehicle_x) -> dict:
        curvature = center.mean_curvature(self.scale)
        radius = math.inf if curvature == 0.0 else abs(1.0 / curvature)

        if radius >= self.straight_radius_m:
            road_shape = "STRAIGHT"
        else:
            road_shape = "CURVE RIGHT" if curvature > 0 else "CURVE LEFT"

        return {
            "curvature": curvature,
            "curve_radius_m": radius,
            "min_curve_radius_m": center.tightest_radius_m(self.scale),
            "road_shape": road_shape,
            # Positive right of centre, matching the steering convention.
            "lateral_offset_m": (vehicle_x - center.x_at(height)) * self.scale.m_per_px_x,
            "heading_rad": center.heading_rad(height, self.scale),
            "visible_distance_m": (center.y_max - center.y_min) * self.scale.m_per_px_y,
            "lane_width_px": right.x_at(height) - left.x_at(height),
            "confidence": min(left.confidence, right.confidence),
        }

    @staticmethod
    def _classify(left, right, left_inferred, right_inferred, left_stale, right_stale):
        if left is None and right is None:
            return DrivingState.STOP
        if left_inferred:
            # The left line is a guess.
            return DrivingState.RIGHT_ONLY
        if right_inferred:
            return DrivingState.LEFT_ONLY
        if left_stale or right_stale:
            return DrivingState.COASTING
        return DrivingState.LANE_KEEPING
