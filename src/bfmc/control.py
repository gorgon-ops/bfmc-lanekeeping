"""Steering = curvature feedforward + pure pursuit or Stanley feedback; speed from curvature.

Pure pursuit already follows curvature through its goal point, so adding the
feedforward commands the same turn twice; Stanley only regulates error and
needs it. ``feedforward="auto"`` pairs them (see README for the lap numbers).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, lfilter, lfilter_zi

from .geometry import MetricScale
from .states import DrivingState
from .tracking import LaneSnapshot

__all__ = ["DriveCommand", "ButterworthSmoother", "SteeringController",
           "pure_pursuit_angle", "stanley_angle", "curvature_speed_limit"]

PURE_PURSUIT = "pure_pursuit"
STANLEY = "stanley"
LAWS = (PURE_PURSUIT, STANLEY)


@dataclass(frozen=True)
class DriveCommand:
    #: Degrees, positive = steer right.
    steering_deg: float
    #: m/s; zero means stop.
    target_speed_ms: float

    #: Before smoothing and the soft limit.
    raw_steering_deg: float = 0.0
    feedforward_deg: float = 0.0
    feedback_deg: float = 0.0
    #: Metres actually used this frame.
    lookahead_m: float = 0.0
    #: Warped pixels; pure pursuit only.
    goal_point: tuple | None = None
    law: str = PURE_PURSUIT


class ButterworthSmoother:
    """Low-order Butterworth: removes per-frame fit jitter with less lag than a moving average."""

    def __init__(self, order: int = 2, cutoff: float = 0.15) -> None:
        if not 0.0 < cutoff < 1.0:
            raise ValueError("cutoff must be a fraction of Nyquist in (0, 1)")
        self.order = order
        self.cutoff = cutoff
        self._b, self._a = butter(order, cutoff, btype="low", analog=False)
        self._steady_state = lfilter_zi(self._b, self._a)
        self.reset()

    def reset(self, value: float = 0.0) -> None:
        self._zi = self._steady_state * value

    def __call__(self, sample: float) -> float:
        filtered, self._zi = lfilter(self._b, self._a, [sample], zi=self._zi)
        return float(filtered[0])


def pure_pursuit_angle(
    lateral_error_m: float, lookahead_m: float, wheelbase_m: float
) -> float:
    """``delta = atan(2 L sin(alpha) / ld)``, radians, with ``sin(alpha) = lateral_error / ld``."""
    if lookahead_m <= 1e-6:
        return 0.0
    return math.atan2(2.0 * wheelbase_m * lateral_error_m, lookahead_m ** 2)


def stanley_angle(
    heading_error_rad: float,
    cross_track_error_m: float,
    speed_ms: float,
    gain: float,
    softening_ms: float,
) -> float:
    """``delta = heading_error + atan(k * e / (v + eps))``, radians; eps keeps it finite at rest."""
    return heading_error_rad + math.atan2(
        gain * -cross_track_error_m, softening_ms + max(speed_ms, 0.0)
    )


def curvature_speed_limit(radius_m: float, max_lateral_accel_ms2: float) -> float:
    if not math.isfinite(radius_m):
        return math.inf
    return math.sqrt(max(0.0, max_lateral_accel_ms2 * radius_m))


class SteeringController:
    def __init__(
        self,
        scale: MetricScale | None = None,
        law: str = PURE_PURSUIT,
        wheelbase_m: float = 0.26,
        max_steer_deg: float = 30.0,
        max_lateral_accel_ms2: float = 2.0,
        min_speed_ms: float = 0.15,
        max_speed_ms: float = 1.20,
        lookahead_base_m: float = 0.25,
        lookahead_time_s: float = 0.45,
        lookahead_min_m: float = 0.18,
        lookahead_max_m: float = 0.90,
        pursuit_gain: float = 1.0,
        stanley_gain: float = 1.4,
        stanley_softening_ms: float = 0.4,
        heading_gain: float = 1.0,
        feedforward: bool | str = "auto",
        assumed_speed_ms: float = 0.4,
        horizon_time_s: float = 1.1,
        butter_order: int = 2,
        butter_cutoff: float = 0.15,
    ) -> None:
        if law not in LAWS:
            raise ValueError(f"unknown control law {law!r}; expected one of {LAWS}")
        self.scale = scale if scale is not None else MetricScale()
        self.law = law
        self.wheelbase_m = wheelbase_m
        self.max_steer_deg = max_steer_deg
        self.max_lateral_accel_ms2 = max_lateral_accel_ms2
        self.min_speed_ms = min_speed_ms
        self.max_speed_ms = max_speed_ms
        self.lookahead_base_m = lookahead_base_m
        self.lookahead_time_s = lookahead_time_s
        self.lookahead_min_m = lookahead_min_m
        self.lookahead_max_m = lookahead_max_m
        self.pursuit_gain = pursuit_gain
        self.stanley_gain = stanley_gain
        self.stanley_softening_ms = stanley_softening_ms
        self.heading_gain = heading_gain
        # 'auto' is the only setting right for both laws.
        self.feedforward = (law == STANLEY) if feedforward == "auto" else bool(feedforward)
        self.assumed_speed_ms = assumed_speed_ms
        self.horizon_time_s = horizon_time_s
        self.smoother = ButterworthSmoother(butter_order, butter_cutoff)

    def reset(self) -> None:
        self.smoother.reset()

    def update(
        self,
        snapshot: LaneSnapshot,
        height: int,
        vehicle_x: float,
        speed_ms: float = 0.0,
    ) -> DriveCommand:
        """A STOP still runs the zero through the smoother, so steering does not snap on recovery.
        """
        halted = snapshot.state is DrivingState.STOP or snapshot.center is None

        # Without an encoder the measured speed is zero, which would pin the
        # lookahead to its minimum.
        control_speed = speed_ms if speed_ms > 1e-3 else self.assumed_speed_ms

        if halted:
            feedforward = feedback = 0.0
            lookahead_m = 0.0
            goal_point = None
        else:
            feedforward = self._feedforward(snapshot)
            feedback, lookahead_m, goal_point = self._feedback(
                snapshot, height, vehicle_x, control_speed
            )

        raw_deg = math.degrees(feedforward + feedback)
        smoothed = self.smoother(raw_deg)

        # tanh saturates instead of clipping, so large errors stay distinguishable.
        steering = float(np.tanh(smoothed / self.max_steer_deg) * self.max_steer_deg)

        return DriveCommand(
            steering_deg=steering,
            target_speed_ms=0.0 if halted else self._target_speed(snapshot),
            raw_steering_deg=raw_deg,
            feedforward_deg=math.degrees(feedforward),
            feedback_deg=math.degrees(feedback),
            lookahead_m=lookahead_m,
            goal_point=goal_point,
            law=self.law,
        )

    # -- parts -------------------------------------------------------------
    def _feedforward(self, snapshot: LaneSnapshot) -> float:
        if not self.feedforward:
            return 0.0
        return math.atan(self.wheelbase_m * snapshot.curvature)

    def _feedback(self, snapshot, height, vehicle_x, speed_ms):
        lookahead_m = float(np.clip(
            self.lookahead_base_m + self.lookahead_time_s * speed_ms,
            self.lookahead_min_m, self.lookahead_max_m,
        ))

        if self.law == STANLEY:
            angle = stanley_angle(
                self.heading_gain * snapshot.heading_rad,
                snapshot.lateral_offset_m,
                speed_ms,
                self.stanley_gain,
                self.stanley_softening_ms,
            )
            return angle, lookahead_m, None

        # Pure pursuit: lookahead distance -> image row -> centreline point.
        lookahead_px = lookahead_m / self.scale.m_per_px_y

        # Never aim past the fitted stretch: extrapolating the parabola is worst in
        # exactly the tight corner where the fit is short.
        horizon_y = max(0.0, snapshot.center.y_min)
        goal_y = float(np.clip(height - lookahead_px, horizon_y, height))
        goal_x = snapshot.center.x_at(goal_y)

        lateral_error_m = (goal_x - vehicle_x) * self.scale.m_per_px_x
        longitudinal_m = max((height - goal_y) * self.scale.m_per_px_y, 1e-6)

        angle = self.pursuit_gain * pure_pursuit_angle(
            lateral_error_m, longitudinal_m, self.wheelbase_m
        )
        return angle, longitudinal_m, (int(goal_x), int(goal_y))

    def _target_speed(self, snapshot: LaneSnapshot) -> float:
        # The tightest radius in view, not under the wheels.
        limit = curvature_speed_limit(
            snapshot.min_curve_radius_m, self.max_lateral_accel_ms2
        )
        speed = min(self.max_speed_ms, limit)

        # Never outdrive the sensor: cap speed at the visible road crossed in
        # horizon_time_s, so the car slows for a corner it cannot see round.
        if snapshot.visible_distance_m > 0.0 and self.horizon_time_s > 0.0:
            speed = min(speed, snapshot.visible_distance_m / self.horizon_time_s)

        # A guessed or remembered boundary is a weaker basis for speed.
        if snapshot.state is not DrivingState.LANE_KEEPING:
            speed *= 0.6
        speed *= 0.5 + 0.5 * float(np.clip(snapshot.confidence, 0.0, 1.0))

        return float(np.clip(speed, self.min_speed_ms, self.max_speed_ms))
