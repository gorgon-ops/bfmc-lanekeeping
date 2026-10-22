from __future__ import annotations

import math

import numpy as np
import pytest

from bfmc.control import (
    PURE_PURSUIT,
    STANLEY,
    ButterworthSmoother,
    SteeringController,
    curvature_speed_limit,
    pure_pursuit_angle,
    stanley_angle,
)
from bfmc.geometry import LanePolynomial, MetricScale
from bfmc.states import DrivingState
from bfmc.tracking import LaneSnapshot

HEIGHT = 480
VEHICLE_X = 320
SCALE = MetricScale(m_per_px_x=0.35 / 320, m_per_px_y=1.2 / 480)


def snapshot(centre_x=320.0, a=0.0, b=0.0, curvature=0.0, offset_m=0.0,
             heading_rad=0.0, radius_m=math.inf, state=DrivingState.LANE_KEEPING,
             confidence=1.0, visible_m=1.2) -> LaneSnapshot:
    centre = LanePolynomial(a, b, centre_x, 0, HEIGHT)
    return LaneSnapshot(
        left=centre.shifted(-160), right=centre.shifted(160), center=centre,
        state=state, curvature=curvature, curve_radius_m=radius_m,
        min_curve_radius_m=radius_m, lateral_offset_m=offset_m,
        heading_rad=heading_rad, confidence=confidence,
        visible_distance_m=visible_m,
    )


class TestControlLaws:
    def test_pure_pursuit_is_neutral_with_no_lateral_error(self):
        assert pure_pursuit_angle(0.0, 0.5, 0.26) == pytest.approx(0.0)

    def test_pure_pursuit_steers_towards_the_goal(self):
        assert pure_pursuit_angle(0.1, 0.5, 0.26) > 0    # goal to the right
        assert pure_pursuit_angle(-0.1, 0.5, 0.26) < 0

    def test_pure_pursuit_softens_with_a_longer_lookahead(self):
        near = pure_pursuit_angle(0.1, 0.3, 0.26)
        far = pure_pursuit_angle(0.1, 0.9, 0.26)
        assert abs(far) < abs(near)

    def test_pure_pursuit_is_safe_at_zero_lookahead(self):
        assert pure_pursuit_angle(0.1, 0.0, 0.26) == 0.0

    def test_stanley_corrects_heading_and_cross_track_separately(self):
        heading_only = stanley_angle(0.2, 0.0, 1.0, 1.4, 0.4)
        assert heading_only == pytest.approx(0.2)

        # Car right of centre: steer left.
        assert stanley_angle(0.0, 0.1, 1.0, 1.4, 0.4) < 0

    def test_stanley_is_finite_at_a_standstill(self):
        assert abs(stanley_angle(0.0, 0.1, 0.0, 1.4, 0.4)) < math.pi / 2

    def test_stanley_cross_track_term_shrinks_with_speed(self):
        slow = abs(stanley_angle(0.0, 0.1, 0.2, 1.4, 0.4))
        fast = abs(stanley_angle(0.0, 0.1, 3.0, 1.4, 0.4))
        assert fast < slow

    def test_curvature_speed_limit_follows_the_physics(self):
        assert curvature_speed_limit(1.0, 2.0) == pytest.approx(math.sqrt(2.0))
        assert curvature_speed_limit(4.0, 2.0) == pytest.approx(math.sqrt(8.0))
        assert curvature_speed_limit(math.inf, 2.0) == math.inf


class TestSmoother:
    def test_rejects_an_invalid_cutoff(self):
        with pytest.raises(ValueError):
            ButterworthSmoother(cutoff=0.0)
        with pytest.raises(ValueError):
            ButterworthSmoother(cutoff=1.5)

    def test_converges_on_a_constant_input(self):
        smoother = ButterworthSmoother()
        assert [smoother(10.0) for _ in range(200)][-1] == pytest.approx(10.0, abs=1e-6)

    def test_attenuates_frame_to_frame_jitter(self):
        smoother = ButterworthSmoother(cutoff=0.1)
        filtered = [smoother(5.0 if i % 2 else -5.0) for i in range(100)][20:]
        assert max(abs(v) for v in filtered) < 1.0

    def test_reset_clears_memory(self):
        smoother = ButterworthSmoother()
        for _ in range(50):
            smoother(20.0)
        smoother.reset()
        assert abs(smoother(0.0)) < 1e-9


class TestFeedforwardPolicy:
    """Pure pursuit already follows curvature; Stanley does not."""

    def test_auto_disables_feedforward_for_pure_pursuit(self):
        assert SteeringController(law=PURE_PURSUIT, feedforward="auto").feedforward is False

    def test_auto_enables_feedforward_for_stanley(self):
        assert SteeringController(law=STANLEY, feedforward="auto").feedforward is True

    def test_an_explicit_setting_still_wins(self):
        assert SteeringController(law=PURE_PURSUIT, feedforward=True).feedforward is True
        assert SteeringController(law=STANLEY, feedforward=False).feedforward is False

    def test_feedforward_turns_towards_the_bend(self):
        controller = SteeringController(scale=SCALE, law=STANLEY)
        command = controller.update(snapshot(curvature=0.8, radius_m=1.25),
                                    HEIGHT, VEHICLE_X, speed_ms=0.5)
        assert command.feedforward_deg > 0

    def test_unknown_law_is_rejected(self):
        with pytest.raises(ValueError):
            SteeringController(law="telepathy")


class TestSteering:
    def test_centred_on_a_straight_commands_no_steering(self):
        controller = SteeringController(scale=SCALE)
        for _ in range(40):
            command = controller.update(snapshot(), HEIGHT, VEHICLE_X, speed_ms=0.6)
        assert command.steering_deg == pytest.approx(0.0, abs=0.05)

    def test_offset_road_steers_towards_it(self):
        controller = SteeringController(scale=SCALE)
        for _ in range(30):
            right = controller.update(snapshot(centre_x=420.0), HEIGHT, VEHICLE_X,
                                      speed_ms=0.6)
        assert right.steering_deg > 0

        controller.reset()
        for _ in range(30):
            left = controller.update(snapshot(centre_x=220.0), HEIGHT, VEHICLE_X,
                                     speed_ms=0.6)
        assert left.steering_deg < 0

    def test_stop_commands_a_straight_wheel_and_no_speed(self):
        controller = SteeringController(scale=SCALE)
        command = controller.update(snapshot(state=DrivingState.STOP),
                                    HEIGHT, VEHICLE_X, speed_ms=0.5)
        assert command.goal_point is None
        assert command.target_speed_ms == 0.0
        assert command.raw_steering_deg == 0.0

    def test_a_missing_centreline_is_treated_as_stop(self):
        controller = SteeringController(scale=SCALE)
        blind = LaneSnapshot(left=None, right=None, center=None)
        assert controller.update(blind, HEIGHT, VEHICLE_X).target_speed_ms == 0.0

    def test_steering_saturates_without_exceeding_the_limit(self):
        controller = SteeringController(scale=SCALE, max_steer_deg=25.0,
                                        pursuit_gain=40.0)
        for _ in range(60):
            command = controller.update(snapshot(centre_x=1500.0), HEIGHT, VEHICLE_X,
                                        speed_ms=0.5)
        assert abs(command.steering_deg) <= 25.0
        assert abs(command.raw_steering_deg) > 25.0

    def test_saturation_compresses_rather_than_clips(self):
        def settle(centre_x):
            controller = SteeringController(scale=SCALE, max_steer_deg=25.0,
                                            pursuit_gain=3.0)
            for _ in range(60):
                command = controller.update(snapshot(centre_x=centre_x), HEIGHT,
                                            VEHICLE_X, speed_ms=0.5)
            return command.steering_deg

        assert 0 < settle(520.0) < settle(700.0) < 25.0

    def test_lookahead_grows_with_speed(self):
        controller = SteeringController(scale=SCALE)
        slow = controller.update(snapshot(), HEIGHT, VEHICLE_X, speed_ms=0.2)
        fast = controller.update(snapshot(), HEIGHT, VEHICLE_X, speed_ms=1.5)
        assert fast.lookahead_m > slow.lookahead_m

    def test_lookahead_never_aims_past_the_fitted_road(self):
        controller = SteeringController(scale=SCALE, lookahead_base_m=5.0)
        short = LanePolynomial(0.0, 0.0, 320.0, y_min=360, y_max=HEIGHT)
        stub = LaneSnapshot(left=short.shifted(-160), right=short.shifted(160),
                            center=short, state=DrivingState.LANE_KEEPING)

        command = controller.update(stub, HEIGHT, VEHICLE_X, speed_ms=1.0)
        assert command.goal_point[1] >= 360

    def test_steering_stays_continuous_through_a_dropout(self):
        controller = SteeringController(scale=SCALE)
        for _ in range(30):
            controller.update(snapshot(centre_x=420.0), HEIGHT, VEHICLE_X, speed_ms=0.6)
        for _ in range(60):
            command = controller.update(snapshot(state=DrivingState.STOP),
                                        HEIGHT, VEHICLE_X)
        assert command.steering_deg == pytest.approx(0.0, abs=1e-3)

    def test_output_is_deterministic(self):
        def run():
            controller = SteeringController(scale=SCALE)
            return [controller.update(snapshot(centre_x=320.0 + i), HEIGHT, VEHICLE_X,
                                      speed_ms=0.8).steering_deg for i in range(20)]

        assert np.allclose(run(), run())


class TestSpeedPolicy:
    def test_a_straight_allows_the_maximum(self):
        controller = SteeringController(scale=SCALE, max_speed_ms=1.2,
                                        horizon_time_s=0.0)
        command = controller.update(snapshot(), HEIGHT, VEHICLE_X, speed_ms=1.0)
        assert command.target_speed_ms == pytest.approx(1.2)

    def test_the_sensing_horizon_binds_before_the_top_speed(self):
        """With a 1.2 m view, 1.2 m/s is already a second of warning: the camera sets the pace."""
        controller = SteeringController(scale=SCALE, max_speed_ms=1.2,
                                        horizon_time_s=1.1)
        command = controller.update(snapshot(visible_m=1.2), HEIGHT, VEHICLE_X,
                                    speed_ms=1.0)
        assert command.target_speed_ms == pytest.approx(1.2 / 1.1, rel=1e-3)

    def test_a_tight_corner_slows_the_car(self):
        controller = SteeringController(scale=SCALE)
        straight = controller.update(snapshot(), HEIGHT, VEHICLE_X, speed_ms=1.0)
        corner = controller.update(snapshot(radius_m=0.5, curvature=2.0),
                                   HEIGHT, VEHICLE_X, speed_ms=1.0)
        assert corner.target_speed_ms < straight.target_speed_ms

    def test_a_short_sensing_horizon_slows_the_car(self):
        controller = SteeringController(scale=SCALE, horizon_time_s=1.0)
        far = controller.update(snapshot(visible_m=1.2), HEIGHT, VEHICLE_X, speed_ms=1.0)
        near = controller.update(snapshot(visible_m=0.4), HEIGHT, VEHICLE_X, speed_ms=1.0)
        assert near.target_speed_ms < far.target_speed_ms
        assert near.target_speed_ms == pytest.approx(0.4, abs=0.01)

    def test_the_horizon_limit_can_be_switched_off(self):
        controller = SteeringController(scale=SCALE, horizon_time_s=0.0)
        command = controller.update(snapshot(visible_m=0.2), HEIGHT, VEHICLE_X,
                                    speed_ms=1.0)
        assert command.target_speed_ms == pytest.approx(1.2)

    def test_a_weak_fit_slows_the_car(self):
        controller = SteeringController(scale=SCALE)
        strong = controller.update(snapshot(confidence=1.0), HEIGHT, VEHICLE_X)
        weak = controller.update(snapshot(confidence=0.2), HEIGHT, VEHICLE_X)
        assert weak.target_speed_ms < strong.target_speed_ms

    def test_a_degraded_state_slows_the_car(self):
        controller = SteeringController(scale=SCALE)
        keeping = controller.update(snapshot(), HEIGHT, VEHICLE_X)
        coasting = controller.update(snapshot(state=DrivingState.COASTING),
                                     HEIGHT, VEHICLE_X)
        assert coasting.target_speed_ms < keeping.target_speed_ms

    def test_speed_never_falls_below_the_minimum(self):
        controller = SteeringController(scale=SCALE, min_speed_ms=0.2)
        command = controller.update(snapshot(radius_m=0.01, confidence=0.01,
                                             visible_m=0.01),
                                    HEIGHT, VEHICLE_X)
        assert command.target_speed_ms == pytest.approx(0.2)
