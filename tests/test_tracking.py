from __future__ import annotations

import math

import pytest

from bfmc.geometry import LanePolynomial, MetricScale
from bfmc.states import DrivingState
from bfmc.tracking import LaneTracker

HEIGHT = 480
VEHICLE_X = 320
SCALE = MetricScale(m_per_px_x=0.35 / 320, m_per_px_y=1.2 / 480)

LEFT = LanePolynomial(0.0, 0.0, 160.0, 0, HEIGHT)
RIGHT = LanePolynomial(0.0, 0.0, 480.0, 0, HEIGHT)


def curved(centre_x: float, a: float = -5e-4) -> LanePolynomial:
    return LanePolynomial(a, 0.0, centre_x, 0, HEIGHT)


@pytest.fixture
def tracker():
    return LaneTracker(memory_frames=3, lane_width_px=320.0, scale=SCALE)


class TestStates:
    def test_both_boundaries_detected_is_lane_keeping(self, tracker):
        snapshot = tracker.update(LEFT, RIGHT, HEIGHT, VEHICLE_X)
        assert snapshot.state is DrivingState.LANE_KEEPING
        assert snapshot.has_both
        assert snapshot.center is not None

    def test_a_lost_boundary_is_replayed_from_memory(self, tracker):
        tracker.update(LEFT, RIGHT, HEIGHT, VEHICLE_X)
        snapshot = tracker.update(None, RIGHT, HEIGHT, VEHICLE_X)

        assert snapshot.state is DrivingState.COASTING
        assert snapshot.left == LEFT
        assert snapshot.left_stale and snapshot.left_age == 1

    def test_memory_expires_and_recovery_takes_over(self, tracker):
        tracker.update(LEFT, RIGHT, HEIGHT, VEHICLE_X)
        for _ in range(3):
            assert tracker.update(None, RIGHT, HEIGHT, VEHICLE_X).state is DrivingState.COASTING

        snapshot = tracker.update(None, RIGHT, HEIGHT, VEHICLE_X)
        assert snapshot.state is DrivingState.RIGHT_ONLY
        assert snapshot.left_inferred
        assert snapshot.left.x_at(HEIGHT) == pytest.approx(RIGHT.x_at(HEIGHT) - 320.0)

    def test_recovery_keeps_the_curvature_of_the_visible_boundary(self, tracker):
        visible = curved(480.0)
        for _ in range(5):
            snapshot = tracker.update(None, visible, HEIGHT, VEHICLE_X)
        assert snapshot.left_inferred
        assert snapshot.left.a == pytest.approx(visible.a)

    def test_nothing_visible_commands_stop(self, tracker):
        for _ in range(5):
            snapshot = tracker.update(None, None, HEIGHT, VEHICLE_X)
        assert snapshot.state is DrivingState.STOP
        assert snapshot.center is None

    def test_detection_resets_the_age(self, tracker):
        tracker.update(LEFT, RIGHT, HEIGHT, VEHICLE_X)
        tracker.update(None, RIGHT, HEIGHT, VEHICLE_X)
        snapshot = tracker.update(LEFT, RIGHT, HEIGHT, VEHICLE_X)
        assert snapshot.left_age == 0
        assert snapshot.state is DrivingState.LANE_KEEPING

    def test_reset_clears_memory(self, tracker):
        tracker.update(LEFT, RIGHT, HEIGHT, VEHICLE_X)
        tracker.reset()
        assert tracker.update(None, None, HEIGHT, VEHICLE_X).state is DrivingState.STOP


class TestWidthCheck:
    def test_a_plausible_pair_passes(self, tracker):
        snapshot = tracker.update(LEFT, RIGHT, HEIGHT, VEHICLE_X)
        assert not snapshot.width_rejected
        assert snapshot.lane_width_px == pytest.approx(320.0)

    def test_two_fits_on_the_same_marking_are_rejected(self, tracker):
        """Both searches locked onto one line would put the centreline on the marking."""
        nearly_same = LanePolynomial(0.0, 0.0, 190.0, 0, HEIGHT, confidence=0.4)
        strong_left = LanePolynomial(0.0, 0.0, 160.0, 0, HEIGHT, confidence=0.9)

        snapshot = tracker.update(strong_left, nearly_same, HEIGHT, VEHICLE_X)

        assert snapshot.width_rejected
        assert snapshot.right_inferred
        assert snapshot.right.x_at(HEIGHT) == pytest.approx(480.0)

    def test_the_weaker_fit_is_the_one_dropped(self, tracker):
        weak_left = LanePolynomial(0.0, 0.0, 300.0, 0, HEIGHT, confidence=0.2)
        strong_right = LanePolynomial(0.0, 0.0, 330.0, 0, HEIGHT, confidence=0.95)

        snapshot = tracker.update(weak_left, strong_right, HEIGHT, VEHICLE_X)

        assert snapshot.width_rejected
        assert snapshot.left_inferred
        assert snapshot.right.x_at(HEIGHT) == pytest.approx(330.0)

    def test_inverted_boundaries_are_rejected(self, tracker):
        snapshot = tracker.update(RIGHT, LEFT, HEIGHT, VEHICLE_X)
        assert snapshot.width_rejected


class TestMeasurements:
    def test_a_centred_car_reports_no_offset(self, tracker):
        snapshot = tracker.update(LEFT, RIGHT, HEIGHT, VEHICLE_X)
        assert snapshot.lateral_offset_m == pytest.approx(0.0, abs=1e-9)

    def test_offset_is_positive_when_the_car_is_right_of_centre(self, tracker):
        # Road shifted left in the image means the car sits right of it.
        snapshot = tracker.update(LEFT.shifted(-40), RIGHT.shifted(-40), HEIGHT, VEHICLE_X)
        assert snapshot.lateral_offset_m > 0

    def test_offset_is_reported_in_metres(self, tracker):
        snapshot = tracker.update(LEFT.shifted(-32), RIGHT.shifted(-32), HEIGHT, VEHICLE_X)
        assert snapshot.lateral_offset_m == pytest.approx(32 * SCALE.m_per_px_x, rel=1e-6)

    def test_a_straight_road_reports_no_curvature(self, tracker):
        snapshot = tracker.update(LEFT, RIGHT, HEIGHT, VEHICLE_X)
        assert snapshot.road_shape == "STRAIGHT"
        assert snapshot.curve_radius_m == math.inf

    def test_curve_direction_matches_the_bend(self, tracker):
        right_bend = tracker.update(curved(160.0, 5e-4), curved(480.0, 5e-4),
                                    HEIGHT, VEHICLE_X)
        assert right_bend.road_shape == "CURVE RIGHT"
        assert right_bend.curvature > 0

        tracker.reset()
        left_bend = tracker.update(curved(160.0, -5e-4), curved(480.0, -5e-4),
                                   HEIGHT, VEHICLE_X)
        assert left_bend.road_shape == "CURVE LEFT"
        assert left_bend.curvature < 0

    def test_visible_distance_reflects_the_fitted_span(self, tracker):
        short_left = LanePolynomial(0.0, 0.0, 160.0, y_min=240, y_max=HEIGHT)
        snapshot = tracker.update(short_left, RIGHT, HEIGHT, VEHICLE_X)
        assert snapshot.visible_distance_m == pytest.approx(240 * SCALE.m_per_px_y)

    def test_confidence_takes_the_weaker_boundary(self, tracker):
        left = LanePolynomial(0.0, 0.0, 160.0, 0, HEIGHT, confidence=0.8)
        right = LanePolynomial(0.0, 0.0, 480.0, 0, HEIGHT, confidence=0.3)
        assert tracker.update(left, right, HEIGHT, VEHICLE_X).confidence == pytest.approx(0.3)
