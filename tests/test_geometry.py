from __future__ import annotations

import math

import numpy as np
import pytest

from bfmc.geometry import LanePolynomial, Line, MetricScale, lane_center, x_at_y

HEIGHT = 480
# The anisotropic scale a bird's-eye warp actually produces.
SCALE = MetricScale(m_per_px_x=0.35 / 320, m_per_px_y=1.2 / 480)


def arc_polynomial(radius_m: float, scale: MetricScale = SCALE,
                   vehicle_x: float = 320.0) -> LanePolynomial:
    """Second-order fit of an arc: sideways deviation ``d^2 / 2R``. Positive radius bends right."""
    ys = np.linspace(0.0, float(HEIGHT), 60)
    ahead_m = (HEIGHT - ys) * scale.m_per_px_y
    xs = vehicle_x + (ahead_m ** 2 / (2.0 * radius_m)) / scale.m_per_px_x
    fit = LanePolynomial.fit(xs, ys, 0.0, float(HEIGHT))
    assert fit is not None
    return fit


class TestLine:
    def test_x_at_y_interpolates(self):
        assert x_at_y(Line(100, 400, 200, 0), 200) == 150

    def test_x_at_y_falls_back_for_degenerate_lines(self):
        assert x_at_y(Line(120, 400, 120, 0), 200) == 120     # vertical
        assert x_at_y(Line(100, 300, 400, 300), 50) == 100    # horizontal


class TestConstruction:
    def test_from_line_reproduces_the_segment(self):
        poly = LanePolynomial.from_line(Line(100, 400, 160, 100))
        assert poly.a == 0.0
        assert poly.x_at(400) == pytest.approx(100)
        assert poly.x_at(100) == pytest.approx(160)

    def test_from_line_rejects_a_horizontal_segment(self):
        with pytest.raises(ValueError):
            LanePolynomial.from_line(Line(0, 300, 400, 300))

    def test_fit_recovers_a_known_polynomial(self):
        ys = np.linspace(0, 480, 40)
        xs = 3e-4 * ys ** 2 - 0.2 * ys + 150.0
        fit = LanePolynomial.fit(xs, ys, 0.0, 480.0)
        assert fit.a == pytest.approx(3e-4, rel=1e-6)
        assert fit.b == pytest.approx(-0.2, rel=1e-6)
        assert fit.c == pytest.approx(150.0, rel=1e-6)

    def test_fit_needs_more_rows_than_the_degree(self):
        assert LanePolynomial.fit([1, 2], [10, 20], 0, 100) is None
        # Plenty of points, but all on two rows: a parabola is unconstrained.
        assert LanePolynomial.fit([1] * 20 + [5] * 20, [10] * 20 + [20] * 20, 0, 100) is None

    def test_fit_rejects_mismatched_inputs(self):
        with pytest.raises(ValueError):
            LanePolynomial.fit([1, 2, 3], [1, 2], 0, 100)

    def test_degree_one_forces_a_straight_fit(self):
        ys = np.linspace(0, 480, 40)
        xs = 3e-4 * ys ** 2 - 0.2 * ys + 150.0
        assert LanePolynomial.fit(xs, ys, 0.0, 480.0, degree=1).a == 0.0


class TestEvaluation:
    def test_x_at_accepts_scalars_and_arrays(self):
        poly = LanePolynomial(0.0, 1.0, 5.0, 0, 480)
        assert poly.x_at(10) == pytest.approx(15.0)
        assert np.allclose(poly.x_at(np.array([0.0, 10.0])), [5.0, 15.0])

    def test_shifted_moves_sideways_only(self):
        poly = arc_polynomial(2.0)
        moved = poly.shifted(40.0)
        assert moved.a == poly.a and moved.b == poly.b
        assert moved.x_at(HEIGHT) == pytest.approx(poly.x_at(HEIGHT) + 40.0)

    def test_points_run_from_the_car_outwards(self):
        points = arc_polynomial(2.0).points(count=8)
        assert points.shape == (8, 2)
        assert points[0][1] > points[-1][1]


class TestPhysicalQuantities:
    @pytest.mark.parametrize("radius_m", [1.0, 2.0, 5.0])
    def test_curvature_matches_the_arc_it_was_fitted_to(self, radius_m):
        poly = arc_polynomial(radius_m)
        assert poly.curve_radius_m(HEIGHT, SCALE) == pytest.approx(radius_m, rel=0.05)

    def test_curvature_sign_is_positive_for_a_right_bend(self):
        assert arc_polynomial(2.0).signed_curvature(HEIGHT, SCALE) > 0
        assert arc_polynomial(-2.0).signed_curvature(HEIGHT, SCALE) < 0

    def test_a_straight_lane_has_infinite_radius(self):
        straight = LanePolynomial(0.0, 0.0, 320.0, 0, HEIGHT)
        assert straight.curve_radius_m(HEIGHT, SCALE) == math.inf
        assert straight.signed_curvature(HEIGHT, SCALE) == 0.0

    def test_heading_is_zero_when_the_lane_points_straight_ahead(self):
        straight = LanePolynomial(0.0, 0.0, 320.0, 0, HEIGHT)
        assert straight.heading_rad(HEIGHT, SCALE) == pytest.approx(0.0)

    def test_heading_is_positive_when_the_lane_runs_right(self):
        # x grows as y decreases, i.e. the lane heads right as it recedes.
        rightwards = LanePolynomial(0.0, -0.5, 320.0, 0, HEIGHT)
        assert rightwards.heading_rad(HEIGHT, SCALE) > 0
        leftwards = LanePolynomial(0.0, 0.5, 320.0, 0, HEIGHT)
        assert leftwards.heading_rad(HEIGHT, SCALE) < 0

    @pytest.mark.parametrize("radius_m", [2.0, 5.0])
    def test_mean_curvature_agrees_with_a_gentle_arc(self, radius_m):
        poly = arc_polynomial(radius_m)
        assert abs(1.0 / poly.mean_curvature(SCALE)) == pytest.approx(radius_m, rel=0.15)

    def test_mean_curvature_averages_rather_than_peaks(self):
        """A parabola's curvature falls off with distance, so the average reports a larger radius
        than the tightest point - why the speed limit uses the tightest segment instead.
        """
        poly = arc_polynomial(1.0)
        at_car = abs(poly.signed_curvature(HEIGHT, SCALE))
        averaged = abs(poly.mean_curvature(SCALE))
        far = abs(poly.signed_curvature(0.0, SCALE))
        assert far < averaged < at_car

    def test_mean_curvature_keeps_the_sign_convention(self):
        assert arc_polynomial(2.0).mean_curvature(SCALE) > 0
        assert arc_polynomial(-2.0).mean_curvature(SCALE) < 0

    def test_mean_curvature_of_a_degenerate_span_falls_back(self):
        stub = LanePolynomial(1e-4, 0.0, 320.0, 479.5, 480.0)
        assert stub.mean_curvature(SCALE) == stub.signed_curvature(480.0, SCALE)

    def test_tightest_radius_is_no_larger_than_the_radius_at_the_car(self):
        poly = arc_polynomial(1.5)
        assert poly.tightest_radius_m(SCALE) <= poly.curve_radius_m(HEIGHT, SCALE) * 1.05

    def test_tightest_radius_of_a_straight_is_infinite(self):
        straight = LanePolynomial(0.0, 0.0, 320.0, 0, HEIGHT)
        assert straight.tightest_radius_m(SCALE) == math.inf


class TestLaneCenter:
    def test_centre_sits_midway_between_parallel_boundaries(self):
        left = arc_polynomial(2.0).shifted(-160.0)
        right = arc_polynomial(2.0).shifted(160.0)
        centre = lane_center(left, right)
        assert centre.x_at(HEIGHT) == pytest.approx(
            (left.x_at(HEIGHT) + right.x_at(HEIGHT)) / 2.0
        )

    def test_centre_keeps_the_shared_curvature(self):
        left = arc_polynomial(2.0).shifted(-160.0)
        right = arc_polynomial(2.0).shifted(160.0)
        assert lane_center(left, right).a == pytest.approx(left.a)

    def test_centre_is_trusted_only_where_both_boundaries_reach(self):
        left = LanePolynomial(0.0, 0.0, 160.0, y_min=200, y_max=480)
        right = LanePolynomial(0.0, 0.0, 480.0, y_min=50, y_max=470)
        centre = lane_center(left, right)
        assert centre.y_min == 200
        assert centre.y_max == 470

    def test_centre_takes_the_weaker_confidence(self):
        left = LanePolynomial(0.0, 0.0, 160.0, 0, 480, confidence=0.9)
        right = LanePolynomial(0.0, 0.0, 480.0, 0, 480, confidence=0.4)
        assert lane_center(left, right).confidence == pytest.approx(0.4)
