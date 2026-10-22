from __future__ import annotations

import numpy as np
import pytest

from bfmc.geometry import LanePolynomial
from bfmc.polyfit import (
    SlidingWindowFinder,
    histogram_peaks,
    robust_fit,
    truncate_to_well_conditioned,
)

HEIGHT, WIDTH = 480, 640


def mask_with_boundaries(left_x=160, right_x=480, curve=0.0, thickness=5,
                         top=0, bottom=HEIGHT):
    mask = np.zeros((HEIGHT, WIDTH), np.uint8)
    for y in range(top, bottom):
        offset = curve * (HEIGHT - y) ** 2
        for base in (left_x, right_x):
            x = int(base + offset)
            start, end = max(0, x - thickness), min(WIDTH, x + thickness)
            if start < end:  # skip rows where the boundary is out of frame
                mask[y, start:end] = 255
    return mask


class TestHistogramPeaks:
    def test_finds_one_peak_either_side_of_centre(self):
        left, right = histogram_peaks(mask_with_boundaries())
        assert abs(left - 160) <= 6
        assert abs(right - 480) <= 6

    def test_reports_none_for_an_empty_mask(self):
        assert histogram_peaks(np.zeros((HEIGHT, WIDTH), np.uint8)) == (None, None)

    def test_ignores_a_side_with_too_few_hits(self):
        mask = np.zeros((HEIGHT, WIDTH), np.uint8)
        mask[470:472, 100] = 255          # two lit pixels: not a lane
        mask[:, 480] = 255
        left, right = histogram_peaks(mask, min_column_hits=5)
        assert left is None and right == 480


class TestRobustFit:
    def test_matches_a_plain_fit_on_clean_data(self):
        ys = np.linspace(0, HEIGHT, 50)
        xs = 5e-4 * ys ** 2 + 100.0
        assert robust_fit(xs, ys).a == pytest.approx(5e-4, rel=1e-3)

    def test_rejects_a_detached_outlier_cluster(self):
        ys = np.linspace(200, HEIGHT, 40)
        xs = np.full_like(ys, 300.0)
        # A small far blob, like a reflection.
        noisy_x = np.concatenate([xs, np.full(6, 600.0)])
        noisy_y = np.concatenate([ys, np.linspace(0, 40, 6)])

        clean = robust_fit(xs, ys)
        recovered = robust_fit(noisy_x, noisy_y)
        assert recovered.x_at(400.0) == pytest.approx(clean.x_at(400.0), abs=15)

    def test_keeps_the_first_fit_when_most_points_disagree(self):
        # Half on each of two lines: no outlier majority, so the first fit stands.
        ys = np.linspace(0, HEIGHT, 40)
        xs = np.where(np.arange(40) % 2 == 0, 200.0, 400.0)
        assert robust_fit(xs, ys) is not None

    def test_returns_none_when_underdetermined(self):
        assert robust_fit(np.array([1.0, 2.0]), np.array([10.0, 20.0])) is None


class TestTruncation:
    def test_leaves_a_straight_fit_alone(self):
        straight = LanePolynomial(0.0, 0.3, 100.0, 0, HEIGHT)
        assert truncate_to_well_conditioned(straight) is straight

    def test_pulls_back_the_horizon_where_the_curve_goes_sideways(self):
        steep = LanePolynomial(3e-3, -3.0, 300.0, 0, HEIGHT)
        truncated = truncate_to_well_conditioned(steep, max_slope=2.1)
        assert truncated.y_min > steep.y_min
        assert abs(truncated.slope_at(truncated.y_min)) <= 2.2

    def test_leaves_a_fit_that_never_goes_sideways(self):
        gentle = LanePolynomial(1e-5, 0.0, 300.0, 0, HEIGHT)
        assert truncate_to_well_conditioned(gentle).y_min == gentle.y_min


class TestSlidingWindowFinder:
    def test_finds_both_straight_boundaries(self):
        result = SlidingWindowFinder().find(mask_with_boundaries())
        assert result.left is not None and result.right is not None
        assert result.left.x_at(HEIGHT) == pytest.approx(160, abs=10)
        assert result.right.x_at(HEIGHT) == pytest.approx(480, abs=10)

    def test_follows_a_curve(self):
        result = SlidingWindowFinder().find(mask_with_boundaries(curve=-8e-4))
        assert result.left.a == pytest.approx(-8e-4, abs=3e-4)
        assert result.right.a == pytest.approx(-8e-4, abs=3e-4)

    def test_both_boundaries_agree_on_the_bend_direction(self):
        result = SlidingWindowFinder().find(mask_with_boundaries(curve=-8e-4))
        assert (result.left.a < 0) == (result.right.a < 0)

    def test_empty_mask_finds_nothing(self):
        result = SlidingWindowFinder().find(np.zeros((HEIGHT, WIDTH), np.uint8))
        assert result.left is None and result.right is None
        assert len(result.left_pixels) == 0

    def test_a_boundary_hugging_the_frame_edge_is_excluded(self):
        mask = mask_with_boundaries()
        mask[:200, 0:6] = 255  # clipped remnant against the left edge

        result = SlidingWindowFinder().find(mask)
        assert result.left is not None
        assert result.left.x_at(HEIGHT) == pytest.approx(160, abs=12)

    def test_the_walk_stops_at_a_gap_rather_than_jumping_it(self):
        # Present low down, absent in the middle, a different stretch at the top.
        mask = np.zeros((HEIGHT, WIDTH), np.uint8)
        mask[300:HEIGHT, 155:165] = 255
        mask[0:60, 300:310] = 255
        mask[:, 476:486] = 255

        result = SlidingWindowFinder().find(mask)
        assert result.left is not None
        assert result.left.y_min >= 200  # never reached the detached stretch

    def test_confidence_falls_with_a_short_fit(self):
        finder = SlidingWindowFinder()
        full = finder.find(mask_with_boundaries())
        stub = finder.find(mask_with_boundaries(top=380))
        assert stub.left is None or stub.left.confidence < full.left.confidence

    def test_a_prior_is_reused_on_the_next_frame(self):
        finder = SlidingWindowFinder()
        mask = mask_with_boundaries()
        first = finder.find(mask)
        second = finder.find(mask, first.left, first.right)

        assert second.tracked
        assert second.left.x_at(HEIGHT) == pytest.approx(first.left.x_at(HEIGHT), abs=8)

    def test_a_stale_prior_falls_back_to_the_window_search(self):
        finder = SlidingWindowFinder()
        nonsense = LanePolynomial(0.0, 0.0, 50.0, 0, HEIGHT)
        result = finder.find(mask_with_boundaries(), nonsense, nonsense)

        assert not result.tracked
        assert result.left.x_at(HEIGHT) == pytest.approx(160, abs=12)

    def test_degree_one_forces_straight_boundaries(self):
        finder = SlidingWindowFinder(degree=1)
        result = finder.find(mask_with_boundaries(curve=-8e-4))
        assert result.left.a == 0.0
