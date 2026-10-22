from __future__ import annotations

import numpy as np

from bfmc.fitting import split_by_slope, weighted_average_lane


def test_split_by_slope_assigns_sides():
    # y grows downwards, so the left boundary has a negative slope.
    left_segment = (100, 400, 160, 200)
    right_segment = (500, 200, 560, 400)
    flat_segment = (100, 300, 400, 305)   # nearly horizontal: noise

    left, right = split_by_slope([left_segment, right_segment, flat_segment],
                                 min_slope=0.3)

    assert left == [left_segment]
    assert right == [right_segment]


def test_split_by_slope_drops_vertical_segments():
    assert split_by_slope([(100, 400, 101, 100)], min_slope=0.3) == ([], [])


def test_weighted_average_lane_recovers_a_known_line():
    def x_of(y):
        return int(100 + 0.25 * (400 - y))

    segments = [(x_of(y), y, x_of(y - 60), y - 60) for y in range(400, 100, -60)]
    lane = weighted_average_lane(segments, height=400, top_frac=0.3)

    assert lane is not None
    assert abs(lane.x1 - 100) <= 2
    assert lane.y1 == 400 and lane.y2 == 120


def test_weighted_average_lane_returns_none_without_usable_segments():
    assert weighted_average_lane([], height=400) is None
    assert weighted_average_lane([(100, 400, 105, 398)], height=400) is None   # too short
    assert weighted_average_lane([(0, 300, 400, 300)], height=400) is None     # too flat


def test_a_short_outlier_cannot_move_the_fit():
    good = [(100, 400, 150, 200), (150, 200, 200, 0)]
    outlier = [(300, 300, 305, 290)]
    assert weighted_average_lane(good, 400) == weighted_average_lane(good + outlier, 400)


def test_the_fit_survives_pixel_noise():
    rng = np.random.default_rng(0)
    segments = []
    for y in range(400, 100, -50):
        jitter = rng.integers(-2, 3, size=2)
        segments.append((100 + int(jitter[0]) + (400 - y) // 4, y,
                         100 + int(jitter[1]) + (450 - y) // 4, y - 50))

    lane = weighted_average_lane(segments, height=400)
    assert lane is not None and abs(lane.x1 - 100) <= 12
