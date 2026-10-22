from __future__ import annotations

import numpy as np
import pytest

from bfmc.config import WarpConfig
from bfmc.control import DriveCommand
from bfmc.geometry import LanePolynomial
from bfmc.overlay import compose, draw_birdseye, draw_hud, draw_lane_region
from bfmc.perception import PerspectiveWarp
from bfmc.states import DrivingState
from bfmc.tracking import LaneSnapshot

WIDTH, HEIGHT = 640, 480
CENTRE = LanePolynomial(0.0, 0.0, 320.0, 0, HEIGHT)

SNAPSHOT = LaneSnapshot(
    left=CENTRE.shifted(-160), right=CENTRE.shifted(160), center=CENTRE,
    state=DrivingState.LANE_KEEPING, confidence=0.8, visible_distance_m=1.2,
)
COMMAND = DriveCommand(steering_deg=4.0, target_speed_ms=0.8,
                       goal_point=(330, 200), lookahead_m=0.5)


@pytest.fixture
def frame():
    return np.full((HEIGHT, WIDTH, 3), 60, np.uint8)


def test_lane_region_shades_the_camera_view(frame):
    warp = PerspectiveWarp(WarpConfig(), WIDTH, HEIGHT)
    before = frame.copy()
    draw_lane_region(frame, warp, SNAPSHOT)
    assert not np.array_equal(frame, before)


def test_lane_region_is_a_no_op_without_boundaries(frame):
    warp = PerspectiveWarp(WarpConfig(), WIDTH, HEIGHT)
    empty = LaneSnapshot(left=None, right=None, center=None)
    before = frame.copy()
    draw_lane_region(frame, warp, empty)
    assert np.array_equal(frame, before)


def test_birdseye_draws_boundaries_and_the_goal_point(frame):
    pixels = np.array([[100, 400], [101, 401]], int)
    result = draw_birdseye(frame, SNAPSHOT, COMMAND, WIDTH // 2, pixels, pixels)
    assert result.shape == (HEIGHT, WIDTH, 3)
    assert result.max() > 60


def test_birdseye_copes_with_a_missing_boundary(frame):
    partial = LaneSnapshot(left=CENTRE.shifted(-160), right=None, center=None)
    draw_birdseye(frame, partial, DriveCommand(0.0, 0.0), WIDTH // 2)


def test_hud_prints_the_numbers(frame):
    result = draw_hud(frame, SNAPSHOT, COMMAND, speed_ms=0.7, fps=25.0)
    assert result.shape == (HEIGHT, WIDTH, 3)


def test_hud_handles_an_infinite_radius(frame):
    draw_hud(frame, SNAPSHOT, COMMAND, speed_ms=0.0)  # curve_radius_m is inf


def test_hud_flags_a_rejected_lane_width(frame):
    rejected = LaneSnapshot(left=CENTRE.shifted(-160), right=CENTRE.shifted(160),
                            center=CENTRE, width_rejected=True)
    plain = draw_hud(frame.copy(), SNAPSHOT, COMMAND, 0.5)
    flagged = draw_hud(frame.copy(), rejected, COMMAND, 0.5)
    assert not np.array_equal(plain, flagged)


def test_compose_stacks_panes_and_converts_masks(frame):
    mask = np.zeros((HEIGHT, WIDTH), np.uint8)
    combined = compose(frame, frame, mask)
    assert combined.shape == (HEIGHT, WIDTH * 3, 3)


def test_compose_matches_differing_heights(frame):
    small = np.zeros((240, 320, 3), np.uint8)
    combined = compose(frame, small)
    assert combined.shape[0] == HEIGHT
