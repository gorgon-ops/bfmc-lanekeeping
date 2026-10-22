from __future__ import annotations

import math

import numpy as np
import pytest

from bfmc.simulator import (
    SAMPLE_STEP_M,
    Track,
    TrackSimulator,
    figure_eight_track,
    oval_track,
)


class TestTrack:
    def test_a_straight_profile_runs_straight(self):
        track = Track.from_profile([(1.0, 0.0)])
        assert track.headings.max() == pytest.approx(track.headings.min())
        assert len(track.points) == pytest.approx(1.0 / SAMPLE_STEP_M, abs=2)

    def test_an_arc_turns_by_length_over_radius(self):
        radius = 2.0
        track = Track.from_profile([(math.pi * radius, 1.0 / radius)])
        turned = track.headings[-1] - track.headings[0]
        assert turned == pytest.approx(math.pi, rel=0.01)

    def test_an_oval_closes_on_itself(self):
        track = oval_track(straight_m=2.0, radius_m=1.0)
        assert np.linalg.norm(track.points[-1] - track.points[0]) < 0.05
        assert track.closed

    def test_nearest_finds_the_point_and_side(self):
        track = Track.from_profile([(2.0, 0.0)])  # heading +y
        index, offset = track.nearest(track.points[40] + np.array([0.1, 0.0]))
        assert index == pytest.approx(40, abs=2)
        # Facing +y, a car displaced towards +x is to the right of the path.
        assert offset == pytest.approx(0.1, abs=0.01)

    def test_offset_sign_is_negative_on_the_left(self):
        track = Track.from_profile([(2.0, 0.0)])
        _, offset = track.nearest(track.points[40] - np.array([0.1, 0.0]))
        assert offset < 0

    def test_boundaries_sit_half_a_lane_either_side(self):
        track = Track.from_profile([(2.0, 0.0)], lane_width_m=0.4)
        gap = np.linalg.norm(track.boundary_points(1)[50] - track.boundary_points(-1)[50])
        assert gap == pytest.approx(0.4, rel=1e-6)

    def test_a_figure_eight_turns_both_ways(self):
        turns = np.diff(figure_eight_track().headings)
        assert turns.max() > 0 and turns.min() < 0


class TestDynamics:
    @pytest.fixture
    def simulator(self, offline_config):
        return TrackSimulator(oval_track(), offline_config)

    def test_driving_straight_advances_along_the_track(self, simulator):
        for _ in range(20):
            simulator.step(0.0, 1.0, dt=0.05)
        assert simulator.car.odometer_m > 0.5
        assert abs(simulator.cross_track_error_m) < 0.01

    def test_positive_steering_turns_right(self, simulator):
        """Positive steering is clockwise, so yaw must decrease; backwards, the car diverges on a
        straight.
        """
        start_yaw = simulator.car.yaw
        for _ in range(20):
            simulator.step(20.0, 1.0, dt=0.05)
        assert simulator.car.yaw < start_yaw

    def test_negative_steering_turns_left(self, simulator):
        start_yaw = simulator.car.yaw
        for _ in range(20):
            simulator.step(-20.0, 1.0, dt=0.05)
        assert simulator.car.yaw > start_yaw

    def test_steering_is_clamped_to_the_mechanical_limit(self, simulator, offline_config):
        wild = TrackSimulator(oval_track(), offline_config)
        for _ in range(10):
            simulator.step(offline_config.vehicle.max_steer_deg, 1.0, dt=0.05)
            wild.step(500.0, 1.0, dt=0.05)
        assert wild.car.yaw == pytest.approx(simulator.car.yaw)

    def test_speed_ramps_rather_than_jumping(self, simulator):
        simulator.step(0.0, 1.0, dt=0.05)
        assert 0.0 < simulator.car.speed_ms < 1.0

    def test_turning_radius_matches_the_bicycle_model(self, simulator, offline_config):
        angle = 20.0
        for _ in range(200):
            simulator.step(angle, 1.0, dt=0.02)
        expected = offline_config.vehicle.wheelbase_m / math.tan(math.radians(angle))
        travelled = simulator.car.odometer_m
        turned = abs(simulator.car.yaw - math.pi / 2)
        assert travelled / turned == pytest.approx(expected, rel=0.05)

    def test_placement_can_introduce_a_known_error(self, simulator):
        simulator.place_on_track(100, lateral_offset_m=0.05)
        assert simulator.cross_track_error_m == pytest.approx(0.05, abs=0.005)

    def test_leaving_the_lane_is_detected(self, simulator):
        simulator.place_on_track(100, lateral_offset_m=0.5)
        assert not simulator.on_track

    def test_an_open_track_reports_its_end(self, offline_config):
        simulator = TrackSimulator(figure_eight_track(), offline_config)
        assert not simulator.at_end_of_track
        simulator.place_on_track(len(simulator.track.points) - 1)
        assert simulator.at_end_of_track

    def test_a_closed_track_never_ends(self, offline_config):
        simulator = TrackSimulator(oval_track(), offline_config)
        simulator.place_on_track(len(simulator.track.points) - 1)
        assert not simulator.at_end_of_track


class TestRendering:
    def test_renders_a_frame_of_the_configured_size(self, offline_config):
        frame = TrackSimulator(oval_track(), offline_config).render()
        assert frame.shape == (offline_config.camera.height,
                               offline_config.camera.width, 3)
        assert frame.dtype == np.uint8

    def test_the_road_is_brighter_than_nothing(self, offline_config):
        simulator = TrackSimulator(oval_track(), offline_config)
        assert simulator.render().max() > 150  # markings are visible

    def test_only_the_road_ahead_is_drawn(self, offline_config):
        """A closed circuit passes close to itself: off-frame track must not be drawn or joined
        across.
        """
        simulator = TrackSimulator(oval_track(straight_m=1.0, radius_m=0.5),
                                   offline_config)
        for run in simulator._to_birdseye(simulator.track.points):
            lateral = np.abs(run[:, 0] - simulator.vehicle_x) * simulator.scale.m_per_px_x
            assert lateral.max() < 1.0

    def test_runs_are_split_at_gaps(self, offline_config):
        simulator = TrackSimulator(oval_track(straight_m=1.0, radius_m=0.5),
                                   offline_config)
        for run in simulator._to_birdseye(simulator.track.points):
            steps = np.abs(np.diff(run[:, 1]))
            assert steps.max() < 50  # no jump between disjoint stretches

    def test_noise_makes_successive_frames_differ(self, offline_config):
        simulator = TrackSimulator(oval_track(), offline_config, noise_sigma=8.0)
        assert not np.array_equal(simulator.render(), simulator.render())

    def test_rendering_is_reproducible_for_a_seed(self, offline_config):
        first = TrackSimulator(oval_track(), offline_config, seed=7).render()
        second = TrackSimulator(oval_track(), offline_config, seed=7).render()
        assert np.array_equal(first, second)
