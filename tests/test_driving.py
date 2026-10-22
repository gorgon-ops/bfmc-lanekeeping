"""Thresholds are looser than the measured results, to catch regressions without failing on noise.
"""

from __future__ import annotations

import pytest

from bfmc.evaluation import LapReport, SimulatedSpeedSensor, drive
from bfmc.pipeline import LaneKeepingPipeline
from bfmc.simulator import TrackSimulator, figure_eight_track, oval_track

LANE_WIDTH_M = 0.35


def lap(track, offline_config, fake_source, steps=320, **overrides) -> LapReport:
    for key, value in overrides.items():
        offline_config.set(key, value)
    simulator = TrackSimulator(track, offline_config)
    pipeline = LaneKeepingPipeline(offline_config, source=fake_source())
    return drive(simulator, pipeline, steps=steps, dt=0.05)


class TestLapPerformance:
    def test_holds_the_lane_round_an_oval(self, offline_config, fake_source):
        report = lap(oval_track(), offline_config, fake_source)
        assert report.completed
        assert report.mean_abs_error_m < 0.03          # under 3 cm in a 35 cm lane
        assert report.max_abs_error_m < LANE_WIDTH_M / 2

    def test_holds_the_lane_through_turns_in_both_directions(self, offline_config,
                                                             fake_source):
        report = lap(figure_eight_track(), offline_config, fake_source)
        assert report.completed
        assert report.mean_abs_error_m < 0.04

    def test_survives_a_tight_corner(self, offline_config, fake_source):
        """0.8 m radius with a 1.2 m view is near the camera's limit: running wide is allowed,
        leaving is not.
        """
        report = lap(oval_track(radius_m=0.8), offline_config, fake_source)
        assert report.completed

    def test_stanley_also_completes_a_lap(self, offline_config, fake_source):
        report = lap(oval_track(), offline_config, fake_source,
                     **{"control.law": "stanley"})
        assert report.completed
        assert report.mean_abs_error_m < 0.04

    def test_steering_stays_smooth(self, offline_config, fake_source):
        report = lap(oval_track(), offline_config, fake_source)
        assert report.steering_effort < 1.0  # degrees of change per frame

    def test_it_mostly_sees_both_boundaries(self, offline_config, fake_source):
        report = lap(oval_track(), offline_config, fake_source)
        keeping = report.state_counts["LANE KEEP"]
        assert keeping > report.frames * 0.7

    def test_it_slows_for_corners(self, offline_config, fake_source):
        report = lap(oval_track(radius_m=0.8), offline_config, fake_source)
        assert report.speeds_ms.min() < report.speeds_ms.max() * 0.8


class TestDetectorComparison:
    def test_curve_fitting_beats_straight_line_fitting(self, offline_config,
                                                       fake_source):
        """A straight-line fit re-aims every frame on a bend and saws at the wheel."""
        curves = lap(oval_track(), offline_config, fake_source)

        offline_config.detection.method = "hough"
        straights = lap(oval_track(), offline_config, fake_source)

        assert curves.mean_abs_error_m < straights.mean_abs_error_m
        assert curves.steering_effort < straights.steering_effort


class TestFeedforwardPairing:
    def test_pure_pursuit_is_better_without_a_curvature_feedforward(
        self, offline_config, fake_source
    ):
        """Pure pursuit already follows curvature; adding feedforward doubles it."""
        auto = lap(oval_track(), offline_config, fake_source)

        offline_config.control.feedforward = True
        doubled = lap(oval_track(), offline_config, fake_source)

        assert auto.mean_abs_error_m < doubled.mean_abs_error_m

    def test_stanley_needs_its_feedforward(self, offline_config, fake_source):
        with_ff = lap(oval_track(), offline_config, fake_source,
                      **{"control.law": "stanley"})

        offline_config.control.feedforward = False
        without = lap(oval_track(), offline_config, fake_source)

        assert with_ff.mean_abs_error_m < without.mean_abs_error_m


class TestRecovery:
    def test_it_recovers_from_a_lateral_offset(self, offline_config, fake_source):
        simulator = TrackSimulator(oval_track(), offline_config)
        simulator.place_on_track(20, lateral_offset_m=0.08)
        pipeline = LaneKeepingPipeline(offline_config, source=fake_source())

        report = drive(simulator, pipeline, steps=120, dt=0.05)

        assert report.completed
        assert abs(report.errors[-1]) < 0.02  # back on the centreline

    def test_it_recovers_from_a_heading_error(self, offline_config, fake_source):
        simulator = TrackSimulator(oval_track(), offline_config)
        simulator.place_on_track(20, heading_error_rad=0.25)
        pipeline = LaneKeepingPipeline(offline_config, source=fake_source())

        report = drive(simulator, pipeline, steps=120, dt=0.05)

        assert report.completed
        assert abs(report.errors[-1]) < 0.03


class TestReport:
    def test_report_summarises_the_run(self, offline_config, fake_source):
        report = lap(oval_track(), offline_config, fake_source, steps=40)
        summary = report.summary()
        assert "cross-track error" in summary
        assert "completed" in summary
        assert report.fps > 0

    def test_an_empty_report_does_not_divide_by_zero(self, offline_config,
                                                     fake_source):
        report = lap(oval_track(), offline_config, fake_source, steps=0)
        assert report.mean_abs_error_m == 0.0
        assert report.steering_effort == 0.0
        assert report.fps == 0.0

    def test_the_simulated_speed_sensor_feeds_the_controller(self, offline_config):
        simulator = TrackSimulator(oval_track(), offline_config)
        sensor = SimulatedSpeedSensor(simulator)
        assert sensor.speed_ms == 0.0

        simulator.step(0.0, 1.0, dt=0.05)
        assert sensor.speed_ms > 0.0
        assert sensor.rpm > 0.0
        assert sensor.start() is sensor
        sensor.stop()

    def test_a_run_that_leaves_the_lane_is_marked_incomplete(self, offline_config,
                                                             fake_source):
        simulator = TrackSimulator(oval_track(), offline_config)
        # on_track allows a full lane width before calling it a departure.
        simulator.place_on_track(20, lateral_offset_m=0.6)
        pipeline = LaneKeepingPipeline(offline_config, source=fake_source())

        report = drive(simulator, pipeline, steps=10, dt=0.05)
        assert not report.completed


def test_a_lap_is_reproducible(offline_config, fake_source):
    first = lap(oval_track(), offline_config, fake_source, steps=60)
    second = lap(oval_track(), offline_config, fake_source, steps=60)
    assert first.mean_abs_error_m == pytest.approx(second.mean_abs_error_m)
