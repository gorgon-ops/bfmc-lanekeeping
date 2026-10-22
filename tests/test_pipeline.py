from __future__ import annotations

import pytest

from bfmc.config import AppConfig
from bfmc.pipeline import LaneKeepingPipeline
from bfmc.states import DrivingState
from bfmc.telemetry import NullSpeedSensor


def drain(pipeline, source):
    result = None
    while True:
        frame = source.read()
        if frame is None:
            return result
        result = pipeline.process(frame)


class TestProcessing:
    def test_a_straight_road_is_tracked(self, offline_config, fake_source, road_frame):
        source = fake_source([road_frame()])
        with LaneKeepingPipeline(offline_config, source=source) as pipeline:
            result = pipeline.process(source.read())

        assert result.snapshot.state is DrivingState.LANE_KEEPING
        assert result.snapshot.has_both
        assert result.command.goal_point is not None

    def test_a_centred_road_needs_almost_no_steering(self, offline_config,
                                                     fake_source, road_frame):
        source = fake_source([road_frame() for _ in range(20)])
        with LaneKeepingPipeline(offline_config, source=source) as pipeline:
            result = drain(pipeline, source)
        assert abs(result.command.steering_deg) < 3.0

    @pytest.mark.parametrize("shift, expected_sign", [(70, 1), (-70, -1)])
    def test_an_offset_road_steers_towards_it(self, offline_config, fake_source,
                                              road_frame, shift, expected_sign):
        source = fake_source([road_frame(shift=shift) for _ in range(20)])
        with LaneKeepingPipeline(offline_config, source=source) as pipeline:
            result = drain(pipeline, source)
        assert result.command.steering_deg * expected_sign > 0.5

    def test_blank_frames_fall_through_to_stop(self, offline_config, fake_source,
                                               blank_frame):
        source = fake_source([blank_frame] * 30)
        with LaneKeepingPipeline(offline_config, source=source) as pipeline:
            result = drain(pipeline, source)

        assert result.snapshot.state is DrivingState.STOP
        assert result.command.steering_deg == pytest.approx(0.0, abs=1e-6)
        assert result.command.target_speed_ms == 0.0

    def test_latency_and_throughput_are_measured(self, offline_config, fake_source,
                                                 road_frame):
        source = fake_source([road_frame() for _ in range(5)])
        with LaneKeepingPipeline(offline_config, source=source) as pipeline:
            result = drain(pipeline, source)
            assert pipeline.fps > 0
        assert result.latency_s > 0


class TestGeometry:
    def test_it_adapts_to_an_unexpected_frame_size(self, offline_config,
                                                   fake_source, road_frame):
        """A camera may ignore the requested resolution; the first real frame wins."""
        offline_config.camera.width = 1280
        offline_config.camera.height = 720
        source = fake_source([road_frame(640, 480)])

        with LaneKeepingPipeline(offline_config, source=source) as pipeline:
            pipeline.process(source.read())
            assert (pipeline.width, pipeline.height) == (640, 480)
            assert pipeline.vehicle_x == 320
            assert pipeline.scale.m_per_px_y == pytest.approx(
                offline_config.metric.view_depth_m / 480)

    def test_the_metric_scale_comes_from_the_config(self, offline_config,
                                                    fake_source):
        with LaneKeepingPipeline(offline_config, source=fake_source()) as pipeline:
            assert pipeline.scale.m_per_px_x == pytest.approx(
                offline_config.metric.lane_width_m / offline_config.tracking.lane_width_px)


class TestLifecycle:
    def test_closing_releases_the_source(self, offline_config, fake_source):
        source = fake_source()
        with LaneKeepingPipeline(offline_config, source=source):
            pass
        assert source.closed

    def test_a_disabled_encoder_uses_a_null_sensor(self, offline_config, fake_source):
        with LaneKeepingPipeline(offline_config, source=fake_source()) as pipeline:
            assert isinstance(pipeline.speed_sensor, NullSpeedSensor)
            assert pipeline.speed_sensor.speed_ms == 0.0

    def test_reset_clears_detector_tracker_and_filter(self, offline_config,
                                                      fake_source, road_frame):
        frames = [road_frame() for _ in range(6)]
        source = fake_source(frames)
        with LaneKeepingPipeline(offline_config, source=source) as pipeline:
            drain(pipeline, source)
            pipeline.reset()
            assert not pipeline.detector.detect(frames[0]).tracked

    def test_default_config_still_builds_a_pipeline(self, fake_source):
        config = AppConfig()
        config.display = False
        config.encoder.enabled = False
        with LaneKeepingPipeline(config, source=fake_source()) as pipeline:
            assert pipeline.vehicle_x == config.camera.width // 2


class TestRunLoop:
    def test_run_stops_when_the_source_is_exhausted(self, offline_config,
                                                    fake_source, road_frame):
        source = fake_source([road_frame() for _ in range(7)])
        with LaneKeepingPipeline(offline_config, source=source) as pipeline:
            assert pipeline.run() == 7

    def test_run_honours_max_frames(self, offline_config, fake_source, road_frame):
        offline_config.max_frames = 3
        source = fake_source([road_frame() for _ in range(10)])
        with LaneKeepingPipeline(offline_config, source=source) as pipeline:
            assert pipeline.run() == 3

    def test_run_calls_the_result_hook_for_every_frame(self, offline_config,
                                                       fake_source, road_frame):
        source = fake_source([road_frame() for _ in range(4)])
        seen = []
        with LaneKeepingPipeline(offline_config, source=source) as pipeline:
            pipeline.run(on_result=seen.append)
        assert len(seen) == 4


class TestOutput:
    def test_summary_mentions_the_state_and_steering(self, offline_config,
                                                     fake_source, road_frame):
        source = fake_source([road_frame()])
        with LaneKeepingPipeline(offline_config, source=source) as pipeline:
            summary = pipeline.process(source.read()).summary()
        assert "LANE KEEP" in summary and "steer" in summary

    def test_a_telemetry_row_is_flat_and_serialisable(self, offline_config,
                                                      fake_source, road_frame):
        source = fake_source([road_frame()])
        with LaneKeepingPipeline(offline_config, source=source) as pipeline:
            row = pipeline.process(source.read()).as_row()

        assert row["state"] == "LANE KEEP"
        assert all(isinstance(v, (int, float, str)) for v in row.values())
        assert {"steering_deg", "target_speed_ms", "curve_radius_m"} <= set(row)

    def test_render_produces_a_debug_view(self, offline_config, fake_source,
                                          road_frame):
        source = fake_source([road_frame()])
        with LaneKeepingPipeline(offline_config, source=source) as pipeline:
            result = pipeline.process(source.read())
            plain = pipeline.render(result)
            with_mask = pipeline.render(result, show_mask=True)

        assert plain.shape[1] == offline_config.camera.width * 2
        assert with_mask.shape[1] > plain.shape[1]
        assert plain.shape[0] == offline_config.camera.height
