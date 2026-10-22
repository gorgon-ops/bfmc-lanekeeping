from __future__ import annotations

import pytest

from bfmc.cli import _config_from_args, build_parser


def parse(*argv):
    return _config_from_args(build_parser().parse_args(list(argv)))


def test_video_switches_to_the_opencv_backend():
    config = parse("--video", "lap.mp4")
    assert config.camera.backend == "opencv"
    assert config.camera.source == "lap.mp4"


def test_explicit_backend_beats_video():
    assert parse("--video", "lap.mp4", "--backend", "picamera2").camera.backend == "picamera2"


def test_output_flags_override_defaults():
    config = parse("--no-display", "--no-encoder", "--max-frames", "5", "--loop-video")
    assert config.display is False
    assert config.encoder.enabled is False
    assert config.max_frames == 5
    assert config.camera.loop_video is True


def test_algorithm_flags_select_the_method_and_law():
    config = parse("--method", "hough", "--law", "stanley", "--no-feedforward")
    assert config.detection.method == "hough"
    assert config.control.law == "stanley"
    assert config.control.feedforward is False


def test_set_overrides_any_key():
    config = parse("--set", "control.lookahead_time_s=0.85",
                   "--set", "tracking.lane_width_px=280",
                   "--set", "detection.window.windows=12")
    assert config.control.lookahead_time_s == 0.85
    assert config.tracking.lane_width_px == 280
    assert config.detection.window.windows == 12


def test_config_file_is_applied_then_overridden_by_flags(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("display: true\ncontrol:\n  law: stanley\n", encoding="utf-8")

    config = parse("--config", str(path), "--no-display")

    assert config.control.law == "stanley"
    assert config.display is False


def test_unknown_override_key_is_reported():
    with pytest.raises(KeyError):
        parse("--set", "control.nope=1")


def test_an_invalid_choice_is_rejected():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--law", "telepathy"])
