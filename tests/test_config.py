from __future__ import annotations

import pytest

from bfmc.config import AppConfig


def test_defaults_are_populated():
    config = AppConfig()
    assert config.camera.backend == "picamera2"
    assert config.detection.method == "sliding_window"
    assert config.control.law == "pure_pursuit"
    assert config.control.feedforward == "auto"
    assert config.encoder.enabled is True


def test_load_without_a_path_returns_defaults():
    assert AppConfig.load() == AppConfig()


def test_metric_scale_is_derived_from_measurements():
    config = AppConfig()
    scale = config.metric_scale(480)
    assert scale.m_per_px_x == pytest.approx(
        config.metric.lane_width_m / config.tracking.lane_width_px)
    assert scale.m_per_px_y == pytest.approx(config.metric.view_depth_m / 480)


def test_metric_scale_follows_the_frame_height():
    config = AppConfig()
    assert config.metric_scale(240).m_per_px_y == pytest.approx(
        config.metric_scale(480).m_per_px_y * 2)


def test_yaml_overlays_only_the_keys_it_names(tmp_path):
    path = tmp_path / "test.yaml"
    path.write_text(
        "camera:\n  backend: opencv\n"
        "control:\n  law: stanley\n"
        "detection:\n  window:\n    windows: 12\n"
        "display: false\n",
        encoding="utf-8",
    )

    config = AppConfig.load(path)

    assert config.camera.backend == "opencv"
    assert config.camera.width == 640            # untouched default
    assert config.control.law == "stanley"
    assert config.control.butter_cutoff == 0.15  # untouched default
    assert config.detection.window.windows == 12
    assert config.detection.window.margin == 60
    assert config.display is False


def test_unknown_keys_are_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("control:\n  lookahed_time_s: 0.9\n", encoding="utf-8")
    with pytest.raises(KeyError):
        AppConfig.load(path)


def test_a_non_mapping_document_is_rejected(tmp_path):
    path = tmp_path / "list.yaml"
    path.write_text("- one\n- two\n", encoding="utf-8")
    with pytest.raises(ValueError):
        AppConfig.load(path)


def test_set_uses_a_dotted_path():
    config = AppConfig()
    config.set("tracking.lane_width_px", 280)
    config.set("detection.window.margin", 75)

    assert config.tracking.lane_width_px == 280
    assert config.detection.window.margin == 75

    with pytest.raises(KeyError):
        config.set("tracking.nope", 1)
    with pytest.raises(KeyError):
        config.set("nope.lane_width_px", 1)
    with pytest.raises(KeyError):
        config.set("display.nested", 1)   # not a section


def test_set_literal_parses_the_value():
    config = AppConfig()
    config.set_literal("control.lookahead_time_s=0.8")
    config.set_literal("display=False")
    config.set_literal("encoder.port=/dev/ttyACM0")

    assert config.control.lookahead_time_s == 0.8
    assert config.display is False
    assert config.encoder.port == "/dev/ttyACM0"

    with pytest.raises(ValueError):
        config.set_literal("display")


def test_to_dict_round_trips_through_apply():
    original = AppConfig()
    original.set("control.law", "stanley")
    rebuilt = AppConfig().apply(original.to_dict())
    assert rebuilt == original


def test_shipped_configs_load():
    for name in ("configs/vehicle.yaml", "configs/replay.yaml", "configs/sim.yaml"):
        assert AppConfig.load(name) is not None
