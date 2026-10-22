"""Typed configuration with YAML overrides. Unknown keys are rejected, so a typo fails loudly."""

from __future__ import annotations

import ast
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any

from .geometry import MetricScale

__all__ = [
    "CameraConfig", "WarpConfig", "MetricConfig", "DetectionConfig",
    "HoughConfig", "MaskConfig", "WindowConfig", "TrackingConfig",
    "VehicleConfig", "ControlConfig", "EncoderConfig", "AppConfig",
]


@dataclass
class CameraConfig:
    #: ``picamera2`` on the vehicle, ``opencv`` for a USB camera or a video file.
    backend: str = "picamera2"
    width: int = 640
    height: int = 480
    #: OpenCV only: camera index or a path to a clip.
    source: int | str = 0
    loop_video: bool = False


@dataclass
class WarpConfig:
    """Trapezoid warped to the bird's-eye view, as fractions of the frame. Calibrate on a straight.
    """

    horizon_y_frac: float = 0.60
    top_left_x_frac: float = 0.25
    top_right_x_frac: float = 0.75
    bottom_left_x_frac: float = 0.0
    bottom_right_x_frac: float = 1.0


@dataclass
class MetricConfig:
    """Everything physical downstream derives from these two, so measure them."""

    lane_width_m: float = 0.35
    view_depth_m: float = 1.20


@dataclass
class MaskConfig:
    #: Wider than a marking, narrower than the frame.
    block_size: int = 51
    brightness_offset: int = -12
    #: Relative to the frame maximum, 0 to 1.
    gradient_threshold: float = 0.18
    #: Lane markings are white or grey.
    use_saturation: bool = True
    max_saturation: int = 90
    blur_kernel: int = 5
    open_kernel: int = 3


@dataclass
class WindowConfig:
    windows: int = 9
    margin: int = 60
    #: Pixels before a window re-centres.
    min_pixels: int = 40
    #: Half-width of the band around the previous fit.
    search_margin: int = 50
    #: Fraction of windows that must find pixels.
    min_confidence: float = 0.35
    bottom_frac: float = 0.5
    #: 2 follows curves; 1 forces a straight fit.
    degree: int = 2
    #: Past this ``dx/dy`` (~65 degrees) ``x = f(y)`` stops modelling a road.
    max_slope: float = 2.1
    #: Rather than jumping a gap to a different stretch of road.
    max_missed_windows: int = 2
    #: Pixels this close to the border belong to a cut-off boundary.
    edge_margin: int = 8
    #: Shorter fits keep position but lose confidence, so the car slows rather than
    #: steering on curvature from a stub of road.
    min_span_frac: float = 0.55


@dataclass
class HoughConfig:
    canny_low: int = 50
    canny_high: int = 150
    blur_kernel: int = 5
    threshold: int = 30
    min_line_length: int = 40
    max_line_gap: int = 50
    #: Before splitting sides.
    min_slope: float = 0.3
    #: Inside the lane fit.
    fit_min_slope: float = 0.2
    min_segment_length: float = 20.0
    #: Fraction of image height.
    lane_top_frac: float = 0.3


@dataclass
class DetectionConfig:
    method: str = "sliding_window"
    mask: MaskConfig = field(default_factory=MaskConfig)
    window: WindowConfig = field(default_factory=WindowConfig)
    hough: HoughConfig = field(default_factory=HoughConfig)


@dataclass
class TrackingConfig:
    memory_frames: int = 15
    #: Warped pixels; calibrate on a straight.
    lane_width_px: float = 320.0
    #: Fractional tolerance on the gap between boundaries.
    width_tolerance: float = 0.35
    #: Metres; above this a curve is reported as straight.
    straight_radius_m: float = 6.0


@dataclass
class VehicleConfig:
    #: Metres.
    wheelbase_m: float = 0.26
    max_steer_deg: float = 30.0
    #: m/s^2 the tyres take before sliding.
    max_lateral_accel_ms2: float = 2.0
    min_speed_ms: float = 0.15
    max_speed_ms: float = 1.20


@dataclass
class ControlConfig:
    law: str = "pure_pursuit"
    #: ``auto``: on for Stanley, off for pure pursuit.
    feedforward: bool | str = "auto"

    #: Clamped base + time * speed, so the car looks a roughly constant time ahead.
    lookahead_base_m: float = 0.25
    lookahead_time_s: float = 0.45
    lookahead_min_m: float = 0.18
    lookahead_max_m: float = 0.90
    pursuit_gain: float = 1.0

    stanley_gain: float = 1.4
    stanley_softening_ms: float = 0.4
    heading_gain: float = 1.0

    assumed_speed_ms: float = 0.4
    #: Raising it brakes earlier for corners the car cannot see round.
    horizon_time_s: float = 1.1

    butter_order: int = 2
    #: Fraction of Nyquist.
    butter_cutoff: float = 0.15


@dataclass
class EncoderConfig:
    enabled: bool = True
    port: str = "/dev/ttyUSB0"
    baud_rate: int = 115200
    #: Metres.
    wheel_circumference_m: float = 0.20


@dataclass
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    warp: WarpConfig = field(default_factory=WarpConfig)
    metric: MetricConfig = field(default_factory=MetricConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    vehicle: VehicleConfig = field(default_factory=VehicleConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    display: bool = True
    max_frames: int = 0

    def metric_scale(self, height: int) -> MetricScale:
        return MetricScale(
            m_per_px_x=self.metric.lane_width_m / self.tracking.lane_width_px,
            m_per_px_y=self.metric.view_depth_m / height,
        )

    @classmethod
    def load(cls, path: str | Path | None = None) -> AppConfig:
        config = cls()
        if path is not None:
            import yaml

            data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
            if not isinstance(data, Mapping):
                raise ValueError(f"{path}: expected a mapping at the top level")
            config.apply(data)
        return config

    def apply(self, data: Mapping[str, Any]) -> AppConfig:
        _apply(self, data)
        return self

    def set(self, dotted_key: str, value: Any) -> AppConfig:
        """Dotted path, e.g. ``control.lookahead_time_s``."""
        target: Any = self
        parts = dotted_key.split(".")
        for part in parts[:-1]:
            target = _child(target, part, dotted_key)
        leaf = parts[-1]
        if not hasattr(target, leaf):
            raise KeyError(f"unknown config key: {dotted_key}")
        setattr(target, leaf, value)
        return self

    def set_literal(self, assignment: str) -> AppConfig:
        """``key.path=value``, the value parsed as Python."""
        key, separator, raw = assignment.partition("=")
        if not separator:
            raise ValueError(f"expected key=value, got {assignment!r}")
        try:
            value = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            value = raw
        return self.set(key.strip(), value)

    def to_dict(self) -> dict:
        return asdict(self)


def _child(target: Any, name: str, dotted_key: str) -> Any:
    if not hasattr(target, name):
        raise KeyError(f"unknown config key: {dotted_key}")
    child = getattr(target, name)
    if not is_dataclass(child):
        raise KeyError(f"{dotted_key}: {name} is not a config section")
    return child


def _apply(target: Any, data: Mapping[str, Any]) -> None:
    for key, value in data.items():
        if not hasattr(target, key):
            raise KeyError(f"unknown config key: {key}")
        current = getattr(target, key)
        if is_dataclass(current) and isinstance(value, Mapping):
            _apply(current, value)
        else:
            setattr(target, key, value)
