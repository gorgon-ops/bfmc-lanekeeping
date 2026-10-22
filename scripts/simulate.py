#!/usr/bin/env python3
"""Drive a simulated lap and score it. No hardware required.

python scripts/simulate.py                       # one lap of the default oval
python scripts/simulate.py --track figure8 --record lap.mp4
python scripts/simulate.py --compare             # benchmark the alternatives
"""

from __future__ import annotations

import argparse
import logging
import sys

from bfmc.config import AppConfig
from bfmc.evaluation import drive
from bfmc.pipeline import LaneKeepingPipeline
from bfmc.simulator import TrackSimulator, figure_eight_track, oval_track

TRACKS = {
    "oval": oval_track,
    "tight": lambda: oval_track(radius_m=0.8),
    "hairpin": lambda: oval_track(radius_m=0.6),
    "figure8": figure_eight_track,
}

COMPARISONS = {
    "pure pursuit (default)": {},
    "stanley": {"control.law": "stanley"},
    "pure pursuit + feedforward": {"control.feedforward": True},
    "hough (straight-line fit)": {"detection.method": "hough"},
    "no horizon speed limit": {"control.horizon_time_s": 0.0},
}


class _NoSource:
    def read(self):
        return None

    def close(self):
        return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("-c", "--config", help="YAML config to start from")
    parser.add_argument("--track", choices=sorted(TRACKS), default="oval")
    parser.add_argument("--steps", type=int, default=400, help="simulation steps")
    parser.add_argument("--dt", type=float, default=0.05, help="seconds per step")
    parser.add_argument("--offset", type=float, default=0.0,
                        help="start this many metres off the centreline")
    parser.add_argument("--heading-error", type=float, default=0.0,
                        help="start this many radians off the track heading")
    parser.add_argument("--noise", type=float, default=None,
                        help="sensor noise sigma; higher is a harsher test")
    parser.add_argument("--record", metavar="FILE", help="write the debug view to a video")
    parser.add_argument("--show-mask", action="store_true",
                        help="include the binary lane mask in the recording")
    parser.add_argument("--compare", action="store_true",
                        help="benchmark the alternatives instead of one run")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="KEY=VALUE", help="override any config value")
    return parser


def make_config(args) -> AppConfig:
    config = AppConfig.load(args.config)
    config.display = False
    config.encoder.enabled = False   # the simulator feeds speed back directly
    for override in args.overrides:
        config.set_literal(override)
    return config


def build(config: AppConfig, args):
    simulator = TrackSimulator(TRACKS[args.track](), config)
    if args.noise is not None:
        simulator.noise_sigma = args.noise
    if args.offset or args.heading_error:
        simulator.place_on_track(20, lateral_offset_m=args.offset,
                                 heading_error_rad=args.heading_error)
    return simulator, LaneKeepingPipeline(config, source=_NoSource())


class _Recorder:
    def __init__(self, path: str, pipeline: LaneKeepingPipeline, show_mask: bool,
                 fps: float) -> None:
        self.path = path
        self.pipeline = pipeline
        self.show_mask = show_mask
        self.fps = fps
        self._writer = None

    def __call__(self, result) -> None:
        import cv2

        rendered = self.pipeline.render(result, self.show_mask)
        if self._writer is None:
            height, width = rendered.shape[:2]
            self._writer = cv2.VideoWriter(
                self.path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (width, height)
            )
        self._writer.write(rendered)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            print(f"wrote {self.path}")


def compare(args) -> int:
    header = (f"{'variant':30} {'done':>5} {'mean':>8} {'rms':>7} "
              f"{'max':>7} {'effort':>7} {'speed':>7}")
    print(f"track: {args.track}    (cross-track error in cm, lane is 35 cm wide)")
    print(header)
    print("-" * len(header))

    for name, overrides in COMPARISONS.items():
        config = make_config(args)
        for key, value in overrides.items():
            config.set(key, value)
        simulator, pipeline = build(config, args)
        report = drive(simulator, pipeline, steps=args.steps, dt=args.dt)
        print(f"{name:30} {('yes' if report.completed else 'NO'):>5} "
              f"{report.mean_abs_error_m * 100:8.2f} {report.rms_error_m * 100:7.2f} "
              f"{report.max_abs_error_m * 100:7.2f} {report.steering_effort:7.2f} "
              f"{report.mean_speed_ms:7.2f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING)

    if args.compare:
        return compare(args)

    config = make_config(args)
    simulator, pipeline = build(config, args)

    recorder = None
    if args.record:
        recorder = _Recorder(args.record, pipeline, args.show_mask, 1.0 / args.dt)

    try:
        report = drive(simulator, pipeline, steps=args.steps, dt=args.dt,
                       on_result=recorder)
    finally:
        if recorder is not None:
            recorder.close()

    print(f"track: {args.track}")
    print(report.summary())
    return 0 if report.completed else 1


if __name__ == "__main__":
    sys.exit(main())
