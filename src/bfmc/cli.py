from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

from .config import AppConfig
from .pipeline import FrameResult, LaneKeepingPipeline

__all__ = ["build_parser", "main"]

log = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bfmc",
        description="Lane keeping for the Bosch Future Mobility Challenge vehicle.",
    )
    parser.add_argument("-c", "--config", help="path to a YAML config file")

    source = parser.add_argument_group("source")
    source.add_argument("--backend", choices=["picamera2", "opencv"],
                        help="frame source (overrides the config file)")
    source.add_argument("--video", help="run against a recorded clip; implies opencv")
    source.add_argument("--loop-video", action="store_true", help="replay the clip forever")
    source.add_argument("--max-frames", type=int, help="stop after N frames")

    algorithm = parser.add_argument_group("algorithm")
    algorithm.add_argument("--method", choices=["sliding_window", "hough"],
                           help="lane detector")
    algorithm.add_argument("--law", choices=["pure_pursuit", "stanley"],
                           help="steering control law")
    algorithm.add_argument("--no-feedforward", action="store_true",
                           help="disable the curvature feedforward term")

    output = parser.add_argument_group("output")
    output.add_argument("--no-display", action="store_true", help="run headless, log only")
    output.add_argument("--show-mask", action="store_true",
                        help="add the binary lane mask to the debug view")
    output.add_argument("--record", metavar="FILE", help="write the debug view to a video")
    output.add_argument("--log-csv", metavar="FILE", help="write per-frame telemetry")

    parser.add_argument("--no-encoder", action="store_true",
                        help="skip the serial speed reader")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="KEY=VALUE",
                        help="override any config value, e.g. --set control.law=stanley")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    return parser


def _config_from_args(args: argparse.Namespace) -> AppConfig:
    config = AppConfig.load(args.config)

    if args.video:
        config.camera.backend = "opencv"
        config.camera.source = args.video
    if args.backend:
        config.camera.backend = args.backend
    if args.loop_video:
        config.camera.loop_video = True
    if args.no_display:
        config.display = False
    if args.no_encoder:
        config.encoder.enabled = False
    if args.max_frames is not None:
        config.max_frames = args.max_frames
    if args.method:
        config.detection.method = args.method
    if args.law:
        config.control.law = args.law
    if args.no_feedforward:
        config.control.feedforward = False

    for override in args.overrides:
        config.set_literal(override)
    return config


class _Recorder:
    def __init__(self, path: str, pipeline: LaneKeepingPipeline, show_mask: bool,
                 fps: float = 20.0) -> None:
        self.path = path
        self.pipeline = pipeline
        self.show_mask = show_mask
        self.fps = fps
        self._writer = None

    def __call__(self, result: FrameResult) -> None:
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
            log.info("Wrote %s", self.path)


class _CsvLogger:
    def __init__(self, path: str) -> None:
        self._handle = Path(path).open("w", newline="", encoding="utf-8")
        self._writer: csv.DictWriter | None = None
        self.path = path

    def __call__(self, result: FrameResult) -> None:
        row = result.as_row()
        if self._writer is None:
            self._writer = csv.DictWriter(self._handle, fieldnames=list(row))
            self._writer.writeheader()
        self._writer.writerow(row)

    def close(self) -> None:
        self._handle.close()
        log.info("Wrote %s", self.path)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    try:
        config = _config_from_args(args)
    except (KeyError, ValueError) as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    try:
        with LaneKeepingPipeline(config) as pipeline:
            sinks = []
            if args.record:
                sinks.append(_Recorder(args.record, pipeline, args.show_mask))
            if args.log_csv:
                sinks.append(_CsvLogger(args.log_csv))

            def on_result(result: FrameResult) -> None:
                for sink in sinks:
                    sink(result)
                if not config.display:
                    log.info("%s", result.summary())

            try:
                frames = pipeline.run(
                    on_result=on_result if sinks or not config.display else None,
                    show_mask=args.show_mask,
                )
            finally:
                for sink in sinks:
                    sink.close()
    except (RuntimeError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    log.info("Processed %d frames at %.1f fps", frames, pipeline.fps)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
