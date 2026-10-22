"""Frame sources. ``picamera2`` is imported lazily so everything else runs off-Pi."""

from __future__ import annotations

import logging
from typing import Protocol

import cv2
import numpy as np

from .config import CameraConfig

__all__ = ["FrameSource", "PiCameraSource", "OpenCVSource", "create_source"]

log = logging.getLogger(__name__)


class FrameSource(Protocol):
    def read(self) -> np.ndarray | None:
        pass

    def close(self) -> None:
        ...


class PiCameraSource:
    def __init__(self, width: int = 640, height: int = 480) -> None:
        from picamera2 import Picamera2

        self._camera = Picamera2()
        self._camera.configure(
            self._camera.create_preview_configuration(main={"size": (width, height)})
        )
        self._camera.start()

    def read(self) -> np.ndarray | None:
        frame = self._camera.capture_array()
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    def close(self) -> None:
        self._camera.stop()


class OpenCVSource:
    def __init__(self, source: object = 0, width: int = 640, height: int = 480,
                 loop_video: bool = False) -> None:
        self._source = source
        self._loop = loop_video
        self._capture = cv2.VideoCapture(source)
        if not self._capture.isOpened():
            raise RuntimeError(f"could not open video source {source!r}")
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read(self) -> np.ndarray | None:
        ok, frame = self._capture.read()
        if not ok:
            if not self._loop:
                return None
            self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self._capture.read()
            if not ok:
                return None
        return frame

    def close(self) -> None:
        self._capture.release()


def create_source(config: CameraConfig) -> FrameSource:
    """Falls back to OpenCV off-Pi."""
    if config.backend == "picamera2":
        try:
            return PiCameraSource(config.width, config.height)
        except ImportError:
            log.warning("picamera2 not available; falling back to the OpenCV backend")
            return OpenCVSource(config.source, config.width, config.height, config.loop_video)
    if config.backend == "opencv":
        return OpenCVSource(config.source, config.width, config.height, config.loop_video)
    raise ValueError(f"unknown camera backend: {config.backend!r}")
