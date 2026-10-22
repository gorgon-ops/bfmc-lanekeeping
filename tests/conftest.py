from __future__ import annotations

from collections.abc import Sequence
from typing import Callable

import numpy as np
import pytest

from bfmc.config import AppConfig
from bfmc.perception import PerspectiveWarp

WIDTH, HEIGHT = 640, 480


class FakeSource:
    def __init__(self, frames: Sequence[np.ndarray] = ()) -> None:
        self.frames: list[np.ndarray] = list(frames)
        self.closed = False
        self._index = 0

    def read(self) -> np.ndarray | None:
        if self._index >= len(self.frames):
            return None
        frame = self.frames[self._index]
        self._index += 1
        return frame

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def offline_config() -> AppConfig:
    config = AppConfig()
    config.camera.width = WIDTH
    config.camera.height = HEIGHT
    config.display = False
    config.encoder.enabled = False
    return config


@pytest.fixture
def fake_source() -> Callable[..., FakeSource]:
    return FakeSource


def _road_frame(width: int = WIDTH, height: int = HEIGHT, shift: int = 0,
                brightness: int = 55, curve: float = 0.0) -> np.ndarray:
    """Drawn in bird's-eye space and projected out through the pipeline's own warp, so
    the lane matches the configured trapezoid and width. ``curve`` > 0 bends right.
    """
    import cv2

    config = AppConfig()
    warp = PerspectiveWarp(config.warp, width, height)
    half_width = config.tracking.lane_width_px / 2.0
    centre_x = width / 2.0 + shift

    birdseye = np.full((height, width, 3), brightness, np.uint8)
    for side in (-1, 1):
        points = np.array([
            [centre_x + side * half_width + curve * (height - y) ** 2, y]
            for y in range(0, height + 1, 8)
        ], np.int32)
        cv2.polylines(birdseye, [points], False, (235, 235, 235), 8)

    return cv2.warpPerspective(birdseye, warp.inverse_matrix, (width, height))


@pytest.fixture
def road_frame() -> Callable[..., np.ndarray]:
    return _road_frame


@pytest.fixture
def blank_frame() -> np.ndarray:
    return np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
