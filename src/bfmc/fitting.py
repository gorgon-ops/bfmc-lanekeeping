from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from .geometry import MIN_DX, Line

__all__ = ["split_by_slope", "weighted_median", "weighted_average_lane"]

Segment = tuple[int, int, int, int]


def split_by_slope(
    segments: Iterable[Segment], min_slope: float
) -> tuple[list[Segment], list[Segment]]:
    """Image y grows downwards, so the left boundary has negative slope; flatter than ``min_slope``
    is noise.
    """
    left: list[Segment] = []
    right: list[Segment] = []
    for x1, y1, x2, y2 in segments:
        dx = x2 - x1
        if abs(dx) < MIN_DX:
            continue
        slope = (y2 - y1) / dx
        if abs(slope) < min_slope:
            continue
        (left if slope < 0 else right).append((x1, y1, x2, y2))
    return left, right


def weighted_median(values: Sequence[float], weights: Sequence[float]) -> float:
    """Median rather than mean, so one spurious segment cannot drag the estimate."""
    order = np.argsort(values)
    sorted_values = np.asarray(values, dtype=float)[order]
    sorted_weights = np.asarray(weights, dtype=float)[order]
    cumulative = np.cumsum(sorted_weights)
    index = int(np.searchsorted(cumulative, cumulative[-1] * 0.5))
    return float(sorted_values[min(index, len(sorted_values) - 1)])


def weighted_average_lane(
    segments: Sequence[Segment],
    height: int,
    min_segment_length: float = 20.0,
    min_slope: float = 0.2,
    top_frac: float = 0.3,
) -> Line | None:
    """Weighted by segment length; ``None`` when nothing survives filtering."""
    if not segments:
        return None

    slopes: list[float] = []
    intercepts: list[float] = []
    weights: list[float] = []

    for x1, y1, x2, y2 in segments:
        dx = x2 - x1
        if abs(dx) < MIN_DX:
            continue
        length = float(np.hypot(dx, y2 - y1))
        if length < min_segment_length:
            continue
        slope = (y2 - y1) / dx
        if abs(slope) < min_slope:
            continue
        slopes.append(slope)
        intercepts.append(y1 - slope * x1)
        weights.append(length)

    if not slopes:
        return None

    slope = weighted_median(slopes, weights)
    intercept = weighted_median(intercepts, weights)

    y_bottom = height
    y_top = int(height * top_frac)
    return Line(
        int((y_bottom - intercept) / slope),
        y_bottom,
        int((y_top - intercept) / slope),
        y_top,
    )
