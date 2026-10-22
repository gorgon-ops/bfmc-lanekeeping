"""Lanes as ``x = a*y**2 + b*y + c`` in the warped frame, y growing downwards.

``y = height`` is the row nearest the vehicle, so looking ahead means decreasing y.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import NamedTuple

import numpy as np

__all__ = ["Line", "MetricScale", "LanePolynomial", "lane_center", "x_at_y"]

#: Below this horizontal span a Hough segment is treated as vertical.
MIN_DX = 2


class Line(NamedTuple):
    x1: int
    y1: int
    x2: int
    y2: int


def x_at_y(line: Line, y: float) -> int:
    x1, y1, x2, y2 = line
    dx = x2 - x1
    if abs(dx) < MIN_DX:
        return int(x1)
    slope = (y2 - y1) / dx
    if abs(slope) < 1e-6:
        return int(x1)
    return int((y - (y1 - slope * x1)) / slope)


@dataclass(frozen=True)
class MetricScale:
    """``m_per_px_x``: lane width / its warped pixel width; ``m_per_px_y``: depth covered / image
    height.
    """

    m_per_px_x: float = 0.0025
    m_per_px_y: float = 0.0100

    def to_metres(self, dx_px: float = 0.0, dy_px: float = 0.0) -> tuple:
        return dx_px * self.m_per_px_x, dy_px * self.m_per_px_y


PIXEL_SCALE = MetricScale(1.0, 1.0)


@dataclass(frozen=True)
class LanePolynomial:
    a: float
    b: float
    c: float
    y_min: float
    y_max: float
    #: Fraction of sampled rows that supported the fit, 0 to 1.
    confidence: float = 1.0

    @classmethod
    def from_line(cls, line: Line, confidence: float = 1.0) -> LanePolynomial:
        x1, y1, x2, y2 = line
        if y1 == y2:
            raise ValueError("a horizontal segment is not a lane boundary")
        b = (x2 - x1) / (y2 - y1)
        c = x1 - b * y1
        return cls(0.0, b, c, min(y1, y2), max(y1, y2), confidence)

    @classmethod
    def fit(
        cls,
        xs: Sequence[float],
        ys: Sequence[float],
        y_min: float,
        y_max: float,
        confidence: float = 1.0,
        degree: int = 2,
    ) -> LanePolynomial | None:
        # x(y) rather than y(x): a near-vertical boundary is single-valued in y.
        if len(xs) != len(ys):
            raise ValueError("xs and ys must be the same length")
        if len(xs) <= degree:
            return None
        if len(set(ys)) <= degree:
            # All samples on too few rows: the fit would be unconstrained.
            return None

        coefficients = np.polyfit(np.asarray(ys, float), np.asarray(xs, float), degree)
        if degree == 1:
            coefficients = np.insert(coefficients, 0, 0.0)
        a, b, c = (float(v) for v in coefficients[-3:])
        if not all(math.isfinite(v) for v in (a, b, c)):
            return None
        return cls(a, b, c, y_min, y_max, confidence)

    def x_at(self, y: float | np.ndarray) -> float | np.ndarray:
        result = self.a * np.asarray(y, float) ** 2 + self.b * np.asarray(y, float) + self.c
        return float(result) if np.isscalar(y) or np.ndim(y) == 0 else result

    def slope_at(self, y: float) -> float:
        """``dx/dy`` at row ``y``."""
        return 2.0 * self.a * y + self.b

    def shifted(self, dx_px: float) -> LanePolynomial:
        return replace(self, c=self.c + dx_px)

    def points(self, count: int = 24) -> np.ndarray:
        ys = np.linspace(self.y_max, self.y_min, count)
        xs = self.x_at(ys)
        return np.stack([xs, ys], axis=1).astype(np.int32)

    def heading_rad(self, y: float, scale: MetricScale = PIXEL_SCALE) -> float:
        """Positive means the lane runs off to the right."""
        return math.atan2(-scale.m_per_px_x * self.slope_at(y), scale.m_per_px_y)

    def signed_curvature(self, y: float, scale: MetricScale = PIXEL_SCALE) -> float:
        """1/m, positive for a right-hand bend."""
        # Re-express the fit in metres: X = A*Y^2 + B*Y + C.
        a_m = self.a * scale.m_per_px_x / (scale.m_per_px_y ** 2)
        b_m = self.b * scale.m_per_px_x / scale.m_per_px_y
        y_m = y * scale.m_per_px_y
        derivative = 2.0 * a_m * y_m + b_m
        return 2.0 * a_m / (1.0 + derivative ** 2) ** 1.5

    def mean_curvature(self, scale: MetricScale = PIXEL_SCALE) -> float:
        """Heading change over the fit divided by its length: a first derivative.

        Against ground-truth arcs of 0.6-5 m radius this errs by 1-23 % where the
        second derivative errs by 3-41 %.
        """
        span_px = self.y_max - self.y_min
        if span_px <= 1.0:
            return self.signed_curvature(self.y_max, scale)

        heading_near = self.heading_rad(self.y_max, scale)
        heading_far = self.heading_rad(self.y_min, scale)
        distance_m = span_px * scale.m_per_px_y
        return (heading_far - heading_near) / distance_m

    def curve_radius_m(self, y: float, scale: MetricScale = PIXEL_SCALE) -> float:
        curvature = self.signed_curvature(y, scale)
        if curvature == 0.0:
            return math.inf
        return abs(1.0 / curvature)

    def tightest_radius_m(
        self, scale: MetricScale = PIXEL_SCALE, segments: int = 4
    ) -> float:
        """The sharpest radius the car can currently see, so it slows before the corner."""
        edges = np.linspace(self.y_min, self.y_max, segments + 1)
        distance_m = (edges[1] - edges[0]) * scale.m_per_px_y
        if distance_m <= 0.0:
            return self.curve_radius_m(self.y_max, scale)

        headings = [self.heading_rad(float(y), scale) for y in edges]
        peak = max(abs(headings[i + 1] - headings[i]) for i in range(segments)) / distance_m
        return math.inf if peak == 0.0 else 1.0 / peak


def lane_center(
    left: LanePolynomial, right: LanePolynomial, confidence: float | None = None
) -> LanePolynomial:
    """Mean of the two fits, trusted only as far as both reach.

    Shifting the longer boundary half a lane across looks like free range but is
    only right on a straight; over a lap it roughly tripled cross-track error.
    """
    shared = min(left.confidence, right.confidence) if confidence is None else confidence
    return LanePolynomial(
        a=(left.a + right.a) / 2.0,
        b=(left.b + right.b) / 2.0,
        c=(left.c + right.c) / 2.0,
        y_min=max(left.y_min, right.y_min),
        y_max=min(left.y_max, right.y_max),
        confidence=shared,
    )
