"""Sliding-window lane search: histogram peaks, windows walked up each boundary, quadratic fit.

Once a fit exists the next frame searches a band around it instead, falling back
to the windows when the band stops finding pixels.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .geometry import LanePolynomial

__all__ = ["LaneSearchResult", "SlidingWindowFinder", "histogram_peaks",
           "robust_fit", "truncate_to_well_conditioned"]


def robust_fit(
    xs: np.ndarray,
    ys: np.ndarray,
    confidence: float = 1.0,
    degree: int = 2,
    inlier_margin_px: float = 25.0,
) -> LanePolynomial | None:
    """One rejection pass: a single detached blob is enough to ruin a least-squares fit."""
    first = LanePolynomial.fit(xs, ys, float(ys.min()), float(ys.max()),
                              confidence=confidence, degree=degree)
    if first is None:
        return None

    residuals = np.abs(xs - first.x_at(ys.astype(float)))
    threshold = max(inlier_margin_px, 2.5 * float(residuals.std()))
    inliers = residuals <= threshold
    if inliers.sum() < len(xs) * 0.5 or inliers.sum() <= degree + 1:
        return first

    kept_x, kept_y = xs[inliers], ys[inliers]
    refit = LanePolynomial.fit(kept_x, kept_y, float(kept_y.min()), float(kept_y.max()),
                               confidence=confidence, degree=degree)
    return refit if refit is not None else first


def truncate_to_well_conditioned(
    fit: LanePolynomial, max_slope: float = 2.1
) -> LanePolynomial:
    """Limit the fit to where its slope stays under ``max_slope``: ``x = f(y)`` degenerates as the
    road turns horizontal.
    """
    if fit.a == 0.0:
        return fit
    for limit in (max_slope, -max_slope):
        y_limit = (limit - fit.b) / (2.0 * fit.a)
        if fit.y_min < y_limit < fit.y_max:
            return LanePolynomial(fit.a, fit.b, fit.c, y_limit, fit.y_max,
                                  fit.confidence)
    return fit


def histogram_peaks(
    mask: np.ndarray, bottom_frac: float = 0.5, min_column_hits: int = 3
) -> tuple:
    """Columns where the boundaries meet the car, from the bottom ``bottom_frac`` of the mask;
    either may be ``None``.
    """
    height, width = mask.shape[:2]
    start_row = int(height * (1.0 - bottom_frac))
    histogram = (mask[start_row:, :] > 0).sum(axis=0)
    midpoint = width // 2

    def peak(columns: np.ndarray, offset: int):
        if columns.size == 0:
            return None
        index = int(np.argmax(columns))
        if columns[index] < min_column_hits:
            return None
        return index + offset

    return peak(histogram[:midpoint], 0), peak(histogram[midpoint:], midpoint)


@dataclass(frozen=True)
class LaneSearchResult:
    left: LanePolynomial | None = None
    right: LanePolynomial | None = None
    left_pixels: np.ndarray = field(default_factory=lambda: np.empty((0, 2), int))
    right_pixels: np.ndarray = field(default_factory=lambda: np.empty((0, 2), int))
    tracked: bool = False


class SlidingWindowFinder:
    def __init__(
        self,
        windows: int = 9,
        margin: int = 60,
        min_pixels: int = 40,
        search_margin: int = 50,
        min_confidence: float = 0.35,
        bottom_frac: float = 0.5,
        degree: int = 2,
        max_slope: float = 2.1,
        max_missed_windows: int = 2,
        edge_margin: int = 8,
        min_span_frac: float = 0.55,
    ) -> None:
        self.windows = windows
        self.margin = margin
        self.min_pixels = min_pixels
        self.search_margin = search_margin
        self.min_confidence = min_confidence
        self.bottom_frac = bottom_frac
        self.degree = degree
        self.max_slope = max_slope
        self.max_missed_windows = max_missed_windows
        self.edge_margin = edge_margin
        self.min_span_frac = min_span_frac

    def find(
        self,
        mask: np.ndarray,
        prior_left: LanePolynomial | None = None,
        prior_right: LanePolynomial | None = None,
    ) -> LaneSearchResult:
        ys, xs = np.nonzero(mask)
        if ys.size == 0:
            return LaneSearchResult()

        height, width = mask.shape[:2]
        tracked = False

        left = right = None
        left_pixels = right_pixels = np.empty((0, 2), int)

        if prior_left is not None or prior_right is not None:
            left, left_pixels = self._search_around(xs, ys, prior_left, height, width)
            right, right_pixels = self._search_around(xs, ys, prior_right, height, width)
            tracked = left is not None or right is not None

        if left is None or right is None:
            base_left, base_right = histogram_peaks(mask, self.bottom_frac)
            if left is None and base_left is not None:
                left, left_pixels = self._slide(xs, ys, base_left, height, width)
            if right is None and base_right is not None:
                right, right_pixels = self._slide(xs, ys, base_right, height, width)
            tracked = tracked and left is not None and right is not None

        left = self._condition(left, height)
        right = self._condition(right, height)
        return LaneSearchResult(left, right, left_pixels, right_pixels, tracked)

    def _condition(self, fit: LanePolynomial | None,
                   height: int) -> LanePolynomial | None:
        """A fit of the nearest few centimetres says little about a corner, so the span is folded
        into confidence.
        """
        if fit is None:
            return None
        fit = truncate_to_well_conditioned(fit, self.max_slope)

        span = (fit.y_max - fit.y_min) / max(1.0, height * self.min_span_frac)
        return replace(fit, confidence=fit.confidence * float(np.clip(span, 0.1, 1.0)))

    def _clipped(self, band_x: np.ndarray, width: int) -> bool:
        """A boundary cut off by the frame leaves a strip hugging the border; fitting it bends the
        curve the wrong way.
        """
        margin = self.edge_margin
        return bool(band_x.min() <= margin or band_x.max() >= width - 1 - margin)

    def _well_conditioned_rows(self, xs, ys, selected, height: int,
                               width: int) -> np.ndarray:
        band_height = max(1, height // self.windows)
        bands = (ys[selected] // band_height).astype(int)
        band_x = xs[selected]

        allowed = []
        for band in np.unique(bands):
            in_band = band_x[bands == band]
            spread = float(np.percentile(in_band, 95) - np.percentile(in_band, 5))
            if spread <= band_height * self.max_slope and not self._clipped(in_band, width):
                allowed.append(band)

        keep = np.zeros_like(selected)
        keep[selected] = np.isin(bands, allowed)
        return keep

    def _connected_to_car(self, ys, selected, height: int) -> np.ndarray:
        """Rows joined to the bottom of the image; anything past a gap of ``max_missed_windows`` is
        another stretch of road.
        """
        band_height = max(1, height // self.windows)
        bands = (ys[selected] // band_height).astype(int)
        occupied = np.unique(bands)
        if occupied.size == 0:
            return np.zeros_like(selected)

        keep = []
        previous = None
        for band in sorted(occupied, reverse=True):  # bottom of the image first
            if previous is not None and previous - band > self.max_missed_windows + 1:
                break
            keep.append(band)
            previous = band

        connected = np.zeros_like(selected)
        connected[selected] = np.isin(bands, keep)
        return connected

    def _search_around(self, xs, ys, prior: LanePolynomial | None,
                       height: int, width: int):
        if prior is None:
            return None, np.empty((0, 2), int)

        centre = prior.x_at(ys.astype(float))
        selected = np.abs(xs - centre) <= self.search_margin
        if int(selected.sum()) < self.min_pixels * 2:
            return None, np.empty((0, 2), int)

        # The band can sweep across a detached stretch of road higher up.
        selected &= self._connected_to_car(ys, selected, height)
        if int(selected.sum()) < self.min_pixels * 2:
            return None, np.empty((0, 2), int)

        rows = ys[selected]
        # Same reason the window walk stops at sideways rows.
        selected &= self._well_conditioned_rows(xs, ys, selected, height, width)
        if int(selected.sum()) < self.min_pixels * 2:
            return None, np.empty((0, 2), int)
        rows = ys[selected]

        # Pixels bunched on a few rows do not constrain a curve.
        spread = np.unique(rows // max(1, height // self.windows)).size / self.windows
        if spread < self.min_confidence:
            return None, np.empty((0, 2), int)

        fit = robust_fit(xs[selected], rows, confidence=min(1.0, spread),
                         degree=self.degree, inlier_margin_px=self.search_margin)
        if fit is None:
            return None, np.empty((0, 2), int)
        return fit, np.stack([xs[selected], rows], axis=1)

    def _slide(self, xs, ys, base_x: int, height: int, width: int):
        """Stops once the boundary is lost: in a corner a different stretch often re-enters near
        the top, and fitting both together bends through the gap, sometimes the wrong way.
        """
        window_height = max(1, height // self.windows)
        current_x = base_x
        collected = []
        hits = 0
        missed = 0

        for index in range(self.windows):
            y_high = height - index * window_height
            y_low = y_high - window_height
            inside = (
                (ys >= y_low) & (ys < y_high)
                & (xs >= current_x - self.margin) & (xs < current_x + self.margin)
            )
            count = int(inside.sum())

            if count >= self.min_pixels:
                band_x = xs[inside]
                # A long sideways smear is the road turning horizontal; better rebuilt from
                # the opposite boundary and the lane width.
                spread = float(np.percentile(band_x, 95) - np.percentile(band_x, 5))
                if spread > window_height * self.max_slope:
                    break
                if self._clipped(band_x, width):
                    break

                hits += 1
                missed = 0
                collected.append(np.stack([band_x, ys[inside]], axis=1))
                # Re-centre so the next window follows the curve.
                current_x = int(band_x.mean())
            else:
                missed += 1
                if missed > self.max_missed_windows:
                    break

            # A boundary that left the frame does not come back as the same boundary.
            if not -self.margin <= current_x <= width + self.margin:
                break

        confidence = hits / self.windows
        if not collected or confidence < self.min_confidence:
            return None, np.empty((0, 2), int)

        pixels = np.concatenate(collected)
        fit = robust_fit(pixels[:, 0], pixels[:, 1], confidence=confidence,
                         degree=self.degree, inlier_margin_px=self.margin)
        if fit is None:
            return None, np.empty((0, 2), int)
        return fit, pixels
