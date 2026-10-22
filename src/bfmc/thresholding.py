"""Binary lane masks that do not depend on the absolute brightness of the scene."""

from __future__ import annotations

import cv2
import numpy as np

__all__ = ["brightness_mask", "gradient_mask", "saturation_mask", "lane_mask"]


def brightness_mask(gray: np.ndarray, block_size: int = 51, offset: int = -12) -> np.ndarray:
    """``block_size`` should be wider than a marking but narrower than the frame."""
    block = max(3, block_size | 1)
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block, offset
    )


def gradient_mask(gray: np.ndarray, threshold: float = 0.18, kernel: int = 3) -> np.ndarray:
    """Lane boundaries run near-vertical in the warped view, so only the x derivative carries
    signal.
    """
    sobel_x = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=kernel))
    peak = float(sobel_x.max())
    if peak <= 0.0:
        return np.zeros_like(gray, dtype=np.uint8)
    normalised = sobel_x / peak
    return ((normalised >= threshold) * 255).astype(np.uint8)


def saturation_mask(frame_bgr: np.ndarray, max_saturation: int = 90) -> np.ndarray:
    hls = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HLS)
    return ((hls[:, :, 2] <= max_saturation) * 255).astype(np.uint8)


def lane_mask(
    frame_bgr: np.ndarray,
    block_size: int = 51,
    brightness_offset: int = -12,
    gradient_threshold: float = 0.18,
    use_saturation: bool = True,
    max_saturation: int = 90,
    blur_kernel: int = 5,
    open_kernel: int = 3,
) -> np.ndarray:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    if blur_kernel >= 3:
        gray = cv2.GaussianBlur(gray, (blur_kernel | 1, blur_kernel | 1), 0)

    mask = cv2.bitwise_or(
        brightness_mask(gray, block_size, brightness_offset),
        gradient_mask(gray, gradient_threshold),
    )

    if use_saturation:
        mask = cv2.bitwise_and(mask, saturation_mask(frame_bgr, max_saturation))

    if open_kernel >= 3:
        # Opening drops isolated speckle without eroding a continuous marking.
        element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel, open_kernel))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, element)

    return mask
