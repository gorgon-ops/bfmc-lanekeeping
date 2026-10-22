from __future__ import annotations

import cv2
import numpy as np
import pytest

from bfmc.config import DetectionConfig, WarpConfig
from bfmc.perception import PerspectiveWarp, create_detector
from bfmc.thresholding import brightness_mask, gradient_mask, lane_mask, saturation_mask

WIDTH, HEIGHT = 640, 480


def road_frame(width=WIDTH, height=HEIGHT, shift=0, brightness=55, marking=235):
    frame = np.full((height, width, 3), brightness, np.uint8)
    horizon = int(height * 0.60)
    frame[:horizon] = 0
    cv2.line(frame, (int(width * 0.10) + shift, height),
             (int(width * 0.38) + shift, horizon), (marking,) * 3, 8)
    cv2.line(frame, (int(width * 0.90) + shift, height),
             (int(width * 0.62) + shift, horizon), (marking,) * 3, 8)
    return frame


class TestThresholding:
    def test_brightness_mask_finds_locally_bright_pixels(self):
        gray = np.full((HEIGHT, WIDTH), 60, np.uint8)
        gray[:, 300:310] = 220
        mask = brightness_mask(gray)
        assert mask[:, 305].mean() > 200
        assert mask[:, 100].mean() < 60

    def test_brightness_mask_is_insensitive_to_overall_light(self):
        """The point of adapting: the same road, brighter, gives the same mask."""
        dim = np.full((HEIGHT, WIDTH), 40, np.uint8)
        dim[:, 300:310] = 150
        bright = dim.astype(np.int16) + 80
        assert np.array_equal(brightness_mask(dim),
                              brightness_mask(np.clip(bright, 0, 255).astype(np.uint8)))

    def test_gradient_mask_finds_vertical_edges(self):
        gray = np.zeros((HEIGHT, WIDTH), np.uint8)
        gray[:, 300:310] = 255
        assert gradient_mask(gray)[:, 295:315].max() == 255

    def test_gradient_mask_handles_a_flat_image(self):
        flat = np.full((HEIGHT, WIDTH), 90, np.uint8)
        assert gradient_mask(flat).max() == 0

    def test_saturation_mask_keeps_grey_and_drops_colour(self):
        frame = np.zeros((10, 2, 3), np.uint8)
        frame[:, 0] = (200, 200, 200)   # grey
        frame[:, 1] = (0, 0, 255)       # saturated red
        mask = saturation_mask(frame)
        assert mask[:, 0].all() and not mask[:, 1].any()

    def test_lane_mask_lights_up_the_markings(self):
        mask = lane_mask(road_frame())
        assert mask.max() == 255
        assert (mask > 0).sum() > 500

    def test_lane_mask_stays_quiet_on_blank_asphalt(self):
        blank = np.full((HEIGHT, WIDTH, 3), 55, np.uint8)
        assert (lane_mask(blank) > 0).mean() < 0.05

    def test_lane_mask_survives_a_brightness_change(self):
        dim = lane_mask(road_frame(brightness=35, marking=150))
        bright = lane_mask(road_frame(brightness=90, marking=255))
        assert (dim > 0).sum() > 500 and (bright > 0).sum() > 500


class TestPerspectiveWarp:
    def test_warp_preserves_the_frame_size(self):
        warp = PerspectiveWarp(WarpConfig(), WIDTH, HEIGHT)
        assert warp(road_frame()).shape == (HEIGHT, WIDTH, 3)

    def test_converging_markings_come_out_parallel(self):
        """After warping, converging markings must be vertical and a constant distance apart."""
        config = WarpConfig()
        warp = PerspectiveWarp(config, WIDTH, HEIGHT)

        frame = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
        horizon = int(HEIGHT * config.horizon_y_frac)
        # Two lines just inside the trapezoid, so they stay off the frame edge.
        inset = 0.04
        for top_frac, bottom_frac in (
            (config.top_left_x_frac + inset, config.bottom_left_x_frac + inset),
            (config.top_right_x_frac - inset, config.bottom_right_x_frac - inset),
        ):
            cv2.line(frame, (int(WIDTH * bottom_frac), HEIGHT - 1),
                     (int(WIDTH * top_frac), horizon), (255, 255, 255), 6)

        def separation(image, row):
            lit = np.nonzero(image[row] > 128)[0]
            return int(lit.max() - lit.min())

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        warped = cv2.cvtColor(warp(frame), cv2.COLOR_BGR2GRAY)

        # In the camera image the markings converge hard towards the horizon.
        raw_convergence = separation(gray, HEIGHT - 20) - separation(gray, horizon + 20)
        # After warping they should hold a near-constant gap.
        warped_convergence = abs(separation(warped, HEIGHT - 20) - separation(warped, 40))

        # The inset is not exactly a straight line under a projective map, so some
        # residual remains; convergence still drops by nearly an order of magnitude.
        assert raw_convergence > 200
        assert warped_convergence < raw_convergence / 5

    def test_unwarp_is_the_inverse_of_warp(self):
        warp = PerspectiveWarp(WarpConfig(), WIDTH, HEIGHT)
        corners = np.float32([[0, 0], [WIDTH, 0], [WIDTH, HEIGHT], [0, HEIGHT]])
        assert np.allclose(warp.unwarp_points(corners), warp.src_points, atol=1.0)


class TestDetectors:
    @pytest.fixture
    def warp(self):
        return PerspectiveWarp(WarpConfig(), WIDTH, HEIGHT)

    @pytest.mark.parametrize("method", ["sliding_window", "hough"])
    def test_both_detectors_find_a_straight_lane(self, warp, method):
        config = DetectionConfig(method=method)
        detection = create_detector(warp, config).detect(road_frame())

        assert detection.left is not None and detection.right is not None
        assert detection.right.x_at(HEIGHT) > detection.left.x_at(HEIGHT)

    @pytest.mark.parametrize("method", ["sliding_window", "hough"])
    def test_both_detectors_find_nothing_in_a_blank_frame(self, warp, method):
        blank = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
        detection = create_detector(warp, DetectionConfig(method=method)).detect(blank)
        assert detection.left is None and detection.right is None

    def test_the_sliding_window_detector_reports_its_pixels(self, warp):
        detection = create_detector(warp, DetectionConfig()).detect(road_frame())
        assert len(detection.left_pixels) > 0
        assert detection.left_pixels.shape[1] == 2

    def test_the_hough_detector_returns_straight_fits(self, warp):
        config = DetectionConfig(method="hough")
        detection = create_detector(warp, config).detect(road_frame())
        assert detection.left.a == 0.0 and detection.right.a == 0.0

    def test_a_prior_is_used_and_can_be_cleared(self, warp):
        detector = create_detector(warp, DetectionConfig())
        frame = road_frame()
        detector.detect(frame)
        assert detector.detect(frame).tracked

        detector.reset()
        assert not detector.detect(frame).tracked

    def test_an_unknown_method_is_rejected(self, warp):
        with pytest.raises(ValueError):
            create_detector(warp, DetectionConfig(method="telepathy"))
