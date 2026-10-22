from __future__ import annotations

import math

import cv2
import numpy as np

from .control import DriveCommand
from .perception import PerspectiveWarp
from .states import DrivingState
from .tracking import LaneSnapshot

__all__ = ["draw_lane_region", "draw_birdseye", "draw_hud", "compose"]

FONT = cv2.FONT_HERSHEY_SIMPLEX

COLOR_LIVE = (0, 255, 255)        # detected this frame
COLOR_STALE = (0, 165, 255)       # replayed from memory
COLOR_INFERRED = (255, 120, 120)  # derived from the opposite boundary
COLOR_CENTER = (0, 255, 0)
COLOR_TEXT = (235, 235, 235)
COLOR_DIM = (160, 160, 160)
COLOR_ALERT = (0, 80, 255)

STATE_COLORS = {
    DrivingState.LANE_KEEPING: (0, 220, 0),
    DrivingState.COASTING: (0, 180, 255),
    DrivingState.LEFT_ONLY: (255, 140, 0),
    DrivingState.RIGHT_ONLY: (255, 140, 0),
    DrivingState.STOP: (0, 60, 255),
}


def _boundary_color(stale: bool, inferred: bool):
    if inferred:
        return COLOR_INFERRED
    if stale:
        return COLOR_STALE
    return COLOR_LIVE


def draw_lane_region(
    frame: np.ndarray, warp: PerspectiveWarp, snapshot: LaneSnapshot, alpha: float = 0.3
) -> np.ndarray:
    """Boundaries projected through the inverse warp: any warp or fit error shows as shading off the
    road.
    """
    if snapshot.left is None or snapshot.right is None:
        return frame

    left = warp.unwarp_points(snapshot.left.points(32))
    right = warp.unwarp_points(snapshot.right.points(32))
    polygon = np.concatenate([left, right[::-1]]).astype(np.int32)

    shade = frame.copy()
    color = STATE_COLORS.get(snapshot.state, COLOR_CENTER)
    cv2.fillPoly(shade, [polygon], color)
    cv2.addWeighted(shade, alpha, frame, 1.0 - alpha, 0, dst=frame)

    if snapshot.center is not None:
        center = warp.unwarp_points(snapshot.center.points(32)).astype(np.int32)
        cv2.polylines(frame, [center], False, COLOR_CENTER, 2)
    return frame


def draw_birdseye(
    warped: np.ndarray,
    snapshot: LaneSnapshot,
    command: DriveCommand,
    vehicle_x: int,
    left_pixels: np.ndarray | None = None,
    right_pixels: np.ndarray | None = None,
) -> np.ndarray:
    height = warped.shape[0]

    # Sparse clusters here explain a low confidence better than any number.
    for pixels, color in ((left_pixels, (90, 90, 255)), (right_pixels, (255, 160, 60))):
        if pixels is not None and len(pixels):
            warped[pixels[:, 1], pixels[:, 0]] = color

    for lane, stale, inferred in (
        (snapshot.left, snapshot.left_stale, snapshot.left_inferred),
        (snapshot.right, snapshot.right_stale, snapshot.right_inferred),
    ):
        if lane is not None:
            cv2.polylines(warped, [lane.points()], False,
                          _boundary_color(stale, inferred), 3)

    if snapshot.center is not None:
        cv2.polylines(warped, [snapshot.center.points()], False, COLOR_CENTER, 2)

    cv2.drawMarker(warped, (int(vehicle_x), height - 6), (255, 255, 255),
                   cv2.MARKER_TRIANGLE_UP, 18, 2)

    if command.goal_point is not None:
        goal_x, goal_y = command.goal_point
        cv2.line(warped, (int(vehicle_x), height), (goal_x, goal_y), (0, 200, 0), 2)
        cv2.circle(warped, (goal_x, goal_y), 7, COLOR_CENTER, -1)
        cv2.circle(warped, (goal_x, goal_y), 9, (0, 0, 0), 1)

    return warped


def _steering_bar(canvas: np.ndarray, origin, width, angle_deg, max_deg):
    x, y = origin
    cv2.rectangle(canvas, (x, y), (x + width, y + 10), (70, 70, 70), 1)
    cv2.line(canvas, (x + width // 2, y - 2), (x + width // 2, y + 12), (110, 110, 110), 1)

    fraction = float(np.clip(angle_deg / max_deg, -1.0, 1.0))
    end = int(x + width // 2 + fraction * width // 2)
    cv2.rectangle(canvas, (x + width // 2, y + 1), (end, y + 9), COLOR_CENTER, -1)


def draw_hud(
    frame: np.ndarray,
    snapshot: LaneSnapshot,
    command: DriveCommand,
    speed_ms: float,
    fps: float = 0.0,
    max_steer_deg: float = 30.0,
) -> np.ndarray:
    panel = frame.copy()
    cv2.rectangle(panel, (0, 0), (330, 214), (0, 0, 0), -1)
    cv2.addWeighted(panel, 0.55, frame, 0.45, 0, dst=frame)

    state_color = STATE_COLORS.get(snapshot.state, COLOR_TEXT)
    cv2.putText(frame, str(snapshot.state), (14, 30), FONT, 0.75, state_color, 2)
    if fps:
        cv2.putText(frame, f"{fps:5.1f} fps", (240, 30), FONT, 0.5, COLOR_DIM, 1)

    cv2.putText(frame, f"steer {command.steering_deg:+6.1f} deg", (14, 60),
                FONT, 0.6, COLOR_TEXT, 1)
    _steering_bar(frame, (14, 68), 300, command.steering_deg, max_steer_deg)

    radius = snapshot.curve_radius_m
    radius_text = "straight" if not math.isfinite(radius) else f"{radius:5.2f} m"

    rows = [
        (f"speed  {speed_ms:5.2f} -> {command.target_speed_ms:4.2f} m/s", COLOR_TEXT),
        (f"offset {snapshot.lateral_offset_m * 100:+6.1f} cm", COLOR_TEXT),
        (f"headng {math.degrees(snapshot.heading_rad):+6.1f} deg", COLOR_TEXT),
        (f"{snapshot.road_shape.lower():<12s} R {radius_text}", COLOR_DIM),
        (f"ff {command.feedforward_deg:+5.1f}  fb {command.feedback_deg:+5.1f}  "
         f"ld {command.lookahead_m:.2f}m", COLOR_DIM),
        (f"conf {snapshot.confidence:.2f}   age L{snapshot.left_age} R{snapshot.right_age}",
         COLOR_DIM),
    ]
    y = 100
    for text, color in rows:
        cv2.putText(frame, text, (14, y), FONT, 0.5, color, 1)
        y += 19

    if snapshot.width_rejected:
        cv2.putText(frame, "lane width rejected", (14, y), FONT, 0.5, COLOR_ALERT, 1)

    return frame


def compose(*panes: np.ndarray) -> np.ndarray:
    prepared = []
    height = max(pane.shape[0] for pane in panes)
    for pane in panes:
        if pane.ndim == 2:
            pane = cv2.cvtColor(pane, cv2.COLOR_GRAY2BGR)
        if pane.shape[0] != height:
            scale = height / pane.shape[0]
            pane = cv2.resize(pane, (int(pane.shape[1] * scale), height))
        prepared.append(pane)
    return np.hstack(prepared)
