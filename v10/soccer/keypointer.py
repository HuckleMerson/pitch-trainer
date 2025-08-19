# from soccer.config import SoccerPitchConfiguration
import numpy as np
import cv2
from typing import List, Tuple

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class SoccerPitchConfiguration:
    width: int = 7000  # [cm]
    length: int = 12000  # [cm]
    penalty_box_width: int = 4100  # [cm]
    penalty_box_length: int = 2015  # [cm]
    goal_box_width: int = 1832  # [cm]
    goal_box_length: int = 550  # [cm]
    centre_circle_radius: int = 915  # [cm]
    penalty_spot_distance: int = 1100  # [cm]

    @property
    def vertices(self) -> List[Tuple[int, int]]:
        return [
            (0, 0),  # 1
            (0, (self.width - self.penalty_box_width) / 2),  # 2
            (0, (self.width - self.goal_box_width) / 2),  # 3
            (0, (self.width + self.goal_box_width) / 2),  # 4
            (0, (self.width + self.penalty_box_width) / 2),  # 5
            (0, self.width),  # 6
            (self.goal_box_length, (self.width - self.goal_box_width) / 2),  # 7
            (self.goal_box_length, (self.width + self.goal_box_width) / 2),  # 8
            (self.penalty_spot_distance, self.width / 2),  # 9
            (self.penalty_box_length, (self.width - self.penalty_box_width) / 2),  # 10
            (self.penalty_box_length, (self.width - self.goal_box_width) / 2),  # 11
            (self.penalty_box_length, (self.width + self.goal_box_width) / 2),  # 12
            (self.penalty_box_length, (self.width + self.penalty_box_width) / 2),  # 13
            (self.length / 2, 0),  # 14
            (self.length / 2, self.width / 2 - self.centre_circle_radius),  # 15
            (self.length / 2, self.width / 2 + self.centre_circle_radius),  # 16
            (self.length / 2, self.width),  # 17
            (
                self.length - self.penalty_box_length,
                (self.width - self.penalty_box_width) / 2
            ),  # 18
            (
                self.length - self.penalty_box_length,
                (self.width - self.goal_box_width) / 2
            ),  # 19
            (
                self.length - self.penalty_box_length,
                (self.width + self.goal_box_width) / 2
            ),  # 20
            (
                self.length - self.penalty_box_length,
                (self.width + self.penalty_box_width) / 2
            ),  # 21
            (self.length - self.penalty_spot_distance, self.width / 2),  # 22
            (
                self.length - self.goal_box_length,
                (self.width - self.goal_box_width) / 2
            ),  # 23
            (
                self.length - self.goal_box_length,
                (self.width + self.goal_box_width) / 2
            ),  # 24
            (self.length, 0),  # 25
            (self.length, (self.width - self.penalty_box_width) / 2),  # 26
            (self.length, (self.width - self.goal_box_width) / 2),  # 27
            (self.length, (self.width + self.goal_box_width) / 2),  # 28
            (self.length, (self.width + self.penalty_box_width) / 2),  # 29
            (self.length, self.width),  # 30
            (self.length / 2 - self.centre_circle_radius, self.width / 2),  # 31
            (self.length / 2 + self.centre_circle_radius, self.width / 2),  # 32
        ]

    edges: List[Tuple[int, int]] = field(default_factory=lambda: [
        (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (7, 8),
        (10, 11), (11, 12), (12, 13), (14, 15), (15, 16),
        (16, 17), (18, 19), (19, 20), (20, 21), (23, 24),
        (25, 26), (26, 27), (27, 28), (28, 29), (29, 30),
        (1, 14), (2, 10), (3, 7), (4, 8), (5, 13), (6, 17),
        (14, 25), (18, 26), (23, 27), (24, 28), (21, 29), (17, 30)
    ])

    labels: List[str] = field(default_factory=lambda: [
        "01", "02", "03", "04", "05", "06", "07", "08", "09", "10",
        "11", "12", "13", "15", "16", "17", "18", "20", "21", "22",
        "23", "24", "25", "26", "27", "28", "29", "30", "31", "32",
        "14", "19"
    ])

    colors: List[str] = field(default_factory=lambda: [
        "#FF1493", "#FF1493", "#FF1493", "#FF1493", "#FF1493", "#FF1493",
        "#FF1493", "#FF1493", "#FF1493", "#FF1493", "#FF1493", "#FF1493",
        "#FF1493", "#00BFFF", "#00BFFF", "#00BFFF", "#00BFFF", "#FF6347",
        "#FF6347", "#FF6347", "#FF6347", "#FF6347", "#FF6347", "#FF6347",
        "#FF6347", "#FF6347", "#FF6347", "#FF6347", "#FF6347", "#FF6347",
        "#00BFFF", "#00BFFF"
    ])


pitch_config = SoccerPitchConfiguration()
pitch_vertices = np.asarray(pitch_config.vertices, dtype=np.float32)

def filter_by_density(kept, radius=80, max_neighbors=80):
    mids = [((x1 + x2) / 2, (y1 + y2) / 2) for x1, y1, x2, y2 in kept]
    filtered = []
    for i, (x1, y1, x2, y2) in enumerate(kept):
        mx, my = mids[i]
        cnt = sum(
            1 for ux, uy in mids if abs(ux - mx) < radius and abs(uy - my) < radius
        )
        if cnt <= max_neighbors:
            filtered.append((x1, y1, x2, y2))
    return filtered


def line_on_mask(x1, y1, x2, y2, mask, samples=20):
    for t in np.linspace(0, 1, samples):
        x = int(x1 + (x2 - x1) * t)
        y = int(y1 + (y2 - y1) * t)
        if mask[y, x] == 0:
            return False
    return True


def detect_pitch_lines_tophat(frame, border_ignore=3):
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 1) basic grass mask
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    grass = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
    grass = cv2.morphologyEx(grass, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    # 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 31))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    _, white_lines = cv2.threshold(tophat, 30, 255, cv2.THRESH_BINARY)

    # 3) restriction to grass
    masked = cv2.bitwise_and(white_lines, grass)

    # 4) Canny + HoughP
    blur = cv2.GaussianBlur(masked, (5, 5), 0)
    edges = cv2.Canny(blur, 30, 100)
    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 360, threshold=30, minLineLength=10, maxLineGap=15
    )

    # 5) only grass segments
    line_mask = np.zeros_like(gray)
    kept_lines = []
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            if line_on_mask(x1, y1, x2, y2, grass, samples=30):
                cv2.line(line_mask, (x1, y1), (x2, y2), 255, 2)
                kept_lines.append((x1, y1, x2, y2))

    # 6) get rid of borders
    line_mask[:border_ignore, :] = 0
    line_mask[-border_ignore:, :] = 0
    line_mask[:, :border_ignore] = 0
    line_mask[:, -border_ignore:] = 0

    # 7) final list reconstruction
    final_kept = []
    for x1, y1, x2, y2 in kept_lines:
        mx, my = (x1 + x2) // 2, (y1 + y2) // 2
        if 0 <= mx < w and 0 <= my < h and line_mask[my, mx] > 0:
            final_kept.append((x1, y1, x2, y2))

    final_kept = filter_by_density(final_kept, radius=80, max_neighbors=80)

    return grass, white_lines, masked, edges, line_mask, final_kept


def point_to_segment_dist(px, py, x1, y1, x2, y2):
    # vectors
    vx, vy = x2 - x1, y2 - y1
    wx, wy = px - x1, py - y1
    # scalar projection w·v / |v|²
    t = (wx * vx + wy * vy) / float(vx * vx + vy * vy)
    t = max(0, min(1, t))
    # projected point
    projx = x1 + t * vx
    projy = y1 + t * vy
    # euclidian distance
    return np.hypot(px - projx, py - projy)


def _closest_point_on_segment(px, py, x1, y1, x2, y2):
    vx, vy = x2 - x1, y2 - y1
    wx, wy = px - x1, py - y1
    denom = vx * vx + vy * vy

    if denom == 0:  # degenerate segment
        return x1, y1, np.hypot(px - x1, py - y1)

    t = (wx * vx + wy * vy) / denom
    t = max(0.0, min(1.0, t))  # clamp to segment
    cx = x1 + t * vx
    cy = y1 + t * vy
    return cx, cy, np.hypot(px - cx, py - cy)


def keypoint_on_line_segments(u, v, segments, tol=3):
    for x1, y1, x2, y2 in segments:
        if point_to_segment_dist(u, v, x1, y1, x2, y2) <= tol:
            return True
    return False


def keypoint_to_closest_segment(u, v, segments, tol=3):
    best_point = None
    best_dist = float("inf")

    for x1, y1, x2, y2 in segments:
        cx, cy, dist = _closest_point_on_segment(u, v, x1, y1, x2, y2)

        # 1) if within tol, return original (u, v)
        if dist <= tol:
            return u, v

        # 2) track closest point otherwise
        if dist < best_dist:
            best_dist = dist
            best_point = (cx, cy)

    return best_point  # closest point on any segment to (u, v)


def detect_valid_keypoints_mask(keypoints, conf_mask, final_kept_lines, tol=15):
    """
    Returns a boolean mask of keypoints that:
    - Pass the given conf_mask (True means confidence is OK)
    - Are close to any line in final_kept_lines (within tol)

    Args:
        keypoints (np.ndarray): shape (N, 3) — [x, y, conf]
        conf_mask (np.ndarray): shape (N,) — True if confidence > threshold
        final_kept_lines (List[List[int]]): list of lines [x1, y1, x2, y2]
        tol (float): distance tolerance in pixels

    Returns:
        valid_mask (np.ndarray): shape (N,), dtype=bool
    """
    N = keypoints.shape[0]
    valid_mask = np.zeros(N, dtype=bool)

    for i in range(N):
        if not conf_mask[i]:
            continue
        x, y = keypoints[i, :2]
        if keypoint_on_line_segments(x, y, final_kept_lines, tol=tol):
            valid_mask[i] = True

    return valid_mask


def get_keypoints_close_to_valid(keypoints, conf_mask, final_kept_lines, tol=5):
    """
    Returns a boolean mask of keypoints that:
    - Pass the given conf_mask (True means confidence is OK)
    - Are close to any line in final_kept_lines (within tol)

    Args:
        keypoints (np.ndarray): shape (N, 3) — [x, y, conf]
        conf_mask (np.ndarray): shape (N,) — True if confidence > threshold
        final_kept_lines (List[List[int]]): list of lines [x1, y1, x2, y2]
        tol (float): distance tolerance in pixels

    Returns:
        valid_mask (np.ndarray): shape (N,), dtype=bool
    """
    N = keypoints.shape[0]
    nw_keypoints = np.copy(keypoints)

    for i in range(N):
        if not conf_mask[i]:
            continue
        x, y = keypoints[i, :2]
        nw_keypoints[i] = keypoint_to_closest_segment(x, y, final_kept_lines, tol=tol)

    return nw_keypoints


import numpy as np
import cv2
from typing import Tuple

def recover_detected_keypoints_only(
    frame_id: int,
    original_frame: np.ndarray,
    keypoints: np.ndarray,
    image_shape: tuple[int, int],
    fallback_value: tuple[float, float] = (-1.0, -1.0),
    conf_thresh: float = 0.1,
    min_correspondences: int = 3,
    ransac_reproj_frac_of_w: float = 0.05,
) -> Tuple[np.ndarray, bool]:

    if keypoints is None or keypoints.size == 0:
        return np.array([]), False
    if keypoints.ndim != 2 or keypoints.shape[1] < 2:
        return np.array([]), False

    h, w = image_shape

    N = pitch_vertices.shape[0]
    scale = np.array(
        [w / pitch_config.length, h / pitch_config.width], dtype=np.float32
    )
    scaled_vertices = pitch_vertices * scale  # (N, 2), float32

    if keypoints.shape[1] >= min_correspondences:
        conf_mask = keypoints[:, 2] > conf_thresh
    else:
        return np.array([]), False

    num_detected = np.sum(conf_mask)
    if num_detected < 4:
        return np.array([]), False

    if keypoints.shape[0] != N:
        return np.array([]), False

    _, _, _, _, _, final_kept = detect_pitch_lines_tophat(original_frame)

    nw_keypoints = get_keypoints_close_to_valid(
        keypoints[:, :2], conf_mask, final_kept_lines=final_kept
    )

    detected = nw_keypoints[conf_mask, :2].astype(np.float32)
    layout = scaled_vertices[conf_mask].astype(np.float32)  # (K, 2)

    ransac_thresh = float(w) * float(ransac_reproj_frac_of_w)
    # return np.array([]), False

    try:
        H, _ = cv2.findHomography(
            layout, detected, method=cv2.RANSAC, ransacReprojThreshold=ransac_thresh
        )
        if H is None:
            return np.array([]), False
    except Exception as e:
        print(e)
        return np.array([]), False

    layout_points = scaled_vertices.reshape(-1, 1, 2).astype(np.float32)  # (N,1,2)
    projected = cv2.perspectiveTransform(layout_points, H).reshape(-1, 2)  # (N,2)

    seed = np.ones(N, dtype=bool)

    valid_mask = detect_valid_keypoints_mask(
        projected, seed, final_kept_lines=final_kept, tol=15
    )
    valid_mask = np.asarray(valid_mask, dtype=bool).reshape(-1)

    if valid_mask.shape[0] != N:
        return np.array([]), False

    full_result = np.full((N, 2), fallback_value, dtype=np.float32)
    full_result[valid_mask] = projected[valid_mask]

    return full_result, True

