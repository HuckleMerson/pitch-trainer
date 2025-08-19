# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, nonecheck=False

# --- Python imports kept as-is ---
import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import List, Tuple

# --- Cython/Numpy types for speed ---
cimport numpy as cnp
from libc.math cimport sqrt, fabs

ctypedef cnp.float32_t f32
ctypedef cnp.float64_t f64
ctypedef cnp.uint8_t   u8
ctypedef Py_ssize_t    isize

cdef inline f64 _hypot(f64 dx, f64 dy) nogil:
    return sqrt(dx*dx + dy*dy)

cdef inline f64 _clamp(f64 t, f64 lo, f64 hi) nogil:
    if t < lo: return lo
    if t > hi: return hi
    return t

# -----------------------------------------------------------------------------
# Data model (unchanged API, Python dataclass)
# -----------------------------------------------------------------------------

@dataclass
class SoccerPitchConfiguration:
    width: int = 7000   # [cm]
    length: int = 12000 # [cm]
    penalty_box_width: int = 4100
    penalty_box_length: int = 2015
    goal_box_width: int = 1832
    goal_box_length: int = 550
    centre_circle_radius: int = 915
    penalty_spot_distance: int = 1100

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
            (self.length - self.penalty_box_length, (self.width - self.penalty_box_width) / 2),  # 18
            (self.length - self.penalty_box_length, (self.width - self.goal_box_width) / 2),  # 19
            (self.length - self.penalty_box_length, (self.width + self.goal_box_width) / 2),  # 20
            (self.length - self.penalty_box_length, (self.width + self.penalty_box_width) / 2),  # 21
            (self.length - self.penalty_spot_distance, self.width / 2),  # 22
            (self.length - self.goal_box_length, (self.width - self.goal_box_width) / 2),  # 23
            (self.length - self.goal_box_length, (self.width + self.goal_box_width) / 2),  # 24
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
        "01","02","03","04","05","06","07","08","09","10",
        "11","12","13","15","16","17","18","20","21","22",
        "23","24","25","26","27","28","29","30","31","32",
        "14","19"
    ])

    colors: List[str] = field(default_factory=lambda: [
        "#FF1493","#FF1493","#FF1493","#FF1493","#FF1493","#FF1493",
        "#FF1493","#FF1493","#FF1493","#FF1493","#FF1493","#FF1493",
        "#FF1493","#00BFFF","#00BFFF","#00BFFF","#00BFFF","#FF6347",
        "#FF6347","#FF6347","#FF6347","#FF6347","#FF6347","#FF6347",
        "#FF6347","#FF6347","#FF6347","#FF6347","#FF6347","#FF6347",
        "#00BFFF","#00BFFF"
    ])

pitch_config = SoccerPitchConfiguration()
pitch_vertices = np.asarray(pitch_config.vertices, dtype=np.float32)

# -----------------------------------------------------------------------------
# Helpers (OPT #1: typed loops + memoryviews)
# -----------------------------------------------------------------------------

def filter_by_density(kept, int radius=80, int max_neighbors=80):
    """
    kept: list[(x1,y1,x2,y2)] or ndarray (N,4)
    O(N^2) neighbor count using typed arrays (no Python tuple unpacking).
    """
    if not kept:
        return []
    cdef cnp.ndarray[f32, ndim=2] a = np.asarray(kept, dtype=np.float32)
    cdef isize n = a.shape[0]
    cdef cnp.ndarray[f32, ndim=1] midx = (a[:,0] + a[:,2]) * 0.5
    cdef cnp.ndarray[f32, ndim=1] midy = (a[:,1] + a[:,3]) * 0.5
    cdef f32[:] mx = midx
    cdef f32[:] my = midy
    cdef isize i, j
    cdef int cnt
    cdef f32 rx = <f32>radius
    cdef f32 ry = <f32>radius
    cdef list out = []
    for i in range(n):
        cnt = 0
        for j in range(n):
            if fabs(mx[i] - mx[j]) < rx and fabs(my[i] - my[j]) < ry:
                cnt += 1
        if cnt <= max_neighbors:
            out.append((<int>a[i,0], <int>a[i,1], <int>a[i,2], <int>a[i,3]))
    return out

def line_on_mask(int x1, int y1, int x2, int y2,
                 cnp.ndarray[u8, ndim=2] mask, int samples=20):
    """
    OPT #2: Replace np.linspace with integer-stepped sampling.
    """
    cdef int H = mask.shape[0]
    cdef int W = mask.shape[1]
    cdef double dt = 1.0 / (samples - 1 if samples > 1 else 1)
    cdef double t = 0.0
    cdef int i, x, y
    cdef double xf, yf
    for i in range(samples):
        xf = x1 + (x2 - x1) * t
        yf = y1 + (y2 - y1) * t
        x = <int>xf
        y = <int>yf
        if (x < 0) or (x >= W) or (y < 0) or (y >= H):
            return False
        if mask[y, x] == 0:
            return False
        t += dt
    return True

# -----------------------------------------------------------------------------
# Detection
# -----------------------------------------------------------------------------

def detect_pitch_lines_tophat(frame, int border_ignore=3):
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 1) basic grass mask
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    grass = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
    grass = cv2.morphologyEx(grass, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    # 2) tophat to enhance white lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 31))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    _, white_lines = cv2.threshold(tophat, 30, 255, cv2.THRESH_BINARY)

    # 3) restriction to grass
    masked = cv2.bitwise_and(white_lines, grass)

    # 4) edges + HoughP
    blur = cv2.GaussianBlur(masked, (5, 5), 0)
    edges = cv2.Canny(blur, 30, 100)
    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 360, threshold=30, minLineLength=10, maxLineGap=15
    )

    # 5) only keep segments that lie on grass
    line_mask = np.zeros_like(gray)
    kept = []
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            if line_on_mask(x1, y1, x2, y2, grass, samples=30):
                cv2.line(line_mask, (x1, y1), (x2, y2), 255, 2)
                kept.append((x1, y1, x2, y2))

    # 6) get rid of borders
    line_mask[:border_ignore, :] = 0
    line_mask[-border_ignore:, :] = 0
    line_mask[:, :border_ignore] = 0
    line_mask[:, -border_ignore:] = 0

    # 7) finalize kept list using midpoint test on mask
    if kept:
        a = np.asarray(kept, dtype=np.int32)
        mx = ((a[:,0] + a[:,2]) // 2).clip(0, w - 1)
        my = ((a[:,1] + a[:,3]) // 2).clip(0, h - 1)
        keep_mask = line_mask[my, mx] > 0
        kept = a[keep_mask].tolist()

    kept = filter_by_density(kept, radius=80, max_neighbors=80)
    return grass, white_lines, masked, edges, line_mask, kept

# -----------------------------------------------------------------------------
# Geometry (typed)
# -----------------------------------------------------------------------------

def point_to_segment_dist(f64 px, f64 py, f64 x1, f64 y1, f64 x2, f64 y2):
    cdef f64 vx = x2 - x1
    cdef f64 vy = y2 - y1
    cdef f64 wx = px - x1
    cdef f64 wy = py - y1
    cdef f64 denom = vx*vx + vy*vy
    if denom == 0.0:
        return _hypot(px - x1, py - y1)
    cdef f64 t = (wx*vx + wy*vy) / denom
    t = _clamp(t, 0.0, 1.0)
    cdef f64 projx = x1 + t * vx
    cdef f64 projy = y1 + t * vy
    return _hypot(px - projx, py - projy)

def _closest_point_on_segment(f64 px, f64 py, f64 x1, f64 y1, f64 x2, f64 y2):
    cdef f64 vx = x2 - x1
    cdef f64 vy = y2 - y1
    cdef f64 wx = px - x1
    cdef f64 wy = py - y1
    cdef f64 denom = vx*vx + vy*vy
    if denom == 0.0:
        return x1, y1, _hypot(px - x1, py - y1)
    cdef f64 t = (wx*vx + wy*vy) / denom
    t = _clamp(t, 0.0, 1.0)
    cdef f64 cx = x1 + t * vx
    cdef f64 cy = y1 + t * vy
    return cx, cy, _hypot(px - cx, py - cy)

cdef inline cnp.ndarray[f32, ndim=2] _ensure_segments_array(segments):
    if isinstance(segments, np.ndarray) and segments.dtype == np.float32 and segments.ndim == 2 and segments.shape[1] == 4:
        return segments
    return np.asarray(segments, dtype=np.float32)

def keypoint_on_line_segments(f64 u, f64 v, segments, f64 tol=3.0):
    cdef cnp.ndarray[f32, ndim=2] seg = _ensure_segments_array(segments)
    cdef f32[:, :] s = seg
    cdef isize i, M = s.shape[0]
    cdef f64 x1,y1,x2,y2
    for i in range(M):
        x1 = s[i,0]; y1 = s[i,1]
        x2 = s[i,2]; y2 = s[i,3]
        if point_to_segment_dist(u, v, x1, y1, x2, y2) <= tol:
            return True
    return False

def keypoint_to_closest_segment(f64 u, f64 v, segments, f64 tol=3.0):
    cdef cnp.ndarray[f32, ndim=2] seg = _ensure_segments_array(segments)
    cdef f32[:, :] s = seg
    cdef isize i, M = s.shape[0]
    cdef f64 x1,y1,x2,y2,cx,cy,d
    cdef f64 best = 1e300
    cdef f64 bx = u
    cdef f64 by = v
    for i in range(M):
        x1 = s[i,0]; y1 = s[i,1]
        x2 = s[i,2]; y2 = s[i,3]
        cx, cy, d = _closest_point_on_segment(u, v, x1, y1, x2, y2)
        if d <= tol:
            return u, v
        if d < best:
            best = d; bx = cx; by = cy
    return bx, by

def detect_valid_keypoints_mask(keypoints, conf_mask, final_kept_lines, f64 tol=15.0):
    """
    keypoints: (N,2) or (N,3); conf_mask: (N,) bool
    """
    cdef cnp.ndarray[f32, ndim=2] seg = _ensure_segments_array(final_kept_lines)
    cdef f32[:, :] s = seg
    cdef isize N = keypoints.shape[0]
    valid = np.zeros((N,), dtype=bool)
    cdef isize i
    cdef f64 x, y
    for i in range(N):
        if not conf_mask[i]:
            continue
        x = float(keypoints[i,0]); y = float(keypoints[i,1])
        if keypoint_on_line_segments(x, y, s, tol):
            valid[i] = True
    return valid

def get_keypoints_close_to_valid(keypoints, conf_mask, final_kept_lines, f64 tol=5.0):
    cdef cnp.ndarray[f32, ndim=2] seg = _ensure_segments_array(final_kept_lines)
    cdef f32[:, :] s = seg
    cdef isize N = keypoints.shape[0]
    out = np.copy(keypoints[:, :2]).astype(np.float32, copy=False)
    cdef isize i
    cdef f64 x, y, nx, ny
    for i in range(N):
        if not conf_mask[i]:
            continue
        x = float(keypoints[i,0]); y = float(keypoints[i,1])
        nx, ny = keypoint_to_closest_segment(x, y, s, tol)
        out[i,0] = nx; out[i,1] = ny
    return out

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def recover_detected_keypoints_only(
    int frame_id,
    original_frame,
    keypoints,
    image_shape,
    fallback_value = (-1.0, -1.0),
    f64 conf_thresh = 0.1,
    int min_correspondences = 3,
    f64 ransac_reproj_frac_of_w = 0.05,
):
    if keypoints is None or keypoints.size == 0:
        return np.array([]), False
    if keypoints.ndim != 2 or keypoints.shape[1] < 2:
        return np.array([]), False

    h, w = image_shape

    N = pitch_vertices.shape[0]
    scale = np.array([w / pitch_config.length, h / pitch_config.width], dtype=np.float32)
    scaled_vertices = pitch_vertices * scale  # (N,2)

    if keypoints.shape[1] >= min_correspondences:
        conf_mask = keypoints[:, 2] > conf_thresh
    else:
        return np.array([]), False

    num_detected = int(np.sum(conf_mask))
    if num_detected < 4:
        return np.array([]), False

    if keypoints.shape[0] != N:
        return np.array([]), False

    _, _, _, _, _, final_kept = detect_pitch_lines_tophat(original_frame)
    final_kept = np.asarray(final_kept, dtype=np.float32).reshape(-1, 4)  # FIX

    # snap detected keypoints to nearest segments (Cython-accelerated)
    nw_keypoints = get_keypoints_close_to_valid(
        keypoints[:, :2], conf_mask, final_kept_lines=final_kept
    )

    detected = nw_keypoints[conf_mask, :2].astype(np.float32)
    layout = scaled_vertices[conf_mask].astype(np.float32)

    ransac_thresh = float(w) * float(ransac_reproj_frac_of_w)

    try:
        H, _ = cv2.findHomography(
            layout, detected, method=cv2.RANSAC, ransacReprojThreshold=ransac_thresh
        )
        if H is None:
            return np.array([]), False
    except Exception:
        return np.array([]), False

    projected = cv2.perspectiveTransform(
        scaled_vertices.reshape(-1, 1, 2).astype(np.float32), H
    ).reshape(-1, 2)

    seed = np.ones(N, dtype=bool)
    valid_mask = detect_valid_keypoints_mask(projected, seed, final_kept_lines=final_kept, tol=15.0)
    valid_mask = np.asarray(valid_mask, dtype=bool).reshape(-1)

    if valid_mask.shape[0] != N:
        return np.array([]), False

    full_result = np.full((N, 2), fallback_value, dtype=np.float32)
    full_result[valid_mask] = projected[valid_mask]
    return full_result, True
