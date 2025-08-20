from soccer.config import SoccerPitchConfiguration
import numpy as np
import cv2
from typing import List, Tuple

pitch_config = SoccerPitchConfiguration()
pitch_vertices = np.asarray(pitch_config.vertices, dtype=np.float32)

# def filter_by_density(kept, radius=80, max_neighbors=80):
#     mids = [((x1 + x2) / 2, (y1 + y2) / 2) for x1, y1, x2, y2 in kept]
#     filtered = []
#     for i, (x1, y1, x2, y2) in enumerate(kept):
#         mx, my = mids[i]
#         cnt = sum(
#             1 for ux, uy in mids if abs(ux - mx) < radius and abs(uy - my) < radius
#         )
#         if cnt <= max_neighbors:
#             filtered.append((x1, y1, x2, y2))
#     return filtered


# def line_on_mask(x1, y1, x2, y2, mask, samples=20):
#     for t in np.linspace(0, 1, samples):
#         x = int(x1 + (x2 - x1) * t)
#         y = int(y1 + (y2 - y1) * t)
#         if mask[y, x] == 0:
#             return False
#     return True


# def detect_pitch_lines_tophat(frame, border_ignore=3):
#     h, w = frame.shape[:2]
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # 1) basic grass mask
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#     grass = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
#     grass = cv2.morphologyEx(grass, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

#     # 2)
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 31))
#     tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
#     _, white_lines = cv2.threshold(tophat, 30, 255, cv2.THRESH_BINARY)

#     # 3) restriction to grass
#     masked = cv2.bitwise_and(white_lines, grass)

#     # 4) Canny + HoughP
#     blur = cv2.GaussianBlur(masked, (5, 5), 0)
#     edges = cv2.Canny(blur, 30, 100)
#     lines = cv2.HoughLinesP(
#         edges, rho=1, theta=np.pi / 360, threshold=30, minLineLength=10, maxLineGap=15
#     )

#     # 5) only grass segments
#     line_mask = np.zeros_like(gray)
#     kept_lines = []
#     if lines is not None:
#         for x1, y1, x2, y2 in lines[:, 0]:
#             if line_on_mask(x1, y1, x2, y2, grass, samples=30):
#                 cv2.line(line_mask, (x1, y1), (x2, y2), 255, 2)
#                 kept_lines.append((x1, y1, x2, y2))

#     # 6) get rid of borders
#     line_mask[:border_ignore, :] = 0
#     line_mask[-border_ignore:, :] = 0
#     line_mask[:, :border_ignore] = 0
#     line_mask[:, -border_ignore:] = 0

#     # 7) final list reconstruction
#     final_kept = []
#     for x1, y1, x2, y2 in kept_lines:
#         mx, my = (x1 + x2) // 2, (y1 + y2) // 2
#         if 0 <= mx < w and 0 <= my < h and line_mask[my, mx] > 0:
#             final_kept.append((x1, y1, x2, y2))

#     # final_kept = filter_by_density(final_kept, radius=80, max_neighbors=80)

#     if len(final_kept) > 60:
#         final_kept.sort(key=lambda seg: (seg[2]-seg[0])**2 + (seg[3]-seg[1])**2)
#         final_kept = final_kept[:30] + final_kept[-30:]

#     return grass, white_lines, masked, edges, line_mask, final_kept

# Reusable kernels (avoid re-allocating every call)
_KERNEL_OPEN_5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
_KERNEL_TOPHAT_31 = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 31))

def _filter_by_density_numpy(kept, radius=80, max_neighbors=80):
    """Vectorized version of your filter_by_density (Chebyshev neighborhood)."""
    if not kept:
        return kept
    segs = np.asarray(kept, dtype=np.int32)
    mids = (segs[:, [0, 1]] + segs[:, [2, 3]]) * 0.5  # (N,2)
    dx = np.abs(mids[:, 0:1] - mids[:, 0])  # (N,N)
    dy = np.abs(mids[:, 1:2] - mids[:, 1])
    # Chebyshev square: abs(dx)<r & abs(dy)<r
    neigh_counts = ((dx < radius) & (dy < radius)).sum(axis=1)
    keep_mask = neigh_counts <= max_neighbors
    return [tuple(x) for x in segs[keep_mask].tolist()]

def _lines_on_mask_batch(segments, mask, samples=20):
    """
    Vectorized line-on-mask test for many segments at once.
    segments: (N,4) int32 (x1,y1,x2,y2), mask: uint8 {0,255}
    Returns boolean array (N,) True if all sampled points lie on nonzero mask.
    """
    if len(segments) == 0:
        return np.empty((0,), dtype=bool)

    segs = segments.astype(np.float32)
    t = np.linspace(0.0, 1.0, samples, dtype=np.float32)  # (S,)
    # (N,1) + (N,1)*(S,) -> (N,S)
    xs = segs[:, 0:1] + (segs[:, 2:3] - segs[:, 0:1]) * t
    ys = segs[:, 1:2] + (segs[:, 3:4] - segs[:, 1:2]) * t
    xs = np.clip(xs.round().astype(np.int32), 0, mask.shape[1]-1)
    ys = np.clip(ys.round().astype(np.int32), 0, mask.shape[0]-1)
    vals = mask[ys, xs]  # (N,S)
    return (vals != 0).all(axis=1)

def detect_pitch_lines_tophat(frame, border_ignore=3):
    h, w = frame.shape[:2]

    # Convert once; keep arrays as uint8
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 1) Grass mask (tighten with open to reduce noise)
    grass = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
    grass = cv2.morphologyEx(grass, cv2.MORPH_OPEN, _KERNEL_OPEN_5, iterations=1)

    # 2) Top-hat for bright lines
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, _KERNEL_TOPHAT_31)
    _, white_lines = cv2.threshold(tophat, 30, 255, cv2.THRESH_BINARY)

    # 3) Restrict to grass
    masked = cv2.bitwise_and(white_lines, grass)

    # 4) Canny + HoughP
    #    Box blur is faster; works similarly here
    blur = cv2.blur(masked, (5, 5))
    edges = cv2.Canny(blur, 30, 100)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 360,
        threshold=30,
        minLineLength=10,
        maxLineGap=15,
    )

    # Prepare border bounds once
    x_lo, x_hi = border_ignore, w - border_ignore - 1
    y_lo, y_hi = border_ignore, h - border_ignore - 1

    kept_lines = []
    line_mask = np.zeros_like(gray, dtype=np.uint8)

    if lines is not None:
        segs = lines[:, 0].astype(np.int32)  # (N,4)
        # Discard segments whose midpoint lies in the ignored border band BEFORE any drawing
        mids = (segs[:, [0, 1]] + segs[:, [2, 3]]) >> 1  # integer midpoint
        in_border = (
            (mids[:, 0] < x_lo) | (mids[:, 0] > x_hi) |
            (mids[:, 1] < y_lo) | (mids[:, 1] > y_hi)
        )
        segs = segs[~in_border]

        # Keep only segments that lie on grass (batched, low Python overhead)
        on_grass = _lines_on_mask_batch(segs, grass, samples=30)
        segs = segs[on_grass]

        # Draw remaining segments
        for x1, y1, x2, y2 in segs:
            cv2.line(line_mask, (x1, y1), (x2, y2), 255, 2)

        kept_lines = [tuple(s) for s in segs.tolist()]

    # 6) Explicitly clear borders (cheap; ensures mask is clean)
    if border_ignore > 0:
        line_mask[:border_ignore, :] = 0
        line_mask[-border_ignore:, :] = 0
        line_mask[:, :border_ignore] = 0
        line_mask[:, -border_ignore:] = 0

    # 7) Final list reconstruction from mask (fast boolean lookup)
    final_kept = []
    if kept_lines:
        segs = np.asarray(kept_lines, dtype=np.int32)
        mids = (segs[:, [0, 1]] + segs[:, [2, 3]]) >> 1  # (N,2)
        mx, my = mids[:, 0], mids[:, 1]
        valid = (
            (mx >= 0) & (mx < w) &
            (my >= 0) & (my < h) &
            (line_mask[my, mx] > 0)
        )
        final_kept = [tuple(s) for s in segs[valid].tolist()]

    # Optional: density filter (vectorized). Uncomment if you found it helpful.
    # final_kept = _filter_by_density_numpy(final_kept, radius=80, max_neighbors=80)

    # Cap number of segments while preserving extremes (same logic as yours)
    if len(final_kept) > 60:
        final_kept.sort(key=lambda seg: (seg[2]-seg[0])**2 + (seg[3]-seg[1])**2)
        final_kept = final_kept[:30] + final_kept[-30:]

    return grass, white_lines, masked, edges, line_mask, final_kept

def detect_pitch_lines_final_fastest(frame, border_ignore=3,
                                     density_radius=80.0,
                                     density_max_neighbors=80):
    h, w = frame.shape[:2]

    # Grass mask (fast morphological open with small kernel)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    grass = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
    cv2.morphologyEx(grass, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), dst=grass)

    # White lines restricted to grass
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 31))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    _, white_lines = cv2.threshold(tophat, 30, 255, cv2.THRESH_BINARY)
    masked = cv2.bitwise_and(white_lines, grass)

    # Edge detection
    edges = cv2.Canny(masked, 30, 100)

    # Detect line segments (direct from edges)
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 360,
                            threshold=30, minLineLength=10, maxLineGap=15)

    if lines is None:
        return []

    # Vectorized midpoint/border check
    kept_lines = []
    for x1, y1, x2, y2 in lines[:, 0]:
        mx, my = (x1 + x2) // 2, (y1 + y2) // 2
        if (border_ignore <= mx < w - border_ignore and
            border_ignore <= my < h - border_ignore and
            grass[my, mx] > 0):
            kept_lines.append((x1, y1, x2, y2))

    # Density pruning (O(n) grid binning)
    if kept_lines:
        mids = np.array([((x1 + x2) * 0.5, (y1 + y2) * 0.5) for x1, y1, x2, y2 in kept_lines], dtype=np.float32)
        r = float(density_radius)
        ix = (mids[:, 0] // r).astype(np.int32)
        iy = (mids[:, 1] // r).astype(np.int32)

        buckets = {}
        for i, key in enumerate(zip(ix, iy)):
            buckets.setdefault(tuple(key), []).append(i)

        keep_mask = np.ones(len(kept_lines), dtype=bool)
        for i, (mx, my, bx, by) in enumerate(zip(mids[:, 0], mids[:, 1], ix, iy)):
            cnt = 0
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    lst = buckets.get((bx + dx, by + dy))
                    if lst:
                        diffs = mids[lst] - (mx, my)
                        cnt += np.count_nonzero((np.abs(diffs[:, 0]) < r) & (np.abs(diffs[:, 1]) < r))
            if cnt > density_max_neighbors:
                keep_mask[i] = False
        final_kept = [seg for seg, keep in zip(kept_lines, keep_mask) if keep]
    else:
        final_kept = []

    return grass, white_lines, masked, edges, [], final_kept

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

from pathlib import Path

final_kept_dir = Path("final_kept")
final_kept_dir.mkdir(parents=True, exist_ok=True)



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

    if True:

        vis_frame = original_frame.copy()
        for line in final_kept:
            x1, y1, x2, y2 = map(int, line)
            cv2.line(vis_frame, (x1, y1), (x2, y2), color=(0, 255, 255), thickness=2)  # Yellow lines

        # Save visualized frame
        out_path = final_kept_dir / f"frame_{frame_id}.jpg"
        cv2.imwrite(str(out_path), vis_frame)

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
        projected, seed, final_kept_lines=final_kept, tol = 15
    )
    valid_mask = np.asarray(valid_mask, dtype=bool).reshape(-1)

    if valid_mask.shape[0] != N:
        return np.array([]), False

    full_result = np.full((N, 2), fallback_value, dtype=np.float32)
    full_result[valid_mask] = projected[valid_mask]

    return full_result, True


# def recover_detected_keypoints_only(
#     frame_id: int,
#     original_frame: np.ndarray,
#     keypoints: np.ndarray,  # (N, 3) -> x, y, conf
#     image_shape: tuple[int, int],  # (h, w)
#     pitch_config=pitch_config,
#     fallback_value: tuple[float, float] = (-1.0, -1.0),
#     conf_thresh: float = 0.1,
#     max_correspondences: int = 32,
#     min_correspondences: int = 4,
#     ransac_reproj_frac_of_w: float = 0.05,
# ) -> Tuple[np.ndarray, bool]:
#     """
#     Estimate homography from detected pitch keypoints and reproject *all* pitch layout points.
#     Returns:
#         full_result: (N, 2) float32 array of reprojected points. Invalid entries set to fallback_value.
#         ok:         bool, True if homography was found with enough correspondences, else False.
#     """

#     # ---- Basic guards ----
#     if keypoints is None or keypoints.size == 0:
#         return np.array([]), False
#     if keypoints.ndim != 2 or keypoints.shape[1] < 2:
#         return np.array([]), False

#     h, w = image_shape
#     pitch_vertices = np.asarray(pitch_config.vertices, dtype=np.float32)  # (N, 2)
#     N = pitch_vertices.shape[0]

#     # ---- Scale pitch model to image size ----
#     scale = np.array(
#         [w / pitch_config.length, h / pitch_config.width], dtype=np.float32
#     )
#     scaled_vertices = pitch_vertices * scale  # (N, 2), float32

#     # ---- Confidence mask from detections ----
#     if keypoints.shape[1] >= 3:
#         conf_mask = keypoints[:, 2] > conf_thresh
#     else:
#         # If no confidence column, assume all detected
#         conf_mask = np.ones(keypoints.shape[0], dtype=bool)

#     # Ensure keypoints length matches vertices count
#     if keypoints.shape[0] != N:
#         # If mismatch, bail out (or adapt here to intersect by id)
#         full_result = np.full((N, 2), fallback_value, dtype=np.float32)
#         return full_result, False

#     # ---- Pitch-line-based validity filter (once per frame) ----
#     # final_kept is used both for selecting reliable correspondences and validating projections
#     _, _, _, _, _, final_kept = detect_pitch_lines_tophat(original_frame)

#     valid_mask_1 = detect_valid_keypoints_mask(
#         keypoints[:, :2], conf_mask, final_kept_lines=final_kept
#     )
#     valid_mask_1 = np.asarray(valid_mask_1, dtype=bool).reshape(-1)
#     if valid_mask_1.shape[0] != N:
#         full_result = np.full((N, 2), fallback_value, dtype=np.float32)
#         return full_result, False

#     # ---- Select up to K reliable correspondences (detected & valid) ----
#     true_idx = np.flatnonzero(valid_mask_1)
#     if true_idx.size == 0:
#         full_result = np.full((N, 2), fallback_value, dtype=np.float32)
#         return full_result, False

#     # Deterministic top-K (first K). Replace with smarter ranking if needed.
#     sel_idx = true_idx[: min(max_correspondences, true_idx.size)]

#     # Need at least 4 for homography
#     if sel_idx.size < min_correspondences:
#         full_result = np.full((N, 2), fallback_value, dtype=np.float32)
#         return full_result, False

#     # detected = keypoints[sel_idx, :2].astype(np.float32)     # (K, 2)
#     detected = keypoints[conf_mask]
#     layout = scaled_vertices[sel_idx].astype(np.float32)  # (K, 2)

#     # ---- Estimate homography ----
#     ransac_thresh = float(w) * float(ransac_reproj_frac_of_w)
#     H, _ = cv2.findHomography(
#         layout, detected, method=cv2.RANSAC, ransacReprojThreshold=ransac_thresh
#     )
#     if H is None:
#         full_result = np.full((N, 2), fallback_value, dtype=np.float32)
#         return full_result, False

#     # ---- Reproject all layout points ----
#     layout_points = scaled_vertices.reshape(-1, 1, 2).astype(np.float32)  # (N,1,2)
#     projected = cv2.perspectiveTransform(layout_points, H).reshape(-1, 2)  # (N,2)

#     # ---- Validate projections with the same pitch-line rule ----
#     # Seed mask: start as all True (we want to cull only by validity function)
#     seed = np.ones(N, dtype=bool)
#     valid_mask_2 = detect_valid_keypoints_mask(
#         projected, seed, final_kept_lines=final_kept
#     )
#     valid_mask_2 = np.asarray(valid_mask_2, dtype=bool).reshape(-1)

#     if valid_mask_2.shape[0] != N:
#         full_result = np.full((N, 2), fallback_value, dtype=np.float32)
#         return full_result, False

#     # ---- Build final result ----
#     full_result = np.full((N, 2), fallback_value, dtype=np.float32)
#     full_result[valid_mask_2] = projected[valid_mask_2]

#     return full_result, True
