from soccer.config import SoccerPitchConfiguration
import numpy as np
import cv2

pitch_config = SoccerPitchConfiguration()

def line_on_mask(x1, y1, x2, y2, mask, samples=20):
    for t in np.linspace(0, 1, samples):
        x = int(x1 + (x2 - x1) * t)
        y = int(y1 + (y2 - y1) * t)
        if mask[y, x] == 0:
            return False
    return True

def recover_centered_pitch_points(
    frame: np.ndarray,
    pitch_config: SoccerPitchConfiguration = pitch_config,
) -> tuple[np.ndarray, bool]:
    """
    Detect one pitch line point in the frame and return a tightly clustered set of pitch layout points around it.
    
    Returns:
        np.ndarray of shape (32, 2): tightly clustered pitch layout points around detected anchor
        bool: whether a valid point was found
    """
    h, w = frame.shape[:2]
    pitch_vertices = np.array(pitch_config.vertices, dtype=np.float32)

    # Scale pitch layout to image coordinates
    scale_x = w / pitch_config.length
    scale_y = h / pitch_config.width
    scaled_vertices = pitch_vertices * np.array([scale_x, scale_y])

    # Step 1: Try to find one valid point on a line
    def get_anchor_point_fast(frame, border_ignore=3):
        h, w = frame.shape[:2]
        
        # Convert to grayscale and HSV
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Basic grass mask
        grass = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
        grass = cv2.morphologyEx(grass, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))  # Smaller kernel for speed

        # Top-hat and white line thresholding
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 31))
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        _, white_lines = cv2.threshold(tophat, 30, 255, cv2.THRESH_BINARY)

        # Mask with grass
        masked = cv2.bitwise_and(white_lines, grass)

        # Edge detection
        blur  = cv2.GaussianBlur(masked, (5, 5), 0)
        edges = cv2.Canny(blur, 30, 100)

        # Hough Line Transform
        lines = cv2.HoughLinesP(edges,
                                rho=1,
                                theta=np.pi/360,
                                threshold=30,
                                minLineLength=10,
                                maxLineGap=15)

        if lines is not None:
            for x1, y1, x2, y2 in lines[:, 0]:
                if line_on_mask(x1, y1, x2, y2, grass, samples=20):  # Reduce samples for speed
                    mx, my = (x1 + x2) // 2, (y1 + y2) // 2

                    # Skip border area and check if point is on grass
                    if (border_ignore <= mx < w - border_ignore and
                        border_ignore <= my < h - border_ignore and
                        grass[my, mx] > 0):
                        return np.array([mx, my], dtype=np.float32)

        return None

    anchor = get_anchor_point_fast(frame)
    if anchor is None:
        return None, False

    # Step 2: Return a mini-pitch layout centered at the detected anchor point
    layout_center = scaled_vertices.mean(axis=0)
    relative_positions = scaled_vertices - layout_center
    shrink_scale = 0.01  # tightly clustered

    tiny_pitch = anchor + shrink_scale * relative_positions
    return tiny_pitch.astype(np.float32)[:10], True
    



def filter_by_density(kept, radius=80, max_neighbors=80):
    mids = [((x1+x2)/2, (y1+y2)/2) for x1,y1,x2,y2 in kept]
    filtered = []
    for i, (x1,y1,x2,y2) in enumerate(kept):
        mx, my = mids[i]
        cnt = sum(1 for ux,uy in mids if abs(ux-mx)<radius and abs(uy-my)<radius)
        if cnt <= max_neighbors:
            filtered.append((x1,y1,x2,y2))
    return filtered

def line_on_mask(x1, y1, x2, y2, mask, samples=20):
    for t in np.linspace(0, 1, samples):
        x = int(x1 + (x2 - x1) * t)
        y = int(y1 + (y2 - y1) * t)
        if mask[y, x] == 0:
            return False
    return True

def recover_centered_pitch_points(
    frame: np.ndarray,
    pitch_config: SoccerPitchConfiguration = pitch_config,
) -> tuple[np.ndarray, bool]:
    """
    Detect one pitch line point in the frame and return a tightly clustered set of pitch layout points around it.
    
    Returns:
        np.ndarray of shape (32, 2): tightly clustered pitch layout points around detected anchor
        bool: whether a valid point was found
    """
    h, w = frame.shape[:2]
    pitch_vertices = np.array(pitch_config.vertices, dtype=np.float32)

    # Scale pitch layout to image coordinates
    scale_x = w / pitch_config.length
    scale_y = h / pitch_config.width
    scaled_vertices = pitch_vertices * np.array([scale_x, scale_y])

    # Step 1: Try to find one valid point on a line
    def get_anchor_point_fast(frame, border_ignore=3):
        h, w = frame.shape[:2]
        
        # Convert to grayscale and HSV
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1) basic grass mask
        hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        grass = cv2.inRange(hsv, (35,40,40), (85,255,255))
        grass = cv2.morphologyEx(grass, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))

        # 2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (31,31))
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        _, white_lines = cv2.threshold(tophat, 30, 255, cv2.THRESH_BINARY)

        # 3) restriction to grass
        masked = cv2.bitwise_and(white_lines, grass)

        # 4) Canny + HoughP
        blur  = cv2.GaussianBlur(masked, (5,5), 0)
        edges = cv2.Canny(blur, 30, 100)
        lines = cv2.HoughLinesP(edges,
                                rho=1,
                                theta=np.pi/360,
                                threshold=30,
                                minLineLength=10,
                                maxLineGap=15)

        # 5) only grass segments
        line_mask  = np.zeros_like(gray)
        kept_lines = []
        if lines is not None:
            for x1,y1,x2,y2 in lines[:,0]:
                if line_on_mask(x1,y1,x2,y2, grass, samples=30):
                    cv2.line(line_mask, (x1,y1), (x2,y2), 255, 2)
                    kept_lines.append((x1,y1,x2,y2))

        # 6) get rid of borders
        line_mask[:border_ignore, :] = 0
        line_mask[-border_ignore:, :] = 0
        line_mask[:, :border_ignore] = 0
        line_mask[:, -border_ignore:] = 0

        # 7) final list reconstruction
        final_kept = []
        for x1,y1,x2,y2 in kept_lines:
            mx, my = (x1+x2)//2, (y1+y2)//2
            if 0 <= mx < w and 0 <= my < h and line_mask[my,mx] > 0:
                final_kept.append((x1,y1,x2,y2))

        final_kept = filter_by_density(final_kept,
                                    radius=80,
                                    max_neighbors=80)
        if len(final_kept) == 0:
            return None
        return np.array(final_kept[0][:2], dtype=np.float32)

    anchor = get_anchor_point_fast(frame)
    if anchor is None:
        return None, False

    # Step 2: Return a mini-pitch layout centered at the detected anchor point
    layout_center = scaled_vertices.mean(axis=0)
    relative_positions = scaled_vertices - layout_center
    shrink_scale = 0.01  # tightly clustered

    tiny_pitch = anchor + shrink_scale * relative_positions
    return tiny_pitch.astype(np.float32)[:10], True
    

def recover_detected_keypoints_only(
    original_frame : np.ndarray,
    keypoints: np.ndarray,
    image_shape: tuple[int, int],
    pitch_config: SoccerPitchConfiguration = pitch_config,
    fallback_value: tuple[float, float] = (-1.0, -1.0),
    conf_thresh: float = 0.5,

) -> tuple[np.ndarray, bool]:
    """
    Estimate homography from detected keypoints and reproject only those same pitch layout points.

    Returns:
        reprojected (N, 2): the projected keypoints (only detected ones)
        indices (N,): indices of the keypoints that were detected and projected
    """
    h, w = image_shape
    pitch_vertices = np.array(pitch_config.vertices, dtype=np.float32)

    scale_x = w / pitch_config.length
    scale_y = h / pitch_config.width
    scaled_vertices = pitch_vertices * np.array([scale_x, scale_y])

    def return_recover_response(original_frame, default_answer):
        # return default_answer, False
        res, is_regular = recover_centered_pitch_points(original_frame)
        # print(res)
        if res is None:
            return default_answer, False
        return res, is_regular

    if keypoints is None:
        print("aaaaa")
        return return_recover_response(original_frame, scaled_vertices)
        # return scaled_vertices, False

    conf_mask = keypoints[:, 2] > conf_thresh
    num_detected = np.sum(conf_mask)
    if num_detected == 0:
        print("bbbbb")
        return return_recover_response(original_frame, scaled_vertices)

    def get_tiny_pitch():
        detected_pts = keypoints[conf_mask, :2].astype(np.float32)
        anchor = detected_pts[0]  # use first detected keypoint as reference
        layout_center = scaled_vertices.mean(axis=0)

        # Create a tiny pitch layout around the anchor
        shrink_scale = 0.01
        relative_positions = scaled_vertices - layout_center
        tiny_pitch = anchor + shrink_scale * relative_positions
        return tiny_pitch[:10]
        # print(num_detected, tiny_pitch)
    if num_detected < 4:
        print("cccc")
        res, is_regular = recover_centered_pitch_points(original_frame)
        if res is None:
            return get_tiny_pitch(), False
        return res, is_regular

    detected = keypoints[conf_mask, :2].astype(np.float32)
    layout = scaled_vertices[conf_mask]

    H, _ = cv2.findHomography(layout, detected, cv2.RANSAC, 12)
    if H is None:
        res, is_regular = recover_centered_pitch_points(original_frame)
        print("ddddd")
        if res is None:
            return get_tiny_pitch(), False
        return res, is_regular
        # return get_tiny_pitch(), False
    

    layout_points = layout.reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(layout_points, H).reshape(-1, 2)

    full_result = np.full((32, 2), fallback_value, dtype=np.float32)
    full_result[conf_mask] = projected

    print("eeee")

    return full_result, True  # reprojected points + their original indices



# def filter_by_density(kept, radius=80, max_neighbors=80):
#     mids = [((x1+x2)/2, (y1+y2)/2) for x1,y1,x2,y2 in kept]
#     filtered = []
#     for i, (x1,y1,x2,y2) in enumerate(kept):
#         mx, my = mids[i]
#         cnt = sum(1 for ux,uy in mids if abs(ux-mx)<radius and abs(uy-my)<radius)
#         if cnt <= max_neighbors and len(filtered) < 30:
#             filtered.append([mx, my])
#     return filtered

# def line_on_mask(x1, y1, x2, y2, mask, samples=20):
#     for t in np.linspace(0, 1, samples):
#         x = int(x1 + (x2 - x1) * t)
#         y = int(y1 + (y2 - y1) * t)
#         if mask[y, x] == 0:
#             return False
#     return True

# def detect_pitch_lines_tophat(frame, border_ignore=3, idx = -1):
#     h, w = frame.shape[:2]
#     print(h, w)
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # 1) basic grass mask
#     hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#     grass = cv2.inRange(hsv, (35,40,40), (85,255,255))
#     grass = cv2.morphologyEx(grass, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))

#     # 2)
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (31,31))
#     tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
#     _, white_lines = cv2.threshold(tophat, 30, 255, cv2.THRESH_BINARY)

#     # 3) restriction to grass
#     masked = cv2.bitwise_and(white_lines, grass)

#     # 4) Canny + HoughP
#     blur  = cv2.GaussianBlur(masked, (5,5), 0)
#     edges = cv2.Canny(blur, 30, 100)
#     lines = cv2.HoughLinesP(edges,
#                             rho=1,
#                             theta=np.pi/360,
#                             threshold=30,
#                             minLineLength=10,
#                             maxLineGap=15)

#     # 5) only grass segments
#     line_mask  = np.zeros_like(gray)
#     kept_lines = []
#     if lines is not None:
#         for x1,y1,x2,y2 in lines[:,0]:
#             if line_on_mask(x1,y1,x2,y2, grass, samples=30):
#                 cv2.line(line_mask, (x1,y1), (x2,y2), 255, 2)
#                 kept_lines.append((x1,y1,x2,y2))

#     # 6) get rid of borders
#     line_mask[:border_ignore, :] = 0
#     line_mask[-border_ignore:, :] = 0
#     line_mask[:, :border_ignore] = 0
#     line_mask[:, -border_ignore:] = 0
    
#     # 7) final list reconstruction
#     final_kept = []
#     for x1,y1,x2,y2 in kept_lines:
#         mx, my = (x1+x2)//2, (y1+y2)//2
#         if 0 <= mx < w and 0 <= my < h and line_mask[my,mx] > 0:
#             final_kept.append((x1,y1,x2,y2))

#     final_kept = filter_by_density(final_kept,
#                                    radius=80,
#                                    max_neighbors=80)

#     return final_kept

