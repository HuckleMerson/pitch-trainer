import numpy as np


def get_segment_covering_all_points(points):
    # Convert list of dicts to Nx2 numpy array
    pts = np.array([[p["x"], p["y"]] for p in points])

    # Compute the centroid
    centroid = np.mean(pts, axis=0)

    # Subtract mean
    centered = pts - centroid

    # PCA: direction of maximum variance (1st principal component)
    u, s, vh = np.linalg.svd(centered)
    direction = vh[0]  # principal axis

    # Project all points onto the direction vector
    projections = centered @ direction

    # Get the extreme projections
    min_proj = projections.min()
    max_proj = projections.max()

    # Convert projections back to 2D points
    pt1 = centroid + min_proj * direction
    pt2 = centroid + max_proj * direction

    return {
        "start": {"x": float(pt1[0]), "y": float(pt1[1])},
        "end": {"x": float(pt2[0]), "y": float(pt2[1])},
    }


points = [
    {"x": 0.9983679641191496, "y": 0.5488886559335578},
    {"x": 0.31282172106410144, "y": 0.997954376151948},
    {"x": 0.9341098430538367, "y": 0.588327448820915},
    {"x": 0.5621526600936203, "y": 0.8259814109883914},
    {"x": 0.8445898687928305, "y": 0.6425702942709335},
    {"x": 0.7704053939624409, "y": 0.6889724383648538},
    {"x": 0.49092292995397224, "y": 0.874354715652142},
    {"x": 0.6653185634282368, "y": 0.7563749569460095},
    {"x": 0.4131370977558238, "y": 0.9276635625242222},
]

print(get_segment_covering_all_points(points))
