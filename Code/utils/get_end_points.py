import cv2
import numpy as np


def farthest_points(pts):
    if len(pts) == 0:
        return np.array([])
    if len(pts) <= 2:
        return pts

    max_dist = 0
    best_pair = None
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            dist = np.linalg.norm(pts[i] - pts[j])
            if dist > max_dist:
                max_dist = dist
                best_pair = (i, j)
    return np.array([pts[best_pair[0]], pts[best_pair[1]]])


def get_end_pts(bw_map):
    if np.count_nonzero(bw_map) < 2:
        return np.array([])

    h, w = bw_map.shape
    bw_mask = (bw_map > 0).astype(np.float32)
    match_mask = np.zeros((h, w), dtype=bool)
    for i, k in enumerate(get_kernels()):
        conv = cv2.filter2D(bw_mask, -1, k)
        if i < 8:
            match_mask |= np.isclose(conv, 2)
        else:
            match_mask |= np.isclose(conv, 3)
    rows, cols = np.where(match_mask)
    pts = np.column_stack([rows, cols])
    return farthest_points(pts)


def get_kernels():
    # 3点方向，逆时针转
    rows = np.array([1, 0, 0, 0, 1, 2, 2, 2])
    cols = np.array([2, 2, 1, 0, 0, 0, 1, 2])
    kernels = np.ones((16, 3, 3), dtype=np.float32)

    x = np.array([1, -1, -1, -1, -1, -1, -1, -1])
    for i in range(8):
        kernels[i, rows, cols] = x
        x = np.roll(x, -1)

    x = np.array([1, 1, -1, -1, -1, -1, -1, -1])
    for i in range(8, 16):
        kernels[i, rows, cols] = x
        x = np.roll(x, -1)
    return kernels
