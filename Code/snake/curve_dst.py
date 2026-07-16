import numpy as np


def get_circle(idx_map, row, col, radius):
    h, w = idx_map.shape
    top, bot = max(0, row - radius), min(h, row + radius + 1)
    left, right = max(0, col - radius), min(w, col + radius + 1)
    i, j = np.meshgrid(np.arange(top, bot), np.arange(left, right), indexing="ij")
    dist = np.hypot(i - row, j - col)
    mask = (dist > 0) & (dist <= radius)
    coords = np.stack([i[mask], j[mask]], axis=1)
    return coords


def get_valid_id(curve_map, row, col, radius):
    coords = get_circle(curve_map, row, col, radius)
    idx = np.unique(curve_map[coords[:, 0], coords[:, 1]])
    local_idx = curve_map[row, col]
    valid_idx = idx[(idx != 0) & (idx != -1) & (idx != local_idx)]
    return valid_idx


def get_weight(h, w, center_h, center_w):
    i, j = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    dist = np.hypot(i - center_h, j - center_w)
    dist[center_h, center_w] = np.inf
    weight = 1.0 / dist
    return weight


def search_curve_dst(curve_map, pt1, pt2, radius):
    h, w = curve_map.shape
    r1, c1 = pt1
    r2, c2 = pt2

    weight_map = get_weight(h, w, r1, c1)
    valid_idx = get_valid_id(curve_map, r1, c1, radius)
    if len(valid_idx) == 0:
        return None

    top, bot = max(0, r1 - radius), min(h, r1 + radius + 1)
    left, right = max(0, c1 - radius), min(w, c1 + radius + 1)
    sub_curve = curve_map[top:bot, left:right]
    sub_weight = weight_map[top:bot, left:right]
    weight_sum = np.zeros(len(valid_idx), dtype=np.float32)
    for i, idx in enumerate(valid_idx):
        weight_sum[i] = sub_weight[sub_curve == idx].sum()

    sorted_indices = np.argsort(weight_sum)[::-1]
    for sorted_idx in sorted_indices:
        curve_idx = valid_idx[sorted_idx]
        nr_sub, nc_sub = np.where(sub_curve == curve_idx)
        nr, nc = nr_sub + top, nc_sub + left
        dist1, dist2 = np.hypot(nr - r1, nc - c1), np.hypot(nr - r2, nc - c2)
        min_idx1, min_idx2 = np.argmin(dist1), np.argmin(dist2)
        if dist1[min_idx1] <= dist2[min_idx2]:
            return (nr[min_idx1], nc[min_idx1])
    return None
