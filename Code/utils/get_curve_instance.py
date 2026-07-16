import cv2
import numpy as np

from .image_draw import color_instance, show_image


def enlarge_curve_instance(label_map, idx, valid_mask, kernel):
    idx_mask = (label_map == idx).astype(np.uint8)
    dilate_mask = cv2.dilate(idx_mask, kernel)

    unfilled_mask = valid_mask & (label_map == 0)
    grow_mask = unfilled_mask & (dilate_mask > 0)
    label_map[grow_mask] = idx  # note: label_map is changed
    return np.any(grow_mask)


def process_curve(bw_mask, ske_num, ske_map):
    valid_mask = bw_mask > 0
    label_map = ske_map.astype(np.int32)
    kernel = np.ones((3, 3), dtype=np.uint8)
    while True:
        expanded = False
        for idx in range(1, ske_num + 1):
            expanded |= enlarge_curve_instance(label_map, idx, valid_mask, kernel)
        if not expanded:
            break
    # show_curve_instance(valid_mask, label_map)
    return label_map


def show_curve_instance(bw_img, inst_img):
    clr_bw = cv2.cvtColor((bw_img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    clr_img = color_instance(inst_img)
    combined = np.hstack((clr_bw, clr_img))
    show_image(combined)


# ---------------- process_curve v2: frontier-based expansion ----------------

_FRONTIER_OFFSETS = np.array(
    [
        [-1, -1],
        [-1, 0],
        [-1, 1],
        [0, -1],
        [0, 1],
        [1, -1],
        [1, 0],
        [1, 1],
    ],
    dtype=np.int32,
)


def get_curve_frontier_growth(label_map, idx, frontier, valid_mask):
    if frontier.size == 0:
        return np.empty((0, 2), dtype=np.int32)

    h, w = label_map.shape
    nb = frontier[:, np.newaxis, :] + _FRONTIER_OFFSETS[np.newaxis, :, :]
    nb = nb.reshape(-1, 2)

    in_bounds = (nb[:, 0] >= 0) & (nb[:, 0] < h) & (nb[:, 1] >= 0) & (nb[:, 1] < w)
    nb = nb[in_bounds]
    if nb.size == 0:
        return np.empty((0, 2), dtype=np.int32)

    nb = np.unique(nb, axis=0)
    rows, cols = nb[:, 0], nb[:, 1]
    grow_mask = valid_mask[rows, cols] & (label_map[rows, cols] == 0)
    return nb[grow_mask]


def process_curve_v2(bw_mask, ske_num, ske_map):
    valid_mask = bw_mask > 0
    label_map = ske_map.astype(np.int32, copy=True)

    frontiers = {}
    active = set()
    for idx in range(1, ske_num + 1):
        rows, cols = np.where(label_map == idx)
        frontier = np.column_stack([rows, cols]).astype(np.int32)
        frontiers[idx] = frontier
        if frontier.size > 0:
            active.add(idx)

    while active:
        expanded = False
        next_active = set()

        for idx in range(1, ske_num + 1):
            if idx not in active:
                continue

            grow_pts = get_curve_frontier_growth(label_map, idx, frontiers[idx], valid_mask)
            if grow_pts.size == 0:
                frontiers[idx] = np.empty((0, 2), dtype=np.int32)
                continue

            label_map[grow_pts[:, 0], grow_pts[:, 1]] = idx
            frontiers[idx] = grow_pts
            next_active.add(idx)
            expanded = True

        if not expanded:
            break

        active = next_active

    return label_map
