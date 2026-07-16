import cv2
import numpy as np

from ..utils.get_end_points import get_end_pts
from ..my_addition.metric import cal_skeleton_len_v2, cal_skeleton_len_v3


def get_next_point(cur_row, cur_col, mask):
    row_offset = [0, -1, 0, 1, -1, -1, 1, 1]
    col_offset = [1, 0, -1, 0, 1, -1, -1, 1]
    for r_off, c_off in zip(row_offset, col_offset):
        nr, nc = cur_row + r_off, cur_col + c_off
        if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1]:
            if mask[nr, nc] == 1:
                return np.array([nr, nc], dtype=np.int32)
    return None


def traverse_skeleton(start_pt, mask, match_mask=None):
    tmp_mask = mask.copy()
    length = 0.0
    cur_pt = start_pt
    total = np.count_nonzero(tmp_mask)
    for _ in range(total - 1):
        r, c = cur_pt
        tmp_mask[r, c] = 0
        next_pt = get_next_point(r, c, tmp_mask)
        if next_pt is None:
            break
        if match_mask is None or (match_mask[r, c] and match_mask[tuple(next_pt)]):
            length += np.linalg.norm(next_pt - cur_pt)
        cur_pt = next_pt
    return length


def cal_skeleton_len(src_idx, src_map, tgt_idx=None, tgt_map=None):
    src_mask = np.where(src_map == src_idx, 1, 0).astype(np.uint8)
    end_pts = get_end_pts(src_mask)
    if end_pts.size == 0:
        return 0.0, 0.0

    ske_len = traverse_skeleton(end_pts[0], src_mask)

    kernel = np.ones((3, 3), np.uint8)
    match_len = 0.0
    if tgt_idx is not None and tgt_map is not None:
        tgt_mask = np.where(tgt_map == tgt_idx, 1, 0).astype(np.uint8)
        tgt_mask1 = cv2.dilate(tgt_mask, kernel)
        match_mask = (src_mask > 0) & (tgt_mask1 > 0)
        match_len = traverse_skeleton(end_pts[0], src_mask, match_mask)
    return ske_len, match_len


def match_skeleton_len(matches, src_ske_map, tgt_ske_map):
    eps = 1e-8
    scores = []
    # fmt = "---> {:06.2f} {:06.2f} {:06.2f} {:06.2f} {:+06.2f} {:+06.2f}"
    for src_idx, tgt_idx in matches:
        # a1, b1 = cal_skeleton_len(src_idx, src_ske_map, tgt_idx, tgt_ske_map)
        # a2, b2 = cal_skeleton_len(tgt_idx, tgt_ske_map, src_idx, src_ske_map)

        # a1, b1 = cal_skeleton_len_v2(src_idx, src_ske_map, tgt_idx, tgt_ske_map)
        # a2, b2 = cal_skeleton_len_v2(tgt_idx, tgt_ske_map, src_idx, src_ske_map)
        a1, b1 = cal_skeleton_len_v3(src_idx, src_ske_map, tgt_idx, tgt_ske_map)
        a2, b2 = cal_skeleton_len_v3(tgt_idx, tgt_ske_map, src_idx, src_ske_map)

        scores.append((b1 + b2) / (a1 + a2 + eps))
        # print(fmt.format(a1, b1, a2, b2, a1 - a2, b1 - b2))

    if not scores:
        return 0.0

    return np.mean(scores)
