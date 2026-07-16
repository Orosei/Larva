import cv2
import numpy as np


def cal_skeleton_len_v2(src_idx, src_map, tgt_idx=None, tgt_map=None):
    src_mask = (src_map == src_idx)

    if not np.any(src_mask):
        return 0.0, 0.0

    ske_len = cal_skeleton_len_fast(src_mask)
    match_len = 0.0
    if tgt_idx is not None and tgt_map is not None:
        tgt_mask = (tgt_map == tgt_idx).astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        tgt_mask1 = cv2.dilate(tgt_mask, kernel)

        match_mask = (src_mask > 0) & (tgt_mask1 > 0)
        match_len = cal_skeleton_len_fast(src_mask, match_mask=match_mask)

    return ske_len, match_len


def cal_skeleton_len_fast(skeleton_mask, match_mask=None):
    M = skeleton_mask > 0

    H = M[:, :-1] & M[:, 1:]
    V = M[:-1, :] & M[1:, :]

    top_left = M[:-1, :-1]
    top_right = M[:-1, 1:]
    bottom_left = M[1:, :-1]
    bottom_right = M[1:, 1:]

    D1 = top_left & bottom_right & ~top_right & ~bottom_left
    D2 = top_right & bottom_left & ~top_left & ~bottom_right

    if match_mask is not None:
        Match = match_mask > 0
        H &= Match[:, :-1] & Match[:, 1:]
        V &= Match[:-1, :] & Match[1:, :]
        D1 &= Match[:-1, :-1] & Match[1:, 1:]
        D2 &= Match[:-1, 1:] & Match[1:, :-1]

    ortho_len = np.sum(H) + np.sum(V)
    diag_len = (np.sum(D1) + np.sum(D2)) * np.sqrt(2)

    return float(ortho_len + diag_len)


# ==================== skeleton length ROI v3 ====================

_SKELETON_LEN_KERNEL = np.ones((3, 3), np.uint8)


def cal_skeleton_len_v3(src_idx, src_map, tgt_idx=None, tgt_map=None):
    src_box = _label_bbox(src_map, src_idx)
    if src_box is None:
        return 0.0, 0.0

    sy1, sx1, sy2, sx2 = src_box
    src_roi = src_map[sy1:sy2, sx1:sx2] == src_idx
    ske_len = cal_skeleton_len_fast(src_roi)

    match_len = 0.0
    if tgt_idx is not None and tgt_map is not None:
        tgt_box = _label_bbox(tgt_map, tgt_idx)
        if tgt_box is None:
            return ske_len, match_len

        tgt_box = _expand_box(tgt_box, 1, tgt_map.shape)
        work_box = _intersect_boxes(src_box, tgt_box)
        if work_box is None:
            return ske_len, match_len

        wy1, wx1, wy2, wx2 = work_box
        dilate_box = _expand_box(work_box, 1, tgt_map.shape)
        dy1, dx1, dy2, dx2 = dilate_box

        src_match_roi = src_map[wy1:wy2, wx1:wx2] == src_idx
        tgt_dilate_roi = (tgt_map[dy1:dy2, dx1:dx2] == tgt_idx).astype(np.uint8)
        tgt_dilate_roi = cv2.dilate(tgt_dilate_roi, _SKELETON_LEN_KERNEL)

        off_y = wy1 - dy1
        off_x = wx1 - dx1
        h = wy2 - wy1
        w = wx2 - wx1
        tgt_match_roi = tgt_dilate_roi[off_y:off_y + h, off_x:off_x + w]

        match_mask = src_match_roi & (tgt_match_roi > 0)
        match_len = cal_skeleton_len_fast(src_match_roi, match_mask=match_mask)

    return ske_len, match_len


def _label_bbox(label_map, label):
    rows, cols = np.where(label_map == label)
    if rows.size == 0:
        return None
    return int(rows.min()), int(cols.min()), int(rows.max()) + 1, int(cols.max()) + 1


def _expand_box(box, pad, shape):
    y1, x1, y2, x2 = box
    height, width = shape
    return (
        max(0, y1 - pad),
        max(0, x1 - pad),
        min(height, y2 + pad),
        min(width, x2 + pad),
    )


def _intersect_boxes(box1, box2):
    y1 = max(box1[0], box2[0])
    x1 = max(box1[1], box2[1])
    y2 = min(box1[2], box2[2])
    x2 = min(box1[3], box2[3])
    if y1 >= y2 or x1 >= x2:
        return None
    return y1, x1, y2, x2
