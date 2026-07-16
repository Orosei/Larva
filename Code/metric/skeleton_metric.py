import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from .skeleton_len import match_skeleton_len
from ..utils.get_skeleton_instance import process_skeleton


def cal_skeleton_metric(pred_mask, label_mask):
    p_num, p_map, _ = process_skeleton(pred_mask)
    l_num, l_map, _ = process_skeleton(label_mask)

    # matches, s1 = match_skeleton_iou(p_num, p_map, l_num, l_map)
    matches, s1 = match_skeleton_iou_v2(p_num, p_map, l_num, l_map)
    
    s2 = match_skeleton_len(matches, p_map, l_map)
    return s1, s2


# ==================== cal_skeleton_metric_v2 ====================

def cal_skeleton_metric_v2(pred_mask, label_cache):
    p_num, p_map, _ = process_skeleton(pred_mask)
    l_num = label_cache["label_skeleton_num"]
    l_map = label_cache["label_skeleton_map"]

    matches, s1 = match_skeleton_iou_v2(p_num, p_map, l_num, l_map)
    s2 = match_skeleton_len(matches, p_map, l_map)
    return s1, s2


def match_skeleton_iou(src_num, src_map, tgt_num, tgt_map):
    iou_matrix = cal_iou_matrix(src_num, src_map, tgt_num, tgt_map)
    matches, avg_score = find_matches(iou_matrix)
    return matches, avg_score


def cal_iou_matrix(src_num, src_map, tgt_num, tgt_map):
    iou_matrix = np.zeros((src_num, tgt_num))
    for src_idx in range(1, src_num):
        src_mask = (src_map == src_idx).astype(np.uint8)
        for tgt_idx in range(1, tgt_num):
            tgt_mask = (tgt_map == tgt_idx).astype(np.uint8)
            iou = cal_iou(src_mask, tgt_mask)
            iou_matrix[src_idx, tgt_idx] = iou
    return iou_matrix


def cal_iou(mask1, mask2, ksize=3, eps=1e-8):
    kernel = np.ones((ksize, ksize), np.uint8)
    dilated_mask1 = cv2.dilate(mask1, kernel)
    inter = np.logical_and(dilated_mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return inter / (union + eps)


def find_matches(iou_matrix):
    rows, cols = linear_sum_assignment(1 - iou_matrix)
    matches, scores = [], []
    for r, c in zip(rows, cols):
        s = iou_matrix[r, c]
        if s > 0:
            matches.append((r, c))
            scores.append(s)
            # print(f"Match: ({r}, {c})  Score: {s:.4f}")
    avg_score = np.mean(scores) if scores else 0.0
    return matches, avg_score


# ==================== cal_iou optimized v2 ====================

def match_skeleton_iou_v2(src_num, src_map, tgt_num, tgt_map):
    iou_matrix = cal_iou_matrix_v2(src_num, src_map, tgt_num, tgt_map)
    matches, avg_score = find_matches(iou_matrix)
    return matches, avg_score


def cal_iou_matrix_v2(src_num, src_map, tgt_num, tgt_map, ksize=3, eps=1e-8):
    iou_matrix = np.zeros((src_num, tgt_num), dtype=np.float64)
    if src_num <= 1 or tgt_num <= 1:
        return iou_matrix

    kernel = np.ones((ksize, ksize), np.uint8)
    pad = ksize // 2
    height, width = src_map.shape

    src_area = np.bincount(src_map.ravel(), minlength=src_num).astype(np.float64)
    tgt_area = np.bincount(tgt_map.ravel(), minlength=tgt_num).astype(np.float64)
    src_boxes = _label_bboxes(src_map, src_num)

    for src_idx in range(1, src_num):
        if src_area[src_idx] == 0:
            continue

        y1, x1, y2, x2 = src_boxes[src_idx]
        if y1 < 0:
            continue

        roi_y1 = max(0, y1 - pad)
        roi_x1 = max(0, x1 - pad)
        roi_y2 = min(height, y2 + pad)
        roi_x2 = min(width, x2 + pad)

        src_roi = (src_map[roi_y1:roi_y2, roi_x1:roi_x2] == src_idx).astype(np.uint8)
        tgt_roi = tgt_map[roi_y1:roi_y2, roi_x1:roi_x2]

        dilated_src_roi = cv2.dilate(src_roi, kernel)
        dilated_inter = np.bincount(tgt_roi[dilated_src_roi > 0].ravel(), minlength=tgt_num).astype(np.float64)
        original_overlap = np.bincount(tgt_roi[src_roi > 0].ravel(), minlength=tgt_num).astype(np.float64)

        union = src_area[src_idx] + tgt_area - original_overlap
        valid = union > eps
        scores = np.zeros(tgt_num, dtype=np.float64)
        scores[valid] = dilated_inter[valid] / (union[valid] + eps)
        scores[0] = 0.0
        iou_matrix[src_idx, 1:] = scores[1:]

    return iou_matrix


def _label_bboxes(label_map, num_labels):
    boxes = np.full((num_labels, 4), -1, dtype=np.int32)
    rows, cols = np.nonzero(label_map)
    if rows.size == 0:
        return boxes

    labels = label_map[rows, cols]
    y_min = np.full(num_labels, label_map.shape[0], dtype=np.int32)
    x_min = np.full(num_labels, label_map.shape[1], dtype=np.int32)
    y_max = np.full(num_labels, -1, dtype=np.int32)
    x_max = np.full(num_labels, -1, dtype=np.int32)

    np.minimum.at(y_min, labels, rows)
    np.minimum.at(x_min, labels, cols)
    np.maximum.at(y_max, labels, rows)
    np.maximum.at(x_max, labels, cols)

    valid = y_max >= 0
    boxes[valid, 0] = y_min[valid]
    boxes[valid, 1] = x_min[valid]
    boxes[valid, 2] = y_max[valid] + 1
    boxes[valid, 3] = x_max[valid] + 1
    return boxes
