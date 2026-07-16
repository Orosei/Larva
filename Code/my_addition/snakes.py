import numpy as np
import copy
import random

from ..my_addition.Model.get_direction import get_pred_direction_v2
from ..snake.curve_dst import search_curve_dst
from ..utils.get_end_points import get_end_pts


def snakes_info_v2(curve_map, ske_map, junc_map, fov=(5, 10)):
    row_offset = [0, -1, -1, -1, 0, 1, 1, 1]
    col_offset = [1, 1, 0, -1, -1, -1, 0, 1]
    max_id = np.max(ske_map)
    info = []
    for id in range(1, max_id):
        mask = (ske_map == id).astype(np.uint8)
        if np.sum(mask) < 10:
            continue

        end_pts = get_end_pts(mask)
        if len(end_pts) != 2:
            continue

        dist = np.linalg.norm(end_pts[0] - end_pts[1])
        radius = int(np.clip(5 + dist // 4, fov[0], fov[1]))
        for i in range(2):
            r, c = end_pts[i]
            flag = all(
                junc_map[r + dr, c + dc] == 0
                for dr, dc in zip(row_offset, col_offset)
                if 0 <= r + dr < ske_map.shape[0] and 0 <= c + dc < ske_map.shape[1]
            )

            if flag:
                dst = search_curve_dst(curve_map, end_pts[i], end_pts[1 - i], radius)
                if dst is not None:
                    info.append([id, tuple(end_pts[i][:2]), dst])
                    break

    return info


def cal_head_move(predictor, nodes, num):
    prob_pred_16 = get_pred_direction_v2(predictor, [nodes[0, :]], use_far=False)
    head_angle = cal_head_angle_v2(nodes, num)
    min_idx, orig_idx = get_optimal_angle_idx_v2(prob_pred_16, head_angle, 2)
    return int(min_idx), False, prob_pred_16, orig_idx


def cal_head_angle_v2(nodes, num):
    pt0 = nodes[0, :]
    pts = nodes[1:num, :]
    rads = np.arctan2(-(pt0[0] - pts[:, 0]), pt0[1] - pts[:, 1])
    rad_avg = mean_angle_rad(rads)
    return np.degrees(rad_avg) % 360


def mean_angle_rad(angles_rad):
    avg_cos = np.mean(np.cos(angles_rad))
    avg_sin = np.mean(np.sin(angles_rad))
    mean_angle = np.arctan2(avg_sin, avg_cos)
    return mean_angle


def get_optimal_angle_idx_v2(angle_weights, ref_angle, top_k):
    sorted_idx = np.argsort(angle_weights)[::-1][:top_k]
    angles = sorted_idx * 22.5
    diff = np.minimum((angles - ref_angle) % 360, 360 - ((angles - ref_angle) % 360))
    optimal_idx = sorted_idx[np.argmin(diff)].copy()
    orig_idx = sorted_idx[np.argmin(diff)].copy()

    if optimal_idx % 2 == 1:
        left_even = optimal_idx - 1
        right_even = optimal_idx + 1
        if right_even >= 16:
            right_even = 0
        optimal_idx = np.random.choice([left_even, right_even], p=[0.5, 0.5])

    return int(optimal_idx / 2), orig_idx


def get_optimal_angle_idx_v3(angle_weights, ref_angle, top_k):
    sorted_idx = np.argsort(angle_weights)[::-1][:top_k]
    angles = sorted_idx * 22.5
    diff = np.minimum((angles - ref_angle) % 360, 360 - ((angles - ref_angle) % 360))

    best_match_loc = np.argmin(diff)
    orig_idx = sorted_idx[best_match_loc]

    if orig_idx % 2 == 1:
        left_even_16 = (orig_idx - 1) % 16
        right_even_16 = (orig_idx + 1) % 16
        candidate_list = [int(left_even_16 / 2), int(right_even_16 / 2)]
    else:
        center_8 = int(orig_idx / 2)
        left_8 = (center_8 - 1) % 8
        right_8 = (center_8 + 1) % 8
        candidate_list = [left_8, center_8, right_8]

    return np.array(candidate_list, dtype=np.int32), orig_idx


def get_optimal_angle_idx_v4(angle_weights, ref_angle, top_k):
    sorted_idx = np.argsort(angle_weights)[::-1][:top_k]
    angles = sorted_idx * 22.5
    diff = np.minimum((angles - ref_angle) % 360, 360 - ((angles - ref_angle) % 360))

    best_match_loc = np.argmin(diff)
    orig_idx = sorted_idx[best_match_loc]
    optimal_idx = orig_idx

    if optimal_idx % 2 == 1:
        left_even = (optimal_idx - 1) % 16
        right_even = (optimal_idx + 1) % 16

        if angle_weights[left_even] >= angle_weights[right_even]:
            optimal_idx = left_even
        else:
            optimal_idx = right_even

    return int(optimal_idx / 2), orig_idx


def get_optimal_angle_idx_v5(angle_weights, ref_angle, top_k, head, dst, dist_limit):
    force_idx = -1

    dist = np.linalg.norm(head - dst)
    if dist <= dist_limit:
        dy = head[0] - dst[0]
        dx = dst[1] - head[1]

        angle_deg = np.degrees(np.arctan2(dy, dx)) % 360
        target_idx = int(round(angle_deg / 22.5)) % 16

        if angle_weights[target_idx] > 0.5:
            force_idx = target_idx

    if force_idx != -1:
        orig_idx = copy.deepcopy(force_idx)
    else:
        sorted_idx = np.argsort(angle_weights)[::-1][:top_k]
        angles = sorted_idx * 22.5
        diff = np.minimum((angles - ref_angle) % 360, 360 - ((angles - ref_angle) % 360))

        best_match_loc = np.argmin(diff)
        orig_idx = sorted_idx[best_match_loc]

    optimal_idx = copy.deepcopy(orig_idx)
    if optimal_idx % 2 == 1:
        left_even = (optimal_idx - 1) % 16
        right_even = (optimal_idx + 1) % 16

        if angle_weights[left_even] >= angle_weights[right_even]:
            optimal_idx = left_even
        else:
            optimal_idx = right_even

    return int(optimal_idx / 2), orig_idx


def get_optimal_angle_idx_v6(angle_weights, ref_angle, top_k, head, dst, dist_limit):
    force_idx = -1

    dist = np.linalg.norm(head - dst)
    if dist <= dist_limit:
        dy = head[0] - dst[0]
        dx = dst[1] - head[1]

        angle_deg = np.degrees(np.arctan2(dy, dx)) % 360
        target_idx = int(round(angle_deg / 22.5)) % 16

        if angle_weights[target_idx] > 0.5:
            force_idx = target_idx

    if force_idx != -1:
        orig_idx = force_idx
    else:
        sorted_idx = np.argsort(angle_weights)[::-1][:top_k]
        angles = sorted_idx * 22.5
        diff = np.minimum((angles - ref_angle) % 360, 360 - ((angles - ref_angle) % 360))

        best_match_loc = np.argmin(diff)
        orig_idx = sorted_idx[best_match_loc]

    optimal_idx = orig_idx

    if optimal_idx % 2 == 1:
        left_even = (optimal_idx - 1) % 16
        right_even = (optimal_idx + 1) % 16

        res = random.choices(
            [left_even, right_even], weights=[angle_weights[left_even], angle_weights[right_even]], k=1)
        optimal_idx = res[0]

    return int(optimal_idx / 2), orig_idx


def get_optimal_angle_idx_v7(angle_weights, ref_angle, head, dst, dist_limit):
    """dst in dist limit or best match head"""
    force_idx = -1
    dist = np.linalg.norm(head - dst)

    """dst"""
    if dist <= dist_limit:
        dy = head[0] - dst[0]
        dx = dst[1] - head[1]

        angle_deg = np.degrees(np.arctan2(dy, dx)) % 360
        target_idx = int(round(angle_deg / 22.5)) % 16

        if angle_weights[target_idx] > 0:
            force_idx = target_idx

    """head"""
    if force_idx != -1:
        orig_idx = force_idx
    else:
        valid_indices = np.where(angle_weights > 0)[0]

        if len(valid_indices) == 0:
            valid_indices = np.arange(16)

        angles = valid_indices * 22.5

        raw_diff = np.abs(angles - ref_angle) % 360
        diff = np.minimum(raw_diff, 360 - raw_diff)

        best_match_pos = np.argmin(diff)
        orig_idx = valid_indices[best_match_pos]

    optimal_idx = orig_idx

    if optimal_idx % 2 == 1:
        left_even = (optimal_idx - 1) % 16
        right_even = (optimal_idx + 1) % 16

        optimal_idx = random.choice([left_even, right_even])

    return int(optimal_idx / 2), orig_idx
