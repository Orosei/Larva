import numpy as np
import torch

from ...snake import get_valid_id, get_weight, get_circle


"""search curve dst"""
def search_curve_dst_v2(comp_map, main_comp_idx, curve_map, pt1, pt2, radius):
    h, w = curve_map.shape
    r1, c1 = pt1
    r2, c2 = pt2

    is_pt1_main = (comp_map[r1, c1] == main_comp_idx)

    valid_idx = get_valid_id(curve_map, r1, c1, radius)
    if len(valid_idx) == 0:
        return None

    weight_map = get_weight(h, w, r1, c1)

    circle_coords = get_circle(curve_map, r1, c1, radius)
    c_rows, c_cols = circle_coords[:, 0], circle_coords[:, 1]

    ids_in_circle = curve_map[c_rows, c_cols]
    weights_in_circle = weight_map[c_rows, c_cols]

    weight_sums = []
    for idx in valid_idx:
        current_sum = weights_in_circle[ids_in_circle == idx].sum()
        weight_sums.append(current_sum)

    sorted_indices = np.argsort(weight_sums)[::-1]

    for sorted_idx in sorted_indices:
        target_id = valid_idx[sorted_idx]

        mask = (ids_in_circle == target_id)
        target_rows = c_rows[mask]
        target_cols = c_cols[mask]

        dist1 = np.hypot(target_rows - r1, target_cols - c1)
        dist2 = np.hypot(target_rows - r2, target_cols - c2)

        min_idx1 = np.argmin(dist1)
        min_idx2 = np.argmin(dist2)

        if dist1[min_idx1] <= dist2[min_idx2]:
            candidate_r, candidate_c = target_rows[min_idx1], target_cols[min_idx1]
            is_dst_main = (comp_map[candidate_r, candidate_c] == main_comp_idx)

            # if is_pt1_main and not is_dst_main:  # ori does not exist
            #     """pt1 in main but dst not"""
            #     continue

            return candidate_r, candidate_c

    return None


def search_curve_dst_v3(predictor, comp_map, main_comp_idx, curve_map, pt1, pt2, radius):
    h, w = curve_map.shape
    r1, c1 = pt1
    r2, c2 = pt2

    is_pt1_main = (comp_map[r1, c1] == main_comp_idx)

    valid_idx = get_valid_id(curve_map, r1, c1, radius)
    if len(valid_idx) == 0:
        return None

    circle_coords = get_circle(curve_map, r1, c1, radius)
    c_rows, c_cols = circle_coords[:, 0], circle_coords[:, 1]
    ids_in_circle = curve_map[c_rows, c_cols]

    """conf"""
    raw_pred_tensor, _ = predictor.dir_pre(np.array([[r1, c1]]))

    squeezed_tensor = raw_pred_tensor.squeeze(0).squeeze(0)
    squeezed_softmax_tensor = torch.softmax(squeezed_tensor, dim=-1)
    squeezed_pred = squeezed_softmax_tensor.cpu().detach().numpy()

    val_0 = squeezed_pred[..., 0]
    val_1_2 = squeezed_pred[..., 1:3]

    merged_1_2 = val_1_2.sum(axis=-1)
    conf_16 = merged_1_2[:16]

    """candidates"""
    candidates = []
    for target_id in valid_idx:
        mask = (ids_in_circle == target_id)
        target_rows = c_rows[mask]
        target_cols = c_cols[mask]

        if len(target_rows) == 0:
            continue

        dist1_all = np.hypot(target_rows - r1, target_cols - c1)
        min_idx = np.argmin(dist1_all)

        cand_r = target_rows[min_idx]
        cand_c = target_cols[min_idx]

        dx = cand_c - c1
        dy = r1 - cand_r

        angle = np.arctan2(dy, dx)
        dir_idx = int(np.round(np.degrees(angle) / 22.5)) % 16
        conf = conf_16[dir_idx]

        candidates.append({
            'r': cand_r,
            'c': cand_c,
            'conf': conf
        })

    num_candidates = len(candidates)
    if num_candidates == 0:
        return None

    """check"""
    candidates.sort(key=lambda x: x['conf'], reverse=True)
    for cand in candidates:
        cand_r = cand['r']
        cand_c = cand['c']
        conf = cand['conf']

        d1 = np.hypot(cand_r - r1, cand_c - c1)
        d2 = np.hypot(cand_r - r2, cand_c - c2)
        pass_dist_check = (d1 <= d2)

        is_dst_main = (comp_map[cand_r, cand_c] == main_comp_idx)
        pass_main_check = not (is_pt1_main and not is_dst_main)

        if num_candidates == 1:
            if conf > 0.5 and pass_dist_check and pass_main_check:
                return cand_r, cand_c
            else:
                break
        else:
            if pass_dist_check and pass_main_check:
                return cand_r, cand_c
            continue

    return None
