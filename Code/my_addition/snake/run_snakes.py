import numpy as np
import cv2


from .curve_dst import search_curve_dst_v2, search_curve_dst_v3
from ..Model.crack.direction import Direction

from ...utils.get_end_points import get_end_pts
from ..setting import add_args


def snakes_info_v3(seg_map, curve_map, ske_map, junc_map, fov=[5, add_args.search_radius]):
    row_offset = [0, -1, -1, -1, 0, 1, 1, 1]
    col_offset = [1, 1, 0, -1, -1, -1, 0, 1]

    binary_map = seg_map.astype(np.uint8)
    comp_num, comp_map, stats, centroids = cv2.connectedComponentsWithStats(binary_map, connectivity=8)
    main_comp_idx = 0

    if comp_num > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]

        if len(areas) > 0:
            max_idx = np.argmax(areas)
            main_comp_idx = max_idx + 1

    max_ske_idx = np.max(ske_map)
    info = []

    for idx in range(1, max_ske_idx + 1):
        mask = (ske_map == idx).astype(np.uint8)
        if np.sum(mask) < 2:
            continue

        end_pts = get_end_pts(mask)
        if len(end_pts) != 2:
            continue

        dist = np.linalg.norm(end_pts[0] - end_pts[1])
        radius = int(np.clip(5 + dist // 4, fov[0], fov[1]))
        for i in range(2):
            r, c = end_pts[i]

            flag = all(junc_map[r + dr, c + dc] == 0
                       for dr, dc in zip(row_offset, col_offset)
                       if 0 <= r + dr < ske_map.shape[0] and 0 <= c + dc < ske_map.shape[1])

            if flag:
                dst = search_curve_dst_v2(comp_map, main_comp_idx, curve_map, end_pts[i], end_pts[1 - i], radius)
                if dst is not None:
                    info.append([idx, tuple(end_pts[i][:2]), dst])
                    break

    return info


def snakes_info_v4(img, seg_map, curve_map, ske_map, junc_map, fov=[5, 10]):
    """direction net"""
    predictor = Direction(img)

    row_offset = [0, -1, -1, -1, 0, 1, 1, 1]
    col_offset = [1, 1, 0, -1, -1, -1, 0, 1]

    binary_map = seg_map.astype(np.uint8)
    comp_num, comp_map, stats, centroids = cv2.connectedComponentsWithStats(binary_map, connectivity=8)
    main_comp_idx = 0

    if comp_num > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]

        if len(areas) > 0:
            max_idx = np.argmax(areas)
            main_comp_idx = max_idx + 1

    max_ske_idx = np.max(ske_map)
    info = []

    for idx in range(1, max_ske_idx + 1):
        mask = (ske_map == idx).astype(np.uint8)
        if np.sum(mask) < 4:
            continue

        end_pts = get_end_pts(mask)
        if len(end_pts) != 2:
            continue

        dist = np.linalg.norm(end_pts[0] - end_pts[1])
        radius = int(np.clip(5 + dist // 4, fov[0], fov[1]))
        for i in range(2):
            r, c = end_pts[i]

            flag = all(junc_map[r + dr, c + dc] == 0
                       for dr, dc in zip(row_offset, col_offset)
                       if 0 <= r + dr < ske_map.shape[0] and 0 <= c + dc < ske_map.shape[1])

            if flag:
                dst = search_curve_dst_v3(
                    predictor, comp_map, main_comp_idx, curve_map, end_pts[i], end_pts[1 - i], radius)

                if dst is not None:
                    info.append([idx, tuple(end_pts[i][:2]), dst])
                    break

    return info
