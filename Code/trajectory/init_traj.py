import numpy as np

from ..utils import get_end_pts
from ..my_addition import add_args
from ..snake import get_valid_id
from ..snake.run_snakes import snake_nodes


def has_other_curve_nearby(curve_mask, point, radius):
    """Return whether another curve ID exists around ``point`` within ``radius``."""
    row, col = int(point[0]), int(point[1])
    valid_ids = get_valid_id(curve_mask, row, col, radius)
    return bool(len(valid_ids) > 0)


def is_adjacent_to_junction(junc_mask, point):
    """Return whether any valid 8-neighbor of ``point`` belongs to a junction."""
    row, col = int(point[0]), int(point[1])
    row_offsets = (0, -1, -1, -1, 0, 1, 1, 1)
    col_offsets = (1, 1, 0, -1, -1, -1, 0, 1)
    height, width = junc_mask.shape[:2]
    return any(
        junc_mask[row + dr, col + dc] != 0
        for dr, dc in zip(row_offsets, col_offsets)
        if 0 <= row + dr < height and 0 <= col + dc < width
    )


def init_traj(ske_mask, junc_mask, curve_mask, fov=[5, add_args.search_radius]):
    max_ske_idx = np.max(ske_mask)
    info = []

    for idx in range(1, max_ske_idx + 1):
        mask = (ske_mask == idx).astype(np.uint8)
        if np.sum(mask) < 2:
            continue

        end_pts = get_end_pts(mask)
        if len(end_pts) != 2:
            continue

        dist = np.linalg.norm(end_pts[0] - end_pts[1])
        radius = int(np.clip(5 + dist // 4, fov[0], fov[1]))
        for i in range(2):
            is_junction_adjacent = is_adjacent_to_junction(junc_mask, end_pts[i])
            has_other_curve = has_other_curve_nearby(curve_mask, end_pts[i], radius)
            if is_junction_adjacent or not has_other_curve:
                continue

            sampled_points = snake_nodes(ske_mask, idx, end_pts[1 - i])
            info.append([idx, i, sampled_points])
            break

    return info