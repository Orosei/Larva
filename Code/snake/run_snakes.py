import cv2
import numpy as np

from .curve_dst import search_curve_dst
from .snakes import Snakes
from ..my_addition import snakes_info_v2, gen_video, add_args
from ..my_addition.snake import snakes_info_v3, snakes_info_v4
from ..my_addition.snake.snakes_info import snakes_info_v5
from ..utils.get_end_points import get_end_pts


def snake_nodes(ske_map, idx, start_pt, step=5):
    mask = (ske_map == idx).astype(np.int32)
    ro, co = (
        np.array([0, -1, 0, 1, -1, -1, 1, 1]),
        np.array([1, 0, -1, 0, 1, -1, -1, 1]),
    )
    h, w = ske_map.shape
    cur_pt = tuple(start_pt)
    ordered, total = [], np.count_nonzero(mask)
    for _ in range(total):
        ordered.append(cur_pt)
        mask[cur_pt] = 0
        nr, nc = cur_pt[0] + ro, cur_pt[1] + co
        vm = (nr >= 0) & (nr < h) & (nc >= 0) & (nc < w)
        valid_nr, valid_nc = nr[vm], nc[vm]
        found = False
        for r, c in zip(valid_nr, valid_nc):
            if mask[r, c]:
                cur_pt = (r, c)
                found = True
                break
        if not found:
            break

    sampled = ordered[::step]
    if ordered and sampled and sampled[-1] != ordered[-1]:
        sampled.append(ordered[-1])
    return np.array(sampled, dtype=np.int32)


def snakes_info(curve_map, ske_map, junc_map, fov=(10, 30)):
    row_offset = [0, -1, -1, -1, 0, 1, 1, 1]
    col_offset = [1, 1, 0, -1, -1, -1, 0, 1]
    max_id = np.max(ske_map)
    info = []
    for id in range(1, max_id):
        mask = (ske_map == id).astype(np.uint8)
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

    # visualize_snake_info(curve_map, ske_map, info)
    return info


def visualize_snake_info(curve_map, seg_map, info, window_name="Snake Info Visualization"):
    h, w = (
        max(curve_map.shape[0], seg_map.shape[0]),
        max(curve_map.shape[1], seg_map.shape[1]),
    )
    curve_vis = np.zeros((h, w, 3), dtype=np.uint8)
    seg_vis = np.zeros((h, w, 3), dtype=np.uint8)
    curve_vis[: curve_map.shape[0], : curve_map.shape[1]] = cv2.cvtColor((curve_map > 0).astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
    seg_vis[: seg_map.shape[0], : seg_map.shape[1]] = cv2.cvtColor((seg_map > 0).astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
    for item in info:
        _, end_pt, dst = item
        end_r, end_c = int(end_pt[0]), int(end_pt[1])
        dst_r, dst_c = int(dst[0]), int(dst[1])
        cv2.circle(curve_vis, (end_c, end_r), 1, (0, 255, 0), -1)
        cv2.circle(curve_vis, (dst_c, dst_r), 1, (0, 0, 255), -1)
        cv2.line(curve_vis, (end_c, end_r), (dst_c, dst_r), (255, 0, 0), 1)
        cv2.circle(seg_vis, (end_c, end_r), 1, (0, 255, 0), -1)
        cv2.circle(seg_vis, (dst_c, dst_r), 1, (0, 0, 255), -1)
        cv2.line(seg_vis, (end_c, end_r), (dst_c, dst_r), (255, 0, 0), 1)
    combined_vis = np.hstack((curve_vis, seg_vis))
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO)
    cv2.imshow(window_name, combined_vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def snakes_loader(ske_map, info, batch_size=100, sort_pts=True, top_k=100):
    ids, pts, dsts = [], [], []
    for id, pt1, dst in info:
        pt = snake_nodes(ske_map, id, pt1, add_args.win)
        if pt.shape[0] > 1:
            ids.append(id)
            pts.append(pt)
            dsts.append(dst)
    if sort_pts:
        sorted_indices = np.argsort([-p.shape[0] for p in pts])
        ids = [ids[i] for i in sorted_indices]
        pts = [pts[i] for i in sorted_indices]
        dsts = [dsts[i] for i in sorted_indices]
    if top_k > 0:
        ids = ids[:top_k]
        pts = pts[:top_k]
        dsts = dsts[:top_k]
    ids_batch = [ids[i : i + batch_size] for i in range(0, len(ids), batch_size)]
    pts_batch = [pts[i : i + batch_size] for i in range(0, len(pts), batch_size)]
    dsts_batch = [dsts[i : i + batch_size] for i in range(0, len(dsts), batch_size)]
    return ids_batch, pts_batch, dsts_batch


def batch_mode(img, label, seg_map, curve_map, ske_map, junc_map, ext_eng_map, img_name):
    SNAKES = Snakes(img, label, seg_map, curve_map, ske_map, ext_eng_map, img_name)

    # info = snakes_info(curve_map, ske_map, junc_map)
    # info = snakes_info_v2(curve_map, ske_map, junc_map)
    info = snakes_info_v3(seg_map, curve_map, ske_map, junc_map)
    # info = snakes_info_v4(img, seg_map, curve_map, ske_map, junc_map)
    # info = snakes_info_v5(img, seg_map, curve_map, ske_map, junc_map)

    ids_batch, pts_batch, dsts_batch = snakes_loader(ske_map, info)

    refine_seg_map = np.zeros((seg_map.shape[0], seg_map.shape[1]), dtype=np.int32)
    refined = False

    in_foreground = np.zeros(2, dtype=np.int32)

    for ids, pts, dsts in zip(ids_batch, pts_batch, dsts_batch):
        SNAKES.Add(ids, pts, dsts)

        # seg_map_tmp = SNAKES.Process(50, 5)
        seg_map_tmp = SNAKES.process_v2(40, add_args.win)

        mask1 = (seg_map_tmp == 1) & (seg_map == 0)
        refine_seg_map[mask1] = 1
        refined |= np.any(mask1)

        fg_count, bg_count = SNAKES.get_foreground_stats()
        in_foreground[0] += fg_count
        in_foreground[1] += bg_count

        SNAKES.clear()

    # gen_video()

    return refined, refine_seg_map, in_foreground
