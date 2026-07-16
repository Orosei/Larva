import cv2
import numpy as np

from Code.snake.snakes import Snakes
from .run import run_snake
from .utils import ini_snakes, dynamic_radius, check_nb_comp, search_dst_v2
from Code.utils import (
    get_sep_ske_v1,
    reorder_comps,
    junc_agg,
    get_end_pts,
)
from Code.snake.setting import args


def visualization_mode(img, seg, ext_eng_map, dis_map):
    shape = seg.shape

    frame_img = np.zeros((shape[0], shape[1] * 2, 3), dtype=np.uint8)
    refine_map = np.zeros((shape[0], shape[1]), dtype=np.uint8)

    frame_img[:, : shape[1], 0] = seg
    frame_img[:, : shape[1], 1] = seg
    frame_img[:, : shape[1], 2] = seg
    frame_img[:, shape[1] :, :] = img

    cv2.namedWindow("Visualization", cv2.WINDOW_NORMAL)
    cv2.imshow("Visualization", frame_img)

    param_dict = {
        "img": img,
        "seg": seg,
        "ext_eng_map": ext_eng_map,
        "dis_map": dis_map,
        "refine_map": refine_map,
    }

    cv2.setMouseCallback("Visualization", mouse_callback, param_dict)

    while True:
        key = cv2.waitKey(100)
        if key == 27:  # 按下ESC键也可退出
            break

    cv2.destroyAllWindows()
    return


def mouse_callback(event, x, y, flags, param):
    img = param["img"]
    seg = param["seg"]
    ext_eng_map = param["ext_eng_map"]
    dis_map = param["dis_map"]
    refine_map = param["refine_map"]

    color = [0, 255, 0]

    shape = seg.shape

    frame_img = np.zeros((img.shape[0], img.shape[1] * 2, 3), dtype=np.uint8)

    if event == cv2.EVENT_LBUTTONDOWN:
        click_pt = np.array([y, x], dtype=np.int32)

        ske_num, ske_map, ske_info, junc_rows, junc_cols = get_sep_ske_v1(seg, include_border=False)

        ske_map, junc_map = junc_agg(ske_num, ske_map, ske_info, junc_rows, junc_cols)

        curve_map = get_sep_curve(seg, ske_num, ske_map)

        comp_num, comp_map, comp_info = reorder_comps(seg)

        ske_idx = click_ske_idx(ske_map, click_pt)

        if ske_idx >= 0:
            print(y, x, ske_idx)
            snakes = Snakes(img, seg, comp_map, curve_map, ske_map, ext_eng_map, dis_map, 1)

            info = try_ini_snake(comp_map, curve_map, ske_map, junc_map, ske_idx, 1)
            num = info.shape[0]

            if num > 0:
                ske_idxs, snakes_pt, dsts, weight_dsts = ini_snakes(ske_map, info)

                snakes.add(ske_idxs[0], snakes_pt[0], dsts[0], weight_dsts[0])

                temp_refine_map, _ = run_snake(snakes, visual=True)

                mask_temp = temp_refine_map == 1
                param["refine_map"][mask_temp] = 1

                mask_refine = refine_map == 1
                mask_refine_tiled = np.tile(mask_refine, (1, 2))

                frame_img[:, : shape[1], :] = seg[..., np.newaxis]
                frame_img[:, shape[1] :, :] = img
                frame_img[mask_refine_tiled, :] = color

                cv2.imshow("Visualization", frame_img)

                snakes.clear()
            return
        else:
            if ske_idx == -1:
                print("邻近无骨架点,请重新选取")
                return
            else:
                print("点击不在分割图内,请重新选取")
                return


def click_ske_idx(ske_map, pt):
    shape = ske_map.shape

    if pt[1] > shape[1]:
        return -2

    row_diff = []
    col_diff = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            row_diff.append(i)
            col_diff.append(j)

    idx = -1

    for j in range(len(row_diff)):
        nb_row = pt[0] + row_diff[j]
        nb_col = pt[1] + col_diff[j]
        if 0 <= nb_row < shape[0] and 0 <= nb_col < shape[1]:
            if ske_map[nb_row, nb_col] > 0:
                idx = ske_map[nb_row, nb_col]
                break

    return idx


def try_ini_snake(comp_map, curve_map, ske_map, junc_map, ske_idx, stage_idx=1):
    shape = ske_map.shape
    info = np.zeros((2, 7), dtype=np.int32)
    count = 0

    row_diff = [0, -1, -1, -1, 0, 1, 1, 1]
    col_diff = [1, 1, 0, -1, -1, -1, 0, 1]

    b_map = np.where(ske_map == ske_idx, 1, 0)
    pt_count = np.sum(b_map)

    if pt_count < 2:
        return info[:0, :]

    end_pts = get_end_pts(b_map)
    if end_pts.shape[0] != 2:
        return info[:0, :]

    dist = np.sqrt(np.sum((end_pts[0] - end_pts[1]) ** 2))
    radius = dynamic_radius(dist)

    for i in range(2):
        row = end_pts[i, 0]
        col = end_pts[i, 1]

        flag1 = True

        for j in range(8):
            nb_row = row + row_diff[j]
            nb_col = col + col_diff[j]

            if 0 <= nb_row < shape[0] and 0 <= nb_col < shape[1]:
                if junc_map[nb_row, nb_col] > 0:
                    flag1 = False
                    break

        if stage_idx == 2:
            flag2 = check_nb_comp(curve_map, row, col, radius)

        if stage_idx == 1:
            flag = flag1
        elif stage_idx == 2:
            flag = flag1 and flag2

        if flag:
            # dst = search_dst(curve_map, row, col, radius)
            dst, weight_dst = search_dst_v2(curve_map, end_pts[i], end_pts[(i + 1) % 2], radius)

            if dst.shape[0] == 2:
                info[count, 0] = ske_idx
                info[count, 3] = end_pts[(i + 1) % 2][0]
                info[count, 4] = end_pts[(i + 1) % 2][1]
                info[count, 5] = dst[0]
                info[count, 6] = dst[1]

                if pt_count > args.win:
                    info[count, 1] = 1
                    info[count, 2] = args.win
                elif pt_count > 1:
                    info[count, 1] = 2
                    info[count, 2] = pt_count - 1

                count += 1

    return info[:count, :]


def search_nb_comp_indices(num_map, row, col, radius):
    shape = num_map.shape

    top = np.max([0, row - radius])
    bot = np.min([shape[0], row + radius + 1])
    left = np.max([0, col - radius])
    right = np.min([shape[1], col + radius + 1])

    cut_map = num_map[top:bot, left:right]
    unique_indices = np.unique(cut_map)

    return unique_indices
