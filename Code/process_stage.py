import numpy as np

from Code.metric.basic_metric import bw_dist
from Code.snake.run_snakes import batch_mode
from Code.utils import process_curve, process_curve_v2
from Code.utils.get_skeleton_instance import process_skeleton
from Code.utils.image_operation import cal_ext_eng1, cal_ext_eng_v4


def update_seg(seg_map, refine_map):
    seg_map[refine_map > 0] = 255
    return seg_map


def update_prob(prob_nmap, refine_prob_nmap):
    np.maximum(prob_nmap, refine_prob_nmap, out=prob_nmap)
    return prob_nmap


def process_img(img, label, seg_map, img_name, iter_num=1):
    seg = seg_map.copy()

    ext_eng = cal_ext_eng_v4(img)  # ori_ext_eng_1
    # ext_eng = 1.0 - bw_dist(seg_map, True)

    i = 0
    flag = True
    while flag and i < iter_num:
        flag, refine_seg, in_foreground = batch_process(img, label, seg, ext_eng, img_name)
        seg = update_seg(seg, refine_seg)
        i += 1

    return seg, in_foreground


def batch_process(img, label, seg_map, ext_eng, img_name):
    ske_num, ske_map, junc_mask = process_skeleton(seg_map)
    # curve_map = process_curve(seg_map, ske_num, ske_map)
    curve_map = process_curve_v2(seg_map, ske_num, ske_map)

    flag, refine_seg, in_foreground = batch_mode(img, label, seg_map, curve_map, ske_map, junc_mask, ext_eng, img_name)
    return flag, refine_seg, in_foreground
