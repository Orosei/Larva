import cv2
import numpy as np

from .basic_metric import get_thick_map
from .comp_metric import Euler_num, mean_without_top_n
from ..utils.get_skeleton_instance import process_skeleton


def prepare_metric_label_cache(label, kernel_size=5):
    g_mask = (label > 0).astype(np.uint8)

    g_map = get_thick_map(g_mask)
    g_thin = (g_map == 1).astype(np.uint8)
    g_thick = (g_map == 2).astype(np.uint8)

    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    g_thin_dilate = cv2.dilate(g_thin, kernel)
    g_thick_dilate = cv2.dilate(g_thick, kernel)

    g_thin_good = g_thin_dilate | (1 - g_thick_dilate)
    g_thick_good = g_thick_dilate | (1 - g_thin_dilate)

    label_skeleton_num, label_skeleton_map, _ = process_skeleton(g_mask)
    label_comp_num, label_comp_labels, label_comp_stats, _ = cv2.connectedComponentsWithStats(g_mask)
    label_comp_area_mean = mean_without_top_n(label_comp_stats[:, 4], 1)
    label_comp_euler = Euler_num(label_comp_labels)

    return {
        "g_mask": g_mask,
        "g_thin_good": g_thin_good > 0,
        "g_thick_good": g_thick_good > 0,
        "label_skeleton_num": label_skeleton_num,
        "label_skeleton_map": label_skeleton_map,
        "label_comp_num": label_comp_num,
        "label_comp_area_mean": label_comp_area_mean,
        "label_comp_euler": label_comp_euler,
    }
