import cv2
import numpy as np


def gen_refine_seg(seg_map, refine_img):
    refine_seg = seg_map.copy().astype(np.uint8)

    mask = (refine_img > 0) & (seg_map == 0)
    refine_seg[mask] = 255

    return refine_seg


def color_diffs_v2(src_img, base_mask, diff_mask, color=(0, 0, 255)):
    is_grayscale = len(src_img.shape) == 2
    if is_grayscale:
        marked_img = cv2.cvtColor(src_img, cv2.COLOR_GRAY2BGR)
    else:
        marked_img = src_img.copy()

    mask = (diff_mask > 0) & (base_mask == 0)
    marked_img[mask] = color

    return marked_img
