import cv2
import numpy as np

from ..metric.comp_metric import mean_without_top_n, Euler_num


def cal_comp_metrics_v2(pred_mask, gt_mask, eps=1e-8):
    num1, labels1, stats1, _ = cv2.connectedComponentsWithStats(pred_mask)
    num2, labels2, stats2, _ = cv2.connectedComponentsWithStats(gt_mask)
    ncce = np.abs((num1 - num2) / (num1 + num2 + eps))
    a1 = mean_without_top_n(stats1[:, 4], 1)
    a2 = mean_without_top_n(stats2[:, 4], 1)
    naae = np.abs((a2 - a1) / (a2 + a1 + eps))  # lower->better
    b1 = Euler_num(labels1)
    b2 = Euler_num(labels2)
    brr = b1 - b2
    return ncce, naae, brr  # lower->0, lower->0, lower->0 are better


# ==================== cal_comp_metrics_v3 ====================

def cal_comp_metrics_v3(pred_mask, label_cache, eps=1e-8):
    num1, labels1, stats1, _ = cv2.connectedComponentsWithStats(pred_mask)
    num2 = label_cache["label_comp_num"]
    a2 = label_cache["label_comp_area_mean"]
    b2 = label_cache["label_comp_euler"]

    ncce = np.abs((num1 - num2) / (num1 + num2 + eps))
    a1 = mean_without_top_n(stats1[:, 4], 1)
    naae = np.abs((a2 - a1) / (a2 + a1 + eps))  # lower->better
    b1 = Euler_num(labels1)
    brr = b1 - b2
    return ncce, naae, brr  # lower->0, lower->0, lower->0 are better
