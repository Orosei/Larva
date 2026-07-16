import cv2
import numpy as np


def Euler_num(label_map):
    if np.max(label_map) == 0:
        # return {"b0": 0, "b1": 0}
        return 0
    b0 = np.max(label_map)
    contours, hierarchy = cv2.findContours((label_map > 0).astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    b1 = 0
    if hierarchy is not None:
        for h in hierarchy[0]:
            if h[3] != -1:
                b1 += 1
    return b0 - b1


def mean_without_top_n(arr, n=1, eps=1e-8):
    return np.sum(np.sort(arr)[:-n]) / (len(arr) - n + eps)


def cal_comp_metrics(pred_mask, gt_mask, eps=1e-8):
    num1, labels1, stats1, _ = cv2.connectedComponentsWithStats(pred_mask)
    num2, labels2, stats2, _ = cv2.connectedComponentsWithStats(gt_mask)
    frr = (num1 - num2) / (num2 + eps)
    a1 = mean_without_top_n(stats1[:, 4], 1)
    a2 = mean_without_top_n(stats2[:, 4], 1)
    agr = (a2 - a1) / (a2 + eps)  # lower->better
    b1 = Euler_num(labels1)
    b2 = Euler_num(labels2)
    brr = b1 - b2
    return frr, agr, brr  # lower->0, lower->0, lower->0 are better
