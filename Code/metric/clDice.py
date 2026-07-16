import numpy as np
import cv2


def _thin_mask(mask):
    bw_map = (mask > 0).astype(np.uint8) * 255
    return cv2.ximgproc.thinning(bw_map, cv2.ximgproc.THINNING_ZHANGSUEN) > 0


def cal_cldice_metric_v2(pred_mask, label_mask, eps=1e-8):
    p_mask = (pred_mask > 0).astype(np.uint8)
    g_mask = (label_mask > 0).astype(np.uint8)

    p_skeleton = _thin_mask(p_mask)
    g_skeleton = _thin_mask(g_mask)

    tprec = np.count_nonzero(p_skeleton & (g_mask > 0)) / (np.count_nonzero(p_skeleton) + eps)
    tsens = np.count_nonzero(g_skeleton & (p_mask > 0)) / (np.count_nonzero(g_skeleton) + eps)
    cldice = 2 * tprec * tsens / (tprec + tsens + eps)
    return (cldice,)
