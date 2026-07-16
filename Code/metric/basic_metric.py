import cv2
import numpy as np

from ..utils import process_curve, process_curve_v2
from ..utils.get_skeleton_instance import process_skeleton


def cal_basic_metric(pred, label, mask=None, eps=1e-8):
    # 如果有 mask，则只在 mask 区域计算
    if mask is not None:
        pred = pred[mask] > 0
        label = label[mask] > 0
    tp = np.sum(label * pred)
    fp = np.sum((1 - label) * pred)
    fn = np.sum(label * (1 - pred))
    pre = tp / (tp + fp + eps)
    rec = tp / (tp + fn + eps)
    f1 = 2 * pre * rec / (pre + rec + eps)
    iou = tp / (tp + fp + fn + eps)
    return f1, pre, rec, iou


# ==================== cal_basic_metric_v2 ====================

def cal_basic_metric_v2(pred, label, mask=None, eps=1e-8):
    if mask is not None:
        pred = pred[mask] > 0
        label = label[mask] > 0
    else:
        pred = pred > 0
        label = label > 0

    tp = np.count_nonzero(label & pred)
    pred_sum = np.count_nonzero(pred)
    label_sum = np.count_nonzero(label)
    fp = pred_sum - tp
    fn = label_sum - tp

    pre = tp / (tp + fp + eps)
    rec = tp / (tp + fn + eps)
    f1 = 2 * pre * rec / (pre + rec + eps)
    iou = tp / (tp + fp + fn + eps)
    return f1, pre, rec, iou


def bw_dist(bw_mask, norm=False):
    kernel = np.ones((3, 3), np.uint8)
    tmp_mask = bw_mask.copy()
    dist_map = np.zeros_like(bw_mask, dtype=np.int32)
    dist_val = 1
    while np.any(tmp_mask):
        erode_mask = cv2.erode(tmp_mask, kernel)
        edge_mask = (tmp_mask > 0) & (erode_mask == 0)
        dist_map[edge_mask] = dist_val
        dist_val += 1
        tmp_mask = erode_mask

    if norm:
        mag_max = np.max(dist_map)
        if mag_max > 0:
            dist_map = dist_map / mag_max

    return dist_map


def curve_dist(curve_map, dist_map):
    instance_labels = np.unique(curve_map)
    instance_labels = instance_labels[instance_labels != 0]  # 去掉背景

    dist_curve_map = np.zeros_like(curve_map, dtype=np.float32)
    dist_list = []
    for label in instance_labels:
        mask = curve_map == label
        dist = np.mean(dist_map[mask])
        dist_curve_map[mask] = dist
        dist_list.append(dist)

    # show_dist_map_split(curve_map, dist_curve_map)
    return dist_curve_map, dist_list


def get_thick_map(bw_mask):
    dist_map = bw_dist(bw_mask)
    ske_num, ske_map, _ = process_skeleton(bw_mask)
    # curve_map = process_curve(bw_mask, ske_num, ske_map)
    curve_map = process_curve_v2(bw_mask, ske_num, ske_map)
    dist_fmap, dists = curve_dist(curve_map, dist_map)
    thick_map = thresh_map(dist_fmap, float(np.mean(dists)))
    return thick_map


def thresh_map(bw_map, thresh=1.0):
    thick_gray = np.zeros_like(bw_map, dtype=np.uint8)
    mask_thin = (bw_map > 0) & (bw_map <= thresh)
    mask_thick = bw_map > thresh
    thick_gray[mask_thin] = 1
    thick_gray[mask_thick] = 2
    return thick_gray


def cal_thick_metric(pred, gt, kernel_size=5):
    p_mask = (pred > 0).astype(np.uint8)
    g_mask = (gt > 0).astype(np.uint8)

    g_map = get_thick_map(g_mask)
    g_thin = (g_map == 1).astype(np.uint8)
    g_thick = (g_map == 2).astype(np.uint8)

    # 膨胀 thin/thick 区域
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    g_thin_dilate = cv2.dilate(g_thin, kernel)
    g_thick_dilate = cv2.dilate(g_thick, kernel)
    g_thin_good = g_thin_dilate | (1 - g_thick_dilate)
    g_thick_good = g_thick_dilate | (1 - g_thin_dilate)
    thin = cal_basic_metric(p_mask, g_mask, g_thin_good > 0)
    thick = cal_basic_metric(p_mask, g_mask, g_thick_good > 0)

    # visualize_thick_thin(g_thin_good, g_thick_good, p_mask)
    return thin, thick


# ==================== cal_thick_metric_v2 ====================

def cal_thick_metric_v2(pred, label_cache):
    p_mask = (pred > 0).astype(np.uint8)

    g_mask = label_cache["g_mask"]
    g_thin_good = label_cache["g_thin_good"]
    g_thick_good = label_cache["g_thick_good"]

    thin = cal_basic_metric(p_mask, g_mask, g_thin_good)
    thick = cal_basic_metric(p_mask, g_mask, g_thick_good)
    return thin, thick


def show_dist_map_split(bw_img, dist_map):
    # 距离图归一化并上色
    dist_map_norm = cv2.normalize(dist_map, None, 0, 255, cv2.NORM_MINMAX)
    dist_map_norm = np.uint8(dist_map_norm)
    dist_map_color = cv2.applyColorMap(dist_map_norm, cv2.COLORMAP_JET)

    # 原始图归一化并转成三通道
    bw_img_norm = cv2.normalize(bw_img, None, 0, 255, cv2.NORM_MINMAX)
    bw_img_norm = np.uint8(bw_img_norm)
    bw_img_color = cv2.cvtColor(bw_img_norm, cv2.COLOR_GRAY2BGR)

    h, w = bw_img.shape
    combined_img = np.zeros((h, w * 2, 3), dtype=np.uint8)
    combined_img[:, :w] = dist_map_color
    combined_img[:, w:] = bw_img_color

    cv2.imshow("Left: Color Distance Map | Right: Original Curve", combined_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def visualize_thick_thin(g_thin_good, g_thick_good, p_mask):
    p_map = np.zeros_like(p_mask, dtype=np.uint8)
    p_map[(g_thin_good > 0) & (g_thick_good == 0) & (p_mask > 0)] = 1
    p_map[(g_thin_good == 0) & (g_thick_good > 0) & (p_mask > 0)] = 2
    p_map[(g_thin_good == 0) & (g_thick_good == 0) & (p_mask > 0)] = 3
    p_map[(g_thin_good > 0) & (g_thick_good > 0) & (p_mask > 0)] = 4
    g_map = np.zeros_like(p_mask, dtype=np.uint8)
    g_map[(g_thin_good > 0) & (g_thick_good == 0)] = 1
    g_map[(g_thin_good == 0) & (g_thick_good > 0)] = 2
    g_map[(g_thin_good == 0) & (g_thick_good == 0)] = 3
    g_map[(g_thin_good > 0) & (g_thick_good > 0)] = 4
    show_maps(p_map, g_map)


def show_maps(p_map, g_map):
    h, w = p_map.shape
    p_color = np.zeros((h, w, 3), dtype=np.uint8)
    g_color = np.zeros((h, w, 3), dtype=np.uint8)

    p_color[p_map == 1] = [255, 0, 0]  # 蓝色 = thin
    p_color[p_map == 2] = [0, 0, 255]  # 红色 = thick
    p_color[p_map == 3] = [0, 255, 0]  # 绿色 = null
    p_color[p_map == 4] = [0, 255, 255]  # 黄色 = common

    g_color[g_map == 1] = [255, 0, 0]  # 蓝色 = thin
    g_color[g_map == 2] = [0, 0, 255]  # 红色 = thick
    g_color[g_map == 3] = [0, 255, 0]  # 绿色 = null
    g_color[g_map == 4] = [0, 255, 255]  # 黄色 = common
    combined = np.hstack((p_color, g_color))
    cv2.imshow("Pred (Left) vs GT (Right)", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
