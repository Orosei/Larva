import argparse
import os
import sys
import time

import cv2
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Code.metric.basic_metric import cal_basic_metric, cal_thick_metric_v2
from Code.metric.clDice import cal_cldice_metric_v2
from Code.metric.metric_excel import cal_average_metrics, save_metric, show_metric
from Code.metric.metric_label_cache import prepare_metric_label_cache
from Code.metric.skeleton_metric import cal_skeleton_metric_v2
from Code.my_addition.metrics import cal_comp_metrics_v3


IMAGE_EXTS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def cal_single_metric_v2(pred, label):
    pred_mask = (pred > 0).astype(np.uint8)
    label_mask = (label > 0).astype(np.uint8)
    label_cache = prepare_metric_label_cache(label_mask)

    base = cal_basic_metric(pred_mask, label_mask)
    thin, thick = cal_thick_metric_v2(pred_mask, label_cache)
    ske = cal_skeleton_metric_v2(pred_mask, label_cache)
    comp = cal_comp_metrics_v3(pred_mask, label_cache)
    topology = cal_cldice_metric_v2(pred_mask, label_mask)

    metric_dict = {
        "1_base": base,
        "1_thin": thin,
        "1_thick": thick,
        "2_ske": ske,
        "3_comp": comp,
        "4_topology": topology,
    }
    return convert_and_format(metric_dict)


def convert_and_format(obj, decimals=4):
    if isinstance(obj, dict):
        return {k: convert_and_format(v, decimals) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(convert_and_format(x, decimals) for x in obj)
    if isinstance(obj, np.generic):
        py_val = obj.item()
        if isinstance(py_val, float):
            return round(py_val, decimals)
        return py_val
    if isinstance(obj, float):
        return round(obj, decimals)
    return obj


def get_basename_mapping(folder_path):
    mapping = {}
    if not os.path.isdir(folder_path):
        print(f"folder not found: {folder_path}")
        return mapping

    for filename in os.listdir(folder_path):
        full_path = os.path.join(folder_path, filename)
        basename, ext = os.path.splitext(filename)
        if os.path.isfile(full_path) and ext.lower() in IMAGE_EXTS:
            mapping[basename] = filename

    return mapping


def batch_evaluate_folders(pred_dir, label_dir, save_path=None):
    pred_mapping = get_basename_mapping(pred_dir)
    label_mapping = get_basename_mapping(label_dir)

    if not pred_mapping or not label_mapping:
        print("No images found. Please check pred_dir and label_dir.")
        return []

    common_basenames = sorted(set(pred_mapping.keys()).intersection(label_mapping.keys()))
    if not common_basenames:
        print("No matched image basename found between pred_dir and label_dir.")
        return []

    print(f"Matched {len(common_basenames)} images. Start calculating metrics...\n")

    metrics_list = []
    for basename in common_basenames:
        t0 = time.perf_counter()

        pred_path = os.path.join(pred_dir, pred_mapping[basename])
        label_path = os.path.join(label_dir, label_mapping[basename])

        pred_img = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        label_img = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        if pred_img is None or label_img is None:
            print(f"Failed to read image, skipped: {basename}")
            continue
        if pred_img.shape != label_img.shape:
            print(f"Shape mismatch, skipped: {basename}, pred={pred_img.shape}, label={label_img.shape}")
            continue

        metric = cal_single_metric_v2(pred_img, label_img)
        t1 = time.perf_counter()

        metrics = {
            "img": pred_mapping[basename],
            "time": t1 - t0,
            **metric,
        }
        metrics_list.append(metrics)
        show_metric(metrics)

    if not metrics_list:
        print("No valid image pair was calculated.")
        return []

    avg_metrics = cal_average_metrics(metrics_list)
    print("\n========== Average Metrics ==========")
    show_metric(avg_metrics)

    if save_path:
        metrics_list.append(avg_metrics)
        output_path = save_metric(save_path, metrics_list)
        print(f"\nmetrics saved to: {output_path}")

    return metrics_list


def parse_args():
    parser = argparse.ArgumentParser(description="Batch calculate binary segmentation metrics.")
    parser.add_argument("--pred_dir", default=r"Dataset\Massachusetts_Roads_top200\val\coarse_seg\iternet\soft_dice_cldice", help="Folder of predicted binary images.")
    parser.add_argument("--label_dir", default=r"Dataset\Massachusetts_Roads_top200\val\labels", help="Folder of label binary images.")
    parser.add_argument("--save_path", default=r"Output\metric\Massachusetts_Roads_top200\val\iternet\soft_dice_cldice", help="Output folder for metrics_*.xlsx.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    batch_evaluate_folders(args.pred_dir, args.label_dir, save_path=args.save_path)
