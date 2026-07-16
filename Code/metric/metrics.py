import os
import time
from pprint import pprint

import numpy as np
import pandas as pd

from .basic_metric import cal_basic_metric, cal_thick_metric, cal_thick_metric_v2
from .clDice import cal_cldice_metric_v2
from .skeleton_metric import cal_skeleton_metric, cal_skeleton_metric_v2
from .metric_label_cache import prepare_metric_label_cache
from ..my_addition.metrics import cal_comp_metrics_v2, cal_comp_metrics_v3


def cal_quick_metric(pred, label):
    pred_mask = (pred > 0).astype(np.uint8)
    label_mask = (label > 0).astype(np.uint8)

    base = cal_basic_metric(pred_mask, label_mask)  # f1 pre rec iou
    thin, thick = cal_thick_metric(pred_mask, label_mask)
    ske = cal_skeleton_metric(pred_mask, label_mask)  # bmIoU SDS
    # comp = cal_comp_metrics(pred_mask, label_mask)
    comp = cal_comp_metrics_v2(pred_mask, label_mask) # NCCE NAAE brr?

    metric_dict = {
        "1_base": base,
        "1_thin": thin,
        "1_thick": thick,
        "2_ske": ske,
        "3_comp": comp,
    }

    standard_dict = convert_and_format(metric_dict)
    return standard_dict


# ==================== cal_quick_metric_v2 ====================

def cal_quick_metric_v2(pred, refine, label):
    pred_mask = (pred > 0).astype(np.uint8)
    refine_mask = (refine > 0).astype(np.uint8)
    label_mask = (label > 0).astype(np.uint8)
    label_cache = prepare_metric_label_cache(label_mask)

    pred_base = cal_basic_metric(pred_mask, label_mask)  # f1 pre rec iou
    pred_thin, pred_thick = cal_thick_metric_v2(pred_mask, label_cache)
    pred_ske = cal_skeleton_metric_v2(pred_mask, label_cache)  # bmIoU SDS
    pred_comp = cal_comp_metrics_v3(pred_mask, label_cache)  # NCCE NAAE brr?
    pred_topology = cal_cldice_metric_v2(pred_mask, label_mask)  # clDice

    refine_base = cal_basic_metric(refine_mask, label_mask)  # f1 pre rec iou
    refine_thin, refine_thick = cal_thick_metric_v2(refine_mask, label_cache)
    refine_ske = cal_skeleton_metric_v2(refine_mask, label_cache)  # bmIoU SDS
    refine_comp = cal_comp_metrics_v3(refine_mask, label_cache)  # NCCE NAAE brr?
    refine_topology = cal_cldice_metric_v2(refine_mask, label_mask)  # clDice

    pred_metric_dict = {
        "1_base": pred_base,
        "1_thin": pred_thin,
        "1_thick": pred_thick,
        "2_ske": pred_ske,
        "3_comp": pred_comp,
        "4_topology": pred_topology,
    }

    refine_metric_dict = {
        "1_base": refine_base,
        "1_thin": refine_thin,
        "1_thick": refine_thick,
        "2_ske": refine_ske,
        "3_comp": refine_comp,
        "4_topology": refine_topology,
    }

    return convert_and_format(pred_metric_dict), convert_and_format(refine_metric_dict)


def convert_and_format(obj, decimals=4):
    if isinstance(obj, dict):
        return {k: convert_and_format(v, decimals) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(convert_and_format(x, decimals) for x in obj)
    elif isinstance(obj, np.generic):
        py_val = obj.item()
        if isinstance(py_val, float):
            return round(py_val, decimals)
        return py_val
    elif isinstance(obj, float):
        return round(obj, decimals)
    else:
        return obj


def sub_metric(a, b):
    """递归对两个嵌套字典对应位置相减"""
    result = {}
    for k in a:
        if k not in b:
            result[k] = a[k]
            continue

        val_a = a[k]
        val_b = b[k]

        if isinstance(val_a, dict) and isinstance(val_b, dict):
            result[k] = sub_metric(val_a, val_b)
        elif isinstance(val_a, (tuple, list)) and isinstance(val_b, (tuple, list)):
            result[k] = tuple(x - y for x, y in zip(val_a, val_b))
        else:
            result[k] = val_a - val_b
    result = convert_and_format(result)
    return result


def flatten_dict(d, parent_key="", sep="_"):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k

        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, (list, tuple)):
            # 处理列表/元组，把索引加入键名
            for i, val in enumerate(v):
                items.append((f"{new_key}_{i}", val))
        else:
            items.append((new_key, v))

    return dict(items)


def save_metric(folder, metrics_list):
    os.makedirs(folder, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(folder, f"metrics_{timestamp}.xlsx")

    df = pd.DataFrame([flatten_dict(m) for m in metrics_list])
    df.to_excel(output_path, index=False, engine="openpyxl")
    return


def show_metric(obj):
    GREEN = "\033[92m"
    RESET = "\033[0m"
    print(f"{GREEN}✅")
    pprint(obj, compact=True)
    print(f"{RESET}")
    return
