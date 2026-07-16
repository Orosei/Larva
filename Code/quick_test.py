import copy
import json
import os
import sys
import time
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Code.metric import cal_quick_metric, cal_quick_metric_v2, save_metric, show_metric, sub_metric
from Code.my_addition import merge_1x3
from Code.process_stage import process_img
from Code.utils.image_draw import save_image, add_bottom_bar
from Code.utils.image_loader import init_loaders

from Code.my_addition import (project_root, delete_files, gen_video_v2, add_args, batch_gen_videos, delete_all,
                              save_output_v2)


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ───────────────────  通常的后缀命名规则  ────────────────────
# ┌──────────────────────────────────────────────────────────┐
# │ 后缀           类型含义          数据类型         取值范围    │
# │ _img         灰度图/彩色图      np.uint8        0 ~ 255    │
# │ _mask        二值掩码图         np.uint8/bool   0 或 1     │
# │ _map         整数映射图         np.uint8        0 ~ 255    │
# │ _nmap        归一化浮点特征图    np.float32      0.0 ~ 1.0  │
# │ _fmap        非归一化浮点特征图  np.float32      0.0 ~ +∞   │
# └──────────────────────────────────────────────────────────┘


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def cal_average_metrics(metrics_list):
    """
    计算所有图片指标的平均值。
    """
    if not metrics_list:
        return None

    count = len(metrics_list)
    # 深拷贝第一个元素作为累加器模板，避免修改原始数据
    # 移除 'img' 键，因为路径不能求平均
    avg_metrics = copy.deepcopy(metrics_list[0])
    if 'img' in avg_metrics:
        del avg_metrics['img']

    # 遍历剩余的每一个结果进行累加
    for i in range(1, count):
        curr = metrics_list[i]

        # 1. 累加时间
        avg_metrics['time'] += curr['time']

        # 2. 累加嵌套的字典指标 (x_coarse, x_refine, x_gain)
        target_keys = ['x_coarse', 'x_refine', 'x_gain']
        for main_key in target_keys:
            if main_key not in avg_metrics: continue

            # 遍历具体的指标项 (1_base, 2_ske, 3_comp 等)
            for sub_key, values in avg_metrics[main_key].items():
                curr_values = curr[main_key][sub_key]
                # 对应位置数值相加
                avg_metrics[main_key][sub_key] = [
                    v1 + v2 for v1, v2 in zip(values, curr_values)
                ]

    # 执行除法求平均并保留4位小数
    avg_metrics['time'] = round(avg_metrics['time'] / count, 4)

    target_keys = ['x_coarse', 'x_refine', 'x_gain']
    for main_key in target_keys:
        if main_key not in avg_metrics: continue

        for sub_key, values in avg_metrics[main_key].items():
            # 列表中的每个值除以总数
            avg_metrics[main_key][sub_key] = [
                round(v / count, 4) for v in values
            ]

    # 添加一个标记，表示这是平均值
    avg_metrics['img'] = "AVERAGE_OF_{}_IMAGES".format(count)

    return avg_metrics


def save_output(img, label, seg_img, refine_img, metrics):
    keys = ["x_coarse", "x_refine", "x_gain"]
    str_list = [
        " | ".join([f"{k}: {v}" for k, v in metrics.items() if k not in keys]),
        " | ".join([f"{k}: {v}" for k, v in metrics[keys[0]].items()]),
        " | ".join([f"{k}: {v}" for k, v in metrics[keys[1]].items()]),
        " | ".join([f"{k}: {v}" for k, v in metrics[keys[2]].items()]),
    ]

    # marked_img, merged_img = merge_2x2(img, label, seg_img, refine_img)
    marked_img, merged_img = merge_1x3(img, label, seg_img, refine_img)
    merged_img = add_bottom_bar(merged_img, str_list)

    paths = [os.path.join(config["output_root"], s) for s in config["output_subdirs"]]
    save_image(paths, metrics["img"], [marked_img, merged_img])
    return


def format_hms(seconds):
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{secs:05.2f}"


def main():
    paths = [os.path.join(config["input_root"], s) for s in config["input_subdirs"]]
    all_loaders = init_loaders(paths, config["color_flags"])
    img_loader, _, _, _ = all_loaders
    img_num = config.get("img_num", 0)
    img_num = img_num if img_num > 0 else img_loader.get_count()

    in_foreground = np.zeros(2, dtype=np.int32)

    print("output path:")
    print(add_args.output_path)
    delete_files(add_args.output_path)
    print("video path:")
    print(add_args.video_path)
    delete_all(add_args.video_path)

    metrics_list = []
    total_t0 = time.perf_counter()
    for i in range(img_num):
        t0 = time.perf_counter()

        # img: color image (np.uint8, 0-255)
        # label: label image (np.uint8, 0-255)
        # seg_img: segmentation (np.uint8, 0-255)

        img_name = os.path.splitext(os.path.basename(img_loader.get_path(i)))[0]

        img, label, seg_img, _ = (loader.get_image(i) for loader in all_loaders)
        refine_seg_img, in_foreground_add = process_img(img, label, seg_img, img_name, 1)
        t1 = time.perf_counter()

        in_foreground += in_foreground_add

        # metric1 = cal_quick_metric(seg_img, label)
        # metric2 = cal_quick_metric(refine_seg_img, label)
        metric1, metric2 = cal_quick_metric_v2(seg_img, refine_seg_img, label)
        metricd = sub_metric(metric2, metric1)

        metrics = {
            "img": img_loader.get_path(i),
            "time": t1 - t0,
            "x_coarse": metric1,
            "x_refine": metric2,
            "x_gain": metricd,
        }
        metrics_list.append(metrics)
        show_metric(metrics)

        # save_output(img, label, seg_img, refine_seg_img, metrics)
        save_output_v2(img, label, seg_img, refine_seg_img, metrics)

    """average metrics"""
    avg_metrics = cal_average_metrics(metrics_list)
    print("\n========== Average Metrics ==========")
    show_metric(avg_metrics)
    if avg_metrics:
        metrics_list.append(avg_metrics)
    save_metric(config["output_root"], metrics_list)

    bg_percent = in_foreground[1] / np.sum(in_foreground)
    print(f"snake move in background percent: {bg_percent:.2f}")

    batch_gen_videos(add_args.video_path)
    # gen_video_v2(add_args.video_path)
    total_t1 = time.perf_counter()
    print(f"main total time: {format_hms(total_t1 - total_t0)}")

    return


if __name__ == "__main__":
    config = load_config(os.path.join(os.path.dirname(__file__), "config.json"))
    parent_dir = os.path.dirname(os.path.dirname(__file__))
    config["input_root"] = os.path.join(parent_dir, config["input_root"])
    config["output_root"] = os.path.join(parent_dir, config["output_root"])

    main()
