import os
import time
import cv2
import copy

# 导入你已有的指标计算和保存函数
from Code.metric.metrics import cal_quick_metric, save_metric, show_metric, sub_metric


def get_basename_mapping(folder_path):
    """
    遍历文件夹，建立 basename 到完整文件名的映射字典。
    """
    mapping = {}
    if not os.path.exists(folder_path):
        print(f"警告：文件夹不存在 -> {folder_path}")
        return mapping

    for filename in os.listdir(folder_path):
        full_path = os.path.join(folder_path, filename)
        if os.path.isfile(full_path):
            # 分离文件名和后缀，例如 "img_01.png" -> "img_01"
            basename = os.path.splitext(filename)[0]
            mapping[basename] = filename

    return mapping


def cal_average_metrics(metrics_list):
    """
    计算所有图片指标的平均值。
    """
    if not metrics_list:
        return None

    count = len(metrics_list)
    # 深拷贝第一个元素作为累加器模板，避免修改原始数据
    avg_metrics = copy.deepcopy(metrics_list[0])
    if 'img' in avg_metrics:
        del avg_metrics['img']

    # 遍历剩余的每一个结果进行累加
    for i in range(1, count):
        curr = metrics_list[i]

        avg_metrics['time'] += curr['time']

        target_keys = ['x_coarse', 'x_refine', 'x_gain']
        for main_key in target_keys:
            if main_key not in avg_metrics: continue

            for sub_key, values in avg_metrics[main_key].items():
                curr_values = curr[main_key][sub_key]
                avg_metrics[main_key][sub_key] = [
                    v1 + v2 for v1, v2 in zip(values, curr_values)
                ]

    # 执行除法求平均并保留4位小数
    avg_metrics['time'] = round(avg_metrics['time'] / count, 4)

    target_keys = ['x_coarse', 'x_refine', 'x_gain']
    for main_key in target_keys:
        if main_key not in avg_metrics: continue

        for sub_key, values in avg_metrics[main_key].items():
            avg_metrics[main_key][sub_key] = [
                round(v / count, 4) for v in values
            ]

    avg_metrics['img'] = "AVERAGE_OF_{}_IMAGES".format(count)

    return avg_metrics


def batch_evaluate_folders(coarse_dir, refine_dir, label_dir, save_path=None):
    """
    通过匹配 basename，批量计算三个文件夹下对应的二值图指标，评估精细化提升。
    """
    coarse_mapping = get_basename_mapping(coarse_dir)
    refine_mapping = get_basename_mapping(refine_dir)
    label_mapping = get_basename_mapping(label_dir)

    if not coarse_mapping or not refine_mapping or not label_mapping:
        print("未找到文件或部分文件夹为空，请检查路径。")
        return

    # 获取三个文件夹中共同的 basename (三者取交集)
    common_basenames = set(coarse_mapping.keys()) \
        .intersection(set(refine_mapping.keys())) \
        .intersection(set(label_mapping.keys()))

    if not common_basenames:
        print("在三个文件夹中没有找到完全匹配的文件名（basename）。")
        return

    print(f"成功匹配到 {len(common_basenames)} 组图像，开始计算指标...\n")

    metrics_list = []

    for basename in common_basenames:
        t0 = time.perf_counter()

        coarse_path = os.path.join(coarse_dir, coarse_mapping[basename])
        refine_path = os.path.join(refine_dir, refine_mapping[basename])
        label_path = os.path.join(label_dir, label_mapping[basename])

        # 读取为单通道灰度图
        coarse_img = cv2.imread(coarse_path, cv2.IMREAD_GRAYSCALE)
        refine_img = cv2.imread(refine_path, cv2.IMREAD_GRAYSCALE)
        label_img = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        if coarse_img is None or refine_img is None or label_img is None:
            print(f"读取失败，跳过: {basename}")
            continue

        # 分别计算粗分割和精分割的指标
        metric_coarse = cal_quick_metric(coarse_img, label_img)
        metric_refine = cal_quick_metric(refine_img, label_img)

        t1 = time.perf_counter()

        # 构造 metrics 字典，此时 x_coarse 和 x_refine 是独立的计算结果
        metrics = {
            "img": refine_mapping[basename],  # 以 refine 图的文件名为准记录
            "time": t1 - t0,
            "x_coarse": metric_coarse,
            "x_refine": metric_refine,
            "x_gain": sub_metric(metric_refine, metric_coarse),  # 计算真实提升
        }

        show_metric(metrics)
        metrics_list.append(metrics)

    # 计算平均指标并展示
    if metrics_list:
        avg_metrics = cal_average_metrics(metrics_list)

        print("\n========== 文件夹批量平均指标 (Average Metrics) ==========")
        show_metric(avg_metrics)

        if save_path:
            metrics_list.append(avg_metrics)
            save_metric(save_path, metrics_list)
            print(f"\n指标已保存至: {save_path}")

    return metrics_list


if __name__ == "__main__":
    BASE_DATASET_DIR = r"D:\Master\Snake\Snake_2026_01_26\Dataset"
    BASE_OUTPUT_DIR = r"D:\Master\Snake\Snake_2026_01_26\Code\my_addition\others\images"

    dataset_name = "CFD"
    phase = "test"
    backbone = "IterNet"

    COARSE_DIR = os.path.join(BASE_DATASET_DIR, dataset_name, phase, "coarse_seg", backbone)
    REFINE_DIR = os.path.join(BASE_OUTPUT_DIR, dataset_name, phase, "combine", backbone)
    LABEL_DIR = os.path.join(BASE_DATASET_DIR, dataset_name, phase, "labels")
    OUTPUT_ROOT = os.path.join(BASE_OUTPUT_DIR, dataset_name, phase, "metrics")

    print(f"[*] 粗分割结果 (COARSE): {COARSE_DIR}")
    print(f"[*] 细化结果   (REFINE): {REFINE_DIR}")
    print(f"[*] 真实标签   (LABEL):  {LABEL_DIR}")
    print(f"[*] 指标输出   (OUTPUT): {OUTPUT_ROOT}")
    print("-" * 50)

    batch_evaluate_folders(COARSE_DIR, REFINE_DIR, LABEL_DIR, save_path=OUTPUT_ROOT)
