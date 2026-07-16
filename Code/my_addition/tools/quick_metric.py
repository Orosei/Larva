import os
import time
import cv2
import copy

# 导入你已有的指标计算和保存函数
from Code.metric.metrics import cal_quick_metric, save_metric, show_metric, sub_metric


# TODO: 请根据你的实际项目结构，从对应文件导入 cal_average_metrics
# 例如，如果它在原来的 main.py 中：
# from main import cal_average_metrics 
# 或者如果移到了 metrics.py 中：
# from Code.metric.metrics import cal_average_metrics


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


def batch_evaluate_folders(pred_dir, label_dir, save_path=None):
    """
    通过匹配 basename，批量计算两个文件夹下对应二值图的指标。
    """
    pred_mapping = get_basename_mapping(pred_dir)
    label_mapping = get_basename_mapping(label_dir)

    if not pred_mapping or not label_mapping:
        print("未找到文件，请检查文件夹路径。")
        return

    # 获取两个文件夹中共同的 basename
    common_basenames = set(pred_mapping.keys()).intersection(set(label_mapping.keys()))

    if not common_basenames:
        print("在两个文件夹中没有找到匹配的文件名（basename）。")
        return

    print(f"成功匹配到 {len(common_basenames)} 对图像，开始计算指标...\n")

    metrics_list = []

    for basename in common_basenames:
        t0 = time.perf_counter()

        pred_path = os.path.join(pred_dir, pred_mapping[basename])
        label_path = os.path.join(label_dir, label_mapping[basename])

        # 使用 OpenCV 读取为单通道灰度图
        pred_img = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        label_img = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        if pred_img is None or label_img is None:
            print(f"读取失败，跳过: {basename}")
            continue

        metric1 = cal_quick_metric(pred_img, label_img)
        t1 = time.perf_counter()

        # cv2.imshow("pred_img", pred_img)
        # cv2.waitKey(0)
        # cv2.imshow("label_img", label_img)
        # cv2.waitKey(0)

        # 构造 metrics 字典，保持与你原有代码结构一致
        metrics = {
            "img": pred_mapping[basename],
            "time": t1 - t0,
            "x_coarse": metric1,
            "x_refine": copy.deepcopy(metric1),
            "x_gain": sub_metric(metric1, metric1),
        }

        show_metric(metrics)
        metrics_list.append(metrics)

    # 计算平均指标并展示
    if metrics_list:
        # print(metrics_list)
        avg_metrics = cal_average_metrics(metrics_list)

        print("\n========== 文件夹批量平均指标 (Average Metrics) ==========")
        # print(avg_metrics)
        show_metric(avg_metrics)

        if save_path:
            metrics_list.append(avg_metrics)
            save_metric(save_path, metrics_list)
            print(f"\n指标已保存至: {save_path}")

    return metrics_list


if __name__ == "__main__":
    # 配置路径
    PRED_DIR = "D:\\Master\\Snake\\Snake_2026_01_26\\Code\\my_addition\\others\\images\\union_seg_output"
    LABEL_DIR = "D:\\Master\\Snake\\Snake_2026_01_26\\Dataset\\CFD\\test\\labels"
    OUTPUT_ROOT = None  # 若不保存可设为 None

    batch_evaluate_folders(PRED_DIR, LABEL_DIR, save_path=OUTPUT_ROOT)
