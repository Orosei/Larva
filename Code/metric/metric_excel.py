import copy
import os
import time
from pprint import pprint

import pandas as pd


def cal_average_metrics(metrics_list):
    if not metrics_list:
        return None

    count = len(metrics_list)
    avg_metrics = copy.deepcopy(metrics_list[0])
    if "img" in avg_metrics:
        del avg_metrics["img"]

    for i in range(1, count):
        curr = metrics_list[i]
        avg_metrics["time"] += curr["time"]

        for key, values in avg_metrics.items():
            if key == "time":
                continue
            curr_values = curr[key]
            avg_metrics[key] = [v1 + v2 for v1, v2 in zip(values, curr_values)]

    avg_metrics["time"] = round(avg_metrics["time"] / count, 4)
    for key, values in avg_metrics.items():
        if key == "time":
            continue
        avg_metrics[key] = [round(v / count, 4) for v in values]

    avg_metrics["img"] = "AVERAGE_OF_{}_IMAGES".format(count)
    return avg_metrics


def flatten_dict(d, parent_key="", sep="_"):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k

        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, (list, tuple)):
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
    return output_path


def show_metric(obj):
    green = "\033[92m"
    reset = "\033[0m"
    print(f"{green}[OK]")
    pprint(obj, compact=True)
    print(f"{reset}")
    return
