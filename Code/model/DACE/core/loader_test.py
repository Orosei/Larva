import json
import os

import cv2
import numpy as np

VALID_EXTS = (".tif", ".png", ".jpg", ".jpeg", ".bmp")


def get_file_paths(args):
    image_dir = args.IMAGE_DIR
    label_dir = args.LABEL_DIR

    label_map = {}
    if label_dir and os.path.isdir(label_dir):
        with os.scandir(label_dir) as dir_entries:
            for entry in dir_entries:
                if entry.is_file() and entry.name.lower().endswith(VALID_EXTS):
                    stem = os.path.splitext(entry.name)[0].lower()
                    label_map[stem] = entry.path

    image_paths = []
    label_paths = []
    with os.scandir(image_dir) as dir_entries:
        for entry in dir_entries:
            if entry.is_file() and entry.name.lower().endswith(VALID_EXTS):
                stem = os.path.splitext(entry.name)[0].lower()
                label_path = label_map.get(stem)
                if label_path is None:
                    print(f"⚠️ No matching ground truth label found for [{stem}]")
                image_paths.append(entry.path)
                label_paths.append(label_path)

    return image_paths, label_paths


def load_image(image_path, label_path=None):
    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Image not found or unreadable: {image_path}")
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    image = gray.astype(np.float32) / 255.0
    label = None
    if label_path:
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        if label is not None:
            label = (label > 0).astype(np.float32)
    return image, label, bgr


def load_point(args, image_path, image_bgr):
    os.makedirs(args.OUTPUT_DIR, exist_ok=True)
    stem = os.path.splitext(os.path.basename(image_path))[0]
    json_path = os.path.join(args.PT_DIR, f"{stem}_points.json")
    return _select_point(json_path, image_path, image_bgr, 10)


def _on_click(event, x, y, _flags, param):
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    pts, canvas, max_n = param["points"], param["canvas"], param["max_points"]
    if len(pts) >= max_n:
        return
    pts.append([x, y])
    cv2.circle(canvas, (x, y), 1, (0, 0, 255), -1)
    cv2.putText(canvas, str(len(pts)), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.imshow(param["window_name"], canvas)


def _select_point(json_path, image_path, image_bgr, max_points):
    if os.path.exists(json_path):
        return _load_json(json_path)

    points = []
    window_name = f"Select Points - {image_path}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)

    canvas = image_bgr.copy()
    cv2.putText(
        canvas,
        f"Click {max_points} points, ENTER to confirm, Q to skip",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )

    cv2.setMouseCallback(
        window_name,
        _on_click,
        {"points": points, "canvas": canvas, "window_name": window_name, "max_points": max_points},
    )
    cv2.imshow(window_name, canvas)
    print(f"👉 请在图像上点击 {max_points} 个起点，按回车键确认...")

    while True:
        key = cv2.waitKey(1) & 0xFF
        if len(points) >= max_points or (key == 13 and points):
            print(f"👉 已选择 {len(points)} 个点")
            break
        if key == 13:
            print("⚠️ 至少选择一个点！")
        if key == ord("q"):
            print("⚠️ 已跳过，未选择任何点。")
            cv2.destroyWindow(window_name)
            return []

    cv2.destroyWindow(window_name)
    _save_json(json_path, points)
    return points


def _load_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    print(f"✅ 已从缓存加载 {len(data)} 个起点: {path}")
    return data


def _save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f)
    print(f"💾 已保存 {len(data)} 个起点到: {path}")
