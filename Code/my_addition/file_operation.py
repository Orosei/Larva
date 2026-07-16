import os
import shutil
import cv2

from .setting import add_args
from .image_operation import merge_1x3
from .utils import gen_refine_seg, color_diffs_v2
from ..utils import add_bottom_bar, process_curve, process_curve_v2, process_skeleton, color_instance


def delete_files(abs_path):
    if not os.path.exists(abs_path):
        os.makedirs(abs_path)

    for item in os.listdir(abs_path):
        item_path = os.path.join(abs_path, item)

        try:
            if os.path.isfile(item_path):
                os.remove(item_path)
        except Exception as e:
            print(f"[Error] 删除文件 {item} 失败: {e}")

    return


def delete_all(abs_path):
    if not os.path.exists(abs_path):
        os.makedirs(abs_path)
        return

    for item in os.listdir(abs_path):
        item_path = os.path.join(abs_path, item)

        try:
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
        except Exception as e:
            print(f"[Error] 删除 {item} 失败: {e}")

    return


def save_output_v2(img, label, seg_img, refine_img, metrics):
    keys = ["x_coarse", "x_refine", "x_gain"]
    str_list = [
        " | ".join([f"{k}: {v}" for k, v in metrics.items() if k not in keys]),
        " | ".join([f"{k}: {v}" for k, v in metrics[keys[0]].items()]),
        " | ".join([f"{k}: {v}" for k, v in metrics[keys[1]].items()]),
        " | ".join([f"{k}: {v}" for k, v in metrics[keys[2]].items()]),
    ]

    ske_num, ske_map, _ = process_skeleton(seg_img)
    # curve_map = process_curve(seg_img, ske_num, ske_map)
    curve_map = process_curve_v2(seg_img, ske_num, ske_map)
    curve_clr = color_instance(curve_map)
    seg_color = color_diffs_v2(curve_clr, seg_img, refine_img)

    refine_seg = gen_refine_seg(seg_img, refine_img)
    # marked_img, seg_color, merged_img = merge_1x3(img, label, seg_img, refine_img)
    marked_img, _, merged_img = merge_1x3(img, label, seg_img, refine_img)
    merged_img = add_bottom_bar(merged_img, str_list)

    paths = [
        add_args.refine_seg_path,
        add_args.marked_refine_path,
        add_args.refine_seg_color_path,
        add_args.bar_img_path
    ]

    save_image(paths, metrics["img"], [refine_seg, marked_img, seg_color, merged_img])
    return


def save_image(paths, img_path, imgs):
    [os.makedirs(p, exist_ok=True) for p in paths]
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    for i, path in enumerate(paths):
        cv2.imwrite(os.path.join(path, f"{base_name}.png"), imgs[i])
    return
