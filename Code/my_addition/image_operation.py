import os

import numpy as np
import cv2

from ..utils.image_draw import color_diffs
from .utils import color_diffs_v2

cur_path = os.path.abspath(__file__)
root_path = cur_path
for _ in range(3):
    root_path = os.path.dirname(root_path)


def gen_video(fps=5):
    img_dir = os.path.join(root_path, "Output\\video")
    output_path = os.path.join(root_path, "Output\\video\\gif.avi")

    imgs = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.jpg')])
    s = cv2.imread(imgs[0]).shape
    w, h = s[1], s[0]
    fourccs = [cv2.VideoWriter_fourcc(*'MJPG'), cv2.VideoWriter_fourcc(*'avc1'),
               cv2.VideoWriter_fourcc(*'mp4v'), cv2.VideoWriter_fourcc(*'XVID')]
    vw = None
    for fourcc in fourccs:
        try:
            vw = cv2.VideoWriter(output_path, fourcc, fps, (w, h), True)
            break
        except:
            continue
    for i in imgs:
        frame = cv2.imread(i)
        if frame is not None: vw.write(frame)
    vw.release()
    print("---success generate---")


def batch_gen_videos(parent_root_path):
    if not os.path.exists(parent_root_path):
        print(f"[Error] 父路径不存在: {parent_root_path}")
        return

    items = sorted(os.listdir(parent_root_path))

    print(f"--- 开始批量处理，根目录: {parent_root_path} ---")
    count = 0

    for item in items:
        current_abs_path = os.path.join(parent_root_path, item)
        if os.path.isdir(current_abs_path):
            print(f"正在处理: {item} ...")
            try:
                gen_video_v2(current_abs_path)
                count += 1
            except Exception as e:
                print(f"[Error] 处理文件夹 {item} 失败: {e}")

    print(f"--- 批量处理完成，共处理 {count} 个文件夹 ---")


def gen_video_v2(abs_path, fps=5):
    img_dir = abs_path
    output_path = os.path.join(abs_path, "video.avi")

    imgs = sorted([
        os.path.join(img_dir, f)
        for f in os.listdir(img_dir)
        if f.lower().endswith('.jpg')
    ])

    if not imgs:
        print(f"[Error] 在 {img_dir} 中没有找到 jpg 图片")
        return

    first_frame = cv2.imread(imgs[0])
    if first_frame is None:
        print(f"[Error] 无法读取图片: {imgs[0]}")
        return

    h, w = first_frame.shape[:2]

    fourccs = [
        cv2.VideoWriter_fourcc(*'MJPG'),
        cv2.VideoWriter_fourcc(*'mp4v'),
        cv2.VideoWriter_fourcc(*'XVID'),
        cv2.VideoWriter_fourcc(*'avc1')
    ]

    vw = None
    success = False

    for fourcc in fourccs:
        try:
            vw = cv2.VideoWriter(output_path, fourcc, fps, (w, h), True)
            if vw.isOpened():
                success = True
                break
        except Exception as e:
            continue

    if not success or vw is None:
        print("[Error] 无法初始化 VideoWriter，请检查编码器")
        return

    print(f"正在生成视频到: {output_path} ...")
    for img_path in imgs:
        frame = cv2.imread(img_path)
        if frame is not None:
            if frame.shape[:2] != (h, w):
                frame = cv2.resize(frame, (w, h))
            vw.write(frame)

    vw.release()
    print("---success generate---")

    return


def draw_dir(pt, img, prob_pred_16, pred_dir):
    idxs = get_top_3_direction(prob_pred_16)

    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]

    for i in range(3):
        idx = idxs[i]
        if idx == pred_dir:
            color = colors[0]
        else:
            color = colors[1]
        angle = idx * (360 / 16)
        rad = np.radians(angle)

        end_y = int(pt[0] - 15 * np.sin(rad))
        end_x = int(pt[1] + 15 * np.cos(rad))

        end_y = np.clip(end_y, 0, img.shape[0] - 1)
        end_x = np.clip(end_x, 0, img.shape[1] - 1)

        cv2.arrowedLine(img, (pt[1], pt[0]), (end_x, end_y), color, thickness=1, tipLength=0.3)

    return img


def draw_dir_v2(pt, img, dir_16_pred, main_dir):
    valid_idxs = np.where(dir_16_pred > 0)[0]

    color_pred = (0, 0, 255)
    color_valid = (0, 255, 0)

    for idx in valid_idxs:
        if idx == main_dir:
            color = color_pred
        else:
            color = color_valid

        angle = idx * (360 / 16)
        rad = np.radians(angle)

        end_y = int(pt[0] - 15 * np.sin(rad))
        end_x = int(pt[1] + 15 * np.cos(rad))

        end_y = np.clip(end_y, 0, img.shape[0] - 1)
        end_x = np.clip(end_x, 0, img.shape[1] - 1)

        cv2.arrowedLine(img, (int(pt[1]), int(pt[0])), (end_x, end_y), color, thickness=1, tipLength=0.3)

    return img


def get_top_3_direction(probs):
    sorted_idx = np.argsort(probs)[::-1][:3]
    return sorted_idx


# def merge_1x3(img, label, seg_map, refine_map):
#     gap_color = (200, 200, 200)
#     gap_width = 8
#
#     label_img = cv2.cvtColor(label, cv2.COLOR_GRAY2BGR)
#     refine_vis = cv2.cvtColor(seg_map, cv2.COLOR_GRAY2BGR)
#
#     marked_refine = color_diffs(refine_vis, seg_map, refine_map, color=(0, 0, 255))
#
#     h, w = img.shape[:2]
#     gap_v = np.full((h, gap_width, 3), gap_color, dtype=np.uint8)
#
#     merged_img = np.hstack((img, gap_v, marked_refine, gap_v, label_img))
#
#     return marked_refine, merged_img


def merge_1x3(img, label, seg_map, refine_map):
    gap_color = (200, 200, 200)
    gap_width = 8

    label_img = cv2.cvtColor(label, cv2.COLOR_GRAY2BGR)
    clr_img = color_comps(seg_map)
    # refine_vis = cv2.cvtColor(clr_img, cv2.COLOR_GRAY2BGR)

    # marked_refine = color_diffs(seg_map, seg_map, refine_map, color=(0, 0, 255))
    # marked_clr = color_diffs(clr_img, seg_map, refine_map, color=(0, 0, 255))

    marked_refine = color_diffs_v2(seg_map, seg_map, refine_map, color=(0, 0, 255))
    marked_clr = color_diffs_v2(clr_img, seg_map, refine_map, color=(0, 0, 255))

    h, w = img.shape[:2]
    gap_v = np.full((h, gap_width, 3), gap_color, dtype=np.uint8)

    merged_img = np.hstack((img, gap_v, label_img, gap_v, marked_clr))

    return marked_refine, marked_clr, merged_img


def color_comps(bw_mask):
    num, label_map = cv2.connectedComponents(bw_mask, connectivity=8)
    clr_img = color_instance(label_map)
    # clr_img1 = brighten_skeleton(label_map * 255, clr_img)
    # show_image(clr_img1)
    return clr_img


def color_instance(id_map):
    clr_img = np.zeros((id_map.shape[0], id_map.shape[1], 3), dtype=np.uint8)
    # 获取所有非0且非-1的ID
    ids = [i for i in np.unique(id_map) if i > 0]

    for i, id_ in enumerate(ids):
        if id_ == 1:
            # 【策略 1】 ID=1 设为鲜绿色 (Green)
            # 这里的 (0, 255, 0) 对应 HSV 的 Hue=60，非常显眼且绝对不是红色
            bgr = (0, 255, 0)
        else:
            # 【策略 2】 其他 ID 使用冷色调区间 (蓝 -> 紫)
            # 我们将 Hue 限制在 [95, 165] 之间
            # 95是青蓝色，165是洋红色。
            # 这样既避开了红色(0/180)，也避开了绿色(60)

            # 使用简单的哈希算法确保不同ID颜色不同，但都在安全区间内
            # 步长 37 是一个质数，有助于颜色在区间内随机分布而不重复
            hue_offset = (id_ * 37) % 70
            hue = 95 + hue_offset

            # 饱和度(S)和明度(V)设为最大，保证颜色鲜艳
            hsv = np.uint8([[[hue, 255, 255]]])

            # 转换回 BGR
            # [0][0] 是因为 cvtColor 输入输出都需要是图像矩阵形式
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
            # 转为 tuple 格式赋值
            bgr = tuple(int(x) for x in bgr)

        clr_img[id_map == id_] = bgr

    clr_img[id_map == -1] = [255, 255, 255]  # -1 标识为白色
    return clr_img
