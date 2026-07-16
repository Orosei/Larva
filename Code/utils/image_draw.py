import os

import cv2
import numpy as np


def color_diffs(src_img, base_mask, diff_mask, color=(0, 0, 255)):
    marked_img = src_img.copy()
    mask = (diff_mask > 0) & (base_mask == 0)
    marked_img[mask] = color
    return marked_img


def save_image(paths, img_path, imgs):
    [os.makedirs(p, exist_ok=True) for p in paths]

    base_name = os.path.splitext(os.path.basename(img_path))[0]
    for i, path in enumerate(paths):
        # show_image(images[i])
        cv2.imwrite(os.path.join(path, f"{base_name}.png"), imgs[i])
    return


def merge_2x2(img, label, seg_map, refine_map):
    gap_color = (200, 200, 200)
    gap_width = 8

    label_img = cv2.cvtColor(label, cv2.COLOR_GRAY2BGR)
    clr_img1 = cv2.cvtColor(seg_map, cv2.COLOR_GRAY2BGR)
    clr_img2 = color_curves(seg_map)
    marked_img1 = color_diffs(clr_img1, seg_map, refine_map)
    marked_img2 = color_diffs(clr_img2, seg_map, refine_map)

    h1, w1 = img.shape[:2]

    gap_h = np.full((h1, gap_width, 3), gap_color, dtype=np.uint8)
    gap_v = np.full((gap_width, w1 + gap_width + w1, 3), gap_color, dtype=np.uint8)

    row1 = np.hstack((img, gap_h, label_img))
    row2 = np.hstack((marked_img1, gap_h, marked_img2))
    merged_img = np.vstack((row1, gap_v, row2))
    return marked_img1, merged_img


def show_image(img, win_name="Image"):
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, img.shape[1], img.shape[0])
    cv2.imshow(win_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return None


def add_bottom_bar(img, str_list, bar_color=(200, 200, 200), text_color=(255, 0, 0)):
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale, thickness = 0.3, 1
    line_imgs = []

    for s in str_list:
        (tw, th), _ = cv2.getTextSize(s, font, font_scale, thickness)
        line_img = np.full((th + 10, w, 3), bar_color, dtype=np.uint8)
        cv2.putText(line_img, s, (10, th + 5), font, font_scale, text_color, thickness)
        line_imgs.append(line_img)

    bar_img = np.vstack(line_imgs)
    return np.vstack((img, bar_img))


def dist_skeleton(id_map):
    skel_map = cv2.ximgproc.thinning((id_map > 0).astype(np.uint8) * 255)
    dist_fmap = cv2.distanceTransform(255 - skel_map, cv2.DIST_L2, 5)
    dist_nmap = cv2.normalize(dist_fmap, None, 0.0, 1.0, cv2.NORM_MINMAX)
    return dist_nmap


def brighten_skeleton(id_map, clr_img, add=100):
    dist_nmap = dist_skeleton(id_map)
    mask = (id_map > 0) & (np.isclose(dist_nmap, 0))
    bright_img = clr_img.copy()
    bright_img[mask] = np.clip(bright_img[mask].astype(np.int32) + add, 0, 255).astype(np.uint8)
    return bright_img


def color_instance(id_map):
    clr_img = np.zeros((id_map.shape[0], id_map.shape[1], 3), dtype=np.uint8)
    ids = [i for i in np.unique(id_map) if i > 0]
    for i, id_ in enumerate(ids):
        hue = int(180 * i / len(ids)) % 180
        hsv = np.uint8([[[hue, 255, 255]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        clr_img[id_map == id_] = bgr[0][0]
    clr_img[id_map == -1] = [255, 255, 255]  # -1 标识为白色
    return clr_img


def color_curves(bw_mask):
    num, label_map = cv2.connectedComponents(bw_mask, connectivity=4)
    clr_img = color_instance(label_map)
    clr_img1 = brighten_skeleton(label_map * 255, clr_img)
    # show_image(clr_img1)
    return clr_img1
