import os

import cv2
import numpy as np

from Code.my_addition.Model import vit_main, vit_int


def cal_angle(label, labels):
    mask = (labels == label).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        if contours[0].shape[0] > 10:
            ellipse = cv2.fitEllipse(contours[0])
            angle = ellipse[2]
            return (90 - angle) % 180
        return -1


def cal_angle_diffs(blob_angle, angle_interval=45):
    if blob_angle == -1:
        return None, -1
    angles = np.arange(0, 180, angle_interval)
    diffs = np.abs(blob_angle - angles)

    first_diff = diffs[0]
    diffs[0] = min(first_diff, abs(blob_angle - 180))
    min_diff_value = np.min(diffs)
    min_id = np.where(diffs == min_diff_value)[0][0]
    return diffs, min_id


def reorder_connected_components(image):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
    region_info = sorted([(stats[i][4], i) for i in range(1, num_labels)])
    new_labels = np.zeros_like(labels)
    angles = [-1]
    angles_id = [-1]
    label_idx = 1
    for area, orig_label in region_info:
        new_labels[labels == orig_label] = label_idx
        angle = cal_angle(orig_label, labels) if area > 10 else -1
        angle_id = cal_angle_diffs(angle)[1] if angle != -1 else -1
        angles.append(angle)
        angles_id.append(angle_id)
        label_idx += 1
    return num_labels, new_labels, angles, angles_id


def adjust_probs_by_angle(probs, blob_angle, factor=3.0, bonus=0.1):
    if blob_angle == -1:  # -1无效值
        return probs

    diffs, id = cal_angle_diffs(blob_angle)
    angle_weights = 1 + np.maximum(0, factor * (1.0 - diffs / 45))
    extended_angle_weights = np.concatenate([angle_weights.T, angle_weights.T])
    probs1 = probs * extended_angle_weights[:, np.newaxis]

    mask = (probs1[id, :] > probs1[4 + id, :])
    probs1[id, :] += mask * bonus
    probs1[4 + id, :] += (~mask) * bonus

    probs1 /= probs1.sum(axis=0)
    return probs1


def crop_with_center(img, center_row, center_col, width, height):
    half_w = width // 2
    half_h = height // 2
    start_r = center_row - half_h
    start_c = center_col - half_w
    end_r = start_r + height
    end_c = start_c + width

    pad_top = max(0, -start_r)
    pad_bottom = max(0, end_r - img.shape[0])
    pad_left = max(0, -start_c)
    pad_right = max(0, end_c - img.shape[1])

    if any([pad_top, pad_bottom, pad_left, pad_right]):
        padded_img = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
    else:
        padded_img = img
    return padded_img[start_r + pad_top:end_r + pad_top, start_c + pad_left:end_c + pad_left]


gsolver = None


def smart_force(row, col, img, field):
    global gsolver
    if gsolver is None:
        gsolver = vit_int(False)

    prob = np.zeros(8)
    h, w = field, field
    sub_im = crop_with_center(img, row, col, w, h)
    cls = vit_main(gsolver, sub_im)
    prob[cls] = 1.0
    return prob


def smart_gravity_prob(rows, cols, image, starttime):
    global gsolver
    if gsolver is None:
        gsolver = vit_int(False)

    probs = gsolver.test_image(image, rows, cols, starttime)
    return probs


def cal_force(row, col, weight_map, forcesectors):
    num_sectors = len(forcesectors)
    fxs = np.zeros(num_sectors)
    fys = np.zeros(num_sectors)
    for i, fs in enumerate(forcesectors):
        fxs[i], fys[i] = one_sector_force(row, col, weight_map, fs.wx, fs.wy)
    return fxs, fys


def one_sector_force(row, col, weight_map, sx, sy):
    h, w = weight_map.shape
    r, c = sx.shape
    half_r, half_c = r // 2, c // 2
    start_row = max(0, row - half_r)
    end_row = min(h, row + half_r + 1)
    start_col = max(0, col - half_c)
    end_col = min(w, col + half_c + 1)

    r1 = 0 if start_row > 0 else (half_r - row)
    c1 = 0 if start_col > 0 else (half_c - col)
    r2 = r if end_row < h else (h - start_row)
    c2 = c if end_col < w else (w - start_col)

    sub_wm = weight_map[start_row:end_row, start_col:end_col]
    wx = sub_wm * sx[r1:r2, c1:c2]
    wy = sub_wm * sy[r1:r2, c1:c2]
    fx = np.sum(wx)
    fy = np.sum(wy)
    # show_image(wx + wy, [])
    return fx, fy


def force_to_prob(fxs, fys):
    fxy = abs(fxs + fys)
    prob = fxy / (fxy.sum() + 1e-6)
    return prob


def get_gravity_prob(rows, cols, weight_map, field, blob, starttime):
    # show_image(weight_map,None)
    angles = np.arange(0, 360, 45)
    rs = np.random.randint(int(field * 0.9), int(field), size=8)
    if blob.angleid != -1 and starttime < 15:
        rs[blob.angleid] += 15
        rs[blob.angleid + 4] += 15

    forcesectors = [ForceSector(i, rs[i], angles[i]) for i in range(8)]
    forcesectors = update_sector_wxy(forcesectors, starttime)

    probs = np.zeros((8, len(rows)))
    for i, (row, col) in enumerate(zip(rows, cols)):
        fxs, fys = cal_force(row, col, weight_map, forcesectors)
        prob = force_to_prob(fxs, fys)
        probs[:, i] = prob

    # probs = adjust_probs_by_angle(probs, blob.angle)
    return probs


def create_distxy_weight(r, starttime):
    row, col = 2 * r + 1, 2 * r + 1
    center_row, center_col = r, r
    i, j = np.meshgrid(np.arange(row), np.arange(col), indexing='ij')
    y, x = i - center_row, j - center_col
    if starttime < 15:
        wx = np.ones_like(x, dtype=np.float64)
        wy = np.ones_like(y, dtype=np.float64)
    else:
        d = np.sqrt(x * x + y * y)
        w = np.maximum(0, 1 - d / 15)
        wx = np.maximum(0, 1 - np.abs(x) / max(row, col))
        wy = np.maximum(0, 1 - np.abs(y) / max(row, col))
        wx *= w
        wy *= w
    return wx, wy


class ForceSector:
    def __init__(self, id, field=20, angle=-1):
        self.id = id
        self.field = field
        self.angle = angle
        self.wx = None
        self.wy = None


def update_sector_wxy(forces, starttime):
    for i, force in enumerate(forces):
        r, angle = force.field, force.angle
        wx, wy = create_distxy_weight(r, starttime)
        sector = create_sectors(r, angle)
        forces[i].wx = wx * sector
        forces[i].wy = wy * sector
    return forces


def create_angle_matrix(r):  # Cartesian coordinate
    d = 2 * r + 1
    j, i = np.meshgrid(np.arange(d), np.arange(d))
    angle_matrix = np.degrees(np.arctan2(r - i, j - r))
    angle_matrix[angle_matrix < 0] += 360
    return angle_matrix


def create_sectors(r, angle, angle_gap=45):
    m = create_angle_matrix(r)
    half_angle = angle_gap / 2
    lower_bounds = (angle - half_angle)
    upper_bounds = (angle + half_angle)
    if lower_bounds > 0:
        sector = np.where((lower_bounds <= m) & (m <= upper_bounds), 1, 0)
    else:  # [-22.5，22.5]度的角度处理
        sector = np.where(((lower_bounds + 360) <= m) | (m <= upper_bounds), 1, 0)
    sector[r, r] = 0
    return sector


def show_image(image, ellipse):
    if image.dtype in [np.float32, np.float64]:
        image = contrast(image)
    colored_image = cv2.applyColorMap(image, cv2.COLORMAP_HOT)
    if ellipse:
        cv2.ellipse(colored_image, ellipse, (0, 255, 0), 2)
    cv2.imshow('Image with Ellipse', colored_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return


def save2pic(img_folder, tag, info, imgorg, walker_rows, walker_cols, walker_map, blob_weight, full_map):
    counter = getattr(save2pic, 'counter', 0)
    save2pic.counter = counter + 1

    img1 = imgorg
    img2 = contrast(walker_map, -1)
    img2 = cv2.applyColorMap(np.clip(img2, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
    img3 = blob_weight * 255
    img3 = cv2.cvtColor(img3.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    img4 = contrast(full_map, -1)
    img4 = cv2.applyColorMap(np.clip(img4, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)

    for y, x in zip(walker_rows, walker_cols):
        cv2.circle(img1, (x, y), 1, (255, 255, 255), -1)
        cv2.circle(img3, (x, y), 1, (0, 0, 255), -1)
        cv2.circle(img4, (x, y), 1, (0, 0, 255), -1)
    clr = (0, 255, 255)
    img1 = cv2.putText(img1, tag, (2, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, clr, 1, lineType=cv2.LINE_AA)
    img2 = cv2.putText(img2, info, (2, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, clr, 1, lineType=cv2.LINE_AA)
    img3 = cv2.putText(img3, info, (2, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, clr, 1, lineType=cv2.LINE_AA)
    img4 = cv2.putText(img4, info, (2, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, clr, 1, lineType=cv2.LINE_AA)
    file_name = f"tmp_%04d_{tag}_{info}.png" % counter
    img_name = os.path.join(img_folder, file_name)
    save2big(img_name, img1, img2, img3, img4)
    return


def save2big(img_name, img1, img2, img3, img4, gap=4):
    h, w, _ = img1.shape
    img_big = np.full((h * 2 + gap, w * 2 + gap, 3), 127)
    img_big[:h, :w] = img1
    img_big[:h, -w:] = img2
    img_big[-h:, :w] = img3
    img_big[-h:, -w:] = img4
    cv2.imwrite(img_name, img_big)
    return


def contrast(image, adjust_factor=1.0):
    image1 = image.copy()
    max_val, min_val = np.max(image1), np.min(image1)
    diff = max_val - min_val
    if adjust_factor == -1:
        image1 = np.clip(image1, 0, 255)
    elif diff != 0:
        image1 = ((image1 - min_val) * adjust_factor / diff) * 255
    else:
        image1 = image1 * 255
    return image1.astype(np.uint8)
