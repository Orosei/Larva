import cv2
import matplotlib.pyplot as plt
import numpy as np

from .image_draw import show_image


_JUNC_KERNELS = None
_JUNC_DILATE_KERNEL = np.ones((3, 3), np.uint8)


def process_skeleton(bw_mask):
    bw_map = (bw_mask > 0).astype(np.uint8) * 255
    ske_map, junc_mask = get_skeleton_instance(bw_map)
    num, ske_map = reorder_labels(ske_map)
    ske_map, junc_mask = enlarge_junc(ske_map, junc_mask)
    num, ske_map = reorder_labels(ske_map)
    return num, ske_map, junc_mask


def reorder_labels(bw_map):
    if bw_map.dtype != np.uint8:
        bw_map = (bw_map > 0).astype(np.uint8)  # 生成新数组，不影响外部

    num, label_map, stats, _ = cv2.connectedComponentsWithStats(bw_map, connectivity=8)
    if num == 1:
        return num, label_map

    areas = stats[1:, 4]
    sorted_indices = np.argsort(areas)
    label_remap = np.zeros(num, np.int32)
    label_remap[1 + sorted_indices] = np.arange(1, num)  # 直接映射新标签
    return num, label_remap[label_map]


def draw_kernels(kernels, cols=4):
    N = len(kernels)
    rows = (N + cols - 1) // cols  # 自动计算行数
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
    axes = axes.flat
    for i in range(N):
        axes[i].imshow(kernels[i], cmap="gray", vmin=0, vmax=1)
        axes[i].set_title(f"K{i+1}")
        axes[i].axis("off")
    plt.tight_layout()
    plt.show(block=True)
    input("按 Enter 继续...")


def get_junc_kernels():
    # 3点方向，逆时针转
    rows = np.array([1, 0, 0, 0, 1, 2, 2, 2])
    cols = np.array([2, 2, 1, 0, 0, 0, 1, 2])
    kernels = np.ones((16, 3, 3), dtype=np.float32)

    x = np.array([1, 0, 1, 0, 1, 0, 0, 0])
    for i in range(8):
        kernels[i, rows, cols] = x
        x = np.roll(x, -1)

    x = np.array([1, 0, 1, 0, 0, 1, 0, 0])
    for i in range(8, 16):
        kernels[i, rows, cols] = x
        x = np.roll(x, -1)

    # draw_kernels(kernels)
    return kernels


def get_junc_kernels_cached():
    global _JUNC_KERNELS
    if _JUNC_KERNELS is None:
        _JUNC_KERNELS = get_junc_kernels()
    return _JUNC_KERNELS


def find_junc(bw_map):
    bw_mask = (bw_map > 0).astype(np.float32)
    junc_mask = np.zeros_like(bw_map, dtype=np.uint8)

    for k in get_junc_kernels_cached():
        conv = cv2.filter2D(bw_mask, -1, k)
        junc_mask[conv == 4] = 1

    junc_mask = cv2.dilate(junc_mask, _JUNC_DILATE_KERNEL, iterations=1)
    junc_mask = junc_mask & bw_map
    return junc_mask


def get_skeleton_instance(bw_map, zero_border=True, border_size=2):
    ske_mask = cv2.ximgproc.thinning(bw_map, cv2.ximgproc.THINNING_ZHANGSUEN)
    if zero_border:
        ske_mask[:border_size, :] = ske_mask[-border_size:, :] = ske_mask[:, :border_size] = ske_mask[:, -border_size:] = 0

    junc_mask = find_junc(ske_mask)
    ske_mask[junc_mask > 0] = 0
    return ske_mask, junc_mask


def draw_diffs(b_ske, a_ske, b_junc, a_junc):
    img = cv2.cvtColor((b_ske > 0).astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
    old_junc = b_junc > 0
    new_junc = ~(b_junc > 0) & (a_junc > 0)
    img[old_junc] = [0, 255, 0]
    img[new_junc] = [0, 0, 255]
    show_image(img)


def enlarge_junc(label_map, junc_mask, max_area=4):
    row_offset = [0, -1, 0, 1, -1, -1, 1, 1]
    col_offset = [1, 0, -1, 0, 1, -1, -1, 1]
    new_junc_mask = junc_mask.copy()

    labels = np.unique(label_map)
    labels = labels[labels != 0]
    areas = np.array([np.sum(label_map == l) for l in labels])
    small_labels = labels[np.where(areas < max_area)]
    for label in small_labels:
        y, x = np.where(label_map == label)
        hits = np.zeros(len(y), dtype=bool)
        for i in range(len(y)):
            nr = y[i] + row_offset
            nc = x[i] + col_offset
            valid = (nr >= 0) & (nr < label_map.shape[0]) & (nc >= 0) & (nc < label_map.shape[1])
            if np.any(new_junc_mask[nr[valid], nc[valid]]):
                hits[i] = True
        new_junc_mask[y[hits], x[hits]] = 1

    new_label_map = label_map.copy()
    new_label_map[new_junc_mask] = 0
    # draw_diffs(label_map, new_label_map, junc_mask, new_junc_mask)
    return new_label_map, new_junc_mask
