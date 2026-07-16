import cv2
import numpy as np
import torch
from scipy.signal import convolve2d
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .extract_skeleton import SkeletonWalker
from .loader_test import get_file_paths
from .trace_utils import get_observation, resolve_indices


def data_loader(args):
    data = generate_data(args)
    dataset = CurvedDataset(data)
    return DataLoader(dataset, batch_size=args.BATCH_SIZE, shuffle=True, num_workers=0)


def generate_data(args):
    data = []
    image_paths, label_paths = get_file_paths(args)
    idxs = resolve_indices(args.IMAGE_IDX, len(image_paths))
    image_paths = [image_paths[i] for i in idxs]
    label_paths = [label_paths[i] for i in idxs]
    total = len(image_paths)

    for image_path, label_path in tqdm(zip(image_paths, label_paths), total=total, desc="generate data"):
        image, label, skeleton = get_img_skel(image_path, label_path)
        seed_points = get_seedpoints(skeleton, args.POINTS_PER_IMAGE, args.ENDPOINT_ONLY)
        curve_mask = (label > 0).astype(np.float32)
        width_map = diameter_map(curve_mask)
        for seed in seed_points:
            slices = slice_trajectory(args, seed, image, skeleton, curve_mask, width_map)
            data.extend(slices)
    print(f"\n✅ generate data done | Total samples: {len(data)}")
    return data


def get_img_skel(image_path, label_path):
    _DILATE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    if image is None or label is None:
        return None, None, None

    img_h, img_w = image.shape
    image = image.astype(np.float32) * (1.0 / 255.0)

    label = cv2.resize(label, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
    label = np.where(label > 65, 255, 0).astype(np.uint8)

    label_dilate = cv2.dilate(label, _DILATE_KERNEL, iterations=1)
    skeleton = cv2.ximgproc.thinning(label_dilate, cv2.ximgproc.THINNING_GUOHALL)
    return image, label, (skeleton > 0).astype(np.float32)


def get_seedpoints(skeleton, max_points, endpoint_only=False):
    min_length = 60
    skel_u8 = (skeleton * 255).astype(np.uint8)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(skel_u8, connectivity=8)
    if n_labels <= 1:
        return np.empty((0, 2))

    areas = stats[1:, cv2.CC_STAT_AREA]
    large_labels = np.where(areas >= min_length)[0] + 1
    large_mask = np.isin(labels, large_labels)

    if endpoint_only:
        kernel = np.ones((3, 3), dtype=np.uint8)
        kernel[1, 1] = 0
        degree = convolve2d(skeleton, kernel, mode="same", boundary="fill")
        y, x = np.where((degree == 1) & large_mask)
    else:
        y, x = np.where(large_mask)
    seeds = np.column_stack([x, y]) if len(x) > 0 else np.empty((0, 2))

    if max_points != -1 and len(seeds) > max_points:
        seeds = seeds[np.random.choice(len(seeds), max_points, replace=False)]
    return seeds


def diameter_map(curve_mask):
    mask = np.uint8(curve_mask > 0)
    radius = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    w = 1.0 + 1.0 / (1.0 + np.exp(-4.0 * (radius - 2.0)))
    return (radius * w).astype(np.float32)


def slice_trajectory(args, seed, img, skeleton, curve_mask, width_map):
    patch_size = args.PATCH_SIZE
    pred_steps = args.PRED_STEPS
    use_width = getattr(args, "USE_WIDTH", False)
    samples = []

    traj = SkeletonWalker(skeleton, args.TRAIN_MAX_LEN).extract(seed)
    if len(traj) <= pred_steps:
        return samples

    end = len(traj) if use_width else len(traj) - pred_steps
    for i in range(1, end, 4):
        history = traj[max(0, i - 2 * pred_steps) : i]
        # history = traj[:i+1]
        target = traj[i : i + pred_steps]
        sample = get_observation(img, history, patch_size, curve_mask, width_map, target, args)
        samples.append(sample)

        # vis_history_target(img, skeleton, history, target, wait=True)

    return samples


class CurvedDataset(Dataset):
    KEYS = ("patch_obs", "traj_gt", "traj_gt_heatmaps", "patch_curve_mask")

    def __init__(self, data_list):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = {k: torch.as_tensor(self.data[idx][k], dtype=torch.float32) for k in self.KEYS}
        item["patch_obs"][0] = self._augment(item["patch_obs"][0])
        return item

    @staticmethod
    def _augment(x):
        if np.random.random() < 0.5:
            x = x * np.random.uniform(0.9, 1.1) + np.random.uniform(-0.1, 0.1) * x.mean()
        if np.random.random() < 0.3:
            x += torch.randn_like(x) * np.random.uniform(0.005, 0.02)
        return torch.clamp(x, -1.0, 1.0)
