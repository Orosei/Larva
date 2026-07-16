from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch

from ..curve_dst import search_curve_dst_v2
from ....model import get_model_weight_paths
from ....model.connectivity.v3_curve_tiny_unet import build_connectivity_predictor
from ....utils.get_curve_instance import process_curve
from ....utils.get_end_points import get_end_pts
from ....utils.get_skeleton_instance import process_skeleton
from ...setting import add_args


DEFAULT_PATCH_SIZE = 48
_PREDICTOR_CACHE = {}


class InvalidCurvePointError(ValueError):
    """Raised when a requested point does not lie on a reconstructed curve."""


@dataclass(frozen=True)
class ModelInputBundle:
    model_input: torch.Tensor
    curve_mask: np.ndarray
    center_curve_id: int
    target_curve_id: int
    point: Tuple[int, int]
    target_point: Tuple[int, int]


def _as_row_col(point: Tuple[int, int], name: str) -> Tuple[int, int]:
    if point is None or len(point) != 2:
        raise ValueError(f"{name} must be a (row, col) point, got {point!r}")
    return int(point[0]), int(point[1])


def _validate_point_in_image(point: Tuple[int, int], shape: Tuple[int, int], name: str) -> None:
    r, c = point
    h, w = shape
    if not (0 <= r < h and 0 <= c < w):
        raise ValueError(f"{name}={point} is outside image bounds {(h, w)}")


def _binary_255_mask(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    return (mask > 0).astype(np.uint8) * 255


def _ensure_rgb_image(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return np.repeat(image[:, :, None], 3, axis=2)
    if image.ndim != 3:
        raise ValueError(f"image must be HxW or HxWxC, got shape={image.shape}")
    if image.shape[2] == 1:
        return np.repeat(image, 3, axis=2)
    if image.shape[2] == 3:
        return image
    if image.shape[2] == 4:
        return image[:, :, :3]
    raise ValueError(f"image must have 1, 3, or 4 channels, got shape={image.shape}")



def _resize_point(point: Tuple[int, int], old_shape: Tuple[int, int], new_shape: Tuple[int, int]) -> Tuple[int, int]:
    old_h, old_w = old_shape
    new_h, new_w = new_shape
    r, c = point
    new_r = int(round(r * new_h / old_h))
    new_c = int(round(c * new_w / old_w))
    return min(max(new_r, 0), new_h - 1), min(max(new_c, 0), new_w - 1)


def _resize_inputs(
    image: np.ndarray,
    coarse_seg: np.ndarray,
    point: Tuple[int, int],
    target_point: Tuple[int, int],
    resize_size: Tuple[int, int] | None,
):
    if resize_size is None:
        return image, coarse_seg, point, target_point

    target_h, target_w = resize_size
    old_shape = image.shape[:2]
    resized_image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    resized_seg = cv2.resize(coarse_seg, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    return (
        resized_image,
        resized_seg,
        _resize_point(point, old_shape, (target_h, target_w)),
        _resize_point(target_point, old_shape, (target_h, target_w)),
    )


def _get_crop_params(center_point, _unused_point, patch_size):
    r, c = _as_row_col(center_point, "center_point")
    half = patch_size // 2
    r_start = r - half
    c_start = c - half
    r_end = r_start + patch_size
    c_end = c_start + patch_size
    return r, c, r_start, c_start, r_end, c_end, patch_size


def _crop_patch_core(image: np.ndarray, r_start: int, c_start: int, r_end: int, c_end: int, patch_size: int) -> np.ndarray:
    if image.ndim not in (2, 3):
        raise ValueError(f"image must be 2D or 3D for crop, got shape={image.shape}")

    patch_shape = (patch_size, patch_size) if image.ndim == 2 else (patch_size, patch_size, image.shape[2])
    patch = np.zeros(patch_shape, dtype=image.dtype)

    src_r_start = max(r_start, 0)
    src_c_start = max(c_start, 0)
    src_r_end = min(r_end, image.shape[0])
    src_c_end = min(c_end, image.shape[1])

    if src_r_start >= src_r_end or src_c_start >= src_c_end:
        return patch

    dst_r_start = src_r_start - r_start
    dst_c_start = src_c_start - c_start
    dst_r_end = dst_r_start + (src_r_end - src_r_start)
    dst_c_end = dst_c_start + (src_c_end - src_c_start)
    patch[dst_r_start:dst_r_end, dst_c_start:dst_c_end] = image[src_r_start:src_r_end, src_c_start:src_c_end]
    return patch


def crop_center_patch(image: np.ndarray, point: Tuple[int, int], patch_size: int):
    """以 point 为中心裁剪 patch，裁剪逻辑与 generate_points_dataset 的 _crop_center_patch 一致。"""
    _, _, r_start, c_start, r_end, c_end, _ = _get_crop_params(point, point, patch_size)
    return _crop_patch_core(image, r_start, c_start, r_end, c_end, patch_size)


def _to_chw_float(image: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float()


def _image_patch_to_tensor(image_patch: np.ndarray) -> torch.Tensor:
    image_patch = image_patch.astype(np.float32, copy=False)
    if image_patch.size and image_patch.max() > 1.0:
        image_patch = image_patch / 255.0
    return _to_chw_float(np.clip(image_patch, 0.0, 1.0))


def _curve_patch_to_tensor(curve_patch: np.ndarray) -> torch.Tensor:
    curve_patch = (curve_patch > 0).astype(np.float32)
    return torch.from_numpy(np.ascontiguousarray(curve_patch)).unsqueeze(0).float()


def rebuild_curve_mask(coarse_seg: np.ndarray) -> np.ndarray:
    source_mask = _binary_255_mask(coarse_seg)
    ske_num, ske_mask, _ = process_skeleton(source_mask)
    return process_curve(source_mask, ske_num, ske_mask).astype(np.int32)


def _build_model_input_bundle(
    image: np.ndarray,
    coarse_seg: np.ndarray,
    point: Tuple[int, int],
    target_point: Tuple[int, int],
    patch_size: int = DEFAULT_PATCH_SIZE,
    resize_size: Tuple[int, int] | None = None,
) -> ModelInputBundle:
    image = _ensure_rgb_image(np.asarray(image))
    coarse_seg = np.asarray(coarse_seg)
    if image.shape[:2] != coarse_seg.shape[:2]:
        raise ValueError(f"image shape {image.shape[:2]} and coarse_seg shape {coarse_seg.shape[:2]} do not match")

    point = _as_row_col(point, "point")
    target_point = _as_row_col(target_point, "target_point")
    _validate_point_in_image(point, image.shape[:2], "point")
    _validate_point_in_image(target_point, image.shape[:2], "target_point")

    image, coarse_seg, point, target_point = _resize_inputs(image, coarse_seg, point, target_point, resize_size)
    coarse_seg = _binary_255_mask(coarse_seg)
    curve_mask = rebuild_curve_mask(coarse_seg)

    center_curve_id = int(curve_mask[point])
    target_curve_id = int(curve_mask[target_point])
    if center_curve_id <= 0:
        raise InvalidCurvePointError(f"point={point} is not on a valid curve: center_curve_id={center_curve_id}")
    if target_curve_id <= 0:
        raise InvalidCurvePointError(f"target_point={target_point} is not on a valid curve: target_curve_id={target_curve_id}")

    image_patch = crop_center_patch(image, point, patch_size)
    center_curve_patch = crop_center_patch(curve_mask == center_curve_id, point, patch_size)
    target_curve_patch = crop_center_patch(curve_mask == target_curve_id, point, patch_size)

    model_input = torch.cat(
        [
            _image_patch_to_tensor(image_patch),
            _curve_patch_to_tensor(center_curve_patch),
            _curve_patch_to_tensor(target_curve_patch),
        ],
        dim=0,
    ).unsqueeze(0)

    return ModelInputBundle(
        model_input=model_input,
        curve_mask=curve_mask,
        center_curve_id=center_curve_id,
        target_curve_id=target_curve_id,
        point=point,
        target_point=target_point,
    )


def build_model_input(
    image: np.ndarray,
    coarse_seg: np.ndarray,
    point: Tuple[int, int],
    target_point: Tuple[int, int],
    patch_size: int = DEFAULT_PATCH_SIZE,
    resize_size: Tuple[int, int] | None = None,
) -> torch.Tensor:
    """Build (1, 5, 48, 48): RGB image + center_curve + target_curve."""
    return _build_model_input_bundle(image, coarse_seg, point, target_point, patch_size, resize_size).model_input


def _infer_dataset_name(dataset_name: str | None = None) -> str | None:
    if dataset_name:
        return str(dataset_name)

    for attr_name in ("dataset_name", "dataset"):
        value = getattr(add_args, attr_name, None)
        if value:
            return str(value)

    output_path = getattr(add_args, "output_path", None)
    if output_path:
        parts = Path(output_path).parts
        for idx, part in enumerate(parts):
            if part == "Output" and idx + 1 < len(parts):
                return parts[idx + 1]

    return None


def _resolve_model_weight_paths(
    dataset_name: str | None = None,
    model_name: str | None = None,
    model_path: str | Path | None = None,
    weight_path: str | Path | None = None,
):
    if model_path is not None and weight_path is not None:
        return Path(model_path), Path(weight_path)

    paths = get_model_weight_paths(_infer_dataset_name(dataset_name), model_name=model_name)
    return Path(model_path or paths.model_path), Path(weight_path or paths.weight_path)


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return device


def _get_predictor(model_path: str | Path, weight_path: str | Path, device: torch.device):
    cache_key = (str(Path(model_path).resolve()), str(Path(weight_path).resolve()), str(device))
    cached = _PREDICTOR_CACHE.get(cache_key)
    if cached is not None:
        return cached

    model = build_connectivity_predictor({}).to(device)
    checkpoint = torch.load(weight_path, map_location=device)
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"checkpoint does not contain 'model_state_dict': {weight_path}")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    _PREDICTOR_CACHE[cache_key] = model
    return model


def predict(
    image: np.ndarray,
    coarse_seg: np.ndarray,
    point: Tuple[int, int],
    target_point: Tuple[int, int],
    model_path: str | Path | None = None,
    weight_path: str | Path | None = None,
    dataset_name: str | None = None,
    model_name: str | None = None,
    threshold: float = 0.5,
    device: str | torch.device | None = None,
) -> bool:
    model_path, weight_path = _resolve_model_weight_paths(
        dataset_name=dataset_name,
        model_name=model_name,
        model_path=model_path,
        weight_path=weight_path,
    )
    bundle = _build_model_input_bundle(image, coarse_seg, point, target_point, DEFAULT_PATCH_SIZE, None)

    device = _resolve_device(device)
    model_input = bundle.model_input.to(device)

    model = _get_predictor(model_path, weight_path, device)

    with torch.no_grad():
        outputs = model(model_input)
        connect_prob = torch.sigmoid(outputs["logits"])

    connected = bool(connect_prob.reshape(-1)[0].item() >= threshold)
    print(connected)
    return connected


def snakes_info_v5(image, seg_map, curve_map, ske_map, junc_map, fov=[5, add_args.search_radius], dataset_name: str | None = None):
    row_offset = [0, -1, -1, -1, 0, 1, 1, 1]
    col_offset = [1, 1, 0, -1, -1, -1, 0, 1]

    binary_map = seg_map.astype(np.uint8)
    comp_num, comp_map, stats, centroids = cv2.connectedComponentsWithStats(binary_map, connectivity=8)
    main_comp_idx = 0

    if comp_num > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]

        if len(areas) > 0:
            max_idx = np.argmax(areas)
            main_comp_idx = max_idx + 1

    max_ske_idx = np.max(ske_map)
    info = []

    for idx in range(1, max_ske_idx + 1):
        mask = (ske_map == idx).astype(np.uint8)
        if np.sum(mask) < 2:
            continue

        end_pts = get_end_pts(mask)
        if len(end_pts) != 2:
            continue

        dist = np.linalg.norm(end_pts[0] - end_pts[1])
        radius = int(np.clip(5 + dist // 4, fov[0], fov[1]))
        for i in range(2):
            r, c = end_pts[i]

            flag = all(junc_map[r + dr, c + dc] == 0
                       for dr, dc in zip(row_offset, col_offset)
                       if 0 <= r + dr < ske_map.shape[0] and 0 <= c + dc < ske_map.shape[1])

            if flag:
                dst = search_curve_dst_v2(comp_map, main_comp_idx, curve_map, end_pts[i], end_pts[1 - i], radius)
                if dst is not None:
                    try:
                        connected = predict(image, seg_map, end_pts[i], dst, dataset_name=dataset_name)
                    except InvalidCurvePointError:
                        connected = False
                    if connected:
                        info.append([idx, tuple(end_pts[i][:2]), dst])
                        break

    return info
