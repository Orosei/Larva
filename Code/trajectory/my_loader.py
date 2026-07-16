"""Data loading helpers for DACE trajectory inference with coarse segmentation."""

import os
from pathlib import Path

import cv2
import numpy as np

from ..model.DACE.core.trace_utils import load_config as load_dace_config


VALID_EXTS = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".gif")


def load_my_config(config_path=None):
    """Load DACE options and resolve the additional coarse-seg directory."""
    if config_path is None:
        config_path = Path(__file__).resolve().parent / "my_test.json"

    config_path = Path(config_path).expanduser().resolve()
    args = load_dace_config(config_path)

    coarse_seg_dir = getattr(args, "COARSE_SEG_DIR", None)
    if coarse_seg_dir:
        coarse_seg_dir = Path(coarse_seg_dir).expanduser()
        if not coarse_seg_dir.is_absolute():
            coarse_seg_dir = config_path.parent / coarse_seg_dir
    else:
        coarse_seg_subdir = getattr(args, "COARSE_SEG_SUBDIR", "coarse_seg")
        coarse_seg_dir = Path(args.DATA_ROOT) / coarse_seg_subdir

    args.COARSE_SEG_DIR = os.path.normpath(os.fspath(coarse_seg_dir))
    return args


def _get_file_map(directory, data_name):
    """Return a case-insensitive stem-to-path map for one data directory."""
    if not directory or not os.path.isdir(directory):
        raise FileNotFoundError(f"{data_name} directory does not exist: {directory}")

    file_map = {}
    with os.scandir(directory) as entries:
        for entry in entries:
            if not entry.is_file() or not entry.name.lower().endswith(VALID_EXTS):
                continue
            stem = os.path.splitext(entry.name)[0].lower()
            if stem in file_map:
                raise ValueError(f"Duplicate {data_name} stem [{stem}] in: {directory}")
            file_map[stem] = entry.path
    return file_map


def get_my_file_paths(args):
    """Return strictly aligned image, label, and coarse-seg paths."""
    image_map = _get_file_map(args.IMAGE_DIR, "image")
    label_map = _get_file_map(args.LABEL_DIR, "label")
    coarse_seg_map = _get_file_map(args.COARSE_SEG_DIR, "coarse-seg")

    stems = sorted(image_map)
    missing_labels = [stem for stem in stems if stem not in label_map]
    missing_coarse_seg = [stem for stem in stems if stem not in coarse_seg_map]
    if missing_labels or missing_coarse_seg:
        messages = []
        if missing_labels:
            messages.append(f"missing labels: {', '.join(missing_labels)}")
        if missing_coarse_seg:
            messages.append(f"missing coarse-seg: {', '.join(missing_coarse_seg)}")
        raise FileNotFoundError("; ".join(messages))

    image_paths = [image_map[stem] for stem in stems]
    label_paths = [label_map[stem] for stem in stems]
    coarse_seg_paths = [coarse_seg_map[stem] for stem in stems]
    return image_paths, label_paths, coarse_seg_paths


def _load_gray(path, data_name):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"{data_name} not found or unreadable: {path}")
    return image


def load_my_image(image_path, label_path, coarse_seg_path):
    """Load one aligned image/label/coarse-seg sample."""
    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Image not found or unreadable: {image_path}")

    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    image = image_gray.astype(np.float32) / 255.0
    label = (_load_gray(label_path, "Label") > 0).astype(np.float32)
    coarse_seg = (_load_gray(coarse_seg_path, "Coarse segmentation") > 0).astype(np.uint8)

    expected_shape = image.shape
    if label.shape != expected_shape or coarse_seg.shape != expected_shape:
        raise ValueError(
            "Image, label, and coarse-seg shapes must match: "
            f"image={expected_shape}, label={label.shape}, coarse_seg={coarse_seg.shape}"
        )

    return image, label, coarse_seg, image_bgr
