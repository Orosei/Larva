import os

import cv2


def init_loaders(paths, color_flags):
    loaders = [ImageLoader(p, is_color=f) for p, f in zip(paths, color_flags)]
    return loaders if check_loaders(loaders) else [None] * 4


def check_loaders(loaders):
    R, G, N = "\033[91m", "\033[92m", "\033[0m"
    name_sets = [set(os.path.splitext(os.path.basename(p))[0] for p in l.img_paths) for l in loaders]
    common = set.intersection(*name_sets)
    if all(s == common for s in name_sets):
        print(f"{G}✅ All loader filenames match!{N}")
        return True
    print(f"{R}⚠️ WARNING: Loader filenames do not match!{N}")
    for i, s in enumerate(name_sets):
        print(f"{R} Loader {i} extra files: {', '.join(sorted(s - common))}{N}")
    return False


def get_img_paths(dir_path):
    valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".gif", ".tiff")
    if not os.path.exists(dir_path):
        raise ValueError("Directory does not exist")
    return sorted(os.path.join(root, file) for root, _, files in os.walk(dir_path) for file in files if file.lower().endswith(valid_exts))


def load_image(path, is_color, to_rgb):
    if not os.path.exists(path):
        print(f"Warning: Image path {path} does not exist.")
        return None
    flag = cv2.IMREAD_COLOR if is_color else cv2.IMREAD_GRAYSCALE
    img = cv2.imread(path, flag)
    if img is None:
        print(f"Warning: Failed to read image {path}.")
        return None
    if is_color and to_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


class ImageLoader:
    def __init__(self, dir_path, is_color=False, to_rgb=False):
        self.img_paths = get_img_paths(dir_path)
        self.is_color = is_color
        self.to_rgb = to_rgb

    def get_count(self):
        return len(self.img_paths)

    def get_path(self, idx):
        if not (0 <= idx < self.get_count()):
            raise IndexError(f"Index {idx} out of range. Max index is {self.get_count() - 1}.")
        return self.img_paths[idx]

    def get_image(self, idx):
        return load_image(self.get_path(idx), self.is_color, self.to_rgb)
