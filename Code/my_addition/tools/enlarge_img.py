import cv2
import os
import numpy as np


def enlarge_by_position(img_path, scale_factor):
    if not isinstance(scale_factor, int) or scale_factor < 1:
        raise ValueError("scale_factor 必须是大于等于 1 的整数")

    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")

    enlarged_img = np.repeat(np.repeat(img, scale_factor, axis=0), scale_factor, axis=1)

    dir_name, file_name = os.path.split(img_path)
    base_name, ext = os.path.splitext(file_name)
    new_file_name = f"{base_name}_x{scale_factor}_pos{ext}"
    new_img_path = os.path.join(dir_name, new_file_name)

    cv2.imwrite(new_img_path, enlarged_img)

    return


if __name__ == "__main__":
    test_img_path = "D:\\Master\\Snake\\Snake_2026_01_26\\Code\\my_addition\\others\\images\\101_cropped_0_1.png"
    scale = 5
    enlarge_by_position(test_img_path, scale)
