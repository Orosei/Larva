import os
import cv2
import numpy as np


def process_masks_v4(mask_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(mask_dir):
        mask_path = os.path.join(mask_dir, filename)

        if not os.path.isfile(mask_path):
            continue

        basename = os.path.splitext(filename)[0]

        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if mask_img is None:
            print(f"[-] 警告: 无法读取图像 '{filename}'，已跳过。")
            continue

        _, mask_bin = cv2.threshold(mask_img, 127, 255, cv2.THRESH_BINARY)
        skeleton = cv2.ximgproc.thinning(mask_bin, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated_mask_4iter = cv2.dilate(mask_bin, kernel, iterations=4)
        eroded_mask = cv2.erode(dilated_mask_4iter, kernel, iterations=1)
        one_pixel_inner_edge = cv2.subtract(dilated_mask_4iter, eroded_mask)

        new_img = np.zeros_like(mask_bin, dtype=np.uint8)
        new_img[skeleton == 255] = 255
        new_img[one_pixel_inner_edge == 255] = 127

        output_filename = f"{basename}.png"
        output_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(output_path, new_img)
        print(f"[+] 成功处理并保存: {output_filename}")


def generate_seeds(mask1_dir, mask2_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    mask1_files_map = {}
    for f in os.listdir(mask1_dir):
        if os.path.isfile(os.path.join(mask1_dir, f)):
            basename = os.path.splitext(f)[0]
            mask1_files_map[basename] = f

    for mask2_filename in os.listdir(mask2_dir):
        mask2_path = os.path.join(mask2_dir, mask2_filename)

        if not os.path.isfile(mask2_path):
            continue

        basename = os.path.splitext(mask2_filename)[0]
        if basename not in mask1_files_map:
            print(f"[-] 警告: 在 mask1 文件夹中找不到与 '{mask2_filename}' 匹配的文件，已跳过。")
            continue

        mask1_filename = mask1_files_map[basename]
        mask1_path = os.path.join(mask1_dir, mask1_filename)

        mask1_img = cv2.imread(mask1_path, cv2.IMREAD_GRAYSCALE)
        mask2_img = cv2.imread(mask2_path, cv2.IMREAD_GRAYSCALE)

        if mask1_img is None or mask2_img is None:
            print(f"[-] 错误: 无法读取 {basename} 的图像数据，已跳过。")
            continue

        _, mask1_bin = cv2.threshold(mask1_img, 127, 255, cv2.THRESH_BINARY)
        _, mask2_bin = cv2.threshold(mask2_img, 127, 255, cv2.THRESH_BINARY)
        fg_seeds = cv2.bitwise_and(mask1_bin, mask2_bin)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated_mask2 = cv2.dilate(mask2_bin, kernel, iterations=1)
        bg_seeds = cv2.subtract(dilated_mask2, mask2_bin)

        new_img = np.zeros_like(mask1_bin)
        new_img[fg_seeds == 255] = 255
        new_img[bg_seeds == 255] = 127

        output_filename = f"{basename}.png"
        output_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(output_path, new_img)
        print(f"[+] 成功生成种子图: {output_filename}")


if __name__ == "__main__":
    BASE_DATASET_DIR = r"D:\Master\Snake\Snake_2026_01_26\Dataset"
    BASE_OUTPUT_DIR = r"D:\Master\Snake\Snake_2026_01_26\Code\my_addition\others\images"

    dataset_name = "CFD"
    phase = "test"
    backbone = "IterNet"

    SEG_DIRECTORY = os.path.join(BASE_DATASET_DIR, dataset_name, phase, "coarse_seg", backbone)
    LABEL_DIRECTORY = os.path.join(BASE_DATASET_DIR, dataset_name, phase, "labels")
    OUTPUT_DIRECTORY = os.path.join(BASE_OUTPUT_DIR, dataset_name, phase, "seeds", backbone)

    print(f"[*] 输入文件夹 (SEG): {SEG_DIRECTORY}")
    print(f"[*] 输入文件夹 (SEG): {LABEL_DIRECTORY}")
    print(f"[*] 输出文件夹 (OUT): {OUTPUT_DIRECTORY}")

    # process_masks_v4(SEG_DIRECTORY, OUTPUT_DIRECTORY)
    generate_seeds(SEG_DIRECTORY, LABEL_DIRECTORY, OUTPUT_DIRECTORY)
