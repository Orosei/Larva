import os
import cv2


def create_mask_union(mask1_dir, mask2_dir, output_dir):
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

        union_mask = cv2.bitwise_or(mask1_bin, mask2_bin)
        output_filename = f"{basename}.png"
        output_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(output_path, union_mask)
        print(f"[+] 成功处理并保存并集: {output_filename}")


if __name__ == "__main__":
    BASE_DATASET_DIR = r"D:\Master\Snake\Snake_2026_01_26\Dataset"
    BASE_OUTPUT_DIR = r"D:\Master\Snake\Snake_2026_01_26\Code\my_addition\others\images"

    dataset_name = "CFD"
    phase = "test"
    refine = "crwcr"
    backbone = "IterNet"

    MASK1_DIRECTORY = os.path.join(BASE_DATASET_DIR, dataset_name, phase, "coarse_seg", backbone)
    MASK2_DIRECTORY = os.path.join(BASE_OUTPUT_DIR, dataset_name, phase, refine, backbone)
    OUTPUT_DIRECTORY = os.path.join(BASE_OUTPUT_DIR, dataset_name, phase, "combine", backbone)

    print(f"[*] 输入掩膜 1 (MASK1): {MASK1_DIRECTORY}")
    print(f"[*] 输入掩膜 2 (MASK2): {MASK2_DIRECTORY}")
    print(f"[*] 输出文件夹 (OUT):   {OUTPUT_DIRECTORY}")

    create_mask_union(MASK1_DIRECTORY, MASK2_DIRECTORY, OUTPUT_DIRECTORY)
