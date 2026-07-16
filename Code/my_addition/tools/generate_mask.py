import cv2
import numpy as np
import os


def generate_masks_contours(src_folder, dst_folder, mode='full'):
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)
        print(f"[Info] 创建输出目录: {dst_folder}")

    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    count = 0

    for filename in os.listdir(src_folder):
        name, ext = os.path.splitext(filename)
        if ext.lower() not in valid_exts:
            continue

        src_path = os.path.join(src_folder, filename)
        img = cv2.imread(src_path)
        if img is None: continue

        h, w = img.shape[:2]
        final_mask = None

        if mode == 'full':
            final_mask = np.full((h, w), 255, dtype=np.uint8)
        elif mode == 'center':
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img

            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            mask_canvas = np.zeros((h, w), dtype=np.uint8)

            if contours:
                max_cnt = max(contours, key=cv2.contourArea)
                cv2.drawContours(mask_canvas, [max_cnt], -1, 255, thickness=-1)

            final_mask = mask_canvas

        # 保存
        dst_filename = f"{name}.png"
        cv2.imwrite(os.path.join(dst_folder, dst_filename), final_mask)
        count += 1

    print(f"--- 处理完成 ---")
    print(f"模式: {mode} (Contours)")
    print(f"共生成 {count} 张 Mask")


if __name__ == "__main__":
    input_dir = r"D:\Master\Snake\Snake_2026_01_26\Dataset\DRIVE\test\images"
    output_dir = r"D:\Master\Snake\Snake_2026_01_26\Dataset\DRIVE\test\masks"
    generate_masks_contours(input_dir, output_dir, mode='center')
