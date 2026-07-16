import cv2
import numpy as np
import os


def overlay_red_strict_bgr(img1_path, img2_path, save_path):
    # 1. 读取两张图片
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    if img1 is None:
        print(f"错误: 无法读取第一张图片，请检查路径 -> {img1_path}")
        return
    if img2 is None:
        print(f"错误: 无法读取第二张图片，请检查路径 -> {img2_path}")
        return

    # 2. 尺寸对齐（必需：确保两张图尺寸一致才能一一对应叠加）
    if img1.shape != img2.shape:
        print(f"提示: 两张图片尺寸不同。正在将图1调整为图2的尺寸...")
        img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]))

    # 3. 核心：创建遮罩 (Mask) - 严格基于 R通道=255
    # OpenCV 的通道顺序是 B(0), G(1), R(2)
    # img1[:,:,2] 获取的是整张图的红通道矩阵
    # (img1[:,:,2] == 255) 生成一个布尔矩阵 (True/False)

    red_condition = (img1[:, :, 0] == 0) & (img1[:, :, 1] == 0) & (img1[:, :, 2] == 255)

    # red_mask = np.where(red_condition)

    final_img = img2.copy()
    final_img[red_condition, 0] = 0
    final_img[red_condition, 1] = 0
    final_img[red_condition, 2] = 255


    # 8. 保存图片
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    cv2.imwrite(save_path, final_img)
    print(f"处理完成！成功叠加并保存至: {save_path}")


# ==========================================
# 使用示例
# ==========================================
if __name__ == "__main__":
    # 使用你自己的图片路径
    source_image = "D:\\Master\\Snake\\Snake_2026_01_26\\Code\\my_addition\\others\\images\\101_cropped_3.png"  # 图1：带有需要提取的“R=255”区域
    background_image = "D:\\Master\\Snake\\Snake_2026_01_26\\Code\\my_addition\\others\\images\\101_cropped_0.png"  # 图2：作为背景
    output_image = "D:\\Master\\Snake\\Snake_2026_01_26\\Code\\my_addition\\others\\images\\101_cropped_0_1.png"

    # 调用函数
    overlay_red_strict_bgr(source_image, background_image, output_image)