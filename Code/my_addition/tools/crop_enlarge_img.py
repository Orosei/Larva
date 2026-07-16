import cv2
import numpy as np
import os


def generate_patched_image(image_path, coords, side_length, zoom_factor, border_thickness):
    """
    输入图片路径，坐标(y,x)，正方形边长，整数放大倍数，边框像素宽度。
    使用 OpenCV 读取和保存图像。
    放大时不使用插值（直接重复像素）。
    完全通过 NumPy 矩阵切片绘制边框，不使用 OpenCV 的绘图功能，不使用 PIL。
    """
    # 确保放大倍数为整数
    zoom_factor = int(zoom_factor)

    # 1. 使用 OpenCV 读取图像 (注意：此处读取为 BGR 格式)
    input_image = cv2.imread(image_path)
    if input_image is None:
        print(f"❌ 错误: 无法读取图像，请检查路径: {image_path}")
        return

    img_h, img_w, _ = input_image.shape

    # 定义高对比度颜色列表 (恢复为 OpenCV 的 BGR 格式！)
    color_palette = [
        (0, 0, 255),  # 红色
        (0, 255, 0),  # 绿色
        (255, 0, 0),  # 蓝色
        (0, 255, 255),  # 黄色
        (255, 0, 255),  # 品红
        (255, 255, 0),  # 青色
        (0, 165, 255),  # 橙色
        (128, 0, 128),  # 紫色
        (255, 255, 255),  # 白色
        (100, 100, 100)  # 灰色
    ]

    # 2. 计算尺寸并创建新画布
    scaled_patch_dim = side_length * zoom_factor
    max_patches_count = img_w // scaled_patch_dim
    num_coords_to_process = min(len(coords), max_patches_count)

    if num_coords_to_process == 0:
        print("⚠️ 警告: 提供的坐标为空，或图片宽度不足以容纳任何放大后的Patch。")

    # 创建新的画布矩阵
    canvas_h = img_h + scaled_patch_dim
    canvas = np.zeros((canvas_h, img_w, 3), dtype=np.uint8)

    # 将原图像素复制到画布的顶部
    canvas[0:img_h, 0:img_w] = input_image.copy()

    # ==========================================
    # 核心：纯 NumPy 实现的边框绘制函数
    # ==========================================
    def draw_border_numpy(target_array, y, x, h, w, color, thickness):
        """通过矩阵切片直接修改边缘像素，实现内边框绘制"""
        # 顶边 (Top): 修改高度为 thickness 的顶部区域
        target_array[y: y + thickness, x: x + w] = color

        # 底边 (Bottom): 修改高度为 thickness 的底部区域
        target_array[y + h - thickness: y + h, x: x + w] = color

        # 左边 (Left): 修改宽度为 thickness 的左侧区域
        target_array[y: y + h, x: x + thickness] = color

        # 右边 (Right): 修改宽度为 thickness 的右侧区域
        target_array[y: y + h, x + w - thickness: x + w] = color

    valid_patches_processed = 0

    # 3. 处理每个 Patch
    for i in range(num_coords_to_process):
        original_y, original_x = coords[i]

        # 边界保护，确保裁剪不越界
        x_start = max(0, min(original_x, img_w - side_length))
        y_start = max(0, min(original_y, img_h - side_length))

        # --- 3.a 裁剪 ---
        patch_cropped = input_image[y_start: y_start + side_length, x_start: x_start + side_length]

        if patch_cropped.shape[0] != side_length or patch_cropped.shape[1] != side_length:
            continue

        # --- 3.b 放大 (严格使用 NumPy 重复像素，无插值) ---
        patch_scaled = np.repeat(np.repeat(patch_cropped, zoom_factor, axis=0), zoom_factor, axis=1)

        # --- 3.c 放置到画布底部 ---
        patch_canvas_x_start = i * scaled_patch_dim
        canvas[
            img_h: img_h + scaled_patch_dim, patch_canvas_x_start: patch_canvas_x_start + scaled_patch_dim] = patch_scaled

        # --- 3.d 绘制边框 (使用纯 NumPy 矩阵切片) ---
        current_color = color_palette[i % len(color_palette)]

        # 在原图位置画框
        draw_border_numpy(
            canvas,
            y_start, x_start,
            side_length, side_length,
            current_color, border_thickness
        )

        # 在放大后的 Patch 位置画框
        draw_border_numpy(
            canvas,
            img_h, patch_canvas_x_start,
            scaled_patch_dim, scaled_patch_dim,
            current_color, border_thickness
        )

        valid_patches_processed += 1

    # 4. 使用 OpenCV 保存生成的图像
    file_dir = os.path.dirname(image_path)
    file_name = os.path.basename(image_path)
    name_part, ext_part = os.path.splitext(file_name)

    save_filename = f"{name_part}_patched{ext_part}"
    save_path = os.path.join(file_dir, save_filename)

    cv2.imwrite(save_path, canvas)
    print(f"✅ 处理完成！成功处理了 {valid_patches_processed} 个 Patch，结果已保存至: {save_path}")


# ================= 运行测试示例 =================
if __name__ == "__main__":
    test_image_path = r"D:\Master\Snake\Snake_2026_01_26\Code\my_addition\others\images\2026_04_26\097_label.png"

    test_coords = [
        (221, 0),
        (175, 89),
        (162, 165),
    ]
    test_side_length = 40
    test_zoom_factor = 4

    generate_patched_image(
        test_image_path,
        test_coords,
        test_side_length,
        test_zoom_factor,
        border_thickness=1
    )