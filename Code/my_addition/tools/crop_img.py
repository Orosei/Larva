import cv2
import os


def crop_and_save_images(image_paths, start_x, start_y, side_length, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"已创建保存目录: {save_dir}")

    for i, img_path in enumerate(image_paths):
        # 1. 使用 OpenCV 读取图片
        img = cv2.imread(img_path)

        # 检查图片是否成功读取
        if img is None:
            print(f"警告: 无法读取图片，请检查路径 -> {img_path}")
            continue

        # 获取原图的尺寸 (高度, 宽度)
        img_height, img_width = img.shape[:2]

        # 计算截取区域的右下角坐标
        end_x = start_x + side_length
        end_y = start_y + side_length

        # 边界安全检查：防止截取范围超出原图大小
        if end_x > img_width or end_y > img_height:
            print(f"警告: 截取区域超出图片 {os.path.basename(img_path)} 的边界。将截取到图片边缘。")
            # 限制在边界内
            end_x = min(end_x, img_width)
            end_y = min(end_y, img_height)

        # 2. 核心操作：使用 NumPy 切片进行截取
        # 注意：切片顺序是 [起始Y:结束Y, 起始X:结束X]
        cropped_img = img[start_y:end_y, start_x:end_x]

        # 3. 准备保存路径
        # 获取原文件名（不含路径和后缀）
        base_name = os.path.basename(img_path)
        file_name_without_ext = os.path.splitext(base_name)[0]

        # 构建新的保存路径，强制指定后缀为 .png
        new_file_name = f"{file_name_without_ext}_cropped_{i}.png"
        save_path = os.path.join(save_dir, new_file_name)

        # 4. 保存为 PNG 格式
        cv2.imwrite(save_path, cropped_img)
        print(f"成功截取并保存: {save_path}")


# ==========================================
# 使用示例
# ==========================================
if __name__ == "__main__":
    # 你的 4 张图片的路径列表
    my_images = [
        "D:\\Master\\Snake\\Snake_2026_01_26\\Dataset\\CFD\\test\\images\\101.jpg",
        "D:\\Master\\Snake\\Snake_2026_01_26\\Dataset\\CFD\\test\\labels\\101.png",
        "D:\\Master\\Snake\\Snake_2026_01_26\\Dataset\\CFD\\test\\coarse_seg\\unet\\101.png",
        "D:\\Master\\Snake\\Snake_2026_01_26\\Output\\CFD\\test\\unet\\marked_refine\\101.png"
    ]

    # 设置参数
    x_coordinate = 167  # 左上角 X 坐标
    y_coordinate = 168  # 左上角 Y 坐标
    square_size = 50  # 正方形边长
    output_folder = "D:\\Master\\Snake\\Snake_2026_01_26\\Code\\my_addition\\others\\images"  # 存储路径

    # 执行函数
    crop_and_save_images(
        image_paths=my_images,
        start_x=x_coordinate,
        start_y=y_coordinate,
        side_length=square_size,
        save_dir=output_folder
    )