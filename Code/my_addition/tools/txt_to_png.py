import cv2
import numpy as np
import os
import glob


def batch_convert_txt_to_png(input_dir, output_dir, width, height):
    if not os.path.isdir(input_dir):
        print(f"错误：找不到输入文件夹 '{input_dir}'")
        return False

    # 2. 如果输出文件夹不存在，则自动创建
    os.makedirs(output_dir, exist_ok=True)

    # 3. 获取所有 txt 文件
    txt_files = glob.glob(os.path.join(input_dir, "*.txt"))

    if not txt_files:
        print(f"在 '{input_dir}' 中没有找到任何 .txt 文件。")
        return False

    print(f"找到 {len(txt_files)} 个 .txt 文件，开始处理...")

    # 4. 遍历处理每个文件
    success_count = 0
    for txt_path in txt_files:
        basename = os.path.splitext(os.path.basename(txt_path))[0]
        output_path = os.path.join(output_dir, f"{basename}.png")

        # 【核心修改 1】创建单通道纯黑画布 (shape 为 H x W，去掉了第三个维度)
        img = np.zeros((height, width), dtype=np.uint8)

        with open(txt_path, 'r') as file:
            lines = file.readlines()

        for line in lines:
            data = line.strip().split()
            if len(data) < 3:
                continue

            coords = [float(x) for x in data[1:]]

            points = []
            for i in range(0, len(coords), 2):
                x = int(coords[i] * width)
                y = int(coords[i + 1] * height)
                points.append([x, y])

            pts = np.array(points, np.int32).reshape((-1, 1, 2))

            # 【核心修改 2】颜色固定为 255 (纯白)
            cv2.fillPoly(img, [pts], color=255)

        # 保存图像
        cv2.imwrite(output_path, img)
        success_count += 1

    print(f"✅ 处理完成！成功生成 {success_count} 张 PNG 图像。输出目录: {output_dir}")
    return True


if __name__ == "__main__":
    # 直接在这里传入你的参数
    SOURCE_FOLDER = r"D:\Master\Snake\Snake_2026_01_26\Dataset\crack-seg\labels\test"
    DEST_FOLDER = r"D:\Master\Snake\Snake_2026_01_26\Dataset\crack-seg\test\labels"
    IMG_W = 416
    IMG_H = 416

    # 执行函数
    batch_convert_txt_to_png(
        input_dir=SOURCE_FOLDER,
        output_dir=DEST_FOLDER,
        width=IMG_W,
        height=IMG_H
    )