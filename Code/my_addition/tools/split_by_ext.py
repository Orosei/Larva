import os
import shutil


def sort_images_by_type(source_folder, jpg_folder, png_folder):
    os.makedirs(jpg_folder, exist_ok=True)
    os.makedirs(png_folder, exist_ok=True)

    jpg_count = 0
    png_count = 0

    for filename in os.listdir(source_folder):
        file_path = os.path.join(source_folder, filename)

        if os.path.isfile(file_path):
            _, ext = os.path.splitext(filename)
            ext = ext.lower()

            if ext in ['.jpg', '.jpeg']:
                target_path = os.path.join(jpg_folder, filename)
                shutil.move(file_path, target_path)
                # 如果你想保留原文件（复制而不是移动），请将上一行替换为: shutil.copy2(file_path, target_path)
                jpg_count += 1

            elif ext == '.png':
                target_path = os.path.join(png_folder, filename)
                shutil.move(file_path, target_path)
                # 同样，如果要复制而不是移动，使用 shutil.copy2
                png_count += 1

    print(f"整理完成！共处理了 {jpg_count} 个 JPG 文件 和 {png_count} 个 PNG 文件。")


if __name__ == "__main__":
    SRC_DIR = r"D:\Master\Snake\Snake_2026_01_26\Dataset\archive\train"
    JPG_DIR = r"D:\Master\Snake\Snake_2026_01_26\Dataset\DeepGlobe2018\train\images"
    PNG_DIR = r"D:\Master\Snake\Snake_2026_01_26\Dataset\DeepGlobe2018\train\labels"

    sort_images_by_type(SRC_DIR, JPG_DIR, PNG_DIR)