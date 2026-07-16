import os
import shutil


def copy_top_k_images(input_dir, output_dir, k):
    if not os.path.exists(input_dir):
        print(f"错误：找不到输入文件夹 '{input_dir}'")
        return False

    os.makedirs(output_dir, exist_ok=True)

    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp')

    image_files = [
        f for f in os.listdir(input_dir)
        if f.lower().endswith(valid_extensions) and os.path.isfile(os.path.join(input_dir, f))
    ]

    if not image_files:
        print(f"在 '{input_dir}' 中没有找到任何支持的图片文件。")
        return False

    image_files.sort()
    top_k_files = image_files[:k]

    print(f"共找到 {len(image_files)} 张图片，准备复制前 {len(top_k_files)} 张...")

    success_count = 0
    for filename in top_k_files:
        src_path = os.path.join(input_dir, filename)
        dst_path = os.path.join(output_dir, filename)

        try:
            shutil.copy2(src_path, dst_path)
            success_count += 1
        except Exception as e:
            print(f"复制文件 '{filename}' 时发生错误: {e}")

    print(f"✅ 处理完成！成功复制 {success_count} 张图片到 '{output_dir}'")
    return True


if __name__ == "__main__":
    SOURCE_FOLDER = r"D:\Master\Snake\Snake_2026_01_26\Dataset\CHN6-CUG\test\images"
    DEST_FOLDER = r"D:\Master\Snake\Snake_2026_01_26\Dataset\CHN6-CUG_top200\test\images"
    K_VALUE = 200

    copy_top_k_images(
        input_dir=SOURCE_FOLDER,
        output_dir=DEST_FOLDER,
        k=K_VALUE
    )