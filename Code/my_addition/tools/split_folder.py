import os
import shutil


def split_images(src_folder, dest_folder_80, dest_folder_20):
    os.makedirs(dest_folder_80, exist_ok=True)
    os.makedirs(dest_folder_20, exist_ok=True)

    valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp'}

    files = [f for f in os.listdir(src_folder) if os.path.splitext(f)[1].lower() in valid_extensions]
    files.sort()

    total_files = len(files)
    if total_files == 0:
        print("源文件夹中没有找到任何图片，请检查路径。")
        return

    split_index = int(total_files * 0.8)

    files_80 = files[:split_index]
    files_20 = files[split_index:]

    print(f"总计找到 {total_files} 张图片。")

    print(f"正在将前 80% ({len(files_80)} 张) 复制到：{dest_folder_80} ...")
    for f in files_80:
        src_path = os.path.join(src_folder, f)
        dest_path = os.path.join(dest_folder_80, f)
        shutil.copy2(src_path, dest_path)

    print(f"正在将后 20% ({len(files_20)} 张) 复制到：{dest_folder_20} ...")
    for f in files_20:
        src_path = os.path.join(src_folder, f)
        dest_path = os.path.join(dest_folder_20, f)
        shutil.copy2(src_path, dest_path)

    print("✅ 排序与分配操作已全部完成！")


if __name__ == "__main__":
    SOURCE_DIR = r"D:\Master\Snake\Snake_2026_01_26\Dataset\CrackLS315_ori\CrackLS315_gt"
    DIR_80 = r"D:\Master\Snake\Snake_2026_01_26\Dataset\CrackLS315\train\labels"
    DIR_20 = r"D:\Master\Snake\Snake_2026_01_26\Dataset\CrackLS315\test\labels"

    split_images(SOURCE_DIR, DIR_80, DIR_20)