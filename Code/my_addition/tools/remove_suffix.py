import os
import shutil
from pathlib import Path


def process_files(src_dir, dest_dir, suffix_to_remove):
    """
    处理指定文件夹中的文件，去除文件名的指定后缀并复制到新文件夹
    """
    src_path = Path(src_dir)
    dest_path = Path(dest_dir)

    # 检查源文件夹是否存在
    if not src_path.exists() or not src_path.is_dir():
        print(f"❌ 错误: 源文件夹 '{src_dir}' 不存在！")
        return

    # 确保目标文件夹存在，如果不存在则自动创建
    dest_path.mkdir(parents=True, exist_ok=True)

    count = 0
    for file_path in src_path.iterdir():
        if file_path.is_file():
            # file_path.stem 获取不含扩展名的文件名 (例如 "report_v1")
            # file_path.suffix 获取扩展名 (例如 ".pdf")
            stem = file_path.stem
            ext = file_path.suffix

            # 检查文件名是否以你指定的后缀结尾
            if stem.endswith(suffix_to_remove):
                # 切片去除后缀
                new_stem = stem[:-len(suffix_to_remove)]
                new_name = new_stem + ext

                # 构建目标路径并复制文件 (copy2 会保留文件的原始元数据，如创建时间)
                dest_file_path = dest_path / new_name
                shutil.copy2(file_path, dest_file_path)

                print(f"✅ 已处理: {file_path.name} -> {new_name}")
                count += 1
            else:
                # 如果没有匹配的后缀，则跳过
                print(f"⚠️ 跳过 (无匹配后缀): {file_path.name}")

    print(f"🎉 [{src_dir}] 处理完成，共成功复制了 {count} 个文件。\n" + "-" * 40 + "\n")


# ================= ⚙️ 配置区域 =================

# 1. 你的两个原始文件夹路径 (可以使用绝对路径或相对路径)
SOURCE_FOLDER_1 = r"D:\Master\Snake\Snake_2026_01_26\Dataset\CHN6-CUG_ori_suffix\test\images"

# 2. 你的两个目标新文件夹路径 (如果不存在，代码会自动创建)
TARGET_FOLDER_1 = r"D:\Master\Snake\Snake_2026_01_26\Dataset\CHN6-CUG\test\images"

# 3. 你需要去除的两种后缀
# 注意：这指的是扩展名前的字符串。
# 例如：文件名是 "data_old.csv"，你想变成 "data.csv"，那么后缀就是 "_old"
SUFFIX_1 = "_sat"  # 对应第一个文件夹要去除的后缀


# ===============================================

if __name__ == "__main__":
    # 处理第一个文件夹
    print(f"开始处理第一个文件夹: {SOURCE_FOLDER_1}")
    process_files(SOURCE_FOLDER_1, TARGET_FOLDER_1, SUFFIX_1)
