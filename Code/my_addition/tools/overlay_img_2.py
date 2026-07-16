import cv2
import numpy as np
import os


def process_masks(path1, path2):
    """
    处理两张掩码图并生成交集结果。

    :param path1: 彩色图 (mask1) 的路径
    :param path2: 二值图 (mask2) 的路径
    """
    # 1. 读取图像
    # mask1 是彩色图，OpenCV 默认以 BGR 格式读取
    mask1 = cv2.imread(path1)
    # mask2 是二值图，直接以灰度模式读取
    mask2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)

    if mask1 is None or mask2 is None:
        print("❌ 图像读取失败，请检查路径是否正确。")
        return

    # 尺寸校验：确保两张图尺寸一致。如果不一致，将 mask2 强转为 mask1 的尺寸
    if mask1.shape[:2] != mask2.shape:
        print("⚠️ 警告：两张图像尺寸不一致，已自动调整二值图(mask2)的尺寸。")
        mask2 = cv2.resize(mask2, (mask1.shape[1], mask1.shape[0]), interpolation=cv2.INTER_NEAREST)

    # 2. 提取逻辑条件 (布尔矩阵)
    # 提取 mask1 中的纯白色部分 (BGR: 255, 255, 255)
    lower_white = np.array([255, 255, 255], dtype=np.uint8)
    upper_white = np.array([255, 255, 255], dtype=np.uint8)
    # inRange 会返回 0 或 255 的灰度图，我们将其转化为 True/False 的布尔掩码
    m1_is_white = cv2.inRange(mask1, lower_white, upper_white) == 255

    # 提取 mask2 中的白色部分(255) 和黑色部分(0)
    m2_is_white = mask2 == 255
    m2_is_black = mask2 == 0

    # 3. 创建结果图 mask3
    # 直接复制 mask1，这样既能保证“红色部分保持不变”，也能保留其他无关区域的原始状态
    mask3 = mask1.copy()

    # 4. 计算交集并赋予新颜色 (注意 OpenCV 的颜色通道是 BGR)
    # 交集 A: mask1 白 且 mask2 白 -> 变绿 (BGR: 0, 255, 0)
    intersect_green = m1_is_white & m2_is_white
    mask3[intersect_green] = [0, 255, 0]

    # 交集 B: mask1 白 且 mask2 黑 -> 变黄 (BGR: 0, 255, 255)
    intersect_yellow = m1_is_white & m2_is_black
    mask3[intersect_yellow] = [0, 255, 255]

    # 5. 保存结果
    # 获取 mask1 的目录
    save_dir = os.path.dirname(path1)
    # 构造新的文件名，例如 "mask3_原图名.png"
    base_name = os.path.basename(path1)
    save_path = os.path.join(save_dir, f"mask3_{base_name}")

    cv2.imwrite(save_path, mask3)
    print(f"✅ 处理完成！结果已保存至: {save_path}")


# ================= 运行测试 =================
if __name__ == "__main__":
    # 请替换为您本地的实际图片路径
    mask1_path = r"D:\Master\Snake\Snake_2026_01_26\Code\my_addition\others\images\2026_04_01\M_roads_UNet_10978735_15_mr.png"
    mask2_path = r"D:\Master\Snake\Snake_2026_01_26\Code\my_addition\others\images\2026_04_01\M_Roads_10978735_15_label.tif"

    process_masks(mask1_path, mask2_path)