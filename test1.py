import numpy as np


def test_junc_check():
    H, W = 3, 3
    ske_map = np.zeros((H, W))
    junc_map = np.zeros((H, W))

    row_offset = [-1, -1, -1, 0, 0, 1, 1, 1]
    col_offset = [-1, 0, 1, -1, 1, -1, 0, 1]

    # --- 场景 A: 中心点 (1,1)，周围全是 0 ---
    print("\n=== 场景 A: 中心点 (1,1)，周围全是 0 ===")
    r, c = 1, 1

    # 这里的列表推导式完全模拟了 all() 里面的逻辑
    bool_list_A = [
        junc_map[r + dr, c + dc] == 0
        for dr, dc in zip(row_offset, col_offset)
        if 0 <= r + dr < ske_map.shape[0] and 0 <= c + dc < ske_map.shape[1]
    ]

    print(f"检测点: ({r}, {c})")
    print(f"all() 里面的布尔值列表: {bool_list_A}")
    print(f"最终结果 flag: {all(bool_list_A)}")  # 应该是 True

    # --- 场景 B: 中心点 (1,1)，但右边 (1,2) 有一个关节点 ---
    print("\n=== 场景 B: 右边有个关节点 ===")
    junc_map[1, 2] = 1  # 设置障碍

    bool_list_B = [
        junc_map[r + dr, c + dc] == 0
        for dr, dc in zip(row_offset, col_offset)
        if 0 <= r + dr < ske_map.shape[0] and 0 <= c + dc < ske_map.shape[1]
    ]

    print(f"检测点: ({r}, {c})")
    # 注意看列表里会出现一个 False，对应 (1,2) 那个点
    print(f"all() 里面的布尔值列表: {bool_list_B}")
    print(f"最终结果 flag: {all(bool_list_B)}")  # 应该是 False

    # --- 场景 C: 角落点 (0,0)，触发边界检测 ---
    print("\n=== 场景 C: 左上角 (0,0)，边界测试 ===")
    r, c = 0, 0
    # 重置地图为全0
    junc_map[:] = 0

    bool_list_C = [
        junc_map[r + dr, c + dc] == 0
        for dr, dc in zip(row_offset, col_offset)
        # 这里会过滤掉越界的偏移量，导致列表变短
        if 0 <= r + dr < ske_map.shape[0] and 0 <= c + dc < ske_map.shape[1]
    ]

    print(f"检测点: ({r}, {c})")
    # 8邻域在角点只有 3 个有效邻居，所以列表长度只有 3
    print(f"all() 里面的布尔值列表: {bool_list_C}")
    print(f"最终结果 flag: {all(bool_list_C)}")


if __name__ == "__main__":
    test_junc_check()