import random

import cv2
import numpy as np
import torch
from drive_loader import skeletonize_and_simplify
from sklearn.metrics import precision_recall_fscore_support

from model2 import DirectionNet


class DynamicVesselTracker:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        model_path = "./model/DirectionNet/DirectionNet_best.pt"
        self.model = DirectionNet().to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()

        self.tracked_points = []
        self.current_points = None
        self.vessel_mask = None
        self.img_size = (1024, 1024)
        self.point_radius = 32

        self.last_main_dir = None
        self.all_history_pixels = []
        self.min_distance = 10

    def add_vessel_occlusion(self, img, mask, center_point,
                                        occlusion_type='rect', num_occlusions=1,
                                        occlusion_size_range=(5, 10), search_radius=50):
        img_with_occl = img.copy()
        mask_with_occl = mask.copy()
        img_with_occl = cv2.resize(img_with_occl, self.img_size)
        mask_with_occl = cv2.resize(mask_with_occl, self.img_size)
        center_y, center_x = center_point

        # 1. 提取以所选点为中心、search_radius为半径的区域内的血管像素
        gray_mask = cv2.cvtColor(mask_with_occl, cv2.COLOR_BGR2GRAY)
        height, width = gray_mask.shape

        # 生成圆形搜索区域掩码
        search_mask = np.zeros_like(gray_mask)
        cv2.circle(search_mask, (center_x, center_y), search_radius, 255, -1)

        # 提取搜索区域内的血管像素坐标（同时满足：在搜索区域内 + 是血管）
        vessel_coords = np.argwhere((gray_mask > 0) & (search_mask > 0))

        # 2. 从附近血管区域中随机选择遮挡中心
        occl_centers = random.sample(list(vessel_coords), min(num_occlusions, len(vessel_coords)))

        # 3. 生成遮挡（黑色填充）
        min_size, max_size = occlusion_size_range
        for (y, x) in occl_centers:
            occl_size = random.randint(min_size, max_size)
            half_size = occl_size // 2

            occl_circle_mask = np.zeros_like(gray_mask)
            cv2.circle(occl_circle_mask, (x, y), half_size, 255, -1)
            img_with_occl[occl_circle_mask > 0] = (0, 0, 0)
            mask_with_occl[occl_circle_mask > 0] = (0, 0, 0)

        return img_with_occl, mask_with_occl

    def preprocess_img(self, img):
        img = cv2.resize(img, self.img_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose(2, 0, 1) / 255.0  # [H,W,C] → [C,H,W]，归一化
        mean = np.array([0.5, 0.5, 0.5]).reshape(3, 1, 1)  # 适配(C,H,W)维度
        std = np.array([0.25, 0.25, 0.25]).reshape(3, 1, 1)
        img = (img - mean) / std
        return torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(self.device)

    def on_mouse_click(self, event, x, y, flags, param):
        global click_coords
        # 捕获鼠标左键单击事件
        if event == cv2.EVENT_LBUTTONDOWN:
            click_coords = (y, x)  # 存储格式：(y, x)，和原有坐标体系一致
            print(f"已选择初始点：像素坐标(y={y}, x={x})")

    # ========== 新增：手动鼠标选点函数 ==========
    def manual_select_start_point(self, img_path):
        """
        显示图像，等待用户鼠标左键点击选择初始点，返回选点坐标(y, x)（适配img_size）
        """
        global click_coords
        click_coords = None

        # 读取并resize图像到模型输入尺寸（保证选点坐标和模型输入匹配）
        img = cv2.imread(img_path)
        img_resized = cv2.resize(img, self.img_size)

        # 创建窗口并绑定鼠标事件
        cv2.namedWindow("Select Initial Tracking Point (Left Click to Choose, Press 'q' to Confirm)", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Select Initial Tracking Point (Left Click to Choose, Press 'q' to Confirm)",
                             self.on_mouse_click)

        # 显示图像，等待用户操作
        while True:
            # 实时显示图像，若已选点则标记
            display_img = img_resized.copy()
            if click_coords is not None:
                y, x = click_coords
                cv2.circle(display_img, (x, y), 3, (0, 0, 255), -1)  # 红色圆点标记选点
                cv2.putText(display_img, f"Selected: ({y}, {x})", (x + 10, y + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            cv2.imshow("Select Initial Tracking Point (Left Click to Choose, Press 'q' to Confirm)", display_img)

            # 按q键确认选点并退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

        # 校验选点是否有效
        if click_coords is None:
            print("警告：未选择初始点，自动使用图像中心作为初始点")
            click_coords = (self.img_size[0] // 2, self.img_size[1] // 2)

        return click_coords

    def init_start_points(self, manual_point = None):

        if manual_point is not None:
            # 使用手动选点
            init_points = np.array([manual_point])
        else:
            img = cv2.imread(args.image_path)
            gray = cv2.cvtColor(cv2.resize(img, self.img_size), cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            sk_label, conn_points, _ = skeletonize_and_simplify(thresh)
            if len(conn_points) >= 1:
                init_points = np.array([random.choice(conn_points)])
            else:
                init_points = np.array([[self.img_size[0] // 2, self.img_size[1] // 2]])

        self.current_points = self.normalize_points(init_points)
        self.tracked_points.append(init_points.copy())
        return self.current_points, init_points[0]

    def calculate_dir_angle(self, dir_idx1, dir_idx2):
        angle_per_dir = 360 / 16
        angle1 = dir_idx1 * angle_per_dir
        angle2 = dir_idx2 * angle_per_dir
        diff = abs(angle1 - angle2)
        return diff if diff <= 180 else 360 - diff

    def normalize_points(self, points):
        H, W = self.img_size
        norm_points = np.zeros((1, len(points), 2), dtype=np.float32)  # [batch, num_points, 2]
        for i, (y, x) in enumerate(points):
            norm_points[0, i, 0] = y / (H - 1) if H > 1 else 0.0
            norm_points[0, i, 1] = x / (W - 1) if W > 1 else 0.0
        return torch.tensor(norm_points, dtype=torch.float32).to(self.device)

    def denormalize_points(self, norm_points):
        H, W = self.img_size
        norm_points_np = norm_points.cpu().numpy()
        norm_points_np = norm_points_np.squeeze(axis=0)
        norm_points_np = norm_points_np.reshape(-1, 2)
        pixel_points = np.zeros_like(norm_points_np, dtype=np.int32)
        for i, (y_norm, x_norm) in enumerate(norm_points_np):
            pixel_points[i, 0] = int(y_norm * (H - 1))
            pixel_points[i, 1] = int(x_norm * (W - 1))
        return pixel_points

    def dynamic_extend_points(self, dir_results, vessel_mask):
        current_pixel = self.denormalize_points(self.current_points)
        dir_codes = dir_results.cpu().numpy().squeeze()
        if dir_codes.ndim == 1:
            dir_codes = dir_codes.reshape(1, 32)

        vessel_mask_bin = np.where(vessel_mask > 0.5, 1, 0).astype(np.uint8)
        next_pixel = []
        for i, (y, x) in enumerate(current_pixel):
            large_view_conf =dir_codes[i, 16:]
            valid_dir_info = []
            for dir_idx in range(16):
                if self.last_main_dir is None:
                    valid_dir_info.append((dir_idx, large_view_conf[dir_idx]))
                else:
                    angle = self.calculate_dir_angle(dir_idx, self.last_main_dir)
                    if angle <= 90:
                        valid_dir_info.append((dir_idx, large_view_conf[dir_idx]))

            valid_dir_info.sort(key=lambda x: x[1], reverse=True)
            valid_dir_idxs = [d[0] for d in valid_dir_info]
            # valid_conf = [d[1] for d in valid_dir_info]

            final_dir_idx = None
            step_size = None
            new_y, new_x = y, x

            base_step = 15

            for dir_idx in valid_dir_idxs:
                angle = dir_idx * (360 / 16)
                rad = np.radians(angle)
                temp_y = y + int(base_step * np.sin(rad))
                temp_x = x + int(base_step * np.cos(rad))
                # 边界约束
                temp_y = np.clip(temp_y, self.point_radius, self.img_size[0] - self.point_radius)
                temp_x = np.clip(temp_x, self.point_radius, self.img_size[1] - self.point_radius)

                # 新点是否在血管区域内
                if vessel_mask_bin[temp_y, temp_x] == 1:
                    final_dir_idx = dir_idx
                    new_y, new_x = temp_y, temp_x
                    step_size = base_step
                    break
                # 移出，减半步长
                elif vessel_mask_bin[temp_y, temp_x] == 0:
                    temp_y_half = y + int((base_step // 2) * np.sin(rad))
                    temp_x_half = x + int((base_step // 2) * np.cos(rad))
                    temp_y_half = np.clip(temp_y_half, self.point_radius, self.img_size[0] - self.point_radius)
                    temp_x_half = np.clip(temp_x_half, self.point_radius, self.img_size[1] - self.point_radius)
                    if vessel_mask_bin[temp_y_half, temp_x_half] == 1:
                        final_dir_idx = dir_idx
                        new_y, new_x = temp_y_half, temp_x_half
                        step_size = base_step // 2
                        break

            if final_dir_idx is None:
                final_dir_idx = self.last_main_dir if self.last_main_dir is not None else 0
                angle = final_dir_idx * (360 / 16)
                rad = np.radians(angle)

                new_y = y + int(2 * np.sin(rad))
                new_x = x + int(2 * np.cos(rad))
                new_y = np.clip(new_y, self.point_radius, self.img_size[0] - self.point_radius)
                new_x = np.clip(new_x, self.point_radius, self.img_size[1] - self.point_radius)
                step_size = 2

            # 更新历史方向
            self.last_main_dir = final_dir_idx
            next_pixel.append([new_y, new_x])

            # 更新追踪点
        next_pixel = np.array(next_pixel)
        self.tracked_points.append(next_pixel.copy())
        self.current_points = self.normalize_points(next_pixel)
        return current_pixel, next_pixel

    def visualize_tracking(self, img, current_pixel, next_pixel, thresh):

        vis_img = img.copy()
        vis_img = cv2.resize(vis_img,self.img_size)
        vessel_mask = thresh.astype(np.uint8) * 255

        mask_color = cv2.applyColorMap(vessel_mask, cv2.COLORMAP_JET)

        cv2.imwrite("vessel.jpg", vessel_mask)

        vis_img = cv2.addWeighted(vis_img, 1, mask_color, 0, 0)

        for i in range(1, len(self.tracked_points)):
            prev_points = self.tracked_points[i - 1]
            curr_points = self.tracked_points[i]
            for (y1, x1), (y2, x2) in zip(prev_points, curr_points):
                # 轨迹线
                cv2.arrowedLine(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 1, tipLength=0.5)
                cv2.rectangle(vis_img, (x1 - self.point_radius // 2, y1 - self.point_radius // 2),
                              (x1+ self.point_radius // 2, y1 + self.point_radius // 2), (0, 0, 255), 2)

        for (y, x) in current_pixel:
            cv2.circle(vis_img, (x, y), 1, (0, 0, 255), -1)
        for (y, x) in next_pixel:
            cv2.circle(vis_img, (x, y), 1, (255, 0, 0), -1)

        return vis_img

    def track_single_image(self, mask_path, img_path, output_path="tracking_result.jpg", max_steps=10,
                           add_occlusion=True, occlusion_type='rect', num_occlusions=3,
                           occlusion_size_range=(20,50), occlusion_search_radius=50):

        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path)
        selected_point = self.manual_select_start_point(img_path)
        self.init_start_points(manual_point=selected_point)
        # 添加血管区域遮挡
        if add_occlusion:
            img, mask = self.add_vessel_occlusion(img, mask, selected_point,
                occlusion_type=occlusion_type,
                num_occlusions=num_occlusions,
                occlusion_size_range=occlusion_size_range,
                search_radius=occlusion_search_radius)
            # 保存带遮挡的图像，方便对比
            cv2.imwrite("img_with_occlusion.jpg", img)
            cv2.imwrite("mask_with_occlusion.jpg", mask)
            print("已生成带遮挡的图像：img_with_occlusion.jpg 和 mask_with_occlusion.jpg")

        H, W = self.img_size
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 5
        video_writer = cv2.VideoWriter("tracking_video.mp4", fourcc, fps, (W, H))

        gray = cv2.cvtColor(cv2.resize(mask, self.img_size), cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh = thresh/255

        img_tensor = self.preprocess_img(img)

        for step in range(max_steps):
            with torch.no_grad():

                outputs = self.model(img_tensor, self.current_points)
                dir_results = outputs["dir"]
                self.vessel_mask = torch.argmax(outputs["vessel"], dim=1).cpu().numpy()
                self.vessel_mask = self.vessel_mask.squeeze()

            current_pixel, next_pixel = self.dynamic_extend_points(dir_results,  self.vessel_mask)
            vis_frame = self.visualize_tracking(img, current_pixel, next_pixel, self.vessel_mask)
            video_writer.write(vis_frame)
            print(f"延伸步数：{step + 1}/{max_steps}，已追踪点组数：{len(self.tracked_points)}")

        f1 = self.calculate_segmentation_f1(gray, self.vessel_mask)
        print("f1 = ",f1)

        video_writer.release()

        final_vis = self.visualize_tracking(img, current_pixel, next_pixel, self.vessel_mask)
        cv2.imwrite(output_path, final_vis)


        cv2.imshow("Dynamic Vessel Tracking (Press 'q' to exit)", final_vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print(f"追踪完成！共延伸 {max_steps} 步")

    def calculate_segmentation_f1(self, gt_mask, pred_mask):
        """
        计算分割结果的精确率、召回率、F1分数（按你的方式调用）
        返回：precision, recall, f1
        """
        # 预处理（和之前一致）
        gt_mask = cv2.resize(gt_mask, self.img_size)
        pred_mask = cv2.resize(pred_mask, self.img_size)
        _, gt_mask_bin = cv2.threshold(gt_mask, 0, 1, cv2.THRESH_BINARY)

        # 展平
        gt_flat = gt_mask_bin.flatten()
        pred_flat = pred_mask.flatten()

        # 调用你指定的函数
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true=gt_flat,
            y_pred=pred_flat,
            average='binary',
            pos_label=1,
            zero_division=0
        )
        return f1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--mask_path', type=str, default=r'D:\hs\DirectionNet - 3\data\drive\test\gt\im0139_vk.png')
    # parser.add_argument('--image_path', type=str, default=r'D:\hs\DirectionNet - 3\data\DirectionNet\test\image\1.tif')

    parser.add_argument('--image_path', type=str, default=r'D:\hs\DirectionNet - 3\data\drive\test\image\im0139_vk.png')

    parser.add_argument('--output_path', type=str, default='tracking_result.jpg', help='输出追踪结果图片路径')
    parser.add_argument('--max_steps', type=int, default=20, help='最大动态延伸步数')

    parser.add_argument('--add_occlusion', type=bool, default=False, help='是否添加血管遮挡')
    parser.add_argument('--occlusion_type', type=str, default='rect', choices=['rect', 'circle'],
                        help='遮挡类型：rect(矩形)/circle(圆形)')
    parser.add_argument('--num_occlusions', type=int, default=5, help='遮挡数量')
    parser.add_argument('--occlusion_size', type=tuple, default=(5, 10), help='遮挡大小范围 (min, max)')
    parser.add_argument('--occlusion_search_radius', type=int, default=50, help='遮挡搜索半径（以所选点为中心，像素）')
    args = parser.parse_args()

    tracker = DynamicVesselTracker()
    tracker.track_single_image(mask_path =args.mask_path ,
                               img_path=args.image_path,
                               output_path=args.output_path,
                               max_steps=args.max_steps,
                               add_occlusion=args.add_occlusion,
                               occlusion_type=args.occlusion_type,
                               num_occlusions=args.num_occlusions,
                               occlusion_size_range=args.occlusion_size,
                               occlusion_search_radius=args.occlusion_search_radius
                               )