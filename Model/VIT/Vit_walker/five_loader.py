import concurrent.futures
import glob
import os
import random

import cv2
import numpy as np
from PIL import Image
from torchvision.datasets import VisionDataset
from tqdm import tqdm


class MyFiveLikeDataset(VisionDataset):
    def __init__(self, dataset, transform, network_size):
        super(MyFiveLikeDataset, self).__init__('data/', None, None, None)
        self.dataset = dataset
        self.transform = transform
        self.network_size = network_size
        self.cache_data = self.load_all_data()

    def load_all_data(self):
        cache_data = []
        avail_mem = calculate_mem()
        item_mem = self.network_size[0] * self.network_size[1] * 4
        item_num = len(self.dataset)
        total_req_mem = item_num * item_mem / 1024 / 1024 / 1024
        print(f'-->转换{item_num}个Tensor，需要内存:{total_req_mem:.2f}GB，{avail_mem:.2f}GB（可用），开始...')

        if avail_mem > total_req_mem:
            for index in tqdm(range(len(self.dataset)), desc="->MyMNISTLikeDataset Fast_Cache", ncols=100):
                data_item = self.getitem(index)
                cache_data.append(data_item)
        else:
            print(f'-->内存不足！')
        return cache_data

    def getitem(self, index):
        img = self.dataset[index][2]
        target = self.dataset[index][0]

        if self.transform:
            img = Image.fromarray(img)
            img = self.transform(img)
        return img, target

    def __getitem__(self, index):
        if len(self.cache_data) > 0:
            one_data = self.cache_data[index]
        else:
            one_data = self.getitem(index)
        return one_data

    def __len__(self):
        return len(self.dataset)


def calculate_mem():
    import psutil
    memory_info = psutil.virtual_memory()
    total_memory_gb = float(memory_info.total) / 1024 / 1024 / 1024
    used_memory_gb = float(memory_info.used) / 1024 / 1024 / 1024
    free_memory_gb = float(memory_info.free) / 1024 / 1024 / 1024
    print(f'-->系统内存: {total_memory_gb:.2f}GB = {used_memory_gb:.2f}GB(占用） + {free_memory_gb:.2f}GB(空闲)')
    return free_memory_gb


def draw_local_direction(input_image, angle_list, center_coordinates=None):
    image_height, image_width = input_image.shape[:2]
    if center_coordinates is None:
        center_coordinates = np.array([image_height // 2, image_width // 2])
    max_arrow_length = min(40, min(image_height, image_width) // 3)
    drawn_image = input_image.copy()
    num_angles = len(angle_list)
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    for i, angle in enumerate(angle_list):
        center = center_coordinates[i]
        color = colors[i % 3]
        arrow_length = max_arrow_length * (1 - 0.7 * i / num_angles)
        end = center + np.array([-np.sin(np.radians(angle)), np.cos(np.radians(angle))]) * arrow_length
        drawn_image = cv2.arrowedLine(drawn_image, tuple(center[::-1]),
                                      tuple(end.astype(int)[::-1]), color, 2, cv2.LINE_AA, tipLength=0.2)
    return drawn_image


def save2pic(img_folder, tag, img10, img20, img30, img40, local_direction, center_coordinates=None):
    counter = getattr(save2pic, 'counter', 0)
    save2pic.counter = counter + 1

    img1 = cv2.cvtColor(img10.astype(np.uint8), cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img20.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    img3 = cv2.cvtColor(img30.astype(np.uint8), cv2.COLOR_BGR2RGB)
    img4 = cv2.cvtColor(img40.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    img4 = draw_local_direction(img4, local_direction, center_coordinates)

    clr = (0, 255, 255)
    img1 = cv2.putText(img1, tag, (0, 8), cv2.FONT_HERSHEY_SIMPLEX, 0.25, clr, 1, lineType=cv2.LINE_AA)
    img3 = cv2.putText(img3, tag, (0, 8), cv2.FONT_HERSHEY_SIMPLEX, 0.25, clr, 1, lineType=cv2.LINE_AA)
    file_name = f"tmp_%04d_{tag}.png" % counter
    img_name = os.path.join(img_folder, file_name)
    save2big(img_name, img1, img2, img3, img4)
    return


def save2big(img_name, img1, img2, img3, img4, gap=4):
    h, w, _ = img1.shape
    img_big = np.full((h * 2 + gap, w * 2 + gap, 3), 127)
    img_big[:h, :w] = img1
    img_big[:h, -w:] = img2
    img_big[-h:, :w] = img3
    img_big[-h:, -w:] = img4
    show_image(img_big)
    # cv2.imwrite(img_name, img_big)
    return


def show_image(image, ellipse=None):
    colored_image = image.astype(np.uint8)  # cv2.applyColorMap(images, cv2.COLORMAP_HOT)
    if ellipse:
        cv2.ellipse(colored_image, ellipse, (0, 255, 0), 2)
    cv2.imshow('Image with mark', colored_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return


def select_white_pixels(label, num, patch):
    pad = patch // 2
    img_with_border_zero = np.copy(label)
    img_with_border_zero[:pad, :] = 0
    img_with_border_zero[-pad:, :] = 0
    img_with_border_zero[:, :pad] = 0
    img_with_border_zero[:, -pad:] = 0
    valid_coords = np.argwhere((img_with_border_zero > 127))
    select_coords = random.sample(list(valid_coords), num) if len(valid_coords) >= num else list(valid_coords)
    return select_coords


class DataLoader:
    def __init__(self, file_dir, image_name, label_name):
        self.image_files, self.label_files = self.load_dataset(file_dir, image_name, label_name)
        self.sectors = None

    def load_dataset(self, file_dir, img_name, lbl_name):
        img_folder = os.path.join(file_dir, img_name)
        lbl_folder = os.path.join(file_dir, lbl_name)
        img_list = [f for ext in ['*.jpg', '*.png', '*.bmp'] for f in glob.glob(os.path.join(img_folder, ext))]

        new_imgs, new_lbls = [], []
        for img in img_list:
            lbl = os.path.join(lbl_folder, os.path.basename(img))
            if os.path.exists(img) and os.path.exists(lbl):
                new_imgs.append(img)
                new_lbls.append(lbl)
        return new_imgs, new_lbls

    def getnum(self):
        return len(self.image_files)

    def getname(self, i):
        return self.image_files[i], self.label_files[i]

    def get_item(self, i, samples, patch):
        image_path, label_path = self.getname(i)
        image = cv2.imread(image_path)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        size = (768, 768)
        image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR_EXACT)
        label = cv2.resize(label, size, interpolation=cv2.INTER_NEAREST)

        coords = select_white_pixels(label, samples, patch)
        infos = [self.crop_patch(image, label, patch, coord) for coord in coords]
        return infos

    def load_data(self, start_index, end_index, samples, patch):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.get_item, index, samples, patch) for index in range(start_index, end_index)]
            total = len(futures)
            with tqdm(total=total, desc="Processing") as pbar:
                data_list = []
                for future in concurrent.futures.as_completed(futures):
                    items = future.result()
                    for item in items:
                        data_list.append(item)
                    pbar.set_description(f"Train/Test Data Generating - {len(data_list)}/{total * samples}")
                    pbar.update(1)
            return data_list

    def load_all_data(self, start_index, end_index, samples, patch):
        cache_data = []
        avail_mem = calculate_mem()
        item_mem = patch * patch * (3 * 4 + 4)
        item_num = (end_index - start_index) * samples
        total_req_mem = item_num * item_mem / 1024 / 1024 / 1024
        print(f'-->载入{item_num}个item，需要内存:{total_req_mem:.2f}GB，{avail_mem:.2f}GB（可用），开始...')

        if avail_mem > total_req_mem:
            cache_data = self.load_data(start_index, end_index, samples, patch)
        else:
            print(f'-->内存不足！')
        return cache_data

    def crop_patch(self, image, label, patch, center_coord):
        pad = patch // 2
        y0, x0 = center_coord[0] - pad, center_coord[1] - pad
        y1, x1 = center_coord[0] + pad, center_coord[1] + pad
        cropped_img = image[y0:y1, x0:x1, :]
        cropped_lbl = label[y0:y1, x0:x1]

        top_idx, top_degree = self.find_angle(cropped_lbl, 20)
        # save2pic("crop_patch", f"{top_degree}", cropped_img, cropped_lbl, cropped_img, cropped_lbl, top_degree)

        return top_idx[0], [x0, y0, x1, y1], cropped_img, cropped_lbl

    def find_angle(self, lbl, dist_thresh):
        angles = np.arange(0, 360, 45)

        if self.sectors is None:
            h, w = lbl.shape
            self.sectors = [ForceSector(max(h, w) // 2, angle) for angle in angles]

        connect_lbl = self.find_connected_255_region(lbl, dist_thresh)
        forces = [np.mean(connect_lbl * s.w) for s in self.sectors]

        top_n = self.get_top_n(forces, 2)
        mapped_angle = angles[top_n]
        return top_n, mapped_angle

    def get_top_n(self, forces, n):
        sorted_indices = np.argsort(forces)[::-1]
        return sorted_indices[:n]

    def find_connected_255_region(self, label, dist_thresh):
        if not np.all(np.isin(label.flatten(), [0, 255])):
            raise ValueError("Input images is not binary.")

        pad = dist_thresh
        img = np.copy(label)
        img_h, img_w = img.shape
        center = (img_h // 2, img_w // 2)
        y0, x0 = center[0] - pad, center[1] - pad
        y1, x1 = center[0] + pad, center[1] + pad
        img[:y0, :] = 0
        img[y1:, :] = 0
        img[:, :x0] = 0
        img[:, x1:] = 0

        if img[center] != 255:
            return np.zeros_like(img)

        visited = set()
        queue = [center]
        while queue:
            curr_y, curr_x = queue.pop(0)
            if (curr_y, curr_x) in visited:
                continue
            visited.add((curr_y, curr_x))
            if img[curr_y, curr_x] == 255:
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    new_y, new_x = curr_y + dy, curr_x + dx
                    if 0 <= new_y < img_h and 0 <= new_x < img_w and (new_y, new_x) not in visited:
                        queue.append((new_y, new_x))

        bin_mat = np.zeros_like(img)
        for y, x in visited:
            bin_mat[y, x] = 255

        return bin_mat


class ForceSector:
    def __init__(self, radius, angle):
        cls = self.__class__
        if not hasattr(cls, 'current_id'):
            cls.current_id = 0
        cls.current_id += 1

        self.idx = cls.current_id
        self.radius = radius
        self.angle = angle
        self.w = self.create_sector(radius, angle)

    def create_sector(self, r, angle):
        w1 = self.create_dist_weight(r)
        w2 = self.create_angle_weight(r, angle)
        w = w1 * w2
        # show_image(w * 255, None)
        return w

    def create_dist_weight(self, r):
        center_row, center_col = r, r
        d = 2 * r
        j, i = np.meshgrid(np.arange(d), np.arange(d))
        y, x = j - center_row, i - center_col
        d = np.sqrt(x * x + y * y)
        w = np.maximum(0, 1 - d / r)
        return w

    def create_angle_matrix(self, r):  # Cartesian coordinate
        d = 2 * r
        j, i = np.meshgrid(np.arange(d), np.arange(d))
        angle_matrix = np.degrees(np.arctan2(r - i, j - r))
        angle_matrix[angle_matrix < 0] += 360
        return angle_matrix

    def create_angle_weight(self, r, angle, angle_gap=45):
        m = self.create_angle_matrix(r)
        half_angle = angle_gap / 2
        lower_bounds = (angle - half_angle)
        upper_bounds = (angle + half_angle)
        if lower_bounds > 0:
            sector = np.where((lower_bounds <= m) & (m <= upper_bounds), 1, 0)
        else:  # [-22.5，22.5]度的角度处理
            sector = np.where(((lower_bounds + 360) <= m) | (m <= upper_bounds), 1, 0)
        sector[r, r] = 0
        return sector
