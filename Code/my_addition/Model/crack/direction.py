import os

import cv2
import numpy as np
import torch

cur_path = os.path.abspath(__file__)
folder_path = os.path.dirname(cur_path)

from Code.my_addition.Model.crack.model4 import DirectionNet
# from .model4 import DirectionNet


class Direction:
    def __init__(self, img):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        model_path =  os.path.join(folder_path, 'DirectionNet_best.pt')
        # model_path = r"D:\hs\DirectionNet - 3\model\crack\DirectionNet_best.pt"
        self.model = DirectionNet().to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        self.img_shape = img.shape
        self.img_size = (1024, 1024)
        self.img = self.preprocess_img(img)
        self.img_feature = self.model.img_feature(self.img)

    def preprocess_img(self, img):
        img = cv2.resize(img, self.img_size)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose(2, 0, 1) / 255.0  # [H,W,C] → [C,H,W]，归一化
        mean = np.array([0.5, 0.5, 0.5]).reshape(3, 1, 1)  # 适配(C,H,W)维度
        std = np.array([0.25, 0.25, 0.25]).reshape(3, 1, 1)
        img = (img - mean) / std
        return torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(self.device)

    def normalize_points(self, points):
        H, W, _ = self.img_shape
        norm_points = np.zeros((1, len(points), 2), dtype=np.float32)  # [batch, num_points, 2]
        for i, (y, x) in enumerate(points):
            norm_points[0, i, 0] = y / (H - 1) if H > 1 else 0.0
            norm_points[0, i, 1] = x / (W - 1) if W > 1 else 0.0
        return torch.tensor(norm_points, dtype=torch.float32).to(self.device)

    def dir_pre(self, point):
        norm_pt = self.normalize_points(point)
        output = self.model.pre_dir(self.img_feature, norm_pt)
        out = torch.argmax(output["dir"],dim=-1)
        return output["dir"], out


if __name__ == "__main__":
    img_path = r'D:\Master\Snake\Snake_2026_01_26\Dataset\quick_test\images\0.png'
    img = cv2.imread(img_path)
    slover = Direction(img)
    point = [(150, 266)]
    output = slover.dir_pre(point)
    print(output)
