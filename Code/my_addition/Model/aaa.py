import numpy as np
import torch
import cv2

from .model2 import DirectionNet


def get_top_2_dirs(img, pt):
    pred_dirs = get_pred_direction(img, pt)
    pred_dirs_near = pred_dirs[:16]
    top_k_dirs_idxs = np.argsort(pred_dirs_near)[::-1][:1] / 2

    return top_k_dirs_idxs


def get_pred_direction(img, pt):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    img_t = process_img(img)
    pt_t = normalize_points(pt, img.shape)

    model = DirectionNet().to(device)
    model.load_state_dict(
        torch.load("D:/Master/Snake/Snake_2026_1_8/Code/snake/drive/DirectionNet_best.pt", map_location=device))

    model.eval()
    with torch.no_grad():
        output = model(img_t, pt_t)
        dir_prob = output["dir"]
        dir_prob = dir_prob.squeeze(0).squeeze(0).cpu().detach().numpy()

    return dir_prob


def process_img(img):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    img = cv2.resize(img, (1024, 1024))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose(2, 0, 1) / 255.0  # [H,W,C] → [C,H,W]，归一化
    mean = np.array([0.5, 0.5, 0.5]).reshape(3, 1, 1)  # 适配(C,H,W)维度
    std = np.array([0.25, 0.25, 0.25]).reshape(3, 1, 1)
    img = (img - mean) / std
    img_t = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(device)

    return img_t


def normalize_points(points, img_shape):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    H, W, _ = img_shape
    norm_points = np.zeros((1, len(points), 2), dtype=np.float32)  # [batch, num_points, 2]

    for i, (y, x) in enumerate(points):
        norm_points[0, i, 0] = y / (H - 1) if H > 1 else 0.0
        norm_points[0, i, 1] = x / (W - 1) if W > 1 else 0.0

    return torch.tensor(norm_points, dtype=torch.float32).to(device)


def trans_16_to_8(direction):
    direction = int(direction)
    new_dir = direction // 2

    if direction % 2 == 0:
        new_dirs = [(new_dir - 1) % 8, new_dir, (new_dir + 1) % 8]
    else:
        new_dirs = [new_dir, (new_dir + 1) % 8]
    return new_dirs
