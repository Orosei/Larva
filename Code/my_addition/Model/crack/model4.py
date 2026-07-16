import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchvision.models import ResNet34_Weights
import numpy as np


class FourierPositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, L: int = 8):
        super().__init__()
        self.L = L
        self.embed_dim = embed_dim
        fourier_dim = 2 * 2 * L  # 2(x/y) * 2(sin/cos) * L(频率层)
        self.proj = nn.Linear(fourier_dim, embed_dim)
        self.fusion_proj = nn.Conv2d(2 * self.embed_dim, self.embed_dim, kernel_size=1)

    def _fourier_encode(self, coords: torch.Tensor) -> torch.Tensor:
        freq = 2 ** torch.arange(self.L, dtype=torch.float32) * np.pi
        freq = freq.to(coords.device)
        spectrum = coords[..., None] * freq
        sin_enc = torch.sin(spectrum)
        cos_enc = torch.cos(spectrum)
        encoding = torch.stack([sin_enc, cos_enc], dim=-2)
        encoding = encoding.reshape(*coords.shape[:-1], -1)
        return encoding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # 动态生成归一化的二维网格坐标（范围[-1,1]）
        y = torch.linspace(-1, 1, H, device=x.device)
        x_coord = torch.linspace(-1, 1, W, device=x.device)
        grid_y, grid_x = torch.meshgrid(y, x_coord, indexing='ij')
        coords = torch.stack([grid_x, grid_y], dim=-1)  # (H, W, 2)

        # 傅里叶编码 + 线性投影
        pos_enc = self._fourier_encode(coords)  # (H, W, 4L)
        pos_enc = self.proj(pos_enc)  # (H, W, C)
        pos_enc = pos_enc.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)，适配特征图维度
        pos_enc = pos_enc.expand(B, -1, -1, -1)  # (B, C, H, W)
        x_with_pos = torch.cat([x, pos_enc], dim=1)
        x_out = self.fusion_proj(x_with_pos)  # (B, C, H, W)
        return x_out


class ResidualAdd(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.adjust = nn.Identity()

    def forward(self, x):
        return x + self.adjust(x)


class DirectionNet(nn.Module):
    def __init__(self, small_fov=24, large_fov=48, num_seg_classes=2,
                 small_fov_classes=16, large_fov_classes=16):
        super(DirectionNet, self).__init__()

        self.small_fov = small_fov
        self.large_fov = large_fov
        self.num_seg_classes = num_seg_classes
        self.small_fov_classes = small_fov_classes
        self.large_fov_classes = large_fov_classes

        # 共享特征提取（ResNet34）
        resnet = models.resnet34(weights=ResNet34_Weights.DEFAULT)
        self.shared_conv1 = resnet.conv1
        self.shared_bn1 = resnet.bn1
        self.shared_relu = resnet.relu
        self.shared_maxpool = resnet.maxpool
        self.shared_layer1 = resnet.layer1
        self.shared_layer2 = resnet.layer2
        self.seg_layer3 = resnet.layer3
        self.seg_layer4 = resnet.layer4

        self.fourier_enc_layer2 = FourierPositionalEncoding(embed_dim=128, L=8)  # layer2输出通道128
        self.fourier_enc_layer3 = FourierPositionalEncoding(embed_dim=256, L=8)  # layer3输出通道256
        self.fourier_enc_small_fov = FourierPositionalEncoding(embed_dim=96, L=8)  # small_fov_regions通道96
        self.fourier_enc_large_fov = FourierPositionalEncoding(embed_dim=160, L=8)  # large_fov_regions通道160

        # 分割分支
        self.seg_up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.seg_conv1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.seg_up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.seg_conv2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.seg_up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.seg_conv3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.seg_up4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.seg_conv4 = nn.Sequential(
            nn.Conv2d(96, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.segment_head = nn.Conv2d(32, self.num_seg_classes, kernel_size=1)

        self.seg_feat_small = nn.Sequential(
            nn.Conv2d(self.num_seg_classes, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=4, padding=1)
        )
        self.seg_feat_large = nn.Sequential(
            nn.Conv2d(self.num_seg_classes, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=8, padding=1)
        )

        # 小视野
        self.feat_reduce_small = nn.Sequential(
            nn.Conv2d(512, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        )
        self.small_fov_process = nn.Sequential(
            nn.Conv2d(96, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            self._create_dwConv_block(64, 3),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Conv2d(32, 16, kernel_size=1)
        )

        # 大视野
        self.feat_reduce_large = nn.Sequential(
            nn.Conv2d(512, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        )
        self.large_fov_process = nn.Sequential(
            nn.Conv2d(160, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            self._create_dwConv_block(64, 3),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=1)
        )

        self.small_fov_head = nn.Sequential(
            nn.Flatten(),
            nn.ReLU(inplace=True),
            nn.Linear(16 * 24 * 24, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, 16 * 3)
        )

        self.large_fov_head = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=1, padding=0),
            nn.Flatten(),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(16 * 47 * 47, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 16 * 3)
        )

    def _create_dwConv_block(self, in_channels, kernel_size):
        mid_channels = in_channels // 2
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                      groups=in_channels, padding=1, stride=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, mid_channels * 2, kernel_size=1, stride=1),
            nn.BatchNorm2d(mid_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels * 2, mid_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(mid_channels),
            ResidualAdd(in_channels=mid_channels)
        )

    def _extract_point_regions(self, features, point_coords, original_size, fov_size):
        B, C, H_feat, W_feat = features.shape
        batch_size, num_points, _ = point_coords.shape
        H, W = original_size
        scale_h, scale_w = H_feat / H, W_feat / W
        x_pix = (point_coords[..., 1] * (W - 1)).int()
        y_pix = (point_coords[..., 0] * (H - 1)).int()
        x_feat = (x_pix * scale_w).int()
        y_feat = (y_pix * scale_h).int()
        half = fov_size // 2
        y1 = torch.clamp(y_feat - half, 0, H_feat)
        y2 = torch.clamp(y_feat + half, 0, H_feat)
        x1 = torch.clamp(x_feat - half, 0, W_feat)
        x2 = torch.clamp(x_feat + half, 0, W_feat)
        regions = []
        for i in range(batch_size):
            for j in range(num_points):
                region = features[i:i + 1, :, y1[i, j]:y2[i, j], x1[i, j]:x2[i, j]]
                h, w = region.shape[2], region.shape[3]
                if h < fov_size or w < fov_size:
                    pad_h = fov_size - h
                    pad_w = fov_size - w
                    region = F.pad(region, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))
                regions.append(region)
        return torch.cat(regions, dim=0)

    def _process_fov(self, seg_output, seg_dec_feat, seg_feat_module,
                     fourier_enc, fov_process_module, fov_head,
                     point_coords, H, W, fov_size):
        seg_feat = seg_feat_module(seg_output)
        fov_base_feat = torch.cat([seg_dec_feat, seg_feat], dim=1)

        fov_regions = self._extract_point_regions(fov_base_feat, point_coords, (H, W), fov_size)
        # fov_regions = fourier_enc(fov_regions)

        fov_masks = self._extract_point_regions(seg_output, point_coords, (H, W), fov_size)
        fov_masks = torch.argmax(fov_masks, dim=1).unsqueeze(1)
        fov_regions = fov_regions * fov_masks

        fov_feat = fov_process_module(fov_regions)
        directions = fov_head(fov_feat)
        directions = directions.view(-1, 16, 3)
        return directions

    def img_feature(self, img):
        batch_size, _, H, W = img.size()

        # 共享特征提取
        x = self.shared_conv1(img)
        x = self.shared_bn1(x)
        x = self.shared_relu(x)
        shared_conv1_feat = x
        x = self.shared_maxpool(x)

        shared_layer1_feat = self.shared_layer1(x)
        shared_layer2_feat = self.shared_layer2(shared_layer1_feat)
        shared_layer2_feat = self.fourier_enc_layer2(shared_layer2_feat)

        shared_layer3_feat = self.seg_layer3(shared_layer2_feat)
        shared_layer3_feat = self.fourier_enc_layer3(shared_layer3_feat)

        shared_layer4_feat = self.seg_layer4(shared_layer3_feat)

        # 分割分支
        seg_dec1 = self.seg_up1(shared_layer4_feat)
        seg_dec1 = torch.cat([seg_dec1, shared_layer3_feat], dim=1)
        seg_dec1 = self.seg_conv1(seg_dec1)

        seg_dec2 = self.seg_up2(seg_dec1)
        seg_dec2 = torch.cat([seg_dec2, shared_layer2_feat], dim=1)
        seg_dec2 = self.seg_conv2(seg_dec2)

        seg_dec3 = self.seg_up3(seg_dec2)
        seg_dec3 = torch.cat([seg_dec3, shared_layer1_feat], dim=1)
        seg_dec3 = self.seg_conv3(seg_dec3)

        seg_dec4 = self.seg_up4(seg_dec3)
        seg_dec4 = torch.cat([seg_dec4, shared_conv1_feat], dim=1)
        seg_dec4 = self.seg_conv4(seg_dec4)

        seg_output = self.segment_head(seg_dec4)
        seg_output = F.interpolate(seg_output, size=(H, W),
                                   mode='bilinear', align_corners=True)
        seg_output = torch.sigmoid(seg_output)

        features = {
            "seg_output": seg_output,
            "seg_dec3": seg_dec3,
            "seg_dec2": seg_dec2,
            "H": H,
            "W": W
        }
        return features

    def pre_dir(self, features, point_coords):
        seg_output = features["seg_output"]
        seg_dec3 = features["seg_dec3"]
        seg_dec2 = features["seg_dec2"]
        H = features["H"]
        W = features["W"]
        batch_size = seg_output.size(0)
        num_points = point_coords.size(1) if batch_size > 0 else 0

        # 小视野处理
        small_directions = self._process_fov(
            seg_output=seg_output,
            seg_dec_feat=seg_dec3,
            seg_feat_module=self.seg_feat_small,
            fourier_enc=self.fourier_enc_small_fov,
            fov_process_module=self.small_fov_process,
            fov_head=self.small_fov_head,
            point_coords=point_coords,
            H=H, W=W,
            fov_size=self.small_fov
        )

        # 大视野处理
        large_directions = self._process_fov(
            seg_output=seg_output,
            seg_dec_feat=seg_dec2,
            seg_feat_module=self.seg_feat_large,
            fourier_enc=self.fourier_enc_large_fov,
            fov_process_module=self.large_fov_process,
            fov_head=self.large_fov_head,
            point_coords=point_coords,
            H=H, W=W,
            fov_size=self.large_fov
        )

        combined_directions = torch.cat([small_directions, large_directions], dim=1)
        combined_directions = combined_directions.view(batch_size, num_points, 32, 3)

        out = {
            "dir": combined_directions,  # 方向输出 (B, N, 32, 3)
            "vessel": seg_output  # 分割输出 (B, 2, H, W)
        }
        return out

    def forward(self, img, point_coords):
        features = self.extract_features(img)
        out = self.predict_directions(features, point_coords)
        return out


# 测试代码：验证拆分后的调用逻辑
if __name__ == "__main__":
    import time

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 初始化模型
    model = DirectionNet(small_fov=24, large_fov=48, num_seg_classes=2).to(device)
    model.eval()  # 推理模式，关闭dropout等训练层

    # 构造测试输入：单张图像（batch_size=1）
    img = torch.randn(1, 3, 512, 512).to(device)
    # 多组不同的点坐标（模拟多次计算不同点的方向）
    point_coords1 = torch.rand(1, 10, 2).to(device)  # 10个点
    point_coords2 = torch.rand(1, 20, 2).to(device)  # 20个点
    point_coords3 = torch.rand(1, 15, 2).to(device)  # 15个点

    # ========== 方式1：原始方式（重复提取特征，耗时） ==========
    print("=== 原始方式（重复计算特征） ===")
    start_time = time.time()
    out1 = model(img, point_coords1)
    out2 = model(img, point_coords2)
    out3 = model(img, point_coords3)
    end_time = time.time()
    print(f"总耗时: {end_time - start_time:.4f} 秒")

    # ========== 方式2：拆分方式（仅提取一次特征，节省时间） ==========
    print("\n=== 拆分方式（仅计算一次特征） ===")
    start_time = time.time()
    # 第一步：仅提取一次特征（耗时的核心步骤）
    features = model.extract_features(img)
    # 第二步：多次传入不同点坐标计算方向（仅计算轻量级的方向分支）
    out1_new = model.predict_directions(features, point_coords1)
    out2_new = model.predict_directions(features, point_coords2)
    out3_new = model.predict_directions(features, point_coords3)
    end_time = time.time()
    print(f"总耗时: {end_time - start_time:.4f} 秒")

    # 验证输出形状正确性
    print("\n=== 输出形状验证 ===")
    print("分割输出形状:", features["seg_output"].shape)  # (1, 2, 512, 512)
    print("10个点的方向输出形状:", out1_new["dir"].shape)  # (1, 10, 32, 3)
    print("20个点的方向输出形状:", out2_new["dir"].shape)  # (1, 20, 32, 3)
    print("15个点的方向输出形状:", out3_new["dir"].shape)  # (1, 15, 32, 3)