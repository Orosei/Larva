import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchvision.models import ResNet34_Weights


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

        resnet = models.resnet34(weights=ResNet34_Weights.DEFAULT)
        self.shared_conv1 = resnet.conv1
        self.shared_bn1 = resnet.bn1
        self.shared_relu = resnet.relu
        self.shared_maxpool = resnet.maxpool
        self.shared_layer1 = resnet.layer1
        self.shared_layer2 = resnet.layer2
        self.seg_layer3 = resnet.layer3
        self.seg_layer4 = resnet.layer4

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
            nn.BatchNorm1d(16*24*24),
            nn.ReLU(inplace=True),
            nn.Linear(16*24*24, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, 16)
        )

        self.large_fov_head = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=1, padding=0),
            nn.Flatten(),
            nn.BatchNorm1d(16*47*47),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(16*47*47, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 16)
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
        batch_size, num_points, _ = point_coords.size()
        B, C, H_feat, W_feat = features.size()
        H, W = original_size
        scale_h, scale_w = H_feat / H, W_feat / W
        feat_fov_h = fov_size
        feat_fov_w = fov_size
        point_regions = []
        for i in range(batch_size):
            for j in range(num_points):
                y_norm, x_norm = point_coords[i, j]
                x = int(x_norm * (W - 1))
                y = int(y_norm * (H - 1))
                x_feat = int(x * scale_w)
                y_feat = int(y * scale_h)
                half_h, half_w = feat_fov_h // 2, feat_fov_w // 2
                y1 = max(0, y_feat - half_h)
                y2 = min(H_feat, y_feat + half_h)
                x1 = max(0, x_feat - half_w)
                x2 = min(W_feat, x_feat + half_w)
                region = features[i:i + 1, :, y1:y2, x1:x2]
                if region.size(2) < feat_fov_h or region.size(3) < feat_fov_w:
                    pad_h = feat_fov_h - region.size(2)
                    pad_w = feat_fov_w - region.size(3)
                    region = F.pad(region, (pad_w // 2, pad_w - pad_w // 2,
                                            pad_h // 2, pad_h - pad_h // 2))
                point_regions.append(region)
        return torch.cat(point_regions, dim=0)

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
        shared_layer3_feat = self.seg_layer3(shared_layer2_feat)
        shared_layer4_feat = self.seg_layer4(shared_layer3_feat)

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

        seg_feat_small = self.seg_feat_small(seg_output)
        seg_feat_large = self.seg_feat_large(seg_output)

        feat_reduce_small = self.feat_reduce_small(shared_layer4_feat)
        feat_reduce_large = self.feat_reduce_large(shared_layer4_feat)

        image_features = {
            "batch_size": batch_size,
            "img_size": (H, W),
            "seg_output": seg_output,
            "seg_dec2": seg_dec2,
            "seg_dec3": seg_dec3,
            "seg_feat_small": seg_feat_small,
            "seg_feat_large": seg_feat_large,
            "feat_reduce_small": feat_reduce_small,
            "feat_reduce_large": feat_reduce_large
        }
        return image_features

    def pre_dir(self, image_features, point_coords):

        batch_size = image_features["batch_size"]
        H, W = image_features["img_size"]
        num_points = point_coords.size(1) if batch_size > 0 else 0

        # 小视野域
        small_fov_input = torch.cat([image_features["seg_dec3"], image_features["seg_feat_small"]], dim=1)
        small_fov_regions = self._extract_point_regions(
            small_fov_input, point_coords, (H, W), self.small_fov
        )

        small_fov_masks = self._extract_point_regions(
            image_features["seg_output"], point_coords, (H, W), self.small_fov
        )
        small_fov_masks = torch.argmax(small_fov_masks, dim=1).unsqueeze(1)
        small_fov_regions = small_fov_regions * small_fov_masks

        small_fov_feat = self.small_fov_process(small_fov_regions)
        small_directions = self.small_fov_head(small_fov_feat)

        # 大视野域
        large_fov_input = torch.cat([image_features["seg_dec2"], image_features["seg_feat_large"]], dim=1)
        large_fov_regions = self._extract_point_regions(
            large_fov_input, point_coords, (H, W), self.large_fov
        )

        large_fov_masks = self._extract_point_regions(
            image_features["seg_output"], point_coords, (H, W), self.large_fov
        )
        large_fov_masks = torch.argmax(large_fov_masks, dim=1).unsqueeze(1)
        large_fov_regions = large_fov_regions * large_fov_masks

        large_fov_feat = self.large_fov_process(large_fov_regions)
        large_directions = self.large_fov_head(large_fov_feat)

        combined_directions = torch.cat([small_directions, large_directions], dim=1)
        combined_directions = combined_directions.view(batch_size, num_points, -1)
        combined_directions = torch.sigmoid(combined_directions)

        out = {
            "dir": combined_directions,
            "vessel": image_features["seg_output"]
        }
        return out

    def forward(self, img, point_coords):

        image_features = self.img_feature(img)
        return self.pre_dir(image_features, point_coords)