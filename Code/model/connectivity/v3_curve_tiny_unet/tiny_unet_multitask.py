"""
Tiny U-Net multitask model for small 48x48 connectivity patches.

Outputs are raw logits:
    logits: (B, 1)
    seg_logits: (B, 1, H, W)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_activation(name: str) -> nn.Module:
    normalized = name.lower()
    if normalized == "relu":
        return nn.ReLU(inplace=True)
    if normalized == "silu":
        return nn.SiLU(inplace=True)
    raise ValueError(f"不支持的 activation={name!r}，可选: relu / silu")


class ConvBlock(nn.Module):
    """Two 3x3 convolutions with same spatial resolution."""

    def __init__(self, in_channels: int, out_channels: int, activation: str = "silu") -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            _make_activation(activation),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            _make_activation(activation),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpsampleBlock(nn.Module):
    """Configurable 2x upsampling for the Tiny U-Net decoder."""

    _TRANSPOSE_MODES = {"conv_transpose", "transpose", "deconv"}
    _INTERPOLATE_MODES = {"bilinear", "interpolate"}

    def __init__(self, channels: int, mode: str = "conv_transpose") -> None:
        super().__init__()
        self.mode = str(mode).lower()
        if self.mode not in self._TRANSPOSE_MODES | self._INTERPOLATE_MODES:
            raise ValueError(
                "UpsampleBlock mode 仅支持 conv_transpose / bilinear，"
                f"收到: {mode}"
            )
        self.transposed_conv = (
            nn.ConvTranspose2d(channels, channels, kernel_size=2, stride=2)
            if self.mode in self._TRANSPOSE_MODES
            else None
        )

    def forward(self, x: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
        if self.transposed_conv is not None:
            x = self.transposed_conv(x)
            if x.shape[-2:] == target_size:
                return x
        return F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)


class TinyUNetMultiTask(nn.Module):
    """
    Two-downsample Tiny U-Net for tiny foreground segmentation plus patch flag.

    For a 48x48 input, the feature path is:
        48x48 -> 24x24 -> 12x12 -> 24x24 -> 48x48
    """

    def __init__(
        self,
        in_channels: int = 5,
        mask_out_channels: int = 1,
        activation: str = "silu",
        max_patch_size: int = 48,
        upsample_mode: str = "conv_transpose",
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.mask_out_channels = int(mask_out_channels)
        self.max_patch_size = int(max_patch_size)

        self.encoder0 = ConvBlock(self.in_channels, 16, activation=activation)
        self.pool0 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder1 = ConvBlock(16, 32, activation=activation)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = ConvBlock(32, 64, activation=activation)

        self.up1 = UpsampleBlock(64, mode=upsample_mode)
        self.up0 = UpsampleBlock(32, mode=upsample_mode)
        self.decoder1 = ConvBlock(64 + 32, 32, activation=activation)
        self.decoder0 = ConvBlock(32 + 16, 16, activation=activation)

        self.mask_head = nn.Conv2d(16, self.mask_out_channels, kernel_size=1)
        self.flag_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flag_head = nn.Sequential(
            nn.Linear(64, 32),
            _make_activation(activation),
            nn.Linear(32, 1),
        )

    def _validate_input(self, x: torch.Tensor) -> None:
        if x.dim() != 4:
            raise ValueError(f"TinyUNetMultiTask 期望输入为 (B, C, H, W)，收到: {tuple(x.shape)}")
        if x.size(1) != self.in_channels:
            raise ValueError(
                f"TinyUNetMultiTask 期望 {self.in_channels} 个输入通道，收到: {x.size(1)}"
            )
        height, width = x.shape[-2:]
        if height > self.max_patch_size or width > self.max_patch_size:
            raise ValueError(
                f"TinyUNetMultiTask 期望 patch 最大为 {self.max_patch_size}x{self.max_patch_size}，"
                f"收到: {height}x{width}"
            )
        bottleneck_height = height // 4
        bottleneck_width = width // 4
        if bottleneck_height <= 6 or bottleneck_width <= 6:
            raise ValueError(
                "TinyUNetMultiTask 不允许两次下采样后到 6x6 或更小，"
                f"收到 bottleneck: {bottleneck_height}x{bottleneck_width}"
            )

    def forward(self, x: torch.Tensor, return_seg: bool | None = None) -> dict[str, torch.Tensor]:
        self._validate_input(x)

        skip0 = self.encoder0(x)
        x = self.pool0(skip0)

        skip1 = self.encoder1(x)
        x = self.pool1(skip1)

        bottleneck = self.bottleneck(x)

        flag_features = self.flag_pool(bottleneck).flatten(1)
        flag_logits = self.flag_head(flag_features)

        x = self.up1(bottleneck, target_size=skip1.shape[-2:])
        x = torch.cat([x, skip1], dim=1)
        x = self.decoder1(x)

        x = self.up0(x, target_size=skip0.shape[-2:])
        x = torch.cat([x, skip0], dim=1)
        x = self.decoder0(x)
        mask_logits = self.mask_head(x)

        return {"logits": flag_logits, "seg_logits": mask_logits}


def build_tiny_unet_multitask(cfg: dict) -> TinyUNetMultiTask:
    """Build TinyUNetMultiTask from the shared v3 model config."""
    model_cfg = cfg.get("model", {})
    return TinyUNetMultiTask(
        in_channels=int(model_cfg.get("in_channels", 5)),
        mask_out_channels=int(model_cfg.get("seg_out_channels", 1)),
        activation=str(model_cfg.get("activation", "silu")),
        max_patch_size=int(model_cfg.get("max_patch_size", 48)),
        upsample_mode=str(model_cfg.get("upsample_mode", "conv_transpose")),
    )
