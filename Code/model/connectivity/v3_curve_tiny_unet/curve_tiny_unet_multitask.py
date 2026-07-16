"""
Curve Tiny U-Net multitask model for v3 connectivity patches.

This model removes the RGB patch-label branch from the dual-branch model. The
remaining curve branch consumes the full 5-channel input, predicts label-curve
logits, downsamples those logits with small 3x3 conv blocks, concatenates the
result with the curve branch bottleneck feature, and applies another convolution
before flattening for connectivity classification.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tiny_unet_multitask import (
    ConvBlock,
    UpsampleBlock,
    _make_activation,
)


class _TinyUNetBackbone(nn.Module):
    """Tiny U-Net feature extractor/decoder without task heads."""

    def __init__(
        self,
        in_channels: int,
        activation: str = "silu",
        upsample_mode: str = "conv_transpose",
    ) -> None:
        super().__init__()
        self.encoder0 = ConvBlock(int(in_channels), 16, activation=activation)
        self.pool0 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder1 = ConvBlock(16, 32, activation=activation)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = ConvBlock(32, 64, activation=activation)

        self.up1 = UpsampleBlock(64, mode=upsample_mode)
        self.up0 = UpsampleBlock(32, mode=upsample_mode)
        self.decoder1 = ConvBlock(64 + 32, 32, activation=activation)
        self.decoder0 = ConvBlock(32 + 16, 16, activation=activation)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        skip0 = self.encoder0(x)
        x = self.pool0(skip0)

        skip1 = self.encoder1(x)
        x = self.pool1(skip1)

        bottleneck = self.bottleneck(x)

        x = self.up1(bottleneck, target_size=skip1.shape[-2:])
        x = torch.cat([x, skip1], dim=1)
        x = self.decoder1(x)

        x = self.up0(x, target_size=skip0.shape[-2:])
        x = torch.cat([x, skip0], dim=1)
        decoder_features = self.decoder0(x)
        return decoder_features, bottleneck


class CurveTinyUNetMultiTask(nn.Module):
    """Single curve Tiny U-Net branch with label-curve segmentation and flag head."""

    def __init__(
        self,
        in_channels: int = 5,
        label_curve_out_channels: int = 1,
        activation: str = "silu",
        max_patch_size: int = 48,
        upsample_mode: str = "conv_transpose",
        logits_fusion_channels: int = 16,
        classifier_hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.label_curve_out_channels = int(label_curve_out_channels)
        self.max_patch_size = int(max_patch_size)

        self.curve_branch = _TinyUNetBackbone(
            self.in_channels,
            activation=activation,
            upsample_mode=upsample_mode,
        )
        self.label_curve_head = nn.Conv2d(16, self.label_curve_out_channels, kernel_size=1)

        logits_fusion_channels = int(logits_fusion_channels)
        self.logits_downsample = nn.Sequential(
            nn.Conv2d(self.label_curve_out_channels, logits_fusion_channels, kernel_size=3, padding=1),
            _make_activation(activation),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(logits_fusion_channels, logits_fusion_channels, kernel_size=3, padding=1),
            _make_activation(activation),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.bottleneck_fusion = nn.Sequential(
            nn.Conv2d(logits_fusion_channels + 64, logits_fusion_channels, kernel_size=3, padding=1),
            _make_activation(activation),
        )

        self.flag_head = nn.Sequential(
            nn.LazyLinear(int(classifier_hidden_dim)),
            _make_activation(activation),
            nn.Linear(int(classifier_hidden_dim), 1),
        )

    def _validate_input(self, x: torch.Tensor) -> None:
        if x.dim() != 4:
            raise ValueError(f"CurveTinyUNetMultiTask 期望输入为 (B, C, H, W)，收到: {tuple(x.shape)}")
        if x.size(1) != self.in_channels:
            raise ValueError(
                f"CurveTinyUNetMultiTask 期望 {self.in_channels} 个输入通道，收到: {x.size(1)}"
            )
        height, width = x.shape[-2:]
        if height > self.max_patch_size or width > self.max_patch_size:
            raise ValueError(
                f"CurveTinyUNetMultiTask 期望 patch 最大为 {self.max_patch_size}x{self.max_patch_size}，"
                f"收到: {height}x{width}"
            )
        bottleneck_height = height // 4
        bottleneck_width = width // 4
        if bottleneck_height <= 6 or bottleneck_width <= 6:
            raise ValueError(
                "CurveTinyUNetMultiTask 不允许两次下采样后到 6x6 或更小，"
                f"收到 bottleneck: {bottleneck_height}x{bottleneck_width}"
            )

    def forward(self, x: torch.Tensor, return_seg: bool | None = None) -> dict[str, torch.Tensor]:
        self._validate_input(x)

        curve_decoder_features, curve_bottleneck = self.curve_branch(x)
        label_curve_logits = self.label_curve_head(curve_decoder_features)

        logits_features = self.logits_downsample(label_curve_logits)
        if logits_features.shape[-2:] != curve_bottleneck.shape[-2:]:
            logits_features = F.interpolate(
                logits_features,
                size=curve_bottleneck.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        flag_features = torch.cat([logits_features, curve_bottleneck], dim=1)
        flag_features = self.bottleneck_fusion(flag_features).flatten(1)
        flag_logits = self.flag_head(flag_features)

        return {
            "logits": flag_logits,
            "seg_logits": label_curve_logits,
            "label_curve_logits": label_curve_logits,
        }


def _resolve_in_channels(value) -> int:
    if isinstance(value, str) and value.lower() == "auto":
        return 5
    return int(value)


def build_curve_tiny_unet_multitask(cfg: dict) -> CurveTinyUNetMultiTask:
    """Build CurveTinyUNetMultiTask from the shared v3 model config."""
    model_cfg = cfg.get("model", {})
    return CurveTinyUNetMultiTask(
        in_channels=_resolve_in_channels(model_cfg.get("in_channels", 5)),
        label_curve_out_channels=int(model_cfg.get("label_curve_out_channels", 1)),
        activation=str(model_cfg.get("activation", "silu")),
        max_patch_size=int(model_cfg.get("max_patch_size", 48)),
        upsample_mode=str(model_cfg.get("upsample_mode", "conv_transpose")),
        logits_fusion_channels=int(model_cfg.get("curve_tiny_unet_logits_fusion_channels", model_cfg.get("dual_branch_logits_fusion_channels", 16))),
        classifier_hidden_dim=int(model_cfg.get("curve_tiny_unet_classifier_hidden_dim", model_cfg.get("dual_branch_classifier_hidden_dim", 32))),
    )
