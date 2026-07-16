import math

import torch
import torch.nn.functional as F
from torch import nn


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half = self.dim // 2
        freq = torch.exp(-math.log(10000) / (half - 1) * torch.arange(half, device=device, dtype=torch.float32))
        angles = t[:, None].float() * freq[None, :]
        emb = torch.cat((angles.sin(), angles.cos()), dim=-1)
        if self.dim % 2:
            emb = F.pad(emb, (0, 1))
        return emb


class Conv2dBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, n_groups=8):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )


class Downsample2d(nn.Conv2d):
    def __init__(self, dim):
        super().__init__(dim, dim, 3, stride=2, padding=1)


class Upsample2d(nn.ConvTranspose2d):
    def __init__(self, dim):
        super().__init__(dim, dim, 4, stride=2, padding=1)


class ConditionalResidualBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim, kernel_size=3, n_groups=8, cond_predict_scale=False):
        super().__init__()
        self.conv1 = Conv2dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups)
        self.conv2 = Conv2dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups)
        self.cond_predict_scale = cond_predict_scale
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, out_channels * 2 if cond_predict_scale else out_channels),
        )
        self.residual = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, c):
        h = self.conv1(x)
        e = self.cond_encoder(c)[..., None, None]
        if self.cond_predict_scale:
            s, b = e.chunk(2, dim=1)
            h = s * h + b
        else:
            h = h + e
        return self.conv2(h) + self.residual(x)


class ConditionalUnet2D(nn.Module):
    def __init__(
        self,
        horizon_steps=16,
        cond_channels=2,
        diffusion_step_embed_dim=256,
        down_dims=(64, 128, 256),
        kernel_size=3,
        n_groups=8,
        cond_predict_scale=False,
    ):
        super().__init__()
        self.horizon_steps = horizon_steps
        dims = [horizon_steps + cond_channels] + list(down_dims)
        d_emb = diffusion_step_embed_dim
        cps = cond_predict_scale

        self.time_encoder = nn.Sequential(
            SinusoidalPosEmb(d_emb), nn.Linear(d_emb, d_emb * 4), nn.Mish(), nn.Linear(d_emb * 4, d_emb)
        )

        self.mid = nn.ModuleList(
            [ConditionalResidualBlock2D(dims[-1], dims[-1], d_emb, kernel_size, n_groups, cps) for _ in range(2)]
        )

        in_out = list(zip(dims[:-1], dims[1:]))
        self.down = nn.ModuleList()
        for i, (c_in, c_out) in enumerate(in_out):
            last = i == len(dims) - 2
            self.down.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock2D(c_in, c_out, d_emb, kernel_size, n_groups, cps),
                        ConditionalResidualBlock2D(c_out, c_out, d_emb, kernel_size, n_groups, cps),
                        Downsample2d(c_out) if not last else nn.Identity(),
                    ]
                )
            )

        self.up = nn.ModuleList()
        for i, (c_out, c_in) in enumerate(reversed(in_out[1:])):
            self.up.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock2D(c_in * 2, c_out, d_emb, kernel_size, n_groups, cps),
                        ConditionalResidualBlock2D(c_out, c_out, d_emb, kernel_size, n_groups, cps),
                        Upsample2d(c_out) if i < len(dims) - 2 else nn.Identity(),
                    ]
                )
            )

        self.final = nn.Sequential(Conv2dBlock(dims[1], dims[1], kernel_size), nn.Conv2d(dims[1], horizon_steps, 1))

    def forward(self, x, t, cond):
        x = torch.cat([x, cond], dim=1)
        t = self.time_encoder(t.expand(x.shape[0]).to(x.device, dtype=torch.long))

        skips = []
        for b1, b2, down in self.down:
            x = b2(b1(x, t), t)
            skips.append(x)
            x = down(x)
        for m in self.mid:
            x = m(x, t)
        for b1, b2, up in self.up:
            x = b2(b1(torch.cat([x, skips.pop()], dim=1), t), t)
            x = up(x)
        return self.final(x)
