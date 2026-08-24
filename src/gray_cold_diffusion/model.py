from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


def _groups(channels: int) -> int:
    for group in (8, 4, 2, 1):
        if channels % group == 0:
            return group
    return 1


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int, max_steps: int):
        super().__init__()
        self.dim = dim
        self.max_steps = max(1, max_steps)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        scale = math.log(10000.0) / max(half - 1, 1)
        freq = torch.exp(torch.arange(half, device=t.device, dtype=torch.float32) * -scale)
        phase = (t.float() / self.max_steps).unsqueeze(1) * freq.unsqueeze(0) * 1000.0
        emb = torch.cat((phase.sin(), phase.cos()), dim=1)
        return F.pad(emb, (0, self.dim - emb.shape[1]))


class TimeResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_ch))
        self.norm2 = nn.GroupNorm(_groups(out_ch), out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time(temb)[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class RestorationUNet(nn.Module):
    def __init__(
        self,
        base_channels: int = 32,
        channel_mults: tuple[int, ...] = (1, 2, 4, 8),
        dropout: float = 0.0,
        diffusion_steps: int = 8,
    ):
        super().__init__()
        channels = [base_channels * m for m in channel_mults]
        time_dim = base_channels * 4
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(base_channels, diffusion_steps),
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.stem = nn.Conv2d(3, channels[0], 3, padding=1)

        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        current = channels[0]
        for index, out_ch in enumerate(channels):
            self.down_blocks.append(nn.ModuleList([
                TimeResBlock(current, out_ch, time_dim, dropout),
                TimeResBlock(out_ch, out_ch, time_dim, dropout),
            ]))
            current = out_ch
            if index < len(channels) - 1:
                self.downsamples.append(nn.Conv2d(out_ch, channels[index + 1], 4, stride=2, padding=1))
                current = channels[index + 1]

        self.mid1 = TimeResBlock(channels[-1], channels[-1], time_dim, dropout)
        self.mid2 = TimeResBlock(channels[-1], channels[-1], time_dim, dropout)

        self.upsamples = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        current = channels[-1]
        for index in range(len(channels) - 2, -1, -1):
            out_ch = channels[index]
            self.upsamples.append(nn.Conv2d(current, out_ch, 3, padding=1))
            self.up_blocks.append(nn.ModuleList([
                TimeResBlock(out_ch * 2, out_ch, time_dim, dropout),
                TimeResBlock(out_ch, out_ch, time_dim, dropout),
            ]))
            current = out_ch

        self.out_norm = nn.GroupNorm(_groups(channels[0]), channels[0])
        self.out = nn.Conv2d(channels[0], 3, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        temb = self.time_mlp(t)
        h = self.stem(x)
        skips = []
        for index, blocks in enumerate(self.down_blocks):
            h = blocks[0](h, temb)
            h = blocks[1](h, temb)
            skips.append(h)
            if index < len(self.downsamples):
                h = self.downsamples[index](h)

        h = self.mid2(self.mid1(h, temb), temb)
        for upsample, blocks, skip in zip(self.upsamples, self.up_blocks, reversed(skips[:-1])):
            h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
            h = upsample(h)
            h = torch.cat((h, skip), dim=1)
            h = blocks[0](h, temb)
            h = blocks[1](h, temb)
        return self.out(F.silu(self.out_norm(h)))
