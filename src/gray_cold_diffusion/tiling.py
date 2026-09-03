from __future__ import annotations

import torch
from torch import nn


def _tile_starts(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size + 1, stride))
    final = length - tile_size
    if starts[-1] != final:
        starts.append(final)
    return starts


def _blend_weight(
    height: int,
    width: int,
    overlap: int,
    *,
    top_edge: bool,
    bottom_edge: bool,
    left_edge: bool,
    right_edge: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    y_weight = torch.ones(height, device=device, dtype=dtype)
    x_weight = torch.ones(width, device=device, dtype=dtype)
    y_overlap = min(overlap, height)
    x_overlap = min(overlap, width)
    if y_overlap:
        ramp = torch.linspace(
            1.0 / (y_overlap + 1), 1.0, y_overlap, device=device, dtype=dtype
        )
        if not top_edge:
            y_weight[:y_overlap] = ramp
        if not bottom_edge:
            y_weight[-y_overlap:] = ramp.flip(0)
    if x_overlap:
        ramp = torch.linspace(
            1.0 / (x_overlap + 1), 1.0, x_overlap, device=device, dtype=dtype
        )
        if not left_edge:
            x_weight[:x_overlap] = ramp
        if not right_edge:
            x_weight[-x_overlap:] = ramp.flip(0)
    return y_weight[:, None] * x_weight[None, :]


class TiledModel(nn.Module):
    """Run a fully convolutional model on overlapping tiles and feather-stitch it."""

    def __init__(self, model: nn.Module, tile_size: int, overlap: int = 64):
        super().__init__()
        self.model = model
        self.tile_size = int(tile_size)
        self.overlap = int(overlap)
        if self.tile_size < 1:
            raise ValueError("tile_size must be >= 1")
        if self.overlap < 0 or self.overlap >= self.tile_size:
            raise ValueError("overlap must satisfy 0 <= overlap < tile_size")

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[0] != 1:
            raise ValueError("tiled inference requires a BCHW tensor with batch size 1")
        if t.ndim != 1 or t.shape[0] != 1:
            raise ValueError("tiled inference requires one timestep")

        height, width = x.shape[-2:]
        if height <= self.tile_size and width <= self.tile_size:
            return self.model(x, t)

        stride = self.tile_size - self.overlap
        y_starts = _tile_starts(height, self.tile_size, stride)
        x_starts = _tile_starts(width, self.tile_size, stride)
        output = None
        weight_sum = None
        for top in y_starts:
            bottom = min(top + self.tile_size, height)
            for left in x_starts:
                right = min(left + self.tile_size, width)
                prediction = self.model(x[..., top:bottom, left:right], t)
                if prediction.shape[-2:] != (bottom - top, right - left):
                    raise RuntimeError("tile prediction changed spatial dimensions")
                if output is None:
                    output = prediction.new_zeros(
                        (1, prediction.shape[1], height, width)
                    )
                    weight_sum = prediction.new_zeros((1, 1, height, width))
                weight = _blend_weight(
                    prediction.shape[-2],
                    prediction.shape[-1],
                    self.overlap,
                    top_edge=top == 0,
                    bottom_edge=bottom == height,
                    left_edge=left == 0,
                    right_edge=right == width,
                    device=prediction.device,
                    dtype=prediction.dtype,
                )[None, None]
                output[..., top:bottom, left:right] += prediction * weight
                weight_sum[..., top:bottom, left:right] += weight

        if output is None or weight_sum is None:
            raise RuntimeError("tiled inference produced no tiles")
        return output / weight_sum.clamp_min(torch.finfo(output.dtype).eps)
