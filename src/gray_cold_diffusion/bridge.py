from __future__ import annotations

import torch
from torch import nn


class GrayBridge(nn.Module):
    """Linear paired bridge from a reference Lab image to gray(raw)."""

    def __init__(self, steps: int = 8):
        super().__init__()
        if steps < 1:
            raise ValueError("steps must be >= 1")
        self.steps = int(steps)
        self.register_buffer("schedule", torch.linspace(0.0, 1.0, self.steps + 1))

    def alpha(self, t: torch.Tensor, ndim: int = 4) -> torch.Tensor:
        t = t.long().clamp(0, self.steps)
        shape = (t.shape[0],) + (1,) * (ndim - 1)
        return self.schedule[t].view(shape)

    def degrade(self, clean_candidate: torch.Tensor, anchor: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        a = self.alpha(t, clean_candidate.ndim)
        return (1.0 - a) * clean_candidate + a * anchor

    @torch.no_grad()
    def sample(self, model: nn.Module, anchor: torch.Tensor, return_trajectory: bool = False):
        x = anchor.clone()
        trajectory = [x.clone()] if return_trajectory else None
        batch = x.shape[0]
        for step in range(self.steps, 0, -1):
            t = torch.full((batch,), step, device=x.device, dtype=torch.long)
            pred_clean = model(x, t)
            current = self.degrade(pred_clean, anchor, t)
            previous = self.degrade(pred_clean, anchor, t - 1)
            x = (x - current + previous).clamp(-1.0, 1.0)
            if trajectory is not None:
                trajectory.append(x.clone())
        return (x, trajectory) if return_trajectory else x
