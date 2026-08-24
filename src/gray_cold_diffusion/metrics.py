from __future__ import annotations

import torch
import torch.nn.functional as F

from .color import lab_denormalize


def psnr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = (pred - target).square().flatten(1).mean(1).clamp_min(1e-10)
    return -10.0 * torch.log10(mse)


def ssim(pred: torch.Tensor, target: torch.Tensor, window: int = 11) -> torch.Tensor:
    """Compact SSIM using a uniform local window; returns one value per image."""
    padding = window // 2
    mu_x = F.avg_pool2d(pred, window, stride=1, padding=padding)
    mu_y = F.avg_pool2d(target, window, stride=1, padding=padding)
    sigma_x = F.avg_pool2d(pred * pred, window, 1, padding) - mu_x.square()
    sigma_y = F.avg_pool2d(target * target, window, 1, padding) - mu_y.square()
    sigma_xy = F.avg_pool2d(pred * target, window, 1, padding) - mu_x * mu_y
    c1, c2 = 0.01**2, 0.03**2
    score = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2)
    )
    return score.flatten(1).mean(1)


def delta_e76(pred_lab_norm: torch.Tensor, target_lab_norm: torch.Tensor) -> torch.Tensor:
    pred = lab_denormalize(pred_lab_norm)
    target = lab_denormalize(target_lab_norm)
    return (pred - target).square().sum(1).sqrt().flatten(1).mean(1)


def trajectory_monotonic_fraction(trajectory: list[torch.Tensor], target_lab: torch.Tensor) -> torch.Tensor:
    errors = torch.stack([delta_e76(state, target_lab) for state in trajectory], dim=1)
    improved = errors[:, 1:] <= errors[:, :-1] + 1e-6
    return improved.float().mean(1)
