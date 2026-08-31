from __future__ import annotations

import torch


def _matrix(values: list[list[float]], ref: torch.Tensor) -> torch.Tensor:
    return torch.tensor(values, device=ref.device, dtype=ref.dtype)


def normalize_rgb(rgb: torch.Tensor) -> torch.Tensor:
    """Map RGB [0,1] to the model range [-1,1]."""
    return rgb * 2.0 - 1.0


def denormalize_rgb(rgb: torch.Tensor) -> torch.Tensor:
    """Map model RGB [-1,1] back to display RGB [0,1]."""
    return ((rgb + 1.0) * 0.5).clamp(0.0, 1.0)


def rgb_channel_mean_gray(rgb: torch.Tensor) -> torch.Tensor:
    """Paper-style grayscale: repeat the per-pixel mean of R, G, and B."""
    gray = rgb.mean(dim=1, keepdim=True)
    return gray.expand_as(rgb)


def adjust_saturation_from_channel_mean(rgb: torch.Tensor, factor: float) -> torch.Tensor:
    """Expand RGB chroma around the same channel-mean gray axis used by the bridge.

    factor=0 is grayscale, factor=1 is the unmodified image, and factor=1.5
    increases each channel's distance from gray by 50%. Values outside the sRGB
    gamut are clipped to [0,1].
    """
    if factor < 0:
        raise ValueError("saturation factor must be >= 0")
    gray = rgb_channel_mean_gray(rgb)
    return (gray + float(factor) * (rgb - gray)).clamp(0.0, 1.0)


def lab_normalize(lab: torch.Tensor) -> torch.Tensor:
    """Map L in [0,100], a/b approximately [-128,127] to roughly [-1,1]."""
    lightness = lab[:, 0:1] / 50.0 - 1.0
    ab = lab[:, 1:3] / 128.0
    return torch.cat((lightness, ab), dim=1)


def lab_denormalize(lab: torch.Tensor) -> torch.Tensor:
    lightness = (lab[:, 0:1] + 1.0) * 50.0
    ab = lab[:, 1:3] * 128.0
    return torch.cat((lightness, ab), dim=1)


def rgb_to_lab(rgb: torch.Tensor) -> torch.Tensor:
    """Differentiable sRGB [0,1] to CIE Lab (D65)."""
    rgb = rgb.clamp(0.0, 1.0)
    linear = torch.where(
        rgb > 0.04045,
        ((rgb + 0.055) / 1.055).pow(2.4),
        rgb / 12.92,
    )
    m = _matrix(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        rgb,
    )
    xyz = torch.einsum("ij,bjhw->bihw", m, linear)
    white = torch.tensor([0.95047, 1.0, 1.08883], device=rgb.device, dtype=rgb.dtype)
    xyz = xyz / white.view(1, 3, 1, 1)

    delta = 6.0 / 29.0
    f = torch.where(
        xyz > delta**3,
        xyz.clamp_min(0).pow(1.0 / 3.0),
        xyz / (3.0 * delta**2) + 4.0 / 29.0,
    )
    fx, fy, fz = f[:, 0:1], f[:, 1:2], f[:, 2:3]
    return torch.cat((116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)), dim=1)


def lab_to_rgb(lab: torch.Tensor) -> torch.Tensor:
    """Differentiable CIE Lab (D65) to sRGB [0,1]."""
    lightness = lab[:, 0:1]
    a_channel = lab[:, 1:2]
    b_channel = lab[:, 2:3]
    fy = (lightness + 16.0) / 116.0
    fx = fy + a_channel / 500.0
    fz = fy - b_channel / 200.0
    f = torch.cat((fx, fy, fz), dim=1)

    delta = 6.0 / 29.0
    xyz = torch.where(f > delta, f.pow(3.0), 3.0 * delta**2 * (f - 4.0 / 29.0))
    white = torch.tensor([0.95047, 1.0, 1.08883], device=lab.device, dtype=lab.dtype)
    xyz = xyz * white.view(1, 3, 1, 1)

    m = _matrix(
        [
            [3.2404542, -1.5371385, -0.4985314],
            [-0.9692660, 1.8760108, 0.0415560],
            [0.0556434, -0.2040259, 1.0572252],
        ],
        lab,
    )
    linear = torch.einsum("ij,bjhw->bihw", m, xyz)
    rgb = torch.where(
        linear > 0.0031308,
        1.055 * linear.clamp_min(0).pow(1.0 / 2.4) - 0.055,
        12.92 * linear,
    )
    return rgb.clamp(0.0, 1.0)


def rgb_to_normalized_lab(rgb: torch.Tensor) -> torch.Tensor:
    return lab_normalize(rgb_to_lab(rgb))


def normalized_lab_to_rgb(lab: torch.Tensor) -> torch.Tensor:
    return lab_to_rgb(lab_denormalize(lab))


def gray_anchor(raw_lab: torch.Tensor) -> torch.Tensor:
    """Keep raw luminance and remove all Lab chroma."""
    return torch.cat((raw_lab[:, 0:1], torch.zeros_like(raw_lab[:, 1:3])), dim=1)
