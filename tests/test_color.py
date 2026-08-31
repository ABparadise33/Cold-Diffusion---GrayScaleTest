import pytest
import torch

from gray_cold_diffusion.color import (
    adjust_saturation_from_channel_mean,
    normalized_lab_to_rgb,
    rgb_channel_mean_gray,
    rgb_to_normalized_lab,
)


def test_rgb_lab_round_trip():
    torch.manual_seed(1)
    rgb = torch.rand(2, 3, 16, 16)
    reconstructed = normalized_lab_to_rgb(rgb_to_normalized_lab(rgb))
    assert (rgb - reconstructed).abs().max().item() < 2e-5


def test_channel_mean_saturation_has_interpretable_endpoints():
    rgb = torch.tensor([[[[0.4]], [[0.5]], [[0.6]]]])
    gray = rgb_channel_mean_gray(rgb)
    unchanged = adjust_saturation_from_channel_mean(rgb, 1.0)
    stronger = adjust_saturation_from_channel_mean(rgb, 1.5)

    assert torch.allclose(gray, torch.full_like(rgb, 0.5))
    assert torch.allclose(unchanged, rgb)
    assert torch.allclose(stronger, torch.tensor([[[[0.35]], [[0.5]], [[0.65]]]]))
    assert torch.allclose(stronger.mean(dim=1), rgb.mean(dim=1))


def test_negative_saturation_is_rejected():
    with pytest.raises(ValueError, match="saturation factor"):
        adjust_saturation_from_channel_mean(torch.rand(1, 3, 2, 2), -0.1)
