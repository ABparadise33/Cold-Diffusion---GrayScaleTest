import torch

from gray_cold_diffusion.color import normalized_lab_to_rgb, rgb_to_normalized_lab


def test_rgb_lab_round_trip():
    torch.manual_seed(1)
    rgb = torch.rand(2, 3, 16, 16)
    reconstructed = normalized_lab_to_rgb(rgb_to_normalized_lab(rgb))
    assert (rgb - reconstructed).abs().max().item() < 2e-5
