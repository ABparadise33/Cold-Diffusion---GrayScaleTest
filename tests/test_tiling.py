import pytest
import torch
from torch import nn

from gray_cold_diffusion.tiling import TiledModel


class AddTime(nn.Module):
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return x + t[:, None, None, None].to(x.dtype) / 10.0


def test_tiled_model_preserves_values_and_rectangular_geometry():
    image = torch.rand(1, 3, 73, 91)
    timestep = torch.tensor([4])
    expected = AddTime()(image, timestep)
    actual = TiledModel(AddTime(), tile_size=32, overlap=8)(image, timestep)

    assert actual.shape == image.shape
    assert torch.allclose(actual, expected, atol=1e-6)


def test_tiled_model_delegates_when_image_fits_one_tile():
    image = torch.rand(1, 3, 20, 24)
    timestep = torch.tensor([2])
    actual = TiledModel(AddTime(), tile_size=32, overlap=8)(image, timestep)
    assert torch.equal(actual, AddTime()(image, timestep))


@pytest.mark.parametrize("overlap", [-1, 32])
def test_tiled_model_rejects_invalid_overlap(overlap: int):
    with pytest.raises(ValueError):
        TiledModel(AddTime(), tile_size=32, overlap=overlap)
