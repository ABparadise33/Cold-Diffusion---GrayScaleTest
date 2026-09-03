from pathlib import Path

from PIL import Image
import torch

from gray_cold_diffusion.io import save_stage_strip, save_trajectory_grid


def test_stage_and_trajectory_strips_are_horizontal(tmp_path: Path):
    images = torch.rand(1, 3, 12, 16)
    comparison_path = tmp_path / "comparison.png"
    save_stage_strip(
        [("raw", images), ("gray", images), ("prediction", images), ("reference", images)],
        comparison_path,
        display_scale=2,
    )
    with Image.open(comparison_path) as comparison:
        assert comparison.width == 4 * 16 * 2
        assert comparison.width > comparison.height

    trajectory_path = tmp_path / "trajectory.png"
    lab_states = [torch.rand(1, 3, 12, 16) * 2 - 1 for _ in range(9)]
    save_trajectory_grid(lab_states, trajectory_path, display_scale=2)
    with Image.open(trajectory_path) as trajectory:
        assert trajectory.width == 9 * 16 * 2
        assert trajectory.width > trajectory.height

    rgb_trajectory_path = tmp_path / "rgb_trajectory.png"
    rgb_states = [torch.rand(1, 3, 12, 16) * 2 - 1 for _ in range(5)]
    save_trajectory_grid(
        rgb_states,
        rgb_trajectory_path,
        display_scale=2,
        color_space="rgb",
    )
    with Image.open(rgb_trajectory_path) as trajectory:
        assert trajectory.width == 5 * 16 * 2


def test_preview_max_side_preserves_aspect_ratio(tmp_path: Path):
    images = torch.rand(1, 3, 60, 120)
    path = tmp_path / "comparison_small.png"
    save_stage_strip(
        [("raw", images), ("prediction", images)],
        path,
        max_side=30,
    )
    with Image.open(path) as comparison:
        assert comparison.width == 60
        assert comparison.height == 15 + 28
