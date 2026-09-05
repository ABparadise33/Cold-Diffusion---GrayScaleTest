import json
import os
from pathlib import Path
import subprocess
import sys

from PIL import Image
import pytest
import torch

from gray_cold_diffusion.model import RestorationUNet


@pytest.mark.parametrize("mode", [
    "cold_gray", "natural_rgb_colorization", "natural_lab_colorization", "gray_oneshot",
])
def test_evaluate_groups_previews_without_changing_geometry(tmp_path: Path, mode: str):
    root = Path(__file__).resolve().parents[1]
    data = tmp_path / "data"
    data.mkdir()
    width, height = 40, 24
    Image.new("RGB", (width, height), (50, 100, 150)).save(data / "sample.png")
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"test": ["sample.png"]}), encoding="utf-8")
    config = {
        "mode": mode,
        "model": {"base_channels": 8, "channel_mults": [1, 2]},
        "diffusion": {"steps": 2},
        "data": {"image_size": 16},
    }
    model = RestorationUNet(8, (1, 2), diffusion_steps=2)
    checkpoint = tmp_path / "model.pt"
    torch.save({"config": config, "ema": model.state_dict(), "step": 2}, checkpoint)
    output = tmp_path / "evaluation"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["MPLCONFIGDIR"] = str(tmp_path / "matplotlib")
    result = subprocess.run([
        sys.executable, str(root / "evaluate.py"),
        "--checkpoint", str(checkpoint), "--raw-dir", str(data),
        "--reference-dir", str(data), "--split-file", str(split),
        "--device", "cpu", "--original-size", "--batch-size", "1",
        "--output-dir", str(output),
    ], cwd=root, env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert not list(output.glob("batch_*.png"))
    assert not list(output.glob("trajectory_*.png"))
    assert json.loads((output / "metrics.json").read_text())["evaluation"]["num_images"] == 1
    iterative = mode != "gray_oneshot"
    with Image.open(output / "batches" / "batch_000.png") as strip:
        assert strip.size == (width * 4, height + 28)
    assert not (output / 'direct_predictions').exists()
    trajectory = output / "trajectories" / "trajectory_000.png"
    if iterative:
        with Image.open(trajectory) as strip:
            assert strip.size == (width * 3, height + 28)
    else:
        assert not trajectory.exists()
