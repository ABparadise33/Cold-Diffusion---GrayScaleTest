import copy
import json
import os
from pathlib import Path
import subprocess
import sys

from PIL import Image
import pytest
import torch
from torch import nn
import yaml

from gray_cold_diffusion.bridge import GrayBridge
from gray_cold_diffusion.color import gray_anchor
from gray_cold_diffusion.diagnostics import check_diagnostic_checkpoint, sample_from_step
from gray_cold_diffusion.model import RestorationUNet


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "configs/div2k_uieb_style_lab_sat1_50k.yaml"


@pytest.mark.parametrize("start", [4, 6, 7, 8])
def test_partial_lab_path_and_oracle_use_original_time_indices(start):
    torch.manual_seed(42)
    clean = torch.rand(1, 3, 12, 16) - .5
    anchor = gray_anchor(clean)
    bridge = GrayBridge(8)
    initial = bridge.degrade(clean, anchor, torch.tensor([start]))
    assert torch.allclose(initial[:, :1], clean[:, :1])
    assert torch.allclose(initial[:, 1:], clean[:, 1:] * (1-start/8))
    calls = []

    class Oracle(nn.Module):
        def forward(self, x, t):
            calls.append(int(t.item()))
            return clean

    result, trajectory = sample_from_step(Oracle(), bridge, initial, anchor, start)
    assert calls == list(range(start, 0, -1))
    assert len(trajectory) == start+1
    assert torch.equal(trajectory[0], initial)
    assert torch.allclose(result, clean, atol=1e-6)
    if start < 8:
        analytic = (initial - (start/8)*anchor) / (1-start/8)
        assert torch.allclose(analytic, clean, atol=1e-6)


def test_full_gray_entry_is_identical_to_existing_algorithm2():
    bridge = GrayBridge(8)
    anchor = gray_anchor(torch.rand(1, 3, 12, 16) - .5)

    class ToyModel(nn.Module):
        def forward(self, x, t):
            return x * .75 + t[:, None, None, None] * .1

    model = ToyModel()
    expected, expected_path = bridge.sample(model, anchor, return_trajectory=True)
    actual, actual_path = sample_from_step(model, bridge, anchor, anchor, 8)
    assert torch.equal(actual, expected)
    assert all(torch.equal(a, b) for a, b in zip(actual_path, expected_path))


@pytest.mark.parametrize("start", [0, 9])
def test_partial_sampler_rejects_invalid_step(start):
    with pytest.raises(ValueError, match="start_step"):
        sample_from_step(nn.Identity(), GrayBridge(8), torch.zeros(1, 3, 8, 8), torch.zeros(1, 3, 8, 8), start)


def test_checkpoint_guard_accepts_runtime_batch_configuration():
    baseline = yaml.safe_load(BASELINE.read_text())
    config = copy.deepcopy(baseline)
    config["training"]["batch_size"] = 16
    config["training"]["grad_accum"] = 1
    config["data"]["num_workers"] = 4
    check_diagnostic_checkpoint(config, baseline, 50000, 50000)


@pytest.mark.parametrize("mismatch", ["step", "timesteps", "mode", "dataset", "saturation"])
def test_checkpoint_guard_refuses_nonmatching_experiments(mismatch):
    baseline = yaml.safe_load(BASELINE.read_text())
    config = copy.deepcopy(baseline)
    step = 50000
    if mismatch == "step":
        step = 15000
    elif mismatch == "timesteps":
        config["diffusion"]["steps"] = 20
    elif mismatch == "mode":
        config["mode"] = "natural_lab_colorization"
    elif mismatch == "dataset":
        config["experiment"] = "uieb_lab_retrain_50k"
    else:
        config["data"]["reference_saturation_factor"] = 1.5
    with pytest.raises(ValueError, match="does not match"):
        check_diagnostic_checkpoint(config, baseline, step, 50000)


def test_diagnostic_cli_writes_original_size_subfolders_and_metadata(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.new("RGB", (40, 24), (70, 130, 160)).save(image_dir / "0803.png")
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"test": ["0803.png"]}))
    config = yaml.safe_load(BASELINE.read_text())
    config["model"] = {"base_channels": 8, "channel_mults": [1, 2], "dropout": 0.0}
    config_path = tmp_path / "smoke_config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    model = RestorationUNet(8, (1, 2), diffusion_steps=8)
    checkpoint = tmp_path / "synthetic_checkpoint.pt"
    torch.save({"step": 50000, "config": config, "ema": model.state_dict()}, checkpoint)
    output = tmp_path / "diagnostic"
    output.mkdir()  # An empty pre-created output directory is safe.
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["MPLCONFIGDIR"] = str(tmp_path / "matplotlib")
    command = [sys.executable, str(ROOT / "diagnose_lab_chroma.py"),
               "--checkpoint", str(checkpoint), "--baseline-config", str(config_path),
               "--image-dir", str(image_dir), "--split-file", str(split),
               "--device", "cpu", "--output-dir", str(output), "--include-gray-control"]
    result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    metadata = json.loads((output / "run_metadata.json").read_text())
    assert metadata["status"] == "complete"
    assert metadata["checkpoint_step"] == 50000
    assert metadata["start_steps"] == [4, 6, 7, 8]
    assert metadata["selected_names"] == ["0803"]
    assert not list(output.glob("*.png"))
    means = json.loads((output / "metrics.json").read_text())["means_by_start_step"]
    assert means["8"]["analytic_delta_e76"] is None
    for start, level_name in [(4, "retain_50pct"), (6, "retain_25pct"), (7, "retain_12.5pct"), (8, "retain_0pct")]:
        level = output / level_name
        for folder in ["inputs", "predictions", "direct_predictions"]:
            with Image.open(level / folder / "0803.png") as image:
                assert image.size == (40, 24)
        with Image.open(level / "trajectories" / "0803.png") as image:
            assert image.size == (40*(start+1), 24+28)
        if start < 8:
            assert means[str(start)]["analytic_delta_e76"] < .01
    # Never overwrite an existing diagnostic run silently.
    repeated = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=60)
    assert repeated.returncode != 0
    assert "not empty" in repeated.stderr


def test_diagnostic_wrapper_defaults_to_verified_50k_partial_starts():
    env = os.environ.copy()
    env["EVAL_DRY_RUN"] = "1"
    result = subprocess.run(["bash", "scripts/diagnose_div2k_lab_4090.sh"],
                            cwd=ROOT, env=env, capture_output=True, text=True, check=True)
    assert "step_050000.pt" in result.stdout
    assert "--expected-checkpoint-step 50000" in result.stdout
    assert "--start-steps 4 6 7" in result.stdout
    assert "--include-gray-control" not in result.stdout
    assert "best.pt" not in result.stdout
