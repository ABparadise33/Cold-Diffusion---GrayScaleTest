import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn
import yaml

from gray_cold_diffusion.bridge import GrayBridge
from gray_cold_diffusion.color import (
    denormalize_rgb, gray_anchor, normalize_rgb, rgb_channel_mean_gray,
)
from gray_cold_diffusion.diagnostics import check_diagnostic_checkpoint, sample_from_step
from gray_cold_diffusion.model import RestorationUNet


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "configs/div2k_uieb_style_lab_sat1_50k.yaml"


@pytest.mark.parametrize("steps,start", [(8, 4), (8, 6), (8, 7), (8, 8), (20, 10), (20, 15), (20, 18), (20, 20)])
def test_partial_lab_path_and_oracle_use_original_time_indices(steps, start):
    torch.manual_seed(42)
    clean = torch.rand(1, 3, 12, 16) - .5
    anchor = gray_anchor(clean)
    bridge = GrayBridge(steps)
    initial = bridge.degrade(clean, anchor, torch.tensor([start]))
    assert torch.allclose(initial[:, :1], clean[:, :1])
    assert torch.allclose(initial[:, 1:], clean[:, 1:] * (1-start/steps), atol=1e-7)
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
    if start < steps:
        analytic = (initial - (start/steps)*anchor) / (1-start/steps)
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


@pytest.mark.parametrize("config_name", [
    "div2k_uieb_style_lab_sat1_50k.yaml", "div2k_lab_sat1_50k.yaml", "div2k_rgb_sat1_50k.yaml",
])
def test_checkpoint_guard_accepts_runtime_batch_configuration(config_name):
    baseline = yaml.safe_load((ROOT / "configs" / config_name).read_text())
    config = copy.deepcopy(baseline)
    config["training"]["batch_size"] = 16
    config["training"]["grad_accum"] = 1
    config["data"]["num_workers"] = 4
    color_space = "rgb" if config["mode"] == "natural_rgb_colorization" else "lab"
    check_diagnostic_checkpoint(config, baseline, 50000, 50000, color_space=color_space)


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


@pytest.mark.parametrize("steps,mode,starts,levels", [
    (8, "cold_gray", [4, 6, 7], ["retain_50pct", "retain_25pct", "retain_12.5pct"]),
    (20, "natural_lab_colorization", [10, 15, 18], ["retain_50pct", "retain_25pct", "retain_10pct"]),
    (20, "natural_rgb_colorization", [10, 15, 18], ["retain_50pct", "retain_25pct", "retain_10pct"]),
])
def test_diagnostic_cli_writes_original_size_subfolders_and_metadata(tmp_path, steps, mode, starts, levels):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.new("RGB", (40, 24), (70, 130, 160)).save(image_dir / "0803.png")
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"test": ["0803.png"]}))
    config = yaml.safe_load(BASELINE.read_text())
    config["mode"] = mode
    config["diffusion"]["steps"] = steps
    config["model"] = {"base_channels": 8, "channel_mults": [1, 2], "dropout": 0.0}
    config_path = tmp_path / "smoke_config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    model = RestorationUNet(8, (1, 2), diffusion_steps=steps)
    checkpoint = tmp_path / "synthetic_checkpoint.pt"
    torch.save({"step": 50000, "config": config, "ema": model.state_dict()}, checkpoint)
    output = tmp_path / "diagnostic"
    output.mkdir()  # An empty pre-created output directory is safe.
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["MPLCONFIGDIR"] = str(tmp_path / "matplotlib")
    color_space = "rgb" if mode == "natural_rgb_colorization" else "lab"
    command = [sys.executable, str(ROOT / f"diagnose_{color_space}_chroma.py"),
               "--checkpoint", str(checkpoint), "--baseline-config", str(config_path),
               "--image-dir", str(image_dir), "--split-file", str(split),
               "--device", "cpu", "--output-dir", str(output), "--include-gray-control"]
    command.extend(["--start-steps", *map(str, starts)])
    result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    metadata = json.loads((output / "run_metadata.json").read_text())
    assert metadata["status"] == "complete"
    assert metadata["checkpoint_step"] == 50000
    assert metadata["diffusion_steps"] == steps
    assert metadata["start_steps"] == starts + [steps]
    assert metadata["selected_names"] == ["0803"]
    assert metadata["color_space"] == color_space
    assert metadata["checkpoint_sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert metadata["diagnostic_source_sha256"] == hashlib.sha256(
        (ROOT / "src/gray_cold_diffusion/chroma_diagnostic.py").read_bytes()
    ).hexdigest()
    assert not list(output.glob("*.png"))
    means = json.loads((output / "metrics.json").read_text())["means_by_start_step"]
    assert means[str(steps)]["analytic_delta_e76"] is None
    for start, level_name in zip(starts + [steps], levels + ["retain_0pct"]):
        level = output / level_name
        for folder in ["inputs", "predictions", "direct_predictions"]:
            with Image.open(level / folder / "0803.png") as image:
                assert image.size == (40, 24)
        with Image.open(level / "trajectories" / "0803.png") as image:
            assert image.size == (40*(start+1), 24+28)
        if color_space == "rgb":
            assert metadata["gray_operator"] == "rgb_channel_mean"
            # Independent RGB calculation: no Lab or luminance-weighted gray.
            source = np.array([70., 130., 160.]) / 255
            expected_rgb = source.mean() + (1-start/steps) * (source-source.mean())
            with Image.open(level / "inputs" / "0803.png") as image:
                assert np.max(np.abs(np.asarray(image)[0, 0] - expected_rgb * 255)) <= 1
        if start < steps:
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
    assert "div2k_lab_sat1_50k/checkpoints/step_050000.pt" in result.stdout
    assert "configs/div2k_lab_sat1_50k.yaml" in result.stdout
    assert "--start-steps 10 15 18" in result.stdout
    assert "div2k_lab_partial_t20_step050000" in result.stdout
    assert "--include-gray-control" not in result.stdout
    assert "best.pt" not in result.stdout


@pytest.mark.parametrize("start", [10, 15, 18, 20])
def test_partial_rgb_path_and_oracle(start):
    torch.manual_seed(42)
    rgb = torch.rand(1, 3, 12, 16)
    clean = normalize_rgb(rgb)
    anchor = rgb_channel_mean_gray(clean)
    bridge = GrayBridge(20)
    initial = bridge.degrade(clean, anchor, torch.tensor([start]))
    gray = rgb.mean(dim=1, keepdim=True).expand_as(rgb)
    expected = gray + (1-start/20) * (rgb-gray)
    assert torch.allclose(denormalize_rgb(initial), expected, atol=1e-7)
    calls = []

    class Oracle(nn.Module):
        def forward(self, x, t):
            calls.append(int(t.item()))
            return clean

    final, path = sample_from_step(Oracle(), bridge, initial, anchor, start)
    assert calls == list(range(start, 0, -1))
    for i, state in enumerate(path):
        expected_state = bridge.degrade(clean, anchor, torch.tensor([start-i]))
        assert torch.allclose(state, expected_state, atol=2e-6)
    assert torch.allclose(denormalize_rgb(final), rgb, atol=1e-6)
    if start == 20:
        expected_final, expected_path = bridge.sample(Oracle(), anchor, return_trajectory=True)
        assert torch.equal(final, expected_final)
        assert all(torch.equal(a, b) for a, b in zip(path, expected_path))


@pytest.mark.parametrize("mode,space", [
    ("natural_rgb_colorization", "lab"), ("natural_lab_colorization", "rgb"),
])
def test_diagnostic_entry_rejects_wrong_color_space_even_with_matching_config(mode, space):
    config = yaml.safe_load(BASELINE.read_text())
    config["mode"] = mode
    with pytest.raises(ValueError, match=f"requires a {space.upper()} model"):
        check_diagnostic_checkpoint(config, config, 50000, 50000, color_space=space)


@pytest.mark.parametrize("factor,step", [(1.25, 50000), (1.5, 50000), (2.0, 50000), (1.0, 15000)])
def test_rgb_guard_refuses_other_factors_or_checkpoint_ages(factor, step):
    baseline = yaml.safe_load((ROOT / "configs/div2k_rgb_sat1_50k.yaml").read_text())
    config = copy.deepcopy(baseline)
    config["data"]["saturation_factor"] = factor
    with pytest.raises(ValueError, match="does not match"):
        check_diagnostic_checkpoint(config, baseline, step, 50000, color_space="rgb")


def test_rgb_wrapper_uses_only_existing_factor1_and_includes_matched_gray_control():
    env = os.environ.copy()
    env["EVAL_DRY_RUN"] = "1"
    env.pop("RGB_DIAGNOSTIC_CHECKPOINT", None)
    result = subprocess.run(["bash", "scripts/diagnose_div2k_rgb_4090.sh"],
                            cwd=ROOT, env=env, capture_output=True, text=True, check=True)
    assert "diagnose_rgb_chroma.py" in result.stdout
    assert "div2k_rgb_sat1_50k/checkpoints/step_050000.pt" in result.stdout
    assert "configs/div2k_rgb_sat1_50k.yaml" in result.stdout
    assert "--expected-checkpoint-step 50000" in result.stdout
    assert "--limit 4" in result.stdout
    assert "--start-steps 10 15 18" in result.stdout
    assert "--include-gray-control" in result.stdout
    assert "div2k_rgb_sat_1.00x_partial_t20_step050000" in result.stdout
    assert "best.pt" not in result.stdout
    assert "train.py" not in result.stdout


def test_rgb_and_lab_select_the_same_seed42_subset():
    import random

    names = json.loads((ROOT / "splits/div2k_valid_all.json").read_text())["test"]
    selected = sorted(random.Random(42).sample(range(len(names)), 4))
    assert [Path(names[i]).stem for i in selected] == ["0804", "0815", "0882", "0895"]
