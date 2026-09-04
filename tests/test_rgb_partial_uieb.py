import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest
import torch
import yaml

from gray_cold_diffusion import chroma_diagnostic, chroma_sweep
from gray_cold_diffusion.diagnostics import check_diagnostic_checkpoint
from gray_cold_diffusion.model import RestorationUNet


ROOT = Path(__file__).resolve().parents[1]


def small_config(factor=1.):
    experiment = chroma_sweep.EXPERIMENTS[factor]
    config = yaml.safe_load((ROOT / "configs" / f"{experiment}.yaml").read_text())
    config["model"] = {"base_channels": 8, "channel_mults": [1, 2], "dropout": 0.0}
    return config


def write_checkpoint(root, factor, weights):
    config = small_config(factor)
    config_path = root / f"config_{factor}.yaml"
    config_path.write_text(yaml.safe_dump(config))
    checkpoint = root / f"model_{factor}.pt"
    torch.save({"step": 50000, "config": config, "ema": weights}, checkpoint)
    return config_path, checkpoint


def run_cli(command, tmp_path):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["MPLCONFIGDIR"] = str(tmp_path / "mpl")
    return subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=60)


@pytest.mark.parametrize("factor", [1., 1.25, 1.5, 2.])
def test_explicit_factor_guard_matches_both_checkpoint_and_baseline(factor):
    config = small_config(factor)
    check_diagnostic_checkpoint(config, config, 50000, 50000, color_space="rgb", expected_saturation_factor=factor)
    other = copy.deepcopy(config)
    other["data"]["saturation_factor"] = 2. if factor != 2. else 1.
    for actual, baseline in [(other, config), (config, other)]:
        with pytest.raises(ValueError, match="saturation_factor"):
            check_diagnostic_checkpoint(actual, baseline, 50000, 50000,
                                        color_space="rgb", expected_saturation_factor=factor)


def test_paired_inputs_outputs_do_not_depend_on_gt_or_saturation_flag(tmp_path):
    raw, reference = tmp_path / "raw", tmp_path / "reference"
    raw.mkdir()
    reference.mkdir()
    names = ["a.png", "b.png"]
    for index, name in enumerate(names):
        Image.new("RGB", (40+index*8, 24), (40, 125, 175)).save(raw / name)
        Image.new("RGB", (40+index*8, 24), (150, 115, 90)).save(reference / name)
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"test": names}))
    torch.manual_seed(42)
    weights = RestorationUNet(8, (1, 2), diffusion_steps=20).state_dict()
    outputs = []
    for run, factor in enumerate([1., 2., 1.]):
        if run == 2:
            # Change only GT. The actual model inputs and predictions must be bit-identical.
            for index, name in enumerate(names):
                Image.new("RGB", (40+index*8, 24), (210, 30, 20)).save(reference / name)
        config, checkpoint = write_checkpoint(tmp_path, factor, weights)
        output = tmp_path / f"result_{run}"
        command = [sys.executable, str(ROOT / "diagnose_rgb_chroma.py"),
                   "--checkpoint", str(checkpoint), "--baseline-config", str(config),
                   "--expected-saturation-factor", str(factor), "--raw-dir", str(raw),
                   "--reference-dir", str(reference), "--split-file", str(split),
                   "--output-dir", str(output), "--start-steps", "15", "--include-gray-control",
                   "--preview-count", "1", "--device", "cpu"]
        result = run_cli(command, tmp_path)
        assert result.returncode == 0, result.stdout + result.stderr
        metrics = json.loads((output / "metrics.json").read_text())
        assert metrics["evaluation"]["task"] == "paired_underwater_transfer"
        assert metrics["evaluation"]["num_images"] == 2  # Paired default does not limit to four.
        assert metrics["means_by_start_step"]["15"]["analytic_raw_delta_e76"] < .01
        assert metrics["means_by_start_step"]["15"]["analytic_delta_e76"] > 1
        assert metrics["means_by_start_step"]["20"]["analytic_raw_delta_e76"] is None
        assert len(list((output / "retain_25pct/predictions").glob("*.png"))) == 2
        assert len(list((output / "retain_25pct/trajectories").glob("*.png"))) == 1
        with Image.open(output / "retain_25pct/inputs/a.png") as image:
            actual = np.asarray(image)[0, 0]
        source = np.array([40., 125., 175.])
        expected = source.mean() + .25*(source-source.mean())
        assert np.max(np.abs(actual-expected)) <= .5
        outputs.append((output, metrics))
    base, base_metrics = outputs[0]
    for output, metrics in outputs[1:]:
        assert metrics["evaluation"]["input_sha256"] == base_metrics["evaluation"]["input_sha256"]
        for level in ["retain_25pct", "retain_0pct"]:
            for folder in ["inputs", "direct_predictions", "predictions", "trajectories"]:
                for path in (base / level / folder).glob("*.png"):
                    assert path.read_bytes() == (output / level / folder / path.name).read_bytes()
    assert outputs[2][1]["evaluation"]["reference_sha256"] != base_metrics["evaluation"]["reference_sha256"]
    assert outputs[2][1]["means_by_start_step"]["15"]["algorithm2_delta_e76"] != base_metrics["means_by_start_step"]["15"]["algorithm2_delta_e76"]
    chroma_sweep.validate_comparable_reports([outputs[0][1], outputs[1][1]])
    broken = copy.deepcopy(outputs[1][1])
    broken["evaluation"]["input_sha256"]["a"]["15"] = "different input"
    with pytest.raises(ValueError, match="input_sha256"):
        chroma_sweep.validate_comparable_reports([base_metrics, broken])


def test_small_four_factor_sweep_end_to_end(tmp_path, monkeypatch):
    # Real subprocess inference with tiny test weights, not real 50k models.
    temp_repo = tmp_path / "repo"
    for folder in ["configs", "splits", "raw", "reference"]:
        (temp_repo / folder).mkdir(parents=True)
    split_items = json.loads((ROOT / "splits/uieb_seed42.json").read_text())["test"]
    (temp_repo / "splits/uieb_seed42.json").write_text(json.dumps({"test": split_items}))
    for item in split_items:
        raw_name, ref_name = (item, item) if isinstance(item, str) else (item["raw"], item["reference"])
        Image.new("RGB", (32, 24), (40, 125, 175)).save(temp_repo / "raw" / raw_name)
        Image.new("RGB", (32, 24), (150, 115, 90)).save(temp_repo / "reference" / ref_name)
    (temp_repo / "diagnose_rgb_chroma.py").write_text(
        "from gray_cold_diffusion.chroma_diagnostic import main\nmain(color_space='rgb')\n"
    )
    weights = RestorationUNet(8, (1, 2), diffusion_steps=20).state_dict()
    for factor, experiment in chroma_sweep.EXPERIMENTS.items():
        config = small_config(factor)
        (temp_repo / "configs" / f"{experiment}.yaml").write_text(yaml.safe_dump(config))
        checkpoint = temp_repo / "outputs" / experiment / "checkpoints/step_050000.pt"
        checkpoint.parent.mkdir(parents=True)
        torch.save({"step": 50000, "config": config, "ema": weights}, checkpoint)
    output = tmp_path / "sweep"
    argv = ["evaluate_rgb_partial_uieb.py", "--checkpoint-root", str(temp_repo / "outputs"),
            "--raw-dir", str(temp_repo / "raw"), "--reference-dir", str(temp_repo / "reference"),
            "--split-file", str(temp_repo / "splits/uieb_seed42.json"),
            "--output-dir", str(output), "--limit", "1", "--preview-count", "1",
            "--device", "cpu", "--skip-extended-metrics"]
    monkeypatch.setattr(chroma_sweep, "ROOT", temp_repo)
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setenv("PYTHONPATH", str(ROOT / "src"))
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("MKL_NUM_THREADS", "1")
    chroma_sweep.main()
    summary = json.loads((output / "saturation_comparison.json").read_text())
    assert summary["input_and_reference_hashes_match"]
    assert len(summary["rows"]) == 8
    assert {row["saturation_factor"] for row in summary["rows"]} == {1., 1.25, 1.5, 2.}
    assert {row["start_step"] for row in summary["rows"]} == {15}
    for method in ["direct", "algorithm2"]:
        images = list((output / "comparisons/retain_25pct" / method).glob("*.png"))
        assert len(images) == 1
        with Image.open(images[0]) as image:
            assert image.size == (32*7, 24+28)  # raw, input, four factors, GT; no resizing
    for label in ["1.00", "1.25", "1.50", "2.00"]:
        metrics = json.loads((output / f"sat_{label}x/metrics.json").read_text())
        assert metrics["evaluation"]["num_images"] == 1
    with pytest.raises(SystemExit):
        chroma_sweep.main()  # No output overwrite.


def test_paired_extended_metrics_use_exported_raw_and_gt(tmp_path, monkeypatch):
    from gray_cold_diffusion import extended_metrics

    torch.set_num_threads(1)
    raw, reference = tmp_path / "raw", tmp_path / "reference"
    raw.mkdir()
    reference.mkdir()
    Image.new("RGB", (32, 24), (30, 120, 160)).save(raw / "a.png")
    Image.new("RGB", (32, 24), (140, 100, 80)).save(reference / "a.png")
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"test": ["a.png"]}))
    config, checkpoint = write_checkpoint(tmp_path, 1., RestorationUNet(8, (1, 2), diffusion_steps=20).state_dict())
    output = tmp_path / "eval"
    calls = []

    def fake_extended(prediction_dir, reference_dir, names, output_csv, device, eval_size, **kwargs):
        assert reference_dir == output / "references"
        assert names == ["a"] and eval_size == 256
        calls.append(Path(prediction_dir).name)
        return {"num_images": 1, "eval_size": eval_size, "means": {"psnr_rgb": 10.}}

    monkeypatch.setattr(extended_metrics, "create_pyiqa_metrics", lambda device: {})
    monkeypatch.setattr(extended_metrics, "evaluate_extended_metrics", fake_extended)
    monkeypatch.setattr(sys, "argv", ["diagnose_rgb_chroma.py", "--checkpoint", str(checkpoint),
                        "--baseline-config", str(config), "--raw-dir", str(raw), "--reference-dir", str(reference),
                        "--split-file", str(split), "--output-dir", str(output), "--device", "cpu",
                        "--start-steps", "15", "--preview-count", "0", "--extended-metrics"])
    chroma_diagnostic.main(color_space="rgb")
    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics["evaluation"]["status"] == "complete"
    assert calls == ["raw", "direct_predictions", "predictions"]
    assert set(metrics["extended_by_start_step"]["15"]) == {"direct", "algorithm2"}


def test_sweep_wrapper_defaults_to_all_four_factors_fixed_25pct():
    env = os.environ.copy()
    env["EVAL_DRY_RUN"] = "1"
    result = subprocess.run(["bash", "scripts/evaluate_rgb_partial_uieb_4090.sh", "all"],
                            cwd=ROOT, env=env, text=True, capture_output=True, check=True)
    assert "--factors 1.0 1.25 1.5 2.0" in result.stdout
    assert "--start-steps 15" in result.stdout
    assert "--limit 0" in result.stdout
    assert "--preview-count 4" in result.stdout
    assert "splits/uieb_seed42.json" in result.stdout
    assert "rgb_partial_uieb_test90_step050000" in result.stdout
    assert "train.py" not in result.stdout
