import os
import json
from pathlib import Path
import subprocess


def test_all_saturation_evaluations_are_explicit_and_extended():
    repo_root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["EVAL_DRY_RUN"] = "1"
    result = subprocess.run(
        ["bash", "scripts/evaluate_div2k_uieb_4090.sh", "all"],
        cwd=repo_root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    lines = [line for line in result.stdout.splitlines() if line.startswith("DRY RUN:")]
    assert len(lines) == 4
    for label, line in zip(("1.00x", "1.25x", "1.50x", "2.00x"), lines):
        assert "--extended-metrics" in line
        assert "--original-size" in line
        assert f"div2k_rgb_sat_{label}_uieb" in line


def test_lab_sat1_evaluation_is_explicit_and_extended():
    repo_root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["EVAL_DRY_RUN"] = "1"
    result = subprocess.run(
        ["bash", "scripts/evaluate_div2k_lab_uieb_4090.sh"],
        cwd=repo_root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    lines = [line for line in result.stdout.splitlines() if line.startswith("DRY RUN:")]
    assert len(lines) == 1
    assert "div2k_lab_sat1_50k/checkpoints/best.pt" in lines[0]
    assert "--extended-metrics" in lines[0]
    assert "--original-size" in lines[0]
    assert "div2k_lab_sat_1.00x_uieb" in lines[0]


def test_full_resolution_div2k_validation_evaluation_supports_both_models():
    repo_root = Path(__file__).resolve().parents[1]
    manifest = json.loads((repo_root / "splits/div2k_valid_all.json").read_text())
    assert manifest["test"] == [f"{index:04d}.png" for index in range(801, 901)]
    environment = os.environ.copy()
    environment["EVAL_DRY_RUN"] = "1"
    result = subprocess.run(
        ["bash", "scripts/evaluate_div2k_validation_4090.sh", "both"],
        cwd=repo_root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    lines = [line for line in result.stdout.splitlines() if line.startswith("DRY RUN:")]
    assert len(lines) == 2
    assert "div2k_lab_sat_1.00x_div2k_val" in lines[0]
    assert "div2k_rgb_sat_1.00x_div2k_val" in lines[1]
    for line in lines:
        assert "--original-size" in line
        assert "--tile-size 512" in line
        assert "--tile-overlap 64" in line
        assert "--preview-max-side 512" in line
        assert "--extended-metrics" in line
