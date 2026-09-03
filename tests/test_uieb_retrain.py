import hashlib
import os
from pathlib import Path
import subprocess

import yaml


def test_original_uieb_config_remains_byte_identical():
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256((root / "configs/cold_gray_50k.yaml").read_bytes()).hexdigest()
    assert digest == "f631f0c340f8b9aced8bb75bc0128aa4945b525c8cd62596529ca3b121382801"


def test_fresh_uieb_retrain_changes_only_recording_fields():
    root = Path(__file__).resolve().parents[1]
    original = yaml.safe_load((root / "configs/cold_gray_50k.yaml").read_text())
    retrain = yaml.safe_load((root / "configs/uieb_lab_retrain_50k.yaml").read_text())

    assert retrain["mode"] == original["mode"] == "cold_gray"
    assert retrain["seed"] == original["seed"] == 42
    assert retrain["model"] == original["model"]
    assert retrain["diffusion"] == original["diffusion"]
    original_training = dict(original["training"])
    retrain_training = dict(retrain["training"])
    assert original_training.pop("validate_every") == 5000
    assert retrain_training.pop("validate_every") == 1000
    assert retrain_training == original_training


def test_fresh_uieb_retrain_dry_run_does_not_resume():
    root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["UIEB_RETRAIN_DRY_RUN"] = "1"
    result = subprocess.run(
        ["bash", "scripts/train_uieb_lab_retrain_4090.sh"],
        cwd=root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    line = next(line for line in result.stdout.splitlines() if line.startswith("DRY RUN:"))
    assert "uieb_lab_retrain_50k.yaml" in line
    assert "outputs/uieb_lab_retrain_50k" in line
    assert "--resume" not in line


def test_fresh_uieb_retrain_evaluation_uses_original_test90():
    root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["EVAL_DRY_RUN"] = "1"
    result = subprocess.run(
        ["bash", "scripts/evaluate_uieb_lab_retrain_4090.sh"],
        cwd=root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    line = next(line for line in result.stdout.splitlines() if line.startswith("DRY RUN:"))
    assert "uieb_lab_retrain_50k/checkpoints/best.pt" in line
    assert "splits/uieb_seed42.json" in line
    assert "--original-size" in line
    assert "--extended-metrics" in line
