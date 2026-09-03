import os
from pathlib import Path
import subprocess

import yaml


CONFIGS = (
    "div2k_uieb_style_lab_sat1_50k.yaml",
    "div2k_uieb_style_lab_sat1_25_50k.yaml",
    "div2k_uieb_style_lab_sat1_5_50k.yaml",
    "div2k_uieb_style_lab_sat2_50k.yaml",
)


def test_uieb_style_div2k_configs_keep_the_original_lab_recipe():
    root = Path(__file__).resolve().parents[1]
    configs = [yaml.safe_load((root / "configs" / name).read_text()) for name in CONFIGS]
    factors = [float(config["data"].get("reference_saturation_factor", 1.0)) for config in configs]
    assert factors == [1.0, 1.25, 1.5, 2.0]
    for config in configs:
        assert config["mode"] == "cold_gray"
        assert config["seed"] == 42
        assert config["diffusion"]["steps"] == 8
        assert config["data"]["image_size"] == 128
        assert config["data"]["validation_preview_name"] == "0803"
        assert config["training"]["validate_every"] == 1000


def test_uieb_style_div2k_train_dry_run_is_explicit():
    root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["DIV2K_UIEB_STYLE_DRY_RUN"] = "1"
    result = subprocess.run(
        ["bash", "scripts/train_div2k_uieb_style_4090.sh", "all", "--resume-if-exists"],
        cwd=root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    lines = [line for line in result.stdout.splitlines() if line.startswith("DRY RUN:")]
    assert len(lines) == 4
    assert all("--raw-dir" in line and "data/DIV2K" in line for line in lines)
    assert all("--reference-dir" in line and "--resume-if-exists" in line for line in lines)


def test_uieb_style_div2k_evaluate_dry_run_preserves_full_size():
    root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["EVAL_DRY_RUN"] = "1"
    result = subprocess.run(
        ["bash", "scripts/evaluate_div2k_uieb_style_4090.sh", "1.0"],
        cwd=root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    line = next(line for line in result.stdout.splitlines() if line.startswith("DRY RUN:"))
    assert "--original-size" in line
    assert "--tile-size 512" in line
    assert "div2k_uieb_style_lab_sat_1.00x_div2k_val" in line
