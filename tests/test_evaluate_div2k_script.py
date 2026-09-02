import os
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
