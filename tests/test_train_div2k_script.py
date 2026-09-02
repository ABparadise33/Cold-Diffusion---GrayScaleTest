import os
from pathlib import Path
import subprocess


def test_all_saturation_command_lists_four_runs_in_order():
    root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["DIV2K_DRY_RUN"] = "1"
    result = subprocess.run(
        [
            "bash",
            "scripts/train_div2k_4090.sh",
            "all",
            "--resume-if-exists",
            "--max-steps",
            "100000",
        ],
        cwd=root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    lines = [line for line in result.stdout.splitlines() if line.startswith("DRY RUN:")]
    assert len(lines) == 4
    expected = ["sat1_50k", "sat1_25_50k", "sat1_5_50k", "sat2_50k"]
    assert all(fragment in line for fragment, line in zip(expected, lines))
    assert all("--resume-if-exists --max-steps 100000" in line for line in lines)
