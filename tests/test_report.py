import csv
from pathlib import Path

from PIL import Image

from gray_cold_diffusion.report import read_training_history, save_training_report, summarize_training


def _write_metrics(path: Path):
    fields = [
        "step",
        "train_l1",
        "val_psnr",
        "val_ssim",
        "val_delta_e76",
        "trajectory_monotonic",
    ]
    rows = [
        [50, 0.30, "", "", "", ""],
        [5000, "", 17.0, 0.80, 22.0, 0.60],
        [10000, "", 17.5, 0.82, 21.0, 0.68],
        [15000, "", 18.0, 0.84, 20.0, 0.75],
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        writer.writerows(rows)


def test_training_report_and_resume_hint(tmp_path: Path):
    metrics_path = tmp_path / "metrics.csv"
    _write_metrics(metrics_path)
    history = read_training_history(metrics_path)
    summary = summarize_training(history)

    assert summary["resume_hint"] == "continue_candidate"
    assert summary["best_psnr_validation"]["step"] == 15000

    report_dir = tmp_path / "report"
    save_training_report(metrics_path, report_dir)
    with Image.open(report_dir / "training_curves.png") as image:
        assert image.width >= 2000
        assert image.height >= 1000
    assert (report_dir / "training_summary.json").is_file()
