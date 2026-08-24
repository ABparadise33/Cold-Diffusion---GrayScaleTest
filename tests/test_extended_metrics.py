import csv

import numpy as np
from PIL import Image
import torch

from gray_cold_diffusion import extended_metrics


class _DummyMetric:
    def __call__(self, *_args):
        return torch.tensor(0.5)


def test_extended_metrics_writes_per_image_and_mean(monkeypatch, tmp_path):
    names = ["sample_a", "sample_b"]
    prediction_dir = tmp_path / "predictions"
    reference_dir = tmp_path / "references"
    prediction_dir.mkdir()
    reference_dir.mkdir()

    for index, name in enumerate(names):
        prediction = np.full((32, 32, 3), 90 + index * 20, dtype=np.uint8)
        reference = np.full((32, 32, 3), 100 + index * 20, dtype=np.uint8)
        Image.fromarray(prediction).save(prediction_dir / f"{name}.png")
        Image.fromarray(reference).save(reference_dir / f"{name}.png")

    dummy_metrics = {
        name: _DummyMetric()
        for name in ("psnr_y", "psnr_rgb", "ssim_y", "ssim_rgb", "ms_ssim", "lpips", "niqe", "musiq", "clipiqa")
    }
    monkeypatch.setattr(extended_metrics, "create_pyiqa_metrics", lambda _device: dummy_metrics)

    cold_scores = {
        "sample_a": {"delta_e76": 12.0, "trajectory_monotonic": 0.75},
        "sample_b": {"delta_e76": 10.0, "trajectory_monotonic": 1.0},
    }
    output_csv = tmp_path / "extended_metrics.csv"
    result = extended_metrics.evaluate_extended_metrics(
        prediction_dir,
        reference_dir,
        names,
        output_csv,
        torch.device("cpu"),
        eval_size=32,
        cold_scores=cold_scores,
    )

    with output_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 3
    assert rows[-1]["image"] == "__mean__"
    assert set(extended_metrics.OUTPUT_METRICS).issubset(rows[-1])
    assert result["num_images"] == 2
    assert result["means"]["delta_e76"] == 11.0
    assert result["means"]["trajectory_monotonic"] == 0.875
    assert output_csv.with_suffix(".md").is_file()

    direct_csv = tmp_path / "direct_metrics.csv"
    direct_result = extended_metrics.evaluate_extended_metrics(
        prediction_dir,
        reference_dir,
        names,
        direct_csv,
        torch.device("cpu"),
        eval_size=32,
        cold_scores={name: {"delta_e76": score["delta_e76"]} for name, score in cold_scores.items()},
        pyiqa_metrics=dummy_metrics,
    )
    with direct_csv.open(newline="", encoding="utf-8") as handle:
        direct_fields = csv.DictReader(handle).fieldnames
    assert direct_result["means"]["delta_e76"] == 11.0
    assert "trajectory_monotonic" not in direct_fields


def test_method_comparison_marks_winner_and_algorithm_only_metric(tmp_path):
    direct = {name: 0.5 for name in extended_metrics.ALL_METRICS}
    direct["delta_e76"] = 12.0
    algorithm2 = {name: 0.6 for name in extended_metrics.ALL_METRICS}
    algorithm2.update({"delta_e76": 10.0, "trajectory_monotonic": 0.875})

    output_csv = tmp_path / "direct_vs_algorithm2.csv"
    rows = extended_metrics.save_method_comparison(direct, algorithm2, output_csv)
    rows_by_metric = {row["metric"]: row for row in rows}

    assert rows_by_metric["psnr_rgb"]["winner"] == "Algorithm 2"
    assert rows_by_metric["lpips"]["winner"] == "Direct"
    assert rows_by_metric["delta_e76"]["winner"] == "Algorithm 2"
    assert rows_by_metric["trajectory_monotonic"]["direct"] is None
    assert output_csv.is_file()
    assert output_csv.with_suffix(".md").is_file()
