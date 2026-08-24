from __future__ import annotations

import csv
import json
from pathlib import Path


METRIC_FIELDS = (
    "train_l1",
    "val_psnr",
    "val_ssim",
    "val_delta_e76",
    "trajectory_monotonic",
)


def read_training_history(path: str | Path) -> dict[str, list[tuple[int, float]]]:
    history = {field: [] for field in METRIC_FIELDS}
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            step = int(row["step"])
            for field in METRIC_FIELDS:
                value = row.get(field, "").strip()
                if value:
                    history[field].append((step, float(value)))
    return history


def summarize_training(history: dict[str, list[tuple[int, float]]]) -> dict:
    validations = []
    validation_by_step: dict[int, dict[str, float]] = {}
    for field in METRIC_FIELDS[1:]:
        for step, value in history[field]:
            validation_by_step.setdefault(step, {})[field] = value
    for step in sorted(validation_by_step):
        row = {"step": step, **validation_by_step[step]}
        if "val_psnr" in row:
            validations.append(row)

    summary = {
        "train_points": len(history["train_l1"]),
        "validation_points": len(validations),
        "resume_rule": {
            "psnr_gain_db": 0.05,
            "delta_e76_reduction": 0.1,
            "monotonic_gain": 0.01,
        },
    }
    if not validations:
        summary["resume_hint"] = "insufficient_validation_data"
        return summary

    summary["latest_validation"] = validations[-1]
    summary["best_psnr_validation"] = max(validations, key=lambda row: row["val_psnr"])
    if len(validations) < 3:
        summary["resume_hint"] = "need_at_least_3_validation_points"
        return summary

    recent = validations[-3:]
    first, last = recent[0], recent[-1]
    changes = {
        "from_step": first["step"],
        "to_step": last["step"],
        "psnr_gain_db": last["val_psnr"] - first["val_psnr"],
        "delta_e76_reduction": first["val_delta_e76"] - last["val_delta_e76"],
        "monotonic_gain": last["trajectory_monotonic"] - first["trajectory_monotonic"],
    }
    summary["recent_3_validation_change"] = changes
    rule = summary["resume_rule"]
    still_improving = (
        changes["psnr_gain_db"] >= rule["psnr_gain_db"]
        or changes["delta_e76_reduction"] >= rule["delta_e76_reduction"]
        or changes["monotonic_gain"] >= rule["monotonic_gain"]
    )
    summary["resume_hint"] = "continue_candidate" if still_improving else "plateau_or_regressing"
    return summary


def save_training_report(metrics_path: str | Path, output_dir: str | Path) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics_path = Path(metrics_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history = read_training_history(metrics_path)
    summary = summarize_training(history)
    summary["metrics_file"] = str(metrics_path.resolve())
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    plots = (
        ("train_l1", "Training L1", "lower is better", "#1f77b4"),
        ("val_psnr", "Validation PSNR", "higher is better", "#2ca02c"),
        ("val_ssim", "Validation SSIM", "higher is better", "#9467bd"),
        ("val_delta_e76", "Validation Delta-E76", "lower is better", "#d62728"),
        ("trajectory_monotonic", "Trajectory monotonic", "higher is better", "#ff7f0e"),
    )
    for axis, (field, title, ylabel, color) in zip(axes.flat, plots):
        points = history[field]
        if points:
            axis.plot([point[0] for point in points], [point[1] for point in points], color=color)
        axis.set_title(title)
        axis.set_xlabel("step")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)

    summary_axis = axes.flat[-1]
    summary_axis.axis("off")
    latest = summary.get("latest_validation", {})
    recent = summary.get("recent_3_validation_change", {})
    text = (
        f"Resume hint: {summary['resume_hint']}\n\n"
        f"Latest step: {latest.get('step', 'n/a')}\n"
        f"PSNR: {latest.get('val_psnr', float('nan')):.4f}\n"
        f"SSIM: {latest.get('val_ssim', float('nan')):.4f}\n"
        f"Delta-E76: {latest.get('val_delta_e76', float('nan')):.4f}\n"
        f"Monotonic: {latest.get('trajectory_monotonic', float('nan')):.4f}\n\n"
        f"Recent PSNR gain: {recent.get('psnr_gain_db', float('nan')):+.4f}\n"
        f"Recent Delta-E reduction: {recent.get('delta_e76_reduction', float('nan')):+.4f}\n"
        f"Recent monotonic gain: {recent.get('monotonic_gain', float('nan')):+.4f}"
    )
    summary_axis.text(0.02, 0.98, text, va="top", family="monospace", fontsize=12)
    fig.suptitle("Training and validation curves", fontsize=16)
    fig.savefig(output_dir / "training_curves.png", dpi=180)
    plt.close(fig)
    return summary
