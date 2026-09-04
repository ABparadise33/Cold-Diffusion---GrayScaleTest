"""Sequential, raw-only UIEB inference with existing RGB saturation checkpoints."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import torch
import yaml

from .chroma_diagnostic import file_sha256
from .data import PairedImageDataset
from .diagnostics import check_diagnostic_checkpoint
from .io import save_stage_strip


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = {
    1.0: "div2k_rgb_sat1_50k",
    1.25: "div2k_rgb_sat1_25_50k",
    1.5: "div2k_rgb_sat1_5_50k",
    2.0: "div2k_rgb_sat2_50k",
}


def validate_comparable_reports(reports):
    if not reports:
        raise ValueError("no completed reports to compare")
    first = reports[0]["evaluation"]
    keys = (
        "task", "color_space", "gray_operator", "checkpoint_step", "diffusion_steps",
        "start_steps", "selected_names", "input_sha256", "reference_sha256",
        "split_sha256", "tile_size", "tile_overlap", "preview_names", "extended_metric_size",
    )
    factors = []
    for report in reports:
        meta = report["evaluation"]
        if meta["status"] != "complete" or meta["task"] != "paired_underwater_transfer" or meta["color_space"] != "rgb":
            raise ValueError("comparison requires completed raw-only paired RGB runs")
        for key in keys:
            if meta.get(key) != first.get(key):
                raise ValueError(f"runs are not comparable: {key} differs")
        if not meta.get("input_sha256") or not meta.get("reference_sha256"):
            raise ValueError("input/reference hashes must be recorded")
        factors.append(meta["training_saturation_factor"])
    if len(set(factors)) != len(factors):
        raise ValueError("duplicate saturation factor in comparison")


def _read_tensor(path):
    with Image.open(path) as image:
        array = np.array(image.convert("RGB"), dtype=np.float32) / 255
    return torch.from_numpy(array).permute(2, 0, 1)


def summarize_runs(run_dirs, output):
    run_dirs = [Path(path) for path in run_dirs]
    reports = [json.loads((path / "metrics.json").read_text()) for path in run_dirs]
    validate_comparable_reports(reports)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    targets = [output / f"saturation_comparison.{ext}" for ext in ("json", "csv", "md")]
    if any(path.exists() for path in targets):
        raise ValueError("comparison files already exist; use a new output directory")
    rows = []
    for report in reports:
        factor = report["evaluation"]["training_saturation_factor"]
        for start, means in report["means_by_start_step"].items():
            for method in ("direct", "algorithm2"):
                row = {
                    "saturation_factor": factor, "start_step": int(start),
                    "retained_color": means["retained_chroma"], "method": method,
                    "raw_psnr_rgb": means["raw_psnr_rgb"], "psnr_rgb": means[f"{method}_psnr_rgb"],
                    "raw_ssim_rgb": means["raw_ssim_rgb"], "ssim_rgb": means[f"{method}_ssim_rgb"],
                    "raw_delta_e76": means["raw_delta_e76"], "delta_e76": means[f"{method}_delta_e76"],
                    "delta_e_improvement_over_raw": means[f"{method}_delta_e_improvement_over_raw"],
                    "output_vs_raw_mae_255": means[f"{method}_vs_raw_mae_255"],
                    "chroma": means[f"{method}_chroma"],
                    "mean_a": means[f"{method}_mean_a"], "mean_b": means[f"{method}_mean_b"],
                }
                extended = report.get("extended_by_start_step", {}).get(start, {}).get(method, {})
                row.update({f"extended_{key}": value for key, value in extended.get("means", {}).items()})
                rows.append(row)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with targets[1].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "status": "complete", "runs": [str(path.resolve()) for path in run_dirs],
        "input_and_reference_hashes_match": True,
        "selected_names": reports[0]["evaluation"]["selected_names"], "rows": rows,
        "interpretation": "Positive delta_e_improvement_over_raw is better than unchanged raw; higher chroma alone is not success.",
    }
    targets[0].write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    lines = [
        "# RGB saturation / UIEB partial-color comparison", "",
        "Same raw-derived inputs and unchanged GT across runs (hashes verified).",
        "Core scores: original-size tensors. Extended scores: legacy IQA preprocessing.",
        "Positive DE improvement means better than leaving raw unchanged.", "",
        "| Factor | Retained | Method | PSNR RGB ↑ | SSIM RGB ↑ | DE76 ↓ | DE improvement over raw ↑ |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(f"| {row['saturation_factor']:.2f}x | {row['retained_color']:.0%} | {row['method']} | "
                     f"{row['psnr_rgb']:.4f} | {row['ssim_rgb']:.4f} | {row['delta_e76']:.4f} | "
                     f"{row['delta_e_improvement_over_raw']:+.4f} |")
    targets[2].write_text("\n".join(lines) + "\n", encoding="utf-8")
    first = reports[0]["evaluation"]
    for start in first["start_steps"]:
        level = f"retain_{(1-start/first['diffusion_steps'])*100:g}pct"
        for name in first["preview_names"]:
            for method, folder in [("algorithm2", "predictions"), ("direct", "direct_predictions")]:
                stages = [
                    ("raw", _read_tensor(run_dirs[0] / "raw" / f"{name}.png")),
                    (f"input t={start}", _read_tensor(run_dirs[0] / level / "inputs" / f"{name}.png")),
                ]
                for run, report in zip(run_dirs, reports):
                    factor = report["evaluation"]["training_saturation_factor"]
                    stages.append((f"{factor:.2f}x {method}", _read_tensor(run / level / folder / f"{name}.png")))
                stages.append(("reference (GT)", _read_tensor(run_dirs[0] / "references" / f"{name}.png")))
                save_stage_strip(stages, output / "comparisons" / level / method / f"{name}.png")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factors", type=float, nargs="+", choices=list(EXPERIMENTS), default=list(EXPERIMENTS))
    parser.add_argument("--checkpoint-root", default=str(ROOT / "outputs"))
    parser.add_argument("--raw-dir", default=str(ROOT / "data/UIEB/raw-890"))
    parser.add_argument("--reference-dir", default=str(ROOT / "data/UIEB/reference-890"))
    parser.add_argument("--split-file", default=str(ROOT / "splits/uieb_seed42.json"))
    parser.add_argument("--output-dir", default=str(ROOT / "evaluation/rgb_partial_uieb_test90_step050000"))
    parser.add_argument("--start-steps", type=int, nargs="+", default=[15])
    parser.add_argument("--include-gray-control", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--preview-count", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--tile-overlap", type=int, default=64)
    parser.add_argument("--skip-extended-metrics", action="store_true", help="quick core-metric inference only")
    args = parser.parse_args()
    if len(set(args.factors)) != len(args.factors):
        parser.error("saturation factors must be unique")
    if args.limit < 0 or args.preview_count < 0 or any(t < 1 or t >= 20 for t in args.start_steps):
        parser.error("limit/preview count must be >=0; start steps must be in 1..19")
    if args.tile_size < 0 or not 0 <= args.tile_overlap < max(args.tile_size, 1):
        parser.error("use 0 <= tile-overlap < tile-size, or both 0")
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        parser.error("output directory is not empty; use --output-dir with a new directory")
    # Preserve the exact FlowIE-compatible Test90 membership, even for a custom split path.
    canonical = PairedImageDataset(args.raw_dir, args.reference_dir, ROOT / "splits/uieb_seed42.json", "test", None)
    dataset = PairedImageDataset(args.raw_dir, args.reference_dir, args.split_file, "test", None)
    canonical_pairs = {(a.name, b.name) for a, b in map(canonical._paths, canonical.items)}
    actual_pairs = [(a.name, b.name) for a, b in map(dataset._paths, dataset.items)]
    if len(actual_pairs) != 90 or len(set(actual_pairs)) != 90 or set(actual_pairs) != canonical_pairs:
        parser.error("split must contain the exact Underwater_FlowIE-compatible seed42 Test90 pairs")
    for item in dataset.items:
        for path in dataset._paths(item):
            if not path.is_file():
                parser.error(f"missing UIEB image: {path}; prepare the dataset before inference")
    jobs = []
    # Validate every checkpoint before spending GPU time on the first factor.
    for factor in args.factors:
        experiment = EXPERIMENTS[factor]
        config_path = ROOT / "configs" / f"{experiment}.yaml"
        checkpoint_path = Path(args.checkpoint_root) / experiment / "checkpoints/step_050000.pt"
        if not checkpoint_path.is_file():
            parser.error(f"missing checkpoint: {checkpoint_path}; no best.pt substitution or new training")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        baseline = yaml.safe_load(config_path.read_text())
        try:
            check_diagnostic_checkpoint(checkpoint["config"], baseline, int(checkpoint.get("step", -1)), 50000,
                                        color_space="rgb", expected_saturation_factor=factor)
        except ValueError as error:
            parser.error(str(error))
        del checkpoint
        jobs.append((factor, checkpoint_path, config_path, output / f"sat_{factor:.2f}x", file_sha256(checkpoint_path)))
    print(f"Verified {len(jobs)} RGB checkpoints at 50000; shared raw-only inputs, starts={args.start_steps}", flush=True)
    for factor, checkpoint, config, run_output, expected_hash in jobs:
        command = [
            sys.executable, "-u", str(ROOT / "diagnose_rgb_chroma.py"),
            "--checkpoint", str(checkpoint), "--baseline-config", str(config),
            "--expected-checkpoint-step", "50000", "--expected-saturation-factor", str(factor),
            "--raw-dir", args.raw_dir, "--reference-dir", args.reference_dir, "--split-file", args.split_file,
            "--output-dir", str(run_output), "--limit", str(args.limit), "--seed", "42",
            "--preview-count", str(args.preview_count), "--device", args.device,
            "--tile-size", str(args.tile_size), "--tile-overlap", str(args.tile_overlap),
            "--start-steps", *map(str, args.start_steps),
        ]
        if args.include_gray_control:
            command.append("--include-gray-control")
        if not args.skip_extended_metrics:
            command.extend(["--extended-metrics", "--extended-metric-size", "256"])
        print(f"Starting saturation={factor:.2f}x -> {run_output}", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
        # Confirm the checkpoint was not replaced between validation and inference.
        report = json.loads((run_output / "metrics.json").read_text())
        if (report["evaluation"]["checkpoint_sha256"] != expected_hash
                or file_sha256(checkpoint) != expected_hash):
            raise RuntimeError("checkpoint changed during the sweep")
    summarize_runs([job[3] for job in jobs], output)
    print(f"RGB UIEB SWEEP COMPLETE: {output}", flush=True)


if __name__ == "__main__":
    main()
