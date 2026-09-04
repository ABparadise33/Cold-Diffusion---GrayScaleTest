"""Partial-chroma inference: natural self-pairs or raw-only paired transfer."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import subprocess

import torch
import yaml

from gray_cold_diffusion.bridge import GrayBridge
from gray_cold_diffusion.color import (
    denormalize_rgb, gray_anchor, normalize_rgb, normalized_lab_to_rgb,
    rgb_channel_mean_gray, rgb_to_normalized_lab,
)
from gray_cold_diffusion.data import PairedImageDataset
from gray_cold_diffusion.diagnostics import check_diagnostic_checkpoint, sample_from_step
from gray_cold_diffusion.io import save_stage_strip, save_tensor_image, select_device
from gray_cold_diffusion.metrics import delta_e76, psnr, ssim
from gray_cold_diffusion.model import RestorationUNet
from gray_cold_diffusion.tiling import TiledModel


def score(rgb, reference, reference_lab):
    lab = rgb_to_normalized_lab(rgb)
    ab = lab[:, 1:] * 128.0
    return {
        "psnr_rgb": float(psnr(rgb, reference).mean().item()),
        "ssim_rgb": float(ssim(rgb, reference).mean().item()),
        "delta_e76": float(delta_e76(lab, reference_lab).mean().item()),
        "chroma": float(ab.square().sum(dim=1).sqrt().mean().item()),
        "mean_a": float(ab[:, 0].mean().item()),
        "mean_b": float(ab[:, 1].mean().item()),
    }


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_revision(root):
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
            text=True, check=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def main(color_space="lab"):
    if color_space not in {"lab", "rgb"}:
        raise ValueError("color_space must be lab or rgb")
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-checkpoint-step", type=int, default=50000)
    parser.add_argument("--expected-saturation-factor", type=float, choices=[1., 1.25, 1.5, 2.], default=1.)
    parser.add_argument("--baseline-config", default=str(root / f"configs/div2k_{color_space}_sat1_50k.yaml"))
    parser.add_argument("--image-dir", default=str(root / "data/DIV2K/DIV2K_valid_HR"))
    parser.add_argument("--raw-dir", help="paired transfer: input/anchor are derived ONLY from raw")
    parser.add_argument("--reference-dir", help="paired transfer: reference is for scoring/display only")
    parser.add_argument("--split-file")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--start-steps", type=int, nargs="+", default=[10, 15, 18])
    parser.add_argument("--include-gray-control", action="store_true")
    parser.add_argument("--limit", type=int, help="fixed-seed subset; default 4 for natural, 0/all for paired")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--tile-overlap", type=int, default=64)
    parser.add_argument("--preview-max-side", type=int, help="optional display-only reduction; no reduction by default")
    parser.add_argument("--preview-count", type=int, help="limit comparison/trajectory examples, not predictions")
    parser.add_argument("--extended-metrics", action="store_true")
    parser.add_argument("--extended-metric-size", type=int, default=256)
    args = parser.parse_args()
    if bool(args.raw_dir) != bool(args.reference_dir):
        parser.error("--raw-dir and --reference-dir must be provided together")
    paired = bool(args.raw_dir)
    if paired and color_space != "rgb":
        parser.error("paired partial-color transfer currently requires the RGB entry point")
    if args.limit is None:
        args.limit = 0 if paired else 4
    if args.split_file is None:
        args.split_file = str(root / ("splits/uieb_seed42.json" if paired else "splits/div2k_valid_all.json"))
    if args.limit < 0:
        parser.error("--limit must be >= 0")
    if args.tile_size < 0 or not 0 <= args.tile_overlap < max(args.tile_size, 1):
        # --tile-size 0 --tile-overlap 0 explicitly disables tiling.
        parser.error("use 0 <= tile-overlap < tile-size, or set both to 0 to disable tiling")
    if args.preview_max_side is not None and args.preview_max_side < 1:
        parser.error("--preview-max-side must be >= 1")
    if args.preview_count is not None and args.preview_count < 0:
        parser.error("--preview-count must be >= 0")
    if args.extended_metric_size < 0:
        parser.error("--extended-metric-size must be >= 0")
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        parser.error(f"output directory is not empty; use a new directory: {output}")
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    baseline = yaml.safe_load(Path(args.baseline_config).read_text(encoding="utf-8"))
    checkpoint_step = int(checkpoint.get("step", -1))
    try:
        check_diagnostic_checkpoint(
            config, baseline, checkpoint_step, args.expected_checkpoint_step, color_space=color_space,
            expected_saturation_factor=args.expected_saturation_factor,
        )
    except ValueError as error:
        parser.error(str(error))
    steps = int(config["diffusion"]["steps"])
    starts = sorted(set(args.start_steps))
    if not starts or any(step < 1 or step >= steps for step in starts):
        parser.error(f"--start-steps must be in [1,{steps-1}]; use --include-gray-control to add full gray")
    if args.include_gray_control:
        starts.append(steps)

    raw_dir = args.raw_dir if paired else args.image_dir
    reference_dir = args.reference_dir if paired else args.image_dir
    dataset = PairedImageDataset(raw_dir, reference_dir, args.split_file, "test", None)
    indices = list(range(len(dataset)))
    if 0 < args.limit < len(indices):
        indices = sorted(random.Random(args.seed).sample(indices, args.limit))
    selected_names = [Path(dataset._paths(dataset.items[i])[0]).stem for i in indices]
    if len(set(selected_names)) != len(selected_names):
        parser.error("image stems must be unique to avoid overwriting outputs")
    for index in indices:
        for path in dataset._paths(dataset.items[index]):
            if not path.is_file():
                parser.error(f"missing image: {path}")
    preview_names = selected_names
    if args.preview_count is not None and args.preview_count < len(selected_names):
        chosen = sorted(random.Random(args.seed).sample(range(len(selected_names)), args.preview_count))
        preview_names = [selected_names[i] for i in chosen]
    device = select_device(args.device)
    model_cfg = config["model"]
    model = RestorationUNet(
        int(model_cfg["base_channels"]), tuple(model_cfg["channel_mults"]),
        float(model_cfg.get("dropout", 0.0)), steps,
    )
    model.load_state_dict(checkpoint["ema"])
    model.to(device).eval()
    del checkpoint
    inference_model = TiledModel(model, args.tile_size, args.tile_overlap) if args.tile_size else model
    bridge = GrayBridge(steps).to(device)
    to_rgb = denormalize_rgb if color_space == "rgb" else normalized_lab_to_rgb
    chroma_label = "RGB-gray" if color_space == "rgb" else "a/b"
    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "status": "running",
        "purpose": (
            "paired underwater transfer; input and anchor come ONLY from raw, GT is scoring/display only"
            if paired else "synthetic partial-chroma diagnostic; input includes target color information"
        ),
        "task": "paired_underwater_transfer" if paired else "natural_self_desaturation",
        "input_source": "raw",
        "analytic_control": (
            "For retained_chroma > 0, g + (input-g)/retained_chroma exactly inverts "
            "the synthetic desaturation before RGB quantization, recovering RAW, not paired GT"
        ),
        "color_space": color_space,
        "gray_operator": "rgb_channel_mean" if color_space == "rgb" else "lab_L_ab_zero",
        "retention_definition": "fraction of RGB-gray" if color_space == "rgb" else "fraction of Lab a/b",
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "code_revision": code_revision(root),
        "diagnostic_source_sha256": file_sha256(__file__),
        "checkpoint_step": checkpoint_step,
        "checkpoint_config": config,
        "training_saturation_factor": args.expected_saturation_factor,
        "baseline_config": str(Path(args.baseline_config).resolve()),
        "baseline_config_sha256": hashlib.sha256(Path(args.baseline_config).read_bytes()).hexdigest(),
        "diffusion_steps": steps,
        "start_steps": starts,
        "retained_chroma": {str(t): 1-t/steps for t in starts},
        "spatial_mode": "original_size",
        "image_dir": str(Path(raw_dir).resolve()),
        "reference_dir": str(Path(reference_dir).resolve()),
        "split_sha256": hashlib.sha256(Path(args.split_file).read_bytes()).hexdigest(),
        "subset_seed": args.seed,
        "selected_names": selected_names,
        "preview_names": preview_names,
        "num_images": len(selected_names),
        "input_sha256": {},
        "reference_sha256": {},
        "tile_size": args.tile_size,
        "tile_overlap": args.tile_overlap,
        "device": str(device),
        "torch_version": str(torch.__version__),
        "extended_metrics_requested": args.extended_metrics,
        "extended_metric_size": args.extended_metric_size if args.extended_metrics else None,
    }
    manifest = output / "run_metadata.json"
    manifest.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"verified checkpoint_step={checkpoint_step} mode={config['mode']} T={steps}", flush=True)
    print(f"images={selected_names}; start_steps={starts}; no training", flush=True)
    rows = []
    with (output / "per_image_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = None
        with torch.no_grad():
            for index in indices:
                item = dataset[index]
                name = item["name"]
                raw = item["raw"].unsqueeze(0).to(device)
                reference = item["reference"].unsqueeze(0).to(device)
                reference_lab = rgb_to_normalized_lab(reference)
                raw_lab = rgb_to_normalized_lab(raw)
                if color_space == "rgb":
                    clean = normalize_rgb(raw)
                    anchor = rgb_channel_mean_gray(clean)
                else:
                    clean = raw_lab
                    anchor = gray_anchor(clean)
                save_tensor_image(reference[0], output / "references" / f"{name}.png")
                save_tensor_image(raw[0], output / "raw" / f"{name}.png")
                metadata["reference_sha256"][name] = file_sha256(output / "references" / f"{name}.png")
                metadata["input_sha256"][name] = {}
                for start in starts:
                    retained = 1-start/steps
                    percent = f"{retained*100:g}"
                    level = output / f"retain_{percent}pct"
                    t = torch.full((1,), start, device=device, dtype=torch.long)
                    initial = bridge.degrade(clean, anchor, t)
                    metadata["input_sha256"][name][str(start)] = hashlib.sha256(
                        initial.detach().cpu().contiguous().numpy().tobytes()
                    ).hexdigest()
                    direct = to_rgb(inference_model(initial, t).clamp(-1, 1))
                    final, trajectory = sample_from_step(inference_model, bridge, initial, anchor, start)
                    sampled = to_rgb(final)
                    initial_rgb = to_rgb(initial)
                    analytic = (
                        to_rgb((initial - (1-retained)*anchor) / retained)
                        if retained > 0 else None
                    )
                    for kind, image in [("inputs", initial_rgb), ("direct_predictions", direct), ("predictions", sampled)]:
                        if image.shape != reference.shape:
                            raise RuntimeError(f"{kind} geometry changed for {name}")
                        save_tensor_image(image[0], level / kind / f"{name}.png")
                    stages = [
                        ("raw (unchanged baseline)", raw),
                        (f"input t={start}/{steps}, {chroma_label}={retained*100:g}%", initial_rgb),
                        ("Direct", direct), ("Algorithm 2", sampled),
                    ]
                    if analytic is not None:
                        stages.append(("analytic inverse (control)", analytic))
                    stages.append(("reference", reference))
                    if name in preview_names:
                        save_stage_strip(stages, level / "batches" / f"{name}.png", max_side=args.preview_max_side)
                        save_stage_strip([
                            (f"t={start-i}/{steps}", to_rgb(state))
                            for i, state in enumerate(trajectory)
                        ], level / "trajectories" / f"{name}.png", max_side=args.preview_max_side)
                    row = {"image": name, "start_step": start, "retained_chroma": retained}
                    for method, image in [("raw", raw), ("input", initial_rgb), ("direct", direct), ("algorithm2", sampled), ("reference", reference)]:
                        row.update({f"{method}_{key}": value for key, value in score(image, reference, reference_lab).items()})
                    analytic_scores = score(analytic, reference, reference_lab) if analytic is not None else None
                    row["analytic_delta_e76"] = analytic_scores["delta_e76"] if analytic_scores else None
                    row["analytic_raw_delta_e76"] = float(
                        delta_e76(rgb_to_normalized_lab(analytic), raw_lab).mean().item()
                    ) if analytic is not None else None
                    if analytic is not None and not row["analytic_raw_delta_e76"] < .01:
                        raise RuntimeError(f"analytic inverse failed to recover raw for {name} t={start}")
                    errors = torch.stack([
                        delta_e76(rgb_to_normalized_lab(to_rgb(state)), reference_lab)
                        for state in trajectory
                    ], dim=1)
                    row["algorithm2_trajectory_monotonic"] = float(
                        (errors[:, 1:] <= errors[:, :-1] + 1e-6).float().mean().item()
                    )
                    for method, image in [("direct", direct), ("algorithm2", sampled)]:
                        row[f"{method}_vs_raw_mae_255"] = float((image-raw).abs().mean().item()*255)
                        row[f"{method}_delta_e_improvement_over_raw"] = row["raw_delta_e76"] - row[f"{method}_delta_e76"]
                    rows.append(row)
                    if writer is None:
                        writer = csv.DictWriter(handle, fieldnames=list(row))
                        writer.writeheader()
                    writer.writerow(row)
                    handle.flush()
                    print(f"{name} t={start}: input_DE={row['input_delta_e76']:.3f} "
                          f"direct_DE={row['direct_delta_e76']:.3f} algorithm2_DE={row['algorithm2_delta_e76']:.3f}", flush=True)
                manifest.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    means = {}
    for start in starts:
        subset = [row for row in rows if row["start_step"] == start]
        means[str(start)] = {}
        for key in subset[0]:
            if key in {"image", "start_step"}:
                continue
            values = [row[key] for row in subset if row[key] is not None]
            means[str(start)][key] = sum(values)/len(values) if values else None
    metadata["status"] = "inference_complete" if args.extended_metrics else "complete"
    metrics = {"evaluation": metadata, "means_by_start_step": means}
    metrics_path = output / "metrics.json"
    manifest.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if args.extended_metrics:
        # Persist inference/core scores before optional model downloads or IQA failures.
        del inference_model, model, trajectory, bridge
        if device.type == "cuda":
            torch.cuda.empty_cache()
        from gray_cold_diffusion.extended_metrics import create_pyiqa_metrics, evaluate_extended_metrics

        iqa = create_pyiqa_metrics(device)
        metrics["extended_raw"] = evaluate_extended_metrics(
            output / "raw", output / "references", selected_names,
            output / "raw_extended_metrics.csv", device, args.extended_metric_size,
            pyiqa_metrics=iqa, progress_label="raw_metrics",
        )
        metrics["extended_by_start_step"] = {}
        for start in starts:
            level = output / f"retain_{(1-start/steps)*100:g}pct"
            metrics["extended_by_start_step"][str(start)] = {}
            for method, folder in [("direct", "direct_predictions"), ("algorithm2", "predictions")]:
                cold_scores = {}
                for row in rows:
                    if row["start_step"] == start:
                        cold_scores[row["image"]] = {"delta_e76": row[f"{method}_delta_e76"]}
                        if method == "algorithm2":
                            cold_scores[row["image"]]["trajectory_monotonic"] = row["algorithm2_trajectory_monotonic"]
                metrics["extended_by_start_step"][str(start)][method] = evaluate_extended_metrics(
                    level / folder, output / "references", selected_names,
                    level / f"{method}_extended_metrics.csv", device, args.extended_metric_size,
                    cold_scores=cold_scores, pyiqa_metrics=iqa, progress_label=f"t{start}_{method}_metrics",
                )
                metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        metadata["status"] = "complete"
        manifest.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"DIAGNOSTIC COMPLETE: {output}", flush=True)
