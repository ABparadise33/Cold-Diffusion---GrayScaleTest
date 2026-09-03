"""Synthetic DIV2K partial-chroma diagnosis, not a restoration benchmark."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import random

import torch
import yaml

from gray_cold_diffusion.bridge import GrayBridge
from gray_cold_diffusion.color import gray_anchor, normalized_lab_to_rgb, rgb_to_normalized_lab
from gray_cold_diffusion.data import PairedImageDataset
from gray_cold_diffusion.diagnostics import check_diagnostic_checkpoint, sample_from_step
from gray_cold_diffusion.io import save_stage_strip, save_tensor_image, select_device
from gray_cold_diffusion.metrics import delta_e76, psnr
from gray_cold_diffusion.model import RestorationUNet
from gray_cold_diffusion.tiling import TiledModel


def score(rgb, reference, reference_lab):
    lab = rgb_to_normalized_lab(rgb)
    ab = lab[:, 1:] * 128.0
    return {
        "psnr_rgb": float(psnr(rgb, reference).mean().item()),
        "delta_e76": float(delta_e76(lab, reference_lab).mean().item()),
        "chroma": float(ab.square().sum(dim=1).sqrt().mean().item()),
        "mean_a": float(ab[:, 0].mean().item()),
        "mean_b": float(ab[:, 1].mean().item()),
    }


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-checkpoint-step", type=int, default=50000)
    parser.add_argument("--baseline-config", default=str(root / "configs/div2k_uieb_style_lab_sat1_50k.yaml"))
    parser.add_argument("--image-dir", default=str(root / "data/DIV2K/DIV2K_valid_HR"))
    parser.add_argument("--split-file", default=str(root / "splits/div2k_valid_all.json"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--start-steps", type=int, nargs="+", default=[4, 6, 7])
    parser.add_argument("--include-gray-control", action="store_true")
    parser.add_argument("--limit", type=int, default=4, help="fixed-seed subset; 0 uses all images")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--tile-overlap", type=int, default=64)
    parser.add_argument("--preview-max-side", type=int, help="optional display-only reduction; no reduction by default")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be >= 0")
    if args.tile_size < 0 or not 0 <= args.tile_overlap < max(args.tile_size, 1):
        # --tile-size 0 --tile-overlap 0 explicitly disables tiling.
        parser.error("use 0 <= tile-overlap < tile-size, or set both to 0 to disable tiling")
    if args.preview_max_side is not None and args.preview_max_side < 1:
        parser.error("--preview-max-side must be >= 1")
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        parser.error(f"output directory is not empty; use a new directory: {output}")
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    baseline = yaml.safe_load(Path(args.baseline_config).read_text(encoding="utf-8"))
    checkpoint_step = int(checkpoint.get("step", -1))
    try:
        check_diagnostic_checkpoint(config, baseline, checkpoint_step, args.expected_checkpoint_step)
    except ValueError as error:
        parser.error(str(error))
    steps = int(config["diffusion"]["steps"])
    starts = sorted(set(args.start_steps))
    if not starts or any(step < 1 or step >= steps for step in starts):
        parser.error(f"--start-steps must be in [1,{steps-1}]; use --include-gray-control to add full gray")
    if args.include_gray_control:
        starts.append(steps)

    dataset = PairedImageDataset(args.image_dir, args.image_dir, args.split_file, "test", None)
    indices = list(range(len(dataset)))
    if 0 < args.limit < len(indices):
        indices = sorted(random.Random(args.seed).sample(indices, args.limit))
    selected_names = [Path(dataset._paths(dataset.items[i])[0]).stem for i in indices]
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
    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "status": "running",
        "purpose": "synthetic partial-chroma diagnostic; input includes target color information",
        "analytic_control": "For retained_chroma > 0, dividing a/b by that fraction exactly inverts this synthetic degradation before RGB quantization",
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_step": checkpoint_step,
        "checkpoint_config": config,
        "baseline_config": str(Path(args.baseline_config).resolve()),
        "baseline_config_sha256": hashlib.sha256(Path(args.baseline_config).read_bytes()).hexdigest(),
        "diffusion_steps": steps,
        "start_steps": starts,
        "retained_chroma": {str(t): 1-t/steps for t in starts},
        "spatial_mode": "original_size",
        "image_dir": str(Path(args.image_dir).resolve()),
        "split_sha256": hashlib.sha256(Path(args.split_file).read_bytes()).hexdigest(),
        "subset_seed": args.seed,
        "selected_names": selected_names,
        "tile_size": args.tile_size,
        "tile_overlap": args.tile_overlap,
        "device": str(device),
        "torch_version": str(torch.__version__),
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
                reference = item["reference"].unsqueeze(0).to(device)
                clean = rgb_to_normalized_lab(reference)
                anchor = gray_anchor(clean)
                save_tensor_image(reference[0], output / "references" / f"{name}.png")
                for start in starts:
                    retained = 1-start/steps
                    percent = f"{retained*100:g}"
                    level = output / f"retain_{percent}pct"
                    t = torch.full((1,), start, device=device, dtype=torch.long)
                    initial = bridge.degrade(clean, anchor, t)
                    direct = normalized_lab_to_rgb(inference_model(initial, t).clamp(-1, 1))
                    final, trajectory = sample_from_step(inference_model, bridge, initial, anchor, start)
                    sampled = normalized_lab_to_rgb(final)
                    initial_rgb = normalized_lab_to_rgb(initial)
                    analytic = (
                        normalized_lab_to_rgb((initial - (1-retained)*anchor) / retained)
                        if retained > 0 else None
                    )
                    for kind, image in [("inputs", initial_rgb), ("direct_predictions", direct), ("predictions", sampled)]:
                        if image.shape != reference.shape:
                            raise RuntimeError(f"{kind} geometry changed for {name}")
                        save_tensor_image(image[0], level / kind / f"{name}.png")
                    stages = [
                        (f"input t={start}/{steps}, a/b={retained*100:g}%", initial_rgb),
                        ("Direct", direct), ("Algorithm 2", sampled),
                    ]
                    if analytic is not None:
                        stages.append(("analytic inverse (control)", analytic))
                    stages.append(("reference", reference))
                    save_stage_strip(stages, level / "batches" / f"{name}.png", max_side=args.preview_max_side)
                    save_stage_strip([
                        (f"t={start-i}/{steps}", normalized_lab_to_rgb(state))
                        for i, state in enumerate(trajectory)
                    ], level / "trajectories" / f"{name}.png", max_side=args.preview_max_side)
                    row = {"image": name, "start_step": start, "retained_chroma": retained}
                    for method, image in [("input", initial_rgb), ("direct", direct), ("algorithm2", sampled), ("reference", reference)]:
                        row.update({f"{method}_{key}": value for key, value in score(image, reference, clean).items()})
                    analytic_scores = score(analytic, reference, clean) if analytic is not None else None
                    row["analytic_delta_e76"] = analytic_scores["delta_e76"] if analytic_scores else None
                    rows.append(row)
                    if writer is None:
                        writer = csv.DictWriter(handle, fieldnames=list(row))
                        writer.writeheader()
                    writer.writerow(row)
                    handle.flush()
                    print(f"{name} t={start}: input_DE={row['input_delta_e76']:.3f} "
                          f"direct_DE={row['direct_delta_e76']:.3f} algorithm2_DE={row['algorithm2_delta_e76']:.3f}", flush=True)
    means = {}
    for start in starts:
        subset = [row for row in rows if row["start_step"] == start]
        means[str(start)] = {}
        for key in subset[0]:
            if key in {"image", "start_step"}:
                continue
            values = [row[key] for row in subset if row[key] is not None]
            means[str(start)][key] = sum(values)/len(values) if values else None
    metadata["status"] = "complete"
    manifest.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (output / "metrics.json").write_text(json.dumps({"evaluation": metadata, "means_by_start_step": means}, indent=2), encoding="utf-8")
    print(f"DIAGNOSTIC COMPLETE: {output}", flush=True)


if __name__ == "__main__":
    main()
