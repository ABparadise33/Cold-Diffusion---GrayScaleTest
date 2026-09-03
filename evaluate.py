from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from gray_cold_diffusion.bridge import GrayBridge
from gray_cold_diffusion.color import (
    denormalize_rgb,
    gray_anchor,
    normalize_rgb,
    normalized_lab_to_rgb,
    rgb_channel_mean_gray,
    rgb_to_normalized_lab,
)
from gray_cold_diffusion.data import PairedImageDataset
from gray_cold_diffusion.extended_metrics import (
    create_pyiqa_metrics,
    evaluate_extended_metrics,
    save_method_comparison,
)
from gray_cold_diffusion.io import save_stage_strip, save_tensor_image, save_trajectory_grid, select_device
from gray_cold_diffusion.metrics import delta_e76, psnr, ssim, trajectory_monotonic_fraction
from gray_cold_diffusion.model import RestorationUNet
from gray_cold_diffusion.report import save_training_report
from gray_cold_diffusion.tiling import TiledModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--output-dir", default="evaluation")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    spatial_group = parser.add_mutually_exclusive_group()
    spatial_group.add_argument("--image-size", type=int, help="square evaluation crop; defaults to training size")
    spatial_group.add_argument(
        "--original-size",
        action="store_true",
        help="preserve each image's original width, height, and aspect ratio",
    )
    parser.add_argument("--preview-count", type=int, default=3)
    parser.add_argument("--preview-scale", type=int, help="display-only upscaling; automatic if omitted")
    parser.add_argument(
        "--preview-max-side",
        type=int,
        help="display-only maximum side for comparison and trajectory tiles",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        help="overlapping inference tile size; recommended for full-resolution DIV2K",
    )
    parser.add_argument("--tile-overlap", type=int, default=64)
    parser.add_argument("--training-metrics", help="metrics.csv; auto-detected from checkpoint if omitted")
    parser.add_argument("--extended-metrics", action="store_true", help="run the 14 FlowIE metrics")
    parser.add_argument(
        "--extended-metric-size",
        type=int,
        default=256,
        help="FlowIE metric resize; keep 256 for comparable future runs",
    )
    args = parser.parse_args()

    device = select_device(args.device)
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model_cfg = config["model"]
    steps = int(config["diffusion"]["steps"])
    model = RestorationUNet(
        int(model_cfg["base_channels"]), tuple(model_cfg["channel_mults"]),
        float(model_cfg.get("dropout", 0.0)), steps,
    ).to(device)
    model.load_state_dict(checkpoint["ema"])
    model.eval()
    bridge = GrayBridge(steps).to(device)
    if args.original_size and args.batch_size != 1:
        parser.error("--original-size requires --batch-size 1 because UIEB image sizes vary")
    if args.tile_size is not None:
        if args.batch_size != 1:
            parser.error("--tile-size requires --batch-size 1")
        if args.tile_size < 1:
            parser.error("--tile-size must be >= 1")
        if args.tile_overlap < 0 or args.tile_overlap >= args.tile_size:
            parser.error("--tile-overlap must satisfy 0 <= overlap < tile-size")
    inference_model = (
        TiledModel(model, args.tile_size, args.tile_overlap)
        if args.tile_size is not None
        else model
    )
    image_size = None if args.original_size else (args.image_size or int(config["data"]["image_size"]))
    dataset = PairedImageDataset(
        args.raw_dir, args.reference_dir, args.split_file, args.split,
        image_size, augment=False,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    totals = {"psnr": 0.0, "ssim": 0.0, "delta_e76": 0.0, "monotonic": 0.0}
    iterative_modes = {
        "cold_gray",
        "natural_rgb_colorization",
        "natural_lab_colorization",
    }
    direct_totals = (
        {"psnr": 0.0, "ssim": 0.0, "delta_e76": 0.0}
        if config["mode"] in iterative_modes
        else None
    )
    count = 0
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = (
        Path(args.training_metrics)
        if args.training_metrics
        else checkpoint_path.resolve().parent.parent / "metrics.csv"
    )
    training_summary = None
    if metrics_path.is_file():
        training_summary = save_training_report(metrics_path, output)
        print(f"training_curves={output / 'training_curves.png'}")
        print(f"resume_hint={training_summary['resume_hint']}")
    else:
        print(f"warning: training metrics not found: {metrics_path}")

    mode = config["mode"]
    preview_saved = 0
    preview_scale = args.preview_scale or (1 if image_size is None else max(1, math.ceil(512 / image_size)))
    exported_names = []
    cold_scores = {}
    direct_scores = {}
    prediction_dir = output / "predictions"
    direct_prediction_dir = output / "direct_predictions"
    evaluation_reference_dir = output / "references"
    with torch.no_grad():
        for batch in loader:
            raw = batch["raw"].to(device)
            reference = batch["reference"].to(device)
            target_lab = rgb_to_normalized_lab(reference)
            if mode == "natural_rgb_colorization":
                raw_state = normalize_rgb(raw)
                anchor = rgb_channel_mean_gray(raw_state)
                state_to_rgb = denormalize_rgb
                trajectory_color_space = "rgb"
            else:
                raw_state = rgb_to_normalized_lab(raw)
                anchor = gray_anchor(raw_state)
                state_to_rgb = normalized_lab_to_rgb
                trajectory_color_space = "lab"
            t = torch.full((raw.shape[0],), steps, device=device, dtype=torch.long)
            if mode in iterative_modes:
                direct_state = inference_model(anchor, t).clamp(-1, 1)
                direct = state_to_rgb(direct_state)
                pred_state, trajectory = bridge.sample(
                    inference_model, anchor, return_trajectory=True
                )
            else:
                direct_state = None
                direct = None
                state = anchor if mode == "gray_oneshot" else raw_state
                pred_state = inference_model(state, t).clamp(-1, 1)
                trajectory = [state, pred_state]
            pred = state_to_rgb(pred_state)
            pred_lab = rgb_to_normalized_lab(pred)
            trajectory_lab = [rgb_to_normalized_lab(state_to_rgb(state)) for state in trajectory]
            if pred.shape[-2:] != raw.shape[-2:]:
                raise RuntimeError(
                    f"prediction geometry changed from {tuple(raw.shape[-2:])} "
                    f"to {tuple(pred.shape[-2:])}"
                )
            if direct is not None and direct.shape[-2:] != raw.shape[-2:]:
                raise RuntimeError(
                    f"direct geometry changed from {tuple(raw.shape[-2:])} "
                    f"to {tuple(direct.shape[-2:])}"
                )
            batch_psnr = psnr(pred, reference)
            batch_ssim = ssim(pred, reference)
            batch_delta_e = delta_e76(pred_lab, target_lab)
            batch_monotonic = trajectory_monotonic_fraction(trajectory_lab, target_lab)
            totals["psnr"] += batch_psnr.sum().item()
            totals["ssim"] += batch_ssim.sum().item()
            totals["delta_e76"] += batch_delta_e.sum().item()
            totals["monotonic"] += batch_monotonic.sum().item()
            if direct_totals is not None:
                batch_direct_psnr = psnr(direct, reference)
                batch_direct_ssim = ssim(direct, reference)
                direct_lab = rgb_to_normalized_lab(direct)
                batch_direct_delta_e = delta_e76(direct_lab, target_lab)
                direct_totals["psnr"] += batch_direct_psnr.sum().item()
                direct_totals["ssim"] += batch_direct_ssim.sum().item()
                direct_totals["delta_e76"] += batch_direct_delta_e.sum().item()
            count += raw.shape[0]
            if args.extended_metrics:
                for image_index, name in enumerate(batch["name"]):
                    save_tensor_image(pred[image_index], prediction_dir / f"{name}.png")
                    save_tensor_image(reference[image_index], evaluation_reference_dir / f"{name}.png")
                    if direct is not None:
                        save_tensor_image(direct[image_index], direct_prediction_dir / f"{name}.png")
                    exported_names.append(name)
                    cold_scores[name] = {
                        "delta_e76": float(batch_delta_e[image_index].item()),
                        "trajectory_monotonic": float(batch_monotonic[image_index].item()),
                    }
                    if direct is not None:
                        direct_scores[name] = {
                            "delta_e76": float(batch_direct_delta_e[image_index].item()),
                        }
            for image_index in range(raw.shape[0]):
                if preview_saved >= args.preview_count:
                    break
                stages = [("raw", raw), ("gray", state_to_rgb(anchor))]
                if direct is not None:
                    stages.extend([("direct", direct), ("Algorithm 2", pred)])
                else:
                    stages.append(("prediction", pred))
                stages.append(("reference", reference))
                save_stage_strip(
                    stages,
                    output / f"batch_{preview_saved:03d}.png",
                    image_index=image_index,
                    display_scale=preview_scale,
                    max_side=args.preview_max_side,
                )
                if mode in iterative_modes:
                    save_trajectory_grid(
                        trajectory,
                        output / f"trajectory_{preview_saved:03d}.png",
                        image_index=image_index,
                        display_scale=preview_scale,
                        color_space=trajectory_color_space,
                        max_side=args.preview_max_side,
                    )
                preview_saved += 1
            print(f"inference {count}/{len(dataset)}")
    metrics = {key: value / count for key, value in totals.items()}
    if direct_totals is not None:
        metrics["direct"] = {key: value / count for key, value in direct_totals.items()}
    split_path = Path(args.split_file)
    metrics["evaluation"] = {
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_step": int(checkpoint.get("step", -1)),
        "split": args.split,
        "num_images": count,
        "image_size": image_size,
        "spatial_mode": "original_size" if image_size is None else "square_crop",
        "tile_size": args.tile_size,
        "tile_overlap": args.tile_overlap if args.tile_size is not None else None,
        "split_file": str(split_path.resolve()),
        "split_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
    }
    if mode in {"natural_rgb_colorization", "natural_lab_colorization"}:
        metrics["evaluation"]["training_saturation_factor"] = float(
            config.get("data", {}).get("saturation_factor", 1.0)
        )
    if "reference_saturation_factor" in config.get("data", {}):
        metrics["evaluation"]["training_reference_lab_saturation_factor"] = float(
            config["data"]["reference_saturation_factor"]
        )

    if args.extended_metrics:
        del inference_model, model, bridge
        if device.type == "cuda":
            torch.cuda.empty_cache()
        pyiqa_metrics = create_pyiqa_metrics(device)
        metrics["extended"] = evaluate_extended_metrics(
            prediction_dir=prediction_dir,
            reference_dir=evaluation_reference_dir,
            names=exported_names,
            output_csv=output / "extended_metrics.csv",
            device=device,
            eval_size=args.extended_metric_size,
            cold_scores=cold_scores,
            pyiqa_metrics=pyiqa_metrics,
            progress_label="algorithm2_metrics",
        )
        if direct_totals is not None:
            metrics["direct_extended"] = evaluate_extended_metrics(
                prediction_dir=direct_prediction_dir,
                reference_dir=evaluation_reference_dir,
                names=exported_names,
                output_csv=output / "direct_metrics.csv",
                device=device,
                eval_size=args.extended_metric_size,
                cold_scores=direct_scores,
                pyiqa_metrics=pyiqa_metrics,
                progress_label="direct_metrics",
            )
            comparison = save_method_comparison(
                metrics["direct_extended"]["means"],
                metrics["extended"]["means"],
                output / "direct_vs_algorithm2.csv",
            )
            metrics["direct_vs_algorithm2"] = comparison
        metrics["evaluation"]["extended_metric_size"] = args.extended_metric_size
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
