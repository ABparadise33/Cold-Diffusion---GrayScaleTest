from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from gray_cold_diffusion.bridge import GrayBridge
from gray_cold_diffusion.color import gray_anchor, normalized_lab_to_rgb, rgb_to_normalized_lab
from gray_cold_diffusion.data import PairedImageDataset
from gray_cold_diffusion.extended_metrics import evaluate_extended_metrics
from gray_cold_diffusion.io import save_stage_strip, save_tensor_image, save_trajectory_grid, select_device
from gray_cold_diffusion.metrics import delta_e76, psnr, ssim, trajectory_monotonic_fraction
from gray_cold_diffusion.model import RestorationUNet
from gray_cold_diffusion.report import save_training_report


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
    parser.add_argument("--image-size", type=int, help="evaluation crop size; defaults to training size")
    parser.add_argument("--preview-count", type=int, default=3)
    parser.add_argument("--preview-scale", type=int, help="display-only upscaling; automatic if omitted")
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
    image_size = args.image_size or int(config["data"]["image_size"])
    dataset = PairedImageDataset(
        args.raw_dir, args.reference_dir, args.split_file, args.split,
        image_size, augment=False,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    totals = {"psnr": 0.0, "ssim": 0.0, "delta_e76": 0.0, "monotonic": 0.0}
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
    preview_scale = args.preview_scale or max(1, math.ceil(512 / image_size))
    exported_names = []
    cold_scores = {}
    prediction_dir = output / "predictions"
    evaluation_reference_dir = output / "references"
    with torch.no_grad():
        for batch in loader:
            raw = batch["raw"].to(device)
            reference = batch["reference"].to(device)
            raw_lab = rgb_to_normalized_lab(raw)
            target_lab = rgb_to_normalized_lab(reference)
            anchor = gray_anchor(raw_lab)
            t = torch.full((raw.shape[0],), steps, device=device, dtype=torch.long)
            if mode == "cold_gray":
                pred_lab, trajectory = bridge.sample(model, anchor, return_trajectory=True)
            else:
                state = anchor if mode == "gray_oneshot" else raw_lab
                pred_lab = model(state, t).clamp(-1, 1)
                trajectory = [state, pred_lab]
            pred = normalized_lab_to_rgb(pred_lab)
            batch_psnr = psnr(pred, reference)
            batch_ssim = ssim(pred, reference)
            batch_delta_e = delta_e76(pred_lab, target_lab)
            batch_monotonic = trajectory_monotonic_fraction(trajectory, target_lab)
            totals["psnr"] += batch_psnr.sum().item()
            totals["ssim"] += batch_ssim.sum().item()
            totals["delta_e76"] += batch_delta_e.sum().item()
            totals["monotonic"] += batch_monotonic.sum().item()
            count += raw.shape[0]
            if args.extended_metrics:
                for image_index, name in enumerate(batch["name"]):
                    save_tensor_image(pred[image_index], prediction_dir / f"{name}.png")
                    save_tensor_image(reference[image_index], evaluation_reference_dir / f"{name}.png")
                    exported_names.append(name)
                    cold_scores[name] = {
                        "delta_e76": float(batch_delta_e[image_index].item()),
                        "trajectory_monotonic": float(batch_monotonic[image_index].item()),
                    }
            for image_index in range(raw.shape[0]):
                if preview_saved >= args.preview_count:
                    break
                save_stage_strip(
                    [
                        ("raw", raw),
                        ("gray", normalized_lab_to_rgb(anchor)),
                        ("prediction", pred),
                        ("reference", reference),
                    ],
                    output / f"batch_{preview_saved:03d}.png",
                    image_index=image_index,
                    display_scale=preview_scale,
                )
                if mode == "cold_gray":
                    save_trajectory_grid(
                        trajectory,
                        output / f"trajectory_{preview_saved:03d}.png",
                        image_index=image_index,
                        display_scale=preview_scale,
                    )
                preview_saved += 1
    metrics = {key: value / count for key, value in totals.items()}
    split_path = Path(args.split_file)
    metrics["evaluation"] = {
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_step": int(checkpoint.get("step", -1)),
        "split": args.split,
        "num_images": count,
        "image_size": image_size,
        "split_file": str(split_path.resolve()),
        "split_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
    }

    if args.extended_metrics:
        del model, bridge, raw, reference, raw_lab, target_lab, anchor, pred_lab, trajectory, pred
        if device.type == "cuda":
            torch.cuda.empty_cache()
        metrics["extended"] = evaluate_extended_metrics(
            prediction_dir=prediction_dir,
            reference_dir=evaluation_reference_dir,
            names=exported_names,
            output_csv=output / "extended_metrics.csv",
            device=device,
            eval_size=args.extended_metric_size,
            cold_scores=cold_scores,
        )
        metrics["evaluation"]["extended_metric_size"] = args.extended_metric_size
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
