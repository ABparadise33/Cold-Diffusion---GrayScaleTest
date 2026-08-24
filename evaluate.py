from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from gray_cold_diffusion.bridge import GrayBridge
from gray_cold_diffusion.color import gray_anchor, normalized_lab_to_rgb, rgb_to_normalized_lab
from gray_cold_diffusion.data import PairedImageDataset
from gray_cold_diffusion.io import save_labeled_grid, save_trajectory_grid, select_device
from gray_cold_diffusion.metrics import delta_e76, psnr, ssim, trajectory_monotonic_fraction
from gray_cold_diffusion.model import RestorationUNet


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
    args = parser.parse_args()

    device = select_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
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
    dataset = PairedImageDataset(
        args.raw_dir, args.reference_dir, args.split_file, args.split,
        int(config["data"]["image_size"]), augment=False,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    totals = {"psnr": 0.0, "ssim": 0.0, "delta_e76": 0.0, "monotonic": 0.0}
    count = 0
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    mode = config["mode"]
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
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
            totals["psnr"] += psnr(pred, reference).sum().item()
            totals["ssim"] += ssim(pred, reference).sum().item()
            totals["delta_e76"] += delta_e76(pred_lab, target_lab).sum().item()
            totals["monotonic"] += trajectory_monotonic_fraction(trajectory, target_lab).sum().item()
            count += raw.shape[0]
            if batch_index < 3:
                save_labeled_grid(
                    [("raw", raw), ("gray", normalized_lab_to_rgb(anchor)), ("prediction", pred), ("reference", reference)],
                    output / f"batch_{batch_index:03d}.png",
                )
                if mode == "cold_gray":
                    save_trajectory_grid(trajectory, output / f"trajectory_{batch_index:03d}.png")
    metrics = {key: value / count for key, value in totals.items()}
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
