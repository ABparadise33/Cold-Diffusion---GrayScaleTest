from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import yaml

from gray_cold_diffusion.bridge import GrayBridge
from gray_cold_diffusion.data import PairedImageDataset, seed_worker
from gray_cold_diffusion.engine import Trainer
from gray_cold_diffusion.io import select_device, set_seed
from gray_cold_diffusion.model import RestorationUNet


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", nargs="?", const="auto")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--grad-accum", type=int)
    parser.add_argument("--num-workers", type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.max_steps is not None:
        config["training"]["max_steps"] = args.max_steps
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.grad_accum is not None:
        config["training"]["grad_accum"] = args.grad_accum
    if args.num_workers is not None:
        config["data"]["num_workers"] = args.num_workers

    set_seed(int(config["seed"]))
    device = select_device(args.device)
    print(f"device={device} mode={config['mode']} max_steps={config['training']['max_steps']}")

    image_size = int(config["data"]["image_size"])
    train_dataset = PairedImageDataset(args.raw_dir, args.reference_dir, args.split_file, "train", image_size, augment=True)
    val_dataset = PairedImageDataset(args.raw_dir, args.reference_dir, args.split_file, "val", image_size, augment=False)
    generator = torch.Generator().manual_seed(int(config["seed"]))
    loader_args = dict(
        batch_size=int(config["training"]["batch_size"]),
        num_workers=int(config["data"]["num_workers"]),
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=generator,
    )
    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=True, **loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False, **loader_args)

    model_cfg = config["model"]
    steps = int(config["diffusion"]["steps"])
    model = RestorationUNet(
        base_channels=int(model_cfg["base_channels"]),
        channel_mults=tuple(model_cfg["channel_mults"]),
        dropout=float(model_cfg.get("dropout", 0.0)),
        diffusion_steps=steps,
    )
    print(f"parameters={sum(p.numel() for p in model.parameters()):,}")
    trainer = Trainer(model, GrayBridge(steps), train_loader, val_loader, config, device)

    if args.resume:
        resume_path = Path(config["output_dir"]) / "checkpoints" / "latest.pt" if args.resume == "auto" else Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"resume checkpoint not found: {resume_path}")
        trainer.load_checkpoint(resume_path)
    trainer.train()


if __name__ == "__main__":
    main()
