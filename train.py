from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import yaml

from gray_cold_diffusion.data import NaturalImageDataset, PairedImageDataset, seed_worker
from gray_cold_diffusion.engine import Trainer
from gray_cold_diffusion.factory import NATURAL_MODES, OFFICIAL_MODE, build_model_and_bridge
from gray_cold_diffusion.io import select_device, set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--raw-dir")
    parser.add_argument("--reference-dir")
    parser.add_argument("--split-file")
    parser.add_argument("--train-dir", help="single-image training directory for natural colorization")
    parser.add_argument("--val-dir", help="single-image validation directory for natural colorization")
    parser.add_argument("--output-dir")
    parser.add_argument("--device", default="auto")
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument("--resume", nargs="?", const="auto")
    resume_group.add_argument(
        "--resume-if-exists",
        action="store_true",
        help="resume latest.pt when present; otherwise start this config from step 0",
    )
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
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        print(
            f"cuda_device={properties.name} "
            f"vram={properties.total_memory / 1024**3:.1f}GB "
            f"torch={torch.__version__} cuda_runtime={torch.version.cuda}"
        )

    image_size = int(config["data"]["image_size"])
    if config["mode"] == OFFICIAL_MODE:
        from gray_cold_diffusion.official_training import preflight_official_run
        preflight_official_run(config, args)
    if config["mode"] in NATURAL_MODES:
        if not args.train_dir or not args.val_dir:
            raise SystemExit(
                f"ERROR: {config['mode']} requires --train-dir and --val-dir"
            )
        saturation_factor = float(config["data"].get("saturation_factor", 1.0))
        train_dataset = NaturalImageDataset(
            args.train_dir,
            image_size,
            saturation_factor=saturation_factor,
            augment=True,
        )
        val_dataset = NaturalImageDataset(
            args.val_dir,
            image_size,
            saturation_factor=saturation_factor,
            augment=False,
        )
        print(
            f"natural_data train={len(train_dataset)} val={len(val_dataset)} "
            f"saturation_factor={saturation_factor:g}"
        )
    else:
        if not args.raw_dir or not args.reference_dir or not args.split_file:
            raise SystemExit(
                "ERROR: paired modes require --raw-dir, --reference-dir, and --split-file"
            )
        reference_saturation_factor = float(
            config["data"].get("reference_saturation_factor", 1.0)
        )
        train_dataset = PairedImageDataset(
            args.raw_dir,
            args.reference_dir,
            args.split_file,
            "train",
            image_size,
            augment=True,
            reference_saturation_factor=reference_saturation_factor,
        )
        val_dataset = PairedImageDataset(
            args.raw_dir,
            args.reference_dir,
            args.split_file,
            "val",
            image_size,
            augment=False,
            reference_saturation_factor=reference_saturation_factor,
        )
        print(
            f"paired_data train={len(train_dataset)} val={len(val_dataset)} "
            f"reference_lab_saturation={reference_saturation_factor:g}x"
        )
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

    if len(train_dataset) < loader_args["batch_size"]:
        raise ValueError("training set is smaller than batch size (drop_last would yield no batches)")
    model, bridge = build_model_and_bridge(config)
    print(f"parameters={sum(p.numel() for p in model.parameters()):,}")
    trainer_class = Trainer
    if config["mode"] == OFFICIAL_MODE:
        from gray_cold_diffusion.official_training import OfficialTrainer
        trainer_class = OfficialTrainer
    trainer = trainer_class(model, bridge, train_loader, val_loader, config, device)

    if args.resume:
        resume_path = Path(config["output_dir"]) / "checkpoints" / "latest.pt" if args.resume == "auto" else Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"resume checkpoint not found: {resume_path}")
        trainer.load_checkpoint(resume_path)
    elif args.resume_if_exists:
        resume_path = Path(config["output_dir"]) / "checkpoints" / "latest.pt"
        if resume_path.exists():
            trainer.load_checkpoint(resume_path)
        else:
            print(f"no checkpoint found; starting from step 0: {resume_path}")
    trainer.train()


if __name__ == "__main__":
    main()
