"""Print the runtime used by a GPU instance and fail on common setup errors."""

from __future__ import annotations

import argparse
import platform
import sys

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--min-vram-gb", type=float, default=0.0)
    args = parser.parse_args()

    print(f"python: {platform.python_version()}")
    print(f"torch: {torch.__version__}")
    print(f"torch_cuda_runtime: {torch.version.cuda}")
    print(f"cuda_available: {torch.cuda.is_available()}")

    if sys.version_info < (3, 10):
        raise SystemExit("ERROR: Python 3.10+ is required")
    if args.require_cuda and not torch.cuda.is_available():
        raise SystemExit("ERROR: CUDA is unavailable; do not start paid training")
    if not torch.cuda.is_available():
        return

    properties = torch.cuda.get_device_properties(0)
    vram_gb = properties.total_memory / 1024**3
    print(f"device: {properties.name}")
    print(f"compute_capability: {properties.major}.{properties.minor}")
    print(f"vram_gb: {vram_gb:.1f}")
    if vram_gb < args.min_vram_gb:
        raise SystemExit(
            f"ERROR: GPU has {vram_gb:.1f} GB VRAM; expected at least {args.min_vram_gb:.1f} GB"
        )

    value = torch.ones(256, 256, device="cuda").square().mean()
    torch.cuda.synchronize()
    print(f"cuda_operation: OK ({value.item():.1f})")


if __name__ == "__main__":
    main()
