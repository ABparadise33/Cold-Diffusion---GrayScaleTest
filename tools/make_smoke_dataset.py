"""Create a tiny paired dataset for CLI and resume smoke tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="/tmp/gray_cold_smoke")
    args = parser.parse_args()
    root = Path(args.output)
    raw_dir = root / "raw"
    reference_dir = root / "reference"
    raw_dir.mkdir(parents=True, exist_ok=True)
    reference_dir.mkdir(parents=True, exist_ok=True)

    records = []
    yy, xx = np.mgrid[0:40, 0:40]
    for index in range(8):
        reference = np.stack(
            (
                (xx * 5 + index * 17) % 256,
                (yy * 6 + index * 11) % 256,
                ((xx + yy) * 3 + index * 23) % 256,
            ),
            axis=-1,
        ).astype(np.uint8)
        raw = reference.astype(np.float32)
        raw[..., 0] *= 0.35
        raw[..., 1] = raw[..., 1] * 0.75 + 25
        raw[..., 2] = raw[..., 2] * 0.85 + 35
        raw = np.clip(raw, 0, 255).astype(np.uint8)
        name = f"sample_{index:02d}.png"
        Image.fromarray(raw).save(raw_dir / name)
        Image.fromarray(reference).save(reference_dir / name)
        records.append({"raw": name, "reference": name})

    manifest = {"seed": 7, "train": records[:4], "val": records[4:6], "test": records[6:]}
    split_file = root / "split.json"
    split_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"raw={raw_dir}")
    print(f"reference={reference_dir}")
    print(f"split={split_file}")


if __name__ == "__main__":
    main()
