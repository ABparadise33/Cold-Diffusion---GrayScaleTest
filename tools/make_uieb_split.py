from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def index_images(folder: Path):
    result = {}
    for path in folder.iterdir():
        if path.is_file() and path.suffix.lower() in EXTENSIONS:
            if path.stem in result:
                raise ValueError(f"duplicate stem {path.stem!r} in {folder}")
            result[path.stem] = path.name
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--output", default="splits/uieb_seed42.json")
    parser.add_argument("--train", type=int, default=720)
    parser.add_argument("--val", type=int, default=80)
    parser.add_argument("--test", type=int, default=90)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw = index_images(Path(args.raw_dir))
    reference = index_images(Path(args.reference_dir))
    common = sorted(set(raw) & set(reference))
    required = args.train + args.val + args.test
    if len(common) < required:
        raise ValueError(f"need {required} pairs, found {len(common)}")
    if len(common) > required:
        print(f"warning: using {required} of {len(common)} matched pairs")
    random.Random(args.seed).shuffle(common)
    common = common[:required]

    def records(stems):
        return [{"raw": raw[stem], "reference": reference[stem]} for stem in stems]

    manifest = {
        "seed": args.seed,
        "train": records(common[:args.train]),
        "val": records(common[args.train:args.train + args.val]),
        "test": records(common[args.train + args.val:]),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {output}: train={args.train} val={args.val} test={args.test}")


if __name__ == "__main__":
    main()
