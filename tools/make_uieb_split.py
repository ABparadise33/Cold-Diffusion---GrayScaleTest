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


def make_manifest(raw, reference, train_count, val_count, test_count, seed):
    """Match Underwater_FlowIE's seed-42 test sampling exactly."""
    common = sorted(set(raw) & set(reference))
    required = train_count + val_count + test_count
    if len(common) < required:
        raise ValueError(f"need {required} pairs, found {len(common)}")
    if len(common) > required:
        print(f"warning: using {required} of {len(common)} matched pairs")
        common = common[:required]

    rng = random.Random(seed)
    # Underwater_FlowIE/split_dataset.py selects its 90-image test set this way.
    test_stems = rng.sample(common, test_count)
    test_set = set(test_stems)
    remaining = [stem for stem in common if stem not in test_set]
    rng.shuffle(remaining)
    train_stems = remaining[:train_count]
    val_stems = remaining[train_count:train_count + val_count]

    def records(stems):
        return [{"raw": raw[stem], "reference": reference[stem]} for stem in stems]

    return {
        "seed": seed,
        "split_method": "Underwater_FlowIE random.sample test set",
        "train": records(train_stems),
        "val": records(val_stems),
        "test": records(test_stems),
    }


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
    manifest = make_manifest(
        raw,
        reference,
        args.train,
        args.val,
        args.test,
        args.seed,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {output}: train={args.train} val={args.val} test={args.test}")


if __name__ == "__main__":
    main()
