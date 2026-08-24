"""Download the UIEB mirror used by Underwater_FlowIE and create a fixed split."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

from huggingface_hub import snapshot_download


EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def index_images(folder: Path) -> dict[str, Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f"missing image directory: {folder}")
    return {
        path.stem: path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in EXTENSIONS
    }


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=str(repo_root / "data" / "UIEB"))
    parser.add_argument("--output", default=str(repo_root / "splits" / "uieb_seed42.json"))
    parser.add_argument("--repo-id", default="Edddddd8787/UIEB")
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="only validate existing raw-890/reference-890 folders and create the split",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    raw_dir = data_root / "raw-890"
    reference_dir = data_root / "reference-890"

    if not raw_dir.is_dir() or not reference_dir.is_dir():
        if args.skip_download:
            raise SystemExit(f"ERROR: place raw-890 and reference-890 under {data_root}")
        data_root.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {args.repo_id} to {data_root} ...")
        snapshot_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            local_dir=data_root,
            allow_patterns=["raw-890/*", "reference-890/*"],
        )

    raw = index_images(raw_dir)
    reference = index_images(reference_dir)
    common = set(raw) & set(reference)
    if len(common) < 890:
        raise SystemExit(
            f"ERROR: expected 890 paired images, found raw={len(raw)}, "
            f"reference={len(reference)}, paired={len(common)}"
        )
    print(f"UIEB OK: raw={len(raw)} reference={len(reference)} paired={len(common)}")

    output = Path(args.output).expanduser().resolve()
    subprocess.run(
        [
            sys.executable,
            str(repo_root / "tools" / "make_uieb_split.py"),
            "--raw-dir",
            str(raw_dir),
            "--reference-dir",
            str(reference_dir),
            "--output",
            str(output),
        ],
        check=True,
        cwd=repo_root,
    )
    print("DATASET READY")
    print(f"raw: {raw_dir}")
    print(f"reference: {reference_dir}")
    print(f"split: {output}")


if __name__ == "__main__":
    main()
