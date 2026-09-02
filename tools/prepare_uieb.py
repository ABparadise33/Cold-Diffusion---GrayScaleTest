"""Download the UIEB mirror used by Underwater_FlowIE and create a fixed split."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from PIL import Image


EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def index_images(folder: Path) -> dict[str, Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f"missing image directory: {folder}")
    return {
        path.stem: path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in EXTENSIONS
    }


def clone_dataset(repo_url: str, data_root: Path) -> None:
    if shutil.which("git") is None:
        raise SystemExit("ERROR: git is required to download UIEB")
    lfs = subprocess.run(
        ["git", "lfs", "version"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if lfs.returncode != 0:
        raise SystemExit(
            "ERROR: git-lfs is required. Install git-lfs, run `git lfs install`, then retry."
        )

    data_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="uieb_clone_", dir=data_root.parent) as temporary:
        clone_root = Path(temporary) / "UIEB"
        print(f"Cloning UIEB with Git LFS: {repo_url}")
        subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, str(clone_root)],
            check=True,
        )
        for directory in ("raw-890", "reference-890"):
            source = clone_root / directory
            if not source.is_dir():
                raise SystemExit(f"ERROR: downloaded repository is missing {directory}")
            shutil.copytree(source, data_root / directory, dirs_exist_ok=True)


def validate_dataset(raw_dir: Path, reference_dir: Path) -> tuple[dict, dict]:
    raw = index_images(raw_dir)
    reference = index_images(reference_dir)
    common = set(raw) & set(reference)
    if len(common) < 890:
        raise SystemExit(
            f"ERROR: expected 890 paired images, found raw={len(raw)}, "
            f"reference={len(reference)}, paired={len(common)}"
        )
    # Git without LFS leaves small text pointers instead of valid images.
    for path in (raw[sorted(common)[0]], reference[sorted(common)[0]]):
        try:
            with Image.open(path) as image:
                image.verify()
        except Exception as error:
            raise SystemExit(
                f"ERROR: {path} is not a valid image; check that Git LFS downloaded the files"
            ) from error
    return raw, reference


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=str(repo_root / "data" / "UIEB"))
    parser.add_argument("--output", default=str(repo_root / "splits" / "uieb_seed42.json"))
    parser.add_argument(
        "--repo-url",
        default="https://huggingface.co/datasets/Edddddd8787/UIEB",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="only validate existing raw-890/reference-890 folders and create the split",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    raw_dir = data_root / "raw-890"
    reference_dir = data_root / "reference-890"

    raw = index_images(raw_dir) if raw_dir.is_dir() else {}
    reference = index_images(reference_dir) if reference_dir.is_dir() else {}
    common = set(raw) & set(reference)
    if len(common) < 890:
        if args.skip_download:
            raise SystemExit(
                f"ERROR: expected 890 existing pairs under {data_root}, found {len(common)}"
            )
        data_root.mkdir(parents=True, exist_ok=True)
        clone_dataset(args.repo_url, data_root)

    raw, reference = validate_dataset(raw_dir, reference_dir)
    common = set(raw) & set(reference)
    print(f"UIEB OK: raw={len(raw)} reference={len(reference)} paired={len(common)}")

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
