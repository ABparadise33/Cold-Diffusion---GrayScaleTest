"""Download and validate the official DIV2K high-resolution train/val images."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from zipfile import BadZipFile, ZipFile

from PIL import Image


BASE_URL = "https://data.vision.ee.ethz.ch/cvl/DIV2K"
CHUNK_SIZE = 8 * 1024 * 1024


@dataclass(frozen=True)
class SplitSpec:
    name: str
    archive: str
    directory: str
    expected_count: int
    first_id: int
    last_id: int

    @property
    def url(self) -> str:
        return f"{BASE_URL}/{self.archive}"


SPECS = {
    "train": SplitSpec("train", "DIV2K_train_HR.zip", "DIV2K_train_HR", 800, 1, 800),
    "val": SplitSpec("val", "DIV2K_valid_HR.zip", "DIV2K_valid_HR", 100, 801, 900),
}


def _format_bytes(value: int | None) -> str:
    if value is None:
        return "unknown"
    return f"{value / 1024**2:.1f} MiB"


def _remote_size(url: str) -> int | None:
    try:
        with urlopen(Request(url, method="HEAD"), timeout=60) as response:
            length = response.headers.get("Content-Length")
            return int(length) if length else None
    except (HTTPError, URLError, TimeoutError):
        return None


def download_with_resume(url: str, destination: Path, retries: int = 5) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    total = _remote_size(url)
    existing = destination.stat().st_size if destination.exists() else 0
    if total is not None and existing == total:
        print(f"Archive already complete: {destination} ({_format_bytes(total)})")
        return
    if total is not None and existing > total:
        raise RuntimeError(
            f"local archive is larger than the official file: {destination} "
            f"({existing} > {total} bytes)"
        )

    for attempt in range(1, retries + 1):
        existing = destination.stat().st_size if destination.exists() else 0
        headers = {"Range": f"bytes={existing}-"} if existing else {}
        request = Request(url, headers=headers)
        try:
            with urlopen(request, timeout=120) as response:
                status = getattr(response, "status", response.getcode())
                resumed = existing > 0 and status == 206
                mode = "ab" if resumed else "wb"
                downloaded = existing if resumed else 0
                response_length = response.headers.get("Content-Length")
                expected = total
                if expected is None and response_length:
                    expected = downloaded + int(response_length)
                print(
                    f"Downloading {url}\n"
                    f"  -> {destination}\n"
                    f"  resume={resumed} start={_format_bytes(downloaded)} "
                    f"total={_format_bytes(expected)}"
                )
                last_report = time.monotonic()
                with destination.open(mode) as handle:
                    while True:
                        chunk = response.read(CHUNK_SIZE)
                        if not chunk:
                            break
                        handle.write(chunk)
                        downloaded += len(chunk)
                        now = time.monotonic()
                        if now - last_report >= 2:
                            percent = ""
                            if expected:
                                percent = f" ({downloaded / expected * 100:.1f}%)"
                            print(f"  {_format_bytes(downloaded)}{percent}", flush=True)
                            last_report = now
            final_size = destination.stat().st_size
            if expected is not None and final_size != expected:
                raise IOError(f"incomplete download: {final_size} of {expected} bytes")
            print(f"Download complete: {destination} ({_format_bytes(final_size)})")
            return
        except (HTTPError, URLError, TimeoutError, OSError) as error:
            if attempt == retries:
                raise RuntimeError(f"download failed after {retries} attempts: {url}") from error
            delay = min(2**attempt, 30)
            print(f"Download interrupted ({error}); retrying in {delay}s ...")
            time.sleep(delay)


def _safe_extract(archive: Path, output_root: Path) -> None:
    try:
        with ZipFile(archive) as handle:
            bad_member = handle.testzip()
            if bad_member:
                raise BadZipFile(f"CRC check failed: {bad_member}")
            root = output_root.resolve()
            for member in handle.infolist():
                destination = (output_root / member.filename).resolve()
                if root != destination and root not in destination.parents:
                    raise BadZipFile(f"unsafe archive member: {member.filename}")
            print(f"Extracting {archive} ...")
            handle.extractall(output_root)
    except BadZipFile as error:
        raise RuntimeError(f"invalid ZIP archive: {archive}: {error}") from error


def validate_split(root: Path, spec: SplitSpec) -> Path:
    image_dir = root / spec.directory
    if not image_dir.is_dir():
        raise FileNotFoundError(f"missing extracted directory: {image_dir}")
    images = sorted(image_dir.glob("*.png"))
    if len(images) != spec.expected_count:
        raise RuntimeError(
            f"{spec.name}: expected {spec.expected_count} PNG files, found {len(images)} in {image_dir}"
        )
    expected_names = {f"{index:04d}.png" for index in range(spec.first_id, spec.last_id + 1)}
    actual_names = {path.name for path in images}
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)[:5]
        unexpected = sorted(actual_names - expected_names)[:5]
        raise RuntimeError(f"{spec.name}: filename mismatch missing={missing} unexpected={unexpected}")
    for path in (images[0], images[-1]):
        with Image.open(path) as image:
            image.verify()
    print(f"DIV2K {spec.name} OK: {len(images)} images in {image_dir}")
    return image_dir


def prepare_split(root: Path, spec: SplitSpec, skip_download: bool, delete_archive: bool) -> Path:
    try:
        return validate_split(root, spec)
    except (FileNotFoundError, RuntimeError) as existing_error:
        archive = root / spec.archive
        if not archive.is_file():
            if skip_download:
                raise RuntimeError(
                    f"{existing_error}; archive is also missing and --skip-download was used"
                ) from existing_error
            download_with_resume(spec.url, archive)
        _safe_extract(archive, root)
        image_dir = validate_split(root, spec)
        if delete_archive:
            archive.unlink()
            print(f"Deleted verified archive to save space: {archive}")
        return image_dir


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=str(repo_root / "data" / "DIV2K"))
    parser.add_argument("--split", choices=["train", "val", "all"], default="all")
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="validate/extract existing local data without network access",
    )
    parser.add_argument(
        "--delete-archives",
        action="store_true",
        help="delete ZIP files only after extraction and validation succeed",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    data_root.mkdir(parents=True, exist_ok=True)
    selected = SPECS.values() if args.split == "all" else [SPECS[args.split]]
    prepared = [
        prepare_split(data_root, spec, args.skip_download, args.delete_archives)
        for spec in selected
    ]
    print("DIV2K DATASET READY")
    for directory in prepared:
        print(directory)


if __name__ == "__main__":
    main()
