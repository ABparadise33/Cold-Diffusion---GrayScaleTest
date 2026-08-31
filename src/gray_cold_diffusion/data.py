from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from .color import adjust_saturation_from_channel_mean


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array.copy()).permute(2, 0, 1)


class PairedImageDataset(Dataset):
    def __init__(
        self,
        raw_dir: str | Path,
        reference_dir: str | Path,
        split_file: str | Path,
        split: str,
        image_size: int | None = 128,
        augment: bool = False,
    ):
        self.raw_dir = Path(raw_dir)
        self.reference_dir = Path(reference_dir)
        self.image_size = None if image_size is None else int(image_size)
        self.augment = augment
        if self.image_size is None and self.augment:
            raise ValueError("original-size loading does not support random crop augmentation")
        with Path(split_file).open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if split not in manifest:
            raise KeyError(f"split {split!r} not found in {split_file}")
        self.items = manifest[split]
        if not self.items:
            raise ValueError(f"split {split!r} is empty")

    def __len__(self) -> int:
        return len(self.items)

    def _paths(self, item):
        if isinstance(item, str):
            return self.raw_dir / item, self.reference_dir / item
        return self.raw_dir / item["raw"], self.reference_dir / item["reference"]

    def _resize_if_needed(self, raw: Image.Image, reference: Image.Image):
        if self.image_size is None:
            return raw, reference
        width, height = raw.size
        if min(width, height) >= self.image_size:
            return raw, reference
        scale = self.image_size / min(width, height)
        size = (round(width * scale), round(height * scale))
        return (
            raw.resize(size, Image.Resampling.BICUBIC),
            reference.resize(size, Image.Resampling.BICUBIC),
        )

    def __getitem__(self, index: int):
        raw_path, reference_path = self._paths(self.items[index])
        with Image.open(raw_path) as image:
            raw = image.convert("RGB")
        with Image.open(reference_path) as image:
            reference = image.convert("RGB")
        if raw.size != reference.size:
            raise ValueError(f"unaligned pair: {raw_path.name} {raw.size} vs {reference_path.name} {reference.size}")
        raw, reference = self._resize_if_needed(raw, reference)
        if self.image_size is None:
            return {
                "raw": _to_tensor(raw),
                "reference": _to_tensor(reference),
                "name": raw_path.stem,
            }
        width, height = raw.size
        if self.augment:
            left = random.randint(0, width - self.image_size)
            top = random.randint(0, height - self.image_size)
        else:
            left = (width - self.image_size) // 2
            top = (height - self.image_size) // 2
        box = (left, top, left + self.image_size, top + self.image_size)
        raw, reference = raw.crop(box), reference.crop(box)
        if self.augment and random.random() < 0.5:
            raw = raw.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            reference = reference.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return {
            "raw": _to_tensor(raw),
            "reference": _to_tensor(reference),
            "name": raw_path.stem,
        }


def seed_worker(worker_id: int):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


class NaturalImageDataset(Dataset):
    """Single-image colorization data with an on-the-fly saturation target."""

    def __init__(
        self,
        image_dir: str | Path,
        image_size: int = 128,
        saturation_factor: float = 1.0,
        augment: bool = False,
    ):
        self.image_dir = Path(image_dir)
        self.image_size = int(image_size)
        self.saturation_factor = float(saturation_factor)
        self.augment = bool(augment)
        if self.image_size < 1:
            raise ValueError("image_size must be >= 1")
        if self.saturation_factor < 0:
            raise ValueError("saturation_factor must be >= 0")
        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"missing natural-image directory: {self.image_dir}")
        self.items = sorted(
            path
            for path in self.image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.items:
            raise ValueError(f"no supported images found in {self.image_dir}")

    def __len__(self) -> int:
        return len(self.items)

    def _resize_if_needed(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        if min(width, height) >= self.image_size:
            return image
        scale = self.image_size / min(width, height)
        size = (round(width * scale), round(height * scale))
        return image.resize(size, Image.Resampling.BICUBIC)

    def __getitem__(self, index: int):
        path = self.items[index]
        with Image.open(path) as image:
            image = self._resize_if_needed(image.convert("RGB"))
        width, height = image.size
        if self.augment:
            left = random.randint(0, width - self.image_size)
            top = random.randint(0, height - self.image_size)
        else:
            left = (width - self.image_size) // 2
            top = (height - self.image_size) // 2
        box = (left, top, left + self.image_size, top + self.image_size)
        image = image.crop(box)
        if self.augment and random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        source = _to_tensor(image)
        target = adjust_saturation_from_channel_mean(
            source.unsqueeze(0), self.saturation_factor
        ).squeeze(0)
        return {
            "raw": source,
            "reference": target,
            "name": path.stem,
        }
