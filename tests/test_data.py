import json

import numpy as np
from PIL import Image

from gray_cold_diffusion.data import PairedImageDataset


def _write_pair(root, name, size):
    raw_dir = root / "raw"
    reference_dir = root / "reference"
    raw_dir.mkdir(exist_ok=True)
    reference_dir.mkdir(exist_ok=True)
    width, height = size
    array = np.zeros((height, width, 3), dtype=np.uint8)
    Image.fromarray(array).save(raw_dir / name)
    Image.fromarray(array).save(reference_dir / name)
    split_file = root / "split.json"
    split_file.write_text(json.dumps({"test": [name]}), encoding="utf-8")
    return raw_dir, reference_dir, split_file


def test_original_size_preserves_rectangular_dimensions(tmp_path):
    raw_dir, reference_dir, split_file = _write_pair(tmp_path, "wide.png", (73, 41))
    dataset = PairedImageDataset(
        raw_dir,
        reference_dir,
        split_file,
        "test",
        image_size=None,
        augment=False,
    )

    sample = dataset[0]

    assert sample["raw"].shape == (3, 41, 73)
    assert sample["reference"].shape == (3, 41, 73)


def test_square_mode_still_center_crops(tmp_path):
    raw_dir, reference_dir, split_file = _write_pair(tmp_path, "wide.png", (73, 41))
    dataset = PairedImageDataset(
        raw_dir,
        reference_dir,
        split_file,
        "test",
        image_size=32,
        augment=False,
    )

    assert dataset[0]["raw"].shape == (3, 32, 32)
