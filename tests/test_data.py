import json

import numpy as np
from PIL import Image
import torch

from gray_cold_diffusion.data import NaturalImageDataset, PairedImageDataset


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


def test_natural_dataset_generates_saturation_target_without_modifying_source(tmp_path):
    image_dir = tmp_path / "natural"
    image_dir.mkdir()
    array = np.zeros((40, 48, 3), dtype=np.uint8)
    array[..., 0] = 90
    array[..., 1] = 120
    array[..., 2] = 150
    Image.fromarray(array).save(image_dir / "sample.png")

    original = NaturalImageDataset(
        image_dir, image_size=32, saturation_factor=1.0, augment=False
    )[0]
    stronger = NaturalImageDataset(
        image_dir, image_size=32, saturation_factor=1.5, augment=False
    )[0]

    assert torch.allclose(original["raw"], original["reference"])
    original_chroma = (original["reference"] - original["reference"].mean(0)).abs().mean()
    stronger_chroma = (stronger["reference"] - stronger["reference"].mean(0)).abs().mean()
    assert stronger_chroma > original_chroma
    with Image.open(image_dir / "sample.png") as source:
        assert np.asarray(source)[0, 0].tolist() == [90, 120, 150]


def test_paired_lab_saturation_changes_only_reference(tmp_path):
    raw_dir, reference_dir, split_file = _write_pair(
        tmp_path, "wide.png", (40, 40)
    )
    raw_array = np.full((40, 40, 3), [20, 80, 140], dtype=np.uint8)
    reference_array = np.full((40, 40, 3), [80, 130, 180], dtype=np.uint8)
    Image.fromarray(raw_array).save(raw_dir / "wide.png")
    Image.fromarray(reference_array).save(reference_dir / "wide.png")
    baseline = PairedImageDataset(
        raw_dir,
        reference_dir,
        split_file,
        "test",
        image_size=32,
        augment=False,
        reference_saturation_factor=1.0,
    )[0]
    stronger = PairedImageDataset(
        raw_dir,
        reference_dir,
        split_file,
        "test",
        image_size=32,
        augment=False,
        reference_saturation_factor=1.5,
    )[0]

    assert torch.equal(stronger["raw"], baseline["raw"])
    assert torch.equal(
        baseline["reference"],
        torch.from_numpy(reference_array[:32, :32].copy()).permute(2, 0, 1).float()
        / 255.0,
    )
    baseline_chroma = (
        baseline["reference"].max(0).values
        - baseline["reference"].min(0).values
    ).mean()
    stronger_chroma = (
        stronger["reference"].max(0).values
        - stronger["reference"].min(0).values
    ).mean()
    assert stronger_chroma > baseline_chroma
