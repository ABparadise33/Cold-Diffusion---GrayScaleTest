import numpy as np
from PIL import Image

from tools.prepare_div2k import SplitSpec, validate_split


def test_validate_div2k_split_checks_expected_ids(tmp_path):
    image_dir = tmp_path / "tiny_hr"
    image_dir.mkdir()
    for image_id in (1, 2):
        array = np.full((8, 9, 3), 40 * image_id, dtype=np.uint8)
        Image.fromarray(array).save(image_dir / f"{image_id:04d}.png")
    spec = SplitSpec(
        name="tiny",
        archive="tiny.zip",
        directory="tiny_hr",
        expected_count=2,
        first_id=1,
        last_id=2,
    )

    assert validate_split(tmp_path, spec) == image_dir
