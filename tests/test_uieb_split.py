import random

from tools.make_uieb_split import make_manifest


def test_test_split_matches_underwater_flowie_random_sample():
    names = [f"image_{index:02d}" for index in range(10)]
    raw = {name: f"{name}.png" for name in names}
    reference = dict(raw)

    manifest = make_manifest(
        raw,
        reference,
        train_count=5,
        val_count=2,
        test_count=3,
        seed=42,
    )

    expected = random.Random(42).sample(sorted(names), 3)
    actual = [record["raw"].removesuffix(".png") for record in manifest["test"]]
    assert actual == expected
    assert len(manifest["train"]) == 5
    assert len(manifest["val"]) == 2
