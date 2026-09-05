import copy
import json
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn

from gray_cold_diffusion.color import denormalize_rgb, normalize_rgb
from gray_cold_diffusion.data import _to_tensor
from gray_cold_diffusion.io import _tensor_to_pil
from gray_cold_diffusion.official_colorization import RGBDecolorization, channel_gray
from gray_cold_diffusion.official_preview import save_full_scene_previews, select_preview_images
from gray_cold_diffusion.official_training import (
    PRE_SPATIAL_SOURCE_SHA256,
    compatible_preview_revision,
    implementation_fingerprint,
)
from gray_cold_diffusion.tiling import TiledModel


def test_five_unique_step_seeded_images_leave_global_rng_untouched():
    paths = [Path(f'{i:04d}.png') for i in range(100)]
    before_python, before_torch = random.getstate(), torch.get_rng_state().clone()
    first, seed = select_preview_images(paths, 42, 1000)
    assert len(first) == len(set(first)) == 5
    assert select_preview_images(list(reversed(paths)), 42, 1000) == (first, seed)
    assert select_preview_images(paths, 42, 2000)[0] != first
    assert select_preview_images(paths, 43, 1000)[0] != first
    assert random.getstate() == before_python
    assert torch.equal(torch.get_rng_state(), before_torch)
    assert len(select_preview_images(paths[:2], 42, 1000)[0]) == 2


def test_preview_selection_rejects_empty_or_ambiguous_inputs():
    with pytest.raises(ValueError):
        select_preview_images([], 42, 1000)
    with pytest.raises(ValueError):
        select_preview_images([Path('a.png')], 42, 1000, 0)
    with pytest.raises(ValueError, match='unique stems'):
        select_preview_images([Path('a.jpg'), Path('a.png')], 42, 1000)


class ColorModel(nn.Module):
    def forward(self, x, t):
        return x * .4 + x.new_tensor([.3, -.1, .15]).view(1, 3, 1, 1)


@pytest.mark.parametrize('sampler', ['paper_algorithm2', 'official_code'])
def test_preview_step_folders_five_scenes_geometry_and_same_sampler_pixels(tmp_path, sampler):
    paths = []
    for index in range(8):
        path = tmp_path / f'{index:04d}.png'
        Image.new('RGB', (25 + index, 17 + index), (30 + index, 110, 180)).save(path)
        paths.append(path)
    trainer = SimpleNamespace(
        config={'seed': 42, 'training': {'preview_count': 5, 'preview_tile_size': 16,
                                      'preview_tile_overlap': 4, 'preview_max_side': 512}},
        step=1000, output=tmp_path / 'output', device=torch.device('cpu'),
        ema=ColorModel(), bridge=RGBDecolorization(2, sampler),
        val_loader=SimpleNamespace(dataset=SimpleNamespace(items=paths)),
    )
    before_python, before_torch = random.getstate(), torch.get_rng_state().clone()
    save_full_scene_previews(trainer)
    first = trainer.output / 'previews/step_001000'
    manifest = json.loads((first / 'preview.json').read_text())
    assert manifest['status'] == 'complete' and manifest['actual_count'] == 5
    assert len(manifest['completed_images']) == 5
    for folder in ('samples', 'trajectories', 'predictions', 'direct_predictions'):
        assert len(list((first / folder).glob('*.png'))) == 5
    for row in manifest['completed_images']:
        with Image.open(row['image']) as image:
            rgb = _to_tensor(image.convert('RGB')).unsqueeze(0)
            width, height = image.size
        anchor = channel_gray(normalize_rgb(rgb))
        model = TiledModel(trainer.ema, 16, 4)
        # Independent old preview computation: same input/model/bridge, new paths only.
        with torch.no_grad():
            predicted = denormalize_rgb(trainer.bridge.sample(model, anchor))[0]
            direct = denormalize_rgb(model(anchor, torch.tensor([2])))[0]
        for folder, tensor in [('predictions', predicted), ('direct_predictions', direct)]:
            with Image.open(first / folder / row['filename']) as image:
                assert image.size == (width, height)
                assert np.array_equal(np.asarray(image), np.asarray(_tensor_to_pil(tensor)))
        with Image.open(first / 'trajectories' / row['filename']) as image:
            assert image.size == (width * 3, height + 28)  # horizontal, full-gray plus two updates
    assert random.getstate() == before_python
    assert torch.equal(torch.get_rng_state(), before_torch)
    saved_files = {path: path.read_bytes() for path in first.rglob('*') if path.is_file()}
    save_full_scene_previews(trainer)  # interrupted run may repeat a step before its checkpoint was saved
    assert json.loads((first / 'retry_001/preview.json').read_text())['status'] == 'complete'
    assert all(path.read_bytes() == content for path, content in saved_files.items())
    trainer.step = 2000
    save_full_scene_previews(trainer)
    second = trainer.output / 'previews/step_002000'
    assert json.loads((second / 'preview.json').read_text())['selected_images'] != manifest['selected_images']
    assert json.loads((first / 'preview.json').read_text()) == manifest
    assert not (trainer.output / 'samples').exists()
    assert not (trainer.output / 'trajectories').exists()


def test_preview_migration_does_not_relax_other_source_or_sampler_checks():
    current = {'sampler': 'paper_algorithm2', 'source_sha256': implementation_fingerprint()}
    old = copy.deepcopy(current)
    old['source_sha256']['official_training.py'] = '789c81f26dfa83f4c739c4d2f12adb87b1769a0af593ac98983af4d6cf9c7b3c'
    del old['source_sha256']['official_preview.py']
    assert compatible_preview_revision(current, current)
    assert compatible_preview_revision(old, current)
    for name in old['source_sha256']:
        modified = copy.deepcopy(old)
        modified['source_sha256'][name] = 'unrecognized hash'
        assert not compatible_preview_revision(modified, current)
    modified = copy.deepcopy(old)
    modified['sampler'] = 'official_code'
    assert not compatible_preview_revision(modified, current)
    pre_spatial = copy.deepcopy(current)
    pre_spatial['source_sha256'] = PRE_SPATIAL_SOURCE_SHA256
    assert compatible_preview_revision(pre_spatial, current)
    pre_spatial['source_sha256'] = {**PRE_SPATIAL_SOURCE_SHA256, 'factory.py': 'unrecognized hash'}
    assert not compatible_preview_revision(pre_spatial, current)
