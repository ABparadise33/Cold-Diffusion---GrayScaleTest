import importlib.util
import json
from pathlib import Path

from PIL import Image
import pytest
import torch

from gray_cold_diffusion.mixed_training import domain_weights, state_stats
from gray_cold_diffusion.color import normalize_rgb

spec = importlib.util.spec_from_file_location('prepare_mixed', Path(__file__).parents[1] / 'tools/prepare_mixed_colorization.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_weights_balance_domains():
    paths = [Path('UIEB__a.png'), Path('DIV2K__a.png'), Path('DIV2K__b.png')]
    weights = domain_weights(paths)
    assert weights[0] == weights[1:].sum()
    with pytest.raises(ValueError):
        domain_weights(paths[1:])


def test_prepare_split_and_content_leakage(tmp_path):
    uieb, train, val = [tmp_path / n for n in ('uieb', 'train', 'val')]
    for folder in (uieb, train, val):
        folder.mkdir()
    for i, name in enumerate(('a.png', 'b.png', 'c.png')):
        Image.new('RGB', (16, 16), (i*30, 100, 20)).save(uieb/name)
    Image.new('RGB', (16, 16), (123, 1, 20)).save(train/'0001.png')
    Image.new('RGB', (16, 16), (124, 1, 20)).save(val/'0801.png')
    split = tmp_path/'split.json'
    split.write_text(json.dumps({'train': ['a.png'], 'val': ['b.png'], 'test': ['c.png']}))
    out = tmp_path/'mixed'
    manifest = module.prepare(uieb, split, train, val, out)
    assert len(manifest['records']['train']) == 2
    assert (out/'train/UIEB__a.png').resolve() == (uieb/'a.png').resolve()
    assert module.prepare(uieb, split, train, val, out) == manifest
    (val/'0801.png').write_bytes((train/'0001.png').read_bytes())
    with pytest.raises(ValueError, match='leakage'):
        module.prepare(uieb, split, train, val, tmp_path/'bad')


def test_chroma_diagnostics_detect_gray_and_clipping():
    rgb = torch.zeros(1, 3, 16, 16)
    rgb[:, 0] = 0.8
    target = normalize_rgb(rgb)
    gray = target.mean(1, keepdim=True).expand_as(target)
    assert state_stats(target, rgb)['chroma_ratio'] == pytest.approx(1)
    assert state_stats(gray, rgb)['chroma_ratio'] < 0.001
    assert state_stats(target * 3, rgb)['clipped_fraction'] > 0
