import csv
import importlib.util
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader
import yaml

from gray_cold_diffusion.data import NaturalImageDataset
from gray_cold_diffusion.mixed_training import MixedTrainer, PRE_MONITOR_MIXED_SHA256, mixed_fingerprint
from gray_cold_diffusion.official_colorization import RGBDecolorization
from gray_cold_diffusion.official_training import OfficialTrainer, implementation_fingerprint

spec = importlib.util.spec_from_file_location('mixed_report', Path(__file__).parents[1]/'tools/report_mixed_monitor.py')
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


class TinyColorizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor([.2, -.1, .1]).reshape(1, 3, 1, 1))
    def forward(self, x, t):
        return x*.8 + self.bias


def make_trainer(tmp_path, output='run', max_steps=50000):
    config = yaml.safe_load((Path(__file__).parents[1]/'configs/uieb_div2k_rgb_fullgray_pilot.yaml').read_text())
    config['output_dir'] = str(tmp_path/output)
    config['model'] = {'architecture': 'test_tiny'}
    config['data'].update(image_size=16, num_workers=0)
    config['diffusion']['steps'] = 2
    config['training'].update(batch_size=1, grad_accum=2, max_steps=max_steps, min_free_disk_gb=0,
                              preview_count=1, diagnostic_count_per_domain=2)
    config['implementation'] = {'source_sha256': implementation_fingerprint(), 'mixed_sha256': mixed_fingerprint()}
    loaders = []
    for split in ('train', 'val'):
        folder = tmp_path/split
        folder.mkdir(exist_ok=True)
        for domain, color in [('UIEB', (170, 90, 60)), ('DIV2K', (30, 180, 80))]:
            path = folder/f'{domain}__a.png'
            if not path.exists():
                Image.new('RGB', (23, 19), color).save(path)
        ds = NaturalImageDataset(folder, image_size=16, augment=split=='train')
        loaders.append(DataLoader(ds, batch_size=1, num_workers=0))
    return MixedTrainer(TinyColorizer(), RGBDecolorization(2), *loaders, config, torch.device('cpu'))


def csv_rows(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def test_fixed_records_reports_and_rng(tmp_path):
    t = make_trainer(tmp_path)
    t.model.train()
    t.ema.eval()
    py, np_state, rng = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    for step in [10000, 20000]:
        t.step = step
        t.write_diagnostics()
    assert random.getstate() == py
    assert np.array_equal(np.random.get_state()[1], np_state[1])
    assert torch.equal(torch.get_rng_state(), rng)
    assert t.model.training and not t.ema.training
    assert t.train_loader.dataset.augment
    rows = csv_rows(t.output/'fixed_color_summary.csv')
    assert len(rows) == 24
    assert {r['domain'] for r in rows} == {'UIEB', 'DIV2K'}
    assert all(int(r['count']) == 1 for r in rows)
    result = report.build_report(t.output, stages=[10000, 20000, 30000])
    assert result['compared_stages'] == [10000, 20000]
    assert result['missing_stages'] == [30000]
    assert len(list((t.output/'monitor_report/image_comparisons').glob('*.png'))) == 4
    # An incomplete retry is ignored, never combined with a completed pass.
    original = (t.output/'fixed_color_per_image.csv').read_text()
    row = csv_rows(t.output/'fixed_color_per_image.csv')[0]
    row.update(pass_id='incomplete-retry', target_sha256='wrong')
    from gray_cold_diffusion.io import append_csv
    append_csv(t.output/'fixed_color_per_image.csv', row)
    report.read_fixed_records(t.output)
    (t.output/'fixed_color_per_image.csv').write_text(original.replace('UIEB__a.png', 'UIEB__changed.png', 1))
    with pytest.raises(ValueError, match='Fixed subset'):
        report.read_fixed_records(t.output)


def test_monitored_checkpoint_migration_and_milestones(tmp_path):
    old = make_trainer(tmp_path, output='old')
    old.config['implementation']['mixed_sha256'] = PRE_MONITOR_MIXED_SHA256
    # Populate real Adam moments before writing a previous-version checkpoint.
    old.model(torch.ones(1, 3, 16, 16), None).square().mean().backward()
    old.optimizer.step()
    old.optimizer.zero_grad()
    old.step = 10000
    old.last_validation = {'delta_e76': 12.}
    old.save_checkpoint()
    assert (old.output/'checkpoints/step_010000.pt').exists()
    new = make_trainer(tmp_path, output='new')
    expected_rng = torch.load(old.output/'checkpoints/latest.pt', weights_only=False)['rng']['torch']
    new.load_checkpoint(old.output/'checkpoints/latest.pt')
    assert new.step == 10000 and new.best_delta_e == 12.
    assert new.config['implementation']['mixed_sha256'] == mixed_fingerprint()
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert torch.equal(new.optimizer.state[new.model.bias]['exp_avg'], old.optimizer.state[old.model.bias]['exp_avg'])
    manifest = json.loads((new.output/'run_manifest.json').read_text())
    assert manifest['config']['implementation']['mixed_sha256'] == mixed_fingerprint()
    for step in [20000, 30000, 40000, 50000]:
        new.step = step
        new.save_checkpoint()
        assert (new.output/f'checkpoints/step_{step:06d}.pt').exists()
    payload = torch.load(old.output/'checkpoints/latest.pt', weights_only=False)
    payload['config']['implementation']['mixed_sha256'] = 'unreviewed-source'
    path = tmp_path/'bad.pt'
    torch.save(payload, path)
    with pytest.raises(ValueError, match='implementation'):
        make_trainer(tmp_path, output='reject').load_checkpoint(path)
    payload['config']['implementation']['mixed_sha256'] = PRE_MONITOR_MIXED_SHA256
    payload['config']['implementation']['source_sha256']['engine.py'] = 'changed-training'
    torch.save(payload, path)
    with pytest.raises(ValueError, match='implementation'):
        make_trainer(tmp_path, output='reject_training_change').load_checkpoint(path)


def test_preview_failure_has_current_checkpoint_and_fixed_records(tmp_path, monkeypatch):
    t = make_trainer(tmp_path)
    t.step = 10000
    t.last_validation = {'delta_e76': 12.}
    calls = []
    def fail(self):
        calls.append(1)
        raise RuntimeError('CUDA error: unspecified launch failure')
    monkeypatch.setattr(OfficialTrainer, '_save_full_scene_preview', fail)
    with pytest.raises(RuntimeError, match='unspecified launch failure'):
        t._save_full_scene_preview()
    assert len(calls) == 1
    assert torch.load(t.output/'checkpoints/latest.pt', weights_only=False)['step'] == 10000
    assert len(csv_rows(t.output/'fixed_color_summary.csv')) == 12
    records = [json.loads(x) for x in (t.output/'debug_diagnostics.jsonl').read_text().splitlines()]
    assert records[-1]['phase'] == 'full_scene_preview_failed'
    monkeypatch.setenv('MIXED_FULL_SCENE_PREVIEWS', '0')
    t._save_full_scene_preview()
    assert len(calls) == 1
    assert 'full_scene_preview_disabled_by_user' in (t.output/'debug_diagnostics.jsonl').read_text()
