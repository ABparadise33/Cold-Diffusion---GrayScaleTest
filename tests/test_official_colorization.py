import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from PIL import Image
import pytest
import torch
from torch import nn
import yaml

from gray_cold_diffusion.bridge import GrayBridge
from gray_cold_diffusion.factory import build_model_and_bridge
from gray_cold_diffusion.official_colorization import OfficialColorizer, RGBDecolorization, channel_gray
from gray_cold_diffusion.official_training import OfficialTrainer

ROOT = Path(__file__).resolve().parents[1]


def test_pinned_upstream_network_body_is_unchanged():
    body = (ROOT / 'src/gray_cold_diffusion/official_convnext.py').read_text().split('import math', 1)[1].strip()
    assert hashlib.sha256(body.encode()).hexdigest() == '41701c38bf7058dcfee4dfcd0f5c730089439b7752c4b9f16552614d03f97a6f'


def upstream_cumulative(x, s, steps):
    # Independent 1x1 sequential-kernel implementation of upstream Linear.
    start = 1.0
    for i in range(s):
        keep = 0 if i == steps - 1 else 1 - (1 / steps) / start
        start *= keep
        weight = keep * torch.eye(3) + (1 - keep) * torch.ones(3, 3) / 3
        x = torch.nn.functional.conv2d(x, weight[..., None, None])
    return x


def test_forward_matches_upstream_all_states_ignores_raw_anchor():
    torch.manual_seed(42)
    x = torch.rand(2, 3, 8, 8) * 2 - 1
    bridge = RGBDecolorization(20)
    for s in range(21):
        t = torch.full((2,), s)
        out = bridge.degrade(x, torch.randn_like(x), t)
        assert torch.allclose(out, upstream_cumulative(x, s, 20), atol=5e-7)
    gray = bridge.degrade(x, None, torch.full((2,), 20))
    assert torch.equal(gray[:, 0], gray[:, 1])
    assert torch.equal(gray[:, 1], gray[:, 2])


class ConstantModel(nn.Module):
    def __init__(self, pred):
        super().__init__()
        self.pred = pred
        self.calls = []

    def forward(self, x, t):
        self.calls.append(t.tolist())
        return self.pred.expand_as(x)


@pytest.mark.parametrize('sampler,updates', [('paper_algorithm2', 20), ('official_code', 19)])
def test_reverse_mean_invariant_no_clamp_and_exact_endpoint(sampler, updates):
    # Deliberately out-of-range prediction catches accidental state clipping.
    pred = torch.tensor([2.4, -.6, -.6]).view(1, 3, 1, 1)
    anchor = torch.zeros_like(pred)
    model = ConstantModel(pred)
    result, states = RGBDecolorization(20, sampler).sample(model, anchor, True)
    expected = anchor + updates / 20 * (pred - channel_gray(pred))
    assert torch.allclose(result, expected, atol=1e-6)
    assert result.max() > 1
    assert len(states) == 21
    assert model.calls == [[s] for s in range(20, 0, -1)]
    assert all(torch.allclose(x.mean(1), anchor.mean(1), atol=1e-6) for x in states)
    assert sum(not torch.equal(a, b) for a, b in zip(states, states[1:])) == updates
    legacy = GrayBridge(20).sample(ConstantModel(pred), anchor)
    assert not torch.allclose(legacy, result)


def test_literal_upstream_sampler_matches_sequential_kernel_reference():
    torch.manual_seed(2)
    x = channel_gray(torch.rand(1, 3, 8, 8))
    pred = torch.rand_like(x)
    expected = x.clone()
    for t in range(19, -1, -1):
        if t:
            expected = expected - upstream_cumulative(pred, t, 20) + upstream_cumulative(pred, t - 1, 20)
    actual = RGBDecolorization(20, 'official_code').sample(ConstantModel(pred), x)
    assert torch.allclose(expected, actual, atol=2e-6)


def test_official_rejects_partial_color_and_legacy_model_selection():
    with pytest.raises(ValueError, match='FULL gray'):
        RGBDecolorization().sample(nn.Identity(), torch.rand(1, 3, 4, 4))
    config = yaml.safe_load((ROOT / 'configs/div2k_official_rgb_sat1_50k.yaml').read_text())
    bad = copy.deepcopy(config)
    bad['data']['saturation_factor'] = 1.25
    with pytest.raises(ValueError, match='saturation_factor'):
        build_model_and_bridge(bad)
    bad = copy.deepcopy(config)
    bad['model']['architecture'] = 'legacy'
    with pytest.raises(ValueError, match='upstream_convnext'):
        build_model_and_bridge(bad)


def test_network_time_mapping_padding_and_gradients():
    torch.set_num_threads(1)
    model = OfficialColorizer(dim=8, dim_mults=(1, 2), steps=20)
    x = torch.rand(1, 3, 16, 24)
    t = torch.tensor([20])
    assert torch.equal(model(x, t), model.network(x, t - 1))
    out = model(torch.rand(1, 3, 17, 25), t)
    assert out.shape == (1, 3, 17, 25)
    out.mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_upstream_ema_warmup_and_counter_phase():
    trainer = OfficialTrainer.__new__(OfficialTrainer)
    trainer.model = nn.Linear(1, 1, bias=False)
    trainer.ema = nn.Linear(1, 1, bias=False)
    trainer.config = {'training': {'ema_decay': .995, 'ema_start_step': 2000, 'ema_update_every': 10}}
    with torch.no_grad():
        trainer.model.weight.fill_(1)
        trainer.ema.weight.fill_(0)
    trainer.step = 1
    trainer.step_ema()
    assert trainer.ema.weight.item() == 1
    with torch.no_grad():
        trainer.model.weight.fill_(2)
    trainer.step = 2
    trainer.step_ema()
    assert trainer.ema.weight.item() == 1
    trainer.step = 2001
    trainer.step_ema()
    assert trainer.ema.weight.item() == pytest.approx(1.005)


def run_cli(args, tmp_path, ok=True):
    env = os.environ.copy()
    env.update(PYTHONPATH=str(ROOT / 'src'), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               MPLCONFIGDIR=str(tmp_path / 'mpl'))
    result = subprocess.run([sys.executable, *args], cwd=ROOT, env=env, text=True,
                            capture_output=True, timeout=120)
    if ok:
        assert result.returncode == 0, result.stdout + result.stderr
    else:
        assert result.returncode != 0
    return result


def test_fresh_training_resume_full_scene_and_both_evaluations(tmp_path):
    train_dir, val_dir = tmp_path / 'train', tmp_path / 'val'
    train_dir.mkdir()
    val_dir.mkdir()
    for index in range(2):
        for folder in (train_dir, val_dir):
            Image.new('RGB', (25, 17), (40 + 40 * index, 120, 200)).save(folder / f'{index:04d}.png')
    config = yaml.safe_load((ROOT / 'configs/div2k_official_rgb_sat1_50k.yaml').read_text())
    config['model'].update(dim=8, dim_mults=[1, 2])
    config['diffusion']['steps'] = 2
    config['data'].update(image_size=16, num_workers=0, validation_preview_name='0000')
    config['training'].update(max_steps=2, batch_size=1, grad_accum=1, log_every=1,
                             validate_every=1, save_every=1, preview_tile_size=16, preview_tile_overlap=4,
                             min_free_disk_gb=0)
    output = tmp_path / 'run'
    config['output_dir'] = str(output)
    cfg_path = tmp_path / 'config.yaml'
    cfg_path.write_text(yaml.safe_dump(config))
    command = ['train.py', '--config', str(cfg_path), '--train-dir', str(train_dir),
               '--val-dir', str(val_dir), '--device', 'cpu']
    run_cli(command, tmp_path)
    assert not (output / 'checkpoints/step_000001.pt').exists()
    checkpoint = output / 'checkpoints/step_000002.pt'
    assert checkpoint.exists()
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert payload['step'] == 2
    assert payload['config']['mode'] == 'official_rgb_colorization'
    assert payload['run_metadata']['data']['val']['count'] == 2
    assert (output / 'training_curves.png').is_file()
    assert (output / 'full_gray_metrics.csv').is_file()
    previews = output / 'previews/step_000002'
    assert json.loads((previews / 'preview.json').read_text())['actual_count'] == 2
    with Image.open(previews / 'predictions/0000.png') as image:
        assert image.size == (25, 17)
    run_cli(command, tmp_path, ok=False)  # never silently overwrite a run
    run_cli([*command, '--resume', '--max-steps', '3'], tmp_path)
    assert torch.load(output / 'checkpoints/latest.pt', weights_only=False)['step'] == 3
    assert (output / 'previews/step_000003/preview.json').is_file()
    # Simulate the exact reviewed source fingerprint of the pre-preview release.
    legacy_payload = copy.deepcopy(payload)
    old_hashes = legacy_payload['config']['implementation']['source_sha256']
    old_hashes['official_training.py'] = '789c81f26dfa83f4c739c4d2f12adb87b1769a0af593ac98983af4d6cf9c7b3c'
    del old_hashes['official_preview.py']
    legacy_payload['config']['training'].pop('preview_count', None)
    legacy_checkpoint = tmp_path / 'legacy.pt'
    torch.save(legacy_payload, legacy_checkpoint)
    migrated_output = tmp_path / 'migrated'
    migrated = run_cli([*command, '--resume', str(legacy_checkpoint), '--max-steps', '3',
                        '--output-dir', str(migrated_output)], tmp_path)
    assert 'verified known preview-only revision' in migrated.stdout
    assert (migrated_output / 'previews/step_000003/preview.json').is_file()
    resumed = torch.load(migrated_output / 'checkpoints/latest.pt', weights_only=False)
    assert resumed['step'] == 3
    assert 'preview_only_resume_migration' in resumed['run_metadata']
    split = tmp_path / 'split.json'
    split.write_text(json.dumps({'test': ['0000.png', '0001.png']}))
    for sampler in ('paper_algorithm2', 'official_code'):
        evaluation = tmp_path / sampler
        run_cli(['evaluate.py', '--checkpoint', str(checkpoint), '--raw-dir', str(val_dir),
                 '--reference-dir', str(val_dir), '--split-file', str(split), '--device', 'cpu',
                 '--original-size', '--batch-size', '1', '--tile-size', '16', '--tile-overlap', '4',
                 '--sampler', sampler, '--output-dir', str(evaluation)], tmp_path)
        metadata = json.loads((evaluation / '其餘/metrics.json').read_text())['evaluation']
        assert metadata['sampler'] == sampler
        assert metadata['num_images'] == 2
        assert metadata['checkpoint_step'] == 2
        assert len(list((evaluation / 'predictions').glob('*.png'))) == 2
        assert not (evaluation / 'references').exists()
        assert (evaluation / '其餘/training_curves.png').is_file()
        with Image.open(evaluation / 'predictions/0000.png') as image:
            assert image.size == (25, 17)
    # Same weights and same gray input give exactly identical Direct predictions.
    assert (tmp_path / 'paper_algorithm2/direct_predictions/0000.png').read_bytes() == (tmp_path / 'official_code/direct_predictions/0000.png').read_bytes()


def test_official_commands_are_fresh_factor1_and_fixed50k():
    env = {**os.environ, 'OFFICIAL_DRY_RUN': '1'}
    train = subprocess.check_output(['bash', 'scripts/train_official_div2k_4090.sh'], cwd=ROOT, env=env, text=True)
    assert '--resume' not in train and 'div2k_official_rgb_sat1_50k.yaml' in train
    evaluation = subprocess.check_output(['bash', 'scripts/evaluate_official_div2k_4090.sh'], cwd=ROOT, env=env, text=True)
    assert evaluation.count('DRY RUN:') == 2
    assert 'step_050000.pt' in evaluation and 'best.pt' not in evaluation
    manifest = json.loads((ROOT / 'splits/div2k_valid_all.json').read_text())
    assert len(manifest['test']) == 100
    assert '--split test' in evaluation
