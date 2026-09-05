import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn

from gray_cold_diffusion.factory import build_model_and_bridge
from gray_cold_diffusion.official_colorization import RGBDecolorization, channel_gray
from gray_cold_diffusion.official_partial import partial_raw_input, retained_color_start_step, sample_from_step

ROOT = Path(__file__).resolve().parents[1]


class Oracle(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.source = source
        self.calls = []

    def forward(self, x, t):
        self.calls.append(int(t[0]))
        return self.source


@pytest.mark.parametrize('percent,start', [(0, 20), (5, 19), (25, 15)])
@pytest.mark.parametrize('sampler', ['paper_algorithm2', 'official_code'])
def test_raw_only_partial_state_exact_time_labels_and_oracle(percent, start, sampler):
    torch.manual_seed(4)
    source = torch.rand(1, 3, 7, 9) * 2 - 1
    gray = channel_gray(source)
    bridge = RGBDecolorization(20, sampler)
    assert retained_color_start_step(20, percent) == start
    state = partial_raw_input(bridge, source, start)
    expected = gray + (percent / 100) * (source - gray)
    assert torch.allclose(state, expected, atol=2e-7)
    original = state.clone()
    model = Oracle(source)
    pred, trajectory = sample_from_step(bridge, model, state, start)
    assert model.calls == list(range(start, 0, -1))
    assert len(trajectory) == start + 1
    assert torch.equal(state, original)
    assert all(torch.allclose(x.mean(1), gray.mean(1), atol=1e-6) for x in trajectory)
    expected = source if sampler == 'paper_algorithm2' else source - (source - gray) / 20
    assert torch.allclose(pred, expected, atol=2e-6)
    if percent == 0:
        previous, old_trajectory = bridge.sample(Oracle(source), gray, return_trajectory=True)
        assert torch.equal(pred, previous)
        assert all(torch.equal(a, b) for a, b in zip(trajectory, old_trajectory))


@pytest.mark.parametrize('percent', [-5, 100, 101, float('nan'), float('inf'), 1, 0.05])
def test_reject_invalid_or_unrepresentable_percentage(percent):
    with pytest.raises(ValueError):
        retained_color_start_step(20, percent)


def test_partial_does_not_clip_internal_states_or_relax_full_gray_guard():
    bridge = RGBDecolorization(20)
    state = torch.tensor([.1, -.1, 0.]).view(1, 3, 1, 1)
    with pytest.raises(ValueError, match='FULL gray'):
        bridge.sample(Oracle(state), state)
    pred, _ = sample_from_step(bridge, Oracle(state * 100), state, 19)
    assert pred.max() > 1 and pred.min() < -1


def test_cli_zero_equivalence_partial_exports_baselines_and_no_gt_leak(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    raw_dir, ref_dir = tmp_path / 'raw', tmp_path / 'gt'
    raw_dir.mkdir()
    ref_dir.mkdir()
    rng = np.random.default_rng(2)
    Image.fromarray(rng.integers(0, 256, (17, 25, 3), dtype=np.uint8)).save(raw_dir / 'a.png')
    Image.new('RGB', (25, 17), (20, 80, 130)).save(ref_dir / 'target.png')
    split = tmp_path / 'split.json'
    split.write_text(json.dumps({'test': [{'raw': 'a.png', 'reference': 'target.png'}]}))
    config = {'mode': 'official_rgb_colorization',
              'model': {'architecture': 'upstream_convnext', 'dim': 8, 'dim_mults': [1, 2]},
              'diffusion': {'steps': 20, 'sampler': 'paper_algorithm2'},
              'data': {'saturation_factor': 1, 'image_size': 16}}
    model, _ = build_model_and_bridge(config)
    checkpoint = tmp_path / 'step_050000.pt'
    torch.save({'config': config, 'ema': model.state_dict(), 'step': 50000}, checkpoint)
    command = [sys.executable, 'evaluate.py', '--checkpoint', str(checkpoint), '--expected-checkpoint-step', '50000', '--include-direct',
               '--raw-dir', str(raw_dir), '--reference-dir', str(ref_dir), '--split-file', str(split),
               '--device', 'cpu', '--original-size', '--batch-size', '1', '--preview-count', '1']
    env = {**os.environ, 'PYTHONPATH': str(ROOT / 'src'), 'OMP_NUM_THREADS': '1',
           'MPLCONFIGDIR': str(tmp_path / 'mpl')}

    def run(label, percent=None, include_direct=True):
        output = tmp_path / label
        args = command + ['--output-dir', str(output)]
        if not include_direct:
            args.remove('--include-direct')
        if percent is not None:
            args += ['--retain-color-percent', str(percent)]
        result = subprocess.run(args, cwd=ROOT, env=env, capture_output=True, text=True, timeout=90)
        assert result.returncode == 0, result.stdout + result.stderr
        return output, json.loads((output / '其餘/metrics.json').read_text())

    original, baseline = run('full_gray')
    zero, zero_metrics = run('zero', 0)
    for directory, name in [('predictions', 'a.png'), ('direct_predictions', 'a.png'),
                            ('batches', 'batch_000.png'), ('trajectories', 'trajectory_000.png')]:
        assert (zero / directory / name).read_bytes() == (original / directory / name).read_bytes()
    for key in ('psnr', 'ssim', 'delta_e76', 'direct', 'monotonic'):
        assert baseline[key] == zero_metrics[key]
    for percent, start in [(5, 19), (25, 15)]:
        output, metrics = run(f'keep_{percent}', percent)
        assert metrics['evaluation']['start_step'] == start
        assert metrics['evaluation']['model_calls'] == start
        assert metrics['evaluation']['effective_reverse_updates'] == start
        assert metrics['evaluation']['state_timesteps'] == list(range(start, -1, -1))
        assert metrics['evaluation']['checkpoint_sha256'] == zero_metrics['evaluation']['checkpoint_sha256']
        assert metrics['baselines']['raw'] == zero_metrics['baselines']['raw']
        assert len(json.loads((output / '其餘/per_image_core.json').read_text())) == 1
        assert not (output / 'references').exists()
        with Image.open(output / 'predictions/a.png') as image:
            assert image.size == (25, 17)
        with Image.open(output / 'trajectories/trajectory_000.png') as image:
            assert image.size == (25 * (start + 1), 17 + 28)
    Image.new('RGB', (25, 17), (250, 200, 0)).save(ref_dir / 'target.png')
    changed, changed_metrics = run('changed_gt', 25)
    assert changed_metrics['delta_e76'] != metrics['delta_e76']
    for directory in ('predictions', 'direct_predictions'):
        assert (changed / directory / 'a.png').read_bytes() == (output / directory / 'a.png').read_bytes()
    no_direct, no_direct_metrics = run('no_direct', 25, include_direct=False)
    assert (no_direct / 'predictions/a.png').read_bytes() == (changed / 'predictions/a.png').read_bytes()
    assert (no_direct / 'trajectories/trajectory_000.png').read_bytes() == (changed / 'trajectories/trajectory_000.png').read_bytes()
    assert no_direct_metrics['evaluation']['direct_evaluated'] is False
    assert 'direct' not in no_direct_metrics
    assert 'direct_mae_255' not in no_direct_metrics['output_vs_raw']
    assert 'direct' not in json.loads((no_direct / '其餘/per_image_core.json').read_text())[0]
    assert not (no_direct / 'direct_predictions').exists()
    with Image.open(no_direct / 'batches/batch_000.png') as image:
        assert image.size == (25 * 4, 17 + 28)
    # Count real network forwards: only the 15 reverse steps, no extra Direct pass.
    # Also exercise optional IQA routing without downloading any learned metric.
    spec = importlib.util.spec_from_file_location('eval_under_test', ROOT / 'evaluate.py')
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    build = evaluator.build_model_and_bridge
    calls, metric_calls = [], []

    def counted_build(config):
        model, bridge = build(config)
        model.register_forward_hook(lambda _model, _inputs, _output: calls.append(1))
        return model, bridge

    def fake_extended(**kwargs):
        metric_calls.append(kwargs['prediction_dir'].name)
        assert kwargs['reference_loader']('a').size == (25, 17)
        return {'means': {}, 'num_images': 1}

    monkeypatch.setattr(evaluator, 'build_model_and_bridge', counted_build)
    monkeypatch.setattr(evaluator, 'create_pyiqa_metrics', lambda _device: {})
    monkeypatch.setattr(evaluator, 'evaluate_extended_metrics', fake_extended)
    args = command[1:] + ['--output-dir', str(tmp_path / 'counted'), '--retain-color-percent', '25', '--extended-metrics']
    args.remove('--include-direct')
    monkeypatch.setattr(sys, 'argv', args)
    evaluator.main()
    assert len(calls) == 15
    assert metric_calls == ['predictions']
    assert not list((tmp_path / 'counted').rglob('direct*'))
    wrong_step = subprocess.run(command + ['--expected-checkpoint-step', '15000', '--output-dir', str(tmp_path / 'bad')],
                                cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)
    assert wrong_step.returncode != 0 and 'expected 15000' in wrong_step.stderr
    assert not (tmp_path / 'bad').exists()


def test_launcher_runs_only_two_requested_uieb_conditions():
    env = {**os.environ, 'OFFICIAL_DRY_RUN': '1'}
    result = subprocess.run(['bash', 'scripts/evaluate_official_partial_uieb_4090.sh'],
                            cwd=ROOT, env=env, capture_output=True, text=True, check=True)
    lines = result.stdout.splitlines()
    assert len(lines) == 2
    for line, percent in zip(lines, (5, 25)):
        assert f'--retain-color-percent {percent}' in line
        assert f'retain_{percent}pct' in line
        assert '--sampler paper_algorithm2' in line
        assert '--raw-dir data/UIEB/raw-890' in line and 'splits/uieb_seed42.json' in line
        assert 'step_050000.pt' in line and '--output-layout compact' in line
        assert '--expected-checkpoint-step 50000' in line
    assert 'best.pt' not in result.stdout and 'train.py' not in result.stdout
