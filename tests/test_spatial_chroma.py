import os
from pathlib import Path
import subprocess
import sys

from PIL import Image
import pytest
import torch
from torch import nn
import yaml

from gray_cold_diffusion.factory import SPATIAL_MODE, build_model_and_bridge
from gray_cold_diffusion.official_colorization import channel_gray
from gray_cold_diffusion.spatial_chroma import SpatialChromaMask


ROOT = Path(__file__).resolve().parents[1]


class Oracle(nn.Module):
    def __init__(self, clean):
        super().__init__()
        self.clean = clean
        self.calls = []

    def forward(self, x, t):
        self.calls.append(t.tolist())
        return self.clean


def test_forward_uses_nested_binary_color_pixels_and_true_gray_endpoint():
    clean = torch.tensor([.9, .2, -.5]).view(1, 3, 1, 1).expand(1, 3, 4, 5).clone()
    gray = channel_gray(clean)
    bridge = SpatialChromaMask(20)
    pixels = clean.shape[-2] * clean.shape[-1]
    scores = ((torch.arange(pixels) + .5) / pixels).view(1, 1, 4, 5)
    previous = torch.ones(1, 1, 4, 5, dtype=torch.bool)
    for t, expected in ((0, 20), (10, 10), (19, 1), (20, 0)):
        state = bridge.degrade(clean, None, torch.tensor([t]), scores)
        colored = (state != gray).any(1, keepdim=True)
        assert int(colored.sum()) == expected
        assert bool((colored <= previous).all())
        assert torch.equal(state, torch.where(colored, clean, gray))
        previous = colored


def test_removed_pixel_is_many_to_one_not_attenuated_chroma():
    # Both colors have channel mean 0.2, so deletion makes them identical.
    colors = torch.tensor([[.8, .0, -.2], [-.2, .8, .0]]).view(2, 3, 1, 1)
    bridge = SpatialChromaMask(20)
    removed = bridge.degrade(colors, None, torch.tensor([20, 20]), torch.ones(2, 1, 1, 1))
    assert torch.equal(removed[0], removed[1])
    assert torch.equal(removed, channel_gray(colors))


def test_algorithm2_oracle_recovers_exact_color_and_uses_one_mask_sequence():
    torch.manual_seed(7)
    clean = torch.rand(2, 3, 8, 9) * 2 - 1
    bridge = SpatialChromaMask(20, sampling_seed=91)
    anchor = channel_gray(clean)
    scores = bridge.sampling_scores(anchor)
    model = Oracle(clean)
    result, trajectory = bridge.sample(model, anchor, True, scores)
    assert torch.allclose(result, clean, atol=2e-6)
    assert len(trajectory) == 21
    assert model.calls == [[s, s] for s in range(20, 0, -1)]
    for index, state in enumerate(trajectory):
        t = torch.full((2,), 20 - index)
        assert torch.allclose(state, bridge.degrade(clean, None, t, scores), atol=2e-6)


def test_sampling_mask_is_reproducible_and_does_not_touch_global_rng():
    anchor = torch.zeros(2, 3, 9, 11)
    bridge = SpatialChromaMask(20, sampling_seed=42)
    before = torch.get_rng_state().clone()
    first = bridge.sampling_scores(anchor)
    second = bridge.sampling_scores(anchor)
    assert torch.equal(first, second)
    assert torch.equal(torch.get_rng_state(), before)
    assert not torch.equal(first[0], first[1])


def test_pilot_config_and_4090_launcher_are_separate_and_resumable():
    config = yaml.safe_load((ROOT / 'configs/div2k_spatial_chroma_sat1_t20_pilot.yaml').read_text())
    assert config['mode'] == SPATIAL_MODE
    assert config['data']['saturation_factor'] == 1
    assert config['diffusion'] == {'steps': 20, 'sampler': 'paper_algorithm2', 'sampling_seed': 42}
    assert config['training']['max_steps'] == 10000
    assert config['training']['preview_count'] == 5
    assert config['training']['preview_direct'] is False
    assert config['training']['validation_direct'] is False
    model, bridge = build_model_and_bridge({**config, 'model': {'architecture': 'upstream_convnext',
                                                                 'dim': 8, 'dim_mults': [1, 2]}})
    assert isinstance(bridge, SpatialChromaMask)
    assert sum(p.numel() for p in model.parameters()) > 0
    env = {**os.environ, 'SPATIAL_DRY_RUN': '1'}
    fresh = subprocess.check_output(['bash', 'scripts/train_spatial_chroma_div2k_4090.sh', '--auto-batch'],
                                    cwd=ROOT, env=env, text=True)
    assert 'div2k_spatial_chroma_sat1_t20_pilot.yaml' in fresh
    assert '--resume' not in fresh and '--batch-size' not in fresh
    resumed = subprocess.check_output(
        ['bash', 'scripts/train_spatial_chroma_div2k_4090.sh', '--resume', '--max-steps', '50000'],
        cwd=ROOT, env=env, text=True,
    )
    assert '--resume' in resumed and '--max-steps 50000' in resumed


def test_rejects_bad_mask_shape_and_partial_sampling_start():
    bridge = SpatialChromaMask(2)
    x = torch.rand(1, 3, 4, 4)
    with pytest.raises(ValueError, match='mask score shape'):
        bridge.degrade(x, None, torch.tensor([1]), torch.rand(1, 1, 3, 4))
    with pytest.raises(ValueError, match='full gray'):
        bridge.sample(Oracle(x), x)


def test_tiny_spatial_training_writes_resumable_checkpoint_and_no_direct_preview(tmp_path):
    train_dir, val_dir = tmp_path / 'train', tmp_path / 'val'
    train_dir.mkdir()
    val_dir.mkdir()
    for index in range(2):
        for folder in (train_dir, val_dir):
            Image.new('RGB', (25, 17), (50 + index * 30, 100, 180)).save(folder / f'{index}.png')
    config = yaml.safe_load((ROOT / 'configs/div2k_spatial_chroma_sat1_t20_pilot.yaml').read_text())
    config['model'].update(dim=8, dim_mults=[1, 2])
    config['diffusion']['steps'] = 2
    config['data'].update(image_size=16, num_workers=0)
    config['training'].update(max_steps=1, batch_size=1, grad_accum=1, log_every=1,
                              validate_every=1, save_every=1, preview_count=1,
                              preview_tile_size=16, preview_tile_overlap=4, min_free_disk_gb=0)
    output = tmp_path / 'output'
    config['output_dir'] = str(output)
    config_path = tmp_path / 'config.yaml'
    config_path.write_text(yaml.safe_dump(config))
    env = {**os.environ, 'PYTHONPATH': str(ROOT / 'src'), 'OMP_NUM_THREADS': '1',
           'MKL_NUM_THREADS': '1', 'MPLCONFIGDIR': str(tmp_path / 'mpl')}
    result = subprocess.run(
        [sys.executable, 'train.py', '--config', str(config_path), '--train-dir', str(train_dir),
         '--val-dir', str(val_dir), '--device', 'cpu'], cwd=ROOT, env=env, text=True,
        capture_output=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    checkpoint = torch.load(output / 'checkpoints/latest.pt', map_location='cpu', weights_only=False)
    assert checkpoint['step'] == 1 and checkpoint['config']['mode'] == SPATIAL_MODE
    preview = output / 'previews/step_000001'
    assert len(list((preview / 'predictions').glob('*.png'))) == 1
    assert len(list((preview / 'samples').glob('*.png'))) == 1
    assert len(list((preview / 'trajectories').glob('*.png'))) == 1
    assert not (preview / 'direct_predictions').exists()
    assert 'direct_predictions' not in checkpoint['run_metadata']['preview']['layout']
    assert 'direct_delta_e76' not in (output / 'full_gray_metrics.csv').read_text().splitlines()[0]
