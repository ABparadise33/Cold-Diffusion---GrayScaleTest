import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

from PIL import Image
import pytest
import torch

from gray_cold_diffusion.factory import build_model_and_bridge

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('natural_diag', ROOT/'tools/diagnose_natural_fullgray.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_natural_diagnostic_uses_fixed_sources_both_weights_and_full_frames(tmp_path):
    torch.set_num_threads(1)
    train, val = tmp_path/'train', tmp_path/'val'
    train.mkdir()
    val.mkdir()
    for folder, offset in [(train, 0), (val, 80)]:
        for i in range(2):
            Image.new('RGB', (25, 17), (offset+i*10, 100, 180)).save(folder/f'{i:04d}.png')
    selected = module.select_images(train, val, 1, 42)
    assert selected == module.select_images(train, val, 1, 42)
    with pytest.raises(ValueError, match='overlap'):
        module.select_images(train, train, 1, 42)
    config = {'mode': 'official_rgb_colorization', 'model': {'architecture': 'upstream_convnext', 'dim': 8, 'dim_mults': [1, 2]},
              'diffusion': {'steps': 2, 'sampler': 'paper_algorithm2'}, 'data': {'image_size': 16}}
    model, _ = build_model_and_bridge(config)
    online = {k: v.clone() for k, v in model.state_dict().items()}
    ema = {k: v.clone() for k, v in online.items()}
    for key in ema:
        if ema[key].is_floating_point():
            ema[key] = ema[key] + .01
    checkpoint = tmp_path/'model.pt'
    torch.save({'config': config, 'model': online, 'ema': ema, 'step': 9}, checkpoint)
    output = tmp_path/'diagnostic'
    env = {**os.environ, 'PYTHONPATH': str(ROOT/'src'), 'OMP_NUM_THREADS': '1', 'MPLCONFIGDIR': str(tmp_path/'mpl')}
    result = subprocess.run([sys.executable, 'tools/diagnose_natural_fullgray.py', '--checkpoint', str(checkpoint),
                             '--train-dir', str(train), '--val-dir', str(val), '--count', '1', '--device', 'cpu',
                             '--output-dir', str(output)], cwd=ROOT, env=env, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout+result.stderr
    summary = json.loads((output/'summary.json').read_text())['rows']
    assert {(r['split'],r['weights']) for r in summary} == {('train','ema'),('train','model'),('val','ema'),('val','model')}
    assert all(r['checkpoint_step'] == 9 and r['fallback_images'] == [] for r in summary)
    for split in ('train','val'):
        name = selected[split][0].name
        for weights in ('model','ema'):
            with Image.open(output/split/weights/'predictions'/name) as im:
                assert im.size == (25,17)
        assert (output/split/'ema/direct_predictions'/name).read_bytes() != (output/split/'model/direct_predictions'/name).read_bytes()
    assert json.loads((output/'source_manifest.json').read_text())['status'] == 'complete'
