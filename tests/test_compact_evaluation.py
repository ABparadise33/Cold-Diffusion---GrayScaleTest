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


def load_organizer():
    spec = importlib.util.spec_from_file_location('organizer', ROOT / 'tools/organize_official_evaluation.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_compact_export_preserves_predictions_and_scores(tmp_path):
    torch.set_num_threads(1)
    raw_dir, gt_dir = tmp_path / 'raw', tmp_path / 'gt'
    raw_dir.mkdir()
    gt_dir.mkdir()
    Image.new('RGB', (25, 17), (20, 120, 170)).save(raw_dir / 'raw.png')
    Image.new('RGB', (25, 17), (80, 100, 130)).save(gt_dir / 'target.png')
    manifest = tmp_path / 'split.json'
    manifest.write_text(json.dumps({'test': [{'raw': 'raw.png', 'reference': 'target.png'}]}))
    config = {'mode': 'official_rgb_colorization',
              'model': {'architecture': 'upstream_convnext', 'dim': 8, 'dim_mults': [1, 2]},
              'diffusion': {'steps': 2, 'sampler': 'paper_algorithm2'},
              'data': {'saturation_factor': 1, 'image_size': 16}}
    model, _ = build_model_and_bridge(config)
    checkpoint = tmp_path / 'weights.pt'
    torch.save({'config': config, 'ema': model.state_dict(), 'step': 50000}, checkpoint)
    env = {**os.environ, 'PYTHONPATH': str(ROOT / 'src'), 'OMP_NUM_THREADS': '1',
           'MPLCONFIGDIR': str(tmp_path / 'mpl')}
    for layout in ('legacy', 'compact'):
        command = [sys.executable, 'evaluate.py', '--checkpoint', str(checkpoint),
                   '--raw-dir', str(raw_dir), '--reference-dir', str(gt_dir),
                   '--split-file', str(manifest), '--original-size', '--batch-size', '1',
                   '--device', 'cpu', '--output-dir', str(tmp_path / layout), '--output-layout', layout]
        result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stdout + result.stderr
    legacy, compact = tmp_path / 'legacy', tmp_path / 'compact'
    for folder, name in [('predictions', 'raw.png'), ('direct_predictions', 'raw.png'),
                         ('batches', 'batch_000.png'), ('trajectories', 'trajectory_000.png')]:
        assert (legacy / folder / name).read_bytes() == (compact / folder / name).read_bytes()
    left, right = json.loads((legacy / 'metrics.json').read_text()), json.loads((compact / '其餘/metrics.json').read_text())
    for metric in ('psnr', 'ssim', 'delta_e76', 'monotonic', 'direct'):
        assert left[metric] == right[metric]
    assert not (compact / 'references').exists()
    assert set(p.name for p in compact.iterdir()) == {'predictions', 'direct_predictions', 'batches', 'trajectories', '其餘'}
    with Image.open(compact / 'predictions/raw.png') as image:
        assert image.size == (25, 17)


def fixture_export(root):
    root.mkdir(parents=True)
    for folder in ('predictions', 'direct_predictions', 'batches', 'trajectories', 'references'):
        (root / folder).mkdir()
        Image.new('RGB', (8, 5)).save(root / folder / 'sample.png')
    (root / 'metrics.json').write_text(json.dumps({'evaluation': {'mode': 'official_rgb_colorization', 'num_images': 1}}))
    (root / 'training_summary.json').write_text('{}')
    Image.new('RGB', (10, 10)).save(root / 'training_curves.png')


def test_existing_results_reorganize_without_inference_or_changing_images(tmp_path):
    tool = load_organizer()
    root = tmp_path / 'export'
    for method in ('paper_algorithm2', 'official_code'):
        fixture_export(root / method)
    original = {p: p.read_bytes() for p in root.rglob('*') if p.is_file() and p.parent.name != 'references'}
    tool.organize(root)
    assert (root / 'paper_algorithm2/references/sample.png').exists()
    tool.organize(root, apply=True)
    for path, content in original.items():
        new_path = path.parent / '其餘' / path.name if path.name in tool.REPORT_FILES else path
        assert new_path.read_bytes() == content
    assert not list(root.rglob('references'))
    tool.organize(root, apply=True)  # idempotent


def test_organizer_rejects_unknown_references_and_conflicts_before_any_move(tmp_path):
    tool = load_organizer()
    root = tmp_path / 'export'
    fixture_export(root)
    unknown = root / 'references/user-note.txt'
    unknown.write_text('keep me')
    with pytest.raises(ValueError, match='unrecognized'):
        tool.organize(root, apply=True)
    assert (root / 'metrics.json').exists() and unknown.exists()
    (root / '其餘').mkdir()
    (root / '其餘/metrics.json').write_text('existing')
    with pytest.raises(FileExistsError, match='overwrite'):
        tool.organize(root, apply=True)
    assert (root / 'predictions/sample.png').exists()


def test_uieb_launcher_selects_underwater_manifest_and_compact_layout():
    env = {**os.environ, 'OFFICIAL_DRY_RUN': '1'}
    result = subprocess.run(['bash', 'scripts/evaluate_official_uieb_4090.sh'],
                            cwd=ROOT, env=env, text=True, capture_output=True, check=True)
    assert result.stdout.count('DRY RUN:') == 2
    assert result.stdout.count('--output-layout compact') == 2
    assert '--raw-dir data/UIEB/raw-890' in result.stdout
    assert 'splits/uieb_seed42.json' in result.stdout
    assert 'step_050000.pt' in result.stdout
    assert 'DIV2K_valid_HR' not in result.stdout
