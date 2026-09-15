"""Fixed natural train/val scenes: compare online/EMA Direct and full sampling."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else _digest(handle)


def _digest(handle):
    result = hashlib.sha256()
    for chunk in iter(lambda: handle.read(4*1024*1024), b''):
        result.update(chunk)
    return result.hexdigest()


def select_images(train_dir, val_dir, count, seed):
    if count < 1:
        raise ValueError('count must be positive')
    selected = {}
    rng = random.Random(seed)
    pools = {name: sorted(Path(folder).glob('*.png')) for name, folder in [('train', train_dir), ('val', val_dir)]}
    if not all(pools.values()):
        raise ValueError('both DIV2K train and validation PNG directories are required')
    if {p.resolve() for p in pools['train']} & {p.resolve() for p in pools['val']}:
        raise ValueError('train and val pools overlap')
    for split, paths in pools.items():
        selected[split] = sorted(rng.sample(paths, min(count, len(paths))))
    if {digest(p) for p in selected['train']} & {digest(p) for p in selected['val']}:
        raise ValueError('selected train/val images share identical file content')
    return selected


def summarize(output):
    rows = []
    for split in ('train', 'val'):
        for weights in ('ema', 'model'):
            folder = output/split/weights/'其餘'
            diagnostics = json.loads((folder/'color_diagnostics.json').read_text())
            metrics = json.loads((folder/'metrics.json').read_text())
            row = {'split': split, 'weights': weights, 'count': len(diagnostics),
                   'checkpoint_step': metrics['evaluation']['checkpoint_step'],
                   'fallback_images': [x['image'] for x in diagnostics if x['inference']['method'] != 'full_image']}
            for kind in ('target', 'gray_input', 'direct', 'sample'):
                row[kind] = {key: sum(x[kind][key] for x in diagnostics)/len(diagnostics)
                             for key in ('chroma', 'chroma_ratio', 'delta_e76')}
            row['sample_improves_gray_count'] = sum(x['sample']['delta_e76'] < x['gray_input']['delta_e76'] for x in diagnostics)
            rows.append(row)
    (output/'summary.json').write_text(json.dumps({'scope': 'fixed diagnostic subsets, not whole-domain scores or object-level labels',
                                                 'rows': rows}, indent=2))
    print(json.dumps(rows, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='outputs/uieb_div2k_rgb_fullgray_pilot/checkpoints/best.pt')
    parser.add_argument('--train-dir', default='data/DIV2K/DIV2K_train_HR')
    parser.add_argument('--val-dir', default='data/DIV2K/DIV2K_valid_HR')
    parser.add_argument('--count', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--oom-tile-size', type=int, default=512)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    selected = select_images(args.train_dir, args.val_dir, args.count, args.seed)
    checkpoint_hash = digest(args.checkpoint)
    out = Path(args.output_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'use a new output directory: {out}')
    out.mkdir(parents=True, exist_ok=True)
    source = {'seed': args.seed, 'checkpoint': str(Path(args.checkpoint).resolve()),
              'checkpoint_sha256': checkpoint_hash, 'input': 'full gray made from the same natural image used as target',
              'selected': {key: [{'path': str(p.resolve()), 'sha256': digest(p)} for p in paths] for key, paths in selected.items()},
              'status': 'in_progress'}
    (out/'source_manifest.json').write_text(json.dumps(source, indent=2))
    evaluator = Path(__file__).resolve().parents[1]/'evaluate.py'
    for split, paths in selected.items():
        manifest = out/f'{split}_split.json'
        manifest.write_text(json.dumps({'test': [p.name for p in paths]}))
        for weights in ('ema', 'model'):
            if digest(args.checkpoint) != checkpoint_hash:
                raise RuntimeError('checkpoint changed during diagnostic')
            command = [sys.executable, '-u', str(evaluator), '--checkpoint', args.checkpoint,
                       '--weights', weights, '--raw-dir', str(paths[0].parent), '--reference-dir', str(paths[0].parent),
                       '--split-file', str(manifest), '--split', 'test', '--device', args.device,
                       '--original-size', '--batch-size', '1', '--include-direct', '--sampler', 'paper_algorithm2',
                       '--preview-count', str(len(paths)), '--preview-max-side', '512', '--output-layout', 'compact',
                       '--oom-tile-size', str(args.oom_tile_size), '--output-dir', str(out/split/weights)]
            subprocess.run(command, check=True)
    if digest(args.checkpoint) != checkpoint_hash:
        raise RuntimeError('checkpoint changed during diagnostic')
    summarize(out)
    source['status'] = 'complete'
    (out/'source_manifest.json').write_text(json.dumps(source, indent=2))


if __name__ == '__main__':
    main()
