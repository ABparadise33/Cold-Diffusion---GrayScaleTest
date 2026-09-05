"""Read-only ZIP/PNG audit and an analytic invertibility control; no model inference."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image


def rgb(source):
    with Image.open(source) as image:
        return np.asarray(image.convert('RGB'), dtype=np.float32) / 255


def interval(values):
    values = np.asarray(values)
    generator = np.random.default_rng(42)
    samples = values[generator.integers(0, len(values), (5000, len(values)))].mean(1)
    return np.quantile(samples, [.025, .975]).tolist()


def audit(archive, raw_dir, split_file):
    archive, raw_dir, split_file = Path(archive), Path(raw_dir), Path(split_file)
    split = json.loads(split_file.read_text())['test']
    paths = {Path(item if isinstance(item, str) else item['raw']).stem:
             raw_dir / (item if isinstance(item, str) else item['raw']) for item in split}
    assert len(paths) == len(split), 'duplicate test stems'
    summary = {'archive': str(archive.resolve()), 'raw_dir': str(raw_dir.resolve()),
               'split_sha256': hashlib.sha256(split_file.read_bytes()).hexdigest(), 'num_images': len(paths),
               'measurement': 'reported unquantized tensor scores; independent original-size PNG pixel differences',
               'bootstrap': '5000 paired image resamples, seed42; not training-seed uncertainty', 'conditions': {}}
    digest = hashlib.sha256()
    with archive.open('rb') as handle:
        for chunk in iter(lambda: handle.read(4 * 1024**2), b''):
            digest.update(chunk)
    summary['archive_sha256'] = digest.hexdigest()
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        assert len(names) == len(set(names)), 'ambiguous duplicate ZIP names'
        checkpoints, raw_scores = set(), []
        for percent, start in [(5, 19), (25, 15)]:
            prefix = f'paper_algorithm2/retain_{percent}pct/'
            report_name, = [name for name in names if name.startswith(prefix) and name.endswith('/metrics.json')]
            rows_name, = [name for name in names if name.startswith(prefix) and name.endswith('/per_image_core.json')]
            metrics = json.loads(z.read(report_name))
            rows = json.loads(z.read(rows_name))
            by_name = {row['image']: row for row in rows}
            assert len(rows) == len(paths) and set(by_name) == set(paths)
            meta = metrics['evaluation']
            assert meta['checkpoint_step'] == 50000 and meta['start_step'] == start
            assert meta['retained_raw_color_percent'] == percent and meta['sampler'] == 'paper_algorithm2'
            assert meta['split_sha256'] == summary['split_sha256'] and meta['num_images'] == len(paths)
            assert meta['spatial_mode'] == 'original_size' and (meta['tile_size'], meta['tile_overlap']) == (256, 32)
            prediction_names = [name for name in names if name.startswith(prefix + 'predictions/') and name.endswith('.png')]
            assert {Path(name).stem for name in prediction_names} == set(paths)
            checkpoints.add(meta['checkpoint_sha256'])
            raw_scores.append([by_name[name]['raw'] for name in sorted(paths)])
            for key in ('psnr', 'ssim', 'delta_e76'):
                assert abs(np.mean([row['prediction'][key] for row in rows]) - metrics[key]) < 1e-5
            differences, large_pixels = [], []
            for name, path in paths.items():
                source = rgb(path)
                prediction = rgb(io.BytesIO(z.read(prefix + f'predictions/{name}.png')))
                assert source.shape == prediction.shape
                difference = np.abs(source - prediction) * 255
                differences.append({'image': name, 'mae_255': float(difference.mean())})
                large_pixels.append(float((difference.max(2) > 5).mean()))
            gain = [row['raw']['delta_e76'] - row['prediction']['delta_e76'] for row in rows]
            summary['conditions'][str(percent)] = {
                'scores': {key: metrics[key] for key in ('psnr', 'ssim', 'delta_e76', 'monotonic')},
                'baselines': metrics['baselines'], 'reported_output_vs_raw': metrics['output_vs_raw'],
                'png_mae_to_raw_255': float(np.mean([row['mae_255'] for row in differences])),
                'mean_pixel_fraction_any_channel_difference_gt5': float(np.mean(large_pixels)),
                'delta_e_gain_over_raw': float(np.mean(gain)), 'delta_e_gain_ci95': interval(gain),
                'images_better_than_raw_delta_e': int(np.sum(np.asarray(gain) > 0)),
                'largest_raw_differences': sorted(differences, key=lambda row: -row['mae_255'])[:5],
                'per_image': rows,
            }
        assert len(checkpoints) == 1 and raw_scores[0] == raw_scores[1]
        summary['reported_checkpoint_sha256'] = checkpoints.pop()
        summary['checkpoint_note'] = 'Matching reported hashes; actual checkpoint bytes are not in the ZIP.'
    # Not neural inference or a new timestep schedule. Subsample each raw scene
    # for an inexpensive numeric control of the known linear operator.
    toy = {str(p): {'float_mae': [], 'quantized_mae': []} for p in (25, 5, 1, .5)}
    for path in paths.values():
        source = rgb(path)[::8, ::8]
        gray = source.mean(2, keepdims=True)
        for percent in (25, 5, 1, .5):
            retention = percent / 100
            state = retention * source + (1 - retention) * gray
            average = state.mean(2, keepdims=True)
            restored = average + (state - average) / retention
            quantized = np.rint(state * 255) / 255
            q_average = quantized.mean(2, keepdims=True)
            q_restored = q_average + (quantized - q_average) / retention
            toy[str(percent)]['float_mae'].append(float(np.abs(restored - source).mean() * 255))
            toy[str(percent)]['quantized_mae'].append(float(np.abs(q_restored - source).mean() * 255))
    summary['analytic_control'] = {
        'method': 'mean(x_t)+(x_t-mean(x_t))/r; no model, no GT, no reconstruction clipping',
        'scope': 'all test raw images at spatial stride8, equal image weights, NumPy float32',
        'warning': 'Quantized input is a separate control, NOT the actual float32 model input.',
        'means_mae_255': {key: {name: float(np.mean(values)) for name, values in row.items()} for key, row in toy.items()},
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', required=True)
    parser.add_argument('--raw-dir', required=True)
    parser.add_argument('--split-file', default='splits/uieb_seed42.json')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f'refusing to overwrite analysis: {output}')
    summary = audit(args.archive, args.raw_dir, args.split_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    small = {**summary, 'conditions': {k: {field: value for field, value in row.items() if field != 'per_image'}
                                     for k, row in summary['conditions'].items()}}
    print(json.dumps(small, indent=2))


if __name__ == '__main__':
    main()
