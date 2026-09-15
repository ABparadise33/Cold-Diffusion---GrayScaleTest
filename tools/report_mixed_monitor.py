"""Build mixed-training curves and matched milestone image comparisons from saved logs."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

METRICS = ('rgb_mae', 'rgb_rmse', 'delta_e76', 'ab_error', 'lab_chroma', 'chroma_ratio')
PHASES = ('gray', 'direct_ema', 'sample_ema')


def read_csv(path):
    with Path(path).open(newline='') as handle:
        return list(csv.DictReader(handle))


def read_fixed_records(run):
    # A completed summary selects its matching pass. Partial/replayed writes must
    # never combine outputs produced by different checkpoint executions.
    summaries = {}
    for row in read_csv(run/'fixed_color_summary.csv'):
        key = (int(row['step']), row['split'], row['domain'], row['phase'])
        summaries[key] = row
    per_image = {}
    for row in read_csv(run/'fixed_color_per_image.csv'):
        key = (int(row['step']), row['split'], row['domain'], row['phase'])
        if key in summaries and row['pass_id'] == summaries[key]['pass_id']:
            per_image.setdefault(key, {})[row['image']] = row
    identities = {}
    for key, summary in summaries.items():
        values = per_image.get(key, {})
        if len(values) != int(summary['count']):
            raise ValueError(f'Incomplete fixed-image records: {key}')
        identity = sorted((name, row['target_sha256']) for name, row in values.items())
        group = key[1:]
        if group in identities and identities[group] != identity:
            raise ValueError(f'Fixed subset/target hashes changed for {group}; compare a consistent run segment')
        identities[group] = identity
        for metric in METRICS:
            if not all(math.isfinite(float(row[metric])) for row in values.values()):
                raise ValueError(f'Nonfinite diagnostic metric: {key}, {metric}')
            mean = sum(float(row[metric]) for row in values.values())/len(values)
            if abs(mean-float(summary[metric])) > 1e-7:
                raise ValueError(f'Metric summary mismatch: {key}, {metric}')
    return summaries, per_image


def build_report(run, output=None, stages=(10000, 20000, 30000, 40000, 50000)):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    from PIL import Image

    run = Path(run).resolve()
    output = Path(output).resolve() if output else run/'monitor_report'
    summaries, per_image = read_fixed_records(run)
    if not summaries:
        raise ValueError('No completed fixed monitoring records')
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.grid': True, 'grid.alpha': .15, 'font.size': 10})
    colors = {'gray': '#888888', 'direct_ema': '#da8726', 'sample_ema': '#167b99'}
    written = []
    for split in sorted({key[1] for key in summaries}):
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout='constrained')
        for row_index, domain in enumerate(('UIEB', 'DIV2K')):
            for ax, metric, label in zip(axes[row_index], ('rgb_mae', 'delta_e76', 'chroma_ratio'),
                                         ('RGB MAE (lower)', 'Lab DE76 (lower)', 'Mean per-image chroma ratio')):
                for phase in PHASES:
                    points = sorted((key[0], float(value[metric])) for key, value in summaries.items()
                                    if key[1:] == (split, domain, phase))
                    if points:
                        ax.plot([s/1000 for s, _ in points], [v for _, v in points],
                                '.-', label=phase, color=colors[phase], markersize=4)
                ax.set(title=f'{domain} / {label}', xlabel='Completed updates (k)')
                ax.xaxis.set_major_locator(MaxNLocator(5))
                ax.legend(fontsize=8)
        fig.suptitle(f'Mixed fixed center-crop diagnostics | {split} | subsets, not full-domain scores', fontsize=14)
        path = output/f'fixed_{split}_trends.png'
        fig.savefig(path, dpi=160)
        plt.close(fig)
        written.append(path.name)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout='constrained')
    if (run/'metrics.csv').exists():
        losses = {int(r['step']): float(r['train_l1']) for r in read_csv(run/'metrics.csv') if r.get('train_l1')}
        points = sorted(losses.items())
        axes[0].plot([s/1000 for s, _ in points], [v for _, v in points], color='#6346a0')
    axes[0].set(title='Training L1 (logged window mean)', xlabel='Completed updates (k)')
    if (run/'full_gray_metrics.csv').exists():
        vals = {int(r['step']): float(r['delta_e76']) for r in read_csv(run/'full_gray_metrics.csv')}
        points = sorted(vals.items())
        axes[1].plot([s/1000 for s, _ in points], [v for _, v in points], '.-', color='#167b99')
    axes[1].set(title='Complete combined validation center crops: DE76', xlabel='Completed updates (k)')
    for ax in axes:
        ax.xaxis.set_major_locator(MaxNLocator(5))
    fig.savefig(output/'training_validation.png', dpi=160)
    plt.close(fig)
    written.append('training_validation.png')
    available = sorted({key[0] for key in summaries})
    selected = sorted(set(stages) & set(available))
    comparison_dir = output/'image_comparisons'
    comparison_dir.mkdir(exist_ok=True)
    images = sorted({(key[1], key[2], name) for key, rows in per_image.items()
                     if key[3] == 'sample_ema' for name in rows})
    for split, domain, name in images:
        pairs = []
        for step in selected:
            key = (step, split, domain, 'sample_ema')
            if name not in per_image.get(key, {}):
                continue
            path = run/'debug_previews'/f'step_{step:06d}'/split/f'{Path(name).stem}.png'
            if not path.exists():
                raise FileNotFoundError(f'Missing comparison preview: {path}')
            expected = per_image[key][name]['preview_sha256']
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f'Preview differs from completed diagnostic pass: {path}')
            pairs.append((step, path))
        if not pairs:
            continue
        fig, axes = plt.subplots(len(pairs), 1, figsize=(12, 2.7*len(pairs)), squeeze=False, layout='constrained')
        for ax, (step, path) in zip(axes[:, 0], pairs):
            with Image.open(path) as im:
                ax.imshow(im, interpolation='nearest')
            ax.axis('off')
            de = float(per_image[(step, split, domain, 'sample_ema')][name]['delta_e76'])
            ax.set_title(f'{step/1000:g}k | recon DE76={de:.4f}')
        fig.suptitle(f'{split} / {name}\nTarget | Gray | Direct EMA | Sample EMA (unchanged saved strips)')
        dest = comparison_dir/f'{split}_{Path(name).stem}.png'
        fig.savefig(dest, dpi=130)
        plt.close(fig)
        written.append(str(dest.relative_to(output)))
    manifest = {'run': str(run), 'available_steps': available, 'requested_stages': list(stages),
                'compared_stages': selected, 'missing_stages': sorted(set(stages)-set(selected)),
                'scope': 'fixed diagnostic subsets; full combined validation and training loss plotted separately',
                'identity_check': 'identical image names and target tensor hashes across recorded steps',
                'replay_rule': 'latest completed pass per step/split/domain/phase',
                'summary_rows': len(summaries), 'files': written,
                'input_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                [run/'fixed_color_summary.csv', run/'fixed_color_per_image.csv']}}
    (output/'report.json').write_text(json.dumps(manifest, indent=2))
    print(f'Mixed monitoring report: {output}; available milestone comparisons: {selected}')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--steps', nargs='+', type=int, default=[10000, 20000, 30000, 40000, 50000])
    args = parser.parse_args()
    build_report(args.run_dir, args.output_dir, args.steps)
