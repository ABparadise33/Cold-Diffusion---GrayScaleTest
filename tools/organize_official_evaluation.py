"""Reorganize completed official evaluation exports without running inference."""
import argparse
import json
from pathlib import Path

REPORT_FILES = (
    'metrics.json', 'training_curves.png', 'training_summary.json',
    'extended_metrics.csv', 'extended_metrics.md', 'direct_metrics.csv', 'direct_metrics.md',
    'direct_vs_algorithm2.csv', 'direct_vs_algorithm2.md',
)


def plan_layout(root):
    root = Path(root).absolute()
    if any(p.is_symlink() for p in (root, *root.parents)):
        raise ValueError('evaluation path must not pass through symlinks')
    if (root / 'metrics.json').is_file() or (root / '其餘/metrics.json').is_file():
        runs = [root]
    else:
        runs = [root / name for name in ('paper_algorithm2', 'official_code')
                if (root / name).is_dir()]
    if not runs:
        raise ValueError(f'no completed official evaluation found: {root}')
    moves, removals, empty_dirs = [], [], []
    for run in runs:
        if run.is_symlink() or (run / '其餘').is_symlink():
            raise ValueError('evaluation directories must not be symlinks')
        metric = run / 'metrics.json'
        if not metric.exists():
            metric = run / '其餘/metrics.json'
        if metric.is_symlink():
            raise ValueError('metrics must not be a symlink')
        metadata = json.loads(metric.read_text())['evaluation']
        if metadata.get('mode') != 'official_rgb_colorization':
            raise ValueError(f'not an official evaluation: {run}')
        prediction_dir = run / 'predictions'
        if prediction_dir.is_symlink():
            raise ValueError('predictions must not be a symlink')
        predictions = list(prediction_dir.glob('*.png'))
        if any(p.is_symlink() or not p.is_file() for p in predictions):
            raise ValueError('invalid prediction entries')
        if len(predictions) != metadata['num_images']:
            raise ValueError(f'incomplete prediction export: {run}')
        for name in REPORT_FILES:
            source, destination = run / name, run / '其餘' / name
            if source.exists() or source.is_symlink():
                if not source.is_file() or source.is_symlink():
                    raise ValueError(f'unexpected report entry: {source}')
                if destination.exists() or destination.is_symlink():
                    raise FileExistsError(f'refusing to overwrite: {destination}')
                moves.append((source, destination))
        refs = run / 'references'
        if refs.exists() or refs.is_symlink():
            if not refs.is_dir() or refs.is_symlink():
                raise ValueError('reference export must be an ordinary directory')
            expected = {p.name for p in predictions}
            for path in refs.iterdir():
                if path.is_symlink() or not path.is_file() or path.name not in expected:
                    raise ValueError(f'unrecognized reference export; refusing deletion: {path}')
                removals.append(path)
            empty_dirs.append(refs)
    return moves, removals, empty_dirs


def organize(root, apply=False):
    # Validate the entire selected export before any changes; never recursively delete.
    moves, removals, empty_dirs = plan_layout(root)
    print(f'reports to move: {len(moves)}; redundant reference PNGs to remove: {len(removals)}')
    if not apply:
        print('Preview only. Add --apply to move reports and remove exported reference copies.')
        return
    for source, destination in moves:
        destination.parent.mkdir(exist_ok=True)
        source.rename(destination)
    for path in removals:
        path.unlink()
    for path in empty_dirs:
        path.rmdir()
    print('DONE: prediction/Direct/batch/trajectory files untouched; reports are in 其餘/.')
    print('Removed reference exports can be recreated from the source UIEB dataset; source data was not touched.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    organize(args.output_dir, args.apply)


if __name__ == '__main__':
    main()
