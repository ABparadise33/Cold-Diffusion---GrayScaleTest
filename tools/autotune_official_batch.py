"""Find a safe microbatch in child processes, then run train.py unchanged."""
import copy
import datetime
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import yaml

from gray_cold_diffusion.batch_probe import select_batch
from gray_cold_diffusion.factory import OFFICIAL_MODE
from gray_cold_diffusion.official_training import preflight_official_run

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from train import parse_args  # noqa: E402


def launch(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    if args.batch_size is not None or args.grad_accum is not None:
        raise ValueError('--auto-batch cannot be mixed with manual --batch-size/--grad-accum')
    if args.device != 'cuda':
        raise ValueError('--auto-batch requires --device cuda on the training GPU')
    config = yaml.safe_load(Path(args.config).read_text())
    if config['mode'] != OFFICIAL_MODE:
        raise ValueError('--auto-batch is only supported for the new official RGB baseline')
    if args.output_dir:
        config['output_dir'] = args.output_dir
    if args.max_steps is not None:
        config['training']['max_steps'] = args.max_steps
    if args.num_workers is not None:
        config['data']['num_workers'] = args.num_workers
    preflight_official_run(copy.deepcopy(config), args)
    for directory in (args.train_dir, args.val_dir):
        if not Path(directory).is_dir():
            raise FileNotFoundError(directory)
    if args.resume:
        checkpoint = Path(config['output_dir']) / 'checkpoints/latest.pt' if args.resume == 'auto' else Path(args.resume)
        if not checkpoint.is_file():
            raise FileNotFoundError(f'resume checkpoint not found: {checkpoint}')
    effective = int(config['training']['batch_size']) * int(config['training']['grad_accum'])
    report_root = ROOT / 'outputs/batch_probes'
    report_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    report_dir = Path(tempfile.mkdtemp(prefix=f'official_{stamp}_', dir=report_root))
    config_path = report_dir / 'probe_config.yaml'
    config_path.write_text(yaml.safe_dump(config))
    print(f'Batch probe reports: {report_dir}', flush=True)
    print('Only OOM/headroom failures fall back; non-OOM errors stop. No training checkpoint is modified.', flush=True)
    report = select_batch(config_path, effective, report_dir)
    chosen = report['selected']
    command = [sys.executable, '-u', str(ROOT / 'train.py'), *argv,
               '--batch-size', str(chosen['batch_size']), '--grad-accum', str(chosen['grad_accum'])]
    report['training_command'] = command
    print(f'SELECTED batch={chosen["batch_size"]}, accumulation={chosen["grad_accum"]}, effective={effective}', flush=True)
    (report_dir / 'batch_probe.json').write_text(json.dumps(report, indent=2))
    # No automatic fallback/restart once real training begins: an unrelated
    # failure must not overwrite checkpoints or silently change an experiment.
    return subprocess.call(command)


if __name__ == '__main__':
    sys.exit(launch())
