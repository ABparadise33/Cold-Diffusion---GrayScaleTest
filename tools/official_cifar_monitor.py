"""Loss and fixed-input diagnostics; never stop or extend training implicitly."""
import contextlib
import csv
import hashlib
import json
import math
from pathlib import Path
import random
from types import SimpleNamespace
import uuid

import numpy as np
import torch


def append_json(path, record):
    with Path(path).open('a') as handle:
        handle.write(json.dumps(record, allow_nan=False) + '\n')


def append_csv(path, row):
    path = Path(path)
    header = not path.exists() or path.stat().st_size == 0
    with path.open('a', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if header:
            writer.writeheader()
        writer.writerow(row)


@contextlib.contextmanager
def isolated_rng(model):
    python_state, numpy_state = random.getstate(), np.random.get_state()
    modes = [(module, module.training) for module in model.modules()]
    try:
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            random.seed(42)
            np.random.seed(42)
            torch.manual_seed(42)
            model.eval()
            with torch.no_grad():
                yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        for module, mode in modes:
            module.training = mode


def choose_indices(targets):
    labels = np.asarray(targets)
    rng = np.random.default_rng(42)
    return sorted(int(i) for label in range(10)
                  for i in rng.choice(np.flatnonzero(labels == label), 100, replace=False))


def initialize(trainer):
    trainer._monitor_run_id = uuid.uuid4().hex
    if not hasattr(trainer, '_monitoring_state'):
        trainer._monitoring_state = {'loss_window': [], 'last_monitor_step': None}
    folder = Path(trainer.results_folder)
    folder.mkdir(parents=True, exist_ok=True)
    append_json(folder/'monitor_runs.jsonl', {'run_id': trainer._monitor_run_id, 'start_step': trainer.step,
                'loss_window_count': len(trainer._monitoring_state['loss_window']),
                'interval': 1000, 'fixed_subset': '100 CIFAR10 test images/class, seed42',
                'role': 'monitoring/validation; not untouched final test', 'automatic_early_stop': False})
    if (trainer._monitoring_state.get('last_monitor_step') != trainer.step
            or not (folder/'fixed_monitor/summary.jsonl').exists()):
        fixed_monitor(trainer)


def record_update(trainer, losses):
    if len(losses) != trainer.gradient_accumulate_every or not all(math.isfinite(v) for v in losses):
        raise ValueError('Missing or nonfinite microbatch losses')
    value = sum(losses) / len(losses)
    row = {'run_id': trainer._monitor_run_id, 'step': int(trainer.step), 'loss': value,
           'microbatches': len(losses), 'lr': trainer.opt.param_groups[0]['lr']}
    folder = Path(trainer.results_folder)
    append_csv(folder/'train_loss_steps.csv', row)
    append_json(folder/'train_loss_steps.jsonl', {**row, 'microbatch_losses': losses})
    window = trainer._monitoring_state['loss_window']
    window.append({'step': int(trainer.step), 'loss': value})
    if trainer.step % 1000 == 0:
        values = np.array([x['loss'] for x in window])
        stats = {'run_id': trainer._monitor_run_id, 'step': int(trainer.step),
                 'start_step': window[0]['step'], 'count': len(window),
                 'mean': float(values.mean()), 'std': float(values.std()),
                 'min': float(values.min()), 'max': float(values.max())}
        append_csv(folder/'train_loss_windows.csv', stats)
        print('TRAIN_LOSS_WINDOW:', json.dumps(stats))
        window.clear()
        fixed_monitor(trainer)


def fixed_monitor(trainer):
    from codex_cifar_resume import log_preview_color
    from torchvision import datasets, transforms
    from torchvision.utils import save_image

    folder = Path(trainer.results_folder)/'fixed_monitor'
    folder.mkdir(parents=True, exist_ok=True)
    with isolated_rng(trainer.ema_model):
        if not hasattr(trainer, '_fixed_monitor_cache'):
            # Official torchvision CIFAR dataset root, independent of training shuffle/augmentations.
            dataset = datasets.CIFAR10(root=trainer.ds.root, train=False, download=False,
                                       transform=transforms.ToTensor())
            indices = choose_indices(dataset.targets)
            images = torch.stack([dataset[i][0]*2-1 for i in indices])
            hashes = [hashlib.sha256(x.contiguous().numpy().tobytes()).hexdigest() for x in images]
            manifest = {'version': 1, 'seed': 42, 'split': 'CIFAR10 test used as monitoring validation',
                        'count': 1000, 'per_class': 100, 'indices': indices, 'target_tensor_sha256': hashes,
                        'metric_schema': 2, 'batch_size': 32, 'sampler_steps': 20, 'weights': 'ema'}
            path = folder/'manifest.json'
            if path.exists() and json.loads(path.read_text()) != manifest:
                raise ValueError('Fixed monitor data/config changed; use a separate experiment folder')
            path.write_text(json.dumps(manifest, indent=2))
            trainer._fixed_monitor_cache = (indices, images)
        indices, images = trainer._fixed_monitor_cache
        device = next(trainer.ema_model.parameters()).device
        totals = {}
        proxy = SimpleNamespace(step=trainer.step, save_and_sample_every=1000, results_folder=folder,
                                metric_split='test_monitor', metric_run_id=trainer._monitor_run_id)
        with (folder/'sampler.log').open('a') as sampler_log:
            for start in range(0, len(images), 32):
                batch = images[start:start+32].to(device)
                with contextlib.redirect_stdout(sampler_log):
                    samples = trainer.ema_model.sample(batch_size=len(batch), img=batch, t=20)
                samples['og'] = batch
                proxy.metric_indices = indices[start:start+len(batch)]
                with contextlib.redirect_stdout(sampler_log):
                    event = log_preview_color(proxy, samples)
                for phase, metrics in event['phases'].items():
                    totals.setdefault(phase, {})
                    for key, value in metrics['mean'].items():
                        totals[phase][key] = totals[phase].get(key, 0.) + value*len(batch)
                if start == 0:
                    preview = folder/f'step_{trainer.step}'
                    preview.mkdir(exist_ok=True)
                    for phase, tensor in samples.items():
                        save_image(((tensor+1)*.5).clamp(0, 1), preview/f'{phase}.png', nrow=6)
        means = {p: {k: v/len(images) for k, v in m.items()} for p, m in totals.items()}
        for phase, metrics in means.items():
            metrics['delta_e76_gain_vs_gray'] = means['xt']['delta_e76']-metrics['delta_e76']
            metrics['rgb_mae_gain_vs_gray'] = means['xt']['rgb_mae']-metrics['rgb_mae']
            append_csv(folder/'summary.csv', {'run_id': trainer._monitor_run_id, 'step': trainer.step,
                       'phase': phase, 'count': len(images), **metrics})
        append_json(folder/'summary.jsonl', {'run_id': trainer._monitor_run_id, 'step': trainer.step,
                    'count': len(images), 'phases': means})
        print('FIXED_MONITOR:', json.dumps({'step': trainer.step, 'recon': means['recon']}))
        trainer._monitoring_state['last_monitor_step'] = int(trainer.step)
