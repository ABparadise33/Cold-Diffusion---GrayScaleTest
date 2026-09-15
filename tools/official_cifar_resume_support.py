"""Checkpoint support for the pinned official CIFAR trainer; no model/sampler edits."""
import json
import os
from pathlib import Path
import random

import numpy as np
import torch

VERSION = 1


def signature(trainer):
    model = trainer.model
    forward = model.forward_process
    return {
        'image_size': list(model.image_size), 'timesteps': model.num_timesteps,
        'loss_type': model.loss_type, 'train_routine': model.train_routine,
        'sampling_routine': model.sampling_routine, 'to_lab': model.to_lab,
        'forward_type': type(forward).__name__,
        'decolor_routine': forward.decolor_routine,
        'decolor_total_remove': forward.decolor_total_remove,
        'decolor_ema_factor': forward.decolor_ema_factor,
        'batch_size': trainer.batch_size, 'grad_accum': trainer.gradient_accumulate_every,
        'ema_decay': trainer.ema.beta, 'ema_every': trainer.update_ema_every,
        'ema_start': trainer.step_start_ema,
        'lr': [group['lr'] for group in trainer.opt.param_groups],
    }


def save_checkpoint(trainer, save_with_time_stamp=False):
    np_state = np.random.get_state()
    state = {
        'checkpoint_version': VERSION, 'step': trainer.step,
        'step_semantics': 'completed_optimizer_updates',
        'model': trainer.model.state_dict(), 'ema': trainer.ema_model.state_dict(),
        'optimizer': trainer.opt.state_dict(), 'signature': signature(trainer),
        'resume_history': getattr(trainer, '_resume_history', []),
        'rng': {'python': random.getstate(), 'torch': torch.get_rng_state(),
                'numpy': [np_state[0], np_state[1].tolist(), int(np_state[2]), int(np_state[3]), float(np_state[4])]},
        'runtime': {'torch': str(torch.__version__), 'cuda': torch.version.cuda},
        'data_loader_resume': 'new iterator; worker prefetch and mid-epoch permutation are not replayed exactly',
    }
    if torch.cuda.is_available():
        state['rng']['cuda'] = torch.cuda.get_rng_state_all()
    folder = Path(trainer.results_folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (f'model_{trainer.step}.pt' if save_with_time_stamp else 'model.pt')
    temporary = path.with_name(path.name + f'.tmp.{os.getpid()}')
    try:
        torch.save(state, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(f'checkpoint saved: {path} completed_updates={trainer.step}')


def load_checkpoint(trainer, load_path):
    state = torch.load(load_path, map_location='cpu', weights_only=False)
    if os.environ.get('CIFAR_INFERENCE_ONLY') == '1':
        trainer.model.load_state_dict(state['model'])
        trainer.ema_model.load_state_dict(state['ema'])
        trainer.step = int(state['step'])
        print(f'Inference weights loaded at step {trainer.step}')
        return
    legacy = 'checkpoint_version' not in state
    if legacy:
        if os.environ.get('CIFAR_ALLOW_LEGACY_RESUME') != '1':
            raise ValueError('Legacy checkpoint has no optimizer/RNG. Set CIFAR_ALLOW_LEGACY_RESUME=1 for the first explicit migration.')
        # Our preceding instructions saved this exact final 10k checkpoint after train().
        # Do not guess the zero-based semantics of an arbitrary intermediate file.
        if state.get('step') != 10000 or Path(load_path).name != 'model_10000.pt':
            raise ValueError('Legacy migration only accepts final model_10000.pt at step10000')
    else:
        if state['checkpoint_version'] != VERSION:
            raise ValueError('Unsupported checkpoint version')
        if state.get('signature') != signature(trainer):
            raise ValueError('Resume model/training signature mismatch')
        if state.get('step_semantics') != 'completed_optimizer_updates':
            raise ValueError('Unknown checkpoint step semantics')
    trainer.model.load_state_dict(state['model'])
    trainer.ema_model.load_state_dict(state['ema'])
    trainer.step = int(state['step'])
    trainer._resume_history = list(state.get('resume_history', []))
    event = {'path': str(Path(load_path).resolve()), 'step': trainer.step,
             'optimizer_restored': not legacy, 'rng_restored': not legacy,
             'note': 'legacy 10k weights+EMA; Adam restarted, old moments cannot be recovered' if legacy else 'optimizer and RNG restored; DataLoader order restarts'}
    if not legacy:
        trainer.opt.load_state_dict(state['optimizer'])
        rng = state['rng']
        random.setstate(rng['python'])
        np_state = rng['numpy']
        np.random.set_state((np_state[0], np.array(np_state[1], dtype=np.uint32), *np_state[2:]))
        torch.set_rng_state(rng['torch'].cpu())
        if 'cuda' in rng and torch.cuda.is_available():
            if len(rng['cuda']) != torch.cuda.device_count():
                raise ValueError('CUDA RNG device count changed; keep the same number of visible GPUs')
            torch.cuda.set_rng_state_all([x.cpu() for x in rng['cuda']])
    trainer._resume_history.append(event)
    folder = Path(trainer.results_folder)
    folder.mkdir(parents=True, exist_ok=True)
    with (folder/'resume_log.jsonl').open('a') as handle:
        handle.write(json.dumps(event) + '\n')
    print('RESUME:', json.dumps(event))


def _preview_lab(value):
    """Display-clipped sRGB [-1,1] BCHW to CIE Lab, D65 (CPU float64)."""
    rgb = value.detach().cpu().double().numpy()
    if rgb.ndim != 4 or rgb.shape[1] != 3 or not np.isfinite(rgb).all():
        raise ValueError('Expected finite BCHW sRGB preview tensors with 3 channels')
    clipped_fraction = ((rgb < -1) | (rgb > 1)).mean(axis=(1, 2, 3))
    rgb = ((rgb + 1) * .5).clip(0, 1).transpose(0, 2, 3, 1)
    linear = np.where(rgb > .04045, ((rgb + .055) / 1.055)**2.4, rgb / 12.92)
    matrix = np.array([[.4124564, .3575761, .1804375],
                       [.2126729, .7151522, .0721750],
                       [.0193339, .1191920, .9503041]])
    xyz = (linear @ matrix.T) / np.array([.95047, 1., 1.08883])
    delta = 6 / 29
    f = np.where(xyz > delta**3, np.cbrt(xyz), xyz / (3 * delta**2) + 4 / 29)
    lab = np.stack([116*f[..., 1]-16, 500*(f[..., 0]-f[..., 1]),
                    200*(f[..., 1]-f[..., 2])], axis=-1)
    return lab, clipped_fraction


def log_preview_color(trainer, samples):
    """Measure existing previews only: no extra sampling or RNG consumption."""
    import csv
    import hashlib
    from datetime import datetime, timezone

    target, _ = _preview_lab(samples['og'])
    target_chroma = np.linalg.norm(target[..., 1:], axis=-1).mean(axis=(1, 2))
    hashes = [hashlib.sha256(x.detach().cpu().float().contiguous().numpy().tobytes()).hexdigest()
              for x in samples['og']]
    event = {'schema_version': 2, 'step': int(trainer.step),
             'utc': datetime.now(timezone.utc).isoformat(),
             'split': getattr(trainer, 'metric_split', 'train_preview'), 'weights': 'ema',
             'preview_index': int(trainer.step // trainer.save_and_sample_every),
             'batch_size': len(target), 'target_tensor_sha256': hashes,
             'metric': 'CIE76, Lab D65, display-clipped sRGB before PNG quantization',
             'comparison': 'paired against og; match dataset indices or target hashes across runs',
             'phases': {}, 'dataset_indices': getattr(trainer, 'metric_indices', None)}
    for name in ['xt', 'direct_recons', 'recon']:
        lab, clipped_fraction = _preview_lab(samples[name])
        if lab.shape != target.shape:
            raise ValueError(f'{name} and target preview shapes differ')
        delta = lab - target
        rgb = ((samples[name].detach().cpu().double().numpy() + 1) * .5).clip(0, 1)
        og_rgb = ((samples['og'].detach().cpu().double().numpy() + 1) * .5).clip(0, 1)
        rgb_delta = rgb - og_rgb
        values = {
            'rgb_mae': np.abs(rgb_delta).mean(axis=(1, 2, 3)),
            'rgb_mse': np.square(rgb_delta).mean(axis=(1, 2, 3)),
            'rgb_rmse': np.sqrt(np.square(rgb_delta).mean(axis=(1, 2, 3))),
            'delta_e76': np.linalg.norm(delta, axis=-1).mean(axis=(1, 2)),
            'ab_error': np.linalg.norm(delta[..., 1:], axis=-1).mean(axis=(1, 2)),
            'chroma': np.linalg.norm(lab[..., 1:], axis=-1).mean(axis=(1, 2)),
            'target_chroma': target_chroma,
            'clipped_fraction': clipped_fraction,
        }
        event['phases'][name] = {
            'mean': {k: float(v.mean()) for k, v in values.items()},
            'per_image': [{k: float(v[i]) for k, v in values.items()} for i in range(len(target))],
        }
    baseline = event['phases']['xt']['mean']['delta_e76']
    folder = Path(trainer.results_folder)
    folder.mkdir(parents=True, exist_ok=True)
    with (folder/'preview_color_metrics.jsonl').open('a') as handle:
        handle.write(json.dumps(event, allow_nan=False) + '\n')
    # V2 adds RGB columns; keep pre-existing V1 CSV intact on upgrade.
    csv_path = folder/'preview_color_metrics_v2.csv'
    header = ['step', 'utc', 'preview_index', 'split', 'weights', 'batch_size', 'phase',
              'rgb_mae', 'rgb_mse', 'rgb_rmse', 'delta_e76', 'ab_error', 'chroma', 'target_chroma', 'clipped_fraction', 'delta_e76_gain_vs_gray']
    needs_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open('a', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        if needs_header:
            writer.writeheader()
        for phase, metrics in event['phases'].items():
            row = {k: event[k] for k in header[:6]}
            row.update(phase=phase, **metrics['mean'],
                       delta_e76_gain_vs_gray=baseline-metrics['mean']['delta_e76'])
            writer.writerow(row)
    print('PREVIEW_COLOR:', json.dumps({'step': event['step'],
          'delta_e76': {k: v['mean']['delta_e76'] for k, v in event['phases'].items()},
          'rgb_mae': {k: v['mean']['rgb_mae'] for k, v in event['phases'].items()},
          'split': event['split']}, allow_nan=False))
    return event
