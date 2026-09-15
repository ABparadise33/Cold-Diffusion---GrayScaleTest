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
