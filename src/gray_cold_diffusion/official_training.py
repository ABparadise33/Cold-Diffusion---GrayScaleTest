"""DIV2K adaptation of the pinned upstream training recipe, with explicit records."""
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess

import torch

from .color import denormalize_rgb, normalize_rgb, rgb_to_normalized_lab
from .engine import Trainer
from .io import append_csv, atomic_torch_save, update_ema
from .metrics import delta_e76, psnr, ssim, trajectory_monotonic_fraction
from .official_colorization import UPSTREAM_COMMIT, channel_gray
from .official_preview import save_full_scene_previews
from .report import save_training_report


PRE_SPATIAL_SOURCE_SHA256 = {
    'official_convnext.py': 'd71636fc0a4305f1eae12607587793a6816b7c68a681dc460170be2d16aa6dad',
    'official_colorization.py': '075c913bbae6e0e315a6e6d612854514e76c70ecd2cbc6664eda0331a6069dde',
    'official_training.py': '93dbac7ae808426de6da27d4d5c1c7f1028e41003c72026798df52847cba4c0c',
    'official_preview.py': '073457955ab7d8926e4f1b7d4e9f66667cdf76d71f00e29950190ff3aff238ec',
    'factory.py': '9771dd28f0120774040b956975fb6570023c0364c6c0b2c3af1dae9b14b44690',
    'engine.py': '1d03b085a3916c805175eac1a55815f4526df928253d2da09c300e212e693e02',
    'data.py': '3d36efdfcfda94b80063891fa1b489e54bd4934226ea4958f2891ffdcada5e29',
    'color.py': '1928debe792621428efa9210ce4efda1d83fd772eca0d202ef42b63ef0b94ffa',
    'io.py': '6f96d84bb17f8d949aeb32cd092b98cb152cb124d6edf0d10531e9761092985e',
    'tiling.py': '09c61ac095572d105449dd6af6745bdcb3fca761a5806d3cc26278a92cc5355c',
}


def implementation_fingerprint():
    base = Path(__file__).parent
    names = ('official_convnext.py', 'official_colorization.py', 'official_training.py', 'official_preview.py',
             'spatial_chroma.py', 'factory.py', 'engine.py', 'data.py', 'color.py', 'io.py', 'tiling.py')
    return {name: hashlib.sha256((base / name).read_bytes()).hexdigest() for name in names}


def compatible_preview_revision(saved, current):
    """Only the reviewed fixed-0803 -> random-five preview migration is allowed.

    All other source hashes and implementation settings remain strict. Do not
    extend this migration to future training/model changes without a new audit.
    """
    if saved == current:
        return True
    if not isinstance(saved, dict) or not isinstance(current, dict):
        return False
    old_hashes = saved.get('source_sha256', {})
    new_hashes = current.get('source_sha256', {})
    candidates = [old_hashes]
    if (old_hashes.get('official_training.py') ==
            '789c81f26dfa83f4c739c4d2f12adb87b1769a0af593ac98983af4d6cf9c7b3c'
            and 'official_preview.py' not in old_hashes):
        candidates.extend([
            {**old_hashes, 'official_training.py': new_hashes.get('official_training.py'),
             'official_preview.py': new_hashes['official_preview.py']},
            {**old_hashes, 'official_training.py': PRE_SPATIAL_SOURCE_SHA256['official_training.py'],
             'official_preview.py': PRE_SPATIAL_SOURCE_SHA256['official_preview.py']},
        ])
    for hashes in candidates:
        if hashes == PRE_SPATIAL_SOURCE_SHA256:
            hashes = new_hashes
        if {**saved, 'source_sha256': hashes} == current:
            return True
    return False


def preflight_official_run(config, args):
    """Fail before replacing a prior run or allocating the large model."""
    cfg = config['training']
    for key in ('max_steps', 'batch_size', 'grad_accum', 'validate_every', 'save_every', 'log_every'):
        if int(cfg[key]) < 1:
            raise ValueError(f'{key} must be >=1')
    if cfg.get('amp', False):
        raise ValueError('this official baseline requires FP32 (amp=false)')
    if float(config['data'].get('saturation_factor', 1)) != 1:
        raise ValueError('this baseline is saturation1 only')
    if int(cfg.get('preview_count', 5)) < 1:
        raise ValueError('preview_count must be >=1')
    out = Path(config['output_dir'])
    if out.exists() and any(out.iterdir()) and not (args.resume or args.resume_if_exists):
        raise FileExistsError(f'run already exists: {out}; explicitly --resume or choose a NEW output')
    if args.resume_if_exists and out.exists() and any(out.iterdir()):
        if not (out / 'checkpoints/latest.pt').is_file():
            raise FileNotFoundError('existing run has no latest.pt; refusing to restart over its records')
    existing_parent = next(p for p in (out, *out.parents) if p.exists())
    free = shutil.disk_usage(existing_parent).free / 1024**3
    if free < float(cfg.get('min_free_disk_gb', 5)):
        raise OSError(f'only {free:.2f} GiB free before training; need checkpoint headroom')
    if not args.train_dir or not args.val_dir:
        raise ValueError('official DIV2K run needs --train-dir and --val-dir')
    if Path(args.train_dir).resolve() == Path(args.val_dir).resolve():
        raise ValueError('train and validation directories must be different')
    spatial = config['mode'] == 'spatial_chroma_colorization'
    config['implementation'] = {
        'upstream_commit': UPSTREAM_COMMIT,
        'source_sha256': implementation_fingerprint(),
        'start': 'full_gray', 'public_time': 's=1..T; upstream label=s-1',
        'sampler': config['diffusion']['sampler'],
    }
    if spatial:
        config['implementation'].update(
            operator='nested spatial RGB-chroma mask; expected retained fraction=(T-t)/T',
            sampling_seed=int(config['diffusion'].get('sampling_seed', config['seed'])),
            status='novel experiment; not the paper RGB amplitude operator',
        )
    config['data']['train_dir'] = str(Path(args.train_dir).resolve())
    config['data']['val_dir'] = str(Path(args.val_dir).resolve())
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    label = 'SPATIAL CHROMA MASK' if spatial else 'OFFICIAL RGB / FULL GRAY'
    print(f'{label} / saturation1 / FP32 / effective batch={cfg["batch_size"] * cfg["grad_accum"]}')
    print(f'sampler={config["diffusion"]["sampler"]}; fresh unless resume explicitly requested')


class OfficialTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_delta_e = math.inf
        self.last_validation = None
        self.run_metadata = self._metadata()
        # Existing run manifest is checked by load_checkpoint before being updated.
        if not (self.output / 'run_manifest.json').exists():
            self._write_manifest()

    def _prepare(self, batch):
        # Sat1 baseline: use the source itself, avoiding even a roundoff-level
        # saturation edit in the historical NaturalImageDataset target helper.
        rgb = batch['raw'].to(self.device, non_blocking=True)
        state = normalize_rgb(rgb)
        return rgb, rgb, state, state, channel_gray(state)

    def _metadata(self):
        data = {}
        for label, loader in [('train', self.train_loader), ('val', self.val_loader)]:
            items = loader.dataset.items
            rows = [{'name': p.name, 'bytes': p.stat().st_size} for p in items]
            data[label] = {'count': len(items), 'files': rows,
                           'listing_sha256': hashlib.sha256(json.dumps(rows).encode()).hexdigest()}
        try:
            root = Path(__file__).resolve().parents[2]
            revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            revision = None
        preview_folders = ['samples', 'trajectories', 'predictions']
        if self.config['training'].get('preview_direct', True):
            preview_folders.append('direct_predictions')
        return {'config': self.config, 'data': data, 'code_commit': revision,
                'parameters': sum(p.numel() for p in self.model.parameters()),
                'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
                'device': str(self.device), 'seed': self.config['seed'],
                'effective_batch': self.grad_accum * self.config['training']['batch_size'],
                'dataset_fingerprint_kind': 'sorted names and byte sizes, NOT image content hashes',
                'data_loader_resume': 'new shuffled iterator; not a bitwise replay of crop/order',
                'validation': 'all fixed center crops; separate random full-scene tiled previews',
                'preview': {'count': int(self.config['training'].get('preview_count', 5)),
                            'every': self.config['training']['validate_every'], 'also_at_final_step': True,
                            'selection': 'local seed+step RNG; validation split only',
                            'layout': f'previews/step_<step>/{{{",".join(preview_folders)}}}'}}

    def _write_manifest(self):
        (self.output / 'run_manifest.json').write_text(json.dumps(self.run_metadata, indent=2))

    def step_ema(self):
        # Upstream increments its counter AFTER the optimizer + EMA branch.
        cfg = self.config['training']
        source_step = self.step - 1
        if source_step % int(cfg['ema_update_every']) == 0:
            if source_step < int(cfg.get('ema_start_step', 2000)):
                self.ema.load_state_dict(self.model.state_dict())
            else:
                update_ema(self.ema, self.model, float(cfg['ema_decay']))

    def _checkpoint_payload(self):
        payload = super()._checkpoint_payload()
        payload['best_delta_e76'] = self.best_delta_e
        payload['run_metadata'] = self.run_metadata
        return payload

    def load_checkpoint(self, path):
        payload = torch.load(path, map_location='cpu', weights_only=False)
        saved = payload.get('config', {})
        for key in ('mode', 'model', 'diffusion', 'data', 'seed'):
            if saved.get(key) != self.config.get(key):
                raise ValueError(f'official resume mismatch: {key}; do not use legacy or different-run weights')
        if not compatible_preview_revision(saved.get('implementation'), self.config.get('implementation')):
            raise ValueError('official resume mismatch: implementation; only the known preview-only migration is allowed')
        preview_migration = saved.get('implementation') != self.config.get('implementation')
        for key in ('learning_rate', 'ema_decay', 'ema_update_every', 'ema_start_step', 'amp'):
            if saved.get('training', {}).get(key) != self.config['training'].get(key):
                raise ValueError(f'official resume training mismatch: {key}')
        old_training = saved['training']
        if old_training['batch_size'] * old_training['grad_accum'] != self.config['training']['batch_size'] * self.grad_accum:
            raise ValueError('official resume effective batch changed')
        previous_data = payload.get('run_metadata', {}).get('data')
        if previous_data != self.run_metadata['data']:
            raise ValueError('official resume dataset listing changed')
        if self.max_steps <= int(payload['step']):
            raise ValueError('max_steps must exceed checkpoint step to continue training')
        self.best_delta_e = float(payload.get('best_delta_e76', math.inf))
        if preview_migration:
            self.run_metadata['preview_only_resume_migration'] = {
                'from_implementation': saved['implementation'],
                'reason': 'fixed 0803 preview replaced by five seeded validation images per step folder',
            }
            print('resume: verified known preview-only revision; training/model fingerprints otherwise unchanged')
        del payload
        super().load_checkpoint(path)
        self.run_metadata['resumed_from'] = str(Path(path).resolve())
        self.run_metadata['resumed_at_step'] = self.step
        self._write_manifest()

    @torch.no_grad()
    def validate(self):
        self.ema.eval()
        include_direct = self.config['training'].get('validation_direct', True)
        names = ['psnr', 'ssim', 'delta_e76', 'monotonic', 'val_l1', 'gray_delta_e76',
                 'pred_chroma', 'target_chroma']
        if include_direct:
            names.extend(['direct_delta_e76', 'direct_val_l1'])
        totals = dict.fromkeys(names, 0.0)
        count = 0
        for batch in self.val_loader:  # no max_val_batches truncation
            _, reference, _, target, anchor = self._prepare(batch)
            t = torch.full((len(target),), self.bridge.steps, device=self.device, dtype=torch.long)
            direct_state = self.ema(anchor, t) if include_direct else None
            state, trajectory = self.bridge.sample(self.ema, anchor, return_trajectory=True)
            pred = denormalize_rgb(state)
            pred_lab, target_lab = rgb_to_normalized_lab(pred), rgb_to_normalized_lab(reference)
            n = len(pred)
            totals['psnr'] += psnr(pred, reference).sum().item()
            totals['ssim'] += ssim(pred, reference).sum().item()
            totals['delta_e76'] += delta_e76(pred_lab, target_lab).sum().item()
            if direct_state is not None:
                direct = denormalize_rgb(direct_state)
                totals['direct_delta_e76'] += delta_e76(rgb_to_normalized_lab(direct), target_lab).sum().item()
                totals['direct_val_l1'] += (direct_state - target).abs().mean().item() * n
            totals['gray_delta_e76'] += delta_e76(rgb_to_normalized_lab(denormalize_rgb(anchor)), target_lab).sum().item()
            totals['val_l1'] += (state - target).abs().mean().item() * n
            totals['pred_chroma'] += (pred_lab[:, 1:].square().sum(1).sqrt() * 128).mean().item() * n
            totals['target_chroma'] += (target_lab[:, 1:].square().sum(1).sqrt() * 128).mean().item() * n
            labs = [rgb_to_normalized_lab(denormalize_rgb(x)) for x in trajectory]
            totals['monotonic'] += trajectory_monotonic_fraction(labs, target_lab).sum().item()
            count += n
        if count == 0:
            raise RuntimeError('empty official validation set')
        metrics = {key: value / count for key, value in totals.items()}
        metrics['chroma_ratio'] = metrics['pred_chroma'] / max(metrics['target_chroma'], 1e-8)
        self.last_validation = metrics
        append_csv(self.output / 'full_gray_metrics.csv', {'step': self.step, 'count': count, **metrics})
        self._save_full_scene_preview()
        return metrics

    @torch.no_grad()
    def _save_full_scene_preview(self):
        save_full_scene_previews(self)

    def save_checkpoint(self, is_best=False):
        checkpoint_dir = self.output / 'checkpoints'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        cfg = self.config['training']
        free = shutil.disk_usage(checkpoint_dir).free / 1024**3
        if free < float(cfg.get('min_free_disk_gb', 5)):
            raise OSError(f'only {free:.2f} GiB free; refusing checkpoint overwrite; free space then resume latest')
        score = self.last_validation['delta_e76'] if self.last_validation else math.inf
        is_best = score < self.best_delta_e
        if is_best:
            self.best_delta_e = score
        payload = self._checkpoint_payload()
        atomic_torch_save(payload, checkpoint_dir / 'latest.pt')
        if is_best:
            atomic_torch_save(payload, checkpoint_dir / 'best.pt')
        milestone = int(cfg.get('checkpoint_milestone_every', 50000))
        if self.step == self.max_steps or (milestone > 0 and self.step % milestone == 0):
            atomic_torch_save(payload, checkpoint_dir / f'step_{self.step:06d}.pt')
        if (self.output / 'metrics.csv').exists():
            save_training_report(self.output / 'metrics.csv', self.output)
