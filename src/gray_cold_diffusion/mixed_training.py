"""Mixed-domain full-gray pilot; fixed diagnostics do not change training RNG."""
import copy
import hashlib
import json
import os
import uuid
from collections import Counter
from pathlib import Path

import torch

from .color import denormalize_rgb, normalize_rgb, normalized_lab_to_rgb, rgb_to_normalized_lab
from .io import save_stage_strip, append_csv
from .metrics import delta_e76, psnr
from .official_colorization import channel_gray
from .official_training import OfficialTrainer


PRE_MONITOR_MIXED_SHA256 = 'c63137ea1d9ecb7eb52361be5c402029912db2ee01162040724823d8977c3d98'


def mixed_fingerprint():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def domain_of(path):
    domain = Path(path).name.split('__', 1)[0]
    if domain not in ('UIEB', 'DIV2K'):
        raise ValueError(f'missing domain prefix: {path}')
    return domain


def domain_weights(items):
    domains = [domain_of(p) for p in items]
    counts = Counter(domains)
    if set(counts) != {'UIEB', 'DIV2K'}:
        raise ValueError('both UIEB and DIV2K are required')
    return torch.tensor([1.0 / counts[d] for d in domains], dtype=torch.double)


def state_stats(state, target):
    rgb = denormalize_rgb(state)
    lab = rgb_to_normalized_lab(rgb)
    target_lab = rgb_to_normalized_lab(target)
    chroma = (lab[:, 1:].square().sum(1).sqrt() * 128).mean()
    target_chroma = (target_lab[:, 1:].square().sum(1).sqrt() * 128).mean()
    return {
        'rgb_mae': (rgb-target).abs().mean().item(),
        'rgb_rmse': (rgb-target).square().mean().sqrt().item(),
        'ab_error': (128*(lab[:, 1:]-target_lab[:, 1:])).square().sum(1).sqrt().mean().item(),
        'finite': bool(torch.isfinite(state).all()),
        'state_min': state.min().item(), 'state_max': state.max().item(),
        'clipped_fraction': ((state < -1) | (state > 1)).float().mean().item(),
        'rgb_chroma_rms_unclipped': (state - channel_gray(state)).square().mean().sqrt().item(),
        'lab_a_mean': (128 * lab[:, 1]).mean().item(),
        'lab_b_mean': (128 * lab[:, 2]).mean().item(),
        'lab_chroma': chroma.item(),
        'chroma_ratio': (chroma / target_chroma.clamp_min(1e-8)).item(),
        'delta_e76': delta_e76(lab, target_lab).mean().item(),
        'psnr': psnr(rgb, target).mean().item(),
    }


class MixedTrainer(OfficialTrainer):
    def __init__(self, *args, **kwargs):
        self.timestep_counts = Counter()
        self.domain_counts = Counter()
        super().__init__(*args, **kwargs)

    def load_checkpoint(self, path):
        payload = torch.load(path, map_location='cpu', weights_only=False)
        old_hash = payload.get('config', {}).get('implementation', {}).get('mixed_sha256')
        del payload
        current = self.config['implementation']['mixed_sha256']
        # One explicitly reviewed diagnostics-only migration; all superclass checks still apply.
        if old_hash == PRE_MONITOR_MIXED_SHA256:
            self.config['implementation']['mixed_sha256'] = old_hash
        try:
            super().load_checkpoint(path)
        finally:
            self.config['implementation']['mixed_sha256'] = current
        if old_hash == PRE_MONITOR_MIXED_SHA256:
            migration = {'kind': 'monitoring_revision_migration', 'step': self.step,
                         'from_mixed_sha256': old_hash, 'to_mixed_sha256': current}
            self._append(migration)
            self.run_metadata['monitoring_revision_migration'] = migration
            self._write_manifest()

    def _metadata(self):
        result = super()._metadata()
        result['sampling'] = 'replacement, probability 0.5 per domain; not exactly half each batch'
        for label, loader in [('train', self.train_loader), ('val', self.val_loader)]:
            result['data'][label]['domains'] = dict(Counter(domain_of(p) for p in loader.dataset.items))
            result['data'][label]['content_sha256'] = {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in loader.dataset.items}
        return result

    def _next_batch(self):
        batch = super()._next_batch()
        self.domain_counts.update(domain_of(name) for name in batch['name'])
        return batch

    def _training_pair(self, *args):
        result = super()._training_pair(*args)
        self.timestep_counts.update(result[2].detach().cpu().tolist())
        return result

    def step_ema(self):
        super().step_ema()
        if self.step % self.config['training']['log_every'] == 0:
            self._append({'kind': 'training_exposure', 'step': self.step,
                          'counts_since_start_or_resume': dict(self.domain_counts),
                          'timestep_counts_since_start_or_resume': dict(self.timestep_counts),
                          'full_gray_count': self.timestep_counts[self.bridge.steps]})

    def _append(self, row):
        with (self.output / 'debug_diagnostics.jsonl').open('a') as handle:
            handle.write(json.dumps(row) + '\n')

    @torch.no_grad()
    def write_diagnostics(self):
        was_training = self.model.training
        ema_was_training = self.ema.training
        self.model.eval()
        self.ema.eval()
        summaries = {}
        pass_id = uuid.uuid4().hex
        try:
            for split, loader in [('train', self.train_loader), ('val', self.val_loader)]:
                dataset = copy.copy(loader.dataset)
                dataset.augment = False
                used = Counter()
                for index, path in enumerate(dataset.items):
                    domain = domain_of(path)
                    if used[domain] >= self.config['training'].get('diagnostic_count_per_domain', 2):
                        continue
                    used[domain] += 1
                    rgb = dataset[index]['raw'][None].to(self.device)
                    target = normalize_rgb(rgb)
                    anchor = channel_gray(target)
                    t = torch.full((1,), self.bridge.steps, device=self.device, dtype=torch.long)
                    direct = self.ema(anchor, t)
                    online = self.model(anchor, t)
                    predicted, trajectory = self.bridge.sample(self.ema, anchor, return_trajectory=True)
                    row = {'kind': 'fixed_center_crop', 'step': self.step, 'split': split,
                           'domain': domain, 'image': str(path), 't': self.bridge.steps,
                           'scope': 'fixed diagnostic subset, not full-domain benchmark',
                           'rgb_roundtrip_max': (denormalize_rgb(target)-rgb).abs().max().item(),
                           'lab_roundtrip_max': (normalized_lab_to_rgb(rgb_to_normalized_lab(rgb))-rgb).abs().max().item(),
                           'endpoint_gray_max': (self.bridge.degrade(target, anchor, t)-anchor).abs().max().item(),
                           'target': state_stats(target, rgb), 'gray': state_stats(anchor, rgb),
                           'direct_ema': state_stats(direct, rgb),
                           'direct_online': state_stats(online, rgb),
                           'sample_ema': state_stats(predicted, rgb),
                           'trajectory': [{'t': self.bridge.steps-i, **state_stats(x, rgb)}
                                          for i, x in enumerate(trajectory)],
                           'timestep_counts_since_start_or_resume': dict(self.timestep_counts)}
                    row['pass_id'] = pass_id
                    row['target_tensor_sha256'] = hashlib.sha256(rgb.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
                    preview_path = self.output / 'debug_previews' / f'step_{self.step:06d}' / split / f'{path.stem}.png'
                    save_stage_strip([('target', rgb), ('gray', denormalize_rgb(anchor)),
                                      ('direct EMA', denormalize_rgb(direct)),
                                      ('sample EMA', denormalize_rgb(predicted))],
                                     preview_path)
                    row['preview_sha256'] = hashlib.sha256(preview_path.read_bytes()).hexdigest()
                    self._append(row)
                    for phase in ['gray', 'direct_ema', 'sample_ema']:
                        values = {key: row[phase][key] for key in ['rgb_mae', 'rgb_rmse', 'delta_e76', 'ab_error', 'lab_chroma', 'chroma_ratio']}
                        append_csv(self.output/'fixed_color_per_image.csv',
                                   {'step': self.step, 'pass_id': pass_id, 'split': split, 'domain': domain, 'image': path.name,
                                    'target_sha256': row['target_tensor_sha256'], 'preview_sha256': row['preview_sha256'], 'phase': phase, **values})
                        summaries.setdefault((split, domain, phase), []).append(values)
        finally:
            self.model.train(was_training)
            self.ema.train(ema_was_training)
        for (split, domain, phase), values in summaries.items():
            append_csv(self.output/'fixed_color_summary.csv',
                       {'step': self.step, 'pass_id': pass_id, 'split': split, 'domain': domain, 'phase': phase,
                        'count': len(values), 'scope': 'fixed center crop diagnostic subset',
                        **{key: sum(x[key] for x in values)/len(values) for key in values[0]}})

    def _save_full_scene_preview(self):
        # Official validation has already populated last_validation here. Persist
        # the completed update before any optional large-image GPU inference.
        self.save_checkpoint()
        self._append({'kind': 'validation_phase', 'step': self.step,
                      'phase': 'checkpoint_saved_before_previews'})
        self.write_diagnostics()
        enabled = os.environ.get('MIXED_FULL_SCENE_PREVIEWS', '1')
        if enabled not in ('0', '1'):
            raise ValueError('MIXED_FULL_SCENE_PREVIEWS must be 0 or 1')
        if enabled == '0':
            self._append({'kind': 'validation_phase', 'step': self.step,
                          'phase': 'full_scene_preview_disabled_by_user'})
            return
        self._append({'kind': 'validation_phase', 'step': self.step,
                      'phase': 'full_scene_preview_start',
                      'cuda_launch_blocking': os.environ.get('CUDA_LAUNCH_BLOCKING', 'unset')})
        try:
            super()._save_full_scene_preview()
        except Exception as error:
            # Do not issue more CUDA operations or reinterpret a launch failure
            # as OOM. The earlier checkpoint is the recovery boundary.
            self._append({'kind': 'validation_phase', 'step': self.step,
                          'phase': 'full_scene_preview_failed', 'error_type': type(error).__name__,
                          'error': str(error)})
            raise
        self._append({'kind': 'validation_phase', 'step': self.step,
                      'phase': 'full_scene_preview_complete'})

    def validate(self):
        self._append({'kind': 'validation_phase', 'step': self.step,
                      'phase': 'full_validation_start'})
        return super().validate()
