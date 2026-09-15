import copy
import importlib.util
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch


def load_tool(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1] / 'tools' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


support = load_tool('official_cifar_resume_support')
patcher = load_tool('patch_official_cifar_resume')


def trainer(tmp_path):
    model = torch.nn.Linear(2, 1)
    model.image_size = (32, 32)
    model.num_timesteps = 20
    model.loss_type = 'l1'
    model.train_routine = 'Final'
    model.sampling_routine = 'x0_step_down'
    model.to_lab = False
    model.forward_process = SimpleNamespace(decolor_routine='Linear', decolor_total_remove=True, decolor_ema_factor=.9)
    return SimpleNamespace(model=model, ema_model=copy.deepcopy(model),
                           opt=torch.optim.Adam(model.parameters(), lr=2e-5), step=10000,
                           batch_size=32, gradient_accumulate_every=2,
                           ema=SimpleNamespace(beta=.995), update_ema_every=10,
                           step_start_ema=2000, results_folder=tmp_path)


def update(t):
    t.opt.zero_grad()
    t.model(torch.ones(3, 2)).square().mean().backward()
    t.opt.step()


def test_restore_adam_rng_and_next_update(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    a = trainer(tmp_path)
    update(a)
    a._monitoring_state = {'loss_window': [{'step': 10000, 'loss': .3}], 'last_monitor_step': 9000}
    support.save_checkpoint(a)
    # Compatible with default torch.load on newer PyTorch, without numpy globals.
    assert torch.load(tmp_path/'model.pt', weights_only=True)['step'] == 10000
    expected = (random.random(), np.random.rand(), torch.rand(3))
    update(a)
    b = trainer(tmp_path/'resumed')
    support.load_checkpoint(b, tmp_path/'model.pt')
    actual = (random.random(), np.random.rand(), torch.rand(3))
    assert actual[:2] == expected[:2]
    assert torch.equal(actual[2], expected[2])
    update(b)
    for x, y in zip(a.model.parameters(), b.model.parameters()):
        torch.testing.assert_close(x, y, rtol=0, atol=0)
    assert b.step == 10000
    assert b._monitoring_state == a._monitoring_state
    assert b._resume_history[-1]['optimizer_restored']


def test_legacy_requires_explicit_migration(tmp_path, monkeypatch):
    a = trainer(tmp_path)
    path = tmp_path/'model_10000.pt'
    torch.save({'step':10000, 'model':a.model.state_dict(), 'ema':a.ema_model.state_dict()}, path)
    monkeypatch.delenv('CIFAR_ALLOW_LEGACY_RESUME', raising=False)
    with pytest.raises(ValueError, match='no optimizer'):
        support.load_checkpoint(a, path)
    monkeypatch.setenv('CIFAR_ALLOW_LEGACY_RESUME', '1')
    support.load_checkpoint(a, path)
    assert not a._resume_history[-1]['optimizer_restored']
    assert len(a.opt.state) == 0
    assert 'Adam restarted' in (tmp_path/'resume_log.jsonl').read_text()


def test_signature_mismatch_and_inference(tmp_path, monkeypatch):
    a = trainer(tmp_path)
    support.save_checkpoint(a)
    b = trainer(tmp_path/'new')
    b.batch_size = 1
    with pytest.raises(ValueError, match='signature mismatch'):
        support.load_checkpoint(b, tmp_path/'model.pt')
    monkeypatch.setenv('CIFAR_INFERENCE_ONLY', '1')
    support.load_checkpoint(b, tmp_path/'model.pt')
    for x, y in zip(a.ema_model.parameters(), b.ema_model.parameters()):
        assert torch.equal(x, y)


def test_atomic_save_preserves_previous(tmp_path, monkeypatch):
    a = trainer(tmp_path)
    support.save_checkpoint(a)
    original = (tmp_path/'model.pt').read_bytes()
    def fail(*args):
        raise OSError('disk full')
    monkeypatch.setattr(torch, 'save', fail)
    with pytest.raises(OSError, match='disk full'):
        support.save_checkpoint(a)
    assert (tmp_path/'model.pt').read_bytes() == original


def test_patcher_completed_update_count_and_idempotence(tmp_path):
    # Minimal executable Trainer with the upstream loop's relevant ordering.
    (tmp_path/'diffusion').mkdir()
    source = tmp_path/'diffusion/diffusion.py'
    source.write_text('''class Trainer:
    def save(self, save_with_time_stamp=False):
        pass
    def load(self, load_path):
        pass
    def train(self):
        while self.step < self.train_num_steps:
            for data in [1]:
                loss = self.model(data)
            self.opt.step()
            if self.step % self.update_ema_every == 0:
                self.step_ema()
            if self.step != 0 and self.step % self.save_and_sample_every == 0:
                if True:
                    sample_dict = {}
                    og_img = None
                    sample_dict['og'] = og_img
                self.save()
            self.step += 1
        print('training completed')
    def save_gif(self):
        pass
''')
    (tmp_path/'train.py').write_text('''trainer = Trainer(
    diffusion, args.dataset_folder,
    image_size = image_size,
)
trainer.train()
trainer.save()
trainer.save(save_with_time_stamp=True)
''')
    patcher.patch(tmp_path)
    first = source.read_text()
    patcher.patch(tmp_path)
    assert source.read_text() == first
    namespace = {'_cifar_log_color': lambda *args: None, '_cifar_monitor_init': lambda *args: None, '_cifar_record_update': lambda *args: None}
    exec('\n'.join(x for x in first.splitlines() if not x.startswith('from codex_cifar_')), namespace)
    t = namespace['Trainer']()
    t.step, t.train_num_steps = 10000, 10003
    t.update_ema_every, t.save_and_sample_every = 10, 1
    updates, saved, ema = [], [], []
    t.model = lambda data: torch.tensor(1.)
    t.opt = SimpleNamespace(step=lambda: updates.append(1))
    t.save = lambda save_with_time_stamp=False: saved.append((t.step, len(updates)))
    t.step_ema = lambda: ema.append(t.step)
    t.train()
    assert len(updates) == 3
    assert saved == [(10001,1), (10002,2), (10003,3), (10003,3), (10003,3)]
    assert ema == [10000]
    assert 'save_with_time_stamp_every=10000' in (tmp_path/'train.py').read_text()


def test_preview_color_reference_values():
    black, _ = support._preview_lab(torch.full((1, 3, 2, 2), -1.))
    white, _ = support._preview_lab(torch.ones(1, 3, 2, 2))
    assert np.linalg.norm(white-black, axis=-1).mean() == pytest.approx(100, abs=1e-4)
    red, _ = support._preview_lab(torch.tensor([1., -1., -1.]).reshape(1, 3, 1, 1))
    np.testing.assert_allclose(red.flatten(), [53.2408, 80.0925, 67.2032], atol=.001)


def test_preview_metrics_append_identity_and_rng(tmp_path):
    import csv
    import json
    t = trainer(tmp_path)
    t.save_and_sample_every = 1000
    original = torch.tensor([1., -1., -1.]).reshape(1, 3, 1, 1).expand(2, 3, 4, 4).clone()
    gray = original.mean(1, keepdim=True).expand_as(original)
    samples = {'og': original, 'xt': gray, 'direct_recons': original, 'recon': original}
    before = original.clone()
    rng = torch.get_rng_state().clone()
    support.log_preview_color(t, samples)
    t.step += 1000
    support.log_preview_color(t, samples)
    assert torch.equal(before, original)
    assert torch.equal(rng, torch.get_rng_state())
    events = [json.loads(x) for x in (tmp_path/'preview_color_metrics.jsonl').read_text().splitlines()]
    assert [e['step'] for e in events] == [10000, 11000]
    event = events[0]
    assert event['phases']['recon']['mean']['delta_e76'] == 0
    assert event['phases']['xt']['mean']['delta_e76'] > 50
    assert event['phases']['xt']['mean']['chroma'] < .001
    assert len(event['phases']['xt']['per_image']) == 2
    assert event['target_tensor_sha256'] == events[1]['target_tensor_sha256']
    with (tmp_path/'preview_color_metrics_v2.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 6
    assert float(rows[2]['delta_e76_gain_vs_gray']) > 0


def test_preview_clipping_and_invalid_input():
    lab, fraction = support._preview_lab(torch.full((1, 3, 1, 1), 2.))
    white, _ = support._preview_lab(torch.ones(1, 3, 1, 1))
    np.testing.assert_array_equal(lab, white)
    assert fraction.tolist() == [1.]
    with pytest.raises(ValueError, match='finite'):
        support._preview_lab(torch.full((1, 3, 1, 1), float('nan')))


def test_rgb_metrics_scale(tmp_path):
    t = trainer(tmp_path)
    t.save_and_sample_every = 1000
    black = torch.full((1, 3, 2, 2), -1.)
    white = torch.ones_like(black)
    event = support.log_preview_color(t, {'og': black, 'xt': white, 'direct_recons': black, 'recon': white})
    for metric in ['rgb_mae', 'rgb_mse', 'rgb_rmse']:
        assert event['phases']['xt']['mean'][metric] == 1.
        assert event['phases']['direct_recons']['mean'][metric] == 0.


def test_evaluation_summary_weights_short_final_batch(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, 'official_cifar_resume_support', support)
    evaluator = load_tool('evaluate_official_cifar10')
    def event(n, gray, result):
        return {'batch_size': n, 'phases': {
            'xt': {'mean': {'delta_e76': gray, 'rgb_mae': gray/100}},
            'recon': {'mean': {'delta_e76': result, 'rgb_mae': result/100}},
        }}
    summary = evaluator.summarize([event(32, 10, 5), event(16, 40, 20)])
    assert summary['count'] == 48
    assert summary['phases']['xt']['delta_e76'] == 20
    assert summary['phases']['recon']['delta_e76'] == 10
    assert summary['phases']['recon']['delta_e76_gain_vs_gray'] == 10
    assert summary['phases']['recon']['rgb_mae_gain_vs_gray'] == pytest.approx(.1)


def test_evaluator_runs_all_batches_and_saves_summary(tmp_path, monkeypatch):
    import json
    import sys
    monkeypatch.setitem(sys.modules, 'official_cifar_resume_support', support)
    evaluator = load_tool('evaluate_official_cifar10')
    root = tmp_path/'official'
    (root/'diffusion').mkdir(parents=True)
    (root/'diffusion/diffusion.py').write_text('# test double')
    checkpoint = tmp_path/'model_100000.pt'
    torch.save({'step':100000, 'ema': {}}, checkpoint)
    monkeypatch.setattr(evaluator.subprocess, 'check_output', lambda *a, **k: evaluator.PIN)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'manual_seed_all', lambda seed: None)
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda: 'synthetic test')
    monkeypatch.setattr(torch.Tensor, 'cuda', lambda self: self)
    batches = []
    class Model(torch.nn.Module):
        def __init__(self, *a, **k):
            super().__init__()
        def cuda(self):
            return self
        def sample(self, batch_size, img, t):
            batches.append(batch_size)
            gray = img.mean(1, keepdim=True).expand_as(img)
            return {'xt': gray, 'direct_recons': img.clone(), 'recon': img.clone()}
    class Dataset:
        def __init__(self, root, train, download, transform):
            assert train is False and download is False
        def __len__(self):
            return 10000
        def __getitem__(self, i):
            return torch.tensor([1., -1., -1.]).reshape(3, 1, 1).expand(3, 2, 2).clone(), 0
    monkeypatch.setitem(sys.modules, 'diffusion', SimpleNamespace(GaussianDiffusion=Model))
    monkeypatch.setitem(sys.modules, 'diffusion.model.get_model', SimpleNamespace(get_model=lambda *a, **k: Model()))
    transforms = SimpleNamespace(Compose=lambda x: x, ToTensor=lambda: None, Normalize=lambda *a: None)
    monkeypatch.setitem(sys.modules, 'torchvision', SimpleNamespace(datasets=SimpleNamespace(CIFAR10=Dataset), transforms=transforms))
    monkeypatch.setitem(sys.modules, 'torchvision.utils', SimpleNamespace(save_image=lambda tensor, path, nrow: Path(path).write_bytes(b'test')))
    output = tmp_path/'evaluation'
    monkeypatch.setattr(sys, 'argv', ['evaluate', '--official-dir', str(root), '--checkpoint', str(checkpoint),
                                    '--output-dir', str(output), '--limit', '33'])
    evaluator.main()
    summary = json.loads((output/'summary.json').read_text())
    assert batches == [32, 1]
    assert summary['count'] == 33 and summary['full_test'] is False
    assert summary['phases']['recon']['rgb_mae'] == 0
    records = [json.loads(x) for x in (output/'preview_color_metrics.jsonl').read_text().splitlines()]
    assert records[-1]['dataset_indices'] == [32]
    assert records[0]['split'] == 'test_subset'
    assert all((output/f'{phase}.png').exists() for phase in ['og', 'xt', 'direct_recons', 'recon'])
