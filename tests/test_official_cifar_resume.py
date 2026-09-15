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
            self.opt.step()
            if self.step % self.update_ema_every == 0:
                self.step_ema()
            if self.step != 0 and self.step % self.save_and_sample_every == 0:
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
    namespace = {}
    exec('\n'.join(first.splitlines()[2:]), namespace)
    t = namespace['Trainer']()
    t.step, t.train_num_steps = 10000, 10003
    t.update_ema_every, t.save_and_sample_every = 10, 1
    updates, saved, ema = [], [], []
    t.opt = SimpleNamespace(step=lambda: updates.append(1))
    t.save = lambda save_with_time_stamp=False: saved.append((t.step, len(updates)))
    t.step_ema = lambda: ema.append(t.step)
    t.train()
    assert len(updates) == 3
    assert saved == [(10001,1), (10002,2), (10003,3), (10003,3), (10003,3)]
    assert ema == [10000]
    assert 'save_with_time_stamp_every=10000' in (tmp_path/'train.py').read_text()
