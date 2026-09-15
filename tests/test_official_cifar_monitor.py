import importlib.util
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import numpy as np
import torch


def load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1]/'tools'/f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


monitor = load('official_cifar_monitor')


def test_loss_accumulation_and_partial_window(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(monitor, 'fixed_monitor', lambda t: calls.append(t.step))
    t = SimpleNamespace(step=998, results_folder=tmp_path, gradient_accumulate_every=2,
                        opt=SimpleNamespace(param_groups=[{'lr':2e-5}]))
    monitor.initialize(t)
    t.step = 999
    monitor.record_update(t, [2., 4.])
    t.step = 1000
    monitor.record_update(t, [4., 6.])
    assert calls == [998, 1000]
    records = [json.loads(x) for x in (tmp_path/'train_loss_steps.jsonl').read_text().splitlines()]
    assert [x['loss'] for x in records] == [3., 5.]
    import csv
    with (tmp_path/'train_loss_windows.csv').open() as handle:
        row = next(csv.DictReader(handle))
    assert float(row['mean']) == 4 and float(row['std']) == 1
    assert int(row['count']) == 2 and int(row['start_step']) == 999
    assert t._monitoring_state['loss_window'] == []


def test_rng_and_mixed_modes_restored_on_failure():
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout())
    model.train()
    model[0].eval()
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    expected = (random.random(), np.random.rand(), torch.rand(1))
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    try:
        with monitor.isolated_rng(model):
            assert not model.training
            random.random()
            np.random.rand()
            torch.rand(1)
            raise RuntimeError('simulated failure')
    except RuntimeError:
        pass
    actual = (random.random(), np.random.rand(), torch.rand(1))
    assert actual[:2] == expected[:2] and torch.equal(actual[2], expected[2])
    assert model.training and not model[0].training and model[1].training


def test_fixed_monitor_stable_subset_and_summary(tmp_path, monkeypatch):
    support = load('official_cifar_resume_support')
    monkeypatch.setitem(sys.modules, 'codex_cifar_resume', support)
    targets = list(range(10))*1000
    selected = monitor.choose_indices(targets)
    assert len(selected) == len(set(selected)) == 1000
    assert np.bincount([targets[i] for i in selected]).tolist() == [100]*10
    assert selected == monitor.choose_indices(targets)
    class Dataset:
        def __init__(self, root, train, download, transform):
            assert train is False and download is False
            self.targets = targets
        def __getitem__(self, i):
            return torch.tensor([1., 0., 0.]).reshape(3, 1, 1), targets[i]
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
        def sample(self, batch_size, img, t):
            assert not self.training and not torch.is_grad_enabled()
            return {'xt': img.mean(1, keepdim=True).expand_as(img), 'direct_recons': img, 'recon': img}
    monkeypatch.setitem(sys.modules, 'torchvision', SimpleNamespace(
        datasets=SimpleNamespace(CIFAR10=Dataset), transforms=SimpleNamespace(ToTensor=lambda: None)))
    monkeypatch.setitem(sys.modules, 'torchvision.utils', SimpleNamespace(
        save_image=lambda tensor, path, nrow: Path(path).write_bytes(b'test')))
    trainer = SimpleNamespace(step=100000, results_folder=tmp_path, ds=SimpleNamespace(root='unused'),
                              ema_model=Model())
    rng = torch.get_rng_state().clone()
    monitor.initialize(trainer)
    trainer.step = 101000
    monitor.fixed_monitor(trainer)
    assert torch.equal(rng, torch.get_rng_state())
    summaries = [json.loads(x) for x in (tmp_path/'fixed_monitor/summary.jsonl').read_text().splitlines()]
    assert [x['step'] for x in summaries] == [100000, 101000]
    assert summaries[0]['phases'] == summaries[1]['phases']
    assert summaries[0]['phases']['recon']['rgb_mae'] == 0
    events = [json.loads(x) for x in (tmp_path/'fixed_monitor/preview_color_metrics.jsonl').read_text().splitlines()]
    assert [i for e in events[:32] for i in e['dataset_indices']] == selected
    assert events[31]['batch_size'] == 8
    assert trainer._monitoring_state['last_monitor_step'] == 101000
    assert (tmp_path/'fixed_monitor/step_101000/recon.png').exists()
