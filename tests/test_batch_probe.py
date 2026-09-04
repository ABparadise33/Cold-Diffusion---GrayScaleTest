import importlib.util
import json
import os
from pathlib import Path
import subprocess

import pytest
import torch
import yaml

from gray_cold_diffusion import batch_probe

ROOT = Path(__file__).resolve().parents[1]
GIB = 1024**3


def good_result(peak=12):
    return {'status': 'ok', 'total_bytes': 24 * GIB, 'initial_free_bytes': 23 * GIB,
            'peak_reserved_bytes': peak * GIB, 'final_free_bytes': (23 - peak) * GIB}


def test_candidate_order_keeps_effective_batch():
    assert batch_probe.candidate_batches(32) == [32, 16, 8, 4, 2, 1]
    assert all(32 % b == 0 for b in batch_probe.candidate_batches(32))


def test_oom_and_memory_margin_are_the_only_retry_cases():
    assert batch_probe.classify_result(42, {'status': 'cuda_oom'}) == 'cuda_oom'
    assert batch_probe.classify_result(0, good_result(22)) == 'insufficient_headroom'
    assert batch_probe.classify_result(0, good_result(12)) == 'pass'
    other_gpu_user = good_result(12)
    other_gpu_user['final_free_bytes'] = GIB
    assert batch_probe.classify_result(0, other_gpu_user) == 'insufficient_headroom'
    with pytest.raises(RuntimeError, match='non-OOM'):
        batch_probe.classify_result(-9, {})  # killed worker is NOT assumed to be CUDA OOM
    with pytest.raises(RuntimeError, match='non-OOM'):
        batch_probe.classify_result(1, {'status': 'error', 'error': 'shape mismatch'})
    with pytest.raises(RuntimeError, match='non-OOM'):
        batch_probe.classify_result(42, {})


def test_selection_falls_back_then_stops_at_largest_safe_divisor(tmp_path):
    batches = []

    def fake_runner(config, batch, effective, directory):
        batches.append(batch)
        if batch == 32:
            return 42, {'status': 'cuda_oom'}
        return 0, good_result(22 if batch == 16 else 12)

    result = batch_probe.select_batch('config.yaml', 32, tmp_path, runner=fake_runner)
    assert batches == [32, 16, 8]
    assert result['selected'] == {'batch_size': 8, 'grad_accum': 4}
    assert [x['status'] for x in result['attempts']] == ['cuda_oom', 'insufficient_headroom', 'pass']
    assert json.loads((tmp_path / 'batch_probe.json').read_text()) == result


def test_non_oom_and_timeout_abort_without_trying_smaller_batches(tmp_path):
    for error in ('shape', 'timeout'):
        directory = tmp_path / error
        directory.mkdir()
        batches = []

        def runner(config, batch, effective, attempt):
            batches.append(batch)
            if error == 'timeout':
                raise subprocess.TimeoutExpired('worker', 300)
            return 1, {'error': 'shape mismatch'}

        with pytest.raises((RuntimeError, subprocess.TimeoutExpired)):
            batch_probe.select_batch('config.yaml', 32, directory, runner=runner)
        assert batches == [32]
        assert json.loads((directory / 'batch_probe.json').read_text())['status'] == 'stopped_on_error'


def test_no_safe_batch_writes_failure_and_does_not_select(tmp_path):
    with pytest.raises(RuntimeError, match='training was NOT started'):
        batch_probe.select_batch('config.yaml', 32, tmp_path,
                                 runner=lambda *args: (42, {'status': 'cuda_oom'}))
    report = json.loads((tmp_path / 'batch_probe.json').read_text())
    assert len(report['attempts']) == 6
    assert report['status'] == 'no_safe_batch' and report['selected'] is None


def test_probe_runs_adam_ema_accumulation_and_validation_on_cpu():
    config = yaml.safe_load((ROOT / 'configs/div2k_official_rgb_sat1_50k.yaml').read_text())
    config['model'].update(dim=8, dim_mults=[1, 2])
    config['data']['image_size'] = 16
    config['diffusion']['steps'] = 2
    result = batch_probe.probe_workload(config, 1, 2, 'cpu', updates=3)
    assert result['status'] == 'ok'
    assert result['grad_accum'] == 2 and result['updates'] == 3
    assert result['seconds_per_update_after_warmup'] > 0
    assert result['checks'] == ['training_forward_backward', 'adam_states', 'ema', 'full_gray_crop_validation']


def test_child_protocol_uses_a_separate_process_and_log(tmp_path, monkeypatch):
    def fake_run(command, stdout, stderr, timeout):
        assert command[1:4] == ['-u', '-m', 'gray_cold_diffusion.batch_probe']
        assert timeout == 300
        path = Path(command[command.index('--result') + 1])
        path.write_text(json.dumps({'status': 'cuda_oom'}))
        stdout.write('CUDA OOM fixture, not real GPU\n')
        return subprocess.CompletedProcess(command, 42)

    monkeypatch.setattr(subprocess, 'run', fake_run)
    code, result = batch_probe.run_child('config.yaml', 32, 32, tmp_path / 'attempt')
    assert code == 42 and result['status'] == 'cuda_oom'
    assert 'fixture' in (tmp_path / 'attempt/worker.log').read_text()


def test_worker_typed_cuda_oom_only(monkeypatch, tmp_path):
    cfg = tmp_path / 'config.yaml'
    cfg.write_text('{}')
    output = tmp_path / 'result.json'
    monkeypatch.setattr(batch_probe.sys, 'argv', ['probe', '--config', str(cfg), '--batch-size', '32',
                                               '--effective-batch', '32', '--result', str(output)])
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)

    def oom(*args):
        raise torch.cuda.OutOfMemoryError('simulated CUDA OOM')

    monkeypatch.setattr(batch_probe, 'probe_workload', oom)
    assert batch_probe.main() == 42
    assert json.loads(output.read_text())['status'] == 'cuda_oom'

    def bug(*args):
        raise RuntimeError('tensor shape bug')

    monkeypatch.setattr(batch_probe, 'probe_workload', bug)
    with pytest.raises(RuntimeError, match='shape bug'):
        batch_probe.main()


def test_auto_script_routes_correctly_and_rejects_manual_conflict():
    env = {**os.environ, 'OFFICIAL_DRY_RUN': '1'}
    for key in ('OFFICIAL_BATCH_SIZE', 'OFFICIAL_GRAD_ACCUM'):
        env.pop(key, None)
    result = subprocess.run(['bash', 'scripts/train_official_div2k_4090.sh', '--auto-batch'],
                            cwd=ROOT, env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert 'tools/autotune_official_batch.py' in result.stdout
    assert '--batch-size' not in result.stdout and '--grad-accum' not in result.stdout
    conflict = subprocess.run(['bash', 'scripts/train_official_div2k_4090.sh', '--auto-batch'],
                              cwd=ROOT, env={**env, 'OFFICIAL_BATCH_SIZE': '4'}, capture_output=True)
    assert conflict.returncode == 2


def test_launcher_only_starts_training_after_selection_and_preserves_resume(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('autotune_tool', ROOT / 'tools/autotune_official_batch.py')
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    monkeypatch.setattr(tool, 'ROOT', tmp_path)
    config = yaml.safe_load((ROOT / 'configs/div2k_official_rgb_sat1_50k.yaml').read_text())
    config['output_dir'] = str(tmp_path / 'run')
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(config))
    train_dir, val_dir = tmp_path / 'train', tmp_path / 'val'
    train_dir.mkdir()
    val_dir.mkdir()
    checkpoint = tmp_path / 'run/checkpoints/latest.pt'
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b'not a real checkpoint; launching is mocked')
    calls = []

    def choose(*args):
        return {'selected': {'batch_size': 8, 'grad_accum': 4}}

    monkeypatch.setattr(tool, 'select_batch', choose)
    monkeypatch.setattr(tool.subprocess, 'call', lambda command: calls.append(command) or 0)
    args = ['--config', str(path), '--train-dir', str(train_dir), '--val-dir', str(val_dir),
            '--device', 'cuda', '--resume', '--max-steps', '100000']
    assert tool.launch(args) == 0
    assert len(calls) == 1
    assert calls[0][-4:] == ['--batch-size', '8', '--grad-accum', '4']
    assert '--resume' in calls[0] and '100000' in calls[0]
    assert checkpoint.read_bytes().startswith(b'not a real')

    def stop(*args):
        raise RuntimeError('all probes failed')

    monkeypatch.setattr(tool, 'select_batch', stop)
    with pytest.raises(RuntimeError, match='all probes'):
        tool.launch(args)
    assert len(calls) == 1  # failed selection cannot launch real training
