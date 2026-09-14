import importlib.util
import json
from pathlib import Path
import subprocess

import pytest

spec = importlib.util.spec_from_file_location('select_gpu_stack', Path(__file__).parents[1]/'tools/select_gpu_stack.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize('capability,expected', [
    ('8.6', ('2.5.1', '0.20.1', 'cu121', '12.1')),
    ('8.9', ('2.5.1', '0.20.1', 'cu121', '12.1')),
    ('12.0', ('2.7.1', '0.22.1', 'cu128', '12.8')),
])
def test_capability_selects_matching_torch_vision(capability, expected):
    assert module.choose_stack([capability]) == expected


def test_mixed_hardware_and_unknown_device():
    assert module.choose_stack(['8.6', '12.0'])[2] == 'cu128'
    with pytest.raises(ValueError):
        module.choose_stack(['99.0'])
    with pytest.raises(ValueError):
        module.choose_stack([])


@pytest.mark.parametrize('name,capability', [('3090', '8.6'), ('4090', '8.9'), ('5090', '12.0')])
def test_old_driver_exact_name_fallback(monkeypatch, name, capability):
    monkeypatch.setattr(module.subprocess, 'run', lambda *a, **k: subprocess.CompletedProcess(a, 1, '', 'invalid query'))
    monkeypatch.setattr(module.subprocess, 'check_output', lambda *a, **k: f'NVIDIA GeForce RTX {name}\n')
    assert module.detect_capabilities()[0] == [capability]
    monkeypatch.setattr(module.subprocess, 'check_output', lambda *a, **k: 'Unknown GPU\n')
    with pytest.raises(ValueError):
        module.detect_capabilities()


def test_cli_records_detected_environment(monkeypatch, tmp_path, capsys):
    record = tmp_path/'environment.json'
    monkeypatch.setattr(module, 'detect_capabilities', lambda: (['12.0'], 'nvidia-smi compute_cap'))
    monkeypatch.setattr('sys.argv', ['select_gpu_stack.py', '--record', str(record)])
    module.main()
    assert capsys.readouterr().out.strip() == '2.7.1 0.22.1 cu128 12.8'
    assert json.loads(record.read_text())['capabilities'] == ['12.0']
