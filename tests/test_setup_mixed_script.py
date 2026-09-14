"""Check new-instance setup orchestration without installing/downloading anything."""
import os
from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.parametrize('mounted', [False, True])
def test_setup_orders_environment_and_data_without_changing_split(tmp_path, mounted):
    root = tmp_path / 'project with spaces'
    (root/'scripts').mkdir(parents=True)
    (root/'.venv/bin').mkdir(parents=True)
    bin_dir = tmp_path/'bin'
    bin_dir.mkdir()
    git = bin_dir/'git'
    git.write_text('#!/bin/sh\nexit 0\n')
    git.chmod(0o755)
    source = Path(__file__).parents[1]/'scripts/setup_mixed.sh'
    shutil.copy(source, root/'scripts/setup_mixed.sh')
    (root/'scripts/setup_cuda.sh').write_text('echo environment >> "$CALL_LOG"\n')
    python = root/'.venv/bin/python'
    python.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$CALL_LOG"\n')
    python.chmod(0o755)
    log = tmp_path/'calls'
    env = {**os.environ, 'PATH': str(bin_dir)+os.pathsep+os.environ['PATH'], 'CALL_LOG': str(log)}
    for key in ('UIEB_REFERENCE_DIR','UIEB_DATA_ROOT','DIV2K_DATA_ROOT','MIXED_DATA_ROOT'):
        env.pop(key, None)
    if mounted:
        env['UIEB_REFERENCE_DIR'] = '/mounted/UIEB/reference-890'
    subprocess.run(['bash', str(root/'scripts/setup_mixed.sh')], env=env, check=True, capture_output=True)
    calls = log.read_text().splitlines()
    assert calls[0] == 'environment'
    assert ('tools/prepare_uieb.py' in '\n'.join(calls)) != mounted
    if not mounted:
        assert '--output ' + str(root/'data/UIEB/download_split.json') in calls[1]
    assert calls[-2].startswith('tools/prepare_div2k.py')
    assert calls[-1].startswith('tools/prepare_mixed_colorization.py')
    assert '--split-file splits/uieb_seed42.json' in calls[-1]
    assert not any('train.py' in call for call in calls)
