"""Opt-in W&B logging: use a stub to test hooks without credentials or network.

This verifies the W&B call contract and recorded metrics; it does not claim
that the real SDK successfully uploaded any online run.
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiment_config import PROJECT_ROOT, RESULTS_ROOT, output_path


class FakeRun:
    id = 'offline-test-id'
    def __init__(self):
        self.logged = []
        self.summary = {}
        self.finished = False
    def log(self, data):
        self.logged.append(dict(data))
    def finish(self):
        self.finished = True


class FakeWandb:
    def __init__(self):
        self.runs = []
        self.init_kwargs = []
        self.metrics = []
    def init(self, **kwargs):
        self.init_kwargs.append(kwargs)
        run = FakeRun()
        self.runs.append(run)
        return run
    def define_metric(self, *args, **kwargs):
        self.metrics.append((args, kwargs))


def test_wandb_disabled_does_not_need_sdk(monkeypatch):
    # --wandb-mode disabled must be equivalent to running without W&B.
    monkeypatch.setitem(sys.modules, 'wandb', None)
    from experiment_config import psro_cli
    monkeypatch.setattr(sys, 'argv', ['run_psro.py', '--scenario','five_node_two_risk',
        '--steps','8','--iterations','1','--init','scratch','--seed','11',
        '--skip-svi','--wandb','--wandb-mode','disabled','--force'])
    psro_cli(preset='quick')


def test_offline_wandb_metrics_and_directories(monkeypatch):
    fake = FakeWandb()
    monkeypatch.setitem(sys.modules, 'wandb', fake)
    from experiment_config import psro_cli
    monkeypatch.setattr(sys, 'argv', ['run_psro.py', '--scenario','five_node_two_risk',
        '--steps','16','--iterations','1','--init','scratch','--seed','11',
        '--skip-svi','--wandb','--wandb-mode','offline','--wandb-project','test-project',
        '--force'])
    psro_cli(preset='quick')
    assert len(fake.runs) == 1
    assert fake.runs[0].finished
    init = fake.init_kwargs[0]
    assert init['mode']=='offline'
    assert init['project']=='test-project'
    assert init['config']['scenario']=='five_node_two_risk'
    assert init['job_type']=='quick-psro'
    assert Path(init['dir']).resolve().is_relative_to(RESULTS_ROOT.resolve())
    for name in ('WANDB_DIR','WANDB_CACHE_DIR','WANDB_DATA_DIR'):
        assert Path(os.environ[name]).resolve().is_relative_to(RESULTS_ROOT.resolve())
    all_logs=fake.runs[0].logged
    assert any(d.get('psro/iteration')==0 for d in all_logs)
    assert any(d.get('psro/iteration')==1 and 'psro/meta_value' in d for d in all_logs)
    assert any('ppo/blue/global_steps' in d for d in all_logs)
    assert any('ppo/red/global_steps' in d for d in all_logs)
    docs=[json.loads(x.read_text()) for x in (RESULTS_ROOT/'data/psro/five_node_two_risk/16steps/scratch').glob('*.json')]
    assert any(x.get('status')=='complete' and x.get('wandb',{}).get('run_id')=='offline-test-id' for x in docs)


def test_shell_wrappers_have_valid_paths():
    for name in ('run_quick_psro.sh','run_long_psro.sh','run_ppo_ablations.sh','run_quick_svi.sh'):
        sh=PROJECT_ROOT/'scripts'/name
        assert sh.is_file()
        assert sh.stat().st_mode & 0o111


def test_requested_wandb_gives_install_instructions_when_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, 'wandb', None)
    from experiment_config import psro_cli
    monkeypatch.setattr(sys, 'argv', ['run_psro.py', '--scenario', 'five_node_two_risk',
        '--steps', '8', '--iterations', '1', '--init', 'scratch', '--seed', '11',
        '--skip-svi', '--wandb', '--wandb-mode', 'offline', '--force'])
    with pytest.raises(RuntimeError, match='requirements-wandb.txt'):
        psro_cli(preset='quick')
