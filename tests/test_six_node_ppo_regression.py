"""Six-node/three-risk regression against original 2026 1k PSRO terminal logs.

Run: python -m pytest tests/test_six_node_ppo_regression.py -q -m slow
These tests execute actual PPO+PSRO (not mocked evaluation or replayed values).
"""
from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
# Historical iteration-1 tuple: (initial V, Blue BR, Red BR, updated restricted V).
REFERENCE = {
    ('scratch', 11): (40.0, 787.2386, 104.9384, 104.9384),
    ('scratch', 12): (440.0, 628.3327, 200.6081, 440.0),
    ('bc', 11): (40.0, 787.2386, 66.1577, 66.1577),
    ('bc', 12): (40.0, 628.3327, 59.4926, 59.4926),
    ('heuristic', 11): (40.0, 787.2386, 66.1577, 66.1577),
    ('heuristic', 12): (40.0, 628.3327, 59.4926, 59.4926),
}

@pytest.mark.slow
@pytest.mark.parametrize('condition,seed', sorted(REFERENCE))
def test_six_node_1k_first_iteration_matches_original(condition, seed, tmp_path):
    """Also checks launching from a completely unrelated working directory."""
    command = [sys.executable, str(ROOT/'experiments/quick/run_psro.py'),
               '--scenario', 'six_node_three_risk', '--init', condition,
               '--seed', str(seed), '--steps', '1000', '--iterations', '1',
               '--threads', '1', '--skip-svi', '--force']
    proc = subprocess.run(command, cwd=tmp_path, capture_output=True,
                          text=True, timeout=100)
    assert proc.returncode == 0, proc.stdout+'\n'+proc.stderr
    directory=ROOT/'results/data/psro/six_node_three_risk/1000steps'/condition
    documents=[json.loads(p.read_text()) for p in directory.glob(f'seed{seed}-*.json')]
    matching=[item for item in documents
              if item.get('status')=='complete'
              and item.get('psro_iterations')==1
              and item.get('blue_actor')=='static_relgnn']
    assert matching, proc.stdout
    run=matching[-1]
    assert len(run['history']) == 1
    row=run['history'][0]
    actual=(run['initial_value'],row['blue_BR'],row['red_BR'],row['meta_after'])
    assert actual == pytest.approx(REFERENCE[(condition, seed)], abs=2.5e-4)
    assert not list(tmp_path.iterdir()), 'Experiment wrote files into the invoking folder'
