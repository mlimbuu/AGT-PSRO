"""Optional 1k/3-iteration regression against archived five-node PSRO logs.

Run with: python -m pytest tests/test_ppo_regression.py -q -m slow
Each parameter executes a genuine Blue+Red PPO PSRO run, not mocked or replayed data.
"""
from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path
import pytest

REFERENCE={
    'scratch':[(963.1574,286.3253,440.0),
               (1046.6999,319.3398,440.0),
               (1030.3609,286.2113,440.0)],
    'bc':[(963.1574,52.3858,52.3858),
          (573.7650,52.1221,52.3858),
          (952.3229,52.2905,52.3858)],
    'heuristic':[(963.1574,52.3858,52.3858),
                 (573.7650,52.1221,52.3858),
                 (952.3229,52.2905,52.3858)],
}

@pytest.mark.slow
@pytest.mark.parametrize('condition',('scratch','bc','heuristic'))
def test_five_node_1k_three_iterations_against_archived_log(condition):
    root=Path(__file__).resolve().parents[1]
    command=[sys.executable,str(root/'experiments/quick/run_psro.py'),
             '--scenario','five_node_two_risk','--init',condition,
             '--seed','11','--steps','1000','--iterations','3',
             '--threads','1','--skip-svi','--force']
    cp=subprocess.run(command,cwd=root,capture_output=True,text=True,timeout=180)
    assert cp.returncode==0,cp.stdout+'\n'+cp.stderr
    results=root/'results/data/psro/five_node_two_risk/1000steps'/condition
    docs=[json.loads(p.read_text()) for p in results.glob('seed11-*.json')]
    matching=[x for x in docs if x.get('status')=='complete' and
              x.get('psro_iterations')==3 and x.get('blue_actor')=='static_relgnn']
    assert matching,cp.stdout
    doc=matching[-1]
    assert len(doc['history'])==3
    for step,expected in zip(doc['history'],REFERENCE[condition]):
        assert (step['blue_BR'],step['red_BR'],step['meta_after'])==pytest.approx(expected,abs=2e-4)
