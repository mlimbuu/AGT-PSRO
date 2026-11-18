"""Summarize 2-vs-3 agent SVI/PSRO experiments; outputs stay under results/."""
from __future__ import annotations
from pathlib import Path
import json
import csv

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'results'/'data'
OUT=DATA/'agent_comparison'
OUT.mkdir(parents=True,exist_ok=True)

def select_run(n):
    directory=DATA/'psro'/('five_node_two_risk' if n==2 else f'five_node_two_risk_{n}agents')/'1000steps'/'scratch'
    files=sorted(directory.glob('seed11-*.json'))
    runs=[json.loads(p.read_text()) for p in files]
    runs=[r for r in runs if r.get('status')=='complete']
    if not runs: return None
    return max(runs,key=lambda r:len(r.get('history',[])))

rows=[]
for n in (2,3):
    scenario='five_node_two_risk' if n==2 else f'five_node_two_risk_{n}agents'
    svi=json.loads((DATA/'svi'/scenario/'ammo1'/'with_support.json').read_text())['exact_svi']
    run=select_run(n)
    if run is None: raise RuntimeError(f'No complete PSRO output for {n} agents')
    for entry in run['history']:
        rows.append({'agents':n,'iteration':entry['iteration'],'svi_value':svi,
                     'meta_value':entry['meta_after'],'abs_gap_to_svi':abs(entry['meta_after']-svi),
                     'blue_BR':entry['blue_BR'],'red_BR':entry['red_BR'],
                     'seed':run['seed'],'ppo_steps_per_oracle':run['ppo_steps_per_oracle']})
path=OUT/'five_node_2vs3_agents_1k_scratch.csv'
with path.open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(f'SAVED {path}')
for n in (2,3):
    subset=[r for r in rows if r['agents']==n]
    print(f'{n} agents: SVI={subset[-1]["svi_value"]:.4f}, iterations={len(subset)}, '
          f'PSRO={subset[-1]["meta_value"]:.4f}, abs_gap={subset[-1]["abs_gap_to_svi"]:.4f}')
