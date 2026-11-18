"""Compare final PSRO restricted values by Blue actor/critic pairing.

Scientific caveat: this plots restricted-game values, not exploitability.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from experiment_config import RESULTS_ROOT, output_path

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--scenario',default='five_node_two_risk')
    p.add_argument('--steps',type=int,default=10000)
    p.add_argument('--init',default='scratch')
    args=p.parse_args()
    directory=RESULTS_ROOT/'data'/'psro'/args.scenario/f'{args.steps}steps'/args.init
    runs={}
    for path in sorted(directory.glob('seed*.json')):
        obj=json.loads(path.read_text())
        if obj.get('status')!='complete': continue
        key=(obj['blue_actor'],obj['blue_critic'])
        runs.setdefault(key,{})[obj['seed']]=float(obj['final'])
    if not runs: raise SystemExit(f'No completed ablation results found in {directory}')
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels=[];means=[];sds=[]
    for key,vals in sorted(runs.items()):
        z=np.array(list(vals.values()),dtype=float)
        labels.append(' / '.join(key));means.append(float(z.mean()))
        sds.append(float(z.std(ddof=1)) if len(z)>1 else 0.0)
    fig,ax=plt.subplots(figsize=(8,4))
    ax.bar(range(len(labels)),means,yerr=sds,capsize=4)
    ax.set_xticks(range(len(labels)),labels,rotation=22,ha='right')
    ax.set_ylabel('Final restricted-game value')
    ax.set_title(f'{args.scenario}: Blue actor / critic, {args.steps} PPO steps')
    fig.tight_layout()
    target=output_path(None,'figures',args.scenario,f'{args.steps}steps','ppo_ablations.png')
    fig.savefig(target,dpi=180)
    print('SAVED',target)
if __name__=='__main__':main()
