"""Plot restricted-game trajectories from results/data/psro. Writes figures to results/figures."""
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
    p.add_argument('--steps',type=int,default=1000)
    args=p.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    directory=RESULTS_ROOT/'data'/'psro'/args.scenario/f'{args.steps}steps'
    if not directory.exists(): raise SystemExit(f'No PSRO results under {directory}')
    fig,ax=plt.subplots(figsize=(7,4))
    palette={'scratch':'#2563EB','bc':'#E1A000','heuristic':'#059669'}
    for condition in ['scratch','bc','heuristic']:
        files=sorted((directory/condition).glob('seed*.json'))
        series=[]
        for path in files:
            run=json.loads(path.read_text())
            if run.get('status')=='complete' and run.get('history'):
                series.append([v['meta_after'] for v in run['history']])
        if not series: continue
        import numpy as np
        n=min(map(len,series))
        values=np.array([seq[:n] for seq in series],dtype=float)
        x=np.arange(1,n+1)
        ax.plot(x,values.mean(axis=0),label=condition,color=palette[condition])
        if len(series)>1:
            sd=values.std(axis=0,ddof=1)
            ax.fill_between(x,values.mean(axis=0)-sd,values.mean(axis=0)+sd,color=palette[condition],alpha=.12)
    ax.set(xlabel='PSRO iteration',ylabel='Restricted-game value',title=f'{args.scenario}: {args.steps} PPO steps')
    ax.legend();fig.tight_layout()
    out=output_path(None,'figures',args.scenario,f'{args.steps}steps','psro.png')
    fig.savefig(out,dpi=180);print('SAVED',out)
if __name__=='__main__':main()
