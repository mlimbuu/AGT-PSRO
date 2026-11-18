"""Architecture ablations use one PSRO engine, not duplicated PPO training code.

Usage:
 python experiments/long/run_ppo_ablations.py --scenario five_node_two_risk --steps 1000 --iterations 1 --seeds 11

All JSON is saved by run_ppo_psro_experiment under results/data/psro.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from dataclasses import asdict
import json
import os
from scenarios import get_scenario
from experiment_config import (RESULTS_ROOT, safe_results_root, output_path,
    base_ppo_config, config_fingerprint)
from psro.psro import run_ppo_psro_experiment

COMBINATIONS=(
    ('mlp','mlp'),
    ('static_relgnn','mlp'),
    ('action_relgnn','mlp'),
    ('static_relgnn','static_relgnn'),
)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--scenario',choices=('five_node_two_risk','six_node_three_risk'),default='five_node_two_risk')
    p.add_argument('--init',choices=('scratch','bc','heuristic'),default='scratch')
    p.add_argument('--seeds',type=int,nargs='+',default=[11,12,13])
    p.add_argument('--steps',type=int,default=10000)
    p.add_argument('--iterations',type=int,default=15)
    p.add_argument('--ammo',type=int,default=1)
    p.add_argument('--threads',type=int,default=1)
    p.add_argument('--output-root',type=Path,default=RESULTS_ROOT)
    p.add_argument('--skip-svi',action='store_true')
    p.add_argument('--force',action='store_true')
    p.add_argument('--wandb',action='store_true')
    p.add_argument('--wandb-project',default=os.getenv('WANDB_PROJECT', 'agt5-psro'))
    p.add_argument('--wandb-entity',default=os.getenv('WANDB_ENTITY') or None)
    p.add_argument('--wandb-mode',choices=('online','offline','disabled'),default=os.getenv('WANDB_MODE', 'online'))
    p.add_argument('--wandb-group',default=None)
    args=p.parse_args()
    args.preset='ppo_ablation'
    safe_results_root(args.output_root)
    scenario=get_scenario(args.scenario)
    for actor,critic in COMBINATIONS:
        args.blue_actor, args.blue_critic=actor,critic
        cfg={**asdict(base_ppo_config()),'total_steps':args.steps}
        for seed in args.seeds:
            h=config_fingerprint(scenario,condition=args.init,seed=seed,
                steps=args.steps,iterations=args.iterations,
                blue_actor=actor,blue_critic=critic,ppo_config=cfg,ammo=args.ammo)
            target=output_path(args.output_root,'data','psro',scenario.name,
                f'{args.steps}steps',args.init,f'seed{seed}-{h}.json')
            if target.exists() and not args.force:
                try:
                    old=json.loads(target.read_text())
                    if old.get('status')=='complete' and old.get('config_fingerprint')==h:
                        print('SKIP',actor,critic,seed)
                        continue
                except (OSError, ValueError): pass
            print('ARCHITECTURE',actor,critic,'SEED',seed,flush=True)
            run_ppo_psro_experiment(args.init,seed,args)
if __name__=='__main__':main()
