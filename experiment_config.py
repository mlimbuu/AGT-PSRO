"""Experiment settings and output-path safety. Generated files are restricted to results/."""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import os
import tempfile
PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = PROJECT_ROOT / 'results'

def safe_results_root(path=None):
    root = (Path(path).expanduser() if path else RESULTS_ROOT).resolve()
    results = RESULTS_ROOT.resolve()
    if root != results and results not in root.parents:
        raise ValueError(f"Output must be within {results}; received {root}")
    root.mkdir(parents=True,exist_ok=True)
    return root

def output_path(results_root, *parts):
    root=safe_results_root(results_root)
    dest=root.joinpath(*map(str,parts)).resolve()
    if root != dest and root not in dest.parents:
        raise ValueError('Output path escapes results/')
    dest.parent.mkdir(parents=True,exist_ok=True)
    return dest

def source_revision():
    """Hash all scientific sources that can affect a run, preventing stale skips after edits."""
    files = [
        *sorted((PROJECT_ROOT / 'game').glob('*.py')),
        *sorted((PROJECT_ROOT / 'svi').glob('*.py')),
        *sorted((PROJECT_ROOT / 'scenarios').glob('*.py')),
        *sorted((PROJECT_ROOT / 'psro').rglob('*.py')),
        PROJECT_ROOT / 'experiment_config.py',
    ]
    h=hashlib.sha256()
    for file in files:
        h.update(str(file.relative_to(PROJECT_ROOT)).encode())
        h.update(file.read_bytes())
    return h.hexdigest()[:12]

def config_fingerprint(scenario, *, condition, seed, steps, iterations,
                       blue_actor, blue_critic, ppo_config, support=True, ammo=1):
    from dataclasses import asdict
    payload={"scenario":asdict(scenario), "condition":condition,
             "seed":seed,"steps":steps,"iterations":iterations,
             "blue_actor":blue_actor,"blue_critic":blue_critic,
             "ppo_config":ppo_config,"support":support,"ammo":ammo,
             "source_revision":source_revision()}
    encoded=json.dumps(payload,sort_keys=True,default=str,separators=(',',':'))
    return hashlib.sha256(encoded.encode()).hexdigest()[:12]

def write_json_atomic(dest, payload):
    """Safely write a result JSON. Reject paths outside this project's results/.

    A unique temporary file avoids collisions between concurrent cloud jobs;
    the final rename is atomic on a single filesystem.
    """
    target = Path(dest).expanduser().resolve()
    results = RESULTS_ROOT.resolve()
    if target == results or results not in target.parents:
        raise ValueError(f"JSON output must be inside {results}; received {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=target.parent,
            prefix=target.name + ".", suffix=".tmp", delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(payload, handle, indent=2, default=str)
        os.replace(temporary_path, target)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

def base_ppo_config():
    from psro.ppo_oracle.red_ppo import PPOConfig
    return PPOConfig(total_steps=6000, rollout_steps=512, max_episode_steps=32,
        hidden_size=64, learning_rate=3e-4, update_epochs=4,
        minibatch_size=128, entropy_coef=0.01, entropy_coef_final=None,
        reward_scale=0.01,include_graph_feature=True,
        include_ammo_feature=True,include_red_reachability_feature=True)


def psro_cli(*, preset='quick'):
    """Common CLI for quick and long modes; both use an identical PSRO implementation."""
    import argparse
    from psro.psro import run_ppo_psro_experiment
    from scenarios import get_scenario
    parser=argparse.ArgumentParser(description=f'{preset} PPO PSRO experiment')
    parser.add_argument('--scenario',choices=('five_node_two_risk','six_node_three_risk'),default='five_node_two_risk')
    parser.add_argument('--ammo',type=int,default=1)
    parser.add_argument('--agents',type=int,choices=(2,3),default=2,help='Number of Blue agents (2 or 3)')
    parser.add_argument('--init',choices=('scratch','bc','heuristic','all'),default='scratch' if preset=='quick' else 'all')
    parser.add_argument('--seed',type=int,default=None)
    parser.add_argument('--seeds',nargs='+',type=int,default=[11] if preset=='quick' else [11,12,13])
    parser.add_argument('--steps',type=int,default=100 if preset=='quick' else 10000)
    parser.add_argument('--iterations',type=int,default=1 if preset=='quick' else 15)
    parser.add_argument('--threads',type=int,default=1)
    parser.add_argument('--blue-actor',choices=('mlp','static_relgnn','action_relgnn','static_relgnn_virtual'),default='static_relgnn')
    parser.add_argument('--blue-critic',choices=('mlp','static_relgnn'),default='mlp')
    parser.add_argument('--output-root',type=Path,default=RESULTS_ROOT)
    parser.add_argument('--skip-svi',action='store_true')
    parser.add_argument('--force',action='store_true')
    parser.add_argument('--wandb',action='store_true')
    parser.add_argument('--wandb-project',default=os.getenv('WANDB_PROJECT', 'agt5-psro'))
    parser.add_argument('--wandb-entity',default=os.getenv('WANDB_ENTITY') or None)
    parser.add_argument('--wandb-mode',choices=('online','offline','disabled'),default=os.getenv('WANDB_MODE', 'online'))
    parser.add_argument('--wandb-group', default=None, help='Optional W&B group; default groups by scenario, budget and Blue architecture')
    args=parser.parse_args()
    args.preset = preset
    safe_results_root(args.output_root)
    if min(args.steps,args.iterations,args.threads,args.ammo)<=0:
        parser.error('--steps, --iterations, --threads, --ammo must all be positive')
    scenario=get_scenario(args.scenario, num_blue_agents=args.agents)
    from dataclasses import asdict
    config={**asdict(base_ppo_config()),'total_steps':args.steps}
    conditions=('scratch','bc','heuristic') if args.init=='all' else (args.init,)
    seeds=(args.seed,) if args.seed is not None else args.seeds
    for condition in conditions:
        for seed in seeds:
            h=config_fingerprint(scenario,condition=condition,seed=seed,steps=args.steps,
                iterations=args.iterations,blue_actor=args.blue_actor,
                blue_critic=args.blue_critic,ppo_config=config,ammo=args.ammo)
            dest=output_path(args.output_root,'data','psro',scenario.name,
                f'{args.steps}steps',condition,f'seed{seed}-{h}.json')
            if dest.exists() and not args.force:
                try:
                    old=json.loads(dest.read_text())
                    if old.get('status')=='complete' and old.get('config_fingerprint')==h:
                        print('SKIP',condition,seed,'already complete')
                        continue
                except (ValueError,OSError):
                    pass
            run_ppo_psro_experiment(condition,seed,args)
