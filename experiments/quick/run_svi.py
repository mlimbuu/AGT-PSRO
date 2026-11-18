"""Exact SVI on a named scenario. Writes only into results/data/svi/."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from scenarios import get_scenario, build_scenario
from svi.layers import solve_ammo_layers
from game.models import PruningMode
from experiment_config import output_path, write_json_atomic

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--scenario',choices=('five_node_two_risk','six_node_three_risk'),default='five_node_two_risk')
    parser.add_argument('--ammo',type=int,default=1)
    parser.add_argument('--agents',type=int,choices=(2,3),default=2)
    parser.add_argument('--without-support',action='store_true')
    parser.add_argument('--tolerance',type=float,default=1e-9)
    args=parser.parse_args()
    scenario=get_scenario(args.scenario, num_blue_agents=args.agents)
    game=build_scenario(scenario,max_ammo=args.ammo).build_game(with_support=not args.without_support)
    solution=solve_ammo_layers(game,pruning_mode=PruningMode.NONE,
                              tolerance=args.tolerance,max_iterations=5000)
    initial=game.canonical_positions(scenario.initial_positions)
    value=float(solution.values[(scenario.initial_graph,args.ammo)][initial])
    destination=output_path(None,'data','svi',scenario.name,f'ammo{args.ammo}',
          'with_support.json' if not args.without_support else 'without_support.json')
    write_json_atomic(destination,{'scenario':scenario.name,'ammo':args.ammo,
          'support':not args.without_support,'exact_svi':value,
          'tolerance':args.tolerance})
    print(f'SVI {scenario.name} ammo={args.ammo} support={not args.without_support} V*={value:.10f}')
    print('SAVED',destination)
    return value
if __name__=='__main__': main()
