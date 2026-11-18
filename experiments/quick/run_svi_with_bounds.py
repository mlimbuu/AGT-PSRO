"""Run exact ammo-layer SVI and report initial-state certified value bounds.

Usage:
  python experiments/quick/run_svi_with_bounds.py --scenario five_node_two_risk
  python experiments/quick/run_svi_with_bounds.py --scenario six_node_three_risk

Writes only beneath results/data/svi/<scenario>/ammo<ammo>/.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment_config import output_path, write_json_atomic
from scenarios import build_scenario, get_scenario
from svi.layers import solve_ammo_layers
from svi.value_bounds import initial_value_bounds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--scenario',
        choices=('five_node_two_risk', 'six_node_three_risk'),
        default='five_node_two_risk',
    )
    parser.add_argument('--ammo', type=int, default=1)
    parser.add_argument('--agents', type=int, choices=(2, 3), default=2,
                        help='Number of Blue agents (default: 2)')
    parser.add_argument('--without-support', action='store_true')
    parser.add_argument('--tolerance', type=float, default=1e-9)
    parser.add_argument('--bound-sweeps', type=int, default=200)
    args = parser.parse_args()

    scenario = get_scenario(args.scenario, num_blue_agents=args.agents)
    game = build_scenario(scenario, max_ammo=args.ammo).build_game(
        with_support=not args.without_support
    )
    solved = solve_ammo_layers(
        game, tolerance=args.tolerance, max_iterations=5000
    )
    initial = game.canonical_positions(scenario.initial_positions)
    v = float(solved.values[(scenario.initial_graph, args.ammo)][initial])
    bounds = initial_value_bounds(
        game,
        solved,
        initial_graph=scenario.initial_graph,
        initial_positions=scenario.initial_positions,
        ammo=args.ammo,
        max_sweeps=args.bound_sweeps,
    )
    result = dict(
        scenario=scenario.name,
        ammo=args.ammo,
        support=not args.without_support,
        svi_value=v,
        lower_bound=bounds.lower,
        upper_bound=bounds.upper,
        bound_gap=bounds.gap,
        bound_sweeps=bounds.sweeps,
        bound_type='initial-state game value interval (pure security updates)',
    )
    filename = 'with_support_bounds.json' if not args.without_support else 'without_support_bounds.json'
    destination = output_path(
        None, 'data', 'svi', scenario.name, f'ammo{args.ammo}', filename
    )
    write_json_atomic(destination, result)
    print('SVI exact value: %.10f' % v)
    print('Certified interval: [%.10f, %.10f]' % (bounds.lower, bounds.upper))
    print('Gap: %.10f, bound sweeps: %d' % (bounds.gap, bounds.sweeps))
    print('SAVED', destination)


if __name__ == '__main__':
    main()
