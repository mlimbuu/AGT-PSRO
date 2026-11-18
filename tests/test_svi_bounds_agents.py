"""Regression: agent-count configuration and certified initial-state SVI bounds."""
import pytest
from scenarios import build_scenario, get_scenario
from svi.layers import solve_ammo_layers
from svi.value_bounds import initial_value_bounds

@pytest.mark.parametrize('agents,expected', [(2,52.0),(3,73.0)])
def test_five_node_svi_bounds(agents,expected):
    scenario=get_scenario('five_node_two_risk',num_blue_agents=agents)
    game=build_scenario(scenario,max_ammo=1).build_game(with_support=True)
    solved=solve_ammo_layers(game,tolerance=1e-9,max_iterations=5000)
    positions=game.canonical_positions(scenario.initial_positions)
    value=float(solved.values[(scenario.initial_graph,1)][positions])
    b=initial_value_bounds(game,solved,initial_graph=scenario.initial_graph,
        initial_positions=scenario.initial_positions,ammo=1)
    assert abs(value-expected)<1e-4
    assert b.lower-1e-5<=value<=b.upper+1e-5
    assert b.gap>=-1e-6
