"""Mathematical regression tests for verified small games."""
import pytest
from game.models import GameState, PruningMode
from scenarios import get_scenario, build_scenario
from svi.layers import solve_ammo_layers
from psro.initialization import random_proper_blue_seed
from psro.psro import make_stay_red_policy, evaluate_policy_pair_exact
from psro.meta_game_matrix import build_empirical_payoff_matrix, solve_meta_game

def test_five_node_svi():
    scenario=get_scenario('five_node_two_risk')
    game=build_scenario(scenario,max_ammo=1).build_game(with_support=True)
    res=solve_ammo_layers(game,pruning_mode=PruningMode.NONE,tolerance=1e-9,max_iterations=5000)
    initial=game.canonical_positions(scenario.initial_positions)
    assert res.values[(scenario.initial_graph,1)][initial] == pytest.approx(52.0,abs=1e-5)

def test_six_node_svi():
    scenario=get_scenario('six_node_three_risk')
    game=build_scenario(scenario,max_ammo=1).build_game(with_support=True)
    res=solve_ammo_layers(game,pruning_mode=PruningMode.NONE,tolerance=1e-9,max_iterations=5000)
    initial=game.canonical_positions(scenario.initial_positions)
    assert res.values[(scenario.initial_graph,1)][initial] == pytest.approx(48.94427177,abs=1e-5)

def test_ammo_and_red_legal_actions():
    scenario=get_scenario('five_node_two_risk')
    game=build_scenario(scenario,max_ammo=1).build_game(with_support=True)
    s=GameState((0,0),1,1)
    assert game.red_actions(s)==(1,2)
    a=game.blue_actions(s)[0]
    assert game.transition(s,a,2).red_ammo==0
    assert game.transition(s,a,1).red_ammo==1
    assert game.red_actions(GameState((0,0),2,0))==(2,)

def test_payoff_matrix_matches_direct_value():
    scenario=get_scenario('five_node_two_risk')
    game=build_scenario(scenario,max_ammo=1).build_game(with_support=True)
    blue=random_proper_blue_seed(game,11)
    red=make_stay_red_policy(game)
    kwargs=dict(initial_graph=scenario.initial_graph,initial_ammo=1,initial_positions=scenario.initial_positions)
    exact=evaluate_policy_pair_exact(game,blue,red,**kwargs)
    matrix=build_empirical_payoff_matrix(game,[blue],[red],**kwargs)
    assert matrix.shape==(1,1)
    assert matrix[0,0]==pytest.approx(exact,abs=1e-7)
    assert solve_meta_game(matrix).value==pytest.approx(exact,abs=1e-7)
    assert exact==pytest.approx(440.0,abs=1e-5)


def test_support_eliminates_matching_risk_only_when_supporter_waits():
    scenario=get_scenario('five_node_two_risk')
    supported=build_scenario(scenario,max_ammo=1).build_game(with_support=True)
    no_support=build_scenario(scenario,max_ammo=1).build_game(with_support=False)
    # Blue robot 0 traverses attacked {1,4}; teammate waits at support node 3.
    state=GameState((1,3),1,1)
    action=(4,3)
    assert action in supported.blue_actions(state)
    c1=supported.stage_cost(state,action)
    c0=no_support.stage_cost(state,action)
    assert c0-c1==pytest.approx(200.0,abs=1e-6)
    # Supporter leaves node 3, so that same risky crossing is not protected.
    departure=(4,2)
    assert departure in supported.blue_actions(state)
    assert supported.stage_cost(state,departure)==pytest.approx(no_support.stage_cost(state,departure))
