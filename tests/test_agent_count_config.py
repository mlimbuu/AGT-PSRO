from scenarios import get_scenario, build_scenario

def test_agent_variants():
    for agents in (2, 3):
        scenario = get_scenario('five_node_two_risk', num_blue_agents=agents)
        assert scenario.num_blue_agents == agents
        assert scenario.initial_positions == (0,) * agents
        game = build_scenario(scenario, max_ammo=1).build_game(with_support=True)
        assert game.spec.num_blue_agents == agents
        assert game.canonical_positions(scenario.initial_positions) == (0,) * agents
        assert (scenario.name.endswith('3agents')) == (agents == 3)
