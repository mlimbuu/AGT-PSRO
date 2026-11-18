"""Named graph configurations: no mutable SMALL_SCENARIO global."""
from .config import BenchmarkConfig, build_scenario
from .five_node_two_risk import SCENARIO as FIVE_NODE_TWO_RISK
from .six_node_three_risk import SCENARIO as SIX_NODE_THREE_RISK
SCENARIOS = {
    "five_node_two_risk": FIVE_NODE_TWO_RISK,
    "six_node_three_risk": SIX_NODE_THREE_RISK,
}
def get_scenario(name: str, num_blue_agents=None) -> BenchmarkConfig:
    try:
        scenario = SCENARIOS[name]
        if num_blue_agents is None or num_blue_agents == scenario.num_blue_agents:
            return scenario
        if num_blue_agents < 1:
            raise ValueError('num_blue_agents must be positive')
        from dataclasses import replace
        return replace(scenario,
            name=f'{scenario.name}_{num_blue_agents}agents',
            num_blue_agents=num_blue_agents,
            initial_positions=(scenario.initial_positions[0],) * num_blue_agents)
    except KeyError as error:
        raise ValueError(f"Unknown scenario {name!r}; choose {tuple(SCENARIOS)}") from error
