"""Five-node, two risky-edge benchmark (exact legacy scenario)."""
from .config import BenchmarkConfig, TerrainEdge, RiskEdge, AttackConfiguration, SupportRelation
G0=0
G1=1
G2=2
G12=3
SCENARIO = BenchmarkConfig(
    name="five_node_two_risk",
    goal_node=4,
    num_blue_agents=2,
    initial_positions=(0,0),
    initial_graph=G1,
    terrain_edges=(
        TerrainEdge(0, 1, 10.0),
        TerrainEdge(1, 4, 10.0),
        TerrainEdge(0, 2, 10.0),
        TerrainEdge(2, 3, 10.0),
        TerrainEdge(3, 4, 10.0),
    ),
    attack_configurations=(
        AttackConfiguration(
            configuration_id=G0,
            risky_edges=(),
            label="safe graph",
            display_name="G0",
        ),
        AttackConfiguration(
            configuration_id=G1,
            risky_edges=(
                RiskEdge(1, 4, 200.0),
            ),
            label="edge {1,4} attacked",
            display_name="G1",
        ),
        AttackConfiguration(
            configuration_id=G2,
            risky_edges=(
                RiskEdge(2, 3, 200.0),
                        ),
            label="edge {2,3} attacked",
            display_name="G2",
        ),
        AttackConfiguration(
                configuration_id=G12,
                risky_edges=(
                    RiskEdge(2, 3, 200.0),
                    RiskEdge(1, 4, 200.0),
                            ),
                            
                label="edge {2, 3} & {1, 4} attacked",
                display_name="G12",
                ),

    ),
    # red_transitions={
    #     G1: (G1, G2, G12),
    #     G2: (G2, G12, G12),
    #     G12: (G1, G2, G12,),
    # },
    red_transitions={
        # G0 is not reachable from the selected initial graph in this setup,
        # but every declared attack configuration must have an action entry.
        G0: (G0,),
        G1: (G1, G2,),
        G2: (G2, G12,),
        G12: (G12,),
        },
    support_relations=(
        SupportRelation(
            support_node=3,
            risk_u=1,
            risk_v=4,
        ),
        SupportRelation(
            support_node=1,
            risk_u=2,
            risk_v=3,
        ),
    ),
    ammo_levels=(1,2,3),
    gamma=1 - 1e-9,
    # gamma=1.0,
    wait_cost=1.0,
    # active_support_wait_cost=0.01,
    active_support_wait_cost=1.0,
    repetitions=3,
    use_agent_symmetry=True,
)

