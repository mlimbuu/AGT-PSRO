"""Six-node, three risky-edge benchmark, from the user-supplied 2026-11-15 configuration."""
from .config import BenchmarkConfig, TerrainEdge, RiskEdge, AttackConfiguration, SupportRelation
G0,G1,G2,G3=0,1,2,3
SCENARIO=BenchmarkConfig(
    name="six_node_three_risk", goal_node=5, num_blue_agents=2,
    initial_positions=(0,0), initial_graph=G1,
    terrain_edges=(TerrainEdge(0,1,10.0),TerrainEdge(1,5,10.0),
      TerrainEdge(0,2,10.0),TerrainEdge(2,3,10.0),TerrainEdge(3,5,10.0),
      TerrainEdge(0,4,10.0),TerrainEdge(4,5,10.0)),
    attack_configurations=(
      AttackConfiguration(G0, (), "safe graph", "G0"),
      AttackConfiguration(G1, (RiskEdge(1,5,200.0),), "edge {1,5} attacked", "G1"),
      AttackConfiguration(G2, (RiskEdge(3,5,25.0),), "edge {3,5} attacked", "G2"),
      AttackConfiguration(G3, (RiskEdge(4,5,50.0),), "edge {4,5} attacked", "G3"),
    ),
    red_transitions={G0:(G0,),G1:(G1,G2,G3),G2:(G2,),G3:(G3,)},
    support_relations=(SupportRelation(4,1,5),), ammo_levels=(1,),
    gamma=1-1e-9, wait_cost=1.0, active_support_wait_cost=1.0,
    repetitions=3, use_agent_symmetry=True,
)
