"""Scenario specification and builder (ported unchanged from main_run_SB.py)."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence
from game.graph_factory import AdversarialGraphBuilder
from game.models import PruningMode

@dataclass(frozen=True)
class TerrainEdge:
    u: int
    v: int
    cost: float


@dataclass(frozen=True)
class RiskEdge:
    u: int
    v: int
    penalty: float


@dataclass(frozen=True)
class AttackConfiguration:
    """One complete snapshot of the risky edges active in a Red graph.

    A Red transition selects the entire next snapshot. One transition may
    therefore activate one edge, several edges, or no new edges.
    """

    configuration_id: int
    risky_edges: tuple[RiskEdge, ...] = ()
    label: str = ""
    display_name: str = ""


@dataclass(frozen=True)
class SupportRelation:
    support_node: int
    risk_u: int
    risk_v: int


@dataclass(frozen=True)
class SolverConfig:
    tolerance: float = 1e-8
    bound_tolerance: float = 1e-8
    max_iterations: int = 5_000


@dataclass(frozen=True)
class BenchmarkConfig:
    name: str
    goal_node: int
    num_blue_agents: int
    initial_positions: tuple[int, ...]
    initial_graph: int

    terrain_edges: tuple[TerrainEdge, ...]
    attack_configurations: tuple[AttackConfiguration, ...]
    red_transitions: Mapping[int, tuple[int, ...]]

    ammo_levels: tuple[int, ...]
    support_relations: tuple[SupportRelation, ...] = ()

    gamma: float = 0.9
    wait_cost: float = 1.0
    active_support_wait_cost: float = 0.01
    use_agent_symmetry: bool = False

    support_options: tuple[bool, ...] = (False, True)
    pruning_modes: tuple[PruningMode, ...] = tuple(PruningMode)

    repetitions: int = 3
    solver: SolverConfig = field(default_factory=SolverConfig)

    # Exhaustive matched-game preflight. This checks that support does not
    # alter actions/transitions or increase any immediate matching cost.
    enable_diagnostics: bool = True
    diagnostic_tolerance: float = 1e-10
    fail_on_diagnostic_violation: bool = True

    def validate(self) -> None:
        if self.num_blue_agents <= 0:
            raise ValueError("num_blue_agents must be positive.")

        if len(self.initial_positions) != self.num_blue_agents:
            raise ValueError(
                "initial_positions must contain exactly "
                f"{self.num_blue_agents} entries."
            )

        if not self.ammo_levels:
            raise ValueError("ammo_levels cannot be empty.")

        if any(ammo < 0 for ammo in self.ammo_levels):
            raise ValueError("ammo levels must be nonnegative.")

        if self.repetitions <= 0:
            raise ValueError("repetitions must be positive.")

        configuration_ids = [
            configuration.configuration_id
            for configuration in self.attack_configurations
        ]
        graph_ids = set(configuration_ids)

        if len(configuration_ids) != len(graph_ids):
            raise ValueError("Attack configuration IDs must be unique.")

        if self.initial_graph not in graph_ids:
            raise ValueError(
                f"initial_graph={self.initial_graph} is not present "
                "in attack_configurations."
            )

        transition_keys = set(self.red_transitions)
        if transition_keys != graph_ids:
            missing = graph_ids - transition_keys
            extra = transition_keys - graph_ids
            raise ValueError(
                "red_transitions must define every graph exactly once. "
                f"Missing={sorted(missing)}, extra={sorted(extra)}."
            )

        for current_graph, next_graphs in self.red_transitions.items():
            if not next_graphs:
                raise ValueError(
                    f"Graph {current_graph} must have at least one successor."
                )
            unknown = set(next_graphs) - graph_ids
            if unknown:
                raise ValueError(
                    f"Graph {current_graph} transitions to unknown graphs "
                    f"{sorted(unknown)}."
                )
            if current_graph not in next_graphs:
                raise ValueError(
                    f"Graph {current_graph} must include its self-loop in "
                    "red_transitions."
                )

        nodes = {
            node
            for edge in self.terrain_edges
            for node in (edge.u, edge.v)
        }

        if self.goal_node not in nodes:
            raise ValueError("goal_node is not present in terrain_edges.")

        missing_positions = set(self.initial_positions) - nodes
        if missing_positions:
            raise ValueError(
                f"Initial positions not in graph: {sorted(missing_positions)}"
            )

        terrain_edge_keys = {
            frozenset((edge.u, edge.v)) for edge in self.terrain_edges
        }
        risky_edge_keys = {
            frozenset((edge.u, edge.v))
            for configuration in self.attack_configurations
            for edge in configuration.risky_edges
        }

        for configuration in self.attack_configurations:
            for edge in configuration.risky_edges:
                if frozenset((edge.u, edge.v)) not in terrain_edge_keys:
                    raise ValueError(
                        f"Risk edge {(edge.u, edge.v)} in graph "
                        f"{configuration.configuration_id} is not a terrain edge."
                    )

        for relation in self.support_relations:
            if relation.support_node not in nodes:
                raise ValueError(
                    f"Support node {relation.support_node} is not in graph."
                )

            relation_edge = frozenset((relation.risk_u, relation.risk_v))
            if relation_edge not in terrain_edge_keys:
                raise ValueError(
                    "Support relation references a non-terrain edge: "
                    f"{(relation.risk_u, relation.risk_v)}."
                )
            if relation_edge not in risky_edge_keys:
                raise ValueError(
                    "Support relation references an edge that is never risky: "
                    f"{(relation.risk_u, relation.risk_v)}."
                )

        # Deliberately do not require every risky edge to have a support
        # relation. A risky edge without one always receives its full penalty.


# ---------------------------------------------------------------------
# Scenario construction
# ---------------------------------------------------------------------

def build_scenario(config: BenchmarkConfig, *, max_ammo: int):
    """
    Construct one scenario for the requested maximum Red ammo level.
    """
    config.validate()

    builder = AdversarialGraphBuilder(
        goal_node=config.goal_node,
        num_blue_agents=config.num_blue_agents,
        max_ammo=max_ammo,
        gamma=config.gamma,
        wait_cost=config.wait_cost,
        active_support_wait_cost=config.active_support_wait_cost,
        indistinguishable_blue_agents=config.use_agent_symmetry,
    )

    builder.add_undirected_edges(
        [(edge.u, edge.v, edge.cost) for edge in config.terrain_edges]
    )

    for attack_configuration in config.attack_configurations:
        risk_mapping = {
            (edge.u, edge.v): edge.penalty
            for edge in attack_configuration.risky_edges
        }
        builder.add_undirected_attack_configuration(
            attack_configuration.configuration_id,
            risk_mapping,
        )

    # Red's action is a graph-to-graph transition. The next graph is a
    # complete risk snapshot, not an individual edge toggle. Every non-self
    # transition consumes one ammo unit in environment.transition().
    for current_graph, next_graphs in config.red_transitions.items():
        for next_graph in next_graphs:
            builder.allow_red_transition(current_graph, next_graph)

    for relation in config.support_relations:
        builder.add_undirected_support_relation(
            relation.support_node,
            relation.risk_u,
            relation.risk_v,
        )

    return builder.build()


