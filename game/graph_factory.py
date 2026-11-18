from __future__ import annotations

from game.compat import dataclass

from dataclasses import field
from typing import Iterable, Mapping

from game.costs import (
    EdgeRiskCost,
    EdgeRiskSpec,
    EdgeSupportCost,
    EdgeSupportSpec,
)
from game.environment import AdversarialGraphGame
from game.models import Edge, GameSpec, GraphId, Node


@dataclass(frozen=True, slots=True)
class GraphScenario:
    """One reusable graph scenario for both experimental conditions.

    The base topology, base movement costs, Red configurations, and Red action
    graph remain identical. `build_game(with_support=False)` creates the
    no-support condition. `build_game(with_support=True)` activates the
    support-node/risky-edge relation.
    """

    nodes: tuple[Node, ...]
    base_edge_costs: Mapping[Edge, float]
    goal_node: Node
    risk_penalty_by_graph: Mapping[GraphId, Mapping[Edge, float]]
    red_action_graph: Mapping[GraphId, tuple[GraphId, ...]]
    support_nodes_by_risky_edge: Mapping[Edge, frozenset[Node]]
    num_blue_agents: int
    max_ammo: int
    gamma: float = 0.99
    allow_wait: bool = True
    wait_cost: float = 0.1
    goal_wait_cost: float = 0.0
    allow_collisions: bool = True
    indistinguishable_blue_agents: bool = False
    supporter_must_wait: bool = True
    supported_penalty_multiplier: float = 0.0
    active_support_wait_cost: float | None = 0.01

    def adjacency(self) -> dict[Node, tuple[Node, ...]]:
        outgoing: dict[Node, set[Node]] = {
            node: set() for node in self.nodes
        }

        for current, successor in self.base_edge_costs:
            if current not in outgoing or successor not in outgoing:
                raise ValueError(
                    f"Edge {(current, successor)} uses an unknown node."
                )
            outgoing[current].add(successor)

        if self.allow_wait:
            for node in self.nodes:
                outgoing[node].add(node)

        # The terminal self-loop is required even when ordinary waiting is
        # disabled. This matches the absorbing-goal convention used by the
        # MATLAB reference implementation and by GameSpec validation.
        outgoing[self.goal_node].add(self.goal_node)

        return {
            node: tuple(sorted(successors))
            for node, successors in outgoing.items()
        }

    def complete_base_costs(self) -> dict[Edge, float]:
        costs = {
            edge: float(value)
            for edge, value in self.base_edge_costs.items()
        }

        if self.allow_wait:
            for node in self.nodes:
                costs.setdefault((node, node), float(self.wait_cost))

        costs[(self.goal_node, self.goal_node)] = float(
            self.goal_wait_cost
        )
        return costs

    def build_spec(self) -> GameSpec:
        graph_ids = tuple(self.red_action_graph)
        if set(graph_ids) != set(self.risk_penalty_by_graph):
            raise ValueError(
                "risk_penalty_by_graph and red_action_graph must use "
                "the same graph IDs."
            )

        base_costs = self.complete_base_costs()
        graph_costs = {
            graph_id: dict(base_costs) for graph_id in graph_ids
        }

        return GameSpec(
            nodes=self.nodes,
            adjacency=self.adjacency(),
            graph_costs=graph_costs,
            red_action_graph=self.red_action_graph,
            goal_node=self.goal_node,
            num_blue_agents=self.num_blue_agents,
            max_ammo=self.max_ammo,
            gamma=self.gamma,
            allow_collisions=self.allow_collisions,
            indistinguishable_blue_agents=(
                self.indistinguishable_blue_agents
            ),
        )

    def validate_relations(self) -> None:
        valid_edges = set(self.complete_base_costs())
        valid_nodes = set(self.nodes)

        if (
            self.active_support_wait_cost is not None
            and self.active_support_wait_cost > self.wait_cost + 1e-12
        ):
            raise ValueError(
                "active_support_wait_cost cannot exceed wait_cost when "
                "support is intended to be optional and cost-reducing. "
                f"Got active_support_wait_cost={self.active_support_wait_cost} "
                f"and wait_cost={self.wait_cost}."
            )

        for graph_id, penalties in self.risk_penalty_by_graph.items():
            for edge, penalty in penalties.items():
                if edge not in valid_edges:
                    raise ValueError(
                        f"Risk edge {edge} in graph {graph_id} is not "
                        "part of the topology."
                    )
                if penalty < 0.0:
                    raise ValueError("Risk penalties must be nonnegative.")

        for risky_edge, support_nodes in (
            self.support_nodes_by_risky_edge.items()
        ):
            if risky_edge not in valid_edges:
                raise ValueError(
                    f"Support relation references unknown edge {risky_edge}."
                )
            invalid_nodes = set(support_nodes) - valid_nodes
            if invalid_nodes:
                raise ValueError(
                    f"Support relation for {risky_edge} references unknown "
                    f"nodes {invalid_nodes}."
                )

    def build_game(
        self,
        *,
        with_support: bool,
    ) -> AdversarialGraphGame:
        self.validate_relations()
        spec = self.build_spec()

        if with_support:
            if self.num_blue_agents < 2:
                raise ValueError(
                    "Support requires at least two Blue agents."
                )
            cost_model = EdgeSupportCost(
                EdgeSupportSpec(
                    risk_penalty_by_graph=self.risk_penalty_by_graph,
                    support_nodes_by_risky_edge=(
                        self.support_nodes_by_risky_edge
                    ),
                    supported_penalty_multiplier=(
                        self.supported_penalty_multiplier
                    ),
                    supporter_must_wait=self.supporter_must_wait,
                    idle_wait_cost=self.wait_cost,
                    active_support_wait_cost=(
                        self.active_support_wait_cost
                    ),
                )
            )
        else:
            cost_model = EdgeRiskCost(
                EdgeRiskSpec(
                    risk_penalty_by_graph=self.risk_penalty_by_graph
                )
            )

        return AdversarialGraphGame(
            spec=spec,
            cost_model=cost_model,
        )

    def build_comparison_pair(
        self,
    ) -> tuple[AdversarialGraphGame, AdversarialGraphGame]:
        """Return `(no_support_game, support_game)`."""

        return (
            self.build_game(with_support=False),
            self.build_game(with_support=True),
        )


@dataclass(slots=True)
class AdversarialGraphBuilder:
    """Mutable convenience builder for a `GraphScenario`."""

    goal_node: Node
    num_blue_agents: int
    max_ammo: int
    gamma: float = 0.99
    allow_wait: bool = True
    wait_cost: float = 0.1
    goal_wait_cost: float = 0.0
    allow_collisions: bool = True
    indistinguishable_blue_agents: bool = False
    supporter_must_wait: bool = True
    supported_penalty_multiplier: float = 0.0
    active_support_wait_cost: float | None = 0.01
    _nodes: set[Node] = field(default_factory=set, init=False)
    _base_edge_costs: dict[Edge, float] = field(
        default_factory=dict, init=False
    )
    _risk_penalty_by_graph: dict[GraphId, dict[Edge, float]] = field(
        default_factory=dict, init=False
    )
    _red_action_graph: dict[GraphId, set[GraphId]] = field(
        default_factory=dict, init=False
    )
    _support_nodes_by_risky_edge: dict[Edge, set[Node]] = field(
        default_factory=dict, init=False
    )

    def add_node(self, node: Node) -> AdversarialGraphBuilder:
        self._nodes.add(node)
        return self

    def add_edge(
        self,
        current: Node,
        successor: Node,
        base_cost: float,
        *,
        bidirectional: bool = False,
    ) -> AdversarialGraphBuilder:
        if base_cost < 0.0:
            raise ValueError("Base movement cost must be nonnegative.")

        self._nodes.update((current, successor))
        self._base_edge_costs[(current, successor)] = float(base_cost)

        if bidirectional:
            self._base_edge_costs[(successor, current)] = float(base_cost)

        return self

    def add_directed_edge(
        self,
        current: Node,
        successor: Node,
        base_cost: float,
    ) -> AdversarialGraphBuilder:
        """Add one directed arc ``(current, successor)``."""

        return self.add_edge(
            current,
            successor,
            base_cost,
            bidirectional=False,
        )

    def add_undirected_edge(
        self,
        node_i: Node,
        node_j: Node,
        base_cost: float,
    ) -> AdversarialGraphBuilder:
        """Represent ``{i, j}`` by equal-cost arcs in both directions.

        Mathematically, the physical undirected edge is stored internally as
        ``(i, j)`` and ``(j, i)``. This preserves the directed-game notation
        while allowing Blue to traverse or backtrack in either direction.
        """

        return self.add_edge(
            node_i,
            node_j,
            base_cost,
            bidirectional=True,
        )

    def add_edges(
        self,
        edges: Iterable[tuple[Node, Node, float]],
    ) -> AdversarialGraphBuilder:
        """Add a collection of directed arcs."""

        for current, successor, base_cost in edges:
            self.add_directed_edge(current, successor, base_cost)
        return self

    def add_undirected_edges(
        self,
        edges: Iterable[tuple[Node, Node, float]],
    ) -> AdversarialGraphBuilder:
        """Add undirected physical edges as symmetric directed arcs."""

        for node_i, node_j, base_cost in edges:
            self.add_undirected_edge(node_i, node_j, base_cost)
        return self

    def add_attack_configuration(
        self,
        graph_id: GraphId,
        risky_edge_penalties: Mapping[Edge, float] | None = None,
    ) -> AdversarialGraphBuilder:
        penalties = {
            edge: float(value)
            for edge, value in (risky_edge_penalties or {}).items()
        }
        if any(value < 0.0 for value in penalties.values()):
            raise ValueError("Risk penalties must be nonnegative.")

        self._risk_penalty_by_graph[graph_id] = penalties
        self._red_action_graph.setdefault(graph_id, set()).add(graph_id)
        return self

    def add_undirected_attack_configuration(
        self,
        graph_id: GraphId,
        risky_edge_penalties: Mapping[Edge, float] | None = None,
    ) -> AdversarialGraphBuilder:
        """Add a Red configuration whose physical risky edges are undirected.

        Each supplied pair ``(i, j): penalty`` is expanded to both
        ``(i, j): penalty`` and ``(j, i): penalty``. Therefore risk applies
        identically in both traversal directions.
        """

        expanded: dict[Edge, float] = {}
        for (node_i, node_j), value in (
            risky_edge_penalties or {}
        ).items():
            penalty = float(value)
            if penalty < 0.0:
                raise ValueError("Risk penalties must be nonnegative.")
            expanded[(node_i, node_j)] = penalty
            expanded[(node_j, node_i)] = penalty

        return self.add_attack_configuration(graph_id, expanded)

    def allow_red_transition(
        self,
        current_graph: GraphId,
        next_graph: GraphId,
        *,
        bidirectional: bool = False,
    ) -> AdversarialGraphBuilder:
        self._red_action_graph.setdefault(current_graph, set()).add(
            next_graph
        )
        if bidirectional:
            self._red_action_graph.setdefault(next_graph, set()).add(
                current_graph
            )
        return self

    def make_red_action_graph_complete(
        self,
    ) -> AdversarialGraphBuilder:
        graph_ids = set(self._risk_penalty_by_graph)
        for current in graph_ids:
            self._red_action_graph.setdefault(current, set()).update(
                graph_ids
            )
        return self

    def add_support_relation(
        self,
        support_node: Node,
        risky_edge: Edge,
    ) -> AdversarialGraphBuilder:
        """Map a support node to one directed risky arc."""

        self._nodes.add(support_node)
        self._support_nodes_by_risky_edge.setdefault(
            risky_edge, set()
        ).add(support_node)
        return self

    def add_undirected_support_relation(
        self,
        support_node: Node,
        node_i: Node,
        node_j: Node,
    ) -> AdversarialGraphBuilder:
        """Apply one support relation to both directions of an edge.

        The support node covers both ``i -> j`` and ``j -> i``.
        """

        self.add_support_relation(support_node, (node_i, node_j))
        self.add_support_relation(support_node, (node_j, node_i))
        return self

    def build(self) -> GraphScenario:
        self._nodes.add(self.goal_node)

        graph_ids = set(self._risk_penalty_by_graph)
        if not graph_ids:
            raise ValueError(
                "Add at least one attack configuration before build()."
            )

        if set(self._red_action_graph) != graph_ids:
            missing = graph_ids - set(self._red_action_graph)
            extra = set(self._red_action_graph) - graph_ids
            raise ValueError(
                "Red action graph/configuration mismatch. "
                f"Missing={missing}, extra={extra}."
            )

        for current, successors in self._red_action_graph.items():
            invalid = set(successors) - graph_ids
            if invalid:
                raise ValueError(
                    f"Graph {current} transitions to unknown IDs {invalid}."
                )
            successors.add(current)

        scenario = GraphScenario(
            nodes=tuple(sorted(self._nodes)),
            base_edge_costs=dict(self._base_edge_costs),
            goal_node=self.goal_node,
            risk_penalty_by_graph={
                graph_id: dict(penalties)
                for graph_id, penalties in (
                    self._risk_penalty_by_graph.items()
                )
            },
            red_action_graph={
                graph_id: tuple(sorted(successors))
                for graph_id, successors in self._red_action_graph.items()
            },
            support_nodes_by_risky_edge={
                edge: frozenset(nodes)
                for edge, nodes in (
                    self._support_nodes_by_risky_edge.items()
                )
            },
            num_blue_agents=self.num_blue_agents,
            max_ammo=self.max_ammo,
            gamma=self.gamma,
            allow_wait=self.allow_wait,
            wait_cost=self.wait_cost,
            goal_wait_cost=self.goal_wait_cost,
            allow_collisions=self.allow_collisions,
            indistinguishable_blue_agents=(
                self.indistinguishable_blue_agents
            ),
            supporter_must_wait=self.supporter_must_wait,
            supported_penalty_multiplier=(
                self.supported_penalty_multiplier
            ),
            active_support_wait_cost=self.active_support_wait_cost,
        )
        scenario.validate_relations()
        scenario.build_spec()
        return scenario


def make_edge_support_demo_scenario(
    *,
    gamma: float = 0.95,
) -> GraphScenario:
    """Small symmetric-edge scenario used in examples and tests.

    The physical environment is undirected but stored as a directed graph:

        0 -- 1 -- 4(goal)
        |         /
        2 -------
        |
        3 -------- 4(goal)

    Red configuration G1 attacks physical edge {1, 4}, so both directed
    arcs (1, 4) and (4, 1) receive the same risk penalty. Node 3 supports
    that physical risky edge in both traversal directions. G0 has no risk.
    """

    builder = AdversarialGraphBuilder(
        goal_node=4,
        num_blue_agents=2,
        max_ammo=1,
        gamma=gamma,
        wait_cost=1.0,
        active_support_wait_cost=0.01,
    )

    builder.add_undirected_edges(
        [
            (0, 1, 1.0),
            (1, 4, 1.0),
            (0, 2, 1.5),
            (2, 4, 1.5),
            (0, 3, 1.0),
            (3, 4, 2.0),
        ]
    )
    builder.add_undirected_attack_configuration(0, {})
    builder.add_undirected_attack_configuration(
        1,
        {(1, 4): 10.0},
    )
    builder.make_red_action_graph_complete()
    builder.add_undirected_support_relation(3, 1, 4)
    return builder.build()
