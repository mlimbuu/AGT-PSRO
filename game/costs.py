from __future__ import annotations

from game.compat import dataclass, strict_zip


from typing import Mapping, Protocol

from game.models import BlueAction, Edge, GameSpec, GameState, GraphId, Node


class StageCostModel(Protocol):
    def __call__(
        self,
        state: GameState,
        blue_action: BlueAction,
        spec: GameSpec,
    ) -> float:
        ...


@dataclass(frozen=True, slots=True)
class EdgeTraversalCost:
    """James-style sum of current-graph edge traversal costs."""

    def __call__(
        self,
        state: GameState,
        blue_action: BlueAction,
        spec: GameSpec,
    ) -> float:
        costs = spec.graph_costs[state.graph_id]
        return float(
            sum(
                costs[(current, successor)]
                for current, successor in strict_zip(
                    state.blue_positions, blue_action
                )
            )
        )


@dataclass(frozen=True, slots=True)
class EdgeRiskSpec:
    """Graph-dependent additional penalties on risky directed edges."""

    risk_penalty_by_graph: Mapping[GraphId, Mapping[Edge, float]]


@dataclass(frozen=True, slots=True)
class EdgeRiskCost:
    """Base edge traversal costs plus unsupported risky-edge penalties."""

    risk: EdgeRiskSpec

    def __call__(
        self,
        state: GameState,
        blue_action: BlueAction,
        spec: GameSpec,
    ) -> float:
        if len(blue_action) != len(state.blue_positions):
            raise ValueError("Blue action dimension does not match state.")

        base_costs = spec.graph_costs[state.graph_id]
        penalties = self.risk.risk_penalty_by_graph.get(state.graph_id, {})

        return float(
            sum(
                base_costs[(current, successor)]
                + float(penalties.get((current, successor), 0.0))
                for current, successor in strict_zip(
                    state.blue_positions, blue_action
                )
            )
        )


@dataclass(frozen=True, slots=True)
class EdgeSupportSpec:
    """Support-node to risky-edge relation.

    A relation such as

        support_nodes_by_risky_edge[(u, v)] = {z1, z2}

    means that another Blue agent occupying z1 or z2 may support a teammate
    traversing directed edge (u, v). By default, the supporter must remain at
    the support node during the transition.

    `supported_penalty_multiplier=0` removes the adversarial risk penalty
    completely when support is active. A value such as 0.25 leaves 25% of it.
    """

    risk_penalty_by_graph: Mapping[GraphId, Mapping[Edge, float]]
    support_nodes_by_risky_edge: Mapping[Edge, frozenset[Node]]
    supported_penalty_multiplier: float = 0.0
    supporter_must_wait: bool = True
    idle_wait_cost: float | None = None
    active_support_wait_cost: float | None = None

    def __post_init__(self) -> None:
        if not (0.0 <= self.supported_penalty_multiplier <= 1.0):
            raise ValueError(
                "supported_penalty_multiplier must lie in [0, 1]."
            )


@dataclass(frozen=True, slots=True)
class EdgeSupportCost:
    """Risky-edge cost with state-dependent support from another agent.

    Timing used here:

    * Traverser i moves from p_i(t) to p_i(t+1).
    * Supporter j is at support node z at time t.
    * If `supporter_must_wait=True`, supporter j must also choose z at t+1.
    * One supporter may support multiple covered traversals in the same step.
    """

    support: EdgeSupportSpec

    def __call__(
        self,
        state: GameState,
        blue_action: BlueAction,
        spec: GameSpec,
    ) -> float:
        if len(blue_action) != len(state.blue_positions):
            raise ValueError("Blue action dimension does not match state.")

        base_costs = spec.graph_costs[state.graph_id]
        penalties = self.support.risk_penalty_by_graph.get(
            state.graph_id, {}
        )

        supported_agents: set[int] = set()
        active_supporters: set[int] = set()

        for traverser_i, (current, successor) in enumerate(
            strict_zip(state.blue_positions, blue_action)
        ):
            risky_edge = (current, successor)
            if risky_edge not in penalties:
                continue

            valid_support_nodes = (
                self.support.support_nodes_by_risky_edge.get(
                    risky_edge, frozenset()
                )
            )

            for supporter_j, supporter_position in enumerate(
                state.blue_positions
            ):
                if supporter_j == traverser_i:
                    continue
                if supporter_position not in valid_support_nodes:
                    continue
                if (
                    self.support.supporter_must_wait
                    and blue_action[supporter_j] != supporter_position
                ):
                    continue

                supported_agents.add(traverser_i)
                active_supporters.add(supporter_j)
                break

        total = 0.0

        for agent_i, (current, successor) in enumerate(
            strict_zip(state.blue_positions, blue_action)
        ):
            edge = (current, successor)

            if current == successor:
                # Preserve the absorbing-goal convention. The no-support model
                # uses the graph's zero-cost goal self-loop, so the support
                # model must do the same. Otherwise merely enabling support
                # adds an artificial cost to every agent already at the goal.
                if current == spec.goal_node:
                    total += base_costs[edge]
                elif (
                    agent_i in active_supporters
                    and self.support.active_support_wait_cost is not None
                ):
                    total += self.support.active_support_wait_cost
                elif self.support.idle_wait_cost is not None:
                    total += self.support.idle_wait_cost
                else:
                    total += base_costs[edge]
            else:
                total += base_costs[edge]

            penalty = float(penalties.get(edge, 0.0))
            if agent_i in supported_agents:
                penalty *= self.support.supported_penalty_multiplier
            total += penalty

        return float(total)


@dataclass(frozen=True, slots=True)
class NodeSupportSpec:
    """Optional node-node support semantics.

    risk_penalty_by_graph[g][r] is the additional cost for an agent
    whose next position is risky node r under graph configuration g.
    support_nodes_by_risk[r] contains the nodes that can support r.
    """

    risk_penalty_by_graph: Mapping[GraphId, Mapping[Node, float]]
    support_nodes_by_risk: Mapping[Node, frozenset[Node]]
    supported_risk_penalty: float = 0.0
    supporter_must_wait: bool = True
    idle_wait_cost: float | None = None
    active_support_wait_cost: float | None = None


@dataclass(frozen=True, slots=True)
class NodeSupportCost:
    """Base edge costs plus state-dependent node support."""

    support: NodeSupportSpec

    def __call__(
        self,
        state: GameState,
        blue_action: BlueAction,
        spec: GameSpec,
    ) -> float:
        if len(blue_action) != len(state.blue_positions):
            raise ValueError("Blue action dimension does not match state.")

        base_costs = spec.graph_costs[state.graph_id]
        penalties = self.support.risk_penalty_by_graph.get(state.graph_id, {})

        active_supporters: set[int] = set()
        supported_agents: set[int] = set()

        for agent_i, risky_node in enumerate(blue_action):
            if risky_node not in penalties:
                continue

            valid_support_nodes = self.support.support_nodes_by_risk.get(
                risky_node, frozenset()
            )

            for agent_j, supporter_position in enumerate(state.blue_positions):
                if agent_i == agent_j:
                    continue
                if supporter_position not in valid_support_nodes:
                    continue
                if (
                    self.support.supporter_must_wait
                    and blue_action[agent_j] != supporter_position
                ):
                    continue

                supported_agents.add(agent_i)
                active_supporters.add(agent_j)
                break

        total = 0.0

        for agent_i, (current, successor) in enumerate(
            strict_zip(state.blue_positions, blue_action)
        ):
            if current == successor:
                # Preserve the zero-cost absorbing goal exactly as in the
                # corresponding no-support game.
                if current == spec.goal_node:
                    total += base_costs[(current, successor)]
                elif (
                    agent_i in active_supporters
                    and self.support.active_support_wait_cost is not None
                ):
                    total += self.support.active_support_wait_cost
                elif self.support.idle_wait_cost is not None:
                    total += self.support.idle_wait_cost
                else:
                    total += base_costs[(current, successor)]
            else:
                total += base_costs[(current, successor)]

            if successor in penalties:
                if agent_i in supported_agents:
                    total += self.support.supported_risk_penalty
                else:
                    total += float(penalties[successor])

        return float(total)
