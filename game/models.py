from __future__ import annotations

from game.compat import dataclass


from enum import Enum
from math import isfinite
from typing import Mapping
try:
    from typing import TypeAlias  # Python 3.10+
except ImportError:
    TypeAlias = object()  # Used only as a type annotation; no runtime semantics.


import numpy as np

Node: TypeAlias = int
GraphId: TypeAlias = int
Edge: TypeAlias = tuple[Node, Node]
BlueAction: TypeAlias = tuple[Node, ...]
RedAction: TypeAlias = GraphId


class PruningMode(str, Enum):
    """Action-pruning mode used by the exact solvers."""

    NONE = "none"
    TEMPORARY_STRICT = "temporary_strict"
    BOUND_CERTIFIED = "bound_certified"
    HYBRID = "hybrid"

    @property
    def uses_temporary_pruning(self) -> bool:
        return self in {
            PruningMode.TEMPORARY_STRICT,
            PruningMode.HYBRID,
        }

    @property
    def uses_bound_pruning(self) -> bool:
        return self in {
            PruningMode.BOUND_CERTIFIED,
            PruningMode.HYBRID,
        }


@dataclass(slots=True)
class PruningStats:
    """Aggregate computational statistics for one solver run.

    Temporary-pruning removals are counted every time a current matrix is
    solved. Bound-certified removals are counted once when the permanent
    per-state masks are constructed.
    """

    matrix_games_solved: int = 0
    original_matrix_entries: int = 0
    solved_matrix_entries: int = 0
    bound_blue_actions_removed: int = 0
    bound_red_actions_removed: int = 0
    temporary_blue_actions_removed: int = 0
    temporary_red_actions_removed: int = 0
    bound_sweeps: int = 0
    bound_state_updates: int = 0

    @property
    def matrix_entry_reduction(self) -> int:
        return self.original_matrix_entries - self.solved_matrix_entries

    @property
    def matrix_entry_reduction_fraction(self) -> float:
        if self.original_matrix_entries == 0:
            return 0.0
        return self.matrix_entry_reduction / self.original_matrix_entries

    def merge(self, other: "PruningStats") -> None:
        for field_name in self.__dataclass_fields__:
            setattr(
                self,
                field_name,
                getattr(self, field_name) + getattr(other, field_name),
            )


@dataclass(frozen=True, slots=True)
class GameState:
    """Complete Markov state of the full game."""

    blue_positions: tuple[Node, ...]
    graph_id: GraphId
    red_ammo: int


@dataclass(frozen=True, slots=True)
class GameSpec:
    """Static definition of the adversarial graph-traversal game."""

    nodes: tuple[Node, ...]
    adjacency: Mapping[Node, tuple[Node, ...]]
    graph_costs: Mapping[GraphId, Mapping[Edge, float]]
    red_action_graph: Mapping[GraphId, tuple[GraphId, ...]]
    goal_node: Node
    num_blue_agents: int
    max_ammo: int
    gamma: float = 0.99
    allow_collisions: bool = True
    indistinguishable_blue_agents: bool = False
    require_red_self_loops: bool = True
    require_zero_cost_goal_loop: bool = True

    def __post_init__(self) -> None:
        if not self.nodes:
            raise ValueError("nodes must not be empty.")
        if len(set(self.nodes)) != len(self.nodes):
            raise ValueError("nodes must be unique.")
        if self.goal_node not in self.nodes:
            raise ValueError("goal_node must belong to nodes.")
        if self.num_blue_agents <= 0:
            raise ValueError("num_blue_agents must be positive.")
        if self.max_ammo < 0:
            raise ValueError("max_ammo must be nonnegative.")
        if not (0.0 < self.gamma <= 1.0):
            raise ValueError("gamma must satisfy 0 < gamma <= 1.")

        node_set = set(self.nodes)
        for node in self.nodes:
            if node not in self.adjacency:
                raise ValueError(f"Missing adjacency entry for node {node}.")
            successors = self.adjacency[node]
            if not successors:
                raise ValueError(f"Node {node} has no available action.")
            invalid = set(successors) - node_set
            if invalid:
                raise ValueError(
                    f"Adjacency for node {node} contains invalid nodes: {invalid}."
                )

        if self.goal_node not in self.adjacency[self.goal_node]:
            raise ValueError("The goal node must have a self-loop.")

        graph_ids = set(self.graph_costs)
        if not graph_ids:
            raise ValueError("At least one graph configuration is required.")
        if set(self.red_action_graph) != graph_ids:
            raise ValueError(
                "red_action_graph keys must exactly match graph_costs keys."
            )

        required_edges = {
            (node, successor)
            for node, successors in self.adjacency.items()
            for successor in successors
        }

        for graph_id, costs in self.graph_costs.items():
            missing = required_edges - set(costs)
            if missing:
                raise ValueError(
                    f"Graph {graph_id} is missing edge costs: {sorted(missing)}."
                )
            for edge in required_edges:
                value = float(costs[edge])
                if not isfinite(value):
                    raise ValueError(
                        f"Graph {graph_id}, edge {edge} has nonfinite cost."
                    )

            if self.require_zero_cost_goal_loop:
                goal_cost = float(costs[(self.goal_node, self.goal_node)])
                if abs(goal_cost) > 1e-12:
                    raise ValueError(
                        f"Goal self-loop must cost zero; got {goal_cost} "
                        f"in graph {graph_id}."
                    )

        for graph_id, successors in self.red_action_graph.items():
            if not successors:
                raise ValueError(
                    f"Red action graph node {graph_id} has no outgoing action."
                )
            invalid = set(successors) - graph_ids
            if invalid:
                raise ValueError(
                    f"Red graph {graph_id} has invalid successors: {invalid}."
                )
            if self.require_red_self_loops and graph_id not in successors:
                raise ValueError(f"Red graph {graph_id} must have a self-loop.")


@dataclass(slots=True)
class MatrixGameSolution:
    """Solution of one zero-sum matrix game.

    Q has Red actions as rows and Blue actions as columns.
    Red maximizes the cost; Blue minimizes it.
    """

    value: float
    blue_policy: np.ndarray
    red_policy: np.ndarray
    blue_value_upper: float
    red_value_lower: float
    duality_gap: float
    blue_security_value: float
    red_security_value: float
    strategy_gap: float


@dataclass(slots=True)
class EquilibriumPolicy:
    """Equilibrium action distribution stored for one state."""

    value: float
    blue_actions: tuple[BlueAction, ...]
    red_actions: tuple[RedAction, ...]
    blue_probabilities: np.ndarray
    red_probabilities: np.ndarray


@dataclass(slots=True)
class ValueIterationResult:
    values: dict[GameState, float]
    policies: dict[GameState, EquilibriumPolicy]
    converged: bool
    iterations: int
    residual: float
    states: tuple[GameState, ...]
    pruning_mode: PruningMode = PruningMode.NONE
    pruning_stats: PruningStats | None = None
    bound_sweeps: int = 0
    bound_gap: float | None = None
