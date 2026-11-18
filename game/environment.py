from __future__ import annotations

from game.compat import dataclass

from dataclasses import field
from itertools import product

from game.costs import EdgeTraversalCost, StageCostModel
from game.models import BlueAction, GameSpec, GameState, RedAction
from game.symmetry import (
    canonicalize_positions,
    unique_actions_mod_agent_symmetry,
)


def has_forbidden_collision(
    positions: tuple[int, ...],
    goal_node: int,
) -> bool:
    """Return whether two agents share a non-goal node.

    Multiple robots are allowed to accumulate at the absorbing goal even when
    ordinary collisions are disabled. This is the usual multi-robot terminal
    convention and avoids rejecting states where some robots have already
    arrived while others are still moving.
    """

    non_goal_positions = [
        position for position in positions if position != goal_node
    ]
    return len(non_goal_positions) != len(set(non_goal_positions))


@dataclass(slots=True)
class AdversarialGraphGame:
    """Game dynamics and action generation."""

    spec: GameSpec
    cost_model: StageCostModel = field(default_factory=EdgeTraversalCost)

    def canonical_positions(
        self, positions: tuple[int, ...] | BlueAction
    ) -> tuple[int, ...]:
        """Map a labeled joint position to its quotient-state representative."""

        position_tuple = tuple(positions)
        if not self.spec.indistinguishable_blue_agents:
            return position_tuple
        return canonicalize_positions(position_tuple)

    def canonical_state(self, state: GameState) -> GameState:
        """Return the canonical state representing an agent-permutation orbit."""

        return GameState(
            blue_positions=self.canonical_positions(state.blue_positions),
            graph_id=state.graph_id,
            red_ammo=state.red_ammo,
        )

    def validate_state(self, state: GameState) -> None:
        if len(state.blue_positions) != self.spec.num_blue_agents:
            raise ValueError("State has the wrong number of Blue positions.")
        if (
            self.spec.indistinguishable_blue_agents
            and state.blue_positions != self.canonical_positions(state.blue_positions)
        ):
            raise ValueError(
                "State is not canonical for indistinguishable Blue agents: "
                f"{state.blue_positions}. Use game.canonical_state(...)."
            )
        invalid_positions = set(state.blue_positions) - set(self.spec.nodes)
        if invalid_positions:
            raise ValueError(
                f"State contains invalid Blue nodes: {invalid_positions}."
            )
        if state.graph_id not in self.spec.graph_costs:
            raise ValueError(f"Unknown graph_id {state.graph_id}.")
        if not (0 <= state.red_ammo <= self.spec.max_ammo):
            raise ValueError(
                f"Ammo {state.red_ammo} lies outside the allowed range."
            )
        if (
            not self.spec.allow_collisions
            and has_forbidden_collision(
                state.blue_positions, self.spec.goal_node
            )
        ):
            raise ValueError("Blue-agent collisions are disabled.")

    def is_terminal(self, state: GameState) -> bool:
        return all(
            position == self.spec.goal_node
            for position in state.blue_positions
        )

    def blue_actions(self, state: GameState) -> tuple[BlueAction, ...]:
        self.validate_state(state)

        if self.is_terminal(state):
            return (
                tuple(
                    self.spec.goal_node
                    for _ in range(self.spec.num_blue_agents)
                ),
            )

        per_agent_successors = tuple(
            self.spec.adjacency[position]
            for position in state.blue_positions
        )

        actions: list[BlueAction] = []
        for joint_action in product(*per_agent_successors):
            action = tuple(joint_action)
            if (
                not self.spec.allow_collisions
                and has_forbidden_collision(action, self.spec.goal_node)
            ):
                continue
            actions.append(action)

        if not actions:
            raise RuntimeError(f"No valid Blue action exists at state {state}.")

        if self.spec.indistinguishable_blue_agents:
            return unique_actions_mod_agent_symmetry(
                state.blue_positions, actions
            )
        return tuple(actions)

    def red_actions(self, state: GameState) -> tuple[RedAction, ...]:
        self.validate_state(state)
        if state.red_ammo == 0:
            return (state.graph_id,)
        return tuple(self.spec.red_action_graph[state.graph_id])

    def stage_cost(self, state: GameState, blue_action: BlueAction) -> float:
        if self.is_terminal(state):
            return 0.0
        return float(self.cost_model(state, blue_action, self.spec))

    def transition(
        self,
        state: GameState,
        blue_action: BlueAction,
        red_action: RedAction,
    ) -> GameState:
        """Apply simultaneous actions using James's timing convention."""

        if blue_action not in self.blue_actions(state):
            raise ValueError(f"Invalid Blue action {blue_action} at state {state}.")
        if red_action not in self.red_actions(state):
            raise ValueError(f"Invalid Red action {red_action} at state {state}.")

        graph_changed = red_action != state.graph_id
        next_state = GameState(
            blue_positions=self.canonical_positions(blue_action),
            graph_id=red_action,
            red_ammo=state.red_ammo - int(graph_changed),
        )
        self.validate_state(next_state)
        return next_state
