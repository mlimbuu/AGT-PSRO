from __future__ import annotations

from collections import deque
from itertools import combinations_with_replacement, product
from typing import Iterable

from game.environment import AdversarialGraphGame, has_forbidden_collision
from game.models import GameSpec, GameState, Node


def all_position_states(spec: GameSpec) -> tuple[tuple[Node, ...], ...]:
    positions: list[tuple[Node, ...]] = []
    joint_position_iterator = (
        combinations_with_replacement(spec.nodes, spec.num_blue_agents)
        if spec.indistinguishable_blue_agents
        else product(spec.nodes, repeat=spec.num_blue_agents)
    )
    for joint_positions in joint_position_iterator:
        position = tuple(joint_positions)
        if (
            not spec.allow_collisions
            and has_forbidden_collision(position, spec.goal_node)
        ):
            continue
        positions.append(position)
    return tuple(positions)


def enumerate_full_state_space(spec: GameSpec) -> tuple[GameState, ...]:
    return tuple(
        GameState(positions, graph_id, ammo)
        for positions in all_position_states(spec)
        for graph_id in spec.graph_costs
        for ammo in range(spec.max_ammo + 1)
    )


def reachable_states(
    game: AdversarialGraphGame,
    initial_state: GameState,
) -> tuple[GameState, ...]:
    initial_state = game.canonical_state(initial_state)
    game.validate_state(initial_state)
    discovered: set[GameState] = {initial_state}
    queue: deque[GameState] = deque([initial_state])

    while queue:
        state = queue.popleft()
        if game.is_terminal(state):
            continue
        for blue_action in game.blue_actions(state):
            for red_action in game.red_actions(state):
                next_state = game.transition(state, blue_action, red_action)
                if next_state not in discovered:
                    discovered.add(next_state)
                    queue.append(next_state)

    return tuple(
        sorted(
            discovered,
            key=lambda s: (s.red_ammo, s.graph_id, s.blue_positions),
        )
    )


def ensure_transition_closed(
    game: AdversarialGraphGame,
    states: Iterable[GameState],
) -> None:
    state_set = set(states)
    for state in state_set:
        if game.is_terminal(state):
            continue
        for blue_action in game.blue_actions(state):
            for red_action in game.red_actions(state):
                next_state = game.transition(state, blue_action, red_action)
                if next_state not in state_set:
                    raise ValueError(
                        "Provided state set is not transition-closed. "
                        f"Missing successor {next_state} from {state}."
                    )
