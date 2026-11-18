from __future__ import annotations

from game.compat import dataclass, strict_zip

import math
import warnings
from collections import deque

from typing import Mapping, Sequence

import numpy as np

from game.environment import AdversarialGraphGame
from svi.matrix_game import solve_zero_sum_matrix_game
from game.models import (
    BlueAction,
    EquilibriumPolicy,
    GameState,
    Node,
    PruningMode,
    PruningStats,
    RedAction,
    ValueIterationResult,
)
from svi.pruning import (
    IntervalPruningResult,
    compose_retained_indices,
    normalize_pruning_mode,
    prune_interval_dominated_actions,
    prune_strictly_dominated_actions,
    restore_full_policy,
)
from game.state_space import (
    ensure_transition_closed,
    enumerate_full_state_space,
    reachable_states,
)


@dataclass(slots=True)
class StateMatrix:
    q: np.ndarray
    blue_actions: tuple[BlueAction, ...]
    red_actions: tuple[RedAction, ...]


@dataclass(slots=True)
class StateBoundPlan:
    lower_values: dict[GameState, float]
    upper_values: dict[GameState, float]
    action_masks: dict[GameState, IntervalPruningResult]
    sweeps: int
    max_gap: float


def build_state_matrix(
    game: AdversarialGraphGame,
    state: GameState,
    continuation_values: Mapping[GameState, float],
) -> StateMatrix:
    """Construct the current Bellman-Shapley matrix at one state."""

    blue_actions = game.blue_actions(state)
    red_actions = game.red_actions(state)
    q = np.empty((len(red_actions), len(blue_actions)), dtype=float)

    for red_index, red_action in enumerate(red_actions):
        for blue_index, blue_action in enumerate(blue_actions):
            next_state = game.transition(state, blue_action, red_action)
            if next_state not in continuation_values:
                raise KeyError(f"No continuation value for {next_state}.")
            q[red_index, blue_index] = (
                game.stage_cost(state, blue_action)
                + game.spec.gamma * continuation_values[next_state]
            )

    return StateMatrix(q, blue_actions, red_actions)


def _build_state_bound_matrices(
    game: AdversarialGraphGame,
    state: GameState,
    lower_values: Mapping[GameState, float],
    upper_values: Mapping[GameState, float],
) -> tuple[np.ndarray, np.ndarray]:
    blue_actions = game.blue_actions(state)
    red_actions = game.red_actions(state)
    lower_q = np.empty((len(red_actions), len(blue_actions)))
    upper_q = np.empty_like(lower_q)

    for red_i, red_action in enumerate(red_actions):
        for blue_i, blue_action in enumerate(blue_actions):
            next_state = game.transition(state, blue_action, red_action)
            stage = game.stage_cost(state, blue_action)
            lower_q[red_i, blue_i] = (
                stage + game.spec.gamma * lower_values[next_state]
            )
            upper_q[red_i, blue_i] = (
                stage + game.spec.gamma * upper_values[next_state]
            )
    return lower_q, upper_q


def _hop_distances_to_goal(game: AdversarialGraphGame) -> dict[Node, int]:
    reverse: dict[Node, list[Node]] = {node: [] for node in game.spec.nodes}
    for current, successors in game.spec.adjacency.items():
        for successor in successors:
            if successor != current:
                reverse[successor].append(current)

    distances = {game.spec.goal_node: 0}
    queue: deque[Node] = deque([game.spec.goal_node])
    while queue:
        successor = queue.popleft()
        for predecessor in reverse[successor]:
            if predecessor not in distances:
                distances[predecessor] = distances[successor] + 1
                queue.append(predecessor)

    missing = set(game.spec.nodes) - set(distances)
    if missing:
        raise ValueError(f"Nodes cannot reach the goal: {sorted(missing)}")
    return distances


def _proper_joint_action(
    game: AdversarialGraphGame,
    state: GameState,
    hops: Mapping[Node, int],
) -> BlueAction | None:
    candidates: list[BlueAction] = []
    for action in game.blue_actions(state):
        proper = all(
            (successor == current)
            if current == game.spec.goal_node
            else hops[successor] < hops[current]
            for current, successor in strict_zip(
                state.blue_positions, action
            )
        )
        if proper:
            candidates.append(action)

    if not candidates:
        return None
    return min(
        candidates,
        key=lambda action: (game.stage_cost(state, action), action),
    )


def _proper_policy_upper_values(
    game: AdversarialGraphGame,
    states: Sequence[GameState],
) -> dict[GameState, float]:
    """Evaluate one feasible goal-reaching policy against worst-case Red."""

    state_set = set(states)
    hops = _hop_distances_to_goal(game)
    rank = {
        state: sum(hops[node] for node in state.blue_positions)
        for state in states
    }
    upper: dict[GameState, float] = {}

    for state in sorted(
        states,
        key=lambda s: (rank[s], s.red_ammo, s.graph_id),
    ):
        if game.is_terminal(state):
            upper[state] = 0.0
            continue
        action = _proper_joint_action(game, state, hops)
        if action is None:
            raise ValueError(
                "Could not construct a proper Blue policy for certified bounds."
            )
        outcomes = []
        for red_action in game.red_actions(state):
            next_state = game.transition(state, action, red_action)
            if next_state not in state_set:
                raise ValueError("State set is not transition-closed.")
            if rank[next_state] >= rank[state]:
                raise ValueError("Proper-policy action did not decrease rank.")
            outcomes.append(
                game.stage_cost(state, action)
                + game.spec.gamma * upper[next_state]
            )
        upper[state] = max(outcomes)
    return upper


def _compute_state_bound_plan(
    game: AdversarialGraphGame,
    states: Sequence[GameState],
    *,
    max_sweeps: int,
    bound_tolerance: float,
    dominance_tolerance: float,
    stats: PruningStats,
) -> StateBoundPlan:
    lower = {state: 0.0 for state in states}
    upper = _proper_policy_upper_values(game, states)
    completed_sweeps = 0

    for sweep in range(1, max_sweeps + 1):
        new_lower: dict[GameState, float] = {}
        new_upper: dict[GameState, float] = {}
        max_change = 0.0

        for state in states:
            if game.is_terminal(state):
                new_lower[state] = 0.0
                new_upper[state] = 0.0
                continue
            lower_q, upper_q = _build_state_bound_matrices(
                game, state, lower, upper
            )
            lower_candidate = float(np.max(np.min(lower_q, axis=1)))
            upper_candidate = float(np.min(np.max(upper_q, axis=0)))
            lower_value = max(lower[state], lower_candidate)
            upper_value = min(upper[state], upper_candidate)
            if lower_value > upper_value + 100.0 * dominance_tolerance:
                raise RuntimeError(
                    f"Certified bounds crossed at state {state}: "
                    f"{lower_value} > {upper_value}."
                )
            if lower_value > upper_value:
                midpoint = 0.5 * (lower_value + upper_value)
                lower_value = midpoint
                upper_value = midpoint
            new_lower[state] = lower_value
            new_upper[state] = upper_value
            max_change = max(
                max_change,
                abs(lower_value - lower[state]),
                abs(upper_value - upper[state]),
            )

        lower = new_lower
        upper = new_upper
        completed_sweeps = sweep
        stats.bound_state_updates += sum(
            not game.is_terminal(state) for state in states
        )
        if max_change <= bound_tolerance:
            break

    masks: dict[GameState, IntervalPruningResult] = {}
    for state in states:
        if game.is_terminal(state):
            continue
        lower_q, upper_q = _build_state_bound_matrices(
            game, state, lower, upper
        )
        mask = prune_interval_dominated_actions(
            lower_q, upper_q, tolerance=dominance_tolerance
        )
        masks[state] = mask
        stats.bound_red_actions_removed += (
            lower_q.shape[0] - len(mask.retained_red_rows)
        )
        stats.bound_blue_actions_removed += (
            lower_q.shape[1] - len(mask.retained_blue_columns)
        )

    stats.bound_sweeps += completed_sweeps
    return StateBoundPlan(
        lower_values=lower,
        upper_values=upper,
        action_masks=masks,
        sweeps=completed_sweeps,
        max_gap=float(max(upper[s] - lower[s] for s in states)),
    )


def _solve_state_matrix(
    state_matrix: StateMatrix,
    *,
    pruning_mode: PruningMode,
    bound_mask: IntervalPruningResult | None,
    dominance_tolerance: float,
    lp_tolerance: float,
    stats: PruningStats,
) -> EquilibriumPolicy:
    q = state_matrix.q
    retained_red_rows = tuple(range(q.shape[0]))
    retained_blue_columns = tuple(range(q.shape[1]))
    reduced_q = q

    if pruning_mode.uses_bound_pruning:
        if bound_mask is None:
            raise RuntimeError("Bound-pruning mode requires a certified mask.")
        retained_red_rows = bound_mask.retained_red_rows
        retained_blue_columns = bound_mask.retained_blue_columns
        reduced_q = bound_mask.reduce(q)

    if pruning_mode.uses_temporary_pruning:
        temporary = prune_strictly_dominated_actions(
            reduced_q,
            tolerance=dominance_tolerance,
        )
        stats.temporary_red_actions_removed += (
            reduced_q.shape[0] - len(temporary.retained_red_rows)
        )
        stats.temporary_blue_actions_removed += (
            reduced_q.shape[1] - len(temporary.retained_blue_columns)
        )
        retained_red_rows = compose_retained_indices(
            retained_red_rows, temporary.retained_red_rows
        )
        retained_blue_columns = compose_retained_indices(
            retained_blue_columns, temporary.retained_blue_columns
        )
        reduced_q = temporary.matrix

    stats.matrix_games_solved += 1
    stats.original_matrix_entries += q.size
    stats.solved_matrix_entries += reduced_q.size
    solution = solve_zero_sum_matrix_game(reduced_q, tolerance=lp_tolerance)

    return EquilibriumPolicy(
        value=solution.value,
        blue_actions=state_matrix.blue_actions,
        red_actions=state_matrix.red_actions,
        blue_probabilities=restore_full_policy(
            solution.blue_policy,
            retained_blue_columns,
            q.shape[1],
        ),
        red_probabilities=restore_full_policy(
            solution.red_policy,
            retained_red_rows,
            q.shape[0],
        ),
    )


def bellman_shapley_residual(
    game: AdversarialGraphGame,
    states: Sequence[GameState],
    values: Mapping[GameState, float],
    *,
    pruning_mode: PruningMode | str = PruningMode.NONE,
    prune_actions: bool | None = None,
    bound_masks: Mapping[GameState, IntervalPruningResult] | None = None,
    dominance_tolerance: float = 1e-10,
    lp_tolerance: float = 1e-9,
    stats: PruningStats | None = None,
) -> float:
    """Compute ``||T(V)-V||_infinity`` using exact current stage games."""

    mode = normalize_pruning_mode(
        pruning_mode, prune_actions=prune_actions
    )
    masks = bound_masks or {}
    local_stats = stats or PruningStats()
    residual = 0.0
    for state in states:
        if game.is_terminal(state):
            candidate = 0.0
        else:
            candidate = _solve_state_matrix(
                build_state_matrix(game, state, values),
                pruning_mode=mode,
                bound_mask=masks.get(state),
                dominance_tolerance=dominance_tolerance,
                lp_tolerance=lp_tolerance,
                stats=local_stats,
            ).value
        residual = max(residual, abs(candidate - values[state]))
    return float(residual)


def shapley_value_iteration(
    game: AdversarialGraphGame,
    *,
    initial_state: GameState | None = None,
    states: Sequence[GameState] | None = None,
    initial_values: Mapping[GameState, float] | None = None,
    tolerance: float = 1e-9,
    max_iterations: int = 10_000,
    pruning_mode: PruningMode | str = PruningMode.NONE,
    prune_actions: bool | None = None,
    dominance_tolerance: float = 1e-10,
    lp_tolerance: float = 1e-9,
    bound_max_sweeps: int = 200,
    bound_tolerance: float = 1e-8,
    require_convergence: bool = True,
) -> ValueIterationResult:
    """Run synchronous Bellman-Shapley value iteration on full states."""

    if max_iterations <= 0 or tolerance <= 0.0:
        raise ValueError("max_iterations and tolerance must be positive.")
    if states is not None and initial_state is not None:
        raise ValueError("Pass either states or initial_state, not both.")

    mode = normalize_pruning_mode(
        pruning_mode, prune_actions=prune_actions
    )
    if states is None:
        state_sequence = (
            enumerate_full_state_space(game.spec)
            if initial_state is None
            else reachable_states(game, initial_state)
        )
    else:
        state_sequence = tuple(states)
        ensure_transition_closed(game, state_sequence)

    if not state_sequence:
        raise ValueError("No states were supplied.")
    if game.spec.gamma == 1.0:
        warnings.warn(
            "gamma=1 removes the contraction guarantee; convergence depends "
            "on SSP/proper-policy assumptions.",
            RuntimeWarning,
            stacklevel=2,
        )

    stats = PruningStats()
    masks: Mapping[GameState, IntervalPruningResult] = {}
    bound_sweeps = 0
    bound_gap: float | None = None
    if mode.uses_bound_pruning:
        plan = _compute_state_bound_plan(
            game,
            state_sequence,
            max_sweeps=bound_max_sweeps,
            bound_tolerance=bound_tolerance,
            dominance_tolerance=dominance_tolerance,
            stats=stats,
        )
        masks = plan.action_masks
        bound_sweeps = plan.sweeps
        bound_gap = plan.max_gap

    values = {
        state: (
            0.0
            if game.is_terminal(state) or initial_values is None
            else float(initial_values.get(state, 0.0))
        )
        for state in state_sequence
    }

    converged = False
    iterate_change = math.inf
    completed_iterations = 0

    for iteration in range(1, max_iterations + 1):
        new_values: dict[GameState, float] = {}
        for state in state_sequence:
            if game.is_terminal(state):
                new_values[state] = 0.0
                continue
            equilibrium = _solve_state_matrix(
                build_state_matrix(game, state, values),
                pruning_mode=mode,
                bound_mask=masks.get(state),
                dominance_tolerance=dominance_tolerance,
                lp_tolerance=lp_tolerance,
                stats=stats,
            )
            new_values[state] = equilibrium.value

        iterate_change = max(
            abs(new_values[state] - values[state])
            for state in state_sequence
        )
        values = new_values
        completed_iterations = iteration
        if iterate_change <= tolerance:
            converged = True
            break

    policies: dict[GameState, EquilibriumPolicy] = {}
    for state in state_sequence:
        if not game.is_terminal(state):
            policies[state] = _solve_state_matrix(
                build_state_matrix(game, state, values),
                pruning_mode=mode,
                bound_mask=masks.get(state),
                dominance_tolerance=dominance_tolerance,
                lp_tolerance=lp_tolerance,
                stats=stats,
            )

    final_residual = bellman_shapley_residual(
        game,
        state_sequence,
        values,
        pruning_mode=mode,
        bound_masks=masks,
        dominance_tolerance=dominance_tolerance,
        lp_tolerance=lp_tolerance,
        stats=stats,
    )
    converged = converged and final_residual <= max(
        tolerance, 10.0 * lp_tolerance
    )

    result = ValueIterationResult(
        values=values,
        policies=policies,
        converged=converged,
        iterations=completed_iterations,
        residual=float(final_residual),
        states=tuple(state_sequence),
        pruning_mode=mode,
        pruning_stats=stats,
        bound_sweeps=bound_sweeps,
        bound_gap=bound_gap,
    )

    if require_convergence and not result.converged:
        raise RuntimeError(
            "Bellman-Shapley value iteration did not converge: "
            f"iterations={result.iterations}, "
            f"iterate_change={iterate_change}, "
            f"Bellman residual={result.residual}."
        )
    return result
