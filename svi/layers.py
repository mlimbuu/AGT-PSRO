from __future__ import annotations

from game.compat import dataclass, strict_zip

import heapq
import math
import warnings
from collections import deque
from dataclasses import field
from typing import Mapping, Sequence

import numpy as np

from game.environment import AdversarialGraphGame
from svi.matrix_game import solve_zero_sum_matrix_game
from game.models import (
    BlueAction,
    EquilibriumPolicy,
    GameState,
    GraphId,
    Node,
    PruningMode,
    PruningStats,
)
from svi.pruning import (
    IntervalPruningResult,
    compose_retained_indices,
    normalize_pruning_mode,
    prune_interval_dominated_actions,
    prune_strictly_dominated_actions,
    restore_full_policy,
)
from game.state_space import all_position_states

PositionState = tuple[Node, ...]
LayerKey = tuple[GraphId, int]


@dataclass(slots=True)
class AmmoLayerResult:
    values: dict[LayerKey, dict[PositionState, float]]
    policies: dict[LayerKey, dict[PositionState, EquilibriumPolicy]]
    converged: dict[LayerKey, bool]
    iterations: dict[LayerKey, int]
    residuals: dict[LayerKey, float]
    pruning_mode: PruningMode = PruningMode.NONE
    pruning_stats: PruningStats = field(default_factory=PruningStats)
    bound_sweeps: dict[LayerKey, int] = field(default_factory=dict)
    bound_gaps: dict[LayerKey, float] = field(default_factory=dict)


@dataclass(slots=True)
class LayerBoundPlan:
    lower_values: dict[PositionState, float]
    upper_values: dict[PositionState, float]
    action_masks: dict[PositionState, IntervalPruningResult]
    sweeps: int
    max_gap: float


def _terminal(game: AdversarialGraphGame, positions: PositionState) -> bool:
    return all(node == game.spec.goal_node for node in positions)


def _validate_position_states(
    game: AdversarialGraphGame,
    position_states: Sequence[PositionState],
) -> None:
    state_set = set(position_states)
    terminal = tuple(
        game.spec.goal_node for _ in range(game.spec.num_blue_agents)
    )
    if terminal not in state_set:
        raise ValueError("Terminal joint position is missing.")

    for positions in position_states:
        for graph_id in game.spec.graph_costs:
            state = GameState(positions, graph_id, 0)
            game.validate_state(state)
            if game.is_terminal(state):
                continue
            for action in game.blue_actions(state):
                if game.canonical_positions(action) not in state_set:
                    raise ValueError(
                        "Position state set is not transition-closed: "
                        f"{positions} -> {game.canonical_positions(action)}."
                    )


def _solve_zero_ammo_dijkstra(
    game: AdversarialGraphGame,
    graph_id: GraphId,
    position_states: Sequence[PositionState],
    *,
    cost_tolerance: float,
) -> tuple[
    dict[PositionState, float],
    dict[PositionState, EquilibriumPolicy],
]:
    """Exact undiscounted deterministic shortest path for ammo zero."""

    terminal = tuple(
        game.spec.goal_node for _ in range(game.spec.num_blue_agents)
    )
    state_set = set(position_states)

    reverse_edges: dict[
        PositionState,
        list[tuple[PositionState, float, BlueAction]],
    ] = {position: [] for position in position_states}

    for positions in position_states:
        state = GameState(positions, graph_id, 0)
        if game.is_terminal(state):
            continue
        for action in game.blue_actions(state):
            cost = game.stage_cost(state, action)
            if not math.isfinite(cost):
                raise ValueError("Stage cost must be finite.")
            if cost < -cost_tolerance:
                raise ValueError(
                    "Dijkstra zero-ammo solver requires nonnegative costs."
                )
            next_positions = game.canonical_positions(action)
            if next_positions not in state_set:
                raise ValueError("Position set is not transition-closed.")
            reverse_edges[next_positions].append((positions, cost, action))

    distances = {position: math.inf for position in position_states}
    chosen_actions: dict[PositionState, BlueAction] = {}
    distances[terminal] = 0.0
    queue: list[tuple[float, PositionState]] = [(0.0, terminal)]

    while queue:
        current_distance, current = heapq.heappop(queue)
        if current_distance > distances[current] + cost_tolerance:
            continue
        for predecessor, edge_cost, action in reverse_edges[current]:
            candidate = current_distance + edge_cost
            if candidate + cost_tolerance < distances[predecessor]:
                distances[predecessor] = candidate
                chosen_actions[predecessor] = action
                heapq.heappush(queue, (candidate, predecessor))

    unreachable = [
        positions
        for positions, value in distances.items()
        if not math.isfinite(value)
    ]
    if unreachable:
        raise ValueError(
            "Some position states cannot reach the goal in graph "
            f"{graph_id}: {unreachable[:5]}"
        )

    policies: dict[PositionState, EquilibriumPolicy] = {}
    for positions, action in chosen_actions.items():
        state = GameState(positions, graph_id, 0)
        actions = game.blue_actions(state)
        blue_probability = np.zeros(len(actions))
        blue_probability[actions.index(action)] = 1.0
        policies[positions] = EquilibriumPolicy(
            value=distances[positions],
            blue_actions=actions,
            red_actions=(graph_id,),
            blue_probabilities=blue_probability,
            red_probabilities=np.array([1.0]),
        )

    return distances, policies


def _build_layer_matrix(
    game: AdversarialGraphGame,
    positions: PositionState,
    graph_id: GraphId,
    ammo: int,
    current_layer_values: Mapping[PositionState, float],
    lower_layer_values: Mapping[LayerKey, Mapping[PositionState, float]],
) -> tuple[np.ndarray, tuple[BlueAction, ...], tuple[GraphId, ...]]:
    state = GameState(positions, graph_id, ammo)
    blue_actions = game.blue_actions(state)
    red_actions = game.red_actions(state)
    q = np.empty((len(red_actions), len(blue_actions)), dtype=float)

    for red_i, next_graph in enumerate(red_actions):
        for blue_i, blue_action in enumerate(blue_actions):
            next_positions = game.canonical_positions(blue_action)
            if next_graph == graph_id:
                continuation = current_layer_values[next_positions]
            else:
                if ammo <= 0:
                    raise RuntimeError(
                        "Red changed graph at zero ammo; transition logic is invalid."
                    )
                continuation = lower_layer_values[
                    (next_graph, ammo - 1)
                ][next_positions]
            q[red_i, blue_i] = (
                game.stage_cost(state, blue_action)
                + game.spec.gamma * continuation
            )

    return q, blue_actions, red_actions


def _build_layer_bound_matrices(
    game: AdversarialGraphGame,
    positions: PositionState,
    graph_id: GraphId,
    ammo: int,
    current_lower: Mapping[PositionState, float],
    current_upper: Mapping[PositionState, float],
    lower_layer_values: Mapping[LayerKey, Mapping[PositionState, float]],
) -> tuple[np.ndarray, np.ndarray]:
    state = GameState(positions, graph_id, ammo)
    blue_actions = game.blue_actions(state)
    red_actions = game.red_actions(state)
    q_lower = np.empty((len(red_actions), len(blue_actions)), dtype=float)
    q_upper = np.empty_like(q_lower)

    for red_i, next_graph in enumerate(red_actions):
        for blue_i, blue_action in enumerate(blue_actions):
            next_positions = game.canonical_positions(blue_action)
            if next_graph == graph_id:
                lower_continuation = current_lower[next_positions]
                upper_continuation = current_upper[next_positions]
            else:
                exact = lower_layer_values[(next_graph, ammo - 1)][
                    next_positions
                ]
                lower_continuation = exact
                upper_continuation = exact

            stage = game.stage_cost(state, blue_action)
            q_lower[red_i, blue_i] = (
                stage + game.spec.gamma * lower_continuation
            )
            q_upper[red_i, blue_i] = (
                stage + game.spec.gamma * upper_continuation
            )

    return q_lower, q_upper


def _hop_distances_to_goal(game: AdversarialGraphGame) -> dict[Node, int]:
    reverse: dict[Node, list[Node]] = {node: [] for node in game.spec.nodes}
    for current, successors in game.spec.adjacency.items():
        for successor in successors:
            if current != successor:
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
    hop_distance: Mapping[Node, int],
) -> BlueAction | None:
    candidates: list[BlueAction] = []
    for action in game.blue_actions(state):
        is_proper = True
        for current, successor in strict_zip(
            state.blue_positions, action
        ):
            if current == game.spec.goal_node:
                is_proper = is_proper and successor == current
            else:
                is_proper = is_proper and (
                    hop_distance[successor] < hop_distance[current]
                )
        if is_proper:
            candidates.append(action)

    if not candidates:
        return None
    return min(
        candidates,
        key=lambda action: (game.stage_cost(state, action), action),
    )


def _universal_discounted_upper(
    game: AdversarialGraphGame,
    position_states: Sequence[PositionState],
) -> float:
    if game.spec.gamma >= 1.0:
        return math.inf

    max_stage = 0.0
    for graph_id in game.spec.graph_costs:
        for positions in position_states:
            state = GameState(positions, graph_id, 0)
            if game.is_terminal(state):
                continue
            max_stage = max(
                max_stage,
                max(
                    game.stage_cost(state, action)
                    for action in game.blue_actions(state)
                ),
            )
    return max_stage / (1.0 - game.spec.gamma)


def _proper_policy_upper_values(
    game: AdversarialGraphGame,
    graph_id: GraphId,
    ammo: int,
    position_states: Sequence[PositionState],
    lower_layer_values: Mapping[LayerKey, Mapping[PositionState, float]],
) -> dict[PositionState, float]:
    """Evaluate a feasible goal-reaching Blue policy against worst-case Red."""

    hops = _hop_distances_to_goal(game)
    rank = {
        positions: sum(hops[node] for node in positions)
        for positions in position_states
    }
    universal = _universal_discounted_upper(game, position_states)
    upper: dict[PositionState, float] = {}

    for positions in sorted(position_states, key=lambda p: rank[p]):
        if _terminal(game, positions):
            upper[positions] = 0.0
            continue

        state = GameState(positions, graph_id, ammo)
        action = _proper_joint_action(game, state, hops)
        if action is None:
            if not math.isfinite(universal):
                raise ValueError(
                    "Could not construct a proper Blue policy needed for "
                    "finite gamma=1 bounds."
                )
            upper[positions] = universal
            continue

        next_positions = game.canonical_positions(action)
        red_outcomes: list[float] = []
        for next_graph in game.red_actions(state):
            if next_graph == graph_id:
                if rank[next_positions] >= rank[positions]:
                    if not math.isfinite(universal):
                        raise ValueError(
                            "Proper-policy action did not decrease goal rank."
                        )
                    continuation = universal
                else:
                    continuation = upper[next_positions]
            else:
                continuation = lower_layer_values[
                    (next_graph, ammo - 1)
                ][next_positions]
            red_outcomes.append(
                game.stage_cost(state, action)
                + game.spec.gamma * continuation
            )
        upper[positions] = max(red_outcomes)

    return upper


def _compute_layer_bound_plan(
    game: AdversarialGraphGame,
    graph_id: GraphId,
    ammo: int,
    position_states: Sequence[PositionState],
    lower_layer_values: Mapping[LayerKey, Mapping[PositionState, float]],
    *,
    max_sweeps: int,
    bound_tolerance: float,
    dominance_tolerance: float,
    stats: PruningStats,
) -> LayerBoundPlan:
    """Construct certified value intervals and permanent action masks.

    Lower updates use Red's pure maximin security value. Upper updates use
    Blue's pure minimax security value. No LP is required during this bound
    preprocessing. The initial upper bound is the worst-case value of a
    feasible goal-reaching Blue policy.
    """

    lower = {positions: 0.0 for positions in position_states}
    upper = _proper_policy_upper_values(
        game,
        graph_id,
        ammo,
        position_states,
        lower_layer_values,
    )

    completed_sweeps = 0
    for sweep in range(1, max_sweeps + 1):
        new_lower: dict[PositionState, float] = {}
        new_upper: dict[PositionState, float] = {}
        max_change = 0.0

        for positions in position_states:
            if _terminal(game, positions):
                new_lower[positions] = 0.0
                new_upper[positions] = 0.0
                continue

            q_lower, q_upper = _build_layer_bound_matrices(
                game,
                positions,
                graph_id,
                ammo,
                lower,
                upper,
                lower_layer_values,
            )

            lower_candidate = float(np.max(np.min(q_lower, axis=1)))
            upper_candidate = float(np.min(np.max(q_upper, axis=0)))

            # Preserve monotone certified sequences despite floating noise.
            lower_value = max(lower[positions], lower_candidate)
            upper_value = min(upper[positions], upper_candidate)
            if lower_value > upper_value + 100.0 * dominance_tolerance:
                raise RuntimeError(
                    "Certified value bounds crossed at "
                    f"layer={(graph_id, ammo)}, positions={positions}: "
                    f"lower={lower_value}, upper={upper_value}."
                )
            if lower_value > upper_value:
                midpoint = 0.5 * (lower_value + upper_value)
                lower_value = midpoint
                upper_value = midpoint

            new_lower[positions] = lower_value
            new_upper[positions] = upper_value
            max_change = max(
                max_change,
                abs(lower_value - lower[positions]),
                abs(upper_value - upper[positions]),
            )

        lower = new_lower
        upper = new_upper
        completed_sweeps = sweep
        stats.bound_state_updates += sum(
            not _terminal(game, p) for p in position_states
        )
        if max_change <= bound_tolerance:
            break

    masks: dict[PositionState, IntervalPruningResult] = {}
    for positions in position_states:
        if _terminal(game, positions):
            continue
        q_lower, q_upper = _build_layer_bound_matrices(
            game,
            positions,
            graph_id,
            ammo,
            lower,
            upper,
            lower_layer_values,
        )
        mask = prune_interval_dominated_actions(
            q_lower,
            q_upper,
            tolerance=dominance_tolerance,
        )
        masks[positions] = mask
        stats.bound_red_actions_removed += (
            q_lower.shape[0] - len(mask.retained_red_rows)
        )
        stats.bound_blue_actions_removed += (
            q_lower.shape[1] - len(mask.retained_blue_columns)
        )

    max_gap = max(
        upper[p] - lower[p] for p in position_states
    )
    stats.bound_sweeps += completed_sweeps
    return LayerBoundPlan(
        lower_values=lower,
        upper_values=upper,
        action_masks=masks,
        sweeps=completed_sweeps,
        max_gap=float(max_gap),
    )


def _solve_layer_matrix(
    q: np.ndarray,
    *,
    pruning_mode: PruningMode,
    bound_mask: IntervalPruningResult | None,
    dominance_tolerance: float,
    lp_tolerance: float,
    stats: PruningStats,
) -> tuple[float, np.ndarray, np.ndarray]:
    full_red_rows = tuple(range(q.shape[0]))
    full_blue_columns = tuple(range(q.shape[1]))
    retained_red_rows = full_red_rows
    retained_blue_columns = full_blue_columns
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
            retained_red_rows,
            temporary.retained_red_rows,
        )
        retained_blue_columns = compose_retained_indices(
            retained_blue_columns,
            temporary.retained_blue_columns,
        )
        reduced_q = temporary.matrix

    stats.matrix_games_solved += 1
    stats.original_matrix_entries += q.size
    stats.solved_matrix_entries += reduced_q.size

    solution = solve_zero_sum_matrix_game(reduced_q, tolerance=lp_tolerance)
    return (
        solution.value,
        restore_full_policy(
            solution.blue_policy,
            retained_blue_columns,
            q.shape[1],
        ),
        restore_full_policy(
            solution.red_policy,
            retained_red_rows,
            q.shape[0],
        ),
    )


def _layer_bellman_residual(
    game: AdversarialGraphGame,
    graph_id: GraphId,
    ammo: int,
    position_states: Sequence[PositionState],
    values: Mapping[PositionState, float],
    lower_layer_values: Mapping[LayerKey, Mapping[PositionState, float]],
    *,
    pruning_mode: PruningMode,
    bound_masks: Mapping[PositionState, IntervalPruningResult],
    dominance_tolerance: float,
    lp_tolerance: float,
    stats: PruningStats,
) -> float:
    residual = 0.0
    for positions in position_states:
        if _terminal(game, positions):
            candidate = 0.0
        else:
            q, _, _ = _build_layer_matrix(
                game,
                positions,
                graph_id,
                ammo,
                values,
                lower_layer_values,
            )
            candidate, _, _ = _solve_layer_matrix(
                q,
                pruning_mode=pruning_mode,
                bound_mask=bound_masks.get(positions),
                dominance_tolerance=dominance_tolerance,
                lp_tolerance=lp_tolerance,
                stats=stats,
            )
        residual = max(residual, abs(candidate - values[positions]))
    return float(residual)


def _solve_one_layer_by_value_iteration(
    game: AdversarialGraphGame,
    graph_id: GraphId,
    ammo: int,
    position_states: Sequence[PositionState],
    lower_layer_values: Mapping[LayerKey, Mapping[PositionState, float]],
    *,
    initial_values: Mapping[PositionState, float] | None,
    tolerance: float,
    max_iterations: int,
    pruning_mode: PruningMode,
    bound_masks: Mapping[PositionState, IntervalPruningResult],
    dominance_tolerance: float,
    lp_tolerance: float,
    stats: PruningStats,
) -> tuple[
    dict[PositionState, float],
    dict[PositionState, EquilibriumPolicy],
    bool,
    int,
    float,
]:
    values = {
        positions: (
            0.0
            if _terminal(game, positions) or initial_values is None
            else float(initial_values.get(positions, 0.0))
        )
        for positions in position_states
    }
    converged = False
    iterations = 0

    for iteration in range(1, max_iterations + 1):
        new_values: dict[PositionState, float] = {}
        for positions in position_states:
            if _terminal(game, positions):
                new_values[positions] = 0.0
                continue
            q, _, _ = _build_layer_matrix(
                game,
                positions,
                graph_id,
                ammo,
                values,
                lower_layer_values,
            )
            value, _, _ = _solve_layer_matrix(
                q,
                pruning_mode=pruning_mode,
                bound_mask=bound_masks.get(positions),
                dominance_tolerance=dominance_tolerance,
                lp_tolerance=lp_tolerance,
                stats=stats,
            )
            new_values[positions] = value

        iterate_change = max(
            abs(new_values[p] - values[p]) for p in position_states
        )
        values = new_values
        iterations = iteration
        if iterate_change <= tolerance:
            converged = True
            break

    policies: dict[PositionState, EquilibriumPolicy] = {}
    for positions in position_states:
        if _terminal(game, positions):
            continue
        q, blue_actions, red_actions = _build_layer_matrix(
            game,
            positions,
            graph_id,
            ammo,
            values,
            lower_layer_values,
        )
        value, blue_policy, red_policy = _solve_layer_matrix(
            q,
            pruning_mode=pruning_mode,
            bound_mask=bound_masks.get(positions),
            dominance_tolerance=dominance_tolerance,
            lp_tolerance=lp_tolerance,
            stats=stats,
        )
        policies[positions] = EquilibriumPolicy(
            value=value,
            blue_actions=blue_actions,
            red_actions=red_actions,
            blue_probabilities=blue_policy,
            red_probabilities=red_policy,
        )

    residual = _layer_bellman_residual(
        game,
        graph_id,
        ammo,
        position_states,
        values,
        lower_layer_values,
        pruning_mode=pruning_mode,
        bound_masks=bound_masks,
        dominance_tolerance=dominance_tolerance,
        lp_tolerance=lp_tolerance,
        stats=stats,
    )
    converged = converged and residual <= max(
        tolerance, 10.0 * lp_tolerance
    )
    return values, policies, converged, iterations, residual


def solve_ammo_layers(
    game: AdversarialGraphGame,
    *,
    position_states: Sequence[PositionState] | None = None,
    tolerance: float = 1e-9,
    max_iterations: int = 10_000,
    pruning_mode: PruningMode | str = PruningMode.NONE,
    prune_actions: bool | None = None,
    dominance_tolerance: float = 1e-10,
    lp_tolerance: float = 1e-9,
    bound_max_sweeps: int = 200,
    bound_tolerance: float = 1e-8,
    require_convergence: bool = True,
) -> AmmoLayerResult:
    """Solve layer games in ammo order ``0, 1, ..., max_ammo``.

    Modes:
      * ``none``: James-style full matrices, no pruning.
      * ``temporary_strict``: rebuild and strictly prune the current Q matrix
        on every exact value-iteration sweep.
      * ``bound_certified``: compute certified value intervals once per layer
        and permanently mask dominated state-actions in that layer.
      * ``hybrid``: bound-certified masks followed by temporary strict pruning.
    """

    if tolerance <= 0.0 or max_iterations <= 0:
        raise ValueError("tolerance and max_iterations must be positive.")
    if bound_max_sweeps <= 0 or bound_tolerance <= 0.0:
        raise ValueError("Bound settings must be positive.")

    mode = normalize_pruning_mode(
        pruning_mode,
        prune_actions=prune_actions,
    )
    positions = (
        all_position_states(game.spec)
        if position_states is None
        else tuple(position_states)
    )
    _validate_position_states(game, positions)

    if game.spec.gamma == 1.0 and game.spec.max_ammo > 0:
        warnings.warn(
            "Higher-ammo gamma=1 layer iteration has no generic contraction "
            "guarantee. Validate SSP assumptions.",
            RuntimeWarning,
            stacklevel=2,
        )

    all_values: dict[LayerKey, dict[PositionState, float]] = {}
    all_policies: dict[
        LayerKey, dict[PositionState, EquilibriumPolicy]
    ] = {}
    converged: dict[LayerKey, bool] = {}
    iterations: dict[LayerKey, int] = {}
    residuals: dict[LayerKey, float] = {}
    bound_sweeps: dict[LayerKey, int] = {}
    bound_gaps: dict[LayerKey, float] = {}
    stats = PruningStats()

    graph_ids = tuple(game.spec.graph_costs)

    for ammo in range(game.spec.max_ammo + 1):
        for graph_id in graph_ids:
            key = (graph_id, ammo)

            # Exact gamma=1, ammo=0 shortest path needs no matrix pruning.
            if ammo == 0 and game.spec.gamma == 1.0:
                values, policies = _solve_zero_ammo_dijkstra(
                    game,
                    graph_id,
                    positions,
                    cost_tolerance=dominance_tolerance,
                )
                all_values[key] = values
                all_policies[key] = policies
                converged[key] = True
                iterations[key] = 1
                residuals[key] = 0.0
                bound_sweeps[key] = 0
                bound_gaps[key] = 0.0
                continue

            masks: Mapping[PositionState, IntervalPruningResult] = {}
            if mode.uses_bound_pruning:
                plan = _compute_layer_bound_plan(
                    game,
                    graph_id,
                    ammo,
                    positions,
                    all_values,
                    max_sweeps=bound_max_sweeps,
                    bound_tolerance=bound_tolerance,
                    dominance_tolerance=dominance_tolerance,
                    stats=stats,
                )
                masks = plan.action_masks
                bound_sweeps[key] = plan.sweeps
                bound_gaps[key] = plan.max_gap
            else:
                bound_sweeps[key] = 0
                bound_gaps[key] = math.nan

            initial_values = (
                None
                if ammo == 0
                else all_values[(graph_id, ammo - 1)]
            )
            result = _solve_one_layer_by_value_iteration(
                game,
                graph_id,
                ammo,
                positions,
                all_values,
                initial_values=initial_values,
                tolerance=tolerance,
                max_iterations=max_iterations,
                pruning_mode=mode,
                bound_masks=masks,
                dominance_tolerance=dominance_tolerance,
                lp_tolerance=lp_tolerance,
                stats=stats,
            )
            (
                all_values[key],
                all_policies[key],
                converged[key],
                iterations[key],
                residuals[key],
            ) = result

            if require_convergence and not converged[key]:
                raise RuntimeError(
                    f"Layer {key} did not converge: "
                    f"iterations={iterations[key]}, "
                    f"Bellman residual={residuals[key]}."
                )

    return AmmoLayerResult(
        values=all_values,
        policies=all_policies,
        converged=converged,
        iterations=iterations,
        residuals=residuals,
        pruning_mode=mode,
        pruning_stats=stats,
        bound_sweeps=bound_sweeps,
        bound_gaps=bound_gaps,
    )
