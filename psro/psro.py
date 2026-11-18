from __future__ import annotations

from game.compat import dataclass, strict_zip

from dataclasses import field
from typing import Union, Hashable, Literal, Mapping, Sequence

import numpy as np

from game.environment import AdversarialGraphGame
from svi.matrix_game import solve_zero_sum_matrix_game
from psro.meta_game_matrix import build_empirical_payoff_matrix
from game.models import BlueAction, GameState, RedAction
from game.state_space import all_position_states

Player = Literal["blue", "red"]
Action = Union[BlueAction, RedAction]
PositionState = tuple[int, ...]
LayerKey = tuple[int, int]


@dataclass(slots=True)
class TabularPolicy:
    """Stationary Markov policy over the complete game state.

    The action map stores probabilities by action value rather than by action
    index. This makes policies robust to legal-action ordering in the game.
    Missing terminal-state entries are handled automatically by
    ``distribution``.
    """

    player: Player
    action_probabilities: dict[GameState, dict[Hashable, float]]
    name: str = "policy"

    def distribution(
        self,
        game: AdversarialGraphGame,
        state: GameState,
    ) -> tuple[tuple[Action, ...], np.ndarray]:
        if self.player == "blue":
            legal: tuple[Action, ...] = tuple(game.blue_actions(state))
        else:
            legal = tuple(game.red_actions(state))

        if game.is_terminal(state):
            # Terminal values are zero, so the precise action is immaterial.
            # Keep a deterministic legal action to make evaluation well-defined.
            probabilities = np.zeros(len(legal), dtype=float)
            probabilities[0] = 1.0
            return legal, probabilities

        stored = self.action_probabilities.get(state)
        if stored is None:
            raise KeyError(
                f"{self.name!r} has no {self.player} policy entry for {state}."
            )

        probabilities = np.array(
            [float(stored.get(action, 0.0)) for action in legal],
            dtype=float,
        )
        if np.any(probabilities < -1e-12):
            raise ValueError(f"{self.name!r} contains a negative probability.")
        probabilities = np.clip(probabilities, 0.0, None)
        total = float(probabilities.sum())
        if total <= 1e-15:
            raise ValueError(
                f"{self.name!r} assigns no probability to legal actions at {state}."
            )
        probabilities /= total
        return legal, probabilities

    def signature(self, *, digits: int = 12) -> tuple:
        rows = []
        for state in sorted(
            self.action_probabilities,
            key=lambda s: (s.red_ammo, s.graph_id, s.blue_positions),
        ):
            action_map = self.action_probabilities[state]
            rows.append(
                (
                    state,
                    tuple(
                        sorted(
                            (
                                repr(action),
                                round(float(probability), digits),
                            )
                            for action, probability in action_map.items()
                            if probability > 10.0 ** (-digits)
                        )
                    ),
                )
            )
        return (self.player, tuple(rows))


@dataclass(slots=True)
class BestResponseResult:
    policy: TabularPolicy
    values: dict[LayerKey, dict[PositionState, float]]
    initial_value: float
    policy_iterations: int


@dataclass(slots=True)
class PSROIteration:
    iteration: int
    blue_population_size: int
    red_population_size: int
    meta_value: float
    blue_meta_strategy: list[float]
    red_meta_strategy: list[float]
    blue_br_value: float
    red_br_value: float
    blue_improvement: float
    red_improvement: float
    oracle_gap: float
    blue_added: bool
    red_added: bool


@dataclass(slots=True)
class PSROResult:
    blue_population: list[TabularPolicy]
    red_population: list[TabularPolicy]
    payoff_matrix: np.ndarray
    blue_meta_strategy: np.ndarray
    red_meta_strategy: np.ndarray
    meta_value: float
    history: list[PSROIteration] = field(default_factory=list)
    converged: bool = False
    semantics: str = "behavioral_mixture_exact_oracle"


def _all_states(game: AdversarialGraphGame) -> tuple[GameState, ...]:
    positions = all_position_states(game.spec)
    return tuple(
        GameState(position, graph_id, ammo)
        for ammo in range(game.spec.max_ammo + 1)
        for graph_id in game.spec.graph_costs
        for position in positions
    )


def _hop_distances_to_goal(game: AdversarialGraphGame) -> dict[int, int]:
    reverse: dict[int, list[int]] = {node: [] for node in game.spec.nodes}
    for current, successors in game.spec.adjacency.items():
        for successor in successors:
            if successor != current:
                reverse[successor].append(current)

    distances = {game.spec.goal_node: 0}
    frontier = [game.spec.goal_node]
    while frontier:
        successor = frontier.pop(0)
        for predecessor in reverse[successor]:
            if predecessor not in distances:
                distances[predecessor] = distances[successor] + 1
                frontier.append(predecessor)

    missing = set(game.spec.nodes) - set(distances)
    if missing:
        raise ValueError(f"Nodes cannot reach the goal: {sorted(missing)}")
    return distances


def tabular_policy_from_ammo_layer_result(
    game: AdversarialGraphGame,
    result,
    *,
    player: Player,
    name: str | None = None,
) -> TabularPolicy:
    """Extract one player's stationary Markov policy from ``AmmoLayerResult``."""

    mapping: dict[GameState, dict[Hashable, float]] = {}
    for (graph_id, ammo), layer_policies in result.policies.items():
        for positions, equilibrium in layer_policies.items():
            state = GameState(positions, graph_id, ammo)
            if player == "blue":
                actions = equilibrium.blue_actions
                probabilities = equilibrium.blue_probabilities
            else:
                actions = equilibrium.red_actions
                probabilities = equilibrium.red_probabilities
            mapping[state] = {
                action: float(probability)
                for action, probability in strict_zip(actions, probabilities)
                if probability > 0.0
            }
    return TabularPolicy(
        player,
        mapping,
        name=name or f"{player}_ammo_layer_policy",
    )


def make_proper_blue_policy(
    game: AdversarialGraphGame,
    *,
    name: str = "blue_proper_seed",
) -> TabularPolicy:
    """Create a deterministic goal-directed seed policy for Blue."""

    hops = _hop_distances_to_goal(game)
    mapping: dict[GameState, dict[Hashable, float]] = {}

    for state in _all_states(game):
        if game.is_terminal(state):
            continue
        actions = game.blue_actions(state)

        def key(action: BlueAction) -> tuple:
            progress = sum(hops[node] for node in action)
            stage = game.stage_cost(state, action)
            return (progress, stage, action)

        chosen = min(actions, key=key)
        mapping[state] = {chosen: 1.0}

    return TabularPolicy("blue", mapping, name=name)


def make_stay_red_policy(
    game: AdversarialGraphGame,
    *,
    name: str = "red_stay_seed",
) -> TabularPolicy:
    """Create a deterministic Red seed that always takes the self-loop."""

    mapping: dict[GameState, dict[Hashable, float]] = {}
    for state in _all_states(game):
        if game.is_terminal(state):
            continue
        actions = game.red_actions(state)
        chosen = state.graph_id if state.graph_id in actions else actions[0]
        mapping[state] = {chosen: 1.0}
    return TabularPolicy("red", mapping, name=name)


def _mixture_distribution(
    game: AdversarialGraphGame,
    state: GameState,
    policies: Sequence[TabularPolicy],
    weights: Sequence[float],
    *,
    player: Player,
) -> tuple[tuple[Action, ...], np.ndarray]:
    if len(policies) != len(weights) or not policies:
        raise ValueError("policies and weights must be nonempty and have equal length.")

    weights_array = np.asarray(weights, dtype=float)
    if np.any(weights_array < -1e-12):
        raise ValueError("Meta-strategy weights must be nonnegative.")
    weights_array = np.clip(weights_array, 0.0, None)
    total = float(weights_array.sum())
    if total <= 1e-15:
        raise ValueError("Meta-strategy weights sum to zero.")
    weights_array /= total

    legal: tuple[Action, ...]
    if player == "blue":
        legal = tuple(game.blue_actions(state))
    else:
        legal = tuple(game.red_actions(state))

    aggregate = np.zeros(len(legal), dtype=float)
    for policy, weight in strict_zip(policies, weights_array):
        if policy.player != player:
            raise ValueError(f"Expected a {player} policy, got {policy.player}.")
        policy_actions, probabilities = policy.distribution(game, state)
        if policy_actions != legal:
            raise RuntimeError("Policy legal-action ordering mismatch.")
        aggregate += float(weight) * probabilities

    aggregate /= aggregate.sum()
    return legal, aggregate


def _solve_linear_layer(
    game: AdversarialGraphGame,
    graph_id: int,
    ammo: int,
    position_states: Sequence[PositionState],
    lower_values: Mapping[LayerKey, Mapping[PositionState, float]],
    blue_distribution,
    red_distribution,
) -> dict[PositionState, float]:
    """Evaluate stationary policies exactly within one ammo/graph layer."""

    index = {position: i for i, position in enumerate(position_states)}
    n = len(position_states)
    a = np.eye(n, dtype=float)
    rhs = np.zeros(n, dtype=float)

    for positions in position_states:
        row = index[positions]
        state = GameState(positions, graph_id, ammo)
        if game.is_terminal(state):
            rhs[row] = 0.0
            continue

        blue_actions, blue_probs = blue_distribution(state)
        red_actions, red_probs = red_distribution(state)

        for blue_action, p_blue in strict_zip(blue_actions, blue_probs):
            if p_blue <= 0.0:
                continue
            stage = game.stage_cost(state, blue_action)
            rhs[row] += float(p_blue) * stage

            next_positions = game.canonical_positions(blue_action)
            for red_action, p_red in strict_zip(red_actions, red_probs):
                probability = float(p_blue * p_red)
                if probability <= 0.0:
                    continue
                if red_action == graph_id:
                    a[row, index[next_positions]] -= game.spec.gamma * probability
                else:
                    if ammo <= 0:
                        raise RuntimeError("Red changed graph with zero ammo.")
                    rhs[row] += (
                        game.spec.gamma
                        * probability
                        * lower_values[(red_action, ammo - 1)][next_positions]
                    )

    try:
        solution = np.linalg.solve(a, rhs)
    except np.linalg.LinAlgError as exc:
        raise RuntimeError(
            "Exact policy evaluation produced a singular linear system. "
            "For gamma=1 this requires SSP/proper-policy handling; the current "
            "PSRO exact oracle is intended for gamma<1."
        ) from exc

    return {
        position: float(solution[index[position]])
        for position in position_states
    }


def evaluate_policy_pair_exact(
    game: AdversarialGraphGame,
    blue_policy: TabularPolicy,
    red_policy: TabularPolicy,
    *,
    initial_graph: int,
    initial_ammo: int,
    initial_positions: tuple[int, ...],
) -> float:
    """Exact discounted value of one fixed Blue/Red stationary-policy pair."""

    if blue_policy.player != "blue" or red_policy.player != "red":
        raise ValueError("evaluate_policy_pair_exact expects Blue then Red policies.")
    if game.spec.gamma >= 1.0:
        raise ValueError(
            "Exact PSRO policy evaluation currently requires gamma < 1. "
            "Use a value such as 1-1e-9 for the SB setup."
        )

    positions = all_position_states(game.spec)
    values: dict[LayerKey, dict[PositionState, float]] = {}

    def blue_distribution(state: GameState):
        return blue_policy.distribution(game, state)

    def red_distribution(state: GameState):
        return red_policy.distribution(game, state)

    for ammo in range(game.spec.max_ammo + 1):
        for graph_id in game.spec.graph_costs:
            values[(graph_id, ammo)] = _solve_linear_layer(
                game,
                graph_id,
                ammo,
                positions,
                values,
                blue_distribution,
                red_distribution,
            )

    initial_positions = game.canonical_positions(initial_positions)
    return float(values[(initial_graph, initial_ammo)][initial_positions])


def _evaluate_blue_deterministic_policy_layer(
    game: AdversarialGraphGame,
    graph_id: int,
    ammo: int,
    positions: Sequence[PositionState],
    lower_values: Mapping[LayerKey, Mapping[PositionState, float]],
    actions_by_position: Mapping[PositionState, BlueAction],
    red_distribution,
) -> dict[PositionState, float]:
    def blue_distribution(state: GameState):
        legal = tuple(game.blue_actions(state))
        if game.is_terminal(state):
            probabilities = np.zeros(len(legal))
            probabilities[0] = 1.0
            return legal, probabilities
        chosen = actions_by_position[state.blue_positions]
        probabilities = np.zeros(len(legal))
        probabilities[legal.index(chosen)] = 1.0
        return legal, probabilities

    return _solve_linear_layer(
        game,
        graph_id,
        ammo,
        positions,
        lower_values,
        blue_distribution,
        red_distribution,
    )


def exact_blue_best_response_behavioral(
    game: AdversarialGraphGame,
    red_population: Sequence[TabularPolicy],
    red_weights: Sequence[float],
    *,
    initial_graph: int,
    initial_ammo: int,
    initial_positions: tuple[int, ...],
    tolerance: float = 1e-10,
    max_policy_iterations: int = 1_000,
    name: str = "blue_exact_br",
) -> BestResponseResult:
    """Exact stationary Blue BR to the *behavioral* mixture of Red policies.

    Important: this mixes the Red population action distributions at each
    Markov state. Standard normal-form PSRO usually samples one whole opponent
    policy at the start of an episode. The behavioral semantics are used here
    deliberately so that the Markov state remains sufficient and the oracle can
    be solved exactly for validation.
    """

    if game.spec.gamma >= 1.0:
        raise ValueError("Exact behavioral BR currently requires gamma < 1.")

    positions = all_position_states(game.spec)
    hops = _hop_distances_to_goal(game)
    values: dict[LayerKey, dict[PositionState, float]] = {}
    policy_map: dict[GameState, dict[Hashable, float]] = {}
    total_iterations = 0

    def red_distribution(state: GameState):
        return _mixture_distribution(
            game,
            state,
            red_population,
            red_weights,
            player="red",
        )

    for ammo in range(game.spec.max_ammo + 1):
        for graph_id in game.spec.graph_costs:
            actions_by_position: dict[PositionState, BlueAction] = {}
            for position in positions:
                state = GameState(position, graph_id, ammo)
                if game.is_terminal(state):
                    continue
                legal = game.blue_actions(state)
                actions_by_position[position] = min(
                    legal,
                    key=lambda action: (
                        sum(hops[node] for node in action),
                        game.stage_cost(state, action),
                        action,
                    ),
                )

            layer_values: dict[PositionState, float] | None = None
            for policy_iteration in range(1, max_policy_iterations + 1):
                total_iterations += 1
                layer_values = _evaluate_blue_deterministic_policy_layer(
                    game,
                    graph_id,
                    ammo,
                    positions,
                    values,
                    actions_by_position,
                    red_distribution,
                )
                stable = True

                for position in positions:
                    state = GameState(position, graph_id, ammo)
                    if game.is_terminal(state):
                        continue
                    red_actions, red_probs = red_distribution(state)
                    current_action = actions_by_position[position]

                    def q_value(blue_action: BlueAction) -> float:
                        q = game.stage_cost(state, blue_action)
                        next_position = game.canonical_positions(blue_action)
                        continuation = 0.0
                        for red_action, p_red in strict_zip(
                            red_actions, red_probs
                        ):
                            if red_action == graph_id:
                                next_value = layer_values[next_position]
                            else:
                                next_value = values[(red_action, ammo - 1)][
                                    next_position
                                ]
                            continuation += float(p_red) * next_value
                        return float(q + game.spec.gamma * continuation)

                    legal = game.blue_actions(state)
                    q_values = {action: q_value(action) for action in legal}
                    best_value = min(q_values.values())
                    current_value = q_values[current_action]
                    if current_value <= best_value + tolerance:
                        chosen = current_action
                    else:
                        chosen = min(
                            (
                                action
                                for action, value in q_values.items()
                                if value <= best_value + tolerance
                            ),
                            key=repr,
                        )
                    if chosen != current_action:
                        actions_by_position[position] = chosen
                        stable = False

                if stable:
                    break
            else:
                raise RuntimeError(
                    f"Blue policy iteration failed to converge in layer "
                    f"(graph={graph_id}, ammo={ammo})."
                )

            assert layer_values is not None
            # Re-evaluate the final improved policy if the last iteration changed it.
            layer_values = _evaluate_blue_deterministic_policy_layer(
                game,
                graph_id,
                ammo,
                positions,
                values,
                actions_by_position,
                red_distribution,
            )
            values[(graph_id, ammo)] = layer_values
            for position, action in actions_by_position.items():
                policy_map[GameState(position, graph_id, ammo)] = {action: 1.0}

    canonical_initial = game.canonical_positions(initial_positions)
    return BestResponseResult(
        policy=TabularPolicy("blue", policy_map, name=name),
        values=values,
        initial_value=float(values[(initial_graph, initial_ammo)][canonical_initial]),
        policy_iterations=total_iterations,
    )


def _evaluate_red_deterministic_policy_layer(
    game: AdversarialGraphGame,
    graph_id: int,
    ammo: int,
    positions: Sequence[PositionState],
    lower_values: Mapping[LayerKey, Mapping[PositionState, float]],
    actions_by_position: Mapping[PositionState, RedAction],
    blue_distribution,
) -> dict[PositionState, float]:
    def red_distribution(state: GameState):
        legal = tuple(game.red_actions(state))
        if game.is_terminal(state):
            probabilities = np.zeros(len(legal))
            probabilities[0] = 1.0
            return legal, probabilities
        chosen = actions_by_position[state.blue_positions]
        probabilities = np.zeros(len(legal))
        probabilities[legal.index(chosen)] = 1.0
        return legal, probabilities

    return _solve_linear_layer(
        game,
        graph_id,
        ammo,
        positions,
        lower_values,
        blue_distribution,
        red_distribution,
    )


def exact_red_best_response_behavioral(
    game: AdversarialGraphGame,
    blue_population: Sequence[TabularPolicy],
    blue_weights: Sequence[float],
    *,
    initial_graph: int,
    initial_ammo: int,
    initial_positions: tuple[int, ...],
    tolerance: float = 1e-10,
    max_policy_iterations: int = 1_000,
    name: str = "red_exact_br",
) -> BestResponseResult:
    """Exact stationary Red BR to the behavioral mixture of Blue policies."""

    if game.spec.gamma >= 1.0:
        raise ValueError("Exact behavioral BR currently requires gamma < 1.")

    positions = all_position_states(game.spec)
    values: dict[LayerKey, dict[PositionState, float]] = {}
    policy_map: dict[GameState, dict[Hashable, float]] = {}
    total_iterations = 0

    def blue_distribution(state: GameState):
        return _mixture_distribution(
            game,
            state,
            blue_population,
            blue_weights,
            player="blue",
        )

    for ammo in range(game.spec.max_ammo + 1):
        for graph_id in game.spec.graph_costs:
            actions_by_position: dict[PositionState, RedAction] = {}
            for position in positions:
                state = GameState(position, graph_id, ammo)
                if game.is_terminal(state):
                    continue
                legal = game.red_actions(state)
                actions_by_position[position] = (
                    graph_id if graph_id in legal else legal[0]
                )

            layer_values: dict[PositionState, float] | None = None
            for policy_iteration in range(1, max_policy_iterations + 1):
                total_iterations += 1
                layer_values = _evaluate_red_deterministic_policy_layer(
                    game,
                    graph_id,
                    ammo,
                    positions,
                    values,
                    actions_by_position,
                    blue_distribution,
                )
                stable = True

                for position in positions:
                    state = GameState(position, graph_id, ammo)
                    if game.is_terminal(state):
                        continue
                    blue_actions, blue_probs = blue_distribution(state)
                    current_action = actions_by_position[position]

                    def q_value(red_action: RedAction) -> float:
                        total = 0.0
                        for blue_action, p_blue in strict_zip(
                            blue_actions, blue_probs
                        ):
                            next_position = game.canonical_positions(blue_action)
                            if red_action == graph_id:
                                next_value = layer_values[next_position]
                            else:
                                next_value = values[(red_action, ammo - 1)][
                                    next_position
                                ]
                            total += float(p_blue) * (
                                game.stage_cost(state, blue_action)
                                + game.spec.gamma * next_value
                            )
                        return float(total)

                    legal = game.red_actions(state)
                    q_values = {action: q_value(action) for action in legal}
                    best_value = max(q_values.values())
                    current_value = q_values[current_action]
                    if current_value >= best_value - tolerance:
                        chosen = current_action
                    else:
                        chosen = min(
                            (
                                action
                                for action, value in q_values.items()
                                if value >= best_value - tolerance
                            ),
                            key=repr,
                        )
                    if chosen != current_action:
                        actions_by_position[position] = chosen
                        stable = False

                if stable:
                    break
            else:
                raise RuntimeError(
                    f"Red policy iteration failed to converge in layer "
                    f"(graph={graph_id}, ammo={ammo})."
                )

            assert layer_values is not None
            layer_values = _evaluate_red_deterministic_policy_layer(
                game,
                graph_id,
                ammo,
                positions,
                values,
                actions_by_position,
                blue_distribution,
            )
            values[(graph_id, ammo)] = layer_values
            for position, action in actions_by_position.items():
                policy_map[GameState(position, graph_id, ammo)] = {action: 1.0}

    canonical_initial = game.canonical_positions(initial_positions)
    return BestResponseResult(
        policy=TabularPolicy("red", policy_map, name=name),
        values=values,
        initial_value=float(values[(initial_graph, initial_ammo)][canonical_initial]),
        policy_iterations=total_iterations,
    )




def run_vanilla_psro_exact(
    game: AdversarialGraphGame,
    *,
    initial_graph: int,
    initial_ammo: int,
    initial_positions: tuple[int, ...],
    initial_blue_policy: TabularPolicy | None = None,
    initial_red_policy: TabularPolicy | None = None,
    max_iterations: int = 20,
    oracle_tolerance: float = 1e-8,
    lp_tolerance: float = 1e-9,
    verbose: bool = False,
) -> PSROResult:
    """Run a small-game PSRO validation loop with exact behavioral BR oracles.

    The restricted meta-game is standard PSRO: rows are complete Red policies,
    columns are complete Blue policies, and its Nash equilibrium is solved by LP.

    For tractable *exact* response computation, however, the opponent meta-policy
    is converted to a state-wise behavioral mixture. This is deliberately a
    validation oracle, not the final MARL oracle. A standard MARL PSRO oracle
    should instead sample one whole opponent policy per episode.
    """

    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive.")
    if game.spec.gamma >= 1.0:
        raise ValueError(
            "run_vanilla_psro_exact currently requires gamma < 1 for exact "
            "linear policy evaluation."
        )

    blue_population = [
        initial_blue_policy or make_proper_blue_policy(game)
    ]
    red_population = [
        initial_red_policy or make_stay_red_policy(game)
    ]
    history: list[PSROIteration] = []
    converged = False

    blue_signatures = {blue_population[0].signature()}
    red_signatures = {red_population[0].signature()}

    payoff_matrix = build_empirical_payoff_matrix(
        game,
        blue_population,
        red_population,
        initial_graph=initial_graph,
        initial_ammo=initial_ammo,
        initial_positions=initial_positions,
    )
    meta_solution = solve_zero_sum_matrix_game(
        payoff_matrix,
        tolerance=lp_tolerance,
    )

    for iteration in range(1, max_iterations + 1):
        meta_solution = solve_zero_sum_matrix_game(
            payoff_matrix,
            tolerance=lp_tolerance,
        )
        meta_value = float(meta_solution.value)

        blue_br = exact_blue_best_response_behavioral(
            game,
            red_population,
            meta_solution.red_policy,
            initial_graph=initial_graph,
            initial_ammo=initial_ammo,
            initial_positions=initial_positions,
            tolerance=oracle_tolerance,
            name=f"blue_br_{iteration}",
        )
        red_br = exact_red_best_response_behavioral(
            game,
            blue_population,
            meta_solution.blue_policy,
            initial_graph=initial_graph,
            initial_ammo=initial_ammo,
            initial_positions=initial_positions,
            tolerance=oracle_tolerance,
            name=f"red_br_{iteration}",
        )

        blue_improvement = max(0.0, meta_value - blue_br.initial_value)
        red_improvement = max(0.0, red_br.initial_value - meta_value)
        oracle_gap = max(0.0, red_br.initial_value - blue_br.initial_value)

        blue_signature = blue_br.policy.signature()
        red_signature = red_br.policy.signature()
        blue_added = blue_signature not in blue_signatures
        red_added = red_signature not in red_signatures

        history.append(
            PSROIteration(
                iteration=iteration,
                blue_population_size=len(blue_population),
                red_population_size=len(red_population),
                meta_value=meta_value,
                blue_meta_strategy=[float(x) for x in meta_solution.blue_policy],
                red_meta_strategy=[float(x) for x in meta_solution.red_policy],
                blue_br_value=float(blue_br.initial_value),
                red_br_value=float(red_br.initial_value),
                blue_improvement=float(blue_improvement),
                red_improvement=float(red_improvement),
                oracle_gap=float(oracle_gap),
                blue_added=blue_added,
                red_added=red_added,
            )
        )

        if verbose:
            print(
                f"PSRO iter={iteration:02d} "
                f"|Pi_B|={len(blue_population)} "
                f"|Pi_R|={len(red_population)} "
                f"meta={meta_value:.9f} "
                f"BR_B={blue_br.initial_value:.9f} "
                f"BR_R={red_br.initial_value:.9f} "
                f"gap={oracle_gap:.3e}",
                flush=True,
            )

        if (
            blue_improvement <= oracle_tolerance
            and red_improvement <= oracle_tolerance
        ):
            converged = True
            break

        if blue_added:
            blue_population.append(blue_br.policy)
            blue_signatures.add(blue_signature)
        if red_added:
            red_population.append(red_br.policy)
            red_signatures.add(red_signature)

        if not blue_added and not red_added:
            break

        payoff_matrix = build_empirical_payoff_matrix(
            game,
            blue_population,
            red_population,
            initial_graph=initial_graph,
            initial_ammo=initial_ammo,
            initial_positions=initial_positions,
        )

    payoff_matrix = build_empirical_payoff_matrix(
        game,
        blue_population,
        red_population,
        initial_graph=initial_graph,
        initial_ammo=initial_ammo,
        initial_positions=initial_positions,
    )
    meta_solution = solve_zero_sum_matrix_game(
        payoff_matrix,
        tolerance=lp_tolerance,
    )

    return PSROResult(
        blue_population=blue_population,
        red_population=red_population,
        payoff_matrix=payoff_matrix,
        blue_meta_strategy=meta_solution.blue_policy,
        red_meta_strategy=meta_solution.red_policy,
        meta_value=float(meta_solution.value),
        history=history,
        converged=converged,
    )

# --- Empirical PPO PSRO runner: numerical operations preserved from run_full_psro_long.py ---
import random
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from scenarios import build_scenario, get_scenario
from game.models import GameState, PruningMode
from game.state_space import all_position_states
from svi.layers import solve_ammo_layers
from svi.matrix_game import solve_zero_sum_matrix_game
from psro.meta_game_matrix import build_empirical_payoff_matrix
from experiment_config import (PROJECT_ROOT, safe_results_root, output_path,
    base_ppo_config, config_fingerprint, write_json_atomic)


def run_ppo_psro_experiment(condition, seed, args):
    # Import PPO only when training, so this file's policy classes can be used
    # independently of PPO without creating import-order-dependent cycles.
    from psro.ppo_oracle.red_ppo import PPOConfig, train_ppo_best_response
    from psro.ppo_oracle.blue_ppo import train_relgnn_blue_best_response
    from psro.initialization import start_populations
    scenario = get_scenario(args.scenario, num_blue_agents=getattr(args, "agents", 2))
    ammo = int(args.ammo)
    torch.set_num_threads(args.threads)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    game = build_scenario(scenario, max_ammo=ammo).build_game(with_support=True)
    blue_pop, red_pop, seed_stats = start_populations(game, condition, seed, scenario=scenario, ammo=ammo)
    blue_signatures = {p.signature(digits=7) for p in blue_pop}
    red_signatures = {p.signature(digits=7) for p in red_pop}
    config = {**asdict(base_ppo_config()), "total_steps": args.steps}
    fingerprint = config_fingerprint(
        scenario, condition=condition, seed=seed, steps=args.steps,
        iterations=args.iterations, blue_actor=args.blue_actor,
        blue_critic=args.blue_critic, ppo_config=config,
        support=True, ammo=ammo)
    result_file = output_path(args.output_root, "data", "psro", scenario.name,
                              f"{args.steps}steps", condition,
                              f"seed{seed}-{fingerprint}.json")
    # Memoize exact pair payoffs by immutable policy identity: old pairs do
    # not change when PSRO adds a new policy. This changes no game mathematics.
    pair_cache = {}
    def pair_value(game, blue, red):
        key = (id(blue), id(red))
        if key not in pair_cache:
            pair_cache[key] = float(evaluate_policy_pair_exact(
                game, blue, red, initial_graph=scenario.initial_graph,
                initial_ammo=ammo, initial_positions=scenario.initial_positions))
        return pair_cache[key]
    def payoff(game, blues, reds):
        matrix = np.empty((len(reds), len(blues)), dtype=float)
        for ri, rp in enumerate(reds):
            for bi, bp in enumerate(blues):
                matrix[ri,bi] = pair_value(game, bp, rp)
        return matrix
    svi = None
    if not args.skip_svi:
        exact = solve_ammo_layers(game, pruning_mode=PruningMode.NONE,
                                  tolerance=1e-9, max_iterations=5000)
        position = game.canonical_positions(scenario.initial_positions)
        svi = float(exact.values[(scenario.initial_graph, ammo)][position])

    # W&B tracking is strictly opt-in. Offline uses the same metrics as online;
    # default Python and shell runs work without installing or authenticating W&B.
    tracker = None
    wandb_enabled = bool(getattr(args, 'wandb', False)) and getattr(args, 'wandb_mode', 'online') != 'disabled'
    if wandb_enabled:
        import os
        # Keep ALL generated W&B files and caches inside results/, even when
        # launched from another working directory or with a different shell.
        wandb_directory = output_path(args.output_root, 'wandb', '.keep').parent
        wandb_cache = output_path(args.output_root, 'cache', 'wandb', '.keep').parent
        wandb_data = output_path(args.output_root, 'wandb', 'data', '.keep').parent
        os.environ['WANDB_DIR'] = str(wandb_directory)
        os.environ['WANDB_CACHE_DIR'] = str(wandb_cache)
        os.environ['WANDB_DATA_DIR'] = str(wandb_data)
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError(
                'W&B requested but not installed. Run: '
                'python -m pip install -r requirements-wandb.txt; '
                'or rerun without --wandb.'
            ) from exc
        run_group = (getattr(args, 'wandb_group', None) or
                     f'{scenario.name}-{args.steps}steps-{args.blue_actor}-{args.blue_critic}')
        run_kind = getattr(args, 'preset', 'psro')
        tracker = wandb.init(
            dir=str(wandb_directory),
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=f'{scenario.name}-{condition}-seed{seed}-{args.steps}steps-{args.blue_actor}-{args.blue_critic}',
            group=run_group,
            job_type=f'{run_kind}-psro',
            tags=['psro', run_kind, scenario.name, condition, f'blue_{args.blue_actor}', f'critic_{args.blue_critic}'],
            mode=args.wandb_mode,
            config={
                'condition': condition, 'seed': seed, 'steps_per_oracle': args.steps,
                'iterations': args.iterations, 'blue_actor': args.blue_actor,
                'blue_critic': args.blue_critic, 'red_actor': 'mlp',
                'red_critic': 'mlp', 'scenario': scenario.name,
                'ammo': ammo, 'with_support': True, 'ppo': config,
                'svi': svi, 'config_fingerprint': fingerprint,
            },
        )
        # The step metrics give W&B correct x-axes for PSRO iterations and
        # PPO updates; Blue and Red track separate cumulative training steps.
        wandb.define_metric('psro/iteration')
        wandb.define_metric('psro/*', step_metric='psro/iteration')
        for player in ('blue', 'red'):
            wandb.define_metric(f'ppo/{player}/global_steps')
            wandb.define_metric(f'ppo/{player}/*', step_metric=f'ppo/{player}/global_steps')

    started = time.perf_counter()
    history = []
    result = {
        "status": "running", "condition": condition, "seed": seed,
        "config_fingerprint": fingerprint, "result_path": str(result_file.relative_to(PROJECT_ROOT)),
        "scenario": scenario.name, "support": True,
        "blue_actor": args.blue_actor, "blue_critic": args.blue_critic,
        "red_oracle": "mlp PPO", "ppo_config": config,
        "ppo_steps_per_oracle": args.steps, "psro_iterations": args.iterations,
        "exact_svi": svi, "bc_seed_stats": seed_stats,
        "wandb": {"enabled": wandb_enabled, "mode": getattr(args, "wandb_mode", "disabled") if wandb_enabled else "disabled",
                  "project": args.wandb_project if wandb_enabled else None,
                  "run_id": getattr(tracker, "id", None) if tracker is not None else None},
        "initial_blue": [p.name for p in blue_pop],
        "initial_red": [p.name for p in red_pop], "history": history,
    }

    def save():
        write_json_atomic(result_file, result)

    try:
        mat = payoff(game, blue_pop, red_pop)
        solution = solve_zero_sum_matrix_game(mat, tolerance=1e-9)
        result["initial_value"] = float(solution.value)
        result["initial_blue_seed_pair_value"] = pair_value(game, blue_pop[0], red_pop[0])
        save()
        if tracker is not None:
            initial = {"psro/iteration": 0,
                       "psro/meta_value": float(solution.value),
                       "psro/blue_population": len(blue_pop),
                       "psro/red_population": len(red_pop)}
            if svi is not None:
                initial["psro/abs_gap_to_svi"] = abs(float(solution.value) - svi)
            tracker.log(initial)
        print("START", condition, "seed", seed, "initial", result["initial_value"], flush=True)

        for iteration in range(1, args.iterations + 1):
            iteration_start = time.perf_counter()
            mat = payoff(game, blue_pop, red_pop)
            solution = solve_zero_sum_matrix_game(mat, tolerance=1e-9)
            v = float(solution.value)
            blue_weights = np.asarray(solution.blue_policy, dtype=float)
            red_weights = np.asarray(solution.red_policy, dtype=float)

            def metric_callback(player):
                if tracker is None:
                    return None
                def emit(metrics):
                    s = int(metrics["steps"])
                    payload = {"ppo/%s/global_steps" % player: (iteration - 1) * args.steps + s,
                               "ppo/%s/local_steps" % player: s,
                               "ppo/%s/psro_iteration" % player: iteration}
                    payload.update({"ppo/%s/%s" % (player, key): val
                                    for key, val in metrics.items()
                                    if key != "steps" and val is not None})
                    tracker.log(payload)
                return emit

            blue_cfg = PPOConfig(**{**config, "seed": seed + iteration - 1})
            red_cfg = PPOConfig(**{**config, "seed": seed + 90 + iteration - 1})
            blue_policy, blue_stats, _ = train_relgnn_blue_best_response(
                game, actor_kind=args.blue_actor, critic_kind=args.blue_critic,
                opponent_population=red_pop, opponent_weights=red_weights,
                initial_graph=scenario.initial_graph, initial_ammo=ammo,
                initial_positions=scenario.initial_positions, config=blue_cfg,
                name="%s_seed%d_blue_br_%d" % (condition, seed, iteration),
                gnn_dim=16, metric_callback=metric_callback("blue"),
            )
            red_policy, red_stats = train_ppo_best_response(
                game, player="red", opponent_population=blue_pop,
                opponent_weights=blue_weights,
                initial_graph=scenario.initial_graph, initial_ammo=ammo,
                initial_positions=scenario.initial_positions, config=red_cfg,
                name="%s_seed%d_red_br_%d" % (condition, seed, iteration),
                metric_callback=metric_callback("red"),
            )
            blue_br = float(sum(float(weight) * pair_value(game, blue_policy, p)
                                for weight, p in zip(red_weights, red_pop)))
            red_br = float(sum(float(weight) * pair_value(game, p, red_policy)
                               for weight, p in zip(blue_weights, blue_pop)))
            blue_sig = blue_policy.signature(digits=7)
            red_sig = red_policy.signature(digits=7)
            blue_added = blue_sig not in blue_signatures
            red_added = red_sig not in red_signatures
            if blue_added:
                blue_pop.append(blue_policy)
                blue_signatures.add(blue_sig)
            if red_added:
                red_pop.append(red_policy)
                red_signatures.add(red_sig)
            mat = payoff(game, blue_pop, red_pop)
            updated = solve_zero_sum_matrix_game(mat, tolerance=1e-9)
            new_value = float(updated.value)
            row = {
                "iteration": iteration, "meta_before": v, "meta_after": new_value,
                "blue_BR": blue_br, "red_BR": red_br,
                "blue_gain": v-blue_br, "red_gain": red_br-v,
                "learned_BR_candidate_gap": red_br-blue_br,
                "blue_added": blue_added, "red_added": red_added,
                "blue_population": len(blue_pop), "red_population": len(red_pop),
                "blue_seconds": float(blue_stats.elapsed_seconds),
                "red_seconds": float(red_stats.elapsed_seconds),
                "iteration_seconds": time.perf_counter()-iteration_start,
                "elapsed_seconds": time.perf_counter()-started,
                "blue_names": [p.name for p in blue_pop],
                "red_names": [p.name for p in red_pop],
                "blue_mixture": [float(w) for w in updated.blue_policy],
                "red_mixture": [float(w) for w in updated.red_policy],
                "restricted_nash_gap": float(updated.strategy_gap),
            }
            history.append(row)
            result["elapsed_seconds"] = row["elapsed_seconds"]
            save()
            if tracker is not None:
                payload = {
                    "psro/iteration": iteration,
                    "psro/meta_value": new_value,
                    "psro/meta_before": v,
                    "psro/blue_best_response": blue_br,
                    "psro/red_best_response": red_br,
                    "psro/blue_gain": row["blue_gain"],
                    "psro/red_gain": row["red_gain"],
                    "psro/learned_br_candidate_gap": row["learned_BR_candidate_gap"],
                    "psro/blue_population": len(blue_pop),
                    "psro/red_population": len(red_pop),
                    "psro/restricted_nash_gap": row["restricted_nash_gap"],
                    "psro/elapsed_seconds": row["elapsed_seconds"],
                    "psro/blue_oracle_seconds": row["blue_seconds"],
                    "psro/red_oracle_seconds": row["red_seconds"],
                }
                if svi is not None:
                    payload["psro/abs_gap_to_svi"] = abs(new_value-svi)
                for player, weights in (("blue", updated.blue_policy),
                                        ("red", updated.red_policy)):
                    payload.update({"mixture/%s/policy_%03d" % (player,i): float(w)
                                    for i,w in enumerate(weights)})
                tracker.log(payload)
            print("ITER", condition, seed, iteration,
                  "V", round(v, 4), "B", round(blue_br, 4),
                  "R", round(red_br, 4), "new", round(new_value, 4),
                  "elapsed", round(row["elapsed_seconds"],1), flush=True)

        result.update({
            "status": "complete", "final": float(updated.value),
            "abs_gap_to_svi": abs(float(updated.value)-svi) if svi is not None else None,
            "total_seconds": time.perf_counter()-started,
            "final_blue_names": [p.name for p in blue_pop],
            "final_red_names": [p.name for p in red_pop],
            "final_blue_mixture": [float(w) for w in updated.blue_policy],
            "final_red_mixture": [float(w) for w in updated.red_policy],
            "payoff_matrix": mat.tolist(),
        })
        save()
        if tracker is not None:
            tracker.summary.update({"final_meta_value": result["final"],
                                    "total_seconds": result["total_seconds"]})
        print("DONE", condition, seed, "final", result["final"], flush=True)
    except BaseException as exc:
        result["status"] = "failed"
        result["error"] = repr(exc)
        save()
        raise
    finally:
        if tracker is not None:
            tracker.finish()

