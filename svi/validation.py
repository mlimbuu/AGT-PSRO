from __future__ import annotations

from game.compat import dataclass



from game.environment import AdversarialGraphGame
from svi.layers import AmmoLayerResult, solve_ammo_layers
from game.models import PruningMode
from svi.shapley import shapley_value_iteration


@dataclass(frozen=True, slots=True)
class SolverCrossCheck:
    max_full_vs_layer_difference: float
    max_mode_difference: dict[PruningMode, float]
    full_iterations: int
    full_residual: float
    all_layers_converged: bool

    @property
    def max_pruned_vs_unpruned_difference(self) -> float:
        return max(self.max_mode_difference.values(), default=0.0)


def max_layer_value_difference(
    first: AmmoLayerResult,
    second: AmmoLayerResult,
) -> float:
    difference = 0.0
    if first.values.keys() != second.values.keys():
        raise ValueError("Layer results have different layer keys.")
    for key, first_values in first.values.items():
        second_values = second.values[key]
        if first_values.keys() != second_values.keys():
            raise ValueError(f"Layer {key} has different position states.")
        for positions, first_value in first_values.items():
            difference = max(
                difference,
                abs(first_value - second_values[positions]),
            )
    return float(difference)


def cross_check_solvers(
    game: AdversarialGraphGame,
    *,
    tolerance: float = 1e-9,
    max_iterations: int = 10_000,
    modes: tuple[PruningMode, ...] = (
        PruningMode.TEMPORARY_STRICT,
        PruningMode.BOUND_CERTIFIED,
        PruningMode.HYBRID,
    ),
) -> SolverCrossCheck:
    """Cross-check full-state/layer equivalence and pruning invariance."""

    full = shapley_value_iteration(
        game,
        tolerance=tolerance,
        max_iterations=max_iterations,
        pruning_mode=PruningMode.NONE,
    )
    baseline = solve_ammo_layers(
        game,
        tolerance=tolerance,
        max_iterations=max_iterations,
        pruning_mode=PruningMode.NONE,
    )

    full_vs_layer = 0.0
    for state, value in full.values.items():
        layer_value = baseline.values[
            (state.graph_id, state.red_ammo)
        ][state.blue_positions]
        full_vs_layer = max(full_vs_layer, abs(value - layer_value))

    mode_differences: dict[PruningMode, float] = {}
    all_converged = all(baseline.converged.values())
    for mode in modes:
        result = solve_ammo_layers(
            game,
            tolerance=tolerance,
            max_iterations=max_iterations,
            pruning_mode=mode,
        )
        mode_differences[mode] = max_layer_value_difference(
            baseline, result
        )
        all_converged = all_converged and all(
            result.converged.values()
        )

    return SolverCrossCheck(
        max_full_vs_layer_difference=float(full_vs_layer),
        max_mode_difference=mode_differences,
        full_iterations=full.iterations,
        full_residual=full.residual,
        all_layers_converged=all_converged,
    )
