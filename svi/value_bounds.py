"""Report certified lower/upper bounds on the SVI game value.

Uses the existing bound-propagation code in svi.layers, independently of
whether action pruning is enabled. Requires a solved lower-ammo layer result.

The bounds are *value bounds*, not exploitability/security certificates for
an arbitrary learned policy.
"""
from __future__ import annotations

from dataclasses import dataclass

from game.environment import AdversarialGraphGame
from game.models import PruningStats
from game.state_space import all_position_states
from svi.layers import AmmoLayerResult, _compute_layer_bound_plan


@dataclass(frozen=True)
class ValueBounds:
    lower: float
    upper: float
    gap: float
    sweeps: int


def initial_value_bounds(
    game: AdversarialGraphGame,
    svi_result: AmmoLayerResult,
    *,
    initial_graph: int,
    initial_positions: tuple[int, ...],
    ammo: int,
    max_sweeps: int = 200,
    tolerance: float = 1e-8,
) -> ValueBounds:
    """Bound V*(initial state) using pure security Bellman updates.

    Previous ammo-layer continuations come from exact SVI. Lower starts at 0,
    upper starts from a proper goal-reaching Blue policy. The monotone updates
    preserve lower <= V* <= upper under the game's nonnegative-cost and
    finite proper-policy assumptions.
    """
    positions = game.canonical_positions(initial_positions)
    key = (initial_graph, ammo)
    if key not in svi_result.values:
        raise ValueError(f"SVI result does not include layer {key}")
    if not svi_result.converged.get(key, False):
        raise ValueError("SVI layer must be converged before using this wrapper")
    for k, ok in svi_result.converged.items():
        if not ok:
            raise ValueError(f"Unconverged lower-ammo continuation layer: {k}")

    if ammo == 0 and game.spec.gamma == 1.0:
        v = float(svi_result.values[key][positions])
        return ValueBounds(v, v, 0.0, 0)

    plan = _compute_layer_bound_plan(
        game,
        initial_graph,
        ammo,
        all_position_states(game.spec),
        svi_result.values,
        max_sweeps=max_sweeps,
        bound_tolerance=tolerance,
        dominance_tolerance=1e-10,
        stats=PruningStats(),
    )
    lower = float(plan.lower_values[positions])
    upper = float(plan.upper_values[positions])
    v = float(svi_result.values[key][positions])
    # Tolerance allows tiny LP and VI numerical deviations.
    if lower > v + 1e-5 or upper < v - 1e-5:
        raise RuntimeError(
            f"Bound verification failed: lower={lower}, SVI={v}, upper={upper}"
        )
    return ValueBounds(lower, upper, upper - lower, plan.sweeps)
