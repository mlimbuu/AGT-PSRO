"""Restricted PSRO meta-game: Red rows, Blue columns, exact pair payoffs."""
from __future__ import annotations
from typing import Sequence
import numpy as np
from game.environment import AdversarialGraphGame
from svi.matrix_game import solve_zero_sum_matrix_game

def build_empirical_payoff_matrix(
    game: AdversarialGraphGame,
    blue_population: Sequence[TabularPolicy],
    red_population: Sequence[TabularPolicy],
    *,
    initial_graph: int,
    initial_ammo: int,
    initial_positions: tuple[int, ...],
) -> np.ndarray:
    """Return Red-rows x Blue-columns exact policy-pair costs."""

    from psro.psro import evaluate_policy_pair_exact
    matrix = np.empty((len(red_population), len(blue_population)), dtype=float)
    for red_index, red_policy in enumerate(red_population):
        for blue_index, blue_policy in enumerate(blue_population):
            matrix[red_index, blue_index] = evaluate_policy_pair_exact(
                game,
                blue_policy,
                red_policy,
                initial_graph=initial_graph,
                initial_ammo=initial_ammo,
                initial_positions=initial_positions,
            )
    return matrix


def solve_meta_game(matrix: np.ndarray, *, tolerance: float = 1e-9):
    """Solve the restricted zero-sum game: Red maximizes, Blue minimizes."""
    return solve_zero_sum_matrix_game(matrix,tolerance=tolerance)
