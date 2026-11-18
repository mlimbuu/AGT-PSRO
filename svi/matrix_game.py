from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

from game.models import MatrixGameSolution


def _normalize_probability_vector(
    probabilities: np.ndarray,
    tolerance: float,
) -> np.ndarray:
    cleaned = np.asarray(probabilities, dtype=float).copy()
    cleaned[np.abs(cleaned) < tolerance] = 0.0
    cleaned = np.clip(cleaned, 0.0, None)
    total = float(cleaned.sum())
    if total <= tolerance:
        raise RuntimeError("LP returned a degenerate probability vector.")
    return cleaned / total


def solve_zero_sum_matrix_game(
    payoff_matrix: np.ndarray,
    *,
    tolerance: float = 1e-9,
) -> MatrixGameSolution:
    """Solve a finite zero-sum cost matrix with independent primal/dual LPs.

    Rows are Red actions and columns are Blue actions. Red maximizes cost;
    Blue minimizes it. In addition to the LP duality gap, the returned mixed
    policies are checked directly through their security values.
    """

    q = np.asarray(payoff_matrix, dtype=float)
    if q.ndim != 2:
        raise ValueError("payoff_matrix must be two-dimensional.")
    if q.shape[0] == 0 or q.shape[1] == 0:
        raise ValueError("payoff_matrix must have at least one row and column.")
    if not np.all(np.isfinite(q)):
        raise ValueError("payoff_matrix contains NaN or infinity.")

    n_red, n_blue = q.shape
    if n_red == 1 and n_blue == 1:
        value = float(q[0, 0])
        return MatrixGameSolution(
            value=value,
            blue_policy=np.array([1.0]),
            red_policy=np.array([1.0]),
            blue_value_upper=value,
            red_value_lower=value,
            duality_gap=0.0,
            blue_security_value=value,
            red_security_value=value,
            strategy_gap=0.0,
        )

    # Blue primal:
    #   min v
    #   s.t. Q x <= v 1, 1^T x = 1, x >= 0.
    blue_result = linprog(
        c=np.concatenate([np.zeros(n_blue), [1.0]]),
        A_ub=np.hstack([q, -np.ones((n_red, 1))]),
        b_ub=np.zeros(n_red),
        A_eq=np.concatenate([np.ones(n_blue), [0.0]])[None, :],
        b_eq=np.array([1.0]),
        bounds=[(0.0, None)] * n_blue + [(None, None)],
        method="highs",
    )
    if not blue_result.success:
        raise RuntimeError(f"Blue minimax LP failed: {blue_result.message}")

    # Red dual:
    #   max w
    #   s.t. y^T Q >= w 1^T, 1^T y = 1, y >= 0.
    red_result = linprog(
        c=np.concatenate([np.zeros(n_red), [-1.0]]),
        A_ub=np.hstack([-q.T, np.ones((n_blue, 1))]),
        b_ub=np.zeros(n_blue),
        A_eq=np.concatenate([np.ones(n_red), [0.0]])[None, :],
        b_eq=np.array([1.0]),
        bounds=[(0.0, None)] * n_red + [(None, None)],
        method="highs",
    )
    if not red_result.success:
        raise RuntimeError(f"Red maximin LP failed: {red_result.message}")

    blue_policy = _normalize_probability_vector(
        blue_result.x[:n_blue], tolerance
    )
    red_policy = _normalize_probability_vector(
        red_result.x[:n_red], tolerance
    )

    blue_value_upper = float(blue_result.x[-1])
    red_value_lower = float(red_result.x[-1])
    duality_gap = blue_value_upper - red_value_lower

    numerical_tolerance = max(1e-8, 100.0 * tolerance)
    if duality_gap < -numerical_tolerance:
        raise RuntimeError(
            "LP duality order was violated: "
            f"Blue upper={blue_value_upper}, Red lower={red_value_lower}."
        )

    # Direct policy verification after numerical cleanup/normalization.
    # Blue's mixed policy guarantees no more than max_r (Qx)_r.
    blue_security_value = float(np.max(q @ blue_policy))
    # Red's mixed policy guarantees at least min_b (y^TQ)_b.
    red_security_value = float(np.min(red_policy @ q))
    strategy_gap = blue_security_value - red_security_value

    if strategy_gap < -numerical_tolerance:
        raise RuntimeError(
            "Computed policy security values have invalid order: "
            f"Blue={blue_security_value}, Red={red_security_value}."
        )
    if strategy_gap > numerical_tolerance:
        raise RuntimeError(
            "LP policies fail the direct minimax security check: "
            f"gap={strategy_gap}."
        )

    value = 0.5 * (blue_security_value + red_security_value)

    return MatrixGameSolution(
        value=float(value),
        blue_policy=blue_policy,
        red_policy=red_policy,
        blue_value_upper=blue_value_upper,
        red_value_lower=red_value_lower,
        duality_gap=max(0.0, float(duality_gap)),
        blue_security_value=blue_security_value,
        red_security_value=red_security_value,
        strategy_gap=max(0.0, float(strategy_gap)),
    )


def expected_matrix_value(
    payoff_matrix: np.ndarray,
    red_policy: np.ndarray,
    blue_policy: np.ndarray,
) -> float:
    return float(
        np.asarray(red_policy, dtype=float)
        @ np.asarray(payoff_matrix, dtype=float)
        @ np.asarray(blue_policy, dtype=float)
    )
