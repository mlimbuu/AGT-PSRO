from __future__ import annotations

from game.compat import dataclass



import numpy as np

from game.models import PruningMode


@dataclass(slots=True)
class PruningResult:
    """Reduced exact payoff matrix and index maps."""

    matrix: np.ndarray
    retained_red_rows: tuple[int, ...]
    retained_blue_columns: tuple[int, ...]


@dataclass(slots=True)
class IntervalPruningResult:
    """Certified retained actions from lower/upper Q-value bounds.

    Bounds certify action elimination but are not the exact stage-game
    payoff matrix. Use :meth:`reduce` on the exact Q matrix before solving.
    """

    retained_red_rows: tuple[int, ...]
    retained_blue_columns: tuple[int, ...]

    def reduce(self, payoff_matrix: np.ndarray) -> np.ndarray:
        q = np.asarray(payoff_matrix, dtype=float)
        if q.ndim != 2:
            raise ValueError("payoff_matrix must be two-dimensional.")
        if self.retained_red_rows and max(self.retained_red_rows) >= q.shape[0]:
            raise ValueError("Retained Red row index is outside payoff_matrix.")
        if (
            self.retained_blue_columns
            and max(self.retained_blue_columns) >= q.shape[1]
        ):
            raise ValueError("Retained Blue column index is outside payoff_matrix.")
        return q[np.ix_(self.retained_red_rows, self.retained_blue_columns)]


def normalize_pruning_mode(
    pruning_mode: PruningMode | str = PruningMode.NONE,
    *,
    prune_actions: bool | None = None,
) -> PruningMode:
    """Normalize the explicit mode and legacy Boolean option.

    ``prune_actions=True`` maps to temporary strict pruning, matching the
    behavior of earlier package versions. New code should use
    ``pruning_mode`` directly.
    """

    mode = (
        pruning_mode
        if isinstance(pruning_mode, PruningMode)
        else PruningMode(str(pruning_mode))
    )

    if prune_actions is None:
        return mode

    legacy_mode = (
        PruningMode.TEMPORARY_STRICT
        if prune_actions
        else PruningMode.NONE
    )
    if mode is not PruningMode.NONE and mode is not legacy_mode:
        raise ValueError(
            "Conflicting pruning_mode and legacy prune_actions arguments."
        )
    return legacy_mode


def _blue_column_is_strictly_dominated(
    q: np.ndarray,
    candidate: int,
    dominator: int,
    tolerance: float,
) -> bool:
    """Blue minimizes: candidate is strictly costlier in every row."""

    return bool(np.all(q[:, candidate] > q[:, dominator] + tolerance))


def _red_row_is_strictly_dominated(
    q: np.ndarray,
    candidate: int,
    dominator: int,
    tolerance: float,
) -> bool:
    """Red maximizes: candidate is strictly smaller in every column."""

    return bool(np.all(q[candidate, :] < q[dominator, :] - tolerance))


def prune_strictly_dominated_actions(
    payoff_matrix: np.ndarray,
    *,
    tolerance: float = 1e-10,
) -> PruningResult:
    """Temporarily eliminate strictly pure-strategy dominated actions.

    The full current Q matrix should be rebuilt and this function rerun on
    every value-iteration sweep because continuation values change.
    """

    q = np.asarray(payoff_matrix, dtype=float)
    if q.ndim != 2 or min(q.shape) == 0:
        raise ValueError("payoff_matrix must be a nonempty matrix.")
    if not np.all(np.isfinite(q)):
        raise ValueError("payoff_matrix contains NaN or infinity.")

    red_rows = list(range(q.shape[0]))
    blue_columns = list(range(q.shape[1]))

    changed = True
    while changed:
        changed = False
        reduced = q[np.ix_(red_rows, blue_columns)]

        if len(blue_columns) > 1:
            removed = False
            for candidate in range(reduced.shape[1]):
                for dominator in range(reduced.shape[1]):
                    if candidate == dominator:
                        continue
                    if _blue_column_is_strictly_dominated(
                        reduced, candidate, dominator, tolerance
                    ):
                        del blue_columns[candidate]
                        changed = True
                        removed = True
                        break
                if removed:
                    break
            if changed:
                continue

        reduced = q[np.ix_(red_rows, blue_columns)]
        if len(red_rows) > 1:
            removed = False
            for candidate in range(reduced.shape[0]):
                for dominator in range(reduced.shape[0]):
                    if candidate == dominator:
                        continue
                    if _red_row_is_strictly_dominated(
                        reduced, candidate, dominator, tolerance
                    ):
                        del red_rows[candidate]
                        changed = True
                        removed = True
                        break
                if removed:
                    break

    return PruningResult(
        matrix=q[np.ix_(red_rows, blue_columns)],
        retained_red_rows=tuple(red_rows),
        retained_blue_columns=tuple(blue_columns),
    )


def prune_interval_dominated_actions(
    q_lower: np.ndarray,
    q_upper: np.ndarray,
    *,
    tolerance: float = 1e-10,
) -> IntervalPruningResult:
    """Certify pure-action dominance using conservative Q intervals.

    Blue column ``c`` is removed when another column ``d`` satisfies

        lower[:, c] > upper[:, d]

    against every retained Red action. Red uses the symmetric maximizer
    condition. The returned mask is permanent for the bounds used to create
    it, because dominance is proven for every exact Q matrix inside those
    intervals.
    """

    lower = np.asarray(q_lower, dtype=float)
    upper = np.asarray(q_upper, dtype=float)
    if lower.shape != upper.shape:
        raise ValueError("q_lower and q_upper must have equal shape.")
    if lower.ndim != 2 or min(lower.shape) == 0:
        raise ValueError("Q bounds must be nonempty matrices.")
    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
        raise ValueError("Q bounds contain NaN or infinity.")
    if np.any(lower > upper + tolerance):
        raise ValueError("Some lower bounds exceed upper bounds.")

    red_rows = list(range(lower.shape[0]))
    blue_columns = list(range(lower.shape[1]))
    changed = True

    while changed:
        changed = False

        if len(blue_columns) > 1:
            removed = False
            for c_local, candidate in enumerate(blue_columns):
                for dominator in blue_columns:
                    if candidate == dominator:
                        continue
                    candidate_lower = lower[
                        np.ix_(red_rows, [candidate])
                    ][:, 0]
                    dominator_upper = upper[
                        np.ix_(red_rows, [dominator])
                    ][:, 0]
                    if np.all(
                        candidate_lower > dominator_upper + tolerance
                    ):
                        del blue_columns[c_local]
                        changed = True
                        removed = True
                        break
                if removed:
                    break
            if changed:
                continue

        if len(red_rows) > 1:
            removed = False
            for r_local, candidate in enumerate(red_rows):
                for dominator in red_rows:
                    if candidate == dominator:
                        continue
                    candidate_upper = upper[
                        np.ix_([candidate], blue_columns)
                    ][0]
                    dominator_lower = lower[
                        np.ix_([dominator], blue_columns)
                    ][0]
                    if np.all(
                        candidate_upper < dominator_lower - tolerance
                    ):
                        del red_rows[r_local]
                        changed = True
                        removed = True
                        break
                if removed:
                    break

    return IntervalPruningResult(
        retained_red_rows=tuple(red_rows),
        retained_blue_columns=tuple(blue_columns),
    )


def compose_retained_indices(
    outer_indices: tuple[int, ...],
    inner_indices: tuple[int, ...],
) -> tuple[int, ...]:
    """Map indices retained in a reduced matrix back to the full matrix."""

    return tuple(outer_indices[index] for index in inner_indices)


def restore_full_policy(
    reduced_policy: np.ndarray,
    retained_indices: tuple[int, ...],
    full_size: int,
) -> np.ndarray:
    full = np.zeros(full_size, dtype=float)
    full[np.asarray(retained_indices, dtype=int)] = reduced_policy
    return full
