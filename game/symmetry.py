from __future__ import annotations

from game.compat import strict_zip

from collections import defaultdict
from typing import Iterable

from game.models import BlueAction, Node


def canonicalize_positions(
    positions: Iterable[Node],
) -> tuple[Node, ...]:
    """Canonical representative for homogeneous, indistinguishable agents.

    The quotient state stores only the multiset of occupied nodes. Sorting is
    sufficient because every Blue agent has the same action set and cost model.
    """

    return tuple(sorted(positions))


def canonicalize_action_within_position_groups(
    current_positions: tuple[Node, ...],
    action: BlueAction,
) -> BlueAction:
    """Remove only permutations of agents sharing the same current node.

    Agents at different current nodes are *not* interchangeable for the current
    transition because swapping their successors can change the traversed edges
    and stage cost. Agents occupying the same node are interchangeable, so their
    chosen successors can be sorted within that equal-position group.
    """

    if len(current_positions) != len(action):
        raise ValueError("Action dimension does not match current positions.")

    grouped_indices: dict[Node, list[int]] = defaultdict(list)
    for index, node in enumerate(current_positions):
        grouped_indices[node].append(index)

    canonical = list(action)
    for indices in grouped_indices.values():
        if len(indices) <= 1:
            continue
        successors = sorted(action[index] for index in indices)
        for index, successor in strict_zip(indices, successors):
            canonical[index] = successor

    return tuple(canonical)


def unique_actions_mod_agent_symmetry(
    current_positions: tuple[Node, ...],
    actions: Iterable[BlueAction],
) -> tuple[BlueAction, ...]:
    """Return one action representative per stabilizer-group orbit."""

    representatives: dict[BlueAction, BlueAction] = {}
    for action in actions:
        canonical = canonicalize_action_within_position_groups(
            current_positions, action
        )
        representatives.setdefault(canonical, canonical)
    return tuple(sorted(representatives))
