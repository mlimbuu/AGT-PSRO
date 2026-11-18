from __future__ import annotations

from game.compat import strict_zip

from game.costs import EdgeTraversalCost
from game.environment import AdversarialGraphGame
from game.models import GameSpec, GameState


def make_berneburg_matlab_reference_game(
    *,
    gamma: float = 1.0 - 1e-9,
    max_ammo: int = 4,
) -> AdversarialGraphGame:
    """Recreate the complete-information example in ``AGTsetup.m``.

    MATLAB graph indices 1, 2, 3 are represented here as Python graph IDs
    0, 1, 2. The physical graph is undirected and is expanded into symmetric
    directed arcs. Only the self-loops explicitly present in the MATLAB setup
    are included: nodes 1, 4, 7, and terminal node 10.
    """

    in_nodes = (
        1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4, 5, 6, 6, 7, 7, 8, 9, 10
    )
    out_nodes = (
        1, 2, 3, 4, 5, 6, 7, 8, 4, 5, 10, 10, 7, 10, 7, 10, 9, 10, 10
    )

    e_self = 1.0
    e_light = 2.0
    e_mid = 4.0
    e_heavy = 8.0

    edge_weight_rows = (
        (
            e_self, e_heavy, e_heavy, e_light, e_light, e_light,
            e_light, e_light, e_self, e_light, e_heavy, e_mid,
            e_mid, e_light, e_self, e_light, e_heavy, e_light, 0.0,
        ),
        (
            e_self, e_mid, e_light, e_mid, e_mid, e_mid,
            e_mid, e_mid, e_self, e_mid, e_light, e_heavy,
            e_light, e_mid, e_self, e_light, e_heavy, e_light, 0.0,
        ),
        (
            e_self, e_light, e_mid, e_mid, e_mid, e_mid,
            e_mid, e_mid, e_self, e_light, e_mid, e_light,
            e_light, e_heavy, e_self, e_heavy, e_light, e_heavy, 0.0,
        ),
    )

    nodes = tuple(range(1, 11))
    adjacency_sets = {node: set() for node in nodes}
    graph_costs = {graph_id: {} for graph_id in range(3)}

    for edge_index, (node_i, node_j) in enumerate(
        strict_zip(in_nodes, out_nodes)
    ):
        adjacency_sets[node_i].add(node_j)
        adjacency_sets[node_j].add(node_i)

        for graph_id, weights in enumerate(edge_weight_rows):
            weight = float(weights[edge_index])
            graph_costs[graph_id][(node_i, node_j)] = weight
            graph_costs[graph_id][(node_j, node_i)] = weight

    # The MATLAB adjacency conversion drops the zero-weight goal loop and
    # then adds it back explicitly.
    adjacency_sets[10].add(10)
    for graph_id in graph_costs:
        graph_costs[graph_id][(10, 10)] = 0.0

    adjacency = {
        node: tuple(sorted(successors))
        for node, successors in adjacency_sets.items()
    }
    red_action_graph = {
        graph_id: (0, 1, 2) for graph_id in range(3)
    }

    spec = GameSpec(
        nodes=nodes,
        adjacency=adjacency,
        graph_costs=graph_costs,
        red_action_graph=red_action_graph,
        goal_node=10,
        num_blue_agents=1,
        max_ammo=max_ammo,
        gamma=gamma,
    )
    return AdversarialGraphGame(spec=spec, cost_model=EdgeTraversalCost())


def matlab_state(
    *,
    position: int,
    graph_index: int,
    ammo: int,
) -> GameState:
    """Convert a 1-based MATLAB graph index to our Python state."""

    if graph_index < 1:
        raise ValueError("MATLAB graph_index must be at least 1.")
    return GameState(
        blue_positions=(position,),
        graph_id=graph_index - 1,
        red_ammo=ammo,
    )
