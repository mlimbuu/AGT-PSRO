"""Policy-population initialization: scratch, BC, heuristic. Not PPO weight warm starts."""
from __future__ import annotations
from collections import deque
import heapq
from time import perf_counter
import random
import numpy as np
import torch
from torch import nn
from game.models import GameState, BlueAction
from game.state_space import all_position_states
from psro.psro import TabularPolicy, make_stay_red_policy, evaluate_policy_pair_exact, exact_blue_best_response_behavioral
from psro.ppo_oracle.red_ppo import PPOConfig
from psro.ppo_oracle.relational_gnn import RelationalGraphBuilder, StaticRelGNNActor
BC_CFG=PPOConfig(total_steps=6000, rollout_steps=512, max_episode_steps=32,
    hidden_size=64,learning_rate=3e-4,update_epochs=4,minibatch_size=128,
    entropy_coef=0.01,entropy_coef_final=None,reward_scale=0.01,seed=11,
    include_graph_feature=True,include_ammo_feature=True,
    include_red_reachability_feature=True)

def random_proper_blue_seed(game, seed):
    """Random tie-breaking among goal-progress actions, without risk/support planning.

    PSRO needs a legal policy to populate its first restricted matrix. The
    shortest-hop constraint ensures eventual arrival; it is not a risk-aware
    or support-aware policy, and it is NOT a learned PPO oracle.
    """
    rng = random.Random(seed)
    distances = {game.spec.goal_node: 0}
    queue = deque([game.spec.goal_node])
    while queue:
        node = queue.popleft()
        for adjacent in game.spec.adjacency[node]:
            if adjacent not in distances:
                distances[adjacent] = distances[node] + 1
                queue.append(adjacent)
    mapping = {}
    for ammo in range(game.spec.max_ammo + 1):
        for graph_id in game.spec.graph_costs:
            for positions in all_position_states(game.spec):
                state = GameState(positions, graph_id, ammo)
                if game.is_terminal(state):
                    continue
                legal = game.blue_actions(state)
                best = min(sum(distances[node] for node in action) for action in legal)
                candidates = [action for action in legal
                              if sum(distances[node] for node in action) == best]
                chosen = rng.choice(candidates)
                mapping[state] = {chosen: 1.0}
    return TabularPolicy("blue", mapping, name="blue_random_proper_seed")


def fixed_snapshot_shortest_values(game, graph_id: int) -> dict[PositionState, float]:
    """Exact deterministic team shortest-path values if Red freezes graph_id forever.

    This is deliberately *not* a game solve. It ignores Red graph-transition dynamics
    and ammo. Support, if present in ``game``, is handled by the actual stage-cost
    function, including the supporter-must-wait semantics.
    """
    positions = tuple(all_position_states(game.spec))
    terminal = game.canonical_positions((game.spec.goal_node,) * game.spec.num_blue_agents)

    reverse: dict[PositionState, list[tuple[PositionState, float, BlueAction]]] = {
        p: [] for p in positions
    }

    # Ammo is irrelevant for a frozen graph because stage cost depends only on the
    # current graph/positions/action. Use ammo=0 merely to form a valid state.
    for p in positions:
        state = GameState(p, graph_id, 0)
        if game.is_terminal(state):
            continue
        for action in game.blue_actions(state):
            q = game.canonical_positions(action)
            c = float(game.stage_cost(state, action))
            # Self-loops are harmless for positive waiting costs but are not needed
            # for shortest paths to the absorbing goal.
            if q == p:
                continue
            reverse[q].append((p, c, action))

    dist = {p: float('inf') for p in positions}
    dist[terminal] = 0.0
    pq: list[tuple[float, PositionState]] = [(0.0, terminal)]
    while pq:
        d, q = heapq.heappop(pq)
        if d > dist[q] + 1e-12:
            continue
        for p, c, _ in reverse[q]:
            nd = d + c
            if nd + 1e-12 < dist[p]:
                dist[p] = nd
                heapq.heappush(pq, (nd, p))

    bad = [p for p, d in dist.items() if d == float('inf')]
    if bad:
        raise RuntimeError(f"Frozen-graph heuristic has states that cannot reach goal: {bad[:5]}")
    return dist


def build_snapshot_blue_heuristic(game, *, name: str) -> tuple[TabularPolicy, dict[int, dict[PositionState, float]]]:
    """Current-snapshot shortest-path policy.

    At each state, Blue pretends the current risk snapshot stays fixed forever and
    takes the first joint action of the exact deterministic shortest path under that
    snapshot. It observes current G, positions and support relations, but *ignores*
    Red transition dynamics and ammo when planning.
    """
    frozen = {
        graph_id: fixed_snapshot_shortest_values(game, graph_id)
        for graph_id in game.spec.graph_costs
    }
    mapping = {}
    for ammo in range(game.spec.max_ammo + 1):
        for graph_id in game.spec.graph_costs:
            values = frozen[graph_id]
            for p in all_position_states(game.spec):
                state = GameState(p, graph_id, ammo)
                if game.is_terminal(state):
                    continue
                legal = game.blue_actions(state)
                chosen = min(
                    legal,
                    key=lambda a: (
                        float(game.stage_cost(state, a)) + values[game.canonical_positions(a)],
                        repr(a),
                    ),
                )
                mapping[state] = {chosen: 1.0}
    return TabularPolicy('blue', mapping, name=name), frozen


def greedy_action(policy: TabularPolicy, state: GameState):
    probs = policy.action_probabilities[state]
    return max(probs, key=probs.get)


def train_static_relgnn_bc_seed(game, red_seed: TabularPolicy, seed: int = 11, *, scenario, ammo: int = 1):
    """BC clone of exact Blue BR to the initial stay-Red seed.

    This affects only Pi_B^0. Later PSRO Blue BRs are random-init PPO as usual.
    """
    exact = exact_blue_best_response_behavioral(
        game, [red_seed], [1.0],
        initial_graph=scenario.initial_graph,
        initial_ammo=ammo,
        initial_positions=scenario.initial_positions,
    )
    teacher = exact.policy
    states = sorted(
        [s for s in teacher.action_probabilities if not game.is_terminal(s)],
        key=lambda s: (s.red_ammo, s.graph_id, s.blue_positions),
    )

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.set_num_threads(1)
    graph = RelationalGraphBuilder(game, BC_CFG)
    actor = StaticRelGNNActor(graph, d=16)
    opt = torch.optim.Adam(actor.parameters(), lr=3e-3)
    examples = []
    for s in states:
        legal = game.blue_actions(s)
        ta = greedy_action(teacher, s)
        examples.append((s, legal, legal.index(ta)))

    started = perf_counter()
    final_acc = 0.0
    final_loss = float('nan')
    epochs = 0
    for epochs in range(1, 81):
        random.shuffle(examples)
        actor.train(); opt.zero_grad(); pending = 0
        for idx, (s, legal, label) in enumerate(examples, start=1):
            logits = actor.logits(s, legal)
            loss = nn.functional.cross_entropy(logits.unsqueeze(0), torch.tensor([label]))
            (loss / 16.0).backward(); pending += 1
            if pending == 16 or idx == len(examples):
                nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                opt.step(); opt.zero_grad(); pending = 0
        if epochs % 10 == 0 or epochs == 80:
            actor.eval(); correct = 0; losses = []
            with torch.no_grad():
                for s, legal, label in examples:
                    logits = actor.logits(s, legal)
                    losses.append(float(nn.functional.cross_entropy(logits.unsqueeze(0), torch.tensor([label])).item()))
                    correct += int(int(torch.argmax(logits).item()) == label)
            final_acc = correct / len(examples)
            final_loss = float(np.mean(losses))
            if final_acc == 1.0 and final_loss < 1e-3:
                break

    mapping = {}
    actor.eval()
    with torch.no_grad():
        for s in states:
            legal = game.blue_actions(s)
            logits = actor.logits(s, legal)
            chosen = legal[int(torch.argmax(logits).item())]
            mapping[s] = {chosen: 1.0}
    clone = TabularPolicy('blue', mapping, name='blue_static_relgnn_BC_seed')
    bc_value = float(evaluate_policy_pair_exact(
        game, clone, red_seed,
        initial_graph=scenario.initial_graph,
        initial_ammo=ammo,
        initial_positions=scenario.initial_positions,
    ))
    return clone, {
        'teacher_exact_br_value': float(exact.initial_value),
        'bc_seed_value_vs_initial_red': bc_value,
        'bc_accuracy': final_acc,
        'bc_loss': final_loss,
        'bc_epochs': epochs,
        'bc_train_seconds': perf_counter() - started,
        'teacher_states': len(states),
    }


def start_populations(game, condition, seed, *, scenario, ammo: int = 1):
    red_seed = make_stay_red_policy(game, name="red_stay_seed")
    seed_stats = {}
    if condition == "scratch":
        # Random legal goal-progress policy for a finite initial payoff matrix.
        # This does not use risk/support planning, BC, or a pretrained PPO model.
        blue_seed = random_proper_blue_seed(game, seed)
    elif condition == "bc":
        # Preserve the existing BC population-seeding procedure; later PPO
        # oracles are randomly initialized, exactly as in the original script.
        blue_seed, seed_stats = train_static_relgnn_bc_seed(game, red_seed, seed=seed, scenario=scenario, ammo=ammo)
    elif condition == "heuristic":
        blue_seed, _ = build_snapshot_blue_heuristic(
            game, name="blue_support_snapshot_shortest_seed")
    else:
        raise ValueError(condition)
    return [blue_seed], [red_seed], seed_stats

