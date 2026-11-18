"""Blue PPO best-response training (same optimizer as legacy RelGNN ablation)."""
from __future__ import annotations
from time import perf_counter
from typing import Sequence
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from game.compat import strict_zip
from game.environment import AdversarialGraphGame
from game.models import BlueAction, GameState
from game.state_space import all_position_states
from psro.psro import TabularPolicy
from psro.ppo_oracle.red_ppo import PPOConfig, PPOOracleStats, _StateCodec, _ActionCodec, _sample_policy_action
from psro.ppo_oracle.mlp import AblationMLPActor
from psro.ppo_oracle.relational_gnn import (
    RelActorKind, RelationalGraphBuilder, SharedMLPCritic, AblationStaticRelGNNCritic,
    RelTransition, _make_actor,
)

def train_relgnn_blue_best_response(
    game: AdversarialGraphGame,
    *,
    actor_kind: RelActorKind,
    opponent_population: Sequence[TabularPolicy],
    opponent_weights: Sequence[float],
    initial_graph: int,
    initial_ammo: int,
    initial_positions: tuple[int, ...],
    config: PPOConfig,
    name: str,
    gnn_dim: int = 16,
    critic_kind: str = "mlp",
    metric_callback=None,
) -> tuple[TabularPolicy, PPOOracleStats, nn.Module]:
    """PPO Blue BR with Static or Action-conditioned typed relational GNN actor."""
    weights = np.asarray(opponent_weights, dtype=float)
    weights = np.clip(weights, 0.0, None)
    weights /= weights.sum()
    seed = int(config.seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    torch.set_num_threads(1)

    graph = RelationalGraphBuilder(game, config)
    state_codec = _StateCodec(game, config)
    actor = (AblationMLPActor(state_codec, _ActionCodec(game, "blue"), hidden=config.hidden_size)
             if actor_kind == "mlp" else _make_actor(actor_kind, graph, gnn_dim))
    critic = (AblationStaticRelGNNCritic(graph, d=gnn_dim) if critic_kind == "static_relgnn"
              else SharedMLPCritic(state_codec.state_dim, hidden=config.hidden_size))
    def critic_state(st):
        if critic_kind == "static_relgnn":
            return critic.value(st)
        return critic(torch.from_numpy(state_codec.encode(st))).reshape(())
    optimizer = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()), lr=config.learning_rate
    )
    ppo_gamma = game.spec.gamma if config.gamma is None else config.gamma

    total_steps = 0
    episode_costs: list[float] = []
    episode_lengths: list[int] = []
    started = perf_counter()
    state = GameState(game.canonical_positions(initial_positions), initial_graph, initial_ammo)
    opponent = opponent_population[int(rng.choice(len(opponent_population), p=weights))]
    episode_cost = 0.0
    episode_length = 0

    while total_steps < config.total_steps:
        batch: list[RelTransition] = []
        target_steps = min(config.rollout_steps, config.total_steps - total_steps)
        while len(batch) < target_steps:
            legal_actions = game.blue_actions(state)
            with torch.no_grad():
                logits = actor.logits(state, legal_actions)
                dist = Categorical(logits=logits)
                aidx_t = dist.sample()
                lp_t = dist.log_prob(aidx_t)
                value_t = critic_state(state)
            aidx = int(aidx_t.item())
            blue_action = legal_actions[aidx]
            red_action = _sample_policy_action(opponent, game, state, rng)
            stage_cost = game.stage_cost(state, blue_action)
            next_state = game.transition(state, blue_action, int(red_action))
            done = game.is_terminal(next_state)
            episode_length += 1
            episode_cost += stage_cost
            truncated = (episode_length >= config.max_episode_steps) and not done
            reward = config.reward_scale * (-stage_cost)
            batch.append(RelTransition(
                state=state,
                action_local_index=aidx,
                log_prob=float(lp_t.item()),
                reward=float(reward),
                value=float(value_t.item()),
                done=done,
                truncated=truncated,
                next_state=next_state,
            ))
            total_steps += 1
            state = next_state
            if done or truncated:
                episode_costs.append(float(episode_cost))
                episode_lengths.append(int(episode_length))
                state = GameState(game.canonical_positions(initial_positions), initial_graph, initial_ammo)
                opponent = opponent_population[int(rng.choice(len(opponent_population), p=weights))]
                episode_cost = 0.0
                episode_length = 0
            if len(batch) >= target_steps:
                break

        # Bootstrap critic values.
        with torch.no_grad():
            nv = np.array([float(critic_state(s.next_state).item()) for s in batch], dtype=np.float32)
        adv = np.zeros(len(batch), dtype=np.float32)
        ret = np.zeros(len(batch), dtype=np.float32)
        gae = 0.0
        for k in range(len(batch)-1, -1, -1):
            smp = batch[k]
            bootstrap = 0.0 if smp.done else float(nv[k])
            delta = smp.reward + ppo_gamma * bootstrap - smp.value
            continuation = 0.0 if (k == len(batch)-1 or smp.done or smp.truncated) else gae
            gae = delta + ppo_gamma * config.gae_lambda * continuation
            adv[k] = gae
            ret[k] = gae + smp.value

        adv_t = torch.from_numpy(adv)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std(unbiased=False) + 1e-8)
        ret_t = torch.from_numpy(ret)
        old_lp = torch.tensor([s.log_prob for s in batch], dtype=torch.float32)
        indices = np.arange(len(batch))
        actor.train(); critic.train()

        for _ in range(config.update_epochs):
            rng.shuffle(indices)
            for start in range(0, len(batch), config.minibatch_size):
                mb = indices[start:start+config.minibatch_size]
                # Many rollout samples revisit the same small-game state.
                # Compute the GNN distribution once per unique state and reuse it.
                dist_cache: dict[GameState, Categorical] = {}
                new_lps = []
                entropies = []
                state_x = []
                for j in mb:
                    smp = batch[int(j)]
                    dist = dist_cache.get(smp.state)
                    if dist is None:
                        legal = game.blue_actions(smp.state)
                        dist = Categorical(logits=actor.logits(smp.state, legal))
                        dist_cache[smp.state] = dist
                    ai = torch.tensor(smp.action_local_index, dtype=torch.long)
                    new_lps.append(dist.log_prob(ai))
                    entropies.append(dist.entropy())
                    state_x.append(state_codec.encode(smp.state))
                new_lp_t = torch.stack(new_lps)
                ent_t = torch.stack(entropies).mean()
                if critic_kind == "static_relgnn":
                    value_cache = {}
                    value_batch = []
                    for j in mb:
                        st = batch[int(j)].state
                        if st not in value_cache:
                            value_cache[st] = critic_state(st)
                        value_batch.append(value_cache[st])
                    val_t = torch.stack(value_batch)
                else:
                    val_t = critic(torch.from_numpy(np.stack(state_x)))
                mb_t = torch.as_tensor(mb, dtype=torch.long)
                ratios = torch.exp(new_lp_t - old_lp[mb_t])
                unclipped = ratios * adv_t[mb_t]
                clipped = torch.clamp(ratios, 1-config.clip_ratio, 1+config.clip_ratio) * adv_t[mb_t]
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = 0.5 * (ret_t[mb_t] - val_t).pow(2).mean()
                if config.entropy_coef_final is None:
                    entropy_coef = config.entropy_coef
                else:
                    progress = min(1.0, max(0.0, total_steps / max(1, config.total_steps)))
                    entropy_coef = (
                        config.entropy_coef
                        + progress * (config.entropy_coef_final - config.entropy_coef)
                    )
                loss = policy_loss + config.value_coef * value_loss - entropy_coef * ent_t
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), config.max_grad_norm)
                optimizer.step()

        if metric_callback is not None:
            metric_callback({
                "steps": int(total_steps), "episodes": len(episode_costs),
                "episode_cost_mean_last10": float(np.mean(episode_costs[-10:])) if episode_costs else None,
                "episode_length_mean_last10": float(np.mean(episode_lengths[-10:])) if episode_lengths else None,
                "policy_loss": float(policy_loss.detach().item()),
                "value_loss": float(value_loss.detach().item()),
                "entropy": float(ent_t.detach().item()),
                "total_loss": float(loss.detach().item()),
            })

    elapsed = perf_counter() - started
    # Snapshot complete Markov stochastic policy for exact evaluation.
    mapping = {}
    actor.eval()
    with torch.no_grad():
        for ammo in range(game.spec.max_ammo + 1):
            for graph_id in game.spec.graph_costs:
                for pos in all_position_states(game.spec):
                    s = GameState(pos, graph_id, ammo)
                    if game.is_terminal(s):
                        continue
                    legal = game.blue_actions(s)
                    probs = torch.softmax(actor.logits(s, legal), dim=-1).cpu().numpy()
                    mapping[s] = {a: float(p) for a, p in strict_zip(legal, probs) if p > 1e-12}
    policy = TabularPolicy("blue", mapping, name=name)
    stats = PPOOracleStats(
        player="blue", steps=total_steps, episodes=len(episode_costs),
        mean_episode_cost=float(np.mean(episode_costs)) if episode_costs else float("nan"),
        mean_episode_length=float(np.mean(episode_lengths)) if episode_lengths else float("nan"),
        elapsed_seconds=float(elapsed), policy_name=name,
    )
    return policy, stats, actor
