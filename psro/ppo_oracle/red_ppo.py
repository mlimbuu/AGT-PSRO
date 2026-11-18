from __future__ import annotations

from game.compat import dataclass, strict_zip

from dataclasses import asdict, field
from time import perf_counter
from typing import Union, Hashable, Literal, Sequence

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from psro.ppo_oracle.mlp import _ActorCritic
from game.environment import AdversarialGraphGame
from svi.matrix_game import solve_zero_sum_matrix_game
from game.models import BlueAction, GameState, RedAction
from psro.meta_game_matrix import build_empirical_payoff_matrix
from psro.psro import (
    PSROIteration,
    TabularPolicy,
    make_proper_blue_policy,
    make_stay_red_policy,
)
from game.state_space import all_position_states

Player = Literal["blue", "red"]
Action = Union[BlueAction, RedAction]


@dataclass(slots=True)
class PPOConfig:
    total_steps: int = 6_000
    rollout_steps: int = 512
    max_episode_steps: int = 32
    hidden_size: int = 64
    learning_rate: float = 3e-4
    gamma: float | None = None
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    update_epochs: int = 6
    minibatch_size: int = 128
    entropy_coef: float = 0.01
    entropy_coef_final: float | None = None
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    reward_scale: float = 0.01
    seed: int = 7
    include_graph_feature: bool = True
    include_ammo_feature: bool = True
    include_red_reachability_feature: bool = False


@dataclass(slots=True)
class PPOOracleStats:
    player: Player
    steps: int
    episodes: int
    mean_episode_cost: float
    mean_episode_length: float
    elapsed_seconds: float
    policy_name: str


@dataclass(slots=True)
class PPOPSROResult:
    blue_population: list[TabularPolicy]
    red_population: list[TabularPolicy]
    payoff_matrix: np.ndarray
    blue_meta_strategy: np.ndarray
    red_meta_strategy: np.ndarray
    meta_value: float
    history: list[PSROIteration] = field(default_factory=list)
    blue_oracle_stats: list[PPOOracleStats] = field(default_factory=list)
    red_oracle_stats: list[PPOOracleStats] = field(default_factory=list)
    converged: bool = False
    semantics: str = "vanilla_psro_episode_level_opponent_sampling_ppo_oracles"




class _StateCodec:
    def __init__(self, game: AdversarialGraphGame, config: PPOConfig) -> None:
        self.game = game
        self.nodes = tuple(game.spec.nodes)
        self.node_index = {node: i for i, node in enumerate(self.nodes)}
        self.graphs = tuple(sorted(game.spec.graph_costs))
        self.graph_index = {graph: i for i, graph in enumerate(self.graphs)}
        self.max_ammo = game.spec.max_ammo
        self.num_agents = game.spec.num_blue_agents
        self.include_graph_feature = bool(config.include_graph_feature)
        self.include_ammo_feature = bool(config.include_ammo_feature)
        self.include_red_reachability_feature = bool(
            config.include_red_reachability_feature
        )
        self.state_dim = self.num_agents * len(self.nodes)
        if self.include_graph_feature:
            self.state_dim += len(self.graphs)
        if self.include_ammo_feature:
            self.state_dim += self.max_ammo + 1
        if self.include_red_reachability_feature:
            self.state_dim += len(self.graphs)

    def encode(self, state: GameState) -> np.ndarray:
        vector = np.zeros(self.state_dim, dtype=np.float32)
        offset = 0
        for agent_index, node in enumerate(state.blue_positions):
            vector[offset + agent_index * len(self.nodes) + self.node_index[node]] = 1.0
        offset += self.num_agents * len(self.nodes)
        if self.include_graph_feature:
            vector[offset + self.graph_index[state.graph_id]] = 1.0
            offset += len(self.graphs)
        if self.include_ammo_feature:
            vector[offset + state.red_ammo] = 1.0
            offset += self.max_ammo + 1
        if self.include_red_reachability_feature:
            # One-step threat topology: encode exactly which Red graph
            # configurations are legal successors from the current state.
            for graph_id in self.game.red_actions(state):
                vector[offset + self.graph_index[graph_id]] = 1.0
        return vector


class _ActionCodec:
    def __init__(self, game: AdversarialGraphGame, player: Player) -> None:
        self.player = player
        if player == "red":
            actions: Sequence[Action] = tuple(sorted(game.spec.graph_costs))
        else:
            seen: set[BlueAction] = set()
            ordered: list[BlueAction] = []
            positions = all_position_states(game.spec)
            for ammo in range(game.spec.max_ammo + 1):
                for graph_id in game.spec.graph_costs:
                    for position in positions:
                        state = GameState(position, graph_id, ammo)
                        for action in game.blue_actions(state):
                            if action not in seen:
                                seen.add(action)
                                ordered.append(action)
            actions = tuple(ordered)
        self.actions = tuple(actions)
        self.index = {action: i for i, action in enumerate(self.actions)}

    def legal_mask(self, game: AdversarialGraphGame, state: GameState) -> np.ndarray:
        legal = game.blue_actions(state) if self.player == "blue" else game.red_actions(state)
        mask = np.zeros(len(self.actions), dtype=np.bool_)
        for action in legal:
            mask[self.index[action]] = True
        return mask

    def action(self, index: int) -> Action:
        return self.actions[index]


@dataclass(slots=True)
class _TransitionSample:
    state: np.ndarray
    action: int
    log_prob: float
    reward: float
    value: float
    mask: np.ndarray
    done: bool
    next_state: np.ndarray
    next_mask: np.ndarray
    truncated: bool


def _sample_policy_action(
    policy: TabularPolicy,
    game: AdversarialGraphGame,
    state: GameState,
    rng: np.random.Generator,
) -> Action:
    actions, probabilities = policy.distribution(game, state)
    index = int(rng.choice(len(actions), p=probabilities))
    return actions[index]


def _masked_distribution(
    model: _ActorCritic,
    states: torch.Tensor,
    masks: torch.Tensor,
) -> tuple[Categorical, torch.Tensor]:
    logits, values = model(states)
    masked_logits = logits.masked_fill(~masks, -1e9)
    return Categorical(logits=masked_logits), values


def _snapshot_tabular_policy(
    game: AdversarialGraphGame,
    model: _ActorCritic,
    state_codec: _StateCodec,
    action_codec: _ActionCodec,
    *,
    player: Player,
    name: str,
) -> TabularPolicy:
    mapping: dict[GameState, dict[Hashable, float]] = {}
    model.eval()
    with torch.no_grad():
        for ammo in range(game.spec.max_ammo + 1):
            for graph_id in game.spec.graph_costs:
                for positions in all_position_states(game.spec):
                    state = GameState(positions, graph_id, ammo)
                    if game.is_terminal(state):
                        continue
                    encoded = torch.from_numpy(state_codec.encode(state)).unsqueeze(0)
                    mask_np = action_codec.legal_mask(game, state)
                    mask = torch.from_numpy(mask_np).unsqueeze(0)
                    distribution, _ = _masked_distribution(model, encoded, mask)
                    probabilities = distribution.probs.squeeze(0).cpu().numpy()
                    action_map: dict[Hashable, float] = {}
                    for index in np.flatnonzero(mask_np):
                        probability = float(probabilities[index])
                        if probability > 1e-12:
                            action_map[action_codec.action(int(index))] = probability
                    mapping[state] = action_map
    return TabularPolicy(player, mapping, name=name)


def train_ppo_best_response(
    game: AdversarialGraphGame,
    *,
    player: Player,
    opponent_population: Sequence[TabularPolicy],
    opponent_weights: Sequence[float],
    initial_graph: int,
    initial_ammo: int,
    initial_positions: tuple[int, ...],
    config: PPOConfig,
    name: str,
    metric_callback=None,
) -> tuple[TabularPolicy, PPOOracleStats]:
    """Train a PPO approximate BR using true PSRO episode-level opponent sampling.

    At the start of every episode, one complete opponent policy is sampled from
    the supplied meta-strategy and then held fixed until termination/truncation.
    Blue minimizes game cost by maximizing reward=-cost; Red maximizes game cost
    by maximizing reward=+cost.
    """

    if not opponent_population:
        raise ValueError("opponent_population cannot be empty")
    weights = np.asarray(opponent_weights, dtype=float)
    if weights.shape != (len(opponent_population),):
        raise ValueError("opponent_weights shape does not match population")
    weights = np.clip(weights, 0.0, None)
    if float(weights.sum()) <= 0.0:
        raise ValueError("opponent_weights must contain positive mass")
    weights /= weights.sum()

    seed = int(config.seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    state_codec = _StateCodec(game, config)
    action_codec = _ActionCodec(game, player)
    model = _ActorCritic(
        state_codec.state_dim,
        len(action_codec.actions),
        config.hidden_size,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    ppo_gamma = game.spec.gamma if config.gamma is None else config.gamma

    samples: list[_TransitionSample] = []
    episode_costs: list[float] = []
    episode_lengths: list[int] = []
    total_steps = 0
    started = perf_counter()

    state = GameState(
        game.canonical_positions(initial_positions),
        initial_graph,
        initial_ammo,
    )
    opponent = opponent_population[int(rng.choice(len(opponent_population), p=weights))]
    episode_cost = 0.0
    episode_length = 0

    while total_steps < config.total_steps:
        batch: list[_TransitionSample] = []
        target_steps = min(config.rollout_steps, config.total_steps - total_steps)

        while len(batch) < target_steps:
            encoded = state_codec.encode(state)
            mask_np = action_codec.legal_mask(game, state)
            states_t = torch.from_numpy(encoded).unsqueeze(0)
            masks_t = torch.from_numpy(mask_np).unsqueeze(0)
            with torch.no_grad():
                distribution, value_t = _masked_distribution(model, states_t, masks_t)
                action_index_t = distribution.sample()
                log_prob_t = distribution.log_prob(action_index_t)
            action_index = int(action_index_t.item())
            learning_action = action_codec.action(action_index)

            if player == "blue":
                blue_action = learning_action
                red_action = _sample_policy_action(opponent, game, state, rng)
            else:
                blue_action = _sample_policy_action(opponent, game, state, rng)
                red_action = learning_action

            assert isinstance(blue_action, tuple)
            stage_cost = game.stage_cost(state, blue_action)
            next_state = game.transition(state, blue_action, int(red_action))
            done = game.is_terminal(next_state)
            episode_length += 1
            episode_cost += stage_cost
            truncated = (episode_length >= config.max_episode_steps) and not done
            signed_reward = -stage_cost if player == "blue" else stage_cost
            reward = config.reward_scale * signed_reward

            next_encoded = state_codec.encode(next_state)
            next_mask_np = action_codec.legal_mask(game, next_state)
            batch.append(
                _TransitionSample(
                    state=encoded,
                    action=action_index,
                    log_prob=float(log_prob_t.item()),
                    reward=float(reward),
                    value=float(value_t.item()),
                    mask=mask_np,
                    done=done,
                    next_state=next_encoded,
                    next_mask=next_mask_np,
                    truncated=truncated,
                )
            )
            total_steps += 1
            state = next_state

            if done or truncated:
                episode_costs.append(float(episode_cost))
                episode_lengths.append(int(episode_length))
                state = GameState(
                    game.canonical_positions(initial_positions),
                    initial_graph,
                    initial_ammo,
                )
                opponent = opponent_population[
                    int(rng.choice(len(opponent_population), p=weights))
                ]
                episode_cost = 0.0
                episode_length = 0

            if len(batch) >= target_steps:
                break

        # Bootstrap each transition directly from its recorded next state.  This
        # is slightly more expensive than a vectorized contiguous GAE pass but
        # correctly handles the episode boundaries produced by opponent resampling.
        next_states_t = torch.from_numpy(np.stack([sample.next_state for sample in batch]))
        next_masks_t = torch.from_numpy(np.stack([sample.next_mask for sample in batch]))
        with torch.no_grad():
            _, next_values_t = _masked_distribution(model, next_states_t, next_masks_t)
        next_values = next_values_t.cpu().numpy()

        advantages = np.zeros(len(batch), dtype=np.float32)
        returns = np.zeros(len(batch), dtype=np.float32)
        gae = 0.0
        for index in range(len(batch) - 1, -1, -1):
            sample = batch[index]
            terminal = sample.done
            bootstrap = 0.0 if terminal else float(next_values[index])
            delta = sample.reward + ppo_gamma * bootstrap - sample.value

            if index == len(batch) - 1:
                continuation = 0.0
            else:
                # Do not carry GAE across terminal/truncation boundaries because
                # a new opponent policy is sampled at the next episode reset.
                boundary = sample.done or sample.truncated
                continuation = 0.0 if boundary else gae
            gae = delta + ppo_gamma * config.gae_lambda * continuation
            advantages[index] = gae
            returns[index] = gae + sample.value

        states = torch.from_numpy(np.stack([sample.state for sample in batch]))
        masks = torch.from_numpy(np.stack([sample.mask for sample in batch]))
        actions = torch.tensor([sample.action for sample in batch], dtype=torch.long)
        old_log_probs = torch.tensor(
            [sample.log_prob for sample in batch], dtype=torch.float32
        )
        advantages_t = torch.from_numpy(advantages)
        returns_t = torch.from_numpy(returns)
        advantages_t = (advantages_t - advantages_t.mean()) / (
            advantages_t.std(unbiased=False) + 1e-8
        )

        model.train()
        n = len(batch)
        indices = np.arange(n)
        for _ in range(config.update_epochs):
            rng.shuffle(indices)
            for start in range(0, n, config.minibatch_size):
                mb = indices[start : start + config.minibatch_size]
                mb_t = torch.as_tensor(mb, dtype=torch.long)
                distribution, values = _masked_distribution(
                    model,
                    states[mb_t],
                    masks[mb_t],
                )
                new_log_probs = distribution.log_prob(actions[mb_t])
                ratios = torch.exp(new_log_probs - old_log_probs[mb_t])
                unclipped = ratios * advantages_t[mb_t]
                clipped = torch.clamp(
                    ratios,
                    1.0 - config.clip_ratio,
                    1.0 + config.clip_ratio,
                ) * advantages_t[mb_t]
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = 0.5 * (returns_t[mb_t] - values).pow(2).mean()
                entropy = distribution.entropy().mean()
                loss = (
                    policy_loss
                    + config.value_coef * value_loss
                    - config.entropy_coef * entropy
                )
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()

        if metric_callback is not None:
            # Collect scalars only; callback never sees tensors or modifies training.
            metric_callback({
                "steps": int(total_steps), "episodes": len(episode_costs),
                "episode_cost_mean_last10": float(np.mean(episode_costs[-10:])) if episode_costs else None,
                "episode_length_mean_last10": float(np.mean(episode_lengths[-10:])) if episode_lengths else None,
                "policy_loss": float(policy_loss.detach().item()),
                "value_loss": float(value_loss.detach().item()),
                "entropy": float(entropy.detach().item()),
                "total_loss": float(loss.detach().item()),
            })

    elapsed = perf_counter() - started
    policy = _snapshot_tabular_policy(
        game,
        model,
        state_codec,
        action_codec,
        player=player,
        name=name,
    )
    stats = PPOOracleStats(
        player=player,
        steps=total_steps,
        episodes=len(episode_costs),
        mean_episode_cost=(float(np.mean(episode_costs)) if episode_costs else float("nan")),
        mean_episode_length=(float(np.mean(episode_lengths)) if episode_lengths else float("nan")),
        elapsed_seconds=float(elapsed),
        policy_name=name,
    )
    return policy, stats


def _expected_blue_cost_against_red_meta(
    game: AdversarialGraphGame,
    blue_policy: TabularPolicy,
    red_population: Sequence[TabularPolicy],
    red_weights: Sequence[float],
    *,
    initial_graph: int,
    initial_ammo: int,
    initial_positions: tuple[int, ...],
) -> float:
    from .psro import evaluate_policy_pair_exact

    return float(
        sum(
            float(weight)
            * evaluate_policy_pair_exact(
                game,
                blue_policy,
                red_policy,
                initial_graph=initial_graph,
                initial_ammo=initial_ammo,
                initial_positions=initial_positions,
            )
            for red_policy, weight in strict_zip(red_population, red_weights)
        )
    )


def _expected_red_cost_against_blue_meta(
    game: AdversarialGraphGame,
    red_policy: TabularPolicy,
    blue_population: Sequence[TabularPolicy],
    blue_weights: Sequence[float],
    *,
    initial_graph: int,
    initial_ammo: int,
    initial_positions: tuple[int, ...],
) -> float:
    from .psro import evaluate_policy_pair_exact

    return float(
        sum(
            float(weight)
            * evaluate_policy_pair_exact(
                game,
                blue_policy,
                red_policy,
                initial_graph=initial_graph,
                initial_ammo=initial_ammo,
                initial_positions=initial_positions,
            )
            for blue_policy, weight in strict_zip(blue_population, blue_weights)
        )
    )


def run_vanilla_psro_ppo(
    game: AdversarialGraphGame,
    *,
    initial_graph: int,
    initial_ammo: int,
    initial_positions: tuple[int, ...],
    blue_ppo: PPOConfig | None = None,
    red_ppo: PPOConfig | None = None,
    initial_blue_policy: TabularPolicy | None = None,
    initial_red_policy: TabularPolicy | None = None,
    max_iterations: int = 4,
    improvement_tolerance: float = 1e-2,
    lp_tolerance: float = 1e-9,
    verbose: bool = False,
) -> PPOPSROResult:
    """Vanilla PSRO with PPO approximate response oracles on both sides.

    The restricted meta-game is solved exactly by LP.  Each PPO oracle samples
    one complete opponent policy from the current meta-strategy at episode start
    and holds it fixed for that episode.  For this small validation game, neural
    policies are snapshotted into complete tabular Markov policies after training,
    which lets the empirical payoff matrix be evaluated exactly rather than by MC.
    """

    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    if game.spec.gamma >= 1.0:
        raise ValueError("Exact meta-game policy evaluation requires gamma < 1")

    blue_cfg = blue_ppo or PPOConfig(seed=7)
    red_cfg = red_ppo or PPOConfig(seed=17)
    blue_population = [initial_blue_policy or make_proper_blue_policy(game)]
    red_population = [initial_red_policy or make_stay_red_policy(game)]
    blue_signatures = {blue_population[0].signature()}
    red_signatures = {red_population[0].signature()}
    history: list[PSROIteration] = []
    blue_stats: list[PPOOracleStats] = []
    red_stats: list[PPOOracleStats] = []
    converged = False

    for iteration in range(1, max_iterations + 1):
        payoff_matrix = build_empirical_payoff_matrix(
            game,
            blue_population,
            red_population,
            initial_graph=initial_graph,
            initial_ammo=initial_ammo,
            initial_positions=initial_positions,
        )
        meta_solution = solve_zero_sum_matrix_game(payoff_matrix, tolerance=lp_tolerance)
        meta_value = float(meta_solution.value)

        blue_policy, blue_stat = train_ppo_best_response(
            game,
            player="blue",
            opponent_population=red_population,
            opponent_weights=meta_solution.red_policy,
            initial_graph=initial_graph,
            initial_ammo=initial_ammo,
            initial_positions=initial_positions,
            config=PPOConfig(**{**asdict(blue_cfg), "seed": blue_cfg.seed + iteration - 1}),
            name=f"blue_ppo_br_{iteration}",
        )
        red_policy, red_stat = train_ppo_best_response(
            game,
            player="red",
            opponent_population=blue_population,
            opponent_weights=meta_solution.blue_policy,
            initial_graph=initial_graph,
            initial_ammo=initial_ammo,
            initial_positions=initial_positions,
            config=PPOConfig(**{**asdict(red_cfg), "seed": red_cfg.seed + iteration - 1}),
            name=f"red_ppo_br_{iteration}",
        )
        blue_stats.append(blue_stat)
        red_stats.append(red_stat)

        blue_br_value = _expected_blue_cost_against_red_meta(
            game,
            blue_policy,
            red_population,
            meta_solution.red_policy,
            initial_graph=initial_graph,
            initial_ammo=initial_ammo,
            initial_positions=initial_positions,
        )
        red_br_value = _expected_red_cost_against_blue_meta(
            game,
            red_policy,
            blue_population,
            meta_solution.blue_policy,
            initial_graph=initial_graph,
            initial_ammo=initial_ammo,
            initial_positions=initial_positions,
        )
        blue_improvement = max(0.0, meta_value - blue_br_value)
        red_improvement = max(0.0, red_br_value - meta_value)
        oracle_gap = max(0.0, red_br_value - blue_br_value)

        blue_signature = blue_policy.signature(digits=7)
        red_signature = red_policy.signature(digits=7)
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
                blue_br_value=float(blue_br_value),
                red_br_value=float(red_br_value),
                blue_improvement=float(blue_improvement),
                red_improvement=float(red_improvement),
                oracle_gap=float(oracle_gap),
                blue_added=blue_added,
                red_added=red_added,
            )
        )

        if verbose:
            print(
                f"PPO-PSRO iter={iteration:02d} "
                f"|Pi_B|={len(blue_population)} |Pi_R|={len(red_population)} "
                f"meta={meta_value:.6f} PPO_BR_B={blue_br_value:.6f} "
                f"PPO_BR_R={red_br_value:.6f} gap={oracle_gap:.3e} "
                f"train_s(B/R)={blue_stat.elapsed_seconds:.2f}/{red_stat.elapsed_seconds:.2f}",
                flush=True,
            )

        # Approximate oracles can occasionally fail to improve.  Treat a joint
        # no-improvement result as convergence for this experimental run rather
        # than claiming an exact Nash certificate.
        if (
            blue_improvement <= improvement_tolerance
            and red_improvement <= improvement_tolerance
        ):
            converged = True
            break

        if blue_added:
            blue_population.append(blue_policy)
            blue_signatures.add(blue_signature)
        if red_added:
            red_population.append(red_policy)
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
    meta_solution = solve_zero_sum_matrix_game(payoff_matrix, tolerance=lp_tolerance)
    return PPOPSROResult(
        blue_population=blue_population,
        red_population=red_population,
        payoff_matrix=payoff_matrix,
        blue_meta_strategy=meta_solution.blue_policy,
        red_meta_strategy=meta_solution.red_policy,
        meta_value=float(meta_solution.value),
        history=history,
        blue_oracle_stats=blue_stats,
        red_oracle_stats=red_stats,
        converged=converged,
    )
