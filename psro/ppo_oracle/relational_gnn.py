from __future__ import annotations

from game.compat import dataclass, strict_zip


from math import inf
from time import perf_counter
from typing import Literal, Sequence
import heapq

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from game.environment import AdversarialGraphGame
from game.models import BlueAction, GameState
from psro.psro import TabularPolicy
from psro.ppo_oracle.red_ppo import PPOConfig, PPOOracleStats, _StateCodec, _ActionCodec, _sample_policy_action
from game.state_space import all_position_states

RelActorKind = Literal["static_relgnn", "action_relgnn", "static_relgnn_virtual"]


def _norm_mat(dst_n: int, src_n: int, pairs: list[tuple[int, int]]) -> torch.Tensor:
    """Mean-aggregation matrix for pairs encoded as (src, dst)."""
    A = torch.zeros(dst_n, src_n, dtype=torch.float32)
    for src, dst in pairs:
        A[dst, src] += 1.0
    return A / A.sum(dim=1, keepdim=True).clamp_min(1.0)


class RelationalGraphBuilder:
    """Build the lifted location/traversal graph for one game specification.

    Dynamic features are computed per state, while FROM/TO/SUPPORTS topology is
    fixed.  This matches the user's Static/Action Rel-GNN design but adapts the
    action-conditioned features to the current support timing: supporter must
    occupy z at time t and wait at z during the teammate's risky traversal.
    """

    def __init__(self, game: AdversarialGraphGame, config: PPOConfig) -> None:
        self.game = game
        self.config = config
        self.nodes = tuple(game.spec.nodes)
        self.node_idx = {v: i for i, v in enumerate(self.nodes)}
        self.graphs = tuple(sorted(game.spec.graph_costs))
        self.graph_idx = {g: i for i, g in enumerate(self.graphs)}
        self.num_agents = game.spec.num_blue_agents
        self.max_ammo = game.spec.max_ammo

        traversals: list[tuple[int, int]] = []
        for u in self.nodes:
            for v in game.spec.adjacency[u]:
                traversals.append((u, v))
        self.traversals = tuple(traversals)
        self.tid = {e: i for i, e in enumerate(self.traversals)}
        self.num_traversals = len(self.traversals)

        # Extract risk/support semantics from the active cost model.
        self.risk_penalty_by_graph: dict[int, dict[tuple[int, int], float]] = {
            g: {} for g in self.graphs
        }
        self.support_nodes_by_edge: dict[tuple[int, int], frozenset[int]] = {}
        self.supported_penalty_multiplier = 1.0

        cm = game.cost_model
        if hasattr(cm, "support"):
            s = cm.support
            self.risk_penalty_by_graph = {
                int(g): {tuple(e): float(p) for e, p in m.items()}
                for g, m in s.risk_penalty_by_graph.items()
            }
            self.support_nodes_by_edge = {
                tuple(e): frozenset(zs)
                for e, zs in s.support_nodes_by_risky_edge.items()
            }
            self.supported_penalty_multiplier = float(s.supported_penalty_multiplier)
        elif hasattr(cm, "risk"):
            r = cm.risk
            self.risk_penalty_by_graph = {
                int(g): {tuple(e): float(p) for e, p in m.items()}
                for g, m in r.risk_penalty_by_graph.items()
            }

        self.support_nodes = frozenset(
            z for zs in self.support_nodes_by_edge.values() for z in zs
        )

        # Typed topology: FROM, TO, SUPPORTS and their reverse messages.
        from_pairs: list[tuple[int, int]] = []
        to_pairs: list[tuple[int, int]] = []
        support_pairs: list[tuple[int, int]] = []
        for tid, (u, v) in enumerate(self.traversals):
            from_pairs.append((self.node_idx[u], tid))
            to_pairs.append((self.node_idx[v], tid))
            for z in self.support_nodes_by_edge.get((u, v), frozenset()):
                support_pairs.append((self.node_idx[z], tid))

        V, T = len(self.nodes), self.num_traversals
        self.A_LT_FROM = _norm_mat(T, V, from_pairs)
        self.A_LT_TO = _norm_mat(T, V, to_pairs)
        self.A_LT_SUPPORT = _norm_mat(T, V, support_pairs)
        self.A_TL_SOURCE = _norm_mat(V, T, [(t, v) for v, t in from_pairs])
        self.A_TL_DEST = _norm_mat(V, T, [(t, v) for v, t in to_pairs])
        self.A_TL_SUPPORTED_BY = _norm_mat(V, T, [(t, v) for v, t in support_pairs])

        # Stable normalization based on the largest immediate edge/risk cost.
        mx = 1.0
        for g in self.graphs:
            for e, base in game.spec.graph_costs[g].items():
                mx = max(mx, abs(float(base)) + abs(float(self.risk_penalty_by_graph.get(g, {}).get(e, 0.0))))
        self.cost_scale = mx

        # Base location features:
        # isSupport, currentOcc, isGoal, Jreal, Jopt, deltaJ,
        # graph one-hot, ammo one-hot, one-step Red reachability.
        self.base_loc_dim = 6 + len(self.graphs) + (self.max_ammo + 1) + len(self.graphs)
        # Action-conditioned version adds next occupancy and stay occupancy.
        self.action_loc_dim = self.base_loc_dim + 2
        # traversal features: risky, supportable, base, unsupported, supported.
        self.trav_dim = 5
        self._loc_cache: dict[GameState, torch.Tensor] = {}
        self._trav_cache: dict[GameState, torch.Tensor] = {}
        self._action_loc_cache: dict[tuple[GameState, tuple[BlueAction, ...]], torch.Tensor] = {}

    def _dijkstra_to_goal(self, state: GameState, optimistic: bool) -> dict[int, float]:
        goal = self.game.spec.goal_node
        rev: dict[int, list[tuple[int, float]]] = {v: [] for v in self.nodes}
        penalties = self.risk_penalty_by_graph.get(state.graph_id, {})
        costs = self.game.spec.graph_costs[state.graph_id]
        for u, v in self.traversals:
            if u == v:
                continue
            base = float(costs[(u, v)])
            penalty = float(penalties.get((u, v), 0.0))
            if optimistic and (u, v) in self.support_nodes_by_edge:
                penalty *= self.supported_penalty_multiplier
            rev[v].append((u, base + penalty))

        d = {v: inf for v in self.nodes}
        d[goal] = 0.0
        pq = [(0.0, goal)]
        while pq:
            dv, v = heapq.heappop(pq)
            if dv != d[v]:
                continue
            for u, w in rev[v]:
                nd = dv + w
                if nd < d[u]:
                    d[u] = nd
                    heapq.heappush(pq, (nd, u))
        return d

    def location_features(self, state: GameState) -> torch.Tensor:
        cached = self._loc_cache.get(state)
        if cached is not None:
            return cached
        j_real = self._dijkstra_to_goal(state, optimistic=False)
        j_opt = self._dijkstra_to_goal(state, optimistic=True)
        V = len(self.nodes)
        X = torch.zeros(V, self.base_loc_dim, dtype=torch.float32)
        reach = set(self.game.red_actions(state))
        for vi, v in enumerate(self.nodes):
            jr = j_real[v] if np.isfinite(j_real[v]) else 2.0 * self.cost_scale
            jo = j_opt[v] if np.isfinite(j_opt[v]) else 2.0 * self.cost_scale
            vals: list[float] = [
                float(v in self.support_nodes),
                sum(p == v for p in state.blue_positions) / self.num_agents,
                float(v == self.game.spec.goal_node),
                float(jr / self.cost_scale),
                float(jo / self.cost_scale),
                float((jr - jo) / self.cost_scale),
            ]
            vals += [float(state.graph_id == g) for g in self.graphs]
            vals += [float(state.red_ammo == a) for a in range(self.max_ammo + 1)]
            vals += [float(g in reach) for g in self.graphs]
            X[vi] = torch.tensor(vals, dtype=torch.float32)
        self._loc_cache[state] = X
        return X

    def action_location_features(
        self, state: GameState, actions: Sequence[BlueAction]
    ) -> torch.Tensor:
        actions_t = tuple(actions)
        key = (state, actions_t)
        cached = self._action_loc_cache.get(key)
        if cached is not None:
            return cached
        base = self.location_features(state)
        B, V = len(actions), len(self.nodes)
        X = base.unsqueeze(0).repeat(B, 1, 1)
        next_occ = torch.zeros(B, V, dtype=torch.float32)
        stay_occ = torch.zeros(B, V, dtype=torch.float32)
        for b, action in enumerate(actions):
            for cur, nxt in strict_zip(state.blue_positions, action):
                next_occ[b, self.node_idx[nxt]] += 1.0 / self.num_agents
                if cur == nxt:
                    stay_occ[b, self.node_idx[cur]] += 1.0 / self.num_agents
        out = torch.cat([X, next_occ.unsqueeze(-1), stay_occ.unsqueeze(-1)], dim=-1)
        self._action_loc_cache[key] = out
        return out

    def traversal_features(self, state: GameState) -> torch.Tensor:
        cached = self._trav_cache.get(state)
        if cached is not None:
            return cached
        penalties = self.risk_penalty_by_graph.get(state.graph_id, {})
        costs = self.game.spec.graph_costs[state.graph_id]
        rows = []
        for e in self.traversals:
            base = float(costs[e])
            penalty = float(penalties.get(e, 0.0))
            supportable = e in self.support_nodes_by_edge
            unsupported = base + penalty
            supported = base + (penalty * self.supported_penalty_multiplier if supportable else penalty)
            rows.append([
                float(penalty > 0.0),
                float(supportable),
                base / self.cost_scale,
                unsupported / self.cost_scale,
                supported / self.cost_scale,
            ])
        out = torch.tensor(rows, dtype=torch.float32)
        self._trav_cache[state] = out
        return out

    def traversal_id(self, u: int, v: int) -> int:
        return self.tid[(u, v)]


class TypedRelBlock(nn.Module):
    def __init__(self, d: int, graph: RelationalGraphBuilder) -> None:
        super().__init__()
        # L -> T typed maps
        self.w_from = nn.Linear(d, d, bias=False)
        self.w_to = nn.Linear(d, d, bias=False)
        self.w_support = nn.Linear(d, d, bias=False)
        # T -> L reverse typed maps
        self.w_source = nn.Linear(d, d, bias=False)
        self.w_dest = nn.Linear(d, d, bias=False)
        self.w_supported_by = nn.Linear(d, d, bias=False)

        self.up_t = nn.Sequential(nn.Linear(4 * d, d), nn.Tanh(), nn.Linear(d, d))
        self.up_l = nn.Sequential(nn.Linear(4 * d, d), nn.Tanh(), nn.Linear(d, d))
        self.norm_t = nn.LayerNorm(d)
        self.norm_l = nn.LayerNorm(d)

        for name in (
            "A_LT_FROM", "A_LT_TO", "A_LT_SUPPORT",
            "A_TL_SOURCE", "A_TL_DEST", "A_TL_SUPPORTED_BY",
        ):
            self.register_buffer(name, getattr(graph, name).clone())

    def forward(self, L: torch.Tensor, T: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if L.dim() == 2:
            mf = self.A_LT_FROM @ self.w_from(L)
            mt = self.A_LT_TO @ self.w_to(L)
            ms = self.A_LT_SUPPORT @ self.w_support(L)
            T = self.norm_t(T + self.up_t(torch.cat([T, mf, mt, ms], dim=-1)))

            msrc = self.A_TL_SOURCE @ self.w_source(T)
            mdst = self.A_TL_DEST @ self.w_dest(T)
            msup = self.A_TL_SUPPORTED_BY @ self.w_supported_by(T)
            L = self.norm_l(L + self.up_l(torch.cat([L, msrc, mdst, msup], dim=-1)))
        else:
            mf = torch.einsum("tv,bvd->btd", self.A_LT_FROM, self.w_from(L))
            mt = torch.einsum("tv,bvd->btd", self.A_LT_TO, self.w_to(L))
            ms = torch.einsum("tv,bvd->btd", self.A_LT_SUPPORT, self.w_support(L))
            T = self.norm_t(T + self.up_t(torch.cat([T, mf, mt, ms], dim=-1)))

            msrc = torch.einsum("vt,btd->bvd", self.A_TL_SOURCE, self.w_source(T))
            mdst = torch.einsum("vt,btd->bvd", self.A_TL_DEST, self.w_dest(T))
            msup = torch.einsum("vt,btd->bvd", self.A_TL_SUPPORTED_BY, self.w_supported_by(T))
            L = self.norm_l(L + self.up_l(torch.cat([L, msrc, mdst, msup], dim=-1)))
        return L, T


class TypedRelEncoder(nn.Module):
    def __init__(self, graph: RelationalGraphBuilder, loc_dim: int, d: int = 16) -> None:
        super().__init__()
        self.graph = graph
        self.loc_embed = nn.Sequential(nn.Linear(loc_dim, d), nn.Tanh())
        self.trav_embed = nn.Sequential(nn.Linear(graph.trav_dim, d), nn.Tanh())
        self.blocks = nn.ModuleList([TypedRelBlock(d, graph), TypedRelBlock(d, graph)])

    def forward(self, X_L: torch.Tensor, X_T: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        L = self.loc_embed(X_L)
        if X_L.dim() == 2:
            T = self.trav_embed(X_T)
        else:
            B = X_L.shape[0]
            T = self.trav_embed(X_T).unsqueeze(0).expand(B, -1, -1)
        for block in self.blocks:
            L, T = block(L, T)
        return L, T



class VirtualTypedRelEncoder(nn.Module):
    """Typed Rel-GNN with one global/virtual node.

    The local FROM/TO/SUPPORT message passing is unchanged. After each local
    block, the virtual node aggregates mean location/traversal embeddings and
    broadcasts global context back to every entity. The virtual node is
    initialized from the already-present global strategic features
    (graph-id, ammo, one-step Red reachability) so no observation information
    is added relative to the Static Rel-GNN baseline.
    """
    def __init__(self, graph: RelationalGraphBuilder, loc_dim: int, d: int = 16) -> None:
        super().__init__()
        self.graph = graph
        self.d = d
        self.loc_embed = nn.Sequential(nn.Linear(loc_dim, d), nn.Tanh())
        self.trav_embed = nn.Sequential(nn.Linear(graph.trav_dim, d), nn.Tanh())
        self.blocks = nn.ModuleList([TypedRelBlock(d, graph), TypedRelBlock(d, graph)])
        # Location feature layout: first six are local; remaining are the
        # repeated global strategic features.
        global_dim = loc_dim - 6
        self.global_init = nn.Sequential(nn.Linear(global_dim, d), nn.Tanh())
        self.global_updates = nn.ModuleList([
            nn.Sequential(nn.Linear(3 * d, d), nn.Tanh(), nn.Linear(d, d))
            for _ in self.blocks
        ])
        self.global_norms = nn.ModuleList([nn.LayerNorm(d) for _ in self.blocks])
        self.to_loc = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in self.blocks])
        self.to_trav = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in self.blocks])
        self.loc_norms = nn.ModuleList([nn.LayerNorm(d) for _ in self.blocks])
        self.trav_norms = nn.ModuleList([nn.LayerNorm(d) for _ in self.blocks])

    def forward(self, X_L: torch.Tensor, X_T: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if X_L.dim() != 2:
            raise ValueError("Virtual Static Rel-GNN expects one state graph, not a candidate-action batch.")
        L = self.loc_embed(X_L)
        T = self.trav_embed(X_T)
        g = self.global_init(X_L[0, 6:])
        for i, block in enumerate(self.blocks):
            L, T = block(L, T)
            pooled_l = L.mean(dim=0)
            pooled_t = T.mean(dim=0)
            g = self.global_norms[i](g + self.global_updates[i](torch.cat([g, pooled_l, pooled_t], dim=-1)))
            L = self.loc_norms[i](L + self.to_loc[i](g).unsqueeze(0))
            T = self.trav_norms[i](T + self.to_trav[i](g).unsqueeze(0))
        return L, T, g


class VirtualStaticRelGNNActor(nn.Module):
    def __init__(self, graph: RelationalGraphBuilder, d: int = 16) -> None:
        super().__init__()
        self.graph = graph
        self.d = d
        self.encoder = VirtualTypedRelEncoder(graph, graph.base_loc_dim, d)
        # Use virtual-node embedding plus the same mean pooled L/T context and
        # per-agent current/traversal/destination embeddings as the baseline.
        in_dim = 3 * d + graph.num_agents * 3 * d
        self.head = nn.Sequential(nn.Linear(in_dim, 72), nn.Tanh(), nn.Linear(72, 1))

    def logits(self, state: GameState, actions: Sequence[BlueAction]) -> torch.Tensor:
        X_L = self.graph.location_features(state)
        X_T = self.graph.traversal_features(state)
        L, T, vg = self.encoder(X_L, X_T)
        g = torch.cat([L.mean(dim=0), T.mean(dim=0), vg], dim=-1)
        rows = []
        for action in actions:
            parts = [g]
            for i, nxt in enumerate(action):
                cur = state.blue_positions[i]
                tid = self.graph.traversal_id(cur, nxt)
                parts.extend([L[self.graph.node_idx[cur]], T[tid], L[self.graph.node_idx[nxt]]])
            rows.append(torch.cat(parts, dim=-1))
        return self.head(torch.stack(rows)).squeeze(-1)


class StaticRelGNNActor(nn.Module):
    def __init__(self, graph: RelationalGraphBuilder, d: int = 16) -> None:
        super().__init__()
        self.graph = graph
        self.d = d
        self.encoder = TypedRelEncoder(graph, graph.base_loc_dim, d)
        in_dim = 2 * d + graph.num_agents * 3 * d
        self.head = nn.Sequential(nn.Linear(in_dim, 72), nn.Tanh(), nn.Linear(72, 1))

    def logits(self, state: GameState, actions: Sequence[BlueAction]) -> torch.Tensor:
        X_L = self.graph.location_features(state)
        X_T = self.graph.traversal_features(state)
        L, T = self.encoder(X_L, X_T)
        g = torch.cat([L.mean(dim=0), T.mean(dim=0)], dim=-1)
        rows = []
        for action in actions:
            parts = [g]
            for i, nxt in enumerate(action):
                cur = state.blue_positions[i]
                tid = self.graph.traversal_id(cur, nxt)
                parts.extend([L[self.graph.node_idx[cur]], T[tid], L[self.graph.node_idx[nxt]]])
            rows.append(torch.cat(parts, dim=-1))
        return self.head(torch.stack(rows)).squeeze(-1)


class ActionRelGNNActor(nn.Module):
    def __init__(self, graph: RelationalGraphBuilder, d: int = 16) -> None:
        super().__init__()
        self.graph = graph
        self.d = d
        self.encoder = TypedRelEncoder(graph, graph.action_loc_dim, d)
        in_dim = 2 * d + graph.num_agents * 3 * d
        self.head = nn.Sequential(nn.Linear(in_dim, 72), nn.Tanh(), nn.Linear(72, 1))

    def logits(self, state: GameState, actions: Sequence[BlueAction]) -> torch.Tensor:
        X_L = self.graph.action_location_features(state, actions)
        X_T = self.graph.traversal_features(state)
        L, T = self.encoder(X_L, X_T)
        g = torch.cat([L.mean(dim=1), T.mean(dim=1)], dim=-1)
        B = len(actions)
        bidx = torch.arange(B, dtype=torch.long)
        pieces = [g]
        for i in range(self.graph.num_agents):
            cur_idx = torch.tensor([self.graph.node_idx[state.blue_positions[i]]] * B, dtype=torch.long)
            tid_idx = torch.tensor([
                self.graph.traversal_id(state.blue_positions[i], action[i]) for action in actions
            ], dtype=torch.long)
            dst_idx = torch.tensor([
                self.graph.node_idx[action[i]] for action in actions
            ], dtype=torch.long)
            pieces.extend([L[bidx, cur_idx], T[bidx, tid_idx], L[bidx, dst_idx]])
        return self.head(torch.cat(pieces, dim=-1)).squeeze(-1)


class SharedMLPCritic(nn.Module):
    def __init__(self, state_dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)



class AblationStaticRelGNNCritic(nn.Module):
    """State value V(s) using the same typed-L/T message-passing layers."""
    def __init__(self, graph, d=16):
        super().__init__()
        self.graph=graph
        self.encoder=TypedRelEncoder(graph,graph.base_loc_dim,d)
        self.head=nn.Sequential(nn.Linear((2+graph.num_agents)*d,72), nn.Tanh(), nn.Linear(72,1))
    def value(self,state):
        L,T=self.encoder(self.graph.location_features(state),self.graph.traversal_features(state))
        agents=torch.cat([L[self.graph.node_idx[pos]] for pos in state.blue_positions])
        x=torch.cat([L.mean(dim=0),T.mean(dim=0),agents])
        return self.head(x).squeeze(-1)

@dataclass(slots=True)
class RelTransition:
    state: GameState
    action_local_index: int
    log_prob: float
    reward: float
    value: float
    done: bool
    truncated: bool
    next_state: GameState


def _make_actor(kind: RelActorKind, graph: RelationalGraphBuilder, d: int) -> nn.Module:
    if kind == "static_relgnn":
        return StaticRelGNNActor(graph, d=d)
    if kind == "action_relgnn":
        return ActionRelGNNActor(graph, d=d)
    if kind == "static_relgnn_virtual":
        return VirtualStaticRelGNNActor(graph, d=d)
    raise ValueError(kind)


