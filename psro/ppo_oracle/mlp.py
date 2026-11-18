"""MLP network architectures used by both PPO oracles."""
from __future__ import annotations
import torch
from torch import nn

class _ActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden_size, action_dim)
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.body(states)
        return self.actor(features), self.critic(features).squeeze(-1)


class AblationMLPActor(nn.Module):
    """State-only MLP with masked global joint-action logits."""
    def __init__(self, state_codec, action_codec, hidden=64):
        super().__init__()
        self.state_codec, self.action_codec = state_codec, action_codec
        self.net = nn.Sequential(nn.Linear(state_codec.state_dim, hidden), nn.Tanh(),
            nn.Linear(hidden,hidden), nn.Tanh(), nn.Linear(hidden,len(action_codec.actions)))
    def logits(self, state, actions):
        x=torch.from_numpy(self.state_codec.encode(state)).unsqueeze(0)
        full=self.net(x).squeeze(0)
        return full[[self.action_codec.index[a] for a in actions]]

