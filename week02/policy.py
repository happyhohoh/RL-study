"""
Week 02
Task: Policy network used by REINFORCE on GridWorld.

The policy outputs a probability distribution over discrete actions:
[P(UP), P(DOWN), P(LEFT), P(RIGHT)]
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MLPPolicy(nn.Module):
    """A minimal MLP policy network for discrete action spaces."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return action probabilities.

        Args:
            state: Tensor with shape [batch_size, state_dim].

        Returns:
            action_probs: Tensor with shape [batch_size, action_dim].
        """
        logits = self.net(state)
        action_probs = torch.softmax(logits, dim=-1)
        return action_probs


if __name__ == "__main__":
    state_dim = 2
    action_dim = 4
    batch_size = 4

    policy = MLPPolicy(state_dim=state_dim, action_dim=action_dim)
    state = torch.randn(batch_size, state_dim)

    with torch.no_grad():
        action_probs = policy(state)

    print("state shape:", state.shape)
    print("action_probs shape:", action_probs.shape)
    print("action_probs:\n", action_probs)
    print("sum over actions:", action_probs.sum(dim=-1))
