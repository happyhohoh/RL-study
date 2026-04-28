"""
Week 01 / Day 01
Task: Implement a minimal PyTorch MLP policy network.

Goal:
- Input: state tensor with shape [batch_size, state_dim]
- Output: action probability tensor with shape [batch_size, action_dim]
"""

import torch
import torch.nn as nn


class MLPPolicy(nn.Module):
    """A minimal policy network for discrete action spaces."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()

        # TODO: Define the network structure.
        # Suggested structure:
        # Linear(state_dim, hidden_dim) -> ReLU -> Linear(hidden_dim, action_dim)
        self.net = nn.Sequential(
            # Write your layers here.
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return action probabilities.

        Args:
            state: Tensor with shape [batch_size, state_dim]

        Returns:
            action_probs: Tensor with shape [batch_size, action_dim]
        """
        # TODO: Pass state through the network and convert logits to probabilities.
        # Hint: use torch.softmax(logits, dim=-1)
        raise NotImplementedError


if __name__ == "__main__":
    state_dim = 4
    action_dim = 3
    batch_size = 4

    policy = MLPPolicy(state_dim=state_dim, action_dim=action_dim)
    state = torch.randn(batch_size, state_dim)

    # TODO: Run the policy and print the output shape.
    # Expected shape: [4, 3]
    action_probs = policy(state)
    print("state shape:", state.shape)
    print("action_probs shape:", action_probs.shape)
    print("action_probs:", action_probs)
    print("sum over actions:", action_probs.sum(dim=-1))
