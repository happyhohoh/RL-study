"""
Week 01 / Day 02
Task: Implement a minimal GridWorld environment without using Gym.

The environment should include:
- reset()
- step(action)
- render()
"""

from __future__ import annotations

import numpy as np


class GridWorldEnv:
    """A minimal 2D GridWorld environment."""

    # Discrete actions
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

    def __init__(self, size: int = 5):
        self.size = size
        self.agent_pos = None
        self.goal_pos = None
        self.obstacles = set()

    def reset(self):
        """Reset the environment and return the initial state."""
        # TODO: Initialize agent position, goal position, and obstacles.
        # Example:
        # self.agent_pos = (0, 0)
        # self.goal_pos = (self.size - 1, self.size - 1)
        # self.obstacles = {(1, 1), (2, 2)}
        raise NotImplementedError

    def step(self, action: int):
        """Execute one action in the environment.

        Args:
            action: 0=up, 1=down, 2=left, 3=right

        Returns:
            next_state: environment state after action
            reward: scalar reward
            done: whether the episode is finished
            info: extra diagnostic information
        """
        # TODO: Compute next position according to action.
        # TODO: Check boundary and obstacle collision.
        # TODO: Compute reward and done.
        raise NotImplementedError

    def get_state(self):
        """Return the current state representation."""
        # TODO: Choose a state representation.
        # Option 1: return agent position only.
        # Option 2: return agent position + goal position.
        # Option 3: return a grid matrix.
        raise NotImplementedError

    def render(self):
        """Print the current grid to the terminal."""
        grid = np.full((self.size, self.size), ".", dtype=str)

        for obs in self.obstacles:
            grid[obs] = "#"

        if self.goal_pos is not None:
            grid[self.goal_pos] = "G"

        if self.agent_pos is not None:
            grid[self.agent_pos] = "A"

        for row in grid:
            print(" ".join(row))
        print()
