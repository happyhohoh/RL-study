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

    def __init__(self, size: int = 5, max_steps: int = 50):
        self.size = size
        self.agent_pos = None
        self.goal_pos = None
        self.obstacles = set()
        self.total_reward = 0.0
        self.max_steps = max_steps
        self.done = False
        self.step_count = 0

    def reset(self):
        """Reset the environment and return the initial state."""
        # TODO: Initialize agent position, goal position, and obstacles.
        self.agent_pos = (0, 0)
        self.goal_pos = (self.size - 1, self.size - 1)
        self.obstacles = {(1, 1), (2, 2)}
        self.step_count = 0
        self.total_reward = 0.0
        self.done = False
        return self.get_state()

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
        if(self.done):
            return self.get_state(), 0, self.done, {"step_count": self.step_count, "total_reward": self.total_reward}
        if(self.step_count >= self.max_steps):
            self.done = True
            return self.get_state(), 0, self.done, {"step_count": self.step_count, "total_reward": self.total_reward}

        next_pos = self.agent_pos
        if(action == 0):
            next_pos = (self.agent_pos[0] - 1, self.agent_pos[1])
        elif(action == 1):
            next_pos = (self.agent_pos[0] + 1, self.agent_pos[1])
        elif(action == 2):
            next_pos = (self.agent_pos[0], self.agent_pos[1] - 1)
        elif(action == 3):
            next_pos = (self.agent_pos[0], self.agent_pos[1] + 1)
        else:
            raise ValueError(f"Invalid action: {action}")

        self.step_count += 1
        if(self.is_out(next_pos)):
            reward = -5 # 越界相当于撞墙，给予负奖励，但不结束
            self.total_reward += reward
            self.done = False
        else:
            self.agent_pos = next_pos
            if(self.agent_pos == self.goal_pos):
                reward = 10
                self.total_reward += reward
                self.done = True
            elif(self.agent_pos in self.obstacles):
                reward = -5
                self.total_reward += reward
                self.done = True
            else:
                reward = -1
                self.total_reward += reward
                self.done = False

        info = {"step_count": self.step_count, "total_reward": self.total_reward}

        return self.get_state(), reward, self.done, info

    def get_state(self):
        """Return the current state representation."""
        # TODO: Choose a state representation.
        # Option 1: return agent position only.
        # Option 2: return agent position + goal position.
        # Option 3: return a grid matrix.
        return np.array(self.agent_pos, dtype=np.float32)
    
    def is_out(self, pos):
        """Check if the position is within bounds and not an obstacle."""
        if pos[0] < 0 or pos[0] >= self.size or pos[1] < 0 or pos[1] >= self.size:
            return True
        return False
    
    def pos_to_index(self, pos):
        row, col = pos
        return row * self.size + col

    def index_to_pos(self, index):
        row = index // self.size
        col = index % self.size
        return (row, col)

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
