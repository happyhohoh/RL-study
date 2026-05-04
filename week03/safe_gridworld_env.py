"""
Week 03
Task: GridWorld with random navigation, obstacles, collision check, and Safety Shield.

Actions:
0: up
1: down
2: left
3: right
"""

from __future__ import annotations

import random
from typing import Iterable

import numpy as np


class SafeGridWorldEnv:
    """A small navigation environment for REINFORCE safety experiments."""

    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

    ACTION_DELTAS = {
        UP: (-1, 0),
        DOWN: (1, 0),
        LEFT: (0, -1),
        RIGHT: (0, 1),
    }

    def __init__(
        self,
        size: int = 6,
        obstacle_count: int = 6,
        max_steps: int = 50,
        use_safety_shield: bool = True,
        shield_strategy: str = "stay",
        seed: int | None = None,
    ):
        self.size = size
        self.obstacle_count = obstacle_count
        self.max_steps = max_steps
        self.use_safety_shield = use_safety_shield
        if shield_strategy not in {"stay", "replace"}:
            raise ValueError("shield_strategy must be 'stay' or 'replace'.")
        self.shield_strategy = shield_strategy
        self.shield_stay_penalty = -5.0
        self.shield_replace_penalty = -2.0
        self.rng = random.Random(seed)

        self.agent_pos: tuple[int, int] | None = None
        self.goal_pos: tuple[int, int] | None = None
        self.obstacles: set[tuple[int, int]] = set()
        self.total_reward = 0.0
        self.step_count = 0
        self.done = False

    def reset(self):
        """Reset with random start, random goal, and random obstacles."""
        self.obstacles = set()
        self.agent_pos = self.sample_empty_cell()
        self.goal_pos = self.sample_empty_cell(exclude={self.agent_pos})
        self.obstacles = self.sample_obstacles(exclude={self.agent_pos, self.goal_pos})

        self.total_reward = 0.0
        self.step_count = 0
        self.done = False
        return self.get_state()

    def step(self, action: int):
        """Execute one action in the environment."""
        if self.done:
            return self.get_state(), 0.0, self.done, self._make_info(
                success=self.agent_pos == self.goal_pos,
                collision=False,
                shield_blocked=False,
            )

        next_pos = self.get_next_pos(action)
        _, should_block = self.shield_action(action)
        is_dangerous = self.is_out(next_pos) or self.is_obstacle(next_pos)
        collision = False
        shield_blocked = False
        success = False

        self.step_count += 1

        if should_block:
            shield_blocked = True
            if self.shield_strategy == "stay":
                reward = self.shield_stay_penalty
                self.done = False
            else:
                reward, success = self._replace_blocked_action()
        elif is_dangerous:
            reward = -10.0
            collision = True
            self.done = True
        else:
            self.agent_pos = next_pos
            if self.agent_pos == self.goal_pos:
                reward = 20.0
                success = True
                self.done = True
            else:
                reward = -1.0

        if self.step_count >= self.max_steps and not self.done:
            self.done = True

        self.total_reward += reward
        info = self._make_info(
            success=success,
            collision=collision,
            shield_blocked=shield_blocked,
        )
        return self.get_state(), reward, self.done, info

    def _replace_blocked_action(self) -> tuple[float, bool]:
        """Replace a dangerous action with a random safe action."""
        safe_actions = self.get_safe_actions()
        if not safe_actions:
            self.done = True
            return -100.0, False

        action = self.rng.choice(sorted(safe_actions))
        self.agent_pos = self.get_next_pos(action)

        if self.agent_pos == self.goal_pos:
            self.done = True
            return 20.0 + self.shield_replace_penalty, True

        return -1.0 + self.shield_replace_penalty, False

    def get_state(self):
        """Return [agent_row, agent_col, goal_row, goal_col]."""
        # TODO: 后续可以把 state 扩展为局部观测矩阵。
        # TODO: 后续可以加入距离目标的 shaping reward。
        assert self.agent_pos is not None
        assert self.goal_pos is not None
        return np.array(
            [
                self.agent_pos[0],
                self.agent_pos[1],
                self.goal_pos[0],
                self.goal_pos[1],
            ],
            dtype=np.float32,
        )

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

    def is_out(self, pos: tuple[int, int]) -> bool:
        row, col = pos
        return row < 0 or row >= self.size or col < 0 or col >= self.size

    def is_obstacle(self, pos: tuple[int, int]) -> bool:
        return pos in self.obstacles

    def get_next_pos(self, action: int) -> tuple[int, int]:
        if action not in self.ACTION_DELTAS:
            raise ValueError(f"Invalid action: {action}")

        assert self.agent_pos is not None
        delta_row, delta_col = self.ACTION_DELTAS[action]
        return self.agent_pos[0] + delta_row, self.agent_pos[1] + delta_col

    def shield_action(self, action: int) -> tuple[int, bool]:
        """Return the original action and whether Safety Shield blocks it.

        当前 Safety Shield 采用方案 A，所以危险动作不会被替换成别的动作；
        真正的拦截逻辑在 step() 中执行，agent 会停留原地并得到惩罚。
        """
        next_pos = self.get_next_pos(action)
        blocked = self.use_safety_shield and (
            self.is_out(next_pos) or self.is_obstacle(next_pos)
        )
        return action, blocked

    def sample_empty_cell(
        self,
        exclude: Iterable[tuple[int, int]] | None = None,
    ) -> tuple[int, int]:
        """Sample one cell that is not excluded and not an obstacle."""
        excluded = set(exclude or set())
        candidates = [
            (row, col)
            for row in range(self.size)
            for col in range(self.size)
            if (row, col) not in excluded and (row, col) not in self.obstacles
        ]
        if not candidates:
            raise ValueError("No empty cell available.")
        return self.rng.choice(candidates)

    def sample_obstacles(
        self,
        exclude: Iterable[tuple[int, int]] | None = None,
    ) -> set[tuple[int, int]]:
        """Sample random obstacles that do not overlap start or goal."""
        excluded = set(exclude or set())
        candidates = [
            (row, col)
            for row in range(self.size)
            for col in range(self.size)
            if (row, col) not in excluded
        ]
        count = min(self.obstacle_count, len(candidates))
        return set(self.rng.sample(candidates, count))

    def _make_info(
        self,
        success: bool,
        collision: bool,
        shield_blocked: bool,
    ) -> dict:
        return {
            "success": bool(success),
            "collision": bool(collision),
            "shield_blocked": bool(shield_blocked),
            "step_count": self.step_count,
            "total_reward": self.total_reward,
        }
    
    def get_safe_actions(self) -> set[int]:
        """Return the set of actions that are not blocked by the Safety Shield."""
        safe_actions = set()
        for action in self.ACTION_DELTAS.keys():
            _, blocked = self.shield_action(action)
            if not blocked:
                safe_actions.add(action)
        return safe_actions


if __name__ == "__main__":
    env = SafeGridWorldEnv(use_safety_shield=True, seed=0)
    state = env.reset()
    print("initial state:", state)
    env.render()

    for action in [0, 3, 3, 1, 1]:
        state, reward, done, info = env.step(action)
        print(f"action={action}, reward={reward}, done={done}, info={info}")
        env.render()
        if done:
            break
