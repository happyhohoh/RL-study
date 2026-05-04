"""
Evaluate a trained Week03 policy.
"""

from __future__ import annotations

from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from safe_gridworld_env import SafeGridWorldEnv
except ImportError:
    from .safe_gridworld_env import SafeGridWorldEnv

from week02.policy import MLPPolicy


def evaluate(
    model_path: str | Path,
    use_safety_shield: bool = True,
    shield_strategy: str = "stay",
    num_episodes: int = 100,
    hidden_dim: int = 64,
    state_dim: int = 4,
    action_dim: int = 4,
):
    model_path = Path(model_path)
    env = SafeGridWorldEnv(
        use_safety_shield=use_safety_shield,
        shield_strategy=shield_strategy,
    )
    policy = MLPPolicy(state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim)
    policy.load_state_dict(torch.load(model_path, map_location="cpu"))
    policy.eval()

    success_count = 0
    collision_count = 0
    total_reward = 0.0
    total_steps = 0
    total_shield_blocked = 0

    for _ in range(num_episodes):
        state = env.reset()
        episode_success = False
        episode_collision = False
        episode_shield_blocked = 0

        while not env.done:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action_probs = policy(state_tensor)

            # Use argmax during evaluation to test the current greedy policy.
            action = torch.argmax(action_probs, dim=-1)
            state, reward, done, info = env.step(action.item())

            episode_success = episode_success or info["success"]
            episode_collision = episode_collision or info["collision"]
            episode_shield_blocked += int(info["shield_blocked"])

        success_count += int(episode_success)
        collision_count += int(episode_collision)
        total_reward += env.total_reward
        total_steps += env.step_count
        total_shield_blocked += episode_shield_blocked

    result = {
        "success_rate": success_count / num_episodes,
        "collision_rate": collision_count / num_episodes,
        "avg_reward": total_reward / num_episodes,
        "avg_steps": total_steps / num_episodes,
        "avg_shield_blocked": total_shield_blocked / num_episodes,
    }

    print("Evaluation result:")
    print(f"success_rate: {result['success_rate']:.2f}")
    print(f"collision_rate: {result['collision_rate']:.2f}")
    print(f"avg_reward: {result['avg_reward']:.2f}")
    print(f"avg_steps: {result['avg_steps']:.1f}")
    print(f"avg_shield_blocked: {result['avg_shield_blocked']:.1f}")
    return result


if __name__ == "__main__":
    results_dir = Path(__file__).resolve().parent / "results"
    default_model = results_dir / "models" / "policy_without_shield.pth"
    if not default_model.exists():
        raise FileNotFoundError(
            f"Model not found: {default_model}. Please run train_reinforce_safe.py first."
        )
    evaluate(default_model, use_safety_shield=False)
