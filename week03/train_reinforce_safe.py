"""
Week03 REINFORCE training on SafeGridWorldEnv.

This script trains two policies:
- without Safety Shield
- with Safety Shield
"""

from __future__ import annotations

from pathlib import Path
import sys

import torch

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from plot_utils import plot_comparison_curves
    from safe_gridworld_env import SafeGridWorldEnv
except ImportError:
    from .plot_utils import plot_comparison_curves
    from .safe_gridworld_env import SafeGridWorldEnv

from week02.policy import MLPPolicy


def compute_returns(rewards: list[float], gamma: float):
    returns = []
    G = 0.0
    for reward in reversed(rewards):
        G = reward + gamma * G
        returns.insert(0, G)
    return torch.tensor(returns, dtype=torch.float32)


def train_one_experiment(
    use_safety_shield: bool,
    model_path: Path,
    shield_strategy: str = "stay",
    num_episodes: int = 1000,
    gamma: float = 0.99,
    lr: float = 1e-3,
    hidden_dim: int = 64,
    state_dim: int = 4,
    action_dim: int = 4,
):
    env = SafeGridWorldEnv(
        use_safety_shield=use_safety_shield,
        shield_strategy=shield_strategy,
    )
    policy = MLPPolicy(state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    history = {
        "episode_rewards": [],
        "episode_steps": [],
        "success_history": [],
        "collision_history": [],
        "shield_blocked_history": [],
        "loss_history": [],
    }

    label = "with shield" if use_safety_shield else "without shield"
    print(f"\nStart training: {label}")

    if tqdm is None:
        print("tqdm is not installed. Run `pip install -r requirements.txt` to see a progress bar.")
        episode_iter = range(num_episodes)
    else:
        episode_iter = tqdm(range(num_episodes), desc=label, unit="episode")

    for episode in episode_iter:
        state = env.reset()
        log_probs = []
        rewards = []
        collision_count = 0
        shield_blocked_count = 0
        success = False

        while not env.done:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action_probs = policy(state_tensor)

            # Use sample() during training to keep exploration.
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
            log_probs.append(action_dist.log_prob(action))

            state, reward, done, info = env.step(action.item())
            rewards.append(reward)
            success = success or info["success"]
            collision_count += int(info["collision"])
            shield_blocked_count += int(info["shield_blocked"])

        returns = compute_returns(rewards, gamma)

        # Return normalization is a stability trick, not the core REINFORCE formula.
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        log_probs_tensor = torch.stack(log_probs).view(-1)
        loss = -torch.sum(log_probs_tensor * returns)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        history["episode_rewards"].append(env.total_reward)
        history["episode_steps"].append(env.step_count)
        history["success_history"].append(float(success))
        history["collision_history"].append(float(collision_count > 0))
        history["shield_blocked_history"].append(float(shield_blocked_count))
        history["loss_history"].append(loss.item())

        recent_rewards = history["episode_rewards"][-100:]
        recent_success = history["success_history"][-100:]
        recent_collision = history["collision_history"][-100:]

        avg_reward = sum(recent_rewards) / len(recent_rewards)
        success_rate = sum(recent_success) / len(recent_success)
        collision_rate = sum(recent_collision) / len(recent_collision)

        if tqdm is not None:
            episode_iter.set_postfix(
                avg_reward=f"{avg_reward:.2f}",
                success=f"{success_rate:.2f}",
                collision=f"{collision_rate:.2f}",
                shield_blocks=shield_blocked_count,
            )
        elif (episode + 1) % 50 == 0 or episode == 0:
            print(
                f"{label} | Episode {episode + 1}, "
                f"AvgReward: {avg_reward:.2f}, "
                f"SuccessRate: {success_rate:.2f}, "
                f"CollisionRate: {collision_rate:.2f}"
            )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(policy.state_dict(), model_path)
    print(f"Saved model to {model_path}")
    return history


def main():
    results_dir = Path(__file__).resolve().parent / "results"
    models_dir = results_dir / "models"
    plots_dir = results_dir / "plots"
    models_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    num_episodes = 10000
    gamma = 0.99
    lr = 1e-3
    hidden_dim = 64
    state_dim = 4
    action_dim = 4

    experiments = [
        {
            "label": "without shield",
            "use_safety_shield": False,
            "shield_strategy": "stay",
            "model_name": "policy_without_shield.pth",
        },
        {
            "label": "shield stay",
            "use_safety_shield": True,
            "shield_strategy": "stay",
            "model_name": "policy_shield_stay.pth",
        },
        {
            "label": "shield replace",
            "use_safety_shield": True,
            "shield_strategy": "replace",
            "model_name": "policy_shield_replace.pth",
        },
    ]

    histories = {}

    for exp in experiments:
        histories[exp["label"]] = train_one_experiment(
            use_safety_shield=exp["use_safety_shield"],
            shield_strategy=exp["shield_strategy"],
            model_path=models_dir / exp["model_name"],
            num_episodes=num_episodes,
            gamma=gamma,
            lr=lr,
            hidden_dim=hidden_dim,
            state_dim=state_dim,
            action_dim=action_dim,
        )

    plot_comparison_curves(histories, results_dir=plots_dir)



if __name__ == "__main__":
    main()
