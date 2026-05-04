"""
Plot helpers for Week03 training.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def moving_average(values, window: int = 50):
    """Compute moving average for smoother learning curves."""
    values = np.asarray(values, dtype=np.float32)
    if len(values) == 0:
        return values
    if len(values) < window:
        return values

    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(values, kernel, mode="valid")


def _plot_single_curve(
    values,
    title: str,
    ylabel: str,
    save_path: Path,
    window: int = 50,
):
    plt.figure(figsize=(8, 5))
    plt.plot(values, alpha=0.25, label="raw")
    smooth = moving_average(values, window=window)
    if len(smooth) > 0:
        start_x = window - 1 if len(values) >= window else 0
        plt.plot(range(start_x, start_x + len(smooth)), smooth, label=f"MA({window})")
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_training_curves(history: dict, results_dir: str | Path = "week03/results/plots"):
    """Save single-experiment curves."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    _plot_single_curve(
        history["episode_rewards"],
        "Episode Reward",
        "Reward",
        results_dir / "reward_curve.png",
    )
    _plot_single_curve(
        history["success_history"],
        "Success",
        "Success",
        results_dir / "success_curve.png",
    )
    _plot_single_curve(
        history["collision_history"],
        "Collision",
        "Collision",
        results_dir / "collision_curve.png",
    )
    _plot_single_curve(
        history["shield_blocked_history"],
        "Shield Blocked Count",
        "Blocked Count",
        results_dir / "shield_blocked_curve.png",
    )


def plot_comparison_curves(
    histories: dict,
    results_dir: str | Path = "week03/results/plots",
    window: int = 50,
):
    """Save comparison curves for with-shield and without-shield experiments."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    specs = [
        ("episode_rewards", "Reward", "reward_curve.png"),
        ("success_history", "Success Rate Signal", "success_curve.png"),
        ("collision_history", "Collision Signal", "collision_curve.png"),
        ("shield_blocked_history", "Shield Blocked Count", "shield_blocked_curve.png"),
    ]

    for key, title, filename in specs:
        plt.figure(figsize=(8, 5))
        for label, history in histories.items():
            values = history[key]
            smooth = moving_average(values, window=window)
            if len(smooth) > 0:
                start_x = window - 1 if len(values) >= window else 0
                plt.plot(
                    range(start_x, start_x + len(smooth)),
                    smooth,
                    label=f"{label} MA({window})",
                )
            else:
                plt.plot(values, label=label)

        plt.title(title)
        plt.xlabel("Episode")
        plt.ylabel(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(results_dir / filename)
        plt.close()
