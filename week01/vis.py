import matplotlib.pyplot as plt

def moving_average(values, window=50):
    averages = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        window_values = values[start:i + 1]
        averages.append(sum(window_values) / len(window_values))
    return averages


def plot_training_progress(
    loss_history,
    total_reward_history,
    step_count_history,
    success_history,
    window=50,
):
    episodes = range(1, len(loss_history) + 1)
    reward_ma = moving_average(total_reward_history, window)
    step_ma = moving_average(step_count_history, window)
    success_rate = moving_average(success_history, window)

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(episodes, loss_history, label='Loss')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('Training Loss over Episodes')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(episodes, total_reward_history, label='Total Reward', color='orange', alpha=0.3)
    plt.plot(episodes, reward_ma, label=f'Reward MA({window})', color='red')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Reward over Episodes')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(episodes, step_count_history, label='Step Count', color='green', alpha=0.3)
    plt.plot(episodes, step_ma, label=f'Step MA({window})', color='darkgreen')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.title('Steps per Episode')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(episodes, success_rate, label=f'Success Rate MA({window})', color='purple')
    plt.xlabel('Episode')
    plt.ylabel('Success Rate')
    plt.ylim(-0.05, 1.05)
    plt.title('Recent Success Rate')
    plt.legend()

    plt.tight_layout()
    plt.show()
