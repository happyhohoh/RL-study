from gridworld_env import GridWorldEnv
from mlp_policy import MLPPolicy
from vis import plot_training_progress
import torch

def main():
    env = GridWorldEnv(size=5, max_steps=20)
    env.reset()
    policy = MLPPolicy(state_dim=2, action_dim=4)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-2)
    env.render()
    # 记录每10轮的loss和total_reward，记录趋势
    loss_history = []
    total_reward_history = []
    step_count_history = []
    success_history = []

    for episode in range(1000):
        env.reset()
        log_probs = []
        rewards = []

        while not env.done:
            state = env.get_state()
            # print("state:", state,"reward:", env.total_reward, "step_count:", env.step_count)
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            # with torch.no_grad():
            #     action_probs = policy(state_tensor)

            action_probs = policy(state_tensor)

            # 用采样选动作
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
            state, reward, done, info = env.step(action.item())
            rewards.append(reward)
            log_prob = action_dist.log_prob(action)
            log_probs.append(log_prob)

        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + 0.99 * G
            returns.insert(0, G)

        # 计算loss
        returns = torch.tensor(returns, dtype=torch.float32)
        log_probs = torch.stack(log_probs).view(-1)  # shape [episode_length]
        loss = -torch.sum(log_probs * returns)

        # 更新策略网络
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 每10轮记录下loss和total_reward
        success = env.agent_pos == env.goal_pos
        loss_history.append(loss.item())
        total_reward_history.append(env.total_reward)
        step_count_history.append(env.step_count)
        success_history.append(float(success))

        if (episode + 1) % 10 == 0:
            recent_success = sum(success_history[-100:]) / len(success_history[-100:])
            print(
                f"Episode {episode + 1}, "
                f"Loss: {loss.item():.3f}, "
                f"Total Reward: {env.total_reward}, "
                f"Steps: {env.step_count}, "
                f"Success: {success}, "
                f"Recent Success Rate: {recent_success:.2f}"
            )

    env.render()
    plot_training_progress(
        loss_history,
        total_reward_history,
        step_count_history,
        success_history,
    )

if __name__ == "__main__":
    main()
