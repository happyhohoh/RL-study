# Week03 前置复盘：REINFORCE 理解误区与修正

这份复盘用来记录进入 Week03 前，对 REINFORCE 和 Safety Shield 的理解整理。重点不是证明哪里错了，而是把容易混淆的地方变成后续学习的经验。

## 1. 我已经理解正确的部分

- 我已经理解 env、policy、train loop 三者的基本关系：环境给 state，policy 根据 state 产生动作，train loop 收集轨迹并更新 policy。
- 我已经理解 reward 和 return 的区别：reward 是一步反馈，return 是从某一步开始往后的累计回报。
- 我已经理解 REINFORCE 是基于完整 episode 的 Monte Carlo 方法，需要等一条轨迹结束后再根据累计回报更新策略。
- 我已经理解 Safety Shield 的基本作用：在 policy 输出动作之后、环境执行动作之前，检查动作是否危险。
- 我已经意识到碰撞率高说明模型不能只看 reward，安全任务必须单独观察 collision rate。

## 2. 我需要修正的理解

### 误区 1：“policy 负责找到最优 action”

修正：

当前阶段 policy 不是直接输出最优动作，而是输出动作概率分布 `pi_theta(a|s)`。训练的目标是让高回报动作在相似状态下概率更大。

### 误区 2：“动作概率乘奖励就是 REINFORCE”

修正：

REINFORCE 的核心不是简单的“概率 × 奖励”，而是使用：

```text
log pi_theta(a_t|s_t) * G_t
```

其中 `log_prob` 是实际采样动作的 log 概率，`G_t` 是从当前时刻开始的累计回报。

### 误区 3：“当前 reward 会受未来影响”

修正：

reward 本身不会受未来影响，reward 是环境当前一步立即给出的反馈。受未来影响的是 return，也就是当前动作的训练信号会包含未来奖励。

### 误区 4：“argmax 选择当前最大概率动作就可以训练”

修正：

训练阶段需要 `sample()` 保持探索，否则随机初始化阶段可能过早陷入错误动作。评估阶段才适合使用 `argmax`。

### 误区 5：“state 只包含 agent 位置也可以做随机目标导航”

修正：

固定目标时 agent 可能记住路线；随机目标时必须把 goal 位置加入 state，否则 agent 不知道要去哪。Week03 应改成：

```text
[agent_row, agent_col, goal_row, goal_col]
```

### 误区 6：“有 reward 提升就说明模型学得好”

修正：

在导航和无人机任务中，reward 高但 collision rate 高，说明模型可能通过危险行为获得收益。安全任务必须同时看 success rate 和 collision rate。

## 3. 核心公式复盘

```text
G_t = r_t + gamma * G_{t+1}

loss = -sum log pi_theta(a_t|s_t) * G_t
```

解释：

- `G_t`：从第 t 步开始的累计回报。
- `log pi_theta(a_t|s_t)`：当前策略选择实际动作的 log 概率。
- 负号：PyTorch 优化器默认最小化 loss，而策略梯度目标是最大化期望回报。

## 4. 本阶段最重要的一句话

REINFORCE 不是直接判断哪个动作对，而是通过 episode 的累计回报 G_t，调整被采样动作的 log probability，让高回报动作更容易再次被选中。

## 5. 下一阶段提醒

Week03 不要继续单纯增加训练轮数，而要完成：

- 固定地图 → 随机地图
- 固定目标 → 随机目标
- 只看 reward → 同时看 success rate 和 collision rate
- policy 直接执行动作 → policy 输出动作后经过 Safety Shield 检查
