# Week 01 笔记

## 环境、策略网络、训练循环

- `GridWorldEnv` 负责环境动力学：位置变化、边界判断、障碍物、奖励、episode 是否结束。
- `MLPPolicy` 负责把 `state` 映射成 `action_probs`。
- 训练循环把两者连起来：先把状态交给 policy，policy 选动作，再把动作交给 env，env 返回下一状态、奖励和 done。

## state 从哪里来

state 由环境产生，不是 policy 产生。

一局 episode 开始时：

```python
state = env.reset()
```

每执行一步动作后：

```python
next_state, reward, done, info = env.step(action)
```

当前 GridWorld 的 state 是 agent 的二维位置：

```python
state = np.array([row, col], dtype=np.float32)
```

所以当前 policy 的输入输出维度是：

```python
policy = MLPPolicy(state_dim=2, action_dim=4)
```

含义是：

```text
state_dim = 2   # [row, col]
action_dim = 4  # UP, DOWN, LEFT, RIGHT
```

如果以后把 goal 也放进状态：

```python
state = np.array([agent_row, agent_col, goal_row, goal_col], dtype=np.float32)
```

那时才需要：

```python
state_dim = 4
```

## 一局 episode 的基本流程

```python
state = env.reset()
done = False

while not done:
    action_probs = policy(state)
    action = sample_from(action_probs)
    next_state, reward, done, info = env.step(action)
    state = next_state
```

训练时，while 循环里主要是在收集这一局的数据：

```python
log_probs = []
rewards = []
```

- `log_probs`：记录 policy 在每一步选择当前 action 的 log 概率。
- `rewards`：记录环境在每一步返回的 reward。

## REINFORCE 的核心思想

REINFORCE 是一种最基础的 policy gradient 方法。

它不直接学习“某个状态的价值是多少”，而是直接调整 policy：

```text
如果某个动作后面带来了高回报，就提高以后在类似状态下选择这个动作的概率。
如果某个动作后面带来了低回报，就降低以后在类似状态下选择这个动作的概率。
```

policy 输出的是动作概率：

```text
state -> [P(UP), P(DOWN), P(LEFT), P(RIGHT)]
```

例如：

```text
[0.10, 0.40, 0.20, 0.30]
```

表示当前状态下，policy 认为：

```text
UP    概率 0.10
DOWN  概率 0.40
LEFT  概率 0.20
RIGHT 概率 0.30
```

训练时不要直接用 `argmax` 固定选择最大概率动作，而是从这个概率分布里采样。这样 agent 才有机会探索不同路径。

## return 是什么

reward 是当前这一步的即时反馈。

return 是从当前这一步开始，未来所有 reward 的折扣累计和。

公式是：

```text
G_t = r_t + gamma * G_{t+1}
```

其中：

- `G_t` 是第 `t` 步的 return。
- `r_t` 是第 `t` 步得到的 reward。
- `gamma` 是折扣因子，常用 `0.99`。

例如一局 rewards 是：

```text
[-1, -1, -1, 10]
```

如果暂时不考虑折扣，那么每一步的 return 大概是：

```text
[7, 8, 9, 10]
```

含义是：

```text
第 0 步之后，总回报是 7
第 1 步之后，总回报是 8
第 2 步之后，总回报是 9
第 3 步之后，总回报是 10
```

## REINFORCE 的 loss

基础写法是：

```python
returns = torch.tensor(returns)
log_probs = torch.stack(log_probs)
loss = -torch.sum(log_probs * returns)
```

它对应的数学直觉是：

```text
loss = - Σ log_prob(action_t | state_t) * return_t
```

其中：

- `log_prob(action_t | state_t)` 表示 policy 在第 `t` 步选择当前动作的 log 概率。
- `return_t` 表示这个动作之后最终带来的累计回报。

为什么要乘起来：

```text
log_prob_t * return_t
```

意思是用 return 给这一步的 action 打分。

如果 `return_t` 高，说明这一步动作整体效果不错，训练会推动 policy 提高这个动作的概率。

如果 `return_t` 低，说明这一步动作整体效果不好，训练会推动 policy 降低这个动作的概率。

为什么前面有负号：

```python
loss = -torch.sum(...)
```

因为 PyTorch 的优化器默认做的是“最小化 loss”，但 policy gradient 的目标是“最大化期望回报”。加负号以后，最小化 loss 就等价于最大化回报。

## loss 是怎么更新 policy 的

优化过程分三步：

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

含义是：

```text
optimizer.zero_grad()
清空上一次留下的梯度。

loss.backward()
根据当前 loss 反向传播，计算 policy 每个参数应该怎么调整。

optimizer.step()
根据梯度真正更新 policy 的参数。
```

如果某一步动作带来的 return 比较高，反向传播会让这一步动作的概率以后更容易变大。

如果某一步动作带来的 return 比较低，反向传播会让这一步动作的概率以后更容易变小。

## Adam 优化器

```python
optimizer = torch.optim.Adam(policy.parameters(), lr=1e-2)
```

这句的意思是：创建一个 Adam 优化器，让它负责更新 `policy` 的所有可训练参数。

- `policy.parameters()`：把 policy 网络里的权重和偏置交给优化器。
- `lr=1e-2`：学习率，也就是每次参数更新的步子大小。

`1e-2` 等于 `0.01`。对于很小的 GridWorld 实验可以先用它。如果训练不稳定，可以试试 `1e-3`。

## returns 和 log_probs 的 shape 对齐

`log_probs * returns` 这一步要求它们的 shape 能一一对应。

假设一局有 `T` 步，最清楚的目标是让二者都是：

```text
[T]
```

也就是：

```text
log_probs.shape == returns.shape
```

### 推荐方式：都变成一维 `[T]`

如果每一步保存的 `log_prob` 是 shape `[1]`，可以在保存时把它变成标量：

```python
log_probs.append(log_prob.squeeze())
```

episode 结束后：

```python
returns = torch.tensor(returns, dtype=torch.float32)
log_probs = torch.stack(log_probs)
```

这时通常是：

```text
returns.shape   -> torch.Size([T])
log_probs.shape -> torch.Size([T])
```

然后就可以：

```python
loss = -torch.sum(log_probs * returns)
```

### 另一种方式：都变成 `[T, 1]`

也可以让二者都变成二维：

```python
returns = torch.tensor(returns, dtype=torch.float32).view(-1, 1)
log_probs = torch.stack(log_probs).view(-1, 1)
```

这时：

```text
returns.shape   -> torch.Size([T, 1])
log_probs.shape -> torch.Size([T, 1])
```

也可以正常相乘。

### 注意

不要写成：

```python
returns = torch.reshape(-1, 1)
```

`reshape` 需要知道要改变谁的形状。应该是：

```python
returns = returns.reshape(-1, 1)
```

或者：

```python
returns = returns.view(-1, 1)
```

训练时可以打印 shape 检查：

```python
print(log_probs.shape, returns.shape)
```
