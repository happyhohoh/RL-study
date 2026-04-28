# 具身智能 / 无人机 RL 工程能力提升计划

## 1. 当前定位

当前阶段的目标不是直接训练 VLA、大模型或复杂无人机系统，而是先补齐以下能力：

1. 强化学习代码实践能力
2. PyTorch 基础工程能力
3. 简化环境建模能力
4. 任务拆解与数据结构设计能力
5. 从“AI 代写代码”转向“自己设计模块，AI 辅助实现”

当前优先做：

- Python
- PyTorch
- 简化 GridWorld 环境
- RL 训练闭环
- 导航、避障、安全规则
- README / GitHub 项目整理

---

## 2. 三周短期目标

本阶段只做 3 周计划，不做长期大规划。

总目标：

```text
重新建立 RL + PyTorch + 小型环境构建的代码手感
```

每周工作量：

- 每周 2–3 天
- 每天 2–3 小时
- 每周总计约 6–8 小时

---

# Week 01：补 PyTorch + RL 代码手感

## 目标

能够自己写一个最小 PyTorch `MLPPolicy`，并理解 RL 中 policy network 是如何根据 state 输出 action 的。

本周暂时不追求训练复杂 agent，只做基础代码和环境封装。

## Day 01：PyTorch 基础 MLPPolicy

### 任务

编写一个最小的 `MLPPolicy`。

输入：

```text
state_dim
```

输出：

```text
action_dim
```

网络结构：

```text
Linear → ReLU → Linear → Softmax
```

### 需要理解

- `nn.Module` 是什么
- `forward()` 是什么
- `tensor.shape` 如何查看
- batch 维度是什么
- 为什么离散动作策略需要输出动作概率
- 为什么 policy network 输出的是动作分布，而不是直接动作

## Day 02：手写极简 GridWorld

### 任务

不使用 Gym，自己手写一个 5×5 的 GridWorld 环境。

环境包含：

- 地图大小
- agent 位置
- goal 位置
- obstacle 位置
- `reset()`
- `step(action)`
- `render()`

动作空间：

```text
0: up
1: down
2: left
3: right
```

建议数据结构：

```python
grid: np.ndarray
agent_pos: tuple[int, int]
goal_pos: tuple[int, int]
obstacles: set[tuple[int, int]]
```

### 需要理解

- 环境 env 本质上是什么
- state 如何表示
- action 如何改变 state
- reward 如何设计
- done 什么时候为 True
- reset 和 step 的职责分别是什么

## Day 03：整理笔记

在 `week01/notes.md` 中回答：

1. 我的 state 是怎么表示的？
2. 我的 action 是怎么作用到环境里的？
3. 我的 reward 是怎么设计的？
4. `reset()` 做了什么？
5. `step(action)` 做了什么？
6. 当前环境和无人机导航问题有什么相似之处？
7. 当前环境和真实无人机导航问题有什么差距？

本周交付物：

```text
week01/
  mlp_policy.py
  gridworld_env.py
  test_env.py
  notes.md
```

---

# Week 02：在自己写的 GridWorld 上跑一个 RL

## 目标

把 RL 理论变成代码闭环。

本周不直接写 PPO，先写更简单的 REINFORCE，用于建立完整训练流程。

## 核心流程

```text
初始化 policy network
↓
agent 与 env 交互
↓
采样一条 episode
↓
记录 states、actions、rewards
↓
计算 return
↓
计算 policy loss
↓
反向传播更新 policy
↓
重复训练
```

## 需要理解

- trajectory
- episode
- return
- policy gradient
- log probability
- reward normalization
- exploration
- reward curve

本周交付物：

```text
week02/
  train_reinforce.py
  policy.py
  gridworld_env.py
  plot_rewards.py
  results/
    reward_curve.png
  notes.md
```

---

# Week 03：加入导航、避障与安全规则

## 目标

将简单 GridWorld 扩展成一个小型导航任务。

重点不是复杂算法，而是理解：

```text
导航任务 + 障碍物 + 碰撞惩罚 + 安全约束
```

## 新增功能

- 随机起点
- 随机目标点
- 随机障碍物
- 最大步数限制
- 碰撞检测
- 到达目标检测

## 奖励设计建议

```text
靠近目标：小奖励
远离目标：小惩罚
撞到障碍物：较大惩罚
到达目标：较大奖励
超过最大步数：结束 episode
```

## Safety Shield

最小规则：

```text
如果下一步动作会撞到障碍物，则拦截该动作，让 agent 停留原地，并给予惩罚。
```

对应无人机中的思想：

```text
RL policy 输出动作
↓
Safety Shield 检查动作是否安全
↓
安全则执行
↓
不安全则拦截或替换动作
```

## 对比实验

比较两组结果：

1. 无 safety shield
2. 有 safety shield

记录指标：

- episode reward
- success rate
- collision count
- collision rate

本周交付物：

```text
week03/
  safe_gridworld_env.py
  train_reinforce_safe.py
  evaluate.py
  results/
    reward_curve.png
    collision_curve.png
  notes.md
  README.md
```

---

# 3. 每周复盘机制

## 固定复盘时间

```text
每周六上午 10:00
```

## 每周复盘提交格式

```text
本周完成：
1.
2.
3.

遇到的问题：
1.
2.
3.

我不理解的地方：
1.
2.
3.

代码 / 截图 / 报错：
粘贴代码、截图或报错信息

我希望你帮我检查：
1. 代码结构是否合理
2. 概念理解是否正确
3. README 是否适合放到 GitHub
4. 是否可以整理成简历项目描述
```

---

# 4. 当前第一步任务

完成 Week 01 / Day 01：

## 编写最小 PyTorch MLPPolicy

要求：

1. 使用 `torch`
2. 使用 `nn.Module`
3. 输入 `state_dim`
4. 输出 `action_dim`
5. 网络结构：

```text
Linear → ReLU → Linear → Softmax
```

6. 测试一个 batch 输入：

```python
state = torch.randn(4, state_dim)
```

7. 输出动作概率，并检查 shape 是否为：

```python
[4, action_dim]
```
