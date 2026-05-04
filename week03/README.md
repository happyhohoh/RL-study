# Week 03：导航、避障与 Safety Shield

## 1. 本周目标

本周在 GridWorld 中加入更接近导航任务的元素：随机起点、随机目标、随机障碍物、碰撞检测和 Safety Shield。目标不是单纯让 reward 变高，而是比较有无安全约束时，智能体在成功率和碰撞率上的差异。

## 2. 相比 Week02 的变化

Week02 更像固定地图上的基础 REINFORCE 练习。Week03 做了以下扩展：

- 固定起点改为随机起点。
- 固定目标改为随机目标。
- 固定障碍物改为随机障碍物。
- state 从 `[agent_row, agent_col]` 扩展为 `[agent_row, agent_col, goal_row, goal_col]`。
- 新增 Safety Shield，用来在执行动作前检查危险动作。
- 训练时同时记录 reward、success、collision 和 shield blocked 等指标。

## 3. 环境设计

环境文件是 `safe_gridworld_env.py`，动作空间保持不变：

- `0`: up
- `1`: down
- `2`: left
- `3`: right

每个 episode 开始时，环境会随机生成：

- agent 起点
- goal 目标点
- obstacle 障碍物

奖励设计：

- 每走一步：`-1`
- 到达目标：`+20`
- 撞障碍物：`-10`
- 越界：`-10`
- Safety Shield 拦截危险动作：`-3`

## 4. Safety Shield 机制

Policy 先输出动作概率分布，然后采样或选择一个动作。Safety Shield 在动作真正执行前检查下一步位置：

- 如果下一步安全，执行原动作。
- 如果下一步越界或撞障碍物，并且 `use_safety_shield=True`，则拦截动作，agent 停留原地，得到 `-3` 惩罚。
- 如果 `use_safety_shield=False`，危险动作会导致 collision，并结束 episode。

当前采用的是最简单的方案 A：危险动作被拦截后停留原地。后续可以比较“停留原地”和“替换为安全动作”两种策略。

## 5. 训练方法：REINFORCE

训练脚本是 `train_reinforce_safe.py`。本周仍然使用 REINFORCE：

```text
G_t = r_t + gamma * G_{t+1}
loss = -sum(log_prob_t * G_t)
```

训练时使用 `sample()` 采样动作，保留探索能力。评估时使用 `argmax()`，观察当前策略最倾向选择的动作。

## 6. 实验指标

训练过程记录：

- `episode_rewards`
- `episode_steps`
- `success_history`
- `collision_history`
- `shield_blocked_history`
- `loss_history`

评估过程输出：

- `success_rate`
- `collision_rate`
- `average_reward`
- `average_steps`
- `average_shield_blocked_count`

## 7. 如何运行

在项目根目录运行：

```bash
python week03/safe_gridworld_env.py
```

启动训练：

```bash
python week03/train_reinforce_safe.py
```

训练会保存模型：

```text
week03/results/models/policy_without_shield.pth
week03/results/models/policy_with_shield.pth
```

并保存曲线：

```text
week03/results/plots/reward_curve.png
week03/results/plots/success_curve.png
week03/results/plots/collision_curve.png
week03/results/plots/shield_blocked_curve.png
```

评估默认会加载 `policy_with_shield.pth`：

```bash
python week03/evaluate.py
```

也可以在 Python 中手动调用：

```python
from week03.evaluate import evaluate

evaluate("week03/results/models/policy_with_shield.pth", use_safety_shield=True)
evaluate("week03/results/models/policy_without_shield.pth", use_safety_shield=False)
```

## 8. 当前 TODO

- 把 state 扩展为局部观测矩阵，让 agent 看到附近障碍物。
- 加入距离目标的 shaping reward，加快导航学习。
- 比较两种 Shield 策略：停留原地 vs 替换成安全动作。
- 保存训练历史为 CSV，方便后续分析。
- 加入多随机种子实验，减少偶然性。

## 9. 和无人机导航的关系

GridWorld 不是无人机真实动力学模型，但它能对应无人机导航中的几个核心问题：

- agent 位置对应无人机位置。
- goal 对应目标航点。
- obstacle 对应建筑物、禁飞区或未知障碍。
- collision rate 对应安全风险。
- Safety Shield 对应飞控或安全层，在策略输出动作后做最后检查。

真实无人机还需要考虑连续空间、速度、加速度、传感器噪声、动力学约束和实时避障。本周的 GridWorld 是一个简化版入口，重点是先把“策略学习”和“安全约束”分清楚。
