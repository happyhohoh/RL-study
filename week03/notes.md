# Week03 学习笔记

## 1. 为什么 state 要加入 goal 位置

固定目标时，agent 只看自己的位置，也可能慢慢记住一条路线。随机目标时，目标每次都变，如果 state 里没有 goal 位置，policy 不知道自己要去哪。

所以 Week03 的 state 改成：

```text
[agent_row, agent_col, goal_row, goal_col]
```

## 2. 为什么随机起点和随机目标更重要

固定起点和固定目标更像“背路线”。随机起点和随机目标会迫使 policy 学习更一般的导航规律，例如向目标靠近、避开障碍，而不是只记住某一条路径。

## 3. obstacle、collision、done 的关系

obstacle 是地图上的障碍物位置。agent 如果尝试走到 obstacle 上，就发生 collision。

在没有 Safety Shield 时，collision 会带来较大负奖励，并结束 episode，也就是 `done=True`。

在启用 Safety Shield 时，危险动作会被提前拦截，agent 停留原地。这时不记为真实 collision，但会记录 `shield_blocked=True`。

## 4. Safety Shield 和 RL policy 的关系

policy 负责根据 state 输出动作概率分布。Safety Shield 不替代 policy 学习，它是在动作执行前增加一层安全检查。

可以理解为：

```text
state -> policy -> action -> safety shield -> environment
```

policy 仍然要学习如何到达目标；shield 只是减少明显危险动作真正发生的机会。

## 5. 为什么只看 reward 不够

导航和无人机任务里，reward 高不一定代表安全。如果一个策略经常撞障碍，但偶尔很快到达目标，平均 reward 可能看起来还不错。

因此必须同时看：

- success rate：是否能到达目标
- collision rate：是否经常发生危险
- average reward：整体收益
- shield blocked count：policy 产生了多少危险动作

## 6. 为什么训练用 sample，评估用 argmax

训练时使用 `sample()`，是为了让 policy 有探索机会。随机初始化时，如果只用 `argmax()`，模型可能过早固定在某个错误动作上。

评估时使用 `argmax()`，是为了观察当前策略最确定、最倾向执行的动作。这时重点不是探索，而是测试训练结果。

## 7. 当前 GridWorld 和真实无人机避障的相似点与差距

相似点：

- 都有当前位置和目标位置。
- 都需要避开障碍物。
- 都需要关注成功率和碰撞率。
- 都可以把 Safety Shield 看作策略外的一层安全约束。

差距：

- GridWorld 是离散格子，真实无人机在连续空间运动。
- GridWorld 动作很简单，真实无人机动作涉及速度、姿态和动力学。
- GridWorld 观察是准确的，真实传感器会有噪声和延迟。
- GridWorld 障碍物是静态的，真实环境可能有动态障碍。

当前阶段先学清楚导航、碰撞和安全约束的基本关系，再逐步进入更真实的环境。

## 8. 根据结果图得到的实验分析

这次对比了三种情况：

- 不使用 Safety Shield。
- 使用 Safety Shield，并采用 stay 策略：危险动作被拦截后停留原地。
- 使用 Safety Shield，并采用 replace 策略：危险动作被替换成一个安全动作。

从 reward 曲线看，整体学习效果更好的反而是不使用 Safety Shield 的情况。一个原因是没有 Safety Shield 时不会产生额外的 shield penalty；而开启 Safety Shield 后，每次危险动作被拦截都会带来额外惩罚，因此平均 reward 会被压低。另一方面，没有 Safety Shield 的训练过程更直接，policy 必须自己承担危险动作带来的后果。

从 success 曲线看，使用 Safety Shield 且采用 replace 策略的效果更好。这符合预期，因为 replace 策略会把危险动作替换成安全动作，相当于环境在一定程度上帮助 agent 修正错误动作，所以更容易到达目标。

从 collision 曲线看，开启 Safety Shield 后安全性最好。危险动作会被提前拦截，因此实际 collision 可以被显著减少，甚至接近完全避免。这说明 Safety Shield 在系统安全保护层面是有效的。

但是从 shield blocked 曲线看，replace 策略下危险动作数量更多。这是一个值得注意的问题：agent 可能意识到即使自己输出了危险动作，Safety Shield 也会替它选择一个安全动作，因此 policy 本身没有足够动力学习真正安全的行为。这种情况说明 replace shield 虽然提高了系统层面的安全性和成功率，但也可能让策略对安全层产生依赖。

因此，我认为后续更合理的设计是：Safety Shield 应该更多作为独立于训练之外的保护机制，而不是长期放在训练环境中替 agent 兜底。也就是说，训练阶段应尽量让 policy 自己学习导航和避障；评估或部署阶段再加入 Safety Shield，作为最后一层安全保护。

更清晰的实验方式可以是：

```text
train without shield
evaluate without shield
evaluate with stay shield
evaluate with replace shield
```

这样可以把两个问题分开分析：

- policy 本身是否学会了安全避障。
- 加上 Safety Shield 后，整个系统是否更加安全。

## 9. 后续上传时需要带上的结果图

后续上传代码时，建议一起上传 `week03/results/plots` 里的四张图，方便结合实验结果进行说明：

- `week03/results/plots/reward_curve.png`
- `week03/results/plots/success_curve.png`
- `week03/results/plots/collision_curve.png`
- `week03/results/plots/shield_blocked_curve.png`
