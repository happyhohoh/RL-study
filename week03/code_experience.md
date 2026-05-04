# Week03 代码经验记录

这个文档用来记录阅读 Week03 代码时遇到的具体问题。目标不是背语法，而是把每次卡住的点变成以后能复习的经验。

## 1. 函数返回值类型提示：`-> tuple[int, int]`

问题：

```python
def get_next_pos(self, action: int) -> tuple[int, int]:
```

这里是不是表示输出为 tuple？

理解：

`action: int` 表示参数 `action` 预期是整数。`-> tuple[int, int]` 表示这个函数预期返回一个包含两个整数的元组。

在 GridWorld 里，位置通常写成：

```python
(row, col)
```

经验：

类型提示主要是给人、IDE 和类型检查工具看的。Python 不会因为写了 `-> tuple[int, int]` 就自动强制返回 tuple，真正返回什么仍然由函数里的 `return` 决定。

## 2. 初始化检查：`assert self.agent_pos is not None`

问题：

```python
assert self.agent_pos is not None
```

这句代码有什么作用？

理解：

`agent_pos` 在环境刚创建时是 `None`，只有调用 `reset()` 之后才会变成真正的位置，例如 `(2, 3)`。

在 `get_state()` 或 `get_next_pos()` 中，代码需要使用：

```python
self.agent_pos[0]
self.agent_pos[1]
```

如果此时 `self.agent_pos` 还是 `None`，程序就无法取下标。因此这句 `assert` 是一个提前检查：确认 agent 已经有位置之后，再继续执行后面的代码。

经验：

如果看到类似：

```python
assert xxx is not None
```

通常说明后面的代码必须依赖 `xxx` 已经初始化。对于环境类来说，常见含义是：需要先调用 `reset()`，再调用 `step()`、`get_state()` 或位置计算函数。

## 3. 避免重复定义：Week03 复用 Week02 的 `MLPPolicy`

问题：

Week03 里一开始又写了一遍 `MLPPolicy`，但 Week02 已经有 `week02/policy.py`，这样会造成重复代码。

理解：

如果两个地方的 policy 网络结构完全一样，就应该优先复用已有模块：

```python
from week02.policy import MLPPolicy
```

经验：

重复代码在学习阶段很常见，但当一个组件已经稳定下来，就可以把它当成“工具模块”继续复用。Week03 真正要变化的是环境、训练指标和 Safety Shield，而不是重新写一遍相同的 MLP。

## 4. 训练进度条：`tqdm`

问题：

训练时终端长时间没有输出，很难判断程序是在正常训练，还是已经卡住。

理解：

Python 常用的进度条库叫 `tqdm`，不是 `tmpd`。它可以把普通循环：

```python
for episode in range(num_episodes):
```

包装成带进度条的循环：

```python
for episode in tqdm(range(num_episodes)):
```

经验：

强化学习训练经常需要跑很多 episode，如果终端没有反馈，会很难判断训练状态。给训练循环加进度条是很实用的工程习惯。

## 5. 中文注释不要把代码挤到同一行

问题：

训练文件里曾经出现过这种情况：中文注释和下一句代码显示在同一行，导致关键代码被注释掉。

风险形式类似：

```python
# 训练阶段使用 sample()，让策略保留探索能力。 action_dist = ...
```

理解：

`#` 后面的内容都会被 Python 当成注释。如果真正要执行的代码不小心被放到同一行，就不会运行。

经验：

看到代码行为不符合预期时，除了看算法逻辑，也要检查缩进、换行和注释。尤其是中文注释显示乱码时，更要确认下一行代码没有被注释“吞掉”。

## 6. 训练输出要分类保存

问题：

训练后模型权重和曲线图片都生成在 `week03/results/` 里，文件混在一起，不方便查看。

理解：

模型权重和可视化图片属于两类不同结果：

```text
week03/results/models/  # 保存 .pth 权重
week03/results/plots/   # 保存 .png 曲线图
```

评估脚本默认加载：

```text
week03/results/models/policy_with_shield.pth
```

训练脚本默认把曲线保存到：

```text
week03/results/plots/
```

经验：

强化学习实验会产生很多文件，最好从一开始就把模型、图片、日志分开。后面如果继续增加 CSV、TensorBoard 日志或不同随机种子结果，也更容易扩展。

## 7. `should_block` 表示是否触发 Safety Shield

问题：

```python
_, should_block = self.shield_action(action)
```

`should_block` 是不是用来判断是否需要触发安全盾？

理解：

是的。`should_block` 是一个布尔值：

```python
True   # 这一步动作危险，需要 Safety Shield 拦截
False  # 这一步动作安全，可以正常执行
```

在当前 Week03 环境里，危险动作指的是：下一步会越界，或者下一步会撞到障碍物。

相关逻辑在 `shield_action()` 里：

```python
blocked = self.use_safety_shield and (
    self.is_out(next_pos) or self.is_obstacle(next_pos)
)
return action, blocked
```

也就是说，只有同时满足两个条件，`should_block` 才会是 `True`：

- 当前环境启用了 Safety Shield：`use_safety_shield=True`
- 下一步位置不安全：越界或撞障碍物

在 `step()` 里，如果 `should_block=True`，就会进入拦截逻辑：

```python
if should_block:
    reward = -3.0
    shield_blocked = True
```

此时 agent 不会真的撞上障碍物，也不会被记为 collision，而是停留原地并得到一个较小惩罚。

经验：

`collision` 表示危险真的发生了；`should_block` 表示危险动作在发生前被 Safety Shield 识别出来；`shield_blocked` 表示这一步确实执行了拦截记录。

## 8. `safe_actions = set()` 为什么用 set

问题：

```python
safe_actions = set()
```

为什么这里使用 `set`？

理解：

`set` 表示集合。它适合用来保存“有哪些动作是安全的”。在 Week03 的动作空间里，动作只有：

```python
0  # up
1  # down
2  # left
3  # right
```

如果我们要遍历这四个动作，并把不会越界、不会撞障碍物的动作收集起来，就可以写成：

```python
safe_actions = set()

for action in [0, 1, 2, 3]:
    next_pos = self.get_next_pos(action)
    if not self.is_out(next_pos) and not self.is_obstacle(next_pos):
        safe_actions.add(action)
```

使用 `set` 的原因：

- 不关心动作顺序，只关心“有哪些动作安全”。
- 自动去重，同一个动作不会被重复保存。
- 判断某个动作是否在集合里很直观，例如 `action in safe_actions`。

经验：

如果后续实现 Safety Shield 方案 B，也就是“危险动作不原地停留，而是替换成一个安全动作”，`safe_actions` 就会很有用。它可以表示当前状态下所有可替代的安全动作。

如果需要随机选一个安全动作，可以再转成 list：

```python
safe_action = self.rng.choice(list(safe_actions))
```

## 9. Shield 替换动作后到达终点怎么处理

问题：

如果触发 Safety Shield，有两种策略：

- 原地停留，给较大惩罚，例如 `-5`，episode 不结束。
- 不停留原地，而是从安全动作里选一个动作执行，并给额外惩罚，例如 `-2`。

疑问是：如果 Shield 替换后的安全动作刚好到达终点，应该怎么写逻辑？

理解：

比较清楚的写法是把奖励拆成两部分：

```text
最终 reward = 环境本身的结果奖励 + Shield 额外惩罚
```

例如：

- 普通走一步：`-1`
- 到达终点：`+20`
- Shield 替换动作惩罚：`-2`

如果 Shield 替换动作后只是走到普通格子：

```text
reward = -1 + (-2) = -3
```

如果 Shield 替换动作后到达终点：

```text
reward = +20 + (-2) = +18
done = True
success = True
shield_blocked = True
collision = False
```

经验：

Shield 替换动作后，仍然要像正常动作一样检查新位置是否到达目标。不要因为动作来自 Shield，就跳过 goal 检查。

推荐逻辑顺序：

```text
1. policy 给出原始动作
2. 判断原始动作是否危险
3. 如果不危险：正常执行
4. 如果危险且 Shield 选择原地停留：reward=-5，不结束
5. 如果危险且 Shield 选择替换动作：
   - 从 safe_actions 中选一个动作
   - 执行替换后的动作
   - 正常检查是否到达 goal
   - 在正常奖励基础上加 Shield 额外惩罚
```
