# RL-study

这是一个面向具身智能 / 无人机强化学习方向的工程能力提升仓库。

当前阶段不直接训练复杂 VLA、大模型或真实无人机系统，而是先从简化环境开始，逐步建立以下能力：

- PyTorch 基础网络编写能力
- 强化学习训练闭环理解能力
- 自定义环境构建能力
- state / action / reward 设计能力
- 模块拆解与数据结构设计能力
- 实验记录、复盘和 README 写作能力

## 当前阶段目标

三周内完成一个从 GridWorld 到安全导航任务的最小强化学习项目。

路线：

```text
GridWorld → REINFORCE → Navigation + Obstacle Avoidance → Safety Shield
```

当前不优先做：

- ROS 工程系统
- AirSim / Gazebo / Isaac Sim
- VLA 大模型训练
- 多无人机协同
- 真实无人机部署

## 每周复盘时间

固定复盘时间：

```text
每周六上午 10:00
```

复盘内容：

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

希望 AI 检查：
1. 代码结构是否合理
2. 概念理解是否准确
3. README 是否适合放到 GitHub
4. 是否可以整理成简历项目描述
```

## 目录结构

```text
RL-study/
  README.md
  learning_plan.md
  requirements.txt

  week01/
    mlp_policy.py
    gridworld_env.py
    test_env.py
    notes.md

  reviews/
    week01_review.md
```

## Week 01 当前任务

第一周目标：补 PyTorch + RL 代码手感。

需要完成：

1. 编写最小 PyTorch `MLPPolicy`
2. 手写极简 `GridWorld` 环境
3. 理解 `reset()`、`step(action)`、`render()` 的职责
4. 在 `notes.md` 中整理 state、action、reward 的设计
5. 周六上午 10:00 完成第一次复盘

## 本地环境建议

第一周可以直接在 Windows 上完成，不需要切换到 Ubuntu。

建议使用 conda 环境：

```bash
conda create -n embodied-uav python=3.10
conda activate embodied-uav
pip install torch numpy matplotlib
```

测试 PyTorch：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

第一周不需要 GPU，即使 `torch.cuda.is_available()` 输出 `False` 也不影响。
