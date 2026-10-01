---
title: "SUN-Reaching-for-Novelty-in-Reinforcement-Learning"
source: https://arxiv.org/pdf/2609.08642v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 03:29:10"
---

# 论文速读：SUN-Reaching-for-Novelty-in-Reinforcement-Learning

## 一句话总结
本文提出 SUN（SUccessor-to-Novelty），一种将后继价值函数（SVF）可达性信号与伪计数新颖性信号相乘融合的无奖励探索框架；通过自适应目标重选策略与成本摊销的轻量级密度估计，SUN 在含不可达状态、不可逆转移及大/无界空间的复杂环境中实现了稳定且均匀的探索覆盖。

## 研究问题与动机
- 现有 GCRL 探索方法通常只优化单一维度：密度/计数类方法（如 MEGA、Skew-Fit）追求高新颖性但易选中不可达目标；距离/成功率类方法追求可达性却忽略稀有状态，两者均无法兼顾“新颖且可达”的直觉。
- 多信号组合常依赖手工调节权重或串行应用，缺乏统一理论解释，且在目标尺度变化时需重新调参。
- 连续状态空间中传统密度估计（KDE、神经密度模型）查询开销过高，难以支撑每步高频目标重选。
- 同类型前沿方法 AdaGoal 与 DISCOVER 依赖 critic 集成方差代理新颖性/不确定性，计算昂贵且在深 RL 设定下校准不良，分别向新颖性或可达性极端倾斜。

## 核心贡献（创新点）
1. **提出乘法融合的 SUN 指标**：将 SVF 可达性值与伪计数新颖性分直接相乘，与已有方法的本质区别在于 multiplicative 结构使两者共享零点与尺度，无需人工平衡系数即可天然拒绝不可达目标。
2. **设计自适应目标选择策略**：基于 SVF 沿最优轨迹应单调非降的理论性质，以“当前状态值低于选择时刻值”为触发器动态重选目标，兼顾 episode 级承诺的稳定性和 step 级决策的灵活性。
3. **提出成本摊销的轻量伪计数估计器**：将固定半径邻居计数计算从查询阶段移至缓冲区插入阶段，实现 O(1) 查询开销，显著优于 KDE 与神经密度模型。
4. **建立严格的理论性质**：证明 SUN 等价于计数奖励的 value function、给出短视阈击中概率的下界，并形式化证明其对不可达目标的严格排斥性。
5. **构建并开源可达性敏感的新基准**：新增 ThreeRoom、FourRoomStuck、GridMaze 三个含不可达区域与不可逆转移的 Gridworld，直接暴露单一信号探索方法的缺陷。

## 方法详解
- **核心指标**：$\text{SUN}(g|s) \triangleq V^{\pi}(s, g) \cdot \nu(g)$，其中 $V^{\pi}(s, g)$ 为 SVF 估计的可达性，$\nu(g)=1/n_g$ 为伪计数新颖性。候选集 $\mathcal{C}_t$ 从回放缓冲区采样，目标选择为 $g_t = \arg\max_{g \in \mathcal{C}_t} \text{SUN}(g|s_t)$。
- **理论支撑**：
  - *Count-bonus 等价性*：SUN 等于奖励 $r_t^g = \mathbf{1}\{s_t=g\}/n_g$ 的目标条件 MDP 的 value function，是单一原则性目标而非启发式叠加。
  - *短视阈击中下界*：$\Pr_\pi[\tau_g \le n] \ge V^\pi(s,g) - \gamma^{n+1}$，SVF 值越高，在 $\mathcal{O}(\log(1/V)/(1-\gamma))$ 步内击中概率越大。
  - *不可达目标抑制*：若 $g$ 对策略类 $\Pi$ 均不可达，则 $V^\pi(s,g)=0$，SUN 恒为 0，与 novelty 值无关。
- **自适应目标选择**：记录选择时刻 $t_{\text{sel}}$ 的 $V^\theta(s_{t_{\text{sel}}}, g_t)$；每步比较 $V^\theta(s_t, g_t)$，若下降则判定目标不可达或 SVF 评估失准，丢弃当前目标并重新采样候选集；目标达成或 episode 开始时亦触发重选。
- **轻量伪计数**：缓冲区每条记录存储特征空间中半径 $\rho$ 内的邻居数量。插入新样本时计算自身计数（邻居数+1）并递增所有邻居计数；查询时直接读取 $O(1)$。采用滑动中位数截断方差进行特征标准化，防止低方差维度放大噪声。
- **与基线公式对比**：SUN 为 $V/n_g$；DISCOVER 为 $\mu(s,g)+\beta\sigma(s,g)$（mean 可达性 + 方差 novelty）；AdaGoal 为 $\sigma(s,g)$（仅集成不确定性）。

## 实验与结果
- **环境设置**：Gridworlds（ThreeRoom, FourRoomStuck, GridMaze）、Classic Control（MountainCar, CartPole, LunarLander, Pendulum, Acrobot）、GCRL Control（PointMaze-S/H, AntMaze-S/H, ArmPush-H）。评测指标为覆盖率（Coverage）与归一化 Shannon 熵（Entropy）。
- **主要结果**：SUN 在所有环境的覆盖率与熵上均一致优于 Random、AdaGoal 与 DISCOVER。Gridworlds 中各方法覆盖率均趋近 1，但 SUN 熵显著领先；连续控制任务中 SUN 曲线上升更快且更早饱和。
- **关键数值提升**：综合 AUC（排除 LunarLander(Full)），Adaptive Multiplicative SUN 较 AdaGoal 覆盖率 +15.1%、熵 +7.6%；较 DISCOVER 覆盖率 +21.0%、熵 +9.2%；较 Random 覆盖率 +67.1%、熵 +26.5%。
- **消融结论**：
  - Reachability-only 几乎不探索；Novelty-only 在存在不可达目标时崩溃；Additive SUN 会随计数增加“饱和”退化为纯可达性并导致聚类；Multiplicative SUN 无此失败模式。
  - Episodic selection 在 FourRoomStuck 等中途目标不可达时彻底失败；Per-step selection 重选过于频繁导致训练不稳定；Adaptive selection 重选率随训练下降至零，性能最优。
- **运行效率**：SUN 虽需更频繁的目标选择，但因伪计数成本前置与单次 critic 计算，在 Gridworlds/Classic Control 上平均比 AdaGoal 快 3.54×、比 DISCOVER 快 3.43×；GCRL 环境中三者耗时相当。

## 相关工作脉络
1. **Reward-free / Max-Entropy Exploration**（Hazan et al., 2019; Jain et al., 2023）：目标最大化状态访问熵，但依赖 Frank-Wolfe 等难扩展至深 RL；SUN 采用 GCRL 范式，通过 goal-selection 隐式诱导均匀覆盖。
2. **Density-based Novelty**（MEGA, Skew-Fit, HFG）：仅关注新颖性，易选中不可达状态；SUN 通过 SVF 乘法项强制过滤不可达目标。
3. **Reachability-based / Proto-Goals**（Schaul et al., 2015; Bagaria et al., 2023）：Proto-Goals 串行先选新颖再选可达，无法互相修正；SUN 联合乘法且支持动态重选。
4. **AdaGoal & DISCOVER**：同用 SVF，但依赖 critic 集成方差代理新颖性，计算重且深度强化学习中校准失效（AdaGoal 偏 novelty，DISCOVER 偏 reachability）；SUN 以轻量伪计数替代，平衡更稳定。
5. **Directed Exploration / NGU**（Badia et al., 2020）：NGU episodic novelty 也用最近邻密度估计，但仅统计 episode 内 novelty 且每次查询重算核和；SUN 统计 lifetime novelty 并将成本移至插入。
6. **Successor Representation**（Dayan, 1993; Eysenbach et al., 2022）：SUN 继承 SVF 作为可达性代理，但创新性地将其与计数 novelty 乘法融合并证明其 count-bonus 等价性。

## 局限性与未来方向
- **高维/图像观测下的计数失效**：在图像等高维空间几乎所有状态
