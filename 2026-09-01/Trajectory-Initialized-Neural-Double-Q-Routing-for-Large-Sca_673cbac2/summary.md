---
title: "Trajectory-Initialized-Neural-Double-Q-Routing-for-Large-Sca"
source: https://arxiv.org/pdf/2608.30512v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 06:40:25"
---

# 论文速读：Trajectory-Initialized Neural Double Q-Routing for Large-Scale Overhead Hoist Transport Systems

## 一句话总结
针对半导体厂站OHT系统中静态最短路径无法感知动态拥堵、传统表格Q-routing经验孤岛且冷启动敏感的问题，提出一种基于共享状态-动作价值网络的轨迹初始化Double Q路由框架；该方法在离散事件仿真中较表格Double Q-routing平均降低0.8%–8.8%的任务完成时间，并在中高车队规模场景下取得最优性能。

## 研究问题与动机
- 核心问题：大规模OHT车队共享受限导向轨，车辆行驶时间受安全间距、交叉口独占访问、下游阻塞及装卸站台争用等时变因素耦合，需在每个分歧节点实时选择下一跳以最小化完成时间。
- 静态最短路径（Dijkstra）采用固定边权重，无法响应实时交通状况与车队耦合延误，容易将车辆导入已排队的拥堵路段。
- 表格Q-routing虽可在线自适应，但每个$(d,i,j)$元组独立存储价值，稀疏访问上下文间无法共享经验，且冷启动期对不准确价值估计极度敏感。
- 直接以神经网络替换表格仅解决表示容量问题，仍需解决在线学习初期的实用性、TD目标稳定性以及短寿命局部队列的快速响应问题。

## 核心贡献（创新点）
1. 提出紧凑共享状态-动作价值网络替代目的地索引表：参数量从约259万降至5825，参数量与$|D|、|V|、|E|$无关。
2. 设计混合轨迹离线预热与在线Double-Q微调流程：利用Dijkstra/Q/Double-Q多策略轨迹做return-to-go回归初始化，解决纯随机初始化冷启动慢、初期路由质量不可靠的根本问题。
3. 引入局部拥堵残差修正并与递归目标解耦：将近期边缘级TD残差以非bootstrap形式叠加至行为评分，避免局部观测噪声通过目标网络递归污染全局价值估计。
4. 提出领域知识引导的事件分层结构化重放：按拥堵严重程度划分五类事件并按固定配额采样，突破均匀重放下罕见阻塞样本被稀释以及PER纯依赖TD误差排序的局限。
5. 提供设置依赖性的系统评估协议：在9组匹配场景下清晰分离架构收益与初始化收益，表明方法在150/200辆场景优于Dijkstra，而在100辆低负载场景仍由Dijkstra主导。

## 方法详解
- **共享价值网络**：状态特征$s$（10维，含节点/目的地归一化索引、最短路径距离$h(i,d)$、进出度、任务阶段编码、近期等待时间）与动作特征$\phi_a(s,j)$（14维，含候选边指标、边长、剩余距离、进度、边占用、节点队列、六项局部压力摘要）拼接为24维向量，经两隐藏层MLP（每层64单元，ReLU）输出标量$Q_\theta(s,a)$，动作值越大越优先。
- **离线预热**：轨迹截断处重设累积，计算return-to-go目标$\hat{G}_t = \sum_{k=t}^{T_\ell} \gamma^{k-t} r_k$，以MSE损失$\mathcal{L}_{\text{prior}}(\theta) = \mathbb{E}[(Q_\theta(s,a) - \hat{G}(s,a))^2]$预训练40 epoch，同步初始化在线网络$Q_\theta$与目标网络$Q_{\bar{\theta}}$。
- **在线Double-Q更新**：折扣因子$\gamma=0.99$，采用$ a^\star = \arg\max_{a'} Q_\theta(s',a')$，目标值$y = r + \gamma(1-z)Q_{\bar{\theta}}(s',a^\star)$（$z=1$时为终态），以Huber损失$\mathcal{L}_{\text{td}}$更新，目标网络Polyak平滑$\bar{\theta} \leftarrow (1-\tau)\bar{\theta} + \tau\theta$（$\tau=0.005$），Adam lr=1e-3，batch=64，梯度范数裁剪至5，每4次观测更新一次。
- **局部拥堵修正**：执行评分$U_t(s,a) = Q_\theta(s,a) + \lambda_t \kappa(s,a) R_{\text{local}}(s,a) - \beta_t(n(s,a)+1)^{-1/2}$，其中$R_{\text{local}}$为指数平均 clipped TD残差（$\alpha=0.05, c=5$），$\lambda_t$与$\beta_t$为线性调度至5000步；修正项仅作用于行为策略评分，不进入bootstrap目标。
- **事件分层结构化重放**：每条转移标记为normal/congested/blocked/near-deadlock/LU-bottleneck，每批次固定抽取40%非正常样本（比例16/12/8/4%）并施加1.0–3.0倍损失权重；缺失类别由其他事件或通用重放填补，非重要性采样修正。

## 实验与结果
- **环境**：单一晶圆厂OHT离散事件模拟器（$|V|=3684$, $|E|=4257$, $|D|=608$）。
- **基线**：Dijkstra、Tabular Q-routing、Tabular Double Q-routing、Neural Single-Q（内部消融）。
- **主要结果**：在100/150/200辆OHT与1.0/1.5/2.0 task/s到达率的9组匹配场景下，QNeuralDouble较Tabular Double Q-routing在所有场景中降低CT Mean 0.8%–8.8%；完成任务数在8/9场景中波动≤1%；CT P95在8/9场景中下降；在6个150/200辆设置中取得全场最低CT Mean，Dijkstra在3个100辆设置中保持最优。
- **启动性能**：离线预热在150/200辆启动场景中使前200s完成任务数提升22.2%–23.1%，CT P95降低7.2%–15.0%，Early CT modestly改善1.3%–2.1%。
- **消融**：移除Double-Q（替换为Single-Q）导致CT Mean恶化+5.78s（最大单项代价），移除局部修正+4.87s，移除结构化重放+1.61s；LC与SR联删产生+7.32s，效应非完全叠加；完整框架较表格Double Q基线差距达17.22s（含表示与初始化整体差异）。

## 相关工作脉络
- **拥堵感知最短路径**：Bartlett等与Gupta等将测量/预测拥堵压缩为边代价后运行Dijkstra，本文跳过边权重预处理，直接在分歧点用共享网络评分候选边。
- **图神经网络交通预测**：Ahn等与Choi等利用GCN/GRU预测未来通行时间或路由级影响后再交由路径规划器，本文学习的是直接输出next-hop的本地决策函数而非全局流量预测。
- **表格Q-routing在OHT的应用**：Hwang & Jang等保留独立$(d,i,j)$表进行在线Q-learning，本文共享参数打破元组经验孤岛，使高频上下文经验可直接惠及低频上下文。
- **地图编码深度Q规划**：Ao等将轨道图栅格化输入Double-DQN，本文采用固定维度状态-动作特征拼接，参数规模与图拓扑解耦，且不依赖显式拓扑卷积。
- **离线RL与演示学习**：Hester等(DQfD)与Kumar等(CQL)在固定数据集上优化策略并施加保守正则，本文仅将多策略轨迹用于价值网络有监督预热，后续完全由在线TD驱动，不引入 imitation loss。
- **
