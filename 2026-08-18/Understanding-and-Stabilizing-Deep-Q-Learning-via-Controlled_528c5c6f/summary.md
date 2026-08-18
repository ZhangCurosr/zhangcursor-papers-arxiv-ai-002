---
title: "Understanding-and-Stabilizing-Deep-Q-Learning-via-Controlled"
source: https://arxiv.org/pdf/2608.16182v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:13:45"
---

# 论文速读：Understanding-and-Stabilizing-Deep-Q-Learning-via-Controlled

## 一句话总结
本文从算子、估计器、参数动态三个互补视角系统剖析了深度 Q 学习训练不稳定的内在交互机制，揭示了奖励驱动的“自强化陷阱”与参数尖峰现象，并据此提出融合跨模型受控引导、集成分位数回归与层级 Spike Ratio 监控重置的稳定化算法，在 Atari-100K 和 Procgen 上取得了领先稳健指标与更强训练鲁棒性。

## 研究问题与动机
- 深度 Q 学习（DQL）训练 notoriously unstable，现有文献多孤立归因于 Bellman 算子的过估计偏差或表征学习退化，缺乏对递归价值估计中多源不稳定因素耦合作用的统一理解。
- 经典 Double Q-learning / Double DQN 仅解耦单一模型内的动作选择与价值评估，未能捕捉奖励触发下动作条件表征泛化所导致的递归放大反馈环。
- 高 replay ratio 下的激进数据复用会加剧参数分布失衡与网络可塑性丧失（primacy bias / dormant neuron），现有干预手段缺乏细粒度、层级的可观测诊断指标。
- 即使回归估计无偏，估计噪声仍会通过小 action gap 扰动贪心决策，进而改变交互数据分布，在控制回路层面引发隐性不稳定。

## 核心贡献（创新点）
- **统一的不稳定性三分解框架**：将 DQL 训练失稳归结为算子级 Bootstrapping 结构偏差、估计级回归噪声对贪心决策的敏感性、以及参数动态失衡三类相互作用机制，突破单一归因范式。
- **发现并形式化“奖励触发自强化陷阱”**：阐明正奖励通过动作条件表征泛化同时推高 $Q(s,a)$ 与 $Q(s',a)$，使 max 算子反复选中原动作，形成递归放大回路（理论上有界，极限为 $r/(1-\gamma)$）。
- **提出三级稳定化原则与算法实例化**：融合跨模型选择-评估解耦、奖励转移强制动作掩码、轨迹级数值有界化、集成分位数回归与层级 Spike Ratio 监控重置，形成完整可落地的稳定训练框架。
- **引入 Spike Ratio 作为训练诊断统计量**：定义 $\mathrm{SpikeRatio}_\ell = \|\theta_\ell\|_\infty / \mathrm{Quantile}_{0.99}(|\theta_\ell|)$，首次将网络可塑性衰退量化为可监控的层级指标，并与 replay ratio 建立显式关联。

## 方法详解
- **受控引导（Controlled Bootstrapping）**：
  - *跨模型解耦*：维护 $m$ 个价值网络集成 $\{\hat{Q}_i\}$，对成员 $i$ 使用另一成员 $j$ 选择 bootstrap 动作，自身评估价值：$y_i = r + \gamma \hat{Q}_i(s', \arg\max_{a'} \hat{Q}_j(s', a'))$，切断同一模型内选择-评估噪声的相关性。
  - *奖励转移动作解耦*：对 $r>0$ 的样本强制将原始动作 $a$ 对应的目标 Q 值置为 $-\infty$，使 $\arg\max$ 必然选择其他动作，直接阻断自强化回路的递归路径。
  - *有界 Bellman 更新*：基于当前贪心轨迹的折扣回报 $G_t$ 定义轨迹级上界 $B(\tau)=\max_t G_t$，约束价值估计的数值膨胀规模。
- **方差感知价值估计**：采用集成分位数回归（Ensemble Quantile Regression），每成员输出 $K$ 个分位数原子，标量 $Q_\theta(s,a)=\frac{1}{K}\sum_i z_\theta^{(i)}(s,a)$；动作选择基于集成均值 $\bar{Q}(s,a)$，双重聚合显著降低贪心决策的估计方差。
- **参数动态调控**：周期性计算各层 Spike Ratio，若某层超过阈值则重置该层参数（同步清空 Adam 动量/方差状态并将 target network 软同步），恢复参数分布平衡与长期可塑性。
- **算法流程（Algorithm 1）**：$\epsilon$-greedy 探索使用 $\
