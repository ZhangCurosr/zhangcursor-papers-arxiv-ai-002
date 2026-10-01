---
title: "The-Surprising-Effectiveness-of-Approximate-Value-Iteration"
source: https://arxiv.org/pdf/2609.09094v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 03:33:11"
field: "多智能体强化学习/博弈论"
keywords: ["Approximate Value Iteration", "Self-Play", "Monte Carlo Tree Search", "Zero-Sum Games", "Value Learning", "Reinforcement Learning", "AlphaZero"]
innovations: ["首次在多个非平凡棋类游戏中用精确 oracle 系统比较 AVI 与 AlphaZero，证明 AVI 可学到更精确价值函数", "提出 negamax 交替结构中误差符号抵消的稳定性解释假设", "通过 cross-inference 实验证明简单价值学习可为复杂搜索提供更强指导"]
benchmarks: ["Connect Four (strong oracle)", "Hex(7x7) (weak oracle)", "F-Games (synthetic)", "Othello (MiniZero baseline)", "Go(9x9) (MiniZero baseline)"]
---

# 论文速读：The-Surprising-Effectiveness-of-Approximate-Value-Iteration

## 一句话总结
本文系统性地验证了在非平凡回合制棋类游戏中，**近似价值迭代（AVI）** 可以稳定训练并学到比 AlphaZero **更精确的价值函数**，其单步前瞻贪婪策略在计算成本大幅降低的情况下仍能与 MCTS 策略竞争。

## 研究问题与动机
1. **核心问题**：AlphaZero 依赖 MCTS + 神经网络的价值判断取得了巨大成功，但 MCTS 带来显著的计算开销和工程复杂度——能否放弃搜索，直接学习价值函数？
2. **已有方法不足**：MCTS-based 方法（AlphaZero、MuZero）需要大量神经网络调用进行深度搜索，训练和推理成本高昂；且 MCTS 的主导地位可能掩盖了简单价值迭代方法的潜力。
3. **预期失败风险**：AVI 结合了函数近似、bootstrap 和 off-policy 学习，属于强化学习"致命三角（deadly triad）"的组合，理论上可能导致数值不稳定或发散。
4. **评估缺失**：已有评测依赖 Elo 评分或胜负匹配，缺乏对价值函数绝对精度的度量；本文首次在多个游戏中引入 ground-truth oracle 进行系统比较。

## 核心贡献（创新点）
1. **首个 oracle-based 系统比较**：首次在 Connect Four、Hex(7x7) 和 F-Games 上，用精确最优值度量神经网络 AVI 与 MCTS-based AlphaZero 的价值误差、策略后悔值和决策质量，而非仅依赖相对胜率。
2. **发现 AVI 的价值学习能力超越 AlphaZero**：AVI 学到的价值函数在两个求解游戏中 MAE 显著低于 AlphaZero，且差距随训练增长；其单步贪婪策略的后悔值和 oracle 错误率在计算成本低一个数量级时仍与之匹敌。
3. **证明简单搜索仍可显著提升 MCTS 表现**：将 AVI 的价值函数嵌入 MiniZero 的 MCTS（cross-inference）后，在相同仿真预算下显著降低了错误率，说明 AVI 的价值为更深搜索提供了更强指导。
4. **揭示交替博弈中 bootstrapping 的稳定性机制**：提出 negamax 备份中误差符号交替抵消的解释假设，为"致命三角"在不发散提供了一个直觉性理论依据。
5. **扩展到更大规模游戏**：在 Othello 和 Go(9x9) 上验证 AVI 训练稳定，且学到的价值函数可提升独立训练的 MCTS 基线，表明方法不限于有精确 oracle 的游戏。

## 方法详解

**AVI 框架（Algorithm 1）**：
- **网络结构**：仅含单一价值网络 $V_\theta(s)$，无策略头（AlphaZero 有独立的策略头 $\pi_\theta$）。
- **数据收集**：使用 $\epsilon$-greedy 自我对弈，策略由当前价值网络的单步 negamax 贪婪产生：
  $$\mathcal{G}(V_{\bar{\theta}})(s) \in \arg\max_{a} [R(s,a) - \gamma V_{\bar{\theta}}(f(s,a))]$$
  每步将状态-目标对 $(s, y_{\bar{\theta}}(s))$ 存入循环回放缓冲区 $\mathcal{D}$，其中目标由单步 negamax 备份计算：
  $$y_{\bar{\theta}}(s) = \max_{a \in \mathcal{A}(s)} [R(s,a) - \gamma V_{\bar{\theta}}(f(s,a))]$$
- **训练**：固定参数 $\bar{\theta}$ 收集数据，然后对回放缓冲区中的 mini-batch 最小化 MSE 损失：
  $$\mathcal{L}_{\text{AVI}}(\theta) = \mathbb{E}_{(s,y)\sim\mathcal{D}}[(V_\theta(s) - y)^2]$$
- **推理**：采用单步前瞻贪婪 $\mathcal{G}(V_\theta)$，仅需评估每个合法动作一次。

**与 AlphaZero 的关键差异**：
| 维度 | AVI | AlphaZero |
|------|-----|-----------|
| 目标生成 | 单步 negamax 备份 | MCTS 模拟 + 终局结果回溯 |
| 网络 | 仅价值头 | 价值头 + 策略头 |
| 搜索 | 无树搜索 | MCTS（S 次仿真） |
| 探索 | $\epsilon$-greedy | PUCT + Dirichlet 噪声 |
| 推理计算 | 最多 7 次前向（Connect Four） | 数百次前向（S=512） |

## 实验与结果

**数据集与评估基准**：
- **Connect Four**（约 $10^{12}$ 状态，强 oracle：alpha-beta 求解器，含最短获胜步数）
- **Hex(7x7)**（约 $10^{22}$ 状态，弱 oracle：MoHex 求解器，仅返回胜负）
- **F-Games**（合成游戏树，三种结构：深窄 h=20 b=2、平衡 h=10 b=5、浅宽 h=5 b=20，各生成 10 个）
- **Othello / Go(9x9)**：无 oracle，以 MiniZero（200 次 MCTS 仿真）为参考基线

**核心结果**：
- **Connect Four**：AVI 的 MAE 显著低于 AlphaZero（即使 AZ 使用 S=512），差距随训练增大；单步贪婪 AVI 的策略后悔值和 oracle 错误率与最强 AZ 相当。直接对战中 AVI 得分 $-0.09 \pm 0.04$，与 AZ(S=512) 基本持平。
- **Hex(7x7)**：AVI 在 MAE、后悔值和 oracle 错误率三项指标上**全面优于**所有 AZ 配置；对战得分 $+0.38 \pm 0.09$，优势明显。
- **F-Games**：AVI 在所有三种树结构上 MAE 均更低；策略质量在深窄树上差距较小（AZ 可用搜索补偿不精确值）。
- **Othello & Go(9x9)**：AVI 训练稳定，greedy 策略逐步提升但仍弱于 MiniZero；但 cross-inference（MiniZero 策略 + AVI 价值）在两种游戏中均**显著提升** MiniZero 的胜率。
- **计算效率**：Connect Four 中 AVI 推理只需最多 7 次前向 vs AZ 的 512 次；Hex 中 49 vs 512。

## 相关工作脉络

1. **AlphaZero（Silver et al., 2018）**：确立 MCTS + DNN 范式；本文与之对照，证明放弃搜索仍可学到更优价值。
2. **Veness et al. (2009) — Bootstrapping from Game Tree Search**：用线性函数近似在 Chess 中获得大师级性能；本文将其思想延伸至神经网络，并探究最小化单步备份的极限。
3. **Perolat et al. (2015) — ADP for Markov Games**：将近似动态规划推广至两玩家零和博弈；本文为该方法提供了大规模实证验证。
4. **Cohen-Solal & Cazenave (2023/2026) — Minimax Strikes Back / Descent**：使用 best-first minimax 构建更强的训练目标；本文采用更简单的单步 negamax 作为对比极端。
5. **Tian et al. (2022) — DAVI**： Doubly-Asynchronous Value Iteration，用采样子集替代穷举最大化；本文讨论其作为未来改进方向的可能性。
6. **Wu et al. (2024) — MiniZero**：作为 Othello/Go 外部基线；本文用其验证 AVI 价值对现有搜索框架的提升能力。

## 局限性与未来方向

**局限性**：
1. 强结论依赖精确 oracle，仅限 Connect Four、Hex(7x7)、F-Games；Othello/Go(9x9) 仅以 MiniZero 为基准，无法衡量与最优策略的距离。
2. 未测试 Go(19x19) 或 Chess 等更大规模游戏。
3. 计算代理指标（forward-equivalent compute）未涵盖所有 wall-clock 开销（如 MCTS 树构建）。
4. $\epsilon$-greedy 探索在宽动作空间（Othello/Go）中效率有限，均匀分散预算导致 greedy 策略弱于 MCTS。

**未来方向**：
1. 改进探索与状态采样（如 DAVI 的采样最大化），结合 learned policy prior 引导动作选择。
2. 构建 hybrid agent：共享表征中学习 AVI 式价值 + policy prior，再配合选择性搜索（如 MCTS）。
3. 深入理论分析 negamax 交替结构中误差抵消机制（bias-cancellation hypothesis）。

## 研究启发与可借鉴点

1. **价值学习与搜索决策可分离设计**：训练时用简单高效的方式学价值，推理时再用复杂搜索转化决策——这一分离思路值得在其他序列决策任务中尝试。
2. **Oracle-based 评估的启示**：在可求解子问题或合成环境中，直接度量价值函数精度和策略后悔值比仅看相对胜率更能揭示方法本质优劣，建议在新方法评估中引入绝对指标。
3. **替代"致命三角"的策略**：在 alternating zero-sum 设置中，negamax 符号翻转可能天然缓解 overestimation bias；这一洞察可推广至其他多玩家 RL 场景的设计。
4. **低成本基线的价值**：以极简实现（无策略头、无树搜索）建立强基线，有助于更公平地量化复杂方法（如 MCTS）的真实边际收益。
5. **Cross-inference 诊断实验设计**：固定搜索策略、仅替换价值函数，可精确隔离价值学习的贡献，这一实验范式适用于任何搜索+学习结合的框架评估。

## 关键术语表

**Approximate Value Iteration（AVI）**：用神经网络替代表格，在采样状态上用单步贝尔曼备份更新价值函数的动态规划方法。

**Negamax**：零和博弈中统一的最大化/最小化形式，所有值从当前行动者视角表达， successor value 取反号。

**Policy Regret（策略后悔值）**：衡量贪婪策略与最优动作之间的价值损失，比纯值误差更能反映决策质量。

**Cross-inference**：保留 MCTS 策略先验不变，仅将价值评估节点替换为 AVI 学习到的价值函数，用于隔离价值函数的贡献。

**Deadly Triad（致命三角）**：函数近似 + Bootstrap + Off-policy 学习的组合，传统上被认为容易导致 RL 训练不稳定或发散。

**Strong/Weak Oracle**：强 oracle 可返回最短获胜步数（区分同结果动作），弱 oracle 仅返回理论胜负结果。

**Forward-equivalent Compute Proxy**：以 $N_{\text{forward}} + 3N_{\text{backward}}$ 近似神经网络计算量，用于跨方法公平比较训练/推理成本。

## 可复现要素

- **代码**：论文声明 "Our code is publicly available"（附注2），基于 JAX + PGX + mctx 实现。
- **数据集**：Connect Four / Hex(7x7) 状态集由论文附录 D/E 描述的方法生成；F-Games 由 Appendix F 的协议程序化生成；Othello/Go(9x9) 使用 PGX 环境。
- **超参数**：详见附录 C Table 3-6；关键值包括 $\epsilon=0.3$（Connect Four/Hex），$\epsilon=0.15$（Othello/Go），学习率 $3\times10^{-4}$，batch size 256，replay buffer 1M。
- **架构**：Connect Four/Hex 使用 4/2 层残差 MLP body + 256 隐藏单元；Othello/Go 使用与 MiniZero 相同的三块残差 backbone。
