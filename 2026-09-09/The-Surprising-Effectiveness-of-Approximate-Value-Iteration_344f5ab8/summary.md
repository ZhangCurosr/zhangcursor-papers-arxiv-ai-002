---
title: "The-Surprising-Effectiveness-of-Approximate-Value-Iteration"
source: https://arxiv.org/pdf/2609.09094v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 03:32:53"
field: "博弈强化学习"
keywords: ["Approximate Value Iteration", "self-play", "Monte Carlo Tree Search", "game-playing AI", "value function learning", "negamax"]
innovations: ["在可解博弈中系统证明AVI价值函数比AlphaZero更准确且训练稳定", "一步贪心策略在极低推理预算下匹敌MCTS策略质量", "AVI价值函数嵌入MCTS可提升搜索性能"]
benchmarks: ["Connect Four", "Hex(7x7)", "F-Games", "Othello", "Go(9x9)"]
---

# 论文速读：The-Surprising-Effectiveness-of-Approximate-Value-Iteration

## 一句话总结
论文系统研究了简单近似值迭代（AVI）在自对弈游戏中的效果，发现在 Connect Four、Hex(7x7) 等中等规模博弈中，AVI 学习到的价值函数比 AlphaZero 更准确，且一步贪心策略的计算开销远低于 MCTS，同时在 Othello 和 Go(9x9) 上也训练稳定并能增强 MCTS 搜索性能。

## 研究问题与动机
- AlphaZero 将 MCTS + 深度神经网络确立为博弈 AI 的主流范式，但 MCTS 在训练和推理阶段均需大量网络调用，计算开销显著
- 直接价值学习方法结合函数逼近、自举和离策略学习（"死亡三角"），被认为不稳定容易发散，鲜有系统研究其现代实现潜力
- 现有评估多依赖 Elo 或相对对局强度，缺乏对价值函数绝对质量的度量，本文利用精确 Oracle（已求解游戏）提供客观评估基准
- 核心问题：在不使用树搜索的情况下，仅通过直接价值迭代能否学到高质量价值函数，并在决策层面具有竞争力？

## 核心贡献（创新点）
1. **首个在可解游戏中系统性对比 AVI 与 AlphaZero 的 Oracle 基准实验**：利用 Connect Four 强 Oracle、Hex(7x7) 弱 Oracle 及合成 F-Games，同时测量价值误差、策略遗憾和对战错误率，弥补纯对局评估的不足。
2. **揭示 AVI 价值学习的高精度与训练稳定性**：AVI 在两种已解博弈中学习到的价值函数 MAE 显著低于 AlphaZero，且 20 个随机种子下均无发散，打破"死亡三角必致不稳定"的预期。
3. **证明一步贪心策略在极低推理预算下仍具竞争力**：AVI 每步仅需最多 7 次（Connect Four）或 49 次（Hex(7x7)）前向传播，即可匹敌 AlphaZero 512 次模拟的决策质量；在更大分支因子游戏（Othello、Go(9x9)）上其价值函数嵌入 MCTS 仍可提升 MiniZero 对局胜率。

## 方法详解
- **Negamax 值函数定义**：最优值满足 $V^*(s) = \max_{a} [R(s,a) - \gamma V^*(f(s,a))]$，所有值和奖励统一从当前行动方视角表达，消除最大化/最小化对称性。
- **AVI 单步备份与目标构建**：固定参数 $\bar{\theta}$，对每个采样状态计算一步 negamax 目标 $y_{\bar{\theta}}(s) = \max_a [R(s,a) - \gamma V_{\bar{\theta}}(f(s,a))]$。
- **训练损失**：采用 MSE 损失 $\mathcal{L}_{\mathrm{AVI}}(\theta) = \mathbb{E}_{(s,y)\sim\mathcal{D}}[(V_\theta(s) - y)^2]$，在循环回放缓冲区上优化。
- **数据收集**：通过 $\epsilon$-greedy 自对弈（$\epsilon=0.3$）沿当前价值函数的单步贪心策略采集状态-目标对，存入滑动窗口缓冲区以混合不同学习阶段数据。
- **与 AlphaZero 的关键差异**：AVI 仅用一步 negamax 备份作为价值目标，无需 MCTS 模拟生成策略分布，也不显式学习策略网络；AlphaZero 使用完整自对弈终局结果作为价值目标，并从 MCTS 访问计数中提取策略目标联合训练。
- **交叉推断（Cross-inference）**：固定 MCTS 策略先验，替换叶节点价值评估为 AVI 价值函数，用于隔离价值函数的搜索指导增益。

## 实验与结果
- **数据集与评估基准**：
  - Connect Four（约 $10^{12}$ 状态，强 Oracle，2×1500 评估态，406 开局）
  - Hex(7x7)（约 $10^{22}$ 状态，弱 Oracle，445 评估态，128 开局）
  - F-Games（三种树结构各 10 个实例，4096 评估态/游戏）
  - Othello 与 Go(9x9)（无 Oracle，对比 MiniZero，分别 54/81 开局）
- **主要数值结果**：
  - **价值误差**：AVI 在 Connect Four 和 Hex(7x7) 上的 MAE 始终低于所有 AlphaZero MCTS 预算配置（S=32~512），且差距随训练扩大。
  - **策略遗憾与错误率**：Connect Four 上贪婪 AVI 遗憾率和 Oracle 对战错误率接近最强 AZ（S=512）；Hex(7x7) 上显著优于所有 AZ 配置。
  - **对局成绩（表2）**：AVI vs AZ(S=512) 在 Connect Four 得分为 $-0.09 \pm 0.04$，在 Hex(7x7) 得分为 $0.38 \pm 0.09$（正值 favor AVI）。
  - **计算成本**：AZ 推理需 512 次前向传播，AVI 仅需 7（Connect Four）/ 49（Hex(7x7)）次；训练阶段前向等效计算量差距更显著。
  - **跨推断（图2）**：用 AVI 价值替换 AZ 价值后，MCTS 错误率在全部预算下下降。
  - **Othello/Go(9x9)（图3）**：AVI 训练稳定，贪婪策略落后 MiniZero；但交叉推断下 MiniZero+AVI 价值在两种游戏上均显著提升对局得分。
  - **F-Games**：三种树结构（深窄/平衡/浅宽）下 AVI 均获得更低价值误差；深窄树上策略遗憾差距较小（MCTS 可补偿不准确叶值）。

## 相关工作脉络
- **AlphaZero (Silver et al., 2018)**：确立 MCTS + 神经网络自对弈范式，本文与其在相同网络骨干和训练协议下公平对比，证明 MCTS 的核心优势在于搜索而非价值学习本身。
- **Veness et al. (2009)**：使用线性函数逼近从 minimax 搜索中自举学习价值，达大师级象棋水平；本文将其思想扩展到深度神经网络，并探索无需搜索的价值学习极限。
- **Perolat et al. (2015) / Scherrer et al. (2012)**：从近似动态规划角度分析双玩家零和马尔可夫博弈的值迭代与策略迭代；本文为极简 one-step negamax 版本提供了系统的实证支撑。
- **Cohen-Solal & Cazenave (2023) / Cohen-Solal (2026, Descent)**：重新审视 minimax 值学习和基于 best-first 搜索的训练目标构造；本文论证最简单的单步备份在深度网络下同样有效，无需复杂搜索构造 target。
- **MuZero (Schrittwieser et al., 2020) / Gumbel AlphaZero (Danihelka et al., 2022)**：在 AZ 架构上改进模型学习或策略优化效率；本文指出这些工作的搜索核心可能被高估，简单值学习值得作为独立方案或搜索组件重新评估。
- **DAVI (Tian et al., 2022)**：用动作子集采样替代穷举最大化，缓解大分支因子下一步备份的计算负担；本文在局限性部分引用其为未来改进 AVI 在大动作空间游戏中的可行方向。

## 局限性与未来方向
- 核心结论基于可精确求解的游戏（Connect Four、Hex(7x7)、F-Games），Othello 和 Go(9x9) 仅以 MiniZero 为基准，无法评估接近最优的程度
- 未测试 Go(19x19) 或国际象棋等顶级规模博弈
- 前向等效计算代理未涵盖构建 MCTS 树等墙钟时间的额外开销
- $\epsilon$-greedy 自对弈在稀疏策略游戏中可能错过关键状态， exploration 机制有待改进
- 未来方向：引入 DAVI 类动作采样、融合 AVI 价值与策略先验的混合网络、探索更有针对性的状态采样策略

## 研究启发与可借鉴点
1. **Oracle-based 评估范式**：在可解博弈中使用精确价值目标和策略遗憾作为绝对度量，比单纯 Elo/胜率更能揭示算法内在差异，值得在可求解任务中推广。
2. **价值学习与搜索解耦的设计思路**：AVI 的训练完全不依赖搜索，但其学到的价值函数仍能提升 MCTS 性能，这为"简单价值学习 + 高级搜索"的模块化架构提供了实证依据。
3. **negamax 交替符号的误差抵消假设**：论文提出 alternating games 中 overestimate 在一层后变为 underestimate，可能缓解 "deadly triad" 的不稳定性，这一假设为理解值迭代在博弈中的稳定性提供了新的理论视角。
4. **极简实现的可复现性**：AVI 仅用 MLP 骨干 + MSE + $\epsilon$-greedy，代码简洁，极适合作为研究基线或教学示例，便于快速验证新想法。

## 关键术语表
- **Approximate Value Iteration (AVI)**：用神经网络近似值函数，通过单步 negamax 备份生成目标值并迭代优化的值学习方法。
- **Negamax**：博弈论中统一最大/最小操作的递归形式，所有值从当前行动方视角表达，公式为 $V(s) = \max_a [R(s,a) - \gamma V(f(s,a))]$。
- **Policy Regret**：策略选择的动作与最优动作之间的价值损失期望，衡量隐式贪心策略的决策质量。
- **Cross-inference**：保持搜索策略先验不变，仅将叶节点价值评估替换为其他方法学到的价值函数的诊断性实验配置。
- **Deadly Triad**：函数逼近、自举（bootstrapping）和离策略学习三个要素的组合，已知在 MDP 中可能导致数值不稳定或发散。
- **Forward-equivalent compute proxy**：以 $N_{\text{forward}} + 3N_{\text{backward}}$ 量化训练阶段的神经网络计算开销的代理指标。
- **Strong/Weak Oracle**：强 Oracle（如 Connect Four 求解器）返回理论结果及距离结局步数；弱 Oracle（如 Hex 求解器）仅返回胜负结果。
- **F-Games**：程序化生成的合成博弈树，深度、分支因子和难度可控，最优值由构造过程已知。

## 可复现要素
- **数据集**：Connect Four（官方规则 + Pascal Pons 求解器）、Hex(7x7)（MoHex 求解器）、F-Games（程序化生成）、Othello/Go(9x9)（PGX 环境）；论文声明代码公开（GitHub）
- **代码**：JAX 实现，基于 PGX 和 mctx，代码已开源（论文脚注 2）
- **关键超参**：$N_{\text{envs}}=128$，$K=128$，batch size=256，lr=$3\times10^{-4}$，AVI $\epsilon=0.3$（Othello/Go 用 0.15），AZ $c_{\text{puct}}=3$，AZ S∈{32,64,128,256,512}，replay buffer=1,000,000
