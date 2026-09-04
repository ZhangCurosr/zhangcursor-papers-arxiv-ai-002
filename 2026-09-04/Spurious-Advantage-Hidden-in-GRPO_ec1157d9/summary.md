---
title: "Spurious-Advantage-Hidden-in-GRPO"
source: https://arxiv.org/pdf/2609.04063v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 12:06:58"
field: "强化学习与大模型对齐"
keywords: ["GRPO", "reinforcement learning with verifiable rewards", "advantage estimation", "spurious advantage", "math reasoning", "search agent"]
innovations: ["揭示GRPO优势估计器在有界答案任务中的虚假优势问题", "提出SIGNBALANCE解耦幅度与组内统计的优势估计器", "系统分析三类触发spurious advantage的场景并提供统一框架"]
benchmarks: ["MATH-7.5K", "GSM8K", "MATH-500", "Minerva-Math", "OlympiadBench", "MMLU-math", "SAT-Math", "AQuA", "AMC", "NQ", "TriviaQA", "PopQA", "HotpotQA", "2WikiMultiHopQA", "MuSiQue"]
---

# 论文速读：Spurious-Advantage-Hidden-in-GRPO

## 一句话总结
论文揭示GRPO优势估计器中存在的"虚假优势(spurious advantage)"问题：在答案空间有限或存在猜测可能的任务中，GRPO会给"猜对"而非"推理对"的rollout赋予相同的高梯度权重，误导策略学习。作者提出SIGNBALANCE，通过解耦幅度与组内正确答案数量的依赖关系，在有界答案数学题和搜索代理任务上取得显著提升。

## 研究问题与动机
- **核心问题**：GRPO的组内归一化优势估计器$(n^+, n^-)$会放大非推理来源的正确回答，因为公式无法区分"推理获得正确答案"和"猜测获得正确答案"两种情况，导致梯度权重混入虚假信号。
- **有界答案任务的误导**：在$k$选项选择题中，无推理的随机猜测以概率$1/k$命中正确答案；GRPO给这类"碰对的"rollout赋予与"推理对的"rollout相同的梯度放大权重，长期训练鼓励模型走捷径。
- **开放数据集中的隐藏有界子集**：即使Nominally open-answer的MATH-7.5K中，55.95%的题目答案形式是有界的（如小整数、简单分数），非推理猜测概率不可忽略。
- **多步搜索代理的outcome-only奖励问题**：搜索轨迹$(T_i, a_i)$不同但终点答案相同，GRPO赋予相同幅度，将无效/冗余搜索路径也纳入高梯度更新。

## 核心贡献（创新点）
1. **首次系统性揭示GRPO的spurious advantage现象**：将组内统计分解为推理来源$p_r$与非推理猜测来源$(1-p_r)\cdot p_g$，证明当$p_g$不可忽略时GRPO梯度混入虚假信号。
2. **识别三类易触发spurious advantage的场景**：有界答案任务（选择题）、开放答案数据集中隐藏的有界子集（如MATH中55.95%题目）、多步搜索代理（outcome-only奖励），提供了统一的分析视角。
3. **提出SIGNBALANCE优势估计器**：幅度与组内$(n^+, n^-)$解耦，仅保留verifier符号，通过stop-gradient每类力平衡恢复零均值，参数-free且无额外推理成本。
4. **在数学推理与搜索代理任务上验证一致性提升**：0.5B/3B规模下，SIGNBALANCE在开放答案任务上匹配GRPO，在有界答案任务上显著超越（SAT-Math +6.26、AQuA +5.90）；搜索代理任务上Avg-6达37.80，超过Search-R1 +1.80。

## 方法详解
**SIGNBALANCE三阶段设计：**

- **Step 1（按类归一化）**：
  $$\hat{A}_i^{(1)} = \frac{r_i - \mu^{c(i)}}{\sigma^{c(i)} + \varepsilon}$$
  对正确/错误子组分别归一化，使正确rollout不再与大量错误rollout比较，但幅度仍依赖$n^+$/$n^-$。

- **Step 2（仅保留符号幅度）**：
  $$\hat{A}_i^{(2)} = \text{sign}(r_i) \cdot c$$
  彻底解耦幅度与$(n^+, n^-)$，但破坏batch-level零均值性质（$n^+ \neq n^-$时正负力不平衡）。

- **Step 3（符号 + 每类力平衡，最终方案）**：
  $$\hat{A}^+ = c, \quad \hat{A}^- = -c \cdot \text{sg}\left[\frac{n^+}{n^-}\right]$$
  其中$\text{sg}[\cdot]$为stop-gradient算子。正确rollout幅度固定为$c$，错误rollout幅度按$n^+/n^-$比例缩放以保持总力平衡，且不引入梯度依赖。

**关键公式推导**：GRPO在二元奖励下的闭式解为$\hat{A}^+ = \sqrt{n^-/n^+}$，幅度完全由组内正确率决定。SIGNBALANCE将其替换为常量$c=1$，错误端乘以$\text{sg}[n^+/n^-]$，使得$\sum_{i \in +}|\hat{A}_i| = \sum_{i \in -}|\hat{A}_i|$。

**与PPO-surrogate的兼容性**：SIGNBALANCE是GRPO损失中优势估计器的直接替换，不修改PPO clipped surrogate公式，无需引入额外参数或外部模型。

## 实验与结果
**实验设置**：
- 骨干模型：Qwen2.5-0.5B-Instruct（主实验）、Qwen2.5-3B-Base（扩展）、Qwen2.5-7B-Instruct（搜索代理）
- 训练数据：MATH-7.5K，G=16 rollouts/question，LR=$1\times10^{-6}$，KL系数$\beta=1\times10^{-3}$，最多1000步
- 基线：PPO、REINFORCE++、RLOO、GRPO、Dr.GRPO、DAPO、BNPO

**主要结果**：

| 任务 | 最强方法 | 关键数字 |
|------|---------|---------|
| 0.5B Open-answer Avg-8 | SIGNBALANCE | 36.61（DAPO 36.27次之）|
| 0.5B SAT-Math | SIGNBALANCE | **71.88** vs GRPO 65.62 (+6.26) |
| 0.5B AQuA | SIGNBALANCE | **35.43** vs GRPO 29.53 (+5.90) |
| 3B Avg-8 | SIGNBALANCE | **43.78** vs GRPO 42.80 (+0.98) |
| 3B Gaokao/AMC/AIME | SIGNBALANCE | +1.2 / +1.7 / +0.8 |
| 搜索代理 Avg-6 | SIGNBALANCE | **37.80** vs Search-R1 36.00 (+1.80) |

**消融实验**：Step 2（仅符号）移除spurious advantage但破坏力平衡，导致MATH-500下降4.80；Step 3恢复力平衡后全面提升。手写非对称boost（Asym. boost: $\hat{A}^+=2.5, \hat{A}^-=-1$）性能介于Step 2和Step 3之间，证明自适应力平衡优于固定非对称。

**稳定性分析**：SIGNBALANCE与GRPO差距在训练前~100步即分离并保持，排除单次幸运checkpoint的解释。

## 相关工作脉络
- **GRPO及变体**：GRPO (Shao et al., 2024) 移除value network，用组内统计估计优势；后续Dr.GRPO、DAPO、BNPO等优化clip范围、采样策略、归一化方式。本文指出这些改进均在"open-answer math"假设下验证，未考虑有界答案场景。
- **Mroueh (2025)** 推导GRPO隐式权重的闭式解$\omega^+(p)=\sqrt{(1-p)/p}$，揭示GRPO等价于自适应加权对比损失——稀有成功rollout获得大权重。本文在此基础上指出：该稀有性可能来自猜测而非推理。
- **Advantage Estimator理论工作**：Yang et al. (2026) 从cross-prompt期望角度分析bias；He et al. (2025a) 从rank-bias角度；Zhang et al. (2026) 从KL约束闭式解角度。本文聚焦于"组内composition含non-reasoning成分"这一未被讨论的情形。
- **Search-R1 (Jin et al., 2025)** 训练多步搜索代理，使用outcome-only奖励。本文发现此类setting天然触发spurious advantage，并验证SIGNBALANCE可进一步提升。
- **StepSearch (Wang et al., 2025b)** 引入step-level reward shaping，在MuSiQue上优于SIGNBALANCE，但属正交改进（流程层面vs优势估计层面）。

## 局限性与未来方向
- **验证场景有限**：仅在数学推理和搜索代理任务上验证，尚未扩展到工具选择（tool selection）、代码生成等更通用setting。
- **缺乏推理能力的定量测量**：虽观察到SIGNBALANCE引导策略走向更有效推理，但未量化reasoning confidence、uncertainty behavior across difficulty levels等指标。
- **多步搜索中的轨迹多样性**：文中承认$(T_i, a_i)$不同时reward相同，但SIGNBALANCE仍只关注最终答案，未对搜索过程本身建模。

## 研究启发与可借鉴点
1. **优势估计与任务结构解耦**：SIGNBALANCE的核心洞察——将verifier sign与magnitude解耦，并独立恢复force balance——可迁移至其他RLVR setting，尤其是答案空间受限时。
2. **训练数据分析的价值**：论文对MATH-7.5K按答案形式分类，揭示55.95%为"有界子集"，这一分析方法（按syntactic shape分类gold answer并估算$p_g$）可用于其他数据集的质量诊断。
3. **停止梯度的巧用**：用stop-gradient处理$n^+/n^-$以恢复force balance而不引入计数依赖，是一种简洁的参数-free技巧，可借鉴于其他需要"统计量参与但不是gradient源"的场景。
4. **实验设计的对照逻辑**：所有方法共享相同backbone、数据、优化和eval pipeline，唯一controlled variable是advantage estimation mechanism，确保结论干净可信。

## 关键术语表
**GRPO (Group Relative Policy Optimization)**：移除value network的RLVR算法，用组内二元奖励的标准化差值估计每个rollout的优势。

**Spurious Advantage (虚假优势)**：GRPO优势估计器中源于非推理成功（如猜测）的梯度权重成分，会误导策略学习。

**SIGNBALANCE**：论文提出的优势估计器，幅度固定且仅保留verifier符号，通过stop-gradient每类力平衡恢复零均值。

**Within-group Composition**：组内正确数$n^+$与错误数$n^-$的比例分布，GRPO用它计算幅度。

**Outcome-only Reward**：仅根据最终答案是否匹配给定奖励，不惩罚中间无效步骤，常见于搜索代理训练。

**Stop-gradient (sg[·])**：阻止梯度通过的算子，使某量参与前向计算但不影响反向传播。

**Per-class Force Balance**：要求组内正确侧与错误侧的总梯度贡献力大小相等，恢复batch-level零均值。

**Random-guess Share**：correct rollouts中可能由随机猜测产生的比例，用于量化spurious advantage的严重程度。

## 可复现要素
- **数据集**：MATH-7.5K（公开）、MMLU-math、SAT-Math、AQuA、AMC（公开）；搜索代理使用Search-R1基准NQ/TriviaQA/PopQA/HotpotQA/2WikiMultiHopQA/MuSiQue（公开）
- **代码/权重**：论文声明"Code will be released"（未提及具体时间）；骨干模型Qwen2.5-0.5B/3B/7B-Instruct开源
- **关键超参**：G=16 rollouts/question，learning rate=$1\times10^{-6}$，KL系数$\beta=1\times10^{-3}$，max steps=1000，global scale $c=1$
- **实现细节**：使用vLLM生成rollouts，reward为基于ground-truth的rule-based binary verifier
