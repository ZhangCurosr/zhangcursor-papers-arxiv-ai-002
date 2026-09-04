---
title: "Sequential-Beats-Joint-On-the-Interplay-between-On-Policy-Di"
source: https://arxiv.org/pdf/2609.04108v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 18:54:07"
field: "大语言模型后训练"
keywords: ["RLVR", "on-policy distillation", "OPD", "post-training", "reasoning LLMs", "sequential training", "policy gradient"]
innovations: ["提出OPD-then-RL两阶段顺序训练方案，系统证明其优于所有联合优化方法", "从pass@k行为、学习动态、参数更新三维度揭示OPD与RLVR的信号干扰机制", "统一token级策略梯度视角分类现有OPD-RLVR混合方法"]
benchmarks: ["Reasoning-Gym (K&K, Zebra, Countdown)", "DeepMath-103K", "MATH-500", "AMC23", "AIME24", "AIME25"]
---

# 论文速读：Sequential-Beats-Joint-On-the-Interplay-between-On-Policy-Di

## 一句话总结
本文系统研究后训练推理LLM中**在线策略蒸馏（OPD）**与**可验证奖励强化学习（RLVR）**的组合策略，发现简单的两阶段**OPD-then-RL**方案在逻辑与数学推理任务上一致优于纯OPD、纯RLVR及所有现有联合优化方法，并通过pass@k行为、学习动态和参数更新三方面机制揭示：顺序训练使OPD负责扩展能力边界、RL负责锐化性能，而联合优化会导致两种信号相互干扰。

## 研究问题与动机
- **核心问题**：RLVR和OPD具有互补性——RLVR使用稀疏序列级奖励直接优化任务目标但探索成本高；OPD提供密集token级监督提升学习效率但仅优化行为代理而非真实任务表现。如何最优组合两者是关键开放问题。
- **现有方法不足**：
  1. 现有混合方法均在单步内融合两种信号（加权相加或教师调制），已被证明对超参数敏感且存在信号干扰。
  2. 联合优化方法在数学推理上反而不如纯OPD，说明两种信号在某些场景下冲突而非互补。
  3. 缺乏对OPD-RL交互机制的系统性理解（为何有时有效、何时失效）。

## 核心贡献（创新点）
1. **统一视角分类现有方法**：从token级策略梯度角度将现有OPD-RLVR混合方法系统整理为两类范式——加权相加（weighted-additive）和教师调制（teacher-modulated），揭示它们均通过修改优势函数或重要性采样比率来融合信号。
2. **提出并验证OPD-then-RL方案**：发现简单的两阶段顺序训练（先OPD后RL）在逻辑和数学推理基准上持续优于纯方法、联合方法及调度方法，逻辑推理任务最高提升达26.7个pass@1点。
3. **三维度机制解释**：通过pass@k分析揭示OPD扩展能力边界而RL锐化的互补机制；通过学习动态证明持续RL能突破教师性能天花板；通过参数更新分析量化显示顺序训练保持OPD更新结构完整性，联合方法产生符号冲突。

## 方法详解
**统一token级策略梯度视角**：RL和OPD目标均可化为通用形式$\nabla_\theta \mathcal{I}^{alg}(\theta) = \mathbb{E}[\sum_{i,t} A_t^{alg,(i)} \nabla_\theta \log \pi_\theta(y_t^{(i)}|h_t^{(i)})]$，区别仅在于token级优势函数$A_t$：
- GRPO优势：$\hat{A}^{(i)} = \frac{R^{(i)} - \text{mean}(\{R^{(j)}\})}{\text{std}(\{R^{(j)}\})}$（组归一化优势）
- OPD优势：$d_t^{(i)} = \log \pi_T(y_t^{(i)}|h_t^{(i)}) - \log \pi_\theta(y_t^{(i)}|h_t^{(i)})$（token级蒸馏优势）

**两类联合优化范式**：
1. **加权相加**：$A_t^{\text{add},(i)} = w_R^{(i,t)}\hat{A}^{(i)} + w_T^{(i,t)}d_t^{(i)}$，如KDRL、SRPO、HDPO。教师信号可翻转优势符号，与验证器冲突。
2. **教师调制**：$A_t^{\text{mod},(i)} = m(d_t^{(i)}) \cdot \hat{A}^{(i)}$，如TRRD、RLSD。教师仅调制幅度不改变符号，作为信任域正则化。

**OPD-then-RL方案**：
$$A_t^{alg,(i)} = \begin{cases} d_t^{(i)}, & \text{training step} \leq S \\ \hat{A}^{(i)}, & \text{otherwise} \end{cases}$$
其中$S$为切换点（默认60步）。OPD阶段最大化$J_{OPD}$扩展解空间覆盖，RL阶段最大化$J_{GRPO}$锐化正确路径概率质量。

## 实验与结果
**实验设置**：
- **学生模型**：Qwen3-1.7B-Base（主实验）、Qwen3-0.6B-Base（泛化验证）
- **教师模型**：Qwen3-8B（冻结）
- **逻辑推理**：Reasoning-Gym的三个任务——Knights & Knaves (K&K)、Zebra Puzzles、Countdown
- **数学推理**：训练集DeepMath-103K，测试集MATH-500、AMC23、AIME24、AIME25

**主要结果**：
| 任务类型 | 方法 | 平均pass@1 | 平均pass@32 |
|---------|------|-----------|------------|
| 逻辑推理 | OPD-then-RL | **80.6** | 98.3 |
| 逻辑推理 | KDRL（最佳联合） | 62.8 | 92.4 |
| 逻辑推理 | GRPO | 49.4 | 68.1 |
| 逻辑推理 | OPD | 53.9 | 94.9 |
| 数学推理 | OPD-then-RL | **31.8** | 58.5 |
| 数学推理 | SRPO（最佳联合） | 31.6 | 56.5 |
| 数学推理 | GRPO | 28.4 | 55.7 |
| 数学推理 | OPD | 31.0 | 55.9 |

- **逻辑推理**：OPD-then-RL领先所有方法11.7-26.7个pass@1点，统计显著性验证通过配对bootstrap检验。
- **数学推理**：OPD-then-RL虽 margin 较小，但仍显著优于6/9的竞争方法，与最强三者持平。
- **关键发现**：加权相加方法在逻辑推理上提升pass@1但侵蚀pass@32（+4.8/-2.0），而在数学推理上反而低于纯OPD（-1.2）。教师调制方法更保守，保持OPD性能同时小幅改善pass@1。

## 相关工作脉络
1. **RLVR方法**：GRPO（Shao et al., 2024）、DeepSeek-R1（Guo et al., 2025）、Kimi-K2.5（Kimi Team, 2025）。本文与之区别：不关注纯RLVR的超参数调优，而是研究其与OPD的交互机制。
2. **知识蒸馏**：off-policy SFT（Muennighoff et al., 2025）、on-policy distillation OPD（Lu and Lab, 2025; Agarwal et al., 2024）。本文定位：将OPD视为RLVR的冷启动而非替代方案。
3. **SFT-then-RL序列方法**：Limozin et al. (2026)提出的SFT冷启动→RL方案。本文扩展：证明OPD比SFT提供更强的冷启动（平均高出4.9 pass@1点），且在RL后差距进一步拉大。
4. **KDRL**（Xu et al., 2025）：加权相加范式代表，添加常数蒸馏项。本文揭示其对超参数β极度敏感（图6消融），且在非设计域（逻辑推理）性能下降。
5. **SRPO**（Li et al., 2026a）：按正确性门控两种信号的加权相加方法。本文实验显示其在数学推理上仍不如OPD-then-RL。
6. **TRRD**（Zhang et al., 2026b）：教师调制范式代表，将教师信号折叠入重要性比率作为信任域正则化。本文证明其保守策略虽保留pass@32但限制了pass@1提升潜力。

## 局限性与未来方向
**自述局限**：
1. **教师-学生配置范围**：仅研究外部教师强于学生的场景，未探索任务特定教师、多教师设置及on-policy自蒸馏（教师为学生自身增强版）。
2. **冻结商用教师**：采用固定教师以隔离信号分析，未与"先RLVR改进教师再蒸馏给学生"的工业流水线模式进行计算预算公平比较。
3. **OPD目标选择**：仅使用reverse-KL目标，未探索full next-token分布、forward-KL或Jensen-Shannon散度等替代目标。

**合理推断的未来方向**：
- 探索OPD-then-RL在自蒸馏、多教师、低资源场景的适用性。
- 研究动态切换策略（如基于验证分数的自适应切换）替代固定步数切换。
- 分析OPD-then-RL与非思维模型（thinking model）的配合效果。

## 研究启发与可借鉴点
1. **方法迁移价值**：顺序训练（sequential training）作为解耦互补信号的有效策略，可推广至其他需要融合多源监督信号的后训练场景（如RLHF+ Preference Alignment的组合）。
2. **实验设计借鉴**：
   - 使用pass@k曲线分析替代单一pass@1指标，能更全面揭示方法的能力边界与锐化能力差异。
   - 参数更新冲突率（SCR）的定量分析提供了机制解释的新工具，可迁移至其他多目标优化冲突诊断。
3. **团队结合创新机会**：
   - 可将OPD-then-RL作为新基线，应用于团队现有的推理增强任务，尤其适合教师模型能力明显强于学生的场景。
   - 探索"OPD-then-RL-then-OPD"的多阶段循环训练，研究阶段间信号传递的最佳模式。
4. **工程实践要点**：
   - 切换点的选择标准：以OPD验证分饱和为标志，而非固定步数。
   - EOS token掩码处理：当师生tokenizer不一致时，需屏蔽EOS位置的教师信号以避免人为惩罚。

## 关键术语表
**RLVR（Reinforcement Learning with Verifiable Rewards）**：使用可验证规则奖励的强化学习方法，通过序列级 outcome reward 直接优化任务目标，无需人工标注偏好数据。

**OPD（On-Policy Distillation）**：在线策略蒸馏，在学生对自身rollout的采样分布上，使用token级teacher信号最小化与teacher的reverse-KL散度，提供密集监督信号。

**pass@k**：在k次采样中至少一次正确的概率，衡量模型能力边界；pass@1反映单次采样正确率，pass@32反映多次采样时的整体能力。

**加权相加范式（Weighted-Additive）**：将RL优势与OPD优势线性加权求和，教师信号可翻转优势符号，代表方法包括KDRL、SRPO、HDPO。

**教师调制范式（Teacher-Modulated）**：用教师信号调制RL优势的幅度而不改变符号，作为信任域正则化，代表方法包括TRRD、RLSD。

**OPD-then-RL**：先执行OPD训练（默认60步）扩展学生覆盖teacher支持的解空间，再切换到纯GRPO锐化概率分布的两阶段顺序训练方案。

**sign-conflict rate (SCR)**：衡量某方法参数更新与OPD更新方向冲突的比例，用于量化联合优化对OPD更新结构的破坏程度。

## 可复现要素
- **数据集**：
  - 逻辑推理：Reasoning-Gym的K&K、Zebra Puzzles、Countdown（通过官方procedural generators生成，共20,000训练样本）
  - 数学推理：DeepMath-103K训练集，MATH-500、AMC23、AIME24、AIME25测试集
  - 数据公开状态：Reasoning-Gym开源，DeepMath-103K和 benchmarks 均为公开标准数据集
- **代码开源**：论文未明确提及代码开源状态，实现基于VERL框架
- **权重开源**：使用公开模型Qwen3系列（1.7B-Base、0.6B-Base、8B）
- **关键超参**：
  - 学习率：$1 \times 10^{-6}$（AdamW）
  - PPO clip ratio：0.2
  - KL惩罚系数：$\beta_{KL} = 0$（无显式KL项）
  - OPD训练步数：60步（切换点）
  - GRPO rollout数：8
  - 批次大小：128（Reasoning-Gym）、128（DeepMath）
  - 最大响应长度：4096（逻辑）、8192（数学）
