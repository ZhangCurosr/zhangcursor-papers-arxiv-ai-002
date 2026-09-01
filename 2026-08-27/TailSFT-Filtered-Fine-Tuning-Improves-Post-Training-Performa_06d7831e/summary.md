---
title: "TailSFT-Filtered-Fine-Tuning-Improves-Post-Training-Performa"
source: https://arxiv.org/pdf/2608.25756v1.pdf
model: agnes-2.5-flash
chunks: 3
summarized_at: "2026-09-01 16:14:52"
field: "大语言模型训练与对齐"
keywords: ["TailSFT", "Filtered Fine-Tuning", "Post-Training", "Coverage", "RL Initialization", "Multi-stage Training", "SFT"]
innovations: ["提出阶段感知SFT范式，主张以下游RL效果评判SFT初始化质量", "设计TailSFT过滤算法，基于初始策略loss offset自适应保留尾部样本", "引入Coverage Ratio诊断指标ρ₁₆，低成本预测TailSFT有效性"]
benchmarks: ["AIME", "MATH-500", "OMEGA-500", "MBPP+", "HumanEval+", "CruxEval-I/O", "LiveCodeBench"]
---

# 论文速读：TailSFT: Filtered Fine-Tuning Improves Post-Training Performance

## 一句话总结
本文提出 **TailSFT**，一种在监督微调（SFT）阶段过滤掉已被初始策略充分建模序列的轻量级方法，将训练资源集中于响应分布的"尾部"，从而提升后续强化学习（RL）训练的覆盖率与最终性能，在 Math 和 Code 任务上均取得显著增益。

---

## 研究问题与动机
- **核心矛盾**：标准 SFT 以交叉熵最小化为目标，但该局部最优与后续 RL 训练的需求并不一致——前者奖励所有训练样本的概率提升，后者依赖初始化模型在有限 rollout 预算内采到 reward-bearing response。
- **标准 SFT 的缺陷**：交叉熵持续拟合"已学好的"响应，将概率质量从有用的、难生成的响应上移走，导致 coverage 下降，即使 cross-entropy 和 pass@1 变好。
- **中间检查点的评判标准**：多阶段训练中，不应仅以当前阶段指标（如 loss、pass@1）评估中间 checkpoint，而应以后续训练阶段的效果来评判其价值。
- **Coverage 的关键性**：RL 成功强依赖初始化模型能否在有限 rollout 预算内覆盖高 reward 响应；`pass@K`（大 K）是 coverage 的可计算代理指标。

---

## 核心贡献（创新点）
1. **提出阶段感知 SFT 范式**：主张前一阶段的局部最优目标不应孤立优化，而应考虑对后续阶段（如 RL）的支撑效果。
   - 与已有工作的本质区别：现有工作多关注单阶段优化（如纯 SFT 或纯 RL），本文首次在 SFT 阶段引入"为下游 RL 初始化"的设计视角。

2. **设计 TailSFT 过滤算法**：基于初始策略 π₀ 的 loss offset 进行序列级过滤，自适应识别并保留"未充分建模"的训练样本。
   - 与已有工作的本质区别：区别于 absolute filtering（固定阈值）或 quantile filtering（每 batch 静态截断），TailSFT 通过相对初始策略的增益排序实现动态、自适应过滤。

3. **引入 Coverage Ratio 诊断指标 ρ₁₆**：仅需基座模型 + 一次标准 SFT 即可预测 TailSFT 是否有效，为实践提供低成本筛选工具。
   - 与已有工作的本质区别：这是首个针对 SFT-to-RL 衔接场景的 coverage 可计算代理指标，而非直接测量最终 RL 性能。

4. **系统性实验验证**：在 OLMo-3 7B 上，覆盖 18 个数据集-基准对，TailSFT 在 15/18 情况下提升 pass@16，且 GRPO 后初始化优势一致。
   - 与已有工作的本质区别：现有 SFT 改进工作多关注 pass@1 或 loss，本文强调大 K coverage 的提升及其对 RL 的传导效应。

---

## 方法详解
**TailSFT 算法（Algorithm 1）**：
- 记录每个训练样本在初始策略 π₀ 下的 loss `ℓ_i^0`
- 每步采样 batch `B_t`，计算当前 loss `ℓ_i^t`
- 按 `ℓ_i^t - ℓ_i^0`（loss offset）**升序**排序，选取增益最大的 `γ_t fraction` 作为过滤集合 `F_t`（即 offset quantile filtering）
- 对被过滤集合外的样本使用**长度归一化的 token-level cross-entropy** 进行梯度更新

**三种过滤变体（消融对比）**：
- **TailSFT**：offset quantile filtering（相对初始策略 π₀）
- **Absolute filtering**：loss 低于固定阈值时停止训练
- **Quantile filtering**：每 batch 过滤 loss 最低的固定分数（无参考基线）

**Coverage Ratio 诊断（Definition 4.1）**：
- 定义 base-reachable 集合 `R_0 = {i : 0.05 < P_{i,16}(π₀) < 0.95}`
- 将 pass@1 转换为估计的 pass@16 值：`f_{16}(p) = 1-(1-p)^16`
- `L` = 标准 SFT 在该集合上损失的 coverage，`G` = 获得的 coverage
- **Coverage ratio `ρ₁₆ = L / G`**；`ρ₁₆ > 1` 意味着标准 SFT 损失的 coverage 多于获得，TailSFT 最可能有效

**理论保证（Theorem 3.1）**：
- offset filtering 可调至达到不低于标准 SFT 或最优 absolute filtering 的 coverage
- π₀ 的作用：不仅作为初始化，还编码了"奖励响应之间的概率相对偏好结构"；offset filtering 通过初始概率自适应决定何时停止更新，而 absolute filtering 施加通用阈值会抹除该结构。

---

## 实验与结果
**实验设置**：
- **模型**：OLMo-3 7B
- **SFT 数据**：
  - Math：OMI（OpenMathInstruct-2，350k 样本，去污染）
  - Code：BigCode Self-OSS-Instruct、Magicoder、OCI（OpenCodeInstruct）
- **评估基准**：AIME 2022–2025、MATH-500 Level 5、OMEGA-500、MBPP+、HumanEval+、CruxEval-I/O、LiveCodeBench（OLMES 协议）
- **Coverage 度量**：pass@16（SFT 后）；RL 后评估用 pass@1
- **控制实验**：Graph Navigation 任务（GPT-2 架构），每个 prompt 含恰好 8 条合法路径，SFT 目标为确定性单路径选择

**关键结果**：
- **Table 1（SFT 结果，18 个数据集-基准对）**：
  - TailSFT 在 **15/18** 情况下改善 pass@16，pass@1 变化混杂
  - 最大提升：**CruxEval-O（BigCode 数据）pass@16 +16.79%**；Magicoder 数据 CruxEval-O **+9.83%**；AIME **+3.07%**
  - OMEGA-500 几乎无变化（-0.20%）
  - 规律：大 K 增益远比 pass@1 稳定，与 coverage 动机一致

- **Figure 4（Coverage Ratio 诊断验证）**：
  - **11 个 ρ₁₆ > 1 的设置中，10 个 TailSFT 正增益、1 个基本不变**，最大增益达 +28.69%（BigCode→CruxEval-O）
  - ρ₁₆ > 1 是充分条件而非必要条件：部分 ρ₁₆ < 1 的设置也受益

- **Table 2（GRPO 后评估）**：
  - 以 TailSFT checkpoint 初始化 GRPO，**所有匹配比较均优于标准 SFT 初始化**：
    - Math（MATH-500 Level 5）：pass@1 **+2.56%**，pass@16 **+3.23%**
    - Math（AIME）：pass@1 **+1.21%**，pass@16 **+3.30%**
    - Code（MBPP+ BigCode）：pass@1 **+3.93%**，pass@16 **+2.73%**
    - Code（MBPP+ Magicoder）：pass@1 **+2.72%**

---

## 相关工作脉络
- **标准 SFT**：以交叉熵最小化为目标，独立优化各样本概率，不关心后续 RL 阶段的 coverage 需求。
- **Active Learning / Curriculum Learning**：关注样本选择策略，但多以当前任务性能为目标，而非面向多阶段训练的初始化优化。
- **Loss-based Filtering**：absolute filtering 使用固定阈值截断，本文指出其会抹除 π₀ 编码的概率偏好结构。
- **RL after SFT（如 GRPO、PPO）**：现有工作关注 RL 算法改进，较少讨论 SFT 初始化对 RL coverage 的影响。
- **Coverage-aware 训练**：本文首次将 pass@K（大 K）作为 SFT 阶段可优化的代理目标，并设计对应的过滤机制。
- **阶段感知训练（Phase-aware Training）**：主张中间 checkpoint 应以对下游任务的支撑效果评判，而非当前阶段指标。

---

## 局限性与未来方向
- **ρ₁₆ 仅为充分条件**：部分 ρ₁₆ < 1 的设置也受益，说明诊断指标存在遗漏，未来需探索更精确的预测指标。
- **过滤粒度为序列级**：当前方法按整个序列的 loss offset 过滤，未考虑序列内部 token 级的差异；未来可探索 token-level filtering。
- **仅验证 on OLMo-3 7B**：模型规模与架构的泛化性有待验证，尤其在大模型（如 70B+）上的表现。
- **RL 算法仅限 GRPO**：未测试 PPO、REINFORCE 等其他 RL 对齐方法，普适性需进一步验证。
- **π₀ 的选择**：当前使用预训练 checkpoint 作为 π₀，未来可探索不同初始化策略对过滤效果的影响。

---

## 研究启发与可借鉴点
- **多阶段训练的初始化设计**：可将"下游任务支撑效果"作为上一阶段优化的隐式目标，适用于 SFT→RL、Pretrain→SFT 等多阶段 pipeline。
- **Coverage 代理指标的设计**：ρ₁₆ 式的诊断指标为低成本筛选有效训练策略提供了思路，可迁移至其他多阶段场景。
- **相对阈值 vs 绝对阈值**：offset filtering 通过参考基线（π₀）实现自适应，比固定阈值更能保留概率结构，可推广至其他过滤场景。
- **长度归一化 loss 的使用**：避免长序列在 loss 计算中的主导效应，可结合其他归一化策略（如难度加权）进一步优化。
- **实验设计的控制任务**：Graph Navigation 任务提供了严格的 coverage 可控实验环境，可作为方法验证的通用测试床。

---

## 关键术语表
- **TailSFT**：一种在 SFT 阶段过滤掉已被初始策略充分建模序列的轻量级微调方法，将训练资源集中于响应分布的"尾部"。
- **Coverage Principle**：RL 成功强依赖初始化模型能否在有限 rollout 预算内采到高 reward 响应，`pass@K`（大 K）是其可计算代理。
- **Offset Quantile Filtering**：按当前 loss 与初始策略 loss 的差值（offset）排序，选取增益最大的样本子集进行训练。
- **Coverage Ratio（ρ₁₆）**：标准 SFT 损失的 coverage 与获得的 coverage 之比，`ρ₁₆ > 1` 预示 TailSFT 有效。
- **Base-reachable 集合**：初始策略 π₀ 下 pass@16 概率介于 0.05~0.95 之间的样本集合，用于诊断 coverage。
- **pass@K**：从 K 次采样中至少有一次正确的概率，衡量模型覆盖高 reward 响应的能力。
- **阶段感知范式**：主张多阶段训练中前一阶段的优化目标应考虑对后续阶段初始化效果的支撑。
- **RL after SFT**：先进行监督微调再以强化学习对齐的 pipeline，本文聚焦 SFT 阶段对 RL 初始化质量的优化。

---

## 可复现要素
- **数据集**：OMI（OpenMathInstruct-2，350k）、BigCode Self-OSS-Instruct、Magicoder、OCI（OpenCodeInstruct）；评估基准 AIME、MATH-500、OMEGA-500、MBPP+、HumanEval+、CruxEval-I/O、LiveCodeBench
- **代码/权重**：论文未提及开源声明
- **关键超参**：过滤比例 γ_t、length-normalized token-level cross-entropy、base-reachable 阈值 0.05/0.95
- **模型**：OLMo-3 7B
- **RL 算法**：GRPO（具体超参论文未详述）

---
