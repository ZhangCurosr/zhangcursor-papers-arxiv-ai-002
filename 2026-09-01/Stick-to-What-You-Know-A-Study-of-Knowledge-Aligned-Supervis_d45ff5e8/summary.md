---
title: "Stick-to-What-You-Know-A-Study-of-Knowledge-Aligned-Supervis"
source: https://arxiv.org/pdf/2608.30987v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 02:33:49"
field: "大模型事实性与幻觉缓解"
keywords: ["knowledge-aligned SFT", "hallucination mitigation", "factuality", "instruction tuning", "claim decomposition", "reciprocal probing"]
innovations: ["提出知识对齐SFT统一框架，区分生成式与估计式对齐", "设计Recall Rewrite方法，通过多问题一致回忆估计模型参数知识", "系统揭示知识对齐强度与事实性/拒绝行为之间的权衡关系"]
benchmarks: ["Wild-Halu", "Biography", "UnknownBench", "OLMES"]
---

# 论文速读：Stick-to-What-You-Know-A-Study-of-Knowledge-Aligned-Supervis

## 一句话总结
本文研究通过**知识对齐的监督微调（SFT）**来缓解大语言模型的**事实性幻觉**问题，将训练目标约束在基础模型已内化的参数知识范围内，提出并统一评估了生成式与估计式对齐方法，并引入两种新变体（Evidence Rewrite 和 Recall Rewrite）。

## 研究问题与动机
1. **SFT 目标超出基础模型知识会驱动幻觉**：当前指令微调（SFT）常使用包含未知事实的训练目标，迫使模型模仿超出其参数知识的回答，从而产生幻觉。
2. **现有缓解方法缺乏统一比较**：已有工作分别提出生成式对齐（如 FLAME）和估计式对齐（如 UNIT_cut），但缺乏在统一框架下的系统对比。
3. **SFT 难以可靠注入新知识**：SFT 阶段数据量相对较小，不适合用于注入新事实关联；更好的方式是检索增强或持续预训练。
4. **拒绝行为与知识边界相关**：SFT 训练目标若要求模型回答其不知道的问题，会鼓励模型在不确定时猜测而非拒绝，违背知识边界。

## 核心贡献（创新点）
1. **提出知识对齐 SFT 的统一框架**：将现有方法和新提出的方法统一为基于声明分解与过滤的数据构建流程，明确区分生成式与估计式两种对齐策略。
2. **引入 Evidence Rewrite 方法**：通过外部证据验证基础模型生成的声明，仅保留有证据支持的声明并重新生成响应，改进了单纯依赖模型自身生成的不可靠性。
3. **提出 Recall Rewrite 方法**：无需外部证据，通过生成探测问题并测试基础模型是否能一致地回忆出原声明，从而更直接地估计模型的参数知识边界。
4. **在多个基准上进行统一实验评估**：在 Wild-Halu 和 Biography 数据集上全面比较不同对齐方法，并分析其在事实性、拒绝行为和一般能力上的权衡。
5. **揭示知识对齐强度与事实性/覆盖度之间的 trade-off**：通过控制已知声明比例（%Known）的消融实验，证明提高训练目标中的知识对齐程度能显著提升事实性，但会增加拒绝率。

## 方法详解
- **统一数据构建流程**：对每个训练样本 (P, R)，分解响应 R 为原子声明集合 C(R|P)，仅保留基础模型参数知识 κ(M_base) 内的声明，然后重写为新的响应 R*。
- **生成式对齐（FLAME）**：对知识寻求型提示，直接用基础模型生成的响应 R̂ 替代原始响应 R，隐含假设所有自生成声明均为已知。
- **估计式对齐（UNIT_cut）**：基于声明条件概率（CCP）阈值过滤声明，仅保留 CCP 高于阈值的声明。
- **Evidence Rewrite**：生成基础模型响应 R̂，利用外部证据（维基百科）验证每个声明的支持性，仅保留有证据支持的声明，并由重写模型生成 R*。
- **Recall Rewrite**：对每个知识依赖型声明 c_n，由教师模型生成 J 个探测问题，从基础模型采样 K 个回答，通过蕴含检查判断每个问题是否支持或矛盾于 c_n。声明被分类为已知当且仅当至少 j_e 个问题得到蕴含回答且至多 j_c 个问题得到矛盾回答。
- **知识分类公式**：声明 c_n 被一致回忆（consistent recall）的条件为：
  |{j : e_{n,j} ≥ k_e}| ≥ j_e 且 |{j : d_{n,j} ≥ k_c}| ≤ j_c
  其中 e_{n,j} 和 d_{n,j} 分别表示对问题 q_{n,j} 的回答中蕴含和矛盾的数量。

## 实验与结果
- **数据集**：OASST1（英语子集，3,468 条样本），评估基准为 Wild-Halu 和 Biography。
- **基线模型**：Standard SFT、FLAME、UNIT_cut、Evidence Rewrite、Recall Rewrite，基于 Qwen 3 4B-Base 和 OLMo 3 7B-Base。
- **主要结果**（Qwen 3 4B）：
  - **Wild-Halu**：Recall Rewrite 达到最高 FActScore（84.1），较 Standard SFT（74.4）提升 9.7 点；Evidence Rewrite 为 78.3，UNIT_cut 为 79.4。
  - **Biography**：Recall Rewrite FActScore 为 76.4，较 Standard SFT（34.1）大幅提升 42.3 点。
- **OLMo 3 7B 结果**：Recall Rewrite（FActScore 82.5 on WildHalu）优于官方 OLMo 3 Instruct 的各阶段 checkpoint。
- **拒绝行为**（UnknownBench）：Recall Rewrite 在所有子任务上取得最高 F1-Score（FalseQA: 68.7, NEC: 68.8, RefuNQ: 69.9），但精度较低（更多误拒）。
- **一般能力**（OLMES）：所有知识对齐方法在 HumanEval+、GSM8K、IFEval、TruthfulQA 上与 Standard SFT 表现相当，无明显下降。
- **消融实验**：将训练目标中已知声明比例从 0% 增至 100%，FActScore 单调上升（WildHalu: 79.5 → 86.1），但支持声明数量下降。

## 相关工作脉络
1. **FLAME (Lin et al., 2024)**：用基础模型生成响应替代金标准响应，是本文 Generation-based alignment 的代表，但未加入验证步骤，无法纠正模型自身的幻觉。
2. **UNIT_cut (Wu et al., 2025)**：基于 CCP 阈值过滤声明，是 Estimation-based alignment 的代表，但依赖 token 级置信度，易受表述影响。
3. **Calderon et al. (2026)**：同期工作，同样用多问题一致回答来估计模型知识，与 Recall Rewrite 思路类似。
4. **Kaplan et al. (2026)**：归因幻觉为事实性遗忘，提出优化正则化，与本文数据层面的对齐策略正交。
5. **FActScore (Min et al., 2023) / VeriScore (Song et al., 2024)**：用于声明分解与验证的评估工具，被本文采纳为事实检查管道。
6. **Retrieval-augmented 方法 (Ovadia et al., 2024; Xu et al., 2023)**：指出 SFT 不适合注入新知识，与本文“对齐而非扩充”的理念一致。

## 局限性与未来方向
1. **知识边界的近似**：参数知识 κ(M_base) 无法直接观测，只能通过行为近似，当前二值分类（已知/未知）忽略了置信度梯度。
2. **评估依赖自动事实检查**：声明分解、检索和验证可能出错，尤其对模糊或需要领域专家的声明。
3. **未充分探索与后续训练阶段的交互**：未研究知识对齐 SFT 与 DPO/RLVR 等后训练阶段的组合效果。
4. **可扩展性限制**：Recall Rewrite 需要多次采样和 entailment 检查，成本较高，难以直接应用于大规模指令微调数据。
5. **数据集规模小**：仅使用 OASST1 英语子集（3.4K 样本），需在更大、多语言、多轮对话等场景验证。
6. **重写管道依赖强教师模型**：gpt-4o-mini/gpt-5-mini 可能引入偏差，未来需探索开源教师或人工审计。

## 研究启发与可借鉴点
1. **声明分解与知识边界检测**：可将原子声明分解作为通用组件，用于检测 SFT 数据中超出模型知识的潜在风险点。
2. **探测式知识验证**：Recall Rewrite 的多问题一致回忆思路可迁移至其他模型评估场景，用于量化模型对特定事实的掌握程度。
3. **知识对齐作为数据筛选策略**：在构建指令微调数据集时，优先保留模型已内化的知识，可降低幻觉训练信号。
4. **拒绝行为建模**：将“拒绝回答”作为显式输出类别纳入训练数据，有助于提升模型在未知问题上的保守性。
5. **统一框架促进公平比较**：将不同对齐方法纳入同一管道，仅改变 GATE/SOURCE/UNKNOWN 钩子，便于控制变量研究。

## 关键术语表
- **Knowledge-aligned SFT**：将监督微调训练目标约束在基础模型已掌握的参数知识内的微调策略。
- **Parametric knowledge κ(M_base)**：基础模型在预训练阶段内化的事实性知识集合。
- **Atomic claim**：可独立验证的最小事实陈述单元。
- **Knowledge-dependent claim**：需要依赖模型参数知识才能正确响应的声明（如事实、程序、结构信息）。
- **Consistent recall**：Recall Rewrite 的判断标准，即基础模型在多次不同问法下能否一致地回忆出原声明。
- **FActScore**：每示例支持声明比例的平均值，将拒绝视为完全支持。
- **UnknownBench**：评估模型拒绝不可回答问题的能力，包含 FalseQA、NEC、RefuNQ 三个子任务。
- **Wild-Halu**：包含约 500 个真实实体的长文事实生成基准，约一半实体无维基百科页面。

## 可复现要素
- **数据集**：OASST1（英语子集，3,468 条样本），公开可用；论文已发布 Recall Rewrite 训练数据及中间流水线输出。
- **代码/权重**：模型权重基于 Qwen 3-4B-Base 和 OLMo 3-7B-Base；训练使用 TRL 库；流水线调用 gpt-4o-mini/gpt-5-mini API。
- **关键超参**：Epoch=5，Batch size=32，学习率=1e-5，Context length=1024；Recall Rewrite 默认过滤器 j_e/k_e/j_c/k_c=2/1/2/1，J=5 个问题，K=2 个回答。
- **评估基准**：Wild-Halu、Biography、UnknownBench、OLMES（HumanEval+, GSM8K, IFEval, TruthfulQA）。
