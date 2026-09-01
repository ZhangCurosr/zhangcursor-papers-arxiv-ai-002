---
title: "What-Proves-You-Wrong-Benchmarking-Language-Models-on-Falsif"
source: https://arxiv.org/pdf/2608.22948v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 01:11:17"
---

# 论文速读：What Proves You Wrong: Benchmarking Language Models on Falsifiable Research Ideation

## 一句话总结
提出了 Lit2Test 基准，通过“六字段合同”强制语言模型将研究设想转化为可证伪的完整测试方案，并采用双向盲评与顺序折叠协议，在剥离位置偏差与表面流畅性干扰后，严格量化前沿模型在可检验科研构思上的能力差异。

## 研究问题与动机
- **核心问题**：如何设计一种评测机制，使模型提出的科研设想在生成时即预-commit 会导致其被推翻的关键观测结果，从而将“想法质量”从主观审美判断转化为可由实验裁决的客观属性。
- **现有方法不足一（自由形式评判）**：直接让 LLM 或人类打分哪篇想法更好，结果严重受文风、表达位置（position bias）影响，且“idea quality”本身缺乏给裁判提供决策规则的公共结构。
- **现有方法不足二（事后答案键打分）**：用已发表的后续论文作为参考标准衡量设想准确度，仅奖励沿单一实现路径的恢复，惩罚合理但走不同方向的研究路线，且极易受知识截止日期污染（knowledge-cutof contamination）。
- **动机**：引入可证伪性作为统一决策基准，使评测仪器自身能经审计，真正衡量模型从真实文献张力中提炼出“最小可执行测试”的能力，而非仅评估文本流畅度或轨迹恢复率。

## 核心贡献（创新点）
- **六字段联合评估单元**：将研究构思形式化为 literature_gap、hypothesis、minimal_test、decisive_metric、supporting_result、falsifying_result 六个必填字段，缺失任一则丧失共同决策结构。与以往仅打分摘要或假设文本的工作本质不同，本文把完整的“最小可证伪测试合同”直接作为评判对象。
- **前瞻性无答案键构建管道**：基于 200 个真实论文邻域（800 篇独立文献）构建，全程不指定目标或后续论文作为答案键。与利用事后答案键或实时检索的方案不同，本文坚持“文献提供上下文、无标准答案”的前瞻性设计，彻底关闭数据泄露路径。
- **可审计的评测协议**：引入双向盲评、顺序折叠（fold）、Case-level bootstrap 置信区间与多类隐性控制项，显式隔离 250 个顺序敏感案例而非强行平均。与常规 LLM-as-judge 评测相比，本文将可靠性审计内嵌于测量流程本身。
- **严格的模型排序与机制诊断**：在 10,000 次 bootstrap 中稳定恢复四模型 Condorcet 序，并通过多维控制证明排序差异源于测试与指标设计质量，而非表面流畅度或格式渲染。

## 方法详解
- **六字段合同设计**：给定固定文献邻域 $c$，模型输出提案 $P = (\text{literature\_gap}, \text{hypothesis}, \text{minimal\_test}, \text{decisive\_metric}, \text{supporting\_result}, \text{falsifying\_result})$。其中 `literature_gap` 锚定文献张力，`hypothesis` 给出带条件与机制的方向性断言，`minimal_test` 要求最小可行实验设计（反宏大主义），`decisive_metric` 命名能裁决该机制的唯一测量，`supporting/falsifying_result` 共同构成双向决策规则。
- **顺序折叠（Order Folding）**：每对提案以 A/B 和 B/A 两种顺序进行盲评。折叠规则为：若两种顺序均指向同一胜者（考虑侧向偏差）则为 order-stable；若胜者翻转或至少一侧返回 tie，则为 order-sensitive。敏感案例单独保留报告，不进入排名聚合，防止位置偏差掩盖不可决比较。
- **聚合与不确定性估计**：order-stable 结果用于 Bradley–Terry 强度估计与 Condorcet 头对头分析；不确定性通过 case-level bootstrap（10,000 次重抽样规范对）计算。当出现完全分离（一方几乎全胜）时，优先报告序数结构（排名、Condorcet 关系）而非幅度值。
- **诊断控制体系**：包含维度分解审计（5 维度结构化评分对比整体判决）、隐藏真实 vs 朴素基线控制、同源渲染控制（分离实质与格式）、单字段明显缺陷操控检查、以及自然主义细微缺陷审计（配合风格匹配的 sham edit 排除表面改写干扰）。

## 实验与结果
- **数据集**：Lit2Test，200 个文献邻域，800 篇独立论文，4 个前沿模型（GPT-5.2, Claude Sonnet 4.6, GLM-5, DeepSeek-V3.2）各生成 1 份提案，共 1,200 个规范对与 2,400 次盲评。
- **评估基线**：Gemini 3.1 Pro (Preview) 为主评委，Doubao Seed 2.0 Pro 为独立二阶评委；同时混入关键词/模板朴素基线作为隐性锚点。
- **主要结果数字**：
  - 折叠后 950 个 order-stable 案例，250 个 order-sensitive 案例。
  - Bradley–Terry 中心化 log-ability：GPT-5.2 (1.26), Claude Sonnet 4.6 (0.74), GLM-5 (−0.73), DeepSeek-V3.2 (−1.28)；全序在 10,000 次 bootstrap 中 100% 恢复。
  - 二阶评委与主评委在 86.1% (1,635/1,900) 的 order-stable 比较上一致；折叠一致率 79.9%。
  - 人类校准（20 邻域，3 名标注员）在 39 个决定性稳定案例中与评委一致率达 87.2% (95% CI 76.9–97.4%)，人类 BT 排序 88.3% 的 bootstrap 样本与机器排序至多差一次翻转。
- **最强结果与提升幅度**：GPT-5.2 相对 DeepSeek-V3.2 的 Head-to-head 战绩为 180-4-16（稳定胜/负/敏感），优势幅度最大；排序驱动力主要来自 `minimal/feasibility` 与 `decisive_metric` 字段，`falsifiability` 充当准入门槛而非区分维度。

## 相关工作脉络
- **AI Idea Bench / IdeaBench**：侧重模型匹配或恢复已发表目标论文，使用事后答案键评分；本文完全摒弃答案键，转为评估前瞻性测试设计能力。
- **
