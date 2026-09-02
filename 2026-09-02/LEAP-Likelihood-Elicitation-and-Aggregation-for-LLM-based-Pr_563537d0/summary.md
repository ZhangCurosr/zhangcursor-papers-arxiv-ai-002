---
title: "LEAP-Likelihood-Elicitation-and-Aggregation-for-LLM-based-Pr"
source: https://arxiv.org/pdf/2609.01337v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 22:15:22"
---

# 论文速读：LEAP-Likelihood-Elicitation-and-Aggregation-for-LLM-based-Pr

## 一句话总结
提出 LEAP 框架，将 LLM 基于证据的概率预测从“整体阅读生成点估计”的单体范式，重构为“逐条证据局部似然提取 + 显式贝叶斯确定性聚合”的两阶段流程，在固定相同证据的前提下显著提升预测准确率、校准度与证据级可审计性。

## 研究问题与动机
- **核心问题**：现有 LLM 预测系统在收集完证据后，通常让模型一次性阅读全部材料并输出单一答案（Monolithic Prediction），该设计难以量化单条证据对最终预测的实际影响。
- **长上下文缺陷**：将多条异构证据拼接成长文本输入 LLM，易触发“中间迷失”（lost in the middle）与事实幻觉，导致关键证据被忽略或曲解。
- **不确定性坍缩**：单体预测强制输出点估计或自由文本理由，会将竞争结果间的真实概率分布压缩为单一答案，掩盖认知不确定性。
- **缺乏可审计性**：用户无法脱离 LLM 生成的非结构化 rationale 独立复核预测逻辑，难以进行误差归因、证据替换或合规审查。

## 核心贡献（创新点）
- **预测阶段的形式化解耦**：将证据-grounded 预测严格拆分为收集与预测两阶段，固定证据集后隔离研究“如何将证据转化为概率预测”这一核心瓶颈。
- **局部似然提取 + 确定性聚合架构**：让 LLM 仅针对单条证据输出结构化似然参数，再由共轭贝叶斯模型完成闭式聚合，避免 LLM 直接承担全局合成与数值运算。
- **统一多类型目标接口**：通过 Gaussian / Categorical / Bernoulli 共轭对无缝支持连续型、单选型与多选性预测，三类目标共享同一证据解析与聚合管道。
- **可复现的证据级归因**：基于留一法后验更新计算每条保留证据的贡献 $\Delta_j$，使预测结果可在证据层面被人工复核、消融与调试。
- **插件化验证与广泛基准**：在自建 ReAct 循环与 4 个外部 Agent CLI 框架上验证，证明 LEAP 可作为即插即用概率技能无缝接入存量系统。

## 方法详解
- **概率模型基础**：假设目标 $\theta$ 与证据 $\mathcal{E}=\{e_1,\ldots,e_n\}$ 满足条件独立 $e_i \perp e_j \mid \theta$，保留集 $\mathcal{R}\subseteq\{1,\ldots,n\}$ 的后验为 $P(\theta\mid\mathcal{E})\propto P
