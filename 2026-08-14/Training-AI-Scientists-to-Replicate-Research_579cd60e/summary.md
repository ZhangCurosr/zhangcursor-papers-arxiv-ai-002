---
title: "Training-AI-Scientists-to-Replicate-Research"
source: https://arxiv.org/pdf/2608.13331v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 04:27:39"
---

# 论文速读：Training-AI-Scientists-to-Replicate-Research

## 一句话总结
本文构建了自动化的 Replica 论文复现任务空间，并引入基于逐任务评分量表的 LLM 裁判与回合级信用分配机制，成功对 27B 参数的 Faraday 智能体进行长程 GRPO 微调；Faraday 以 Coding Agent as a Tool 范式调用 Codex，在多项 ML 与分布外 AI-for-science 复现任务上显著超越 Claude Opus 4.8 与 GPT-5.5，展现出更接近人类科研严谨性的开放探索能力。

## 研究问题与动机
- **科学复现的天然“欠定义”属性**：论文是对研究过程的高度压缩，复现必须推断缺失细节、动态调整实验规模，属于开放式假设驱动探索，而非封闭问题的标准求解。
- **现有 AI 智能体的训练错配**：主流 agent 多在具有明确目标函数与可验证奖励的封闭环境中训练，面对需自行设计实验、评估中间结果的长周期科研任务时缺乏适应力。
- **自动化研究系统缺乏可放大的奖励信号**：autoresearch、AlphaEvolve 等系统依赖可爬山式的标量 reward，而通用论文复现难以事前精确定义单一验证函数；同时人工构建高质量复现基准成本极高。
- **机器学习领域的复现危机**：ML 论文复现失败率持续走高，亟需可扩展的 AI 方案来验证、巩固已有科研结论并缓解人工复核压力。

## 核心贡献（创新点）
1. **Replica 自动化任务空间**：提出三阶段视觉-语言管道（扫描-定位-不可逆红涂），从 100 篇跨年代经典论文中自动生成 310 个图文复现任务，实现任务规模的指数级扩展。*与以往依赖人工筛选或局部代码 mask 的基准相比，该方法在保持高质量的同时彻底摆脱了对人类标注的依赖。*
2. **Rubric-based 低噪声裁判**：设计五维度逐任务评分量表（视觉还原、主张支持、机制实现、预算利用、科研诚信），并通过多采样聚合与人类专家对齐验证，有效解决了长周期非可验证任务的奖励稀疏与高方差问题。*相较于让 LLM 直接做主观 peer-review，该裁判聚焦可观测的代码行为与实验设计，信噪比显著提升。*
3. **Turn-level Credit Assignment 稳定长程 GRPO**：在 GRPO 训练中引入裁判生成的回合级信用权重，按 Token 数归一化后缩放 per-token advantage，在不改变整体 reward 尺度的前提下实现精细的过程激励。*与均匀分配 credit 或仅依赖最终结果打分的方法相比，该设计保留了关键科学决策信号，大幅缓解了长 horizon 训练崩溃。*
4. **CAT 范式与 27B AI Scientist 训练**：提出 Coding Agent as a Tool 架构，Faraday 专注科研规划与假设迭代，将代码实现委托给 Codex，实现小参数模型驾驭大算力工程工具的能力跃迁。*区别于将研究流程硬编码为多智能体工作流或复杂 harness 的方案，该范式以极简接口实现了更强的可迁移性与成本控制。*

## 方法详解
- **Replica 任务生成流水线**：使用 Gemini 2.5 Pro 对论文进行扫描（提取所有正文结果图与 caption）、定位（LLM-verifier 修复循环确定边界框）、红涂（从 PDF 中不可逆移除图）。每个红涂图构成三元组任务（caption, gold plot, redacted paper），共 242 训练任务、68 测试任务，覆盖 1990–2026 年 ML 与 AI-for-science 论文。
- **Rubric Judge 设计**：以 Claude Opus 4.7 基于简短元提示自动生成
