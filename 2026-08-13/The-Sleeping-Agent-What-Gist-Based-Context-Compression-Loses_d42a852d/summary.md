---
title: "The-Sleeping-Agent-What-Gist-Based-Context-Compression-Loses"
source: https://arxiv.org/pdf/2608.11775v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 02:39:18"
---

# 论文速读：The-Sleeping-Agent-What-Gist-Based-Context-Compression-Loses

## 一句话总结
本文以生物启发的 SWC（Salience-Weighted Consolidation）框架为诊断探针，系统揭示通用 gist 摘要压缩在长程对话代理中会**选择性丢失时间锚点**；仅需在压缩提示词中增加一句强制保留时间表达的指令，即可使 Temporal 类问题准确率提升 +0.314，且几乎不影响 Multi-hop 与 Single-hop 性能。

## 研究问题与动机
- 长程 LLM 代理在上下文溢出时普遍采用反应式摘要压缩，但压缩对不同类型记忆检索的影响机制缺乏系统性诊断。
- 现有结构化记忆系统（LightMem、Mem0、MemMachine 等）虽在 LoCoMo 上取得高分，但架构复杂，难以隔离“哪些组件真正起作用”。
- 标准 gist 摘要提示词仅要求保留事实、因果与决策，未显式保护日期/时间，导致代理处理时序依赖问题时出现系统性失效。
- 仅报告聚合准确率会掩盖特定任务类型的性能塌陷，亟需细粒度评估与内容保留剖面分析。

## 核心贡献（创新点）
1. **提出 SWC 作为可控诊断探针**：借鉴睡眠记忆巩固理论设计双阶段流程，以标准化方式剥离 gist 压缩效应，定位失败模式而非直接竞争性能。
2. **揭示“时间锚点选择性丢失”机制**：通过保真度量化证明通用摘要在保留实体与事件结构的同时，会系统性丢弃时间表达（仅 3.05% 保留）。
3. **提供精准的单句提示词修复**：强制保留具体日期、时长等表达后，时间保留率跃升至 62.39%（≈20 倍），精准恢复 Temporal 任务性能而不拖累其他类型。
4. **建立“分类评估 + 保真度剖面”诊断范式**：证明聚合分数可能误导判断，建议记忆系统评估必须配套类别级性能分解与关键信息保留率报告。

## 方法详解
- **设计原则**：前向调度（idle 窗口触发而非容量硬限）、显著性加权保留（chunks 按重要性分层）、分阶段压缩（先评分后摘要）。
- **Stage 1：显著性评分**：每个对话 chunk 综合三个信号计算得分（权重 Downstream similarity 0.4 / Recency 0.3 / Information density 0.3），相似性由 `all-MiniLM-L6-v2` 本地计算。按阈值划分：高优先级（≥0.6，原文保留）、中优先级（0.3–0.6，gist 摘要）、低优先级（<0.3，丢弃）。最近两个 session 与 session 0 始终归入高优先级；超 4,000 token 预算时按显著性从低到高裁剪。
- **Stage 2：Gist 摘要**：
  - **SWC-Full**：用 Claude Haiku（temperature 0）生成 3–5 句摘要，提示词保留事实、因果、决策与计划，丢弃闲聊/重复/纯情绪，**未要求保留时间**。
  - **SWC-Temporal**：与 SWC-Full 完全相同，仅在提示词首句增加：`"You MUST preserve verbatim: all specific dates, times, durations, ages, and temporal expressions (e.g. '7 May', 'last Tuesday', 'at 9pm')."，并配置独立 REM cache。
- **调度与实现**：每 3 个 session 触发一次压缩；显著性评分可在普通 CPU 离线运行（单轮对话耗时 30 min–2.5 h）。

## 实验与结果
- **数据集与设置**：LoCoMo 文本子集，10 段多轮对话（平均 ~600 turn / 16k tokens）。共 1,935 个纯文本 QA，主分析集为 Category 1–4 的 1,501 题；Category 5（对抗题）因评分规则奖励“删除内容”而被排除。QA 生成：Claude Sonnet 4.6；压缩与 Judge：Claude Haiku（temperature 0）。
- **基线条件**：Truncation、Sliding window、SWC-Full、SWC-Temporal、Full context（仅作上界，仅评测 conv 0–1）。
- **主要结果（Table 1，1,501 QA）**：
  - SWC-Temporal: **0.468** [0.444, 0.495]
  - SWC-Full: **0.379** [0.354, 0.404]
  - Sliding window: **0.238** [0.216, 0.260]
  - Truncation: **0.171** [0.153, 0.190]
- **分类结果（Table 2）**：SWC-Temporal 相较 SWC-Full 在 Category 2（Temporal）提升 **+0.314 [0.254, 0.375]**，方向在全部 10 段对话中一致为正；
