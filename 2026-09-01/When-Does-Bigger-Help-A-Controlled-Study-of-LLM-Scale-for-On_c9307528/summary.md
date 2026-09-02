---
title: "When-Does-Bigger-Help-A-Controlled-Study-of-LLM-Scale-for-On"
source: https://arxiv.org/pdf/2608.31118v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 06:43:08"
---

# 论文速读：When-Does-Bigger-Help-A-Controlled-Study-of-LLM-Scale-for-On

## 一句话总结
本文在严格固定的 OntoLearner RAG 评测管道下，系统扫描了 Qwen3.5/3.6 与 GPT 系列中 13 个不同参数量级（0.8B 至 122B）的模型，揭示参数规模对 ontology learning (OL) 的性能影响呈显著任务与领域依赖：规模化主要提升精确率而非召回率，中等规模密集模型（27B）在术语分类上可超越更大规模的 MoE 稀疏模型，但非 taxonomy 关系抽取在抽象领域仍面临几乎无法靠堆参数突破的瓶颈。

## 研究问题与动机
- 现有 LLM 辅助本体学习研究多跨家族、跨架构对比，参数量始终与模型家族、训练策略、领域适配耦合，缺乏在固定协议下将“规模”作为独立变量的族内连续扫描。
- 基础模型领域的 scaling law 多聚焦预训练 loss 或通用基准，下游专项知识工程任务（如 OL）的规模响应曲线尚未被实证刻画，任务复杂度与领域抽象度如何调节规模红利仍不明确。
- 开源权重量化模型与闭源前沿模型在本体学习中的实际边界不清，架构类型（Dense vs MoE）与版本对齐迭代是否会引发非线性甚至倒退的性能变化，缺乏可控实验证据。
- 科学数据（生物医学、材料科学）的异构性与概念抽象度差异，如何影响 LLM 规模收益的获取程度，尚未在统一评测体系下得到量化回答。

## 核心贡献（创新点）
- **族内参数尺度控制扫描**：在 OntoLearner 固定 RAG 管道下首次实现 Qwen3.5/3.6 与 GPT 家族 0.8B→122B 的连续尺度评测，有效分离了参数规模与架构/家族/对齐策略的耦合干扰。
- **任务-领域双维度的规模敏感性图谱**：揭示 OL 三任务（Term Typing ≺ Taxonomy Discovery ≺ Non-Taxonomic RE）的规模收益呈非线性阶梯分布，并指出高抽象度领域（如 MDS）对参数扩展呈现强抵抗力。
- **密集 vs MoE 的 task-dependent 逆向发现**：证明中等规模密集模型（Qwen3.5-27B）在术语分类任务上显著优于规模大 4 倍的 MoE 稀疏模型（122B-A10B），而层级推理任务则反之，打破“越大越好”的直觉。
- **版本对齐惩罚的实证记录**：在同一尺度下首次观察到 Qwen3.6-27B 相比 3.5-27B 的意外性能回退，以及 GPT-5.6 系列因过度保守对齐导致的极端低召回现象，提示商业化迭代可能牺牲特定下游能力。

## 方法详解
- **评估框架**：采用 OntoLearner 检索增强生成流水线，将上下文获取与 LLM 生成解耦，所有变体在完全相同的嵌入模型、检索配置、提示模板与解码条件下运行，确保观测差异仅源于模型规模与架构。
- **语义向量索引**：使用冻结的特征提取器 $E(\cdot)$（固定为 `Qwen3-Embedding-4B`）将目标本体实体映射至共享高维向量空间：$\mathbf{v}_i = E(\text{text}(c_i)) \in \mathbb{R}^d$，构建基于余弦相似度的稠密向量索引。
- **相似度检索**：推理时将候选查询 $q$ 编码为 $\mathbf{q} = E(\text{text}(q))$，计算 $\text{Sim}(\mathbf{q}, \mathbf{v}_i) = \frac{\mathbf{q} \cdot \mathbf{v}_i}{\|\mathbf{q}\|_2 \|\mathbf{v}_i\|_2}$，固定 top-$k
