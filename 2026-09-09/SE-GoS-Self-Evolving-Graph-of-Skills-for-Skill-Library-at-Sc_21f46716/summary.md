---
title: "SE-GoS-Self-Evolving-Graph-of-Skills-for-Skill-Library-at-Sc"
source: https://arxiv.org/pdf/2609.08228v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 03:27:31"
field: "LLM Agent 技能检索"
keywords: ["Skill Retrieval", "Graph-of-Skills", "Self-Evolving", "Personalized PageRank", "Tool-Augmented LLM", "Training-Free", "Execution Trace Distillation"]
innovations: ["从执行轨迹蒸馏拓扑/权重/描述三维度更新，无需训练参数或修改检索代码", "用确定语义重叠图替代 LLM 先验，以执行证据（共现/I-O schema/失败模式）构建 skill 依赖结构", "置信度加权插值保护低证据边，支持多轮部署下的稳定收敛"]
benchmarks: ["SkillsBench", "ALFWorld"]
---

# 论文速读：SE-GoS: Self-Evolving Graph-of-Skills for Skill Library at Scale

## 一句话总结
SE-GoS 是一种训练无关（training-free）的离线进化框架，利用 LLM agent 的历史执行轨迹对 Skill Graph 进行拓扑更新、边权重强化和描述文本优化，在不改动检索代码、不训练任何参数、不修改技能内容的前提下，将静态技能检索图转变为可自我进化的检索基础设施。

## 研究问题与动机
- **技能库扩展后检索成为瓶颈**：随着技能库从几十扩展到上千条目，Agent 的核心挑战从"是否使用技能"转移到"如何有效检索相关技能子集"，全量加载 token 成本线性增长且关键技能被淹没在超长上下文中。
- **向量检索忽略功能依赖链**：基于嵌入相似度检索常选到高层求解器而遗漏其低层解析/前置工具，存在"prerequisite gap"（功能前提缺失）。
- **GoS 等图结构检索是静态的**：已有 Graph-of-Skills (GoS) 通过依赖感知图结构实现可扩展检索，但图一旦构建就不再更新，无法从执行反馈中自我修正错误依赖或丢失的有用关系。
- **执行轨迹中的免费信号未被系统利用**：每次 trial 已记录"哪些技能被检索、哪些实际被使用、任务是否成功"，但这些信号从未被系统性蒸馏为改进检索图的依据。

## 核心贡献（创新点）
1. **隔离并解决静态图瓶颈**：首次明确指出 GoS 类结构检索的静态性限制，并证明所需修复信号已内生于每次执行轨迹中（检索集合、使用集合、任务奖励），无需额外数据采集。
2. **提出三维度训练无关进化机制**：拓扑诱导与剪枝（workflow/dependency/avoid 边）、边权重强化（Hebbian 风格）、单次文本梯度节点描述优化，三者作用于不同可更新轴（离散边集、连续权重、节点文本），互不覆盖。
3. **揭示"改图即改检索"的理论原理**：由于 Personalized PageRank 直接读取边权重，重连/重加权图即可改变检索行为，而检索流水线保持为固定消费者，无需修改任何 SKILL.md 或检索代码。
4. **三轴系统刻画（全基准/多轮曲线/保留集）**：在 SkillsBench 上展示单轮进化增益、多轮迭代下的过拟合拐点（第 3 轮下降至 54.0%），以及在 disjoint 37 题保留集上的泛化提升（+5.4 分），证明进化图学到的是可迁移结构而非记忆。

## 方法详解
**冷启动图**：从确定性语义相似度图出发，基于 signature-token Jaccard 重叠构建 k=1 语义边（863 条），无需 LLM 关系验证器、无需 embedding 服务，与部署场景完全匹配。

**1. 拓扑更新（Topology Update）**
- **Workflow 边诱导**：对每次成功 trial，检索种子技能 u 与实际使用技能 v（u≠v）共现一次则计数 +1；达到阈值后添加有向 workflow 边，权重公式：`w_wf(u→v) = min(0.9, 0.6 + 0.05 × (c_uv − 1))`。
- **Dependency 边认证**：当成功 trial 中 u 先于 v 被使用，且 I/O schema 重叠度 ≥ ζ（ζ=0.6，与 GoS 离线规则相同），添加 dependency 边。
- **Avoid 边**：在失败 trial 中至少 θ_avoid=2 次共现且从不在成功 trial 中共现的技能对，添加 weight=0 的 avoid 边（传播算子不可见，仅在 bundle 组装时用于排除冲突共加载）。
- **软剪枝**：若技能 v 作为边头被检索 ≥ θ_obs=2 次但从未被使用，将其所有 incoming 语义边权重 ×0.5，低于 floor 的边直接删除。

**2. 边权重更新（Edge Update）**
- 对奖励 r_t > 0 的 trial，对每条指向实际使用技能 v 的边 (u→v) 施加 Hebbian 式增量：`w^(t+1)(u→v) = w^(t)(u→v) + η · r_t`，η=0.1 为强化率。
- 置信度加权插值（多轮/冷启动保护）：`w(e) = (1−λ_c)·w_0(e) + λ_c·ẇ(e)`，其中 λ_c = n_e/(n_e+n_0)，证据少时偏向先验。

**3. 节点描述更新（Node Update）**
- 针对在失败 trial 中被使用但排名低于 n_rank=3 的技能，借鉴 ProTeGi 文本梯度下降做单次局部优化：
  1. **梯度**：LLM critic 输出描述 d_s 为何未能在查询 q 下排入 top-n_rank 的自然语言批评 g_q；
  2. **编辑**：LLM editor 沿 g_q 反方向修订 d_s，最多追加 ≤50 token；
  3. **Monte-Carlo 探索**：生成 p=2 个 paraphrase 变体；
  4. **离线选择**：用 `E(q, s; d)`（离线重跑检索计算排名）评估每个候选，选取使累积排名最优者 `d_s*`；配合 no-eviction guard 防止改写后把实际使用技能踢出 top-K。

**推理阶段**：完全不变，仅读取演化后的图 G^(T)、权重 w^(T)、描述 d*，复用 GoS 原有三步检索流程（lexical seeding → reverse-aware PPR → budgeted reranking/hydration）。

## 实验与结果
**数据集**：SkillsBench（1,000 技能、87 任务，v1.1 版本 byte-identical），ALFWorld dev split。

**基线**：Vanilla（全量加载）、Vector 检索、静态 GoS、SkillDAG（在相同协议下重新测量）、SE-GoS。

**主结果（deepseek-v4-flash-0731 / SkillsBench）**：
| 方法 | Reward (R%) ↑ | Tokens (T M) ↓ | Runtime (s) ↓ |
|------|-------------|---------------|---------------|
| Vanilla | 46.2 | 5.06 | 771.4 |
| Vector | 38.7 | 3.11 | 790.7 |
| GoS (静态) | 52.4 | 3.67 | 843.8 |
| SkillDAG | 55.3 | 3.62 | 862.9 |
| **SE-GoS** | **59.4** | **3.45** | 883.7 |

- SE-GoS 相对静态 GoS **+7.0 pp**，相对全量加载节省约 **32% tokens**；
- **保留集泛化**（50 训练/37 测试，disjoint）：SE-GoS 58.3% vs GoS 52.9%，**+5.4 pp**，证明学到的是可迁移结构。
- **多轮曲线**：Round 1 → 59.4%，Round 2 → 59.8%（持平），Round 3 → 54.0%（过拟合回落），说明单轮部署最优。
- **消融**：边权重贡献最大（+4.9），拓扑（+1.7）、节点（+1.2）次之；Edge+Node 组合达 59.1%。
- 跨模型（minimax-m2.7、gpt-5.2-codex）均呈现一致趋势，SE-GoS 均为最优。

## 相关工作脉络
1. **GoS (Liu et al., 2026)**：SE-GoS 的直接基础，提供依赖感知图检索；GoS 的 workflow/semantic/alternative 关系由 LLM 关系验证器离线构建，SE-GoS 用执行轨迹替代这一 LLM 先验，实现从静态到动态的跃迁。
2. **SkillDAG (Zhao et al., 2026)**：同为 self-evolving + training-free，但冷启动依赖 ~200 LLM 调用的 pair classifier + embedding service，且在线编辑含自然语言 justification；SE-GoS 冷启动为确定性 token-overlap 图，结构完全来自执行计数，无 LLM 先验。
3. **HippoRAG / ToolRerank / CRAFT / ControlLLM**：均基于 PPR 类图检索机制，但图边由模型抽取而非执行见证，且不随使用进化；SE-GoS 将图边来源从模型判断切换为执行证据。
4. **Voyager (Wang et al., 2023) 及后续**：通过执行轨迹累积/合成新技能，改变 Agent 知识；SE-GoS 不改变技能库内容，只重构检索 traversed 的图结构。
5. **ProTeGi (Pryzant et al., 2023) / TextGrad**：文本梯度优化方法论来源；SE-GoS 将其裁剪为单次、skill-local 的离线变体，适配技能描述优化场景。
6. **SkillGraph-RL (Li et al., 2026b)**：训练型技能图进化（RL 训练），SE-GoS 与之形成对比：完全无需参数更新，适合部署资源受限场景。

## 局限性与未来方向
- **单轮主结果**：多轮迭代在第 3 轮出现回退（54.0%），论文建议单轮部署，但未充分刻画窗口化（windowed）部署下的长期行为。
- **Used-set 提取可能低估**：间接技能使用（通过 shell/code-exec 间接引用）可能未被轨迹提取完全捕获，影响权重更新的证据量。
- **评估范围有限**：仅在 SkillsBench（1,000 技能）和 ALFWorld dev 上验证，未覆盖 200/500/2,000 等更多技能规模，也未在更多 backbone 上测试。
- **冷启动与收敛性**：置信度加权插值提供了保护机制，但长时程冷启动收敛性仅部分分析，缺少理论保证。

## 研究启发与可借鉴点
- **"检索基础设施作为经验吸收面"的设计哲学**：将 agent 的执行经验沉淀到检索图而非修改模型权重或技能内容，提供了一个低成本、可审计、可回滚的经验复用范式，可迁移至工具检索、文档检索等场景。
- **从执行轨迹蒸馏结构化信号的通用方法**：workflow co-occurrence → 工作流边；I/O schema + 顺序使用 → 依赖边；失败共现 → 避免边——这一套"轨迹→边类型"的映射规则可作为图结构抽取的通用设计模板。
- **单次文本梯度优化的 skill-local 变体**：借鉴 ProTeGi 但仅做单轮编辑+离线选择，避免了完整 APO 的计算开销，适合大规模技能库的离线维护阶段。
- **置信度加权插值作为安全网**：λ_c = n_e/(n_e+n_0) 的机制可在多轮部署中防止低证据边过早主导图结构，是经验驱动图更新的可复用正则化策略。
- **与团队方向的潜在结合**：若团队关注工具检索或 RAG 场景中的图结构检索，SE-GoS 的"离线进化+固定消费端"架构可无缝集成到现有 GoS/HippoRAG 管线中，且无需额外 embedding 服务，适合边缘/低资源部署。

## 关键术语表
- **SE-GoS**：Self-Evolving Graph-of-Skills，一种从执行轨迹离线进化技能检索图的训练无关框架。
- **Graph-of-Skills (GoS)**：基于 typed 有向图的结构化技能检索方法，通过 reverse-aware Personalized PageRank 传播依赖感知的相关性。
- **Personalized PageRank (PPR)**：以种子节点为起点的图随机游走算法，本文用于在技能依赖图上扩散检索相关性。
- **Workflow edge**：从检索种子到实际使用技能的有向边，由成功 trial 中的检索-使用共现计数诱导。
- **Avoid edge**：weight=0 的有向边，由失败 trial 中的高频共现诱导，在 bundle 组装阶段用于排除有害共加载。
- **Textual gradient descent (ProTeGi-style)**：用 LLM 输出自然语言梯度批评描述缺陷，再用 LLM 沿反方向编辑描述的优化范式。
- **Prerequisite gap**：向量检索遗漏低层前置工具的问题，即语义相似的高层求解器并非执行所需的完整技能集。
- **Confidence-weighted interpolation**：将进化权重与静态先验按证据计数加权混合的安全机制，防止低证据边主导检索。

## 可复现要素
- **数据集**：SkillsBench v1.1（commit d75b2187, 2026-06-14），1,000 技能、87 任务；论文声明 task packages 与上游 v1.1 byte-identical（instruction、verifier、task.toml 零差异）；代码/数据开源状态：论文未明确声明仓库链接，SkillsBench 本身为公开 benchmark（arXiv:2602.12670）。
- **代码/权重**：论文未提供开源代码仓库链接，附录含完整算法伪代码（Algorithm 1-3）和超参数表（Table 5），可实现性高。
- **关键超参**：ξ=0（纯 lexical seeding）、α=0.2（PPR restart）、η=0.1（reinforcement rate）、n_rank=3、θ_obs=2、θ_avoid=2、ζ=0.6、p=2（paraphrase count）、edit ≤50 tokens、no-eviction guard=on。
