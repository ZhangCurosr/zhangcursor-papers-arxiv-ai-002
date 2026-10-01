---
title: "SE-GoS-Self-Evolving-Graph-of-Skills-for-Skill-Library-at-Sc"
source: https://arxiv.org/pdf/2609.08228v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 03:27:28"
field: "大语言模型智能体技能检索"
keywords: ["skill retrieval", "graph-of-skills", "self-evolving", "LLM agents", "PPR", "training-free", "SkillsBench"]
innovations: ["提出SE-GoS无训练演化框架，从执行轨迹自动更新技能图谱的拓扑、边权重和节点描述", "用确定性执行计数替代LLM关系验证，实现无LLM prior的结构化检索", "证明单轮离线演化可提升可迁移结构，多轮重复导致过拟合"]
benchmarks: ["SkillsBench", "ALFWorld"]
---

# 论文速读：SE-GoS-Self-Evolving-Graph-of-Skills-for-Skill-Library-at-Sc

## 一句话总结
论文提出 SE-GoS（Self-Evolving Graph-of-Skills），一种无需训练的框架，通过将执行轨迹转化为拓扑更新、边权重强化和描述优化三类结构变化，使技能检索图谱从静态变为可进化，在 SkillsBench 上将任务奖励从 52.4% 提升至 59.4%，同时将输入 token 减少约三分之一。

## 研究问题与动机
1. **技能检索成为大规模 LLM agent 的瓶颈**：当技能库从数十增长到数千时，如何检索有限且相关的技能子集成为关键障碍。
2. **现有方法存在结构性缺陷**：全量加载 token 成本线性增长且模型易淹没关键技能；向量检索忽略功能前置依赖，导致"前提缺口"（prerequisite gap）问题。
3. **静态图谱无法自我修正**：GoS 等图结构检索方法一旦构建即固定不变，无法从执行反馈中修复错误依赖、缺失关系或描述匹配失败等问题。
4. **执行轨迹是免费信号**：每次任务尝试已记录检索结果、实际使用技能和任务成功状态，但未被系统性地蒸馏为图谱改进。

## 核心贡献（创新点）
1. **揭示静态图检索瓶颈并提取修复信号**：指出 GoS 的静态图局限，证明修复信号已存在于每次试验的执行轨迹中——检索了哪些技能、使用了哪些、是否成功。
2. **提出三类互补的无训练更新机制**：拓扑诱导与剪枝（workflow/dependency/avoid 边）、边权重强化（Hebbian 式增权）、单轮文本梯度节点描述优化，三者作用于离散边集、连续权重和节点文本三个正交维度。
3. **解释为何只需编辑图谱即可改变检索行为**：PPR 算子直接读取边权重，重构图谱即改变检索路径，无需修改检索代码、技能内容或训练任何参数。
4. **建立三维部署评估体系**：完整基准对比（flat/vector/static/self-evolving）、多轮演化曲线（是否应重复更新）、独立测试集验证（泛化能力）。

## 方法详解

### 整体架构
SE-GoS 保持 GoS 检索流程完全不变，仅更新底层图谱 $G = (V, E, w, \phi)$，演化过程分两阶段离线执行：Phase A 提取执行信号，Phase B 按依赖顺序应用三类更新。

### 经验信号收集
每个任务试验 $t$ 产生轨迹 $\mathcal{T}_t = (q_t, r_t, B_t, U_t, \text{tokens}_t)$，其中 $r_t \in [0,1]$ 为验证器奖励，$B_t$ 为检索集合，$U_t$ 为实际使用的技能集（通过工具调用中的源码路径匹配提取）。

### 1. 拓扑更新（Topology Update）
- **工作流边（workflow）**：成功试验中检索种子 $u$ 且使用技能 $v$（$u \neq v$）时，增加共现计数 $c_{uv}$，权重公式：$w_{wf}(u \to v) = \min(0.9, 0.6 + 0.05(c_{uv}-1))$。
- **依赖边（dependency）**：成功试验中 $u$ 先于 $v$ 使用且 I/O schema 重叠度 $\geq \zeta=0.6$ 时添加。
- **避免边（avoid）**：失败试验中至少 $\theta_{avoid}=2$ 次共现且从未在成功试验中出现时添加，权重为 0，在 bundle 组成时丢弃已选 avoid 伙伴的候选。
- **软剪枝（soft pruning）**：被检索 $\theta_{obs}$ 次以上但从未被使用的技能 $v$，其所有入边权重乘以 0.5。

### 2. 边权重更新（Edge Update）
对每个正奖励试验 $t$ 和每个实际使用技能 $v \in U_t$，强化所有指向 $v$ 的入边：
$$w^{(t+1)}(u \to v) = w^{(t)}(u \to v) + \eta \cdot r_t$$
其中 $\eta = 0.1$ 为强化率。置信加权插值作为多轮部署的安全保护：
$$w(e) = (1 - \lambda_c(e)) w_0(e) + \lambda_c(e) \tilde{w}(e), \quad \lambda_c(e) = \frac{n_e}{n_e + n_0}$$

### 3. 节点更新（Node Update）
针对成功试用但排名低于 $n_{rank}=3$ 的技能，采用单轮 ProTeGi 式文本梯度优化：
1. **梯度**：LLM critic 分析查询 $q$、当前描述 $d_s$ 和检索证据，输出自然语言批评 $g_q$。
2. **编辑**：LLM editor 沿相反方向修订描述，最多添加 50 tokens。
3. **探索**：paraphrase 模型生成 $p=2$ 个变体。
4. **选择**：离线重跑检索，选择能使 $s$ 进入 top-$n_{rank}$ 的描述：
$$d_s^* = \arg\max_{d \in \text{Cands}(s)} \sum_{q \in Q_s} E(q, s; d)$$
附带 no-eviction guard：拒绝将实际使用技能从 top-K 中排除的改写。

### 演化后检索
推理时沿用原 GoS 流程，仅读取演化后的图：
$$\rho_i(q) = \mathbf{s}_i^*(q; G^{(T)}, w^{(T)}, d^*) + \mu m_i(q; d^*)$$

## 实验与结果

### 实验设置
- **数据集**：SkillsBench（87 任务、1000 技能库）、ALFWorld dev split
- **模型**：deepseek-v4-flash-0731（主实验）、minimax-m2.7、gpt-5.2-codex
- **基线**：Vanilla（全量加载）、Vector（向量检索）、GoS（静态图）、SkillDAG（自演化对比）
- **协议**：单轮演化（50 任务训练、37 任务测试的 held-out 设置）

### 主要结果（deepseek-v4-flash-0731 on SkillsBench）
| 方法 | Reward (R↑) | Tokens (T↓) | Runtime (S↓) |
|------|-------------|-------------|--------------|
| Vanilla | 46.2% | 5.06M | 771.4s |
| Vector | 38.7% | 3.11M | 790.7s |
| GoS (static) | 52.4% | 3.67M | 843.8s |
| SkillDAG | 55.3% | 3.62M | 862.9s |
| **SE-GoS** | **59.4%** | **3.45M** | 883.7s |

- **提升幅度**：相比静态 GoS +7.0 个百分点（52.4% → 59.4%），超越 SkillDAG 4.1 个点
- **效率优势**：token 消耗比 Vanilla 低 32%，接近向量检索的压缩水平
- **跨模型验证**：minimax-m2.7（18.7% → 28.5%，+9.8pp）、gpt-5.2-codex（34.4% → 38.1%，+3.7pp）

### Held-out 泛化验证
在 50 训练/37 测试划分上，SE-GoS 从 52.9% 提升至 58.3%（+5.4pp），证明增益来自可迁移结构而非记忆化。

### 多轮演化曲线
- Round 1：59.4%（最佳）
- Round 2：59.8%（ plateau ）
- Round 3：54.0%（过拟合下降）
- 边缘数从 863 → 990 → 1375 → 1502

### 组件消融（$2^3$ 因子实验）
- 单独拓扑：54.1%（+1.7）
- 单独边权重：57.3%（+4.9）← 主效应最大
- 单独节点：53.6%（+1.2）
- 全组合：59.4%

## 相关工作脉络
1. **GoS（Liu et al., 2026）**：提出有类型技能图的依赖感知检索，但图谱静态不变；SE-GoS 在其基础上增加执行驱动的演化能力。
2. **SkillDAG（Zhao et al., 2026）**：同为 self-evolving + training-free，但依赖 LLM pair classifier 和 HyDE 嵌入服务进行冷启动；SE-GoS 用确定性 token-overlap 图替代，结构来自执行计数而非模型判断。
3. **SkillGraph（Li et al., 2026b）**：通过强化学习训练技能图，需模型参数更新；SE-GoS 完全不训练参数。
4. **HippoRAG（Jiménez Gutiérrez et al., 2024）**：基于 PPR 的 RAG 检索，但图边来自模型提取而非执行；SE-GoS 同样使用 PPR 但图结构由执行轨迹构建。
5. **Voyager（Wang et al., 2023）** 及后续工作：通过探索累积技能库、重写或增强技能；SE-GoS 不动技能内容，仅重构检索图谱。
6. **ProTeGi（Pryzant et al., 2023）**：自动提示优化的梯度下降方法；SE-GoS 借鉴其文本梯度思想，适配为单轮、技能局部的描述优化。

## 局限性与未来方向
1. **单轮报道为主**：多轮演化在 round 2 后 plateau、round 3 过拟合，未深入探索窗口化部署策略。
2. **使用集提取局限性**：间接技能使用可能导致证据低估（如通过脚本间接调用的技能）。
3. **评估范围受限**：仅在 1000 技能规模、87 任务的 SkillsBench 和 ALFWorld dev split 上验证，未覆盖 200/500/2000 规模。
4. **冷启动与收敛性**：置信加权插值保护低证据边，但长时演化收敛性未充分表征。
5. **关系边界限制**：避免 induction 的 alternative 关系（可替代性），因观测轨迹无法验证反事实等价性。

## 研究启发与可借鉴点
1. **结构化检索+执行反馈的解耦设计**：保持检索管道不变，仅更新底层图谱，使演化过程可审计、可回滚，适合生产部署。
2. **三类正交更新的分工原则**：拓扑（离散结构）、权重（连续概率）、描述（文本语义）分别作用于图谱的不同层面，互不干扰且相互增强。
3. **无 LLM prior 的结构推断**：用确定性计数规则替代 LLM 关系验证，降低部署成本且使增益可归因于演化而非先验。
4. **离线单轮演化优于迭代训练**：多轮实验显示反复更新导致过拟合，单轮一次性更新是更稳健的部署策略。
5. **信噪分离的实验设计**：held-out 设置分离泛化与记忆化，$2^3$ 因子消融隔离各组件贡献，噪声带校准统计显著性。

## 关键术语表
**Graph-of-Skills (GoS)**：将技能库建模为有类型有向图，通过 reverse-aware PPR 扩散检索依赖感知的技能束。
**Personalized PageRank (PPR)**：带重启概率的图随机游走算法，用于从种子节点向结构重要的前置技能传播相关性。
**Workflow edge**：从执行轨迹中诱导的工作流依赖边，表示检索种子与实际使用技能的共现模式。
**Avoid edge**：从失败共现中诱导的排除边，标记共同加载会降低任务表现的技能对。
**No-eviction guard**：节点更新的保护机制，拒绝将实际使用技能从检索 top-K 中排除的描述改写。
**Confidence-weighted interpolation**：多轮演化中的安全机制，按证据数量加权混合演化权重与静态先验。
**Lexical-only seeding**：仅使用词法 token 重叠而非向量嵌入进行种子检索，消除对 embedding service 的依赖。

## 可复现要素
- **数据集**：SkillsBench（1000 技能、87 任务）、ALFWorld dev split
- **代码开源**：论文未明确声明代码仓库链接
- **权重开源**：不涉及模型训练，无权重需要开源
- **关键超参**：$\alpha=0.2$（PPR restart）、$\eta=0.1$（reinforcement rate）、$\zeta=0.6$（schema threshold）、$\theta_{obs}=2$、$\theta_{avoid}=2$、$n_{rank}=3$、$p=2$（paraphrase count）、edit ≤ 50 tokens
