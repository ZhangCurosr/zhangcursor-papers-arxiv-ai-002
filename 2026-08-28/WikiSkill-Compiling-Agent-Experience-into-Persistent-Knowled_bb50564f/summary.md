---
title: "WikiSkill-Compiling-Agent-Experience-into-Persistent-Knowled"
source: https://arxiv.org/pdf/2608.27454v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 22:14:01"
field: "Agent 技能自动发现与进化"
keywords: ["agent skill evolution", "persistent knowledge base", "multi-agent systems", "in-context skill refinement", "cross-model transfer"]
innovations: ["引入持久 Wiki 层将执行经验编译为跨迭代累积的结构化知识，使技能更新建立在已验证的知识基础上", "证明技能发现能力与执行能力可分离：强模型演化出的技能可迁移至弱模型并使用，且有时优于自演化技能"]
benchmarks: ["LiveMathematicianBench", "SealQA", "SpreadsheetBench", "OfficeQA", "ALFWorld"]
---

# 论文速读：WikiSkill: Compiling Agent Experience into Persistent Knowledge for Skill Evolution

## 一句话总结
本文提出 WikiSkill，一个将智能体执行经验持续编译为结构化知识（Wiki）并与可执行技能协同进化的框架；在五个跨域基准和五个模型上均优于现有技能进化方法，且演化出的技能可在不同模型间有效迁移。

## 研究问题与动机
- **经验保存断层**：现有技能进化方法（EvoSkill、SkillOpt、Trace2Skill）在多轮迭代中依赖散落在历史痕迹里的洞察，未能将学习到的内容作为独立的知识表示进行积累与组织。
- **缺乏跨迭代记忆**：前作未维护"已学内容"的持续演化层，导致后续技能更新难以基于已被验证的综合知识，而是基于碎片化的优化历史。
- **模型能力与技能发现的耦合**：技能进化与模型扩展（scaling）的关系尚未系统研究；技能发现能力与技能执行能力的解耦也未明确。
- **可迁移性未知**：由某一模型演化出的技能能否跨模型/跨模型族转移、是否优于自演化技能，尚缺乏实证。

## 核心贡献（创新点）
1. **三层层架构 + 持久 Wiki 层**：将智能体工作区分割为 Raw（不可变轨迹）、Wiki（累积知识）、Skills（可执行程序）三层，使技能进化建立在跨迭代沉淀的知识之上；与前作"仅维护痕迹或历史反馈"的本质区别在于 Wiki 是独立于技能本身、持续增长的显式知识表示。
2. **Wiki 协调的四组件进化循环**：推理智能体（限 Wiki 访问）、Wiki 维护器（模式归并）、技能提议器（ReAct 自主诊断）、门控回滚机制；与前作的单阶段或固定轮次的 trace 分析不同，该方法在每轮迭代中都通过 Wiki 保留被拒绝提案的 diff，避免重蹈覆辙。
3. **系统化的跨模型技能迁移评估**：首次在大尺度上验证"技能发现 vs 技能执行是可分离能力"——例如 Qwen-3.5-9B 使用 Qwen-3.6-27B 的 SpreadSheet 技能达到 50.5%，显著高于自身演化的 33.6%。
4. **消融证实 Wiki 的关键作用**：移除持久 Wiki 后，Gemini-3.5-Flash 在 5 个基准上的平均性能从 63.7% 降至 48.7%（−15.0 pts），证明跨迭代知识累积是性能提升的核心来源。

## 方法详解
- **三层层架构（§3.1）**：
  - **Raw Layer（raw/）**：存储每一轮训练 rollout 产生的不可变执行轨迹 τ，包含推理过程、工具调用、工具返回和最终答案；供 Wiki Maintainer 和 Skill Proposer 读取。
  - **Wiki Layer（wiki/）**：持久化知识库，包含 pattern 目录（markdown 文件，记录失败模式/成功策略及可操作的 workaround）、`logs.md`（演化日志）、`skill-impact.md`（提案 diff + 验证分 + 接受/拒绝结果）和 `index.md`（模式索引）。跨迭代不重置，只增不改（接受/拒绝状态写入 impact tracker）。
  - **Skills Layer（skills/）**：每个技能目录含 `SKILL.md`（含 frontmatter 元数据、指令、适用条件）与 `PURPOSE.md`（回溯到 Wiki 中启发改动的 pattern）。
- **进化循环（§3.2）**：
  1. **Inference Agent**：使用当前活跃技能集 $S_{k-1}$ 在 $\mathcal{D}_{\mathrm{train}}$ 上做 rollout，技能全文注入系统提示；训练期间禁止访问 Wiki（消融 §5.1 表明允许访问会削弱技能质量）。
  2. **Wiki Maintainer**：对训练轨迹做分层采样（≤8 条：最多 5 条失败 + 最多 3 条成功，每条 ≤15k 字符），生成中间 Wiki 状态 $W'_k = \mathcal{M}_{\mathrm{WM}}(W_{k-1}, \mathcal{T}_{\mathrm{sample},k})$；通过根因分析提取模式并 patch 更新，同步修订 `index.md` 和追加 `logs.md`。
  3. **Skill Proposer**：以 ReAct 方式自主探索（$P_k = \mathcal{M}_{\mathrm{P}}(W'_k, S_{k-1}, \mathcal{T}_{\mathrm{train},k})$）；先读取 `wiki/index.md` 和 `skill-impact.md` 判断哪些方向已被尝试，再通过 `read_file` 按需检视模式页面和原始轨迹，最终产出对单个技能的原子性提案（新建或 patch）。
  4. **Gating & Rollback**：候选技能 $S'_k = \mathrm{Apply}(S_{k-1}, P_k)$ 在 $\mathcal{D}_{\mathrm{val}}$ 上评估；若 $\mathcal{R}(\mathcal{T}_{\mathrm{val},k}) > \mathcal{R}_{\mathrm{best}}$ 则接受，否则回滚 $S_k \leftarrow S_{k-1}$；但 **Wiki 永不回滚**，并通过 `Update(W'_k, P_k, \mathcal{R}, a_k)` 记录决策形成审计链。若 $\mathcal{R}_{\mathrm{best}} = 1.0$ 则提前终止。
- **超参与实现细节**：采样预算上限 8 条/轮（5 fail + 3 pass）；每条轨迹截断 15k 字符；每轮 batch 取全量 $B=N_{\mathrm{train}}$；$T_{\mathrm{ReAct}} \in [10, 20]$ 轮；统计显著性用 paired bootstrap（1000 次，p<0.05）。

## 实验与结果
- **数据集**：LiveMathematicianBench（数学推理）、SealQA（网页搜索 QA）、SpreadsheetBench（电子表格操作）、OfficeQA（长文档 QA）、ALFWorld（交互式具身任务）。
- **模型**：Qwen-3.5-4B / 9B / 3.6-27B、Gemma-4-31B-It、Gemini-3.5-Flash（共 5 个，开源 + 闭源）。
- **基线**：No-skill、Trace2Skill、EvoSkill、SkillOpt。
- **主要结果**（Table 1，五模型平均）：
  - WikiSkill 相对最强基线的平均提升：+3.3 / +5.1 / +10.0 / +5.8 / +12.0 pts（对应四档模型）。
  - Qwen 系列内 WikiSkill 提升随规模递增：4B +12.3、9B +17.5、27B +23.9 pts；SpreadSheet 增幅分别为 +6.5 / +9.3 / +40.9。
  - 小模型 + 技能可超越大模型无技能：Qwen-3.5-9B w/ WikiSkill 47.4% 平均 > Qwen-3.6-27B no-skill 39.4%。
  - Gemini-3.5-Flash 上 WikiSkill：LiveMath 33.0%→72.6%、SpreadSheet 50.5%→76.6%、OfficeQA 48.6%→60.7%。
- **跨模型迁移（Table 2）**：
  - Qwen-3.6-27B 的 SpreadSheet 技能使 Qwen-3.5-9B 从 24.3%→50.5%，显著高于自身 33.6%。
  - Qwen-3.5-4B→Gemma-4-31B 在 LiveMath 上实现 73.1%（自身 56.7%），小模型技能可反哺大模型。
  - 负面迁移也存在：Qwen-3.5-4B 的 SpreadSheet 技能使 Gemini-3.5-Flash 从 50.5% 跌至 18.1%，原因包括编码低层级 workaround 以及碎片化诊断引入多余工具调用耗尽交互预算。
- **消融（Table 3，Gemini-3.5-Flash）**：
  - 移除 Wiki 累积（Proposer 无 Wiki 访问）：Avg 48.7% → 63.7%（+15.0 pts）。
  - Inference Agent 训练期允许 Wiki 访问：Avg 63.7% → 60.9%（−2.8 pts），LiveMath 72.6%→64.8%。
- **统计**：三独立重复取平均；pairwise bootstrap 检验 p<0.05。

## 相关工作脉络
1. **Trace2Skill（Ni et al., 2026）**：基于并行 trace 分析 + 分层合并的三阶段 pipeline；本文区别于其"一次性合并所有 patch"而非跨迭代持续积累的 Wiki。
2. **EvoSkill（Alzubi et al., 2026）**：将技能进化建模为 frontier 搜索，仅向 Proposer 喂入失败轨迹 + 平坦反馈历史；WikiSkill 的 Wiki 保留了成功策略、被拒提案 diff 及其根因，提供更丰富的历史语境。
3. **SkillOpt（Yang et al., 2026）**：六阶段 ReflACT pipeline，多轮 reflect/aggregate/select/update/evaluate；WikiSkill 将其压缩为单轮 ReAct 提议 + Wiki 累积，API 复杂度由 $O(K_{\mathrm{opt}} \cdot N_{\mathrm{train}}/B)$ 降至 $O((1+T_{\mathrm{ReAct}}) \cdot N_{\mathrm{train}}/B)$ 且与 $N_{\mathrm{train}}$ 无关（全 batch 时恒为常数）。
4. **Agent Skill 基础概念**（Anthropic 2026; Li et al. 2026; Zhang et al. 2025; Chen et al. 2026a 等）：Skill 作为模块化文件系统目录承载领域流程知识；本文继承此形式但增加跨迭代 Wiki 层。
5. **Skill Retrieval 工作**（Cho et al. 2026; Shi et al. 2026; Su et al. 2026; Zheng et al. 2026）：关注从技能库中选择相关技能；本文明确剥离检索环节（直接全量注入），聚焦于"技能质量本身"的进化。
6. **Agent Harness 优化**（Lee et al. 2026; Lou et al. 2026; Lin et al. 2026）：优化 prompt/context/tools/memory/workflow 整体；与 WikiSkill 互补——后者在 harness 固定的前提下专门进化可复用程序性技能。

## 局限性与未来方向
- **未评估技能检索**：全量注入 active skills 到 prompt，回避了随技能数量增长带来的检索/触发问题，实际场景仍待解决。
- **严格门控排除中性提案**：只接受 $\Delta > 0$ 的变更，可能错过"短期持平但为后续迭代铺路"的中性提案；更灵活的接纳标准是未来方向。
- **Wiki 无自动修剪**：随迭代持续累积的 pattern 和日志未设计自动 prune 机制，长期运行可能出现知识膨胀。
- **缺少超长 horizon 任务**：基准覆盖长文档和多步工具调用，但不包含数百步环境动作或持续数小时的在线场景；单 rollout 内的持续适应方法亟待研究。

## 研究启发与可借鉴点
1. **"发现 vs 执行"的能力解耦**：本文证实"由强模型发现技能"与"由弱模型执行技能"可分离，且前者有时更优（小模型技能迁移到大模型）。可借鉴到本研究中的跨模型知识蒸馏或元技能库构建。
2. **三层层隔离设计**：Raw（只读）/ Wiki（累积只增）/ Skill（可变但受 gate 保护）的分工清晰，适合作为本团队后续在任意 skill-evolution pipeline 中引入"历史审计链"的基础范式。
3. **被拒提案 diff 的持久化**：通过 `skill-impact.md` 记录 rejected proposal 的完整 patch 与得分，使 Proposer 不再重复相同错误——这一低成本"负样本库"可直接复用到其他基于提议的自动优化框架。
4. **Inference Agent 训练期禁 Wiki 访问**：消融发现让推理者同时访问 Wiki 会"偷吃"知识从而削弱技能本身的进化驱动力；这一约束对其他"技能 + 知识"双轨系统具有通用参考价值。
5. **分层采样 + 截断策略**：每轮最多采 8 条轨迹（5 失败 + 3 成功，单条 ≤15k 字符）的预算控制方式，兼顾诊断覆盖面与上下文窗口限制，可迁移到相似场景的资源受限优化。

## 关键术语表
- **WikiSkill**：一种将智能体经验编译为持久结构化知识库（Wiki）并与可执行技能协同进化的框架。
- **Skill（智能体技能）**：模块化文件系统目录，包含 SKILL.md（指令与适用条件）和 PURPOSE.md（回溯到 Wiki 模式的来源说明），用于将领域知识封装为可复用代理资源。
- **Raw Layer**：存储每轮训练 rollout 的不可变执行轨迹（观察、行动、工具调用、答案），供维护器和提议器诊断使用。
- **Wiki Layer**：持久知识库，含 pattern 页面、演化日志（logs.md）和技能影响追踪（skill-impact.md），跨迭代只增不删、不接受回滚。
- **Skill Proposer**：以 ReAct 方式自主探索 Wiki 和原始轨迹、诊断根因、生成原子性技能新建/patch 提案的 LLM 智能体。
- **Gating and Rollback**：在验证集上评估候选技能，仅当分数提升时接受并更新 $\mathcal{R}_{\mathrm{best}}$，否则回滚技能集；Wiki 始终保持。
- **Persistent Knowledge Accumulation**：通过 Wiki 将跨迭代的成功策略与失败根因系统化地累积起来，使后续技能更新建立在日益完善的知识基础上。
- **Cross-model Skill Transfer**：由某一模型演化出的技能可注入到其他模型（甚至更小模型）的系统中使用，揭示技能发现能力与执行能力的可分离性。

## 可复现要素
- **数据集**：LiveMathematicianBench、SealQA、SpreadsheetBench、OfficeQA、ALFWorld；论文未声明自有数据开源，但引用源均为公开基准。
- **代码/权重**：论文未提及代码开源声明（URL 指向 arXiv PDF，未附 GitHub 链接）。
- **关键超参**：每轮采样 ≤8 条轨迹（最多 5 失败 + 3 成功）；每条轨迹上限 15k 字符；batch size $B=N_{\mathrm{train}}$（全量）；ReAct 轮数 $T_{\mathrm{ReAct}} \in [10, 20]$；bootstrap 检验 1000 次、p<0.05；门控阈值严格 $\Delta > 0$；$\mathcal{R}_{\mathrm{best}}=1.0$ 时提前终止。
- **环境**：开源模型通过 vLLM 部署（Qwen/Gemma）；Gemini-3.5-Flash 使用官方 API。
