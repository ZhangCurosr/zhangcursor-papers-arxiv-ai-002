---
title: "Measure-Before-You-Manage-Evaluating-Agent-Working-Memory-in"
source: https://arxiv.org/pdf/2608.31057v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 22:15:50"
field: "Agent 记忆系统与评估"
keywords: ["Agent Working Memory", "Coding Agents", "Memory Management", "SWE-bench", "Object-Aware Compression", "Retrieval-Based Memory"]
innovations: ["首次对编程 Agent 工作记忆进行类型化四维核算（size/retention/representation/compression）", "揭示校准增益不可迁移：OA在校准集显著优于FIFO但在held-out不显著", "提出四层次评估框架（stored state/delivered context/management work/outcome）并以实证证明同预算≠同交付上下文"]
benchmarks: ["SWE-bench Lite", "Terminal-Bench probe"]
---

# 论文速读：Measure-Before-You-Manage-Evaluating-Agent-Working-Memory-in

## 一句话总结
本文系统刻画了编程 Agent 工作记忆中不同类型对象的语义异构性（工具输出、代码工件、指令、Agent 状态在大小、留存、表征和压缩行为上显著不同），并通过两个语义感知管理策略（对象感知压缩 OA 与检索式策略 GA）的案例研究，揭示"校准增益未必能迁移到未见任务"及"名义预算相同不等于实际交付上下文与管理工作量相同"的核心发现，最终提出四层次评估框架（存储状态、交付上下文、管理工作量、任务/过程结果）。

## 研究问题与动机
- **工作记忆语义异构性未被系统化度量**：现有记忆管理机制（保留/压缩/淘汰/检索）往往将工具输出、代码工件、指令、Agent 生成状态等视为统一 token 池，但它们在尺寸分布、生命周期、表示形式上差异显著。
- **评估指标单一**：当前研究多用"token 预算"或"代码修复成功率"作为主要终点，忽略了管理的实际开销（辅助调用、嵌入计算、墙钟时间）与"交付上下文"的真实差异。
- **校准/过拟合风险缺乏系统刻画**：在同一开发集上调参获得的优势，在独立 held-out 集上是否稳定？本文给出否定证据。
- **SWE-bench 类评测的局限性**：已有工作（如 MemGPT、SWE-agent、SWE-bench）关注记忆层级与检索，但缺少对编程 Agent 任务局部工作记忆的精细类型核算与多粒度评估。

## 核心贡献（创新点）
1. **首次对编程 Agent 工作记忆进行类型化核算**：基于 55 条完整轨迹，对 1,350 个 in-context 对象按类型统计 size/retention/representation/compression 四维特征，发现工具输出占 55.5% 内容体积但 artifact 经留存加权后贡献 38.9% 留存成本——单一体积指标会严重低估 artifact。
2. **提出并评估两种语义感知管理策略作为对照案例**：对象感知压缩（OA，手工启发式）与基于检索的策略（GA，适配自 Generative Agents 的 recency/relevance/importance 三分量打分）。两者均非"通用最优策略"，而是用于揭示评估陷阱。
3. **揭示"校准增益不可迁移"的实证规律**：OA 在校准集（15 个 SymPy 任务）上相对 FIFO 减少 1.633 次重复调用（Holm p=0.0146），但在 held-out（8 个跨仓库任务）上仅减少 0.500 次且不显著；说明单一指标改进不可直接外推。
4. **提出四层次评估框架**：将评估拆分为 stored state、delivered context、management work、task/process outcome 四个独立度量层，并指出共享 token 预算不等于共享交付上下文（如 FIFO 与 GA 在 10 个臂对中仅 6/8 任务交付量相差 ≤10%）。

## 方法详解
- **工作记忆对象模型**：四种类型——Instructions（受保护的 system/task 文本）、Artifacts（源码视图）、Tool outputs（执行反馈）、Agent state（模型生成文本）。每个对象含 ID、size、creation step、eviction step、representation（raw/compressed/summary/pointer）及可选 path/version/dependencies。
- **留存与体积核算公式**：$r_o = \max(0, \min(e_o, T) - c_o)$，内容体积 $V_c = \sum_{o \in c} s_o$，留存加权成本 $R_c = \sum_{o \in c} s_o r_o$。两者均为会计量，非 provider 账单或 KV-cache 占用。
- **OA 策略（对象感知压缩）**：评分公式 $u_o = \max(10^{-6}, b_t e^{-d_t a_o} m_o[1+0.35\min(k_o,5)](1-p_t z_o)q_o)$，最终分数 $S_o = u_o / \max(s_o^{\text{rendered}}, 1)^{0.5}$。type 参数（Table C.1）：Instruction 最低衰减（$d=0$）、最高 base（1.0）；Tool output 最高衰减（$d=0.18$）、最低 base（0.5）。最多 4 轮降级（raw→compressed→summary→pointer）。
- **GA 策略（检索式）**：$S_o^{\text{GA}} = 0.5 \cdot 0.99^{s-\ell_o} + 3.0 \cos(\tilde{e}_o, \tilde{e}_q) + 2.0 \tilde{I}_o$，三分量 min-max 归一化。relevance 用 BAAI/bge-small-en-v1.5 局部嵌入；importance 用一次额外 LLM 调用评分（1–10）；时钟在 admission 时更新而非自动 prompt inclusion。
- **主评估终点**：repeat call——若当前 tool invocation 的 tool name + sorted-key JSON args 与历史某次完全匹配则计 1。此为 process regularity 指标，不等价于修复成功。
- **预算规则**：早期 $B = \max(\lfloor fP \rfloor, B_{\min})$；检索阶段 $B_t = \max(\lfloor 0.15P_t \rfloor, \lfloor F_t/0.8 \rfloor + 1)$，其中 18/24 任务实际预算高于名义 15%。

## 实验与结果
- **数据集与规模**：SWE-bench Lite 来源，55 条完整轨迹用于表征（8 个仓库：SymPy 31、pytest 8、seaborn 5、pylint 4、Sphinx 4、Django 2、Flask 1、requests 1）；15 个 SymPy 任务校准；8 个 held-out 任务；24 个检索任务中 8 个完成全部 6 臂（48 个 paired cells）。
- **表征结果（Figure 1 & Table B.1）**：
  - 工具输出：585 个对象，中位数 73 tokens，平均留存 8.61 步，占 55.5% 体积 / 40.2% 留存成本。
  - Artifact：165 个对象，中位数 624 tokens，平均留存 10.71 步，占 28.3% 体积 / 38.9% 留存成本。
  - 压缩比差异：Artifact 压缩/raw = 0.150 vs Tool output = 0.673。
- **OA 策略结果（Table 2）**：
  - 校准集：OA–FIFO ∆=−1.633（p=0.0049, Holm p=0.0146）；OA–LRU ∆=−1.533（p=0.0215）；OA–UC ∆=−0.733（p=0.0156）。
  - Held-out：OA–FIFO ∆=−0.500（p=0.5000，仅 2/8 非零差）；OA–UC ∆=−1.000（p=0.0625, Holm p=0.1875）——无一通过 Holm 校正。
- **GA 策略结果（Table 2 & D.2）**：
  - GA–FIFO ∆=−0.375（95% CI [−1.250, +0.375]，p=0.7500）；GA–LRU-D ∆=−0.125（p=0.3594）；GA–OA ∆=+0.500（p=0.2188）。
  - 无对比通过 Holm 校正（全为 1.0000）。
- **管理开销（Table 3）**：GA 额外 285 次 importance 调用；OA 额外 169 次 summary 调用。GA 相对于 FIFO 平均增加 +67.45 秒 wall time。
- **真实系统回放**：NVIDIA GB10 上 Qwen2.5-Coder-32B，32,768 token 硬限——无约束臂在 1 个任务达到 37,883 tokens，6/25 步溢出；所有约束臂 ≤16,643 tokens，体现硬交付边界。
- **最强结果**：OA 在校准集上相对 FIFO 减少 1.633 次重复调用（统计显著）；但其 held-out 泛化失败。

## 相关工作脉络
1. **MemGPT**（Packer et al., 2023）：LLM 作为 OS 的分层记忆系统，面向文档分析与多会话对话；本文聚焦任务局部工作记忆与语义异构性，不涉及跨会话迁移。
2. **Generative Agents**（Park et al., 2023）：社会仿真中的记忆检索 + 反思/规划；本文仅借用其 recency/relevance/importance 打分组件并适配到编码场景，明确剥离 reflection/planning/social。
3. **SWE-agent**（Yang et al., 2024）：关注 Agent-computer interface 对软件工程行为的影响；本文聚焦工作记忆管理策略本身，而非接口设计。
4. **Lindenbauer et al. (2025)**：比较 observation masking 与 LLM summarization 在 SWE-bench Verified 上的效果；本文不比较这两种方法，而是剖析评估维度缺失问题。
5. **Chintalapati et al. (2026)**：memory condenser 在科学发现任务上的 quality/token cost；本文限定于编码 Agent，引入类型化核算与四层次框架。
6. **Omri et al. (2026)**：agent memory 的 construction/retrieval/generation 成本建模；本文与之互补——前者是宏观系统画像，本文是任务局部工作记忆的精细审计。

## 局限性与未来方向
- **样本量小且仓库聚类**：55 条轨迹覆盖 8 个仓库，无法代表更广泛的编程任务；held-out 仅 8 任务，统计功效有限。
- **模型 revision 未锁定**：claude-opus-4.8 alias 不对应具体版本，辅助请求 temperature 未记录，无法精确复现。
- **生命周期信号有缺陷**：automatic prompt inclusion 会刷新 access clock，使 OA 的 age/reuse 输入退化；artifact version 事件可能仅是渲染事件而非语义过期。
- **缺乏有效 repair 评估**：后一阶段 48 个 paired cells 无正式修复验证；agent finish 信号不能填补此缺口。
- **未做超参搜索日志**：OA 参数为"calibration-informed defaults"，无完整搜索账本，存在调参选择偏差。
- **未来方向**：(1) 在更大规模、跨领域编程 Agent 上验证四层次框架；(2) 开发纠正版生命周期信号；(3) 探索语义感知的端到端训练策略（而非手工启发式）；(4) 建立统一的 token 预算→交付上下文映射标准。

## 研究启发与可借鉴点
1. **类型化核算框架可直接迁移**：对任何 Agent 系统的记忆对象，均可按"类型 × 尺寸 × 留存 × 压缩比"四维审计，避免"单一 token 池"模型的误导性结论。
2. **评估需分离四层指标**：任何新记忆管理策略均应报告 stored state / delivered context / management work / outcome，避免"同预算即同实验条件"的谬误。
3. **repeat call 作为 process regularity 指标的实用性**：在缺乏 gold patch 或完整 repair 验证时，重复工具调用可作为过程稳定性的代理终点，值得在 ablation 研究中推广。
4. **Budget label ≠ effective budget**：在报告实验时须同时报告 nominal fraction、absolute cap、measurement unit 及 actual delivered tokens，三者缺一不可。
5. **检索式策略的高管理开销警示**：GA 增加 285 次 importance 调用（约 +67s wall time），提示语义检索的隐性成本不可忽视；轻量策略（如 UC）在相同 cap 下可能更均衡。

## 关键术语表
- **Agent Working Memory**：编程 Agent 在执行过程中临时维护的上下文集合，包含工具输出、代码工件、指令和内部状态。
- **Object-Aware (OA) Compression**：基于对象类型/年龄/访问计数/过期标记的启发式压缩策略，通过四档降级（raw→compressed→summary→pointer）控制上下文规模。
- **Retrieval-Based Policy (GA)**：适配自 Generative Agents 的三分量打分（近期性×相关性×重要性），通过 BAAI/bge-small-en-v1.5 嵌入与 LLM 评分实现上下文选择。
- **Repeated Call**：主评估终点；tool name 与 sorted-key JSON args 完全匹配的前次调用计为一次重复，反映过程regularity。
- **Delivered Context**：实际送入模型上下文的 token 量，区别于名义 budget label；同一 cap 下不同策略的 delivered context 可能显著不等。
- **Retention-Weighted Cost ($R_c$)**：$\sum s_o r_o$，将对象 size 乘以其留存步数，刻画长期驻留对象的实际上下文负担。
- **Four-Level Evaluation Framework**：stored state / delivered context / management work / task or process outcome 四层次评估框架，用于全面刻画记忆管理策略的效果与代价。
- **SWE-bench Lite**：本文使用的编程 Agent 基准，源自真实 GitHub issue，55 条轨迹覆盖 8 个开源仓库。

## 可复现要素
- **数据集**：SWE-bench Lite 相关轨迹存档于作者工作库（路径：A: results/traces/, results/policy_tight/, results/policy_w3/, results/policy_heldout/；G: results/phase3_*/），但论文声明**不提供公开匿名 release**。
- **代码**：审计脚本（analysis/build_appendix_evidence.py）仅用标准库，可从附录定位；完整代码未开源。
- **模型**：claude-opus-4.8 alias（gateway prefix 匿名，revision unpinned）；嵌入模型 BAAI/bge-small-en-v1.5 (rev 5c38ec7c)；tokenizer 为 Qwen/Qwen2.5-Coder-32B-Instruct (rev 381fc969f78e)。
- **关键超参**：OA 的 type base $b_t$、decay $d_t$、stale penalty $p_t$（见 Table C.1）；GA 的 0.5/3.0/2.0 三权重及 0.99 衰减系数；预算 floor 1,200 token（tight sweep）/ 6,000 token（wide sweep）。
- **环境**：Python 3.12.7, NumPy 1.26.4, SciPy 1.13.1, pandas 2.2.2；serving replay 为 NVIDIA GB10, Ubuntu 24.04.3, CUDA 13.0。
- **复现等级**：论文自行回答 NeurIPS checklist 为"不可完全复现"（Q4=No, Q5=No），因 served model revision 与完整 request provenance 缺失。
