---
title: "StarHarness-Evolving-Harnesses-with-Stratified-Search-for-En"
source: https://arxiv.org/pdf/2608.24804v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 06:03:54"
field: "Agent 系统与工具调用优化"
keywords: ["agent harness", "harness evolution", "enterprise agents", "tool use", "stratified search", "prompt optimization", "cross-model transfer"]
innovations: ["分层搜索协议：按失败模式分层构建进化池并分离 proposer-visible/hidden 任务以实现严格泛化评估", "冻结模型的 harness 跨模型迁移：用 GPT-5.4 进化的 harness 无需重新进化即可提升 Qwen/GPT 多模型性能", "三项环境专业化形式分析：interface repair、environment conventions、operational knowledge search compression"]
benchmarks: ["ITBench SRE", "EnterpriseOps-Gym ITSM", "AutomationBench Finance"]
---

# 论文速读：StarHarness: Evolving Harnesses with Stratified Search for Enterprise Environments

## 一句话总结
StarHarness 是一种面向企业级工具丰富场景的 Agent harness 进化框架，在模型权重冻结的前提下，通过分层搜索（stratified search）自动演化与环境匹配的 prompt、工具接口、技能模块与执行策略，在 ITBench、EnterpriseOps-Gym 和 AutomationBench 三个基准上实现全基准性能提升 20–35 个百分点，且进化结果可跨模型（GPT/Qwen）零成本迁移。

## 研究问题与动机
- **企业级工具任务中存在持续性的"模型–环境不匹配"**：工具丰富场景涉及有状态后端、跨步骤依赖、领域约定和精确最终状态验证，而工具 schema 通常省略这些上下文，导致默认 harness 性能低下。
- **现有 harness 优化方法缺乏严格的泛化评估**：如 Meta-Harness 在同一组任务上进行搜索和报告，无法区分搜索性能与泛化性能，存在过拟合风险。
- **关键科学问题**：从紧凑任务子集进化出的 harness 能否不过拟合地泛化到未见过任务？进化结果能否跨不同模型（GPT → Qwen）迁移？harness 进化修复了哪些类型的交互失败？
- **已有方法的不足**：prompt 优化（如 GEPA）仅搜索指令级内容；SIA 等方法同时更新权重和 harness，成本高；现有基准（WorkArena、Terminal-Bench 2.0）与真正企业级有状态工作流场景存在差距。

## 核心贡献（创新点）
1. **高效分层 harness 进化协议**：按基线失败模式分层构建紧凑搜索池，并将 proposer-visible 搜索任务与 proposer-hidden 选择任务分离，保留 held-out 任务用于严格泛化评估；与 Meta-Harness 等方法的本质区别在于引入了任务级隔离和 proposer 不可见选择集，避免搜索-评估数据泄露。
2. **冻结模型的专用 harness 跨任务与跨模型迁移**：在三个企业级基准上，用 GPT-5.4 进化的 harness 无需重新进化即可直接应用于 Qwen3.6、GPT-5.4-mini、GPT-5.5 等模型，均带来显著提升；本质区别于需要 per-model 微调的方法。
3. **三项可解释的环境专业化形式**：识别出 interface repair（接口修复）、environment conventions（环境约定显式化）和 operational knowledge search compression（操作知识压缩搜索），并将它们与 agent 精确率、收敛性和效率变化关联；这是对该领域"进化学到了什么"的首次系统化分析。

## 方法详解
- **整体框架**：StarHarness 是一个基于 Oh My Pi（Pi harness 变体）构建的代码 harness，负责 proposer、验证和评估循环的执行；它修改独立的 Stirrup harness（被评估对象），不触碰模型权重和基准本身。
- **可编辑表面**：包括 prompt 与任务框架、工具定义和 schema、参数预处理、skills、MCP providers、subagent 结构、上下文管理、验证逻辑和终止逻辑。
- **优化目标**：最大化 $J(h; \mathcal{D}_{\mathrm{holdout}})$，其中 $h^* = \arg\max_{h \in \mathcal{H}} J(h; \mathcal{D}_{\mathrm{holdout}})$；由于 holdout 不可访问，用 proposer-visible 搜索任务和 proposer-hidden 选择任务近似。
- **三阶段循环**：proposal（proposer 读取当前 harness + 搜索集 traces + ledger 历史，输出候选 patch）→ validation（作用域检查、import 检查、单任务 smoke test）→ evaluation（在隐藏选择集上评估，按确定性规则接受/拒绝）。
- **任务分层策略（§3.2）**：从 N 个任务中保留 $N'$ 可复现任务，采样 $K \approx N/2$ 构建进化池；按基线失败模式（wrong_tool、context_loss、missing_evidence、premature_conclusion）、基线分数、verifier pass rate 三个维度分层；选择集与搜索集在以上分布上匹配但 proposer 不可见其内容；剩余 $N' - K$ 为 holdout。
- **两种搜索策略**：Hill climbing（单前缘贪心搜索，仅接受严格改进）和 tree search（多候选节点探索，保留替代假设后再用 hill climbing 精细化）；后者在 EnterpriseOps-Gym 上先用 8 个 tree-search patch 发现交互性 schema/prompt/tool 失败，再用 4 个 hill-climbing patch 做针对性修复。
- **Guardrail 约束（§3.4）**：候选 patch 为 git diff，作用域限于可编辑目录；禁止：按 task ID 分支、硬编码答案、在 prompt 中包含 verifier/assertion 内容、ground-truth 表、benchmark 特定答案映射。

## 实验与结果
- **数据集**：
  - ITBench SRE：40 个 Kubernetes 根因分析场景（告警、事件、traces、指标、拓扑 → JSON 诊断）
  - EnterpriseOps-Gym ITSM：103 个 ITSM oracle 任务（incident/problem/change/knowledge/user management），SQL verifier 校验最终数据库状态
  - AutomationBench Finance：100 个财务工作流任务（AP/AR/expenses/reporting/bookkeeping），跨 47 个模拟 SaaS 应用，programmatic assertion 评分
- **模型**：进化使用 GPT-5.4（medium reasoning）作为 agent 和 proposer；迁移评估使用 Qwen3.6-27B、GPT-5.4-mini、GPT-5.4、GPT-5.5（medium/high）。
- **基线**：未修改 Stirrup 默认 harness；对比 GEPA (Pi) 提示优化。
- **主要结果**：
  - StarHarness 相对 GEPA (Pi) 提升：ITBench **+13.8 pp**、EnterpriseOps-Gym **+22.3 pp**、AutomationBench **+17.6 pp**
  - 相对默认 harness（GPT-5.4）：ITBench **+35.0 pp**（40.0%→75.0%）、EnterpriseOps-Gym **+20.4 pp**（23.3%→43.7%）、AutomationBench **+26.1 pp**（57.1%→83.2%）
  - **最强结果**：AutomationBench GPT-5.5 + StarHarness 达到 84.9%（+25.3 pp over baseline）
  - **Inference 成本降低**：ITBench -17%、EnterpriseOps-Gym -53%、AutomationBench -29%
- **泛化与迁移**：
  - Held-out 泛化（GPT-5.4）：ITBench +31.7 pp、EnterpriseOps-Gym +15.1 pp、AutomationBench +29.3 pp，与进化集增益接近
  - Cross-model 迁移：Qwen3.6-27B 在 ITBench 上从 25.6%→70.0%（+44.4 pp），是最显著的一次迁移提升

## 相关工作脉络
1. **Meta-Harness (Lee et al., 2026)**：端到端 harness 优化，但在同一 89 任务集上搜索和报告 TerminalBench-2 性能，缺乏严格泛化评估；StarHarness 引入任务分离和 proposer-hidden 选择集解决此问题。
2. **GEPA (Agrawal et al., 2025)**：基于文本反馈的提示进化方法，仅搜索指令和示例层；StarHarness 搜索空间扩展到工具 schema、MCP provider、subagent 结构和执行策略，覆盖更完整的 harness 层次。
3. **SIA (Hebbar et al., 2026)**：联合更新 harness 和模型权重；StarHarness 聚焦 frozen model 场景，edit 作为普通代码变更可测试和回退，部署风险更低。
4. **DSPy (Khattab et al., 2024)**：声明式 LM 调用编译；属于 prompt/demonstration 层面的优化范式；StarHarness 进一步处理 enterprise 场景中特有的有状态后端、跨步骤依赖和领域约定问题。
5. **Terminal-Bench 2.0 (Merrill et al., 2026) / WorkArena (Drouin et al., 2024)**：分别评估孤立 CLI 环境和浏览器知识工作；StarHarness 选取的三个基准覆盖真正企业级有状态工作流（ITSM、SRE 诊断、财务自动化），更具代表性。
6. **Rethinking harness evolution evaluation (Wang et al., 2026)**：呼吁更严格的 harness 进化评估标准；StarHarness 遵循此建议，通过分层隔离明确区分搜索性能与泛化性能。

## 局限性与未来方向
- **局限**：进化仅在一个模型（GPT-5.4）上完成，未在更大模型范围或更多环境上验证；树搜索与爬山搜索的比较在单一基准上进行，非因果 head-to-head 对比；无法从配对比较中隔离单个 patch 的因果贡献；部分评分与原始 benchmark 论文设置不同（AutomationBench），不可直接横向比较。
- **未来方向**：联合进化 harness 和模型权重（through RL），使 scaffold 和 policy 共同适配企业交互协议；探索更小的企业专用模型以降低推理成本；扩展到更多企业场景和模型族。

## 研究启发与可借鉴点
1. **分层隔离评估范式可直接迁移**：proposer-visible search / proposer-hidden selection / held-out evaluation 的三分法可作为 harness 进化研究的默认评估协议，防止隐秘过拟合；适合与本团队的工具调用优化研究结合。
2. **失败模式分层采样策略**：按 wrong_tool、context_loss 等失败模式分层构建进化池，比随机采样更能保证搜索覆盖 diverse failure modes；可用于其他 agent 系统调优场景。
3. **Interface repair 作为通用调试策略**：StarHarness 发现的大量增益来自 schema 精简、null/empty 参数过滤、compounding schema 保留等接口修复，这对任何 tool-rich agent 部署都是实用启发——优先检查工具接口与后端约定的对齐，而非盲目扩展 prompt。
4. **操作知识外置为确定性工具**：将重复性工作（日期计算、财务算术、电子表格操作）从 open-ended reasoning 移到确定性 tool，是压缩搜索的高效手段；可与本团队的 tool grounding 研究结合。
5. **Guardrail 约束设计**：禁止 task-ID 分支、硬编码答案、verifier 泄露等约束，为 harness 进化中的防作弊设计提供了具体可操作的清单，值得在后续工作中复用。

## 关键术语表
**StarHarness**：ServiceNow 提出的冻结模型 harness 进化框架，通过分层搜索自动演化 prompt、工具接口、技能和执行策略以适配企业环境。
**Stratified Search（分层搜索）**：按基线失败模式、任务分数和 verifier pass rate 对任务分层采样，构建紧凑进化池并分离搜索/选择/保留集。
**Proposer-visible / Proposer-hidden**：proposer 可访问搜索集 traces 和结果，但不可见选择集内容，防止搜索-评估数据泄露。
**Stirrup**：Artificial Analysis 提供的轻量级 agent harness 框架，作为本文被优化的目标 harness。
**MCP（Model Context Protocol）**：标准化工具接口协议，定义 agent 与外部服务之间的工具 schema 和调用方式。
**Interface Repair**：harness 进化学到的一类环境专业化形式，指修复工具 schema、参数处理、空值过滤等接口层面的不匹配。
**Guards / Guardrails**：进化过程中的约束机制，禁止按 task ID 分支、硬编码答案、verifier 内容泄露等不当行为。
**Held-out Generalization**：在进化过程中从未暴露给 proposer 的任务上评估进化后 harness 的性能，衡量真正的泛化能力。

## 可复现要素
- **数据集**：ITBench（公开，Jha et al., 2025）、EnterpriseOps-Gym（公开，Malay et al., 2026）、AutomationBench（公开，Shepard & Salimans, 2026）
- **代码/权重**：代码开源，https://github.com/ServiceNow/StarHarness；模型权重为商业模型（GPT-5.4、Qwen3.6），未开源
- **关键超参**：进化池大小 $K \approx N/2$；进化步数 4–12 accepted changes per environment； proposer 为 GPT-5.4 medium reasoning；树搜索 + 爬山搜索序列执行（仅在 EnterpriseOps-Gym 上）
- **模型**：GPT-5.4 (medium reasoning) 用于进化；迁移模型包括 Qwen3.6-27B、GPT-5.4-mini、GPT-5.4、GPT-5.5 (medium/high)
- **评估协议**：沿用 Artificial Analysis 官方评分；AutomationBench 使用 Finance-100 subset 和 Stirrup harness（与原文设置不同）
