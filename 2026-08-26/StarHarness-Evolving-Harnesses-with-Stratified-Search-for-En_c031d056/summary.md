---
title: "StarHarness-Evolving-Harnesses-with-Stratified-Search-for-En"
source: https://arxiv.org/pdf/2608.24804v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 06:04:35"
---

# 论文速读：StarHarness-Evolving-Harnesses-with-Stratified-Search-for-En

## 一句话总结
StarHarness 提出了一种在固定模型权重的前提下，通过分层任务采样、Proposer-hidden 隔离与护栏约束自动演化企业环境自适应 Agent Harness 的框架；在 ITBench、EnterpriseOps-Gym 与 AutomationBench 三个基准上实现 20–35 个百分点的绝对提升，并验证了跨模型与保留任务的泛化能力。

## 研究问题与动机
- 企业级 Agent 需在工具密集、状态持久、策略隐式的环境中运行，默认 Harness 常导致工具调用失败、上下文丢失、误诊与无效循环等模型-环境不匹配问题。
- 现有 Prompt/Harness 优化方法缺乏严格的泛化评估协议，容易在搜索集上过拟合，且难以区分“搜索表现”与真实环境适应能力。
- 冻结模型权重仅靠外层 Harness 适配能否高效改善企业任务表现，且演化结果能否跨模型迁移，尚缺系统性验证。
- 需要建立可隔离、可审计、带安全护栏的演化管线，防止 Proposer 将任务特解或 Verifier 逻辑硬编码进 Harness。

## 核心贡献（创新点）
- 提出基于失败模式分层与 Proposer-hidden 任务隔离的 Harness 演化协议，首次在三个企业基准上量化区分搜索表现与泛化边界；与 Meta-Harness 等同场景方法相比，本文严格分离搜索集、选择集与保留集，避免 Discovery 设定下的虚高报告。
- 设计 Git-diff 作用域护栏与两级搜索策略（树搜索探索 + 爬山利用），在防止过拟合的同时保留多假设竞争；与 SIA 等权重+Harness 联合更新方法相比，本文坚持冻结模型参数，使演化产物可作为普通代码变更测试与回滚。
- 证明冻结专用 Harness 可无缝迁移至 GPT 与 Qwen 模型家族，且对演化集外的保留任务保持稳定增益；现有工作多在同一模型族内评估，本文首次在跨架构家族上验证外部适配的普适性。
- 通过轨迹分析归纳出接口修复、环境约定显式化与操作知识压缩搜索三类适配模式，揭示 Harness 演化的内在机制；区别于仅报告最终得分的研究，本文提供可解释的执行轨迹归因。

## 方法详解
- **目标形式化**：$h^* = \arg\max_{h \in \mathcal{H}} J(h; \mathcal{D}_{\text{holdout}})$，其中 $J$ 为固定模型 $M$ 在基准上的平均任务得分；演化过程中不可访问保留集，仅以搜索集与选择集近似目标。
- **任务划分与分层采样**：从 $N$ 任务中保留 $N'$ 个可复现任务，按基线失败模式（wrong_tool、context_loss、missing_evidence、premature_conclusion 等）、基线得分与 Verifier 通过率采样 $K \approx N/2$ 构建演化池。将演化池划分为 Proposer-visible 搜索集与 Proposer-hidden 选择集，并匹配三者分布；剩余 $N' - K$ 为 Holdout 集，全程不参与提案或接受决策。
- **三阶段演化循环**：Proposer 读取当前 Harness 与搜索集轨迹返回候选 Patch；Validator 检查作用域、导入依赖与单任务冒烟测试；通过者先执行 Proposer-selected test flip，若翻转成功则评估于 Hidden Selection 集。仅当选择集平均分严格提升（或持平且 Verifier 指标提升）时提交为新前沿，否则回滚。
- **搜索策略**：Hill Climbing 维护单前沿，迭代仅接受严格改进的 $\Delta_t$；Tree Search 维护候选节点集合，每个节点记录父指针、累积 Patch、搜索轨迹、验证状态与选择分，支持探索失败模式、调试失败候选、合并兼容节点，最佳存活节点继承为后续前沿。两阶段顺序衔接（先树搜索探索，后爬山利用）。
- **护栏机制**：禁止基于 Task ID 分支、硬编码答案、在 Prompt 中写入 Verifier/断言内容、访问 Ground-truth 表或隐藏状态、Benchmark-specific 答案映射；确保 Patch 仅编码可复用环境行为。
- **持久化账本（Ledger）**：记录 Patch、会话、评估工件、前沿得分、每任务结果、接受/丢弃假设；树搜索模式下还保存候选节点与父链接，支持回溯与多路径竞争。

## 实验与结果
- **数据集与基线**：ITBench SRE（40 个 K8s 根因分析场景）、EnterpriseOps-Gym ITSM（103 个 ITSM 工作流，SQL 校验）、AutomationBench Finance（100 个财务工作流，跨 47 个 SaaS，程序化断言）；基线为未修改 Stirrup 默认 Harness、Pi/Codex Harness、GEPA 提示优化。
- **主要结果**：StarHarness 在三基准上全面最强，相对 GEPA(Pi) 分别提升 +13.8、+22.3、+17.6 pp；绝对提升约 20–35 pp，仅需 4–12 次接受变更。
- **成本与效率**：推理成本下降 17%（ITBench）、53%（EnterpriseOps-Gym）、29%（AutomationBench）；轨迹显示 Turn 数、工具调用数显著减少，误报诊断大幅下降。
- **冻结迁移**：同一演化 Harness 无重构直接迁移至 GPT-5.5、GPT-5.4-mini、Qwen3.5/3.6，ITBench 最高达 +46.3 pp（GPT-5.4-mini），EnterpriseOps-Gym 最高 +20.6 pp（Qwen3.6-27B），AutomationBench 最高 +40.4 pp（GPT-5.4-mini）。
- **保留集泛化**：GPT-5.4 在 Evolution 集提升 22–45 pp，Holdout 集仍获 +15.1 至 +31.7 pp，验证无明显过拟合。

## 相关工作脉络
- **Prompt/Harness 优化**（GEPA、Meta-Harness）：Meta-Harness 使用保留集但仍在同一 89 任务基准上搜索并报告 TerminalBench-2 成绩，属 Discovery 设定；本文通过任务级分离与 Proposer-hidden 机制严格区分搜索与泛化，回应 Wang et al. (2026) 对评估严谨性的呼吁。
- **Agent 架构/工作流编译**（DSPy、AFlow）：侧重声明式编译或自动化工作流生成；本文聚焦企业状态型环境的代码级 Harness 适配，搜索空间覆盖 Prompt、Tool Schema、MCP Provider、Subagent 与执行策略。
- **自进化 Agent**（SIA）：联合更新 Harness 与模型权重；本文坚持冻结权重，将演化产物作为可回滚的外部配置，更适合企业生产环境的合规与审计需求。
- **企业/状态型 Agent 基准**（WorkArena、Terminal-Bench 2.0、Tau-bench、ToolSandbox）：涵盖浏览器任务、命令行、交互式工具使用；本文选取的三基准均强调持久状态、策略约束与最终状态校验，更贴近运维与财务自动化场景。
- **评估方法论**（Wang et al., 2026）：强调 Harness 演化需控制泄漏与区分搜索/泛化；本文的分层采样+Proposer-hidden selection 协议直接对齐该倡议，并提供公开代码与完整 Ledger 以提升可重复性。

## 局限性与未来方向
- 论文自述未来方向为 Harness 与模型权重的协同演化（RL），目前冻结权重的适配上限受限于底层模型本身的能力边界。
- 树搜索与爬山为顺序衔接设计，非因果对照实验，难以剥离两者独立贡献与最优配比。
- 护栏依赖人工定义的 Forbidden changes 列表，对新型规避策略或隐式数据泄露的覆盖有限。
- 多个 Patch 存在累积交互效应，论文承认无法从配对比较中隔离单个 Patch 的因果贡献。
- 跨模型迁移仅在 GPT 与 Qwen 同架构家族内验证，跨架构（如 MoE vs Dense、不同推理协议）的泛化尚未测试。

## 研究启发与可借鉴点
- **分层任务隔离协议**可直接迁移至团队内部的 Auto-Prompt/Harness 优化管线，作为防过拟合的标准评估基线。
- **Proposer-hidden selection + held-out evaluation** 的两段式验证设计适合集成到现有 Agent CI/CD 流程，实现自动化迭代与安全发布。
- **Git-diff 作用域护栏**（禁 Task-ID 分支、禁篡改 Verifier）是工业界部署 Agent 自动优化的安全实践，可作为团队工具链的强制约束模板。
- **轨迹分析归纳的三类适配模式**（接口修复/显式约定/知识压缩）可作为后续分析 Agent 失败模式的分类框架，指导定向优化优先级。
- 可与团队现有的 Prompt 调优或
