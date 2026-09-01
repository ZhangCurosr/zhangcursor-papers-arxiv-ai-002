---
title: "Verify-Smarter-Evolve-Further-Efficient-Harness-Evolution-th"
source: https://arxiv.org/pdf/2608.27311v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 23:39:47"
field: "Agent 系统与工具调用优化"
keywords: ["agent harness", "harness evolution", "behavior-aware verification", "propose-and-verify", "LLM agent", "trajectory diagnosis", "budget-aware optimization"]
innovations: ["行为感知验证任务选择：为每个候选修改动态选择与其目标行为相关的验证任务而非固定批次", "可归因证据门控：仅当改进可归因于目标修改且无明确回归时才接受候选，避免机会性过拟合", "HARNESSLENS 框架：预算感知、harness 无关的自动演化，集成上下文探索-轨迹诊断-演化更新闭环"]
benchmarks: ["τ²-bench Retail", "τ³-bench Banking Knowledge", "Terminal-Bench 2.0", "BIRD Mini-Dev (Challenging)"]
---

# 论文速读：Verify Smarter, Evolve Further: Efficient Harness Evolution through Behavior-Aware Verification

## 一句话总结
本文提出 **HARNESSLENS**，一种预算感知的 Agent harness 自动演化框架，通过**行为感知验证**（为每个候选修改选择与其目标行为相关的验证任务，并结合可归因证据门控）显著减少无效 rollout，在三个 agent harness 和四个 benchmark 上实现平均 7.6–13.6% 的性能提升，且消耗远少于现有方法。

## 研究问题与动机
1. **Harness 演化缺乏自主性**：Agent harness 决定 LLM 如何感知任务、使用工具和执行行为，但手动构建和适配不同模型/环境的 harness 成本高昂。
2. **现有 propose-and-verify 范式浪费验证预算**：主流方法对每个候选修改使用相同固定/随机任务集验证，大量 rollout 落在与修改目标无关的行为上，稀释了修改信号。
3. **聚合分数掩盖特定回归**：固定验证集可能无法覆盖受影响的特定行为，导致验证结果无法揭示改进是否真正生效，且允许机会性提升和未被检测的回归累积。
4. **验证任务未适配修改的行为效应**：即使有轨迹诊断或任务采样改进，已有方法仍未将验证任务与每个修改的行为效应动态适配，也未系统化复用证据指导后续演化。

## 核心贡献（创新点）
1. **行为感知验证范式**：针对每个候选修改选择与其目标行为相关的验证任务，而非对所有修改使用同一固定批次——本质区别在于验证任务按修改的行为效应动态适配。
2. **可归因证据门控（Attributable-Evidence Gate）**：仅当观察到的改进可归因于目标修改且无明确回归时才接受候选——与仅依赖聚合通过率的方法根本不同，避免噪声驱动的有害编辑累积。
3. **HARNESSLENS 框架**：一个 budget-aware、harness-independent 的演化框架，运行时发现用户可配置组件及其更新机制，迭代地 propose-diagnose-verify-review——区别于针对特定 harness 设计的方法，具有跨框架通用性。
4. **闭环证据复用机制**：每次 rollout 既评估当前候选又为后续演化积累证据，验证失败时返回原始 harness 而非继续退化——基线方法（如 Self-Harness、Meta-Harness）约一半的 harness-benchmark 对表现低于初始性能。

## 方法详解
**HARNESSLENS** 包含三个阶段，由确定性控制器协调，受固定交互预算 B 约束：

1. **Context Exploration（上下文探索）**：
   - **Task-Space Exploration**：根据查询、环境指令、工具描述组织 TRAIN 任务，按主要用户目标分组，记录跨目标重叠——不执行任务，不接触轨迹或 TEST。
   - **Harness-Space Exploration**：检查框架文档和运行时行为，识别用户可配置组件（instructions、skills、tool descriptions、agent roles 等）及其可更新范围和受影响行为。

2. **Trajectory Diagnosis（轨迹诊断）**：
   - **Experience Extraction**：将轨迹总结为可复用经验和 recurring deficiencies，链接到支持轨迹；相同奖励的轨迹可能对应不同行为。
   - **Experience Analysis**：结合任务分组和组件清单生成 modification proposals，每个 proposal 指定目标行为、支持轨迹和可影响行为的组件；移除缺乏足够证据支持的 proposal。

3. **Harness Evolution（harness 演化）**：
   - **Candidate Proposal**：基于支持证据、历史尝试、回归风险和剩余预算选择 proposal，从当前 confirmed harness 复制并应用修改 δⱼ 得到候选 Ĥⱼ，轻量级运行时检查确认修改生效。
   - **Behavior-Aware Verification**：选择 ≥5 个 TRAIN 任务构成验证批次（含 conversion、positive control、preservation、diagnostic 四类），分别在 Hⱼ 和 Ĥⱼ 下以匹配条件 rollout；通过 paired trajectory diagnosis 判断 δⱼ 是否改善目标行为且无回归。
   - **Harness Review and Update**：接受条件：(a) 有可归因的正向证据；(b) 无可归因的回归；(c) confirmation batch（≤2 原有任务 + 其余新鲜任务）上 primary metric 提升。候选不被接受时保持 Hⱼ 不变，实现单调改进保证。

关键公式：
- 目标优化：$\mathcal{H}^* = \arg\max_{\mathcal{H} \in \mathbb{H}_\mathcal{F}} \mathbb{E}[R(x,\tau)]$，s.t. $C(\mathcal{H}_0 \to \mathcal{H}) \le B$
- 更新规则：$\mathcal{H}_{j+1} = \tilde{\mathcal{H}}_j$（接受）或 $\mathcal{H}_j$（否则）

## 实验与结果
- **数据集/Benchmark**：τ²-bench Retail (30 train / 40 test)、τ³-bench Banking Knowledge (30/67)、Terminal-Bench 2.0 (30/59)、BIRD Mini-Dev Challenging (30/72)；TRAIN 每任务 2 trials。
- **Agent Harnesses**：OpenCode v1.17.13、Codex CLI v0.144.4、Pi Coding Agent v0.80.10；LLM 均为 deepseek-v4-flash-preview。
- **基线**：Self-Harness（4800 rollouts）、Meta-Harness（660 rollouts）、HarnessFix（300 rollouts）；HARNESSLENS 预算 B=200 units（含 LLM sessions + rollouts）。
- **主要结果**：
  - OpenCode：Banking 25.37%（+4.47 vs H₀）、BIRD 45.83%（+8.33 vs H₀）；Retail 85.00%、Terminal 33.90%（无退步）
  - Codex：BIRD 47.22%（+9.72 vs H₀）；其他持平
  - Pi：Banking 33.33%（+13.93 vs H₀）；Retail 85.00% 持平
  - **平均提升 7.6–13.6%**；12 个 harness-benchmark 对中 8 个取得最佳或并列最佳
  - **稳定性**：HARNESSLENS 从不低于 H₀，而 Self-Harness/Meta-Harness 在 24 对中约一半低于 H₀
- **消融**：去掉行为感知批次选择（Fixed/Random/RHO Batch）几乎无改善；去掉归因门控仅看指标（Metric-Only Gate）在 Banking/BIRD 无 held-out 提升，说明两个机制缺一不可。

## 相关工作脉络
1. **Self-Harness (Zhang et al., 2026)**：每次迭代评估当前 harness 和 3 个候选在全部 30 任务×2 trials 上，共 4800 rollouts；固定批次验证导致修改信号被稀释。
2. **Meta-Harness (Lee et al., 2026b)**：端到端优化 harness 参数，10 轮迭代各 1 候选；同样使用固定训练集验证，平均性能略低于 H₀。
3. **HarnessFix (Chen et al., 2026)**：聚焦轨迹诊断修复缺陷，3 轮迭代×最多 2 次重试；最小基线预算但仍超 HARNESSLENS 1.5 倍。
4. **RHO-based task selection (Pan et al., 2026)**：预选择 coreset 缩小验证集；消融显示其效果接近随机批次，远不如行为感知动态选择。
5. **SkillBoost (Lin et al., 2026a)**：发现优化的 skills 在相似任务间迁移最佳；与本文结论一致——高任务多样性使跨任务泛化修改更难识别。
6. **GEPA (Agrawal et al., 2026)**：reflective prompt evolution vs RL；展示了 prompt 级微调的有效性边界，HARNESSLENS 扩展到 richer harness components。

## 局限性与未来方向
1. **模型和 harness 覆盖有限**：仅在一个模型族（deepseek-v4-flash-preview）和三个开源 harness 上验证，未全面测试更多模型架构和闭源 agent 系统。
2. **预算未归一化到实际成本**：交互预算以 LLM sessions + task trials 计数，未标准化 token 消耗、延迟和货币成本，跨 harness/benchmark 难以直接比较实际开销。
3. **任务多样性限制泛化修改识别**：Terminal-Bench 等高多样性 benchmark 上难以找到跨任务一致的改进信号，可能限制在开放域任务流中的表现。
4. **未来方向**：扩展到更多模型/框架/开放部署环境；引入 token/ latency/ cost 归一化预算；结合非平稳任务流（Liu et al., 2026）和长运行环境（Karten et al., 2026）持续演化。

## 研究启发与可借鉴点
1. **行为感知验证任务选择**：可为任何 propose-and-verify 优化流程（prompt tuning、workflow 合成、skill 演化）提供模板——先建立任务-行为映射，再按候选修改的目标行为动态选验证集。
2. **可归因证据门控设计**：将"指标提升"与"行为改变归因"解耦，避免机会性过拟合；可移植到 LLM 程序优化、RLHF 奖励模型训练等需要防 regressions 的场景。
3. **配对 rollout + 独立 confirmation batch**：验证轮与确认轮分离、保留 ≤2 旧任务其余用新鲜任务的策略，平衡信号强度与泛化保障，适合样本效率敏感的实验协议。
4. **Harness-Space Exploration 的运行时组件发现**：自动识别可配置接口及其影响范围，无需人工标注 editable 组件，对异构 agent 框架适配有直接参考价值。
5. **单调改进保证机制**：不接受则回滚的设计使演化过程从"搜索"变为"过滤"，在 safety-critical agent 部署中降低引入 regressions 的风险。

## 关键术语表
**Agent Harness**：决定 LLM agent 如何感知任务、使用工具和执行行为的配置集合，包括 instructions、skills、tool descriptions、memory、permissions、agent roles 等用户可配置组件。
**Behavior-Aware Verification**：为每个候选修改动态选择与其目标行为相关的验证任务，而非对所有候选使用固定任务集，以提高验证信号的信噪比。
**Attributable-Evidence Gate**：要求观察到的改进必须可归因于目标修改（通过轨迹对比识别行为变化）且无明确回归，才允许候选通过验证。
**Interaction Budget (B)**：演化过程的总消耗上限，计为 LLM sessions 数 + task trials 数，控制 Context Exploration、Trajectory Diagnosis 和 Harness Evolution 的全部成本。
**Paired Rollout**：在同一任务集和匹配 trial 条件下分别运行当前 harness 和候选 harness，以消除环境随机性并直接对比行为差异。
**Experience Extraction / Analysis**：从轨迹中提取可复用经验与 recurring deficiencies，并结合任务分组和组件清单生成 evidence-supported 的 modification proposals。
**Confirmation Batch**：验证轮通过后额外使用的 fresh 任务集（≤2 原有任务 + 其余未用 TRAIN 任务），用于最终确认改进的稳定性和泛化性。

## 可复现要素
- **数据集**：τ²-bench Retail、τ³-bench Banking Knowledge、Terminal-Bench 2.0、BIRD Mini-Dev（Challenging subset）；TRAIN/TEST 划分种子 42，具体任务 ID 在匿名补充材料中提供。
- **代码**：已开源，https://github.com/jhxu5214/HarnessLens。
- **权重**：使用 deepseek-v4-flash-preview API，未重新训练模型权重。
- **关键超参**：预算 B = 200 units；每任务 trials K = 2；验证批次大小 ≥5；confirmation batch 保留 ≤min(2, n-1) 验证任务；各角色 steps 上限（exploration 30、editor 40、diagnosis/evolution 60）；context limit 65536 tokens，output limit 24576 tokens。
- **环境限制**：禁用外部检索工具；benchmark 侧固定 model/provider/tool/inference limits，候选修改不可绕过。
