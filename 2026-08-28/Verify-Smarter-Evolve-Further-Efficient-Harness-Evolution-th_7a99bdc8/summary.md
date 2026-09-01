---
title: "Verify-Smarter-Evolve-Further-Efficient-Harness-Evolution-th"
source: https://arxiv.org/pdf/2608.27311v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 23:39:49"
field: "Agent 自动化优化与自我改进"
keywords: ["agent harness", "behavior-aware verification", "harness evolution", "propose-and-verify", "attribution gate", "budget-constrained optimization"]
innovations: ["行为感知的动态验证任务选择，针对不同修改匹配相关任务而非固定集", "可归因证据门控机制，要求轨迹层面可归因改进方可接受候选", "预算感知双阶段验证确认框架 HARNESSLENS，在 200 单位预算下超越更高预算基线"]
benchmarks: ["τ²-bench Retail", "τ³-bench Banking Knowledge", "Terminal-Bench 2.0", "BIRD Mini-Dev Challenging"]
---

# 论文速读：Verify-Smarter-Evolve-Further-Efficient-Harness-Evolution-through-Behavior-Aware-Verification

## 一句话总结
本文提出 **HARNESSLENS**，一种预算感知、行为感知的 Agent Harness 自动演化框架，通过为每个候选修改动态选择与其意图行为相关的验证任务，并结合可归因证据门控机制，在显著更少的交互预算下实现比现有基线更优的 Harness 演化效果（平均提升 7.6–13.6%）。

## 研究问题与动机
- **核心问题**：Agent Harness（包含指令、技能、工具描述、权限等组件）的配置调整目前主要依赖手动工程，且针对不同模型/环境的配置难以直接迁移；如何使 Harness 能基于交互轨迹自主持续演化？
- **现有方法不足**：现有 propose-and-verify 方法通常对所有候选修改使用**固定或随机采样的同一批验证任务**，当不同修改针对不同行为时，大量任务与候选修改无关，导致：①无效 rollout 浪费预算；②聚合指标掩盖特定行为的回归；③无法发现意图行为是否真正改善。
- **已有改进仍不够**：尽管近年方法改进了轨迹诊断（Lin et al., 2026b; Chen et al., 2026）或通过预选核心集/分阶段筛选降低成本，但验证任务仍未针对每次修改的**行为效应**自适应调整，评估证据也未被系统复用指导后续演化。

## 核心贡献（创新点）
1. **行为感知的验证任务选择**：首次提出按每次修改的预期行为动态选择验证任务的策略，而非对所有候选使用共享固定任务集，显著减少无关行为上的无效 rollout。
2. **HARNESSLENS 框架**：一个与 Harness 无关、预算感知的自动化演化框架，运行时发现用户可配置组件及其更新机制，对交互轨迹进行诊断，并迭代提出、验证、审核 Harness 修改。
3. **可归因证据门控（Attributable-Evidence Gate）**：要求候选改进必须能从轨迹层面归因到目标行为的变化，而非仅靠聚合通过率提升即可通过，有效避免噪声驱动的有害修改被接受。
4. **系统性实验验证**：在三个 Agent Harness（OpenCode、Codex CLI、Pi）和四个基准（τ²-bench Retail、τ³-bench Banking Knowledge、Terminal-Bench 2.0、BIRD Mini-Dev Challenging）上验证，以最小预算取得最优或并列最优结果，平均提升 7.6–13.6%。

## 方法详解
HARNESSLENS 包含三个核心阶段，受固定交互预算 B 约束：

### 4.1 Context Exploration（上下文探索）
- **Task-Space Exploration**：遍历 TRAIN 任务的查询、环境指令、工具描述，按用户主目标分组，形成任务群结构，用于后续验证任务选择和回归检查。不执行任何任务。
- **Harness-Space Exploration**：分析框架 F 的配置与文档，识别用户可配置组件（指令、技能、工具/参数描述、Agent 角色、运行时扩展等）及其可更新范围，记录变更可能影响的预期行为。

### 4.2 Trajectory Diagnosis（轨迹诊断）
- **Experience Extraction**：从当前 Harness 下的执行轨迹中提取可复用经验（成功策略）和 recurring deficiencies（反复出现的不足），并与支持轨迹关联。
- **Experience Analysis**：将提取的证据与任务分组、可编辑组件结合，生成修改提案（proposal），每个提案明确目标行为、支撑轨迹、可调用的组件。剔除缺乏足够证据支撑的提案。

### 4.3 Harness Evolution（Harness 演化）
每一轮迭代由演化 Agent 执行三个子步骤：
1. **Candidate Proposal**：选择一个 proposal，应用到当前确认 Harness $\mathcal{H}_j$ 的副本，生成候选 $\widetilde{\mathcal{H}}_j = (\mathcal{F}, \delta_j(\mathbf{h}_j))$，并通过轻量级运行时检查验证修改是否生效。
2. **Behavior-Aware Verification**：从 TRAIN 中选取至少 5 个相关任务组成验证批次，包括：直接验证目标行为的转换任务（conversion）、与支撑轨迹关联的正向控制任务、用于检测回归的保留任务（preservation）、用于测试修改范围的诊断任务（diagnostic）。在当前 Harness 与候选 Harness 下以配对条件运行轨迹，再通过 Trajectory Diagnosis 判断改进是否可归因、是否有回归。
3. **Harness Review and Update**：仅当候选同时满足**可归因正向证据 + 无可归因回归**时进入确认阶段；确认批次最多保留 2 个原验证任务，其余用新任务填充。只有在确认批次上主要指标提升时才正式更新 $\mathcal{H}_{j+1} = \widetilde{\mathcal{H}}_j$，否则保持 $\mathcal{H}_j$。

**预算会计**：每单位 LLM session 或 task trial 计 1 单位预算，总预算 B=200。首轮 30 任务×2 trial = 60 单位；验证+确认一轮约 50–68 单位。

## 实验与结果
- **数据集/基准**：τ²-bench Retail（TRAIN 30/TEST 40）、τ³-bench Banking Knowledge（30/67）、Terminal-Bench 2.0（30/59）、BIRD Mini-Dev Challenging（30/72）；TEST 全程对演化过程盲态隔离。
- **Agent Harness**：OpenCode v1.17.13、Codex CLI v0.144.4、Pi Coding Agent v0.80.10；LLM 均为 deepseek-v4-flash-preview。
- **基线**：Self-Harness（4800 rollouts）、Meta-Harness（660 rollouts）、HarnessFix（300 rollouts）；HARNESSLENS 总预算 B=200（含 LLM session + task trial）。
- **主要结果**（Table 1，pass@1 on TEST）：
  - **OpenCode**：HARNESSLENS 在 Banking（25.37 vs 22.39 HarnessFix）和 BIRD（45.83 vs 40.28 HarnessFix）上最优；AVG = 47.53%。
  - **Codex**：HARNESSLENS 在 BIRD 上达 47.22%（远超所有基线）；AVG = 44.06%。
  - **Pi**：HARNESSLENS 在 Banking 上达 33.33%（显著提升）；AVG = 49.67%，四项基准中三项最优。
  - **平均提升**：相对初始 Harness 提升 7.6–13.6%。
- **消融实验**（Table 2，OpenCode）：移除行为感知选择（Fixed Batch / Random Batch / RHO-based Batch）导致 Banking 和 BIRD 上无改善；仅用指标门控（Metric-Only Gate）在 Banking 和 BIRD 上无提升，证明两个机制缺一不可。
- **关键结论**：更多验证 rollout ≠ 更好结果（Self-Harness 预算最多但表现不如 HarnessFix）；行为感知验证起的是**过滤器**而非搜索加速器的作用，无法归因的迭代直接返回 H₀，避免性能回退。

## 相关工作脉络
1. **Self-Harness (Zhang et al., 2026)**：对候选修改在所有 TRAIN 任务上固定验证，预算 4800 rollouts；本文用行为感知批次+预算 200 单位超越其性能。
2. **Meta-Harness (Lee et al., 2026b)**：端到端优化 Harness，验证集固定；本文强调验证任务应适配每次修改的行为目标而非全局固定集。
3. **HarnessFix (Chen et al., 2026)**：聚焦轨迹诊断与修复，但验证任务同样固定；本文在此基础上引入可归因证据门控，防止聚合指标掩盖的真实回归。
4. **SkillBoost (Lin et al., 2026a)**：发现优化后的技能在相似任务间自然迁移；本文与之呼应——高任务多样性（如 Terminal-Bench）下难找到跨任务泛化的有效修改，解释了行为感知选择的重要性。
5. **RHO-based task selection (Pan et al., 2026)**：预筛选核心集方法；消融显示其在行为特异性验证上不如本文的动态任务选择（表2中 RHO-based 变体仅 BIRD 略优于 H₀）。
6. **GEPA (Agrawal et al., 2026)**：Reflective prompt evolution；与本文的区别在于本文面向 richer harness 组件（技能、工具描述、Agent 角色等）而不仅是 prompt，且每次修改验证任务不同。

## 局限性与未来方向
- 实验仅覆盖一个模型族（deepseek-v4-flash-preview）和三个开源 Harness，在更多模型架构和开放部署环境中的有效性尚待验证。
- 预算计量以 LLM session 和 task trial 为单位，未标准化 token 消耗、延迟和 monetary cost，跨角色/基准的实际计算成本对比存在偏差。
- 高任务多样性场景（如 Terminal-Bench）下，能同时被 TRAIN 验证且广泛适用于 TEST 的有效修改更难发现，未来需探索更细粒度的行为匹配策略。
- 当前框架假设固定 base model 和固定 framework，未来可扩展至 framework 层面的联合优化。

## 研究启发与可借鉴点
1. **行为感知验证任务选择**：将验证任务从"全局固定集"改为"针对每次修改的行为目标动态选择"，这一思路可迁移至任何基于 propose-and-verify 的智能体优化框架（如 prompt 优化、workflow 演化）。
2. **可归因证据门控**：要求改进必须在轨迹层面可归因而非仅靠聚合指标，有效防止过拟合 TRAIN 噪声；可作为通用筛选机制嵌入各类自动化 Agent 改进流程。
3. **预算会计的精细设计**：将 LLM session 和 task trial 统一计量并前置校验剩余预算是否支持完整两阶段（验证+确认）周期，避免中途断崖式耗尽；适合所有交互预算受限的自动化实验场景。
4. **双阶段验证确认机制**：初验发现可归因改进后，用新批次确认以提高泛化信心；可借鉴于 test-time adaptation、continual learning 等需要防止灾难性遗忘的场景。
5. **任务多样性分析视角**：本文揭示了"TRAIN 中行为一致性越高、演化收益越大"的现象，未来研究可主动利用此规律，在演化前对 TRAIN 集做行为聚类以评估演化潜力。

## 关键术语表
**Harness**：决定 Agent 如何感知任务、使用工具和执行行为的配置系统，包含指令、技能、工具描述、权限、Agent 角色等组件。
**Behavior-Aware Verification**：根据每次候选修改的预期行为效应，动态选择与其相关的验证任务集，而非对所有候选使用同一固定任务集。
**Attributable-Evidence Gate**：仅当改进能从轨迹层面明确归因到目标行为变化、且无可归因回归时，才接受候选修改的门控机制。
**Trajectory Diagnosis**：从 Agent 执行轨迹中提取可复用经验与反复出现的缺陷，并生成带证据支撑的修改提案的分析模块。
**Context Exploration**：演化初期的探索阶段，包括任务空间分组（Task-Space）和可编辑组件识别（Harness-Space），为后续行为感知验证提供结构基础。
**Pass@1**：在盲态 TEST 集上每个任务仅一次 trial 的成功率，作为最终泛化性能指标。
**Interaction Budget (B)**：约束演化过程的总交互预算单位数，包含所有 LLM session 和 task trial 消耗。
**Conversion Task / Preservation Task / Diagnostic Task**：验证批次中四类任务标签，分别用于直接验证目标行为、检测意外回归、测试修改范围边界。

## 可复现要素
- **数据集**：τ²-bench Retail、τ³-bench Banking Knowledge、Terminal-Bench 2.0、BIRD Mini-Dev Challenging；TRAIN/TEST 划分种子为 42，详细任务 ID 见匿名补充材料。
- **代码开源**：https://github.com/jhxu5214/HarnessLens（论文声明）
- **模型**：deepseek-v4-flash-preview；Agent Harness：OpenCode v1.17.13、Codex CLI v0.144.4、Pi Coding Agent v0.80.10
- **关键超参**：预算 B=200 单位；每任务 trial 数 K=2；验证批次至少 5 个任务；探索/诊断角色最多 60 agent steps，上下文限制 65,536 tokens，输出限制 24,576 tokens。
- **未提及**：具体学习率、temperature 等模型生成超参（仅说明使用 high-reasoning profile）；各基准的硬件配置。
