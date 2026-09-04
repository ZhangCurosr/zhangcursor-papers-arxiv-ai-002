---
title: "Lose-the-Order-Keep-the-Hierarchy-Deordering-HTN-Plans"
source: https://arxiv.org/pdf/2609.03912v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 18:51:48"
field: "HTN规划与计划优化"
keywords: ["HTN planning", "plan deordering", "partial-order planning", "MaxSAT", "plan optimization", "hierarchical task network"]
innovations: ["将Backström的PRF去序算法扩展至含分解强制序的HTN场景", "将Muise等人的Partial Weighted MaxSAT最小去序方法适配至HTN规划形式化", "首次在IPC 2023 Partial-Order HTN基准上系统评估事后去序策略"]
benchmarks: ["IPC 2023 Partial-Order HTN"]
---

# 论文速读：Lose-the-Order-Keep-the-Hierarchy-Deordering-HTN-Plans

## 一句话总结
本文首次系统研究层次任务网络（HTN）规划中的**计划去序**（plan deordering）问题，将经典规划中的 PRF 算法和 MaxSAT 去序技术适配到 HTN 分解约束下，在 IPC 2023 Partial-Order HTN 基准上实现了约束数量最高约 28% 的显著减少。

## 研究问题与动机
1. **HTN 计划优化被严重忽视**：现有 HTN 规划研究主要集中在计划生成（plan generation），而对已生成计划的事后优化（post-plan optimization）关注极少。
2. **经典规划的"去序"技术在 HTN 中缺失**：计划去序（移除多余排序约束、保持计划合法性）在经典规划中已有深入研究，但在 HTN 分解约束下尚未被显式研究。
3. **全序计划在实际应用中存在效率损失**：如消防任务中，不冲突的 "dispatch-ground-crew" 与 "dispatch-helicopter" 被强制序化，剥夺了并行执行的可能性。
4. **既有 HTN 规划器（如 Optiplan）直接生成偏序计划，但与事后去序策略的定位不同**：本文方法可独立应用于已有完全序计划，提供互补方案。

## 核心贡献（创新点）
1. **将 Backström (1998) 的 PRF 算法扩展至 HTN 域**：在经典并发检查基础上，显式纳入由分解树诱导的强制性偏序约束（$\prec_H$），确保去序不破坏层次结构。
2. **将 Muise et al. (2012) 的 Partial Weighted MaxSAT 去序方法适配至 HTN**：构造包含分解约束硬子句、因果正确性约束和传递闭包的 MaxSAT 编码，保证得到**最小去序**（minimal deordering）。
3. **首次在 IPC 2023 Partial-Order HTN 基准上系统评估 HTN 计划去序**：与直接生成偏序计划的 Optiplan 对比，证明事后去序同样能有效减少约束并缩短关键路径。
4. **开源实现与实验协议**：论文提供了 HTN-PRF 与 HTN-MaxSAT 两个实现，以 PANDA 生成的序列计划为输入进行事后优化。

## 方法详解
### HTN-PRF 算法
- 输入：HTN 问题 $\mathcal{P}$ 和其解 $\langle \bar{a}, \prec, D_t \rangle$。
- 遍历原有序约束对 $(a_i, a_j)$，若约束属于分解树诱导的强制序 $\prec_H$，则保留；否则检查**简单并发**（simple concurrency）：
  - 若存在命题 $l$ 使得两动作在 add/pre、add/del 或 pre/del 上存在冲突（$a_i \# a_j$），则保留该约束。
  - 否则，移除该约束。
- 输出传递闭包 $\prec'^+$ 下的偏序 HTN 解，**不保证最小去序**。

### HTN-MaxSAT 去序
- 将每个约束对 $(a_i, a_j)$ 编码为布尔变量 $\kappa(a_i, a_j)$，加入 dummy 起始动作 $a_I$ 和目标动作 $a_G$。
- **硬子句（必须满足）**：
  - 分解强制序：$\kappa(a_i, a_j), \forall (a_i, a_j) \in \prec_H$（公式1）。
  - 无自环、传递闭包、动作在 $a_I$ 与 $a_G$ 之间（公式2-4）。
  - **因果正确性**：若 $a_i$ 为 $a_j$ 的 precondition $l$ 提供支持，则任何 delete $l$ 的动作不得插入其间（公式5-6）。
  - 禁止反转原始顺序：$\neg \kappa(a_j, a_i),\ 0 \leq i < j \leq k$（公式7）。
- **软子句（最大化满足以最小化约束数）**：对每个非强制序约束添加 $w(\neg \kappa(a_i, a_j)) = 1$（公式8），求解器将尽可能置 $\kappa = false$（即删除约束）。

## 实验与结果
- **数据集**：IPC 2023 Partial-Order HTN 基准（PCP、Rover、Satellite、Transport、Um-Translog 共 71 个实例）。
- **输入计划来源**：PANDA 规划器生成的序列计划。
- **对比基线**：Optiplan（Firsov 2025，直接生成偏序 HTN 计划的规划器）。
- **核心指标**：平均约束数、平均关键路径长度。
- **关键数字**：
  - **约束减少幅度（vs 原始全序计划）**：Satellite 域最高达 **27.67%**（HTN-MaxSAT）；Rover 域 26.92%；全部域合计 9.99%（HTN-MaxSAT）vs 9.84%（HTN-PRF）。
  - **关键路径缩短**：Rover 域最大降幅 27.57%（HTN-MaxSAT）；全部域合计 11.03%（HTN-MaxSAT）vs 10.91%（HTN-PRF）。
  - PCP 域完全有序，约束与关键路径均无改善。
  - **与 Optiplan 对比**：HTN-MaxSAT 约束均值（120.52）显著低于 Optiplan（137.90），但 Optiplan 生成的计划本身更大，直接可比性受限。
- **结论**：两种方法均显著减少约束；HTN-MaxSAT 略优于 HTN-PRF；效果高度依赖输入计划质量。

## 相关工作脉络
1. **Backström (1998)** — 经典规划中 PRF（Plan Relaxation by Feasibility）去序算法，本文将其扩展纳入 HTN 分解强制序。
2. **Muise, Beck & McIlraith (2012, 2016)** — 经典规划中基于 Partial Weighted MaxSAT 的最小去序方法，本文将其硬约束集合扩展以兼容 HTN 层级结构。
3. **Bercher, Haslum & Muise (2024)** — 计划优化综述，明确区分"计划生成"与"计划优化"，为本文定位提供理论框架。
4. **Optiplan（Firsov 2025）** — 直接生成偏序 HTN 计划的规划器，本文与其对比以验证事后去序的竞争力。
5. **PANDA（Holler 2023）** — 作为输入计划的生成器，其生成的序列质量直接影响去序效果。
6. **Geier & Bercher (2011)** — HTN 规划的形式化基础（task insertion 模型），本文以此为基础定义分解树与强制序。

## 局限性与未来方向
1. **依赖输入计划质量**：去序效果与 PANDA 生成的原始序列计划质量强相关；更难实例往往无解，限制了整体提升。
2. **PCP 域完全无效**：在该域中分解约束覆盖了全部必要序，去序算法无法消除任何约束。
3. **仅考虑简单并发**：HTN-PRF 使用最简单的并发判据，可能存在更精细的因果链接分析空间。
4. **未涉及时间维度**：当前方法不考虑行动持续时间与时间窗口，仅处理逻辑序约束。
5. **未来方向**：论文暗示可扩展至时序 HTN 规划（temporal HTN），结合 Optiplan 的在线偏序生成与本文的事后优化进行混合策略研究。

## 研究启发与可借鉴点
1. **MaxSAT 编码范式可迁移**：将去序问题转化为 Partial Weighted MaxSAT 的形式化框架（硬子句保 correctness、软子句求最优）可直接迁移到其他规划 formalism 的偏序优化场景。
2. **"保持强制序 + 移除可选序"的设计思路**：先识别由领域结构（如层次分解、资源独占）诱导的不可消除约束，再对剩余约束做优化，是一种通用且安全的方法论。
3. **与现有条目规划器结合**：将本文事后去序管线嵌入 PANDA/Optiplan 等规划器内部，可作为 plan post-processing 模块，无需修改规划器核心逻辑。
4. **IPC 基准可直接复用**：IPC 2023 Partial-Order HTN benchmark 是评估 HTN 偏序质量的标准集，后续研究可直接在此框架下对比。
5. **关键路径作为辅助指标**：除约束数量外，关键路径长度更能反映实际执行效率增益，建议后续工作将其作为主要评估指标之一。

## 关键术语表
- **HTN（Hierarchical Task Network）规划**：基于任务递归分解的规划形式化，通过方法（methods）将抽象任务展开为可执行_primitive_动作序列。
- **Plan Deordering（计划去序）**：在保持计划合法性的前提下，移除序列计划中不必要的排序约束，使其变为偏序计划以提升并行执行能力。
- **Simple Concurrency（简单并发）**：两动作因 add/pre、add/del 或 pre/del 存在命题层面的冲突，则视为不可并发，必须保留其顺序。
- **Minimal Deordering（最小去序）**：在偏序计划中无法再移除任何额外排序约束而不破坏合法性的去序结果。
- **Partial Weighted MaxSAT**：一种 SAT 变体，要求所有硬子句（hard clauses）必须满足，同时最大化满足的软子句（soft clauses）权重之和。
- **Decomposition Tree（分解树）**：描述从初始抽象任务到primitive动作的完整递归展开结构的树形表示，其边关系构成强制偏序 $\prec_H$。
- **Critical Path Length（关键路径长度）**：偏序计划中最长链式动作序列的长度，反映计划的最小串行执行耗时下界。
- **Optiplan**：Firsov (2025) 提出的 HTN 规划器，直接在搜索过程中生成偏序计划，无需事后去序。

## 可复现要素
- **数据集**：IPC 2023 Partial-Order HTN benchmarks（由 IPC 官方发布，公开可获取）。
- **代码开源**：论文标注了实现来源（脚注1），但正文未提供 GitHub 链接；需联系作者获取。
- **基线规划器**：PANDA（Holler 2023，开源）用于生成输入序列计划；Optiplan（Firsov 2025）作为对比基线。
- **关键超参**：实验环境为 Linux + Intel Core Ultra 9 185H / 32GB RAM；每实例上限 30 分钟运行时间、10GB 内存；未提及具体 MaxSAT 求解器版本。
