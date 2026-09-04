---
title: "Lose-the-Order-Keep-the-Hierarchy-Deordering-HTN-Plans"
source: https://arxiv.org/pdf/2609.03912v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 12:04:13"
---

# 论文速读：Lose-the-Order-Keep-the-Hierarchy-Deordering-HTN-Plans

## 一句话总结
本文首次将经典规划中的计划去序（deordering）技术适配至层次任务网络（HTN）规划场景，提出HTN-PRF与HTN-MaxSAT两种算法，在严格保留层级分解强制序的前提下，有效移除冗余动作序约束，显著提升部分序计划的执行灵活性与并发潜力。

## 研究问题与动机
- HTN规划研究长期聚焦于计划生成（plan generation），对已生成计划的后期优化（post-plan optimization）关注明显不足，尤其是计划去序方向存在研究空白。
- 经典STRIPS规划中成熟的去序方法（如PRF、MaxSAT）未考虑HTN特有的任务分解层级结构，直接套用会破坏由分解方法诱导的强制序约束$\prec_H$，导致计划失效。
- 实际调度场景（如消防任务）中，全序计划会引入不必要的串行限制；若两动作互不干扰，移除其序约束可提升并行执行效率与系统灵活性。
- 现有HTN规划器（如Optiplan）虽能直接输出部分序计划，但缺乏针对已有全序计划的精细化后期去序流程，且本文旨在填补HTN去序理论的空白。

## 核心贡献（创新点）
1. **首次系统定义HTN计划去序问题**：将去序概念从经典规划拓展至带层级分解约束的HTN形式化体系，明确区分“层级强制序”与“可移除冗余序”。
2. **提出HTN-PRF启发式去序算法**：在Backström经典并发检查基础上引入$\prec_H$判断，计算高效但不保证全局最优。
3. **提出HTN-MaxSAT精确去序算法**：将去序建模为Partial Weighted MaxSAT问题，通过硬子句严格保证层级序、因果支撑与顺序不可逆，通过软子句最大化冗余序移除量，保证输出为给定输入下的最小去序（minimal deordering）。
4. **构建完整评测体系并与Optiplan对比**：在IPC 2023 Partial-Order HTN基准上验证两种方法的有效性，揭示后处理去序与直接生成部分序策略的差异与互补性。

## 方法详解
- **形式化基础**：采用Geier & Bercher (2011)的HTN定义，计划表示为$(\bar{a}, \prec, D_t)$，其中$\prec_H \subseteq D_t$为分解方法诱导的强制序约束，去序过程中必须完整保留。
- **HTN-PRF算法**：遍历原计划所有序对$(a_i, a_j)$。若$(a_i, a_j) \in \prec_H$则强制保留；否则进行简单并发检测（simple concurrency），即检查是否存在命题$l$满足以下任一冲突：$l \in add(a_i) \land l \in pre(a_j)$、$l \in add(a_i) \land l \in del(a_j)$、$l \in pre(a_i) \land l \in del(a_j)$。若触发非并发条件则保留，否则移除该序约束。最终返回$\prec$的传递闭包$\prec'^+$。
- **HTN-MaxSAT算法**：引入布尔变量$\kappa(a_i, a_j)$表示保留$a_i \prec a_j$的序约束。
  - **硬子句**：(1) 强制保留所有$\prec_H$约束；(2) 禁止自环$\neg\kappa(a_i, a_i)$；(3) 保证每动作位于起止虚拟动作$a_I, a_G$之间；(4) 传递性$\kappa(a_i,a_j) \wedge \kappa(a_j,a_k) \to \kappa(a_i,a_k)$；(5)-(6) 因果正确性：每个前置条件$l$必须由某添加者$a_i$支撑，且其间不能有删除$l$的动作$a_k$；(7) 禁止反转原始计划中的相对顺序。
  - **软子句**：对每个非层级诱导且非起止关联的序约束添加$w(\neg\kappa(a_i, a_j))=1$，目标为最大化满足的软子句总数（即最大化移除的冗余约束）。求解器输出满足全部硬约束的最优解，保证结果的最小性。

## 实验与结果
- **数据集与环境**：IPC 2023 Partial-Order HTN基准（PCP、Rover、Satellite、Transport、Um-Translog），共71个实例。测试平台为Intel Core Ultra 9 185H / 32GB RAM，单实例限时30分钟、限内存10GB。
- **基线与评估指标**：以PANDA生成的全序计划为输入，对比HTN-PRF（$H_P$）、HTN-MaxSAT（$H_M$）与直接生成部分序的Optiplan（$OP$）。指标为平均序约束数量与平均临界路径长度，并计算相对初始全序计划的改进百分比（%Diff）。
- **主要结果**：
  - **序约束减少**：$H_M$在所有领域均优于或持平$H_P$。相比初始计划，ALL领域平均减少约9.84%（$H_P$）与9.99%（$H_M$）；Satellite领域降幅最大（27.67%）；Rover领域$H_M$比$H_P$多移除近9%的约束。$OP$因生成的计划规模更大，保留的约束数反而高于两种去序方法。
  - **临界路径缩短**：$H_M$与$H_P$表现接近，ALL领域平均缩短约10.91%~11.03%；Rover领域缩短最多（$H_P$: 24.32%，$H_M$: 27.57%）。PCP领域完全有序，无任何缩减。
  - **结论**：两种方法均能显著降低序约束与临界路径，但去序收益高度依赖输入全序计划的质量；较难实例潜在收益更大，但PANDA在部分难实例上未能求解，限制了全覆盖验证。

## 相关工作脉络
- **Backström (1998)**：提出经典规划的PRF去序算法，仅依赖简单并发判断。本文将其适配HTN，本质区别是必须额外将分解树强制序$\prec_H$作为不可移除的硬约束。
- **Muise et al. (2012, 2016)**：提出基于Partial Weighted MaxSAT的经典规划最优去序方法。本文扩展其编码，关键差异在于引入层级序与HTN因果支撑链，确保去序不破坏任务细化逻辑。
- **Optiplan (Firsov, 2025)**：直接生成部分序HTN计划的规划器。本文与之对比，指出Opt
