---
title: "Towards-Numerical-TOHTN-Planning-with-SMT-based-HTN-SAT-Enco"
source: https://arxiv.org/pdf/2609.03938v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 23:34:44"
field: "层次任务网络规划与数值推理"
keywords: ["HTN planning", "SMT encoding", "numerical planning", "TOHTN", "cPDT", "SibylSmt", "benchmark"]
innovations: ["将HTN-SAT布尔编码扩展为SMT以支持数值变量与约束", "提出首个数值TOHTN规划形式化定义", "发布七个数值TOHTN基准家族及实例生成器"]
benchmarks: ["Alchemy", "Backpack", "Gripper-colors", "Minecraft", "Overcooked", "Transport-Fuel", "Transvasement", "Coverage"]
---

# 论文速读：Towards-Numerical-TOHTN-Planning-with-SMT-based-HTN-SAT-Enco

## 一句话总结
本文提出将现有SAT-based全序HTN（TOHTN）规划器的布尔编码扩展为SMT编码，首次系统地支持数值变量与约束推理，并提供了首个数值TOHTN基准测试套件。实验表明，这一简单扩展已构成有竞争力的数值HTN规划基线。

## 研究问题与动机
- **HTN规划缺乏数值推理能力**：现有主流HTN规划器几乎全部不支持数值约束（如资源、成本、时间），限制其在物流、机器人、调度等真实场景的应用。
- **仅有两个数值HTN规划器**：已知支持数值推理的HTN规划器仅有Siadex和Aries，覆盖严重不足。
- **SAT-based方法更易扩展**：相比启发式搜索方法需要大幅改造，SAT-based HTN-SAT编码器可通过"升格到SMT"自然支持数值变量，无需改变搜索框架。
- **缺乏统一评测基准**：数值HTN规划领域没有公开的基准测试套件，阻碍了系统比较与后续研究。

## 核心贡献（创新点）
1. **首个数值TOHTN规划的形式化定义**：在PDDL2.1/HDDL 2.1风格下定义含数值流体的TOHTN问题与解，填补形式化空白。
2. **SAT→SMT的增量编码扩展**：将现有HTN-SAT的布尔编码"原地"升级为SMT编码，仅需为每个节点-数值流体对引入算术变量，并保持原布尔子句不变。
3. **七大家族数值TOHTN基准套件**：提供TRANSPORT-FUEL、TRANSVASEMENT等七个基准家族及实例生成器，所有数据与代码开源。
4. **SibylSmt规划器与实验基线**：基于Z3求解器实现SMT编码的TOHTN规划器SibylSmt，与Siadex、Aries对比验证其有效性。
5. **简洁而有效的数值框架公理**：通过( Pf ≠ Pf⁺ )⇒(¬Pprim ∨ ⋁Pa)的框架公理，精确刻画数值流体仅在对应动作执行时改变。

## 方法详解
**结构基础：Compact Path Decomposition Tree (cPDT)**
- cPDT是在AND/OR树之上引入显式时间步的结构：根节点含初始抽象任务cI；抽象任务的每个分解方法产生按顺序排列的子任务子节点；原语任务则被传播到第一个子节点，其余子节点填ε（空操作）。
- 从左到右扫描叶节点即得到候选计划序列。

**布尔编码保留部分**
- 初始任务/状态/目标公理（仅在首次编码时生成）
- 叶节点任务唯一性约束：⋁t P_t，且两两互斥
- 原语任务的前提与效果传播：Pa ⇒ ⋀p∈precond(P_p)，Pa ⇒ ⋀p∈effect⁺(P⁺_p)等
- 框架公理：¬Pp ∧ P⁺_p ⇒ (¬Pprim ∨ ⋁a p∈effects(a) Pa)
- 层级约束：节点P与其第一个子节点P¹的命题值相同；抽象任务激活时恰选一个方法展开
- 叶节点全为原语任务：⋀_{P∈leaves} Pprim

**SMT数值扩展部分**
- 对每个节点P和数值流体f∈F，引入实数值变量Pf，表示该节点处f的取值。
- 初始数值状态：∀f∈F: Rf = vI(f)
- 数值目标：∀ξ∈gN: ⟦ξ⟧G成立
- 数值前提：∀a, ∀ξ∈precondN(a): Pa ⇒ ⟦ξ⟧P
- 数值效果（赋值）：∀(f=ξ)∈effectN(a): Pa ⇒ (P⁺_f = ⟦ξ⟧P)
- 数值框架公理：(Pf ≠ P⁺_f) ⇒ (¬Pprim ∨ ⋁{a | f∈effectN(a)} Pa)
- 数值层级约束：∀f∈F: Pf = P¹_f

**增量生成机制**
- 初始/目标数值约束一次性生成；数值前提/效果/框架公理随cPDT扩展在新增叶节点上增量生成。
- 所有布尔子句保持不变，只需在求解器中添加数值变量与约束。

## 实验与结果
- **实现**：SibylSmt，基于Z3 SMT求解器 + BFS cPDT扩展策略。
- **基线**：Siadex（SHOP衍生，约束推理）、Aries（混合CP/SAT）。
- **硬件**：Intel Core i7-12700H，32GB RAM，单实例时限10分钟。
- **基准**：7个数值TOHTN领域，每领域20个实例（Coverage领域140个实例）。

| 领域 | SibylSmt | Aries | Siadex |
|------|----------|-------|--------|
| Alchemy 20 | 10.54 | 7.79 | 12.82 |
| Backpack 20 | 9.99 | 4.59 | 0 |
| Gripper-colors 20 | 9.54 | 7.82 | 6.81 |
| Minecraft 20 | 9.11 | 4.09 | 19.91 |
| Overcooked 20 | 12.58 | 0 | 1 |
| Transport-Fuel 20 | 7.89 | 1.61 | 0 |
| Transvasement 20 | 7.17 | 5.82 | 0 |
| Coverage 140 | 91 | — | 42 |

- **核心结论**：
  - SibylSmt在多数领域获得更高覆盖率与敏捷得分（agile score），整体鲁棒性更强。
  - Siadex在递归分支强烈的领域（如Backpack）表现差，因方法选择难以被当前状态引导。
  - 所有规划器均未饱和全部实例，说明该方向仍有较大提升空间。

## 相关工作脉络
1. **Schreiber et al. (2019) TreeRex / Behnke et al. (2018) totSAT**：提出HTN-SAT增量编码框架与cPDT结构，本文在其布尔编码基础上直接升级为SMT。
2. **Castillo et al. (2006) Siadex**：最早支持数值约束的HTN规划器之一（基于SHOP），采用约束传播而非SAT/SMT，本文在其基础上做系统对比。
3. **Bit-Monnot (2023) Aries**：混合CP/SAT规划器，支持数值推理，代表另一条技术路线；本文展示纯SMT扩展方案的竞争力。
4. **Quenard et al. (2024) SibylSat / (2025) SibylSatOpt**：本文作者先前工作，分别将SAT/SMT扩展至贪心搜索与MaxSAT最优搜索；本文是当前工作的基线版本。
5. **Pellier et al. (2022) HDDL 2.1**：提出含时间的HTN形式化，本文的数值流体定义沿袭其风格。
6. **Fox & Long (2003) PDDL2.1**：经典数值规划形式化，本文数值前提/效果的语法与此一致。

## 局限性与未来方向
- **仅支持全序（TO）分解**：未扩展至任意序（partial-order）HTN规划。
- **无最优性保证**：当前为BFS+SAT可行性搜索，未集成成本优化（后续SibylSatOpt已在贪心/MaxSAT方向探索）。
- **数值表达式受限**：仅支持赋值型数值效果（f := ξ），不支持条件赋值、增强赋值等更丰富操作。
- **缺少时间维度**：未结合持续时间、时序约束，与HDDL 2.1等含时间形式化距离较远。
- **基准规模有限**：目前仅七个领域，需更多真实应用案例验证。

## 研究启发与可借鉴点
1. **"升格编码"而非"重写求解"**：将SAT编码直接升级SMT，避免重建搜索框架，是向更 expressive 逻辑迁移的可复用策略。
2. **cPDT结构可复用**：该结构将HTN分解与时间步显式化，是后续扩展至部分序、时间、成本等维度的良好骨架。
3. **数值框架公理的简洁形式**：(Pf ≠ P⁺_f) ⇒ (¬Pprim ∨ ⋁Pa) 以一条蕴含式精确刻画"值变必由动作引起"，可推广至带条件效果、增赋值的扩展。
4. **增量生成机制**：初始/目标约束一次性生成，中间约束随cPDT扩展增量添加，兼顾求解效率与内存控制，值得在后续最优/启发式变体中沿用。
5. **开源基准的代码友好**：实例生成器随论文开源，为社区提供统一比较平台，是推进领域发展的标准做法。

## 关键术语表
- **HTN (Hierarchical Task Network)**：层次任务网络规划，通过递归分解抽象任务生成执行序列的规划范式的总称。
- **TOHTN (Totally-Ordered HTN)**：全序HTN，要求方法内子任务严格按顺序执行的HTN子集。
- **cPDT (Compact Path Decomposition Tree)**：压缩路径分解树，将HTN的AND/OR分解树与显式时间步结合的结构，是HTN-SAT编码的核心载体。
- **SMT (Satisfiability Modulo Theories)**：模理论可满足性，在布尔SAT基础上加入算术、数组等理论求解器，支持数值约束推理。
- **Numeric fluent (数值流体)**：取实数值的状态变量，区别于布尔命题，用于建模资源、数量、成本等连续量。
- **Frame axiom (框架公理)**：描述"未改变的量保持不变"的逻辑约束，在编码中避免全量列举未变变量。
- **Agile score (敏捷得分)**：综合覆盖率和求解速度的评测指标，定义为min(1, 1 - log(t)/log(T))，t为实际求解时间，T为时限。
- **Incremental encoding (增量编码)**：随cPDT扩展逐步添加约束而非全量重编码，以降低重复计算与内存消耗。

## 可复现要素
- **数据集**：七个数值TOHTN基准家族（Alchemy, Backpack, Gripper-colors, Minecraft, Overcooked, Transport-Fuel, Transvasement, Coverage），含实例生成器，**论文声明代码与数据已开源**（arxiv来源页面通常附GitHub链接）。
- **代码**：SibylSmt规划器源码与基准生成器应随论文开源，具体仓库地址需查阅论文补充材料或arXiv页面。
- **SMT求解器**：Z3（De Moura & Bjørner, 2008）。
- **关键超参**：论文未提及具体超参数（如Z3策略、BFS节点展开上限等），仅声明单次实例时限10分钟、CPU配置i7-12700H/32GB RAM。
