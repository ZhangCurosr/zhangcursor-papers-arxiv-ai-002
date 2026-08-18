---
title: "SkillShapley-Boundary-Adaptive-Shapley-Valuation-for-Skill-S"
source: https://arxiv.org/pdf/2608.13173v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 04:16:01"
field: "LLM Agent 可解释性与归因"
keywords: ["Shapley Value", "LLM Agent", "Skill Attribution", "Cooperative Game", "Prompt Optimization", "Explainable AI"]
innovations: ["提出 SkillShapley 将技能步骤归因建模为合作博弈 Shapley 值问题", "设计 BAES 缓存感知两阶段近似算法高效逼近精确 Shapley 值", "验证 Shapley 归因的行为意义并提供技能剪枝/修订/创建的工作流指导"]
benchmarks: ["SkillsBench"]
---

# 论文速读：SkillShapley-Boundary-Adaptive-Shapley-Valuation-for-Skill

## 一句话总结
论文提出 SkillShapley 框架，将 LLM Agent 技能的步骤级贡献建模为合作博弈中的 Shapley 值问题，并设计 BAES（Boundary-Adaptive Edge Shapley）方法，通过缓存感知的两阶段自适应采样，以较少配置评估成本逼近精确 Shapley 归因，进而指导技能剪枝、修订与创建。

## 研究问题与动机
- **核心问题**：Agent 技能通常由多步自然语言指令构成，但现有工作仅评估端到端技能性能，缺乏对单个步骤贡献的细粒度归因，导致技能设计依赖试错，无法精准识别冗余或关键步骤。
- **现有方法不足**：Prompt 优化/压缩方法作用于 token 级别，无法直接归因语义步骤；工作流优化改进执行结构但不量化步骤价值；精确 Shapley 枚举在步骤数增多时计算不可行。
- **实证动机**：在低步骤数技能中，精确 Shapley 排名产生差异化步骤值，而 Individual/LOO 基线产生大量并列；移除 Shapley 高排名步骤导致显著性能下降，验证其行为意义。
- **计算瓶颈**：技能配置评估成本高（需运行 LLM Agent），核心开销是新配置执行次数而非算术比较，需利用已缓存配置的 one-flip 边复用。

## 核心贡献（创新点）
1. **建立技能步骤归因的合作博弈形式化框架**：将固定技能划分为 n 个语义步骤作为玩家，保留步骤子集构成联盟，基准性能作为效用函数，首次明确定义步骤级 Shapley 归因目标。与已有 Prompt/token 级归因方法的本质区别在于以语义完整步骤为单位，而非文本片段。
2. **提出 BAES 缓存感知近似算法**：设计 warmup 全覆盖+自适应采集两阶段策略，优先评估能生成最多高优先级 one-flip 边的新配置，复用缓存边减少重复计算。与传统 Monte Carlo/Quasi-Monte Carlo Shapley 估计器的本质区别在于利用配置缓存结构而非均匀随机采样，适应技能评估的成本模型。
3. **验证 Shapley 归因的行为意义并提供技能工程指导**：在 SkillsBench 上证明 Shapley 排名移除曲线下降最快，且归因模式揭示高价值步骤是"程序性桥接"角色；为技能剪枝、修订、创建提供可操作的工作流（假设-编辑-验证循环）。与既有技能优化工作的本质区别在于基于归因证据驱动编辑，而非黑盒式优化。

## 方法详解
- **问题形式化**：技能 $X = (x_1, \ldots, x_n)$ 划分为 n 个语义步骤，联盟 $S \subseteq N$ 对应保留步骤子集 $X_S$（保持原顺序），效用 $v(S) = \frac{1}{M}\sum_{j=1}^M r(o_j(S), y_j)$ 为基准成功率的经验估计。目标为 Shapley 值 $\phi_i$（式5/6）。
- **BAES 两阶段设计**：
  - **Warmup 阶段**：从锚点配置（空集、全集、所有单点、所有 (n-1) 子集）开始构建缓存 D，然后贪心选择能暴露最多已观测 stratum 的新配置（TopPriority 规则，每轮最多3个），直到 stratum 排名稳定性达标（Kendall's τ 指数移动平均低于峰值的 1/e，至少60个 stratum）。
  - **Adaptive 阶段**：对未评估配置 C，计算获取分数 $A(C) = \sum_{i: C \triangle \{i\} \in \mathcal{D}} a_{i,k} \cdot b(v(C \triangle \{i\}))$，其中 stratum 分配分数 $a_{i,k} = \sqrt{\widehat{\sigma}_{i,k}^2 + \epsilon}/\sqrt{m_{i,k}+1}$，缓存奖励权重 $b(y) = 1 + \max\{0, \text{round}((1-y)/g)\}$ 偏好低奖励邻域。每次选取 $A(C)$ 最大配置评估并更新缓存。
- **停止规则**：监测归一化标准误差 NSE 轨迹斜率，当最近 decorrelation window 内斜率非递减且至少85个 stratum 被观测时停止；预算上限 B 为总评估次数。
- **估算输出**：$\widehat{\phi}_i = \frac{1}{n}\sum_{k=0}^{n-1} \widehat{\mu}_{i,k}$，缺失 stratum 用同 size 均值填充或0。

## 实验与结果
- **数据集**：SkillsBench 三个低步骤技能——ofer-letter-generator/docx（n=10，1024精确配置）、manufacturing-fjsp-optimization/fjsp-baseline-repair（n=9，512配置）、dialogue-parser/dialogue-graph（n=11，2048配置）。每个配置在3个基准实例上评估，奖励离散为 {0, 1/3, 2/3, 1}，g=1/3。Agent harness 为 OpenHands，temperature T=0。
- **基线方法**：Individual（单步评估）、Leave-One-Out（删除单步）、Random Removal、LeastCore、Monte Carlo Shapley、Quasi-Monte Carlo Shapley、paired MC Shapley、size-k-truncated Shapley。
- **主要结果**：
  - **行为有效性**（Figure 2）：Shapley 移除曲线在三个任务上下降最快，显著优于 Individual/LOO/Random/LeastCore。
  - **近似精度**（Figure 3）：BAES 在相同 unique-configuration 预算下达到更低的 MAE（相对精确 Shapley），在 B=3n² 预算下优于所有 MC 变体。
  - **缓存效率**：10玩家技能中，99预算下 BAES Phase 1 产生206条可复用 one-flip 边，MC permutation 仅130条（其中115唯一）。
  - **Token 成本**（Figure 4）：token 数不严格随 coalition size 单调，说明归因编辑不一定成比例节省 token，但可精准移除低价值内容。
- **最强结果**：BAES 在匹配预算下近似误差最低，且缓存边复用率提升约 79%（206 vs 115 唯一边）。

## 相关工作脉络
- **Agent Skill 优化**（SkillsBench、SkillAxe、SkillReducer、SkCC）：本文定位差异在于前述工作将技能视为整体进行优化/压缩/编译，本文首次在固定技能内对语义步骤进行 Shapley 归因。
- **Instruction Attribution**（SHAP-based、TokenSHAP、LLMLingua）：本文差异在于归因单位是完整语义步骤而非 token/span，且目标是从基准性能视角量化贡献而非解释模型内部注意力。
- **Shapley 估计器**（Monte Carlo、Stratified、FastSHAP、paired MC）：本文差异在于针对技能评估"新配置成本高、缓存边可复用、奖励离散"的独特成本模型设计 BAES，而非通用特征归因场景。
- **Cooperative Game 理论**（Shapley 原始工作、LeastCore）：本文应用 Shapley 于新领域（LLM 技能步骤），并解决 exact enumeration 不可行的近似问题。

## 局限性与未来方向
- **步骤数限制**：精确 Shapley 仅在低步骤数技能可行，BAES 适用于中等规模但超高维技能仍需验证。
- **动态流程不适用**：步骤数可变或强耦合流水线（移除中间步骤导致整个流程崩溃）使合作博弈定义模糊，Shapley 值可能反映结构必要性而非步骤实用性。
- **主观评估挑战**：高度依赖人工评分或模糊标准的任务难以定义稳定效用函数 v(S)。
- **多技能场景未探索**：当前局限于单技能归因，未来可扩展至多技能 Agent pipeline。

## 研究启发与可借鉴点
- **缓存感知的 Shapley 近似设计**：BAES 利用 one-flip 边复用的思想可迁移至其他高成本配置评估场景（如超参搜索、A/B 测试），将"评估新配置"与"复用边际边"分离处理。
- **离散奖励下的不确定性度量**：以奖励粒度 g 归一化标准误差（NSE）作为停止准则，避免对低于分辨力的数值变化过度敏感，适用于任何离散/粗粒度奖励的系统。
- **两步策略（覆盖+ exploitation）**：Warmup 先建立 stratum 覆盖再进入自适应采集，避免早期随机波动锁定窄 stratum，可借鉴于主动学习或贝叶斯优化中的探索-利用平衡。
- **技能工程工作流**："假设-归因-编辑-验证"的闭环为技能自动化提供了可操作范式，可与本团队的方向（如自动 prompt 生成/优化）结合。
- **行为验证协议**：Top-ranked 移除曲线作为归因有效性的诊断工具，可复用于其他归因方法的对比评估。

## 关键术语表
**SkillShapley**：基于 Shapley 值的 LLM Agent 技能步骤级归因框架。
**BAES (Boundary-Adaptive Edge Shapley)**：一种缓存感知的两阶段 Shapley 近似算法，通过 warmup 覆盖和自适应采集高效逼近精确值。
**Shapley Value**：合作博弈中衡量每个玩家边际贡献期望的值，满足效率、对称性、零贡献、可加性公理。
**Stratum (i, k)**：按玩家 i 和联盟大小 k 划分的边际贡献子空间，BAES 在此维度上统计均值和方差。
**One-flip Edge**：两个仅相差一个步骤的联盟之间的比较边，BAES 缓存并复用这些边以减少重复评估。
**Coalition**：联盟，即保留特定步骤子集的技能变体，对应合作博弈中的一个玩家子集。
**Reward Granularity (g)**：基准奖励的最小可分辨增量，用于归一化不确定性和计算缓存奖励权重。

## 可复现要素
- **数据集**：SkillsBench（公开基准，论文引用 [11]）
- **代码/权重**：论文未提及开源声明
- **关键超参**：warmup 预算 R = ⌊0.4B⌋，总预算 B = 3n²；奖励粒度 g=1/3；temperature T=0；ankor 配置包含空集、全集、单点、(n-1)子集；TopPriority 每轮最多3个配置；排名稳定性阈值 1/e；最小观测 stratum 数60（warmup）/85（adaptive）；NSE 停止规则中的 δ=10⁻⁸
- **Agent Harness**：OpenHands
- **评估配置**：每个配置在3个基准实例上评估，奖励取值 {0, 1/3, 2/3, 1}
