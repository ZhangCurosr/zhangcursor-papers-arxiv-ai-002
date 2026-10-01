---
title: "SKILLADAM-STABLE-AND-EFFICIENTSKILL-EVOLUTION-FOR-AGENTS"
source: https://arxiv.org/pdf/2609.08944v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 03:28:00"
field: "Agent Skill 优化与自演化"
keywords: ["Agent Skills", "Skill Self-Evolution", "Prompt Optimization", "Discrete Optimization", "Adam-inspired Framework"]
innovations: ["将Adam一阶/二阶矩思想功能性地迁移到离散技能空间，设计Evolving Issue Tracker和volatility-driven edit budget", "在七个benchmark上实现SOTA性能的同时将优化Token消耗降低约67%", "通过跨模型迁移和Judge评估证明技能质量与泛化能力的双重提升"]
benchmarks: ["SearchQA", "SpreadsheetBench", "OfficeQA", "DocVQA", "LiveMathematicianBench", "ALFWorld", "DeepPlanning"]
---

# 论文速读：SKILLADAM-STABLE-AND-EFFICIENT SKILL-EVOLUTION-FOR-AGENTS

## 一句话总结
本文提出 SKILLADAM，一种受 Adam 优化器启发的框架，通过在离散技能空间中维护"优化记忆"（Evolving Issue Tracker）和"波动驱动的编辑预算"（volatility-driven edit budget），实现 Agent 技能的稳定且高效自演化，在七个 benchmark 上达到 SOTA 性能的同时显著降低优化成本。

## 研究问题与动机
- **高质量技能获取成本高**：Expert-written skills 依赖人工编写，耗时长；LLM 直接生成的 skills 通常需进一步打磨（SkillAxe 报告）。
- **现有自演化方法优化不稳定**：既有迭代优化方法（如 SkillAxe、SkillOpt）缺乏对"改什么"和"改多少"的可靠控制，易产生迭代局部反馈导致的反复覆盖问题。
- **方向稳定性（Direction Stability）缺失**：每轮仅基于有限样本获得反馈，不同轮次的修改可能针对不同问题，导致有效修正被后续迭代覆盖。
- **更新自适应（Update Adaptivity）缺失**：修改幅度应与近期案例级改进的一致性相匹配，但现有方法缺乏对改进波动性的估计与响应。

## 核心贡献（创新点）
1. **提出 SKILLADAM 框架**：将 Adam 的优化思想功能性地迁移到离散技能空间，而非直接数值计算梯度，适用于不可微的自然语言指令文档优化。
2. **设计 Evolving Issue Tracker（EIT）**：作为 Adam 一阶矩的函数类比，维护问题列表及其求解历史，稳定更新方向，避免重复发现已知失败模式。
3. **设计 volatility-driven edit budget**：作为 Adam 二阶矩的函数类比，通过追踪近期案例级改进的波动性自适应控制每次修改的幅度。
4. **七项 benchmark 上的 SOTA 性能**：在短程和长程任务上均取得最优或次优结果，DP-Avg 相对 SkillOpt 提升 6.7pp，同时优化阶段 Token 消耗减少约 67%，API 请求减少约 69%。

## 方法详解
- **初始技能构建**：从执行轨迹和评估反馈中构建初始技能 $S_0$，作为迭代起点。
- **Rollout 与反馈收集**：每轮 $t$ 采样 mini-batch $B_t$，使用当前技能 $S_{t-1}$ 执行 Agent，得到轨迹 $\mathcal{T}_t$ 和评估反馈 $\mathcal{F}_t^{\mathrm{roll}}$。
- **Evolving Issue Tracker（优化记忆）**：
  - 结构：$M_t = \{I_j : I_j = (p_j, z_j, \mathcal{A}_j)\}$，其中 $p_j$ 为错误模式，$z_j$ 为当前状态，$\mathcal{A}_j$ 为既往求解尝试及结果。
  - 更新：$\mathcal{U}_{\mathrm{EIT}}$ 比较 roll-out 与 validation 结果，链接新失败到已有问题、创建新问题、记录尝试结果，失败重现时重新打开问题。
  - 功能类比：等价于 Adam 的一阶矩 $m_t^{\mathrm{Adam}}$，累积历史信息以稳定更新方向。
- **波动驱动的编辑预算**：
  - 案例级改进：$\delta_{t,i} = s(\mathcal{E}_{t,i}^{\mathrm{val}}) - s(\mathcal{E}_{t,i}^{\mathrm{roll}})$。
  - 波动估计：$\widehat{V}_t = \mathrm{Var}(\{\delta_{t,i}\})$，历史加权更新 $V_t = \beta_2 V_{t-1} + (1-\beta_2)\widehat{V}_t$。
  - 编辑预算：$\sigma_{t+1} = \max(b_{\min}, \lfloor b_{\mathrm{base}}[1 - \mathrm{clip}(V_t/V_{\max}, 0, 1)]\rfloor)$，高波动时缩小预算，低波动时允许更大修改。
  - 功能类比：等价于 Adam 的二阶矩 $v_t^{\mathrm{Adam}}$，自适应调整更新幅度。
- **技能更新与接受门控**：
  - Patch 生成：$g_t = \mathcal{G}_{\mathrm{LLM}}(S_{t-1}, \mathcal{T}_t, \mathcal{F}_t^{\mathrm{roll}}, M_{t-1}, \sigma_t)$，在记忆和预算约束下生成修改。
  - 候选评估：$\mathcal{F}_t^{\mathrm{val}} = \mathcal{F}(\widetilde{S}_t, B_t)$，在同一 batch 上对比。
  - 接受门控：$a_t = G_\theta(S_{t-1}, \widetilde{S}_t, \mathcal{F}_t^{\mathrm{roll}}, \mathcal{F}_t^{\mathrm{val}}) \in \{0,1\}$，主指标达到阈值且保护指标未退化时才接受。

## 实验与结果
- **Benchmark 设置**：5 个短程（SearchQA、SpreadsheetBench、OfficeQA、DocVQA、LiveMathematicianBench）+ 2 个长程（ALFWorld、DeepPlanning），共 10 个评估切片。
- **基线方法**：NoSkill、HumanSkill、LLMSkill、Trace2Skill、TextGrad、GEPA、SkillOpt。
- **主要结果**：
  - 短程：SKILLADAM 在 SearchQA（87.5%）、Spreadsheet（81.1%）、DocVQA（92.3%）、LiveMath（67.7%）取得最佳，OfficeQA 与 SkillOpt 并列 72.1%。
  - 长程：ALFWorld 89.6%（最优），DP-Shopping 45.0%、DP-Travel 11.7%（SkillOpt 分别为 41.7%、1.7%），DP-Avg 28.3% vs SkillOpt 21.7%（+6.7pp）。
  - 相对 LLMSkill 平均提升 31.20%，相对 HumanSkill 平均提升 14.45%。
- **消融实验**：去掉双机制（M+B→无）后 DP-Avg 从 28.3% 降至 19.2%；仅去预算（-B）降至 21.7%；仅去记忆（-B,-M）降至 19.2%，双机制互补。
- **跨模型迁移**：GPT-5.5 训练技能直接部署到 GPT-5.4-mini，SKILLADAM 平均保留率 81.3% vs SkillOpt 76.2%，平均目标得分 67.8% vs 63.1%。
- **训练成本**：DeepPlanning 上 SKILLADAM 总 Token 消耗 74.0M vs SkillOpt 226.6M（-67.3%），API 请求 2,830 vs 9,071（-68.8%），以每百万 Token 的 DP-Avg 计效率约为 SkillOpt 的 4 倍。
- **优化动态**：Figure 4 显示 SKILLADAM 首轮即找到强修订且后续保持稳定，SkillOpt 接受决策波动较大。

## 相关工作脉络
- **Agent Skills 与 Skill 构造**：Trace2Skill 从执行轨迹合成统一技能目录；SkillAxe 迭代诊断并精炼 LLM 生成技能；CoEvoSkills 联合进化技能生成器与验证器；SkillOS 训练策展人维护外部技能库；本文聚焦单一自然语言技能文档的迭代优化，与 SkillOpt 设定最接近。
- **Prompt 优化**：OPRO 从历史候选生成新指令；ProTeGi 将错误反馈转为文本梯度；TextGrad 通过计算图传播文本反馈；GEPA 利用回退反馈进化 prompt；本文将类似思想迁移到可复用的 Agent Skill，目标从单次 prompt 变为跨任务通用的技能文档。
- **Skill 自演化方法**：AutoManual 增量更新结构化规则并编译为手册；Voyager 存储可检索的可执行程序；ExpeL 保留自然语言经验；本文的核心区分在于引入了类似优化器状态的持久记忆和自适应预算机制。

## 局限性与未来方向
- **消融独立性不足**：论文说明消融实验为累积式（依次移除），未完全隔离记忆模块和预算模块在共存时的交互效应。
- **跨模型迁移评估不完整**：仅测试了 GPT-5.5→GPT-5.4-mini，未验证更远距离迁移或开源模型场景。
- **接受门控依赖固定阈值**：不同 benchmark 的指标阈值需人工设定，未讨论自动校准方案。
- **短程与长程的均衡性**：在长程任务（尤其是 DP-Travel）上仍有较大提升空间（11.7%），说明复杂规划场景下技能自演化仍有挑战。

## 研究启发与可借鉴点
- **优化器思想的离散化迁移**：将 Adam 的一阶/二阶矩概念功能性地映射到文本空间，为离散优化问题提供了通用的设计范式，可迁移至其他文本/程序生成场景。
- **波动性驱动的自适应编辑**：用历史加权方差估计改进的不一致性，进而控制修改幅度，这一机制可推广到任何基于反馈的离散迭代优化任务。
- **优化记忆机制**：结构化记录问题状态和求解历史，避免重复试错，对多轮 prompt 优化、代码生成调试等场景具有参考价值。
- **实验设计**：跨模型迁移评测、技能质量 Judge 评估、优化动态可视化三种互补的评估方式，为后续工作提供了完整的验证框架。
- **与本团队的结合机会**：可将 EIT 和 edit budget 机制集成到本团队的数据准备/代码生成 Agent 中，用于自动迭代优化系统提示或工具调用规范。

## 关键术语表
- **Agent Skill**：由 Anthropic 定义的模块化指令包，独立于模型参数存储，可为 Agent 赋予领域专用能力。
- **Evolving Issue Tracker（EIT）**：SKILLADAM 中的优化记忆模块，以结构化问题列表形式记录错误模式、状态及求解历史，类比 Adam 一阶矩。
- **Volatility-driven Edit Budget**：基于近期案例级改进波动性自适应调整编辑幅度的机制，高波动时缩小修改范围，类比 Adam 二阶矩。
- **Rollout**：在 mini-batch 上使用当前技能执行 Agent 并收集轨迹与评估反馈的过程。
- **Acceptance Gate**：基于主指标阈值和保护指标边界决定是否接受候选技能更新的门控函数。
- **Direction Stability**：确保有效修正在迭代间累积而非被后续局部反馈覆盖的属性。
- **Update Adaptivity**：根据近期改进的一致性自适应调节每次修订幅度的属性。
- **DP-Avg**：DeepPlanning 的 Shopping 和 Travel 两项准确率的非舍入平均值。

## 可复现要素
- **数据集**：SearchQA、SpreadsheetBench、OfficeQA、DocVQA、LiveMathematicianBench、ALFWorld、DeepPlanning（论文提供详细引用，均为公开 benchmark）。
- **代码开源**：是，仓库地址 https://github.com/ruc-datalab/SkillAdam。
- **权重开源**：论文未提及独立权重发布，代码仓库应包含完整实现。
- **关键超参**：mini-batch size $k$、EMA 系数 $\beta_2$、基础编辑预算 $b_{\mathrm{base}}$、最小预算 $b_{\min}$、波动饱和阈值 $V_{\max}$、随机种子 $\xi$；目标模型使用 GPT-5.5（temperature=1.0）和 Claude Sonnet 4.5（temperature=0.0）。
