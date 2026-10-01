---
title: "SKILLADAM-STABLE-AND-EFFICIENTSKILL-EVOLUTION-FOR-AGENTS"
source: https://arxiv.org/pdf/2609.08944v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 03:28:09"
field: "智能体技能优化"
keywords: ["Agent Skill", "Skill Self-Evolution", "Prompt Optimization", "Discrete Optimization", "LLM Agent"]
innovations: ["将 Adam 一阶矩二阶矩思想功能性地类比到离散技能空间，实现稳定高效优化", "设计进化问题跟踪器（EIT）作为优化记忆，累积跨迭代证据稳定方向", "设计波动驱动的编辑预算，基于改进一致性自适应控制修改幅度"]
benchmarks: ["SearchQA", "SpreadsheetBench", "OfficeQA", "DocVQA", "LiveMathematicianBench", "ALFWorld", "DeepPlanning"]
---

# 论文速读：SKILLADAM-STABLE-AND-EFFICIENTSKILL-EVOLUTION-FOR-AGENTS

## 一句话总结
SKILLADAM 是一种受 Adam 优化器启发的智能体技能自演化框架，通过优化记忆稳定迭代方向、波动驱动的编辑预算自适应控制修改幅度，在七个基准上实现了更稳定、高效的技能优化。

## 研究问题与动机
1. 智能体技能（Agent Skills）能为冻结的大模型注入领域知识，但高质量技能的构建依赖专家人工编写，成本高昂且难以规模化。
2. 现有的技能自演化方法（如 SkillOpt、Trace2Skill 等）通常采用启发式更新策略，优化过程不稳定（方向易被覆盖）且迭代效率低（盲目修订）。
3. 核心挑战一：**方向稳定性**（Direction Stability）——每次迭代仅基于有限样本反馈，后续修订可能推翻前期有效修正，导致优化方向漂移。
4. 核心挑战二：**更新适应性**（Update Adaptivity）——每次修订的幅度需根据近期案例级改进的一致性动态调整：改进不一致时应收紧修改范围，改进一致时可扩大修改。

## 核心贡献（创新点）
1. **提出 SKILLADAM 框架**：将 Adam 优化器的一阶矩和二阶矩思想功能性地类比到离散技能空间，实现稳定高效的技能自演化。
2. **设计进化问题跟踪器（EIT）**：作为优化记忆，记录已识别问题及其历史解决方案结果，使后续修订能累积有效修正而非重复试错。
3. **设计波动驱动的编辑预算**：基于近期案例级改进的波动性自适应控制修改幅度，在高波动时收紧范围以避免破坏已有约束。
4. **全面实验验证**：在七个基准（涵盖短/长视程任务）上取得 SOTA 性能，同时优化成本显著降低（Token 消耗减少约 67%）。
5. **跨模型迁移优势**：优化后的技能在源模型与目标模型间迁移时保持更高任务分数和保留率。

## 方法详解
1. **框架循环**：从执行轨迹和评估反馈初始化初始技能 $S_0$，之后每轮迭代 $t$：
   - 随机采样 mini-batch $B_t$，用当前技能 $S_{t-1}$ 执行得到轨迹 $\mathcal{T}_t$ 和反馈 $\mathcal{F}_t^{\text{roll}}$。
   - 结合上一轮优化状态（记忆 $M_{t-1}$、预算 $\sigma_t$）通过 LLM 生成候选修改 $g_t$。
   - 应用修改得到候选技能 $\tilde{S}_t$，在相同 batch 上评估得到 $\mathcal{F}_t^{\text{val}}$。
   - 通过多指标接受门控 $G_\theta$ 决定接受（$a_t=1$）或拒绝（$a_t=0$），更新技能 $S_t$。
   - 用 rollout 与 validation 对比结果进化记忆 $M_t$ 并更新波动估计 $V_t$、预算 $\sigma_{t+1}$。
2. **进化问题跟踪器（EIT）**：
   - 维护结构化问题列表 $M_t = \{I_j\}$，每项 $I_j = (p_j, z_j, \mathcal{A}_j)$ 记录问题模式、当前状态及历史解决方案与结果。
   - 更新函数 $\mathcal{U}_{\text{EIT}}$ 将新失败关联到已有问题或新建条目，记录尝试结果，若相同失败重现则重新打开问题。
   - 功能类比 Adam 一阶矩：累积跨迭代证据以稳定更新方向。
3. **波动驱动的编辑预算**：
   - 计算 case-level 改进 $\delta_{t,i} = s(\mathcal{E}_{t,i}^{\text{val}}) - s(\mathcal{E}_{t,i}^{\text{roll}})$，均值 $\bar{\delta}_t$。
   - 估计当前波动 $\widehat{V}_t = \text{Var}(\{\delta_{t,i}\})$，历史加权 $V_t = \beta_2 V_{t-1} + (1-\beta_2)\widehat{V}_t$。
   - 预算 $\sigma_{t+1} = \max\left(b_{\min}, \lfloor b_{\text{base}}\left[1 - \text{clip}\left(\frac{V_t}{V_{\max}}\right)\right]\rfloor\right)$。
   - 功能类比 Adam 二阶矩与有效学习率：波动高则预算小（收敛修改范围），波动低则预算大（允许更广修改）。
4. **接受门控**：基于基准的主要指标与辅助指标设定阈值，要求至少一项指标达到改进阈值且所有保护指标未发生退化，才接受候选技能。

## 实验与结果
- **数据集**：7 个基准，包括 5 个短视程（SearchQA、SpreadsheetBench、OfficeQA、DocVQA、LiveMathematicianBench）和 2 个长视程（ALFWorld、DeepPlanning）。
- **基线**：NoSkill、HumanSkill、LLMSkill、Trace2Skill、TextGrad、GEPA、SkillOpt。
- **主要结果**：
  - **短视程**：SKILLADAM 在五个基准上均取得最佳或次佳；平均比 HumanSkill 提升 14.45%、比 LLMSkill 提升 31.20%；较 SkillOpt 在 DocVQA 提升 1.21%、LiveMath 提升 1.20%。
  - **长视程**：ALFWorld 达 89.6%（SkillOpt 87.3%）；DeepPlanning Shopping 45.0%（41.7%）、Travel 11.7%（1.7%）、DP-Avg 28.3%（21.7%，+6.7pp）。
  - **消融**：移除波动预算后 DP-Avg 降至 21.7%，同时移除记忆后降至 19.2%。
  - **跨模型迁移**（GPT-5.5 → GPT-5.4-mini）：平均目标分数 67.8%（SkillOpt 63.1%），平均保留率 81.3%（76.2%）。
  - **优化成本**：SKILLADAM 总 Token 消耗 74.0M（SkillOpt 226.6M，-67.3%），API 请求 2830 次（9071 次，-68.8%），同时性能更优。
- **动态分析**：SKILLADAM 在第一轮即找到强修订并保持稳定，而 SkillOpt 接受修订波动较大、效率较低。

## 相关工作脉络
1. **Agent Skills**：Anthropic 定义技能包；Voyager、AutoManual 等外部化可重用能力；本文聚焦自然语言指令文档的迭代优化，与 ExpeL/Agent Workflow Memory 形成对比。
2. **Prompt Optimization**：OPRO、ProTeGi、TextGrad、GEPA、ERM 等优化提示文本；本文目标为跨任务实例复用的技能，且引入多指标接受门控与持久优化状态。
3. **Skill Self-Evolution**：Trace2Skill 汇总轨迹教训；SkillAxe、EvoSkill、CoEvoSkills、SkillOS 各自迭代/联合演化技能；SkillOpt 与本文设置最接近，但本文首次共同维护持久记忆与波动估计。

## 局限性与未来方向
1. **局限性**：实验局限于特定模型（GPT-5.5/Claude Sonnet 4.5）与基准；记忆与波动的耦合效应未在消融中完全隔离；接受门控依赖人工设定的指标阈值。
2. **未来方向**：探索更通用的指标聚合函数 $\Phi$；研究记忆结构的压缩与高效检索；将波动估计推广至多指标场景；扩展到更多智能体任务领域（如代码生成、数据科学）。

## 研究启发与可借鉴点
1. **方法可迁移**：Adam 的矩估计思想可迁移至其他离散文本优化问题（如提示工程、代码生成），通过记忆累积稳定方向。
2. **实验设计**：跨模型迁移实验有效评估技能可移植性；结合自动指标与人工 Judge 多维度评估技能质量。
3. **动态可视化**：绘制迭代索引与累积 Token 消耗下的接受/拒绝曲线，直观对比优化效率。
4. **团队结合点**：可将 EIT 机制嵌入本团队的 Prompt 优化流水线，减少重复试错；波动预算可用于控制文本修改的激进程度，提升稳定性。

## 关键术语表
- **Agent Skill**：模块化指令与资源包，为智能体提供领域专项能力。
- **Skill Self-Evolution**：利用执行反馈自动迭代改进技能文档的过程。
- **Evolving Issue Tracker (EIT)**：记录问题模式、状态及历史解决方案的优化记忆结构。
- **Volatility-driven Edit Budget**：基于近期改进一致性动态调整修改范围的机制。
- **Direction Stability**：确保有效修正在多轮迭代中不被覆盖的属性。
- **Update Adaptivity**：根据证据可靠性自适应控制修订幅度的属性。
- **Acceptance Gate**：基于多指标阈值决定是否接受候选技能的门控。
- **Case-level Improvement**：单个任务实例上候选技能相对于当前技能的指标变化。

## 可复现要素
- **数据集**：七个基准均公开（SearchQA、SpreadsheetBench、OfficeQA、DocVQA、LiveMathematicianBench、ALFWorld、DeepPlanning），部分需授权使用。
- **代码/权重**：代码已开源（https://github.com/ruc-datalab/SkillAdam），权重未提及。
- **关键超参**：mini-batch size $k$、EMA 系数 $\beta_2$、预算参数 $b_{\text{base}}$、$b_{\min}$、$V_{\max}$、随机种子 42；接受门控阈值（多指标）；模型配置：GPT-5.5（medium reasoning、temperature=1.0、max output=16384 tokens）、Claude Sonnet 4.5（temperature=0.0）。
