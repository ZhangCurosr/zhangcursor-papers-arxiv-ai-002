---
title: "SkillShapley-Boundary-Adaptive-Shapley-Valuation-for-Skill-S"
source: https://arxiv.org/pdf/2608.13173v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 04:16:46"
field: "LLM Agent 可解释性与技能工程"
keywords: ["Shapley value", "LLM agent skill", "step attribution", "active approximation", "SkillsBench", "prompt engineering"]
innovations: ["将技能步骤归因形式化为合作博弈 Shapley 估值", "BAES 缓存感知两阶段自适应近似", "以顶步移除曲线验证归因行为有效性并给出可测量编辑闭环"]
benchmarks: ["SkillsBench offer-letter-generator", "SkillsBench manufacturing-fjsp-optimization", "SkillsBench dialogue-parser"]
---

# 论文速读：SkillShapley: Boundary-Adaptive Shapley Valuation for Skill Step Attribution in LLM Agents

## 一句话总结
本文提出 SkillShapley，将 LLM Agent 的技能（skill）视为合作博弈，用 Shapley 值量化每个语义步骤对任务成功的边际贡献；并设计边界自适应 Shapley（BAES）两阶段近似算法，在极少新配置评估预算下高效还原步骤重要性排名，为技能剪枝/改写/生成提供可测量依据。

## 研究问题与动机
- 技能是驱动 LLM Agent 执行长程程序任务的关键外部指令，但当前技能设计依赖人工试错或执行轨迹，缺乏对“哪些具体步骤真正起作用”的细粒度归因。
- 现有 prompt 优化/压缩与工作流优化多把技能作为整体单元处理，无法回答固定技能内部单步贡献这一原子问题。
- 精确 Shapley 计算随步数指数爆炸；而在技能评估中，**新配置的执行成本远高于已有配置的“一次翻转”边缘复用成本**，且奖励离散、容易饱和，促使需要面向该成本结构与奖励分布的专用近似策略。
- 研究者希望验证三件事：Shapley 值对自然语言步骤是否具有行为意义、BAES 能否以远少于全枚举的独特配置预算还原排名、归因结果能否给出可落地的剪枝/改写/创建指导。

## 核心贡献（创新点）
1. **将技能步骤归因形式化为合作博弈 Shapley 估值问题**：步骤作为玩家、保留步骤子集作为联盟、基准性能作为效用，使“哪一步真正在做事”成为可计算目标。与单纯整体 prompt 优化或 token 级压缩的本质区别在于归因单元是**语义程序步骤**而非文本片段。
2. **提出 BAES（Boundary-Adaptive Edge Shapley）两阶段缓存感知近似**：先用 warmup 建立跨联盟尺寸的广覆盖并稳定层优先级序，再用 adaptive 阶段沿高方差/低采样层与低奖励邻居方向主动选取下一次评估。与通用采样式 Shapley 估算器的本质区别在于**以配置缓存和可复用 one-flip 边缘为核心资源进行自适应分配**。
3. **在 SkillsBench 上系统验证 Shapley 的行为有效性与 BAES 的预算效率**：精确 Shapley 在顶步移除时下降最快；在匹配独特配置预算下，BAES 比 MC/Quasi-MC/paired MC/截断 Shapley 等基线达到更低逼近误差，并以相同预算产生更多可复用边缘证据。与以往仅比较采样效率工作的区别在于**同时验证行为目标有效性与低预算近似优势**。
4. **给出面向技能工程的可操作案例与流程**：高价值步骤多扮演“过程桥接”角色（连接条件→决策/API/回退），低价值步骤多为局部正确但动作不完整的支持性信息；并提供“假设—编辑—再评估”的可测量闭环。与仅停留在解释性分析工作的区别在于**直接面向技能剪枝、改写与创建的编辑建议**。

## 方法详解
- **问题建模**：固定技能分割为 $n$ 个语义连贯步骤 $X=(x_1,\ldots,x_n)$；子集 $S\subseteq N$ 构成保留这些步骤并保持原序的技能变体 $X_S$。联盟价值 $v(S)=\frac{1}{M}\sum_{j=1}^M r(o_j(S),y_j)$，多为二分成功率的经验均值。前后缀元数据与辅助资源固定，确保变化仅来自保留步骤。
- **目标量**：步骤 $i$ 的 Shapley 值 $\phi_i=\sum_{S\subseteq N\setminus\{i\}}\frac{|S)!(n-|S|-1)!}{n!}[v(S\cup\{i\})-v(S)]$，等价按联盟尺寸分层后的边际期望平均。
- **BAES 两阶段流程**：
  - **Warmup**：先评估锚点配置（空集、全集、所有单点与所有 $(n{-}1)$ 子集），再用 greedy cache-aware 规则选择暴露高优先级层且能产生多条可复用 one-flip 边的中间尺寸配置，直到 RankUnstable 判定层优先级排序趋于稳定（Kendall’s $\tau$ 的 EMA 下降至峰值的 $1/e$ 以下，且已观测至少 60 个层）。
  - **Adaptive**：对未评估配置 $C$，按 $A(C)=\sum_{i: C\triangle\{i\}\in\mathcal{D}} a_{i,k}\cdot b(v(C\triangle\{i\}))$ 打分后选最大者执行并更新缓存与层统计。其中分配分数 $a_{i,k}=\sqrt{\hat\sigma_{i,k}^2+\epsilon}/\sqrt{m_{i,k}+1}$ 偏向高方差/低采样层；缓存奖励权重 $b(y)=1+\max\{0,\mathrm{round}((1-y)/g)\}$ 优先靠近低奖励已缓存状态的邻居。
  - **停止与估计**：以归一化标准误差 NSE 在近端去相关窗口的斜率是否持续下降来判断 UncertDec；最终按层均值聚合得 $\hat\phi_i=\frac{1}{n}\sum_k \hat\mu_{i,k}$，并对极少数空层做同尺寸均值回填。
- **成本观**：核心开销是**新配置数量**而非算术边缘数；BAES 优势来自同一配置预算内产生的可复用 one-flip 边际证据更多。

## 实验与结果
- **数据集与设置**：SkillsBench 三类低步数技能：offer-letter-generator/docx（$n{=}10$, 1024 精确配置）、manufacturing-fjsp-optimization/fjsp-baseline-repair-with-downtime-and-policy（$n{=}9$, 512）、dialogue-parser/dialogue-graph（$n{=}11$, 2048）。每配置在 3 个实例上评分，奖励粒度 $g{=}1/3$；Agent 框架 OpenHands，温度 $T{=}0$，共享同一段基准子集。
- **精确 Shapley 行为验证**：按各方法排名移除顶步后，Full Shapley 的成功率下降曲线最陡，显著优于 Individual、Leave-One-Out、Random Removal、LeastCore，证明 Shapley 排名与“真正重要的步骤”强相关。
- **BAES 逼近效率**：在匹配独特配置预算下，BAES 的 Shapley 误差曲线最低，优于 MC Shapley、Quasi-MC Shapley、paired MC Shapley 与 size-k-truncated Shapley。在 10 步 pilot 中，相同 99 配置预算下 BAES Phase1 产生 206 条可复用 one-flip 边缘，而 MC 仅 130 条（唯一 115 条）。
- **Token 成本诊断**：Dialogue Parser 显示 token 数并不与联盟尺寸单调对应，提示归因指导编辑旨在去除低价值内容而非保证成比例省 token。
- **代表性结论数字**：高价值步骤如 FJSP P9（代码块）$\phi{=}0.1155$，低价值步骤如 Offer Letter P2（纯文本）$\phi{=}-0.0194$，体现价值异质性与任务依赖性。

## 相关工作脉络
- **Agent Skills / SkillsBench**：将技能作为结构化程序知识并提供跨任务确定性验证器；本文定位差异在于不优化整套技能，而是对**固定技能内的单步做归因**。
- **Prompt/Instruction 优化与压缩**（LLMLingua/LongLLMLingua 等）：关注可压缩 token/片段；本文归因单元是**语义步骤**，直接面向程序性作用而非文本冗余。
- **Shapley 归因家族**（SHAP/Data Shapley/MC 变体/FastSHAP 等）：多数假设联盟评估相对廉价；本文针对**新配置执行昂贵、可复用 one-flip 边缘有价值**的技能评估成本模型。
- **Skill 优化/精炼/编译**（Skill set optimization/SkillAxe/SkillReducer/SkCC 等）：面向技能集合或跨框架便携；本文聚焦**单个固定技能内部的步骤贡献解析**。
- **LeastCore 等核心理论工具**：作为精确参考对比之一；本文强调在低预算下以行为 ranking 恢复为目标的实用近似。

## 局限性与未来方向
- 精确参考仅在低步数技能上可行，高步数场景仍依赖近似；动态长度工作流与高度主观评估会使合作博弈定义不稳。
- 强耦合流水线中删除中间步可能导致整条管线崩溃，此时大 Shapley 值可能反映结构必要性而非步骤可用性，存在解释偏差风险。
- 当前评估基于固定基准子集与 $T{=}0$，对随机性或 richer evaluation signal 的泛化未充分展开。
- 未来可扩展精确锚点规模、改进不确定性驱动的编辑决策、并将单技能归因拓展到多技能 Agent 管线与更丰富的评估信号。

## 研究启发与可借鉴点
- **缓存感知的 active Shapley 思路可迁移**：凡“新实验配置昂贵、相邻比较可复用”的场景（超参组合、模块拼装、RAG 组件选择）均可套用 warmup 覆盖+adaptive 选点策略。
- **按联盟尺寸分层的方差/采样分配**：$a_{i,k}$ 将不确定性显式绑定到尺寸层，便于诊断“哪些组合规模最需要样本”，对其它特征/模块归因实验设计有参考价值。
- **行为有效性优先于数值无偏**：在粗糙奖励域中用“顶步移除曲线”验证排名可用，比追求低数值误差更贴合工程需求。
- **程序桥接模式作为技能写作规范**：高价值步骤往往是“条件→决策/API/回退”的连接点；可在团队内沉淀“桥接步骤检查清单”提升技能编写质量。
- **归因驱动的编辑闭环**：假设—编辑—再评估的结构可直接嵌入 CI/技能库维护流程，把试错变为可测量迭代。

## 关键术语表
- **SkillShapley**：基于 Shapley 值的 LLM Agent 技能步骤级归因框架。
- **BAES（Boundary-Adaptive Edge Shapley）**：面向技能评估成本结构的缓存感知两阶段 Shapley 近似方法。
- **联盟/Coalition**：固定技能中保留的部分步骤子集及其保持原序构成的技能变体。
- **边际贡献/One-flip Edge**：向某联盟加入或去掉单步引起的效用变化。
- **层/Stratum $(i,k)$**：玩家 $i$ 在联盟尺寸为 $k$ 时的边际分布。
- **奖励粒度 $g$**：离散奖励的最小可观察增量，用于判断有效不确定性下降。
- **分配分数 $a_{i,k}$**：综合层内方差与样本数的优先级分数，指导自适应采样。
- **缓存奖励权重 $b(y)$**：对靠近低奖励已缓存状态的邻居给予更高选取优先。

## 可复现要素
- 数据集：SkillsBench（公开基准）；本文使用其中 offer-letter-generator、fjsp-baseline-repair、dialogue-parser 三组任务-技能对。
- 代码/权重：论文未明确提供开源仓库与权重链接，实现细节在附录 A/B 给出足够参数与启发规则。
- 关键超参：温度 $T{=}0$；配置预算 $B{=}3n^2$，warmup 上限 $R{=\lfloor 0.4B\rfloor}$；奖励粒度 $g{=}1/3$；停止阈值与 EMA 半衰期、去相关窗等在附录 B 详述。
- 评估设定：每配置 3 个实例；OpenHands  harness；前后缀元数据与辅助资源固定；跨方法共享同一基准子集。
