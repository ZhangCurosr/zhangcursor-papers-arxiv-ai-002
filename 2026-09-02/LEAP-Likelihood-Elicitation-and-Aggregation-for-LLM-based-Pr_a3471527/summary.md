---
title: "LEAP-Likelihood-Elicitation-and-Aggregation-for-LLM-based-Pr"
source: https://arxiv.org/pdf/2609.01337v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 12:08:58"
field: "LLM-based probabilistic forecasting"
keywords: ["probabilistic forecasting", "LLM agent", "Bayesian inference", "likelihood elicitation", "evidence aggregation", "forecasting benchmark"]
innovations: ["提出LEAP框架，将LLM作为局部证据解释器并与确定性概率聚合解耦", "设计共轭后验更新支持连续/单选/多选三种输出类型", "构建包含依赖聚类/可靠性采样/离群拒绝的完整保障机制"]
benchmarks: ["FutureX", "GAIA", "BrowseComp"]
---

# 论文速读：LEAP-Likelihood-Elicitation-and-Aggregation-for-LLM-based-Pr

## 一句话总结
论文针对 LLM 基于搜索证据的最终预测阶段提出 LEAP 方法，将每个证据项单独交给 LLM 进行局部似然解释，再通过确定性概率模型（共轭先验 + 贝叶斯更新）聚合为后验分布，从而在保持可审计性的同时提升预测精度与校准质量。

## 研究问题与动机
- **核心问题**：现有 LLM 预测系统在收集到证据后，通常将全部证据打包输入 LLM 做一次整体预测（称为 Monolithic Prediction），导致两个缺陷：（1）预测不透明，无法隔离单个证据项对最终结果的影响；（2）在长上下文下 LLM 易遗忘或幻觉，且将支持不确定性的分散结果坍缩为单一答案。
- **现状不足**：已有方法（如 BIRD、Nafar 等）将 LLM 判断嵌入结构化概率模型，但未聚焦"固定证据集→概率预测"这一明确预测步骤；此外多数工作关注检索/搜索阶段，而非证据消费方式。
- **动机**：借鉴概率建模中"证据解释与聚合分离"的思路，让 LLM 只做局部证据解读，确定性概率模型完成最终聚合，使预测可审计、可复查（保留 leave-one-out 贡献度）。

## 核心贡献（创新点）
1. **提出 LEAP 框架**：将预测阶段拆分为"局部参数提取 + 确定性概率聚合"两步，LLM 仅评估单个证据对目标的似然含义，不再一次性读取全部证据输出答案。*与已有工作的本质区别*：不同于 BIRD/Nafar 从 LLM 抽取因子来构建或参数化贝叶斯网络，LEAP 直接从已收集证据中逐条提取似然参数，并针对连续/单选一/多选三种输出类型分别设计了共轭更新。
2. **构建统一评估协议**：固定证据集、时间截断、模型知识截止，严格隔离检索与预测两个阶段，使性能差异完全归因于预测消费方式。*与已有工作的本质区别*：多数 Agent 基准评估整系统（搜索+推理+生成），本文协议固定证据只比较预测消费阶段。
3. **设计工程保障措施**：包括依赖聚类（合并同来源证据）、可靠性采样（对局部似然重复抽取以降噪）、离群拒绝（以先验均值4个标准差为阈值排除异常估计）。*本质区别*：将传统概率推断中的依赖处理与校准机制显式引入 LLM 证据消费流程。
4. **跨模型与跨框架验证**：在 5 个基础模型与 4 个外部 Agent CLI 框架（DeerFlow、Hermes、OpenClaw、MiroFlow）上均显著优于 Monolithic 基线，证明方法通用性。*与已有工作区别*：不仅验证单一模型，还以即插即用模块形式对接现有 Agent 框架。

## 方法详解
- **任务定义**：预测任务表示为 $T = (q, t_{\text{freeze}}, \tau, \mathcal{O})$，其中 $t_{\text{freeze}}$ 为可用证据的时间截止，$\tau$ 为输出类型（连续/单选一/多选），$\mathcal{O}$ 为候选选项集合。
- **两阶段分解**：先通过 Agent 循环收集证据 $\mathcal{E} = \{e_1, \ldots, e_n\}$，再进入预测阶段，不对 $\mathcal{E}$ 做任何额外检索。
- **概率模型**：设先验 $\theta \sim P_0(\theta)$，各证据条件独立 $e_i|\theta \stackrel{\text{ind}}{\sim} P_i(e_i|\theta)$，后验 $P(\theta|\mathcal{E}) \propto P_0(\theta)\prod_{i\in\mathcal{R}} P_i(e_i|\theta)$，其中 $\mathcal{R}\subseteq\{1,\ldots,n\}$ 为去重保留集合。
- **共轭选择**：
  - 连续型：高斯先验+高斯似然，后验精度 $\tau_{\text{post}} = \tau_0 + \eta\sum_{i\in\mathcal{R}}\tau_i$，后验均值 $\mu_{\text{post}} = \frac{1}{\tau_{\text{post}}}(\mu_0\tau_0 + \eta\sum_{i\in\mathcal{R}}\mu_i\tau_i)$。
  - 单选型：类别先验 + 多项似然，后验 $\log\pi_{\text{post}}(k) \propto \alpha\log\pi_0(k) + \eta\sum_{i\in\mathcal{R}}w_i\log L_i(k)$，再做 softmax。
  - 多选型：独立伯努利先验，后验 logit $\text{logit}(\pi_{\text{post}}(k)) = \text{logit}(\pi_0(k)) + \eta\sum_{i\in\mathcal{R}}w_i\ell_i(k)$，各选项边际独立，无需归一化和为1。
- **参数提取**：先验参数由历史数据或一次无证据的 LLM 调用生成（禁止见证据以避免双重计数）；每个证据项 $e_i$ 独立送入 LLM，只暴露 $T$ 和 $e_i$，输出结构化似然 schema 和来源依赖键。
- **确定性聚合**：用闭式更新计算后验，读出预测分布；对每条保留证据 $j$ 计算 leave-one-out 贡献 $\Delta_j$（排除 $j$ 后后验的变化量）。
- **保障措施**：
  - **依赖聚类**：用 LLM 输出的依赖键聚类同源证据，每组保留一个代表。
  - **可靠性采样**：对同一证据重复抽取 $R$ 次，以采样一致性调整似然幅度。
  - **离群拒绝**：连续任务中以先验均值4个标准差为阈值剔除异常似然估计。
  - **温度 $\eta$**：全局缩放因子，实验默认 $\eta=1.0$，对结果影响较小。

## 实验与结果
- **数据集**：FutureX（157 题）、GAIA（99 题）、BrowseComp（91 题），共 347 题，输出类型分布：单选266、多选29、连续52。另取 60 题诊断子集用于消融分析。
- **评估协议**：严格时间截断（$t_{\text{freeze}}$）、共享证据集、模型知识截止早于目标时间；指标含 FutureX 综合分、Accuracy、Brier、Spherical、NCRPS、ECE、Overconfidence。
- **主结果（Agent Loop + 5 模型）**：LEAP 在 FutureX 上提升 3.6–18.1 分，Spherical 提升 2.5–15.1 分；GPT-5.4-mini 上 NCRPS 提升超 50 分。Brier 在部分模型（DeepSeek-V3.2、Gemini-3.1-Flash-Lite）上与 Monolithic 接近但校准显著改善。
- **外框架结果（4 CLI）**：FutureX 提升 7.0–13.8 分，Brier 降低 12.9–23.7 分；整体宏平均 FutureX +9.8、Accuracy +4.7、Brier -16.5、Spherical +14.1、NCRPS +12.9。
- **校准分析**：在诊断子集上，LEAP 将 ECE 从 0.184 降至 0.088（近减半）、Overconfidence 从 0.317 降至 0.150；预测越远（7→60天）LEAP 优势越大。
- **消融**：移除先验导致 FutureX 跌破基线（0.643 vs 0.651），是先验最重要的组件；移除依赖聚类降 1.9 分；可靠性采样有小幅但一致的增益。

## 相关工作脉络
- **AutoCast / AutoCast++ / ForecastBench / Prophet Arena**：构建带时间戳的预测基准和动态评测体系，关注世界事件预测能力，但不解决固定证据消费方式问题。
- **BIRD（Feng et al., 2025）**：用 LLM 生成因子和粗粒度概率判断参数化贝叶斯模型进行推断；与 LEAP 区别在于 BIRD 从 LLM 抽取模型因子，LEAP 直接从证据提取每条似然并保留证据级审计。
- **Nafar et al. (2026)**：从 LLM 抽取条件概率参数化预定义领域贝叶斯网络；LEAP 不设预定义网络结构，完全从证据逐条自底向上构造似然。
- **Bayesian Linguistic Forecaster（Murphy, 2026）**：在迭代搜索过程中维护结构化信念状态并通过 shrinkage 和校准组合多轮预测；LEAP 将检索与预测完全解耦，在证据固定后做一次确定性更新。
- **ERASER / Faithful CoT**：关注可解释性与忠实推理；LEAP 继承"解释与计算分离"思想，让 LLM 提供局部解释、确定性模型执行聚合。
- **WebGPT / ReAct / ToolLLM**：改进检索与工具调用阶段；LEAP 假定上游检索已完成，专注于预测消费步骤。

## 局限性与未来方向
- **仅限预测阶段**：不改进上游证据收集，若证据本身稀疏/过时/弱相关，LEAP 只能表达更大不确定性，无法恢复未检索信息。
- **评测范围局限**：仅覆盖英文预测与信息检索/浏览任务，未扩展到中文、专业科学领域、更长预测周期及不同 Agent 设计。
- **推理开销增加**：每条证据需单独 LLM 调用（可能多次采样），token 消耗和延迟高于 Monolithic（median 延迟相近，p95 更高）。
- **未来方向**：跨语言/跨领域扩展、与检索阶段联合优化、探索更高效的证据交互方式。

## 研究启发与可借鉴点
- **可迁移方法**：LLM 作为"局部证据解释器"+ 确定性概率模型的分离设计可推广到其他需要证据聚合的任务（如风险预测、医学诊断辅助）。
- **实验设计借鉴**：固定证据集、严格时间截断、模型知识截止控制，使得预测阶段的公平比较完全归因于方法差异，值得在同类工作中复用。
- **工程保障机制**：依赖聚类、可靠性采样、离群拒绝三个组件的组合可作为通用可靠预测的模板。
- **审计价值**：leave-one-out 证据贡献度 $\Delta_j$ 的实现简单且可复现，可直接用于构建预测可审计系统。
- **即插即用范式**：以 probability skill 形式对接已有 Agent CLI 框架（无需修改上游），为现有系统升级提供低成本路径。

## 关键术语表
- **Monolithic Prediction**：将全部证据一次性输入 LLM 输出最终预测的传统范式，缺乏证据级可审计性。
- **LEAP（Likelihood Elicitation and Aggregation for Probabilistic forecasting）**：本文提出的预测框架，将 LLM 作为局部证据解释器，通过确定性概率模型聚合为后验分布。
- **似然提取（Likelihood Elicitation）**：LLM 独立分析单个证据项，输出描述该证据对目标各候选结果支持程度的结构化参数。
- **共轭后验更新**：利用共轭先验-似然配对得到的闭式后验更新，保证聚合过程确定性且可微分解。
- **Leave-one-out 贡献度（$\Delta_j$）**：排除某条证据后重新计算后验，前后差异即为该证据对最终预测的贡献。
- **依赖聚类**：将来源相同的证据归为一组，每组仅保留一个代表项进入聚合，避免重复计数。
- **可靠性采样**：对同一证据重复多次独立抽取，以采样一致性程度调整局部似然幅度。
- **ECE（Expected Calibration Error）**：衡量预测置信度与实际准确率之间差距的校准指标，值越低表示校准越好。

## 可复现要素
- **数据集**：FutureX（Apache-2.0）、GAIA（gated，text-only mirror 可用）、BrowseComp（MIT）；论文声明为 research-only 评估，未开源新数据集，原始 artifact 见原 benchmark。
- **代码/权重**：论文未声明开源代码，模型均为 API 调用（DeepSeek-V3.2、Gemini-3.1-Flash-Lite、Claude-Haiku-4.5、GPT-5.4-mini、Grok-4.20-Fast）。
- **关键超参**：Agent 循环预算 $B=10$ 轮，最多 4 工具调用/轮，最多 6 条证据参与似然阶段，最多 60 次似然调用/任务，每条证据 10 次可靠性采样，$\eta=1.0$，异常阈值 4 个标准差；上述数值为评估前固定，未做网格搜索。
