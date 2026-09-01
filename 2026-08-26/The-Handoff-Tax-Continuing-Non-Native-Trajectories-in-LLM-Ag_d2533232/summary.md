---
title: "The-Handoff-Tax-Continuing-Non-Native-Trajectories-in-LLM-Ag"
source: https://arxiv.org/pdf/2608.24358v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 16:11:06"
field: "LLM Agent系统与推理优化"
keywords: ["LLM Agent", "model handoff", "cost-quality tradeoff", "coding agent", "trajectory continuation", "model routing"]
innovations: ["首次系统量化长程Agent中模型交接的成本质量惩罚(交接税)", "发现交接方向性二元对立:升级应减少前缀轨迹而降级应保留前缀轨迹", "揭示交接税的两种计算机制:每步成本增加vs步数增加"]
benchmarks: ["SWE-bench Verified", "LiC", "BrowseComp"]
---

# 论文速读：The-Handoff-Tax-Continuing-Non-Native-Trajectories-in-LLM-Ag

## 一句话总结
本文首次系统研究了LLM编码Agent在执行长程任务时中途切换模型（handoff）的成本—质量权衡，发现全轨迹原始升级（Raw escalation）只能恢复不到一半的质量优势但成本显著增加（称为"交接税"），而降级（downshift）反而能提供有利的中间点；不同方向的交接对继承轨迹信息的偏好呈现明显的方向性二元对立。

## 研究问题与动机
- **长程Agent任务中的模型切换成本**：编码Agent单次任务往往涉及数十次模型调用、工具使用和多步代码编辑，用户需要根据任务难度动态选择模型，但切换会导致接收方继承非原生轨迹。
- **交接对质量和成本的影响缺乏系统性理解**：尽管coding-agent产品（如Kiro、Codex、Claude Code）已暴露模型切换命令，但关于"从一个模型继续另一个模型的非原生轨迹"如何影响最终成本与质量仍不清楚。
- **现有路由/级联方法无法刻画交接效应**：已有工作聚焦于模型选择和请求分发（cascades、routers），或仅在最终轮研究单向漂移，未考虑交接界面（interface）的设计选择。
- **不同交接方向的价值不对称性**：HC模型继续LC轨迹（升级）与LC模型继续HC轨迹（降级）可能存在本质不同的计算机制和成本结构。

## 核心贡献（创新点）
1. **首次系统性研究长程编码Agent中途中断交接问题**：通过Claude和GPT两个模型族、两个方向、七个切换点、四种界面共58,000次Agent运行，揭示了交接对成本—质量的系统性影响。
2. **定义并量化"交接税"（Handoff Tax）现象**：Raw升级仅能恢复不到一半的HC质量优势（QRec=47% for Claude, 36% for GPT），同时成本高达LC的4.0×/6.1×，为Claude甚至不如直接重启HC。
3. **发现交接方向性二元对立（Directional Duality）**：减少LC轨迹信息提升升级质量，而保留HC轨迹信息提升降级质量，揭示HC轨迹会"锚定"HC接收方而LC轨迹为LC接收方提供指导。
4. **揭示交接税的两种不同计算机制**：Raw升级溢价主要来自每步HC调用成本增加（Claude每步贵2.2×），而Traj-drop降级代价主要来自LC需要更多步骤（1.6×/2.0×）。
5. **扩展至不同信息动态场景验证交接价值的条件性**：在LiC（迟到的需求）中升级表现更优，在BrowseComp（渐进搜索）中升级几乎恢复质量但仍无法节省成本。

## 方法详解
- **实验框架**：使用SWE-bench Verified基准，mini-swe-agent scaffold，在同一Docker环境中运行，保留前置模型的working tree（代码编辑）不变，仅改变传递给后缀模型的轨迹信息。
- **四种交接界面**：
  - **Raw**：完整传递前缀轨迹$\mathcal{T}_{1:K}$
  - **Compact_pre**：前缀模型生成摘要$\widehat{\mathcal{T}}_{1:K}$，后缀模型仅接收摘要
  - **Compact_suf**：后缀模型自行阅读前缀轨迹并生成摘要
  - **Traj-drop**：不传递任何轨迹信息，仅保留working tree $\mathcal{W}_K$
- **切换点设计**：按难度分桶（easy <15min, medium 15min–1h, hard >1h）内计算前缀模型单模型终止步数的百分位数，取{p5, p10, p15, p25, p35, p45, p50}共七个切换点。
- **评估指标**：
  - Pass rate（通过率）、Mean cost（USD）、Mean steps
  - **Quality Recovery (QRec)**：$\frac{R_m(K) - R_{LC}(K)}{R_{HC}(K) - R_{LC}(K)} \times 100$，衡量恢复HC质量优势的比例
  - **Cost-Savings Retention (CSRet)**：$\frac{C_{HC}(K) - C_m(K)}{C_{HC}(K) - C_{LC}(K)} \times 100$，衡量保留LC成本优势的比例
- **成本分解机制分析**：将后交接成本拆分为"每步成本"和"步数"两个维度，揭示交接税的来源。
- **对照实验**：Abort + HC fresh（前缀仅计费后重启HC）、LC-full + HC-full（完整LC运行后重启HC），验证Raw升级是否被纯重启策略支配。
- **外部验证**：扩展至LiC（535样本，5类任务）和BrowseComp（200样本）验证不同信息动态下的交接模式。

## 实验与结果
- **数据集**：SWE-bench Verified（500实例），LiC（535实例，5类任务），BrowseComp（200实例）。
- **模型**：Claude Haiku 4.5 / Opus 4.7 和 GPT-5.6 Luna / Sol。
- **规模**：58,000次Agent运行，200万次LLM API调用，360亿token。
- **核心结果（Tab. 1）**：
  - **Claude升级**：Raw QRec=47%，CSRet=-285%（比HC更贵）；Compact_pre QRec=60%，CSRet=-11%；Traj-drop QRec=64%，CSRet=-30%。Raw升级（$1.61）> Abort+HC fresh（$0.90）> HC-only（$0.72）。
  - **GPT升级**：Raw QRec=36%；Traj-drop QRec=84%（最高）。
  - **Claude降级**：Raw QRec=50%，CSRet=80%（有利中间点）；Traj-drop QRec=28%，CSRet=59%（最差）。
  - **GPT降级**：Raw QRec=79%，CSRet=14%；Traj-drop QRec=53%。
- **成本机制（Fig. 4）**：
  - Raw升级使Claude每步HC成本增加2.2×（相比Compact_pre）
  - Traj-drop降级使Claude每步LC步数增加1.6×
- **难度条件**（Tab. 4）：Claude在hard任务上，所有简化上下文接口均比HC-only更便宜且恢复65-74%质量。
- **LiC（Tab. 2）**：迟到需求场景下升级QRec=86%，优于降级QRec=31%。
- **BrowseComp（Tab. 3）**：升级QRec=95.8%但CSRet=-30%（无成本节省）。
- **置信区间**（Tab. 8）：Traj-drop vs Raw的翻转效应在两个模型族、两个方向均统计显著（95% CI不含0）。

## 相关工作脉络
1. **模型路由/级联**：FrugalGPT (Chen et al., 2023)、HybridLLM (Ding et al., 2024)、Routellm (Ong et al., 2025)、UCCI (Kotte, 2026)、C3PO (Valkanas et al., 2025)——这些工作决定"哪个模型处理哪个请求"，但未考虑交接界面的设计。
2. **多轮路由决策**：Zhang et al. (2026a,b)、Hemadri et al. (2026)——扩展到多轮交互中的模型选择，但仍假设轨迹自然延续而非显式交接。
3. **SWE-Router**：Son et al. (2026)——使用LC部分轨迹决定继续还是重启，但明确不转移轨迹，与本文研究"如何交接"形成对比。
4. **单向漂移研究**：Khraishi et al. (2026)——仅研究最终轮的模型切换漂移，未考虑中途交接和不同界面。
5. **中断任务交接债务**：KC & Budathoki (2026)——研究Agent接手中断任务时的"再发现成本"，但其模型被评估为后续者而非具有不同成本曲线的LC-HC对。
6. **定位差异**：本文聚焦实际用户模式——在运行期间切换不同能力和成本的商用模型，系统性变化交接方向、时机和界面，测量"继续非原生轨迹"对端到端质量和货币成本的影响。

## 局限性与未来方向
- **仅测试两个模型族**：Claude和GPT各一对，推广到其他厂商需验证。
- **主要基准单一**：SWE-bench Verified为主要实验场，外部验证（LiC、BrowseComp）仅使用Raw交接，界面结论主要在coding setting成立。
- **切换点固定**：使用预定义的百分位切换点，未来需研究自适应/进度感知策略。
- **硬任务子集小**：N̄≈24-27 per cell，难度条件结论视为探索性。
- **单次episode**：未估计相同配置下重复运行的方差。
- **成本依赖定价**：美元成本结论依赖于当前provider定价和prompt-cache率。
- **未来方向**：开发结构化交接（选择性保留/摘要/丢弃轨迹组件）、联合优化交接界面与receiver选择及切换时机、扩展至多模型族/多轮次接力。

## 研究启发与可借鉴点
1. **交接界面设计应作为独立优化变量**：Routing决定"谁执行下一步"，但Handoff Interface决定"接收方继承什么信息"，两者应联合优化而非假设轨迹自然延续。
2. **方向性二元对立的接口选择原则**：升级时应减少前缀轨迹（Traj-drop优于Raw），降级时应保留前缀轨迹（Raw优于Traj-drop），为agent架构设计提供明确指导。
3. **成本分解分析方法可迁移**：将后交接成本拆解为"每步成本"和"步数"两维度，精确定位交接税的计算机制来源，可作为类似研究的通用分析框架。
4. **难度/信息动态条件的性能分化**：同一交接策略在不同任务难度和信息到达模式下表现迥异，提示未来研究需关注"何时切换"的条件化策略设计。
5. **对照组设计价值**：Abort+HC fresh和LC-full+HC-full作为restart控制，有效分离了"继续轨迹"vs"放弃轨迹"的影响，实验设计严谨。

## 关键术语表
- **Handoff Tax（交接税）**：接收方继承非原生轨迹时导致的成本—质量惩罚现象，表现为升级时质量恢复有限但成本显著上升。
- **LC / HC**：Low-cost/Low-capability 和 High-cost/High-capability 模型的缩写，分别指代廉价低能力和昂贵高能力的模型。
- **QRec（Quality Recovery，质量恢复率）**：交接策略恢复的HC质量优势占LC-HC质量差距的比例，0为LC水平，100为HC水平。
- **CSRet（Cost-Savings Retention，成本节省保留率）**：交接策略保留的LC成本优势占LC-HC成本差距的比例，100为LC成本，0为HC成本。
- **Traj-drop（轨迹丢弃）**：交接时不传递任何前缀轨迹信息，仅保留working tree的代码编辑状态。
- **Compact_pre / Compact_suf**：分别由前缀模型或后缀模型生成轨迹摘要，作为交接的压缩界面。
- **Directional Duality（方向性二元对立）**：HC轨迹对LC接收方有帮助但对HC接收方有害，LC轨迹反之的不对称现象。
- **Switched Subset（切换子集）**：仅在全部四种策略下均发生切换的实例上进行公平比较的评估集合。

## 可复现要素
- **数据集**：SWE-bench Verified（500实例，公开）；LiC（535实例，公开）；BrowseComp（200实例，公开）
- **代码/权重**：论文未提及开源，实验使用mini-swe-agent scaffold + LiteLLM v1.83.14 + Bedrock OpenAI-compatible端点
- **模型**：Claude Haiku 4.5 / Opus 4.7, GPT-5.6 Luna / Sol
- **关键超参**：temperature=0（Haiku），medium/high reasoning effort（GPT/Opus），150-step cap，切换百分位数{5,10,15,25,35,45,50}
- **摘要prompt**：标准化模板提供，可复现
- **成本来源**：LiteLLM v1.83.14 model-pricing registry + Bedrock价格，含cache pricing
