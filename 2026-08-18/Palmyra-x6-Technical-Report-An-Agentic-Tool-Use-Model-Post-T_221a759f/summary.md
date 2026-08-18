---
title: "Palmyra-x6-Technical-Report-An-Agentic-Tool-Use-Model-Post-T"
source: https://arxiv.org/pdf/2608.16620v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 04:28:38"
field: "大语言模型后训练与 Agentic 系统"
keywords: ["Agentic 模型", "锚定监督微调", "ASFT", "Muon 优化器", "工具调用", "MoE 架构", "合成数据生成"]
innovations: ["提出 ASFT 方法，结合 DFT 概率加权与 KL 锚定损失，在 744B 参数模型上仅用 626 条轨迹实现 Agentic 能力提升且保持基座能力", "Muon+Adam 混合优化器适配 MoE+MLA 架构，通过 Muon Split 变体保留 MLA 预训练几何结构", "端到端合成数据管线含消毒、效率上限、验证器评分与作弊过滤多层门控，确保轨迹质量"]
benchmarks: ["MCP-Atlas", "FinanceBench", "IFBench", "BFCL Core", "FORTRESS", "Washington Post ModelSlant", "TruthfulQA", "BBQ", "OR-Bench"]
---

# 论文速读：Palmyra-x6-Technical-Report-An-Agentic-Tool-Use-Model-Post-T

## 一句话总结
论文介绍了 Writer 公司开发的 Agentic 模型 Palmyra x6，基于 744B 参数的 GLM-5.2 MoE 基座模型，通过仅 626 条高质量合成轨迹的锚定监督微调（ASFT）加上 Muon+Adam 混合优化器，实现了在工具调用、多步规划等 Agentic 任务上的显著提升，同时保持基座模型的核心能力不被退化。

## 研究问题与动机
- **核心问题**：如何在最小化对基座模型已有能力的损害下，高效地将工具调用与 Agentic 行为"注入"到已具备强通用能力的大模型中。
- **现有方法不足**：传统 SFT 使用外部专家生成的文本作为训练目标，导致模型输出分布远离自身自然分布，引发能力退化（catastrophic forgetting）；全量重新训练成本过高且不必要。
- **数据质量 vs. 规模困境**：大规模低质训练数据效果有限，而小样本高质量数据若缺乏适当的约束机制，极易过度偏离基座分布。
- **安全与对齐风险**：Agentic 模型暴露于工具调用环境中，训练过程中的行为漂移可能导致部署时的安全风险。

## 核心贡献（创新点）
- **ASFT 训练框架**：提出锚定监督微调方法，结合 DFT token 概率权重与 KL 锚定损失（K=0.1），在微调期间维持输出分布靠近基座模型，防止能力退化。
- **小样本高效 Agentic 训练**：仅需 626 条合成轨迹、单轮 epoch 即可在 744B 参数 MoE 模型上实现显著的 Agentic 能力提升，验证了"less is more"理念在 Agentic 场景的适用性。
- **Muon+Adam 混合优化器**：对 2D 权重矩阵采用 Muon 优化器（通过 Newton-Schulz 迭代正交化动量缓冲），对 1D 参数保留 Adam，兼顾优化效率与收敛稳定性。
- **端到端合成数据管线**：设计了自蒸馏细调（SDFT 改编）的数据生成流程，包含去演示、工具调用验证、作弊过滤、LLM 评审等多重质量门控，确保训练数据的高效性与真实性。

## 方法详解
- **ASFT 损失函数**：总损失由两部分组成——DFT 加权负对数似然项 $-\text{mean}(P \cdot \log p_\theta)$，其中 $P = \exp(\log p_\theta(y_t)).\text{detach()}$ 是对自身 token 概率的权重；以及 KL 锚定项 $K \cdot \text{KL}(\pi_{\text{ref}} \| \pi_\theta)$，采用 $k_3$ 估计器计算，参考分布 $\pi_{\text{ref}}$ 为冻结的 GLM-5.2 基座模型，KL 权重 $K = 0.1$。
- **Mu​on 优化器设计**：对注意力投影、密集 FFN 和 MoE expert FFN 的 2D 权重矩阵应用 Muon，通过 Newton-Schulz 迭代（通常 5 步）对动量缓冲 $M_t$ 进行近似正交化，等效替换为最近半正交矩阵；采用 Muon Split 变体处理 MLA 投影，按 head 拆分后独立正交化，避免跨 head 耦合；对嵌入层、输出头、RMSNorm 增益、router 权重、偏置和标量参数继续使用 Adam（$\beta_1=0.9, \beta_2=0.98$）。
- **训练配置**：学习率 $5 \times 10^{-7}$，余弦衰减至 $5 \times 10^{-8}$，warmup 占比 0.1，初始 LR $1 \times 10^{-8}$，权重衰减 0.1，全局批次大小 16，序列长度 65,536，BF16 精度，FlashAttention 后端。
- **并行策略**：TP=8（含序列并行）、PP=4、EP=16、ETP=1、CP=2、DP=1，使用 CPU-offloaded 优化器状态以适配 744B MoE 模型的显存需求。
- **合成数据生成管线**：教师模型生成轨迹后，经过"消毒"提取策略级演示（保留推理框架和工具调用顺序，去除答案、数字、日期和工具参数），注入学生模型的 system prompt 作为参考；学生模型在 $T=1.0$、top-p=0.95 下采样，最多 30 turn、50,000 token、6 次重试；设置努力上限（最多 20 次工具调用、4 次连续同工具调用、3 次错误调用、3 次重复调用）；验证器通过 judge 模型三次投票（1.0/0.5/0.0 三级评分）评估轨迹质量；作弊过滤器检测四元组重叠 >0.8 或无工具调用直接输出答案的情况。

## 实验与结果
- **内部基准对比**：Palmyra x6 相对于 Writer Agent 的前代默认模型在所有 6 个内部基准上均有显著提升，最大增益为 MCP-Atlas（+0.320）、FinanceBench（+0.305）和 IFBench（+0.304）。
- **外部模型对比**：在 BFCL Core 上以 0.785 分位居第一，六个基准的平均分 0.765 为同期模型最高。
- **安全与偏见评估**：在 Washington Post ModelSlant 政治偏见评测中，Palmyra x6 在 80% 的高争议问题上呈现双面对比所有其他模型更为均衡，政治拒绝不匹配率仅为 1.2%（全部 8 个模型中第二低）；TruthfulQA 事实准确度 80.6%，与基座模型 81.5% 基本持平；FORTRESS 对抗安全评估中，配合系统提示词的 Palmyra x6 得分为 67.0（较基座提升 8.6 分）， benign 帮助性得分 96.4（基座 97.2），说明系统提示词是安全性的关键组件。
- **基线方法**：对比对象包括 Writer Agent 前代默认模型以及 DeepSeek V4、Kimi K3 等前沿模型。

## 相关工作脉络
- **Anchored Supervised Fine-Tuning (ASFT)** [1]：本文核心方法来源，首次在 ICLR 2026 提出，本文将其应用于 Agentic 工具调用场景并验证了在 744B 参数规模上的有效性。
- **Self-Distillation Fine-Tuning (SDFT)** [11]：让模型作为自身教师的设计思路启发了本文的合成数据生成管线，但本文将其从 token 级 KL 蒸馏扩展为采样+过滤的在线策略数据生成范式。
- **LIMO/LIMI "Less is More"** [12][13]：本文在数据量控制上与之呼应，论证了少量高质量轨迹即可驱动 Agentic 能力提升，而非依赖大规模数据淹没。
- **Muon Optimizer** [2][17]：本文采用 Muon 处理 2D 权重矩阵，并结合 Muon Split 变体适配 MLA 架构，扩展了 Muon 在 MoE+MLA 架构中的适用性。
- **DFT Token-Probability Weighting** [14]：损失函数中的概率加权策略源自该工作，本文将其与 KL 锚定结合，形成更稳定的微调方案。
- **IndexCache (DSA IndexShare)** [7]：基座模型 GLM-5.2 的稀疏注意力索引跨层复用机制，本文继承该架构并在部署时要求推理栈支持 IndexShare（如 SGLang ≥ 0.5.13.post1）。

## 局限性与未来方向
- **领域覆盖有限**：模型仅在 12 个私有数据集覆盖的任务域内接受过训练，超出这些领域（尤其是工具生态系统差异较大的场景）的性能未见验证。
- **非 Agentic 任务依赖基座**：模型在 Agentic 任务上的提升主要来自微调，而非 Agentic 的通用任务能力完全继承自基座，未对纯语言理解任务进行针对性优化。
- **安全对齐假设未充分验证**：论文承认 KL 锚定可能限制对齐行为漂移，但明确指出"未通过严格实验验证"这一假设。
- **单轮训练的限制**：仅训练一个 epoch 可能未能充分挖掘数据潜力，未来可探索多轮训练与 curriculum 设计的结合。
- **合成数据的潜在偏差**：所有训练数据由机器生成，虽经多重过滤，但仍可能存在合成数据特有的分布偏差或遗漏真实世界复杂场景。

## 研究启发与可借鉴点
- **"少即是多"的数据策略**：在 744B 参数模型上仅用 626 条高质量轨迹即实现显著能力跃升，为团队后续在大规模模型上的小样本高效微调提供了实证支撑，值得在本团队 Agentic 方向上复现验证。
- **ASFT 的通用性**：DFT 概率加权 + KL 锚定的组合方案可有效缓解 SFT 导致的能力退化，该设计可迁移至本团队的其他后训练场景（如多轮对话、代码生成等），避免基座能力损失。
- **Muon + Adam 混合优化器**：对 MoE 模型的高效训练方案，尤其适合大参数规模的微调任务，建议在本团队的后续训练中尝试 Muon 替代 Adam 以提升收敛效率和降低优化器内存占用。
- **合成数据管线的多层门控设计**：从轨迹消毒、效率上限、验证器评分到作弊过滤的完整链路设计，为本团队构建 Agentic 训练数据提供了可直接借鉴的工程范式。
- **安全评估的系统视角**：FORTRESS 评测结果显示系统提示词对安全性能的贡献（+8.6 分），提示团队在部署 Agentic 模型时应将模型微调与系统级安全设计视为整体。

## 关键术语表
- **ASFT (Anchored Supervised Fine-Tuning)**：锚定监督微调，通过在 SFT 损失中加入 KL 锚定项，约束微调后模型分布靠近冻结的基座模型，防止能力退化。
- **DFT (Detached Fine-Tuning) Token Probability Weighting**：将每个 token 的 NLL 乘以模型自身的 detached 概率作为权重，聚焦于模型已 confident 的 token，降低噪声影响。
- **Muon Optimizer**：MomentUm Orthogonalized by Newton-Schulz 优化器，通过对 2D 权重矩阵的动量缓冲进行正交化，使各奇异值方向的更新幅度趋于均衡。
- **KL Anchor**：以 $k_3$ 估计器计算的 per-token KL 散度惩罚项，约束当前模型输出分布与冻结基座模型分布的偏离程度，本文取 $K=0.1$。
- **GLM-5.2 GlmMoeDsa 架构**：结合 Sparse MoE FFN、Multihead Latent Attention (MLA) 和 DeepSeek Sparse Attention (DSA) indexer 的基础架构，支持 744B 总参数、~40B 活跃参数。
- **IndexShare**：DSA indexer 的跨层索引复用机制，不同注意力层共享同一组稀疏索引选择，降低显存和计算开销。
- **SDFT (Self-Distillation Fine-Tuning)**：自蒸馏微调，让模型在给定专家演示的 context 下生成自己的输出，教师信号来自模型自身，减少输出分布偏移。
- **Muon Split**：将 MLA 的投影权重按 head 拆分为独立子矩阵，分别进行 Newton-Schulz 正交化后再合并，避免跨 head 的无效耦合。

## 可复现要素
- **数据集**：12 个私有数据集（金融研究、数据分析/编程、医疗代理、RAG、模拟世界、MCP 工具套件），共计 626 条合成轨迹；论文未公开数据集。
- **代码/框架**：训练框架为 slime [20]（THUDM 开源的后训练框架），推理需支持 DSA IndexShare 的推理栈（如 SGLang ≥ 0.5.13.post1）。
- **权重**：BF16 权重及 FP4 量化变体已发布，但受 Writer 商业许可证约束，非完全开源。
- **关键超参**：KL 权重 $K=0.1$，学习率 $5 \times 10^{-7}$（余弦衰减至 $5 \times 10^{-8}$），warmup 占比 0.1，全局批次大小 16，序列长度 65,536，Muon momentum=0.95，权重衰减 0.1。
- **硬件**：生产训练使用 8 节点 × 8 × NVIDIA H200 GPU（共 64 GPU），开发阶段使用 12 节点（共 96 GPU）。
- **并行配置**：TP=8, PP=4, EP=16, CP=2, DP=1，CPU-offloaded 优化器状态。
