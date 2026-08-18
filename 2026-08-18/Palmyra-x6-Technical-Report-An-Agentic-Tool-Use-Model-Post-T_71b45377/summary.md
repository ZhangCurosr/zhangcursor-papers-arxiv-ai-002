---
title: "Palmyra-x6-Technical-Report-An-Agentic-Tool-Use-Model-Post-T"
source: https://arxiv.org/pdf/2608.16620v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:08:10"
field: "Agentic 大模型微调"
keywords: ["Agentic LLM", "Anchored SFT", "Muon optimizer", "tool-use", "self-distillation", "MoE post-training", "synthetic trajectory", "model safety"]
innovations: ["ASFT锚定微调在626条轨迹上成功微调744B MoE模型而不损伤基座能力", "Muon+Adam混合优化器适配GLM-5.2 GlmMoeDsa架构的post-training实践", "自蒸馏式多轮工具调用合成数据生成管线（verifier+cheating filter双重门控）"]
benchmarks: ["MCP-Atlas", "FinanceBench", "IFBench", "BFCL Core", "FORTRESS", "Washington Post ModelSlant", "BBQ", "TruthfulQA", "OR-Bench"]
---

# 论文速读：Palmyra-x6-Technical-Report-An-Agentic-Tool-Use-Model-Post-T

## 一句话总结
Palmyra x6 是 Writer 公司基于开源 GLM-5.2（744B MoE，~40B 激活）构建的 Agentic 工具使用模型，通过锚定监督微调（ASFT）在仅 626 条高质量合成轨迹上完成单轮训练，显著提升多步骤规划、工具调用与长程任务完成能力，在多个 Agentic 基准上超越前代默认模型及多家人工智能前沿模型。

## 研究问题与动机
- **小数据高效微调大模型**：传统做法需从头预训练或完整 mid-training + post-training 链路；本文证明高质量数据 + 合适目标函数仅需 post-training 即可适配特定能力。
- **SFT 能力退化问题**：纯监督微调（SFT）的梯度将模型推向外部专家分布，破坏已有能力；需要一种"不损伤基础能力"的微调机制。
- **Agentic 工具使用场景**：企业场景中模型需在 MCP 套件、Web 搜索、代码执行沙箱、RAG 管道等工具丰富环境中完成多步规划与长程任务。
- **安全与行为漂移控制**：微调过程中需防止模型在工具调用行为上的退化与对齐偏移，同时保留 base model 的拒绝与安全能力。

## 核心贡献（创新点）
1. **ASFT（锚定监督微调）配方落地**：结合 DFT token 概率加权与 KL 锚点（K=0.1），实现仅 626 条轨迹微调 744B 参数模型而不损害基础能力。
2. **Muon + Adam 混合优化器适配 MoE**：对 2D 权重矩阵使用 Muon（正交化牛顿-舒尔茨），对 embedding/router/norm 等使用 Adam，在 GLM-5.2 MoE 架构上验证可行性。
3. **自蒸馏式合成轨迹生成管线**：将 SDFT 思想扩展到多轮工具调用场景，通过 teacher demonstration → student on-policy rollout → verifier + cheating filter 生成高质量自一致数据。
4. **面向 Agentic 场景的系统化安全评估**：结合 FORTRESS  adversarial 评测、政治偏见配对测试、Washington Post ModelSlant 等，证明 ASFT 在保持安全能力方面的有效性。

## 方法详解

### 模型架构
- 继承 **GLM-5.2 GlmMoeDsa** 架构：**MoE FFN（256 experts, top-8）+ MLA（64 heads）+ DSA indexer + IndexShare（跨层复用稀疏注意力索引）+ RoPE + 1 next-n multi-token prediction 层**。
- 总参数 744B / 激活 ~40B，78 层（前 3 层 dense，后 75 层 MoE），hidden size 6144，vocab 154,880，context length 65,536 tokens。

### 数据合成管线
- **教师-学生自蒸馏**：teacher 生成带 reasoning trace 的完整轨迹 → 裁剪为 strategy-level demonstration（保留推理大纲与工具调用顺序，遮蔽具体数值/日期/参数）→ 注入 student system prompt → student 调用真实工具生成 on-policy rollout。
- **两重过滤门**：① Verifier 对比 teacher 参考答案打分（high-quality 1.0 / low-quality 0.5 / incorrect 0.0 / refusal 0.0，≥0.5 通过，三次投票取平均）；② 作弊过滤器（4-gram 重叠 >0.8、显式引用 demonstration、有工具可用却无 tool call 则拒绝）。
- **Effort caps**：最多 20 次 tool call、最多 4 次连续相同工具、最多 3 次 error、最多 3 次重复相同调用，超限即放弃重试。
- **12 数据集主配方**：金融研究、数据分析/编码、医学 agent、MCP 工具套件、模拟世界长程任务、RAG，共 626 条精选轨迹。

### 训练目标：ASFT
$$\mathcal{L} = -\mathrm{mean}(P \cdot \log p_\theta) + K \cdot \mathrm{KL}(\pi_{\mathrm{ref}} \| \pi_\theta), \quad K=0.1$$
- **DFT 加权**：每个 token 的 NLL 乘以其自身 detached 概率 $P(y_t)=\exp(\log p_\theta(y_t)).\mathrm{detach()}$，聚焦模型已有信心的 token。
- **KL 锚点**：使用 $k_3$ 估计器 $k_3 = \pi_{\mathrm{ref}}/\pi_\theta - 1 - \log(\pi_{\mathrm{ref}}/\pi_\theta)$，token-wise 计算，防止分布漂移。

### 优化器：Muon + Adam
- **Muon** 用于 attention projection、dense FFN、MoE expert FFN 的 2D 权重矩阵：momentum buffer 经 Newton-Schulz 迭代正交化，RMS-matched scaling 统一有效步长；采用 **Muon Split**（按 head 拆分 MLA 投影后再正交化，避免跨 head 耦合）。
- **Adam** 用于 embedding、output head、RMSNorm gain、router、bias、scalars。
- 单 momentum（≈0.95），无二阶矩估计，state 减半；weight decay=0.1；optimizer state CPU-offload。

### 训练配置
- lr=$5\times10^{-7}$，cosine decay 至 $5\times10^{-8}$，warmup 0.1 fraction（初始 $1\times10^{-8}$），weight decay=0.1，global batch size=16，context=65536，TP8·PP4·EP16·ETP1·CP2·DP1，BF16 权重，FP32 梯度/all-reduce，FlashAttention。

## 实验与结果

### 评估协议
- 在 WRITER 内部评估基础设施中对比 Palmyra x6 vs. Writer Agent 前代默认模型（prior default），以及 vs. 五款公开 frontier 模型；分数范围 0–1，误差条为一个标准误。

### 主要结果
- **vs. prior default**：Palmyra x6 在全部六个基准上领先，最大提升：**MCP-Atlas +0.320**、**FinanceBench +0.305**、**IFBench +0.304**。
- **vs. frontier models**：Palmyra x6 在 **BFCL Core 达 0.785**，六基准平均 **0.765**，为 cohort 最高。

### 安全与偏见评估
- **Washington Post ModelSlant**：80% 热点问题呈现双方观点，净左倾最小。
- **政治配对评测（556 families）**：英语无统计显著质量不对称，paired refusal mismatch <5%，BBQ 去歧准确率 90.9%+，TruthfulQA 80.6%（基座 81.5%）。
- **FORTRESS**：模型+system message 对抗安全评分 67.0（基座 58.4，提升 8.6 分），良性有用性 96.4；无 system message 时 55.4/97.4，与基座基本一致。

## 相关工作脉络
1. **ASFT (Zhu et al., ICLR 2026)**：本文锚定微调的直接方法来源；核心区别在于本文将其应用于 744B MoE Agentic 场景，并配合 Muon 优化器与合成轨迹管线。
2. **SDFT (Shenfeld et al., 2026)**：自蒸馏微调思想源头；本文将其从单轮文本扩展到多轮工具调用场景，引入 verifier + cheating filter 双重门控。
3. **Muon (Jordan et al., 2024; Liu et al., 2025)**：正交化优化器；本文首次在其 Muon Split 变体上验证 GLM-5.2 MLA 结构的适配可行性。
4. **DFT (Wu et al., 2025)**：token 概率加权训练；本文将其与 KL 锚点结合，形成 ASFT 损失函数。
5. **LIMO / LIMI ("Less is more")**：少数据高效微调理念的前置工作；本文以 626 条轨迹微调 744B 模型为其提供了 Agentic 场景下的实证支持。
6. **GLM-5.2 (z.ai, 2026)**：基座模型；本文未修改架构，仅继承 GlmMoeDsa 并在其上进行 post-training。

## 局限性与未来方向
- **能力专业化边界**：非 Agentic 任务行为完全由基座决定，模型本身无额外通用能力提升。
- **数据覆盖有限**：仅 12 个任务域、单次 epoch、626 条轨迹，域外工具生态或未见过场景可能效果有限。
- **安全假设未完全验证**：KL 锚点保留对齐/拒绝行为的假设仅停留在理论推断，论文明确声明"未用严格实验验证"。
- **部署依赖**：推理需支持 DSA IndexShare 的构建（如 SGLang ≥ 0.5.13.post1），增加部署复杂度。

## 研究启发与可借鉴点
- **"少而精"数据的可行性**：对于大模型定向能力增强，626 条经过严格过滤的高质量轨迹足以产生显著提升，呼应 LIMO/LIMI 理念，适合本团队在高价值但数据稀缺的场景中借鉴。
- **ASFT + Muon 混合优化的工程范式**：KL 锚点控制漂移 + Muon 加速 2D 矩阵更新的组合，可在 MoE 架构的 post-training 中作为默认配置参考。
- **自蒸馏轨迹生成管线设计**：teacher demonstration → student on-policy rollout → verifier + leakage filter 的两层门控机制，可迁移至其他需要合成 agent 数据的任务。
- **系统化安全评估框架**：政治偏见配对测试 + FORTRESS adversarial 评测的组合，为 Agentic 模型的安全评估提供了可复用的 benchmarking 模板。
- **工具调用 effort caps 的工程实践**：对 tool call 次数、连续重复、error 率的硬上限约束，是防止 agent 陷入循环/浪费的有效工程技巧。

## 关键术语表
**ASFT（Anchored Supervised Fine-Tuning）**：锚定监督微调，在 SFT 损失基础上增加 KL 锚点项以限制分布漂移的方法。
**Muon**：基于 Newton-Schulz 正交化的优化器，将 2D 权重矩阵视为几何对象而非独立标量集合，提升训练效率。
**SDFT（Self-Distillation Fine-Tuning）**：自蒸馏微调，模型同时作为 teacher（含 demonstration context）和 student（仅任务 prompt），消除 SFT 的分布错配。
**DFT（Detached Fine-Tuning）**：用模型自身 detached 概率加权每个 token 的 NLL，聚焦已有信心 token 的训练信号。
**GlmMoeDsa**：GLM-5.x 系列采用的 MoE + MLA + DeepSeek Sparse Attention indexer 的混合架构。
**DSA IndexShare**：跨层复用稀疏注意力 indexer 的选中索引，减少重复计算，但对推理栈有兼容性要求。
**MCP（Model Context Protocol）**：标准化的工具调用协议框架，本文评估的重要场景之一。
**FORTRESS**：前沿安全风险评测基准，含 500 条专家编写 adversarial prompts（CBRNE、政治暴力、犯罪金融等）及对应 benign 配对。

## 可复现要素
- **数据集**：12 个私有数据集（金融、编码、医学、MCP、模拟世界、RAG），合成轨迹 626 条，**未公开**。
- **代码/权重**：BF16 权重已发布；FP4 量化版可用于高效推理；推理需 SGLang ≥ 0.5.13.post1 或等效支持 DSA IndexShare 的栈；**许可为 Writer 商业许可**。
- **关键超参**：KL weight K=0.1，lr=$5\times10^{-7}$（cosine decay 至 $5\times10^{-8}$），global batch size=16，single epoch，sequence length=65,536，weight decay=0.1，Muon momentum≈0.95（Nesterov）。
