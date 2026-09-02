---
title: "Instella-MoE-Technical-Report"
source: https://arxiv.org/pdf/2609.00791v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 12:08:30"
field: "高效稀疏 MoE 语言模型训练"
keywords: ["Mixture-of-Experts", "Gated MLA", "FarSkip-Collective", "fully open LLM", "MoE training", "DPO", "RL post-training", "MOPD"]
innovations: ["Gated MLA 注意力门控机制提升表达力", "FarSkip-Collective 通信-计算重叠提升训练/推理效率", "完整多阶段可复现 MoE 训练管线（含反馈驱动 SFT 数据筛选）"]
benchmarks: ["HELMET", "RULER", "MMLU", "GSM8K", "HumanEval+", "IFEval", "AIME24/25", "LCB", "BBH", "AGIEval", "AlpacaEval 2"]
---

# 论文速读：Instella-MoE-Technical-Report

## 一句话总结
Instella-MoE 是一个完全开源的 16B 总参数 MoE 语言模型（每 token 激活 2.8B 参数），通过 Gated MLA 注意力增强和 FarSkip-Collective 通信-计算重叠设计，在 AMD Instinct GPU 上从头训练，实现了领先的开源性能。

## 研究问题与动机
1. 当前 AI 进展主要由专有模型（GPT-5.6、Claude Fable 5、Gemini 3.1 Pro 等）主导，模型权重、训练数据和方法均不透明，阻碍科学理解、可复现性和公平访问。
2. 即使近年出现大量"open-weight"模型（GLM-5.3、Qwen3.8-Max、Llama 4 等），其预训练语料、预处理流程和完整训练配方仍不完整公开，研究者无法完全复现结果或审计数据污染。
3. 尽管稀疏激活的 MoE 架构已成为提升 LLM 性价比的核心策略，但**完全开源且性能竞争力强的 MoE 模型仍然罕见**（OLMoE-1B-7B 仅有 6.9B 总参数/1.3B 激活参数）。
4. 在 AMD 硬件平台上缺乏可复现的完整 MoE 训练流程与透明研究成果。

## 核心贡献（创新点）
1. **Gated Multi-head Latent Attention (Gated MLA)**：在 MLA 输出投影前插入输入条件 sigmoid 门控，动态调节每个注意力头的输出通道；相比 vanilla MLA，200B token 消融实验中平均提升 +0.47 分，代码和 MMLU 增益最大。
2. **FarSkip-Collective 通信-计算重叠架构**：修改 MoE 层连接方式，使 Dispatch/Combine all-to-all 通信可与独立计算重叠，预训练吞吐提升 12.7%，推理 TTFT 吞吐提升 39.2%；消融显示对模型能力无显著损害（50.38 vs 50.33）。
3. **完整多阶段训练管线**：涵盖预训练→中期训练（model souping）→长上下文扩展→反馈驱动 SFT→DPO→IF-RL+MOPD，所有阶段权重、配置、数据配比和代码全部开源，支持透明可复现研究。
4. **DPO 阶段的负载平衡抑制**：发现 MoE 在 DPO 中若保持负载均衡机制会导致严重路由漂移（top-1 expert 差异从 5% 升至 54%），因此禁用 bias 更新和辅助 loss，从而保留推理能力。

## 方法详解
**架构设计**：
- 27 层 Decoder-only Transformer（1 dense + 26 MoE），总参数 16B，激活参数 2.8B/token
- 每层 MoE：64 个路由专家（FFN size 1,408）中选 top-6，加 2 个共享专家（FFN size 2,816，fused）
- Router 使用 DeepSeek-V3 风格的 bias-based loss-free 负载均衡，target load 速率 $1 \times 10^{-3}$，辅助 loss 系数 $1 \times 10^{-4}$

**Gated MLA 公式**：
$$y_t = W_o [\text{Sigmoid}(W_g x_t) \odot \hat{o}_t]$$
其中 $x_t$ 为 RMSNorm 输入，$\hat{o}_t$ 为多头注意力拼接输出，$W_g$ 为门控投影。

**FarSkip-Collective 原理**：
- 标准 Transformer：$h_{k} = \sum_{i=1}^{k} f_i(h_{i-1})$，通信依赖前序层完成
- FarSkip 使用部分/过时激活 $\hat{h}_k$ 进行下一层计算，同时通信并行：
  - $\text{moe-in}_k = h_{k-1}$（避开 attention 输出依赖）
  - $\text{atm-in}_k = h_{k-2} + \text{attn-out}_{k-1} + \text{shared-exp-out}_{k-1}$（跳过 routed expert）
- 将 Dispatch/Combine 重叠窗口扩展至整个 layer 时长

**训练管线细节**：
- 预训练：7.1T tokens，4K 上下文，global BS=4096，peak LR=$4 \times 10^{-4}$，WSD scheduler，MTP loss 权重 0.3
- 中期训练：~100B tokens，三个 variant 通过 model souping 融合
- 长上下文扩展：YaRN RoPE（$\theta=8\times10^6$），Stage 1 用 Dolma 3 Longmino 100B（~194B tokens），Stage 2 用 37.32B STEM 混合数据恢复数学/代码能力
- SFT：Phase 1 用 Dolci-Think-SFT-7B + 针对性技能切片；Phase 2 用**反馈驱动数据筛选**（judge 模型诊断错误→reflection 模型生成检索查询→embedding 检索，512K 预算）
- DPO：Dolci-Think-DPO-7B，delta learning 思想（强 Qwen3-32B vs 弱 Qwen3-0.6B 对比），$\beta=5.0$，禁用负载平衡
- RL：Stage 1 IF-RLVR（GRPO+DAPO+R3，1400 steps）；Stage 2 MOPD（domain-routed teacher，token-level reverse KL）

## 实验与结果
**预训练基准（Base checkpoint）**：
- 12 项基准平均：**76.7**（fully open 模型最高）
- 超越 OLMo-3-7B（70.1）、SmolLM3-3B（70.5）、OLMoE-1B-7B（61.9）
- 编码能力突出：HumanEval+ 65.7，MBPP+ 57.6
- 长上下文：HELMET 41.5 / RULER 79.4（64K）

**后训练基准（Think checkpoint）**：
- 12 项基准平均：**73.2**（fully open 模型最高）
- 超越 OLMo-3-7B-Think（72.0）、Gemma-4-E4B（70.5）、Qwen3.5-4B（69.7）
- IFEval 83.7，AIME25 73.4，LCB 54.3

**效率提升**：
- FarSkip-Collective：预训练吞吐 +12.7%，推理 TTFT 吞吐 +39.2%

**消融结果**：
- Gated MLA vs Vanilla MLA：avg 50.33 vs 49.86（+0.47）
- Gated MLA + FarSkip：avg 50.38（与 Gated MLA 持平）
- RL 消融：IF-RL 单独使用导致 AIME/GPQA 回退，MOPD 恢复大部分 IF 增益同时保持数学/代码性能

## 相关工作脉络
1. **OLMoE (Muennighof et al., 2025)**：建立了完全开源 MoE 基线（6.9B total / 1.3B active），Instella-MoE 在其基础上将激活参数提升至 2.8B，并实现完整开源管线。
2. **Marco-MoE (Jiang et al., 2026)**：通过 dense checkpoint upcycling 将稀疏训练扩展到多语言；Instella-MoE 则完全从头训练，数据更透明。
3. **DeepSeek-V3 (DeepSeek-AI, 2024a)**：采用细粒度 shared-plus-routed MoE 设计；Instella-MoE 借鉴此设计并引入 Gated MLA 和 FarSkip-Collective 改进。
4. **Gated Attention (Qiu et al., 2025)**：提出输入条件门控注意力；Instella-MoE 将其适配到 MLA 框架并验证其有效性。
5. **FarSkip-Collective (Dukler et al., 2026)**：原论文提出概念；本文进行了大规模架构消融并部署到 16B 级别 MoE 的实际训练中。
6. **MOPD (Ma et al., 2026)**：多教师在线蒸馏方法；本文将其应用到 IF 专家知识与通用能力的整合。

## 局限性与未来方向
- **硬件依赖**：全程基于 AMD Instinct MI300X/MI325X，在其他 GPU 架构（如 NVIDIA H100）上的表现未评估。
- **长上下文小幅回退**：Stage 2 STEM 恢复后，HELMET 和 RULER 分数相比 Stage 1 有所下降（43.7→41.5，83.9→79.4），体现长上下文与短上下文 STEM 能力的 trade-off。
- **模型规模限制**：16B total / 2.8B active 仍处于中小规模，与前沿 proprietary 模型（如 GPT-5.6、Gemini 3.1 Pro）存在数量级差距。
- **论文未明确讨论**的潜在方向：多模态扩展、更高效的路由机制、更大规模的 fully open MoE 探索。

## 研究启发与可借鉴点
1. **反馈驱动数据筛选**：通过 judge 模型诊断 student 错误→reflection 模型生成检索查询→embedding 检索的闭环，可在 SFT 后期显著提升针对性能力（+1.5 平均，IFEval +4.8）。
2. **DPO 中禁用 MoE 负载平衡**：发现 DPO 低学习率下维持 bias 更新会导致严重路由漂移（54% top-1 expert 变化），禁用后效果显著改善——此为 MoE 偏好优化中的重要工程经验。
3. **多阶段长上下文扩展策略**：Stage 1 用多样化长文本适应 64K，Stage 2 用 STEM 密集数据恢复能力，平衡长上下文与短期技能。
4. **MOPD 领域路由蒸馏**：通过 domain router 将 IF rollout 路由到 IF-RL teacher，其他 rollout 路由到 DPO teacher（self-anchor），实现能力整合而不灾难性遗忘。
5. **Model souping 用于中期训练**：三个 variant 独立训练后等权融合，优于任何单一 variant（avg 80.90 vs 79.23-79.73）。

## 关键术语表
**Gated MLA**：在 Multi-head Latent Attention 输出投影前插入输入条件 sigmoid 门控，动态调节各注意力头输出的低秩压缩表示。
**FarSkip-Collective**：修改 MoE 层连接以允许 Dispatch/Combine all-to-all 通信与 attention/shared-expert 计算重叠的系统级优化。
**Multi-Token Prediction (MTP)**：在预训练/中期训练中附加的单层预测头，额外预测下一个 token 以提供丰富训练信号。
**Direct Preference Optimization (DPO)**：通过对比 chosen/rejected 响应对直接优化策略，无需显式 reward model。
**Multi-Teacher On-Policy Distillation (MOPD)**：通过领域路由将 student rollout 分发给不同 frozen teacher，以 token-level reverse KL 整合专项能力。
**Rollout Routing Replay (R3)**：记录采样时的 expert assignment 并在训练前向中重放，确保 $\log\pi^{train}$ 与采样路径一致。
**Truncated Importance Sampling (TIS)**：对 policy ratio 施加上下界截断，修正推理与训练引擎间的数值不匹配。
**Delta Learning**：偏好优化的核心假设——驱动改进的是 chosen 与 rejected 响应的质量差，而非 chosen 的绝对质量。

## 可复现要素
- **代码**：开源，GitHub 仓库 `github.com/AMD-AGI/Instella-MoE`
- **权重**：所有训练阶段 checkpoint（Pretrain/Midtrain/Base/SFT/DPO/Think）全部开源
- **训练配置**：完整超参数表格（Table 3）、数据配比（Table 4）公开
- **预训练数据**：Nemotron-CC-v2、Nemotron-CC-Math-v1、MegaMath、FineMath、RefineCode 等公开数据集
- **关键超参**：EP=8，bf16 mixed precision，peak LR=$4\times10^{-4}$（pretrain）/$2\times10^{-4}$（midtrain/longctx）/$1\times10^{-4}$（SFT/DPO/RL）
