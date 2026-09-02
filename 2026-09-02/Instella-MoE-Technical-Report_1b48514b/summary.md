---
title: "Instella-MoE-Technical-Report"
source: https://arxiv.org/pdf/2609.00791v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 19:25:33"
field: "大规模稀疏语言模型训练与训练系统"
keywords: ["Mixture-of-Experts", "fully open LLM", "Gated MLA", "FarSkip-Collective", "RL post-training", "multi-stage training"]
innovations: ["Gated MLA：在 MLA 输出端加入输入条件 sigmoid 门控以提升表达能力", "FarSkip-Collective：通过部分/延迟激活打破 EP all-to-all 依赖，实现通信-计算重叠", "MOPD：领域路由双教师 on-policy 蒸馏，将 IF-RL 专项能力合并回主干而不遗忘"]
benchmarks: ["MMLU", "GSM8K", "HumanEval+", "IFEval", "AIME24/25", "LiveCodeBench", "HELMET", "RULER", "BBH", "GPQA", "AlpacaEval 2"]
---

# 论文速读：Instella-MoE-Technical-Report

## 一句话总结
本文提出了 Instella-MoE，一个完全开源的 16B 总参数/2.8B 激活参数 MoE 语言模型，在 AMD Instinct GPU 上从零训练完成；通过 Gated MLA 和 FarSkip-Collective 架构创新，配合多阶段训练管道，在基座模型和 post-training 后均达到同参数量级最强开源表现。

## 研究问题与动机
- 主流大模型进展多由闭源商业系统（如 GPT-5.6、Claude Fable 5）主导，权重、数据、方法不透明，阻碍科学复现与可审计性。
- 当前"开源权重"模型仍多未公开完整训练配方、预处理管线与数据配比，研究者无法复现或审计数据污染。
- 稀疏 MoE 架构已成为改善计算-性能性价比的核心策略，但具备竞争力且训练完全透明的"全开源 MoE"仍稀缺。
- MoE 的专家并行（EP）通信开销会阻塞计算；如何在保证性能的同时实现高效训练/推理，是需要解决的系统瓶颈。

## 核心贡献（创新点）
- 提出 Instella-MoE-16B-A3B 全开源 MoE 基座与 Think 系列权重，覆盖从 pretrain→midtrain→long-context→SFT→DPO→RL 的完整可复现流程。
  - 与已有工作的本质区别：同时开源每一阶段 checkpoint、数据配比、训练配置与代码，且基于 AMD GPU 从零训练，而非对现有权重进行 upcycle。
- 设计 Gated Multi-head Latent Attention（Gated MLA），在 MLA 的注意力输出与投影之间加入输入条件 sigmoid 门控。
  - 与已有 MLA 的差异：额外引入可学习门控以逐 head/value 通道调制注意力输出，提升表达能力，尤其在代码与知识类基准上增益明显。
- 设计 FarSkip-Collective 层间连通模式，使 attention 与 shared expert 计算不再依赖 Dispatch/Combine 的 all-to-all 结果，从而将通信与计算重叠。
  - 与标准 MoE 的差异：放宽对前一层完整激活的同步等待，允许用部分/延迟激活推进后续计算，显著减少通信气泡。
- 提出面向 MoE 的 RL 稳定训练方案：DPO 阶段关闭负载均衡器偏置更新与辅助 loss 以避免路由漂移；RL 阶段引入 Rollout Routing Replay（R3）+ token-level TIS 保证 train/inference 一致性；并用 MOPD 将 IF 专项 RL 收益蒸馏回 DPO 模型而避免灾难性遗忘。
  - 与通用 RL/DPO 的差异：针对 MoE 路由不稳定性与 teacher-student 领域路由分歧做专门适配。

## 方法详解
- **模型结构**：decoder-only Transformer，27 层（1 个稠密 FFN + 26 个 MoE）；每 MoE 层 top-6/64 routed experts（FFN 维度 1,408）+ 2 个 shared experts（融合 FFN 维度 2,816）；router top-K 缩放因子 2.5；hidden size 2,048；16 头 Gated MLA，KV latent rank=512，QK content/RoPE=96/32，value dim=128；RMSNorm + SwiGLU + RoPE；vocab 128,896（DeepSeek-V3 tokenizer）。
- **Gated MLA**：$y_t = W_o[\operatorname{Sigmoid}(W_g x_t) \odot \hat{o}_t]$，门控 $W_g$ 由 RMSNorm 归一化后的层输入 $x_t$ 计算，对每个 head 的 $d_v$ 个值通道做逐元素乘法调制后再经 $W_o$ 输出。
- **负载均衡**：主用 bias-based loss-free 方案（bias 仅加到 router score 不改 mixture weights），更新速率 $1\times10^{-3}$；辅用 sequence-level 辅助 loss（系数 $1\times10^{-4}$）；DPO 阶段冻结 bias 并将辅助 loss 系数置零。
- **Multi-Token Prediction（MTP）**：预训练/中期训练阶段在顶层附加 1 层 MTP head，预测下一个未来 token，pretrain 权重 0.3、midtrain 权重 0.1；长上下文及之后禁用。
- **FarSkip-Collective**：标准 Transformer 中 $h_k=f_1(h_0)+\dots+f_k(h_{k-1})$；FarSkip 改用可用激活 $\hat{h}_k$ 推进下一子块，使 Dispatch 与 attention/shared-expert 计算并行。具体：$\text{moe-in}_k=\hat{h}_k(\text{moe})=h_{k-1}$，$\text{atm-in}_k=h_{k-2}+\text{attn-out}_{k-1}+\text{shared-exp-out}_{k-1}$，从而实现 Dispatch/Combine 与 layer 内独立计算的完全重叠。
- **训练管道**：
  - Pre-training：7.1T tokens，context 4K，global BS=4096，AdamW（$\beta_1=0.9,\beta_2=0.95$），WD=0.1，WSD LR peak $4\times10^{-4}$、min $4\times10^{-5}$，warmup 2000。
  - Mid-training：~100B tokens，三变体（v1/v2/v3 逐步上调 STEM/reasoning 子集），peak LR $2\times10^{-4}$，equal-weight model souping。
  - Long-context Stage 1：~194B tokens，64K，YaRN（$\theta=8\times10^6$），doc masking；Stage 2：~20B tokens，STEM-balanced 混合恢复数学/代码。
  - SFT：~2.9M 样本，32K context，2 epochs；phase-2 采用 feedback-driven curation（judge 打分→结构化 error analysis→reflection model 生成检索 query→embedding 近邻选取 512K）。
  - DPO：150K contrastive pairs（Qwen3-32B vs Qwen3-0.6B，delta learning），$\beta=5.0$，0.75 epoch，关闭负载均衡。
  - IF-specialized RL：GRPO + DAPO/Dr.GRPO 改进（zero-gradient filter、active sampling、token-level loss、无 KL、clip-higher、无 std normalization、TIS、R3），1400 steps，verifier-based reward（约束满足比例），think-format gate。
  - MOPD：domain-routed 双教师蒸馏（IF prompt→IF-RL teacher，其余→frozen DPO teacher），token-level reverse KL，advantage $A_t=\log\pi_{teacher}-\log\pi_{student}$，自锚定防止非 IF 能力退化。
- **基础设施**：Primus（Megatron-LM 后端，ROCm，EP=8，FlashAttention，bf16）用于 pretrain–DPO；Miles（SGLang rollout + Ray 编排）用于 RL；FarSkip 在 backward 通过 race-safe async op + Sequence Number API 实现梯度回传安全。

## 实验与结果
- **数据与基线**：与 OLMo-3-7B、SmolLM3-3B、OLMoE-1B-7B、Moonlight-16B-A3B、Gemma-4-E4B、Qwen3.5-4B 等 fully open 与 open-weight 模型对比。
- **Base model（4K→64K 扩展后）**：12 项预训练基准平均 76.7，最高 fully open；WinoGrande 86.5、HumanEval+ 65.7；HELMET avg 41.5、RULER avg 79.4（64K）。
- **Post-trained Think**：11 项指令/推理/数学/代码/聊天基准平均 73.2，超过 fully open OLMo-3-7B-Think（71.97）及 open-weight Gemma-4-E4B（70.47）、Qwen3.5-4B（69.73）；IFEval 83.70、AIME25 73.40、LCB 54.30 领先。
- **架构消融（200B tokens）**：Vanilla MLA 平均 49.86 → Gated MLA 50.33（+0.47，代码/MMLU/GPQA 增益最大）→ +FarSkip 50.38。
- **RL 消融**：IF-RL 将 IFEval 从 77.1 提升至 84.1 但 AIME/GPQA 下降；MOPD 恢复为 IFEval 83.7 且 11 项平均 75.7，证明蒸馏策略有效。
- **效率**：FarSkip 预训练吞吐 +12.7%；expert-parallel SGLang 推理 TTFT 吞吐 +39.2%。
- **SFT 数据选择**：feedback-driven curation 较 uniform sampling 平均 +1.5（IFEval +4.8、AIME25 +3.6、LCB +3.2）。
- **DPO 路由稳定性**：开启 load balancing 时 top-1 专家与 SFT 不同比例 54%、top-6 Jaccard 0.42；关闭后分别降至 5% 和 0.94，证明 DPO 需关闭负载均衡以保留路由。

## 相关工作脉络
- OLMoE-1B-7B / OLMo-3-7B：先前的 fully open 基座，Instella-MoE 在激活参数更少（2.8B vs 7B）下实现更高基准分，体现 MoE 效率优势。
- SmolLM3-3B：fully open dense 模型，Instella-MoE 在同等/更低激活规模下全面超越。
- Moonlight-16B-A3B：open-weight MoE 对照；Instella-MoE 作为 fully open 版本达到相近甚至更优表现，强调透明度价值。
- Qwen3.5-4B / Gemma-4-E4B：open-weight 同/更大 active 参数基线；post-training 阶段 Instella-MoE Think 超越，体现多阶段 pipeline 的有效性。
- DeepSeek-V2/V3 MLA 与 MoE 设计：Gated MLA 在其 MLA 基础上增加输入条件门控；FarSkip-Collective 在其 EP 依赖结构上引入重叠策略。
- OLMo 3 RL / DAPO / Dr.GRPO / STAT：本文在 GRPO 家族改进上集成零梯度过滤、TIS、active sampling 等，并用 feedback-driven curation 区别于 STAT 的预定义技能分类选择。

## 局限性与未来方向
- 长上下文短程能力牺牲：Stage 2 STEM 恢复使 HELMET/RULER 均值较 Stage 1 下降（43.7→41.5、83.9→79.4），在超长语境任务上仍有提升空间。
- MoE 路由对齐难题：尽管 DPO 阶段关闭负载均衡有效缓解漂移，但 RL 阶段仍需依赖 R3+TIS 校正 train/inference 不一致，说明 MoE 动态路由与策略优化的耦合仍未根本解决。
- 仅训练于 AMD Instinct 平台：系统级优化（Primus/Miles/FarSkip）针对 AMD 硬件 tuned，跨平台迁移需要再验证。
- 仅文本模态：当前为 text-only，未来可能向多模态扩展。
- 训练数据依赖第三方开放集合（Nemotron、Dolma 系列等），其长期可用性与版权状态未深入讨论。

## 研究启发与可借鉴点
- Feedback-driven data curation：用 judge 模型生成结构化 error analysis→reflection model 转换为检索 query→embedding 近邻采样，相比均匀采样可稳定提升特定弱项；该流程可迁移到任何 SFT 阶段的能力修补。
- Delta-learning DPO 配对：用强模型与弱模型的响应对做对比，而非两份高质量响应，降低对高质量偏好数据的依赖；适合算力受限场景。
- Multi-Teacher On-Policy Distillation：将专项 RL 产生的能力通过 domain-routed 双教师蒸馏回主干，避免单目标 RL 引起的多任务退化；可推广到多能力（代码/数学/指令）分别 RL 后的融合。
- FarSkip-Collective 通信重叠范式：通过引入部分/延迟激活打破 all-to-all 依赖链，适用于任何 EP 密集的 MoE/序列并行组合场景，有望复制到其它开源框架。
- MTP 衰减权重策略：pretrain 用 0.3、midtrain 降至 0.1、长上下文关闭，体现多目标训练中辅助 loss 应随数据质量与序列长度动态退火的设计思路。

## 关键术语表
- **Mixture-of-Experts (MoE)**： decoder 层内包含多个并行专家 FFN，每 token 由 router 选 top-K 专家处理，实现总参数大、单 token 计算小的稀疏架构。
- **Gated MLA**：在 DeepSeek MLA 的注意力输出后加入输入条件 sigmoid 门控，逐 head/value 通道调制后再投影，增强注意力表达力。
- **FarSkip-Collective**：一种 MoE 层连通设计，用前一层部分/延迟激活驱动后续子块，使得 Dispatch/Combine 与 attention/shared-expert 计算在时间上重叠，减少通信气泡。
- **Model souping**：将多个在不同数据配比下训练的 checkpoint 按等权相加，常能获得优于任一单模型的泛化表现。
- **Delta learning（DPO 场景）**：偏好对由强-弱模型响应构成，训练信号来源于两者差距，即使强响应本身不可继续模仿也能有效优化。
- **IF-RLVR**：instruction-following reinforcement learning with verifiable rewards，利用确定性约束检查函数给出 [0,10] 分数，避免 reward model 被长推理劫持。
- **MOPD（Multi-Teacher On-Policy Distillation）**：在 student 自身 on-policy rollout 上，按 prompt 领域路由到不同 frozen teacher，用 token-level reverse KL 蒸馏合并专项能力。
- **Rollout Routing Replay (R3) + TIS**：R3 回放 rollout 阶段的专家分配以统一 log-prob 路径；TIS 对残差重要性采样比率做截断，联合消除 MoE train/inference 不一致。

## 可复现要素
- 数据集：主要使用公开/开源数据集（Nemotron-CC、Dolma 3 Dolmino、MegaMath、FineMath、RefineCode、TxT360、Instella-GSM8K-synthetic、Dolci-Think 系列）；license 多为依赖型或 ODC-BY-1.0。
- 代码/权重：论文声明开源全部阶段 checkpoint、训练配置、数据配比、训练代码与评测协议；模型仓库见 github.com/AMD-AGI/Instella-MoE。
- 关键超参：EP=8，bf16；pretrain LR peak $4\times10^{-4}$，WSD；midtrain/long-ctx LR $1\text{-}2\times10^{-4}$；SFT LR $1\times10^{-4}$、2 epochs；DPO $\beta=5.0$、LR $8\times10^{-8}$；RL LR $1\times10^{-6}$、GRPO clip (0.2, 0.272)、TIS clip [0.5, 1.5]/[0.5, 2.0]；R3 启用。
