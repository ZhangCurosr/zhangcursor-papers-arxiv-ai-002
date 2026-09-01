---
title: "Q-Strata-Hierarchical-Bit-Allocation-for-Mixed-Precision-Qua"
source: https://arxiv.org/pdf/2608.30564v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 23:43:54"
field: "低比特量化与MoE压缩"
keywords: ["MoE", "mixed-precision quantization", "mixed-precision quantization", "weight-only PTQ", "Jensen-Shannon divergence", "greedy descent", "lazy evaluation"]
innovations: ["双层分配器：内层代理缓存块内候选、外层懒贪心直接优化模型级JSD", "搜索维度从3LE降至L，捕获跨块耦合", "外层搜索作为稠密模型通用分配器，6.9×少于AMQ查询"]
benchmarks: ["WikiText2", "PIQA", "BoolQ", "WinoGrande", "ARC-easy", "ARC-challenge", "HellaSwag", "MMLU", "GSM8K"]
---

# 论文速读：Q-Strata-Hierarchical-Bit-Allocation-for-Mixed-Precision-Qua

## 一句话总结
本文提出 Q-STRATA，一种针对 MoE LLM 的双层混合精度量化分配器：内层利用低成本代理缓存每个 MoE 块的候选分配，外层直接优化模型级目标（JSD）进行跨块预算分配，显著降低了搜索空间并捕获了块间耦合效应。在 Mixtral-8x7B、Qwen1.5-MoE 和 DeepSeek-V2-Lite 上，Q-STRATA 在低比特区域（1.75–2.25 bit）均取得了最低的 WikiText2 困惑度。

## 研究问题与动机
- MoE 模型（如 Qwen3-30B-A3B）的 MoE 块包含 $3LE = 3 \times 48 \times 128 = 18{,}432$ 个专家线性层，搜索空间 $|\mathcal{Q}|^{3LE}$ 远超稠密模型（448 层），传统黑盒搜索不可行。
- 现有 MoE MPQ 方法（MxMoE、MC-MoE）在每个块内分配但保持块间统一预算，GEMQ 通过可加代理跨块分配，均无法直接优化耦合各块的模型级目标。
- 内层分配可用低成本代理可靠排序，而跨块分配需通过组装模型的端到端前向传播才能体现耦合，因此需要分层优化策略。
- 实践部署需要在不增加推理延迟的前提下大幅压缩 MoE 模型存储（专家权重压缩 7.1×–9.1×）。

## 核心贡献（创新点）
- 提出双层分配器 Q-STRATA：内层用块级代理缓存帕累托前沿候选，外层以每块一个预算为决策变量直接优化模型级 JSD，将搜索维度从 $|\mathcal{Q}|^{3LE}$ 降至 $|\mathcal{B}|^L$。
- 引入带延迟评估（lazy evaluation）的贪心下降求解外层 ILP 困难问题，每步仅需重算少量边际值，实际每步平均评估次数 R≈2.5，较急切下降加速 8–12×。
- 将外层搜索复用为稠密模型独立分配器：在 Llama-2-7B 上用 HQQ 量化器以 1,378 次查询匹敌 AMQ（10,474 次）并在 8K 校准数据下以 6.9× 更少查询超越。
- 在三个主流 MoE LLM 及更大的 Qwen3-30B-A3B（18,432 个专家层）上系统验证，1.75 bit 下对 MxMoE 的 WikiText2 困惑度降低超过一半（Mixtral：12.14 vs 25.20），MMLU 较均匀 GPTQ 提升 15.7–17.9 点。
- 提供与 GEMQ 在相同路由微调（RFT）与旋转设置下的公平对比，Q-STRATA 在所有 18 组（模型×比特）条件下均优于 GEMQ。

## 方法详解
**问题形式化**：给定候选量化器集合 $\mathcal{Q}$，为每个专家线性层 $(l,e,i)$ 选择 $x_{l,e,i}\in\mathcal{Q}$，最小化模型级目标 $\mathcal{L}(\mathbf{x})$（JSD），满足平均比特 $\bar{b}(\mathbf{x})\le\tau$。

**内层阶段（Block-level candidate caching）**：
- 使用块输出重建误差代理 $\mathcal{D}_l(\mathbf{x}_l)=\sum_{e,i}\|\hat{h}_l^{(e,i)}(x_{l,e,i})-h_l\|^2$，每项由单次块前向传播预计算。
- 对共享预算网格 $\mathcal{B}=\{\beta^{(1)},\dots,\beta^{(K)}\}$（步长 0.125，K=25），将每个 $\beta$ 下的最优分配表述为多选择背包问题（MCKP），用 Gurobi ILP 精确求解，缓存帕累托前沿 $S_l=\{\mathbf{x}_l^\star(\beta):\beta\in\mathcal{B}\}$。

**外层阶段（Budget-level assembly）**：
- 决策变量简化为每块预算 $\beta_l\in\mathcal{B}$，通过组装 $\mathbf{x}(\boldsymbol{\beta})=(\mathbf{x}_1^\star(\beta_1),\dots,\mathbf{x}_L^\star(\beta_L))$ 后评估 $\mathcal{L}$。
- 从最高预算角点 $(\beta^{(K)},\dots,\beta^{(K)})$ 出发，每步将损失增量 $\delta_l=\mathcal{L}(\mathbf{x}(\beta^{-l}))- \mathcal{L}(\mathbf{x}(\boldsymbol{\beta}))$ 最小的块降一级，直到 $\frac{1}{L}\sum_l\beta_l\le\tau$。
- 延迟贪心：用小根堆缓存每个 $\delta_l$ 及其计算步数 $t$，弹出时若陈旧则重新计算；经验表明 $\delta_l$ 随压缩增强近单调增长，满足低估条件。
- 单趟下降可输出所有目标预算下的分配。

## 实验与结果
- **数据集/模型**：Mixtral-8×7B-Instruct（46.7B-A12.9B）、Qwen1.5-MoE-A2.7B（14.3B-A2.7B）、DeepSeek-V2-Lite（15.7B-A2.4B），扩展至 Qwen3-30B-A3B；校准来自 WikiText2 train，序列长 4096。
- **基线**：均匀 GPTQ、MxMoE、GEMQ（shared，去除 RFT 与旋转）；量化器集为组大小 128 的非对称 GPTQ（1/2/3/4 bit）。
- **主要结果**（WikiText2，越低越好）：
  - 2.25 bit：Mixtral 5.62 / Qwen1.5-MoE 7.98 / DeepSeek-V2-Lite 6.74，均为最低；MMLU/GSM8K 在 2.0 bit 下较均匀 GPTQ 分别提升 15.7–17.9 和 3.6–8.0 点。
  - 2.00 bit：Mixtral 6.90 / Qwen1.5-MoE 8.97 / DeepSeek-V2-Lite 7.41。
  - 1.75 bit：Mixtral 12.14（MxMoE 25.20）、Qwen1.5-MoE 11.84、DeepSeek-V2-Lite 9.35。
- **外层消融**（Mixtral，Table 2）：Lazy greedy 在 1.75 bit 达 JSD 749.4，One-shot ILP 为 823.4，Uniform 为 1036.5；Gap 随预算收紧扩大。
- **Pruning-free 对比**（DeepSeek-V2-Lite，Table 3）：Direct cell-ILP 需 15.4k 次端到端评估（>8× Q-STRATA），JSD 仅略优（1.75 bit: 364.0 vs 410.6），说明内层剪枝几乎无损。
- **成本**：混洗搜索单次离线成本 23–89 GPU-hours（H100），部署后无额外推理延迟；模型体积压缩 5.2–6.3×，Decode 加速约 1.46×。

## 相关工作脉络
- **MxMoE**：块内同 Q-STRATA 的线性层粒度代理，但所有块共享统一预算 $\tau$，缺少外层跨块优化；是 Q-STRATA 的内层-only 对应。
- **GEMQ**：基于梯度加权的可加代理在专家粒度做全局 ILP；忽略块间耦合，本工作在其相同 RFT/旋转协议下仍全面超越。
- **MC-MoE**：在线性规划中手工组合重建误差与路由统计分配每专家位宽，同样保持块间统一预算。
- **AMQ / Q-palette / ScaleBits**：稠密模型的 MPQ 搜索；ScaleBits 亦利用递减回报性质，但其搜索空间更粗粒度（tile 级）且按敏感度排序而非实测边际。
- **HAWQ/HAWQv2/Higgs**：稠密模型的可加代理 MPQ；本工作的内层 proxy 思想与其同源，但 MoE 场景下需双层分解。
- **TWLA / AQ / HAQ**：激活/权重联合或 NAS+量化搜索；本工作专注权重-only PTQ，与 TWLA 的激活分配构成互补轴。

## 局限性与未来方向
- 外层每次评估为一次端到端前向传播，即使懒评估仍比一次拟合代理后闭式求解开销大；适合一次性离线搜索，但不适合在线迭代。
- 外层贪心下降依赖近单调边际假设（DR-submodularity 的经验近似），存在极少数反转，理论上不保证全局最优。
- 当前方法未联合优化路由门控权重与量化分配（除 GEMQ 风格的 RFT 外）；异构比特宽专用 CUDA kernel 亦未覆盖。
- 1 bit 极端量化下旋转效果模型依赖性强（Mixtral 恶化、Qwen1.5-MoE 改善），全局最优旋转策略待探索。

## 研究启发与可借鉴点
- **双层分解范式可迁移**：当优化空间可分为"块内细粒度+块间粗粒度"两层，且层间目标性质不同（代理 vs 端到端）时，可沿用"代理缓存+全局目标优化"的架构。
- **懒贪心 + staleness check 的工程技巧**：适用于任何边际可缓存、搜索路径单调的离散组合优化；建议在递减排列/子模最大化等任务中复用。
- **校准集代表性验证**：通过 per-(layer,expert) 路由分布的 TV 距离（calib vs test 约 3–5× 小于 calib vs uniform）建立置信，可推广到其他 MoE 压缩工作。
- **统一对比协议的价值**：与 GEMQ 共享 quantizer set、旋转、RFT 后再比较，结论更可信；建议团队后续消融也采用同等协议。
- **外层搜索作为稠密模型通用分配器**：只需将决策粒度从 block 换为 layer，并用 $\Delta\mathcal{L}_i/(w_i\Delta)$ 作为 budgeted greedy 键，即可无缝迁移。

## 关键术语表
**Mixed-Precision Quantization (MPQ)**：为模型不同线性层分配不同比特宽度的后训练量化策略，在固定总比特预算下最小化质量损失。
**Jensen-Shannon Divergence (JSD)**：本文用作模型级目标 $\mathcal{L}$，衡量量化模型与全精度模型在下一词元分布上的差异。
**Block reconstruction proxy $\mathcal{D}_l$**：将块输出误差近似为各线性层孤立量化的失真之和，用于内层快排与 ILP。
**Inner stage**：对每个 MoE 块在预算网格 $\mathcal{B}$ 上以 ILP 求解 MCKP，缓存帕累托候选集合 $S_l$。
**Outer stage**：从每个块的缓存集合中选一个预算 $\beta_l$，以懒贪心下降直接最小化端到端 JSD。
**Lazy evaluation (Minoux)**：用小根堆缓存边际评估，仅在陈旧时重算，将急切下降的 $O(L^2K)$ 降至 $O(RLK)$，R≈2.5。
**DR-submodularity**：递减回报性质，保证懒贪心的低估条件；本文经验验证其近成立。
**Router Fine-Tuning (RFT)**：在校准数据上微调路由门控（1 epoch、lr=1e-4），用于与 GEMQ 公平对比。

## 可复现要素
- **数据集**：WikiText2（train 用于校准，test 用于评估）；零样本评测 PIQA、BoolQ、WinoGrande、ARC-easy/challenge、HellaSwag。
- **模型**：Mixtral-8×7B-Instruct、Qwen1.5-MoE-A2.7B、DeepSeek-V2-Lite、Qwen3-30B-A3B。
- **代码**：已开源 https://github.com/snu-mllab/Q-Strata/tree/main。
- **量化器**：GPTQ group-128 非对称，1/2/3/4 bit（旋转设置沿用 MxMoE 协议）；稠密实验另用 HQQ 2/3/4 bit。
- **预算网格**：1.25–4.25 bit，步长 0.125，K=25。
- **校准数据量**：内层 64 条序列、外层 32 条序列；最终 GPTQ 量化用 128 条序列。
- **JSD 计算**：取 top-1000 token logits 以降低显存。
- **评测工具**：lm-eval-harness v0.4.5，context length 4096（MoE）/2048（稠密）；5-shot MMLU/GSM8K。
- **硬件与成本**：H100；搜索成本 23–88.7 GPU-hours（表 14）。
- **关键超参**：学习率/epoch 仅 RFT 提及（lr=1e-4，1 epoch）；组大小 128；目标比特 2.25/2.00/1.75。
