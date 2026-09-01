---
title: "Q-Strata-Hierarchical-Bit-Allocation-for-Mixed-Precision-Qua"
source: https://arxiv.org/pdf/2608.30564v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 23:43:56"
field: "大模型高效推理/量化"
keywords: ["Mixed-Precision Quantization", "Mixture-of-Experts", "LLM Compression", "Bit Allocation", "Lazy Greedy"]
innovations: ["双层分配器：内层代理排序候选 Pareto frontier，外层模型级目标优化跨 block 预算", "懒贪心下降将 O(L²K) 评估降至 O(RLK)，R≈2.5", "通用外层搜索可用于稠密模型，query 数远少于 AMQ"]
benchmarks: ["WikiText2 perplexity", "6-task zero-shot accuracy", "MMLU", "GSM8K"]
---

# 论文速读：Q-Strata-Hierarchical-Bit-Allocation-for-Mixed-Precision-Qua

## 一句话总结
Q-STRATA 提出了一种双层混合精度量化分配器，针对 MoE LLM 的专家层位宽分配问题，通过内层基于低成本代理排序候选分配、外层基于模型级目标优化块间预算分配，在低比特 regime 下显著降低了 Wiki-Text2 perplexity。

## 研究问题与动机
- **MoE 模型量化空间爆炸**：每个 MoE block 含 E 个 expert，每个 expert 有 up/gate/down 三层 linear layer，总搜索空间为 $|\mathcal{Q}|^{3LE}$，远大于稠密模型（如 Qwen3-30B-A3B 的 MoE 层有 18,432 个 linear layer）。
- **现有方法局限**：现有 MPQ 方法要么使用 additive proxy（GEMQ），忽略 block 间的耦合效应；要么仅在同一 block 内分配但所有 block 预算统一（MxMoE、MC-MoE），无法跨 block 优化。
- **直接黑盒搜索不可行**：高维离散空间下，基于模型评估的进化搜索/强化学习等方法计算开销过大。
- **双层结构自然存在**：block 内部的位宽分配可用低成本 proxy 可靠排序，而 block 之间的分配耦合只能通过模型级目标 $\mathcal{L}$ 体现，这为分层优化提供了天然依据。

## 核心贡献（创新点）
- **双层分配器设计**：将原 $3LE$ 维位宽分配问题降为 $L$ 个 per-block 预算选择，内层用代理排序候选、外层用模型级目标优化跨 block 分配。
- **内层 Pareto frontier 缓存**：对每个 block 在精细预算网格上求解 MCKP（用 ILP），缓存一条 cost-distortion Pareto frontier（每个预算水平一个候选）。
- **懒贪心外循环**：从最贵预算角出发，每次降低使模型级目标 $\mathcal{L}$ 上升最小的 block 预算，利用 lazy evaluation（min-heap + 过期检测）减少 $O(L^2K)$ 到 $O(RLK)$ 次前向传播评估。
- **通用性验证**：外层搜索不依赖 MoE 结构，可直接用于稠密模型（Llama-2-7B），以极少 query 数匹配 AMQ 效果。
- **实证 DR-submodularity**：实验验证了 $\mathcal{L}$ 的边际损失增长近似单调，支持 lazy 加速的正确性。

## 方法详解
**问题形式化**：最小化模型级目标 $\mathcal{L}(\mathbf{x})$（JSD between quantized vs full-precision next-token distributions），约束为平均位宽 $\bar{b}(\mathbf{x}) \le \tau$。

**内层（Inner Stage）**：
- 对每个 block $l$ 和预算 $\beta \in \mathcal{B}$，求解：
$$\mathbf{x}_l^\star(\beta) = \arg\min_{\mathbf{x}_l \in \mathcal{X}_l} \mathcal{D}_l(\mathbf{x}_l) \quad \text{s.t. } \bar{b}_l(\mathbf{x}_l) \le \beta$$
- $\mathcal{D}_l$ 是 block 输出重建误差代理（各 linear layer 独立失真之和），通过一次 block 前向预计算。
- 用 ILP（Gurobi）精确求解 MCKP，生成候选缓存 $S_l = \{\mathbf{x}_l^\star(\beta): \beta \in \mathcal{B}\}$。

**外层（Outer Stage）**：
- 优化：$\beta^\star = \arg\min_{\beta \in \mathcal{B}^L} \mathcal{L}(\mathbf{x}(\beta))$ s.t. $\frac{1}{L}\sum_l \beta_l \le \tau$。
- 懒贪心下降：初始全用最高预算，每次 pop 最小 $\delta_l$ 的 block，降低其预算一级，用 min-heap 存储最近 marginal，过期则 REFRESH 重算。
- 单次 sweep 覆盖所有目标预算，R ≈ 2.5 次 evals/move，相比 eager 加速约 10×。

## 实验与结果
- **数据集/模型**：Mixtral-8×7B-Instruct (46.7B-A12.9B)、Qwen1.5-MoE-A2.7B (14.3B)、DeepSeek-V2-Lite (15.7B-A2.4B)、Qwen3-30B-A3B。
- **评估**：WikiText2 perplexity（context 4096）、6-task zero-shot average accuracy。
- **基线**：Uniform GPTQ、MxMoE、GEMQ（含 shared/RFT variants）。
- **主要结果（2.25 bits）**：
  - Mixtral: Q-STRATA 5.62 vs MxMoE 5.69 vs GEMQ(shared) 9.39
  - Qwen1.5-MoE: 7.98 vs 8.07 vs 10.76
  - DeepSeek-V2-Lite: 6.74 vs 6.78 vs 7.61
- **低比特突破**：Mixtral 1.75 bits 时 Q-STRATA 12.14 对 MxMoE 25.20（halves perplexity），acc +7.9 pts。
- **稠密模型验证**：Llama-2-7B 2.5-3.5 bits 下 lazy greedy 匹配 AMQ（JSD 0.1569 vs 0.1572），但 query 少 6.9-51×。
- **MMLU/GSM8K（2.0 bits）**：Q-STRATA 比 uniform GPTQ 提升 15.7-17.9 MMLU pts，GSM8K 从 ~0 提升到 3.7-8.1。

## 相关工作脉络
- **MPQ 作为 MCKP**：HAWQ/HAWQv2 (Dong et al., 2019, 2020)、Q-Palette (Lee & Song, 2025) 等用 Hessian/梯度代理求和 + ILP/Dynamic Programming。
- **黑盒 MPQ**：AMQ (Lee et al., 2025) 用 AutoML-style 搜索 + 模型级目标，但搜索空间大。
- **MoE 量化**：MC-MoE (Huang et al., 2025) 用 hand-crafted score + LP；MxMoE (Duanmu et al., 2025) 用 block-level 代理但不跨 block 分配；GEMQ (Deng et al., 2026) 全局分配但依赖 additive gradient proxy。
- **ScaleBits (Li et al., 2026)**：concurrent work，tile-level 量化 + diminishing returns + 贪心，但搜索空间更细，用 sensitivity 估计而非实测 marginals。
- **Activation 量化**：TWLA (Zhao et al., 2026) 的 ILA-AMP 做 inter-layer activation bit 分配，与本方法的 weight-only 正交。

## 局限性与未来方向
- **搜索开销仍重于 proxy-only 方法**：每次评估需一次 assembled model 前向传播，虽 lazy 加速但总成本仍高于一次拟合代理后闭式求解的方法。
- **DR-submodularity 非严格成立**：实验观察到少量反转，lazy 加速为启发式，理论上不能保证等价 eager 搜索。
- **1-bit 极端场景下 rotation 效果模型相关**：对 Mixtral 1-bit 有害，需根据模型选择是否 rotation。
- **部署时 kernel 优化未覆盖**：当前 serving 用统一 bit dequantization kernel，异构 bitwidth 专用 kernel 可进一步提速。
- **未来方向**：降低外层搜索成本、结合 kernel-aware 的 mixed-precision search formulation。

## 研究启发与可借鉴点
- **双层分解思想**：将高维组合优化拆为"局部精细排序 + 全局粗粒度优化"，适用于大规模分层模型架构（如 MoE、nested transformers）。
- **Lazy 加速在黑色目标搜索中的适用性**：验证 DR-submodularity 的近似成立后，可大幅减少黑盒目标评估次数。
- **代理 fidelity 校验协议**：通过 random allocation + Spearman correlation（JSD 0.98-0.99 vs GEMQ proxy 0.28-0.42）定量比较代理质量。
- **校准数据代表性分析**：用 TV distance 检查 calibration routing distribution 与 test 分布的偏差，保障分配决策稳定。
- **统一量化器集公平对比**：所有 baseline 共享同一 GPTQ 量化器集（1/2/3/4-bit, group 128），隔离"allocation"与"quantizer"效果。

## 关键术语表
- **Mixed-Precision Quantization (MPQ)**：为不同线性层分配不同位宽，在固定预算下最小化质量损失。
- **Mixture-of-Experts (MoE)**：将 FFN 拆分为多个 expert，每 token 只激活少数 expert 的大模型架构。
- **Block reconstruction proxy ($\mathcal{D}_l$)**：单次 block 前向预计算的 linear-layer 独立失真之和，近似 block 级量化影响。
- **Jensen-Shannon Divergence (JSD)**：量化模型与满精度模型 next-token 分布间的散度，作为模型级目标 $\mathcal{L}$。
- **Multiple-Choice Knapsack Problem (MCKP)**：每层选一个量化器（item），约束总 bit 成本，最小化失真之和。
- **Lazy greedy descent**：用 min-heap 缓存边际增益，仅在 stale 时重算，减少黑盒目标评估次数。
- **DR-submodularity**：递增函数 $G$ 满足 $G(\beta^{+l}) - G(\beta) \ge G((\beta')^{+l}) - G(\beta')$，保证 lazy 加速精确性。
- **Router Fine-Tuning (RFT)**：微调 routing gate 参数以适配量化后的 expert 分布。

## 可复现要素
- **数据集**：WikiText2（train 用于 calibration，test 用于 evaluation）；6-task zero-shot（PIQA, BoolQ, WinoGrande, ARC-easy/challenge, HellaSwag）。
- **代码**：已开源 https://github.com/snu-mllab/Q-Strata/tree/main
- **权重**：未提及开放量化权重，代码可复现。
- **关键超参**：
  - 量化器集：GPTQ group-128 asymmetric 1/2/3/4-bit
  - 预算网格 $\mathcal{B}$：1.25 到 4.25 bits，步长 0.125，K=25
  - Calibration 序列数：inner stage 64、outer stage 32、final GPTQ 128（seq len 4096）
  - JSD 计算：top-1000 token logits
- **实验环境**：H100 GPU，GemLite kernels for serving benchmark
