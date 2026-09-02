---
title: "Latent-Recurrent-Thoughts-Recurrent-Refinement-of-Proposed-L"
source: https://arxiv.org/pdf/2609.01117v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 12:09:51"
field: "大语言模型推理增强"
keywords: ["latent reasoning", "frozen LLM", "chain-of-thought", "recurrent refinement", "parameter-efficient", "continuous space reasoning", "TRM", "soft token"]
innovations: ["用任务专用 proposer + TRM 递归 refiner 替代通用助手+能量 refiner，在冻结解码器前实现多步隐空间迭代精炼", "首次将 TRM 等递归推理器与预训练 LLM 配对，计算深度与参数规模解耦", "提出 bounded residual correction 参数化与 truncated-gradient unrolling 联合训练策略，以 11.2M 参数在 5 基准上达 54.1% 平均准确率"]
benchmarks: ["Countdown-4", "Sudoku", "HumanEval", "MBPP", "StrategyQA"]
---

# 论文速读：Latent Recurrent Thoughts: Recurrent Refinement of Proposed Latents for Reasoning with Frozen LLMs

## 一句话总结
本文提出 **LRT（Latent Recurrent Thoughts）**，将任务专用 proposer 与递归迭代 refiner（TRM）结合，在冻结 LLM 的连续隐空间中完成推理——proposer 生成初始隐向量，refiner 通过多级残差修正迭代精细化，冻结解码器最终解码出答案，以仅 11.2M 可训练参数在符号与自然语言推理上显著超越既有冻结解码器方法。

## 研究问题与动机
- **CoT 的离散空间局限**：链式思维在离散 token 空间进行，每步必须从固定词表中生成文本，错误会传播，且诱发高质量 CoT 的前提是已有可模仿的推理轨迹（trace supervision），这对无 trace 的符号任务不可行。
- **连续隐空间推理的有效性未确立**：将中间状态以向量形式注入模型隐空间可规避上述约束，但现有冻结解码器方法（SoftCoT、EBM-CoT）中，通用助手作为 proposer 在远离自然语言分布的符号任务上会主动损害解码器表现（Countdown-4 仅 5.9%/8.4%，低于无推理的 Direct 基线 27.8%）。
- **能量模型 refiner 计算深度不足**：EBM-CoT 的能量 refiner 仅沿固定标量场的梯度校准隐向量，无法实现多步搜索式计算，对复杂约束满足问题过浅。
- **递归推理器尚未与预训练 LLM 组合**：TRM/GRAM/HRM 等递归推理器虽参数高效且计算深度可独立于参数量扩展，但此前均作为独立求解器从头训练，缺乏语言建模先验；LRT 首次将其复用于冻结预训练 LLM 的隐空间迭代精炼。

## 核心贡献（创新点）
1. **提出 LRT 框架**：将任务专用 bidirectional Transformer encoder 作为 proposer 与 TRM 递归 refiner 配对，置于冻结 LLM 解码器之前，实现连续隐空间中的多步推理。
2. **任务专用 proposer 替代通用助手**：proposer 直接以共享词表 E 嵌入输入 x，经降维映射与可学习 query vectors 生成 K 个 base latents，解决了通用助手在短输入/非分布内样本上生成误导向量（off-manifold）的问题。
3. **递归迭代 refiner 以有界残差形式精炼隐向量**：利用 TRM 的双时间尺度状态（快速 scratch state $z_L$、慢速整合状态 $z_H$），在每个 fast update 阶段重新注入外部信号 u，输出 $\Delta = P'_\uparrow(z_H)$ 作为有界残差修正，使 $L^\star = L^{(0)} + \Delta$，计算深度（45 次 transition pass）与参数规模完全解耦。
4. **受控对比下的系统性胜利**：在相同冻结 Qwen3-8B 解码器、提示、训练数据与预算下，LRT 在 5 个基准上平均 54.1%，大幅领先 SoftCoT（29.5%）与 EBM-CoT（33.5%）；在 Countdown-4 上达 56.7%，甚至超越从头训练的扩散求解器 MGDM（52.0%）。
5. **机制分析与可复现性**：通过因子分解 ablation（2×3 grid）、线性 probe、NLL 与余弦相似度分析，定量证实计算分布于 refiner 与 decoder 而非任一端，并开源全部代码与权重（MIT License）。

## 方法详解

**整体管线**（公式化表述）：
$$L^{(0)} = g_\psi(x), \quad L^\star = L^{(0)} + r_\phi(L^{(0)}), \quad \hat{y} = \mathcal{M}(I, x, L^\star)$$
其中 $g_\psi$ 为 proposer，$r_\phi$ 为 refiner，$I$ 为任务指令，$\mathcal{M}$ 为冻结的 Qwen3-8B 解码器（参数量 8.2B）。

**Proposer（§3.2）**：
- 双向 Transformer encoder，输入 x 经冻结解码器词表 $E \in \mathbb{R}^{|V|\times d}$（$d=4096$）嵌入后，通过可学习投影 $P_\downarrow: \mathbb{R}^d \to \mathbb{R}^{d'}$（$d'=256$）降维至工作维度（输出缩放 $\sqrt{d'}$）。
- 追加 $K=32$ 个可学习 query vectors，经两个 pre-norm Transformer 块（self-attention + SwiGLU）交互后，提取 query 位置输出，再经 $P_\uparrow: \mathbb{R}^{d'} \to \mathbb{R}^d$ 升维至解码器嵌入空间，得到 $L^{(0)} \in \mathbb{R}^{K \times d}$。
- 可训练参数 ≈ 4.2M。

**Recurrent Refiner（§3.3）**：
- 基于 TRM（Jolicoeur-Martineau, 2025）的递归网络，维护两个工作维度状态 $z_L, z_H \in \mathbb{R}^{K \times d'}$，初始化自可学习 buffer $z_L^0, z_H^0$。
- base latents 经独立投影 $P'_\downarrow$ 得到外部信号 $u = P'_\downarrow(L^{(0)}) \in \mathbb{R}^{K \times d'}$，每个 high-level cycle 执行 T=4 次 fast update 后 1 次 slow update：
$$z_L \leftarrow f(z_L, z_H + u) \quad (\times T), \qquad z_H \leftarrow f(z_H, z_L)$$
其中 $f$ 为 single transition block（与 proposer 同架构类、独立权重）。
- **关键设计**：u 在每个 fast update 重新注入（非仅在初始化），确保精炼锚定于原问题；最终输出有界残差 $\Delta = P'_\uparrow(z_H) \in \mathbb{R}^{K \times d}$， refined latents $L^\star = L^{(0)} + \Delta$。
- 损失函数：$\mathcal{L} = \mathcal{L}_{CE}(\mathcal{M}(I, x, L^\star), y) + \lambda \|\Delta\|^2$，$\lambda=0.01$ 防止漂移。
- Truncated-gradient unrolling（Algorithm 1）：$S=3$ 个 outer iteration、$H=3$ 个 high-level cycle，前 $S\cdot H - 1 = 8$ 个 cycle 用 stop-gradient 预热，仅最后一个 cycle 参与反向传播，总计算量等价于 45 次 transition block 前向。
- 可训练参数 ≈ 7.0M。

**Injection & Decoding（§3.4）**：
- 解码器接收拼接输入 $[I; x; L^\star]$，$L^\star$ 已输出在 $\mathbb{R}^d$ 空间无需再投影；指令 I 提供 instruction-following prior，$L^\star$ 提供 instance-specific 推理内容；$\theta$ 全程冻结。

**Two-Stage Training（§3.5）**：
- Stage 1：只训 proposer，$L^{(0)}$ 直接注入 $[I; x; L^{(0)}]$，端到端优化 CE。
- Stage 2：冻结 $g_\psi$，预计算并缓存 $L^{(0)}$，仅训 refiner 以 $\mathcal{L}_{CE} + 0.01\|\Delta\|^2$ 优化。
- 全程仅使用最终答案监督（StrategyQA 含参考 rationale），不使用任何推理 trace。

## 实验与结果

**数据集与度量**：
| 数据集 | 类型 | 度量 | 训练/评估规模 |
|---|---|---|---|
| Countdown-4（CD4） | 符号（算术组合） | exact solve rate | 100k / 1k（hold-out targets） |
| Sudoku | 符号（约束满足） | exact solve rate | 100k / 1k |
| HumanEval | 自然语言（Python 合成） | pass@1 | eval-only / 164 |
| MBPP | 自然语言（Python 合成） | pass@1 | 374 / 500 |
| StrategyQA | 自然语言（yes/no 问答） | accuracy | 2,061 / 229 |

**主要结果（Table 1，均值±标准差，3 seeds）**：

| 方法 | CD4 | Sudoku | HumanEval | MBPP | StrategyQA | Avg. |
|---|---|---|---|---|---|---|
| Direct (no CoT) | 27.8±1.6 | 17.3±0.3 | 13.4±1.1 | 28.7±1.8 | 67.4±6.1 | 30.9 |
| Zero-Shot CoT | 30.0±0.6 | 23.9±0.2 | 15.9±0.5 | 35.4±1.6 | 69.9±1.4 | 35.0 |
| SoftCoT | 5.9±0.2 | 10.4±0.1 | 20.7±0.9 | 40.2±2.1 | 70.2±6.5 | **29.5** |
| EBM-CoT | 8.4±0.1 | 17.2±0.4 | 25.0±1.6 | 46.1±3.6 | 71.0±5.7 | 33.5 |
| **LRT（Ours）** | **56.7±1.9** | **49.2±1.5** | 37.8±3.3 | **51.5±1.7** | 75.1±2.3 | **54.1** |

- **最强结果**：CD4 上 56.7%，较零样本 CoT 提升 **+26.7 个百分点**，超越从头训练的 MGDM（52.0%）；Sudoku 上 49.2%，较 EBM-CoT 提升 **+32.0 个百分点**；Avg. 54.1% 较 EBM-CoT 提升 **+20.6%**。
- **重要对比**：思考模式（thinking mode）Qwen3-8B 在 CD4 上达 85.3%，但每样本消耗 144.9 TFLOP，约为 LRT（≈1 TFLOP）的 **145 倍**；Sudoku 思考模式仅 0.5%，远低于 LRT 的 49.2%。

**关键 Ablation（Table 3, 5, 9, 10, 13, 14）**：
- 因子分解（Table 3）：generic proposer + 无 refiner = 12.3% Avg.；加 task-dedicated proposer 升至 37.2%；再加 recurrent refiner 升至 47.9%（三基准均值）。recurrent refiner 增益约为 energy refiner 的 **3 倍**（+10.7 vs +3.4）。
- Residual penalty λ=0.01 为最优；λ=0 时残差漂移严重（3.9× norm，-2.3 Avg. 点）。
- $K_{infer}=4$ 为最优推理 latent 数（训练 $K=32$ 保留为 scratch space）。
- Joint training 对比两阶段训练低 3.1  Avg. 点；冻结 proposer 几乎无代价（-0.5 点）。
- 移除指令 I 导致 -3.4 点。
- 推理时深度外推（9→12→15 cycles）持续微幅提升。
- 参数缩放（7M→14M→28M refiner）在 CD4 上单调上升（56.7→66.3→69.1），Headroom 显著。

## 相关工作脉络

1. **SoftCoT（Xu et al., 2025a）**：冻结通用助手 LM 生成 soft thoughts 注入解码器，无 refiner；LRT 的核心区别在于用 task-dedicated proposer + recurrent refiner 替代通用助手+无精炼。
2. **EBM-CoT（Chen et al., 2025b）**：在 SoftCoT 基础上加入能量-based refiner，沿标量场梯度校准隐向量；LRT 将其替换为可执行多步向量变换的 recurrent reasoner，计算深度与参数量解耦。
3. **Coconut / CODI / Token-Assorted（Hao et al., 2025; Shen et al., 2025; Su et al., 2025）**：需要 fine-tune 或全参数训练解码器；LRT 保持解码器完全冻结，仅训练 <0.2% 参数，避免 catastrophic forgetting。
4. **TRM / HRM / GRAM / EqR（Jolicoeur-Martineau, 2025; Wang et al., 2025; Baek et al., 2026; Huang et al., 2026）**：均为从头训练、单一任务的 standalone solver，无语言能力；LRT 首次将 TRM 复用为 refiner，与预训练 LLM 协同，实现符号推理+自然语言的统一框架。
5. **Prefix-tuning / P-tuning v2（Li & Liang, 2021; Liu et al., 2022）**：参数预算匹配（≈11M）下在 CD4 上仅达 12.5%/28.4%，远不及 LRT 的 56.7%，证明 instance-conditioned latent computation 是增益来源而非单纯参数开销。
6. **MGDM / 离散扩散求解器（Ye et al., 2025）**：从头训练符号专用模型在 Sudoku 上近乎完美（96.9%），但无法处理自然语言；LRT 以峰值精度换取多领域通用性。

## 局限性与未来方向

- **Per-task 训练协议**：每个任务族需训练独立的 proposer 和 refiner，非 zero-shot 通用方法；跨任务迁移能力未验证。
- **训练依赖答案监督**：需有答案标签的成对数据；对于无监督或 trace-free 且无答案的任务不适用（尽管 1k 样本即可接近完整性能）。
- **符号任务峰值精度不及专用求解器**：Sudoku 上 49.2% vs. EqR 的 99.8%，TRM 在 Countdown-4 上仅 1.2%（因 seq-to-seq 适配简陋），专用 solver 仍有显著差距。
- **推理计算随深度增长**：refiner 的递归 unroll 增加推理开销（虽远小于 CoT 生成开销）；full BPTT 比 truncated 内存高 6×、步时长 3× 但仅 ±0.1 点收益。
- **机械分析为相关性证据**：线性 probe（57.5% vs 52.5% baseline）、收敛轨迹等仅证明线性可分性边界，非计算分布的因果证明。
- **仅评测 5 个基准与 3 个 decoder 尺寸**：更大规模 decoder 或更广任务分布下的泛化性待验证。

## 研究启发与可借鉴点

1. **TRM/递归推理器的冻结 LLM 复用最**：将现成递归推理模块（TRM 等）作为 refiner 接入冻结 LLM，是一个参数高效的推理增强范式，可迁移至代码生成、数学推理等多种场景。
2. **Task-dedicated proposer 的设计原则**：共享输入词表 E 实现几何对齐、bidirectional encoder + query vectors 的 one-pass proposal 结构，可作为 generic-to-specific 转换的标准模板。
3. **Truncated-gradient unrolling 的工程价值**：仅对最后 cycle 反向传播，以前续 40/45 次 stop-gradient 预热 latent state，大幅降低训练内存（6× 缩减）和时间（3× 缩减）而几乎无损精度，值得在深度递归架构中推广。
4. **Train/inference asymmetry 的 latent 数量设计**：训练时用大量 query（K=32）作 scratch space，推理时只取前 K_inf=4，是一种"训练容量大、推理轻量"的参数高效策略。
5. **残差修正参数化（$L^\star = L^{(0)} + \Delta$）+ 权重衰减 $\lambda\|\Delta\|^2$**：保持精炼过程锚定于 proposal，避免 drift 到 instance-agnostic 区域，是隐空间 refine 的通用正则手段。

## 关键术语表

**Chain-of-Thought（CoT）**：让 LLM 逐步输出一系列中间推理 token 以改善复杂任务表现的 prompting 技术；本文指其离散 token 空间的局限。

**Latent Thought / Soft Token**：以连续向量而非离散 token 形式注入解码器输入序列的隐状态，作为模型内部的"软推理"载体。

**Proposer**：将问题实例 x 映射为 K 个 base latent 向量的小型编码器模块；本文中专指 task-dedicated bidirectional encoder。

**Recurrent Refiner（TRM）**：基于 TRM 递归网络的精炼器，通过双时间尺度状态迭代更新 latent，输出有界残差修正 base latents。

**Truncated-Gradient Unrolling**：深度递归展开训练中仅对最后若干步计算梯度、前序步骤使用 stop-gradient 的技术，实现深度推理与低训练成本的兼顾。

**Frozen Decoder**：全程不参与梯度更新的大规模预训练 LLM 解码器，仅提供序列建模与语言先验；本文特指 Qwen3-8B。

**Energy-Based Refiner（EBM-CoT）**：沿标量能量场梯度校准 latent 的 refiner，仅执行单步能量下降而非多步迭代计算。

**Bounded Residual Correction**：refiner 输出 $\Delta$ 而非完整 $L^\star$，通过 $L^\star = L^{(0)} + \Delta$ 保持精炼锚定于 proposal 的参数化形式。

## 可复现要素

- **数据集**：Countdown-4（合成，100k 训练/1k 评估，targets hold-out）、Sudoku（合成，100k/1k）、HumanEval（公开，eval-only）、MBPP（公开，374/500）、StrategyQA（公开，2,061/229）。
- **代码开源**：是，MIT License，仓库 https://github.com/czl-david/latent-recurrent-thoughts。
- **权重开源**：是（论文声明包含 trained module weights）。
- **关键超参**：
  - 工作维度 $d'=256$，训练 latent 数 $K=32$，推理 latent 数 $K_{infer}=4$
  - Refiner unroll：$S=3, H=3, T=4$（共 45 次 transition pass，仅最后 5 次反向传播）
  - Residual penalty $\lambda=0.01$
  - Optimizer：AdamW，weight decay=0.01，cosine LR，5% warmup，gradient clipping=1.0
  - Stage 1 peak LR=$3\times10^{-4}$，Stage 2 peak LR=$2\times10^{-4}$
  - Batch size=64，Epochs=30/stage，bfloat16，单卡 96GB GPU
  - 训练集规模敏感性：CD4 上 1k 样本即达 52.7%
