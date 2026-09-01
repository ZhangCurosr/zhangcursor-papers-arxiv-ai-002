---
title: "VisCache-Visual-KV-Cache-Pruning-for-Efficient-Vision-Large"
source: https://arxiv.org/pdf/2608.24063v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 11:17:13"
field: "视觉语言模型高效推理"
keywords: ["KV Cache Compression", "Vision-Language Model", "Video Understanding", "Inference Acceleration", "Token Pruning", "Efficient VLM"]
innovations: ["双阶段粗到细视觉KV缓存剪枝框架", "抛物线层预算分配与非对称键值融合策略", "提示感知关键帧选择与层感知压缩协同"]
benchmarks: ["ActivityNet Captions", "DREAM1K", "NExTQA", "ActivityNet-QA", "EgoSchema", "MVBench"]
---

# 论文速读：VisCache: Visual KV Cache Pruning for Efficient Vision Large Language Model Inference

## 一句话总结
VisCache 提出了一种无需训练、即插即用的双阶段视觉 KV 缓存剪枝框架，通过提示感知关键帧筛选与抛物线层预算分配结合的非对称键值压缩，在仅保留 19–28% KV 缓存的情况下实现最高 2.35× 推理加速，同时在长视频理解任务上保持或超越完整缓存性能。

## 研究问题与动机
- **视觉 KV 缓存开销巨大**：长视频产生海量视觉 token，导致 KV cache 占据大量 GPU 显存并加剧显存带宽竞争，同时 attention 计算随序列长度二次增长，严重制约 VLLM 的长上下文推理效率。
- **现有压缩方法存在信息损失**：量化易受激活值异常值影响，低秩分解会损害 attention 容量；更根本的问题是这些粗粒度压缩策略忽视了模型内部信息流的结构性冗余特征。
- **视觉冗余具有高度结构化特征**：仅有部分视频帧与查询相关，不同 transformer 层对下游推理的贡献不均，且 keys 和 values 在 attention 计算中扮演本质不同的角色，需要联合考虑时序相关性、层间重要性和键值非对称性。

## 核心贡献（创新点）
1. **提出双阶段粗到细的 VisCache 框架**：在输入层通过轻量级 scout VLM 进行提示感知时序过滤，在模型层通过 PruneKV 进行层感知 KV 压缩，两者协同工作，与仅做单层压缩的方法形成本质区别。
2. **设计 PruneKV 抛物线层预算分配机制**：依据 Equation (3) 在早期层保留更多 token（编码细粒度空间细节），在深层逐步减少预算，相比 PyramidKV 的等差数列和 PDrop 的几何衰减，在激进压缩下更稳定。
3. **引入非对称键值更新策略**：针对 dropped keys 丢弃而对应 values 通过相似度加权聚合融合到 retained tokens 中，利用 attention 中 keys 作为相关性选择器、values 作为信息载体的功能非对称性，避免上下文信息不可逆丢失。
4. **实现跨架构即插即用与零训练部署**：Scout 模型可兼容 CLIP/BLIP/OpenCLIP 等不同 VLM，PruneKV 与 KV 缓存量化方法（KIVI、FlatQuant）正交可组合，在 Qwen2.5-VL 3B/32B、Qwen3-VL、LLaVA-OneVision 等多架构上验证有效性。

## 方法详解
**Stage 1: Prompt-Aware Scout for Temporal Redundancy Filtering**
- 使用轻量级 VLM（如 CLIP ViT-B/32）作为 scout，分别通过文本编码器和视觉编码器将 prompt $T$ 和帧 $f$ 映射到共享嵌入空间，得到 $\mathbf{h}_t$ 和 $\mathbf{h}_f$。
- 采用最大边际相关性（MMR）准则平衡 prompt 相关性与帧间多样性：
$$\lambda \cdot \text{sim}(\mathbf{h}_f, \mathbf{h}_t) - (1 - \lambda) \cdot \max_{f' \in \Omega} \text{sim}(\mathbf{h}_f, \mathbf{h}_{f'})$$
- 迭代选择得分最高的帧加入集合 $\Omega$，直至达到目标保留率 $p$，形成紧凑且多样化的关键帧序列。

**Stage 2: PruneKV Layer-Aware Visual KV Cache Compression**
- **Token 重要性评分**：聚合所有层对所有视觉 token 的 attention 权重作为重要性分数：
$$s_v = \frac{1}{L} \sum_{l=1}^{L} \sum_{i=1}^{N} A_{i,v}^l$$
- **抛物线层预算分配**：设定截断阈值 $h$，对保留层 $l \in [1, h]$ 分配预算比例：
$$b_l = 1 - \frac{(l-1)^2}{2(h-1)^2}$$
满足 $b_1 = 1$、$b_h = 0.5$，在浅层缓慢衰减、深层快速衰减，归一化后满足 $\sum_{l=1}^{h} b_l = h \cdot m$。
- **非对称键值更新**：将 token 划分为 keep set $\mathcal{C}_k$ 和 drop set $\mathcal{C}_d$，keys 直接丢弃，values 通过相似度过重分布矩阵融合：
$$\Phi = \text{Softmax}\left(\frac{\mathbf{V}_k \mathbf{V}_d^\top}{\tau}\right) \in \mathbb{R}^{b_l \times (n-b_l)}$$
$$\mathbf{V}_k^{\text{new}} = \mu \mathbf{V}_k + (1-\mu)(\Phi \mathbf{V}_d)$$
其中 $\mu$ 平衡原始信息与重分布信息，$\tau$ 控制分布锐度。

**整体 RR 计算**：$\text{RR} = p \times q \times \frac{h}{L} \times m$，例如 $p=0.75, q=0.67, h=\frac{3}{4}L, m=0.75$ 时 RR ≈ 28%。

## 实验与结果
**数据集与基线**：在 ActivityNet Captions（ActCap）、DREAM1K（视频摘要）、NExTQA、ActivityNet-QA（ActQA）、EgoSchema（VQA）、MVBench（20项多任务）上评估；基线包括 PyramidKV、FastV、PDrop、Q-Frame。

**主要结果（Qwen2.5-VL-3B）**：
- 40% RR 时 FLOPs 降至完整缓存的 9%，优于 PyramidKV（13%）；28% RR 时平均 VQA 准确率 45.64%，**超越完整缓存（44.16%）**。
- 19% RR 时 FLOPs 仅 6%，平均准确率 44.85%，仍高于多数基线在 40% RR 的表现。

**主要结果（Qwen2.5-VL-32B）**：
- 28% RR 时 FLOPs 降至 12%，平均准确率 56.86%，优于所有基线在 40% RR 的结果。
- 19% RR 时 FLOPs 仅 10%，EgoSchema 准确率 65.00%，显著优于 Q-Frame（56.00）和 PDrop（57.00）。

**MVBench 多任务评估**：VisCache 在 3B 和 32B 模型上均取得最高平均准确率（52.6% 和 55.4%），在时间推理（AS、UA）和细粒度感知（AA）任务上优势明显。

**系统效率**：在 ActCap 上 19% RR 时实现 2.35× E2E 加速，TPOT 从 118.81ms 降至 45.60ms；GPU 总显存降至 3.68 GB，KV cache 仅 0.02 GB。

**消融验证**：抛物线分配优于固定/等差/几何分配；值融合机制在所有预算策略下均提升性能；共享全局 token 排名优于层独立排名。

## 相关工作脉络
- **PyramidKV**（Cai et al., 2024）：按等差数列逐层缩减 KV 预算形成金字塔结构，本文抛物线分配在浅层保留更多 token，激进压缩下更稳定。
- **PDrop**（Xing et al., 2024）：将层划分为阶段并按几何衰减分配预算，本文指出过度激进压缩会损害细粒度视觉理解。
- **FastV**（Chen et al., 2024a）：基于视觉 attention 稀疏性选择性保留 KV 条目，本文在此基础上增加层间预算自适应和键值非对称处理。
- **Q-Frame**（Zhang et al., 2025a）：查询感知帧选择与多分辨率适配，本文采用 MMR 准则同时考虑相关性与多样性。
- **VisionZip / SparseVLM**：基于 attention 或跨模态相关性进行 token 剪枝或合并，本文强调忽略 key/value 功能非对称性是现有方法的共同局限。

## 局限性与未来方向
- **Scout 模型与主模型表征对齐问题**：轻量级 VLM 的视觉编码器可能与主 LLM backbone 存在表征差异，导致筛选的关键帧未必是下游模型最优选择；未来可探索 scout-backbone 协同适配。
- **Prefilling 阶段需存储全层 attention 分数**：当前 PruneKV 实现需在前向传播期间保存所有层的 attention 权重，引入额外显存开销；未来需探索更省内存的分数计算方法。
- **Scout 模型选择存在性能波动**：实验显示 CLIP/OpenCLIP 优于 BLIP，语义对齐能力影响筛选质量，需进一步优化 scout 模型选择策略。

## 研究启发与可借鉴点
- **键值非对称性利用**：attention 中 keys 决定选择、values 承载信息的本质差异可推广至其他 KV 压缩场景，如 LLM 长上下文推理中的 cross-attention 压缩。
- **抛物线预算分配优于线性/几何**：基于"浅层保留细节、深层抽象表示"的认知科学洞察，可迁移至文本 LLM 的层间 token 保留策略设计。
- **双阶段粗到细范式**：输入级时序过滤 + 模型级结构化压缩的组合思路，适用于任何涉及多模态长序列的高效推理场景。
- **即插即用与正交可组合性**：VisCache 与量化方法（KIVI、FlatQuant）正交，可进一步叠加压缩，为混合压缩策略提供设计参考。

## 关键术语表
**KV Cache**：自回归生成过程中缓存的 Key 和 Value 矩阵，用于避免重复计算，视觉 token 会显著膨胀其体积。
**Retention Ratio (RR)**：保留的视觉 KV 缓存占完整缓存的比例，VisCache 整体 RR = $p \times q \times \frac{h}{L} \times m$。
**Maximal Marginal Relevance (MMR)**：信息检索中平衡相关性与多样性的重排序准则，本文用于关键帧选择。
**Parabolic Budget Allocation**：按抛物线规律分配层间 KV 预算，浅层高保留、深层低保留，公式为 $b_l = 1 - \frac{(l-1)^2}{2(h-1)^2}$。
**Asymmetric Key-Value Update**：对 dropped tokens 丢弃 keys 但通过加权聚合将 values 融合至 retained tokens，保留上下文信息。
**Scout VLM**：轻量级视觉语言模型（如 CLIP），用于前置时序过滤，与主模型解耦。

## 可复现要素
- **数据集**：ActivityNet Captions、DREAM1K、NExTQA、ActivityNet-QA、EgoSchema、MVBench（公开基准）
- **代码开源**：是，https://github.com/Wlklk/VisCache
- **关键超参**：$\lambda = 0.7$（MMR 相关-多样权衡）、$h = \frac{3}{4}L$（层截断阈值）、$m = 0.75$（平均层预算）、$\tau = 1.0$（温度）、$\mu = 0.7$（融合权重）
- **基座模型**：Qwen2.5-VL-3B/32B-Instruct、Qwen3-VL-4B-Instruct、LLaVA-OneVision
- **硬件环境**：4 × NVIDIA A100 GPU（80GB）
