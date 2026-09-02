---
title: "SinkPruner-Sink-Free-Visual-Token-Pruning-for-Multimodal-Lar"
source: https://arxiv.org/pdf/2609.01004v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 22:17:27"
field: "多模态大模型高效推理"
keywords: ["visual token pruning", "multimodal large language models", "attention sink", "high-norm outlier", "inference acceleration", "training-free"]
innovations: ["发现高范数异常Token在特征和空间维度高度冗余，提出scale-free top-ρ规则识别", "提出cascading剪枝框架，通过视觉净化器预处理降低下游注意力汇聚和弥散", "视觉净化模块可作为即插即用组件增强现有剪枝方法性能"]
benchmarks: ["MMBench", "MME", "POPE", "ScienceQA", "VQA-v2", "MMStar", "MMMU", "AI2D", "MM-Vet", "MVBench", "SEED-Bench", "VideoMME", "NextQA"]
---

# 论文速读：SinkPruner: Sink-Free Visual Token Pruning for Multimodal Large Language Models

## 一句话总结
论文提出 SinkPruner，一种无需训练的视觉 Token 剪枝框架，通过识别并过滤视觉编码器中的高范数异常 Token（high-norm outlier tokens）来缓解注意力汇聚（attention sink）和注意力弥散问题，从而显著提升 MLLM 推理效率，在 LLaVA-1.5 上以 89% Token 削减保留 96.5% 性能，在 Qwen2.5-VL 上以 88.9% 削减保留 91.8% 性能。

## 研究问题与动机
- **计算瓶颈**：MLLM 处理图像时生成大量视觉 Token（如 LLaVA-NeXT 可达 2880 tokens，视频每秒 1fps 可达 200 万 tokens），Transformer 自注意力机制的二次复杂度导致推理成本过高。
- **高范数异常 Token 被误保留**：现有方法（如 VisionZip、FastV）依赖 CLS attention 或 text-visual attention 选择重要 Token，但高范数异常 Token（通常来自背景区域）因特征范数异常大，会吸引大量注意力，被错误地优先保留，浪费 Token 预算。
- **注意力汇聚（Attention Sink）与弥散**：LLM 解码器中存在大规模激活（massive activations）导致少数 Token 聚集过多注意力，同时 text-visual attention 弥散使模型难以对相关 Token 形成可靠排序，尤其在激进压缩比下失效。

## 核心贡献（创新点）
- **发现高范数异常 Token 的冗余性**：首次系统分析并证明高范数 Token 在特征和空间维度均高度冗余（与邻居相似性高、组内特征坍缩），且与注意力异常尖峰空间对齐。
- **提出无需训练的级联剪枝框架 SinkPruner**：通过视觉净化器（visual sanitizer）预处理视觉流，先过滤高范数异常 Token，再在 LLM 解码器早期层进行文本引导剪枝，实现 coarse-to-fine 剪枝。
- **揭示上游净化对下游注意力机制的增益**：经验证，视觉净化将 LLM 解码器中的注意力 sink Token 比例从 14.23% 降至 3.85%，text-visual attention 熵从 6.36 降至 4.85，显著提升后续剪枝可靠性。
- **高范数过滤模块的可移植性**：作为即插即用模块，可无缝集成到现有剪枝方法（如 VisionZip）中，带来稳定性能提升（MMB +2.6%，POPE +0.9%）。

## 方法详解
### 高范数异常 Token 识别
- 对视觉 Token $X = \{x_1, \ldots, x_N\} \in \mathbb{R}^{N \times D}$，计算每个 Token 的 $L_2$ 范数 $n_i = \|x_i\|_2$。
- 使用 scale-free top-ρ 规则：按范数降序排列，取 top ρ 作为高范数集合 $X_{high}$，其余为低范数候选集 $X_{low}$，默认 ρ = 1%。
- 高范数 Token 在空间上与其 4-近邻相似性极高，在特征空间中组内余弦相似度接近 1（特征坍缩）。

### 视觉净化器（Visual Sanitizer）
1. **高范数 Token 聚合**：将 $X_{high}$ 中的 Token 通过平均池化合并为单个 sink Token：
   $$x_{sink} = \frac{1}{|X_{high}|} \sum_{x \in X_{high}} x$$
2. **显著性保留**：从 $X_{low}$ 中选 CLS attention 最高的 top-$k_{res}$ 个 Token 构成 $X_{res}$：
   $$X_{res} = \{x \in X_{low} \mid \text{Rank}(A_{cls}(x)) \leq k_{res}\}$$
3. **多样性选择**：对剩余候选 $R = X_{low} \setminus X_{res}$，迭代选择与当前已选集合 S 相似度最低的 Token，避免特征坍缩：
   $$x^* = \arg\min_{x \in R} \left(\max_{s \in S} \text{CosSim}(x, s)\right)$$
4. 最终净化序列：$Z = [x_{sink}, X_{res}, X_{div}]$

### 文本引导剪枝器（Text-Guided Pruner）
- 在 LLM 解码器早期层（LLaVA-1.5 为 layers 2, 6, 15），利用 text-to-vision attention 评估每个视觉 Token 的相关性：
  $$\tilde{p}_j = \frac{1}{L_t} \sum_{i=1}^{L_t} \text{Softmax}(Q_{text} \cdot K_{vis}^\top)_{i,j}$$
- 保留 $\tilde{p}$ 最高的 top-K 个 Token。

### 无 CLS Token 编码器的适配
- 对于如 Qwen2.5-VL 等无 [CLS] Token 的视觉编码器，使用收到的平均自注意力作为显著性得分：
  $$s_j = \frac{1}{HN} \sum_{h=1}^{H} \sum_{i=1}^{N} A_{ij}^{(h)}$$

## 实验与结果
### 图像理解基准（LLaVA-1.5-7B）
| 方法 | 保留 Token | 平均性能保留率 |
|------|-----------|---------------|
| Upper Bound | 576 | 100% |
| VisionZip (CVPR25) | 64 (↓88.9%) | 93.2% |
| HoloV (NeurIPS25) | 64 (↓88.9%) | 94.7% |
| **SinkPruner** | 64 (↓88.9%) | **96.5%** |
| VisPruner (ICCV25) | 32 (↓94.4%) | 87.2% |
| **SinkPruner** | 32 (↓94.4%) | **91.2%** |

### 动态分辨率架构（Qwen2.5-VL-7B）
- 剪枝比 66.7%：保留 98.6% 性能
- 剪枝比 77.8%：保留 96.3% 性能
- 剪枝比 88.9%：保留 91.8% 性能，超过 HoloV +5.6、VisionZip +4.1

### 视频理解基准（Qwen2.5-VL-7B，80% 剪枝）
- MVBench: 66.70 (Full: 68.10)
- SEED-Bench: 61.80 (Full: 62.18)
- VideoMME: 58.59 (Full: 60.67)
- **平均保留 98.0%**，超越 DART (96.6%) 和 DivPrune (96.0%)

### 推理效率（POPE，90% 剪枝）
- 总推理时间从 19:28 降至 12:59（↓33.3%）
- Prefill 延迟 37.1 ms，Decode 延迟 86.0 ms
- 性能保留 97.1%，显著优于 VisionZip (91.8%) 和 FastV (65.5%)

### 消融实验
- 移除视觉净化器：MMB 从 61.6 降至 51.4（-10.2）
- 移除文本引导剪枝器：MMB 从 61.6 降至 59.4（-2.2）
- 不移除高范数 Token：MME 从 1754 降至 1706（-48）
- 高范数聚合略优于直接移除

## 相关工作脉络
- **Vision-centric 方法**（VisionZip、Faster-VLM、VisPruner）：依赖 CLS attention 选择 Token，但被高范数异常 Token 误导，优先保留冗余背景 Token。
- **Text-guided 方法**（FastV、SparseVLM）：利用 LLM 解码器 cross-modal attention 选择 Token，但受注意力汇聚和弥散影响，在激进剪枝下不可靠。
- **Token Merging/Dropping 方法**（ToMe、MustDrop、PDrop、PruMerge）：通过特征相似性合并或渐进式丢弃减少 Token，未针对高范数异常现象进行专门处理。
- **Hierarchical/Adaptive 方法**（HiDrop、AutoPrune、HoloV）：考虑层级保留或动态预算分配，但未解决上游视觉冗余对下游注意力的干扰。
- **SinkPruner 定位**：首次系统性地将高范数异常 Token 识别为视觉冗余的核心来源，通过上游净化为下游文本引导剪枝提供干净输入，形成 cascading 剪枝范式。

## 局限性与未来方向
- **离线推理假设**：当前方法假设输入为固定长度（静态图像或有限视频片段），未支持在线流式场景（如机器人连续感知）。
- **TextVQA 性能略逊**：在高分辨 tiled 编码下，小文字 patch 可能具有高范数，被错误聚合，未来可考虑引入局部边缘密度 exemt 机制。
- **超参数敏感性**：虽然经验证对 ρ、层位置、保留调度等具有鲁棒性，但未进行全网格搜索，当前配置为默认值而非最优。
- **视频应用扩展**：当前视频实验仅针对 Qwen2.5-VL，未见对其他视频 MLLM（如 Video-LLaVA）的验证。

## 研究启发与可借鉴点
- **高范数异常的识别可作为通用先验**：该现象在 CLIP 和 DINOv2 等不同编码器中均存在，可作为视觉 Token 冗余的通用检测信号，适用于多种 MLLM 架构。
- **级联剪枝范式（上游净化 + 下游引导）**：通过预处理净化视觉输入，为后续剪枝提供更干净的注意力景观，这一思路可迁移到其他需要注意力选择的任务中。
- **无需训练的即插即用模块设计**：视觉净化器无需微调，可直接集成到现有方法（如 VisionZip）中带来稳定提升，验证了该问题的本质性和解法的有效性。
- **注意力熵与 sink ratio 可作为诊断指标**：论文引入了 text-visual attention entropy 和 sink token ratio 作为量化注意力质量的指标，可用于评估和改进其他剪枝方法。
- **Scale-free 的 top-ρ 规则**：通过相对排名而非绝对阈值识别高范数 Token，避免了不同编码器间范数尺度差异带来的校准问题，具有良好可移植性。

## 关键术语表
- **High-norm outlier tokens**：视觉编码器中特征 $L_2$ 范数异常大的 Token，通常来自背景区域，在特征和空间维度高度冗余，易被错误保留。
- **Attention sink**：LLM 解码器中少数 Token 聚集过多注意力权重（无论语义相关性），导致大量计算资源浪费在非关键 Token 上。
- **Attention dispersion**：text-visual attention 过度分散，模型难以对查询相关区域形成明确聚焦，降低剪枝可靠性。
- **Visual sanitizer**：SinkPruner 的核心模块，通过识别和聚合高范数异常 Token，净化视觉序列，为下游剪枝提供干净输入。
- **Text-guided pruner**：利用 LLM 解码器中 text-to-vision attention 评估视觉 Token 与查询的相关性，保留最相关的 Token。
- **Scale-free top-ρ rule**：通过 Token 范数的相对排名（而非绝对阈值）识别高范数异常，避免不同编码器范数尺度差异带来的校准需求。
- **Feature collapse**：高范数 Token 在特征空间中高度相似（组内余弦相似度接近 1），表明其信息量低、冗余度高。
- **Cascading pruning**：级联剪枝策略，先在视觉编码器层面进行粗粒度净化，再在 LLM 解码器早期层进行细粒度文本引导剪枝。

## 可复现要素
- **数据集**：MMBench、MMB-CN、MME、POPE、ScienceQA、VQA-v2、TextVQA、MMStar、MMMU、AI2D、MM-Vet、MVBench、SEED-Bench、NextQA、VideoMME；图像和视频基准均为公开数据集。
- **代码开源**：https://github.com/LaVi-Lab/SinkPruner
- **模型权重**：使用开源的 LLaVA-1.5-7B、LLaVA-1.5-13B、LLaVA-NeXT-7B、Qwen2.5-VL-7B
- **关键超参**：ρ = 1%（高范数比例）；LLaVA-1.5 剪枝层为 (2, 6, 15)；Qwen2.5-VL 无 CLS Token，使用接收注意力替代；batched diversity selection 中 batch size b = 16
- **硬件**：NVIDIA A800-SXM4-80GB GPU
- **框架**：Python 3.10, PyTorch 2.1.2, CUDA 12.1
