---
title: "SinkPruner-Sink-Free-Visual-Token-Pruning-for-Multimodal-Lar"
source: https://arxiv.org/pdf/2609.01004v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 15:47:49"
---

# 论文速读：SinkPruner: Sink-Free Visual Token Pruning for Multimodal Large Language Models

## 一句话总结
针对多模态大语言模型（MLLM）因超长视觉Token序列导致的推理计算瓶颈，本文提出了一种免训练的粗到细级联剪枝框架SinkPruner。该方法通过视觉清洗器预先过滤高范数背景异常Token并聚合为单一Sink Token，有效抑制注意力汇聚与注意力弥散，再结合文本引导的多阶段剪枝保留语义对齐Token，在削减近90%视觉Token的同时实现极高的性能保留。

## 研究问题与动机
1. **视觉序列过长引发二次复杂度瓶颈**：MLLM将图像/视频投影为数千甚至数百万Token送入LLM解码器，Transformer自注意力机制的 $O(n^2)$ 复杂度导致推理延迟与显存开销巨大，严重制约边缘端与长视频场景部署。
2. **视觉中心剪枝受“高范数陷阱”误导**：VisionZip、Faster-VLM等方法依赖[CLS]注意力保留Token，但高范数异常Token（通常来自均质背景）会掠夺异常高的注意力权重，被错误判定为重要Token，挤占真正携带语义信息的低范数Token预算。
3. **文本引导剪枝受注意力汇聚与弥散干扰**：在LLM解码器端利用文本-视觉注意力选Token时，Massive Activations（注意力汇聚）使模型将大量注意力锚定在少数无语义Sink Token上，剩余Token注意力分布极度分散（高熵），导致按注意力排序的剪枝决策在高压缩率下不可靠。

## 核心贡献（创新点）
1. **系统揭示高范数异常Token的双重冗余本质**：从空间维度（高邻居相似性、集中于同质背景）与特征维度（极高 intra-set pairwise cosine similarity、表示坍缩）证明其冗余性，指出其是现有注意力剪枝失效的根本病理。
2. **提出免训练的视觉清洗器（Visual Sanitizer）**：采用尺度无关的 top-ρ 规则自动切分高低范数Token，通过平均池化将异常Token聚合为单一 Sink Token，并结合 CLS 注意力筛选与基于余弦距离的多样性去重策略，输出纯净且空间多样的视觉子集。
3. **设计级联式文本引导渐进剪枝框架**：在清洗后的序列基础上，于LLM解码器早期层利用累积文本-视觉注意力进行多阶段Token保留，将注意力汇聚比例降低约73%，并显著压低文本-视觉注意力熵，使下游语义对齐更可靠。
4. **验证强架构泛化性与模块可迁移性**：框架无缝适配固定网格（LLaVA系列）与动态分辨率（Qwen2.5-VL）架构，且在视频理解任务上表现优异；视觉清洗器可作为正交即插即用模块独立提升 VisionZip 等基线 0.9%~2.6% 性能。

## 方法详解
- **整体流程**：SinkPruner 采用“先视觉清洗、后文本引导剪枝”的级联架构，全程免训练，仅在推理阶段运行。
- **视觉清洗器（Visual Sanitizer）**：
  - **高范数识别**：计算视觉Token的 $L_2$ 范数 $n_i = \|x_i\|_2$，按数值相对排名取顶部 $\rho$（默认 $\rho=1\%$）为高范数集合 $\mathbf{X}_{high}$，其余为 $\mathbf{X}_{low}$。该策略为尺度无关，无需针对特定编码器设定绝对阈值。
  - **Sink Token聚合**：利用 $\mathbf{X}_{high}$ 内部极高相似性，通过平均池化压缩为单代理Token $x_{sink} = \frac{1}{|\mathbf{X}_{high}|} \sum_{x \in \mathbf{X}_{high}} x$，保留全局背景概览。
  - **混合保留策略**：从 $\mathbf{X}_{low}$ 中按 $\text{CLS}$ 注意力排序选取显著性集合 $\mathbf{X}_{res}$；对剩余候选 $\mathcal{R}$ 执行迭代最远点采样：每次选取与已选集合 $\mathcal{S}$ 最大余弦相似度最低的Token加入 $\mathbf{X}_{div}$，兼顾显著性与空间多样性。最终净化序列 $\mathbf{Z} = [x_{sink}, \mathbf{X}_{res}, \mathbf{X}_{div}]$。
  - **无[CLS]编码器适配**：针对 Qwen2.5-VL 等无 CLS Token 的视觉塔，使用倒数第二层所有注意力头的平均接收注意力 $s_j = \frac{1}{HN}\sum_{h,i} A_{ij}^{(h)}$ 替代 CLS 注意力，无需修改主体流水线。
- **文本引导剪枝器（Text-Guided Pruner）**：
  - 在LLM解码器指定层（如 LLaVA-1.5 的 layer 2, 6, 15）执行多阶段渐进剪枝。
  -
