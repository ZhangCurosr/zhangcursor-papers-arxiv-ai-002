---
title: "Token-Oriented-Semantic-Communication-with-Pretrained-Vision"
source: https://arxiv.org/pdf/2608.25410v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 16:16:17"
field: "语义通信与边缘AI"
keywords: ["语义通信", "token通信", "学习型图像压缩", "Vision Transformer", "协同推理", "边缘计算"]
innovations: ["利用ViT与LIC的空间一一对应实现token粒度的选择性latent传输", "分层选择性注意力滚降以单次前向传播实现高效token相关性估计", "单代理token参数高效适配大ViT处理不完整输入"]
benchmarks: ["ImageNet-1K"]
---

# 论文速读：Token-Oriented Semantic Communication with Pretrained Vision Transformers

## 一句话总结
本文提出了一种基于预训练ViT的模块化token导向语义通信框架，通过ViT patch token与LIC latent之间的空间一一对应关系，利用token级任务相关性选择性传输压缩图像latent，并在服务端用可学习代理token弥补缺失区域，实现了优于现有语义通信方案、手工编码器和任务无关LIC模型的高效率客户端-服务器协同推理。

## 研究问题与动机
1. **通信成本高昂**：直接传输token embedding需发送高维向量，即使只传输部分token也带来较大比特负载。
2. **跨模型嵌入空间互操作性差**：不同模型因架构、预训练方式、下游任务差异导致embedding空间难以对齐，限制模块化解耦部署。
3. **现有端到端方案的局限性**：VQ-based token通信等方法需联合训练收发器并共享codebook，降低了模块化程度，增加了重训练成本，且通信策略与特定下游任务强耦合。
4. **先前token-oriented方法在压缩效率上的不足**：已有的token-oriented方案在像素域传输图像块，未能充分利用学习型图像压缩（LIC）的统计冗余消除能力。

## 核心贡献（创新点）
1. **Token-aligned LIC（token对齐学习型图像压缩）**：利用ViT patch token与LIC latent向量的一一空间对应关系，将token级任务相关性直接映射为latent选择性传输索引，无需修改LIC架构即可实现token粒度的语义压缩——与先前在像素域传输patch或标量量化的方法本质不同，利用了LIC的统计冗余获得更优率失真性能。
2. **Layer-selective attention rollout（分层选择性注意力滚降）**：仅在ViT的中间层范围内（而非全部层或最后一层）聚合attention权重来估计token任务相关性，只需单次前向传播即可获得比last-layer attention和全层rollout更准确的关注度图，且避免了gradient-weighted attention所需的反向传播开销。
3. **Learnable surrogate token substitution（可学习代理token替换）**：在服务端用单个可学习token替换未选中区域的视觉token，以参数高效方式（仅D=1024个参数）适配大ViT处理不完整输入，同时保持原始预训练模型对完整输入的预测行为不变——区别于直接重建推断或全量微调服务器的方法。

## 方法详解
框架由三个预训练组件协调工作：**轻量级客户端ViT（DeiT-Tiny）**、**学习型图像压缩（LIC）模型**、**大容量服务端ViT（DeiT-III-Large）**，无需端到端联合训练。

1. **Token对齐LIC原理**：ViT的patch大小（16×16）与LIC编码器的空间下采样因子（4次stride-2卷积，下采样16倍）一致，使得每个视觉token $\mathbf{x}_i^t$ 与第i个latent向量 $\widehat{\mathbf{x}}_i^c$ 形成一一空间对应，selection map $\mathbf{s} \in \{0,1\}^N$ 可直接索引latent进行选择性熵编码。
2. **选择性传输**：客户端根据阈值δ按相关性降序选取token，构建选择集合 $S$ 和未选集合 $\mathcal{U}$，仅对 $i \in S$ 的latent进行算术编码，未选位置完全跳过编码，传输比特流 $(\widehat{\mathbf{X}}_S^c, \widehat{\mathbf{Z}}, \mathbf{s})$。
3. **预测均值插补**：服务端在解码前，用hyper decoder预测的均值 $\widehat{\pmb{\mu}}_i$ 填充未选latent位置：$\widetilde{\mathbf{x}}_i^c = s_i \widehat{\mathbf{x}}_i^c + (1-s_i)\widehat{\pmb{\mu}}_i$，保留局部统计一致性，提升选中区域的重建质量。
4. **分层选择性注意力滚降**：从第 $L_s$ 层到第 $L_e$ 层聚合注意力权重：$\mathbf{R} = \widetilde{\mathbf{A}}_{L_e}\widetilde{\mathbf{A}}_{L_e-1}\cdots\widetilde{\mathbf{A}}_{L_s}$，其中 $\widetilde{\mathbf{A}}_l = \mathbb{E}_h[\mathbf{A}_{l,h}] + \mathbf{I}$，再按公式(5)归一化提取token级任务相关性 $\bar{\mathbf{r}}$；本文选取 $(L_s, L_e)=(7,12)$ 避免早期层噪声和末层背景漂移。
5. **代理token替换**：服务端构造输入序列 $\widetilde{\mathbf{X}}^t$，将未选位置的token替换为可学习向量 $\mathbf{x}_{\text{sur}}$：$\widetilde{\mathbf{x}}_{i+1}^t = s_i\widehat{\mathbf{x}}_i^t + (1-s_i)\mathbf{x}_{\text{sur}} + \mathbf{e}_i$，仅优化 $\mathbf{x}_{\text{sur}}$ 而冻结骨干网络。

## 实验与结果
- **数据集**：ImageNet-1K，图像裁剪至224×224。
- **基线**：先前的token-oriented语义通信方法（selected-patch传输[33]、importance-aware quantization[34]）、手工编码器（JPEG2000、WebP、BPG）、任务无关LIC模型（Balle et al.[26,27]、ELIC[28]）。
- **主要结果**：
  - 在1.24 bpp时达到**85.83%**准确率，仅低于服务器理论上限86.81%约0.98个百分点；WebP需约1.57 bpp、BPG需约1.82 bpp才能匹配，优势显著。
  - 结合熵感知图像传输（EIT）后，在**0.94 bpp**时达到**85.89%**准确率。
  - 在主要运行区域内，token-aligned LIC、layer-selective attention rollout、surrogate token三者均优于对应消融基线。
- **最强结果**：在$\lambda=0.3$、$\delta$调优下，提出的框架在低于2.5 bpp的范围内持续优于所有对比方法。

## 相关工作脉络
1. **VQ-based token通信**（如[14,22-24]）：需端到端联合训练和共享codebook，缺乏模块化；本文方法无需联合训练，各组件可独立替换。
2. **先前token-oriented通信**（如[31-34]）：在像素域传输图像块（原始或标量量化），不能利用LIC的统计冗余；本文在LIC latent域操作，获得更高压缩效率。
3. **Last-layer attention相关性估计**（如[33]）：仅用末层注意力，易受背景漂移影响；本文分层聚合中间层注意力，准确性更优。
4. **Attention rollout**（[35]）：累积全部层注意力，包含早期层噪声；本文剔除噪声层，精度更高。
5. **Gradient-weighted attention**（[36]）：需反向传播计算梯度，资源开销大；本文方法仅需单次前向传播即可达到 comparable accuracy。
6. **视觉prompt tuning**（[37]）： prepend少量prompt token进行任务适配；本文仅用单个surrogate token替换缺失位置，目标是为缺失信息补偿而非任务迁移。

## 局限性与未来方向
1. **空间对齐假设的限制**：依赖ViT patch大小与LIC下采样因子一致的架构假设，对高分辨率图像或非标准patch大小的场景需额外设计。
2. **单代理token表达能力有限**：当前仅用一个可学习向量统一替换所有未选位置，对缺失信息结构复杂时可能不足以充分补偿。
3. **仅验证于图像分类任务**：框架的通用性需在目标检测、分割等其他视觉任务中进一步验证。
4. **未与信道编码深度集成**：尽管在packet-erasure channel上验证了鲁棒性，但实际无线信道（如衰落信道）下的端到端联合优化有待探索。
5. **未来方向**：可扩展至视频/多模态token通信、探索更灵活的跨模型token对齐机制、与信道编码联合设计以提升恶劣信道下的性能。

## 研究启发与可借鉴点
1. **Token级相关性指导特征压缩**：利用ViT内部token relevance来决定哪些特征被传输/压缩，而非直接传输embedding，可有效解耦通信策略与模型embedding空间，这一思路可迁移到其他多模态协同推理场景。
2. **分层注意力聚合的实用性**：发现early层attention噪声大、late层attention有背景漂移，选择中间层范围聚合可兼顾准确性与计算效率，该设计模式可直接复用于其他基于attention的任务重要性估计任务。
3. **参数高效的不完整输入适配**：单代理token替换方案以极低成本（D个参数）实现服务端模型对缺失输入的适配，同时保持原始完整输入性能不变，该"冻结骨干+少量可学习token"范式可推广到遮挡鲁棒性、跨区域融合等场景。
4. **理论分析与工程实现结合**：论文给出了选择性传输率减少的理论上界（式10-11），定量刻画了side-information overhead与selection ratio的关系，为后续工作提供理论参照基准。
5. **可复现的实验设计**：代理token在模拟实际传输流程（emulation）下进行训练，确保训练-测试分布一致，这一训练协议对类似"缺失信息补全"任务具有参考价值。

## 关键术语表
**Token-oriented semantic communication**：以transformer token粒度执行语义通信的方法，利用token级任务相关性决定传输内容而非完整embedding。
**Layer-selective attention rollout**：仅在选定层范围内聚合自注意力权重以估计token任务相关性，避免早期层噪声和末层背景漂移。
**Token-aligned LIC**：利用ViT patch token与LIC latent的空间一一对应关系，将token选择图直接索引latent向量的学习型图像压缩方法。
**Surrogate token**：可学习的单一token向量，用于在服务端替换未传输区域的视觉token，以参数高效方式适应不完整输入。
**Predicted-mean imputation**：服务端用hyper decoder预测的条件期望值填补缺失latent位置，保持局部统计一致性。
**Entropy-aware image transmission (EIT)**：客户端在置信度高时本地完成分类而不传输图像，仅在min-entropy超过阈值时才发起服务器推理。
**Rate-accuracy trade-off**：通信比特率与下游任务分类准确率之间的折衷关系，是语义通信的核心评估指标。

## 可复现要素
- **数据集**：ImageNet-1K（公开）
- **代码/权重**：论文未明确声明代码开源状态；使用了公开的Pretrained DeiT-Tiny/DeiT-III-Large和LIC模型
- **关键超参**：token选择阈值δ∈[0.5, 0.98]；LIC超参λ∈{0.0075, 0.03, 0.1, 0.2, 0.4, 1}；分层注意力范围$(L_s, L_e)=(7,12)$；代理token训练超参$(\delta_{\text{sur}}, \lambda_{\text{sur}})=(0.35, 0.03)$；训练轮数20 epoch
