---
title: "Token-Oriented-Semantic-Communication-with-Pretrained-Vision"
source: https://arxiv.org/pdf/2608.25410v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 16:15:56"
field: "语义通信与边缘智能"
keywords: ["语义通信", "Token通信", "边缘计算", "ViT", " Learned Image Compression", "协作推理"]
innovations: ["利用ViT与LIC的空间对齐实现token粒度选择性latent传输", "层选择性注意力回滚高效估计token任务相关性", "单向量可学习代理token实现参数高效的服务器端适配"]
benchmarks: ["ImageNet-1K"]
---

# 论文速读：Token-Oriented Semantic Communication with Pretrained Vision Transformers

## 一句话总结
本文提出了一种基于预训练模型的token导向语义通信框架，通过ViT patch token与LIC latent vector的空间对齐关系，在不直接传输token embedding的前提下实现token粒度选择性传输；核心利用层选择性注意力回滚估计任务相关性，并结合可学习代理token补偿服务器端推理中的信息缺失，在ImageNet上实现了优于现有语义通信方案和传统编解码器的率-准确率权衡。

## 研究问题与动机
- **通信成本过高**：直接传输token embedding需要发送高维向量（维度随模型容量增大），即使减少传输数量也会产生大量负载。
- **跨模型互操作性差**：不同架构、预训练流程和下游目标独立训练的模型，其token embedding空间存在显著差异，难以兼容。
- **现有方案的局限**：端到端联合训练方案虽有效但降低模块化、增加重新训练成本，且通信策略与特定下游任务耦合；已有token-oriented方法仅传输像素域图像块，缺乏高效的压缩效率。
- **资源受限边缘设备的需求**：边缘侧资源有限，需在不修改底层模型的前提下实现高效的协作推理。

## 核心贡献（创新点）
- **Token-aligned Learned Image Compression（令牌对齐的 learned image compression）**：利用ViT patch大小与LIC编码器空间下采样因子的一致性（均为16），实现token级别的task relevance到latent向量的直接映射，选择性传输任务相关latent而非整个embedding矩阵，与仅传输像素域patch的方法本质区别在于利用了LIC的统计冗余去除能力。
- **Layer-selective Attention Rollout（层选择性注意力回滚）**：通过在选定的中间层范围内聚合注意力权重来估计token级任务相关性（DeiT-Tiny设为第7至12层），避免早期层的噪声干扰和最后一层的背景漂移问题，相比全层rollout和梯度加权注意力，仅需单次前向传播即可达到同等甚至更优的准确性。
- **Learnable Surrogate Token Substitution（可学习代理token替换）**：在服务器端用单个可学习的代理token替换未选中区域的token位置，以冻结主干网络的方式优化分类目标，参数效率极高（仅D=1024个参数），同时保持完整输入行为不变，弥补了直接重建推理和仅使用选中patch推理的不足。

## 方法详解
**整体框架**：客户端部署轻量级ViT（DeiT-Tiny）和神经压缩器，服务器部署大型ViT（DeiT-III-Large）和解压模块，三者均为预训练模型，无需端到端联合训练。

**关键流程**：
1. 客户端从输入图像提取视觉token，通过层选择性注意力回滚计算token级任务相关性$\bar{\mathbf{r}}$
2. 基于阈值$\delta$构建token选择映射$\mathbf{s} \in \{0,1\}^N$，按相关性降序累积选择直到超过阈值
3. Token-aligned LIC利用空间对齐特性，仅对选中位置的latent进行算术编码，未选中位置由超先验预测均值填充
4. 传输$(\widehat{\mathbf{X}}_S^c, \widehat{\mathbf{Z}}, \mathbf{s})$，服务器接收后重建图像，并用代理token替换未选中区域token
5. 服务器ViT对修正后的token序列进行分类推理

**关键技术细节**：
- **空间对齐机制**：ViT的patch size=P=16，LIC编码器通过4个stride-2卷积下采样，两者均将输入分辨率降至1/P，实现一一对应关系
- **选择性latent传输**：利用mean-scale hyperprior架构中latent元素的条件独立性，仅编码选中位置的latent，速率降低近似线性于选择比例$\rho$
- **预测均值插补**：未选中latent用超解码器预测的条件期望$\widehat{\mu}_i$填充，公式为$\widetilde{\mathbf{x}}_i^c = s_i\widehat{\mathbf{x}}_i^c + (1-s_i)\widehat{\pmb{\mu}}_i$
- **代理token训练**：冻结服务器ViT主干，仅优化单个$\mathbf{x}_{sur} \in \mathbb{R}^{1\times D}$，训练时模拟实际pipeline（$\delta_{sur}=0.35, \lambda_{sur}=0.03$）

## 实验与结果
- **数据集**：ImageNet-1K，图像裁剪 resize 至(224,224)
- **基线对比**：
  - 先前的token-oriented语义通信方案（selected-patch transmission [33]、importance-aware quantization [34]）
  - 手工编解码器（JPEG2000、WebP、BPG）
  - 任务无关LIC模型（Balle et al. [26]-[28]）
- **主要结果**：
  - 在1.24 bpp时达到85.83%准确率，仅比服务器完整输入准确率（86.81%）低0.98个百分点；相比之下WebP和BPG需约1.57 bpp和1.82 bpp才能匹配
  - 结合Entropy-aware Image Transmission (EIT)后，在0.94 bpp达到85.89%准确率
  - 在包擦除信道下，surrogate token substitution相比直接重建推理和选中patch推理具有最佳鲁棒性
- **提升幅度**：相对于先前的token-oriented方法在相同bpp下获得更高准确率；相比手工编解码器在<2 bpp regime表现更优；相比任务无关LIC在<2.5 bpp范围有优势

## 相关工作脉络
- **VQ-Based Token Communications [14],[22]-[24]**：通过矢量量化码本传输token表示，需端到端联合训练，模块性差；本文方法避免直接传输token embedding，保持模块化
- **Prior Token-Oriented Communications [31]-[34]**：同样利用token级相关性指导传输，但操作于像素域传输图像块；本文在LIC latent域操作，获得更优的率效率
- **Learned Image Compression [25]-[27]**：关注重构保真度的自回归/非自回归压缩模型；本文在此基础上引入任务相关性引导的选择性传输
- **Attention-Guided Importance Estimation [35],[36]**：last-layer attention、attention rollout、gradient-weighted attention；本文的layer-selective rollout结合精度与计算效率优势
- **Visual Prompt Tuning [37] & Masked Image Modeling [38],[39]**：可学习输入token的已有应用；本文的surrogate token结合两者特点，针对缺失信息补偿场景设计

## 局限性与未来方向
- **依赖空间对齐假设**：方法前提是ViT patch size与LIC下采样因子一致，对架构变体需额外适配
- **仅评估图像分类任务**：未验证在检测、分割等其他视觉任务上的泛化性
- **EIT为可选附加组件**：文中指出entropy-aware image transmission并非本文贡献，仅为补充说明
- **延迟未详细分析**：主要关注率-准确率权衡，端到端推理延迟的开销未深入讨论
- **信道模型简化**：采用 packet-erasure channel 评估鲁棒性，未考虑更复杂的无线信道 impairments

## 研究启发与可借鉴点
- **空间对齐设计思想**：ViT与LIC的自然空间对应关系可用于其他transformer-LIC协同场景，如跨模态通信或视频压缩
- **层选择性聚合策略**：排除噪声层、保留关键层的注意力聚合思路可迁移至其他需要token重要性估计的任务
- **参数效率极高的适配方式**：单向量可学习token的服务器端适配方案，为联邦学习、模型卸载等场景提供低开销adaptation范式
- **无需端到端训练的模块化设计**：在资源受限边缘设备部署大模型时，分离式推理架构具有实际应用价值
- **率-准确率的细粒度控制**：仅调整δ即可在窄带宽范围内自适应，为动态信道环境提供灵活调控手段

## 关键术语表
- **Token-oriented semantic communication**：在transformer token粒度上实现语义通信原则，通过task relevance决定传输内容而非完整特征
- **Learned Image Compression (LIC)**：基于深度学习的图像压缩方法，利用神经网络学习图像统计特性以达到接近Shannon极限的压缩效率
- **Layer-selective attention rollout**：仅在选定范围的transformer层上进行注意力权重累积，平衡准确性与计算开销
- **Surrogate token**：单个可学习向量，用于替换未传输区域的token位置，以冻结主干网络方式优化分类目标
- **Mean-scale hyperprior**：LIC中的一种熵模型，同时预测latent的均值和方差以精确建模条件分布
- **Rate-accuracy trade-off**：通信码率与下游任务准确率之间的权衡关系，是语义通信的核心评估指标
- **Packet-erasure channel**：模拟无线信道中数据包丢失的信道模型，用于评估系统的鲁棒性

## 可复现要素
- **数据集**：ImageNet-1K（公开可用）
- **代码/权重**：论文未明确提及代码开源状态；DeiT-Tiny、DeiT-III-Large、LIC模型使用公开预训练权重
- **关键超参**：
  - 层选择范围：$(L_s, L_e) = (7, 12)$（DeiT-Tiny）
  - 代理token训练参数：$\delta_{sur} = 0.35, \lambda_{sur} = 0.03$
  - 测试时阈值扫描范围：$\delta \in [0.5, 0.98]$，$\lambda \in \{0.0075, 0.03, 0.1, 0.2, 0.4, 1\}$
  - LIC通道维度：latent $D_c = 192$，hyperprior $= 128$
