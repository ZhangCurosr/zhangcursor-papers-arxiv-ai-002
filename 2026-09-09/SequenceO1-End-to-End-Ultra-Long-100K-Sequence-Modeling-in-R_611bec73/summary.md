---
title: "SequenceO1-End-to-End-Ultra-Long-100K-Sequence-Modeling-in-R"
source: https://arxiv.org/pdf/2609.08443v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 03:29:41"
field: "工业推荐系统长序列建模"
keywords: ["推荐系统", "长序列建模", "Sketch Attention", "端到端排序", "低秩缓存"]
innovations: ["原型侧归一化的Sketch Attention实现可缓存固定大小sketch", "双时间尺度STCA推理（近期10K+超长sketch）", "训练-serving端KVCache+MRLB使100K路径开销降至O(1)"]
benchmarks: ["Douyin离线AUC/UAUC", "Douyin/Douyin Lite在线A/B（Finish, Activeness, Duration）"]
---

# 论文速读：SequenceO1-End-to-End-Ultra-Long-100K-Sequence-Modeling-in-R

## 一句话总结
本文提出SequenceO1，一种在抖音全量流量下部署的端到端推荐排序框架，通过Sketch Attention将100K级超长期用户行为序列压缩为固定大小、可缓存的用户端sketch，再结合STCA进行双时间尺度推理，显著降低训练与推理计算开销。

## 研究问题与动机
- **100K超长期历史是实际瓶颈**：抖音单个用户一年的消费行为可累积至10^5量级，包含稳定偏好、周期性模式等关键信号，但细粒度排序阶段受严格延迟与训练吞吐量约束。
- **现有方法的系统性不足**：截断丢失长期信号；多阶段检索（如TWIN V2）将历史压缩与最终排序目标解耦；直接缩放STCA到100K因线性复杂度仍面临巨大特征存储、通信与计算成本。
- **100K不仅是单层计算问题**：是端到端系统工程挑战，涉及特征存储、主机-设备通信、分布式训练与-serving全链路，而非仅优化单层注意力计算。
- **压缩与复用是核心机会**：超长期历史具有可压缩性与可复用性，若能提取仅依赖用户侧的固定大小表示并跨target/请求缓存，则可大幅降低重复计算。

## 核心贡献（创新点）
- **端到端100K排序框架**：提出SA+STCA的双时间尺度推理架构，首次将端到端长序列排序扩展至100K级，与多阶段检索（TWIN V2）的本质区别在于联合优化最终排序目标。
- **可缓存的低秩sketch机制（Sketch Attention）**：设计目标无关的可学习原型压缩模块，输出固定长度$R^{k×d}$的sketch，参数规模与原始序列长度$N$无关，实现长度无关的差异化压缩。
- **缓存优先的训练与推理系统**：引入本地KVCache（训练端MRLB+-serving端缓存）、FlashSA算子与pipeline lift，将cache-hit路径的开销降至$O(1)$（相对于原始序列长度），使100K建模在生产可行。

## 方法详解
- **Sketch Attention (SA)**：给定历史嵌入$X∈R^{n×d}$，学习$k$个全局共享原型$P^{(0)}∈R^{k×d}$，计算原型-词元相似度$S=(P^{(0)}W_Q)(XW_K)^T/√d$，采用**原型侧归一化**（prototype-wise softmax）而非标准token侧归一化：$A_{j,i}=exp(S_{j,i})/Σ_{j'}exp(S_{j',i})$，使每个行为词元在原型上分配质量，再聚合得sketch $\tilde{X}=AX∈R^{k×d}$。
- **堆叠精炼**：通过残差连接与SwiGLU FFN迭代$N_{sa}=2$次 refine sketch，提升表达能力同时保持参数高效。
- **双时间尺度STCA推理**：近期$X_r$（10K）经STCA处理捕获时效信号；超长历史sketch经宽度适配器$W_A$映射到STCA宽度$D$后，通过STCA处理超长信号；两者与目标特征经MixFormer轻量融合。
- **训练端本地KVCache**：缓存用户级sketch，以user id+模型版本+时间戳为key，TTL 3小时，训练hit rate约0.5；-serving端TTL 1小时，hit rate约0.6。
- **MRLB（Multi-Request Level Batching）**：同用户多请求分组复用sketch计算，训练reuse ratio $R_{train}=40$，推理$R_{infer}=300$。
- **FlashSA算子**：流式block-wise计算原型-词元相似度与sketch聚合，避免显式存储$k×N$中间矩阵，减少HBM传输。
- **Pipeline Lift**：因sketch仅依赖用户侧，可将SA计算上移至上游阶段，与召回并行，miss时仍可掩盖延迟。

## 实验与结果
- **离线轻量消融（STCA(512)基线）**：最终SA配置（$k=1K, N_{sa}=2, d=128$）达到Finish AUC +1.07%；原型侧归一化相比token侧在多个engagement指标上均有提升（Table 1）。
- **压缩方法对比（Table 3）**：SA（+1.07%）优于TWIN V2 KMeans（+0.30%）、Lightning-Attention（+0.83%）、position-bucket queries（+0.72%）。
- **生产环境离线评估（Table 4）**：替代TWIN V2后，Finish AUC +0.29%，Dislike AUC +1.29%，UAUC在Favourite上+1.47%、Dislike上+3.63%。
- **在线A/B（Table 5）**：Douyin Finish +2.33%、Douyin Lite +3.49%，30日活跃度、时长、评论、点赞均显著提升，Dislike下降约-7%至-6%。
- **计算节省（Figure 1）**：在训练端$R=40$、hit=0.5时，100K下SequenceO1相比直接STCA节省约49.9× FLOPs；推理端$R=300$、hit=0.6时节省约63.9×。
- **保留83%直接缩放收益**：Figure 1(c)显示100K截断（平均85K）下，SequenceO1保留直接扩展STCA增益的83%（+1.07% vs +1.29%）。

## 相关工作脉络
- **TWIN V2**（Si et al., 2024）：离线层次聚类压缩全生命周期行为，在线检索后target attention；本质是多阶段分离，非端到端联合优化。
- **STCA**（Guan et al., 2025）：Stacked Target-to-History Cross Attention，单query交叉注意力复杂度$O(L)$，已部署于10K，但直接扩展到100K仍受线性增长限制。
- **Linformer**（Wang et al., 2020）：低秩注意力近似，使用显式长度相关的投影矩阵；SA是隐式、数据依赖、长度无关的压缩，侧重可缓存性。
- **Perceiver/Set Transformer**：使用inducing points或latent arrays压缩输入；SA采用原型侧归一化保障全覆盖，与target-agnostic压缩目标对齐。
- **LLaTTE / SOLARIS**：两阶段表示迁移系统（异步user model→在线ranker），存在表征瓶颈与目标错位；SequenceO1保持端到端联合优化。
- **FlashAttention**（Dao et al., 2022/2023）：IO-aware精确注意力算子；本文FlashSA是其思想在recommendation场景的适配，处理ragged batching下的sketching。

## 局限性与未来方向
- **固定sketch容量假设**：不同用户行为冗余度不同，统一$k$可能无法自适应；未来可探索分层或多分辨率sketch。
- **缓存时效性权衡**：TTL策略需平衡新鲜度与命中率，极端情况下miss成本仍存在（虽经MRLB摊销）。
- **未评估sketch检索扩展**：分配模式可支持未来基于sketch token的行为检索或局部精炼，但当前未实现。
- **扩展至million-scale需进一步验证**：虽设计支持，但实际部署在更高量级可能面临新的存储/通信瓶颈。

## 研究启发与可借鉴点
- **"压缩-推理"分离范式**：将用户侧通用压缩与target条件推理解耦，可实现跨target复用，适用于多种长序列场景。
- **原型侧归一化 vs token侧归一化**：在需要全覆盖的target-agnostic压缩任务中，原型侧softmax更能避免信息丢失，值得在其它压缩型attention中尝试。
- **缓存友好算子设计**：FlashSA流式处理避免大中间矩阵显式化，对工业界GPU内存受限场景有借鉴价值。
- **端到端替代多阶段流水线**：用可缓存sketch替代TWIN V2等两阶段模块，简化系统同时提升性能，证明联合优化优于模块级最优。
- **FLOP量化分析框架**：附录G给出完整的GEMM主导复杂度推导与命中/未命中cost分解，为后续系统benchmark提供方法论参考。

## 关键术语表
- **Sketch Attention (SA)**：将超长期历史压缩为固定大小、目标无关的sketch的注意力模块，采用原型侧归一化实现全覆盖。
- **STCA（Stacked Target-to-History Cross Attention）**：堆叠的单query target-history交叉注意力，复杂度$O(L)$，用于细粒度偏好匹配。
- **MRLB（Multi-Request Level Batching）**：按用户分组多个请求复用用户侧计算（如sketch），提升训练吞吐。
- **Prototype-wise normalization**：沿原型维度做softmax归一化，使每个token分配到各原型，而非token选择机制。
- **FlashSA**：流式block-wise计算SA的融合算子，避免$k×N$中间矩阵驻留HBM。
- **Local KVCache**：训练端缓存用户级sketch的本地存储，按TTL+容量淘汰管理。
- **Pipeline lift**：将sketch计算提前至上游阶段，与召回并行以掩盖延迟。
- **Low-rank caching**：将超长序列沿长度维度压缩为低秩固定表示并缓存复用。

## 可复现要素
- **数据集**：抖音生产离线数据集（内部数据，未公开）；在线A/B测试在Douyin与Douyin Lite。
- **代码/权重**：论文未提及开源；框架已全量部署于抖音生产环境。
- **关键超参**：SA宽度$d=128$，原型数$k=1024$（消融用1K），STCA宽度$D=1024$（16头×64），$N_{sa}=2$，$N_{stca}=4$；训练hit rate=0.5，推理hit rate=0.6；MRLB ratio训练40、推理300；TTL训练3h/推理1h。
