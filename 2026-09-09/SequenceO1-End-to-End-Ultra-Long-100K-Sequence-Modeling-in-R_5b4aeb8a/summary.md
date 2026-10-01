---
title: "SequenceO1-End-to-End-Ultra-Long-100K-Sequence-Modeling-in-R"
source: https://arxiv.org/pdf/2609.08443v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 03:29:45"
field: "推荐系统长序列建模"
keywords: ["recommendation systems", "long-sequence modeling", "Sketch Attention", "cache amortization", "end-to-end ranking", "industrial recommender"]
innovations: ["Sketch Attention: prototype-wise normalization 实现 target-agnostic 固定大小可缓存 sketch 压缩", "Cache-first 100K 训练/推理系统: local KVCache + MRLB + FlashSA kernel 协同实现 O(1) cache-hit 路径", "双时间尺度 STCA 推理: 10K recent suffix + fixed-size ultra-long sketch 联合优化"]
benchmarks: ["Douyin offline AUC/UAUC", "Douyin online A/B (Finish, Duration, Activeness)", "Douyin Lite online A/B"]
---

# 论文速读：SequenceO1-End-to-End-Ultra-Long-100K-Sequence-Modeling-in-R

## 一句话总结
本文提出了 **SequenceO1**，一种在抖音全流量部署的端到端超长期（100K）序列建模框架，通过 **Sketch Attention（SA）** 将超长用户行为历史压缩为固定大小、可缓存的用户侧 sketch，再结合双时间尺度 STCA 推理与训练/推理侧缓存复用机制，实现了 100K 序列下训练与推理计算成本分别降低约 **49.9×** 和 **63.9×**，同时在线 A/B 测试获得 Finish +2.33%（抖音）/ +3.49%（抖音极速版）的显著提升。

## 研究问题与动机
1. **100K 级用户历史建模的工程瓶颈**：短视频推荐中单用户一年消费行为可达 10⁵ 量级，包含稳定偏好与周期性模式，但细排阶段受严格延迟与训练吞吐约束，无法直接端到端建模。
2. **现有方案的缺陷**：截断、多阶段检索（如 TWIN V2）或长度外推（如 STCA/HSTU）要么牺牲端到端优化，要么仍需大量特征存储/通信/计算成本；STCA 的 target-conditioned cross attention 是 query-dependent 的，跨 target 复用性差。
3. **核心观察**：超长用户历史应具备**可压缩性**与**可复用性**——若能将 100K 序列压缩为仅依赖 user-side 信号的固定大小 sketch，并一次性计算后跨 target、跨请求复用，则可避免反复存储、传输与处理原始超长序列。
4. **目标**：在保留端到端优化路径的同时，协同设计模型压缩机制与系统缓存复用策略，使 100K 序列建模在工业生产中可行。

## 核心贡献（创新点）
1. **端到端 100K 序列细排框架**：将端到端长序列推荐扩展至 100K 历史，提出 SA+STCA 双分支架构；与 TWIN V2 等多阶段检索方案本质区别在于全程联合优化，无需离线聚类或在线检索两阶段分离。
2. **Sketch Attention（SA）作为可缓存的低秩压缩模块**：使用可学习 prototype 与 prototype-wise normalization，将 n=100K 历史压缩为固定长度 k（通常 512–1024）的 target-agnostic sketch；与 Perceiver/Linformer 的本质区别在于归一化轴改为 prototype 维度，保证全量历史覆盖而非提前选择，且输出可直接缓存跨 target 复用。
3. **Cache-first 训练与推理系统设计**：训练侧 local KVCache（TTL 3h，命中率≈0.5）与推理侧共享缓存接口（TTL 1h，命中率≈0.6），配合 MRLB（训练复用比 R=40、推理复用比 R=300）、pipeline lift 与 FlashSA fused kernel，使 cache-hit 路径对原始超长序列长度呈 O(1)；与 KV/prefix caching 的本质区别在于缓存对象是固定大小 user-only sketch，缓存 footprint 不随 n 增长。

## 方法详解
- **Sketch Attention (SA)**：给定历史嵌入 X ∈ R^{n×d} 与 k 个可学习 prototype P ∈ R^{k×d}，计算 prototype–token 亲和度 S = (P W_Q)(X W_K)^T / √d，然后沿 **prototype 维度**做 softmax（而非 token 维度），得到分配矩阵 A ∈ R^{k×n}，其中 A_{j,i} = p(j|i) 表示第 i 个 token 向第 j 个 prototype 分配的质量；最终 sketch 为 X̃ = A X ∈ R^{k×d}。该操作是 target-agnostic 的、长度参数无关的（参数规模 O(kd+d²) 与 n 无关）、完全可微的。
- **Stacked Refinement**：使用 N_sa=2 层 SA 迭代细化，每层含 residual connection + LayerNorm + SwiGLU FFN，提升压缩容量。
- **双时间尺度 STCA 推理**：① **近期分支**：对最近 L_r=10K 行为直接使用 STCA（single-query cross-attention），捕获 recency-sensitive 信号；② **超长期分支**：先将 SA sketch 经宽度适配器 W_A 映射至 STCA 宽度 D=1024，再对固定长度 k 的 sketch 做 STCA，成本与原始 n 无关；③ 两分支经轻量级 fusion（生产中使用 MixFormer）送入下游 ranker。
- **训练侧 Local KVCache**：以 user_id + model_version + history_timestamp + sketch_config 为 key 缓存 X̃_u，TTL=3h，capacity eviction；cache miss 时仅额外执行一次 user-only SA 计算。
- **Multi-Request Level Batching (MRLB)**：将同一用户短时间窗口内的多个 request 分组，SA sketch 只需计算一次复用 R 个 target；request-specific 的 recent suffix 与 candidate 特征仍 per-request 计算。
- **Pipeline Lift**：由于 X̃_u 仅依赖 user-side 信号，可将 SA 计算上移至请求入口或上游阶段，与 recall 并行；cache hit 时恒定时间查找即可供给 sketch，cache miss 时并行执行不阻塞主链路。
- **FlashSA Kernel**：针对 ragged batching 下的 SA 计算设计 fused kernel，分块流式计算 affinity、prototype-wise softmax（max/log-sum-exp）与最终聚合，避免显式 materialize k×n 矩阵，降低峰值显存并提升吞吐。

## 实验与结果
- **数据集与设置**：抖音全量生产数据；离线评估含轻量 ablation 设置（baseline: STCA(512)）与完整生产设置（baseline: STCA 10K + lifelong TWIN V2）；在线 A/B 持续一个月，覆盖抖音与抖音极速版。
- **SA vs Vanilla Attention**：prototype-wise normalization 在多任务 UAUC 上全面优于 token-wise（Table 1）：like +0.10%，follow +0.20%，clicmmt +0.09%，comment +0.19%，share +0.14%，finish +0.01%。
- **SA 架构消融**（Table 2）：最终配置 k=1K, N_sa=2, d=128，较 STCA(512) baseline 获 Finish AUC **+1.07%**；去掉 action-side SwiGLU fusion 损失 -0.20%，降维 d=64 损失 -0.10%。
- **压缩方法对比**（Table 3）：SA (+1.07%) 优于 TWIN V2 KMeans (+0.30%)、chunk mean pooling (+0.46%)、position-bucket queries (+0.72%)、Lightning-Attention comp. (+0.83%)、recent-behavior query init. (+0.78%)。
- **完整生产离线评估**（Table 4）：以 STCA 10K + TWIN V2 为 baseline，SequenceO1 移除 TWIN V2 后仍全指标提升：Finish ΔAUC +0.29%、ΔUAUC +0.40%；Dislike ΔAUC +1.29%、ΔUAUC +3.63%。
- **在线 A/B**（Table 5）：抖音整体 Finish +2.33%、Duration +1.50%、Activeness +0.20%；抖音极速版 Finish +3.49%、Duration +1.67%；各活跃度分层均显著正向。
- **计算效率**（Figure 1）：100K 下 SequenceO1 训练 FLOPs 较直接 STCA 低 **49.9×**，推理 FLOPs 低 **63.9×**；100K truncation（平均 85K）下保留直接缩放 STCA 增益的 83%（+1.07% vs +1.29%）。

## 相关工作脉络
1. **TWIN/TWIN V2（快手）**：离线层次聚类压缩终身行为→在线检索目标相关子集→cluster-aware target attention；SequenceO1 定位差异在于端到端联合优化，移除两阶段分离，用可缓存 sketch 替代离线聚类+在线检索管线。
2. **STCA（抖音前作）**：train-sparsely/infer-densely，单 query cross-attention 将复杂度降至 O(L)；局限在于推理成本仍随原始历史长度线性增长且 query-dependent，无法跨 target 复用，本文通过 SA 压缩解决此问题。
3. **HSTU/ULTRA-HSTU（Meta）**：随机长度采样+长度外推；外推范围仍与训练长度耦合（训练~4.4K 外推至 16K），100K 外推仍需大量训练上下文，本文方案不依赖外推而是直接压缩建模。
4. **LLaTTE/SOLARIS**：异步上游 user model 生成表示→传输至在线 ranker；存在表征瓶颈、目标错位与新鲜度约束，且 transfer efficiency 仅 42%–53%；SequenceO1 保持 ultra-long 路径与 final ranker 联合优化，无需独立上游模型。
5. **Linformer/Set Transformer/Perceiver**：低秩注意力近似或 inducing points 压缩长序列；差异在于这些方法多面向 self-attention 近似或 target-aware 选择，而 SA 设计目标是 target-agnostic、可缓存、长度参数无关的用户侧 sketch，归一化轴改为 prototype 维度以保证全量覆盖。
6. **FlashAttention（Dao et al.）**：IO-aware attention kernel；本文 FlashSA 受其启发但针对 SA 的 prototype-wise softmax 与 ragged batching 定制，避免 materialize k×n 中间矩阵。

## 局限性与未来方向
- **Sketch 容量固定**：k=1K 在 100K 场景有效，但面对 million-scale 历史是否需要分层/多分辨率 sketch 尚未验证。
- **Prototype 覆盖假设**：prototype-wise normalization 假设每个 token 都能分配到某 prototype，但在极端稀疏或噪声历史下可能存在信息丢失风险。
- **缓存一致性与新鲜度**：TTL 机制依赖用户行为变化缓慢的假设，对高频兴趣突变的 user 可能引入 staleness。
- **论文自述未来方向**：分层或多分辨率 sketch、跨用户/行为模式的自适应容量、sketch-to-behavior 检索接口、localized refinement 等。

## 研究启发与可借鉴点
1. **归一化轴设计对复用性的影响**：prototype-wise vs token-wise softmax 的选择直接影响压缩表示是"选择子集"还是"全量覆盖"，这一设计原则可迁移至其他需要 target-agnostic 压缩的场景（如用户画像构建、长文档摘要）。
2. **Cache-first 系统思维**：将"可缓存性"作为模型设计的 first-class constraint（而非事后优化），使 sketch 成为 user-only 的固定大小状态，这一思路适用于任何涉及重复计算 user-side 特征的推荐/广告系统。
3. **Fused kernel 针对非标准 attention 模式**：FlashSA 对 prototype-wise softmax + ragged batching 的定制优化，展示了如何将 IO-aware 思想扩展到非标准 attention 变体，对工业界定制化 kernel 开发有参考价值。
4. **双时间尺度解耦**：recent suffix（10K，精确 recency）+ ultra-long sketch（固定 k，稳定偏好）的分层推理模式，可推广至其他需要兼顾短期动态与长期稳定的序列建模任务。
5. **FLOP 分析透明化**：附录 G 给出完整的 GEMM-dominant FLOP 公式与 hit-rate 敏感性推导，为后续 work 提供可直接复用的复杂度建模模板。

## 关键术语表
- **Sketch Attention (SA)**：一种将超长序列压缩为固定大小 prototype-based sketch 的 attention 变体，通过 prototype-wise softmax 实现 token-to-prototype 质量分配，输出 target-agnostic 且长度参数无关的用户侧表示。
- **STCA（Stacked Target-to-History Cross Attention）**：单层单 query 的 target-history cross attention 堆叠结构，移除 history–history attention 使复杂度从 O(L²) 降至 O(L)，是抖音生产中长序列建模的核心模块。
- **MRLB（Multi-Request Level Batching）**：将同一用户短时间窗口内的多个 request 归并为一组，使 user-only 的 SA sketch 计算仅执行一次并复用 R 个 target，实现跨请求的计算摊销。
- **Local KVCache**：训练侧维护的以 user_id 等为 key 的本地缓存，存储 SA 生成的固定大小 sketch，TTL=3h 内命中则跳过 100K 特征读取与 sketch 计算。
- **FlashSA**：针对 SA 计算的 fused GPU kernel，分块流式完成 affinity 计算、prototype-wise softmax 统计与聚合，避免 materialize k×n 中间矩阵，适配 ragged batching。
- **Prototype-wise normalization**：SA 中沿 prototype 维度对每个 token 做 softmax，使每个 token 的质量分布到所有 prototype（而非每个 prototype 独立选择 token），保证超长期压缩的全量历史覆盖。
- **Pipeline lift**：将仅依赖 user-side 信号的 SA sketch 计算从细排阶段上移至请求入口或上游并行执行，cache hit 时恒定时间查找，cache miss 时不阻塞主链路。
- **Low-rank caching**：指将超长序列沿 length 维度压缩为固定大小 representation 并缓存复用的范式，区别于传统 KV caching（per-position 状态随长度线性增长）。

## 可复现要素
- **数据集**：抖音生产数据（内部数据集，未公开）；离线评估使用全量生产日志。
- **代码/权重**：论文未提及开源；模型在抖音全流量部署（私有系统）。
- **关键超参**：SA sketch 长度 k=1024（ablation 中 512–2K  Tested）、SA 宽度 d=128、SA 层数 N_sa=2、STCA 头数 h=16、head dim d_h=64、STCA 宽度 D=1024、STCA 层数 N_stca=4、recent suffix 长度 L_r=10K、训练 MRLB 复用比 R_train=40、推理复用比 R_infer=300、训练缓存 TTL=3h（命中率≈0.5）、推理缓存 TTL=1h（命中率≈0.6）。
