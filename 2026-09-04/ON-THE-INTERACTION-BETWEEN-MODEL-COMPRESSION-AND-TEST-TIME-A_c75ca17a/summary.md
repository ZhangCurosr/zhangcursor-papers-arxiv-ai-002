---
title: "ON-THE-INTERACTION-BETWEEN-MODEL-COMPRESSION-AND-TEST-TIME-A"
source: https://arxiv.org/pdf/2609.03604v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 12:04:41"
field: "模型压缩与测试时适应交互"
keywords: ["model compression", "test-time adaptation", "structured pruning", "representational expressivity", "gradient alignment", "plasticity loss"]
innovations: ["提出首个压缩-TTA交互诊断框架，从表征表达性与子空间兼容性双维度量化压缩对适应能力的破坏", "揭示静默可塑性丧失现象并建立梯度退化/主动发散两种故障模式的理论机制", "区分压缩效应与容量效应，证明可适应性损失源于结构化压缩而非单纯参数减少"]
benchmarks: ["CIFAR-10-C", "ImageNet-C"]
---

# 论文速读：ON-THE-INTERACTION-BETWEEN-MODEL-COMPRESSION-AND-TEST-TIME-ADAPTATION

## 一句话总结
本文系统分析了结构化模型压缩与测试时适应（TTA）之间的交互作用，揭示了压缩模型在保留源域精度的同时会出现"静默可塑性丧失"（silent plasticity loss），即TTA性能随压缩程度加剧而显著退化；通过表征表达性与子空间兼容性两个维度的诊断框架，阐明了熵最小化与一致性优化目标在压缩下均存在梯度退化机制。

## 研究问题与动机
- **核心问题**：结构化压缩是否会削弱模型在分布偏移下的测试时适应能力？是否存在能保持可适应性的压缩策略？
- **现有不足一**：压缩与模型适应长期被独立研究，两者交互机制缺乏系统性理解（Liang et al., 2025; Choudhary et al., 2020）。
- **现有不足二**：现有TTA研究几乎均假设模型为稠密状态，忽略了边缘部署中模型必须压缩的现实约束（Mirza et al., 2022; Wang et al., 2021）。
- **现有不足三**：已有涉及压缩与适应交互的工作仅局限于量化方法（Xiao et al., 2024; Deng et al., 2025），结构化剪枝与折叠的表征几何影响尚未阐明。
- **动机延伸**：边缘设备（如Raspberry Pi Zero）因内存限制难以运行梯度计算，压缩不仅是推理效率需求，更是启用设备端适应的前提条件。

## 核心贡献（创新点）
- **提出首个压缩-TTA交互诊断框架**：从"表征表达性"（CKA对齐度、AME熵）与"子空间兼容性"（梯度余弦相似度）两个互补维度量化压缩对适应能力的破坏，区别于仅评估源域精度的传统压缩基准。
- **揭示"静默可塑性丧失"现象**：发现压缩模型在监督微调（Oracle）下仍保持高准确率，但TTA性能随压缩度加剧显著下降，且该差距在ImageNet-C上比CIFAR-10-C更为严重。
- **建立梯度退化的理论机制**：从理论上证明熵最小化目标（Eq. 4-5）与KL一致性目标（Eq. 6）在压缩导致预测趋近均匀分布时均发生梯度退化；同时定义了"主动发散"模式——TTA梯度不仅信号弱化，还反向偏离监督优化方向。
- **方法选择影响显著**：Wanda和Taylor能较好保留可恢复的表征结构，OBD和Mag-ℓ₂退化严重；Folding作为无校准数据方法在较高稀疏度下仍保持相对稳定恢复能力。
- **区分压缩效应与容量效应**：通过与同参数规模从零训练的较小稠密模型对比（Fig. 19），证明可适应性损失源于结构化压缩的表征破坏而非单纯模型尺寸减小。

## 方法详解
- **结构化压缩形式化**：将压缩建模为投影操作 $\Pi: \mathbb{R}^D \to \mathbb{R}^d$（$d \ll D$），涵盖两类方法：（1）结构化剪枝：坐标选择算子，对通道或注意力头施加二元掩码；（2）模型折叠：基于k-means聚类的通道聚合，将相似通道替换为质心。
- **适配子空间定义**：TTA更新 $\theta^* = \theta_0 + \Delta$，其中 $\Delta \in \mathcal{U}$ 为允许的更新方向集合（如归一化层参数），Oracle使用相同子空间但基于有标签数据。
- **表征表达性度量一（CKA）**：使用Centered Kernel Alignment衡量压缩模型与稠密模型在残差后隐藏表示的对齐度，取最退化层的CKA值作为代表统计量。
- **表征表达性度量二（AME）**：定义Activation Map Entropy为激活Gram矩阵归一化特征值的Shannon熵，量化表征谱多样性；压缩后 $G_l$ 从 $C_l \times C_l$ 缩减至 $C_l' \times C_l'$，熵上限被限制为 $\log C_l'$。
- **子空间兼容性度量**：计算TTA梯度与Oracle梯度在适配子空间 $\mathcal{U}$ 上的余弦相似度（Eq. 2）：$\cos(\mathbf{g}_{\text{TTA}}, \mathbf{g}_{\text{Oracle}}) = \frac{\mathbf{g}_{\text{TTA}} \cdot \mathbf{g}_{\text{Oracle}}}{\|\mathbf{g}_{\text{TTA}}\| \cdot \|\mathbf{g}_{\text{Oracle}}\|}$。
- **熵最小化梯度退化证明**：对 $\mathcal{L}_{\text{ent}} = -\sum_c p_c \log p_c$ 求导得 $\nabla_\phi \mathcal{L}_{\text{ent}} = -\sum_c \log(Cp_c) \nabla_\phi p_c$；当预测趋近均匀分布 $p_c = 1/C$ 时所有系数消失，梯度退化为零向量；梯度范数上界为 $\|\nabla_\phi \mathcal{L}_{\text{ent}}\|_2 \leq (\sum_c |\log(Cp_c)|) \max_c \|\nabla_\phi p_c\|_2$。
- **KL一致性梯度退化**：SPA目标 $\mathcal{L}_{\text{KL}} = \sum_c q_c \log(q_c/p_{\text{aug},c})$ 的梯度为 $-\sum_c (q_c/p_{\text{aug},c}) \nabla_\phi p_{\text{aug},c}$；当干净视图与增强视图预测均趋近均匀时各比值趋近1且梯度求和为零。
- **监督交叉熵梯度保持**：$\nabla_\phi \mathcal{L}_{\text{CE}} = -(1/p_y) \nabla_\phi p_y$，即使在低置信度下梯度仍指向单一方向且被 $1/p_y$ 放大，不会退化。
- **两种故障模式**：（1）梯度退化性：预测趋近均匀导致TTA信号减弱；（2）主动发散：压缩保留"高置信错误预测"时，大范数梯度与监督方向相反。

## 实验与结果
- **数据集**：CIFAR-10-C和ImageNet-C（severity level 5，15种腐败类型）；ResNet-18（11.1M参数）和ViT-Base（86M参数）在对应源数据集预训练。
- **压缩方法**：Mag-ℓ₂剪枝、Wanda（权重×激活）、Taylor剪枝、OBD（Hessian）、Folding（k-means通道聚类）；共5种结构化压缩策略。
- **TTA方法**：ResNet-18使用SAR（熵最小化+锐度感知），ViT-Base使用SPA（一致性目标）；Oracle变体使用相同优化设置仅替换为监督交叉熵。
- **核心发现一**：Oracle-TTA差距随压缩度扩大（Fig. 1），CIFAR-10-C上梯度对齐在45%-65%稀疏度间发生符号反转，ImageNet-C上熵目标在最小稀疏度即出现主动发散，而一致性目标在全部测试稀疏度保持正对齐。
- **核心发现二**：TTA在低稀疏度即开始破坏表征（ResNet-18/ImageNet-C上2%即出现），且无法恢复；Oracle持续恢复表征至稠密模型水平（Fig. 2）。
- **核心发现三**：AME分析显示压缩引入熵上限约束，Oracle能更有效重分配特征值质量，TTA则受限于退化信号无法充分恢复（Fig. 3）。
- **最强结果对比**：Wanda和Taylor在保持较高CTA可恢复性的同时维持较好的梯度对齐；Folding在无校准数据方法中表现最稳定，在ViT-Base 39%稀疏度下仍显著优于Mag-ℓ₂。
- **提升幅度**：在CIFAR-10-C上，排除SAR可靠性过滤器中的"高置信错误"样本后，梯度对齐恢复至≈+0.9（Fig. 18），揭示了主动发散的主要成因。

## 相关工作脉络
- **结构化剪枝与折叠**：Li et al. (2017) Magnitude剪枝、Sun et al. (2024) Wanda、Molchanov et al. (2019) Taylor、LeCun et al. (1989) OBD、Wang et al. (2025) Folding；本文定位差异在于不仅评估源域精度，更关注压缩对TTA可适应性的影响。
- **测试时适应**：TENT（Wang et al., 2021）、SAR（Niu et al., 2023）、EATA（Niu et al., 2022）、SPA（Niu et al., 2025）、PEA（Xiao et al., 2026b）、FOA（Niu et al., 2024）；本文填补了这些方法在压缩场景下的性能退化机制空白。
- **Loss Landscape与模式连通性**：Frankle et al. (2020)线性模式连通性、Saukh et al. (2026)折叠对loss landscape的影响；本文扩展了这一视角，关注压缩是否可通过适应补偿landscape变化。
- **压缩与鲁棒性**：Liebenwein et al. (2021)过参数化对分布偏移鲁棒性的必要性、Hooker et al. (2020)压缩对低频子群体的偏差放大；本文将观察扩展到可塑性（plasticity）而非仅鲁棒性。
- **量化-TTA交互**：Xiao et al. (2024) TTAQ、Deng et al. (2025)；本文是唯一系统分析结构化压缩（剪枝+折叠）与TTA交互的工作，区别于已有的量化研究。
- **塑性丧失**：Lyle et al. (2023)、Dohare et al. (2024)深度学习中的塑性丧失；本文首次在压缩语境下定义并实证了"静默塑性丧失"（压缩模型保留源域精度但丧失适应能力的现象）。

## 局限性与未来方向
- **架构与模型规模受限**：仅评估了ResNet-18和ViT-Base，未涉及更大规模模型（如LLM）的压缩-TTA交互。
- **TTA方法覆盖有限**：每个架构仅使用一种BP-based TTA方法（SAR/SPA），BP-free方法（PEA/FOA）仅在精度曲线层面报告（Fig. 9），其梯度诊断未深入分析。
- **压缩类型局限**：仅研究结构化压缩（剪枝+折叠），未涵盖非结构化剪枝、量化及其组合形式。
- **实验设置固定**：主要在severity level 5下评估，虽附有severity 3和multi-seed结果，但未探索跨领域持续适应场景。
- **未来方向**：（1）扩展至BP-free TTA方法的诊断分析；（2）设计压缩感知的TTA目标以在压缩下保持可适应性；（3）扩展到LLM等大模型场景。

## 研究启发与可借鉴点
- **压缩策略选择应优先考虑可适应性而非仅源域精度**：Wanda/Taylor保留的表征结构更易被TTA恢复，OBD/Mag-ℓ₂即使源域精度相近也会导致严重退化，建议将TTA恢复能力纳入压缩准则评估。
- **梯度对齐可作为压缩-TTA兼容性的诊断指标**：余弦相似度能揭示从信号弱化到主动发散的渐进退化过程，建议作为压缩方法选择的辅助评估工具。
- **压缩预算与TTA目标需联合设计**：Entropy-based方法在复杂分布偏移（ImageNet-C）下对压缩极度敏感，Consistency-based方法（SPA）在相同条件下保持更好对齐，建议根据部署场景的偏移复杂度匹配TTA目标。
- **AME与CKA的组合诊断框架可迁移至其他适配场景**：同时测量表征对齐度与谱多样性，辅以梯度分析，可全面刻画压缩对模型适应能力的多维影响。
- **控制变量设计值得借鉴**：通过与同参数规模从零训练的稠密模型对比，严格区分了结构化压缩效应与模型容量效应，这一实验设计可用于其他压缩方法的影响归因分析。

## 关键术语表
- **Silent Plasticity Loss（静默可塑性丧失）**：压缩模型在源域精度保持的同时，因表征结构破坏而丧失通过测试时信号进行适应的能力，表现为Oracle-TTA差距随压缩加剧而扩大。
- **CKA（Centered Kernel Alignment）**：衡量两个神经网络层表示之间相似性的核对齐度量，用于评估压缩模型与稠密模型表征的对齐程度。
- **AME（Activation Map Entropy）**：基于激活Gram矩阵归一化特征值的Shannon熵，量化网络层激活的谱多样性和有效秩。
- **Gradient Degeneracy（梯度退化）**：压缩导致模型预测趋近均匀分布，使熵最小化TTA目标的梯度系数 $\log(Cp_c)$ 趋近零，适配信号衰减。
- **Active Divergence（主动发散）**：压缩保留高置信错误预测时，TTA梯度具有大范数但方向与监督优化方向相反，主动损害模型性能。
- **Adaptation Subspace（适配子空间）**：TTA过程中允许更新的参数集合 $\mathcal{U}$（如BN/LayerNorm参数），压缩会缩减该空间的维度。
- **Structured Compression（结构化压缩）**：通过移除或合并完整计算单元（如卷积滤波器、注意力头）实现压缩的方法，包括剪枝和折叠。
- **Test-Time Adaptation（TTA）**：仅利用无标签测试数据，在推理阶段在线更新模型参数以适应分布偏移的范式。

## 可复现要素
- **数据集**：CIFAR-10-C、ImageNet-C（公开基准，Hendrycks & Dietterich, 2019）；ResNet-18和ViT-Base预训练权重来自标准实现。
- **代码/权重**：论文未明确声明代码开源状态（投稿时间为2025年，arxiv链接可能为预印本阶段）。
- **关键超参**：ResNet-18稀疏度 $r \in \{0.0, 0.01, \ldots, 0.95\}$（对应sparsity $s \in [0\%, 94.9\%]$）；ViT-Base稀疏度 $r \in \{0.0, 0.1, \ldots, 0.7\}$（对应sparsity $s \in [0\%, 45.8\%]$）； corruption severity level 5；batch size B=64；SAR/SPA使用原文超参。
- **硬件环境**：AMD EPYC 9654 96核处理器单线程评估延迟/FLOPs/内存。
- **随机种子**：每架构-数据集配置使用3个不同随机种子训练checkpoint，multi-seed结果见Appendix。
