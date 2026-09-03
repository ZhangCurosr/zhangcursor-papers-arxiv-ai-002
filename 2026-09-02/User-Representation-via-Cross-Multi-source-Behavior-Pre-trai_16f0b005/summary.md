---
title: "User-Representation-via-Cross-Multi-source-Behavior-Pre-trai"
source: https://arxiv.org/pdf/2609.01057v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-03 00:53:55"
field: "用户表征预训练与多源行为建模"
keywords: ["User Representation", "Pre-training", "Multi-source Behavior", "Mobile Game Recommendation", "Cross-source Modeling", "Cascaded Prediction"]
innovations: ["跨源级联预测任务体系（来源→类别→ID→操作）逐步细化用户意图", "GASA+CGFA双注意力机制显式处理volume-granularity mismatch", "设备级多源异构行为预训练框架CM-PTM统一建模跨源关联与源内时序"]
benchmarks: ["OPPO工业游戏推荐数据集（6个任务G1-G6）", "AUC", "R@P_0.95"]
---

# 论文速读：User-Representation-via-Cross-Multi-source-Behavior-Pre-training

## 一句话总结
提出CM-PTM（Cross Multi-source Behavior Pre-Training Model），一种面向移动设备级别多源异构行为日志的用户表征预训练框架，通过层级级联的mask-then-predict代理任务统一建模跨源意图关联与细粒度行为动态，显著提升下游移动游戏推荐性能。

## 研究问题与动机
- **多源异构行为的结构复杂性被忽视**：现有用户表征预训练方法集中于单App或单一行为源建模，而移动设备上用户意图分散于系统级应用（App Store/Game Center，有细粒度行为）与第三方应用（仅能观测到安装/启动/卸载等粗粒度事件）等多源异构行为流中，直接聚合会破坏源内时序连续性。
- **行为量与粒度的严重不对称**：第三方应用行为量大但粒度粗糙，系统应用粒度细但行为量少，这种volume-granularity mismatch导致传统Transformer对token一视同仁的建模方式失效。
- **跨源意图交织难以捕捉**：用户真实意图（如一款游戏的下载决策）常跨越多个源（广告曝光→App Store浏览→浏览器搜索→最终未下载），单序列预训练无法显式建模此类跨源关联。
- **判别式增强易破坏跨源语义**：现有判别式预训练依赖mask/裁剪/替换等数据增强，可能将"App Store搜索"错误替换为"GameApp启动"，破坏跨源相关性，导致意图表征退化。

## 核心贡献（创新点）
1. **首个面向移动设备级多源多粒度行为日志的用户表征预训练框架CM-PTM**：与现有方法聚焦单序列或单源行为的本质区别在于，CM-PTM将设备视为行为数据的基本组织单元，显式建模多源交互与粒度不对称性。
2. **粒度感知自注意力（GASA）与跨粒度融合注意力（CGFA）的协同架构设计**：与标准Transformer统一处理所有token的本质区别在于，GASA捕获同粒度源内行为连续相关性，CGFA捕获跨源宏微观意图关联，二者拼接形成源特定+跨源关联双重表征。
3. **跨源级联预测（Inter/Intra-Source Cascaded Prediction）代理任务体系**：将传统NBP拆解为"预测下一行为来源→预测App类别→预测App ID→预测具体操作"四层级联决策链，与并行多任务学习的本质区别在于每层预测条件化于前序高层意图状态，逐步缩小假设空间并缓解跨源歧义噪声。
4. **离线大规模实证+在线A/B测试双验证**：在OPPO 194万用户、8.3万App的真实工业数据集上验证，下游游戏推荐任务AUC提升1.05%~2.85%，在线A/B对比DeepFM提升1.28%/1.07%。

## 方法详解
**整体架构**：由"多粒度用户编码器（Multi-Granularity User Model）"和"跨源级联预测模块（Cross-Source Cascaded Prediction）"两部分组成，遵循两阶段范式——Stage 1自监督预训练学习通用用户表征，Stage 2表征冻结/微调后接入LightGBM进行下游游戏推荐。

**1）行为嵌入（Behavior Embedding）**：
- 收集三类来源行为：第三方GameApps（Install/Uninstall/Launch）、App Store（Search/Click）、Game Center（Search/Click）。
- 每源行为序列经Embedding后补零或截断至长度l，加位置编码$P^k$。

**2）粒度感知自注意力GASA（Granularity-Aware Self-Attention）**：
- 对源$k$的行为嵌入$E_u^k$计算多头自注意力，捕获同粒度源内行为连续相关性，输出$g_u^k$。

**3）跨粒度融合注意力CGFA（Cross-Granularity Fusion Attention）**：
- 以源1的行为为Query，源2+3的拼接行为为Key/Value，计算多头交叉注意力；对称地对各源分别计算，输出$c_u^k$，捕获跨源意图关联。

**4）用户表征拼接**：
- 预训练阶段：每源独立使用$\boldsymbol{u}^k = \text{Concat}(c_u^k, g_u^k)$完成各自代理任务，避免异源噪声干扰。
- 下游阶段：全量拼接$\boldsymbol{u} = \text{Concat}(c_u^1, c_u^2, c_u^3, g_u^1, g_u^2, g_u^3)$，最大化表征信息量。

**5）跨源级联预测模块**：
- **Inter-Source阶段（$T_1$）**：用共享源预测塔$f_{src}$对每源表征输出logits $r_1^k$，经softmax得$\hat{p}_{src}^k$；同时用Info模块$f_{info}^1$生成上下文向量$z_1^k$作为先验知识传递给下一层。标签取自全局时序排布中被屏蔽的下一条行为所属源。
- **Intra-Source阶段（$T_2, T_3, T_4$）**：
  - $T_2$：预测下一App的Category（类别）。
  - $T_3$：预测具体App ID，条件化于$T_2$输出的上下文$z_1^k$。
  - $T_4$：预测操作类型（Install/Launch/Uninstall），仅对第三方GameApps有效（App Store/Game Center仅含Search/Click，语义区分度低故跳过）。
  - 每层$T_i$将$T_{i-1}$产生的$z_{i-1}^k$通过$f_{attn}^i$（自注意力融合）与$r_i^k$融合，经softmax输出$\hat{p}_i^k$；$T_2, T_3$进一步产生$z_i^k$供后续使用。
- 所有$f_{src}, f_{info}^i, f_T^i$均为2层MLP（ReLU激活），$f_{attn}^i$为自注意力。

**6）损失函数**：
- $\mathcal{L}_{INTER} = -\sum_{k=1}^3 \log(\hat{p}_{src, s_u}^k)$
- $\mathcal{L}_{INTRA} = -\sum_{k=1}^3 \sum_{T \in \mathcal{A}_k} \sum_{i=0}^{N(T)-1} (y_T^k=i)\log(\hat{p}_T^k)$
- $\mathcal{L} = \lambda \mathcal{L}_{INTER} + (1-\lambda)\mathcal{L}_{INTRA}$，$\lambda=1/2$

## 实验与结果
- **预训练数据集**：OPPO设备日志，1947683用户、83341 App，3源行为：GameApp操作均次144.93、App Store搜索均次22.60、App Store点击均次62.21、Game Center搜索均次1.12、Game Center点击均次3.84。
- **下游任务**：6个移动游戏推荐任务$G_1 \sim G_6$，涵盖"拉新吸引老游戏"（$G_1$: Mini World, $G_2$: Eggy Party, $G_3$: Sky）、"召回老用户玩老游戏"（$G_4$: Honor of Kings）、"拉新吸引新游戏"（$G_5$: Crystal of Atlan, $G_6$: Journey to West）。
- **评估指标**：AUC、R@P$_{0.95}$。
- **基线**：PeterRec、PTUM、BERT4Rec（生成式）；CLUE、CCL、AdaptSSR（判别式）；MAFN（混合式）。不支持多源的基线将多源行为按时间合并为单序列。
- **主要结果**：CM-PTM在$G_1 \sim G_5$上AUC均显著优于所有基线，提升幅度+1.05%~+2.85%（如$G_1$: 0.8542 vs BERT4Rec 0.8370，+2.15%绝对值；$G_2$: 0.8411 vs 0.8109，+3.02%绝对值）。$G_6$ AUC略低于Best baselines（-0.08%），但R@P$_{0.95}$保持领先（0.9288），对新游戏推荐场景更具实际价值。
- **消融**：移除粗粒度数据AUC降0.57~2.28pp；移除细粒度降2.28~2.64pp；单序列建模降1.52~1.76pp；移除$T_1$（跨源预测）对老游戏降0.52pp、对新游戏降1.87pp；级联结构替换为并行预测降2.35~2.83pp。
- **在线A/B测试**：OPPO真实流量289万用户，CM-PTM vs DeepFM：AUC +1.28%（0.8192 vs 0.8064），R@P$_{0.95}$ +1.07%（0.3656 vs 0.3549）。
- **效率**：预训练耗时5.74h（双A100），介于PeterRec（4.32h）与MAFN（6.53h）之间；在线推理仅增加$d_u=384$维特征，开销可忽略。

## 相关工作脉络
- **PeterRec [10] / PTUM [11] / BERT4Rec [17]**：生成式单序列行为预训练，通过Masked Behavior Prediction或Next Behavior Prediction学习用户表征；CM-PTM的区别在于引入多源结构约束和跨源级联预测，而非简单将多源行为拼成单序列。
- **CLUE [12] / CCL [16] / AdaptSSR [15]**：判别式对比学习预训练，通过数据增强构造正负对；CM-PTM认为此类方法可能破坏跨源语义（如将App Store搜索替换为GameApp启动），而生成式级联预测更保真。
- **MAFN [29]**：结合生成+判别学习App多类型行为（单App内）；CM-PTM进一步将建模单元从App内行为扩展到设备级跨App/跨源行为。
- **多行为建模（[35]-[45]）**：通常针对单一来源的多类型行为（如click/buy/cart）；CM-PTM处理的是异构来源（系统App vs 第三方游戏）且粒度不对等的行为。
- **跨域推荐（CDR）方法（[46]-[53]）**：聚焦不同领域间的知识迁移；CM-PTM聚焦同一设备上的跨源行为关联，不涉及领域适配或差分隐私等CDR特有机制。

## 局限性与未来方向
- **时间感知机制缺失**：当前仅用位置编码捕捉短窗口内相对顺序，未显式建模绝对时间间隔（如用户上次访问App Store与本次启动GameApp之间的天数），作者建议作为未来方向。
- **行为源数量受限**：当前仅建模3个行为源（GameApps/App Store/Game Center），未覆盖社交媒体、短视频等其他常见行为渠道（如图1中提到的TikTok广告曝光未被捕获）。
- **Task $T_4$覆盖不全**：App Store和Game Center跳过操作预测仅保留类别和App ID预测，可能损失搜索vs点击的细粒度意图信号。
- **下游任务单一**：虽提及用户画像、欺诈检测等通用下游，但实验仅聚焦游戏推荐，未验证在其他领域（如电商、内容）的泛化性。
- **推理时的硬标签传递问题**：当前级联使用可微上下文传递而非硬标签，但实际部署时仍需将隐式预测转换为可解释决策链路。

## 研究启发与可借鉴点
- **多源行为建模的"先宏观后微观"设计哲学**：将复杂意图分解为层级决策链（来源→类别→ID→操作），每一层条件化于前序高层状态，可有效降低假设空间并缓解标签歧义——该思路可迁移至任何多源异构行为场景（如电商+内容+社交跨域）。
- **粒度不对称融合的结构化设计**：GASA+CGFA双注意力机制明确区分"源内细粒度时序"与"跨源粗粒度关联"，为volume-granularity mismatch问题提供了显式架构解决方案，优于简单拼接或平均池化。
- **生成式预训练在跨源场景的天然优势**：判别式数据增强（mask/crop/substitution）容易跨源破坏语义一致性，而生成式mask-then-predict在预测层面保持源结构不变——这一观察对选择下游预训练范式具有指导价值。
- **工业级离线+在线双验证闭环**：从194万用户预训练→6个游戏场景离线评测→289万用户A/B测试的完整链路，为团队搭建端到端验证体系提供参考。
- **轻量化部署友好**：预训练产出仅增加384维特征，下游LightGBM即可承接，推理开销几乎为零，适合资源受限的移动端场景。

## 关键术语表
**CM-PTM**：Cross Multi-source Behavior Pre-Training Model的缩写，本文提出的面向移动设备级多源多粒度行为日志的用户表征预训练模型。
**GASA（Granularity-Aware Self-Attention）**：粒度感知自注意力机制，用于捕获同一行为源内部的细粒度时序行为相关性。
**CGFA（Cross-Granularity Fusion Attention）**：跨粒度融合注意力机制，通过交叉注意力捕获不同行为源之间的宏微观意图关联。
**Inter-Source Phase ($T_1$)**：跨源预测阶段，预测用户下一条行为来自哪个行为源（GameApps/App Store/Game Center）。
**Intra-Source Phase ($T_2/T_3/T_4$)**：源内预测阶段，在确定来源后依次预测App类别、App ID和具体操作类型。
**Cascaded Prediction**：级联预测，每层预测条件化于前序层产生的可微上下文向量$z_i^k$，而非硬标签，逐步细化用户意图。
**R@P$_{N}$**：在精度不低于$N$的约束下所能达到的最大召回率，本文设$N=0.95$，适合评估高精准要求场景（如新用户拉新）。
**Volume-Granularity Mismatch**：行为数据量与粒度之间的不对等现象，第三方应用行为量大但仅含粗粒度事件，系统应用粒度细但行为量少。

## 可复现要素
- **数据集**：OPPO设备级行为日志，预训练数据2023/07/13~07/19（1947683用户，83341 App）；下游任务$G_1 \sim G_6$训练集采样2023/07/20~07/25，测试集采样2023/07/23~07/28及之后一周。数据集为OPPO内部工业数据，未公开。
- **代码/权重**：论文未声明开源代码或预训练权重。
- **关键超参**：嵌入维度$d=64$，用户表征维度$d_u=384$（6×64），序列最大长度$l=256$，注意力头数$H=2$，批次大小512，学习率1e-4，$\lambda=1/2$，预训练训练/验证集比例9:1，优化器Adam。
