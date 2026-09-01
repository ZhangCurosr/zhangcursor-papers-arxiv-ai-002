---
title: "Reading-the-News-Adapting-Large-Language-Models-to-Swedish-J"
source: https://arxiv.org/pdf/2608.30609v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 23:44:35"
field: "低资源语言领域适配"
keywords: ["continued pre-training", "Swedish NLP", "domain adaptation", "experience replay", "LoRA", "news journalism", "instruction following", "low-resource language"]
innovations: ["系统验证了经验回放混合策略对瑞典语新闻领域 CPT 的有效性，BCES 混合最优", "发现 Instruct Vector 仅与 LoRA 而非 FFT 兼容，揭示了 PEFT 方法与 IV 的适配规律", "构建首个瑞典语新闻领域六任务基准 BonEval，证明通用基准无法捕捉领域适配增益"]
benchmarks: ["BonEval", "EuroEval Swedish", "SwedishFacts", "SvD SEO Headline", "AB Summary"]
---

# 论文速读：Reading-the-News-Adapting-Large-Language-Models-to-Swedish-Journalism-Through-Continued-Pre-Training

## 一句话总结
本文针对瑞典语新闻这一低资源、细粒度领域，利用从 2170 万篇瑞典新闻文章中精心清洗的 BonCorpus（1460 万篇/85 亿 token），通过持续预训练（CPT）+ 经验回放的方式适配 Ministral 3 模型，并提出了首个覆盖六种编辑任务的瑞典语新闻领域评测基准 BonEval。结果表明 CPT 可显著提升生成质量与事实知识，但对判别性任务有害，且需配合经验回放才能避免灾难性遗忘；LoRA + Instruct Vector 组合效果最佳。

## 研究问题与动机
- **新闻领域作为 LLM 适配目标被严重忽视**：现有工作主要关注金融、法律、生物医学等垂直领域，而新闻业极少作为 CPT 适配目标被系统研究。
- **瑞典语在北欧语言中处于"被遗忘的角落"**：丹麦语（Zhang et al., 2025）、挪威语（Samuel et al., 2025）、冰岛语（Gogoulou et al., 2024）等均有系统性 CPT 研究，瑞典语却仅有有限方法细节的探索或小模型研究。
- **新闻语料质量高但噪声大**：跨刊转载导致大量近重复文章，格式迁移遗留超链接、HTML 残片等 artifacts，需要系统的清洗与去重管道。
- **缺乏针对瑞典语新闻领域的评测基准**：既有 EuroEval 等通用瑞典语基准无法捕捉领域内适配收益，特别是生成任务上的提升。

## 核心贡献（创新点）
- **构建了第一个面向瑞典语新闻的大规模专用预训练语料 BonCorpus**：从 Bonnier News 出版社 1991–2026 年的 2170 万篇文章中，经清洗去重后保留 1460 万篇（8.5B tokens），覆盖日报、晚报、财经、地方及生活方式等多元类型。*与从零预训练（如 GPT-SW3、Silo AI Viking）相比，本工作以较低计算成本实现了同类语料的精加工与适配。*
- **提出了首个六任务瑞典语新闻领域基准 BonEval**：覆盖 headline（标题生成）、lead（导语生成）、summary（要点摘要）、topic（话题分类）、entity（核心实体识别）、quiz（事实问答）三类任务，均从 BonCorpus 2022 年后文章采样并排除训练集。*区别于 EuroEval 等通用基准，本基准专为编辑工作流设计，能敏感反映 CPT 的领域增益。*
- **系统验证了经验回放（Experience Replay）组合对 CPT 的有效性**：以 10% 比例混合代码/英文/瑞典文通用数据，BCES 混合（BonCorpus 70% + 三者各 10%）取得最高 BonEval 分，纯 BonCorpus 因灾难性遗忘得分最低。*与简单全量领域数据训练相比，证明了回放数据构成是成功适配的关键。*
- **揭示了 Instruct Vector（IV）与 PEFT 方法的兼容性差异**：LoRA + IV 显著优于 FFT + IV；作者假设 LoRA 较小的参数漂移使模型更接近原始指令微调版本的空间。*这为"CPT + IV"的组合使用提供了方法选择依据。*
- **警示了通用基准的评估盲区**：BonLM 在 EuroEval 上整体退化，但在 BonEval 上显著提升，说明已有瑞典语通用基准无法可靠衡量领域适配的真实收益。

## 方法详解
- **BonCorpus 数据管道**：① 解析聚合 21.7M 文章，Unicode 归一化，正则去除外链/广告/作者署名；② 基于文本统计（行数、字数、type-token ratio ≥ 0.2）、fastText 语言识别（分 > 0.9）、元数据过滤低质/模板/新闻稿等；③ MinHash LSH（112 哈希函数，相似度阈值 0.25）进行模糊去重，Levenshtein 距离 < 0.2 判为重复，取每个连通分量最长文章，最终得到 14.6M 篇（8.5B tokens）。
- **经验回放混合策略**：以 Common Corpus 为回放源，采样 code、English、Swedish 三类通用数据，每类占混合体的 10%（最大混合 BCES 中 BonCorpus 占 70%），共 8 种组合。回溯法：每类单独 10% 即足以缓解遗忘。
- **PEFT 方法**：对比 LoRA（rank=256, α=512, dropout=0，应用于所有 transformer 模块）、LLaMA Pro（每第 6 层后插入一层，约 466M 可训练参数，与 LoRA 的 395M 相当）、Full Fine-Tuning（FFT）。
- **Instruct Vector（IV）构造**：将指令微调模型参数减去基础模型参数得到 instruct 向量，无额外训练，直接加到 CPT 适应后的模型上。LoRA + IV 在 3B 实验中 BonEval 平均分达 42.27，8B 达 46.02，为最佳。
- **训练配置**：最小 3B/最大 8B 的 Ministral 3 系列，Global batch size=64，sequence length=16384，cosine schedule + 10% warmup，max lr=5e-5，DeepSpeed ZeRO-2 + FlashAttention-3 + Liger kernels，3B 实验 8000 步，8B 实验 4 个 epoch 等价步数，总 GPU 时长约 6500 小时。

## 实验与结果
- **数据集**：BonCorpus（14.6M/8.5B tokens）、BonEval（6 任务；Headline/Lead/Topic/Entity 各 8192 条，Summary 100 条，Quiz 11960 条），补充测试用 Svenska Dagbladet SEO Headline、Aftonbladet Summary、SwedishFacts。
- **基线模型**：Ministral 3 8B Base/Instruct/Reasoning、GPT-SW3 6.7B、Llama SW3 8B、Apertus 8B。
- **BonEval 核心结果**：BonLM LoRA + IV 平均分 46.02（CHRF3++ 生成任务 + MCC 分类/问答），超越 Ministral Base（41.59）和所有开源基线；FFT（44.14）、LoRA（44.47）次之。
- **各任务趋势**：CPT 一致提升 Headline/Lead/Summary 生成质量与 Quiz 事实知识（Quiz 提升最显著，FFT: 60.12→68.86, LoRA+IV: 60.12→66.95）；Topic/Entity 等判别任务轻微退化。
- **EuroEval 结果**：BonLM 整体低于 Ministral Base，因 CPT 导致的灾难性遗忘，在 LLM 翻译版 MMLU/HellaSwag 上衰退尤其明显；仅在 SUC 3.0、ScaLA 等原生瑞典任务上持平或略优。
- **最强结果**：BonLM 8B LoRA + IV 在 BonEval 上平均 46.02（+4.43 vs Base），在 SvD SEO Headline（30.14）、AB Summary（47.18）、SwedishFacts（38.56）补充任务上均大幅领先。

## 相关工作脉络
- **BloombergGPT / MediaGPT / NewsGPT**：针对金融或中文新闻的从零预训练，需海量算力，本工作证明 CPT 可作为低资源场景的高效替代路径。
- **GPT-SW3 / Llama SW3**：均从零或粗粒度 CPT 构建瑞典语基础模型，但缺少新闻领域的深度适配与针对性评测；本工作在此基础上证明领域 CPT 仍有增量价值。
- **丹麦语/挪威语/冰岛语 CPT 研究**：本工作与这些北欧语言适配路径形成横向对照，证明瑞典语（Scandinavian 最大语种）同样适用该范式。
- **经验回放在 CPT 中的应用**：Chaudhry et al. (2019)、Scialom et al. (2022) 等提出回放缓解遗忘，本文进一步量化了不同回放类型（code/EN/SV）对新闻领域任务的具体增益模式。
- **LoRA 与 Instruct Vector 结合**：Biderman et al. (2024) 证明 LoRA 减少遗忘，Huang et al. (2024) 提出 instruct vector；本文首次系统验证二者在 CPT 场景下的组合兼容性问题，发现 IV 仅与 LoRA 而非 FFT 兼容。
- **EuroEval / Superlim 等瑞典语通用基准**：本文通过对比实验揭示这些基准在衡量领域适配增益上的局限性，呼应 Chen et al. (2024)、Kuulmets & Fishel (2023) 关于翻译基准文化偏差的论点。

## 局限性与未来方向
- **数据与代码不公开**：因商业与法律原因，BonCorpus、BonLM 模型与代码均不可复现，降低研究可重复性。
- **单模型家族限制泛化**：仅基于 Ministral 3 系列实验，其他模型家族（如 Llama/Qwen）可能呈现不同结论。
- **原始预训练数据未知**：无法评估 BonCorpus/BonEval 与基础模型预训练的潜在重叠及其影响。
- **缺少下游任务微调**：未将 BonLM 针对具体编辑用例（如标题生成、文风修正）进行 further fine-tuning，限制了实际效用验证。
- **仅依赖自动评测**：无人工评估，无法断言模型在真实新闻编辑室中的实用价值。
- **未来方向**：下游编辑任务微调、与记者合作设计人工评估协议、探索多模型家族的泛化验证。

## 研究启发与可借鉴点
- **经验回放的混合比例设计可直接迁移**：采用 10% 比例的多类型通用数据（code/EN/SV）混合用于语言/领域适配，是一种低成本防遗忘的有效策略，可推广至其他低资源语言或垂直领域。
- **LoRA + Instruct Vector 组合值得复用**：在 CPT 后恢复指令遵循能力时，LoRA 适配后再叠加 IV 比直接 FFT + IV 稳定得多；这对"领域适应 + 对话接口"的工程管线有参考价值。
- **构建领域专属评测基准的必要性**：EuroEval 等通用基准会掩盖 CPT 的领域增益，建议在适配研究中同步构建或选用针对目标领域的专用基准，以便准确捕捉生成与事实类任务提升。
- **去重与清洗管道可复用**：MinHash LSH + Levenshtein 连通分量去重的流程，适用于任何包含大量近重复内容的开放域语料（如社交媒体、专利、政府文档）。
- **长序列打包策略**：best-fit-decreasing packing + 压平消除 padding，适合处理长度差异极大的文章类数据，可在类似任务中借鉴。

## 关键术语表
**Continued Pre-training（CPT）**：在已有预训练模型基础上继续使用目标领域语料进行自监督训练，以扩展模型在该领域的能力。
**Experience Replay（经验回放）**：在 CPT 训练数据中混入少量原始预训练分布样本（如代码、通用文本），以缓解灾难性遗忘。
**LoRA（Low-Rank Adaptation）**：参数高效微调方法，通过冻结原权重并在注意力模块中注入低秩分解矩阵进行训练，限制参数更新幅度以减少遗忘。
**Instruct Vector（IV）**：从指令微调模型参数中减去基础模型参数得到的差值向量，直接叠加到 CPT 模型上以恢复指令遵循能力，无需额外训练。
**Catastrophic Forgetting（灾难性遗忘）**：模型在继续学习新领域数据时，原有通用能力显著退化的现象。
**PEFT（Parameter-Efficient Fine-Tuning）**：仅更新模型一小部分参数的适配技术，包括 LoRA、LLaMA Pro 等。
**BonCorpus**：本文构建的瑞典语新闻预训练语料，经清洗去重后含 1460 万篇文章、85 亿 token。
**BonEval**：本文提出的瑞典语新闻领域评测基准，包含标题生成、导语生成、要点摘要、话题分类、实体识别、知识问答共 6 项任务。

## 可复现要素
- **数据集**：BonCorpus（未公开，因商业/法律原因）；BonEval（未公开）
- **代码**：论文未开源
- **模型权重**：论文未公开
- **关键超参**：AdamW，β₁=0.9/0.95，β₂=1e-8，weight decay=0.1，cosine schedule，linear warmup 10%，max lr=5e-5，min lr=5e-6，gradient clipping=1.0，sequence length=16384，global batch size=64，bfloat16，LoRA rank=256/α=512/dropout=0，总 GPU 时长约 6500 小时（NVIDIA DGX H200）
