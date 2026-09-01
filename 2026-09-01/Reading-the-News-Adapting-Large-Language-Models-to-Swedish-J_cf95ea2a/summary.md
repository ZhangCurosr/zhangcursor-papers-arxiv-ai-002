---
title: "Reading-the-News-Adapting-Large-Language-Models-to-Swedish-J"
source: https://arxiv.org/pdf/2608.30609v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 23:44:36"
field: "低资源语言领域适配"
keywords: ["continued pre-training", "domain adaptation", "Swedish NLP", "news journalism LLM", "LoRA", "instruct vector", "experience replay", "low-resource language"]
innovations: ["首次系统性地将 CPT 适配到瑞典新闻领域，构建 BonCorpus 与 BonEval", "发现经验回放 BCES 配方是避免灾难性遗忘的关键，纯新闻语料导致 -2.75 分下降", "揭示 LoRA 与 IV 的高兼容性（参数漂移 0.08 vs FFT 0.18），LoRA+IV 达成 BonEval 最高 46.02"]
benchmarks: ["BonEval", "EuroEval (Swedish)", "SWEReC", "SUC 3.0", "ScaLA", "MultiWikiQA", "SweDN", "MMLU", "HellaSwag"]
---

# 论文速读：Reading-the-News-Adapting-Large-Language-Models-to-Swedish-J

## 一句话总结
通过持续预训练（CPT）将 LLM 适配到瑞典新闻领域，构建了大规模高质量瑞典新闻语料 BonCorpus（14.6M 篇、8.5B tokens）和 6 任务领域评测基准 BonEval；发现 CPT 配合经验回放（experience replay）可显著提升生成质量和事实知识，但仅对 LoRA 适配方式可兼容 instruct vector（IV）以提升指令遵循能力。

## 研究问题与动机
- 通用 LLM 在细分/低资源语言（如瑞典语）和特定领域（如新闻业）中的能力薄弱，而直接从零预训练大型模型需要海量算力，不经济；
- 新闻业虽在经典 NLP 任务中资源丰富（WSJ、Reuters、CNN/Daily Mail），但作为 LLM 适配目标领域几乎未被系统研究，仅有的少数工作聚焦英语金融或中文新闻；
- 瑞典语作为北欧主要语言之一，缺乏面向新闻领域的 CPT 研究（已有工作仅聚焦从 scratch 预训练或规模较小的模型）；
- 评估资源匮乏：瑞典新闻领域仅有少量评测任务（lead paragraph、summarisation、SEO headline），现有 EuroEval/Superlim 基准未能有效捕获领域内性能变化。

## 核心贡献（创新点）
- **首次系统性地将 LLM CPT 适配到瑞典新闻领域**：此前仅有一个研究关注过新闻作为 LLM 适配目标，且非瑞典语；本文填补了"语言×领域"双重缺口；
- **构建 BonCorpus：从 21.7M 原始文章经清洗→去重→过滤 pipeline 得到 14.6M 高质量瑞典新闻，总量 8.5B tokens**：引入 MinHash LSH + Levenshtein 距离图连通分量去重策略，减少近重复和格式残留，这是领域语料工程的核心方法论贡献；
- **构建 BonEval：6 个编辑任务的领域评测基准（headline/lead/summary 生成、topic 分类、entity NER、quiz 问答）**：覆盖生成、判别、知识密集型三类任务，弥补瑞典新闻领域评测空白；
- **揭示 CPT 经验回放配方的关键性**：纯 BonCorpus（B）导致灾难性遗忘（-2.75），而加入 code+English+Swedish 回放（BCES）后平均提升 +1.75（相对 base），说明代理数据组成是 CPT 成败的关键；
- **发现 LoRA + IV 的组合最优且 LoRA 与 IV 高度兼容（而 FFT 与 IV 不兼容）**：参数漂移度量（LoRA 0.08 vs FFT 0.18）解释了 IV 兼容性的差异，这是一个新的可复现设计启示。

## 方法详解
- **BonCorpus 数据工程**：来源为 Bonnier News 1991–2026 年 21.7M 篇瑞典文章；预处理分四步：①解析聚合统一格式；② Unicode 规范化 + 正则去除链接/广告/作者署名等噪声；③基于文本统计（min 20 words、newline 2–2000、type-token ratio ≥ 0.2）、fastText 语言识别（>0.9）、去模板类文章（体育比分、天气预报、房地产等自动生成都排除）的过滤器；④MinHash LSH（112 hash、阈值 0.25）做近似去重 + Levenshtein 距离阈值 < 0.2 + 图连通分量选最长文，最终 14.6M 篇、8.5B tokens。
- **BonEval 六任务**：Headline（标题生成，CHRF3++）、Lead（导语生成，CHRF3++）、Summary（要点摘要 3–5 bullet，CHRF3++）、Topic（19 类 IPTC 话题分类，MCC）、Entity（NER，仅人物/组织/地点中心实体，micro F1）、Quiz（11,960 道多选题，15 类，涵盖瑞典及全球知识，MCC）。除 Quiz 外都来自 BonCorpus，文章发布年份 ≥ 2022 以最小化与原预训练的 overlap。
- **数据混合策略**：以 BonCorpus 为主，从 Common Corpus 分别采样 Code（C）、English（E）、Swedish（S）作为 10% token 的回放源，构成 B/BC/BE/BS/BCE/BCS/BES/BCES 八种 mixture；BCES 最优。
- **模型与训练**：base 为 Ministral 3 3B/8B；PEFT 对比 LoRA（rank=256, α=512, dropout=0）和 LLaMA Pro（每 6 层插入 1 层）；FFT 使用 AdamW、cosine LR schedule（max 5e-5）、global batch=64、seq_len=16384、bfloat16、DeepSpeed ZeRO-2 + FlashAttention-3。
- **Instruct Vector（IV）方法**：计算 instruction-tuned 与 base 模型的参数差向量 Δθ_instruct，将其无权重叠加到 CPT 后的模型参数上（即 θ_CPT+IV = θ_CPT + Δθ_instruct），实现 training-free 指令遵循恢复。

## 实验与结果
- **3B 数据混合选择（Table 1）**：base 39.73；B 纯新闻下降至 36.98（-2.75，灾难性遗忘）；BCES 最高 41.48（+1.75）；LoRA 41.32，LLaMA Pro 41.17；LoRA + IV 达到 3B 实验最高 42.27；FFT + IV 反而降至 39.07。
- **8B 对比（Table 2 / Table 3）**：基线中 Ministral 3 8B Base 在 EuroEval 上均分 58.73 最高；BonLM 8B 在 BonEval 上 LoRA + IV 均值 46.02，超越所有基线；各生成任务（Headline CHRF 29.76/Lead 35.01/Summary 49.88）CPT 均显著提升；Topic 和 Entity 判别任务略有下降（Topic 70.54 vs Base 72.84；Entity 23.99 vs 29.97）；Quiz 从 60.12 升至 66.95–71.36（领域知识获得）。
- **EuroEval 上 CPT 下降明显**：MMLU 从 58.10 降至 30.62（FFT）/47.18（LoRA），HellaSwag 从 43.68 降至 17.66/36.82，说明通用能力被新闻领域分布偏移所侵蚀，LoRA 的约束效应显著缓解此问题。
- **关键数值**：LoRA + IV 在 BonEval 上绝对最高分 46.02；Quiz 是 CPT 增益最大任务（+11.23 absolute）；FFT + IV 是唯一负向组合（BonEval 38.58 vs FFT 44.14，-5.56）。

## 相关工作脉络
- **BloombergGPT (Wu et al., 2023)**：首个金融领域 CPT 大模型，证明领域适配价值，但面向英语高资源场景，与本文瑞典新闻的"低资源+垂直领域"定位不同；
- **News GPT (Yao et al., 2024)**：中文新闻领域 CPT，是极少数新闻适配工作，但语言/文化语境迥异，且评测任务单一；
- **SnakModel (Zhang et al., 2025, Danish) / Norwegian CPT (Samuel et al., 2025)**：北欧语言 CPT 研究脉络，本文是对最后一块北欧拼图（瑞典语）的补全；
- **GPT-SW3 (Ekgren et al., 2024) / Llama SW3 (AI Sweden, 2024) / Apertus (Hernández-Cano et al., 2026)**：瑞典/多语从零预训练模型，本文定位为"CPT 低成本适配"路径，算力门槛远低于 from-scratch；
- **Task Arithmetic / Chat Vector (Ilharco et al., 2023; Huang et al., 2024)**：IV 方法的理论基础，本文首次验证 IV 与 CPT 在 LoRA 下的兼容性并给出参数漂移解释；
- **EuroEval / SweReC / SUC 3.0 / MultiWikiQA / MMLU-SV**：通用瑞典语评测，本文指出机器翻译/LLM 合成题存在文化偏见（Section 5.5），凸显 native 评测 BonEval 的必要性。

## 局限性与未来方向
- **数据/代码/模型不可公开**（商业与法律原因），削弱复现性；
- **仅在 Ministral 家族内部实验**，其他模型族（如 Llama/Qwen）下结论可能不同；
- **未做下游微调**（如具体 headline generation、style correction），无法评估 CPT 的实际编辑应用价值；
- **原预训练数据未披露**，无法量化 BonCorpus/BonEval 与原训练集的 overlap 及其影响；
- **仅依赖自动评测**（CHRF/MCC/F1），未开展人类评估，难以断言真实新闻室效用；
- **未来方向**：下游编辑任务微调、新闻室人机协作评估、扩展至其他北欧语言/垂直领域、探索训练-free 方法避免 IV 对 FFT 的反效果。

## 研究启发与可借鉴点
- **经验回放的代理数据选择需多元**：Code+English+Swedish 三源混合（BCES）最优，单一源或纯领域数据均引发遗忘；可借鉴于任何语言的领域 CPT，建议至少保留 10% 异构 replay；
- **IV 与 PEFT 方法的兼容性差异显著**：LoRA 参数漂移小（0.08 vs 0.18）使其更易叠加 IV，而 FFT 则适得其反；后续研究应区分 PEFT 选择后再评估 IV；
- **去重图连通分量策略可直接迁移**：MinHash LSH + Levenshtein + 图聚类选最长文，适用于任何长文本语料的近重复过滤，且已开源 pipeline 思路可复现；
- **BonEval 的"生成-判别-知识"三分法框架**可作为新闻/法律/医疗等垂直领域的评测模板；
- **EuroEval 类翻译基准对领域 CPT 评价存在系统性偏差**：建议任何非英语低资源研究必须配套 native 评测集，否则得出"CPT 有害"的伪结论。

## 关键术语表
- **Continued Pre-training (CPT)**：在已有基座模型上继续使用目标域语料进行下一词预测预训练，以适配领域而保持通用能力；
- **Experience Replay**：CPT 时混入一定比例历史/代理数据（代码、英语等）以缓解灾难性遗忘；
- **Catastrophic Forgetting**：CPT 对领域数据过度拟合导致原预训练通用能力大幅退化；
- **Low-Rank Adaptation (LoRA)**：冻结主权重，仅训练低秩分解的更新矩阵 ΔW=AB，参数量少、抗遗忘；
- **Instruct Vector (IV)**：instruction-tuned 模型与 base 模型参数差的向量，无训练地叠加到 CPT 模型以恢复指令遵循；
- **Task Arithmetic**：IV 的理论基础，假设模型行为可由参数空间中的任务向量加减来组合；
- **MinHash LSH**：基于 min-hash signature 和 locality-sensitive hashing 的近重复文档检索；
- **BonCorpus / BonEval / BonLM**：本文构建的瑞典新闻语料库、评测基准、适配模型的统一命名体系。

## 可复现要素
- 数据集：BonCorpus（14.6M 篇、8.5B tokens）因商业/法律原因**不公开**；Common Corpus 为开源伦理数据（Langlais et al., 2026），可用于 replay；
- 代码：论文明确声明**不公开**；
- 模型权重：**不公开**（Intended for internal Bonnier News deployment）；
- 基座模型：Ministral 3 3B/8B（Mistral，Liu et al., 2026）；
- 训练超参：AdamW β1=0.9/β2=0.95/ε=1e-8/WD=0.1、cosine LR（max 5e-5→min 5e-6，10% warmup）、clip=1.0、batch=64、seq=16384、bfloat16、ZeRO-2 + FA-3 + Liger；LoRA rank=256/α=512/dropout=0；
- 实验算力：总计 6,500 GPU-hours（DGX H200），3B 8K steps ≈ 1 epoch，8B 对应 4 epochs。
