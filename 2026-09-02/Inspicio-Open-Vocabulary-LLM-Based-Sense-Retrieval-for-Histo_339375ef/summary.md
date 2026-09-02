---
title: "Inspicio-Open-Vocabulary-LLM-Based-Sense-Retrieval-for-Histo"
source: https://arxiv.org/pdf/2609.00998v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 12:08:26"
field: "历史语言计算语义学"
keywords: ["Word Sense Disambiguation", "Open-Vocabulary WSD", "Historical Languages", "Large Language Models", "WordNet", "Dense-Sparse Hybrid Retrieval", "Latin", "Ancient Greek"]
innovations: ["提出无需源语言词义库存的开放词汇表LLM驱动检索流水线INSPICIO", "设计字面+自然双轨翻译与定义/词根生成的两阶段提示策略", "通过密集定义相似度与稀疏词根匹配的混合检索+MMR重排在OEWN中召回历史语言token的候选义项"]
benchmarks: ["Perception-Verb Test Set (Latin + Ancient Greek, 150 tokens)", "PREMOVE (preverbed motion verbs, ~2800 tokens)", "Diachronic Italian (100 tokens from MIDIA corpus)"]
---

# 论文速读：Inspicio: Open-Vocabulary, LLM-Based Sense Retrieval for Historical Languages

## 一句话总结
本文提出了 **INSPICIO**，一套面向拉丁语、古希腊语等历史低资源语言的开放词汇表词义检索流水线；该方法完全无需源语言词义库存或词-义映射，通过指令微调 LLM 生成句子的双轨英语翻译、候选定义与候选词根，再经密集/稀疏混合检索与 MMR 重排在 Open English WordNet（OEWN）中召回目标词的候选 synset，在感知动词测试集上达到 96% Recall@50。

## 研究问题与动机
1. **传统 WSD 的基础假设在历史语言上失效**：标准词义消歧预设源语言存在完整 sense inventory 与 word-to-sense mapping，而绝大多数历史语言（及低资源语言）的专用 WordNet 尚未建成或严重不完整。
2. **手动构建 WordNet 成本高昂，自动迁移继承噪声**：基于 expand 方法的跨语言迁移往往从源语言继承覆盖缺口与标注噪声，难以直接应用于古典语料。
3. **现有历史语言 WSD 工作严重依赖平行标注数据**：如 Ghinassi et al. (2024) 通过英语-拉丁语平行语料传播标注，Ghizzota et al. (2025, 2026) 依赖 SemEval-2020 Latin 子集，一旦源语言无标注则无从下手。
4. **需要一个 truly zero-shot 且可扩展的检索框架**：研究者希望在不依赖任何源语言 sense 标注的前提下，仍能为目标 token 召回高质量候选义项，服务于词典编撰、知识图谱构建与银标数据生成。

## 核心贡献（创新点）
1. **提出了首个面向历史/低资源语言的开放词汇表检索式 WSD 流水线 INSPICIO**：与 Bejgu et al. (2024) Word Sense Linking 依赖源语言映射不同，INSPICIO 完全不依赖任何源语言 sense inventory 或 word-to-sense mapping。
2. **设计了"双轨翻译 + 定义/词根生成"的两阶段 LLM 提示策略**：先产生字面与自然两种英语翻译以保留词源线索与习用解读，再以 lexographer 角色生成 1–3 条候选定义与 1–5 个候选词根；与 Meconi et al. (2025) 纯生成式解释不同，本文通过检索将其锚定到稳定可解释的 OEWN 空间。
3. **构建了双语感知动词评估数据集（拉丁语 72 / 古希腊语 78）并开源全部代码与数据**：覆盖共和晚期至帝国时期拉丁语与古风至晚期希腊语，Cohen's κ=0.895；公开仓库使基线可复现。
4. **系统验证了英语枢轴策略在跨语言/历时场景下的有效性**：在 PREMOVE（81.65%）与历时意大利语（91%）两个 out-of-domain / cross-lingual 设定下均保持竞争力，证明 OEWN 枢轴可推广至印欧语系内更广泛的历史语言。
5. **消融实验揭示密集/稀疏检索的互补性与翻译阶段的独立价值**：移除翻译阶段 Recall@50 从 96% 降至 92%；移除 lemma boost 降至 94%，证实双轨信号的不可互相替代性。

## 方法详解
INSPICIO 逐 token 处理，输入为目标 token、字典词根、所在句子与语言标签，输出为 OEWN synset 排序列表及中间产物。整体流程如下：

1. **翻译与假设生成（两阶段零样本 LLM 调用）**
   - **第一次调用**：以 `T=1.0`（特定模型文档另有规定除外）生成两条英语翻译——**literal**（贴近源语言词序与词法，保留词源线索）与 **natural**（符合现代英语习惯，暴露习用解读）。
   - **第二次调用**：输入原句、目标 token、词根及两条翻译，输出 JSON：1–3 条按可能性排序的词典式英文定义，以及 1–5 个候选英文词根或短多词表达；温度调至 `T=0.8` 以保证辞书输出稳定。Prompt 明确要求：定义动词本身而非动词+否定、区分字面与隐喻义、动词语义与论元语义分离，以规避 co-composition 与 negation 误读。

2. **嵌入与密集检索**
   - 构建 4 个 OEWN 子索引（名词/动词/形容词/副词），以动词分区为例：每个 synset 表示为 lemmas + gloss + examples + hypernyms + lexname 的拼接短文档（初步实验表明该表示优于仅用 gloss 或仅用 lemma）。
   - 使用句子嵌入模型对所有 synset 文档与 LLM 生成的每条定义 `d` 进行编码，存储在 ChromaDB 集合中；所有向量为 L2 归一化，点积即余弦相似度。
   - 对每条定义 `d` 检索 top `N=50` 最近 synset。

3. **定义排名衰减加权聚合**
   $$
   S_{\text{base}}(s) = \sum_{d=1}^{D} w_d \cdot \cos(\mathbf{e}_d, \mathbf{e}_s)
   $$
   其中 `w = (1.0, 0.75, 0.5)`，反映 LLM 首定义置信度最高；相似度仅对进入 top-N 池的定义计算，避免全量重编码。

4. **稀疏词根匹配检索**
   - 并行地将 LLM 生成的候选词根查询预建倒排索引（每个英文词根 → 包含该词根的 synset 集合），召回在嵌入空间中语义距离较远但在词法上被翻译锚定的 synset，尤其利于 idiomatic 对应关系。

5. **融合打分**
   $$
   S_{\text{final}}(s) = 
   \begin{cases}
   S_{\text{base}}(s) \cdot (1 + \gamma \cdot \mathbf{1}[s \in \mathcal{L}]), & \text{if } S_{\text{base}}(s) > 0 \\
   \beta, & \text{if } S_{\text{base}}(s) = 0 \land s \in \mathcal{L}
   \end{cases}
   $$
   其中 `\gamma=0.8` 为密集+词根双重支持的 boost，`\beta=0.65` 为仅词根召回的 fallback 分数；合并后取 top 500 进入下一阶段。

6. **MMR 多样性重排序**
   $$
   \text{MMR}(s) = \lambda \cdot S_{\text{final}}(s) - (1-\lambda) \cdot \max_{s' \in S} \cos(\mathbf{e}_s, \mathbf{e}_{s'})
   $$
   取 `\lambda=0.8`，在保留顶部相关性的同时打散 OEWN 细粒度带来的近义聚类，最终截断至 `K=50`。

7. **可审计输出**：每 token 输出 JSON 记录包含两条翻译、候选定义/词根、top-K synset 及其分数与 lemma-match 标志，便于词典学家追踪与人工修正。

## 实验与结果
- **评测数据集**：
  - 感知动词集：拉丁语 72 + 古希腊语 78 = 150 例，来自 PREMOVE Base Corpus，Cohen's κ=0.895。
  - PREMOVE：约 2800 例拉丁/古希腊 preverbed motion verbs，跨越公元前 8 世纪至公元 2 世纪。
  - Diachronic Italian：100 例意大利语运动动词，来自 MIDIA 语料，Cohen's κ=0.914。
- **模型矩阵**：6 个 LLM（DeepSeek V3.2 / V4 Pro、Kimi K2.6、GLM 5.1、Qwen 3.5 397B A17B、Mistral Medium 3.5）× 6 个 Embedding（text-embedding-3-large、KaLM-Embedding-Gemma3-12B-2511、Qwen3-Embedding-8B、Cohere Embed v4、Harrier-OSS-27B、jina-embeddings-v5-text-small）。
- **主要结果（Recall@50）**：
  - **感知动词集**：最佳组合 **DeepSeek V4 Pro + KaLM-Embedding-Gemma3-12B** 达到 **96.00%**；KaLM 在 6 个 LLM 中占 5 个最佳列，DeepSeek V4 Pro 在 6 个 Embedding 列中占 4 个最佳行。
  - **PREMOVE**：81.65%（较感知动词下降约 15 个百分点，归因于 motion verbs 语义更广、Zipfian 分布与历时隐喻漂移）。
  - **Diachronic Italian**：91.00%（尽管意大利语动词经历了显著词汇化与 figurative drift，英语枢轴策略仍然有效）。
- **消融（感知动词集，最佳配置）**：
  - Full pipeline：96.00% / Recall@20=90.67%
  - − translation stage：92.00%（降幅最大，翻译双轨提供定义器无法单独恢复的线索；但仅省一次 LLM 调用可减半推理成本）
  - − lemma boost：94.00%（密集与稀疏检索互补）
  - − definition-rank decay：96.00%（影响不显著）
  - − MMR re-ranking：96.67%（mmr 对 aggregate recall 影响有限但显著提升 top-K 多样性，保留于默认配置）

## 相关工作脉络
1. **Bejgu et al. (2024) Word Sense Linking**：将 WSD 形式化为 raw text 中 span 到参考库存的链接任务，但不依赖 word-to-sense mapping；其 retriever-reader 架构在源语言映射缺失时性能急剧退化，而 INSPICIO 完全绕过该假设。
2. **Meconi et al. (2025)**：发现 LLM 在自由生成 gloss 时比从固定候选列表中挑选更准确（最高 98%）；本文与其理念相近但进一步将生成内容作为检索 query 锚定到 OEWN，兼顾生成灵活性与输出可解释性。
3. **Bamman & Burns (2020)；Lendvai & Wick (2022)**：拉丁语 BERT 二分类/多分类 WSD，受限于 Lewis and Short 与 Thesaurus Linguae Latinae 的粗粒度标注；INSPICIO 不使用监督微调，而是零样本检索。
4. **Ghinassi et al. (2024)**：通过平行语料将英语 WSD 标注传播到拉丁语；需要双语平行语料与源语言标注，INSPICIO 完全不需要任何源语言标注数据。
5. **Ghizzota et al. (2025, 2026)**：在 SemEval-2020 Latin 上比较 zero-shot/fine-tuned LLM 及 GRAG；依赖既有标注集，INSPICIO 可应用于无任何标注的历史语言。
6. **Farina & Ciletti (2025)；Farina et al. (2026)**：针对 preverbed motion verbs 与地理名词的 LLM 驱动标注实验；本文扩展了这一范式，将其形式化为通用的开放词汇表检索流水线。

## 局限性与未来方向
- **仅验证了动词**：尽管流水线 PoS 无关且已构建四类 OEWN 索引，但名词/形容词/副词尚未系统评测，泛化性有待检验。
- **多阶段 LLM 推理引入随机性**：翻译阶段使用高温度采样会放大 run-to-run 波动；虽然中间产物可审计追踪错误来源，但无法完全消除不确定性。
- **英语枢轴的固有偏置**：OEWN 反映当代英语词汇化模式，部分源语言义项在英语中无精确对应，pipeline 只能返回最接近的 synset。
- **未来方向**：
  - 加入第二阶段 LLM-based reranker，从 top-K 候选中选出最终义项，缩小检索与消歧之间的差距。
  - 生成银标 sense 数据集，反哺拉丁语/古希腊语等语言的 WordNet 自举（如 Marchesi et al. 2025、Santoro et al. 2025）。
  - 探索 agentic LLM 在无约束环境下的自主检索能力。
  - 在具备标注资源的语言上尝试 supervised fine-tuning 以进一步提升性能。

## 研究启发与可借鉴点
1. **"字面+自然"双轨翻译提示设计**值得广泛借鉴：同一句子生成两种风格的翻译，可分别激活词源/构词线索与习用/隐喻解读，显著提升后续定义生成的覆盖率，且该策略语言无关。
2. **密集定义相似度 + 稀疏词根倒排 + MMR 重排的混合检索架构**对细粒度词义库（如 WordNet）极为适用，能在保持高召回的同时缓解近义聚类问题；该模块可直接复用到其他基于 WordNet 的检索任务。
3. **定义排名衰减加权（rank decay）**与 **lemma boost/fallback 融合打分**提供了轻量且可解释的多信号整合方案，超参数含义清晰（置信度衰减、双重支持奖励），便于迁移到其他多源检索融合场景。
4. **全链路审计输出（JSON 记录每一步中间产物）**为下游词典学家提供了可追溯的纠错路径，这一设计理念对任何面向人文学者的 NLP 工具都具有重要参考价值。
5. **英语枢轴策略**证明了借助成熟高密度资源（OEWN）桥接低资源/历史语言的有效性，未来可与多语 WordNet（如 MultiWordNet）或跨语言向量空间结合，进一步削弱单一语言的lexicalization bias。

## 关键术语表
**Word Sense Disambiguation (WSD)**：根据上下文确定多义词在句中的具体义项，是词汇语义学的核心任务。
**Open-Vocabulary WSD / Word Sense Linking**：放弃源语言封闭词义库假设，直接从 raw text 中将 span 链接到外部参考库存的开放词汇表范式。
**Open English WordNet (OEWN)**：McCrae et al. (2020) 发布的开源英文 WordNet，涵盖名词/动词/形容词/副词四个词性分区，是本文检索的目标知识库。
**Synset**：WordNet 中一组同义词（synonym set），代表一个抽象的义项，由 lemmas、gloss、examples、hypernyms 等构成。
**Maximal Marginal Relevance (MMR)**：在信息检索中兼顾相关性与多样性的重排序算法，通过惩罚已选结果附近的候选来打散聚类。
**Dense retrieval**：基于句子/定义嵌入向量与库向量余弦相似度进行的近邻搜索。
**Sparse retrieval**：基于词根倒排索引的直接词条匹配检索。
**Zero-shot WSD**：完全不依赖源语言 sense 标注数据，仅通过提示与外部知识库完成消歧的设定。

## 可复现要素
- **代码与数据**：论文声明代码、prompts 与评估数据已发布于 GitHub（论文标注脚注 1）。
- **数据集**：
  - 自建 bilingual perception-verb 集（Latin 72 + Ancient Greek 78），来源为 PREMOVE Base Corpus；
  - PREMOVE（Farina, 2025），约 2800 例 preverbed motion verbs；
  - Diachronic Italian 子集（100 例），来自 MIDIA 语料。
- **模型**：
  - LLM：DeepSeek V3.2、DeepSeek V4 Pro、Kimi K2.6、GLM 5.1、Qwen 3.5 397B A17B、Mistral Medium 3.5（均通过官方 API，使用默认解码参数）；
  - Embedding：text-embedding-3-large、KaLM-Embedding-Gemma3-12B-2511、Qwen3-Embedding-8B、Cohere Embed v4、Harrier-OSS-27B、jina-embeddings-v5-text-small。
- **关键超参**：`N=50`（密集检索 top-K）、`K=50`（最终输出）、`λ=0.8`（MMR 相关性权重）、`γ=0.8`（lemma boost）、`β=0.65`（fallback 分）、`w=(1.0, 0.75, 0.5)`（定义排名衰减）、翻译温度 `T=1.0`、定义温度 `T=0.8`。
- **Prompt 模板**：附录 A 提供了翻译 prompt、定义/词根生成 prompt 与 embedding prompt 的完整 Markdown 版本。
