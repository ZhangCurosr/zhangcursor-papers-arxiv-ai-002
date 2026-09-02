---
title: "Inspicio-Open-Vocabulary-LLM-Based-Sense-Retrieval-for-Histo"
source: https://arxiv.org/pdf/2609.00998v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 19:25:12"
field: "历史语言计算语言学/词义消歧"
keywords: ["Word Sense Disambiguation", "Open Vocabulary Retrieval", "Historical Languages", "Large Language Models", "OEWN", "Mixed Dense-Sparse Retrieval"]
innovations: ["零源语词义库存的开放词汇检索管道，通过 LLM 翻译+定义+lemma 双通道检索链接到 OEWN", "定义排名衰减加权与 lemma boost 的混合评分机制，兼顾密集语义与稀疏词元覆盖", "MMR 多样化重排缓解 OEWN 细粒度近义重复，支持可审计零样本 WSD 管道"]
benchmarks: ["Perception-Verb Test Set (Latin + Ancient Greek, 150 tokens)", "PREMOVE (Latin + Ancient Greek preverbed motion verbs, ~2800 tokens)", "Diachronic Italian Motion Verb Dataset (100 tokens)"]
---

# 论文速读：Inspicio-Open-Vocabulary-LLM-Based-Sense-Retrieval-for-Histo

## 一句话总结
INSPICIO 是一个面向历史语言与低资源语言的开放词汇词义检索管道，通过指令微调 LLM 将源语言上下文中的目标词翻译、定义和候选词元映射到 Open English WordNet (OEWN) 义原，无需任何源语言词义词典或词-义映射；在拉丁语/古希腊语感知动词测试集上以 DeepSeek V4 Pro + KaLM-Embedding 组合达到 96% Recall@50。

## 研究问题与动机
- **传统 WSD 的基础假设在历史语言中失效**：现代 WSD 依赖（i）目标语言存在完整词义词典、（ii）歧义 span 已被标注或可被可靠识别、（iii）存在词到义的候选映射；这些对大多数历史语言并不成立。
- **历史语言缺乏词义标注资源**：拉丁语、古希腊语等的高质量 WSD 数据极其稀缺，现有最大跨语言评测框架仅覆盖 18 种现代语言。
- **手工构建 WordNet 成本高昂且自动迁移会引入噪声与覆盖缺口**：扩展法（expand method）往往继承源语言的覆盖缺陷与错误。
- **英语枢纽资源（OEWN）丰富但未被充分利用于开放词汇检索**：OEWN 是目前最全面的类 WordNet 资源之一，且已作为拉丁/希腊语 WordNet 构建项目的枢纽语言；论文尝试利用这一点实现零样本开放词汇检索。

## 核心贡献（创新点）
- **零源语词义库存的开放词汇检索管道**：INSPICIO 不依赖任何源语言 sense inventory 或 word-to-sense mapping，通过 LLM 生成英语假设词义并链接到 OEWN，与现有需映射的 WSD 系统形成本质区别。
- **翻译驱动的密集-稀疏混合检索设计**：将 LLM 生成的字面/自然双译、候选定义与候选 lemma 并行驱动密集语义检索与稀疏词元匹配，弥补单一检索路径在 idiom 或细粒度语义差下的缺失。
- **定义排名衰减加权聚合（Definition-Rank Decay Scoring）**：针对不同定义返回的 synset 候选池，按 LLM 给出的置信排名施加递减权重（1.0 / 0.75 / 0.5），使最可能释义对最终排序贡献更大。
- **Lemma 增强得分与回退机制**：为同时命中密集与稀疏两个检索池的 synset 施加 γ=0.8 提升，并为仅由 lemma 命中的 synset 设置 β=0.65 保底分数，避免其被密集检索淹没。
- **MMR 多样化重排以提升可解释性**：在合并池中以 λ=0.8 进行 Maximal Marginal Relevance 重排，缓解 OEWN 细粒度导致的顶部近义重复问题，便于下游词典审查。

## 方法详解
INSPICIO 按 token 逐条处理，输入为目标词、字典词元、所在句与语言标签，输出为 OEWN synset 的排名列表及中间制品（可审计）。

- **阶段一：翻译与假设生成（LLM 零样本调用）**
  - 第一次调用：以较高温度（T=1.0）让 LLM 生成两种英语翻译——字面译（贴近源语言结构与词序，保留词源线索）与自然译（地道现代英语，捕捉习语性解读）。
  - 第二次调用：在提供原句、目标词、词元与两份翻译后，要求 LLM 输出 1-3 条按可信度排序的词典式定义与 1-5 条候选英语词元/短多词表达（JSON）。提示明确规避常见陷阱：区分动词自身语义与论元语义、处理否定、包含字面与隐喻读法。温度降至 T=0.8 以稳定词义输出。

- **阶段二：嵌入与密集检索**
  - 构建 4 类 OEWN 索引（名词/动词/形容词/副词），本实验聚焦动词分片。每个 synset 表示为 lemmas + gloss + examples + hypernyms + lexname 的拼接文档。
  - 使用同一 sentence-embedding 模型编码所有文档，存入 ChromaDB，嵌入做 L2 归一化，内积等价于余弦相似度。
  - 对每条 LLM 生成的定义 d 编码后，检索 Top N=50 的近邻 synset。

- **阶段三：定义排名衰减聚合**
  公式：
  $$ S_{\text{base}}(s) = \sum_{d=1}^{D} w_d \cdot \cos(\mathbf{e}_d, \mathbf{e}_s) $$
  权重 w=(1.0, 0.75, 0.5)，仅在定义 d 进入 Top-50 时才计入相似项；复用检索列表而非全量重算以降低成本。

- **阶段四：稀疏词元检索**
  建立从英语 lemma 到含该词元的 synset 集合的倒排索引（一次性构建并缓存），将仅凭词汇对应而嵌入距离较远但仍相关的高价值候选纳入池。

- **阶段五：合并与最终得分**
  公式：
  $$ S_{\text{final}}(s) = \begin{cases} S_{\text{base}}(s) \cdot (1 + \gamma \cdot \mathbf{1}[s \in \mathcal{L}]), & \text{if } S_{\text{base}}(s) > 0 \\ \beta, & \text{if } S_{\text{base}}(s) = 0 \land s \in \mathcal{L} \end{cases} $$
  其中 γ=0.8（双通道协同提升），β=0.65（仅 lemma 命中的保底分）；合并后取 Top 500 进入下一阶段。

- **阶段六：MMR 多样化重排**
  公式：
  $$ \text{MMR}(s) = \lambda \cdot S_{\text{final}}(s) - (1 - \lambda) \cdot \max_{s' \in S} \cos(\mathbf{e}_s, \mathbf{e}_{s'}) $$
  λ=0.8 偏向相关性同时打破近似同义簇；最终截取 K=50。

- **阶段七：可审计输出**
  每条样本输出 JSON 记录：两版翻译、定义、候选词元、Top-K synset 与分数、lemma 命中标志及每定义贡献，便于词典学家追溯与修正。

## 实验与结果
- **数据集**
  - 主评测：新构建的双语感知动词数据集，150 条（拉丁语 72 + 古希腊语 78），选自 PREMOVE Base Corpus，覆盖早期到晚期希腊与早古典到后古典拉丁；两注释员独立标注并解决分歧，Cohen's κ=0.895。
  - 域外评测：PREMOVE 数据集，约 2800 条前缀动词运动义项，跨越 8 世纪 BCE 至 2 世纪 CE， senses 直接编码为 OEWN synsets。
  - 跨语言评测：100 条意大利语历时运动动词（来自 MIDIA 语料库，从中世纪晚期到当代），与拉丁语 PREMOVE 词条对齐；κ=0.914。

- **模型配置**：6 种指令微调 LLM × 6 种 sentence-embedding 模型（共 36 组合）：
  - LLM：DeepSeek V3.2、DeepSeek V4 Pro、Kimi K2.6、GLM 5.1、Qwen 3.5 397B A17B、Mistral Medium 3.5
  - Embedding：text-embedding-3-large、KaLM-Embedding-Gemma3-12B-2511、Qwen3-Embedding-8B、Cohere Embed v4、Harrier-OSS-27B、jina-embeddings-v5-text-small

- **主要结果（Recall@k）**
  - **感知动词集**：最优组合 DeepSeek V4 Pro + KaLM-Embedding-Gemma3-12B 达到 **Recall@50 = 96%**；KaLM 在 6 种 LLM 中 5 次最强，DeepSeek V4 Pro 在 6 种 embedding 中 4 次最强（Table 1）。
  - **PREMOVE 域外**：同样最优组合 Recall@50 = **81.65%**（Table 2），低于感知动词约 15 点，符合更分散的语义与 Zipfian 分布。
  - **历时意大利语**：Recall@50 = **91.00%**，高于 PREMOVE，说明英语枢纽策略在源语言本身被 LLM 充分预训练的情况下仍然有效。
  - **消融（Table 2）**
    - 移除翻译阶段：96% → 92%（降幅最大，说明双译提供独立于 gloss generator 的信息；单次 LLM 调用虽可减半推理成本但会牺牲约 4%）。
    - 移除 lemma boost：96% → 94%（密集与稀疏互补短板）。
    - 移除定义排名衰减 / 移除 MMR：均无明显下降；MMR 因有助于多样性而被保留。

- **结论**：RQ1-RQ4 均获正面回应——无源语词典仍能产生高召回候选池；LLM 与 embedding 选型相互影响但 KaLM/DeepSeek V4 Pro 最稳健；密集+稀疏组合带来可测量增益；在域外与跨语言数据上仍具竞争力。

## 相关工作脉络
- **Bejgu et al. (2024) Word Sense Linking**：形式化为从原始文本直接链接到参考库存的任务，但在源语言映射不完整时性能急剧下降；本文与之相对的区别在于彻底放弃源语映射假设并采用开放词汇检索。
- **Bevilacqua et al. (2020) & Meconi et al. (2025) 生成式 WSD**：将 WSD 当作 gloss 生成任务，证明自由定义比固定候选选择更准确（Meconi 最高 98%）；INSPICIO 沿用这一生成思路但把目标锁定到稳定 OEWN synset 而非仅产生定义。
- **Bamman & Burns (2020)、Lendvai & Wick (2022) 拉丁语 BERT**：依赖二进制分类式 WSD 与 Thesaurus Linguae Latinae 等有限标注；本文零样本路线完全不依赖此类训练数据。
- **Ghinassi et al. (2024) 平行语料传播**：将英语 WSD 标注通过平行语料传播到拉丁语；本文不依赖平行语料，通过英语枢纽实现跨语言对齐。
- **Ghizzota et al. (2025, 2026) LLM WSD 与知识图谱**：比较 zero-shot/fine-tuned LLM 及 LLM-assisted annotation 对拉丁语的影响；本文定位为与其互补的 open-vocabulary retrieval-first 管道。
- **PREMOVE (Farina, 2025)、Marchesi et al. (2025) 历史语言 WSD**：现有工作多针对特定动词类或需要已有 WordNet；本文在感知动词与 PREMOVE 上均验证了通用性，并为 bootstrapping 新 WordNet 提供银标来源。

## 局限性与未来方向
- **评测范围局限于动词**：虽然架构对词类无关且已建四类索引，但名词、形容词、副词仍需验证。
- **随机性传播风险**：两次 LLM 调用引入采样不确定性，尤其翻译阶段使用高温度会放大误差链；可通过可审计中间产物追溯，但无法完全消除。
- **英语枢纽的语言偏置**：OEWN 体现现代英语词汇化模式，某些源语义项在英语中无完全对应，只能返回最接近 synset。
- **未来方向**：引入 LLM-based reranker 进行 top-K→top-1/2 精排；以银标数据补强历史语言专用 WordNet 构建；探索 supervised fine-tuning 与 agentic LLM 自主检索设定。

## 研究启发与可借鉴点
- **混合检索设计范式**：密集定义相似 + 稀疏词元倒排 + MMR 多样性，可直接迁移至其他低资源语言或跨语言 WSD 场景，作为零样本 sense linking 的通用模板。
- **翻译双轨策略的价值**：字面+自然双译同时保留词源与习语信息，可作为其他语言处理中弥补单译歧义的有效手段；即便为节省成本只用单次调用，也可考虑蒸馏该信息。
- **定义排名衰减权重的自适应设计**：w=(1.0, 0.75, 0.5) 简单有效，可进一步根据 LLM 置信度评分或校准后校准，提升对不同模型/语料的泛化性。
- **银标管道化用于资源建设**：INSPICIO 输出的审计痕迹可直接支撑 Lexical Semantic Change、WordNet bootstrapping 等项目，建议与团队现有的历史语言语料管线对接。
- **评估指标建议**：除 Recall@k 外，未来可补充 Precision@k、MRR、Coverage@k 及人工审计通过率，以更全面衡量开放词汇检索系统的可用性。

## 关键术语表
- **WSD（Word Sense Disambiguation）**：词义消歧，判断语境中歧义词的具体义项。
- **OEWN（Open English WordNet）**：开源英语 WordNet，包含丰富的 synset、gloss、例子与层级结构，本文作为外部知识索引。
- **Synset（Synonym Set）**：同义词集合，WordNet 中表达同一概念的所有词元的集合，为 WSD 的基本目标单元。
- **Sense Inventory（词义词典）**：目标语言中所有已标注义项的清单；传统 WSD 假设其存在，本文不再依赖。
- **Definition-Rank Decay Scoring**：对 LLM 多定义产生的检索结果按定义排名施加递减权重加权求和的聚合策略。
- **MMR（Maximal Marginal Relevance）**：在检索结果中兼顾相关性（relevance）与多样性（diversity）的重排算法，本文 λ=0.8。
- **Zero-shot WSD**：无需目标语言标注数据即可进行词义消歧；INSPICIO 的核心设定。
- **Silver-standard Annotation**：由系统自动生成的非人工校验义项标注，可用于后续训练或资源建设。

## 可复现要素
- **数据集**
  - 感知动词双语评测集（150 条，拉丁语 72 + 古希腊语 78）：论文声明发布在 GitHub。
  - PREMOVE 数据集（Farina, 2025，约 2800 条）：已公开发布。
  - 历时意大利语数据集（100 条，MIDIA 语料）：论文声明发布在 GitHub。
- **代码/权重**：GitHub 开源（论文 URL 含 GitHub 链接说明：code、prompts、evaluation data 均已开源）。
- **关键超参**：翻译温度 T=1.0、定义/lemma 温度 T=0.8；定义数 1-3、lemma 候选 1-5；N=50（密集检索 top-K）、γ=0.8、β=0.65、MMR λ=0.8、最终截断 K=50、合并后截断 500。
- **模型**：DeepSeek V4 Pro、KaLM-Embedding-Gemma3-12B-2511 为最佳组合；其余模型均在附录给出 API 调用参数与 prompt 模板。
