---
title: "When-Names-Cross-Scripts-A-Source-Grounded-Benchmark-for-His"
source: https://arxiv.org/pdf/2608.23507v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 01:12:40"
field: "历史NLP与跨语言实体消歧"
keywords: ["historical entity reconciliation", "cross-script name matching", "provenance-controlled benchmark", "selective prediction", "abstention", "named entity linking", "multilingual NLP", "evidence-grounded evaluation"]
innovations: ["提出MHER来源控制基准，将历史实体消歧形式化为pairwise证据条件任务并允许弃权", "设计mention×source级别独立证据与shuffle-context伪造控制，分离grounding与文本长度/主题捷径", "跨五个异构生成模型验证source-grounded证据使配对准确率提升12.96–94.44pp并在同名不同人困难案例达24/25"]
benchmarks: ["MHER (Mongol-world Historical Entity Reconciliation)", "MHER-Name-only TEST (316 pairs)", "MHER-Source-grounded paired TEST (108 pairs)", "MHER-Context-only", "MHER-Shuffled-Context A/B/C", "MHER-Expert-unresolved (10 cases)"]
---

# 论文速读：When Names Cross Scripts: A Source-Grounded Benchmark for Historical Entity Reconciliation in the Mongol World

## 一句话总结
论文提出 **MHER**（Mongol-world Historical Entity Reconciliation），一个以来源追溯控制为核心的二元配对基准，用于研究跨语言/跨书写系统的历史人物实体消歧问题；实验表明，在84位蒙古世界历史人物构成的396对名称中，补充来源可溯源的独立历史证据可使五个大模型的配对TEST准确率提升 **12.96–94.44个百分点**，并在"同名不同人"困难案例上从 0/25 提升至 24/25。

## 研究问题与动机
- **名字≠身份**：同一历史人物在不同语言/书写系统/转写传统下可呈现显著不同的表面形式，而不同人物也可能拥有完全相同的名称，仅靠字符串匹配无法解决历史实体消歧。
- **现有基线的不足**：传统多语言实体链接任务要求将mention映射到预定义KB；历史NER/EL资源（如Chinese historical newspapers、KE-MHISTO、Mahan¯ama、MHEL-LLaMo）虽扩展了语种覆盖，但仍未解决"pairwise identity without KB dependency"这一结构性差异。
- **Context构造的泄漏风险**：若上下文中隐含等价关系提示或重复出现，模型可能通过浅层捷径获得高分，而非真正利用来源证据进行推理。
- **缺乏证据-基线分离的评测框架**：现有benchmark难以区分"模型是否使用了正确的来源证据"与"模型是否利用了额外的文本长度/主题相似性"。

## 核心贡献（创新点）
1. **将历史实体消歧形式化为证据条件任务**：区分"name-only"与"source-grounded"两种输入条件，允许模型在证据不足时主动弃权（ambiguous），而非强制二分类。与已有KB依赖型EL的本质区别在于不假设完整实体清单已知。
2. **提出MHER基准**：涵盖84位历史人物、396对name-only样本与160对source-grounded配对子集（DEV 52对，TEST 108对），按canonical entity做entity-disjoint划分，避免跨 split 的身份泄漏。
3. **引入leakage-aware的evidence干预与伪造控制**：构建了context-only、三种确定性shuffle-context控制，以分离"文本存在"与"正确来源对齐"的贡献，并对原始construction中的实体级上下文重复进行了修复审计。
4. **超越强制二分类的评测体系**：结合精确率、macro-F1、弃权率、选择性准确率、概率质量（Brier/log loss）、hard-negative挑战切片（near-surface、identical-surface、cross-language、cross-script）及expert-unresolved sanity check。
5. **在五个异构模型上验证跨政策一致性**：四个API模型（GPT-5.6 Terra/Luna/Sol、Gemini 3.7 Flash）与一个开源模型Qwen3-8B在source-grounded条件下准确率趋近，证明了source evidence的稳定性与泛化性。

## 方法详解
- **任务形式化**：给定两个mention $m_i = (s_i, \ell_i, \sigma_i)$（surface form、language、script/transcription tradition），目标为推断二元关系 $y(m_a, m_b) \in \{\text{SAME, DIFFERENT}\}$，模型可额外输出AMBIGUOUS表示弃权。
- **四种评测条件**：
  - **Name-only**：仅提供 $(m_a, m_b)$，共316对TEST样本。
  - **Source-grounded**：提供 $(m_a, e_a), (m_b, e_b)$，其中 $e_i$ 来自mention×source级别的可溯源独立证据（描述年代、亲属、职官、地域、事件参与等），禁止直接包含等价声明或泄露alternative names。
  - **Context-only**：移除name surfaces，仅保留 $(e_a, e_b)$，用于检验历史文本本身的信息含量。
  - **Shuffled-context**（A/B/C三种）：冻结同一context pool但刻意破坏mention-evidence对齐，并在prompt中明示，作为"informed misgrounding stress test"。
- **Gold标签来源**：由curator-reviewed的scholarly registry构建，84位primary entities经系统化复审；10个expert-unresolved案例单独评估。
- **评估指标**：Acc、macro-F1、Abstention Rate、Selective Accuracy、Mean $p(\text{gold})$、Multiclass Brier Score、Log Loss；配对分析使用McNemar检验、200,000次bootstrap CI、entity-cluster bootstrap与leave-one-entity-out敏感性。
- **Baseline体系**：lexical baselines（exact match、Unicode Levenshtein、Unidecode+Levenshtein、char 2-5-gram TF-IDF）、context baselines（token Jaccard、six-feature logistic classifier）、multilingual embedding baseline（gemini-embedding-2）。
- **模型阵容**：GPT-5.6 Terra/Luna/Sol、Gemini 3.7 Flash、Qwen3-8B（本地量化部署）。

## 实验与结果
- **非生成基线**（Name-only on 316 TEST）：Gemini Embedding 2最佳，Acc=64.87%；Best context baseline（Token Jaccard on 108 subset）Acc=76.85%，Gemini Embedding 2 Acc=86.11%。
- **Name-only生成模型表现高度政策依赖**：Terra Acc=35.13%（Abst=63.61%, Sel.Acc=96.52%）；Luna Acc=6.01%（Abst=93.99%, Sel.Acc=100%）；Sol Acc=4.11%（Abst=95.89%, Sel.Acc=100%）；Gemini Acc=16.77%（Abst=83.23%, Sel.Acc=100%）；Qwen3-8B Acc=67.09%（Abst=0%, 强制二分类）。
- **Source-grounded配对提升**（108 TEST对）：
  - Terra：32.41% → 100.00%（Δ=+67.59pp, W→C=73, C→W=0）
  - Luna：6.48% → 99.07%（Δ=+92.59pp, W→C=100, C→W=0）
  - Sol：4.63% → 99.07%（Δ=+94.44pp, W→C=102, C→W=0）
  - Gemini：20.37% → 100.00%（Δ=+79.63pp, W→C=86, C→W=0）
  - Qwen3-8B：75.00% → 87.96%（Δ=+12.96pp, W→C=23, C→W=9, McNemar $p=0.0201$, Bonferroni调整后$p≈0.100$）
- **Context-only vs Source-grounded**：API模型均因name恢复而提升（Terra +12.96, Luna +18.52, Sol +6.48, Gemini +5.56）；Qwen3-8B反降6.48pp（Context-only 94.44% vs Source 87.96%, $p=0.0923$ 不显著），揭示surface-form interference。
- **Shuffle控制**：所有模型在shuffled条件下准确率大幅下降（Terra 13.89-16.67%, Luna 11.11-22.22%, Sol 0.93-4.63%, Gemini ~50%, Qwen 57.41-63.89%），证明正确grounding的必要性。
- **Identical-surface hard negatives（5对）**：Name-only全模型 0/25；Source-grounded 24/25正确（Sol在1例弃权）。
- **Expert-unresolved（10对）**：所有模型在显式signal下均返回AMBIGUOUS（50/50），验证弃权协议可行性。
- **Relevant strongest result**：GPT-5.6 Sol达到最大配对增益（+94.44pp）并接近满分；所有API模型在Source-grounded条件下Selective Accuracy≈100%。

## 相关工作脉络
1. **Multilingual Entity Linking**：Pan et al. (2024) 282语种命名实体链接、Botha et al. (2020) 100+语种KB链接、mGENRE autoregressive multilingual EL——均要求mention→KB映射；本文聚焦pairwise $m_a \equiv m_b$，不假设完整KB已知。
2. **Pairwise Identity Inference**：Green et al. (2012) 跨语料实体聚类；MOLEMAN提及级表征——结构相近但面向当代语料；本文将其迁移至"证据需跨语言/书写重建"的历史场景。
3. **Cross-Script Name Processing**：ParaNames (~1.4亿名/1680万实体/400+语种)、Sälevä & Lignos (2024)、Jayakumar et al. (2026) 脚本壁垒综述——强调surface correspondence≠identity relation。
4. **Historical NLP / Long-tail EL**：Blouin et al. (2024) 中国近代报刊、KE-MHISTO (Graciotti et al. 2025) 多语种长尾、Mahan¯ama (Sarkar et al. 2025) 梵语文学实体——本文补充pairwise视角与provenance控制。
5. **MHEL-LLaMo**（Santini et al. 2026）：六语种历史EL+LLM置信度选择——同为历史EL但未做mention×source级别证据干预与misgrounding falsification。
6. **Statistical Record Linkage**：Fellegi–Sunter (1969) 多字段匹配理论；Abramitzky et al. (2021) 历史人口链接中地理/亲属信息的增益——本文继承"evidence-combination"视角但面向手写/多文种史料。
7. **Attributed Generation / Falsification**：ALCE (Gao et al. 2023)、ExpertQA (Malaviya et al. 2024) 侧重输出侧引用验证；本文在输入侧通过shuffled context控制证据对齐，与contrast-set (Gardner et al. 2020) 精神一致。
8. **Selective Prediction / Abstention**：Xin et al. (2021)、Wen et al. (2025) 综述——本文明确区分"gold identity二元"与"model action三元（SAME/DIFFERENT/AMBIGUOUS）"，避免将uncertainty误判为错误。

## 局限性与未来方向
- **域范围有限**：仅覆盖蒙古世界人物，未验证其他历史时期/区域/实体类型（组织、地点、事件）。
- **证据理想化**：当前context为紧凑且聚焦的paraphrase；真实档案常含冗长/矛盾/缺失文献，需扩展到retrieve-then-reconcile场景。
- **Gold依赖人工curator**：84 entities经复审但非独立双盲标注；历史身份本身存在学术争议，十例unresolved案例不足以统计检验。
- **数据规模较小**：396对name-only与108对source-grounded相比大规模当代EL benchmark偏小；尽管entity-disjoint与重复实体敏感性分析稳健，但跨社群泛化仍待验证。
- **Prominence仅为代理变量**：head/mid/long-tail按public visibility粗分类，非真实pretraining频率测量，无法直接断言memorization。
- **模型复现受限**：商业API版本迭代快；Qwen3-8B为固定量化checkpoint；prompt、运行配置、frozen project state未在本次arXiv版本中公开。
- **Unresolved条件单向**：仅在name-only下评估；source-grounded unresolved因证据中易含uncertainty语言而存在泄漏风险，未来需设计neutral evidence。
- **开源计划**：数据/代码/复现包计划于正式发表后公开，受source redistribution constraints限制；模型可见证据将以project-authored paraphrase+biblio citation形式提供。

## 研究启发与可借鉴点
1. **Provenance作为benchmark语义的一部分**：在引入外部context的NER/EL/RE任务中，应审计mention×source对齐、跨样本重复、explicit equivalence cue等泄漏路径，而非仅关注文本内容。
2. **Shuffle/mismatched control作为informed stress test**：显式告知模型证据已被错位，可区分"模型利用正确grounding"与"模型仅依赖文本长度/主题分布"，比blind noise injection更直接。
3. **Abstention作为独立能力评估维度**：对低资源/长尾实体场景，强制二分类会掩盖模型的真实知识边界；在prompt中提供third action并配合expert-unresolved sanity check，有助于评估selective prediction。
4. **Same-pair source-independence约束**：构建gold same对时要求两侧证据独立attested，防止shared canonical description被两遍复用导致shortcut——此设计可直接迁移至任何pairwise linking benchmark。
5. **跨政策模型的对比价值**：选择不同abstention倾向的模型（conservative API vs aggressive open-weight）可分离"knowledge"与"decision policy"，避免单一分数误判；建议后续工作报告Selective Accuracy与Abstention Rate联合解读。
6. **Surface-form interference诊断**：Qwen3-8B案例显示，当strong context已足够时，恢复name可能激活unsupported alias hypothesis并引发false merge；可在误差分析中引入"context-only vs source-grounded"差分编码。

## 关键术语表
- **Historical Entity Reconciliation**：针对历史人物name attestations的pairwise身份判断任务，目标为推断两个mention是否指向同一历史个体，不要求映射到现代KB。
- **Source-Grounded Evidence**：与mention × source记录绑定的独立可溯源证据（年代、职官、亲属、地域、事件等），禁止直接陈述等价关系或泄露替代名称。
- **Entity-Disjoint Split**：按canonical person划分DEV/TEST，确保同一人的不同名称变体不出现在对立split中，防止身份泄漏。
- **Selective Prediction / Abstention**：模型在证据不足时主动输出AMBIGUOUS（弃权）而非强制二分类；不同于EL中的NIL预测（后者针对KB无候选）。
- **Shuffled-Context Control**：保持同一context pool但冻结破坏mention-evidence对齐的确定性错配，并在prompt中明示，用于检验grounding依赖性。
- **Leakage-Aware Construction**：审计并排除上下文中的exact duplication、explicit equivalence statement、registered alias等可能预测gold label的捷径信号。
- **Identical-Surface Hard Negative**：两mention表面名称完全相同的配对；仅凭字符串无法区分，必须依赖历史证据（年代/亲属/职官等）。
- **McNemar Paired Test**：用于比较同一108对样本在Name-only与Source-grounded两种条件下的准确率变化显著性。

## 可复现要素
- **数据集**：MHER，包含84 primary entities、396 Name-only pairs、160 Source-grounded pairs、10 Expert-unresolved cases；**当前arXiv版本未公开**，计划正式发表后公开（受source redistribution constraints限制）。
- **代码/权重**：benchmark construction pipeline、prompt、evaluation script、frozen project state（含execution records、exact artifacts identity）**未公开**，计划发表后开源。
- **模型**：GPT-5.6 Terra/Luna/Sol（OpenAI Responses API）、Gemini 3.7 Flash（Google Gemini API）、Qwen3-8B（本地量化部署）；**API版本与量化checkpoint细节未公开**，需查阅frozen project record。
- **关键超参**：阈值与baseline拟合在DEV选择并冻结；McNemar检验、200,000次bootstrap、entity-cluster bootstrap；Prompt细节与条件instruction未公开。
- **评估脚本**：未公开；论文声明正式发表后计划提供reproducibility package。
