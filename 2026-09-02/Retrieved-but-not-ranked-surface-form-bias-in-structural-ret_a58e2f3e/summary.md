---
title: "Retrieved-but-not-ranked-surface-form-bias-in-structural-ret"
source: https://arxiv.org/pdf/2609.01556v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-03 00:52:46"
---

# 论文速读：Retrieved-but-not-ranked-surface-form-bias-in-structural-ret

## 一句话总结
本文在数学竞赛题和代理轨迹两个不相关领域，系统研究了当表面词汇与底层结构分离时的嵌入检索行为，发现嵌入模型严重锚定于字面token而非任务结构；同时证明LLM重排序可恢复大量性能，且一个廉价的词汇重排序控制器的符号翻转可作为基准测试表面变化机制的低成本诊断工具。

## 研究问题与动机
1. **现有评估盲区**：当前嵌入检索基准主要在"表面形式与语义指向一致"的易 case 下验证，缺乏对结构检索（structural retrieval）——即查询与答案共享底层技术/程序但无共词——的系统评估。
2. **单一领域结论不可靠**：单领域研究会将嵌入检索特性与基准构建特性混杂；本文通过跨域协议揭示同一控制工具（词汇重排序）在不同基准中符号相反，这是单一领域无法观察到的。
3. **前作方法缺陷**：Kohar & Krishnan (2025) 的轨迹检索基准仅用单一384维编码器、相关性标签来自低人评一致性(Cohen's kappa 0.178)的LLM评判、且未实现两阶段检索；MathNet (Alshammari et al., 2026) 虽展示27个模型数学等价检索失败(Recall@1<5%)但未测试重排序。
4. **重排序效果的领域依赖性未知**：已有工作指出无关上下文会损害LLM求解(Shi et al., 2023, MathNet观察到RAG得分低于零样本)，但未在配对设计中比较oracle vs adversarial检索对下游性能的影响。

## 核心贡献（创新点）
1. **双领域统一评估协议**：提出分层伪装、严格命中基线、精确机会基线(hypergeometric)、rank-1失败分类学和bootstrap置信区间的共享协议，在数学(500 query/117,088 corpus)和轨迹(118 query/336 trajectories)两个无关领域验证，首次系统性揭示结构检索的字面锚定偏差。
2. **词汇重排序控制的符号翻转发现**：同一朴素词汇重排序器在数学中损害检索（对抗性表面变化使词汇信号反方向），在轨迹中改善检索（偶然性表面变化使词汇信号为真实相关），其符号可作为区分基准测试"对抗性vs偶然性"表面变化机制的低成本诊断工具。
3. **LLM重排序的不可迁移性证明**：三个独立LLM裁判（Gemini-j, GLM-j, Haiku-j）在两个领域均有效，但效应量跨越八倍以上、等级剖面 disagree、各域各有不同离群裁判；一个裁判在域内查询风格切换时收益下降超三分之一，证明单一重排序效应量不可跨设定迁移。
4. **污染效应的量化定位**：在数学竞赛题重排序中发现知名竞赛题（IMO/USAMO/APMO, n=57）的收益显著高于其他来源（+19.8 pts, CI [+6.7, +33.2]），部分恢复可归因于裁判的记忆而非纯推理，污染归属虽具相关性而非因果性但被实证定位。
5. **配对下游null及其机制定位**：在包含刻意坏检索条件的配对实验中，oracle检索与对抗性检索对DeepSeek-v4-flash求解器无统计差异（McNemar p=0.678）；完整答案分析揭示69.5%零样本准确率实为"是否截断"的代理变量（完成答案准确率达97-100%），检索有效余量接近于零。

## 方法详解
**查询与语料库构建**：
- 数学：500条来自MathNet-Retrieve的查询（每条为源竞赛题的改写），语料库为全量117,088项MathNet集合，分EASY/HARD两层伪装（由Gemini-family模型生成），每条查询含1个gold（源问题）及约3个词汇相似但结构不同的near-miss诱饵。
- 轨迹：118条查询（源基准40条+自行抽取78条来自ALFWorld valid_unseen），语料库336条ALFWorld轨迹，按六种任务类型（pick-and-place, heat-and-place, examine-in-light等）定义procedure。

**Gold定义**：
- 数学：严格Hit@k仅计数gold；宽松Hit@k额外计入planted near-miss。二者差距为核心测量指标。
- 轨迹：相关性由任务类型定义（rule-derived从元数据提取，经人工审计验证）。两层伪装要求：(i)不同目标对象；(ii)不同目标对象+不同容器（使gold与query共享最可能驱动词汇相似的token为零）。

**嵌入检索**：
- 数学：gemini-embedding-001、Qwen3-Embedding-8B（DeepInfra服务）
- 轨迹：+MiniLM-L6-v2（源基准编码器，作弱基线对比）
- 每query嵌入一次，全库余弦相似度排序，无shortlisting。

**重排序阶段**（均在相同top-10候选列表上操作）：
- **词汇控制**：对top-10按三种信号的加权平均重新排序——Jaccard重叠（小写词token, 无词干提取）、Levenshtein编辑距离比、长度比，各信号∈[0,1]。数学比较完整问题文本；轨迹比较query指令与candidate任务描述。机制消融用更窄的Jaccard：动词-only（14个ALFWorld行动动词闭集）vs名词-only（残差非动词非stopword token）。
- **LLM裁判**：三个独立训练裁判（Gemini 3.1 Flash-Lite, GLM-5.2 fp8, Claude Haiku 4.5，temperature=0），两种提示变体（terse和chain-of-thought）。每个裁判返回最佳候选置于rank-1，其余保持嵌入顺序。不可解析输出回退至嵌入顺序。

**指标与统计**：
- 主要指标：可恢复差距占比 = (Hit@1后 - Hit@1前) / (Hit@10 - Hit@1前)
- Bootstrap 95% CI：10,000次重采样（queries层面），seed分别为12345/54321/42
- Judge比较：配对bootstrap差异（10,000次重采样）
- 部署比较：McNemar检验（per-query hit翻转）
- 下游效用：McNemar检验（配对per-problem正确性）

**下游效用设计**：
- Solver：DeepSeek-v4-flash，32,768 token输出预算
- 210道MathNet问题，三条件：无上下文、刻意坏检索（lexically similar但structurally unrelated）、gold+solution
- 双独立评分器（Gemini 3.1 Flash-Lite主，GLM-5.2稳健性检查），0-7分制，≥6为正确，人评一致性96-99%

**验证门控**：
- MathNet Table 4 easy-tier数值在1 point内复现；轨迹MEDIUM MAP在0.007内复现；超出则停止研究。

## 实验与结果
**数学领域（n=500, 117,088 corpus）**：
- **基线**（表1）：Hard tier严格Hit@1为0.0% [0.0, 0.0]（bootstrap从未含命中），Gemini-emb的Hit@10为55.4% [51.0, 59.8]，Qwen-emb为21.0% [17.6, 24.6]；Lenient Hit@10近100%。结论：检索成功但排名失败。
- **失败结构**：84-98%的miss为query自身的planted near-miss；95.2-99.8%的误命中候选比gold更词汇相似。典型案例：不等式方向翻转的字符相同问题以cosine 0.860排名第1，而真正结构等价的gold在rank 4（cosine 0.821）。
- **词汇控制**：损害检索——Gemini-emb easy tier损失9.1%可恢复差距，Qwen-emb损失4.8%，hard tier无变化（flat）。
- **LLM重排序**（表2，terse prompt）：
  - Gemini-j：Easy 20.6-35.6%，Hard 10.5-44.4%（hard tier反而优于easy tier）
  - GLM-j：Easy 10.1-12.0%，Hard 10.5-18.1%（flat to mildly rising）
  - Haiku-j：Easy 58.1-63.3%，Hard 5.4-6.7%（唯一随伪装加重而降级的裁判）
  - CoT使Gemini-j easy tier翻倍（44.7-55.4%），但hard tier减半（22.7-26.7%）；GLM-j CoT被排除（63.3%截断）
- **污染**（表3）：知名竞赛 vs 其他来源，Gemini-j+Gemini-emb细胞置信区间 [+6.7, +33.2] 不含零，其余五细胞方向一致但 individually indistinguishable from zero。Fisher test四组合均p<0.01。
- **部署分歧**：同权重Qwen3-Embedding-8B的双服务，5/6指标一致，但hard-tier Hit@10存在17 vs 1的discordant flips（McNemar p=0.00014）。

**轨迹领域（n=118, 336 corpus）**：
- **字面锚定证据**（表4）：Definition (i) "不同对象"时产线模型约chance水平；Definition (ii) "不同对象+不同容器"时三嵌入器均低于chance（Qwen-emb 11.0% vs chance 14.1%，Gemini-emb 8.5%，MiniLM 6.8%），表明actively steered away而非indifferent。
- **词汇控制翻转**（核心结果）：帮助检索——Qwen-emb +25.9% [11.3, 41.2]，Gemini-emb +36.4% [23.4, 50.0]，MiniLM +32.1% [18.5, 47.1]，CI均不含零。Verb-only变体复现大部分效果，noun-only贡献微弱(0-12.5%)。
- **LLM重排序**（表5，terse）：排名反转——GLM-j领先（68.5-75.8%），Gemini-j次之（42.6-58.9%），Haiku-j最低（43.9-48.2%）。GLM-j在query风格间稳定(68.3-76.0%)，Gemini-j在新模板化查询上收益骤降至36.6-40.0%（vs 原始paraphrased的61.5-62.5%）。
- **失败分类**：裁判miss中72-88%为sibling miss（同对象同任务类型不同receptacle），远低于嵌入器的结构性混淆。

**下游效用（n=210, DeepSeek-v4-flash）**：
- 准确率在各条件间平坦（67-70%），McNemar none vs gold: p=0.678，none vs dumb: p=0.648
- 困难子集（64/210零样本失败）：gold recover 10题(15.6%)但同时break 13题已解，net = -3；bad retrieval同样net = -3
- **机制**：127/210题完成于预算内，完成率准确率97.6-100%；headline 69.5%为零样本准确率实为截断代理（truncation proxy）
- 知名竞赛子集(30/210)完整答案准确率99.1-100%，与non-famous(97.6%)无显著差异，排除纯记忆解释

## 相关工作脉络
1. **MathNet (Alshammari et al., 2026)**：展示27个嵌入模型数学等价检索失败(Recall@1<5%)，归因于表面重叠；未测试重排序。本文在其基础上补充两阶段评估、现代嵌入器和结构化失败分析。
2. **Kohar & Krishnan (2025) 程序记忆基准**：发现ALFWorld轨迹检索generalization cliff；局限为单一384维编码器、LLM相关性标签低人评一致性(Cohen's kappa=0.178)、两阶段检索留为future work。本文提供完整实现+现代嵌入器+规则推导的exhaustive标签。
3. **Nogueira & Cho (2019) Retrieve-then-rerank**：标准IR范式。本文贡献在于在结构相关性而非主题相关性下评估该范式，并配备针对LLM裁判的控制设计。
4. **Shi et al. (2023) Context that hurts**：无关上下文损害LLM问题求解。MathNet观察到RAG得分低于零样本但未探究。本文通过配对设计+刻意坏条件量化link。
5. **Zheng et al. (2023) LLM-as-judge**：评估可扩展但引入裁判训练分布。本文通过污染分析具体化该担忧，证明部分重排序收益可归因于记忆而非推理。

## 局限性与未来方向
1. 数学实验仅用单一seed，bootstrap CI仅量化样本内不确定性；三个裁判为小样本，效应量八倍跨度表明变异性下限估计。
2. 轨迹领域仅一个数据集族、336-item语料库、118 queries（低于150目标），任务类型标签为规则推断+样本审计而非全量手工标注。
3. 污染归因为相关而非因果；知名竞赛子集可能在training exposure之外还有其他差异；MathNet等价题和decoys由Gemini生成，家族亲和可能贡献Gemini-j优势及单一显著污染cell。
4. 下游null仅针对一个solver且有效余量近零；31%截断率为声明性 Caveat，未随budget增加解决，暗示截断追踪问题难度而不仅是budget。
5. 数学变体和decoys为LLM生成，词汇控制约束但未消除generation-process confounds。

## 研究启发与可借鉴点
1. **低成本诊断工具**：词汇重排序控制器的符号翻转（positive/negative）可作为判断基准测试"对抗性vs偶然性"表面变化机制的廉价诊断——本文团队在设计新基准时可直接采用此控制。
2. **pipeline分层错误分类学**：本文揭示embedder错误为结构性混淆而judge错误主要为伪装失败（sibling/near-miss为主），后续工作可按此框架区分各stage的错误类型再评估改进。
3. **截断审计的必要性**：本文八项完整性事件中最关键的是silent truncation（解析为有效输出），验证门控、finish-reason检查、per-condition截断率报告应作为标准实践。
4. **下游效用的headroom约束**：检索质量→求解性能的transfer受solver"有效余量"制约；评估RAG时应在控制截断后报告complete-answer准确率，否则headline数字可能误导。
5. **精确机会基线的重要性**：轨迹领域通过hypergeometric exact chance baseline揭示below-chance retrieval，证明当gold-set size可变时需报告精确机会而非平均机会。

## 关键术语表
**Structural retrieval**：从共享底层结构（相同数学技术/任务程序）但表面词汇不同的项目中检索，要求gold与query的surface form分离。

**Surface-form bias**：嵌入模型锚定于字面token相似性而非任务结构相似性的系统性偏差，导致结构等价但词汇不同的正确答案被排名压制。

**Recoverable gap**：Hit@10与Hit@1之间的差距，代表当前检索器已检索但未能排第一的候选比例，是重排序器理论上可补充的性能空间。

**Lexical control / sign flip**：朴素词汇重排序器在对抗性表面变化基准中损害检索（负号），在偶然性表面变化基准中改善检索（正号），其符号反映基准构建机制。

**Below-chance retrieval**：检索性能显著低于超几何精确机会基线，表明模型actively steered away而非indifferent于跨对象/容器的结构匹配。

**Strict vs Lenient Hit@k**：Strict要求gold与query共享结构且不共享字面token；Lenient额外接受词汇相似的near-miss，二者差距量化表面相似陷阱的强度。

**Truncation proxy**：当solver在token budget限制下大量截断时， headline准确率实质反映"能否完成推导"而非"求解能力"，造成对检索增益的误判。

**Contamination probe**：通过比较well-known vs other-source候选的重排序收益，检测LLM裁判的记忆效应——显著正向gap表明裁判可能记住了而非推理出答案。

## 可复现要素
- **数据集**：MathNet-Retrieve（公开，500 queries/117,088 corpus）；ALFWorld（公开valid_unseen split，336 trajectories）
- **代码/权重**：代码开源于 https://github.com/nabirarashid/structural-retrieval；raw results、per-query cached responses、spend logs、benchmark correction patches均在仓库
- **关键超参**：Bootstrap重采样10,000次；LLM温度=0；solver输出预算=32,768 tokens；seed=42（数学）；总计实验成本$17.24/4,338次API调用
- **嵌入器服务**：gemini-embedding-001（DeepInfra）、Qwen3-Embedding-8B（DeepInfra+lab-hosted双部署对比）、MiniLM-L6-v2（源发布）

<!--META
{"keywords": ["structural retrieval", "surface-form bias", "LLM reranking", "retrieval evaluation", "embedding models", "math reasoning", "agent trajectories"], "field": "信息检索与RAG评估", "innovations": ["发现嵌入检索在结构检索任务中严重锚定字面surface form而非底层结构，且在轨迹领域表现为below-chance retrieval", "
