---
title: "Unseen-Harm-Measuring-Cross-Script-Safety-Inconsistency-with"
source: https://arxiv.org/pdf/2608.24191v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 11:16:42"
field: "多语言 NLP 安全评估"
keywords: ["仇恨 speech 检测", "跨脚本一致性", "乌尔都语", "LLM 安全评估", "低资源语言", "Missed-in-Urdu", "WOAH"]
innovations: ["提出 Missed-in-Urdu 指标量化跨脚本安全盲区", "六数据集四条件跨脚本 LLM 评估框架与统计验证体系", "首个 WOAH 十年文献中乌尔都语覆盖度全面审计"]
benchmarks: ["HateInsights", "Cyberbullying", "Abusive Tweets", "HS-RU-20", "RU-EN Emotion", "HateXplain"]
---

# 论文速读：Unseen-Harm-Measuring-Cross-Script-Safety-Inconsistency-with

## 一句话总结
本文对五个主流 LLM 进行了跨脚本安全评估，揭示当同一内容以乌尔都语 Nastaliq 脚本或英文翻译呈现时，模型分类结果存在系统性不一致——中位数"标签不稳定性"达 18.0%，中位数"Missed-in-Urdu"率为 4.3%，即模型能识别英文中的仇恨内容却对乌尔都语原文视而不见。

## 研究问题与动机
- **WOAH 文献覆盖盲区**：乌尔都语（全球第十大语言，2.46 亿使用者）在 ALW/WOAH 会议十年间（2017–2025）零篇专题论文，而 WOAH 6（2022）曾明确呼吁关注乌尔都语。
- **脚本多样性带来评估挑战**：乌尔都语内容在社交媒体上以 Nastaliq 脚本、Roman Urdu（拉丁转写）、Urdu-English 代码切换等多种形式出现，不同脚本可能触发不同的模型行为。
- **跨脚本一致性缺失的影响**：现有仇恨检测研究主要面向英语，若模型仅在翻译形式上识别伤害而忽略原始脚本，将导致大规模用户群体的安全防护漏洞。
- **小规模开放模型风险更高**：假设开源/小参数模型比前沿闭源模型对乌尔都语脚本更不鲁棒，这一假设尚未被系统验证。

## 核心贡献（创新点）
- **WOAH 文献全面审计**：通过 ACL Anthology API 枚举全部 205 篇论文，量化乌尔都语代表性与全球十大语言地位的落差，填补社区资源分布的实证空白。
- **提出"Missed-in-Urdu"指标**：定义并测量"模型在英文翻译中标记为有害、但在原始乌尔都语脚本中当作正常"的比例，揭示脚本条件化安全行为的系统性偏差。
- **六数据集跨脚本四条件实验设计**：在 Nastaliq 原文（C1）、英文翻译（C2）、Roman Urdu 转写（C3）、代码切换（C4）四个条件下评估五个 LLM，隔离脚本、转写、代码切换效应。
- **统计验证框架**：使用 McNemar 检验、卡方检验与 Stuart-Maxwell 检验（Bonferroni 校正）证明观察到的差异具有统计显著性，而非随机噪声。
- **资源缺口识别**：首次明确指出不存在专用 Nastaliq Urdu-English 代码切换仇恨数据集，并将该结构性缺失作为研究发现本身。

## 方法详解
- **数据集与标注体系统一**：六个数据集（HateInsights、Cyberbullying、Abusive Tweets、HS-RU-20、RU-EN Emotion、HateXplain）的原始标签被统一映射为三分类：Hate、Offensive、Normal，以保证跨数据集可比较。
- **四条件实验设计**：
  - C1（原始 Nastaliq）：模型直接对乌尔都语原文分类。
  - C2（英文翻译）：由同一模型将 C1 文本翻译为英文后再分类，隔离模型自身翻译一致性与分类一致性。
  - C3（Roman Urdu 转写）：使用 HS-RU-20 提供的拉丁转写形式，隔离转写效应。
  - C4（代码切换代理）：使用 RU-EN Emotion 语料（Roman Urdu-English 混合），以情感标签映射危害标签作为代理。
- **两个核心指标**：
  - **Label Instability**：C1 与 C2 分类结果不一致的样本比例。
  - **Missed-in-Urdu**：C2 中标为 Hate/Offensive 但 C1 中标为 Normal 的样本比例，代表"漏过的伤害"。
- **统计检验**：配对二分类结果用 McNemar 检验；双向非对称比较用卡方检验；三分类分布偏移用 Stuart-Maxwell 检验；多重比较采用 Bonferroni 校正（α = 0.005）。
- **提示模板标准化**：所有模型使用相同系统提示进行零样本分类，拒绝响应（REFUSED）行被排除。

## 实验与结果
- **数据集规模**：五个乌尔都语脚本数据集共约 4,396 条唯一文本，每个模型有效评估样本约 4,531–4,553 条。
- **标签不稳定性**：前沿模型（GPT-4o、Claude Sonnet 4.5、Gemini 2.5 Flash）介于 15.9%–18.0%；开源模型（Llama-3.1、Qwen-2.5）高达 27.3%–31.6%，约为前沿模型的两倍。
- **Missed-in-Urdu 率**：GPT-4o（2.4%）最低，Qwen-2.5（9.9%）最高，中位数 4.3%（SD = 2.9%）。
- **HateXplain 控制**：英语数据集上两项指标均为 0.0%，排除流水线误差。
- **统计显著性**：所有前沿 vs 开源模型对比均高度显著（p < 0.0005）；Stuart-Maxwell 检验确认所有模型的 C1-C2 标签分布均发生显著偏移（p < 0.01）。
- **最强结果**：GPT-4o 在两项指标上表现最优，Missed-in-Urdu 仅 2.4%，显著低于 Claude Sonnet 4.5（p < 0.0005）。

## 相关工作脉络
- **Ghorbanpour et al. (2025)**：评估 LLM 在八种非英语语言的仇恨检测，覆盖五大自然语言但遗漏乌尔都语，本文补充其空白。
- **Chan et al. (2024)**：证明翻译对代码混合内容无效，支持本文"必须在原始脚本上评估"的核心论点。
- **Nozza (2021)**：发现零样本跨语言迁移在低资源语言上失败，为本研究的跨脚本设计提供动机。
- **Ahmad et al. (2025)**：乌尔都语仇恨检测的唯一相关工作，但仅评估单一模型和单一数据集，未涉及跨脚本一致性。
- **Dey et al. (2024)**：在三种南亚低资源语言上发现"翻译到英语再分类"优于原始语言提示，本文将其结论限定在单语言条件并揭示跨脚本风险。
- **Vargas (2024) / HausaHate**：豪萨语仇恨研究提供结构类比，表明低资源语言差距是系统性而非孤立问题。

## 局限性与未来方向
- 数据集规模有限（每集 700–1,000 条），部分数据因运行中断未达目标。
- C2 翻译由被测模型自身完成，无法区分翻译质量与分类一致性。
- 1,216 条模型与黄金标签不一致的样本未经人工审查，无法区分模型失败还是标注偏差。
- RU-EN Emotion 以情感标签代理危害标签，仅为近似。
- ACL Anthology 审计仅基于标题和摘要，可能遗漏未提及乌尔都语但实际使用其数据的研究。
- 所有模型通过商业 API 访问，版本不可复现。
- **未来方向**：构建专用 Nastaliq Urdu-English 代码切换仇恨数据集；对模型-标注分歧进行定性审查；探索脚本鲁棒性对齐训练方法。

## 研究启发与可借鉴点
- **"Missed-in-X"指标设计可迁移**：该指标概念可推广至其他低资源语言（如孟加拉语、斯瓦希里语等），衡量模型在原文与翻译间的系统偏差。
- **四条件实验设计范式**：原文→翻译→转写→代码切换的控制变量框架，适用于任何多脚本/多语言变体的安全评估研究。
- **统计检验组合策略**：McNemar + 卡方 + Stuart-Maxwell + Bonferroni 校正的完整显著性验证链路，可作为多模型跨条件比较的标准报告模板。
- **HateXplain 控制集思路**：用同语言但不同脚本的英语数据集验证流水线一致性，可有效排除非语言因素的评估噪声。
- **WOAH 文献审计方法**：通过 ACL Anthology API 枚举标题/摘要的关键词匹配策略，可快速评估任意语言在特定会议上的代表性。

## 关键术语表
- **Missed-in-Urdu**：模型在英文翻译中识别为有害但在原始乌尔都语脚本中判定为正常的内容比例，衡量脚本条件化安全盲区。
- **Label Instability**：同一内容在不同脚本条件（C1 vs C2）下获得不同分类标签的样本比例。
- **Nastaliq Urdu**：乌尔都语的传统书法脚本，向右下方倾斜，与拉丁转写形式（Roman Urdu）视觉差异显著。
- **Roman Urdu**：使用拉丁字母拼写的乌尔都语，常见于巴基斯坦社交媒体。
- **Stuart-Maxwell 检验**：McNemar 检验的三分类扩展，用于检验配对条件下三类标签边际分布是否发生显著变化。
- **C1/C2/C3/C4 实验条件**：分别指原始 Nastaliq 乌尔都语、英文翻译、Roman Urdu 转写、Urdu-English 代码切换四种评估条件。
- **Frontier Models**：指 GPT-4o、Claude Sonnet 4.5、Gemini 2.5 Flash 等顶级闭源模型。
- **Open-weight Models**：指 Llama-3.1-8B 和 Qwen-2.5-7B 等开源权重模型。

## 可复现要素
- **数据集**：全部六个数据集公开可获取（HateInsights、Cyberbullying、Abusive Tweets、HS-RU-20、RU-EN Emotion 需注册，HateXplain 公开）。
- **代码**：匿名代码仓库 https://anonymous.4open.science/r/WOAHJul26-047C（审稿期间），录用后将公开。
- **模型访问**：通过 OpenAI、Anthropic、Google、Together.ai 和 Groq API 调用，非本地部署。
- **关键超参**：零样本分类，无 in-context examples；Bonferroni 校正 α = 0.005（10 次比较）；Python 环境，Windows 机器，API 限流为主要约束。
