---
title: "Token-Eficient-Data-Reasoning-Agents-via-Adaptive-Structurin"
source: https://arxiv.org/pdf/2608.31082v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 06:38:39"
---

# 论文速读：Token-Eficient-Data-Reasoning-Agents-via-Adaptive-Structurin

## 一句话总结
本文提出**代理数据裂纹（agentic data cracking）**机制，在推理Agent打开非结构化文档回答问题时，利用已加载上下文的KV缓存以极低边际成本派生子Agent进行推测性结构提取，使后续相关查询可直接通过结构化读取复用证据，从而在保持高准确率的同时显著降低Agent数据推理的Token与API成本。

## 研究问题与动机
1. **成本瓶颈**：现有Agent处理复杂数据推理任务时需反复加载大型非结构化文档，单问题最高消耗近百万Token、近1美元成本，主要源于Prefill密集型操作。
2. **预结构化不可行**：文档蕴含的实体/属性/关系远超任何单一工作负载所需，全量提取是不定长Decode，且未知哪些结构会被实际复用。
3. **查询驱动的语义局部性**：真实工作负载中，相关问题会反复打开同一文档，每次查询会暴露真正有价值的结构与关系，具备跨查询复用潜力。
4. **推理层的共享前缀机会**：文档载入后KV Cache已存在，通过共享前缀服务与Prompt缓存，第二次生成仅需承担小额Decode开销，为推测性提取提供了经济基础。

## 核心贡献（创新点）
1. **提出agentic data cracking范式**：将数据库Cracking的“查询驱动、增量组织”思想引入AI推理，在文档被打开回答问题时低边际成本地派生子Agent进行推测性结构提取，而非仅缓存答案。
2. **设计证据锚定的扩展RDF数据模型**：裂解对象包含主体、关系、客体、基数（singular/list）、单位及来源文档与文本区域，兼顾结构化复用与可追溯验证。
3. **构建自适应读取与推理协议**：引入目录（catalogue）视图让Agent在打开文档前探查可用主体-关系对，命中则走低成本结构化读取，未命中则保留原始文档作为保守回退路径。
4. **实证验证成本-精度权衡**：在FanOutQA及希区柯克案例研究中，仅扩展单条相关查询即可实现平均成本下降53%（FanOutQA）至67%（案例），第10百分位节省达9倍，且精度无统计显著损失。

## 方法详解
- **裂解对象数据模型（Write）**：每个对象表示为扩展RDF边 $c = \langle s, r, o, \kappa, u, \varepsilon \rangle$，其中 $s, r, o$ 来自开放域；$\kappa \in \{\text{singular}, \text{list}\}$ 区分单边完成还是列表收集（list需全量提取后才可复用）；$u$ 记录规范单位；$\varepsilon = \langle d, \rho \rangle$ 锚定证据来源文档 $d$ 及文本区域 $\rho$。
- **推测性提取（Cracking Sub-agent）**：主Agent打开文档后，子Agent从已加载上下文派生，复用共享Prefix的KV缓存，规避重复Prefill。子Agent不仅提取当前查询缺失的事实，还基于语义推理推测可能服务于未来相关查询的实体、属性与关系，在固定Decode预算（论文设为4K Token）内输出约束JSON。
- **读取/推理协议（Read/Reasoning）**：主Agent新增受限工具调用，支持按主体、关系及主体-关系对进行结构化读取。系统提供目录视图 $\text{Cat}(d) \triangleq \pi_{s,r,\kappa}\left(\sigma_{\text{doc}(\varepsilon)=d}(C)\right)$，使Agent在打开文档前即可探查可用结构；命中则直接返回紧凑证据与溯源，未命中则fallback至原始文档加载。
- **不变量与鲁棒性保障**：要求每次Crack覆盖完整逻辑文档以避免部分数据歧义；结构化读取失败或不完整时自动回退原始文档，确保探索性提取不损害答案精度。

## 实验与结果
- **设置**：基于Claude-Haiku-4.5与FanOutQA官方规范实现基线Agent与ADC系统，开启Prompt缓存。为模拟真实局部性，FanOutQA测试集每条问题前追加一条由claude-opus-5生成并经人工校验的相关问题（重叠实体集、不同属性），构建温蓄库；另设希区柯克电影案例（20条探索问题+10条测试问题）。
- **Token与成本**：FanOutQA平均Prefill从189K降至87K Token，单问题成本从\$0.26降至\$0.12（降幅53%）；案例研究Prefill从565K降至161K，成本从\$0.81降至\$0.27（降幅67%）。Decode始终低于3K Token。
- **准确率**：LLM-as-judge准确率42% vs 43%（p=0.39，无统计显著差异）；词级匹配准确率下降2个百分点，主要源于答案规范化，核心精度保持。
- **分位数收益**：中位数成本比达3.4倍；第10百分位问题节省9倍；90百分位因无复用且需支付Cracking开销，成本为基线的1.24倍。约四分之三问题实现净收益。
- **预算效率**：4K Token裂解预算产生约150个对象，带来12%额外开销；规避单次文档打开即可摊销该开销，后续每少开一次均为净节省。

## 相关工作脉络
1. **OpenIE与知识图谱构建**（OpenIE [7]、DeepDive [33]）：侧重无模式抽取或离线全量构建；本文方法介于两者之间，查询驱动、在线推测、按需提取。
2. **图RAG与数据空间**（Graph RAG [6]、Dataspaces [26]、Query-driven doc analytics [14]）：图RAG预先建图；本文随查询流增量积累，利用语义局部性替代固定模式。
3. **语义缓存**（GPTCache [2]）：仅命中近似重复问题；本文提取的可复用结构支持不同属性/不同问题的查询，泛化性更强。
4. **KV Cache
