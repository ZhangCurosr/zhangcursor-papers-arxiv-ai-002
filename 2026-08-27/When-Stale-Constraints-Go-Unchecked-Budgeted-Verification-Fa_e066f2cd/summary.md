---
title: "When-Stale-Constraints-Go-Unchecked-Budgeted-Verification-Fa"
source: https://arxiv.org/pdf/2608.25553v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 19:00:24"
field: "LLM Agent Memory Safety"
keywords: ["agent memory", "verification budget", "provenance", "stale constraints", "LLM agents", "temporal validity", "budgeted reasoning"]
innovations: ["首次因果识别预算受限代理在过时约束下的可避免错误比例", "提出Ceiling-bound评估框架量化分配vs推理错误", "OpenTimestamps+Bitcoin锚定的零篡改实验注册流程"]
benchmarks: ["Nakayashiki 2026 growth scenario", "procurement held-out domain"]
---

# 论文速读：When-Stale-Constraints-Go-Unchecked-Budgeted-Verification-Fa

## 一句话总结
论文研究了在有限验证预算下，AI代理继承包含过时约束的记忆时能否发现并纠正该错误，发现代理倾向于将验证资源分配给其他记录而非关键溯源路径，导致约77%的场景下仍遵循过时记忆；但将其中一个验证槽强制分配给关键溯源路径可将正确决策率提升约74个百分点，证明大部分错误可通过优化验证分配策略避免。

## 研究问题与动机
1. **核心问题**：当代理继承的存储记忆中的约束已被更新的权威记录废止时，在有限的验证预算下，代理能否通过检查溯源路径发现这一废止？若不能，这种错误是否可通过重新分配验证预算来避免？
2. **现有方法的不足**：之前的研究已测量了代理在验证预算下如何将检查集中在已持有的计划所支持的记忆上，但所有约束都是真实的，因此无法评估当约束过时时的危害；现有工作未区分"验证去向"与"验证后是否导致正确决策"这两个问题。
3. **安全动机**：持久化记忆系统中的可审计性承诺并未转化为实际的审计行为，代理只检查少量溯源链接后就行动，这种分配决策在推理时发生，可能危及系统安全。
4. **理论缺口**：现有研究假设失效证据已在上下文中或可由存储层处理，但没有研究改变预算受限代理选择检查哪条记录这一变量。

## 核心贡献（创新点）
1. **首次直接测量过时继承溯源在固定验证预算下的可避免错误比例**：通过随机化验证策略而非仅观察分配模式，识别出分配策略对下游错误的因果效应，证明约74%的错误本可通过预算内重新分配避免。
2. **分离了"相关性检索"与"成本评估"两个决策层**：证明代理最相关的记忆（陈述对诱惑行动的约束）恰恰是代理选择不去检查其溯源路径的记忆，揭示了相关性与验证必要性之间的不对称。
3. **提出了结构化 Ceiling 概念用于评估干预效果**：将干预效果的理论上限定义为 `100×(1−Pr(V|native))`，即原生分配下未检查目标路径的概率，发现实际改善幅度接近该理论上限。
4. **开发了可验证的实验框架**：使用OpenTimestamps在Bitcoin区块中锚定实验规范哈希，实现从零篡改的完整实验记录，为可复现性研究树立新标准。

## 方法详解
1. **世界与仪器设计**：复用六记忆增长场景（Nakayashiki, 2026），代理继承六条包含ID和整合日期的单行记忆，面临指标下降的情境，有五个候选行动，验证预算k=2条源记录；目标记忆(memory 73)涉及针对折扣的约束，其源记录报告保留损失和禁止重用。
2. **时间结构建模**：`t₀=day31`写入源记录S₀并从中整合记忆；`t₁=day68`（仅在废止世界）写入更新的权威记录S₁并标记为替代S₀，记忆不重新整合；`day76`代理继承存储并分配预算；`day79`做出决策。历史溯源不变，变更的是当前记录状态。
3. **验证请求机制**：每次验证请求命名一个记忆ID，存储返回历史源记录S₀原文、状态行（"current — no superseding record on file"或"superseded on day t₁ by S₁"），以及若S₀被替代则在其下方返回S₁；发现更正只需一次请求成本。
4. **三因素完全区组设计**：(1)记忆形式F：stated（含正证据+量化负面结果+禁止条款）或removed（相同正证据+中性量化事实+中性条款）；(2)世界W：valid（S当前）或superseded（S₁废止约束）；(3)验证策略P：native（返回代理自选的两条记录）、forced-critical（目标路径+代理首个其他选择）、forced-noncritical（种子随机非目标记录+代理首个选择）。
5. **结果度量**：V=目标路径在turn-1被命名；R=目标记录被返回；Y=turn-2决策与存储标记的当前记录一致（在superseded世界中选择不执行被约束的行动）。主要估计量为forced-critical与native之间Y的风险差异，使用模型分层bootstrap 95%区间(B=4,000)。
6. **实验规模与注册**：2×2×3=12单元×6模型×25=1,800集数/主实验；另含1,800集数复制实验、900集数持有领域实验、900集数修正后稳健性复制；全部5,400集数确认性实验；实验规范在首个确认性集数前96秒通过OpenTimestamps时间戳锚定于Bitcoin区块。

## 实验与结果
1. **原生分配极少选择陈述约束的溯源路径**：约束存在时，代理在primary run中仅20.1%(181/900)、replication run中23.1%(208/900)的episode中命名目标溯源路径；而当约束被移除时，该率升至66.9%和72.9%，差距约+47至+50个百分点。
2. **约束过时导致高错误率**：在superseded世界中，native分配下决策遵循过时记忆而非当前记录的占比达77.3%(primary)、74.7%(replication)、74.7%(procurement held-out)。
3. **强制关键路径大幅减少错误**：在primary run中，强制分配一个槽到关键溯源路径使当前记录一致决策从34/150提升至145/150，增加+74.0点[68.0, 80.0]，理论Ceiling为78.7点，实际接近上限；replication run提升+72.7点，procurement run提升+61.3点，修正后稳健性run提升+73.3点。
4. **六模型全部受益**：在每个run中，六个模型（Claude Opus 5、Sonnet 5、Haiku 4.5、GPT-5.6 Sol、Terra、Luna）均显示正向改善（5/6或6/6模型）；效果幅度与每个模型的原生验证率相关，Sonnet 5原生验证率高则改善小(+16点)，Opus 5原生验证率低则改善大(+88点)。
5. **约束与记录一致时无干预效果**：在valid世界（记录确认陈述约束），强制关键路径仅改变决策0.7至2.0点，证明干预效果取决于被检查记录的内容而非检查行为本身。
6. **持有领域发现的上下文不一致**：原始procurement run存在时间不一致（合同"now expires in 3 days"与源记录"onboarding 6 weeks"及day-71文本"expires in 14 days"矛盾），修正后稳健性replication将句子改为"expires in 11 days"，效果从+61.3点回升至+73.3点，验证结果稳健。

## 相关工作脉络
1. **Nakayashiki [2026]（同作者先作）**：测量了验证预算下代理如何将检查集中在已持有计划的记忆上，但未测试约束过时的情况；本文复用同一仪器但回答不同问题——不是"验证去哪儿"而是"验证去哪儿在world移动后是否造成可避免的错误"。
2. **Wu et al. [2025], Hu et al. [2025], Uddin et al. [2026]（LongMemEval等基准）**：研究知识更新、冲突解决和遗忘任务，但失效证据已在context中或可由store处理，未改变预算受限代理选择检查哪条记录这一变量。
3. **Chao et al. [2026]（STALE基准）**：测试LLM代理是否能感知记忆何时失效，但同样假设无效化证据可用，未涉及验证预算分配问题。
4. **Yadav [2026], Zhou et al. [2026]（TEPA等）**：研究存储层的时效有效性或废止记录撤销，将信号置于检索层，本文测量该信号未能到达代理验证选择时的成本。
5. **Lin et al. [2026]（BAGEN）、Fang et al. [2026]、Wang & Xu [2026]（AllocBench）**：研究代理是否合理花费工具调用预算，本文关注记忆溯源路径的验证预算分配而非工具调用。
6. **Fei et al. [2026]**：论证每条记录的provenance在selection layer被 compromising时不足，本文的selection layer是代理在已呈现记录中的预算选择，二者层面不同。

## 局限性与未来方向
1. **合成受控环境**：仅六记忆存储、两个脚本化世界、单次请求返回替代记录的归档语义；真实存储更大且发现可能需多跳。
2. **人为安装过时状态**："当前真理"由归档标记定义，替代记录的内容为世界的固定特征，非自然产生。
3. **未测量 prevalence**：未展示陈述约束在实际部署管道中过时出现的频率，也未研究自然整合是否产生过时状态。
4. **干预是 oracle**：forced-critical使用实验者对关键路径的知识，识别可避免错误比例而非可部署的验证策略。
5. **仅六模型两个提供商**：方向一致性6/6强于幅度一致性；不同模型在不同世界的原生验证率差异大（同一模型在growth world验证率0-12%，在procurement world验证率80-100%）。
6. **任务与schema特定性**：单一system prompt、JSON schema、归档消息格式； Twelve wording families和两个domain仅覆盖记忆文本。
7. **未建立中介机制**：代理的turn-1选择是post-treatment；设计识别策略效应而非通过选择的路径。
8. **未隔离原生低估验证的机制**：未操作置信度、计划或显著性来解释为何代理不检查关键路径。
9. **非关键控制设计受限**：在部分episode中丢弃代理自身的目标选择，结果不包含在主结论中。
10. **未来方向**：开发可从可观测元数据预测原生低估验证路径的策略（如新鲜度/替代信号），而非依赖实验者知识；研究更大规模真实存储中的多跳发现成本；测量实际部署中过时约束的出现频率。

## 研究启发与可借鉴点
1. **预算约束下的因果识别设计**：通过固定预算、随机化验证策略（native vs forced-critical），在观测相同budget下识别分配策略对错误的因果效应，而非仅描述性测量；可迁移至其他agent工具调用/检索预算研究。
2. **Ceiling-bound 效果评估框架**：将干预效果的理论上限定义为 `100×(1−Pr(未检查关键路径))`，使实验结果可与结构性上限比较，判断错误是分配问题还是推理问题；适用于各类verification budget研究。
3. **上下文不一致的审计方法**：在held-out run完成后通过read rationales发现文本矛盾，并设计pre-specified robustness replication修复后再运行，保持原始结果不动并并列报告；可作为可复现研究的标准流程。
4. **OpenTimestamps+Bitcoin锚定的实验注册**：将完整实验规范哈希后提交OpenTimestamps日历并锚定于Bitcoin区块，实现零篡改承诺；可为其他ML实验的可复现性提供技术模板。
5. **相关性-验证必要性不对称的发现**：证明最相关的记忆恰恰是最不被验证的，这一不对称性源于约束本身产生的"settled"感知；可启发设计freshness/supersession信号，将其与semantic relevance分离。

## 关键术语表
**Supersession（替代）**：新权威记录S₁在某日期取代旧记录S₀成为current record，但历史溯源链接M→S₀保持不变且不可修改。

**Stale memory（过时记忆）**：记忆的源记录已被替代记录废止其内容，但记忆本身未重新整合；过时性是(记忆内容, 当前记录)对的属性，非溯源链接的属性。

**Verification budget（验证预算）**：代理在推理时可查询的源记录数量上限，本文为k=2，代理在turn-1命名最多两个记忆ID，archive返回对应记录。

**Provenance path（溯源路径）**：从记忆M指向其源记录S₀的历史链接，是唯一能通过验证请求获得替代记录S₁的路径。

**Native allocation（原生分配）**：archive返回代理在turn-1自行命名的两条记录，代表代理无干预下的验证选择。

**Forced-critical（强制关键路径）**：干预策略，将两个验证槽中的一个替换为目标记忆的溯源路径，另一个保留代理的首个选择，预算不变。

**Current record cur(S₀)**：若S₀存在替代则返回替代记录S₁，否则返回S₀本身；代理通过验证请求可获得此信息。

**Risk difference（风险差异RD）**：forced-critical与native之间当前记录一致决策Y的差值，为主要估计量，使用模型分层bootstrap 95%区间估计。

## 可复现要素
- **数据集**：合成生成的六记忆场景，两个世界（growth decline + procurement）；非公开基准数据集，由实验脚本生成
- **代码**：完整释放，包括episode文件、SHA256 manifest、OpenTimestamps证明、frozen analysis scripts、independent recomputation scripts、generator script；每数字均由脚本发出，无手动输入
- **权重**：使用商业模型API（Claude Opus 5/Sonnet 5/Haiku 4.5 via Anthropic Messages API；GPT-5.6 Sol/Terra/Luna via OpenAI Responses API），未训练自有模型
- **关键超参**：验证预算k=2；bootstrap B=4,000；client timeout 120s；并发8；reasoning effort medium（GPT模型）；strict JSON schema
- **外部注册**：OSF项目axsnm（主/复制/原持有out）、hdm75（修正持有out）；Bitcoin区块964062和964064锚定
