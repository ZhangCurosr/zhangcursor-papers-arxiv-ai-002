---
title: "Walking-on-the-DARKSIDE"
source: https://arxiv.org/pdf/2608.23370v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 01:11:20"
---

# 论文速读：Walking-on-the-DARKSIDE

## 一句话总结
本文提出 DARKSIDE，一种基于“否定之路”（via negativa）的连贯性审计层，叠加于 POLANYI++ 之上；通过在 OWL2 扩展知识图谱（XKG）中显式维护排除轨迹（NegativeTrail）与担保轴（Warrant Axis），对 LLM 前向推理的输出进行风险分级（TRUSTLESS/SUPERVISED/UNSAFE），在 BSBench 对抗性无意义问题基准上将模型识别伪装性谬误的覆盖度提升至 1.96/2（97.0% 满分）。

## 研究问题与动机
- **模式-路径结构性缺口**：LLM 擅长局部模式识别，但内部缺乏对连贯论述所需“排除路径”的持续追踪能力（pattern-vs-path gap），导致其在长程或迷宫型任务中易产生漂移。
- **伪专业输入的结构性脆弱**：面对语法严谨、框架权威但语义虚构的问题（如 BSBench 中的虚假框架/权威），未经脚手架引导的 LLM 倾向直接回应而非质疑前提，将谬误重构为结构化输出。
- **现有 LAG 方法继承相同缺陷**：POLANYI++ 虽能通过启发式与本体现在驱动单次前向推理生成 XKG，但错误假设与合法三元组被联合产出，自动化推理器难以检测，缺乏对“未提及/已排除”内容的显式追踪。
- **否定工作（negative work）的外部化缺失**：人类在委托 LLM 执行复杂任务时需持续补偿路径追踪成本，现有系统缺乏可机器解析的排除累积机制，导致幻觉在代际传递中放大。

## 核心贡献（创新点）
1. **DARKPOLANYI 功能化规范**：将 heuristics、methods 与 ontology priors 作为一等公民输入编译为 OIS 以驱动单次 LLM 前向推理；与已有 LAG 方法的本质区别在于将“否定审计”显式嵌入同一路径，而非事后纠错。
2. **OWL2 否定审计词汇表**：形式化 NegativeTrail、ConstancyViolation、Encapsulated/Vain Contradiction、LabyrinthSignature 与 DelegationRiskAssessment；与已有本体设计模式的本质区别在于将其直接映射到主动推理框架（NegativeTrail = 生成模型，ConstancyViolation = 预测误差）。
3. **担保轴（Warrant Axis）作为认识论防火墙**：对输入中的每个命名指代表征进行四类分类（Warranted/Unattested/Misattributed/Fabricated）并设确定性升级规则；与既有事实性分类体系的本质区别在于它是针对“用户输入承诺”的先行分类，且在升级前不依赖外部知识检索。
4. **三模式审计架构（Mode A/B/C）**：Mode A 后验审计、Mode B 元审计、Mode C 在线预提交约束；与已有幻觉检测方法的本质区别在于核心是追踪“排除累积”而非单纯比对事实，Mode C 的 BLOCK 动作可自动触发 REPHRASE/RETRACT 等修复策略。

## 方法详解
- **POLANYI++ 基座函数**：E = f(I, O, S, B, H, T, M, U)，包含三阶段编排器：Phase A（LLM 打分生成 YAML 报告，用 Gemini 3 Flash）、Phase B（纯 Python 汇编器解析 YAML 与带插入点的指令文件，生成任务特定 OIS）、Phase C（单次 LLM 前向推理，以 OIS 为系统提示输出 XKG，用 Gemini 3 Pro）。XKG 在 `rdfs:comment` 中嵌入包含 Executive Semantics、Hidden Dynamics、Strategic Anomalies 与 Methodological Outputs 的可读叙述。
- **NegativeTrail 数据结构**：`dark:NegativeTrail` 作为话语时间维度的排除累积器；每个 `dark:Exclusion` 是对“下一内容不会出现”的预测，`exclusionStrength` 对应主动推理中的精度（逆方差）；新承诺 cⱼ 与已累积排除内容 eₖ 冲突且无 PERSPECTIVE/HYPOTHETICAL 正当理由时触发 `dark:ConstancyViolation`。
- **担保轴与升级规则**：OIS 注入 `REIFICATION_MANDATE_WARRANT`，强制将输入命名指代表征化为带 `dark:warrantSource`、`dark:warrantNote`、`dark:warrantConfidence` 的个体；聚合为 `dark:WarrantProfile`，计算 fabricatedRate 与 unsupportedRate；升级规则：`if fabricatedRate > 0 OR unsupportedRate > 0.40: escalate DelegationRiskAssessment to UNSAFE`。
- **矛盾分类五重测试**：对每个 ContradictionEvent 运行测试判断是否为 Encapsulated（悖论、不可靠叙述者、反讽、修辞、荒诞）；默认为 Vain。
- **主动推理形式化对齐**：NegativeTrail = 生成模型；Prediction = Exclusion；PredictionError = ConstancyViolation；PathIntegrity = 变分自由能（accuracy vs complexity 分解）；Mode C 的 PreCommitCheck 构成一次完整推断循环（观察-预测-比对误差-更新后验-行动）。

## 实验与结果
- **数据集**：BSBench，100 个跨软件/DevOps（≈46%）、金融/会计（≈16%）、物理（≈13%）、法律/合规（≈11%）、医疗（≈10%）的伪专业无意义问题，每项标注 13 类谬误技术与金标准理由。
- **评估设置**：两组对照。pol_gem（DARKPOLANYI，Phase C 用 Gemini 3 Pro）；gem_baseline（裸 Gemini 3 Pro）。独立评判器为 Claude Sonnet 4.6（temperature=0.0），使用单轴 0–2 覆盖度量规（0=完全遗漏，1=部分提及，2=明确指出核心谬误）。评判文本仅限 XKG 中的自然语言合成部分，不含 OWL 图本身，避免格式偏好污染。
- **主要结果**：DARKPOLANYI 在 99 个有效样本上均分 1.96/2，完美识别率 97.0%（96/99）；基线 Gemini 3 Pro 在 94 个有效样本上均分 1.84/2，完美识别率 91.5%
