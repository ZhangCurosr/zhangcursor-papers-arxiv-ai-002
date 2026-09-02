---
title: "Predicting-Program-Exit-Code-with-LLMs-and-Programming-Langu"
source: https://arxiv.org/pdf/2609.00579v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 12:13:12"
field: "LLM程序语义理解与评测"
keywords: ["Program Executability Prediction", "Operational Semantics", "K Framework", "KeywordSwap", "KeywordObf", "Code LLMs", "Semantic Shifts", "PLSemanticsBench"]
innovations: ["提出PrEx任务以简洁可执行性判断评估LLM对显式形式语义的规则遵从", "构建带五类语义错误的配对C*数据集并在两种形式化下系统评测", "设计KeywordSwap与KeywordObf语义偏移以分离预训练先验与给定规则应用"]
benchmarks: ["PrEx", "PLSemanticsBench", "CRUXEval", "REval"]
---

# 论文速读：Predicting-Program-Exit-Code-with-LLMs-and-Programming-Langu

## 一句话总结
本文提出程序可执行性预测（PrEx）任务，通过向LLM提供形式化语义规则并要求其判断C*程序是否可执行及违反的具体规则，揭示当前开源代码LLM严重依赖预训练先验而非系统性应用给定语义，且在语义偏移与程序复杂度上升时性能显著下降。

## 研究问题与动机
- LLM在代码生成、翻译等软件工程中表现优异，但其是否真正理解程序底层语义仍不明确；现有能力更多源于模式匹配与数据记忆，而非逻辑模拟执行。
- 既有工作（如PLSemanticsBench）任务复杂度高，模型普遍低分，无法区分失败源于多步推理复杂度还是语义推理的根本缺陷；需要更基础的基线任务分离追踪生成与语义判断。
- 缺乏系统评估LLM是否在给定显式语义下遵循规则、还是退回预训练先验的受控实验设置（如改变运算符含义或使用新符号）。
- 需要覆盖不同来源与复杂度的程序（人工编写、LLM翻译、Fuzzer生成）及多种语义形式化（细粒度小步语义S与粗粒度K框架），以检验语义推理的泛化能力。

## 核心贡献（创新点）
- 提出PrEx（Program Executability Prediction）任务：给定程序与形式语义，预测执行成功或语义错误，并在失败时指出违反的具体规则，输出简洁但评估维度丰富。
- 构建带语义错误标注的C*数据集：在PLSemanticsBench有效程序基础上，通过五种语义感知变换生成一一对应的无效变体，覆盖break/continue越界、除零、模零、未声明变量等错误类型。
- 设计两种语义偏移条件KeywordSwap与KeywordObf：前者交换熟悉运算符语义，后者用新单token符号替换标准符号，强制模型依赖给定规则而非先验关联。
- 在两种语义形式化（S小步语义与K框架重写语义）与三类程序拆分（Human-Written、LLM-Translated、Fuzzer-Generated）上进行系统评测，量化LLM对给定规则的遵从程度。
- 揭示LLM的系统性失败模式：在标准设置下对短程序仍会误判，且在语义偏移与复杂度提升时假成功与错规则激增，表明模型主要依赖预训练先验。

## 方法详解
- 语言与语法规则：使用C*小命令式语言（类C语法），以EBNF给出完整语法，避免真实语言的歧义与复杂性，便于形式化语义映射。
- 语义形式化S（小步操作语义）：采用Gentzen风格推理规则，每条规则表示原子计算步骤，显式处理操作数归约、变量查找、除法/模零等，细粒度追踪执行状态。
- 语义形式化K（K框架）：基于重写规则，以较大块变换程序配置，内置重写机制隐式处理中间表达式归约，规则颗粒度较粗但语义等价。
- 语义偏移构造：KeywordSwap将运算符语义互换（如+改为减法），测试模型覆盖先验的能力；KeywordObf用新颖单token符号替代标准符号，消除预训练词形线索，同时控制token开销。
- 提示结构：每个样本输入包含C*语法、所选语义形式化规则集与待测程序；要求输出##success##或##error##，并在错误时给出违反规则编号。非推理模型比较直接回答与CoT，推理模型仅用默认CoT。
- 错误变换流程：基于ANTLR解析器-访问者，对每个有效程序随机应用五种变换之一，保证每个无效程序仅违反一条语义规则，形成一一对应的有效/无效配对。

## 实验与结果
- 数据集规模与划分：491个有效C*程序，生成2455个无效程序；三类拆分Human-Written（中位19行/81 token）、LLM-Translated（中位106行/538 token）、Fuzzer-Generated（中位786行/9081 token），复杂度递增。
- 模型基线：Qwen、DeepSeek、Ministral系列多尺寸开源代码/推理模型，含非推理与CoT变体；每种{S,K}×{Standard,KeywordSwap,KeywordObf}×{non-CoT,CoT}配置运行三次取均值。
- 最强Human-Written结果：DeepSeek-Qwen 32B、Ministral 3 14B-CoT、Qwen2.5-Coder 32B-CoT在标准S/K语义下最高达99%（如DeepSeek-Qwen 32B在K/Standard为99%），但整体对给定规则的遵从仍不稳定。
- 语义偏移下降显著：Human-Written下KeywordSwap中位下降19pp、KeywordObf中位下降32pp；最强模型平均下降约15–23pp，最大单配置下降达59pp（Ministral 3 14B在K/KeywordObf由90%降至32%）。
- 复杂度下降更剧烈：相较Human-Written，LLM-Translated平均下降2.9–8.7pp，Fuzzer-Generated平均下降13–33pp；最大单配置下降55pp（Qwen2.5-Coder 14B-CoT在K/KeywordObf由78%降至23%）。
- 错误类型差异：关键字相关错误（continue-outside-loop、break-outside-loop）在KeywordObf下退化更明显；Divide-by-zero在KeywordSwap下因先验偏差而下降；所有可胜任模型在Fuzzer拆分上均出现13–40pp均值下降（中位24pp）。
- 主要失败模式：Human-Written以错规则为主（同错误族内相邻规则混淆，如Rule 73与76）；LLM-Translated与Fuzzer-Generated假成功显著增加，且malformed输出增多，说明模型常忽略给定语义而依赖代码表层模式。

## 相关工作脉络
- PLSemanticsBench及其PredState/PredRule/PredTrace任务：侧重多步执行追踪与状态推理，复杂度较高；本文以PrEx为更基础的可执行性判定点，分离追踪与判断，并以语义偏移检验规则遵从。
- LLM作为预测执行器的研究（如NExT、PredEx、CodeFlow、INTERPRETER类工作）：通过训练、程序分析或图模型提升隐式语义预测，但未在提示中显式供给形式规则，难以判断模型是否真正应用给定语义。
- 代码执行与运行时行为评测（CRUXEval、REval、Horà预测测试用例通过/失败）：面向熟悉Python语义的真实程序，评估输出或中间行为，无形式规则对照与语义偏移设置。
- 标识符与命名敏感性研究（如Wang等表明替换名称会显著降低CodeBERT性能）：支撑本文KeywordSwap/KeywordObf动机，证明模型易受表层符号影响而非纯逻辑推理。
- 形式验证与终止性研究（如SV-Comp上LLM接近专门验证器但难产机器有效见证）：与本文共同指向LLM在严格语义与形式属性上的脆弱性，但PrEx提供更可控的规则跟随评测。
- 死代码消除与程序分析辅助（DCE-LLM等）：利用LLM进行局部判断与修补，依赖模型内化语义；本文强调在未内化情况下显式规则供给时的实际遵从度。

## 局限性与未来方向
- 仅评估C*这一小型教学语言，结论向真实工业语言与大规模程序的推广仍需验证。
- 任务聚焦单一可执行性判断与规则指认，未覆盖更丰富的运行时行为（如覆盖率、路径、增量一致性）与多步解释生成。
- 语义偏移只覆盖符号替换层面，未系统考察控制流宏变换、嵌套深度或跨域语料分布漂移的影响。
- 评估以准确率与规则指认为主，缺少对模型推理过程的结构化分析（如CoT逻辑链质量、错误分类学动态演化）。
- 未来可在更大规模、多语言语义上扩展PrEx，结合微调/检索增强以检验能否显著提升对给定规则的遵从，并探索模型在语义偏移下的失效边界与可纠正性。

## 研究启发与可借鉴点
- 以显式形式规则与受控语义偏移评估LLM的“规则遵从”而非仅看表面准确率，可作为后续工作检验模型是否真正使用提供知识的通用评测范式。
- 语义偏移设计（KeywordSwap/KeywordObf）对识别模型先验依赖具有迁移价值，可延伸至其他编程任务（如代码翻译正确性、静态分析）的鲁棒性评测。
- 配对有效/无效样本并绑定单一错误规则的数据构造策略，能精确分解错误类型，便于定位模型薄弱环节与制定针对性改进。
- 多形式化对比（细粒度S与粗粒度K）提示规则颗粒度可能影响模型遵循程度；后续可研究何种形式化表达最利于LLM解析与应用。
- 分层拆分（人工/LLM翻译/Fuzzer）与复杂度递增设置可用于绘制模型能力曲线，为训练数据合成与课程学习提供依据。

## 关键术语表
- **PrEx（Program Executability Prediction）**：给定程序与形式语义，预测程序执行成功或语义错误并指认违反规则的新任务。
- **C\***：一种类C语法的小命令式语言，用于可控的形式语义研究与基准构建。
- **S语义（Small-step operational semantics）**：细粒度操作语义，每步对应一条原子推理规则，显式归约操作数。
- **K框架（K-framework）**：基于重写的语义框架，以较大块变换程序配置并利用内置重写机制处理中间表达式。
- **KeywordSwap**：语义偏移类型，交换熟悉运算符的语义以测试模型覆盖预训练关联的能力。
- **KeywordObf**：语义偏移类型，用新单token符号替换标准符号以消除预训练词形线索。
- **PLSemanticsBench**：先前评估LLM编程语义理解的工作，提供PredState/PredRule/PredTrace任务与C*有效程序集。
- **False success / Wrong rule / Malformed output**：本文错误分类；假成功指将无效程序判为有效，错规则指判错违反条款，畸形输出指无法解析的回答。

## 可复现要素
- 数据集：PrEx数据集公开，含Human-Written、LLM-Translated、Fuzzer-Generated拆分与2946个程序（491有效+2455无效），见https://github.com/EngineeringSoftware/prex。
- 代码：论文提供数据集与评测入口；具体推理脚本与超参以仓库为准，论文未单独列出完整超参表。
- 权重/模型：使用开源Qwen、DeepSeek、Ministral系列模型（3B/7B/14B/32B等），包含非推理与CoT变体。
- 关键设置：{S,K}×{Standard,KeywordSwap,KeywordObf}×{non-CoT,CoT}配置，推理模型仅用默认CoT，每配置运行三次取均值。
