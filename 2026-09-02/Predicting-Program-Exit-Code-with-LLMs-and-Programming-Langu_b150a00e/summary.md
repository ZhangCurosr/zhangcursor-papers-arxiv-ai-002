---
title: "Predicting-Program-Exit-Code-with-LLMs-and-Programming-Langu"
source: https://arxiv.org/pdf/2609.00579v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 19:27:38"
---

# 论文速读：Predicting-Program-Exit-Code-with-LLMs-and-Programming-Langu

## 一句话总结
提出程序可执行性预测（PrEx）任务，检验LLM在给定形式语义规则时是系统性遵循提示还是依赖预训练先验；实验表明，即使任务简化为二元判定，当前主流编码LLM在语义偏移与高复杂度程序上仍严重依赖表层模式记忆，显式语义推理能力薄弱。

## 研究问题与动机
- **核心问题**：LLM在代码生成与翻译中表现优异，但其是否真正理解编程语言底层语义、能否像解释器一样系统性追踪执行路径，仍缺乏受控实证。
- **现有基准局限**：PLSemanticsBench等任务要求生成多步执行轨迹或中间状态，模型全面低分，无法区分失败源于“多步推理复杂”还是“根本缺乏语义判定能力”。
- **研究动机**：需要更基础的任务基线，剥离轨迹生成负担，仅要求模型判断程序是否可执行并在出错时指出违反的规则，从而精确定位语义推理的能力阈值。
- **机制诊断需求**：传统评测使用标准语法，模型可能靠记忆而非规则推导作答；引入语义偏移（KeywordSwap/Obf）可强制模型覆盖预训练符号关联，检验其是否真正“阅读并应用”了提供的形式规则。

## 核心贡献（创新点）
1. **提出PrEx任务**：要求模型基于给定的C*文法、形式语义规则与程序，输出`##success##`或`##error## + 违规规则编号`。与PLSemanticsBench等多步推理基准相比，该任务将评估目标聚焦于单一语义判定决策，剥离了长轨迹生成的混淆因素。
2. **构建匹配的有效/无效程序数据集**：基于491个有效程序，通过语义感知AST转换生成2455个覆盖5类错误的无效变体。与仅含正确程序的代码基准相比，本研究显式构造了结构对齐的负样本，使评测能够同时检验真阳性判定与假阳性幻觉。
3. **设计KeywordSwap与KeywordObf语义偏移协议**：前者交换标准运算符语义，后者用新颖单token符号替换操作符。该设计直接针对模型“记忆优先”倾向，与依赖标准符号分布的传统benchmark相比，能更干净地分离预训练先验与提示内规则遵循的贡献。
4. **系统化评测形式语义形式与程序复杂度的交互影响**：在S（细粒度小步语义）与K（粗粒度重写框架）两种形式化体系下，结合Human-Written/LLM-Translated/Fuzzer-Generated三划分的复杂度梯度，首次量化了语义推理能力随规则粒度和程序结构复杂度变化的衰减曲线。

## 方法详解
- **任务定义与提示结构**：模型输入包含三部分：C*语言的EBNF文法、指定的形式语义规则集（S或K）、待评估程序。输出强制限定为`##success##`或`##error## Rule N`。CoT变体仅在提示末尾追加简短推理引导，不改变核心评测逻辑。
- **形式语义体系**：
  - **Small-step operational semantics (S)**：采用Gentzen风格推理规则，每条规则仅描述一个原子计算步骤（如`v1 + v2 → v3`），中间求值需显式推导，适合检验细粒度规则应用。
  - **K-framework (K)**：基于Rewrite-based重写语义，将多个原子步骤合并为粗粒度配置转换规则，依赖框架内置重写引擎隐式求值，更接近工业级形式化语义描述。
- **语义偏移构造**：
  - **KeywordSwap**：交换运算符语义（如`+`映射为减法，`-`映射为加法），保留符号可见性但篡改其语义绑定。
  - **KeywordObf**：将标准操作符替换为全新单token符号（如`★`、`▲`、`□`），完全切断预训练词汇表关联；所有新符号均保证为单token，避免提示长度膨胀与分词歧义。
- **数据集生成流水线**：使用ANTLR-based parser-visitor遍历合法程序AST，随机施加5种语义错误转换：`break-outside-loop`、`continue-outside-loop`、`divide-by-zero`、`modulo-by-zero`、`variable-use-before-declare`。每个有效程序生成5个一一对应的无效变体，确保错误类型单一且可追溯。
- **评测配置**：正交组合`{S, K} × {Standard, KeywordSwap, KeywordObf} × {non-CoT, CoT}`，每个配置运行3次取均值，以规则级精确匹配（可执行性+规则编号均正确）作为判定标准。

## 实验与结果
- **数据集与基线**：共2946个C*程序（491有效 + 2455无效），划分为Human-Written（中位19行）、LLM-Translated（中位106行）、Fuzzer-Generated（中位786行）。基线包含随机猜测（约16-18%）及DeepSeek-Qwen/Ministral/Qwen2.5-Coder系列（3B-32B，含CoT）。
- **最强结果**：DeepSeek-Qwen 32B在K语义/Standard条件下达到**99%**准确率，较随机基线提升约81pp，展现了在短小、标准语法程序上的上限。
- **复杂度衰减**：Human-Written → LLM-Translated → Fuzzer-Generated，最强模型平均准确率分别下降**2.9-8.7pp**与**19-33.3pp**；最大单配置跌幅达**55pp**（Qwen2.5-Coder 14B-CoT在K/KeywordObf下从78%跌至23%）。
- **语义偏移效应**：Human-Written上，KeywordObf的中位数准确率降幅（32pp）显著高于KeywordSwap（19pp），表明当表层符号线索被彻底消除时，模型对提示规则的应用能力急剧恶化。
- **失败模式分布**：Human-Written上错误多为同族规则混淆（如continue/break越界混淆）；LLM-Translated与Fuzzer-Generated上**False success**成为主导，模型频繁将违反规则的程序误判为合法； malformed输出在复杂划分中比例上升。
- **核心结论**：即使输出空间被压缩至离散二元判定，LLM仍高度依赖预训练分布而非系统性应用给定形式语义；程序结构越复杂、符号先验越弱，规则遵循能力退化越显著。

## 相关工作脉络
- **PLSemanticsBench [39]**：本文的直接前身，提出PredState/PredRule/PredTrace三任务；定位差异在于PLSemanticsBench侧重多步轨迹生成，本文PrEx将其简化为单步可执行性判定，以分离“推理深度”与“语义遵循”的贡献。
- **CRUXEval [10] / REval [5]**：面向真实Python代码的输出预测与运行时行为一致性评测；本文与之不同，不提供真实运行环境，而是注入显式形式语义规则，专注检验模型对“规则驱动”而非“
