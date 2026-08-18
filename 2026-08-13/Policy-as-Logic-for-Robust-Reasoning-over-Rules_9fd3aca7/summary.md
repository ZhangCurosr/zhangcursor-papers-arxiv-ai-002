---
title: "Policy-as-Logic-for-Robust-Reasoning-over-Rules"
source: https://arxiv.org/pdf/2608.11905v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 02:35:27"
---

# 论文速读：Policy-as-Logic-for-Robust-Reasoning-over-Rules

## 一句话总结
论文提出 Policy-as-Logic (PaL) 框架，将策略/规则决策严格拆分为“LLM 事实抽取”与“ASP 符号求解”两个独立阶段。在客观数值型规则场景（航空、税务、NBA）中，该方法在准确率与输入扰动鲁棒性上显著超越 Policy-as-prompt 与 Policy-as-code 基线，同时将推理 token 消耗降低约一个数量级。

## 研究问题与动机
- **核心问题**：在涉及书面政策（税务、定价、内容审核等）的自动化决策中，端到端 LLM 生成方法对输入细微变化（语气、语序、表述风格）极度敏感，导致决策结果不稳定，且缺乏可解释性与审计能力。
- **现有方法不足**：
  1. **Policy-as-prompt** 将完整政策文本作为上下文注入 LLM，token 消耗巨大，且 LLM 在处理离散实体与复杂条件分支时逻辑推理能力不可靠。
  2. **Policy-as-code** 依赖 LLM 生成可执行代码，业务规则越复杂代码生成错误率越高，调试与修复成本难以接受。
  3. 现有神经符号工作多聚焦抽象逻辑推理任务，缺乏对真实世界“客观规则 vs 主观信念”混合政策场景的系统性鲁棒性评测。

## 核心贡献（创新点）
1. **提出 PaL 解耦架构**：通过分离自然语言理解与符号推理，让 LLM 专注事实抽取，让 ASP 求解器负责确定性规则演绎，从架构层面切断生成随机性对最终决策的影响。
2. **系统化扰动鲁棒性评测协议**：引入六类语言重述扰动（冗长、改写、干扰、误导上下文、欢快/沮丧情感）并辅以 llm-as-a-judge 语义一致性校验，首次在多难度层级下量化验证神经符号策略推理的稳定性。
3. **划定客观/主观政策的适用边界**：实证表明 PaL 在基于知识/数值的客观规则中优势显著，而在依赖概念判断的主观规则（HR 审核）中收益有限，为方法选型提供明确的数据依据。
4. **实现 ~10x Token 效率提升**：推理时仅传入策略模式（schema）而非完整政策文档，在保持高精度的同时大幅降低 API 调用成本与延迟。

## 方法详解
方法采用五阶段流水线，核心思想是**非单调逻辑 + 提取/求解严格分离**：
- **语义解析 (Semantic Parsing)**：离线使用 LLM（实验中使用 Claude Opus 4.7）将自然语言政策文档 $P$ 翻译为 Answer Set Program (ASP)。程序需覆盖所有规则、例外与默认情况，同时生成供抽取阶段使用的 JSON schema 与领域映射表。
- **事实抽取 (Extraction)**：在线推理时，LLM 仅根据 schema 从用户查询 $x$ 中提取结构化事实（JSON 列表），不接触完整政策文本，避免上下文长度限制与策略混淆。
- **基址化 (Grounding)**：利用解析阶段生成的映射表，将 JSON 事实转换为命题逻辑原子，处理类型映射（如布尔→命题原子）、自定义列表展开及数值缩放（如 cents→dollars），生成不含变量的命题形式 ASP 程序。
- **求解 (Solver)**：调用 Clingo v5.8.0 计算基址化程序的稳定模型（Answer Set）。针对多解歧义场景（如行李费用最小化），在 ASP 中嵌入 weak constraints / optimization directives 以唯一化决策。
- **解释 (Interpretation)**：将求解输出的决策原子按领域规则映射回原始数值/类别输出。
- **计算特性**：基址化搜索空间通常与程序谓词数量同阶；答案集规模极小（实验中 $|M| \leq 3$），因此除 LLM 调用外不引入额外延迟。唯一的不确定性来源被限制在事实抽取阶段。

## 实验与结果
- **数据集**：RuleArena（Airline, Tax, NBA）与 PolyGuard（HR）；样本量分别为 n=300, 300, 216, 300。
- **评估基线**：Policy-as-prompt（0-shot/1-shot）结合 GPT-OSS 120B、Qwen-2.5 72B、Llama-3.3 70B、Granite-4.1 8B；Policy-as-code 采用 DeonticBench 文献报告值。
- **主要结果**：
  - **Airline**：PaL 准确率 0.94–1.00，鲁棒性 0.93–0.98；最优基线（GPT-OSS 0-shot）仅 0.38 Acc / 0.27 Rob。即使仅用 8B 小模型 Granite，PaL 也达 0.61 Acc / 0.62 Rob，相对基线提升超 60 倍。
  - **Tax**：所有基线准确率均 <0.10 且鲁棒性跌至 0.00；PaL 稳定维持在 0.31 Acc / 0.2
