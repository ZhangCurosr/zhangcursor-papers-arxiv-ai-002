---
title: "SAGE-From-Direct-Answering-to-Evidence-Grounded-Inference-fo"
source: https://arxiv.org/pdf/2608.24011v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 11:13:53"
---

# 论文速读：SAGE: From Direct Answering to Evidence-Grounded Inference for Chinese Ancient Document Understanding

## 一句话总结
SAGE 提出了一种面向中文古籍理解的多智能体证据 grounding 推理框架，将传统 LVLM 的单遍直接生成范式重构为“任务规划—工具中介证据获取—主张级验证—有界重规划”的有状态推理过程，显著提升了 AncientDoc 基准上的准确率、证据支撑度与可诊断性。

## 研究问题与动机
- **古籍理解的复合复杂性**：中文古籍具有竖排布局、异体字/繁简混用、无标点、高密度注疏与隐含历史文化语义等特征，远超现代文档理解的感知与推理难度。
- **直接回答范式的固有缺陷**：现有 LVLM 系统普遍采用单遍 direct-answering，将阅读、证据收集、推理与验证压缩为不透明的一步生成，导致输出过度自信、证据支撑薄弱，且错误难以定位。
- **缺乏学术严谨的推理机制**：人类学者处理古籍时会系统阅读、检索参考、逐条验证主张，并在证据不足时主动标注不确定性或放弃；现有方法缺乏 claim-level 验证与 abstention 能力。
- **需要可追溯的推理过程**：仅优化最终答案分数无法反映模型是否真正“读懂”了页面，需显式暴露证据路径、验证报告与执行轨迹，以支撑诊断与可信部署。

## 核心贡献（创新点）
- **范式重构**：将中文古籍理解从直接答案生成 reformulate 为证据 grounding 推理，系统额外输出支撑证据 $E$、验证报告 $R$、abstention 指示 $z$ 与执行轨迹 $\tau$。
- **多智能体协调架构**：设计 Scholarly Planning Agent、Tool-Augmented Execution Agent 与 Scholar Verifier Agent，在 Constrained Shared-State Runtime 下实现任务感知规划、受限工具调用、主张级验证与有界重规划。
- **性能超越规模依赖**：在 AncientDoc 上，SAGE+Qwen3.5-9B 在 8 项指标中 7 项超越规模大 10 倍以上的单体直接回答 LVLM（如 GPT-4o、Gemini2.5-Pro、Qwen2.5-VL-72B），证明结构化证据 grounding 比单纯 scaling 模型更有效。
- **可诊断的轨迹分析**：通过运行时轨迹揭示任务自适应的工具调度规律与验证器引导重规划的实证增益（支持率与置信度显著提升），为系统级可解释性提供量化依据。

## 方法详解
- **共享状态与输出形式**：输入图像 $I$ 与问题 $Q$，运行第 $k$ 步维护状态 $s_k=(y, P_k, X_k, E_k, \hat{A}_k, R_k, b_k, \tau_k)$，最终输出 $O=(A, E, R, z, \tau)$。
- **Scholarly Planning Agent**：以问题 $Q$ 与初始预算 $b_0$ 为输入，预测任务类型 $y\in\{\text{ocr, translation, reasoning\_qa, knowledge\_qa, linguistic\_qa}\}$，生成结构化计划 $P_0=(y, \mathbf{a}_0, \rho_0, b_0)$，明确各环节的证据需求与动作序列；重规划时基于当前状态与验证反馈动态更新计划。
- **Evidence-Grounded Execution Agent**：在受限动作空间 $\{\text{read, normalize, extract, retrieve, synthesize}\}$ 内执行计划，更新页面文本 $X_k$ 与累积证据 $E_k$，并生成候选答案 $\hat{A}_k$（视为待验证假设而非终态输出）。
- **Scholar Verifier Agent**：将 $\hat{A}_k$ 分解为原子主张 $C_k=\{c_1,\dots,c_n\}$，对每个主张评估支撑关系 $v_i=\text{Verify}(c_i, Q, X_k, E_k)\in\{\text{SUP, CON, INS, N/A}\}$，聚合为证据报告 $R_k$，输出支持率、置信度与 abstention 建议。
- **Constrained Shared-State Runtime**：作为系统级调度器，不自主决策，仅维护状态、校验工具调用合法性、限制预算，并按规则 $d_k=\text{Rule}(s_k)$ 输出 $\{\text{STOP, REPLAN, ABSTAIN}\}$；提供共享记忆、固定工具注册表、轨迹日志与预算熔断机制。

## 实验与结果
- **数据集与指标**：AncientDoc 基准，主实验聚焦 Translation、Reasoning QA、Knowledge QA、Linguistic QA 四类，采用 CHRF++ 与 BERTScore F1（BS-F1）评估，排除 OCR（因其主要依赖外部读取工具质量）。
- **基线设置**：对比 8 款代表性直接回答 LVLM（DeepSeek-VL2、LLaVA-OneVision-72B、InternVL3-78B、Qwen2.5-VL-72B、GPT-4o、Gemini2.5-Pro 等），SAGE 在 InternVL3-8B、Qwen2.5-VL-7B、Qwen3.5-9B 三个 backbone 上进行匹配对比。
- **主要结果**：SAGE 在所有三组 backbone 的八项指标上均超越对应 Direct 基线；知识 QA 与语言 QA 提升最显著（如 InternVL3-8B：知识 QA +6.09/ +6.80，语言 QA +4.09/ +8.55）。
- **最强结果**：Qwen3.5-9B+SAGE 在 8 项指标中 7 项登顶，翻译、知识 QA、语言 QA 全面超越规模大一个数量级的单体直接回答模型，仅 Reasoning QA BS-F1 略低于 Qwen2.5-VL-72B Direct。
- **消融与轨迹**：移除 Planner 或 Verifier 均导致 BS-F1 明显下降；关闭 Local Retrieval 仅使 Knowledge QA 小幅回落（说明核心收益来自页面证据组织与验证而非外部知识）；Verifier 触发重规划的样本中，trace 推导的支持率从 69.23% 升至 74.45%，置信度从 67.99% 升至 71.59%。

## 相关工作脉络
- **Document VQA / OCR Benchmarks（DocVQA、OCRBench、MMLongBench-Doc）**：侧重现代文档与场景文本的读取定位，SAGE 转向古籍特殊版面与隐性语义，强调推理过程的证据显式化而非仅评测终态答案。
- **Ancient Chinese / 历史文献基准（AC-EVAL、C³Bench、WenyanBENCH、HisDoc1B）**：多聚焦纯文本理解、翻译或单一模态，SAGE 处理图文混排古籍页面，将任务拆解为“视觉阅读—文本规整—证据检索—主张验证”的协同流程。
- **领域专用模型（AnchiBERT、WenyanGPT、TongGu）**：依赖持续预训练与指令微调的模型中心化路线，SAGE 是模型无关的推理时框架，不改变 backbone 权重，通过多智能体协调与工具调用实现证据 grounding。
- **通用 Agentic LVLM**：常面临幻觉工具调用、无界推理与不可控状态累积问题，SAGE 通过受限动作空间、固定证据池与预算熔断机制，将 agent 行为严格约束在可验证、可追溯的路径内。

## 局限性与未来方向
- **评估范围受限**：仅在 AncientDoc 的任务分布与数据集合上验证，泛化至其他历史文献库、不同版式或古代语言尚未检验。
- **验证统计非人工标注**：Verifier 生成的支持率与置信度为系统自诊断指标，不能替代人工事实验证，需结合专家标注进一步提升可信度。
- **推理开销增加**：规划、工具调用、验证与可能的重规划带来额外的延迟与计算成本，需在未来工作中探索更高效的状态更新与早退机制。
- **未来方向**：引入更强/更多元的外部证据源、接入人工反馈进行 grounding 校准、优化多智能体通信效率，并拓展至跨语言/跨朝代古籍理解。

## 研究启发与可借鉴点
- **状态机式受限运行时**：用共享状态+固定工具注册表+预算规则替代自由对话式 agent，可有效抑制幻觉调用与无界推理，适用于医疗、法律等高可靠性 agent 系统。
- **Claim-level 验证与 Abstention**：将答案拆解为原子主张逐条判定支撑关系，并允许证据不足时主动放弃，可显著提升系统的输出可靠性与可诊断性。
- **任务自适应的证据路径**：规划器根据任务语义动态选择动作序列（如知识 QA 触发检索，翻译/推理 QA 依赖页面文本），避免固定 pipeline 的僵化，值得迁移至长上下文多模态理解。
- **Trace-driven 自我修正**：利用验证器反馈驱动有界重规划，使系统具备“生成—评估—补证”的闭环能力，为构建可解释、可迭代的推理 agent 提供工程范式。

## 关键术语表
- **Evidence-Grounded Inference**：将答案生成建立在显式收集与验证的结构化证据之上，而非仅依赖模型内部参数的隐式生成。
- **Constrained Shared-State Runtime**：维护规划、执行、验证三模块共享上下文状态、工具访问权限与预算限制，并依据预设规则驱动推理流程控制的核心基础设施。
- **Claim-level Verification**：将候选答案分解为原子主张，逐一判定其与已收集证据的支持、矛盾或不足关系，生成结构化验证报告。
- **Bounded Replanning**：在剩余预算允许范围内，根据验证器反馈动态调整后续动作序列，以实现证据补全或推理路径修正。
- **Abstention**：当证据不足以支撑确定结论且预算耗尽时，系统主动选择放弃输出或返回不确定声明的容错机制。
- **Local Evidence Pool**：预构建并冻结的离线知识库，存储古籍相关短片段证据，供推理时按术语检索调用，防止评测期间外部信息污染。
- **AncientDoc**：专为中文古籍理解设计的多模态基准，涵盖页面 OCR、白话翻译、基于页面的推理 QA、知识 QA 与语言变体 QA。

## 可复现要素
- 数据集：AncientDoc（遵循其官方评估协议；论文未提供独立开源链接，需访问原项目页面获取）。
- 代码/权重：论文未明确声明完整开源，但使用 Qwen-VL
