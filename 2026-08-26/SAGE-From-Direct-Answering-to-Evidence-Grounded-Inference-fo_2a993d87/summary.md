---
title: "SAGE-From-Direct-Answering-to-Evidence-Grounded-Inference-fo"
source: https://arxiv.org/pdf/2608.24011v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 11:13:58"
field: "多模态文档理解"
keywords: ["证据驱动推理", "多智能体框架", "中文古籍理解", "视觉语言模型", "主张级验证", "AncientDoc"]
innovations: ["将中文古籍理解重构为证据驱动的有界推理过程，而非单次直接回答", "提出规划-取证-验证三角色智能体与受限共享状态运行时，支持 abstention 与 traceable 诊断", "在 AncientDoc 上以 Qwen3.5-9B 骨干在 8 项指标中 7 项超越更大规模单体 LVLM"]
benchmarks: ["AncientDoc"]
---

# 论文速读：SAGE: From Direct Answering to Evidence-Grounded Inference for Chinese Ancient Document Understanding

## 一句话总结
SAGE 提出一种证据驱动的多智能体框架，将中文古籍理解从 LVM 单次直接回答范式重构为"规划→工具辅助取证→主张级验证→有界重规划"的推理过程，在 AncientDoc 基准上以 Qwen3.5-9B 为骨干获得七个指标的 SOTA，显著超越参数量大数倍的单体大模型。

## 研究问题与动机
- **直接回答范式的固有缺陷**：当前 LVLM 通常将阅读、取证、推理、验证压缩为一次前向传播，输出高度自信但证据支撑薄弱的答案，且出错后难以诊断。
- **中文古籍理解的高复杂度**：竖排版式、繁体/异体字、缺标点、密集注释与隐性历史文化语义，要求模型同时完成视觉阅读、古汉语解读与知识 grounding，远超现代文档理解。
- **人机实践差距**：史学家面对复杂问题时系统性地阅读、查考、核实并在证据不足时保持克制（abstain），现有 LVLM 不具备此类能力。
- **单纯 scaling 不足以解决问题**：实验显示更小的模型配合结构化证据 grounding 可击败远大的单体 LVLM，说明推理过程设计比参数规模更重要。

## 核心贡献（创新点）
1. **范式重构**：将中文古籍理解重新定义为"证据驱动的推理"而非直接生成，输出除最终答案外还暴露证据、验证报告、abstention 决策与执行轨迹。与既有工作仅优化单步生成的本质差异在于把可解释性与可靠性嵌入推理本身。
2. **SAGE 多智能体框架**：提出规划、执行、验证三角色智能体 + 受限共享状态运行时，支持任务感知规划、工具中介取证、主张级验证与有界重规划。与单纯 RAG 或 chain-of-thought 的区别在于引入预算约束、运行时规则控制与 abstention 机制。
3. **系统的实证验证**：在 AncientDoc 基准上以三种骨干模型全面评估，SAGE+Qwen3.5-9B 在 8 项指标中 7 项最优，消融与 trace 分析证实规划、工具执行与验证各自贡献。与现有古文理解工作的差异在于聚焦推理过程的可诊断性而非最终答案 accuracy  alone。

## 方法详解
- **共享状态运行时**（Core Runtime）：维护统一状态 $s_k = (y, P_k, X_k, E_k, \hat{A}_k, R_k, b_k, \tau_k)$，其中 $y$ 为任务类型预测，$P_k$ 为当前计划，$X_k$ 为已识别/归一化页文本，$E_k$ 为累积证据，$\hat{A}_k$ 为候选答案，$R_k$ 为验证报告，$b_k$ 为剩余预算，$\tau_k$ 为执行轨迹。运行时不固定动作序列，而是按规则在 STOP / REPLAN / ABSTAIN 间决策：证据充分则 STOP；证据缺失且有预算则 REPLAN；证据不足且预算耗尽则 ABSTAIN。
- **学术规划智能体**（Scholarly Planning Agent）：输入问题 $Q$ 与初始预算 $b_0$，输出计划 $P_0 = (y, \mathbf{a}_0, \rho_0, b_0)$，其中 $y$ 预测任务类型（OCR / Translation / Reasoning QA / Knowledge QA / Linguistic QA），$\mathbf{a}_0$ 为有序动作序列，$\rho_0$ 指定证据需求与验证标准。重规划时基于当前状态和剩余预算更新计划。
- **工具中介执行智能体**（Evidence-Grounded Execution Agent）：在受限动作空间内执行 read / normalize / extract / retrieve / synthesize。read 调用 Qwen-VL-OCR 识别竖排古籍；normalize 用 OpenCC 做繁简转换与异体归一；extract 为轻量规则式术语抽取；retrieve 搜索冻结的本地知识证据池（最多 8 条 snippet）。状态更新为 $(X_k, E_k) = \text{Exec}(I, Q, P_k, s_k)$，候选答案 $\hat{A}_k = \text{Solve}(Q, X_k, E_k, P_k)$ 作为待验证假设。
- **学者验证智能体**（Scholar Verifier Agent）：将候选答案拆解为原子主张 $C_k = \{c_1, \dots, c_n\}$，逐条验证 $v_i = \text{Verify}(c_i, Q, X_k, E_k) \in \{\text{SUP}, \text{CON}, \text{INS}, \text{N/A}\}$，聚合生成证据报告 $R_k = \text{Aggregate}(C_k, V_k, X_k, E_k)$，包含支持率、矛盾/不足主张、置信度与 abstention 建议。Knowledge QA 要求更严格的 claim-evidence grounding，Translation / Reasoning / Linguistic QA 侧重页内一致性。
- **预算约束**：默认每例最大调用次数——OCR 3 次、翻译 5 次、推理 QA 5 次、知识 QA 7 次、语言学 QA 6 次；检索每轮最多返回 8 条 snippet。所有工具调用、证据、验证结果与决策均记录于 $\tau$ 支持事后诊断。

## 实验与结果
- **数据集**：AncientDoc（Yu et al., 2025），含 OCR / Translation / Reasoning QA / Knowledge QA / Linguistic QA 五类任务。主评四类理解任务（OCR 因侧重工具质量被排除），指标为 CHRF++ 与 BERTScore F1（BS-F1）。
- **骨干模型**：InternVL3-8B、Qwen2.5-VL-7B、Qwen3.5-9B，均与 matched direct-baseline 对比。
- **最强结果**：SAGE + Qwen3.5-9B 在 8 项指标中 7 项达 SOTA：
  - Translation：CHRF++ 16.01（vs. Direct 7.71，**+8.30**）、BS-F1 75.29（vs. 68.03，+7.26）
  - Knowledge QA：CHRF++ 11.81（vs. 6.80，**+5.01**）、BS-F1 71.12（vs. 66.83，+4.29）
  - Linguistic QA：CHRF++ 7.72（vs. 1.83，**+5.89**）、BS-F1 65.35（vs. 55.63，+9.72）
  - Reasoning QA：CHRF++ 9.35（vs. 6.88，+2.47）、BS-F1 69.10（vs. 68.56，+0.54）
  - 超越 GPT-4o、Gemini2.5-Pro、Qwen2.5-VL-72B 等远大规模单体模型的多数指标。
- **消融（Qwen3.5-9B，Knowledge QA）**：w/o Planner CHRF++ 9.34 / BS-F1 67.80；w/o Page Reader 11.58 / 70.79；w/o Retrieval 11.78 / 70.89；w/o Verifier 11.23 / 70.08；Full SAGE 11.81 / 71.12。Page Reader 移除后小幅下降，说明页级证据是主要贡献；Retrieval 贡献较温和；Planner 与 Verifier 对 BS-F1 尤为关键。
- **Trace 分析**：知识 QA 中 90.39% 触发 extract/retrieve，推理 QA 仅 9.42%，体现任务自适应；Replan 后支持率 69.23%→74.45%，置信度 67.99%→71.59%。

## 相关工作脉络
- **AncientDoc（Yu et al., 2025）**：最直接的对标基准，覆盖 OCR→翻译→推理/知识/语言学 QA 多任务。本文与它的差异：AncientDoc 评估直接回答的最终输出，本文进一步评估推理过程的可 grounding 性与可诊断性。
- **AC-EVAL / C³Bench / 伏羲 / WenyanBENCH**：纯文本古文理解与翻译基准。本文的定位差异在于从文本扩展到**页面图像**的多模态场景，并引入证据 grounding 与验证机制。
- **HisDoc1B / BABM-LLM**：历史文档/古籍资源与数据集。本文与之互补：前者侧重语料构建，本文侧重**推理框架与系统方法**。
- **AnchiBERT / WenyanGPT / TongGu**：面向古汉语的领域预训练/指令微调模型。本文与之正交：SAGE 是**模型无关的推理时框架**，不依赖新训练模型。
- **DocVQA / TextVQA / OCRBench / MMLongBench-Doc**：现代文档理解基准。本文强调古风文档在版式、异体字、隐性语义上的特殊挑战，指出通用文档 benchmark 不足。
- **RAG / Agent 工作**：已有检索增强与多智能体系统。SAGE 的差异化在于为古文场景定制了**有界共享状态运行时、主张级验证、abstention 机制**，并显式暴露执行轨迹。

## 局限性与未来方向
- **评估范围有限**：仅在 AncientDoc 的任务与数据分布上验证，未扩展到更广泛的历史收藏、不同版式或**其他古代语言**。
- **验证信号为系统自生成**：Verifier 与 trace 统计来自系统内部报告，**非人工事实标注**，应解释为运行时行为指标而非答案正确性的确证。
- **推理成本增加**：规划、工具调用、验证与重规划引入额外开销；未来需探索更高效的执行策略。
- **未来方向**：更强的证据源（如开放检索）、人机协同验证、更低成本的证据采集与验证流程。

## 研究启发与可借鉴点
1. **范式迁移**："直接回答 → 证据驱动推理"的范式可推广至其他需要高可信度的垂直领域（法律、医学、学术文献），核心是可暴露证据链与验证报告。
2. **受限共享状态运行时设计**：预算控制 + 规则决策（STOP/REPLAN/ABSTAIN） + 轨迹记录的组合，为构建**可诊断、可干预**的智能体系统提供了可直接复用的架构模板。
3. **主张级验证 + 有界重规划**：将答案拆解为原子主张逐条验证，再以验证反馈驱动重规划，这一机制可复用于任何需要 claim-level grounding 的多模态 QA 任务。
4. **冻结本地证据池**：评估时锁定证据源以确保可复现性与公平对比，这一实验设计值得在基准评测中推广。
5. **trace-based 行为分析**：除最终指标外，通过行动模式与验证置信度变化诊断系统行为，可作为方法论文的**标配补充分析**。

## 关键术语表
**SAGE**：Evidence-Grounded multi-agent framework，面向中文古籍理解的规划-取证-验证一体化推理框架。
**AncientDoc**：覆盖 OCR / 翻译 / 推理 QA / 知识 QA / 语言学 QA 的五任务中文古籍多模态基准。
**Shared-State Runtime**：协调规划、执行、验证三模块的受限运行时，维护统一状态并施加预算与规则控制。
**Claim-Level Verification**：将候选答案拆解为原子主张，逐条判定 SUP / CON / INS / N/A 的验证粒度。
**ABSTAIN**：证据不足时系统主动放弃生成答案的决策，避免幻觉输出。
**Local Evidence Pool**：预构建的离线知识片段库，推理时冻结不可更新，保障可复现性。
**CHRF++ / BS-F1**：本文采用的生成质量评估指标，分别为字符级 n-gram F 分与 BERTScore F1。
**Task-Aware Planning**：根据预测任务类型动态选择证据路径与工具序列的规划策略。

## 可复现要素
- **数据集**：AncientDoc（公开），评估遵循官方协议；冻结本地证据池（构建方法见附录 C.5，来源为 web search 收集后过滤清洗的古文 QA 相关知识片段）。
- **代码/权重**：论文未明确声明代码开源；骨干模型为 InternVL3-8B、Qwen2.5-VL-7B、Qwen3.5-9B（均开源权重）。
- **关键超参**：默认预算上限——OCR 3 次 / 翻译 5 次 / 推理 QA 5 次 / 知识 QA 7 次 / 语言学 QA 6 次；检索上限每轮最多 8 条 snippet；使用确定性/低温度解码。
- **工具实现**：Page Reader 基于 Qwen-VL-OCR；Char Normalizer 基于 OpenCC；Term Extractor 为轻量规则组件；Local Evidence Retriever 使用精确匹配、子串匹配、年份等价匹配、中文重叠等透明匹配规则。
- **提示词**：附录 B 提供 Task Classification / Planning / Page Reading / Task-Specific Answer / Verifier 全套 prompt 模板。
