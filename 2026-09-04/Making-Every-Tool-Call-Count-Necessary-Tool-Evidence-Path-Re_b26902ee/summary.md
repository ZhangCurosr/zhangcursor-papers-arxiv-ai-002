---
title: "Making-Every-Tool-Call-Count-Necessary-Tool-Evidence-Path-Re"
source: https://arxiv.org/pdf/2609.03493v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 15:32:03"
field: "多模态智能体工具调用与强化学习"
keywords: ["Agentic VLM", "Tool Use", "Reinforcement Learning", "Process Reward", "Vision-Language Model", "Multi-step Reasoning", "OOD Generalization"]
innovations: ["提出NTEP双阶段过程奖励机制，显式监督预调用意图对齐与后调用信息提取", "引入非重复目标正则化器压制饱和式工具调用，实现准确率与调用效率双赢", "构建教师蒸馏+失败补全的NTEP自动化构造流水线，实现工具无关的证据路径监督"]
benchmarks: ["MMSearch", "HR-MMSearch", "InfoSeek", "MAT-Search", "V* Bench", "HR-Bench 4K", "HR-Bench 8K"]
---

# 论文速读：Making-Every-Tool-Call-Count-Necessary-Tool-Evidence-Path-Re

## 一句话总结
本文提出 NTEP（Necessary Tool-Evidence Path）注释方案与 NTEP-R 奖励机制，通过对智能体视觉语言模型（VLM）的**预调用意图对齐**与**后调用信息提取**进行细粒度过程监督，并引入重复目标正则化惩罚冗余调用，在统一三工具框架下显著提升了搜索导向准确率与工具使用效率。

## 研究问题与动机
1. **现有监督信号过于粗粒度**：当前 agentic VLM 训练主要依赖最终答案正确性或单纯的工具调用次数作为奖励，无法准确反映每次工具调用的实际必要性与有效性。
2. **预调用意图偏差（Pre-call misalignment）**：模型常发起偏离目标或冗余的工具调用，未能主动检索解答所必需的关键证据。
3. **后调用信息提取失败（Post-acquisition failure）**：即使工具返回了有价值的上下文，模型也经常在后续推理中未能成功抽取所需证据，导致工具调用白耗。
4. **缺乏证据级信用分配机制**：终端答案监督无法区分“有效证据获取路径”与“碰巧答对的路径”，需建立细粒度的过程级监督以保证每次调用都严格推进推理进程。

## 核心贡献（创新点）
1. **提出 NTEP 注释方案**：将推理轨迹显式分解为有序的“答案关键步骤”，每步独立定义证据目标、预期工具与必须抽取的信息；与现有仅关注任务完成的方法本质不同，本文首次将每个工具调用的**证据角色**显式建模并监督。
2. **设计 NTEP-R 双阶段奖励机制**：通过冻结的语义 Judge 分别评估预调用意图对齐（Align）与后调用信息获取（Acquire），并将两者与终端答案奖励结合；区别于仅依赖结果反馈的 ToolRL/RLTR 等，本文实现可归因到单次调用过程的价值分配。
3. **引入非重复目标正则化器（Non-repeated-goal regularizer）**：对已满足 NTEP 目标的重复调用施加显式惩罚；相比单纯增加调用数量或放宽约束的基线，该设计直接压制饱和式调用，实现准确率与调用效率的双赢。
4. **构建双分支 NTEP 构造流水线**：利用强教师模型从目标策略的 warm-up  rollout 中蒸馏有效路径（成功样本剪枝冗余、失败样本补全缺失步骤），在保持工具无关性的同时显著提升训练数据覆盖率与路径质量。

## 方法详解
1. **NTEP 形式化**：每个必要步骤建模为工具中介的证据转移 $g_j \xrightarrow{t_j} e_j$，其中 $g_j$ 为调用前证据目标，$t_j$ 为期望工具，$e_j$ 为从返回观测中必须提取的信息。样本路径表示为 $\mathcal{P}(x) = ((g_j, t_j, e_j))_{j=1}^N$，不规定具体查询字符串或绑定框，保证工具无关性。
2. **路径构建（Teacher Distillation）**：教师模型（Gemini-3.1-Pro）处理目标策略 warm-up rollout：成功轨迹剔除冗余/非关键交互保留证据支撑步；失败轨迹识别缺失证据转移并补写，统一输出标准化 $(g_j, t_j, e_j)$ 三元组。冻结后作为 GRPO 阶段的参考路径。
3. **双阶段语义评判**：对每条 rollout 中的第 $i$ 次调用，计算二元指标：
   - $\alpha_{i,j} = \text{Align}(r_i, c_i; g_j, t_j)$：预调用状态与目标对齐程度。
   - $\beta_{i,j} = \text{Acquire}(r_i^+; e_j)$：调用后推理是否显式捕获必要信息（$\beta \le \alpha$）。
4. **过程奖励聚合**：统计四类指标 $H_{goal}$（命中目标数）、$H_{info}$（命中信息数）、$M_{goal}$（无目标偏移调用）、$D_{dup}$（重复对齐冗余调用），组合为：
   $$R_{NTEP}(\tau,x) = \frac{1}{N}\left((1-\lambda_g)H_{info} + \lambda_g H_{goal} - \lambda_g M_{goal} - \lambda_d D_{dup}\right)$$
5. **GRPO 集成**：总奖励 $R_g = R_{ans} + R_{fmt} + \lambda R_{NTEP}$，其中 $R_{ans}$ 为答案正确性奖励，$R_{fmt}$ 为 XML 协议格式奖励。采用 token 级组相对优势 $\widehat{A}_g^{\text{NTEP}} = (R_g - \bar{R}_x)/(s_x + \epsilon)$ 更新策略，无需额外 critic。推理阶段仅输入图像、问题与工具接口，NTEP 与 Judge 完全脱离。

## 实验与结果
- **数据集**：MMSearch、HR-MMSearch、InfoSeek、MAT-Search（搜索导向）；V* Bench、HR-Bench 4K/8K（细粒度视觉），共 7 个基准。
- **基线**：端到端 VLM（Qwen3-VL-8B/30B、GPT-5、Grok-4、Gemini-2.5-Pro、Claude-Sonnet-4.6）；RL agent pipelines（Vision-R1-7B、MM-DeepResearch-8B、DeepEyes-7B/V2、Thyme-RL、Chain-of-Focus、V-Thinker、PixelReasoner、Visual-ARFT-Search、WebWatcher、SenseNova-MARS-8B）。
- **最强结果**：NTEP-8B 在统一三工具（image_zoom_in_tool、image_search_tool、text_search_tool）协议下取得 **70.34%** 综合平均准确率，超越最强同 harness RL 基线 SenseNova-MARS-8B **2.03 个百分点**。搜索导向均值从 58.12% 提升至 60.55%，视觉均值从 81.90% 提升至 83.40%。
- **工具使用效率**：搜索基准平均调用数从 2.54 降至 **1.89**，视觉基准从 1.86 降至 **1.08**；必要且有效调用率提升至 **78.8%**（Case-5 从 39.0% 升至 **58.7%**）。
- **消融结论**：仅保留预调用目标对齐会导致搜索均值崩塌至 34.58%；完整 NTEP-R 将平均调用压至 1.55（较 goal-only 的 5.36 下降约 71%，较 Answer Reward Only 的 3.11 下降约 50%）。
- **OOD 跨工具迁移**：仅在搜索任务上训练的模型，在评估时开放未见过的新工具（zoom），视觉均值从 75.01% 跃升至 **82.32%**，平均调用从 1.679 降至 **1.294**，证明证据导向策略具备工具无关的可迁移性。
- **Judge 可靠性**：人工盲审 100 例，准确率 97.0%，Cohen’s κ = 0.94，验证自动化过程奖励的测量可信度。

## 相关工作脉络
1. **Agentic VLM Tool Use**：MMSearch-R1、SenseNova-MARS、MM-DeepResearch、DeepEyes、Vision-R1 等展示了 VLM 掌握多轮工具交互的能力，但大多仅监督任务最终完成或工具调用行为本身，未显式刻画每次调用的证据必要性。
2. **RL for Agent-Tool Interaction**：ToolRL、RLTR、Atom-Searcher、TA-MDP 等引入过程/成本/原子奖励，但奖励设计仍停留在轨迹级或动作级，缺乏将“预调用意图-后调用提取”解耦的细粒度证据路径监督。
3. **文本领域搜索 RL**：Search-R1、R1-Searcher、StepSearch、WebThinker 等在纯文本场景验证了强化学习驱动的动态知识获取，本文将其范式扩展至多模态并引入视觉操作（裁剪/缩放）与跨模态证据匹配。
4. **过程奖励与信用分配**：现有方法多依赖 outcome-only 或浅层 process reward，本文通过 $H_{goal}/H_{info}/M_{goal}/D_{dup}$ 四元组实现调用级的可归因信用分配，填补了证据级监督的空白。
5. **OOD 工具泛化**：传统 RL agent 易过拟合训练期工具集，本文证明将奖励锚定于抽象证据状态而非具体工具 ID，可使策略自然泛化至未见接口。

## 局限性与未来方向
1. **路径质量受限于教师模型**：NTEP 由更强模型蒸馏生成，失败样本的补全分支本质上更难构造，残留噪声需依赖终端答案奖励对冲。
2. **语义 Judge 带来推理/训练开销**：每步预调用与后调用判定需调用冻结大模型，虽通过并发控制与长度截断缓解，但仍高于纯 outcome-only RL。
3. **工具集与语言覆盖有限**：当前仅验证三工具（裁剪、图像检索、文本检索）与英语检索场景，更丰富的工具家族与多语言语料是自然扩展方向。
4. **重复目标正则化的权衡**：当前 $\lambda_d$ 偏向效率最优前沿，对追求最大化重试容错的应用场景需按需求调整惩罚权重。
5. **工具选择监督缺失**：残差失败分析显示 “wrong tool selection” 仍是最大误差来源，未来需结合工具选型细粒度监督进一步压缩无效调用。

## 研究启发与可借鉴点
1. **双阶段过程奖励设计**：将工具调用拆分为“调用前意图对齐”与“调用后信息提取”独立评判，可作为类似 multi-step reasoning with tool execution 任务的通用奖励范式。
2. **失败轨迹补全（Completion Branch）构造策略**：利用教师模型识别缺失证据转移并反向生成 NTEP，显著扩充困难样本覆盖，可直接迁移至其他需要构造过程监督数据的 RL 训练 pipeline。
3. **证据状态去耦工具 ID**：将信用分配锚定在抽象证据目标 $(g_j, e_j)$ 而非具体工具名，是实现零样本/少样本新工具迁移的有效机制，适用于开放域 agent 系统。
4. **XML 协议显式暴露状态转换**：通过 `<thinking>`、`<tool_call>`、`<tool_response>`、`<answer>` 结构化协议配合格式奖励 $R_{fmt}$，可稳定抓取出预/后调用状态，便于后续接入自动化评测或人类审计。
5. **Group-Relative Token-level 优化适配**：在 GRPO 框架中保留 clip 策略但将 advantage 计算下沉至 token 级并注入过程奖励，兼顾多轮交互序列的探索空间与训练稳定性。

## 关键术语表
- **NTEP（Necessary Tool-Evidence Path）**：显式定义每个答案关键步骤所需的外部证据目标、预期调用工具及必须从观测中提取的信息序列的注释方案。
- **NTEP-R（NTEP Reward）**：基于 NTEP 的过程级奖励机制，对预调用意图对齐与后调用信息提取进行双重监督，并惩罚重复目标调用。
- **Pre-call Goal Alignment**：评估模型在发起工具调用前的推理状态与当前待完成证据目标的一致性。
- **Post-call Information Acquisition**：评估模型在接收工具返回后，是否在后续推理中显式捕获并利用了必要的证据信息。
- **Non-repeated-goal Regularizer**：对已满足 NTEP 目标的工具调用施加负向惩罚，抑制饱和式/重复检索行为。
- **H_goal / H_info / M_goal / D_dup**：过程奖励的四个统计指标，分别衡量目标命中数、信息提取数、无目标偏移调用数与重复对齐冗余数。
- **Completion Branch**：针对失败 rollout 由教师模型补全缺失证据步骤的路径构造分支，与 Extractive Branch 共同构成训练数据源。
- **Case-5 Rate**：轨迹中所有工具调用均为必要且有效、且最终答案正确的样本占比，用于衡量整体调用效率与正确率的联合表现。

## 可复现要素
- **数据集**：MMSearch、HR-MMSearch、InfoSeek、MAT-Search、V* Bench、HR-Bench 4K、HR-Bench 8K（部分数据集受非商业/研究许可限制，详见附录 E）。
- **代码/权重开源状态**：论文未明确声明开源（官方链接仅指向 arXiv 原文）。
- **关键超参**：基座模型 Qwen3-VL-8B-Instruct；GRPO group size G=8；上下文 32K，最大响应 16K；工具响应截断 8K token；Optimizer AdamW lr=1e-5，weight decay 0.01，gradient clip 1.0；训练 20 epochs；奖励权重配置：Info hit +0.7、Goal hit +0.3、Goal miss -0.3、Duplicate -0.1，process reward scale 0.5；Judge 模型 Qwen3-VL-Plus，temperature=0，8K 输出上限；Rollout sampling temperature=1.0，top-p=1.0，presence penalty=1.5。
