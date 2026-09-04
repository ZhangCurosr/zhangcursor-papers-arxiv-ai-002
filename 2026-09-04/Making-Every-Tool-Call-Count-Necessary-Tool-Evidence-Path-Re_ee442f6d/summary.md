---
title: "Making-Every-Tool-Call-Count-Necessary-Tool-Evidence-Path-Re"
source: https://arxiv.org/pdf/2609.03493v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 21:24:37"
field: "多模态Agent工具使用强化学习"
keywords: ["Agentic VLM", "Tool-use reinforcement learning", "Process reward", "Vision-language model", "Evidence-path supervision", "GRPO", "Cross-tool OOD transfer"]
innovations: ["提出NTEP-R双阶段过程奖励机制，显式监督预调用目标对齐与后调用信息提取", "引入非重复目标正则化显式惩罚冗余工具调用", "证明证据条件化训练实现跨工具OOD转移（搜索→视觉zoom）"]
benchmarks: ["MMSearch", "HR-MMSearch", "InfoSeek", "MAT-Search", "V* Bench", "HR-Bench 4K", "HR-Bench 8K"]
---

# 论文速读：Making-Every-Tool-Call-Count-Necessary-Tool-Evidence-Path-Re

## 一句话总结
论文提出NTEP-R（Necessary Tool-Evidence Path Reward），一种面向Agent型视觉-语言模型的细粒度过程奖励机制，通过在GRPO训练中显式监督每个工具调用的"预调用目标对齐"与"后调用信息提取"两个阶段，并引入非重复目标正则化，显著提升搜索导向准确率与工具使用效率，同时在七个人像基准上达到最强RL-Agent水平。

## 研究问题与动机
1. **现有训练范式监督粗糙**：现有方法主要基于最终答案正确性或单纯工具调用次数进行评估，未能对"证据获取"和"信息利用"进行充分监督，导致奖励信号与工具调用的实际必要性脱节。
2. **预调用偏移（Pre-call misalignment）**：模型常发出冗余或偏离目标工具调用，这些调用未能真正寻求必要证据。
3. **后获取失败（Post-acquisition failure）**：即使工具返回了有价值上下文，模型也常未能从观测中提取推理所需的必要信息。
4. **关键未解挑战**：如何实现"证据层面信用分配"——即不仅判断工具是否被调用，还要确保Agent的意图是真正必要的且检索信息被准确利用。

## 核心贡献（创新点）
1. **NTEP标注方案**：显式定义每个训练实例中的关键证据里程碑（证据目标、预期工具、需提取信息三元组），解决现有结果驱动方法缺乏细粒度意图监督的问题。
2. **NTEP-R双阶段过程奖励**：区别于仅依赖最终答案的Outcome-only奖励，NTEP-R将监督分解为预调用目标对齐（Align）和后调用信息提取（Acquire）两个独立二元判断，使每个工具调用严格推进推理过程。
3. **非重复目标正则化**：显式惩罚重复针对已满足NTEP目标的工具调用，从机制上压缩冗余操作，而非仅依赖最终答案间接引导。
4. **跨工具OOD转移能力**：证明模型学到的是可迁移的"证据寻求纪律"而非固定工具序列，在仅训练搜索任务的情况下，未见zoom工具可使视觉平均准确率提升7.31个百分点且调用次数下降。

## 方法详解
**NTEP定义**：将每个必要步骤建模为工具介导的证据转换 $g_j \xrightarrow{t_j} e_j$，其中 $g_j$ 为调用前证据目标， $t_j$ 为预期工具， $e_j$ 为需从返回观测中提取的必要信息。整个NTEP为有序三元组序列 $\mathcal{P}(x) = ((g_j, t_j, e_j))_{j=1}^N$。

**两阶段构建**：先用教师模型（Gemini-3.1-Pro）蒸馏目标策略warm-up rollout生成冻结NTEP——对成功rollout修剪冗余交互，对失败rollout补全缺失证据步骤。

**双阶段评估函数**：
- **Align**（预调用）：判断当前推理状态 $r_i$ 和工具调用 $c_i$ 是否与某未满足NTEP步骤的目标 $g_j$ 对齐。
- **Acquire**（后调用）：判断调用后推理状态 $r_i^+$ 是否显式捕获了必要信息 $e_j$。

**奖励公式**：
$$R_{\text{NTEP}}(\tau, x) = \frac{1}{N}\left((1-\lambda_g)H_{\text{info}} + \lambda_g H_{\text{goal}} - \lambda_g M_{\text{goal}} - \lambda_d D_{\text{dup}}\right)$$
其中 $H_{\text{goal}}$ 为目标命中数，$H_{\text{info}}$ 为信息提取命中数，$M_{\text{goal}}$ 为偏离目标调用，$D_{\text{dup}}$ 为重复目标调用。

**组合奖励**：$R_g = R_{\text{ans}} + R_{\text{fmt}} + \lambda R_{\text{NTEP}}$，在GRPO的group-relative advantage计算中代入，实现轨迹级优化。

**XML交互协议**：强制模型在 $\thinking$ 标签中表达意图，在 $\tool\_call$ 中发出JSON工具调用，环境注入 $\tool\_response$，后续 $\thinking$ 捕获提取结果，最终 $\answer$ 终止。

## 实验与结果
**数据集**：七个图像基准——MMSearch、HR-MMSearch、InfoSeek、MAT-Search（搜索导向）；V* Bench、HR-Bench 4K/8K（视觉感知）。

**基线对比**：
- 端到端VLM：Qwen3-VL-8B/30B、GPT-5、Grok-4、Gemini-2.5-Pro、Claude-Sonnet-4.6
- RL-Agent：Vision-R1-7B、MM-DeepResearch-8B、DeepEyes-7B/V2-7B、Thyme-RL、Chain-of-Focus-7B、V-Thinker-7B、PixelReasoner-7B、Visual-ARFT-Search、WebWatcher-7B、SenseNova-MARS-8B

**主要结果（Table 1）**：
- NTEP-8B在统一三工具框架下取得**70.34%**平均准确率，超越最强同 harness RL-Agent（SenseNova-MARS-8B: 68.31%）**+2.03pp**。
- 搜索平均（S-Avg.）：**60.55%** vs 58.12%（+2.43pp）；视觉平均（V-Avg.）：**83.40%** vs 81.90%（+1.50pp）。
- 搜索导向任务上NTEP-8B（60.55%）大幅超越最强端到端基线Gemini-2.5-Pro（45.92%）**+14.63pp**。

**工具使用效率（Figure 3）**：
- 搜索基准：平均工具调用从2.54降至**1.89**；视觉基准：从1.86降至**1.08**。
- Case-5率（所有调用均必要+最终答案正确）：从39.0%提升至**58.7%**。

**跨工具OOD转移（Figure 6）**：
- 仅训练搜索任务（无zoom工具），评测时暴露zoom工具：视觉平均准确率从75.01%提升至**82.32%**，平均调用从1.679降至**1.294**。

**消融（Table 6）**：
- 去掉信息获取+非重复目标正则化后，Search Avg. 从59.44骤降至34.58，总调用从1.55增至5.36。
- 仅答案奖励（Answer Reward Only）虽达69.63%整体平均，但总调用3.11，远低于NTEP-R的1.55。

**理论分析（Appendix A.6）**：
- Proposition 1证明浪费调用个体不可行，真正重试在成功率 $p > \lambda_d/(1-\lambda_g) = 1/7$ 时仍有正期望收益。
- Proposition 2证明最优停止：所有目标满足后继续调用纯为惩罚，调用需求由证据路径设定而非预算。
- Proposition 3证明证据条件化训练使值函数随工具集单调递增，支持跨工具OOD转移。

## 相关工作脉络
1. **Search-R1/R1-Searcher/R1-Searcher++**：文本领域LLM通过RL学习搜索行为的开创性工作，但仅评估搜索动作的outcome，无证据层面细粒度监督。
2. **MMSearch-R1/SenseNova-MARS/MM-DeepResearch/WebWatcher**：将搜索RL扩展至多模态，但主要关注Agent能否完成任务或学习工具行为，未显式指定每次调用的证据角色。
3. **DeepEyes/DeepEyesV2/Vision-R1/Visual-ARFT/Thyme-RL/V-Thinker/Chain-of-Focus/PixelReasoner**：聚焦视觉操作与交互式图像推理，奖励设计多为outcome-based或粗粒度process reward，缺乏"必要证据获取-提取"双阶段监督。
4. **ToolRL/RLTR/Atom-Searcher/TA-MDP**：工具学习奖励分解相关工作，ToolRL侧重工具学习本身，RLTR奖励"好过程无正确结果"，均不如NTEP-R将证据路径作为样本特定的过程监督锚点。
5. **定位差异**：NTEP-R不预设刚性推理轨迹，而是定义证据里程碑（tool-agnostic），允许模型自由 formulate 工具参数和中间步骤，同时通过semantic judge提供可审计的细粒度过程信号。

## 局限性与未来方向
1. **NTEP质量依赖教师模型**：路径由更强模型（Gemini-3.1-Pro）蒸馏，完成分支（reconstruction from failed rollouts）比提取分支更难，残留路径噪声仍由答案奖励锚定。
2. **推理开销增加**：步级语义判断相比outcome-only RL引入额外服务延迟，通过bounded judge concurrency控制但仍需权衡。
3. **工具多样性有限**：实验仅覆盖三种工具（crop/zoom、image search、text search）且以英文检索为主， richer tool families和multilingual corpus为自然扩展方向。
4. **工具选择监督仍有提升空间**：failure mode分析显示"wrong tool selection"仍是最大残留错误（~14.5%），暗示工具选择层面的监督可作为补充方向。
5. **非重复目标正则化参数敏感**：当前 $\lambda_d$ 设为0.1达成效率-精度折中，但对需最大重试的应用场景需调整。

## 研究启发与可借鉴点
1. **双阶段过程监督范式可迁移**：将工具调用拆分为"意图对齐→信息提取"两阶段独立评估，适用于任何需要外部证据的多步推理任务（如代码生成中的API调用、数学证明中的定理引用）。
2. **非重复目标正则化机制**：通过 $D_{\text{dup}}$ 显式惩罚已满足目标的重复调用，是一种简洁有效的效率正则化，可借鉴于Agent轨迹压缩。
3. **样本特定的冻结NTEP设计**：教师蒸馏生成样本专属路径并冻结，既避免overfitting到单一轨迹模板，又保留任务结构性约束，可在其他RLHF/GRPO场景中复用。
4. **跨工具OOD转移的实验设计**：仅训练部分工具集，测试时暴露新工具验证策略可迁移性，是评估Agent泛化能力的有力范式。
5. **理论保证+实证验证的闭环**：Proposition 1-3从理论上刻画奖励函数的最优停止、浪费定价和工具集单调性，并与实验严格对应，为后续工作提供可验证的理论基线。

## 关键术语表
**NTEP (Necessary Tool-Evidence Path)**：必要工具-证据路径，显式定义每个回答关键步骤中的证据目标、预期工具、需提取信息三元组的标注方案。
**NTEP-R**：基于NTEP的过程奖励机制，通过双阶段语义判断和非重复目标正则化评估工具调用有效性。
**Pre-call goal alignment (Align)**：预调用阶段评估，判断Agent的推理意图和工具调用是否针对某个必要证据目标。
**Post-call information acquisition (Acquire)**：后调用阶段评估，判断Agent是否从工具返回观测中提取了必要信息。
**Non-repeated-goal regularizer**：非重复目标正则化项，惩罚重复针对已满足NTEP目标的工具调用。
**GRPO (Group Relative Policy Optimization)**：组相对策略优化，本文用于轨迹级RL优化的基础算法。
**Evidence-seeking discipline**：证据寻求纪律，指模型学到的可迁移的"按需取证-适时停止"策略，而非记忆固定工具序列。
**Case-5 rate**：全部工具调用均为必要且被使用、最终答案正确的轨迹比例。

## 可复现要素
| 要素 | 详情 |
|------|------|
| 基础模型 | Qwen3-VL-8B-Instruct（vision tower frozen） |
| 训练数据 | 历史7,774例池（FVQA/DeepEyes-4K/Visual-Probe/VDR/Search-R1-text）+ 搜索导向4,855例池 |
| 工具集 | image_zoom_in_tool, image_search_tool, text_search_tool（统一三工具接口） |
| 代码开源 | 论文未提及 |
| 权重开源 | 论文未提及（提到checkpoint但无具体链接） |
| 关键超参 | G=8 trajectories/input, batch=128 prompts, lr=1e-5, λ=0.5, λ_g=0.3, λ_d=0.1, 20 epochs, 32K context, 16K response |
| Judge | Qwen3-VL-Plus（temperature 0, 8K token cap） |
| 数据公开 | 使用公开benchmark（MMSearch/InfoSeek/V*Bench/HR-Bench等），但论文未声明自有数据开源 |
