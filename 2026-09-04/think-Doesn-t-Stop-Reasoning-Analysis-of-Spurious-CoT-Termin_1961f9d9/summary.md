---
title: "think-Doesn-t-Stop-Reasoning-Analysis-of-Spurious-CoT-Termin"
source: https://arxiv.org/pdf/2609.03633v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 15:34:08"
field: "大语言模型推理效率与可解释性"
keywords: ["Chain-of-Thought", "Early Exit", "Large Reasoning Models", "Mechanistic Interpretability", "Attention Intervention", "Spurious Termination"]
innovations: ["提出虚假CoT终止概念并系统刻画EoT再生现象", "设计无训练注意力偏置方法EAB建立因果证据", "跨模型/方法/基准的细粒度行为分析与对照实验设计"]
benchmarks: ["GSM8K", "MATH-500", "AMC 2023", "AIME 2024", "GPQA-Diamond"]
---

Doesn't Stop Reasoning: Analysis of Spurious CoT Termination"。

让我先理解论文的核心内容：
1. 研究了CoT早期退出方法中注入的结束思考标记（EoT, </think>）导致的虚假CoT终止现象
2. 发现注入的EoT并不总是能干净地触发从推理到回答的转换
3. 提出了Exit-token Attention Biasing (EAB)方法来增加对注入EoT的注意力
4. 在4个LRM、5个基准和2个早期退出方法上验证了效果

现在我需要按照用户指定的格式输出笔记：
- 一句话总结
- 研究问题与动机
- 核心贡献
- 方法详解
- 实验与结果
- 相关工作脉络
- 局限性与未来方向
- 研究启发与可借鉴点
- 关键术语表
- 可复现要素
- META信息块

让我仔细阅读论文并提取关键信息。
</think>

# 论文速读：think-Doesn-t-Stop-Reasoning-Analysis-of-Spurious-CoT-Termin

## 一句话总结
本文系统研究了CoT早期退出方法中注入的**结束思考标记（EoT, `</think>`）**在大型推理模型（LRMs）中导致的**虚假CoT终止（spurious CoT termination）**现象——即注入的EoT未能干净触发推理→回答转换，而是引发模型在回答阶段继续生成类推理内容，直到再次生成EoT为止。作者提出**退出标记注意力偏置（EAB）**作为因果探测工具，证实增加对注入EoT的注意力可有效减少该现象。

## 研究问题与动机
1. **早期退出方法的隐性缺陷**：训练无关的CoT早期退出方法通过在中间位置注入EoT强制触发推理→回答转换，但实际观察发现这往往导致回答阶段长度膨胀（甚至延长10倍以上），反而抵消了效率收益。
2. **格式匹配≠行为控制**：现有方法假设外部插入EoT标记即可控制模型的内部状态转换，但未验证该标记是否真正被后续生成过程"注意到"并生效。
3. **现象缺乏细粒度刻画**：已有工作仅观察到"退出后模型可能继续推理"，但未系统分析EoT再生的分布特征、前置行为模式及其与注意力机制的因果关系。

## 核心贡献（创新点）
1. **现象定义与刻画**：首次将"注入EoT后回答阶段仍出现类推理行为直至再次生成EoT"的现象定义为**虚假CoT终止**，并从再生率（ERR）、回答长度膨胀、Wait标记分布、boxed表达式位置等多维度建立表征体系。
2. **注意力因果探测框架**：提出**Exit-token Attention Biasing (EAB)**——通过在解码阶段对注入EoT键位置的注意力logit添加常数偏置α，实现无训练干预，首次在注意力层面建立EoT关注程度与终止行为的因果关系。
3. **机制解释与对照验证**：证明ERR随早期退出压缩率升高而递增；偏置目标仅限EoT时有效（Offset-1甚至产生反向效果），说明效应具标记特异性而非序列长度效应。
4. **跨模型/方法的泛化验证**：在4个尺度跨越数量级的LRM（R1-Distill-1.5B/14B、Qwen3-14B、QwQ-32B）、5个数学/科学推理基准及2种早期退出方法（DEER、DynaSoR）上系统性验证，揭示不同模型家族中现象成因差异（如QwQ-32B主要受学习生成模式驱动而非注意力不足）。

## 方法详解
**EAB（Exit-token Attention Biasing）核心设计**：

- **符号定义**：完整token序列$x=(x_1,\ldots,x_T)$，注入EoT位于位置$I$；所有$i>I$的token构成回答阶段。$s_{i,j}^{(l,h)}$表示第$l$层第$h$头从query位置$i$到key位置$j$的softmax前注意力分数。

- **偏置公式**：
$$
\tilde{s}_{i,I}^{(l,h)} = s_{i,I}^{(l,h)} + \alpha \quad \text{for all } i > I, l, h
$$
其余位置的logit保持不变（$j \neq I$），修改后的注意力权重经softmax重新归一化：$\tilde{A}_{i,j}^{(l,h)} = \mathrm{softmax}_j(\tilde{s}_{i,:}^{(l,h)})$。

- **实现细节**：仅作用于回答阶段（$i>I$），不改变推理阶段计算；通过patch transformers的attention forward pass实现，prefill使用原Flash Attention路径，decode路由至xformers.memory_efficient_attention以支持任意加法偏置掩码。

- **对照方法**：
  - **Double-EoT**：在注入EoT后追加第二个EoT token作为对照
  - **Block-EoT**：将回答阶段所有位置的EoT logit置为$-\infty$
  - **Ans-Prefix**：在注入EoT前添加显式转换短语"Based on the reasoning up until now, I will now present my final answer."
  - **Post-Box/Post-Ans-Box**：在EoT后立即插入$\boxed{}$模板

## 实验与结果
**实验设置**：
- **模型**：DeepSeek-R1-Distill-Qwen 1.5B/14B、Qwen3-14B、QwQ-32B
- **基准**：GSM8K、MATH-500、AMC 2023、AIME 2024、GPQA-Diamond
- **早期退出方法**：DEER（在Wait处探测答案置信度）、DynaSoR（固定间隔探测一致性）
- **指标**：EoT再生率（ERR）、回答阶段长度（#Ans）、准确率（Acc）

**关键结果**：
| 模型 | 方法 | MATH | AMC | AIME | GSM8K | GPQA | AVG |
|------|------|------|-----|------|-------|------|-----|
| R1-Distill-14B | DEER | 19.8% | 22.5% | 26.7% | 3.3% | 9.1% | 16.3% |
| R1-Distill-14B | DynaSoR | 35.8% | 40.0% | 40.0% | 8.0% | 18.2% | 28.4% |
| QwQ-32B | DEER | 40.4% | 57.5% | 50.0% | 9.6% | 12.1% | 33.9% |

- **EoT再生与长度膨胀关联**：发生EoT再生的样本，其中位数回答长度是无再生样本的2~22倍（QwQ-32B + DynaSoR在AIME上超15倍）
- **EAB核心效果**（R1-Distill-14B + DEER, α=4）：
  - ERR从16.3%降至3.8%（↓12.5pp）
  - 平均回答长度从1,252 token降至667 token（↓47%）
  - 准确率从73.3%提升至75.1%（+1.8pp）
- **EAB选择性**：对已存在虚假终止的样本ERR降至25%以下，对无虚假终止样本几乎无影响（ERR维持≈0%）
- **模型差异**：QwQ-32B的EoT再生主要由习得生成模式驱动（No-CoT ERR>60%），EAB改善有限且偶尔降低准确率

## 相关工作脉络
1. **CoT早期退出方法**：DEER（Yang et al., 2026）通过Wait处探测答案置信度触发退出；DynaSoR（Fu et al., 2026）基于固定间隔探测答案一致性；本文聚焦这两种方法中EoT注入引发的副作用，区别于前置prompt跳过推理阶段的研究（Zhu et al., 2025; Zhang et al., 2025d）。
2. **推理控制与压缩**：TokenSkip（Xia et al., 2025）可控压缩CoT；COT-Valve（Ma et al., 2025）长度可压缩CoT调优；本文从"终止信号失效"角度揭示效率优化的另一类隐性成本。
3. **注意力干预**：Zhang et al. (2024) 后验注意力重加权；Li et al. (2023) 激活空间干预；本文在EoT这一特定标记上建立因果探测，提供可复用的"标记级注意力调控"范式。
4. **机制解释学**：Thought Anchors（Bogdan et al., 2025）识别关键推理步；Thinking Sparks！（Park et al., 2025）发现涌现注意力头；本文沿此脉络聚焦"阶段转换信号"的内在处理机制。
5. **过思考分析**：ThoughtTerminator（Pu et al., 2025）基准评测过度推理；本文从"强制终止"角度补充理解LRMs的推理结束机制。

## 局限性与未来方向
1. **因果推断的间接性**：无法直接观测退出标记后的内部状态，依赖行为指标（ERR、Wait分布、boxed位置）和干预实验（EAB）间接推断机制。
2. **EAB的工程化挑战**：自适应偏置强度选择标准缺失；现有推理后端（如vLLM）不支持逐步注意力访问，需定制实现。
3. **单一标记视角**：仅关注EoT本身，未系统考察其前后上下文（如Full-CoT中boxed-before-EoT模式、Ans-Prefix效果）对转换信号的协同作用。
4. **模型家族差异未深挖**：QwQ-32B的EoT再生机制（习得模式 vs. 注意力不足）与其他模型存在本质差异，未进一步建模分类。

## 研究启发与可借鉴点
1. **EAB范式可迁移**：任何依赖特殊分隔符（如`<|end|>`、`***`）触发状态转换的框架，均可借用此类"注意力偏置探测"诊断信号是否真正生效。
2. **对照组设计严谨**：同时对比Double-EoT（序列操纵）、Block-EoT（强制屏蔽）、Ans-Prefix（提示工程）等正交对照，清晰剥离"注意力机制"与"表面格式"的贡献。
3. **细粒度分段分析**：将回答阶段拆分为pre-regen（注入EoT至再生EoT）与post-regen两段，分别统计#Token、#Wait、boxed密度，精准定位行为异常段落。
4. **模型差异诊断框架**：通过No-CoT ERR基线区分"机制性不足"（如R1-Distill系列）与"习得模式漂移"（如QwQ-32B），为后续方法选择提供决策依据。
5. **准确率-效率权衡的可解释分析**：指出EAB导致的准确率下降主要源于早期退出点移除的推理内容不可补偿，而非干预本身有害——这一归因为后续工作优化退出策略提供明确方向。

## 关键术语表
**Chain-of-Thought (CoT)**：大模型在输出最终答案前逐步生成中间推理步骤的技术，显著提升复杂任务表现。
**Large Reasoning Model (LRM)**：经过强化学习等训练、显式支持长链推理的模型家族，如DeepSeek-R1、QwQ系列。
**End-of-Think (EoT)**：特殊标记`</think>`，与`<think>`配对构成显式两阶段格式，标记推理阶段结束。
**Early Exit**：在推理尚未自然结束时强制终止的策略，以节省计算成本。
**Spurious CoT Termination**：注入EoT后模型未在后续回答阶段干净转换，而是继续生成类推理内容直至再次产生EoT的现象。
**EoT Regeneration Rate (ERR)**：回答阶段中发生EoT再生的样本比例，本文核心评估指标。
**Exit-token Attention Biasing (EAB)**：在回答阶段生成时对注入EoT键位置的注意力logit施加常数偏置α的无训练干预方法。
**Pre-regen / Post-regen**：将回答阶段以最后一次再生EoT为界分割为"注入EoT至再生EoT"（pre-regen）和"再生EoT之后"（post-regen）两段。

## 可复现要素
- **数据集**：GSM8K、MATH-500、AMC 2023、AIME 2024、GPQA-Diamond（均公开可用）
- **模型权重**：DeepSeek-R1-Distill-Qwen-1.5B/14B（MIT License）、Qwen3-14B/QwQ-32B（Apache 2.0）
- **代码开源**：是，GitHub仓库 https://github.com/Seunghee-Koh/Spurious-CoT-Termination
- **关键超参**：
  - EAB偏置强度α：默认2，R1-Distill-14B用4
  - 推理阶段预算上限：R1-Distill/QwQ为总预算（16384 token）的60%，Qwen3为80%
  - 探测间隔：DynaSoR固定256 token
  - 退出阈值：DEER用Wait处答案置信度>0.95

---
