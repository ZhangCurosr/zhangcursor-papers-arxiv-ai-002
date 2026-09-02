---
title: "StateSwap-Probing-Support-Elimination-Hidden-States-in-Multi"
source: https://arxiv.org/pdf/2609.01081v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 22:18:16"
field: "大语言模型可解释性与推理机制"
keywords: ["mechanistic interpretability", "activation substitution", "prompt framing", "multiple-choice reasoning", "contrastive activation steering", "state tracking"]
innovations: ["提出untrained [STATE] token作为残差流read-write接口探测框架敏感表征", "发现并定位SUP/ELIM诱导的中间层可分离激活峰并通过跨框架替换提升准确率与一致性", "证明双框架均值差转向比对比激活加法(CAA)在层级响应上更稳定"]
benchmarks: ["MMLU-17", "MedQA-CH"]
---

# 论文速读：StateSwap: Probing Support–Elimination Hidden States in Multiple-Choice Questions

## 一句话总结
论文提出 StateSwap 框架，通过引入未训练的 `[STATE]` 特殊 token 作为残差流干预接口，在支持型（SUP）与排除型（ELIM）两种语义等价但表述不同的提示框架下，探测并定位模型中间层可分离的决策相关隐藏状态，并通过跨框架激活替换系统性地改变模型预测、提升准确率与跨框架一致性。

## 研究问题与动机
1. LLM 在面对相同的多选题时，采用 SUP 框架（"找出正确选项"）与 ELIM 框架（"找出错误选项"）会产生不一致的答案，但现有研究对这种分歧的内部表征机制缺乏理解。
2. 已有工作表明过程消除（Process-of-Elimination, PoE）策略可以提升 LLM 多选题性能，但也有研究报道排除型提示反而降低准确率，说明两种框架的语义等价性并不保证行为一致性。
3. 现有的提示工程与推理增强方法（如 prompt ensembling、Chain-of-Thought）主要关注输出层面的聚合，未从**激活空间干预**的角度揭示框架敏感表征的因果作用。
4.  mechanistic interpretability 中的 activation patching 与 steering 方法已证明内部表征具有行为相关性，但尚无工作系统研究提示框架差异如何塑造并体现在中间层隐藏状态中。

## 核心贡献（创新点）
1. **提出 `[STATE]` 接口与双框架对照协议**：通过插入无预训练词汇语义的特殊 token 并严格对齐 token 位置，构建了最小差异的双框架输入，使框架效应可从文本内容中隔离。*与已有方法的区别在于，它不依赖参数微调或连续 prompt 优化，而是直接操作原始残差流中的固定位置激活。*
2. **定位中间层可分离的框架敏感表征**：发现 SUP 与 ELIM 诱导的 `[STATE]` 激活在浅层高度混合、在中间层显著分离、在深层又趋于收敛，且该模式在 Qwen-2.5-7B 和 GLM-4-9B 上保持一致。*区别于此前认为决策相关表征均匀分布或仅存在于浅/深层的观点，本文证明存在明确的"中间层峰"。*
3. **激活替换提升准确率与跨框架一致性**：在定位窗口内交换框架敏感的 `[STATE]` 激活后，两种模型的 ACC 和 Jaccard 均显著提升（MMLU-17 上 Jaccard 最高提升约 11.57%，ACC 提升约 0.79%~3.14%），且方向不对称——替换后更倾向于增强错误排除确定性。*与单纯 prompt ensembling 不同，替换效应超出了投票聚合所能解释的范围。*
4. **平均差转向方向比对比激活加法（CAA）更稳定**：在去掉 `[STATE]` 接口的纯人口级别 steering 实验中，双框架均值差方向的层级响应方差显著低于匹配的 CAA 方向（`a=-1` 时 GLM 层间波动从 50.54 降至 19.90 分，Qwen 从 60.89 降至 28.08 分）。*为框架对照类 steering 方向的选择提供了实证依据。*

## 方法详解
1. **输入构造与状态接口**：对每个问题 $q$，构造配对提示 $\mathbf{x}_{\text{SUP}}(q) = \text{Concat}(\text{SUP}, q, [\text{STATE}])$ 与 $\mathbf{x}_{\text{ELIM}}(q) = \text{Concat}(\text{ELIM}, q, [\text{STATE}])$；指令段通过右填充对齐至相同 token 长度，确保 `[STATE]` 始终位于最终位置 $t_S = T$；`[STATE]` 通过 `tokenizer.add_special_tokens` 注册、嵌入矩阵 resize 后保持初始化值不变，不参与训练。
2. **状态替换（State Substitution）**：选定连续干预窗口 $W \subseteq \{1,\dots,L\}$，对框架 $I$ 的 prompt 在执行过程中缓存 $\mathbf{s}^{(k)}_{\text{cache}}(\bar{I}, q)$，并在每个 $k \in W$ 层后覆写 $\mathbf{s}^{(k)}(I, q) \leftarrow \mathbf{s}^{(k)}_{\text{cache}}(\bar{I}, q)$，其余残差流分量保持不变，继续正常自回归生成。
3. **干预窗口定位**：对每个层 $l$ 与特征窗口 $[j:j+w)$，计算配对 Cohen's $d$ 统计量；两种标量化方式：① 随机方向诊断 $\phi^{\text{rand}}_{j,w}(\mathbf{z}) = \mathbf{u}_{[j:j+w)}^\top \mathbf{z}$（每轮随机采样单位向量 $\mathbf{u}$）；② 平均差诊断 $\phi^{\text{md}}_{l,j,w}(\mathbf{z}) = \hat{\mathbf{v}}_l(j,w)^\top \mathbf{z}$，其中 $\mathbf{v}_l(j,w) = \frac{1}{N}\sum_i (\mathbf{s}^{{\text{ELIM}},i}_{[j:j+w),l} - \mathbf{s}^{{\text{SUP}},i}_{[j:j+w),l})$；按 Top-K 均值聚合得到每层强度 $\tilde{d}_l$，选取 $\tilde{d}_l \geq 0.8 \cdot \max_{l'} \tilde{d}_{l'}$ 的相邻层形成 $W$（Qwen: Layers 11–20；GLM: Layers 21–40），该阈值独立于下游任务性能。
4. **评估协议**：每个框架内使用两种不同模板的提示各进行一次 greedy decoding（do_sample=False, max_new_tokens=1024），组内严格多数投票；输出解析为 `Correct options:[...]` 与 `Wrong options:[...]` 集合，评估 ACC、Jaccard（跨框架正确题号集的交集/并集）、LAC（长度感知余弦相似度）与 BERTScore-F1。

## 实验与结果
- **数据集**：MMLU-17（4,607 题，17 个推理导向子集）与 MedQA-CH（1,015 题，中文医学多选题），共 5,622 题；50 条校准样本仅用于定位窗口，不进入测试。
- **模型**：Qwen-2.5-7B-Instruct、GLM-4-9B，zero-shot、greedy decoding、固定 seed=123。
- **主要结果（Table 1）**：
  - Qwen-2.5-7B / MMLU-17：ACC(SUP) 65.52→66.31（+0.79），ACC(ELIM) 65.38→74.36（+8.98），Jaccard 66.53→78.10（+11.57）。
  - Qwen-2.5-7B / MedQA-CH：ACC(SUP) 77.12→80.26（+3.14），ACC(ELIM) 77.45→81.58（+4.13），Jaccard 75.46→78.10（+2.64）。
  - GLM-4-9B / MMLU-17：ACC(SUP) 66.47→68.74（+2.27），ACC(ELIM) 69.21→72.36（+3.15），Jaccard 72.96→78.21（+5.25）。
  - GLM-4-9B / MedQA-CH：ACC(SUP) 73.54→76.82（+3.28），ACC(ELIM) 77.92→82.48（+4.56），Jaccard 78.67→82.48（+3.81）。
- **强结果**：MMLU-17 上 GLM-4-9B Jaccard 提升 5.25 个百分点；MedQA-CH 上 Qwen-2.5-7B ELIM ACC 提升 8.98 个百分点（从 65.38 到 74.36）。
- **Ablation**：① `[STATE]` vs 最终内容 token：后者 LAC≈0.978、BERTScore-F1≈0.944，几乎无行为改变；② 随机/零替换：LAC≈0.62–0.66，显著破坏稳定性；③ `[STATE]` 错位放置（插于选项前/问题前）：>95% 生成失败或不可解析；④ 随机 label 控制：可分离性始终处于低位，排除假阳性。
- **Ensemble 对比（Table C.1/C.2）**：Dual StateSwap ensemble 在两种模型上均超越朴素 SUP+ELIM ensemble 与加权投票方案，证明增益非简单聚合所致。
- **Steering 对比（Figure 7/Table C.5）**：双框架均值差方向在 `a=-2,-1,2` 时精度全面优于 CAA，层间波动更窄。

## 相关工作脉络
1. **Process-of-Elimination (PoE) / Elimination-based reasoning**（Ma & Du, 2023; Zhu et al., 2025; Wang et al., 2026）：关注 prompt 层面利用排除策略提升多选题性能，但未深入内部表征机制；本文从激活空间提供因果证据，解释 PoE 有效/失效的表征根源。
2. **Activation patching & causal tracing**（Meng et al., 2022; Heimersheim & Nanda, 2024）：通过局部修改激活验证行为相关性；本文继承干预思想，但目标不是事实编辑而是**框架敏感性表征**的探测与转移。
3. **Contrastive Activation Addition (CAA)**（Rimsky et al., 2024）与 **Activation steering**（Li et al., 2023; Zou et al., 2023; Lee et al., 2025）：通过数据集级对比方向引导模型输出；本文提出基于提示框架对照的 mean-difference direction，并证明其在层间稳定性上优于 CAA。
4. **Prompt/Prefix tuning**（Lester et al., 2021; Li & Liang, 2021; Liu et al., 2022）：学习连续嵌入以控制行为；本文不使用任何训练，仅利用固定 untrained token 作为 read-write 接口，属于 zero-cost intervention。
5. **Self-consistency / Prompt ensembling**（Wang et al., 2022; Lin et al., 2024）：通过多路推理聚合提升鲁棒性；本文通过替换内部状态实现类似效果，但机制不同——聚合是输出级加权，StateSwap 是表示级因果干预。
6. **State tracking & synthetic mechanistic studies**（Zhang et al., 2025; Lu et al., 2025）：证明 Transformer 可在中间/深层 MLP 神经元中发展出状态表示机制；本文实证发现自然语言任务中框架敏感状态同样集中于中间层。

## 局限性与未来方向
1. 实验局限于四选项多选题与确定性 greedy decoding，未验证在采样解码、长链多步生成或开放生成任务下的有效性。
2. 方法依赖结构化 MCQ 输出解析（`Correct/Wrong options`），扩展至开放域问答需重新设计配对提示与一致性度量。
3. 引入 untrained `[STATE]` token 虽经随机初始化鲁棒性验证，但仍可能引入轻微分布偏移，无法保证效应完全独立于接口设计。
4. 需要白盒激活访问，黑盒 API 场景下不可直接应用；仅用 prompt 变体投票降噪，未集成进端到端系统。
5. 讨论部分指出 SUP/ELIM 可能隐含更细粒度的域/置信度/策略异质性，当前分析对这些 within-class 结构进行了 marginalize，后续可做分层或多因子分解。

## 研究启发与可借鉴点
1. **Untrained special token 作为干预接口**：通过注册不携带先验词汇语义的特殊 token 并固定其位置，可在几乎零成本的前提下获得一个"干净"的残差流读写端口，适用于各类表示探测与干预研究。
2. **双框架最小差异对照设计**：保持问题、选项、语言、解码全部一致，仅反转"选对/选错"的指令语义，可有效剥离框架效应与其他混杂因素，为表征可分离性提供干净因果检验。
3. **中间层峰定位策略**：使用配对 Cohen's d + Top-K 均值聚合 + 0.8 阈值经验准则定位干预窗口，避免依赖下游任务性能调参，该方法可迁移至其他表征对齐/干预任务。
4. **响应稳定性指标的联合使用**：同时报告 LAC 与 BERTScore-F1 可区分"结构敏感替换"与"随机扰动"——前者保持高语义相似度，后者显著退化，为干预有效性提供多维证据。
5. **均值差转向 vs CAA 的比较协议**：在控制校准样本、层、系数、answer 边界 hook 一致的条件下直接对比两类方向，为 steering method 选择提供了可复用的评测范式。

## 关键术语表
- **SUP / ELIM framing**：支持型（寻找正确选项）与排除型（寻找错误选项）两种逻辑等价但表述不同的多选题提示框架。
- **[STATE] token**：论文引入的未训练特殊 token，注册于词表末尾、嵌入保持初始化，作为残差流中固定的读-写接口，无先验词汇语义。
- **State substitution（激活替换）**：在选定的连续层窗口内，将框架 $I$ 的 `[STATE]` 残差激活替换为互补框架 $\bar{I}$ 的缓存激活，保持文本输入与其余计算不变。
- **Cohen's d（配对）**：跨问题配对差异 $(\mathbf{s}_{\text{ELIM}} - \mathbf{s}_{\text{SUP}})$ 的标准化均值，用于衡量两框架在特征窗口上的可分离强度。
- **Mean-difference steering direction**：由多实例配对表征均值差归一化得到的向量，用于在 answer 边界处施加层wise 线性干预。
- **Jaccard index（跨框架）**：两个框架下正确预测题号集合的交集除以并集，衡量模型在 SUP 与 ELIM 间决策一致性。
- **LAC（Length-Aware Cosine Similarity）**：句子嵌入余弦相似度乘以长度比惩罚项，评估干预前后完整响应的全局语义相似度。
- **Intervention window W**：基于 separability 诊断选出的连续 Transformer 层区间（Qwen: 11–20；GLM: 21–40），仅在此窗口执行替换。

## 可复现要素
- **数据集**：MMLU-17（官方 test split，4,607 题）、MedQA-CH（官方 test split，1,015 题）；校准集 50 题来自 train split。
- **代码开源**：https://github.com/Cha0Ga0/SWAPSTATE
- **模型**：Qwen-2.5-7B-Instruct、GLM-4-9B（open-weight）。
- **解码**：greedy（do_sample=False，max_new_tokens=1024，torch_dtype=float16，KV cache 开启），seed=123。
- **上下文与填充**：截断至模型最大上下文长度，右侧 PAD 最多 64 token；指令段 pad 对齐保证 `[STATE]` 处于相同绝对位置。
- **超参数**：论文未单独声明 learning rate / weight decay 等（因 inference-only），但明确校准集大小 N=50、bootstrap resamples=500、Top-K 聚合、阈值 0.8、特征窗口 w 搜索范围（默认 w=6）等细节；详见 Appendix A.3 / B。

---
