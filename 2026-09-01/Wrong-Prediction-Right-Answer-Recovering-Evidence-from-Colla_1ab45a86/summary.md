---
title: "Wrong-Prediction-Right-Answer-Recovering-Evidence-from-Colla"
source: https://arxiv.org/pdf/2608.31068v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 15:44:29"
field: "大模型可解释性与评测"
keywords: ["LLM interpretability", "reasoning evaluation", "output calibration", "probing", "score collapse", "additive correction"]
innovations: ["揭示readout gap证明答案表征存在于hidden state但被输出层bias遮蔽", "目标标签无关的2参数加法校正恢复per-example推理证据", "TF-IDF-missed与permutation null严格的per-example验证框架"]
benchmarks: ["ProofWriter", "ANLI", "FOLIO", "Controlled Synthetic Logic"]
---

# 论文速读：Wrong-Prediction-Right-Answer-Recovering-Evidence-from-Collapsed-LLM-Sequence-Scores

## 一句话总结
论文发现LLM在推理任务上"答错"往往不是因为缺乏推理能力，而是输出层的score坍塌（structural bias导致频繁标签被过度预测）。通过仅用25个无标签样本拟合两个加法偏移量，可恢复9–34个准确率点数，证明内部表征已编码正确答案但被晚期输出瓶颈遮蔽。

## 研究问题与动机
- **核心问题**：LLM在few-shot/zero-shot推理基准上表现为随机猜测（如33% accuracy），是否真的缺乏推理能力，还是仅因输出层score distortion导致？
- **现有方法不足**：（1）Hidden-state probe证明模型内部能解码正确答案，但属于"外部观察者"，不证明原生序列生成流程使用了该信息；（2）Output calibration方法通过全局分布重平衡提升aggregate accuracy，但无法保证per-example逻辑修复；（3）现有评测直接将wrong prediction等同于reasoning deficit，混淆了representation failure与scoring bottleneck。
- **关键现象**：Qwen3.5系列模型在balanced three-way label task（true/false/unknown）上，full candidate-answer string score坍塌到chance（~0.333），且99.9%预测集中在单一标签（unknown），形成predictive collapse。

## 核心贡献（创新点）
1. **诊断协议揭示readout gap**：首次系统对比answer-slot probe（~0.83）、same-position label logits（~0.57）与full string score（~0.33），证明信息损失发生在vocabulary projection与sequence aggregation两层。
2. **目标标签无关的最小校正器**：仅用两个自由度（固定一reference offset为0）拟合全局决策边界偏移，不访问ground-truth labels，不改变within-label ranking，仅作为threshold shift。
3. **严格的per-example验证框架**：引入TF-IDF-missed slice、count-preserving permutation baseline、deterministic subsampling三重控制，排除lexical shortcut与label-distribution matching解释。
4. **sample efficiency突破**：仅25个无标签样本即可使校正有效（30/30随机子采样均正增益），且从25到1000样本准确率变化≤0.022，表明恢复的是pre-existing threshold bias而非relearned mapping。
5. **跨模型与跨任务转移**：校正效果从Qwen3.5-2B/4B/9B转移到OLMo-2-1B与Llama-3.1-8B，并在synthetic lexical、ProofWriter、ANLI R2、FOLIO四类基准上保持一致性。

## 方法详解
**三步诊断流程（Figure 1）**：
1. **Probe验证表征存在**：在frozen hidden state上训练multinomial logistic regression（答案槽位置：`Answer: <answer>`标记后第一个token前），评估held-out accuracy。
2. **定位score瓶颈**：对比answer-slot probe、same-position label logits（$z_y = h^a(x)^\top w_y + b_y$）、first-label-token score、full candidate-answer string score（$s_y(x) = \sum_{t=1}^{|a_y|} \log p_\theta(a_{y,t}|x,r,a_{y,<t})$），识别information loss发生在unembedding geometry还是sequence aggregation。
3. **最小加法校正**：
   - 给定无标签in-domain score集$D_{id}$，校正预测：$\hat{y}_c(x) = \arg\max_y s_y(x) + c_y$
   - 固定$c_{ref}=0$（uniform shift不影响argmax），剩余两参数通过grid search最小化预测分布与目标prior $\pi$的MSE：$c^* = \arg\min_c \sum_y (q_c(y; D_{id}) - \pi_y)^2$
   - 校正后保留relative score differences，不改within-label ranking，仅移除shared label-level bias $\beta_y$。

**关键设计原则**：
- 零梯度更新模型参数，所有组件在frozen outputs上操作。
- Fit set与eval set严格disjoint。
- 校正器capacity有限（仅2自由度），无法learn新reasoning mapping。

## 实验与结果
**数据集**：
- Controlled synthetic logic（ID/Depth OOD/Lexical OOD/Stress，各1000样本，balanced three-way labels）
- ProofWriter（1000 ID fit + 1000 OOD eval）
- ANLI R2（R1为fit set，R2为adversarial held-out）
- FOLIO（203 fit + 1001 eval，sparse diagnostic）

**基线模型**：Qwen3.5-2B/4B/9B（post-trained）、Qwen3.5-Base系列、OLMo-2-1B、Llama-3.1-8B、Pythia-410M/12B（边界控制）

**核心结果**：
| 模型 | 任务 | Raw | Cal. | △ | Macro-F1 |
|------|------|-----|------|---|----------|
| Qwen3.5-4B | Synthetic lexical | 0.333 | 0.570 | +237 per-example | 0.572 |
| Qwen3.5-9B | Synthetic lexical | 0.333 | 0.602 | +269 | 0.591 |
| Qwen3.5-4B | ProofWriter | 0.333 | 0.653 | +320 | 0.650 |
| Qwen3.5-9B | ProofWriter | 0.334 | 0.678 | +344 | 0.678 |
| Qwen3.5-4B | ANLI R2 | 0.478 | 0.571 | +93 | 0.573 |
| Qwen3.5-9B | FOLIO | 0.348 | 0.638 | +291 | 0.632 |

**最强结果**：Qwen3.5-9B在ProofWriter上从chance恢复到0.678（+34.4个百分点），且permutation null gap达+0.344（p<0.001）。

**关键控制验证**：
- TF-IDF-missed硬例：Qwen3.5-4B/9B仍保持0.622/0.643，排除lexical shortcut。
- 25样本拟合：30/30随机子采样均正增益，饱和曲线显示>1000样本仅+0.022。
- Prior扰动：worst-case增益仍为正（Synthetic +0.193/+0.242，ProofWriter +0.281/+0.294）。
- Cross-family：OLMo-2-1B (+0.204)、Llama-3.1-8B (+0.144) 正增益；Pythia无显著恢复（边界控制）。
- Source-task ensemble融合：Source only 0.700 → Source + Calibrated LM scores 0.720（+20 per-example paired gain）。

## 相关工作脉络
1. **Probing vs. output scoring**：Alain & Bengio (2018)、Burns et al. (2023)、Belrose et al. (2025) 证明hidden states可decode答案，但未解决native generation利用问题；本文通过additive correction填补这一gap。
2. **Softmax bottleneck与unembedding misalignment**：Yang et al. (2017) 指出linear projection约束输出表达能力；本文将其具体化为label-bias $\beta_y$主导raw score。
3. **Output calibration**：Zhao et al. (2021) verbalizer normalization、Wang & Liu (2025) diagonal scaling等通过全局分布重平衡提升accuracy；本文强调aggregate improvement不能证明per-example reasoning修复，提出per-example gain与permutation null作为严格验证。
4. **Logical-reasoning benchmarks**：ProofWriter (Tafjord et al., 2021)、FOLIO (Han et al., 2024)、ANLI (Nie et al., 2020) 提供balanced labels与known ground truth；本文将其作为diagnostic testbed而非leaderboard。
5. **Surface form competition**：Holtzman et al. (2021) 指出高概率答案未必正确；本文发现collapsed distribution下模型将所有概率mass集中于单一标签（如unknown），而非simple surface-form competition。
6. **Chain-of-thought prompting**：Wei et al. (2022) 通过中间推理token分散表征负担；本文推测single-token classification将表征负担集中于collapse-prone unembedding step，为CoT提供mechanistic动机。

## 局限性与未来方向
- **Prior假设显式**：校正依赖已知label prior（本文所有任务balanced），未解决imbalanced/unknown deployment场景。
- **Observational而非causal**：Triangulation证明representation与scoring divergence，但未isolate neural circuits responsible for bottleneck。
- **非普遍适用**：Pythia模型无recovery（无collapsed ranking可供rescue），near-ceiling配置无headroom。
- **Additive校正仅修复shared bias**：无法处理per-example distortions（如surface-form competition、context-dependent unembedding misalignment），probe-sequence gap仍有0.17-0.20残余。
- **范围限制**：当前协议限于forced-choice tasks，open-ended generation需定义robust candidate-answer strings与permutation controls。

## 研究启发与可借鉴点
1. **诊断协议设计**：三阶段pipeline（probe→native readout→校正）可作为通用框架，用于审计任意LLM的"假性推理缺陷"，为benchmark evaluation提供更细粒度诊断。
2. **最小容量校正器验证**：2参数grid search + permutation null控制，证明校正恢复的是pre-existing结构而非relearned mapping，为calibration方法提供可证伪范式。
3. **TF-IDF-missed hard slice**：通过排除bag-of-words可解样本，构建adversarial evaluation regime，可直接迁移至其他NLI/reasoning任务的shortcut审计。
4. **与source-task transfer的互补性**：Source ensemble + calibrated sequence scores融合（0.700→0.720）表明native scores携带adapter未覆盖的target-domain evidence，为multi-model fusion提供新思路。
5. **Base vs. post-trained对比**：Table 11显示collapse在pretraining阶段已存在（Qwen3.5-Base string score ~0.33-0.37），instruction tuning仅amplify bias，提示retraining或alignment阶段需关注output-layer calibration。

## 关键术语表
**Readout gap**：Probe准确率高但native sequence score坍塌的现象，表明答案信息存在于hidden state但未被输出层暴露。
**Answer-slot probe**：在`Answer: <answer>`标记后的hidden state上训练的线性分类器，避免generated token的cascade effect。
**Full candidate-answer string score**：对固定候选答案字符串进行teacher-forced logprobability求和得到的native zero-shot分数。
**Prior-conditioned additive offset**：仅用两个自由度的全局偏移量校正label bias，不改变candidate relative ranking。
**Per-example gain**：校正后正确但原始错误的答案数减去原始正确但校正后错误的答案数，衡量per-instance修复效果。
**Count-preserving permutation baseline**：保持预测label直方图不变但随机打乱预测结果，用于排除distribution-matching解释。
**TF-IDF-missed slice**：从评估集中排除bag-of-words分类器可解样本的子集，用于审计lexical shortcut。
**Scoring bottleneck**：信息从hidden state到native output的转换过程中，因unembedding geometry与sequence aggregation导致的information loss。

## 可复现要素
- **数据集**：Controlled synthetic logic（固定script与seed生成，论文提供详细construction protocol）；ProofWriter、ANLI、FOLIO均为公开benchmark。
- **代码/权重**：使用Qwen3.5、OLMo-2、Llama-3.1、Pythia open-weight checkpoints（huggingface可下载）；偏移校正为确定性grid search，实现简单。
- **关键超参**：Fit set大小（25/100/500/1000）；anchor label固定为unknown（offset=0）；prior设为uniform（三标签balanced）；probe训练2000行、3 seeds。
- **协议细节**：Format-matched prompt ending with `Answer: <answer>`；left padding至max length 1024；训练/评估split严格disjoint。
