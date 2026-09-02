---
title: "Wrong-Prediction-Right-Answer-Recovering-Evidence-from-Colla"
source: https://arxiv.org/pdf/2608.31068v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 15:44:18"
field: "大语言模型可解释性与评估"
keywords: ["LLM interpretability", "reasoning diagnostics", "output calibration", "readout gap", "sequence scoring", "few-shot correction"]
innovations: ["三阶段position-matched诊断协议定位推理信号在输出层丢失位置", "2参数无标签加法修正仅需25样本即可恢复崩溃序列得分", "TF-IDF-missed+permutation null+deterministic subsampling三重逐例验证框架"]
benchmarks: ["ProofWriter", "ANLI R2", "FOLIO", "Controlled Logic (synthetic lexical/depth)"]
---

# 论文速读：Wrong-Prediction-Right-Answer-Recovering-Evidence-from-Collapsed-LLM-Sequence-Scores

## 一句话总结
论文揭示大语言模型在推理任务中常见的"零样本失败"并非真正缺乏推理能力，而是输出层的结构性评分偏差导致正确答案被掩盖；通过仅用两个参数的无标签加法修正，可从原始序列得分中恢复9–34个准确率点，证实内部逻辑仍完整保留。

## 研究问题与动机
- **错误预测 ≠ 能力缺失**：现有评估默认模型输出错误答案即代表缺乏推理能力，但未区分"内部表示"与"外部表达"两个独立机制。
- **隐藏状态探针显示正确答案已被编码**：多项研究表明模型内部早已计算出正确答案，但native sequence scoring在最终输出时因结构性偏见（如偏好常见标签"unknown"）而坍缩。
- **既有方法无法逐例验证**：probe方法仅证明理论可解码性，不证明原生评分路径是否利用该信息；现有校准方法（如verbal normalization）仅改善全局分布，无法确认逐实例的逻辑对齐。
- **输出层几何畸变是瓶颈**：unembedding矩阵的固定线性变换将复杂推理流形映射到词表空间时产生几何失真，导致正确答案被频繁标签覆盖。

## 核心贡献（创新点）
1. **诊断协议揭示"readout gap"**：系统性比较hidden-state probe、single-token logit与full sequence score，定位推理信号在输出层投影与序列聚合两阶段丢失。
2. **无标签最小修正方法**：仅需2个参数（全局偏置偏移）拟合25个未标注样本，即可恢复崩溃的序列得分，不触碰模型内部权重。
3. **逐例强度验证框架**：引入TF-IDF-missed slice、label-count permutation baseline、deterministic subsampling三重控制，排除表面词匹配与分布匹配解释。
4. **跨模型与跨任务迁移验证**：在Qwen3.5-2B/4B/9B、OLMo-2-1B、Llama-3.1-8B及ProofWriter、ANLI、FOLIO、controlled logic四类任务上一致有效。

## 方法详解
- **三阶段诊断管道**：(i) 验证答案是否编码于隐藏状态（probe）；(ii) 测试冻结输出层是否能暴露该信息（same-position logits vs. full string score）；(iii) 施加最小加法修正并逐例验证恢复。
- **全候选答案字符串得分**：对固定续写$ a_y $计算teacher-forced log概率之和 $ s_y(x) = \sum_{t=1}^{|a_y|} \log p_\theta(a_{y,t}|x,r,a_{y,<t}) $，raw prediction为 $\arg\max_y s_y(x)$。
- **Answer-slot probe**：读取"Answer: <answer>"标记后、标签token前的隐藏状态 $ h^a(x) \in \mathbb{R}^d $，训练独立逻辑回归超平面，绕过词表投影约束。
- **原生得分几何**：$ z_y = h^a(x)^\top w_y + b_y $，若结构偏置 $ b_y $ 过大或 $\|w_y\|$ 偏向特定词汇子集，预测坍缩。
- **Prior-conditioned offset correction**：$ \hat{y}_c(x) = \arg\max_y s_y(x) + c_y $，固定一个参考偏移为0，通过确定性网格搜索最小化预测分数与目标先验的平方差，仅调整全局决策边界，不改任何case内排名。
- **三重控制**：TF-IDF-missed slice排除浅层词重叠捷径；label-count permutation baseline验证per-example对齐超出直方图匹配；25–500样本子采样验证低维阈值偏置假设。

## 实验与结果
- **数据集**：Controlled logic（synthetic lexical/depth/stress）、ProofWriter、ANLI R2、FOLIO，均为平衡三元标签（true/false/unknown）。
- **模型**：主实验Qwen3.5-2B/4B/9B（post-trained与Base），跨家族验证OLMo-2-1B、Llama-3.1-8B，Pythia作边界对照。
- **Readout gap**（Table 1）：Qwen3.5-9B answer-slot probe达0.830（lexical OOD），而full string score仅0.333（chance），gap统计显著（$ p<.001 $）。Base checkpoint同样复现，表明先于instruction tuning存在。
- **修正恢复**（Table 2）：Qwen3.5-4B/9B在synthetic lexical从0.333升至0.570/0.602（+23.7/+26.9点）；ProofWriter升至0.653/0.678（+32.0/+34.4点）；ANLI R2从0.478→0.571；FOLIO从0.348→0.638。
- **样本效率**（Table 3/Table 20）：仅25个未标注样本即实现正增益，30/30 subsample均为正；25→1000样本变化≤0.022。
- **捷径控制**（Table 4）：TF-IDF-missed subset上Qwen3.5-4B/9B仍保持0.622/0.643；permutation null gap达+0.287/+0.305。
- **跨家族迁移**（Table 6）：OLMo-2-1B +0.204、Llama-3.1-8B +0.144；Pythia-410M/12B无显著恢复，划定能力边界。
- **最强结果**：Qwen3.5-9B在ProofWriter OOD从0.334→0.678（+34.4点，paired $ p<.001 $）。

## 相关工作脉络
- **Probing研究**（Alain & Bengio 2018; Burns et al. 2023; Belrose et al. 2025）：证明LLM内部编码推理信息，但未验证native scoring是否利用——本文通过additive correction桥接此gap。
- **Softmax bottleneck与unembedding对齐**（Yang et al. 2017; Xiao et al. 2025）：解释输出层几何畸变，本文将其实证定位为sequence-level aggregation瓶颈。
- **输出校准方法**（Zhao et al. 2021; Wang & Liu 2025; Sanz-Guerrero & von der Wense 2025）：仅改善全局分布指标，本文强调per-example gain与null-gap作为诊断标准。
- **Logical reasoning benchmarks**（ProofWriter, ANLI, FOLIO; Clark et al. 2020; Tafjord et al. 2021; Nie et al. 2020; Han et al. 2024）：用作结构化测试床，分离表征失败与评分瓶颈。
- **CoT与单token分类对比**（Wei et al. 2022）：本文暗示collapse源于多跳推理压缩至单token分类，为CoT有效性提供机制解释。

## 局限性与未来方向
- **先验假设限制**：校准依赖已知label prior，对真实部署中类别不平衡或未知分布场景不适用。
- **观测性非因果性**：probe、logits与修正得分仅三角定位分叉点，未隔离神经回路层面的因果机制。
- **部分恢复上限**：修正仅移除全局标签级偏置$\beta_y$，无法修复surface-form competition或context-dependent unembedding错位等逐例畸变。
- **适用范围边界**：Pythia未出现坍缩 Ranking故无法恢复；高基线配置（如ANLI-9B raw 0.638）无恢复空间。
- **扩展至开放生成**：当前方法依赖forced-choice format与candidate-answer string，推广至free-form generation需重新定义控制机制。

## 研究启发与可借鉴点
- **诊断协议可复用**：position-matched readout chain（prompt-end probe → answer-slot probe → same-position logits → full string score）可作为LLM能力评估的标准审计管线。
- **最小修正框架**：2-parameter additive offset fitted on unlabeled data的思路可迁移至其他存在output-layer bias的任务（如多分类、NLI、事实验证）。
- **三重控制设计**：TF-IDF-missed + permutation null + deterministic subsampling组合可有效排除捷径解释，建议纳入后续benchmark评估协议。
- **评估实践启示**：在判定模型"缺乏推理能力"前，应先检查prediction histogram是否坍缩于单一标签；排行榜评分可能因default calibration差异而不公平惩罚某些模型。
- **与CoT结合机会**：可将collapsed ranking诊断与chain-of-thought扩展结合，验证多步生成是否缓解单token unembedding bottleneck。

## 关键术语表
- **Readout gap**：hidden-state probe准确率显著高于native sequence scoring的现象，表明答案信息存在于内部表示但未被输出层暴露。
- **Answer-slot probe**：在"Answer: <answer>"标记后的隐藏状态上训练的线性分类器，绕过词表投影约束直接解码标签。
- **Prior-conditioned offset correction**：无标签加法修正，通过两个自由参数全局偏移候选得分以匹配目标label prior，不改case内相对排名。
- **TF-IDF-missed slice**：从评测集中剔除bag-of-words分类器可解的实例，保留仅依赖深层推理的hard subset。
- **Label-count permutation baseline**：保持修正后预测的label直方图不变、随机打乱配对关系，检验是否仅靠分布匹配获得增益。
- **Softmax bottleneck**：输出层单次线性变换+softmax限制表达能力，即使隐藏状态包含复杂推理流形也无法充分映射到词表空间。
- **Unembedding misalignment**：预训练词表投影矩阵$W$与bias向量$b$的方向不利于特定任务标签，导致结构偏置压倒上下文信号。

## 可复现要素
- **数据集**：Controlled logic由固定脚本与seed生成（见Appendix B），ProofWriter/ANLI/FOLIO为公开基准；论文声明evaluation split与fit set disjoint。
- **代码/权重**：Qwen3.5、OLMo-2-1B、Llama-3.1-8B、Pythia均为open-weight checkpoint（见Table 7 revision manifest）；计算管线使用确定性grid search拟合offset。
- **关键超参**：offset仅2自由参数（三标签固定一参考）；fit样本数25–1000；probe训练2000行×3 seeds；最大长度1024；mean/sum score归一化均测试。
- **不确定度报告**：bootstrap intervals、paired statistical tests、30次deterministic subsample重复、多seed再生评估。
