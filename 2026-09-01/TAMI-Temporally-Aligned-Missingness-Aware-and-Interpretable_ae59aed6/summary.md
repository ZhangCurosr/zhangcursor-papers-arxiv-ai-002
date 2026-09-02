---
title: "TAMI-Temporally-Aligned-Missingness-Aware-and-Interpretable"
source: https://arxiv.org/pdf/2608.30857v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 12:06:24"
field: "多模态心理健康计算"
keywords: ["multimodal fusion", "temporal alignment", "missingness-aware", "interpretable AI", "mental health screening", "mild cognitive impairment", "remote interviews"]
innovations: ["自适应时间分箱实现异构采样率多模态特征的细粒度时间对齐", "模态-时间步缺失掩码显式区分真实缺失与合法零值", "问题上下文条件化融合 + 多级 IG 归因框架联合提供模态/问题/时间三维度解释"]
benchmarks: ["GDS≥10 抑郁风险分类", "GAD-7≥5 焦虑风险分类"]
---

# 论文速读：TAMI-Temporally-Aligned-Missingness-Aware-and-Interpretable

## 一句话总结
本文提出 TAMI 多模态融合框架，通过自适应时间分箱对齐异构采样率特征、显式编码模态级缺失掩码、并以问题上下文条件化融合，用于从远程临床访谈中预测轻度认知障碍（MCI）老年人的抑郁与焦虑。实验表明细粒度时间对齐带来最大性能增益，且仅用开放式问题回答（中位时长 5.1 分钟）即可达到与完整访谈（19 分钟）相当的抑郁筛查效果。

## 研究问题与动机
- **时间错位**：现有方法采用序列索引配对（sequence-index pairing）关联不同模态特征，导致跨模态时序覆盖不对齐——一个数秒级窗口特征（如声学）与单个帧级观测（如面部）被错误关联，产生高达 22 秒的时间偏移，远超人类感知整合的 200 ms 窗口。
- **缺失值处理不当**：远程录制存在不均匀模态丢失（遮挡、模糊、帧丢失），但 prior work 以零填充缺失值，使真实缺失与合法的近零测量无法区分。
- **缺乏细粒度可解释性**：现有方法仅关注部分归因层级（模态级或问题级），未联合 attributing 到模态、问题和访谈时刻，难以支持临床精细解读。
- **问题上下文未充分融入**：既有 MCI 人群精神健康预测忽略问题上下文；部分通用人群方法仅对部分模态施加问题条件，未在细粒度时间分辨率上联合条件化跨模态行为。

## 核心贡献（创新点）
1. **自适应时间分箱对齐**：将每个问答段划分为 1 秒级时间桶（bin），按时间戳分配帧级特征、按重叠区间分配窗口级特征，构建共享时间轴；与 prior 序列索引配对方法的本质区别在于对齐依据是原始录音时间而非序列位置。
2. **模态-时间步缺失掩码**：为每个模态特征的每个时间桶引入二元可用性掩码，与零填充值同时输入；与 prior 零填充法的本质区别在于显式区分"真实缺失"与"合法近零信号"。
3. **问题条件化融合**：将问题嵌入逐时间桶加到融合多模态 token 上，覆盖全部 8 种模态特征；与 prior 仅在文本模态或仅对声学/文本分别条件的方法的本质区别在于细粒度时间分辨率上的跨模态联合条件化。
4. **多级 IG 归因分析**：基于 Integrated Gradients 聚合出模态重要性、问题重要性、问题内模态重要性、时间重要性四个层级；与 prior 仅考察子集层级的方法的本质区别在于首次在跨被试全访谈范围联合提供三维度解释。
5. **开放式问题高效筛查发现**：证明仅用 4 道开放式问题（中位 5.1 分钟）的抑郁 AUROC 与全访谈（19 分钟）无显著差异（p>0.05）；与 prior 工作的本质区别在于为缩短访谈协议提供了数据驱动证据。

## 方法详解
**特征提取**：8 种模态特征，包括帧级（AUs&LMs D=155、head pose D=3、eyegaze D=2，1 fps）、窗口级（eGeMAPS D=88、ComParE D=6373→PCA 至 512、Wav2Vec2 D=1024，2s 窗口/1s hop、rPPG 心率 D=1，6s 窗口/1s hop）、及词级语言特征（WhisperX 转录 + RoBERTa，D=768）。

**自适应时间分箱**：每段回答时长 $L_{ij}$ 划分为 $B_{ij}=\text{clamp}(\text{round}(L_{ij}/w)+1, B_{\min}, B_{\max})$ 个时间桶（$w=1\text{s}$，$B_{\min}=1$，$B_{\max}=64$）；超长回答按比例扩大 bin 宽度而非截断。帧级特征按时间戳归属 bin，窗口级特征归属所有重叠 bin 并平均，词级转录按时间戳拼接后用 RoBERTa 编码为 bin 级语言表征。

**缺失掩码**：若 bin 内无有效观测则 $M_{ij,k}^{(m)}=0$ 且特征向量零填充；否则 $M_{ij,k}^{(m)}=1$，与零填充值共存，使模型可区分缺失与合法低信号。

**投影骨干与融合**：线性骨干（直接投影）vs 非线性骨干（per-modality Transformer 自注意力，掩码作 attention mask）；投影后拼接掩码向量，再经 $W_f$ 投影至融合 token $\mathbf{z}_{i,s}$。

**问题条件化**：$\tilde{\mathbf{z}}_{i,s}=\mathbf{z}_{i,s}+W_q\mathbf{q}_{ij}$，其中 $\mathbf{q}_{ij}$ 为第 $j$ 题的 Sentence-BERT 嵌入，映射到维度 $d$ 后逐桶相加。

**预测**：前置 [CLS] 的 Transformer Encoder（隐藏维 $d=128$，线性骨干无 per-modality 编码器、非线性含 1 层 per-modality Transformer，共同 Encoder 2 层）→ 线性头 → 二分类 logit。

**多级归因**：对标准化后零向量基线计算 IG，逐 bin 平均特征维度得 $\phi_{i,s}^{(m)}$，再聚合至模态重要性 $I_m$、问题重要性 $I_q$、时间重要性 $R_u$（20 个相对桶）。

## 实验与结果
- **数据集**：Emory University CEP 项目，49 名 MCI 老年人（均值年龄 73.4±8.1 岁，47% 女性），Zoom 远程半结构化访谈；抑郁阈值 GDS≥10（21 阳性/28 阴性），焦虑阈值 GAD-7≥5（17 阳性/31 阴性，1 人缺失）。
- **评估协议**：被试无关 5-fold 交叉验证 × 3 次重复（15 runs），一侧配对 t 检验（p<0.05）。
- **最强结果**：抑郁 Non-linear+T 达 **AUROC 0.68±0.04**（较 baseline +0.10，p≈0.05）；焦虑 Linear+T 达 **AUROC 0.69±0.09**（较 baseline +0.11，p=0.05）。
- **关键 ablation**：时间对齐（T）带来最大增益（Δ≥0.1 AUROC）；缺失掩码对焦虑 Linear+T 提升显著（0.69 vs 0.64）；问题条件化（Q）在已对齐模型上未带来统计显著提升（p>0.05）。
- **开放式问题效率**：仅用 4 道开放式问题（中位 5.1 min）抑郁 AUROC=0.67，与全访谈 0.68 无显著差异（p>0.05）；焦虑 open-only AUROC=0.59，显著低于全访谈（p≈0.02）。
- **模态重要性**：抑郁以 eyegaze 为主导（66%），ComParE 第二（11.2%）；焦虑 eyegaze（44%）与 head pose（37.2%）均衡分布。GDS/GAD-7 量表问题均未进入 Top 问题组。

## 相关工作脉络
- **序列索引配对基线** [26][27]：按序列位置关联特征，忽视时间覆盖；本文以时间分箱替代，确保帧级/窗口级/词级特征共享同一时间轴。
- **问题条件化 prior** [23][25][34]：Zhang et al. [25] 对声学+文本做时间池化后与问题 gated addition；Niu et al. [34] 对声学与文本分别做 cross-attention；Guo et al. [23] 仅条件化文本；本文在所有 8 种模态融合 token 的每个时间桶上统一加问题嵌入。
- **缺失处理 prior** [27][31]：零填充缺失值；Gimeno-Gomez et al. [28] 提出 per-frame presence mask；本文在此基础上将 mask 置于答案对齐的时间桶上，与问题上下文共享时间轴。
- **MCI 人群精神健康预测** [21][32][33]：prior 多忽略问题上下文且未做跨模态时间对齐；本文首次在该人群中联合三者。
- **可解释性 prior** [20][21][28][31]：modality-level（性能比较或 IG）或 question-level（attention）单独使用；本文首次联合提供 modality×question×time 三维度归因。
- **跨模态对齐通用方法**：emotion recognition [81]、human activity recognition [82] 均涉及异构采样率特征对齐，本文的时间分箱策略可直接迁移。

## 局限性与未来方向
- 样本量小（N=49），统计功效有限，部分 ablation 差异未达显著；需在更大多样队列中验证。
- 面部分割/声学特征提取器均在通用人群数据上预训练，对 MCI 老年人远程录制的可靠性可能受限。
- 问答分段目前依赖人工标注起始/结束时间戳，部署时需自动对齐（如 WhisperX 转录与已知问题匹配），自动化精度待验证。
- 未来可扩展至伴有抑郁/焦虑症状的神经精神疾病人群，并探索更短访谈协议的临床落地。

## 研究启发与可借鉴点
1. **自适应时间分箱策略**：将异构采样率特征统一到共享时间轴的思路，可迁移至任何多模态时间序列任务（如情感识别、活动识别、医疗时序融合），避免序列索引配对引入的虚假跨模态关联。
2. **缺失掩码 + 零填充联合输入**：将可用性掩码拼接到特征向量后送入融合层，以显式区分"缺失"与"合法零值"；适用于任何存在不均匀丢帧/丢信号的远程数据采集场景。
3. **多级 IG 归因聚合框架**：从特征级→bin 级→模态/问题/时间层级的聚合公式（式 4–11）具有模块化和可复现性，可直接套用至其他需要多粒度解释的医疗 AI 模型。
4. **开放式问题效率验证范式**：通过比较"全访谈 vs 子集问题"的性能差异（结合 IG 归因验证一致性），为缩短临床筛查协议提供数据驱动依据；可作为后续工作效率优化实验的标准模板。
5. **线性 vs 非线性投影骨干的对比设计**：解耦"per-modality 时间建模"与"跨模态融合时间建模"两个设计选项，有助于厘清各自贡献；该消融范式值得在多模态融合研究中推广。

## 关键术语表
**TAMI**：Temporally-Aligned, Missingness-Aware, and Interpretable 多模态融合框架的缩写，用于 MCI 老年人远程访谈中的抑郁/焦虑筛查。
**MCI（Mild Cognitive Impairment）**：轻度认知障碍，阿尔茨海默病及相关痴呆的前驱临床状态，特征为认知下降超出正常衰老但日常生活功能保留。
**AUROC**：Receiver Operating Characteristic 曲线下面积，衡量二分类模型区分能力，值越高表示性能越好。
**Integrated Gradients（IG）**：基于公理的深度网络归因方法，通过沿零向量基线到输入点的积分路径计算每个特征对预测的贡献。
**rPPG（remote Photoplethysmography）**：从视频信号中远程估计心率等生理指标的计算机视觉技术。
**Sequence-index pairing**：将不同模态特征按序列位置一一对应而非按时间戳对齐的旧有做法，会导致跨模态时间错位。
**Temporal binning**：将连续时间轴划分为固定宽度时间桶（bin），按时间覆盖关系将异构特征分配到桶中的对齐策略。
**Modality-timestep missingness mask**：二元掩码，标记每个模态在每个时间桶是否有有效观测，使零填充缺失与合法零值可区分。

## 可复现要素
- **数据集**：Emory University Charlie and Harriet Shaffer Cognitive Empowerment Program（CEP），49 名 MCI 老年人远程访谈；论文未声明公开数据集链接。
- **代码/权重**：论文未提及开源。
- **关键超参**：隐藏维 $d=128$，$B_{\min}=1$，$B_{\max}=64$，bin 宽 $w=1\text{s}$（超长时按比例扩大），Transformer 层数（非线性骨干 per-modality 1 层，共同 Encoder 2 层），训练 10 epoch，Adam lr=0.001，batch size=4，class-balanced sampler，5-fold CV × 3 repetitions。
- **特征提取器**：Py-Feat（AUs&LMs）、L2CS-Net（eyegaze）、openSMILE（eGeMAPS/ComParE）、Wav2Vec2（Hugging Face Transformers）、WhisperX+RoBERTa-base（语言）、pyVHR（rPPG 心率）。
