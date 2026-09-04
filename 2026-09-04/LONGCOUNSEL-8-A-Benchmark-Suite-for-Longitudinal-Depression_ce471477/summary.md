---
title: "LONGCOUNSEL-8-A-Benchmark-Suite-for-Longitudinal-Depression"
source: https://arxiv.org/pdf/2609.03507v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 12:03:23"
field: "多会话咨询对话中的抑郁纵向追踪"
keywords: ["longitudinal depression tracking", "counseling dialogue benchmark", "PHQ-8", "synthetic dialogue generation", "behavior cue translation", "self-report label", "trajectory type evaluation"]
innovations: ["首次构建同时覆盖多会话咨询对话+会话级PHQ-8标签+实证症状构成+大规模生成的基准套件LONGCOUNSEL-8", "提出实证启发式状态构建与间接行为实现的合成数据管线，通过self-report/controlled state分离避免标签泄露", "揭示change MAE与direction accuracy排名反转、worsening轨迹系统性困难、历史利用非单调性等纵向评估关键现象"]
benchmarks: ["LONGCOUNSEL-8 (LC8-QWEN / LC8-LUNA / LC8-GPT-5.4-MINI)", "DAIC-WOZ", "REALCBT", "PSYCHE-D", "NHANES DPQ_L", "PsychEval"]
---

# 论文速读：LONGCOUNSEL-8: A Benchmark Suite for Longitudinal Depression Tracking from Multi-Session Counseling Dialogues

## 一句话总结
论文提出 LONGCOUNSEL-8，一个包含 7,749 条五会话咨询轨迹的基准套件，通过整合 PSYCHE-D 纵向轨迹与 NHANES 症状构成，以实证驱动的方式模拟真实客户端状态，填补了多会话咨询对话 + 会话级标准化抑郁标签的空白，并为抑郁纵向追踪任务提供了从静态单会话预测向可靠纵向跟踪迁移的评测基础。

## 研究问题与动机
- **现有数据资源不匹配任务需求**：DAIC-WOZ 仅提供单会话访谈，REALCBT/KokoroChat/PsyDial 等包含多会话但无会话级抑郁标签；纵向健康研究提供量表测量但无对齐的咨询对话，导致无法评估"多会话咨询对话中的抑郁纵向追踪"任务。
- **构建基准面临三大挑战**：①纵向一致性与多样性并存（同一客户端跨会话保持连贯身份同时支持多样轨迹）；②症状进展必须扎根于实证模式（而非随机生成）；③需在自然对话中表达受控抑郁状态，同时避免暴露目标标签。
- **现有评估指标存在误导性**：平均分数误差（MAE）低的模型不一定能正确识别改善/恶化方向，说明幅度估计与方向识别是两种不同能力，但缺乏统一纵向评测框架。
- **纵向信息利用机制不明确**：新增历史会话未必持续提升趋势预测准确率，现有方法对恶化轨迹的系统性偏差尚未被系统揭示。

## 核心贡献（创新点）
- **首次提出多会话+会话级 PHQ-8 标签对齐的咨询对话基准**：LONGCOUNSEL-8 同时覆盖 counseling dialogue、repeated sessions、standardized depression label、empirical grounding 四个维度，是现有资源中唯一集齐四要素的 benchmark（对比 Table 1）。
- **构建了一套"实证启发→间接行为实现"的数据生成管线**：将 PSYCHE-D 的五次访问轨迹 + NHANES 的 PHQ-8 症状构成向量映射为行为提示（behavior cues），再经 REALCBT 对话形式统计引导 LLM 生成自然对话，避免直接暴露量表语言。
- **设计了受控状态保真度验证协议**：通过后会话自我报告（self-report）恢复受控症状状态，量化 exact match、MAE、趋势一致性，验证合成数据的 label fidelity。
- **揭示了纵向评估中三个关键发现**：①change MAE 与 direction accuracy 呈排名反转，说明幅度误差与方向识别是分离能力；②所有评估方法在 worsening 轨迹上 MAE 最高，呈现跨轨迹类型的系统性偏差；③全历史输入可提升 MAE 却恶化方向准确率，历史利用存在 trade-off。

## 方法详解
- **Empirical grounding and state construction**：从 PsychEval 抽取 client profile（传记、主诉、信念），锚定 PSYCHE-D 的 PHQ-9 五次访问总轨迹；按总分数组 NHANES DPQ 完整 PHQ-9 响应向量，采样与目标总分匹配的 item 级症状组合，取前 8 项构成 latent PHQ-8 向量 y^(t)，跳过第 9 项（self-harm 高度敏感）。
- **From symptom scores to dialogue（行为提示翻译）**：将每个 symptom-score 单元格翻译为 12–30 词的 lived-experience behavior cue，禁止提及 PHQ 术语、响应选项或数值分数；通过 cue 预测试（给 role-playing agent 完成 PHQ-8 校验）保留最佳候选，每个单元格保留 10 条，生成时每条症状采样 5 条。
- **Counselor-client 对话生成**：counselor 获得公开 context + turn-form card（基于 REALCBT 统计的句数/长度/问句比例/话语开场），client 额外获得 private cues + qualitative total-burden band + disclosure beat；双向 prompt 均不包含任何数值分数或 PHQ 症状名，client 通过 partial disclosure、情感、具体实例间接表达状态。
- **Self-report labels**：会话结束后 client simulator 完成 PHQ-8（shuffle 题目与选项顺序 ×5 次后取平均），产生 simulated self-report 标签；该标签用作训练监督信号，而 latent PHQ-8 作为评估 ground truth，二者分开使用以避免标签泄露。
- **Trajectory 分类规则**：基于端点差值与内部极差划分四种类型：improving（end - start ≤ -5）、worsening（end - start ≥ 5）、fluctuating（端点差 <5 但内部极差 ≥5）、stable（其余）；阈值 5 对应 PHQ-9 最小临床重要差异（MCID）。

## 实验与结果
- **数据集规模**：三套独立生成数据（LC8-QWEN/LC8-LUNA/LC8-GPT-5.4-MINI），共 7,749 条五会话轨迹、38,745 个咨询会话；LC8-QWEN 含 2,583/369/738 训练/验证/测试轨迹，LC8-LUNA 同源同量，LC8-GPT-5.4-MINI 含 258/37/74 且为 profile-disjoint。
- **Label fidelity**：自报告与受控状态总均方误差在 0.294–0.541；变化方向（|Δy|≥5）恢复率在 99.96%–100%；完整轨迹类型一致性 92.1%–95.4%（LC8-LUNA 最高）；重度范围（PHQ-8≥15）MAE 为 1.33–3.38，偏向高估。
- **Baseline 方法**：AIDA（结构特征提取+线性回归）、LMIQ（LLM 问卷完成+随机森林）、Milintsevich et al.（症状预测）、Lau et al.（监督双编码器）、EnsemBERT（多损失集成编码器），均在 LC8-QWEN 冻结 split 上报 5-seed 均值。
- **核心结果（LC8-QWEN，Table 8）**：Lau et al. (Qwen3-Emb) 当前分数 MAE 最低 2.594，方向准确率最高 0.833；EnsemBERT 变化 MAE 最低 2.606 但方向准确率仅 0.479；metadata-only control 的 MAE 为 3.903，证明对话文本贡献 34% 的 MAE 下降。
- **关键发现**：
  1. **Magnitude vs. Direction 分离**：EnsemBERT（Change MAE 2.606 / Direction 0.479）与 Lau et al.（Change MAE 3.059 / Direction 0.833）排名反转，证明两者是可训练的不同能力。
  2. **Worsening 轨迹系统性困难**：所有方法在 worsening 轨迹上 MAE 最高（3.13–4.04），稳定轨迹最低（2.22–3.04），无方法在所有轨迹类型上公平。
  3. **历史利用的非单调性**：AIDA 用全历史使 Change MAE 下降 0.183 但 Direction 错误率从 0.258 升至 0.317；LMIQ 用全历史降低 Change MAE 0.265 但方向估计区间重叠，提示历史窗口设计需与模型架构联动。
  4. **Profile-disjoint LC8-LUNA 复现相同反转模式**，增强了结论鲁棒性。

## 相关工作脉络
- **DAIC-WOZ**：唯一提供会话级 PHQ-8 标注的对话资源，但每参与者仅单次半结构化访谈，不支持多会话纵向建模；本文填补"多会话+标准标签"空白。
- **REALCBT**：提供真实 CBT 会话与对话形式统计，被本文用作 turn-form card 的分布来源与语言相似度基线；本文在其对话形式之上叠加纵向抑郁标签与实证症状构成。
- **PsychEval / TheraPhase / MusPsy**：均提供多会话咨询对话，但缺乏会话级抑郁监督；本文通过 PHQ-8 标签体系弥补这一缺失，并增加 PSYCHE-D 纵向轨迹锚定。
- **PSYCHE-D**：10,036 参与者、每季度 PHQ-9 测量的纵向队列，提供五访问总分数轨迹作为本文 synthetic trajectories 的骨架。
- **NHANES DPQ_L**：大规模人群 PHQ-9 响应向量，用于按总分分组采样 item-level 症状构成，支撑 symptom composition 的实证 grounding。
- **已有 transcript-based 评估方法**（AIDA/LMIQ/Lau et al./EnsemBERT/Milintsevich）：本文为这五类主流方法提供首次统一的纵向评测协议，覆盖 current-state、consecutive change、direction、trajectory type、history-use 五个维度。

## 局限性与未来方向
- **实证接地范围受限**：profile 与 PHQ-8 轨迹来自不同资源（PsychEval + PSYCHE-D），profile-state 兼容性仅在聚合层面验证，缺少联合收集的真实纵向咨询数据。
- **生成器多样性有限**：仅使用三种 LLM（Qwen3.5、GPT-5.6 Luna、GPT-5.4-mini）生成数据，跨模型的语言合理性与偏差模式需更多评估。
- **重度会话自评校准偏差**：PHQ-8≥15 的会话 self-report 系统性偏向更高严重度，MAE 显著高于平均值（LC8-QWEN 达 3.382），提示极端状态需特殊处理。
- **会话状态表达稀疏性**：真实咨询中每次会话仅表达 PHQ-8 子集，本文生成数据虽覆盖全状态但临床实际对话更稀疏，需进一步连接独立临床状态验证协议。
- **未探索动态表征学习**：现有 baseline 主要使用静态编码或简单拼接历史，未充分挖掘时序建模、记忆检索、变化点检测等专用机制。

## 研究启发与可借鉴点
- **行为提示翻译（cue translation）设计可直接迁移**：将量表 item-score 映射为 12–30 词 lived-experience 描述、经 pre-test 筛选保留高保真候选的方案，适用于其他心理量表（GAD-7、BDI 等）的对话生成任务。
- **训练/评估标签分离策略**：self-report 作训练监督、latent PHQ-8 作评估 target，避免 label leakage 的同时保留真实自评噪声，对 synthetic benchmark 构建具有通用参考价值。
- **Magnitude-Direction 双指标联合报告范式**：本文证明单一 MAE 可能掩盖方向识别缺陷，建议未来纵向健康预测任务同步报告 change MAE 与 direction accuracy，避免指标误导。
- **轨迹类型分层评估**：将 improving/worsening/fluctuating/stable 作为独立评估层，能揭示模型的"公平性缺口"，建议作为纵向心理健康预测的标准报告协议。
- **历史窗口 sweep 实验设计**：固定 split、改变 context window 大小、观察多指标非单调变化，为对话历史利用机制研究提供可复用的实验范式。

## 关键术语表
- **PHQ-8**：Patient Health Questionnaire-8，包含 8 项抑郁症状条目、每项 0–3 分、总分 0–24 的标准化工具，本文用作会话级抑郁状态标签。
- **PSYCHE-D**：Prediction of Severity Change–Depression，一项为期一年、每季度 PHQ-9 测量的 10,036 人纵向队列，提供五访问总分数轨迹作为生成骨架。
- **Behavior Cue**：由 symptom-score 单元格翻译而成的 12–30 词 lived-experience 描述，作为 client agent 的私有输入，引导其自然表达目标严重度而不暴露量表语言。
- **Self-report Label**：会话结束后 client simulator 完成的 PHQ-8 问卷（shuffle×5 取平均），作为训练监督信号，与 latent controlled state 区分使用。
- **Trajectory Type**：基于端点差值与内部极差划分的四种病程模式：improving（≥5 分改善）、worsening（≥5 分恶化）、fluctuating（端点小但内部波动大）、stable。
- **Change MAE vs. Direction Accuracy**：前者衡量相邻会话分数变化幅度的平均误差，后者衡量 |Δy|≥5 的大变化方向（改善/恶化）识别准确率，本文证明二者可呈排名反转。
- **Turn-form Card**：基于 REALCBT 统计为每个说话者-阶段组合预定义的句子数、长度、问句比例与话语开场，约束对话表层形式但不决定症状内容。
- **Latent PHQ-8 State**：由 NHANES 症状构成采样的 8 维向量及其总分，作为不可见的受控评估 target，不进入对话生成 prompt 也不提供给模型。

## 可复现要素
- **数据集**：LONGCOUNSEL-8 三套数据（LC8-QWEN/LC8-LUNA/LC8-GPT-5.4-MINI）附于补充材料，将在 HuggingFace 以 CC BY-NC 4.0 发布；包含 transcript.jsonl、ground_truth.json、self_report.json 三份核心文件及 split manifests。
- **代码/权重**：数据集生成配置与 manifest 已公开；评估方法（AIDA/LMIQ/Lau et al./EnsemBERT/Milintsevich）基于原始仓库重建；Qwen3.5-35B-A3B 与 Qwen3-Embedding-8B 权重可通过 HuggingFace 获取（Apache 2.0）；OpenAI GPT 模型通过 API 访问。
- **关键超参**：Qwen 生成 T=0.6、p=0.95、k=20，单轮 token cap 512，reasoning budget 4,096；Lau et al. lr=3×10⁻⁴、weight decay=0.01、max 200 epochs、patience 20；EnsemBERT lr=10⁻³、weight decay=10⁻⁴、batch=64、max 50 epochs、patience 8；AIDA 特征提取 temperature=0、最多 3 次解析尝试；LMIQ 树数 100/200/300、深度 10/20/30 通过 validation MSE 选择。
- **随机种子**： cohort 分配 seed=42，Qwen 额外 seed=0；bootstrap 95% 区间 seed=20260825/20260828。
- **训练/评估协议**：所有 learned methods 在 self-report 标签上拟合、validation 上选模、在 latent PHQ-8 上评估；5-seed 平均；profile-overlap 与 profile-disjoint 双协议并行报告。
