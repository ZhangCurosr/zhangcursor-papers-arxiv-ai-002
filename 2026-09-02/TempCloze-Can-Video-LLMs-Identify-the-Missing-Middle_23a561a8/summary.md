---
title: "TempCloze-Can-Video-LLMs-Identify-the-Missing-Middle"
source: https://arxiv.org/pdf/2609.01515v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 22:19:04"
---

# 论文速读：TempCloze-Can-Video-LLMs-Identify-the-Missing-Middle

## 一句话总结
本文提出 **TempCloze**，一个基于视频完形填空的视觉时序推理评测基准，通过给定视频首尾片段要求模型从同源候选中选出真正的缺失中段，从而切断语言捷径；大规模评测揭示当前 Video-LLMs 在时间对齐（Alignment）维度上存在显著且一致的性能瓶颈。

## 研究问题与动机
- 现有 Video-LLM 时序基准（如 TempCompass、TVBench）多依赖文本选项或多轮问答，模型可利用选项措辞、答案共现模式或语言先验走“捷径”，得分难以反映真实视觉时序理解。
- 已有研究指出纯语言模型在无多模态输入时即可凭先验超越随机基线 25% 以上，亟需一种更直接、以视觉片段比对为核心的评测范式。
- 当前评估偏重事件排序或细粒度定位，缺乏对“首尾约束下精确时间边界匹配（Missing Middle）”能力的系统检验。
- 传统基准中候选片段常来自不同视频或领域，外观差异易被模型利用；需要控制变量，迫使模型聚焦时序证据。

## 核心贡献（创新点）
- **提出 TempCloze 视频完形基准**：将时序推理任务形式化为首尾片段补全中段，从评估格式上直接阻断语言捷径对得分的污染。
- **三维度同源干扰项设计**：将候选片段的错误来源正交分解为 Semantic（事件内容）、Alignment（时间边界）与 Progression（内部演进）三个维度，实现细粒度能力诊断。
- **系统性大规模评测**：评测 10 个商业与 21 个开源 Video-LLM，首次明确指出 **Alignment 是当前视觉时序推理的首要瓶颈**，而非事件语义或局部动作理解。
- **多维行为诊断框架**：提供错误模式分析、跨维度混合干扰分析、候选顺序置换、上下文方向消融、可见跨度/帧密度缩放及测试时缩放（Pass@k）实验，揭示模型决策的稳定性与敏感性特征。

## 方法详解
- **任务形式**：视频 $V$ 按时间划分为 $B=C_{0,s}$、$M=C_{s,e}$、$E=C_{e,T}$。模型接收 $(B, E)$，从候选集 $\mathcal{V}_d=\{M, D_d^1, D_d^2, D_d^3\}$（$d \in \{S, A, P\}$）中选出唯一正确的缺失中段。
- **干扰项构造规则**：
  - **Semantic (S)**：选取时长相同且与目标区间无重叠的其他片段，考察模型能否从首尾推断核心事件内容。
  - **Alignment (A)**：保持内容大致合理，但扰动时间边界。包含 Advanced（前移 $\ell/2$）、Deferred（后移 $\ell/2$）与 Expanded（向两侧各扩展 $\ell/2$），考察对时间边界的敏感度。
  - **Progression (P)**：时间区间正确但破坏内部顺序。包含 Reversed（倒放）、Reordered（子事件序列重排）与 Repeated（短片段重复），考察事件方向与连续性感知。
- **数据构建管线**：从 LVD-2M、EgoLife、MiraData、FAVOR-Bench、CaReBench、Video-TT、Daily-Omni 七类源中筛选；经时长过滤（12–90s）、GPTo3 推理模型评估完形适配性、质量过滤（码率>200 kbps、清晰度>30）及 Farnebäck 光流运动验证（平均幅值>1.0，最多重采样 3 次），最终保留 **1,521** 条视频。
- **评估配置**：默认每片段均匀采样 16 帧（单维度共 96 帧）；开放模型通过 vLLM 部署于 A6000 GPU；构造 TempCloze-Mixed（300 条跨维度混合干扰）与 TempCloze-Hard（全模型错误最多的 150 条）用于深度分析。

## 实验与结果
- **数据集与基线**：TempCloze 1,521 视频；评测 10 个 Proprietary（Seed1.8, Qwen3.5-Plus, Gemini2.5-Pro/Flash, GPT5.4, Claude4.6-Sonnet/Opus, Gemini3-Flash, Seed1.6, Grok4.1）与 21 个 Open-source 模型。
- **Dimension Accuracy**：
  - **Semantic**：商业平均 70.73%（Seed1.8 最高 96.25%），开源平均 34.00%（Qwen3.5-397B 达 75.94%）。
  - **Alignment**：商业平均 48.13%（Seed1.8 最高 76.92%），开源平均 26.54%，远低于人类基线 98.00%，为**首要瓶颈**。
  - **Progression**：商业平均 67.72%，开源平均 36.97%。
- **Cumulative Accuracy**：商业平均 3/3 正确率 33.24%（Seed1.8 最佳 70.81%），开源平均 7.15%；人类基线 92.00%。开源模型错误高度集中于 0/3 或 1/3，商业模型多表现为窄幅 2/3 遗漏。
- **错误模式**：Alignment 错误以 **Expanded** 为主导（如 Seed1.8 在 75% 的 A 错误中选择 Expanded）；Progression 错误以 **Reversed** 为主导（GPT5.4 占 67%）。两者共同特征是视觉内容合理但时序结构不匹配。
- **行为敏感性**：候选顺序置换对均值影响小（方差<4%），但 Flip Rate 与 Clip Flip Rate 高达 26%–60%，表明精度稳定不等于决策稳定；模型更依赖 Beginning 上下文而非 Ending；增加可见跨度或帧密度通常损害 Alignment 表现；测试时缩放（Pass@k）可提升绝对得分，但不改变 S/A/P 的相对排序。

## 相关工作脉络
- **时序推理基准**（Temporal-Bench、TVBench、TempCompass、LongVideoBench、MVBench、VideoMME）：多延续 VideoQA 范式，依赖文本选项或自由生成，易受语言先验污染；TempCloze 转向纯视觉片段对比，从评估形式上切断捷径。
- **Cloze 类视频理解**（MovieFIB、FIBER、VideoBERT、VCP、MaskFeat、VideoMAE）：将完形填空用作预训练或表示学习的自监督目标；本文将其转化为下游 Video-LLM 的测评任务，聚焦推理诊断而非表征学习。
- **多模态语言捷径研究**（Balepur et al., 2024；Xiao et al., 2024）：证实 PLM 可凭文本共现与选项模式获得虚假高分；本文通过同源候选设计从数据侧消除外观与文本共现干扰。
- **长镜头/第一人称视频源**（LVD-2M、EgoLife、MiraData、FAVOR-Bench）：提供连续时空上下文；本文基于此类数据构建基准，验证长时程与视域一致性对时序边界诊断的价值。
- **测试时思考机制**（Think vs Instruct 对比）：现有工作多关注推理 token 对生成质量的提升；本文发现思考对 Alignment 增益显著（Seed1.8 +14.99%），但对开源小模型可能以牺牲 S/P 为代价，揭示思考机制在时序任务中的非单调效应。

## 局限性与未来方向
- 任务局限于四选一共形填空，无法覆盖开放式生成、对话叙事理解、音频 grounding 或未约束的未来预测。
- 同源候选设计虽抑制外观捷径，但也大幅提高任务难度（要求精确时间边界匹配），可能低估模型在更宽松场景下的时序泛化。
- 数据来源偏向 long-take、egocentric 与精细动作视频，尚未覆盖多场景切换、非连续剪辑或多样拍摄风格。
- 评测结果为当前模型快照，API
