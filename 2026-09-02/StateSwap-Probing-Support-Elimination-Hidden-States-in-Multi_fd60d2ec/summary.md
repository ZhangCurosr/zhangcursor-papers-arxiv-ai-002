---
title: "StateSwap-Probing-Support-Elimination-Hidden-States-in-Multi"
source: https://arxiv.org/pdf/2609.01081v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 15:48:44"
field: "大语言模型可解释性与推理干预"
keywords: ["activation substitution", "mechanistic interpretability", "prompt framing", "contrastive activation addition", "multiple-choice reasoning", "residual stream steering"]
innovations: ["引入未训练[STATE]token作为残差流接口，实现实例级SUP/ELIM交叉激活交换并验证其因果行为效应", "提出基于Cohen's d与滑动特征窗的无监督层定位方案，在中间层发现框架敏感表征的可分离峰", "构建无接口的双框架mean-difference steering方向，在多层多系数下相比匹配CAA显示更高的精度保留与更窄的层间波动"]
benchmarks: ["MMLU-17", "MedQA-CH"]
---

# 论文速读：StateSwap: Probing Support–Elimination Hidden States in Multiple-Choice Questions

## 一句话总结
本文在控制推理条件下，通过引入未训练的 [STATE] 特殊 token 作为残差流接口，发现支持导向（SUP）与排除导向（ELIM）提示在同一选择题上诱导出的中间层隐藏状态具有可分离性；交换这两种提示对应的 [STATE] 激活能系统性地改变模型预测并提升跨框架一致性，为"逻辑等价的决策目标可编码出可干预的差异化隐状态"提供了因果性实验证据。

## 研究问题与动机
1. **行为不一致性的表征根源问题**：SUP（"哪项正确？"）与 ELIM（"哪项错误？"）在语义与答案键上完全等价，但 LLM 在实际推理中对此两种框架的表现时好时坏，甚至出现矛盾结论；二者是否诱导了可区分的内部表征，是尚未被直接验证的核心假设。
2. **现有方法无法隔离表征差异**：过往 prompting / PoE 类工作（如 Ma & Du, 2023; Zhu et al., 2025; Fu et al., 2025）仅在输出层面比较准确率，无法判定性能波动源于解码随机性还是表征层面的结构性差异；即便 activation steering 类方法（CAA、Lee et al., 2025）能定向操控行为，也缺乏针对"提示框架"这一变量的受控实例级干预证据。
3. **干预接口与定位机制缺乏**：已有 mechanistic interpretability 研究（如 Heimersheim & Nanda, 2024; Meng et al., 2022）证明 activation patching 的有效性，但在 MCQ 场景下，如何识别最敏感的层窗口、如何构建不受词汇/句法混淆的对照对、如何区分"结构化状态交换"与"无序扰动"，均无系统方案。

## 核心贡献（创新点）
1. **发现并验证了 [STATE] 表征中的框架可分离性**：在相同 MCQ 下，SUP 与 ELIM 在中间层（Qwen-2.5-7B 的 layers 11–20、GLM-4-9B 的 layers 21–40）诱导出线性可分的 [STATE] 隐状态，且该模式经随机标签控制与 bootstrap 稳健性检验后依然显著；与 prior work 相比，本文首次在同一实例对上直接观测到"等价决策目标 → 差异化隐表示"的几何结构，而非仅通过分类器间接推断。
2. **提出 StateSwap 的实例级激活交换协议**：通过缓存一个框架的 [STATE] 激活并替换另一框架对应层窗口的状态，可在固定文本输入下系统性地翻转预测；与 CAA 等数据集级方向对比不同，StateSwap 作用于具体题对的隐藏状态，提供从表征到行为的因果链路证据。
3. **揭示 mean-difference steering 比匹配 CAA 方向具有更稳定的层wise响应**：去除 [STATE] 接口后，在人口水平上对比双框架均值差方向与原 CAA 方向的 steering 效果，在中等系数（a = −1, −2, 2）下 Dual-Framing 方向精度显著更高（GLM: 59.03 vs. 37.16 / 69.67 vs. 48.85 / 68.15 vs. 60.35；Qwen: 71.21 vs. 52.26 / 81.61 vs. 75.97 / 74.77 vs. 62.90），且层间波动幅度更窄（a=−1 时 GLM 从 50.54 pts 降至 19.90 pts），表明双框架对比本身即可作为更鲁棒的 steering 先验。

## 方法详解
1. **输入构造与 [STATE] 接口**：对每道题目 q，构造一对仅指令不同的 prompt：x_SUP(q) = Concat(SUP, q, [STATE]) 与 x_ELIM(q) = Concat(ELIM, q, [STATE])，[STATE] 位于序列末端位置 t_S = T；为保证相同绝对索引，两条指令通过 PAD 填充至等长 token 数。模型为因果 Transformer f_θ（L 层），记 h_t^(l) ∈ R^d 为 block l 后位置 t 的残差流表示，则 [STATE] 在层 l 的隐状态定义为 s^(l) = h_(t_S)^(l)。[STATE] 为新增注册但未预训练的 special token，embedding 保持随机初始化不更新，作为最小化 read–write 接口。
2. **状态交换（State Substitution）**：选定连续层窗口 W ⊆ {1,…,L}，对框架 I 下的 prompt x_I(q)，在其每个 k ∈ W 处将缓存的互补框架激活 s^k_cache(Ī, q) 覆盖原 s^k(I, q)，后续计算照常进行至自回归生成。每次干预仅覆盖 post-block 残差向量，下游通过残差传播受影响，其他位置向量不变。特征窗切片仅用于诊断定位，实际交换为完整 d 维状态。
3. **层窗口定位（Diagnostic）**：采用滑动特征窗口 s_[j:j+w)^(l) ∈ R^w，在成对 ELIM–SUP 差上计算 Cohen's d。两种标量化方式：① Random-direction：采样随机单位向量 u，投影 φ^(rand)_{j,w}(z) = u_[j:j+w)^⊤ z；② Mean-difference：构造数据级均值差方向 v_l(j,w) = (1/N) Σ_i (s^((l))_ELIM,i,[j:j+w) − s^((l))_SUP,i,[j:j+w))，归一化后投影。每层按 Top-K 均值聚合各窗口 |d_l(j,w)| 得到层强度 d̃_l，取 ≥ 0.8 × max(d̃_l') 的层并合并为连续窗口 W。该阈值固定、不依赖下游任务性能。校准集仅 50 题，测试阶段使用官方 test split。
4. **评估与集成协议**：采用确定性贪心解码（do_sample=False, max_new_tokens=1024）消除采样噪声。每框架由两条不同 prompt 实例化组成，组内严格多数投票。指标包括 ACC（正确选项集合判定）、Jaccard（跨框架决策交集/并集）、LAC（长度感知余弦相似度）、BERTScore-F1（token 级语义一致）。扩展分析包含 ensemble ablation、错误消除计数（E_i^p = |W_i^p \ {y_i}|）的右移/左移比例、以及去掉 [STATE] 后的无接口 CAA-style steering 对比。

## 实验与结果
- **数据集**：MMLU-17（17 个侧重推理的子领域，共 4,607 题，英文）与 MedQA-CH（中文医学 QA，1,015 题），合计 5,622 题；50 题训练集仅用于窗口定位。
- **模型**：Qwen-2.5-7B-Instruct、GLM-4-9B，zero-shot，chat template 与解码超参固定，torch_dtype=float16，seed=123。
- **主要数值结果（Table 1）**：
  - Qwen-2.5-7B / MMLU-17：Base ACC_SUP=65.52 / ACC_ELIM=66.53 / Jaccard=75.46；Sub ACC_SUP=66.31 / ACC_ELIM=74.36 / Jaccard=76.63 → ELIM ACC +7.83 pts、Jaccard +1.17 pts。
  - Qwen-2.5-7B / MedQA-CH：Base 77.12 / 77.45 / 75.46；Sub 80.26 / 81.58 → ELIM ACC +4.13 pts、Jaccard +6.12 pts。
  - GLM-4-9B / MMLU-17：Base 66.47 / 69.21 / 72.96；Sub 68.74 / 72.36 / 76.63 → ELIM ACC +3.15 pts、Jaccard +3.67 pts。
  - GLM-4-9B / MedQA-CH：Base 73.54 / 75.91 / 78.67；Sub 76.82 / 77.92 / 82.48 → ELIM ACC +2.01 pts、Jaccard +3.81 pts。
  - **最强结果**：Qwen-2.5-7B 在 MedQA-CH 下 SUB_ELIM 达到 81.58%（相对 Base 77.45% 提升 4.13 pts）；双 StateSwap 集成（4 路）在 MMLU-17 上达到 82.25%（Qwen）/ 80.18%（GLM），优于直接 SUP+ELIM 朴素集成。
- **可分离性（EQ1）**：PCA 可视化显示早期层混叠、中间层清晰分离、晚期层再次减弱；随机标签控制在 10 次置换下始终维持低分，排除窗口化/投影/不平衡伪影。
- **界面特异性（EQ3）**：① 在最终内容 token 上交换仅使响应相似度维持 ~1.0（Table E.3 LAC median: [STATE]=0.8766 vs. 最终token=0.9780），[STATE] 接口远更有效；② 零化/随机替换显著降低 LAC/BERTScore-F1，而跨框架交换保持高相似度；③ 将 [STATE] 移至非末位导致 >95% 生成失效（不连贯或无法解析），证明位置敏感性。
- **双框架 Steering（Section 5.4 / Table C.5）**：去掉 [STATE] 后，双框架均值差方向在 GLM 与 Qwen 的多个层与 a ∈ {−2,−1,1,2} 下均保持显著更高的平均精度，层间极差明显收窄。

## 相关工作脉络
1. **Process-of-Elimination / Prompting 系列**：Ma & Du (2023)、Zhu et al. (2025)、Fu et al. (2025)、Wang et al. (2026)、Zhao & Zhang (2025) 证明排除式提示可提升 MCQ 性能；Balepur et al. (2024) 则报告其可能损害准确率。本文解释这种"矛盾"根源在于不同框架诱导了不同的隐状态，提供表征层面的统一视角。
2. **Activation Patching / Mechanistic Interpretability**：Heimersheim & Nanda (2024)、Meng et al. (2022)、Olsson et al. (2022) 证明局部组件与行为因果相关。本文继承因果追踪思想，但将干预从注意力头/MLP 单元扩展至"整段 [STATE] 残差流 + 连续层窗口"的实例级交换范式。
3. **Contrastive Activation Addition（CAA）与 Steering**：Rimsky et al. (2024)、Li et al. (2023)、Turner et al. (2024)、Zou et al. (2023)、Lee et al. (2025) 展示数据集级方向可系统性改变输出。本文的贡献在于：① 在实例级别做配对交换而非数据集统计方向；② 将对比构造从"相同指令+相反选项"（CAA）切换到"相同选项+相反框架"，揭示后者在层wise稳定性上的优势。
4. **参数高效提示与接口方法**：Lester et al. (2021) (Prompt Tuning)、Li & Liang (2021) (Prefix Tuning)、Liu et al. (2022) (P-tuning) 学习连续 embedding 来操控行为；本文不使用任何训练，仅将未训练 [STATE] 作为零成本接口，避免了训练引入的分布偏移。
5. **Chain-of-Thought / 隐式状态表示**：Chen et al. (2025) 用 sparse autoencoder 刻画 CoT 激活；Zhang et al. (2025)、Lu et al. (2025) 从有限状态机角度研究隐状态追踪。本文关注同一任务内部因措辞差异而产生的细粒度状态分化，而非任务本身的状态管理机制。

## 局限性与未来方向
1. **实验设置受限**：仅四选项 MCQ 与确定性贪心解码，未验证 sampling-based decoding、长多步生成或开放文本下的可迁移性。
2. **接口依赖**：方法绑定 [STATE] token 的固定末位插入，尽管随机初始化鲁棒性实验显示影响极小，但仍不能声称效果完全独立于接口设计；黑盒模型下尚未经检验。
3. **框架对比的混淆风险**：作者承认 SUP/ELIM 作为观测标签仍可能混杂领域、不确定性、响应策略等因素；未做分层解耦，Hierarchical 分析可作为未来扩展。
4. **外部泛化未知**：双框架 steering 虽在无 [STATE] 设置下有效，但其对非 MCQ 任务（需要共享评估目标的 paired prompt + 开放式匹配度量）的扩展路径尚未探索。

## 研究启发与可借鉴点
1. **未训练特殊 token 作为零成本干预接口**：无需任何微调即可在残差流中建立 read–write 接入点，该方法可移植到 CoT 分解、多轮对话状态追踪等场景，作为探测表征结构的通用探针。
2. **Mean-difference vs. CAA 的对比实验设计**：以"改变什么变量、保持什么固定"为准则构造对照方向（双框架：固定答案、变指令；CAA：固定指令、变答案），可推广到任意成对提示差异（如语言、推理链长度、风格）的 steering 先验挖掘。
3. **特征窗切片 + Cohen's d 的层定位流程**：滑动窗 + Top-K 聚合 + 0.8 阈值的层选方案不依赖下游性能调优，具备跨模型、跨任务的通用定位价值，可直接复用到其他 mechanistic probing 管线。
4. **不对称转移的定量度量（Right/Left Shift 计数）**：通过错误消除计数的方向分布判断干预行为的语义偏向（本文发现 Sub 更常增强错误排除），为激活干预的效果分析提供细粒度诊断指标，可迁移至事实编辑、偏见消减等研究。

## 关键术语表
**[STATE] 接口**：作者新注册但未预训练的 special token，追加到 prompt 末尾，作为残差流中读写的零训练开销表示接口。
**Support-oriented (SUP) vs. Elimination-oriented (ELIM)**：两种逻辑等价的 MCQ 提示框架，前者要求模型识别正确选项，后者要求识别错误选项。
**Activation Substitution / StateSwap**：将一道题在某一框架下缓存的 [STATE] 残差激活替换到另一框架对应层窗口的干预操作。
**Cohen's d 层定位**：在特征窗切片上计算成对 ELIM–SUP 差值的标准化均值，用于定量衡量各层表征的可分离强度。
**Mean-difference steering**：跨数据集实例平均得到的 SUP–ELIM 差方向，用于无 [STATE] 时的人口级激活操控。
**Jaccard 跨框架重叠**：两框架各自答对题目索引集的交集比并集，衡量决策一致性而非单框架准确率。
**LAC（Length-Aware Cosine Similarity）**：句子级语义余弦相似度乘以长度比值惩罚项，用于量化干预后输出的漂移程度。
**Error-elimination Right/Left Shift**：子题级错误排除计数在干预前后的变化方向，衡量干预是否系统性增强了排除确定性。

## 可复现要素
- **数据集**：MMLU（官方 test split）、MedQA-CH（中文子集，公开）；50 题校准集来自训练 split 仅用于窗口定位，不进入测试。
- **代码**：已开源，地址 https://github.com/Cha0Ga0/SWAPSTATE。
- **权重**：使用开源模型 Qwen-2.5-7B-Instruct 与 GLM-4-9B（zero-shot，无参数更新）。
- **关键超参**：greedy decoding（do_sample=False）、max_new_tokens=1024、float16、seed=123、右 padding ≤64 tokens；窗口定位阈值固定为 0.8×max(d̃_l)；steering 系数 a ∈ {−2,−1,0,1,2}；特征窗 w 在实验中遍历多个候选。
- **训练/微调**：无，纯推理时干预（inference-time intervention），模型参数全程冻结。
