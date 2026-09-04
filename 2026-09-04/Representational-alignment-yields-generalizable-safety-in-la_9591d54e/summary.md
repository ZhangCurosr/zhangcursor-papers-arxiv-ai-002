---
title: "Representational-alignment-yields-generalizable-safety-in-la"
source: https://arxiv.org/pdf/2609.04022v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 15:32:26"
field: "大语言模型安全与对齐"
keywords: ["LLM安全对齐", "表示相似性优化", "对抗鲁棒性", "道德分类", "原型理论", "jailbreak防御", "表征分析"]
innovations: ["提出ReSO方法直接对齐LLM潜在表示与人类道德分类结构，不监督生成响应", "系统性揭示23个开源模型中道德分类表征的缺陷（原型分离度低、典型性梯度弱）", "建立RSA与攻击成功率的剂量-效应关系（R²=0.86），证明表征对齐带来泛化安全性"]
benchmarks: ["HarmBench", "DeceptionBench", "OpenRT", "XSTest", "Flames", "Ethics Benchmark", "MoReBench", "MMLU-Pro", "HaluEval"]
---

# 论文速读：Representational-alignment-yields-generalizable-safety-in-la

## 一句话总结
本文提出表示相似性优化（ReSO），直接将LLM潜在表示与人类道德分类结构对齐，无需监督生成响应即可显著提升模型在对抗攻击下的安全泛化能力；相比DPO等标准行为对齐方法，ReSO虽在显式道德判断上增益有限，但能一致降低各类jailbreak攻击成功率，且保持通用能力。

## 研究问题与动机
1. **现有对齐方法的脆弱性**：当前LLM对齐主要优化可观察响应（如DPO、RLHF），但当相同有害意图以不熟悉或对抗形式呈现时（如角色扮演绕过），模型仍易受攻击；人类却能凭道德概念的层级结构快速识别。
2. **LLM内部道德分类结构薄弱**：跨23个开源模型（0.6B–235B参数）的分析显示，多数模型未能有效分离对立道德类别（如authority–subversion平均余弦相似度0.65，care–harm为0.31），且类别内典型性梯度保持不足（峰值Spearman ρ < 0.55）。
3. **行为对齐未重构潜在分类**：同一模型谱系中，base、instruction-tuned、safeguard变体的表征轨迹高度一致，表明现有对齐流程并未实质性重组底层道德分类结构。
4. **原型分类能否提升泛化安全性**：若将人类基于原型的分级分类原则注入LLM表示空间，是否能带来超越响应级优化的对抗鲁棒性？

## 核心贡献（创新点）
1. **系统性揭示LLM道德表征缺陷**：首次跨23个开源模型、10个道德类别量化评估原型分离度、典型性梯度和线性可解码性，证明现有对齐方法未能重组底层道德分类。
2. **提出ReSO表示相似性优化方法**：基于Bradley-Terry排序目标直接对齐潜在表示间相似度与人类道德判断的分类关系，不监督生成响应、不使用reward model。
3. **建立RSA–ASR的剂量–效应关系**：在Qwen3-8B训练中追踪20个checkpoint，发现验证RSA解释86%的HarmBench ASR变异，揭示表征对齐强度与安全性的直接关联。
4. **验证跨架构、跨规模泛化**：在Qwen3（8B/14B/32B）和gpt-oss-20b（MoE）上均一致验证ReSO能降低27种red-team攻击中的多数ASR，证明方法不依赖特定架构或规模。
5. **控制实验证明结构重要性**：shuffled-label控制（保留优化压力但打乱标签对应）仅恢复部分ReSO效果，证明是"分类结构对齐"本身而非单纯优化强度驱动安全性提升。

## 方法详解
- **数据构建**：从Social-Chemistry-101提取251,334个原子判断，基于Moral Foundations Theory（MFT）将5对道德基础分解为10个virtue/vice类别，每个动作编码为稀疏10维向量**h**_a，分量大小为confidence-weighted典型性（judgement score × agreement score）。
- **表示提取与校正**：对16,135个动作计算layer-wise residual stream表示，mean-pool后减去每层全局均值μ^(l)校正各向异性（anisotropy），得到中心化表示**ẑ**_a^(l)。
- **ReSO损失函数**：
  - **结构损失**：在每个解码器层独立应用Bradley-Terry目标，从每批中抽取有序三元组(i,j,k)（分别来自virtue/neutral/vice段），保留满足s_H(i,j) − s_H(i,k) ≥ δ_h = 0.2的有效三元组；模型侧相似度s_M(i,j) = cos(**ẑ**_i, **ẑ**_j)，结构损失为：
    $$\mathcal{L}_{struct} = \sum_{l=1}^{L} w_l \frac{1}{|\mathcal{T}|}\sum_{(i,j,k)\in\mathcal{T}} -\log\sigma\left(\frac{\Delta_{ijk}^{(l)} - \delta}{\tau}\right)$$
    其中δ = 0.05，τ = 0.1，w_l = 1/L。该目标对潜在空间的旋转/缩放不变。
  - **保留损失**：在50,000条通用文档上进行KL散度正则化ℒ_pres = D_KL(p_θ || p_ref)，防止能力退化，β = 1（按验证RSA选取）。
  - **总损失**：ℒ = ℒ_struct + βℒ_pres。
- **对比方法DPO**：使用相同251,334标注，构建成对偏好数据（chosen = 正确pole，rejected = 对立pole），β_DPO = 0.1，其余超参与ReSO保持一致。
- **训练设置**：全参数训练（冻结input/output embedding），AdamW优化，8×H200 GPU；学习率随模型规模调整（8B/14B: 1e-5，32B: 1e-6，gpt-oss-20b: 5e-6）；1,000步，每25步验证。

## 实验与结果
- **数据集**：Social-Chemistry-101（251,334标注，201,023训练/25,170验证/25,141测试）。
- **模型**：Qwen3-8B/14B/32B（dense）、gpt-oss-20b（MoE）；对比基线：base、DPO、ReSO、shuffled control。
- **9个OOD基准**：HaluEval、MMLU-Pro（通用能力）；Ethics Benchmark、Flames（中文）、MoReBench（道德推理）；XSTest（拒绝平衡）；HarmBench、DeceptionBench、OpenRT（27种攻击：GCG、AutoDAN、PAIR、DrAttack等）。
- **主要结果**：
  - **HarmBench ASR**（越低越好）：Qwen3-8B从26.17%降至14.72%（−11.45pp），14B从22.48%降至13.33%（−9.15pp），32B从19.00%降至13.67%（−5.33pp），gpt-oss-20b从3.33%降至1.33%（−2.00pp）；DPO反向增加ASR。
  - **DeceptionBench ASR**：ReSO分别降低21.67、13.22、12.11和2.71pp。
  - **OpenRT 27种攻击**：ReSO在Qwen3-8B/14B/32B上分别减少23/23/19种攻击的ASR；DPO恶化22/23/22种。
  - **Flames（中文）**：ReSO提升Flames得分（8B: 63.80→67.24），DPO下降（63.80→61.15），证明跨语言能力保持更好。
  - **XSTest**：ReSO同步提升unsafe refusal和safe refusal，gpt-oss-20b上从54.38升至88.95；DPO虽提升balanced score至93.03，但牺牲安全性。
  - **通用能力**：ReSO对MMLU-Pro影响≤1.87pt，HaluEval≤1.29pt；DPO导致gpt-oss-20b HaluEval下降3.72pt。
  - **训练动态**：Qwen3-8B上RSA从0.060升至0.24（4倍），ASR从24.2%降至15.1%，RSA–ASR平面拟合R² = 0.855；step 700后RSA回退伴随ASR上升，证明剂量效应。
- **最强结果**：gpt-oss-20b的HarmBench ASR降至1.33%（绝对值最低），Qwen3-8B的DeceptionBench降至31.61%（相对降幅最大，53.28→31.61，−21.67pp）。

## 相关工作脉络
1. **DPO / RLHF类行为对齐**：Rafailov et al. (2023) DPO直接优化响应分布，本文证明其高效学习显式道德判断但不重组潜在分类，且增加jailbreak脆弱性。
2. **Constitutional AI / 规则对齐**：Bai et al. (2022) 通过AI反馈约束输出，与DPO类似仍停留在响应级，本文提出从表征结构层面增强泛化。
3. **Representation Analysis of LLMs**：Shani et al. (2026) 讨论LLM压缩与语义的权衡，本文将其扩展到道德分类维度，揭示现有模型原型分级结构保留不足。
4. **Red Teaming / Jailbreak 评估**：HarmBench (Mazeika et al. 2024)、OpenRT (Wang et al. 2026) 提供标准化攻击评估，本文首次在27种攻击下系统比较ReSO与DPO的鲁棒性差异。
5. **Prototype Theory 在AI中的应用**：Rosch (2024) 原型分类理论启发本文，将"分级典型性"形式化为稀疏向量并用于监督表示对齐。
6. **Anisotropy Correction**：Ethayarajh (2019)、Elhage et al. (2021) 提出Transformer激活的各向异性校正，本文将其应用于道德表征分析。

## 局限性与未来方向
1. **道德框架的局限性**：参考分类基于MFT十维度，不能穷尽人类道德认知，且存在文化差异；ReSO可兼容更细粒度或替代理论，但未验证其他道德体系。
2. **仅评估文本攻击**：OpenRT评测限于text-only HarmBench split，未覆盖多模态或语音场景。
3. **能力轻微退化风险**：gpt-oss-20b上DPO导致HaluEval下降3.72pt，ReSO虽缓解但需进一步调优保留损失权重。
4. **未探索分层/模块化对齐**：当前ReSO对各层均匀加权（w_l = 1/L），未探索哪些层对道德分类最重要或是否应差异化优化。
5. **规模化到更大模型未验证**：最大仅到235B base模型的表征分析，但训练实验仅到32B，超大模型（>100B）的ReSO效果未知。

## 研究启发与可借鉴点
1. **表征对齐可作为行为对齐的互补路径**：对于安全关键应用，可将ReSO作为DPO/RLHF的预处理或并行目标，兼顾显式性能与对抗鲁棒性。
2. **RSA作为安全指标的可迁移性**：RSA–ASR的强相关（R² = 0.86）提示可用中间表示分析替代昂贵red-teaming，用于早期训练监控。
3. **三阶段采样构造对比三元组**：从virtue/neutral/vice三段抽取有序三元组并施加δ_h = 0.2阈值过滤，可推广至其他结构化分类任务（如价值观、法律合规）。
4. **各向异性校正+mean-pooling的表征提取流程**：适用于任何需要分析Transformer内部概念的评估管线。
5. **Shuffled-label控制实验设计**：保留优化压力但破坏标签对应，可有效分离"结构信息"与"优化强度"的贡献，值得在其他对齐研究中复用。

## 关键术语表
- **Representational Similarity Optimization (ReSO)**：直接将LLM潜在表示相似度与人类分类关系对齐的训练方法，不监督生成响应。
- **Representational Similarity Analysis (RSA)**：量化模型内部表示结构与外部标签结构之间对应关系的分析工具，本文用Spearman相关衡量。
- **Attack Success Rate (ASR)**：红队攻击中成功绕过安全防御的比例，越低表示模型越鲁棒。
- **Moral Foundations Theory (MFT)**：道德心理学理论，将道德划分为care-harm、fairness-cheating等5对基础维度。
- **Prototype Theory**：认知科学中的分类理论，认为概念以典型实例为中心、成员按典型性梯度组织。
- **Anisotropy Correction**：校正Transformer激活空间中主导方向（各向异性）的技术，通过减去层均值实现。
- **Bradley-Terry Ranking Objective**：用于成对/三元组排序的目标函数，本文用于约束模型相似度符合人类道德排序。
- **Generalizable Safety**：指模型在未见过的对抗形式或上下文变化下仍能保持安全行为的特性。

## 可复现要素
- **数据集**：Social-Chemistry-101（公开），论文清理后得251,334原子判断；训练/验证/测试分割见Methods。
- **代码**：已开源，GitHub: https://github.com/LingyuLi-Cogs/ReSO
- **模型**：Qwen3-8B/14B/32B、gpt-oss-20b（开源权重）；表征分析覆盖23个开源模型。
- **关键超参**：δ_h = 0.2（三元组筛选阈值）、δ = 0.05（相似度margin）、τ = 0.1（温度）、β = 1（保留损失权重）、学习率8B/14B: 1e-5、32B: 1e-6、gpt-oss-20b: 5e-6；训练1,000步，每25步验证。
- **硬件**：8×NVIDIA H200 GPU。
- **评估基准**：HaluEval、MMLU-Pro、Ethics Benchmark、Flames、MoReBench、XSTest、HarmBench、DeceptionBench、OpenRT（27种攻击）。
