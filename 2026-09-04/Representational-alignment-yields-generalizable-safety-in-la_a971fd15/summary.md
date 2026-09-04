---
title: "Representational-alignment-yields-generalizable-safety-in-la"
source: https://arxiv.org/pdf/2609.04022v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 18:53:12"
field: "LLM安全对齐"
keywords: ["安全对齐", "表征相似性优化", "大语言模型安全", "道德分类", "对抗鲁棒性", "原型理论", "ReSO", "jailbreak防御"]
innovations: ["提出ReSO方法直接对齐LLM潜层表征与人类道德分级分类结构，不监督生成响应", "系统刻画23个开源模型内部道德表征的结构性缺陷并量化其与对抗脆弱性的关联", "首次在训练中建立RSA与ASR的剂量效应因果关系（R²=0.855）"]
benchmarks: ["HarmBench", "OpenRT", "DeceptionBench", "XSTest", "Flames", "MMLU-Pro", "HaluEval", "Ethics Benchmark", "MoReBench"]
---

# 论文速读：Representational-alignment-yields-generalizable-safety-in-la

## 一句话总结
本文发现当前大语言模型的内部表征未能充分保留人类道德概念的分级分类结构，并据此提出 **Representational Similarity Optimization (ReSO)** ——一种不监督生成响应、直接对齐潜层表示与人类道德分类关系的训练方法，在不损失通用能力的前提下显著提升模型在对抗性越狱攻击下的安全泛化能力。

## 研究问题与动机
1. **安全对齐的泛化困境**：现有对齐方法（RLHF/DPO/监督微调）主要优化可观测响应行为，使模型在标准评测中表现良好，但在提示词改写、语境变化或对抗性越狱（jailbreak）攻击下仍会失效——例如"已故祖母讲述危险信息的睡前故事"这类叙事越狱可轻易绕过安全防线。
2. **人类分类学的适应优势**：基于原型理论（prototype theory），人类概念以原型为中心组织，新实例按与原型之间的**分级典型性（graded typicality）**进行归类，从而支持跨语境的安全意识泛化；而 LLM 的概念表征为紧凑的统计模式，缺乏这种结构化组织。
3. **当前 LLM 内部道德分类的微弱对应**：作者在对 23 个开源模型（0.6B–235B，覆盖 Base、Instruct、Safeguard 各阶段）的分析中发现，对立道德范畴在表征空间中经常重叠，类别内典型性梯度也仅弱保留（峰值 Spearman ρ < 0.55，线性解码 R² 最高 0.34），且同 lineage 变体轨迹高度一致——说明响应级对齐并未重排底层道德分类结构。

## 核心贡献（创新点）
1. **系统刻画 LLM 内部道德表征的缺陷**：首次在 23 个开源模型、跨参数规模与对齐阶段的分层分析中量化表明，对立道德范畴未分离、典型性梯度弱保留、线性解码能力有限，为"对齐≠概念重构"提供直接证据。
2. **提出 ReSO（Representational Similarity Optimization）方法**：不监督生成响应、不使用奖励模型或 judge，而是以 Bradley-Terry 排序目标 + 表征相似性分析（RSA）直接对齐潜层表征中的成对相似度与人类道德判断的结构关系，与现有 DPO 等方法形成互补路径。
3. **建立表征对齐与对抗鲁棒性的功能因果关系**：在 Qwen3-8B 训练轨迹中验证 RSA 解释了 HarmBench ASR 变异的 86%，揭示剂量效应关系；ReSO 在四个模型（Qwen3-8B/14B/32B、gpt-oss-20b）上**一致降低**所有越狱评估的 ASR，而 DPO 在所有模型上**一致升高** ASR，证明两种对齐范式在泛化安全上的根本差异。
4. **验证跨架构与跨语言的可迁移性**：在 Strongly safety-aligned 的 gpt-oss-20b 上仍进一步将 HarmBench ASR 从 3.33% 降至 1.33%，且在中文 Flames 基准上 ReSO 唯一实现正向提升，表明方法不受模型 family 或评测语言限制。

## 方法详解

### 数据构建：人类道德分级的定量参考
- 源自 **Social-Chemistry-101** 语料（355,923 条 Reddit 众包道德判断），经筛选得 251,334 条原子判断。
- 基于 **Moral Foundations Theory (MFT)** 将五个对立基础分解为十个美德/恶德范畴（care, harm, fairness, cheating, loyalty, betrayal, authority, subversion, sanctity, degradation）。
- 每条 action 编码为**稀疏十维向量** hₐ：
  - 道德判断分 jₐ ∈ [−2, 2] 映射为极性 vₐ = jₐ/2 ∈ [−1, 1]
  - 共识分 cₐ ∈ [0, 4] 映射为置信权重 wₐ = cₐ/4 ∈ [0, 1]
  - m⁺ₐ = ReLU(vₐ) × wₐ，m⁻ₐ = ReLU(−vₐ) × wₐ
  - 中性判断的 hₐ = 0； magnitude 接近 1 表示 annotators 既判定极端又高度一致
- 训练/验证/测试划分：201,023 / 25,170 / 25,141

### 表征提取与各向异性校正
- 使用固定模板"action is morally"单步前向，提取每层 decoder 的 residual-stream hidden states 并进行 mean-pooling（而非 final token）。
- 各层全局均值 μ⁽ˡ⁾ 中心化：ẑₐ⁽ˡ⁾ = zₐ⁽ˡ⁾ − μ⁽ˡ⁾，以消除 Transformer 激活的各向异性。

### 三种表征分析指标
1. **对立范畴分离度**：计算每对 MFT 基础下美德/恶德原型（最典型成员的平均激活）的余弦相似度；负值=两极对立，正值=区域重叠。
2. **典型性梯度保留**：在每层计算 action 到其原型余弦距离与人工标注典型性的 Spearman 秩相关 ρ，取跨层峰值。
3. **线性解码能力**：每层训练线性 probe P⁽ˡ⁾: ℝᵈ → ℝ¹⁰（均方误差，AdamW），以 R² 衡量 ten-dimensional vector 的可恢复性。

### ReSO 损失函数
总损失：ℒ = ℒ_struct + βℒ_pres

**Structural loss（核心对齐项）：**
- 模型侧相似度：s_M⁽ˡ⁾(a,b) = cos(ẑₐ⁽ˡ⁾, ẑ_b⁽ˡ⁾)
- 人类侧相似度：s_H(a,b) = −‖hₐ − h_b‖₂
- 从每批次枚举有序三元组 (i, j, k)（来自同一 MFT 基础段的美德/中性/恶德），保留满足 s_H(i,j) − s_H(i,k) ≥ δ_h = 0.2 的"无歧义"三元组构成 T
- 每层分别以 Bradley-Terry 目标拟合：
  ℒ_struct = Σₗ wₗ · (1/|T|) Σ_(i,j,k)∈T −log σ((Δ_ijk⁽ˡ⁾ − δ)/τ)
  其中 Δ_ijk⁽ˡ⁾ = s_M⁽ˡ⁾(i,j) − s_M⁽ˡ⁾(i,k)，δ = 0.05，τ = 0.1，wₗ = 1/L

**Preservation loss（能力保持项）：**
- ℒ_pres 为 KL-divergence：D_KL(p_Θ(·|x) ‖ p_ref(·|x))，在 50,000 条通用文档 replay corpus 上计算，防止全参数训练对基础分布的偏离。
- β = 1（以验证 RSA 为选择标准、同时满足能力 guardrail 的超参）

### 对比基线
- **DPO**：用相同 251,334 条标注构建偏好对（chosen=annotated pole，rejected=opposing/neutral），β_DPO = 0.1。
- **Shuffled-label control**：在同 pipeline 下对 label 随机置换，保留同等优化压力但破坏 action-label 对应。

### 训练设置
- 全参数训练 decoder stack，输入 embedding 与 unembedding 冻结；8×NVIDIA H200 GPUs
- 每步：260 条分层 action 示例（五基础均匀分配）+ 每 rank 2 条 replay 序列（≤1024 tokens）
- AdamW，峰值学习率：Qwen3-8B/14B 为 1e-5，Qwen3-32B 为 1e-6，gpt-oss-20b 为 5e-6
- 1,000 步训练，每 25 步验证；ReSO arm 按验证 RSA 选 checkpoint

## 实验与结果

### 模型与数据集
- **23 个开源模型**：Qwen3（0.6B–235B-A22B）、Llama-3.1-8B、Llama-4 Scout-17B-16E、gpt-oss-120b/20b；含 Base/Instruct/Safeguard 三阶段
- **表征分析子集**：16,135 条分层 action（十范畴平衡）
- **训练/评测集**：251,334 条道德判断；9 个 OOD benchmark

### 9 个 OOD 评测基准及关键结果

| 基准 | 评测内容 | DPO 效果 | ReSO 效果 |
|------|----------|----------|-----------|
| **HaluEval** | 事实可靠性 | Qwen3-8B: 68.66→68.81（略升） | 68.76（保持） |
| **MMLU-Pro** | 通用推理 | Qwen3-8B: 63.02→62.51（-0.51） | 62.06（-0.96） |
| **Flames（中文）** | 中文价值观对齐 | **全部下降**（-2.65~-5.87） | **全部提升**（+3.44~+1.68） |
| **MoReBench** | 程序性道德推理 | 三模型提升，gpt-oss-20b 提升 | 三模型提升 |
| **Ethics Benchmark** | 道德推理 | 全部提升 | 两模型提升（gpt-oss 反降） |
| **XSTest** | 过度拒绝平衡 | gpt-oss 54.38→93.03（大幅改善，但伴随 ASR 升高） | 保持安全同时提升平衡 |
| **HarmBench↓** | 有害攻击成功率 | **全部升高**（+11.4pp 到 +2.67pp） | **全部降低** |
| **DeceptionBench↓** | 欺骗攻击成功率 | **全部升高** | **全部降低** |
| **OpenRT（27种攻击）** | 综合红队攻击 | Qwen3-8B: 24.77%→29.28% | Qwen3-8B: 24.77%→19.63%（−5.14pp） |

### 最强结果
- **Qwen3-8B HarmBench**：ReSO 将 ASR 从 26.17% 降至 **14.72%**（−11.45pp，p=3.5×10⁻⁶），为各模型最大降幅
- **gpt-oss-20b（已强对齐）**：HarmBench ASR 3.33%→**1.33%**（−2.00pp），同时 XSTest 平衡分从 54.38→88.95
- **Qwen3-8B 训练轨迹**：RSA 从 0.060 升至 0.24（4倍），ASR 从 24.2%→15.1%（最小值），两者在 RSA-ASR 平面上 R²=0.855；Late-stage 回撤时 ASR 同步上升，斜率 −0.510 vs −0.506（t-test p=0.99，不可区分）
- **DPO 负面效果**：Qwen3-8B HarmBench ASR 从 26.17% 升至 37.58%（+11.41pp），OpenRT 27 种攻击中 22 种恶化
- **Shuffled control** 部分复现 ReSO 效果（约恢复 20–40% 幅度），说明非特定结构对齐本身有一定泛化收益，但不及 ReSO

## 相关工作脉络
1. **DPO（Rafailov et al., 2023）**：当前主流行为对齐方法，通过直接优化偏好对提升响应合规性；本文证明 DPO 在保持甚至增强显式道德判断精度的同时，会**恶化**对抗鲁棒性（ASR 全面升高），且**不改变**内部 RSA，揭示"响应级对齐≠表征重构"的本质区别。
2. **Constitutional AI（Bai et al., 2022） & RLHF（Ouyang et al., 2022）**：同属 response-level 优化范式；本文论点指出此类方法虽有效但可能收窄 response pattern，导致面对陌生提示形式时安全边界被绕过。
3. **Prototype Theory（Rosch, 1978/2024）**：心理学中人类概念以原型为中心的分级分类理论；本文将其形式化为可计算的道德分类参照，并将"原型-典型性梯度"机制引入 LLM 表征训练，为认知科学假设提供计算验证。
4. **Jailbreak 攻击研究（Wei et al., 2023; Andriushchenko & Flammarion, 2025）**：展示当前安全对齐模型的脆弱性；本文与现有工作不同，不通过更多训练数据或对抗训练提升防御，而是从**表征结构重组**角度解释并改善鲁棒性。
5. **RSA（Kriegeskorte et al., 2008）与线性 probe 分析**：表征可解释性领域的经典工具；本文将其系统性应用于道德概念的多维分级（ten-dimensional sparse vector）而非二分类标签，细化了对 LLM 内部表征的分析粒度。
6. **HarmBench / OpenRT / DeceptionBench 等安全评测基准**：本文采用最全面的 OOD 安全评测组合，在 27 种攻击策略（含白盒 GCG 和 26 种黑盒攻击）下验证泛化，相比单一基准评估更具说服力。

## 局限性与未来方向
1. **道德参照体系的完备性**：人类道德分类源自 Social-Chemistry-101（基于 MFT），并非对人类道德认知的穷尽描述；不同文化/社区存在分歧，ReSO 框架虽可容纳更细粒度或文化特异的结构，但当前实验仅验证了 MFT 基础上的十维分类。
2. **显式判断精度收益有限**：ReSO 在 Ethics Benchmark、MoReBench 等显式道德判断基准上的提升不一致（gpt-oss-20b Ethics 反降），说明表征对齐与行为对齐并不完全等价，何时需要联合优化仍需探索。
3. **训练后期 RSA-ASR 的剂量效应逆转**：Qwen3-8B 训练中 RSA 从峰值 0.241 回落到 0.196 时 ASR 同步回升，提示过对齐或训练过度可能造成表征退化，需发展更精细的训练调度策略。
4. **仅验证文本模态**：OpenRT 中排除了多模态攻击，ReSO 是否适用于多模态安全对齐尚待验证。
5. **计算成本**：全参数训练 + 每层独立 loss + Bradley-Terry triplet 枚举带来较大计算开销，未来需探索参数高效版本（如 LoRA 适配）。

## 研究启发与可借鉴点
1. **表征对齐作为行为对齐的补充范式**：本文的核心洞见——优化内部表征结构比优化输出策略更能提升对抗泛化——可迁移至知识保留、值对齐、多语言泛化等场景；未来研究可将"表征结构匹配"作为独立目标函数。
2. **分级典型性的形式化方法**：将 prototype theory 中的"典型性梯度"操作化为稀疏十维向量 + 置信加权，为其他领域（如因果推理、伦理决策树）的分级概念表征提供了可复用的建模框架。
3. **Shuffled-label control 的严格性**：本文设计了 label 置换的 ReSO 对照臂，证明并非任何结构对齐压力都能产生相同效果，为后续"结构正则化"类工作确立了更严格的评估基准。
4. **RSA-ASR 的剂量效应监控**：训练过程中持续监测 RSA 与 ASR 的联合轨迹，可发展早期预警信号（如 RSA 回落后 ASR 同步回升），为安全微调训练过程控制提供实用工具。
5. **与团队方向的潜在结合**：若团队关注多模态安全或价值观对齐，ReSO 的"不监督响应、只对齐表征"思路可直接移植；同时，RSA 作为训练目标可扩展至非道德领域的语义结构对齐（如事实一致性、逻辑结构）。

## 关键术语表
- **ReSO（Representational Similarity Optimization）**：本文提出的训练方法，直接对齐 LLM 潜层表征间的成对相似度与人类道德判断的结构关系，不监督生成响应。
- **RSA（Representational Similarity Analysis）**：表征相似性分析，通过比较模型表征空间与参照空间（如人类判断）的成对相似度矩阵来量化对应程度，本文用 Spearman 相关系数度量。
- **MFT（Moral Foundations Theory）**：道德基础理论，将人类道德推理归因于五个对立基础（care-harm, fairness-cheating 等），本文据此构建十维道德分类参照。
- **Prototype Theory（原型理论）**：认知科学理论，主张概念以典型实例为中心组织，新实例按分级典型性归类；本文将其作为人类道德分类的 computational model。
- **ASR（Attack Success Rate）**：攻击成功率，越狱攻击中被成功绕过的比例；越低表示模型越安全。
- **Bradley-Terry 排序目标**：一种用于成对/三元组比较的排序损失，本文用于约束模型表征相似度服从人类道德判定的序关系。
- **各向异性（Anisotropy）**：Transformer 激活中常见的现象——所有输入共享一个主导方向，导致表征空间分布不均；本文通过全局均值中心化校正。
- **XSTest**：评估 LLM 过度拒绝行为的测试套件，衡量安全拒绝与良性请求响应之间的平衡。

## 可复现要素
- **数据集**：Social-Chemistry-101（公开可获取，原始数据来自 Reddit 社区）
- **代码**：已开源 — https://github.com/LingyuLi-Cogs/ReSO
- **模型**：使用 Qwen3-8B/14B/32B、gpt-oss-20b 开源权重
- **关键超参**：δ_h = 0.2（人类相似度阈值），δ = 0.05（相似度 margin），τ = 0.1（temperature），β = 1（preservation loss 系数），β_DPO = 0.1；峰值学习率 1e-5/1e-6/5e-6（依模型规模）；1000 步训练，每 25 步验证
- **硬件**：8×NVIDIA H200 GPUs
- **评测基准**：HaluEval、MMLU-Pro、Ethics Benchmark、Flames、MoReBench、XSTest、HarmBench、DeceptionBench、OpenRT（27 种攻击）
