---
title: "SciMIF-Understanding-Multimodal-Instruction-Following-in-Sci"
source: https://arxiv.org/pdf/2608.25973v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 18:58:55"
field: "多模态科学智能评测"
keywords: ["多模态大语言模型", "指令遵循", "科学推理", "基准评测", "约束遵循", "跨学科"]
innovations: ["构建专家驱动的10类功能约束分组与42条学科适配约束的层次化taxonomy", "提出高保真指令注入管线，在不改变参考答案前提下系统叠加科学/通用约束", "揭示参数规模与科学指令遵循能力无线性关系及正确性与遵循性弱耦合的发现"]
benchmarks: ["SciMIF", "CFBench", "SciIF", "MM-IFEngine", "IFEval"]
---

# 论文速读：SciMIF-Understanding-Multimodal-Instruction-Following-in-Sci

## 一句话总结
论文提出了 SciMIF，一个面向科学领域的多模态指令遵循基准，覆盖化学、地理、生物、材料科学和物理五大学科，通过专家构建的 10 类功能约束分组（共 42 条具体约束）和 2,527 条样本，系统评估 MLLM 在科学任务中遵循复杂指令的能力。

## 研究问题与动机
1. **科学评估仅看答案正确性存在缺陷**：现有科学基准（如 MMMU、SciBench）主要判断最终答案是否正确，但"答案正确"不等于"输出可用"——模型可能给出科学上正确的答案却违反单位、格式或分析步骤要求，或严格遵守格式却依赖错误的科学推理。
2. **科学指令遵循与通用指令遵循存在本质差异**：科学指令具有三个独特属性——① 依赖领域知识（如化学键数、官能团数量需理解科学实体而非表面文本操作）；② 跨学科语义差异大（同一术语约束在化学、地理、生物中要求完全不同）；③ 频繁涉及多模态输入（分子结构、显微图像、物理图表等）。
3. **现有指令遵循基准缺乏科学维度**：IFEval、FollowBench、MM-IFEngine 等通用基准不覆盖科学知识约束；SciIF 虽涉及科学任务但仅限纯文本，且其约束未从学科表征和惯例中系统推导。
4. **参数规模扩展不等于指令遵循能力提升**：初步实验发现模型规模增大并不带来约束遵循的线性提升，指示了当前对齐策略的不足。

## 核心贡献（创新点）
1. **构建了专家驱动的 Scientific Constraint Taxonomy**：基于 22 项任务、13 个数据集、5 个学科的系统分析，提出了 10 个功能约束分组（Procedure/Number/Method/Unit/Format/Terminology/Precision/Letter/Structure/Selection）及其在各自学科中的具体实现，与 SciIF 的通用流程约束形成本质区别。
2. **设计了可扩展的高保真指令注入管线**：提出"种子准备→约束识别→约束注入→人工验证"的自动化框架，在不改变参考答案的前提下系统地叠加科学约束和通用约束，并通过双重自动验证（$\mathbb{I}_{included} \cap \mathbb{I}_{unchanged}$）确保质量。
3. **构建了首个支持多模态的科学指令遵循基准 SciMIF**：包含 2,527 条样本（27.50% 含多模态输入），覆盖 5 个学科和 10 个功能分组，同时支持 CSR、ISR、DRFR 三种评估指标以及科学正确性与指令遵循性的分离评估。
4. **揭示了 MLLM 科学指令遵循的系统性局限**：跨学科性能差异显著（化学和地理最难）、细粒度约束（Letter/Number/Format）是主要瓶颈、模型规模与指令遵循能力无线性关系、科学正确性与指令遵循性弱耦合（CF<30%）。

## 方法详解
**约束体系设计（两层结构）**：
- **域层（Domain）**：General + 5 个科学学科（Chemistry/Geography/Biology/Material/Physics）。
- **分组层（Group）**：10 个功能约束分组，每组的学科实例化各不相同（如 Terminology 在化学中要求有效分子命名，在地理中要求地址层级格式）。

**数据构建流程（Algorithm 1）**：
1. 种子样本表示为 $x = (q, I, a, t)$，其中 $q$ 为文本问题、$I$ 为可选视觉输入、$a$ 为参考答案、$t$ 为任务类型。
2. 确定适用约束集 $\mathcal{C}_x = \mathcal{C}_x^{\text{sci}} \cup \mathcal{C}_x^{\text{gen}}$。
3. 识别原始查询中已存在的约束 $C_o = f_r(q, C_x)$，避免冗余注入。
4. 科学约束注入：$q_d = f_d(q, C_s)$，按学科术语调整措辞但保持操作要求不变。
5. 通用约束注入：从 $C_g$ 中抽样最多 N=3 个不同类别，逐类注入并通过双重验证 $I_{included}(q_g, c_n) \wedge I_{unchanged}(q_g, a)$；失败最多重试 k=3 次。
6. 人工验证：两名 annotator 独立检查逻辑连贯性和约束保真度，884 条样本经过修订。

**评估指标**：
- **CSR（Constraint Satisfaction Rate）**：$\frac{1}{m}\sum_{i=1}^{m}\left(\frac{1}{n_i}\sum_{j=1}^{n_i}s_{i,j}\right)$，各约束满足率的均值。
- **ISR（Instruction Satisfaction Rate）**：$\frac{1}{m}\sum_{i=1}^{m}s_i$，所有约束均满足的指令比例。
- **DRFR（Decomposed Requirements Following Ratio）**：$\frac{\sum_i\sum_{j=1}^{m_i}r'_{i,j}}{\sum_i m_i}$，细粒度需求的整体满足比。
- 验证方式：确定性约束使用 Script-Based（正则、结构化解析、RDKit 等），过程性约束使用 LLM-as-a-Judge（GPT-4.1）。

## 实验与结果
**数据集统计**：2,527 条样本，来自 13 个源数据集（ChemEval、IMAGEO-Bench、LAB-Bench、MatCha、PhysUniBench 等），平均每条问题 211.03 tokens，27.50% 含多模态输入。

**模型评估范围**：4 个闭源（GPT-5.2、Grok-4-Fast、Gemini-3.1-Pro-Preview、Claude-Sonnet-4.6）+ 5 个开源系列（Qwen3.5-27B/35B-A3B/122B-A10B/397B-A17B，InternVL3.5-8B/14B/38B）。

**关键结果**：
- **最强整体表现**：GPT-5.2 以 ISR 65.67% 位居闭源第一，领先第二名 Grok-4-Fast（53.04%）12.63 个百分点；开源最强为 Qwen3.5-397B-A17B（57.39%）。
- **学科差异显著**：生物（ISR 86.58%~72.82%）和材料（88.93%~78.07%）表现最好；化学（ISR 46.33%~62.02%）和地理难度最高。GPT-5.2 在化学上仅得 46.33% ISR，显著低于生物学 72.82%。
- **规模悖论**：InternVL3.5 系列 8B（51.37%）与 38B（51.64%）几乎持平；Qwen3.5 系列 27B（51.37%）→ 122B（50.41%）反而下降。参数扩展不带来线性提升。
- **通用 vs 科学约束**：GPT-5.2 在科学约束上 DRFR 为 88.74%，显著高于通用约束的 74.65%。
- **约束分组难度**：Letter（35.33%~53.90%）、Number（43.23%~68.62%）、Format（48.76%~82.31%）最难；Procedure（76.74%~94.19%）、Method（94.55%~98.38%）相对简单。
- **正确性与遵循性的解耦**：CF 类别（同时正确且遵循）占比低于 30%；约 20% 样本为 CV（正确但违反约束）；多模型 IF 比例超过 30%。Pearson χ² 检验显示 5/6 模型显著相关但 φ 系数仅 0.0672~0.2069，Jaccard 相似度仅 27.04%~36.40%。

## 相关工作脉络
1. **IFEval（Zhou et al. 2023）**：通用指令遵循基准，9 组 25 条约束，541 样本，纯文本；SciMIF 与其本质区别在于覆盖科学约束和多模态输入。
2. **FollowBench（Jiang et al. 2024）**：通用基准，评估不同难度约束组合，820 样本；不涉及科学领域特定约束。
3. **MM-IFEngine（Ding et al. 2025）**：将指令遵循扩展到多模态输入，但仍是通用领域，不覆盖科学知识约束。
4. **SciIF（Su et al. 2026）**：科学指令遵循基准，3 组 10 条约束，1,244 样本，4 个学科；但其约束未从学科惯例中系统推导，且不支持多模态输入，约束类型更少。
5. **CFBench（Zhang et al. 2025b）**：通用约束遵循基准，引入 CSR 和 ISR 指标，被 SciMIF 采用为评估基础；但同样缺乏科学维度。
6. **ScienceQA/MMMU/SciBench/MathVista**：科学推理基准，主要评估最终答案正确性，不系统检验输出是否满足显式约束。

## 局限性与未来方向
1. **学科覆盖有限**：仅覆盖 5 个代表性学科，地球科学、医学、环境科学等重要领域尚未纳入，可能无法全面反映科学指令遵循的能力图谱。
2. **指令注入的人工干预依赖**：虽然引入了自动化验证，但科学约束的选择和注入仍需领域专家参与，大规模扩展到其他学科的成本较高。
3. **多模态样本占比较低（27.50%）**：部分学科（化学、生物）完全为纯文本，限制了多模态指令遵循能力的全面评估。
4. **评估工具的限制**：LLM-as-a-Judge 可能存在Judge偏差，Script-Based 验证对开放推理过程的评估有限。
5. **未来方向**：面向约束环境的实际应用训练、结合外部数值计算工具/符号引擎的结构感知方法、科学知识与指令遵循的联合优化。

## 研究启发与可借鉴点
1. **约束分解与独立评估框架可直接复用**：将复合指令拆解为独立可验证的子约束（CSR/DRFR），并分别评估"科学正确性"和"指令遵循性"的分离视角，对评测体系设计有借鉴价值。
2. **学科自适应的约束实例化策略**：同一功能约束组在不同学科中有截然不同的语义实现（如 Terminology 约束），这种"抽象分组+具体实例化"的两层设计可用于构建其他专业领域的评测基准。
3. **参数规模≠指令遵循能力的发现**：对团队模型扩缩实验设计有启发——在科学/专业领域中，单纯的 Scaling Up 可能不足以提升细粒度约束遵循能力，需要针对性的对齐优化。
4. **多模态 vs 纯文本的性能退化模式**：地理、材料、物理的多模态子集 DRFR 显著低于纯文本子集（如地理差距达 23.25pp），提示在构建多模态科学应用时需特别关注视觉输入对指令遵循的干扰。
5. **Constraint Injection Pipeline 的可迁移性**：双验证机制（$I_{included} \cap I_{unchanged}$）可用于其他领域的数据增强与评测构建，确保新约束不影响原有答案正确性。

## 关键术语表
**MLLM（Multimodal Large Language Model）**：融合视觉与语言能力的多模态大语言模型，是本文评估的对象。
**CSR（Constraint Satisfaction Rate）**：约束满足率，衡量所有约束的平均满足比例。
**ISR（Instruction Satisfaction Rate）**：指令满足率，衡量所有约束均被满足的指令比例。
**DRFR（Decomposed Requirements Following Ratio）**：细粒度需求遵循比，从更细粒度评估指令满足情况。
**科学约束（Scientific Constraint）**：根植于特定学科知识和惯例的约束，如化学分子命名有效性、地理地址层级等。
**通用约束（General Constraint）**：跨学科适用的功能性约束，如输出格式、大小写、数值精度等。
**CF/CV/IF/IV 分类**：科学正确性与指令遵循性的四象限分类，CF=Correct&Followed，CV=Correct but Violated，IF=Incorrect but Followed，IV=Incorrect and Violated。

## 可复现要素
- **数据集**：2,527 条样本，来源为 13 个已公开科学数据集；原始数据使用受各自许可证约束，SciMIF 将发布处理后的样本及数据处理脚本。
- **代码**：论文声明数据与代码将在 https://github.com/shenye7436/SciMIF 开源。
- **模型权重**：使用 GPT-5.2、Grok-4-Fast、Gemini-3.1-Pro-Preview、Claude-Sonnet-4.6 等闭源 API 及 Qwen3.5、InternVL3.5 等开源模型进行评估，权重均来自各自发布渠道。
- **关键超参**：N=3（每样本注入的通用约束类别数），k=3（约束注入重试上限）；评估中使用 DeepSeek-Chat 进行答案一致性验证，GPT-4.1 作为 LLM-as-a-Judge 和约束提取器。
- **评估工具**：RDKit（化学验证）、CompassVerifier-32B（科学正确性判断）。
