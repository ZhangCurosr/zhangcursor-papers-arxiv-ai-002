---
title: "S-martCirc-Self-supervised-Smart-Circuit-Discovery"
source: https://arxiv.org/pdf/2609.00755v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 15:46:06"
field: "大语言模型可解释性"
keywords: ["mechanistic interpretability", "circuit discovery", "large language models", "interpretability", "functional roles", "self-supervised"]
innovations: ["提出计算转换与信息传播的通用功能角色二分法", "基于token相似度变化的可微功能解释指标FI", "重要性-功能交替联合优化框架S3martCirc"]
benchmarks: ["IOI task on GPT-2/Llama/Qwen", "Acronym Prediction task on GPT-2/Llama/Qwen"]
---

# 论文速读：S³martCirc: Self-supervised Smart Circuit Discovery

## 一句话总结
S³martCirc 是一个统一的自监督框架，将大语言模型（LLM）的**电路发现**（找到关键组件）与**功能解释**（理解组件作用）从传统的顺序两阶段改为联合优化，通过抽象出"计算转换"和"信息传播"两种通用功能角色，显著提升了电路发现的性能和对人类验证电路的恢复能力。

---

## 研究问题与动机

1. **两阶段范式的根本缺陷**：现有机械可解释性（MI）方法先独立发现重要节点（如 attention head），再主观解释其功能，忽视了"节点重要性源于其功能"与"功能定义依赖其对任务的贡献"之间的内在相互依存关系。

2. **功能角色缺乏通用性**：前作给出的功能标签（如"S-inhibition head"、"name mover"）高度绑定特定节点和任务，无法跨任务泛化，也难以嵌入自动化流水线。

3. **功能解释主观且不可量化**：当前方法依赖人类定性分析，缺乏可直接用于优化目标的定量指标，限制了可扩展性。

4. **组件交互被忽略**：独立评估节点的方法（如 patching-based）无法捕捉组件间的重要交互，导致遗漏关键节点。

---

## 核心贡献（创新点）

1. **通用功能角色二分法**：将细粒度任务特定解释抽象为"计算转换（functional）"和"信息传播（passthrough）"两类通用角色，**本质区别**在于刻画节点"如何处理信息"而非"处理什么内容"，实现任务无关（task-agnostic）。

2. **可量化的功能解释指标（FI）**：基于输入/输出 token 相似结构变化率来定量判定节点类型，**本质区别**在于用数学度量替代主观判断，使功能角色可直接嵌入优化目标。

3. **交替联合优化框架**：通过 Important Node Discovery 和 Important Node Classification 两个阶段的交替训练，显式建模重要性与功能角色的互依关系，**本质区别**在于打破顺序两阶段的割裂，实现 bidirectional influence。

4. **软掩码 + 稀疏性正则**：引入可学习的软掩码系数（$c_{n,u}, c_{n,p}, c_{n,f}$）配合稀疏性惩罚项，在保证可微性的同时推动电路稀疏化，**本质区别**在于不同于硬剪枝或单次优化方法。

---

## 方法详解

### 节点掩码系数建模
对每个节点 $n$，定义三个可学习概率系数：
- $c_{n,u}$：节点不重要的概率
- $c_{n,p}$：节点为 passthrough 角色的概率
- $c_{n,f}$：节点为 functional 角色的概率

约束：$c_{n,u} + c_{n,p} + c_{n,f} = 1$。

评估时用硬指示函数：
$$\hat{c}_i = \mathbb{1}[i = \arg\max_j c_j]$$

### Important Node Discovery（关键节点发现）
联合优化两个目标：
- **忠实度**：最小化掩码后输出分布与原始分布的 KL 散度
- **任务性能**：最小化交叉熵损失

综合损失：
$$L_{\text{disc.}} = \alpha \cdot \sum_{v \in V} P_{\text{orig}}(v) \log \frac{P_{\text{orig}}(v)}{P_{\text{mask}}(v)} - (1 - \alpha) \cdot \log P_{\text{mask}}(y)$$

其中 $c_{n,p} + c_{n,f} > c_{n,u}$ 的节点被选为候选电路节点。

### Important Node Classification（关键节点分类）
提出**功能解释指标 FI**：
$$\text{FI}(X, Y) = \frac{\|\text{Sim}(X) - \text{Sim}(Y)\|_F}{\|\text{Sim}(X)\|_F}$$

其中 $\text{Sim}(A) = \hat{A}\hat{A}^T \odot (\mathbf{1}\mathbf{1}^T - I)$ 衡量 token 间相似度结构（行方向 L2 归一化后的余弦相似度，排除自相似）。

FI 越大 → 节点改变了 token 关系结构 → functional node；FI 越小 → 保留结构 → passthrough node。

分类损失：
$$L_{\text{class.}} = \sum_{n_i \in N_i} \left[ -c_{n_i,f} \cdot \text{FI}(X_{n_i}, Y_{n_i}) - \frac{c_{n_i,p}}{\text{FI}(X_{n_i}, Y_{n_i})} \right]$$

### 稀疏性正则与交替训练
总损失加入三项正则：
- **Confidence**：推动系数趋向 {0, 1}
- **Validity**：确保三系数和为 1
- **Sparsity**：惩罚节点被选入电路

训练流程：先 warmup 阶段仅优化 $L_{\text{disc.}}$ 发现候选节点，之后交替执行 Discovery 和 Classification 阶段，实现双向优化。

---

## 实验与结果

### 实验设置
- **模型**：GPT-2、Llama 3.2 1B、Qwen 3 1.7B
- **任务**：IOI（间接宾语识别）、Acronym Prediction（首字母缩略词预测）
- **基线**：Activation Patching、Attribution Patching、Attribution Patching IG、IBCircuit、Pruning、Random
- **评估指标**：Circuit Size、Accuracy Drop（Acc.）、Logit Difference（Logit.）

### 核心结果（RQ1）
S³martCirc **全面超越所有基线**：

| 模型 | 任务 | S³martCirc Acc. | 最强基线 Acc. | 提升 |
|------|------|-----------------|---------------|------|
| GPT-2 | Acronym | **-97.50%** | -54.67% (Act.) | +42.83pp |
| GPT-2 | IOI | **-95.99%** | -70.71% (Act.) | +25.28pp |
| Llama | Acronym | **-57.33%** | -59.17% (Act.) | 相当 |
| Llama | IOI | **-69.00%** | -5.67% (Attr. IG) | +63.33pp |
| Qwen | Acronym | **-100.00%** | -100.00% (Act.) | 持平 |
| Qwen | IOI | **-100.00%** | -33.00% (Act.) | +67.00pp |

**关键发现**：S³martCirc 在 comparable 电路规模下造成更大的性能下降，说明选出的节点更关键。例如 Llama-IOI 中，S³martCirc 用 35 个节点达到 -69%，而 Attr. IG 用 47.67 个节点仅 -5.67%。

### 电路恢复能力（RQ2）
- **IOI 任务**：S³martCirc 恢复人类验证电路的比例显著高于其他方法，接近 Activation Patching，远超优化类基线。
- **Acronym 任务**：恢复 8 个已知节点中的 4 个（50%），且功能分类与人工分析一致（如 Previous Token 1.0 正确判为 passthrough，Letter Mover 11.4 判为 functional）。

### 超参敏感性（RQ3）
- **k 值**（分类阶段步数）：k=6 时性能稳定，从 k=2 的 -71% 提升到 k=6 的 -96%
- **α 值**（KL vs CE 权重）：α=0.75 最优，说明忠实度目标应主导，但 CE 项不可缺（α=1.0 时性能骤降）

### 消融实验（RQ4）
- **交替训练至关重要**：S³martCirc-DF（发现→分类顺序执行）在 Acronym 上 Acc. 仅 -4.33%，远低于联合优化的 -97.5%
- **Warmup 必要**：去掉 warmup（S³martCirc-NW）性能下降到 IOI 67%、Acronym 66%

### 运行效率（Table 2）
S³martCirc 运行时间远低于 patching 类基线（GPT-2-Acronym：197s vs Act. 6239s），与优化类基线相当。

---

## 相关工作脉络

1. **Mechanistic Interpretability & Circuit Discovery**：Wang et al. (2022) 手工验证 IOI 电路；Nanda et al. (2023) 提出 progress measures；本文定位差异——不再依赖人工分析，而是自动联合发现+解释。

2. **Patching-based 方法**：Activation Patching (Heimersheim & Nanda, 2024)、Attribution Patching (Syed et al., 2023)、Attr. IG (Hanna et al., 2024)——独立评估节点贡献，忽略组件交互；本文通过可微掩码捕捉交互。

3. **Optimization-based 方法**：IBCircuit (Bian et al., 2026) 用信息瓶颈；Pruning (Bhaskar et al., 2024) 基于梯度剪枝——只建模"哪些节点重要"，不建模"功能是什么"；本文补充了功能角色的量化嵌入。

4. **Cross-task component reuse**：Merullo et al. (2024) 发现组件在不同任务中以一致角色复用；本文受此启发，提出 task-agnostic 的功能角色二分法。

5. **Structural equivalence of circuits**：Haklay et al. (2026) 证明不同结构可实现相同功能；本文的抽象角色定义恰好兼容这种等价性。

---

## 局限性与未来方向

1. **角色粒度较粗**：两种通用角色无法还原人工分析中的细粒度语义标签（如 "negative name mover" vs "name mover"）。

2. **任务相对性**：角色定义依赖于任务特定激活，同一节点在不同任务中可能被赋予不同标签。

3. **仅在两个经典 MI 任务上验证**：IOI 和 Acronym 均为已有充分人工分析的简单任务，在更复杂真实任务上的表现待验证。

4. **仅覆盖 attention head 和 MLP neuron**：未扩展到 layer、embedding 等其他组件层级。

---

## 研究启发与可借鉴点

1. **"重要性-功能"联合优化范式**：打破顺序两阶段的思维定式，将互依变量纳入统一目标函数，可迁移至其他可解释性场景（如视觉 Transformer 电路发现）。

2. **基于相似度变化的定量角色度量（FI）**：用 token 间关系的改变程度来刻画节点功能，是一种无需人工标注的可微度量，值得在其它结构分析任务中借鉴。

3. **交替训练 + warmup 策略**：在相互依赖的双阶段优化中，先用单一目标初始化再交替精修，是稳定训练的有效技巧。

4. **通用抽象优先于细粒度解释**：为获得可量化、可泛化的指标，主动牺牲语义特异性，这一设计哲学对构建可扩展 MI 流水线有启发。

5. **可微软掩码替代硬剪枝**：用连续概率系数实现可微的节点选择，便于端到端优化，可在模型压缩、特征选择等领域复用。

---

## 关键术语表

**Mechanistic Interpretability (MI)**：通过逆向工程神经网络，将其行为还原为人可理解的算法机制的研究方向。

**Circuit Discovery**：在 LLM 的计算图中识别对特定任务至关重要的子图（组件集合）。

**Functional Role Dichotomy**：将节点功能抽象为两类——"计算转换（functional）"和"信息传播（passthrough）"，关注"如何处理"而非"处理什么"。

**FI (Functional Interpretation) Metric**：基于输入输出 token 相似度结构变化率来定量判定节点功能角色的指标。

**Soft Masking Coefficients**：$c_{n,u}, c_{n,p}, c_{n,f}$ 三个可学习概率系数，分别表示节点不重要、passthrough、functional 的概率。

**Activation Patching**：通过替换节点激活值为噪声/中间值，衡量其对输出 logit 的影响。

**Attribution Patching**：用 clean-corrupted 激活差与 logit 梯度相乘，线性近似节点贡献。

**IBCircuit**：基于信息瓶颈目标的优化型电路发现方法，同时追求行为忠实度和电路紧凑性。

---

## 可复现要素

- **数据集**：论文自建 IOI 和 Acronym 数据集（各 1000 samples），**未公开**；模板和数据生成逻辑在附录中详细描述
- **代码**：已开源 —— https://anonymous.4open.science/r/s3martcirc
- **模型权重**：使用标准 GPT-2、Llama 3.2 1B、Qwen 3 1.7B 开源权重
- **关键超参范围**：学习率 0.001–0.01、epochs 50–75、warmup 5–60、sparsity系数 0.3–1.0、α 0.25–0.75、k=6
- **实验环境**：NVIDIA RTX A4000 16GB / A16 16GB / A6000 48GB GPU 集群

---
