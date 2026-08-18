---
title: "Topological-Attribution-Distance-TAD-Revealing-Segment-Level"
source: https://arxiv.org/pdf/2608.16775v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 06:52:41"
field: "LLM可解释性与RAG归因"
keywords: ["Topological Attribution", "RAG Explainability", "Persistent Homology", "Incident Log Analysis", "LLM Interpretability", "Evidence Attribution", "Wasserstein Distance"]
innovations: ["首次将段级归因形式化为拓扑比较问题，建立同伦推理与嵌入几何的显式联系", "提出TAD指标，通过层聚合Wasserstein距离量化检索段对LLM输出的几何影响", "设计√N分组筛选-确认两阶段自适应策略，将计算复杂度从O(N)降至O(√N)"]
benchmarks: ["GOAD网络安全日志数据集（587条日志，20条ground-truth）"]
---

# 论文速读：Topological-Attribution-Distance-TAD-Revealing-Segment-Level

## 一句话总结
论文提出了 **Topological Attribution Distance (TAD)**，一种基于拓扑数据分析的段级归因指标，通过计算 LLM 响应嵌入空间几何在不同检索日志消融前后的拓扑变化（持久同调 Wasserstein 距离），实现网络安全事件日志与模型生成的可信溯源。

## 研究问题与动机
1. RAG 系统在网络安全运营中日益普及，但检索本身不能保证生成结果的正确解释，**信任问题**成为关键瓶颈。
2. 现有基于 LLM 裁判的方法（如 Self-RAG、RAGAS）受限于模型自身偏见和幻觉，且未与参数化知识完全解耦。
3. 标记级归因方法（LEA、SHAP、LIME 等）在**高度重叠的结构化数据**（如安全日志）上失效：相似日志共享大量 Token，导致关键区分信号被稀释，无法捕捉"段级整体归因"。
4. 现有拓扑方法（Fitz 等、Fay 等）仅关注单 token 或批次统计，**缺乏对单条生成响应的段级归因能力**，无法处理多源证据的独立影响。

## 核心贡献（创新点）
1. **首次将段级归因问题形式化为拓扑比较问题**，建立同伦推理与嵌入空间几何之间的显式联系，填补了 token 级到段级归因的空白。
2. **提出 TAD 指标**，通过持久同调（H₀/H₁）与 Wasserstein 距离量化每个检索日志对 LLM 输出几何的全局影响，本质区别在于不再依赖 Token 重叠或语义相似度，而是度量模型内部表示的几何变形。
3. **设计"筛选-确认"两阶段自适应策略**（√N 分组），将计算复杂度从 O(N) 降至 O(√N)，解决了大规模日志池下的可扩展性问题。
4. **在真实 GOAD 网络攻击日志数据集上验证**，TAD 在 Direct/Regular/Indirect 三种难度场景下均以 97% 平均准确率领先，Precision 提升超过 39%，尤其在无关键词重叠的 Indirect 场景中优势显著。

## 方法详解
1. **嵌入空间拓扑建模**：将 LLM 每层隐藏状态中的响应 Token Embedding 视为点云 X ∈ ℝ^(T×d)，采用 L∞ 距离度量，构建 Vietoris-Rips 复形 filtration，捕捉 token 间的语义连通结构。
2. **持久同调计算**：对每层隐藏状态提取 H₀（连通分量）和 H₁（环）持久同调图 Dgm₀/Dgm₁，每个点 (bᵢ, dᵢ) 表示拓扑特征的出生-死亡尺度。
3. **Wasserstein 距离度量**：定义 (∞, 1)-Wasserstein 距离 W₁^(∞)(P, Q)，通过最优部分匹配量化两个持久同调图之间的几何差异；TAD 公式为各层距离之和：
   TAD(P, Q) = Σₗ W₁^(∞)(Dgm₀(Pₗ), Dgm₀(Qₗ))
4. **筛选-确认两阶段流程**：
   - **Screen 阶段**：将 N 条日志分为 √N 个批次，逐批消融后计算 TAD，找出距离最大的 hot batch。
   - **Confirm 阶段**：在 hot batch 内进行单条日志消融，使用 gap 检测（连续 TAD 值最大间隙）识别 spike logs（最关键证据）。
5. **全上下文拼接**：所有日志按顺序拼接为完整 context，确保自回归生成的因果链可追溯，每次消融仅移除目标日志而其他变量保持不变。

## 实验与结果
1. **数据集**：来自 GOAD 环境的真实网络攻击 Windows 事件日志，经 Gestalt Pattern Matching 压缩后共 587 条日志（20 条 ground-truth 攻击日志），覆盖 pass-the-hash、lateral movement、defense evasion 等多阶段攻击行为。
2. **评估基线**：LLM-as-judge、In-line citation、Cosine Similarity（gte-modernbert-base）、ROUGE-L、LEA（作者先前工作）。
3. **主要结果**（平均 across 4 个 LLM：Qwen3-4B、Gemma3-4B、Qwen2.5-7B、Granite4.1-8B）：
   - **Accuracy**：TAD 达 97%，比最优基线（In-line citation / LLM-as-judge）高 1.60%–6.63%。
   - **F1-score**：TAD 达 60%–71%，比最优基线高 7.35%–17.63%。
   - **Precision**：TAD 在 Direct/Regular/Indirect 分别为 94.23%/86.90%/57.70%，比最优基线提升 **39.47%/47.46%/40.88%**，显著减少误报。
   - Indirect 场景（零关键词重叠）中 TAD 仍准确定位关键日志，证明其不依赖词汇匹配。
4. **最强结果**：Qwen3-4B Direct 场景 Accuracy=0.9847、F1=0.7097；H₁ 在 Indirect 场景 F1 略优于 H₀（如 Granite4.1-8B 达 0.7097 vs 0.3750），但因计算复杂度高（cubic）仍以 H₀ 为默认。

## 相关工作脉络
1. **Self-RAG / RAGAS / ARES**：LLM 裁判类 RAG 评估方法，依赖模型自评；TAD 通过数学可追溯的嵌入几何替代自评，解耦于模型参数化知识。
2. **LEA（作者先前工作）**：基于线性依赖分析的 Token 级归因；TAD 突破 Token 粒度，实现段级整体归因，克服高重叠日志下的失效问题。
3. **Fitz et al. (Hidden Holes)**：比较 Transformer/LSTM 各层拓扑复杂度；本文扩展至 RAG 段级溯源，并强调 H₀ 连通分量的关键作用。
4. **Gardinazzi et al. (Persistent Topological Features)**：用 zigzag persistence 识别冗余层；本文借鉴"每层均影响最终决策"的洞察，聚合全层 Wasserstein 距离。
5. **Fay et al. (The Shape of Adversarial Influence)**：用持久同调检测对抗样本；仅关注 last-token 和批次统计；本文支持单条响应的实时段级归因。
6. **Draganov & Skiena / Rottach et al.**：用 TDA 分析词嵌入空间几何；本文将其扩展至 LLM 隐藏状态与 RAG 归因场景。

## 局限性与未来方向
1. 计算复杂度与响应 Token 数 T 呈 O(T²)（H₀）或 O(T³)（H₁），长响应场景下仍有开销。
2. 当前使用 H₀ 作为默认，H₁ 在 Indirect 场景表现更好但未成为主流，缺乏系统性的维度选择准则。
3. 各 LLM 的 TAD 性能存在差异（如 Gemma3-4B 弱于其他模型），反映模型所学表征质量的影响，非方法论本身问题。
4. 实验基于固定 canonical response，未评估模型生成质量本身的差异；真实部署中每个模型的输出可能不同。
5. 未来方向：分析各层对归因的相对重要性、探索可扩展的 Vietoris-Rips 构造变体、将 TAD 推广至更广泛的 Agentic-AI 工作流证据溯源。

## 研究启发与可借鉴点
1. **段级归因的拓扑视角**：将归因问题转化为嵌入空间几何变形的度量问题，为"哪些输入段驱动了输出"提供了超越 token 重叠和语义相似的新范式，可迁移至任何 RAG 证据溯源场景。
2. **"筛选-确认"两阶段策略**：√N 分组 + gap 检测的自适应归因框架，在保持精度的同时将计算量从 O(N) 降至 O(√N)，值得在大规模上下文归因任务中复用。
3. **层聚合而非单层依赖**：借鉴 Gardinazzi 的发现，跨所有 Transformer 层聚合 Wasserstein 距离比依赖单一深层更具鲁棒性，适用于任何需要层重要性建模的解释任务。
4. **Gap-based spike 检测**：对排序后的 TAD 值求相邻间隙最大值作为截断阈值，替代人工设定的 top-k，具有自适应优势，可直接迁移到其他排序型归因任务。
5. **与团队结合点**：若团队关注多源证据归因、Agent 决策溯源或安全合规审计，TAD 的拓扑度量框架可与现有的 token 级归因管线互补，形成"token→segment→source"的多粒度归因体系。

## 关键术语表
**TAD (Topological Attribution Distance)**：一种段级归因指标，通过比较全上下文与逐段消融后 LLM 响应嵌入的持久同调 Wasserstein 距离，量化各检索段对输出的几何影响。
**Persistent Homology (持久同调)**：TDA 核心工具，追踪拓扑特征（连通分量、环等）在 filtration 尺度下的出生与死亡，输出持久同调图或条形码。
**Wasserstein Distance (Wasserstein 距离)**：衡量两个持久同调图之间几何差异的距离度量，通过最优部分匹配最小化特征点的运输成本。
**Vietoris-Rips Complex**：由点云及其距离阈值构建的单纯复形，距离不超过 ε 的点集形成 simplex，是多尺度拓扑分析的标准构造。
**H₀ Homology**：计数嵌入空间中连通分量的同调维度，反映 token embedding 的聚类结构，TAD 默认使用以控制计算复杂度。
**Screen-then-Confirm**：TAD 的自适应归因流程，先 √N 分组批量筛查再在候选组内逐条确认，平衡精度与计算效率。
**GOAD (Game of Active Directory)**： intentionally vulnerable Windows AD 训练环境，本文用于构建真实网络攻击日志数据集。
**Canonical Response**：实验中固定使用的标准响应文本（由 Claude-Opus-4.5 生成），确保跨模型对比的可控性。

## 可复现要素
- **数据集**：GOAD 环境 Windows 事件日志，经压缩后 587 条日志（20 条 ground-truth）；论文声明代码和数据已开源，见 https://github.com/RezzFayyazi/TAD
- **代码/权重**：代码开源；模型使用 Qwen3-4B、Gemma3-4B、Qwen2.5-7B、Granite4.1-8B（公开模型）
- **关键超参**：screen 分组数 k=⌈√N⌉，最小分组阈值 τ=9（Algorithm 1），默认同调维度 H₀，距离度量 L∞，Wasserstein 阶数 (∞,1)
- **实验环境**：Intel Xeon E5-2650 ×2 CPU、256GB RAM、NVIDIA Tesla L40S GPU
- **基准模型**：Alibaba-NLP/gte-modernbert-base（Cosine Similarity 嵌入模型）
