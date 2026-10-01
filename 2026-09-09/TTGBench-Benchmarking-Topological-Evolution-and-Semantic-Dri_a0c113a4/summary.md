---
title: "TTGBench-Benchmarking-Topological-Evolution-and-Semantic-Dri"
source: https://arxiv.org/pdf/2609.08226v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 03:32:48"
field: "时变图表示学习"
keywords: ["temporal graph learning", "benchmark", "topological evolution", "semantic drift", "large language model", "graph neural network"]
innovations: ["首个联合评测时变图结构演化与语义漂移的统一基准", "提出双重波动性数据集并揭示TGNN与LLM范式的根本能力鸿沟"]
benchmarks: ["TTGBench", "DGB", "TGB", "DTGB", "DyGLib"]
---

# 论文速读：TTGBench-Benchmarking-Topological-Evolution-and-Semantic-Drift-in-Text-Attributed-Temporal-Graphs

## 一句话总结
论文提出 TTGBench 基准，首次联合评估时变文本图的**拓扑结构演化**（TLP）与**语义漂移**（TNC），构建6个具有双重波动性的高新颖度真实数据集，揭示当前 TGNN 与 LLM 预测范式之间存在根本性能力鸿沟。

## 研究问题与动机
1. **语义演化被忽视**：现有基准（如 DGB、TGB、DTGB）几乎只关注结构演化（TLP），对语义漂移的支持极有限；即便有 TNC 任务，也多为简单二元分类（如用户是否被封号），无法反映真实场景中连续、多类的语义漂移。
2. **数据集链接重复率过高**：主流数据集（如 DyGLib 引用的基准）存在大量重复边，导致模型性能虚高（部分方法 AP > 98%），掩盖了在高新颖度环境下的真实能力。
3. **缺乏统一评测框架**：尚无同时支持多类/多标签 TNC 与时变链接预测的标准化基准，阻碍了对模型联合建模结构与语义能力的公平比较。
4. **两种主流范式存在结构性偏置**：TGNN 依赖消息传递聚合结构信息，LLM 依赖文本理解，但二者均难以兼顾拓扑演化与语义追踪，需系统诊断其瓶颈。

## 核心贡献（创新点）
1. **首个联合评测 TLP + TNC 的统一基准**：构建 TTGBench，首次同时评估时变图的结构演化与语义漂移，填补多类/多标签 TNC 评测空白。
2. **提出"双重波动性"（Dual Volatility）数据集特性**：6 个真实数据集具有持续高密度的链接 novelty（≈1.0）与高频语义漂移（drift ratio 常 > 0.8），远超现有基准的挑战性。
3. **揭示 TGNN 与 LLM 范式的根本性能力鸿沟**：TGNN-Predictors 在 TLP 上显著优于 LLM-Predictors，但在 TNC 上全面失败；反之 LLM-Predictors 在 TNC 上领先，TLP 仅接近随机水平。
4. **诊断性分析揭示三类架构瓶颈**：TGNN 存在"语义盲视"（architectural semantic blindness）、"表征锁定"（representation lock-in）和"结构惯性"（structural inertia under semantic volatility）；LLM-Predictors 受限于输入级结构编码不足与高昂计算成本。
5. **提供首个面向时变文本图的 LLM-as-Predictor 评测体系**：将 LLaGA、GraphGPT、TGTalker 等方法适配到时变图场景，建立 ICL 与 SFT 两类范式的系统性对比。

## 方法详解
### 任务定义
- **时变链接预测（TLP）**：给定历史时变图 $\mathcal{G}(\le t)$，预测未来时刻 $t' > t$ 是否存在边 $(u,v)$：
  $$P(e=(u,v,t') \mid \mathcal{G}(\le t)) = \sigma(f_\theta^{(\text{link})}(u,v,\mathcal{G}_{\le t}))$$
- **时变节点分类（TNC）**：
  - 多类：$P(y_u(t')=c \mid u, \mathcal{G}\le t) = \text{Softmax}(\cdot)_c$
  - 多标签：$P(\mathbf{Y}_{u,c}(t')=1 \mid u, \mathcal{G}\le t) = \sigma(f_{\theta,c}^{(\text{node})}(u, \mathcal{G}\le t))$

### 数据集构造（Dual Volatility）
| 数据集 | 节点数 | 边数 | Novelty | 类别数 | 任务类型 |
|---|---|---|---|---|---|
| FOOD | 33,773 | 401,057 | 1.0 | 9 | TLP + 多标签 TNC |
| IMDB | 32,371 | 310,891 | 1.0 | 8 | TLP + 多标签 TNC |
| Librarything | 51,201 | 785,690 | 1.0 | - | TLP |
| Beeradvocate | 99,361 | 1,586,573 | 0.99 | 7 | TLP + 多类 TNC |
| Ratebeer | 139,538 | 2,924,105 | 1.0 | 8 | TLP + 多类 TNC |
| Amazon-Kindle | 215,504 | 5,621,343 | 1.0 | 8 | TLP + 多类 TNC |

- **结构波动性**：交互密度持续波动 + top 5% hub 活动高度不稳定（网络中心不断迁移）
- **语义波动性**：label drift ratio 在 FOOD/IMDB 常 > 0.8；转移矩阵呈非对称高密度簇（非线性、非平凡演化轨迹）

### 评测方法（17 种）
**TGNN-Predictors（11 种）**：
- 纯 TGNN：JODIE、DyRep、TGAT、TGN、CAWN、TCL、GraphMixer、DyGFormer、FreeDyG
- LLM-as-Enhancer：LKD4DyTAG、CROSS（冻结 LLM 提取语义后蒸馏/融合至 GNN）

**LLM-Predictors（6 种）**：
- ICL：TGTalker、DST-v1、DST-v2（将 (u,v,t) 三元组线性序列化，无训练）
- SFT：LLaGA-ND、LLaGA-HO、GraphGPT（基于 BFS 邻居序列，微调 projector/embedding layer，骨干 Qwen3-8B）

### 评估协议
- 划分：40%/10%/50% 时序切分（transductive + inductive）
- TLP 负采样：random / historical / inductive 三种策略；指标：AP、AUROC、MRR
- TNC 指标：Macro-F1、Balanced Accuracy（bACC）
- 实现：TGNN 沿用 DyGLib 协议，先 TLP 预训练再初始化 TNC；LLM 使用 Qwen3-8B + Sentence-BERT 文本编码

## 实验与结果
### TLP 主结果（Transductive, Random Negative Sampling）
| 方法 | FOOD AP | IMDB AP | Librarything AP | Beeradvocate AP | Amazon-Kindle AP | Ratebeer AP |
|---|---|---|---|---|---|---|
| **JODIE** | **72.96** | 60.30 | **75.76** | 79.75 | 82.81 | 92.62 |
| TGN | 77.74 | 45.40 | 82.60 | 83.46 | 71.77 | 91.55 |
| DyRep | 72.83 | 64.79 | 73.81 | **89.61** | 78.35 | 85.87 |
| GraphMixer | 66.98 | 57.01 | 62.13 | 86.02 | 81.53 | 78.53 |
| LLaGA-HO | 61.95 | 54.37 | 56.87 | 58.00 | 72.64 | 56.09 |
| EdgeBank | 50.00 | 50.00 | 50.00 | 50.51 | 50.00 | 50.07 |

**结论**：记忆型 TGNN（JODIE、TGN、DyRep）全面领先；EdgeBank（纯记忆基线）≈ 随机，验证高 novelty 环境下记忆失效。

### TNC 主结果
| 方法 | FOOD Macro-F1 | IMDB Macro-F1 | Beeradvocate Macro-F1 | Amazon-Kindle Macro-F1 | Ratebeer Macro-F1 |
|---|---|---|---|---|---|
| **DST-v2** | **37.59** | 29.14 | 16.12 | 24.64 | 13.34 |
| GraphGPT | 36.74 | 25.79 | 16.31 | 24.28 | 18.93 |
| DyGFormer | 36.58 | 20.05 | 12.91 | 20.19 | 15.52 |
| LLaGA-HO | 30.37 | 20.66 | 15.09 | 21.01 | 16.94 |
| JODIE | 25.95 | 12.65 | 7.00 | 17.60 | 7.86 |
| TGN | 21.98 | 8.51 | 8.35 | 13.22 | 7.34 |

**结论**：LLM-Predictors（DST-v2、GraphGPT）全面压制 TGNNs；最差 TGNN 方法 Macro-F1 低至 7–8%。

### 诊断实验
- **文本消融**：去除文本特征对 TGNN TLP 影响巨大，但对 TNC 影响小；LLM-Predictors 相反。
- **LP vs E2E（TGNN TNC）**：端到端微调几乎无提升甚至下降（Table 22），证实表征锁定。
- **语义漂移频率敏感性**：TGNN 性能随漂移频率急剧恶化，LLM-Predictors 保持稳健甚至改善。
- **效率**：GraphGPT 训练耗时 ≈ 600,000s、13.5 TFLOPs，远超 TGNN；所有方法显式线性扩展。

## 相关工作脉络
1. **DGB / TGB（Poursafaei et al., 2022; Huang et al., 2023）**：早期时变图基准，侧重 TLP，无文本属性与节点语义任务。本文与其定位差异在于补充语义演化评测并解决高重复边问题。
2. **DTGB（Zhang et al., 2024）**：引入文本属性时变图，但仍以 TLP 为主，TNC 未覆盖多类/多标签场景。本文拓展至联合结构-语义评测。
3. **DyGLib（Yu et al., 2023）**：提供统一评测管道，但 TNC 任务为简单二元分类（如用户封禁），无法建模复杂语义漂移。本文明确填补多类/多标签 TNC 空白。
4. **TGB-Seq（Yi et al., 2025）**：识别既往基准边重复过高的问题并提出高 novelty 数据，但未涉及语义演化。本文在其基础上进一步引入语义波动性。
5. **LLM-as-Predictor 静态图工作（LLaGA、GraphGPT）**：原设计面向静态图，本文首次将其适配到时变图并系统评测其结构建模瓶颈。
6. **TGTalker / LLM4DyG（Huang et al., 2025; Zhang et al., 2024）**：探索 LLM 解构时空信息的方法，本文在其基础上扩展至多任务联合评测并比较 ICL/SFT 两种训练范式。

## 局限性与未来方向
1. **LLM-Predictors 计算开销过大**：GraphGPT 需 60 万秒训练时间与 13.5 TFLOPs，难以在真实场景中部署；高效轻量架构是必要方向。
2. **单一骨干与固定超参**：LLM 统一使用 Qwen3-8B，未探索不同规模/架构 LLM 的表现差异；SFT 方法沿袭静态图设置，时间敏感结构编码仍有改进空间。
3. **未覆盖异质/多模态时变图**：数据集虽含文本边属性，但未涉及图像、音频等多模态语义；节点与边文本长度分布右偏（短文本稀疏、长文本多义），模型鲁棒性未充分检验。
4. **未来方向**：① 设计统一结构-语义联合建模架构；② 提升 LLM 的结构推理能力；③ 增强 TGNN 的语义自适应能力；④ 探索更高效的可部署范式。

## 研究启发与可借鉴点
1. **"双重波动性"可作为数据集设计准则**：在构建新基准时，应同时量化 structural novelty（边重复率）与 semantic drift ratio（标签转移频率），避免高重复导致的性能虚高。
2. **ICL vs SFT 的结构性对比框架**：将 LLM-as-Predictor 分为序列化三元组（ICL）与 BFS 邻居序列（SFT）两条技术路线进行系统性对比，为后续 LLM+图工作提供清晰的研究坐标。
3. **诊断实验设计值得复用**：文本消融、LP vs E2E 微调对比、漂移频率敏感性分析三层诊断体系，可有效剥离模型失败原因，建议纳入同类工作。
4. **跨范式能力鸿沟的发现**：TGNN 与 LLM 分别擅长结构/语义维度，提示后续工作可探索"双通路"或"交替优化"的统一架构，而非单一路径。
5. **效率-性能 trade-off 定量分析**：除准确率外，系统报告训练时间、GPU 显存、参数量与 FLOPs，为实际应用选型提供实用参考。

## 关键术语表
- **Temporal Link Prediction (TLP)**：基于历史交互序列预测未来时刻两个节点间是否会产生新边，用于刻画时变图的结构演化。
- **Temporal Node Classification (TNC)**：预测节点在未来时刻的语义标签，支持多类/多标签设定，用于刻画节点语义漂移。
- **Dual Volatility**：论文提出的数据集属性，包含 Structural Volatility（高链接新颖度、密度/ hub 活动剧烈波动）与 Semantic Volatility（标签高频转移、转移矩阵非平凡）。
- **TGNN-Predictor**：以 Temporal GNN 为预测主干的范式，包括纯 TGNN 与 LLM-as-Enhancer 混合方法，强于结构建模弱于语义追踪。
- **LLM-Predictor**：直接使用大语言模型作为预测器的范式，包括 ICL 与 SFT 两类，强于语义追踪弱于结构建模。
- **Representation Lock-in**：TGNN 在 TLP 预训练后表征已强烈偏向拓扑模式，即使端到端微调也难以被语义监督信号重塑的现象。
- **Novelty**：测试集中新出现边（此前未观测到的节点对）的比例；本文所有数据集 novelty ≈ 1.0，远高于传统基准（0.02–0.67）。
- **Label Drift Ratio**：某时刻相比前一时刻标签发生变化的节点占比；FOOD/IMDB 常 > 0.8，Amazon-Kindle 约 0.3。

## 可复现要素
- **数据集**：6 个真实数据集（FOOD、IMDB、Librarything、Beeradvocate、Ratebeer、Amazon-Kindle），论文未声明单独开源链接，通常随代码仓库一并发布。
- **代码**：TGNN 方法沿用 DyGLib 代码库；LLM 方法（LLaGA、GraphGPT、TGTalker 等）沿用其原始实现；论文未提供 TTGBench 独立代码仓库。
- **权重**：Qwen3-8B 开源权重；Sentence-BERT 默认编码器。
- **关键超参**：TGNN 网格搜索调优；LLM SFT 学习率 2e-5、batch 16（LLaGA）、4/8（GraphGPT）、1 epoch；LLM 最大输出长度 4096 tokens；训练/验证/测试切分 40%/10%/50%。
- **硬件**：NVIDIA RTX A6000（48GB）。
