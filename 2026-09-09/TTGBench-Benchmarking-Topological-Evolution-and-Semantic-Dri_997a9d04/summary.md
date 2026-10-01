---
title: "TTGBench-Benchmarking-Topological-Evolution-and-Semantic-Dri"
source: https://arxiv.org/pdf/2609.08226v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 03:32:22"
field: "动态图机器学习"
keywords: ["时序图学习", "链接预测", "节点分类", "大语言模型", "基准测试"]
innovations: ["首个联合评估结构演化与语义漂移的文本丰富时序图基准", "揭示TGNN与LLM预测器在结构与语义建模上的根本性能力鸿沟"]
benchmarks: ["TTGBench", "DGB", "DyGLib", "DTGB", "TGB-Seq"]
---

# 论文速读：TTGBench: Benchmarking Topological Evolution and Semantic Drift in Text-attributed Temporal Graphs

## 一句话总结
本文提出了首个联合评估时变图结构中演化（TLP）与语义漂移（TNC）的基准测试 TTGBench，包含6个文本丰富的真实数据集，揭示出 TGNN 与 LLM 预测器在结构与语义建模能力上存在显著鸿沟。

## 研究问题与动机
- **现有基准偏向结构演化**：大多数时变图学习基准（如 DGB、TGB、DTGB）主要关注时序链接预测（TLP），对语义演化支持有限
- **语义任务设置过于简化**：虽有 DyGLib 支持时序节点分类（TNC），但仅限于简单的二元分类（如用户是否被封号），无法捕捉现实中的连续语义漂移
- **数据集边重复率高导致性能虚高**：常用数据集存在大量重复边，许多方法在基准上达到接近饱和的结果（>98%），掩盖了模型在真实高新颖性环境下的实际能力

## 核心贡献（创新点）
- **首个联合评估基准**：TTGBench 是首个同时支持多类（multi-class）和多标签（multi-label）时序节点分类的基准，填补了语义演化评估的关键空白
- **双 volatility 数据集设计**：构建6个真实文本丰富数据集，具有结构 volatility（高边新颖性、低重复率）和语义 volatility（高频标签漂移）
- **揭示范式能力鸿沟**：系统评估17种前沿方法，首次清晰揭示 TGNN-Predictors 擅长结构预测但语义跟踪失败，LLM-Predictors 表现相反的根本性局限
- **诊断性分析**：从架构语义盲视、表征锁死、结构惯性三个维度深入解释 TGNN 在语义跟踪上失败的原因

## 方法详解
**TTGBench 数据集构建**：
- 6个真实文本丰富的时序图数据集：FOOD（33K节点，多标签TNC）、IMDB（32K节点，多标签TNC）、Librarything（51K节点，仅TLP）、Beeradvocate（99K节点，多类TNC）、Ratebeer（140K节点，多类TNC）、Amazon-Kindle（216K节点，多类TNC）
- **结构 volatility**：高边新颖性（Novelty≈1.0）、低重复率、持续变化的网络密度与节点中心转移
- **语义 volatility**：高频语义漂移（FOOD/IMDB 漂移率>0.8），非平凡演化轨迹（不对称、高聚集的转移矩阵）

**任务定义**：
- **TLP**：$P(e=(u,v,t')|\mathcal{G}(\leq t)) = \sigma(f_\theta^{(link)}(u,v,\mathcal{G}_{\leq t}))$
- **TNC（多类）**：$P(y_u(t')=c|u,\mathcal{G}_{\leq t}) = \text{Softmax}(f_\theta^{(node)}(u,\mathcal{G}_{\leq t}))_c$
- **TNC（多标签）**：$P(Y_{u,c}(t')=1|u,\mathcal{G}_{\leq t}) = \sigma(f_{\theta,c}^{(node)}(u,\mathcal{G}_{\leq t}))$

**评估方法**：
- TGNN-Predictors：9种纯TGNN（JODIE、DyRep、TGAT、TGN、CAWN、TCL、GraphMixer、DyGFormer、FreeDyG）+ 2种LLM增强型（LKD4DyTAG、CROSS）
- LLM-Predictors：3种ICL方法（TGTalker、DST-v1、DST-v2）+ 3种SFT方法（LLaGA-ND、LLaGA-HO、GraphGPT）

## 实验与结果
**评估指标**：TLP用AP和AUROC（随机负采样、transductive设定）；TNC用Macro-F1和bACC

**核心发现**：
| 类别 | TLP表现 | TNC表现 | 代表方法 |
|------|---------|---------|----------|
| TGNN-Predictors | 强（JODIE 82.81% AP, TGN 92.62% AP） | 弱（JODIE 25.95% F1, TGN 21.98% F1） | JODIE, TGN, DyRep |
| LLM-Predictors | 弱（LLaGA-HO 72.64% AP, GraphGPT 55.09% AP） | 强（DST-v2 37.59% F1, GraphGPT 36.74% F1） | DST-v2, GraphGPT |

**关键结论**：
- EdgeBank（纯记忆基线）在所有数据集上性能接近随机（AUROC≈50），证明高新颖性下记忆失效
- 内存式TGNN（JODIE、DyRep、TGN）在TLP上表现最强
- SFT显著优于ICL，显式结构化输入帮助LLM-Predictors，但学习范式仍是瓶颈
- LLM-as-Enhancer方法（CROSS、LKD4DyTAG）在TLP上具竞争力，但继承了TGNN的语义跟踪弱点
- TGNN在语义漂移加快时性能急剧下降，LLM-Predictors保持稳定甚至提升

**效率分析**：GraphGPT训练耗时近60万秒、13.5 TFLOPs，远超所有TGNN；所有方法随数据规模近似线性扩展

## 相关工作脉络
- **DGB [24]**：提出link novelty概念，识别现有基准的高重复边问题，但仅关注TLP
- **DyGLib [39]**：统一评估框架，支持TNC但仅限于二元分类
- **DTGB [40]**：引入文本属性图，但仍以结构预测为核心
- **TGB-Seq [38]**：进一步识别边重复问题并提出高novelty数据集，但未涉及语义演化
- **LLM-as-Enhancer [27, 41]**：LKD4DyTAG、CROSS利用冻结LLM增强TGNN语义表示，但仍依赖TGNN聚合架构
- **LLM-as-Predictor [3, 29]**：LLaGA、GraphGPT将LLM直接用于图预测，原为静态图设计，本文首次适配至时序场景

## 局限性与未来方向
- **模型能力鸿沟难以调和**：TGNN与LLM分别在结构与语义上存在根本性架构偏差，单一范式难以兼顾
- **效率与性能的trade-off**：LLM-Predictors在语义跟踪上表现优异但计算开销巨大，限制了实际应用
- **SFT方法适配局限**：原为静态图设计的LLM-as-Predictor方法在时序场景下无法完全捕捉动态特性
- **缺少统一模型**：尚无方法能同时有效建模结构演化与语义漂移

## 研究启发与可借鉴点
- **双任务联合评估框架**：将TLP与TNC在同一基准上联合评估，可同时检验模型的结构与语义建模能力
- **高novelty数据集设计**：通过控制边重复率来避免性能虚高，更真实反映模型能力
- **消融分析范式**：通过移除文本属性、比较LP与E2E训练策略等方式系统诊断模型失败原因
- **可扩展性评估**：从静态效率扩展到渐进式数据规模增长，提供部署实践指导

## 关键术语表
- **Temporal Link Prediction (TLP)**：基于历史交互预测未来可能形成的边，评估结构演化建模能力
- **Temporal Node Classification (TNC)**：预测节点在给定时刻的语义标签，评估语义漂移追踪能力
- **Dual Volatility**：结构volatility（高边新颖性、动态密度变化）与语义volatility（高频标签漂移）的并称
- **TGNN-Predictors**：以时序图神经网络为预测骨干的方法，包括纯TGNN和LLM增强型
- **LLM-Predictors**：直接使用LLM作为预测器，分为ICL（无需训练）和SFT（微调）两类
- **Linear Probing (LP)**：仅训练分类头、冻结预训练主干网络的评估策略
- **EdgeBank**：无参数的纯记忆基线方法，记录已观察到的交互用于预测

## 可复现要素
- **数据集**：6个文本丰富时序图数据集（FOOD、IMDB、Librarything、Beeradvocate、Ratebeer、Amazon-Kindle），论文未明确声明公开链接
- **代码**：TGNN方法使用DyGLib代码库，LLM方法基于Qwen3-8B骨干，具体代码开源情况论文未明确提及
- **关键超参**：训练/验证/测试集按40%/10%/50%时序划分；LLM为Qwen3-8B；学习率2e-5（LLaGA）/2e-3（GraphGPT）；batch size 16/4/8；训练1 epoch
- **硬件**：NVIDIA RTX A6000 GPU（48GB）
