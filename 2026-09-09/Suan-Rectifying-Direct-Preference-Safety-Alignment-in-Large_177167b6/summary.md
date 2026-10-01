---
title: "Suan-Rectifying-Direct-Preference-Safety-Alignment-in-Large"
source: https://arxiv.org/pdf/2609.08634v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 03:30:50"
field: "大语言模型安全对齐"
keywords: ["安全对齐", "直接偏好优化", "过拒绝", "似然位移", "大语言模型", "Suan"]
innovations: ["从梯度层面重新设计偏好优化目标，避免 DPO 类方法的似然位移与过拒绝", "引入单调可解释的正则超参 β，实现参考策略距离的稳定控制", "无需安全标签过滤即实现安全性与帮助性的综合最优权衡"]
benchmarks: ["Malicious Instruct", "HarmBench", "AdvBench", "SORRY-Bench", "XS-Test", "OR-Bench", "AlpacaEval", "MT-Bench", "ArenaHard", "MMLU", "ARC", "NoveltyBench"]
---

# 论文速读：Suan-Rectifying-Direct-Preference-Safety-Alignment-in-Large

## 一句话总结
本文提出 Suan，一种直接在梯度层面设计偏好优化目标的安全对齐算法，通过自定义梯度加权与正则化项，在多项安全基准上同时实现低攻击成功率、低拒绝率与高回答质量，优于 DPO/IPO/SafeDPO 等现有直接对齐方法。

## 研究问题与动机
- **过拒绝（over-refusal）与效用退化**：现有 DPO、SafeDPO 等直接对齐算法为提升安全性常常以牺牲合规性和响应质量为代价，对良性提示出现过度拒绝；同时存在"似然位移"（likelihood displacement）问题，导致偏好样本的似然被系统性削弱。
- **梯度行为缺乏可解释性**：标准 DPO 对低概率和高概率样本对赋予相同的梯度权重，且其 surrogate 正则化（替代 KL 散度）会产生与真实 KL 不同的极小值，诱发训练不稳定和输出退化。
- **理论假设不成立**：已有理论保障通常假设行为策略严格等于参考策略，实际训练中这一前提并不满足，使得基于此的缓解手段（如在正样本上预训练 SFT）无法恢复理论性质。
- **开源安全对齐方法仍然脆弱**：尽管工业界使用闭源多层护栏取得了较好效果，但公开可复现的开源模型安全对齐方法仍易受多样化对抗性 jailbreak 攻击影响。

## 核心贡献（创新点）
1. 在多种模型族、数据集与基准上系统梳理了安全偏好优化中的过拒绝与效用退化问题，给出了量化对比（DPO/SafeDPO 在安全提升的同时伴随显著质量下降与过拒绝）。
2. 提出 Suan：绕过变分推导，直接在梯度层面设计偏好项与正则项，获得可解释、训练动态更稳定的闭式损失。
3. 通过八种开源模型的广泛实验，Suan 在攻击成功率、过拒绝率和帮助性三个维度均取得最优或接近最优的综合权衡，显著优于 DPO/IPO/SafeDPO。
4. 证明了 Suan 的超参数 β 具有单调、可解释的约束强度调节作用，而 DPO 在极端 β 下会出现非单调甚至质量崩塌行为。

## 方法详解
- **偏好梯度项**：从通用 DAA 梯度 $\nabla_\theta \log \pi_\theta(y^+|x) - \nabla_\theta \log \pi_\theta(y^-|x)$ 出发，引入基于偏好边际 $\tau$ 的 sigmoid 加权，低质量样本对获得更大梯度信号：$\nabla_\theta \mathcal{L}_P(\theta) = -\sigma(\log \frac{\pi_\theta(y^-|x)}{\pi_\theta(y^+|x)} - \tau)(\nabla_\theta \log \pi_\theta(y^+|x) - \nabla_\theta \log \pi_\theta(y^-|x))$。
- **正则化梯度项**：为避免 $k_2$ 估计器系数随 $\log \pi_\theta$ 线性增长而导致正则项主导训练，将 KL 正则的梯度重新缩放为 $(\sigma(\log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}) - \frac{1}{2})\nabla_\theta \log \pi_\theta(y|x)$，使偏好梯度与正则梯度量级相当。
- **闭式损失**：两项分别对应 $\mathcal{L}_P(\theta) = -\log \sigma(\log \pi_\theta(y^+|x) - \log \pi_\theta(y^-|x) + \tau)$ 和 $\mathcal{L}_R(\theta) = \sum_{y \in \{y^+, y^-\}} \log \cosh(\log \frac{1}{2} \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)})$，总目标为 $\mathcal{L}_{\text{Suan}} = \mathcal{L}_P + \beta \mathcal{L}_R$，其中 $\tau$ 控制偏好边际、β 控制正则强度。
- **无需额外过滤**：与 SafeDPO 不同，Suan 不依赖安全标签或动态数据过滤，直接使用标准偏好对训练；通过正则项抑制对已正确响应的过度优化，从而减少过拒绝，同时保持输出质量。

## 实验与结果
- **模型与数据**：Mistral-12B、Falcon3-7B、Llama-3.1-8B、Gemma-2-9B、Qwen-3-8B、Yi-1.5-9B、DeepSeek-7B、OLMo-3-7B，均在 Alpaca 上做 SFT，再用 PKU-Safe-RLHF（安全偏好集）训练。
- **安全基准**：Malicious Instruct、HarmBench、AdvBench、SORRY-Bench，指标为 Attack Success Rate (ASR)，越低越好。
- **合规基准**：XS-Test、OR-Bench，指标为 Over-refusal Rate，越低越好。
- **质量基准**：AlpacaEval、MT-Bench、ArenaHard（LLM-as-a-judge，1-5 分）；以及 ARC、MMLU 事实知识；NoveltyBench（多样性与 Utility）。
- **关键数字**：以 Llama-3.1-8B 为例，Suan 在 MMLU 上达 57.11 vs DPO 52.30，在 ArenaHard 上 Suan 4.21 vs DPO 3.77 vs SafeDPO 2.50；XS-Test 过拒绝 Suan 6.2 vs DPO 20.2 vs SafeDPO 20.1；OR-Bench Suan 3.6 vs DPO 29.9 vs SafeDPO 21.9。安全 ASR 方面，Suan 在多数基线中保持低值（如 AdvBench Suan 8.5 vs DPO 3.1 但 SafeDPO 1.4 更优，综合品质 Suan 更优）。
- **消融**：τ=1、β=0.1 为最优组合；DPO 在 β=0.01 时出现严重质量崩塌（ASR 失去意义），而 Suan 的 β 单调控制参考距离，性能稳定。
- **最强结果**：Suan 在综合三维（安全性、合规性、帮助性）上均达到最优或接近最优，且在 ARC/MMLU 事实保留上优于 DPO/SafeDPO，避免知识遗忘；多样性（Distinct/Utility on NoveltyBench）亦保持高水平。

## 相关工作脉络
- **DPO**（Rafailov et al., 2023）：本文将其作为最直接基线，指出 DPO 的梯度对低/高概率样本对一视同仁，且隐含正则与真实 KL 不等价，导致似然位移。
- **IPO**（Azar et al., 2023）：通过固定间隔目标缓解 DPO 过拟合，本文显示 IPO 过拒绝控制较好但安全增益有限，Suan 在安全与质量的综合权衡上更优。
- **SafeDPO**（Kim et al., 2025）：引入安全标签进行动态过滤与损失修正，本文认为其依赖额外标注且触发严重过拒绝，Suan 在无过滤条件下实现相近安全效果。
- **General DAA 统一框架**（Tang et al., 2024）：本文在此基础上进一步指出该类方法共有的梯度结构性缺陷（likelihood displacement），Suan 从根源上重新设计梯度。
- **Likelihood displacement**（Razin et al., 2025）：本文实验验证并引用该理论分析，Suan 的正则项明确用于抵消此类偏移。
- **RLHF/GRPO 等在线方法**：作为离线 DPO 类方法的上界参照，Suan 在不依赖 reward model 和在线采样的前提下达到可比性能。

## 局限性与未来方向
- **基线覆盖有限**：主要对比 DPO/IPO/SafeDPO，未与更多近期偏好优化方法及主流 RL 框架（如 GRPO、PPO）系统比较。
- **对抗鲁棒性待验证**：虽在多红队基准上表现良好，但对多样化演进型 jailbreak 攻击的鲁棒性未做深入评估。
- **大模型可扩展性未验证**：实验集中在 7B-12B 规模，对于更大参数规模或 Long-context Large Reasoning Models（LRMs）的表现尚不清楚。
- **数据集依赖**：主实验使用 PKU-Safe-RLHF，用 HH-RLHF 训练时出现轻微性能下降，需进一步探索跨数据集泛化。

## 研究启发与可借鉴点
- **梯度级目标设计思路**：绕过变分推导、直接从梯度结构出发设计加权方案，可有效规避隐式正则与真实 KL 的偏差，适用于其他偏好优化场景。
- **正则项量级均衡**：将 KL 正则梯度的系数缩放至与偏好梯度同阶，避免正则主导训练，这一技巧对多目标对齐任务具有普遍参考价值。
- **单调超参数可解释性**：Suan 中 β 对参考距离的单调控制提供了比 DPO β 更直观的超参调优路径，可借鉴到其它离线对齐方法中。
- **似然位移的实证诊断**：论文通过 preferred likelihood 演化曲线直观展示各方法的偏移程度，这一可视化诊断方式可直接用于团队后续研究的方法对比。
- **无需安全标签的轻量方案**：Suan 在不依赖安全元数据过滤的前提下实现安全提升，降低了数据标注成本，适合资源受限场景。

## 关键术语表
- **Direct Preference Optimization (DPO)**：一种无需显式奖励模型、直接从偏好对数据优化策略的离线对齐方法，通过 Bradley-Terry 模型与 RLHF 的闭式等价推导得出。
- **Likelihood Displacement（似然位移）**：DPO 类等直接对齐算法在优化过程中偏好响应似然被系统性降低的现象，源于梯度结构导致概率质量向非目标方向偏移。
- **Over-refusal（过拒绝）**：模型在遭遇良性或边缘提示时过度拒绝的现象，通常由安全对齐过程中过度保守的策略偏移引起。
- **Attack Success Rate (ASR)**：红队基准中恶意提示成功诱导不安全输出的比例，用于量化模型的安全性防御能力。
- **Over-refusal Rate**：在合规基准中良性提示被模型错误拒绝的比例，用于衡量对齐后模型的过度保守程度。
- **PKU-Safe-RLHF**：包含约 27K 条安全偏好样本的数据集，每条含 prompt、正负响应及安全标签，本文用作主训练数据。
- **Suan**：本文提出的直接偏好优化算法，通过在梯度层面设计偏好项与正则项的闭式损失，实现安全、合规与帮助性的综合优化。
- **LLM-as-a-judge**：利用大语言模型对生成回答进行自动评分的评价范式，本文使用 Flow-Judge-v0.1 在 1-5 分制下评估帮助性与相关性。

## 可复现要素
- **数据集**：PKU-Safe-RLHF（CC-BY-NC-4.0）、Alpaca（CC-BY-NC-4.0）、HH-RLHF（MIT）；评估基准多为 MIT/Apache/CC-BY-NC 许可。
- **代码/权重**：论文声明代码和数据集将在发表后公开（"The code and datasets will become available upon publication"），目前尚未开源。
- **关键超参**：τ=1（偏好边际）、β=0.1（正则强度）；LoRA r=16、α=16；学习率峰值 2×10⁻⁴，warmup 50 步；batch size=2，gradient accumulation=4；4-bit NF4 量化。
- **训练平台**：单卡 NVIDIA GH200 96 GB GPU，单 epoch 训练。
