---
title: "Suan-Rectifying-Direct-Preference-Safety-Alignment-in-Large"
source: https://arxiv.org/pdf/2609.08634v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 03:30:44"
field: "大模型安全对齐"
keywords: ["Direct Preference Optimization", "Safety Alignment", "Large Language Models", "Over-refusal", "Likelihood Displacement", "Gradient Design"]
innovations: ["直接在梯度层面设计偏好优化目标，绕过变分推导", "引入有界正则化梯度项抑制过拒绝和似然偏移"]
benchmarks: ["Malicious Instruct", "HarmBench", "AdvBench", "SORRY-Bench", "XS-Test", "OR-Bench", "AlpacaEval", "ArenaHard", "MT-Bench", "ARC", "MMLU", "NoveltyBench"]
---

# 论文速读：Suan-Rectifying-Direct-Preference-Safety-Alignment-in-Large

## 一句话总结
本文提出 Suan，一种直接在梯度层面设计损失函数的直接对齐算法，通过显式的正则化项抑制过拒绝（over-refusal）和似然偏移（likelihood displacement），在不损害回复质量的前提下实现更优的安全对齐效果。

## 研究问题与动机
- 现有直接对齐算法（DPO、IPO、SafeDPO）在安全对齐过程中容易出现"过拒绝"问题，即模型对良性提示也拒绝回答，导致可用性严重下降。
- 现有方法的梯度设计存在"似然偏移"问题：优化目标是偏好响应的相对梯度差，可能导致模型将概率质量从目标响应上移走，进而降低输出质量。
- 传统 KL 散度正则化的替代方案（如 k₂ 估计器）在实际训练中梯度系数随 log π_θ 线性增长，容易主导训练动态，造成训练不稳定。
- 开放权重模型的安全对齐方法远未解决，学术提出的公开算法难以匹敌闭源系统的安全控制能力。

## 核心贡献（创新点）
- **梯度级直接设计**：绕过标准变分推导，直接在梯度层面设计偏好优化目标，使训练动态更可解释。
- **双项损失函数**：将训练目标分解为偏好梯度项（L_P）和正则化梯度项（L_R），前者控制偏好学习，后者稳定策略漂移。
- **抑制过拒绝与似然偏移**：通过在偏好梯度项中引入阈值 τ，降低对已正确样本的过度优化；正则化项防止策略远离参考模型。
- **跨模型验证**：在 8 种开源模型（Mistral-12B、Llama-3.1-8B、Gemma-2-9B 等）上验证，Suan 在安全性、合规性和帮助性三个维度均达到最优权衡。

## 方法详解
- **偏好梯度项**（Proposition 3.1）：
  从通用 DAA 形式出发，提取共同梯度结构，引入自定义权重机制优先处理模型表现不佳的响应对：
  ∇_θ L_P(θ) = -σ(log(π_θ(y⁻|x)/π_θ(y⁺|x)) - τ)(∇_θ log π_θ(y⁺|x) - ∇_θ log π_θ(y⁻|x))
  其中 τ > 0 控制偏好 margin，当偏好差超过 τ 时梯度权重降低，避免过优化。

- **正则化梯度项**（Proposition 3.2）：
  针对标准 k₂ 估计器梯度系数随 log π_θ 线性增长的问题，重新设计正则化项使其梯度系数有界：
  ∇_θ L_R(θ) = Σ_{y∈{y⁺,y⁻}} (σ(log(π_θ(y|x)/π_ref(y|x))) - 1/2)∇_θ log π_θ(y|x)
  该项以 sigmoid 形式约束梯度系数，防止正则化项主导训练动态。

- **完整目标函数**（Definition 3.4）：
  L_Suan(θ) = L_P(θ) + βL_R(θ)，其中 L_P 和 L_R 分别对应 closed-form 损失：
  - L_P(θ) = -log σ(log π_θ(y⁺|x) - log π_θ(y⁻|x) + τ)
  - L_R(θ) = Σ log(cosh(½ log(π_θ(y|x)/π_ref(y|x))))
  β 控制正则化强度，τ 控制偏好 margin，通过消融实验确定最优值 τ=1、β=0.1。

## 实验与结果
- **数据集**：SFT 使用 Alpaca（52K 指令-响应对）；偏好优化使用 PKU-SafeRLHF-30K（27K 安全偏好对）。
- **模型**：8 种开源模型（Mistral-12B、Falcon3-7B、Llama-3.1-8B、Gemma-2-9B、Qwen-3-8B、Yi-1.5-9B、DeepSeek-7B、OLMo-3-7B）。
- **基准**：安全性（Malicious Instruct、HarmBench、AdvBench、SORRY-Bench）；合规性（XS-Test、OR-Bench）；帮助性（AlpacaEval、MT-Bench、ArenaHard）；知识保持（ARC、MMLU）；多样性（NoveltyBench）。
- **主要结果**：
  - Suan 在所有基准上达到最优安全-效用权衡，ASR 低于 DPO/SafeDPO 但显著优于 SFT/IPO。
  - 过拒绝率：Suan 在 XS-Test 和 OR-Bench 上平均约 6-8%，而 DPO 高达 20-45%。
  - 帮助性评分：Suan 在 AlpacaEval、ArenaHard、MT-Bench 上均与 SFT 参考模型相当或更优。
  - 似然偏移分析：DPO/SafeDPO 在训练过程中 preferred 响应 likelihood 明显下降，Suan 保持稳定。
  - 知识保持：Suan 在 ARC/MMLU 上无明显性能下降，而 DPO/SafeDPO 在部分模型上出现显著退化。

## 相关工作脉络
- **DPO**（Rafailov et al., 2023）：通过闭式损失直接优化偏好，无需显式奖励模型；但存在似然偏移和过拒绝问题。
- **IPO**（Azar et al., 2023）：引入固定 margin 正则化缓解过拟合；但在安全性提升上有限。
- **SafeDPO**（Kim et al., 2025）：引入安全标签动态过滤偏好对；但会导致严重过拒绝。
- **Safe-RLHF**（Dai et al., 2023）：通过约束 RL 显式最小化不安全输出概率；计算开销大且依赖奖励模型。
- **Likelihood Displacement 研究**（Razin et al., 2025）：揭示 DAA 梯度结构导致偏好响应 likelihood 系统性下降，本文方法正是针对此问题的设计。

## 局限性与未来方向
- 仅在 open-weight 模型上验证，未与 PPO/GRPO 等在线 RL 方法对比。
- 对渐进式 jailbreak 攻击（如 Tree of Attacks、AutoDAN）的鲁棒性未充分测试。
- 未探索在大型推理模型（LRMs）上的应用，其长上下文生成更容易受到攻击。
- 超参数 τ 和 β 需逐数据集调优，缺乏自动选择机制。

## 研究启发与可借鉴点
- **梯度级设计思维**：绕过变分推导直接构造梯度，可应用于其他偏好优化场景，避免间接推导带来的次优行为。
- **有界正则化梯度**：使用 sigmoid/tanh 约束正则化项梯度系数，防止训练动态失衡，此技巧可迁移至 RLHF 的 KL 正则化设计。
- **多基准综合评价框架**：同时评估安全性（ASR）、合规性（过拒绝率）、帮助性（LLM-as-judge）、知识保持（ARC/MMLU）和多样性（NoveltyBench），为安全对齐研究提供完整评估范式。
- **消融实验设计**：对 τ 和 β 进行系统扫参并可视化非单调行为，揭示 DPO 在极端 β 值下的退化机制，为超参数调优提供参考。

## 关键术语表
- **Direct Alignment Algorithms (DAAs)**：无需显式奖励模型，直接从偏好数据优化策略的直接对齐算法家族，包括 DPO、IPO、SafeDPO 等。
- **Likelihood Displacement**：DAA 优化过程中偏好响应的对数概率系统性下降的现象，导致输出质量退化。
- **Over-refusal**：安全对齐后模型对良性提示也拒绝回答的行为，是安全与效用权衡中的核心问题。
- **Preference Margin (τ)**：控制偏好优化强度的超参数，决定多少偏好差才会触发显著梯度更新。
- **k₂ Estimator**：一种 KL 散度的采样估计器，在 DAA 理论分析中常用作正则化项，但其梯度性质在实际训练中存在问题。
- **PKU-SafeRLHF**：包含约 27K 安全偏好对的开源数据集，每对响应附带布尔安全标签。
- **LLM-as-a-judge**：使用大语言模型作为裁判对生成结果进行质量评分的自动化评估方法。

## 可复现要素
- **数据集**：Alpaca（CC-BY-NC-4.0）、PKU-SafeRLHF（CC-BY-NC-4.0）、HH-RLHF（MIT）；基准包括 Malicious Instruct、HarmBench、AdvBench、SORRY-Bench、XS-Test、OR-Bench、AlpacaEval、ArenaHard、MT-Bench、ARC、MMLU、NoveltyBench。
- **代码**：论文声明代码和权重将在发表后开源。
- **超参**：τ=1、β=0.1、学习率 2×10⁻⁴、warmup 50 步、batch size 2、gradient accumulation 4、LoRA r=16 α=16、4-bit NF4 量化、单 epoch。
