---
title: "TailSFT-Filtered-Fine-Tuning-Improves-Post-Training-Performa"
source: https://arxiv.org/pdf/2608.25756v1.pdf
model: agnes-2.5-flash
chunks: 3
summarized_at: "2026-09-01 16:15:36"
---

# 论文速读：TailSFT-Filtered-Fine-Tuning-Improves-Post-Training-Performa

## 一句话总结
TailSFT 提出在监督微调（SFT）阶段通过在线计算相对初始策略的损失下降量，动态 mask 已充分拟合的序列，从而在不改动两阶段 SFT+RL 流程的前提下重塑梯度方向，显式提升专家响应的覆盖率并显著改善后续强化学习阶段的最终性能。

## 研究问题与动机
- **SFT 平等优化导致覆盖率瓶颈**：传统 SFT 对所有样本一视同仁，模型易过度拟合已掌握序列，浪费算力且压缩后续 RL 阶段的探索空间。
- **现有数据选择方法依赖额外假设**：重要性采样、动态剪枝、influence-based 筛选等需目标样本、梯度计算或辅助损失，多面向 pretrain 或隐式 RL 设计，实施复杂。
- **SFT 局部指标与 post-RL 表现脱节**：高 SFT 准确率无法预测后训练效果，局部优化反而可能降低 pass@large-k 与 held-out 泛化能力。
- **缺乏 RL 前的显式覆盖率干预**：现有 RL/Sharpening 理论侧重已覆盖行为内的质量重分配，缺少在 SFT 阶段主动扩展“可用响应集合”的机制。

## 核心贡献（创新点）
1. **序列级动态损失屏蔽机制**：仅依赖 base/reference policy 的概率估计在线 mask 已改善样本，无需目标样本、influence 矩阵或辅助正则项，实现文档/序列粒度的低成本过滤。
2. **相对裁剪损失（OFF）的理论形式化**：在 expert conditional setting 下严格对比 ERM、绝对裁剪（ABS）与相对裁剪（OFF），证明 OFF 能更优地逼近目标分布并保障相对梯度方向。
3. **即插即用的 post-RL 性能对齐**：不改变两阶段 pipeline 与任何 RL 算法，仅修改 SFT 阶段哪些样本接收梯度，即可无缝衔接现有 on-policy RL 流程。
4. **代理指标驱动的因果验证范式**：以 held-out generalization loss 与 pass@large-k 替代原始 SFT 分数评估 SFT 干预效果，并与完整 RL run 匹配验证覆盖率增益的真实性。

## 方法详解
- **专家条件设定**：expert policy 建模为 $\pi^\star(y) \propto \pi_{\text{ref}}(y) \cdot \mathbf{1}\{y \in S\}$，与 $\beta \to 0$ 极限下的 RLHF 形式等价。优化目标为最小化前向 KL 散度 $D_{KL}(\pi \| \pi_{\text{ref}})$。
- **三种对比 Loss**：
  - **ERM**：$L_{\text{ERM}}(\pi) = \frac{1}{n}\sum_i -\log\pi(y_i)$，标准全量优化。
  - **ABS**：$L_{\text{ABS},\alpha}(\pi) = \frac{1}{n}\sum_i \max(-\log\pi(y_i) + \log\alpha,\, 0)$，忽略 $\pi(y_i) \geq \alpha$ 的高置信样本。
  - **OFF（TailSFT 核心）**：$L_{\text{OFF},\beta}(\pi) = \frac{1}{n}\sum_i \max(-\log\pi(y_i) + \log(\beta \cdot \pi_{\text{ref}}(y_i)),\, 0)$，忽略 $\pi(y_i) \geq \beta\pi_{\text{ref}}(y_i)$ 的样本，以相对参考概率为阈值。
- **在线屏蔽流程**：训练过程中持续比较当前策略 $\pi$ 与初始/base 策略 $\pi_{\text{ref}}$ 在各序列上的 loss；当某序列 loss 下降至阈值以下（表明已充分拟合），则该序列后续迭代不参与梯度计算，迫使优化器将预算分配给尚未覆盖的 expert 响应。
- **无辅助依赖**：无需 verifier、mode labels、熵正则或 self-distillation，仅依靠监督数据与 reference policy 实现梯度重塑。

## 实验与结果
- **实验设置**：论文在本段主要提供理论分析与相关工作定位，具体数据集名称、超参数网格与基线数值结果需见原文实验部分；实证部分通过匹配完整 RL run 验证 SFT 干预对 post-RL 性能的提升。
- **基线覆盖**：系统对比重要性采样/动态剪枝、RHO-LOSS/Dataset Cartography、Rho-1/CoLoR/Token Cleaning/GREATS/LESS、Confidence Caps/Entropy Regularization/Self-Distillation/Token-level Clipping/Prefix-conditioned SFT，以及 Bansal/Fu/Huang/Zhang 等 SFT-RL 耦合方案。
- **核心结论**：TailSFT 在保持标准两阶段流程不变的前提下，以相对裁剪（OFF）替代绝对裁剪或全量 ERM，显著提升下游 RL 后的模型表现；理论证明 OFF 在逼近 expert conditional distribution 时具备梯度方向优势；held-out generalization loss 与 pass@large-k 被证实为比原始 SFT 分数更可靠的代理指标。具体提升幅度与基准测试集编号详见原文实验表格。

## 相关工作脉络
1. **数据选择与多样性保持**（Katharopoulos & Fleuret 2018; Qin et al. 2024; Mindermann et al. 2022; Lin et al. 2024 等）：prior 方法多依赖梯度/influence 或目标样本，面向 pretrain 或隐式 RL；TailSFT 在序列粒度在线 mask，直接面向 post-RL 覆盖率。
2. **SFT 与 RL 耦合策略**（Bansal et al. 2026; Fu et al. 2026; Huang et al. 2026; Zhang et al. 2026b; Yoshihara et al. 2025; Niu et al. 2026）：现有工作或混合中间 checkpoint、联合加权演示、改变 RL 辅助目标；TailSFT 保持顺序阶段不变，仅过滤 SFT 梯度。
3. **覆盖率与测试时计算**（Chen et al. 2026a; Brown et al. 2024; Snell et al. 2025 等）：形式化覆盖率或依赖 verifier/mode labels/rollout 预算分配；TailSFT 将覆盖率操作化为在线 SFT 规则，无需测试时扩展。
4. **RLVR、锐化与探索**（Yue et al. 2025; Wu & Choi 2025; Wen et al. 2026; Huang et al. 2025a）：侧重 RL 阶段的质量提升或奖励激发；TailSFT 在 RL 前通过监督数据预优化覆盖率，与 RL 探索方法正交互补。
5. **分阶段模型开发诊断**（Kang et al. 2026; Liu et al. 2023; Zeng et al. 2025; Springer et al. 2025; Chu et al. 2025）：指出 SFT 分数/pretrain perplexity 无法预测 post-RL 表现；TailSFT 提供可操作的 SFT 干预并以匹配 RL run 验证有效性。

## 局限性与未来方向
- **Reference
