---
title: "REOPD-Reliability-Adaptive-Reward-Extrapolation-for-On-Polic"
source: https://arxiv.org/pdf/2608.11698v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 02:35:59"
---

# 论文速读：REOPD: Reliability-Adaptive Reward Extrapolation for On-Policy Distillation

## 一句话总结
本文提出 REOPD，通过 token 级兼容性权重与 micro-batch 级有界自适应预算的协同，对 on-policy 蒸馏中的教师-参考 log-ratio 残差进行细粒度、可靠性适配的外推；该方法无需额外验证器或 rollout，在数学、代码及多教师混合蒸馏任务上均达到或超越固定系数 ExOPD/G-OPD。

## 研究问题与动机
1. **固定全局系数导致奖励黑客与训练不稳定**：ExOPD 使用单一标量 λ 统一放大教师-参考 log-ratio 残差，易过度放大极端峰值，引发 reward hacking 与优化震荡。
2. **域间最优 λ 差异大且调参成本高昂**：数学任务偏好 λ≈1.25，代码任务偏好 λ≈1.5，每次切换任务需全量训练+评估扫描，仍可能选错。
3. **现有自适应方法改动了对齐路径**：AdaKD、TIP、Prune-OPD 等修改教师对齐项或 rollout 过程，而本文希望保留标准 OPD 对齐信号，仅对“超越教师的残差部分”进行控制。
4. **缺乏纯白盒的在线残差控制机制**：依赖 verifier 或 outcome label 的方法（SCOPE、SG-OPD、Reward-gated OPD）需要额外监督；本文旨在仅利用白盒 OPD 已计算的 log-probability 实现自适应。

## 核心贡献（创新点）
1. **将固定系数外推重新形式化为残差控制问题**，明确指出均匀缩放的风险，并提出“保留对齐、只控残差”的目标解耦思路。
2. **提出 REOPD 双层自适应框架**：token 级兼容度权重 $q_{b,i,t}$ 过滤不可靠残差，micro-batch 级有界预算 $\gamma_b$ 动态调控整体外推强度，合成有效系数 $\lambda_{b,i,t}=1+\gamma_b q_{b,i,t}$。
3. **设计零额外开销的在线控制器**：仅依赖学生、教师、参考模型的 log
