---
title: "Witnesses-Explain-Anomalies"
source: https://arxiv.org/pdf/2609.03826v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 21:27:46"
---

# 论文速读：Witnesses-Explain-Anomalies

## 一句话总结
本文提出 WAND，一种专为表格数据设计的无监督异常检测器，通过将数据投影到单位球面上的多个方向并度量其超出次高斯基线（由中位数/MAD 校准）的程度来打分，而那些触发异常的“见证方向（witness directions）”本身就是逐特征的归因解释，获取零额外查询成本，且整个打分管线可微分，可直接嵌入可学习系统中。

## 研究问题与动机
- **核心问题**：现有无监督异常检测器（Isolation Forest、LOF、ECOD、COPOD 等）仅输出单一异常分数，缺乏对“为何被标记为异常”的特征级解释。
- **事后解释的缺陷**：业界标准做法是事后附加 SHAP/LIME，需对检测器重采样数百至数千次才能近似特征贡献，计算开销巨大，且仅是对原检测器行为的黑盒近似，维度升高时质量显著退化。
- **理论空白**：主流检测器虽在污染样本上单次扫描打分，但均摊线性代价于样本量 $n$，且缺乏与异常数量相关的探测效率（probe-efficiency）理论保证；同时，可微分、带覆盖保证的原
