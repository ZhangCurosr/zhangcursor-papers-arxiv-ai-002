---
title: "UniDot-A-Unified-Network-for-Sequence-Modeling-and-Feature-I"
source: https://arxiv.org/pdf/2608.16797v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 06:59:08"
---

# 论文速读：UniDot-A-Unified-Network-for-Sequence-Modeling-and-Feature-I

## 一句话总结
本文提出 UniDot，一种面向大规模推荐点击后转化率（CVR）预测的统一架构，将特征交互模型与序列建模模型通过因子分解机（FM）内积这一统一原语融合进同一可堆叠宏块中，凭借双总线并行、FM Highway 显式路由及多路径互学习训练策略，在 TAAC × KDD Cup 2026 工业赛道取得 AUC 0.83217 的亚军成绩。

## 研究问题与动机
1. **双轨演进的架构割裂**：工业推荐系统长期独立发展两类模型——基于静态多维特征（用户画像×物品属性）的特征交互模型（如 DeepFM、DCN-v2、Wukong）与基于用户行为历史的序列建模模型（如 DIN、DIEN、TWIN），线上系统往往仅做松散耦合。
2. **协同过滤内积被隐式化导致泛化弱化**：现有统一工作多将用户-物品协同信号作为深层交互堆叠的涌现属性；而在广告转化场景中，推理时绝大多数候选对用户-物品对均为训练未见的新组合，隐式内积难以充分保证对新组合的泛化能力。
3. **多域行为序列的高效统一处理缺失**：不同行为域（seq_a~seq_d）的序列长度与字段分布各异，缺乏一个共享嵌入、一次编码、按需提取的管道，且
