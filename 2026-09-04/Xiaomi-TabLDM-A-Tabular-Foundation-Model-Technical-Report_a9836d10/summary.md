---
title: "Xiaomi-TabLDM-A-Tabular-Foundation-Model-Technical-Report"
source: https://arxiv.org/pdf/2609.03880v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 21:27:49"
---

# 论文速读：Xiaomi-TabLDM-A-Tabular-Foundation-Model-Technical-Report

## 一句话总结
本文提出 Xiaomi-TabLDM，一种基于上下文学习（ICL）的表格基础模型，完全依赖结构因果模型（SCM）合成数据预训练，无需任务特定微调即可在分类与回归任务上实现高精度预测，并通过双流特征分组、稀疏 MoE 与测试时计算扩展，在多项基准上取得领先性能与优异的能效比。

## 研究问题与动机
1. 传统表格预测长期依赖数据集特定的监督训练与超参调优（如 XGBoost、CatBoost 与 AutoGluon），缺乏跨任务的可迁移能力。
2. 现有 ICL 表格基础模型（如 TabPFN 系列、TabICLv2 等）在模型容量扩展、推理效率与回归任务一致性上仍存在瓶颈，难以同时兼顾高精度与低计算成本。
3. 需要一种统一框架，既能通过大规模合成先验学习丰富的特征交互，又能通过架构设计与测试时计算策略突破单一前向推理的性能上限。

## 核心贡献（创新点）
1. 提出基于上下文学习的表格基础模型 Xiaomi-TabLDM，无需微调即可完成分类与回归；与早期仅依赖单次前向推理的 ICL 模型相比，本文进一步引入测试时计算扩展机制，使推理性能可随额外算力持续提升。
2. 设计双流特征分组与轻量级 AttnRes 残差连接，打破特征排列对称性并改善深层梯度传播；与 TabICLv2 等使用固定分组与标准加法残差的方案相比，本文机制更灵活且能自适应聚合多尺度历史表征。
3. 在 ICL 预测层引入稀疏混合专家（Sparse MoE）并配合 load-balance 与 router z-loss 正则化；与全密度 FFN 或单纯堆叠模型规模的方法相比，该设计实现总参数容量的线性扩展而单次推理计算保持恒定。
4. 在四个主流基准（TALENT、TabArena、BCCO、OpenML-CTR23）上系统验证，尤其回归任务取得第一或第二名；与依赖真实数据继续预训练或重度调参的基线（如 TabFM、AutoGluon）相比，本文方法以显著更低的训练与推理耗时逼近最强性能。

## 方法详解
- **整体流程**：将输入表格分为列级嵌入、行级聚合、ICL 预测三个阶段，直接以带标签样本作为上下文进行单次或多次前向推理，无需更新模型参数。
- **双流特征分组（Dual-stream Feature Grouping）**：每个特征列与经循环移位选出的另外两列组成三元组，共享线性投影 $E[i,j] = \text{Linear}(x_{i,(j+\delta_1)\bmod m}, x_{i,(j+\delta_2)\bmod m}, x_{i,(j+\delta_3)\bmod m})$；第一流采用固定二进偏移 $(1,2,4)$，第二流采用宽度自适应几何偏移 $(1,\lfloor\sqrt{s}\rfloor,s)$，两流输出相加，有效打破特征排列对称性并覆盖不同尺度的列间交互。
- **列级 Set Transformer 编码**：每组三元组通过诱导点注意力（inducing-point attention）以小批量 learned inducing tokens 聚合行维度信息，生成列嵌入；注意力使用 query-aware scalable softmax (QASSMax) 以提升长上下文泛化。
- **轻量级 AttnRes 残差连接**
