---
title: "Prior-Audit-Repair-Context-Shifts-LLM-Verifier-Thresholds-To"
source: https://arxiv.org/pdf/2608.16003v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 04:29:26"
---

# 论文速读：Prior-Audit-Repair-Context-Shifts-LLM-Verifier-Thresholds-To

## 一句话总结
本文揭示了在 LLM 自动化检查流水线中，将已完成的“审计→修复”上下文提前注入核查器，会显著降低其对正确推理轨迹的误报率（最高下降 11.5 pp）。该效应方向与既有极性漂移理论预测相反，信号检测论分析证实变化源于判定标准(c)的移动而非判别力(d')的提升，且被消除的误报中 82% 实为虚假判定，说明该阈值偏移在当前运行点具有净收益。

## 研究问题与动机
- **核心问题**：当语言模型担任核查器(checker)且下游连接修复器(fixer)时，这种流水线布线方式本身是否会静默改变核查器的判断输出？
- **现有研究混淆变量**：Jin & Chen (2026) 发现要求解释与修复的提示格式会增加误判，但“修复”与“审计”绑定在同一响应中，效应与输出格式混杂；Khullar et al. (2026) 发现的自我宽容偏差依赖作品位置而非明确标注，二者均未隔离出“已完成 repair episode”的独立影响。
- **理论预测冲突**：Temkit (2026) 的累积消息漂移(polarity drift)理论指出负面历史会引发更强的判断偏移，据此应预测“报告错误的上下文提高误报率”，但本文预期并验证了相反方向。
- **评估维度单一**：既有 LLM-as-judge 与 CriticBench/CriticEval 工作多关注 critique 与 correction 的质量，缺乏对“已完成的 repair 经历如何反作用于后续 critique 阈值”的上游追问。

## 核心贡献（创新点）
1. **首次因果分离并量化“审计→修复”上下文对核查器误报率的独立效应。** 通过 byte-identical 任务设定与完整对照矩阵，证明该效应是流水线拓扑的隐性产物，而非输出格式、归属标注或位置安排的副产品。
2. **发现反向极性漂移：报告错误的审计上下文反而进一步降低误报率。** 打破了 Temkit (2026) 所建立的“负面历史→更严格判定”的直觉框架，证明 pipeline 角色经历比单纯的情绪极性更能主导阈值移动。
3. **利用信号检测理论定位变化机制为判定标准(c)移动，而非判别力(d')提升。** 系统联合报告 FAR 与 Detection Rate，证明效应完全由阈值右移（更不愿 flag）驱动，并在 15/15 组合中经受多重校正。
4. **提供人工审计证据表明该阈值移动在当前运行点具有净收益。** 基线误报中 82% 为虚构/错误判定，修复请求消除的主要是算术与代数类幻觉 flag，说明宽容偏移在实际部署中未必有害。

## 方法详解
- **数据集与任务划分**：使用 ProcessBench 数学推理轨迹。从 1,101 条 human-verified-correct 轨迹中划分出 929 条 clean targets，并构造等量匹配的 929 条 labelled-incorrect 轨迹用于计算检测率。两臂严格 disjoint 且按 source (GSM8K, MATH, Olympiad-Bench, OmniMath) stratified 匹配。
- **2×2 因子与五重对照设计**：
  - 主操纵：在 target prompt 前注入一个在另一 item 上冻结于 T=0 的“审计→修复”episode。
  - 2×2 因子：归属 (AS/US = self vs AO/UO = peer) × 角色位置 (assistant turn vs user turn)，四项均长度匹配且脱标签后文本 byte-identical。
  - 关键对照：`AF`（非审计等长 filler，隔离“是否有上下文”）；`AV`（仅保留审计，删除修复）；`AN`（inert continuation，修复请求替换为重述结论）；`AX`（在错误 trace 上生成真实修复）；`AXN`（AX 的 inert 版本）。
- **评估与统计约定**：
  - 主指标 FAR 在 T=0.7 下每 item 采样 8 次，避免 T=0 argmax 硬化导致概率性阈值移动不可见。
  - 信号检测分解：计算 $c = -0.5[z_F + z_H]$（c 越大表示越保守/宽容）与 $d'$；因 clean 与 incorrect 臂独立重采样，$\text{SE}(\Delta d')$ 恰好为 $\text{SE}(\Delta c)$ 的两倍，故 d' 检验敏感度天然减半。
  - 统计推断：20,000 次 cluster bootstrap（以 episode 为聚类单元，处理 episode 重复使用带来的依赖），Holm–Bonferroni 家族校正；p-value 与 95% CI 联动报告。

## 实验与结果
- **数据集与基线**：ProcessBench；基线包含 R0（无上下文）、AF（长度匹配 filler）、AV、AN、AX、AXN，覆盖三种模型（Qwen3.6-27B, Qwen3.6-35B-A3B, Ministral-3-14B）与五种语义等价 wording (F1–F5)。
- **主效应**：相对于 AF，episode 使 FAR 下降 **2.8 ~ 11.5 pp**，在 15/15 model×wording 组合中全部显著，相对基线降低 **9%~25%**（Table 1）。AF − R0 在两个 Qwen 模型上为 null，证明效应来自“审计内容”而非“任意前序上下文”。
- **极性漂移证伪（Table 2）**：Ministral 上 `AX - AS = -5.62 pp`（5/5 wording 全部显著），即“模型已发现并修复错误”的上下文比“未发现错误”的上下文更宽容，方向与极性漂移预测完全相反。Qwen3.6-27B 因 37/50 修复仍输出 correct 导致该对照退化；35B-A3B 仅在 F1 显著。
- **组件分解**：修复内容 (AX-AXN) 在 Qwen 系列主导；错误 verdict 上下文 (AXN-AN) 在 Ministral 主导（最大 -8.96 pp）；修复请求 (AS-AN) 在三模型均有存活。无单一通用组件，不同模型由不同部分驱动。
- **信号检测与收益（Figure 2 / §6）**：$\Delta c$ 在 15/15 组合同向移动且 13/15 survive correction；$\Delta d'$ 0/15 survive（考虑到检验敏感度减半，实际估计多为正偏）。Balanced accuracy 在 15/15 组合上升，6/15 显著
