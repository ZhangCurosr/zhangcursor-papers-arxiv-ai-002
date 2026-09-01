---
title: "One-note-in-three-a-verified-census-of-three-deployed-AI-scr"
source: https://arxiv.org/pdf/2608.31017v1.pdf
model: agnes-2.5-flash
chunks: 4
summarized_at: "2026-09-01 23:41:59"
---

# 论文速读：One-note-in-three-a-verified-census-of-three-deployed-AI-scr

## 一句话总结
本文对三类已部署的AI临床速记模型开展大规模审计，通过双族质疑面板与四维评审标准对照实验，量化了审查尺度对质量评估的支配性影响，最终提炼出618条经严格验证的临床速记错误发现及结构化分类体系。

## 研究问题与动机
- **核心问题**：已部署AI临床速记工具在实际问诊场景中产生哪些系统性错误？其“错误率”指标受评审标准与审查流程的影响程度有多大？
- **现有方法不足**：已发表的published audit研究在错误率估算上差异高达8.5倍，主要源于评审仪器（instrument）与审查标准不一致，而非速记产品本身差异；单一模型或宽松标准易产生自我宽容（self-leniency）偏差，导致错误被系统性低估。
- **研究动机**：构建一套可复现、多维度、抗偏见的审计框架，显式剥离“速记产品质量”与“评审标准噪声”，并建立结构化的错误分类与严重度评级体系，为医疗AI安全评估提供可比基线。

## 核心贡献（创新点）
- **双族独立质疑面板设计**：引入来自不同模型族的两名skeptic进行交叉反驳，本质区别在于通过跨族独立验证排除单一模型的体系性盲区与自我宽容。
- **评审标准对照实验量化仪器效应**：在同一候选池上并行测试宽松/严格标准，揭示评审尺度差异可导致验证错误率相差8.45倍，填补了审计研究中“标准-结果”耦合关系的空白。
- **Salience/Severity双轨临床错误分级**：将重要性重评（Salience）与临床后果量表（Severity）解耦并联合应用于survivors，本质区别在于从“类型标签统计”升级为“风险加权评估”。
- **可复用的聚类错误谱系**：从618条验证发现中提炼17个机制聚类（含Tier 1/Omission+Tier 2/Mechanism双层框架），为后续自动检测与防御模板提供结构化先验。

## 方法详解
- **候选人发现流水线**：原始候选池13,678条，经发现模型（discovery model）的importance filter初筛，保留5,898条high+medium重要性候选。
- **双审稿人质疑面板**：两位skeptic均来自不同模型族，获完整note+transcript，指令为“只要可反驳就反驳”。Harsher skeptic为`anthropic/claude-opus-5`（temperature=1.0，6000-token上限），反驳率90.6%；Gentler skeptic为`openai/gpt-5.5`，反驳率81.3%。分歧提交Tiebreak仲裁。
- **裁决聚合逻辑**：双审稿人阶段产生4,663次一致反驳、812次分歧、423次一致保留；Tiebreak（`openai/gpt-5.4`，高推理）裁决中195次维持发现、617次否决，最终落地618条已验证发现。两族独立一致率86.2%，Tiebreak偏向更严格方（531:281）。
- **双维评级体系**：
  - **Salience**：评审小组对候选发现的重要性重评（high/medium/low），决定进入严重度评级的候选。
  - **Severity**：书面临床评分量表，仅用于survivors；critical（可能改变临床行动/安全性）、supporting（降质记录但不改变后续）、peripheral（无实际后果）。
- **聚类与分层架构**：618条发现经密度分组得17个聚类（563条）+55条未分配。固定顶层两级：Tier 1分为omission/addition/wrong output/irrelevant四大类；Tier 2为机制层。16个聚类归入Tier-2，1个（retracted-device group）单独报告。
- **审查标准对照实验**：在同一批1,295条候选上测试四组配置：Harsher alone、Gentler alone、完整面板+tiebreak、同模型lenient baseline，报告验证比例、Wilson 95%置信区间与聚类bootstrap 95%置信区间。

## 实验与结果
- **数据集**：三类已部署AI临床速记模型的门诊问诊记录（原文未披露具体数据集名称与公开状态）。
- **评估基线**：四组评审标准对照（Harsher alone / Gentler alone / 完整面板+tiebreak / 同模型lenient）。
- **主要结果数字**：
  - 最终验证发现：618条（17聚类563条 + 55未分配）。
  - 严重度分布：476 critical / 140 supporting / 1 peripheral（共617条有评级）。
  - 高频失败模式：过敏状态与用药清单遗漏（111条）、伪造患者身份（93条）、工作诊断被删除（53条）、相对时间转伪造日历日期（44条）、远程问诊史记为客观检查（42条）等。
  - 可计率聚类（跨≥10会诊）：伪造身份(34)、过敏/用药遗漏(33)、工作诊断删除(22)、伪造日期(17)；其余4个聚类因基于极少量问诊，仅作失败模式列举。
  - 重要性重评显著移动：Discovery称1,211高/4,687中；Panel重评为2,014高/2,377中/1,507低。按salience存活率：high 15.7% / medium 8.1% / low 7.2%（约2倍差距）。
  - 审查标准对照（1,295候选池）：同模型lenient验证率79.00%（Wilson 95% CI [76.69, 81.13]）；Harsher alone 9.34%，Gentler alone 17.76%，完整面板+tiebreak 10.35%。Strict vs Lenient比率8.45x。
-
