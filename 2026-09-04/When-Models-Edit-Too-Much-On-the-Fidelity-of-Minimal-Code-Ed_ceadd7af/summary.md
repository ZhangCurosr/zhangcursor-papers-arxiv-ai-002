---
title: "When-Models-Edit-Too-Much-On-the-Fidelity-of-Minimal-Code-Ed"
source: https://arxiv.org/pdf/2609.04061v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 21:27:21"
---

# 论文速读：When-Models-Edit-Too-Much-On-the-Fidelity-of-Minimal-Code-Ed

## 一句话总结
本文指出当前前沿 LLM 在代码修复任务中普遍存在“过度编辑（over-editing）”现象，即补丁功能正确但修改范围远超必要；通过构建基于 BigCodeBench 的已知最小修复真值基准，证明了编辑保真度是可独立度量与训练的维度，且可通过保留提示与强化学习显著优化。

## 研究问题与动机
- **评测维度缺失**：现有代码基准（如 HumanEvalFix、CanItEdit、SWE-bench）主要报告 Pass@1/Pass@k 或测试通过率，无法区分“正确修复”与“必要修复”，导致高冗余补丁被误认为高质量输出。
- **工程维护成本**：在 brownfield 维护场景中，冗余重写会增大 diff churn、增加代码审查负担、引入额外认知复杂度，并可能掩盖测试覆盖之外的回归风险。
- **默认行为策略错位**：模型倾向于将修复任务理解为“生产健壮代码”而非“执行局部修补”，即使强模型（如 GPT-5.5）也能以 0.823 的 Pass@1 附带 0.299 的超额编辑距离。
- **规模与推理的边界**：强化推理与模型放大并不单调改善编辑保真度，说明过度编辑是提示/训练分布层面的策略偏好，而非单纯的能力上限。

## 核心贡献（创新点）
- **提出编辑保真度作为独立评估轴并构建已知最小修复真值基准**；与以往仅依赖测试通过率或自由编辑指令的基准不同，本文通过逆向注入 AST 级干扰使最小补丁成为可计算对齐目标，将“过度编辑”形式化为可量化的 $E_{Lev}$ 与 Added CC。
- **系统量化前沿 LLM 的过度编辑行为并揭示其策略错位本质**；与多数工作将大补丁视为正常生成交叉不同，本文首次在同一基准上证明高 Pass@1 可与高冗余编辑共存，且推理预算、模型规模均不能单调消除该现象。
- **证明最小编辑偏好可通过强化学习稳定习得并在域外泛化**；与 SFT/DPO 易过拟合训练干扰族或牺牲正确率不同，本文的 GRPO-style 组合奖励在保持 LiveCodeBench 通用能力（+0.6%）的同时显著降低 OOD 超额编辑距离。
- **提出细粒度过度编辑分类法并验证其跨模型一致性**；与以往仅用行数、token 数或 CodeBLEU 衡量补丁大小不同，本文归纳出防御性泛化、数据流重写、契约漂移等模式，揭示模型常因“粒度错位”而非单纯风格差异进行过度重写。

## 方法详解
- **基准构建**：从 BigCodeBench 采样 400 个 Python 函数任务，对参考解注入 1~2 个预定义 AST 级干扰（共 568 处，涵盖比较运算符、边界条件、切片索引、排序顺序等），仅保留干扰后测试失败样例；逆向干扰即为已知最小修复补丁（gold repair），平均仅 1.63 个 token 编辑、91.8% 不超过 2 个 token。
- **评估指标**：
  - `Pass@1`：修复后通过所有测试的比例。
  - `Excess Levenshtein distance`：$E_{Lev}(M) = d(M, C) - d(G, C)$，其中 $C$ 为破损代码、$G$ 为黄金补丁、$M$ 为模型输出，$d(\cdot,\cdot)$ 为去除注释后的归一化 token 级 Levenshtein 距离。
  - `Added Cognitive Complexity`：修复后代码与黄金补丁的 SonarSource 认知复杂度差值，捕获分支/嵌套等结构性开销。
- **提示干预**：通用提示末尾追加保留子句 “...but keep as much of the original code as possible”，保持系统提示、任务描述与测试套件不变，仅改变模型的任务框架（task-framing）。
- **后训练奖励设计**：使用 DeepCoder 数据（每样本 1~10 处干扰，共 4,141 例）进行 SFT/rSFT/DPO/RL 对比。RL 采用 GRPO-style 组相对优势估计，奖励函数为：
  - 失败/不可解析：$r(M) = -0.2$
  - 通过：$r(M) = \lambda_{exec} - \lambda_{edit} \cdot e(M)$，其中 $e(M) = d(M,C) - d(G,C)$，$\lambda_{exec}=0.1$、$\lambda_{edit}=1.0$。
  - 该奖励在保障执行正确的同时线性惩罚超额编辑量，使 $e(M)>0.3$ 的补丁收益低于部分错误补丁，强制模型聚焦局部修复。

## 实验与结果
- **数据集与基线**：评估集为 400 个 BigCodeBench 任务；训练集为 4,141 个 DeepCoder 样本；OOD 验证使用 20 种保留干扰族 + Defects4J Java 真实 Bug。基线覆盖 20+ 前沿闭源/开源码模型（GPT-5.5/5.4、Claude Opus/Sonnet 4.x、Gemini、DeepSeek V3.2、GLM-5、Qwen3-Coder 等）及 SFT/rSFT/DPO/RL 训练变体。
- **前沿模型表现**：泛化提示下平均超额 Levenshtein 距离从 0.195 降至 0.131，新增认知复杂度下降 26.6%（$p < 10^{-4}$），Pass@1 提升 2.3 个百分点（paired bootstrap 95% CI [+1.49, +3.05]）。最强非推理模型 Claude Opus 4.7 在通用提示下已实现 Pass@1 0.918、Excess Lev 0.070、Added CC 0.084。
- **RL 训练最优结果**：OOD 评估中 RL 达到 Pass@1 0.782、Excess Lev 0.050、Added CC 0.185，显著优于 SFT（OOD Pass@1 暴跌至 0.458）与 DPO；LiveCodeBench v6 从 32.6% 升至 33.2%（+0.6%），是唯一未退化通用编码能力的训练方法。
- **最强结果与提升幅度**：相比未训练基线（OOD Excess Lev 0.089），全参数 RL 将超额距离压缩近 44%（0.050），且 LoRA rank=64 可近乎追平（0.051），证明最小编辑偏好可由小参数适配器捕获。
- **过度编辑模式**：对 530 个高冗余补丁标注显示，64.2% 为防御性泛化（添加宽泛校验/类型转换），63.2% 为数据流重写（整体重算任务），34.7% 为契约漂移，切片与列表索引类干扰触发最大冗余（Excess Lev 0.353）。

## 相关工作脉络
- **CanItEdit / EDIT-Bench / CodeEditorBench**：关注自然语言指令驱动的代码编辑能力，但未以“已知最小修复”为锚点量化补丁冗余度，无法分离功能正确性与编辑忠实度。
- **CREF / AdaPatcher / PAFT / PRepair**：显式追求更小或更忠实补丁，多依赖运行时定位器或偏好学习；本文在相同
