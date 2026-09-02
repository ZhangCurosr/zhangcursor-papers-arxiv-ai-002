---
title: "One-Policy-Any-Budget-Internalizing-Budget-Aware-Search-via"
source: https://arxiv.org/pdf/2609.00813v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 19:26:37"
field: "LLM Agentic Search"
keywords: ["budget-aware search", "reinforcement learning", "LLM agents", "tool use", "curriculum learning", "search efficiency"]
innovations: ["Progressive scaffold removal for budget-aware search internalization", "Sliding-window adaptive budget sampling for multi-budget training", "Correctness-gated composite reward with adaptive efficiency weighting"]
benchmarks: ["NQ", "TriviaQA", "PopQA", "HotpotQA", "2Wiki-MultiHopQA", "MuSiQue", "Bamboogle"]
---

# 论文速读：One-Policy-Any-Budget: Internalizing Budget-Aware Search via Reinforcement Learning

## 一句话总结
论文提出 AnySearch，一种通过强化学习将预算感知搜索能力内部化的框架，使单个策略能够在任意预算约束下自适应调整搜索行为，无需在部署时重新训练或切换模型。

## 研究问题与动机
- **固定预算训练的策略无法适应变化约束**：现有 RL-based search agents 主要在固定预算下训练，部署时预算变化则策略失效。
- **实际场景预算高度多变**：从延迟敏感应用（极少搜索调用）到深度研究任务（允许大量探索），单一策略需覆盖全预算范围。
- **外部依赖导致训练-推理差距**：BATS 等方法依赖外部预算追踪器，预算感知能力未内化到策略本身，推理时需额外模块配合。
- **简单截断搜索调用不足以解决问题**：agent 需要学会何时搜索、何时推理，有效分配有限预算而非机械地控制调用次数。

## 核心贡献（创新点）
1. **训练脚手架渐进移除机制**：通过 Phase I 显式预算状态注入 + 结构化推理提示引导策略学习预算感知模式，Phase II 完全移除脚手架使其内化为策略能力，消除推理时的外部依赖。
2. **滑动窗口自适应预算采样**：Phase II 根据各预算水平近期准确率动态调整采样分布，聚焦弱项同时保留均匀覆盖，解决多预算水平训练的采样效率问题。
3. **复合奖励的自适应效率权重**：基于每组查询准确率动态调整效率信号强度，高准确率查询强化效率优化、低准确率查询优先保证正确性，避免硬约束下的性能退化。
4. **单策略泛化至未见预算**：在训练预算范围（1-5）外（6-8）仍保持单调递增的性能曲线，证明策略真正学会了预算比例分配而非记忆固定策略。

## 方法详解
**任务形式化**：将预算感知搜索建模为资源受限序列决策问题，agent 每步观测状态 $s_t = (h_t, b_t)$（上下文 + 剩余预算），执行 Reason/Search/Answer 动作，预算约束 $\sum \mathbb{I}(a_t \in \mathcal{A}_{tool}) \leq B$。

**训练脚手架（Phase I）**：
- 预算状态注入：每轮输入 `<budget>remaining=R; used=U; total=T</budget>` 使预算完全可观测
- 结构化推理提示：要求 agent 在 `<think>` 块中进行双因素分析：信息充分性评估 + 预算条件策略决策

**两阶段课程学习**：
- Phase I（前 100 步）：预算从 $B_{max}=5$ 线性衰减至 1，每预算级别 20 步，脚手架激活
- Phase II（后 400 步）：移除脚手架，仅输入总预算 $B$；采用滑动窗口（$W=20$）追踪各预算级最近准确率 $\bar{R}^{(t)}(b)$

**自适应采样分布**：
$$P^{(t)}(B{=}b) = (1{-}\lambda)\cdot\frac{\bar{R}_{max}^{(t)}{-}\bar{R}^{(t)}(b){+}\epsilon}{\sum_{b'}(\bar{R}_{max}^{(t)}{-}\bar{R}^{(t)}(b')+\epsilon)} + \lambda\cdot\frac{1}{B_{max}}$$
其中 $\lambda=0.6$ 平衡聚焦弱项与均匀覆盖。

**复合奖励设计**：
$$R_{total} = \alpha\cdot R_{acc} + \beta\cdot R_{format} + \delta\cdot R_{length} + \gamma_q\cdot R_{tool}$$
- 工具奖励分解：$R_{tool} = R_{abs} \cdot R_{rel}$
  - 绝对信号：$R_{abs} = \mathbb{I}_{ans} \cdot \frac{B_{total}-B_{used}}{B_{total}}$（正确回答按比例奖励节省预算）
  - 相对信号：$R_{rel} = \mathbb{I}_{ans} \cdot \left(1 - \frac{B_{used}-B_{min}^+}{B_{max}^+-B_{min}^++\xi}\right)$（相对于同组最优轨迹的效率比较）
- 自适应权重：$\gamma_q = \gamma_{max} \cdot \frac{1}{G}\sum_{i=1}^G \mathbb{I}_{ans}^{(i)}$，按组准确率缩放效率信号强度

## 实验与结果
**数据集与设置**：
- 训练：NQ（79k）+ HotpotQA（90k），$B_{max}=5$
- 评估：7 个 QA 基准（3 通用 + 4 多跳），三个 backbone（Qwen2.5-7B、Llama-3.1-8B、Qwen3-4B）
- 检索：E5-base-v2 + 2018 Wikipedia dump（~18M passages），top-k=3

**主要结果（Qwen2.5-7B-Instruct，B=5）**：
| 方法 | 通用QA平均 | 多跳QA平均 | 总体平均 |
|------|-----------|-----------|---------|
| BATS | 0.260 | 0.197 | 0.227 |
| Search-o1 | 0.377 | 0.217 | 0.266 |
| Search-R1 | 0.503 | 0.318 | 0.397 |
| ZeroSearch | 0.524 | 0.291 | 0.387 |
| StepSearch | 0.528 | 0.300 | 0.403 |
| **AnySearch** | **0.551** | **0.331** | **0.431** |

AnySearch 相对最强基线 StepSearch 提升 **+6.9%**（0.403→0.431）。

**跨预算泛化**：在训练范围外的 B=6,7,8 上，AnySearch 保持单调递增性能（Bamboogle: B=5→0.40, B=8→0.42），而 Search-R1 在 B=5 后 plateau 于 ~0.37。

**效率优势**：在 HotpotQA (B=6) 上，AnySearch TP=0.354 vs Search-R1 TP=0.212（+67%）；总 token 消耗最低。

## 相关工作脉络
1. **预算感知推理**：BRPO/BudgetThinker 针对 token 预算通过 RL 训练，但推理时仍需显式预算信号；AnySearch 将预算感知完全内化。
2. **工具预算约束 agent**：OTC-PO/IKEA 通过奖励塑形减少冗余调用，但缺乏显式预算接口；AnySearch 支持任意约束指定。
3. **BATS**：引入预算状态追踪但依赖外部追踪器，推理时需持久化模块；AnySearch 消除了这一外部依赖。
4. **RL-based 搜索 agent**：Search-R1/R1-Searcher/ReSearch/Deep-Researcher 优化多轮交互轨迹但未显式建模预算分配；AnySearch 填补动态工具调用预算训练的空白。
5. **稀疏奖励缓解**：StepSearch/AutoRefine 引入步级监督；AnySearch 通过复合奖励的绝对+相对信号联合优化效率。
6. **模拟检索训练**：ZeroSearch/SSRL 用 LLM 模拟检索环境降低成本；AnySearch 使用真实搜索引擎但通过减少调用次数降低实际成本。

## 局限性与未来方向
- **预算定义单一**：当前仅建模为离散搜索调用计数，实际部署涉及多维度成本（延迟、费用、系统负载），需扩展为连续多目标成本 formulation。
- **骨干模型能力上限**：AnySearch 优化了搜索决策效率，但无法补偿模型自身知识缺失或检索语料未覆盖的答案。
- **静态语料限制**：训练与评估使用 2018 年 Wikipedia 快照，未测试时间适应性；扩展至持续更新的开放网页环境是未来方向。

## 研究启发与可借鉴点
1. **脚手架渐进移除策略**：通过显式引导（Phase I）建立目标行为模式，再移除辅助使策略内化（Phase II），可有效缩小训练-推理差距，适用于其他需要结构化决策能力的任务。
2. **滑动窗口自适应采样**：用近期表现而非全局统计指导采样分布，能更快速响应训练过程中的能力变化，适合多难度/多模式混合的训练场景。
3. ** correctness-gated 效率奖励**：效率信号乘以准确率指示器（$\mathbb{I}_{ans}$），避免在难以回答的问题上牺牲正确性换取效率，平衡了准确性与资源消耗的权衡。
4. **工具调用成本建模**：通过 OpenRouter 价格对比证明单次搜索 ≈ 20,000-100,000 output tokens，为 agent 系统的成本优化提供了量化依据。

## 关键术语表
**AnySearch**：本文提出的框架，通过强化学习将预算感知搜索能力内部化到单个策略中。
**Training Scaffold**：Phase I 使用的显式预算状态注入和结构化推理提示，用于引导策略学习预算感知决策模式。
**Adaptive Budget Sampling**：Phase II 基于滑动窗口准确率的预算采样策略，动态聚焦训练于弱项预算水平。
**Tool Productivity (TP)**：每单位外部搜索调用正确回答的问题数，衡量检索效率的核心指标。
**Composite Reward**：结合准确率、格式、长度和工具效率的复合奖励函数，通过自适应权重平衡各信号。
**Internalization**：指预算感知能力从外部脚手架转移到策略内部参数的过程，推理时不再依赖显式预算状态。
**GRPO**：Group Relative Policy Optimization，本文采用的强化学习优化算法。

## 可复现要素
- **数据集**：训练集 NQ + HotpotQA（公开）；评估集 NQ/TriviaQA/PopQA/HotpotQA/2Wiki-MultiHopQA/MuSiQue/Bamboogle（公开）
- **代码**：https://github.com/xwsun01/AnySearch（已开源）
- **模型权重**：使用 Qwen2.5-7B-Instruct、Llama-3.1-8B-Instruct、Qwen3-4B（公开权重）
- **关键超参**：$B_{max}=5$，$G=5$（group size），$\lambda=0.6$，$W=20$（滑动窗口），$L_{limit}=2048$，$L_{tol}=1024$；学习率 $1\times10^{-6}$，KL penalty $\beta_{KL}=0.001$
- **硬件**：8× NVIDIA H800 (80GB)，训练 500 steps
