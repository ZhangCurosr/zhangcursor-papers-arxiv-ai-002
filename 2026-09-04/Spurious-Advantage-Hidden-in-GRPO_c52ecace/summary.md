---
title: "Spurious-Advantage-Hidden-in-GRPO"
source: https://arxiv.org/pdf/2609.04063v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 18:54:27"
field: "大语言模型强化学习训练"
keywords: ["GRPO", "reinforcement learning", "advantage estimation", "spurious advantage", "RLVR", "math reasoning", "search agent"]
innovations: ["揭示GRPO优势估计器中虚假优势现象：组内组成无法区分推理成功与猜测成功", "提出SIGNBALANCE方法：解耦幅度与组内组成，保持符号并恢复力平衡", "统一分析三类场景：有界答案任务、开放式数据中的有界子情况、多轮搜索代理"]
benchmarks: ["MATH-7.5K", "GSM8K", "MATH-500", "SAT-Math", "AQuA", "NQ", "HotpotQA", "2WikiMultiHopQA"]
---

# 论文速读：Spurious-Advantage-Hidden-in-GRPO

## 一句话总结
本文揭示了GRPO优势估计器中隐藏的"虚假优势"现象：在答案空间有界（如多选题）或开放式答案集中存在有界子情况时，通过猜测而非推理获得正确结果的rollout仍会被分配高优势幅度，误导策略学习侥幸猜测行为而非有效推理。为此提出SIGNBALANCE方法，通过将rollout幅度与组内正确/错误组成解耦，在开放式数学任务上与GRPO持平，在封闭式答案任务和搜索代理任务上显著提升。

## 研究问题与动机
- **核心问题**：GRPO的优势估计器在二元奖励下，正确rollout的优势幅度|Â⁺|=√(n⁻/n⁺)仅取决于组内正确数量n⁺，无法区分"通过推理得到正确答案"和"通过猜测候选答案得到正确答案"两种情形。
- **现有方法不足**：
  - 现有GRPO变体工作（clip范围优化、归一化改进、采样策略重塑等）均围绕组内相对统计设计，未考虑任务答案空间结构带来的虚假信号。
  - 当p_g（非推理成功率）不可忽略时，组内正确rollout中包含大量猜测成分，GRPO的高幅度会放大这一虚假学习信号，导致策略偏向猜测行为。
  - 开放式数学训练集（如MATH-7.5K）实际包含55.95%的有界答案子问题（如小整数、简单分数等），同样受虚假优势影响。
  - 多轮搜索代理使用结果级奖励（outcome-only reward），不同轨迹长度的正确rollout获得相同幅度，无法区分有效推理路径与冗余搜索路径。

## 核心贡献（创新点）
- **揭示GRPO的虚假优势现象**：首次系统分析GRPO优势估计器在答案空间有界场景下的局限性，证明其将猜测成功与推理成功同等放大，为重新审视RLVR中信用分配提供新视角。
- **识别三类虚假优势显著场景**：形式化界定 bounded-answer tasks、open-answer sets中的bounded sub-cases、multi-turn search agents三种场景，建立统一分析框架，填补了现有工作仅关注开放式数学推理的理论空白。
- **提出SIGNBALANCE优势估计器**：设计幅度与组内组成无关的估计器，保留验证器符号、使用全局尺度c，并通过stop-gradient按类重缩放恢复零均值力平衡，参数-free且不改变PPO代理损失。
- **全面实验验证**：在0.5B、3B模型规模的数学推理任务及7B搜索代理任务上进行系统评估，证明SIGNBALANCE在封闭式答案基准（SAT-Math +6.26、AQuA +5.90）和搜索代理任务（Avg-6提升1.80）上显著优于GRPO，同时在开放式数学任务上与GRPO持平。

## 方法详解
- **问题建模**：将每提示成功率分解为p_q = p_r + (1-p_r)·p_g，其中p_r为推理成功率，p_g为非推理成功率（猜测概率）。GRPO优势幅度√(n⁻/n⁺)作用于期望为Gp_q的n⁺，包含推理驱动和猜测驱动两部分，后者即为虚假优势。
- **Step 1（类内归一化）**：ĤA_i^(1) = (r_i - μ^(c(i))) / (σ^(c(i)) + ε)，正确/错误rollout分别归一化，但幅度仍依赖类内数量n⁺或n⁻。
- **Step 2（纯符号幅度）**：ĤA_i^(2) = sign(r_i)·c，幅度完全解耦于组内组成，但破坏batch-level零均值属性（∑|Â⁺|≠∑|Â⁻|当n⁺≠n⁻）。
- **Step 3（带力平衡的符号幅度，SIGNBALANCE）**：Â⁺=c，Â⁻=-c·sg[n⁺/n⁻]，其中sg[·]为stop-gradient算子。按类重缩放恢复力平衡∑|Â⁺|=∑|Â⁻|，同时保持per-rollout幅度常数c不依赖(n⁺,n⁻)。
- **实现细节**：全局尺度c=1，直接替换GRPO损失中的优势估计公式，无额外参数、无外部模型、无推理开销。算法伪代码见附录F。

## 实验与结果
- **数据集与评估**：
  - 数学推理：Qwen2.5-0.5B-Instruct训练MATH-7.5K（G=16 rollouts/question，max length 8192），评估8个基准（GSM8K、MATH-500、Minerva-Math、OlympiadBench为开放式；MMLU-math、SAT-Math、AQuA、AMC为封闭式）。
  - 规模泛化：Qwen2.5-3B-Base评估更难题库（含Gaokao、College Math、AIME24、AMC23）。
  - 搜索代理：Qwen2.5-7B-Instruct，6个文本QA基准（NQ、TriviaQA、PopQA单跳；HotpotQA、2WikiMultiHopQA、MuSiQue多跳），最多B=4次搜索调用。
- **主要结果**：
  - **0.5B数学**：SIGNBALANCE Avg-8=36.61 vs GRPO 34.24（+2.37）。封闭式基准提升显著：SAT-Math 71.88 vs 65.62（+6.26）、AQuA 35.43 vs 29.53（+5.90）、AMC 10.84 vs 6.02（+4.82）；开放式基准与GRPO相当（GSM8K 49.66 vs 49.89）。
  - **3B数学**：SIGNBALANCE Avg-8=43.78 vs GRPO 42.80（+0.98），领先基准数从4/8增至7/8，差距随规模扩大而稳定。
  - **搜索代理**：SIGNBALANCE Avg-6=37.80，超越Search-R1（36.00）+1.80、StepSearch（36.44）+1.36；在2WikiMultiHopQA上达35.20 vs Search-R1 27.58（+7.62）。
- **消融实验**：Step 2（纯符号）移除组内计数依赖后SAT-Math提升+6.26但MATH-500下降4.80（力平衡破坏）；Step 3完整方法在封闭式基准上持续领先，Ablation表5显示力平衡恢复是关键。
- **稳定性分析**：Figure 5显示SIGNBALANCE在训练全程（前100步即分离）持续优于GRPO，排除偶然checkpoint解释。

## 相关工作脉络
- **GRPO及变体**（Shao et al., 2024; Liu et al., 2025; Yu et al., 2026; Xiao et al., 2025等）：聚焦clip范围、归一化、采样策略优化，评估集中于开放式数学，假设组内组成纯粹反映推理能力，本文指出其在有界答案场景下存在系统性偏差。
- **优势估计器理论分析**（Mroueh, 2025）：推导GRPO二元奖励下的闭式权重ω⁺(p)=√((1-p)/p)，解释为罕见成功rollout获得高权重，本文沿用该形式但扩展至p_g>0场景，揭示其学习信号中混杂的猜测成分。
- **多目标复合奖励方法**（Ichihara et al., 2025）：通过额外奖励项抑制reward hacking，属过程控制；SIGNBALANCE从优势估计源头消除虚假信号，无需改变奖励设计。
- **搜索代理训练**（Jin et al., 2025; Wang et al., 2025b等）：现有工作优化步级奖励或搜索策略，本文证明即使相同外层GRPO循环，仅替换优势估计器即可提升多轮搜索任务性能（Avg-6 +1.80 vs Search-R1）。
- **负样本利用方法**（Zhu et al., 2026; Nan et al., 2025）：聚焦wrong rollouts的信用分配，本文关注correct rollouts内部来源区分，二者互补。
- **对比定位**：现有工作从"组内相对排名/量化"角度改进GRPO，本文从"绝对幅度解耦"角度切入，首次将答案空间结构纳入优势估计设计考量。

## 局限性与未来方向
- **验证范围局限**：仅在数学推理和搜索代理任务验证，未扩展到工具选择等更复杂多步决策场景（作者推测工具库有限时GRPO同样存在虚假优势）。
- **缺乏推理能力量化**：虽观察SIGNBALANCE引导策略学习更有效推理，但未定量测量推理置信度、不确定性行为等指标。
- **固定全局尺度**：c=1为经验设定，未探索自适应尺度学习或任务特定优化。
- **未来方向**：扩展至工具使用Agent、多智能体协作；设计推理过程可度量指标验证信号纯净度；探索scale c的自动化搜索机制。

## 研究启发与可借鉴点
- **答案空间结构分析框架**：可将p_g（猜测概率）作为任务特性指标，指导优势估计器选择；对多选题、分类任务等天然有界场景，应优先验证虚假优势影响。
- **力平衡与幅度解耦的 trade-off 设计**：SIGNBALANCE Step 2→3的消融揭示力平衡对开放答案任务的重要性，后续方法需在"消除组内依赖"与"保持batch统计属性"间权衡。
- **推理-猜测信号分离的通用思路**：可推广至其他RLVR场景（如代码生成、多步规划），设计过程监督与结果监督联合的优势估计，进一步隔离虚假信号。
- **训练轨迹稳定性诊断**：Figure 5的checkpoint-to-checkpoint分析可作为优势估计器鲁棒性评估标准，新方法应证明性能提升非偶然。
- **MATH数据集再分析价值**：Table 1的语法-答案形状分类方法可直接复用于其他数学基准，量化"名义开放实则混合"问题的虚假优势比例。

## 关键术语表
**Spurious Advantage（虚假优势）**：GRPO优势估计器将猜测成功与推理成功同等赋予高幅度，导致策略被误导学习侥幸行为而非有效推理的虚假学习信号。
**Within-group Composition（组内组成）**：组内正确rollout数n⁺与错误rollout数n⁻的比例关系，GRPO优势幅度|Â⁺|=√(n⁻/n⁺)仅依赖此统计量。
**Bounded-answer Tasks（有界答案任务）**：答案空间为有限集合的任务（如k选项多选题），随机猜测即以1/k概率获得正确结果。
**Outcome-only Reward（结果级奖励）**：仅依据最终答案匹配给予二元奖励，不区分到达答案的轨迹过程，多轮搜索代理常用。
**Sign-Balance（符号平衡）**：SIGNBALANCE核心设计，保持验证器符号、幅度常数c，通过stop-gradient重缩放恢复batch级零均值力平衡。
**Random-guess Share（随机猜测占比）**：正确rollout中可由无推理策略产生的比例，MMLU-math上达30.1%，AQuA上达53.4%。
**Stop-gradient（梯度停止）**：符号sg[·]，使重缩放比例n⁺/n⁻仅影响幅度值而不参与反向传播，避免引入额外优化动态。

## 可复现要素
- **数据集**：MATH-7.5K（约7500数学问题）、MATH-500、GSM8K、Minerva-Math、OlympiadBench、MMLU-math、SAT-Math、AQuA、AMC、Gaokao、College Math、AIME24、AMC23、NQ、TriviaQA、PopQA、HotpotQA、2WikiMultiHopQA、MuSiQue；均公开可用。
- **代码**：论文声明"Code will be released"，当前未开源。
- **权重**：基础模型Qwen2.5-0.5B-Instruct、Qwen2.5-3B-Base、Qwen2.5-7B-Instruct公开；SIGNBALANCE训练权重未提及。
- **关键超参**：G=16 rollouts/question，max response length=8192，learning rate=1×10⁻⁶，KL coefficient β=1×10⁻³，max 1000 steps；搜索代理B=4 search invocations；全局尺度c=1。
- **实现细节**：vLLM生成rollouts，rule-based verifier精确匹配gold answer，best checkpoint selection by Avg-8。
