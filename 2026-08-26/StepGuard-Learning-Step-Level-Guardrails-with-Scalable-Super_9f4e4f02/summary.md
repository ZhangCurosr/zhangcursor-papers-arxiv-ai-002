---
title: "StepGuard-Learning-Step-Level-Guardrails-with-Scalable-Super"
source: https://arxiv.org/pdf/2608.24777v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 06:05:55"
field: "Agent安全与护栏"
keywords: ["agent safety", "guardrail", "step-level guard", "GRPO", "synthetic data generation", "safety-utility tradeoff", "tool-use safety"]
innovations: ["提出StepGen前缀对齐数据引擎，自动构建共享执行前缀的安全/不安全对比轨迹对并提供步骤级监督", "提出Balance-GRPO训练算法，根据类间准确率差距动态重加权GRPO优势以减小防御偏差", "构建4B步级防护模型StepGuard，在执行前检查与事后审计双重模式下实现开源模型最优的安全-效用平衡"]
benchmarks: ["ATBench", "R-Judge", "ASSE Security", "TS-Bench-Dojo", "TS-Bench-Harm", "AgentDojo", "AgentDyn", "AgentHarm"]
---

# 论文速读：StepGuard: Learning Step-Level Guardrails with Scalable Supervision and Safety–Utility Balancing

## 一句话总结
本文提出了 **StepGuard**，一个基于 Qwen3-4B 的步级防护模型，支持对候选工具调用进行执行前安全检查，以及对已完成轨迹进行事后安全审计；通过自动数据引擎 **StepGen** 和动态校准训练方法 **Balance-GRPO**，实现了在 AgentDojo 和 AgentDyn 上将攻击成功率（ASR）降低 77.3%，同时任务效用仅下降 2.8 点的优异安全-效用平衡。

## 研究问题与动机
- **现有防护的粒度局限**：大多数现有 guardrail 系统评估已完成的轨迹或孤立输入，对执行前的步级（step-level）候选工具调用缺乏前置监控能力，无法在风险行动执行前进行阻断。
- **可扩展的高质量步级监督数据缺失**：真实世界中的不安全执行事件极为稀疏，人工构建成本高且难以扩展；已有数据在规模、风险覆盖度以及同决策点上下文匹配的安全/不安全行动对方面存在明显不足。
- **安全-效用平衡缺乏可控性**：现有训练方法（如 DPO/GRPO 的简单变体）主要优化整体性能，未根据防护模型在安全类与不安全类动作上的观察准确度差距来动态调整训练权重，导致过度防御（over-defense）或防御不足（under-defense）的偏差难以消除。
- **防御偏差的普遍存在**：实证表明现有防护模型在安全/不安全两类样本上的准确率存在显著差异，且预测稳定性与偏差程度正相关，亟需显式校准机制。

## 核心贡献（创新点）
- **提出 StepGuard 步级防护模型**：以 Qwen3-4B-Instruct 为基础，支持执行前候选动作检查和已完成轨迹审计的双重模式；与已有工作相比，StepGuard 是首个明确针对 agent 工具调用场景进行细粒度步骤级安全防护并兼顾上下文感知的小型开放权重模型。
- **设计 StepGen 前缀对齐自动数据引擎**：通过风险锚点规划、对照分支重生成（Refuse/Aware/Benign）和结构化过滤，生成共享执行前缀的安全/不安全轨迹对；与 AuraGen 等仅停留在计划级监督的工作不同，StepGen 提供细粒度步骤级标注和工具语义匹配的对照组。
- **提出 Balance-GRPO 训练算法**：在标准 GRPO 基础上引入基于类级准确率差距的动态优势加权机制（\(\omega_i\) 和 \(c_i\)），在不修改 rollout prompt 和原始奖励的前提下自动强化表现较差的类别；与静态权重调整（如 70:30 upsampling）不同，该方法在线根据实时观察到的安全-不安全准确率偏差自适应调节。
- **系统化的安全-效用评估体系**：在 5 个静态基准（ATBench、R-Judge、ASSE Security、TS-Bench-Dojo、TS-Bench-Harm）和 3 个运行时环境（AgentDojo、AgentDyn、AgentHarm）上进行全面评测，并提供了推理开销分析（每调用仅增加约 600ms 延迟，占任务总时长 7.24%）。

## 方法详解

### StepGen 数据引擎
- **风险锚点轨迹构建**：采样风险源（risk source）、失效模式（failure mode）和危害类型（harm type）三元组 \((r, f, h)\)，由规划器构建结构化执行计划 \(P\)，指定工具调用、参数约束与血缘关系，并标注首个不安全动作 \(a_{i^\star}\) 为风险锚点。
- **前缀对齐对比分支生成**：保留 unsafe 轨迹在锚点前的执行前缀 \(\tau_{<i^\star}^U\)，分别以 Refuse（拒绝执行）和 Aware（识别风险后采用安全替代方案）模式重生成后缀：
  \[
  \tau^m = \tau_{<i^\star}^U \circ \text{ReRoll}(P_{\geq i^\star}, m), \quad m \in \{\text{Refuse}, \text{Aware}\}
  \]
  同时独立生成使用相同工具子集但无对抗背景的良性轨迹（benign tool-reuse）。
- **步骤级标注与质量过滤**：每个动作获得 Safe/Unsafe 标签及五字段结构化解释（evidence、intent、consequence、decision、risk-source）；通过机械验证器 \(\phi_{\text{mech}}\)（检查 JSON 格式、参数血缘、锚点正确性、注入标记位置）和 LLM 质量审计器 \(\phi_{\text{qual}}\)（六个语义维度，总分 0–18，阈值 \(\eta = 14\)）两级过滤。

### StepGuard 训练流程
- **冷启动 SFT**：在 3K 条 StepGen 生成样本上进行 token-level 交叉熵损失微调，输入为完整轨迹或"上下文+候选动作"，输出为安全标签、风险类别、风险步骤定位及安全推理。
- **Balance-GRPO 在线校准**：
  - **结构化奖励**：\(r_{i,j} = 0.5 \cdot \mathbb{I}[\hat{Y}_{i,j}=Y_i] + 0.5 \cdot \mathbb{I}[\hat{Y}_{i,j}=Y_i] \cdot \mathbb{I}[\hat{R}_{i,j}=R_i]\)，风险类别正确仅在安全判断正确时计分。
  - **归一化优势**：\(A_{i,j} = \frac{r_{i,j} - \mu_i}{\sigma_i + \delta}\)。
  - **动态重加权**：类别计数校正因子 \(c_i = \text{clip}(\frac{|B|}{2n_{Y_i}}, c_{\min}, c_{\max})\)；准确率差距因子 \(\omega_i\)，当 \(\Delta_{\text{cls}} = \text{Acc}_{\text{safe}} - \text{Acc}_{\text{unsafe}}\) 偏离目标 \(g_0\) 时给准确率较低类别更大权重：
    \[
    \omega_i = \begin{cases} 1 + \lambda \max(-\bar{\Delta}_{\text{cls}}, 0), & Y_i = \text{safe} \\ 1 + \lambda \max(\bar{\Delta}_{\text{cls}}, 0), & Y_i = \text{unsafe} \end{cases}
    \]
    最终平衡优势 \(A'_{i,j} = c_i \cdot \omega_i \cdot A_{i,j}\)。
  - **最终目标函数**：标准 clipped GRPO 目标，\(\mathcal{I}(\theta) = \mathbb{E}\left[\frac{1}{G}\sum_{j=1}^G \min\left(\rho_{i,j} A'_{i,j}, \text{clip}(\rho_{i,j}, 1-\epsilon, 1+\epsilon) A'_{i,j}\right)\right] - \beta D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})\)。

## 实验与结果
- **评测基准**：轨迹级 ATBench、R-Judge、ASSE Security；步级 TS-Bench-Dojo、TS-Bench-Harm；运行时 AgentDojo、AgentDyn、AgentHarm。
- **主要结果**：
  - **静态评测**：StepGuard（4B）在轨迹级达到 83.0 Acc / 83.3 F1，在步级达到 84.8 Acc / 84.1 F1，在所有开源 agent-guard 模型中取得最高平均准确率；与 GPT-5.4（闭源）的轨迹级 83.0 Acc / 84.4 F1 基本持平。
  - **运行时评测**：在 AgentDojo 上 ASR=1.2（较无防护的 25.1 下降 95.2%）、Utility=90.7；在 AgentDyn 上 ASR=9.3（较无防护的 21.2 下降 56.1%）、Utility=66.7；两项合计相对无防护设置平均 ASR 降低 77.3%，平均 utility 仅下降 2.8 分。
  - **Balance-GRPO 消融**：相比 vanilla GRPO，将安全-不安全准确率差距从 13.0 降至 8.0，AgentDojo 上 utility 提升 5.4 分（85.3→90.7），ASR 仅增加 0.3。
  - **泛化能力**：仅在 2 类风险源上训练后，在未见的 6 类 ATBench 风险源上仍达到 74.9 Acc / 78.1 F1，显著优于基础模型（49.6/35.6）。
  - **推理开销**：每次防护调用平均 599.9ms / 195.5 tokens，占 AgentDojo 任务总时长约 7.24%。

## 相关工作脉络
- **TS-Guard**（Mou et al., 2026）：同样在步级评估候选工具调用，但缺乏前缀对齐的对照数据生成机制，且未显式校准安全/不安全类别间的性能差距，在 TS-Bench-Harm 上 F1 为 76.4，低于 StepGuard 的 75.2（Dojo 侧 93.0 vs 83.5）。
- **Safiron**（Huang et al., 2025）：在计划级（plan-level）进行前置安全评估，无法定位具体风险步骤；在 ATBench 上 F1 仅 56.4，显著低于 StepGuard 的 89.1。
- **AgentDoG-Qwen3-4B**（Liu et al., 2026b）：聚焦于诊断性防护框架，但在步级评测中 Safe Acc 仅 26.5，存在严重过度防御，安全性-可用性平衡明显劣于 StepGuard。
- **ShieldAgent-THU**（Zhang et al., 2024c）：基于知识增强的 guard agent，F1 达 65.5 但 Safe Acc 仅 42.3，防御偏差达 -54.2，且预测不稳定性高（Safe Flips=8.7）。
- **Qwen3-Guard / LlamaGuard3 / ProGuard**：内容级 LLM guard，主要面向孤立输入/输出审核，在需要多步工具调用上下文推理的 agent 安全评测中表现显著偏弱（如 Qwen3-Guard 在 TS-Bench-Dojo 上 F1 为 0.0）。
- **Meta-SecAlign-70B**（Chen et al., 2026）：模型级对齐基线，需大规模重新训练底层 agent，资源成本远高于本文的独立部署型 guard 方案。

## 局限性与未来方向
- **数据依赖合成生成**：StepGen 基于 LLM 和模拟器生成轨迹，可能存在覆盖盲区、教师模型偏差和标注错误；需要进一步引入真实世界日志或人类验证数据。
- **评估基准的局限性**：静态/动态评测覆盖代表性场景但尚未涵盖完全开放的 tools 生态、更长周期工作流、多 agent 交互环境及自适应对抗攻击者。
- **非形式化安全保证**：StepGuard 是启发式防护而非形式化验证，仍存在假阳性/假阴性，且部署引入额外推理延迟和拦截后的替代策略需求。
- **部分风险边界模糊**：错误分析显示学术诚信违规、制度性规范违反等非技术性安全风险识别能力有限，未来可扩展至更广义的 agent 行为规范。

## 研究启发与可借鉴点
- **前缀对齐对照数据生成范式**：StepGen 的"固定执行前缀 + 风险锚点分支"设计可迁移至其他需要细粒度因果归因的监督学习场景（如自主机器人决策、代码生成安全校验），是低成本构建高质量对比训练数据的通用思路。
- **基于实时性能差距的动态类别加权**：Balance-GRPO 的核心思想（用观察到的类间准确率偏差在线调整梯度权重，而不修改 reward/prompt）可推广至其他存在类别不平衡或防御偏差的 RLHF/GRPO 训练任务。
- **五字段结构化安全推理输出**：evidence → intent → consequence → decision → risk-source 的链式解释框架可作为多步推理 guard 的通用输出 schema，便于后续模块消费和人工审查。
- **工具血缘（lineage）追踪辅助安全判断**：在参数标注中区分 USER / TOOL / SYSTEM 来源，为模型提供因果溯源信号；该方法可被借鉴到数据隐私保护和可信信息流追踪研究中。
- **运行时 guard 的开销量化与部署设计**：论文提供了详细的单调用延迟、token 生成量和时间占比数据，为后续研究在设计低成本 guard 时提供了可直接复用的性能基准和部署参数参考。

## 关键术语表
- **StepGuard**：本文提出的 4B 步级防护模型，支持执行前候选动作检查和事后轨迹审计两种模式。
- **StepGen**：自动数据引擎，通过风险锚点规划和前缀对齐对比分支生成，构造步骤级标注的安全/不安全轨迹对。
- **Balance-GRPO**：改进的 GRPO 训练算法，根据安全/不安全类的实时准确率差距动态重加权归一化优势，减少防御偏差。
- **ASR（Attack Success Rate）**：攻击成功率，衡量在注入攻击后 agent 成功执行有害行动的比例，越低越好。
- **Utility**：任务效用分数，衡量 agent 在受防护情况下完成良性任务的能力，越高越好。
- **Prefix-aligned trajectory**：前缀对齐轨迹，指在风险锚点之前具有完全相同执行历史的对照轨迹组。
- **Risk anchor（风险锚点）**：轨迹中被指定为首个不安全决策的关键步骤，用于隔离风险评估与执行上下文。
- **Defense bias（防御偏差）**：防护模型在安全类与不安全类样本上表现不一致的倾向，分为过度防御（over-defense）和防御不足（under-defense）。

## 可复现要素
- **数据集**：StepGen 生成池共 10,815 条记录，其中 3K 用于 SFT、4K 用于 RL；训练数据未公开，但 StepGen 代码已开源（github.com/zheng977/StepGuard）。
- **代码/权重**：代码已开源，模型权重通过 HuggingFace/ninty-seven/StepGuard 提供。
- **关键超参**：SFT 学习率 \(2 \times 10^{-5}\)、batch size 16、2 epochs；RL 学习率 \(5 \times 10^{-7}\)、rollout batch 64 prompts × 8 responses、clip range 0.2/0.28、\(\lambda = 2.0\)、deadband 0.02、权重范围 [0.75, 1.5]；最大序列长度 16,384 tokens。
- **硬件**：4 × NVIDIA H200 GPU，SFT 约 0.5h，RL 约 1.5h。
- **评估协议**：静态评测使用 greedy decoding（temperature=0）、context 长度 32K；运行时评测统一使用 Qwen3.6-35B-A3B 作为 agent backbone。
