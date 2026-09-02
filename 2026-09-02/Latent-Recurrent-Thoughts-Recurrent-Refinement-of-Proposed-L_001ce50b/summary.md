---
title: "Latent-Recurrent-Thoughts-Recurrent-Refinement-of-Proposed-L"
source: https://arxiv.org/pdf/2609.01117v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 22:14:59"
field: "大模型推理方法"
keywords: ["latent reasoning", "frozen LLM", "recurrent refinement", "chain-of-thought", "continuous-space reasoning", "parameter-efficient"]
innovations: ["任务专用提议器+TRM递归精炼器的冻结解码器推理框架", "截断梯度展开实现多步递归精炼而不增加训练负担", "残差参数化约束精炼器以有界修正方式工作避免漂移"]
benchmarks: ["Countdown-4", "Sudoku", "HumanEval", "MBPP", "StrategyQA"]
---

# 论文速读：Latent-Recurrent-Thoughts-Recurrent-Refinement-of-Proposed-Latents

## 一句话总结
本文提出 **Latent Recurrent Thoughts (LRT)**，通过在冻结的LLM前端部署任务专用提议器 + 递归精炼器（TRM），在连续隐空间中进行多步迭代推理，以极小可训练参数量（11.2M）超越既有冻结解码器方法和零样本CoT。

## 研究问题与动机
- 现有Chain-of-Thought (CoT) 在离散token空间推理，误差传播且依赖可模仿的推理轨迹。
- 连续隐空间推理是可行替代路径，但已有方法（SoftCoT使用冻结通用助手LM、EBM-CoT添加能量模型精炼）在非语言推理场景下崩溃（如Countdown-4仅5.9%和8.4%，远低于零样本CoT的30.0%）。
- 通用助手生成的潜变量可能偏离解码器输入流形，产生反作用；而能量精炼器只能沿标量场梯度微调，无法进行多步约束传播。
- 核心问题：如何在不更新解码器参数的条件下，获得高质量、可被递归优化的实例条件化潜变量？

## 核心贡献（创新点）
1. **任务专用提议器 + 递归精炼器组合**：将通用LLM助手替换为任务专用双向Transformer编码器，并引入TRM递归网络进行多步有界残差修正，而非单次前馈或能量下降。
2. **冻结解码器下的分布式计算架构**：首次将递归理由器（TRM）与预训练冻结LLM配对，前者承担迭代计算，后者贡献序列建模与语言能力，两者分工明确。
3. **截断梯度展开训练策略**：仅对最后一个递归周期进行反向传播，前期40步仅作状态预热，使深度递归在推理时展开而不增加训练计算负担。
4. **跨符号/自然语言统一框架**：在答案监督但无推理轨迹的符号任务（Countdown-4、Sudoku）和编程/问答任务（HumanEval、MBPP、StrategyQA）上均有效，克服了既有方法离开自然语言语料后的崩溃问题。

## 方法详解
- **管道公式**：$L^{(0)} = g_\psi(x)$，$L^\star = L^{(0)} + r_\phi(L^{(0)})$，$\hat{y} = \mathcal{M}(I, x, L^\star)$，其中$g_\psi$为提议器，$r_\phi$为TRM精炼器，$\mathcal{M}$为冻结Qwen3-8B解码器。
- **任务专用提议器**：共享解码器词嵌入表，经可学习投影$P_\downarrow: \mathbb{R}^d \to \mathbb{R}^{d'}$（$d'=256$）压缩，拼接K=32个可学习查询向量，经2个双向Transformer块（self-attention + SwiGLU）后读取查询位置输出，再经$P_\uparrow: \mathbb{R}^{d'} \to \mathbb{R}^d$还原，生成$L^{(0)} \in \mathbb{R}^{K \times d}$。约4.2M参数。
- **递归精炼器**：维护快/慢两层状态$z_L, z_H \in \mathbb{R}^{K \times d'}$。每个高层周期内执行T=4次快更新$z_L \leftarrow f(z_L, z_H + u)$（$u$为提议器输出的下投影，每步重新注入以锚定问题），再1次慢更新$z_H \leftarrow f(z_H, z_L)$。最终残差$\Delta = P'_\uparrow(z_H)$， refined latents $L^\star = L^{(0)} + \Delta$。加残差正则$\lambda\|\Delta\|^2$（$\lambda=0.01$）防止漂移。约7M参数。
- **两阶段训练**：Stage 1训练提议器（冻结精炼器）；Stage 2冻结提议器、训练精炼器，仅用最终答案的交叉熵+残差正则。全程梯度穿过冻结解码器激活但不更新其权重。
- **推理设置**：训练用K=32，推理仅取前$K_{infer}=4$；精炼器展开$S \times H = 9$个高层周期（共45次transition pass），仅在最后周期反向传播。

## 实验与结果
- **数据集**：Countdown-4（生成，100k训练/1k评测）、Sudoku（生成，100k/1k）、HumanEval（164评测）、MBPP（374训练/500评测）、StrategyQA（2061训练/229评测）。
- **评估指标**：Countdown-4/Sudoku精确求解率、HumanEval/MBPP pass@1、StrategyQA准确率。
- **最强结果**：
  - Countdown-4：**56.7%**（对比零样本CoT 30.0%，SoftCoT 5.9%，EBM-CoT 8.4%）
  - Sudoku：**49.2%**（对比零样本CoT 23.9%）
  - HumanEval：**37.8%**（对比EBM-CoT 25.0%）
  - MBPP：**51.5%**
  - StrategyQA：**75.1%**
  - 五任务平均：**54.1%**（对比EBM-CoT 33.5%，零样本CoT 35.0%）
- LRT在Countdown-4上甚至超越了从头训练的扩散求解器MGDM（52.0%），在Sudoku上虽不及专用求解器（TRM 97.2%，EqR 99.8%），但唯一同时处理语言任务。

## 相关工作脉络
1. **SoftCoT**（Xu et al., 2025a）：冻结通用助手LM生成软Thoughts，无精炼步骤，在符号任务上崩溃。
2. **EBM-CoT**（Chen et al., 2025b）：保留通用提议器，添加能量模型进行单步梯度精炼，仍无法处理非语言推理。
3. **Coconut / CODI / Token-Assorted**：通过微调解码器或插入隐状态实现连续推理，违反冻结解码器设定。
4. **TRM / HRM / GRAM / EqR**：从头训练的递归/能量求解器，在单一符号任务上强，但无语言接口，无法跨域泛化。
5. **Prefix-tuning / P-tuning v2**：参数高效提示适配，在本研究设定下仅达28.4%（Countdown-4），证明实例条件化+递归计算不可替代。
6. **Thinking-mode prompting**（Qwen3）：消耗8.9–453.9 TFLOP/example，比LRT（≈1–2 TFLOP）高数倍至数百倍，但在Sudoku上仅0.5%，凸显token空间搜索在约束满足任务上的局限。

## 局限性与未来方向
- **任务专用训练**：每类任务需重新训练提议器和精炼器，非zero-shot通用模型；跨任务泛化未验证。
- **符号精度上限**：在Sudoku等纯符号任务上，专用从头训练求解器（如EqR 99.8%）仍远超LRT（49.2%）。
- **推理延迟随深度增长**：递归展开增加计算步数，虽因截断训练保持低成本，但深度不可无限扩展。
- **机制分析为相关性**：线性探针仅测量线性可解码信息，无法严格定位计算分布。
- **未来方向**：跨任务迁移、更优的TRM序列到序列适配、更大规模解码器的扩展验证。

## 研究启发与可借鉴点
1. **提议器-精炼器解耦设计**：任务专用编码器+递归精炼器的组合模式可迁移至其他需要连续隐空间推理的任务（如规划、约束满足）。
2. **截断梯度展开**：仅在最后周期反向传播、前期stop-gradient热身的策略，在保持深度递归推理能力的同时控制训练成本，可作为递归模型训练的通用技巧。
3. **残差参数化（$L^\star = L^{(0)} + \Delta$）**：强制精炼器以有界修正形式工作，避免漂移，这一约束可推广到其他隐变量 refinement 场景。
4. **train/inference latent count 非对称性**：训练用大K=32作scratch space，推理仅用K=4，揭示了潜变量容量与注入规模的分离设计思路。
5. **答案监督无轨迹的可行性**：证明在连续隐空间进行推理无需中间步骤标注，对缺乏推理轨迹数据的新任务具有借鉴价值。

## 关键术语表
- **Chain-of-Thought (CoT)**：让模型逐步输出推理中间步骤以提升复杂任务表现的技术。
- **Frozen Decoder**：保持参数不变的预训练语言模型，仅接受额外输入进行推理。
- **Soft Tokens / Latent Vectors**：连续嵌入向量，作为虚拟token注入解码器输入序列。
- **TRM (Tiny Recursive Model)**：小型递归网络，通过重复应用同一transition函数实现深层计算而参数不变。
- **Residual Correction**：精炼器输出对基础潜变量的有界修正量，而非完全重新生成。
- **Truncated Gradient Unrolling**：仅在展开的最后一步进行反向传播，前期步骤仅warmup状态。
- **Answer Supervision**：仅使用最终答案标签训练，不提供中间推理过程标注。
- **Task-Dedicated Proposer**：针对特定任务家族训练的编码器，生成初始潜变量。

## 可复现要素
- **数据集**：Countdown-4和Sudoku由作者按公开协议合成（Gandhi et al. 2024; Ye et al. 2025）；HumanEval、MBPP、StrategyQA为公开数据集。
- **代码/权重**：代码和训练权重以MIT许可在 https://github.com/czl-david/latent-recurrent-thoughts 开源。
- **关键超参**：$d'=256$，$K=32$（训练）/$K_{infer}=4$（推理），$S=3, H=3, T=4$，$\lambda=0.01$，AdamW（$\beta_1=0.9, \beta_2=0.999, \epsilon=10^{-8}$），cosine LR schedule（5% warmup），peak LR $3\times10^{-4}$（Stage 1）/$2\times10^{-4}$（Stage 2），batch size 64，30 epochs，bfloat16，单卡96GB GPU。
