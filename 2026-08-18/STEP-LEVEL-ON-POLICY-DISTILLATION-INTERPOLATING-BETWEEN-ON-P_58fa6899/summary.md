---
title: "STEP-LEVEL-ON-POLICY-DISTILLATION-INTERPOLATING-BETWEEN-ON-P"
source: https://arxiv.org/pdf/2608.16333v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 04:32:19"
field: "大语言模型知识蒸馏"
keywords: ["on-policy distillation", "step-level supervision", "black-box distillation", "SFT", "reasoning", "agent"]
innovations: ["提出SOPD融合SFT长程监督与OPD on-policy状态覆盖", "建立SFT到OPD的步长插值理论联系", "设计packed attention mask实现并行step-level蒸馏"]
benchmarks: ["ALFWorld", "AIME24", "AIME25", "HMMT25-Feb", "HMMT25-Nov", "DeepMath"]
---

# 论文速读：STEP-LEVEL-ON-POLICY-DISTILLATION-INTERPOLATING-BETWEEN-ON-P

## 一句话总结
论文提出 Step-Level On-Policy Distillation (SOPD)，通过将学生生成的完整轨迹划分为自然步骤并在每个步骤上请求教师生成独立的目标响应，在保持 on-policy 状态覆盖的同时提供连贯的多步监督；SOPD 在极限情况下分别退化为 SFT（单步）和 forward-KL OPD（单 token），并在 ALFWorld 和数学推理任务上显著超越两种基线。

## 研究问题与动机
1. **标准 token 级 OPD 的修正碎片化**：单次 rollout 中教师仅在当前位置提供单 token 指导，无法展开一条完整、连贯的正确修复路径，后续目标仍基于错误的学生前缀条件化。
2. **SFT 的 off-policy 状态分布偏差**：SFT 使用教师生成的完整序列进行 teacher-forcing，训练上下文来自教师数据而非当前学生，导致训练态与推理态不匹配。
3. **DAgger 类方法改变学生轨迹**：现有方法通过在 student 轨迹中插入教师响应实现长程监督，但这实质上改变了学生原本会产生的轨迹，削弱 on-policy 性质。
4. **黑盒蒸馏的实用性需求**：标准 OPD 需要访问教师 logits，难以直接应用于黑盒教师；SOPD 仅需教师生成文本，且每轮调用生成量接近一次完整响应，黑盒成本与 SFT 相当。

## 核心贡献（创新点）
1. **提出 SOPD 统一 SFT 与 OPD**：SOPD 保留完整学生轨迹的状态分布，同时在每个自然步骤上请求教师生成连贯的多 token 目标，本质区别在于教师目标仅在步骤内部自回归展开，不污染后续步骤的上下文。
2. **支持黑盒蒸馏且训练效率可控**：SOPD 不需要教师 logits，仅需教师生成文本；通过 packed attention mask 将多条 step-level 监督实例并行处理，教师调用可完全并行发出，无需等待环境交互。
3. **建立 SFT ↔ OPD 的插值理论联系**：SOPD 在单步极限下等价于序列 SFT，在单 token 极限下等价于 sampled forward-KL OPD，为两种范式提供了统一的视角。
4. **步级平衡损失设计**：先对每个教师目标内部 token 取平均，再跨所有步骤平均，避免长目标主导训练，并提供长度归一化的风险估计。
5. **在 agent 和推理任务上双重验证**：ALFWorld  Seen/Unseen 成功率分别提升 18.57/21.64 点，数学推理四个基准平均提升 9.9 点。

## 方法详解
- **步骤划分**：根据领域结构将学生轨迹 $\tau_S = (s_1, s_2, \ldots, s_K)$ 切分为自然步骤。数学推理使用空行段落边界或显式标题（如"Step"/"Solution"）截断，超过 1024 token 时强制截断；ALFWorld agent 中一个助手响应及其解析出的动作构成一步。
- **学生轨迹固定访问状态**：rollout 阶段完整记录每个步骤前的学生前缀 $h_k$，确保 $h_k \sim d_k^{\pi_{\theta_r}}$，教师目标不影响学生访问的状态分布。
- **教师目标独立生成**：对每个 $h_k$，教师独立生成目标 $\tilde{s}_k \sim q_T(\cdot \mid h_k)$，每个目标仅以原始 prompt 和 preceding student steps 为上下文，不引入早期教师文本，因此各目标条件独立，可并行生成。
- **步级平衡损失**：
  $$\mathcal{L}_{\mathrm{SOPD}}(B) = \frac{1}{|S_B|} \sum_{(h_k, \tilde{s}_k) \in S_B} \frac{1}{|\tilde{s}_k|} \sum_{j=1}^{|\tilde{s}_k|} -\log \pi_\theta(\tilde{s}_{k,j} \mid h_k, \tilde{s}_{k,<j})$$
  内层平均防止长目标主导，外层平均赋予每个步骤等权。
- **Packed Attention Mask 实现**：将输入 prompt、完整学生轨迹及各教师目标打包为单一序列；每条教师目标仅可 attend 自身的前缀和学生前缀，阻止其看到后续学生步骤和其他教师目标，实现单条 forward pass 内并行处理。
- **极限连接**：单步 ($K=1$) 时退化为序列 SFT；单 token 时 teacher 采样 cross-entropy 的期望等价于 student 访问前缀上的 forward-KL OPD。

## 实验与结果
- **ALFWorld（3B 学生 / 7B RL 教师）**：
  - Valid Seen：SOPD SR = **84.29%**（+18.57 vs Vanilla OPD 的 65.72%），交互轮次 11.20（↓3.53）。
  - Valid Unseen：SOPD SR = **82.09%**（+21.64 vs OPD 的 60.45%），轮次 11.88（↓4.33）。
  - Hard：SOPD SR = 10.74%，轮次 28.15（最短）。
  - SOPD Seen SR 超出 TCOD-F2B（81.43%）约 2.86 点。
- **数学推理（Qwen3-4B 学生/教师同规模）**：
  - AIME24：SOPD **71.8%**（+9.9 vs OPD 的 61.9%）。
  - AIME25：**67.4%**（+10.4）。
  - HMMT25-Feb：**38.9%**（+6.4）。
  - HMMT25-Nov：**52.7%**（+13.1）。
  - Avg₄ = **57.7** vs OPD 的 47.7（+9.9）。
- **训练动态**：随训练推进，学生生成的自然推理步骤数显著增加，每步教师监督 token 数从 update 1 的 29.0 增至 final 的 45.2，说明模型学会组织更多结构化推理步骤。

## 相关工作脉络
1. **Trajectory-Refined Distillation (TRD, Jiang et al., 2026)**：指出单次 rollout 的 token-wise OPD 只能提供碎片化修正，无法展开完整 counterfactual 修复路径；SOPD 与其本质区别在于不重写整个学生轨迹，而是保留完整轨迹并在每个自然步骤处插入独立的教师局部修正。
2. **On-policy Expert Correction (OEC, Lauffer et al., 2025)**：在轨迹中途切换专家控制，仅对奖励过滤后的专家后缀做 SFT；SOPD 的区别在于不改变轨迹结构，每一步都以学生实际访问的状态为锚点生成独立目标。
3. **TCOD (Wang et al., 2026)**：通过时间课程控制多轮轨迹中学生/教师的控制区间；SOPD 不涉及课程调度，而是按自然步骤顺序对所有学生访问状态进行并行教师查询。
4. **Guided-OPD (Li et al., 2026) / ReOPD (Liao et al., 2026)**：前者在每轮按课程混合师生动作并执行教师动作到环境中，后者回放预收集的教师前缀以避免重复交互；SOPD 的关键差异是教师目标从不进入环境执行，完全离线生成。
5. **Standard On-Policy Distillation (Agarwal et al., 2024)**：在 student 生成的每个 token 位置查询教师 logits 并最小化 reverse KL；SOPD 不依赖 logits 访问，且提供多 token 连贯目标而非单 token 密集反馈。
6. **ExOPD (Yang et al., 2026)**：利用 reward extrapolation 让学生超越单一教师；SOPD 不使用任何 reward 信号，纯通过 step-level teacher generation 实现 imitative distillation。

## 局限性与未来方向
- **步骤划分依赖启发式规则**：数学推理使用段落边界/标题检测，agent 使用环境 turn 界定，自动识别的鲁棒性有待验证；缺乏通用的自动化步骤边界检测方法。
- **黑盒蒸馏的单步生成质量瓶颈**：教师只生成单步目标时，若教师在该状态下质量不稳定，可能限制 SOPD 上限；long-horizon 一致性仍有损失。
- **教师调用量随步骤数线性增长**：虽然可并行，但步骤数增多会直接增加推理开销，在长轨迹场景下成本可能较高。
- **未探索与 RL/偏好优化方法的结合**：论文定位为纯蒸馏方法，未讨论与 GRPO、RLVR 等强化学习信号融合的潜力。
- **未测试更大规模模型**：实验仅在 3B/4B 级别验证，大模型下的缩放行为尚不明确。

## 研究启发与可借鉴点
1. **Packed attention mask 的通用设计**：将多个教师目标与对应学生前缀打包入单序列并通过精细注意力掩码隔离，可在其他需要多目标监督的场景（如多轮对话蒸馏、代码生成）中复用。
2. **Step balancing 损失的层次化归一化策略**：先内层 token 平均再外层步骤平均的设计可有效防止长序列主导梯度，对任何变长多目标训练任务都有借鉴价值。
3. **SFT ↔ OPD 的插值统一视角**：通过将步长作为连续超参数，可将不同粒度的监督信号纳入统一框架，为后续设计"粒度自适应"蒸馏器提供理论参考。
4. **黑盒蒸馏的实用性路径**：SOPD 仅需教师文本而非 logits，且每轮生成量接近单次 SFT 响应，为工业界使用闭源大模型（如 GPT-4/Claude）作为黑盒教师提供了低成本可行方案。
5. **可结合团队现有方向**：若团队关注多轮对话 agent 或复杂推理链的训练，SOPD 的自然步骤划分逻辑可直接迁移至对话 turn 或推理 chain 的结构化蒸馏中。

## 关键术语表
- **On-Policy Distillation (OPD)**：学生在自身生成的轨迹上查询教师分布进行学习，解决 off-policy 方法中训练态与推理态不匹配的问题。
- **Step-Level On-Policy Distillation (SOPD)**：将学生轨迹划分为自然步骤，在每个步骤前缀上请求教师生成独立目标，融合 SFT 长程监督与 OPD on-policy 状态覆盖。
- **Trajectory-Refined Distillation (TRD)**：通过重写完整学生轨迹构建修复路径，弥补 token-wise OPD 只提供碎片化修正的缺陷。
- **Step Balancing Loss**：先对每条教师目标内部 token 取交叉熵平均，再跨所有步骤取平均的损失设计，防止长目标主导训练。
- **Packed Attention Mask**：将多条 step-level 监督实例打包为单一序列，并通过注意力掩码隔离不同目标间的干扰，实现并行高效训练。
- **Black-box Distillation**：无需访问教师 logits，仅依赖教师生成文本完成蒸馏，适用于闭源模型场景。
- **Forward-KL OPD**：最小化教师分布对以学生分布的前向 KL 散度，SOPD 在单 token 极限下等价于此目标。
- **Natural Step Boundary**：根据领域结构定义的步骤分割点，如数学中的段落/标题边界或 agent 中的环境 turn 分界。

## 可复现要素
- **数据集**：ALFWorld（公开， embodied QA 交互环境）；数学推理使用过滤后的 DeepMath（difficulty ≥ 6，57,000 条）。
- **代码/权重**：论文声明训练和评估配置详见 Appendix A；提交源码包含图表生成脚本和聚合记录；教师模型使用 langfeng01/GiGPO-Qwen2.5-7B-Instruct-ALFWorld。
- **关键超参**：
  - 数学 SOPD：lr = $2 \times 10^{-6}$，10 轮更新，每轮 1,024 prompts，温度 1.0/top-p 1.0，教师贪婪解码，最大响应 32,768 token，步安全上限 1,024 token。
  - 数学 OPD：lr = $10^{-5}$，reverse KL。
  - ALFWorld SOPD：lr = $10^{-6}$，batch 64，max staleness 2，训练响应上限 512 token，prompt 上限 10,240 token，最多 30 turn。
  - 评估：AIME/HMMT 各 30 题 × 32 采样；ALFWorld 温度 0.4，上限 4,096 token。
