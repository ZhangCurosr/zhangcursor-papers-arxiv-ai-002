---
title: "STEP-LEVEL-ON-POLICY-DISTILLATION-INTERPOLATING-BETWEEN-ON-P"
source: https://arxiv.org/pdf/2608.16333v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:11:01"
field: "大语言模型蒸馏与训练"
keywords: ["on-policy distillation", "knowledge distillation", "SFT", "agent", "mathematical reasoning", "black-box distillation"]
innovations: ["提出 Step-Level On-Policy Distillation (SOPD)，在自然步骤粒度上提供连贯的多 token 教师纠正", "建立 SOPD 与 SFT/OPD 的理论连接，形成监督粒度的统一视角", "设计 packed attention mask 实现多 step-level 监督实例的并行高效训练"]
benchmarks: ["ALFWorld", "AIME24", "AIME25", "HMMT25-Feb", "HMMT25-Nov", "DeepMath"]
---

# 论文速读：STEP-LEVEL-ON-POLICY-DISTILLATION-INTERPOLATING-BETWEEN-ON-P

## 一句话总结
本文提出 Step-Level On-Policy Distillation (SOPD)，通过将教师监督粒度从 token 级扩展到自然步骤级，在保留 OPD on-policy 状态覆盖的同时提供连贯的多 token 局部纠正，在 ALFWorld agent 交互和数学推理任务上均显著超越传统 SFT 和 OPD。

## 研究问题与动机
1. 标准 token-level OPD 只能提供片段化纠正：学生在错误轨迹上生成时，教师仅在每个位置给出单个 token 指导，无法展开完整正确的纠正路径。
2. 标准 OPD 后续 token 的条件仍基于原始错误前缀，而非包含早期教师纠正的新前缀，导致纠正信号不连贯。
3. DAgger 类方法虽保留 SFT 的长程监督，但会将教师响应插入学生轨迹，显著改变学生原本会产生的轨迹状态分布。
4. 现有方法难以同时获得 on-policy 训练状态覆盖和连贯的多 token 局部纠正。

## 核心贡献（创新点）
1. 提出 SOPD 方法，将 OPD 的 on-policy 属性与 SFT 的长程监督相结合；与标准 OPD 不同，SOPD 仅需教师生成的文本响应而非 logits，天然支持黑盒蒸馏。
2. 建立 SOPD 与 SFT/OPD 的理论连接：步长趋近整个响应时退化为序列 SFT，步长趋近单个 token 时近似为 forward-KL OPD，为两种范式提供了统一视角。
3. 设计 packed attention mask 实现高效并行训练：将 prompt、学生轨迹和多个 step-level 教师目标打包进单个序列，单次前向传播处理所有监督实例。
4. 在 ALFWorld 和数学推理两个不同域验证有效性：ALFWorld Seen 成功率从 65.72% 提升至 84.29%（+18.57 点），数学推理四基准平均准确率从 47.7% 提升至 57.7%（+10 点）。

## 方法详解
1. **学生轨迹生成与步骤划分**：学生 π_θr 生成完整轨迹 τ_S = (s_1, s_2, ..., s_K)，s_k 为第 k 个自然步骤，h_k 为该步骤前的学生前缀（静态推理中 h_k = (x, s_<k)，交互环境中还包含环境返回的观察）。数学任务中以 blank-line 段落边界或 "Step"/"Solution" heading 划分，ALFWorld 中以每个 assistant response + parsed action 为一个 step，最大 1,024 token 安全上限。
2. **教师目标生成**：对每个记录的前缀 h_k，独立生成教师目标 s̃_k ~ q_T(·|h_k)。每个查询仅包含原始 prompt 和前面学生步骤，不包含之前教师目标，确保目标与学生实际访问状态对齐且目标间条件独立。
3. **步骤平衡损失函数**：
   L_SOPD(B) = (1/|S_B|) Σ_{(h_k, s̃_k) ∈ S_B} (1/|s̃_k|) Σ_{j=1}^{|s̃_k|} -log π_θ(s̃_{k,j} | h_k, s̃_{k,<j})
   内层平均防长步骤主导，外层平均赋各步骤等权；轨迹步骤越多贡献越大。
4. **Packed Attention Mask**：将 prompt、学生步骤和教师目标打包入单序列，目标 s̃_k 只能 attend 到 x、s_<k 及其自身因果前缀，屏蔽 s_≥k 和其他教师目标，loss label 仅覆盖教师 token。
5. **极限连接**：K=1 时退化为序列 teacher-forcing SFT 损失；step length=1 token 时取期望等价于 student-visited prefix 上的 sampled forward-KL 目标。

## 实验与结果
**ALFWorld Agent 交互**（Qwen2.5-3B 学生，Qwen2.5-7B-RL 教师）：
- 数据集：140 Valid Seen / 134 Valid Unseen / 121 Hard tasks
- SOPD Seen SR=84.29%（Vanilla OPD 65.72%，+18.57 点；TCOD-F2B 81.43%，+2.86 点），Unseen SR=82.09%（OPD 60.45%，+21.64 点），Hard SR=10.74%
- SOPD 同时获得最少交互轮数（Seen 11.20 轮 vs OPD 14.73 轮）
- 最强结果：Unseen 成功率 82.09%，比 Vanilla OPD 提升 21.64 点，比 TCOD-F2B 提升 2.90 点

**数学推理**（Qwen3-4B 学生/教师同规模对比）：
- 数据集：Filtered DeepMath（57,000 examples，difficulty ≥ 6），GRPO 训练教师
- 基准：AIME24 / AIME25 / HMMT25-Feb / HMMT25-Nov
- SOPD 准确率：71.8% / 67.4% / 38.9% / 52.7%，Avg_4=57.7%
- 相对 OPD 提升：AIME24 +9.9 点，AIME25 +10.4 点，HMMT25-Feb +6.4 点，HMMT25-Nov +13.1 点
- 最强结果：HMMT25-Nov 达 52.7%，比 OPD 提升 13.1 点；Avg_4 比 OPD 提升 10 点

## 相关工作脉络
1. **Off-policy sequence distillation (Kim & Rush, 2016)**：在教师生成序列上训练；SOPD 保留学生诱导前缀，用局部教师生成替代，实现 on-policy 状态覆盖。
2. **Standard OPD (Agarwal et al., 2024)**：在 student-generated tokens 上查询教师 logits 最小化 token-level divergence；SOPD 用教师生成文本替代 logit 访问，支持黑盒蒸馏且提供多 token 连贯纠正。
3. **TRD (Jiang et al., 2026)**：指出单 rollout 监督不含完整反事实纠正轨迹，通过重写学生轨迹构建纠正路径；SOPD 保留完整学生轨迹，在每个自然步骤独立生成局部纠正。
4. **TCOD (Wang et al., 2026)**：通过时间课程分配多轮轨迹控制区间；SOPD 保持自然时间顺序，查询每个学生访问步骤，从不执行教师目标。
5. **OEC (Lauffer et al., 2025)**：从学生 rollout 中途切换专家并对 reward-filtered 专家后缀应用 SFT；SOPD 提供完整学生轨迹上每个步骤的独立监督。
6. **ExOPD (Yang et al., 2026)**：使用奖励外推使学生超越单一教师；SOPD 不使用任何奖励信号，纯基于教师文本监督。

## 局限性与未来方向
1. 步骤划分依赖领域特定的自然边界规则（数学段落/heading、ALFWorld 环境 turn），需人工设计或启发式，泛化到无天然边界的任务（如自由对话）存疑。
2. 仅在 agent 交互和数学推理两个域验证，代码生成、多轮对话等其他长程推理场景的适用性待探索。
3. 教师目标通过 safety cap 截断（如 1,024 tokens），可能丢失步骤后半段的有用信息。
4. 黑盒蒸馏仅依赖教师生成文本而非完整分布，无法完全捕捉教师的不确定性，理论上不如 logit-level OPD 信息丰富。

## 研究启发与可借鉴点
1. **Step-level 监督思想可跨域迁移**：对多轮对话、代码生成、规划等需要分段推理的任务，通过合理定义"步骤"边界即可获得连贯局部纠正，无需修改核心框架。
2. **Packed attention mask 的高效训练设计**值得直接复用：将多个监督实例打包到单序列并行处理，显著提升 GPU 利用率，适合各类序列到序列蒸馏场景。
3. **长度归一化的 step-balanced loss** 避免了长目标主导训练，该技巧可推广到其他序列生成任务的监督信号加权，如对话系统或工具调用轨迹。
4. **SOPD 与 SFT/OPD 的粒度统一视角**启发了从"监督步长"维度重新审视蒸馏方法谱系，未来可在此连续统上自动搜索最优步长。
5. **教师生成并行化**：由于各教师目标条件独立，可在学生轨迹完成后批量并行生成，大幅降低黑盒蒸馏的延迟成本，适合实际部署。

## 关键术语表
**On-policy Distillation (OPD)**：让学生在自己生成的轨迹状态上学习教师监督信号，确保训练状态分布与推理时一致。
**Step-Level On-Policy Distillation (SOPD)**：本文方法，将学生轨迹划分为自然步骤，在每个步骤前缀处独立请求教师生成局部纠正目标。
**Trajectory-Refined Distillation (TRD)**：指出标准 OPD 只能提供片段化纠正的局限，通过重写完整学生轨迹构建连贯纠正路径。
**Packed Attention Mask**：将多个监督实例打包进单序列并通过注意力掩码隔离的并行训练机制。
**Step Balancing Loss**：先对每个教师目标内 token 平均交叉熵再对所有步骤平均的损失设计，防止长步骤主导。
**Forward KL Distillation**：最小化教师分布对学生分布的前向 KL 散度，使学生对教师分布做均值拟合。
**Black-box Distillation**：仅需访问教师文本输出而无需 logit 或内部状态的蒸馏方式。
**Natural Step Boundary**：按任务语义定义的步骤分割点（如数学段落边界、agent 环境 turn）。

## 可复现要素
- **数据集**：DeepMath（filtered, 57,000 examples, difficulty ≥ 6）用于数学推理；ALFWorld（140 Valid Seen + 134 Valid Unseen + 121 Hard）用于 agent 任务。论文未明确声明公开，但 submission source 包含 figure-generation scripts 和 aggregated records。
- **代码/权重**：论文未明确提及代码开源；模型使用 Qwen2.5-3B-Instruct、Qwen2.5-7B-RL（langfeng01/GiGPO-Qwen2.5-7B-Instruct-ALFWorld）、Qwen3-4B non-thinking 及对应 RL 教师。
- **关键超参**：
  - 数学 SOPD：batch=1,024 prompts/update，10 updates，lr=2e-6，temperature=1.0，top-p=1.0，max_response=32,768 tokens，teacher greedy decoding；步骤边界：blank-line / "Step"/"Solution" heading，1,024-token 安全上限
  - 数学 OPD：lr=1e-5，sampled-token reverse KL
  - ALFWorld SOPD：Explorer 16 tasks/batch，Trainer 64 experiences/batch，lr=1e-6，temperature=1.0，staleness=2，max 30 environment turns，response cap=512 tokens（训练）/4,096 tokens（评估）
