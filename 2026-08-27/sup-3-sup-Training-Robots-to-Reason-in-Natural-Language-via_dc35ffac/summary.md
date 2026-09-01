---
title: "sup-3-sup-Training-Robots-to-Reason-in-Natural-Language-via"
source: https://arxiv.org/pdf/2608.26053v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 19:01:16"
field: "具身智能与机器人操作"
keywords: ["robotic manipulation", "vision-language model", "chain-of-thought reasoning", "reinforcement learning", "imitation learning", "test-time compute"]
innovations: ["提出两阶段后训练R³，通过mid-training初始化+单步rubric-based RL从指令数据中训练VLM自由语言推理", "设计VLM-as-judge奖励使模型可从仅含expert instruction的离线数据中学习无需多步在线rollout", "提供因果证据表明测试时生成推理是性能增益来源而非仅提升表征"]
benchmarks: ["Language Table", "Bimanual Grocery Packing"]
---

# 论文速读：R³: Training Robots to Reason in Natural Language via Reinforcement Learning

## 一句话总结
本文提出 ℛ³，一种将现成VLM训练为机器人推理器的两阶段后训练方案：先通过中期训练在少量专家推理轨迹上初始化推理风格，再通过基于评分的单步RL从大量无推理标签的离线动作数据中进一步提升；在Language Table和双臂grocery packing上，ℛ³显著优于指令-only模仿学习基线，并证明自由形式语言推理可作测试时计算机制引导底层策略。

## 研究问题与动机
- 长周期机器人操作需要跟踪部分进展、推理物体关系、从错误中恢复、并引导有噪声的底层策略，但现有方法主要把结构化CoT仅作为训练时监督信号，测试时推理带来的额外收益有限。
- 自由形式自然语言推理是否能作为“测试时计算”机制来提升机器人操纵的泛化与性能尚不清楚。
- 现有具身推理工作多依赖手工设计的结构化中间表示（如边界框、坐标、子目标图像等），缺乏对VLM本身在真实操作中生成灵活推理并指导动作的研究。
- 数据获取的现实约束：高质量推理轨迹昂贵且稀疏，而仅含专家指令的动作数据更易获取，需要设计能在“推理标注稀少 + 指令标注丰富”条件下生效的训练范式。

## 核心贡献（创新点）
- 提出 ℛ³ 两阶段后训练流程：先mid-train于专家推理轨迹以初始化推理风格，再用单步rubric-based RL在更广的仅含指令的离线数据上改进，本质区别于仅用结构化CoT做辅助训练监督的做法。
- 设计 rubric-based VLM-as-a-judge 奖励机制，使模型可从仅含 expert instruction 的离线数据中学习，避免昂贵的多步在线rollout与长期信用分配。
- 在 Language Table 与双臂 grocery packing 两个受控benchmark上系统验证，证明自由形式语言推理能在seen与OOD任务上提升exploration与generalization，并显著优于instruction-only imitation learning。
- 提供因果证据（VQA诊断、非推理策略对照、推理预算截断实验）说明推理不仅是表征学习的副产品，更是可被调度用于提升泛化的测试时compute。
- 开源项目页与实验细节（含ECoT对照、judge校验、消融），便于后续研究与对比。

## 方法详解
- **分层架构**：高层VLM πθ(z_t, u_t | x_t, g) 生成自由形式推理 z_t 与短周期指令 u_t；低层固定策略 π_lo(a_t | s_t, u_t) 执行动作块。
- **Stage I — Mid-training (SFT)**：用专家reasoner（Gemini 3 Flash）在任务上收集的 multi-turn 轨迹（含推理 z_t 与指令 u_t）做标准 next-token prediction；同时使用成功与失败轨迹以覆盖 partial progress、mistake、recovery 等模式；目标是初始化 manipulation-oriented 的推理风格（约束跟踪、自我修正、空间推理）。
- **Stage II — Single-step Rubric-based RL**：在仅含 expert instruction u_t* 与 history x_t 的离线数据上，采样 (z_t, u_t) ~ πθ，由 rubric-based VLM judge 给出标量奖励 R；采用 Dr.GRPO 优化，优势 A^(k)=R^(k) − mean(R)。
- **奖励设计**：奖励 = 指令准确性奖励 + 长度惩罚（鼓励足够长的推理）；Language Table 用 Qwen3.5-35B-A3B 作为 judge，按 1.0/0.5/0.25/0.0 四级评分（精确匹配/副词不匹配/语义匹配/不匹配）。
- **历史上下文**：以上一轮完整响应（z_{t-1}, u_{t-1}）作为 interaction history，显著提升 expert pass@1。
- **RL 关键工程细节**：
  - Reasoning context imputation：RL 数据缺少历史推理，从中期训练模型采样 48 个响应，若其上一轮指令与 expert 一致则复用，否则仅保留上一指令。
  - 过滤重复步骤：避免模型在RL中利用重复指令作为 reward shortcut。
  - Grocery packing 使用 exact string-match 奖励（指令空间有限）。
- **ECoT 对照实现**：在 Reasoning 前插入 end-effector 与 object 坐标，并对比 “online-collected” vs “post-hoc” 推理生成；实验表明结构化 ECoT 在本设定下未见增益。

## 实验与结果
- **数据集与环境**：
  - Language Table：14 类长周期 block 排列任务；训练分 T_M（6个 mid-training 任务）、T_R（3个 RL 任务）、T_O（5个 OOD 评估任务）。
  - Bimanual grocery packing：12 个 unseen 任务，双 xArm-7 + MuJoCo。
- **模型与基线**：Base Qwen3.5-4B；Instruction-only IL（mid-only / full）；ℛ³ 多_variant（mid only、RL only、1/4 mid）；ECoT 变体。
- **Language Table 主要数字（Table 2）**：
  - T_M 中分布任务：group 65.8%（ℛ³） vs IL 64.7%；V 69.2% vs IL 40.9%；clear_qtr 93.8% vs IL 91.8%。
  - T_R：iV 57.5% vs IL 38.9%；gris 47.8% vs IL 58.1%（此处 IL 较强但 ℛ³ 整体OOD更强）。
  - T_O 外分布任务：diag_line 30.9% vs IL 16.7%；mid 51.0% vs IL 42.3%；iL 37.2% vs IL 27.3%；clear_half 74.5% vs IL 69.7%。
- **Grocery packing 主要数字（Table 5）**：Mean success 47.9%（ℛ³ RL only） vs 38.0%（IL w/o reason）；Mean progress 73.1% vs 65.4%。
- **推理预算消融（Table 3）**：截断到 50/100 tokens 会显著下降（如 group 65.8→53.1/60.9），去除推理则下降更多（group 65.8→39.8），证明 test-time 推理直接贡献性能。
- **ECoT 对照（Table 4）**：加入结构化 ECoT 组件普遍小幅下降或持平，支持自由形式推理更适合长周期、需闭环 replanning 的任务。
- **结论**：ℛ³ 在 in-distribution 与 OOD 任务上均优于 instruction-only IL；mid-training 提供强 prior，RL 在此基础上精修行为分布。

## 相关工作脉络
- ECoT / CoT-VLA 等：以结构化视觉/坐标/子目标作为推理中间表示并用于训练；本文证明在长周期封闭-loop 操控中自由形式自然语言推理更重要。
- SteerVLA、Inner Monologue、SARL 等：以语义接口或语言命令引导通用 VLA；本文聚焦训练 VLM 自身生成自由推理以作为测试时 compute。
- RT-2、Octo、OpenVLA、GR00T、Gemini Robotics 等：主流 VLA 骨干未显式引入 reasoning；本文与其在“是否训练推理”上定位不同。
- Chen et al. [8]：论证结构化 CoT 收益主要来自训练时监督；本文通过截断/对比实验进一步说明测试时推理本身仍有独立增益。
- 任务与运动规划（TAMP）、visual foresight、latent planning 等：采用符号/预测/潜变量结构；本文用 VLM 端到端学习自然语言推理并以层级冻结低层策略解耦评估推理贡献。
- Refletion/WebAgent-r1 等交互式 agent 的 online RL；本文采用单步离线 RL 规避昂贵多步 rollouts，作为向 online 延伸的前置工作。

## 局限性与未来方向
- 仅在两种模拟环境验证，未在真实机器人/高噪声感知下检验鲁棒性。
- Stage II 使用 VLM-as-judge 评估语义一致性，优化的是 surrogate objective 而非最终任务成功；可能引入 judge 偏差。
- 数据生成仍依赖 Gemini 作为 expert reasoner；post-hoc 生成与在线收集的相近表现可能受限于同模型生成，真实 human policy → 不同 reasoner 的情形待验证。
- 高层推理与低层策略分离可能带来 mismatch；未来可探索联合训练与同步机制。
- 单步离线 RL 无法直接利用任务完成/中间进展/恢复行为的长期反馈。

## 研究启发与可借鉴点
- **两阶段“SFT 初始化 + 单步 RL 精炼”**的训练范式值得迁移：在“推理标注稀缺、指令标注充足”的数据预算下具有实用价值。
- **Rubric-based VLM judge 替代字符串匹配**的奖励设计，适用于开放指令空间（non-finite）的任务，可推广至多模态/开放域策略训练。
- **History as last full response** 的设计简洁有效，能在长周期任务中维持计划连贯性；可推广至其他需要跨步状态跟踪的具身/agent 任务。
- **截断推理预算的消融**提供强有力的因果论证思路：在固定 checkpoint 上只改变 test-time 计算量，可直接归因推理贡献。
- **Context imputation + 去重**的工程技巧对 RL 稳定训练具有参考价值；尤其是避免 reward hacking 的过滤策略。
- 与本团队潜在结合：可将此框架应用于需要推理规划的视觉导航/装配/抓取流水线，或以本团队的预训练 VLM 骨干替换 Qwen3.5-4B 以检验可扩展性。

## 关键术语表
- **ℛ³**：一种两阶段后训练方法，将现成 VLM 训练为可在操作中生成自由语言推理并指导低层策略的高层 reasoner。
- **Mid-training**：在正式 RL 之前，用专家推理轨迹对预训练 VLM 做 SFT，使其获得任务导向的推理行为先验。
- **Single-step RL**：在当前观察与历史条件下采样推理与指令，以 rubric-based 奖励即时评估并更新策略的离线强化学习 formulation。
- **VLM-as-a-judge**：用更大 VLM 根据 rubric 判断候选指令与专家指令的语义一致性并给出标量奖励。
- **Test-time compute**：模型在推理阶段通过额外生成推理轨迹来消耗计算，以提升难问题性能。
- **ECoT（Embodied Chain-of-Thought）**：在推理链中注入视觉/坐标等结构化地物信息以增强 grounding 的方法。
- **Reasoning context imputation**：在 RL 阶段用中期训练模型补齐历史推理，以维持连贯的交互上下文。
- **Dr.GRPO**：用于群体采样与优势归一化的策略梯度优化算法。

## 可复现要素
- 数据集：Language Table（公开 benchmark）；双臂 grocery packing 数据集引用匿名工作 [2]（preparation manuscript），需进一步确认开源状态。
- 代码/权重：项目页 https://robotic-reasoner.github.io/；模型使用 Qwen3.5-4B（开源）与 Qwen3.5-35B-A3B judge。
- 关键超参：SFT lr 1e-6、2 epochs、batch 128、bf16、8 GPU；RL lr 2e-6、4/8 epochs、batch 32、rollouts per prompt 12、temperature 1.0、clip 0.2/0.3、max response 1024、length penalty 阈值 T=80（论文未逐一列出全部细节，见附录 Table 8/9）。
