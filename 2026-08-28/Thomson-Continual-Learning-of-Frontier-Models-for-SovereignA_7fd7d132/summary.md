---
title: "Thomson-Continual-Learning-of-Frontier-Models-for-SovereignA"
source: https://arxiv.org/pdf/2608.27147v1.pdf
model: agnes-2.5-flash
chunks: 8
summarized_at: "2026-09-02 12:09:22"
field: "大语言模型持续学习与对齐"
keywords: ["Continual Learning", "Sovereign AI", "Model Alignment", "Large Language Models", "Legal AI", "Test-Time Scaling", "Data-Centric AI"]
innovations: ["提出基于价值重对齐、持续预训练、多阶段行为训练的持续学习框架以低成本实现主权AI", "设计Fisher-Routed Directional Ablation实现低KL成本的行为编辑对齐", "构建数据中心化管线与模型合并策略实现π形性能提升与能力保持"]
benchmarks: ["CapTrack", "Scales++", "Stanford LegalBench", "COLIEE Task 1", "AIME 2026", "GPQA-Diamond", "SWE-bench Pro", "Humanity's Last Exam", "Promptfoo Safety Benchmark"]
---

# 论文速读：Thomson-Continual-Learning-of-Frontier-Models-for-SovereignAI

## 一句话总结
论文展示了通过**持续学习（Continual Learning）**范式，以约 **4000 万美元** 总成本和 **3周** 训练时间，在开放权重基线模型上实现了接近甚至局部超越多个闭源旗舰模型的 π 形性能提升，为**主权 AI（SovereignAI）** 开发提供了一套低成本、可复现的“模型工厂”蓝图。

## 研究问题与动机
1.  **主权 AI 的算力与成本门槛过高**：实现前沿模型性能传统上依赖从头预训练的巨额资金（约数百万美元级）和算力，限制了学术机构、中小企业等广泛主体。
2.  **开放权重模型难以触及前沿**：尽管开放模型易于获取和定制，但其基线性能与顶级闭源系统（如 Opus, Gemini, GPT 系列）存在显著差距。
3.  **持续学习易导致灾难性遗忘**：针对特定领域（如法律）的微调或对齐训练，通常会损害模型的通用能力（如编程、数学、知识检索）。
4.  **缺乏可复现的系统性开发路径**：现有研究多聚焦单一技术点（如 DPO、RL），缺乏将数据治理、价值对齐、持续预训练、复杂任务训练端到端整合并验证的完整 pipeline。

## 核心贡献（创新点）
1.  **提出并验证了“主权 AI 持续学习框架”**：首次系统性地展示了通过**价值重对齐、持续预训练、两阶段行为/技能训练**的组合，能以极低成本在开放模型上复现前沿性能，打破了“必须从头预训练”的依赖。
2.  **设计了 Fisher-Routed Directional Ablation 与 Constitutional DPO 结合的价值重对齐方法**：通过将编辑方向路由到低敏感坐标（基于 Fisher 信息）并最小化对其它行为的干扰，大幅降低了重新对齐过程中的 KL 散度成本（降低 2.0×–5.0×），相比标准 abliteration 方法更优雅地剥离有害行为。
3.  **构建了数据中心化的持续预训练（CPT）管线与模型合并策略**：从超大规模语料池（>19T tokens）中精心筛选出微调数据集（200B tokens），并通过“激进 CPT + 保守合并比例”的策略，在提升目标领域能力的同时，有效保护甚至恢复通用能力，实现了 π 形性能模式。
4.  **开发并开源了多代理 Deep Research Harness 及其训练机制**：改编 DeerFlow，设计了规划器 - 工作者 - 报告器的拓扑结构，并提出了将单代理 RL 学到的改进泛化到多代理生产环境的训练混合物（off-policy 前缀 + on-policy 续接），关键维度（如事实性、完整性）获得提升。
5.  **提供了详尽的训练计算预算分解与基础设施工程方案**：公布了完整的 FLOP 计算（~4.6×10^23 FLOP）、硬件配置（最多 368×B200 GPU）、以及包含三级缓存、非共置训练/生成架构、推理加速等在内的完整工程实践，提高了研究的可复现性。

## 方法详解
1.  **价值重对齐（Value Re-alignment）**：
    *   **Constitutional DPO**：基于公共 AI 宪法（Public AI Constitution），使用人类专家构建的偏好对（Human Expert Pairs）和由多个开源模型集成改写生成的合成对（Synthetic Pairs）进行 1 epoch 的长度归一化 DPO 训练。联合搜索最优的 $\beta$, 学习率, NLL 权重和数据混合比例，以最大化重对齐得分并约束一般能力损失。
    *   **Fisher-Routed Directional Ablation**：用于抑制特定的 misaligned 行为。核心是 rank-one weight edit ($ \Delta W = b a^\top $)。方向 $v$ 由敏感集与中性集在最终 prompt token 处的平均残差流激活差确定。写入向量 $b$ 由两部分组成：沿 $v$ 方向的纠正项（$-\lambda v$）和一个正交于 $v$ 的分量 $b_\perp$，后者通过最小化对其他能力的影响来确定。利用对角 Fisher 信息矩阵估计，以 ridge fraction $\rho = 0.05$ 将修正导向低敏感坐标。每次 trial 通过 LoRA adapter 实现。
2.  **持续预训练（Continual Pre-training / Mid-Training）**：
    *   **数据中心化子选择**：从 >19T tokens 原始语料中，经清洗、精确去重、长度过滤（cap 256k tokens）、防污染（13-gram overlap）、质量过滤、任务/用例分布匹配、合成数据生成等步骤，精选出 200B tokens 的数据集。包含精心构建的 replay 数据（用于能力恢复）和基于 BeyondWeb 改进的合成数据（覆盖六大核心能力）。
    *   **训练与模型合并**：使用 seq len 8,192，global batch 512，inverse square root decay 学习率调度。通过 sweeps LR 和 token budget 定义 CPT strength，并与 pre-CPT checkpoint 进行线性插值合并（merge ratio 从 1:9 到 10:0）。结论是**激进 CPT + 保守合并比例**能在领域和通用能力双轴上同时达到最优。
3.  **行为/技能/工具训练**：
    *   **两阶段 DPO**：第一阶段进行广义偏好对齐；第二阶段针对智能体专项能力进行偏好训练。
    *   **两阶段 RL**：
        *   **Stage 1（短上下文多样化查询）**：在多代理 harness 记录的合法中间状态前缀上分别训练 planner, agent, compaction, reporter。优势在于 off-policy 前缀提供更广的状态覆盖和样本复用，on-policy 续接提供直接策略改进。重点优化 Claim-Entailment Composition 等具体技能，奖励曲线显示 factuality 和 completeness 提升最显著。
        *   **Stage 2（长上下文端到端 Deep Research）**：使用改编的 DeerFlow 多代理拓扑，进行端到端训练。测试时扩展（Test-Time Scaling）采用 Fusion 策略（N=4，3候选+1次融合）显著优于 reward-model selection，能回收约 40% 的可用 headroom。
4.  **训练基础设施与性能优化**：
    *   **硬件与并行**：基于 Nvidia B200 GPU (8×/节点) 和 InfiniBand/NVLink/NVSwitch 网络。使用 NeMo-RL + Ray 集群调度，模型并行基于 Megatron-LM，生成基于 vLLM。采用 Non-colocation 架构，生成与训练节点分离，减少资源竞争。
    *   **训练栈定制**：解决了 GRPO 采样多样性（唯一 seed）、Advantage Grouping（仅追踪首条 user message）、长度归一化 train-inference ratio（masked mean）等问题。实现了 Cohort Reward Reconciliation 以处理 judge 超时/限流导致的奖励缺失。
    *   **多级文档缓存**：进程内 (LRU, zstd 压缩)、共享文件系统 (按司法管辖区和 GUID 分片)、跨运行归档，命中率高达 98.2%。
    *   **推理加速**：Kubernetes 编排，跨地域/云的多集群 Service Mesh 路由，支持 Thinking Budget 和动态 post-training 量化（FP8）。

## 实验与结果
1.  **评测基准**：使用 CapTrack 元基准跟踪能力漂移（CAN/WILL/HOW 三维度），Scales++ 用于语义去重/剪枝开发评估集。全面基准包括：AIME 2026, GPQA-Diamond, Humanity's Last Exam, IFEval, SimpleQA-Verified, SWE-bench Pro, Terminal-Bench 2.1, WritingBench, FaithEval, GDPval, tau2-bench, Stanford LegalBench, COLIEE Task 1, 以及内部构建的 ContractScrub, InsuficiencyBench 等。
2.  **主要性能结果**：
    *   **Thomson-1.0-Large vs 竞品**：Overall Avg. **78.5**，仅次于 Opus 4.8 (79.5)，超越 Gemini 3.1 Pro (78.0), GLM-5.2 (77.2), GPT-5.4 (76.5), DeepSeek-V4 Pro (71.9) 等。法律领域平均 **78.4**，与 Opus 持平。Deep Research (88.9), Human Queries (89.2), Summarisation (90.0) 表现突出。政治中立性 **93.4**。轻微遗忘体现在 Coding (39.9 vs 基线 57.4) 和 Terminal-Bench 2.1 (54.0→48.3) 上。
    *   **Thomson-1.0-Small (35B) vs 小模型竞品**：Overall Avg. **74.6**，超越 Snowdon 1.1-Small (71.7), Gemma4 31B (71.2), Qwen3.6-35B (71.7), Haiku 4.5 (68.2)。政治中立性 **98.5** 显著领先。
    *   **能力保持与提升**：相比基线 Qwen3.5-397B，Humanity's Last Exam (+3.7pp), GDPval (+4.1pp), WritingBench (+2.4pp) 得到提升。SWE-bench Pro 基本持平 (33.8 vs 33.2)。
    *   **系统级盲评**：在 3,035 条专家任务（84% 单轮/16% 多轮）的盲评中，Thomson-1.0-Large 在 53-62% 的比较中胜出，对手仅 29-33%。法律查询胜率尤其高（对 GPT-5.5/Opus 约 2/3）。
3.  **安全与对齐**：在 Promptfoo 安全红队基准（11,167 个案例）上，Thomson-1.0-Large 在 Reasoning 模式下达到 **93.35%** 安全得分，是 Delta 提升最大的模型（+3.18pp）。在所有 7 个风险类别中均优于或持平 DeepSeek/GPT-5.6，但总体仍低于 Qwen3.5 基线 (93.55%)。
4.  **测试时扩展**：Fusion (N=4) 为 Thomson-1.0-Large 带来显著增益（Tax Eval v3 +8.77），且 Single-pass Thomson (71.9 macro) 优于 OOB Qwen3.5 + Fusion (71.2)，推理成本仅为后者的 1/4。
5.  **成本与效率**：总开发成本 ≈ **USD 40M**（人力+算力+专家+合作），训练时长 **3 周**，最终训练成本保守估计 **< USD 450,000**。总计算量 ~**4.6×10^23 FLOP** (≈ 8.8×10^4 GPU-hours)。

## 相关工作脉络
1.  **Constitutional AI [5,6]**：本文借鉴了基于宪法进行 AI 对齐的思想，但将其具体化为用于 DPO 训练的数据构建指南和 LLM judge 标准，并与 Fisher-Routed Directional Ablation 技术结合以实现更低 KL 成本的重对齐。
2.  **Abliteration [1]**：本文使用的 Fisher-Routed Directional Ablation 是对 abliteration 方法的改进，核心区别在于引入了 Fisher 信息来路由编辑方向，寻找对非目标行为影响最小的正交通道，从而降低对齐过程中的能力损失。
3.  **DPO [30] & RL [31]**：本文的两阶段 DPO 和两阶段 RL 是对标准偏好对齐和强化学习框架的工程化扩展和应用，重点解决了多代理环境下的训练稳定性、奖励稀疏性、以及从单代理到多代理的泛化问题。
4.  **DeerFlow [26] / LangGraph [27] / ReAct [29]**：本文的 Deep Research Harness 借鉴了这些多代理框架的设计思想（规划-执行-报告拓扑），但进行了定制化改造以适应法律和研究场景，并设计了专门的训练混合物来优化其组件。
5.  **Open-Source Frontier Models (Qwen3.5/3.6, Gemma4)**：本文以这些先进的开放权重模型作为基线，证明了通过持续的、受控的 finetuning 和 alignment 过程，可以显著缩小与闭源旗舰模型的性能差距，挑战了“必须从头预训练”的假设。
6.  **DatologyAI 等数据-centric 合作**：与 DatologyAI 的合作体现了数据-centric AI 的方法论，即通过系统性的数据筛选、去重、合成和质量控制（而非单纯增加数据量）来提升微调效果，这是与很多仅依赖扩大预训练数据规模的工作不同的定位。

## 局限性与未来方向
1.  **代码与权重开源程度**：论文表明“部分代码和模型权重已开放”，但未明确说明所有组件（尤其是训练基础设施、奖励函数、评估 harness）是否完全开源，可能影响完全复现性。
2.  **安全评估的局限性**：安全红队评估使用了 LLM-as-judge (GLM-5.2) 且未进行人工校准，可能引入裁判偏差。同时，使用的非自适应攻击套件可能无法捕捉所有真实世界的高级越狱策略。
3.  **领域泛化性待进一步验证**：虽然声称方法具有“泛化性”，但主要验证集中在法律和研究领域。对于其他专业领域（如医疗、金融、科学）的持续学习 pipeline 效果和潜在风险尚需更多实证。
4.  **对昂贵硬件的依赖**：尽管成本大幅降低，但仍需数百块高端 B200 GPU，这对于资源极其有限的机构来说仍是门槛。如何进一步降低硬件需求是未来方向。
5.  **合成数据的质量与幻觉**：持续预训练中使用的合成数据（基于 BeyondWeb 改进）可能存在潜在幻觉或偏差，如何更好地抑制而非传播幻觉是未来的改进点。
6.  **多文档推理能力**：论文在合成数据部分提到“未来工作”包括改进多文档推理能力，暗示当前版本在此方面可能存在不足。

## 研究启发与可借鉴点
1.  **“模型工厂”流水线思维**：将主权 AI 开发分解为**价值重对齐 -> 持续预训练 -> 行为/工具训练**的标准化、可复现阶段，并为每个阶段设计了具体的技术选型和评估闭环，这种工程化思维对构建可定制 AI 系统非常有借鉴价值。
2.  **Fisher-Routed 编辑作为低 KL 成本对齐技术**：将对齐干预（如抑制有害行为）与模型参数空间的敏感性（Fisher 信息）相结合，通过正交通道写入编辑，是一种值得深入研究和应用于其他对齐场景的优雅技术。
3.  **数据中心化与 Replay 策略**：从海量候选数据中通过严格流程筛选出小比例高质量数据用于持续预训练，并精心构建 replay 数据集以对抗遗忘，这为在资源受限下进行高效 finetuning 提供了重要参考。
4.  **多代理环境的训练泛化机制**：提出的“off-policy 前缀 + on-policy 续接”的训练混合物，以及将单代理 RL checkpoint 迁移到多代理 harness 的经验，为解决复杂 agent 系统的训练-部署 gap 提供了思路。
5.  **测试时扩展（Fusion）的实用价值**：证明了简单的 Fusion 策略在保持低推理成本的同时，能显著提升复杂任务（如法律研究）的最终表现，且优于复杂的 reward-model selection，这是一种高性价比的性能提升技巧。
6.  **专家协作与 IAA 提升流程**：论文附录详细描述了如何通过 LLM 辅助筛查评估指南脆弱性、运行校准轮次、有目的地构建校准批次等流程来提升专家标注的一致性，这对于依赖人类反馈进行对齐的研究极具参考价值。

## 关键术语表
*   **SovereignAI**：主权 AI，指一个国家、机构或社区能够自主控制其 AI 模型的开发、部署、数据和使用，而不依赖单一外部供应商的技术路线。
*   **Continual Learning (CL)**：持续学习，指模型在不遗忘先前所学知识的前提下，持续从新数据或新任务中学习的能力。本文特指在开放基座模型上进行多阶段微调以提升性能的范式。
*   **π-shaped Performance**：π形性能，指模型在特定目标领域（如法律）的能力显著提升，同时通用能力几乎不下降（甚至略有提升）的性能模式，形似希腊字母 π。
*   **Constitutional DPO**：宪法直接偏好优化，一种结合人工智能宪法原则和 DPO 算法的对齐方法，用于生成符合特定价值和安全标准的偏好数据对。
*   **Fisher-Routed Directional Ablation**：Fisher 路由方向性消融，一种通过估计模型参数的 Fisher 信息矩阵，将行为编辑向量路由到低敏感参数方向，以最小化对齐干预对其它模型能力影响的权重编辑技术。
*   **CapTrack**：能力跟踪元基准，一个用于系统性地监测模型在持续学习过程中能力漂移（按 CAN-能力/WILL-意愿/HOW-方式 维度）的评估框架。
*   **DeerFlow**：一个开源的多代理工作流框架，本文以其为蓝本改编构建了面向深度研究（Deep Research）的法律场景多代理 Harness。
*   **Test-Time Scaling / Fusion**：测试时扩展/融合，指在推理阶段，通过多次采样（N 个候选）并利用融合策略（而非单一 reward model 选择）聚合结果，以提升模型最终输出质量的方法。

## 可复现要素
*   **数据集**：使用了多种公开和私有数据集。公开基准包括 AIME 2026, GPQA-Diamond, SWE-bench Pro, Terminal-Bench 2.1, Stanford LegalBench, COLIEE Task 1 等。内部构建或合作数据集包括 ContractScrub, InsuficiencyBench, 以及用于 CapTrack 和 Scales++ 评估的子集。原始语料池 >19T tokens，精选出 200B tokens 用于 CPT。**部分数据可能受限于合作或隐私协议未完全公开**。
*   **代码/权重**：**部分代码和模型权重已开放**。具体开源范围需查阅论文原文及官方发布渠道。
*   **关键超参**：
    *   CPT: seq len 8,192, global batch 512, LR sweeps: {2e-6, 5e-6, 1e-5, 2e-5}, Token budget sweeps: {20, 40, 100, 200}B.
    *   Constitutional DPO: β ∈ {5, 10}, lr ∈ {1e-7, 2e-7}, λ ∈ {0.9, 1.0}.
    *   Fisher-Routed Ablation: ridge fraction ρ = 0.05.
    *   RL Stage 1: off-policy/on-policy 混合训练。
    *   Test-Time Fusion: 最优候选数 N=4。
    *   量化：动态 FP8，lm_head 和 MoE 门控保留全精度。
