---
title: "Thomson-Continual-Learning-of-Frontier-Models-for-SovereignA"
source: https://arxiv.org/pdf/2608.27147v1.pdf
model: agnes-2.5-flash
chunks: 8
summarized_at: "2026-09-01 19:04:09"
field: "持续学习与模型对齐"
keywords: ["Continual Learning", "DPO", "RL", "SovereignAI", "Deep Research Agent", "GSPO", "Legal LLM", "Test-Time Scaling"]
innovations: ["无SFT的DPO+RL两阶段训练链路实现π-shaped性能改进", "基于counterfactual completions和SNR筛选的偏好对构建与reward设计框架", "低预算（<4000万美元）持续学习达到frontier级法律与通用能力"]
benchmarks: ["LegalBench", "PRBench Legal Hard", "GDPval", "SWE-bench Pro", "Humanity's Last Exam", "Tau2-bench", "COLIEE", "CaseHOLD", "ReClor"]
---

# 论文速读：Thomson-Continual-Learning-of-Frontier-Models-for-SovereignAI

## 一句话总结
本研究证明：通过对开放权重模型（Qwen3.5-397B / Qwen3.6-35B）进行全量持续学习（continual learning），仅用≤36人团队、3个月周期、总成本约4000万美元（其中Large模型训练成本<45万美元），即可使中规模模型达到甚至超越同期前沿闭源模型（如GPT-5.5/5.6、Claude Sonnet 5）的法律与通用能力表现，实现"低预算主权AI"的可行性论证。

## 研究问题与动机
- **前沿模型被少数超资方垄断**：当前 frontier 级模型依赖数千亿美元级 compute 和数百人团队，普通机构无法独立构建、部署、治理自己的 AI。
- **持续学习可突破compute瓶颈**：通过持续学习（continual learning）对开放权重模型进行全量权重更新，有望在远低于常规训练的 compute 预算下实现等效的多代际性能跃升。
- **遗忘问题尚未被系统消除**：窄领域持续适配常导致非目标领域能力衰退（forgetting）；本文旨在实现"π-shaped"性能改进——目标领域大幅提升的同时非目标领域不降反升。
- **Deep Research 智能体存在系统性弱点**：不调用工具直接作答、编造引用、证据压缩失真、报告不完整等问题普遍存在，需要专门的训练流水线解决。

## 核心贡献（创新点）
1. **π-shaped 持续学习训练流水线**：首次系统性证明在开放权重模型上经持续学习可达到 frontier 级性能，且遗忘问题几乎完全消除，非目标领域性能意外提升。
2. **无SFT的三模块post-training设计**：明确规避独立 SFT 阶段（因其易导致快速遗忘），采用两阶段DPO（广泛偏好对齐→agentic专业化）+ 两阶段RL（短上下文多样查询→长上下文Deep Research端到端强化学习），DPO仅作RL的off-policy warm start。
3. **轨迹级偏好对构建框架**：基于counterfactual completions（固定prefix重跑节点隔离单一决策效应）结合Teacher-Guided Behavioural Repair、Factuality-Based Quality Selection、Programmatic Tool-Use Correction三种信号构建偏好对，适配Planner/Researcher/Compaction/Reporter四类智能体节点。
4. **SNR筛选框架用于Reward设计**：提出信噪比筛选机制（SNR² = advantage方差/judge噪声方差−1），仅用于过滤而非优化目标，显著提升reward有效性；配合Batched judging、Adaptive judge capacity、Adaptive reasoning effort等工程实践。
5. **能力课程（capability curriculum）与GSPO算法适配**：两阶段RL构成从局部能力到全局协调的能力课程，使用GSPO（Group Sequence Policy Optimisation）算法，配合异步rollouts（policy lag可达5版本）、inference policy log prob记录等技巧，在B200集群上实现高效训练。

## 方法详解

### 基础模型与整体架构
- 基座：**Qwen3.5-397B**（Large）与 **Qwen3.6-35B**（Small），进行全量权重更新（full-weight updates），不使用参数高效微调（PEFT）或大规模蒸馏。
- 算力：Nvidia B200集群，峰值 H = 2.25×10¹⁵ FLOP/s（dense BF16），最大368块GPU并行。

### 三模块开发流水线

#### 模块一：Value Re-alignment（价值重对齐）
- **Constitutional DPO**：基于 Public AI Constitution 进行偏好对齐。
- **Fisher-Protected Directional Ablation**：保护关键参数方向，防止持续学习中核心能力退化。

#### 模块二：Continual Pre-training / Mid-Training
- 数据中心的数据增强与混合优化（Bayesian Optimisation校准数据混合）。
- **Model merging** 技术保护通用能力，防止灾难性遗忘。
- 语义空间低密度区域引导人类专家数据收集。

#### 模块三：Post-Training（DPO + RL）
- **两阶段DPO**：
  - Stage 1（382步）：广泛偏好对齐，建立基础行为偏好。
  - Stage 2（67步）：在Stage 1 checkpoint基础上，context length扩展至64k tokens，更大KL系数、更低学习率、单epoch，聚焦 agentic 专业化。
  - DPO仅作为RL的off-policy warm start，不作为终端优化步骤。
- **两阶段RL（能力课程）**：
  - RL Stage 1：单步训练，混合 diverse queries 与通用能力任务；包含来自agentic trajectory关键节点前缀的偏好样本，训练 report writing、tool summarisation 等局部能力。
  - RL Stage 2：多轮 rollout，使用完整 context window，训练 planning/tool use/evidence integration 的全局协调。
- **GSPO算法**（Group Sequence Policy Optimisation）：
  - 群体相对优势估计，无需单独训练 value network。
  - 优势估计：$\hat{A}_i = (r(x, y_i) - \text{mean}_j r(x, y_j)) / \text{std}_j r(x, y_j)$
  - 使用 DAPO 的非对称 clipping bounds，省略 KL 惩罚项。
  - 关键技巧：
    - Inference Policy Log Probabilities：用 vLLM rollout engine 记录的 token log probs 定义 $\pi_{\theta_{old}}$，避免异步训练多版本 policy 的 recompute 误差。
    - Asynchronous Rollouts：rollout 和 reward 计算与策略优化异步，policy lag 可达 5 个版本。
    - Reward Normalisation within Groups：组内标准化奖励，防止高方差 collection 主导梯度。
    - Overlong Filtering：RL Stage 1 对达到生成长度上限的单轮响应进行过滤。

### 深度研究智能体Harness
- 改编自 DeerFlow + LangGraph，planner-worker 架构。
- Worker 为受限工具调用预算的 ReAct loop，通过 query-conditioned compaction 处理长文档，确保引用可追溯。
- 四类节点：Planner → Researcher → Compaction → Reporter。
- 离线前缀（多智能体 harness 记录的中间状态）与在线轨迹续写结合，支持单智能体收敛环境学到的能力向生产拓扑迁移。

### 偏好对构建（三种信号）
1. **Teacher-Guided Behavioural Repair**（Planner/Researcher）：以模型身份（更强vs更弱 agentic model）定义偏好，无需 per-example judge。
2. **Factuality-Based Quality Selection**（Compaction/Reporter）：提取声明评估来源支撑度，factuality 分差 < 0.10 的偏好对被剔除。
3. **Programmatic Tool-Use Correction**（Researcher）：基于程序化谓词，若 retrieval 是必要动作则"未调用工具即作答"为 rejected，"调用 ≥1 个检索工具"为 chosen。

### Reward设计与SNR筛选
- **LLM-as-a-Judge 双属性评估**：SME calibration（judge分数是否与专家判断一致）+ Discriminative reliability（稳定差异能否从judge噪声中区分）。
- **SNR²筛选**：$\text{SNR}^2 = \widehat{\sigma}_{adv}^2 / \widehat{\sigma}_{judge}^2 - 1$，用于过滤而非优化目标。
- **Constitutional RL Rewards**：基于 Promptfoo 对抗测试框架，覆盖 >100 个国家争议性议题，刻意惩罚"回避讨论"行为。
- **Reward库**覆盖五大类：Diverse Queries / Deep Research / Citations / General Capability / Legal Task Completion，分 Judge 与 Verifiable 两种类型。

### 评估框架：DeCE（Decomposed Criteria-based Evaluation）
- 将评估分解为**精确度（Precision）**和**召回率（Recall）**两个正交维度。
- 从每个 gold answer 自动提取 Required Information 和 Helpful Information。
- 与人类专家判断相关性：Pearson r=0.78, Spearman ρ=0.76。
- 仅 11.95% 的自动提取标准需要人工修订。

## 实验与结果

### 训练算力统计
| 模型 | 总FLOP | GPU-hours | CPT利用率 | DPO利用率 | RL1利用率 | RL2利用率 |
|------|--------|-----------|----------|----------|----------|----------|
| Thomson-1.0-Large | ≈4.6×10²³ | ≈8.8×10⁴ | U≈0.85 | U≈0.66 | U≈0.35 | U≈0.56 |
| Thomson-1.0-Small | ≈1.6×10²³ | ≈3.5×10⁴ | — | — | — | — |
| Snowdon-1.0-Large | 5.11×10²⁰ | 210.2 | — | — | — | — |

- 两模型均未触及欧盟 AI Act 10²⁵ FLOP 阈值。
- CPT+RL Stage 2 占总量 85%。

### 通用基准对比（vs Qwen3.5-397B）
| 基准 | Qwen3.5 397B | Thomson 1.0-Large | Δ |
|------|-------------|-------------------|---|
| AIME 2026 | 93.3 | 90.0 | −3.3 |
| FaithEval-Inconsistent | 99.3 | 99.1 | −0.2 |
| GDPval | 89.4 | **93.5** | **+4.1** |
| GPQA-Diamond | 87.0 | 89.1 | +2.1 |
| Humanity's Last Exam | 24.8 | **28.5** | **+3.7** |
| IFEval | 91.4 | 92.4 | +1.0 |
| MGSM | 91.8 | 91.3 | −0.5 |
| MMLU-Pro | 88.1 | 87.0 | −1.1 |
| SimpleQA-Verified | 54.1 | 49.9 | −4.2 |
| SWE-bench Pro | 33.8 | 34.0 | +0.2 |
| Tau2: Telecom | 98.3 | **98.3** | 0 |
| Terminal-Bench 2.1 | **54.0** | 48.3 | **−5.7** |
| WritingBench | 77.9 | **80.3** | **+2.4** |

- **Small 模型**：AIME 2026 Thomson 1.0-Small = **90.0**（vs Qwen3.6-35B 86.7）；Tau2 Telecom 达到 **100.0**（满分）。

### 专家盲评结果
| 对比 | Thomson胜率 | 对手胜率 |
|------|------------|---------|
| 整体查询 | 53–62% | 29–33% |
| 法律查询 vs GPT-5.5 | ~66% | — |
| 法律查询 vs GPT-5.6 Terra | 64% | 26% |
| 通用查询 vs GPT-5.5 | 59% | 27% |
| 通用查询 vs GPT-5.6 Sol | 47% | 34% |
| 通用查询 vs GPT-5.6 Terra | 43% | 40% |

### DeCE法律查询评估（相对于各竞品）
- Thomson 在**完整性**和**实用性**维度全面领先：
  - vs GPT-5.5：完整性 +7.9，实用性 +7.4
  - vs Opus 4.8：完整性 +7.5，实用性 +8.3
  - vs GPT-5.6 Terra：完整性 +9.5，实用性 +9.1

### 测试时间扩展（Test-Time Scaling）
- **Fusion 方法**（N=4：3个候选+1次综合调用）：
  - 学术法律基准：68.42 → 70.70（+2.28）
  - 法律研究基准：73.37 → **78.58**（+5.21）
  - Tax Eval v3：44.04 → **52.81**（+8.77）
- N=4后收益不再累积，N=8时4/5基准持平或下降。
- **结论**：后训练贡献大于推理时扩展，瓶颈在利用阶段（exploitation）而非探索阶段。

### 法律专项基准（Thomson-1.0-Large）
| 基准 | 得分 |
|------|------|
| PRBench Hard | 31.6 |
| LegalBench | 82.3 |
| LEXam MCQ4 | 82.9 |
| MBE | 90.8 |
| ContractScrub | — |
| Query Suff. | — |
| Harvey LAB | — |

### Claim-Entailment 训练轨迹（RL Stage 1, Small）
- Entailed claims：47% → 65%（+18pp，r=0.75）
- Uncited claims：13% → ~1%（r=−0.67，训练中点附近饱和）
- Contradicting claims：先上升至峰值9.0%，再缓慢下降，净变化−0.2pp

## 相关工作脉络
1. **SFT-based continual fine-tuning**（如传统LoRA/全参SFT链路）：本文明确规避独立SFT阶段，证明其易导致快速遗忘；DPO+RL组合在无SFT条件下实现等效对齐效果。
2. **DPO-only 对齐范式**（LLaMA-2-Chat, Vicuna等）：本文DPO仅作RL的off-policy warm start，强调RL的on-policy深度探索才是能力突破的关键，与纯DPO路线形成本质差异。
3. **GRPO/RLHF类后训练**（DeepSeek-R1, Kimi等）：本文使用GSPO（Group Sequence Policy Optimisation）替代传统PPO，无需单独训练value network，并通过异步rollouts和inference log prob记录提升训练效率。
4. **Agent benchmark评估**（SWE-bench, GDPval, tau2-bench等）：本文扩展评估至Agentic Deep Research、DeCE分解评估框架，提出将评估分解为Precision/Recall两个正交维度，与现有holistic评分方式形成互补。
5. **Constitutional AI / AI safety对齐**（Constitutional AI, Anthropic等）：本文将其延伸至>100个国家争议性议题的Constitutional RL Rewards，并刻意惩罚"回避讨论"行为，在合规性与能力间取得平衡。
6. **低资源/持续学习前沿模型**（Continual Learning社区工作）：本文证明在<400万美元预算下可实现多代际性能跃升，挑战了"frontier模型必须千亿美元级训练"的共识。

## 局限性与未来方向
- **小模型在大模型基线前仍有差距**：Thomson-1.0-Small（35B）在多数通用基准上仍落后于Qwen3.5-397B，仅在特定领域（如Tau2 Telecom满分）有突出表现。
- **Terminal-Bench 2.1 大幅下降（−5.7）**：持续学习导致终端操作类能力显著退化，说明多领域适配仍面临能力迁移失衡挑战。
- **RL Stage 1利用率方差大（35.0±16.5%）**：表明部分训练阶段存在计算资源利用不均衡问题，可进一步通过动态batch sizing或early exit优化。
- **Claim-Entailing训练中的"先升后降"现象**：Contradicting claims在训练中先上升至峰值9.0%再缓慢下降，提示reward signal可能存在阶段性噪声，需更精细的训练调度。
- **法律领域强但通用领域相对弱化**：虽然整体能力显著提升，但在SimpleQA-Verified（−4.2）、MMLU-Pro（−1.1）等通用知识基准上仍有小幅下降，主权AI的全面泛化能力仍需增强。

## 研究启发与可借鉴点
1. **无SFT的DPO+RL链路设计**：可直接复用于本团队的后训练阶段，尤其是"DPO仅作为RL warm start"这一思路，避免了SFT导致的快速遗忘问题。
2. **Counterfactual completions偏好对构建**：固定prefix重跑节点隔离单一决策效应的方法，适用于任何多智能体系统的偏好数据自动构建，减少人工标注成本。
3. **SNR筛选框架**：提出的信噪比筛选机制可用于我们团队reward design pipeline，帮助过滤低质量judge信号，提升RL训练稳定性。
4. **DeCE分解评估方法**：将评估分解为Precision/Recall正交维度的思路，适用于我们团队的评测体系建设，尤其适合法律/金融等需要严谨推理的领域。
5. **能力课程（capability curriculum）训练理念**：从局部能力到全局协调的两阶段RL设计，可作为我们后续多阶段训练调度的参考模板。
6. **与领域专家紧密协作的工作流**：三步LLM辅助流程（起草→漏洞筛查→残留检测）及校准轮次设计，可直接迁移至我们团队的benchmark构建环节。

## 关键术语表
- **Continual Learning**：在已有模型基础上持续注入新数据/新任务进行训练，而非从头预训练，常用于在有限预算下实现能力迭代。
- **DPO（Direct Preference Optimisation）**：通过直接优化偏好对（chosen/rejected）来对齐模型输出，无需显式训练reward model。
- **GSPO（Group Sequence Policy Optimisation）**：基于群体相对优势估计的强化学习算法，无需单独训练value network，适合长轨迹agent训练。
- **π-shaped性能改进**：目标领域性能显著提升，同时非目标领域性能不降反升的持续学习现象，打破传统"遗忘-增益"trade-off。
- **DeCE（Decomposed Criteria-based Evaluation）**：将评估结果分解为精确度（Precision）和召回率（Recall）两个正交维度的自动化评估方法。
- **SNR筛选框架**：基于信噪比（优势信号方差/judge噪声方差）过滤低质量reward信号的框架，用于提升RL训练效率。
- **SovereignAI**：指由本地/主权机构独立构建、部署、治理的AI系统，摆脱对少数超资方闭源模型的依赖。
- **Counterfactual Completions**：固定轨迹prefix重跑智能体节点，比较不同模型产生的completion以隔离单一决策效应的方法。

## 可复现要素
- **数据集**：部分公开基准（Stanford LegalBench, COLIEE, CaseHOLD, ReClor, SuperGPQA, LEXam, PRBench等），部分内部构建（AALP, Parentheticals, Finance Agent v2等）；论文未提及所有自定义数据集是否开源。
- **代码/权重**：开源模型版本为 Thomson-1.0-Small (35B)；大模型版本（Thomson-1.0-Large）未提及是否开源。
- **关键超参**：DPO Stage 2 context length=64k tokens；RL policy lag=5个版本；Fusion N=4；DAPO非对称clipping bounds；KL惩罚项省略。
- **硬件要求**：Nvidia B200集群，Large模型峰值H=2.25×10¹⁵ FLOP/s。
- **论文未提及**：具体的learning rate数值、batch size、warmup策略细节、DAPO clipping bound具体参数值。
