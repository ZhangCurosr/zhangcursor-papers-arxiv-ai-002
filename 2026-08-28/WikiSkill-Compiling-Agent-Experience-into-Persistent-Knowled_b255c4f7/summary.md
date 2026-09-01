---
title: "WikiSkill-Compiling-Agent-Experience-into-Persistent-Knowled"
source: https://arxiv.org/pdf/2608.27454v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 22:13:35"
field: "Agent技能进化与知识累积"
keywords: ["Agent Skill Evolution", "Persistent Knowledge Base", "WikiLayer", "Self-Improving Agents", "Cross-model Transfer"]
innovations: ["提出Raw/Wiki/Skill三层架构实现经验到持久知识的编译", "Wiki永不回滚策略保障跨迭代知识累积", "揭示技能进化与模型缩放的互补性及跨模型迁移能力"]
benchmarks: ["LiveMathematicianBench", "SealQA", "SpreadSheetBench", "OfficeQA", "ALFWorld"]
---

# 论文速读：WikiSkill-Compiling-Agent-Experience-into-Persistent-Knowled

## 一句话总结
本文提出了WikiSkill框架，通过将Agent执行经验持续汇总到持久化知识库（wiki）中，实现智能体技能的协同进化；该框架在五个多样化基准上 consistently 优于现有技能进化方法，并揭示了技能进化与模型缩放之间的互补关系及跨模型迁移能力。

## 研究问题与动机
- **核心问题**：如何让Agent从历史执行经验中系统化地积累知识，以支持长期、可复用的技能进化？
- **现有方法不足**：已有工作（如EvoSkill、Trace2Skill、SkillOpt）虽然能通过迭代优化技能，但所积累的洞察分散于优化历史中，缺乏独立的、持续演化的知识表示层，限制了跨迭代的有效复用。
- **设计缺口**：前序方法未将"已学内容"维护为独立的知识表示，导致Skill Proposer难以利用历史拒绝决策、错误模式重复出现等问题。

## 核心贡献（创新点）
1. **提出WikiSkill三层架构**：将Agent工作空间划分为Raw Layer（不可变执行轨迹）、Wiki Layer（持久化知识结构）和Skill Layer（可执行程序知识），使技能开发能建立在日益完善的知识基础上。
   - *区别*：现有方法缺乏专门的知识积累层，经验信息直接用于技能更新而无长期记忆机制。

2. **设计Wiki Maintainer与Skill Proposer协同机制**：Wiki Maintainer对采样轨迹进行根因分析并更新wiki，Skill Proposer基于wiki和轨迹自主生成技能提案，形成持续迭代的进化循环。
   - *区别*：不同于Trace2Skill等直接将轨迹映射到skill patch的方法，本文通过wiki层实现知识的中间表示与跨迭代保留。

3. **引入严格验证门控与wiki永不回滚策略**：技能更新需通过validation split提升才接受，但wiki积累永远保留，确保历史知识不被遗忘。
   - *区别*：EvoSkill等采用有限前沿策略，可能丢失早期有用知识；WikiSkill保证知识只增不减。

4. **系统揭示技能进化与模型能力的交互规律**：发现更大模型从技能进化中获益更多（Qwen-4B/9B/27B分别提升12.3/17.5/23.9分），且小模型+技能可超越无技能大模型（9B with WikiSkill > 27B without skill）。
   - *区别*：此前工作未系统研究技能发现能力与技能执行能力的分离问题。

5. **证明跨模型技能迁移有效性**：不同模型进化的技能可跨模型使用，且有时优于自进化技能（如Qwen-27B进化的技能可使Qwen-9B在ALFWorld达到70.2% vs 63.4%）。
   - *区别*：首次系统性验证技能作为可移植知识的潜力。

## 方法详解
### 三层知识架构
- **Raw Layer**：存储不可变的执行轨迹$\tau_i$，包含推理过程、工具调用及结果，供Wiki Maintainer和Skill Proposer分析。
- **Wiki Layer**：维护结构化知识，包含`patterns/`目录（记录成功策略/失败模式）、`logs.md`（进化日志）和`skill-impact.md`（提案审计追踪）。wiki跨迭代累积，永不回滚。
- **Skill Layer**：包含活跃技能集$S$，每个技能含`SKILL.md`（指令内容）和`PURPOSE.md`（追溯至wiki pattern）。

### 进化循环机制
1. **Inference Agent**：使用当前技能集$S_{k-1}$执行训练集$D_{train}$上的rollout，生成轨迹$\mathcal{T}_{train,k}$。训练期间禁止访问wiki（消融实验证实此设计）。
2. **Wiki Maintainer** $\mathcal{M}_{WM}$：对采样轨迹子集$\mathcal{T}_{sample,k}$（最多8条：≤5失败+≤3成功）进行根因分析，更新wiki：
   $$W'_k \leftarrow \mathcal{M}_{WM}(W_{k-1}, \mathcal{T}_{sample,k})$$
3. **Skill Proposer** $\mathcal{M}_P$：以ReAct方式自主访问wiki和轨迹，诊断失败并生成原子提案$P_k$：
   $$P_k \leftarrow \mathcal{M}_P(W'_k, S_{k-1}, \mathcal{T}_{train,k})$$
4. **Gating and Rollback**：应用提案得$S'_k$，在validation split上评估，按阈值$\mathcal{R}_{best}$决定是否接受：
   $$S_k \leftarrow \begin{cases} S'_k & \text{if } \mathcal{R}(\mathcal{T}_{val,k}) > \mathcal{R}_{best} \\ S_{k-1} & \text{otherwise} \end{cases}$$
   无论技能是否接受，wiki状态$W_k$均更新并保留。

### 关键设计细节
- Skill Proposer采用多轮ReAct交互（约10-20步），动态检索相关pattern和轨迹。
- 每轮进化后程序化追加提案元数据到`skill-impact.md`，形成可审计的历史记录。
- 若validation score达1.0则提前终止进化循环。

## 实验与结果
### 数据集与基线
- **五基准**：LiveMathematicianBench（数学推理）、SealQA（网络搜索）、SpreadSheetBench（电子表格操作）、OfficeQA（长上下文文档QA）、ALFWorld（交互式具身任务）。
- **五模型**：Qwen-3.5-4B/9B、Qwen-3.6-27B、Gemma-4-31B、Gemini-3.5-Flash。
- **基线**：Trace2Skill、EvoSkill、SkillOpt，以及no-skill baseline。

### 主要结果
- **整体优势**：WikiSkill在五模型上平均性能均最优，较最强基线提升3.3-12.0分（如Gemini-3.5-Flash平均68.1% vs 56.1%）。
- **模型缩放互补**：Qwen系列中，WikiSkill提升幅度随模型增大（4B:+12.3, 9B:+17.5, 27B:+23.9）；Qwen-3.5-9B with WikiSkill (47.4%) > Qwen-3.6-27B no skill (39.4%)。
- **跨基准差异**：SpreadSheet增益最大（27B模型+40.9分），OfficeQA对4B模型略降（28.5% vs 30.2% baseline）。
- **跨模型迁移**：Qwen-27B技能使Qwen-9B在SpreadSheet达50.5% vs 自进化33.6%；小模型技能也可迁移至大模型（4B→31B在LiveMath提升显著）。

### 最强结果
- Gemini-3.5-Flash + WikiSkill在LiveMath达72.6%（baseline 33.0%，+39.6分）、SpreadSheet达76.6%（baseline 50.5%，+26.1分）。
- Qwen-3.6-27B + WikiSkill在SpreadSheet达81.7%（baseline 40.8%，+40.9分）。

## 相关工作脉络
1. **Trace2Skill (Ni et al., 2026)**：并行轨迹分析+层次化合并patch；WikiSkill通过wiki层实现跨迭代知识累积，避免重复失败提案。
2. **EvoSkill (Alzubi et al., 2026)**：基于前沿候选程序搜索；WikiSkill不依赖有限前沿，而是持续扩展知识图谱。
3. **SkillOpt (Yang et al., 2026)**：六阶段ReflACT流水线；WikiSkill简化为双Agent协作（Maintainer+Proposer），知识持久化。
4. **GEPA (Agrawal et al., 2026)**：通用prompt优化器；本文聚焦专用skill-evolution框架，证明其优于通用方法。
5. **Skill retrieval工作 (Cho et al., 2026; Shi et al., 2026)**：关注技能选择；WikiSkill专注技能质量本身，假设技能已全量注入。
6. **Karpathy LLM Wiki观点**：启发本文 idea，将经验编译为持久化、复利式知识。

## 局限性与未来方向
- **技能检索未评估**：当前直接将技能注入prompt，未测试大规模技能库中的检索/触发问题。
- **严格门控限制**：仅接受validation提升的提案，排除中性但可能为后续迭代奠基的proposal。
- **缺乏wiki修剪机制**：pattern持续累积无自动 pruning，长期演化可能导致知识冗余。
- **未覆盖超长horizon任务**：当前基准多为短时交互，缺乏数百步环境交互或小时级任务测试。
- **未来方向**：在线技能适应（单次长rollout内持续精炼）、更灵活的acceptance criteria、wiki自动管理。

## 研究启发与可借鉴点
1. **三层架构可复用**：Raw/Wiki/Skill分离设计适用于其他Agent自我改进系统，尤其是需要历史知识累积的场景。
2. **wiki永不回滚策略**：与技能rollback形成对比，保证经验只增不减，这一理念可迁移至持续学习系统。
3. **Skill Proposer的ReAct探索模式**：自主决定检索哪些pattern和轨迹再提proposal，比直接喂全量轨迹更高效。
4. **分层采样策略**：Wiki Maintainer采样≤5失败+≤3成功轨迹，平衡根因分析与成功模式保留，可借鉴于其他trace分析任务。
5. **技能发现vs执行的分离研究**：本文实证证明两者是可分离的能力，启发后续工作分别优化这两阶段。

## 关键术语表
**WikiSkill**：将Agent执行经验编译为持久化知识库以支持技能进化的框架。  
**Raw Layer**：存储不可变执行轨迹的层级，包含完整交互历史。  
**Wiki Layer**：维护结构化、累积性知识的层级，包含pattern目录和进化日志。  
**Skill Layer**：包含可被Agent使用的程序性技能集合。  
**Wiki Maintainer**：分析轨迹并更新wiki的Agent组件。  
**Skill Proposer**：基于wiki和轨迹生成技能修改提案的Agent。  
**Gating and Rollback**：基于validation性能决定技能更新是否接受的机制。  
**Cross-model Transfer**：一个模型进化的技能在另一模型上评估的现象。

## 可复现要素
- **数据集**：五基准均引用自已有公开工作（LiveMath、SealQA、SpreadSheetBench、OfficeQA、ALFWorld），划分与工具设置与 prior work 一致。
- **代码/权重**：论文未明确声明开源，但提供了完整prompt和算法描述（Appendix E）。
- **关键超参**：batch size=$N_{train}$（全量处理）；采样上限8条轨迹（≤5失败+≤3成功）；单轨迹截断15000字符；ReAct步数约10-20步。
