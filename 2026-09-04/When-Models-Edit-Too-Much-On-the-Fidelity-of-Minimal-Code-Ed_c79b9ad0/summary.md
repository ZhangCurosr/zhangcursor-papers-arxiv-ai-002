---
title: "When-Models-Edit-Too-Much-On-the-Fidelity-of-Minimal-Code-Ed"
source: https://arxiv.org/pdf/2609.04061v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 15:33:26"
field: "代码大语言模型评估与后训练"
keywords: ["代码编辑保真度", "over-editing", "最小化代码修复", "强化学习微调", "LLM代码生成", "编辑距离评估"]
innovations: ["构建含已知最小修复ground truth的400题代码编辑保真度评估基准，填补可度量编辑量维度的空白", "发现显式preservation提示可显著降低over-editing且不牺牲正确性，且强化学习可持久化学习最小编辑风格优于SFT/DPO"]
benchmarks: ["BigCodeBench", "DeepCoder", "LiveCodeBench v6", "Defects4J"]
---

# 论文速读：When-Models-Edit-Too-Much-On-the-Fidelity-of-Minimal-Code-Ed

## 一句话总结
论文系统研究了代码修复中 LLM 的"过度编辑"（over-editing）问题——模型能在测试通过的前提下重写大量不必要的代码；通过构建已知最小修复基准评估 20+ 前沿模型，发现显式 preservation 提示可显著降低冗余编辑，且通过强化学习（RL）可直接让模型学会最小化编辑风格而不损害通用编码能力。

## 研究问题与动机
- **现有评估只关注功能正确性**：当前代码 LLM 基准（如 HumanEval、SWE-bench）主要用 Pass@1/k 衡量"是否修对"，但无法反映模型修改了原代码多少内容，导致"正确但过度重写"的修复被遗漏。
- **Brownfield 维护场景的需求**：实际开发中需保留原始实现意图、减少 code review 负担并避免 regression，尤其是对已有代码做局部修复时，最小化编辑至关重要。
- **模型默认行为偏向"生产鲁棒代码"**：实验显示即使是最强模型（如 GPT-5.5），在通过测试的同时会自行添加输入校验、dtype 转换、NaN 屏蔽等无关逻辑，属于任务框架错配。
- **推理增强和模型规模并非万灵药**：推理变体和非推理变体表现不一致；模型从 14B 扩到 32B 时，Excess Levenshtein 反而上升（0.108→0.127），说明单纯 scaling 无法解决过度编辑。

## 核心贡献（创新点）
1. **构建了首个含已知最小修复 ground truth 的大规模代码编辑保真度评估基准**：基于 400 道 BigCodeBench 题目，注入受控 AST 级别变异，每个任务都有明确的 minimal patch 作为参照，填补了"可度量编辑保真度"的空白。
2. **在 20+ 前沿模型上首次量化 over-editing 现象**：发现高 Pass@1 可与高冗余编辑量共存（如 GPT-5.5 High 的 excess Lev. 是 Claude Opus 4.7 的 4 倍以上），证明编辑保真度是独立于功能正确性的新质量维度。
3. **证实显式 preservation 提示可显著改善编辑保真度**：仅增加一句"尽可能保留原始代码"，可使聚合 excess Levenshtein 从 0.195 降至 0.131，降低认知复杂度 26.6%，并提升 Pass@1 2.3 个百分点。
4. **首次系统比较 SFT/rSFT/DPO/RL 在最小化编辑学习上的表现**：发现 RL 在跨域泛化上最优（OOD Pass@1=0.782，Excess Lev.=0.050），而 SFT 严重过拟合训练分布，DPO 次之；LoRA rank=64 即可几乎恢复全参数 RL 的保真度收益。
5. **提出 edit-fidelity reward 设计并在 Qwen3 系列上验证可扩展性**：奖励函数结合执行正确性（λ_exec=0.1）与编辑距离惩罚（λ_edit=1.0），从 4B 到 14B 模型均稳步降低 excess edit，且不损害 LiveCodeBench 通用编码能力。

## 方法详解
- **基准构建**：从 BigCodeBench 采样 400 个 Python 任务，对每个 reference solution 注入 1–2 个预定义 AST-level 变异（共 14 类：比较运算符、range 边界、排序顺序、累加器初始化、算术运算符、边界守卫、列表索引、函数调用替换、复制移除、布尔常量、数值常量、slice 边界、条件取反、range step 修改），仅保留未通过原始测试的样本；gold patch 即为变异的逆向操作。
- **核心评估指标**：
  - **Pass@1**：修复通过所有测试的比例。
  - **Excess Normalized Levenshtein Distance**：$E_{Lev}(M) = d(M, C) - d(G, C)$，其中 $C$ 为含 bug 程序，$G$ 为 gold 修复，$M$ 为模型输出；$d(\cdot,\cdot)$ 为去除注释和格式后对 token 序列计算的归一化编辑距离。
  - **Added Cognitive Complexity**：使用 Python AST visitor 计算 $CC(M) - CC(G)$，衡量修复带来的额外结构复杂度。
- **Prompt 设计**：在 generic prompt（"Fix and complete my function."）基础上追加 preservation clause（"…but keep as much of the original code as possible"）。
- **RL 训练框架**：以 PRIME-RL（GRPO-style group-relative RL）训练 Qwen3-4B-Instruct，每样本 K=16 次 rollout；奖励函数：
  $$r(M) = \lambda_{exec} \cdot \mathbb{1}[\text{passed}] - \lambda_{edit} \cdot e(M)$$
  其中 $e(M)=d(M,C)-d(G,C)$，失败/不可解析样本得 -0.2；默认 $\lambda_{exec}=0.1$，$\lambda_{edit}=1.0$。
- **对比方法**：SFT（程序化最小修复监督训练）、rSFT（reject sampled SFT，保留编辑距离最小的 3 个候选）、DPO（偏好最小编辑候选 vs 最大编辑候选）、LoRA 参数高效 RL（rank 1–64 扫描）。

## 实验与结果
- **数据集**：400 道 BigCodeBench Python 任务（平均 10.4 行可执行代码，gold repair 50.2% 为单 token 编辑、91.8% 不超过 2 token）；训练用 4,141 个 DeepCoder 样本；OOD 测试含 20 类未见过变异族；跨语言验证用 Defects4J Java bug 集。
- **前沿模型评估**：覆盖 20+ 模型（Claude/GPT/Gemini/DeepSeek/Grok/GLM/Kimi/Qwen/Mistral 等），含 reasoning 变体。最强保真模型为 Claude Opus 4.7（Excess Lev.=0.0704），最差为 GPT-5（Excess Lev.=0.4379）。
- **Prompt 效果**：preservation 提示使聚合 Excess Lev. 从 0.195 降至 0.131（↓32.8%），Added CC 降 26.6%（paired signed-rank $p<10^{-4}$），Pass@1 升 2.3 pp（95% CI [+1.49, +3.05]）；效果在重采样、改写提示词、开权模型（Qwen2.5-Coder 14B/32B）上一致。
- **推理与规模效应**：reasoning 效果因模型而异（非单调）；Qwen2.5-Coder 从 14B→32B，Pass@1 上升但 Excess Lev. 从 0.108 升至 0.127。
- **最触发的过编辑变异类型**：slice bounds（Excess Lev.=0.353）、list indexing（0.244）、conditional inversion（Added CC=1.46）。
- **Over-editing 分类（530 个高冗余样本）**：Defensive generalization（64.2%）、Data-flow rewrite（63.2%）、Contract drift（34.7%）、Feature accretion（23.6%）。
- **训练实验（Table 3，OOD）**：SFT 在域外崩溃（Pass@1 0.458）；rSFT 0.780；DPO 0.787/0.092；**RL 最优 0.782/0.050/0.185**，LCB 保持 +0.6pp。
- **LoRA 实验**：rank=64 时 Excess Lev.=0.051，接近全参数 RL 的 0.050，LCB 微增 +0.001。
- **Reward ablation（Table 5）**：正确性 alone→Excess Lev.=0.189；edit-only→0.070；Lev+CogC→0.151；**Full reward→0.050** 最优。
- **跨语言迁移（Defects4J，Table 6）**：RL 保持通过率不变（4B: 7.4%→7.0%；14B: 14.9%→14.5%），但 Excess Lev. 分别降至 0.060 和 0.074。
- **对比 DPOP（AdaPatcher）**：Full-parameter DPOP: Pass@1=0.718/Excess=0.064；RL: 0.782/0.050；LoRA r=64 DPOP: 0.783/0.082；RL: 0.797/0.051。

## 相关工作脉络
1. **正确性导向的编程评估**（Chen et al., 2021; Jain et al., 2024; Zhuo et al., 2025）：本文的核心对比基线——这些工作只测 Pass@1/k，不测量编辑量，本文将其扩展为同时度量编辑保真度的独立维度。
2. **Instructed Editing 基准**（Cassano et al., 2024 CanItEdit; Guo et al., 2025 CodeEditorBench; Chi et al., 2025 EDIT-Bench）：CanItEdit 报告了 ExcessCode 但主要依赖弱 oracle；本文通过已知的 gold patch 实现精确度量，可区分"合理替代修复"与"真实冗余编辑"。
3. **显式追求小 patch 的修复系统**：
   - **CREF**（Yang et al., 2024）：面向编程辅导的对话式修复，侧重 patch precision 而非最小化。
   - **AdaPatcher**（Dai et al., 2025）：结合运行时 trace 定位与 DPO 偏好学习，本文在其 offline 偏好阶段基础上比较，发现 on-policy RL 泛化更好。
   - **PAFT**（Yang et al., 2026）：pre-train-time 保真微调，与本文 post-training RL 路线形成对照。
   - **PRepair**（Ke et al., 2026）：并发工作，研究 edit-aware RL，与本文目标相近但基于不同基准。
4. **代码相似性度量**（CodeBLEU, Ren et al., 2020）：本文在 Appendix B.1 对比 CodeBLEU 与 token Levenshtein，发现 CodeBLEU 可能因 n-gram 重叠而误判大重写为"更相似"，故选择 token-level 编辑距离。
5. **推理模型的指令遵循**（Fu et al., 2025b; Li et al., 2025; Wen et al., 2024）：本文发现强推理能力与显式约束存在交互问题（如 GPT-5.5 Reasoning 反而增加 cognitive complexity），延续了"推理增强未必改善约束遵循"这一观察。

## 局限性与未来方向
- **基准可控性限制**：使用的是人工注入的 AST 级局部变异，比真实仓库 bug 和多文件变更简单；结果需在实际 brownfield 场景中验证。
- **语言与模型家族局限**：评估以 Python 为主，训练基于 Qwen 系列；跨语言泛化仅在 Defects4J（Java）小规模验证，绝对修复率仍低。
- **人类评估规模有限**：metric 验证仅 3 位标注员 × 100 对补丁；需要更大规模人评。
- **Open-ended 生成、架构重构、功能扩展**不在本工作范围内；更广泛的代码维护任务（如性能优化）是否适用仍需探索。
- **奖励函数设计**：cognitive complexity 加入 reward 反而有害（表5），未来需更细粒度的结构复杂度度量。
- **可扩展性**：未做全面超参搜索，LoRA rank 与 rollout budget 的组合仍需进一步探索。

## 研究启发与可借鉴点
1. **"已知最小修复"基准设计思路可迁移**：对于任何需要评估"编辑最小性"的任务（如文档翻译编辑、配置修改、SQL 重写），均可采用"人工注入局部扰动 + 已知逆操作作为 gold patch"的思路来构建可测基准。
2. **Excess Levenshtein + Added Cognitive Complexity 双指标体系值得复用**：前者度量编辑量、后者度量结构复杂度，二者互补且经人类验证（κ=0.897/0.939），可作为代码编辑任务的通用评估方案。
3. **Preservation prompt 作为零成本 baseline**：仅需在 user message 末尾追加一句指令即可显著改善编辑保真度，可作为所有代码编辑 agent 的默认 prompt 策略，无需额外训练。
4. **RL 优于 SFT/DPO 用于学习编辑风格偏好**：SFT 易过拟合狭窄合成分布，RL 通过 on-policy 探索可保留通用能力（LCB +0.6pp vs SFT -14.9pp），这一结论可推广到其他"风格化生成"任务（如代码格式化、注释生成）。
5. **LoRA rank=64 即可捕获最小编辑偏好**：说明编辑保真度本质上是"风格偏好"而非新编码技能，后续工作可用更轻量的参数高效方式部署此类模型。

## 关键术语表
**Over-editing**：模型在修复 bug 时，在通过所有测试的前提下修改了远超必要最小量的代码的行为。
**Excess Normalized Levenshtein Distance**：模型编辑距离与 gold 最小修复距离之差（$d(M,C)-d(G,C)$），正值表示模型改多了，核心保真度指标。
**Added Cognitive Complexity**：修复后代码的 SonarSource 认知复杂度减去 gold 修复的复杂度，衡量引入的结构冗余。
**Preservation Prompting**：在请求末尾追加"尽可能保留原始代码"指令，引导模型进入"保真修复"模式而非"重新实现"模式。
**Gold Patch**：由人工注入的 AST 变异唯一确定的最小逆向操作，作为 edit-fidelity 评估的可信 ground truth。
**rSFT（Reject-sample SFT）**：从多个候选修复中筛选出编辑距离最小的若干样本进行监督训练。
**DPOP（DPO-positive）**：AdaPatcher 提出的偏好学习方法，以最小编辑修复为正样本、最大编辑为负样本进行 DPO 训练。
**GRPO（Group Relative Policy Optimization）**：PRIME-RL 使用的组内相对优势估计 RL 算法，本文用它训练最小化编辑策略。

## 可复现要素
- **数据集**：BigCodeBench（Apache 2.0）400 题 + 人工注入 AST 变异；DeepCoder（MIT）4,141 训练样本；LiveCodeBench v6（MIT）；Defects4J（学术研究用）。**评估变异族列表见 Appendix A.1，OOD 变异族见 Appendix D.2**。
- **代码/框架**：LlamaFactory（Apache 2.0）用于 SFT/rSFT/DPO；PRIME-RL（Apache 2.0）用于 GRPO-style RL。**论文未明确声明专属开源代码仓库**。
- **关键超参**：SFT/rSFT/DPO 学习率 $10^{-5}$，3 epochs；DPO beta=0.1；RL 学习率 $10^{-6}$，K=16 rollouts，group-mean baseline；RL 奖励权重 $\lambda_{exec}=0.1$，$\lambda_{edit}=1.0$；失败样本 reward=-0.2；LoRA rank sweep {1, 8, 16, 32, 64}。
- **推理设置**：temperature=1，thinking 模型 budget=10,000 tokens；详见 Appendix A.4。
