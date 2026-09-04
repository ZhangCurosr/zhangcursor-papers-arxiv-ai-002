---
title: "Sequential-Beats-Joint-On-the-Interplay-between-On-Policy-Di"
source: https://arxiv.org/pdf/2609.04108v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 12:06:34"
field: "推理大模型后训练"
keywords: ["On-Policy Distillation", "RLVR", "Reasoning LLMs", "Post-training", "Knowledge Distillation", "Reinforcement Learning", "Sequential Training"]
innovations: ["提出 OPD-then-RL 两阶段序列训练框架，系统超越所有单步联合优化基线", "从 pass@k、学习动态和参数更新三角度揭示 OPD 扩展能力边界、RL 进行 sharpening 的互补机制", "证明 OPD 作为 RL 冷启动优于 SFT，且验证分数是切换时机的有效信号"]
benchmarks: ["Reasoning-Gym Knights & Knaves", "Reasoning-Gym Zebra Puzzles", "Reasoning-Gym Countdown", "MATH-500", "AMC23", "AIME24", "AIME25"]
---

# 论文速读：Sequential-Beats-Joint-On-the-Interplay-between-On-Policy-Distillation-and-RLVR

## 一句话总结
论文提出 OPD-then-RL 序列训练框架，先在 On-Policy Distillation 阶段让学生模型学习教师分布，再切换至纯 RLVR 阶段强化高奖励行为，该两阶段策略在逻辑与数学推理任务上系统性地超越所有单步联合优化基线及纯 OPD/RLVR。

## 研究问题与动机
- **RLVR 奖励稀疏**：基于可验证奖励的强化学习（RLVR）仅提供序列级 outcome reward，对中间推理步骤缺乏细粒度监督，需大量 trial-and-error 才能涌现高奖励行为。
- **OPD 仅是行为代理**：On-Policy Distillation（OPD）在学生学习轨迹上提供密集的 token-level 监督，但优化的是师生分布差异而非真实任务目标，不能保证最大化任务性能。
- **联合优化存在信号干扰**：现有混合方法在单步内通过加权相加或教师调制融合两种信号，论文发现两种互补信号相互纠缠时会发生干扰，限制彼此潜力。
- **序列策略未被充分探索**：将 OPD 与 RLVR 分为两个独立阶段（先 OPD 后 RLVR）是一种简单但未被系统研究的替代方案，论文旨在验证其有效性并揭示其内在机制。

## 核心贡献（创新点）
- **统一视图与范式分类**：在 token-level 策略梯度框架下系统梳理现有 OPD-RLVR 混合方法，将其归纳为加权加性（weighted-additive）和教师调制（teacher-modulated）两大范式，厘清各类方法的设计逻辑。
- **提出 OPD-then-RL 序列框架**：首次系统证明两阶段顺序训练（OPD 后接纯 RLVR）在多个逻辑与数学推理基准上 consistently 超越纯 OPD、纯 RLVR 及所有联合优化变体，最大提升达 26.7 pass@1。
- **多维度机制解释**：从 pass@k 行为、学习动态和参数更新三个角度提供一致的解释——OPD 扩展学生覆盖的教师支持解空间，RL 在该空间内进行 sharpening；联合优化则因信号冲突牺牲能力扩展。
- **实用配方与经验发现**：指出 OPD 验证分数是切换至 RL 的关键信号，且 OPD 作为 RL 冷启动优于传统 SFT，为后续后训练流水线提供可直接采用的 recipe。

## 方法详解
- **统一 token-level 视角**：RL（GRPO）和 OPD 的目标均可转化为带特定 token-level advantage 的策略梯度上升形式，所有混合方法共享同一 PPO-clipped surrogate，仅优势函数和重要性采样比率不同。
- **加权加性范式**：将教师项作为独立求和项加入，如 KDRL 的 $A_t^{add} = \hat{A} + \beta d_t$，后续变体（KDRL-mask、SRPO、HDPO）根据 rollout 正确性或全错组选择性激活教师信号。
- **教师调制范式**：用教师信号调整奖励优势的幅度而不改变符号，如 TRRD 将教师信号融入重要性采样比率作为 trust-region 正则，RLSD 用 clipped 教师因子直接缩放优势值。
- **OPD-then-RL 两阶段设计**：前 S 步（默认 60 步）使用 OPD 目标 $\mathcal{L}_{OPD}$ 最小化师生轨迹分布的 reverse-KL 散度；之后切换至纯 GRPO 目标，仅依赖可验证奖励进行强化学习，完全移除教师信号以避免干扰。
- **切换点选择准则**：OPD 阶段验证集 pass@1 分数是决定何时切换的核心信号——分数仍在快速提升时继续 OPD 可提高 RL 上限，分数饱和后切换即可；论文默认选择第 60 步（OPD 快速提升阶段的末尾）。

## 实验与结果
- **数据集**：逻辑推理使用 Reasoning-Gym 的三个程序化任务（Knights & Knaves、Zebra Puzzles、Countdown），数学推理使用 DeepMath-103K 训练集及 MATH-500、AMC23、AIME24、AIME25 四个评测集。
- **模型配置**：教师模型 Qwen3-8B（non-thinking 模式），学生模型 Qwen3-1.7B-Base（附录 B 补充 Qwen3-0.6B-Base 验证泛化性），所有方法使用相同训练步数预算和统一超参数（学习率 1e-6、PPO clip 0.2、batch size 128 等）。
- **主要结果**：OPD-then-RL 在逻辑推理上平均 pass@1 达 80.6（K&K 92.6、Zebra 59.7、Countdown 89.6），超越所有联合方法 11.7–26.7 个点；在数学推理上平均 pass@1 达 31.8（MATH-500 70.1、AIME25 7.8），与最强基线持平或显著领先。
- **最强结果与提升幅度**：在 Knights & Knaves 任务上，OPD-then-RL 的 pass@1（92.6）较次优的 KDRL-Annealing（86.3）提升 6.3 个点，较纯 GRPO（28.1）提升 64.5 个点；数学推理 AIME24 上 pass@1 达 10.1，较纯 GRPO（6.7）提升 51%。
- **统计显著性**：配对 bootstrap 检验（20,000 次重采样）显示，OPD-then-RL 在数学推理 pass@1 上较六种基线显著领先，与三种最强基线持平；逻辑推理各任务均显著领先所有对比方法。

## 相关工作脉络
- **RLVR 推理训练**：以 DeepSeekMath、DeepSeek-R1 为代表，利用可验证奖励通过策略梯度直接优化推理能力，但稀疏奖励导致探索效率低；本文在其基础上引入密集教师信号并研究两者最佳组合方式。
- **知识蒸馏与 OPD**：传统 off-policy SFT 存在 exposure bias，OPD 通过在学生学习轨迹上提供 token-level 教师监督大幅提升学习效率，已广泛应用于 Qwen3、GLM-5、DeepSeek-V4 等工业后训练管线。
- **OPD-RLVR 联合优化方法**：KDRL、SRPO、HDPO 等将教师信号以加权相加方式融入 RL 优势函数；TRRD、RLSD 将教师信号用于调制优势幅度；本文指出这些方法因信号纠缠而产生干扰，性能受限。
- **SFT-then-RL 序列训练**：先前工作（如 Limozin et al., 2026）探索 off-policy SFT 作为 RL 冷启动，本文证明 on-policy OPD 作为冷启动优于 SFT，且在 OPD 后接 RL 可突破教师性能天花板。
- **教师增强 RL 策略**：部分工作先通过 RLVR 增强教师模型再蒸馏给学生，本文的 OPD-then-RL 可视为该策略的低成本近似，因 RL 阶段在学生已贴近教师分布的空间内进行 sharpening。

## 局限性与未来方向
- **教师-学生配置范围有限**：当前研究聚焦外部强教师设定，未探索任务特定教师、多教师设置及 on-policy self-distillation（教师为学生自身增强版本）等场景，OPD-then-RL 在这些设定下是否持续占优待验证。
- **冻结开箱即用的教师**：采用固定教师以保持监督信号稳定，但工业管线常先通过 RLVR 增强教师再蒸馏；论文指出 OPD-then-RL 可视为该策略的部分代理，但未进行计算预算对齐的受控比较。
- **OPD 目标函数单一**：仅使用标准的 token-level reverse-KL 目标，未探索其他目标如 full next-token distribution 上的 reverse-KL、forward-KL 或 Jensen-Shannon 散度，这些目标可能在模式覆盖和生成多样性上产生不同的冷启动效果。
- **学生模型规模限制**：实验主要在 0.6B–1.7B 学生模型上进行，虽已验证跨家族（OLMo）泛化，但在更大规模模型上的表现和 Scaling Law 特性尚未探索。

## 研究启发与可借鉴点
- **信号分离优于信号融合**：当两种训练信号具有不同作用机制（如 OPD 负责能力扩展、RL 负责行为 sharpening）时，顺序执行可能比单步联合优化更能避免干扰，这一设计思想可迁移至其他多目标后训练场景。
- **OPD 作为 RL 冷启动的价值**：OPD 不仅提供比 SFT 更强的初始化，且其 on-policy 特性使学生分布与教师对齐，为后续 RL 探索提供更优质的起点；该经验可直接用于构建推理模型的训练流水线。
- **pass@k 分析揭示机制**：通过评估不同采样数 k 下的性能变化，可清晰区分方法的能力扩展效应与概率集中效应，这种分析方法可用于诊断其他训练策略的内在行为。
- **切换点由验证分数决定**：实践中无需复杂调度策略，只需监控 OPD 阶段的验证集 pass@1，当其不再显著提升时切换至 RL 即可；这一简单准则具有普适性。
- **参数更新冲突度量**：提出的 sign-conflict rate（SCR）可量化不同训练目标对关键参数的方向一致性，为分析多目标训练的兼容性提供可复用的评估工具。

## 关键术语表
**On-Policy Distillation (OPD)**：在学生自身采样轨迹上进行的 token-level 教师蒸馏，最小化师生分布的 reverse-KL 散度，提供密集监督信号。
**Reinforcement Learning with Verifiable Rewards (RLVR)**：利用可验证的规则奖励通过策略梯度优化模型，典型算法如 GRPO，奖励稀疏但直接对齐任务目标。
**Pass@k**：对每个问题采样 k 个回答，至少有一个正确的比例；pass@1 衡量单次生成质量，pass@k（k>1）衡量能力边界和多样性。
**Weighted-Additive Paradigm**：将教师蒸馏优势和奖励优势以加权求和方式合并为单一 token-level 优势，如 KDRL、SRPO 等方法。
**Teacher-Modulated Paradigm**：用教师信号调整奖励优势的幅度而不改变其符号，教师仅作为 trust-region 正则或优势缩放因子，如 TRRD、RLSD。
**Sign-Conflict Rate (SCR)**：衡量某方法参数更新方向与 OPD 更新方向不一致的比例，用于量化不同训练信号间的冲突程度。
**Cold Start**：RL 训练前的初始化阶段，本文比较 OPD 和 SFT 两种冷启动策略对后续 RL 性能的影响。
**Learning Dynamics**：训练过程中验证 pass@1、KL 散度、熵等指标的演化曲线，用于揭示不同方法的优化行为和收敛特性。

## 可复现要素
- **数据集**：Reasoning-Gym（Knights & Knaves、Zebra Puzzles、Countdown）、DeepMath-103K（训练）、MATH-500、AMC23、AIME24、AIME25（评测）；数据集公开可获取。
- **代码开源**：论文未明确声明代码开源仓库；实验基于 VERL 框架实现。
- **模型权重**：使用 Qwen3-8B 作为教师、Qwen3-1.7B-Base 和 Qwen3-0.6B-Base 作为学生；模型权重需从官方渠道获取。
- **关键超参数**：学习率 1e-6、PPO clip ratio 0.2、batch size 128、rollouts 每组 8、梯度裁剪 1.0、OPD 训练步数 60、max response length 逻辑推理 4096/数学推理 8192；详见附录 Table 4。
- **训练细节**：全部在 VERL 框架上实现，vLLM rollout，FSDP sharding，bf16 精度；OPD 阶段不对 EOS token 位置计算教师信号（因 tokenizer 不匹配）。
