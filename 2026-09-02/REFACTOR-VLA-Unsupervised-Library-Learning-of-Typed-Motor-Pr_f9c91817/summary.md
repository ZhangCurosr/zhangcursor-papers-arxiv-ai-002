---
title: "REFACTOR-VLA-Unsupervised-Library-Learning-of-Typed-Motor-Pr"
source: https://arxiv.org/pdf/2609.01215v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 15:45:23"
field: "机器人技能学习与表示学习"
keywords: ["VLA", "skill discovery", "wake-sleep", "behavioral equivalence", "library learning", "LIBERO", "contrastive learning"]
innovations: ["基于世界模型rollout的行为等价核BEK，将bisimulation度量扩展至轨迹片段级", "类型化lambda程序发射器与库条件动作解码器的结合", "证明训练目标形状优于模型容量，InfoNCE辅助损失显著提升技能聚类质量"]
benchmarks: ["LIBERO-object", "LIBERO-spatial", "LIBERO-goal", "LIBERO-10"]
---

# 论文速读：REFACTOR-VLA-Unsupervised-Library-Learning-of-Typed-Motor-Pr

## 一句话总结
REFACTOR-VLA 提出了一种"醒/睡"架构的无监督技能发现框架，通过基于潜在世界模型 rollout 的行为等价核（BEK）对连续动作片段进行聚类，并结合类型化 lambda 程序提取可重用技能库。关键发现是：世界模型的训练目标形状（而非模型容量）才是决定技能表示质量的核心因素，添加 InfoNCE 对比损失后在全部 4 个 LIBERO 套件上超越最强基线（平均 NMI 提升 Δ = +0.184）。

## 研究问题与动机
- **长期任务的技能分解困境**：当前 VLA 模型（如 OpenVLA、π₀、RT-2）为"单体式"架构，直接生成原始动作命令，难以处理长视野多步任务，且学习到的行为缺乏可解释性和可重用性。
- **行为等价性的判定难题**：现有技能发现方法（如 AtomicVLA、AtomSkill、BLADE、LRLL）要么依赖对比嵌入聚类，要么依赖未校准到机器人动力学的 LLM 判断等价性，无法识别"产生相同物理效果但动作路径不同"的片段。
- **醒/睡框架在连续动作空间的迁移障碍**：DreamCoder 等符号域的成功经验难以直接应用于连续动作，因为物理演示极少共享相同轨迹，需要动力学感知的等价度量。
- **世界模型容量的误区**：社区存在"更大世界模型总是更好"的假设，本文通过对照实验证明模型容量并非决定技能聚类质量的关键。

## 核心贡献（创新点）
- **行为等价核（BEK）**：将 MDP 状态级 bisimulation 度量扩展至连续轨迹片段，通过潜在世界模型 M_φ 的 rollout 计算值函数差与 k 步 Wasserstein 距离的组合发散度量 D_φ(τ, τ')，实现动力学校准的行为等价性判定。
- **类型化 lambda 发射器与库条件动作解码器（LCAD）**：在 Hindley–Milner 类型系统约束下生成结构化合规的 lambda 程序，结合 rectified-flow 解码器实现技能条件的动作生成，首次在生产 LIBERO 数据上得到结构化任务语言技能库。
- **MDL 门控联合 admission 标准**：候选抽象必须同时满足 BEK 声理性（ε=0.05）、回报保留（K_v=32 次验证 rollout）和最小描述长度增益（>4 nats）三重条件，确保策略重构的性能损失有理论界。
- **三阶段醒/睡交替训练循环**：世界模型预热（Phase A）→ 唤醒阶段策略优化（Phase B）→ 睡眠阶段技能发现（Phase C），形成端到端的自洽闭环。
- **训练目标形状优于模型容量的实证结论**：将 M_φ 从 188M 扩至 430M 参数在全部 4 个套件上导致 NMI 下降，而添加 SupCon InfoNCE 辅助损失在 188M 参数下实现 4/4 套件全面超越最强基线（mean Δ = +0.184）。

## 方法详解
**整体架构**：REFACTOR-VLA 包含四个核心模块——主干 VLA 策略 π_θ（含 Typed Program Emitter TPE 和 Library-Conditioned Action Decoder LCAD）、潜在世界模型 M_φ、类型化 λ 程序库 F_t、以及程序使用后验 ρ_t。

**世界模型 M_φ**：采用 DreamerV3 风格分层世界模型，冻结 DINOv2-base 视觉编码器，因果 Transformer 处理交错 [img, state, action] token，包含后验 q_φ(z_t|x_t, h_{t-1}) 和前验 p_φ(z_t|h_{t-1}) 头，重建目标在 DINOv2 特征空间。默认配置 188.16M 总参数（101.58M 可训练），LIBERO 无奖励列故 w_ret=0。

**行为等价核 BEK**：对于 MDP M=(S,A,P,R,γ) 和轨迹片段 τ=(s₀,a₀,...,s_H)，定义：
$$D_φ(τ, τ') = w_R \mathbb{E}_{s~ρ}|V_φ^τ(s) - V_φ^{τ'}(s)| + w_E \mathbb{E}_s \mathcal{W}_2(P_φ^k(·|s,τ), P_φ^k(·|s,τ'))$$
其中 V_φ^τ(s) 为在 s 处插入 τ 后的期望回报，P_φ^k 为 k 步潜 rollout 分布。使用固定 k KMeans 聚类，k 匹配各 LIBERO 套件的任务数。

**Siamese 摊销器 k_χ**：2.4M 参数 Siamese Transformer 生成 L2 归一化嵌入，pairwise BEK 用 1-<k_χ(τ), k_χ(τ')> 近似。从冻结 Phase A M_φ 进行严格特征蒸馏（MSE），梯度仅来自学习到的动力学，无任务标签参与。

**Typed Program Emitter（TPE）**：因果解码器 Transformer，Hindley–Milner 类型词汇 Σ={Twist, Wrench, GripperPhase, Pose, Lang}，grammar e::=prim | λx:τ.e | e₁e₂ | seq[e₁,...,eₖ] | repeat(e,n) | branch(c,e₁,e₂)。每步 beam 解码时使用 Robinson 统一器进行类型检查。

**Library-Conditioned Action Decoder（LCAD）**：4 层 Transformer（d_model=384，~10.9M 可训练参数），接收类型化项和当前状态，输出 16 步 action chunk，使用 rectified-flow matching（10 Euler steps）。

**醒/睡交替循环**：每轮外迭代 t=1,...,T 执行三步：
- **Wake**：训练 TPE（交叉熵损失）和 LCAD（rectified-flow MSE）
- **Sleep**：训练 BEK 头 k_χ 对抗 M_φ rollout 蒸馏目标
- **Library refactor**：top-down 语法反统一，admission 三条件：BEK 声理性 ≤ ε_t、回报保留 ≤ ε=0.05（K_v=32 验证）、MDL 增益 > δ_MDL=4 nats，选取 top K_max=8 候选，修剪低使用条目

**Return Preservation 理论保证（Lemma 1）**：若 admitted abstraction e 在每个调用点满足 |V_φ^τ(s) - V_φ^e(s)| ≤ ε，且 M_φ sup-norm 回报误差 ≤ η，则用 e 重构 π 每调用导致的真实期望回报退化上界为 (ε+2η)/(1-γ)。

## 实验与结果
**数据集**：LIBERO 完整基准套件，含四个 LeRobot v3 子集：libero_object_image、libero_spatial_image、libero_goal_image、libero_10_image（LONG）。

**评估基线**：AtomicVLA、AtomSkill、BLADE、LRLL，以及 ~7B OpenVLA 冻结提取器。

**主要结果**：
- Phase C BEK NMI（n=3 multi-seed）：Object 0.462±0.021、Spatial 0.867±0.025、Goal 0.915±0.013、LIBERO-10 0.754±0.010，均值 0.749
- 相比最强基线平均提升 Δ = +0.184（Object +0.114、Spatial +0.250、Goal +0.201、10 +0.170）
- 跨提供者 seed 收敛（n=12 pairs）：mean NMI=0.705，95% CI [0.683, 0.729]
- InfoNCE 辅助损失使 NMI 相对提升 mean +0.252（+50.7% relative），恢复 task_index supervised UB 的 26–98.5%

**容量反证实验**：M_φ 从 188M 扩至 430M（LR 调优后 Phase A loss 严格占优：0.3555 vs 0.4176），Phase C NMI 在 4/4 套件上全面下降（Object -0.040、Spatial -0.146、Goal -0.018、10 -0.090），证伪"容量假设"。

**技能库发现**：在 libero_object 上发现首个结构化 LIBERO 任务语言库——3 个抽象（1211 nats），LCAD 使用其中 2 个，成功重写全部 256/256 采样演示。运动基元子空间 Σ_motor 在所有 MDL 阈值下 admit 0 个抽象。

## 相关工作脉络
- **AtomicVLA / AtomSkill**：基于 Gumbel-gated MoE 路由或 VLM 提名关键帧进行聚类，等价性算子未校准到系统动力学；REFACTOR-VLA 用 BEK 提供动力学感知的等价度量。
- **BLADE / LRLL**：依赖未校准的 LLM 判断等价性或合成 PDDL 式前置/后置条件；REFACTOR-VLA 的理论保证来自 bisimulation metric 而非 LLM 判断。
- **DreamCoder / LILO**：在离散符号域成功的 wake/sleep 库学习框架，依赖 syntactic anti-unification；REFACTOR-VLA 将 anti-unification 应用于连续动作片段，用 BEK 替代纯语法距离。
- ** Castro & Zhang 的 bisimulation 度量**：定义状态级发散 D(s,s')=w_R|R(s)-R(s')|+w_E W₂(P(·|s),P(·|s'))；REFACTOR-VLA 将其从单状态扩展至轨迹片段级 D_φ(τ,τ')。
- **Option-Critic / CompILE / Relay Policy Learning**：时序抽象选项学习经典方法，等价性算子作用于单状态值或单 token 分割；REFACTOR-VLA 在片段级动力学上进行聚类。

## 局限性与未来方向
- **任务标签依赖**：Phase A 的 InfoNCE 辅助损失需要 task_index 标签，在开放语料（如 OXE-mixed）中不可用；自监督 episode_contrast 在 libero_10 上失败。
- **Wasserstein 项的关键作用**：去除 W₂ 分量导致 libero_object NMI 下降 61%（0.462→0.110），说明 k 步 rollout 分布匹配比值函数差更重要。
- **运动基元库无法发现抽象**：Σ_motor 子空间在所有 MDL 阈值下 admit 0 个抽象，原因在于 typed-lambda token 词汇表缺乏非平凡 anti-unification 的 lifting 结构。
- **跨提供者 0.70 阈值处于边界**：n=12 时 95% CI 下限 0.683 低于 0.70 严格门控，仅 libero_goal 干净通过。
- **LIBERO 无奖励列**：导致 sup-norm 回报误差 η 无法测量，仅在 RecursivePourEnv 合成环境中测得 η_sup=0.205±0.039。
- **未来方向**：Real-robot 迁移至 RoboCasa、跨 embodiment 泛化、label-free contrastive sampler 改进。

## 研究启发与可借鉴点
- **训练目标形状优先于模型容量**：对于技能表示学习，设计有 discriminative 信号的训练目标（如 InfoNCE）比简单扩大世界模型更有效，这一结论可迁移至其他 representation learning 场景。
- **行为等价性的动力学校准思路**：用 learned world model rollout 替代 LLM 判断或纯对比嵌入聚类，为 continuous action skill discovery 提供了理论严谨的等价性定义。
- **MDL 门控与回报保留的联合 admission 机制**：为库学习中的抽象质量控制提供了可组合的三条件框架，兼顾压缩效率与性能保障。
- **跨提供者 seed 收敛评估协议**：n=12 pairwise NMI + bootstrap CI 的评估方式提供了更稳健的技能发现方法比较基准。
- **类型化程序诱导与连续动作的结合**：Hindley–Milner 类型约束下的 lambda 程序发射器展示了符号结构如何与连续控制接口桥接。

## 关键术语表
- **Behavioral-Equivalence Kernel (BEK)**：基于潜在世界模型 rollout 的轨迹片段发散度量 D_φ，结合值函数差与 k 步 Wasserstein 距离，用于动力学校准的聚类。
- **Wake/Sleep Architecture**：交替优化两阶段的训练范式，wake 阶段用自底向上识别连接训练策略，sleep 阶段用自顶向下生成连接发现并压缩技能。
- **Typed Lambda Emitter (TPE)**：在 Hindley–Milner 类型约束下生成结构化合规 lambda 程序的因果解码器，词汇表覆盖 Twist/Wrench/GripperPhase/Pose/Lang。
- **Library-Conditioned Action Decoder (LCAD)**：以类型化程序为条件的 rectified-flow 动作解码器，输出 16 步 action chunk。
- **Minimum Description Length (MDL) Gate**：抽象 admission 的压缩效率门控，要求候选抽象的 MDL 增益超过阈值（δ_MDL=4 nats）。
- **Return-Preservation Gate**：保证策略重构不显著降低性能的门控，要求 BEK 发散 ≤ ε（ε=0.05）在 K_v 次验证 rollout 上成立。
- **Siamese Amortizer (k_χ)**：2.4M 参数的 Siamese Transformer，从冻结 M_φ 蒸馏得到，用于高效近似 pairwise BEK。
- **Supervised Contrastive InfoNCE Loss**：Phase A 世界模型预热的辅助对比损失，最大化片段隐变量与任务标签的互信息，改善下游聚类质量。

## 可复现要素
- **数据集**：LIBERO benchmark suite（4 个子集），公开可用；RecursivePourEnv 合成环境
- **代码/权重**：论文未明确声明开源状态，但提供了完整的 checkpoint 路径描述和实验协议细节
- **关键超参**：
  - M_φ：188.16M 参数（d_model=1024, 8 layers），LR=3×10⁻⁴（188M）/1×10⁻⁴（430M retune）
  - k_χ：2.4M 参数，2000 BEK steps
  - LCAD：10.9M 参数（d_model=384, 4 layers）
  - Phase A：4000 SGD steps，DDP，BF16
  - BEK：w_R, w_E 权重，k=10（短套件）/k=9（libero_10）
  - Admission：ε=0.05, K_v=32, δ_MDL=4 nats, K_max=8
  - InfoNCE：w_infoNCE=0.5, τ=0.1
- **硬件**：8×H100 GPUs（训练），单 H100（评估）
