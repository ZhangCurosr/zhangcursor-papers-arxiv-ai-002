---
title: "REFACTOR-VLA-Unsupervised-Library-Learning-of-Typed-Motor-Pr"
source: https://arxiv.org/pdf/2609.01215v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 19:28:21"
---

# 论文速读：REFACTOR-VLA-Unsupervised-Library-Learning-of-Typed-Motor-Pr

## 一句话总结
本文提出 REFACTOR-VLA，一种基于醒眠（wake/sleep）架构的无监督技能发现系统，通过动力学校准的行为等价核（BEK）在睡眠阶段聚类轨迹片段，并在觉醒阶段由类型化 Lambda 程序库驱动 rectified-flow 动作解码器执行策略；实验证明，世界模型 $M_\phi$ 的训练目标形状（而非参数量容量）是决定技能表示质量与跨基准泛化的核心杠杆。

## 研究问题与动机
- 现有 VLA 模型（OpenVLA、$\pi_0$、RT-2、RDT-1B 等）为单体架构，直接输出原始电机命令或短动作序列，缺乏可复用、可解释的高层行为抽象，导致长 horizon 任务表现退化。
- 现有技能发现工作回避了“何时两条动作序列是行为等价的”这一核心难题：AtomicVLA/AtomSkill 依赖聚类对比嵌入，BLADE/LRLL 依赖未校准于机器人自身动力学的 LLM 判断，无法区分物理等效但动作路径不同的片段。
- 传统醒眠框架与程序归纳方法（如 DreamCoder）仅在离散符号域有效，物理演示极少共享完全相同的 token 序列，句法反同构无法直接迁移至连续动作空间。
- “世界模型越大越好”的容量假设缺乏实证支撑，盲目放大 $M_\phi$ 可能引入冗余表征，反而损害下游技能聚类的可分性。

## 核心贡献（创新点）
1. **行为等价核（BEK） formulation**：将 MDP 状态级 bisimulation 散度推广至连续轨迹片段级，定义 $D_\phi(\tau,\tau')$ 为潜在世界模型 rollout 的值函数差与 k 步 Wasserstein 距离的加权和，并给出固定 k KMeans 聚类 NMI 收敛速率的理论保证（Theorem 2）。与仅依赖状态值或句法距离的基线本质不同。
2. **Typed Program Emitter (TPE) + 库条件动作解码器（LCAD）**：在 Hindley–Milner 类型约束下生成结构化合法的类型化 $\lambda$ 项，LCAD 以 rectified-flow matching 方式根据活跃库条目生成动作 chunk。与 DreamCoder/LILO 的纯句法压缩不同，本文以行为等价性替代语法同一性作为程序归纳驱动力。
3. **MDL 门控的醒眠交替循环**：睡眠阶段通过语法反同构提取候选抽象，仅当同时满足 BEK 声理性、回报保持（$\varepsilon=0.05$）与 MDL 增益（$>4$ nats）时才 admit 入库，为策略重构提供了形式化的性能损失上界（Lemma 1）。
4. **揭示“目标形状 vs 容量”双杠杆机制**：实证证伪容量假设（188M→430M 使 NMI 在 4/4 suite 下降），并证明在 Phase A 添加轻量级 SupCon InfoNCE 辅助损失可全面提升聚类质量，平均超越最强基线 $\Delta = +0.184$ NMI。

## 方法详解
- **四模块架构**：主干 VLA 策略 $\pi_\theta$（
