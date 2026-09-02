---
title: "StudyBench-Can-Self-Evolution-Squeeze-Textbooks-for-Olympiad"
source: https://arxiv.org/pdf/2609.00787v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 22:18:47"
---

# 论文速读：StudyBench-Can-Self-Evolution-Squeeze-Textbooks-for-Olympiad

## 一句话总结
本文提出了 **StudyBench**，一个受控的物理学科基准，用于直接测量自演化方法将固定教材转化为可迁移解题能力的效率；实验发现现有主流方法的 Application Set 提升几乎无法传递到 Olympiad 难度的 Transfer Set，且存在显著的 **Guidance Gap** 与 **Compute Plateau**，证明剩余瓶颈是算法设计问题而非数据或算力问题。

## 研究问题与动机
- 自演化（Self-evolution）的核心承诺是模型无需依赖有限高质量数据，即可持续吸收环境新知并将其转化为可迁移的独立解题能力，而非仅仅记忆或复述。
- 现有静态高难考试（如 AIME、Humanity’s Last Exam）只能给出最终分数，混淆了算法、训练数据与基础模型的贡献；动态/终身学习基准侧重交互流内的局部适应，无法测量“经验→可迁移能力”的转化效率。
- 直接测量该转化面临三大障碍：**能力差距消失**（Base Model 已能稳定解决）、**目标不可达**（训练材料不足以推导答案）、**归因混淆**（方法/数据/模型变量耦合），目前尚无同时消除这三点的评估体系。
- 因此缺乏一个可控、可归因、且能直接量化“材料吸收→能力迁移”效率的基准，严重阻碍了自演化方法的横向对比与进展评估。

## 核心贡献（创新点）
- **提出 StudyBench 受控基准**：构建包含 11 本经典物理教材（三层训练材料）与两级测试集（Application Set / Transfer Set）的标准化评测环境，从设计上同时保证 Capability Gap、Reachability 与 Controlled Attribution。
- **定义并量化 Guidance Gap 与 Compute Plateau**：通过教科书上下文指导消融实验揭示“方法内化能力”与“外部提示可达上限”之间的巨大鸿沟；通过计算曲线 profiling 证明各方法在耗尽算力前早已饱和。
- **系统 benchmark 多类自演化方法**：在三种不同能力层级的基础模型（Llama-3.2-3B-Instruct、Qwen3-8B、Opus 4.7）上评测 corpus/SFT、标签 RL、上下文演化、无数据自对弈等代表方法，首次实证吸收提升几乎不迁移到奥赛难度。
- **开源评测框架与防污染管线**：发布代码与指令层数据集，提供双教师（DeepSeek V4 Pro / GLM-5.1）可达性验证、两阶段规则+LLM 验证器以及彻底的训练材料脱敏流程。
- **与已有工作的本质区别**：不同于 SE-Bench、NewtonBench 等依赖开放或提交者自选数据的基准，StudyBench 固定训练语料与测试项并设置指导天花板，将“算法贡献”从“数据/模型贡献”中严格剥离，使自演化进步成为可测量的科学目标。

## 方法详解
- **训练材料分层**：11 本物理教材经 MinerU 解析后提取三层嵌套材料：Corpus（原始段落，~6.0M tokens）、Instructions without Answer（646 题）、Instructions with Answer（1,420 题），覆盖主流自演化方法的监督范式。
- **能力过滤器（Capability Filter）**：以 Qwen3-8B（thinking mode）在 pass@8 下跑两轮池子，仅保留其稳定失败的题目；教材题按子学科均衡采样，极端稀疏学科额外
