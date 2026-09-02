---
title: "Science-sandboxes-measure-the-scientific-capability-of-AI-ag"
source: https://arxiv.org/pdf/2608.30165v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 02:31:40"
field: "AI for Science"
keywords: ["science sandbox", "AI agent", "scientific reasoning", "MPRA", "protein folding", "rule discovery", "benchmark"]
innovations: ["提出science sandbox框架区分指标优化与规则推断", "构建MPRAbox和CodonBox两个生物领域沙盒", "实证揭示前沿AI在陌生规则空间中科学推理退化"]
benchmarks: ["MPRAbox", "CodonBox"]
---

# 论文速读：Science sandboxes measure the scientific capability of AI agents

## 一句话总结
论文提出"science sandboxes"框架，通过在封闭循环中让AI agent执行实验、接收反馈、修订假设，来区分"优化指标"与"真正理解规则"的差异；在MPRAbox（调控序列设计）和CodonBox（发明生物规则发现）两个沙盒中，前沿AI能超越人类基线实现定量优化，但在脱离熟悉生物先验的规则空间中科学推理能力迅速退化。

## 研究问题与动机
- **核心问题**：如何评估AI agent是否具备真正的科学推理能力，而非仅仅通过局部搜索优化指标分数？
- **现有基准不足**：传统benchmark只关注最终分数，无法区分"找到高分样本"与"理解系统规则"。
- **科学进步的本质**：不仅在于找到解决方案，更在于理解"为什么有效"的规则，并用这些理解设计更优实验。
- **AI当前局限**：前沿模型在熟悉领域表现优异，但一旦规则超出预训练生物先验，其推理能力会急剧下降。

## 核心贡献（创新点）
- **提出science sandbox框架**：建立"实验-反馈-假设修订"闭环，通过记录agent的实验选择与推理轨迹来评估科学能力。
- **构建MPRAbox用于调控序列设计评估**：让agent设计50,000条200bp增强子库以训练泛化性序列-活性模型，使用14个隐藏测试集（含5个真实实验标签、9个oracle标签）综合评估。
- **构建CodonBox用于发明规则发现评估**：设计隐藏遗传规则的简化蛋白质折叠世界，测试agent从纯实验数据推断未知翻译规则的的能力。
- **提出wet/damp/dry三层oracle设计**：湿实验（物理）、潮湿模型（数据驱动模拟）、干燥规则（人为发明），形成从真实到虚构的评估光谱。
- **首次实证揭示前沿AI的"优化-理解鸿沟"**：Claude等模型能在熟悉生物学任务中超越人类基线，但在陌生规则空间中只能机械优化而无法推断真正规则。

## 方法详解
- **沙盒通用结构**：包含specimens（可提交测试的实体，如DNA序列）、assays（测量属性方法）、sealed oracle（隐藏反馈机制）。每轮agent提交样本，oracle返回报告，agent更新lab notebook并选择下一轮实验。
- **MPRAbox核心流程**：
  - 每轮agent生成N=50,000条200bp序列库；
  - 使用Malinois模型（776,474条MPRA实验数据训练的CNN）预测每条序列在K562/HepG2/SK-N-SH三种细胞系中的活性；
  - 用该库从零训练sequence-to-activity模型（相同架构，不同权重）；
  - 在14个隐藏测试集上计算Pearson r，返回均值作为综合评分；
  - 支持单轮（M=1）和多轮（M=30）模式。
- **CodonBox核心流程**：
  - 基于Dill's HP lattice protein folding模型，16个残基链在二维格点上最大化非连续H-H接触数；
  - 隐藏密码子表将核苷酸序列翻译为H/P链，密码子长度k∈{2,3,4}，字母表大小j∈{4,6,8}；
  - 部分世界设置silent positions或interacting positions增加推断难度；
  - agent每轮提交一条序列，oracle返回fitness score（最高9分）。
- **三类oracle**：Wet（真实实验）、Damp（Malinois模拟）、Dry（14种人为发明规则如GC balance、Fibonacci positions、English words cipher等）。

## 实验与结果
- **单轮无先验**：Claude Opus 4.7中位性能r=0.774，GPT-5.5为r=0.655，Gemini 3.5 Flash为r=0.680；最佳人类策略均值r=0.763。
- **单轮有先验**：Claude提升至r=0.781，GPT提升至r=0.760，Gemini提升至r=0.751。
- **30轮多轮**：Claude全部4次运行超越最佳人类策略，定量提升 modest；agent逐步发展出结构化探索策略（如控制变量、GC分层）。
- **Dry oracle**：Claude无法推断任何隐藏规则（包括English word cipher、Fibonacci rule等），仅能实现分数优化。
- **CodonBox**：Claude在10轮内达到最大fitness=9；在j=4,k=2/3时正确推断密码子结构；j=6时完全无法发现结构；遇到silent+interaction组合规则时只能记忆有效codon而无法归纳一般规则。
- **核心结论**：定量优化与规则推断是两种独立能力，当前AI在前者上已超越人类基线，在后者上仍显著不足。

## 相关工作脉络
- **Empowering biomedical discovery with AI agents**（Gao et al., 2024）与**Accelerating scientific discovery with co-scientist**（Gottweis et al., 2026）：聚焦AI agent在生物医学发现中的工具执行能力，缺少对推理过程的系统性评估。
- **DiscoveryBench**（Majumder et al., 2024）和**LAB-bench**（Laurent et al., 2024）：衡量LLM在生物学研究任务上的工具使用能力，但未区分优化与理解。
- **PaperBench**（Starace et al., 2025）和**BixBench**（Mitchener et al., 2025）：分别评估AI复现研究与计算生物学能力，评估对象仍是固定任务而非开放科学探究过程。
- **Virtual lab of AI agents**（Swanson et al., 2025）：多agent系统自动设计SARS-CoV-2纳米抗体，侧重于执行而非科学推理评估。
- 本文定位差异：首次建立可区分"指标优化"与"规则推断"的闭环评估框架，强调保留agent推理轨迹进行定性分析。

## 局限性与未来方向
- Wet实验成本高、耗时长、存在噪声；Damp oracle继承训练数据偏差；Dry oracle缺乏真实物理复杂性。
- 仅评估了单一agent（Claude）在多轮实验中的表现，未系统比较多种架构。
- 定性评估依赖人工或独立AI judge阅读lab notebook，尚未实现完全自动化。
- 未来可构建更多领域的science sandbox（化学、物理学等），并通过BroadBox开源平台促进社区共建；开发自动化qualitative evaluation pipeline。

## 研究启发与可借鉴点
- **保留推理轨迹作为评估维度**：lab notebook机制为后续研究提供了可复用范式，可将agent的假设形成、实验设计、理论修订过程结构化记录并量化评估。
- **控制先验知识暴露**：通过改变任务 framing（MPRA-framed/unframed/symbolic）分离生物先验与推理能力的影响，值得在其他领域借鉴。
- **分层oracle设计**：wet/damp/dry三层次可从不同角度解耦系统的可重复性、可扩展性与可解释性评估需求。
- **多轮迭代中的"控制变量"策略**：agent自发发展出"一次只改变一个变量"的实验设计原则，可作为agent训练中的启发式引导信号。
- **与团队结合机会**：可将此框架迁移至材料科学、药物发现等领域，构建领域特定的science sandbox，评估团队内部开发的agent系统的科学推理能力。

## 关键术语表
- **Science sandbox**：封闭式实验测试床，agent通过多轮实验-反馈-假设修订循环学习系统规则。
- **Oracle（wet/damp/dry）**：反馈机制，湿实验来自真实物理测定，潮湿来自训练模型模拟，干燥来自人为发明规则。
- **MPRA（Massively Parallel Reporter Assay）**：高通量报告基因实验，并行测量数千至数百万DNA序列的调控活性。
- **Malinois**：训练于776,474条MPRA数据的深度卷积神经网络，用于预测序列活性。
- **CodonBox**：简化遗传系统沙盒，agent需从实验反馈推断隐藏的密码子翻译规则。
- **HP lattice model**：Dill提出的蛋白质折叠简化模型，将氨基酸分为疏水(H)和极性(P)，在二维格点上最大化H-H接触。
- **Lab notebook**：agent记录的实验推理轨迹，包含假设、观察、理论修订，用于定性评估科学能力。
- **Scientific reasoning deterioration**：指agent在脱离熟悉生物先验的规则空间中，从"理解规则"退化为"机械优化"的现象。

## 可复现要素
- **数据集**：MPRAbox使用Gosai et al. episomal MPRA数据、UK Biobank/GTEx variant数据、Sei注释、DHS index、随机基因组窗口和合成序列；全部来自公开来源。
- **代码开源**：是，https://github.com/asr2210/science-sandbox
- **模型权重**：Malinois预训练权重来自原始发表（Gosai et al., 2024）
- **关键超参**：MPRA库大小N=50,000，序列长度200bp；CodonBox残基数16，最多500轮查询；下游模型训练batch size=512，Adam lr=3.27×10⁻³，CosineAnnealingWarmRestarts调度。
- **Agent配置**：Claude Opus 4.7（Claude Code）、GPT-5.5（Codex CLI）、Gemini 3.5 Flash（Gemini CLI），默认provider参数，无第三方scaffold。
