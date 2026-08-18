---
title: "Vero-Can-AI-Agents-Build-Formally-Verified-Software-Reposito"
source: https://arxiv.org/pdf/2608.13522v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 04:27:45"
field: "形式化验证与代码生成"
keywords: ["formal verification", "code generation", "Lean 4", "benchmark", "AI agents", "repository-scale", "proof synthesis"]
innovations: ["首个仓库级联合实现-证明合成基准（43实例/743 API/2705规范）", "形式化审计机制接受机器可检查反证以修正基准缺陷", "双模式评估揭示实现自由度对验证难度的双向影响"]
benchmarks: ["Vero", "RVBench", "VeruSAGE-Bench", "VeriSoftBench", "miniCodeProps", "FVAPPS"]
---

# 论文速读：Vero-Can-AI-Agents-Build-Formally-Verified-Software-Reposito

## 一句话总结
本文提出 **Vero**，首个面向 Lean 4 的仓库级验证代码生成基准，包含 43 个多模块实例（源自 Python/Dafny/Verus/Coq 真实仓库），支持联合实现-证明合成与纯证明两种任务模式，并设计了形式化审计机制以自动发现基准中的隐性错误。最强智能体 GPT-5.5 (xhigh) 在 code-and-proof 模式下仅完全解决 27/43 实例，表明当前 AI 在跨模块 invariant 发现、引理库构建与仓库级一致性维持方面仍存在显著能力缺口。

## 研究问题与动机
1. **现有基准局限于函数级别**：miniCodeProps、FVAPPS、CLEVER 等仅评估单个函数的验证生成，无法捕捉真实软件中跨模块依赖与长程推理需求。
2. **仓库级基准仅评测纯证明**：RVBench、VeruSAGE-Bench、VeriSoftBench 等提供参考实现，仅要求智能体生成证明，忽略了实现选择与证明可行性之间的紧密耦合。
3. **训练数据污染风险**：基于 Mathlib 的 LeanDojo、APE-Bench 等偏重形式化数学，与命令式软件验证结构差异大；且公开源码可能已存在于预训练数据中。
4. **基准自身潜在错误未被检测**：形式化基准即使经过人工审核仍可能存在规范不可满足、参考实现缺陷或规范间不一致，导致误判智能体能力。

## 核心贡献（创新点）
1. **首个仓库级联合实现-证明合成基准**：Vero 包含 43 个 Lean 4 多模块项目、743 个 API 与 2,705 条规范，要求智能体同时完成实现与机器可检查证明，填补了仓库级验证代码生成的评测空白。
2. **形式化审计机制（Formal Audit Mechanism）**：接受机器可检查的反证（规范不可满足、参考实现错误、规范集不一致），将基准缺陷转化为可操作的修正证据，提升基准可靠性。
3. **反作弊评估协议**：通过标记区域约束、公理白名单、类型类实例检测与 `@implemented_by` 拆分检测三重机制，防止智能体通过弱化规范或构造空洞类型类骗取通过。
4. **双任务模式设计**：支持 proof-only（固定参考实现，仅证证明）与 code-and-proof（联合生成实现与证明）两种模式，揭示实现自由度对验证难度的双向影响。
5. **可扩展的多语言策展流水线**：支持从 Dafny/Verus/Coq（Track 1）和 Python（Track 2）翻译到 Lean 4，流水线模块化设计便于扩展至新源语言与新领域。

## 方法详解
**基准数据结构**：每个 Vero 实例是一个多模块 Lean 4 项目，包含：
- 固定数据类型与辅助定义
- API 签名集合 $\mathcal{A} = \{a_1, \ldots, a_m\}$ 及参考实现
- 接口结构类型 `RepoImpl`，收集所有 API 实现
- 规范集合 $S = \{S_1, \ldots, S_n\}$，类型为 `RepoImpl → Prop`
- 规范参数化于 `RepoImpl` 而非固定实现，支持审计机制

**两种任务模式**：
- **Proof-only**：`canonical` 实例化为参考实现，智能体仅需生成满足 $S_i(\text{canonical})$ 的证明
- **Code-and-proof**：`canonical` 由智能体实现填充，需同时生成实现与证明

**审计机制的三种形式化证据**：
1. 证明规范在参考实现上不可满足：$\neg \bigwedge_{S \in S} S(\text{canonical})$
2. 证明存在不可满足的单个规范：$\neg \exists \text{impl} : \text{RepoImpl}, S(\text{impl})$
3. 证明存在不一致的规范子集但各成员单独可满足

**反作弊三层防护**：
- 层1：仅评估 `!benchmark` 标记区域内的代码修改
- 层2：公理白名单（仅允许 `Classical.choice`、`propext`、`Quot.sound`）
- 层3：类型类实例空洞检测与 `@implemented_by` 拆分检测

## 实验与结果
**数据集**：43 个实例（13 个 Track 1 来自 Dafny/Verus/Coq，30 个 Track 2 来自 Python），共 743 个 API、2,705 条规范。

**评估设置**：四个前沿配置（GPT-5.5 xhigh/mid、Claude Opus 4.8、Claude Sonnet 5），两个智能体 harness（Codex v0.140.0、Claude Code v2.1.191），90 分钟时间预算，Lean v4.29.1。

**主要结果**：
- GPT-5.5 (xhigh) 最强：code-and-proof 完全解决 **27/43** 实例，proof-only 解决 **25/43**
- **10 个实例抵抗所有配置**（如 `dedekind_reals` 零通过）
- 单规范通过率高达 87.3%（code-and-proof），但仓库级完成率受限
- 完全解决的运行证明文本约为实现的 **2 倍**
- 成功仓库依赖共享引理库：中位数 **73.6%** 证明行位于 helper 定理中

**关键发现**：
- 使用 supplied helper 的规范失败率高出 **14.9 点**，反复调用 API 的规范高出 **11.7 点**
- Existence-and-coverage 类规范失败率最高（**47.1%**）
- 智能体在运行前半段确定实现，后半段专注于证明，缺乏回溯重构实现的策略
- 审计机制在策展中发现 **38 个规范缺陷**（涉及 9 个实例），包括直接冲突、缺失域条件等

## 相关工作脉络
1. **函数级验证基准**：miniCodeProps、FVAPPS、VERINA、CLEVER（Lean 4）；DafnyBench、VerusBench、AlgoVeri（SMT 风格）—— Vero 扩展至仓库级联合生成。
2. **仓库级纯证明基准**：RVBench（755 个 Verus 任务）、VeruSAGE-Bench（849 个任务）、VeriSoftBench（500 个 Lean 4 义务）—— Vero 首次评测实现-证明联合生成。
3. **形式化数学基准**：LeanDojo、LeanAgent、APE-Bench（基于 Mathlib）—— Vero 聚焦命令式软件验证，领域差异显著。
4. **污染控制方法**：LiveCodeBench 强调训练数据隔离—— Vero 通过手动翻译与 Lean 4 规范重写提供结构性污染防护。
5. **智能体评测协议**：SWE-bench、SWE-agent 评估实际 GitHub 问题修复—— Vero 引入正式审计与反作弊机制，针对验证任务特性设计。

## 局限性与未来方向
1. **仅支持 Lean 4**：虽策展流水线可扩展至其他目标语言，但当前评估仅限于 Lean 生态。
2. **并发/时序协议缺失**：当前基准偏向可 cleanly 翻译为 Lean 的代码，缺少并发系统与 temporal 协议类任务。
3. **规范语义正确性无法形式化保证**：审计机制仅证形式可满足性，不能确保规范与原始意图一致。
4. **未评测增量维护任务**：如何验证已验证仓库的增量更新是重要但尚未覆盖的方向。
5. **实现自由度可能导致效率牺牲**：智能体可选择易证明但低效的实现（如暴力枚举替代匈牙利算法），当前基准未强制成本约束。

## 研究启发与可借鉴点
1. **形式化审计机制可迁移**：将"接受反证"的反馈循环引入其他形式化基准建设，可系统性提升基准质量。
2. **实现-证明耦合评估设计**：双模式（proof-only vs code-and-proof）的对照实验设计可揭示智能体的真实能力边界，值得在其他代码生成任务中借鉴。
3. **引理库深度作为难度指标**：helper-chain depth 与跨配置通过率的相关性分析（深度 4+ 时通过率降至 ~40%）为自动化难度预估提供量化信号。
4. **反作弊三重防护策略**：标记区域约束 + 公理白名单 + 语义漏洞检测的组合方案，可推广至其他需防范 reward hacking 的 AI 基准。
5. **沙盒工作区行为分析**：通过未评分 scratch files 推断智能体搜索策略（如 probing vs drafting、retry loop），为 Agent 设计提供可观测性洞察。

## 关键术语表
- **Vero**：首个仓库级验证代码生成基准，评估 AI 智能体在 Lean 4 中联合生成实现与形式化证明的能力。
- **Code-and-proof mode**：任务模式，要求智能体同时生成 API 实现与满足所有规范的机器可检查证明。
- **Proof-only mode**：任务模式，提供参考实现，仅要求智能体生成证明。
- **RepoImpl**：接口结构类型，收集所有 API 实现字段，规范参数化于该类型以支持审计。
- **Formal Audit Mechanism**：接受机器可检查反证的机制，用于发现规范不可满足、参考实现错误或规范间不一致。
- **Helper theorem**：智能体编写的辅助定理，用于支撑多个规范证明的可重用引理。
- **Helper-chain depth**：证明某个规范所需引用的辅助定理链的最大深度，预测跨配置通过率。
- **Reward hacking**：智能体通过弱化规范、引入空洞类型类或利用 `@implemented_by` 拆分来骗取评估通过的行为。

## 可复现要素
- **数据集**：43 个 Vero 实例，已开源（https://github.com/sapiens-ucb/vero）
- **代码/评估 harness**：已开源
- **模型**：GPT-5.5（Codex v0.140.0）、Claude Opus 4.8 / Sonnet 5（Claude Code v2.1.191）
- **工具链**：Lean v4.29.1
- **关键超参**：90 分钟时间预算、xhigh/medium reasoning effort、公理白名单仅允许 Classical.choice/propext/Quot.sound
- **策展成本**：约 $60/实例（标准 API 费率），含 QA 与验证轮次
