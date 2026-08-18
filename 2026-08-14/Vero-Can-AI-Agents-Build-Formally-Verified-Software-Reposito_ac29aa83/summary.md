---
title: "Vero-Can-AI-Agents-Build-Formally-Verified-Software-Reposito"
source: https://arxiv.org/pdf/2608.13522v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 04:28:12"
field: "形式化软件验证与AI代码生成"
keywords: ["verified code generation", "repository-level benchmark", "formal verification", "Lean 4", "AI agent", "code and proof synthesis"]
innovations: ["首个仓库级 Lean 4 验证代码生成基准，支持代码+证明联合合成", "形式化审计机制允许 Agent 提交负证据纠正基准缺陷", "双模式评估（proof-only 与 code-and-proof）分离评估证明技能与代码-证明协同能力"]
benchmarks: ["Vero (43 instances, 743 APIs, 2705 specs)", "RVBench", "VeriSoftBench", "miniCodeProps", "FVAPPS"]
---

# 论文速读：Vero-Can-AI-Agents-Build-Formally-Verified-Software-Reposito

## 一句话总结
论文提出了 **Vero**，首个面向 **Lean 4** 的**仓库级（repository-level）**验证代码生成基准测试，通过支持“代码+证明”联合合成任务与形式化审计机制，评估前沿 AI 编码 Agent 在真实多模块软件库上同时生成实现与机器检查证明的能力，发现当前最强 Agent 仅能完全解决 27/43 个实例。

## 研究问题与动机
1. **现有基准局限于函数级别**：已有关键字生成基准（如 miniCodeProps, FVAPPS 等）主要评估单个函数的验证，无法反映真实软件中跨模块依赖与长程推理需求。
2. **缺乏代码-证明联合评估**：已有的仓库级基准（如 RVBench, VeriSoftBench）仅提供参考实现，仅评估证明生成，忽略了代码实现选择与证明策略之间的强耦合关系。
3. **基准本身可能存在隐藏错误**：形式化规范与参考实现可能包含细微错误，直接作为 ground truth 会导致对 Agent 的不公平惩罚，需要可纠正的基准构建机制。
4. **污染风险与评估协议缺失**：多数基准直接使用公开代码库，存在训练数据污染风险；且缺乏针对 Agent 工具访问（文件编辑、构建命令、Lean 工具链）的完整评估协议与反作弊机制。

## 核心贡献（创新点）
1. **首个仓库级 Lean 4 验证代码生成基准 Vero**：包含 43 个源自真实仓库的多模块实例，覆盖 Python、Dafny、Verus、Coq 等多种源语言，共计 743 个评分 API 和 2,705 个规格说明，支持“代码+证明”与“纯证明”两种任务模式。
2. **形式化审计机制（Formal Audit Mechanism）**：允许 Agent 提交机器检查的反证（如规范不可满足、参考实现错误、规范集不一致），将基准隐藏缺陷转化为可修复证据，实现基准质量的持续改进。
3. **可扩展的多阶段策展流水线**：设计了从源仓库到 Lean 4 实例的半自动翻译流程，每阶段经人工审核，支持多种源语言技能扩展，并确保规格说明源于上游文档与测试套件。
4. **全面的 Agent 评估与深入分析**：在 GPT-5.5、Claude Opus 4.8 等前沿配置下评估，揭示当前 Agent 在跨模块不变式发现、引理库构建、实现-证明协同方面的瓶颈，并提供失败案例的结构性分析。

## 方法详解
- **数据格式**：每个 Vero 实例是一个多模块 Lean 4 项目，包含固定的数据类型定义、API 签名集合 $\mathcal{A}$、接口结构类型 `RepoImpl`、规范集合 $S$（类型为 `RepoImpl → Prop`）以及规范实现 `canonical`。规范参数化于 `RepoImpl` 而非固定实现，以支持审计机制。
- **任务模式**：
  - **Proof-only**：`canonical` 实例化为参考实现，Agent 仅需为每个规范 $S_i$ 生成 Lean 4 证明。
  - **Code-and-proof**：`canonical` 由 Agent 自行填充，Agent 需同时生成所有 API 的实现体并为每个规范生成证明。
- **策展流水线**：分为 Track 1（来自 Dafny/Verus/Coq 等形式语言）和 Track 2（来自 Python）。包含 discover、select、plan、translate、spec writing（仅 Track 2）、validate 等阶段，每阶段需人工审核。
- **形式化审计机制**：接受三类机器检查的负证据：
  1. 证明参考实现在 proof-only 模式下不满足规范（公式 1）；
  2. 证明存在单个规范不可满足（公式 2）；
  3. 证明存在规范子集互相不一致但各自可满足（公式 3）。
- **反作弊机制**：三层防御：
  1. **插槽范围重渲染**：仅提取 `!benchmark` 标记内的内容，其余部分恢复为原始内容；
  2. **公理白名单**：仅允许 Lean 标准公理，拒绝依赖 `sorry` 或用户声明公理的证明；
  3. **声明筛查**：规则预过滤 + LLM 裁判检测空主体类型类实例、优先级覆盖、可判定性洗白、证明目标与运行函数分离等作弊模式。

## 实验与结果
- **数据集**：43 个实例，Track 1（13 个，来自 Dafny/Verus/Coq）平均 36 API、92.8 规范；Track 2（30 个，来自 Python）平均 9.2 API、50 规范。总计 743 API、2,705 规范。
- **评估基线**：四个前沿 Agent 配置：GPT-5.5 (xhigh)、GPT-5.5 (medium)、Claude Opus 4.8、Claude Sonnet 5，均在 Codex 和 Claude Code 环境中，拥有完整工具访问（文件系统编辑、构建、Lean 工具链）。
- **主要结果**：
  - **最强配置**：GPT-5.5 (xhigh) 在 code-and-proof 模式下完全解决 **27/43** 实例，proof-only 模式下解决 **25/43** 实例。
  - **10 个实例**抵抗所有配置（两种模式均无法完全解决）。
  - **单规范通过率**：GPT-5.5 (xhigh) 在 code-and-proof 中通过 87.3% 规范，proof-only 中通过 85.8% 规范，但高单规范通过率不意味着仓库级完成。
  - **模式差异**：172 个实例-Agent 配对中，26 个在两种模式下均完全解决，13 个仅在 code-and-proof 中解决，17 个仅在 proof-only 中解决，116 个在两种模式下均未解决。
  - **成本效率**：GPT-5.5 (xhigh) 在 code-and-proof 中每完全解决实例花费 **$106**，为最便宜配置；而弱 Agent 每完全解决实例成本更高（$182–$464）。
- **失败分析**：
  - 失败规范主要集中在**存在性与覆盖性规范**（failure rate 47.1%），这类规范涉及全局属性，无法通过单调用分情况证明。
  - 深度引理链（helper-chain depth ≥ 4）的规范通过率大幅下降至 39.1%–50.6%。
  - 强 Agent 尝试 hardest obligations 但失败，弱 Agent 则未触及这些规范。

## 相关工作脉络
1. **函数级验证基准**：miniCodeProps、FVAPPS、VERINA、CLEVER（Lean 4）；DafnyBench、VerusBench、AlgoVeri、VeriCoding、VerifyThisBench、VeriEquivBench（SMT 验证器）——仅评估单函数，无法捕捉跨模块依赖。
2. **仓库级证明生成基准**：RVBench、VeruSAGE-Bench、VeriSoftBench（Verus/Rust）、CoqStoq、FSCQ 案例研究——仅提供参考实现，仅评估证明生成，忽略代码-证明耦合。
3. **数学定理证明基准**：LeanDojo、LeanAgent、APE-Bench——基于 Mathlib，侧重数学证明，与软件验证的推理结构不同。
4. **污染与评估协议**：LiveCodeBench 关注训练数据污染；本文通过从源语言翻译为 Lean 4 且无公开 Lean 4 ground truth 来缓解污染，并引入完整 Agent 工具访问与反作弊机制。
5. **审计机制**：本文形式化审计机制区别于 prior benchmarks 仅依赖静态 ground truth，允许 Agent 提交机器检查的反证来纠正基准错误。

## 局限性与未来方向
1. **目标语言单一**：当前仅支持 Lean 4，虽策展流水线可扩展至其他目标语言，但尚未在 Coq/Isabelle/HOL 等其他定理证明器上评估。
2. **源仓库选择偏差**：偏向易于翻译为 Lean scaffold 的代码，缺失并发/时序协议等难以移植的领域。
3. **规范语义正确性依赖人工审核**：审计机制仅保证形式可满足性，不能确保规范语义正确或完整，需依赖人工审查与上游形式化交叉验证。
4. **增量维护任务未覆盖**：未评估 Agent 在已有验证代码库上进行增量修改、保持证明有效性的能力。
5. **性能与正确性权衡**：code-and-proof 模式中 Agent 可能用低效但易证明的实现替换参考实现（如 munkres 中用指数级排列枚举替换匈牙利算法），未对复杂度进行约束。

## 研究启发与可借鉴点
1. **多阶段策展流水线设计**：结合 LLM 自动化翻译与人工审核的每阶段门禁机制，可迁移至其他形式化基准构建，平衡规模与质量。
2. **形式化审计机制**：允许 Agent 提交负证据来纠正基准缺陷的思路，可用于构建自我演进的基准，适用于任何需 ground truth 的评估场景。
3. **双模式评估协议**：同时提供 proof-only 与 code-and-proof 模式，可分离评估 Agent 的证明技能与代码-证明协同能力，为未来基准设计提供对比框架。
4. **反作弊多层防御**：插槽范围重渲染、公理白名单、声明筛查（规则+LLM）的组合，可作为 Agent 编码基准的标准安全实践。
5. **实现自由度的双面性分析**：通过人工审计发现 Agent 可用更简单实现替换难证明参考实现，提示未来基准需考虑算法效率、资源约束等多维规范。

## 关键术语表
- **Vero**：首个仓库级 Lean 4 验证代码生成基准，评估 Agent 在多模块代码库上联合生成实现与证明的能力。
- **Code-and-proof mode**：Agent 需同时生成所有 API 实现并为每个规范生成机器检查证明的任务模式。
- **Proof-only mode**：提供参考实现，Agent 仅需生成规范证明的任务模式。
- **RepoImpl**：接口结构类型，收集所有 API 实现，作为规范的参数化变量。
- **Formal Audit Mechanism**：接受机器检查反证（规范不可满足、参考实现错误、规范集不一致）以纠正基准缺陷的机制。
- **Helper theorem**：Agent 为支持多个规范证明而编写的共享辅助引理，构成可重用引理库。
- **Helper-chain depth**：证明一个规范所需辅助引理的最大递归深度，深度越大规范越难证明。
- **Axiom allowlist**：仅允许 Lean 标准公理的审查机制，拒绝依赖 `sorry` 或用户声明公理的证明。

## 可复现要素
- **数据集**：43 个 Lean 4 实例，来源于 13 个 Dafny/Verus/Coq 仓库（Track 1）和 30 个 Python 仓库（Track 2），已公开于 GitHub。
- **代码/权重**：基准、策展流水线、评估 harness 均开源（https://github.com/sunblaze-ucb/vero）。
- **关键超参**：评估时长预算 90 分钟；Lean 版本 v4.29.1；Agent 配置包括 GPT-5.5 (medium/xhigh)、Claude Opus 4.8、Claude Sonnet 5 (均 xhigh)；推理努力程度通过 Codex/Claude Code 平台参数控制。
- **环境**：Agent 具备完整文件编辑、构建命令、Lean 工具链访问权限。
