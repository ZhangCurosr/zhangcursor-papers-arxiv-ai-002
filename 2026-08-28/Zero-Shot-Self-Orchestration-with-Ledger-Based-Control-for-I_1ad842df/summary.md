---
title: "Zero-Shot-Self-Orchestration-with-Ledger-Based-Control-for-I"
source: https://arxiv.org/pdf/2608.26480v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 22:14:01"
field: "多智能体大模型系统与代码生成"
keywords: ["multi-agent LLM", "zero-shot orchestration", "manager-worker scaffold", "code generation", "LiveCodeBench", "cost-effectiveness", "shared workspace"]
innovations: ["训练无关的零样本 manager-worker 编排框架，动态任务分配无学习组件", "共享文件系统账本外化 agent 状态，配合 sample-test verifier 与截断摘要器", "系统量化 scaffold 的成本-精度 tradeoff，证明以更低成本逼近顶级模型"]
benchmarks: ["LiveCodeBench v6 hard split (LCB-100)"]
---

# 论文速读：Zero-Shot Self-Orchestration with Ledger-Based Control for Improved LLM Coding Performance

## 一句话总结
本文在无训练、无基准调优的前提下，提出一种基于共享文件系统账本的 manager-worker 编排框架，在 LiveCodeBench 上对 9 个模型（9B～2.8T 参数）进行系统评估，证明该脚手架能显著提升中等/较小模型的代码生成准确率，且以更低成本逼近顶级闭源模型的性能。

## 研究问题与动机
- **多智能体收益混杂**：现有文献中 multi-agent 是否优于 single-model 的证据混合，且对比往往同时改变了 token 预算、工具调用、prompt 等多个变量，难以归因。
- **训练依赖未知**：编排策略的收益是否需要学习（trained orchestrator）或任务特定微调？本文回答：零样本固定 prompt 即可产生显著收益。
- **成本-精度权衡不清**：加 scaffold 花费约三倍 token，但换来的是比升级到更大模型更便宜的精度提升路径——论文系统量化了这一 tradeoff。
- **上下文管理缺失**：长输出易被截断（truncation）或模型陷入自我验证循环（looping），共享账本可将状态外化、降低单次调用负载。

## 核心贡献（创新点）
1. **训练无关的零样本自编排框架**：manager 使用固定 prompt 动态规划、分配任务、决定终止；与 Fugu/Conductor 等需 RL 训练协调器的本质区别在于完全无学习组件。
2. **共享文件系统账本（Ledger Workspace）**：用 plan.md / tasks.json / notes.md / solution.py 五个文件作为跨调用持久状态，替代单一对话历史；与 ARIADNE/LbMAS 的黑板架构相比，无需 MCTS 或 reward model，是更轻量的纯任务列表循环。
3. **v2 脚手架的三项增量设计**：sample-test verifier（步骤 5 执行公开测试用例反馈给 manager）、cut-off summarizer（截断 worker 的半成品思路摘要）、size-bounded workspace prompt（有界大小避免 prompt 膨胀）；原版 scaffold 缺少这三项。
4. **发现并修复 LiveCodeBench 评测器 Bug**：`sys.stdin.buffer.readline()` mock 实现错误导致 311 个输出被误判，修复后重评分数，论文所有数字均基于修正版。
5. **系统化的成本-精度分析**：通过 cap-matching replay 隔离截断效应，证明 GPT-5.6-Terra + manager（$11.71/pass）几乎匹敌 Fable 5 单调用（$61.11/pass），Qwen3.8-27B + manager（$51.75）可本地自托管。

## 方法详解
**工作空间（Workspace）文件**：
- `<ws>/task.md`：问题陈述
- `<ws>/plan.md`：manager 撰写的总体策略（3-6 句）
- `<ws>/tasks.json`：任务列表，每项含 `{id, desc, status, result}`
- `<ws>/notes.md`：累积的思路/发现/部分证明
- `<ws>/solution.py`：当前最优代码

**控制流（v2 脚手架，共 6 步）**：
1. **Manager - plan**：读取问题，写策略 + 3-6 个种子任务。
2. **Worker - brainstorm**：首个 worker 不写代码，只识别核心难点、候选方法及陷阱，追加到 `notes.md`。
3. **Manager - manage（循环）**：整合计划与头脑风暴，去重、标记完成、添加新子任务；若问题已解则终止，否则指定下一个任务。
4. **Worker - do the task**：新鲜 worker 执行单一任务，重写 `solution.py`，追加记录并提议剩余步骤。
5. **Verifier - 运行样本测试**：执行程序与公开样例对比（stdin 格式，覆盖 73/100 题）；pass/fail 反馈给 manager 作为 ground truth，失败则强制进入 fix-or-switch 任务。
6. **Finalizer**：循环因预算耗尽/manager 无新任务时触发，产出最终解；若已有干净答案则跳过。

**安全 guard**：
- `MAX_ITERS = 10` 轮 manager→worker 周期
- 无进展守卫：manager 重复分配同一任务即停止
- 截断摘要器：hit cap 的 worker 由短调用将其半成品思路摘要

**温度设置**：plan 写作为 0.3，brainstorm 为 0.4，任务执行与任务列表整理为 0.2。

**Baseline**：同一模型、同一温度（0.2）、同一 problem，单次调用，无共享工作空间、无循环、无其他角色。

## 实验与结果
**数据集**：LiveCodeBench（release_v6）hard split，最新 100 道竞赛编程题（LCB-100）。

**模型（9 个）**：
- 开放权重：Qwen3.5-9B、Qwen3.6-35B-A3B、Qwen3.8-27B（本地 vLLM FP8）
- 闭源前沿：Opus-5、GPT-5.6-Terra、GPT-5.6-Luna（OpenAI API）、Claude Fable 5（Anthropic API）、Kimi-K3（~2.8T MoE）、Minimax-M3（428B）

**主要结果（§2.1，128k cap，reasoning ON，5 passes，v2 scaffold）**：

| 模型 | Single | Manager | Δ |
|------|--------|---------|---|
| GPT-5.6-Terra | 77.0 ± 1.0 | **85.0 ± 1.0** | **+8.0** |
| GPT-5.6-Luna | 67.2 ± 4.3 | **77.8 ± 2.0** | **+10.6 ± 5.1** |
| Qwen3.8-27B | 63.0 ± 4.1 | **86.4 ± 2.7** | **+23.4 ± 6.6** |
| Opus-5 | 85 | **91** | +6（单 pass） |

- 最强单调用：**Claude Fable 5 = 87.4 ± 1.1**
- Qwen3.8-27B + manager 达 86.4，与 Fable 5 差距 1.0 点（p=0.73，不显著）
- 截断救援：Qwen3.8-27B 单调用在 128k 下被截断 150/500 次，manager 仅 5/500 次；35 个无代码的失败中 manager  rescued 25 个（贡献 ~5 点，占 Δ 的约 1/5）
- GPT-5.6-Terra + manager 以 **$11.71/pass** 逼近 Fable 5 的 **$61.11/pass**（p=0.59 不差）
- Qwen3.8-27B + manager **$51.75/pass**，可本地自托管

**Reasoning OFF 结果（§2.4，OpenRouter，16k ×5 passes）**：
- Kimi-K3：**+30.4**（p<10⁻⁴）；128k 下单 pass **+42**
- Minimax-M3：**+11.0**（p<10⁻⁴）；128k 下单 pass **+12**
- Qwen3.5-9B：**+7.2**（p<10⁻³）
- Qwen3.6-35B：**-1.2**（16k）/ **-9**（128k）—— 唯一显著退步

## 相关工作脉络
1. **Fugu（Sakana AI）** 与 **Conductor（ICLR 2026）**：训练式协调器，RL 学习编排策略；本文是其训练无关对应物，回答"多少收益无需训练即可获得"。
2. **MetaGPT / ChatDev / AutoGen / CAMEL**：固定流水线式多智能体，角色顺序预定义；本文 manager 每步动态重新策展任务列表并自主决定终止，非静态 pipeline。
3. **Trinity（ICLR 2026）**：固定三角色结构 + 进化策略训练的角色分配；本文无任何学习，仅有固定 prompt 的零样本编排。
4. **ARIADNE / LbMAS**：黑板驱动的多智能体；本文共享文件系统账本类似黑板，但采用单 manager 的简单任务列表循环，无 MCTS、无 reward model。
5. **Mixture-of-Agents（ICLR 2025）**：多层聚合合成；本文不涉及层级聚合，是串行 manager-worker 循环。
6. **Wang et al. (EMNLP 2024)** 与 **Tran & Kiela (2026)**：等 token 预算下的 multi-agent 公平比较；本文不试图等 token 比较，而是问"额外 token 是否买来更好解"，并进一步比较不同模型+scaffold 的组合性价比。

## 局限性与未来方向
- **服务提供方可靠性噪声**：OpenRouter 路由导致 §2.4 部分结果的 run-to-run 变异；§2.1 通过 pinned backend 缓解，但未完全消除。
- **单一基准家族**：仅评测竞赛编程（LiveCodeBench）；AIME/MATH/GPQA 等数学基准因顶模接近天花板无法测量 improvement。
- **Qwen3.6-35B 退步**：reasoning off 时 -9 点，说明 deliberation 可能将正确方案引入更差方向（§4.4 案例分析）。
- **Qwen3.5-9B reasoning ON 不可用**：返回 reasoning-only 空响应，限制 scaffold 对小模型的适用边界。
- **未来方向（论文自述）**：引入 fresh-perspective workers（无上下文的独立 worker 防止错误锚定）和 comprehensive test verifier（全量生成测试或形式化验证替代仅依赖公开样例）。

## 研究启发与可借鉴点
1. **零样本编排的可行性**：固定 prompt + 共享账本即可产生显著增益，无需训练协调器；可迁移至其他需要多步推理的任务（如数学证明、文档分析）。
2. **Context 外化的通用设计**：将状态写入有界文件（而非累积对话历史）是控制 prompt 膨胀的有效手段，适用于任何 multi-turn agent 系统。
3. **截断救援机制**：cut-off summarizer 将半成品思路摘要化而非丢弃，是一个值得复用的容错设计，尤其在 API 输出 token 受限场景。
4. **评测器缺陷的公平性影响**：本文揭示的 `sys.stdin.buffer.readline` mock bug 提醒：任何 benchmark 复用第三方 harness 时需检查代码风格兼容性，否则 leaderboard 会产生系统性偏差。
5. **成本-精度 Frontier 分析方法**：cap-matching replay + 单变量对照实验设计，可推广至其他 agent 系统的 ablation 研究。

## 关键术语表
**Zero-shot self-orchestration**：推理时编排，协调器无需训练且无任务特定演示，仅凭固定 prompt 动态决策。
**Ledger-based control / Shared filesystem workspace**：用 plan/task/notes/solution 等文件作为跨 agent 调用的持久化状态存储，替代对话上下文。
**Manager-worker scaffold**：单个 manager agent 负责规划与任务分配，多个 worker agent 各自执行单一子任务，通过共享工作空间协调。
**Pass@1**：N 次独立尝试中至少一次成功的概率估计，此处 N=1（单 pass）或 N=5（5 passes 均值）。
**Cap-matching replay**：将高 token cap 下生成的输出截断至目标长度后重新评分，以隔离截断效应而不重新运行模型。
**Verifier-gated loop**：v2 脚手架特有设计，每次 worker 产出候选代码后执行公开测试样例，pass/fail 结果反馈给 manager 作为终止/继续依据。
**Thinking tokens / Reasoning effort**：模型生成的链式思考文本，按输出 token 计费并占用 max_tokens 预算。
**Finish-reason=length**：模型输出达到 token 上限被强制截断，与正常结束（stop）相对。

## 可复现要素
- **数据集**：LiveCodeBench release_v6 hard split（100 题）—— [论文声明公开](https://github.com/LiveCodeBench/LiveCodeBench)
- **代码**：论文未提供官方仓库链接，但描述足够详细；workspace 文件结构与控制流可在 §3.1 完整复现
- **权重**：Qwen3.8-27B-FP8（HuggingFace 公开）；其余为 API 模型
- **关键超参**：`MAX_ITERS=10`，manager temperature=0.3/0.4，worker temperature=0.2，output cap=128k（pinned）/16k（OpenRouter），thinking on/off
- **硬件**：Qwen3.8-27B 本地 vLLM 推理；其余通过 OpenAI/Anthropic/OpenRouter API
- **Evaluator 修复**：论文附带 `sys.stdin.buffer.readline` mock 补丁描述，修复后的分数已重算
