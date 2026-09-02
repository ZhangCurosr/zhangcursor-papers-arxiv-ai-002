---
title: "Making-Prospective-Memory-SLM-Shaped-Typed-Intention-Stores"
source: https://arxiv.org/pdf/2609.01272v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 19:25:36"
field: "Agent Memory / Prospective Memory"
keywords: ["prospective memory", "small language models", "agent memory", "training-free scaffolding", "typed intention store", "PM-Bench"]
innovations: ["首次系统性形式化 Agent 前瞻性记忆操作，定义类型化意图三元组 (φ,α,σ)", "提出 PIS 训练无关架构，通过 Form-Revise-Filter-Decide 四算子循环实现 PM，无需 PEFT 或蒸馏", "首次在小模型上验证 PM 能力，Gemma-E2B 从 4.2% 跃升至 66.2%，DeepSeek-Chat 达 82.9% Set-F1 刷新 SOTA"]
benchmarks: ["PM-Bench"]
---

# 论文速读：Making Prospective Memory SLM-Shaped: Typed Intention Stores for Small-Model Agents

## 一句话总结
本文提出**前瞻性意图存储（PIS）**框架，将 Agent 的前瞻性记忆（PM）建模为类型化的状态追踪任务，通过 Form-Revise-Filter-Decide 四个算子的训练无关（training-free）循环实现"延迟承诺→触发执行"的核心能力；在 PM-Bench 上，DeepSeek-Chat 达到 82.9% Set-F1 刷新 SOTA，Gemma-E2B 等小模型从不足 7% 跃升至 66.2%，显著超越现有大模型基线。

---

## 研究问题与动机

1. **前瞻性记忆（PM）是 Agent 的关键短板**：现有工作集中于回顾性记忆（过去知识检索/缓存），而"在合适时机执行延迟意图"的能力被严重忽视；即使前沿 LLM 在 PM-Bench 上也仅达 65.1% Set-F1。
2. **现有 Memory×SLM 系统未覆盖 PM**：已发表的小模型记忆管线（如 LightMem-SLM、DuoMem）依赖选择器 LoRA 或教师轨迹蒸馏，且仍服务于回顾性场景，不维护可更新的待办集合（due set）。
3. **PM 本质上是 schema-constrained 的状态追踪**：与 SLM 擅长的受限工具调用同构，若将动作空间类型化，小模型即可胜任，无需额外训练。
4. **缺乏系统化的 PM 形式化与开源评测**：Agent 领域的 PM 研究稀少，公开基准仅有 PM-Bench，TriggerBench 等也存在前瞻性能远低于回顾性能的差距。

---

## 核心贡献（创新点）

1. **首次系统性形式化 Agent 的前瞻性记忆操作**：将延迟承诺建模为类型化三元组 $I = (\varphi, \alpha, \sigma)$，定义 due-set 决策算子作用于外部存储，区别于回顾性存储的 $sim(e,q)$ 优化目标。
2. **提出 PIS（Prospective Intention Store）训练无关架构**：通过 Form–Revise–Filter–Decide 四步循环实现 $(\pi, F)$ 映射，无需 PEFT、选择器微调或教师蒸馏，直接可用冻结 backbone 推理。
3. **首次研究 SLM 在前瞻性记忆上的能力**：不仅让 DeepSeek-Chat 超越已发表的大模型 scaffold（82.9% vs 65.1%），更令 Gemma-E2B 等小模型从 4.2% 提升至 66.2%，同时保持低 token 消耗与低延迟。

---

## 方法详解

### 3.1 问题形式化

- **意图三元组**：$I = (\varphi, \alpha, \sigma)$，其中 $\varphi$ 为触发条件，$\alpha$ 为预期动作，$\sigma \in \{\text{pending}, \text{done}, \text{canceled}\}$ 为状态。
- **隐式通道（hidden channels）**：触发条件常依赖 portal/dashboard/sensor 等不可见状态，需主动查询 $c \in \mathcal{C}_t$ 后合并入证据 $V_t$。
- **理想待办集**：$D_t^\star = \{ \alpha \mid I \in \mathcal{P}_t,\, \sigma = \text{pending},\, V_t \models \varphi \}$，评估指标为 trajectory Set-F1。
- **决策函数**：$\hat{D}_t = \pi(\mathcal{P}_t, V_t)$，更新函数：$\mathcal{P}_{t+1} = F(\mathcal{P}_t, V_t, \hat{D}_t)$。

### 3.2 Store and Decision Interface（核心公式）

$$
\hat{D}_t = \pi(\mathcal{P}_t, V_t),\quad \mathcal{P}_{t+1} = F(\mathcal{P}_t, V_t, \hat{D}_t),\quad D_t^\star = \{ \alpha \mid I \in \mathcal{P}_t,\, \sigma=\text{pending},\, V_t \models \varphi \}
$$

时钟谓词由规则判定；事件/通道谓词由语言 judge 在证据充足后判定。

### 3.3 四大算子

| 算子 | 功能 | 执行方式 |
|------|------|----------|
| **Form** | 将叙事文本拆解为类型化行 $(\varphi, \alpha, \text{pending})$ 并入存储 | $\mathcal{P}_t \gets \mathcal{P}_t \cup \text{Form}(V_t)$ |
| **Revise** | 对 pending 意图应用单条 typed patch：reschedule（改写 $\varphi$）、override（改写 $\alpha$）、cancel（设 $\sigma=$ canceled） | $\rho_t(I; V_t)$ 返回修改后的 $I'$ |
| **Filter** | 按结构规则（日期/时间窗口/离散事件标签）将 $\mathcal{P}_t$ 缩减为候选资格板 $B_t$；未查询的 channel 意图保留为 check 目标而非丢弃 | $B_t = \{ I \in \mathcal{P}_t \mid \sigma=\text{pending},\, \text{eligible}(I; V_t) \}$ |
| **Decide** | 先由 channel judge 请求 $c \in \mathcal{C}_t$ 查询（Observation），再基于 $(B_t, V_t, X_t)$ 输出待办集 $\hat{D}_t$，并将已执行意图标为 done | $\kappa(I; \hat{D}_t)$ 更新 $\sigma$ |

**关键设计**：
- Filter 和 $\kappa$ 为纯结构操作；Form、Revise、channel judge、Decide 需要语言 grounding。
- Revise 是 belief update（更新已有意图），非检索——这与回顾性记忆的本质差异。
- Observation 是主动性 channel 查询，模拟人类主动查看 portal/sensor 的行为。

### Algorithm 1：单步 PIS 流程

```
1: P_t ← P_t ∪ Form(V_t)
2: P_t ← Revise(P_t, V_t)
3: B_t ← Filter(P_t, V_t)
4: while channel judge on B_t requests c ∈ C_t do
5:     V_t ← V_t ∪ {c}   # 查询隐藏通道
6: end while
7: (D̂_t, P_{t+1}) ← Decide(B_t, V_t, X_t)
```

---

## 实验与结果

### 数据集与基准
- **PM-Bench**（Liu & Gabriel, 2026）：7 天模拟活动流，含匿名前瞻菜单、诱饵动作（lure）、隐藏状态通道；主指标为 micro Set-F1（预测 vs 真值待办集），辅助指标：update miss、cross-day miss、FA/step。

### 主要结果（DeepSeek-Chat）

| Setup | Set-F1 | Update miss | Cross-day miss | FA/step |
|-------|--------|-------------|----------------|---------|
| single（无存储） | 67.7 | 55.6 | 57.1 | 6.3 |
| + Naive RAG | 46.5 | 88.9 | 100.0 | 57.5 |
| + Mem0 | 48.7 | 100.0 | 57.1 | 6.3 |
| + A-Mem | 54.1 | 100.0 | 42.9 | 8.8 |
| + Letta | 52.2 | 66.7 | 71.4 | 25.0 |
| + LightMem-style | 51.5 | 66.7 | 57.1 | 38.8 |
| + MemoryOS-style | 58.3 | 66.7 | 42.9 | 8.8 |
| **+ PIS（本文）** | **82.9** | **22.2** | **0.0** | 8.8 |

**关键发现**：所有七种回顾性记忆基线均低于 single（67.7%），PIS 是唯一大幅超越的，Set-F1 达 **82.9%**（相对最佳回顾基线 +24.8 个百分点），且 update miss 仅 22.2%、cross-day miss 为 0.0%。

### 小模型结果（Gemma-E2B）

| Setup | Set-F1 |
|-------|--------|
| single | 4.2 |
| 七种回顾性记忆（最高） | 6.6 |
| **+ PIS** | **66.2** |

Gemma-E2B 在 PIS 加持下提升约 **10×**，cross-day miss 降至 28.6%，证明类型化 intent store 对 SLM 极度有效。

### 其他 SLM（Qwen3.5-4B / Qwen3-8B）

- Qwen3.5-4B + PIS：**70.1%**（高于 Qwen3-8B + PIS 的 57.2%），因 3.5-4B checkpoint 专为 agentic 任务设计，对 typed board 遵循更可靠。
- 回顾性基线在 Qwen 系列上整体失效，无法稳定超越 single。

### 计算开销（Table 3）

- PIS 的 wall-clock 与 token 开销中等（DeepSeek: 16.4 min / 2.08M tokens），显著低于 A-Mem（22 min / 7.84M）和 Letta（23.9 min / 8.11M），但远高于 naive single。
- 开销主要来自 channel enrichment 与更长 board 带来的 choose/query 调用增加，但换来的是最高的 Set-F1。

---

## 相关工作脉络

1. **回顾性 Agent 记忆**（MemGPT/Letta、Mem0、A-Mem、LightMem、MemoryOS）：以 $sim(e,q)$ 为核心，存储 episodic/semantic notes 供后续检索，未覆盖"触发-执行"的时间条件逻辑；PIS 与此本质不同，存储对象为可修订的 $(\varphi,\alpha,\sigma)$ 而非静态 notes。
2. **Memory×SLM 系统**（LightMem-SLM、DuoMem）：需 selector LoRA 或 teacher distillation，服务于回顾性任务；PIS 零训练，直接作用于前瞻性场景。
3. **SLM 受控工具调用**（TinyAgent、Octopus）：1B–7B 模型在 typed 接口上匹配大模型；本文将此类思想扩展到 PM 这一新领域。
4. **Cognitive science PM 研究**（Einstein & McDaniel, 1990）：区分事件型与时基型延迟意图，为 PIS 的类型化 trigger 设计提供理论基础。
5. **PM-Bench / TriggerBench**（Liu & Gabriel, 2026；Zhang et al., 2026b）：首次孤立评测 Agent 的 PM 能力，揭示 LLM 在该技能上的可靠性不足，PIS 直接针对此 gap。

---

## 局限性与未来方向

1. **评测基准稀缺**：目前仅有 PM-Bench 一个公开 agent PM 基准，结论的外部效度有待验证。
2. **未探索 fine-tuning**：PIS 为训练无关方案，fine-tune Form/Revise/Decide 接口可能进一步提升性能，但未在本文中展开。
3. **Operator 贡献未隔离**：Form/Revise/Filter/Decide 各自的 ablation 尚未系统分析。
4. **未来方向**：（i）对 Form–Decide 接口做 fine-tuning；（ii）设计更多 Prospective Memory 基准；（iii）开展 operator 级消融实验。

---

## 研究启发与可借鉴点

1. **"能力归因于架构而非尺度"的设计范式**：将 schema-constrained 任务外包给代码（Filter、$\kappa$ 为纯结构），仅把 open-ended 语义 grounding 留给模型，使 SLM 可匹敌甚至超越 LLM scaffold——这一 factorization 思路可迁移至其他受限 Agent 任务。
2. **观察（Observation）是 PM 的核心环**：PIS 中 channel judge 主动查询隐藏状态的机制，弥补了纯被动回忆的缺陷；这对任何涉及隐式环境状态的 Agent 任务均有借鉴价值。
3. **训练无关（training-free）的可复现性**：无需 LoRA 或蒸馏即可获得性能跃升，降低了部署门槛，适合资源受限场景的快速验证。
4. **小模型 × 类型化接口的协同优势**：Gemma-E2B 从 4.2% 到 66.2% 的跨跃证明，只要动作空间有 typed schema，SLM 即可胜任原本认为需 LLM 的任务；此结论可推广至其他结构化记忆/规划场景。
5. **错误可定位至算子边界**：Form/Revise/Filter/Decide 四步隔离使得性能瓶颈可逐一诊断，为后续 ablation 和 fine-tuning 提供清晰入口。

---

## 关键术语表

**Prospective Memory (PM)**：在正确未来线索出现时执行延迟意图的能力，区别于回顾性记忆（对过去知识的回忆）。

**Typed Intention Store**：将意图存储为类型化三元组 $I=(\varphi,\alpha,\sigma)$ 的外部数据结构，$\varphi$ 触发条件、$\alpha$ 动作、$\sigma$ 状态。

**Due Set**：某时刻应执行的候选动作集合，理想值为 $D_t^\star$，Agent 预测值为 $\hat{D}_t$，以 Set-F1 评估。

**Hidden Channel**：Agent 不可直接观测但影响触发条件的隐式状态源（portal/dashboard/sensor），需主动查询才能获取。

**Form–Revise–Filter–Decide Loop**：PIS 的单步四算子循环，Form 提取意图、Revise 修订意图、Filter 缩小候选、Decide 查询通道并输出待办集。

**Set-F1**：预测待办集与真值待办集之间的 micro F1，衡量多动作同时预测的准确性。

**Update Miss**：在 reschedule/cancel/override 更新敏感切片上的遗漏率，衡量 PM 的动态修正能力。

**Cross-Day Miss**：在前一天植入、次日触发的意图上的遗漏率，衡量跨时间窗口维持承诺的能力。

---

## 可复现要素

- **数据集**：PM-Bench（synthetic week），论文声明将在接受后 release 方法与代码（目前未明确开源状态）。
- **代码/权重**：论文未正式开源（声明 "We will release the method and code upon acceptance"）。
- **关键超参**：论文未详细列出，主要依赖预训练 backbone（DeepSeek-Chat、Gemma-E2B、Qwen3.5-4B、Qwen3-8B）的冻结推理。
- **评测协议**：PM-Bench 官方协议，主指标 Set-F1，辅助指标 update miss / cross-day miss / FA/step。
