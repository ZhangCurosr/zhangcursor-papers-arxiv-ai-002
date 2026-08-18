---
title: "Quipu-A-Governed-Bitemporal-Knowledge-Graph-Store"
source: https://arxiv.org/pdf/2608.16813v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 04:30:57"
---

# 论文速读：Quipu-A-Governed-Bitemporal-Knowledge-Graph-Store

## 一句话总结
论文提出 Quipu，一种可嵌入的知识图谱存储系统，通过将“门禁写入、全要素双时态、分区信任非扩延组合、治理内生化”四项传统默认机制全部反转，解决了 AI Agent 高速自主写库场景下的数据污染、信任洗白与审计不可判定问题，并给出确定性基准 Census 与内容级退化测试证明其严格性与可审计性。

## 研究问题与动机
1. **写入库债务（D1）**：传统存储默认“先接受后清洗”，下游读者在审查窗口期即可消费未验证事实；Agent 写入速率远超人工审查能力，导致脏数据指数累积。
2. **单/无时间轴（D2）**：即使支持双时态的存储也仅将数据层时间化，信任标注、策略决策与验证规则仍为 latest-only，无法回答“T 时刻什么被信任/被允许”。
3. **扁平信任（D3）**：命名图组合采用静默联合（silent union），低信任或隔离图的事实加入后继承高信任图的名誉，信任无法被机械降级。
4. **治理外置（D4）**：策略生活在 dashboard、prompt 或 middleware 中，策略规范 Σ 与存储状态独立漂移，审计退化为日志考古且不可机械判定。

## 核心贡献（创新点）
1. **四默认机制的统一反转实现**：Quipu 在单一可嵌入存储中原生支持写入门禁、全栈双时态、分区信任格与非扩延组合；与已有工作本质区别：前作仅在某一方面有探索，本文证明四项机制相互强化，单独实现均无法支撑 Agent 规模治理。
2. **GS1–GS6 存储治理契约**：提炼门禁写入、判决持久化、分区权威、非扩延组合、在库可判定审计、按时重放六项原则；与已有工作本质区别：将 SARC 等框架的“治理即架构”思想从 Agent 循环下沉至存储原语层，使 Σ 与 trace 成为可查询的事实。
3. **Census 与内容级退化基准**：提出确定性多写入手绘生命周期与 DEMM-Bench 内容退化评估；与已有工作本质区别：全程无需 LLM judge，字节级可重复，并将证据充分性度量从“容器存在”转向“内容可重构”，直接量化 overclaim 风险。

## 方法详解
- **双时态 EAVT 日志与三值操作**：底层追加式事实日志形如 `(e, a, v, g, tx, valid_from, valid_to, op)`，op 为三值：`assert`（断言）、`retract`（逻辑关闭 valid interval，保留历史）、`tombstone`（标记组合视图中的缺席而不修改底层，使分层叠加语义严谨）。
- **命名图、Overlay 与元数据分区**：`committed` 图自根；`overlay` 图在创建时单次绑定到 committed parent，重新绑定报错（绑定不可伪造）。数据集为图的任意子集，组合遵循 semilattice 而非 silent union。
- **标签格（Label Lattice）与组合不变式**：每个分区携带 freshness/trust/durability/policy 四轴标签，存于预留 meta-graph。组合满足机器可检查的同态不变式：`label(A∪B) = label(A)⊓label(B)`；freshness 与 trust 按 meet 折叠，obligation 按 join 折叠。结果为单位对 `(fold, coverage)`，coverage ∈ {Empty, None, Partial, Full}。未声明不隐式升权，跨链信任比较报错而非静默整数比较，过期即视为缺席而非未知。
- **门禁写入（GS1）**：策略以图内实体形式定义（target type、SPARQL ask claim、boundary、effect）。写入时暂存事实进入 open savepoint，claim 针对 **pending post-state** 求值；未触碰受管 target type 的写入零策略评估开销。区分 pre-state 会漏检“孤立合法、组合违规”的写入。
- **判决持久化与签名（GS2）**：每次门禁决策（allow/deny/unknown）作为带 ed25519 签名的时间索引事实异步 flush，签名将 actor 与 principal chain 封入 evidence hash。拒绝的 write delta 被回滚，但 verdict 记录保留，确保审计对象不被清洗。
- **升级与权限相交（GS3）**：`require-approval` 拒绝生成 decision request 事实存活；人类响应绑定同一 evidence hash 后放行。权限沿 principal chain 取交集，空交集显式拒绝；重标图需 meta-graph 授权。
- **在库可判定审计（GS5）**：Σ、trace T、verdicts 均以事实存于被治理图中。审计器以 O(|T|·|C|) 四遍扫描（coverage、class-placement、outcome consistency、attribution）决定 `T |= Σ`，不访问模型或 prompt。violation 与 incompleteness 严格区分，避免混合隐瞒。
- **按时重放（GS6）**：规则版本通过 bitemporal registry 管理，加载命名集时关闭旧版本而非覆盖，移除为 close 而非 delete；`as-of-T` 查询可同时拉取当时的数据、标签与策略。

## 实验与结果
- **数据集与基准**：Census（确定性多写入手绘生命周期，seeded SplitMix64，无 LLM judge）；Yupana 真实 writer trace（5 条 Pre-Action Gate 决策回放）；DEMM-Bench（8 条件 × 8 属性族 = 64 案例，512 属性级判定）。
- **基线**：同脚本关闭门禁的 ungated 控制臂；SARC reference checker；DEMM-Bench 的 container-presence baselines（trace/ledger/schema presence）。
- **主要结果**：
  - RQ1：gated 臂内零策略命中写入中位数 1.3 ms，合规治理写入 2.7 ms；control 臂仅 0.7 ms（GS3 授权相交开销不可避免）。
  - RQ2：种植缺陷 6 个，gated 终态 0/6，ungated 终态 6/6。
  - RQ3：
