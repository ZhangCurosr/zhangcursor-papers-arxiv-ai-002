---
title: "MutMem-V2-Cryptographically-Authorized-Mutation-in-Persisten"
source: https://arxiv.org/pdf/2609.01235v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 19:25:58"
field: "Agent 记忆与可审计协议"
keywords: ["persistent agent memory", "cryptographic authorization", "tamper-evident logging", "portable verification", "Merkle order", "PoisonedRAG"]
innovations: ["定义13+5r版本的跨对象谓词图与闭集失败词实现可移植验证", "双语言Node/Python独立验证器72/72判定一致与42/42生产conformance一致", "三种突变终端(authorized_transition/signed_noop/occurrence_observation)的完整枚举"]
benchmarks: ["LongMemEval", "LoCoMo", "PoisonedRAG adaptation"]
---

# 论文速读：MutMem V2: Cryptographically Authorized Mutation in Persistent Agent Memory — Portable Verification and Reproducible Evidence

## 一句话总结
MutMem V2 在 V1 基础上定义了**完整可移植的密码学验证协议**，使第三方能够从干净安装出发独立复核证据；但其实证结果（LongMemEval、LoCoMo、PoisonedRAG 等）仍标注为"历史 V1"，未进行 V2 重新运行。

## 研究问题与动机
1. **验证不可移植**：V1 虽有追加式记忆变异与签名溯源设计，但缺乏从干净安装到可核查证据的完整路径，审核者必须信任生产数据库或运行时。
2. **对象边界未显式化**：V1 的对象边界、字节编码、信任根、失败原因、结果基数与终端状态均未形成显式规范。
3. **证据与实证分离不足**：V1 未明确区分"协议级一致性证明"与"独立实证复现"，导致读者可能误读 V1 的基准结果为 V2 结果。
4. **审计闭环缺失**：缺乏"请求-授权-召回-突变-终端收据"的完整跨对象谓词链，难以做面向第三方的结构化审计。

## 核心贡献（创新点）
1. **版本化召回披露信封**：定义 `13 + 5r` 对象的确定性成员集合与精确字节编码，以域分离 SHA-256 承诺绑定，与 V1 实现解耦。
2. **跨对象谓词图**：涵盖外部信任锚、Actor/Housekeeper 身份epoch、撤销状态、有效授权、请求-收据配对、Merkle 顺序与终端收据校验，形成可机判定的判定网络。
3. **三种突变终端的完整枚举**：`authorized_transition`、`signed_noop`、`occurrence_observation`，并在协议层区分授权转换与观察性记录。
4. **双语言独立验证器**：独立 Node / Python 实现在 72 个结构与密码学向量上达成精确判定一致；另提供 42 个生产派生 conformance 用例，覆盖 28 类必需场景且全部一致。
5. **可再生出版证据架构**：所有公开表格从自哈希聚合体重建；提供确定性证据再生命令；明确声明"V2 不重新标定 V1 的实证结果"。

## 方法详解
- **对象承诺**：对每个对象体做 JCS 规范 JSON 编码后，采用 `H(D_o ‖ LP₄(k) ‖ LP₄(s) ‖ BE₄(|J(b)|) ‖ J(b))` 计算，禁止非有限数、超限嵌套与越界 body。
- **有序 Merkle 根**：按 RFC 6962 构建 leaf/node 域分离的 Merkle 树，防止节点编码被误认为叶节点。
- ** bundles 承诺**：`B = H(D_b ‖ LP₄(u) ‖ LP₄(c) ‖ f ‖ BE₈(r) ‖ R)`，其中外部信任指纹 `f` 必须从外部通道获得，避免循环信任。
- **五类跨对象谓词**：Authority / Admission / Decision composition / Per-result evidence / Terminal receipt；失败词汇是闭集，新增失败需走新版本。
- **突变体承诺**：`B_μ = H(D_μ ‖ J(M))`，其中 M 包含召回承诺、结果证据、收据、事件、签名 valence 与可选认知投影。
- **统计契约**：采用 Wilson 区间（z=1.9599…）、McNemar 精确检验、Holm 校正与 Cohen's κ；所有派生数值容忍度为 `10⁻¹²`。

## 实验与结果
- **验证向量**：72/72 结构/密码学向量上 Node 与 Python 实现完全一致；10/10 退出标准通过。
- **Conformance corpus**：42/42 生产派生用例一致，覆盖 28 个必需类。
- **Canary 显式标记穿越**：120 个终端单元中 60/60 标记被检测（100%，Wilson 下界 93.98%），0/60 干净单元误报（0%，Wilson 上界 6.02%），仅支持显式标记路径，非通用检测器。
- **Clean installer**：Node v26.8.1 一次干净安装通过首次启动、重启与调度器就绪；注册 8 条 Guide 记忆、0 条实验记忆。
- **历史 V1 基准**（Table 2，仅作为历史参考，非 V2 重跑）：
  - LongMemEval LLM judged：459/500（91.80%，CI 89.06–93.90%），生成器 GPT-5.4，裁判 GPT-5.6 Terra。
  - LoCoMo LLM judged：1472/1986（74.12%，CI 72.15–76.00%）。
  - LoCoMo token F1：58.20（bootstrap 区间因私有行不可用而省略）。
  - PoisonedRAG 适应版 N=100：induced ASR = 1/98（1.02%）；clean 2/100，attacked 3/100。
  - 突变延迟：median 4.865 ms，p95 5.674 ms（单次机器描述值）。
  - 盲审系统/裁判一致：193/200（96.5%，κ=0.9108），由系统作者执行。
- **PoisonedRAG ablation**（Table 3）：A1/A2/A3 将 retrieval@5 从 94/100 降至 0/100；clean accuracy 维持在 68–69%；attacked accuracy 从 40% 恢复至 65–71%。

## 相关工作脉络
1. **Tamper-evident logging**（Crosby & Wallach, 2009）：密码学承诺使历史篡改可检测，本文借鉴其思想并扩展到 agent 记忆场景。
2. **Certificate Transparency / RFC 6962**：定义本文有序 Merkle 构造，用于披露证据的确定性排序。
3. **EdDSA / JCS / 长度帧编码**：均为成熟原语，本文组合使用而非原创。
4. **LongMemEval / LoCoMo / PoisonedRAG**： motivating benchmarks；本文与其区别在于后者的任务聚焦检索质量，而非可移植授权与证据审计协议。
5. **MutMem V1**：引入保留式记忆变异与追加式结果证据；V2 与之的本质差异是“协议级可移植验证契约”而非新记忆引擎。

## 局限性与未来方向
1. **无 V2 实证重跑**：所有基准结果仍为历史 V1，V2 未重新测量。
2. **无独立实证复现**：验证器虽独立但实验未被第三方团队重跑。
3. **私有 V1 行不可用**：部分 bootstrap 区间与详细行无法独立重建。
4. **Canary 仅覆盖显式标记**：不能推广到任意 poison、prompt injection 或社会工程。
5. **PoisonedRAG 为适配版**：非上游全量语料，泛化未证。
6. **SABER 操作数缺失**：因私有 artifact 不在验证保管中而全部省略。
7. **安装覆盖受限**：仅完成 Node v26.8.1 单一平台一次安装资格；其他运行时无自动继承。
8. **外部信任锚不可或缺**：若证据与所有信任锚同时被替换，协议无法防御。
9. **arXiv ID 尚未分配**：论文为提交候选，归档 ID 待接收后补录。

## 研究启发与可借鉴点
1. **闭集失败词汇 + 版本化谓词**：通过将失败原因列表化、禁止未声明异常，极大增强跨实现判决一致性，可迁移到任何需要审计闭环的安全子系统。
2. **协议/实证分层声明**：明确区分"V2 协议结果"与"历史 V1 实证"，并用 claim map 逐条绑定 code/verifier/denominator/limitation，值得在可复现性论文中作为标准模板。
3. **自哈希聚合体驱动出版**：所有公开表格从单一确定性证据根重建，降低论文– artifact 漂移风险。
4. **双语言独立验证器同向判定**：以 72 个正/负向量验证跨语言 parity，比单纯报告准确率更能证明协议可移植性。
5. **与团队方向结合机会**：若团队研究 agent 记忆持久化/安全审计/RAG 抗投毒，可将本协议的"跨对象谓词图+闭集失败词+Merkle 顺序"作为可复用验证层，叠加团队现有检索模块。

## 关键术语表
- **Canonical JSON (JCS)**：RFC 8785 定义的确定性 JSON 序列化方案，消除空白与键序差异，保证不同实现产生相同字节。
- **Domain separation**：通过在哈希输入前缀不同域名字符串，防止同一 hash 函数下叶节点与内部节点被混淆。
- **Merkle order**：以 RFC 6962 方式将对象有序列表哈希为树根，任何增删换序均可被检测到。
- **Authorized transition / Signed noop / Occurrence observation**：三种互斥的突变终端，分别对应权重更新、签名保留不变、仅记录发生事实。
- **Cross-object predicate**：横跨多个对象的一致性约束，用于校验授权链、请求-收据配对与结果顺序。
- **Portable verification**：验证器仅消费公开证据对象，不依赖生产者运行时，实现跨平台裁决。
- **Clean installer qualification**：从干净安装到首次启动、重启与调度器就绪的一次资格性演示。
- **Claim-to-evidence disposition**：将论文每一量化/安全声明与代码、验证器、分母与局限性逐条绑定的映射表。

## 可复现要素
- **数据集**：LongMemEval、LoCoMo、Natural Questions/BEIR、PoisonedRAG 文本**不随包重分发**；仓库提供下载器与哈希。
- **代码**：GitHub 仓库 https://github.com/wallidsaydi-creator/HOM-AIMOS，AGPL-3.0-or-later；公共发布标签 v1.0.0，commit fcda26d1e05c729185c09dd11446db2f79c14a0d。
- **独立验证器**：Node（参考）与 Python（独立）两份实现，均随仓库提供。
- **关键命令**：
  - 离线验证：`npm run verify`
  - 确定性证据再生：`npm run evidence:regenerate`
  - 完整复现：`npm run reproduce -- --installed-instance canonical --agent-id "$AGENT_ID" --full --benchmark both --protocol canonical-blind-v1`
- **关键超参/环境**：Node v26.8.1；PostgreSQL 及扩展；并发与协议配置需记录；历史模型身份（GPT-5.4 生成、GPT-5.6 Terra 裁判、GPT-5.5/5.6 Terra 中毒道）保留。
- **公开承诺**：18 种对象 schema、39 条召回谓词向量、15 条突变向量、37 种召回失败码；所有表格可从自哈希聚合体重建。
