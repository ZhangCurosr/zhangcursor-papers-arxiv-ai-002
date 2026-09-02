---
title: "Relational-Core-Graph-Analytics"
source: https://arxiv.org/pdf/2609.01525v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 19:29:17"
field: "图数据库与查询优化"
keywords: ["graph analytics", "Cypher", "SQL translation", "columnar database", "zero-ETL", "LDBC SNB", "lakehouse"]
innovations: ["模式原生映射：识别关系模式中五种连通表达并翻译为六类 JOIN 策略，无需重编码为节点/边表", "单路径翻译架构：集中分析所有 schema 决策并通过枚举穷举匹配保证编译期覆盖", "零 ETL 零集群：将图查询推入已有 ClickHouse/Databricks 或 chdb 进程执行，无需独立集群"]
benchmarks: ["LDBC SNB (SF1, SF10)", "OnTime Flights", "Apache AGE 重编码税微基准"]
---

# 论文速读：Relational-Core-Graph-Analytics

## 一句话总结
本文提出 ClickGraph/DeltaGraph 系统，将图查询语言 Cypher 直接翻译为关系型 SQL，在 ClickHouse、Databricks 等列式数据库上以零 ETL 方式查询现有关系架构，证明列式关系引擎在分析型图查询上可匹敌甚至超越原生图引擎，且能无限扩展至内存图引擎无法承载的规模。

## 研究问题与动机
- 企业数据天然存在于关系型仓库/湖仓中（客户、订单、事件、日志），而 GraphRAG 等新兴应用需要在此类数据上进行多跳图遍历，但传统做法需将数据拷贝至独立图引擎，产生 ETL 管道、同步成本与第二集群运维负担。
- 节点/边属性图并非对连通数据的更忠实建模，而是对关系表中已显式存在的关系的重新编码；重建这些关系在查询时执行仅为纯开销。
- 原生图引擎（如 Neo4j）采用 tuple-at-a-time 指针追逐循环，无法向量化执行，难以利用数十年来列式存储、向量执行、代价优化等成熟技术。
- SQL/PGQ 标准已被 ISO 采纳并进入 Oracle 23ai，证明"图查询应内建于关系引擎"已被主流认可，但现有实现仍依赖 PROPERTY GRAPH 视图声明，而非直接在现有列式仓库的原始模式上运行。

## 核心贡献（创新点）
- **模式原生映射（Schema-native mapping）**：识别五种关系模式中的连通数据表达（标准、外键边、反规范、多态、复合 ID），并将 Cypher 直接翻译为六类 JOIN 策略，无需将数据重编码为节点/边表；与 Apache AGE/AgensGraph 等 Tier 2 系统仅在相同 SQL 引擎上增加一层重编码相比，避免了"隐式邻接"重建开销。
- **单路径翻译架构**：将所有 schema 决策在一个分析点集中计算（PatternSchemaContext::analyze），下游通过枚举穷举匹配消费，新增变体必须在编译期处理所有消费者，配合 ratchet 测试强制杜绝原始 flag 分支；相较此前分散在 4,800+ 行代码中的 ad-hoc 检查，消除了"修复一类模式后另一类回归"的 ping-pong bug。
- **零 ETL + 零集群架构**：与 PuppyGraph 同属 Tier 3 零 ETL，但 PuppyGraph 需独立部署专用集群，而 ClickGraph 将图查询直接推入组织已有的 ClickHouse/Databricks 集群或 chdb 进程内执行，无需新集群。
- **Bolt 协议兼容与 LLM 辅助发现**：完整实现 Neo4j Bolt v5.8 协议、db.labels/db.relationshipTypes 等 schema 自省过程及 WebSocket 传输，使 Neo4j Browser、AWS graph-notebook 等图形客户端可直接连接；并通过 cg schema discover + LLM 生成映射，降低模式发现门槛。
- **开源 SQL 翻译形式带来的开放性**：生成的查询是普通 SQL，可被重写、优化；引擎层可接入最坏情况最优连接、因子化处理等图专用优化；递归 CTE 与扁平 JOIN 链的可观测差异使得性能瓶颈可诊断、可修复。

## 方法详解
- **四层运行模式**：Server 模式（暴露 HTTP + Bolt v5.8 对接远程 ClickHouse）、Embedded 模式（通过 chdb 在进程内直接查询 Parquet/S3/Iceberg/Delta Lake）、Remote 模式（本地翻译、远程执行）、SQL-only 模式（仅翻译不执行，用于调试）。
- **六阶段查询管线**：Parse → Plan → Optimize → Render → Generate SQL → Execute；入口为 Cypher，出口为 ClickHouse 或 Databricks SQL。
- **五种模式 × 六种 JOIN 策略**：
  - Standard：三表 JOIN（node–edge–node）。
  - FK-edge：两表 JOIN 或自连接，锚点反转。
  - Denormalized：零 JOIN 单表扫描，多跳变为 edge-to-edge 自相关。
  - Polymorphic：加入 discriminator 谓词 + UNION 展开处理通配端点。
  - Composite-id：标识符转为 SQL 元组，等值 JOIN 展开为 N 列 key-zip。
  - EdgeToEdge / CoupledSameRow：处理反规范多跳及同行耦合边。
- **递归 CTE vs 扁平 JOIN 策略**：精确长度路径（*N）自动展开为扁平内连接链；有界范围（*min..max）目前仍生成 WITH RECURSIVE，无界路径（*1..）保留递归。图 3 显示，扁平 JOIN 可由优化器重排、推过滤器、跨分片并行；递归 CTE 为顺序不动点。BI3 查询因无界递归回复路径在 SF10 上从 1.8s 膨胀至 241s（~130×），暴露递归 CTE 边界。
- **优化层**：投影与过滤下推、锚点选择、统计信息驱动规划；未来可接入 GRainDB/DuckPGQ 风格的最坏情况最优连接与因子化处理。

## 实验与结果
- **LDBC SNB 基准（自身测量）**：单机 32 核 121GB 内存，ClickHouse 26.7，SF1（9.9K 人/165K 边/1.05M 帖）上 41 个官方查询中 26 个通过，中位延迟 Interactive Short 23ms、Interactive Complex 130ms、BI 367ms。SF10（67K 人/1.75M 边/7.84M 帖/17.4M 评论）覆盖率仍为 26/41，26 个通过查询中位数仅增长 2.5×，点查 S1/S4 基本持平；唯一异常 BI3（无界递归回复路径）从 1.8s 增至 241s。
- **PuppyGraph 发布基准的外部证据**：OnTime 航班数据集（12.28M 边），Neo4j 社区版 vs 列式引擎对比——ClickGraph 在 19.57M 航班的单节点 5 轮中位数：q1=91ms、q2=85ms、q3=21ms、q4=31ms；Neo4j 分别为 ~41s/~42s/~345s/~515s，速度提升约 370×、380×、4,300×、5,000×（两个数量级至四个数量级）。
- **重编码税隔离微基准（Claim B）**：PostgreSQL 18.6 单机，100K 顶点/2M 边的相同数据，一次以 native FK-join SQL 执行，一次以 Apache AGE 节点/边表执行，结果完全一致；2-hop 种子节点 1.9ms vs 13.0ms（6.8×）、3-hop 10.3ms vs 24.3ms（2.4×）、全图 2-hop 560.6ms vs 2,211.7ms（3.9×），在 0.56s 的查询上增加 1.6s 延迟是可感知的。
- **正确性**：通过全部 402 个 openCypher TCK 读取场景，DeltaGraph 与 ClickHouse 间 22 项 LDBC 结果一致性校验均通过。

## 相关工作脉络
- **Grail (CIDR 2015)**：最早提出图查询编译为 SQL 的系统，在 PageRank、SSSP、WCC 上与 GraphLab/Giraph 竞争并在超内存时表现更稳健；但 Grail 仍需将数据导入 vertex/edge 表，本文扩展至仓库规模与模式原生查询。
- **GRainDB (CIDR 2022)**：扩展 DuckDB 加入基于 RID 的指针连接与邻接列表索引；其"SQL 引擎内部可用于图负载优化"的思路被继承，但 GRainDB 仍是关系上的节点/边叠加层，本文通过零 ETL 消除该层。
- **DuckPGQ (CIDR 2023/VLDB)**：SQL/PGQ 的 DuckDB 参考实现，采用内存 CSR 加速路径查找并主张最坏情况最优连接；本文与其同属 Tier 3 但区别在于直接在现有列式仓库的原始模式上翻译，而非 PROPERTY GRAPH 视图声明。
- **Apache AGE / AgensGraph**：Tier 2 关系核心图存储，基于 PostgreSQL，要求将数据加载到自有标签式节点/边表并通过 JOIN 重建关系；本文在相同 SQL 引擎层面证明了 native FK-join SQL 比 AGE 快 2.4–6.8×，孤立出"重编码税"。
- **PuppyGraph**：与本文同为 Tier 3 零 ETL，但 PuppyGraph 需独立集群；Benchmark 直接对比显示 ClickGraph 在单节点达到 PuppyGraph 满缓存亚秒级性能，无第二集群成本。
- **Kùzu / LadybugDB**：Tier 1 存储但采用列式/向量化架构，与 ClickGraph 共享"OLAP 图查询需向量化"理念，但 Kùzu 仍是独立图数据库；本文定位为其上游互补而非直接替代。

## 局限性与未来方向
- **仅只读 OLAP**：不涉及写操作与 OLTP 事务性遍历；低延迟单记录路径查找仍是原生引擎优势区。
- **单节点评估**：未验证分布式扩展；Grail 2015 年遗留的分布式执行问题待解。
- **无直接 TigerGraph / 原生图引擎对比**：Neo4j 对比依赖 PuppyGraph 发布数据；未来需同机三向对照。
- **递归 CTE 边界**：当前仅精确长度路径（*N）展开为扁平 JOIN；有界范围（*min..max）仍用递归 CTE，导致 BI3 在 SF10 上出现 ~130× 膨胀；未来方向包括范围展开为 UNION ALL 固定链。
- **递归端点跨 WITH/UNWIND 屏障的重匹配**：15 个失败查询中 5 个受此结构限制；需在翻译层扩展递归 CTE 端点解析能力。
- **chdb 嵌入模式能力有限**：当前不支持将临时结果表提升为新图模式；未来可构建可组合的 per-step schema，支持多步图分析与 Agent 工具链调用。

## 研究启发与可借鉴点
- **模式发现自动化**：cg schema discover + LLM 生成映射的思路可迁移至任何"SQL-on-existing-schema"的集成场景，显著降低零 ETL 部署门槛。
- **递归 CTE vs 扁平 JOIN 的可观测性**：通过对比两者在不同数据规模下的性能分化（如本文 BI3 的 130× 膨胀），可为图查询优化提供明确的诊断指标与改进优先级。
- **单路径翻译 + 枚举穷举匹配的工程范式**：PatternSchemaContext::analyze 将所有 schema 决策集中在一个分析点，配合 ratchet 测试保证编译期覆盖，可作为复杂查询翻译器的通用设计模板。
- **重编码税的隔离测量方法**：在相同引擎、相同数据、仅改变表示形式的条件下比较性能（本文 §6.1），为评估"模式层成本"提供了可复现的基准设计。
- **与 Vector/Embedding 场景结合**：chdb 嵌入模式支持临时结果表物化，可将图遍历产物直接与向量检索、RAG pipeline 衔接，构成面向 GraphRAG 的端到端分析工作流。

## 关键术语表
- **Tier 1/2/3**：按图数据存储位置划分的三层架构——Tier 1 为原生图存储（邻接列表），Tier 2 为关系核心图存储（自有节点/边表），Tier 3 为零 ETL 原生模式翻译（无拷贝）。
- **Property Graph（属性图）**：以节点、边及其属性为核心的图数据模型，传统图引擎的标准存储格式。
- **Re-encoding Tax（重编码税）**：将关系表中已显式存在的关系（如外键、联合行）重编码为通用节点/边表所带来的查询时重建开销。
- **Zero-ETL**：不将数据从源仓库/湖仓复制到独立图存储，直接查询原位数据。
- **Worst-case-optimal Join（最坏情况最优连接）**：在 join 结果大小的最坏情况下仍保持最优时间复杂度的连接算法，适用于图遍历场景。
- **Factorized Processing（因子化处理）**：将关系查询结果以有向无环图形式共享公共子表达式，避免重复物化，用于压缩多源路径查找输出。
- **Recursive CTE（递归公用表表达式）**：SQL 中用于表达可变长度路径的标准机制，以顺序不动点方式执行，可扩展性弱于扁平 JOIN。
- **Bolt Protocol v5.8**：Neo4j 定义的二进制网络协议，用于客户端与图数据库之间传输 Cypher 语句与结果记录。

## 可复现要素
- **数据集**：LDBC SNB（SF1、SF10）、OnTime 航班（2020–2022，12.28M 边/19.57M 航班）、PostgreSQL 单机微基准（100K 顶点/2M 边社交图）——均为公开或可复现来源。
- **代码**：ClickGraph / DeltaGraph 开源，仓库 https://github.com/genezhang/clickgraph；文档 https://genezhang.github.io/clickgraph；Docker 镜像 genezhang/clickgraph:latest。
- **权重**：不适用（纯翻译引擎，无模型权重）。
- **关键超参**：LDBC 评测使用 5 轮迭代取中位值；ClickHouse 26.7、PostgreSQL 18.6 版本；机器配置：LDBC 评测 32 核 121GB RAM，OnTime 评测 m6a.2xlarge（8 vCPU/32GB）。
