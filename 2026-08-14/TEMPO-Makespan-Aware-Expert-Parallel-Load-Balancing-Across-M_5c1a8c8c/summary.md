---
title: "TEMPO-Makespan-Aware-Expert-Parallel-Load-Balancing-Across-M"
source: https://arxiv.org/pdf/2608.13057v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 04:18:08"
field: "MoE推理负载均衡"
keywords: ["Mixture-of-Experts", "Expert Parallelism", "Load Balancing", "MoE Serving", "Makespan Optimization", "Inference Acceleration"]
innovations: ["两段式max-affine成本模型量化HBM权重流式传输与compute padding双regime", "固定费用makespan调度问题的NP-hardness证明与additive近似保证", "零边际内图成本的SGLang集成与拓扑感知两阶段source-replica分配"]
benchmarks: ["Qwen3-235B-FP8", "DeepSeek-V3", "Testbed A(8-GPU)", "Testbed B(2-node EP16)", "DeepGEMM fp8 grouped GEMM"]
---

# 论文速读：TEMPO-Makespan-Aware-Expert-Parallel-Load-Balancing-Across-M

## 一句话总结
本文针对MoE模型专家并行（EP）推理中的负载均衡问题，提出TEMPO——一种基于实测硬件成本模型的makespan感知调度器，通过同时建模显存瓶颈（权重流式传输）和计算瓶颈（M-tile填充），在混合 regime 场景下将端到端吞吐量提升最高15.5%，并精确预测收益窗口。

## 研究问题与动机
1. **MoE服务的同步瓶颈**：每个MoE层都以all-to-all通信为同步点，整个批次的处理时间等于最慢GPU的时间（makespan），现有方法的放层分配与批次级调度之间存在显著不平衡。
2. **代理指标的隐含假设不成立**：现有方法优化token计数（LPLB、UltraEP）或激活专家数（METRO），但实测表明每专家FFN时间并非单一线性函数——存在n*≈156-168 tokens的拐点。
3. **两阶段成本结构**：低于n*时HBM权重加载占主导（平坦区域，激活floor代价附着于副本而非token）；高于n*时grouped GEMM的128-token M-tile填充使分拆专家制造额外padding计算。
4. **混合regime是常态**：真实decode批量中92-100%的batch同时包含两种regime——热专家处于线性区、冷专家处于平坦区，单一代理在任何情况下都不是全局最优。

## 核心贡献（创新点）
1. **两段式max-affine成本模型**：首次用`t = max(a + bG, c + βN)`量化expert并行中隐藏的激活floor（b）和compute线性项（β），通过黑盒校准在10分钟内完成拟合，精度误差4-8%。
2. **固定费用makespan问题的形式化**：将每批次dispatch建模为NP-hard的固定费用调度问题，并在两个退化极限（b→0为token LP、β→0为半匹配）分别给出多项式解法，揭示问题难度来自regime交互。
3. **tempo fast启发式算法**：成本感知的贪心初始化+增广链激活重平衡（平坦区移动）+瓶颈局部搜索+token-LP/Round-robin集成切换，在1.9ms内求解（比MILP快550倍），全复制下有O OPT + max(b, βn_max)的加法近似保证。
4. **拓扑感知多节点扩展**：提出两阶段source-replica配对（同节点优先），在2-node EP16上将有向流量的负收益转为+4%至+7.2%提升。
5. **零边际内图成本集成**：Fused CUDA kernel合并概率调度与计数收集，solver out-of-process运行，消除GIL竞争，对SGLang无额外kernel/collective开销。

## 方法详解
**成本模型构建**：
- 公式(1)：`t_g = max(a + b·G_g, c + β·N_g)`，其中G_g为GPU g上激活的(expert, replica)对数，N_g为token总数
- 参数b（激活floor）≈1.74µs(Qwen3)至14.8µs(DSv3)，对应HBM带宽负载重专家权重的物理含义
- 公式(2)扩展（tile-aware）：`t = max(a + bG + b₂(T-G), c + βN)`，T=Σ⌈n_e/128⌉，b₂≈b/3捕捉后续tile阶梯

**问题形式化**：
- 公式(3)：min max_g max(a + bΣz_e,g, c + βΣx_e,g)，受限于x_e,g≥0且Σx_e,g=n_e
- Theorem 1证明NP-hardness：通过Balanced Partition归约，两GPU全复制情形下决策问题为NP-complete
- Lemma 1-2：b→0时退化为token LP（LPLB/UltraEP），β→0时退化为半匹配问题

**求解器设计**：
- 阶段1：按token降序整专家放置于边际成本最低的副本
- 阶段2：1-2步增广链激活重平衡（对应半匹配的截断版本）
- 阶段3：瓶颈局部搜索+三分搜索部分迁移（目标函数为分段线性单峰）
- 阶段4：集成(token-LP候选+Round-robin证书)，>1%切换容忍度避免过度切换

**通信与拓扑扩展**：
- 公式(4)三参数模型：`t_g = max(a + bG_g, c + βN_g, c₂ + γN_g)`
- 两阶段拓扑感知：先求解GPU份额，再同节点优先配对source-replica

## 实验与结果
**数据集与测试床**：
- Testbed A：8-GPU，DSv3/Qwen3-30B expert shape，fp8 masked DeepGEMM
- Testbed B：2×8节点，RoCE 8×400G，DeepSeek-V3/Qwen3-235B-FP8

**基线对比**：static、EPLB-even、token-LP(LPLB)、METRO、LLEP-R、MoonEP

**核心结果**：
| 场景 | 提升幅度 |
|------|----------|
| 模拟相位图 | 相比最优固定策略win up to 15.5%(DSv3)/11.8%(DSv2) |
| Qwen3-235B End-to-end | 长上下文+decode流量吞吐+4-6%，p99 TPOT -15.6% |
| DSv3 on Testbed B | 无增益（位于win region外，通信主导）|
| 2-node EP16 Qwen3 | 拓扑感知split后+7.2%吞吐 |
| vs SGLang shipped LPLB | 1.4-1.7×吞吐(含架构优势) |

**关键观察**：
- 相同batch的不同代理dispatch在模拟block time上相差1.4-1.6×(p95达1.7×)
- 92-100%的decode batch同时包含≥20%平坦区expert和≥20%线性区token
- TEMPO在每个evaluated grid point保持在最佳固定基线±1%内

## 相关工作脉络
1. **Token-proxies (EPLB/LPLB/UltraEP/FlexMoE)**：假设时间∝token，忽略激活floor，本文证明在memory-bound regime下token balancing会fragment cold experts并支付隐藏代价。
2. **Activation-proxies (METRO)**：假设时间∝expert数，仅适用于纯平坦区；本文在compute-bound regime证明其损失高达2.8×。
3. **Weight-moving balancers (MoonEP)**：通过在线prefetch冗余expert实现完美token平衡，但fetch代价(283-502µs)超过MoE层预算(95-107µs)，仅在prefill极端skew下有效。
4. **Latency-aware方法 (ViBE)**：按设备线性速率平衡，但未建模regime非线性，本文的成本模型补足了这一缺陷。
5. **理论工作 (MoE-Serving)**：证明GPU-quota粒度下的NP-hardness，本文补充per-batch dispatch粒度的复杂度分析，两者不可互相推导。

## 局限性与未来方向
1. **模型空间为主的数据**：头部数字来自校准模拟器，中间区域transfer error 2-6pp，中程验证依赖wall-clock微基准。
2. **测试床规模限制**：DSv3完整规模(58 MoE层, EP32+)仍为外推；更深层次可能需要更精细的traffic term。
3. **静态校准需求**：kernel/driver更新需重新运行10分钟校准流程。
4. **模型的底层假设**：(G,N)模型无法表达uniform per-slot load偏好（第三维度），该维度决定近ties。
5. **受限replica集的近似保证缺失**：Theorem 2的加法保证仅在全复制下成立，部分replica情形为open problem。

## 研究启发与可借鉴点
1. **黑盒成本校准范式**：脱离理论推导，直接在部署kernel上 sweep (G,N)网格拟合成本表，10分钟即可捕获真实硬件行为——适用于任何底层kernel稳定的系统。
2. **phase diagram驱动的方法论**：不追求universal win，而是绘制"何时用哪种策略"的地图，并通过两个边界模型（inside/outside win region）证伪预测——提升工程论文的说服力。
3. **degenerate limit分解复杂度**：证明NP-hard问题在两个极端参数下退化为多项式，为设计混合启发式提供理论锚点——可迁移至其他混合约束调度问题。
4. **architecture isolation技巧**：通过将token-LP移植到相同worker中比较，分离objective贡献与架构贡献（Section 5.12），避免将工程收益误归因于算法创新。
5. **topology-aware source splitting作为独立模块**：先求解负载再分配源-副本配对的两阶段设计，可解耦optimize和locality concerns，便于模块化复用。

## 关键术语表
**Expert Parallelism (EP)**：将MoE模型的各expert sharding到多GPU上，每层执行dispatch all-to-all → grouped GEMM → combine all-to-all的并行模式。
**Makespan**：多GPU同步场景下，以时间最长的GPU为基准的latency，MoE block的cost即为max_g(per-GPU time)。
**Activation Floor (b)**：HBM流式传输一个expert权重的成本(µs)，附着于activated replicas而非tokens，是memory-bound regime的主导项。
**M-tile Padding**：grouped GEMM将tokens向上取整到128-token tile，fragment一个expert跨k个replica会制造k次独立padding。
**Phase Diagram**：以batch size、skew、replication为轴的收益地图，标记不同区域的最优固定策略。
**Staleness**：solver使用上一window的计数而非当前计数，导致5-7pp的收益损失。
**Drift**：placement随routing变化而过时，实验通过16-window窗口模拟。
**Additive Approximation**：round-robin placement的makespan不超过OPT + max(b, βn_max)的理论界。

## 可复现要素
- **数据集**：公开路由trace(Qwen3-30B, DSv2-Lite × wikitext/GSM8K)，Testbed A/B未公开
- **代码**：论文未提及开源，但提供SGLang patch(D篇7个patch points)
- **权重**：未公开
- **关键超参**：n*≈156-168(tokens/expert)，β∈[0.0108, 0.0945](µs/token)，b∈[1.74, 14.78](µs/expert)，b₂≈b/3，切换容忍度τ=1%，solver refresh周期200ms
- **校准时间**：~10分钟/(kernel, dtype, hardware)组合
