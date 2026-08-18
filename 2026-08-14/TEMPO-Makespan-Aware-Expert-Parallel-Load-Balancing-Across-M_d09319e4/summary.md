---
title: "TEMPO-Makespan-Aware-Expert-Parallel-Load-Balancing-Across-M"
source: https://arxiv.org/pdf/2608.13057v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 04:18:48"
---

# 论文速读：TEMPO-Makespan-Aware-Expert-Parallel-Load-Balancing-Across-M

## 一句话总结
TEMPO是一种面向MoE专家并行（EP）推理的Makespan感知调度器，通过约10分钟黑盒校准揭示专家FFN成本在内存受限（flat）与计算受限（linear）双区间的非线性特性，将per-batch调度形式化为固定费用makespan问题，并在SGLang中以毫秒级进程外求解器落地；其核心主张并非“永远更优”，而是构建可预测收益边界的相图，在真实旗舰模型（Qwen3-235B）上稳定捕获吞吐与尾延迟增益，同时在通信主导或新鲜placement场景下诚实返回零收益。

## 研究问题与动机
- **代理指标的隐性线性假设在实测中崩溃**：现有生产调度器（EPLB/LPLB/UltraEP优化token计数，METRO优化激活专家数）分别假设时间∝tokens 或 时间∝experts，但DeepGEMM fp8 grouped GEMM实测显示：单expert耗时在n*≈156–168 tokens/expert存在拐点，拐点前为平坦段（HBM加载权重主导），拐点后为线性段（compute主导）。
- **双机制在单batch内共存**：真实decode流量中92–100%的batch同时包含冷expert（flat regime）与热expert（linear regime），单一代理策略会在对方的主场产生30–70%的系统性定价偏差；同batch下四种代理调度在建模block time上相差1.4–1.6×（p95 1.7×）。
- **分裂expert的隐藏成本未被捕捉**：token平衡会碎片化冷expert，重复触发激活地板b；而分组GEMM以128为M-tile对齐，分裂一次即向上取整一次，制造额外的padded compute。
- **缺乏统一的目标函数与可迁移的预测框架**：现有工作未在同一模型下刻画内存流与计算填充的交互，也缺少能指导部署“何时用何种策略”的可验证预测工具。

## 核心贡献（创新点）
1. **双区间max-affine专家成本模型**：用`t = max(a + bG, c + βN)`统一定价HBM权重流开销（b）与M-tile填充斜率（β），并扩展引入单层台阶参数b₂建模完整阶梯；与现有工作的本质区别在于首次将物理可解释的激活地板与tile对齐惩罚纳入单一可校准目标，而非依赖单一线性代理。
2. **固定费用makespan调度的形式化与复杂度刻画**：证明该问题在双GPU全复制下已是NP-hard（归约自Balanced Partition），但在纯内存（β→0）或纯计算（b→0）退化极限下分别退化为多项式时间的semi-matching与token LP；理论揭示了“代理各自在家成立、混合区间才是困难根源”的本质。
3. **tempo fast四阶段启发式求解器**：成本感知贪心种子 + 增广链激活重平衡（针对flat regime） + 瓶颈局部搜索与部分迁移（针对linear regime） + 带1%切换容差的Ensemble；在《=2ms内逼近10s MILP，且在全复制下继承O(OPT+max(b,βn_max)))加性保证。
4. **零关键路径开销的SGLang集成与相图方法论**：将概率路由与count collection融合为单个CUDA kernel，求解器跑在独立numpy进程避免GIL竞争；同时输出覆盖batch size×skew×replication×shape的相图，预测每种固定策略的胜负边界，并在Qwen3-235B与DeepSeek-V3上 bracket win region 完成可证伪验证。

## 方法详解
- **成本模型校准（Black-box calibration）**：在(G, N)网格上运行部署kernel（DeepGEMM fp8 masked grouped GEMM），CUDA-graph计时、权重轮换防L2复用、固定expected m冻结JIT tuner；对每对(G,N)拟合`t_g = max(a + bG_g, c + βN_g)`，拟合误差4–8%。Testbed B上进一步引入tile-aware项`t = max(a + bG + b₂(T−G), c + βN)`，T=∑⌈nₑ/128⌉，b₂≈b/3，将multi-tile区域误差从~10%降至2.0%。
- **问题形式化**：给定expert token数nₑ、副本集R(e)、分配xₑ,₉与激活指示zₑ,₉，目标为`min max_g max(a + bΣzₑ,₉, c + βΣxₑ,₉)`。Theorem 1通过Balanced Partition归约证NP-hard；Lemma 1/2分别给出β→0与b→0极限下的多项式对应。
- **tempo fast求解器**：
  1. **Cost-aware seeding**：按token降序将整expert放至边际成本最低的副本，避免在flat regime碎片化冷expert。
  2. **Augmenting-chain activation rebalancing**：在flat regime瓶颈为max G_g，直接单步移动会抬升目的地；采用1/2步截断的半匹配增广链，多项式时间内降低最大激活数。
  3. **Bottleneck local search with partial migrations**：在linear regime对瓶颈GPU进行三元搜索部分拆分，目标在单变量下为分段线性单峰，迭代≤300次。
  4. **Ensemble with switching tolerance (τ=1%)**：同时对token-LP、round-robin证书A₃（Theorem 2）与局部搜索解在统一模型下打分，仅当某候选比当前最优好>1%才切换；避免模型近tie被硬件未建模偏好（如uniform per-slot load）驱动。
- **通信与拓扑感知扩展**：加入第三项`c₂ + γN_g`拟合all-to-all流量，形成三项max-affine模型；多节点部署采用两阶段split：先用原求解器确定per-GPU份额xₑ,₉，再对每个expert按same-node-first规则配对源到副本，保证compute makespan不变的同时最小化跨节点流量。
- **SGLang集成架构**：in-graph仅保留一个融合kernel（probabilistic dispatch + atomic cumulative count）；refresher thread在side stream快照计数器，gloo group聚合后递交独立numpy进程求解；表更新采用pinned staging + race-safe发布，单行撕裂时fallback至均匀分布，天然避免除零或路由至无该expert的副本。

## 实验与结果
- **数据集/模型/硬件**：Qwen3-30B/235B-FP8、DeepSeek-V2-Lite、DeepSeek-V3-0324；工作负载含ShareGPT、Wikitext、GSM8K、OASST1、GovReport；硬件为8-GPU Testbed A (NVLink) 与 2×8 Testbed B (RoCE 8×400G/node)。
- **评估基线**：EPLB-even（uniform）、token-LP（LPLB）、METRO（activation balancing）、static placement、MoonEP（weight-moving）、LLEP-R。
- **相图与仿真**：TEMPO在所有网格点保持在最佳固定策略±1%内；混合区最高提升15.5%（DSv3, EP32–64外推）；METRO在compute regime深层代价高达2.8×，token-LP在memory regime仅5–17%，不对称显著。
- **Testbed A 8-GPU EP8微基准**：B=32时TEMPO比EPLB-even优11–14%、比token-LP优7%；B=2048时METRO偏离最佳7–11%；TEMPO全跨度内保持在±5%（≈run noise）。
- **Testbed B 端到端 serving**：
  - Qwen3-235B（inside win region）：GovReport吞吐+
