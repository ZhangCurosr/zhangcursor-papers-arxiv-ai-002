# Reducing Batch Makespan for LLM Agent Workloads through Inference–Environment Coordination

Wei Feng, Meiling Chen, Boxiao Chen, Lu Liu, Bin Zi, Libin Ye, Xingyu Lu<sup>∗</sup>, Lei Xu, Jun Zhou

Ling Team, Ant Group

<sup>∗</sup>Corresponding author

Efficient batch execution of agent tasks is critical to LLM training and development workflows, including benchmark evaluation, synthetic-data generation, and RL rollouts. Since these tasks alternate between LLM inference and environment interaction, we present coordinated optimizations to reduce batch makespan, the time required to complete all tasks in a batch. For hybrid LLMs, a workload-based resource model guides GPU-memory allocation between key-value (KV) cache and recurrent state and provides a reference sample concurrency to exploit parallelism while limiting recomputation. Trajectory-aware routing and scheduling promote cache reuse while balancing worker load. Under constrained environment capacity, we decouple sample admission from environment execution and introduce Chain-Advancing Action Scheduling (CAAS), which prioritizes long chains using observed progress to shorten the batch tail. Controlled replay combines actual inference with recorded environment responses and durations for ablations of individual optimizations. Live evaluations with Ling-3.0-Flash and Ling-3.1-Flash on the BFCL, SWE-bench Multilingual, WideSearch, and MCP-Atlas benchmarks demonstrate makespan reductions of 10.4%–65.0%, while evaluation scores remain aligned. Furthermore, these methods have been successfully applied to training and evaluation workflows for the Ling-3.x model series, accelerating the model-development cycle.

Date: November 11, 2026 Correspondence: sing.lxy@antgroup.com

![](images/3104a020536c9225946dbb6f093b43afc2175f9cb4b00f740198dcbf104bbd54.jpg)

## 1 Introduction

The training and development of agentic LLMs rely on workflows such as benchmark evaluation, synthetic-data generation, and RL rollouts. Repeated across model-development iterations, these workflows often execute agent tasks in batches, with downstream stages awaiting the entire batch. Batch completion time therefore affects development efficiency, motivating our focus on reducing batch makespan, the time required to complete all tasks in a batch, rather than the request-level latency typically optimized in conventional serving, such as time to first token (TTFT) and time per output token (TPOT) (Kwon et al., 2023).

We refer to an agentic task instance as a sample and its complete execution as a trajectory. Figure 1 illustrates how a batch of such trajectories is executed, each alternating between LLM inference and environment interaction. Tasks are admitted under a sample-concurrency limit, and their inference requests are routed to workers and ordered by local schedulers. Actions produced by the model are queued as environment jobs and dispatched under an execution limit. Model responses and environment results are appended to the task’s context before the next inference turn. A final response or terminal environment action completes the task and releases its admission slot. The batch finishes when all tasks complete.

![](images/7d8d88153397d1e129016729d611bd1ab11e1ab32f3265d1f6b215d8f4146afe.jpg)  
Figure 1 Illustration of batched execution of multi-turn LLM agent tasks. Active tasks are those admitted but not yet completed, including tasks waiting for inference or environment interaction. Each maintains its own context while sharing inference and environment-execution resources. Purple labels indicate where our methods are applied. Capacity separation allows the sample-admission limit $C _ { S }$ and environment-execution limit $C _ { E }$ to be configured independently. Our workload-aware capacity model guides the GPU memory allocation between KV cache and recurrent state at deployment; the allocation remains unchanged during batch execution.

Successive requests within a trajectory share growing input prefixes, creating opportunities for cache reuse. Hybrid models such as Ling-3.0-Flash (Ling Team, 2026a), Qwen3.8-Max (Qwen Team, 2026), and Kimi K3 (Kimi Team, 2026) require both KV cache and recurrent state for cross-turn reuse. For a given model, KV-cache memory grows with the number of cached tokens, whereas each recurrent state has a fixed size. Thus, longer contexts increase KV-cache demand per trajectory, while admitting more concurrent trajectories requires retaining more recurrent states as well as their associated KV cache. SGLang manages these states in separate pools within a shared cachememory budget. If a default or empirically chosen allocation ratio does not accurately reflect the target workload’s cache-demand characteristics, one pool can be exhausted while substantial capacity remains unused in the other. This imbalance limits the number of trajectories whose caches can be retained together. Our workload-aware capacity model quantifies the target workload’s recurrent-state and KV-cache requirements to balance memory allocation between the two pools at deployment and provide a reference sample concurrency, enabling greater parallelism while preserving cross-turn cache reuse. The resulting allocation remains fixed during inference serving.

Even with suitable allocation, reuse depends on where and when requests execute. With multiple inference workers, keeping a trajectory’s requests on the same worker favors reuse, but strict affinity can concentrate load and lengthen queues. Even on a worker holding reusable state, a request can wait long enough for that state to be evicted. In addition, model responses and environment results accumulate across turns, potentially producing very long agent contexts. Identifying cache-reuse opportunities through token-prefix matching can then incur substantial computational overhead. To reduce this overhead, we use a shared trajectory label to identify requests from the same trajectory at low cost, giving the router an affinity key and letting the worker screen waiting requests before querying the prefix cache for scheduling priority. We implement this approach through two complementary methods. Load-bounded consistent hashing (LBCH) routing uses the labels to preserve cross-turn worker affinity while accounting for worker load. Routing-prefix (RP)

scheduling uses them to prioritize requests from recently served trajectories and promote timely cache reuse at low scheduling cost.

Environment resources may also be constrained by CPU budgets or external-service quotas. A single sample-concurrency limit, as used in the agent implementations of MCP-Atlas and τ<sup>3</sup>-bench (Bandi et al., 2026; Shi et al., 2026), couples the two stages. When the environment supports less concurrency than the inference service, raising this shared limit to keep inference busy can overload environment execution. Conversely, lowering it to protect the environment can leave inference workers without ready requests while active samples await environment-interaction results. We therefore decouple sample admission from environment execution, allowing more samples to remain active while an independent execution limit protects the environment. This improves stage overlap but also makes environment-side queuing more likely as more admitted samples submit jobs to the limited execution slots. Their dispatch order determines when each trajectory can resume inference. Repeated waits can defer a long trajectory’s later stages until most other samples have finished, leaving few ready inference requests and prolonging the batch tail. When multiple environment jobs await execution, Chain-Advancing Action Scheduling (CAAS) uses observed progress to prioritize those from potentially long trajectories. This aims to expose their subsequent stages sooner and shorten the batch tail without requiring knowledge of remaining trajectory lengths.

Our contributions are summarized as follows.

1. Workload-based and trajectory-aware inference optimization. A hybrid-cache capacity model guides allocation and sample concurrency, complemented by LBCH routing for affinity–load balance and routing-prefix scheduling for timely cross-turn reuse.

2. Inference–environment coordination. Independent sample and environment concurrency controls improve stage overlap, while CAAS effectively advances long execution chains to shorten the batch tail.

3. Controlled replay and live end-to-end validation. Trace-driven replay combines actual inference with recorded environment responses and durations for controlled comparisons and ablations. Live evaluations with Ling-3.0-Flash and Ling-3.1-Flash across four well-known benchmarks demonstrate substantial reductions in batch makespan, while evaluation scores remain aligned (Section 5.4). The proposed methods are also applied in Ling-3.x model development, where batched agent workloads such as benchmark evaluation, synthetic-data generation, and RL rollouts play a key role. By accelerating these workloads, the methods substantially improve overall development efficiency.

The remainder of this paper presents the methods and experimental evaluations supporting these contributions. Section 2 reviews related work. Sections 3 and 4 present inference-side optimizations and inference–environment coordination, respectively. Section 5 evaluates the proposed methods through controlled replay and live execution. Section 6 discusses future directions, and Section 7 concludes the paper.

## 2 Related Work

Prefix caching. Prefix caching avoids repeated computation over shared prompt prefixes. Serving systems support prefix reuse through mechanisms such as SGLang’s RadixAttention (Zheng et al., 2024) and vLLM’s automatic prefix caching (vLLM Project, 2026). Mooncake, Tutti, and CacheFlow extend cache management across distributed or multi-tier settings (Qin et al., 2024; Qiu et al., 2026; Nian et al., 2026). Li et al. (2026) survey these broader cache-management techniques. BatchLLM exploits globally available prefix structure and throughput-oriented batching for large inference batches (Zheng et al., 2026), whereas we target cross-turn reuse within agent trajectories.

Cache management for hybrid LLMs. Cache reuse in hybrid LLMs requires retaining both KV cache and recurrent state. AVMP manages separate KV-cache and state-space-model (SSM) state pools through a unified virtual page table (Nguyen, 2026). It can transfer capacity between pools at runtime when allocation fails in one pool and the other has enough spare capacity. Marconi selects recurrent states for caching based on their reuse potential and guides eviction using recency and computation saved relative to the combined footprint of the KV cache and recurrent state (Pan et al., 2025). Our workload-aware capacity model uses trajectory profiles to select a static partition of the cache budget and a reference sample concurrency before serving, with the goal of increasing joint trajectory-cache residency.

Cache-aware routing and scheduling. Preble uses global prefix-cache information and computation estimates to route requests across workers (Srivatsa et al., 2024). Its local scheduler groups requests by the fraction of cached prompt tokens and selects requests from each group to form a batch, assigning larger shares to higher-priority groups. DualMap hashes a prompt prefix to two candidate workers and uses a time-to-first-token (TTFT) service-level objective to balance affinity and load (Yuan et al., 2026). PEEK maintains an incremental radix tree over pending requests to identify shared prefixes even before they are cached (Xie et al., 2026). It uses these relationships to schedule prefix-sharing requests and protect cache entries needed by queued requests. These methods use prompt-prefix information to guide routing or scheduling. In contrast, our LBCH routing and RP scheduling use trajectory labels to identify and exploit potential cache-reuse opportunities.

Agent-aware inference scheduling. Autellix schedules LLM calls using observed program-level attained service and uses program locality in cross-engine routing (Luo et al., 2025). Helium represents workflows as query plans and optimizes workflow makespan through rewriting, caching, and cost-based scheduling (Wadlom et al., 2026). SAGA uses predicted workflow structure for cache management and workflow-level scheduling (Guo et al., 2026). ConServe uses observable conversation state to place the first-turn prefill and maintains decoder affinity across turns (Ding et al., 2026). RP links successive turns of an agent trajectory through shared labels and recent serving activity, without tracking program-level attained service or requiring workflow graphs.

Environment-aware serving and action scheduling. SAGA uses expected tool-call durations to guide cache retention (Guo et al., 2026), while MORI uses relative tool-call idleness to guide cache placement across GPU and CPU memory (Xia et al., 2026). Both use environment behavior to inform inference-side cache decisions. ARL-Tangram directly schedules external actions, using first-come, first-served (FCFS) to select candidates and elastic resource allocation to reduce the sum of action completion times (Xiao et al., 2026). We instead order pending environment jobs under a fixed execution limit with separate sample admission. CAAS uses observed trajectory progress to advance long execution chains and shorten the batch tail without changing individual job service times.

## 3 Optimizing Inference Efficiency

This section presents our methods for improving inference efficiency, which are implemented in SGLang (Zheng et al., 2024). We first examine cache allocation for hybrid models, then present cross-worker routing and in-worker scheduling to improve cross-turn reuse.

## 3.1 Workload-Aware Hybrid-Cache Allocation

Hybrid models such as Ling-3.0-Flash (Ling Team, 2026a) combine Multi-head Latent Attention (MLA) with Kimi Delta Attention (KDA), requiring KV cache and recurrent state, respectively. Each inference worker maintains separate pools for these states within a shared cache-memory budget. Cache reuse requires both types of state for the reused prefix to remain available, so the more constrained pool limits how many trajectories can retain reusable state.

KV-cache demand grows with cached context length, whereas recurrent state occupies a fixed number of slots per request. As a trajectory extends its prefix, the latest reusable recurrent state summarizes that prefix and supports subsequent computation without retaining states from earlier turns. We therefore retain one recurrent state per trajectory, keeping its recurrent-state memory requirement fixed as the number of turns increases.

At low sample concurrency, performance is relatively insensitive to the cache ratio when both pools have sufficient capacity. Higher concurrency can shorten makespan, but also increases cache demand, so an imbalanced allocation can cause eviction and recomputation that offset parallelism gains. We therefore use a workload-aware capacity model to guide cache allocation and sample concurrency, aiming to sustain high cross-turn reuse while increasing parallelism.

Model development often involves repeated execution of batched agent workloads, including benchmark evaluation, synthetic-data generation, and RL rollouts, across successive checkpoints. For similar task distributions, the mean context length often changes little between nearby checkpoints, allowing historical trajectory profiles to provide reusable estimates for guiding cache allocation and sample concurrency.

## 3.1.1 Joint Capacity Model

Let M be the GPU memory budget available to the KV-cache and recurrent-state pools after model weights and non-cache runtime reservations. Let $M _ { \mathrm { k v } }$ and $M _ { \mathrm { s t a t e } }$ denote the memory allocated to the two pools, respectively, and define $r = M _ { \mathrm { s t a t e } } / M _ { \mathrm { k v } }$ . The resulting partition is

$$
M _ { \mathrm { s t a t e } } ( r ) = M \frac { r } { 1 + r } , \quad M _ { \mathrm { k v } } ( r ) = M \frac { 1 } { 1 + r } .\tag{1}
$$

For workload ${ \mathcal { D } } ,$ , we combine historical trajectory profiles with model and runtime parameters to estimate the per-trajectory cache footprints as

$$
F _ { \mathrm { k v } } ( \mathcal { D } ) = L _ { \mathrm { k v } } ( \mathcal { D } ) c _ { \mathrm { k v } } , \quad F _ { \mathrm { s t a t e } } = S _ { \mathrm { s t a t e } } c _ { \mathrm { s t a t e } } ,\tag{2}
$$

where $L _ { \mathrm { k v } } ( \mathcal { D } )$ is the average number of KV-cache entries required by a complete trajectory in ${ \mathcal { D } } ,$ measured at its final inference turn, $S _ { \mathrm { s t a t e } }$ is the number of recurrent-state slots required by each inference request under the deployment configuration, and $c _ { \mathrm { k v } }$ and $c _ { \mathrm { s t a t e } }$ are their respective unit memory costs. The quantities $S _ { \mathrm { s t a t e } } , c _ { \mathrm { k v } } .$ , and $c _ { \mathrm { s t a t e } }$ are fixed for a given model and runtime configuration. Because the accumulated context generally grows across turns, the final-turn KVcache allocation provides a conservative workload estimate, although it is not treated as a strict upper bound.

Based on the pool allocations in Equation 1 and the workload-derived cache footprints in Equation $^ { 2 , }$ we approximate the number of trajectories whose cache state can be retained in each pool simultaneously using the following continuous estimates:

$$
C _ { \mathrm { k v } } ( r ; \mathcal { D } ) = \frac { M _ { \mathrm { k v } } ( r ) } { F _ { \mathrm { k v } } ( \mathcal { D } ) } , \quad C _ { \mathrm { s t a t e } } ( r ) = \frac { M _ { \mathrm { s t a t e } } ( r ) } { F _ { \mathrm { s t a t e } } } .\tag{3}
$$

Under this per-trajectory footprint approximation, we estimate the joint residency capacity as the minimum of the two pool-specific estimates:

$$
C ( r ; \mathcal { D } ) = \operatorname* { m i n } \{ C _ { \mathrm { k v } } ( r ; \mathcal { D } ) , C _ { \mathrm { s t a t e } } ( r ) \} .\tag{4}
$$

The joint capacity $C ( r ; \mathcal { D } )$ estimates how many trajectories can simultaneously retain both KV cache and recurrent state on a worker at ratio $r ,$ providing a reference for sample concurrency. The analytical model treats the available cache memory as continuously divisible and abstracts away allocator-level details such as recurrent-state slot granularity, KV-page granularity, non-contiguous memory placement, and runtime fragmentation.

## 3.1.2 Capacity-Balancing Estimator

Under the continuous capacity model, joint residency is limited by the pool that can support fewer trajectories. Reallocating memory from the other pool to this limiting pool increases the joint capacity until both pools support the same number of trajectories. Setting $C _ { \mathrm { k v } } ( r ; \mathcal { D } ) = C _ { \mathrm { s t a t e } } ( r )$ gives

$$
\frac { M _ { \mathrm { k v } } ( r ) } { F _ { \mathrm { k v } } ( \mathcal { D } ) } = \frac { M _ { \mathrm { s t a t e } } ( r ) } { F _ { \mathrm { s t a t e } } } ,\tag{5}
$$

which yields the balancing target

$$
r ^ { \mathrm { b a l } } ( \mathcal { D } ) = \frac { F _ { \mathrm { s t a t e } } } { F _ { \mathrm { k v } } ( \mathcal { D } ) } = \frac { S _ { \mathrm { s t a t e } } c _ { \mathrm { s t a t e } } } { L _ { \mathrm { k v } } ( \mathcal { D } ) c _ { \mathrm { k v } } } .\tag{6}
$$

At this balancing ratio, the two pools support the same number of trajectories, and the joint capacity simplifies to

$$
C ( r ^ { \mathrm { b a l } } ( \mathcal { D } ) ; \mathcal { D } ) = \frac { M } { F _ { \mathrm { k v } } ( \mathcal { D } ) + F _ { \mathrm { s t a t e } } } .\tag{7}
$$

This balancing ratio may not be optimal in practice, since the workload-wide mean KV-cache footprint may not reflect the cache demand of the trajectories active at a given time. Optional local calibration can identify a better-performing nearby ratio. We use rˆ to denote the selected cache ratio, which can be either the model-derived value or a locally calibrated value. For deployment, the continuous capacity estimate $C ( \hat { r } ; \mathcal { D } )$ is rounded down to obtain the integer reference concurrency ⌊C(rˆ; D)⌋ per worker. For W homogeneous workers, we use the aggregate reference concurrency

$$
C _ { I } ^ { \star } ( \hat { r } ; \mathcal { D } ) = W \left\lfloor C ( \hat { r } ; \mathcal { D } ) \right\rfloor .\tag{8}
$$

The selected ratio rˆ stays fixed during inference serving. The reference concurrency does not impose a hard limit, since timely cache reuse may support higher concurrency. The results in Section 5.2.1 show that the model-derived cache ratios and reference concurrency settings provide effective operating points for the evaluated workloads.

## 3.2 Cross-Worker Trajectory-Aware Routing

We consider a distributed inference deployment comprising a gateway and multiple homogeneous workers. Each worker is a model-serving replica that independently schedules its assigned inference requests and maintains its own device-local KV-cache and recurrent-state pools. The gateway routes each inference turn to one of these workers. We focus on worker-local cache reuse without cross-worker cache transfer, so a later turn can reuse its trajectory cache only on the worker that currently holds the corresponding cached state.

Routing decisions based solely on load balancing may send successive turns of the same trajectory to different workers, missing opportunities for cache reuse. Conversely, prioritizing trajectory affinity without considering worker load can concentrate requests on a few workers. Requests may then queue at these workers while others remain idle, reducing overall inference efficiency. We adapt consistent hashing with bounded loads (Mirrokni et al., 2016) to per-turn inference routing. LBCH uses a shared trajectory label for affinity and current in-flight request counts for load control. This routing method requires every inference request in trajectory k to carry the same trajectory label $\kappa _ { k }$ . Let $\mathcal { W }$ denote the worker set and $W = | \mathcal { W } |$ its size. These workers are organized on a hash ring ${ \mathcal { R } } ,$ and $\kappa _ { k }$ is hashed to a position on $\mathcal { R }$ . The first worker clockwise from that position is defined as the home worker $h ( \kappa _ { k } )$ of trajectory k. Let $L ( w )$ denote the number of in-flight inference requests assigned to worker w that are either queued or running, and let $\begin{array} { r } { L _ { \mathrm { t o t } } = \sum _ { w \in \mathcal { W } } L ( w ) } \end{array}$ . A worker is eligible when

$$
L ( w ) \leq \frac { L _ { \mathrm { t o t } } + 1 } { W } \alpha ,\tag{9}
$$

where the +1 term accounts for the request currently being assigned, and $\alpha \geq 1$ is the load-bound factor. Values above 1 permit controlled load skew in favor of trajectory affinity, whereas $\alpha = 1$ gives the tightest eligibility threshold. The router selects $h ( \kappa _ { k } )$ if it is eligible. Otherwise, it traverses the ring clockwise with the successor operator succ $\pi ^ { ( w ) }$ , which returns the next worker clockwise from w on ${ \mathcal { R } } ,$ and selects the first eligible successor. If no successor is eligible, it falls back to the home worker. Therefore, the resulting procedure, summarized in Algorithm 1, relies only on trajectory labels and worker loads, without inspecting prompt text or predicting cache contents. The experiments in Section 5.2.3 demonstrate that LBCH reduces request-load imbalance across workers while maintaining high cache reuse.

Algorithm 1 Trajectory-aware routing with load-bounded consistent hashing (LBCH)   
Input: trajectory label $\kappa _ { k } ,$ hash ring ${ \mathcal { R } } ,$ loads $\{ L ( w ) \} _ { w \in \mathcal { W } } ,$ , load-bound factor α   
Output: selected worker   
1: $w _ { 0 } \gets h ( \kappa _ { k } )$   
2: $\begin{array} { r } { \tau  \frac { L _ { \mathrm { t o t } } + 1 } { W } \alpha } \end{array}$   
3: if $L ( w _ { 0 } ) \leq \tau$ then   
4: return $w _ { 0 }$   
5: end if   
6: w ← w<sub>0</sub>   
7: repeat   
8: w ← succ<sub>R</sub>(w)   
9: if w ̸= w and $L ( w ) \leq \tau$ then   
10: return w   
11: end if   
12: until $w = w _ { 0 }$   
13: return $w _ { 0 }$

## 3.3 In-Worker Routing-Prefix Scheduling

Worker affinity alone cannot guarantee cache reuse. At high sample concurrency, a request may wait in the worker’s queue long enough for its trajectory’s cached state to be evicted. When the request is eventually served, the worker must recompute the affected prefix. Routing-prefix prioritizes queued requests from recently served trajectories to promote timely reuse while limiting prefix-matching overhead.

For each worker, let $\mathcal { Q } = \left( q _ { 1 } , \ldots , q _ { n } \right)$ be its waiting queue, A its trajectory-label set with capacity $C _ { A } ,$ and C its radix prefix cache. The trajectory-label set is updated using a least recently used (LRU) rule. Algorithm 2 summarizes the policy in three stages: scoring the waiting requests, building the next batch, and updating A. Each request q has a trajectory label $\kappa ( q )$ , a cached prefix-hit length $\ell ( q )$ , and a queue-entry time $t _ { \mathrm { q u e u e } } ( q )$ . It also carries a binary prefix-query flag $\delta _ { q } \in \{ 0 , 1 \}$ initialized to 1 when the request is created. During scoring, requests whose labels belong to A form the first tier. The operator $\mathrm { P r e f i x H i t } ( \mathcal { C } , q )$ returns the maximum prefix-match length for $q$ in ${ \mathcal { C } } ,$ measured in tokens. For each first-tier request with $\delta _ { q } = 1 _ { \cdot }$ , the scheduler queries C via PrefixHit $( \mathcal { C } , q )$ , records the resulting value as $\ell ( q )$ , and sets $\delta _ { q }$ to 0; when $\delta _ { q } = 0$ , the recorded $\ell ( q )$ is reused. For requests with $\kappa ( q ) \notin { \mathcal { A } } .$ , the scheduler skips the prefix-cache query. The priority key is defined as

$$
p ( q ) = \left\{ \begin{array} { l l } { ( 0 , - \ell ( q ) , t _ { \mathrm { q u e u e } } ( q ) ) , } & { \kappa ( q ) \in \mathcal { A } , } \\ { ( 1 , 0 , t _ { \mathrm { q u e u e } } ( q ) ) , } & { \kappa ( q ) \notin \mathcal { A } . } \end{array} \right.
$$

The operator $\mathrm { S o r t } _ { \mathrm { l e x } } ^ { \uparrow } ( \mathcal { Q } , p )$ orders the requests in Q by ascending lexicographic order of their priority keys $p ( q )$ . In this ordering, the first component places requests with labels in A ahead of the others, the second ranks those requests by decreasing prefix-hit length, and the third preserves queue-entry order when the preceding components are tied.

Algorithm 2 Routing-prefix policy   
Input: waiting queue Q, label set A, capacity $C _ { A } ,$ radix prefix cache C   
Output: admitted batch B, remaining queue Q, updated label set A   
1: for all $q \in \mathcal { Q }$ do   
2: if κ(q) ∈ A then   
3: if $\delta _ { q } = 1$ then   
4: $\dot { \ell } ( q ) \gets \operatorname { P r e f i x H i t } ( \mathcal { C } , q )$   
5: $\delta _ { q } \gets 0$   
6: end if   
7: $p ( q ) \gets ( 0 , - \ell ( q ) , t _ { \mathrm { q u e u e } } ( q ) )$   
8: else   
9: $p ( q ) \gets ( 1 , 0 , t _ { \mathrm { q u e u e } } ( q ) )$   
10: end if   
11: end for   
12: $\mathcal { Q }  \mathrm { S o r t } _ { \mathrm { l e x } } ^ { \uparrow } ( \mathcal { Q } , p )$   
13: $( B , \mathcal { Q } ) $ BuildBatch(Q)   
14: for all $q \in B$ do   
15: A ← LRUUpdate $( \mathcal { A } , \kappa ( \boldsymbol { q } ) , C _ { \mathcal { A } } )$   
16: end for   
17: return $B , \mathcal { Q } , A$

After this reordering, the existing batch builder scans Q in order and admits requests subject to the current resource limits. This operation is denoted by $( \boldsymbol { \mathcal { B } } , \boldsymbol { \mathcal { Q } } ) \gets \mathrm { B u i l d B a t c h } ( \boldsymbol { \mathcal { Q } } )$ , where B is the admitted batch and Q is the remaining queue. For admitted requests only, A is updated using their labels according to

$$
\mathrm { L R U U p d a t e } ( \mathcal { A } , \kappa , C _ { \mathcal { A } } ) = \mathrm { t a i l } _ { C _ { \mathcal { A } } } \big ( ( \mathcal { A } \setminus \{ \kappa \} ) \big | | \langle \kappa \rangle \big ) ,
$$

where ∥ appends κ at the most-recent position and $\mathrm { t a i l } _ { C _ { A } }$ retains the $C _ { A }$ most recent labels.

The label set A records recent batch admissions, not GPU cache residency. Its LRU updates are independent of GPU cache eviction, so labels and cached state can become inconsistent in two ways. First, a label may remain in A after the corresponding cached state has been evicted. If eviction occurs before the prefix lookup, the resulting hit length may be short or zero, potentially lowering the request’s rank within the first tier. If eviction occurs after the lookup, the stored hit length may become stale because subsequent priority calculations reuse it without another lookup. Second, a label may be removed from A while corresponding cached state remains on the worker: removing a label does not evict that state. A request with this label skips the priority-related prefix lookup and enters the second tier, potentially missing an opportunity to prioritize reuse of the remaining cached state.

The capacity $C _ { A }$ therefore controls label coverage, not GPU cache allocation. Larger sets offer more reuse opportunities at higher prefix-query cost, while smaller sets reduce this cost but may miss reuse opportunities. We use $C _ { \mathcal { A } } = \lfloor C ( \hat { r } ; \mathcal { D } ) \rfloor$ as a model-informed default, matching the set capacity to the estimated number of trajectories whose KV cache and recurrent state one worker can retain under the selected cache ratio. Section 5.2.2 evaluates whether this RP design preserves cache reuse and limits priority-calculation cost under cache pressure.

## 3.4 Integration with SGLang

At deployment, we apply the selected cache ratio rˆ through SGLang’s native mamba\_full\_memory\_- ratio parameter. Since each trajectory requires only its latest reusable recurrent state, as explained in Section 3.1, we use SGLang’s native path-based Mamba-state eviction with mamba\_max\_states\_- per\_path=1 to prune older states as the trajectory advances. We add LBCH as a routing policy in SGLang’s router and RP as a scheduling policy in each worker. Both policies use the same trajectory label carried as request metadata. LBCH selects a worker using this label and in-flight request counts. RP reorders the waiting queue while reusing SGLang’s existing prefix-cache lookup and resource-constrained batch builder. Sample admission remains controlled by the agent execution layer, as described in Section 4.

## 4 Coordinating Inference and Environment Interaction

In the agent execution layer, we decouple sample admission from environment execution by setting their concurrency limits independently, and use CAAS to dispatch pending environmentinteraction jobs. Trajectory k has $N _ { k }$ inference stages ${ \cal I } _ { k , n } ,$ with $E _ { k , n }$ denoting the corresponding environment-interaction stage, when present. Execution follows

$$
I _ { k , 1 } \to E _ { k , 1 } \to I _ { k , 2 } \to \cdots \to I _ { k , N _ { k } } \left( \to E _ { k , N _ { k } } \right)\tag{10}
$$

and completes after the final inference stage or, if present, the corresponding environment stage. Completing a nonterminal environment stage releases the next inference turn, so each trajectory has at most one outstanding environment stage.

## 4.1 Separating Sample Admission from Execution Capacity

Although all samples are eligible at batch start, admitting them all at once can overload inference and environment services. The sample-admission limit $C _ { S }$ bounds the number of active samples. Each sample holds a slot from admission to completion, including while waiting for inference or environment interaction. Waiting samples are admitted in submission order as slots become available. Each environment-interaction stage is modeled as a non-preemptive job comprising one or more operations. The environment-execution capacity $C _ { E }$ limits the number of concurrently running jobs.

In practice, agent-evaluation harnesses such as MCP-Atlas (Bandi et al., 2026) and $\tau ^ { 3 }$ -bench (Shi et al., 2026) commonly use a single task-level concurrency limit. We model this as a unified concurrency gate that constrains sample admission by both inference and environment capacity, giving

$$
C _ { S } ^ { \mathrm { u n i f i e d } } = \operatorname* { m i n } \{ C _ { I } ^ { \star } ( \hat { r } ; \mathcal { D } ) , C _ { E } \} ,\tag{11}
$$

where $C _ { I } ^ { \star }$ is the inference-side reference concurrency from Section 3.1. When $C _ { E } < C _ { I } ^ { \star }$ , this coupling can underutilize GPUs while samples await environment results. We instead default to

$$
C _ { S } ^ { \mathrm { d e c o u p l e d } } = C _ { I } ^ { \star } ( \hat { r } ; \mathcal { D } )\tag{12}
$$

and independently cap running environment-interaction jobs at $C _ { E }$ . Pending jobs retain sample slots but no environment-execution slots. Figure 11 demonstrates the benefits of improved inference–environment overlap.

## 4.2 Chain-Advancing Action Scheduling

Capacity separation allows more active trajectories under the same environment-execution limit, which can increase queueing for environment jobs. Their dispatch order determines which trajectory resumes inference next. In practice, a few long trajectories often prolong batch completion after most samples have finished, further reducing GPU utilization. CAAS therefore aims to identify and advance potentially long trajectories earlier through environment-job scheduling to shorten batch makespan.

We consider a homogeneous execution pool where any pending job can use any available slot. When a slot becomes available, CAAS selects the next job without changing the agent’s actions or job service times. For job $j ,$ let $a _ { j }$ be its slot-request time, $n _ { j }$ the number of completed inference turns in its trajectory, including the turn that generated job $j ,$ , and $\tau _ { m }$ the observed duration of its mth inference turn. With hyperparameters $B > 0$ and $\Delta _ { I } > 0$ , we define

$$
g _ { j } = \left\lfloor \frac { a _ { j } } { B } \right\rfloor ,\tag{13}
$$

$$
\bar { \tau } _ { j } = \frac { 1 } { n _ { j } } \sum _ { m = 1 } ^ { n _ { j } } \tau _ { m } ,\tag{14}
$$

$$
c _ { j } = \left\lceil \bar { \tau } _ { j } / \Delta _ { I } \right\rceil .\tag{15}
$$

Here $g _ { j }$ groups jobs by arrival time, while $c _ { j }$ discretizes their observed mean inference duration. When job j joins the pending queue, CAAS assigns it the priority key

$$
p _ { E } ( j ) = ( g _ { j } , - n _ { j } , c _ { j } , e _ { j } ) ,\tag{16}
$$

where $e _ { j }$ is a globally increasing entry sequence. Each job’s key remains fixed while it waits, and pending jobs are sorted in ascending lexicographic order. The ordering first protects jobs in earlier arrival bands through $g _ { j } ,$ then prioritizes deeper revealed chains through $- n _ { j }$ . Within the same band and depth, $c _ { j }$ provides a coarse tie breaker favoring a shorter estimated inference return, with $e _ { j }$ resolving remaining ties in FCFS order. A smaller B favors arrival order, whereas a larger B gives chain depth more influence. Duration bucketing reduces noise in the inference-duration tie breaker.

CAAS makes online dispatch decisions using only trajectory state observed in the current run, without requiring sample-specific historical profiles or predictions of total or remaining execution time. With future trajectory lengths unknown, CAAS uses completed inference turns as a proxy for potentially long trajectories and prioritizes deeper chains within each arrival band. This allows subsequent stages to be revealed and executed sooner and reduces the risk that long chains are left to dominate batch completion. Arrival bands prevent returning trajectories from repeatedly overtaking shallower jobs waiting in earlier bands, as can happen under depth-only ordering. Section 5.3 shows that CAAS reduces batch tail and makespan over FCFS after capacity separation.

## 5 Experimental Evaluation

Our evaluation examines how the proposed optimizations reduce batch makespan, both individually and in combination. Section 5.1 first describes the replay framework and experimental setup. Using controlled replay, we evaluate how cache allocation, sample concurrency, scheduling, and routing affect inference efficiency (Section 5.2), and isolate the effects of capacity separation and CAAS under constrained environment capacity (Section 5.3). Live experiments then evaluate their combined effects across models of different sizes (Section 5.4).

## 5.1 Controlled Replay Framework and Setup

Real-world environment interactions vary with network conditions, provider load, rate limits, retries, and response drift, making repeatable comparisons across system configurations difficult. To support controlled comparisons, we combine actual model inference with recorded environment responses and service durations from historical agent trajectories. Replay runs within each comparison use the same initial messages, stage sequences, per-turn target output lengths, and recorded environment responses and durations.

Figure 2 illustrates the turn-by-turn replay of a trajectory. At each inference turn, the input consists of the initial user message followed, in execution order, by model responses generated in preceding replay turns and recorded environment responses released earlier in the replay. A deployed SGLang service performs actual inference, generating the same number of output tokens as the corresponding historical turn. If an environment stage follows, the trajectory waits for an available execution slot and then pauses for the recorded job duration. The framework subsequently appends the recorded environment response to the context, making the next inference turn ready.

No environment actions execute during replay. Stage dependencies and environment durations remain fixed, while inference latency and cache reuse arise from actual serving. The replay timeline includes measured inference times, recorded environment durations, and queueing delays under the selected admission limits, execution capacity, and scheduling policies. Batch makespan is the elapsed time until all trajectories complete. Replay therefore measures execution efficiency rather than semantic correctness, as generated text need not reproduce the recorded action.

All replay experiments use Ling-3.0-Flash with thinking enabled, served by SGLang on H20 GPUs with four-way tensor parallelism (TP4) per worker. The model combines KDA and Multi-head Latent Attention layers at a 5:1 ratio with a sparse MoE backbone (Ling Team, 2026a). Inferenceonly replay covers SWE-bench Multilingual (SWE) (Khandpur et al., 2025), BFCL-v4 (BFCL) (Patil et al., 2025), and WideSearch (Wong et al., 2025), bypassing environment durations and capacity limits. Coordination replay uses MCP-Atlas (Bandi et al., 2026) with recorded environment stages. Within each replay comparison, we keep the checkpoint, sampling parameters, workload, hardware budget, submission order, and target output lengths fixed. Appendix A.1 provides the inferenceonly replay workload details.

![](images/4388df515ba6bc5f0363f5ab47b7ddf503e72fee2e73283dcfd883d80d6b1d94.jpg)  
Figure 2 Overview of the controlled trace-driven agent replay framework. The numbered markers indicate optimization locations. ⃝1 Capacity separation: the sample-admission limit $C _ { S }$ is set independently of the environment-action concurrency limit $\ C _ { E } . \textcircled { 2 }$ Inference optimizations: cache allocation, routing-prefix scheduling, and LBCH routing. ⃝3 CAAS for job ordering.

## 5.2 Inference-Side Evaluation

## 5.2.1 Cache allocation and admission operating point

We evaluate the joint effect of cache allocation and sample admission on a single TP4 worker with FCFS scheduling. Using Equation 6, we obtain $r ^ { \mathrm { b a l } } = 0 . 0 9$ for SWE, $r ^ { \mathrm { b a l } } = 2 . 2 6$ for BFCL, and $r ^ { \mathrm { b a l } } = 0 . 1 0$ for WideSearch. The corresponding reference concurrency, $C _ { \mathrm { r e f } } = \lfloor C ( r ^ { \mathrm { b a l } } ; \mathcal { D } ) \rfloor$ , is 98, 859, and 98, respectively. We perform a grid search over cache ratios and sample-concurrency limits around these model-derived reference points. Figure 3 summarizes the resulting makespans.

Joint operating region. Under FCFS scheduling, the model-derived operating points yield makespans close to the best measured values across all three workloads, leaving only modest room for further improvement through local search. The analytical capacity model abstracts away allocation granularity and runtime fragmentation. Nevertheless, these results show that it provides a simple yet effective heuristic for jointly selecting the cache ratio and sample concurrency by accounting for trajectory token footprints and the number of trajectories that can remain resident.

The sweeps around the model-derived operating points show that the benefit of increasing concurrency depends on the cache ratio. At a fixed ratio, raising concurrency from below $C _ { \mathrm { r e f } }$ can initially shorten makespan, but further increases may lead to sharp slowdowns or, on SWE, six-hour timeouts. Conversely, at a fixed concurrency, makespan can decrease and then increase as the ratio grows. Furthermore, at sufficiently low concurrency, makespan is relatively insensitive to the ratio, whereas ratios near $r ^ { \mathrm { b a l } }$ generally yield shorter makespans at higher concurrency. We next examine cache reuse to explain these trends.

Cache reuse and the cost of excess concurrency. This limited sensitivity at low concurrency is consistent with both cache pools having sufficient capacity to retain reusable state across a broad range of ratios. At higher concurrency, an imbalanced partition can make one pool a bottleneck, whereas ratios near $r ^ { \mathrm { b a l } }$ balance the two residency capacities and help preserve cross-turn reuse.

![](images/908ae99273197b0f1d48ff58cdef757a2477a38c20eb795507e22a4e34b36356.jpg)  
Figure 3 Cache allocation and sample-concurrency sweeps on (a) SWE, (b) BFCL, and (c) WideSearch. Cell values are complete-batch makespans in minutes; the panels use separate color scales. Second tick-label lines indicate multiples of $r ^ { \mathrm { b a l } }$ or $C _ { \mathrm { { r e f } } } ,$ except for the default ratio. Magenta outlines mark the best measured points. TO denotes a six-hour timeout and is excluded from the continuous color scale.

Figures 4 and 5 relate the concurrency trends to cache reuse and system throughput at each workload’s best measured cache ratio. Cache reuse measures the fraction of input tokens served from cache, while new-prefill work counts input tokens processed without cached reuse. We also report the time-averaged number of requests in service, including waiting, and output throughput, measured as successful output tokens per second.

Cached input tokens (left) New-prefill tokens (right)

![](images/d49d356fb5d51345c41306fe5c605ca73519dd2fb256599552dae0c3f99607a4.jpg)

![](images/83c882b8557ca8372a83b8ec380286273cbfb027c4a95713a34720067c8e9241.jpg)

![](images/a84d55a2a5da89024595d367099cd51bc4b9965fe7e99ad5afcf30b367826a41.jpg)  
Figure 4 Cache reuse and cumulative new-prefill work versus sample concurrency at the best measured ratio for each workload: (a) SWE, $r = 0 . 1 3 5 ;$ (b) BFCL, r = 1.13; (c) WideSearch, $r = 0 . 1 5$ . Blue circles use the left axis and orange squares the right axis. Every point represents a complete batch. Concurrency configurations are equally spaced on the horizontal axis. Shading highlights the concurrency ranges where cache reuse drops sharply and prefill recomputation increases (SWE: 147–157; BFCL: 644–859; WideSearch: 123–147).

Within the efficient operating region where cache reuse remains high and stable, increasing concurrency leaves new-prefill work largely unchanged while reducing makespan. The mean number of requests in service and output throughput increase together (Figure 5). This pattern is consistent with higher concurrency improving GPU utilization through increased inference parallelism,

Mean requests in service (left) Output throughput (right)

thereby raising system throughput without substantial additional prefill work.

![](images/f6ad771695c36351969fb67d2a7ce8d43a02b96c39ba4a770d77a10d6027263b.jpg)

![](images/91be604e9371331b9c0944155dd8d5578defe986fb11ec8eed63e33f7d472d81.jpg)

![](images/07e7ea96bdb112bb300eb5687028bd9236b8450e7cee5a0f1f6ea8021940ea71.jpg)  
Figure 5 Mean requests in service (left) and output throughput (right) versus sample concurrency at the same cache ratios as Figure 4: (a) SWE; (b) BFCL; (c) WideSearch. Concurrency configurations are equally spaced on the horizontal axis. Shading highlights the concurrency ranges where output throughput drops despite increasing requests in service (SWE: 147–157; BFCL: 644–859; WideSearch: 123–147).

The shaded regions in Figures 4 and 5 highlight where this relationship reverses. Increasing $C _ { S }$ from 147 to 157 for SWE, from 644 to 859 for BFCL, and from 123 to 147 for WideSearch causes cache reuse to decline and new-prefill work to rise to approximately eleven, three, and nine times its previous level, respectively. More requests reside in the engine, yet output throughput falls and makespan increases. Higher concurrency increases the demand for cached state from concurrently active trajectories. The resulting memory pressure can cause reusable cache entries to be evicted, reducing prefix reuse and increasing prefill recomputation.

Therefore, different cache ratios support different concurrency ranges for effective cross-turn cache reuse. Within these ranges, higher concurrency can improve inference parallelism and throughput; beyond them, prefix recomputation can outweigh the gains from additional parallelism. The cache ratio and sample concurrency should be selected jointly.

## 5.2.2 In-worker scheduling

We compare FCFS, SGLang’s native longest prefix match (LPM) and depth-first search weighting (DFS-weight), and routing-prefix scheduling on a single TP4 worker using the same inference-only workloads. We fix the cache ratio at the model-derived balanced allocation and vary sample concurrency to evaluate scheduling performance under low, moderate, and high cache pressure. Policy settings are provided in Appendix A.1.

Performance under cache pressure. Figure 6 compares complete-batch makespans. At low concurrency, makespans are generally close across scheduling policies, consistent with limited cache competition and little queueing, leaving relatively few opportunities to benefit from reordering requests. At the two highest tested concurrency levels, all three cache-aware policies achieve shorter makespans than FCFS across the three workloads. Together with the cache-reuse measurements discussed below, this contrast highlights the importance of identifying and prioritizing requests with reusable cached state to accelerate multi-turn agent inference under cache pressure. Among the three cache-aware policies, routing-prefix achieves shorter makespans than LPM and DFS-weight at most tested concurrency settings.

(a) SWE  
![](images/8f2fc77193f31bbd081f7f1f79db77cf9995e5f0e4d84b03be9e1abbdc6ec894.jpg)  
Figure 6 In-worker scheduling: (a) SWE; (b) BFCL; (c) WideSearch. Cell values are complete-batch makespans in minutes, with separate logarithmic color scales. Magenta outlines mark the minimum makespan for each workload before rounding. The hatched cell marked > 13 h denotes a timeout for SWE FCFS at $C _ { S } = 2 8 9$ under the 13-hour limit.

Cache reuse and scheduling overhead. Figure 7 relates these makespan differences to cache reuse and new-prefill work. At the lowest concurrency, the cache metrics are close across policies. As concurrency increases, routing-prefix continues to achieve high cache reuse with little additional prefill work.

![](images/27bd803933df1c65f0df592f1dae13bc199c144146703fecd140fa00988636d4.jpg)

(b) BFCL  
![](images/2ac970d060eec2e0b5c03f5c75e79f7b2ce6ab76756de200cb24ab2631b8a523.jpg)

![](images/b059086a3ffb48a34d87e279eac619df243cc5b31118f112552683dcc52433c2.jpg)

![](images/b55e1808c5dd6be08edc7decbef0e2f032bf4d5f550d2960b3ea2f147de1373e.jpg)

![](images/8296450a1bc932cfe0051bd0b77e0d4d2285b821a7c7956b6a157d3d68460425.jpg)

![](images/13424474145a99a543a819502c01b96310e3a31acc2a7c8abe9d1ed14ab8deba.jpg)  
Figure 7 Cache reuse (top) and cumulative new-prefill work (bottom) from the scheduling runs under the configurations in Figure 6. Each point represents a complete batch; the SWE FCFS timeout at $C _ { S } = 2 8 9$ is omitted. The SWE new-prefill axis is logarithmic. The LPM, DFS-weight, and routing-prefix curves nearly overlap on WideSearch.

On SWE, routing-prefix and DFS-weight maintain similarly high cache reuse and low new-prefill work across the tested concurrency levels, whereas LPM shows a sharp drop in reuse and a corresponding rise in new-prefill tokens at the highest concurrency. On BFCL, routing-prefix achieves higher cache reuse and fewer new-prefill tokens than LPM and DFS-weight at the two highest concurrency levels, with a much greater advantage over LPM. On WideSearch, all three cache-aware policies maintain similarly high cache reuse and low new-prefill work, with their curves nearly overlapping. Together, these results highlight routing-prefix’s consistently high cache reuse and low new-prefill work across the evaluated workloads and concurrency levels.

Cache reuse alone, however, does not fully explain why routing-prefix achieves shorter makespans even when cache metrics are similar across policies. We therefore examine scheduling overhead. Table 1 reports cumulative priority-calculation time at the highest tested effective sample concurrency for each workload. Routing-prefix incurs less overhead than DFS-weight on all three workloads and less than LPM on SWE and WideSearch. LPM repeatedly matches prefixes for queued requests when computing priorities, while DFS-weight additionally traverses the cache tree. Routing-prefix limits this work through selective prefix matching and reuse of recorded hit lengths. LPM has lower overhead on BFCL because it frequently falls back to FCFS when its waiting queue exceeds SGLang’s built-in threshold of 128 requests. This fallback is intended to limit prefix-matching and sorting overhead on long queues, but bypasses cache-aware prioritization. Its lower calculation cost therefore does not by itself imply more efficient inference execution.

Table 1 Cumulative wall-clock time spent computing scheduling priorities over a complete batch at the highest tested effective sample concurrency for each workload. Bold values mark the minimum in each row.
<table><tr><td>Workload</td><td> $C _ { S }$ </td><td>LPM (s)</td><td>DFS-weight (s)</td><td>routing-prefix (s)</td></tr><tr><td>SWE</td><td>289</td><td>735.00</td><td>1704.25</td><td>303.05</td></tr><tr><td>BFCL</td><td>1950</td><td>7.65</td><td>172.15</td><td>10.91</td></tr><tr><td>WideSearch</td><td>200</td><td>62.82</td><td>144.08</td><td>39.97</td></tr></table>

By prioritizing queued requests from recently served trajectories, routing-prefix promotes reuse before their cached state is evicted. Trajectory-aware scheduling complements cache allocation and sample admission. An appropriate cache ratio and concurrency setting can preserve crossturn reuse and achieve short makespans even with FCFS. Routing-prefix helps sustain reuse with low scheduling overhead as cache competition intensifies, making performance less sensitive to increased concurrency.

## 5.2.3 Cross-worker routing

We compare SGLang’s native cache-aware routing (CA), native consistent hashing (CH), and LBCH on two TP4 workers using the same inference-only workloads. All workers use routing-prefix scheduling, and the cache ratio and sample-concurrency limit are kept fixed across routing policies within each workload. The workload-specific configurations and policy parameters are listed in Appendix A.1.

Batch makespan and cache reuse. Figure 8 shows that LBCH achieves the shortest measured makespan on all three workloads, with reductions of 1.5%–13.6% relative to CA and 1.9%–3.8% relative to CH. Figure 9 compares cache reuse and new-prefill work. Cache reuse rates are similar across policies within each workload (98.4%–98.8% on SWE, 86.2%–87.2% on BFCL, and 95.2%– 95.7% on WideSearch), and LBCH does not consistently minimize new-prefill work. Therefore, similar cache reuse does not necessarily translate into similar makespans, motivating a closer examination of how routing policies distribute requests across workers.

Load distribution. To assess load distribution across workers, we use two complementary metrics. Let $N _ { w } ( t )$ denote the number of in-flight inference requests, including queued and executing requests, on worker w at time t.

(c) WideSearch

New-prefill work (bars, left axis) Cache reuse (line, right axis)

![](images/1787b87bc1186c0e5cf8472f8d3046a3f681736da75959172b900f9362c0c72c.jpg)

(b) BFCL  
![](images/4729ddbd3969f9a9b454e72bdbccc3f1c9407e0dc2e13439dd4a7b7dc8819195.jpg)

![](images/67e9013ae2f7c37d2089d3e5afcc7b32d7aa5b9b51a4e19dc0fc774392cb5b15.jpg)  
Figure 8 Cross-worker routing on two TP4 workers: (a) SWE; (b) BFCL; (c) WideSearch. Bars show measured batch makespan in minutes; the panels use separate vertical scales that do not start at zero.

(a) SWE  
![](images/2c1675b75c033750cfbdc078f4c2bcd3fe8d962df8dcb3eca38223fb9ccfc68f.jpg)

(b) BFCL  
![](images/0963e40bfeab4cd465b4bb33b7f2645e1d435f622afdf4da3d8a8ccb35eb3d66.jpg)

![](images/dc03570ea87501c37f8ce6572408a34bc8c4c4e46589c5a2e58c40b6fd649189.jpg)  
Figure 9 Cache reuse and new-prefill work for the runs in Figure 8: (a) SWE; (b) BFCL; (c) WideSearch. Bars show new-prefill work on the left axis; the purple line shows cache reuse on the right axis. Workload panels use separate scales for new-prefill work.

Let T span from the start of the first request to the completion of the last request, and let $J = \{ t \in$ $[ 0 , T ] : \bar { N } _ { 1 } ( t ) + N _ { 2 } ( t ) > 1 \}$ denote the intervals with more than one in-flight request across both workers. We denote the time-averaged request-count imbalance over these retained intervals by $\beta ,$ defined as

$$
\beta = \frac { 1 } { | J | } \int _ { J } d ( t ) d t ,\tag{17}
$$

where $\left| J \right|$ is the total retained duration and $d ( t )$ is the instantaneous request-count imbalance,

$$
d ( t ) = \frac { | N _ { 1 } ( t ) - N _ { 2 } ( t ) | } { N _ { 1 } ( t ) + N _ { 2 } ( t ) } , \qquad t \in { \cal J } .\tag{18}
$$

This metric gives equal weight to each unit of retained elapsed time. We also measure output-token imbalance, denoted by $\gamma _ { \mathrm { o u t } } ,$ to capture differences in generation workload, since request counts alone do not capture differences in output length. Let $D _ { w }$ be the total number of output tokens generated by requests executed on worker w over the batch:

$$
D _ { w } = \sum _ { i \in \mathcal { R } _ { w } } O _ { i } ,\tag{19}
$$

where $O _ { i }$ is request $i \prime \mathrm { s }$ actual output length and $\mathcal { R } _ { w }$ contains requests actually executed on worker w. We define $\gamma _ { \mathrm { o u t } }$ as

$$
\gamma _ { \mathrm { o u t } } = \frac { \left| D _ { 1 } - D _ { 2 } \right| } { D _ { 1 } + D _ { 2 } } .\tag{20}
$$

![](images/b2fbe14e45207d5523325f17b757f8ee162cc67c5fa7d6741acf0cb9aee6bf5b.jpg)  
Figure 10 Routing load distribution on (a) SWE, (b) BFCL, and (c) WideSearch. Top: time-averaged request-count imbalance over intervals with more than one in-flight request across both workers; each unit of retained time has equal weight. Bottom: output-token imbalance over the complete batch. Each row uses a common percentage scale. Lower values indicate more even distributions of requests over time and generated tokens across workers, respectively.

A lower $\beta$ indicates more balanced in-flight request counts across workers over time, while a lower $\gamma _ { \mathrm { o u t } }$ indicates a more balanced distribution of generated tokens across workers over the complete batch. Note that these metrics characterize load distribution rather than directly measuring computational cost. Figure 10 shows that LBCH achieves lower request-count imbalance than CA and CH across all three workloads. Its advantage over CA is particularly pronounced on SWE and BFCL, where output-token imbalance is also substantially lower. Compared with CH, LBCH improves both metrics on SWE and WideSearch, while on BFCL it achieves lower request-count imbalance with nearly identical output-token imbalance. Together with the cache-reuse results, these observations show that LBCH maintains high reuse while improving load balance, even though it does not always minimize new-prefill work. These results highlight the importance of jointly considering cache reuse and load distribution when routing multi-turn agent requests.

## 5.3 Inference–Environment Coordination

Using the replay framework in Section 5.1, we evaluate capacity separation and CAAS under constrained environment-execution capacity on historical trajectories from MCP-Atlas (Bandi et al., 2026), whose tasks rely on external MCP services that may degrade under high concurrency or encounter upstream rate limits. The inference deployment uses two TP4 workers with routingprefix scheduling and LBCH. Guided by the capacity model in Section 3.1, we use $C _ { S } = 5 0$ as the inference-side admission setting. To isolate the effect of capacity separation, we compare unified admission $( C _ { S } = C _ { E } )$ with decoupled admission at $C _ { S } \in \{ 1 0 , 2 5 , 5 0 \}$ for environment-execution limits of $C _ { E } \in \{ 3 , 5 \}$ , both with FCFS job scheduling. We then compare CAAS with FCFS at $C _ { S } = 5 0$ under the same $C _ { E } ,$ changing only job ordering while keeping the inference configuration and recorded workload unchanged. Configuration details and supporting background on environment constraints are provided in Appendix A.2.

Capacity separation. Figure 11 shows the effect of capacity separation at fixed environmentexecution capacity.

![](images/73198feb16554605a2ba1ae87012cd2a37d3c27fae8f8ebc2e2a541a345441c1.jpg)  
Figure 11 Capacity separation under FCFS job scheduling: batch makespan (top) and mean in-flight inference requests (bottom). The first point uses unified admission $C _ { S } = C _ { E } .$ . In-flight requests are measured from client request launch to response completion and averaged over the replay makespan.

Increasing sample admission from the unified baseline to $C _ { S } = 1 0$ reduces makespan by approximately one third at both environment-execution capacities. Capacity separation thus provides substantial gains even at modest sample concurrency. Further increases bring smaller improvements, with makespan fluctuating within a narrow range at $C _ { S } = 2 5 , 5 0$

The mean number of in-flight inference requests helps explain this pattern. It rises substantially even when sample admission is increased only to $C _ { S } = 1 0 .$ , but barely changes when $C _ { S }$ doubles from 25 to 50 at either environment-execution capacity. By supplying more concurrent inference work, capacity separation can improve GPU utilization and thereby increase overall system throughput. Admitting more samples allows inference and environment-interaction stages to overlap across trajectories, yet the limited environment-execution capacity constrains the release of subsequent inference work. Additional samples may then spend more time waiting for environment interaction rather than supplying concurrent inference work, so the makespan gains level off. At matched sample concurrency, $C _ { E } = 5$ supports more in-flight requests and shorter makespans than $C _ { E } = 3 ,$ consistent with earlier release of subsequent inference work.

CAAS and the completion tail. We evaluate CAAS at $C _ { S } = 5 0$ after capacity separation, where it reduces makespan relative to FCFS by 1.9% at $C _ { E } = 3$ and 3.5% at $C _ { E } = 5$ . To examine how CAAS affects completion of the last few tasks, we measure the duration of the batch tail. Measuring time from the first task admission, let $t _ { q }$ denote when the completed fraction first reaches or exceeds $q \% ,$ for $0 < q \leq 1 0 0 ; t _ { 1 0 0 }$ is the batch makespan. The final-p% tail is the interval from $t _ { 1 0 0 - p }$ to t<sub>100</sub>, with duration

$$
L _ { p } = t _ { 1 0 0 } - t _ { 1 0 0 - p } , \qquad 0 < p < 1 0 0 .\tag{21}
$$

We use $p = 5$ to analyze the replay completion tail below.

Figure 12 shows that CAAS shortens the tail interval relative to FCFS by 19.0% at $C _ { E } = 3$ and 21.4% at $C _ { E } ~ = ~ 5$ . Although CAAS reaches 95% completion later, it finishes the batch earlier. Prioritizing jobs from trajectories with greater revealed depth can delay other trajectories, but allows subsequent stages of deeper chains to execute earlier, shortening the final completion tail. This behavior supports its design goal of exposing and advancing later stages of long execution chains before they become batch-ending bottlenecks, using only trajectory progress observed during the current run. These gains come from changing job dispatch order, not service times.

![](images/4b673b206c7c78612990f33e55fd1bcd8b24af5bd17f707703eae6880b0de811.jpg)

![](images/0a513464370d148531f801f58a25a801af8fbd057e9a9b2cf9083c8393c8bf79.jpg)  
Figure 12 Completion of the final 5% of trajectories at $C _ { S } = 5 0$ . Both policies use the same absolute replay time axis within each panel. The final-5% tail duration is denoted by L .

## 5.4 End-to-End Validation

We conduct live evaluations on BFCL, SWE, WideSearch, and MCP-Atlas using Ling-3.0-Flash (124B total and 5.1B active parameters) (Ling Team, 2026a) and Ling-3.1-Flash (560B total and 25B active parameters) (Ling Team, 2026b). We omit BFCL-v4 for Ling-3.1-Flash, as scores are nearly saturated and we no longer use this benchmark in model development. The two models use two TP4 workers on eight H20 GPUs and two TP8 workers on sixteen H20 GPUs, respectively. Within each model–workload comparison, we fix the checkpoint, task set, sampling parameters, and hardware, varying only cache allocation, sample concurrency, inference scheduling and routing, and, for MCP-Atlas, capacity separation and environment-job scheduling. Live makespan includes all samples and retry time, with no exclusion of samples that are retried or ultimately fail. For BFCL, SWE, and WideSearch, we tune sample concurrency for the original baselines with the default cache ratio, FCFS scheduling, and cache-aware routing fixed, targeting GPU saturation without substantial inference queueing. The optimized configuration combines cache allocation and sample concurrency settings guided by the workload-aware capacity model with RP scheduling and LBCH routing. For each model, SWE includes a Baseline-2 configuration that uses the same sample concurrency as the optimized configuration while retaining the baseline cache allocation and inference policies. Detailed configurations and operating points are provided in Appendix A.3 and Table A2.

Table 2 Live inference-side results. Each metric is calculated per run and averaged equally over three runs per configuration. Input token counts include cached tokens. Token totals are in millions. Imbalance metrics follow Section 5.2.3.
<table><tr><td></td><td></td><td></td><td>Makespan</td><td colspan="2">Tokens (M)</td><td>Reuse</td><td colspan="2">Imbalance (%)</td><td></td></tr><tr><td>Model</td><td>Workload</td><td>Config.</td><td>(min)</td><td>Input</td><td>Output</td><td>H (%)</td><td>β</td><td>Yout</td><td>Score</td></tr><tr><td>3.0</td><td>BFCL</td><td>Baseline</td><td>68.4</td><td>141.1</td><td>3.9</td><td>84.2</td><td>10.6</td><td>2.9</td><td>73.0</td></tr><tr><td></td><td></td><td>Opt</td><td>49.2</td><td>141.6</td><td>3.9</td><td>84.5</td><td>17.0</td><td>3.1</td><td>75.65</td></tr><tr><td></td><td>SWE</td><td>Baseline-1</td><td>166.9</td><td>1286.2</td><td>8.5</td><td>98.3</td><td>17.2</td><td>13.0</td><td>72.4</td></tr><tr><td></td><td></td><td>Baseline-2</td><td>197.7</td><td>1371.4</td><td>8.7</td><td>87.5</td><td>17.2</td><td>7.7</td><td>71.3</td></tr><tr><td></td><td></td><td>Opt</td><td>149.5</td><td>1347.9</td><td>8.7</td><td>98.6</td><td>16.4</td><td>3.5</td><td>72.6</td></tr><tr><td></td><td>WideSearch</td><td>Baseline</td><td>186.7</td><td>261.9</td><td>10.9</td><td>79.9</td><td>11.5</td><td>1.4</td><td>73.6</td></tr><tr><td></td><td></td><td>Opt</td><td>116.7</td><td>311.1</td><td>10.0</td><td>81.7</td><td>15.6</td><td>4.6</td><td>72.7</td></tr><tr><td>3.1</td><td>SWE</td><td>Baseline-1</td><td>234.1</td><td>1073.3</td><td>11.1</td><td>98.2</td><td>15.1</td><td>11.6</td><td>87.16</td></tr><tr><td></td><td></td><td>Baseline-2</td><td>304.1</td><td>1102.8</td><td>11.5</td><td>80.4</td><td>15.9</td><td>12.0</td><td>86.0</td></tr><tr><td></td><td></td><td>Opt</td><td>185.1</td><td>1045.7</td><td>11.0</td><td>98.3</td><td>16.2</td><td>3.3</td><td>88.33</td></tr><tr><td></td><td>WideSearch</td><td>Baseline</td><td>459.9</td><td>415.7</td><td>25.0</td><td>83.2</td><td>8.8</td><td>1.7</td><td>78.6</td></tr><tr><td></td><td></td><td>Opt</td><td>359.1</td><td>430.0</td><td>25.1</td><td>80.7</td><td>17.6</td><td>2.5</td><td>78.1</td></tr></table>

Table 2 reports mean makespan, token counts, cache reuse, worker-load imbalance, and evaluation scores over three runs per configuration for the first three workloads, where environment-execution capacity is sufficient and does not constrain sample admission. For Ling-3.0-Flash, the optimized configuration reduces mean makespan by 28.1% on BFCL, 10.4% on SWE, and 37.5% on WideSearch relative to their original baselines. For Ling-3.1-Flash, the corresponding reductions are 20.9% on SWE and 21.9% on WideSearch. These results demonstrate that the proposed optimizations scale effectively to larger models while still achieving substantial makespan reductions. Stochastic model sampling and environment variability cause trajectories and token counts to differ across live runs. The makespan reductions, even with higher input-token volumes on some workloads, support the effectiveness of the optimizations under realistic execution variability.

On BFCL and WideSearch, optimized runs achieve shorter makespans despite higher worker-load imbalance, while maintaining comparable cache reuse. These results suggest a more effective trade off among concurrency, cache reuse, and worker-load balance, where the benefits of sustaining cache reuse at higher concurrency can outweigh a slight deterioration in worker-load balance. The SWE comparison further shows that increased concurrency alone does not explain these gains. For Ling-3.0-Flash, cache reuse falls from 98.3% in Baseline-1 to 87.5% in the higher-concurrency Baseline-2. This decline is consistent with increased cache pressure and eviction, leading to additional recomputation that can offset parallelism gains. At the same concurrency as Baseline-2, however, the optimized configuration maintains 98.6% reuse and reduces makespan by 24.4%. Ling-3.1-Flash shows a similar pattern: with the same sample concurrency, Opt maintains 98.3% cache reuse versus Baseline-2’s 80.4% and reduces makespan by 39.1%. Consistent with the controlled experiments in Sections 5.2.1–5.2.3, these results demonstrate the benefits of coordinated cache allocation, scheduling, and routing in preserving cache reuse at higher concurrency and reducing batch makespan. The latency measurements further clarify how these batch-level gains relate to individual request performance. Compared with the concurrency-matched Baseline-2, Opt reduces TTFT by 24.9% and 50.2%, and TPOT by 51.1% and 40.2%, for Ling-3.0-Flash and Ling-3.1- Flash, respectively. Relative to Baseline-1, Opt uses higher concurrency, increasing competition for inference resources; this is consistent with higher per-request TTFT and TPOT despite shorter batch makespan, reflecting the distinction between batch efficiency and per-request latency. Appendix A.3 provides a detailed analysis of the SWE results.

Table 3 Live MCP-Atlas results, averaged equally over three 500-task runs per configuration. Tail durations $L _ { 5 }$ and $L _ { 1 0 }$ are measured from each run’s own 95% and 90% completion points to its final task termination, including both successful and failed tasks. Mean in-flight is the time average of recorded inference calls over the full task-execution window, computed from native trajectory start/end timestamps and averaged equally across runs.
<table><tr><td colspan="3"></td><td rowspan="2">Mean</td><td colspan="3"></td></tr><tr><td>Model</td><td>Configuration</td><td>Makespan (min)</td><td>in-flight</td><td> $L _ { 5 } ( \mathrm { m i n } )$ </td><td> $L _ { 1 0 } ( \mathrm { m i n } )$  Score</td></tr><tr><td rowspan="3">3.0</td><td>Baseline</td><td>466.1</td><td>1.2</td><td>38.9</td><td>73.7</td><td>65.5</td></tr><tr><td>Opt-FCFS</td><td>168.2</td><td>7.3</td><td>67.6</td><td>79.2</td><td>66.6</td></tr><tr><td>Opt-CAAS</td><td>163.0</td><td>7.9</td><td>66.0</td><td>73.4</td><td>65.6</td></tr><tr><td rowspan="3">3.1</td><td>Baseline</td><td>333.5</td><td>1.6</td><td>26.1</td><td>43.5</td><td>81.0</td></tr><tr><td>Opt-FCFS</td><td>140.5</td><td>12.5</td><td>49.8</td><td>54.4</td><td>80.9</td></tr><tr><td>Opt-CAAS</td><td>128.5</td><td>11.5</td><td>35.3</td><td>38.6</td><td>81.3</td></tr></table>

For MCP-Atlas, Table 3 reports mean results over three 500-task runs per configuration, evaluating the integrated inference and environment optimizations under constrained environment capacity. Relative to the unified-admission baseline, inference-side optimizations and capacity separation (Opt-FCFS) reduce mean makespan from 466.1 to 168.2 minutes (63.9%) for Ling-3.0-Flash and from 333.5 to 140.5 minutes (57.9%) for Ling-3.1-Flash, with environment capacity unchanged for each model. Mean in-flight inference calls increase from 1.2 to 7.3 for Ling-3.0-Flash and from 1.6 to 12.5 for Ling-3.1-Flash, consistent with more inference work being available while other trajectories await environment interaction.

Compared with Opt-FCFS, Opt-CAAS further reduces mean makespan by 3.1% for Ling-3.0-Flash and 8.5% for Ling-3.1-Flash. Using the tail durations defined in Equation 21, the corresponding reductions in $L _ { 5 }$ and $L _ { 1 0 }$ are 2.4% and 7.3% for Ling-3.0-Flash, and 29.1% and 29.0% for Ling-3.1- Flash. These live results are consistent with the benefits of CAAS observed in the controlled replay experiments in Section 5.3, although live trajectories vary across runs.

Although Opt-CAAS completes the batch much earlier, its final 5% takes longer than the baseline’s. This apparent discrepancy can be explained by differences in the tasks that finish last. Under limited admission, the baseline’s final tasks are largely those admitted last, rather than necessarily those with the longest trajectories. Broader admission lets shorter trajectories finish while longer ones continue, so the optimized tail is more concentrated in long trajectories. The final 25 tasks under Opt-CAAS average 27.9 and 15.6 turns for Ling-3.0-Flash and Ling-3.1-Flash, respectively, compared with 10.1 and 7.9 under the baseline. When matched by task ID, the Opt-CAAS tail tasks also require more turns in baseline runs. This difference helps explain why Opt-CAAS can finish the batch much earlier while taking longer to complete its final $5 \% ,$ so $L _ { 5 }$ alone should not be used to compare overall execution efficiency across different admission policies.

Across the evaluated model–benchmark combinations, evaluation scores remain aligned between the baseline and optimized configurations, indicating that the efficiency gains do not come at the expense of response quality.

## 6 Discussion and Future Work

Although this work focuses on batch makespan in LLM training and development workflows, its underlying principles may extend to other agent workloads. For example, in personal-agent services, workload-aware cache allocation, RP scheduling, and LBCH routing could preserve cross-turn reuse while supporting concurrent users. These mechanisms could reduce prefix recomputation and shorten the prefill component of TTFT. Reducing redundant prefill may also help reduce TPOT by leaving more capacity for ongoing decoding. Separating sample admission from environment-execution capacity could let ready trajectories use GPUs while others wait for tools or external services. In workflows with parallel subagents, the chain-advancing principle behind CAAS could inform tool-job scheduling to advance subagents that delay downstream aggregation or overall task completion.

Many directions remain for improving agent trajectory execution efficiency and further reducing makespan. These include extending workload-aware cache allocation and trajectory-aware scheduling to hierarchical caching while accounting for cache capacity and transfer costs, and investigating cache eviction policies for hybrid models in agent workloads. Another important direction is makespan optimization subject to application service-level objectives (SLOs), such as TTFT and TPOT targets, which is relevant to datacenters seeking to improve task throughput while meeting user-facing latency requirements.

## 7 Conclusion

We present coordinated inference and environment optimizations to reduce batch makespan for multi-turn agents served by hybrid LLMs. A workload-aware capacity model guides cache allocation and sample concurrency, while RP and LBCH promote timely cross-turn reuse at low scheduling cost and balance trajectory affinity with worker load. Under constrained environment resources, capacity separation improves inference–environment overlap, and CAAS advances long execution chains to shorten the batch tail. Controlled replay evaluates individual mechanisms, while live experiments with Ling-3.0-Flash and Ling-3.1-Flash demonstrate batch makespan reductions of 10.4%–65.0% across four well-established benchmarks without compromising model performance. Their successful application in developing the Ling-3.x model series demonstrates the practical value of these optimizations in improving model-development efficiency.

## References

C. Bandi et al. MCP-Atlas: A Large-Scale Benchmark for Tool-Use Competency with Real MCP Servers. arXiv:2602.00933, 2026. Code: https://github.com/scaleapi/mcp-atlas.

D. Guo, J. Wu, and S.-M. Yiu. SAGA: Workflow-Atomic Scheduling for AI Agent Inference on GPU Clusters. arXiv:2605.00528, 2026.

Kimi Team. Kimi K3: Open Frontier Intelligence. arXiv:2607.24653, 2026.

W. Kwon et al. Efficient Memory Management for Large Language Model Serving with PagedAttention. SOSP, 2023.

J. Li, T. Wang, and Y. Chen. From Tensor Buffer to Distributed Memory Hierarchy: A Survey of KV Cache Management for LLM Serving. arXiv:2607.02574, 2026.

Ling Team. Ling-3.0-flash: Intelligence through Efficiency. Technical blog, July 24, 2026. https://developer.ant-ling.com/zh-CN/blogs/ling-3.0-flash-release. Official model card, 2026a. https://huggingface.co/inclusionAI/Ling-3.0-flash.

M. Luo et al. Autellix: An Efficient Serving Engine for LLM Agents as General Programs. arXiv:2502.13965, 2025.

A. X. Nguyen. Asymmetric Virtual Memory Paging for Hybrid Mamba-Transformer Inference. arXiv:2605.22416, 2026.

S. Nian et al. CacheFlow: Efficient LLM Serving with 3D-Parallel KV Cache Restoration. arXiv, 2026.

R. Pan et al. Marconi: Prefix Caching for the Era of Hybrid LLMs. MLSys, 2025.

S. G. Patil et al. The Berkeley Function Calling Leaderboard (BFCL): From Tool Use to Agentic Evaluation of Large Language Models. ICML, 2025. BFCL-v4: https://gorilla.cs.berkeley.edu/leaderboard.html.

R. Qin et al. Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving. arXiv:2407.00079, 2024.

S. Qiu et al. Tutti: Making SSD-Backed KV Cache Practical for Long-Context LLM Serving. arXiv:2605.03375, 2026.

Qwen Team. Qwen3.8-2.4T-A95B. Official model card, 2026. https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B.

Q. Shi et al. τ-Knowledge: Evaluating Conversational Agents over Unstructured Knowledge. arXiv:2603.04370, 2026. Code: https://github.com/sierra-research/tau2-bench.

V. Srivatsa et al. Preble: Efficient Distributed Prompt Scheduling for LLM Serving. arXiv:2407.00023, 2024.

vLLM Project. Automatic Prefix Caching. vLLM Documentation. https://docs.vllm.ai/en/stable/design/prefix\_caching/.

N. Wadlom, J. Shen, and Y. Lu. Efficient LLM Serving for Agentic Workflows: A Data Systems Perspective. arXiv:2603.16104, 2026.

T. Xia et al. Idleness is Relative: Exploiting Tool-Call Idle Windows for Offloading in Agentic Systems with MORI. arXiv:2606.00866, 2026.

B. Xiao et al. ARL-Tangram: Unleash the Resource Efficiency in Agentic Reinforcement Learning. arXiv:2603.13019, 2026.

B. Xie et al. PEEK: Predictive Queue-Informed KV Cache Management for LLM Serving. arXiv:2607.02525, 2026.

Y. Yuan et al. DualMap: Enabling Both Cache Affinity and Load Balancing for Distributed LLM Serving. arXiv:2602.06502, 2026.

L. Zheng et al. SGLang: Efficient Execution of Structured Language Model Programs. NeurIPS, 2024.

J. Ding et al. Observation, Not Prediction: Conversation-Level Disaggregated Scheduling for Agentic Serving. arXiv:2606.01839, 2026.

R. Wong et al. WideSearch: Benchmarking Agentic Broad Info-Seeking. arXiv:2508.07999, 2025.

V. Mirrokni, M. Thorup, and M. Zadimoghaddam. Consistent Hashing with Bounded Loads. arXiv:1608.01350, 2016.

K. Khandpur, K. Lieret, C. E. Jimenez, O. Press, and J. Yang. SWE-bench Multilingual. Official benchmark description, 2025. https://www.swebench.com/multilingual.html.

Ling Team. Ling-3.1-flash. Official model card, 2026b. https://huggingface.co/inclusionAI/Ling-3.1-flash.

Z. Zheng et al. BatchLLM: Optimizing Large Batched LLM Inference with Global Prefix Sharing and Throughput-oriented Token Batching. MLSys, 2026.

## A Experimental Configuration Details

## A.1 Inference-Side Experiments

Table A1 lists the inference-only workloads. The SWE replay set contains 289 trajectories because only successful historical executions were collected; failed executions were excluded. Selected SWE trajectories have long contexts and deep chains; BFCL lengths vary substantially and include single-turn trajectories to evaluate our methods on short trajectories; WideSearch repeatedly invokes inference for broad information seeking (Khandpur et al., 2025; Patil et al., 2025; Wong et al., 2025). Replay measures execution efficiency for these recorded workloads; it does not evaluate semantic correctness or cover all failures and retries.

Table A1 Historical inference-only replay workloads. Output totals are from the routing replay runs. Live task counts are reported separately in Table A2.
<table><tr><td>Workload</td><td>Recorded trajectories</td><td>Inference requests</td><td>Output tokens</td></tr><tr><td>SWE</td><td>289</td><td>25,416</td><td>6,000,784</td></tr><tr><td>BFCL</td><td>5,106</td><td>17,029</td><td>1,441,547</td></tr><tr><td>WideSearch</td><td>200</td><td>5,672</td><td>4,736,147</td></tr></table>

Cache allocation and admission. The experiments in Section 5.2.1 use one TP4 worker with FCFS scheduling.   
The cache is flushed before each timed run, and complete-batch makespan excludes deployment and warmup.   
The tested cache ratios and sample-concurrency limits are shown in Figure 3; TO denotes a six-hour timeout.

In-worker scheduling. In Section 5.2.2, the cache ratio is fixed at the model-derived balanced allocation: 0.09 for SWE, 2.26 for BFCL, and 0.10 for WideSearch. LPM and DFS-weight use SGLang’s default policy parameters. Figure 6 reports the tested sample-concurrency limits; the SWE FCFS run at the largest effective concurrency of 289 is marked as a timeout under the 13-hour limit.

Cross-worker routing. The replay comparison in Section 5.2.3 uses two TP4 workers with routing-prefix scheduling. The fixed $( r , C _ { S } )$ values are (0.09, 196) for SWE, (2.26, 2000) for BFCL, and (0.10, 200) for WideSearch. CA and CH use their native default parameters, and LBCH uses $\alpha = 1 . 2 5$ . Request imbalance is averaged over the intervals with more than one in-flight request across both workers, as defined in Equation 17. Output imbalance uses the generated token totals assigned to each worker.

## A.2 Inference–Environment Coordination

Environment-capacity settings. MCP-Atlas (Bandi et al., 2026) tasks depend on external MCP services, which may degrade under high concurrency or encounter upstream rate limits such as HTTP 429 errors. The runner configuration referenced for these experiments defaulted to five concurrent tasks.<sup>1</sup> This motivates evaluating constrained environment-execution settings, rather than assuming that environment capacity grows with inference capacity. We report results at $C _ { E } \in \{ 3 , 5 \}$

Replay configuration. Section 5.3 uses 495 successful historical trajectories with recorded environment responses and job service times. Two TP4 workers use a cache ratio of $r = 0 . 1 4 _ { \cdot }$ , routing-prefix scheduling, and LBCH routing. CAAS uses a time-band width of B = 30 seconds and an inference-duration bucket width of $\Delta _ { I } = 1$ second. Compared replay runs keep the model, hardware, and recorded workload fixed.

## A.3 Live Execution Settings

Both models enable thinking mode and use a maximum context length of 256K tokens in the live experiments.   
Table A2 summarizes the task sets and operating points.

For MCP-Atlas, the baseline uses unified admission with $C _ { S } = C _ { E } = 3$ . Both Opt variants use the same inference configuration and decoupled admission at $C _ { S } = 5 0$ while retaining $C _ { E } = 3$ . They differ only in FCFS versus CAAS environment dispatch.

For the live evaluations in Section 5.4, to use the same load-balance criteria for CA and LBCH, CA sets balance\_abs\_threshold=0 and balance\_rel\_threshold=1.6666667, while LBCH uses $\alpha = 1 . 2 5$ . Live CAAS uses the same parameters as the replay experiments: B = 30 seconds and $\Delta _ { I } = 1$ second.

Table A2 Live task sets and operating points. All baseline cache ratios are 0.9. B1 and B2 denote SWE Baseline-1 and Baseline-2. The two MCP-Atlas Opt variants share the listed settings. A dash in $C _ { E }$ indicates that environment execution does not constrain sample admission.
<table><tr><td></td><td></td><td></td><td colspan="3">Sample concurrency  $C _ { S }$ </td><td></td><td></td></tr><tr><td>Model</td><td>Workload</td><td>Tasks</td><td>Baseline/B1</td><td>B2</td><td>Opt</td><td>Opt ratio</td><td> $C _ { E }$ </td></tr><tr><td>3.0</td><td>BFCL</td><td>5,106</td><td>100</td><td></td><td>300</td><td>1.5</td><td>一</td></tr><tr><td></td><td>SWE</td><td>300</td><td>100</td><td>250</td><td>250</td><td>0.1</td><td>一</td></tr><tr><td></td><td>WideSearch</td><td>200</td><td>40</td><td>一</td><td>100</td><td>0.25</td><td>一</td></tr><tr><td></td><td>MCP-Atlas</td><td>500</td><td>3</td><td>一</td><td>50</td><td>0.14</td><td>3</td></tr><tr><td>3.1</td><td>SWE</td><td>300</td><td>50</td><td>100</td><td>100</td><td>0.2</td><td>一</td></tr><tr><td></td><td>WideSearch</td><td>200</td><td>15</td><td>一</td><td>40</td><td>0.27</td><td>一</td></tr><tr><td></td><td>MCP-Atlas</td><td>500</td><td>3</td><td>一</td><td>50</td><td>0.34</td><td>3</td></tr></table>

Live WideSearch also uses the evaluated model for summarization, which can introduce additional uncached input and help explain its lower aggregate cache reuse relative to replay.

Additional SWE diagnostics. Table A3 provides a closer comparison of the tuned baseline, concurrencymatched baseline, and Opt.

Table A3 Live SWE diagnostics under the tuned baseline, concurrency-matched baseline, and Opt. New-prefill counts uncached input tokens. TTFT and TPOT are averaged across requests within each run, then across runs. Mean API in-flight averages outstanding API requests, including those waiting, over each live run.
<table><tr><td></td><td colspan="3">Ling-3.0-Flash</td><td colspan="3">Ling-3.1-Flash</td></tr><tr><td>Metric</td><td>Baseline-1</td><td>Baseline-2</td><td>Opt</td><td>Baseline-1</td><td>Baseline-2</td><td>Opt</td></tr><tr><td>Sample concurrency</td><td>100</td><td>250</td><td>250</td><td>50</td><td>100</td><td>100</td></tr><tr><td>Makespan (min)</td><td>166.9</td><td>197.7</td><td>149.5</td><td>234.1</td><td>304.1</td><td>185.1</td></tr><tr><td>Cache reuse (%)</td><td>98.3</td><td>87.46</td><td>98.58</td><td>98.2</td><td>80.41</td><td>98.27</td></tr><tr><td>New-prefill (M)</td><td>22.2</td><td>170.90</td><td>18.98</td><td>19.3</td><td>211.81</td><td>17.93</td></tr><tr><td>Mean API in-flight</td><td>40.9</td><td>105.2</td><td>76.7</td><td>35.5</td><td>71.0</td><td>63.7</td></tr><tr><td>TTFT (s)</td><td>2.90</td><td>10.08</td><td>7.57</td><td>2.54</td><td>11.81</td><td>5.88</td></tr><tr><td>TPOT (ms/token)</td><td>28.6</td><td>85.55</td><td>41.81</td><td>24.3</td><td>60.84</td><td>36.37</td></tr></table>

We also include Baseline-LowConcurrency as a reference configuration. Baseline-LowConcurrency uses $C _ { S } = 2 0$ with the default cache ratio, FCFS scheduling, and cache-aware routing. Its live SWE runs yield mean makespans of 208.1 and 263.0 minutes for Ling-3.0-Flash and Ling-3.1-Flash, respectively, over three runs per configuration. Tuning sample concurrency under this baseline configuration yields Baseline-1 at $C _ { S } = 1 0 0$ and 50, respectively, reducing makespan to 166.9 and 234.1 minutes (19.8% and 11.0%) while maintaining cache reuse above 98%. This is consistent with the concurrency benefits observed in Section 5.2.1.

Increasing sample concurrency from Baseline-1 to Baseline-2 lowers cache reuse and increases new-prefill work, request latency, and makespan for both models. At the same sample concurrency as Baseline-2, Opt restores cache reuse to above 98% and reduces new-prefill tokens by 88.9% and 91.5%, respectively. TTFT falls by 24.9% and 50.2%, and TPOT falls by 51.1% and 40.2%. Together with makespan reductions of 24.4% and 39.1%, these results show that Opt maintains high cache reuse at high concurrency, benefiting from both parallelism and cache reuse to shorten makespan.

At the same sample concurrency as Baseline-2, Opt has fewer mean in-flight requests, lower TTFT and TPOT, and shorter makespan. Together, these observations are consistent with faster request completion and task progress. Relative to Baseline-1, Opt uses higher concurrency, increasing competition for inference resources; this is consistent with higher per-request TTFT and TPOT despite shorter batch makespan, distinguishing batch efficiency from per-request latency.