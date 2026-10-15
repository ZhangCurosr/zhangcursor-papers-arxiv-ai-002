# SCALING LLM INFERENCE VIA PARALLELISM: THROUGHPUT-LATENCY TRADE-OFF

Javad Mirzaei<sup>∗</sup>   
Office of CTO   
Dell Technologies Inc.   
Canada   
javad.mirzaei@dell.com   
Jeebak Mitra   
Office of CTO   
Dell Technologies Inc.   
Canada   
jeebak.mitra@dell.com

## ABSTRACT

Large Language Model (LLM) serving systems must balance latency and throughput while efficiently utilizing distributed GPU resources. This challenge arises because inference consists of two fundamentally different phases: prefill, which determines time-to-first-token (TTFT), and decode, which determines inter-token latency (ITL). The use of tensor parallelism (TP), pipeline parallelism (PP), and hybrid configurations exposes different trade-offs between these latency dimensions, mak ing the optimal parallelism configuration highly dependent on workload characteristics, latency requirements, and context length. Moreover, despite the widespread adoption of distributed inference, there is still limited understanding of how parallelism configurations should be selected under varying workload conditions and service-level objectives (SLO). In this paper, we develop a systematic framework for characterizing LLM inference parallelism via throughput-latency trade-offs, SLO-aware serving, and long-context scalability. We introduce the notion of parallelism operating regions by constructing TTFT-ITL feasible sets and Pareto frontiers across different workload compositions, thereby revealing a systematic transition from TP-dominant to PP-dominant configurations as workloads become increasingly prefill-heavy. We further formulate SLO-aware parallelism selection using goodput, showing that stringent TTFT requirements favor PP-dominated configurations, whereas relaxed latency constraints progressively shift the optimality toward TP-dominated configurations with higher serving capacity. Finally, we demonstrate that increasing context length fundamentally changes the relative advantages of parallelism configurations, with PP-dominant con figurations exhibiting more favorable scalability and SLO attainment in the long-context regime.

## 1 Introduction

Large language models (LLMs) have become a foundational component of modern AI applications, powering conversational assistants, code-generation systems, retrieval-augmented generation workflows, and increasingly complex agentic pipelines. As these applications move from offline processing to interactive services, inference efficiency has become a primary serving challenge. Modern LLM serving systems must simultaneously satisfy stringent latency requirements, sustain high request throughput, and efficiently utilize expensive GPU resources. Achieving these objectives is particularly difficult because autoregressive inference consists of two fundamentally different phases: prefill, which processes the input context in parallel, and decode, which generates output tokens autoregressively. These phases exhibit distinct computational, memory, and communication characteristics, which creates an inherent tradeoff between time-to-first-token (TTFT) and inter-token latency (ITL) [1, 2]. Consequently, evaluating serving systems using only a single latency or throughput metric is often insufficient to characterize real-world performance. Instead, systems increasingly consider service-level objectives (SLO) and goodput<sup>2</sup> as serving objectives [3, 4].

The challenge is further compounded by the diversity of LLM workloads. Requests can vary substantially in input length, output length, concurrency, and latency requirements, causing different phases of the inference pipeline to become dominant under different operating conditions [5, 6]. There is no single parallelization configuration that is universally optimal across serving scenarios. Tensor parallelism (TP), pipeline parallelism (PP), and their hybrid configurations impose different computation, synchronization, communication, and memory trade-offs, and their efficacy depends strongly on model size, sequence length, hardware topology, and workload composition [7–9]. As a result, the use of parallelism should not be viewed as a static universal deployment decision; rather, it is a workloaddependent operating point that balances competing latency and throughput objectives. Dynamic reconfiguration and parallelism-switching approaches have therefore been proposed to adapt execution strategies as workload conditions change [10–12].

A second challenge stems from the SLO constraints encountered in production deployments. In production environments, maximizing raw throughput is often insufficient because only requests that satisfy their latency requirements constitute useful service [2, 13]. Moreover, the relative importance of TTFT and throughput depends on application requirements. For example, interactive applications generally demand fast responsiveness, whereas batch or offline workloads may prioritize overall serving capacity and cost efficiency [4, 14]. The problem becomes even more pronounced for long-context inference, where increasing context length increases prefill computation, communication overhead, key-value (KV) cache memory consumption, and scheduling complexity [15, 16]. Under these conditions, the preferred parallelism and scheduling strategy can change dramatically as sequence lengths, workload heterogene ity, and latency requirements evolve, thereby making static parallelism configuration choices inefficient [10, 17].

In this paper, we develop a systematic framework to understand and select parallelism configuration for LLM inference through three complementary perspectives: latency trade-offs, SLO-aware serving, and long-context scalability. First, we introduce the notion of a parallelism operating region by constructing TTFT-ITL feasible sets and Pareto frontiers across different workload compositions, revealing a systematic transition from heavy TP-dominant to PP-dominant configurations as workloads move from decode-heavy to prefill-heavy regimes. Second, we formulate SLO-aware parallelism selection, which enables parallelism configurations to be evaluated according to both serving capacity and latency compliance. We show that the optimal configuration shifts from PP-dominant to TP-dominant strategies as TTFT requirements are relaxed. Finally, we extend the analysis to long-context serving and show that sequencelength growth fundamentally changes the relative advantages of different parallelism strategies, increasingly favoring PP-dominant configurations under stringent latency constraints. Together, these results provide a unified view of how workload characteristics, latency requirements, and context length jointly determine the optimal operating region for LLM inference parallelism, thereby offering practical guidance for deploying and adapting serving systems across diverse application scenarios.

## 1.1 Related Works

Latency-oriented LLM serving: Recent works on LLM serving have shown that throughput alone is insufficient to characterize system performance, mainly because TTFT and ITL depend on distinct phases of inference and exhibit different sensitivities to system design choices. During prefill, computation scales with input sequence length, whereas decoding proceeds iteratively while repeatedly accessing a growing KV cache. Consequently, configurations that maximize token throughput may still deliver poor interactive performance. Recently, attention has shifted toward latency-aware metrics. Sarathi-Serve [1] examines the trade-off among batching, throughput, and streaming latency, while DistServe [2] introduces goodput as the throughput of requests that satisfy latency requirements. This perspective is further reinforced by holistic evaluation frameworks in [18, 19], which show that prompt length, output length, request arrival patterns, and hardware topology jointly determine performance. As a result, modern LLM serving systems are increasingly evaluated in terms of their ability to satisfy TTFT and ITL constraints rather than optimizing a single latency or throughput metric.

The separation between prefill and decode phases has also motivated architectural innovations that explicitly address resource utilization during inference. When long prefill computations and latency-sensitive decoding requests share the same resources, contention can simultaneously degrade TTFT and ITL. To mitigate this effect, Nexus [20] ex plores proactive prefill-decode disaggregation and chunked-prefill scheduling, while [3] studies the conditions where aggregation or disaggregation yields superior goodput. These observations suggest that the efficacy of any parallelism configuration depends not only on its computational efficiency but also on how resources are allocated between pre fill and decode workloads. Consequently, the choice of a parallelism configuration is closely coupled with broader serving-system architectural decisions.

To scale increasingly large models and long-context workloads, existing systems employ combinations of data parallelism (DP), TP, and PP. Hybrid approaches are often necessary because model weights, activations, KV caches, computation, and communication scale differently across hardware platforms and workloads [7]. DP provides high request-level concurrency and benefits from efficient KV-cache management techniques such as PagedAttention [21], while TP enables model execution beyond the memory limits of a single accelerator and has motivated adaptive mechanisms for workload-dependent reconfiguration [11,14]. PP further expands model capacity but introduces additional pipeline scheduling and load-balancing challenges, particularly for long-context workloads [15]. More recently, dynamic approaches such as on-the-fly parallelism switching [12] have emphasized that no single parallelism configuration is universally optimal. Instead, the appropriate configuration depends on workload characteristics, latency objectives, and changing demand conditions.

SLO-aware serving: Moving from throughput-centric evaluation toward TTFT and/or ITL constrained serving, recent work has increasingly adopted goodput as the primary optimization objective. Unlike raw throughput, goodput measures only the requests that satisfy their latency requirements, thereby directly reflecting user-perceived quality of service. DistServe [2] demonstrates how architectural decisions such as prefill-decode disaggregation can significantly improve goodput under latency constraints, while Cascade [13] extends this perspective through SLO-aware resource allocation for heterogeneous workloads. Together, these studies emphasize that maximizing token processing rates alone is insufficient, and effective serving systems must balance resource utilization against the ability to consistently meet latency targets across diverse request classes.

A recurring theme in the literature is the close interaction between parallelism configuration and request scheduling. Parallelism techniques determine the available compute capacity, communication overhead, and memory footprint, whereas scheduling policies decide how these resources are allocated to incoming requests. Prior work has shown that scheduling decisions strongly influence latency outcomes, especially in the presence of long input prompts and mixed workloads. To address this challenge, Sarathi-Serve [1] introduced chunked prefill to prevent large prompts from monopolizing execution and to enable better interleaved decode operations, thereby improving responsiveness without sacrificing throughput. Nexus further advances this idea by exploring the scheduling granularity of prefill workloads in shared-resource environments [20]. These works collectively suggest that achieving high goodput requires coordinated optimization of both system configuration and runtime scheduling rather than addressing either problems in isolation.

The challenge becomes significantly more pronounced in multi-SLO and heterogeneous serving environments, where requests exhibit different latency requirements, priorities, and resource demands. GoodServe [22] investi gates goodput-aware serving for agentic inference workloads spanning heterogeneous resources, while Nitsum [14], Niyama [23], and SLOs-Serve [4] explore adaptive parallelism, QoS-aware scheduling, and dynamic request routing to meet differentiated service objectives. Infinite-LLM [24] addresses the challenge of highly variable context lengths through attention-layer disaggregation. At the same time, uncertainty in output length introduces additional complex ity because future KV-cache consumption and decode time are unknown at admission. Jenga [25] addresses memory efficiency through a heterogeneity-aware allocation framework, while [26] highlights the trade-off between conservative memory reservation and aggressive admission policies under uncertain decode demand. These studies point toward a control framework where long-term resource allocation, parallelism selection, memory management, and dynamic scheduling decisions must be jointly optimized to maximize goodput while maintaining SLO compliance.

Serving for long context: As context lengths continue to grow, traditional model-parallel techniques alone become insufficient to sustain efficient LLM serving. Long-context workloads increase attention computation during prefill, expand KV-cache requirements, and impose higher memory-access costs during decoding, thereby affecting TTFT and ITL through different mechanisms. TTFT is primarily influenced by the cost of processing increasingly large input prompts, while ITL degrades as each decoding step repeatedly accesses a larger context state. Medha [27] highlights this challenge by targeting inference at multi-million-token context lengths, demonstrating that sequence state and KV-cache management can become the dominant scalability bottlenecks. To address this issue, recent studies have focused on context parallelism (CP), which distributes context tokens and associated state across multiple accelerators [16,28–33]. Adaptive CP techniques have been explored for production environments [16], while Untied Ulysses [32] improves memory efficiency through headwise chunking and PecSched [33] reduces communication overhead for sequence-parallel execution. Collectively, these studies suggest that CP is an effective mechanism when sequence length and KV-cache growth dominate resource consumption, although its communication costs can outweigh its benefits for shorter-context workloads.

Long-context serving also strengthens the interaction between scheduling policies, pipeline execution, and attention mechanisms. While chunked prefill has emerged as an effective mechanism for reducing interference and preserving decode responsiveness, the computational cost of successive chunks is not uniform because later chunks attend to progressively longer prefixes. The work in [15] addresses this challenge through virtual pipeline parallelism (VPP) that better balances chunked-prefill execution under long-context workloads, thereby linking pipeline scheduling directly to the TTFT-ITL trade-off discussed earlier. Beyond dense-attention models, recent work has further shown that the choice of attention mechanism can significantly influence parallelism configuration efficacy. SparseServe [34] demonstrates that dynamic sparse-attention mechanisms introduce irregular computation and communication patterns that require specialized scheduling and distribution policies to fully exploit available parallelism. Together, these works indicate that efficient long-context serving requires a holistic approach that jointly considers context parallelism, pipeline scheduling, memory management, and attention algorithms, extending the broader theme that optimal serving performance emerges from coordinated management of both system architecture and runtime execution policies.

## 2 Parallelism Operating Region

During inference, a request goes through prefill and decode phases, each with fundamentally different execution characteristics. In particular, during prefill, the input is processed in parallel, while during decode, output tokens are generated autoregressively. A key challenge is that these phases are not computationally homogeneous. When multiple GPUs are used, inter-GPU communication becomes an additional challenge along with computation overheads [35]. To illustrate this effect, we represent a parallelism configuration by

$$
p = ( N _ { \mathrm { T P } } , N _ { \mathrm { P P } } ) ,\tag{1}
$$

where $N _ { \mathrm { T P } }$ denotes the tensor-parallel degree and $N _ { \mathrm { P P } }$ the pipeline-parallel degree. Then, the per request total latency is given by

$$
T _ { \mathrm { r e q } } ^ { ( p ) } = T _ { \mathrm { p r e f i l l } } ^ { ( p ) } + T _ { \mathrm { d e c o d e } } ^ { ( p ) } ,\tag{2}
$$

where $T _ { \mathrm { p r e f i l l } } ^ { ( p ) }$ and $T _ { \mathrm { d e c o d e } } ^ { ( p ) }$ represent the prefill and decoding latency, respectively. Such decomposition immediately exposes a fundamental trade-off in LLM serving, where TTFT is primarily determined by the efficiency of prefill, whereas sustained token generation is determined by the efficiency of decoding. Consequently, optimizing a serving system for low TTFT does not necessarily produce the lowest per-token latency or highest token generation throughput. TP distributes the computation of each layer across $N _ { \mathrm { T P } }$ GPUs, reducing the computation assigned to each GPU, while introducing collective communication. PP instead distributes model layers across $N _ { \mathrm { P P } }$ consecutive stages, thereby reducing the amount of fine-grained inter-GPU synchronization, however, overheads related to pipeline scheduling and bubbles <sup>3</sup> are inevitable [35]. We can therefore express the latency of a given parallelism configuration as follows

$$
T _ { q } ^ { ( p ) } = \tau _ { q } ^ { ( p ) } + \eta _ { q } ^ { ( p ) } + \gamma _ { q } ^ { ( p ) } ,\tag{3}
$$

where $\tau _ { q } ^ { ( p ) } , \eta _ { q } ^ { ( p ) }$ , and $\gamma _ { q } ^ { \left( p \right) }$ represent computation, communication/synchronization, and scheduling overhead, respectively, assuming parallelism configuration $p ,$ and $q \in \{ \mathrm { p r e f i l l }$ , decode . Different parallelism configurations exhibit different interplay between the components in (3). For example, with TP, increasing $N _ { \mathrm { T P } }$ reduces the computation per GPU, while increasing the amount of inter-GPU communication. Thus, for a given phase $q ,$ the reduction in computation $( \mathrm { i } . \mathrm { e } . , \tau _ { q } ^ { ( p ) } )$ may outweigh the communication cost $( \mathrm { i } . \mathrm { e } . , \eta _ { q } ^ { ( p ) } )$ . As the computation per iteration becomes smaller, however, communication becomes increasingly visible in the critical path. This effect is particularly important during decoding, where each iteration produces only a small number of new tokens. TP can therefore provide strong decode efficiency while potentially paying a larger communication cost during prefill. PP exhibits a complementary behavior. By assigning contiguous layers to different pipeline stages, PP avoids much of the fine-grained collective communication of TP, which makes it attractive for compute-intensive execution of prefill. However, its decode efficiency can be limited by pipeline scheduling and bubble effects, particularly when all stages are not fully occupied [35]. To characterize such competing effects, we represent the workload by

$$
w = ( L _ { \mathrm { i n } } , L _ { \mathrm { o u t } } , C ) ,\tag{4}
$$

where $L _ { \mathrm { i n } }$ is the input sequence length (ISL), $L _ { \mathrm { o u t } }$ is the output sequence length (OSL), and $C$ is the request concurrency. We also define $T _ { \mathrm { T T F T } } ^ { ( p , w ) }$ and $T _ { \mathrm { I T L } } ^ { ( p , w ) }$ random variables representing TTFT and ITL of a request served under parallelism $p$ and workload w, respectively. For given serving policy $s ,$ model $\mathcal { M } .$ , hardware platform , parallelism configuration p, executing workload w produces a latency tuple<sup>4</sup>

$$
\ell ( p , w ) = ( \mathrm { T T F T } ^ { \mathrm { a v g } } ( p , w ) , \mathrm { I T L } ^ { \mathrm { a v g } } ( p , w ) ) ,\tag{5}
$$

where $\mathrm { T T F T } ^ { \mathrm { a v g } } ( p , w ) = \frac { } { } \mathrm { m e a n } ( T _ { \mathrm { T T F T } } ^ { ( p , w ) } ) ^ { 5 }$ captures the average TTFT across all serving requests, whereas $\mathrm { I T L } ^ { \mathrm { a v g } } ( p , w ) = \mathrm { m e a n } ( T _ { \mathrm { I T L } } ^ { ( p , w ) } )$ captures the average sustained latency of autoregressive token generation. The tuple in (5) provides two complementary views of inference latency. We deliberately retain these quantities as separate dimensions rather than combining them into a single metric, since their relative importance changes with the composition of the workload. For a request producing $L _ { \mathrm { o u t } }$ tokens, the total request latency is given by (2). Such decomposition highlights why a single latency measure can obscure important differences between parallelism configurations. In particular, for specific workload $w ,$ parallelism configuration $p _ { i }$ and $p _ { j }$ may have substantially different TTFT and ITL characteristics while producing similar total latency.

## 2.1 Latency Feasibility Set and Pareto Frontier

In order to characterize the latency feasibility set, we consider a constant total workload with fixed concurrency and total sequence length, denoted as $L _ { \mathrm { s e q } }$ . We define

$$
\mathcal { W } = \{ ( L _ { \mathrm { i n } } , L _ { \mathrm { o u t } } , C ) | L _ { \mathrm { i n } } + L _ { \mathrm { o u t } } = L _ { \mathrm { s e q } } \} ,\tag{6}
$$

as the set of feasible workloads, with

$$
\mathcal { P } = \{ ( N _ { \mathrm { T P } } , N _ { \mathrm { P P } } ) | N _ { \mathrm { T P } } N _ { \mathrm { P P } } = N \} ,\tag{7}
$$

representing the set of available parallelism configurations, where N denotes the total number of available GPUs. Note that fixing $L _ { \mathrm { s e q } }$ defines a particular workload scale for the analysis. We fix the total sequence length to define a consistent workload scale, so that comparisons across different $L _ { \mathrm { i n } }$ and $L _ { \mathrm { o u t } }$ values reflect changes in workload composition rather than changes in the overall request size. This allows us to construct the Pareto frontier for a well-defined workload family and isolate how the prefill and decode imbalance affects the latency trade-off. As we scale $L _ { \mathrm { s e q } } ,$ the underlying workload changes and a new latency feasible set and Pareto frontier must consequently be constructed. It is also worth mentioning that, changing input and output lengths independently changes multiple aspects of the workload simultaneously. For example, increasing $L _ { \mathrm { i n } } .$ , will increase the amount of prefill computation, while increasing $L _ { \mathrm { o u t } }$ will lead to more autoregressive decode iterations. If the total sequence length is also allowed to vary, differences in latency could instead be attributed simply to differences in the overall workload. By fixing $L _ { \mathrm { s e q } } ,$ the approximate total workload stays constant while changing its composition between prefill and decode. The corresponding workload-conditioned feasible set is therefore given by

$$
\mathcal { L } ( w , p ) = \{ \ell ( p , w ) | w \in \mathcal { W } , p \in \mathcal { P } \} .\tag{8}
$$

For each specific workload w, a Pareto frontier can be constructed independently. A configuration $p _ { i }$ dominates $p _ { j }$ under the workload w if

$$
\mathrm { T T F T } ^ { \mathrm { a v g } } ( p _ { i } , w ) \leq \mathrm { T T F T } ^ { \mathrm { a v g } } ( p _ { j } , w ) \quad \mathrm { a n d } \quad \mathrm { I T L } ^ { \mathrm { a v g } } ( p _ { i } , w ) \leq \mathrm { I T L } ^ { \mathrm { a v g } } ( p _ { j } , w ) ,\tag{9}
$$

with at least one inequality being strict. The Pareto-optimal configuration set is therefore

$$
\mathcal { P } ^ { * } ( w ) = \{ p \in \mathcal { P } | \nexists p ^ { ' } \mathrm { s . t . } p ^ { ' } \prec _ { w } p \} .\tag{10}
$$

The corresponding TTFT–ITL Pareto frontier is given by

$$
\mathcal { F } ( w ) = \{ \ell ( p , w ) \in \mathcal { L } ( w , p ) | p \in \mathcal { P } ^ { * } ( w ) \} .\tag{11}
$$

Note that $\mathcal { F } ( w )$ contains only those points for which no other configuration simultaneously provides lower TTFT and lower ITL. The frontier therefore represents the best achievable trade-offs between TTFT and ITL for that particular workload. The formulations in (8) and (11) provide a precise foundation for the notion of parallelism operating regions, i.e., the region of the workload space where different parallelism configurations occupy favorable positions on the latency frontier. This formulation further distinguishes two questions: (1) what TTFT–ITL combinations can be achieved, and (2) which configurations provide the best trade-offs between TTFT and ITL, such that improving one metric would require sacrificing the other. The first is captured by the feasible set $\mathcal { L } ( w )$ , while the second is captured by the Pareto frontier $\mathcal { F } ( w )$

For a given $L _ { \mathrm { s e q } }$ and concurrency, we vary the input/output composition, construct the feasible latency set for every resulting workload, and extract its Pareto frontier. Note that feasible set and its corresponding Pareto frontier are workload-dependent. Therefore, the objective is not to identify a single globally optimal parallelism configuration, but to characterize how the TTFT–ITL trade-off and its Pareto-optimal configurations evolve with the workload.

## 2.2 Experiment Results on Pareto Frontier

Here, we characterize the latency trade-off exposed by different parallelism configurations. For the purpose of evaluation, we employ Qwen2.5-14B-Instruct [36] with BF16 tensor type on 8 NVDIA H200 (141GB) [37]. All experiments are performed using vLLM v0.15.0 as an inference engine [38]. For the sake of descriptive notations throughout the experiments, we use the mapping in Table 1 for parallelism configuration across 8 GPUs

Table 1: Notation mapping.
<table><tr><td rowspan=1 colspan=1> $p$ </td><td rowspan=1 colspan=1>(8, 1)</td><td rowspan=1 colspan=1> $\overline { { ( 4 , 2 ) } }$ </td><td rowspan=1 colspan=1>(2,4)</td><td rowspan=1 colspan=1> $\overline { { ( 1 , 8 ) } }$ </td></tr><tr><td rowspan=1 colspan=1>Notation</td><td rowspan=1 colspan=1>TP8</td><td rowspan=1 colspan=1>TP4PP2</td><td rowspan=1 colspan=1>TP2PP4</td><td rowspan=1 colspan=1>PP8</td></tr></table>

![](images/453d789bfa4c9e657fdc06047aa50bbb3426bedfd52a5ba2a18b7e39e2ed9265.jpg)  
Figure 1: TTFT-ITL feasible region and Pareto frontier for different parallelism configurations. TP-dominant configurations achieve lower ITL but higher TTFT, while PP-dominant configurations achieve lower TTFT at the expense of higher ITL. As workloads transition from decode-heavy to prefill-heavy, the Pareto-optimal operating point shifts systematically from TP-dominant configurations, through hybrid configurations, toward PP-dominant configurations.

Fig. 1 depicts the latency feasible region and the corresponding Pareto frontier comparing different parallelism configurations. During inference, we consider batch size to be 128, maximum GPU memory utilization of 0.8 and prefill chunking is enabled. The rest of serving parameters remain as default values. We also consider a set of workloads $w = ( L _ { \mathrm { i n } } , L _ { \mathrm { o u t } } , 1 2 8 )$ , where $L _ { \mathrm { i n } } + L _ { \mathrm { o u t } } = 1 2 0 0$ . Indeed, the total sequence length is fixed, $\mathrm { i . e . , } L _ { \mathrm { s e q } } = 1 2 0 0$ , while varying only the composition of the request by changing $L _ { \mathrm { i n } }$ and $L _ { \mathrm { o u t } }$ . Each point in the feasible region therefore represents the latency obtained by one parallelism configuration for a particular workload composition. For each parallelism configuration $p ,$ moving across the points corresponds to moving from decode-heavy workloads, with relatively small $L _ { \mathrm { i n } }$ and large $L _ { \mathrm { o u t } }$ , toward prefill-heavy workloads, with relatively large $L _ { \mathrm { i n } }$ and small $L _ { \mathrm { o u t } }$ . Since the total sequence length remains fixed, the figure isolates the effect of changing the relative contribution of prefill and decode rather than simply increasing the overall workload size. Therefore, the collection of points corresponding to the available parallelism configurations defines the latency feasible set corresponding to given workload. The resulting frontier reveals a clear and systematic transition across parallelism configurations.

In Fig. 1, we observe that TP8 occupies the low-ITL/high-TTFT end of the frontier, PP8 occupies the low-TTFT/high-ITL end, while the hybrid configurations provide intermediate trade-offs. The contrastive behavior of TP8 and PP8 can be explained by the fundamentally different compute-communication trade-off induced by TP and PP. During prefill, the model processes a relatively large amount of input computation in a highly parallel manner. TP distributes each layer’s computation across GPUs, but requires collective communication between GPUs during the execution of the layers. Because prefill involves substantial activation processing, these communication operations can contribute significantly to the critical path. PP, in contrast, partitions the model into sequential stages and largely avoids the fine-grained collective communication required by TP. At sufficiently high concurrency, the pipeline can also remain well utilized. Consequently, PP8 can achieve substantially lower TTFT than TP8, placing PP8 toward the bottom of the figure.

The behavior is reversed during decode. Each decoding iteration processes only the newly generated token, making the computation per iteration relatively small. Under these conditions, the communication and synchronization costs associated with pipeline execution become more prominent relative to useful computation [35]. A deep PP configuration must propagate each token through multiple pipeline stages, and the limited amount of computation available per decoding step makes pipeline scheduling overhead more difficult to amortize. TP8, on the other hand, distributes the computation of every layer across all available GPUs, thereby reducing the computation required by each GPU and providing a highly parallel execution path for each decoding iteration. Although TP still incurs collective communication, its lower per-token computation time results in substantially lower ITL. TP8 therefore occupies the low-ITL end of the frontier.

![](images/42434d91b67ae16a7aacf2ae220534f977a34cffdbc297adc52e73122714b91a.jpg)  
Figure 2: TTFT vs. ITL for various OSL. The effect of OSL is not uniform across parallelism configurations. Longer OSL mainly improves ITL through better amortization of decode overheads, with the largest benefit observed for TPdominant configurations.

This creates a TP-PP latency trade-off depicted in Fig. 1. TP8 provides the fastest sustained token generation but incurs a higher cost to produce the first token. On the other hand, PP8 provides the fastest first response but sacrifices pertoken generation latency. The hybrid configurations, TP2PP4 and TP4PP2, lie between these two extremes because they inherit the characteristics of both TP and PP. Increasing the TP degree progressively favors decode efficiency, while increasing the PP degree progressively favors prefill efficiency.

It is worth mentioning that, as workload transitions from decode-heavy to prefill-heavy, the Pareto-optimal operating point shifts systematically from TP-dominant configurations, through hybrid configurations, toward PP-dominant configurations. To explain this, consider a decode-heavy workload, characterized by a relatively small $L _ { \mathrm { i n } }$ and large $L _ { \mathrm { o u t } } .$ In this regime, the request spends comparatively more of its execution in autoregressive decoding. Consequently, the sustained per-token cost becomes increasingly important, and configurations with lower ITL provide more favorable latency behavior. We observe that TP8 configuration lies at the most favorable end of the frontier for this regime, making high-degree TP an effective configuration for decode-heavy workloads.

As $L _ { \mathrm { i n } }$ increases and $L _ { \mathrm { o u t } }$ decreases, the workload becomes increasingly balanced between prefill and decode. Neither phase overwhelmingly determines the latency behavior, and the advantage of an extreme TP or PP configuration diminishes. The hybrid configurations consequently become increasingly attractive, occupying the middle portion of the Pareto frontier. Moreover, the two hybrid configurations provide a gradual transition between the TP- and PP-dominant regimes, i.e., TP4PP2 retains more of the TP characteristics and therefore favors lower ITL, whereas TP2PP4 shifts further toward the PP characteristics and favors lower TTFT. Finally, as $L _ { \mathrm { i n } }$ becomes large relative to $L _ { \mathrm { o u t } }$ , workload becomes prefill-heavy. The cost of processing the input context dominates the first token latency, and the advantage of PP becomes increasingly pronounced. PP8 therefore moves toward the favorable end of the latency trade-off, providing the lowest TTFT among the evaluated configurations, albeit with the highest ITL. In this regime, the additional decode efficiency of TP8 provides diminishing benefit because relatively little of the request is spent generating output tokens.

Note that the four configurations should not be viewed as having a fixed global performance ranking. Instead, each configuration occupies a different position in the TTFT-ITL design space, and the desirable operating point changes as the workload moves through different prefill - decode regimes. The Pareto frontier provides a compact representation of this behavior by exposing the set of non-dominated latency trade-offs available for each workload.

In Fig. 2, we consider the workload w $= ( 1 0 0 0 , L _ { \mathrm { o u t } } , 2 5 6 )$ , and isolate the effect of OSL on the TTFT-ITL trade-off by varying $L _ { \mathrm { o u t } } \in \{ 2 0 0 , 4 0 0 , 8 0 0 \}$ . As $L _ { \mathrm { o u t } }$ increases, the latency frontier expands primarily along the ITL dimension, while its TTFT range remains largely unchanged. This behavior is expected because TTFT is dominated by the prefill phase, whose workload is determined primarily by the fixed input length and is therefore relatively insensitive to the number of tokens subsequently generated. In contrast, a longer $L _ { \mathrm { o u t } }$ provides more decoding iterations over which per-request startup, scheduling, and pipeline effects can be amortized. Consequently, the effective measured ITL decreases as $L _ { \mathrm { o u t } }$ increases, exposing a wider range of achievable TTFT–ITL trade-offs. Thus, increasing OSL does not substantially change the TTFT, but allows the system to better amortize decode-related overheads over a longer generation process.

Importantly, the effect of OSL is not uniform across parallelism configurations. PP8 shows relatively little improvement in ITL as OSL increases, whereas the benefit becomes progressively larger for TP2PP4 and, most notably, TP8. This reflects the different strengths of the parallelism strategies during decoding. PP-dominant configurations incur pipeline-related synchronization and stage-propagation costs on each decoding iteration, mainly because these costs remain significant throughout generation, increasing the number of output tokens provides limited improvement in the measured per-token latency [35]. In contrast, TP-dominant configurations are better suited to autoregressive decoding. Once the decoding pipeline is established, the computation can be distributed across more GPUs and the associated per-request overheads are amortized over increasingly many decoding iterations. As a result, longer OSL increasingly favors TP-dominant configurations, causing their points to move toward lower ITL while their TTFT remains largely unchanged. The figure therefore shows that OSL does not merely shift the overall latency frontier; it changes the rela tive benefit of parallelism configurations, increasingly favoring TP as the workload becomes more decode-intensive.

## 3 SLO-Aware Parallelism

In the previous section, we characterized the fundamental latency trade-offs induced by different parallelism configurations through the TTFT-ITL Pareto frontier. This perspective is important because it reveals the inherent operating regions of TP-, PP-, and hybrid-parallel execution and identifies the set of non-dominated latency trade-offs available for a given workload. However, practical serving systems are rarely optimized for latency alone. Instead, they must deliver as much useful service capacity as possible while satisfying application-level latency requirements. A configuration with higher throughput is not necessarily preferable if it produces excessive latency for a significant fraction of requests, while a configuration with excellent latency may leave substantial system capacity underutilized.

Moreover, latency requirements are inherently asymmetric. For example, reducing TTFT from 2(s) to 1(s) can be highly valuable for an interactive application, whereas the same improvement may provide little additional benefit once the application’s responsiveness target has already been met. This motivates extending the latency trade-off analysis to an SLO-aware serving perspective, where parallelism configurations are evaluated according to how much useful throughput they can sustain under a given latency requirement. For a parallelism configuration $p ,$ let $\lambda _ { p } ( w )$ denote its achieved request throughput for workload w. Our goal is to find the optimal parallelism configuration that maximizes the throughput while respecting the TTFT constraints <sup>6</sup>. To do so, let $F _ { p } ( t | w )$ denote the empirical CDF of $T _ { \mathrm { T T F T } } ^ { ( p , w ) }$ , i.e., $F _ { p } ( t | w ) = \mathrm { P r } ( T _ { \mathrm { T T F T } } ^ { ( p , w ) } \leq t )$ . In particular, $F _ { p } ( t | w )$ represents the fraction of the requests that satisfies the latency requirement for parallelism configuration $p$ and workload w. Assuming $T _ { s }$ to be the target TTFT SLO constraint, goodput is defined as

$$
G _ { p } ( w , T _ { s } ) = \lambda _ { p } ( w ) F _ { p } ( T _ { s } | w ) .\tag{12}
$$

Thus, goodput can be interpreted as the product of two competing properties of a parallelism configuration: I) the raw serving capacity $\lambda _ { p } ( w )$ and II) the probability of satisfying the latency requirement $F _ { p } ( T _ { s } | w )$ . Such decomposition is important because maximizing raw throughput alone can favor a configuration that could generate many requests but violates the TTFT requirement for a substantial fraction of them. As we later show in this section, and under tight SLO requirement, each parallelism exhibits a trade-off between these components. The goal is to choose a parallelism configuration that delivers the largest amount of useful serving capacity for a given latency requirement. Therefore, we formalize SLO-aware parallelism selection as follows

$$
p ^ { * } ( w , T _ { s } ) = \arg \operatorname* { m a x } _ { p \in \mathcal { P } } G _ { p } ( w , T _ { s } ) .\tag{13}
$$

According to (13), each configuration $p \in \mathcal P$ contributes to goodput according to both its achievable request throughput $\lambda _ { p } ( w )$ and the fraction of requests that satisfy the SLO, $\bar { F _ { p } ( T _ { s } | w ) }$ . The optimization therefore does not simply favor the configuration with the highest throughput or the lowest TTFT; instead, it balances the two through their product. As $T _ { s }$ varies, the relative contribution of SLO compliance also changes, causing the maximizing configuration $p ^ { * } ( w , T _ { s } )$ to potentially change. Therefore, the optimization in (13) tries to capture the best parallelism configuration that makes the most effective use of the available hardware for a given level of latency requirement.

![](images/23f72ff9c897a78b58fda0e6adb88dc2715cc4f4313ad63c354641d6b46e2f1f.jpg)  
TTFT (s)

![](images/13556fe15719e6196140abd9a58240c3947427dd5b76da8b31abe0429a8a4f99.jpg)  
ITL (ms)  
Figure 3: Left Plot: TTFT distribution (Prefill), Right plot: ITL distribution (decode). PP-dominant configurations achieve lower TTFT, while TP-dominant configurations achieve lower ITL, thereby highlighting the latency trade-off between prefill and decoding across different parallelisms.

Note that, when solving (13), there is a fundamental trade-off between latency compliance and serving capacity across different parallelism configurations. To show this, for two configurations $p _ { i }$ and $p _ { j }$ , the optimal $\breve { T } _ { s } ^ { * }$ should solve $G _ { p _ { i } } ( w , T _ { s } ^ { * } ) = G _ { p _ { j } } ( w , T _ { s } ^ { * } )$ , or equivalently $\lambda _ { p _ { i } } ( w ) F _ { p _ { i } } ( T _ { s } ^ { * } | w ) = \check { \lambda _ { p _ { j } } } ( w ) F _ { p _ { j } } ( T _ { s } ^ { * } | w )$ , and therefore

$$
\frac { F _ { p _ { i } } ( T _ { s } ^ { * } | w ) } { F _ { p _ { j } } ( T _ { s } ^ { * } | w ) } = \frac { \lambda _ { p _ { j } } ( w ) } { \lambda _ { p _ { i } } ( w ) } .\tag{14}
$$

For tight TTFT SLO, configurations with favorable TTFT distributions, such as PP-dominant configurations, can achieve substantially higher SLO attainment, which allows them to overcome their lower raw throughput. As the SLO is relaxed $( T _ { s } \to \infty )$ , the compliance probability $F _ { p } ( T _ { s } | w ) \to 1 \forall p \in \mathcal { P }$ , and therefore goodput approaches to raw throughput, i.e., $G _ { p } ( w , T _ { s } ) \stackrel { \textstyle > } {  } \lambda _ { p } ( w )$ . Thus, parallelism configurations with higher serving capacity, such as TP-dominant configurations, eventually become preferable.

## 3.1 Experiment Results on SLO-Aware Parallelism Selection

Fig. 3 demonstrates the latency trade-offs by examining the distribution of request latency. In this experiment, we fix the workload as $w = ( 1 2 0 0 , 2 8 0 0 , 2 5 6 )$ where $L _ { \mathrm { s e q } } = 4 0 0 0$ . The left plot in Fig. 3 shows the TTFT CDF during prefill, where PP8 provides the most favorable distribution, with its CDF rising more rapidly than the other configurations. This indicates that a larger fraction of requests achieve a lower TTFT under PP8. This is mainly due to that fact that prefill is compute-intensive, and PP partitions the model across stages while avoiding the fine-grained collective communication required by TP. As a result, PP8 can perform efficiently during the prefill, particularly when the amount of computation is sufficient to amortize pipeline overhead. Conversely, as we move towards TP-dominant configurations, such as TP8, they tend to exhibit a wider and slower-rising TTFT distribution, reflecting the additional communication associated with distributing each layer’s computation across all GPUs. It is worth noting that hybrid configurations provide intermediate behavior, which reinforces the observation that increasing the PP degree tends to favor TTFT.

The right plot in Fig. 3 shows the opposite behavior during autoregressive decoding. Here, TP8 provides the most favorable ITL distribution, with its CDF shifted substantially toward lower values, while PP8 exhibits wider ITL distribution. During decoding, each iteration processes only a small amount of new computation, making communication and synchronization overhead more significant relative to useful computation. TP8 can exploit all available GPUs to reduce the per-token computation time, thereby making it particularly effective for sustained token generation, whereas the sequential propagation of tokens through multiple pipeline stages makes PP8 less effective in this regime.

![](images/278f7611cee25440ba542a8163a53ba7a0a54ef0c1a89acb3081303b297ca734.jpg)  
Figure 4: Latency compliance and serving capacity trade-off for fixed workload $w = ( 1 2 0 0 , 2 8 0 0 , 2 5 6 )$ with $L _ { \mathrm { s e q } } =$ 4000. Filled marker in each $T _ { s }$ , indicates the parallelism configuration that yields the maximum goodput for the corresponding $T _ { s }$

The hybrid configurations lie between these two extremes, with their behavior determined by the balance between TP and PP. Together, the two CDFs provide a phase-level explanation for the TTFT–ITL Pareto frontier shown in Fig. 1, where PP-dominant configurations favor the prefill phase and therefore TTFT, whereas TP-dominant configurations favor the decode phase and therefore ITL.

Fig. 4 provides a more direct decomposition of the SLO-aware goodput trade-off by plotting each configuration in the throughput-SLO-attainment space. Recall that goodput is $G _ { p } ( { \breve { T } } _ { s } ) \stackrel { \cdot } { = } \lambda _ { p } F _ { p } ( T _ { s } ) ^ { 7 }$ , so a configuration must simultaneously provide high serving throughput $\lambda _ { p }$ and high SLO attainment $F _ { p } \big ( T _ { s } \big )$ . As shown in Fig. 4, these two quantities are fundamentally in tension. Configurations with higher throughput (i.e., TP-dominant) generally achieve a smaller fraction of requests within a given TTFT SLO, while configurations with stronger TTFT performance (PP-dominant) tend to sacrifice throughput for a better TTFT SLO attainment. This occurs because the parallelism choices that accelerate prefill and reduce TTFT, particularly PP-dominant configurations, do so at the expense of decode efficiency and therefore sustained request throughput; TP-dominant configurations exhibit the opposite behavior [35]. Consequently, maximizing raw throughput alone is insufficient for maximizing goodput.

More importantly, this observation makes the SLO-dependent shift in the optimal parallelism explicit. For each $T _ { s } ,$ the filled marker identifies the configuration maximizing the goodput, illustrating that the optimal point is determined by the balance between throughput capacity and SLO compliance rather than either quantity independently. Under stringent TTFT requirements $( T _ { s } = 5$ and 10(s)), SLO attainment is the dominant differentiating factor. PP8 achieves substantially higher $F _ { p } ( T _ { s } )$ , which compensates for its lower throughput and results in the highest goodput. As the SLO is progressively relaxed to 20 and 25(s), the SLO-attainment advantage of PP8 diminishes as more requests from all configurations satisfy the requirement. The throughput advantage of TP-dominant configurations can then dominate, shifting the goodput optimality toward the hybrid configurations and ultimately toward TP8. Thus, the plot reveals a fundamental capacity–compliance trade-off induced by parallelism. In particular, PP-dominant configurations trade throughput for stronger latency compliance, whereas TP-dominant configurations trade latency compliance for higher serving capacity. The optimal strategy consequently moves from PP to hybrid to TP as the SLO is relaxed, reinforcing the earlier observation for the operating-region in Fig. 1.

Fig. 5 depicts goodput vs. TTFT SLO for various parallelism configurations. The observation in this figure indicates how goodput captures the interplay between a configuration’s serving capacity and its ability to satisfy a latency requirement as shown in Fig. 4. Under a tight TTFT SLO, we observe that PP8 can achieve the highest goodput despite having a lower raw throughput λ than TP-dominant configurations. The reason is that PP8 provides substantially higher SLO attainment, $\tilde { F _ { \mathrm { P P 8 } } } ( T _ { s } )$ , because of its superior TTFT behavior. PP8 therefore wins whenever its compliance advantage is sufficiently large to compensate for its throughput disadvantage

![](images/7d1f8ca86dba955e6094e384eec8499364387c88cb94c8b5f326e61ec083ea8f.jpg)  
Figure 5: Goodput vs. TTFT SLO for fixed workload $w = ( 1 2 0 0 , 2 8 0 0 , 2 5 6 )$ where $L _ { \mathrm { s e q } } = 4 0 0 0$ . As TTFT SLO is relaxed, the optimal parallelism shifts from PP-dominant configurations to TP-dominant configurations.

$$
\lambda _ { \mathrm { P P 8 } } F _ { \mathrm { P P 8 } } ( T _ { s } ) > \lambda _ { \mathrm { T P 8 } } F _ { \mathrm { T P 8 } } ( T _ { s } ) ,\tag{15}
$$

or equivalently

$$
\frac { F _ { \mathrm { P P 8 } } ( T _ { s } ) } { F _ { \mathrm { T P 8 } } ( T _ { s } ) } > \frac { \lambda _ { \mathrm { T P 8 } } } { \lambda _ { \mathrm { P P 8 } } } .\tag{16}
$$

Thus, at stringent SLOs, the ability to meet the latency requirement matters more than raw serving capacity. For example, as shown in Fig. 4, TP8 may serve requests at a higher rate, but a large fraction of those requests may fail the TTFT requirement and consequently contribute less to goodput. As the TTFT SLO is relaxed, such behavior progressively reverses. The SLO-attainment probability $F _ { p } ( T _ { s } )$ increases for all configurations and approaches one as the constraint becomes sufficiently loose. Consequently,

$$
\operatorname* { l i m } _ { T _ { s } \to \infty } G _ { p } ( T _ { s } ) = \operatorname* { l i m } _ { T _ { s } \to \infty } \lambda _ { p } F _ { p } ( T _ { s } ) = \lambda _ { p } ,\tag{17}
$$

so goodput increasingly reflects the underlying throughput capacity of each configuration. The latency-compliance ad vantage of PP8 therefore becomes less valuable, while the higher decoding throughput of TP-dominant configurations becomes increasingly important. This produces the observed transition from PP8 at stringent SLOs, through hybrid configurations at intermediate SLOs, to TP8 under relaxed SLOs. Together with the latency trade-offs established earlier in Section 2, this result shows that the choice of parallelism is fundamentally SLO-dependent. PP-dominant parallelism is advantageous when latency compliance is the primary constraint, whereas TP-dominant parallelism becomes preferable when the latency requirement can be relaxed enough to exploit its greater serving capacity.

## 4 Parallelism for Long Context

The preceding sections characterize parallelism primarily through the latency trade-offs, and show that the preferred configuration depends on the workload composition and latency requirement. However, as LLM workloads move toward the long-context workloads, the ability of a parallelism configuration to accommodate the workload becomes critical. Increasing the input sequence length not only increases prefill computation and TTFT, but also increases the memory required to store the model’s intermediate states and KV cache, potentially changing both performance and feasible operating range. Thus, a configuration that is attractive for shorter requests may become inefficient or even infeasible as context length grows. To isolate this effect, we fix the $L _ { \mathrm { o u t } }$ , and just increase $L _ { \mathrm { i n } } .$ , thereby increasing the request’s context size while keeping the decode workload fixed. This setup allows us to study the evolution of throughput, latency, SLO compliance, and ultimately the feasible operating range of different parallelism configurations. In this setting, parallelism is no longer only a mechanism for trading TTFT against ITL; it also determines how far the system can scale in context length before performance degrades or the workload become infeasible.

![](images/f2a95dcb36583952b1630e5b844e0f731d168a341af0f5a3fe47e2d3356c6251.jpg)  
T<sub>s</sub> = 5(s)

![](images/dae56da43e5e1ea3967559450bc0fc75849077a86ba1a099b76ba14fd512984d.jpg)  
T<sub>s</sub> = 2(s)  
Figure 6: Goodput vs. Sequence Length for workload $w = ( L _ { \mathrm { i n } } , 1 0 0 0 , 1 2 8 )$ where $L _ { \mathrm { i n } }$ varies to construct values of x-axis. Left plot: $T _ { s } = 5 \bar { ( \mathrm { s ) } }$ . Right Plot: $T _ { s } = 2 ( \mathrm { s } )$ . Increasing sequence length progressively favors PP-dominant parallelism over TP-dominant parallelism, highlighting PP’s stronger scalability for long context workloads.

Fig. 6 depicts the goodput against the sequence length for $T _ { s } = 2 ( \mathrm { s } )$ and 5(s). As sequence length increases, in both scenarios, we observe a systematic shift in the relative advantage of parallelism configurations toward PP-dominated strategies, indicating that PP-dominant parallelism scales more favorably with long-sequence workloads. In the left plot in Fig. 6, where $T _ { s } = 5 ( \mathrm { s } )$ , at shorter sequence length, TP-dominant configurations outperform their PP-dominant counterparts, mainly due to their higher serving capacity and strong decode efficiency. However, as the input sequence length grows, prefill becomes increasingly dominant and the communication overhead associated with TP execution becomes more consequential. PP-dominant configurations, in contrast, avoid much of this fine-grained collective communication and therefore experience a slower degradation in TTFT and SLO attainment. Consequently, the initial throughput advantage of TP-dominant configurations is progressively eroded as sequence length increases, which allows PP-dominant configurations to become increasingly competitive and eventually achieve higher goodput.

This effect is particularly pronounced under stringent TTFT requirements as indicated in the right plot in Fig. 6, where $T _ { s } = 2 ( s )$ . From (12), where $G _ { p } ( w , T _ { s } ) = \lambda _ { p } ( \bar { w } ) F _ { p } ( T _ { s } | w )$ , we note that long-context scaling is determined not only by the throughput $\lambda _ { p } ( w )$ , but also by how rapidly SLO attainment $F _ { p } ( T _ { s } | w )$ deteriorates with increasing sequence length. Our results indicate that PP-dominant configurations maintain higher SLO attainment over a wider range of sequence lengths (illustrated in Fig. 7), which enables them to sustain useful throughput farther into the long-context regime. Thus, the key observation is not simply that PP performs better at the largest sequence length, but that PPdominant parallelism exhibits more favorable scaling with sequence length, with its advantage becoming increasingly important as the TTFT SLO becomes tighter.

Fig. 7 isolates the SLO-attainment component of goodput and shows the sensitivity of $F _ { p } ( T _ { s } | w )$ with respect to sequence length under both tight and relaxed TTFT SLO requirement. In particular, for each parallelism configuration we plot<sup>8</sup>

$$
F _ { p } ( T _ { s } | L _ { \mathrm { s e q } } ) = P r ( T _ { \mathrm { T T F T } } ^ { ( p , L _ { \mathrm { s e q } } ) } < T _ { s } | L _ { \mathrm { s e q } } ) .\tag{18}
$$

In the left plot of Fig. 7, as the sequence length increases, $F _ { p } ( T _ { s } \mid L _ { \mathrm { s e q } } )$ consistently decreases across all configurations. This reflects the growing cost of processing the input sequence during prefill. Longer sequence increases the work required before the first token can be produced, thereby shifting the TTFT distribution toward larger values and causing an increasing fraction of requests to violate the required TTFT SLO. This effect becomes substantially sharper for the tighter $T _ { s } = 2 ( \mathrm { s } )$ as shown in the right plot of Fig. 7. Intuitively, a relaxed SLO lies farther into the right side of the TTFT distribution, so moderate increases in TTFT with sequence length may leave many requests still compliant. In contrast, a tighter threshold lies much deeper in the low-latency portion of the distribution; where even a modest shift in TTFT therefore causes a large reduction in the fraction of requests satisfying the TTFT SLO. Thus, tighter latency requirements amplify the impact of sequence length on SLO attainment.

![](images/979605ed9dfd9fee3ad54f0a1b23c01a031d3fb7fa9259f297806e92ca493a2d.jpg)  
T<sub>s</sub> = 5(s)

![](images/e2c494c81ec70495bb21d8ed4449f62b5de298306f54e60e4031e5b04023c402.jpg)  
T<sub>s</sub> = 2(s)  
Figure 7: CDF vs. Sequence Length for workload $w = ( L _ { \mathrm { i n } } , 1 0 0 0 , 1 2 8 )$ where $L _ { \mathrm { i n } }$ varies to construct values of x-axis. Left plot: $T _ { s } = 5 ( \mathrm { s } )$ . Right plot: $\bar { T } _ { s } = 2 ( \mathrm { s } )$ . PP-dominant configurations maintain higher SLO attainment as sequence length increases, especially under tight TTFT SLOs.

The observation in Fig. 7 further indicates that different parallelism configurations exhibit different sequence length scaling behavior. The relevant quantity is not only the absolute TTFT at a particular sequence length, but how rapidly SLO attainment degrades as sequence length grows. In particular, the PP-dominant configurations retain a larger $F _ { p } ( T _ { s } \mid L _ { \mathrm { s e q } } )$ over increasing sequence lengths, while TP8 exhibits a substantially faster decline, especially under the tighter SLO. This difference directly propagates to goodput through

$$
G _ { p } ( L _ { \mathrm { s e q } } , T _ { s } ) = \lambda _ { p } ( L _ { \mathrm { s e q } } ) F _ { p } ( T _ { s } | L _ { \mathrm { s e q } } ) .\tag{19}
$$

Although TP-dominant configurations may have higher throughput $\lambda _ { p } ,$ , that advantage can be progressively overwhelmed when their SLO-attainment probability falls more rapidly with sequence length. Conversely, PP-dominant configurations can sustain a larger fraction of SLO-compliant requests as the sequence length grows, thereby allowing their goodput to degrade more gracefully and eventually surpass TP-dominant configurations. The key insight is that long-sequence scalability is governed not only by throughput degradation, but also by the rate at which latency-SLO attainment deteriorates with sequence length. This provides direct explanation for the increasing advantage of PP-dominant parallelism observed in Fig. 6, particularly under stringent TTFT requirements.

## 5 Conclusions

In this paper, we presented a systematic study of LLM inference parallelism from the perspectives of latency tradeoffs, SLO-aware serving, and long-context scalability. We showed that TP, PP, and their hybrid combinations occupy distinct operating points in the TTFT-ITL design space, resulting in a workload-dependent Pareto frontier rather than a single globally optimal configuration. Our analysis revealed a clear transition from TP-dominant configurations for decode-heavy workloads to PP-dominant configurations for prefill-heavy workloads, which highlights the importance of considering workload composition when selecting inference parallelism. We further demonstrated that average latency metrics alone are insufficient to characterize serving performance and that TTFT and ITL should be treated as complementary dimensions when evaluating distributed inference systems.

Building upon this latency-centric view, we formulated SLO-aware parallelism selection through goodput and showed that the optimal configuration depends jointly on serving capacity and latency compliance. Our results indicate that stringent TTFT requirements favor PP-dominant configurations, whereas relaxed SLOs progressively shift the opti mality toward TP-dominant configurations with higher throughput capacity. Finally, we demonstrated that increasing context length fundamentally changes the relative advantages of parallelism strategies, with PP-dominant configura tions exhibiting more favorable scaling and maintaining higher SLO attainment in the long-context regime. Collectively, these findings suggest that parallelism should be treated not as a fixed deployment choice, but as a workload-, SLO-, and context-dependent operating decision. We believe that the proposed operating-region framework provides a useful foundation for future adaptive serving systems capable of dynamically selecting parallelism configurations in response to changing workload conditions and application requirements.

## Acknowledgments

The authors would like to acknowledge the support of the Dell leadership team and thank colleagues across Dell for their valuable discussions, constructive feedback, and technical insights that contributed to this research.

## References

[1] A. Agrawal, N. Kedia, A. Panwar, J. Mohan, N. Kwatra, B. Gulavani, A. Tumanov, and R. Ramjee, “Taming throughput-latency tradeoff in LLM inference with sarathi-serve,” in 18th USENIX symp. on operating sys. design and implementation (OSDI 24), pp. 117–134, 2024.

[2] Y. Zhong, S. Liu, J. Chen, J. Hu, Y. Zhu, X. Liu, X. Jin, and H. Zhang, “DistServe: Disaggregating prefill and decoding for goodput-optimized large language model serving,” in 18th USENIX Symposium on Operating Systems Design and Implementation (OSDI 24), pp. 193–210, 2024.

[3] C. Wang, P. Zuo, Z. Chen, Y. Liang, Z. Yu, and M. Yang, “Prefill-decode aggregation or disaggregation? unifying both for goodput-optimized LLM serving,” 2025. Available: https://arxiv.org/abs/2508.01989.

[4] S. Chen, Z. Jia, S. Khan, A. Krishnamurthy, and P. B. Gibbons, “SLOs-Serve: Optimized serving of multi-SLO LLMs,” 2025. Available: https://arxiv.org/abs/2504.08784.

[5] C. Ruan, Y. Chen, D. Tian, Y. Shi, Y. Wu, J. Li, and C. Li, “DynaServe: Unified and elastic execution for dynamic disaggregated LLM serving,” 2025. Available: https://arxiv.org/abs/2504.09285.

[6] Q. Wang, “Taming request imbalance: SLO-aware scheduling for disaggregated LLM inference,” 2026. Available:https://arxiv.org/abs/2605.02329.

[7] H. Amer, R. Karim, A. Pourranjbar, W. Zhang, W. Ahmed, and B. Chen, “Distributed hybrid parallelism for large language models: Comparative study and system design guide,” 2026. Available: https://arxiv.org/abs/2602.09109.

[8] L. Xu, K. K. Suresh, Q. Anthony, N. Alnaasan, and D. K. Panda, “Characterizing communication patterns in distributed large language model inference,” 2025. Available: https://arxiv.org/abs/2507.14392.

[9] T. Guo, X. Zhang, J. Du, Z. Chen, N. Xiao, and Y. Lu, “gLLM: Global balanced pipeline parallelism system for distributed LLM serving with token throttling,” 2025. Available: https://arxiv.org/abs/2504.14775.

[10] M. Hidayetoglu, A. Qiao, M. Wyatt, J. Rasley, Y. He, and S. Rajbhandari, “Shift parallelism: Low-latency, high-throughput LLM inference for dynamic workloads,” 2026. Available: https://arxiv.org/abs/2509.16495.

[11] H. Yuan, K. Zheng, Y. Bai, Y. Zhang, Y. Zhang, B. Wu, X. Gao, and D. Cheng, “ReMP: Low-downtime runtime model-parallelism reconfiguration for LLM serving,” 2026. Available: https://arxiv.org/abs/2606.18741.

[12] S. Gao, J. Yin, F. Wang, and W. Dong, “FLYING SERVING: On-the-fly parallelism switching for large language model serving,” 2026. Available: https://arxiv.org/abs/2602.22593.

[13] M. Adnan, R. Mahapatra, P. J. Nair, D. Berger, P. Zardoshti, R. Fonseca, and E. Choukse, “Cascade: Exploiting SLO-aware latency budget for fair and high goodput LLM inference serving,” 2026. Available: https://arxiv.org/abs/2608.06557.

[14] V. Srivatsa, Z. He, P. Guo, D. Li, and Y. Zhang, “Nitsum: Serving tiered LLM requests with adaptive tensor parallelism,” 2026. Available: https://arxiv.org/abs/2605.05467.

[15] Y. Shi, X. Wang, J. Gao, J. Luo, X. Zhou, F. Liu, K. Luo, X. Li, X. Guo, and L. Feng, “VPP: Virtual pipeline parallelism for efficient chunked prefill in long-context LLM inference,” 2026. Available: https://arxiv.org/abs/2608.26523.

[16] J. Guo, R. Wang, P. Huang, Z. Lv, Z. Wang, K. Liu, T. Lan, L. Qu, X. Wang, and T. Yang, “Adaptive context parallelism for production LLM serving,” 2026. Available: https://arxiv.org/abs/2609.04774.

[17] “Deadline-aware adaptive prefill chunking for efficient large language model serving,” 2026. Available: https://arxiv.org/abs/2609.07883.

[18] A. Agrawal, A. Agarwal, N. Kedia, J. Mohan, S. Kundu, N. Kwatra, R. Ramjee, and A. Tumanov, “Etalon: Holistic performance evaluation framework for LLM inference systems,” 2024. Available: https://arxiv.org/abs/2407.07000.

[19] S. Usami, V. Vishwanath, and E. W. Bethel, “Prefill/decode-aware evaluation of LLM inference on emerging AI accelerators,” 2026. Available: https://arxiv.org/abs/2606.17104.

[20] X. Shi, C. Cai, J. Du, and Z. Jia, “Nexus: Proactive intra-GPU disaggregation of prefill and decode in LLM serving,” 2025. Available: https://arxiv.org/abs/2507.06608.

[21] W. Kwon, Z. Li, S. Zhuang, Y. Sheng, L. Zheng, C. H. Yu, J. E. Gonzalez, H. Zhang, and I. Stoica, “Efficient memory management for large language model serving with pagedattention,” 2023. Available: https://arxiv.org/abs/2309.06180.

[22] B. Du, B. Huangfu, Y. Luo, C. Chen, Z. Li, M. Yu, X. Fan, and M. Guo, “GoodServe: Towards high-goodput serving of agentic LLM inferences over heterogeneous resources,” 2026. Available: https://arxiv.org/abs/2605.16867.

[23] K. Goel, J. Mohan, N. Kwatra, R. S. Anupindi, and R. Ramjee, “Niyama: Breaking the silos of LLM inference serving,” 2025. Available: https://arxiv.org/abs/2503.22562.

[24] B. Lin, C. Zhang, T. Peng, H. Zhao, W. Xiao, M. Sun, A. Liu, Z. Zhang, L. Li, X. Qiu, S. Li, Z. Ji, T. Xie, Y. Li, and W. Lin, “Infinite-LLM: Efficient LLM service for long context with distattention and distributed KVCache,” 2024. Available: https://arxiv.org/abs/2401.02669.

[25] C. Zhang, K. Du, S. Liu, W. Kwon, X. Mo, Y. Wang, X. Liu, K. You, Z. Li, M. Long, J. Zhai, J. Gonzalez, and I. Stoica, “Jenga: Effective memory management for serving LLM with heterogeneity,” 2025. Available: https://arxiv.org/abs/2503.18292.

[26] J. Cheng, D. T. Do, and D. T. Nguyen, “Robust KV cache management for LLM serving under output token length uncertainty,” 2026. Available: https://arxiv.org/abs/2607.16892.

[27] A. Agrawal et al., “No request left behind: Tackling heterogeneity in long-context LLM inference with Medha,” 2025. Available: https://arxiv.org/abs/2409.17264.

[28] C. Jiang, Z. Cai, Y. Tian, . Jia, Y. Wang, and C. Wu, “DCP: Addressing input dynamism in long-context training via dynamic context parallelism,” 2025. Available: https://arxiv.org/abs/2510.10620.

[29] Y. Niu, H. Xiao, D. Liu, W. Zhou, and J. Li, “DHP: Efficient scaling of MLLM training with dynamic hybrid parallelism,” 2026. Available: https://arxiv.org/abs/2602.21788.

[30] Z. Wang, E. Liu, L. Jiang, Z. Yu, Z. Pan, Y. Guan, Y. Wang, and Y. Ding, “FlashCP: Load-balanced communication-efficient context parallelism for LLM training,” 2026. Available: https://arxiv.org/abs/2606.08476.

[31] Y. Zhao, X. Nie, K. Zhu, S. Ma, Z. Lai, H. Hao, Y. Zhou, B. Kasikci, and I. Stoica, “Unleashing scalable context parallelism for foundation models pre-training via FCP,” 2026. Available: https://arxiv.org/abs/2605.08524.

[32] R. Ghadia, M. Abraham, S. Vorobyov, and M. Ryabinin, “Untied Ulysses: Memory-efficient context parallelism via headwise chunking,” 2026. Available: https://arxiv.org/abs/2602.21196.

[33] Z. Zhang and H. Shen, “PecSched: Preemptive and efficient cluster scheduling for LLM inference,” 2025. Available: https://arxiv.org/abs/2409.15104.

[34] Q. Zhou, P. Yin, P. Zuo, and J. Cheng, “SparseServe: Unlocking parallelism for dynamic sparse attention in long-context LLM serving,” 2025. Available: https://arxiv.org/abs/2509.24626.

[35] J. Mirzaei and J. Mitra, “Characterizing parallelism strategies in LLM inference: Fundamental computecommunication trade-offs,” 2026. Available: https://arxiv.org/abs/2610.05305.

[36] A. Yang, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu, C. Li, D. Liu, F. Huang, H. Wei, H. Lin, J. Yang, J. Tu, J. Zhang, J. Yang, J. Yang, J. Zhou, J. Lin, K. Dang, K. Lu, K. Bao, K. Yang, L. Yu, M. Li, M. Xue, P. Zhang, Q. Zhu, R. Men, R. Lin, T. Li, T. Tang, T. Xia, X. Ren, X. Ren, Y. Fan, Y. Su, Y. Zhang, Y. Wan, Y. Liu, Z. Cui, Z. Zhang, and Z. Qiu, “Qwen2.5 technical report,” 2025. Available: https://arxiv.org/abs/2412.15115.

[37] NVIDIA, “NVIDIA H200 tensor core GPU: Supercharging AI and HPC workloads.” https://dam-cdn.nvd. orangelogic.com/AssetLink/5o2qgy5d2835ve2pm11i62kv8mphqta8.pdf, 2026. Accessed: 2026-09-15.

[38] vLLM v0.15.0, “vLLM: Easy, fast, and cheap LLM serving for everyone.” https://docs.vllm.ai/en/v0. 15.0/, 2026. Accessed: 2026-09-15.