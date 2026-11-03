# PARALLELISM-ALIGNED PRUNING AND ROUND-AWARE CONTEXT REUSE FOR ON-DEVICE MULTIMODAL AGENTS

Peipei Jiao, Mingkang Liu, Weiming Meng, Lichen Pang, Tianyuan Liu, and Junyan L

Juhaokan Technology Co., Ltd.

cliff.pan@hisense.com

Qingdao, China

## ABSTRACT

On-device multimodal agents are limited by two deployment mismatches: pretrained attention structures may not divide evenly across the compute cores of an edge accelerator, causing irregular partitions; and standard prefix caching keeps historical visual tokens in the persistent KV cache, whose capacity grows with the number of rounds. We present PARC, which combines parallelismaligned structured pruning with lifetime-aware KV-cache reorganization. At the model level, it uses first-order Taylor importance and grouped-query-attention-aware selection to make attention dimensions divisible by the target core count while preserving the query-to-key/value ratio and head dimension. At runtime, it separates the system prompt, persistent text history, and transient visual observations by lifetime, updating text history during tool execution so that historical images leave the persistent cache and more interaction rounds fit a fixed budget. On 2,062 GUI actions, the incremental interface shows no accuracy regression, and task-aware recovery training restores 98.7% of the action-type accuracy gap and the coordinate hit rate to the original level. Under seven-core deployment, hardware-aligned pruning reduces vision-encoder latency by 11.4%, improves decoding throughput by 23.8%, and lowers per-heap high-water memory by 12.7%; a parameter-matched control confirms that the speedup stems mainly from core alignment rather than parameter reduction.

Keywords on-device inference · multimodal agents · structured pruning · neural processing unit · KV cache · incremental prefill

## 1 Introduction

Vision–language models are evolving from static visual question answering into interactive agents that perceive an environment, generate structured actions, invoke tools, and respond to new observations, as in embodied navigation, mobile automation, and multi-round visual dialogue. Compared with offline inference, such applications jointly require low time to first token (TTFT), high decoding throughput, accurate spatial grounding, and bounded device memory.

Two mismatches limit on-device efficiency. First, general-purpose architectures are not necessarily aligned with the parallel resources of an edge accelerator: when a compiler partitions attention heads across compute cores, a non-divisible dimension creates uneven shards, and the slowest shard determines the synchronization latency. Second, standard prefix caching treats the prefix as a contiguous sequence, so historical visual observations either keep consuming persistent cache capacity or, once removed, break direct reuse of the following text state; the KV-cache pressure thus grows with the number of rounds.

Existing pruning methods optimize parameter count or FLOPs and rarely treat the discrete core count as a structural constraint, while KV-cache research focuses on prefix reuse, eviction, or capacity rather than the differing lifetimes of prompts, text history, and images in tool-using agents. We therefore propose PARC (Parallelism-Aligned Pruning and Round-Aware Context Reuse), illustrated in Fig. 1.

The contributions of this work are threefold:

![](images/870a65e6feb1abfa27117b819d2d493fdf33fd741df0636a7bbe09fcf6dfb2b4.jpg)

![](images/41c8f3fff805f05be78340018f3c1d51f745cbb2618fe32a81fbbef84be176b0.jpg)  
Figure 1: Overview of PARC. Model dimensions are aligned with available cores, while fixed prompts and roundspecific inputs are managed according to their lifetimes.

• We formulate grouped-query-attention (GQA)-aware head pruning under a hardware parallelism constraint, preserving the GQA ratio, per-head dimension, and rotary-position compatibility while producing more regular partition candidates and reducing the risk of uneven shards or local serialization.

• We introduce a round-aware inference schedule that separates persistent text from transient images and commits new text history during tool execution, thereby reorganizing KV-cache state by context lifetime, reducing persistent visual-token occupancy, and extending the feasible interaction length under a fixed cache budget.

• We conduct deployment experiments on a seven-core edge NPU and systematically evaluate direct pruning, recovery training, and multi-round incremental inference on 2,062 GUI actions, using accuracy and device measurements to quantify effects on visual encoding, context prefill, autoregressive decoding, and memory usage.

## 2 Related Work

## 2.1 Structured Pruning and Hardware-Aware Compression

Unstructured methods such as SparseGPT remove individual weights and can achieve high sparsity [2]. Their realized speedup, however, depends on sparse storage formats and specialized kernels. Structured pruning removes complete channels, layers, heads, or tensor groups and retains dense regular tensors that are more readily supported by existing conversion and operator-fusion pipelines.

The Taylor criterion estimates the loss change caused by removing a structure using parameter–gradient products. DepGraph and Torch-Pruning explicitly track operator dependencies so that coupled input and output dimensions are pruned consistently [3]. Prior studies commonly target FLOPs, parameter count, or a uniform pruning ratio. PARC instead introduces the accelerator core count into the feasible set of attention dimensions and optimizes partition regularity in addition to model size.

## 2.2 KV-Cache Optimization

Autoregressive transformers create key and value tensors during prefill and repeatedly read them during decoding [1]. H2O retains recent and high-importance tokens [4], while SnapKV selects salient cache positions from an observation window near the end of a prompt [5]. PagedAttention improves memory utilization through non-contiguous logical pages [6], and Gist Tokens compress long prompts into reusable representations [7]. These methods mainly decide which cache entries to retain or how to store them. PARC is complementary: it changes prefill boundaries and separates fixed, accumulating, and transient context segments.

## 2.3 Interactive Multimodal Agents

ReAct interleaves language reasoning with external actions [8]. Platforms such as LEGENT provide continuous perception, language interaction, action execution, and trajectory generation [9]. Most related systems focus on agent capability and environment construction, with inference performed on server GPUs. On-device deployment introduces a tighter coupling among visual encoding, text prefill, cache storage, and tool scheduling. PARC addresses this systems-level coupling without assuming a specific graphical interface.

## 3 Problem Formulation

## 3.1 Model–Hardware Parallelism Mismatch

Consider a multimodal transformer deployed on an accelerator with C available cores. The model exposes several potentially parallel dimensions, including language query heads $H _ { q }$ , language key/value heads $H _ { k v }$ , and vision heads $H _ { v }$ . The compiler may partition one or more of these dimensions across the cores. When a partition dimension is not divisible by C, the resulting workload can contain an uneven final shard. For example, assigning eight equal-cost groups to seven cores requires one core to execute two groups or requires the compiler to introduce padding. Because synchronized operators complete at the pace of the slowest core, reducing the average work is insufficient if the critical shard remains unchanged.

The structural objective is therefore not merely to minimize the number of heads. Given an admissible accuracy loss ϵ, we seek target dimensions

$$
\mathcal { H ^ { \star } } = \left( H _ { q } ^ { \star } , H _ { k v } ^ { \star } , H _ { v } ^ { \star } \right)\tag{1}
$$

that minimize measured device latency while satisfying divisibility, GQA consistency, and accuracy constraints:

$$
\begin{array} { r } { H _ { q } ^ { \star } \bmod { C } = 0 , } \\ { } \\ { H _ { v } ^ { \star } \bmod { C } = 0 , } \\ { \Delta \mathcal { A } ( \mathcal { H } ^ { \star } ) \leq \epsilon . } \end{array}
$$

$$
\begin{array} { c } { { H _ { k v } ^ { \star } \mathrm { ~ m o d ~ } C = 0 , } } \\ { { H _ { q } ^ { \star } } } \\ { { H _ { k v } ^ { \star } } } \end{array}\tag{2}
$$

Here, A denotes task accuracy. The final constraint cannot be established from structure alone and motivates recovery fine-tuning and task-level evaluation.

## 3.2 Redundant Multi-Round Prefill

Let $L _ { P } , L _ { E } ^ { ( r ) } , L _ { q } ^ { ( r ) }$ , and $L _ { a } ^ { ( r ) }$ be the token lengths of the system prompt, visual observation, request, and response at round r. If every round serializes the complete multimodal history without separating transient images from persistent text, the number of tokens presented to prefill over R rounds is approximately

$$
N _ { \mathrm { f u l l } } = \sum _ { r = 1 } ^ { R } \left[ L _ { P } + \sum _ { i = 1 } ^ { r } L _ { E } ^ { ( i ) } + \sum _ { i = 1 } ^ { r } L _ { q } ^ { ( i ) } + \sum _ { i = 1 } ^ { r - 1 } L _ { a } ^ { ( i ) } \right] .\tag{3}
$$

The repeated-image term grows quadratically with R when the per-round observation length is approximately constant. This growth is particularly expensive because a visual observation must first pass through the vision encoder and is then represented by many visual tokens in the language model.

With successful prefix reuse and transient visual observations, the number of newly prefilled tokens is approximated by

$$
N _ { \mathrm { P A R C } } \approx L _ { P } + \sum _ { r = 1 } ^ { R } \left( L _ { E } ^ { ( r ) } + L _ { q } ^ { ( r ) } + L _ { a } ^ { ( r ) } \right) .\tag{4}
$$

This counts newly computed prefill tokens; previously cached text still participates in attention, so cache reads and KV capacity still grow with conversation length.

![](images/f392c2336b72d5f72106b142e814f607d0f412f9471e074d1b533d7b1c47f4ce.jpg)  
Figure 2: Illustrative non-divisible and aligned mappings. The method generalizes to other core counts and partition granularities.

## 3.3 KV-Cache Capacity Under a Fixed Budget

Let B denote the KV-cache capacity in tokens, with approximately constant per-round lengths $L _ { E } , L _ { q } , L _ { a } .$ . Retaining all historical images bounds the number of complete rounds by $R _ { \mathrm { f u l l } } = { \bar { \lfloor ( B - L _ { P } ) / ( { \bar { L } } _ { E } + L _ { q } + L _ { a } ) \rfloor } }$ , whereas keeping only the current image resident gives $R _ { \mathrm { P A R C } } = \lfloor ( B - L _ { P } - L _ { E } ) / ( \bar { L } _ { q } + L _ { a } ) \rfloor$ . For example, with $B = 3 0 7 2 .$ $\bar { L _ { P } } \bar { = } \bar { 1 } 0 2 \bar { 4 } , \bar { L _ { E } } = 2 2 0$ , and $\phantom { 0 } \overline { { L } } _ { a } = 1 0 0$ , full retention supports six rounds and transient retention up to 18. This is an analytical capacity illustration, not a measured device limit, and is distinct from the measured device-memory footprint reported in Table 3.

## 3.4 Joint Objective

PARC targets foreground interaction latency, decomposed per round into vision, prefill, decode, and scheduling terms. Structural pruning primarily affects the first three, whereas round-aware reuse reduces redundant prefill and overlaps history maintenance with tool execution; validating the method therefore requires jointly measuring accuracy, per-stage latency, decoding throughput, and cache usage.

## 4 Method

## 4.1 Parallelism Constraint

Let H be the number of approximately equal-cost attention partitions and $C$ the number of available accelerator cores. Under static uniform partitioning, the slowest core processes $\lceil H / C \rceil$ partitions. We define the discrete partition utilization as

$$
\eta ( H , C ) = { \frac { H } { C \lceil H / C \rceil } } .\tag{5}
$$

If H is not divisible by C, some cores receive an additional partition, so a target $H ^ { \star }$ with $H ^ { \star }$ mod $C = 0$ is a candidate for regular mapping (Fig. 2). Equation (5) is a structural screening metric, not a predicted end-to-end speedup; the realized mapping must still be verified from the compiled graph.

## 4.2 GQA-Aware Language-Head Pruning

Let $H _ { q }$ and $H _ { k v }$ denote the original query- and key/value-head counts, with GQA ratio $g = H _ { q } / H _ { k v }$ . For C available cores, PARC selects

$$
H _ { k v } ^ { \star } = m C , \qquad H _ { q } ^ { \star } = g H _ { k v } ^ { \star } ,\tag{6}
$$

where m is a positive integer selected under the accuracy or latency budget. This construction makes both target dimensions divisible by C while preserving g.

For query head h in layer l, we use the first-order Taylor score

$$
I _ { l , h } = \operatorname* { m e a n } \left| W _ { l , h } ^ { Q } \odot \frac { \partial \mathcal { L } } { \partial W _ { l , h } ^ { Q } } \right| .\tag{7}
$$

Scores are aggregated over the g query heads associated with each KV head. The highest-scoring KV groups are retained subject to (6). The output dimensions of $W ^ { Q } , W ^ { K }$ , and $W ^ { V }$ and the corresponding input dimension of $W ^ { O }$ are sliced consistently. When gradients are unavailable, the score falls back to the $\ell _ { 2 }$ norm. Layer count, residual width, vocabulary, and per-head rotary dimension remain unchanged.

## 4.3 Vision-Head Pruning

For an original vision-head count $H _ { v }$ , the target is $H _ { v } ^ { \star } = n C$ , where n is a positive integer and $H _ { v } ^ { \star } \le H _ { \iota }$ . Heads are ranked using the magnitude of their query projection. The corresponding rows of the fused QKV projection and columns of the output projection are removed together. The per-head dimension and cross-modal output width remain unchanged, limiting changes to downstream interfaces.

## 4.4 Dependency-Consistent Pruning Procedure

Head removal changes multiple coupled tensor dimensions. For each language layer, PARC aggregates query-head importance within the original GQA groups and retains the highest-scoring KV groups with their query heads; with per-head dimension $d _ { h }$ , the retained indices slice $W ^ { Q } / W ^ { O }$ to $H _ { q } ^ { \star } d _ { h }$ and $W ^ { K } / W ^ {  V }$ to $H _ { k v } ^ { \star } d _ { h }$ , together with any bias vectors. For a vision block with a fused projection $W ^ { Q K V } \in \mathrm { ~ \mathbb { R } ^ { 3 H _ { v } d _ { v } \times d _ { m o d e l } } ~ }$ , the same selected-head indices are applied to the query, key, and value ranges and to the matching output-projection columns. This preserves the block output width, leaving the residual path and cross-modal connector unchanged. After pruning, PARC updates the model metadata and validates the result in three stages: projection shapes and divisibility, a finite floating-point forward pass, and successful target-backend conversion of all modified operators. These checks distinguish a mathematically valid head selection from a deployable model artifact.

## 4.5 Backend Conversion and Deployment Adaptation

The pruned floating-point checkpoint is first validated in the source framework and is then converted and quantized for the target backend. The key requirement is backend independent: because the query, KV, and vision projection widths differ from the original architecture, any target runtime must read the updated configuration and preserve the pruned tensor dimensions and the fixed per-head encoding dimensions, rather than reconstructing them from the original model shape (for example, by inferring the per-head dimension from hidden width divided by head count). Platform-specific conversion details for the RKNN/Qwen3-VL path used in this work are given in Section 5.

## 4.6 Target Selection and Recovery

The feasible targets in (2) form a discrete set. For a fixed core count, PARC enumerates candidates that preserve the original GQA ratio and per-head dimension. Each candidate can be screened using parameter count and the utilization proxy in (5), but the final choice should be based on measured latency and task accuracy. An important control is a model with a similar parameter count but a non-divisible head configuration; it separates the benefit of removing parameters from the benefit of producing regular partitions.

Removing attention heads changes the learned subspace even when tensor interfaces remain compatible. We therefore treat recovery fine-tuning as part of the complete deployment pipeline rather than an optional post-processing step. Either low-rank adaptation [11] or full-parameter training can be used on structured action, spatial-grounding, and tool-use examples. The recovery set should contain coordinate-sensitive examples because the results in Section 5 show that localization is substantially more fragile than coarse action recognition.

## 4.7 Round-Aware Incremental Context Reuse

Let P denote the system prompt, and let $E ^ { ( r ) } , q ^ { ( r ) }$ , and $a ^ { ( r ) }$ denote the visual observation, user request, and model response at round r. The persistent textual history before round r is

$$
M ^ { ( r - 1 ) } = [ q ^ { ( 1 ) } , a ^ { ( 1 ) } , \ldots , q ^ { ( r - 1 ) } , a ^ { ( r - 1 ) } ] .\tag{8}
$$

P is fixed for the task, $M ^ { ( r - 1 ) }$ grows across rounds, and $E ^ { ( r ) }$ is transient. Past visual observations are not stored in persistent history.

![](images/a23a00b7af621e96075b16f1f95f1f96e9e4cf9e1641b3d6876a13f3c0bbcfd0.jpg)  
Figure 3: Alternating visible inference and masked text-only KV-cache reconstruction.

At initialization, the runtime prefills P once and stores $K V ( P )$ . At visible round $r ,$ the business layer submits $[ E ^ { ( r ) } , q ^ { ( r ) } ]$ in history-retention mode. The runtime reuses the retained text-only state $K V ( P , M ^ { ( r - 1 ) } )$ , appends the current visual and query tokens, and generates $a ^ { ( r ) }$ . The cache produced by this call is transient because it contains $E ^ { ( r ) }$

After $a ^ { ( r ) }$ is produced, the business layer starts the requested tool and serializes the complete cumulative text history $M ^ { ( r ) } = [ \acute { q } ^ { ( 1 ) } , a ^ { ( 1 ) } , \ldots , q ^ { ( r ) } , a ^ { ( r ) } ]$ . During the tool-execution window, it submits $M ^ { ( r ) }$ in a text-only maintenance call in text-cache reconstruction mode with minimal generation. In this mode, the previous cache is not retained as the final session state; nevertheless, the runtime performs token-prefix matching and directly reuses the matching prefix KV entries, computing only the unmatched suffix. The call therefore replaces the transient image-bearing state with $K V ( P , M ^ { ( r ) } )$ and removes $E ^ { ( r ) }$ from the persistent cache. Its output is masked and is not exposed as another assistant response. The next visible call returns to history-retention mode with $[ E ^ { ( r + 1 ) } , q ^ { ( r + 1 ) } ]$ , and the procedure repeats. Figure 3 summarizes the state transition

Through prefix matching, this schedule avoids recomputing the system prompt, previous images, and committed text, and prevents historical visual tokens from accumulating in the persistent cache, freeing budget for more interaction rounds. It does not eliminate textual-history growth, so the realized benefit depends on prefix-hit rate, visual-token count, and tool duration.

## 4.8 Runtime Workflow and State Management

The prototype has four logical layers: an application layer (screen capture, request formatting, action execution, task boundaries), a round-aware scheduler that owns the textual history and chooses between visible inference and masked maintenance, the multimodal runtime exposing session-level KV state, and the target backend on the NPU. As shown in Fig. 3, a visible round submits the current image and request and releases the image buffer after vision inference, while a masked maintenance call during tool execution reconstructs the text-only persistent state; the new logical round is committed only after it succeeds. Two details are essential for correctness: the appended history must use exactly the same tokenizer and chat delimiters as visible inference, or prefix matching fails; and the authoritative history stays in the application layer, so the runtime cache can be rebuilt after any error or eviction.

## 4.9 Cache-State Invariants and Failure Handling

Immediately after visible inference, the transient cache may contain $E ^ { ( r ) }$ . After the masked maintenance call, the logical persistent state must represent

$$
\begin{array} { r } { K ^ { \left( r \right) } = K V \Big ( P , M ^ { \left( r \right) } \Big ) , } \end{array}\tag{9}
$$

while no previous observation $E ^ { ( i ) } , i \leq r .$ , is part of the persistent logical input. Equation (9) is a post-maintenance state invariant: a new visual observation may be processed transiently, but the scheduler returns to a text-only persistent state before the next round. This distinction prevents image history from growing without removing textual evidence required for subsequent decisions.

The maintenance call is useful only when it overlaps tool execution: if the tool finishes first, the next observation waits for the update or a synchronized fallback; if prefix matching fails or inference is interrupted, the state is rebuilt from the authoritative history and the logical round is not advanced until the update succeeds. At task completion, cancellation, or eviction, the business layer clears both the runtime KV state and its history metadata, avoiding silent divergence between the visible conversation and the internal cache.

## 5 Experiments and Results

## 5.1 Experimental Setup

We use a task-adapted Qwen3-VL-4B checkpoint [10], obtained by merging a balanced mixture of GUI-task LoRA adapters, and deploy it to an RK1828 edge NPU. The original language module contains 32 query heads and eight KV heads, and the vision encoder contains 16 attention heads. PARC changes these counts to 28, seven, and 14, respectively. Taylor importance is estimated on 128 WikiText-2 calibration sequences of length 2,048. The resulting checkpoint retains all 36 language layers and contains 4.31 billion parameters, compared with 4.4378 billion before pruning, a reduction of approximately 128 million parameters (2.9%).

The checkpoint is converted and quantized using the RKNN toolchain under a W4A16 configuration (4-bit weights and 16-bit activations). Unless stated otherwise, the “Pruned” model in the device measurements of Tables 2 and 3 refers to the deployed pruned-and-recovered checkpoint after this conversion and quantization, not the directly pruned model without recovery. For this Qwen3-VL implementation, the vision QKV and output projections use the explicit retained width $H _ { v } ^ { \star } d _ { v }$ rather than the original hidden width. The vision RoPE path reads the fixed 64-dimensional head size from the updated configuration, avoiding the invalid inference 1024/14 after pruning, while the rotary dimension remains 32; the text RoPE path also handles a missing rope\_scaling field. The runtime’s history-control flag maps history-retention and text-cache reconstruction modes to values 1 and 0, respectively. Before task evaluation, identical text-only and image–text inputs are used to check the source and converted models for functional consistency.

In this work, recovery is performed with LoRA applied to all linear projections, using rank 8, scaling factor 16, and dropout 0.05. We train for two epochs with a cosine schedule, a peak learning rate of $\mathrm { 1 \times 1 0 ^ { - 5 } }$ , 50 warmup steps, an effective batch size of 128, a maximum sequence length of 8,192, and bf16 precision. The recovery data is disjoint from the 2,062 evaluation actions. In our runs, the procedure is not sensitive to the recovery-set size: training on 5,000 and on 50,000 examples yields comparable recovered accuracy.

We compare the original, directly pruned, and recovered models using both one-shot inference and the multi-round GUI incremental interface. All accuracy runs use the same generation configuration.

The test set contains 2,062 GUI-action steps evaluated under the multi-round GUI workflow and covers click, application launch, text entry, wait, back, and swipe actions. We report action-type accuracy, coordinate hit rate, and exact-match accuracy. Coordinate hit rate evaluates whether a predicted position falls inside the target region or satisfies a distance threshold. The configured distance threshold is 50.0. It is computed over samples with a valid coordinate prediction, whose count is reported in parentheses. For each model checkpoint, the same action set and generation configuration are evaluated using both conventional one-shot inference and the GUI incremental workflow. The paired aggregate metrics therefore measure whether multi-round state management changes task accuracy.

## 5.2 Multi-Round GUI Accuracy

As shown in Table 1, all aggregate metrics differ by at most 0.5 percentage points between one-shot and multi-round GUI inference across the three model checkpoints, indicating no material accuracy regression from the multi-round interface. The eight-core baseline in Table 1 is close to the original model’s seven-core results, so it is reported as a deployment reference rather than evidence that core count changes model accuracy.

Table 1: Accuracy on 2,062 GUI Actions
<table><tr><td>Model</td><td>NPU Cores</td><td>Interface</td><td>Type Acc.</td><td>Coord. Hit</td><td>Exact Match</td></tr><tr><td>Original</td><td>8</td><td>Baseline</td><td>83.4</td><td>85.2 (757)</td><td>77.9</td></tr><tr><td>Original</td><td>7</td><td>One-shot</td><td>83.1</td><td>85.7 (759)</td><td>77.3</td></tr><tr><td>Original</td><td>7</td><td>Multi-round GUI</td><td>83.1</td><td>85.7 (759)</td><td>77.4</td></tr><tr><td>Pruned</td><td>7</td><td>One-shot</td><td>67.1</td><td>36.9 (759)</td><td>43.8</td></tr><tr><td>Pruned</td><td>7</td><td>Multi-round GUI</td><td>67.4</td><td>36.4 (758)</td><td>43.8</td></tr><tr><td>Recovered</td><td>7</td><td>One-shot</td><td>83.0</td><td>86.3 (759)</td><td>71.5</td></tr><tr><td>Recovered</td><td>7</td><td>Multi-round GUI</td><td>82.9</td><td>85.8 (758)</td><td>71.2</td></tr></table>

![](images/5a7f2523774b50f0ec3045bf8b3daf731cde58a7b34227ea7d5f44d3a8a08266.jpg)  
Figure 4: Incremental-interface accuracy before pruning, after direct pruning, and after recovery training. Coordinate hit rate is conditional on a valid coordinate prediction.

Direct pruning causes a substantial loss. Under the incremental interface, action-type accuracy, coordinate hit rate, and exact match decrease by 15.7, 49.3, and 33.6 percentage points, respectively. Recovery training raises these metrics to 82.9%, 85.8%, and 71.2%. Relative to the original incremental model, it closes 98.7% of the action-type accuracy gap and 81.5% of the exact-match gap. Under the incremental interface, the recovered coordinate hit rate reaches 85.8%, essentially matching the original model’s 85.7%, so coordinate localization is restored to the original-model level. The residual gap relative to the original model is concentrated in exact match (71.2% versus 77.4%) rather than in action-type accuracy or coordinate hit rate. These results show that hardware-aligned structure alone does not preserve model ability: direct pruning affects spatial grounding more strongly than aggregate action recognition, but recovery training closes both the action-type and coordinate gaps. Structural regularity and task accuracy must therefore be treated as separate objectives, with task-relevant recovery training included in the deployment pipeline.

## 5.3 Model Inference Performance

We report system-prompt loading time, visual encoding latency, language-model prefill throughput, and autoregressive decoding throughput. For $N _ { \mathrm { o u t } }$ generated tokens and decoding duration T<sub>decode</sub>,

$$
\mathrm { T P S } = \frac { N _ { \mathrm { o u t } } } { T _ { \mathrm { d e c o d e } } } , \qquad \mathrm { T P O T } = \frac { T _ { \mathrm { d e c o d e } } } { N _ { \mathrm { o u t } } } .\tag{10}
$$

Table 2 reports RKNN inference measurements for the original model using eight and seven NPU cores, for the aligned pruned model (28/7/14 heads) using seven cores, and for a parameter-matched non-aligned control described below. Each reported value is the mean of ten repeated runs under the same configuration.

Table 2: RKNN Model Inference Performance
<table><tr><td>Model</td><td>NPU cores</td><td>Params</td><td>System prompt (s)</td><td>Vision input</td><td>ViT (ms)</td><td>Prefll (tok/s)</td><td>Decode (tok/s)</td></tr><tr><td>Original</td><td>8</td><td>4.4378B</td><td>0.848</td><td> $6 4 0 \times 3 5 2$ </td><td>184.84</td><td>729.59</td><td>79.17</td></tr><tr><td>Original</td><td>7</td><td>4.4378B</td><td>1.24</td><td> $6 4 0 \times 3 5 2$ </td><td>223.67</td><td>650.59</td><td>62.32</td></tr><tr><td>Aligned pruned</td><td>7</td><td>4.31B</td><td>1.04</td><td> $6 4 0 \times 3 5 2$ </td><td>198.28</td><td>651.04</td><td>77.12</td></tr><tr><td>FFN-pruned (non-aligned)</td><td>7</td><td>4.3140B</td><td>1.23</td><td> $6 4 0 \times 3 5 2$ </td><td>225.72</td><td>655.00</td><td>62.89</td></tr></table>

Table 3: NPU Heap High-Water Usage During Inference
<table><tr><td>Heap</td><td>Original (MB)</td><td>Pruned (MB)</td><td>Reduction (%)</td></tr><tr><td>n0</td><td>524.93</td><td>470.39</td><td>10.4</td></tr><tr><td>n1</td><td>476.71</td><td>416.72</td><td>12.6</td></tr><tr><td>n2</td><td>480.98</td><td>415.42</td><td>13.6</td></tr><tr><td>n3</td><td>481.64</td><td>410.54</td><td>14.8</td></tr><tr><td>n4</td><td>474.45</td><td>410.56</td><td>13.5</td></tr><tr><td>n5</td><td>468.04</td><td>409.93</td><td>12.4</td></tr><tr><td>n6</td><td>462.80</td><td>409.28</td><td>11.6</td></tr><tr><td>n7</td><td>0.18</td><td>0.18</td><td>0.0</td></tr><tr><td>Total</td><td>3369.73</td><td>2943.03</td><td>12.7</td></tr></table>

Relative to the original seven-core configuration, the aligned pruned seven-core model reduces ViT latency from 223.67 to 198.28 ms (11.4%) and increases decoding throughput from 62.32 to 77.12 tokens/s (23.8%). Prefill throughput is essentially unchanged (650.59 to 651.04 tokens/s); prefill is compute-bound and is therefore far less sensitive to the head-count reduction than decoding, which explains why it does not improve.

To separate the effect of parameter reduction from the effect of core alignment, we add a parameter-matched non-aligned control. Starting from the original model, this control keeps the attention head configuration unchanged (32 query and eight key/value heads, which are not divisible by seven), and instead reduces parameters only in the feed-forward intermediate dimension using the same first-order Taylor criterion, reaching 4.3140 billion parameters, essentially the same size as the aligned pruned model (4.31 billion). Despite this matched parameter budget, the non-aligned control attains only 62.89 tokens/s of decoding throughput and 225.72 ms of ViT latency, close to the original seven-core model and far from the aligned pruned model’s 77.12 tokens/s and 198.28 ms. Because the two pruned models share essentially the same parameter count and differ mainly in whether the head count is divisible by the core count, the roughly 22.6% decoding-throughput gap between them is attributable to core alignment rather than to parameter reduction. This supports the central claim that aligning attention dimensions with the core count, not merely reducing head count, drives the measured speedup. Divisibility remains a structural precondition for regular partitioning. The current RKNN toolchain does not expose per-core execution traces, so we rely on the parameter-matched control above together with the per-heap memory distribution in Section 5 as indirect evidence of the mapping, and leave direct per-core tracing to future work.

## 5.4 Memory Usage

Before model loading, the max used size of every NPU heap is 96 bytes, providing a cleared baseline for both measurements. Table 3 reports the per-heap high-water values observed during inference.

The aggregate of the per-heap high-water values decreases by 426.70 MB, from 3.370 to 2.943 GB, corresponding to a 12.7% reduction under the reported RKNN memory metric. All compute heaps n0–n6 show reductions of 10.4–14.8%, while the small n7 management heap remains unchanged. The inference load is carried by the seven compute heaps n0–n6 with comparable footprints, while the management heap n7 stays negligible, which is consistent with execution partitioned across the seven cores; this memory view is a coarse indicator only, and direct per-core execution traces remain the definitive way to confirm the mapping. Because maxima from different heaps are not guaranteed to occur at exactly the same instant, the aggregate should be interpreted as a sum of per-heap high-water marks rather than a synchronized process-level peak.

Table 4: Per-Round Runtime Behavior on Seven Cores (Aligned Pruned Model)
<table><tr><td>Round</td><td>Strategy</td><td>Vision</td><td>Input tok.</td><td>TTFT (ms)</td><td>Prefill (tok/s)</td></tr><tr><td rowspan="3">1</td><td>full</td><td>640 × 352</td><td>238</td><td>302.09</td><td>787.85</td></tr><tr><td>round-aware</td><td>640 × 352</td><td>238</td><td>302.48</td><td>786.84</td></tr><tr><td>round-aware (mask)</td><td></td><td>627</td><td>140.48</td><td>4463.43</td></tr><tr><td rowspan="3">2</td><td>full</td><td>640 × 352</td><td>236</td><td>321.30</td><td>734.51</td></tr><tr><td>round-aware</td><td>640 × 352</td><td>236</td><td>310.69</td><td>759.59</td></tr><tr><td>round-aware (mask)</td><td></td><td>644</td><td>144.23</td><td>4465.18</td></tr><tr><td rowspan="3">3</td><td>full</td><td>640 × 352</td><td>240</td><td>332.09</td><td>722.70</td></tr><tr><td>round-aware</td><td>640 × 352</td><td>240</td><td>314.18</td><td>763.90</td></tr><tr><td>round-aware (mask)</td><td></td><td>665</td><td>146.30</td><td>4545.36</td></tr><tr><td rowspan="3">4</td><td>full</td><td>640 × 352</td><td>235</td><td>338.95</td><td>693.32</td></tr><tr><td>round-aware</td><td>640 × 352</td><td>235</td><td>314.38</td><td>747.50</td></tr><tr><td>round-aware (mask)</td><td></td><td>681</td><td>149.25</td><td>4562.75</td></tr><tr><td rowspan="3">5</td><td>full</td><td>640 × 352</td><td>241</td><td>353.36</td><td>682.02</td></tr><tr><td>round-aware</td><td>640 × 352</td><td>241</td><td>315.91</td><td>762.87</td></tr><tr><td>round-aware (mask)</td><td></td><td>703</td><td>151.97</td><td>4625.97</td></tr><tr><td rowspan="3">6</td><td>full</td><td>640 × 352</td><td>236</td><td>359.70</td><td>656.10</td></tr><tr><td>round-aware</td><td>640× 352</td><td>236</td><td>315.55</td><td>747.89</td></tr><tr><td>round-aware (mask)</td><td></td><td>720</td><td>154.89</td><td>4648.52</td></tr></table>

## 5.5 Round-Aware Runtime Behavior

To examine how the round-aware schedule behaves across rounds, we run the aligned pruned model on seven cores under two strategies and track both the feasible number of rounds and the per-round cost. In the full-history strategy, each visible round prepends the accumulated history, including all previous images, to the current image and query. In the round-aware strategy, each visible round processes only the current image and query, while the growing textual history is committed by a masked maintenance call issued during the tool-execution window; the maintenance call is labeled “round-aware (mask)” and produces no visible response. Table 4 reports, for the first six rounds, the per-round visual input, prefill time (used here as the time to first token, TTFT), and prefill throughput.

The primary benefit of the schedule is capacity, not per-round speed. Under the full-history strategy, every visible round retains all previous images in the persistent KV cache, which therefore fills quickly: on this device the cache budget is exhausted at the sixth round, so roughly six rounds are the practical limit. Under the round-aware strategy, historical visual tokens are removed from the persistent cache and only the current image is resident, so the same fixed cache budget sustains about twenty interaction rounds, more than a 3× increase, without changing the per-round inference procedure. This measured six-versus-twenty-round gap confirms the token-budget analysis in Section III-C on the actual device. Crucially, this capacity gain does not cost extra per-round latency. The visible-round TTFT of the round-aware strategy stays essentially flat (302.48 to 315.55 ms) and matches that of full history at the rounds where full history is still feasible, so users see no slowdown. The growing textual history is committed by the masked maintenance call issued during tool execution; its input grows across rounds while its prefix is reused (its prefill throughput exceeds 4,400 tokens/s, indicating that only the unmatched suffix is recomputed), and because it runs inside the tool-execution window its cost is hidden from the user-facing path rather than added to it. In short, the schedule extends the feasible interaction length under a fixed cache budget while keeping foreground latency unchanged.

## 5.6 Discussion and Limitations

PARC combines two optimizations that act at different stages: parallelism-aligned pruning changes model execution, whereas lifetime-aware KV-cache reorganization reduces persistent visual-token retention and repeated multimodal prefill. Explicit KV-cache occupancy and prefix-hit-rate readouts, not exported by the current toolchain, would further quantify the round-aware schedule, and we leave these measurements to future work.

The present study covers one 4B-scale vision–language model, one edge-NPU platform, and GUI actions. Device latency and throughput are reported as the mean of ten repeated runs per configuration; the parameter-matched nonaligned control in Section 5 already isolates core alignment from parameter reduction as the main source of the measured speedup, and direct per-core execution traces, which the current toolchain does not expose, would provide further confirmation of the physical mapping. The dataset is also imbalanced across action types, so aggregate accuracy can conceal minority-class behavior; class-wise results or balanced metrics would provide a more complete assessment.

Finally, round-aware reuse does not eliminate textual-history growth, and it assumes that request–response text is sufficient after each visual round. Tasks that require details visible only in an earlier image may need selective visual-memory retention or a textual visual summary, introducing an additional accuracy–memory trade-off.

## 6 Conclusion

We introduced PARC, a framework that jointly addresses irregular core partitioning and KV-cache lifetime mismatch through parallelism-aligned structured pruning and round-aware context reuse. By retaining only textual history between visual rounds, PARC also reduces persistent KV-cache occupancy and increases the feasible number of interactions under a fixed cache budget. On 2,062 GUI actions evaluated in the multi-round workflow, the incremental interface produces accuracy comparable to one-shot inference, while direct pruning without recovery training substantially degrades fine-grained localization. Recovery training then closes 98.7% and 81.5% of the action-type and exact-match gaps, restoring coordinate localization to the original-model level.

In the seven-core edge-NPU comparison, the aligned pruned model reduces ViT latency from 223.67 to 198.28 ms and increases decoding throughput from 62.32 to 77.12 tokens/s, while prefill throughput remains nearly unchanged. Aggregate per-heap high-water memory usage decreases from 3.370 to 2.943 GB (12.7%). A parameter-matched nonaligned control, which prunes only the feed-forward dimension and keeps the original attention heads, reaches the same parameter budget (4.314 versus 4.31 billion) yet attains only 62.89 tokens/s, close to the unpruned model; this isolates core alignment, rather than parameter reduction alone, as the main source of the measured speedup. The framework is independent of a specific graphical interface and can be applied to other multimodal perception–reasoning–action loops.

## References

[1] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, “Attention Is All You Need,” in Advances in Neural Information Processing Systems (NeurIPS), 2017.

[2] E. Frantar and D. Alistarh, “SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot,” in Proc. 40th Int. Conf. Mach. Learn. (ICML), 2023.

[3] G. Fang, X. Ma, M. Song, M. B. Mi, and X. Wang, “DepGraph: Towards Any Structural Pruning,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2023.

[4] Z. Zhang, Y. Sheng, T. Zhou, T. Chen, L. Zheng, R. Cai, Z. Song, Y. Tian, C. Ré, C. Barrett, Z. Wang, and B. Chen, “H<sub>2</sub>O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models,” in Advances in Neural Information Processing Systems (NeurIPS), 2023.

[5] Y. Li, Y. Huang, B. Yang, B. Venkitesh, A. Locatelli, H. Ye, T. Cai, P. Lewis, and D. Chen, “SnapKV: LLM Knows What You Are Looking for Before Generation,” in Advances in Neural Information Processing Systems (NeurIPS), 2024.

[6] W. Kwon, Z. Li, S. Zhuang, Y. Sheng, L. Zheng, C. H. Yu, J. E. Gonzalez, H. Zhang, and I. Stoica, “Efficient Memory Management for Large Language Model Serving with PagedAttention,” in Proc. 29th Symp. Operating Syst. Principles (SOSP), 2023.

[7] J. Mu, X. L. Li, and N. Goodman, “Learning to Compress Prompts with Gist Tokens,” in Advances in Neural Information Processing Systems (NeurIPS), 2023.

[8] S. Yao, J. Zhao, D. Yu, N. Du, I. Shafran, K. Narasimhan, and Y. Cao, “ReAct: Synergizing Reasoning and Acting in Language Models,” in Int. Conf. Learn. Represent. (ICLR), 2023.

[9] Z. Cheng, Z. Wang, J. Hu, S. Hu, A. Liu, Y. Tu, P. Li, L. Shi, Z. Liu, and M. Sun, “LEGENT: Open Platform for Embodied Agents,” in Proc. 62nd Annu. Meeting Assoc. Comput. Linguistics: System Demonstrations, Bangkok, Thailand, 2024, pp. 335–345.

[10] S. Bai et al., “Qwen3-VL Technical Report,” arXiv:2511.21631, 2025.

[11] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen, “LoRA: Low-Rank Adaptation of Large Language Models,” in Int. Conf. Learn. Represent. (ICLR), 2022.