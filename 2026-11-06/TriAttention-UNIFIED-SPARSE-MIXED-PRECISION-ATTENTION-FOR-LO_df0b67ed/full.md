# TriAttention: UNIFIED SPARSE MIXED-PRECISION ATTENTION FOR LONG-CONTEXT LLM INFERENCE

Hanzhi Zhang 1 Qiao Zhang2 Qinglei Cao2 Heng Fan1 Yan Huang 1 Kewei Sha3 Yunhe Feng 1

## ABSTRACT

Sparsity and quantization reduce the cost of attention by skipping interactions and lowering the precision of those retained. However, useful interactions differ in their precision requirements: some tolerate low-precision scores while remaining sensitive to removal. We therefore introduce TriAttention, which assigns query-key (QK) tiles to skipping, 4-bit, or 8-bit computation, using 4-bit to retain quantization-tolerant interactions at lower cost. To represent compactly, we encode each head's computation regions with two diagonal-distance boundaries, keeping policy storage independent of context length. Offline calibration then evaluates candidate boundary pairs using normalized projected-output errors and complete-tile execution costs. Using these measurements, a model-wide allocator selects each head's boundaries to minimize calibration error under a shared computation budget. During prefill, the selected boundaries generate precision-grouped traversals for a fused kernel that skips omitted tiles and combines contributions from both QK precisions into one normalized attention output. The quantized keys and values prepared during prefill are then reused in a compressed key-value (KV) cache for decoding, reducing repeated reads of prompt data. Experiments show that TriAttention achieves end-to-end generation speedups of up to 7.88× over the Torch attention and 2.65× over the BF16 baseline. Meanwhile, TriAttention maintains quality comparable to the original models on general-purpose and long-context tasks. These results show how a compact retention-and-precision policy translates local sensitivities into attention computation and repeated KV-cache access.The implementation is https://github.com/HanzhiZhang-Ulrica/TriAttention.

## 1 INTRODUCTION

Efficient deployment of large language models (LLMs) requires reducing inference latency while preserving model quality across conversational assistance, code generation, reasoning, and document summarization (Grattafiori et al. 2024; Bai et al., 2024). For attention, this involves both matrix computation during prompt processing (prefill) and repeated key-value (KV) cache reads during token generation (decode) (Dao et al., 2022; Yang et al., 2025). Sparsity reduces attention computation by skipping selected interactions, while quantization accelerates query-key (QK) multiplication through lower-precision arithmetic (Jiang et al., 2024; Zhang et al., 2025b). However, attention interactions differ in their sensitivity to skipping and reduced precision: some can be removed with little effect on the output, while other useful contributions tolerate lower QK precision but remain sensitive to removal. Model quality therefore depends on both which interactions are retained and the precision used to compute their contributions.

Figure 1 contrasts efficient attention strategies and illustrates their long-context QA and inference-throughput differences. Dense attention kernels reduce memory transfers and improve GPU utilization, yet still evaluate a quadratic number of causal interactions (Dao et al., 2022; Dao, 2024). To reduce the interaction count, structured sparse methods select attention patterns or spans, but retained regions still incur floating-point computation. Therefore, further savings from sparsity require excluding more context, and discarding useful interactions degrades model quality (Lai et al., 2025; Jiang et al., 2024; Fu et al., 2024; Zhu et al., 2025). Quantized attention increases QK multiplication throughput with lower-bit arithmetic. However, using a uniformly higher precision protects sensitive regions but spends unnecessary computation on regions that tolerate lower precision, while reducing precision to suit tolerant regions introduces excessive error in sensitive ones and degrades model quality (Zhang et al., 2025b; 2024a). Sparse quantized attention combines skipping with low-bit QK computation, yet retained regions still share one precision even though some require higher score accuracy and others tolerate further quantization (Zhang et al., 2025a). This leads to our central question: how can attention jointly allocate retention and precision for efficient GPU execution?

![](images/66cf92f577ef2e89d423bfc74661601aa5e5cc1796794b345c3b8b22ee1f3949.jpg)

![](images/e83ffc1b433f4fd50da385ff67f9e33668c85d860dc3176889ad56080683ffd9.jpg)  
Figure 1. Attention computation choices and inference performance. Top left: attention maps and token-interaction graphs compare dense, sparse, quantized, and sparse-quantized designs with TriAttention, which jointly assigns skipping, 4-bit, and 8-bit. Bottom left: LooGLE single-hop QA F1 at 64K. Right: throughput across input lengths. Both evaluations use Llama-3.2-3B-Instruct.

Our key insight is that attention regions differ in their sensitivity to skipping and reduced score precision. Exploiting these differences lets us preserve useful context at lower cost while reserving higher precision for sensitive regions. However, translating this flexibility into efficient inference requires resolving two coupled challenges. For allocation, comparable reductions in computation produce different output errors across layers and heads, so a compression level tolerated by some regions introduces excessive errors in less tolerant regions and degrades model quality. For execution, an explicit tile-action map grows quadratically with context length. Query-block traversal metadata also competes with staged K/V operands for shared memory.

To address these challenges, we introduce TriAttention, a sparse mixed-precision attention method that jointly assigns skipping, 4-bit, and 8-bit QK computation within each head. To control policy-metadata growth, we define each layer– head configuration using two diagonal-distance boundaries that describe entire computation regions with storage independent of context length. To compare sensitivity across layers and heads, offline calibration measures each candidate boundary pair's error after output projection and normalizes it by the layer's reference attention-output energy. We pair these errors with complete-tile costs, so allocation accounts for the full computation performed on retained tiles. Using these comparisons, a global allocator selects a boundary pair under a shared budget, preserving more computation in sensitive regions and concentrating savings in more tolerant ones. During prefill, the selected boundaries generate only the current query block's retained tile indices, keeping traversal metadata local to the work staged in shared memory. The fused kernel processes these indices in precision groups to reuse query operands, skips omitted tiles, and combines retained contributions into one normalized attention output. At prompt completion, the selected ranges determine which prepared quantized keys and values are packed into the decode cache, extending the allocation's savings from prefill to repeated prompt-data reads.

Across the evaluated workloads, TriAttention achieves endto-end generation speedups of up to 7.88× over models using Torch attention and 2.65× over the BF16 baseline. These efficiency gains are achieved while maintaining quality comparable to the original models on general-purpose tasks and long-context question answering.

## Our contributions are:

1. Three-Level QK Computation. We develop a tile-level organization of skipping, 4-bit, and 8-bit QK computation that matches graded sensitivity with graded computational cost. The shared tile unit makes retention and precision jointly selectable within each head.

2. Compact Distance Encoding. We represent each head's computation regions with two diagonal-distance boundaries, providing a common representation for offline calibration and local, precision-grouped GPU traversal while keeping policy storage independent of context length.

3. Sensitivity-Guided Global Allocation. We develop an offline allocation rule that compares normalized output errors and execution costs of alternative three-level assignments. A model-wide budget coordinates these choices across layers and heads, prioritizing computation where it most reduces projected-output error.

4. Cross-Phase Execution. Fused prefill executes the selected computation regions, while decoding reuses the prepared quantized keys and values in a compressed KV cache to reduce repeated prompt-data reads.

## 2 RELATED WORK

Efficient attention connects GPU execution, interaction selection, and numerical precision.

Hardware-efficient attention execution improves data reuse, work partitioning, and kernel specialization. Tiled fusion keeps score computation, online softmax, and value aggregation on chip, reducing intermediate transfers while dense query-key computation remains quadratic (Dao et al., 2022; Dao, 2024; Ye et al., 2025). Workload-aware partitioning of query blocks or KV segments across thread blocks improves occupancy and load balance, although KV splitting adds coordination and partial-output reduction (Dao, 2024; Sanovar et al., 2025; Ye et al., 2025). Asynchronous designs overlap transfers, matrix multiplication, and reduce partial-softmax synchronization to improve pipeline parallelism(Shah et al., 2024; Hong et al., 2024). Meanwhile, programmable systems specialize fused kernels for score transformations, masks, and KV-cache layouts, supporting attention variants while leaving interaction selection to applications (Dong et al., 2025; Ye et al., 2025).

Structured sparse attention selects tokens, spans, heads, or interaction blocks. Streaming windows retain initial attention sinks and recent tokens, bounding KV working sets but denying windowed heads direct access to excluded distant evidence (Xiao et al., 2024; 2025). For heterogeneous heads, calibration selects per-head spans or full-context retrieval and streaming roles, leaving retained QK precision fixed (Fu et al., 2024; Xiao et al., 2025). Input-dependent methods use sampled-query attention and lightweight block-importance estimates to select columns, diagonals, or blocks, adding online scoring and sparse-index construction (Jiang et al. 2024; Lai et al., 2025; Zhu et al., 2025; Xu et al., 2025). Estimated cumulative attention mass adapts retention per input but omits output errors from quantizing retained scores (Lai et al., 2025; Zhu et al., 2025; Xu et al., 2025). Learningbased methods distill block-importance predictors from attention maps, requiring extra predictor training changes (Gao et al., 2024b; Yuan et al., 2025).

Quantized and mixed-precision attention controls arithmetic precision and operand representations. Blockwise scales and integer kernels accelerate 8-bit QK, but outlier-dominated scales reduce resolution for smaller values (Zhang et al. 2025b; Kang et al., 2025). Smoothing and orthogonal transforms reduce outlier error, while finer groups, probability rescaling, and accumulation control enable lower-bit integer and floating-point attention with uniform QK precision within heads despite regional sensitivity differences (Zhang et al., 2024a; Shah et al., 2024; Zhang et al., 2026). Beyond scale selection, spatial mixed precision gives diagonal neighborhoods higher precision, yet computes all unmasked interactions, including low-impact ones (Ding et al., 2026; Sharratt, 2026). KV-cache quantization tailors scale axes, quantization levels, and outlier handling to key and value distributions, reducing bytes per entry while cached tokens and per-step reads grow with context (Liu et al., 2024; Hooper et al., 2024). Joint sparse-quantized systems add block skipping to low-bit QK, but retained regions share precision(Zhang et al., 2025a). Other systems combine selective KV access or eviction with cache quantization, reducing traffic but requiring online page or token scoring during generation (Yang et al., 2025; Zhang et al., 2024b).

## 3 PRELIMINARIES

Prefill computes attention for all prompt positions, allowing many queries to reuse each key and value. Decode processes one new query per request and rereads the accumulated KV cache each step. For a context of length $L ,$ dense attention evaluates $\mathcal { O } ( L ^ { 2 } )$ causal interactions during prefill and $\mathcal { O } ( L )$ per decode step (Yang et al., 2025).

Both phases use the same attention operation. Let $Q \in$ R $\phantom { - } ) L _ { q } \times \bar { d }$ contain the queries and $K , V \in \bar { \mathbb { R } } ^ { L _ { k } \times d }$ the keys and values, where d is the head dimension. The output is

$$
P = \mathrm { s o f t m a x } \left( \frac { Q K ^ { \top } } { \sqrt { d } } + M \right) , \qquad O = P V ,\tag{1}
$$

where M applies the causal mask and softmax operates. QK multiplication produces attention scores, and PV multiplication aggregates values using the attention probabilities in $P .$ Grouped-query attention (GQA) allows several query heads to share the same K and V (Yang et al., 2025).

Integer operands and scales. Integer quantization represents a group of values X through integer codes $\widehat { X }$ and a scale $s _ { X }$ , with $X \approx s _ { X } { \widehat { X } }$ (Zhang et al., 2025b). The corresponding operand scales restore the numerical magnitude of integer products. Products from different groups or precisions can then be accumulated in common units.

Tiled attention and partial aggregation. Fused attention kernels pair query and key blocks into tiles, stage operands in shared memory, and keep intermediate scores and probabilities on chip (Dao et al., 2022). For one query row, a subset of keys produces a partial state $( m _ { r } , l _ { r } , o _ { r } )$ : its maximum score, the sum of exponentiated scores relative to that maximum, and the correspondingly weighted value sum. Partial states over disjoint key subsets are combined by aligning their score maxima:

$$
\begin{array} { l } { m = \displaystyle \operatorname* { m a x } _ { r } m _ { r } , } \\ { l = \displaystyle \sum _ { r } \exp ( m _ { r } - m ) l _ { r } , } \\ { o = \displaystyle \sum _ { r } \exp ( m _ { r } - m ) o _ { r } , \qquad O = o / l . } \end{array}\tag{2}
$$

This rule supports both online accumulation across successive tiles and parallel reduction across key segments. The partials use common score and value units so that the denominator and numerator describe the same attention weights.

![](images/bdf879305f3dd19cf91d2171352b2b6ba300aee69fdcfb7b9590fc07355e4d39.jpg)  
Figure 2. Offline calibration and compact distance encoding. Left: candidate boundary pairs are evaluated by normalized projected-output error. Center: errors and complete-tile execution costs guide allocation under a shared computation budget. Full-model prediction checks select the final boundary table. Right: two diagonal-distance boundaries per layer and query head encode 8-bit, 4-bit, and skipped QK regions, with the first key block and diagonal tiles protected.

## 4 TRIATTENTION

TriAttention uses regional sensitivity to assign attention tiles to skipping, 4-bit, or 8-bit QK computation. The diagonaldistance boundaries define each layer-head configuration for both offline candidate evaluation and precision-grouped traversal. Offline calibration evaluates candidate boundary pairs, and a model-wide allocator uses their errors and costs to select a boundary table under a shared computation budget (Figure 2). During inference, this boundary table guides fused prefill and compressed KV-cache for decode.

## 4.1 Compact Distance Encoding

Long contexts enlarge the attention tile map, while each GPU thread block has a limited shared-memory budget for traversal metadata and staged operands. We organize the three QK actions by distance from the attention diagonal. Each boundary pair describes a query head's retained context and QK precision regions.

For query block i and key block j, let $\Delta = i - j$ be their distance in blocks. A configuration $c = ( w _ { 8 } , w _ { 4 } )$ specifies the extent of the 8-bit QK region and the overall retained region, with $1 \leq w _ { 8 } \leq w _ { 4 }$ . We protect the first key block and the diagonal tile with 8-bit QK, preserving access to the prompt beginning and the query's own block. For the remaining tiles, the configuration assigns

$$
a _ { c } ( \Delta ) = \left\{ \begin{array} { l l } { 8 { \cdot } \mathrm { b i t } , } & { 1 \leq \Delta < w _ { 8 } , } \\ { 4 { \cdot } \mathrm { b i t } , } & { w _ { 8 } \leq \Delta < w _ { 4 } , } \\ { \mathrm { s k i p } , } & { \Delta \geq w _ { 4 } . } \end{array} \right.\tag{3}
$$

Extending $w _ { 4 }$ retains additional context at 4-bit precision, while increasing $w _ { 8 }$ assigns 8-bit computation. We select a separate pair for every layer and query head from a candidate set C that includes full-context configurations.

The compact distance encoding in Figure 2 stores two integers per layer and query head, keeping policy storage independent of context length. At execution time, each thread block uses these boundaries to generate only the retained key-block indices for its query block. The indices are grouped by precision, so their group boundaries specify the arithmetic paths. Calibration evaluates these boundary pairs, and allocation selects one pair per layer and query head to form the boundary table.

![](images/c20d1d051784fc0b5f16815f9be37361f0d9e15dc3d3c5dbb0d3285ba7d8b469.jpg)  
Figure 3. Per-layer QK sensitivity in Llama-3.2-3B-Instruct. Rows are query heads; columns are diagonal-distance bands ordered near to far. Colors show layer-normalized projected-output error from 4-bit QK or skipping, with nearer bands kept in 8-bit.

![](images/84db26b07666c82bc8f7ddabf3123b877984ba7700515650ce48639f83502b20.jpg)

![](images/a8207218c5f0af06b9162c66105a92bdea0985f39fd0d43a7bc7301989c32203.jpg)  
Figure 4. Distribution of high-sensitivity QK bands in Llama-3.2- 3B-Instruct. Ten bars per layer. Each bar reports the percentage of heads whose error lies in the model-wide top 20% of layer-head– band errors, ranked separately for 4-bit and skipping.

## 4.2 Sensitivity-Guided Global Allocation

Figures 3 and 4 show that output errors from skipping and 4-bit QK vary across layers, heads, and diagonal-distance bands. We therefore evaluate candidate boundary pairs for each layer and query head and select a boundary table under a shared computation budget.

![](images/c19178ad2a32d9610fe96ec7f4229860a0fb9382a257fad5de942fd7e504a35b.jpg)  
Figure 5. Fused prefill with compact, precision-grouped traversal. The illustrated query block visits protected tiles 0 and 6, 8-bit QK tiles 4 and 5, and 4-bit QK tile 3. Protected tiles use 16-bit PV, while other retained tiles use 8-bit PV. Shared-memory staging and query reuse feed one register-resident attention state, followed by normalization, value restoration, and 16-bit output.

Measuring the effect on the layer output. To compare these configurations, we measure how each one changes a head's contribution to the representation passed through the model by the attention output projection. So we evaluate its error after this projection. We normalize each head's projected-output error by the layer's reference attentionoutput energy, making errors comparable across layers.

For calibration input x, let $O _ { \ell h } ^ { c } ( x )$ and $O _ { \ell h } ^ { \mathrm { r e f } } ( x )$ denote the candidate and dense 16-bit reference outputs of query head h in layer l at the scored calibration positions. Let $W _ { \ell h } ^ { O }$ be the block of the output projection that maps this head into the model's hidden dimension, and let $\begin{array} { r } { Y _ { \ell } ^ { \mathrm { r e f } } = \sum _ { h } O _ { \ell h } ^ { \mathrm { r e f } } W _ { \ell h } ^ { O } } \end{array}$ be the complete reference attention output. Over calibration prompts D, we score the candidate by

$$
e _ { \ell h } ( c ) = \frac { 1 } { | \mathcal { D } | } \sum _ { x \in \mathcal { D } } \frac { \left. \left( O _ { \ell h } ^ { c } ( x ) - O _ { \ell h } ^ { \mathrm { r e f } } ( x ) \right) W _ { \ell h } ^ { O } \right. _ { F } ^ { 2 } } { \left. Y _ { \ell } ^ { \mathrm { r e f } } ( x ) \right. _ { F } ^ { 2 } } .\tag{4}
$$

A lower score identifies a configuration that preserves the head's projected contribution more closely. All candidates' reference output continues to the next layer. Their scores thus compare the combined effects of context retention and quantized computation under common inputs.

Allocating a shared computation budget. Each error score is paired with the computation needed to produce that output. At a common target context length, let $n _ { 8 } ( c )$ and $n _ { 4 } ( c )$ count the candidate's retained 8-bit and 4-bit QK tiles, excluding the protected tiles shared by all candidates. Profiling complete tiles, including QK, softmax, and PV, gives the relative cost γ of a tile using 4-bit QK compared with one using 8-bit QK. We estimate normalized cost as $C ( c ) = [ n _ { 8 } ( c ) + \gamma n _ { 4 } ( c ) ] / n _ { \mathrm { a l l } 8 }$ , where $n _ { \mathrm { a l l } 8 }$ is the corresponding count when all eligible tiles use 8-bit QK. This cost prices both context extension and increased precision.

For each layer and head, we choose $c \ell h$ to minimize summed error under a shared normalized-cost budget B:

$$
\operatorname* { m i n } _ { \{ c _ { \ell h } \in \mathcal { C } \} } \sum _ { \ell , h } e _ { \ell h } ( c _ { \ell h } ) \mathrm { ~ s . t . ~ } \sum _ { \ell , h } C ( c _ { \ell h } ) \leq B .\tag{5}
$$

We generate candidate boundary tables by sweeping a shared cost weight $\lambda ~ \geq ~ 0$ , with each head selecting $\begin{array} { r } { c _ { \ell h } ( \lambda ) = \arg \operatorname* { m i n } _ { c \in { \mathcal C } } [ e _ { \ell h } ( c ) + \lambda C ( c ) ] } \end{array}$ . The shared weight assigns the same price to computation in every head, giving wider context or higher precision priority where it yields greater error reduction per added cost. Per budget, we retain the feasible sampled table with the lowest summed error.

The resulting tables are evaluated in the complete model on separate selection prompts. We choose the lowest-cost table that satisfies the prescribed thresholds for mean nexttoken KL divergence from the reference model and the mean increase in teacher-forced response negative log-likelihood within every context-length group. Appendix A.1 specifies the calibration prompts, candidate grid, cost-weight sweep, and selection thresholds.

## 4.3 Fused Prefill with Compact Traversal

Given the selected boundary table and a layer's Q, K, and V, prefill prepares the required operand formats and constructs a compact traversal for each query block. The traversal determines which K/V tiles enter shared memory and which arithmetic path processes them. The kernel accumulates their contributions into one attention output. Figure 5 follows the selected tiles through this execution.

Preparing reusable operands. After rotary position embedding, the runtime centers keys and applies a common orthogonal signed Hadamard transform to queries and keys to control their dynamic range before quantization. It prepares the 8-bit and 4-bit Q/K formats for the selected table, with their scales. Value formats share channelwise centering and scaling statistics, allowing reconstruction after aggregation. Preparation produces 16-bit and 8-bit values for prefill and 4-bit values for the compressed prompt KV cache. GPU memory holds the packed operands and statistics.

Constructing the shared-memory traversal. Each thread block reads its head's two boundaries and combines them with its query-block index to generate the retained key-block indices. Threads write these indices cooperatively in shared memory, with consecutive segments for protected tiles, 8-bit QK tiles, and 4-bit QK tiles. Segment boundaries specify the arithmetic path, so each list entry carries only a keyblock index. The kernel consumes this list to stage the corresponding packed K/V tiles and reuses the query representation within each pass. Skipping removes tiles from the list and their associated QK, softmax, and PV work. This organization places the current query block's traversal data alongside its operand buffers, linking the compact distance policy directly to on-chip execution.

![](images/5ba4c16c8524aea0048e6ac38731af33717534a33c7850acc7c49768b703e5b9.jpg)  
Figure 6. Mixed-precision decode over the compressed prompt KV cache. Alternating shared-memory buffers B0/B1 stage retained KV tiles for 8-bit (blue) or 4-bit (green) computation. The kernel reuses query fragments and maintains attention state in registers across tiles Each key segment writes partial states for subsequent merging with other segments and the 16-bit generated-token tail.

Executing the precision groups. Protected tiles use 8-bit QK with 16-bit PV. Both remaining QK precisions use 8-bit probabilities and 8-bit values for PV. Tile-local probability scaling preserves resolution within each retained tile. The integer path uses the same probability codes for its normalization sum and PV product, and their scale places both contributions in the shared state. All groups update the same attention state in registers, keeping QK, probability formation, and PV accumulation fused across the traversal. After the final tile, the kernel normalizes and restores the value coordinates to produce O for the layer's output projection. Quantization scales, probability coding, and shared-state updates are detailed in Appendix A.2. Prepared prompt K/V and statistics then supply the cache.

## 4.4 Decoding from the Compressed Cache

To reduce repeated prompt-data reads, we use the selected boundaries to compact the prepared quantized K/V into a compressed prompt KV cache. Cache construction applies each head's boundaries to the final prompt query block, identifies the retained prompt tiles, and packs this selection once for reuse throughout generation. Each new query attends to this compressed prompt KV cache and a 16-bit tail containing generated-token keys and values. Figure 6 shows the computation over the retained prompt data.

Mapping head decisions to shared storage. The policy specifies context per query head, while GQA shares physical K/V storage across a group of query heads. We retain the union of the group's requested tiles and store each key tile once at the highest requested precision. A head mask identifies which query heads retain the tile, and participating heads compute QK at its stored precision. Compact buffers store retained K/V tiles with indices, masks, and reconstruction statistics. Values use 8-bit storage for the first prompt block and a recent window at the prompt's end, with 4-bit storage for the other retained blocks. This combines shared storage with each head's selected context and reduces prompt-data reads at each generation step.

Sharing reads and parallelizing the retained context. The incoming query receives the same rotation and quantization used in prefill, allowing it to consume the stored prompt keys directly. Queries sharing a KV head form rows of one matrix operation, so each loaded KV tile serves the participating heads in the group. Since decode supplies one query position per request, the kernel splits the retained key tiles across GPU thread blocks to expose parallelism along the context. Each block processes its assigned segment using the stored key precision and probability codes matched to the value precision, returning $( m _ { r } , l _ { r } , o _ { r } )$ for merging.

Producing the output and extending the context. Newly generated keys and values are appended to the 16-bit tail. The merge kernel computes attention over this tail using the prompt's key-centering offset. It combines the prompt partials and restores their value coordinates with the stored statistics. With both sources in common score and value units, the kernel merges their contributions and normalizes the result to produce the current query's output. Appendix A.3 details compact traversal, shared cache indexing, and prompt-tail merging.

Table 1. End-to-end inference efficiency for Llama-3.2-3B-Instruct on A100-SXM4-40GB, with 65,536 input tokens, 128 output tokens, and batch size one. Peak VRAM covers generation including prefill. Cell colors are relative to BF16 under each metric's direction: green denotes equal or better results, blue worse results, and gray BF16. Bold and underlined values mark the best and worst results among Ours and Others; displayed ties share the mark.
<table><tr><td rowspan="2">Metric</td><td rowspan="2">Unit</td><td rowspan="2">Upper Ref. BF16</td><td rowspan="2">Ours</td><td colspan="5">Others</td></tr><tr><td>TriAttention</td><td>MInference FlexPrefill</td><td>SageAttn</td><td>MoA</td><td>SpargeAttn</td></tr><tr><td>TPOT↓</td><td>ms/token</td><td>26.64</td><td>10.06</td><td>26.91</td><td>26.79</td><td>26.93</td><td>176.64</td><td>26.92</td></tr><tr><td>Throughput ↑</td><td>tok/s</td><td>14.06</td><td>22.80</td><td>14.23</td><td>17.39</td><td>14.59</td><td>4.64</td><td>13.97</td></tr><tr><td>Peak VRAM↓</td><td>GiB</td><td>15.65</td><td>11.76</td><td>15.77</td><td>16.40</td><td>15.65</td><td>15.77</td><td>16.82</td></tr><tr><td>Energy ↓</td><td>J/1K tokens</td><td>24788.26</td><td>15737.62</td><td>21536.48</td><td>18358.31</td><td>23459.16</td><td>71430.99</td><td>23365.47</td></tr></table>

![](images/e74a0b20eb6d74e3dad56c442652ce77dd4764f24035b2586ab14dd916d66363.jpg)  
Figure 7. Generation-throughput ratio on A100-SXM4-40GB. Llama-3.2-3B-Instruct. Rows use batch sizes 1, 2, 4, and 8; columns use 128, 2,048, 8,192, 16,384 input tokens. Each panel varies output length over 128, 256, 512, and 1024 tokens. BF16 is normalized to 1×; higher is better. Method colors and order are fixed across panels.

## 5 EXPERIMENTS

## 5.1 Experimental Setup

We evaluate inference efficiency and model quality for general and long-context tasks, with allocation controls examining the value of head-specific retention and precision. The main evaluations focus on Llama-3.2-3B-Instruct, with computation-level ablations also covering Qwen2- 7B-Instruct and Ministral-3-8B-Instruct-2512-BF16. Appendix D.2 reports additional quality results for the latter two models.

Datasets and Metrics. We evaluate language modeling using WikiText-2 word perplexity and language understanding using accuracy on HellaSwag, MMLU, ARC-Challenge, LAMBADA, PIQA, and WinoGrande. Long-context question answering is assessed on nine LV-Eval (Yuan et al.,

2024) datasets at the 16K, 32K, and 64K length levels, using the benchmark's task-specific F1 and ROUGE-L metrics.

Configuration and Comparisons. TriAttention uses 64×64 attention tiles. For the full method, the target mean normalized computation cost is 0.80. BF16 provides the originalmodel reference with scaled dot-product attention (SDPA), while Torch attention uses the math backend. The lowbit reference, al1\_int 4, retains all causal tiles and uses INT4 QK outside the protected first and diagonal tiles. We compare against MInference, FlexPrefill, SageAttention2, MoA, and SpargeAttn (Jiang et al., 2024; Lai et al., 2025; Zhang et al., 2024a; Fu et al., 2024; Zhang et al., 2025a). Layer-shared, model-shared, and randomized boundary assignments test allocation granularity and sensitivity-guided placement. Appendix B.1 provides model configurations, task settings, and allocation-control definitions.

Table 2. General-task results for Llama-3.2-3B-Instruct. Accuracy is reported in percent (higher is better); WikiText-2 uses word perplexity (lower is better). BF16 is the upper reference; a11\_int 4 is the lower reference. Colors compare with BF16: green denotes equal or better results, blue worse results, and gray BF16. Bold marks the best result among Ours and Others; displayed ties share the mark.
<table><tr><td rowspan="2">Benchmark</td><td rowspan="2">Metric</td><td>Upper Ref.</td><td>Lower Ref.</td><td>Ours</td><td colspan="5">Others</td></tr><tr><td>BF16</td><td>all_int4</td><td>TriAttention</td><td>MInference</td><td>FlexPrefill</td><td>SageAttn</td><td>MoA</td><td>SpargeAttn</td></tr><tr><td>WikiText-2</td><td>PPL↓</td><td>12.49</td><td>12.59</td><td>12.50</td><td>12.53</td><td>12.74</td><td>12.52</td><td>12.64</td><td>12.43</td></tr><tr><td>HellaSwag</td><td>Acc. ↑</td><td>71.52</td><td>71.53</td><td>72.03</td><td>71.62</td><td>71.63</td><td>71.82</td><td>71.65</td><td>71.61</td></tr><tr><td>MMLU</td><td>Acc. ↑</td><td>60.66</td><td>60.26</td><td>61.32</td><td>60.65</td><td>60.64</td><td>60.86</td><td>61.28</td><td>60.79</td></tr><tr><td>ARC-Challenge</td><td>Acc. ↑</td><td>52.56</td><td>51.71</td><td>52.97</td><td>52.73</td><td>52.47</td><td>51.79</td><td>53.07</td><td>52.05</td></tr><tr><td>LAMBADA</td><td>Acc. ↑</td><td>64.23</td><td>64.33</td><td>64.83</td><td>64.39</td><td>64.49</td><td>64.25</td><td>64.82</td><td>64.39</td></tr><tr><td>PIQA</td><td>Acc. ↑</td><td>77.31</td><td>77.15</td><td>77.65</td><td>76.71</td><td>77.26</td><td>76.88</td><td>77.04</td><td>76.71</td></tr><tr><td>WinoGrande</td><td>Acc. ↑</td><td>71.43</td><td>71.11</td><td>71.85</td><td>71.51</td><td>71.59</td><td>71.82</td><td>70.80</td><td>70.24</td></tr></table>

Hardware and Measurement. Primary efficiency measurements use an NVIDIA A100-SXM4 40 GB GPU, with additional system evaluations on H100 NVL and RTX 4090 GPUs. We report time per output token (TPOT), completerequest generation throughput, peak GPU memory, and energy per 1,000 generated tokens. The workload sweeps vary prompt length, output length, and batch size, with complete-request measurements covering both prefill and decode. Timing, energy, and memory-measurement protocols are specified in Appendix B.2.

## 5.2 Inference Efficiency

We first examine whether the selected attention representation improves complete generation requests. On Llama with a 64K prompt, TriAttention reaches 22.80 tokens/s, compared with 14.06 for BF16 and 17.39 for FlexPrefill, yielding 1.62× and 1.31× throughput, respectively (Table 1). The improvement accompanies a reduction in TPOT from 26.64 to 10.06 ms relative to BF16. Peak allocation decreases from 15.65 to 11.76 GiB, and energy per generated token falls by 36.5%. These measurements establish that the attention implementation's benefits extend to inference with the rest of the model included.

The comparison with FlexPrefill highlights the importance of carrying attention savings into generation. Its TPOT is 26.79 ms, compared with 10.06 ms for TriAttention, alongside the complete-request throughput improvement. Figure 7 evaluates this combination across different batch sizes and input/output lengths. Prompt processing occurs once per request, while each generated token reuses the compressed prompt cache. The workload sweep therefore evaluates how the prefill and decode paths contribute as the balance between prompt processing and generation changes. Appendix D.1 extends the throughput and TPOT comparisons across models, request shapes, and GPU platforms.

FINDINGS Joint retention and precision improve complete-request throughput while reducing tokengeneration latency, peak memory, and energy.

## 5.3 General-Purpose Performance

Having established inference gains, we examine whether the same boundary table preserves general prediction quality across tasks. The table selected through summary and retrieval calibration is reused across the benchmarks, testing whether its retention and precision choices support broader model capabilities. On Llama, TriAttention has word perplexity 12.50, compared with 12.49 for BF16 (Table 2). All six accuracy scores exceed the BF16 reference by less than 0.7 percentage points, with MMLU increasing from 60.66% to 61.32%. Near-reference quality therefore extends across both language modeling and the evaluated understanding tasks under one allocation.

The full-context al1\_int 4 reference provides a complementary comparison by retaining all causal tiles and applying 4-bit QK outside the protected regions. Its perplexity is 12.59 and MMLU accuracy is 60.26%, while TriAttention achieves lower perplexity and higher accuracy on all six tasks. These results support jointly selecting retained context and score precision to preserve predictions alongside the measured inference gains. The following long-context QA evaluation examines whether this quality extends from prefill predictions to answer generation with repeated access to the compressed prompt cache.

FINDINGS One model-level retention-and-precision policy preserves general-task quality across benchmarks while supporting efficient inference.

## 5.4 Long-Context Question Answering

Long-context QA evaluates whether the context retained during prefill remains useful throughout answer generation. Table 3 reports nine tasks at the 16K, 32K, and 64K dataset levels using the same calibrated boundary table, alongside controls for boundary sharing and placement. Dataset identifiers are ① cmrc\_mixup, ② dureader\_mixup, ③ hotpotwikiqa\_mixup, ④ lic\_mixup, ⑤ loogle\_CR\_mixup, ⑥ loogle\_MIR\_mixup,⑦ loogle\_SD\_mixup, ⑧ multifieldqa\_en\_mixup, and⑨multifieldqa\_zh\_mixup.

Table 3. LV-Eval QA results for Llama-3.2-3B-Instruct. Dataset identifiers follow Section 5.4. Scores use official task-specific metrics on a 0–100 scale; higher is better. Length labels follow the official dataset levels. BF16 is the upper reference; a11\_int 4 is the lower reference. Colors compare with BF16: green denotes equal or better results, blue worse results, and gray BF16. Bold marks the best result among Ours, Others, and Ablations; displayed ties share the mark
<table><tr><td rowspan="2"></td><td rowspan="2">Dataset Level</td><td>Upper Ref. Lower Ref.</td><td></td><td>Ours</td><td colspan="5">Others</td><td colspan="3">Ablations</td></tr><tr><td>BF16</td><td>all_int4</td><td>TriAttention</td><td>MInference FlexPrefill SageAttn</td><td></td><td></td><td>MoA</td><td></td><td>SpargeAttn Layer Same</td><td>All Same</td><td>Random</td></tr><tr><td rowspan="4">①</td><td>16K</td><td>21.33</td><td>18.10</td><td>21.94</td><td>26.93</td><td>27.11</td><td>29.79</td><td>13.54</td><td>20.85</td><td>20.66</td><td>20.05</td><td>19.65</td></tr><tr><td>32K</td><td>18.49</td><td>15.34</td><td>18.07</td><td>11.55</td><td>11.74</td><td>13.50</td><td>10.02</td><td>16.40</td><td>16.50</td><td>17.47</td><td>13.38</td></tr><tr><td>64K</td><td>13.95</td><td>11.32</td><td>14.61</td><td>5.31</td><td>5.39</td><td>5.77</td><td>6.21</td><td>13.72</td><td>11.63</td><td>6.47</td><td>4.30</td></tr><tr><td>16K</td><td></td><td>16.22</td><td></td><td>14.61</td><td>14.92</td><td>16.67</td><td>14.57</td><td>15.33</td><td>15.84</td><td>15.56</td><td>16.20</td></tr><tr><td rowspan="4">②</td><td>32K</td><td>17.59</td><td>15.48 15.16</td><td>16.84 17.01</td><td>12.01</td><td>11.64</td><td>14.59</td><td>14.96</td><td>16.87</td><td>15.10</td><td>16.31</td><td>15.14</td></tr><tr><td>64K</td><td>17.00</td><td>15.25</td><td>17.92</td><td>7.96</td><td>6.81</td><td>11.22</td><td>13.36</td><td>16.39</td><td>15.35</td><td>14.20</td><td>14.37</td></tr><tr><td>16K</td><td>23.76</td><td></td><td></td><td>13.97</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>32K</td><td>29.57</td><td>20.26 28.78</td><td>24.22 28.81</td><td>7.81</td><td>13.16 7.20</td><td>15.27 9.11</td><td>15.91 19.04</td><td>20.94 28.43</td><td>18.68 21.63</td><td>21.98 23.25</td><td>22.66 24.61</td></tr><tr><td rowspan="3">③</td><td>64K</td><td>22.66</td><td>22.68</td><td>25.53</td><td>4.26</td><td>3.86</td><td>5.50</td><td>19.20</td><td>22.75</td><td>22.89</td><td>13.54</td><td>17.63</td></tr><tr><td></td><td>11.05</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>16K 32K</td><td>12.03</td><td>10.12</td><td>11.53</td><td>9.82</td><td>9.28</td><td>10.69</td><td>7.88</td><td>10.75</td><td>10.53</td><td>9.50</td><td>11.09</td></tr><tr><td rowspan="3">④</td><td>64K</td><td>9.57</td><td>10.01 7.33</td><td>12.67 10.85</td><td>4.88 1.12</td><td>4.48 1.07</td><td>5.69 1.44</td><td>6.72</td><td>9.72</td><td>12.02 7.41</td><td>11.35 4.47</td><td>7.47 1.93</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>4.15</td><td>9.42</td><td></td><td></td><td></td></tr><tr><td>16K</td><td>25.27</td><td>23.86</td><td>25.64</td><td>15.59</td><td>14.05</td><td>19.94</td><td>19.28</td><td>25.53</td><td>23.85</td><td>24.75</td><td>23.87</td></tr><tr><td rowspan="4">⑤</td><td>32K</td><td>25.70 24.83</td><td>23.41</td><td>25.75</td><td>12.23</td><td>10.95</td><td>15.92</td><td>18.15</td><td>25.39</td><td>23.00</td><td>21.29</td><td>23.05</td></tr><tr><td>64K</td><td></td><td>24.17</td><td>25.46</td><td>7.85</td><td>6.67</td><td>11.24</td><td>18.43</td><td>22.15</td><td>20.57</td><td>15.59</td><td>14.62</td></tr><tr><td>16K</td><td>19.12</td><td>18.90</td><td>20.16</td><td>12.32</td><td>11.39</td><td>13.93</td><td>15.44</td><td>19.78</td><td>17.21</td><td>17.64</td><td>17.17</td></tr><tr><td>32K</td><td>20.11</td><td>18.81</td><td>19.17</td><td>9.26</td><td>7.40</td><td>10.54</td><td>13.68</td><td>18.85</td><td>16.12</td><td>13.58</td><td>13.11</td></tr><tr><td rowspan="3">⑦</td><td>64K</td><td>19.09</td><td>18.89</td><td>19.65</td><td>7.24</td><td>5.08</td><td>9.45</td><td>9.74</td><td>17.58</td><td>17.38</td><td>7.94</td><td>10.00</td></tr><tr><td>16K</td><td>54.01</td><td>51.31</td><td>54.42</td><td>38.16</td><td>35.24</td><td></td><td>42.12 37.20</td><td>51.22</td><td>49.34</td><td>51.59</td><td>50.49</td></tr><tr><td>32K</td><td>49.88</td><td>46.84</td><td>50.74</td><td>20.09</td><td>21.40</td><td>22.50</td><td>29.38</td><td>47.21</td><td>44.03</td><td>35.39</td><td>41.88</td></tr><tr><td rowspan="3">⑧</td><td>64K</td><td>48.40</td><td>44.82</td><td>47.94</td><td>10.24</td><td>10.69</td><td>13.21</td><td>19.63</td><td>45.45</td><td>45.51</td><td>17.07</td><td>20.17</td></tr><tr><td>16K</td><td>34.19</td><td>33.12</td><td>33.26</td><td>23.28</td><td>23.43</td><td></td><td>25.18 23.59</td><td>36.27</td><td>32.08</td><td>33.39</td><td>33.07</td></tr><tr><td>32K</td><td>32.95</td><td>30.55</td><td>31.50</td><td>12.70</td><td>10.28</td><td>13.94</td><td>20.76</td><td>32.99</td><td>30.68</td><td>23.18</td><td>27.54</td></tr><tr><td rowspan="3">⑨</td><td>64K</td><td>36.77</td><td>33.60</td><td>35.19</td><td>9.42</td><td>7.68</td><td>13.37</td><td>23.98</td><td>33.44</td><td>35.36</td><td>22.53</td><td>21.71</td></tr><tr><td>16K</td><td>18.42</td><td>19.79</td><td>19.54</td><td>15.99</td><td>12.64</td><td>18.19</td><td>16.02</td><td>19.56</td><td>20.56</td><td>19.37</td><td>19.26</td></tr><tr><td>32K 64K</td><td>12.81 11.59</td><td>11.86 10.64</td><td>14.53</td><td>11.33</td><td>10.25 4.02</td><td>14.57</td><td>7.43</td><td>13.61</td><td>10.87</td><td>14.31</td><td>12.67</td></tr></table>

LooGLE-SD shows how this policy preserves answer quality as context grows. TriAttention scores 54.42, 50.74, and 47.94 at 16K, 32K, and 64K, respectively, following BF16's 54.01, 49.88, and 48.40 with differences below one point at every level. At 64K, its score also exceeds 44.82 for a11\_int 4 and 45.45 for SpargeAttn. The comparison with al1\_int 4, which retains every causal tile, supports selecting score precision together with retained context. Selective mixed-precision attention preserves stronger answer quality here than retaining the full context with uniformly low QK precision outside the protected regions.

Tasks requiring multiple pieces of evidence provide a complementary check. At 64K, TriAttention scores 19.65 on LooGLE-MIR, compared with 19.09 for BF16 and 17.58 for SpargeAttn. On LooGLE-CR, the corresponding scores are 25.46, 24.83, and 22.15. HotpotWikiQA also reaches 25.53, compared with 22.66 for BF16 and 22.75 for SpargeAttn. These comparisons extend quality preservation across different QA tasks under one model-level allocation. They assess the complete prefill-and-decode path: the retained context must support generated answers after being packed into the compressed prompt cache.

The allocation controls examine how this behavior depends on the freedom to assign different boundaries. Layer Same shares one boundary pair across the query heads of each layer, while All Same uses a common 32K 8-bit window throughout the model, with the first block protected. On LooGLE-SD at 64K, their scores are 45.51 and 17.07, compared with 47.94 for TriAttention. On LooGLE-CR, the corresponding scores are 20.57, 15.59, and 25.46. Sharing boundaries couples the retained ranges and precision choices of heads with different sensitivities. Head-specific boundaries let the allocator preserve wider ranges or higher precision where they most reduce projected-output error supporting stronger QA performance in these comparisons.

Random then tests the value of matching the selected configurations to head sensitivity. It permutes the deployed boundary pairs across layers and heads, preserving both their collection and the aggregate modeled computation cost. On LooGLE-SD, its gap from TriAttention widens from 3.93 points at 16K to 27.77 points at 64K, where Random scores 20.17 against 47.94. LooGLE-MIR shows a corresponding separation at 64K, with 10.00 for Random and 19.65 for TriAttention. The growing separation links long-context quality to which heads receive the wider ranges and higher precision. Sensitivity-guided placement therefore makes the selected computation more effective for QA within the same modeled budget.

FINDINGS The selected ranges preserve useful context throughout answer generation, while sensitivityguided placement achieves stronger long-context QA than randomized assignment at the same modeled computation cost.

## 5.5 Ablation and Analysis

The preceding QA controls examine the placement and sharing of boundary configurations. We now examine the computation choices available to the allocator: whether combining skipping with two retained precision levels reduces the prefill time needed to meet the same quality-selection requirements.

Computation-level controls. Without 4-bit imposes w8 = w4, leaving each non-protected region a choice between 8-bit QK and skipping. Without Skip imposes w4 = ∞, retaining the full causal context while assigning its 4-bit and 8-bit regions. Each control is independently reallocated and selected under the same quality-selection thresholds, allowing its remaining choices to adapt to the validation requirements. Table 4 compares the resulting prefill attention latency on A100 across three models and contexts from 32K to 128K.

Table 4. Effect of three-level computation on A100 prefill attention latency. Latency includes Q/K preparation, V packing, and attention across all layers. Ablations are independently reallocated under the same quality-selection thresholds. Lower is better.
<table><tr><td>Model</td><td>Context</td><td>TriAttention (s)</td><td>Without 4-bit (s)</td><td>Without Skip (s)</td></tr><tr><td rowspan="3">Llama</td><td>32K</td><td>0.654</td><td>0.660</td><td>0.682</td></tr><tr><td>64K</td><td>2.242</td><td>2.259</td><td>2.647</td></tr><tr><td>128K</td><td>7.347</td><td>7.568</td><td>10.324</td></tr><tr><td rowspan="3">Qwen2</td><td>32K</td><td>0.781</td><td>0.810</td><td>0.845</td></tr><tr><td>64K</td><td>2.752</td><td>2.924</td><td>3.232</td></tr><tr><td>128K</td><td>10.134</td><td>11.151</td><td>13.010</td></tr><tr><td rowspan="3">Ministral</td><td>32K</td><td>1.066</td><td>1.106</td><td>1.086</td></tr><tr><td>64K</td><td>3.856</td><td>4.078</td><td>4.182</td></tr><tr><td>128K</td><td>14.266</td><td>15.449</td><td>16.402</td></tr></table>

Lowering the cost of retained context. TriAttention achieves the lowest latency in all nine settings. At 128K, the full design reduces Qwen2 latency from 11.151 s without 4-bit to 10.134 s, and Ministral latency from 15.449 to 14.266 s, reductions of 9.1% and 7.7%, respectively. Without the intermediate level, retaining an interaction also commits it to 8-bit QK computation. The 4-bit region separates these choices, allowing lower-cost retention alongside an 8-bit region for more sensitive interactions. Since the measurements include Q/K preparation and V packing across all layers, these reductions reflect the benefit over complete attention calls after preparing the mixed-precision operands.

Removing work as context grows. Skipping reduces latency by eliminating QK, normalization, and PV for omitted tiles. For Llama, the reduction relative to Without Skip increases from 4.1% at 32K to 28.8% at 128K, where latency falls from 10.324 to 7.347 s. Qwen2 shows the same length-dependent pattern, with reductions increasing from 7.6% to 22.1%; the relative benefit also grows across the three lengths for Ministral. Full retention adds interactions throughout the causal map as context expands. Allowing selected distance regions to be skipped removes their complete tile cost, complementing the lower QK cost of interactions that remain.

Adapting the retention-precision trade-off. The faster restricted configuration also changes with the workload. For Ministral at 32K, Without Skip takes 1.086 s and Without 4-bit takes 1.106 s. At 128K, their ordering reverses, with 16.402 and 15.449 s, respectively, while the full design remains fastest at both lengths. Thus, the relative efficiency of retaining all context at mixed precision and retaining selected context at 8 bits varies even within one model. Independently adjustable boundaries let the allocator combine these alternatives. The measured advantage of the full design supports jointly choosing how far attention extends and which precision is used within that range.

FINDINGSThree computation levels reduce both the cost of retained interactions and the number of executed tiles, lowering prefill latency under common quality-selection requirements.

## 6 CONCLUSION

TriAttention demonstrates that useful attention interactions can be preserved at lower cost by jointly allocating retention and QK precision according to regional sensitivity. Compact distance boundaries carry these choices through fused prefill and compressed-cache decoding, yielding faster generation while preserving general-task and long-context quality. Allocation controls further show that computation placement affects long-context quality at a fixed modeled cost. These findings support a broader principle for efficient attention: evaluate which interactions to retain together with the precision they require, and make that joint policy govern both GPU computation and repeated context access.

## REFERENCES

Bai, Y., Lv, X., Zhang, J., Lyu, H., Tang, J., Huang, Z., Du, Z., Liu, X., Zeng, A., Hou, L., et al. Longbench: A bilingual, multitask benchmark for long context understanding. In Proceedings of the 62nd annual meeting of the association for computational linguistics (volume 1: Long papers), pp. 3119–3137, 2024.

Dao, T. Flashattention-2: Faster attention with better parallelism and work partitioning. In International Conference on Learning Representations, volume 2024, pp. 35549– 35562, 2024.

Dao, T., Fu, D., Ermon, S., Rudra, A., and Ré, C. Flashattention: Fast and memory-efficient exact attention with io-awareness. Advances in neural information processing systems, 35:16344–16359, 2022.

Ding, Y., Zhang, X., Wang, Y., Chen, Y., and Guo, J. Diagonal-tiled mixed-precision attention for efficient lowbit mxfp inference. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 3420–3428, 2026.

Dong, J., Feng, B., Guessous, D., Liang, Y., and He, H. Flexattention: A programming model for generating fused attention variants. Proceedings of Machine Learning and Systems, 7, 2025.

Fabbri, A. R., Li, I., She, T., Li, S., and Radev, D. Multi-news: A large-scale multi-document summarization dataset and abstractive hierarchical model. In Proceedings of the 57th annual meeting of the association for computational linguistics, pp. 1074–1084, 2019.

Fu, T., Huang, H., Ning, X., Zhang, G., Chen, B., Wu, T., Wang, H., Huang, Z., Li, S., Yan, S., et al. Moa: Mixture of sparse attention for automatic large language model compression. 2024.

Gao, L., Tow, J., Abbasi, B., Biderman, S., Black, S., DiPofi, A., Foster, C., Golding, L., Hsu, J., Le Noac'h, A., Li, H., McDonell, K., Muennighoff, N., Ociepa, C., Phang, J., Reynolds, L., Schoelkopf, H., Skowron, A., Sutawika, L., Tang, E., Thite, A., Wang, B., Wang, K., and Zou, A. The language model evaluation harness, 07 2024a. URL https://zenodo.org/records/12608602.

Gao, Y., Zeng, Z., Du, D., Cao, S., Zhou, P., Qi, J., Lai, J., So, H. K.-H., Cao, T., Yang, F., et al. Seerattention: Learning intrinsic sparse attention in your llms. arXiv preprint arXiv:2410.13276, 2024b.

Grattafiori, A., Dubey, A., Jauhri, A., Pandey, A., Kadian, A., Al-Dahle, A., Letman, A., Mathur, A., Schelten, A., Vaughan, A., et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

Hong, K., Dai, G., Xu, J., Mao, Q., Li, X., Liu, J., Chen, K., Dong, Y., and Wang, Y. Flashdecoding++: Faster large language model inference with asynchronization, flat gemm optimization, and heuristics. Proceedings of Machine Learning and Systems, 6:148–161, 2024.

Hooper, C., Kim, S., Mohammadzadeh, H., Mahoney, M. W., Shao, Y. S., Keutzer, K., and Gholami, A. Kvquant: Towards 10 million context length llm inference with kv cache quantization. Advances in Neural Information Processing Systems, 37:1270–1303, 2024.

Jiang, H., Li, Y., Zhang, C., Wu, Q., Luo, X., Ahn, S., Han, Z., Abdi, A. H., Li, D., Lin, C.-Y., et al. Minference 1.0: Accelerating pre-filling for long-context llms via dynamic sparse attention. Advances in Neural Information Processing Systems, 37:52481–52515, 2024.

Kang, H., Bharadwaj, S., Hensman, J., Krishna, T., Rühle, V., and Rajmohan, S. Turboattention: Efficient attention approximation for high throughputs llm. Proceedings of Machine Learning and Systems, 7, 2025.

Lai, X., Lu, J., Luo, Y., Ma, Y., and Zhou, X. Flexprefill: A context-aware sparse attention mechanism for efficient long-sequence inference. arXiv preprint arXiv:2502.20766, 2025.

Liu, Z., Yuan, J., Jin, H., Zhong, S., Xu, Z., Braverman, V., Chen, B., and Hu, X. Kivi: A tuning-free asymmetric 2bit quantization for kv cache. arXiv preprint arXiv:2402.02750, 2024.

Peng, B., Quesnelle, J., Fan, H., and Shippole, E. Yarn: Efficient context window extension of large language models. In International Conference on Learning Representations, volume 2024, pp. 31932–31951, 2024.

Sanovar, R., Bharadwaj, S., St Amant, R., Rühle, V., and Rajmohan, S. Leanattention: Hardware-aware scalable attention mechanism for the decode-phase of transformers. Proceedings of Machine Learning and Systems, 7, 2025.

Shah, J., Bikshandi, G., Zhang, Y., Thakkar, V., Ramani, P., and Dao, T. Flashattention-3: Fast and accurate attention with asynchrony and low-precision. Advances in Neural Information Processing Systems, 37:68658–68685, 2024.

Sharratt, J. Thriftattention: Selective mixed precision for long-context fp4 attention. arXiv preprint arXiv:2605.23081, 2026.

Xiao, G., Tian, Y., Chen, B., Han, S., and Lewis, M. Efficient streaming language models with attention sinks. In International Conference on Learning Representations, volume 2024, pp. 21875–21895, 2024.

Xiao, G., Tang, J., Zuo, J., Guo, J., Yang, S., Tang, H., Fu, Y., and Han, S. Duoattention: Efficient long-context llm inference with retrieval and streaming heads. In International Conference on Learning Representations, volume 2025, pp. 37228–37253, 2025.

Xu, R., Xiao, G., Huang, H., Guo, J., and Han, S. Xattention: Block sparse attention with antidiagonal scoring. arXiv preprint arXiv:2503.16428, 2025.

Yang, S., Guo, J., Tang, H., Hu, Q., Xiao, G., Tang, J., Lin, Y., Liu, Z., Lu, Y., and Han, S. Lserve: Efficient long-sequence llm serving with unified sparse attention. Proceedings of Machine Learning and Systems, 7, 2025.

Ye, Z., Chen, L., Lai, R., Lin, W., Zhang, Y., Wang, S., Chen, T., Kasikci, B., Grover, V., Krishnamurthy, A., et al. Flashinfer: Efficient and customizable attention engine for llm inference serving. Proceedings of Machine Learning and Systems, 7, 2025.

Yuan, J., Gao, H., Dai, D., Luo, J., Zhao, L., Zhang, Z., Xie, Z., Wei, Y., Wang, L., Xiao, Z., et al. Native sparse attention: Hardware-aligned and natively trainable sparse attention. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 23078–23097, 2025.

Yuan, T., Ning, X., Zhou, D., Yang, Z., Li, S., Zhuang, M., Tan, Z., Yao, Z., Lin, D., Li, B., et al. Lv-eval: A balanced long-context benchmark with 5 length levels up to 256k. arXiv preprint arXiv:2402.05136, 2024.

Zhang, J., Huang, H., Zhang, P., Wei, J., Zhu, J., and Chen, J. Sageattention2: Efficient attention with thorough outlier smoothing and per-thread int4 quantization. arXiv preprint arXiv:2411.10958, 2024a.

Zhang, J., Xiang, C., Huang, H., Wei, J., Xi, H., Zhu, J., and Chen, J. Spargeattention: Accurate and training-free sparse attention accelerating any model inference. arXiv preprint arXiv:2502.18137, 2025a.

Zhang, J., Zhang, P., Zhu, J., Chen, J., et al. Sageattention: Accurate 8-bit attention for plug-and-play inference acceleration. In International Conference on Learning Representations, volume 2025, pp. 71566–71585, 2025b.

Zhang, J., Wang, H., Zhang, P., Xu, X., Huang, H., Jiang, K. Chen, J., Zhu, J., et al. Sageattention3: Microscaling fp4 attention for inference and an exploration of 8-bit training. Advances in Neural Information Processing Systems, 38: 53901–53931, 2026.

Zhang, Z., Liu, S., Chen, R., Kailkhura, B., Chen, B., and Wang, Z. Q-hitter: A better token oracle for efficient llm inference via sparse-quantized kv cache. Proceedings of Machine Learning and Systems, 6:381–394, 2024b.

Zhu, Q., Duan, J., Chen, C., Liu, S., Li, X., Feng, G., Lv, X., Chuanfu, X., Lin, D., and Yang, C. Sampleattention: Near-lossless acceleration of long context llm inference with adaptive structured sparse attention. Proceedings of Machine Learning and Systems, 7, 2025.

## A CALIBRATION AND IMPLEMENTATION DETAILS

The boundary table connects offline sensitivity measurements to the operands used during inference. This appendix details how calibration selects that table, how low-bit products recover attention scores and outputs, and how execution addresses and merges the resulting representations.

## A.1 Calibration and Boundary-Table Selection

Calibration inputs and scored positions. We combine document summarization and archive-record retrieval to evaluate both distributed context use and access to specific evidence. The 16 calibration prompts span 16K, 32K, 64K, and 128K contexts. Summary prompts use Multi-News (Fabbri et al., 2019) documents and reference summaries capped at 256 response tokens. Retrieval prompts place a unique numeric record at document depths 0.2, 0.5, or 0.8 and request its value at the end. Each prompt uses the model's chat template and reference response, with a 129,024-token budget for the 128K construction to accommodate both. Calibration and selection use separate documents, disjoint from LV-Eval and LongEval.

We score the final 1,024 positions, covering the prompt ending and teacher-forced response, so the evaluated queries attend over the long input. A dense BF16 pass supplies common Q/K/V for all candidates and propagates the reference output to the next layer. Candidate errors therefore reflect their retention and precision choices under the same layer input. Equation 4 normalizes each prompt's projected-output error before averaging, giving every prompt equal weight

Candidate ranges and computation costs. Geometrically spaced boundaries cover local through full-context configurations with a compact candidate set. Measured in 64-token blocks, the grid is

$$
\begin{array} { r l } & { \mathcal { W } _ { 8 } = \{ 1 , 2 , 4 , . . . , 5 1 2 , \infty \} , } \\ & { \mathcal { W } _ { 4 } = \{ 1 , 2 , 4 , . . . , 2 0 4 8 , \infty \} , } \\ & { \mathcal { C } = \{ ( w _ { 8 } , w _ { 4 } ) : w _ { 8 } \in \mathcal { W } _ { 8 } , } \\ & { \quad \quad \quad w _ { 4 } \in \mathcal { W } _ { 4 } , w _ { 8 } \leq w _ { 4 } \} , } \end{array}
$$

These 86 pairs jointly vary retained context and QK precision. Setting $w _ { 4 } = \infty$ retains the full causal context, while $w _ { 8 } = w _ { 4 }$ assigns 8-bit QK to retained non-protected tiles.

The allocator evaluates non-protected tile costs at 65,536 tokens using the measured complete-tile ratio $\gamma = 0 . 9 0 5$ For $N _ { H }$ layer-query-head pairs, a target mean cost τ gives the total budget $B = N _ { H } \tau$ in Equation 5. We evaluate $\lambda = 0$ and 400 logarithmically spaced positive values from $1 0 ^ { - 9 } \ \mathrm { t o \ 1 0 ^ { 1 } }$ , retaining the feasible sampled table with the lowest summed calibration error for each target budget. The deployed mean-cost targets are 0.80 for Llama-3.2-

3B-Instruct and 0.85 for Qwen2-7B-Instruct and Ministral-  
3-8B-Instruct-2512-BF16.

Complete-model selection. The assembled boundary tables are evaluated on separate summary and retrieval prompts. KL divergence measures changes in the BF16 next-token distribution, while response NLL measures the likelihood assigned to the reference answers. We average each metric over response tokens within a prompt, then over prompts in the 16K/32K, 64K, and 128K groups. Each group must satisfy mean KL at most 0.002 and mean NLL increase at most 0.01 nats. This applies the quality requirements separately across context lengths before selecting the lowestcost qualifying table for inference.

## A.2 Operand Quantization and Shared Normalization

Queries and keys in common score coordinates. Key centering removes the prompt-wide key offset, and a shared orthogonal transform redistributes channel magnitudes before quantization (Zhang et al., 2024a; Shah et al., 2024; Zhang et al., 2026). These operations prepare the same Q/K inputs for both precision paths. After rotary position embedding, let $\mu _ { K }$ be the mean prompt key vector for one KV head and U the fixed orthogonal signed Hadamard transform. The prepared operands are $\widetilde Q = Q U$ and $\widetilde { K } = ( K - \mathbf 1 \mu _ { K } ^ { \top } ) U$ Since $U U ^ { \dagger } = I $

$$
\widetilde Q \widetilde K ^ { \top } = Q K ^ { \top } - ( Q \mu _ { K } ) \mathbf { 1 } ^ { \top } .\tag{6}
$$

Centering subtracts the same quantity from every legal score in a query row and therefore preserves its softmax before quantization. The orthogonal transform redistributes channel magnitudes without changing the centered dot products.

For bit-width $r ~ \in ~ \{ 4 , 8 \}$ , symmetric quantization uses $a _ { r } = 2 ^ { r - 1 } - 1$ and absmax scale $s _ { X } ^ { ( r ) } = \operatorname* { m a x } | X | / a _ { r }$ per quantization group (Zhang et al., 2025b). Codes round $X / s _ { X } ^ { ( r ) }$ to the nearest integer, with zero-valued groups producing zero codes. Queries use one scale per token and query head; keys use one scale per KV head and group of 16 consecutive tokens across their channels. The 4-bit and 8-bit codes represent the same transformed operands with separate scales. For query-token index t and key-token index k, the restored score is

$$
\widetilde { s } _ { t k } ^ { ( r ) } = \frac { s _ { Q , t } ^ { ( r ) } s _ { K , g ( k ) } ^ { ( r ) } } { \sqrt { d } } \left. \widehat { Q } _ { t } ^ { ( r ) } , \widehat { K } _ { k } ^ { ( r ) } \right. ,\tag{7}
$$

where $g ( k )$ is key $k ' s$ 16-token group. Restored scores at both precisions enter the same probability computation.

Values in shared reconstruction coordinates. Sharing channel statistics across value formats lets the kernel accumulate their contributions together and reconstruct the output once. We center and scale each prompt value channel to prepare the FP16, INT8, and INT4 representations (Zhang et al., 2024a). For each KV head and channel c, preparation computes the prompt mean $\mu _ { V , c }$ and

$$
\sigma _ { V , c } = \frac { \operatorname* { m a x } _ { k } | V _ { k , c } - \mu _ { V , c } | } { 1 2 7 } .
$$

A constant channel uses scale one. The normalized values are $V ^ { \prime } = \left( V - \mu _ { V } \right) \oslash \sigma _ { V }$ . Protected prefill tiles store $V ^ { \prime }$ in FP16, while other retained prefill tiles use the rounded INT8 codes $\widehat { V } _ { 8 }$ . For distant decode tiles, $\widehat { V } _ { 4 } = \mathrm { r o u n d } ( 7 V ^ { \prime } / 1 2 7 )$ is stored in signed 4-bit form and reconstructed in the same normalized coordinates as $( 1 2 7 / 7 ) \widehat { V } _ { 4 }$

Probability coding and accumulation. Tile-local scaling gives each tile its own probability-coding range, while a separate factor preserves its weight relative to other tiles. For one query row, we encode the restored scores in base-two units and merge their contributions into the shared attention state. For one query row and the current key tile, let $s _ { k }$ be the restored score multiplied by $\log _ { 2 } e ,$ , and let $b = \operatorname* { m a x } _ { k \in \mathcal { K } } { s _ { k } }$ where K contains this row's legal keys in the tile.

Using the row's query scale and the four key-group scales at QK precision $r ,$ the largest score multiplier is

$$
\eta _ { \mathrm { m a x } } = \frac { \log _ { 2 } e } { \sqrt { d } } s _ { Q } ^ { ( r ) } \operatorname* { m a x } _ { u \in \{ 0 , 1 , 2 , 3 \} } s _ { K , u } ^ { ( r ) } .\tag{8}
$$

We use the probability-coding base

$$
\begin{array} { r } { \varepsilon _ { \mathrm { t i l e } } = 0 . 7 5 \eta _ { \mathrm { m a x } } + 2 . 4 \times 1 0 ^ { - 7 } | b | , } \\ { b ^ { \prime } = b + \varepsilon _ { \mathrm { t i l e } } . \qquad } \end{array}\tag{9}
$$

The shared base provides rounding headroom while keeping all four key groups on the same probability scale.

The probability codes are

$$
\widehat { p } _ { k } = \left\lfloor \mathrm { r o u n d } _ { 1 / 2 5 6 } \left( a _ { P } 2 ^ { s _ { k } - b ^ { \prime } } + \rho _ { k } \right) \right\rfloor ,
$$

where $\mathrm { r o u n d _ { 1 / 2 5 6 } }$ denotes rounding to the nearest multiple of $1 / 2 5 6$ in the FP32 packing step. Prefill uses $\rho _ { k } = 1 / 2$ Decode uses $\rho _ { k } = ( D _ { k }$ mod $2 5 6 ) / 2 5 6$ , where $D _ { k }$ is the integer QK dot product for key k. The maximum code is $a _ { P } = 2 5 5$ for 8-bit PV and $a _ { P } = 1 5$ for 4-bit PV. Masked keys receive zero probability.

Let $z _ { k }$ be the stored value expressed in $V ^ { \prime }$ coordinates. The factors α and $\beta$ align the accumulated state and the current tile to the updated maximum (Dao et al., 2022):

$$
\begin{array} { l } { m ^ { \prime } = \operatorname* { m a x } ( m , b ) , } \\ { \displaystyle \alpha = 2 ^ { m - m ^ { \prime } } , \qquad \beta = 2 ^ { b ^ { \prime } - m ^ { \prime } } / a _ { P } , } \\ { \displaystyle l ^ { \prime } = \alpha l + \beta \sum _ { k } \widehat { p } _ { k } , } \\ { \displaystyle \sigma ^ { \prime } = \alpha o + \beta \sum _ { k } \widehat { p } _ { k } z _ { k } . } \end{array}\tag{10}
$$

The same quantized weights update the denominator and weighted value sum. Protected tiles contribute through FP16 $\mathrm { P V }$ into the same FP32 attention state. After the retained tiles have been processed, value restoration gives

$$
O = \mu _ { V } + \sigma _ { V } \odot ( o / l ) .\tag{11}
$$

## A.3 Compact Traversal and Compressed-Cache Execution

From boundaries to local work. The contiguous distance ranges let each query block construct its traversal from the two stored boundaries. Processing the protected, 8-bit QK, and 4-bit QK groups separately reuses the corresponding query representation while successive K/V tiles are staged. On RTX 4090, direct index calculation and reduced value staging lower the per-thread-block shared-memory footprint Across thread blocks, KV-group-major scheduling places heads sharing K/V together and visits query blocks in descending order to group accesses to common operands.

Mapping logical retention to physical storage. GQA query heads share physical K/V, so cache construction combines their requested prompt tiles into $\begin{array} { r } { \mathcal { R } _ { g } = \bigcup _ { h \in \mathcal { H } _ { a } } \mathcal { R } _ { h } . } \end{array}$ where $\mathcal { R } _ { h }$ is head $h \mathrm { { s } }$ retained set at the final prompt query block. Each tile is stored once at the highest requested key precision. A head mask preserves the participating query heads, and per-KV-head offsets locate the packed tiles in contiguous buffers. This shares stored operands while retaining each head's selected context.

Values use INT8 for the first and most recent 16 prompt blocks and packed INT4 elsewhere in the retained set. At head dimension 128, a 64-token INT4 K or V tile uses 4 KiB, half the 8 KiB required by INT8, before scales and metadata. The packed prompt cache is reused throughout generation, while new K/V enter the BF16 tail.

Combining prompt and generated context. The prompt numerator is accumulated in normalized $V ^ { \prime }$ coordinates, whereas the BF16 tail numerator uses original values. They must be reconciled before the final normalization. Let $( m _ { \mathrm { p } } , l _ { \mathrm { p } } , o _ { \mathrm { p } } )$ be the prompt state obtained by aligning and merging the thread-block partials, and let $( m _ { \mathrm { t } } , l _ { \mathrm { t } } , o _ { \mathrm { t } } )$ be the tail state. Tail scores use the prompt's key-centering offset. With $m = \operatorname* { m a x } ( m _ { \mathrm { p } } , m _ { \mathrm { t } } )$ , the final output is

$$
O = \frac { 2 ^ { m _ { \mathrm { p } } - m } \bigl ( \sigma _ { V } \odot o _ { \mathrm { p } } + \mu _ { V } l _ { \mathrm { p } } \bigr ) } { 2 ^ { m _ { \mathrm { p } } - m } l _ { \mathrm { p } } + 2 ^ { m _ { \mathrm { t } } - m } l _ { \mathrm { t } } } .\tag{12}
$$

The term $\sigma _ { V } \odot o _ { \mathrm { p } } + \mu _ { V } l _ { \mathrm { p } }$ restores the unnormalized prompt numerator; the exponential factors align prompt and tail maxima before combining numerators and denominators.

## B EXPERIMENTAL PROTOCOLS

## B.1 Models, Tasks, and Comparison Settings

The evaluated checkpoints are Llama-3.2-3B-Instruct, Qwen2-7B-Instruct, and Ministral-3-8B-Instruct-2512- BF16. Their respective numbers of query heads, KV heads, and decoder layers are (24, 8, 28), (28, 4, 28), and (32, 8, 34), with head dimension 128. The original projection weights are retained, and the attention implementation consumes their Q/K/V outputs.

General quality. General evaluation uses the Language Model Evaluation Harness (Gao et al., 2024a) with the recorded task configurations and dataset snapshots. MMLU uses five-shot prompts and subject-macro accuracy. These likelihood-based evaluations run the prefill attention path with the model's native positional configuration.

Long-context QA. LV-Eval (Yuan et al., 2024) uses the official dataset prompts and task-specific scorers at the 16K, 32K, and 64K dataset levels, with batch size one, greedy generation, and each task's configured response limit. Each prompt is processed in a single prefill pass before answer generation. Scores are reported separately for each dataset– length pair on a 0–100 scale, using keyword-aware F1 or DuReader ROUGE-L as specified by the benchmark. The length levels identify the supplied datasets; actual tokenized lengths follow the model's tokenizer.

References and allocation controls. BF16 uses default Py-Torch SDPA dispatch, while Torch attention restricts SDPA to its math backend. The al1\_int 4 reference retains all causal tiles and changes non-protected QK computation to INT4; the protected first and diagonal tiles retain INT8 QK and FP16 PV. MoA loads FP16 model weights, while the other quality comparisons use BF16 weights. The H100 quality environment uses Python 3.12, PyTorch 2.10 with CUDA 12.6, and Transformers 5.18.

Layer Same selects one boundary pair per layer by aggregating its heads’ candidate errors and sharing the resulting pair across those heads. All Same assigns $( w _ { 8 } , w _ { 4 } ) = ( 5 1 2 , 5 1 2 )$ throughout the model, corresponding to a 32K local 8-bit range with the first block protected. Random uses seed zero to permute the deployed boundary pairs across layers and heads, preserving the configuration collection and its aggregate modeled prefill cost.

## B.2 Measurement and Ablation Protocols

Complete-request measurements. System evaluation uses an NVIDIA A100-SXM4 40 GB GPU, H100 NVL, and GeForce RTX 4090. Within each model, comparisons reuse the same stored input token IDs across methods and GPUs. The core grid varies prompt length over{128, 2048, 8192, 16384, 32768, 65536, 131072}, batch size over {1,2,4,8}, and output length over {128, 256, 512, 1024}. The figures display the indicated prompt groups, with out-of-memory outcomes marked separately. Qwen2 system runs apply fourfold YaRN scaling (Peng et al., 2024) across methods. For prompts of at least 64K tokens, feed-forward computation is processed in 16K-position chunks across methods.

Two warm-up runs precede five timed repetitions. Each repetition generates 1,024 tokens greedily, with synchronized checkpoints at the reported output lengths. For a checkpoint containing $L _ { \mathrm { o u t } }$ generated tokens per request, TPOT is the elapsed time from the first token to that checkpoint divided by $L _ { \mathrm { o u t } } - 1$ . Complete-request throughput divides the batch's generated-token count by elapsed time from request start, including prefill. Latency, throughput, and energy use medians across repetitions, while peak GPU allocation uses the maximum observed allocation. NVIDIA Management Library (NVML) measures GPU energy during generation, reported in joules per 1,000 generated tokens.

Before timing, CUDA graphs capture decode and prefill with prompts up to 16K tokens. Timed requests include operand preparation, attention, and prompt-cache construction. Baselines use released generation interfaces.

Attention-level timing and storage. The efficiency ablations use seeded BF16 Q/K/V tensors with each model's head geometry and per-layer boundary table. CUDA events time complete attention calls, and per-layer median times are summed over the indicated layers. Prefill timing includes Q/K preparation, V packing, and attention computation; decode timing measures one attention step after prompt-cache construction. The PV comparison uses the first, middle, and last decoder layers. Prompt-cache storage counts retained K/V and associated metadata after prefill, excluding model weights, generated-token storage, and temporary workspace.

## C DESIGN VALIDATION

Head-specific allocation determines where attention computation is retained. Its execution cost also depends on the available precision levels, operand-staging space, and the representation accessed at every decode step.

## C.1 Value-Aggregation Precision

Value-aggregation precision. Retained tiles still require PV computation, making its precision another source of execution cost. With retained tiles and 8-bit QK fixed, Table 5 changes non-protected PV from 16 to 8 bits over the first, middle, and last layers. The 8-bit path is faster in all settings, reducing Ministral's 128K latency from 882.1 to 762.2 ms. Lower-bit value aggregation therefore complements the savings from retention and QK precision.

Table 5. Effect of value-aggregation precision on A100 prefill latency. The comparison covers the first, middle, and last attention layers with identical retained tiles and 8-bit QK computation. Nonprotected tiles use either 8-bit or 16-bit PV. Lower is better.
<table><tr><td>Model</td><td>Context</td><td>8-bit PV (ms)</td><td>16-bit PV (ms)</td></tr><tr><td rowspan="3">Llama</td><td>32K</td><td>72.9</td><td>79.2</td></tr><tr><td>64K</td><td>251.7</td><td>275.3</td></tr><tr><td>128K</td><td>865.0</td><td>992.7</td></tr><tr><td rowspan="3">Qwen2</td><td>32K</td><td>77.1</td><td>83.6</td></tr><tr><td>64K</td><td>269.4</td><td>295.1</td></tr><tr><td>128K</td><td>985.2</td><td>1,089.4</td></tr><tr><td rowspan="3">Ministral</td><td>32K</td><td>64.5</td><td>69.7</td></tr><tr><td>64K</td><td>216.0</td><td>236.1</td></tr><tr><td>128K</td><td>762.2</td><td>882.1</td></tr></table>

FINDINGS Low-bit PV reduces value-aggregation cost with retained tiles and QK precision fixed.

## C.2 Compact On-Chip Execution and Scheduling

Table 6. Shared-memory footprint and prefill execution on RTX 4090. Direct boundary-based indexing and reduced value staging decrease per-CTA shared-memory usage, allowing more concurrent CTAs per SM. Both layouts preserve equivalent attention computation. Lower latency is better. CTA denotes a GPU thread block, and SM denotes a streaming multiprocessor.

(a) On-chip resource usage
<table><tr><td>Metric</td><td>Original</td><td>Compact</td></tr><tr><td>Shared memory/CTA (KiB)</td><td>37</td><td>30</td></tr><tr><td>Resident CTAs/SM</td><td>2</td><td>3</td></tr><tr><td>Resident warps/SM</td><td>8</td><td>12</td></tr></table>

(b) Prefill attention latency
<table><tr><td>Model</td><td>Context</td><td>Original (s)</td><td>Compact (s)</td></tr><tr><td rowspan="2">Llama</td><td>32K</td><td>0.505</td><td>0.466</td></tr><tr><td>128K</td><td>5.609</td><td>5.149</td></tr><tr><td rowspan="2">Qwen2</td><td>32K</td><td>0.588</td><td>0.546</td></tr><tr><td>128K</td><td>7.565</td><td>7.036</td></tr><tr><td rowspan="2">Ministral</td><td>32K</td><td>0.810</td><td>0.743</td></tr><tr><td>128K</td><td>10.725</td><td>9.813</td></tr></table>

Shared-memory organization. Traversal metadata and staged K/V share the thread block's limited on-chip storage, so reducing their combined footprint allows more blocks to reside on an SM. On RTX 4090, direct boundary-based indexing together with reduced value staging lowers shared memory per thread block from 37 to 30 KiB (Table 6). The resource reduction changes how many thread blocks fit on an SM: a 7 KiB decrease per block raises residency from two blocks to three. At 128K, the compact layout reduces prefill latency by 8.2%, 7.0%, and 8.5% for Llama, Qwen2, and Ministral, respectively; Ministral latency falls from 10.725 to 9.813 s. This links traversal metadata and value staging through their shared resource budget: reducing their combined footprint enables a different execution occupancy for the same attention computation.

Ordering work that shares K/V. Query heads sharing a KV head repeatedly access common operands, motivating KV-group-major scheduling. With the selected tiles fixed and outputs bitwise identical, this ordering reduces 128K prefill latency from 7.803 to 7.347 s for Llama and from 15.092 to 14.266 s for Ministral (Table 7).

Table 7. Effect of KV-group-aware CTA scheduling on A100 prefill attention latency. Both schedules execute the same selected tiles and arithmetic operations, producing bitwise-identical outputs. KV-group-major scheduling places query heads sharing a KV head together to improve K/V data reuse. Lower is better.

<table><tr><td colspan="3">KV-Group-Major</td><td rowspan="2">Heaviest-First (s)</td></tr><tr><td>Model</td><td>Context</td><td>(s)</td></tr><tr><td rowspan="3">Llama</td><td>32K</td><td>0.654</td><td>0.681</td></tr><tr><td>64K</td><td>2.242</td><td>2.359</td></tr><tr><td>128K</td><td>7.347</td><td>7.803</td></tr><tr><td rowspan="3">Qwen2</td><td>32K</td><td>0.781</td><td>0.782</td></tr><tr><td>64K</td><td>2.752</td><td>2.790</td></tr><tr><td>128K</td><td>10.134</td><td>10.339</td></tr><tr><td rowspan="3">Ministral</td><td>32K</td><td>1.066</td><td>1.102</td></tr><tr><td>64K</td><td>3.856</td><td>4.053</td></tr><tr><td>128K</td><td>14.266</td><td>15.092</td></tr></table>

FINDINGS Reducing the combined footprint of traversal metadata and operand staging increases thread-block residency and accelerates the same computation.

## C.3 Prompt-Cache Storage and Decode

Table 8. Prompt KV-cache storage after prefill. The full-retention compressed cache stores all prompt tiles with 8-bit keys and mixedbit values. TriAttention additionally applies its selected retention and key-precision policy. Sizes exclude model weights and temporary inference workspace. Lower is better.

<table><tr><td>Model</td><td>Context</td><td>BF16 (GiB)</td><td>Full-Retention Compressed (GiB)</td><td>TriAttention (GiB)</td></tr><tr><td rowspan="3">Llama</td><td>32K</td><td>3.50</td><td>1.33</td><td>1.21</td></tr><tr><td>64K</td><td>7.00</td><td>2.64</td><td>2.10</td></tr><tr><td>128K</td><td>14.00</td><td>5.27</td><td>3.74</td></tr><tr><td rowspan="3">Qwen2</td><td>32K</td><td>1.75</td><td>0.66</td><td>0.66</td></tr><tr><td>64K</td><td>3.50</td><td>1.32</td><td>1.28</td></tr><tr><td>128K</td><td>7.00</td><td>2.64</td><td>2.51</td></tr><tr><td rowspan="3">Ministral</td><td>32K</td><td>4.25</td><td>1.61</td><td>1.47</td></tr><tr><td>64K</td><td>8.50</td><td>3.21</td><td>2.80</td></tr><tr><td>128K</td><td>17.00</td><td>6.40</td><td>5.45</td></tr></table>

Persistent prompt storage. Cache size depends on both the number of retained tiles and their stored precision.

At 128K, Llama's prompt cache occupies 14.00 GiB in BF16, 5.27 GiB with all tiles retained using 8-bit keys and mixed-bit values, and 3.74 GiB with the selected policy (Table 8). Low-bit representation supplies the first reduction; retention and key-precision selection save a further 29.0% over the full-retention compressed cache.

Table 9. Effect of prompt-cache retention and value precision on decode attention latency on A100. Each entry measures one decode step summed across all layers. Full Retention keeps all prompt tiles with 8-bit keys. The 8-bit Values control retains the selected context and key precision and uses 8-bit values with corresponding 8-bit probability codes on every retained tile. Lower is better.

<table><tr><td>Model</td><td>Context</td><td>TriAttention (ms)</td><td>Full Retention (ms)</td><td>8-bit Values (ms)</td></tr><tr><td rowspan="3">Llama</td><td>32K</td><td>1.62</td><td>1.68</td><td>1.95</td></tr><tr><td>64K</td><td>2.39</td><td>2.69</td><td>2.93</td></tr><tr><td>128K</td><td>3.79</td><td>4.77</td><td>4.72</td></tr><tr><td rowspan="3">Qwen2</td><td>32K</td><td>0.93</td><td>0.93</td><td>1.32</td></tr><tr><td>64K</td><td>1.81</td><td>1.82</td><td>2.15</td></tr><tr><td>128K</td><td>2.79</td><td>2.84</td><td>3.41</td></tr><tr><td rowspan="3">Ministral</td><td>32K</td><td>1.96</td><td>2.07</td><td>2.37</td></tr><tr><td>64K</td><td>3.02</td><td>3.29</td><td>3.73</td></tr><tr><td>128K</td><td>5.19</td><td>5.82</td><td>6.52</td></tr></table>

Repeated decode work. The compressed prompt cache is read at every generation step, so its retention and precision choices also affect recurring attention cost. Full Retention keeps all prompt tiles with 8-bit keys, while 8-bit Values preserves the selected context and key precision and uses 8-bit values with matching 8-bit probability codes. At 128K,

Ministral takes 5.19 ms with mixed-bit values and 6.52 ms with 8-bit Values (Table 9).

The additional benefit of the selected cache policy grows with context length. For Llama, its storage reduction over the full-retention compressed cache increases from 9.0% at 32K to 29.0% at 128K. Over the same lengths, decode attention latency decreases from 1.68 to 1.62 ms and from 4.77 to 3.79 ms, corresponding to reductions of 3.6% and 20.5%. The paired measurements link the policy's smaller persistent prompt cache to lower per-step attention cost.

FINDINGS Applying the selected policy to compressed KV storage reduces the persistent prompt working set and the attention cost repeated during generation.

## D EXTENDED EVALUATION

The selected boundary table is reused across tasks and request shapes. Additional models test whether reuse preserves prediction and evidence use, while the workload grids show how complete-request gains vary with generation length and GPU platform.

## D.1 Workload and Hardware Scaling

The workload grids vary prompt processing, generation length, and batch-level parallelism. Each comparison uses the same model, GPU, and request shape. Torch attention provides the SDPA math-backend reference. Generation-throughput ratios divide method throughput by BF16 throughput, while TPOT speedups divide BF16 TPOT by method TPOT.

![](images/7603de0be721ff03d0c98efb0ad4e41446109a105144fbdee320f57e09c4a119.jpg)  
Figure 8. TPOT speedup on A100-SXM4-40GB. Llama-3.2-3B-Instruct. Rows use batch sizes 1, 2, 4, and 8; columns use 128, 2,048. 8,192, 16,384 input tokens. Each panel varies output length over 128, 256, 512, and 1024 tokens. BF16 is normalized to 1 ×; higher is better. Method colors and order are fixed across panels.

![](images/331f985827f297fa83477463bd5e595ed19dace113d03c142de0a9e33e506a89.jpg)

![](images/ebd4593a452ed66e774c13892f83d644654b963d7d1c799e8b1dab7ae889f10f.jpg)

![](images/568e3f19d487fd1d7765ce328565b6f559af96a2e46bd9ece07fd420964751ab.jpg)

![](images/be70c5b4771b5da38f8bb9c99bf880175b89674594ae317738f4250f4aebbb4f.jpg)

![](images/b131c24c82e8eba528f8edd59a7792a32df2717e1a6c73d8c98187aa8332e3bf.jpg)

![](images/cf2230ee9c5ad79a4e960905e85ce2f6d1b8ed16c932130cb737806605987422.jpg)

![](images/dcc4355b4d989068cd5a0048fd4e23ae70e1527535c3cbd77b107982a66d90a8.jpg)

![](images/16bd9930b630b1027e6b8ae1d6da6a0acaeb741dc2125e5a45de8f9a7da33a11.jpg)

![](images/3aaa11cb340b6057ea10a5a925a0a12092f7ed57c59c882ed4c92c22b0b3c997.jpg)

![](images/fb8aac6ca0966786c3648385629913ad62098e5bda0708e77e40bba40ce621c0.jpg)

![](images/e909532829b9873571bc43a6119e8aede945cf2c9e2c30a611c0b1cbf5e050f1.jpg)

![](images/7b19a30d45d432eb69ff64f2ea7fe28ea2e1f4715d21f1386a9b81ad2a49b419.jpg)  
Figure 9. Generation-throughput ratio on A100-SXM4-40GB. Qwen2-7B-Instruct. Rows use batch sizes 1, 2, 4, and 8; columns use 128, 2,048, 8,192 input tokens. Each panel varies output length over 128, 256, 512, and 1024 tokens. BF16 is normalized to 1×; higher is better. Method colors and order are fixed across panels.

A100. Output length changes the balance between onetime prompt processing and repeated token generation. For Llama with a 16K prompt and batch size eight, increasing the output length from 128 to 1,024 tokens changes complete-request throughput from 137.07 to 394.97 tokens/s, compared with 88.55 to 156.58 tokens/s for BF16. The throughput advantage grows from 1.55× to 2.52× as the prompt cost is distributed across a longer continuation.

![](images/58196eb6580a61659daadd8cd29e6576fc8e590afa37791717628c11604e2463.jpg)

![](images/ddb3dc87c77fc0d12e82169b653ade6d4aa6dd33bf9d8a857c3a777439357d71.jpg)

![](images/78b5000116c11558224f600fe1b553b4293ad23e063725d83dec992b99f0fb5e.jpg)

![](images/a56db4022274b2e2f6079744d885ac85b45406a4af88b290499eb96f67674221.jpg)

![](images/f6406939ebd1fe1485ae86a085c1efb6df4cf2aa052bc9e514b28b6959f6775c.jpg)

![](images/dcd85bf46e2e4990ac71ed53f20c7118c0d652ac358ddfa7dad6771e6596357d.jpg)  
Figure 10. TPOT speedup on A100-SXM4-40GB. Qwen2-7B-Instruct. Rows use batch sizes 1 and 2; columns use 128, 2,048, 8,192 input tokens. Each panel varies output length over 128, 256, 512, and 1024 tokens. BF16 is normalized to 1×; higher is better. Method colors and order are fixed across panels.

Batch Size 4, Input Length 128

![](images/db5fa92bf1d270f7bceee1c7d5ff1ebc35c0db737290ea388f6904cb384c8f98.jpg)

![](images/958e37b5057b546a40369796bc79b9c3aa36625161adf0a619e21de391c02a45.jpg)

![](images/485938bb20bf0d39e4b0bb28908ea300e85da54ca5a5c75745e175d1e3d54b74.jpg)

![](images/7bfb21aa2ffd4dececcaa43322add5aa46836b5f9fc06fa30a0f1d610ac01b76.jpg)

![](images/daa4f632e5579ff9bff409a3e32367d1524705581b179b150808a66bdf8ac363.jpg)  
Figure 10. Continued. Rows use batch sizes 4 and 8; panel settings follow the first part.

![](images/7105e4a0761ce37807a23068e41ce0ead4448cac3a74b1c4f520def4e15aa1ed.jpg)

![](images/c23042d897b6de15c283c5d895c0ea43ca41c627698504eb93b50779bd58daa1.jpg)

Batch Size 1, Input Length 8K  
![](images/c1a2b247bc30b187c6439939c22794e3b686746113f86ec592aee4074a3725b1.jpg)

![](images/5520d8dee6178d083bbd5da8247e115fc38df8aa95517b1b38024c25c7091922.jpg)

![](images/add36f2627d025250491d2208c404f54666f74c89d7deedbd3d9a9ae4af22169.jpg)

![](images/067d5c5229a45c76b7d64baca8a9e6633b6d13c9c6d62b16a2fb1a4b1a433f14.jpg)

![](images/98b2bbe4a179e5e816312bfa7ae020668592fc9cdc44695e8b576f94568f0eba.jpg)

![](images/75dfae8c203ce759d16fce9046cdd94345e4385a191f2c9a619c698ee8417826.jpg)

![](images/03bdee89cfcbe28c5bf4d4ff250101fd2c6e9833da8999c7fe9f541c9cf030f7.jpg)

![](images/bbe8075ed5e9c55222f5358448bdb373728a27b02c80411e879239988540ddd8.jpg)

![](images/06e229e27e561894ff2f8fe29740b9a298c0f4b966791a4fa06fa8ac015b0285.jpg)

![](images/0d36a3313a322348f51419b65329eda8a70c27e6b5fa218873abbfa4c14219e8.jpg)  
Figure 11. Generation-throughput ratio on A100-SXM4-40GB. Ministral-3-8B-Instruct-2512-BF16. Rows use batch sizes 1, 2, 4, and 8; columns use 128, 2,048, 8,192 input tokens. Each panel varies output length over 128, 256, 512, and 1024 tokens. BF16 is normalized to 1 ×; higher is better. Method colors and order are fixed across panels.

Batch Size 1, Input Length 2K

![](images/0a3cbf0c2c03f4a7015b3613b38c635f1eae42b98a61a3ad58a96869af52ef00.jpg)

![](images/f5622f72b1c468a4b514fd050c32f45679b98b0774d0fde4e75b19fcb12b063f.jpg)

![](images/429a8ff416d4c1a4044ac26dec4beffb00e6b675a01d017cd52f38519fae36f2.jpg)

![](images/8ff1e647ed867a5c075c9d16bb5a2f2893d513d540ced930b090c8435f8860aa.jpg)

![](images/3b93cfefa634f85d1773a5b707092e579ee4f7e830ac82b503edc8a05b689527.jpg)

![](images/51a8b1953241fec5526c2eb226bf25e25626ab49190fad52b1200a83456d3f74.jpg)

![](images/9d4d05c76603477ebab481128cbf81bbfd4b73ec144ae95cf9770a3355631b08.jpg)

![](images/eb4c10c3290a6a780e47080f3157dff1a15feabbc2b5c38b03d2f0ec0797b09e.jpg)

![](images/d9bae15cd1c064420a85abaebc05ea6d45db39994dde39974b9375931c814c80.jpg)

![](images/3c3a6e418430d668049606796aaf5b0833607cde3c20a9b8c8edd5f54c428aea.jpg)

![](images/ef669bb65a31b9d11d56be694b31040866b14937c3cbdb7e9c275e3aed392db5.jpg)

![](images/8a28fab70347cd2590a7b716a6cd1987d0d01088d9f5fcc55469274cc087c20a.jpg)  
Figure 12. TPOT speedup on A100-SXM4-40GB. Ministral-3-8B-Instruct-2512-BF16. Rows use batch sizes 1, 2, 4, and 8; columns use 128, 2,048, 8,192 input tokens. Each panel varies output length over 128, 256, 512, and 1024 tokens. BF16 is normalized to 1×; higher is better. Method colors and order are fixed across panels.

The long-context sweep reaches 42.95 tokens/s for Llama with a 128K prompt, 1,024 generated tokens, and batch size one, compared with 16.19 tokens/s for BF16, giving the 2.65× speedup. Figure 8 complements the main-text throughput grid, and Figures 9–12 cover the other models.

![](images/410a4823876bae2797a8bd74b386f7ac4af342c003c9dbc19063d0f3f93a477b.jpg)

![](images/6a84387de383927d685c8efca8276a8f497b451fa5ef01cf100b4e79ea37faa9.jpg)

![](images/c5742692b1c772200f3169b9962cd6598c11d5d7c60c23347a3297641b412e8e.jpg)

![](images/1d08f511f7a691fc62e9c4bad93758f159ba08eafa54e4146dbd81b7edeee31a.jpg)

![](images/2714e55c9e5f861af7a9c42335d5c484067181373672aa768649bd6618f7c1f3.jpg)

![](images/56bc0c354da731a62761bcbe68ccde5b9a4e2c678e556a69b3b65034c8aed46d.jpg)  
Figure 13. Generation-throughput ratio on H100 NVL. Llama-3.2-3B-Instruct (prompt group 1 of 2). Rows use batch sizes 1 and 2; columns use 128, 2,048, 8,192 input tokens. Each panel varies output length over 128, 256, 512, and 1024 tokens. BF16 is normalized to 1 ×; higher is better. Method colors and order are fixed across panels.

Batch Size 4, Input Length 128

Batch Size 1, Input Length 128  
![](images/7c48830fd6b0e320cec263a9ec434821494078df3a3407847d1c0e1044fb2df4.jpg)

![](images/ce7b1d3b74595912ea61581b8c14b0498a398571665d5d80a345d251e04bc728.jpg)

Batch Size 4, Input Length 8K  
![](images/eea9dc277bd987705584980cf29128acdb4bb0436da24117c571d31eea36d7a8.jpg)

![](images/00d7d42ab789f418ada4158389a40d347b724d22ff8f046f1d83e145af3382a2.jpg)

![](images/0a8f45f822b1ddbb711dbeff1c933dbb643faaf341cd5c1e9671d63ade41adc0.jpg)

![](images/eb16e02f3769aec103e14f96fe14a4b20ba7eb3f0906c6d0e04e2e3cf428fd80.jpg)  
Figure 13. Continued. Rows use batch sizes 4 and 8; panel settings follow the first part.

![](images/c7376dfd6cba3d7c4ef100166d1f789591dfb06e41bd9167f5b82b3eb489b801.jpg)

![](images/e77f5e8ee484cf148117c070caba79a09b391221014ca55dc6c4a53d2fdaac03.jpg)

Batch Size 1, Input Length 32K  
![](images/0cc370f4b7fdcbccc47863ccd7ae185749495f550a831201a3f15e3a3c2454aa.jpg)

![](images/417d371438845bf59a1c7cf97b9d5c3e7ece947701bef15559aabb33ee073df3.jpg)

![](images/033efbadf622871060b92e22a40a401342206c6797e93c4fcf52dd145741d195.jpg)

![](images/82d7b260d1d848a495e40dc4f600dc2ae6d1530fd2baa6605d6e355af0d87494.jpg)

![](images/a28d7d1a0a54966d566ff89b02ab1d2a3e7b9ecc06451e2bc3e63aa08bba4e53.jpg)

![](images/d69ccb8c1b1d7a3ad3df17fd188efc346a81f435fd8ff38dfb34f6cccb7122ad.jpg)

![](images/b75385062a48d6c13d0d732013d8816bad297caf7c97a82b8eba94fd06f679a3.jpg)

![](images/52742a1e16de7b56c64e22b2c79809a102084fed79af3d309f95985710d046ca.jpg)

![](images/10ce1837e4740c9fee901fe0e2abafad2f1c2c1f62c8e75900bfc63503f0a8aa.jpg)

![](images/2e7aa822d2eb132945a83b2586e84d35af51c36b45a0e409bc1e8fab201e634d.jpg)  
Figure 14. Generation-throughput ratio on H100 NVL. Llama-3.2-3B-Instruct (prompt group 2 of 2). Rows use batch sizes 1, 2, 4, and 8; columns use 128, 16,384, 32,768 input tokens. Each panel varies output length over 128, 256, 512, and 1024 tokens. BF16 is normalized to 1×; higher is better. The 128-token column repeats the first group as a shared reference. Panels share colors and order.

![](images/91143d938aa2ed3d9e275e351f060462a2244a73003b12791159dc0f11de6ac9.jpg)

![](images/79f9d8d84b97d06987ec2cf883a12317ca82d0ea5b4bb039d0b6e5da98f906f6.jpg)

![](images/30716b79b9565cef4b0c1f6b21edad38404c69c48cb85b7855c3feb769172c70.jpg)

![](images/30a7bcd39fb857fe8e5f12c2da259b6bef3d3d2fc0db25dbce843207063b4711.jpg)

![](images/ab84993731db9bdd07b1b9af8debf0f6b6e7f6314c32ea44705d3d7f7ee15d2c.jpg)

![](images/9bdd0ac63deed84321b3efc9e73af3a10023bd87966b32eb27e2ef2803705f12.jpg)

![](images/0fc37db91e8ab4343211709234ce4b73df5f98590f57af70a971b42af6f9fca3.jpg)

![](images/cb455891f3dc698f7a6513bdcbf0205d2cae765187642296554e750b90e50bf0.jpg)

![](images/a12b8d7db5d946574ba5a17697d8116aa54bf7657f8abfe97106a9d4ff68fd32.jpg)

![](images/620a1a1ce43c72207f0861f15246a4d1f63e318a4844aec426e4533befa05c09.jpg)

![](images/d46ebff81b2549d500e54afea999bba5b9780729c5ac783a36fbc4c945d2c744.jpg)

![](images/4908046688ad57bf64a3cab5d4b23f5a11f22a632e0919d78560bb02f11b8691.jpg)  
Figure 15. TPOT speedup on H100 NVL. Llama-3.2-3B-Instruct (prompt group 1 of 2). Rows use batch sizes 1, 2, 4, and 8; columns use 128, 2,048, 8,192 input tokens. Each panel varies output length over 128, 256, 512, and 1024 tokens. BF16 is normalized to 1 ×; higher is better. Method colors and order are fixed across panels.

H100 NVL. On H100 NVL, Llama reaches 318.36 tokens/s with an 8K prompt, 128 generated tokens, and batch size four, compared with 40.42 tokens/s for Torch attention and 223.61 tokens/s for BF16 (Figures 13–18 and 21–24).

These complete-request measurements give throughput gains of 7.88× and 1.42× over the respective references.

![](images/40497bd6df2f948ff1275b46a9681fad913b0a94c48d3cfc96d44012236479fc.jpg)

![](images/d9aab28d466bd7d4e80747b5a6433b6e8eaaf19c9133b8aabf221a1387e80d47.jpg)

![](images/df3f4488abe907887674264c6ec26151e8f11465ed1425521bfe3abbda292e97.jpg)

![](images/9e59b47830496b8d36e52ba6de9b18612019cfaafca95c3e0289ae18cd7f1d32.jpg)

![](images/3f02c47d2637db88c55cf0f3e9fef2b92abcf6207587101a9a267ce326169ac2.jpg)

![](images/bb22d517b0df288826e5d25ebdf8dfdf5b87d3f2de7b4fa522a5f6a9e42b3038.jpg)  
Figure 16. TPOT speedup on H100 NVL. Llama-3.2-3B-Instruct (prompt group 2 of 2). Rows use batch sizes 1 and 2; columns use 128. 16,384, 32,768 input tokens. Each panel varies output length over 128, 256, 512, and 1024 tokens. BF16 is normalized to 1×; higher is better. The 128-token column repeats the first group as a shared reference. Method colors and order are fixed across panels.

Batch Size 1, Input Length 8K  
![](images/84842cc82a1cacd4dbb73f70a47361d7e1aaa9f743d1ae2170d7d9c3758f4804.jpg)

![](images/799307e7dcbb8bb2c7b9553de408f8135d996333ea79864edb157aae1f255d6a.jpg)

![](images/4edc55ca282d591079064ba3d73e6252a51c749217123d5bbbbdbf73902a1940.jpg)

![](images/08a6945c37b11f6be5fa97d39ee0fa253582fbb25b33ccb8ab15182182183466.jpg)

![](images/dff4db1f87a5ff65ca986e7ea9111f6b392a1960bee1df90a03a10805dde4360.jpg)  
Figure 16. Continued. Rows use batch sizes 4 and 8; panel settings follow the first part.

![](images/af12fc5d8ff9bc1c34ca8f9a62bb286d3ba917498aee287cf44bd13cc4eeb80b.jpg)

![](images/7e190100b2392c8e8a0f6731aa0b801b1bf01a2bb57f08bd2f1699f078f22275.jpg)

![](images/b690fe4060f96971d9049758f13ab68770de92d11131be01e7678efe5c1f26d1.jpg)

![](images/d2f4fff8cbfc52ea108f3e84d54ec61ff9e8fd9d6e849ca024d91cf615ceec11.jpg)

![](images/c99524b3e0080751bf91c144ee96243130dbaddc95cea68fb871530294769faf.jpg)

![](images/7887cbf3da495904ddbd3c3709c23e76da5a750baaadeb26d8ac08268520ec92.jpg)

![](images/62698172d9671ebf07094ccee23f91529d19be508dc2a3b09e4f8efabbf8e51e.jpg)

![](images/b1dcdfda3551355b0b65df431181e640dbb6e0b49ef69605a7cd69c1ca39e424.jpg)

![](images/937ec935d7d3ce1d7b5b86b9f026a007d8fc9152d069134ed9f2dcf5132d7c47.jpg)

![](images/ae4a82d26a9086a3987d972615bd5c5b3a051a2095a21ef232d2fdff539e459b.jpg)

![](images/ba4237d076e6ec81aa33e36537d15e906326f896a468b8112145e583919d3a64.jpg)

![](images/1e8b7cb639e08aaead51f61c76ad11630653c1687fc4a60f24085565bf06d50f.jpg)  
Figure 17. Generation-throughput ratio on H100 NVL. Qwen2-7B-Instruct (prompt group 1 of 2). Rows use batch sizes 1, 2, 4, and 8; columns use 128, 2,048, 8,192 input tokens. Each panel varies output length over 128, 256, 512, and 1024 tokens. BF16 is normalized to 1 ×; higher is better. Method colors and order are fixed across panels.

Batch Size 1, Input Length 8K  
Batch Size 1, Input Length 128  
Batch Size 1, Input Length 128  
![](images/4470ba1178c64547ab8c8f40da62304c87e953b35fc64806733c1245818e23b8.jpg)

Batch Size 1, Input Length 2K  
![](images/90561b824b85019f0fa38f56a7102c048ad3d8245dd32d3237a3daa689b80de6.jpg)

Batch Size 1, Input Length 8K  
![](images/294620938e9834010ec1266d0a54b011e9c392e9f13b245d71427ad6bbc897c3.jpg)

![](images/688e4ec0172cab2eb0fab33bd3c1d47cd558f8ab413e331878ad3aea6ba7fa96.jpg)

![](images/c5e37572d227be6684d5ce4c9fa539c03b5e8f41247be2e93ef1c8c7c8e31ca8.jpg)

![](images/51d955310330eee1be8ce91ce978d5af5eff8e414aecb3e324b3977a4bfe8dc9.jpg)

![](images/3459d946a3886f312ada866a58d14acfae9eb5bdf6b9ebb26eaa572bafc100be.jpg)

![](images/4d75b7482a312f9b4d13da3184547de1576ba55f13852644d1780fe33c92852e.jpg)

![](images/4798335b319a207a60bd353711470843bdeb086c55faa23a3e7923d9ae8420e7.jpg)

![](images/859a6cdc77300e33ef9d5a2628d148206c7301a5626f29c7b723773f74bc0f75.jpg)

![](images/3b61267c127fb13e7fedd6ef87d98b7403b7b79f68abc5d207bd08d095e6a0fb.jpg)

![](images/8164347ed6f4e95ac9c0403c954c5b41f3c5e93e069cffa84846c55e73d6282d.jpg)  
Figure 18. TPOT speedup on H100 NVL. Qwen2-7B-Instruct (prompt group 1 of 2). Rows use batch sizes 1, 2, 4, and 8; columns use 128, 2,048, 8,192 input tokens. Each panel varies output length over 128, 256, 512, and 1024 tokens. BF16 is normalized to 1×; higher is better. Method colors and order are fixed across panels.

![](images/8f2ea71d97c645e817331693543ca7b0b8d0f68a07d06ebe1ac1b40f8f3ce7c7.jpg)

![](images/344eb8db6b8a04bfe3ede6e09dc9f241d92190f95de9c478cad23306ca627114.jpg)

![](images/1733bafdeeab9c52a92a3370c76a61ce521398c8bb1e4960d5729556cd7f8e79.jpg)

![](images/47764ee1ede000f0e87df85734fcdb179d5d0fca4f5bf90a2107338e41fb1fb1.jpg)

![](images/bf0cae026f8d4c94ea2b9d3e32cc1801c7acd8e7699a2f9681a3d8bc6d5858a4.jpg)

![](images/b7753c470a805ae361b364984375c4d585626391d5b111627d46acae3100981d.jpg)  
Figure 19. Generation-throughput ratio on RTX 4090. Llama-3.2-3B-Instruct. Rows use batch sizes 1 and 2; columns use 128, 2,048, 8,192 input tokens. Each panel varies output length over 128, 256, 512, and 1024 tokens. BF16 is normalized to 1×; higher is better. Method colors and order are fixed across panels.

RTX 4090. The compact-layout implementation also improves complete-request throughput on RTX 4090 (Figures 19–20 and 25–28). With an 8K prompt and batch size four, Llama reaches 167.69 tokens/s for 128 output tokens and 316.54 tokens/s for 1,024 output tokens, compared with 111.39 and 164.97 for BF16.

Batch Size 1, Input Length 8K  
![](images/bbaf449ae02ba53ad9191f3d07e5c8f0be9d73710173b3cd26185c5b15b3736a.jpg)

![](images/48cadfb571e22f9aedeca0fd884e6d34c4c9c97d476466b4c6b2ea275c612d3f.jpg)

![](images/8cac3c8d263625cca2e523f1f6d52ebce38a98c63b90df75cfe594798bd2ee5e.jpg)

![](images/1ecb05ab88ab089b3bd56b2b4b06006d1a7b5c892c51a4fe3fc6230a026a8431.jpg)

![](images/f76d6c02f33c94ec5e10c84ede72b065d8f4c29f44bc34377bf5bc2297a42735.jpg)  
Figure 19. Continued. Rows use batch sizes 4 and 8; panel settings follow the first part.

![](images/d3ac68293b95b6871ddd03f469e14ae8a4895d6b58338cd1df19396b5e9ba440.jpg)

![](images/64deae21c5a0c10a490d1fb2da4e606d0e0a38839f3fdb5f2302e4440cd95697.jpg)

![](images/aeff4ef27c7ea9d53757e0cc04bd3a4b7591c6f7b1bca65a15a76ff6b0fbaf6e.jpg)

![](images/c67df18a33bdf7a43fc1b08f2960f8b62a9b26e23c20a9e2119efcf44f399530.jpg)

![](images/e3a164a19858893670dec296e76e8d2e1f281659c18c9c9cf9509eb856b599f2.jpg)

![](images/780807dfaaa57eb83db972d6be9987d8d06689c2d66d040fadb57fb8e8918d4c.jpg)

![](images/85b6328e18c8c85e39c8047e4b74f7ca7acaa83a10c2415947aeff6f22efa5ab.jpg)

![](images/68c6a7d406f6f1f811c3e6f4cba3460a06ada2d1a56f5bfe4230ceca54b84e50.jpg)

![](images/231abde5dca286086409575db146ff7db16e70eb3945d84ebecb8ae63750f870.jpg)

![](images/eb31312ad731aa4d353ef59984db62d241f1fb1706b67b9e0d9957032af16abb.jpg)

![](images/40b4680e9f8160113509824d227cfcd62d2e470ec62aa32bc5f221df2ba185bb.jpg)

![](images/0c08da4e7b055e617aeb74bac2a6056c2c1874438208c187442e8b04e0ac892f.jpg)  
Figure 20. TPOT speedup on RTX 4090. Llama-3.2-3B-Instruct. Rows use batch sizes 1, 2, 4, and 8; columns use 128, 2,048, 8,192 input tokens. Each panel varies output length over 128, 256, 512, and 1024 tokens. BF16 is normalized to 1×; higher is better. Method colors and order are fixed across panels.

The corresponding ratios increase from 1.51× to 1.92×, extending this trend to this GPU.

FINDINGS In A100, H100 NVL, and RTX 4090 comparisons, throughput gains over BF16 grow with continuation length. Longer continuations increase decode efficiency's role in complete-request performance.

![](images/7e40ae4fcbcadf59657950ebca86830317a7a05f3acd92c6d6812cf2951e55d3.jpg)

![](images/c2da6385c587b0094df9e82b3e48c6a7e6027f0e97d6f59487230b6501a07464.jpg)

![](images/d90c3ecbb6d0e861192cfb8dea410c1122eac536863c72c40ca58490350bed4d.jpg)

![](images/eee8d5665be0e1f6241096e7ddcfeb6ae76089321c990c689a01724c8f25e049.jpg)

![](images/2738c02b12898844b81ef0891cd8b7bc06387f852bb728866fc51956d7d3fc7a.jpg)

![](images/d9ec5f619c7153cad3249291a44de7c0a14e069f30eb3522e9005da90900ef1f.jpg)

![](images/0ef7fed490b860889c7a8737e5ad5844683041d5b7ef658a85f374601260cf25.jpg)

![](images/e009a75de62f69c0d73a828360208763c416a084831ea8426c879cdc0755de24.jpg)

![](images/cdb8de9218eb80911cd92efe87acfc3aedd0649cb16e4abddef2cfa5a915583a.jpg)

![](images/f10162e013fc7b4645d6ccbabe7a0d80ad948649f7d61d81ede06936bb6c0e50.jpg)

![](images/31f23b8ddeaa87291f0e325b80a6547a8137e441ce399fb0bd0a685534af5271.jpg)

![](images/ec054a762e111b1ae42fb592b8e43c18b9199d8fa2124cb42b46ad3509d16ddb.jpg)

![](images/85d05112826edabecc17123dcbe9bfd69c50d650225565230e366520150c1f0b.jpg)

![](images/46089852cb1a0850ddba81c8a151e00f2a918bcd354e2141750820ea69a26992.jpg)

![](images/9afd3200920aa02ac8febd8f474d5d86e0c95f641251451c3c545243d0029c05.jpg)

![](images/31667b47200b4bb308fb855edd7a9565b833e16fa5ec6ee5d6c4cd5add6e21bb.jpg)  
Figure 21. Generation-throughput ratio on H100 NVL. Qwen2-7B-Instruct (prompt group 2 of 2). Rows use batch sizes 1, 2, 4, and 8; columns use 128, 16,384, 32,768, 65,536 input tokens. Each panel varies output length over 128, 256, 512, and 1024 tokens. BF16 is normalized to 1×; higher is better. The 128-token column repeats the first group as a shared reference. Panels share colors and order.

![](images/6e667b156488bd54af5145a760d9b80efa3a32e761b99cc9cb0b532e5207d7be.jpg)

![](images/cfd9382246549abfc6ef11e214d6f9c6b1224f64095db1153c82da51a46cec91.jpg)

![](images/1706207542912bd9141d3f50a58814e24508e0687ecbdb3fd365fc0566138fce.jpg)

![](images/2db57a24f7a442e4bdf0980974c0b09e42f01507943b4fce97e2d19d79064977.jpg)

![](images/5be6549f82462326068f0d522b196fe8236c11487860d465b4bad8d0943c6bad.jpg)

![](images/3384fa6d2c6173010e62914b78d09cf5396ae111c11d89573dbe96021bbdd421.jpg)

![](images/fb3d5af185a02cc266cef3b0e1519d5198cb8886e95ff858a54cf4fe8abec2df.jpg)

![](images/1c665c3293d16a0a818585d6109bcc26f5f79f5f0a00644177b471ef8947ee6f.jpg)

![](images/3c1b2c9610a14fa51972195ba5f16a0a17575824c5e6b3883a514ebe7b5af420.jpg)

![](images/4e28ae4a2483ec735851babb9c4235e5452d905281f8edfcdf4664598995fa99.jpg)

![](images/67ce6afb3b0a2ba995f0feb33d5cda757b865e9565a9d9737ac577c25ad41cdc.jpg)

![](images/94a447936f01b8b649601e448ece971704868a96f8bee5206f800fc8ccd488aa.jpg)

![](images/bc4a325a1737073d6e78190331b00855048389ef89bd0b4e2190d86cd26d33a6.jpg)

![](images/4b6f04ca47e963006bb39eee6fc592cbac6c1aaf70ea816d402a37f06c1b473e.jpg)

![](images/ea43ffdef8dee77a8d7a4d2fb6e433433483f59122d7642e4839dd203a512bc6.jpg)

![](images/4a8ea54367c4d513ccce1272d2d5269747c9930ba95259121c9a43d8ef5e1d0c.jpg)  
Figure 22. TPOT speedup on H100 NVL. Qwen2-7B-Instruct (prompt group 2 of 2). Rows use batch sizes 1, 2, 4, and 8; columns use 128, 16,384, 32,768, 65,536 input tokens. Each panel varies output length over 128, 256, 512, and 1024 tokens. BF16 is normalized to 1 ×; higher is better. The 128-token column repeats the first group as a shared reference. Method colors and order are fixed across panels

Batch Size 1, Input Length 8K

Batch Size 1, Input Length 128

![](images/1b27b10117d655f70e4f1601f4d18c7600d01a87a161ea63d43a7e4536c73473.jpg)

![](images/698da214ecf64b3470698eab18f6ce7700f1e4ff7a260cbdfe0769f0ee1bfba0.jpg)

![](images/32c3249c9c564234f78b5e9fbe6543d568eb59f5b90fc6b123409faffcd09d0a.jpg)

![](images/c6278728acccd1e3f1289f077ffafe363c7d38aa207f8f2b2a41e7e821d04613.jpg)

![](images/c755d30ff7783615bc58b7c0bfb3d723d6ec09362f86810154beda892b162f2f.jpg)

![](images/c1cf5b9c867b4d7a4368c9eef0e98b208894e85130dde3d774f3120b7ce3024a.jpg)

![](images/139b6a07ddb748cf6f5e01842618b326644926e28a90661ae18b13088af11c53.jpg)

![](images/22bacf6e5cb12017c859dbaa2322f3a8793d3764c1822acaaa2f16814188d859.jpg)

![](images/4609f074cb6abe95d0148b18b9ea79ccad9d2e51a3203f92ac46118d10a06e6c.jpg)

![](images/64044db48c6e092674f3add0065fe7d26f62728d7f00977831f6e73ec20521f9.jpg)

![](images/5b453a636737d09311bdfce4680fe668b7f79290392661281bcedfda99fd2dc6.jpg)

![](images/740c9cf0f1bc6b30c4486c03b9fa0949b68c7a93149307fdda884f89ddc9ccf9.jpg)

![](images/b92526d45daba30c04bcb1175dbb67061ad6de9f7eb6957e514263e7e614b21d.jpg)

![](images/08bbbb8d19516df6eb4fefd9505394eae530a452ed5a4298b3f5fb7da6ace3aa.jpg)

![](images/04ea4d625189beebb35a6e94e00270393d3b012d43a412af27cbdecdeecfd34e.jpg)

![](images/e37b44b006df70a9eee0abd351e83be1751b138e80b8151f2df6d5675f7734db.jpg)

Figure 23. Generation-throughput ratio on H100 NVL. Ministral-3-8B-Instruct-2512-BF16. Rows use batch sizes 1, 2, 4, and 8; columns use 128, 2,048, 8,192, 16,384 input tokens. Each panel varies output length over 128, 256, 512, and 1024 tokens. BF16 is normalized to 1×; higher is better. Method colors and order are fixed across panels.  
![](images/e9a1e84f82586a4a55874efe264cb73d5ccf187fc0c2aafd4ee07366e399d5a0.jpg)

![](images/2af452c1711617f07b859705a03288ca7548c580207ab9d91788201f4d6280e8.jpg)

![](images/4076c3feaa6c69e8f02bc90a7b0f6cee87b25ad5f9ad533969cf06056817f370.jpg)

![](images/3863227b822039ceacb2800f5f7baad483d9b297c18e456ded2ebf0f8556453f.jpg)

![](images/8e0df397a171b1756c20722c09cd468e80025e247cb5bb9f136005cd10830392.jpg)

![](images/a2c3afc3fc9fb885383ea636d47f4f121b9659076fc4317c72cacaaecbcc2471.jpg)

![](images/a1844d22da4c51ce54462fea92e85f19171bef75ad7b9fd2c98e7a6def776054.jpg)

![](images/286503f7e46a3e13488f6ebdde69d3bc19967d2856d0b6cf5e8ccd07e0ae3eb1.jpg)

![](images/ca943273a57a5d54e882aa8ed580c25178e36cabf7fee65a6ed4852b6d71b06d.jpg)

![](images/d673b020eb55c865e68383ae3a7f5dd9f2534b221f186433573137411a8e99b9.jpg)

![](images/99a09946ec1f5cd5c06258a9f36704c6ba37bced7327b069225e35c93a334338.jpg)

![](images/33b3dfb35168e5b970e362fa01b9afc15bd2995cd1ed8bddd8acfebe50b7e2f5.jpg)

![](images/5646188497bc90f819cddc11f05aad55ca046b8fecbf545760e342799aee4b84.jpg)

![](images/7a35a49d53552359f9976cdc7793c8fe175e26a247bbfa6da1e348d4b1e0ba80.jpg)

![](images/b1c15bb1d1f5f9209d3a2805bf3b85a1ea0031162d05e13ec26d099b0f56974c.jpg)

![](images/b40b9c87dab31d622df0e782602dc80bc8ff7ff200c0e3ed1f17aa49c5ae2769.jpg)  
Figure 24. TPOT speedup on H100 NVL. Ministral-3-8B-Instruct-2512-BF16. Rows use batch sizes 1, 2, 4, and 8; columns use 128, 2,048, 8,192, 16,384 input tokens. Each panel varies output length over 128, 256, 512, and 1024 tokens. BF16 is normalized to 1 ×; higher is better. Method colors and order are fixed across panels.

![](images/4322b15f1c7a4c711e7aa3670b066d31442aa6008554f5145bb6ea9562d05d6a.jpg)

![](images/3ed7d3b86666627d0159fd73080d65b9d2f5a73e63e6e88bc82636c96a64dc6b.jpg)

![](images/32060e5fd9c46349621f146003518fba56b48894fdc0c3fe6f2c087d8484bec8.jpg)

![](images/83d205298c062130864604be984ba7b13b8143d030939465c0db84819e704b66.jpg)

![](images/e885e49cf437e0e71658023cce19220ddfe9c55e68e1127f639771ef0471e22d.jpg)

![](images/1ab192a5f262004982f7e65ec580e80d0e28b58671a542a348a8ffb6183066f3.jpg)

![](images/114450d5761db9948edbe194f6f3c5ba50c6ac6a924a0fec46c0c72987be2683.jpg)

![](images/2313d431501fd944274b754b06ea11ac1314f3febf88bed359581b0d0b920382.jpg)

![](images/78e63e254fda236ca5cd1cb43c0a71187865328c3af46ca1bcf5af6b28cb6234.jpg)

![](images/63b22ccda7c3996459b9a26f02b9dfabd39f4818c64804e5e3a54ca35cf9d0d1.jpg)

![](images/a0218980eeee4bdf8c9caf078c9372b967d3325f37a11da9ca252f7b3b53ca47.jpg)

![](images/f529036d2fa45bf6fb487de54678258bd3e8421f3687c159e482e59905ebbdb2.jpg)

![](images/d8f1671be039015ffe4133634066ad161174d684acc6fed50285851e2430df29.jpg)

![](images/6e96f2c0d79b7504127220c8f493e146b7e1459e48ea6a2660973a506a2588d7.jpg)

![](images/43beac9f6c16b5039d27c4d5a39d4dc0fe265421469a602fc2a907de116c30a8.jpg)

![](images/e92bf924b669e101e1d87b353564411f353bee9889c288a732e055820bfeb544.jpg)  
Figure 27. Generation-throughput ratio on RTX 4090. Ministral-3-8B-Instruct-2512-BF16. Rows use batch sizes 1, 2, 4, and 8; columns use 128, 2,048 input tokens. Each panel varies output length over 128, 256, 512, and 1024 tokens. BF16 is normalized to 1×; higher is better. Panels share colors and order.

Figure 25. Generation-throughput ratio on RTX 4090. Qwen2- 7B-Instruct. Rows use batch sizes 1, 2, 4, and 8; columns use 128, 2,048 input tokens. Each panel varies output length over 128, 256, 512, and 1024 tokens. BF16 is normalized to 1×; higher is better. Method colors and order are fixed across panels.  
![](images/89faf0b7704ed9757c0b88e0978848cc5728e109eb258e0044184d81a240f8cb.jpg)

![](images/c81608db431b386c85649b2e368130d866de130dca51bfe98c74c9ac20742f2b.jpg)

![](images/536f5ea77de65a374f4ae17d29306c61d9cbe901703236cefeb24c93db93ae86.jpg)

![](images/71acba6641869077583d011795ec11b7bab0f153598619174a7071ec3be8d46c.jpg)

![](images/bbfdfa10cf5023b085fab05ec31a56e907db55c8ac00a66f8de9ff0cc775d424.jpg)

![](images/f5600bf763657cf4b18b43b38f35bae0d7257a6a8df21681ab695acec849c615.jpg)

![](images/e1e891a233776b88bb195205f3bb3da8324e278e9fac85771e2e8eb3dcfdecd1.jpg)

![](images/e6d23499b7f59468b4c68871dbfbe8b2aefd4613e2cc190cfd8ab81eb2fd40de.jpg)

![](images/bedf0a024adb7515a02373ddf4c196fdfe77759aa8c181890e5fe82e8281659c.jpg)

![](images/e7e422e58cbe9a6db94b0b025daa33ad32cb7b76302ac7f73795b1ba12e933c1.jpg)

![](images/efb0626ea2096412a57e810adaab62dff1604e48045e79e5cfdd95444bf545a2.jpg)

![](images/d99272372319e0134b0e865e5de2f1b301447e832d7fde7f22ed88c9a900adf4.jpg)

![](images/462130ea81c8ef35182bdba40f09396b71d384c872fe999ae4dd83f3c85ea7d8.jpg)

![](images/71c7f2a0d24b97dff8e481d5c576699d9433e3829ad17e2c653ac751ca8dc7fa.jpg)

![](images/e56bde4735f989f5ad66489a48cf724183151d1910a802c7df7b3476f68b55ee.jpg)

![](images/bc2a020e3919007fb782fe9b783c6811132916937c7b28ca0cc3f55fd7579221.jpg)  
Figure 26. TPOT speedup on RTX 4090. Qwen2-7B-Instruct. Rows use batch sizes 1, 2, 4, and 8; columns use 128, 2,048 input tokens. Each panel varies output length over 128, 256, 512, and 1024 tokens. BF16 is normalized to 1 ×; higher is better. Method colors and order are fixed across panels.  
Figure 28. TPOT speedup on RTX 4090. Ministral-3-8B-Instruct-2512-BF16. Rows use batch sizes 1, 2, 4, and 8; columns use 128, 2,048 input tokens. Each panel varies output length over 128, 256, 512, and 1024 tokens. BF16 is normalized to 1×; higher is better. Method colors and order are fixed across panels.

## D.2 General and Long-Context Quality

Table 10. General-task results for Qwen2-7B-Instruct. Accuracy is reported in percent (higher is better); WikiText-2 uses word perplexity (lower is better). BF16 is the upper reference; a11\_int 4 is the lower reference. Colors compare with BF16: green denotes equal or better results, blue worse results, and gray BF16. Bold marks the best result among Ours and Others; displayed ties share the mark.
<table><tr><td rowspan="2">Benchmark</td><td rowspan="2">Metric</td><td>Upper Ref.</td><td>Lower Ref.</td><td>Ours</td><td colspan="5">Others</td></tr><tr><td>BF16</td><td>all_int4</td><td>TriAttention</td><td>MInference</td><td>FlexPrefill</td><td>SageAttn</td><td>MoA</td><td>SpargeAttn</td></tr><tr><td>WikiText-2</td><td>PPL↓</td><td>9.68</td><td>9.73</td><td>9.68</td><td>9.67</td><td>10.15</td><td>9.68</td><td>9.75</td><td>9.70</td></tr><tr><td>HellaSwag</td><td>Acc. ↑</td><td>80.62</td><td>80.62</td><td>81.12</td><td>80.63</td><td>80.65</td><td>80.65</td><td>80.65</td><td>80.62</td></tr><tr><td>MMLU</td><td>Acc. ↑</td><td>71.62</td><td>71.16</td><td>72.07</td><td>71.45</td><td>71.60</td><td>71.34</td><td>71.64</td><td>71.51</td></tr><tr><td>ARC-Challenge</td><td>Acc. ↑</td><td>62.37</td><td>62.29</td><td>62.96</td><td>62.37</td><td>62.54</td><td>62.46</td><td>62.37</td><td>62.29</td></tr><tr><td>LAMBADA</td><td>Acc. ↑</td><td>70.04</td><td>69.94</td><td>70.44</td><td>70.06</td><td>69.84</td><td>69.94</td><td>70.11</td><td>70.06</td></tr><tr><td>PIQA</td><td>Acc. ↑</td><td>80.52</td><td>80.52</td><td>81.02</td><td>80.52</td><td>80.47</td><td>80.79</td><td>80.52</td><td>80.52</td></tr><tr><td>WinoGrande</td><td>Acc. ↑</td><td>76.56</td><td>76.32</td><td>76.82</td><td>76.24</td><td>76.48</td><td>76.80</td><td>76.24</td><td>76.95</td></tr></table>

Table 11. LV-Eval QA results for Qwen2-7B-Instruct. Dataset identifiers follow Section 5.4. Scores use official task-specific metrics on a 0–100 scale; higher is better. Length labels follow the official dataset levels. BF16 is the upper reference; a11\_int 4 is the lower reference. Colors compare with BF16: green denotes equal or better results, blue worse results, and gray BF16. Bold marks the best result among Ours, Others, and Ablations; displayed ties share the mark
<table><tr><td rowspan="2"></td><td rowspan="2">Dataset Level</td><td rowspan="2">Upper Ref. BF16</td><td rowspan="2">Lower Ref. all_int4 TriAttention</td><td rowspan="2">Ours</td><td colspan="4">Others</td><td colspan="4">Ablations</td></tr><tr><td>MInference FlexPrefill</td><td></td><td>SageAttn</td><td>MoA SpargeAttn</td><td></td><td>Layer Same</td><td>All Same Random</td><td></td></tr><tr><td rowspan="3">①</td><td>16K</td><td>23.85</td><td>22.22</td><td>23.49</td><td>29.00</td><td>26.13</td><td>32.60</td><td>16.32</td><td>20.85</td><td>22.94</td><td>21.78</td><td>23.12</td></tr><tr><td>32K</td><td>24.74</td><td>19.99</td><td>25.65</td><td>15.00</td><td>11.89</td><td>16.90</td><td>16.55</td><td>21.19</td><td>23.01</td><td>23.26</td><td>20.49</td></tr><tr><td>64K</td><td>27.46</td><td>20.68</td><td>29.70</td><td>6.40</td><td>8.98</td><td>9.10</td><td>19.40</td><td>26.15</td><td>24.05</td><td>17.07</td><td>21.15</td></tr><tr><td rowspan="3">②</td><td>16K</td><td>16.00</td><td>14.07</td><td>16.62</td><td>14.10</td><td>15.74</td><td>14.20</td><td>14.86</td><td>15.87</td><td>14.56</td><td>14.74</td><td>13.59</td></tr><tr><td>32K</td><td>16.42</td><td>14.94</td><td>16.88</td><td>11.50</td><td>15.24</td><td>13.50</td><td>16.20</td><td>15.61</td><td>15.87</td><td>15.60</td><td>16.03</td></tr><tr><td>64K</td><td>13.70</td><td>12.20</td><td>14.45</td><td>9.50</td><td>12.65</td><td>10.80</td><td>13.49</td><td>13.19</td><td>13.89</td><td>11.83</td><td>13.07</td></tr><tr><td rowspan="3">③</td><td>16K</td><td>12.43</td><td>16.64</td><td>13.36</td><td>14.30</td><td>12.21</td><td>12.12 11.91</td><td></td><td>12.74</td><td>11.89</td><td>12.27</td><td>12.76</td></tr><tr><td>32K</td><td>9.33</td><td>14.04</td><td>9.94</td><td>11.46</td><td>15.16</td><td>9.76</td><td>10.50</td><td>8.73</td><td>9.85</td><td>7.98</td><td>10.51</td></tr><tr><td>64K</td><td>11.82</td><td>13.21</td><td>12.18</td><td>11.15</td><td>13.14</td><td>10.84</td><td>9.95</td><td>10.97</td><td>7.69</td><td>10.24</td><td>7.09</td></tr><tr><td rowspan="3">④</td><td></td><td>14.89</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>15.37</td></tr><tr><td>16K 32K</td><td>16.04</td><td>14.82</td><td>16.87</td><td>15.79</td><td>15.47</td><td>15.65</td><td>13.75</td><td>15.67</td><td>14.62</td><td>15.07</td><td>12.15</td></tr><tr><td>64K</td><td>14.01</td><td>12.53 12.91</td><td>14.11 14.69</td><td>14.88 12.78</td><td>14.17 13.96</td><td>13.66 13.70</td><td>10.44 11.62</td><td>13.61 12.94</td><td>11.95 12.90</td><td>13.30 9.90</td><td>10.95</td></tr><tr><td rowspan="3">⑤</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>16K</td><td>18.24 18.04</td><td>18.94</td><td>20.31</td><td>14.90</td><td>17.09</td><td>15.80</td><td>16.74</td><td>17.14</td><td>18.15</td><td>15.30</td><td>15.12</td></tr><tr><td>32K 64K</td><td>17.57</td><td>18.15 16.01</td><td>18.58 17.70</td><td>12.60 9.00</td><td>16.08</td><td>13.80</td><td>17.78</td><td>18.48</td><td>15.37</td><td>16.46</td><td>16.37 16.31</td></tr><tr><td rowspan="3">⑥</td><td></td><td></td><td></td><td></td><td></td><td>10.29</td><td>11.40</td><td>14.49</td><td>14.75</td><td>16.60</td><td>12.88</td><td></td></tr><tr><td>16K</td><td>17.33</td><td>15.02</td><td>17.27</td><td>15.20</td><td>13.65</td><td>15.10</td><td>14.08</td><td>16.06</td><td>15.93</td><td>15.35</td><td>17.47</td></tr><tr><td>32K 64K</td><td>15.41 14.67</td><td>14.88</td><td>15.91</td><td>10.80</td><td>12.26</td><td>10.80</td><td>13.99</td><td>15.44</td><td>14.19</td><td>11.76</td><td>15.55</td></tr><tr><td rowspan="3">⑦</td><td></td><td></td><td>14.79</td><td>14.31</td><td>8.10</td><td>9.46</td><td>9.50</td><td>10.83</td><td>13.37</td><td>11.87</td><td>8.73</td><td>10.91</td></tr><tr><td>16K</td><td>38.49</td><td>34.96</td><td>38.65</td><td>28.50</td><td>29.07</td><td>31.40</td><td>30.63</td><td>37.38</td><td>37.94</td><td>37.15</td><td>37.40</td></tr><tr><td>32K</td><td>39.45</td><td>33.05</td><td>39.03</td><td>18.10</td><td>20.71</td><td>21.00</td><td>29.02</td><td>38.05</td><td>36.28</td><td>29.49</td><td>34.35</td></tr><tr><td rowspan="3">⑧</td><td>64K</td><td>34.89</td><td>32.02</td><td>32.77</td><td>11.40</td><td>14.53</td><td>13.70</td><td>21.04</td><td>33.13</td><td>33.17</td><td>15.32</td><td>24.09</td></tr><tr><td>16K</td><td>30.14</td><td>29.08</td><td>31.34</td><td>21.20</td><td>23.89</td><td>22.60</td><td>22.24</td><td>29.65</td><td>28.59</td><td>28.73</td><td>26.49</td></tr><tr><td>32K</td><td>25.83</td><td>24.15</td><td>26.31</td><td>15.30</td><td>17.67</td><td>17.90</td><td>22.73</td><td>22.92</td><td>24.62</td><td>19.14</td><td>23.06</td></tr><tr><td rowspan="3">⑨</td><td>64K</td><td>25.55</td><td>24.17</td><td>27.66</td><td>12.80</td><td>15.95</td><td>15.90</td><td>21.10</td><td>24.37</td><td>27.40</td><td>19.58</td><td>23.33</td></tr><tr><td>16K</td><td>35.42</td><td>35.86</td><td>36.81</td><td>28.70</td><td>22.74</td><td>31.40</td><td>28.45</td><td>32.26</td><td>34.41</td><td>32.46</td><td>32.15</td></tr><tr><td>32K 64K</td><td>29.52 27.05</td><td>27.28 22.55</td><td>28.94 25.38</td><td>19.30 13.00</td><td>15.98 15.22</td><td>23.50 16.10</td><td>20.48 23.79</td><td>28.98 26.18</td><td>25.29 23.74</td><td>27.30 21.51</td><td>26.35 23.52</td></tr></table>

General prediction across models. Each additional model reuses its calibrated boundary table across tasks. Qwen2- 7B-Instruct and Ministral-3-8B-Instruct-2512-BF16 match BF16 word perplexity at the reported precision, at 9.68 and 9.73, respectively, and all six accuracy gaps are below 0.6 percentage points (Tables 10 and 12). Near-reference predictions support model-level allocation across language modeling, reasoning, and understanding tasks.

Long-context answer generation. LV-Eval additionally exercises the compressed prompt cache throughout generation. At 64K, Qwen2 scores 29.70 on CMRC, compared with 27.46 for BF16 and 26.15 for SpargeAttn; Ministral scores 20.71 on LooGLE-MIR, compared with 19.03 and 18.89, respectively (Tables 11 and 13). The same model-specific boundaries therefore preserve useful context for generated answers as well as likelihood-based prediction.

Table 12. General-task results for Ministral-3-8B-Instruct-2512-BF16. Accuracy is reported in percent (higher is better); WikiText-2 uses word perplexity (lower is better). BF16 is the upper reference; a11\_int 4 is the lower reference. Colors compare with BF16: green denotes equal or better results, blue worse results, and gray BF16. Bold marks the best of Ours and Others, including displayed ties.
<table><tr><td rowspan="2">Benchmark</td><td rowspan="2">Metric</td><td>Upper Ref.</td><td>Lower Ref.</td><td>Ours</td><td colspan="5">Others</td></tr><tr><td>BF16</td><td>all_int4</td><td>TriAttention</td><td>MInference</td><td>FlexPrefill</td><td>SageAttn</td><td>MoA</td><td>SpargeAttn</td></tr><tr><td>WikiText-2</td><td>PPL↓</td><td>9.73</td><td>9.75</td><td>9.73</td><td>9.72</td><td>9.84</td><td>9.73</td><td>9.89</td><td>9.75</td></tr><tr><td>HellaSwag</td><td>Acc. ↑</td><td>78.79</td><td>78.80</td><td>79.30</td><td>78.84</td><td>78.85</td><td>78.95</td><td>78.87</td><td>78.85</td></tr><tr><td>MMLU</td><td>Acc. ↑</td><td>77.85</td><td>77.68</td><td>78.35</td><td>77.82</td><td>77.92</td><td>77.94</td><td>77.65</td><td>77.60</td></tr><tr><td>ARC-Challenge</td><td>Acc. ↑</td><td>69.03</td><td>69.11</td><td>69.61</td><td>69.20</td><td>69.03</td><td>69.03</td><td>68.94</td><td>68.94</td></tr><tr><td>LAMBADA</td><td>Acc. ↑</td><td>70.99</td><td>71.07</td><td>71.53</td><td>71.12</td><td>71.05</td><td>71.22</td><td>71.43</td><td>71.12</td></tr><tr><td>PIQA</td><td>Acc. ↑</td><td>80.52</td><td>80.36</td><td>80.86</td><td>80.30</td><td>80.30</td><td>80.58</td><td>80.36</td><td>80.30</td></tr><tr><td>WinoGrande</td><td>Acc. ↑</td><td>77.98</td><td>77.11</td><td>78.40</td><td>77.51</td><td>77.19</td><td>77.35</td><td>77.58</td><td>77.51</td></tr></table>

Table 13. LV-Eval QA results for Ministral-3-8B-Instruct-2512-BF16. Dataset identifiers follow Section 5.4. Scores use official taskspecific metrics on a 0–100 scale; higher is better. Length labels follow the official dataset levels. BF16 is the upper reference; a11\_i nt 4 is the lower reference. Colors compare with BF16: green denotes equal or better results, blue worse results, and gray BF16. Bold marks the best result among Ours, Others, and Ablations; displayed ties share the mark.
<table><tr><td rowspan="2">Dataset Level</td><td rowspan="2"></td><td rowspan="2">Upper Ref. Lower Ref. BF16</td><td rowspan="2">all_int4</td><td rowspan="2">Ours</td><td colspan="5">Others</td><td colspan="3">Ablations</td></tr><tr><td>TriAttention</td><td>MInference FlexPrefill SageAttn</td><td></td><td></td><td>MoA SpargeAttn</td><td></td><td>Layer Same All Same Random</td><td></td></tr><tr><td rowspan="3">①</td><td>16K</td><td>39.51</td><td>39.14</td><td>40.52</td><td>39.30</td><td>38.44</td><td>38.22</td><td>21.53</td><td>40.02</td><td>39.03</td><td>38.01</td><td>31.35</td></tr><tr><td>32K</td><td>41.20</td><td>39.42</td><td>39.77</td><td>42.35</td><td>40.29</td><td>39.48</td><td>23.61</td><td>39.23</td><td>38.87</td><td>35.72</td><td>27.81</td></tr><tr><td>64K</td><td>40.11</td><td>36.26</td><td>39.53</td><td>42.02</td><td>36.72</td><td>39.79</td><td>22.37</td><td>39.37</td><td>39.68</td><td>17.46</td><td>20.14</td></tr><tr><td rowspan="3">②</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>16K</td><td>13.68</td><td>13.90</td><td>14.11</td><td>13.51</td><td>12.78</td><td>12.64</td><td>13.23</td><td>12.91</td><td>13.05</td><td>13.37</td><td>13.04</td></tr><tr><td>32K</td><td>15.26</td><td>13.93</td><td>15.85</td><td>14.84</td><td>15.62</td><td>14.51</td><td>13.84</td><td>12.68</td><td>14.42</td><td>15.03</td><td>12.82</td></tr><tr><td rowspan="3">③</td><td>64K</td><td>17.46</td><td>15.42</td><td>16.91</td><td>14.86</td><td>16.44</td><td>15.68</td><td>14.78</td><td>15.87</td><td>15.57</td><td>14.24</td><td>14.22</td></tr><tr><td>16K</td><td>31.15</td><td>29.48</td><td>32.04</td><td>30.48</td><td>26.82</td><td>30.69</td><td>31.40</td><td>29.95</td><td>31.46</td><td>31.36</td><td>29.52</td></tr><tr><td>32K</td><td>32.58</td><td>33.21</td><td>34.34</td><td>29.93</td><td>25.34</td><td>30.70</td><td>26.33</td><td>31.89</td><td>31.40</td><td>25.52</td><td>29.03</td></tr><tr><td rowspan="3"></td><td>64K</td><td>25.90</td><td>27.63</td><td>27.81</td><td>22.70</td><td>23.26</td><td>26.11</td><td>22.70</td><td>25.35</td><td>23.02</td><td>15.46</td><td>21.49</td></tr><tr><td>16K</td><td>21.65</td><td>20.70</td><td></td><td></td><td></td><td></td><td></td><td></td><td>20.47</td><td></td><td></td></tr><tr><td>32K</td><td>21.21</td><td>20.34</td><td>21.22 21.13</td><td>19.63 19.98</td><td>18.61 20.11</td><td>19.93 20.52</td><td>14.49</td><td>20.14</td><td>20.72</td><td>20.37</td><td>15.62</td></tr><tr><td rowspan="3">④</td><td>64K</td><td>20.77</td><td>20.39</td><td>20.04</td><td>20.27</td><td>19.43</td><td>19.58</td><td>14.63</td><td>20.09</td><td>19.20</td><td>17.38 10.37</td><td>15.51</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>15.00</td><td>19.46</td><td></td><td></td><td>12.50</td></tr><tr><td>16K</td><td>27.62</td><td>28.52</td><td>30.46</td><td>26.91</td><td>27.36</td><td>26.79</td><td>28.87</td><td>28.54</td><td>28.29</td><td>28.56</td><td>27.51</td></tr><tr><td rowspan="3">⑤</td><td>32K</td><td>26.46</td><td>26.98</td><td>29.22</td><td>25.97</td><td>25.57</td><td>25.35</td><td>25.14</td><td>27.81</td><td>26.55</td><td>22.67</td><td>23.85</td></tr><tr><td>64K</td><td>26.93</td><td>26.03</td><td>27.34</td><td>26.23</td><td>25.71</td><td>25.99</td><td>20.80</td><td>25.72</td><td>27.06</td><td>15.79</td><td>19.82</td></tr><tr><td>16K</td><td>21.07</td><td>20.04</td><td>21.64</td><td>20.75</td><td>21.25</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="3">⑥</td><td>32K</td><td>19.49</td><td>19.76</td><td>20.02</td><td>19.23</td><td>19.96</td><td>19.62 19.08</td><td>17.79</td><td>19.48 19.08</td><td>19.51 18.42</td><td>18.63 13.92</td><td>17.51</td></tr><tr><td>64K</td><td>19.03</td><td>19.35</td><td>20.71</td><td>18.18</td><td>17.87</td><td>17.99</td><td>16.32 13.44</td><td>18.89</td><td>19.29</td><td>8.27</td><td>15.93</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>11.44</td></tr><tr><td rowspan="3">⑦</td><td>16K</td><td>57.89</td><td>56.53</td><td>60.26</td><td>57.92</td><td>57.36</td><td>55.76</td><td>45.50</td><td>57.08</td><td>58.90</td><td>56.96</td><td>53.15</td></tr><tr><td>32K</td><td>57.31</td><td>55.04</td><td>57.56</td><td>55.92</td><td>53.84</td><td>56.54</td><td>42.82</td><td>56.53</td><td>56.36</td><td>44.55</td><td>39.54</td></tr><tr><td>64K</td><td>55.03</td><td>53.15</td><td>55.33</td><td>56.59</td><td>53.32</td><td>54.98</td><td>33.74</td><td>54.00</td><td>55.75</td><td>22.05</td><td>24.72</td></tr><tr><td rowspan="3">⑧</td><td>16K</td><td>33.14</td><td>30.05</td><td>34.92</td><td>31.41</td><td>30.18</td><td>33.77</td><td>23.81</td><td>31.39</td><td>31.04</td><td>31.00</td><td>26.39</td></tr><tr><td>32K</td><td>31.90</td><td>30.06</td><td>30.50</td><td>27.60</td><td>31.12</td><td></td><td>30.28 25.88</td><td>30.12</td><td>29.37</td><td>25.18</td><td>25.17</td></tr><tr><td>64K</td><td>31.27</td><td>29.05</td><td>33.82</td><td>31.39</td><td>28.99</td><td>30.11</td><td>25.42</td><td>29.32</td><td>33.81</td><td>19.64</td><td>21.01</td></tr><tr><td rowspan="3">⑨</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>16K</td><td>41.97</td><td>44.18</td><td></td><td>41.93</td><td>40.49</td><td></td><td>41.46 28.79</td><td>44.56</td><td>43.49</td><td>41.87</td><td>41.83</td></tr><tr><td>32K 64K</td><td>41.76 44.43</td><td>43.10 43.89</td><td>42.7 45.80</td><td>39.41 45.88</td><td>36.03 38.34</td><td>40.18 40.69</td><td>26.82 28.68</td><td>40.91 40.62</td><td>39.17 44.43</td><td>39.25 26.53</td><td>34.34 22.25</td></tr></table>

Sensitivity-guided placement. Random preserves the selected boundary pairs and their aggregate modeled cost but changes which heads receive them. On LooGLE-SD at 64K, Qwen2 scores 32.77 with TriAttention and 24.09 with Random, while Ministral scores 55.33 and 24.72. All Same, which applies one 32K local configuration throughout the model, scores 15.32 and 22.05. The randomized control establishes the value of sensitivity-guided placement at fixed modeled cost, while the shared-range control examines restricting all heads to one context range.

The separation from Random widens at longer dataset levels. On LooGLE-SD, the gap increases from 1.25 points at 16K to 8.68 at 64K for Qwen2, and from 7.11 to 30.61 points for Ministral. Because Random preserves the boundary-pair collection and aggregate modeled cost, this trend identifies configuration placement as increasingly consequential in these longer-context evaluations. The same amount of modeled computation supports stronger QA quality when its retained ranges and precisions follow head sensitivity.

FINDINGS At the same modeled computation cost, sensitivity-guided boundary placement preserves longcontext QA quality better than random placement.