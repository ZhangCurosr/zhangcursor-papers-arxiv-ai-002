# SPLIT-KV: EFFICIENT CONTEXT PARALLELISM FOR TRAINING DIFFUSION DRAFTERS

Geng Zhang <sup>\*</sup> <sup>1</sup> Fangzhao Zhang <sup>\*</sup> <sup>2</sup> Cong Xie <sup>3</sup> Tiantian Fan <sup>3</sup> Yang You <sup>1</sup>

## ABSTRACT

Diffusion drafters accelerate speculative decoding by denoising multiple draft tokens in parallel. Their training uses anchor sampling to construct compact draft queries and performs cross-attention with a block-sparse mask over a much longer context. We empirically show that acceptance length remains stable as context scales with a fixed query budget, while increasing the query budget yields diminishing quality gains, jointly establishing a practical regime with queries far shorter than the context. No existing context parallelism (CP) method jointly exploits this query–context asymmetry and meets the nonlinear, sample-dependent load-balancing requirement created by anchor sampling. We present Split-KV, an exact CP method that keeps key/value shards local and limits per-layer attention communication to query-sized tensors, making this communication independent of context length. Its global query view further enables a cost-aware longest-processing-time-first (LPT) strategy that balances context key/value chunks by their realized attention cost. At a CP size of 64, Split-KV retains 85–91% weak-scaling efficiency and achieves up to 3.67× speedup over the fastest available CP baseline. LPT achieves <1% imbalance and plans three orders of magnitude faster than the best load-balancing baseline. Code is available at https://github.com/zxgx/verl.

## 1 INTRODUCTION

Speculative decoding accelerates large language model inference by using a lightweight drafter to propose multiple tokens and verifying them in one target-model forward pass (Leviathan et al., 2023; Chen et al., 2023). Diffusion drafters generate proposals by denoising a block of tokens in parallel, avoiding the sequential forward passes required by autoregressive drafters (Arriola et al., 2025; Christopher et al., 2025; Li et al., 2026a). Diffusion drafter training uses anchor sampling to construct compact draft queries and performs cross-attention with a block-sparse mask over a potentially much longer context.

Diffusion drafter training decouples the attention query length Q, determined by the anchor budget, from the context length S, determined by the training samples. However, how the relationship between Q and S affects drafter quality is still underexplored: scaling S could require proportionally more queries to preserve acceptance length, eliminating the systems benefit of this separation. Our controlled SFT studies instead show that acceptance length remains stable as context scales with a fixed query budget, while increasing the query budget yields diminishing quality gains. Together, these results establish a practical Q ≪ S regime for longcontext diffusion drafter training.

Existing context parallelism (CP) methods do not fully exploit this training regime. Ulysses, Ring Attention, and RingX retain attention communication that grows with S, even when Q remains fixed (Jacobs et al., 2024; Liu et al., 2024; Zhu, 2024; Yin et al., 2025). LV-XAttn keeps key/value tensors stationary and makes its attention communication independent of context length, but streams queryside state through a sequential hop loop and preserves contiguous key/value shards (Chang & Venkataraman, 2025). Thus, context-independent attention communication alone does not address the distinct load-balancing requirement of diffusion-drafter training.

Anchor sampling also changes how attention work is distributed. Unlike causal self-attention, whose per-position cost follows a regular linear profile, sampled anchors produce a nonlinear and sample-dependent context-access profile. Zigzag balances causal self-attention by pairing mirrored positions (Zhu, 2024; Yin et al., 2025), but this position-based rule cannot capture anchor-induced context access. Dynamic Context Parallelism (DCP) handles general attention masks through hypergraph partitioning but incurs substantial planning overhead, while RingX and LV-XAttn provide no load-balancing mechanism (Jiang et al., 2025; Yin et al., 2025; Chang & Venkataraman, 2025). Existing CP methods therefore either balance this workload inaccurately, plan it expensively, or do not balance it at all.

We introduce Split-KV to address both limitations. Split-KV keeps key/value shards local throughout the forward and backward passes and limits per-layer attention communication to query-sized tensors. It exactly reconstructs the original attention while making this communication independent of context length. The resulting global query view decouples context key/value work from query ownership, enabling a cost-aware longest-processing-time-first (LPT) strategy that assigns key/value chunks by their realized attention cost.

Our contributions are summarized as follows:

• We show that acceptance length remains stable as context scales with a fixed query budget, while increasing the query budget yields diminishing quality gains, jointly establishing a practical Q ≪ S regime.

• We design Split-KV, an exact CP method that keeps key/value shards local and limits per-layer attention communication to query-sized tensors, making this communication independent of context length.

• We design a cost-aware LPT strategy that uses Split-KV’s global query view to balance the nonlinear, sample-dependent attention work.

• At a CP size of 64, Split-KV retains 85–91% weakscaling efficiency and achieves up to 3.67× speedup over the fastest available CP baseline; LPT achieves < 1% imbalance and plans three orders of magnitude faster than the best load-balancing baseline, DCP.

## 2 RELATED WORK

## 2.1 Speculative Decoding with Learnable Drafters

Speculative decoding accelerates a target model by generating a multi-token draft at low cost and verifying it in one target-model forward pass, thereby alleviating memorybound decoding (Leviathan et al., 2023; Chen et al., 2023). Extensive work studies autoregressive drafters, including the EAGLE family (Li et al., 2026b) and MTP (DeepSeek-AI et al., 2025). These drafters follow the target-model architecture but use only a few layers to reduce draft-generation cost. Diffusion drafters instead denoise a block of draft tokens in each forward pass. DFlash (Chen et al., 2026) adopts a block-diffusion architecture (Arriola et al., 2025) and uses fewer drafter layers than DiffuSpec (Li et al., 2026a) and SpecDiff (Christopher et al., 2025; Sandler et al., 2026) while retaining high acceptance length. Domino (Huang et al., 2026) and DSpark (Cheng et al., 2026) modify the dependency structure within diffusion blocks to improve acceptance length, but retain the same anchor-based blockdiffusion training pattern targeted by Split-KV.

## 2.2 Context Parallelism

Context parallelism (CP) distributes long sequences across devices to reduce per-device computation and memory at the cost of additional communication. Sequence parallelism distributes sequence-dependent activations across devices (Korthikanti et al., 2023). Ulysses uses all-to-all communication to exchange sequence shards for attention-head shards (Jacobs et al., 2024). Although Ulysses balances attention work across devices, its CP size is limited by the model’s attention-head count. Ring Attention instead partitions the sequence dimension and circulates key/value shards during local attention computation (Li et al., 2023; Liu et al., 2024). Zigzag Ring Attention pairs chunks at mirrored sequence positions to balance the triangular workload induced by causal self-attention (Yin et al., 2025; Zhu, 2024). Dynamic Context Parallelism (DCP) uses hypergraph partitioning to balance dynamic attention workloads while accounting for communication constraints (Jiang et al., 2025). Beyond methods for self-attention, RingX<sup>1</sup> (Yin et al., 2025) and LV-XAttn (Chang & Venkataraman, 2025) distribute bidirectional cross-attention by broadcasting queries in the forward pass. These methods target BERT-like (Liu et al., 2019) or multimodal (Ye et al., 2025) workloads without causal query–key dependencies and do not address the load imbalance introduced by anchor sampling.

## 3 MOTIVATION

## 3.1 Diffusion Drafter Training

Diffusion drafters use cross-attention with a block-sparse mask, a pattern distinct from the self-attention used by LLMs and autoregressive drafters. We use DFlash (Chen et al., 2026) as the concrete architecture throughout the paper. The same attention pattern appears in subsequent diffusion drafters such as Domino (Huang et al., 2026) and DSpark (Cheng et al., 2026), so the Split-KV design applies to them as well.

Specifically, a micro-batch of b prompt–response samples is packed into S context tokens $y _ { 1 } , \ldots , y _ { S }$ and fed to the target model to extract intermediate hidden states. DFlash fuses and projects these hidden states into context features $c \in$ $\mathbb { R } ^ { S \times \bar { H } }$ with hidden size H, then projects c into the context keys and values at each drafter layer. For the attention queries, anchor sampling selects A response positions per sequence. Each anchor opens a block whose remaining B−1 positions are mask tokens to be denoised as draft tokens.

Table 1. Per-rank training cost for a diffusion drafter layer. Multi-head attention is used for brevity. H is hidden size, $P$ is CP size. Communication includes both forward and backward passes. Peak memory counts input and output tensors and communication buffers. Red highlights mark baseline limitations jointly addressed by Split-KV.
<table><tr><td></td><td>communication</td><td>peak memory</td><td>load balancing</td></tr><tr><td>no CP (P=1)</td><td></td><td> $( 2 ( S + Q ) + 2 Q ) H$ </td><td>N/A</td></tr><tr><td>Ulysses</td><td> ${ \frac { P - 1 } { P } } \ \cdot \ { \frac { 4 H ( S + 2 Q ) } { P } }$ </td><td> $\begin{array} { r } { \left( 4 ( S + Q ) + 2 Q \right) \frac { H } { P } } \end{array}$ </td><td>Head sharding</td></tr><tr><td>Ring Attention</td><td> ${ \frac { 2 ( 3 P - 2 ) } { P } } \cdot ( S + Q ) H$ </td><td> $\begin{array} { r } { \left( 4 ( S + Q ) + 2 Q \right) \frac { H } { P } } \end{array}$ </td><td>Zigzag</td></tr><tr><td>RingX</td><td> $\begin{array} { r } { \frac { P - 1 } { P } \cdot 2 H ( 2 S + 3 Q ) } \end{array}$ </td><td> $\displaystyle \left( 4 ( S + Q ) + 2 Q \right) \frac { H } { P }$ </td><td>None</td></tr><tr><td>LV-XAttn</td><td> $\frac { 2 ( 3 P - 2 ) } { P } \cdot Q H$ </td><td> $\textstyle { \bigl ( } 2 ( S + Q ) + 8 Q { \bigr ) } { \frac { H } { P } }$ </td><td>None</td></tr><tr><td>Split-KV</td><td> $\frac { P - 1 } { P } \cdot 4 Q H$ </td><td> $\left( 2 ( S + Q ) + 2 P Q \right) \frac { H } { P }$ </td><td>Cost-aware LPT</td></tr></table>

![](images/6ecde213aa14a9ec29be1f26d38de9a211f6c11a4d784c2ee2a2090ce833a9ec.jpg)  
Figure 1. The DFlash block-sparse mask for one micro-batch with two draft blocks $\left( B { = } 3 \right)$ . One cell is one query–key pair. Each block’s anchor row $a _ { i }$ and its B−1 masked rows ([M]) attend the context window $[ s _ { i } , a _ { i } )$ of their own sample and their own block.

Across the micro-batch, this process produces $N = b A$ blocks and a query length of $Q = N B$ . Anchor i lies at position $a _ { i }$ in the sequence beginning at $s _ { i }$ and attends to its context window $\mathcal { W } _ { i } = [ s _ { i } , a _ { i } )$ and all B tokens in its own block. The attention is therefore causal over the context and bidirectional within each block. As shown in Figure 1, FlexAttention processes all N blocks in one pass with Q queries and a key/value sequence comprising S context tokens and Q mask tokens (Dong et al., 2024). Throughout this paper, S denotes the packed context length, Q denotes the packed query length, and the key/value length is $S + Q$

## 3.2 Limitation 1: Communication Ignores Query-Context Asymmetry

Query–context asymmetry Diffusion drafter training decouples the packed context length S, determined by the training samples, from the packed query length $Q ,$ determined by the anchor budget. The impact of the two factors on the final drafter quality is still underexplored. To determine whether this separation creates a useful $Q \ll S$ regime, we conduct two controlled SFT studies using DFlash drafters for Qwen3.5-9B and Qwen3.5-35B-A3B. Fix Q–Scale S holds the query budget fixed while increasing the maximum per-sequence context length. Fix S–Scale Q instead fixes the maximum context length and increases the query budget through the number of sampled anchors. Figure 2 reports mean drafter acceptance length across GSM8K (Cobbe et al., 2021), MATH-500 (Lightman et al., 2024), HumanEval (Chen et al., 2021), MBPP (Austin et al., 2021), and MT-Bench (Zheng et al., 2023). Appendix A provides the detailed setup and results.

![](images/babbd105f731afbb1a1a0ee7aa6c4caec1ea58b3bbc26dcf7b67375b6bfed64e.jpg)  
Figure 2. Mean acceptance length over five downstream tasks for two controlled SFT studies. (a) Fix Q–Scale S varies the maximum context length at a fixed query budget. (b) Fix S–Scale $Q$ varies the query budget at a fixed maximum context length.

Figure 2(a) shows that, with Q fixed, scaling S changes the five-task mean acceptance length by at most 0.82% for either model. Thus, increasing the context does not require a proportional increase in the query budget to preserve drafter quality. Figure 2(b) then fixes a long context and scales Q. Acceptance length increases monotonically, confirming that query budget matters, but the gain from each doubling diminishes: the final doubling improves the mean by only 0.98% and 1.57% for two models separately. The first study shows that Q need not grow with S, while the second identifies a quality-saturation region at a query budget substantially smaller than the context. Together, they establish our core observation: acceptance length remains stable as context scales with a fixed query budget, while in creasing the query budget yields diminishing quality gains, creating a practical $Q \ll S$ regime.

![](images/d01bca265f982a79b053adcc476406f9e84419b2734f0a46a2d19394e26ba722.jpg)  
Figure 3. Number of queries attending key x in a 16-token sample. Causal self-attention has a linear profile whose mirrored positions have constant total load. Diffusion drafter anchor sampling $( B =$ 2, anchors {6, 10, 11, 12}) produces a piecewise profile.

Limitations of existing context parallelisms Table 1 shows that most existing CP methods do not exploit this query–context asymmetry. Ulysses communicates query/key/value tensors through all-to-all collectives (Jacobs et al., 2024), Ring Attention circulates key/value shards (Liu et al., 2024; Zhu, 2024), and RingX broadcasts key/value tensors and reduces their gradients in the backward pass (Yin et al., 2025). Their total communication therefore contains an S-dependent term and grows with the context even when Q is fixed. LV-XAttn makes attention communication independent of S, but its P2P loop circulates running output and gradient state along with Q and provides no mechanism for the anchor-induced load imbalance discussed next (Chang & Venkataraman, 2025).

## 3.3 Limitation 2: Inadequate Load Balancing

Nonlinear context access With the anchor sampling, context tokens at different positions incur different amounts of attention works, creating a unique load balance requirement different from causal self-attention.

Let the per-position load $c ( x )$ be the number of query rows that attend the key at absolute context position x. For causal self-attention on a sequence of length $L ,$ this load is linear:

$$
c _ { \mathrm { s e l f } } ( x ) = L - x .\tag{1}
$$

Zigzag relies on $c _ { \mathrm { s e l f } } ( x ) + c _ { \mathrm { s e l f } } ( L - 1 - x ) = L + 1$ to achieve perfect load balance (Zhu, 2024).

Diffusion drafters instead sample anchors over the response. Each anchor contributes B queries that read only the context prefix before it, so

$$
\begin{array} { c } { { \displaystyle c _ { \mathrm { a n c h o r } } ( x ) = B \sum _ { i = 0 } ^ { N - 1 } \mathbf { 1 } [ s _ { i } \leq x < a _ { i } ] , } } \\ { { C _ { p } = \displaystyle \sum _ { x \in \mathcal { K } _ { p } } c _ { \mathrm { a n c h o r } } ( x ) , } } \end{array}\tag{2}
$$

where $\displaystyle { { \cal K } _ { p } }$ is the set of context positions assigned to rank $p$ and $C _ { p }$ is its context-dependent attention work. As Figure 3 shows, this load is flat over the prompt and drops only at sampled anchors in the response. Therefore, Anchor sampling changes the linear causalprofile into a sampledependent, nonlinear context-access profile.

Limitations of existing balancing strategies Table 1 summarizes the balancing strategy coupled to each CP method. Ulysses evenly shards attention heads, but its CP size $P$ is bounded by the number of heads and thus cannot scale to arbitrary CP sizes (Jacobs et al., 2024). Ring Attention adopts Zigzag (Zhu, 2024), whose mirrored-position rule is inaccurate for the nonlinear, sample-dependent profile as shown in Figure 3. DCP can represent arbitrary sparse masks through hypergraph partitioning, but its per-step planning is expensive (Section 5.5) (Jiang et al., 2025). The bidirectional cross-attention methods RingX and LV-XAttn do not provide any load-balancing mechanism and preserve contiguous context shards (Yin et al., 2025; Chang & Venkataraman, 2025).

Consequently, existing methods neither fully exploit the query–context asymmetry in their parallel design nor account for the nonlinear attention workload induced by anchor sampling in their load balancing.

## 4 SPLIT-KV: CONTEXT PARALLELISM FOR DIFFUSION DRAFTERS

This section introduces Split-KV, a context-parallel method designed for diffusion-drafter training with $Q \ll S .$ . Section 4.1 presents the parallelism design and analyzes its communication, computation and memory overhead. Section 4.2 analyzes the load skew introduced by anchor sampling and presents a low-cost longest-processing-time-first (LPT) load-balancing strategy built upon Split-KV’s parallelism.

## 4.1 Split-KV Parallelism Design

Motivated by ${ \cal Q } \ll { \cal S }$ , Split-KV starts from the asymmetry between the Q query-source rows $\boldsymbol { X } ~ \in ~ \mathbb { R } ^ { \tilde { Q } \times H }$ and the $S + Q$ key/value-source rows $U \in \mathbb { R } ^ { ( S + Q ) \times H }$

![](images/73ffeb84f6cc9ecf5440ce51a81bbd720458cef697ac973d32e3d18e4c1046f3.jpg)  
Figure 4. Split-KV exploits $Q \ll S$ by all-gathering the global query while keeping the long key/value shards local. The global query view decouples key/value-chunk work from query ownership and enables the cost-aware balancing design.

Split-KV partitions $\begin{array} { r } { X = \bigsqcup _ { p = 0 } ^ { P - 1 } X _ { p } } \end{array}$ and $\begin{array} { r } { U = \bigsqcup _ { p = 0 } ^ { P - 1 } U _ { p } } \end{array}$ by rows across P devices, but replicates only the projected queries $q \ = \ \mathrm { Q P r o j } ( X )$ while keeping $k _ { p } , v _ { p } \ =$ KeyValueProj $( U _ { p } )$ at rank p. For one query and attention head, let $z _ { p } = q k _ { p } ^ { \top } / \sqrt { d } + \mathcal { M } _ { p }$ denote the masked scores over rank p’s local keys, with $\mathcal { M } _ { p }$ being the block-sparse mask and masked entries set to −∞, the local attention result and its log-sum-exp (LSE) are

$$
\begin{array} { l } { \displaystyle \ell _ { p } = \log \sum _ { j } \exp ( z _ { p , j } ) , } \\ { \displaystyle o _ { p } = \sum _ { j } \exp ( z _ { p , j } - \ell _ { p } ) v _ { p , j } } \end{array}\tag{3}
$$

Because the local key sets form a disjoint partition of the complete key/value sequence, the single-device result satisfies

$$
\begin{array} { l } { { \ell = \displaystyle \log \sum _ { p = 0 } ^ { P - 1 } \exp ( \ell _ { p } ) } , } \\ { { \ o = \displaystyle \sum _ { p = 0 } ^ { P - 1 } \exp ( \ell _ { p } - \ell ) o _ { p } = \sum _ { p = 0 } ^ { P - 1 } \bar { o } _ { p } } } \end{array}\tag{4}
$$

where $\bar { o } _ { p } = \exp ( \ell _ { p } - \ell ) o _ { p }$ is rank $p \mathbf { \hat { s } }$ additive contribution to the globally normalized output. Writing OProj as its linear map and adding any bias once after reduction, linearity further gives

$$
Y _ { p } = \left[ \sum _ { p = 0 } ^ { P - 1 } \mathrm { O P r o j } ( \bar { \sigma } _ { p } ) \right] _ { p } = \left[ \mathrm { O P r o j } \left( \sum _ { p = 0 } ^ { P - 1 } \bar { \sigma } _ { p } \right) \right] _ { p }\tag{5}
$$

where $[ \cdot ] _ { p }$ selects the query rows owned by rank $p$ and is implemented together with the summation by a sum reducescatter. Equations 3–5 therefore reconstruct the singledevice attention exactly without moving any key/value rows;

Algorithm 1 Split-KV for one draft layer on rank p. AG,   
AR, and RS denote all-gather, all-reduce, and reduce  
scatter.   
1: Input: $\overline { { \boldsymbol { X _ { p } } , \boldsymbol { U _ { p } } , \boldsymbol { M _ { p } } } }$ ▷ local query rows, key/value-source   
rows, attention mask   
2: Forward pass:   
3: $q _ { p } \gets \mathrm { Q P r o j } ( X _ { p } )$   
4: launch $q \gets \mathbf { A G } ( q _ { p } )$   
5: $k _ { p } , v _ { p } \gets \mathrm { K e y }$ ValueProj(U<sub>p</sub>) ▷ overlaps AG   
6: wait for q   
7: $( o _ { p } , \ell _ { p } ) \bar {  } \mathrm { A t t n } ( q , k _ { p } , v _ { p } ; \mathcal { M } _ { p } )$   
8: $( \ell , \bar { o } _ { p } ) \mathop {  } \mathrm { \mathbf { M e R G E } A R } ( o _ { p } , \ell _ { p } )$   
9: $\bar { y } _ { p } \gets \mathrm { O P r o j } ( \bar { o } _ { p } )$   
10: $\dot { Y } _ { p } \gets \mathrm { R S } _ { \mathrm { s u m } } \dot { ( \bar { y } _ { p } ) }$   
11: return Y<sub>p</sub>   
12: Backward pass:   
13: $g  \mathbf { A G } ( d Y _ { p } )$   
14: do¯ ← OProjBwd(g)   
15: $( \dot { d o _ { p } } , \dot { d \ell _ { p } } ) \longleftarrow \mathbf { M E R G E B W D A R } \big ( \dot { d } \bar { o } _ { p } \big )$   
16: $( d q ^ { ( p ) } , d k _ { p } , d v _ { p } ) \gets \mathrm { A t }$ tnBwd<sub>p</sub>   
17: launch $d q _ { p } \gets \mathrm { R S } _ { \mathrm { s u m } } ( d q ^ { ( p ) } )$   
18: $d U _ { p } \gets \bar { \mathrm { K e y V a l u e P r o j B w d } } ( d k _ { p } , d v _ { p } )$ ▷ overlaps RS   
19: wait for dq<sub>p</sub>   
20: $d X _ { p } \gets \mathrm { Q l }$ ProjBwd(dq<sub>p</sub>)   
21: return $d X _ { p } , d U _ { p }$

the forward pass communicates only q and its projected output, and the backward pass communicates their corresponding gradients as depicted in Figure 4.

RingX (Yin et al., 2025) and LV-XAttn (Chang & Venkataraman, 2025) expose one query shard at each communication step. Under the anchor sampling, a key/value chunk’s cost varies with the active query shard, so balancing aggregate work does not balance every synchronized communication step. Split-KV instead all-gathers the complete q before local attention, making every rank evaluate the full realized query workload against its local key/value chunks at once. This collapses the step-dependent workload into one destination-independent cost per key/value chunk, enabling the cost-aware assignment in Section 4.2, while $Q \ll S$ keeps the additional query memory small.

(a) A two-sample micro-batch with 4 anchors per sample  
![](images/ba0e8439a797ff2e29754ed189a175ed9b54889aed54e81be1ca97ddcac38c01.jpg)

(b) Zigzag: chunk each sample by position  
![](images/b8a713ebe70bda0c5cf774390f3b3e20213db4c2ef1bb66297bca916711c53af.jpg)  
(c) LPT (ours): global chunks, cost-aware assignment

![](images/e3e61a9879f67ce01978ffd0b67ea90f6a2162ab41e266a912253e22d673fb98.jpg)

![](images/f01410e81f764cc2f306997156ae7e33d86ababe435a05b770f4880fec6330ab.jpg)  
12 global two-token chunks; color is the cost-aware LPT assignment.

![](images/55404c8be1a35aa749a634d057d6343924eaaf3c6a123b3b1d57e86a8d3e5ceb.jpg)  
Figure 5. Load balancing of the context keys in Split-KV for the same two packed samples and sampled anchors. A cell shows its position and realized context cost $c _ { \mathrm { a n c h o r } } ( x )$ ; chunk cells show the chunk label above its summed context cost. Zigzag independently cuts each sample into $2 P$ chunks and pairs mirrored positions, yielding $( C _ { 0 } , \dots , C _ { 3 } ) = [ 2 4 , 2 4 , 3 0 , 3 4 ]$ . LPT cuts the global packed context into PJ chunks and assigns them by realized chunk cost, yielding $( C _ { 0 } , \dots , C _ { 3 } ) = [ 2 8 , 2 8 , 2 8 , 2 8 ]$ . Adding the identical $\bar { D } = 2 B ^ { 2 } = 8$ gives $( W _ { 0 } , \dots , W _ { 3 } ) = [ 3 2 , 3 2 , 3 8 , 4 2 ]$ and $( W _ { 0 } , \dots , W _ { 3 } ) = [ 3 6 , 3 6 , 3 6 , 3 6 ]$ , respectively.

Algorithm 1 instantiates this decomposition on rank $p ,$ where $X _ { p } , U _ { p } ,$ and $\mathcal { M } _ { p }$ are the local query rows, the local key/value-source rows comprising the context shard and owned block rows, and the corresponding block-sparse mask. In the forward pass, the query all-gather is overlapped with the key/value projection (lines 4–6), after which lines 7– 10 evaluate Equations 3–5. The backward pass applies the dual schedule (lines 13–21), overlapping the query-gradient reduce-scatter with the key/value-projection backward computation while keeping $d k _ { p }$ and $d v _ { p }$ local.

Table 1 summarizes the communication and peak-memory costs of Split-KV. The forward-pass all-gather and reducescatter each move $\scriptstyle { \frac { P - 1 } { P } } Q H$ elements per rank, and their backward-pass duals move the same amount. Thus Split-KV communicates $\scriptstyle { \frac { P - 1 } { P } } 4 Q H$ elements per layer. For a fixed micro-batch size and anchor budget, Q remains fixed as S grows, so this traffic is independent of the context length; the LSE statistics contribute only a lower-order term. For block i, its B queries attend to $a _ { i } - s _ { i }$ context rows and all B rows in the same block, producing $B ( a _ { i } - s _ { i } ) + B ^ { 2 }$ unmasked query–key pairs. The complete block-sparse attention therefore evaluates $T _ { \mathrm { a t t n } } = \textstyle \bar { B } \sum _ { i = 0 } ^ { N - 1 } ( a _ { i } - s _ { i } ) + N B ^ { 2 }$ pairs, and balanced Split-KV assigns an average attention cost of

$T _ { \mathrm { a t t n } } H / P$ to each rank. At the forward-pass peak, each rank simultaneously holds a local key/value shard of length $( S + Q ) / P$ , the all-gathered queries, and the partial projected outputs, giving a peak of $2 ( S + Q ) H / P + 2 Q H$ . The backward pass has the same peak memory. Having established context-independent communication and destinationindependent chunk costs, we now turn to assigning those chunks across ranks.

## 4.2 Cost-Aware LPT Load Balancing for Split-KV

Cost-aware LPT Using the global queries exposed by Split-KV, we address the nonlinear workload identified in Section 3.3 with a cost-aware heuristic based on longest processing time (LPT) first strategy. Our cost-aware LPT partitions the packed context globally into PJ equal chunks of $m = S / ( P J )$ rows, where each rank must receive J chunks. This one-dimensional assignment relies on Split-KV’s global query view: every rank evaluates the complete query set against whichever context chunks it owns. The aggregate cost of each key/value chunk is therefore independent of its destination rank and can be represented by one scalar, whereas hop-based methods expose a different cost for each active query shard and require a joint assignment over chunks and communication steps. Because every rank has the global anchor metadata $\boldsymbol { \mathcal { A } } = \{ ( s _ { i } , a _ { i } ) \} _ { i = 0 } ^ { N - 1 }$ , it can independently price chunk j, whose left boundary is $\lambda _ { j }$ , as

$$
\hat { c } _ { j } \ = \ B \sum _ { i = 0 } ^ { N - 1 } \big ( \operatorname * { m i n } ( a _ { i } , \lambda _ { j } + m ) - \operatorname * { m a x } ( s _ { i } , \lambda _ { j } ) \big ) ^ { + } ,\tag{6}
$$

where $( \cdot ) ^ { + }$ denotes the positive part. We sort chunks by $\hat { c } _ { j } ^ { \phantom { \dagger } } ,$ , the context work of chunk j, and repeatedly assign the remaining chunk with the longest estimated processing time (the highest cost) to the least-loaded rank with fewer than J chunks. A final locality matching maps the resulting bundles to physical ranks without changing their loads.

The plan itself requires no communication; one all-to-all then redistributes the context-source rows according to the assignment once per training step, before any draft layer executes. This step-level key/value redistribution is separate from the per-layer communication reported in Table 1 and is amortized across all draft layers. The draft blocks in Figure 1 remain local. With balanced block ownership they add the same $D = ( N / P ) B ^ { 2 }$ work to every rank. Thus rank p has total work $W _ { p } = C _ { p } + D$ . We define the load-imbalance ratio as $\rho = \operatorname* { m a x } _ { p } W _ { p } / { \overline { { W } } }$ , where $\begin{array} { r } { \overline { { W } } = P ^ { - 1 } \sum _ { p } W _ { p } } \end{array}$

A concrete example Figure 5 compares Zigzag, the position-based strategy discussed in Section 3.3, with costaware LPT on one realized micro-batch. Panel (a) shows two packed samples with different lengths and anchor locations, which induce the nonlinear per-position costs from Equation 2; here $u \in \{ 0 , 1 \}$ indexes the sample. Panel (b) applies Zigzag independently to each sample, where $Z _ { j } ^ { ( u ) }$ denotes chunk j, and pairs mirrored positions without considering their realized costs. Although every rank receives the same number of chunks, their full loads become [32, 32, 38, 42]. Panel (c) instead forms global chunks $L _ { j }$ , prices them with Equation 6, and assigns them by LPT, producing [36, 36, 36, 36]. Because the block-local contribution $D = 8$ is identical across ranks, equalizing the context contribution $C _ { p }$ reduces ρ from 1.167 to 1.000.

## 4.3 Implementation Details

Our implementation is based on VeOmni, a PyTorch-native backend that composes FSDP2 (Zhao et al., 2023) with sequence parallelism and expert parallelism on multidimensional device meshes (Ma et al., 2025). Our implementation comprises approximately 4.3K lines of Python for constructing the drafter, capturing target-model hidden states, sampling anchors, building block-sparse masks, balancing the key/value chunks, computing the denoising loss and metrics, and integrating optimizer and checkpointing.

Beyond supporting diffusion-drafter training and implementing Split-KV, efficient long-context execution requires an additional system optimization. Packed, padding-free batches reduce computation but expose variable local key/value lengths that trigger repeated FlexAttention compilation. We therefore round each local key/value length up to a multiple of 4096, forming static-shape buckets whose compiled kernels can be reused. Padding is masked out before attention and loss computation.

Table 2. Architectures of the evaluated DFlash drafters, where $n _ { q }$ $n _ { k v } ,$ , and $d _ { h }$ denote the numbers of query heads, key/value heads, and the head dimension, respectively.
<table><tr><td>Target</td><td>Layers</td><td> $d _ { \mathrm { m o d e l } }$ </td><td> $d _ { \mathrm { f f n } }$ </td><td> $n _ { q }$ </td><td> $n _ { k v }$ </td><td> $d _ { h }$ </td></tr><tr><td>Qwen3.5-35B-A3B</td><td>8</td><td>2048</td><td>6144 32</td><td></td><td></td><td>4128</td></tr><tr><td>Qwen3.5-27B</td><td>5</td><td>5120 17408 32</td><td></td><td></td><td></td><td>8128</td></tr></table>

## 5 EVALUATION

## 5.1 Experiment Setup

Testbed We conduct all experiments on up to eight nodes, each equipped with eight accelerators. Within a node, each accelerator has 600 GB/s of aggregate bidirectional intranode bandwidth; across nodes, communication goes through RDMA with 400 Gb/s of inter-node bandwidth.

Workload We evaluate DFlash drafters for Qwen3.5-35B-A3B and Qwen3.5-27B. The target models are from the official Qwen3.5 model collection<sup>2</sup>, and the drafter model architectures are summarized in Table 2 following the original DFlash configurations<sup>3</sup> (Chen et al., 2026).

Evaluation protocol Because diffusion drafters are typically trained for multiple epochs (Chen et al., 2026; Huang et al., 2026; Cheng et al., 2026), we follow the offline training setting to directly feed target-model hidden states to the drafter and measure only drafter training step time. This offline setting excludes target-model feature extraction time. To eliminate variable-length sequences as a confounding factor, we use randomly generated fixed-length samples with a 1:3 prompt-to-response ratio. Following DFlash, we set the diffusion block size to $B = 1 6$ and vary the anchor budget A to control the query length. We report per-sample context and query lengths when describing each benchmark configuration. For micro-batch size b, the packed S and Q defined in Section 3 are obtained by multiplying the corresponding per-sample lengths by b.

To isolate context parallelism, we set the DP size to one and world size equal to CP size $P \in \{ 8 , 1 6 , 3 2 , 6 4 \}$ . We report end-to-end drafter training step time averaged over three steps after five warmup steps.

Baselines To evaluate CP methods, we compare Split-KV with four representative baselines. Ulysses (Jacobs et al.,

![](images/9c2e3e95ad71308d6270b13b75646effe03e17b9d3b547962e987fc250f07fb5.jpg)  
Figure 6. Drafter training step time under weak scaling. Each bar is labeled with absolute time in seconds and weak-scaling efficiency $E _ { \mathrm { w e a k } } ( P ) = T _ { 8 } / T _ { P }$ , separated by a slash. Lower time and higher efficiency are better; 100% efficiency denotes unchanged drafter training step time as the context and P grow proportionally. N/A denotes an unsupported configuration.

2024) shards attention heads through all-to-all communication. Ring Attention (Liu et al., 2024; Zhu, 2024) rotates key/value shards with a zigzag-balanced schedule. RingX (Yin et al., 2025) moves queries in the forward pass, then broadcasts and reduces key/value gradients in the backward pass. LV-XAttn (Chang & Venkataraman, 2025) circulates query-side attention state through P2P rings.

To fairly evaluate load balancing, all baselines use Split-KV and differ only in how context key/value chunks are assigned to ranks. No balance preserves the contiguous chunk assignment. Zigzag applies a mirrored chunk order (Zhu, 2024). DCP invokes KaHyPar for hypergraph partitioning (Jiang et al., 2025; Schlag et al., 2016). LPT is our cost-aware strategy.

## 5.2 Weak Scaling of Drafter Training Step Time

Following a weak-scaling configuration, we increase P from 8 to 64 and scale the per-sample context length proportionally from 128k to 1M while fixing the per-sample query length at 8k and $b = 8 .$ . Consequently, the packed context length S grows from 1M to 8M, while the global packed query length remains $Q = 6 4 \mathrm { k }$ . The local context shard remains $S / P = 1 2 8 \mathrm { k \Omega }$ rows per rank. Figure 6 reports the drafter training step time of every method at each scale.

Split-KV has the lowest reported drafter training step time in six of the eight model-scale configurations, ties Ulysses in a seventh, and provides $1 . 0 1 { \times } { - 2 } . 8 1 { \times }$ speedup where it is strictly faster. Across the 8× scale-out, its drafter training step time remains within 3.67–4.02 s for Qwen3.5-35B-A3B and 3.33–3.94 s for Qwen3.5-27B. At $P = 6 4$ , these results correspond to 91% and 85% weak-scaling efficiency, respectively, while Split-KV is $2 . 8 1 \times$ and 2.19× faster than the fastest available baseline. The near-flat times reflect that its attention communication is independent of context length; the remaining efficiency loss comes from collective latency and synchronization as $P$ increases, even though the query payload stays fixed.

Ring Attention and RingX degrade rapidly because their training communication still carries context-sized tensors, whose size grows with S in this experiment. Ulysses remains competitive at small scales but is unsupported at $P = 6 4$ because of its head-sharding constraint. LV-XAttn’s attention communication is also independent of context length, but its sequential hop loop and contiguous, unbalanced key/value placement keep it slower than Split-KV.

## 5.3 Strong Scaling of Drafter Training Step Time

Following a strong-scaling configuration, we fix the total workload per training step at $b = ~ 4$ , with a per-sample context length of 256k and a per-sample query length of 8k, while increasing P from 8 to 64. The packed lengths are therefore $S = 1 \mathrm { M }$ and $Q = 3 2 \mathrm { k \Omega }$ , and the local context shard $S / P$ decreases from 128k to 16k rows per rank. Figure 7 reports the drafter training step time of every method as the fixed workload is distributed over more ranks.

Split-KV has the lowest reported drafter training step time in seven of the eight model-scale configurations and provides $1 . 0 4 \times - 3 . 6 7 \times$ speedup over the fastest available baseline in those settings. It reduces drafter training step time from 3.46 to 0.78 s for Qwen3.5-35B-A3B and from 2.98 to 0.87 s for Qwen3.5-27B, corresponding to 4.44× and 3.43× speedups under an $8 \times$ increase in parallelism. At $P = 6 4$ , Split-KV therefore retains 55% and 43% strong-scaling efficiency, respectively. The efficiency gap arises because local attention work shrinks with $S / P ,$ whereas the global query payload remains fixed and collective latency and synchronization grow with P.

Ulysses is comparable to Split-KV at the smallest scale for Qwen3.5-27B but is unsupported at $P = 6 4$ . Ring Attention and RingX retain context-dependent traffic, while LV-XAttn avoids that traffic but executes a sequential query-state hop loop without balancing key/value work. Consequently, even though LV-XAttn is the fastest available baseline at $P = 6 4$ Split-KV remains 3.67× and 2.55× faster for Qwen3.5- 35B-A3B and Qwen3.5-27B, respectively.

![](images/444bf7abfa7139857a6e937fd949f6d2d595f3e22f4753df6cd3c724f03e375c.jpg)  
Figure 7. Drafter training step time under strong scaling. Each bar is labeled with absolute time in seconds and strong-scaling efficiency $E _ { \mathrm { s t r o n g } } ( P ) = ( T _ { 8 } / T _ { P } ) / ( P / 8 )$ , separated by a slash. Lower time and higher efficiency are better; the gap from 100% quantifies the scale-out loss. N/A denotes an unsupported configuration.

## 5.4 Query-to-Context Ratio Ablation

We evaluate query-to-context sensitivity on a fixed fournode cluster with P = 32 (32 accelerators). Across all configurations, we hold the per-sample context length at $S _ { \mathrm { s a m p l e } } = 1 2 8 \mathrm { k \Omega }$ and the packed query length at $Q = 6 4 \mathrm { k \Omega }$ We vary the per-sample query length $Q _ { \mathrm { s a m p l e } }$ together with the micro-batch size b using $( Q _ { \mathrm { s a m p l e } } , b ) \bar { = } ( 3 2 \mathrm { k } , 2 )$ (16k, 4), (8k, 8), and (4k, 16), so that $Q \ = \ b Q _ { \mathrm { s a m p l e } } \ \mathrm { r e }$ mains fixed. The packed context length $S = b S _ { \mathrm { s a m p l e } }$ is therefore 256k, 512k, 1M, and 2M, giving $Q { : } S$ ratios of 1:4, 1:8, 1:16, and 1:32. Because each query attends only within its own sample, the nominal query–context attention work $Q S _ { \mathrm { s a m p l e } }$ also remains fixed.

![](images/4853ecddb9b82b3dbbd2a6cec3d8af69244fb39aa4d6f272b2716c6a5da7e568.jpg)  
Figure 8. Drafter training step time as the query-to-context ratio decreases. Lower is better; labels report absolute time in seconds.

Figure 8 demonstrates the central property of Split-KV: decreasing $Q { : } S$ does not increase its attention communication because only query-sized tensors are exchanged. Because the figure reports the full training step rather than attention communication alone, residual growth can still come from context-side computation and the larger micro-batch. For Qwen3.5-35B-A3B, increasing $S / Q$ from 4 to 32 raises Split-KV time by only 1.11×, compared with 2.18× for Ulysses, 5.63× for Ring Attention, and 4.35× for RingX; Split-KV is fastest throughout, and its speedup over Ulysses grows from 1.01× to 1.97×. For Qwen3.5-27B, Split-KV grows by 1.44×, versus 2.05× for Ulysses, 6.32× for Ring Attention, and 4.09× for RingX. Ulysses is 1.10× faster at Q:S = 1:4, but Split-KV overtakes it at 1:8 and reaches a 1.29× speedup at 1:32. LV-XAttn grows by only 1.17× and 1.22× for the two models because its communication is also query-sized, but its sequential hop schedule leaves it $1 . 9 8 \times - 3 . 4 8 \times$ slower than Split-KV across all settings.

## 5.5 Load-Balancing Quality and Planning Cost

We isolate load balancing under a common Split-KV transport, so the strategies differ only in how they assign context chunks. We fix the per-sample query length at 8k and $b = 4 .$

![](images/b6a09ccddd779d0ebbe1fa835d3813d6293170f59342d4b50b2318a1f499b720.jpg)  
Figure 9. Average work imbalance under varying context lengths and CP sizes. Each label reports avg (std), with both values measured in percentage points. The lower panel magnifies the cost-aware strategies.

and evaluate two scaling studies: context-length scaling from 16k to 256k per sample at $P = 8 ,$ and CP-size scaling from $P = 8$ to 64 at a fixed per-sample context length of 256k. We report percentage imbalance as $1 0 0 ( \rho - 1 )$ , where $\rho = \operatorname* { m a x } _ { p } W _ { p } / { \overline { { W } } }$ is the load-imbalance ratio, together with the planning latency required to construct each assignment. Both drafters use the same anchors, block partition, and query-head count and therefore induce identical work distributions.

Figure 9 summarizes each strategy by its average and standard deviation because their relative ordering is invariant across both scaling studies. No balance ignores the nonlinear access profile and averages 46.49% and 55.26% imbalance under context-length and CP-size scaling. Zigzag reduces these values to 7.94% and 8.01%, but its positionbased pairing leaves residual skew. The cost-aware methods are substantially more accurate: DCP averages 0.66% and 0.80%, while LPT achieves the best balance at 0.38% and 0.21%.

Figure 10 shows that comparable assignment quality does not imply comparable planning cost. For Qwen3.5-35B-A3B, Zigzag takes 0.01–0.05 ms and LPT takes 0.38– 1.24 ms across both scaling studies. DCP instead grows from 16.92 to 148.70 ms with context length and from 148.38 to 1164.64 ms with P because it invokes a general hypergraph partitioner; Qwen3.5-27B exhibits the same trend.

Combining Figures 9 and 10, LPT achieves the best load balance with low planning overhead: it is nearly three orders of magnitude faster to plan than DCP while further reducing imbalance. End to end, LPT achieves a 1.15×–1.42× speedup in drafter training step time over No balance, whose details are extended to Appendix B.

![](images/3ca3072a8e0ca887ea58eb777843b41e84f352f39d1b9dd8f1c789da3b44d303.jpg)  
Figure 10. Planning latency across context-length and CP-size scaling for Qwen3.5-35B-A3B. The shared logarithmic axis labels every decade explicitly so that all three planners remain visible. No balance has zero planning cost and is omitted.

## 6 CONCLUSION

Diffusion drafter training exposes a distinct CP regime: acceptance length remains stable as context scales with a fixed query budget, while increasing the query budget yields diminishing quality gains, yielding $Q \ll S ;$ meanwhile, anchor sampling induces nonlinear, sample-dependent attention work. Split-KV addresses both properties by keeping key/value shards local and limiting per-layer attention communication to query-sized tensors, making this communication independent of context length. The global query view enables cost-aware LPT assignment that balances the realized workload. At a CP size of 64, Split-KV retains 85–91% weak-scaling efficiency and achieves up to 3.67× speedup over the fastest available CP baseline; LPT maintains sub-1% imbalance with planning latency nearly three orders of magnitude lower than the best available load-balancing baseline, DCP. These results show that context parallelism for structured attention should jointly design communication and load balancing around the model’s training semantics rather than inherit schedules designed for self-attention.

## REFERENCES

Arriola, M., Gokaslan, A., Chiu, J., Yang, Z., Qi, Z., Han, J., Sahoo, S., and Kuleshov, V. Block diffusion: Interpolating between autoregressive and diffusion language models. In International Conference on Learning Representations, volume 2025, pp. 50726–50753, 2025.

Austin, J., Odena, A., Nye, M., Bosma, M., Michalewski, H., Dohan, D., Jiang, E., Cai, C., Terry, M., Le, Q., and Sutton, C. Program synthesis with large language models, 2021.

Chang, T.-T. and Venkataraman, S. Lv-xattn: Distributed cross-attention for long visual inputs in multimodal large language models. arXiv preprint arXiv:2502.02406, 2025.

Chen, C., Borgeaud, S., Irving, G., Lespiau, J.-B., Sifre, L., and Jumper, J. Accelerating large language model decoding with speculative sampling. arXiv preprint arXiv:2302.01318, 2023.

Chen, J., Liang, Y., and Liu, Z. Dflash: Block diffusion for flash speculative decoding. arXiv preprint arXiv:2602.06036, 2026.

Chen, M., Tworek, J., Jun, H., Yuan, Q., de Oliveira Pinto, H. P., Kaplan, J., Edwards, H., Burda, Y., Joseph, N., Brockman, G., Ray, A., Puri, R., Krueger, G., Petrov, M., Khlaaf, H., Sastry, G., Mishkin, P., Chan, B., Gray, S., Ryder, N., Pavlov, M., Power, A., Kaiser, L., Bavarian, M., Winter, C., Tillet, P., Such, F. P., Cummings, D., Plappert, M., Chantzis, F., Barnes, E., Herbert-Voss, A., Guss, W. H., Nichol, A., Paino, A., Tezak, N., Tang, J., Babuschkin, I., Balaji, S., Jain, S., Saunders, W., Hesse, C., Carr, A. N., Leike, J., Achiam, J., Misra, V., Morikawa, E., Radford, A., Knight, M., Brundage, M., Murati, M., Mayer, K., Welinder, P., McGrew, B., Amodei, D., McCandlish, S., Sutskever, I., and Zaremba, W. Evaluating large language models trained on code, 2021.

Cheng, X., Yu, X., Shao, C., Li, J., Xiong, Y., Qian, Y., Zhu, J., Ma, S., Zhang, X., Ye, J., Chen, Q., Deng, C., Yu, J., Dai, D., Zhang, Z., Wei, Y., Tan, Y., Yang, W., Xu, R., Wu, Y., Xu, Z., Wang, X., Chen, M., Tian, R., Bi, X., Hao, Z., Chen, S., Cao, H., Zhang, W., Xu, A., Zhang, H., Zhao, D., and Liang, W. Dspark: Confidencescheduled speculative decoding with semi-autoregressive generation, 2026.

Christopher, J. K., Bartoldson, B. R., Ben-Nun, T., Cardei, M., Kailkhura, B., and Fioretto, F. Speculative diffusion decoding: Accelerating language generation through diffusion. In Proceedings of the 2025 Conference of the Nations ofthe Americas Chapter ofthe Associationfor

Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pp. 12042–12059, 2025.

Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L., Plappert, M., Tworek, J., Hilton, J., Nakano, R., Hesse, C., and Schulman, J. Training verifiers to solve math word problems, 2021.

DeepSeek-AI, Liu, A., Feng, B., Xue, B., Wang, B., Wu, B., Lu, C., Zhao, C., Deng, C., Zhang, C., Ruan, C., Dai, D., Guo, D., Yang, D., Chen, D., Ji, D., Li, E., Lin, F., Dai, F., Luo, F., Hao, G., Chen, G., Li, G., Zhang, H., Bao, H., Xu, H., Wang, H., Zhang, H., Ding, H., Xin, H., Gao, H., Li, H., Qu, H., Cai, J. L., Liang, J., Guo, J., Ni, J., Li, J., Wang, J., Chen, J., Chen, J., Yuan, J., Qiu, J., Li, J., Song, J., Dong, K., Hu, K., Gao, K., Guan, K., Huang, K., Yu, K., Wang, L., Zhang, L., Xu, L., Xia, L., Zhao, L., Wang, L., Zhang, L., Li, M., Wang, M., Zhang, M., Zhang, M., Tang, M., Li, M., Tian, N., Huang, P., Wang, P., Zhang, P., Wang, Q., Zhu, Q., Chen, Q., Du, Q., Chen, R. J., Jin, R. L., Ge, R., Zhang, R., Pan, R., Wang, R., Xu, R., Zhang, R., Chen, R., Li, S. S., Lu, S., Zhou, S., Chen, S., Wu, S., Ye, S., Ma, S., Wang, S., Zhou, S., Yu, S., Zhou, S., Pan, S., Wang, T., Yun, T., Pei, T., Sun, T., Xiao, W. L., Zeng, W., Zhao, W., An, W., Liu, W., Liang, W., Gao, W., Yu, W., Zhang, W., Li, X. Q., Jin, X., Wang, X., Bi, X., Liu, X., Wang, X., Shen, X., Chen, X., Zhang, X., Chen, X., Nie, X., Sun, X., Wang, X., Cheng, X., Liu, X., Xie, X., Liu, X., Yu, X., Song, X., Shan, X., Zhou, X., Yang, X., Li, X., Su, X., Lin, X., Li, Y. K., Wang, Y. Q., Wei, Y. X., Zhu, Y. X., Zhang, Y., Xu, Y., Huang, Y., Li, Y., Zhao, Y., Sun, Y., Li, Y., Wang, Y., Yu, Y., Zheng, Y., Zhang, Y., Shi, Y., Xiong, Y., He, Y., Tang, Y., Piao, Y., Wang, Y., Tan, Y., Ma, Y., Liu, Y., Guo, Y., Wu, Y., Ou, Y., Zhu, Y., Wang, Y., Gong, Y., Zou, Y., He, Y., Zha, Y., Xiong, Y., Ma, Y., Yan, Y., Luo, Y., You, Y., Liu, Y., Zhou, Y., Wu, Z. F., Ren, Z. Z., Ren, Z., Sha, Z., Fu, Z., Xu, Z., Huang, Z., Zhang, Z., Xie, Z., Zhang, Z., Hao, Z., Gou, Z., Ma, Z., Yan, Z., Shao, Z., Xu, Z., Wu, Z., Zhang, Z., Li, Z., Gu, Z., Zhu, Z., Liu, Z., Li, Z., Xie, Z., Song, Z., Gao, Z., and Pan, Z. Deepseek-v3 technical report, 2025.

Dong, J., Feng, B., Guessous, D., Liang, Y., and He, H. Flex attention: A programming model for generating optimized attention kernels. arXiv preprint arXiv:2412.05496, 2(3):4, 2024.

Huang, J., Zhang, Y., Zhang, Q., Lin, H., Xu, H., and Zhang, L. Domino: Decoupling causal modeling from autoregressive drafting in speculative decoding. arXiv preprint arXiv:2605.29707, 2026.

Jacobs, S. A., Tanaka, M., Zhang, C., Zhang, M., Aminadabi, R. Y., Song, S. L., Rajbhandari, S., and He, Y. System optimizations for enabling training of extreme

long sequence transformer models. In Proceedings of the 43rd ACM Symposium on Principles of Distributed Computing, pp. 121–130, 2024.

Jiang, C., Cai, Z., Tian, Y., Jia, Z., Wang, Y., and Wu, C. Dcp: Addressing input dynamism in long-context training via dynamic context parallelism. In Proceedings of the ACM SIGOPS 31st Symposium on Operating Systems Principles, pp. 221–236. ACM, October 2025. doi: 10. 1145/3731569.3764849.

Korthikanti, V. A., Casper, J., Lym, S., McAfee, L., Andersch, M., Shoeybi, M., and Catanzaro, B. Reducing activation recomputation in large transformer models. Proceedings ofMachine Learning and Systems, 5:341–353, 2023.

Leviathan, Y., Kalman, M., and Matias, Y. Fast inference from transformers via speculative decoding. In International conference on machine learning, pp. 19274–19286. PMLR, 2023.

Li, G., Fu, Z., Fang, M., Zhao, Q., Tang, M., Yuan, C., and Wang, J. Diffuspec: Unlocking diffusion language models for speculative decoding. In Findings of the Association for Computational Linguistics: ACL 2026, pp. 20896–20910, 2026a.

Li, S., Xue, F., Baranwal, C., Li, Y., and You, Y. Sequence parallelism: Long sequence training from system perspective. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 2391–2404, 2023.

Li, Y., Wei, F., Zhang, C., and Zhang, H. Eagle-3: Scaling up inference acceleration of large language models via training-time test. Advances in Neural Information Processing Systems, 38:136737–136756, 2026b.

Lightman, H., Kosaraju, V., Burda, Y., Edwards, H., Baker, B., Lee, T., Leike, J., Schulman, J., Sutskever, I., and Cobbe, K. Let’s verify step by step. In International Conference on Learning Representations, volume 2024, pp. 39578–39601, 2024.

Liu, H., Zaharia, M., and Abbeel, P. Ring attention with blockwise transformers for near-infinite context. In International Conference on Learning Representations, volume 2024, pp. 3992–4008, 2024.

Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., Levy, O., Lewis, M., Zettlemoyer, L., and Stoyanov, V. Roberta: A robustly optimized bert pretraining approach. arXiv preprint arXiv:1907.11692, 2019.

Ma, Q., Zheng, Y., Shi, Z., Zhao, Z., Jia, B., Huang, Z., Lin, Z., Li, Y., Yang, J., Peng, Y., Zhang, Z., and Liu, X. Veomni: Scaling any modality model training with model-centric distributed recipe zoo, 2025.

Sandler, J., Christopher, J., Hartvigsen, T., and Fioretto, F. Specdiff-2: Scaling diffusion drafter alignment for faster speculative decoding. Proceedings of Machine Learning and Systems, 8:1128–1147, 2026.

Schlag, S., Henne, V., Heuer, T., Meyerhenke, H., Sanders, P., and Schulz, C. k-way hypergraph partitioning via nlevel recursive bisection. In Proceedings of the Eighteenth Workshop on Algorithm Engineering and Experiments (ALENEX), pp. 53–67, 2016.

Ye, J., Xu, H., Liu, H., Hu, A., Yan, M., Qian, Q., Zhang, J., Huang, F., and Zhou, J. mplug-owl3: Towards long image-sequence understanding in multi-modal large language models. In International Conference on Learning Representations, volume 2025, pp. 98891–98913, 2025.

Yin, J., Palash, M., Shankar, M., and Wang, F. Ringx: Scalable parallel attention for long-context learning on hpc. In Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis, pp. 1395–1408, 2025.

Zhao, Y., Gu, A., Varma, R., Luo, L., Huang, C.-C., Xu, M., Wright, L., Shojanazeri, H., Ott, M., Shleifer, S., Desmaison, A., Balioglu, C., Damania, P., Nguyen, B., Chauhan, G., Hao, Y., Mathews, A., and Li, S. Pytorch fsdp: Experiences on scaling fully sharded data parallel, 2023.

Zheng, L., Chiang, W.-L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., Lin, Z., Li, Z., Li, D., Xing, E., Zhang, H., Gonzalez, J., and Stoica, I. Judging llm-as-a-judge with mt-bench and chatbot arena. In Oh, A., Naumann, T., Globerson, A., Saenko, K., Hardt, M., and Levine, S. (eds.), Advances in Neural Information Processing Systems, volume 36, pp. 46595–46623. Curran Associates, Inc., 2023. doi: 10.52202/075280-2020.

Zhu, Z. ring-flash-attention: Ring attention implementation with flash attention. https://github.com/ zhuzilin/ring-flash-attention, 2024. Accessed: 2026-08-26.

# Split-KV: Efficient Context Parallelism for Training Diffusion Drafters

Supplementary Appendix

## A DRAFTER QUALITY ACROSS QUERY AND CONTEXT LENGTHS

Training setup We train DFlash drafters for the frozen Qwen3.5-9B and Qwen3.5-35B-A3B target models on their model-specific regenerated SFT corpora, a mixture of 800K samples from NVIDIA Nemotron Post-Training Dataset $\nabla 2 ^ { 1 }$ and CodeAlpaca<sup>2</sup>. Following the original DFlash recipe (Chen et al., 2026), training uses cross-entropy loss, block size $B = 1 6$ , loss-decay factor 7.0, weight decay 0.1, gradient clipping at 1.0, and a cosine learning-rate schedule with a 4% warmup. The 35B-A3B drafter uses the original learning rate $6 \times 1 0 ^ { - 4 }$ . For the 9B drafter, this rate causes recurrent gradient spikes, so we reduce it to $2 \times 1 0 ^ { - 4 }$ while leaving the remaining optimization settings unchanged.

Both studies keep the global batch size at 256 and follow the same six-epoch recipe. Fix Q–Scale S uses 512 anchors per sequence, giving $Q \ = \ 8 \mathrm { k }$ at $B \ = \ 1 6 ,$ , and varies the maximum per-sequence context length over {4k, 16k, 64k}. Fix S–Scale Q fixes the maximum persequence context length at 64k and varies the anchor budget over {128, 256, 512, 1024}, giving Q ∈ {2k, 4k, 8k, 16k}. Each maximum context length is a right-truncation limit rather than a requirement that every sample fill the sequence. All runs terminate at step 18,684 for Qwen3.5-9B and step 18,744 for Qwen3.5-35B-A3B; the 64k, 512-anchor run is shared by the two studies. All variants use Split-KV for drafter training.

Evaluation protocol We evaluate GSM8K, MATH-500, HumanEval, MBPP, and MT-Bench with the official DFlash benchmark and task-specific prompt formatters. Each drafter–task pair starts a fresh vLLM server on two accelerators with tensor parallelism two, 15 speculative tokens, and at most 32,768 batched tokens. We issue 128 sequential requests with thinking enabled, at most 2,048 generated tokens, temperature zero, top-p one, and top-k one. The reported acceptance length equals one plus accepted draft tokens divided by draft steps, including the target model’s bonus token. We measure acceptance length rather than downstream answer accuracy.

Results For Fix Q–Scale S, Table 1 shows no monotonic relationship between maximum training context length and acceptance length. Across 4k–64k, the relative task-average span, computed as the difference between the maximum and minimum divided by the minimum, is only 0.82% for Qwen3.5-35B-A3B and 0.70% for Qwen3.5-9B, and the best context length varies by task. Because each variant is evaluated once, we treat these sub-1% differences as equivalent rather than as statistically significant rankings. Thus, scaling S does not require scaling Q to preserve drafter quality over the tested range.

For Fix S–Scale Q, acceptance length increases monotonically with the query budget on both target models, so reducing Q is not free. However, the marginal gain decreases across successive doublings of Q: the mean gains are 0.16, 0.14, and 0.06 for Qwen3.5-35B-A3B and 0.13, 0.12, and 0.09 for Qwen3.5-9B. Consequently, the final doubling of Q improves mean acceptance length by only 0.98% and 1.57%, respectively. Together, the two studies show that the query budget need not grow with the context and that quality exhibits diminishing returns as that budget increases, establishing a practical $Q \ll S$ regime.

## B IMPACT OF LOAD BALANCING ON DRAFTER TRAINING STEP TIME

To evaluate the end-to-end effect of load balancing, we fix the per-sample context and query lengths at 256k and 8k, respectively, with $b \ = \ 4 ,$ and measure how assignment quality and planning latency jointly affect drafter training step time as P increases. Figure 1 reports speedup over No balance, so values below one indicate that planning costs more than balancing saves.

LPT is the only strategy that improves both drafters at every scale, yielding 1.15×–1.42× speedup over No balance. Zigzag remains beneficial except for a slight regression on Qwen3.5-27B at $P = 6 4$ , but its residual imbalance prevents it from matching LPT. DCP performs competitively at small P yet falls below No balance from $P = 3 2$ for Qwen3.5-35B-A3B and from P = 16 for Qwen3.5-27B as partitioning dominates the shrinking compute time. At P = 64, DCP is 3.11× and 3.24× slower than LPT for the two drafters. Thus, LPT converts near-perfect balance into reductions in drafter training step time without placing an expensive planner on the training critical path. Although planning can be moved into data preparation for offline drafter training, it remains on the critical path when the drafter is trained online during reinforcement learning (Chen et al., 2025; Li et al., 2026).

Table 1. Acceptance length on five downstream tasks. Fix Q–Scale S holds 512 anchors per sequence and varies the maximum context length; Fix S–Scale Q holds the maximum context length at 64k and varies the anchor budget. With block size B = 16, the anchor budgets {128, 256, 512, 1024} correspond to query lengths {2k, 4k, 8k, 16k}. Avg. is the unweighted mean of the five displayed task values.
<table><tr><td>Target model</td><td>S</td><td>Q</td><td>GSM8K</td><td>MATH-500</td><td>HumanEval</td><td>MBPP</td><td>MT-Bench</td><td>Avg.</td></tr><tr><td>Fix Q-Scale S</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Qwen3.5-35B-A3B</td><td>4k</td><td>8k</td><td>5.98</td><td>6.99</td><td>7.00</td><td>5.96</td><td>4.48</td><td>6.08</td></tr><tr><td></td><td>16k</td><td>8k</td><td>6.09</td><td>7.05</td><td>7.03</td><td>5.98</td><td>4.52</td><td>6.13</td></tr><tr><td></td><td>64k</td><td>8k</td><td>6.09</td><td>6.95</td><td>7.05</td><td>5.97</td><td>4.54</td><td>6.12</td></tr><tr><td>Qwen3.5-9B</td><td>4k</td><td>8k</td><td>5.71</td><td>6.59</td><td>6.30</td><td>5.50</td><td>4.40</td><td>5.70</td></tr><tr><td></td><td>16k</td><td>8k</td><td>5.70</td><td>6.51</td><td>6.44</td><td>5.56</td><td>4.47</td><td>5.74</td></tr><tr><td></td><td>64k</td><td>8k</td><td>5.77</td><td>6.56</td><td>6.42</td><td>5.53</td><td>4.37</td><td>5.73</td></tr><tr><td>Fix S-Scale Q</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Qwen3.5-35B-A3B</td><td>64k</td><td>2k</td><td>5.84</td><td>6.58</td><td>6.73</td><td>5.66</td><td>4.31</td><td>5.82</td></tr><tr><td></td><td>64k</td><td>4k</td><td>5.97</td><td>6.78</td><td>6.91</td><td>5.83</td><td>4.40</td><td>5.98</td></tr><tr><td></td><td>64k</td><td>8k</td><td>6.09</td><td>6.95</td><td>7.05</td><td>5.97</td><td>4.54</td><td>6.12</td></tr><tr><td></td><td>64k</td><td>16k</td><td>6.19</td><td>7.04</td><td>7.11</td><td>6.01</td><td>4.55</td><td>6.18</td></tr><tr><td>Qwen3.5-9B</td><td>64k</td><td>2k</td><td>5.48</td><td>6.21</td><td>6.16</td><td>5.23</td><td>4.31</td><td>5.48</td></tr><tr><td></td><td>64k</td><td>4k</td><td>5.64</td><td>6.34</td><td>6.30</td><td>5.44</td><td>4.32</td><td>5.61</td></tr><tr><td></td><td>64k</td><td>8k</td><td>5.77</td><td>6.56</td><td>6.42</td><td>5.53</td><td>4.37</td><td>5.73</td></tr><tr><td></td><td>64k</td><td>16k</td><td>5.79</td><td>6.70</td><td>6.54</td><td>5.58</td><td>4.51</td><td>5.82</td></tr></table>

![](images/dcefacba965444cd7c6c0b72a1adefa2777aeab8a8ea5bb960ff5c997972157b.jpg)  
Figure 1. Speedup in drafter training step time for each loadbalancing strategy over No balance with per-sample context and query lengths of 256k and 8k, respectively. The dashed line denotes performance equal to No balance.

Chen, J., Liang, Y., and Liu, Z. Dflash: Block diffusion for flash speculative decoding. arXiv preprint arXiv:2602.06036, 2026.

Chen, Q., Liu, Z., Sun, P., Li, S., Wang, G., Liu, Z., Wen, Y., Feng, S., and Zhang, T. Respec: Towards optimizing speculative decoding in reinforcement learning systems, 2025.

## REFERENCES

Li, Y., Jiang, H., Xu, Y., Yang, J., Zhang, Y., Cao, Y., Shen, Y., Zhou, F., Men, R., Zhang, J., Yang, A., Yu, B., Zheng, B., Huang, F., Lin, J., Liu, D., and Zhou, J. Breaking entropy bounds: Accelerating rl training via mtp with rejection sampling, 2026.