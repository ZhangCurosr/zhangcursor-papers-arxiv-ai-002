# RACE: ROPE-ALIGNED ATTENTION COMPRESSION BASED ON QUERY-KEY ACTIVATION ENERGIES

Irene Tenison<sup>1∗</sup>, Abdulrahman Alabdulkareem<sup>1</sup>, Daniel Karl I. Weidele<sup>2</sup>, Nima Dehmamy<sup>2</sup>, Dan Gutfreund<sup>2</sup>

<sup>1</sup>MIT CSAIL <sup>2</sup>MIT-IBM Computing Research Lab

## ABSTRACT

Head-dimension compression can reduce the memory required by the key-value (KV) cache, but existing methods often reconstruct full-dimensional representations during inference, adding significant decoding FLOPs overhead. We formalize the design considerations for converting pretrained attention into a natively reduced-dimensional form that reduces both KV-cache storage and attention computation using a calibration set. Specifically, we consider reductions that (1) apply a shared projection to queries and keys so that they remain in a common reduced feature space, (2) derive this projection jointly from query and key activations to better preserve their interaction, and (3) are compatible with the pretrained RoPE transformation. To obtain a reconstruction-free realization of this reduction, we restrict it to coordinate selection and theoretically show that RoPE compatibility then holds if and only if complete rotary pairs are retained. Building on this characterization, we introduce RACE (RoPE-Aligned Attention Compression based on activation Energy), a post-training method that selects rotary pairs using their joint query-key activation energy. We theoretically show that this criterion minimizes a separable, calibration-dependent upper bound on post-RoPE querykey product approximation error. The resulting reductions are folded into the model weights and require neither runtime reconstruction nor specialized kernels. Across three model families and different compression levels, RACE outperforms structured pruning and remains competitive with reconstruction-based methods on language-modeling, zero-shot, and long-context tasks. At 50% KV-cache retention and 256K context, RACE reduces total inference FLOPs by 47.8%, yielding 1.79× higher throughput, 43.2% lower time-to-first-token (TTFT), and 28.1% lower allocated GPU memory in comparison to an uncompressed model. We also show that RACE is compatible with token eviction or compaction as well as quantization based KV-cache compression methods.

## 1 INTRODUCTION

Autoregressive language-model inference is increasingly constrained by the key-value (KV) cache, especially at long contexts. Each attention layer stores a key and value vector for every preceding token, so cache memory grows linearly with context length and with the dimensionality of the cached representations. Reducing the key and value head dimensions offers the possibility of decreasing both KV-cache memory and native attention computation. Realizing these savings is more difficult than merely storing a low-dimensional encoding. Several head-dimension compression methods project keys and values into compact representations but reconstruct full-dimensional keys during inference to effectively apply rotary positional embeddings (RoPE) (Saxena et al., 2024; Chang et al., 2025). Such methods reduce cache storage but introduce decompression and additional arithmetic into the attention path during inference. To obtain the full benefit of head-dimension compression, the attention layer itself must operate directly in the reduced dimensions.

This raises a more fundamental question: when can a pretrained RoPE-based attention layer be converted into a natively lower-dimensional attention layer without reconstructing the discarded coordinates? Since attention logits depend on the post-RoPE product $Q K ^ { \top }$ , we view head-dimension reduction as an approximate matrix multiplication (AMM) problem along the shared feature dimension. However, preserving this interaction in the post-RoPE space is not sufficient for native inference. It must be realizable before RoPE. This imposes an additional structural constraint that the reduction must commute with the pretrained rotary transformation. We characterize this condition exactly for coordinate-selection reductions.

![](images/b826fe811530efb52cf77d824923942cdf0673496497c55e6943eae1dfee61f7.jpg)  
KV Cache Retained (%)

![](images/5873cc27b063ebd7725ac0c99c73ae399a0396de4513190900f5baf1b7597004.jpg)  
KV Cache Retained (%)  
Figure 1: Quality-efficiency trade-off across KV-cache retention levels on Llama-3.2-3B-Instruct. Left: average zero-shot accuracy. Right: relative decoding FLOPs on a logarithmic scale, normalized to full attention (1×). The dashed line represent the uncompressed model. RAP and Thin Keys have decoding FLOPs that overlap those of RACE, while Thin Keys’ accuracy falls below the displayed y-axis. RACE combines accuracy competitive with reconstruction-based methods with the low decoding cost ofnative dimensional reduction.

Table 1: Comparison of head-dimension KV-cache compression methods. RACE combines a shared, jointly selected, RoPE-compatible Q/K reduction with native reduced-dimensional execution. FLOPs are normalized to full attention. See A.4 for derivations of FLOPs.
<table><tr><td colspan="3">Excecution</td><td colspan="4">Q/K Reduction Criterion</td><td colspan="4">Efficiency</td></tr><tr><td>Method</td><td>Reconst- ruction free</td><td>Custom- Kernel free</td><td>Fwd.-only Calibration</td><td>Shared Q/K Map</td><td>Joint Q/K Signal</td><td>Native RoPE</td><td>KV Cache Retained</td><td>Attn. Params Retained</td><td>Prefill FLOPs</td><td>Decode FLOPs</td></tr><tr><td>Attention (Vaswani et al., 2017)</td><td>-</td><td></td><td></td><td></td><td></td><td></td><td>100%</td><td>100%</td><td>x1</td><td>x1</td></tr><tr><td>EigenAttention (Saxena et al., 2024)</td><td>x</td><td>√</td><td>√</td><td>x</td><td>√</td><td>x</td><td>50%</td><td>69.8%</td><td>×0.75</td><td>×11.42</td></tr><tr><td>PALU (Chang et al., 2025)</td><td>x</td><td>x</td><td>√</td><td>x</td><td>x</td><td>x</td><td>50%</td><td>58.3%</td><td>×0.75</td><td>×21.92</td></tr><tr><td>Thin Keys (Yao et al., 2026)</td><td>√</td><td>√</td><td>√</td><td>x</td><td>x</td><td>x</td><td>75%</td><td>50%</td><td>×0.75</td><td>×0.75</td></tr><tr><td>RAP (Xin et al., July 2026)</td><td>√</td><td>√</td><td>x</td><td>√</td><td>x</td><td>√</td><td>50%</td><td>50%</td><td>×0.5</td><td>×0.5</td></tr><tr><td>RACE (Ours)</td><td>√</td><td>√</td><td>√</td><td>√</td><td>1</td><td>√</td><td>50%</td><td>50%</td><td>×0.5</td><td>×0.5</td></tr></table>

These observations define an admissible family of native Q/K reductions: queries and keys can share a reduced feature space, and the reduction must respect the pretrained RoPE structure. The remaining question is which admissible dimensions to retain. We introduce RACE, which ranks complete rotary pairs using joint activation energy for each RoPE pair m, $E _ { m } = \| Q _ { m } \| _ { F } ^ { 2 } + \| K _ { m } \| _ { F } ^ { 2 }$ and retains the highest-energy pairs. This criterion minimizes a separable, calibration-dependent upper bound on post-RoPE query-key product approximation error. The resulting Q/K reduction, together with a calibrated value subspace, is folded into the attention projections, enabling RoPE, caching, and attention to operate entirely in the reduced dimensions without runtime reconstruction or specialized kernels. Summarizing our contributions:

• Constrained-AMM formulation. We formulate native attention compression as preservation of the post-RoPE query-key product subject to pre-RoPE realizability constraint.

• Structural Characterization. We prove that a coordinate-selection reduction can commute with the pretrained RoPE if and only if it retains complete rotary pairs, characterizing the admissible family of coordinate reductions.

• Forward-only pair selection within the family. RACE ranks RoPE pairs using joint query-key activation energy, minimizing a separable calibration-dependent upper bound.

• Empirical validation. Across multiple model families and compression levels, RACE remains competitive with reconstruction-based methods while reducing KV-cache memory, attention cost, parameters, TTFT, and runtime overhead without specialized kernels.

## 2 RELATED WORK

Compression along other KV-cache axes. KV-cache memory can be reduced along the token, precision, or head-dimension axes as shown in Fig. 8. Token-level methods reduce the number of cached positions (S in Eq.7), through eviction, retention, or compaction (Zhang et al., 2023; Xiao et al., 2024; Li et al., 2024; Cai et al., 2024; Fu et al., 2025; Shi et al., 2025). Quantization methods instead reduce the number of bits used for each cached element, (b in Eq.7), (Liu et al., 2024; Hooper et al., 2024; Wang et al., 2025). Our method preserves the retained token positions and reduces the number of elements in every key and value vector. It is therefore complementary to both token reduction and quantization; we evaluate representative combinations in Sec. 4.3.

![](images/f4464f0837d160ec2454720d23dadd972f960193a68c2830f7128c76a422b1c5.jpg)  
Figure 2: Native attention compression along the head-dimension. Reconstruction-based methods (right) may cache low-dimensional representations yet return to the original feature dimension during inference. This is often done using $A _ { K } ^ { l o w }$ and $B _ { K } ^ { l o w }$ that are decomposition of $W _ { K }$ . We instead seek a native reduction (left) that can be applied before RoPE and folded into the pretrained projections, so that attention is executed natively in the reduced dimension. Dashed boxes indicate the corresponding full-dimensional operations.

Head-dimension KV-cache compression. Head-dimension methods reduce the width of the cached representations $( d _ { k }$ and $d _ { v }$ in Eq.7) for keys and values respectively. EigenAttention (Saxena et al., 2024) derives low-rank subspaces from calibrated activations, whereas PALU (Chang et al., 2025) factorizes attention projection weights. The dense low-rank compression subspaces in these and similar methods are not generally invariant under the pretrained RoPE transformation and so these methods recover full-dimensional keys before applying RoPE. This can be compute-intense and so, methods like PALU (Chang et al., 2025) mitigate this by using a custom kernel that does not materialize the full-dimensional reconstructed key tensor. However the inference FLOPs remains dominated by the dense reconstruction basis.

Thin Keys (Yao et al., 2026) avoids key reconstruction by reducing the shared Q/K dimension, but retains all dimensions of Values. A concurrent preprint, RAP (Xin et al., July 2026), performs RoPE-aligned structured parameter pruning using empirical-Fisher scores computed from gradients of $W _ { K }$ and arrives at a similar RoPE-aligned native architecture from a different starting point to RACE. While RAP approaches the problem as structured parameter pruning, RACE begins from the problem of constrained-AMM as shown in Sec.3.1. The methods can therefore produce similar reduced-dimensional architectures, but differ in their compression objective and calibration signal. Additionally, our calibration is forward-only and does not depend on a task loss or the batchsensitivity of Fisher estimates (Tenison et al., 2026). Figure 1 compares our solution to previous techniques in terms of accuracy and FLOPs across a range of compression levels.

Methods including MatryoshkaKV (Lin et al., 2025), STAR-KV (Bhatnagar et al., 2026), ThinK (Xu et al., 2025), and OjaKV (Zhu et al., 2026) additionally study heterogeneous ranks or inputdependent compression. These methods primarily address where or when to allocate compression across layers, heads, or inputs. Our work instead addresses how to construct a fixed, RoPEcompatible reduced representation that operates natively without runtime selection, adaptation, or full-dimensional reconstruction.

## 3 METHOD

Problem Formulation. Consider an attention head with pre-RoPE activations $Q , K \in \mathbb { R } ^ { S \times d _ { k } }$ and $V \in \mathbb { R } ^ { S \times d _ { v } }$ . Given target dimensions $r _ { k } \ < \ d _ { k }$ and $r _ { v } \ < \ d _ { v }$ , we seek reduced-dimensional representations

$$
Q ^ { \prime } = Q P _ { Q } , \qquad K ^ { \prime } = K P _ { K } , \qquad V ^ { \prime } = V P _ { V } ,\tag{1}
$$

with $P _ { Q } , P _ { K } \in \mathbb { R } ^ { d _ { k } \times r _ { k } }$ and $P _ { V } \in \mathbb R ^ { d _ { v } \times r _ { v } }$ , such that the resulting layer stores only $K ^ { \prime }$ and $V ^ { \prime }$ in the KV cache and performs attention directly in dimensions $r _ { k }$ and $r _ { v }$ , without reconstructing the discarded coordinates at inference.

For native reduced-dimensional execution, the Q/K reductions must ultimately be applied to the $p r e \AA -$ RoPE activations so that they can be folded into the query and key projection weights. However, the quantity that determines attention logits is the post-RoPE query-key product. We therefore first characterize the low-dimensional Q/K representation that preserves this interaction, and then determine when it can be realized equivalently before RoPE. Finally we introduce RACE, a criterion that satifies these conditions and effectively convert a pretrained attention to a natively reduceddimensional attention.

## 3.1 NATIVE ATTENTION COMPRESSION AS CONSTRAINED AMM

Let $\widetilde Q \ = \ \mathrm { R o P E } ( Q ) , \widetilde K \ = \ \mathrm { R o P E } ( K )$ . Since attention logits depend on the post-RoPE product $\widetilde Q \widetilde K ^ { \top }$ , we formulate Q/K head-dimension reduction as an approximate matrix multiplication (AMM) problem. We are seeking a column-orthonormal reduction $\mathbf { \bar { \boldsymbol { P } } } \in \mathbb { R } ^ { d _ { k } \times r _ { k } }$ such that,

$$
( \widetilde { Q } P ) ( \widetilde { K } P ) ^ { \top } = \widetilde { Q } P P ^ { \top } \widetilde { K } ^ { \top } \approx \widetilde { Q } \widetilde { K } ^ { \top } .\tag{2}
$$

This is consistent with classical matrix-sketching and AMM results, where a common lowerdimensional map is applied along the shared dimension of the two operands (Woodruff, 2014; Cohen et al., 2016). Detailed explanation is given in A.5.

This view motivates two design principles. First, queries and keys should be represented in a shared low-dimensional subspace, since their interaction is evaluated along the same feature dimension. We therefore seek a common reduction $P _ { Q K }$ for both operands. Second, when this subspace is chosen from calibration data, its selection criterion should reflect both matrices whose product is being approximated. Because the target interaction depends jointly on $\widetilde { Q }$ and $\widetilde { K }$ , we construct $P _ { Q K }$ from statistics of both query and key activations rather than either operand in isolation as in the concurrent work, RAP(Xin et al., July 2026) or ThinKeys(Yao et al., 2026). This is validated empirically in Fig.12.

These observations specify how the Q/K interaction should be reduced, but not yet whether the reduction can be executed natively before RoPE to get the desired systems benefits. Applying $P _ { Q K }$ only after RoPE would still require full-dimensional query and key projections and full-dimensional rotary encoding, reducing the KV cache width without fully reducing the attention computation as in reconstruction-based methods. Native execution therefore requires the shared reduction to be realizable before RoPE. Section 3.1 establishes how the query-key interaction should be reduced, but not whether the shared reduction can be realized before RoPE. Section 3.2 characterizes the additional structural condition under which this is possible.

## 3.2 ROPE-COMPATIBLE QUERY–KEY REDUCTION

Section 3.1 motivates representing queries and keys through a shared low-dimensional map. To apply this reduction before RoPE, it must preserve the corresponding blocks of the pretrained rotary transformation as detailed in A.3.

Let $R _ { t } = \operatorname { d i a g } ( R _ { 1 , t } , \ldots , R _ { M , t } ) \in \mathbb { R } ^ { d _ { k } \times d _ { k } }$ be the pretrained RoPE operator at position $t ,$ where each $R _ { m , t } \in \mathbb { R } ^ { 2 \times 2 }$ acts on one rotary pair. For a retained pair set ${ \mathcal { S } } \subseteq \{ 1 , \ldots , M \}$ , define $R _ { t } ^ { ( S ) } =$ d $\displaystyle \mathrm { l a g } ( R _ { m , t } : m \in \mathcal { S } ) \in \mathbb { R } ^ { r _ { k } \times r _ { k } }$ . Thus, $R _ { t } ^ { ( S ) }$ is the restriction of the original pretrained operator to the retained pairs. We say that a reduction $P _ { Q K } \in \mathbb { R } ^ { d _ { k } \times r _ { k } }$ is RoPE-compatible if

$$
R _ { t } P _ { Q K } = P _ { Q K } R _ { t } ^ { ( S ) } \qquad { \mathrm { ~ f o r ~ a l l ~ } } t .\tag{3}
$$

This condition ensures that applying the reduction before RoPE is equivalent to applying fulldimensional RoPE and then restricting to the retained coordinates. Consequently, query/key projection, rotary encoding, and attention can all be performed directly in dimension $r _ { k }$ . A simple admissible family of $P _ { Q K }$ is obtained by retaining complete RoPE pairs. The following theorem shows that this family exactly characterizes RoPE-compatible coordinate-selection reductions.

Algorithm 1 RACE: Post-Training RoPE-Aligned Attention Compression   
Require: Pretrained model M, calibration set $\mathcal { D } _ { \mathrm { c a l } }$ , target dimensions $r _ { k } , r _ { v }$   
1: for each attention layer do   
2: Collect post-RoPE $Q , K , V$ activations on $\mathcal { D } _ { \mathrm { c a l } }$   
3: For RoPE pairs, compute $E _ { m } = \| Q _ { m } \| _ { F } ^ { 2 } + \| \widetilde { K } _ { m } \| _ { F } ^ { 2 }$   
4: Construct $P _ { Q K }$ that selects the top- ${ \cdot } r _ { k } / 2$ pairs based on $E _ { m }$   
5: Fold in $P _ { Q K } . \mathrm { ~ } W _ { Q } ^ { \mathrm { { f o l d } } } = W _ { Q } P _ { Q K } ; W _ { K } ^ { \mathrm { { f o l d } } } = W _ { K } P _ { Q K }$   
6: Compute the $ { \mathrm { t o p } } \cdot  { r _ { v } }$ right singular vectors of V to get $P _ { V }$   
7: Fold in $P _ { V } . \ W _ { V } ^ { \mathrm { \hat { f } o l d } } = \mathbf { \tilde { { W } } } _ { V } P _ { V } \mathbf { \tilde { ; } } W _ { O } ^ { \mathrm { f o l d } } = P _ { V } ^ { \top } W _ { O }$   
8: end for   
9: Optional: recover performance of $\mathcal { M } ^ { \prime }$ from $\mathcal { M }$   
10: return Native attention-compressed mode $\mathcal { M } ^ { \prime }$

Theorem 1 (RoPE-compatible coordinate selection). Suppose for every rotary pair m, there exists a position t such that sin $\phi _ { m , t } \neq 0 .$ . A coordinate-selection matrix $\dot { P _ { Q K } } \in \dot { \mathbb { R } ^ { d _ { k } \times r _ { k } } }$ satisfies $E q .$ 3 for all t if and only if it selects exactly the coordinates belonging to a union of complete rotary pairs indexed by ${ \mathcal { S } } \subseteq \{ 1 , \ldots , M \}$ , with the selected coordinates ordered consistently with $R _ { t } ^ { ( S ) }$

The proof is provided in A.6. The remaining question is which admissible pairs should be retained.   
Section 3.3 introduces a calibration criterion for ranking the admissible pairs.

## 3.3 CALIBRATION-GUIDED SELECTION WITHIN THE ROPE-COMPATIBLE FAMILY

Section 3.2 characterizes a family of structurally valid native reductions by restricting selection to complete RoPE pairs. What remains is no longer a compatibility question, but a ranking question over this admissible pairs to determine what pairs should be retained. Because the approximation target in Section 3.1 depends on both queries and keys, we rank candidate pairs using a joint cali bration statistic over the two operands.

Let $\begin{array} { r c l } { \mathcal { P } _ { \mathrm { r o p e } } } & { = } & { \{ ( i _ { m } , j _ { m } ) \} _ { m = 1 } ^ { M } } \end{array}$ denote the RoPE pairs of an attention head, and let $\begin{array} { r l } { Q _ { m } } & { { } = } \end{array}$ $Q _ { : , \{ i _ { m } , j _ { m } \} } , K _ { m } = K _ { : , \{ i _ { m } , j _ { m } \} }$ . We assign pair m the joint query-key energy

$$
\begin{array} { r } { E _ { m } = \| Q _ { m } \| _ { F } ^ { 2 } + \| K _ { m } \| _ { F } ^ { 2 } . } \end{array}\tag{4}
$$

For target dimension $r _ { k }$ , we retain the $r _ { k } / 2$ pairs with the largest energies, $\begin{array} { r l } { S } & { { } = } \end{array}$ $\mathrm { T o p K } _ { m \in [ M ] } \left( E _ { m } , \frac { r _ { k } } { 2 } \right)$ and construct $P _ { Q K } = [ e _ { i } ] _ { i \in \mathcal { I } _ { Q K } }$ where $\textstyle { \mathcal { T } } _ { Q K } = \bigcup _ { m \in S } \{ i _ { m } , j _ { m } \}$

The following result provides a calibration-dependent justification for this selection rule.

Proposition 1 (Pair-energy approximation bound). Let $\widetilde Q = \mathrm { R o P E } ( Q )$ and $\widetilde K = \mathrm { R o P E } ( K )$ , and let $\widetilde { Q } _ { m } , \widetilde { K } _ { m }$ denote the coordinates associated with RoPE pair m. For any retained pair set S,

$$
\left\| \widetilde { Q } \widetilde { K } ^ { \top } - \sum _ { m \in \cal { S } } \widetilde { Q } _ { m } \widetilde { K } _ { m } ^ { \top } \right\| _ { F } \leq \sum _ { m \not \in \cal { S } } \left\| \widetilde { Q } _ { m } \right\| _ { F } \left\| \widetilde { K } _ { m } \right\| _ { F } \leq \frac { 1 } { 2 } \sum _ { m \not \in \cal { S } } E _ { m } .\tag{5}
$$

Consequently, among all subsets containing the same number of complete RoPE pairs, selecting the pairs with the largest $E _ { m }$ minimizes the separable upper bound in Eq. 5. Proof of this proposition is given in A.6. Proposition 1 does not imply that the selected pair set minimizes the exact post-RoPE query-key product error. Rather, it shows that the proposed top-energy rule exactly minimizes a worst-case, separable calibration-dependent upper bound in the right-hand side of Eq. 5. We use this result as a tractable forward-only criterion for choosing within the RoPE-compatible family. We evaluate the robustness of this calibration-dependent selection empirically in Section 4.5.

Value compression. Values are not subject to RoPE, so we follow the standard low-rank activation projection used in prior head-dimension compression methods (Saxena et al., 2024). Given calibration activations $\dot { V } \in \mathbb { R } ^ { S _ { \mathrm { c a l } } \times d _ { v } }$ , we retain the top- $\boldsymbol { \cdot } \boldsymbol { r } _ { v }$ right singular vectors $P _ { V } \in \mathbb R ^ { d _ { v } \times r _ { v } }$ and compute $V ^ { \prime } = V P _ { V }$

![](images/df9669cfb2a82dcf4cbf76c4acd4ab7437b6d09198a005f0b35cfcb8bd2451e4.jpg)  
(a) Llama-3.2-3B-Instruct Perplexity

![](images/a53d76a287d92d6aa4125b415b30dfcc249f2fce4ffc6d83ceea170c1f6cabd4.jpg)  
(b) Llama-3.2-3B-Instruct Average zero-shot accuracy

![](images/e7a08bcdfad75c9d6eb22ef93c917cce87de58a42b20ceadee7f030a28659f6e.jpg)  
(c) Llama-3.2-3B-Instruct Long-Bench Score

![](images/f72db0960db3d7d0d076f9969c571fb454afd3de7f3d63c6696a78b219db35d2.jpg)  
(d) Qwen2.5-7B Perplexity

![](images/f35cb1c1d22f7f2a3f5c71bb9c7744ccfcfc76f8336771b4385d61daa1283027.jpg)  
(e) Qwen2.5-7B Average zero-shot accuracy

![](images/0b24c6b1ac34dcf3bfec29c1f8bbbe508097b5eb9ce3eea284aadc796440ab9a.jpg)  
(f) Qwen2.5-7B LongBench Score

Figure 3: Quality across KV-cache retention ratios on Llama-3.2-3B-Instruct (Top) and Qwen2.5-7B (Bottom). WikiText perplexity (lower is better), and mean accuracy across 15 zeroshot tasks and mean LongBench score on 16 english tasks (higher is better).

## 3.4 FOLDING INTO NATIVE REDUCED-DIMENSIONAL ATTENTION

Having obtained the RoPE-compatible query-key selector $P _ { Q K }$ and the value subspace $P _ { V }$ , we fold both transformations directly into the pretrained attention projections. For queries and keys,

$$
Q ^ { \prime } = X W _ { Q } ^ { \mathrm { f o l d } } , K ^ { \prime } = X W _ { K } ^ { \mathrm { f o l d } } \qquad \mathrm { w h e r e } , \qquad W _ { Q } ^ { \mathrm { f o l d } } = W _ { Q } P _ { Q K } , W _ { K } ^ { \mathrm { f o l d } } = W _ { K } P _ { Q K } .\tag{6}
$$

$Q ^ { \prime }$ and $K ^ { \prime }$ are $r _ { k }$ -dimensional queries and keys to which the restricted RoPE operator from Section 3.2 is applied. Hence, neither the discarded query/key coordinates nor a full-dimensional RoPE representation is materialized during inference.

For values, we similarly fold the calibrated subspace into the value and output projections - $W _ { V } ^ { \mathrm { f o l d } } =$ $W _ { V } P _ { V } , W _ { O } ^ { \mathrm { f o l d } } = P _ { V } ^ { \top } W _ { O }$ and $V ^ { \prime } = X W _ { V } ^ { \mathrm { f o l d } } , Y ^ { \prime } = ( A ^ { \prime } V ^ { \prime } ) W _ { O } ^ { \mathrm { f o l d } }$ where $A ^ { \prime }$ is computed entirely in the reduced dimension $r _ { k }$ using the retained RoPE pairs.

This folding converts the original pretrained attention layer into a native architecture with query/key dimension $r _ { k }$ and value dimension $r _ { v } .$ The resulting model stores only reduced-dimensional keys and values and performs RoPE, query-key comparison, value aggregation, and output projection without reconstructing the discarded dimensions. Consequently, the reduction decreases KV-cache storage, attention arithmetic, and attention projection parameters simultaneously.

Recovery. For aggressive compression, we recommend a short one-time post-fold recovery stage using KL distillation from the uncompressed model (Fig. 10). We update the reduced attention parameters $W _ { Q } ^ { \mathrm { f o l d } } , W _ { K } ^ { \mathrm { f o l d } } , W _ { V } ^ { \mathrm { f o l d } } , W _ { O } ^ { \mathrm { f o l d } }$ while keeping the rest of the model frozen. Recovery is performed directly on the natively compressed checkpoint using 1M FineWeb-Edu tokens. Its effect is modest at upto 75% KV retention but substantial at 50%, where it restores much of the performance. Therefore, while moderate compression requires little or no adaptation; for 50% retention, we recommend a short one-time recovery stage for RACE and RAP.

## 4 EXPERIMENTS AND RESULTS

Implementation Details. We implement RACE in PyTorch and Hugging Face Transformers. Calibration uses 32,768 FineWeb-Edu tokens, with statistics and SVDs computed in FP32 during calibration and inference is performed in BF16. We respect each architecture’s physical rotary-pair layout and retain the original frequency indices and RoPE scaling rules; selected pairs are not reindexed as consecutive frequencies. Compressed checkpoints resize the attention projections and cache dimensions while maintaining the $\scriptstyle { \frac { - 1 } { \sqrt { d _ { k } } } }$ scaling, and run using standard layers and PyTorch attention operations. Thus, RACE requires no specialized CUDA/Triton kernel. While all the results in Sec.4 were reported with SDPA attention backend, Tab.2 shows systems results with eager and flash attention backends. Although different kernels optimize for wall-clock time, the number of FLOPs remian the same. Although optional, all the results of RACE and the concurrent work, RAP involve recovery with KL-distillation over approximately 1M FineWeb-Edu tokens as detailed in A.7.

![](images/84378e4a3c12019cb9bfb6395d305583d45387de36bf3c41a0ad0d64ecb7ac95.jpg)  
(a) Prefill throughput speedup

![](images/5c8463bac9149a28be70babe15110f09d55615289cab13fdfb57cb7c24ff7dd7.jpg)  
(b) Decode throughput speedup

Overall Throughput Speedup  
![](images/6b72e97fbee907a9ecfe866a2806d893ff1cdb0dc040c22a72ea3eda311a37db.jpg)  
(c) Overall throughput speedup

![](images/ca5eac950b4ad16546bceef90c683c61967e40f7d224431b40c8e0796c2e4957.jpg)  
(d) Time to first token

![](images/713a5072407108c9786c63e98e1cdf6b1dab5931e3b8e3670fa3f600d09e0f80.jpg)  
(e) GPU allocated memory

![](images/95caa526b77459f2d1524b291c9762706953d80079a1eff8a76578b56737a80c.jpg)  
(f) Total inference FLOPs  
Figure 4: Long-context inference efficiency on Llama-3.2-3B-Instruct. The top row reports prefill, decoding, and overall throughput speedups over the uncompressed model (higher is better). The bottom row reports time to first token, allocated GPU memory, and total inference FLOPs (lower is better). Results are shown at 75% and 50% KV-cache retention for prompt lengths of 64K, 128K, and 256K tokens, with the output length fixed at 128 tokens. These results are average over 10 runs.

## 4.1 QUALITY AT MATCHED KV-CACHE BUDGETS

We evaluate Qwen2.5-7B, Llama-3.2-3B-Instruct, and Granite-4.1-3B against RoPE-compatible implementations of PALU-GLRD (Chang et al., 2025), EigenAttention(Saxena et al., 2024), Thin Keys(Yao et al., 2026) and concurrent work, RAP (Xin et al., July 2026). The full results are shown in Table 3 in the Appendix. Fig.3 shows the same on Qwen2.5-7B and Llama-3.2-3B-Instruct for RACE, the concurrent work and structured pruning baseline: RAP, and the reconstruction-based baseline: PALU. We finetune reduced attention weights of both RAP and RACE for 1M tokens of FineWeb-Edu using KL distillation from the respective uncompressed model for fair comparison. We report WikiText perplexity, mean accuracy across 15 zero-shot tasks (Tab.3), and LongBench score average over 16 English tasks (Tab.4). At 75% KV-cache retention, RACE preserves 98.5%, 96.5%, and 97.5% of the uncompressed zero-shot accuracy on Qwen, Granite, and Llama, respectively. Across compression levels, on Llama and Qwen models (Figure 3), RACE closely matches PALU on zero-shot accuracy and LongBench without runtime reconstruction. RACE consistently outperforms RAP. RACE also achieves the lowest perplexity at 50% and 75% retention.

![](images/45af8982bea6e567de5ced46a2fd1f1ae43a87d639d8fe14137e0cdee5c07768.jpg)

![](images/72ef84d65258e39dec7bbaf180d011526101e06827694a9301a32c00f0a92d5b.jpg)

![](images/7e90335df2dfd2d71f4f91737596467699dd5b62c1f89e74b36ebed709d8743c.jpg)  
Figure 5: Composing RACE with token-level compression and quantization. Combining RACE with $\mathrm { H _ { 2 } O } .$ , SnapKV, and KIVI further reduces total KV-cache retention. We report average zero-shot accuracy (left; higher is better) over 5 tasks as described in 5 and WikiText perplexity (right; lower is better).  
Figure 6: Effect of calibration data on compressed-model quality. Values show change on evaluation with the respective calibration data relative to FineWeb-Edu calibration.

## 4.2 MEASURED SYSTEMS BENEFITS

Reducing KV-cache storage does not automatically reduce inference cost. As discussed in Section 1, this requires native reduced attention. Such native reduced attention methods like RACE eliminates reconstruction at the architectural level, exposing standard reduced-dimensional attention operations that can be supported by general-purpose backends. Figure 4 shows that these measured gains become larger with prompt length as attention accounts for a greater fraction of inference cost. At 256K tokens, 50% retention achieves 1.76×, 1.90×, and 1.79× prefill, decoding, and overall throughput speedups, respectively. It also reduces TTFT by 43.22%, allocated GPU memory by 28.06%, and total inference FLOPs by 47.75% relative to the uncompressed model. The end-to-end memory reduction is smaller than the 50% KV-cache reduction because total allocated memory also includes model parameters, activations, and other components unaffected by the proposed method. Figure.11 shows the memory reduction with compression on total model parameters, attention parameters and KV cache. While parameter memory is unaffected by sequence length, KV Cache memory grows with it.

## 4.3 COMPATIBILITY WITH TOKEN-DIMENSION COMPRESSION AND QUANTIZATION

RACE reduces the KV-cache feature dimension or head dimension and can therefore be combined with methods that compress its token dimension or numerical precision. We apply 75% RACE retention before H2O and SnapKV token compression and KIVI quantization, each with a 50% cache footprint, yielding 37.5% combined retention. Figure 5 shows that applying RACE to an already compressed model causes greater degradation than applying it to the uncompressed model, indicating that the effects of the two compression stages are not independent. Nevertheless, the combinations provide an additional 25% reduction in cache size and feature-dependent attention FLOPs as well as attention parameters. The degradation is smaller with KIVI, particularly in perplexity, suggesting that RACE is especially complementary to quantization. Overall, RACE can augment existing KV-cache methods with further memory, bandwidth, and, for token compression, decoding-FLOP reductions.

## 4.4 EFFECT OF CALIBRATION DATA

Our selection criterion explained in Section 3.3 is derived from a calibration-dependent upper bound on the query-key approximation error, so the retained subspace is expected to reflect the calibration distribution. Figure 6 evaluates this dependence by varying the calibration dataset and reporting performance relative to FineWeb-Edu calibration. Note that for WikiText, the perplexity change is sign-inverted so that positive values consistently denote improvement. Using the same dataset as the downstream task generally gives the strongest results - upto +5.4% on the evaluated datasets with respect to calibration on FineWen-Edu, and calibration on a related task family may also improve performance as observed in evaluation on arc-c with calibration on SciQ. However, the differences are modest (< 6%) for most tasks. Therefore, FineWeb-Edu remains a strong generic calibration source, especially when in-distribution task-specific samples are unavailable or the downstream task family is unknown.

<table><tr><td></td><td>Energy (Ours)</td><td>Magnitude</td><td>Attn. Product</td><td>Grad. Fisher</td></tr><tr><td>Backprop-free</td><td>√</td><td>√</td><td>√</td><td>x</td></tr><tr><td>Task-loss-free</td><td>√</td><td>√</td><td>√</td><td>x</td></tr><tr><td>Explicit O(S2) scoring</td><td>x</td><td>x</td><td>√</td><td>x</td></tr><tr><td>Time (sec)</td><td>2.717</td><td>2.925</td><td>11.616</td><td>8.290</td></tr><tr><td>Throughput (tok./sec)</td><td>12061.6</td><td>11202.4</td><td>2821</td><td>3952.7</td></tr><tr><td>Memory (GB)</td><td>6.142</td><td>6.142</td><td>10.592</td><td>9.218</td></tr></table>

(a) Calibration cost of alternative datadependent RoPE-pair scoring criteria.

![](images/fe3cc8f9f56e405dcd0c93a8403526dc0bc9dff516650a0e597a995840d945e5.jpg)  
(b) Perplexity under alternative pair-selection criteria.

![](images/723abd42d21c692be2351ee3e846e0d821edb7a2853680d4e400309d2cfc55a1.jpg)  
(c) Compression behavior of wide, standard, & narrow heads.  
Figure 7: Analysis of RoPE-pair selection and compression structure. (a) Computational requirements of RoPE-compatible scoring criteria. (b) Perplexity under alternative pair-selection criteria. All methods retain complete RoPE pairs. (c) Effect of attention-head width at matched model size across compression ratios.

## 4.5 ABLATION: SELECTION CRITERIA

The shared Q/K reduction and Rope-compatibility constraint define a family of native coordinate reductions; the remaining options are on how to rank candidate rotary pairs. We first compare RACE’s joint Q/K activation energy with otherwise identical Q-only and K-only selectors. As shown in Fig. 7(b), K-only selection degrades perplexity more strongly, while joint Q/K selection performs best overall. A similar result was observed for Q-only (Table 6). This is consistent with preservation of the query-key interaction: the average normalized error $\lVert Q K ^ { \top } - Q _ { S } K _ { S } ^ { \top } \rVert _ { F } / \lVert Q K ^ { \top } \rVert _ { F } ^ { \cdot }$ is 0.1159, 0.1120, and 0.1051 for Q-only, K-only, and joint Q/K selection, respectively.

We further compare RACE with random, magnitude, attention-product, and gradient-Fisher scoring. Random selection is data-oblivious; energy(RACE), magnitude and attention-product use joint Q/K forward statistics; and gradient-Fisher uses joint Q/K gradients from backward-pass on the calibration data. RACE achieves the lowest perplexity across the evaluated retention ratios among these selectors. Table 7(a) further compares calibration cost of methods that uses Rope-compatible shared joint Q/K statistics for selection. Attention-product scoring explicitly incurs ${ \dot { O } } ( S ^ { 2 } )$ computation and gradient-Fisher requires backpropagation and a task loss. Accordingly, athese methods incur substantially higher calibration time and peak memory. Joint QK energy selection therefore offers a favorable trade-off between perfromance and calibration cost.

## 4.6 TOWARDS MODELS WITH FEWER AND WIDER HEADS

If post-training attention compression is anticipated at deployment time, larger head dimensions may be a useful architectural choice during pretraining. We study three matched ∼ 245 M-parameter GPT-NeoX-style models pretrained from scratch on FineWeb-Edu for 2.5B tokens: wide (4 × 256), standard (8 × 128), and narrow (16 × 64). All three reach comparable uncompressed WikiText validation perplexity. Figure 7(c) shows that wider-head models retain perplexity more effectively under aggressive RACE compression. At each retention ratio, the models have approximately matched aggregate retained attention dimension, KV-cache memory, attention FLOPs, and compressed attention parameter count; the primary difference is how this capacity is partitioned across heads. Wider heads expose more RoPE pairs from which RACE can select, providing greater flexibility in structured dimensional reduction. Thus, when post-training attention compression is a foreseeable deployment requirement, allocating attention width across fewer, wider heads during pretraining may yield a morefavorable quality-efficiency trade-off.

## 5 CONCLUSION

In this work, we study when and how pretrained RoPE attention can be converted into a natively lower-dimensional form that reduces KV-cache storage, attention computation, and parameters without runtime reconstruction. Viewing Q/K compression as constrained approximate matrix multiplication motivates a shared reduction derived jointly from queries and keys. This reduction has to be

RoPE-compatible to enable reconstruction-free compression. We further show that, for coordinateselection reductions family, RoPE compatibility holds if and only if complete rotary pairs are retained. RACE selects within this admissible family using joint Q/K activation energy, a forwardonly criterion that minimizes a separable calibration-dependent upper bound on post-RoPE product error. The resulting reductions are folded directly into the projection weights, enabling attention to execute entirely in the compressed dimensions without specialized kernels. Across multiple model families and compression levels, RACE outperforms evaluated reconstruction-free baselines and remains competitive with reconstruction-based methods while reducing KV-cache memory, attention cost, and latency.

## AI USE STATEMENT

We used generative AI tools to assist with editing portions of the manuscript, retrieving related works, generating and debugging experimental and plotting code, and formatting figures and tables. We also used AI tools to discuss and refine aspects of theoretical presentation and interpretation of experimental results. All AI-assisted technical suggestions, mathematical arguments, code, and interpretations were independently reviewed and verified by the authors against the underlying derivations, implementations, and experimental outputs. No AI generated datasets were used in this work. We take responsibility for the final content of this work, including all text, claims, code, and artifacts produced with AI assistance.

## ETHICS STATEMENT

The proposed method is intended to reduce the memory and computational cost of long-context inference, which may improve the accessibility and efficiency of deploying language models. As with other efficiency techniques, however, reducing inference cost can also lower the barrier to large-scale model deployment; the downstream societal effects therefore depend on how the resulting models are used. Our work does not modify model objectives, safety mechanisms, or training data, and we do not make claims about changes to model safety, fairness, or robustness beyond the evaluated performance metrics

## REPRODUCIBILITY STATEMENT

We provide sufficient implementation and experimental details to reproduce the results in this work. A.7 specifies the RACE calibration and folding procedure, RoPE-pair handling, value compression, optional KL-based recovery, model and benchmark configurations, long-context evaluation settings, hardware/software environment, precision, number of tokens, kernels, and analytical FLOP derivations. Detailed per-task results, selector ablations, recovery settings, and systems measurements are also provided in the A.8.

## REFERENCES

Priyansh Bhatnagar, Ashkan Moradifirouzabadi, Se-Hyun Yang, SeungJae Lee, Jungwook Choi, and Mingu Kang. Star-kv: Low-rank kv cache compression via soft thresholding for adaptive rank control, 2026. URL https://arxiv.org/abs/2606.08382.

Zefan Cai, Yichi Zhang, Bofei Gao, Yuliang Liu, Tianyu Liu, Keming Lu, Wayne Xiong, Yue Dong, Baobao Chang, Junjie Hu, and Wen Xiao. Pyramidkv: Dynamic kv cache compression based on pyramidal information funneling. arXiv preprint arXiv:2406.02069, 2024.

Chi-Chih Chang, Wei-Cheng Lin, Chien-Yu Lin, Chong-Yan Chen, Yu-Fang Hu, Pei-Shuo Wang, Ning-Chi Huang, Luis Ceze, Mohamed S. Abdelfattah, and Kai-Chiang Wu. Palu: Kv-cache compression with low-rank projection. In International Conference on Learning Representations, 2025.

Michael B. Cohen, Jelani Nelson, and David P. Woodruff. Optimal approximate matrix product in terms of stable rank. In 43rd International Colloquium on Automata, Languages, and Programming (ICALP 2016), volume 55 of Leibniz International Proceedings in Informatics, pp. 11:1–11:14, 2016. doi: 10.4230/LIPIcs.ICALP.2016.11.

Yu Fu, Zefan Cai, Abedelkadir Asi, Wayne Xiong, Yue Dong, and Wen Xiao. Not all heads matter: A head-level kv cache compression method with integrated retrieval and reasoning. In International Conference on Learning Representations, 2025.

Coleman Hooper, Sehoon Kim, Hiva Mohammadzadeh, Michael W. Mahoney, Yakun Sophia Shao, Kurt Keutzer, and Amir Gholami. Kvquant: Towards 10 million context length llm inference with kv cache quantization. arXiv preprint arXiv:2401.18079, 2024.

Yuhong Li, Yingbing Huang, Bowen Yang, Bharat Venkitesh, Acyr Locatelli, Hanchen Ye, Tianle Cai, Patrick Lewis, and Deming Chen. Snapkv: Llm knows what you are looking for before generation. arXiv preprint arXiv:2404.14469, 2024.

Bokai Lin, Zihao Zeng, Zipeng Xiao, Siqi Kou, TianQi Hou, Xiaofeng Gao, Hao Zhang, and Zhijie Deng. MatryoshkaKV: Adaptive KV compression via trainable orthogonal projection. In The Thirteenth International Conference on Learning Representations, 2025. URL https: //openreview.net/forum?id=BQwsRy1h3U.

Zirui Liu, Jiayi Yuan, Hongye Jin, Shaochen Zhong, Zhaozhuo Xu, Vladimir Braverman, Beidi Chen, and Xia Hu. Kivi: A tuning-free asymmetric 2bit quantization for kv cache. arXiv preprint arXiv:2402.02750, 2024.

Utkarsh Saxena, Gobinda Saha, Sakshi Choudhary, and Kaushik Roy. Eigen attention: Attention in low-rank space for kv cache compression. In Findings of the Association for Computational Linguistics: EMNLP 2024, pp. 15332–15344, 2024.

Dachuan Shi, Yonggan Fu, Xiangchi Yuan, Zhongzhi Yu, Haoran You, Sixu Li, Xin Dong, Jan Kautz, Pavlo Molchanov, and Yingyan Celine Lin. Lacache: Ladder-shaped kv caching for efficient long-context modeling of large language models. In International Conference on Machine Learning, 2025.

Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024.

Irene Tenison, Soumyajit Chatterjee, Fahim Kawsar, and Mohammad Malekzadeh. AdaBet: Gradient-free layer selection for efficient training of deep neural networks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems, 2017.

Hao Wang, Ligong Han, Kai Xu, and Akash Srivastava. Squat: Subspace-orthogonal kv cache quantization. In Conference on Language Modeling, 2025.

David P. Woodruff. Sketching as a tool for numerical linear algebra. Foundations and Trends in Theoretical Computer Science, 10(1–2):1–157, 2014.

Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. Efficient streaming language models with attention sinks. In International Conference on Learning Representations, 2024.

Jihao Xin, Tian Lyu, David Keyes, Hatem Ltaief, and Marco Canini. Rap: Kv-cache compression via rope-aligned pruning, July 2026. URL https://arxiv.org/abs/2602.02599.

Yuhui Xu, Zhanming Jie, Hanze Dong, Lei Wang, Xudong Lu, Aojun Zhou, Amrita Saha, Caiming Xiong, and Doyen Sahoo. Think: Thinner key cache by query-driven pruning. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview. net/forum?id=n0OtGl6VGb.

Hengshuai Yao, Xing Chen, Ahmed Murtadha, and Guan Wang. Thin keys, full values: Reducing kv cache via low-dimensional attention selection, 2026. URL https://arxiv.org/abs/ 2603.04427.

Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tianlong Chen, Lianmin Zheng, Ruisi Cai, Zhao Song, Yuandong Tian, Christopher Re, Clark Barrett, Zhangyang Wang, and Beidi Chen. H2o: Heavy- ´ hitter oracle for efficient generative inference of large language models. In Advances in Neural Information Processing Systems, 2023.

Yuxuan Zhu, David H. Yang, Mohammad Mohammadi Amiri, Keerthiram Murugesan, Tejaswini Pedapati, and Pin-Yu Chen. OjaKV: Context-aware online low-rank KV cache compression. In Maria Liakata, Viviane P. Moreira, Jiajun Zhang, and David Jurgens (eds.), Findings of the Associationfor Computational Linguistics: ACL 2026, pp. 10161–10178, San Diego, California, United States, July 2026. Association for Computational Linguistics. ISBN 979-8-89176-395- 1. doi: 10.18653/v1/2026.findings-acl.494. URL https://aclanthology.org/2026. findings-acl.494/.

## A APPENDIX

## A.1 AXES OF KV-CACHE COMPRESSION

For a model with L layers, batch size $B ,$ context length $S , H _ { k v }$ key–value heads, key dimension $d _ { k }$ value dimension $d _ { v }$ , and b bytes per element, the KV-cache memory is

$$
M _ { \mathrm { K V } } = B L S H _ { k v } ( d _ { k } + d _ { v } ) b .
$$

This expression exposes three independent factors that can be targeted by compression.

Token-level methods reduce $S$ by evicting, retaining, or consolidating cached positions. Headdimension methods reduce $d _ { k }$ and $d _ { v }$ for each retained position. Quantization reduces $b ,$ the storage precision of each element. As illustrated in Figure $^ { 8 , }$ these approaches operate on different factors of the cache size and can therefore be applied independently or jointly. In particular, combining token and head-dimension compression changes the cache size to

$$
M _ { \mathrm { K V } } ^ { \prime } = B L S ^ { \prime } H _ { k v } ( r _ { k } + r _ { v } ) b ,
$$

where $S ^ { \prime } \leq S , r _ { k } \leq d _ { k }$ , and $r _ { v } \leq d _ { v }$

## A.2 ATTENTION AND KV CACHING.

For hidden states $X ~ \in ~ \mathbb { R } ^ { S \times d _ { \mathrm { m o d e l } } }$ and attention weights $W _ { Q } , W _ { K } \ \in \ \mathbb { R } ^ { d _ { \mathrm { m o d e l } } \times d _ { k } }$ and $W _ { V } \in$ $\mathbb { R } ^ { d _ { \mathrm { m o d e l } } \times d _ { v } }$ , an attention layer computes $Q = X W _ { Q } , \qquad K = X W _ { K } , \qquad V = X W _ { V }$ . For each head, let $Q _ { h } , K _ { h } \in \mathbb { R } ^ { S \times d _ { k } }$ and $V _ { h } \in \mathbb { R } ^ { S \times d _ { v } }$ . Its output is $\begin{array} { r } { O _ { h } = A _ { h } V _ { h } = \mathrm { s o f t m a x } \left( \frac { Q _ { h } K _ { h } ^ { \top } } { \sqrt { d _ { h } } } \right) V _ { h } } \end{array}$ During autoregressive decoding, previously computed keys and values are stored in the KV cache. For L layers, batch size $B ,$ , context length S, $H _ { k v }$ KV heads, $H _ { q }$ Q heads, and b bytes per element, its memory is

$$
M _ { \mathrm { K V } } = B L S H _ { k v } ( d _ { k } + d _ { v } ) b .\tag{7}
$$

Thus, reducing $d _ { k }$ and $d _ { v }$ decreases cache memory without removing cached tokens. It also reduces the query-key and attention-value products, whose context-dependent decoding costs scale as $\mathcal { O } ( B S \dot { H } _ { q } d _ { k } )$ and $O ( B S H _ { q } d _ { v } )$ , respectively. Their prefill costs scale as $O ( B \bar { S ^ { 2 } } H _ { q } d _ { k } )$ and ${ \mathcal O } ( B S ^ { 2 } H _ { q } \bar { d } _ { v } )$ , respectively. Complete FLOP expressions are provided in A.4.

## A.3 ROPE-COMPATIBLE REDUCTION

Rotary positional embeddings. RoPE partitions the rotary query and key dimensions into coordinate pairs and applies a position-dependent two-dimensional rotation to each pair (Su et al., 2024). For the pair associated with frequency $\theta _ { i }$ at position t,

$$
R _ { i } ( t ) = \left[ \begin{array} { c c } { \cos ( t \theta _ { i } ) } & { - \sin ( t \theta _ { i } ) } \\ { \sin ( t \theta _ { i } ) } & { \cos ( t \theta _ { i } ) } \end{array} \right] .
$$

Although implementations may store paired coordinates consecutively or across two halves of the head dimension, the two coordinates remain coupled by the rotation. A coordinate-selection method

![](images/825389a069c0367b7625f33b9feac8ab4b3f42adf93d074f03e0ad86fceb3fe1.jpg)

![](images/cd6fd084b7e1c4e5d59884be3756e4ba81a81a83b3424792637965538561a664.jpg)

![](images/89609bbed8cd923b03fc11d208323ab18b3c101af45343d589ea0dc85c267f6f.jpg)

![](images/6048fe1d42843bee59a247e3ca1dc4a3cc46abcbd3e1c9cd95d91ca8abac8903.jpg)  
Figure 8: Token- and head- dimension KV-cache compression. Token-level methods reduce the number of retained cache entries $\mathrm { N , }$ whereas head-dimension methods reduce the dimensionality of each cached representation. These approaches act on orthogonal axes of the KV cache and can be combined.

can therefore preserve the pretrained RoPE operation without reconstruction only if it retains or removes complete rotary pairs. Architectures with partial RoPE additionally contain non-rotary coordinates, which need not satisfy this pairwise constraint.

Logical pairs and physical coordinate layouts. Let $d _ { \mathrm { r o t } } \leq d _ { k }$ denote the number of query and key coordinates to which RoPE is applied, and let $M = d _ { \mathrm { r o t } } / 2$ . We represent the logical rotary pairs as

$$
\mathcal { P } _ { \mathrm { r o p e } } = \{ ( i _ { m } , j _ { m } ) \} _ { m = 1 } ^ { M } ,
$$

where $( i _ { m } , j _ { m } )$ identifies the two physical coordinates rotated using the m-th rotary frequency. The mapping from logical pairs to physical coordinates depends on the model implementation. With one-based indexing, common layouts include

$$
( i _ { m } , j _ { m } ) = \left\{ { \begin{array} { l l } { ( 2 m - 1 , 2 m ) , } & { { \mathrm { a d j a c e n t } } \mathrm { { \cdot p a i r } } { \mathrm { l a y o u t } } , } \\ { ( m , m + M ) , } & { { \mathrm { s p l i t } } \mathrm { { \cdot h a l f } } { \mathrm { l a y o u t } } . } \end{array} } \right.
$$

The former is used by GPT-J-style rotary implementations, whereas the latter is used by LLaMAstyle implementations. Our selection operates on these logical pairs rather than on neighboring memory locations, so the same procedure applies to either layout after constructing the appropriate pair map.

Restriction of the original RoPE operator. Let $S \subseteq [ M ]$ be the retained set of rotary pairs and let $P s \in \{ 0 , 1 \} ^ { d _ { k } \times 2 | S | }$ contain the standard basis vectors corresponding to both coordinates of every pair in S. The reduced RoPE operator is formed by retaining the original rotation block associated with each selected pair:

$$
R _ { t } ^ { ( S ) } = \bigoplus _ { m \in S } R _ { t , m } .
$$

Provided that the columns of $P _ { S }$ and the blocks of $R _ { t } ^ { ( S ) }$ use the same pair ordering,

$$
R _ { t } P _ { S } = P _ { S } R _ { t } ^ { ( S ) } .
$$

Consequently, for row-wise query activations,

$$
( Q R _ { t } ) P _ { S } = ( Q P _ { S } ) R _ { t } ^ { ( S ) } ,
$$

and analogously for keys. Thus, selecting coordinates before RoPE and applying the restricted operator produces exactly the selected coordinates of the original RoPE output. This is an algebraic restriction of the pretrained operator, not an approximation introduced by the reduced-dimensional implementation.

Preserving pretrained rotary frequencies. Reducing the query–key dimension must not cause the rotary frequencies to be recomputed from the new dimension $r _ { k }$ . Standard RoPE implementations derive frequencies from their original rotary dimension; reinitializing RoPE with $r _ { k }$ would generally assign different angles to the retained coordinates and would therefore alter the pretrained positional encoding. Instead, for every retained pair m $\in S .$ , we copy its original frequency $\theta _ { m }$

$$
\pmb { \theta } ^ { ( S ) } = [ \theta _ { m } ] _ { m \in \mathcal { S } } .
$$

The selected pairs may be stored in score order or original-coordinate order, provided that the queries, keys, and frequencies use the same ordering. In our implementation, we retain the original pair order after selection to make the folded checkpoints deterministic.

For models using scaled RoPE, the same principle applies: the model’s original position-dependent frequency transformation and any associated attention scaling are preserved, and only the entries corresponding to selected pairs are gathered. This avoids silently replacing the pretrained positional encoding with a new reduced-dimensional frequency schedule.

Partial RoPE. Some architectures apply RoPE to only $d _ { \mathrm { r o t } } < d _ { k }$ coordinates. Let $\mathcal { T } _ { \mathrm { N o P E } }$ denote the complementary non-rotary coordinates. The complete-pair constraint is imposed only on $\mathcal { P } _ { \mathrm { r o p e } } ;$ non-rotary coordinates may be selected individually. If $\mathcal { I } \subseteq \mathbb { Z } _ { \mathrm { N o P E } }$ is the retained non-rotary set, the complete selector is

$$
P _ { Q K } = [ P _ { S } P _ { \mathcal { I } } ] , \qquad r _ { k } = 2 | \boldsymbol { S } | + | \mathcal { I } | ,
$$

and the corresponding reduced positional operator is

$$
R _ { t } ^ { ( S , \mathcal { I } ) } = R _ { t } ^ { ( S ) } \oplus I _ { | \mathcal { I } | } .
$$

The identity block leaves the retained non-rotary coordinates unchanged. Hence, the same commutation relation holds for architectures with full or partial RoPE.

Grouped-query attention. Under grouped-query attention, several query heads attend to a shared key head. For each KV head $^ { g , }$ let $\mathcal { G } _ { g }$ denote its associated query heads. We compute the Q/K pair statistics jointly over the key activations of head g and the query activations from all heads in $\mathcal { G } _ { g }$ A single pair set $ { \boldsymbol { S } } _ { g }$ is then applied to the shared key head and to every query head in its group. This ensures that each query and the key cache against which it is evaluated use identical retained coordinates and rotary frequencies. Pair sets may differ across KV heads, but they cannot differ among query heads that interact with the same cached key representation.

Implementation consequence. After selection, the columns indexed by $P _ { Q K }$ are folded into $W _ { Q }$ and $W _ { K }$ , and the RoPE module is restricted to the corresponding original frequencies. The resulting layer produces $r _ { k }$ -dimensional queries and keys directly. Neither the discarded coordinates nor a full-dimensional RoPE output is materialized during prefill or decoding.

## A.4 INFERENCE FLOPS ANALYSIS

We analyze the attention-related floating-point operations (FLOPs) incurred during prefill and autoregressive decoding. We count a multiply-add as two FLOPs and omit element-wise operations such as softmax, normalization, and RoPE, which are lower-order relative to the matrix multiplications considered here.

Let B denote the batch size, $S$ the context length, d the model hidden dimension, $H _ { q }$ the number of query heads, and $H _ { \mathrm { k v } }$ the number of key-value heads. The original query/key and value head dimensions are $d _ { k }$ and $d _ { v }$ , respectively. After compression, their corresponding dimensions are $r _ { k }$ and $r _ { v }$

For prefill, projections are applied to all S input tokens and attention is computed over an $S \times S$ attention matrix. For decoding, we report the FLOPs for generating one new token given a cache of S previous tokens.

Full Attention For standard attention, the query, key, and value projections during prefill cost

$$
F _ { \mathrm { p r o j } } ^ { \mathrm { p r e } } = 2 B S d \left( H _ { q } d _ { k } + H _ { \mathrm { k v } } d _ { k } + H _ { \mathrm { k v } } d _ { v } \right) .\tag{8}
$$

The query-key product and attention-value product cost

$$
\begin{array} { r } { F _ { Q K } ^ { \mathrm { p r e } } = 2 B S ^ { 2 } H _ { q } d _ { k } , } \end{array}\tag{9}
$$

$$
\begin{array} { r } { F _ { A V } ^ { \mathrm { p r e } } = 2 B S ^ { 2 } H _ { q } d _ { v } , } \end{array}\tag{10}
$$

and the output projection costs

$$
\begin{array} { r } { F _ { O } ^ { \mathrm { p r e } } = 2 B S H _ { q } d _ { v } d . } \end{array}\tag{11}
$$

Thus,

$$
\begin{array} { r } { F _ { \mathrm { F u l l } } ^ { \mathrm { p r e } } = 2 B S d \left( H _ { q } d _ { k } + H _ { \mathrm { k v } } d _ { k } + H _ { \mathrm { k v } } d _ { v } \right) + 2 B S ^ { 2 } H _ { q } ( d _ { k } + d _ { v } ) + 2 B S H _ { q } d _ { v } d . } \end{array}\tag{12}
$$

During decoding, projections are required only for the newly generated token:

$$
F _ { \mathrm { p r o j } } ^ { \mathrm { d e c } } = 2 B d \left( H _ { q } d _ { k } + H _ { \mathrm { k v } } d _ { k } + H _ { \mathrm { k v } } d _ { v } \right) .\tag{13}
$$

The query attends to all S cached keys and values, giving

$$
\begin{array} { r } { F _ { Q K } ^ { \mathrm { d e c } } = 2 B S H _ { q } d _ { k } , } \end{array}\tag{14}
$$

$$
F _ { A V } ^ { \mathrm { d e c } } = 2 B S H _ { q } d _ { v } ,\tag{15}
$$

while the output projection costs

$$
\begin{array} { r } { F _ { O } ^ { \mathrm { d e c } } = 2 B H _ { q } d _ { v } d . } \end{array}\tag{16}
$$

Therefore,

$$
\begin{array} { r } { F _ { \mathrm { F u l l } } ^ { \mathrm { d e c } } = 2 B d \left( H _ { q } d _ { k } + H _ { \mathrm { k v } } d _ { k } + H _ { \mathrm { k v } } d _ { v } \right) + 2 B S H _ { q } ( d _ { k } + d _ { v } ) + 2 B H _ { q } d _ { v } d . } \end{array}\tag{17}
$$

For long contexts, the cache-length-dependent term dominates:

$$
F _ { \mathrm { F u l l , L C } } ^ { \mathrm { d e c } } \simeq 2 B S H _ { q } ( d _ { k } + d _ { v } ) .\tag{18}
$$

EigenAttention with RoPE EigenAttention projects attention representations K and V into learned low-dimensional subspaces. In its RoPE-compatible formulation, compressed keys must be decompressed into the original dimension, which is RoPE-compatible, before query-key attention.

During prefill, the projection costs

$$
F _ { p r o j } ^ { \mathrm { p r e } } = 2 B S d ( H _ { q } d _ { k } + H _ { k v } r _ { k } + H _ { k v } r _ { V } ) .\tag{19}
$$

Transforming the compressed keys to the original key space costs

$$
F _ { K , \mathrm { r e c } } ^ { \mathrm { p r e } } = 2 B S H _ { \mathrm { k v } } r _ { k } d _ { k } .\tag{20}
$$

Since queries and reconstructed keys interact in the original key dimension, the query-key product is the same as full attention. The value representation remains compressed to dimension $r _ { v } ,$ giving

$$
F _ { A V } ^ { \mathrm { p r e } } = 2 B S ^ { 2 } H _ { q } r _ { v } \qquad F _ { O } ^ { \mathrm { p r e } } = 2 B S H _ { q } r _ { v } d .\tag{21}
$$

Thus, the prefill cost is

$$
F _ { \mathrm { E i g e n } } ^ { \mathrm { p r e } } = 2 B S \left[ d ( H _ { q } d _ { k } + H _ { k v } r _ { k } + H _ { k v } r _ { V } ) + H _ { \mathrm { k v } } r _ { k } d _ { k } + S H _ { q } ( d _ { k } + r _ { v } ) + H _ { q } r _ { v } d _ { k } + H _ { \mathrm { k v } } r _ { v } \right] .\tag{22}
$$

During decoding, all S cached compressed keys must be reconstructed before RoPE, giving

$$
F _ { p r o j } ^ { \mathrm { d e c } } = 2 B d ( H _ { q } d _ { k } + H _ { k v } r _ { k } + H _ { k v } r _ { V } ) . \qquad F _ { K , \mathrm { r e c } } ^ { \mathrm { d e c } } = 2 B S H _ { \mathrm { k v } } r _ { k } d _ { k } .\tag{23}
$$

The subsequent query-key, attention-value operations, and output cost

$$
\begin{array} { r } { F _ { Q K } ^ { \mathrm { d e c } } = 2 B S H _ { q } d _ { k } \qquad F _ { A V } ^ { \mathrm { d e c } } = 2 B S H _ { q } r _ { v } , \qquad F _ { O } ^ { \mathrm { d e c } } = 2 B H _ { q } r _ { v } d . } \end{array}\tag{24}
$$

Therefore, the cache-length-dependent cost of generating one token is

$$
F _ { \mathrm { E i g e n , L C } } ^ { \mathrm { d e c } } = 2 B S \left[ H _ { \mathrm { k v } } r _ { k } d _ { k } + H _ { q } ( d _ { k } + r _ { v } ) \right] .\tag{25}
$$

This additional $O ( B S H _ { k v } r _ { k } d _ { k } )$ term arises from reconstructing cached keys into the RoPEcompatible space.

PALU with G-LRD and RoPE PALU (Chang et al., 2025) applies group-head low-rank decomposition (G-LRD) to the key and value projection matrices. Let s denote the number of KV heads in each group and $\begin{array} { r } { G = \frac { H _ { \mathrm { k v } } } { c } } \end{array}$ the number of groups. For each key group g, PALU factorizes the concatenated projection of s heads as

$$
W _ { K } ^ { ( g ) } \approx A _ { K } ^ { ( g ) } B _ { K } ^ { ( g ) } ,
$$

where

$$
A _ { K } ^ { ( g ) } \in \mathbb R ^ { d \times r _ { k } ^ { ( g ) } } , \qquad B _ { K } ^ { ( g ) } \in \mathbb R ^ { r _ { k } ^ { ( g ) } \times s d _ { k } } .
$$

The intermediate representation produced by $A _ { K } ^ { ( g ) }$ is stored in the KV cache. For RoPE-based attention, the cached key latent is reconstructed to the original $s d _ { k }$ -dimensional group representation before applying RoPE and computing attention.

Assuming uniform group ranks $r _ { k } ^ { ( g ) } = r _ { k } ^ { G }$ and $r _ { v } ^ { ( g ) } = r _ { v } ^ { G }$ , the Q/K/V projection cost during prefill is

$$
F _ { \mathrm { p r o j } } ^ { \mathrm { p r e } } = 2 B S d \left( H _ { q } d _ { k } + G r _ { k } ^ { G } + G r _ { v } ^ { G } \right) .\tag{26}
$$

For one decoding step,

$$
F _ { \mathrm { p r o j } } ^ { \mathrm { d e c } } = 2 B d \left( H _ { q } d _ { k } + G r _ { k } ^ { G } + G r _ { v } ^ { G } \right) .\tag{27}
$$

For each group, reconstructing S cached key latents from dimension $r _ { k } ^ { G }$ to $s d _ { k }$ costs $2 B S r _ { k } ^ { G } s d _ { k }$ Across all $G$ groups,

$$
\begin{array} { r } { F _ { K , \mathrm { r e c } } ^ { \mathrm { p r e } } = 2 B S G r _ { k } ^ { G } s d _ { k } = 2 B S H _ { \mathrm { k v } } r _ { k } ^ { G } d _ { k } . } \end{array}\tag{28}
$$

The same cache-wide reconstruction is required for each decoding step:

$$
F _ { K , \mathrm { r e c } } ^ { \mathrm { d e c } } = 2 B S H _ { \mathrm { k v } } r _ { k } ^ { G } d _ { k } .\tag{29}
$$

Because reconstructed keys have the original head dimension $d _ { k } .$ , the query-key products and value products cost

$$
F _ { Q K } ^ { \mathrm { p r e } } = 2 B S ^ { 2 } H _ { q } d _ { k } , \qquad F _ { Q K } ^ { \mathrm { d e c } } = 2 B S H _ { q } d _ { k } .\tag{30}
$$

$$
F _ { A V } ^ { \mathrm { p r e } } = 2 B S ^ { 2 } G r _ { V } ^ { G } , \qquad F _ { A V } ^ { \mathrm { d e c } } = 2 B S G r _ { V } ^ { G } .\tag{31}
$$

PALU folds the value-side reconstruction into the output projection, so attention-value aggregation can be performed in the grouped latent value space. Let the total latent value width across all groups be $d _ { V } ^ { \mathrm { l a t e n t } } = G r _ { v } ^ { G }$ . The precise arithmetic depends on the mapping between query heads and value groups; we therefore retain the value-side latent width explicitly.

The fused output projection maps the concatenated latent value representation back to the model dimension, giving

$$
F _ { O } ^ { \mathrm { p r e } } = 2 B S G r _ { v } ^ { G } d , \qquad F _ { O } ^ { \mathrm { d e c } } = 2 B G r _ { v } ^ { G } d .\tag{32}
$$

The prefill cost is

$$
F _ { P A L U } ^ { p r e } = 2 B S \left[ d \left( H _ { q } d _ { k } + G r _ { k } ^ { G } + G r _ { v } ^ { G } \right) + H _ { \mathrm { k v } } r _ { k } ^ { G } d _ { k } + S H _ { q } d _ { k } + G r _ { V } ^ { G } ( S + d ) \right] .\tag{33}
$$

Ignoring the S-independent projection and output terms, the dominant cache-length-dependent decoding cost is

$$
\begin{array} { r } { F _ { \mathrm { P a l u , L C } } ^ { \mathrm { d e c } } \simeq 2 B S \left( H _ { \mathrm { k v } } r _ { k } ^ { G } d _ { k } + H _ { q } d _ { k } + G r _ { v } ^ { G } \right) . } \end{array}\tag{34}
$$

Ours Our method reduces the architectural query/key and value head dimensions to $r _ { k }$ and $r _ { v }$ respectively. All compression transformations are folded into the query, key, value, and output projection matrices. Consequently, neither key nor value reconstruction is required during inference. The native-reduced dimensional attention FLOPs are as follows

During prefill,

$$
\begin{array} { r } { F _ { \mathrm { O u r s } } ^ { \mathrm { p r e } } = 2 B S d \left( H _ { q } r _ { k } + H _ { \mathrm { k v } } r _ { k } + H _ { \mathrm { k v } } r _ { v } \right) + 2 B S ^ { 2 } H _ { q } ( r _ { k } + r _ { v } ) + 2 B S H _ { q } r _ { v } d . } \end{array}\tag{35}
$$

During decoding,

$$
\begin{array} { r } { F _ { \mathrm { O u r s } } ^ { \mathrm { d e c } } = 2 B d \left( H _ { q } r _ { k } + H _ { \mathrm { k v } } r _ { k } + H _ { \mathrm { k v } } r _ { v } \right) + 2 B S H _ { q } ( r _ { k } + r _ { v } ) + 2 B H _ { q } r _ { v } d . } \end{array}\tag{36}
$$

The cache-length-dependent decoding cost is therefore

$$
F _ { \mathrm { O u r s , L C } } ^ { \mathrm { d e c } } \simeq 2 B S H _ { q } ( r _ { k } + r _ { v } ) .\tag{37}
$$

For $d _ { k } = d _ { \tau }$ and $r _ { k } = r _ { v } = d _ { k } / 2$ , this term is exactly half that of full attention:

$$
\frac { F _ { \mathrm { O u r s , L C } } ^ { \mathrm { d e c } } } { F _ { \mathrm { F u l l , L C } } ^ { \mathrm { d e c } } } = \frac { r _ { k } + r _ { v } } { d _ { k } + d _ { v } } = \frac { 1 } { 2 } .\tag{38}
$$

RAP follows similar compressed attention as Ours and therefore the prefill and decoding FLOPs would be alike. Similarly, Think Keys Full Values also compresses attention while keeping Values in full dimension $r _ { v } = d _ { v }$

## A.4.1 DERIVATION OF TABLE 1

Table 1 compares methods under the representative grouped-query attention configuration $d = 1 5 3 6 ,$ $H _ { q } = 6 , \bar { H _ { k v } } = 2 .$ , and $d _ { k } = d _ { v } = 1 2 8$ . For RACE, RAP, and EigenAttention, we use $r _ { k } = r _ { v } = 6 4$ while Thin Keys uses $r _ { k } = 6 4$ and $r _ { v } = 1 2 8 .$ . For PALU G-LRD, the two KV heads form a single group $( G = 1 , s = 2 )$ with group ranks $r _ { k } ^ { G } = r _ { v } ^ { G } = 1 2 8 ;$ these ranks are defined per group rather than per head. We count a multiply–add as two FLOPs and omit elementwise operations. For prefill, we report the coefficient of the dominant $O ( S ^ { 2 } )$ attention computation, while for single-token decoding, we report all operations that scale linearly with the cached sequence length S. Common factors such as $2 B S ^ { 2 }$ for prefill and 2BS for decoding therefore cancel in the normalized ratios. This convention captures the distinction between native dimensional reduction, where attention is performed directly in the compressed representation, and reconstruction-based methods, which incur additional cache-length-dependent work during decoding.

RACE, RAP, and Thin Keys. Both methods execute attention natively with $r _ { k } ~ = ~ r _ { v } ~ = ~ 6 4$ without reconstructing the discarded dimensions. Their cache retention is

$$
\rho _ { \mathrm { c a c h e } } = { \frac { H _ { k v } ( r _ { k } + r _ { v } ) } { H _ { k v } ( d _ { k } + d _ { v } ) } } = { \frac { 6 4 + 6 4 } { 1 2 8 + 1 2 8 } } = 0 . 5 .
$$

The same dimensional reduction applies to the Q, K, V, and output projections, giving $P _ { \mathrm { n a t i v e } } / P _ { \mathrm { f u l l } } = 0 . 5$ . Since both the query–key and attention–value products are evaluated directly in dimensions $r _ { k }$ and $r _ { v } ,$ , the normalized dominant prefill and decoding costs are also

$$
\rho _ { \mathrm { p r e f i l l } } = \rho _ { \mathrm { d e c o d e } } = { \frac { H _ { q } ( r _ { k } + r _ { v } ) } { H _ { q } ( d _ { k } + d _ { v } ) } } = 0 . 5 .
$$

For the Thin Keys baseline, $r _ { v } = 1 2 8$ as they do not compress values. This gives $\rho _ { c a c h e } = 0 . 7 5$ and $\rho _ { p r e f i l l } = \rho _ { d e c o d e } = 0 . 7 5$

EigenAttention. EigenAttention stores keys and values in reduced dimensions $r _ { k } = r _ { v } = 6 4$ giving 50% KV-cache retention. Following the parameter-counting convention in Table 1, we count the compressed projections together with the learned key and value bases:

$$
P _ { \mathrm { E i g e n } } = d \left( H _ { q } d _ { k } + H _ { k v } r _ { k } + H _ { k v } r _ { v } + H _ { q } r _ { v } \right) + H _ { k v } ( r _ { k } d _ { k } + r _ { v } d _ { v } ) .\tag{39}
$$

For the representative configuration used for parameter counting, $d = 1 5 3 6 , H _ { q } = H _ { k v } = 6 $ $d _ { k } = d _ { v } = 1 2 8$ , and $r _ { k } = r _ { v } = 6 4$ , this gives

$$
\frac { P _ { \mathrm { E i g e n } } } { P _ { \mathrm { f u l l } } } = 0 . 6 9 8
$$

During prefill, key reconstruction scales as $O ( S )$ , while the query-key and attention-value products scale as $\mathsf { \bar { O } } ( S ^ { 2 } )$ . Hence, asymptotically the dominant prefill cost is

$$
\rho _ { \mathrm { p r e f l l } } ^ { \mathrm { E i g e n } } = { \frac { H _ { q } ( d _ { k } + r _ { v } ) } { H _ { q } ( d _ { k } + d _ { v } ) } } = { \frac { 1 2 8 + 6 4 } { 1 2 8 + 1 2 8 } } = 0 . 7 5 .
$$

For decoding, however, every cached compressed key must be mapped back to the original key dimension before query-key attention. The cache-length-dependent cost $F _ { \mathrm { E i g e n , L C } } ^ { \mathrm { d e c } } \simeq$ $2 B S \left[ H _ { k v } r _ { k } d _ { k } + H _ { q } ( d _ { k } + r _ { v } ) \right]$ . Normalizing by full attention, $F _ { \mathrm { F u l l , L C } } ^ { \mathrm { d e c } } = 2 B S H _ { q } ( d _ { k } + \Breve { d _ { v } } )$ , gives

$$
\rho _ { \mathrm { d e c o d e } } ^ { \mathrm { E i g e n } } = \frac { H _ { k v } r _ { k } d _ { k } + H _ { q } ( d _ { k } + r _ { v } ) } { H _ { q } ( d _ { k } + d _ { v } ) } = 1 1 . 4 2 .
$$

PALU. PALU G-LRD jointly compresses groups of KV heads. For the parameter comparison in Table 1, we use $H _ { k v } = 2$ heads divided into $G = 1$ groups of $s \ : = \ : 2$ heads, with group ranks $r _ { k } ^ { G } = r _ { v } ^ { \tilde { G } } = 1 2 8$ . The resulting cache stores

$$
C _ { \mathrm { P A L U } } = G ( r _ { k } ^ { G } + r _ { v } ^ { G } ) = 3 ( 1 2 8 + 1 2 8 ) = 7 6 8
$$

scalars per token, corresponding to 50% of the full KV cache.

Following the parameter-counting convention in Table 1, we count the query projection, both lowrank factor matrices for the grouped key and value projections, and the fused value-output map: $P _ { \mathrm { P A L U } } = d H _ { q } d _ { k } + G r _ { k } ^ { G } ( d + \stackrel { \smile } { s } d _ { k } \stackrel {  } { ) } + G r _ { v } ^ { \overline { { G } } } ( d + s d _ { v } ) \stackrel {  } { + } \acute { d G } r _ { v } ^ { G }$ which gives

$$
\frac { P _ { \mathrm { P A L U } } } { P _ { \mathrm { f u l l } } } = 0 . 5 8 3 .
$$

PALU reconstructs cached keys into the original key dimension before query-key attention, while value aggregation is performed directly in the grouped latent value space. Consequently, the dominant cache-length-dependent decoding cost is

$$
\begin{array} { r } { F _ { \mathrm { P A L U , L C } } ^ { \mathrm { d e c } } \simeq 2 B S \left[ H _ { k v } r _ { k } ^ { G } d _ { k } + H _ { q } d _ { k } + G r _ { v } ^ { G } \right] . } \end{array}
$$

Normalizing by full attention gives

$$
\rho _ { \mathrm { d e c o d e } } ^ { \mathrm { P A L U } } = \frac { H _ { k v } r _ { k } ^ { G } d _ { k } + H _ { q } d _ { k } + G r _ { v } ^ { G } } { H _ { q } ( d _ { k } + d _ { v } ) } . = 2 1 . 9 2 .
$$

Fore prefill,

$$
\rho _ { \mathrm { p r e f i l } } ^ { \mathrm { P A L U } } = \frac { H _ { q } d _ { k } + G r _ { v } ^ { G } } { H _ { q } ( d _ { k } + d _ { v } ) } . = 0 . 7 5
$$

## A.5 APPROXIMATE MATRIX MULTIPLICATION VIEW OF ATTENTION COMPRESSION

This section formalizes the connection between head-dimension compression and approximate matrix multiplication (AMM), and clarifies how our RoPE-constrained coordinate selection relates to existing sampling-based AMM results.

AMM formulation. Consider a single attention head with post-RoPE query and key matrices $Q , K \in \mathbb { R } ^ { S \times d _ { k } }$ . The pre-softmax attention interaction is $L = \bar { Q } K ^ { \top }$

Following the standard AMM formulation, let

$$
A = Q ^ { \top } \in \mathbb { R } ^ { d _ { k } \times S } , \qquad B = K ^ { \top } \in \mathbb { R } ^ { d _ { k } \times S } ,
$$

so that $A ^ { \top } B = Q K ^ { \top }$ . Given a dimensionality-reducing operator $\Pi \in \mathbb { R } ^ { r _ { k } \times d _ { k } }$ , AMM replaces this product by

$$
( \Pi A ) ^ { \top } ( \Pi B ) = Q \Pi ^ { \top } \Pi K ^ { \top } .\tag{40}
$$

Thus, dimensionality reduction along the shared row dimension of A and B is exactly dimensionality reduction along the head-feature dimension of $Q$ and $K$

Stable-rank AMM. For conforming matrices A and $B ,$ (Cohen et al., 2016) define a $( k , \epsilon )$ spectral-norm AMM property of the form

$$
\left\| ( \Pi A ) ^ { \top } ( \Pi B ) - A ^ { \top } B \right\| _ { 2 } \leq \epsilon { \sqrt { \left( \| A \| _ { 2 } ^ { 2 } + { \frac { \| A \| _ { F } ^ { 2 } } { k } } \right) \left( \| B \| _ { 2 } ^ { 2 } + { \frac { \| B \| _ { F } ^ { 2 } } { k } } \right) } } .\tag{41}
$$

They show that suitable oblivious embeddings can satisfy this property using

$$
m = \mathcal { O } \left( \frac { k + \log ( 1 / \delta ) } { \epsilon ^ { 2 } } \right)
$$

rows with probability at least $1 - \delta$

The stable rank of a matrix A is

$$
\operatorname { s r } ( A ) = { \frac { \| A \| _ { F } ^ { 2 } } { \| A \| _ { 2 } ^ { 2 } } } .\tag{42}
$$

Setting

$$
k \gtrsim \operatorname* { m a x } \{ \operatorname { s r } ( A ) , \operatorname { s r } ( B ) \}
$$

makes the Frobenius-to-spectral terms in $\operatorname { E q } .$ . equation 41 controlled by the corresponding spectral norms. Consequently, the required sketch dimension can scale with the effective spectral dimension of the operands rather than their ambient dimension. Applied to attention with $\dot { A } = Q ^ { \top }$ and $B =$ $K ^ { \top }$ , this provides theoretical motivation for reducing $d _ { k }$ when the query and key activations have low stable rank.

From row sampling in AMM to feature selection in attention. AMM sampling methods act on the shared row dimension of the two operands. Under the identification $A = \widetilde Q ^ { \top } ; B = \widetilde K ^ { \top }$ the rows of A and B correspond to the feature dimensions of $\widetilde { Q }$ and $\widetilde { K }$ . Let

$$
P _ { \mathcal { T } } = [ e _ { i _ { 1 } } , \dots , e _ { i _ { r } } ] \in \{ 0 , 1 \} ^ { d _ { k } \times r _ { k } }
$$

select the coordinate set $\mathcal { T } ,$ and take $\Pi = P _ { \mathcal { I } } ^ { \top }$ . The AMM approximation then becomes

$$
\begin{array} { r } { ( \Pi A ) ^ { \top } ( \Pi B ) = \widetilde { Q } P _ { \mathcal { T } } P _ { \mathcal { T } } ^ { \top } \widetilde { K } ^ { \top } } \\ { = ( \widetilde { Q } P _ { \mathcal { T } } ) ( \widetilde { K } P _ { \mathcal { T } } ) ^ { \top } . } \end{array}\tag{43}
$$

Thus, row selection in AMM is exactly feature selection in attention, with the same selection matrix applied to queries and keys.

Equation 43 motivates two design choices. First, queries and keys should use an identical reduction $P _ { \mathcal { T } }$ or in the context of RACE, $\bar { P } _ { Q K }$ rather than independently constructed maps $P _ { Q }$ and $P _ { K }$

Separate maps would produce $( \tilde { Q } P _ { Q } ) ( \tilde { K } P _ { K } ) ^ { \top } = \tilde { Q } P _ { Q } P _ { K } ^ { \top } \tilde { K } ^ { \top }$ , which is not the shared-sketch AMM approximation $\widetilde { Q } P _ { Q K } P _ { Q K } ^ { \top } \widetilde { K } ^ { \top }$

Second, because the approximation target is the joint product $\widetilde Q \widetilde K ^ { \top }$ , a data-dependent choice of $P _ { Q K }$ should account for both operands. Selecting dimensions using only query or only key statistics can discard features that have low importance for one operand but materially affect their interaction.

These observations determine the form of the reduction - a common, jointly informed Q/K selector, but not its precise selection rule. General AMM results show that appropriately chosen subsets of the shared dimension can preserve matrix products. RACE further restricts this family to selectors that are compatible with RoPE and derives an activation-energy criterion for choosing among them.

Relation to our selector. RACE’s compression operator is a more restricted member of this general family. For each attention head, we use an unweighted binary selection matrix $P _ { Q K } =$ $[ e _ { i } ] _ { i \in \pmb { \mathbb { Z } } _ { Q K } }$ , where $\mathcal { T } _ { Q K }$ is chosen from calibration activations. Unlike the general data-dependent sampling operators covered by AMM theory, our selector must additionally satisfy the structural constraints induced by RoPE. Specifically, if $\mathbf { \mathcal { P } } _ { \mathrm { r o p e } } = \{ ( i _ { m } , j _ { m } ) \} _ { m = 1 } ^ { M }$ denotes the rotary coordinate pairs, then

$$
\mathcal { I } _ { Q K } = \bigcup _ { m \in \cal { S } } \{ i _ { m } , j _ { m } \} , \qquad | { \cal { S } } | = \frac { r _ { k } } { 2 } .\tag{44}
$$

The selected set is therefore a union of complete two-dimensional rotary blocks.

Energy-based selection and scope of the guarantee. The AMM results above establish that lowdimensional and sampling-based representations capable of preserving matrix products exist under appropriate conditions. They do not imply that every coordinate subset, or specifically our RoPEconstrained subset, satisfies the same AMM bound.

Our method chooses among the admissible RoPE pairs using calibration energy. For pair $( i _ { m } , j _ { m } )$ we define

$$
E _ { m } = \| Q _ { : , i _ { m } } \| _ { 2 } ^ { 2 } + \| Q _ { : , j _ { m } } \| _ { 2 } ^ { 2 } + \| K _ { : , i _ { m } } \| _ { 2 } ^ { 2 } + \| K _ { : , j _ { m } } \| _ { 2 } ^ { 2 } ,\tag{45}
$$

and retain the $r _ { k } / 2$ highest-energy pairs. This criterion favors coordinates that carry substantial query/key activation energy while remaining compatible with RoPE.

We therefore use stable-rank AMM and sampling-based AMM as theoretical motivation for reducing the shared query–key dimension, rather than claiming a direct AMM guarantee for the proposed energy-based selector. The exact property guaranteed by our construction is structural: selecting complete rotary pairs permits RoPE and attention to be executed natively in the reduced space.

## A.6 PROOFS

Proof of Theorem 1. [RoPE-compatible coordinate selection]

Proof. Suppose $P _ { Q K }$ selects exactly the coordinates belonging to the complete rotary pairs indexed by ${ \mathcal { S } } \subseteq \{ 1 , \ldots , M \}$ , with its columns ordered consistently with the blocks of $R _ { t } ^ { ( S ) }$ . Since

$$
R _ { t } = \operatorname { d i a g } ( R _ { 1 , t } , \ldots , R _ { M , t } )
$$

acts independently on each rotary pair, applying $R _ { t }$ to the selected coordinates applies precisely the blocks $R _ { m , t }$ for $m \in { \mathcal { S } }$ . Therefore,

$$
R _ { t } P _ { Q K } = P _ { Q K } \mathrm { d i a g } ( R _ { m , t } : m \in \mathcal { S } ) = P _ { Q K } R _ { t } ^ { ( S ) }
$$

for every t. Hence $P _ { Q K }$ is RoPE-compatible.

Conversely, suppose

$$
R _ { t } P _ { Q K } = P _ { Q K } R _ { t } ^ { ( S ) } \qquad { \mathrm { ~ f o r ~ a l l ~ } } t .
$$

The right-hand side lies in col $. ( P _ { Q K } )$ , and therefore

$$
R _ { t } \operatorname { c o l } ( P _ { Q K } ) \subseteq \operatorname { c o l } ( P _ { Q K } ) \qquad { \mathrm { f o r ~ a l l ~ } } t .
$$

Thus, the coordinate subspace selected by $P _ { Q K }$ must be invariant under every $R _ { t }$

Consider any rotary pair $( i _ { m } , j _ { m } )$ . Suppose, for contradiction, that $P _ { Q K }$ selects $i _ { m }$ but not $j _ { m }$ . Then $e _ { i _ { m } } \in \mathrm { c o l } ( P _ { Q K } )$ , whereas $e _ { j _ { m } } \not \in \mathrm { c o l } ( P _ { Q K } )$ . For the m-th rotary block,

$$
R _ { m , t } = \left[ \begin{array} { c c } { \cos \phi _ { m , t } } & { - \sin \phi _ { m , t } } \\ { \sin \phi _ { m , t } } & { \cos \phi _ { m , t } } \end{array} \right] ,
$$

so, up to the coordinate convention for the pair,

$$
R _ { t } e _ { i _ { m } } = \cos \phi _ { m , t } e _ { i _ { m } } + \sin \phi _ { m , t } e _ { j _ { m } } .
$$

By assumption, pair m undergoes a nontrivial rotation for at least one position, so there exists t such that sin $\phi _ { m , t } \neq 0$ . For this $t , R _ { t } e _ { i _ { m } }$ has a nonzero component along $e _ { j _ { m } }$ and therefore

$$
R _ { t } e _ { i _ { m } } \notin \mathrm { c o l } ( P _ { Q K } ) ,
$$

contradicting the invariance established above.

The same argument applies if $j _ { m }$ is selected but $i _ { m }$ is not. Hence, for every rotary pair, either both coordinates are selected or neither is selected. Therefore the coordinates selected by $P _ { Q K }$ form a union of complete rotary pairs, indexed by some ${ \mathcal { S } } \subseteq \{ 1 , \ldots , M \}$

## Proof of Proposition 1. [Error bound for RoPE-pair truncation]

Proof. Partition the post-RoPE query and key matrices into RoPE-aligned coordinate pairs,

$$
\widetilde { \cal Q } = [ \widetilde { Q } _ { 1 } , \ldots , \widetilde { Q } _ { M } ] , \qquad \widetilde { \cal K } = [ \widetilde { K } _ { 1 } , \ldots , \widetilde { K } _ { M } ] ,
$$

where each

$$
\tilde { Q } _ { m } , \tilde { K } _ { m } \in \mathbb { R } ^ { S \times 2 } .
$$

Because matrix multiplication along the feature dimension decomposes additively across these blocks,

$$
\tilde { Q } \tilde { K } ^ { \top } = \sum _ { m = 1 } ^ { M } \tilde { Q } _ { m } \tilde { K } _ { m } ^ { \top } .
$$

For a retained pair set $S \subseteq [ M ]$ , the corresponding truncated product is

$$
\widetilde { Q } _ { S } \widetilde { K } _ { S } ^ { \top } = \sum _ { m \in S } \widetilde { Q } _ { m } \widetilde { K } _ { m } ^ { \top } .
$$

Hence the truncation error would be,

$$
\widetilde { Q } \widetilde { K } ^ { \top } - \widetilde { Q } _ { S } \widetilde { K } _ { S } ^ { \top } = \sum _ { m \notin S } \widetilde { Q } _ { m } \widetilde { K } _ { m } ^ { \top } .\tag{46}
$$

Applying the triangle inequality for the Forbenius norm gives

$$
\left\| \widetilde { Q } \widetilde { K } ^ { \top } - \widetilde { Q } _ { S } \widetilde { K } _ { S } ^ { \top } \right\| _ { F } \leq \sum _ { m \not \in S } \left\| \widetilde { Q } _ { m } \widetilde { K } _ { m } ^ { \top } \right\| _ { F } .\tag{47}
$$

Since for any conforming matrices A and $B , \left. A B ^ { T } \right. _ { F } \leq \left. A \right. _ { F } \left. B \right. _ { F }$ , the above equation can be written as

$$
\left\| \widetilde { Q } \widetilde { K } ^ { \intercal } - \widetilde { Q } s \widetilde { K } _ { s } ^ { \intercal } \right\| _ { F } \leq \sum _ { m \not \in \mathcal { S } } \left\| \widetilde { Q } _ { m } \right\| _ { F } \left\| \widetilde { K } _ { m } \right\| _ { F } .\tag{48}
$$

Finally, by the arithmetic–geometric mean inequality, we obtain

$$
\| \widetilde { Q } _ { m } \| _ { F } \| \widetilde { K } _ { m } \| _ { F } \leq \frac { 1 } { 2 } \left( \| \widetilde { Q } _ { m } \| _ { F } ^ { 2 } + \| \widetilde { K } _ { m } \| _ { F } ^ { 2 } \right) = \frac { 1 } { 2 } E _ { m } .\tag{49}
$$

Combining the above inequalities yields

$$
\left| \left\| \widetilde { Q } \widetilde { K } ^ { \top } - \widetilde { Q } s \widetilde { K } _ { s } ^ { \top } \right\| _ { 2 } \leq \sum _ { m \not \in S } \left\| \widetilde { Q } _ { m } \right\| _ { F } \left\| \widetilde { K } _ { m } \right\| _ { F } \leq \frac { 1 } { 2 } \sum _ { m \not \in S } E _ { m } . \right\|\tag{50}
$$

For a fixed budget $| S | = k ,$ , minimizing the right-hand side is equivalent to maximizing

$$
\sum _ { m \in S } E _ { m } .
$$

Therefore, the subset consisting of the k largest-energy RoPE pairs minimizes this upper bound among all subsets of k pairs. □

Evaluation in the pre- or post-RoPE space Because each RoPE pair is transformed by an orthogonal $2 \times 2$ rotation,

$$
\left\| \widetilde { Q } _ { m } \right\| _ { F } = \left\| Q _ { m } \right\| _ { F } \qquad \left\| \widetilde { K } _ { m } \right\| _ { F } = \left\| K _ { m } \right\| _ { F }
$$

Hence,

$$
E _ { m } = \left\| \widetilde { Q } _ { m } \right\| _ { F } ^ { 2 } + \left\| \widetilde { K } _ { m } \right\| _ { F } ^ { 2 } = \left\| Q _ { m } \right\| _ { F } ^ { 2 } + \left\| K _ { m } \right\| _ { F } ^ { 2 }
$$

The pair-energy criterion can therefore be evaluated equivalently in the pre- or post-RoPE space. The implementation accumulates the joint Q/K second moment and ranks each pair by the sum of its two diagonal entries, which is proportional to this $E _ { m }$ . In pour implementations we use the post-RoPE values for the calibration.

Connection to attention-output perturbation. Let

$$
Z = \frac { \widetilde Q \widetilde K ^ { \top } } { \sqrt { d _ { k } } } , \qquad Z _ { S } = \frac { \widetilde Q _ { S } \widetilde K _ { S } ^ { \top } } { \sqrt { d _ { k } } } .
$$

The Jacobian of softmax at a probability vector $p$ is $\mathrm { d i a g } ( p ) - p p ^ { \top }$ , whose spectral norm is at most $1 / 2$ . Applying this result row-wise gives

$$
\| A - A _ { S } \| _ { F } \leq { \frac { 1 } { 2 } } \| Z - Z _ { S } \| _ { F } .
$$

Because $Z - Z _ { S } \in \mathbb { R } ^ { S _ { \mathrm { c a l } } \times S _ { \mathrm { c a l } } }$ <sup>al</sup> ,

$$
\begin{array} { r l } & { \left\| Z - Z _ { \cal S } \right\| _ { { \cal F } } \leq \sqrt { { \cal S } _ { \mathrm { c a l } } } \left\| Z - Z _ { \cal S } \right\| _ { 2 } } \\ & { \qquad = \frac { \sqrt { { \cal S } _ { \mathrm { c a l } } } } { \sqrt { d _ { \cal k } } } \left\| \widetilde Q \widetilde K ^ { \top } - \widetilde Q _ { \cal S } \widetilde K _ { \cal S } ^ { \top } \right\| _ { 2 } . } \end{array}
$$

Table 2: Comparison of methods that compress KV Cache along head dimension. RACE operates natively in the reduced dimensions without reconstruction. See A.4.1 for the derivations.
<table><tr><td></td><td>Total Inference TFLOPs</td><td>TTFT (ms)</td><td>Prefill T.put (tok/sec)</td><td>Decode T.put (tok/sec)</td><td>Total T.put (tok/sec)</td><td>GPU Memory (GB)</td></tr><tr><td>Eager</td><td>3.973</td><td>41.051</td><td>13937</td><td>45.186</td><td>224.44</td><td>5.884</td></tr><tr><td>SDPA</td><td>3.973</td><td>28.854</td><td>17755.36</td><td>56.864</td><td>282.904</td><td>5.882</td></tr><tr><td>Flash Attention 3</td><td>3.973</td><td>34.170</td><td>15054</td><td>46.23</td><td>230.127</td><td>5.882</td></tr></table>

For fixed values V,

$$
\begin{array} { r l } {  { \| A V - A s V \| _ { F } \leq \| A - A s \| _ { F } \| V \| _ { 2 } } } \\ & { \leq \frac { \sqrt { S _ { \mathrm { c a l } } } \| V \| _ { 2 } } { 4 \sqrt { d _ { k } } } \displaystyle \sum _ { m \not \in \mathcal { S } } E _ { m } , } \end{array}
$$

where the final inequality follows from Proposition 1.

More generally, if the downstream calibration loss is locally $L _ { \mathrm { d o w n } ^ { - 1 } }$ Lipschitz with respect to the attention output, then

$$
| \mathcal { L } _ { \mathrm { d o w n } } ( A V ) - \mathcal { L } _ { \mathrm { d o w n } } ( A _ { \mathcal { S } } V ) | \leq L _ { \mathrm { d o w n } } \frac { \sqrt { S _ { \mathrm { c a l } } } \| V \| _ { 2 } } { 4 \sqrt { d _ { k } } } \sum _ { m \not \in S } E _ { m } .
$$

This is a conditional, calibration-dependent bound and does not imply that energy selection globally minimizes language-model loss.

## A.7 IMPLEMENTATION DETAILS

Software and model execution. We implement RACE in PyTorch and Hugging Face Transformers. Our implementation supports GPT-NeoX, Llama, Qwen2, Pythia, and Granite-style attention modules, including both multi-head and grouped-query attention. Although all models reported in the main paper use full-RoPE or RoPE applied to all dimensions, our implementation supports partial RoPE as well by applying pairwise selection only within the rotary-encoded subspace while leaving any non-rotary dimensions unconstrained by the RoPE-pair requirement. Pre-trained models are loaded and evaluated in BF16 on GPU, while calibration statistics and projection construction use higher-precision accumulation during calibration. The folded compressed model uses standard attention operations and does not require a custom CUDA or Triton kernel. Unless otherwise stated, we use PyTorch scaled dot-product attention (SDPA) for the compressed runtime. The implementation also supports eager attention and FlashAttention-3 as shown below.

Furthermore, PALU rely on method-specific execution paths for efficient inference, including custom fused kernels. We were unable to reliably reproduce these optimized kernels in our hardware/software environment and therefore avoid presenting potentially misleading wall-clock comparisons. Instead, we compare algorithmic attention FLOPs across methods and measure RACE end-to-end using standard PyTorch attention backends. This separates implementation-specific kernel efficiency from the architectural distinction that RACE performs attention directly in the reduced dimensions without runtime reconstruction. The original rotary-frequency indices and the model’s RoPE scaling rule are preserved; selected frequencies are not re-indexed as consecutive frequencies. For grouped-query attention (GQA), all query heads associated with a given KV head use the same selected key subspace. For partial RoPE models like Pythia models, the RoPE pairs are considered as pairs and the non-RoPE dimesnisons are paired with adjacent dimensions as in interleaved RoPE as shown in Figure 9.

Calibration in RACE. For each attention layer, we collect query and key activations after the pretrained RoPE transformation. Our implementation also support calibration with pre-RoPE queries and keys and we also show that they are equivalent in A.6. We preserve the physical rotary-pair layout of the original architecture rather than assuming that adjacent tensor coordinates necessarily form a rotary pair. For each rotary pair m, we accumulate the joint activation-energy score as in Eq.4 and retain the $r _ { k } / 2$ highest-scoring pairs. Selection is performed independently for each layer and attention head (or KV head under grouped-query attention), subject to retaining both coordinates of every selected RoPE pair.

Note that models like Llama, Qwen, and Granite have full RoPE where all dimensions have RoPE and in such cases the sequential selection shown in 9 Left will not be present. Such models also have sequential RoPE $[ x _ { 0 } , x _ { d / 2 } ] , [ x _ { 1 } , x _ { d / 2 + 1 } ] .$

Our implementations supports both sequential and interleaved $[ x _ { 0 } , x _ { 1 } ]$ , [x<sub>2</sub>, x<sub>3</sub>], . . . RoPE models. The values are not subject to RoPE. We employ $P _ { v } ,$ the right signular vectors of V for the compression. This do not follow selection as shown in Figure 9 Right.

![](images/b91f2e7d7d35b4355ef9f344829f91f4c8cacf283b6ff093560a7f01898db859.jpg)

![](images/ac41528d7a6f12134a7b07d39e027ac2bca99e0c3f1efa7574bcfc4c18499f4c.jpg)  
Figure 9: Representative RACE $P _ { Q K }$ and $P _ { V }$ matrices before folding. Left: Each column corresponds to one compressed Q/K coordinate and each row to an original head coordinate. Nonzero entries indicate the coordinates retained by $P _ { Q K }$ . Right: Each column corresponds to one compressed V coordinate and each row to an original head coordinate.

Optional post-compression recovery. RACE is a post-training compression method and can be evaluated immediately after constructing and folding the reduced Q/K and V subspaces into the pretrained attention projections. At aggressive compression ratios, however, this discrete change in attention dimensionality can perturb the pretrained representation enough that a short optimization stage is highly beneficial. Although optional, all results in the main paper involve this recovery. Following folding, we use the original uncompressed checkpoint as a frozen teacher and optimize the compressed model using KL distillation on FineWeb data. The implementation uses a temperature of $\overset T { = } 2$ and minimizes the divergence between the teacher and compressed-model output distributions. We train for 250 optimizer steps with sequences of length 512, a micro-batch size of 2, and gradient accumulation over 4 micro-batches, corresponding to approximately 1.0M training tokens.

Figure 10 separates the effect of this recovery from the compression procedure itself by reporting both the directly folded models $( ^ { 6 6 } \mathrm { w } / \mathrm { o } \ \mathrm { F T } ^ { 3 } )$ and their recovered counterparts. At 75% and 87.5% KV-cache retention, the directly compressed models are already relatively close to their recovered versions, indicating that moderate dimensionality reduction introduces limited disruption. In contrast, at 50% retention, both RACE and RAP experience substantially larger degradation without recovery, particularly in perplexity. The optimization stage is not necessary to define or execute the compressed model. RACE produces a valid reconstruction-free reduced-dimensional attention layer immediately after folding. Rather, recovery provides an optional mechanism for adapting the pretrained weights when the dimensionality change becomes sufficiently aggressive. Moreover, the same recovery procedure also substantially improves RAP, suggesting that much of the recoverable degradation at aggressive compression reflects adaptation to the compressed representation rather than a method-specific training advantage.

Systems Details All systems experiments were conducted on a single NVIDIA H100 80GB HBM3 GPU. The machine used NVIDIA driver 590.48.01 with CUDA 13.1, and all reported memory, latency, throughput, and FLOP-related measurements were collected on this hardware configuration. Unless otherwise noted, experiments were run on a single GPU without model parallelism using SDPA kernels, so comparisons between the base model and compressed variants reflect change induced by the compression method rather than differences in hardware allocation.

## A.8 RESULTS

Language Modeling and Zero-Shot Performance The main quality experiments evaluate Qwen2.5-7B, Llama-3.2-3B-Instruct, and Granite-4.1-3B. We compare against PALU-GLRD, EigenAttention, Thin Keys, and RAP at matched or reported KV-cache budgets. The full results are presented in Table 3 while only RAP, a reconstruction-free method like RACE, and PALU-GLRD, a reconstruction-based method, are included in Figure 3. Across all three model families, RACE consistently outperforms the reconstruction-free baselines RAP and Thin Keys, while remaining

![](images/7d46d5f44339409623375916fac981a91e4db46aa7b34a5bdc3b9ec5e3868164.jpg)  
(a) Llama-3.2-3B-Instruct Perplexity

![](images/82c3cd000a42733918f318b1cbbd0f24619f9f98f5227f169b43c6cbab479375.jpg)  
(b) Llama-3.2-3B-Instruct Average zero-shot accuracy

![](images/8f1b0a557554c56c57220969e34c288ba121bca0ac045d55fdcbadb61f6aa85e.jpg)  
(c) Qwen2.5 7B Perplexity

![](images/d4327e7f20d1b247bbbf5af5034950f0f12cc48132fa32e131ab69c05d91463a.jpg)  
(d) Qwen2.5 7B Average zero-shot accuracy

Figure 10: Effect of optional post-compression recovery. Perplexity and average zero-shot accuracy before and after a short recovery stage for RACE and RAP at different KV-cache retention ratios. “w/o FT” denotes the folded compressed checkpoint evaluated directly, while the solid curves apply post-fold KL distillation from the uncompressed model. Recovery has little effect at mild compression but substantially restores performance at 50% retention, particularly in perplexity.

![](images/5a6d317b136043ae5ca9b00fd289ed3b630465f90f202929b87c604c8acba983.jpg)  
(a) Total Parameter Memory

![](images/58f58eb82ed3d9cc199ce590986b9a699dd9ccebcd0cfbeb0cca77e6f2a61e9d.jpg)  
(b) Attention Parameter Memory

![](images/365f0ef5d10ef76a8668cc3ce03944ed32d044e0c8e489964e4a2d1277616efd.jpg)  
(c) KV Cache Memory  
Figure 11: Memory scaling of RACE with context length. Total model parameter memory (left), attention-parameter memory (center), and KV-cache memory (right) for the base model and RACE at 75% and 50% KV retention across 64K–256K-token contexts. Parameter memory is independent of sequence length and decreases after folding the reduced-dimensional projections, while KVcache memory scales with context length; RACE correspondingly reduces both the fixed attentionparameter footprint and the context-dependent cache footprint.

competitive with the reconstruction-based EigenAttention and PALU. The difference is most pronounced under aggressive compression: At 50% KV-cache retention, RACE achieves the lowest WikiText perplexity across all three model families while remaining competitive with PALU in average zero-shot accuracy. among the evaluated compression methods on each of Qwen2.5, Granite, and Llama. At more moderate compression, RACE remains close to the reconstruction-based methods; at 75% retention it preserves 98.5%, 96.5%, and 97.5% of the corresponding uncompressed average zero-shot accuracy on Qwen, Granite, and Llama, respectively. The per-task results also show that the effect of compression is not uniform across benchmarks: some tasks remain close to the uncompressed baseline even at lower retention, whereas others are more sensitive to reduced attention dimensionality. Table 4 presents the per-task results of Qwen2.5-7B and Llama-3.2-3B-Instruct on 16 English tasks in LongBench. RACE outperforms RAP at all evaluated retention levels and exceeds PALU at 87.5% and 75% retention on both model families. At 50% retention, RACE remains competitive with PALU, outperforming it on Qwen while trailing it modestly on Llama. Together, these results show that the quality-compression behavior is consistent across model families rather than being specific to a single architecture, while retaining the reconstruction-free execution of RACE.

Additional Systems Results The systems results in Figure 11 show that RACE reduces both the fixed attention-parameter footprint and the context-dependent KV-cache footprint. As expected, the reduction in attention parameter memory aligns with the compression. However, since the folded projection weights are smaller in comparison to the result of the model parameters and memory consumption elements, the reduction in memory is less prominant in the total parameter memory.

Table 3: Detailed quality results for head-dimension compression. We report WikiText perplexity and per-task zero-shot accuracy across the evaluated model families and KV-cache retention ratios. The final column reports the average zero-shot accuracy across tasks. For each KV Cache compression level per model, we highlight the best performing perplexity and average zero-shot accuracy. RACE is comparable to reconstruction-based methods while not requiring the reconstruction or the additional FLOPs.
<table><tr><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2"></td><td colspan="3"></td><td rowspan="2"></td><td colspan="3"></td><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2"></td><td colspan="3"></td><td rowspan="2"></td><td rowspan="2"></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="3"></td><td rowspan="3"></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td rowspan="3"></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>75%</td><td>64</td><td>128</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.628</td></tr><tr><td>RACE (Ours)</td><td>50% 87.50% 87.50% 75%</td><td>64 112 96 96 64</td><td>64 112 128 96 128</td><td>12.011 14.691 10.254 10.259 10.968</td><td>0.630 0.485 0.704 0.715</td><td>0.789 0.747 0.794 0.799</td><td>0.769 0.707 0.777 0.781</td><td>0.611 0.605 0.724 0.707</td><td>0.762 0.758 0.817 0.794</td><td>0.963 0.948 0.967 0.961</td><td>0.442 0.396 0.462 0.470</td><td>0.493 0.444 0.538 0.517</td><td>0.774 0.758 0.849 0.846</td><td>0.389 0.341 0.411 0.409</td><td>0.498 0.403 0.507 0.524</td><td>0.680 0.430 0.670 0.680</td><td>0.401 0.303 0.510 0.490</td><td>0.402 0.305 0.518 0.529</td><td>0.741 0.606 0.886 0.884</td><td>0.623 0.549 0.676 0.674</td></tr><tr><td>Base</td><td>75% 50%</td><td>64 64</td><td>64 64</td><td>10.916 13.538</td><td>0.668 0.690 0.562</td><td>0.794 0.795 0.785</td><td>0.765 0.764 0.713</td><td>0.722 0.687 0.656</td><td>0.838 0.789 0.748 Granite 4.100 3B</td><td>0.963 0.957 0.944</td><td>0.456 0.454 0.438</td><td>0.562 0.510 0.475</td><td>0.825 0.832 0.769</td><td>0.392 0.394 0.385</td><td>0.480 0.521 0.464</td><td>0.630 0.640 0.470</td><td>0.451 0.520 0.403</td><td>0.481 0.495 0.419</td><td>0.872 0.858 0.770</td><td>0.660 0.661 0.600</td></tr><tr><td></td><td>100% 87.50%</td><td>48 48</td><td>64 48</td><td>11.578 11.775</td><td>0.662 0.671</td><td>0.794 0.746 0.796 0.736</td><td>0.695 0.693</td><td>0.790 0.679</td><td>0.960 0.780 0.958</td><td>0.434 0.426</td><td>0.504 0.502</td><td>0.751</td><td>0.736</td><td>0.366 0.380</td><td>0.476 0.466</td><td>0.570 0.530</td><td>0.461 0.490</td><td>0.414 0.419</td><td>0.813</td><td>0.629 0.626</td></tr><tr><td>PALU</td><td>75% 75% 50% 87.50%</td><td>32 32 48</td><td>64 32 64</td><td>13.615 11.918 17.483</td><td>0.626 0.675 0.539</td><td>0.787 0.738 0.799 0.731 0.760 0.646</td><td>0.693 0.628</td><td>0.736 0.776 0.652</td><td>0.949 0.957 0.914</td><td>0.430 0.426 0.412</td><td>0.500 0.500 0.445</td><td>0.626 0.748 0.708</td><td>0.385 0.376 0.362</td><td></td><td>0.471 0.468 0.456</td><td>0.520 0.540 0.440</td><td>0.412 0.490 0.333</td><td>0.380 0.422 0.332</td><td>0.806 0.772 0.818 0.642</td><td>0.601 0.628 0.551</td></tr><tr><td>Eigen Attention</td><td>75% 75% 50%</td><td>48 32 32 48</td><td>48 64 32 64</td><td>11.775 13.481 29.25 23.361</td><td>0.671 0.577 0.162 0.313</td><td>0.796 0.785 0.724 0.744</td><td>0.736 0.723 0.545</td><td>0.693 0.657 0.539 0.588</td><td>0.779 0.757 0.566</td><td>0.958 0.952 0.765</td><td>0.426 0.416 0.37</td><td>0.502 0.513 0.354</td><td>0.736 0.729 0.388</td><td>0.379 0.376 0.301</td><td>0.466 0.440 0.429</td><td>0.53 0.56 0.35</td><td>0.49 0.421 0.186</td><td>0.419 0.360 0.255</td><td>0.806 0.761 0.196</td><td>0.626 0.601 0.408</td></tr><tr><td>Thin Keys Full Values</td><td>87.50% 75% 87.50% 75%</td><td>32 48 48</td><td>64 64 48</td><td>26.024 40.573 11.928</td><td>0.221 0.152 0.642</td><td>0.719 0.694 0.791</td><td>0.587 0.586 0.544 0.732</td><td>0.517 0.531</td><td>0.641 0.643 0.560</td><td>0.874 0.813 0.760</td><td>0.388 0.356 0.340</td><td>0.387 0.357 0.315</td><td>0.646 0.583 0.603</td><td>0.343 0.294 0.286</td><td>0.411 0.395 0.432</td><td>0.27 0.230 0.220</td><td>0.157 0.235 0.206</td><td>0.265 0.257 0.244</td><td>0.365 0.215 0.194</td><td>0.465 0.428 0.405</td></tr><tr><td>RAP</td><td>75% 50% 87.50%</td><td>32 32 48 48</td><td>64 32 64</td><td>12.795 13.963 17.766 12.092</td><td>0.575 0.563 0.359</td><td>0.782 0.780 0.700</td><td>0.712 0.706 0.641</td><td>0.678 0.662 0.639 0.588</td><td>0.784 0.755 0.754 0.684</td><td>0.942 0.948 0.945 0.914</td><td>0.430 0.424 0.428 0.350</td><td>0.478 0.453 0.449 0.403</td><td>0.777 0.759 0.724 0.605</td><td>0.382 0.373 0.344 0.344</td><td>0.469 0.467 0.438 0.405</td><td>0.530 0.340 0.270 0.190</td><td>0.490 0.373 0.275 0.206</td><td>0.396 0.357 0.273 0.243</td><td>0.782 0.672 0.444 0.262</td><td>0.620 0.577 0.535 0.460</td></tr><tr><td>RACE (Ours)</td><td>75% 75% 50%</td><td>32 32 128</td><td>48 64 32</td><td>13.148 13.041 16.210</td><td>0.633 0.611 0.586 0.516</td><td>0.789 0.787 0.776 0.757</td><td>0.733 0.718 0.687 0.647</td><td>0.672 0.670 0.659 0.586</td><td>0.783 0.773 0.763 0.686</td><td>0.961 0.957 0.957 0.918</td><td>0.436 0.432 0.426 0.418</td><td>0.491 0.482 0.480 0.417</td><td>0.752 0.735 0.757 0.683</td><td>0.348 0.355 0.363 0.337</td><td>0.456 0.441 0.430 0.462</td><td>0.510 0.500 0.430 0.380</td><td>0.491 0.462 0.373 0.295</td><td>0.438 0.406 0.344</td><td>0.819 0.782 0.622 0.624</td><td>0.621 0.607 0.577 0.537</td></tr><tr><td>Base</td><td>100% 87.50% 87.50%</td><td>112 96 96 64</td><td>128 112 128 96</td><td>13.397 13.818 13.561</td><td>0.659 0.649 0.656</td><td>0.757 0.706 0.756 0.703 0.754 0.705</td><td></td><td>0.680 0.680 0.680</td><td>Llama 3.200 3B Instruct 0.681 0.932 0.657 0.933 0.670 0.935</td><td>0.360 0.374 0.370</td><td></td><td>0.460 0.463</td><td>0.784 0.776</td><td>0.410 0.411</td><td>0.496 0.467</td><td>0.470 0.480</td><td>0.353 0.333</td><td>0.322 0.463 0.442</td><td>0.769 0.761</td><td>0.599 0.592</td></tr><tr><td>PALU</td><td>75% 75% 50% 87.50%</td><td>64 112 96</td><td>128 64 112 128</td><td>14.719 14.395 22.435 14.618</td><td>0.637 0.638 0.543 0.633</td><td>0.742 0.757 0.695 0.745 0.757</td><td>0.689 0.688 0.594 0.701 0.706</td><td>0.670 0.677 0.635 0.676 0.684 0.684</td><td>0.653 0.653 0.591 0.702 0.924</td><td>0.918 0.933 0.904 0.362</td><td>0.382 0.380 0.344 0.464</td><td>0.445 0.452 0.423 0.371</td><td>0.775 0.773 0.757 0.739 0.769</td><td>0.409 0.414 0.417 0.394 0.411</td><td>0.487 0.446 0.478 0.440 0.470</td><td>0.470 0.450 0.440 0.400 0.430</td><td>0.343 0.382 0.333 0.235 0.314</td><td>0.450 0.415 0.424 0.331 0.433</td><td>0.772 0.754 0.749 0.613 0.767</td><td>0.595 0.585 0.583 0.522 0.587</td></tr><tr><td>Eigen Attention</td><td>87.50% 75% 75% 50% Thin Keys 87.50% 75%</td><td>96 64 64 96 64 112</td><td>96 128 64 128 128 84.815</td><td>13.557 16.347 14.552 30.778 56.350</td></table>

Additionally parameter memory is independent of context length. KV-cache memory scales with sequence length and therefore yields increasingly large absolute savings at longer contexts. At 256K tokens, the gap between the base model and the compressed variants becomes substantial, illustrating that the main systems benefit of RACE grows with context length while preserving native reduced-dimensional execution.

Compatibility with Token compression and Quantization Table 5 reports the full per-task results when RACE is combined with token-compression and KV-quantization methods. Across H2O, SnapKV, and KIVI, applying RACE on top of an already compressed KV cache provides additional multiplicative memory reduction, but also introduces further accuracy and perplexity degradation. This effect is generally more pronounced for token-compression methods than for quantization.

For H2O and SnapKV, reducing the head dimension with RACE consistently lowers the total KVcache footprint from 50% to 37.5% and 25%, or from 25% to 18.75% and 12.5%, depending on the token-retention ratio. The corresponding degradation is gradual across most zero-shot tasks, while WikiText perplexity increases more noticeably at aggressive combined compression. For example, with H2O at 50% token retention, adding RACE at 75% and 50% head retention reduces total KVcache usage to 37.5% and 25%, respectively, while average zero-shot accuracy decreases from 0.706 to 0.674 and 0.604. SnapKV follows a similar trend, with average accuracy decreasing from 0.710 to 0.676 and 0.612 under the same settings.

Table 4: Per-task results of Qwen2.5 7B and Llama3.2 3B Instruct on LongBench. We report zero shot longbench score across the evaluated model families and KV-cache retention ratios. The final column reports the average across tasks. For each KV Cache compression level per model, we highlight the best performing average result.
<table><tr><td></td><td>KV Cache Retained</td><td>Narrative QA</td><td>Qasper</td><td>Multifield</td><td>Hotpot QA</td><td>Wikimqa</td><td>Musique</td><td>gov_report</td><td>qmsum</td><td>multi_news</td><td>tree triviaqa</td><td>samsum</td><td>passage _count</td><td>passage. retrieval.en</td><td>lee</td><td>repobench</td><td>Average</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>19.15</td><td>Llama3.2 3B Instruct. 33.91</td><td>23.5</td><td></td><td>71</td><td></td><td></td><td></td><td>75</td><td></td><td>52.5</td></tr><tr><td>Uncompressed</td><td>100% 87.5%</td><td>21.25 24.81</td><td>39.89</td><td>50.06</td><td>47.31</td><td>36.75 36.52</td><td>17.9</td><td>33.95</td><td></td><td>26.64 26.98</td><td>88.41</td><td>42.62</td><td>8.5</td><td></td><td>52.01 51.26</td><td></td><td>43.03 42.05</td></tr><tr><td>RACE</td><td></td><td></td><td>33.42</td><td>49.61</td><td>47.52</td><td></td><td></td><td>22.73</td><td></td><td></td><td>90.47 87.78</td><td>42.91 41.16</td><td>4 3.5</td><td>63.5 58.5</td><td>44.34</td><td>53.9 44.45</td><td></td></tr><tr><td></td><td>75%</td><td>22.2</td><td>30.79</td><td>45.3</td><td>40.44</td><td>35.29</td><td>15.96</td><td>32.88</td><td></td><td>25.77 59.5</td><td>78.45</td><td>35.32</td><td>1.18</td><td></td><td>34.83</td><td>35.91</td><td>38.88 26.51</td></tr><tr><td></td><td>50%</td><td>11.45</td><td>15.08</td><td>25.46</td><td>22.86</td><td>20.01</td><td>15.52</td><td>21.44</td><td>20.5</td><td>20.97</td><td></td><td></td><td></td><td></td><td>5.75</td><td></td><td></td></tr><tr><td>RAP</td><td>87.5% 75%</td><td>23.49</td><td>32.55</td><td>46.14</td><td>41.67</td><td>36.7</td><td>14.63</td><td>30.91</td><td>22.6</td><td>26.31</td><td>87.58</td><td>40.96</td><td></td><td></td><td>45 49.3</td><td>52.24</td><td>38.94</td></tr><tr><td></td><td></td><td>19.49</td><td>27.26</td><td>41.91</td><td>30.35</td><td>31.14</td><td>13.96</td><td>25.55</td><td>21.63</td><td>25.39</td><td>84.46</td><td>39.81</td><td></td><td></td><td>23 47.55</td><td>49.75</td><td>34.33</td></tr><tr><td></td><td>50%</td><td>7.79</td><td>6.42</td><td>19</td><td>6.16</td><td>6.81</td><td>8.77</td><td>9.2</td><td>7.94</td><td>5.48</td><td>40.2</td><td>11.99</td><td></td><td></td><td>5 34.8</td><td>32.67</td><td>14.64</td></tr><tr><td></td><td>87.5%</td><td>21.9</td><td>31.34</td><td>49.22</td><td>45.26</td><td>36.56</td><td>17.96</td><td>31.82</td><td>23.32</td><td>25.58</td><td>89.59</td><td>30.3</td><td>2.79</td><td></td><td>74.5 46.9 56.5</td><td>50.69</td><td>40.30</td></tr><tr><td>PALU</td><td>75%</td><td>21.91</td><td>30.94</td><td>46.85</td><td>40.73</td><td>32.06</td><td>17.78</td><td>30.95</td><td>22.63</td><td>24.6</td><td>88.7</td><td>28.8</td><td>2.21</td><td></td><td>43.62</td><td>38.81</td><td>37.07</td></tr><tr><td></td><td>50%</td><td>20.47</td><td>30.26</td><td>37.77</td><td>33.88</td><td>29.15</td><td>14.18</td><td>24.79</td><td>22.24</td><td>21.8</td><td>79.63</td><td>25.5</td><td></td><td></td><td>9 25.74</td><td>34.71</td><td>28.63</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>Qwen2.5 7B</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Uncompressed</td><td>100%</td><td>17.56</td><td>14.86</td><td>27.7</td><td>29.5</td><td>15.51</td><td>11.51</td><td>30.96 23.07</td><td></td><td>26.75 72.5</td><td>89.31</td><td>45.83</td><td>3.5</td><td>24.5</td><td>63.07</td><td>51.58</td><td>34.23</td></tr><tr><td>RACE</td><td>87.5%</td><td>17.98</td><td>20.87</td><td>41.23</td><td>41.75</td><td>31.3</td><td>18.85</td><td>28 22.99</td><td></td><td>26.7 72.5</td><td>90.25</td><td>44.98</td><td>2.5 3</td><td></td><td>53.35</td><td>44.22</td><td>36.03</td></tr><tr><td></td><td>75%</td><td>17.11</td><td>18.33</td><td>40.39</td><td>41.08</td><td>30.36</td><td>18.76</td><td>25.78</td><td>22.27</td><td>26.25 70.5</td><td>87.13</td><td>44.07</td><td></td><td></td><td>40.51</td><td>46.8</td><td>33.80</td></tr><tr><td></td><td>50%</td><td>4.9</td><td>14.27</td><td>22.66</td><td>19.21</td><td>7.23</td><td>3.84</td><td>18.37</td><td>21.1</td><td>20.89</td><td>84.26</td><td>39.01</td><td>2.8</td><td></td><td>37.1</td><td>38.07</td><td>25.70</td></tr><tr><td></td><td>87.5%</td><td>13.1</td><td>21.46</td><td>27.31</td><td>10.36</td><td>18.43</td><td>14.52</td><td>26.22</td><td>21.15</td><td>25.7 72.5</td><td>89.56</td><td>44.53</td><td></td><td></td><td>44.44</td><td>50.62</td><td>30.74</td></tr><tr><td>RAP</td><td>75%</td><td>5.78</td><td>21.38</td><td>26.8</td><td>13.88</td><td>11.9</td><td>5.26</td><td>23.31</td><td>23.32</td><td>23.75 68.5 52</td><td>86.99</td><td>44.21</td><td></td><td></td><td>44.79</td><td>42.49</td><td>28.12</td></tr><tr><td></td><td>50%</td><td>5.16</td><td>9.4</td><td>15.36</td><td>7.57</td><td>8.48</td><td>4.09</td><td>15.67</td><td>21.49</td><td>18.78</td><td>73.1</td><td>33.92</td><td>1.17</td><td>3</td><td>4.5 2.8 27.8</td><td>42.13</td><td>21.18</td></tr><tr><td></td><td>87.5%</td><td>7.94</td><td>12.25</td><td>29.8</td><td>36.21</td><td>26.83</td><td>18.46</td><td>29.05</td><td>22.63</td><td>26.35</td><td>89.49</td><td>34.25</td><td></td><td>2</td><td>19 49.72</td><td>40.75</td><td>32.33</td></tr><tr><td>PALU</td><td></td><td></td><td>16.48</td><td></td><td></td><td></td><td>18.8</td><td>27.25</td><td>22.58</td><td>72.5</td><td></td><td></td><td></td><td>4</td><td>10 41.39 8.5</td><td></td><td></td></tr><tr><td></td><td>75% 50%</td><td>8.15 4.15</td><td>8.95</td><td>28.76 23.03</td><td>36.94 10.8</td><td>28.71 9.37</td><td>5.83</td><td>21.21</td><td>21.25</td><td>25.16 70.5 22.3 69.5</td><td>89.93 83.81</td><td>34.45 33.82</td><td>1.5</td><td></td><td>31.17</td><td>33.24 31.17</td><td>31.02 24.15</td></tr></table>

Table 5: Per-task results of Qwen2.5 7B on Wikitext (perplexity) and zero-shot accuracy on combining RACE with Token compression and quantization methods.
<table><tr><td>Tokens /Precision Retained</td><td>RACE KV Cache Reatained</td><td>Total KV Cache Retained</td><td>wikitext</td><td>hellaswag</td><td>piqa</td><td>arc_challenge</td><td>boolq</td><td>mmlu_ college_cs</td><td>Average</td></tr><tr><td colspan="10">H20 (Token Compression)</td></tr><tr><td rowspan="4">512 (50%)</td><td>100%</td><td>50%</td><td>12.66</td><td>0.66</td><td>0.85</td><td>0.48</td><td>0.82</td><td>0.72</td><td>0.706</td></tr><tr><td>75%</td><td>37.5%</td><td>15.8</td><td>0.65</td><td>0.83</td><td>0.45</td><td>0.81</td><td>0.63</td><td>0.674</td></tr><tr><td>50%</td><td>25%</td><td>17.44</td><td>0.62</td><td>0.79</td><td>0.44</td><td>0.74</td><td>0.43</td><td>0.604</td></tr><tr><td>100%</td><td>25%</td><td>13.62</td><td>0.66</td><td>0.85</td><td>0.48</td><td>0.79</td><td>0.66</td><td>0.688</td></tr><tr><td rowspan="3">256 (25%)</td><td>75%</td><td>18.75%</td><td>16.12</td><td>0.65</td><td>0.83</td><td>0.45</td><td>0.78</td><td>0.63</td><td>0.668</td></tr><tr><td>50%</td><td>12.5%</td><td>21.81</td><td>0.62</td><td>0.79</td><td>0.42</td><td>0.71</td><td>0.4</td><td>0.588</td></tr><tr><td colspan="11">SnapKV (Token Compression)</td></tr><tr><td rowspan="4">512 (50%)</td><td>100%</td><td>50%</td><td>12.55</td><td>0.68</td><td>0.86</td><td>0.48</td><td>0.82</td><td>0.71</td><td>0.71</td></tr><tr><td>75%</td><td>37.5%</td><td>15.61</td><td>0.65</td><td>0.83</td><td>0.46</td><td>0.81</td><td>0.63</td><td>0.676</td></tr><tr><td>50%</td><td>25%</td><td>18.63</td><td>0.62</td><td>0.79</td><td>0.45</td><td>0.76</td><td>0.44</td><td>0.612</td></tr><tr><td>100%</td><td>25%</td><td>12.76</td><td>0.68</td><td>0.85</td><td>0.48</td><td>0.81</td><td>0.7</td><td>0.704</td></tr><tr><td>256 (25%)</td><td>75% 50%</td><td>18.75% 12.5%</td><td>15.66 18.44</td><td>0.65 0.62</td><td>0.83 0.79</td><td>0.45 0.46</td><td>0.79 0.74</td><td>0.63 0.44</td><td>0.67 0.61</td></tr><tr><td colspan="10"></td></tr><tr><td rowspan="3">2 bits (25%)</td><td>100%</td><td>25%</td><td>17.04</td><td>KIVI (Quantization) 0.69</td><td>0.86</td><td>0.48</td><td>0.78</td><td>0.7</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td>0.84</td><td></td><td>0.76</td><td>0.6</td><td>0.702</td></tr><tr><td>75%</td><td>18.75%</td><td>18.17</td><td>0.66 0.64</td><td>0.8</td><td>0.48 0.44</td><td>0.71</td><td>0.42</td><td>0.668</td></tr><tr><td rowspan="3">4 bits (50%)</td><td>50%</td><td>12.5%</td><td>21.81</td><td></td><td>0.86</td><td>0.48</td><td>0.78</td><td>0.7</td><td>0.602 0.702</td></tr><tr><td>100%</td><td>50%</td><td>15.7 16.58</td><td>0.69</td><td>0.84</td><td>0.48</td><td>0.78</td><td>0.63</td><td></td></tr><tr><td>75% 50%</td><td>37.5% 25%</td><td>17.14</td><td>0.66 0.64</td><td>0.8</td><td>0.44</td><td>0.74</td><td>0.42</td><td>0.678 0.608</td></tr></table>

RACE is comparatively more complementary with KIVI. With 4-bit quantization, adding RACE at 75% and 50% head retention reduces the total KV cache from 50% to 37.5% and 25%, while average accuracy decreases from 0.702 to 0.678 and 0.608. Similar behavior is observed with 2- bit quantization. Overall, the results suggest that RACE composes cleanly with both token and precision compression, but the cost of applying additional head-dimension compression is larger once information has already been removed along the token dimension. In contrast, quantization leaves more redundancy for RACE to exploit, resulting in a milder degradation at comparable total cache budgets.

Ablations We first examine whether the retained RoPE pairs should be selected using information from queries and keys jointly or from either projection alone. Across all retention ratios, the joint Q/K energy criterion gives the strongest overall results. At 87.5% retention, Q-only and K-only selection remain close to the joint criterion, but the gap becomes clearer as compression increases. At 75% retention, joint Q/K energy achieves 0.61 average accuracy with a WikiText perplexity of

Error norm per layer relative to uncompressed QK  
![](images/a802bd42fb72eb2b41cc7adbffb41e37a5c773fed1403e427e7e71c902798d43.jpg)  
Figure 12: Layer-wise normalised approximation error as in Eq.5 with respect to uncompressed model for joint and single-sided energy selectors. Joint Q/K selection yields the lowest average error compared to Q-only or K-only selection.

Table 6: Ablation of selection criteria for RoPE-pair retention. We compare RACE’s joint Q/K activation-energy criterion against magnitude, random selection, gradient Fisher, attention-score selection, and Q-only or K-only energy. Results are reported on WikiText perplexity and zero-shot accuracy across nine tasks at 87.5%, 75%, and 50% KV-cache retention. Joint Q/K energy is consistently strongest overall, with the advantage becoming more pronounced at higher compression.
<table><tr><td>KV Cache Retained</td><td>Selection Criteria</td><td>wikitext</td><td>piqa</td><td>wino</td><td>acr_c</td><td>sciq</td><td>arc_easy</td><td>boolq</td><td>race</td><td>mmlu_cs</td><td>mmlu_phy</td><td>Avg</td></tr><tr><td rowspan="6">87.5%</td><td>Joint QK Energy (RACE)</td><td>13.88</td><td>0.74</td><td>0.67</td><td>0.46</td><td>0.93</td><td>0.70</td><td>0.77</td><td>0.42</td><td>0.48</td><td>0.35</td><td>0.61</td></tr><tr><td>Magnitude</td><td>14.58</td><td>0.74</td><td>0.64</td><td>0.44</td><td>0.89</td><td>0.68</td><td>0.71</td><td>0.41</td><td>0.37</td><td>0.25</td><td>0.57</td></tr><tr><td>Random</td><td>14.95</td><td>0.71</td><td>0.63</td><td>0.45</td><td>0.89</td><td>0.67</td><td>0.65</td><td>0.39</td><td>0.35</td><td>0.25</td><td>0.55</td></tr><tr><td>Gradient Fisher</td><td>14.58</td><td>0.76</td><td>0.66</td><td>0.44</td><td>0.87</td><td>0.69</td><td>0.74</td><td>0.41</td><td>0.38</td><td>0.25</td><td>0.58</td></tr><tr><td>Attention Score</td><td>14.13</td><td>0.74</td><td>0.66</td><td>0.44</td><td>0.92</td><td>0.69</td><td>0.75</td><td>0.40</td><td>0.40</td><td>0.27</td><td>0.59</td></tr><tr><td>Energy Q only</td><td>14.02</td><td>0.75</td><td>0.66</td><td>0.45</td><td>0.93</td><td>0.69</td><td>0.75</td><td>0.42</td><td>0.40</td><td>0.32</td><td>0.60</td></tr><tr><td rowspan="8">75%</td><td>Energy K only</td><td>14.03</td><td>0.76</td><td>0.67</td><td>0.45</td><td>0.93</td><td>0.67</td><td>0.75</td><td>0.43</td><td>0.45</td><td>0.30</td><td>0.60</td></tr><tr><td>Joint QK Energy (RACE)</td><td>14.95</td><td>0.76</td><td>0.66</td><td>0.43</td><td>0.93</td><td>0.70</td><td>0.76</td><td>0.42</td><td>0.48</td><td>0.33</td><td>0.61</td></tr><tr><td>Magnitude</td><td>15.64</td><td>0.73</td><td>0.61</td><td>0.43</td><td>0.88</td><td>0.65</td><td>0.67</td><td>0.39</td><td>0.34</td><td>0.24</td><td>0.55</td></tr><tr><td>Random</td><td>17.74</td><td>0.72</td><td>0.56</td><td>0.39</td><td>0.84</td><td>0.62</td><td>0.60</td><td>0.36</td><td>0.25</td><td>0.21</td><td>0.50</td></tr><tr><td>Gradient Fisher</td><td>14.76</td><td>0.75</td><td>0.60</td><td>0.42</td><td>0.86</td><td>0.67</td><td>0.73</td><td>0.39</td><td>0.40</td><td>0.27</td><td>0.57</td></tr><tr><td>Attention Score</td><td>14.87</td><td>0.74</td><td>0.64</td><td>0.42</td><td>0.91</td><td>0.67</td><td>0.73</td><td>0.40</td><td>0.42</td><td>0.32</td><td>0.58</td></tr><tr><td>Energy Q only</td><td>16.78</td><td>0.74</td><td>0.66</td><td>0.42</td><td>0.91</td><td>0.68</td><td>0.75</td><td>0.41</td><td>0.43</td><td>0.33</td><td>0.59</td></tr><tr><td>Energy K only</td><td>16.17</td><td>0.75</td><td>0.65</td><td>0.44</td><td>0.92</td><td>0.68</td><td>0.74</td><td>0.42</td><td>0.43</td><td>0.33</td><td>0.60</td></tr><tr><td rowspan="6">50%</td><td>Joint QK Energy (RACE)</td><td>16.02</td><td>0.69</td><td>0.59</td><td>0.38</td><td>0.89</td><td>0.62</td><td>0.69</td><td>0.37</td><td>0.34</td><td>0.33</td><td>0.54</td></tr><tr><td>Magnitude</td><td>18.73</td><td>0.71</td><td>0.54</td><td>0.32</td><td>0.77</td><td>0.49</td><td>0.50</td><td>0.31</td><td>0.23</td><td>0.22</td><td>0.45</td></tr><tr><td>Random</td><td>26.09</td><td>0.62</td><td>0.50</td><td>0.30</td><td>0.65</td><td>0.44</td><td>0.47</td><td>0.27</td><td>0.26</td><td>0.22</td><td>0.41</td></tr><tr><td>Gradient Fisher</td><td>17.80</td><td>0.72</td><td>0.55</td><td>0.35</td><td>0.80</td><td>0.53</td><td>0.44</td><td>0.33</td><td>0.30</td><td>0.27</td><td>0.48</td></tr><tr><td>Attention Score</td><td>17.86</td><td>0.70</td><td>0.59 0.57</td><td>0.33</td><td>0.73</td><td>0.49</td><td>0.50</td><td>0.35</td><td>0.33</td><td>0.31</td><td>0.48</td></tr><tr><td>Energy Q only</td><td>21.01</td><td>0.72</td><td></td><td>0.35</td><td>0.86</td><td>0.55</td><td>0.66</td><td>0.37</td><td>0.30</td><td>0.33</td><td>0.52</td></tr><tr><td></td><td>Energy K only</td><td>20.32</td><td>0.71</td><td>0.57</td><td>0.35</td><td>0.87</td><td>0.54</td><td>0.66</td><td>0.38</td><td>0.30</td><td>0.32</td><td>0.52</td></tr></table>

14.95, compared with 0.59/16.78 for Q-only and 0.60/16.17 for K-only. At 50% retention, the joint criterion retains 0.54 average accuracy and 16.02 perplexity, while both Q-only and K-only fall to 0.52 average accuracy with substantially higher perplexity.

This trend is also reflected directly in the approximation of the uncompressed attention interaction. Figure 12 shows the layer-wise error norm of the compressed QK<sup>⊤</sup> product relative to the uncompressed model. Joint Q/K selection gives the lowest average error, 0.105, compared with 0.112 for K-only and 0.116 for Q-only, and the reduction is observed across most layers rather than being concentrated in a small subset. Together, the downstream results and the layer-wise approximation error support deriving the shared selector from both Q and K rather than from either side in isolation.

We next compare different scoring criteria that all operate on joint Q/K information. Joint activation energy consistently outperforms magnitude, random selection, gradient-Fisher scoring, and attention-based scoring, with the differences becoming larger at stronger compression. At 50% retention, RACE achieves 0.54 average accuracy, compared with 0.48 for gradient Fisher and attention score, 0.45 for magnitude, and 0.41 for random selection. The same trend is reflected in WikiText perplexity, where joint Q/K energy also achieves the lowest value. Notably, the gradientand attention-based alternatives do not recover the performance of the simple forward-pass energy criterion despite requiring more complex statistics. Overall, these ablations indicate that both components of the selector are important: the ranking should be derived jointly from Q and K, and joint activation energy provides an effective criterion for identifying the RoPE pairs that best preserve the original attention interaction.