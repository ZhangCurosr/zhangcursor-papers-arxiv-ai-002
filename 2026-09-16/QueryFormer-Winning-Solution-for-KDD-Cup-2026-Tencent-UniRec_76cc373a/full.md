# QueryFormer: Winning Solution for KDD Cup 2026 Tencent UniRec Challenge

Yuanzhe Zhou Wuhan University

## Abstract

Post-click conversion rate (pCVR) prediction requires jointly mod eling feature interactions and sequential user behaviors. The KDD Cup 2026 Tencent UniRec Challenge calls for a unified architecture addressing both. We observe that existing unified architectures often generate query tokens—the central information hub— with projection-based multi-layer perceptrons (MLPs), without explicit token-to-query attention for refining the query side. We propose QueryFormer, centered on a stackable unified field– sequence block that bridges non-sequential multi-field features and behavioral sequences, and provide a latency-aware scaling study over view width �, model width, depth, data, and compute. The block generates queries through cross-attention and packs sequence queries into shared-parameter attention. QueryFormer secured 1st place in the Industrial Track, achieving an oficial test area under the ROC curve (AUC) of 0.83254; a modest postcompetition scale-up reached 0.832713. Within our grid, �-scaling improves validation AUC from 0.84540 to 0.84615 and beats Hy-Former at comparable budgets. Ablation identifies query generation as the largest contributor. Packed shared-parameter cross-attention keeps �=8 inference latency to only 1.89× that of �=1, positioning the bridge as an eficient stackable unified block.

## 1 Introduction

The KDD Cup 2026 Tencent UniRec Challenge requires post-click conversion rate (pCVR) models that jointly handle multi-field features and user behavior sequences. Our solution, QueryFormer, won the Industrial Track. The key observation is that existing unified architectures often generate query tokens with projectionbased multi-layer perceptrons (MLPs), which provide nonlinear context compression but lack explicit token-to-query attention. Since the field-to-sequence cross-attention channel ⟨�, �⟩ (nonsequential field tokens attending to behavioral sequence tokens) is dominant [16], query quality directly determines what sequence evidence is retrieved and how user-item context is aggregated.

QueryFormer redesigns query processing as a stronger querygeneration mechanism. It combines token-level self/cross attention, item-to-sequence retrieval, and cross-attention-based query generation (Figure 1). Its reusable unit is a stackable unified field–sequence block: field and behavior tokens enter a token-in/token-out bridge, generated query probes retrieve sequence evidence, and packed shared-parameter attention keeps multi-view execution eficient. Ablations show that replacing SeqQueryCrossAttn with an MLP causes the largest degradation, and scaling the embedding matrix width � gives consistent gains. Figure 2 previews the scaling results against HyFormer.

Our key contributions are:

Zhaoyang Zeng Sun Yat-sen University

(1) Query Generation Mechanism: cross-attention-driven query generation adds explicit token-to-query attention beyond projection-based MLP generation.

(2) Unified Block: a token-in/token-out field–sequence bridge composes with token-based recommenders and uses packed shared-parameter attention.

(3) Latency-Aware Scaling Study: � = 4 independent tokenization columns form a shape- $\left( B _ { s } , H , T , d _ { \mathrm { m o d e l } } \right)$ embedding matrix, enabling batch-parallel scaling across views, depth, width, data, and compute.

(4) Industrial Validation: 1st place in the KDD Cup 2026 Industrial Track (test AUC 0.83254) with consistent scaling trends across multiple dimensions.

## 2 Related Work

## 2.1 Feature Interaction and Sequence Modeling

DeepFM [5], DCN-V2 [22], Fi-GNN [15], and DLRM [17] model static feature interactions through product, cross-network, graph, or embedding-interaction layers. Sequential recommenders such as DIN [27], DIEN [26], SIM [18], BERT4Rec [19], SASRec [11], and HSTU [24] extract interest from behavior histories. These lines are usually optimized as separate modules; QueryFormer instead lets field tokens and behavior evidence interact through a unified query pipeline.

## 2.2 Token-Based Architectures

Recent token-based models [3, 8, 25, 29] represent heterogeneous recommendation features as tokens processed by Transformer-style blocks. EST [16] shows that ⟨�, �⟩ cross-attention is the dominant interaction channel, but it does not optimize how the query side is generated. HyFormer and OneTrans reuse MLP-generated queries; LENS, GAP-Net, and HeMix calibrate MLP queries with gates, cascades, or mixed static/dynamic components. QueryFormer takes a stronger step: it generates query probes through cross-attention and repeatedly refines them through a query-centric pipeline (Table 1). In Table 1, Bridge describes how explicitly a method connects nonsequential field tokens with behavioral sequence evidence during query generation.

Table 1: Unified-block positioning. Tok→Q: explicit tokento-query attention; token I/O: token input/output; Packed: shared-parameter packed attention for multi-view queries.
<table><tr><td>Method</td><td>Query</td><td>Tok→Q</td><td>Bridge</td><td>Views</td><td>Packed</td></tr><tr><td>HyFormer [8]</td><td>MLP</td><td>×</td><td>implicit</td><td>X</td><td>X</td></tr><tr><td>HeMix [21]</td><td>Mixed</td><td>X</td><td>mixed</td><td>X</td><td>X</td></tr><tr><td>LENS [23]</td><td>Gate</td><td>×</td><td>gated</td><td>X</td><td>X</td></tr><tr><td>GAP-Net [12]</td><td>Cascade</td><td>X</td><td>cascade</td><td>X</td><td>X</td></tr><tr><td>QueryFormer</td><td>Cross-Attn</td><td>√</td><td>token I/O</td><td>H</td><td>√</td></tr></table>

![](images/c6ff32825d58b1d4cbee05a04b05b790d05df670b0f9e913c5f536de2be49610.jpg)  
Figure 1: QueryFormer architecture. The reusable stackable query-generation bridge takes non-sequential field tokens and behavioral sequence tokens as input, returns generated query probes and mixed representations, and preserves the � dimension as parallel tokenization columns with shape- $\left( B _ { s } , H , T , d _ { \bf m o d e l } \right)$ views, where $B _ { s }$ is batch size and � is token count, before percolumn outputs are combined.

![](images/82e51c17ff03c01bf83ffd0c357ee9f1b938a9c5ae4ce9b51d98294dcc429e24.jpg)

![](images/f39708a242435a32704573513bc5a517aca927f82d928f0e0bd0e44eed19f64a.jpg)  
Figure 2: Scaling evidence for the stackable query-generation bridge. (a) Parameter scaling: QueryFormer traces a stronger AUC–dense-parameter frontier than HyFormer [8] within the explored budgets, with separate $d _ { \mathbf { m o d e l } } .$ - and �-scaling paths. (b) Data eficiency: QueryFormer $( P _ { \mathrm { d e n s e } } { = } 8 7 . 1 M , H { = } 4 )$ remains above HyFormer $( P _ { \mathrm { d e n s e } } { = } 1 0 5 . 8 \mathrm { M } )$ from about 20% to full training data. Both x-axes are logarithmic; $P _ { \mathrm { d e n s e } }$ denotes dense parameter count.

Recent scaling studies in recommendation [14] motivate treating capacity allocation as a first-class design axis. For optimization, we combine Muon-style dense updates [1, 10, 28], Adagrad for sparse embeddings [4], and exponential moving average (EMA) weight averaging [9].

## 3 Preliminaries

Unified token-based architectures embed heterogeneous features into a shared $d _ { \mathrm { m o d e l } }$ -dimensional space. We denote non-sequential (NS) user tokens as $N _ { u } ,$ item tokens as $N _ { i } ,$ their union as $N =$ $N _ { u } \cup N _ { i }$ , and behavioral sequence tokens from � domains as $B =$ $\{ B _ { 1 } , . . . , B _ { S } \}$ . For two token sets � and $Y , \langle X , Y \rangle$ denotes a crossattention interaction using � as queries and � as keys/values. We use this notation below to distinguish item-to-sequence retrieval from behavior-initialized query generation.

Existing architectures often generate queries through projection based MLPs:

$$
\mathbf { q } = \mathrm { M L P } \left( \mathrm { p o o l } ( B ) \oplus \mathbf { N } \right)\tag{1}
$$

where ⊕ denotes concatenation. Such MLPs are nonlinear and can fuse pooled context, but they do not explicitly let query tokens attend to field or sequence tokens. Recent methods calibrate query quality via gating [23] or cascading [12]; QueryFormer instead generates queries through context-aware attention.

## 4 Methodology

## 4.1 Overview

QueryFormer processes user/item sparse features, dense features, four behavior domains, and temporal context through a unified token pipeline. Its two tiers are Query-Driven Architecture (Section 4.2), where three attention stages build high-quality query representations, and Supporting Components (Section 4.3), which provide tokenization and interaction infrastructure.

## 4.2 Query-Driven Architecture

Design Principles.

(1) Generate through attention, not projection alone. Cross-attention over NS tokens yields data-dependent query probes beyond projection-only summaries.

(2) Refine through interaction. Queries are enriched via self-attention and cross-attention before they retrieve behavioral evidence.

(3) Scale through diversity. The embedding matrix provides � independent tokenization views, improving the explored scaling frontier.

4.2.1 Query-Generation Pipeline. The query-generation pipeline updates the NS context and then constructs final query probes:

$$
N ^ { \prime } = \mathrm { F e a t u r e I n t e r a c t } ( N _ { u } , N _ { i } ) ,\tag{2}
$$

$$
R = { \mathrm { S e q R e t r i e v e } } ( N _ { i } ^ { \prime } , B ) ,\tag{3}
$$

$$
Q = \mathrm { Q u e r y } \mathrm { G e n e r a t e } ( \mathrm { p o o l } ( B ) , N ^ { \prime } \cup R ) ,\tag{4}
$$

$$
Y = \operatorname { M i x } ( Q , N ^ { \prime } , B ) .\tag{5}
$$

Here $N _ { i } ^ { \prime }$ denotes the item-token slice of $N ^ { \prime }$ . This pipeline exposes the reusable bridge component of QueryFormer: NS tokens first retrieve sequence evidence, then generated query tokens re-query the refined NS context. The bridge targets the field–sequence interaction point of token-based recommenders: it takes field tokens and behavior tokens as input, and returns generated query probes plus mixed representations for downstream towers. Because its interface is token-in/token-out, the bridge is architecturally stackable with existing unified recommenders rather than tied to a single backbone. It is also computationally stackable: sequence queries are concatenated and evaluated by shared-parameter cross-attention, amortizing key/value projections and graphics processing unit (GPU) kernel launches instead of invoking cross-attention serially. Table 2 gives the operator-to-mechanism mapping.

Table 2 lists the four attention mechanisms organized into three architectural stages. Together they form a query-centric refinement pipeline: queries are enriched via self-attention, cross-interact with counterpart tokens, attend to behavioral sequences, and drive query generation through explicit token-to-query attention.

Table 2: Attention mechanisms by architectural stage. For SeqQueryCrossAttn, $Q _ { B }$ denotes behavior-initialized query probes.
<table><tr><td>Stage</td><td>Mechanism</td><td>Type</td><td>Interaction</td></tr><tr><td>Co-Transformer</td><td>QuerySelfAttn QueryCrossAttn</td><td>SelfAttn CrossAttn</td><td> $\langle N _ { u } , N _ { u } \rangle , \langle N _ { i } , N _ { i } \rangle$   $\langle N _ { u } , N _ { i } \rangle$ </td></tr><tr><td>QuerySeqCrossAttn</td><td>一</td><td>CrossAttn</td><td></td></tr><tr><td>SeqQueryCrossAttn</td><td></td><td>CrossAttn</td><td> $\langle N _ { i } , B \rangle$   $\langle Q _ { B } , \dot { N } \rangle$ </td></tr></table>

![](images/0c0b0ac740759d5e9094e06460ceed3917d4398f70c6a48151a3da6fe2a261e3.jpg)

![](images/ff18abcdf1d21cd06869b4c11826b09fe73eb3e36cad9338e468fb809d3b94d2.jpg)  
Figure 3: Validation AUC and LogLoss vs. training steps for $H \in \{ 1 , 2 , 4 , 8 \}$ at $d _ { \mathrm { m o d e l } } { = } 2 7 2$

4.2.2 Embedding Matrix. Instead of a flat token sequence, Query-Former uses an embedding matrix with � independent columns. Each column tokenizes the same input through its own pipeline, producing a tensor of shape $\left( B _ { s } , H , T , d _ { \mathrm { m o d e l } } \right)$ , where $B _ { s }$ is batch size and � is token count. This implements view diversity in one batch-parallel forward pass and exposes a direct scaling axis. Figure 3 shows that increasing � from 1 to 8 consistently improves validation AUC and LogLoss at fixed $d _ { \mathrm { m o d e l } } = 2 7 2$

4.2.3 Co-Transformer. The Co-Transformer contains two layers. Each layer first applies QuerySelfAttn within user and item token sets, then QueryCrossAttn between user and item tokens with gated fusion. This separates within-field refinement from cross-side interaction and lets us ablate the two efects independently.

4.2.4 Retrieval. QuerySeqCrossAttn retrieves behavior evidence by using aggregated item NS tokens as queries over each of the four behavior domains. The retrieved vectors summarize item-relevant historical evidence and are appended to the NS token set.

4.2.5 SeqQueryCrossAtn. SeqQueryCrossAttn generates $N _ { q } = 2$ query probes per sequence domain. Mean/max-pooled behavior summaries initialize probes, and cross-attention over refined NS tokens makes them context-aware:

$$
\begin{array} { r } { \ P _ { d , k } = \mathrm { C r o s s A t t n } _ { d } ( \ P _ { d , k } ^ { ( 0 ) } , { \bf N } , { \bf N } ) . } \end{array}\tag{6}
$$

## 4.3 Supporting Components

4.3.1 NS Token Construction. NS tokens are built from user/item integer features, dense feature slices, temporal features, item ID, and four retrieved sequence tokens, yielding 26 NS tokens.

4.3.2 Dense Feature Fusion. User and item dense features are processed by DenseFusionModel, which combines a low-rank DCN-V2 layer $( L = 1 ,$ , rank $r = 8 )$ , a SiLU-activated MLP, and Squeeze-and-Excitation (SENet) recalibration [7].

4.3.3 Sequence Token Embedding. Each behavior domain embeds side-information independently, projects it to $d _ { \mathrm { m o d e l } }$ , and adds a 65-bucket recency embedding.

4.3.4 Blocks. Following HyFormer and OneTrans [8, 25], $K = 2$ lightweight processing blocks evolve behavior sequences with Swish-Gated Linear Unit (SwiGLU) encoders, attend from queries to sequences, and mix tokens through parameter-free permutation.

## 4.4 Training Recipe

4.4.1 Multi-Loss Supervision. The training objective combines binary cross-entropy (BCE) with an auxiliary mean absolute error (MAE) loss on log-transformed conversion time $( \lambda = 0 . 1 )$ . Removing it reduces validation AUC from 0.84606 to 0.84590, a smaller efect than any attention ablation.

4.4.2 Optimization. We use MuonPlus for dense parameters, Adagrad for sparse embeddings, CosineAnnealingLR, gradient clipping at ℓ -norm 1.0, EMA decay 0.999, bfloat16, and torch.compile.

## 5 Experiments

## 5.1 Experimental Setup

Input data is stored in Apache Parquet format with a 90/10 Row-Group-level train/validation split across 142 feature columns and four behavioral domains (a, b, c, d). All experiments use a fixed sequence limit $L _ { \mathrm { s e q } } = 2 5 6 \colon$ sequences are padded to 256 positions and longer histories are truncated to the most recent positions. Labels include conversion type and log-transformed conversion timestamp.

We evaluate on the Round 2 Industrial Track dataset using area under the ROC curve (AUC) as the primary metric, coupled with LogLoss (logarithmic loss) and inference latency constraints. In this leaderboard setting, AUC diferences around 10<sup>−4</sup> were rankingrelevant; the top two oficial submissions difer by 0.00037. Query-Former targets the competition’s innovation criteria with a stackable field–sequence bridge and a latency-aware scaling study over $H , K , d _ { \mathrm { m o d e l } } ,$ training length, data, and compute cost. Evidence is organized as follows: Table 3 compares the optimized recipe with HyFormer, Table 4 isolates the bridge mechanisms, Table 6 measures packed-attention latency, and Table 7/Figure 2 summarize multi-axis scaling.

Validation AUC is consistently higher than test AUC (best val 0.84631 vs. test 0.83254), reflecting the expected distribution shift between temporally proximate train/val splits and the held-out test period. The relative ordering was preserved for submitted architectural and scaling changes, so we report validation AUC for most experiments due to limited test submission slots. Models are trained on NVIDIA GPUs with multi-GPU distributed data parallel (DDP) via torchrun. The PyTorch implementation uses a custom IterableDataset with pre-allocated NumPy bufers, row-groupbalanced DDP sharding, EMA-shadow validation, and sidecar configs for checkpoint recovery. For final submission, training on the full 35M samples (no validation holdout) yields ∼+0.0001 AUC, and ensemble distillation with multiple variants supervising a single student [2, 13, 20, 30] yields ∼+0.0007 test AUC at single-model inference cost. Due to the competition timeline, we did not exhaustively sweep all high-capacity configurations; the post-competition result suggests additional gains remain available.

Table 4: Ablation of attention mechanisms $( H { = } 4 , d _ { \mathbf { m o d e l } } { = } 2 7 2 )$
<table><tr><td>Configuration</td><td>Best AUC</td><td>Δ</td></tr><tr><td>Full Model (baseline)</td><td>0.84606</td><td></td></tr><tr><td>- Replace SeqQueryCrossAttn with MLP</td><td>0.84568</td><td>-0.00038</td></tr><tr><td>- Remove QueryCrossAttn</td><td>0.84577</td><td>-0.00029</td></tr><tr><td>- Remove QuerySeqCrossAttn</td><td>0.84583</td><td>-0.00023</td></tr><tr><td>- Remove QuerySelfAttn</td><td>0.84586</td><td>-0.00020</td></tr><tr><td>- Remove all four attention mechanisms</td><td>0.84496</td><td>-0.00110</td></tr><tr><td></td><td>All runs: K=2, 4 epochs.</td><td></td></tr></table>

![](images/0a35e0a2e26adeae135cb2689864acf00014339cc81d9aa2f6223150c1849796.jpg)

![](images/f1f60364431b2d8372d7ae410fd1c654930a1491272a3a2c861a2d3dfbdb97c4.jpg)

![](images/8e4f1160b210138be80fe407ba34a01b054941bdaabb9e1dbcf149b2ec18064a.jpg)

![](images/43bafb752412a9fb95e592a678dd0ecc1514a72fbfd39b262c0f3a70886467d4.jpg)  
Figure 4: Per-epoch ablation of four attention mechanisms.

Table 3: Validation comparison under the optimized training recipe. Dense parameters are in millions; all runs use $L _ { \mathrm { s e q } } { = } 2 5 6$
<table><tr><td>Model</td><td>H</td><td>Dense</td><td>Val AUC</td><td>LogLoss</td></tr><tr><td>HyFormer [8]</td><td>一</td><td>105.8</td><td>0.84477</td><td></td></tr><tr><td>QueryFormer</td><td>4</td><td>87.1</td><td>0.84606</td><td>0.210309</td></tr><tr><td>QueryFormer</td><td>4</td><td>220.8</td><td>0.84631</td><td>0.210241</td></tr></table>

## 5.2 Ablation Study

Table 4 reports ablation results; Figure 4 shows per-epoch trends. The degradation order—cross-attention query generation causes the largest drop, followed by cross-attention, sequence attention, and self-attention—confirms that query quality is the primary performance driver. Removing all four mechanisms compounds the loss (Δ = −0.00110), validating the additive benefit of each component.

## 5.3 Competition Results

Table 5 shows the final leaderboard of the Industrial Track. Query-Former ranks 1st with a test AUC of 0.83254. Post-competition, a

modest parameter scale-up improved test AUC to 0.832713; due to limited post-competition time, this configuration was not exhaustively tuned and likely remains below the best attainable setting.

Table 5: Industrial Track leaderboard (top 3), with post competition improvement.
<table><tr><td>Rank</td><td>Team</td><td>Test AUC</td><td>Δ</td></tr><tr><td>1st</td><td>QueryFormer(日之光面)</td><td>0.83254</td><td></td></tr><tr><td>2nd</td><td>RClaw</td><td>0.83217</td><td>-0.00037</td></tr><tr><td>3rd</td><td>load_state_dict</td><td>0.83145</td><td>-0.00109</td></tr><tr><td>一</td><td>QF-scaled-post</td><td>0.832713</td><td>+0.000173</td></tr></table>

## 5.4 Scaling Analysis

Model capacity was a binding constraint in this competition: the 35M-sample dataset demanded representations beyond a singlecolumn architecture. The scaling study asks which stackable axis gives the best AUC-latency trade-of: view width �, block depth �, model width $d _ { \mathrm { m o d e l } }$ , or training compute. We report packedattention latency separately because Round 2 imposed tight infer ence limits.

Table 6: Latency-aware � scaling of the packed bridge on 31.35M training samples and 3.48M validation samples.
<table><tr><td>H</td><td>Train Time</td><td>Eval Time</td><td>Eval Throughput</td><td>Latency / sample</td></tr><tr><td>1</td><td>43.7 min</td><td>1.8 min</td><td>32.2k samples/s</td><td>31.1 µs</td></tr><tr><td>2</td><td>56.0 min</td><td>2.0 min</td><td>29.0k samples/s</td><td>34.5 µs</td></tr><tr><td>4</td><td>80.1 min</td><td>2.5 min</td><td>23.2k samples/s</td><td>43.1 µs</td></tr><tr><td>8</td><td>137.3 min</td><td>3.4 min</td><td>17.0k samples/s</td><td>58.7 µs</td></tr></table>

Table 7: Multi-axis scaling grid (SwiGLU encoders; params in millions). Unspecified values use �=2, $H { = } 4 , d _ { \mathbf { m o d e l } } { = } 2 7 2 .$ , and 4 epochs.
<table><tr><td>Setting</td><td>Sparse</td><td>Dense</td><td>Val AUC</td><td>LogLoss</td></tr><tr><td>H=1</td><td>530</td><td>50</td><td>0.84540</td><td>0.210651</td></tr><tr><td>H=2</td><td>563</td><td>62</td><td>0.84575</td><td>0.210432</td></tr><tr><td>H=4</td><td>628</td><td>87</td><td>0.84606</td><td>0.210309</td></tr><tr><td>H=8</td><td>757</td><td>137</td><td>0.84615</td><td>0.210225</td></tr><tr><td>K=1</td><td>628</td><td>77</td><td>0.84598</td><td>0.210302</td></tr><tr><td>K=3</td><td>628</td><td>97</td><td>0.84616</td><td>0.210270</td></tr><tr><td>11 epochs</td><td>628</td><td>87</td><td>0.84616</td><td>0.210712</td></tr><tr><td>d=68</td><td>557</td><td>10</td><td>0.84462</td><td>0.210942</td></tr><tr><td>d=136</td><td>581</td><td>27</td><td>0.84552</td><td>0.210540</td></tr><tr><td>d=204</td><td>604</td><td>53</td><td>0.84593</td><td>0.210360</td></tr><tr><td>d=544</td><td>722</td><td>221</td><td>0.84631</td><td>0.210241</td></tr></table>

Table 7 presents a scaling analysis across four dimensions within our explored grid. First, increasing � from 1 to 8 monotonically improves AUC and LogLoss, making view diversity the strongest observed scaling direction. Under Round-2’s tight latency constraint, Table 6 identifies �=4 as the practical operating point for a GPUeficient unified architecture: it gains +0.00066 AUC over �=1 with validation time increasing from 1.8 to 2.5 minutes. The packed shared-attention implementation keeps latency growth sublinear: �=8 is only 1.89× slower than �=1 (58.7 vs. 31.1 �s/sample), rather than approaching an 8× serial cost. Second, increasing the block count from �=1 to �=3 gives a smaller gain at fixed �=4 and $d _ { \mathrm { m o d e l } } { = } 2 7 2$ , suggesting that depth-only stacking is not the dominant path in our grid. Third, extending training from 4 to 11 epochs gives only +0.00010 AUC and worse LogLoss, suggesting compute saturation. Fourth, increasing $d _ { \mathrm { m o d e l } }$ improves AUC but with diminishing returns as dense parameters grow quadratically. Together, these results support multi-axis stackability: the bridge scales across � views, depth, width, data/compute, and packed execution, rather than only repeated layers.

5.4.1 Architectural Scaling Eficiency. Figure 2 compares Query-Former and HyFormer [8] under the optimized recipe (MuonPlus, EMA, Adagrad, SwiGLU). QueryFormer achieves higher AUC at comparable dense-parameter budgets, while data scaling shows the same trend from about 20% to full data. Fitting $\Delta \mathrm { A U C } = C \cdot P ^ { k }$ following UniMixer [6], where � denotes dense parameter count and QF/HF abbreviate QueryFormer/HyFormer, gives:

$$
\Delta A \mathrm { U C } _ { \mathrm { Q F } , d } = 3 . 8 2 \times 1 0 ^ { - 4 } \cdot P ^ { 0 . 2 8 6 } ( R ^ { 2 } { = } 0 . 8 9 6 ) ,\tag{7}
$$

$$
\Delta \mathrm { A U C } _ { \mathrm { Q F } , H } = 8 . 7 { \times } 1 0 ^ { - 6 } \cdot P ^ { 0 . 9 2 4 } ~ ( R ^ { 2 } { = } 0 . 8 1 1 ) ,\tag{8}
$$

$$
\Delta \mathrm { A U C _ { H F } } = 1 . 9 { \times } 1 0 ^ { - 4 } \cdot P ^ { 0 . 3 2 6 } .\tag{9}
$$

Within the explored range, �-scaling is close to linear while $d _ { \mathrm { m o d e l } } -$ scaling saturates. We view these fits as empirical scaling diagnostics rather than universal laws. They suggest a practical rule for this dataset: allocate budget to view diversity before representation depth; broader high-capacity sweeps remain future work.

## 6 Conclusion

QueryFormer improves unified recommendation by generating query tokens through explicit token-to-query attention rather than projection-only MLP generation. As a unified block, its reusable contribution is a token-in/token-out field–sequence bridge with efficient packed attention. As a scaling study, it identifies view width � as a strong axis alongside depth, width, data, and compute. The evidence is consistent across three paths: cross-attention query generation is the strongest ablated component, � exposes a stronger scaling axis than standard width in our grid, and packed execution keeps �=8 latency to only 1.89× that of �=1. QueryFormer won the KDD Cup 2026 Industrial Track, with post-competition results indicating further headroom under larger sweeps. Limitations include reliance on one large-scale dataset and the latency cost of wider embedding matrices.

## References

[1] Devin, Andrey Ogurtsov, and Yuanzhe Zhou. 2025. CMI—Detect Behavior with Sensor Data, 1st Place Solution. Kaggle competition solution. Accessed: 2026-07- 05.

[2] Haoran Ding, Wenlin Zhao, Yuchen Jiang, Juren Li, Jie Zhu, Xinchun Li, Yishujie Zhao, Yi Zhang, Ao Qiao, Jianhui Dong, et al. 2026. Rec-Distill: An Industrial Distillation Pipeline for Large-Scale Recommendation Models. arXiv preprint arXiv:2605.29755 (2026).

[3] Kaize Ding, Albert Jiongqian Liang, Bryan Perozzi, Ting Chen, Ruoxi Wang, Lichan Hong, Ed H Chi, Huan Liu, and Derek Zhiyuan Cheng. 2023. HyperFormer: Learning expressive sparse feature representations via hypergraph transformer. In Proceedings ofthe 46th international ACM SIGIR conference on research and development in information retrieval. 2062–2066.

[4] John Duchi, Elad Hazan, and Yoram Singer. 2011. Adaptive subgradient methods for online learning and stochastic optimization. Journal of machine learning research 12, 7 (2011).

[5] Huifeng Guo, Ruiming Tang, Yunming Ye, Zhenguo Li, and Xiuqiang He. 2017. DeepFM: a factorization-machine based neural network for CTR prediction. arXiv preprint arXiv:1703.04247 (2017).

[6] Mingming Ha, Guanchen Wang, Linxun Chen, Xuan Rao, Yuexin Shi, Tianbao Ma, Zhaojie Liu, Yunqian Fan, Zilong Lu, Yanan Niu, et al. 2026. UniMixer: A Unified Architecture for Scaling Laws in Recommendation Systems. arXiv preprint arXiv:2604.00590 (2026).

[7] Jie Hu, Li Shen, and Gang Sun. 2018. Squeeze-and-excitation networks. In Proceedings of the IEEE conference on computer vision and pattern recognition. 7132–7141.

[8] Yunwen Huang, Shiyong Hong, Xijun Xiao, Jinqiu Jin, Xuanyuan Luo, Zhe Wang, Zheng Chai, Shikang Wu, Yuchao Zheng, and Jingjian Lin. 2026. HyFormer: Revisiting the Roles of Sequence Modeling and Feature Interaction in CTR Prediction. arXiv preprint arXiv:2601.12681 (2026).

[9] Pavel Izmailov, Dmitrii Podoprikhin, Timur Garipov, Dmitry Vetrov, and An drew Gordon Wilson. 2018. Averaging weights leads to wider optima and better generalization. arXiv preprint arXiv:1803.05407 (2018).

[10] Keller Jordan, Yuchen Jin, et al. 2024. Muon: An Optimizer for Hidden Layers in Neural Networks. Blog post. Available at https://kellerjordan.github.io/posts/ muon/.

[11] Wang-Cheng Kang and Julian McAuley. 2018. Self-attentive sequential recommendation. In 2018 IEEE international conference on data mining (ICDM). IEEE, 197–206.

[12] Shenqiang Ke, Jianxiong Wei, and Qingsong Hua. 2026. GAP-Net: Calibrating User Intent via Gated Adaptive Progressive Learning for CTR Prediction. arXiv preprint arXiv:2601.07613 (2026).

[13] Weijiang Lai, Beihong Jin, Jiongyan Zhang, Yiyuan Zheng, Jian Dong, Jia Cheng, Jun Lei, and Xingxing Wang. 2025. Exploring Scaling Laws of CTR Model for Online Performance Improvement. In Proceedings of the Nineteenth ACM Conference on Recommender Systems. 114–123.

[14] Guoming Li, Shangyu Zhang, Junwei Pan, Wentao Ning, Jin Chen, Gengsheng Xue, Chao Zhou, Shudong Huang, Haijie Gu, and Menglin Yang. 2026. Expand More, Shrink Less: Shaping Efective-Rank Dynamics for Dense Scaling in Recommendation. arXiv preprint arXiv:2605.23191 (2026).

[15] Zekun Li, Zeyu Cui, Shu Wu, Xiaoyu Zhang, and Liang Wang. 2019. Fi-gnn: Modeling feature interactions via graph neural networks for ctr prediction. In Proceedings of the 28th ACM international conference on information and knowledge management. 539–548.

[16] Mingyang Liu, Yong Bai, Zhangming Chan, Sishuo Chen, Xiang-Rong Sheng, Han Zhu, Jian Xu, and Xinyang Chen. 2026. EST: Towards Eficient Scaling Laws in Click-Through Rate Prediction via Unified Modeling. arXiv preprint arXiv:2602.10811 (2026).

[17] Maxim Naumov, Dheevatsa Mudigere, Hao-Jun Michael Shi, Jianyu Huang, Narayanan Sundaraman, Jongsoo Park, Xiaodong Wang, Udit Gupta, Carole-Jean Wu, Alisson G Azzolini, et al. 2019. Deep learning recommendation model for personalization and recommendation systems. arXiv preprint arXiv:1906.00091 (2019).

[18] Qi Pi, Guorui Zhou, Yujing Zhang, Zhe Wang, Lejian Ren, Ying Fan, Xiaoqiang Zhu, and Kun Gai. 2020. Search-based user interest modeling with lifelong sequential behavior data for click-through rate prediction. In Proceedings ofthe 29th ACM International Conference on Information & Knowledge Management. 2685–2692.

[19] Fei Sun, Jun Liu, Jian Wu, Changhua Pei, Xiao Lin, Wenwu Ou, and Peng Jiang. 2019. BERT4Rec: Sequential recommendation with bidirectional encoder representations from transformer. In Proceedings of the 28th ACM international conference on information and knowledge management. 1441–1450.

[20] Jiaxi Tang and Ke Wang. 2018. Ranking distillation: Learning compact ranking models with high performance for recommender system. In Proceedings ofthe 24th ACM SIGKDD international conference on knowledge discovery & data mining. 2289–2298.

[21] Fangye Wang, Guowei Yang, Xiaojiang Zhou, Song Yang, and Pengjie Wang. 2026. Query-Mixed Interest Extraction and Heterogeneous Interaction: A Scalable CTR Model for Industrial Recommender Systems. arXiv preprint arXiv:2602.09387 (2026).

[22] Ruoxi Wang, Rakesh Shivanna, Derek Cheng, Sagar Jain, Dong Lin, Lichan Hong, and Ed Chi. 2021. Dcn v2: Improved deep & cross network and practical lessons for web-scale learning to rank systems. In Proceedings of the web conference 2021. 1785–1797.

[23] Yuan Wang, Yue Liu, Jun Zhang, and Jie Jiang. 2026. LENS: A Staged Design for Interaction Granularityin Sequential CTR Prediction. arXiv preprint arXiv:2605.25583 (2026).

[24] Jiaqi Zhai, Lucy Liao, Xing Liu, Yueming Wang, Rui Li, Xuan Cao, Leon Gao, Zhaojie Gong, Fangda Gu, Michael He, et al. 2024. Actions speak louder than words: Trillion-parameter sequential transducers for generative recommendations. arXiv preprint arXiv:2402.17152 (2024)

[25] Zhaoqi Zhang, Haolei Pei, Jun Guo, Tianyu Wang, Yufei Feng, Hui Sun, Shaowei Liu, and Aixin Sun. 2026. Onetrans: Unified feature interaction and sequence modeling with one transformer in industrial recommender. In Proceedings ofthe ACM Web Conference 2026. 8162–8170.

[26] Guorui Zhou, Na Mou, Ying Fan, Qi Pi, Weijie Bian, Chang Zhou, Xiaoqiang Zhu, and Kun Gai. 2019. Deep interest evolution network for click-through rate prediction. In Proceedings ofthe AAAI conference on artificial intelligence, Vol. 33. 5941–5948.

[27] Guorui Zhou, Xiaoqiang Zhu, Chenru Song, Ying Fan, Han Zhu, Xiao Ma, Yanghui Yan, Junqi Jin, Han Li, and Kun Gai. 2018. Deep interest network for click-through rate prediction. In Proceedings ofthe 24th ACM SIGKDD international conference on knowledge discovery & data mining. 1059–1068.

[28] Yuanzhe Zhou. 2026. NFL Big Data Bowl 2026—Prediction, 2nd Place Solution. Kaggle competition solution. Accessed: 2026-07-05.

[29] Yifeng Zhou, Yuehong Hu, Zhixiang Feng, Junwei Pan, Kaihui Wu, Hanyong Li, Shangyu Zhang, Shudong Huang, Zhangbin Zhu, Chengguo Yin, et al. 2026. TokenFormer: Unify the Multi-Field and Sequential Recommendation Worlds. arXiv preprint arXiv:2604.13737 (2026).

[30] Jieming Zhu, Jinyang Liu, Weiqi Li, Jincai Lai, Xiuqiang He, Liang Chen, and Zibin Zheng. 2020. Ensembled CTR prediction via knowledge distillation. In Proceedings ofthe 29th ACM international conference on information & knowledge management. 2941–2958.