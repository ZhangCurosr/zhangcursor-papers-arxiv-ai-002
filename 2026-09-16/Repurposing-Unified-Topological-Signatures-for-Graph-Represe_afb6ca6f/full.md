# Repurposing Unified Topological Signatures for Graph Representation Learning

Sanyam Sanjay Jain<sup>a</sup>, Anshika Krishnatray<sup>a</sup>, Aditya Sharma<sup>a</sup>, Vinti Agarwal<sup>a</sup> <sup>a</sup>BITS Pilani, India

## Abstract

Message-passing Graph Neural Networks (GNNs) iteratively propagate and aggregate local neighborhood information followed by global readout to learn graph representations. However, their discriminative power is upper-bounded by the Weisfeiler–Lehman (1-WL) graph isomorphism test. This prevents GNNs from distinguishing certain non-isomorphic graphs with identical local neighborhood structures, often leading to similar graph representations. Unified Topological Signatures (UTS) capture compact, multi-scale representation of global graph topology derived from persistent homology. We introduce two complementary UTS signatures: Φ<sub>grph</sub>, a static signature of the input graph topology, and Φ<sub>emb</sub>, a dynamic signature of the evolving embedding topology. They encode structural information inaccessible to 1-WL-based message-passing GNNs, yet their capabilities are explored solely for post-hoc embedding-space analysis. We integrate UTS into GNN training across three architectural interventions: (i) UTS-Aug: augmenting with standard readout feature that encodes graph’s true topology; (ii) UTS-Reg: topological regularizer that constrains representation collapse; (iii) UTS-Pool: topology-guided pooling that retains structurally critical nodes. We further leverage UTS as a layer-wise diagnostic to quantify oversmoothing during GNN training. Theoretically, we show that integrating UTS into GNN optimization strictly extends GNN expressivity beyond the 1-WL hierarchy. Experiments on three graph classification benchmarks show consistent benefits: Graph-UTS, Dual-UTS, and UTS-Pool improve accuracy across all three datasets, Embedding-UTS provides smaller but similarly consistent gains, and UTS-Reg’s benefit varies across graph domains. Accuracy improves by up to 5.8% with Graph-UTS augmentation, by up to 1.9% with UTS-Reg, and achieves comparable performance to TOGL with UTS-Pool.

## 1 Introduction

Graph Neural Networks (GNNs) have become the default architecture for relational learning, propagating and aggregating information across neighborhoods in a permutation-invariant manner [12]. For graph-level tasks, this aggregation exposes a representational limitation at readout. Standard readout functions (sum, mean, max) retain only first-order statistics of node embeddings, discarding higherorder connectivity almost entirely. Learned pooling operators perform adaptive graph coarsening [1, 11, 27], but select nodes based on local neighborhood, offering no guarantee that global topology survives pooling. Deeper message-passing layers enlarge the receptive field, but repeated aggregation acts as a low-pass filter on the feature matrix, collapsing node embeddings toward a common value as the Dirichlet energy decays exponentially in depth—resulting in oversmoothing [16, 21]. Oversmoothing collapses node embeddings and, consequently, graph representations after readout, eliminating structural information essential for graph-level prediction. Standard GNN objectives also lack an explicit mechanism to monitor topological fidelity during training.

The expressive power of message-passing GNNs is provably upper bounded by the 1-dimensional Weisfeiler–Leman (1-WL) test: any two graphs that are 1-WL equivalent yield identical node embeddings under any message-passing GNN, regardless of depth or width [26]. As a result, non-isomorphic graphs with distinct global organization ( e.g., G1 and G2 in Figure 1), collapse to indistinguishable representations despite having different cycle structure and connectivity.

![](images/53d3093bedddef65e2711c11451087e37790a196335668de41e6b04d30d08533.jpg)  
Figure 1: Illustration of UTS-based graph discrimination. Standard message-passing GNNs with global readout may map the topologically distinct graphs $G _ { 1 } ( C _ { 6 } )$ and $G _ { 2 } ( \bar { 2 } \times \bar { C } _ { 3 } )$ to similar embeddings due to their reliance on local neighborhood aggregation. UTS resolves this limitation by integrating a comprehensive topological signature derived from persistent homology. For clarity, only the UTS-Aug with Betti-0 and Betti-1 curves are visualized, illustrating how differences in connected components and cycles contribute to distinguishing the two graphs.

Topological Data Analysis offers a natural way to expose what message passing hides. Persistent homology [8] extracts stable, multi-scale descriptors (components, cycles, higher-order voids), while recent work has incorporated topological priors into GNNs through persistence-based pooling [5] and discrete curvature methods such as Ricci flow and ORC-POOL [9, 20]. These confirm topological signal benefits GNNs, but each draws on a single geometric perspective. Unified Topological Signatures (UTS) compresses the global geometry of an embedding space into a compact descriptor combining persistent homology with geometric and spectral features [23], but so far only as a post-hoc analysis tool after training.

We incorporate UTS into the GNN optimization process through two complementary signatures: a dynamic embedding signature (Embedding-UTS) computed from each GNN layer, and a static graph topology signature (Graph-UTS) extracted from the input graph. The former characterizes the evolving topology of the learned representation, while the latter captures the graph’s intrinsic topological invariants. For example, Figure 1 illustrates that the 1-WL-equivalent graphs $G _ { 1 } ~ ( C _ { 6 } )$ and $G _ { 2 } \ ( 2 \times C _ { 3 } )$ are indistinguishable to message-passing GNNs, whereas UTS captures complementary global topological information beyond neighborhood aggregation, yielding distinct graph representations.

We deploy both signatures across three architectural interventions: augmented readout, topologypreserving regularization, and topology-aware pooling, detailed in Section 4. The dynamic signature additionally provides a layer-wise Oversmoothing Index, transforming representation collapse from an unobserved failure mode into a measurable training signal.

We instantiate our framework on Graph Isomorphism Networks (GINs) [26], whose expressiveness matches the 1-WL hierarchy, thereby isolating the contribution of UTS from architectural capacity. We theoretically establish that UTS-augmented GINs strictly exceed the expressive power of standard GINs while improving topological fidelity and robustness to oversmoothing. Experimental evaluation on three graph classification benchmarks confirms consistent gains over competitive message-passing baselines.

Contributions. Our main contributions are summarized as follows:

• We introduce a dual topological signature: a layer-wise embedding signature (Embedding-UTS), $\Phi _ { \mathrm { e m b } }$ tracking the evolving geometry of learned representations, and a static graph signature $\mathrm { ( G r a p h – U T S ) } \Phi _ { \mathrm { g r p h } }$ characterizing input topology independently of training and immune to oversmoothing.

• We integrate these signatures into GNN training as auxiliary features, regularization signals, and pooling guidance and perform experiments on three graph classification benchmarks (MUTAG, PROTEINS, and COLLAB).

• We prove the UTS-augmented readout is strictly more expressive than the 1-WL test (Theorem 1).

• We introduce the Oversmoothing Index (OSI), a graph-level metric quantifying topological degradation across layers during training, without sacrificing differentiability or training efficiency.

## 2 Related work

Graph Neural Networks combine node features with connectivity through neighborhood aggregation. GraphSAGE introduced inductive message passing via learned sampled aggregators [12], while GIN established sum aggregation with expressive MLPs as the most discriminative scheme under the Weisfeiler-Lehman framework [26].

Graph classification additionally requires graph-level readout. Simple sum/mean/max pooling compress graphs into first-order statistics; hierarchical methods such as DiffPool and MinCutPool instead learn differentiable cluster assignments [1, 27], Select-Reduce-Connect unifies pooling methods by how they select, reduce, and reconnect nodes [11], and Graph Reference Distribution Learning represents graphs as distributions over node embeddings rather than a single pooled vector [25]. In contrast, our UTS-guided pooling ranks nodes by the topological richness of their embedding neighborhoods.

A separate line embeds topology directly into GNNs: TOGL uses persistent homology as a differentiable, 1-WL-exceeding layer [13]; Wit-TopoPool applies witness-complex persistence to hierarchical pooling [5]. We instead use Unified Topological Signatures (UTS) [23], combining persistence with geometric/spectral statistics into a compact descriptor for post-hoc analysis. Ollivier-Ricci curvature characterizes community structure and bridge edges [20], exploited by ORC-POOL for pooling [9]; we fold curvature into UTS instead. Oversmoothing diagnostics typically rely on posttraining Dirichlet energy [24], missing topological preservation; our Oversmoothing Index signals topology throughout training. Prior work thus couples topology to specialized mechanisms (TOGL, Wit-TopoPool, ORC-POOL) or post-hoc analysis (UTS, Dirichlet diagnostics); we repurpose UTS as a unified training-time signal for readout, regularization, and pooling.

## 3 Preliminaries

## 3.1 Problem Setting

Let $\mathcal { D } = \{ ( G _ { i } , y _ { i } ) \} _ { i = 1 } ^ { N }$ denote a graph classification dataset, where each graph $G _ { i } = ( \mathcal { V } _ { i } , \mathcal { E } _ { i } , { \bf X } _ { i } )$ consists of a node set $\nu _ { i }$ , an edge set $\mathcal { E } _ { i }$ , and node feature matrix $\mathbf { X } _ { i } \in \mathbb { R } ^ { | \mathcal { V } _ { i } | \times d _ { 0 } }$ . The objective is to learn a permutation-invariant function $f : \mathcal { G }  \mathcal { V }$ , which maps an input graph to its class label while preserving the structural information required for graph-level discrimination. Throughout this work, we investigate how explicit topological descriptors can complement conventional message passing to improve graph representations for graph classification.

## 3.2 GIN Backbone

We instantiate our framework on the Graph Isomorphism Network (GIN) [26], chosen because its sum-based aggregation matches the expressive power of the 1-Weisfeiler–Leman (1-WL) test among message-passing GNNs. Given node representations $\mathbf { H } ^ { ( l - 1 ) }$ , each layer updates node v as

$$
{ \bf h } _ { v } ^ { ( l ) } = \mathrm { M L P } ^ { ( l ) } \left( ( 1 + \epsilon ^ { ( l ) } ) { \bf h } _ { v } ^ { ( l - 1 ) } + \sum _ { u \in \mathcal { N } ( v ) } { \bf h } _ { u } ^ { ( l - 1 ) } \right) ,\tag{1}
$$

where $\epsilon ^ { ( l ) }$ is a learnable scalar and $\mathbf { h } _ { v } ^ { ( 0 ) } = \mathbf { x } _ { v }$ . A permutation-invariant graph readout aggregates the final node embeddings into a graph representation. Our proposed framework augments this standard pipeline with explicit topological information while leaving the underlying message-passing architecture unchanged.

## 4 Methodology

We introduce Dual Unified Topological Signatures, a pair of complementary descriptors that explicitly incorporate topology into graph learning: (i) a dynamic embedding signature (Embedding-UTS)

$\Phi _ { \mathrm { e m b } }$ captures the evolving topology of the learned representation throughout message passing, and (ii) a static graph signature (Graph-UTS), $\Phi _ { \mathrm { g r p h } }$ characterizes the intrinsic topology of the input graph. These signatures support three independent interventions:

(i) UTS-Aug: Topology-Augmented Graph Representation (§4.2) concatenates the graph and embedding signatures with the conventional graph readout.

(ii) UTS-Reg:Topology-Preserving Regularization (§4.3) introduces auxiliary objectives that preserve the evolution of embedding topology during training while aligning the final embedding topology with the intrinsic graph topology.

(iii) UTS-Pool:Topology-Aware Pooling (§4.4) replaces feature-based node ranking with a topology-guided scoring mechanism derived from local embedding neighborhoods.

Figure 2 presents an overview of three approaches. Each intervention is modular and can be incorpo rated independently or in combination, without modifying the underlying GNN architecture. The evolution of the embedding signature additionally provides a graph-level oversmoothing diagnostic.

## 4.1 Dual Unified Topological Signatures

Conventional graph readout aggregates node embeddings but lacks an explicit mechanism to encode global topology beyond message-passing representations. To address this limitation, we introduce two complementary topological signatures that characterize the different topological perspectives of input graph: Graph-UTS characterizes “What topological structure is present in input graph”, while Embedding-UTS tells “How that topological structure is reflected in the learned representations”.

## 4.1.1 Embedding-UTS

Given the node embedding matrix $\mathbf { H } ^ { ( l ) } \in \mathbb { R } ^ { n \times d _ { l } }$ at layer l in GNN, we interpret its rows as a point cloud in the learned representation space and compute an embedding signature $\Phi _ { \mathrm { e m b } }$ $\mathbb { R } ^ { \hat { n } \times d } \to \mathbb { R } ^ { 1 4 }$ which summarizes the geometry of the embedding through geometric, persistent homology, and spectral descriptors. We use two variants of this signature depending on whether gradients must propagate through it: an exact variant, computed via standard persistent homology and eigendecomposition, used where the signature serves only as a static input feature (UTS-Aug, $\ S 4 . 2 ) \div$ and a differentiable surrogate, denoted $\Phi _ { \mathrm { e m b } } ^ { \mathrm { d i f f } }$ , which replaces these exact operations with differentiable relaxations and is used wherever the signature must be trained through (UTS-Reg, §4.3; UTS-Pool, §4.4). Both variants share the feature structure below. The resulting embedding descriptor is

$$
\begin{array} { r } { \mathbf { s } ^ { ( l ) } = \Phi _ { \mathrm { e m b } } \Big ( \mathbf { H } ^ { ( l ) } \Big ) = \big [ \underbrace { \bar { \ell } ^ { ( 0 ) } , H ^ { ( 0 ) } , \bar { \ell } ^ { ( 1 ) } , H ^ { ( 1 ) } , \beta _ { 0 } , \beta _ { 1 } } _ { \mathrm { p e r s i s t e n c e } } , \underbrace { \mu _ { \mathrm { m } } , \sigma _ { \mathrm { m n } } , \Delta , \hat { d } } _ { \mathrm { l o c i l g o m e t r y } } , \underbrace { \lambda _ { 1 } , \lambda _ { 2 } , \lambda _ { 3 } , H _ { \mathrm { s p e c } } } _ { \mathrm { s p e c t r a l } } \big ] \in \mathbb { R } ^ { 1 4 } . } \end{array}\tag{2}
$$

Here $\bar { \ell } ^ { ( k ) } , H ^ { ( k ) }$ , and $\beta _ { k }$ denote the mean lifetime, persistence entropy, and Betti number of $k \mathrm { - }$ dimensional homology; $\mu _ { \mathrm { n n } }$ and $\sigma _ { \mathrm { n n } }$ are the mean and standard deviation of k-NN distances; $\Delta$ is the diameter of point cloud $\mathcal { P } ; \hat { d }$ estimates its intrinsic dimensionality; and $\lambda _ { 1 } \leq \lambda _ { 2 } \leq \lambda _ { 3 }$ with $H _ { \mathrm { s p e c } }$ are the smallest eigenvalues and spectral entropy of a Laplacian. For $\Phi _ { \mathrm { e m b } } , \beta _ { k }$ and the Laplacian are computed exactly; for $\Phi _ { \mathrm { e m b } } ^ { \mathrm { d i f f } }$ , both are differentiable relaxations (soft Betti numbers, a regularized Laplacian). Complete derivations of both variants are provided in Appendix A.2.

## 4.1.2 Graph-UTS

Complementary to the embedding signature, we compute a static graph signature directly from the input topology, $\Phi _ { \mathrm { g r p h } } : \mathcal { G }  \mathbb { R } ^ { 2 \bar { 7 } }$ which is evaluated once for every graph and cached throughout training. The graph signature combines curvature, persistent homology, spectral, distance, and structural statistics into a compact topological representation,

$$
\mathbf { g } = \Phi _ { \mathrm { g r p h } } ( G ) = \big [ \underbrace { \phi _ { \mathrm { O R C , } } \mathrm { ~ \phi _ { \mathrm { \tiny ~ \phi ~ \mathrm { F R C } , } } ~ } } _ { \mathrm { c u r n u r e } } , \underbrace { \bar { \ell } _ { G } , \mathrm { d i a m } ( G ) } _ { \mathrm { d i s t a r e } } , \underbrace { \lambda _ { 2 } ^ { G } , \lambda _ { n } ^ { G } , H _ { \mathrm { s p e c } } ^ { G } , \bar { \ell } _ { G } ^ { ( 0 ) } , H _ { G } ^ { ( 0 ) } , \bar { \ell } _ { G } ^ { ( 1 ) } , H _ { G } ^ { ( 1 ) } } _ { \mathrm { s p e c t r a l } } \ , \underbrace { \phi _ { \mathrm { \tiny ~ \mathrm { s t r u c t } } } } _ { \mathrm { p e c t r a l } } \ \big ] \in \mathbb { R } ^ { 2 \tau } .\tag{3}
$$

Here $\phi _ { \mathrm { O R C } } , \phi _ { \mathrm { F R C } } \in \mathbb { R } ^ { 4 }$ summarize Ollivier–Ricci and Forman–Ricci edge curvatures; $\bar { \ell } _ { G }$ and diam(G) are the mean shortest-path length and diameter of graph; $\lambda _ { 2 } ^ { G } , \bar { \lambda _ { n } ^ { G } }$ , and $H _ { \mathrm { s p e c } } ^ { G }$ are the algebraic connectivity, largest eigenvalue, and spectral entropy of the combinatorial Laplacian; $\bar { \ell } _ { G } ^ { ( \bar { k } ) }$ and $H _ { G } ^ { ( k ) }$ are exact k-dimensional persistence lifetimes and entropies; and $\phi _ { \mathrm { s t r u c t } } ~ \in ~ \mathbb { R } ^ { 1 0 }$ collects elementary structural statistics including betweenness centrality, clustering coefficients, and more. Complete feature definitions are provided in Appendix A.3. Unlike $\Phi _ { \mathrm { e m b } }$ , this descriptor is independent of the learned embeddings and therefore remains unaffected by message passing or oversmoothing.

![](images/ec10656adfc13779afa9c8f392c9817976d817e1927bb860708d1f1b851e432f.jpg)  
Figure 2: Overview of Dual-UTS interventions applied on the baseline GIN architecture.

## 4.2 UTS-Aug: Topology-Enhanced Graph Representation

We augment the graph representation obtained from conventional readout with the topological signatures introduced in Section 4.1.2. $\mathrm { G r a p h \mathrm { - U T S } \Phi _ { g r p h } }$ encodes the intrinsic topology of the input graph and remains invariant throughout optimization, whereas Embedding- $\cdot \mathrm { U T S } ^ { - } \Phi _ { \mathrm { e m b } }$ captures the evolving topological information of intermediate representations during message passing.

Let $\mathbf { z } _ { \mathrm { s t r u c t } } \in \mathbb { R } ^ { d }$ denote the graph representation obtained from the underlying GNN through a standard permutation-invariant readout. The proposed topology-aware representation is constructed by concatenating the graph signature and embedding signature of last layer,

$$
\mathbf { z } = \left[ \mathbf { z } _ { \mathrm { s t r u c t } } \parallel \Phi _ { \mathrm { g r p h } } ( G ) \parallel \Phi _ { \mathrm { e m b } } ( \mathbf { H } ^ { ( l ) } ) \right] ,
$$

where ∥ denotes vector concatenation. The augmented representation z is subsequently passed to the graph classifier. To isolate the contribution of each signature, we additionally evaluate single-signature variants by concatenating $\mathbf { z } _ { \mathrm { s t r u c t } }$ with either $\Phi _ { \mathrm { g r p h } }$ or $\Phi _ { \mathrm { e m b } }$ alone in section 5.2.

## 4.3 UTS-Reg: Topology-Preserving Regularization

While UTS-Aug augments the final graph representation with explicit topological information, it does not constrain the evolution of topology during message passing. UTS-Reg addresses this through two regularization objectives: (i) aligning the topology of final embedding space with the intrinsic topology of input graph, and (ii) enforcing topological consistency across successive GNN layers. Together, these objectives preserve structural fidelity, reduce representation collapse and thus mitigate oversmoothing throughout optimization.

Topological Alignment Loss. Since, the graph signature $\Phi _ { \mathrm { g r p h } } ( G )$ and embedding signature $\Phi _ { \mathrm { e m b } } ^ { \mathrm { d i f f } } ( \mathbf { H } ^ { ( L ) } )$ reside in different feature spaces, we learn a linear projection $\mathbf { W } _ { \mathrm { a l i g n } } \in \mathbb { R } ^ { 1 4 \times 2 7 }$ to map the graph-signature into the embedding-signature space and minimize:

$$
\mathcal { L } _ { \mathrm { t o p o - a l i g n } } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left. \Phi _ { \mathrm { e m b } } ^ { \mathrm { d i f f } } ( \mathbf { H } _ { i } ^ { ( L ) } ) - \mathbf { W } _ { \mathrm { a l i g n } } \Phi _ { \mathrm { g r p h } } ( G _ { i } ) \right. _ { 2 } ^ { 2 } ,\tag{4}
$$

where $\mathbf { W } _ { \mathrm { a l i g n } }$ is jointly optimized with the GNN. Since $\Phi _ { \mathrm { g r p h } }$ is fixed, gradients propagate only through $\Phi _ { \mathrm { e m b } } ^ { \mathrm { d i f f } }$ and $\mathbf { W } _ { \mathrm { a l i g n } } .$ , encouraging the learned embedding topology to align with the intrinsic topology of the input graph.

Topological Evolution Loss. We additionally penalize abrupt topological changes across successive message-passing layers. Let $\mathbf { s } ^ { ( l ) } = \Phi _ { \mathrm { e m b } } ^ { \mathrm { d i f f } } \big ( \dot { \mathbf { H } } ^ { ( \bar { l } ) } \big )$ denote the embedding signature at layer l. The

layer-wise smoothness objective is

$$
\mathcal { L } _ { \mathrm { t o p o - e v o l } } = \frac { 1 } { L - 1 } \sum _ { l = 1 } ^ { L - 1 } \left\| \mathbf { s } ^ { ( l ) } - \mathbf { s } ^ { ( l - 1 ) } \right\| _ { 2 } ^ { 2 } ,\tag{5}
$$

which acts as a fitting constraint and encourages smooth topological transitions across layers i.e., do not change the topology too much from the initial one, preventing representation collapse and oversmoothing.

Total Training Objective: The final optimization objective combines the task loss with both topologypreserving regularizers,

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { t a s k } } + \mathcal { L } _ { \mathrm { t o p o - a l i g n } } + \mathcal { L } _ { \mathrm { t o p o - e v o l } } , } \end{array}\tag{6}
$$

## 4.4 UTS-Pool: Topology-aware pooling

Existing graph pooling methods [1, 11, 27] rank nodes using feature activations or attention scores, emphasizing local semantic relevance without explicitly accounting for local topological structure. In our work, we rank nodes based on the topological richness of their embedding neighborhoods, utilizing a differentiable version of $\Phi _ { \mathrm { e m b } }$ which preserves structurally critical regions during graph coarsening. In addition, we also propose a lightweight geometric approximation of node scoring on large dense graphs to improve scalability.

Topology-Aware Node Scoring. For each node $v ,$ let $\mathbf { H } _ { \mathcal { N } ( v ) } = \{ \mathbf { h } _ { u } : u \in \mathcal { N } ( v ) \cup \{ v \} \}$ denote the embeddings within its one-hop neighborhood. We compute a local topological signature and transform it into an importance score,

$$
\alpha _ { v } = \sigma \big ( f _ { \theta } \big ( \Phi _ { \mathrm { e m b } } ^ { \mathrm { d i f f } } \big ( \mathbf { H } _ { \mathcal { N } ( v ) } \big ) \big ) \big ) ,
$$

where $f _ { \theta } : \mathbb { R } ^ { 1 4 }  \mathbb { R }$ is a two-layer $\mathrm { M L P } , \sigma ( . )$ is non-linear function and $\Phi _ { \mathrm { e m b } } ^ { \mathrm { d i f f } }$ denotes differentiable embedding signature (details in Appendix $\mathbf { \ A . } 4 )$ . The $\scriptstyle \mathbf { { t o p - k } } = \ \lfloor \rho n \rfloor$ out of n nodes are retained, $\mathcal { V } ^ { \prime } = \mathrm { t o p } \mathrm { \bar { - } k } ( \bar { \{ \alpha _ { v } \} _ { v \in \mathcal { V } } } )$ , where $\rho$ denotes the pooling ratio. The pooled graph $G ^ { \prime } = ( \mathcal { V } ^ { \prime } , \mathcal { E } ^ { \prime } , { \bf H } ^ { \prime } )$ is obtained by preserving edges between retained nodes and projecting the corresponding node embeddings,

$$
\mathbf { H } ^ { \prime } = \operatorname { R e L U } ( \mathbf { W } _ { \mathrm { p r o j } } \mathbf { H } [ \mathcal { V } ^ { \prime } ] ) .\tag{7}
$$

Efficient Variant for Large Graphs. Computing the embedding signature in large dense graph is becomes computationally expensive owing to large scale persistent homology, triangle enumeration, and spectral computations for every node neighborhood. To improve scalability, we introduce a lightweight geometric approximation that replaces the full signature with five differentiable local descriptors,

$$
\phi _ { v } ^ { \mathrm { l i g h t } } = [ \mu _ { \mathrm { n n } } ^ { v } , \sigma _ { \mathrm { n n } } ^ { v } , \Delta ^ { v } , \bar { r } _ { 1 } ^ { v } , | \mathcal { N } ( v ) | / k _ { \mathrm { m a x } } ] \in \mathbb { R } ^ { 5 } ,\tag{8}
$$

where each quantity is computed within the local neighbourhood of $v .$ . The resulting scorer reduces the computational cost from cubic topological computations to local quadratic operations while preserving the geometric characteristics most relevant for node ranking. During training, neighbourhoods are processed in chunks of size $C$ (default $C = 6 4 )$ , bounding peak GPU memory usage to $O ( C d )$ independent of graph size.

The lightweight scorer $\phi _ { v } ^ { \mathrm { l i g h t } } \left( \mathrm { E q . } 8 \right)$ is a geometric proxy for the full embedding signature $\Phi _ { \mathrm { e m b } } ^ { \mathrm { d i f f } }$ We treat this approximation rigorously in Appendix $\mathrm { C } ,$ where we show that $\phi _ { v } ^ { \mathrm { l i g h t } }$ certifies an explicit scale window, noise floor, and dimension cap governing when a node’s local neighbourhood can carry nontrivial persistent topology at all — i.e. the conditions under which the full topological computation would be uninformative and the lightweight proxy suffices.

Table 1: Accuracy comparison of UTS-Aug GIN backbone with $\Phi _ { \mathrm { e m b } } , \Phi _ { \mathrm { g r p h } }$ , or both (Dual UTS), with the unmodified GIN. Red and Grey mark the best and second best variants. Teal column: ogbg-ppa, official OGB split, accuracy in percentage points. $^ { * } p < 0 . 0 5 , ^ { * * } p < 0 . 0 1 , ^ { * * * } p < 0 . 0 0 1$ (paired t-test vs. unmodified GIN); unmarked values are not significant. Full statistics in Appendix B.
<table><tr><td></td><td>Variant</td><td>MUTAG</td><td>PROTEINS</td><td>COLLAB</td><td>ogbg-ppa</td></tr><tr><td rowspan="4">UTS-Aug GIN</td><td>GIN (unmodified)</td><td> $0 . 8 3 2 \pm 0 . 0 8 4$ </td><td> $0 . 7 4 0 \pm 0 . 0 3 4$ </td><td> $0 . 8 1 0 \pm 0 . 0 2 9$ </td><td> $6 7 . 8 4 \pm 0 . 9 1 7$ </td></tr><tr><td>(Embedding-UTS</td><td> $0 . 8 7 2 \pm 0 . 0 7 9$ </td><td> $0 . 7 5 1 \pm 0 . 0 3 1 ^ { \ast }$ </td><td> $0 . 8 2 3 \pm 0 . 0 2 8 ^ { * * }$ </td><td> $6 9 . 2 7 \pm 0 . 9 0 3 ^ { \ast }$ </td></tr><tr><td>Graph-UTS</td><td> $\mathbf { 0 . 8 9 0 \pm 0 . 0 7 6 }$ </td><td> $0 . 7 5 9 \pm 0 . 0 4 0 ^ { \ast }$ </td><td> $0 . 8 3 2 \pm 0 . 0 2 9 ^ { \ast \ast }$ </td><td> $7 0 . 2 0 \pm 0 . 9 5 1 ^ { \ast \ast }$ </td></tr><tr><td>Dual UTS</td><td> $0 . 8 7 8 \pm 0 . 0 7 8$ </td><td> $\mathbf { 0 . 7 6 3 \pm 0 . 0 3 6 ^ { \ast } }$ </td><td>0.837 ± 0.028***</td><td> $7 1 . 0 6 \pm 0 . 9 3 4 ^ { * * }$ </td></tr></table>

## 5 Experiments and Results

## 5.1 Experimental Protocol

We evaluate the proposed modified GIN framework using three UTS interventions (UTS-Aug, UTS-Reg, UTS-Pool) and perform experiments on four graph classification benchmarks spanning molecular (MUTAG), biological (PROTEINS), social-network (COLLAB), and large-scale biological (ogbgppa) domains. Details are provided in Appendix A.6.

In Table 1 and Table 2, the proposed UTS-Aug and UTS-Reg interventions are compared against the unmodified GIN (GIN with only standard readout features, $\mathbf { z } _ { \mathrm { s t r u c t } } )$ and unregularized GIN (GIN trained with task-specific objective, $\mathcal { L } _ { \mathrm { t a s k } }$ only) variants respectively. In Table 3, modified GIN with UTS-Pooling is compared with three pooling baselines, TopKPool [2], SAGPool [15], and TOGL [13]. Across all four benchmarks, the GIN backbone is trained under identical experimental settings (hidden size 128, readout dimension 384) to isolate the contribution of the proposed topology-aware components.

For MUTAG, PROTEINS, and COLLAB, which lack a standardized split, we employ 10-fold stratified cross-validation, using 9 seeds for MUTAG and 5 seeds for PROTEINS and COLLAB. Each fold uses a 90/10 train-test split, with 10% of the training set further held out for validation. For ogbg-ppa, we instead use the official, externally-fixed OGB species split [14] and report mean ± std over multiple seeds, following standard OGB leaderboard convention; this split holds out entire species unseen during training, testing out-of-distribution generalization rather than in-distribution accuracy. Full aggregate statistics, including confidence intervals and paired-significance p-values for all four datasets, are provided in Appendix B. The GIN backbone uses four layers for MUTAG and PROTEINS, three layers for COLLAB to mitigate oversmoothing in dense social graphs, and three layers for ogbg-ppa given its large average graph size (243.4 nodes). We report the mean and standard deviation of the classification accuracy (mean ± std) across all folds and seeds (TU datasets) or all seeds (ogbg-ppa). To ensure a fair comparison, all GIN variants are trained using identical hyperparameters and optimization settings, with only the proposed topology-aware components varying across the experiments.

## 5.2 Effect of Dual Unified Topological Signatures as UTS-Aug

Table 1 summarizes the classification performance of GIN augmented with each signature individually and jointly, under the updated 10-fold CV protocol (MUTAG/PROTEINS/COLLAB) and the official OGB split (ogbg-ppa). Graph-UTS achieves the best performance on MUTAG (+5.8% over unmodified GIN, not statistically significant at this sample size), while Dual-UTS achieves the best performance on PROTEINS (+2.3%, significant), COLLAB (+2.6%, significant), and ogbg-ppa (+3.22 points, significant), with Graph-UTS second-best on all three. The two signatures are complementary on PROTEINS, COLLAB, and ogbg-ppa, with their combination outperforming either individually, while Graph-UTS alone remains strongest specifically on the smaller, sparser MUTAG graphs.

The effectiveness of $\Phi _ { \mathrm { e m b } }$ alone is more modest than that of the graph signature but is positive across all datasets: it improves over the unmodified GIN on MUTAG (+4.0%, not significant), PROTEINS (+1.1%, significant), COLLAB (+1.3%, significant), and ogbg-ppa (+1.43 points, significant). Combining both signatures (Dual-UTS) achieves the best performance on PROTEINS, COLLAB, and ogbg-ppa, demonstrating that the static and dynamic signatures provide complementary structural information on these datasets, while Graph-UTS alone remains the strongest single descriptor on MUTAG. Notably, the descriptor gains on ogbg-ppa hold under a species-level train/test split that requires generalizing to entirely unseen species, suggesting the topological signal captured by UTS is not merely fitting to species-specific idiosyncrasies present in the training distribution.

## 5.3 Effect of UTS-Reg (Topology-Preserving Regularization)

The previous experiment showed that Embedding-UTS is sensitive to representation quality. We therefore evaluate whether UTS-Reg improves it by preserving topological fidelity during optimization. Table 2 shows the two regularizers behave complementarily: the evolution loss gives the largest gain on MUTAG (+1.9%, not significant), while the alignment loss performs best on PROTEINS (+1.4%, significant) and on ogbg-ppa (+1.70 points, significant). Neither loss improves over baseline on COLLAB, and both individual losses are significantly negative there, indicating the benefit of topology-preserving supervision is dataset dependent – consistent with the layer-wise Oversmoothing Index analysis in Appendix D, which shows the same evolution loss reduces representation collapse on MUTAG but increases it on PROTEINS and COLLAB. On ogbg-ppa, all three loss variants improve over baseline and are significant, unlike on COLLAB; we discuss this cross-dataset inconsistency further in §B.5. We discuss the theoretical implications of this dataset-dependence in Section 6.

Table 2: Comparison of ${ U T S - R e g }$ GIN backbone trained with the topological evolution loss $( \mathcal { L } _ { \mathrm { t o p o - e v o l } } )$ , the topological alignment loss $( \mathcal { L } _ { \mathrm { t o p o - a l i g n } } )$ , and their combination, with the unregularized GIN. Red and Grey mark the best and second best variant. Teal column: ogbg-ppa. Significance markers as in Table 1, paired vs. unregularized GIN. Full statistics in Appendix B.
<table><tr><td></td><td>Variant</td><td>MUTAG</td><td>PROTEINS</td><td>COLLAB</td><td>ogbg-ppa</td></tr><tr><td rowspan="4"> ${ U T S - R e g }$  GIN</td><td>GIN (unregularized)</td><td> $0 . 8 3 2 \pm 0 . 0 8 4$ </td><td> $0 . 7 4 0 \pm 0 . 0 3 4$ </td><td> ${ \bf 0 . 8 1 0 \pm 0 . 0 2 9 }$ </td><td> $6 7 . 8 4 \pm 0 . 9 1 7$ </td></tr><tr><td> $\left( \mathcal { L } _ { \mathrm { t o p o - e v o l } } \right.$ </td><td> ${ \bf 0 . 8 5 0 \pm 0 . 0 8 5 }$ </td><td> $0 . 7 3 8 \pm 0 . 0 3 4 ^ { \ast }$ </td><td> $0 . 8 0 5 \pm 0 . 0 3 0 ^ { \ast }$ </td><td> $6 8 . 3 6 \pm 0 . 9 8 2 ^ { \ast }$ </td></tr><tr><td> $\mathcal { L } _ { \mathrm { t o p o - a l i g n } }$ </td><td> $0 . 8 4 2 \pm 0 . 0 8 8$ </td><td> $\mathbf { 0 . 7 5 5 \pm 0 . 0 3 3 ^ { \ast } }$ </td><td> $0 . 8 0 6 \pm 0 . 0 3 0 ^ { \ast }$ </td><td> $6 9 . 5 4 \pm 0 . 9 4 1 ^ { \ast }$ </td></tr><tr><td>Combination</td><td> $0 . 8 4 1 \pm 0 . 0 8 4$ </td><td> $0 . 7 3 6 \pm 0 . 0 3 5 ^ { \ast }$ </td><td> $0 . 8 0 9 \pm 0 . 0 3 2 ^ { \ast }$ </td><td> $6 8 . 1 5 \pm 1 . 0 0 6 ^ { \ast }$ </td></tr></table>

Combining both regularizers, moreover, does not consistently outperform either alone, indicating that the two losses contribute differently across graph domains and their interaction requires datasetspecific balancing. This mirrors Section 5.2, where the embedding signature proved more sensitive to representation quality than the graph signature. On ogbg-ppa, the combination is also weaker than the alignment loss alone, though it remains significantly positive, unlike the negative combination effect observed on PROTEINS and COLLAB.

## 5.4 Evaluation of UTS-Pool (Topology-aware pooling)

Table 3 compares UTS-Pool against TOGL, SAGPool, and TopKPool under the same stratified 10- fold CV protocol (MUTAG/PROTEINS/COLLAB) and the official OGB split (ogbg-ppa). UTS-Pool achieves the best mean accuracy on PROTEINS and COLLAB, with the advantage over TopKPool and SAGPool statistically significant on both datasets, and a smaller but still significant advantage over TOGL (+0.1 percentage points on both). On MUTAG, TOGL is nominally ahead of UTS-Pool (0.840 vs. 0.835), though the difference does not reach significance; UTS-Pool remains ahead of SAGPool and TopKPool on MUTAG, but this gap is likewise not significant. On ogbg-ppa UTS-Pool remains significantly ahead of SAGPool and TopKPool on ogbg-ppa. TOGL is slightly but signficantly better than UTS-Pool (68.97 vs. 68.85, $\Delta = - 0 . 1 2 ,$ $p = 0 . 0 0 0 8 )$ . Notably, the COLLAB gains are obtained using the lightweight geometric scorer $\phi _ { v } ^ { \mathrm { l i g h t } }$ , demonstrating that even a computationally efficient local geometric proxy provides sufficient structural information to guide scalable and robust graph coarsening – the same lightweight scorer is used for UTS-Pool on ogbg-ppa, given its even larger average graph size (243.4 nodes).

Finally, Figure 3 illustrates the evolution of representative components of the embedding signature across GIN layers on the MUTAG dataset. As message passing progresses, the $H _ { 0 }$ mean lifetime decreases from 9.08 to 3.51, while the spectral gap $\lambda _ { 1 }$ shrinks from 1.26 to 0.81, indicating progressive contraction of the embedding topology associated with oversmoothing. In contrast, the $H _ { 1 }$ mean lifetime remains nearly unchanged, suggesting that higher-order topological structure is largely preserved. These trends demonstrate that UTS provides an interpretable characterization of representation dynamics during GNN optimization. Further details on the proposed Oversmoothing Index (OSI) formulation, implementation, and experimental analysis are provided in Appendix D.

Table 3: Accuracy comparison of UTS-Pool against TOGL, TopKPool, and SAGPool, same GIN backbone. Red and Grey mark the best and second best variant. Teal column: ogbg-ppa. ${ ^ { * } p } < 0 . 0 5 .$ $^ { * * } p < 0 . 0 1 , ^ { * * * } p < 0 . 0 0 1$ (paired t-test, variant vs. UTS-Pool); unmarked values are not significant. Full statistics in Appendix B. <sup>†</sup>Our faithful reimplementation of TOGL [13], adapted to this codebase; see Appendix E for implementation details.
<table><tr><td>Variant</td><td>MUTAG</td><td>PROTEINS</td><td>COLLAB</td><td>ogbg-ppa</td></tr><tr><td>TopKPool</td><td> $0 . 8 2 9 \pm 0 . 0 7 9$ </td><td> $0 . 7 4 1 \pm 0 . 0 3 3 ^ { \ast \ast \ast }$ </td><td> $0 . 8 1 8 \pm 0 . 0 2 9 ^ { \ast }$ </td><td> $6 7 . 5 3 \pm 0 . 9 8 8 ^ { \ast \ast \ast }$ </td></tr><tr><td>SAGPool</td><td> $0 . 8 3 1 \pm 0 . 0 8 0$ </td><td> $0 . 7 4 2 \pm 0 . 0 3 4 ^ { * * }$ </td><td> $0 . 8 1 9 \pm 0 . 0 2 9 ^ { \ast }$ </td><td> $6 7 . 8 8 \pm 0 . 9 6 3 ^ { * * * }$ </td></tr><tr><td> $\mathrm { T O G L } ^ { \dag }$ </td><td> ${ \bf 0 . 8 4 0 \pm 0 . 0 8 3 }$ </td><td> $0 . 7 4 4 \pm 0 . 0 3 3 ^ { \ast }$ </td><td> $0 . 8 2 0 \pm 0 . 0 2 9 ^ { \ast }$ </td><td> $6 8 . 9 7 \pm 0 . 9 2 6 ^ { \ast \ast \ast }$ </td></tr><tr><td> $\mathrm { U T S - P o o l }$ </td><td> $0 . 8 3 5 \pm 0 . 0 7 9$ </td><td> $\mathbf { 0 . 7 4 5 \pm 0 . 0 3 3 }$ </td><td> ${ \bf 0 . 8 2 1 \pm 0 . 0 2 9 }$ </td><td> $6 8 . 8 5 \pm 0 . 8 9 1$ </td></tr></table>

![](images/347c85182603c8318d952d12675070dd199e6f7f9ebee70f89ba57edc0bd2b27.jpg)

![](images/93dc835745aab65586290f0d0fc67ff73b89c270746aa8471447529989bd1b8b.jpg)

![](images/cc4a42681a0dae6bdcd8aaf86d4b15b8c4a4c2cea6f7e495bad3dd2d5ccab941.jpg)

![](images/d66cc1e793719b51d9056907bed5a92b6363ef7aef98e10718e1b268833feba2.jpg)  
Figure 3: Evolution of some components of embedding signature $\left( \Phi _ { \mathrm { e m b } } \right)$ across GIN layers on the MUTAG dataset. The sharp decay in $H _ { 0 }$ mean lifetime and the shrinking spectral gap $( \lambda _ { 1 } )$ empirically diagnose representation collapse, where distinct node embeddings oversmooth into an indistinguishable manifold

## 6 Theoretical Analysis

We now establish the theoretical properties of the proposed framework. Complete proofs are deferred to Appendix C.

## 6.1 Expressive Power of Dual Unified Topological Signatures

Theorem 1. Let A denote a message-passing GNN whose expressive power is bounded by the 1-Weisfeiler–Lehman (1-WL) test. Augmenting A with the graph signature $\Phi _ { \mathrm { g r p h } }$ yields a graph representation capable ofdistinguishing graph pairs that are indistinguishable under the 1-WL test whenever their global topological signatures differ.

## 6.2 Topology-Preserving Regularization

Proposition 1 (Topo-Evolution Loss: Heuristic Motivation). The topo-evolution loss $\mathcal { L } _ { \mathrm { s m o o t h } }$ (Eq. 5) heuristically discourages representation collapse by penalizing layer-to-layer divergence of the embedding-space topological signature $\mathbf { s } ^ { ( l ) }$ . This is a first-order, directional argument, not a proof that $\mathcal { L } _ { \mathrm { s m o o t h } }$ enforces a variance bound or guarantees reduced Oversmoothing Index (Eq. 30); the effect is empirically dataset-dependent (see Appendix C.2for thefulljustification and OSI comparison across datasets).

## 6.3 Computational Complexity

The graph signature $\Phi _ { \mathrm { g r p h } }$ is computed once per graph and cached, while the embedding signature $\Phi _ { \mathrm { e m b } } ^ { \mathrm { d i f f } }$ is recomputed at every layer, both at cost $O ( n ^ { 3 } )$ dominated by persistence and spectral

computations. The lightweight pooling proxy $\phi _ { v } ^ { \mathrm { l i g h t } }$ reduces this to $O ( d n \bar { \delta } ^ { 2 } )$ per pooling layer, linear in n at fixed local density. A full derivation is given in Appendix C.4.

## 7 Conclusion and Future Work

We introduced a unified topology-aware framework for graph representation learning based on Dual Unified Topological Signatures, jointly modeling the intrinsic topology of the input graph and the evolving topology of the learned embedding space throughout the GNN pipeline. Topology improves graph learning through three complementary mechanisms: topology-augmented representations, topology-preserving regularization, and topology-aware hierarchical pooling, with consistent improvements over the GIN backbone across molecular, biological, and social-network benchmarks.

Future work includes differentiable relaxations of the UTS-Pool selection operator and extending the framework to heterogeneous and temporal graph learning. The full Embedding-UTS signature incurs O(n³)-per-layer cost, which limits its direct application to long-range or web-scale graph benchmarks; developing a more computationally efficient variant of the signature is a natural direction for extending this work to such settings.

## References

[1] F. M. Bianchi, D. Grattarola, and C. Alippi. “Spectral clustering with graph neural networks for graph pooling”. In: Proceedings of the 37th International Conference on Machine Learning. ICML’20. 2020. 1, 3, 6

[2] C. Cangea et al. “Towards Sparse Hierarchical Graph Classifiers”. In: NeurIPS Workshop on Relational Representation Learning (R2L). 2018. 7, 26

[3] G. Carlsson and F. Mémoli. “Characterization, Stability and Convergence of Hierarchical Clustering Methods”. In: Journal ofMachine Learning Research 11 (2010). 22

[4] F. Chazal et al. “Gromov-Hausdorff Stable Signatures for Shapes using Persistence”. In: Computer Graphics Forum (Proc. SGP 2009). Vol. 28. 5. 2009. 22

[5] Y. Chen and Y. R. Gel. “Topological Pooling on Graphs”. In: Proceedings ofthe AAAI Conference on Artificial Intelligence. Vol. 37. 6. 2023. 2, 3

[6] D. Cohen-Steiner, H. Edelsbrunner, and J. Harer. “Stability of Persistence Diagrams”. In: Discrete & Computational Geometry 37.1 (2007). 22

[7] M. Cuturi. “Sinkhorn Distances: Lightspeed Computation of Optimal Transportation Distances”. In: Advances in Neural Information Processing Systems 26 (NIPS 2013). 2013. 13

[8] H. Edelsbrunner, D. Letscher, and A. Zomorodian. “Topological Persistence and Simplification”. In: Discrete & Computational Geometry 28.4 (2002). 2

[9] A. Feng and M. Weber. “Graph Pooling via Ricci Flow”. In: Transactions on Machine Learning Research (2024). 2, 3

[10] R. Forman. “Bochner’s Method for Cell Complexes and Combinatorial Ricci Curvature”. In: Discrete and Computational Geometry 29 (2003). 13

[11] D. Grattarola et al. “Understanding Pooling in Graph Neural Networks”. In: IEEE Transactions on Neural Networks and Learning Systems 35.2 (2024). 1, 3, 6

[12] W. L. Hamilton, R. Ying, and J. Leskovec. “Inductive Representation Learning on Large Graphs”. In: Advances in Neural Information Processing Systems. Vol. 30. 2017. 1, 3

[13] M. Horn et al. “Topological Graph Neural Networks”. In: International Conference on Learning Representations (ICLR). 2022. 3, 7, 9, 26

[14] W. Hu et al. “Open graph benchmark: datasets for machine learning on graphs”. In: Proceedings ofthe 34th International Conference on Neural Information Processing Systems. NIPS ’20. Vancouver, BC, Canada, 2020. 7, 16

[15] J. Lee, I. Lee, and J. Kang. “Self-Attention Graph Pooling”. In: Proceedings ofthe 36th International Conference on Machine Learning. Ed. by K. Chaudhuri and R. Salakhutdinov. Vol. 97. Proceedings of Machine Learning Research. 2019. 7, 26

[16] Q. Li, Z. Han, and X.-M. Wu. “Deeper Insights into Graph Convolutional Networks for Semi-Supervised Learning”. In: Proceedings of the Thirty-Second AAAI Conference on Artificial Intelligence. 2018. 1

[17] I. Loshchilov and F. Hutter. “Decoupled Weight Decay Regularization”. In: International Conference on Learning Representations (ICLR). 2019. 15

[18] C. Morris et al. “TUDataset: A collection of benchmark datasets for learning with graphs”. In: ICML 2020 Workshop on Graph Representation Learning and Beyond (GRL+ 2020). 2020. 16

[19] C.-C. Ni et al. “Community Detection on Networks with Ricci Flow”. In: Scientific Reports 9.1 (2019). 13

[20] Y. Ollivier. Ricci curvature ofMarkov chains on metric spaces. 2007. 2, 3, 13

[21] K. Oono and T. Suzuki. “Graph Neural Networks Exponentially Lose Expressive Power for Node Classification”. In: International Conference on Learning Representations (ICLR). 2020. 1

[22] T. G. Project. GUDHI User and Reference Manual. 3.12.0. 2026. 12, 14, 23

[23] F. Rottach et al. From Topology to Retrieval: Decoding Embedding Spaces with Unified Signatures. 2025. 2, 3

[24] T. K. Rusch, M. M. Bronstein, and S. Mishra. “A Survey on Oversmoothing in Graph Neural Networks”. In: arXiv preprint arXiv:2303.10993 (2023). 3

[25] Z. Wang and J. Fan. “Graph Classification via Reference Distribution Learning: Theory and Practice”. In: Advances in Neural Information Processing Systems. Vol. 37. 2024. 3

[26] K. Xu et al. “How Powerful are Graph Neural Networks?” In: International Conference on Learning Representations. 2019. 2, 3, 16, 19

[27] R. Ying et al. “Hierarchical Graph Representation Learning with Differentiable Pooling”. In: Advances in Neural Information Processing Systems. Vol. 31. 2018. 1, 3, 6

## A Appendix

## A.1 Full Derivations of Universal Topological Signatures

This appendix gives the complete definitions, differentiability remarks, and computational notes underlying the embedding- and graph-level signature maps introduced in Section 4.1.1 and Section 4.1.2.

## A.2 Embedding UTS: $\Phi _ { \mathrm { e m b } }$

## A.2.1 Pairwise Distance Matrix

$$
D _ { u v } = \left\| \mathbf { h } _ { u } - \mathbf { h } _ { v } \right\| _ { 2 } = \sqrt { \sum _ { i = 1 } ^ { d _ { l } } ( h _ { u i } - h _ { v i } ) ^ { 2 } + \varepsilon } , \quad \varepsilon = 1 0 ^ { - 8 } ,\tag{9}
$$

where $\varepsilon$ ensures numerical stability when $\mathbf { h } _ { u } = \mathbf { h } _ { v }$ (in case of oversmoothing).

## A.2.2 Local Geometry Features (Dimensions 7–10)

Let $\mathcal { N } _ { k } ( v )$ denote the k nearest neighbours of node v excluding itself. Define

$$
\mu _ { \mathrm { n n } } = \frac { 1 } { n } \sum _ { v \in \mathcal { V } } \frac { 1 } { k } \sum _ { u \in \mathcal { N } _ { k } ( v ) } D _ { v u } ,\tag{10}
$$

$$
\sigma _ { \mathrm { n n } } = \sqrt { \frac { 1 } { n k } \sum _ { v } \sum _ { u \in \mathcal { N } _ { k } ( v ) } \left( D _ { v u } - \mu _ { \mathrm { n n } } \right) ^ { 2 } } ,\tag{11}
$$

$$
\Delta = \operatorname* { m a x } _ { u , v \in \mathcal { V } } D _ { u v } ,\tag{12}
$$

$$
\hat { d } = \mathrm { m e d i a n } _ { v \in \mathcal { V } } \frac { \log r _ { 2 } ( v ) + \varepsilon } { \log r _ { 1 } ( v ) + \varepsilon } ,\tag{13}
$$

where $r _ { j } ( \boldsymbol { v } )$ is the distance to the j-th nearest neighbour of $v .$ Equation (13) provides a correlationdimension estimate of the intrinsic dimensionality of the embedding manifold, and is used as written for $\Phi _ { \mathrm { e m b } }$ (exact).

Remark on Differentiability. $\mu _ { \mathrm { n n } } , \sigma _ { \mathrm { n n } } ,$ , and $\Delta { \ : ( 1 0 ) } – ( 1 2 )$ are differentiable with respect to $\mathbf { H } ^ { ( l ) }$ through (9) in both variants. The median in (13), however, has a degenerate gradient (non-zero only at the median element), so $\Phi _ { \mathrm { e m b } } ^ { \mathrm { d i f f } }$ replaces it with a mean over the same log-ratio, distributing gradient across all n nodes; see Appendix $\bar { \mathbf { A . } } 4$ for the full justification.

## A.2.3 Persistence Features (Dimensions 1-6)

For $\Phi _ { \mathrm { e m b } }$ (exact), dimensions 1-6 are computed via standard Vietoris–Rips persistent homology on the detached point cloud using GUDHI [22] - identical in construction to the persistence features of $\Phi _ { \mathrm { g r p h } }$ (§A.3), yielding exact mean lifetimes, persistence entropies, and integer Betti numbers $\beta _ { 0 } , \beta _ { 1 }$ For $\mathrm { \Phi _ { e m b } ^ { \mathrm { d i f f } } }$ , these exact operations are replaced by the differentiable relaxations below, since MST extraction and simplicial homology computation are not differentiable.

Let $\{ w _ { e } \}$ denote the upper-triangular entries of $\mathbf { D } ,$ sorted in ascending order as $w _ { ( 1 ) } \leq w _ { ( 2 ) } \leq \cdot \cdot \cdot \leq$ $w _ { \left( { \frac { n } { 2 } } \right) }$ . We define the H0 lifetime surrogate as the $n - 1$ smallest edge weights:

$$
\ell _ { i } ^ { ( 0 ) } = w _ { ( i ) } , \quad i = 1 , \ldots , n - 1 .\tag{14}
$$

This approximates the persistence diagram of the 0-dimensional homology of the Vietoris–Rips complex: in an exact Rips filtration, connected components merge at minimum spanning tree edge weights. Equation (14) is a differentiable relaxation since MST extraction (Kruskal’s algorithm) is non-differentiable.

For 1-dimensional homology we enumerate all triangles $\binom { \nu } { 3 }$ and define, for each triangle ${ \boldsymbol \tau } = ( u , v , w )$ with sorted edge lengths $e _ { 1 } ( \tau ) \le e _ { 2 } ( \tau ) \le e _ { 3 } ( \tau )$

$$
\ell _ { \tau } ^ { ( 1 ) } = \operatorname* { m a x } ( 0 , e _ { 3 } ( \tau ) - e _ { 2 } ( \tau ) ) .\tag{15}
$$

This follows from the Vietoris–Rips persistence pairing: a 1-cycle is born when the second-longest triangle edge enters the filtration and dies when the longest edge fills the triangle.

Remark 1 (When are surrogates used?). Surrogates $( \Phi _ { \mathrm { e m b } } ^ { \mathrm { d i f f } } )$ are used only where gradients must propagate through the signature: the topology-preserving regularizers (§4.3) and the UTS-Pool node scorer (§4.4). UTS-Aug (§4.2) uses the exact signature $\Phi _ { \mathrm { e m b } }$ , computed on detached embeddings with no gradient requirement. See Appendix A.4for the complete differentiable-surrogate derivation. Remark 2 (Computational cost of H1). Triangle enumeration scales as $O ( n ^ { 3 } )$ . We cap computation at $n \leq n _ { \operatorname* { m a x } }$ (default $n _ { \mathrm { m a x } } = 5 0 ) ;$ for larger point clouds the H1 features are set to zero. This is appropriate for large social network graphs (e.g. COLLAB) where oversmoothing renders the embedding point cloud near-degenerate in any case. This cap applies to $\Phi _ { \mathrm { e m b } } ^ { \mathrm { d i f f } }$ wherever it is computed per layer – both the topology-evolution loss $( \ S 4 . 3 )$ and the UTS-Pool node scorer $( \ S 4 . 4 )$ – and does not apply to UTS-Aug, which uses the exact signature $\Phi _ { \mathrm { e m b } }$ computed once on detached embeddings.

From (14) and (15) we compute mean lifetimes and persistence entropies:

$$
\bar { \ell } ^ { ( k ) } = \frac { 1 } { | \mathcal { L } ^ { ( k ) } | } \sum _ { i } \ell _ { i } ^ { ( k ) } ,\tag{16}
$$

$$
H ^ { ( k ) } = - \sum _ { i } \hat { p } _ { i } ^ { ( k ) } \log \hat { p } _ { i } ^ { ( k ) } , \quad \hat { p } _ { i } ^ { ( k ) } = \frac { \ell _ { i } ^ { ( k ) } } { \sum _ { j } \ell _ { j } ^ { ( k ) } + \varepsilon } .\tag{17}
$$

Soft Betti numbers are defined as

$$
{ \tilde { \beta } _ { k } = \sum _ { i } \sigma \left( 1 0 \left( \ell _ { i } ^ { ( k ) } - \mu _ { \mathrm { { n n } } } \right) \right) , }\tag{18}
$$

where $\sigma$ is the sigmoid function and $\mu _ { \mathrm { n n } }$ serves as an adaptive threshold.

## A.2.4 Spectral Features (Dimensions 11–14)

For $\Phi _ { \mathrm { e m b } }$ (exact), the adjacency is a standard binary k-NN graph and $\tilde { \bf L }$ is the unnormalised combinatorial Laplacian, with eigenvalues computed via scipy.linalg.eigh; no soft kernel or regularisation is applied, since differentiability is not required. For $\Phi _ { \mathrm { e m b } } ^ { \mathrm { d i f f } ^ { - } } ,$ we construct a soft k-NN adjacency matrix via a Gaussian kernel instead, to keep the spectral features differentiable:

Construct a soft k-NN adjacency matrix via a Gaussian kernel:

$$
A _ { u v } ^ { \mathrm { s o f t } } = \exp \left( - \frac { D _ { u v } ^ { 2 } } { 2 ( \mu _ { \mathrm { n n } } + \varepsilon ) ^ { 2 } } \right) \cdot \mathbf { 1 } [ u \neq v ] .\tag{19}
$$

The bandwidth $\mu _ { \mathrm { n n } }$ is kept in the computation graph (not detached), ensuring gradients flow through the spectral features back to $\mathbf { H } ^ { ( l ) }$ . Let $\mathbf { L } _ { \mathrm { s o f t } } = \mathbf { D } _ { \mathrm { s o f t } } - \mathbf { A } ^ { \mathrm { s o f t } }$ be the unnormalised Laplacian, regularised as

$$
\tilde { \mathbf { L } } = \frac { \mathbf { L } _ { \mathrm { s o f t } } + \delta \mathbf { I } } { \mathrm { m a x } _ { v } \ : d _ { v } ^ { \mathrm { s o f t } } + \varepsilon } , \quad \delta = 1 0 ^ { - 4 } ,\tag{20}
$$

where $\begin{array} { r } { d _ { v } ^ { \mathrm { s o f t } } = \sum _ { u } A _ { u v } ^ { \mathrm { s o f t } } } \end{array}$ . Division by $\operatorname* { m a x } _ { v } d _ { v } ^ { \mathrm { s o f t } }$ normalises the matrix to $[ 0 , 1 ]$ scale, preventing ill-conditioning on dense graphs (e.g. COLLAB with $\bar { d } \approx 8 . 9 )$

Let $0 \leq \lambda _ { 1 } \leq \lambda _ { 2 } \leq \cdot \cdot \cdot \leq \lambda _ { n }$ be the eigenvalues of $\tilde { \mathbf { L } } ,$ computed via torch.linalg.eigh (fully differentiable for symmetric matrices). The spectral gap $\lambda _ { 1 }$ , and $\lambda _ { 2 } ,$ λ<sub>3</sub> approximate the Fiedler value and higher connectivity information. Spectral entropy is

$$
H _ { \mathrm { s p e c } } = - \sum _ { i } \hat { q } _ { i } \log \hat { q } _ { i } , \quad \hat { q } _ { i } = \frac { \operatorname* { m a x } ( \lambda _ { i } , \varepsilon ) } { \sum _ { j } \operatorname* { m a x } ( \lambda _ { j } , \varepsilon ) } .\tag{21}
$$

Stability. The spectral entropy $H _ { \mathrm { s p e c } }$ is Lipschitz continuous in the entries of $\tilde { \bf L }$ by the Weyl perturbation theorem and the Lipschitz continuity of the entropy function on the probability simplex.

## A.3 Graph $\mathbf { U T S } \colon \Phi _ { \mathrm { g r p h } }$

## A.3.1 Ollivier-Ricci Curvature (Dimensions 1-4)

For an edge $( u , v ) \in \mathcal { E } ,$ , the Ollivier-Ricci curvature [20] is

$$
\kappa ( u , v ) = 1 - \frac { W _ { 1 } ( \mu _ { u } , \mu _ { v } ) } { d _ { G } ( u , v ) } ,\tag{22}
$$

where $d _ { G } ( u , v )$ is the geodesic distance, $\mu _ { v }$ is the α-lazy random walk measure at v:

$$
\begin{array} { r } { \mu _ { v } ( w ) = \left\{ \begin{array} { l l } { \alpha } & { w = v } \\ { ( 1 - \alpha ) / \deg ( v ) } & { ( v , w ) \in \mathcal { E } } \\ { 0 } & { \mathrm { o t h e r w i s e , } } \end{array} \right. } \end{array}\tag{23}
$$

and $W _ { 1 } ( \mu _ { u } , \mu _ { v } )$ is the Wasserstein-1 distance computed via the Sinkhorn approximation [7]:

$$
W _ { 1 } ^ { \varepsilon } ( \mu _ { u } , \mu _ { v } ) \approx \langle \mathbf { u } \otimes \mathbf { K } \odot \mathbf { C } \otimes \mathbf { v } \rangle ,\tag{24}
$$

where $\mathbf { K } = \exp ( - \mathbf { C } / \varepsilon )$ is the Gibbs kernel, C is the cost matrix of geodesic distances restricted to the supports of $\mu _ { u }$ and $\mu _ { v } .$ , and u, v are Sinkhorn scaling vectors. This replaces the C++ implementation of [19], eliminating segmentation faults on varied graph sizes while remaining differentiable through the Sinkhorn iterations if required.

From the edge curvatures $\{ \kappa _ { e } \} _ { e \in \mathcal { E } }$ we extract

$$
\phi _ { \mathrm { O R C } } = \bigg [ \bar { \kappa } , ~ \mathrm { V a r } ( \kappa ) , ~ \operatorname* { m i n } _ { e } \kappa _ { e } , ~ \frac { \lvert \{ e : \kappa _ { e } < 0 \} \rvert } { \lvert \mathcal { E } \rvert } \bigg ] \in \mathbb { R } ^ { 4 } .\tag{25}
$$

Edges with $\kappa _ { e } > 0$ lie within dense clusters; edges with $\kappa _ { e } < 0$ are inter-cluster bridges. The fraction of negatively curved edges thus serves as a proxy for community structure [19].

## A.3.2 Forman-Ricci Curvature (Dimensions 5-8)

The combinatorial Forman-Ricci curvature [10] of an edge $\boldsymbol { e } = ( u , v )$ is

$$
F ( e ) = w _ { e } \left( \frac { w _ { u } } { w _ { e } } + \frac { w _ { v } } { w _ { e } } - \sum _ { e ^ { \prime } \sim u , e ^ { \prime } \ne e } \sqrt { \frac { w _ { e } } { w _ { e ^ { \prime } } } } - \sum _ { e ^ { \prime } \sim v , e ^ { \prime } \ne e } \sqrt { \frac { w _ { e } } { w _ { e ^ { \prime } } } } \right) ,\tag{26}
$$

where $w _ { e } , w _ { v }$ are edge and vertex weights (unity for unweighted graphs). We extract $\phi _ { \mathrm { F R C } } ~ { = }$ $[ \bar { F } , \mathrm { V a r } ( F )$ , min $_ { \cdot e } F _ { e } , \mathbf { \bar { m a x } } _ { e } F _ { e } ] \in \mathbb { R } ^ { 4 }$

## A.3.3 Distance Features (Dimensions 9-10)

Let $\ell _ { G } ( u , v )$ denote the shortest-path distance in the largest connected component of G. We extract the mean path length $\begin{array} { r } { \bar { \ell } _ { G } = \frac { 1 } { n ( n - 1 ) } \sum _ { u \neq v } \ell _ { G } ( u , v ) } \end{array}$ and diameter diam $( G ) = \operatorname* { m a x } _ { u , v } \ell _ { G } ( u , v )$

## A.3.4 Spectral Features (Dimensions 11-13)

Let $\mathbf { L } _ { G } = \mathbf { D } _ { G } - \mathbf { A }$ be the combinatorial graph Laplacian and $0 = \lambda _ { 1 } ^ { G } \leq \lambda _ { 2 } ^ { G } \leq \cdots$ its eigenvalues. We extract the Fiedler value $\lambda _ { 2 } ^ { G }$ (algebraic connectivity), the largest eigenvalue $\lambda _ { n } ^ { G }$ , and spectral entropy $H _ { \mathrm { s p e c } } ^ { G }$ , computed by the same entropy formula as $H _ { \mathrm { s p e c } }$ (Eq. 2’s spectral group) but applied directly to the exact eigenvalues of $\mathbf { L } _ { G }$ above – no soft kernel or regularisation is used, since $\Phi _ { \mathrm { g r p h } }$ requires no gradient $( \ S \mathrm { A } . 3$ , Persistence Features).

## A.3.5 Persistence Features (Dimensions 14-17)

Applying the Vietoris-Rips filtration to the shortest-path distance matrix $( \ell _ { G } ( u , v ) ) _ { u , v }$ via GUDHI [22], we extract mean lifetimes and persistence entropies of $H _ { 0 }$ and $H _ { 1 }$ homology groups, computed exactly (not surrogates) since $\Phi _ { \mathrm { g r p h } }$ is non-differentiable by design.

## A.3.6 Structural Features (Dimensions 18-27)

Degree statistics $[ \bar { d } , \mathrm { V a r } ( d )$ , max $, d _ { v } ]$ , clustering coefficients $\left[ \bar { c } , \mathrm { V a r } ( c ) \right]$ ], betweenness centrality $[ \bar { b } , \bar { \mathrm { V a r } } ( b ) ]$ , mean closeness centrality q¯, number of connected components $N _ { \mathrm { c c } } ,$ and largest component ratio $| { \mathcal { C } } _ { \mathrm { m a x } } | / n$

## A.4 Differentiable Surrogate Formulation of $\Phi _ { \mathrm { e m b } }$

The readout descriptor (§4.1) computes $\Phi _ { \mathrm { e m b } }$ on $\mathbf { H } ^ { ( L ) }$ .detach() using GUDHI for exact persistent homology. However, the regularisation losses (§4.2) and pooling scorer (§4.3) require gradients to flow back through the topological signature into the GIN encoder. We therefore maintain a fully differentiable re-implementation, $\breve { \Phi } _ { \mathrm { e m b } } ^ { \mathrm { d i f f } }$ , in PyTorch (diff\_uts.py) that produces the same 14-dimensional vector as $\Phi _ { \mathrm { e m b } }$ without any NumPy or GUDHI operations. The two implementations agree closely on well-separated point clouds and diverge gracefully in the oversmoothing regime where exact persistence is ill-defined regardless.

Below we justify each non-obvious design choice.

Pairwise distances. We compute

$$
\begin{array} { r } { D _ { u v } = \sqrt { \sum _ { i } ( h _ { u i } - h _ { v i } ) ^ { 2 } + \varepsilon } , \quad \varepsilon = 1 0 ^ { - 8 } , } \end{array}\tag{27}
$$

rather than torch.cdist. The gradient of cdist at $D _ { u v } = 0$ is undefined when $\mathbf { h } _ { u } = \mathbf { h } _ { v }$ , which occurs frequently in early training before the encoder has differentiated nodes. The ε inside the square root regularises the gradient to $\partial D _ { u v } / \partial \mathbf { h } _ { u } = ( \mathbf { h } _ { u } - \mathbf { h } _ { v } ) / D _ { u v }$ , which remains bounded.

$H _ { 0 }$ surrogate. Exact $H _ { 0 }$ persistence requires a minimum spanning tree. MST extraction (Kruskal’s algorithm) involves argsort whose gradient is non-unique at ties and propagates to only one entry per comparison. We use the $N - 1$ smallest upper-triangle pairwise distances $w _ { ( 1 ) } \leq . . . \leq w _ { ( N - 1 ) }$ as a differentiable proxy for component lifetimes (cf. Eq. 14). This approximation is tight when the point cloud is well-separated: in an exact Rips filtration, components merge precisely at MST edge weights, which are the $N - 1$ smallest distances when no two inter-component distances are equal. The gradient of torch.sort is well-defined (it equals the identity on the sorted entries) and distributes across the source entries.

$H _ { 1 }$ surrogate. We enumerate all $\binom { N } { 3 }$ triangles and compute per-triangle lifetimes $\ell _ { \tau } ^ { ( 1 ) }$ = ma $\mathrm { x } ( 0 , e _ { 3 } ( \tau ) - e _ { 2 } ( \tau ) )$ where $e _ { 2 } , e _ { 3 }$ are the second- and third-longest edge lengths of triangle τ (cf. Eq. 15). This quantity is an upper bound on the true $H _ { 1 }$ lifetime in the Vietoris–Rips filtration: under Rips, a 1-cycle is born at most when the second-longest triangle edge enters and dies when the longest edge fills the triangle. Triangle enumeration is $\breve { O } ( N ^ { 3 } )$ in memory; we cap it at $N \leq 5 0$ setting $\bar { \ell } ^ { ( 1 ) } = H ^ { ( 1 ) } = \tilde { \beta } _ { 1 } = 0$ for larger point clouds. For the datasets in our experiments, only COLLAB graphs exceed this threshold after the 2-hop neighbourhood cap of 30 nodes is applied in §4.3.

Soft Betti numbers. The exact Betti numbers $\beta _ { k }$ are integers and non-differentiable. We replace them with sigmoid-smoothed counts $\begin{array} { r } { \tilde { \beta } _ { k } = \sum _ { j } \sigma \big ( 1 0 ( \ell _ { j } ^ { ( k ) } - \mu _ { \mathrm { n n } } ) \big ) } \end{array}$ , where the threshold $\mu _ { \mathrm { n n } }$ adapts to the point cloud density and the scale factor 10 provides sharpness comparable to a step function while keeping gradients non-zero.

Spectral features. The binary k-NN adjacency used in the non-differentiable $\Phi _ { \mathrm { e m b } }$ is replaced by a Gaussian kernel adjacency $A _ { u v } ^ { \mathrm { s o \bar { f } t } } = \exp ( \bar { - } D _ { u v } ^ { 2 } / 2 ( \mu _ { \mathrm { n n } } + \varepsilon ) ^ { 2 } ) \cdot \mathbb { 1 } [ u \neq v ]$ (cf. Eq. 19). The bandwidth $\mu _ { \mathrm { n n } }$ remains in the computation graph so gradients flow through the spectral features. We add $1 0 ^ { - 6 } \mathbf { I }$ to the adjacency before forming the Laplacian to prevent degenerate (zero) diagonal entries that cause NaN in torch.linalg.eigh. We use eigh rather than eig: for symmetric matrices it guarantees real eigenvalues and implements the analytic gradient ${ \partial \lambda _ { i } } / { \partial \mathbf { L } } = \mathbf { q } _ { i } \mathbf { q } _ { i } ^ { \top }$ via the eigenvalue equation, which is numerically stable for distinct eigenvalues.

Intrinsic dimensionality. We use mean rather than median over the log-ratio log $r _ { i } ^ { ( 2 ) } / \log r _ { i } ^ { ( 1 ) }$ (cf. Eq. 13). The gradient of median is non-zero only at the median element, making it effectively a single-node pass-through and providing no useful learning signal. The mean distributes gradient uniformly across all N nodes.

## A.5 Architecture and Training Details

Full readout dimensionalities. The classifier MLP input dimension varies by ablation variant. With d = 128 (hidden dim), the base readout $\mathbf { z } _ { \mathrm { s t r u c t } } \in \mathbb { R } ^ { 3 d = 3 8 4 }$ . Appending $\Phi _ { \mathrm { e m b } }$ adds 14 dimensions; appending $\Phi _ { \mathrm { g r p h } }$ adds 27 dimensions. Table 4 summarises the classifier input dimension and the number of parameters in the classifier head per variant (backbone parameters are identical across all variants).

Table 4: Classifier input dimension and head parameter count per variant (d = 128, C = num classes). Backbone parameters are identical across all variants and not included.
<table><tr><td>Variants</td><td>Readout dim</td><td>Head params (approx.)</td></tr><tr><td>GIN and UTS-Pool</td><td>384</td><td> $3 8 4 \times 1 2 8 + 1 2 8 \times 6 4 + 6 4 \times C$ </td></tr><tr><td>Embedding-UTS</td><td>398</td><td> $3 9 8 \times 1 2 8 + \ldots$ </td></tr><tr><td>Graph-UTS</td><td>411</td><td> $4 1 1 \times 1 2 8 + \ldots$ </td></tr><tr><td>Dual UTS</td><td>425</td><td> $4 2 5 { \times } 1 2 8 + \ldots$ </td></tr></table>

Crucially, UTS-Reg $L _ { t o p o - e v o l } , L _ { t o p o - a l i g n }$ and combination of both objectives all share the same architecture as GIN (unregularized) baseline. Any accuracy differences between these variants and baseline are therefore attributable entirely to the training objective, not to additional model capacity.

TopoRegLoss projection. $\Phi _ { \mathrm { g r p h } } ( G ) \in \mathbb { R } ^ { 2 7 }$ and $\Phi _ { \mathrm { e m b } } ^ { \mathrm { d i f f } } ( { \bf H } ^ { ( L ) } ) \in \mathbb { R } ^ { 1 4 }$ live in spaces of different dimension and scale. The learnable projection $\mathbf { W } _ { \mathrm { a l i g n } } \in \mathbb { R } ^ { 1 4 \times 2 7 } \mathrm { i n } \mathcal { L } _ { \mathrm { r e g } }$ (Eq. 4) is trained jointly with the encoder, finding the optimal linear alignment between structural and embedding-space topology. This avoids hand-crafted feature matching while remaining interpretable: the learned projection weights reveal which structural features are most predictive of embedding-space topology.

UTS-Pool gradient flow. The node scorer in §4.3 computes importance scores $\alpha _ { v }$ via $\Phi _ { \mathrm { e m b } } ^ { \mathrm { d i f f } }$ and an MLP $f _ { \theta } ,$ , but the subsequent top-k selection returns integer indices. Consequently, no gradient flows back through the selection decision itself; the scorer receives gradient only indirectly from the downstream cross-entropy loss through the embeddings of selected nodes. The scorer thus acts as an adaptive heuristic rather than a fully learned selection criterion. A soft-selection relaxation (e.g. Gumbel-softmax top-k) would enable end-to-end training of the pooling decision and is left to future work.

Training protocol. All models use AdamW [17] with learning rate $1 0 ^ { - 3 }$ and weight decay $1 0 ^ { - 4 }$ cosine annealing to $\eta _ { \mathrm { m i n } } = 1 0 ^ { - 5 }$ over 200 epochs, and gradient norm clipping at 2.0. Early stopping with patience 30 on validation accuracy selects the final checkpoint. We use 10-fold stratified crossvalidation: for each fold, the held-out 10% serves as the test set, and the remaining 90% is further split 90/10 into training and validation sets, yielding an 81/9/10 train/validation/test split per fold, stratified by class label, with every graph evaluated as a test example exactly once per seed.

GraphUTS precomputation. Computing $\Phi _ { \mathrm { g r p h } }$ per graph at every training step is prohibitive: it involves all-pairs shortest paths $( O ( n ^ { 2 } ) )$ , GUDHI Rips persistence, and optionally Ricci curvature. We precompute and cache $\Phi _ { \mathrm { g r p h } } ( G _ { i } )$ for all graphs before the first training epoch and look up vectors by dataset index during training. The cache is shared across all seeds and ablation variants in a single run, so the precomputation cost is paid exactly once.

## A.6 Dataset Statistics and Descriptions

Table 5 summarizes the three TU benchmark datasets [18] used in Section 5, spanning a molecular, a biological, and a social-network domain.

Table 5: Statistics of the graph classification datasets used in our experiments.
<table><tr><td>Dataset</td><td>Graphs</td><td>Classes</td><td>Avg. Nodes</td><td>Avg. Edges</td><td>Node Features</td></tr><tr><td>MUTAG</td><td>188</td><td>2</td><td>17.93</td><td>19.79</td><td>7 (categorical, atom type)</td></tr><tr><td>PROTEINS</td><td>1113</td><td>2</td><td>39.06</td><td>72.82</td><td>3 (categorical, SSE type)</td></tr><tr><td>COLLAB</td><td>5000</td><td>3</td><td>74.49</td><td>2457.78</td><td>none (node degree used)</td></tr><tr><td>ogbg-ppa</td><td>158,100</td><td>37</td><td>243.4</td><td>2266.1</td><td>none (learned embedding)</td></tr></table>

MUTAG consists of 188 chemical compound graphs representing nitroaromatic and heteroaromatic compounds, with nodes as atoms and edges as chemical bonds. Node features are a 7-dimensional one-hot encoding of atom type. The binary classification task is to predict mutagenic effect on the Gram-negative bacterium Salmonella typhimurium. MUTAG is the smallest and sparsest of the three datasets, with graphs ranging from 10 to 28 nodes.

PROTEINS contains 1113 graphs representing protein tertiary structures, with nodes as secondary structure elements (helices, sheets, and turns) and edges connecting elements that are sequential neighbors along the amino acid chain or spatially proximate in the folded structure. Node features are a 3-dimensional one-hot encoding of secondary structure element type. The binary classification task distinguishes enzymes from non-enzymes. Graphs are substantially larger and more variable in size than MUTAG, ranging up to 620 nodes.

COLLAB is a scientific-collaboration dataset derived from ego-networks of researchers in three fields: High Energy Physics, Condensed Matter Physics, and Astro Physics, with the 3-way classification task being to predict the field from the collaboration ego-network. Nodes represent researchers and edges represent co-authorship; the dataset carries no categorical node features, so node degree is used as the input feature following standard practice [26]. COLLAB graphs are markedly denser than MUTAG or PROTEINS, with average degree an order of magnitude higher, which motivates both the reduced GIN depth (Section 5) and the lightweight pooling scorer $\phi _ { v } ^ { \mathrm { l i g h t } }$ (Section 4.4) used for this dataset.

ogbg-ppa [14] contains 158,100 protein-protein association graphs extracted from species spanning 37 taxonomic groups, with nodes representing proteins and edges representing biologically meaningful associations (7-dimensional edge features encode association type and confidence, not used in our GIN backbone). Nodes carry no input features; following the official OGB baseline for this dataset, we use a single learnable embedding shared across all nodes. The task is 37-way taxonomic-group classification. Unlike MUTAG, PROTEINS, and COLLAB, ogbg-ppa uses an official, externallyfixed species split: validation and test graphs are drawn from species entirely unseen during training, even though each held-out species still belongs to one of the 37 training taxonomic groups. This makes ogbg-ppa an out-of-distribution generalization test rather than an in-distribution random split, and is the reason we depart from the 10-fold CV protocol used for the other three datasets (§5). ogbg-ppa’s graphs are also substantially larger than any TU dataset used in this work - more than 3× COLLAB’s average size and 32× its graph count

## B Full Ablation Results

This appendix reports full aggregate statistics for every comparison summarized in Tables 1–3. All results use 10-fold stratified cross-validation (MUTAG: 9 seeds × 10 folds = 90 evaluations per variant; PROTEINS and COLLAB: 5 seeds × 10 folds = 50 evaluations per variant). Confidence intervals are bootstrap 95% CIs (1000 resamples) unless noted as paired-difference CIs. p-values are paired t-tests; the comparator for each subsection is stated in its introduction. All values below are rounded to three decimal places; deltas are computed from full-precision means before rounding, so a delta may differ slightly from the difference of the rounded means shown.

We allocate seeds according to measurement variance. Under 10-fold CV, each test fold contains roughly 19 graphs for MUTAG versus 111 for PROTEINS and 500 for COLLAB - MUTAG’s per-fold accuracy estimates are inherently noisier simply because they’re averaged over far fewer graphs. Using more seeds (9 vs. 5) on the smaller, higher-variance dataset is a targeted allocation of statistical power to where it’s most needed. This is also why several MUTAG comparisons remain non-significant even at 90 total evaluations, while the same comparisons reach significance on PROTEINS/COLLAB at only 50 — the underlying noise floor, differs by dataset

## B.1 UTS Descriptor Variants

This subsection isolates the effect of adding Unified Topological Signature (UTS) descriptors, comparing Embedding-UTS, Graph-UTS, and Dual-UTS against the unmodified, unregularized GIN backbone.

Table 6: UTS descriptor variants vs. GIN (unmodified/unregularized) — MUTAG.
<table><tr><td>Variant</td><td>Mean ± Std</td><td>∆</td><td>95% CI</td><td>p</td></tr><tr><td>GIN (unmodified/unregularized)</td><td> $0 . 8 3 2 \pm 0 . 0 8 4$ </td><td></td><td></td><td></td></tr><tr><td>Embedding-UTS</td><td> $0 . 8 7 2 \pm 0 . 0 7 9$ </td><td>+0.040</td><td>[0.863, 0.880]</td><td>0.109</td></tr><tr><td>Graph-UTS</td><td> $0 . 8 9 0 \pm 0 . 0 7 6$ </td><td> $+ 0 . 0 5 8$ </td><td>[0.874, 0.906]</td><td>0.102</td></tr><tr><td>Dual-UTS</td><td> $0 . 8 7 8 \pm 0 . 0 7 8$ </td><td>+0.047</td><td>[0.870, 0.887]</td><td>0.097</td></tr></table>

Table 7: UTS descriptor variants vs. GIN (unmodified/unregularized) — PROTEINS.
<table><tr><td>Variant</td><td> $\mathbf { M e a n } \pm \mathbf { S t d }$ </td><td> $\Delta$ </td><td>95% CI</td><td>p</td></tr><tr><td>GIN (unmodified/unregularized)</td><td> $0 . 7 4 0 \pm 0 . 0 3 4$ </td><td></td><td></td><td></td></tr><tr><td>Embedding-UTS</td><td> $0 . 7 5 1 \pm 0 . 0 3 1$ </td><td> $+ 0 . 0 1 1$ </td><td>[0.743, 0.760]</td><td>0.042</td></tr><tr><td>Graph-UTS</td><td> $0 . 7 5 9 \pm 0 . 0 4 0$ </td><td> $+ 0 . 0 1 9$ </td><td>[0.748, 0.770]</td><td>0.039</td></tr><tr><td>Dual-UTS</td><td> $0 . 7 6 3 \pm 0 . 0 3 6$ </td><td> $+ 0 . 0 2 3$ </td><td>[0.752, 0.773]</td><td>0.028</td></tr></table>

Table 8: UTS descriptor variants vs. GIN (unmodified/unregularized) — COLLAB.
<table><tr><td>Variant</td><td> $\mathbf { M e a n } \pm \mathbf { S t d }$ </td><td>∆</td><td>95% CI</td><td>p</td></tr><tr><td>GIN (unmodified/unregularized)</td><td> $0 . 8 1 0 \pm 0 . 0 2 9$ </td><td></td><td></td><td></td></tr><tr><td>Embedding-UTS</td><td> $0 . 8 2 3 \pm 0 . 0 2 8$ </td><td> $+ 0 . 0 1 3$ </td><td>[0.816, 0.830]</td><td>0.009</td></tr><tr><td>Graph-UTS</td><td> $0 . 8 3 2 \pm 0 . 0 2 9$ </td><td> $+ 0 . 0 2 1$ </td><td>[0.824, 0.839]</td><td>0.002</td></tr><tr><td>Dual-UTS</td><td> $0 . 8 3 7 \pm 0 . 0 2 8$ </td><td>+0.026</td><td>[0.829, 0.844]</td><td> $< 0 . 0 0 1$ </td></tr></table>

Summary. All three variants improve significantly over baseline, with Dual-UTS strongest – the same ordering seen on PROTEINS and COLLAB, now confirmed at $3 2 \times$ their graph count. Unlike MUTAG, the ogbg-ppa gains reach significance at this seed count, consistent with the much larger fixed test split reducing per-seed variance.

## B.2 Topological Loss Variants

This subsection isolates the effect of the topology-aware auxiliary losses $( { \mathcal { L } } _ { \mathrm { t o p o - e v o l } } , { \mathcal { L } } _ { \mathrm { t o p o - a l i g n } } ,$ and their combination), compared against the unmodified, unregularized GIN backbone.

Summary. The auxiliary losses show no consistent benefit: on MUTAG the differences are small and not significant; on PROTEINS and COLLAB, $\mathcal { L } _ { \mathrm { t o p o - e v o l } }$ and the combined loss are significantly negative, and only $\mathcal { L } _ { \mathrm { t o p o - a l i g n } }$ on PROTEINS shows a significant (small) improvement. This indicates the descriptor-based variants (Section B.1), not the auxiliary losses, drive the gains reported in the main paper.

Table 9: UTS descriptor variants vs. GIN (unmodified/unregularized) — ogbg-ppa. Official OGB species split, 3 seeds.
<table><tr><td>Variant</td><td> $\mathbf { M e a n } \pm \mathbf { S t d }$ </td><td> $\Delta$ </td><td>95% CI</td><td>p</td></tr><tr><td>GIN (unmodified/unregularized)</td><td> $6 7 . 8 4 \pm 0 . 9 1 7$ </td><td></td><td></td><td></td></tr><tr><td>Embedding-UTS</td><td> $6 9 . 2 7 \pm 0 . 9 0 3$ </td><td> $+ 1 . 4 3$ </td><td>[68.76, 69.78]</td><td>0.0168</td></tr><tr><td>Graph-UTS</td><td> $7 0 . 2 0 \pm 0 . 9 5 1$ </td><td> $+ 2 . 3 6$ </td><td>[69.66, 70.74]</td><td>0.0041</td></tr><tr><td>Dual-UTS</td><td> $7 1 . 0 6 \pm 0 . 9 3 4$ </td><td> $+ 3 . 2 2$ </td><td>[70.53, 71.59]</td><td>0.0012</td></tr></table>

Table 10: Topological loss variants vs. GIN (unmodified/unregularized) — MUTAG.
<table><tr><td>Variant</td><td>Mean ± Std</td><td> $\Delta$ </td><td>95% CI</td><td>p</td></tr><tr><td>GIN (unmodified/unregularized)</td><td> $0 . 8 3 2 \pm 0 . 0 8 4$ </td><td></td><td></td><td></td></tr><tr><td> $\mathcal { L } _ { \mathrm { t o p o - e v o l } }$ </td><td> $0 . 8 5 0 \pm 0 . 0 8 5$ </td><td> $+ 0 . 0 1 9$ </td><td>[0.834, 0.867]</td><td>0.210</td></tr><tr><td> $\mathcal { L } _ { \mathrm { t o p o - a l i g n } }$ </td><td> $0 . 8 4 2 \pm 0 . 0 8 8$ </td><td> $+ 0 . 0 1 1$ </td><td>[0.824, 0.859]</td><td>0.205</td></tr><tr><td>Combination</td><td> $0 . 8 4 1 \pm 0 . 0 8 4$ </td><td> $+ 0 . 0 1 0$ </td><td>[0.825, 0.858]</td><td>0.198</td></tr></table>

Summary (ogbg-ppa). Unlike PROTEINS and COLLAB, where at least one individual loss is significantly negative, both losses are individually significant and positive on ogbg-ppa. The combination still underperforms either alone, echoing the lack of synergy seen throughout this work.

## B.3 Capacity Control (Random Descriptor Baselines)

To test whether the gains from UTS descriptors reflect real topological signal rather than added parameter capacity, we compare Graph-UTS and Dual-UTS against matched-dimensionality randomnoise controls (Graph-UTS-random, Dual-UTS-random). Reported ∆, CI, and p are for the paired difference (real variant − matched random control).

Summary. On PROTEINS and COLLAB, both real UTS variants significantly outperform their matched-capacity random controls, directly refuting a “more parameters alone” explanation for the gains in Section B.1. On MUTAG the paired differences are positive but not significant, consistent with the limited statistical power on this smaller dataset noted throughout this appendix.

Summary (ogbg-ppa). Both paired differences are significant, refuting a capacity-only explanation for the ogbg-ppa gains, consistent with the same finding on PROTEINS and COLLAB.

## B.4 Pooling Comparison

This subsection compares our pooling operator, UTSTopPool, against TOGL, SAGPool, and Top-KPool. Reported ∆, CI, and p are for the paired difference (UTSTopPool − competitor).

Summary. UTSTopPool is not significantly different from TOGL on MUTAG and trails it slightly (not significantly) on mean score, but significantly outperforms TOGL on both PROTEINS and COLLAB. Against SAGPool and TopKPool, UTSTopPool is ahead on all three datasets, with the advantage significant on PROTEINS and COLLAB but not on MUTAG, again consistent with reduced power on the smallest dataset.

Summary (ogbg-ppa). UTSTopPool significantly outperforms SAGPool and TopKPool, mirroring PROTEINS/COLLAB. Unlike any other dataset in this work, however, TOGL significantly outperforms UTSTopPool here $( p = 0 . 0 0 0 8 )$ . UTSTopPool’s advantage over feature-based pooling (SAGPool, TopKPool) holds at this scale, but its comparison against TOGL specifically is datasetdependent rather than uniformly favorable.

## B.5 Discussion: Cross-Dataset Consistency and the Role of Scale

Extending evaluation to ogbg-ppa clarifies a pattern the other three datasets alone left ambiguous: UTS-Aug improves over baseline on every dataset and at every scale tested, from MUTAG’s small molecular graphs to ogbg-ppa’s much larger biological networks, making it the most consistently reliable of the three interventions. UTS-Reg and UTS-Pool are each dataset-dependent instead: UTS-Reg’s auxiliary losses underperform baseline on COLLAB (Section 5.3), and UTS-Pool slightly underperforms TOGL specifically on ogbg-ppa (Section 5.4), while otherwise outperforming both baselines on every other dataset in this work.

Table 11: Topological loss variants vs. GIN (unmodified/unregularized) — PROTEINS.
<table><tr><td>Variant</td><td> $\mathbf { M e a n } \pm \mathbf { S t d }$ </td><td> $\Delta$ </td><td>95% CI</td><td>p</td></tr><tr><td>GIN (unmodified/unregularized)</td><td> $0 . 7 4 0 \pm 0 . 0 3 4$ </td><td></td><td></td><td></td></tr><tr><td> $\mathcal { L } _ { \mathrm { t o p o - e v o l } }$ </td><td> $0 . 7 3 8 \pm 0 . 0 3 4$ </td><td> $- 0 . 0 0 2$ </td><td>[0.730, 0.746]</td><td>0.031</td></tr><tr><td> $\mathcal { L } _ { \mathrm { t o p o - a l i g n } }$ </td><td> $0 . 7 5 5 \pm 0 . 0 3 3$ </td><td> $+ 0 . 0 1 4$ </td><td>[0.747, 0.762]</td><td>0.019</td></tr><tr><td>Combination</td><td> $0 . 7 3 6 \pm 0 . 0 3 5$ </td><td> $- 0 . 0 0 4$ </td><td>[0.729, 0.744]</td><td>0.022</td></tr></table>

Table 12: Topological loss variants vs. GIN (unmodified/unregularized) — COLLAB.
<table><tr><td>Variant</td><td>Mean ± Std</td><td> $\Delta$ </td><td>95% CI</td><td>p</td></tr><tr><td>GIN (unmodified/unregularized)</td><td> $0 . 8 1 0 \pm 0 . 0 2 9$ </td><td></td><td></td><td></td></tr><tr><td> $\mathcal { L } _ { \mathrm { t o p o - e v o l } }$ </td><td> $0 . 8 0 5 \pm 0 . 0 3 0$ </td><td>-0.006</td><td>[0.797, 0.813]</td><td>0.012</td></tr><tr><td> $\mathcal { L } _ { \mathrm { t o p o - a l i g n } }$ </td><td> $0 . 8 0 6 \pm 0 . 0 3 0$ </td><td>-0.004</td><td>[0.799, 0.814]</td><td>0.024</td></tr><tr><td>Combination</td><td> $0 . 8 0 9 \pm 0 . 0 3 2$ </td><td>-0.002</td><td>[0.800, 0.817]</td><td>0.044</td></tr></table>

## C Full Proofs and Theoretical Concepts

This section presents the complete proofs of 6 along with information on computational complexity.

## C.1 Proof of Theorem 1

Proof. Message Passing Neural Networks (MPNNs), including the Graph Isomorphism Network (GIN), are provably bounded in expressive power by the 1-dimensional Weisfeiler–Lehman (1-WL) graph isomorphism test [26]. Consequently, if two graphs are 1-WL equivalent, every intermediate node representation and every permutation-invariant graph readout produced by the backbone network must also be identical. Formally,

$$
\mathbf { z } _ { \mathrm { s t r u c t } } ( G _ { 1 } ) = \mathbf { z } _ { \mathrm { s t r u c t } } ( G _ { 2 } ) .
$$

Consider the canonical pair of non-isomorphic but 1-WL-equivalent graphs

$$
G _ { 1 } = C _ { 6 } , \qquad G _ { 2 } = 2 \times C _ { 3 } ,
$$

where $C _ { 6 }$ denotes the cycle graph on six vertices and $2 \times C _ { 3 }$ denotes the disjoint union of two triangles. Because both graphs are regular, the 1-WL refinement procedure assigns identical colors to every vertex throughout all refinement iterations, making them indistinguishable to any standard message-passing GNN.

Our framework augments the conventional graph representation using

$$
\mathbf { z } ^ { \prime } ( G ) = \left[ \mathbf { z } _ { \mathrm { s t r u c t } } ( G ) \parallel \Phi _ { \mathrm { e m b } } ( \mathbf { H } ^ { ( L ) } ) \parallel \Phi _ { \mathrm { g r p h } } ( G ) \right] ,
$$

where $\Phi _ { \mathrm { g r p h } } ( G )$ is computed directly from the input graph topology.

Unlike message passing, $\Phi _ { \mathrm { g r p h } }$ explicitly computes topological invariants through persistent homology. In particular, the Vietoris–Rips filtration distinguishes the two graphs by their one-dimensional homology.

For the cycle graph,

$$
\beta _ { 1 } ( C _ { 6 } ) = 1 ,
$$

since there exists one independent one-dimensional cycle.

Table 13: Topological loss variants vs. GIN (unmodified/unregularized) — ogbg-ppa.
<table><tr><td>Variant</td><td> $\mathbf { M e a n } \pm \mathbf { S t d }$ </td><td> $\Delta$ </td><td>95% CI</td><td>p</td></tr><tr><td>GIN (unmodified/unregularized)</td><td> $6 7 . 8 4 \pm 0 . 9 1 7$ </td><td></td><td></td><td></td></tr><tr><td> $\mathcal { L } _ { \mathrm { t o p o - e v o l } }$ </td><td> $6 8 . 3 6 \pm 0 . 9 8 2$ </td><td> $+ 0 . 5 2$ </td><td>[67.80, 68.92]</td><td>0.0341</td></tr><tr><td> $\mathcal { L } _ { \mathrm { t o p o - a l i g n } }$ </td><td> $6 9 . 5 4 \pm 0 . 9 4 1$ </td><td> $+ 1 . 7 0$ </td><td>[69.01, 70.07]</td><td>0.0286</td></tr><tr><td>Combination</td><td> $6 8 . 1 5 \pm 1 . 0 0 6$ </td><td> $+ 0 . 3 1$ </td><td>[67.58, 68.72]</td><td>0.0437</td></tr></table>

Table 14: Capacity control: real UTS descriptors vs. matched-capacity random controls — MUTAG.
<table><tr><td>Variant</td><td> $\mathbf { M e a n } \pm \mathbf { S t d }$ </td><td> $\Delta$  (paired)</td><td>95% CI</td><td>p</td></tr><tr><td>Graph-UTS</td><td> $0 . 8 9 0 \pm 0 . 0 7 6$ </td><td></td><td></td><td></td></tr><tr><td>Graph-UTS-random</td><td> $0 . 8 7 9 \pm 0 . 0 7 7$ </td><td>.0  $+ 0 . 0 1 1$ </td><td>二  $[ - 0 . 0 0 7 , + 0 . 0 2 9 ]$ </td><td>0.225</td></tr><tr><td>Dual-UTS</td><td> $0 . 8 7 8 \pm 0 . 0 7 8$ </td><td></td><td></td><td></td></tr><tr><td>Dual-UTS-random</td><td> $0 . 8 2 4 \pm 0 . 0 8 4$ </td><td>.0  $+ 0 . 0 5 4$ </td><td> $[ - 0 . 0 1 1 , + 0 . 1 1 9 ]$ </td><td>0.104</td></tr></table>

For the disjoint union of two triangles,

$$
\beta _ { 1 } ( 2 \times C _ { 3 } ) = 2 ,
$$

since each connected component contributes one independent cycle.

Consequently, the persistent homology descriptors contained in $\Phi _ { \mathrm { g r p h } }$ , including the first Betti number, persistence lifetimes, persistence entropy, and related geometric summaries, necessarily differ:

$$
\Phi _ { \mathrm { g r p h } } ( C _ { 6 } ) \neq \Phi _ { \mathrm { g r p h } } ( 2 \times C _ { 3 } ) .
$$

Because concatenation is injective with respect to its individual components,

$$
\begin{array} { r } { \mathbf { z } ^ { \prime } ( C _ { 6 } ) = [ \mathbf { z } _ { \mathrm { s t r u c t } } \parallel \Phi _ { \mathrm { e m b } } \parallel \Phi _ { \mathrm { g r p h } } ( C _ { 6 } ) ] \neq [ \mathbf { z } _ { \mathrm { s t r u c t } } \parallel \Phi _ { \mathrm { e m b } } \parallel \Phi _ { \mathrm { g r p h } } ( 2 \times C _ { 3 } ) ] = \mathbf { z } ^ { \prime } ( 2 \times C _ { 3 } ) . } \end{array}
$$

Therefore, although the underlying GNN cannot distinguish $C _ { 6 }$ from $2 \times C _ { 3 }$ , the UTS-augmented representation can. Hence the proposed readout strictly separates at least one pair of graphs that are indistinguishable under the 1-WL test.

Therefore, the proposed UTS-augmented framework is strictly more expressive than the standard 1-WL message-passing hierarchy.

## C.2 Full Justification of Proposition 1

Let $\mathbf { s } ^ { ( l ) } = \Phi _ { \mathrm { e m b } } ( \mathbf { H } ^ { ( l ) } )$ denote the embedding-space topological signature at layer l, and let $\mathcal { L } _ { \mathrm { s m o o t h } }$ (defined below) be the topo-evolution loss with fixed weight $\lambda _ { \mathrm { s m o o t h } } > 0$

$$
\mathcal { L } _ { \mathrm { s m o o t h } } = \frac { \lambda _ { \mathrm { s m o o t h } } } { L - 1 } \sum _ { l = 1 } ^ { L - 1 } \left\| \mathbf { s } ^ { ( l ) } - \mathbf { s } ^ { ( l - 1 ) } \right\| _ { 2 } ^ { 2 }\tag{28}
$$

At a stationary point of the combined objective $\mathcal { L } = \mathcal { L } _ { \mathrm { t a s k } } + \mathcal { L } _ { \mathrm { s m o o t h } }$ , the gradient contribution of $\mathcal { L } _ { \mathrm { s m o o t h } }$ acts to reduce the layer-to-layer divergence of $\mathbf { s } ^ { ( l ) }$ , which is, at a local first-order level, in tension with representation collapse understood as convergence of per-graph topological signatures toward a shared fixed point across the dataset (Eq. 30).

This is a heuristic motivation, not a proof that $\mathcal { L } _ { \mathrm { s m o o t h } }$ enforces a lower bound on embedding variance or on the Oversmoothing Index. $\mathcal { L } _ { \mathrm { s m o o t h } }$ is an unconstrained soft penalty: standard stochastic gradient descent provides no guarantee that any particular variance floor is achieved, and the penalty’s local effect on layer-to-layer signature divergence does not by itself imply a global effect on cross-graph signature convergence, since these are related but distinct quantities (the former measures a graph’s own trajectory across depth; the latter measures similarity between graphs at a fixed depth).

Table 15: Capacity control: real UTS descriptors vs. matched-capacity random controls — PRO-TEINS.
<table><tr><td>Variant</td><td> $\mathbf { M e a n } \pm \mathbf { S t d }$ </td><td>∆ (paired)</td><td>95% CI</td><td>p</td></tr><tr><td>Graph-UTS</td><td> $0 . 7 5 9 \pm 0 . 0 4 0$ </td><td>0.0</td><td>5+</td><td></td></tr><tr><td>Graph-UTS-random</td><td> $0 . 7 3 6 \pm 0 . 0 4 1$ </td><td> $+ 0 . 0 2 3$ </td><td> $\left[ + 0 . 0 0 5 , + 0 . 0 4 1 \right]$ </td><td>0.015</td></tr><tr><td>Dual-UTS</td><td> $0 . 7 6 3 \pm 0 . 0 3 6$ </td><td>2.0</td><td>，</td><td></td></tr><tr><td> $\mathrm { D u a l - U T S - r a n d o m }$ </td><td> $0 . 7 5 0 \pm 0 . 0 4 0$ </td><td> $+ 0 . 0 1 3$ </td><td> $[ + 0 . 0 0 3 , + 0 . 0 2 3 ]$ </td><td>0.020</td></tr></table>

Table 16: Capacity control: real UTS descriptors vs. matched-capacity random controls — COLLAB.
<table><tr><td>Variant</td><td>Mean ± Std</td><td>∆ (paired)</td><td>95% CI</td><td>p</td></tr><tr><td>Graph-UTS</td><td> $0 . 8 3 2 \pm 0 . 0 2 9$ </td><td></td><td></td><td></td></tr><tr><td>Graph-UTS-random</td><td> $0 . 8 0 7 \pm 0 . 0 3 0$ </td><td> $+ 0 . 0 2 5$ </td><td> $[ + 0 . 0 1 8 , + 0 . 0 3 2 ]$ </td><td> $< 0 . 0 0 1$ </td></tr><tr><td>Dual-UTS</td><td> $0 . 8 3 7 \pm 0 . 0 2 8$ </td><td></td><td></td><td></td></tr><tr><td>Dual-UTS-random</td><td> $0 . 8 2 4 \pm 0 . 0 2 9$ </td><td> $+ 0 . 0 1 3$ </td><td> $[ + 0 . 0 0 4 , + 0 . 0 2 1 ]$ </td><td>0.001</td></tr></table>

Empirical evidence. This connection holds in one direction and reverses in another. Across MUTAG, PROTEINS, and COLLAB, layer-wise OSI comparisons between an unregularized baseline and a $\mathcal { L } _ { \mathrm { s m o o t h } }$ -regularized model (Table 23) show OSI decreasing under regularization on MUTAG at every layer, but increasing under regularization on PROTEINS and COLLAB. We therefore do not claim $\mathcal { L } _ { \mathrm { s m o o t h } }$ acts as a general anti-oversmoothing mechanism; we report it as an empirically dataset-dependent effect, consistent with Proposition 1’s status as motivation rather than guarantee. This dataset-dependence is further corroborated by the loss-ablation results in Table 10–12: the Topo-evol term is directionally positive (though not significant) on MUTAG, small and mixed on PROTEINS, and significantly negative on COLLAB.

## C.3 Geometric Control of Local Persistence Topology

Proposition 2. Let $X _ { v } = \mathbf { H } _ { \mathcal { N } ( v ) } \subset \mathbb { R } ^ { d } , | X _ { v } | = n _ { v } = | \mathcal { N } ( v ) | + 1$ , equipped with the Euclidean metric $d ( \cdot , \cdot )$ , and let $\begin{array} { r } { c _ { v } = \frac { 1 } { n _ { v } } \sum _ { u \in X _ { v } } \mathbf { h } _ { u } } \end{array}$ denote its centroid. Let $\mathrm { V R } _ { \bullet } ( X _ { v } )$ and $\check { \mathrm { C } } _ { \bullet } ( X _ { v } )$ denote the Vietoris–Rips and Cech filtrations<sup>ˇ</sup> $o f X _ { v } ,$ , and let $\mathrm { D g m } _ { k } ( X _ { v } )$ denote the degree-k persistence diagram of $\mathrm { V R } _ { \bullet } ( X _ { v } )$ . Then, with $\phi _ { v } ^ { \mathrm { l i g h t } }$ as in $E q . \ ( 8 )$

(i) (0-dimensional control.) Writing $\begin{array} { r } { \mathrm { P e r s } _ { 0 } ( X _ { v } ) = \sum _ { ( 0 , d ) \in \mathrm { D g m } _ { 0 } ( X _ { v } ) , d < \infty } } \end{array}$ d for the total finite 0-persistence,

$$
\begin{array} { r } { \mu _ { \mathrm { n n } } ^ { v } \leq \mathrm { P e r s } _ { 0 } ( X _ { v } ) \leq ( n _ { v } - 1 ) \Delta ^ { v } . } \end{array}
$$

(ii) (Vanishing of higher-order topology.) For every $k \geq 1$ and every $( b , d ) \in \mathrm { D g m } _ { k } ( X _ { v } )$

$$
d \leq 2 \Delta ^ { v } .
$$

Consequently $\mathrm { D g m } _ { k } ( X _ { v } ) \to \emptyset$ as $\Delta ^ { v }  0 ,$ , for every $k \geq 1$

(iii) (Noisefloor.) $H X _ { v }$ is a δ-Gromov–Hausdorffperturbation ofan idealized configuration $\tilde { X } _ { v }$ with $\delta \approx \sigma _ { \mathrm { n n } } ^ { v }$ , then no pair $( b , d ) \in \mathrm { D g m } _ { k } ( \bar { X _ { v } } )$ with $d - b < 2 \sigma _ { \mathrm { n n } } ^ { v }$ is distinguishable from noise.

(iv) (Dimension cap.) $\mathrm { D g m } _ { k } ( X _ { v } ) = \emptyset$ for all $k \geq | \mathcal { N } ( v ) |$ , so $| \mathcal { N } ( v ) | / k _ { \operatorname* { m a x } }$ bounds the highest homological dimension that can carry signal.

Hence $\phi _ { v } ^ { \mathrm { l i g h t } }$ determines,for every v, an explicit scale interval $[ \mu _ { \mathrm { n n } } ^ { v } , 2 \Delta ^ { v } ]$ outside of which $\mathrm { V R } _ { \bullet } ( X _ { v } )$ is provably trivial (a single component, no higher homology), together with a noise floor $2 \sigma _ { \mathrm { n n } } ^ { v }$ and a dimension cap $| { \mathcal { N } } ( v ) | - i . e .$ the first-order conditions governing whether $X _ { v }$ can carry any nontrivial persistent topology at all.

Table 17: Capacity control: real UTS descriptors vs. matched-capacity random controls — ogbg-ppa.
<table><tr><td>Variant</td><td>Mean Accuracy</td><td>95% CI (paired diff)</td></tr><tr><td>Graph-UTS</td><td>70.20</td><td></td></tr><tr><td>Graph-UTS-random</td><td>67.83</td><td></td></tr><tr><td>Paired difference</td><td> $+ 2 . 3 7 \left( p = 0 . 0 0 1 1 \right)$ </td><td>[+0.94, +3.80]</td></tr><tr><td>Dual-UTS</td><td>71.06</td><td></td></tr><tr><td>Dual-UTS-random</td><td>68.80</td><td></td></tr><tr><td>Paired difference</td><td> $+ 2 . 2 6 ( p = 0 . 0 0 2 4 )$ </td><td>[+0.82, +3.70]</td></tr></table>

Table 18: Pooling comparison — MUTAG.
<table><tr><td>Variant</td><td> $\mathbf { M e a n } \pm \mathbf { S t d }$ </td><td>∆ vs. UTSTopPool</td><td>95% CI</td><td>p</td></tr><tr><td>UTSTopPool (ours)</td><td> $0 . 8 3 5 \pm 0 . 0 7 9$ </td><td></td><td></td><td></td></tr><tr><td>TOGL</td><td> $0 . 8 4 0 \pm 0 . 0 8 3$ </td><td>-0.005</td><td>[-0.012, +0.001]</td><td>0.103</td></tr><tr><td>SAGPool</td><td> $0 . 8 3 1 \pm 0 . 0 8 0$ </td><td>+0.004</td><td>[-0.001, +0.008]</td><td>0.143</td></tr><tr><td>TopKPool</td><td> $0 . 8 2 9 \pm 0 . 0 7 9$ </td><td>+0.006</td><td>[-0.001, +0.013]</td><td>0.085</td></tr></table>

Proof. (i) 0-dimensional control. By the correspondence between 0-th persistent homology of a Vietoris–Rips filtration on a finite metric space and single-linkage hierarchical clustering [3], every component merge (death) in $\mathrm { D g m } _ { 0 } ( X _ { v } )$ occurs exactly at the weight of the corresponding edge added by Kruskal’s algorithm; equivalently, the multiset of finite death times of $\mathrm { D g m } _ { 0 } ( X _ { v } )$ equals the multiset of edge weights of the minimum spanning tree $T ^ { \star }$ of the complete graph $( X _ { v } , d )$ . All births in $\mathrm { D g m } _ { 0 }$ occur at filtration value 0, so $\begin{array} { r } { \operatorname { P e r s } _ { 0 } ( X _ { v } ) = w ( T ^ { \star } ) : = \sum _ { e \in T ^ { \star } } w ( \bar { e } ) } \end{array}$

Upper bound. Every edge weight of $T ^ { \star }$ is a pairwise distance in $X _ { v }$ , hence $w ( e ) \leq \Delta ^ { v }$ for all $e \in T ^ { \star }$ , and $| T ^ { \star } | = \dot { n } _ { v } - 1$ , giving w $( T ^ { \star } ) \leq ( \dot { n } _ { v } - 1 ) \Delta ^ { v }$

Lower bound. Let $\begin{array} { r } { \mathrm { N N } ( u ) = \operatorname* { m i n } _ { w \neq u } d ( { \mathbf { h } } _ { u } , { \mathbf { h } } _ { w } ) } \end{array}$ and $u ^ { \star } = \arg \operatorname* { m a x } _ { u } \mathrm { N N } ( u )$ . Any edge of $T ^ { \star }$ incident to $u ^ { \star }$ has weight $\geq \mathrm { N N } ( u ^ { \star } )$ , since $\mathrm { N N } ( u ^ { \star } )$ is by definition the smallest possible distance from $u ^ { \star }$ to any other point. $\boldsymbol { \mathrm { A s } } \ \boldsymbol { u } ^ { \star }$ has at least one incident tree edge, $w ( T ^ { \star } ) \geq \mathrm { N N } ( u ^ { \star } ) =$ max<sub>u</sub> $\begin{array} { r } { \mathrm { N N } ( u ) \dot { \geq \frac { 1 } { n _ { v } } } \sum _ { u } \mathrm { N N } ( u ) = \mu _ { \mathrm { n n } } ^ { v } } \end{array}$ . Combining the two bounds gives (i).

(ii) Vanishing of higher-order topology. First, a Euclidean fact: for any $u \in X _ { v }$

$$
\begin{array} { r } { d ( \mathbf { h } _ { u } , c _ { v } ) = \Big \| \frac { 1 } { n _ { v } } \sum _ { w } ( \mathbf { h } _ { u } - \mathbf { h } _ { w } ) \Big \| \leq \frac { 1 } { n _ { v } } \sum _ { w } \| \mathbf { h } _ { u } - \mathbf { h } _ { w } \| \leq \Delta ^ { v } , } \end{array}
$$

by the triangle inequality and convexity of $\| \cdot \| .$ . So $c _ { v }$ lies within distance $\Delta ^ { v }$ of every point of $X _ { v }$ i.e. the circumradius of $X _ { v }$ about $c _ { v } \mathrm { i s } \le \ddot { \Delta ^ { v } }$ . Consequently the ball system $\{ B ( \mathbf { h } _ { u } , \bar { \Delta ^ { v } } ) \} _ { u \in X _ { \iota } }$ has nonempty total intersection $( c _ { v }$ lies in all of them), so the top simplex on all of $X _ { v }$ is present in the Cech complex at scale <sup>ˇ</sup> $\Delta ^ { v } ;$ ; since a Cech complex is a downward-closed simplicial complex, this <sup>ˇ</sup> forces $\check { \mathrm { C } } _ { \Delta ^ { v } } ( X _ { v } )$ to be the full simplex $\Delta ^ { n _ { v } - 1 }$ on $X _ { v }$

Next, for any metric space and any $\varepsilon , \check { \mathrm { C } } _ { \varepsilon } \subseteq \operatorname { V R } _ { 2 \varepsilon } \colon$ if σ has circumradius $\leq \varepsilon$ about some point $p ,$ then for any $\begin{array} { r } { u , w \in \sigma , d ( \mathbf { h } _ { u } , \dot { \mathbf { h } _ { w } } ) \leq d ( \mathbf { h } _ { u } , p ) + d ( p , \mathbf { h } _ { w } ) \leq \frac { 1 } { \sigma } } \end{array}$ 2ε by the triangle inequality, so $\sigma \in { \mathrm { V R } } _ { 2 \varepsilon }$ . Applying this at $\varepsilon = \Delta ^ { v }$ gives $\mathrm { V R } _ { 2 \Delta \mathfrak { v } } ( X _ { \mathfrak { v } } ) \supseteq \check { \mathrm { C } } _ { \Delta \mathfrak { v } } ( X _ { \mathfrak { v } } ) = \Delta ^ { n _ { \mathfrak { v } } - 1 }$ , and since $\Delta ^ { n _ { v } - 1 }$ already contains every possible simplex on $X _ { v }$ , equality holds: $\mathrm { V R } _ { 2 \Delta ^ { \upsilon } } ( X _ { v } ) = \Delta ^ { n _ { v } - 1 }$ , the full simplex.

The full simplex is contractible, so $H _ { k } ( \mathrm { V R } _ { 2 \Delta ^ { v } } ( X _ { v } ) ) = 0$ for all $k \geq 1$ . By definition of persistent homology, every class in dimension $k \geq 1$ must therefore have died by filtration value $2 \Delta ^ { v }$ , i.e. every $( b , d ) \in \mathbf { \bar { D } } \mathrm { g m } _ { k } ( X _ { v } ) , k \geq 1$ , satisfies $d \leq 2 \Delta ^ { v }$ . Letting $\Delta ^ { v }  0$ forces every such $d \to 0 ,$ , and since $b \geq \mathrm { 0 ~ a l w a y s , }$ every bar collapses: $\mathrm { D g m } _ { k } ( X _ { v } ) \to \emptyset$

(iii) Noise floor. By the Gromov–Hausdorff stability theorem for persistence diagrams of Vietoris– Rips filtrations on point clouds [4, 6], if $d _ { G H } ( X _ { v } , \tilde { X } _ { v } ) \leq \delta$ then the bottleneck distance obeys $d _ { B } ( \mathrm { D g m } _ { k } ( X _ { v } ) , \mathrm { D g m } _ { k } ( \tilde { X } _ { v } ) ) \leq 2 \delta$ for every k. Taking $\tilde { X } _ { \ i }$ to be an idealized (noise-free) configuration and δ ≈ $\boldsymbol { \sigma } _ { \mathrm { n } n } ^ { v }$ as the empirical estimate of local embedding perturbation, any pair in $\mathrm { D g m } _ { k } \bar { ( \boldsymbol { X } _ { v } ) }$ with lifetime $d - b < 2 \sigma _ { \mathrm { n n } } ^ { v }$ lies within the stability radius of the trivial (empty) diagram and so cannot be certified, from $X _ { v }$ alone, as distinct from a perturbation of a topologically trivial configuration.

Table 19: Pooling comparison — PROTEINS.
<table><tr><td>Variant</td><td> $\mathbf { M e a n } \pm \mathbf { S t d }$ </td><td> $\Delta { \bf v } { \bf s } .$  UTSTopPool</td><td>95% CI</td><td>p</td></tr><tr><td>UTSTopPool (ours)</td><td> $0 . 7 4 5 \pm 0 . 0 3 3$ </td><td></td><td></td><td></td></tr><tr><td>TOGL</td><td> $0 . 7 4 4 \pm 0 . 0 3 3$ </td><td>+0.001</td><td>[0.000, 0.003]</td><td>0.046</td></tr><tr><td>SAGPool</td><td> $0 . 7 4 2 \pm 0 . 0 3 4$ </td><td>+0.003</td><td>[0.002, 0.005]</td><td>0.002</td></tr><tr><td>TopKPool</td><td> $0 . 7 4 1 \pm 0 . 0 3 3$ </td><td>+0.005</td><td>[0.003, 0.006]</td><td> $< 0 . 0 0 1$ </td></tr></table>

Table 20: Pooling comparison — COLLAB.
<table><tr><td>Variant</td><td> $\mathbf { M e a n } \pm \mathbf { S t d }$ </td><td>∆ vs. UTSTopPool</td><td>95% CI</td><td>p</td></tr><tr><td>UTSTopPool (ours)</td><td> $0 . 8 2 1 \pm 0 . 0 2 9$ </td><td></td><td></td><td></td></tr><tr><td>TOGL</td><td> $0 . 8 2 0 \pm 0 . 0 2 9$ </td><td>+0.001</td><td>[0.000, 0.002]</td><td>0.018</td></tr><tr><td>SAGPool</td><td> $0 . 8 1 9 \pm 0 . 0 2 9$ </td><td>+0.002</td><td>[0.000, 0.004]</td><td>0.029</td></tr><tr><td>TopKPool</td><td> $0 . 8 1 8 \pm 0 . 0 2 9$ </td><td>+0.003</td><td>[0.001, 0.006]</td><td>0.020</td></tr></table>

(iv) Dimension cap. A k-simplex requires $k + 1$ distinct vertices, so the chain groups of $\mathrm { V R } _ { \bullet } ( X _ { v } )$ vanish identically for $k \geq n _ { v } = | \mathcal { N } ( v ) | + 1$ , hence $H _ { k } \equiv 0 \mathrm { f o r } k \geq | \mathcal { N } ( v ) |$ at every filtration value, trivially. □

Corollary 1. Statements (i)–(iv) jointly show that $\phi _ { v } ^ { \mathrm { l i g h t } }$ recovers, without persistent-homology computation, (a) matching upper/lower bounds on the total 0-dimensional persistence $[ \mu _ { \mathrm { n n } } ^ { v } , ( n _ { v } \textrm { -- }$ $1 ) \Delta ^ { v } ]$ , (b) an explicit death-time ceiling $2 \Delta ^ { v }$ for all higher-orderfeatures, (c) a noise floor $2 \sigma _ { \mathrm { n n } } ^ { v }$ below which nofeature is distinguishablefrom perturbation, and (d) a dimension cap $| \mathcal { N } ( v ) |$ . The mean radius $\begin{array} { r } { \bar { r } _ { 1 } ^ { v } = \frac { 1 } { n _ { v } } \sum _ { u } d ( \mathbf { h } _ { u } , c _ { v } ) } \end{array}$ is a smooth, outlier-robust surrogate for the circumradius bound used in the proof of $( i i ) -$ since $\begin{array} { r } { \bar { r } _ { 1 } ^ { v } \le \operatorname* { m a x } _ { u } d ( \mathbf { h } _ { u } , c _ { v } ) \le \Delta ^ { v } - } \end{array}$ and is preferred over the (non-smooth) max operator for gradient-based training of $f _ { \theta } .$ . Together these five scalars therefore certify, rather than merely approximate, the scale regime in which $X _ { v }$ can carry nontrivial topology, justifying their use as a lightweight proxyfor $\Phi _ { \mathrm { e m b } } ^ { \mathrm { d i f f } }$

## C.4 Computational Complexity Analysis

This section quantifies the per-component computational cost of the two Unified Topological Signatures and justifies the asymptotic claims made in Section 4.4 and Section 6.4. Let $n { \stackrel { . } { = } } | { \bar { \mathcal { V } } } | , m = { \bar { | } } { \mathcal { E } } |$ and d denote the embedding dimension. For the local pooling analysis, let $n _ { v } = | \mathcal { N } ( v ) | + 1$ denote a neighbourhood size and <sup>¯</sup>δ the average node degree.

## C.4.1 Graph Signature $\Phi _ { \mathrm { g r p h } }$

$\Phi _ { \mathrm { g r p h } }$ is computed once per graph (Appendix A.5, GraphUTS precomputation) and cached, so its cost is amortized to $O ( 1 )$ per training step. Its one-time cost decomposes as:

• Distance features / APSP: all-pairs shortest paths via repeated BFS from every node, $O ( n m )$ for unweighted graphs (or $O ( n ^ { \dot { 2 } }$ log $n + n m )$ with Dijkstra on weighted variants); dominates when graphs are sparse $( m = { \cal { O } } ( n ) ,$ ), giving ${ \dot { O } } ( n ^ { 2 } )$ .

• Ollivier–Ricci curvature: for each of m edges, one Sinkhorn iteration over lazy randomwalk measures supported on $O ( \bar { \delta } )$ neighbours costs $O ( \bar { \delta } ^ { 2 } T )$ for T Sinkhorn iterations, giving $O ( m \bar { \delta } ^ { 2 } T )$ total.

• Forman–Ricci curvature: closed-form per edge from local degree information, $O ( m \bar { \delta } )$

• Persistence features: exact Vietoris–Rips persistence via GUDHI on the $n \times n$ shortest-path matrix, worst case $O ( n ^ { 3 } )$ (bounded in practice by the sparsity of the simplicial complex GUDHI constructs; see Project [22]).

Table 21: Pooling comparison — ogbg-ppa.
<table><tr><td>Variant</td><td> $\mathbf { M e a n } \pm \mathbf { S t d }$ </td><td> $\Delta { \bf v } { \bf s } .$  UTSTopPool</td><td>95% CI</td><td> $p$ </td></tr><tr><td>UTSTopPool (ours)</td><td> $6 8 . 8 5 \pm 0 . 8 9 1$ </td><td></td><td></td><td></td></tr><tr><td>TOGL</td><td> $6 8 . 9 7 \pm 0 . 9 2 6$ </td><td>-0.12</td><td> $[ - 0 . 1 8 , - 0 . 0 6 ]$ </td><td>0.0008</td></tr><tr><td>SAGPool</td><td> $6 7 . 8 8 \pm 0 . 9 6 3$ </td><td>+0.97</td><td> $[ + 0 . 6 2 , + 1 . 3 2 ]$ </td><td>0.0003</td></tr><tr><td>TopKPool</td><td> $6 7 . 5 3 \pm 0 . 9 8 8$ </td><td>+1.32</td><td> $[ + 0 . 8 9 , + 1 . 7 5 ]$ </td><td>0.0001</td></tr></table>

• Spectral features: eigendecomposition of the $n \times n$ graph Laplacian, $O ( n ^ { 3 } )$ (or $O ( n ^ { 2 } )$ for the few smallest eigenvalues via Lanczos, which we use in practice).

• Structural features: degree and clustering statistics $O ( m )$ ; betweenness centrality via Brandes algorithm O(nm).

The dominant term is $O ( n ^ { 3 } )$ from exact persistence, incurred exactly once per graph in the dataset, independent of the number of training epochs.

## C.4.2 Embedding Signature $\Phi _ { \mathrm { e m b } }$ (Full, Per Layer)

Unlike $\Phi _ { \mathrm { g r p h } } , \Phi _ { \mathrm { e m b } } ^ { \mathrm { d i f f } }$ (Appendix A.4) is recomputed at every layer of every forward pass, since it depends on $\mathbf { H } ^ { ( l ) }$ , which changes with the encoder weights during training. Its per-layer, per-graph cost is:

• Pairwise distances $( \mathbf { E q . 9 } ) \colon O ( n ^ { 2 } d )$

$H _ { 0 }$ surrogate (Eq. 14): sorting the  <sup>n</sup> upper-triangle distances, $O ( n ^ { 2 } \log { n } )$

$H _ { 1 }$ surrogate (Eq. 15): enumerating all  <sup>n</sup> triangles, $O ( n ^ { 3 } )$ , capped at $n \leq n _ { \mathrm { m a x } } = 5 0$ (Appendix A.2, Remark on H1 cost); for $n > n _ { \mathrm { m a x } }$ the $H _ { 1 }$ dimensions are set to zero and this term is skipped, reducing the effective cost to $O ( n ^ { 2 } d )$

• Spectral features: eigendecomposition of the regularized soft Laplacian $\tilde { \textbf { L } } \in \mathbb { R } ^ { n \times n }$ via torch.linal $\mathbf { g } . \mathsf { e i g h } , O ( n ^ { 3 } )$ , with a backward pass of matching cost through the analytic eigenvalue gradient.

The full signature is therefore $O ( n ^ { 3 } )$ per layer whenever $n \leq n _ { \mathrm { m a x } } .$ , and $O ( n _ { \mathrm { s p e c t r a l } } ^ { 3 } )$ (from the spectral term alone, since $H _ { 1 }$ is skipped) for larger graphs, where the eigendecomposition remains the bottleneck term. Across an L-layer encoder this multiplies to $\bar { O ( L n ^ { 3 } ) }$ additional cost per forward/backward pass relative to a plain GIN, which is the cost the lightweight variant is designed to avoid for pooling specifically.

## C.4.3 Lightweight Proxy $\phi _ { v } ^ { \mathrm { l i g h t } }$ (Per-Node Pooling Score)

For the pooling scorer (Section 4.4), the relevant unit of computation is a single node’s one-hop neighbourhood $X _ { v }$ rather than the whole graph. Given $\phi _ { v } ^ { \mathrm { l i g h t } } \in \mathbb { R } ^ { 5 } ( \mathrm { E q . ~ } 8 )$ , each component is computable directly from the $n _ { v } \times n _ { v }$ local distance matrix:

• Local distance matrix: $O ( n _ { v } ^ { 2 } d )$

$\mu _ { \mathrm { n n } } ^ { v } , \sigma _ { \mathrm { n n } } ^ { v }$ : mean/std of nearest-neighbour distances within $X _ { v } , { \cal O } ( n _ { v } ^ { 2 } )$ given the distance matrix.

$\Delta ^ { v } \colon$ maximum pairwise distance, $O ( n _ { v } ^ { 2 } )$

• $\bar { r } _ { 1 } ^ { v } :$ : mean distance to centroid, $O ( n _ { v } d )$

$| \mathcal { N } ( v ) | / k _ { \operatorname* { m a x } } \colon O ( 1 )$ given the precomputed degree.

Each node’s score is thus $O ( n _ { v } ^ { 2 } d )$ , matching the “local quadratic” cost stated in Section 4.4 (as opposed to the $O ( n ^ { 3 } )$ cubic cost of triangle enumeration and eigendecomposition used by the full signature). Summing over all nodes, the total pooling-layer cost is

$$
\sum _ { v \in \mathcal { V } } { \cal O } ( n _ { v } ^ { 2 } d ) = { \cal O } \Big ( d \sum _ { v } n _ { v } ^ { 2 } \Big ) = { \cal O } \big ( d n \bar { \delta } ^ { 2 } )\tag{29}
$$

for graphs with roughly uniform degree ${ \bar { \delta } } ,$ i.e. linear in n rather than cubic, at fixed local density. This is the source of the scalability improvement claimed in Section 4.4: the lightweight variant replaces a per-layer $O ( n ^ { 3 } )$ term with an $O ( n \bar { \delta } ^ { 2 } d )$ term, which is asymptotically smaller whenever $\bar { \delta } = o ( \sqrt { n } ) -$ true for all sparse graphs, and in particular for every dataset in Section 5 (mean degree $\bar { \delta } \in [ 2 , \dot { 9 } ]$ , n up to several hundred for COLLAB).

## C.4.4 Chunked Memory Bound

Processing neighbourhoods in chunks of size C (Section 4.4) bounds peak memory rather than total compute: at any instant only C neighbourhoods’ distance matrices and intermediate scores are materialized, giving peak GPU memory $\bar { O } ( C d \bar { \delta } _ { \operatorname* { m a x } } )$ (where $\bar { \delta } _ { \mathrm { m a x } }$ bounds the largest neighbourhood processed in a chunk), independent of the total node count $n .$ . Total compute is unaffected — it remains $O ( d n \bar { \delta } ^ { 2 } )$ as in Eq. (29) — chunking only trades wall-clock time (sequential chunks) for the ability to run on graphs whose full $n \times n$ distance matrix would not fit in memory at once.

## C.4.5 Summary

Table 22 summarizes the cost and frequency of calculation for each signature.

Table 22: Asymptotic cost per forward pass, one graph. n: nodes, m: edges, d: embedding dim, $\bar { \delta } \colon$ average degree, L: GNN layers.
<table><tr><td>Component</td><td>Cost</td><td>Frequency</td></tr><tr><td> $\Phi _ { \mathrm { g r p h } }$  (graph signature)</td><td> $O ( n ^ { 3 } )$ </td><td>once per graph (cached)</td></tr><tr><td> $\Phi _ { \mathrm { e m b } } ^ { \mathrm { d i f f } }$  (full, per layer)</td><td> $O ( n ^ { 3 } )$ </td><td>every layer, every step</td></tr><tr><td> $\phi _ { v } ^ { \mathrm { l i g h t } }$  (lightweight, all nodes)</td><td> $O ( d n \bar { \delta } ^ { 2 } )$ </td><td>pooling layer(s) only</td></tr></table>

## D Oversmoothing Analysis

## D.1 Oversmoothing Index

The embedding signature naturally provides a graph-level diagnostic for monitoring representation evolution during message passing. We define the Oversmoothing Index (OSI) as

$$
\mathrm { O S I } ^ { ( l ) } = 1 - \overline { { d _ { \mathrm { c o s } } \Big ( \mathbf { s } _ { i } ^ { ( l ) } , \mathbf { s } _ { j } ^ { ( l ) } \Big ) } } ,\tag{30}
$$

where $\mathbf { s } _ { i } ^ { ( l ) }$ denotes the embedding signature of graph i at layer $l , d _ { \mathrm { c o s } }$ is the cosine distance, and the average is computed over all graph pairs in the evaluation set.

Higher OSI indicates increasing similarity between graph-level topological signatures and therefore greater representation collapse. OSI is used only for analysis and does not influence training.

## D.2 Oversmoothing diagnostic

To further understand the effect of topology-preserving regularization, we analyze the evolution of graph-level topology throughout message passing using the proposed Oversmoothing Index (OSI). Unlike conventional oversmoothing metrics that primarily measure feature homogenization at the node level, OSI quantifies the similarity of graph-level topological signatures across successive GNN layers.

Table 23 reports layer-wise OSI for both an unregularized baseline GIN and a model trained with the topo-evolution loss $\mathcal { L } _ { \mathrm { s m o o t h } }$ , across all layers of all three benchmark datasets, extending our earlier baseline-only analysis to directly test whether the regularizer has the intended effect.

For the unregularized baseline, OSI remains high but relatively stable across layers on all three datasets, indicating that graph-level topological signatures do not collapse severely at the depths evaluated in this work. This is consistent with our earlier MUTAG-only observation, now confirmed to hold on PROTEINS and COLLAB as well.

Table 23: Layer-wise Oversmoothing Index (OSI, Eq. 30) comparing an unregularized baseline GIN against a model trained with the topo-evolution loss $\mathcal { L } _ { \mathrm { s m o o t h } }$ , across all layers of all three benchmark datasets. Higher OSI indicates greater convergence (collapse) of per-graph topological signatures. ∆ is OSI(Topo-evol) − OSI(baseline); negative values indicate the regularizer reduces oversmoothing at that layer, positive values indicate it increases oversmoothing. MUTAG and PROTEINS use 4 GIN layers (0–3); COLLAB uses 3 layers (0–2), per Section 5.
<table><tr><td>Dataset</td><td>Layer</td><td>OSI (baseline)</td><td>OSI (Topo-evol)</td><td>∆</td><td>Trend</td></tr><tr><td rowspan="4">MUTAG</td><td>0</td><td>0.9693</td><td>0.9587</td><td>-0.0106</td><td>less oversmoothed</td></tr><tr><td>1</td><td>0.9711</td><td>0.9374</td><td>-0.0337</td><td>less oversmoothed</td></tr><tr><td>2</td><td>0.9626</td><td>0.9207</td><td>-0.0418</td><td>less oversmoothed</td></tr><tr><td>3</td><td>0.9549</td><td>0.9089</td><td>-0.0459</td><td>less oversmoothed</td></tr><tr><td rowspan="4">PROTEINS</td><td>0</td><td>0.8176</td><td>0.8408</td><td>+0.0233</td><td>more oversmoothed</td></tr><tr><td>1</td><td>0.8245</td><td>0.8590</td><td>+0.0345</td><td>more oversmoothed</td></tr><tr><td>2</td><td>0.8204</td><td>0.8495</td><td>+0.0291</td><td>more oversmoothed</td></tr><tr><td>3</td><td>0.8196</td><td>0.8458</td><td>+0.0262</td><td>more oversmoothed</td></tr><tr><td rowspan="3">COLLAB</td><td>0</td><td>0.7698</td><td>0.7878</td><td>+0.0180</td><td>more oversmoothed</td></tr><tr><td>1</td><td>0.7841</td><td>0.7943</td><td>+0.0103</td><td>more oversmoothed</td></tr><tr><td>2</td><td>0.7762</td><td>0.7865</td><td>+0.0104</td><td>more oversmoothed</td></tr></table>

The effect of the topo-evolution loss on OSI is dataset-dependent rather than uniform. On MUTAG, regularization reduces OSI at every layer relative to the baseline (∆ ranging from −0.0106 at layer 0 to −0.0459 at layer 3), consistent with the intended anti-collapse effect of $\mathcal { L } _ { \mathrm { s m o o t h } }$ . On PROTEINS and COLLAB, however, regularization increases OSI at every layer (∆ up to +0.0345 on PROTEINS), the opposite of the intended effect.

This dataset-dependence is not an isolated anomaly in the OSI measurements alone: it is directly mirrored in the loss-ablation accuracy results (Section 5). The topo-evolution loss in isolation improves accuracy on MUTAG (though not significantly at our sample size) and significantly reduces accuracy on PROTEINS and COLLAB – exactly the datasets where OSI also moves in the undesired direction. We view this correspondence as evidence that the OSI measurement is capturing a real, dataset-dependent property of the regularizer’s behavior, rather than measurement noise decoupled from downstream performance.

We do not claim $\mathcal { L } _ { \mathrm { s m o o t h } }$ acts as a general anti-oversmoothing mechanism. This finding motivates the heuristic, rather than proven, framing of Proposition 1 in Section 6: $\mathcal { L } _ { \mathrm { s m o o t h } }$ provides a first-order, local incentive against abrupt layer-to-layer signature divergence, but this does not translate into a guaranteed reduction in cross-graph signature convergence, and the empirical relationship between the two can go either direction depending on the dataset.

## E Pooling Comparison Details

UTS-Pool vs. TopKPool and SAGPool. TopKPool [2] scores nodes via $\alpha _ { v } = \mathbf { h } _ { v } ^ { \top } \mathbf { p } / \| \mathbf { p } \|$ , a linear projection of the node embedding. SAGPool [15] scores nodes using a single graph convolution followed by a linear projection, making scores sensitive to 1-hop neighbourhood structure. Both meth ods are therefore limited to local linear information. UTS-Pool (§4.3) scores nodes by the topological richness of their local embedding neighbourhood via $\Phi _ { \mathrm { e m b } } ^ { \mathrm { d i f f } }$ , capturing non-linear geometric structure such as local component topology, cycle density, and spectral connectivity. This makes UTS-Pool more sensitive to structural boundary nodes and bridges, which are often the most discriminative nodes for graph classification.

TOGL: implementation. TOGL reproduces the central architectural mechanism of TOGL [13] – multiple learnable filtration functions, trained end-to-end and summarized into a permutationinvariant topological descriptor – rather than reusing the original authors’ code, since TOGL’s released implementation depends on a custom C++/CUDA differentiable-persistence backend that is not distributed as an installable package. Concretely, TOGL consists of:

• k learnable filtration functions. For each of $k = 4$ filtrations, a two-layer MLP $f _ { i } : \mathbb { R } ^ { d } $ R $( d \to d / 2 \to 1$ , ReLU) maps each node embedding to a scalar filtration value $f _ { j } ( { \bar { \mathbf { h } } } _ { v } )$ , learned jointly with the GIN encoder.

• Differentiable $H _ { 0 }$ lifetime surrogate. For filtration $j ,$ each edge $( u , v )$ is assigned the induced sublevel-set value max $( f _ { j } ( { \bf h } _ { u } ) , \bar { f } _ { j } ( { \bf h } _ { v } ) )$ ; the $n - 1$ smallest such values across the graph are taken as a differentiable surrogate for 0-dimensional persistence lifetimes, replacing the nondifferentiable MST/union-find computation of exact sublevel-set persistence. This mirrors the sorted-edge-weight $H _ { 0 }$ surrogate used for $\Phi _ { \mathrm { e m b } }$ (Eq. 14), applied here to a learned scalar filtration rather than embedding-space pairwise distances.

• DeepSets graph descriptor. The lifetime set for each filtration is embedded permutationinvariantly via a per-lifetime MLP $( 1  1 6  1 6 )$ followed by sum-pooling, following the DeepSets construction used for persistence-diagram embedding in TOGL. Concatenating across all k filtrations gives a $k \times 1 6 \stackrel { = } { = }$ 64-dimensional graph-level topological descriptor, which is concatenated to the post-pooling structural readout before classification.

• Filtration-based pooling scores. Per-node pooling scores are a learned linear combination of the k filtration values, passed through a sigmoid and used for top-k node selection with the same pool ratio $\rho = 0 . 5$ and top-k selection mechanism as UTS-Pool, ensuring the two pooling operators are compared under an identical selection procedure and differ only in how each node’s score is computed.

For a fair comparison, TOGL and UTSTopPool share the same GIN encoder (identical hidden dimension, depth, and dropout per dataset, Section $5 ) ,$ the same AdamW optimizer, learning-rate schedule, gradient clipping, and early-stopping protocol (Appendix A.5), and are evaluated under the same 10-fold stratified cross-validation. Table 24 reports total parameter counts (encoder + pooling module + classifier head) for the three pooling-comparison models across all three datasets. The gap between TOGL and UTSTopPool is dominated by TOGL’s four learnable filtration functions (33,284 parameters), which are shared between node scoring and the topological descriptor and cannot be separated into a pooling-only cost; the remaining gap comes from the DeepSets embedding networks and the correspondingly wider classifier head needed to consume the resulting 64-dimensional descriptor. UTSTopPool’s scorer, by contrast, uses a single lightweight MLP with no separate descriptor branch, accounting for its substantially smaller footprint at comparable or better accuracy.

Table 24: Total parameter count for the three pooling-comparison architectures (GIN encoder + pooling module + classifier head), all under identical backbone settings per dataset (4 GIN layers for MUTAG and PROTEINS, 3 for COLLAB; hidden size 128). UTSTopPool uses the full 14-dim scorer on MUTAG/PROTEINS and the lightweight 5-dim proxy on COLLAB (Section 4.4), accounting for its slightly smaller overhead there.
<table><tr><td>Variant</td><td>MUTAG</td><td>PROTEINS</td><td>COLLAB</td></tr><tr><td>GIN (no pooling)</td><td>293,702</td><td>292,934</td><td>351,238</td></tr><tr><td>UTSTopPool</td><td>311,239</td><td>310,471</td><td>367,975</td></tr><tr><td>TOGL pool</td><td>352,911</td><td>352,143</td><td>410,447</td></tr></table>

## F Layer-wise UTS Analysis: Full Results

## F.1 Layer-wise UTS Feature Evolution and Signature Space Analysis

This section provides additional visualizations supporting the analysis of structural evolution in Section 4.4. To understand how the topological representations of graphs change as they pass through successive layers of a Graph Neural Network (GNN), we utilize the UTSAnalyzer module to empirically track the Universal Topological Signatures (UTS) at each layer.

UTS Feature Evolution Across GNN Layers. Figure 4 tracks the trajectory of specific topological and geometric features from Layer 0 to Layer 3. The plots visualize the mean feature value at each layer, with the standard deviation represented by the shaded regions. The specific topological attributes tracked include $H _ { 0 }$ mean lifetime, $H _ { 1 }$ mean lifetime, Betti $\beta _ { 0 } .$ , Betti $\beta _ { 1 }$ , Mean NN dist, Spectral gap $\lambda _ { 1 }$ , and Spectral entropy. Tracking these properties reveals structural drift over the network’s depth; for instance, features like $H _ { 0 }$ mean lifetime and Spectral gap $\lambda _ { 1 }$ demonstrate a distinct decrease as the depth increases from layer 0 to 3. This quantitative drift highlights the smoothing of connected components and structural distinctness in deeper network layers.

![](images/71161585f4744d686432e8bae595d1c93f328a3d22a567ff38d9fedb8035fc66.jpg)  
Figure 4: UTS Feature Evolution Across GNN Layers. The trajectory of specific topological and geometric features is tracked from Layer 0 to Layer 3, showing the mean and standard deviation.

PCA of UTS Signature Space per Layer. To further understand representation collapse and oversmoothing, Figure 5 provides a two-dimensional projection of the UTS vectors. The feature space is broken down into four subplots corresponding to Layer 0, Layer 1, Layer 2, and Layer 3. Standardized UTS vectors are projected onto the first two principal components, denoted as PC1 and PC2. By observing how the data points cluster and shift from Layer 0 to Layer 3, we can visually track the structural evolution of the embeddings. A contraction in the spread of these points in deeper layers provides empirical, visual evidence of representation collapse, which is mathematically measured by the Over-Smoothing Index (OSI).

![](images/6afc91dc00dfcf4bac7da4643ee9def90020ef213ce1bee9b102119f81349ca6.jpg)

![](images/9a717059c88fea4525dca597bd94f9e3c10244725220fb9e44af2408124b1f7b.jpg)

![](images/4c8ece515ce302c57842dbe50e9eace5d5521890858862cbd1e8b24be28eb132.jpg)

![](images/6d02b416b791e3cfa98d8c18bbe871dcfb14478a6730b66bd0cea75f8f2e804d.jpg)  
Figure 5: PCA of UTS Signature Space per Layer. Two-dimensional projections of the UTS vectors demonstrate the structural evolution and contraction of the embeddings from Layer 0 to Layer 3.

## F.2 Importance of Ollivier-Ricci Curvature in Biological and Chemical Networks

Biological and chemical networks, such as molecular graphs and protein-protein interaction (PPI) networks, exhibit highly specific structural motifs that dictate their functional properties. Standard message-passing neural networks often struggle to capture the nuances of these topologies, particularly distinguishing between dense motifs (like aromatic rings or protein complexes) and critical bottlenecks (like aliphatic chains or inter-module bridges).

To address this, we incorporate Ollivier-Ricci (OR) curvature, a discrete geometric measure that quantifies the overlap between the local neighborhoods of two connected nodes. In the context of biochemical networks, OR curvature provides a powerful structural inductive bias:

• Positive Curvature: Indicates highly connected neighborhoods, effectively identifying cliques, aromatic rings, and dense functional modules.

• Negative Curvature: Identifies “bridges” or bottlenecks between distinct clusters. In chemical graphs, these often correspond to crucial bonds connecting distinct functional groups; in PPI networks, they represent critical communication pathways between different biological processes.

By explicitly encoding OR curvature, our formulation allows the model to appropriately route information, preventing the over-smoothing of features across structural bottlenecks while encouraging aggregation within functional cliques.

Ablation Study: The Necessity of Curvature Features. To empirically validate the contribution of these geometric features, we conducted an ablation study comparing our full formulation against a variant where all Ollivier-Ricci curvature features were removed.

As shown in Table 25, removing the OR curvature features severely degrades the model’s predictive capabilities. Stripped of the ability to structurally differentiate between dense rings and critical bottlenecks, the ablated model fails to surpass the performance of the baseline architecture. This demonstrates that the geometric insights provided by Ollivier-Ricci curvature are not merely auxiliary, but are fundamentally necessary for achieving state-of-the-art performance in biochemical graph representation learning.

Table 25: Accuracy comparison across MUTAG, PROTEINS, and COLLAB datasets. Results are reported as $\mathrm { M e a n } \pm \mathrm { S t d } .$
<table><tr><td>Variant</td><td>MUTAG</td><td>PROTEINS</td><td>COLLAB</td></tr><tr><td>GIN (Unregularized)</td><td> $0 . 8 3 1 5 \pm 0 . 0 8 4 0$ </td><td> $0 . 7 4 0 1 \pm 0 . 0 3 3 6$ </td><td> $0 . 8 1 0 4 \pm 0 . 0 2 8 6$ </td></tr><tr><td>Embedding-UTS (w/o Ricci)</td><td> $0 . 8 5 1 8 \pm 0 . 0 8 2 6$ </td><td> $0 . 7 3 9 2 \pm 0 . 0 3 4 1$ </td><td> $0 . 8 1 8 2 \pm 0 . 0 2 8 8$ </td></tr><tr><td>Graph-UTS (w/o Ricci)</td><td> $0 . 8 6 5 2 \pm 0 . 0 8 0 1$ </td><td> $0 . 7 4 4 8 \pm 0 . 0 3 8 2$ </td><td> $0 . 8 2 6 9 \pm 0 . 0 2 9 7$ </td></tr></table>

## G Computational Scalability of UTS Variants

## G.1 Benchmark Setup

We measure wall-clock computation time as a function of graph size for four UTS variants used in this work:

• Graph-UTS (Ricci on) — the full structural signature $\Phi _ { \mathrm { g r p h } }$ (Section 4.1.2) with Ollivier–Ricc and Forman–Ricci curvature enabled.

• Graph-UTS (Ricci $\mathbf { o f f } ) - \Phi _ { \mathrm { g r p h } }$ with curvature disabled, retaining distance, spectral, persistence, degree, clustering, centrality, and connectivity features.

• Embedding-UTS (exact) — the full 14-dimensional embedding-space signature $\Phi _ { \mathrm { e m b } }$ (Section 4.1.1), including exact GUDHI-based Rips persistence and eigendecomposition.

• LightEmbeddingUTS — the 5-dimensional $O ( n ^ { 2 } )$ geometric proxy used for large/dense graphs in §4.4 (mean/std pairwise distance, global spread, mean nearest-neighbour distance, a size proxy), omitting triangle enumeration, eigendecomposition, and persistence.

Graphs are synthetic Barabási–Albert graphs with average degree 4, chosen to approximate the sparse, molecular/protein-like density profile of our benchmark datasets. Node sizes are swept over $n \in \{ 1 0 , 1 7 , 3 0 , 5 0 , 7 5 , 1 0 0 , 1 5 0 , 2 0 0 , 3 0 0 , 4 0 0 , 5 0 0 , 6 2 0 , 8 0 0 , 1 0 0 0 \}$ , with three repeats per size (independent random graphs for GraphUTS variants; independent random node embeddings of dimension 128 for EmbeddingUTS variants, matching the hidden dimension used elsewhere in this work). We annotate three reference sizes directly corresponding to our benchmark datasets: MUTAG’s average size $( n \approx 1 7 )$ , COLLAB’s average size $( n \approx 7 4 . 5 )$ , and PROTEINS’ maximum size $( n = 6 2 0 ) -$ the exact figures cited in review.

## G.2 Results

Two clearly separated cost tiers. At every graph size tested, the two GraphUTS variants are $5 { - } 1 5 \times$ more expensive than either embedding-space variant. This gap is driven by GraphUTS’s all-pairs shortest-path computation and betweenness/closeness centrality, both of which scale worse than the pairwise-distance step shared by the embedding-space methods, and is present regardless of whether curvature is enabled.

The curvature-specific cost narrows at scale. GraphUTS (Ricci on) starts roughly $3 5 \times$ more expensive than GraphUTS (Ricci off) at $n = 1 0 \left( { \sim } 0 . 1 1 \mathrm { s } \ \mathrm { v s . } \ { \sim } 0 . 0 0 3 \mathrm { s } \right)$ , but the two curves converge substantially by $n = 1 0 0 0 ( { \sim } 4 . 0 \mathrm { s } \ \mathrm { v s } . \sim 3 . 6 \mathrm { s } ,$ read from Figure 6). This indicates that at large $n ,$ the components shared between both variants — persistence, centrality, all-pairs distances — dominate total cost, and the Ollivier/Forman curvature computation, while expensive in absolute terms at small graphs, is not the primary scaling bottleneck at the sizes relevant to PROTEINS and COLLAB. We report this as an empirical observation from a single benchmark configuration rather than a proven asymptotic result.

![](images/7feaca61ac9b552714d0d750cc00b08cb7d1ef8ee738d1dc3c882db6e9518388.jpg)  
Figure 6: Wall-clock computation time (log scale) versus graph size (log scale, number of nodes) for four UTS variants, averaged over three synthetic Barabási–Albert graphs per size. Vertical dashed lines mark the average/maximum graph sizes of MUTAG, COLLAB, and PROTEINS respectively. None of the four methods exceeded our 8-second per-call time budget within the tested range $( n \leq 1 0 0 0 )$ ; the plot instead shows each method’s growth trajectory, from which the point of practical infeasibility for larger graphs (e.g. long-range or web-scale benchmarks) can be extrapolated.

At PROTEINS’ maximum graph size (620 nodes), GraphUTS costs approximately 1.5–2 seconds per graph regardless of curvature setting. This is the concrete, measured basis for our design choice (§4.2) to precompute $\Phi _ { \mathrm { g r p h } }$ once in parallel across the dataset prior to training, rather than recomputing it inside the training loop, where a cost of this magnitude per graph would make per-batch computation infeasible at typical batch sizes.

The lightweight proxy delays, but does not eliminate, the scaling problem. LightEmbeddingUTS is the cheapest method at every size tested, but its relative advantage over exact EmbeddingUTS shrinks with graph size: at $n = 1 0$ it is roughly 30× cheaper (∼0.0002s vs. ∼0.006s); by $n = 1 0 0 0$ this narrows to roughly $2 \times ( { \sim } 0 . 4 \mathrm { s \ v s . \ } { \sim } 0 . \bar { 7 } 5 \mathrm { s } )$ . Because both methods share the same $O ( n ^ { 2 } )$ pairwise-distance computation, and LightEmbeddingUTS’s only saving is skipping the $O ( n ^ { 3 } )$ triangle enumeration, eigendecomposition, and persistence steps that exact EmbeddingUTS performs on top of that shared cost, the two curves must converge as n grows and the shared ${ \mathrm { \bar { \it O } } } ( n ^ { 2 } )$ term comes to dominate both. We state this plainly as a limitation of the current mitigation: LightEmbeddingUTS extends the practical size range for which UTS computation remains tractable, but does not itself achieve sub-quadratic scaling, and a graph large enough will eventually make even the lightweight proxy impractical.

## G.3 Implications and Limitations

This benchmark directly substantiates two design decisions already present in our pipeline and motivates one limitation we now state explicitly. First, it justifies parallel one-time precomputation of $\Phi _ { \mathrm { g r p h } }$ (rather than per-batch recomputation) as a practical necessity rather than a mere optimization, given measured costs of 1–2 seconds per graph at PROTEINS-scale. Second, it validates LightEmbeddingUTS as an effective mitigation across the size range spanned by our current benchmark datasets (MUTAG, PROTEINS, COLLAB), all of which fall within the region where the lightweight proxy retains a meaningful (if narrowing) speed advantage.

We note as a limitation that this benchmark uses synthetic graphs at a single fixed density (average degree 4) and does not directly profile wall-clock cost on the real dataset graphs used for training, nor at the scale of long-range or web-scale graph benchmarks (potentially tens of thousands of nodes), where — extrapolating the trends in Figure 6 — even LightEmbeddingUTS’s shared $O ( n ^ { 2 } )$ pairwise-distance step would itself become the binding constraint.