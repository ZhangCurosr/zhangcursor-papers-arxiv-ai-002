# Synergistic Fusion of Topological Structure and Temporal Semantics of Mobility for Urban Region Embedding

Namwoo Kim<sup>∗</sup>, Jeeyun Chang<sup>†</sup>, Kanghoon Lee<sup>‡</sup>, and Yoonjin Yoon<sup>∗§</sup>

<sup>∗</sup>Urban AI Institute, Korea Advanced Institute of Science and Technology (KAIST), Daejeon, Republic of Korea <sup>†</sup>Graduate School of Data Science, KAIST, Daejeon, Republic of Korea

<sup>‡</sup>Department of Industrial Engineering and Systems, KAIST, Daejeon, Republic of Korea

<sup>§</sup>Department of Civil and Environmental Engineering, KAIST, Daejeon, Republic of Korea

Abstract—Urban region embeddings have shown promising results in diverse urban sensing tasks such as crime, income, and service-call prediction. Recent methods improve representation quality by integrating mobility data with auxiliary modalities, using cross-view attention or contrastive objectives to align heterogeneous features into a unified region representation. However, leveraging the temporal dynamics of human mobility remains under-explored. Regional inflow and outflow fluctuate throughout the day, and inter-region connections emerge, persist, and dissolve over time. Moreover, prevailing fusion strategies combine views additively and miss the joint signal that emerges only when views co-occur. To address these gaps, we propose Mobility Stream–Structure Synergy (MoSS), which derives complementary views from mobility data: a Sequence view that preserves each region’s hourly inflow/outflow profile, and a Structure view based on zigzag persistence diagrams that capture how regional connectivity emerges, persists, and dissolves over time. A synergy module then extracts emergent representations from the co-occurrence of these views through multi-degree interactions, explicitly capturing higher-order signal across views. Extensive experiments on New York City and Chicago show that MoSS achieves state-of-the-art performance across three downstream tasks using mobility data alone, outperforming baselines that rely on auxiliary modalities.

Index Terms—Urban region embedding, Time series modeling, Zigzag persistence

## I. INTRODUCTION

U <sup>RBAN</sup> <sup>region</sup> <sup>embedding</sup> <sup>maps</sup> <sup>each</sup> <sup>city</sup> <sup>region</sup> <sup>to</sup> <sup>a</sup>latent vector, enabling downstream tasks such as crime latent vector, enabling downstream tasks such as crime prediction, region popularity estimation, population density inference, land-use clustering, and socioeconomic analysis [1]– [8]. A region, however, cannot be characterized in isolation: its functional identity emerges from how it interacts with other regions — residential blocks outflow to business districts in the morning, commercial zones absorb inflows on weekends, and so on. To capture such interactions, human mobility data have been widely adopted, since every trip between two regions is an observable act of interaction that, in aggregate, encodes the functional organization of the city [9], [10].

A substantial body of work uses human mobility as the primary signal for region embedding. Early approaches modeled origin-destination (OD) co-occurrence with skip-gramstyle objectives or random walks on flow graphs [2], [3]. Subsequent work reconstructs OD matrices or conditional trip distributions [4]–[6], and the most recent approaches apply contrastive learning over inflow/outflow embeddings [7], [11]. These studies collectively establish that mobility carries rich functional information.

A complementary line of work enriches region representations by combining mobility with auxiliary sources — POIs, check-ins, road networks — and proposes a variety of fusion strategies. Prior studies have explored diverse strategies for multi-view urban representation learning. Early approaches focused on learning view-shared representations and aggregating them with adaptive weighting schemes [4]. Subsequent work modeled mobility patterns through graph clustering and crossview attention mechanisms [5], while others built heterogeneous region graphs with prompt-based task adaptation [12]. More recent methods have emphasized fine-grained fusion and consistency learning, such as dual attention over regions and features [13] and mutual-information-based consistency objectives across views [7]. Recent advances further leverage graph pre-training, task-aware prompting [14], and structureaware contrastive learning for joint multi-view representation learning [15]. A concurrent line of work explicitly separates cross-view shared and view-private signals in this multi-modal setting [16]. Collectively, these methods show that the fusion of heterogeneous urban signals can improve downstream performance.

Despite these advances, current approaches leave two gaps under-explored. The first concerns how mobility itself is modeled. Mobility-oriented methods typically model mobility either as a static graph or as independently processed temporal snapshots, so they fail to capture both region-specific temporal dynamics—weekday/weekend periodicity, peak/off-peak transitions, and autocorrelation patterns—and the evolving connectivity structure among regions, where inter-region relations emerge, persist, and disappear over time [17]. Figure 1 makes both facets concrete on a Manhattan pair: two regions with comparable aggregate outflow magnitudes nevertheless differ sharply in temporal dependency (Figure 1(a, b)) as well as in inter-region connectivity (Figure 1(c, d)).

(a) Outflow time series near-identical volume  
![](images/9a42e240fffd2d6122ac122bcc150d52fc7868878cc75a6fabbdd152b5babd60.jpg)

(b) Autocorrelation temporal dependency differs  
![](images/cc08fc12c298bba3a16e8c797430e0b0bd351865340a12bd7bd63c47921e138c.jpg)

(c) Mobility connectivity of R33  
![](images/077d1c714395f5f5f6382965e9dd39de87ce4d089beb1c5cf7fe64e7e8d37f50.jpg)  
Fig. 1: Same volume, different connectivity dynamics. Two Manhattan regions, $R _ { 3 3 }$ and $R _ { 7 8 }$ , with near-identical aggregate mobility over the 31-day study period. (a) Outflow time series over the full month: the two regions are comparable in aggregate volume. (b) Autocorrelation up to a 72 h lag: $R _ { 3 3 }$ exhibits a strong 24 h periodicity while $R _ { 7 8 }$ shows a much weaker oscillation — their temporal dependency structures differ markedly. $( \mathbf { c } , \mathbf { d } )$ Inter-region connectivity at three consecutive hours (Fri 23:00 → Sat 01:00) over which the two regions also share similar hourly outflow $( 5 4 / 4 8 \to 3 0 / 3 6 \to 1 8 / 1 7$ trips). Despite this volume parity, neighborhood dynamics diverge sharply. Solid nodes denote neighbors persistent from the previous snapshot; outlined nodes are new entrants.

The second gap concerns how views are fused. Even when both temporal and structural views are made available, prevailing multi-view fusion strategies—attention-based aggregation [4], [5], [12], [13] or contrastive alignment [7], [15]— combine views additively and tend to miss the co-occurrencebased signal that emerges only when the two are considered jointly. A region’s functional identity often depends on the specific co-occurrence of its temporal rhythm and its connectivity neighborhood, so capturing this multiplicative interaction between temporal and structural patterns is essential for translating these two complementary streams into an effective region embedding that jointly reflects both signals.

To address these gaps, we propose Mobility Stream–Structure Synergy (MoSS), a region embedding framework that captures both per-region temporal dynamics and the evolving interregion mobility structure from a single mobility stream, and explicitly models cross-view synergy. MoSS consists of two complementary streams. A Sequence stream captures each region’s temporal semantics by modeling its hourly inflow and outflow series with dilated convolutions that span daily and weekly periodicities. A Structure stream captures each region’s topological structure by tracking the evolution of its connectivity neighborhood through zigzag persistence diagrams [18]. The two streams produce complementary views per region, which we fuse via a synergy module built on a shared–private feature decomposition together with multidegree interactions. Empirical evaluation on New York City and Chicago shows that MoSS achieves state-of-the-art region embedding performance from a single mobility input, without relying on any auxiliary modality.

To summarize, our contributions are as follows.

• We propose a dual-stream mobility representation consisting of a sequence stream and a structure stream — to the best of our knowledge, the first application of zigzag persistent homology to urban region embedding.

• To model higher-order cross-view interactions, we introduce a synergy module that combines shared–private decomposition with multi-degree multiplicative interactions.

• Extensive experiments on New York City and Chicago show that MoSS achieves state-of-the-art urban region embedding using mobility data alone, outperforming baselines that rely on auxiliary modalities such as POIs, check-ins, or land use.

## II. RELATED WORK

## A. Human Mobility in Urban Region Embedding

Human mobility has long been a core signal for urban region embedding. Early studies such as ZE-Mob [3] and HDGE [2] primarily modeled mobility using static origin–destination (OD) statistics or transition graphs. MGFN [5] extended this line of work by introducing multiple mobility patterns derived from grouped OD snapshots, but these patterns were still constructed from static aggregations and did not capture the continuous temporal dynamics of mobility flows. Subsequent studies incorporated additional urban modalities such as POIs, land use, satellite imagery, and street views [4], [6], [11]–[13], [19]–[23]. Other recent approaches explored prompt learning and contrastive learning frameworks for urban representation learning [7], [12], [14], [15], [24].

In summary, mobility is still often modeled as a static signal, typically using aggregated OD matrices, transition graphs, or temporally grouped snapshots. While these representations have proven effective for capturing large-scale mobility structure, they generally summarize observations over time and therefore provide only limited access to the temporal continuity and dependency inherent in human movement patterns. As a result, the dynamic evolution of inter-region interactions, including how connections form, persist, and change across hours and days, remains less explored. To address this gap, we model mobility as fine-grained temporal dynamics and capture both its flow-level temporal patterns and its evolving connectivity topology within a unified framework.

## B. Multi-View Approaches in Urban Region Embedding

A growing body of work leverages diverse modalities and views to learn comprehensive and semantically rich urban region embeddings [25]. These multi-view methods cluster into three fusion families. Attention-based fusion [4], [5], [12], [13] combines per-view embeddings through input-dependent weighted sums, often arranged hierarchically across intraview, inter-view, and region levels. Contrastive alignment [7], [11], [14], [15] reshapes view-specific embeddings via crossview agreement objectives—mutual information maximization, structure-aware contrastive losses, or prompt-based selfsupervision—to enforce consistency. Shared–private decomposition, established for domain adaptation and multimodal sentiment analysis [26], [27], has been recently adapted to urban region embedding by ComSRE [16] on mobility and POI views, instantiated with a contrastive alignment and a differential orthogonality penalty.

These three families, however, remain limited in modeling higher-order cross-view interactions. Informative patterns that emerge only through the joint presence of multiple views are not explicitly represented in additive fusion [28]. Prior work has shown that explicitly modeling multiplicative interactions can improve multimodal representation learning [29]–[31]. Building on this intuition, we introduce a synergy module that models cross-view interactions, including pairwise (secondorder) and triple-wise (third-order) multiplicative terms across views. Unlike attention or contrastive alignment, synergy module explicitly incorporates these higher-order interaction terms into the representation pathway, providing a compositional inductive bias for multi-view urban region embedding.

## C. Topological Data Analysis for Time-Varying Data

Persistent homology, a central tool in topological data analysis (TDA), summarizes the multi-scale shape of data as a persistence diagram, and has been used as a complementary signal in deep representation learning, both for time series [32] and for tracking structural change in time-varying graphs [33]. However, a standard filtration grows monotonically, so it cannot represent features that vanish and re-emerge. Zigzag persistence [18] removes this restriction by allowing inclusions in alternating directions, which allows such features to be tracked directly (Sec. III-D), and a growing line of work applies it to temporal and graph-structured data. It has been used to detect bifurcations in dynamical systems [34] and to summarize temporal networks as a single zigzag diagram, recovering daily and weekly commuting periodicities in transportation that connectivity and centrality statistics miss [35]. More recent work integrates zigzag persistence into graph neural networks, either as a time-aware topological layer [36] or as a compact filtration-curve summary used for time-series forecasting [37].

Across these settings, zigzag persistence has proven effective in capturing how connectivity evolves over time, but it has not yet been used for urban region embedding.

## III. PRELIMINARIES

This section introduces the topological tools that MoSS uses to summarize the temporal evolution of inter-regional mobility. We refer to [38], [18], and [35] for a complete treatment.

## A. Clique Complex of a Graph

A simplicial complex K over a vertex set V is a collection of subsets (simplices) closed under taking subsets. A subset of size k+1 is a k-simplex, so that vertices, edges, and triangles are the 0-, 1-, 2-, and higher-dimensional simplices respectively.

Definition 1 (Clique complex): Let $G \ = \ ( \nu , { \mathcal { E } } )$ be a finite undirected graph. The clique complex of G up to dimension D is

$$
{ \mathcal { X } } _ { D } ( G ) = \{ \sigma \subseteq \gamma : 1 \leq | \sigma | \leq D { \mathrm { + 1 } } , \sigma { \mathrm { ~ i s ~ a ~ c l i q u e ~ i n ~ } } G \} .\tag{1}
$$

## B. Homology and Betti Numbers

For each $k \geq 0 ,$ let $C _ { k } ( \mathcal { K } )$ be the $\mathbb { F } _ { 2 } .$ -vector space spanned by the k-simplices of K. The boundary map $\partial _ { k } \colon C _ { k } ( K ) \to$ $C _ { k - 1 } ( \mathcal { K } )$ sends a k-simplex to the formal sum of its $( k { - } 1 )$ faces, and satisfies $\partial _ { k - 1 } \circ \partial _ { k } = 0$ . Thus, the k-th homology group is

$$
H _ { k } ( \mathcal { K } ) = \ker ( \partial _ { k } ) / \operatorname { i m } ( \partial _ { k + 1 } ) ,\tag{2}
$$

i.e., the cycles modulo the boundaries.

Definition 2 (Betti number): The k-th Betti number of a simplicial complex K is the dimension of its k-th homology group over $\mathbb { F } _ { 2 } ,$

$$
\beta _ { k } ( { \mathcal K } ) \ : = \ : \dim \ : H _ { k } ( { \mathcal K } ) .\tag{3}
$$

Geometrically, $\beta _ { k }$ counts the number of independent kdimensional holes in $\kappa \colon ~ \beta _ { 0 }$ is the number of connected components (separate pieces), $\beta _ { 1 }$ the number of 1-dimensional loops (tunnels through the complex), and $\beta _ { 2 }$ the number of enclosed 2-dimensional cavities.

A static $\beta _ { k }$ is one snapshot of structure. Capturing how features appear and disappear over time requires a notion of persistence.

## C. Persistent Homology

A filtration is a nested sequence

$$
{ \cal K } ^ { ( 1 ) } \subseteq { \cal K } ^ { ( 2 ) } \subseteq \cdots \subseteq { \cal K } ^ { ( T ) } .\tag{4}
$$

Persistent homology tracks each topological feature as it appears and disappears across this filtration sequence, recording it as a point $( b , d )$ , where b denotes the filtration index at which the feature appears and d the index at which it disappears. The collection of such points is the persistence diagram in dimension k.

A filtration only ever grows, so it cannot represent features that disappear and reappear. This is a structural property of urban mobility where edges between regions emerge during peak hours and dissolve afterwards [33], [35].

## D. Zigzag Persistent Homology

Zigzag persistent homology [18] replaces the one-way filtration of Eq. (4) by a sequence of inclusions in alternating directions,

$$
{ \mathcal K } ^ { ( 1 ) } \hookrightarrow { \mathcal K } ^ { ( 1 , 2 ) } \gets { \mathcal K } ^ { ( 2 ) } \hookrightarrow \cdots  \joinrel  { \mathcal K } ^ { ( T ) } ,\tag{5}
$$

where we take $K ^ { ( t , t + 1 ) } = K ^ { ( t ) } \cup K ^ { ( t + 1 ) }$ . The forward inclusion ${ \mathcal { K } } ^ { ( t ) } \hookrightarrow { \mathcal { K } } ^ { ( t , t + 1 ) }$ adds simplices that appear in $\mathcal { K } ^ { ( t + 1 ) }$ but not in $\mathcal { K } ^ { ( t ) }$ , and the backward inclusion $\dot { \mathcal { K } } ^ { ( t + 1 ) } \hookrightarrow \mathcal { K } ^ { ( t , t + 1 ) }$ adds simplices that exist in $\mathcal { K } ^ { ( t ) }$ but not in $\mathcal { K } ^ { ( t + 1 ) }$ . The output is again a multiset of points $( b , d )$ , but indexed along the zigzag positions rather than along a monotone parameter, so that features which vanish and re-emerge are represented natively rather than lost.

Two structural properties of urban mobility motivate the use of zigzag persistence. First, edges in inter-regional mobility are inherently transient, so the standard filtration of Eq. (4) cannot represent dissolution. Second, the macro-level role of a region depends on when its connections appear and dissolve, rather than only on the static set of connections that eventually exist.

## IV. METHOD

## A. Model Overview

MOSS represents each region from two complementary viewpoints of the same mobility stream and fuses them through a synergy module (Figure 2). The Sequence stream (Sec. IV-B) encodes each region’s hourly inflow/outflow series with a dilated TCN. The Structure stream (Sec. IV-C–IV-E) encodes how the inter-region connections around a region emerge, persist, and dissolve over the observation window using zigzag persistence. The resulting view tensors are then composed by the synergy module (Sec. IV-F), which internally factors each view into shared and view-private embeddings and aggregates them through multi-degree interaction that captures first-order, second-order, and third-order interactions across views.

## B. Sequence Stream

Each region i is associated with two observed series of trip volumes (inflow and outflow), recording how many trips leave and arrive at the region over the full observation window of length T. We stack the two normalized series along the channel axis to form a two-channel input $\mathbf { x } _ { i } = [ \bar { \mathbf { x } } _ { i } ^ { \mathrm { { i n f l o w } } } ; \bar { \mathbf { x } } _ { i } ^ { \mathrm { { o u t f l o w } } } ] \in$ $\mathbb { R } ^ { 2 \times T }$

A single dilated temporal convolutional network [39], [40] processes both flow directions jointly. After a 1×1 projection of the two-channel input to hidden width $C _ { h }$ , the series passes through a stack of $L _ { \mathrm { t c n } }$ residual blocks; the ℓ-th block applies two width-preserving 1D convolutions of kernel size k with dilation $2 ^ { \ell }$ , with a GELU pre-activation before each convolution and a residual shortcut around the pair,

$$
{ \bf h } _ { \ell } = { \bf h } _ { \ell - 1 } + \mathrm { C o n v } _ { 2 ^ { \ell } } ^ { ( 2 ) } \Big ( \mathrm { G E L U } \big ( \mathrm { C o n v } _ { 2 ^ { \ell } } ^ { ( 1 ) } \big ( \mathrm { G E L U } ( { \bf h } _ { \ell - 1 } ) \big ) \big ) \Big ) .\tag{6}
$$

Exponentially growing dilation gives the topmost block a receptive field that spans both daily and weekly periodicities using only a logarithmic number of layers, while the residual path keeps gradients well behaved. A final $1 \times 1$ projection maps the channel dimension to D-dimension, after which we reduce the time axis by max-pooling to produce a sequenceview embedding

$$
\mathbf { V } _ { i } ^ { \mathrm { s e q } } = \mathrm { m a x p o o l } _ { t } \Big ( \mathrm { C o n v } ^ { ( \mathrm { o u t } ) } \big ( \mathbf { h } _ { L _ { \mathrm { t c n } } } \big ) \Big ) \in \mathbb { R } ^ { D } .\tag{7}
$$

## C. Structure Stream: Connectivity-graph Construction

Figure 3 gives a visual roadmap of the structure stream: a temporal sequence of region-centric connectivity-graphs (Figure 3a) is summarized as per-edge lifespans (Figure 3b), and zigzag persistent homology turns those lifespans into a persistence diagram (Figure 3c) that the structure stream then encodes.

Starting from the raw OD matrices $\{ \mathbf { F } ^ { ( t ) } \} _ { t = 1 } ^ { T }$ with $\mathbf { F } ^ { ( t ) } \in$ R $\mathbf { \Psi } _ { \cdot } N \times N$ , we condense the stream into a shorter representative period of length $T ^ { \prime }$ (e.g., a typical week of hourly snapshots) by averaging OD matrices that share the same within-period phase across repeated cycles. The resulting averaged matrices are then binarized so that each entry records only whether a flow exists between two regions.

For each region $i ,$ the connectivity-graph $G _ { i } ^ { ( t ) }$ at time index t is the subgraph of $G ^ { ( t ) }$ induced by i and the regions connected to it (Figure 3a). We construct two directional variants: the outflow graph ${ \dot { G } } _ { i } ^ { \mathrm { o u t } , ( t ) }$ , which includes edges where i is the source, and the inflow graph $G _ { i } ^ { \mathrm { i n } , ( t ) }$ , which includes edges where region i is the destination. These two graphs capture complementary aspects of the functional role of region i. The temporally aligned sequences $G _ { i } ^ { \mathrm { i n } , ( t ) }$ and $G _ { i } ^ { \mathrm { o u t } , \top }$ are then used as input to the zigzag persistence computation. For simplicity, we use $\textsf { O } \in$ {out, in} to denote either direction in what follows.

## D. Structure Stream: Zigzag Persistence Computation

For each region i, we build a clique complex $\mathcal { K } _ { i } ^ { \circ , ( t ) }$ over the connectivity-graph $G _ { i } ^ { \circ , ( t ) }$ at every time index, and connect consecutive complexes through the zigzag diagram in Eq. (5). Zigzag persistent homology in dimension 0 returns a persistence diagram $\mathrm { P D } _ { i } ^ { \circ }$ that summarizes how the corresponding feature class (connected components) of the mobility neighborhood of i emerges, persists, and dissolves over the time window.

![](images/8abe3cfe63ce480e0e149887d714af7b0a1e871276c7c3fca844d2e1ac9976bf.jpg)  
Fig. 2: Overview of MoSS. Left. Two complementary streams encode each region’s mobility. The Sequence (TS) view (top) feeds the per-region hourly inflow/outflow series into a temporal convolutional encoder (TS Encoder), and the Structure (PD) view (bottom) constructs the region-centric connectivity-graph at each time step, computes its zigzag persistence diagram, and encodes the diagram with a permutation-invariant set encoder (PD Encoder). The structure encoder produces sourceand destination-side view embeddings. Right. The synergy module fuses the three views. It first applies a shared–private decomposition that factors each view into a cross-view shared embedding (filled triangles) and a view-private embedding (dashed circles), and then aggregates the resulting embeddings through multi-degree interaction that captures first-order, second-order, and third-order interactions in a single per-region embedding.

![](images/e2bb3815ff19e7818dafc796641a9628bd77454a732eab237a088434d60ba965.jpg)

![](images/3cbf4ab08e2625107349838d59a96c2e7f2b2cce7015959b7fd9d3b56133229f.jpg)

Persistence diagram (<sub>H0</sub>)(c)  
![](images/40de59a9665807633f9cb1782e29bf8bcbc625db0c79f722682a2ae7b0b659cc.jpg)  
Fig. 3: Structure stream: from a temporal sequence of connectivity-graphs to a persistence diagram. (a) Connectivitygraph $G _ { v } ^ { ( t ) }$ at three consecutive time indices (top row), and the union graph $G _ { v } ^ { ( t ) } \cup \stackrel { \bullet } { G _ { v } } ^ { ( t + 1 ) }$ between every consecutive pair (middle row); filled green nodes are neighbors active at that time, faded gray nodes are inactive. (b) Per-edge lifespans across the window. The bar segments mark the time indices at which the edge is present, exposing the emerge / persist / dissolve dynamics that no single snapshot captures. (c) The resulting zigzag persistence diagram with $H _ { 0 }$ features; the diagonal marks zero lifespan.

Figure 3 illustrates this construction on a four-node toy. Figure 3 (a) shows the per-time graphs $G _ { v } ^ { ( t ) }$ for $t = 1 , 2 , 3$ , together with the consecutive unions $\dot { G } _ { v } ^ { ( t ) } \cup \dot { G } _ { v } ^ { ( t + 1 ) }$ . These union graphs occupy the half-integer slots $t = 1 . 5 , 2 . 5$ along the filtration axis, sitting between the per-time snapshots so that edges added or removed across consecutive times are accounted for. Figure 3 (b) renders each edge’s lifespan as a horizontal bar. As edges enter or leave between consecutive complexes, the number of connected components $( \beta _ { 0 } ,$ , the rank of $H _ { 0 } )$ of the current K rises and falls, and the persistence diagram in (c) collects these changes into one point per component: birth b marks the step where a new component first separates out, death d marks the step where it merges back into an older one (or is pinned to $d = T$ if it survives the window).

The red point $( b , d ) = ( 2 . 5 , 3 )$ traces a single such event. Edge $( v , 1 )$ is present through $K _ { 2 }$ and the union $K _ { 2 , 3 }$ but disappears at $K _ { 3 }$ , which disconnects node 1 from $v { \mathrm { s } }$ component. The resulting singleton is born at the preceding union step $b = 2 . 5$ and, since $( v , 1 )$ does not reappear, persists to the window’s end $d = T = 3 .$

The other two points are read similarly. (1, 3) is $v { \mathrm { s } }$ own component, born at $K _ { 1 }$ and alive throughout the window, while (1, 1.5) is the briefly-isolated node 3 that merges into v’s component as soon as edge (v, 3) first enters at $K _ { 1 , 2 }$

We restrict the computation to $H _ { 0 }$ because the structure stream targets connectivity. $\beta _ { 0 }$ counts the connected components of a region’s mobility neighborhood, so it responds directly to whether regions remain linked to its neighbors. When a region becomes disconnected from its neighborhood, a new component appears, and when regions reconnect, components merge. Therefore, tracking $H _ { 0 }$ across the zigzag captures the connectivity changes that motivate the structure stream, while higher-dimensional features $( \beta _ { 1 }$ and above) are not needed for this connectivity-focused purpose.

## E. Structure Stream: Persistence-diagram Encoder

The zigzag persistence computation returns, for each region i and direction $\circ \in \{ \mathrm { o u t } , \mathrm { i n } \}$ , a multiset of birth–death pairs $( b ^ { \circ } , d ^ { \circ } )$ that characterize the emergence and disappearance of connected components over time in the mobility neighborhood of region i along direction ◦. We augment each pair with its lifespan $l ^ { \circ } = d ^ { \circ } - b ^ { \circ }$ , yielding the per-region, per-direction persistence multiset

$$
\begin{array} { r l r } & { } & { \left\{ p _ { i , 1 } ^ { \circ } , \ldots , p _ { i , M _ { i } } ^ { \circ } \right\} , \qquad p _ { i , j } ^ { \circ } = ( b _ { i , j } ^ { \circ } , d _ { i , j } ^ { \circ } , l _ { i , j } ^ { \circ } ) \in \mathbb { R } ^ { 3 } , } \end{array}\tag{8}
$$

where $M _ { i }$ denotes the cardinality. Our goal is to map this multiset to a fixed-size vector $\mathbf { V } _ { i } ^ { \circ } \in \mathbb { R } ^ { D }$

a) Permutation invariance.: Points in the persistence diagram carry no canonical ordering, since the birth–death pairs index topological features that all coexist during the zigzag and have no natural total order. An order-sensitive encoder would treat the diagram as an order sequence rather than a set, so the resulting embedding would depend on an arbitrary indexing convention rather than on the topological information the diagram actually encodes. To rule this out by construction, the encoder $f _ { \theta } ^ { \circ }$ must satisfy, for any multiset $\{ p _ { 1 } , \hdots , p _ { M } \}$ of diagram points,

$$
f _ { \theta } ^ { \circ } \bigl ( \bigl \{ p _ { \pi ( 1 ) } , \ldots , p _ { \pi ( M ) } \bigr \} \bigr ) \ = \ f _ { \theta } ^ { \circ } \bigl ( \bigl \{ p _ { 1 } , \ldots , p _ { M } \bigr \} \bigr ) \qquad \forall \pi \in S _ { M } .\tag{9}
$$

b) Encoder.: The canonical way to build a function that satisfies Eq. (9) is to compose a per-point transformation shared across all points with a symmetric aggregator over the resulting features [41], [42]. We instantiate this as

$$
\mathbf { V } _ { i } ^ { \circ } \ = \ f _ { \theta } ^ { \circ } \big ( \{ p _ { i , 1 } ^ { \circ } , \ldots , p _ { i , M _ { i } } ^ { \circ } \} \big ) \ = \ \zeta \Big ( \{ \phi ( p _ { i , j } ^ { \circ } ) \} _ { j = 1 } ^ { M _ { i } } \Big ) \ \in \ \mathbb { R } ^ { D } ,\tag{10}
$$

where $\phi _ { \theta } \colon { \mathbb { R } } ^ { 3 }  { \mathbb { R } } ^ { D }$ is a stack of $1 \times 1$ convolutions along the point axis with ReLU activations — equivalent to applying the same MLP to every diagram point independently — and ζ is coordinate-wise max-pooling across the $M _ { i }$ feature vectors. Both $\phi$ and $\zeta$ depend only on the multiset of their inputs and not on the order in which the points are presented, so $f _ { \theta }$ satisfies Eq. (9) by construction.

Through $f _ { \theta } ^ { \circ }$ for $\mathrm { ~ o ~ } \in \ \{ \mathrm { o u t } , \mathrm { i n } \}$ , we obtain two embeddings $\mathbf { V } ^ { \mathrm { o u t } }$ and $\mathbf { \bar { V } } ^ { \mathrm { i n } } \mathrm { ~ i n ~ } \mathbb { R } ^ { N \times D }$ . Each row encodes when the connected components of a region’s neighborhood emerge, persist, and disappear.

## F. Synergy Module

The two encoder streams produce three view embeddings per region: a sequence view V<sup>seq</sup> from the temporal series and two directional views V<sup>out</sup>, $\mathbf { V } ^ { \mathrm { i n } }$ from the zigzag persistence diagrams, collected as $\mathcal { V } = \{ \mathrm { s e q } , \mathrm { o u t } , \mathrm { i n } \}$ . We take synergy to mean the useful features that arise only when two or more of these views co-occur — signal that is uninformative, or not present at all, in any single view observed in isolation. Recovering such features requires that the fusion stage receive, from each view, only what that view contributes distinctively: if shared content were left inside each view, every crossview product would repeatedly recombine the same redundant signal, causing the fused representation to be dominated by redundant cross-view correlations rather than genuinely complementary interactions.

The synergy module therefore fuses the three views in two stages. First, a shared–private decomposition splits each view into a cross-view shared embedding and a view-private embedding. Second, a multi-degree interaction applies a low-rank tensor product over the three private embeddings, enabling the fused representation to capture higher-order interactions across views.

a) Shared–private decomposition.: For each view embedding $\mathbf { V } ^ { v } \in \mathbb { R } ^ { D }$ , a shared projection $g _ { \mathrm { s h a r e d } } : \mathbb { R } ^ { D } \to \mathbb { R } ^ { D }$ with parameters tied across views and a view-private projection $\dot { \boldsymbol g } _ { \mathrm { p r i v a t e } } ^ { v } : \mathbb { R } ^ { D }  \mathbb { R } ^ { D }$ produce [26], [27]

$$
\begin{array} { r l } & { \mathbf { S } ^ { v } = g _ { \mathrm { s h a r e d } } ( \mathbf { V } ^ { v } ) \in \mathbb R ^ { D } , } \\ & { \mathbf { P } ^ { v } = g _ { \mathrm { p r i v a t e } } ^ { v } \big ( \mathbf { V } ^ { v } \big ) \in \mathbb R ^ { D } , } \end{array} \quad \quad v \in \mathcal { V } ,\tag{11}
$$

each implemented as a two-layer MLP (Linear–GELU– Dropout–Linear). Two auxiliary losses shape the decomposition. A cross-view Central Moment Discrepancy (CMD) [43] pulls the shared embeddings toward a common distribution by matching their first K central moments across view pairs,

$$
\mathcal { L } _ { \mathrm { a l i g n } } = \sum _ { \{ v , w \} \subset \mathcal { V } } \mathrm { C M D } _ { K } \big ( \mathbf { S } ^ { v } , \mathbf { S } ^ { w } \big ) ,\tag{12}
$$

while an orthogonality penalty decorrelates $\mathbf { S } ^ { v } \perp \mathbf { P } ^ { v }$ within each view and $\mathbf { P } ^ { v } \perp \mathbf { P } ^ { w }$ across views,

$$
\mathcal { L } _ { \mathrm { o r t h } } = \sum _ { v \in \mathcal { V } } \mathrm { D e c o r r } \left( \mathbf { S } ^ { v } , \mathbf { P } ^ { v } \right) + \sum _ { \{ v , w \} \subset \mathcal { V } } \mathrm { D e c o r r } \left( \mathbf { P } ^ { v } , \mathbf { P } ^ { w } \right) ,\tag{13}
$$

where Decorr $\begin{array} { r l r } { ( { \bf A } , { \bf B } ) } & { { } = } & { \| \tilde { \bf A } ^ { \top } \tilde { \bf B } \| _ { F } ^ { 2 } } \end{array}$ is the squared Frobenius cross-product on row-centered, row-normalized matrices A<sup>˜</sup> , B<sup>˜</sup> . Together these objectives drive the decomposition toward its intended structure: $\mathcal { L } _ { \mathrm { a l i g n } }$ collapses the three shared embeddings $\{ \mathbf { S } ^ { v } \} _ { v \in \mathcal { V } }$ toward a common cross-view representation that captures content present across all views, while ${ \mathcal { L } } _ { \mathrm { o r t h } }$ keeps each private $\mathbf { P } ^ { v }$ distinguishable from both its within-view shared embeddings and the other views’ privates, so that $\mathbf { P } ^ { v }$ retains only signal specific to view v.

b) Multi-degree Interaction.: With the shared–private decomposition in place, the synergy module fuses only the three private embeddings $\{ \mathbf { p } ^ { \mathrm { s e q } } , \mathbf { p } ^ { \mathrm { o u t } } , \mathbf { p } ^ { \mathrm { i n } } \}$ , each in $\mathbb { R } ^ { \bar { D } }$ . Synergy, as we defined it above, requires interactions across views; to capture all such interactions of every degree within a single object, we first augment each private with a constant unit,

$$
\widetilde { \mathbf { p } } ^ { v } = [ 1 ; \mathbf { p } ^ { v } ] \in \mathbb { R } ^ { D + 1 } , \qquad v \in \mathcal { V } .\tag{14}
$$

The outer product of the augmented privates yields the fusion tensor

$$
\mathcal { T } = \widetilde { \mathbf { p } } ^ { \mathrm { s e q } } \otimes \widetilde { \mathbf { p } } ^ { \mathrm { o u t } } \otimes \widetilde { \mathbf { p } } ^ { \mathrm { i n } } \in \mathbb { R } ^ { ( D + 1 ) \times ( D + 1 ) \times ( D + 1 ) } .\tag{15}
$$

Because the zeroth coordinate corresponds to the prepended constant unit, the tensor simultaneously encodes first-order terms $( { \bf e . g . } , p _ { i } ^ { \mathrm { s e q } } )$ , second-order interactions $( { \bf e . g . } , p _ { i } ^ { \mathrm { s e q } } p _ { j } ^ { \mathrm { o u t } } )$ , and full third-order interactions $p _ { i } ^ { \mathrm { s e q } } p _ { j } ^ { \mathrm { o u t } } p _ { k } ^ { \mathrm { i n } }$

A synergy embedding is computed via a learned tensor projection, where a weight tensor $\mathcal { W } \in \mathbb { R } ^ { D \times ( D + 1 ) ^ { | \nu | } }$ linearly combines the entries of $\tau$ as follows:

$$
Y _ { \mathrm { s y n } , m } = \sum _ { i , j , k } \mathcal { W } _ { m , i , j , k } \ : \mathcal { T } _ { i j k } .\tag{16}
$$

Direct parameterization is computationally expensive because W alone contains $D ( D + 1 ) ^ { | \nu | }$ parameters. To obtain a more

parameter-efficient formulation, we approximate W using a low-rank tensor factorization following [31]:

$$
{ \boldsymbol { \mathcal { W } } } \approx \sum _ { r = 1 } ^ { R } \mathbf { w } _ { r } ^ { \mathrm { y } } \otimes \mathbf { u } _ { r } ^ { \mathrm { s e q } } \otimes \mathbf { u } _ { r } ^ { \mathrm { o u t } } \otimes \mathbf { u } _ { r } ^ { \mathrm { i n } } ,\tag{17}
$$

collecting the view factors $\mathbf { u } _ { r } ^ { v } \in \mathbb { R } ^ { D + 1 }$ into $\mathbf { U } ^ { v } \in \mathbb { R } ^ { R \times ( D + 1 ) }$ and the output factors ${ \bf w } _ { r } ^ { \mathrm { y } } \in \mathbb { R } ^ { D }$ into ${ \bf W } _ { \mathrm { y } } ~ \in ~ \mathbb { R } ^ { D \times R }$ Substituting (17) into (16) yields an equivalent factorized form in which the fusion reduces to an element-wise Hadamard product across views in rank space,

$$
\mathbf { z } = \bigodot _ { v \in \mathcal { V } } \mathbf { U } ^ { v } \widetilde { \mathbf { p } } ^ { v } \in \mathbb { R } ^ { R } , \qquad \mathbf { Y } _ { \mathrm { s y n } } = \mathbf { W } _ { \mathrm { y } } \mathbf { z } \in \mathbb { R } ^ { D } .\tag{18}
$$

This compression preserves the multi-degree interaction structure described above.

Finally, the synergy output $\mathbf { Y } _ { \mathrm { s y n } }$ is recombined with the shared content as follows:

$$
\mathbf { Y } = \mathrm { L a y e r N o r m } \big ( \mathbf { Y } _ { \mathrm { s y n } } \big ) + \mathrm { L a y e r N o r m } \big ( \bar { \mathbf { S } } \big ) \in \mathbb { R } ^ { D } ,\tag{19}
$$

where $\begin{array} { r } { \bar { \bf S } = \frac { 1 } { | \mathcal { V } | } \sum _ { v \in \mathcal { V } } { \bf S } ^ { v } } \end{array}$ denotes the mean shared embeddings across views.

## G. Trip-distribution Loss

We supervise the learned embeddings using the empirical OD distribution $\textbf { M } \in \ \mathbb { R } ^ { N \times N }$ , obtained from the observed OD matrix.

The synergy embedding $\textbf { Y } \in \ \mathbb { R } ^ { D }$ is mapped to a region representation $\textbf { Z } = \bar { f _ { \mathrm { h e a d } } } ( \mathbf { Y } ) ~ \in ~ \mathbb { R } ^ { N \times \bar { H } }$ by a two-layer MLP head that expands the view dimension D to the final embedding dimension H, and then projected into source and destination embeddings,

$$
\begin{array} { r } { \mathbf { Z } ^ { \mathrm { s r c } } = \mathbf { Z } \mathbf { W } _ { \mathrm { s r c } } ^ { \top } , \qquad \mathbf { Z } ^ { \mathrm { d s t } } = \mathbf { Z } \mathbf { W } _ { \mathrm { d s t } } ^ { \top } . } \end{array}\tag{20}
$$

Then, the mobility distributions are computed as

$$
\widehat { \mathbf { Q } } _ { i , j } ^ { \mathrm { o u t } } = \frac { \exp \left( \mathbf { z } _ { i } ^ { \mathrm { s r c } } { \cdot } \mathbf { z } _ { j } ^ { \mathrm { d s t } } \right) } { \sum _ { k = 1 } ^ { N } \exp \left( \mathbf { z } _ { i } ^ { \mathrm { s r c } } { \cdot } \mathbf { z } _ { k } ^ { \mathrm { d s t } } \right) } ,\tag{21}
$$

$$
\widehat { \mathbf { Q } } _ { i , j } ^ { \mathrm { i n } } = \frac { \exp \bigl ( \mathbf z _ { i } ^ { \mathrm { d s t } } \cdot \mathbf z _ { j } ^ { \mathrm { s r c } } \bigr ) } { \sum _ { k = 1 } ^ { N } \exp \bigl ( \mathbf z _ { i } ^ { \mathrm { d s t } } \cdot \mathbf z _ { k } ^ { \mathrm { s r c } } \bigr ) } .\tag{22}
$$

The predicted transition distributions are matched against the empirical OD distributions in both directions using the following loss:

$$
\mathcal { L } _ { \mathrm { m o b } } = - \sum _ { i , j } \left[ \mathbf { M } _ { i , j } \log \widehat { \mathbf { Q } } _ { i , j } ^ { \mathrm { o u t } } + \mathbf { M } _ { j , i } \log \widehat { \mathbf { Q } } _ { i , j } ^ { \mathrm { i n } } \right] .\tag{23}
$$

H. Total Training Objective

MoSS is trained end-to-end by minimizing

$$
\mathcal { L } _ { \mathrm { t o t a l } } = \mathcal { L } _ { \mathrm { m o b } } + \lambda _ { \mathrm { a l i g n } } \mathcal { L } _ { \mathrm { a l i g n } } + \lambda _ { \mathrm { o r t h } } \mathcal { L } _ { \mathrm { o r t h } } ,\tag{24}
$$

where $\lambda _ { \mathrm { a l i g n } } , \lambda _ { \mathrm { o r t h } } \ge 0$ control the strength of cross-view shared alignment and dual orthogonality terms, respectively.

TABLE I: Dataset statistics and sources. Counts (regions, trips, and events) are reported as totals. Income is reported as the average of per-region median household income.
<table><tr><td>Data</td><td>NYC</td><td>CHI</td><td>Source</td></tr><tr><td>Regions</td><td>180</td><td>77</td><td>U.S. Census Bureau [46] and CHIDP [45]</td></tr><tr><td>Taxi trips</td><td>9,779,714</td><td>3,368,049</td><td>NYCOD [44] and CHIDP [45]</td></tr><tr><td>Crime</td><td>35,335</td><td>18,200</td><td>NYCOD [44] and CHIDP [45]</td></tr><tr><td>Income</td><td>$84,600</td><td>$74,734</td><td>NYCOD [44] and Chicago Health Atlas [47]</td></tr><tr><td>Service calls</td><td>516,187</td><td>24,350</td><td>NYCOD [44] and CHIDP [45]</td></tr></table>

## V. EXPERIMENTS

## A. Datasets and Tasks

To evaluate the effectiveness of the proposed model MoSS, experiments are conducted using real-world data from two U.S. cities: New York City (NYC) and Chicago (CHI). Table I summarizes the datasets.

The region division uses 180 census tracts in NYC and 77 community areas in CHI. For each region, we aggregate taxitrip records by pickup and drop-off region to form an hourly OD tensor, which serves as the raw mobility input to MoSS. The mobility data is obtained from the NYC Taxi & Limousine Commission [44] for NYC and from the City of Chicago data portal [45] for CHI. The embeddings of the region are evaluated in three downstream prediction tasks.

• Crime prediction. Predict the annual crime count of each region from its embedding.

• Income prediction. Predict the median household income of each region.

• Service-call prediction. Predict the annual 311 servicecall count for each region.

## B. Implementation Details

a) Zigzag persistence.: We extract only $H _ { 0 }$ persistence, which tracks connectivity by counting connected components. For each region, we compute one zigzag $H _ { 0 }$ diagram per direction (inflow/outflow), which yields two persistence-diagram views per region.

b) Persistence-diagram encoder.: The PD encoder of Eq. (10) takes each persistence point as a 3-dim feature (birth, death, persistence) and applies $L _ { \phi } = 4$ point-wise convolutions with widths $C _ { 1 } = 3 2 , C _ { 2 } = 6 4 , C _ { 3 } = 1 2 8 \ /$ , and final width D, followed by max-pooling along the point axis.

c) Temporal encoder.: The two outflow / inflow time series are stacked into a single 2-channel input and processed jointly by a dilated TCN with hidden width $C _ { h } = 3 2 , L _ { \mathrm { t c n } } = 1 0$ residual blocks with kernel size $k _ { \mathrm { t s } } = 3 .$ , dilation $2 ^ { \ell }$ at the ℓ-th block, and max-pooling over the time axis to width D.

d) Optimization.: Adam optimizer is used throughout. Hyperparameters are tuned separately for each city. For NYC, we set the learning rate to $1 0 ^ { - 3 }$ , the alignment weight $\lambda _ { \mathrm { a l i g n } }$ to 1, and the synergy rank R to 8. For Chicago, the learning rate is $8 \times 1 0 ^ { - 4 } , \lambda _ { \mathrm { a l i g n } } = 5 0$ , and $R = 4$ . Shared settings across both cities include $\lambda _ { \mathrm { o r t h } } = 5 0$ , output dimension $H = 1 4 4$ , view dimension $D = 1 6 ,$ and dropout rate 0.1.

TABLE II: Categorization of baselines and MoSS along modality and fusion strategy.
<table><tr><td>Method</td><td>Modality</td><td>Fusion strategy</td></tr><tr><td>MVURE [4]</td><td>mobility + POI + check-in</td><td>attention-based</td></tr><tr><td>MGFN [5]</td><td>mobility</td><td>attention-based</td></tr><tr><td>HREP [12]</td><td>mobility + POI + geographic neighbor</td><td>attention-based</td></tr><tr><td>ReCP [7]</td><td>mobility + POI</td><td>contrastive</td></tr><tr><td>MVJC [15]</td><td>mobility + POI + check-in</td><td>contrastive</td></tr><tr><td>ComSRE [16]</td><td>mobility + POI</td><td>shared-private decomposition</td></tr><tr><td>MoSS (ours)</td><td>mobility</td><td>shared-private + synergy</td></tr></table>

## C. Baselines

We compare against six representative urban region embedding methods that span the main families used to integrate human mobility with auxiliary information.

• MVURE [4] adds a self-attention layer over view-specific embeddings and fuses them with adaptive view weights.

• MGFN [5] learns separate source and destination embeddings from a multi-graph fusion of typical-week OD patterns.

• HREP [12] couples a heterogeneous region encoder with prompt-based downstream specialization, supervised by mobility, geographical, and POI signals.

• ReCP [7] contrasts attribute and mobility autoencoders with a dual prediction objective.

• MVJC [15] adds a structure-aware contrastive objective to mitigate the false-negative problem among functionally similar regions.

• ComSRE [16] disentangles commonality and viewspecific representations across multiple urban views through attention-based fusion and contrastive learning.

Table II positions these baselines and MoSS along modality and fusion strategy.

## D. Evaluation Protocol

Following prior work [4], [7], [12], [13], we evaluate the frozen region embeddings with a Ridge regressor under 5- fold cross-validation, as the number of regions is small. We report the mean absolute error (MAE), root mean squared error (RMSE), and coefficient of determination $R ^ { 2 }$ as mean and standard deviation over 5 runs.

## E. Main Results

We compare MOSS against six recent region-representation baselines—MVURE, MGFN, HREP, ReCP, MVJC, and ComSRE—on three downstream prediction tasks (crime, income, service call) in two cities, New York City (N=180) and Chicago (N=77). All baselines are trained with their authors released code. Each configuration is evaluated with five runs; we report mean ± std of MAE, RMSE, and $R ^ { 2 }$ in Table III. a) Overall performance.: MOSS achieves state-of-the-art performance on every task in terms of MAE, RMSE, and $R ^ { 2 }$ The largest $R ^ { 2 }$ gains over the strongest per-column baseline appear on service call (+0.040 on NY over MVURE; +0.100 on CHI over HREP) and on income on CHI (+0.069 over HREP/MGFN). On crime, RMSE drops by 12.0% on

NY (HREP, $8 8 . 4 3  7 7 . 8 2 )$ and 4.4% on CHI (ComSRE, 117.94 → 112.77). Crucially, the same model configuration and the same set of region embeddings are used for all three tasks within a city, so the uniform gains reflect a more transferable representation rather than per-task tuning— several baselines that are competitive on one task collapse on another (e.g. ComSRE is the strongest baseline on CHI crime, $R ^ { 2 } { = } 0 . 5 4 8$ , but falls to 0.187 on CHI income; ReCP’s CHI service call $R ^ { 2 }$ varies by ±0.157).

b) Cross-city consistency.: Existing baselines exhibit performance variability across cities and tasks. For instance, Com-SRE achieves the best performance on CHI crime prediction $( R ^ { 2 } { = } 0 . 5 4 8 )$ but drops to the middle of the pack on NY crime (0.450, compared to HREP’s 0.642). Likewise, MGFN attains strong performance on CHI income prediction $( R ^ { 2 } { = } 0 . 6 3 0 )$ yet degrades markedly on NY income (0.460), while HREP, which leads on NY crime (0.642), is overtaken on CHI crime (0.513). In contrast, MOSS ranks first in mean across all tasks and both cities, demonstrating stable generalization across different urban structures, prediction targets, and target scales.

c) Stability.: MOSS attains the best mean on every NY task together with consistently low seed variance — the lowest on NY crime and service call — and remains competitive in variance on CHI while maintaining state-of-the-art predictive performance. In comparison, several baselines show either larger variability or lower overall accuracy depending on the task. For example, ReCP on CHI service call exhibits a large fluctuation (±0.157), while MGFN on CHI service call and MVURE across CHI tasks also display relatively high variance (0.205 and 0.087–0.094, respectively). Conversely, some methods such as ComSRE and MVJC achieve relatively small standard deviations on NY tasks (0.020), but with lower mean performance than MOSS.

## F. Ablation Study

To isolate the contribution of each component of our model, we ablate four modules:

• w/o Seq. Replace the dilated-TCN encoding of each region’s hourly inflow/outflow time series with a rowwise embedding of its row in the (row-normalized) OD matrix.

• w/o Strct. Remove structure stream.

• w/o SP. Remove the shared–private decomposition: the three raw view embeddings feed the synergy module directly, with no shared embedding and no align ment/orthogonality regularization.

• w/o Syn. Replace the synergy module with a plain concatenation of the shared embeddings and the three private embeddings.

The results for all four variants are shown in Table IV and we make the following observations.

a) Effectiveness of sequence stream.: Replacing the sequence stream with a row-wise OD-row embedding (w/o Seq) degrades the model, with magnitude varying by city and task. The resulting drop in downstream performance shows that the time-evolving volume semantics carries predictive information that neither the aggregate OD-row embedding nor the topological views recover.

b) Effectiveness of structure stream.: Dropping the persistence-diagram views (w/o Strct) consistently degrades performance across all settings. Prediction across cities and tasks empirically confirms that this persistent connectivity structure carries signal that neither the raw temporal sequence nor a single-view embedding can substitute.

c) Effectiveness of shared–private decomposition.: Removing the shared–private decomposition (w/o SP) feeds the raw view embeddings directly into the synergy module, with no shared embedding and no alignment/orthogonality regularization. The drop is consistent on NYC (e.g. Crime $0 . 7 2 3 \  \ 0 . 6 2 2$ Income $0 . 5 2 0  0 . 4 3 2 )$ and particularly severe on CHI Crime $( 0 . 5 8 7  0 . 2 0 8 )$ . This shows that the synergy module benefits from private embeddings that are disentangled from a common shared component, through shared–private decomposition.

d) Effectiveness of synergy module.: Replacing the multilinear interaction with a plain concatenation of the shared embedding and the three private embeddings (w/o Syn) also degrades performance across both cities. This confirms that the performance gain does not arise merely from combining multiple view embeddings but from explicitly modeling their multiplicative interactions through the synergy mechanism.

## G. Hyperparameter Analysis

![](images/ded4790a49ad8b24cd26851e5e7e172ad48a439f4b2107fafc5e0be242735013.jpg)

![](images/bf93f5824f9026682f9130033a5d0e100ecd42ed793a705723f56c8c98f1795d.jpg)

![](images/3ed1ada39c0a9aa5a4e1c90bd25cc6530703311fbf5ee598e0fb6087a84c800f.jpg)  
Fig. 4: Impact of (a) alignment weight $\lambda _ { \mathrm { a l i g n } } .$ (b) orthogonality weight $\lambda _ { \mathrm { o r t h } } .$ , and (c) final embedding dimension $H ,$ on NYC and CHI. Each curve reports per-task $R ^ { 2 \prime }$ ↑.

We analyze MOSS’s sensitivity to the following hyperparameters.

• Alignment weight $\lambda _ { \mathrm { a l i g n } } \in \{ 0 . 1 , \ 1 , \ 1 0 , \ 5 0 \}$

• Orthogonality weight $\lambda _ { \mathrm { o r t h } } \in \{ 0 . 1 , \ 1 , \ 1 0 , \ 5 0 \}$

• Final embedding dimension $H \in \{ 6 4 , \ 9 6 , \ 1 4 4 , \ 1 9 2 \}$ a) Weight parameters $\lambda _ { a l i g n }$ and $\lambda _ { o r t h } .$ : The two weights affect the three tasks differently (Figure 4a,b). Crime is the most robust, varying within 0.02 in $R ^ { 2 }$ across both weight ranges on both cities. Service call is the most responsive, and the two cities respond in opposite directions to $\lambda _ { \mathrm { a l i g n } } \mathrm { : }$ CHI rises monotonically from 0.466 at $\lambda _ { \mathrm { a l i g n } } { = } 0 . 1$ to 0.601 at $\lambda _ { \mathrm { a l i g n } } { = } 5 0$ while NY declines from 0.465 to 0.426 across the same range.

TABLE III: Downstream task performance on (a) New York and (b) Chicago. Each cell reports MAE ↓ /RMSE ↓ $/ R ^ { 2 } \uparrow$ as mean ± std over 5 runs, evaluated with a frozen embedding and a Ridge regressor under 5-fold cross-validation; best result in bold.  
(a) New York City (N = 180 regions)
<table><tr><td></td><td colspan="3">Crime</td><td colspan="3">Income</td><td colspan="3">Service Call</td></tr><tr><td>Model</td><td>MAE↓</td><td>RMSE↓</td><td> $R ^ { 2 } \uparrow$ </td><td>MAE↓</td><td>RMSE↓</td><td> $R ^ { 2 } \uparrow$ </td><td>MAE↓</td><td>RMSE↓</td><td> $R ^ { 2 } \uparrow$ </td></tr><tr><td>MVURE</td><td> $6 6 . 9 7 \pm 2 . 5 3 $ </td><td> $9 1 . 3 2 \pm 3 . 1 1$ </td><td> $0 . 6 1 8 \pm 0 . 0 2 6$ </td><td> $2 4 , 6 7 9 . 2 5 \pm 5 5 5 . 0 0$ </td><td> $3 4 , 0 3 9 . 7 7 \pm 1 , 4 1 9 . 3 2$ </td><td> $0 . 4 5 4 \pm 0 . 0 4 6$ </td><td> $1 , 3 8 8 . 5 2 \pm 7 3 . 9 8 $ </td><td> $2 , 1 1 9 . 6 1 \pm 7 9 . 5 2$ </td><td> $0 . 4 0 2 \pm 0 . 0 4 5$ </td></tr><tr><td>MGFN</td><td> $7 2 . 6 5 \pm 1 . 9 1$ </td><td> $9 5 . 9 2 \pm 3 . 3 3$ </td><td> $0 . 5 7 9 \pm 0 . 0 2 9$ </td><td> $2 5 , 3 4 8 . 5 1 \pm 4 5 3 . 3 4$ </td><td> $3 3 , 8 9 2 . 8 6 \pm 9 0 6 . 8 7$ </td><td> $0 . 4 6 0 \pm 0 . 0 2 9$ </td><td> $1 , 5 6 1 . 4 9 \pm 6 4 . 5 1$ </td><td> $2 , 2 9 7 . 6 5 \pm 7 4 . 4 3$ </td><td> $0 . 2 9 7 \pm 0 . 0 4 6$ </td></tr><tr><td>HREP</td><td> $6 7 . 5 2 \pm 3 . 4 0$ </td><td> $8 8 . 4 3 \pm 3 . 8 2 $ </td><td> $0 . 6 4 2 \pm 0 . 0 3 0$ </td><td> $2 4 , 0 5 0 . 4 1 \pm 6 0 9 . 8 6$ </td><td> $3 2 , 8 1 1 . 2 6 \pm 6 1 3 . 6 4$ </td><td> $0 . 4 9 4 \pm 0 . 0 1 9$ </td><td> $1 , 4 2 1 . 5 7 \pm 5 9 . 5 6$ </td><td> $2 , 1 7 4 . 0 2 \pm 5 3 . 0 1$ </td><td> $0 . 3 7 1 \pm 0 . 0 3 1$ </td></tr><tr><td>ReCP</td><td> $7 8 . 4 7 \pm 2 . 2 5$ </td><td> $1 0 5 . 6 2 \pm 3 . 7 1$ </td><td> $0 . 4 8 9 \pm 0 . 0 3 5$ </td><td> $2 3 , 5 9 1 . 5 1 \pm 7 8 6 . 4 2$ </td><td> $3 3 , 7 5 2 . 5 7 \pm 9 0 4 . 2 4$ </td><td> $0 . 4 6 4 \pm 0 . 0 2 9$ </td><td> $1 , 6 3 8 . 6 9 \pm 1 4 . 1 6$ </td><td> $2 , 3 7 9 . 7 5 \pm 9 2 . 4 5$ </td><td> $0 . 2 4 6 \pm 0 . 0 5 8$ </td></tr><tr><td>MVJC</td><td> $9 6 . 5 6 \pm 1 . 6 1$ </td><td> $1 2 6 . 2 0 \pm 1 . 4 5$ </td><td> $0 . 2 7 2 \pm 0 . 0 1 7$ </td><td> $2 4 , 8 5 4 . 6 7 \pm 3 9 5 . 8 6$ </td><td> $3 4 , 0 4 8 . 9 7 \pm 6 2 3 . 9 7$ </td><td> $0 . 4 5 5 \pm 0 . 0 2 0$ </td><td> $1 , 7 3 0 . 2 7 \pm 3 2 . 0 7$ </td><td> $2 , 5 5 0 . 7 6 \pm 1 9 . 6 9$ </td><td> $0 . 1 3 5 \pm 0 . 0 1 3$ </td></tr><tr><td>ComSRE</td><td> $8 3 . 3 2 \pm 1 . 2 3$ </td><td> $1 0 9 . 6 2 \pm 1 . 6 7$ </td><td> $0 . 4 5 0 \pm 0 . 0 1 7$ </td><td> $2 5 , 9 7 2 . 6 8 \pm 2 0 9 . 8 4$ </td><td> $3 6 , 7 8 5 . 3 9 \pm 3 5 2 . 5 8$ </td><td> $0 . 3 6 4 \pm 0 . 0 1 2$ </td><td> $1 , 5 2 7 . 6 4 \pm 2 1 . 5 0$ </td><td> $2 , 2 3 3 . 1 9 \pm 2 7 . 1 8$ </td><td> $0 . 3 3 7 \pm 0 . 0 1 6$ </td></tr><tr><td>MoSS (ours)</td><td> ${ \pm } 8 . 5 4 \pm 1 . 4 7$ </td><td> ${ 7 7 . 8 2 \pm 1 . 8 8 }$ </td><td> $\mathbf { 0 . 7 2 3 \pm 0 . 0 1 3 }$ </td><td> $\mathbf { 2 2 } , \mathbf { 5 7 0 . 6 1 } \pm \mathbf { 5 5 } 3 . 3 6$ </td><td> $\mathbf { 3 1 , 9 6 1 . 6 6 \pm 5 4 6 . 6 5 }$ </td><td> ${ \bf 0 . 5 2 0 \pm 0 . 0 1 7 }$ </td><td> $\mathbf { 1 } , 3 5 \mathbf { 0 } . 5 \mathbf { 1 } \pm 2 5 . 8 5$ </td><td> $\mathbf { 2 , 0 4 8 . 4 9 \pm 1 7 . 7 0 }$ </td><td> $\mathbf { 0 . 4 4 2 \pm 0 . 0 1 0 }$ </td></tr></table>

(b) Chicago (N = 77 community areas)
<table><tr><td></td><td colspan="3">Crime</td><td colspan="3">Income</td><td colspan="3">Service Call</td></tr><tr><td>Model</td><td> $\mathrm { M A E \downarrow }$ </td><td> $\begin{array} { r } { \mathrm { R M S E } \downarrow } \end{array}$ </td><td> $R ^ { 2 } \uparrow$ </td><td>MAE↓</td><td> $\mathrm { R M S E } \downarrow$ </td><td> $R ^ { 2 } \uparrow$ </td><td> $\mathbf { M A E } \downarrow$ </td><td>RMSE↓</td><td> $R ^ { 2 } \uparrow$ </td></tr><tr><td>MVURE</td><td> $1 0 6 . 6 5 \pm 9 . 3 3$ </td><td> $1 3 7 . 6 3 \pm 1 1 . 0 4$ </td><td> $0 . 3 8 6 \pm 0 . 0 9 4$ </td><td> $1 7 , 5 9 6 . 4 8 \pm 1 , 3 3 8 . 9 5$ </td><td>22,888.09 ± 1,753.51</td><td> $0 . 4 3 6 \pm 0 . 0 8 7$ </td><td> $1 9 6 . 3 2 \pm 1 5 . 7 0$ </td><td> $2 7 7 . 7 4 \pm 2 1 . 9 6$ </td><td> $0 . 3 9 7 \pm 0 . 0 9 3$ </td></tr><tr><td>MGFN</td><td> $1 0 5 . 9 8 \pm 4 . 9 5$ </td><td> $1 4 1 . 9 9 \pm 7 . 5 8$ </td><td> $0 . 3 4 8 \pm 0 . 0 6 9$ </td><td> $1 4 , 2 5 1 . 8 2 \pm 1 , 3 3 7 . 3 3$ </td><td> $1 8 , 5 0 0 . 4 6 \pm 1 , 6 3 0 . 9 4$ </td><td> $0 . 6 3 0 \pm 0 . 0 6 7$ </td><td> $2 3 0 . 3 5 \pm 2 9 . 0 3$ </td><td> $3 2 7 . 0 0 \pm 3 8 . 8 5$ </td><td> $0 . 1 5 8 \pm 0 . 2 0 5$ </td></tr><tr><td>HREP</td><td> $9 1 . 3 1 \pm 5 . 3 1$ </td><td> $1 2 2 . 8 0 \pm 5 . 0 9$ </td><td> $0 . 5 1 3 \pm 0 . 0 4 1$ </td><td> $1 3 , 9 8 5 . 9 9 \pm 1 , 1 7 6 . 4 1$ </td><td> $1 8 , 5 0 6 . 6 4 \pm 1 , 6 7 6 . 3 3$ </td><td> $0 . 6 3 0 \pm 0 . 0 6 8$ </td><td> $1 7 8 . 3 6 \pm 4 . 5 9$ </td><td> $2 5 3 . 4 1 \pm 8 . 2 6$ </td><td> $0 . 5 0 1 \pm 0 . 0 3 2$ </td></tr><tr><td>ReCP</td><td> $9 9 . 2 2 \pm 8 . 8 2 $ </td><td> $1 3 5 . 9 6 \pm 8 . 2 7$ </td><td> $0 . 4 0 2 \pm 0 . 0 7 3$ </td><td> $1 6 , 0 3 8 . 7 5 \pm 1 , 4 8 6 . 2 6$ </td><td>22,514.94± 1,845.25</td><td> $0 . 4 5 3 \pm 0 . 0 8 8$ </td><td> $1 9 5 . 5 4 \pm 2 6 . 8 3$ </td><td> $2 8 0 . 6 2 \pm 3 7 . 1 7$ </td><td> $0 . 3 7 8 \pm 0 . 1 5 7$ </td></tr><tr><td>MVJC</td><td> $1 1 4 . 0 8 \pm 3 . 6 8$ </td><td> $1 4 8 . 2 1 \pm 3 . 2 2$ </td><td> $0 . 2 9 2 \pm 0 . 0 3 0$ </td><td> $1 6 , 8 7 8 . 6 3 \pm 4 3 3 . 1 6$ </td><td> $2 1 , 6 3 9 . 3 7 \pm 5 2 8 . 6 2$ </td><td> $0 . 4 9 8 \pm 0 . 0 2 5$ </td><td> $1 8 5 . 1 3 \pm 8 . 0 0$ </td><td> $2 7 1 . 2 9 \pm 1 0 . 9 3$ </td><td> $0 . 4 2 8 \pm 0 . 0 4 5$ </td></tr><tr><td>ComSRE</td><td> $8 3 . 8 5 \pm 4 . 8 8$ </td><td> $1 1 7 . 9 4 \pm 1 0 . 3 7$ </td><td> $0 . 5 4 8 \pm 0 . 0 7 7$ </td><td> $2 0 , 4 2 3 . 1 6 \pm 6 8 5 . 8 5$ </td><td> $2 7 , 5 2 4 . 3 1 \pm 1 , 3 3 7 . 3 2$ </td><td> $0 . 1 8 7 \pm 0 . 0 8 1$ </td><td> $2 3 6 . 2 7 \pm 3 . 2 1$ </td><td> $3 1 4 . 8 5 \pm 5 . 5 3 $ </td><td> $0 . 2 3 0 \pm 0 . 0 2 7$ </td></tr><tr><td>MoSS (ours)</td><td> $\mathbf { 8 2 . 4 5 \pm 7 . 9 0 }$ </td><td> ${ \bf 1 1 2 . 7 7 \pm 9 . 8 8 }$ </td><td> $\mathbf { 0 . 5 8 7 \pm 0 . 0 7 2 }$ </td><td> $\mathbf { 1 2 , 4 8 5 . 6 4 \pm 8 6 9 . 8 3 }$ </td><td> ${ \bf 1 6 , 7 3 3 . 4 3 \pm 9 5 4 . 4 4 }$ </td><td> $\mathbf { 0 . 6 9 9 \pm 0 . 0 3 5 }$ </td><td> ${ \bf 1 5 6 . 6 9 \pm 5 . 8 8 }$ </td><td> $\pm 2 6 . 5 7 \pm 1 0 . 4 8$ </td><td> $\mathbf { 0 . 6 0 1 \pm 0 . 0 3 7 }$ </td></tr></table>

TABLE IV: Ablation study on NY and CHI. We remove or replace one module at a time: w/o Seq replaces the dilated-TCN sequence encoder with a row-wise OD-row embedding, w/o Strct drops the structure stream, w/o SP removes the shared–private decomposition (feeding raw view embeddings into the synergy module), and w/o Syn replaces the synergy module with plain concatenation. Each cell reports per-task $R ^ { 2 } \uparrow$ (mean over 5 runs); best results in bold.
<table><tr><td></td><td colspan="3">NY</td><td colspan="3">CHI</td></tr><tr><td>Variant</td><td>Crime</td><td>Income</td><td>Service Call</td><td>Crime</td><td>Income</td><td>Service Call</td></tr><tr><td>Full</td><td>0.723</td><td>0.520</td><td>0.442</td><td>0.587</td><td>0.699</td><td>0.601</td></tr><tr><td>w/o Seq</td><td>0.558</td><td>0.441</td><td>0.378</td><td>0.404</td><td>0.577</td><td>0.284</td></tr><tr><td>w/o Strct</td><td>0.646</td><td>0.481</td><td>0.406</td><td>0.442</td><td>0.504</td><td>0.422</td></tr><tr><td>w/o SP</td><td>0.622</td><td>0.432</td><td>0.395</td><td>0.208</td><td>0.576</td><td>0.326</td></tr><tr><td>w/o Syn</td><td>0.582</td><td>0.477</td><td>0.395</td><td>0.414</td><td>0.454</td><td>0.426</td></tr></table>

## H. Model Size and Training Efficiency

TABLE V: Trainable parameters and per-epoch training time of MOSS and the six baselines on NY and CHI; lower is better for both metrics.
<table><tr><td></td><td colspan="2"> $\mathsf { N Y } \left( N = 1 8 0 \right)$ </td><td colspan="2"> $\mathrm { C H I } \left( N = 7 7 \right)$ </td></tr><tr><td>Model</td><td>Params</td><td>ms/epoch</td><td>Params</td><td>ms/epoch</td></tr><tr><td>MVURE</td><td>222 K</td><td>91.0</td><td>222 K</td><td>23.7</td></tr><tr><td>MGFN</td><td>13.39M</td><td>27.3</td><td>3.46M</td><td>24.5</td></tr><tr><td>HREP</td><td>188 K</td><td>26.6</td><td>188 K</td><td>26.2</td></tr><tr><td>ReCP</td><td>289 K</td><td>155.4</td><td>235 K</td><td>166.9</td></tr><tr><td>MVJC</td><td>829 K</td><td>51.4</td><td>558 K</td><td>49.5</td></tr><tr><td>ComSRE</td><td>458 K</td><td>29.4</td><td>353 K</td><td>28.7</td></tr><tr><td>MoSS (ours)</td><td>157 K</td><td>62.0</td><td>157K</td><td>61.6</td></tr></table>

b) Final embedding dimension.: On CHI, all three tasks peak at $H ~ = ~ 1 4 4$ and decline at $H \ = \ 1 9 2$ , with the worst performance at H = 64 (Figure 4c). NY shows two different patterns: crime and income improve monotonically up to $H \ = \ 1 9 2 \ ( 0 . 6 9 9 \ \to \ 0 . 7 2 9$ and $0 . 5 0 1  0 . 5 3 2 )$ , whereas service call peaks early at $H = 9 6 ~ ( 0 . 4 7 6 )$ and declines at larger dimensions.

We additionally compare MOSS against the six baselines along two practical axes: number of trainable parameters and wall-clock time per training epoch. Parameter counts are obtained by enumerating all trainable tensors in each model after instantiation with the hyperparameters used to produce Table III. Per-epoch time is measured on a single NVIDIA RTX 3090 (24 GB) with 20 measured epochs after 3 warmup epochs. Numbers are reported in Table V.

a) Parameter count.: MOSS uses ∼157K trainable parameters on both cities, the smallest among the compared methods. It uses fewer parameters than HREP (188K), and is 85× smaller than MGFN on NY (13.39M → 157K) and 22× smaller on CHI $( 3 . 4 6 \mathbf { M }  1 5 7 \mathrm { K } )$

b) Training time.: MOSS incurs a moderately higher perepoch cost than most baselines, with only MVURE (NY) and ReCP being slower, because its sequence and structure encoders together with the multi-degree synergy fusion add perstep overhead beyond plain attention or contrastive objectives. This reflects a deliberate trade-off: these components raise the per-epoch cost yet keep MOSS compact in parameters and drive its accuracy gains.

## VI. CONCLUSION

We presented MoSS, an urban region embedding framework that combines mobility time series with zigzag persistence diagrams of the time-evolving connectivity-graphs. A shared– private decomposition, paired with cross-view alignment and orthogonality losses, fuses these complementary signals into a unified region representation refined by multi-degree interaction. Although MoSS draws solely on mobility data, it outperforms urban region embedding baselines that additionally rely on auxiliary modalities such as POI and check-in, while using a fraction of the parameters of the strongest baselines. We view the temporal–topological pairing as a general design pattern for representing urban regions and leave its extension to additional modalities, longer time scales, and cross-city transfer for future work.

## REFERENCES

[1] W. Zhang, J. Han, Z. Xu, H. Ni, T. Lyu, H. Liu, and H. Xiong, “Towards urban general intelligence: A review and outlook of urban foundation models,” ACM Trans. Intell. Syst. Technol., 2026.

[2] H. Wang and Z. Li, “Region representation learning via mobility flow,” in Proc. ACM Int. Conf. Inf. Knowl. Manage., 2017, pp. 237–246.

[3] Z. Yao, Y. Fu, B. Liu, W. Hu, and H. Xiong, “Representing urban functions through zone embedding with human mobility patterns,” in Proc. Int. Joint Conf. Artif. Intell., 2018, pp. 3919–3925.

[4] M. Zhang, T. Li, Y. Li, and P. Hui, “Multi-view joint graph representation learning for urban region embedding,” in Proc. Int. Joint Conf. Artif. Intell., 2020, pp. 4431–4437.

[5] S. Wu, X. Yan, X. Fan, S. Pan, S. Zhu, C. Zheng, M. Cheng, and C. Wang, “Multi-graph fusion networks for urban region embedding,” in Proc. Int. Joint Conf. Artif. Intell., 2022.

[6] N. Kim and Y. Yoon, “Effective urban region representation learning using heterogeneous urban graph attention network (hugat),” IEEE Access, 2025.

[7] Z. Li, W. Huang, K. Zhao, M. Yang, Y. Gong, and M. Chen, “Urban region embedding via multi-view contrastive prediction,” in Proc. AAAI Conf. Artif. Intell, 2024, pp. 8724–8732.

[8] T. Li, S. Xin, Y. Xi, S. Tarkoma, P. Hui, and Y. Li, “Predicting multilevel socioeconomic indicators from structural urban imagery,” in Proc. ACM Int. Conf. Inf. Knowl. Manage., 2022, pp. 3282–3291.

[9] J. Yuan, Y. Zheng, and X. Xie, “Discovering regions of different functions in a city using human mobility and pois,” in Proc. ACM SIGKDD Int. Conf. Knowl. Discovery Data Mining, 2012, pp. 186–194.

[10] Y. Zheng, L. Capra, O. Wolfson, and H. Yang, “Urban computing: concepts, methodologies, and applications,” ACM Trans. Intell. Syst. Technol., vol. 5, no. 3, pp. 1–55, 2014.

[11] L. Zhang, C. Long, and G. Cong, “Region embedding with intra and inter-view contrastive learning,” IEEE Trans. Knowl. Data Eng., vol. 35, no. 9, pp. 9031–9036, 2022.

[12] S. Zhou, D. He, L. Chen, S. Shang, and P. Han, “Heterogeneous region embedding with prompt learning,” in Proc. AAAI Conf. Artif. Intell, 2023, pp. 4981–4989.

[13] F. Sun, J. Qi, Y. Chang, X. Fan, S. Karunasekera, and E. Tanin, “Urban region representation learning with attentive fusion,” in Proc. IEEE Int. Conf. Data Eng., 2024, p. 4409–4421.

[14] J. Jin, Y. Song, D. Kan, H. Zhu, X. Sun, Z. Li, X. Sun, and J. Zhang, “Urban region pre-training and prompting: a graph-based approach,” in Proc. ACM SIGKDD Int. Conf. Knowl. Discovery Data Mining, 2025, p. 1071–1082.

[15] Y. Lin, Y. Xu, L. Jiang, and P. Wang, “Comprehensive urban region representation learning via multi-view joint learning and contrastive learning,” in Proc. AAAI Conf. Artif. Intell, 2026, p. 15261–15268.

[16] Z. Li, H. Jia, K. Zhao, W. Huang, and M. Chen, “Multi-view urban region embedding via commonality-specificity disentanglement,” in Proc. ACM SIGKDD Int. Conf. Knowl. Discovery Data Mining, 2026.

[17] P. Wang, Y. Fu, J. Zhang, X. Li, and D. Lin, “Learning urban community structures: A collective embedding perspective with periodic spatialtemporal mobility graphs,” ACM Trans. Intell. Syst. Technol., vol. 9, no. 6, pp. 1–28, 2018.

[18] G. Carlsson and V. De Silva, “Zigzag persistence,” Found. Comput. Math., vol. 10, no. 4, pp. 367–405, 2010.

[19] M. Chen, H. Jia, Z. Li, W. Huang, K. Zhao, Y. Gong, H. Xu, and H. Dai, “Region embedding with adaptive correlation discovery for predicting urban socioeconomic indicators,” IEEE Trans. Knowl. Data Eng., vol. 38, no. 2, pp. 1280–1291, 2026.

[20] Y. Liu, X. Zhang, J. Ding, Y. Xi, and Y. Li, “Knowledge-infused contrastive learning for urban imagery-based socioeconomic prediction,” in Proc. Int. World Wide Web Conf., 2023, pp. 4150–4160.

[21] Q. Zhang, X. Gao, H. Wang, D. Huang, S.-M. Yiu, and H. Yin, “Hgaurban: Heterogeneous graph autoencoding for urban spatial-temporal learning,” in Proc. ACM Int. Conf. Inf. Knowl. Manage., 2025, p. 4139–4148.

[22] Y. Zhang, Y. Fu, P. Wang, X. Li, and Y. Zheng, “Unifying interregion autocorrelation and intra-region structures for spatial embedding via collective adversarial learning,” in Proc. ACM SIGKDD Int. Conf. Knowl. Discovery Data Mining, 2019, pp. 1700–1708.

[23] Y. Li, W. Huang, G. Cong, H. Wang, and Z. Wang, “Urban region representation learning with openstreetmap building footprints,” in Proc. ACM SIGKDD Int. Conf. Knowl. Discovery Data Mining, 2023, pp. 1363–1373.

[24] F. Sun, Y. Chang, E. Tanin, S. Karunasekera, and J. Qi, “Flexireg: Flexible urban region representation learning,” in Proc. ACM SIGKDD Int. Conf. Knowl. Discovery Data Mining, 2025, p. 2702–2713.

[25] Y. Fu, P. Wang, J. Du, L. Wu, and X. Li, “Efficient region embedding with multi-view spatial networks: A perspective of locality-constrained spatial autocorrelations,” in Proc. AAAI Conf. Artif. Intell, vol. 33, 2019, p. 906–913.

[26] K. Bousmalis, G. Trigeorgis, N. Silberman, D. Krishnan, and D. Erhan, “Domain separation networks,” in Adv. Neural Inf. Process. Syst., 2016, p. 343–351.

[27] D. Hazarika, R. Zimmermann, and S. Poria, “Misa: Modality-invariant and-specific representations for multimodal sentiment analysis,” in Proc. ACM Int. Conf. Multimedia, 2020, pp. 1122–1131.

[28] J. Zhao, X. Xie, X. Xu, and S. Sun, “Multi-view learning overview: Recent progress and new challenges,” Inf. Fusion, vol. 38, pp. 43–54, 2017.

[29] A. Zadeh, M. Chen, S. Poria, E. Cambria, and L.-P. Morency, “Tensor fusion network for multimodal sentiment analysis,” in Proc. Conf. Empir. Methods Nat. Lang. Process., 2017, pp. 1103–1114.

[30] Z. Yu, J. Yu, J. Fan, and D. Tao, “Multi-modal factorized bilinear pooling with co-attention learning for visual question answering,” in Proc. IEEE Int. Conf. Comput. Vis., 2017, pp. 1839–1848.

[31] Z. Liu, Y. Shen, V. B. Lakshminarasimhan, P. P. Liang, A. Bagher Zadeh, and L.-P. Morency, “Efficient low-rank multimodal fusion with modalityspecific factors,” in Proc. Annu. Meeting Assoc. Comput. Linguistics, 2018, pp. 2247–2256.

[32] N. Kim, H. Baik, and Y. Yoon, “Topocl: Topological contrastive learning for time series,” IEEE Trans. Neural Netw. Learn. Syst., pp. 1–12, 2026.

[33] M. Hajij, B. Wang, C. Scheidegger, and P. Rosen, “Visual detection of structural changes in time-varying graphs using persistent homology,” in Proc. IEEE Pacific Vis. Symp., 2018, pp. 125–134.

[34] S. Tymochko, E. Munch, and F. A. Khasawneh, “Using zigzag persistent homology to detect hopf bifurcations in dynamical systems,” Algorithms, vol. 13, no. 11, p. 278, 2020.

[35] A. Myers, D. Munoz, F. A. Khasawneh, and E. Munch, “Temporal˜ network analysis using zigzag persistence,” EPJ Data Sci., p. 6, 2023.

[36] Y. Chen, I. Segovia, and Y. R. Gel, “Z-gcnets: Time zigzags at graph convolutional networks for time series forecasting,” in Proc. Int. Conf. Mach. Learn., vol. 139, 2021, pp. 1684–1694.

[37] Y. Chen, Y. Gel, and H. V. Poor, “Time-conditioned dances with simplicial complexes: Zigzag filtration curve based supra-hodge convolution networks for time-series forecasting,” in Adv. Neural Inf. Process. Syst., vol. 35, 2022, pp. 8940–8953.

[38] H. Edelsbrunner and J. Harer, Computational topology: an introduction. American Mathematical Soc., 2010.

[39] A. Van Den Oord, S. Dieleman, H. Zen, K. Simonyan, O. Vinyals, A. Graves, N. Kalchbrenner, A. Senior, K. Kavukcuoglu, et al., “Wavenet: A generative model for raw audio,” arXiv preprint arXiv:1609.03499, vol. 12, no. 1, 2016.

[40] S. Bai, J. Z. Kolter, and V. Koltun, “An empirical evaluation of generic convolutional and recurrent networks for sequence modeling,” arXiv preprint arXiv:1803.01271, 2018.

[41] M. Zaheer, S. Kottur, S. Ravanbakhsh, B. Poczos, R. R. Salakhutdinov, and A. J. Smola, “Deep sets,” in Adv. Neural Inf. Process. Syst., 2017.

[42] C. R. Qi, H. Su, K. Mo, and L. J. Guibas, “Pointnet: Deep learning on point sets for 3d classification and segmentation,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit., 2017.

[43] W. Zellinger, T. Grubinger, E. Lughofer, T. Natschlager, and¨ S. Saminger-Platz, “Central moment discrepancy (cmd) for domaininvariant representation learning,” in Proc. Int. Conf. Learn. Represent., 2017.

[44] NYC OpenData, “NYC OpenData: Open data for all new yorkers,” 2024, accessed: 2026-05-13. [Online]. Available: https://opendata. cityofnewyork.us/

[45] Chicago Data Portal, “Chicago data portal,” 2024, accessed: 2026-05-13. [Online]. Available: https://data.cityofchicago.org/

[46] U.S. Census Bureau, “Measuring america’s people, places, and economy,” 2024, accessed: 2026-05-13. [Online]. Available: https: //www.census.gov/

[47] Chicago Health Atlas, “Median household income (Indicator INC),” https://chicagohealthatlas.org/indicators/INC, 2024, U.S. Census Bureau, American Community Survey (Table B19013).