# UniOMA: Unified Optimal-Transport Multi-Modal Structural Alignment for Robot Perception

Xinrui Zu, Mark Hoogendoorn, Kevin Sebastian Luck, Shujian Yu

Vrije Universiteit Amsterdam, the Netherlands

## Abstract

Achieving generalizable and well-aligned multimodal representations remains a core challenge in robot learning. While recent approaches have attempted to align modalities by modeling conditional or higher-order statistical dependencies, they do not explicitly consider the structural coherence across modalities. In this work, we propose a novel multimodal alignment method that augments existing contrastive losses with a geometry-aware Gromov–Wasserstein (GW) distance-based regularization. To this end, we encode intra-modal geometry for each modality and reduce cross-modal structural discrepancies through a dynamically learned GW barycenter, thereby encouraging structural consistency in addition to instance-level contrastive alignment. We apply this optimal-transport-based alignment strategy to robot perception tasks involving underexplored modalities such as force and tactile signals, where modality data often exhibit varying sample densities. Extensive experiments on five robotic benchmarks show that our method improves structural alignment and downstream performance over state-of-the-art multimodal contrastive baselines across a wide range of downstream tasks, including robot and environment state prediction. Moreover, our GW-based regularization can be seamlessly integrated into most InfoNCE-like objectives.

Key Words— Robot perception, Multimodal alignment, Gromov–Wasserstein distance.

## I Introduction

The integration of information from diverse sources or modalities has received increasing attention across a wide range of AI applications, including image/video/text generation [1, 2], healthcare [3], autonomous systems [4], scientific discovery [5], and robotics, particularly robot learning and perception [6, 7]. Recent advances in contrastive self-supervised learning (CSSL) [8–11], particularly those leveraging InfoNCE losses [12], have shown strong performance in aligning heterogeneous modalities into a shared representation space [13]. Such alignment has enabled zero-shot cross-moda retrieval, transfer, generation, and completion [13–17], while also providing a foundation for learning multimodal representations in robotic systems. In this setting, heterogeneous observations, such as visual, tactile, and proprioceptive modalities, are aligned for tasks like manipulation, imitation learning, reward learning, and perception [6, 18, 19]. By maximizing agreement between paired modalities of the same instance while minimizing similarity between distinct instances, CSSL encourages the learning of invariant and semantically meaningful features.

Despite their empirical success, existing CSSL methods fundamentally cast multimodal alignment as a binary classification problem [20]: paired samples are encouraged to be close, while unpaired samples are pushed apart, without explicitly modeling the continuous intra-modal distance geometry. As a result, multimodal representations may appear statistically aligned at the population level, yet disagree in how samples relate to one another within each modality. We refer to this phenomenon as the structural alignment gap in contrast to the modality gap [21], which characterizes the instance-level separation of modalities in the embedding space.

![](images/bb35ff4eeca7da127a575758570098f94426aca4dc1e614e2f748191a3fe6ae1.jpg)  
(a) 2-modality samples

![](images/0be59afb455fe0edd2750b14ea622e677a993ac1c0a4b228500b1e92f30743b5.jpg)  
(b) InfoNCE view

![](images/ff20e47b40af66022614ee6424da8a390433f1bc1afa4e3fcb9de4db51d60a19.jpg)  
(c) Structure view  
Figure 1: Structural alignment gap. (a) Samples from two modalities (x and y) show strong instance-level correspondence, while the intra-modal structures are diferent; (b) the cross-modal InfoNCE similarity matrices $( S _ { x  y }$ and $S _ { y  x } ,$ whose $( i , j )$ -th element is sim $( x _ { i } , y _ { j } )$ and $\sin ( y _ { i } , x _ { j } )$ respectively) are similar and both lead to a small InfoNCE loss $( \ell _ { \mathrm { C L I P } } ^ { ( x  y ) }$ and $\ell _ { \mathrm { C L I P } } ^ { ( y  x ) }$ as defined in Eq. 1), indicating that existing contrastive alignment losses cannot distinguish modality-specific structural diferences; (c) the intra-modal similarity (kernel) structures difer $\left( K _ { x } \ \mathrm { v s . } \ K _ { y } \right)$ , where $K _ { y }$ exhibits cluster structure absent from $K _ { x }$ . Thus, alignment with InfoNCE loss does not imply structural consistency across modalities.

Crucially, the structural alignment gap is not an implementation artifact but a theoretical limitation of InfoNCE-style objectives. InfoNCE and related contrastive losses can be interpreted as variational lower bounds on mutual information [22, 23]. However, mutual information is invariant under certain invertible reparameterizations of the variables [24,25], and is therefore underdetermined with respect to modality-specific geometry. Consequently, while efective at capturing populationlevel statistical dependence, they do not by themselves constrain the intra-modal geometry to be consistent across modalities.

Fig. 1 illustrates this gap: the cross-modal InfoNCE similarities remain compatible, while the intra-modal kernels $K _ { x }$ and $K _ { y }$ (Sec. III-C) difer. Thus, instance-level correspondence is preserved, yet the intra-modal relational geometry is inconsistent, which InfoNCE-like objectives alone do not penalize.

This structural alignment gap is particularly critical in robotics, where multimodal sensor streams are neither i.i.d. nor structureless: trajectories form subclusters [26], contact events induce discontinuities [27], and proprioceptive signals follow physical constraints [28, 29]. These robotic settings motivate us to test whether explicit relational alignment complements instance-level correspondence. To address the identified structural alignment gap, we introduce a Unified Optimaltransport Multi-modal structural Alignment (UniOMA) framework that supports alignment across three or more modalities. UniOMA augments contrastive learning with a structure-aware regularization based on Gromov–Wasserstein (GW) distances and barycenters [30, 31]. In our formulation, observations from each modality are represented as a metric space through intra-moda similarity matrices. A dynamic GW barycenter is then computed as the structural consensus across modalities, and each modality is softly aligned to this consensus by minimizing weighted GW distances. To our knowledge, UniOMA is the first method to employ a GW barycenter as a structural-consensus regularizer for contrastive representation learning. The modality weights are optimized end-to-end alongside encoder parameters, enabling adaptive contributions of diferent modalities to the structural consensus. By aligning each modality to a shared structural consensus rather than every other modality, this barycentric formulation reduces structural couplings from O(M<sup>2</sup>) to O(M), where M is the number of modalities, and thus is well suited to settings with

three or more modalities.

In summary, our main contributions are:

(C1) We identify and formalize the structural alignment gap, demonstrating why InfoNCE-style objectives do not explicitly enforce intra-modal structural consistency, supported by synthetic analysis.

(C2) We propose UniOMA, a structure-aware multimodal alignment framework based on Gromov– Wasserstein distance and barycenters, which is well suited to settings with $3 +$ modalities.

(C3) We conduct extensive evaluations on five robotic benchmarks (including a new real-world robot-centric dataset with a 6-DoF manipulator), spanning heterogeneous modalities, 3+ modality settings, and complementary downstream tasks, showing that UniOMA improves downstream performance in most evaluated settings and achieves better structural alignment.

## II Background and Related Work

## A Alignment via InfoNCE and Extensions to More than Two Modalities

Unlike multimodal fusion [32, 33], which typically requires all modalities to be present at inference, alignment into a shared embedding space remains functional even if some modalities are missing, enabling zero-shot retrieval, generation, and modality completion [34]. A representative example is CLIP [13], which trains modality-specific encoders $f _ { \theta } ^ { ( 1 ) } , \bar { f } _ { \theta } ^ { ( 2 ) }$ using an InfoNCE-style objective to align paired samples from two modalities. Given a batch of N paired samples $\{ ( \mathbf { x } _ { i } ^ { ( 1 ) } , \mathbf { x } _ { i } ^ { ( 2 ) } ) \} _ { i = 1 } ^ { N }$ where $\mathbf { x } _ { i } ^ { ( m ) }$ denotes the input of the i-th sample from modality $m _ { \colon }$ , each encoder maps its input to an embedding $\mathbf { z } _ { i } ^ { ( m ) } = f _ { \theta } ^ { ( m ) } ( \mathbf { x } _ { i } ^ { ( m ) } )$ ). CLIP then uses an InfoNCE-style objective to identify the correct cross-modal pair among N candidates:

$$
\ell _ { \mathrm { C L I P } } ^ { ( 1  2 ) } ( \boldsymbol { \theta } ) = - \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \log \frac { \exp ( \sin ( \mathbf { z } _ { i } ^ { ( 1 ) } , \mathbf { z } _ { i } ^ { ( 2 ) } ) / \tau ) } { \sum _ { j = 1 } ^ { N } \exp ( \sin ( \mathbf { z } _ { i } ^ { ( 1 ) } , \mathbf { z } _ { j } ^ { ( 2 ) } ) / \tau ) } ,\tag{1}
$$

where sim $( \cdot , \cdot )$ is the similarity between the embeddings $\mathbf { z } _ { i } ^ { ( m ) } = f _ { \theta } ^ { ( m ) } ( \mathbf { x } _ { i } ^ { ( m ) } ) , m = 1 , 2$ , and $\tau$ denotes a temperature parameter. The final CLIP objective symmetrizes Eq. (1) by taking the average via

$$
\mathcal { L } _ { \mathrm { C L I P } } ^ { ( 1 , 2 ) } ( \boldsymbol { \theta } ) = \frac { 1 } { 2 } ( \ell _ { \mathrm { C L I P } } ^ { ( 1 \to 2 ) } ( \boldsymbol { \theta } ) + \ell _ { \mathrm { C L I P } } ^ { ( 2 \to 1 ) } ( \boldsymbol { \theta } ) ) ,\tag{2}
$$

where $\ell _ { \mathrm { C L I P } } ^ { ( 2  1 ) }$ is the reverse direction $2  1$ . In general, this InfoNCE-based objective captures the statistical correlation, providing a lower bound on the mutual information [22, 23] between the anchor modality $\chi ^ { ( 1 ) }$ and modality $\mathcal { X } ^ { ( 2 ) }$

$$
I ( \chi ^ { ( 1 ) } ; \chi ^ { ( 2 ) } ) \geq \log N - 2 \mathcal { L } _ { \mathrm { C L I P } } ^ { ( 1 , 2 ) } ( \theta ) .\tag{3}
$$

Despite their empirical success, InfoNCE-like objectives discretize the continuous similarity structure into a binary positive–negative signal, which can result in embedding spaces that are statistically aligned across modalities while failing to preserve intra-modal geometric relationships.

Real-world systems, particularly in robotics, often involve three or more modalities. Existing approaches typically extend CLIP to three modalities by summing all pairwise contrastive losses [14, 15, 35–45]:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { C M C } } ^ { ( 1 , 2 , 3 ) } ( \theta ) = \mathcal { L } _ { \mathrm { C L I P } } ^ { ( 1 , 2 ) } ( \theta ) + \mathcal { L } _ { \mathrm { C L I P } } ^ { ( 1 , 3 ) } ( \theta ) + \mathcal { L } _ { \mathrm { C L I P } } ^ { ( 2 , 3 ) } ( \theta ) . } \end{array}\tag{4}
$$

![](images/8a8aaaf3c1cf33dd29d4a44f822f14b2cdb3dd4fe9be82b0293fb5167f96459d.jpg)  
Figure 2: UniOMA in two stages. Stage 1 (left): given a mini-batch of heterogeneous observations, UniOMA builds one input-space similarity matrix ${ \bf K } _ { { \bf x } } ^ { ( m ) }$ per modality and estimates a Gromov– Wasserstein (GW) barycenter $\mathbf { C } _ { \mathbf { x } } ^ { \ast } .$ , which serves as a batch-wise structural consensus. Stage 2 (middle): modality-specific encoders produce embeddings $\mathbf { z } ^ { ( m ) }$ and the corresponding embeddingspace kernels ${ \bf K } _ { { \bf z } } ^ { ( m ) }$ ; the encoders are updated by matching each ${ \bf K } _ { { \bf z } } ^ { ( m ) }$ to $\mathbf { C } _ { \mathbf { x } } ^ { \ast }$ through the UniOMA objective. Aligning each modality to a single consensus avoids pairwise $O ( M ^ { 2 } )$ couplings, so the number of alignment terms grows linearly with M. Right: UniOMA applied to the observation encoders of an end-to-end RL policy under modality dropout (Sec. IV-A and Table IV).

Such pairwise extensions neglect higher-order dependencies among modalities. To address this issue, Symile [46] formulates triple-wise contrastive objectives as:

$$
{ \mathcal { L } } _ { \mathrm { S y m i l e } } ^ { ( 1 , 2 , 3 ) } ( \theta ) = { \frac { 1 } { 3 } } [ \ell ^ { ( 1  2 , 3 ) } ( \theta ) + \ell ^ { ( 2  1 , 3 ) } ( \theta ) + \ell ^ { ( 3  1 , 2 ) } ( \theta ) ] .\tag{5}
$$

Here, $\ell ^ { ( 1  2 , 3 ) }$ is the InfoNCE-like loss for one positive triple and $N - 1$ negative triples given by

$$
\ell ^ { ( 1  2 , 3 ) } ( \theta ) = - \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \log \frac { \exp ( \langle \mathbf { z } _ { i } ^ { ( 1 ) } , \mathbf { z } _ { i } ^ { ( 2 ) } , \mathbf { z } _ { i } ^ { ( 3 ) } \rangle / \tau ) } { \sum _ { j = 1 } ^ { N } \exp ( \langle \mathbf { z } _ { i } ^ { ( 1 ) } , \mathbf { z } _ { j } ^ { ( 2 ) } , \mathbf { z } _ { j } ^ { ( 3 ) } \rangle / \tau ) } ,\tag{6}
$$

where each term $\ell ^ { ( 1  2 , 3 ) }$ compares one positive triple against N−1 negatives, $\langle \cdot , \cdot , \cdot \rangle$ is the coordinatewise sum of the element-wise product. More recently, GRAM [47] replaces the cosine similarity with the Gramian volume $\operatorname { V o l } ( \cdot , \ldots , \cdot )$ spanned by embeddings from modalities, providing a higher-order, groupwise compatibility score:

$$
\begin{array} { l } { { \mathcal { E } _ { \mathrm { G R A M } } ^ { ( 1 , \dots , M ) } ( \theta ) = \displaystyle \frac { 1 } { 2 } ( \ell _ { \mathrm { D 2 A } } ^ { ( 1  2 , \dots , M ) } ( \theta ) + \ell _ { \mathrm { A 2 D } } ^ { ( 1  2 , \dots , M ) } ( \theta ) ) , } } \\ { { \ell _ { \mathrm { D 2 A } } ^ { ( 1  2 , \dots , M ) } ( \theta ) } } \\ { { = - \displaystyle \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \log \frac { \exp ( - \mathrm { V o l } ( { \bf z } _ { i } ^ { ( 1 ) } , { \bf z } _ { i } ^ { ( 2 ) } , \dots , { \bf z } _ { i } ^ { ( M ) } ) / \tau ) } { \sum _ { j = 1 } ^ { N } \exp ( - \mathrm { V o l } ( { \bf z } _ { j } ^ { ( 1 ) } , { \bf z } _ { i } ^ { ( 2 ) } , \dots , { \bf z } _ { i } ^ { ( M ) } ) / \tau ) } , } } \\ { { \ell _ { \mathrm { A 2 D } } ^ { ( 1  2 , \dots , M ) } ( \theta ) } } \\ { { = - \displaystyle \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \log \frac { \exp ( - \mathrm { V o l } ( { \bf z } _ { i } ^ { ( 1 ) } , { \bf z } _ { i } ^ { ( 2 ) } ) , \dots , { \bf z } _ { i } ^ { ( M ) } ) / \tau ) } { \sum _ { j = 1 } ^ { N } \exp ( - \mathrm { V o l } ( { \bf z } _ { i } ^ { ( 1 ) } , { \bf z } _ { j } ^ { ( 2 ) } , \dots , { \bf z } _ { j } ^ { ( M ) } ) / \tau ) } , } } \end{array}
$$

where $\ell _ { \mathrm { D 2 A } }$ and $\ell _ { \mathrm { A 2 D } }$ are bidirectional cross-modal contrastive losses with modality 1 as the anchor (A): The D2A loss matches the non-anchor modality group, consisting of modalities 2 through M, to modality 1, whereas the A2D loss matches the modality 1 embedding to the non-anchor modality group. $\operatorname { V o l } ( \cdot , \ldots , \cdot )$ is the volume of the M-dimensional parallelotope formed by the embedding vectors $\mathbf { z } ^ { ( m ) }$ , with smaller volumes corresponding to higher groupwise similarity.

These methods make progress on multimodal alignment but still remain limited to instance-level dependencies, overlooking intra-modal structure. More advanced objectives (e.g., CoMM [48] and TRIANGLE [49]) strengthen cross-modal matching, yet still do not explicitly align intra-modal relational geometry. [50] introduces a standard optimal transport (OT) regularizer for cross-modal consistency; however, the matching is performed at the distributional level of the embeddings z, thereby ignoring intra-modal structure in the observation space. In contrast, we align structures by encoding each modality by its intra-modal similarity (kernel) and enforcing a GW-based consensus in the input space X , preserving centroid geometry rather than embedding-level OT matching.

## B Multimodal Representation Learning in Robotics

Robotics is inherently multimodal: vision, force–torque, tactile sensing, and proprioception provide complementary views of the robot–environment system. While multimodal representation learning has been extensively studied in vision–language settings, its exploration in robotics remains limited. Existing work, including the recent Vision–Language–Action (VLA) models, has primarily focused on modality fusion or transfer [51–56].

By comparison, alignment of robotic perception modalities into a shared representation space remains underexplored. Recent eforts [57, 58] demonstrate cross-modal retrieval and perception, while [26,59] show that cross-modal or cross-temporal alignment can yield transferable representations. These developments highlight that robotic perception data is inherently structured, which is shaped by trajectories, contact events, and physical constraints, motivating alignment methods that preserve intra-modal geometry across modalities rather than relying solely on feature fusion. Importantly, our work addresses a complementary setting to recent VLA models. Rather than focusing on language-conditioned policy learning, we study multimodal alignment for robotic perception using physically grounded sensor modalities, such as force, tactile sensing, and proprioception, with the goal of learning structure-preserving multimodal representations.

## III UniOMA: Unified Optimal-Transport Multi-Modal Structural Alignment

Our proposed UniOMA aligns three or more heterogeneous modalities by combining instance-level statistical correspondence with structure-level geometric consistency. As illustrated in Fig. 2, UniOMA operates in two stages. Given a mini-batch of multimodal observations, it first constructs modality-specific input-space kernels and estimates a GW barycenter as the batch-wise structural consensus. It then updates the modality-specific encoders by aligning their embedding-space kernels to this consensus while optimizing a contrastive objective.

## A Problem Statement

Consider a robot interacting with its environment over time. At each time step t, the robot receives a multimodal observation $\mathbf { o } _ { t } = ( \mathbf { x } _ { t } ^ { ( 1 ) } , \mathbf { x } _ { t } ^ { ( 2 ) } , \ldots , \mathbf { x } _ { t } ^ { ( M ) } )$ , where $\mathbf { x } _ { t } ^ { ( m ) } \in \mathcal { X } ^ { ( m ) }$ is the observation from modality $m ,$ such as vision, proprioception, tactile sensing, force–torque signals, or environment states. In some cases, the action $\mathbf { a } _ { t }$ may also be included as one modality. The goal of multimodal alignment is to learn modality-specific encoders $f ^ { ( m ) } : \mathcal { X } ^ { ( m ) }  \mathbb { R } ^ { d } , m = 1 , \dots , M$ , that map each input $\mathbf { x } ^ { ( m ) }$ to a shared latent representation $\mathbf { z } ^ { ( m ) } = f ^ { ( m ) } ( \mathbf { x } ^ { ( m ) } ) \in \mathbb { R } ^ { d }$ . For paired observations originating from the same underlying instance or physical event, the learned embeddings are expected to be semantically compatible and highly similar across modalities. At the same time, efective alignment should preserve the relational structure within each modality, so that the geometry among samples in one modality is consistently reflected in the shared representation space.

Fig. 2 shows examples of such modalities and how UniOMA aligns them.

## B Gromov–Wasserstein Distance

The Gromov–Wasserstein (GW) distance [30, 31] extends optimal transport (OT) [60], which seeks the minimum cost of transporting one probability measure into another within the same metric space, to measures supported on diferent metric spaces. Since a cross-modal cost $c ( \mathbf { x } , \mathbf { y } )$ is generally dificult to define, GW replaces it with a relational cost that only requires the two intra-modal similarity matrices.

Definition 1 (Gromov–Wasserstein Distance). Let $\mathcal { X } _ { d _ { \mathbf { x } } , \mu }$ and $\mathcal { D } _ { d _ { \mathbf { y } } , \nu }$ be two metric–measure spaces, with distance metrics $d _ { \mathbf { x } } , ~ d _ { \mathbf { y } }$ and probability measures $\mu , \nu .$ . The GW distance between them is defined as:

$$
\begin{array} { r l } & { d _ { g w } ( \mu , \nu ) : = } \\ & { \underset { \pi \in \Pi ( \mu , \nu ) } { \operatorname* { i n f } } \int _ { \mathcal { X } ^ { 2 } \times \mathcal { Y } ^ { 2 } } c ( d _ { \mathbf x } ( \mathbf x , \mathbf x ^ { \prime } ) , d _ { \mathbf y } ( \mathbf y , \mathbf y ^ { \prime } ) ) \ d \pi ( \mathbf x , \mathbf y ) d \pi ( \mathbf x ^ { \prime } , \mathbf y ^ { \prime } ) } \end{array}\tag{7}
$$

where $c \left( d _ { \mathbf { x } } ( \mathbf { x } , \mathbf { x } ^ { \prime } ) , d _ { \mathbf { y } } ( \mathbf { y } , \mathbf { y } ^ { \prime } ) \right)$ is a relational distance measuring the discrepancy between the sample pairs $( { \bf x } , { \bf x } ^ { \prime } )$ and $( \mathbf { y } , \mathbf { y } ^ { \prime } )$

Minimizing the GW distance thus matches the relational geometry of two modalities rather than their raw coordinates.

For discrete samples, the continuous GW objective can be estimated from finite-sample intramodal similarity matrices. We use the squared relational loss on symmetric similarity kernels and identify it as GW distance.

Theorem 1 (Kernelized empirical GW distance [31]). Let $\mathcal { X } = \{ x _ { i } \} _ { i = 1 } ^ { I }$ and $\mathcal { Y } = \{ y _ { j } \} _ { j = 1 } ^ { J }$ be two sets of samples with empirical measures $\hat { \mathbf { p } } _ { \mathbf { x } } = \frac { 1 } { I } \mathbf { 1 } _ { I }$ and $\begin{array} { r } { \hat { \bf p } _ { \bf y } = \frac { 1 } { J } { \bf 1 } _ { J } } \end{array}$ . Let $\mathbf { K _ { x } } \in \mathbb { R } ^ { I \times I }$ and $\dot { \mathbf { K } } _ { \mathbf { y } } \in \mathbb { R } ^ { J \times J }$ denote their intra-modal similarity matrices. The empirical GW distance between the two sample sets is defined as:

$$
\begin{array} { r l } & { \hat { d } _ { g w } ( \mathbf { K } _ { \mathbf { x } } , \mathbf { K } _ { \mathbf { y } } ) : = \underset { \mathbf { T } \in \Pi ( \hat { \mathbf { p } } _ { \mathbf { x } } , \hat { \mathbf { p } } _ { \mathbf { y } } ) } { \operatorname* { m i n } } } \\ & { \qquad \quad \quad \displaystyle \sum _ { i , i ^ { \prime } , j , j ^ { \prime } } \big ( ( \mathbf { K } _ { \mathbf { x } } ) _ { i i ^ { \prime } } - ( \mathbf { K } _ { \mathbf { y } } ) _ { j j ^ { \prime } } \big ) ^ { 2 } T _ { j i } T _ { j ^ { \prime } i ^ { \prime } } . } \end{array}
$$

where T is the doubly-stochastic matrix that models the transport plan coupling the two empirical measures.

In practice, we estimate the optimal transport plan $\mathbf { T } ^ { * }$ via iterative OT solvers (see Alg. 2 in Appendix A for details) and compute the final GW distance between $\{ { \bf x } _ { i } \} _ { i = 1 } ^ { I }$ and $\{ \mathbf { y } _ { j } \} _ { j = 1 } ^ { J }$ as:

$$
\begin{array} { r l } & { \hat { d } _ { g w } ( \mathbf { K } _ { \mathbf { x } } , \mathbf { K } _ { \mathbf { y } } ) = \hat { \mathbf { p } } _ { \mathbf { x } } ^ { \top } ( \mathbf { K } _ { \mathbf { x } } \odot \mathbf { K } _ { \mathbf { x } } ) \hat { \mathbf { p } } _ { \mathbf { x } } } \\ & { \qquad + \hat { \mathbf { p } } _ { \mathbf { y } } ^ { \top } ( \mathbf { K } _ { \mathbf { y } } \odot \mathbf { K } _ { \mathbf { y } } ) \hat { \mathbf { p } } _ { \mathbf { y } } } \\ & { \qquad - 2 \mathrm { t r } ( \mathbf { K } _ { \mathbf { x } } ^ { \top } \mathbf { T } ^ { * \top } \mathbf { K } _ { \mathbf { y } } \mathbf { T } ^ { * } ) , } \end{array}\tag{8}
$$

where ⊙ denotes element-wise multiplication. The first two terms are independent of T only when the kernels and marginals are fixed. Thus minimizing the GW distance over T is equivalent to maximizing the trace term in Eq. 8. During encoder optimization, we keep the embedding-dependent quadratic term. Eq. (8) enables cross-modal alignment directly from intra-modal similarity structures, without the need for an explicit cross-modal cost function or extra neural potential models [61–63].

## C Structural Consensus

To preserve intra-modal structure during alignment, we represent each modality $\chi ^ { ( m ) }$ by an intramodal similarity matrix $\mathbf { K } _ { \mathbf { x } } ^ { ( m ) } \in \mathbb { R } ^ { N _ { m } \times N _ { m } }$ , where $( \mathbf { K } _ { \mathbf { x } } ^ { ( m ) } ) _ { i j } = \sin ( \mathbf { x } _ { i } ^ { ( m ) } , \mathbf { x } _ { j } ^ { ( m ) } )$ encodes the pairwise similarity between samples and $N _ { m }$ is the number of observations from modality $m ,$ which can difer across modalities. Such kernel matrices provide a unified representation of relational structure across heterogeneous modalities, abstracting away diferences in raw input dimensionality or temporal resolution.

We construct ${ \bf K } _ { { \bf x } } ^ { ( m ) }$ using modality-specific similarity functions. For high-dimensional modalities such as image tensors, we first extract feature representations with a pretrained encoder and then compute pairwise similarities in the resulting feature space, alleviating the sparsity and dimensionality issues associated with raw inputs. For sequential or time-series modalities such as force–torque signals, we use the time-series clustering kernel (TCK; [64]) to more efectively capture temporal dynamics. Additional implementation details are provided in Supplementary Material, Sec. I.

The goal is to identify a shared structural consensus that captures the common relational patterns across modalities. We assume that the modalities share these relational patterns only partially, for example through common trajectories and contact events, rather than having identical geometry; the consensus represents the shared part, and the contrastive term retains modality-specific information. We define this consensus as a Gromov–Wasserstein barycenter [31] of the intra-modal similarity matrices.

Definition 2 (Structural Consensus of Multimodal Data). Given intra-modal kernel matrices $\{ { \bf K } _ { \bf x } ^ { ( m ) } \} _ { m = 1 } ^ { M }$ , the structural consensus is defined as the barycenter:

$$
\mathbf { C } _ { \mathbf { x } } ^ { * } = \arg \operatorname* { m i n } _ { \mathbf { C } _ { \mathbf { x } } \in \mathcal { M } } \sum _ { m = 1 } ^ { M } \lambda _ { m } \cdot \hat { d } _ { g w } ( \mathbf { C _ { x } } , \mathbf { K } _ { \mathbf { x } } ^ { ( m ) } ) ,\tag{9}
$$

where M denotes the space of symmetric positive definite (SPD) matrices, $\hat { d } _ { g w }$ is the empirical GW distance (Eq. 8), and $\{ \lambda _ { m } \}$ are learnable modality weights satisfying $\lambda _ { m } \geq 0$ and $\begin{array} { r } { \sum _ { m } \lambda _ { m } = 1 } \end{array}$

In practice, $\mathbf { C } _ { \mathbf { x } } ^ { \ast }$ is estimated by an iterative barycenter optimization scheme (Alg. 3 in $\mathrm { A p \mathrm { - } }$ pendix A). The central idea of UniOMA is to use this batch-wise consensus as a structural anchor: during training, each modality is aligned by minimizing the GW distance between its embeddinginduced kernel $\mathbf { K } _ { \mathbf { z } } ^ { ( m ) }$ , which is the embedding-space similarity matrix of $\mathbf { z } ^ { ( m ) } = f _ { \theta } ^ { ( m ) } ( \mathbf { x } ^ { ( m ) } )$ , and the consensus $\mathbf { C } _ { \mathbf { x } } ^ { \ast }$ . The weights determine each modality’s contribution to the consensus and serve as an empirical indicator of modality-wise agreement with the learned structural consensus, rather than a direct measure of modality importance. Detailed implementations are in the next section.

## D UniOMA Objective and Algorithm

Given the batch-wise structural consensus $\mathbf { C } _ { \mathbf { x } } ^ { \ast }$ in Sec. III-C, UniOMA augments a standard contrastive term ${ \mathcal L } _ { \mathrm c }$ with a structure-aware regularizer

$$
\mathcal { L } _ { \mathrm { U n i O M A } } ( \theta ) = \mathcal { L } _ { \mathrm { c } } ( \theta ) + \alpha \sum _ { m = 1 } ^ { M } \lambda _ { m } \cdot \hat { d } _ { g w } ( \mathbf { C } _ { \mathbf { x } } ^ { * } , \mathbf { K } _ { \mathbf { z } } ^ { ( m ) } ) ,\tag{10}
$$

where ${ \bf K } _ { { \bf z } } ^ { ( m ) }$ is the embedding-space similarity matrix of $\mathbf { z } ^ { ( m ) } = f _ { \boldsymbol { \theta } } ^ { ( m ) } ( \mathbf { x } ^ { ( m ) } )$ . With the plan $\mathbf { T } ^ { ( m ) * }$ of Alg. 2 held fixed and $\begin{array} { r } { \hat { \bf p } = \frac { 1 } { N } { \bf 1 } _ { N } } \end{array}$ , the structural term of modality m is

$$
\begin{array} { r l } & { \hat { d } _ { g w } ( \mathbf { C } _ { \mathbf { x } } ^ { * } , \mathbf { K } _ { \mathbf { z } } ^ { ( m ) } ( \theta ) ) = \hat { \mathbf { p } } ^ { \top } ( \mathbf { C } _ { \mathbf { x } } ^ { * } \odot \mathbf { C } _ { \mathbf { x } } ^ { * } ) \hat { \mathbf { p } } } \\ & { \qquad + \hat { \mathbf { p } } ^ { \top } ( \mathbf { K } _ { \mathbf { z } } ^ { ( m ) } ( \theta ) \odot \mathbf { K } _ { \mathbf { z } } ^ { ( m ) } ( \theta ) ) \hat { \mathbf { p } } } \\ & { \qquad - 2 \mathrm { t r } \big ( \mathbf { C } _ { \mathbf { x } } ^ { * \top } \mathbf { T } ^ { ( m ) * \top } \mathbf { K } _ { \mathbf { z } } ^ { ( m ) } ( \theta ) \mathbf { T } ^ { ( m ) * } \big ) , } \end{array}\tag{11}
$$

where the first term is constant and the gradients change θ through ${ \bf K } _ { { \bf z } } ^ { ( m ) }$ in the second and third terms. We add an entropy term $\begin{array} { r } { \beta \sum _ { m = 1 } ^ { M } \lambda _ { m } \log ( \lambda _ { m } + \varepsilon ) } \end{array}$ with $\beta = 1$ to discourage the concentration of the weights on a single modality. The scalar α balances contrastive discrimination and structural coherence, and the learnable weights $\{ \lambda _ { m } \}$ determine each modality’s contribution to the consensus. Implementation details for estimating $\mathbf { C } _ { \mathbf { x } } ^ { \ast }$ and evaluating $\hat { d } _ { g w } ( \cdot , \cdot )$ are in Appendix A (see also Fig. 2).

Algorithm 1: UniOMA Training( $\{ \mathcal { X } ^ { ( m ) } \} _ { m = 1 } ^ { M } , \eta , \alpha \}$   
Input: Multimodal dataset $\{ \mathcal { X } ^ { ( m ) } \} _ { m = 1 } ^ { M } ,$ learning rate η, structural weight α   
Initialize $\{ f ^ { ( m ) } ( \cdot ) \} _ { m = 1 } ^ { M }$ , modality weights $\lbrace \lambda _ { m } \rbrace _ { m = 1 } ^ { M } ;$   
while not converged do   
// Stage 1: consensus estimation   
Sample batch $\{ \mathbf { x } _ { i } ^ { ( m ) } \} _ { i = 1 } ^ { N _ { m } }$ for modalities $\{ \mathcal { X } ^ { ( m ) } \} _ { m = 1 } ^ { M } ;$   
for m ← 1 to M do   
Compute ${ \bf K } _ { \bf x } ^ { ( m ) }$ for the batch $\{ \mathbf { x } _ { i } ^ { ( m ) } \} _ { i = 1 } ^ { N _ { m } } ;$   
Estimate the structural consensus $\mathbf { C } _ { \mathbf { x } } ^ { \ast }$ via Alg. 3;   
// Stage 2: alignment update   
$\mathbf { z } _ { i } ^ { ( m ) } \gets f _ { \theta } ^ { ( m ) } ( \mathbf { x } _ { i } ^ { ( m ) } )$ for all i, m;   
for m ← 1 to M do   
$\mathbf { T } ^ { ( m ) ^ { * } } \gets \mathsf { 0 T P 1 a n } ( \mathbf { C _ { x } ^ { * } } , \mathbf { K } _ { \mathbf { z } } ^ { ( m ) } )$ via Alg. 2;   
Compute $\hat { d } _ { g w } ( \mathbf { C } _ { \mathbf { x } } ^ { * } , \mathbf { K } _ { \mathbf { z } } ^ { ( m ) } )$ via $\operatorname { E q . }$ (11);   
Compute the contrastive learning loss ${ \mathcal { L } } _ { \mathrm { c } } ;$   
Compute L<sub>UniOMA</sub> via Eq. (10) and add the entropy term;   
$\theta \gets \theta - \eta \nabla _ { \theta } \mathcal { L } _ { \mathrm { I } }$ <sub>UniOMA</sub>;   
$\lambda _ { m } \gets \lambda _ { m } - \eta \nabla _ { \lambda _ { m } }$ L<sub>UniOMA</sub> for $m = 1 , \ldots , M ;$   
return $\{ f _ { \theta } ^ { ( m ) } \} _ { m = 1 } ^ { M } , \{ \lambda _ { m } \} _ { m = 1 } ^ { M }$

Why this design? (1) scalable to $M \geq 3$ . Aligning every modality to one consensus avoids $O ( M ^ { 2 } )$ pairwise couplings. (2) flexible to heterogeneous and asynchronous modalities. GW distance [30] supports diferent dimensionalities and unequal sample counts across modalities. This is particularly advantageous in robot perception, and we empirically validate this property in Sec. IV-G. (3) Interpretable. The modality weights $\{ \lambda _ { m } \}$ serve as an empirical indicator of modality-wise agreement with the learned structural consensus (Sec. IV-F).

The overall training procedure is in Alg. 1. At each iteration, UniOMA first estimates a batchwise structural consensus $\mathbf { C } _ { \mathbf { x } } ^ { \ast }$ from modality-specific input-space kernels, and then updates the encoders by aligning the embedding-induced kernels $\{ \mathbf { K } _ { \mathbf { z } } ^ { ( m ) } \} _ { m = 1 } ^ { M }$ to this consensus while optimizing the contrastive objective. Both the encoder parameters θ and the modality weights $\lbrace \lambda _ { m } \rbrace _ { m = 1 } ^ { M }$ are updated by stochastic gradient descent.

A shared consensus could suppress information from a more informative modality; probes on the frozen encoders show that UniOMA does not reduce the task-relevant information of the information-rich modality and can increase that of the joint representation (Supplementary Material, Sec. III-G).

## IV Experiments

![](images/2a0c860eadda1a1e34e570ed87e298982949ce0ed5cb4163d7e3a7caa255e527.jpg)  
Figure 3: Downstream tasks. (A) Regression (VFD): a head on the concatenated frozen embeddings predicts the next-step end-efector orientation. (B) Classification (VFP): a classifier on the frozen embeddings predicts binary contact. (C) Cross-modal retrieval (VAT): a query from one modality ranks items from another by embedding similarity (MAP for Vis→Aud, Vis→Tact, and Tact→Aud). (D) End-to-end RL: the encoders are trained within the policy while UniOMA aligns their embeddings, and modalities are dropped to test robustness.

We evaluate UniOMA on five robot-perception benchmarks with diferent sensor combinations: VFD (Vision–Force–Depth) and VFP (Vision–Force–Proprioception) from the Vision&Touch dataset [6, 65], MuJoCo Push [28, 66], VAT (Vision–Audio–Tactile) from ObjectFolder 2.0 [57, 67], and our real-world VIP (Vision–IMU–Proprioception) dataset. To assess scalability beyond three modalities, we further include a 4–7 modality classification task on the Vision&Touch dataset (Sec. IV-E). All methods share the same backbones and training schedule (Supplementary Material, Sec. I).

## A Tasks and Datasets

As illustrated in Fig. 3, the evaluation covers four downstream tasks: regression, classification, cross-modal retrieval, and end-to-end reinforcement learning (RL) under modality dropout.

Table I: Comparative results on downstream tasks (regression, classification, and cross-modal retrieval). Performance is measured by MSE (↓), Acc. (% ↑), and MAP (↑) (mean ± std over 10 seeds; VFD and VFP use a trajectory-level 80/10/10 training/validation/test split). Arrows denote retrieval direction. Gray rows correspond to our method (UniOMA). Adding the GW term improves its contrastive baseline in most paired comparisons, and all methods achieving the best performance for each task are UniOMA variants (highlighted in brown).
<table><tr><td rowspan="2">Method</td><td colspan="2">Regression (MSE)↓</td><td colspan="2">Classification(%)↑</td><td colspan="3">VAT retrieval (MAP) ↑</td></tr><tr><td>V&amp;F&amp;D (×10−3) MuJoCo</td><td></td><td>V&amp;F&amp;D</td><td>V&amp;F&amp;P</td><td></td><td>Vis→Aud Vis→Tact Tact→Aud</td><td></td></tr><tr><td colspan="8">Methods with and without our GW regularizer</td></tr><tr><td>Pairwise [35]</td><td>1.34±0.17</td><td>0.44±0.07</td><td>89.01±0.04</td><td>94.29±0.02</td><td>0.25±0.07</td><td>0.41±0.11</td><td>0.10±0.01</td></tr><tr><td>Pairwise+GW(ours)</td><td>1.26±0.11</td><td>0.38±0.09</td><td>91.50±0.02</td><td>94.71±0.02</td><td>0.36±0.05</td><td>0.60±0.03</td><td>0.12±0.02</td></tr><tr><td>Symile [46]</td><td>2.89±0.08</td><td>0.28±0.04</td><td>89.28±0.05</td><td>93.65±0.05</td><td>0.10±0.02</td><td>0.21±0.05</td><td>0.08±0.01</td></tr><tr><td>Symile+GW(ours)</td><td>2.30±0.07</td><td>0.23±0.02</td><td>91.93±0.02</td><td>93.72±0.03</td><td>0.13±0.03</td><td>0.15±0.03</td><td>0.14±0.03</td></tr><tr><td>GRAM [47]</td><td>3.43±0.12</td><td>0.52±0.07</td><td>92.09±0.03</td><td>93.48±0.04</td><td>0.13±0.02</td><td>0.34±0.05</td><td>0.15±0.01</td></tr><tr><td>GRAM+GW(ours)</td><td>2.33±0.05</td><td>0.30±0.06</td><td>92.91±0.03</td><td>93.80±0.03</td><td>0.79±0.10</td><td>0.58±0.04</td><td>0.16±0.01</td></tr><tr><td colspan="8">Independent methods (different architectures or objectives)</td></tr><tr><td>CoMM [48]</td><td>1.75±0.05</td><td>0.26±0.04</td><td>92.06±0.01</td><td>94.09±0.02</td><td>0.21±0.10</td><td>0.40±0.15</td><td>0.08±0.03</td></tr><tr><td>OTLC [50]</td><td>1.44±0.12</td><td>0.40±0.07</td><td>92.26±0.02</td><td>94.29±0.03</td><td>0.37±0.05</td><td>0.58±0.04</td><td>0.09±0.01</td></tr><tr><td>TRIANGLE [49]</td><td>3.77±0.11</td><td>0.41±0.06</td><td>92.89±0.03</td><td>93.76±0.04</td><td>0.17±0.07</td><td>0.39±0.10</td><td>0.13±0.03</td></tr></table>

VFD (Vision–Force–Depth). We evaluate two tasks on VFD: (1) next-step end-efector orientation prediction (regression, 4D), and (2) trajectory-pair discrimination (classification, binary). For both tasks, inputs are third-person RGB ([b×3×128×128]), force–torque histories ([b×32×6]), and depth ([b×1×128×128]). For trajectory-pair discrimination, given a pair of triplets (vision–force– depth), the model identifies whether the pair is from the same trajectory.

VFP (Vision–Force–Proprioception). We evaluate next-step contact prediction (classification, binary). Inputs are RGB ([b×3×128×128]), force–torque histories ([b×32×6]), and end-efector pose ([b×7]). The task is to classify whether the end-efector is in contact with the object.

MuJoCo Push. We evaluate next-step object-position prediction in a planar pushing task where a Franka Emika Panda arm pushes a puck. Inputs are low-resolution grayscale images ([b×1×32×32]), current force–torque with the contact flag ([b×7]), and end-efector position ([b×3]). The task is to predict the object’s next-step 2D position on the table.

VAT (Vision–Audio–Tactile). We evaluate cross-modal retrieval using mean average precision (MAP). The dataset provides per-object visual, sound, and tactile observations. Queries and retrievals are built across {Vis, Aud, Tact}; we report direction-specific MAP, e.g., Vis→Tact.

VIP (Vision–IMU–Proprioception). To evaluate whether UniOMA remains efective on realworld robotic data beyond simulation and benchmark datasets, we introduce a new robot-centric dataset collected on a 6-DoF robotic manipulator (Fig. 4). In each trajectory, the robot performs a reaching motion toward a sampled end-efector target. The task is to predict the current end-efector position (regression, 3D). Each sample consists of three synchronized modalities: (1) third-person RGB ([b×3×64×64]), (2) proprioception given by the current joint angles ([b×6]), and (3) normalized inertial measurement unit (IMU) signals from the joints ([b×6×6]). We report the mean Euclidean distance between the predicted and measured end-efector positions for this task in Table III. Data collection, dataset scale, sensor setup, ground truth, and the data split are described in Supplementary Material, Sec. II, together with illustrative examples of the modalities.

End-to-end RL. We further evaluate UniOMA on two closed-loop RL tasks in Isaac Lab [68]: locomotion and contact-rich peg insertion, using Proximal Policy Optimization (PPO) [69] for all methods. We assess policy robustness under modality dropout and manipulation performance under full observations. Detailed settings are provided in Supplementary Material, Sec. I. As illustrated in Fig. 2 and Fig. 3(D), each task uses three observation modalities provided to the actor: torso IMU states ([b×6]), joint positions ([b×12]), and joint velocities ([b×12]) for locomotion; and end-efector pose ([b×7]), velocity ([b×6]), and wrist force ([b×3]) for peg insertion. For locomotion, we introduce joint-velocity dropout after 500 training iterations and continue training until iteration 1,000, with the resulting learning curves shown in Fig. 2. We evaluate single-modality dropout at test time using policies trained for 500 iterations. Table IV shows the average rewards of the locomotion task over 10 episodes and the average success rate of the peg insertion task over 512 episodes.

![](images/c359cd7a276c41cb53d91a8f2a32a983a11b5e963aae10374fabc216f44c2682.jpg)  
Figure 4: VIP setup. The VIP dataset is collected on a real 6-DoF robotic manipulator (environment, left). At test time, third-person vision, per-joint IMU, and proprioception (joint angles) are embedded and concatenated to a single trainable head that predicts the current end-efector position (ee-pos, 3D). This applies the regression protocol of Fig. 3(A), and the corresponding results are in Table III.

## B Implementation Details

Encoders, optimizer, temperature, and schedules are shared across methods (fusion heads difer in CoMM). VFD and VFP are split at the trajectory level into training, validation, and test trajectories at a ratio of $8 0 / 1 0 / 1 0 ;$ validation data are used for early stopping and checkpoint selection, and test data only for the final evaluation. We compute input-space kernels $\{ \mathbf { K } _ { \mathbf { x } } ^ { ( m ) } \}$ (RBF for images with the median-heuristic bandwidth; TCK for time-series/force; RBF for other signals) and estimate the batch-wise consensus $\mathbf { C } _ { \mathbf { x } } ^ { \ast }$ using iterative barycenter updates (Appendix A). Hyperparameters, TCK settings, convergence diagnostics, and end-to-end RL PPO settings are detailed in Supplementary Material, Sec. I and Sec. III-B.

Table II: Scalability analysis with 4–7 modalities. We report trajectory-pair classification accuracy (mean ± std over 10 seeds) together with wall-clock time per epoch, its ratio to Pairwise (in parentheses), and peak GPU memory for all three methods: Pairwise contrastive learning, OTLC, and Pairwise+GW (UniOMA, highlighted in gray ). Peak GPU memory was measured on an NVIDIA H100 (batch size 32) and is the same for all three methods at each M. UniOMA achieves consistently higher accuracy and becomes faster than OTLC when $M \geq 6$
<table><tr><td rowspan="2">Modality Combination</td><td rowspan="2">M</td><td rowspan="2">Mem. (GiB)</td><td colspan="2">Pairwise</td><td colspan="2">OTLC</td><td colspan="2">Pairwise+GW(ours)</td></tr><tr><td> $\operatorname { A c c } .$ </td><td>Time</td><td> $\operatorname { A c c . }$ </td><td>Time</td><td> $\operatorname { A c c } .$ </td><td>Time</td></tr><tr><td> $\mathrm { V + F { + } P { + } D }$ </td><td>4</td><td>1.20</td><td>89.94±0.03</td><td> $1 1 0 . 3 8 { \pm } 1 . 7 4 \mathrm { s }$ </td><td>92.07±0.03</td><td> $1 3 5 . 5 7 { \scriptstyle \pm 2 . 9 2 8 \ ( 1 . 2 3 \times ) }$ </td><td>92.39±0.02</td><td>201.36±7.61s (1.82×)</td></tr><tr><td> $\mathrm { V + F { + } P { + } D { + } A }$ </td><td>5</td><td>1.22</td><td> $9 0 . 7 2 { \scriptstyle \pm 0 . 0 3 }$ </td><td> $1 2 9 . 4 4 { \pm } 1 . 9 2 \mathrm { s }$ </td><td>92.51±0.03</td><td>178.63±3.11s (1.38×)</td><td> $\mathbf { 9 3 . 0 4 } { \scriptstyle \pm 0 . 0 2 }$ </td><td>225.89±5.44s (1.75×)</td></tr><tr><td> $\mathrm { V + F + P + D + A + C }$ </td><td>6</td><td>1.24</td><td>89.12±0.04</td><td> $1 5 0 . 7 7 { \scriptstyle \pm 2 . 5 1 8 }$ </td><td> $9 1 . 0 3 { \pm } 0 . 0 3 $ </td><td>268.41±6.83s (1.78×)</td><td>92.11±0.03</td><td>248.52±6.12s (1.65×)</td></tr><tr><td> $\mathrm { V + F + P + D + A + C + O }$ </td><td>7</td><td>1.76</td><td>87.95±0.05</td><td> $1 7 1 . 4 2 { \scriptstyle \pm 3 . 1 2 8 }$ </td><td>89.84±0.04</td><td>382.77±10.44s (2.23×)</td><td>91.02±0.03</td><td>273.36±7.40s (1.59×)</td></tr></table>

Table III: End-efector position prediction (regression) on Vision + IMU + Proprioception (VIP). We report the mean Euclidean position error $( \times 1 0 ^ { - 2 } \mathrm { m } \downarrow )$ ; mean ± std over 10 seeds). The gray row denotes our method (UniOMA). For reference, the final two rows report a proprioception-only MLP and calibrated forward kinematics, which uses the robot’s kinematic model and therefore indicates the accuracy attainable with privileged model knowledge; both are evaluated against motion-capture ground truth.
<table><tr><td>Method</td><td>Modality</td><td>Position error  $\downarrow ( \times 1 0 ^ { - 2 } \mathrm { m } )$  End-effector Pos.</td></tr><tr><td>Pairwise [35]</td><td>V+I+P</td><td>6.32±0.15</td></tr><tr><td>Symile [46]</td><td>V+I+P</td><td>4.59±0.12</td></tr><tr><td>GRAM [47]</td><td>V+I+P</td><td>6.10±0.18</td></tr><tr><td>TRIANGLE [49]</td><td>V+I+P</td><td>5.87±0.09</td></tr><tr><td>Pairwise+GW(ours)</td><td>V+I+P</td><td>4.02±0.10</td></tr><tr><td>Proprioception-only MLP</td><td>P</td><td>5.24±0.15</td></tr><tr><td>Calibrated forward kinematics</td><td>P</td><td>4.18±0.14</td></tr></table>

## C Results: Comparisons on Downstream Tasks

We compare UniOMA against state-of-the-art multimodal alignment methods. First, we consider a family of flexible $I n f o N C E - s t y l e$ frameworks: (i) Pairwise (CMC) [35], which sums pairwise contrastive losses; (ii) Symile [46], extending InfoNCE to triple-wise contrastive objectives; and (iii) GRAM [47], which replaces cosine similarity with Gramian volume similarity for $M \geq 3$ modalities. These methods share a contrastive backbone, allowing our GW regularizer to be incorporated in a plug-and-play manner; we therefore also report “ +GW” variants for (i), (ii), and (iii).

In addition, we compare against three recent, more specialized multimodal objectives: (iv) OTLC [50], which augments pairwise InfoNCE with a standard OT regularizer; (v) TRIAN-GLE [49], which aligns three modalities via triangle area similarity; and (vi) CoMM [48], a strong fusion-based baseline with a dedicated multimodal architecture. All methods are trained with matched optimizer, batch size, temperature, and learning budget (see Supplementary Material, Sec. I).

In Table I, adding the GW term improves its contrastive baseline in most paired comparisons, and the best method for each task is a GW-regularized variant; on VIP (Table III), Pairwise+GW has the lowest error. The gains are large for VFD regression with Symile and GRAM and for Vis→Aud retrieval with GRAM, and small for VFP classification, where all methods reach 93.5–94.7%.

Table IV further evaluates policy robustness in the end-to-end RL setting under observation modality dropout. We compare the unaligned PPO baseline with Pairwise and Pairwise+GW (UniOMA), where the baseline uses the same training pipeline without aligning the modality encoders. The training-time dropout curves in Fig. 2 show that when the joint-velocity modality is removed after 500 iterations, alignment-based policies retain higher reward than the unaligned baseline. The test-time dropout in Table IV shows that UniOMA achieves the highest mean reward under al dropout cases. These results suggest that UniOMA alignment improves RL policy robustness when observations are missing. On peg insertion, Pairwise+GW reaches a success rate of 96.6% under full observations, compared with 92.2% for Pairwise and 90.8% for Baseline PPO; the learning curves are in Supplementary Material, Sec. III-F.

![](images/6c1ceadc3cd3782b06617fd7f730e6dbf5ff8716c91ea121d17253ccb78334db.jpg)  
Figure 5: (a–d) Final learned modality weights $\{ \lambda _ { m } \}$ for each task (VFD, VFP, MuJoCo Push, VAT). Each radar chart shows per-modality weights that sum to 1, showing which modalities agree most with the learned structural consensus in each dataset. (e) Ablation on VFD. One modality is downsampled by $\times \textstyle { \frac { 1 } { 2 } }$ per batch. The plot shows UniOMA’s adaptive redistribution of $\{ \lambda _ { m } \}$ toward intact modalities while retaining signal from the undersampled one. (f) Accuracy under the same ablation (Top-1, %). Pairwise vs. Pairwise+GW (UniOMA). The outer polygon indicates consistent gains from the GW regularizer across all downsampled cases.

Table IV: RL policy performance under full observations and robustness under modality dropout. Locomotion: we report test time average 10-episode reward on an Isaac Lab task: Isaac-Velocity-Flat-Unitree-Go1-v0 (↑; mean ± std over 3 seeds). ‘IMU’ removes the IMUrelated modality, ‘Pos.’ removes joint-position modalities, and $\mathrm { \Omega ^ { \left. \right.} V e l .  }$ removes joint-velocity modalities. Peg insertion: success rate on the Factory task (%, ↑; mean ± std over 5 seeds), with end-efector pose, end-efector velocity, or wrist force removed at test time. The gray rows denote UniOMA.
<table><tr><td>Method</td><td>Full obs.</td><td colspan="3">Modality dropped at test time</td></tr><tr><td>Locomotion (reward)</td><td></td><td>IMU</td><td>Pos.</td><td>Vel.</td></tr><tr><td>Baseline PPO</td><td> $\mathbf { 4 0 . 7 9 } \mathrm { \pm 0 . 3 5 }$ </td><td>9.98±2.39</td><td>15.29±2.16</td><td>13.86±2.41</td></tr><tr><td>Pairwise</td><td> $4 0 . 5 0 { \scriptstyle \pm 0 . 3 5 }$ </td><td> $2 6 . 5 0 { \scriptstyle \pm 1 . 7 6 }$ </td><td> $3 2 . 7 4 { \scriptstyle \pm 0 . 9 3 }$ </td><td> $2 8 . 3 1 { \scriptstyle \pm 1 . 5 7 }$ </td></tr><tr><td>Pairwise+GW (ours)</td><td> $4 0 . 7 0 { \scriptstyle \pm 0 . 4 1 }$ </td><td> $\mathbf { 3 1 . 5 1 } { \pm } \mathbf { 0 . 9 4 }$ </td><td>34.73±0.71</td><td> $\mathbf { 2 9 . 0 4 } \pm \mathbf { 1 . 1 9 }$ </td></tr><tr><td>Peg insertion (%)</td><td></td><td>Pose</td><td>Vel.</td><td>Force</td></tr><tr><td>Baseline PPO</td><td>90.8±5.2</td><td>3.6±1.8</td><td>72.3±10.5</td><td>91.7±6.6</td></tr><tr><td>Pairwise</td><td>92.2±2.4</td><td> $2 . 9 { \pm } 1 . 9 $ </td><td>66.0±10.6</td><td>92.3±4.0</td></tr><tr><td>Pairwise+GW (ours)</td><td> $\mathbf { 9 6 . 6 { \pm 1 . 5 } }$ </td><td> ${ \bf 4 . 5 \pm 1 . 3 }$ </td><td> ${ \bf 7 8 . 5 \pm 7 . 4 }$ </td><td>93.9±3.9</td></tr></table>

## D Results: Structural Alignment and Cross-modal Transfer

Table V measures structural alignment directly on VFD and VFP. We compare each embeddingspace kernel of 256 held-out test samples with a fixed input-kernel reference, the arithmetic mean K<sup>¯</sup> of their input-space kernels, using pairwise similarity rank correlation, 10-nearest-neighbor overlap, squared-loss GW distance, and the adjusted Rand index of spectral clusterings. Unlike the GW barycenter $\mathbf { C } _ { \mathbf { x } } ^ { \ast }$ used in training, K<sup>¯</sup> does not depend on the trained encoders or the modality weights and is the same for all methods. Cross-modal transfer freezes the encoders, fits a ridge probe for next-step end-efector orientation on one modality, and applies it unchanged to another (illustrated in Supplementary Material, Fig. 5); we report the average $R ^ { 2 }$ over modality pairs. Adding the GW term to Pairwise improves most structural measures and cross-modal transfer on both datasets. Relational controls that use the same input kernels (Own, Uniform, CKA, and pairwise GW) improve some of the measures, but none matches our GW term on rank correlation, GW distance, and transfer together.

Table V: Structural alignment and cross-modal transfer on VFD and VFP. For each modality, the embedding-space kernel of 256 held-out test samples is compared with the input-kernel reference (mean input-space kernel) by the rank correlation of pairwise similarities, 10-nearestneighbor overlap, GW distance, and the adjusted Rand index of spectral clusterings, averaged over modalities. Transfer is the $R ^ { 2 }$ of a ridge probe fitted on one modality and applied to another (Sec. IV-D); all values are mean ± std over five seeds. Own, Uniform, and CKA add a relational loss to the own input kernel, to the mean input kernel, or a CKA loss to a weighted reference; Pairwise GW aligns the embedding kernels of modality pairs without a consensus. The gray rows denote our method; bold marks the best value in each column.
<table><tr><td rowspan="2">Method</td><td colspan="4">Direct structural-alignment measurements</td><td>Cross-modal transfer</td></tr><tr><td></td><td>Rank correlation ↑ 10-nearest-neighbor overlap↑ GW distance ↓</td><td> $( \times 1 0 ^ { - 2 } )$ </td><td>Adjusted Rand index↑</td><td> $R ^ { 2 } \uparrow$ </td></tr><tr><td>VFD</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Pairwise [35]</td><td>0.318±0.045</td><td>0.205±0.006</td><td>2.78±0.10</td><td>0.146±0.008</td><td>0.228±0.258</td></tr><tr><td>+ Own</td><td>0.408±0.013</td><td>0.213±0.010</td><td>2.79±0.30</td><td>0.159±0.025</td><td>−0.208±0.209</td></tr><tr><td>+ Uniform</td><td>0.417±0.024</td><td>0.218±0.004</td><td>2.63±0.03</td><td>0.160±0.017</td><td>0.194±0.188</td></tr><tr><td>+ CKA</td><td>0.432±0.030</td><td>0.223±0.006</td><td>2.72±0.08</td><td>0.156±0.019</td><td>0.076±0.152</td></tr><tr><td>+ Pairwise GW</td><td>0.305±0.019</td><td>0.208±0.004</td><td>2.82±0.05</td><td>0.141±0.008</td><td>0.366±0.034</td></tr><tr><td>+ GW (ours)</td><td>0.481±0.042</td><td>0.219±0.009</td><td>2.59±0.04</td><td>0.163±0.010</td><td>0.396±0.175</td></tr><tr><td>VFP</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Pairwise [35]</td><td>0.392±0.035</td><td>0.246±0.006</td><td>3.25±0.06</td><td>0.150±0.011</td><td>0.283±0.130</td></tr><tr><td>+ Own</td><td>0.551±0.009</td><td>0.259±0.004</td><td>3.45±0.18</td><td>0.154±0.020</td><td>-0.889±0.201</td></tr><tr><td>+ Uniform</td><td>0.564±0.019</td><td>0.248±0.005</td><td>2.74±0.05</td><td>0.157±0.008</td><td>0.373±0.044</td></tr><tr><td>+ CKA</td><td>0.554±0.019</td><td>0.256±0.003</td><td>2.88±0.10</td><td>0.162±0.008</td><td>0.320±0.127</td></tr><tr><td>+ Pairwise GW</td><td>0.420±0.026</td><td>0.248±0.004</td><td>3.30±0.05</td><td>0.159±0.015</td><td>0.314±0.028</td></tr><tr><td>+ GW (ours)</td><td>0.575±0.020</td><td>0.257±0.007</td><td>2.70±0.05</td><td>0.164±0.007</td><td>0.390±0.090</td></tr></table>

Fig. 6 tracks the GW distance to the input-kernel reference during training: with the GW term, it falls within the first epochs and ends lower than with Pairwise on both datasets. Fig. 4 of the Supplementary Material shows the corresponding similarity matrices on VFP: after training with the GW term, the structure of each modality is closer to the input-kernel reference, with rank correlations of 0.51, 0.67, and 0.56 for vision, force, and proprioception, against 0.35, 0.50, and 0.36 with Pairwise.

## E Results: Eficiency and Scalability

We evaluate scalability on a challenging trajectory-consistency classification task with up to seven modalities (vision, force, proprioception, depth, action, contact, and optical flow). The task determines whether two multimodal samples originate from the same trajectory; details are provided in Supplementary Material, Sec. II. Results for $M \in \{ 4 , 5 , 6 , 7 \}$ modalities are reported in Table II.

As the number of modalities increases, distributional heterogeneity grows and alignment becomes more dificult. Consequently, pairwise contrastive and OT-based baselines decrease in classification accuracy for $M \geq 6$ . In contrast, UniOMA consistently achieves the highest accuracy across all modality counts, demonstrating scalability in high-modality settings.

![](images/4494d9c94ce0d24d4d067f32a6b1b5c3e112857b115143600b7b8d57be30061c.jpg)  
Figure 6: GW distance to the input-kernel reference during training. Lines show the mean ± standard deviation over three seeds.

Table II shows that UniOMA’s wall-clock time per epoch grows approximately linearly with M, while that of OTLC grows superlinearly. For $M \geq 6$ , UniOMA is strictly faster per epoch than the OTLC baseline, with identical peak memory usage.

We further observe stable convergence of the GW barycenter with a small number of solver iterations $( T _ { \mathrm { m a x } } = 5 )$ , as shown in Supplementary Material, Fig. 6. Ablations over $T _ { \operatorname* { m a x } } \in \{ 3 , \ldots , 7 \}$ (Supplementary Material, Table II) confirm that performance is insensitive to this choice. Overall, mini-batch GW inference introduces only moderate overhead and remains practical when scaling modalities.

## F Results: Modality Weights

UniOMA learns modality weights $\{ \lambda _ { m } \}$ that determine each modality’s contribution to the consensus (Appendix A). Fig. 5 shows the modality weights on each dataset; these values are an empirical indicator of modality-wise agreement with the learned structural consensus rather than a measure of the importance of a modality for the task.

Weight sensitivity and kernel bandwidth. To better understand the learned modality weights, we tested their sensitivity to cross-modal disagreement and kernel bandwidth on VFD and VFP. First, we destroyed cross-modal pairing by replacing images in 75% of the training samples with images from other samples. The vision weight decreased from 0.46 to 0.23 on VFD and from 0.63 to 0.38 on VFP. Second, we varied the kernel bandwidth of one modality at a time by factors of 0.25, 0.5, 2, and 4, while keeping the data unchanged. The corresponding weight changes remained below 0.05.

These results suggest that the weights reflect the tested cross-modal corruption and show limited sensitivity to kernel bandwidth. We further compared learned weights with fixed uniform weights $( \lambda _ { m } = 1 / M )$ under the same GW objective. Learned weights showed higher cross-modal transfer $R ^ { 2 }$ under corruption (−0.26 vs. −0.58 on VFD; −1.44 vs. −2.17 on VFP). Thus, the weights serve as an empirical indicator of modality-wise agreement with the learned structural consensus.

## G Ablation Study: Unequal Modality Sampling

To evaluate robustness to asynchronous robot perception, we ablate UniOMA on the VFD classification by downsampling one modality per mini-batch by $\times \frac 1 2$ (vision/force/depth ↓ batch). Specifically, we downsample one modality (e.g., b=32) while keeping others at b=64, which yields unequal sample frequencies and removes strict one-to-one pairing across modalities. We compare UniOMA (Pairwise+GW) with a contrastive-only baseline (Pairwise). As shown in Fig. 5(f), UniOMA (Pairwise+GW) consistently outperforms Pairwise under all downsampling cases, showing that UniOMA remains efective without strict one-to-one cross-modal pairing.

Interpretability (weights). Fig. 5(e) reports the learned modality weights under each downsampling setting: UniOMA shifts the weight toward intact modalities while still retaining non-trivial mass on the undersampled one. Across datasets (Fig. 5(a–d)), the aggregated weights further show that the modality agreeing most with the learned structural consensus difers between tasks.

Interpretability (structure). We visualize the mini-batch GW barycenter in Supplementary Material, Sec. III-D, which shows coherent geometric patterns shared across modalities, indicating that UniOMA aligns intra-modal structure rather than raw features.

## V Conclusions, Limitations, and Future Work

We revisit multimodal alignment through structural consistency: while pointwise correspondences are statistically strong in existing methods, intra-modal geometries can still disagree across modalities. UniOMA reduces this gap by combining contrastive learning with a GW-barycenter regularizer that aligns 3+ modalities to a shared structural consensus. Concretely, UniOMA represents each modality as an intra-modal similarity (kernel) space, computes a dynamic GW barycenter as the structural consensus, and aligns each modality to this consensus with learned modality weights. This barycentric formulation avoids pairwise couplings across modalities, and is well suited to scaling to more modalities. Across VFP, VFD, MuJoCo Push, VAT, VIP, and 4–7 modality settings, UniOMA improves regression, classification, and cross-modal retrieval in most paired comparisons, brings the similarity structure of each modality closer to a fixed input-kernel reference, and learns dataset-specific modality weights that serve as an empirical indicator of modality-wise agreement with the learned structural consensus.

Limitations. UniOMA introduces additional computation due to the barycentric GW updates, but we explicitly quantify this overhead in Table II. Importantly, the runtime scales linearly with the number of modalities, rather than superlinearly, which makes UniOMA practical for 3+ modalities. A second consideration is the choice of intra-modal similarity kernels used to form ${ \bf K } _ { { \bf x } } ^ { ( m ) }$ . While kernel mismatch can afect the learned consensus geometry, we partially address this sensitivity through the kernel-bandwidth test of Sec. IV-F and the feature-extractor swap of Supplementary Material, Sec. III-H, which empirically validate the robustness of UniOMA under reasonable kernel choices.

The shared structural consensus is most appropriate when common physical factors induce related relational patterns across modalities. If the modalities capture largely distinct information, alignment may ignore useful modality-specific structure. In addition, the learned weights are an empirical indicator of modality-wise agreement with the learned structural consensus and do not directly measure modality importance. The entropy term discourages but does not prevent the concentration of the weights on one modality. Better structural alignment does not necessarily yield better downstream performance: the gains in Table I are small on tasks close to saturation, and one comparison shows a decrease. Simpler relational losses that use the same input-space kernels can be comparable to or better than the GW term on the structural measures (Table V). Finally, the GW term increases the training time per epoch by 1.6–1.8× relative to Pairwise (Table II), which may not be justified when the modalities share little structure or when the contrastive baseline is already close to saturation.

Future work. Building on these findings, a promising direction is large-scale real-robot alignment under heterogeneous sampling rates, missing segments, and asynchronous sensor streams, which calls for more robust couplings and barycenter estimation in the presence of temporal misalignment. Another direction is extending the framework beyond symmetric similarity kernels to asymmetric or directed similarities (e.g., temporal precedence or causal structure), enabling the consensus geometry to capture directional relations that arise in contact-rich robotics data.

## References

[1] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, “High-resolution image synthesis with latent difusion models,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2022, pp. 10 684–10 695.

[2] M. Mirza and S. Osindero, “Conditional generative adversarial nets,” arXiv preprint arXiv:1411.1784, 2014.

[3] J. N. Acosta, G. J. Falcone, P. Rajpurkar, and E. J. Topol, “Multimodal biomedical ai,” Nature Medicine, vol. 28, no. 9, pp. 1773–1784, Sep. 2022.

[4] D. Feng, C. Haase-Schütz, L. Rosenbaum, H. Hertlein, C. Gläser, F. Timm, W. Wiesbeck, and K. Dietmayer, “Deep multi-modal object detection and semantic segmentation for autonomous driving: Datasets, methods, and challenges,” IEEE Transactions on Intelligent Transportation Systems, vol. 22, no. 3, pp. 1341–1360, 2021.

[5] S. Steyaert, M. Pizurica, D. Nagaraj, P. Khandelwal, T. Hernandez-Boussard, A. J. Gentles, and O. Gevaert, “Multimodal data fusion for cancer biomarker discovery with deep learning,” Nature Machine Intelligence, vol. 5, no. 4, pp. 351–362, Apr. 2023.

[6] M. A. Lee, Y. Zhu, K. Srinivasan, P. Shah, S. Savarese, L. Fei-Fei, A. Garg, and J. Bohg, “Making sense of vision and touch: Self-supervised learning of multimodal representations for contact-rich tasks,” in 2019 International conference on robotics and automation (ICRA). IEEE, 2019, pp. 8943–8950.

[7] R. Calandra, A. Owens, M. Upadhyaya, W. Yuan, J. Lin, E. H. Adelson, and S. Levine, “The feeling of success: Does touch sensing help predict grasp outcomes?” 2017.

[8] K. He, H. Fan, Y. Wu, S. Xie, and R. Girshick, “Momentum contrast for unsupervised visual representation learning,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2020, pp. 9729–9738.

[9] T. Chen, S. Kornblith, M. Norouzi, and G. Hinton, “A simple framework for contrastive learning of visual representations,” in International conference on machine learning. PMLR, 2020, pp. 1597–1607.

[10] J.-B. Grill, F. Strub, F. Altché, C. Tallec, P. H. Richemond, E. Buchatskaya, C. Doersch, B. A. Pires, Z. D. Guo, M. G. Azar, B. Piot, K. Kavukcuoglu, R. Munos, and M. Valko, “Bootstrap your own latent a new approach to self-supervised learning,” in Proceedings of the 34th International Conference on Neural Information Processing Systems, ser. NIPS ’20. Red Hook, NY, USA: Curran Associates Inc., 2020.

[11] X. Chen and K. He, “Exploring simple siamese representation learning,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2021, pp. 15 750–15 758.

[12] A. v. d. Oord, Y. Li, and O. Vinyals, “Representation learning with contrastive predictive coding,” arXiv preprint arXiv:1807.03748, 2018.

[13] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark et al., “Learning transferable visual models from natural language supervision,” in International conference on machine learning. PMLR, 2021, pp. 8748–8763.

[14] R. Girdhar, A. El-Nouby, Z. Liu, M. Singh, K. V. Alwala, A. Joulin, and I. Misra, “Imagebind: One embedding space to bind them all,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2023, pp. 15 180–15 190.

[15] S. Chen, H. Li, Q. Wang, Z. Zhao, M. Sun, X. Zhu, and J. Liu, “Vast: A vision-audio-subtitletext omni-modality foundation model and dataset,” Advances in Neural Information Processing Systems, vol. 36, pp. 72 842–72 866, 2023.

[16] B. Zhu, B. Lin, M. Ning, Y. Yan, J. Cui, H. Wang, Y. Pang, W. Jiang, J. Zhang, Z. Li et al., “Languagebind: Extending video-language pretraining to n-modality by language-based semantic alignment,” arXiv preprint arXiv:2310.01852, 2023.

[17] H. Luo, L. Ji, M. Zhong, Y. Chen, W. Lei, N. Duan, and T. Li, “Clip4clip: An empirical study of clip for end to end video clip retrieval and captioning,” Neurocomputing, vol. 508, pp. 293–304, 2022.

[18] S. Nair, A. Rajeswaran, V. Kumar, C. Finn, and A. Gupta, “R3m: A universal visual representation for robot manipulation,” 2022.

[19] S. Karamcheti, S. Nair, A. S. Chen, T. Kollar, C. Finn, D. Sadigh, and P. Liang, “Languagedriven representation learning for robotics,” in Robotics: Science and Systems (RSS), 2023.

[20] T. Wang and P. Isola, “Understanding contrastive representation learning through alignment and uniformity on the hypersphere,” in Proceedings of the 37th International Conference on Machine Learning, ser. ICML’20. JMLR.org, 2020.

[21] W. Liang, Y. Zhang, Y. Kwon, S. Yeung, and J. Zou, “Mind the gap: understanding the modality gap in multi-modal contrastive representation learning,” in Proceedings of the 36th International Conference on Neural Information Processing Systems, ser. NIPS ’22. Red Hook, NY, USA: Curran Associates Inc., 2022.

[22] A. Kraskov, H. Stögbauer, and P. Grassberger, “Estimating mutual information,” Physical Review E, vol. 69, no. 6, p. 066138, Jun. 2004.

[23] B. Poole, S. Ozair, A. Van Den Oord, A. Alemi, and G. Tucker, “On variational bounds of mutual information,” in International conference on machine learning, 2019, pp. 5171–5180.

[24] T. M. Cover and J. A. Thomas, Elements of Information Theory. Wiley, Apr. 2005.

[25] J. B. Kinney and G. S. Atwal, “Equitability, mutual information, and the maximal information coeficient,” Proceedings of the National Academy of Sciences, vol. 111, no. 9, pp. 3354–3359, Feb. 2014.

[26] P. Sermanet, C. Lynch, Y. Chebotar, J. Hsu, E. Jang, S. Schaal, and S. Levine, “Time-contrastive networks: Self-supervised learning from video,” 2018 IEEE International Conference on Robotics and Automation (ICRA), pp. 1134–1141, 2017.

[27] M. Guo, Y. Jiang, A. E. Spielberg, J. Wu, and K. Liu, “Benchmarking rigid body contact models,” in Proceedings of The 5th Annual Learning for Dynamics and Control Conference, ser. Proceedings of Machine Learning Research, N. Matni, M. Morari, and G. J. Pappas, Eds., vol. 211. PMLR, 15–16 Jun 2023, pp. 1480–1492.

[28] M. A. Lee, B. Yi, R. Martín-Martín, S. Savarese, and J. Bohg, “Multimodal sensor fusion with diferentiable filters,” in 2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), IEEE. IEEE, Oct. 2020, pp. 10 444–10 451.

[29] G. Welch and G. Bishop, “An introduction to the kalman filter,” USA, Tech. Rep., 1995.

[30] G. Peyré, M. Cuturi, and J. Solomon, “Gromov-wasserstein averaging of kernel and distance matrices,” in International conference on machine learning. PMLR, 2016, pp. 2664–2672.

[31] F. Gong, Y. Nie, and H. Xu, “Gromov-wasserstein multi-modal alignment and clustering,” in Proceedings of the 31st ACM International Conference on Information & Knowledge Management, ser. CIKM ’22. ACM, Oct. 2022, pp. 603–613.

[32] J. Lu, D. Batra, D. Parikh, and S. Lee, ViLBERT: pretraining task-agnostic visiolinguistic representations for vision-and-language tasks. Red Hook, NY, USA: Curran Associates Inc., 2019.

[33] L. H. Li, M. Yatskar, D. Yin, C.-J. Hsieh, and K.-W. Chang, “Visualbert: A simple and performant baseline for vision and language,” arXiv preprint arXiv:1908.03557, 2019.

[34] C. Jia, Y. Yang, Y. Xia, Y.-T. Chen, Z. Parekh, H. Pham, Q. Le, Y.-H. Sung, Z. Li, and T. Duerig, “Scaling up visual and vision-language representation learning with noisy text supervision,” in Proceedings of the 38th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, M. Meila and T. Zhang, Eds., vol. 139. PMLR, 18–24 Jul 2021, pp. 4904–4916.

[35] Y. Tian, D. Krishnan, and P. Isola, “Contrastive multiview coding,” in European conference on computer vision. Springer, 2020, pp. 776–794.

[36] H. Akbari, L. Yuan, R. Qian, W.-H. Chuang, S.-F. Chang, Y. Cui, and B. Gong, “Vatt: Transformers for multimodal self-supervised learning from raw video, audio and text,” Advances in neural information processing systems, vol. 34, pp. 24 206–24 221, 2021.

[37] J.-B. Alayrac, A. Recasens, R. Schneider, R. Arandjelović, J. Ramapuram, J. De Fauw, L. Smaira, S. Dieleman, and A. Zisserman, “Self-supervised multimodal versatile networks,” Advances in neural information processing systems, vol. 33, pp. 25–37, 2020.

[38] B. Chen, A. Rouditchenko, K. Duarte, H. Kuehne, S. Thomas, A. Boggust, R. Panda, B. Kingsbury, R. Feris, D. Harwath et al., “Multimodal clustering networks for self-supervised learning from unlabeled videos,” in Proceedings of the IEEE/CVF international conference on computer vision, 2021, pp. 8012–8021.

[39] J. Liu, S. Chen, X. He, L. Guo, X. Zhu, W. Wang, and J. Tang, “Valor: Vision-audio-language omni-perception pretraining model and dataset,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024.

[40] J. Huang, Y. Li, J. Feng, X. Wu, X. Sun, and R. Ji, “Clover: Towards a unified video-language alignment and fusion model,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 14 856–14 866.

[41] S. Mai, Y. Zeng, S. Zheng, and H. Hu, “Hybrid contrastive learning of tri-modal representation for multimodal sentiment analysis,” IEEE Transactions on Afective Computing, vol. 14, no. 3, pp. 2276–2289, 2022.

[42] S. Moon, A. Madotto, Z. Lin, A. Dirafzoon, A. Saraf, A. Bearman, and B. Damavandi, “Imu2clip: Multimodal contrastive learning for imu motion sensors from egocentric videos and text,” arXiv preprint arXiv:2210.14395, 2022.

[43] N. Shvetsova, B. Chen, A. Rouditchenko, S. Thomas, B. Kingsbury, R. S. Feris, D. Harwath, J. Glass, and H. Kuehne, “Everything at once-multi-modal fusion transformer for video retrieval,” in Proceedings of the ieee/cvf conference on computer vision and pattern recognition, 2022, pp. 20 020–20 029.

[44] H. Xue, Y. Sun, B. Liu, J. Fu, R. Song, H. Li, and J. Luo, “Clip-vip: Adapting pre-trained image-text model to video-language representation alignment,” arXiv preprint arXiv:2209.06430, 2022.

[45] A. Guzhov, F. Raue, J. Hees, and A. Dengel, “Audioclip: Extending clip to image, text and audio,” in ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2022, pp. 976–980.

[46] A. Saporta, A. M. Puli, M. Goldstein, and R. Ranganath, “Contrasting with symile: Simple model-agnostic representation learning for unlimited modalities,” Advances in Neural Information Processing Systems, vol. 37, pp. 56 919–56 957, 2024.

[47] G. Cicchetti, E. Grassucci, L. Sigillo, and D. Comminiello, “Gramian multimodal representation learning and alignment,” arXiv preprint arXiv:2412.11959, 2024.

[48] B. Dufumier, J. Castillo-Navarro, D. Tuia, and J.-P. Thiran, “What to align in multimodal contrastive learning?” arXiv preprint arXiv:2409.07402, 2024.

[49] G. Cicchetti, E. Grassucci, and D. Comminiello, “A triangle enables multimodal alignment beyond cosine similarity,” arXiv preprint arXiv:2509.24734, 2025.

[50] S. Zhu and D. Luo, Enhancing Multi-modal Contrastive Learning via Optimal Transport-Based Consistent Modality Alignment. Springer Nature Singapore, Nov. 2024, pp. 157–171.

[51] M. Shridhar, J. Thomason, D. Gordon, Y. Bisk, W. Han, R. Mottaghi, L. Zettlemoyer, and D. Fox, “Alfred: A benchmark for interpreting grounded instructions for everyday tasks,” in 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, Jun. 2020.

[52] A. Brohan et al., “RT-1: Robotics transformer for real-world control at scale,” arXiv preprint arXiv:2212.06817, 2022.

[53] D. Driess et al., “PaLM-E: An embodied multimodal language model,” in Proceedings of the 40th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 202. PMLR, 23–29 Jul 2023, pp. 8469–8488.

[54] M. J. Kim et al., “OpenVLA: An open-source vision-language-action model,” arXiv preprint arXiv:2406.09246, 2024.

[55] D. Ghosh et al., “Octo: An open-source generalist robot policy,” in Proceedings of Robotics: Science and Systems, Delft, Netherlands, July 2024.

[56] Physical Intelligence, K. Black et al., “π<sub>0.5</sub>: A vision-language-action model with open-world generalization,” arXiv preprint arXiv:2504.16054, 2025.

[57] J. Wojcik, J. Jiang, J. Wu, and S. Luo, “A case study on visual-audio-tactile cross-modal retrieval,” in 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2024, pp. 12 472–12 478.

[58] A. Dutta, E. Burdet, and M. Kaboli, “Visuo-tactile based predictive cross modal perception for object exploration in robotics,” in 2024 IEEE International Symposium on Robotic and Sensors Environments (ROSE). Chemnitz, Germany: IEEE, 2024.

[59] M. Zambelli, Y. Aytar, F. Visin, Y. Zhou, and R. Hadsell, “Learning rich touch representations through cross-modal self-supervision,” in Proceedings of the 2020 Conference on Robot Learning, ser. Proceedings of Machine Learning Research, J. Kober, F. Ramos, and C. Tomlin, Eds., vol. 155. PMLR, 16–18 Nov 2021, pp. 1415–1425.

[60] C. Villani et al., Optimal transport: old and new. Springer, 2008, vol. 338.

[61] A. Korotin, D. Selikhanovych, and E. Burnaev, “Neural optimal transport,” arXiv preprint arXiv:2201.12220, 2022.

[62] A. Korotin, D. Selikhanovych, and E. Burnaev, “Kernel neural optimal transport,” arXiv preprint arXiv:2205.15269, 2022.

[63] M. Arjovsky, S. Chintala, and L. Bottou, “Wasserstein generative adversarial networks,” in International conference on machine learning. PMLR, 2017, pp. 214–223.

[64] K. Ø. Mikalsen, F. M. Bianchi, C. Soguero-Ruiz, and R. Jenssen, “Time series cluster kernel for learning similarities between multivariate time series with missing data,” Pattern Recognition, vol. 76, pp. 569–581, Apr. 2018.

[65] P. P. Liang, Y. Lyu, X. Fan, Z. Wu, Y. Cheng, J. Wu, L. Chen, P. Wu, M. A. Lee, Y. Zhu, R. Salakhutdinov, and L.-P. Morency, “Multibench: Multiscale benchmarks for multimodal representation learning,” arXiv preprint arXiv:2107.07502, 2021.

[66] E. Todorov, T. Erez, and Y. Tassa, “Mujoco: A physics engine for model-based control,” in 2012 IEEE/RSJ International Conference on Intelligent Robots and Systems. IEEE, Oct. 2012.

[67] R. Gao, Z. Si, Y.-Y. Chang, S. Clarke, J. Bohg, L. Fei-Fei, W. Yuan, and J. Wu, “Objectfolder 2.0: A multisensory object dataset for sim2real transfer,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 10 598–10 608.

[68] NVIDIA, M. Mittal, P. Roth et al., “Isaac lab: A gpu-accelerated simulation framework for multi-modal robot learning,” 2025.

[69] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, “Proximal policy optimization algorithms,” 2017, arXiv:1707.06347.

## Supplementary References

[1] K. Ø. Mikalsen, F. M. Bianchi, C. Soguero-Ruiz, and R. Jenssen, “Time series cluster kernel for learning similarities between multivariate time series with missing data,” Pattern Recognition, vol. 76, pp. 569–581, Apr. 2018.

[2] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, and N. Houlsby, “An image is worth 16x16 words: Transformers for image recognition at scale,” 2020.

[3] Y. Gong, Y.-A. Chung, and J. Glass, “AST: Audio spectrogram transformer,” in Proc. Interspeech 2021, 2021, pp. 571–575.

[4] R. Flamary, N. Courty, A. Gramfort, M. Z. Alaya, A. Boisbunon, S. Chambon, L. Chapel, A. Corenflos, K. Fatras, N. Fournier, L. Gautheron, N. T. Gayraud, H. Janati, A. Rakotomamonjy, I. Redko, A. Rolet, A. Schutz, V. Seguy, D. J. Sutherland, R. Tavenard, A. Tong, and T. Vayer, “Pot: Python optimal transport,” Journal of Machine Learning Research, vol. 22, no. 78, pp. 1–8, 2021. [Online]. Available: http://jmlr.org/papers/v22/20-451.html

[5] N. Bonneel, M. van de Panne, S. Paris, and W. Heidrich, “Displacement interpolation using lagrangian mass transport,” ACM Transactions on Graphics, vol. 30, no. 6, pp. 1–12, Dec. 2011.

[6] F. Gong, Y. Nie, and H. Xu, “Gromov-wasserstein multi-modal alignment and clustering,” in Proceedings of the 31st ACM International Conference on Information & Knowledge Management, ser. CIKM ’22. ACM, Oct. 2022, pp. 603–613.

[7] D. Barber and F. Agakov, “The im algorithm: a variational approach to information maximization,” in Proceedings of the 17th International Conference on Neural Information Processing Systems, ser. NIPS’03. Cambridge, MA, USA: MIT Press, 2003, p. 201–208.

[8] B. Poole, S. Ozair, A. Van Den Oord, A. Alemi, and G. Tucker, “On variational bounds of mutual information,” in International conference on machine learning, 2019, pp. 5171–5180.

[9] M. Oquab, T. Darcet, T. Moutakanni, H. Vo, M. Szafraniec, V. Khalidov, P. Fernandez, D. Haziza, F. Massa, A. El-Nouby et al., “DINOv2: Learning robust visual features without supervision,” Transactions on Machine Learning Research, 2024.

[10] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2016, pp. 770–778.

[11] C. Higuera, A. Sharma, C. K. Bodduluri, T. Fan, P. Lancaster, M. Kalakrishnan, M. Kaess, B. Boots, M. Lambeta, T. Wu, and M. Mukadam, “Sparsh: Self-supervised touch representations for vision-based tactile sensing,” in Proceedings of the Conference on Robot Learning (CoRL), 2024.

## A Implementation Details

## A Optimal Transport Plan Estimation

Algorithm 2: OTPlan(K<sup>ˆ</sup> , K)   
Input: Kernel matrices $\hat { \mathbf { K } } \in \mathbb { R } ^ { \hat { N } \times \hat { N } } , \mathbf { K } \in \mathbb { R } ^ { N \times N }$   
Output: Optimal transport matrix $\mathbf { T } ^ { * }$   
Initialize $\begin{array} { r } { \mathbf p \gets \frac { 1 } { N } \mathbf 1 _ { N } , \quad \hat { \mathbf p } \gets \frac { 1 } { \hat { N } } \mathbf 1 _ { \hat { N } } , \quad \mathbf T \gets \hat { \mathbf p } \mathbf p ^ { \top } ; } \end{array}$   
while not converged do   
// Apply network simplex algorithm:   
$\begin{array} { r } { \hat { \mathbf { T } }  \arg \operatorname* { m a x } _ { \mathbf { T } ^ { \prime } \in \Pi ( \hat { \mathbf { p } } , \mathbf { p } ) } \mathrm { t r } ( \hat { \mathbf { K } } ^ { \top } \mathbf { T } ^ { \prime \top } \mathbf { K } \mathbf { T } ) } \end{array}$ ;   
// Line search to find the minimum:   
$a \gets - 2 \mathrm { t r } ( \hat { \mathbf { K } } ^ { \top } ( \hat { \mathbf { T } } - \mathbf { T } ) ^ { \top } { \mathbf { K } ( \hat { \mathbf { T } } - \mathbf { T } ) ) } \mathrm { : }$   
$b \gets { \mathrm { t r } } \big ( \big ( ( \hat { \mathbf { K } } \odot \hat { \mathbf { K } } ) \hat { \mathbf { p } } \mathbf { 1 } ^ { \top } + \mathbf { 1 p } ^ { \top } ( \mathbf { K } \odot \mathbf { K } ) \big ) ^ { \top } ( \hat { \mathbf { T } } - \mathbf { T } ) \big )$   
$c \gets - 2 \big ( \mathrm { t r } ( \hat { \mathbf { K } } ^ { \top } \mathbf { T } ^ { \top } \mathbf { K } ( \hat { \mathbf { T } } - \mathbf { T } ) )$   
$+ \operatorname { t r } ( \hat { \mathbf { K } } ^ { \top } ( \hat { \mathbf { T } } - \mathbf { T } ) ^ { \top } \mathbf { K } \mathbf { T } ) )$   
if $a > 0$ then   
$\lfloor \tau  \operatorname* { m i n } ( 1 , \operatorname* { m a x } ( 0 , - \frac { b + c } { 2 a } ) )$   
else   
$\begin{array} { r } { \underline { \boldsymbol { \tau } } \gets \left\{ \begin{array} { l l } { 1 , } & { \mathrm { i f } \ a + b + c < 0 , } \\ { 0 , } & { \mathrm { o t h e r w i s e } . } \end{array} \right. ; } \end{array}$   
$\mathbf { T }  ( 1 - \tau ) \mathbf { T } + \tau \hat { \mathbf { T } } ;$   
return T;

Algorithm 2 computes an empirical OT plan T by minimizing the GW distance of Eq. (8) for fixed kernels; since its first two terms do not depend on $\mathbf { T } \in \Pi ( { \hat { \mathbf { p } } } , \mathbf { p } )$ , this is equivalent to solving the quadratic program

$$
\operatorname* { m a x } _ { \mathbf { T } \in \Pi ( \hat { \mathbf { p } } , \mathbf { p } ) } ~ f ( \mathbf { T } ) ~ : = ~ \mathrm { t r } ( \hat { \mathbf { K } } ^ { \top } \mathbf { T } ^ { \top } \mathbf { K } \mathbf { T } ) ,
$$

where $\hat { \mathbf { K } } , \mathbf { K } \in \mathbb { R } ^ { N \times N }$ are intra-domain similarity (or distance) matrices and $\Pi ( { \hat { \mathbf { p } } } , \mathbf { p } ) = \left\{ \mathbf { T } \geq 0 \mid \mathbf { T 1 } = \right.$ $\hat { \mathbf p } , \ \mathbf T ^ { \top } \mathbf { 1 } = \mathbf p \}$ is the transportation polytope (doubly-stochastic when $\begin{array} { r } { \hat { \mathbf { p } } = \frac { 1 } { \hat { N } } \mathbf { 1 } _ { \hat { N } } , \mathbf { p } = \frac { 1 } { N } \mathbf { 1 } _ { N } \big ) } \end{array}$ . Here ⊙ is the Hadamard product, so $( \hat { \mathbf { K } } \odot \hat { \mathbf { K } } )$ and $( \mathbf { K } \odot \mathbf { K } )$ are elementwise squares of the corresponding kernels, which makes b compact. We initialize with the independent coupling $\mathbf { T } = \hat { \mathbf { p } } \mathbf { p } ^ { \top }$ and iterate a Conditional Gradient (Frank–Wolfe; FW) update.

The linear oracle (network simplex) and the closed-form line search are given in Supplementary Material, Sec. I-F.

## B GW Barycenter Estimation

For fixed couplings $\mathbf { T } ^ { ( m ) }$ , the update of $\mathbf { C _ { x } }$ in Alg. 3 is the closed-form minimizer of the barycenter objective of Def. 2, where $\oslash$ denotes the element-wise division; the derivation is given in Supplementary Material, Sec. I-G.

Algorithm 3: GW Barycenter Estimation (mini-batch)   
Input: Intra-modal similarity matrices $\{ { \bf K } _ { \bf x } ^ { ( m ) } \} _ { m = 1 } ^ { M }$ (batch size $N _ { m }$ per modality with   
min $\left\{ N _ { m } \right\} = N )$ , modality weights $\lbrace \lambda _ { m } \rbrace _ { m = 1 } ^ { M }$ with $\begin{array} { r } { \lambda _ { m } \geq 0 , ~ \sum _ { m } \lambda _ { m } = 1 } \end{array}$ , uniform   
marginal $\begin{array} { r } { \hat { \bf p } = \frac { 1 } { N } { \bf 1 } _ { N } } \end{array}$ , max iters $T _ { \mathrm { m a x } }$   
Output: Batch-wise structural consensus (GW barycenter) $\mathbf { C } _ { \mathbf { x } } ^ { \ast } \in \mathbb { R } ^ { N \times N }$   
Initialize $\mathbf { C _ { x } }$ as the weighted average of ${ \bf K } _ { { \bf x } } ^ { ( m ) }$ ;   
for $t  0$ to $T _ { \mathrm { m a x } } - 1$ do   
for m ← 1 to M do   
$\Big \lfloor \mathbf { T } ^ { ( m ) } \gets 0 \mathrm { T P l a n } \Big ( \mathbf { C _ { x } } , \mathbf { K } _ { \mathbf { x } } ^ { ( m ) } \Big )$ ; // Alg. 2   
$\begin{array} { r } { \widetilde { \mathbf { C } } \gets \sum _ { m = 1 } ^ { M } \lambda _ { m } \mathbf { T } ^ { ( m ) } \mathbf { K } _ { \mathbf { x } } ^ { ( m ) } \mathbf { T } ^ { ( m ) \top } } \end{array}$ ;   
$\mathbf { C _ { x } } \gets \tilde { \mathbf { C } } \oslash \left( \hat { \mathbf { p } } \hat { \mathbf { p } } ^ { \top } \right)$ ;   
return $\mathbf { C _ { x } ^ { * } }  \mathbf { C _ { x } }$

# Appendix B

Supplementary Material for

UniOMA: Unified Optimal-Transport Multi-Modal Structural Alignment for Robot Perception

Xinrui $\mathrm { Z u } .$ , Mark Hoogendoorn, Kevin Sebastian Luck, Shujian Yu Vrije Universiteit Amsterdam, the Netherlands

## I Implementation Details

## A Implementation: Time-Series Cluster Kernel

We use the Time-series Cluster Kernel (TCK; [1]) to build intra-modal similarity matrices for time-series modalities (e.g., force/torque). TCK fits an ensemble of diagonal covariance Gaussian mixture models (GMMs) with informative priors and computes a posterior membership vector per sample

$$
{ \bf { I } } _ { i } ( { q } ) = \bigl ( \pi _ { 1 } ^ { ( i ) } ( q ) , \ldots , \pi _ { G _ { q } } ^ { ( i ) } ( q ) \bigr ) ^ { \top } , \qquad \sum _ { g = 1 } ^ { G _ { q } } \pi _ { g } ^ { ( i ) } ( q ) = 1 ,
$$

where each component $\pi _ { g } ^ { ( i ) } ( q )$ is the posterior responsibility of mixture g for sequence i under the q-th GMM, i.e.

$$
\pi _ { g } ^ { ( i ) } ( q ) = p \big ( z { = } g \big | \mathbf { x } _ { i } ^ { ( q ) } ; \hat { \theta } _ { q } \big ) ,
$$

where z is the latent mixture index, $\hat { \theta } _ { q }$ is the MAP-EM estimate of the q-th model parameters, and $\mathbf { x } _ { i } ^ { ( q ) }$ is the subsequence of i restricted to the time window and variable subset chosen by that ensemble member.

The final kernel is the sum of posterior inner products over the ensemble:

$$
( K _ { \mathrm { T C K } } ) _ { i j } \ \gets \ \sum _ { q \in \mathcal { Q } } \mathbf { \Pi } \Pi _ { i } ( q ) ^ { \top } \mathbf { \Pi } \Pi _ { j } ( q ) ,
$$

which is positive semidefinite as a sum of linear kernels. In practice, for time-critical training, we fit the TCK ensemble (the GMM parameters) on the training trajectories only. With this ensemble fixed, we compute the posteriors of the training, validation, and test samples separately and precompute one symmetric matrix per split, ${ \bf K } _ { \mathrm { T C K } } \in \mathbb { R } ^ { N \times N }$ over the N samples of that split; no kernel entry couples samples of diferent splits. During mini-batch training, the intra-modal similarity submatrix for an index set ${ \mathcal { T } } \subseteq \{ 1 , \ldots , N \}$ is obtained by simple indexing

$$
\mathbf { K } _ { \mathrm { b a t c h } } \ = \ K _ { \mathrm { T C K } } [ \mathcal { T } , \mathcal { T } ] ,
$$

thus avoiding repeated TCK fits inside the inner learning loop. Training uses only the training-split matrix, and the test-split matrix is used only for the final evaluation; no test trajectory enters TCK fitting, training, or model selection. We follow the original TCK protocol to induce ensemble diversity (random time windows and variable subsets, random initializations, and varying mixture counts), and we cache per-member posteriors to enable fast posterior lookups at test time. See §4.1–4.4 of [1] for modeling details. In practice, we set the maximal number of mixtures C and the number of randomizations Q as the only user-set hyperparameters. We set C=30 and Q=15 for force/torque signals in the VFP and VFD settings.

## B Implementation: Pre-trained Features

We use pre-trained feature extractors for some modalities to produce modality-specific features whose pairwise similarities form the input-space kernels $\{ { \bf K } _ { \bf x } ^ { ( m ) } \} _ { m = 1 } ^ { M }$ used by our structural consensus $\mathbf { C } _ { \mathbf { x } } ^ { \ast }$ . For the time-series modality (e.g., force/torque), we directly use the TCK method to obtain the input-space kernels.

Pre-trained feature extractors (frozen).

• Vision / Depth / Tactile / Optical Flow: Vision Transformer (ViT-B/16) [2], taking the final [CLS] embedding. Single-channel inputs (e.g., depth) are replicated to 3 channels before preprocessing.

• Force: Time-Series Cluster Kernel (TCK; [1]) directly forms $\mathbf { K } _ { \mathbf { x } } ^ { ( \mathrm { f o r c e } ) } ~ ( \mathrm { S e c . ~ I - A } )$

• Audio (VAT): A frozen Audio Spectrogram Transformer (AST-B, AudioSet-pretrained; [3]) on log-mel spectrograms; we take the [CLS] embedding and build ${ \bf K } _ { \bf x } ^ { \mathrm { ( a u d ) } }$ with an RBF kernel (median heuristic, Sec. I-C).

• Other modalities: RBF kernel on frozen features.

![](images/15f0a28e40b5a94dbdb6cf3371412f9fde27f552d3bd69abdf52c7697ebc2110.jpg)  
Figure 1: GW barycenter of input-space kernels on the VFP dataset. Left: intra-modal similarity matrices $K _ { \mathbf { x } } ^ { ( 1 ) }$ (Vision), $K _ { \mathbf { x } } ^ { ( 2 ) }$ (Force), and $K _ { \mathbf { x } } ^ { ( 3 ) }$ (Proprioception), each min–max normalized for display. Right: the batch-wise structural consensus $\mathbf { C } _ { \mathbf { x } } ^ { \ast }$ obtained by solving $\underset { { \bf C } _ { \bf x } } { \operatorname* { m i n } } \sum _ { m = 1 } ^ { 3 } \lambda _ { m } d _ { g w } ( { \bf C _ { x } } , K _ { \bf x } ^ { ( m ) } )$ (with $\lambda _ { m } { = } 1 / 3$ here). The barycenter preserves recurrent block/trajectory patterns shared across modalities while smoothing modality-specific artifacts, and is later used to regularize the embeddingspace geometry in Stage 2. The batch size is 64.

## C Implementation: Modality Encoders

To avoid architectural confounds, all methods share identical backbones and training schedules. In UniOMA (Stage 2), each modality encoder $\mathcal { E } _ { \theta } ^ { ( m ) }$ produces a feature $\mathbf { h } ^ { ( m ) } \in \mathbb { R } ^ { d _ { h } }$ , which is passed through a modality-specific MLP projector $g _ { \theta } ^ { ( m ) }$ to a shared embedding size d=256:

$$
\mathbf { z } ^ { ( m ) } = \mathbf { \Phi } _ { g _ { \theta } } ^ { ( m ) } ( \mathcal { E } _ { \theta } ^ { ( m ) } ( \mathbf { x } ^ { ( m ) } ) ) \in \mathbb { R } ^ { d } , \quad f _ { \theta } ^ { ( m ) } = g _ { \theta } ^ { ( m ) } \circ \mathcal { E } _ { \theta } ^ { ( m ) }
$$

and the embedding-space kernel within a mini-batch is

$$
\begin{array} { r } { \big ( \mathbf { K } _ { \mathbf { z } } ^ { ( m ) } \big ) _ { i j } = \exp \Big ( - \gamma \| \mathbf { z } _ { i } ^ { ( m ) } - \mathbf { z } _ { j } ^ { ( m ) } \| _ { 2 } ^ { 2 } \Big ) , \qquad \gamma \ = \ \Big ( 2 \ \mathrm { m e d i a n } _ { a , b } \| \mathbf { z } _ { a } ^ { ( m ) } - \mathbf { z } _ { b } ^ { ( m ) } \| _ { 2 } ^ { 2 } \Big ) ^ { - 1 } , } \end{array}
$$

where the median is taken over the squared distances of the mini-batch (median heuristic); the input-space RBF kernels use the same rule. (Stage 1 input-space kernels $\{ \mathbf { K } _ { \mathbf { x } } ^ { ( m ) } \}$ are computed independently using frozen extractors; see Sec. I-B.)

Backbones.

• Vision / Depth / Tactile (image-based). A 2D CNN (ResNet-18). Single-channel inputs (e.g., depth, some tactile images) are replicated to 3 channels before feeding into the backbone.

• Force (time series). A 1D temporal ConvNet built from stacked causal Conv1D layers (kernel size 2, stride 2) with LeakyReLU activations; the final feature map is flattened to obtain a fixed-length embedding.

• Proprioception. A 3-layer MLP with LeakyReLU activations, mapping the low-dimensional pose / joint vector to the shared embedding space.

• Audio (VAT). A 1D CNN with three convolutional blocks (channels $1 \to 6 4 \to 1 2 8 \to 2 5 6 .$ kernel size 5, stride 2), each followed by ReLU, then AdaptiveAvgPool1d(1), flatten(), and a final Linear $\cdot ( 2 5 6 \to d _ { h } )$

• Action. A small MLP mapping the action vector to a compact embedding, implemented as $\mathrm { L i n e a r } ( d _ { a }  3 2 ) \mathrm { - L e a k y R e L U - L i n e a r } ( 3 2  3 2 ) \mathrm { - L e a k y R e L U }$

• Contact. A lightweight MLP applied to the binary contact state, using a structure analogous to the action branch (two Linear layers with LeakyReLU) to obtain a 32-dimensional embedding.

• Optical Flow. We take one channel of the dense optical-flow field (e.g., horizontal component or magnitude), resize it to 128×128, replicate it to 3 channels, and feed it through the same ResNet-18 image encoder as for RGB and depth.

We fix the projector output to d=256, use the same temperature τ for the contrastive term, and share optimizer, batch size, and schedule across methods. UniOMA augments the contrastive loss with a GW-barycenter regularizer (weight α) and learnable modality weights $\{ \lambda _ { m } \}$ (softmaxparameterized to enforce $\lambda _ { m } \geq 0$ and $\Sigma _ { m } \lambda _ { m } = 1 )$ . Encoders and projectors are trained end-to-end with the UniOMA objective; the structure-aware term is computed on $\{ \mathbf { K } _ { \mathbf { z } } ^ { ( m ) } \}$ , while Stage 1 kernels $\{ \mathbf { K } _ { \mathbf { x } } ^ { ( m ) } \}$ remain fixed within each epoch.

## D Implementation: Hyper-parameters

Shared training. Unless otherwise noted, all methods use the same backbone–projector settings. We optimize with AdamW (learning rate $3 \times 1 0 ^ { - 4 }$ , weight decay $1 0 ^ { - 4 } , \beta _ { 1 } { = } 0 . 9 , \beta _ { 2 } { = } 0 . 9 9 9 )$ , batch size 64, and temperature $\tau { = } 0 . 1$ . Each modality head outputs a d=256-dimensional embedding via a lightweight MLP projector (shared width across modalities). We train for 200 epochs with early stopping on the validation metric when applicable, and report mean±std over 10 independent seeds, unless otherwise noted (the structural and weight analyses use 30 epochs and five seeds; Sec. III-A).

Stage-1 input-space kernels. Pre-trained feature extractors for vision/depth/tactile (ViT-B/16   
via timm) are frozen to compute $\{ \mathbf { K } _ { \mathbf { x } } ^ { ( m ) } \}$ . For force/torque we use TCK with max mixtures C=30

and randomizations $Q { = } 1 5$ following Sec. I-A. For VAT audio, we use AST-B as in Sec. I-B to form features and then an RBF kernel. To avoid repeated online estimation during Stage 2, we compute the force kernel of each split once with the TCK ensemble fitted on the training trajectories and cache it; mini-batch kernels $\mathbf { K } _ { \mathrm { b a t c h } } ^ { \mathrm { ( f o r c e ) } }$ are obtained by submatrix indexing.

Stage-2 embedding-space kernels. All modalities use the same Gaussian kernel

$$
\begin{array} { r } { \left( \mathbf { K } _ { \mathbf { z } } ^ { ( m ) } \right) _ { i j } = \exp \big ( - \gamma \| \mathbf { z } _ { i } ^ { ( m ) } - \mathbf { z } _ { j } ^ { ( m ) } \| _ { 2 } ^ { 2 } \big ) , } \end{array}
$$

with $\gamma$ set by the median heuristic of Sec. I-C.

UniOMA-specific. The GW regularization weight $\alpha = 1 0 0 0$ . Modality weights $\{ \lambda _ { m } \}$ are learnable with a softmax parameterization $\begin{array} { r } { ( \lambda _ { m } \geq 0 , ~ \sum _ { m } \lambda _ { m } = 1 ) } \end{array}$ and initialized uniformly; the entropy term of the training objective uses $\beta = 1$ and $\varepsilon = 1 0 ^ { - 9 }$ . For the coupling oracle in OTPlan we use a Frank–Wolfe linearization; the linear subproblem is solved with a network-simplex transportation solver. The line search on the FW segment uses the closed-form quadratic coeficients $( a , b , c )$ derived in Sec. I-F. GW barycenter iterations are run with a maximum of $T _ { \mathrm { m a x } } { = } 5$ per inner-loop (we analyze this choice in Sec. III-B).

To further justify these design choices, we provide a hyper-parameter ablation in Sec. III-C, where we evaluate the efects of α and $T _ { \mathrm { m a x } }$

## E PPO Settings for End-to-End RL

Table I lists the PPO and rollout settings of the two RL tasks. Within each task, the baseline, pairwise alignment, and UniOMA share these settings and difer only in the alignment loss. The peg-insertion task uses the 8 mm peg and hole of Isaac Lab Factory (diameters 7.99 mm and 8.1 mm, i.e., about 0.11 mm clearance) with a Franka arm. An episode lasts 10 s (150 policy steps), and a trial counts as successful when the peg is centered within 2.5 mm of the hole axis and seated within 1 mm of the target depth.

## F Optimal Transport Plan Estimation: Linear Oracle and Line Search

This section details Alg. 2 (OTPlan) of the main paper, which maximizes $f ( \mathbf { T } ) = \mathrm { t r } ( \hat { \mathbf { K } } ^ { \top } \mathbf { T } ^ { \top } \mathbf { K } \mathbf { T } )$ over $\scriptstyle \Pi ( \hat { \mathbf { p } } , \mathbf { p } )$ , which is equivalent to minimizing the GW distance of Eq. (8) of the main paper for fixed kernels.

Network simplex algorithm. At each iteration, we linearize f and solve

$$
\hat { \mathrm { ~ \bf ~ T ~ } } \in \arg \operatorname* { m a x } _ { \mathrm { { \bf ~ T } ^ { \prime } \in \Pi ( \hat { \mathrm { ~ p } } , \mathrm { { p } } ) } } \langle \mathrm { { \bf ~ T } ^ { \prime } } , \mathrm { { \nabla } } \nabla f ( \mathrm { { \bf ~ T } } ) \rangle .
$$

For $f ( \mathbf { T } ) = \mathrm { t r } ( \hat { \mathbf { K } } ^ { \top } \mathbf { T } ^ { \top } \mathbf { K } \mathbf { T } )$ , we use the gradient form

$$
\nabla f ( \mathbf { T } ) = \mathbf { K } \mathbf { T } \hat { \mathbf { K } } + \mathbf { K } ^ { \top } \mathbf { T } \hat { \mathbf { K } } ^ { \top } ,
$$

which reduces to 2 K T K<sup>ˆ</sup> when ${ \bf K } , \hat { \bf K }$ are symmetric, so that $\langle \mathbf { T } ^ { \prime } , \nabla f ( \mathbf { T } ) \rangle = 2 \operatorname { t r } ( \hat { \mathbf { K } } ^ { \top } \mathbf { T } ^ { \prime \top } \mathbf { K } \mathbf { T } )$ as in Alg. 2. The oracle is a linear transportation problem. We implement it using a network simplex $[ 4 , 5 ]$

Line search. Define the search segment $\mathbf { T } ( \tau ) = ( 1 - \tau ) \mathbf { T } + \tau \hat { \mathbf { T } } , \tau \in [ 0 , 1 ]$ . Substituting $\mathbf { T } ( \tau )$ into the GW distance of Eq. (8) of the main paper yields the univariate quadratic $d ( \tau ) = d ( 0 ) + a \tau ^ { 2 } + ( b + c ) \tau$ where $d ( \tau )$ is the GW distance at $\mathbf { T } ( \tau )$ , with the coeficients of Alg. 2 of the main paper: a and c come from the trace term and b from the two quadratic terms $( b = 0$ when T and $\hat { \mathbf { T } }$ satisfy the marginal constraints exactly). The code computes $( a , b , c )$ and picks the minimizer on [0, 1]: $\tau ^ { \star } = \operatorname* { m i n } ( 1 , \operatorname* { m a x } ( 0 , - ( b + c ) / ( 2 a ) ) )$ if $a > 0$ , otherwise $\tau ^ { \star } \in \{ 0 , 1 \}$ by comparing endpoints. We then set $\mathbf { T } = \mathbf { T } ( \tau ^ { \star } )$

Table I: PPO and rollout settings for the RL experiments (Go1 locomotion and peg insertion). These settings are shared by the baseline, pairwise alignment, and UniOMA variants unless otherwise stated.
<table><tr><td>Setting</td><td>Go1 locomotion</td><td>Peg insertion</td></tr><tr><td>Environment</td><td>Isaac-Velocity-Flat-Unitree-Go1-vO</td><td>Factory peg insertion (+ wrist</td></tr><tr><td>RL algorithm</td><td>PPO</td><td>force) PPO</td></tr><tr><td>Parallel environments</td><td>4096</td><td>512</td></tr><tr><td>Rollout length</td><td>24 steps/env</td><td>64 steps/env</td></tr><tr><td>Transitions per PPO it- eration</td><td>98,304</td><td>32,768</td></tr><tr><td>Mini-batch size</td><td>24,576 transitions</td><td>4,096 transitions</td></tr><tr><td>Learning iterations</td><td>500 for test-time dropout; 1000 for training</td><td>300</td></tr><tr><td>Training-dropout activa- iteration 500</td><td>dropout</td><td>none</td></tr><tr><td>tion Optimizer</td><td>Adam</td><td>Adam</td></tr><tr><td>Learning rate</td><td>1.0×10−3, adaptive schedule</td><td>1.0×10−4, adaptive schedule</td></tr><tr><td>Desired KL</td><td>0.01</td><td>0.008</td></tr><tr><td>Discount factor γ</td><td>0.99</td><td>0.995</td></tr><tr><td>GAE parameter λ</td><td>0.95</td><td>0.95</td></tr><tr><td>PPO epochs per itera- tion</td><td></td><td>4</td></tr><tr><td>Mini-batches per itera- tion</td><td></td><td>8</td></tr><tr><td>PPO clip parameter</td><td>0.2</td><td>0.2</td></tr><tr><td>Value loss coefficient</td><td>1.0</td><td>2.0</td></tr><tr><td>Entropy coefficient</td><td>0.01</td><td>0.0</td></tr><tr><td>Max gradient norm</td><td>1.0</td><td>1.0</td></tr><tr><td>Clipped value loss</td><td>enabled</td><td>enabled</td></tr><tr><td>Observation normaliza- disabled tion</td><td></td><td>enabled</td></tr><tr><td>Actor/critic hidden lay- [128, 128, 128] ers</td><td></td><td>[512, 256, 128]</td></tr><tr><td>Activation</td><td>ELU</td><td>ELU</td></tr><tr><td>Action distribution</td><td>Gaussian, initial std 1.0</td><td>Gaussian, initial std 1.0</td></tr><tr><td>Training seeds</td><td>3</td><td>5</td></tr><tr><td>Evaluation</td><td>average 10-episode reward</td><td>average 512-episode success rate</td></tr></table>

## G GW Barycenter Update

Consider the barycenter objective (Def. 2 of the main paper):

$$
\begin{array} { r l } & { \mathbf { C } _ { \mathbf { x } } ^ { * } = \underset { \mathbf { C } _ { \mathbf { x } } \in \mathcal { M } } { \arg \operatorname* { m i n } } \displaystyle \sum _ { m = 1 } ^ { M } \lambda _ { m } \cdot \hat { d } _ { g w } \Big ( \mathbf { C } _ { \mathbf { x } } , \mathbf { K } _ { \mathbf { x } } ^ { ( m ) } \Big ) , } \\ & { \qquad \quad \lambda _ { m } \geq 0 , \quad \displaystyle \sum _ { m = 1 } ^ { M } \lambda _ { m } = 1 . } \end{array}
$$

By Eq. (8) of the main paper, each term consists of the quadratic term of $\mathbf { C _ { x } } .$ , the quadratic term of ${ \bf K } _ { { \bf x } } ^ { ( m ) }$ , and a trace term, and the barycenter minimizes their weighted sum. Fix couplings $\{ \mathbf { T } ^ { ( m ) } \} _ { m = 1 } ^ { M }$

with $\mathbf { T } ^ { ( m ) } \in \Pi ( \hat { \mathbf { p } } , \mathbf { p } ^ { ( m ) } )$ for the current consensus $\mathbf { C _ { x } } .$ , and define

$$
\mathbf { A } ^ { \left( m \right) } : = \mathbf { T } ^ { \left( m \right) } \mathbf { K } _ { \mathbf { x } } ^ { \left( m \right) } \mathbf { T } ^ { \left( m \right) ^ { \top } } \mathbf { \Lambda } \in \mathbf { \Gamma } \mathbb { R } ^ { N \times N } ,
$$

as $\mathbf { C _ { x } }$ -independent constants; since $\Sigma _ { m } \lambda _ { m } = 1$ , the objective reduces to

$$
\begin{array} { r l r } {  { \mathcal { I } ( \mathbf { C } _ { \mathbf { x } } ) } } & { = \hat { \mathbf { p } } ^ { \top } ( \mathbf { C } _ { \mathbf { x } } \odot \mathbf { C } _ { \mathbf { x } } ) \hat { \mathbf { p } } }  \\ & { } & { \quad - 2 \displaystyle \sum _ { m = 1 } ^ { M } \lambda _ { m } \operatorname { t r } ( \mathbf { C } _ { \mathbf { x } } ^ { \top } \mathbf { A } ^ { ( m ) } ) + \mathrm { c o n s t . } } \end{array}
$$

Following the standard GW-barycenter normalization (as in Eq. (8) of [6]), we take the derivative with respect to $\mathbf { C _ { x } }$ and set it to zero

$$
\begin{array} { r l } & { \displaystyle \frac { \partial \mathcal { I } ( { \mathbf { C } _ { \mathbf { x } } } ) } { \partial \mathbf { C } _ { \mathbf { x } } } = 2 \left( \hat { \mathbf { p } } \hat { \mathbf { p } } ^ { \top } \right) \odot \mathbf { C } _ { \mathbf { x } } - 2 \sum _ { m } \lambda _ { m } \mathbf { A } ^ { ( m ) } = \mathbf { 0 } } \\ & { \displaystyle \Rightarrow \quad \mathbf { C } _ { \mathbf { x } } = \left( \sum _ { m = 1 } ^ { M } \lambda _ { m } \mathbf { A } ^ { ( m ) } \right) \oslash \left( \hat { \mathbf { p } } \hat { \mathbf { p } } ^ { \top } \right) , } \end{array}
$$

i.e.

$$
\begin{array} { r l } & { \mathbf { \bar { C } _ { x } } \gets \mathbf { \widetilde { C } } \otimes \big ( \hat { \mathbf { p } } \hat { \mathbf { p } } ^ { \top } \big ) , } \\ & { \quad \quad \quad \quad \widetilde { \mathbf { C } } = \displaystyle \sum _ { m = 1 } ^ { M } \lambda _ { m } \mathbf { T } ^ { ( m ) } \mathbf { K } _ { \mathbf { x } } ^ { ( m ) } \mathbf { T } ^ { ( m ) } } .  \end{array}\tag{1}
$$

Here $\oslash$ denotes the element-wise division.

## II Datasets and Preprocessing

We detail the exact splitting, windowing, and per-modality preprocessing used in our experiments. Unless specified, all randomization uses a fixed seed (seed=42), and splits are performed at the file/trajectory level to avoid leakage.

VFD / VFP (Vision–Force–Depth / Vision–Force–Proprioception). The data contain 3,000 trajectories; with 48 samples per trajectory, this gives 144,000 samples. We split the data at the trajectory level with seed = 42 into training, validation, and test trajectories at a ratio of 80/10/10 (48 samples per trajectory). Validation data are used for early stopping and checkpoint selection, and test data only for the final evaluation; the TCK model is fitted on the training trajectories only. Each episode has length 48. For time step t, we form a fixed history window of length L for force (default L=32) and read targets at t+1. RGB images are center-cropped to 128×128, normalized by ImageNet statistics (mean [0.485, 0.456, 0.406], std [0.229, 0.224, 0.225]). Depth is stored as (128, 128, 1), normalized by mean 0.5/std 0.5, and used as single-channel tensors. Force–torque histories are truncated to the last L steps. The resulting tensor has shape $[ b \times L \times 6 ]$ Proprioception is parsed from the first 7 pose components (end efector position/orientation) in the loader and returned as [b×7] at the current step.

Tasks. For VFD, we follow the main text: (1) next-step end-efector orientation regression (4D) and (2) trajectory-pair discrimination (binary classification of whether two samples come from the same trajectory), using $( \mathrm { R G B } _ { t } , \mathrm { F / T } _ { t - L + 1 : t } , \mathrm { D e p t h } _ { t } )$ as inputs. For VFP, we perform next-step contact prediction (binary) using $( \mathrm { R G B } _ { t } , \dot { \mathrm { F / T } } _ { t - L + 1 : t } , \mathrm { P r o p r i o } _ { t } )$ ; class balance is enforced by uniform sampling across trajectories.

![](images/541ad16c48fb26433edf72e297a9c297aa721e6bde2a46c6af93464c5ce5c8bb.jpg)  
Figure 2: VFD overview. Synchronized windows of RGB images, force–torque signals (last L=32 steps), and depth camera images. Images are center-cropped to 128×128 and normalized. Depth images are normalized with mean/std 0.5.

MuJoCo Push. A planar pushing task with a Franka Panda arm interacting with a puck. The dataset contains 1,310 trajectories of 240 time steps, of which 1,000 are used for training, 10 for validation, and 300 for testing. Image modality: we use sequences of grayscale frames. Each sample contains a length S = 32 subsequence of 32×32 frames, forming tensors of shape (B, S, 1, 32, 32). The force modality concatenates the current force–torque signal and the contact flag, forming tensors of shape (B, 7), and the end-efector position modality forms (B, 3).

VAT (Vision–Audio–Tactile). We assemble object-level triplets from per-class folders. The data cover 20 objects with 425, 75, and 75 samples per object and modality for training, validation, and testing (8,500, 1,500, and 1,500 in total); 12 tuples whose visual image is blank are excluded. The tactile observations are the GelSight images rendered by ObjectFolder 2.0 (160×120), and each audio clip lasts 3 s at 44.1 kHz. We use predefined train / validate / test directory structures over a fixed object list. Labels for the retrieval tasks are integer-encoded. Visual and tactile images are resized to 246×246 and normalized by ImageNet statistics. Audio is loaded at its native sampling rate; at test time, the raw waveform is truncated to TARGET\_LENGTH = 132,300 samples. The final shape of the tensors is (B, 132300).

Task. Cross-modal retrieval with relevance at the object identity level; we report direction-specific MAP on the test set.

VIP (Vision–IMU–Proprioception). To complement benchmark and simulation-based evaluations, we further assess UniOMA on a real-world robot-centric dataset collected on a physical 6-DoF robotic manipulator. This dataset is designed to evaluate whether the learned multimodal structural alignment generalizes to noisy and embodiment-specific sensing commonly encountered in real robotic systems.

Data collection. The robot is a commercial 6-DoF serial manipulator with a reach of 900 mm, operated without payload and without contact. Each trajectory is a reaching motion toward a sampled end-efector target, executed by a trained reaching policy that commands joint increments of at most $1 0 ^ { \circ }$ per step. The initial joint configuration is sampled uniformly from $[ \pm 1 8 0 ^ { \circ } , \pm 3 0 ^ { \circ } , \pm 9 0 ^ { \circ } , \pm 1 8 0 ^ { \circ } , \pm 1 8 0 ^ { \circ } , \pm 1 8 0 ^ { \circ } ]$ , so that the trajectories cover varied initial configurations and reaching targets. All data were collected in a single workspace under constant indoor lighting.

Scale and split. The dataset contains 1,000 trajectories of 100 steps recorded at 5 Hz (20 s each), i.e., 100,000 synchronized samples. We split it at the trajectory level into training, validation, and test trajectories at a ratio of 80/10/10 (80,000/10,000/10,000 samples); training and test trajectories are drawn from the same distribution, and there are no held-out conditions.

Modalities and preprocessing. Each sample consists of three synchronized modalities: (1) a third-person RGB image captured from a fixed external camera on a tripod about 2 m from the robot at $3 2 0 \times 2 4 0$ , resized to $6 4 \times 6 4$ and normalized using ImageNet statistics (the end-efector is not visible in every frame); (2) proprioceptive signals given by the current joint angles of the 6-DoF arm, represented as a $[ b \times 6 ]$ vector; and (3) inertial measurements (IMU) from the six joint modules, each reporting three linear and three angular accelerations, forming tensors of shape $[ b \times 6 \times 6 ]$ . Each of the 36 inertial channels is standardized with the mean and variance of the training trajectories. The three modalities are hardware-triggered and recorded as single-step readings at the 5 Hz control frequency, without temporal windowing.

Task: end-efector position regression. Given the multimodal input $\left( \mathrm { R G B } _ { t } , \mathrm { P r o p r i o } _ { t } , \mathrm { I M U } _ { t } \right)$ , the model is trained to predict the current Cartesian position of the end-efector (regression, 3D). The ground-truth end-efector position is measured by an optical motion-capture system from markers rigidly attached to the end-efector and is expressed in a table-fixed frame; it is not computed from the joint encoders. We report the mean Euclidean distance between the predicted and measured end-efector positions $( \times 1 0 ^ { - 2 } \mathrm { m } )$ on the held-out test set, as summarized in Table III of the main paper.

MultiBench 4–7 Modality Scalability. To evaluate whether UniOMA scales beyond three modalities, we construct an additional downstream classification benchmark using a multi-sensor subset of the MultiBench dataset. This setting allows us to progressively increase the number of modalities and test whether the learned structural consensus remains stable as modality count grows.

Modalities. We select seven heterogeneous sensing streams commonly used in robotic manipulation:

Vision (RGB) : [3×128×128], Depth : $[ 1 \times 1 2 8 \times 1 2 8 ]$

Force–Torque : [6], Proprioception : [7], Action : $[ d _ { a } ]$

Contact state : [1], Optical Flow : $[ 2 \times 1 2 8 \times 1 2 8 ]$

For a given experiment with $M \in \{ 4 , 5 , 6 , 7 \}$ modalities, we take the first M modalities from this list. All modalities are independently normalized using training-set statistics following the MultiBench protocol.

Task: trajectory-consistency classification. Given two multimodal or single-modal samples, the model must classify whether they originate from the same trajectory. Positive pairs are sampled from two timesteps of the same trajectory; negatives are sampled across distinct trajectories. This task directly evaluates whether embeddings preserve the trajectory-level structure across multiple modalities.

![](images/97547019c74164f1067fe0456552dc2fa0605c32a07ef45fa19a79e0fbd4a05b.jpg)  
Figure 3: VIP dataset overview. Example RGB observations from the real-world robot dataset. Images are captured by a third-person camera while the robot executes reaching trajectories. These visual observations are synchronized with joint proprioception and IMU signals.

## III Additional Experiments

## A Structural Metrics and Cross-modal Transfer

All models of Table V of the main paper (including the relational controls and the direct pairwise GW baseline) and of the weight analyses in Sec. IV-F of the main paper (mispaired images; learned vs. uniform weights) are trained on VFD and VFP for a fixed 30 epochs and evaluated at the final checkpoint, over five seeds (42–46). The structural measures in Table V of the main paper use 256 fixed test samples: the frames at time steps 0, 12, 24, and 36 of 64 test trajectories. For each modality, the embedding-space kernel is an RBF kernel with the median bandwidth on the normalized projected embeddings. As the evaluation reference we use the arithmetic mean of the input-space kernels of the same samples, $\begin{array} { r } { \bar { \bf K } = \frac { 1 } { M } \sum _ { m } { \bf K } _ { \bf x } ^ { ( m ) } } \end{array}$ (the input-kernel reference), not the GW barycenter C<sup>∗</sup> used in training; K<sup>¯</sup> does not depend on the trained encoders or the modality weights and is the same for all methods. Each measure is averaged over the modalities.

• Rank correlation: Spearman correlation between the upper-triangular of-diagonal entries of the two matrices.

• Neighborhood preservation (kNN-10): the average fraction of each sample’s 10 nearest neighbors (excluding itself) that the two matrices share.

• GW distance: the squared-loss GW distance to the input-kernel reference.

• Clustering consistency (ARI): the adjusted Rand index between eight-cluster spectral clusterings of the embedding kernel and of K<sup>¯</sup> .

The training curves of the GW distance (Fig. 6 of the main paper) evaluate the same samples and reference K<sup>¯</sup> at epochs 0, 1, 2, 5, 10, 20, and 30 over three seeds. Fig. 4 shows the similarity matrices on VFP before and after training, from one training seed; their rows and columns are ordered by hierarchical clustering of K<sup>¯</sup> and show ranks of pairwise similarity, so that colors are comparable across panels and do not depend on the kernel scale.

![](images/5780dc81d3db17aafa474334c3e8b003160fbbee7f98e6b2f2eed83cbc1f3acf.jpg)  
Figure 4: Similarity structure before and after training on VFP. Each row is one modality. Left: the input-kernel reference K<sup>¯</sup> (mean of the input-space kernels) of 256 held-out test samples, repeated in every row. Right: the embedding-space similarity matrix of that modality at epoch 0 and after 30 epochs of Pairwise training or training with the GW term; $\rho$ is the rank correlation with K<sup>¯</sup> . All matrices share one sample order and show ranks of pairwise similarity.

For cross-modal transfer, the projected embeddings are frozen. A ridge probe for next-step end-efector orientation is fitted on the training embeddings of one modality, with its regularization selected from $\{ 1 0 ^ { - 6 } , \ldots , 1 0 ^ { - 1 } \}$ by validation $R ^ { 2 }$ on the same modality, and is then applied without refitting to the test embeddings of every other modality. We report the $R ^ { 2 }$ averaged over all ordered modality pairs. Fig. 5 illustrates the protocol.

![](images/18d3ceaf0fa1083974cd18d55631ef994adcee6cb570025ae322c0e050613d6b.jpg)  
Figure 5: Cross-modal transfer protocol. A ridge probe for the next-step end-efector orientation is fitted on the frozen projected embeddings of modality a and applied with the same weights to the test embeddings of another modality b. A high $R ^ { 2 }$ indicates that the two modalities place the samples consistently in the shared embedding space.

## B Hyper-parameter Analysis

Number of GW barycenter iterations $T _ { \mathrm { m a x } }$ . Let $\mathbf { C } ^ { ( t ) }$ be the consensus at inner-loop iteration t in Alg. 3 of the main paper. We monitor the relative Frobenius change $\Delta _ { t } = \| \mathbf C ^ { ( t ) } - \|$ $\mathbf { C } ^ { ( t - 1 ) } \| _ { F } / \| \mathbf { C } ^ { ( t - 1 ) } \| _ { F }$ and the trace objective

$$
\sum _ { m } \lambda _ { m } \operatorname { t r } \bigl ( \mathbf { C } ^ { ( t ) \top } \mathbf { T } ^ { ( m ) \top } \mathbf { K } _ { \mathbf { x } } ^ { ( m ) } \mathbf { T } ^ { ( m ) } \bigr ) .\tag{2}
$$

Both stabilize rapidly; after $t { = } 5$ further changes are negligible $( \Delta _ { t } < 1 0 ^ { - 3 } )$ . We therefore fix $T _ { \mathrm { m a x } } = 5$ for all reported results.

![](images/8bf446a7a201d513972b24c1ffadf8273df1921b2e56c469b85c01ef669d9bca.jpg)  
Figure 6: GW barycenter inner-loop: structural consensus across iterations t in Alg. 3 of the main paper. By t=5, both geometry and objective are efectively stable; we therefore choose $T _ { \mathrm { m a x } } { = } 5$ (batch size 64).

Table II: Ablation of UniOMA hyper-parameters on the VFD classification task (Top-1 accuracy, $\% ,$ mean ± std over 10 seeds). We vary the GW weight α and the number of barycenter iterations $T _ { \mathrm { m a x } }$ around the default settings. UniOMA is robust across a wide range of values; our default choices (in bold) lie near the center of each stable regime.
<table><tr><td colspan="2">GW weight α</td></tr><tr><td> $\alpha = 2 0 0$ </td><td> $9 2 . 1 2 \pm 0 . 0 2$ </td></tr><tr><td> $\alpha = 5 0 0$ </td><td> $9 2 . 3 2 \pm 0 . 0 2$ </td></tr><tr><td> $\alpha = 1 0 0 0$ </td><td> $9 2 . 4 4 \pm 0 . 0 2$ </td></tr><tr><td> $\alpha = 2 0 0 0$ </td><td> $9 2 . 4 0 \pm 0 . 0 4$ </td></tr><tr><td> $\alpha = 5 0 0 0$ </td><td> $9 2 . 2 8 \pm 0 . 0 5$ </td></tr><tr><td>Barycenter iterations</td><td> $T _ { \mathrm { m a x } }$ </td></tr><tr><td> $T _ { \mathrm { m a x } } = 3$ </td><td> $9 2 . 0 5 \pm 0 . 0 3$ </td></tr><tr><td> $T _ { \mathrm { m a x } } = 4$ </td><td> $9 2 . 2 7 \pm 0 . 0 3$ </td></tr><tr><td> $T _ { \mathrm { m a x } } = 5$ </td><td> $9 2 . 4 4 \pm 0 . 0 2$ </td></tr><tr><td> $T _ { \mathrm { m a x } } = 6$ </td><td> $9 2 . 4 5 \pm 0 . 0 2$ </td></tr><tr><td> $T _ { \mathrm { m a x } } = 7$ </td><td> $9 2 . 4 4 \pm 0 . 0 2$ </td></tr></table>

## C Ablation: Hyper-parameter Efects

We empirically ablate two key hyper-parameters of UniOMA—the GW regularization weight α and the number of barycenter iterations $T _ { \mathrm { m a x } }$ . Table II summarizes results on the VFD classification task (Top-1 accuracy, mean ± std over 10 seeds). Performance is stable across a broad range of values around our default settings; too small α under-utilizes structural alignment and too large α marginally over-regularizes the embeddings. The solver iteration number $T _ { \mathrm { m a x } }$ shows a clear plateau around $T _ { \mathrm { m a x } } = 5$ , confirming that a small number of GW barycenter iterations is suficient in practice.

## D Additional Qualitative Visualizations

For the benefit of the reader, we provide further qualitative visualizations of the structural alignment induced by UniOMA. Fig. 7 summarizes five aspects on VFP: (a) a t-SNE embedding of the input space of each modality; (b) input-space similarity matrices for each modality; (c) the corresponding structural consensus; (d) embedding-space similarity matrices after training; and (e) a t-SNE embedding of the learned representations. Together, these views illustrate how UniOMA preserves shared structure while remaining interpretable.

## E Entropy Regularization of the Modality Weights

The training objective includes the entropy term $\begin{array} { r } { \beta \sum _ { m } \lambda _ { m } \log ( \lambda _ { m } + \varepsilon ) } \end{array}$ (Sec. III-D of the main paper). Table III compares the learned modality weights on VFD and VFP with and without this term after 30 epochs. Without the entropy term, the largest weight increases from 0.46 to 0.52 on VFD and from 0.63 to 0.83 on VFP, and the force weight decreases from 0.11 to 0.03 and from 0.13 to 0.06. The term therefore reduces weight concentration. Fig. 8 shows the weights during training in the same runs. With the entropy term, the weights leave the uniform initialization within the first epoch and stay within 0.02 of their final values after 6 epochs on VFD and 1 epoch on VFP. Without the term, the vision weight rises further, increases over 0.6 early in training, and stabilizes at 0.52 on VFD.

![](images/28b6a95664eade0172a02c67bc1dafe5b18448e87e9631ffa49b74045d8ed33d.jpg)  
Figure 7: Qualitative visualizations of UniOMA on the VFP benchmark. We choose a mini-batch with size 256 to illustrate the interpretation of the structural consensus and the aligned embeddings. Rows correspond to the modalities (image, force, and proprioception). First column: t-SNE visualizations of the input modalities, showing clear sub-cluster structures (colors indicate groups of samples that are similar in the consensus). Second column: Input-space similarity matrices ${ \bf K } _ { { \bf x } } ^ { ( m ) }$ , showing block structure and modality-specific artifacts. Middle: The structural consensus $\mathbf { C } _ { \mathbf { x } } ^ { \ast }$ , which preserves the shared block structure while smoothing modality-specific noise. Fourth column: Embedding-space similarity matrices ${ \bf K } _ { { \bf z } } ^ { ( m ) }$ after training, which show the block structure of the consensus for all three modalities. Last column: t-SNE of the learned embeddings, where the groups form coherent clusters across modalities, indicating successful structural alignment beyond pairwise correspondence. All matrices share the same sample order, marked by the color strip on their left.

Table III: Final modality weights with and without the entropy term. Mean ± std over five seeds after 30 epochs.
<table><tr><td>Dataset</td><td>Modality</td><td>With entropy</td><td>Without entropy</td></tr><tr><td rowspan="3">VFD</td><td>Vision</td><td>0.460±0.001</td><td> $0 . 5 1 8 { \scriptstyle \pm 0 . 0 0 1 }$ </td></tr><tr><td>Force</td><td>0.108±0.001</td><td> $0 . 0 3 2 { \scriptstyle \pm 0 . 0 0 0 }$ </td></tr><tr><td>Depth</td><td>0.432±0.001</td><td> $0 . 4 5 1 { \scriptstyle \pm 0 . 0 0 1 }$ </td></tr><tr><td rowspan="3">VFP</td><td>Vision</td><td> $0 . 6 3 4 { \scriptstyle \pm 0 . 0 0 2 }$ </td><td> $0 . 8 3 2 { \scriptstyle \pm 0 . 0 0 1 }$ </td></tr><tr><td>Force</td><td> $0 . 1 3 0 { \scriptstyle \pm 0 . 0 0 1 }$ </td><td> $0 . 0 5 7 { \scriptstyle \pm 0 . 0 0 1 }$ </td></tr><tr><td>Proprioception</td><td> $0 . 2 3 6 { \scriptstyle \pm 0 . 0 0 2 }$ </td><td> $0 . 1 1 1 { \scriptstyle \pm 0 . 0 0 1 }$ </td></tr></table>

## F Learning Curves of Peg Insertion

Fig. 9 shows the learning progress of the peg-insertion task under full observations (Table IV of the main paper), over five seeds. The success rate is evaluated with the deterministic policy on 512 episodes at iterations 0, 50, 100, 150, 200, and 250 and at the final iteration; the training reward is logged at every PPO iteration. Pairwise+GW has the highest mean success rate at every checkpoint from iteration 50 onward and ends at 96.6%, compared with 92.2% for Pairwise and 90.8% for Baseline $\mathrm { P P O } ;$ at the end of training, the training rewards of the three methods are similar.

![](images/7be1f960fe8ea4bc85d6d86d166c7d566804fbf445e2b9b2d4ec60ed90383e42.jpg)  
Figure 9: Peg-insertion learning curves under full observations. (a) Success rate of the deterministic policy at the saved checkpoints; the numbers give the final values. (b) Training reward per PPO iteration, smoothed over five iterations. Lines show the mean and bands ±1 standard deviation over five seeds.

![](images/274f225387b3edb2318827d6c500227bd785062419fa12c18f4fac0cbc8bf790.jpg)  
Figure 8: Modality weights during training with and without the entropy term. Same runs as Table III (VFD and VFP, 30 epochs). Top: lines show the mean and bands ±1 standard deviation over five seeds; the dotted line marks uniform weights. Bottom: the standard deviation over the five seeds (running mean over one epoch); the dotted line marks 0.005.

## G Information-Retention Analysis

A natural concern with structural alignment is that enforcing a shared geometry may suppress information from a more informative modality. We examine this in VFD and VFP by treating image embeddings as the information-rich modality $Z _ { R }$ and force embeddings as the information-poor modality $Z _ { P }$ . We freeze the trained Pairwise/UniOMA encoders and fit lightweight probes $\hat { f } _ { \phi }$ from $Z _ { R } , Z _ { P } $ , and $[ Z _ { R } , Z _ { P } ]$ to two targets Y: the next-step end-efector orientation (four-dimensional, ridge regression) and the binary contact label (logistic regression). For the orientation, we use the held-out Gaussian residual measure [7, 8]:

$$
\widehat { I } _ { \mathrm { p r o b e } } ( Z ; Y ) = \operatorname* { m a x } \left\{ 0 , \frac { 1 } { 2 } \log _ { 2 } \frac { \operatorname* { d e t } ( \widehat { \Sigma } _ { Y } ) } { \operatorname* { d e t } ( \widehat { \Sigma } _ { Y - \widehat { f } _ { \phi } ( Z ) } ) } \right\} ,\tag{3}
$$

where $\widehat { \Sigma } _ { Y }$ and $\widehat { \Sigma } _ { Y - \hat { f } _ { \phi } ( Z ) }$ denote the covariance matrices of the target and the prediction residual, respectively. This measure is a mutual-information lower bound only for a Gaussian target, so we read it as task information that a linear probe can decode rather than as $I ( Z ; Y )$

For contact, we report ${ \widehat { H } } ( Y ) - { \widehat { \mathrm { C E } } }$ on the test split, where ${ \widehat { H } } ( Y )$ is the empirical entropy of the binary contact label and $\begin{array} { r } { \widehat { \mathrm { C E } } = - \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \log _ { 2 } q _ { \phi } ( y _ { i } \mid z _ { i } ) } \end{array}$ is the cross-entropy of the logistic probe $q _ { \phi }$ on the n test samples. This quantity estimates the variational lower bound $I ( Z ; Y ) \ge H ( Y ) - \mathrm { C E } \left[ 7 \right]$ on held-out data, and it is at most $\widehat { H } ( Y ) = 0 . 9 5 5$ bits. Both quantities are instances of the bound $I ( Z ; Y ) \ge H ( Y )$ − CE with a decoder matched to the target: a Gaussian decoder for the continuous orientation, which gives $\operatorname { E q . }$ (3) when $H ( Y )$ is replaced by its Gaussian value, and a Bernoulli (logistic) decoder for the binary contact label.

Here, “rich information” refers to task-relevant predictive information in the embedding, rather than total information in the raw signal. Table IV shows that, compared with Pairwise, UniOMA increases the image-only and joint orientation measures by 1.6% on VFP and by 0.8–0.9% on VFD, and changes the contact measures by at most 1.5%. This suggests that alignment with the information-poor force modality does not degrade the information-rich image modality.

Table IV: Task-information retention on VFD and VFP. GW alignment does not reduce task information in the image features: compared with Pairwise, UniOMA increases orientation information in image features by 0.8–1.6% and changes the contact information of each feature set by at most 1.5%. We report the information (in bits) about the target that a probe recovers from frozen encoder features: the Gaussian residual measure (Eq. (3)) for the next-step end-efector orientation, and the empirica ${ \widehat { H } } ( Y ) - { \widehat { \mathrm { C E } } }$ for the binary contact label, which is at most $\widehat { H } ( Y ) = 0 . 9 5 5$ . We fit the probes on the training trajectories, regularize on validation, and evaluate once on the test split (mean ± std over 5 seeds); $\Delta$ is the relative change with respect to Pairwise. Gray columns denote our method.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Features</td><td colspan="3">Next-step end-effector orientation (bits)</td><td colspan="3">Contact (bits, at most 0.955)</td></tr><tr><td>Pairwise</td><td>Ours</td><td> $\Delta$ </td><td>Pairwise</td><td>Ours</td><td>∆</td></tr><tr><td rowspan="3">VFD</td><td>Image</td><td>11.60±0.04</td><td>11.69±0.04</td><td>+0.8%</td><td>0.549±0.009</td><td>0.541±0.003</td><td>-1.5%</td></tr><tr><td>Force</td><td>0.59±0.01</td><td>0.62±0.04</td><td>+5.1%</td><td>0.701±0.004</td><td>0.698±0.008</td><td>-0.4%</td></tr><tr><td>Image + force</td><td>11.71±0.03</td><td>11.81±0.05</td><td>+0.9%</td><td>0.718±0.004</td><td>0.715±0.008</td><td>-0.4%</td></tr><tr><td rowspan="3">VFP</td><td>Image</td><td>12.18±0.06</td><td>12.37±0.07</td><td>+1.6%</td><td>0.548±0.005</td><td>0.544±0.002</td><td>-0.7%</td></tr><tr><td>Force</td><td>0.59±0.04</td><td>0.62±0.04</td><td>+4.9%</td><td>0.701±0.005</td><td>0.699±0.006</td><td>-0.4%</td></tr><tr><td>Image + force</td><td>12.31±0.06</td><td>12.51±0.07</td><td>+1.6%</td><td>0.719±0.003</td><td>0.713±0.005</td><td>-0.8%</td></tr></table>

## H Choice of the Pre-trained Feature Extractor

The input-space kernel of the image modality is built from frozen ViT-B/16 features (Sec. I-B). To test whether the structural gains depend on this choice, we replaced ViT-B/16 by DINOv2 $\mathrm { V i T - B / 1 4 }$ [9] or an ImageNet-pretrained ResNet-18 [10], retrained our method on VFD and VFP (30 epochs, five seeds), and evaluated the structural measures of Sec. III-A against the same ViTbased input-kernel reference as in Table V of the main paper. Table V shows that the gains over Pairwise are preserved with both feature extractors, and ResNet-18 features give slightly higher rank correlation and lower GW distance than the default. The structural alignment therefore does not rely on a particular pre-trained encoder.

Table V: Structural alignment with diferent feature extractors for the image input-space kernel. All rows are evaluated against the ViT-based input-kernel reference of Table V of the main paper on the same 256 held-out samples (mean ± std over five seeds).
<table><tr><td>Method</td><td>Rank correlation ↑</td><td>kNN-10↑</td><td>GW dist. ↓  $( \times 1 0 ^ { - 2 } )$ </td><td>ARI↑</td></tr><tr><td colspan="5">VFD</td></tr><tr><td>Pairwise</td><td> $0 . 3 1 8 { \scriptstyle \pm 0 . 0 4 5 }$ </td><td> $0 . 2 0 5 { \scriptstyle \pm 0 . 0 0 6 }$ </td><td> $2 . 7 8 { \pm } 0 . 1 0 $ </td><td> $0 . 1 4 6 { \scriptstyle \pm 0 . 0 0 8 }$ </td></tr><tr><td>Ours, ViT-B/16 (default)</td><td> $0 . 4 8 1 { \scriptstyle \pm 0 . 0 4 2 }$ </td><td> $0 . 2 1 9 { \scriptstyle \pm 0 . 0 0 9 }$ </td><td> $2 . 5 9 { \pm } 0 . 0 4$ </td><td> $0 . 1 6 3 { \scriptstyle \pm 0 . 0 1 0 }$ </td></tr><tr><td> $\mathrm { O u r s , D I N O v 2 \ V i T \mathrm { - } B / 1 4 }$ </td><td> $0 . 4 8 0 { \scriptstyle \pm 0 . 0 2 8 }$ </td><td> $0 . 2 2 3 { \scriptstyle \pm 0 . 0 0 4 }$ </td><td> $2 . 5 2 { \pm } 0 . 0 5$ </td><td> $0 . 1 6 5 { \scriptstyle \pm 0 . 0 1 2 }$ </td></tr><tr><td>Ours, ResNet-18</td><td> $0 . 5 0 4 { \scriptstyle \pm 0 . 0 3 0 }$ </td><td> $0 . 2 2 4 { \scriptstyle \pm 0 . 0 0 5 }$ </td><td> $2 . 5 0 { \scriptstyle \pm 0 . 0 6 }$ </td><td> $0 . 1 5 7 { \scriptstyle \pm 0 . 0 2 1 }$ </td></tr><tr><td colspan="5">VFP</td></tr><tr><td>Pairwise</td><td> $0 . 3 9 2 { \scriptstyle \pm 0 . 0 3 5 }$ </td><td>0.246±0.006</td><td> $3 . 2 5 { \pm } 0 . 0 6$ </td><td> $0 . 1 5 0 { \scriptstyle \pm 0 . 0 1 1 }$ </td></tr><tr><td>Ours,  $\mathrm { V i T - B / 1 6 ~ ( d e f a u l t ) }$ </td><td> $0 . 5 7 5 { \scriptstyle \pm 0 . 0 2 0 }$ </td><td> $0 . 2 5 7 { \scriptstyle \pm 0 . 0 0 7 }$ </td><td> $2 . 7 0 { \scriptstyle \pm 0 . 0 5 }$ </td><td> $0 . 1 6 4 { \scriptstyle \pm 0 . 0 0 7 }$ </td></tr><tr><td>Ours,  $\mathrm { D I N O v 2 \ V i T { - } B / 1 4 }$ </td><td> $0 . 5 7 4 { \scriptstyle \pm 0 . 0 1 4 }$ </td><td> $0 . 2 5 5 { \scriptstyle \pm 0 . 0 0 2 }$ </td><td> $2 . 5 1 { \pm } 0 . 1 0 $ </td><td> $0 . 1 6 2 { \scriptstyle \pm 0 . 0 2 5 }$ </td></tr><tr><td>Ours, ResNet-18</td><td> $0 . 6 0 6 { \scriptstyle \pm 0 . 0 1 0 }$ </td><td> $0 . 2 5 9 { \scriptstyle \pm 0 . 0 0 5 }$ </td><td> $2 . 4 6 { \pm } 0 . 0 6$ </td><td> $0 . 1 5 3 { \scriptstyle \pm 0 . 0 1 3 }$ </td></tr></table>

Tactile input-space kernel. Tactile images difer substantially from natural RGB images, so we also replaced the ViT-B/16 features of the tactile modality on VAT, while keeping the audio and visual kernels unchanged, with two frozen encoders: an ImageNet-supervised ResNet-50 (global average-pooled 2048-dimensional features; 224×224 inputs with ImageNet normalization) and the tactile-specific encoder Sparsh [11] (sparsh-dino-base, a ViT-B on two stacked backgroundsubtracted tactile frames, mean of the patch tokens; since VAT touches are single images, the frame is repeated and the mean training touch image serves as the background). The tactile input-space kernel is an RBF kernel with the median heuristic on these features (Sec. I-C), and retrieval is evaluated as in Table I of the main paper. Table VI shows that retrieval remains above Pairwise in all three directions with both encoders; ResNet-50 features give the highest Vis→Tact and Tact→Aud MAP, and the tactile-specific Sparsh features do not improve over the generic encoders.

Table VI: Cross-modal retrieval on VAT with diferent feature extractors for the tactile input-space kernel (MAP, mean ± std). The Pairwise and ViT-B/16 rows are those of Table I of the main paper.
<table><tr><td>Method</td><td>Vis→Aud↑</td><td>Vis→Tact ↑</td><td>Tact→Aud↑</td></tr><tr><td>Pairwise</td><td>0.25±0.07</td><td>0.41±0.11</td><td>0.10±0.01</td></tr><tr><td>Ours, ViT-B/16 (default)</td><td>0.36±0.05</td><td>0.60±0.03</td><td>0.12±0.02</td></tr><tr><td>Ours, ResNet-50 (ImageNet)</td><td> $0 . 3 2 { \scriptstyle \pm 0 . 0 4 }$ </td><td>0.65±0.03</td><td>0.15±0.02</td></tr><tr><td>Ours, Sparsh</td><td> $0 . 3 1 { \pm } 0 . 0 6$ </td><td>0.56±0.10</td><td>0.14±0.04</td></tr></table>