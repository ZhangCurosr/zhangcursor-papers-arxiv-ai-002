# Uncertainty-Aware Probabilistic Constrained Clustering from Entangled Pairwise Supervision

Shaojie Zhang Ke Chen

Department of Computer Science, The University of Manchester, Manchester M13 9PL, U.K. {shaojie.zhang,ke.chen}@manchester.ac.uk

## Abstract

Pairwise constrained clustering typically relies on hard must-link/cannot-link labels, whereas realistic pairwise supervision may be real-valued and entangle intrinsic ambiguity, expert judgment, and stochastic corruption. Existing deep constrained clustering (DCC) methods mainly target hard, expert-agnostic constraints, treating soft labels mostly numerically rather than semantically. We formalize this setting as uncertainty-aware probabilistic constrained clustering (UPCC), defining a canonical aleatoric target through a heterogeneous observation process and analyzing its conditional identifiability. We introduce ProbPair, an angular pairwise objective for probabilistic relations, and build ECI-PP, an estimator–corrector–integrator framework that refines imperfect supervision via belief estimation, correction, and reliability-aware integration. Across challenging probabilistic supervision settings, experiments on diverse benchmarks show that ECI-PP outperforms state-of-the-art DCC methods and remains robust with a shared default configuration.

## 1 Introduction

Clustering is a fundamental tool for uncovering structure in data, yet its unsupervised nature often yields partitions that may not align with domain knowledge [1, 2, 3]; pairwise constrained clustering (CC) addresses this by using weak instance-level pairwise supervision [4, 5]. Traditional CC methods [6, 7, 8, 9, 10, 11], however, often struggle with high-dimensional and complex data, motivating deep constrained clustering (DCC). Existing DCC methods can be broadly organized into two paradigms: end-to-end DCC [12, 13, 14, 15], which reformulates clustering as a pseudo-classification problem with anchors, and deep constraint embedding [16, 17], which instead learns clustering-friendly representations from pairwise supervision and, in more recent advances, further moves pairwise learning from Euclidean to angular space to better reconcile positive and negative relations [3].

Despite these advances, the supervision setting remains largely idealized. Existing work typically assumes hard binary pairwise relations, whereas supervision in practice often lies on a continuum. Moreover, such real-valued pairwise labels may simultaneously reflect intrinsic aleatoric ambiguity, expert-conditioned and pair-dependent epistemic judgment [18], and additional stochastic corruption from annotation or recording pipelines. For example, when comparing two clinical cases, a recorded observation may reflect ambiguity in the cases themselves, systematic clinician judgment bias, and documentation or data-entry errors. These considerations motivate what we term uncertaintyaware probabilistic constrained clustering (UPCC): a more specific and realistic setting in which probabilistic pairwise relations arise from a heterogeneous observation process.

In this paper, we formalize the UPCC setting and develop a corresponding solution. We specify an observation model that distinguishes the aleatoric target from expert-conditioned judgment and stochastic corruption, and analyze when this canonical target is identifiable on observable pairs. To learn under UPCC, we first introduce ProbPair, a deep constraint embedding approach linking probabilistic pairwise relations to angular representation learning. Built on it, we further develop a

Estimator–Corrector–Integrator ProbPair (ECI-PP) framework, in which Estimators provide model beliefs, Correctors perform structured correction and suppress residual discordance, and the Integrator learns from the resulting surrogate supervision toward the underlying aleatoric relation.

Our main contributions are: (i) We formalize UPCC through a heterogeneous observation model and analyze identifiability of the canonical aleatoric target on observable pairs. (ii) We introduce ProbPair, a deep constraint embedding approach linking probabilistic pairwise relations to angular embeddings. (iii) We develop ECI-PP, a practical framework for aleatoric relation learning under entangled supervision. (iv) We empirically show across diverse benchmarks that ECI-PP outperforms state-of-the-art DCC methods and is robust under fallible, heterogeneous, and corrupted supervision.

## 2 Related work

Beyond binary pairwise constraints. Existing CC methods either extend binary constraints only numerically, without semantically modeling intermediate supervision, or assign intermediate values to the confidence of a pre-defined relation type rather than a unified probability. Advanced deep methods typically employ logistic losses [19, 3] compatible with real-valued constraints in [0, 1]. However, end-to-end DCC methods [12, 14, 20, 21, 22, 15] remain tied to hard anchor semantics, with attendant anchor misalignment [23] and error propagation [24, 25], while deep constraint embedding methods [3, 17, 16] avoid anchors but still require a binary regime for their theoretical geometric guarantees. Likewise, the deep generative approach [2] encodes constraints as a signed prior over discrete assignments, precluding a unified treatment of real-valued relations. Earlier non-deep approaches considered intermediate supervision more explicitly: (i) probabilistic treatments [26, 27] encoded the activation probability of a positive same-cluster tie, but offered neither negative relations nor a unified notion of pairwise affinity; and (ii) fuzzy-constraint formulations attach an intermediate belief [28], degree [29], or penalty [30, 31, 32, 33] only after a discrete relation type has been specified. Taken together, these lines of work do not address uncertain relational judgment expressed by probabilistic labels under a unified semantics. By contrast, our ProbPair formulation, built on deep constraint embedding, directly models such unified probabilistic supervision, accommodating both vague annotations in practice and inherently ambiguous aleatoric relations.

Imperfect pairwise supervision. While imperfect annotations in CC have long been recognized [34, 35, 36, 37, 38, 39, 1, 2, 15], more realistic observation settings remain underexplored. Existing studies treat simplified noise through constraint-set robustness [34, 36], coarse prior uncertainty over assignments [2], or expert-level reliability in semi-crowdsourced settings [38, 1]. VolMaxDCC [15] further introduces confusion-based structured annotation modeling beyond unstructured labe degradation or flipping. However, it still lacks heterogeneous multi-expert modeling and remains restricted to discrete binary-noise settings, leaving expert-specific, pair-dependent epistemic distortion unmodeled. By contrast, we consider genuinely entangled observations, arising from intrinsic aleatoric relations, expert-conditioned epistemic judgments, and stochastic corruption. Such a setting is reminiscent of learningfrom crowds, where expert behavior is commonly modeled under a fixed class-label scheme through expert-specific confusion or output layers [40, 41], or through expertise coupled with instance difficulty [42, 43]. Yet these strategies rely on a restricted class-label channel and are therefore not directly transferable to deep constraint embedding, where a highly flexible pairwise learner can readily absorb structured distortion and random corruption into its learned geometry. We therefore develop ECI-PP, a multi-role iterative framework built on ProbPair that uses model-side beliefs to correct structured distortion and filter residual inconsistency, yielding refined surrogate supervision that progressively guides learning toward the underlying aleatoric relation.

## 3 UPCC formulation and canonical target

We study uncertainty-aware probabilistic constrained clustering (UPCC), where expert-indexed realvalued pairwise labels arise from a heterogeneous observation process. Let $\mathcal { X } = \{ x _ { j } \} _ { j = 1 } ^ { | \mathcal { X } | }$ be a dataset to be partitioned into C clusters, and let $\mathcal { C } = \{ ( a _ { i } , b _ { i } , e _ { i } , y _ { i } ) \} _ { i = 1 } ^ { | \mathcal { C } | }$ denote the observed soft constraints, where $( a _ { i } , b _ { i } )$ indexes the pair $( x _ { a _ { i } } , x _ { b _ { i } } ) , e _ { i } \in \{ 1 , \ldots , E \}$ records the expert identity, and $y _ { i } \in [ 0 , 1 ]$ is the observed pairwise relation. The formulation targets a canonical pairwise probabilistic relation, while a final hard partition is treated as a downstream summary induced by that relation.

## 3.1 Observation model

We view each observed pairwise relation as arising from three factors: (i) an underlying aleatoric relation in the data $( \mathrm { e . g . }$ ., intrinsic clinical similarities between a common cold and the flu), (ii) an expert-conditioned epistemic judgment built on that relation (e.g., a novice doctor systematically overrating their similarity due to limited knowledge), and (iii) a subsequent stochastic corruption stage accounting for accidental recording noise (e.g., random clerical errors in medical records).

Aleatoric relation. Let $s$ be the set of all C-partitions of $x ,$ , and let $S \in S$ be a latent random partition. For any index pair $( a , b )$ , define the aleatoric co-membership relation

$$
R _ { a b } ^ { \star } : = \operatorname* { P r } \left( x _ { a } { \mathrm { ~ a n d ~ } } x _ { b } { \mathrm { ~ b e l o n g ~ t o ~ t h e ~ s a m e ~ c l u s t e r ~ u n d e r ~ } } S \mid { \mathcal X } \right) \in [ 0 , 1 ] .
$$

This quantity serves as the canonical target of the formulation and remains well defined even when the latent partition is intrinsically ambiguous.

Epistemic judgment. Let $\sigma ( \cdot )$ denote the sigmoid function, and define its clamped inverse logit $\mathbf { \Lambda } _ { \varepsilon } ( u ) : = \mathrm { l o g i t } \big ( \operatorname* { m i n } \{ \operatorname* { m a x } \{ u , \varepsilon \} , 1 - \varepsilon \} \big )$ with $\varepsilon \ll 1$ . We also introduce a deterministic pair descriptor $\phi _ { a b } = \phi ( x _ { a } , x _ { b } )$ , which serves as the covariate through which expert-dependent effects are parameterized. For expert $e ,$ let $m _ { e } ( \phi _ { a b } ) \in \mathbb { R }$ denote a structured mean effect, and let

$$
u _ { e , a b } \mid e , \phi _ { a b } \sim \mathcal { N } \big ( 0 , \tau _ { e } ^ { 2 } ( \phi _ { a b } ) \big ) , \qquad \tau _ { e } ( \phi _ { a b } ) > 0 ,
$$

be a centered residual perturbation. The latent expert judgment is generated as

$$
y _ { e , a b } ^ { \mathrm { j u d } } : = \sigma \big ( \log \mathrm { i t } _ { \varepsilon } ( R _ { a b } ^ { \star } ) + m _ { e } ( \phi _ { a b } ) + u _ { e , a b } \big ) \in ( 0 , 1 ) .\tag{1}
$$

Under this construction, the judgment channel has the logistic-normal density

$$
p _ { \mathrm { j u d } } \bigl ( y \mid R _ { a b } ^ { \star } , e , \phi _ { a b } \bigr ) = \frac { 1 } { y ( 1 - y ) } \varphi \Bigl ( \log \mathrm { i t } ( y ) ; \log \mathrm { i t } _ { \varepsilon } ( R _ { a b } ^ { \star } ) + m _ { e } ( \phi _ { a b } ) , \tau _ { e } ^ { 2 } ( \phi _ { a b } ) \Bigr )\tag{2}
$$

for $y \in ( 0 , 1 )$ , where $\varphi ( \cdot ; \mu , s ^ { 2 } )$ denotes the Gaussian density; see Appendix A.1 for the derivation. The same judgment channel also gives rise to the corresponding mean distortion in probability space,

$$
\bar { \epsilon } _ { e , a b } ( \boldsymbol { R } _ { a b } ^ { \star } , \phi _ { a b } ) : = \mathbb { E } \Big [ y _ { e , a b } ^ { \mathrm { j u d } } - R _ { a b } ^ { \star } \mid { R } _ { a b } ^ { \star } , e , \phi _ { a b } \Big ] .\tag{3}
$$

Stochastic corruption. To capture unstructured recording noise beyond the latent judgment, we introduce a corruption indicator $c _ { e , a b } \sim$ Bernoulli(π ), where $\pi _ { e } \in [ 0 , 1 )$ is the corruption probability for expert e, and define the final observed relation by

$$
y _ { e , a b } \mid c _ { e , a b } = 0 \sim p _ { \mathrm { j u d } } ( \cdot \mid R _ { a b } ^ { \star } , e , \phi _ { a b } ) , \qquad y _ { e , a b } \mid c _ { e , a b } = 1 \sim \mathrm { U n i f o r m } ( 0 , 1 ) .
$$

Equivalently, letting $\mathbf { 1 } _ { [ 0 , 1 ] } ( \cdot )$ denote the uniform density on $[ 0 , 1 ]$ , the conditional observation law is

$$
p \big ( \boldsymbol { y } \mid { \cal R } _ { a b } ^ { \star } , e , \phi _ { a b } \big ) = ( 1 - \pi _ { e } ) p _ { \mathrm { j u d } } \big ( \boldsymbol { y } \mid { \cal R } _ { a b } ^ { \star } , e , \phi _ { a b } \big ) + \pi _ { e } \mathbf { 1 } _ { [ 0 , 1 ] } ( \boldsymbol { y } ) .\tag{4}
$$

## 3.2 Canonical target and identifiability

We now formalize the canonical target within a structural setup for identifiability. To obtain a finite-dimensional canonical estimand, we parameterize

$$
R _ { a b } ^ { \star } = r _ { \theta } ( x _ { a } , x _ { b } ) , \qquad m _ { e } ( \phi _ { a b } ) = m _ { \eta _ { e } } ( \phi _ { a b } ) , \qquad \tau _ { e } ( \phi _ { a b } ) = \tau _ { \zeta _ { e } } ( \phi _ { a b } ) ,
$$

and collect the parameters into $\Xi = ( \theta , \{ \eta _ { e } \} _ { e = 1 } ^ { E } , \{ \zeta _ { e } \} _ { e = 1 } ^ { E } , \{ \pi _ { e } \} _ { e = 1 } ^ { E } )$ . This parameterization makes the canonical target explicit under the specific observation family. Since $\phi _ { a b }$ is introduced only as a deterministic pair descriptor without additional representational constraints, the identification of the canonical relation term $r _ { \theta } ( x _ { a } , x _ { b } )$ is governed by the structural conditions introduced below.

We first fix the trivial additive ambiguity in the location decomposition:

Assumption 3.1 (Centering on observable support). For each expert e with nonzero observation probability, $\mathbb { E } _ { ( a , b ) | e } [ m _ { \eta _ { e } } ( \phi _ { a b } ) ] = 0$ , where the expectation is taken over the observable-pair distribution conditional on expert e.

This centering removes any remaining expert-specific constant shift on the observable support: Lemma 3.2 (Gauge fixing). Suppose both $\{ m _ { \eta _ { e } } \} _ { e = 1 } ^ { E }$ and $\{ m _ { \eta _ { e } ^ { \prime } } \} _ { e = 1 } ^ { E }$ satisfy Assumption $3 . l . \ U f$ for each expert e with nonzero observation probability there exists a constant $k _ { e } \in \mathbb { R }$ such that $m _ { \eta _ { e } } ( \phi _ { a b } ) = m _ { \eta _ { e } ^ { \prime } } ( \phi _ { a b } ) - k _ { e }$ for all observable triples $( a , b , e )$ , then $k _ { e } = 0 f o r$ every such expert e.

The proof is given in Appendix A.2. The logistic-normal clean channel in Eq. (2), combined with the observation law in Eq. (4), yields the following injectivity property:

Lemma 3.3 (Injectivity of the observation family). Let $\mu : = \mathrm { l o g i t } _ { \varepsilon } ( R _ { a b } ^ { \star } ) + m _ { e } ( \phi _ { a b } ) , s : = \tau _ { e } ( \phi _ { a b } )$ and $\pi : = \pi _ { e } .$ For the observation law rewritten as

$$
p ( y \mid \mu , s , \pi ) : = ( 1 - \pi ) { \frac { 1 } { y ( 1 - y ) } } \varphi { \big ( } \log \mathrm { i t } ( y ) ; \mu , s ^ { 2 } { \big ) } + \pi \mathbf { 1 } _ { [ 0 , 1 ] } ( y ) , \qquad y \in ( 0 , 1 ) ,\tag{5}
$$

the mapping $( \mu , s , \pi ) \mapsto p ( \cdot \mid \mu , s , \pi )$ is injective on $\mathbb { R } \times ( 0 , \infty ) \times [ 0 , 1 )$

The proof is given in Appendix A.3. The canonical formulation is completed by the following structural separability condition on the location term log $\mathrm { i t } _ { \varepsilon } \big ( r _ { \theta } ( x _ { a } , x _ { b } ) \big ) + \bar { m } _ { \eta _ { e } } ( \phi _ { a b } \dot { ) }$

Assumption 3.4 (Structural separability). Under Assumption 3.1, if for all observable triples $( a , b , e )$

$$
\mathrm { l o g i t } _ { \varepsilon } \bigl ( r _ { \theta } ( x _ { a } , x _ { b } ) \bigr ) + m _ { \eta _ { e } } ( \phi _ { a b } ) = \mathrm { l o g i t } _ { \varepsilon } \bigl ( r _ { \theta ^ { \prime } } ( x _ { a } , x _ { b } ) \bigr ) + m _ { \eta _ { e } ^ { \prime } } ( \phi _ { a b } ) ,
$$

then for each expert e there exists a constant $k _ { e } \in$ R such that for all such observable triples,

$$
\mathrm { l o g i t } _ { \varepsilon } \big ( r _ { \theta } ( x _ { a } , x _ { b } ) \big ) = \mathrm { l o g i t } _ { \varepsilon } \big ( r _ { \theta ^ { \prime } } ( x _ { a } , x _ { b } ) \big ) + k _ { e } , \qquad m _ { \eta _ { e } } ( \phi _ { a b } ) = m _ { \eta _ { e } ^ { \prime } } ( \phi _ { a b } ) - k _ { e } .
$$

Assumption 3.4 provides the key structural requirement for the identifiability argument; boundary cases clarifying this condition are discussed in Appendix ${ \bf A . 4 } .$ . Writing the support of observable pairs as $\mathcal { P } _ { \mathrm { o b s } } : = \{ ( a , b )$ : e such that $( a , b , e )$ is observable , we then have the identifiability result: Theorem 3.5 (Conditional canonical identifiability of the aleatoric relation). $L e t \Xi a n d \Xi ^ { \prime }$ satisfy the observation model and Assumption 3.1 and $3 . 4 .$ Assumefurther that ${ r _ { \theta } } ( x _ { a } , x _ { b } ) , \ r _ { \theta ^ { \prime } } ( x _ { a } , x _ { b } ) \in$ $[ \varepsilon , 1 - \varepsilon ] f o r a l l \left( a , b \right) \in \mathcal { P } _ { \mathrm { o b s } } .$ Ifthe induced conditional laws ofthe observed response coincide under Ξ and $\dot { \Xi } ^ { \prime } f o r$ every observable triple $( a , b , e )$ , then $r _ { \theta } ( x _ { a } , x _ { b } ) = r _ { \theta ^ { \prime } } ( x _ { a } , x _ { b } )$ for all $( a , b ) \in \mathcal { P } _ { \mathrm { o b s } }$ Hence the canonical aleatoric relation $R _ { a b } ^ { \star } = r _ { \theta } ( x _ { a } , x _ { b } )$ is identifiable on observable pairs.

The proof is in Appendix A.5. Under the specific observation family, Theorem 3.5 gives conditional identifiability of the canonical aleatoric relation on observable pairs at the law level. In the practical single-observation regime, however, estimation still relies on the shared parametric structure imposed across pairs and experts rather than repeated observations of the same pair–expert combination.

## 4 Method

We first introduce a probability-aware pairwise learning formulation that links probabilistic pairwise relations to clustering-oriented representations. Building on this formulation and motivated by the theoretical observation model, we develop a practical learning framework in which a surrogate supervision route guides learning from entangled supervision toward the aleatoric relation.

## 4.1 ProbPair: probability-aware pairwise representation learning

We establish a representation learning formulation under probabilistic pairwise supervision. Given supervised pairs $\{ ( a _ { i } , b _ { i } , y _ { i } ) \} _ { i = 1 } ^ { | { \mathcal { C } } | }$ , where $y _ { i } \in [ 0 , 1 ]$ denotes the probabilistic relation for pair $( x _ { a _ { i } } , x _ { b _ { i } } )$ we aim to learn clustering-oriented representations. To this end, we adopt a deep constraint embedding formulation with encoder $f _ { \psi } : \mathcal { X } \xrightarrow { } \mathcal { Z } \subset \mathbb { R } ^ { D }$ and latent code $z _ { j } = f _ { \psi } ( x _ { j } )$ . We work in angular space, where bounded cosine similarity provides a natural geometric quantity for probabilistic calibration. To connect probabilistic pairwise relations with the latent representation space, we introduce the following probabilistic readout:

$$
\hat { y } _ { a _ { i } b _ { i } } : = \sigma \big ( ( \cos ( z _ { a _ { i } } , z _ { b _ { i } } ) - m ) / T \big ) \in ( 0 , 1 ) ,\tag{6}
$$

where $m \in \mathbb { R }$ and $T > 0$ are learnable parameters controlling the operating point and sharpness of the mapping. The resulting pairwise objective, referred to as the ProbPair loss, is given by

$$
\mathcal { L } _ { \mathrm { P P } } = - \frac { 1 } { | \mathcal { C } | } \sum _ { i = 1 } ^ { | \mathcal { C } | } \left[ y _ { i } \log \hat { y } _ { a _ { i } b _ { i } } + ( 1 - y _ { i } ) \log ( 1 - \hat { y } _ { a _ { i } b _ { i } } ) \right] .\tag{7}
$$

![](images/d2753084cfeab9f42f565e953e8774281a177b5a47064c5bb42e898b35170924.jpg)  
Figure 1: Overview of the ECI-PP pipeline. The Estimators are trained with the ProbPair objective (Eq. (8)) and produce out-of-fold beliefs ${ \hat { y } } _ { i } ^ { \mathrm { o o f } }$ via the readout in Eq. (6). The Correctors take $\left[ \hat { y } _ { i } ^ { \mathrm { o o f } } ; \phi _ { i } \right]$ as input, with $\phi _ { i }$ the pair-i descriptor, and learn probability-space corrections $\hat { \Delta } _ { i }$ via Eq. (10), yielding corrected relations $y _ { i } ^ { \mathrm { { \bar { c o r } } } }$ . Reliability-aware screening produces reliability weights $w _ { i } ,$ and BC fusion forms the refined supervision $y _ { i } ^ { \mathrm { B C } }$ via Eqs. (11) and (12). The Integrator learns embeddings from the refined supervision; its Integrator-side relations $y _ { i } ^ { \mathrm { i n t } }$ are fed back to the Estimators for iterative refinement, while the final embeddings ${ \mathcal { Z } } ^ { \mathrm { i n t } }$ are clustered into the final partition.

Unlike hard-constraint objectives, Eq. (7) directly accommodates non-binary relation strengths. As a result, the induced geometry need not collapse to a purely binary regime, but can preserve both confident and uncertain pairwise structure.

We further include a reconstruction regularizer to avoid degenerate representations and preserve the global structure of the data. Since the representation is learned in angular space, reconstruction is performed from normalized latent embeddings, as in [3]. Specifically, with decoder $g _ { \psi ^ { \prime } } : \mathcal { Z } \to \mathcal { X }$ and Norm( ) denoting $\ell _ { 2 }$ normalization, we decode from $\hat { x } _ { j } = g _ { \psi ^ { \prime } } ( \mathrm { N o r m } ( z _ { j } ) )$ ) and use reconstruction loss $\begin{array} { r } { \mathcal { L } _ { \mathrm { r e c } } = \frac { 1 } { | \mathcal { X } | } \sum _ { j = 1 } ^ { | \mathcal { X } | } \| x _ { j } - \hat { x } _ { j } \| _ { 2 } ^ { 2 } } \end{array}$ . With $\lambda _ { \mathrm { r e c } } > 0$ balancing pairwise relation fitting and reconstruction regularization, the overall objective is

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { P P } } + \lambda _ { \mathrm { r e c } } \mathcal { L } _ { \mathrm { r e c } } . } \end{array}\tag{8}
$$

Minimizing Eq. (8) yields optimal normalized embeddings $\mathcal { Z } _ { \mathrm { n o r m } } = \{ \mathrm { N o r m } ( z _ { j } ) \} _ { j = 1 } ^ { | \mathcal { X } | }$ that are shaped by probabilistic pairwise relations while being regularized by reconstruction. These angular embeddings can then be used by a downstream clustering algorithm to obtain the final partition, and can also produce probabilistic relation estimates for instance pairs through Eq. (6).

## 4.2 Estimator–Corrector–Integrator ProbPair

Building on the above formulation, we develop a practical framework motivated by the observation model. Rather than learning directly from entangled supervision, it progressively refines the supervision signal by correcting the structured observational component, screening the residual inconsistency, and feeding the refined supervision back into learning, thereby yielding an iterative Estimator–Corrector–Integrator ProbPair (ECI-PP) design for learning toward the aleatoric relation. The overall ECI-PP flow is summarized in Fig. 1.

Estimator. To enable the subsequent learning of structured observational effects from entangled supervision, ECI-PP first constructs a model-side reference that is less coupled to the raw observations. In flexible pairwise representation learning, beliefs fitted on the same constraints used for training can absorb noisy observations in-sample and are therefore unreliable for this role. We accordingly partition into K disjoint folds $\{ \mathcal { C } ^ { ( k ) } \} _ { k = 1 } ^ { K }$ and instantiate $K$ non-shared Estimators $\{ \mathrm { E s t } ^ { ( k ) } \} _ { k = 1 } ^ { K }$ where $\operatorname { E s t } ^ { ( k ) } { = } ( f _ { \omega _ { k } } , g _ { \omega _ { k } ^ { \prime } } , m _ { k } , T _ { k } )$ denotes the encoder, decoder, and fold-specific readout parameters. Each $\mathrm { E s t } ^ { ( k ) }$ is trained on ${ \mathcal { C } } \setminus { \mathcal { C } } ^ { ( k ) }$ by minimizing in Eq. (8), and then produces holdout beliefs on $\mathcal { C } ^ { ( k ) }$ through Eq. (6). Aggregating these predictions over all folds yields a full set of out-of-fold beliefs $\{ \hat { y } _ { i } ^ { \mathrm { o o f } } \} _ { i = 1 } ^ { | \mathcal { C } | }$ , which serves as the model-side reference. To initialize the estimator, we weight each observed pair in ProbPair loss by an information-based decisiveness score, defined as

$$
\kappa _ { i } = { \frac { D _ { \mathrm { K L } } { \big ( } \mathrm { B e r n } ( y _ { i } ) \| \mathrm { B e r n } ( { \bar { y } } ) { \big ) } } { ( 1 - y _ { i } ) D _ { \mathrm { K L } } { \big ( } \mathrm { B e r n } ( 0 ) \| \mathrm { B e r n } ( { \bar { y } } ) { \big ) } + y _ { i } D _ { \mathrm { K L } } { \big ( } \mathrm { B e r n } ( 1 ) \| \mathrm { B e r n } ( { \bar { y } } ) { \big ) } } } \in [ 0 , 1 ] ,\tag{9}
$$

where $\begin{array} { r } { \bar { y } = \frac { 1 } { | \mathcal { C } | } \sum _ { i = 1 } ^ { | \mathcal { C } | } y _ { i } , D _ { \mathrm { K L } } ( \cdot \| \cdot ) } \end{array}$ denotes the Kullback–Leibler divergence, and $\mathrm { B e r n } ( \cdot )$ denotes the Bernoulli law. This weighting emphasizes more decisive judgments, which are typically expressed with greater expert confidence, and thereby stabilizes the initial out-of-fold beliefs.

Corrector. Given the Estimators’ out-of-fold beliefs, the Corrector aims to learn structured expertspecific corrections in the observed supervision. These corrections are motivated by the epistemic component defined in logit space (Section 3.1), whereas for practical stability we work with its mean probability-space distortion $\bar { \epsilon } _ { e , a b } ( R _ { a b } ^ { \star } , \phi _ { a b } )$ (see Eq. (3)), because small probability discrepancies near the $0 { \dot { / } } 1$ boundaries induce disproportionately large logit residuals. Although approximate and not preserving the variance structure encoded by $\tau _ { e } ( \phi _ { a b } ) , \bar { \epsilon } _ { e , a b } ( R _ { a b } ^ { \star } , \phi _ { a b } )$ remains a reasonable proxy because it captures the structured mean effect induced by the epistemic channel, making the resulting Corrector a practical surrogate module.

We take ${ \hat { y } } _ { i } ^ { \mathrm { o o f } }$ as the current proxy for $R _ { a _ { i } b _ { i } } ^ { \star }$ and introduce expert-specific Correctors $\{ \mathrm { C o r } _ { e } \} _ { e = 1 } ^ { E }$ in probability space, where each $\mathrm { C o r } _ { e }$ is parameterized by a network $q _ { \nu _ { e } }$ on ${ \mathcal { T } } _ { e } : = \{ i : e _ { i } = e \}$ with input $[ \hat { y } _ { i } ^ { \mathrm { o o f } } ; \phi _ { i } ] ;$ here we instantiate $\phi _ { i } = \left[ x _ { a _ { i } } \odot x _ { b _ { i } } ; ~ \lvert x _ { a _ { i } } - x _ { b _ { i } } \rvert \right]$ with  denoting element-wise product, as a simple and stable generic pair feature. Writing $\Delta _ { i } : = \hat { y } _ { i } ^ { \mathrm { o o f } } - y _ { i } \in ( - 1 , 1 )$ , the Corrector output is $\hat { \Delta } _ { i } : = q _ { \nu _ { e _ { i } } } \left( \left[ \hat { y } _ { i } ^ { \mathrm { o o f } } ; \phi _ { i } \right] \right) \in \left( - 1 , 1 \right)$ . Since $\Delta _ { i }$ is an empirical target formed from the observed $y _ { i }$ and the proxy ${ \hat { y } } _ { i } ^ { \mathrm { o o f } }$ , it may also contain stochastic corruption and proxy error in addition to structured epistemic effects. We therefore train each small expert-specific Corrector with regularization:

$$
\mathcal { L } _ { \mathrm { c o r } } ^ { ( e ) } = \frac { 1 } { | \mathcal { T } _ { e } | } \sum _ { i \in \mathcal { T } _ { e } } \left[ \rho ( \hat { \Delta } _ { i } - \Delta _ { i } ) + \lambda _ { \mathrm { c o r } } | \hat { \Delta } _ { i } | ^ { 2 } \right] .\tag{10}
$$

Here, $\rho ( \cdot )$ is the Huber loss and $\lambda _ { \mathrm { c o r } } > 0$ weights the $\ell _ { 2 }$ regularizer, which biases the limited-capacity Corrector toward shared learnable structure rather than arbitrary pair-specific fluctuations.

Integrator. The Integrator, parameterized as Int ${ \bf \Omega } = ( f _ { \omega } , g _ { \omega ^ { \prime } } , m , T )$ , is the learner that integrates the refined supervision into a clustering-oriented embedding. To build this refined supervision from the Corrector output, we first form a corrected relation $y _ { i } ^ { \mathrm { c o r } }$ , quantify the remaining inconsistency by ${ \mathrm { g a p } } _ { i } ,$ and convert it into a screening weight $w _ { i }$ , with $\gamma > 0$ controlling the sharpness of down-weighting:

$$
y _ { i } ^ { \mathrm { { c o r } } } : = \mathrm { s o f t c l i p } _ { \xi } ( y _ { i } + \hat { \Delta } _ { i } ) , \qquad \mathrm { { g a p } } _ { i } : = \left| { \mathrm { s o f t c l i p } } _ { \xi } ( \hat { y } _ { i } ^ { \mathrm { { o o f } } } ) - y _ { i } ^ { \mathrm { c o r } } \right| , \qquad w _ { i } : = ( 1 - \mathrm { g a p } _ { i } ) ^ { \gamma _ { 1 } } .\tag{11}
$$

Here, softclip $\begin{array} { r } { \mathbf { \nabla } \cdot \mathbf { \mu } ( u ) : = { \xi } ^ { - 1 } } \end{array}$ softplus $( \xi u ) - \xi ^ { - 1 }$ softplus $( \xi ( u - 1 ) )$ with $\xi > 0$ softly maps values into (0, 1) while preserving gradient flow near the boundaries; applying the same map to ${ \hat { y } } _ { i } ^ { \mathrm { o o f } }$ keeps the gap computation in the same bounded space. The residual ${ \mathrm { g a p } } _ { i }$ quantifies the unexplained discrepancy that remains after structured correction. While the observation model associates this discrepancy with stochastic corruption and residual epistemic variation, gap may also reflect proxy error in ${ \hat { y } } _ { i } ^ { \mathrm { o o f } }$ and imperfect Corrector fitting in practice. We therefore treat it uniformly as a reliability signal for heuristic screening through $w _ { i }$ . We then set $n _ { i } : = n _ { 0 } w _ { i }$ with $n _ { 0 } > 0$ , and construct the Bayesian-confidence supervision

$$
y _ { i } ^ { \mathrm { B C } } : = ( y _ { i } + n _ { i } y _ { i } ^ { \mathrm { c o r } } ) / ( 1 + n _ { i } ) ,\tag{12}
$$

which keeps $y _ { i }$ as a fixed prior while letting $y _ { i } ^ { \mathrm { c o r } }$ contribute according to its screened reliability. The Integrator then optimizes $\mathcal { L } \left( \mathrm { E q . } \left( 8 \right) \right)$ using $\bar { y } _ { i } ^ { \mathrm { B C } }$ as the pairwise target, with $w _ { i }$ multiplying each per-constraint term in <sub>PP</sub> (Eq. (7)) before averaging, yielding Integrator embeddings.

Iterative refinement and deployment. ECI-PP operates by passing supervision through the Estimator–Corrector–Integrator pipeline to produce, via Eq. (6), Integrator-side relations $y _ { i } ^ { \mathrm { i n t } }$ <sup>t</sup>, and then feeding $( y _ { i } ^ { \mathrm { i n t } } , w _ { i } )$ back to the Estimators for iterative refinement. Warm-up initializes this loop by starting the Estimators from $( y _ { i } , \kappa _ { i } )$ ; subsequent rounds replace this with $( y _ { i } ^ { \mathrm { i n t } } , w _ { i } )$ . At each round, the refreshed Estimator beliefs are passed again through the Corrector–Integrator pipeline to update $y _ { i } ^ { \mathrm { c o r } } , \mathrm { g a p } _ { i } , w _ { i }$ , and $y _ { i } ^ { \mathrm { B C } }$ , thereby yielding a practical finite-stage refinement procedure that progressively improves the surrogate supervision toward the aleatoric relation. After refinement, the final normalized Integrator embeddings $\mathcal { Z } ^ { \mathrm { i n t } } : = \{ \mathrm { N o r m } ( f _ { \omega } ( x _ { j } ) ) \} _ { j = 1 } ^ { | \mathcal { X } | }$ encode the learned pairwise relation structure; applying an unsupervised clustering algorithm to ${ \mathcal { Z } } ^ { \mathrm { i n t } }$ then yields the final partition. Appendix B provides algorithmic details and complexity analysis.

## 5 Experiments

## 5.1 Experimental settings

Our experiments evaluate ECI-PP under UPCC along four axes: (i) comparison with DCC and probabilistic pairwise baselines, (ii) robustness to expert quality, corruption, and multi-expert variation, (iii) held-out diagnostics of internal dynamics, and (iv) sensitivity and ablation of key designs.

Datasets. We adopt six image benchmarks (CIFAR100-20, CIFAR10 [44], FMNIST [45], ImageNet10 [46], MNIST [47], STL10 [48]) and two text benchmarks (Reuters subset [49], RCV1-10 [3]), covering varying class counts and class imbalance. Dataset details and splits are in Appendix C.1.

Compared methods. We compare ECI-PP with six baselines applicable to probabilistic supervision: VanillaDCC [19] as a basic logistic pairwise baseline, VolMaxDCC [15] with explicit noise modeling, CIDEC [14] and SpherePair [3] as state-of-the-art end-to-end DCC and deep constraint embedding methods, respectively, and ProbPair/Weighted ProbPair as direct ablations. ProbPair uses only in Eq. (8), while Weighted ProbPair additionally uses the information-based weight in Eq. (9).

Constraint generation. To instantiate UPCC, we generate constraints with trained expert classifiers rather than deriving binary labels from ground-truth classes. Experts are trained on controlled labeled subsets; uniformly sampled constraint pairs assigned to expert e are given a judgment $y _ { e , a b } ^ { \mathrm { j u d } } =$ $\langle p _ { a } ^ { e } , { p } _ { b } ^ { e } \rangle \in [ 0 , 1 ]$ from the predictive distributions, and the recorded $y _ { i }$ is then produced through the corruption channel in Section 3.1. For single-expert supervision, settings lv0.1/lv0.01/lv0.001 vary the labeled-data fraction used to train the expert. Multi-expert settings multi2/multi3/multi10 use multiple experts with complementary familiar/unfamiliar categories, where unfamiliar categories act as expert blind spots. See Appendix C.3 for the full procedure.

Protocol. For a comparative study, we report clustering metrics (ACC/NMI/ARI) over five trials; embedding-based methods use K-means on learned representations, while end-to-end DCC baselines use native outputs. The main comparison uses the default lv0.01 single-expert and multi3 multiexpert regimes, both with corruption probability 0.3 and 9k constraints in total (3k per expert under multi3). To assess robustness, we vary single-expert quality, corruption probability, and multi-expert configuration with matched constraint budgets. To probe ECI-PP beyond final clustering scores, we conduct held-out internal-signal diagnostics and sensitivity/ablation studies.

Implementation. For fair comparison, we follow established fully connected architectures and the unsupervised pretraining protocol in [3, 15]: encoder-based methods use a 500–500–2000 backbone with embedding dimension 10 except 20 for CIFAR100-20, while VanillaDCC and VolMaxDCC use their native 512–512 classifier architectures. For baselines, we use reported best settings where available; VolMaxDCC follows its original validation-based search with 1,000 training samples held out. Except in sensitivity/ablation studies, ECI-PP uses one global setting without dataset-specific tuning: 5 estimator folds, Corrector 64–16, λ = 0.5, γ = 10, n = 10, ξ = 20, and $\lambda _ { \mathrm { r e c } } = 0 . 0 2$

Full experimental details are provided in Appendix C to support reproducibility.

## 5.2 Experimental results

Main comparison. Table 1 compares our ECI-PP against baselines under the lv0.01 single-expert and multi3 multi-expert settings, both with corruption probability 0.3 and 9k constraints; additional constraint-budget results appear in Appendix D.1. Overall, ECI-PP is strongest, ranking first in 41 out of 48 test entries (2 expert regimes 8 datasets 3 metrics) and within the top two in 46/48, while one of the ProbPair-family methods is best in 45/48, supporting the ProbPair formulation under UPCC. Under single-expert supervision, ECI-PP outperforms the strong non-ProbPair baseline, SpherePair, in most entries (4–10% absolute NMI margins); further comparisons without pretraining make this pattern clearer (Appendix D.2). Under multi-expert supervision, the NMI margin over SpherePair widens to 6–32%. Appendix D.3 further compares baselines with an expert-aware ensemble wrapper. As an ablation signal, Weighted ProbPair shows a clearer advantage over ProbPair in the multiexpert setting, suggesting that information-based weighting is more useful when low-decisiveness constraints can indicate expert unfamiliarity. A remaining caveat is the severely imbalanced RCV1-10: SpherePair retains the best ACC, reflecting the advantage of its angular geometry for imbalanced pair structures, while ECI-PP still gives the best NMI/ARI, indicating stronger grouping consistency.

Table 1: Comparative performance (%) (ACC, NMI, ARI) across datasets for methods under lv0.01 and multi3 settings, with corruption probability 0.3 and 9k constraints. Blue and black represent training and test results, respectively. Best results are in bold, and second-best are underlined.
<table><tr><td colspan="3">ACC</td><td>Vanilla- DCC 16.8, 17.0</td><td>VolMax- DCC 46.1, 45.9</td><td>CIDEC 14.3, 13.6</td><td>SpherePair 50.2, 50.5</td><td>ProbPair</td><td>Weighted ProbPair</td><td>ECI-PP (Ours)</td></tr><tr><td rowspan="9">Single- expert 1v0.01</td><td>CIFAR100</td><td>NMI ARI</td><td>13.5, 14.6 6.0,6.3</td><td>45.6,46.1 30.4,30.5</td><td>11.0, 11.2 2.8,2.3</td><td>43.0,44.2 32.1, 32.6</td><td>49.4,49.5 43.2, 44.2 30.6,30.9</td><td>50.3,50.8 44.5, 45.7 31.5,32.1</td><td>52.7,52.9 49.2, 49.9 36.2, 36.5</td></tr><tr><td>CIFAR10</td><td>ACC NMI ARI</td><td>38.3,38.7 29.9,31.2 23.2, 24.0</td><td>82.5,82.5 71.9,72.2 68.3,68.3</td><td>44.3,44.0 36.1, 37.6 21.1, 21.3</td><td>84.9,85.7 72.4, 74.1 71.2, 72.5</td><td>85.9,86.0 75.0,75.4 73.0,73.2</td><td>86.6,86.7 76.2, 76.8 74.3, 74.5</td><td>87.8,87.8 79.0, 79.1 76.7,76.8</td></tr><tr><td>FMNIST</td><td>ACC NMI ARI</td><td>39.8, 40.2 30.1, 31.3 21.5,22.2</td><td>66.5,65.5 57.2, 56.8 46.4, 45.2</td><td>36.1, 35.7 29.7, 30.5 15.7, 15.6</td><td>72.0, 71.2 60.4, 60.9 53.0, 52.3</td><td>73.6, 72.5 62.9, 62.9 54.8,53.7</td><td>74.6, 73.6 63.9, 64.0 55.3,54.3</td><td>72.4, 71.1 65.7, 65.0 55.5,53.9</td></tr><tr><td>ImageNet10</td><td>ACC NMI ARI</td><td>39.0, 41.5 27.6, 33.7 21.1,25.2</td><td>87.4,88.1 77.8, 79.2 75.2, 76.1</td><td>63.5, 68.3 47.2, 57.5 35.5, 42.2</td><td>85.1,89.1 71.0,79.2 70.7,78.2</td><td>89.2, 90.5 80.1, 83.2 78.5, 80.7</td><td>89.1,90.3 80.3, 82.7 78.4,80.2</td><td>92.8, 93.1 88.1, 88.5 85.8, 86.2</td></tr><tr><td>MNIST</td><td>ACC NMI ARI</td><td>38.9,40.3 29.5, 32.2 23.1,24.8</td><td>75.6,76.9 59.3, 61.6 56.5,58.7</td><td>47.0, 46.7 41.4, 43.7 24.7, 24.2</td><td>87.8,89.1 75.4, 78.0 75.4,77.8</td><td>89.0,90.0 76.9,79.0 77.4, 79.4</td><td>89.9,90.8 78.3, 80.2 79.1, 80.9</td><td>90.7, 91.0 80.7, 81.6 80.8,81.5</td></tr><tr><td>REUTERS</td><td>ACC NMI ARI ACC</td><td>69.7,77.0 30.5,43.3 36.8,50.4</td><td>71.7, 76.8 34.3, 43.4 40.9,50.4</td><td>72.9, 79.3 35.7, 48.1 41.8, 54.9</td><td>77.9,81.8 48.7, 56.3 52.9, 61.1</td><td>78.2, 80.8 53.2, 58.3 55.5,61.0</td><td>80.4, 82.7 54.5, 59.2 58.6, 63.4</td><td>85.1, 86.6 61.4, 64.4 66.6, 70.0</td></tr><tr><td>STL10</td><td>NMI ARI ACC</td><td>35.5,37.8 22.9,27.6 17.1, 19.8 46.2, 46.4</td><td>79.0,80.1 66.7,69.3 62.8, 64.7</td><td>57.8,63.3 39.3,48.5 29.8,37.3</td><td>76.1, 79.7 59.5,66.5 57.2,63.3</td><td>81.6, 83.4 68.2, , 72.1 65.8,68.9</td><td>81.6,83.1 67.9,71.6 65.8, 68.4</td><td>85.4, 85.8 75.8, 76.6 72.8,73.4</td></tr><tr><td>RCV1-10</td><td>NMI ARI ACC</td><td>15.5, 15.9 19.8,20.1 15.4, 15.8</td><td>61.3,61.8 35.7,36.7 44.7,45.4 42.8, 42.5</td><td>42.5,44.3 29.5, 30.0 26.2,28.3 12.8, 12.1</td><td>71.2, 71.4 60.5,61.3 61.0, 61.4</td><td>52.7,53.0 53.8,54.1 43.6,43.7</td><td>55.2,55.1 57.4,57.5 47.1,47.0</td><td>69.0,68.9 65.5,65.8 62.0, 62.0</td></tr><tr><td>CIFAR100</td><td>NMI ARI ACC</td><td>12.8, 13.6 5.8,5.8 32.4,32.8</td><td>43.4, 43.8 27.1,27.1 76.5,76.6</td><td>10.0, 10.5 2.1,2.0 33.2, 31.6</td><td>44.3,44.4 40.5,41.5 27.1,27.6</td><td>43.3,43.4 40.0, 40.9 26.0, 26.2</td><td>46.1, 46.5 41.8, 43.0 28.0, 28.5</td><td>48.6, 48.8 48.1, 48.5 32.9,32.9</td></tr><tr><td rowspan="7">Multi- expert multi3</td><td>CIFAR10</td><td>NMI ARI</td><td>29.1, 30.9 21.2, 22.4</td><td>73.1,73.2 28.7,29.1 65.8,65.9 12.3, 12.1</td><td></td><td>73.8,74.5 64.7,66.5</td><td>81.3,81.7 71.5, 72.3</td><td>81.4, 81.7 72.0, 72.6</td><td>81.6, 82.1 77.8,77.8</td></tr><tr><td>FMNIST</td><td>ACC NMI ARI</td><td>35.2, 35.6 31.4,32.8 20.4,21.3</td><td>59.4,59.1 59.2, 59.0 45.0,44.3</td><td>28.6, 28.8 28.3, 30.6 14.5, 15.4</td><td>59.0, 60.5 61.0, 61.3 56.6,57.8 46.8,47.4</td><td>67.1, 67.8 66.6, 66.7 60.7, 61.2 49.9,50.0</td><td>67.9, 68.4 68.0, 67.6 61.7, 62.0 51.9,51.8</td><td>73.1, 73.0 65.3,64.8 66.0, 65.6</td></tr><tr><td>ImageNet10</td><td>ACC NMI ARI</td><td>31.8,34.2 22.0,26.8 15.8, 18.7</td><td>89.9,90.5 87.8,88.7 83.9, 84.9</td><td>39.4,38.8 31.0,34.8 14.3, 15.6</td><td>80.7,85.0 65.6,73.8 63.4, 71.0</td><td>87.4,89.2 76.1, 80.1 75.0, 78.3</td><td>90.1, 91.3 80.1, 83.0 79.8, 82.0</td><td>53.4, 52.8 92.4, 92.6 87.7, 88.1 85.4, 85.7</td></tr><tr><td>MNIST</td><td>ACC NMI ARI</td><td>32.2,33.3 26.5,29.3 18.4, 20.1</td><td>67.1,67.8 57.5,58.9 50.9,52.1</td><td>31.8, 31.0 28.8,30.2 13.2, 13.4</td><td>78.6,80.0 70.8,74.0 66.3, 69.1</td><td>80.5, 81.8 72.1,74.7 68.7,71.2</td><td>85.1, 86.1 75.0,77.2 73.4,75.5</td><td>84.7,85.1 79.9,80.7 77.1,77.5</td></tr><tr><td>REUTERS</td><td>ACC NMI ARI ACC</td><td>55.3,59.2 14.7, 22.8 19.2, 26.2 29.2, 30.7</td><td>67.5,68.0 45.4, 46.5 47.0,47.9 71.0,71.1</td><td>57.3,59.1 18.3,25.7 20.4,23.1 31.4,31.6</td><td>62.1,69.9 25.6,37.2 28.0, 40.7</td><td>74.5,79.0 44.6,52.8 50.2, 58.6</td><td>81.0, 83.9 51.5, 57.9 57.9, 64.2</td><td>88.3, 89.6 65.4, 68.7 72.9,76.0 87.9,87.7</td></tr><tr><td>STL10</td><td>NMI ARI ACC</td><td>21.5, 26.0 15.2, 17.7 44.3,44.6</td><td>70.2, 69.9 59.5,59.1 48.3,49.0</td><td>24.9, 31.7 13.6, 16.9 32.3, 32.0</td><td>67.8,71.4 53.2, 59.7 47.8,53.3 55.7, 55.9</td><td>81.1,83.0 66.1, 70.2 63.7,67.1 43.4,43.5</td><td>83.1,84.3 69.4, 72.0 67.4, 69.4 47.8,47.9</td><td>78.3, 78.5 75.9, 75.8 53.4,53.3</td></tr><tr><td>RCV1-10</td><td>NMI ARI</td><td>14.3, 14.7 17.8, 18.0</td><td>31.4, 32.1 30.2,30.9</td><td>14.1, 13.5 10.8, 10.3</td><td>51.9,52.7 43.6,44.2</td><td>46.6,47.4 34.4, 34.7</td><td>51.6, 52.0 38.2,38.4</td><td>58.9, 59.0 46.2, 46.1</td></tr></table>

![](images/2c9dc9a5e81bf0038ac30b4a6576f1279e1859c95a14720b5f7fd4a9613a4673.jpg)  
Figure 2: Test NMI performance (mean std over 5 runs) of all models across datasets under varying (A) expert quality, (B) corruption probability, and (C) multi-expert configuration, with 9k constraints.

Robustness across supervision conditions. We further stress-test all methods by varying one supervision factor in UPCC at a time. Around the 9k default supervision in Table 1, Fig. 2 reports test NMI as we vary (A) expert quality, (B) corruption probability, and (C) multi-expert configuration; ACC/ARI results are deferred to Appendix D.4. Across the three sweeps, the trends largely follow the expected supervision-quality ordering: stronger experts, lower corruption, and multi-expert settings with reduced expert blind spots (as detailed in Appendix C.3) generally yield better performance, while the gap between ECI-PP and strong ProbPair-family baselines often narrows in easier regimes. The stability advantage of ECI-PP is broadly visible across datasets, particularly on FMNIST, ImageNet10, and STL10, and is most pronounced under increasing corruption, where most baselines degrade sharply while ECI-PP remains comparatively stable. Overall, these results are consistent with the role of our ECI framework in improving robustness to imperfect and heterogeneous supervision.

Held-out diagnostics. We examine whether ECI-PP’s internal signals evolve as intended on sample-disjoint held-out constraints. Fig. 3 shows the MNIST case under multi3 with corruption rate 0.3, using benchmark labels only to form an external hard co-membership reference for diagnostics, rather than the canonical aleatoric relation R<sup>⋆</sup>. Panels (A–C) report Brier scores of the estimated, corrected, and integrated relations, and panel (D) reports the residual discrepancy between corrected relations and estimator beliefs on uncorrupted constraints; lower is better. Panels

![](images/54223b7eaec7e4c8da8cc209838cb027ca737e315e3462357c01defcd527e7dd.jpg)  
Figure 3: Held-out diagnostic evolution over 500 training iterations on MNIST under multi3 with corruption rate 0.3 (mean std over 5 runs). (A–C) Brier scores for estimated, corrected, and integrated relations; (D) residual discrepancy between corrected relations and estimator beliefs on uncorrupted constraints; (E,F) AUC/AP measuring the alignment between reliability-aware screening and injected corruption; (G,H) the same screening diagnostic with oracle-generated pairwise supervision before corruption.

(E,F) compare reliability-aware screening with the injected corruption indicator using AUC/AP, where higher values indicate better corruption separation. (G,H) repeat the same screening diagnostic with oracle-generated clean supervision before corruption, removing the expert epistemic uncertainty present in (E,F). The curves improve rapidly and then stabilize: relation Brier scores decrease, the post-correction residual discrepancy drops, and the screening signal aligns with corruption above chance, while the oracle-setting curves in (G,H) more closely track corruption as expected. Overall, these held-out trends are consistent with the intended behavior of the ECI framework, from internal relation estimates to reliability-aware screening; full cross-dataset diagnostics are in Appendix D.5.

Sensitivity and ablation. We vary the main ECI-PP design choices around the default configuration in Section 5.1 to assess sensitivity and ablate key components. The results (see Appendix D.6) show that the default setting is robust over broad ranges: larger K gives only mild gains relative to its cross-fitting cost, information-based warm-up is mildly helpful, small Correctors with moderate regularization are sufficient, and the screening and boundary parameters $( \gamma , n _ { 0 } , \xi )$ are generally insensitive. Reconstruction strength allows tuning flexibility, but the shared default remains reliable overall. These results indicate that ECI-PP is robust without dataset-specific hyperparameter tuning.

## 6 Conclusion

This paper addressed a realistic probabilistic constrained clustering setting, where pairwise supervision entangles aleatoric, epistemic, and stochastic factors. We formalized the constraint-observation process and its conditional identifiability, introduced ProbPair for angular learning from probabilistic relations, and built ECI-PP as a practical framework for such supervision. While existing methods largely rely on idealized constraints, ECI-PP outperforms state-of-the-art alternatives and remains robust under fallible, heterogeneous, and corrupted supervision without dataset-specific tuning.

These results come with several qualifications. The identifiability analysis is population-level and conditional; fully factor-specific finite-sample estimation would require stronger assumptions or additional observations beyond the current surrogate residual structure. Accordingly, the Corrector and reliability signal are practical surrogates rather than estimators of individual latent factors; a more theory-aligned extension would model structured correction, residual epistemic variation, and corruption-aware screening separately. Empirically, our experiments use controlled expert simulations, leaving real-world annotator-provided constraints as a natural next step. Finally, richer Corrector pair descriptors and principled stopping criteria may further improve robustness and training adaptivity.

## References

[1] Yucen Luo, Tian Tian, Jiaxin Shi, Jun Zhu, and Bo Zhang. Semi-crowdsourced clustering with deep generative models. Advances in Neural Information Processing Systems, 31, 2018. 1, 2

[2] Laura Manduchi, Kieran Chin-Cheong, Holger Michel, Sven Wellmann, and Julia Vogt. Deep conditional gaussian mixture model for constrained clustering. Advances in Neural Information Processing Systems, 34:11303–11314, 2021. 1, 2, 23

[3] Shaojie Zhang and Ke Chen. Angular constraint embedding via spherepair loss for constrained clustering. In Advances in Neural Information Processing Systems 39, 2025. 1, 2, 5, 7, 17, 18, 19, 23, 24, 25, 38

[4] Ian Davidson and Sugato Basu. A survey of clustering with instance level constraints. ACM Transactions on Knowledge Discoveryfrom data, 1(1-41):2–42, 2007. 1

[5] Germán González-Almagro, Daniel Peralta, Eli De Poorter, José-Ramón Cano, and Salvador García. Semi-supervised constrained clustering: an in-depth overview, ranked taxonomy and future research directions. Artificial Intelligence Review, 58:157, 2025. 1

[6] Kiri Wagstaff and Claire Cardie. Clustering with instance-level constraints. AAAI/IAAI, 1097(577-584):197, 2000. 1

[7] Kiri Wagstaff, Claire Cardie, Seth Rogers, Stefan Schrödl, et al. Constrained k-means clustering with background knowledge. In ICML, volume 1, pages 577–584, 2001. 1

[8] Zhengdong Lu and Todd Leen. Semi-supervised learning with penalized probabilistic clustering. Advances in Neural Information Processing Systems, 17, 2004. 1

[9] Mikhail Bilenko, Sugato Basu, and Raymond J Mooney. Integrating constraints and metric learning in semi-supervised clustering. In Proceedings of the twenty-first International Conference on Machine Learning, page 11, 2004. 1

[10] Sugato Basu, Arindam Banerjee, and Raymond J Mooney. Active semi-supervision for pairwise constrained clustering. In Proceedings ofthe 2004 SIAM International Conference on Data Mining, pages 333–344. SIAM, 2004. 1

[11] Brian Kulis, Sugato Basu, Inderjit Dhillon, and Raymond Mooney. Semi-supervised graph clustering: a kernel approach. In Proceedings ofthe 22nd International Conference on Machine Learning, pages 457–464, 2005. 1

[12] Yen-Chang Hsu, Zhaoyang Lv, Joel Schlosser, Phillip Odom, and Zsolt Kira. A probabilistic constrained clustering for transfer learning and image category discovery. In Proceedings ofthe CVPR 2018 Deep-Vision Workshop, 2018. 1, 2

[13] Yazhou Ren, Kangrong Hu, Xinyi Dai, Lili Pan, Steven CH Hoi, and Zenglin Xu. Semisupervised deep embedded clustering. Neurocomputing, 325:121–130, 2019. 1, 18, 23, 24

[14] Hongjing Zhang, Tianyang Zhan, Sugato Basu, and Ian Davidson. A framework for deep constrained clustering. Data Mining and Knowledge Discovery, 35:593–620, 2021. 1, 2, 7, 18, 23, 24

[15] Tri Nguyen, Shahana Ibrahim, and Xiao Fu. Deep clustering with incomplete noisy pairwise annotations: A geometric regularization approach. In International Conference on Machine Learning, pages 25980–26007. PMLR, 2023. 1, 2, 7, 18, 23, 24

[16] Sharon Fogel, Hadar Averbuch-Elor, Daniel Cohen-Or, and Jacob Goldberger. Clusteringdriven deep embedding with pairwise constraints. IEEE computer graphics and applications, 39(4):16–27, 2019. 1, 2

[17] Abu Quwsar Ohi, Muhammad Firoz Mridha, Farisa Benta Safir, Md Abdul Hamid, and Muhammad Mostafa Monowar. Autoembedder: A semi-supervised dnn embedding system for clustering. Knowledge-Based Systems, 204:106190, 2020. 1, 2

[18] Eyke Hüllermeier and Willem Waegeman. Aleatoric and epistemic uncertainty in machine learning: An introduction to concepts and methods. Machine Learning, 110(3):457–506, 2021. 1

[19] Yen-Chang Hsu, Zhaoyang Lv, Joel Schlosser, Phillip Odom, and Zsolt Kira. Multi-class classification without multi-class labels. In International Conference on Learning Representations, 2019. 2, 7, 18

[20] Marek Smieja, Łukasz Struski, and Mário AT Figueiredo. A classification-based approach to <sup>´</sup> semi-supervised clustering with pairwise constraints. Neural Networks, 127:193–203, 2020. 2

[21] Tian Tian, Jie Zhang, Xiang Lin, Zhi Wei, and Hakon Hakonarson. Model-based deep embedding for constrained clustering analysis of single cell rna-seq data. Nature communications, 12(1):1873, 2021. 2

[22] Jann Goschenhofer, Bernd Bischl, and Zsolt Kira. Constraintmatch for semi-constrained clustering. In 2023 International Joint Conference on Neural Networks (IJCNN), pages 1–10. IEEE, 2023. 2

[23] Yi Wen, Suyuan Liu, Xinhang Wan, Siwei Wang, Ke Liang, Xinwang Liu, Xihong Yang, and Pei Zhang. Efficient multi-view graph clustering with local and global structure preservation. In Proceedings ofthe 31st ACM International Conference on Multimedia, pages 3021–3030, 2023. 2

[24] Jian Dai, Zhenwen Ren, Yunzhi Luo, Hong Song, and Jian Yang. Tensorized anchor graph learning for large-scale multi-view clustering. Cognitive Computation, 15(5):1581–1592, 2023. 2

[25] Suyuan Liu, Qing Liao, Siwei Wang, Xinwang Liu, and En Zhu. Robust and consistent anchor graph learning for multi-view clustering. IEEE Transactions on Knowledge and Data Engineering, 2024. 2

[26] Martin HC Law, Alexander Topchy, and Anil K Jain. Model-based clustering with probabilistic constraints. In Proceedings ofthe 2005 SIAM International Conference on Data Mining, pages 641–645. SIAM, 2005. 2

[27] Martin HC Law, Alexander Topchy, and Anil K Jain. Clustering with soft and group constraints. In Joint IAPR International Workshops on Statistical Techniques in Pattern Recognition (SPR) and Structural and Syntactic Pattern Recognition (SSPR), pages 662–670. Springer, 2004. 2

[28] Irene Diaz-Valenzuela, M Amparo Vila, and Maria J Martin-Bautista. On the use of fuzzy constraints in semisupervised clustering. IEEE Transactions on Fuzzy Systems, 24(4):992–999, 2015. 2

[29] Zhen Wang, Shan-Shan Wang, Lan Bai, Wen-Si Wang, and Yuan-Hai Shao. Semisupervised fuzzy clustering with fuzzy pairwise constraints. IEEE Transactions on Fuzzy Systems, 30(9):3797–3811, 2021. 2

[30] Jian-Ping Mei, Huajiang Lv, Jiuwen Cao, and Weihua Gong. Pairwise constrained fuzzy clustering: Relation, comparison and parallelization. International Journal of Fuzzy Systems, 21(6):1938–1949, 2019. 2

[31] Jian-Ping Mei and Huajiang Lv. Semi-supervised fuzzy c-means regularized with pairwise constraints. In 2018 14th International Conference on Natural Computation, Fuzzy Systems and Knowledge Discovery (ICNC-FSKD), pages 781–786. IEEE, 2018. 2

[32] Violaine Antoine, Benjamin Quost, M-H Masson, and Thierry Denoeux. Cevclus: evidential clustering with instance-level constraints for relational data. Soft Computing, 18(7):1321–1335, 2014. 2

[33] Feng Li, Shoumei Li, and Thierry Denœux. k-cevclus: Constrained evidential clustering of large dissimilarity data. Knowledge-Based Systems, 142:29–44, 2018. 2

[34] Dan Pelleg and Dorit Baras. K-means with large and noisy constraint sets. In European Conference on Machine Learning, pages 674–682. Springer, 2007. 2

[35] Thiago F Covoes, Eduardo R Hruschka, and Joydeep Ghosh. A study of k-means-based algorithms for constrained clustering. Intelligent Data Analysis, 17(3):485–505, 2013. 2

[36] Xiatian Zhu, Chen Change Loy, and Shaogang Gong. Constrained clustering with imperfect oracles. IEEE transactions on neural networks and learning systems, 27(6):1345–1357, 2015. 2

[37] Hongfu Liu, Zhiqiang Tao, and Yun Fu. Partition level constrained clustering. IEEE transactions on pattern analysis and machine intelligence, 40(10):2469–2483, 2017. 2

[38] Yale Chang, Junxiang Chen, Michael H Cho, Peter J Castaldi, Edwin K Silverman, and Jennifer G Dy. Multiple clustering views from multiple uncertain experts. In International Conference on Machine Learning, pages 674–683. PMLR, 2017. 2

[39] Hongjing Zhang, Sugato Basu, and Ian Davidson. A framework for deep constrained clustering algorithms and advances. In Machine Learning and Knowledge Discovery in Databases: European Conference, ECML PKDD 2019, Würzburg, Germany, September 16–20, 2019, Proceedings, Part I, pages 57–72. Springer, 2020. 2

[40] Alexander Philip Dawid and Allan M Skene. Maximum likelihood estimation of observer error-rates using the em algorithm. Journal ofthe Royal Statistical Society: Series C (Applied Statistics), 28(1):20–28, 1979. 2

[41] Filipe Rodrigues and Francisco Pereira. Deep learning from crowds. In Proceedings of the AAAI conference on artificial intelligence, volume 32, 2018. 2

[42] Jacob Whitehill, Ting-fan Wu, Jacob Bergsma, Javier Movellan, and Paul Ruvolo. Whose vote should count more: Optimal integration of labels from labelers of unknown expertise. Advances in Neural Information Processing Systems, 22, 2009. 2

[43] Peter Welinder, Steve Branson, Pietro Perona, and Serge Belongie. The multidimensional wisdom of crowds. Advances in Neural Information Processing Systems, 23, 2010. 2

[44] Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images. 2009. 7, 17

[45] Han Xiao, Kashif Rasul, and Roland Vollgraf. Fashion-mnist: a novel image dataset for benchmarking machine learning algorithms. arXiv preprint arXiv:1708.07747, 2017. 7, 17

[46] Jianlong Chang, Lingfeng Wang, Gaofeng Meng, Shiming Xiang, and Chunhong Pan. Deep adaptive image clustering. In Proceedings of the IEEE international conference on computer vision, pages 5879–5887, 2017. 7, 17

[47] Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner. Gradient-based learning applied to document recognition. Proceedings ofthe IEEE, 86(11):2278–2324, 1998. 7, 17

[48] Adam Coates, Andrew Ng, and Honglak Lee. An analysis of single-layer networks in unsupervised feature learning. In Proceedings of the fourteenth international conference on artificial intelligence and statistics, pages 215–223. JMLR Workshop and Conference Proceedings, 2011. 7, 17

[49] Junyuan Xie, Ross Girshick, and Ali Farhadi. Unsupervised deep embedding for clustering analysis. In International Conference on Machine Learning, pages 478–487. PMLR, 2016. 7, 17, 18, 24

[50] Victor Chernozhukov, Denis Chetverikov, Mert Demirer, Esther Duflo, Christian Hansen, Whitney Newey, and James Robins. Double/debiased machine learning for treatment and structural parameters. Econometrics Journal, 21:C1–C68, 2018. 15

[51] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A largescale hierarchical image database. In 2009 IEEE conference on Computer Vision and Pattern Recognition, pages 248–255. IEEE, 2009. 17

[52] Xifeng Guo, Long Gao, Xinwang Liu, and Jianping Yin. Improved deep embedded clustering with local structure preservation. In IJCAI, volume 17, pages 1753–1759, 2017. 18, 24

[53] Yunfan Li, Peng Hu, Zitao Liu, Dezhong Peng, Joey Tianyi Zhou, and Xi Peng. Contrastive clustering. In Proceedings of the AAAI conference on artificial intelligence, volume 35, pages 8547–8555, 2021. 23

[54] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 770–778, 2016. 23

[55] Geoffrey E Hinton and Ruslan R Salakhutdinov. Reducing the dimensionality of data with neural networks. Science, 313(5786):504–507, 2006. 23

[56] Pascal Vincent, Hugo Larochelle, Yoshua Bengio, and Pierre-Antoine Manzagol. Extracting and composing robust features with denoising autoencoders. In Proceedings ofthe 25th International Conference on Machine learning, pages 1096–1103. ACM, 2008. 23

## A Additional theoretical details

## A.1 Derivation of the logistic-normal judgment density

Starting from Equation (1), define

$$
\mu _ { e , a b } : = \mathrm { l o g i t } _ { \varepsilon } ( R _ { a b } ^ { \star } ) + m _ { e } ( \phi _ { a b } ) .
$$

Conditionally on $( R _ { a b } ^ { \star } , e , \phi _ { a b } )$ , we then have

$$
\mathrm { l o g i t } \left( y _ { e , a b } ^ { \mathrm { j u d } } \right) = \mu _ { e , a b } + u _ { e , a b } , \qquad u _ { e , a b } \sim \mathcal { N } \bigl ( 0 , \tau _ { e } ^ { 2 } ( \phi _ { a b } ) \bigr ) ,
$$

so that

$$
\mathrm { l o g i t } \left( \boldsymbol { y } _ { e , a b } ^ { \mathrm { j u d } } \right) \sim \mathcal { N } \big ( \mu _ { e , a b } , \tau _ { e } ^ { 2 } ( \phi _ { a b } ) \big ) .
$$

For a generic $y \in ( 0 , 1 )$ , write $v = \log \mathrm { i t } ( y )$ . By the change-of-variables formula,

$$
p _ { \mathrm { j u d } } ( y \mid R _ { a b } ^ { \star } , e , \phi _ { a b } ) = \varphi \Bigl ( v ; \mu _ { e , a b } , \tau _ { e } ^ { 2 } ( \phi _ { a b } ) \Bigr ) \left. \frac { d v } { d y } \right. , \qquad v = \mathrm { l o g i t } ( y ) , \quad y \in ( 0 , 1 ) .
$$

Using $\begin{array} { r } { \frac { d v } { d y } = \frac { d \log \mathrm { i t } ( y ) } { d y } = \frac { 1 } { y ( 1 - y ) } } \end{array}$ , we obtain

$$
p _ { \mathrm { j u d } } ( y \mid R _ { a b } ^ { \star } , e , \phi _ { a b } ) = \frac { 1 } { y ( 1 - y ) } \varphi \Big ( \log \mathrm { i t } ( y ) ; \log \mathrm { i t } _ { \varepsilon } ( R _ { a b } ^ { \star } ) + m _ { e } ( \phi _ { a b } ) , \tau _ { e } ^ { 2 } ( \phi _ { a b } ) \Big ) , \qquad y \in ( 0 , 1 ) ,
$$

which is exactly Equation (2).

## A.2 Proof of Lemma 3.2

Proof. Fix an expert e with nonzero observation probability. By assumption,

$$
m _ { \eta _ { e } } ( \phi _ { a b } ) = m _ { \eta _ { e } ^ { \prime } } ( \phi _ { a b } ) - k _ { e }
$$

for all observable triples $( a , b , e )$ . Taking expectation over the observable-pair distribution conditional on e yields

$$
\mathbb { E } _ { ( a , b ) | e } [ m _ { \eta _ { e } } ( \phi _ { a b } ) ] = \mathbb { E } _ { ( a , b ) | e } [ m _ { \eta _ { e } ^ { \prime } } ( \phi _ { a b } ) ] - k _ { e } .
$$

Both expectations vanish by Assumption 3.1, hence $k _ { e } = 0 . ^ { 1 }$ Since e was arbitrary over experts with nonzero observation probability, the conclusion holds for every such expert. □

## A.3 Proof of Lemma 3.3

Proof. By Equations (2) and (4), the observation law takes the form Equation (5).

Assume that

$$
p ( \cdot \mid \mu , s , \pi ) = p ( \cdot \mid \mu ^ { \prime } , s ^ { \prime } , \pi ^ { \prime } )
$$

on (0, 1), where $p ( \boldsymbol { y } \mid \mu , s , \pi )$ is the density defined in Equation (5). Since $\mathbf { 1 } _ { [ 0 , 1 ] } ( y ) = 1$ for $y \in ( 0 , 1 )$ , equality of the two densities means that

$$
\left( 1 - \pi \right) { \frac { 1 } { y ( 1 - y ) } } \varphi { \bigl ( } \log \mathrm { i t } ( y ) ; \mu , s ^ { 2 } { \bigr ) } + \pi = \left( 1 - \pi ^ { \prime } \right) { \frac { 1 } { y ( 1 - y ) } } \varphi { \bigl ( } \log \mathrm { i t } ( y ) ; \mu ^ { \prime } , s ^ { \prime ^ { 2 } } { \bigr ) } + \pi ^ { \prime }
$$

for all $y \in ( 0 , 1 )$

Write $v = \log \mathrm { i t } ( y )$ , so that $y = \sigma ( v )$ and $y ( 1 - y ) = \sigma ( v ) ( 1 - \sigma ( v ) )$ . Multiplying both sides by $y ( 1 - y )$ gives

$$
\begin{array} { r } { ( 1 - \pi ) \varphi ( v ; \mu , s ^ { 2 } ) + \pi \sigma ( v ) ( 1 - \sigma ( v ) ) = ( 1 - \pi ^ { \prime } ) \varphi ( v ; \mu ^ { \prime } , { s ^ { \prime } } ^ { 2 } ) + \pi ^ { \prime } \sigma ( v ) ( 1 - \sigma ( v ) ) } \end{array}
$$

for all $v \in \mathbb { R }$

As $v  + \infty$ , the Gaussian terms decay faster than $e ^ { - v }$ , whereas

$$
\sigma ( v ) ( 1 - \sigma ( v ) ) = \frac { e ^ { - v } } { ( 1 + e ^ { - v } ) ^ { 2 } } \sim e ^ { - v } .
$$

Therefore,

$$
\operatorname* { l i m } _ { v \to + \infty } e ^ { v } \Big [ ( 1 - \pi ) \varphi ( v ; \mu , s ^ { 2 } ) + \pi \sigma ( v ) ( 1 - \sigma ( v ) ) \Big ] = \pi ,
$$

and similarly the right-hand side tends to $\pi ^ { \prime }$ . Hence $\pi = \pi ^ { \prime }$

Subtracting the common contamination term then yields

$$
\varphi ( v ; \mu , s ^ { 2 } ) = \varphi ( v ; \mu ^ { \prime } , { s ^ { \prime } } ^ { 2 } ) \qquad { \mathrm { f o r ~ a l l ~ } } v \in \mathbb { R } .
$$

Injectivity of the Gaussian family implies $\mu = \mu ^ { \prime }$ and $s = s ^ { \prime }$ . Therefore $( \mu , s , \pi ) = ( \mu ^ { \prime } , s ^ { \prime } , \pi ^ { \prime } )$ proving the claim. 口

## A.4 Boundary cases for structural separability

Assumption 3.4 concerns the decomposition of the identifiable location term logit $( r _ { \theta } ( x _ { a } , x _ { b } ) )$ + $m _ { \eta _ { e } } ( \phi _ { a b } )$ into a canonical relation component and an expert-specific epistemic component. Two boundary cases clarify this requirement. (i) If the pair descriptor $\phi _ { a b }$ is uninformative, for example constant over observable pairs for a given expert, then $m _ { \eta _ { e } } ( \phi _ { a b } )$ reduces to an expert-specific constant and cannot capture pair-dependent epistemic effects; the intended epistemic component is then not represented by the chosen model class. (ii) Conversely, if $\phi _ { a b }$ is overly rich and the function class for $m _ { \eta _ { e } } ( \phi _ { a b } )$ overlaps with that of logit $\mathbf { \rho } _ { \varepsilon } ( r _ { \theta } ( x _ { a } , x _ { b } ) )$ , then shared pair-dependent variation can be moved between the two terms while preserving their sum. For example, a sufficiently small zero-mean shift lying in this overlap can be added to the canonical location term and subtracted from the expert-specific term without violating centering or changing the observation law. The condition therefore rules out such interchangeability, ensuring that canonical aleatoric variation and structured expert-specific variation play distinct explanatory roles on the observable support.

## A.5 Proof of Theorem 3.5

Proof. Assume that the induced conditional laws coincide under $\Xi$ and $\Xi ^ { \prime }$ for every observable triple $( a , b , e )$ . For each such triple, define

$$
\mu _ { \Xi } ( a , b , e ) : = \mathrm { l o g i t } _ { \varepsilon } \big ( r _ { \theta } ( x _ { a } , x _ { b } ) \big ) + m _ { \eta _ { e } } ( \phi _ { a b } ) , \qquad s _ { \Xi } ( a , b , e ) : = \tau _ { \zeta _ { e } } ( \phi _ { a b } ) ,
$$

and define $\mu { \boldsymbol { \Xi } } ^ { \prime } \left( a , b , e \right)$ and $s _ { \Xi ^ { \prime } } ( a , b , e )$ analogously. By Equations (4) and (5), the conditional law of the observed response for $( a , b , e )$ is exactly

$$
p \big ( y \mid \mu _ { \Xi } ( a , b , e ) , s _ { \Xi } ( a , b , e ) , \pi _ { e } \big ) ,
$$

and similarly under $\Xi ^ { \prime }$ . Since the two conditional laws coincide, Lemma 3.3 implies that for every observable triple $( a , b , e )$

$$
\mu _ { \Xi } ( a , b , e ) = \mu _ { \Xi ^ { \prime } } ( a , b , e ) , \qquad s _ { \Xi } ( a , b , e ) = s _ { \Xi ^ { \prime } } ( a , b , e ) , \qquad \pi _ { e } = \pi _ { e } ^ { \prime } .
$$

In particular,

$$
\log \mathrm { i t } _ { \varepsilon } \bigl ( r _ { \theta } ( x _ { a } , x _ { b } ) \bigr ) + m _ { \eta _ { e } } ( \phi _ { a b } ) = \log \mathrm { i t } _ { \varepsilon } \bigl ( r _ { \theta ^ { \prime } } ( x _ { a } , x _ { b } ) \bigr ) + m _ { \eta _ { e } ^ { \prime } } ( \phi _ { a b } )
$$

for all observable triples $( a , b , e )$

Applying Assumption 3.4, for each expert e there exists a constant $k _ { e } \in \mathbb { R }$ such that for all observable triples $( a , b , e )$

$$
\begin{array} { r } { \log \mathrm { i t } _ { \varepsilon } \big ( r _ { \theta } ( x _ { a } , x _ { b } ) \big ) = \log \mathrm { i t } _ { \varepsilon } \big ( r _ { \theta ^ { \prime } } ( x _ { a } , x _ { b } ) \big ) + k _ { e } , \qquad m _ { \eta _ { e } } ( \phi _ { a b } ) = m _ { \eta _ { e } ^ { \prime } } ( \phi _ { a b } ) - k _ { e } . } \end{array}
$$

By Lemma 3.2, each $k _ { e }$ on the observable expert support must be zero. Therefore,

$$
\begin{array} { r } { \log \mathrm { i t } _ { \varepsilon } \big ( r _ { \theta } ( x _ { a } , x _ { b } ) \big ) = \log \mathrm { i t } _ { \varepsilon } \big ( r _ { \theta ^ { \prime } } ( x _ { a } , x _ { b } ) \big ) \qquad \mathrm { f o r ~ a l l ~ } ( a , b ) \in \mathcal { P } _ { \mathrm { o b s } } . } \end{array}
$$

Since $r _ { \theta } ( x _ { a } , x _ { b } ) , r _ { \theta ^ { \prime } } ( x _ { a } , x _ { b } ) \in [ \varepsilon , 1 - \varepsilon ]$ on ${ \mathcal { P } } _ { \mathrm { o b s } } ,$ the clamped logit satisfies $\mathrm { l o g i t } _ { \varepsilon } ( u ) = \mathrm { l o g i t } ( u )$ on this interval and is therefore injective there. It follows that

$$
\begin{array} { r } { r _ { \theta } ( x _ { a } , x _ { b } ) = r _ { \theta ^ { \prime } } ( x _ { a } , x _ { b } ) \qquad \mathrm { f o r ~ a l l ~ } ( a , b ) \in \mathcal { P } _ { \mathrm { o b s } } , } \end{array}
$$

which proves the theorem.

## B Algorithm and complexity analysis

## B.1 Algorithm

We provide the detailed ECI-PP procedure introduced in Section 4, summarized in Algorithm 1.

## B.2 Computational complexity analysis

For an Estimator/Integrator backbone $( f , g , m , T )$ , let $c _ { f }$ and $c _ { g }$ denote the per-instance training costs of the encoder $f$ and decoder g, respectively, and let $c _ { m , T } = { \cal O } ( 1 )$ denote the cost of updating the scalar readout parameters $( m , \bar { T } )$ . We write $c _ { \mathrm { p a i r } } = O ( c _ { f } + c _ { m , T } )$ for the amortized per-constraint cost of the ProbPair-style pairwise term, and $c _ { \mathrm { r e c } } = O ( c _ { f } + c _ { g } )$ for the per-instance reconstruction cost. Thus, one backbone epoch over constraints and instances $\mathcal { X }$ costs $O ( | \mathcal { C } | c _ { \mathrm { p a i r } } + | \mathcal { X } | c _ { \mathrm { r e c } } )$ In particular, the pairwise term scales with the number of constraints, while the reconstruction term scales linearly with $| \mathcal { X } |$ as the usual autoencoder regularization cost. Notably, each Corrector $\mathrm { C o r } _ { e } = q _ { \nu _ { \mathrm { \scriptsize { i } } } }$ is a small expert-specific MLP, rather than an encoder–decoder backbone, and its capacity is intentionally kept more limited than that of the Estimator/Integrator networks so that it learns only shared structured expert-specific patterns rather than arbitrary fitting. We write $c _ { \mathrm { c o r } }$ for its amortized per-constraint update cost.

Under this notation, with $N _ { \mathrm { w u } }$ warm-up epochs and $N _ { \mathrm { e p } }$ total epochs, the warm-up stage costs $O \big ( N _ { \mathrm { w u } } ( K + 1 ) \big ( | { \mathcal C } | c _ { \mathrm { p a i r } } + | { \mathcal X } | c _ { \mathrm { r e c } } \big ) + J _ { \mathrm { c o r } } | { \mathcal C } | c _ { \mathrm { c o r } } \big )$ , where $J _ { \mathrm { c o r } }$ is the average number of optimization passes used when fitting the Correctors. Each of the $T _ { \mathrm { r e f } } : = N _ { \mathrm { e p } } - N _ { \mathrm { w u } }$ subsequent refinement rounds costs $O \big ( ( K + 1 ) \big ( | \mathcal { C } | c _ { \mathrm { p a i r } } + | \mathcal { X } | c _ { \mathrm { r e c } } \big ) + J _ { \mathrm { c o r } } | \mathcal { C } | c _ { \mathrm { c o r } } \big )$ . The total training complexity is

$$
\begin{array} { r } { O \big ( \big ( N _ { \mathrm { w u } } + T _ { \mathrm { r e f } } \big ) ( K + 1 ) \big ( | \mathcal { C } | c _ { \mathrm { p a i r } } + | \mathcal { X } | c _ { \mathrm { r e c } } \big ) + \big ( 1 + T _ { \mathrm { r e f } } \big ) J _ { \mathrm { c o r } } | \mathcal { C } | c _ { \mathrm { c o r } } \big ) . } \end{array}
$$

The dominant overhead comes from the K-fold cross-fitting of the Estimators, namely the repeated backbone updates scaled by $( K + 1 ) \big ( | { \mathcal C } | c _ { \mathrm { p a i r } } + | { \mathcal X } | c _ { \mathrm { r e c } } \big )$ . This overhead is justified by the need for out-of-fold beliefs: the Estimators are introduced not as a redundant ensemble, but to provide modelside references decoupled from the specific constraints they are later used to assess, thereby reducing self-confirmation and overfitting to noisy supervision. This use of held-out predictions is analogous in spirit to debiased/double machine learning [50]: both reduce in-sample contamination, although ECI-PP uses them to obtain cleaner belief proxies under noisy supervision rather than to pursue semiparametric efficiency. By contrast, the expert-specific correction stage does not scale as $O \bar { ( } E | \mathcal { C } | )$ , because each constraint is routed only to its corresponding expert module and $\begin{array} { r } { \sum _ { e = 1 } ^ { E } | \mathcal { T } _ { e } | = | \mathcal { C } | } \end{array}$ . Thus, the per-expert Correctors add only a linear-in- refinement cost, which is largely unavoidable under heterogeneous expert supervision.

Algorithm 1 ECI-PP for learning under UPCC   
Require: training data $\mathcal { X } = \{ x _ { j } \} _ { j = 1 } ^ { | \mathcal { X } | }$ ; constraints $\mathcal { C } = \{ ( a _ { i } , b _ { i } , e _ { i } , y _ { i } ) \} _ { i = 1 } ^ { | \mathcal { C } | }$ ; number of experts $E ;$   
number of estimator folds $K ;$ total training epochs $N _ { \mathrm { e p } } ;$ warm-up epochs $N _ { \mathrm { w u } } ;$ hyperparameters   
$\lambda _ { \mathrm { { r e c } } }$ (reconstruction weight), $\lambda _ { \mathrm { c o r } }$ (Corrector regularization), $\xi$ (soft-clipping sharpness), γ   
(screening sharpness), $n _ { 0 }$ (Bayesian-confidence scale); clustering routine Clustering( )   
1: Partition $\mathcal { C }$ into K disjoint folds $\{ \mathcal { C } ^ { ( k ) } \} _ { k = 1 } ^ { K }$ and define ${ \mathcal { T } } _ { e } : = \{ i : e _ { i } = e \}$ for each expert e   
2: Initialize $\{ \mathrm { E s t } ^ { ( k ) } = \big ( f _ { \omega _ { k } } , g _ { \omega _ { k } ^ { \prime } } , m _ { k } , T _ { k } \big ) \} _ { k = 1 } ^ { K } , \{ \mathrm { C o r } _ { e } = q _ { \nu _ { e } } \} _ { e = 1 } ^ { E } ,$ and Int ${ \bf \Omega } = ( f _ { \omega } , g _ { \omega ^ { \prime } } , m , T )$   
3: Stage I: Warm-up   
4: Compute decisiveness weights $\{ \kappa _ { i } \} _ { i = 1 } ^ { | { \mathcal C } | }$ by Eq. (9)   
5: for $n = 1 , \ldots , N _ { \mathrm { w u } }$ do   
6: for $k = 1 , \ldots , K$ do   
7: Update $\mathrm { E s t } ^ { ( k ) }$ on ${ \mathcal { C } } \backslash { \mathcal { C } } ^ { ( k ) }$ by gradient descent on Eq. (8) with $y _ { i }$ and weights $\kappa _ { i }$   
8: end for   
9: end for   
10: Aggregate holdout predictions from $\{ \mathrm { E s t } ^ { ( k ) } \} _ { k = 1 } ^ { K }$ to obtain $\{ \hat { y } _ { i } ^ { \mathrm { o o f } } \} _ { i = 1 } ^ { | \mathcal { C } | }$ via Eq. (6)   
11: for $e = 1 , \ldots , E$ do   
12: Update $\mathrm { C o r } _ { e }$ on $\mathcal { T } _ { e }$ by gradient descent on Eq. (10) until convergence   
13: end for   
14: Compute $\{ y _ { i } ^ { \mathrm { c o r } } , \mathrm { g a p } _ { i } , w _ { i } \} _ { i = 1 } ^ { | \mathcal { C } | }$ by Eq. (11)   
15: Set $n _ { i } \gets n _ { 0 } w _ { i }$ and compute Bayesian-confidence supervision $\{ y _ { i } ^ { \mathrm { B C } } \} _ { i = 1 } ^ { | { \mathcal C } | }$ by Eq. (12)   
16: for $n = 1 , \ldots , N _ { \mathrm { w u } }$ do   
17: Update Int by gradient descent on Eq. (8) with $y _ { i } ^ { \mathrm { B C } }$ and weights $w _ { i }$   
18: end for   
19: Stage II: Iterative refinement   
20: for $\breve { t } = N _ { \mathrm { w u } } + 1 , \ldots , N _ { \mathrm { e p } }$ do   
21: Read out Integrator-side relations $\{ y _ { i } ^ { \mathrm { i n t } } \} _ { i = 1 } ^ { | { \mathcal C } | }$ from the Integrator embeddings via Eq. (6)   
22: for $k = 1 , \ldots , K$ do   
23: Update $\mathrm { E s t } ^ { ( k ) }$ on ${ \mathcal { C } } \setminus { \mathcal { C } } ^ { ( k ) }$ by gradient descent on Eq. (8) with $y _ { i } ^ { \mathrm { i n t } }$ and weights $w _ { i }$   
24: end for   
25: Aggregate updated holdout predictions to refresh $\{ \hat { y } _ { i } ^ { \mathrm { o o f } } \} _ { i = 1 } ^ { | { \mathcal C } | }$   
26: for $e = 1 , \ldots , E$ do   
27: Update $\mathrm { C o r } _ { e }$ on $\mathcal { T } _ { e }$ by gradient descent on Eq. (10) until convergence   
28: end for   
29: Refresh $\{ y _ { i } ^ { \mathrm { c o r } } , \mathrm { g a p } _ { i } , w _ { i } \} _ { i = 1 } ^ { | \mathcal { C } | }$ by Eq. (11)   
30: Set $n _ { i } \gets n _ { 0 } w _ { i }$ and refresh Bayesian-confidence supervision $\{ y _ { i } ^ { \mathrm { B C } } \} _ { i = 1 } ^ { | { \mathcal C } | }$ by Eq. (12)   
31: Update Int by gradient descent on Eq. (8) with $y _ { i } ^ { \mathrm { B C } }$ and weights $w _ { i }$   
32: end for   
33: Stage III: Clustering and unseen-instance assignment   
34: Obtain the final normalized Integrator embeddings $\mathcal { Z } ^ { \mathrm { i n t } } \gets \{ \mathrm { N o r m } ( f _ { \omega } ( x _ { j } ) ) \} _ { j = 1 } ^ { | \mathcal { X } | }$   
35: Apply Clustering $\left( \mathcal { Z } ^ { \mathrm { i n t } } \right)$ to obtain the final partition $\widehat { S }$   
36: Compute cluster centroids $\{ \mu _ { c } \}$ from ${ \mathcal { Z } } ^ { \mathrm { i n t } }$ and $\widehat { S }$   
37: for each unseen instance x˜ do   
38: Compute $\tilde { z } ^ { \mathrm { i n t } } \gets \mathrm { N o r m } ( f _ { \omega } ( \tilde { x } ) )$   
39: Assign x˜ to the nearest centroid in $\{ \mu _ { c } \}$   
40: end for

More broadly, the practical cost profile of ECI-PP should be interpreted together with its training protocol: it does not require an additional expert-aware wrapper in multi-expert settings (see Appendix C.4), and unlike methods such as VolMaxDCC, its performance is not tightly tied to an extensive hyperparameter-search stage. Meanwhile, Appendix D.2 shows that ECI-PP is less dependent on unsupervised pre-training, which can be advantageous in resource-limited settings, and an overall training-time comparison is reported in Appendix E. A discussion of the choice of K is provided in Appendix D.6.

## C Details of experimental settings

We provide supplementary information for the experimental setup in Section 5.1, including the benchmark datasets, compared methods, constraint-generation procedure, experimental protocol, and implementation details.

## C.1 Datasets

We use eight benchmark datasets covering image and text clustering tasks, with different numbers of classes and class-balance properties. Their details are as follows.

CIFAR100-20<sup>2</sup> [44]: A 20-class version of CIFAR100 obtained by using the 20 superclasses as clustering targets. The original CIFAR100 dataset contains 60,000 real-world 32 32 color images from 100 fine-grained classes, grouped into 20 superclasses. In our experiments, the 20 superclasses are treated as ground-truth clusters, with 3,000 images per superclass.

CIFAR10<sup>3</sup> [44]: A natural-image dataset containing 60,000 32  32 color images from 10 object categories, with 6,000 images per category.

FMNIST<sup>4</sup> [45]: A fashion-product image dataset containing 70,000 grayscale 28  28 images from 10 categories. The official split contains 60,000 training images and 10,000 test images.

ImageNet10<sup>5</sup> [46]: A 10-class subset of ImageNet [51], containing 13,000 color images in total, with 1,300 images per class.

MNIST<sup>6</sup> [47]: A handwritten-digit dataset containing 70,000 grayscale 28  28 images from 10 digit classes. The official split contains 60,000 training images and 10,000 test images.

Reuters subset<sup>7</sup> [49]: A text benchmark derived from the RCV1 corpus<sup>8</sup>, using tf–idf features over the 2,000 most frequent words. It contains four root categories, Corporate/Industrial (CCAT), Economics (ECAT), Government/Social (GCAT), and Markets (MCAT), treated as ground-truth clusters. The four categories contain 4,840, 3,470, 2,673, and 1,017 samples, respectively, giving a mildly imbalanced benchmark. The official split contains 10,000 training samples and 2,000 test samples.

STL10<sup>9</sup> [48]: An image dataset containing 13,000 96  96 color images from 10 object categories, with 1,300 images per category.

RCV1-10<sup>10</sup> [3]: An imbalanced 10-category subset of RCV1 constructed from single-label articles in categories C14, C18, C313, C42, E21, E311, GDEF, GODD, GWELF, and M13. It contains 177,669 documents represented by tf–idf features over the 2,000 most frequent word stems. The class distribution is highly skewed, ranging from 903 documents in GWELF to 53,127 documents in M13, making it a benchmark for evaluating clustering under severe class imbalance.

These benchmarks follow common evaluation practice in deep clustering and constrained clustering: MNIST, FMNIST, Reuters subset, CIFAR10, ImageNet10, STL10, and CIFAR100-20 have been widely used in prior deep clustering/DCC studies [49, 52, 13, 14, 15, 3], while RCV1-10 follows the imbalanced text benchmark introduced in [3]. For MNIST, FMNIST, and Reuters subset, we use the official train/test splits, yielding 60,000/10,000, 60,000/10,000, and 10,000/2,000 training/test samples, respectively. For CIFAR100-20, CIFAR10, ImageNet10, STL10, and RCV1-10, we randomly split each dataset into 80% training and 20% testing, yielding 48,000/12,000, 48,000/12,000, 10,400/2,600, 10,400/2,600, and 142,135/35,534 training/test samples, respectively. All probabilistic constraints for training are generated from the training split only.

## C.2 Compared methods

We summarize the compared methods and their use under probabilistic pairwise supervision. Methods tied to hard binary constraints or discrete signed priors are not included, as they do not natively operate on the real-valued probabilistic supervision considered in UPCC.

VanillaDCC. VanillaDCC [19] is an end-to-end DCC method based on the Meta Classification Likelihood (MCL) loss. Given the soft cluster-assignment vector ${ \pmb q } _ { j } = ( q _ { j 1 } , \dots , q _ { j C } )$ for each instance $x _ { j }$ , the pairwise co-clustering probability of a constrained pair is $P _ { i } ^ { \mathrm { c o } } = \pmb { q } _ { a _ { i } } \pmb { q } _ { b _ { i } } ^ { \top }$ . The MCL loss is

$$
\mathcal { L } _ { \mathrm { M C L } } = - \frac { 1 } { | \mathcal { C } | } \sum _ { i = 1 } ^ { | \mathcal { C } | } [ y _ { i } \log P _ { i } ^ { \mathrm { c o } } + ( 1 - y _ { i } ) \log ( 1 - P _ { i } ^ { \mathrm { c o } } ) ] .
$$

Although originally used for binary constraints, this logistic/BCE-form objective can directly take real-valued targets $y _ { i } \in [ 0 , 1 ]$ ], so VanillaDCC serves as a simple end-to-end DCC baseline under probabilistic supervision.

VolMaxDCC. VolMaxDCC [15] extends MCL by introducing an explicit noise-modeling mechanism through a confusion-adjusted co-clustering probability. Concretely, it replaces $P _ { i } ^ { \mathrm { c o } }$ with $\begin{array} { r } { P _ { i } ^ { \prime \mathrm { c o } } = \pmb { q } _ { a _ { i } } B \pmb { q } _ { b _ { i } } ^ { \top } } \end{array}$ , where $B$ is derived from a learnable confusion matrix, and adds a volume maximization term:

$$
{ \mathcal { L } } _ { \mathrm { V o l M a x } } = - { \frac { 1 } { | { \mathcal { C } } | } } \sum _ { i = 1 } ^ { | { \mathcal { C } } | } [ y _ { i } \log P _ { i } ^ { r \mathrm { c o } } + ( 1 - y _ { i } ) \log ( 1 - P _ { i } ^ { r \mathrm { c o } } ) ] - \lambda \log \operatorname* { d e t } ( { \mathcal { Q } } ^ { \top } { \mathcal { Q } } ) ,
$$

where $\mathcal { Q }$ is the assignment matrix. The first term allows the model to account for systematic annotation confusion, while the volume term encourages separated and distinguishable cluster assignments. We therefore include VolMaxDCC as the compared baseline with explicit noise modeling. At the same time, its design is still rooted in noisy binary supervision and assignment-level separation, whereas UPCC requires preserving a continuum of probabilistic pairwise relations rather than only correcting corrupted hard relations.

CIDEC. CIDEC [14] extends the DEC/IDEC deep embedding clustering framework [49, 52] by incorporating pairwise constraints through the MCL loss. It uses an autoencoder to obtain latent embeddings, initializes $C$ cluster anchors by K-means, and computes a soft assignment vector $\pmb q _ { j } = ( q _ { j 1 } , \dots , q _ { j C } )$ for each instance. Its unsupervised clustering term follows the DEC-style KL objective

$$
\mathcal { L } _ { \mathrm { { D E C } } } = \sum _ { j = 1 } ^ { | \mathcal { X } | } \sum _ { c = 1 } ^ { C } p _ { j c } \log \frac { p _ { j c } } { q _ { j c } } ,
$$

where $p _ { j }$ is the sharpened target distribution derived from $\mathbf { \delta } _ { \mathbf { \delta } _ { q _ { j } } . }$ . CIDEC combines this clustering loss with a reconstruction loss and the MCL pairwise constraint loss. In our setting, its MCL component uses the real-valued $y _ { i }$ directly through the same BCE-form target as VanillaDCC, making CIDEC a strong end-to-end baseline compatible with probabilistic supervision.

SpherePair. SpherePair [3] is a deep constraint embedding method that learns normalized angular representations with an autoencoder. For binary hard constraints, it defines an angular BCE-form loss

$$
\mathcal { L } _ { \mathrm { a n g } } = - \frac { 1 } { | \mathcal { C } | } \sum _ { i = 1 } ^ { | \mathcal { C } | } \left[ y _ { i } \log s _ { a _ { i } b _ { i } } ^ { + } + ( 1 - y _ { i } ) \log ( 1 - s _ { \mathrm { h a r d } , a _ { i } b _ { i } } ^ { - } ) \right] ,
$$

where $s _ { a _ { i } b _ { i } } ^ { + }$ is the must-link similarity and $s _ { \mathrm { h a r d } , a _ { i } b _ { i } } ^ { - }$ is the cannot-link transformed similarity:

$$
\begin{array} { r } { s _ { a _ { i } b _ { i } } ^ { + } = \frac { 1 } { 2 } \big ( \cos ( \theta _ { z _ { a _ { i } } , z _ { b _ { i } } } ) + 1 \big ) , \qquad s _ { \mathrm { h a r d } , a _ { i } b _ { i } } ^ { - } = \frac { 1 } { 2 } \big ( \cos ( \operatorname* { m i n } \{ \omega \theta _ { z _ { a _ { i } } , z _ { b _ { i } } } , \pi \} ) + 1 \big ) . } \end{array}
$$

Here ω controls the negative zone of angular size $\pi / \omega ;$ the SpherePair theory fixes $\omega = 2 ,$ , giving the $\pi / 2$ negative-zone boundary and the corresponding conflict-free equidistant geometry for hard must-link/cannot-link constraints. Since this original formulation is defined for binary relation types, applying SpherePair to UPCC requires a continuous counterpart for intermediate labels. We therefore keep the same $\omega = 2$ angular scale and use

$$
\begin{array} { r } { s _ { \mathrm { s o f t } , a _ { i } b _ { i } } ^ { - } = \frac { 1 } { 2 } \big ( \cos ( 2 \theta _ { z _ { a _ { i } } , z _ { b _ { i } } } ) + 1 \big ) = \cos ^ { 2 } ( \theta _ { z _ { a _ { i } } , z _ { b _ { i } } } ) } \end{array}
$$

in the BCE-form loss with each real-valued $y _ { i } \in [ 0 , 1 ]$ ]:

$$
\mathcal { L } _ { \mathrm { a n g } } ^ { \mathrm { s o f t } } = - \frac { 1 } { | \mathcal { C } | } \sum _ { i = 1 } ^ { | \mathcal { C } | } \left[ y _ { i } \log s _ { a _ { i } b _ { i } } ^ { + } + ( 1 - y _ { i } ) \log ( 1 - s _ { \mathrm { s o f t } , a _ { i } b _ { i } } ^ { - } ) \right] .
$$

Under this soft extension, the attractive endpoint $y _ { i } = 1$ is minimized at $\theta _ { z _ { a _ { i } } , z _ { b _ { i } } } = 0$ , while the fully repulsive endpoint $y _ { i } = 0$ is minimized at $\theta _ { z _ { a _ { i } } , z _ { b _ { i } } } = \pi / 2$ , matching the orthogonal negative-zone boundary selected by the original $\omega = 2 ~ $ SpherePair geometry. For intermediate labels, the preferred pairwise angle varies continuously between these two endpoints; in the single-pair case, the optimum satisfies cos $\theta _ { z _ { a _ { i } } , z _ { b _ { i } } } = y _ { i } / ( 2 - \dot { y } _ { i } )$ . SpherePair also uses the normalized-embedding reconstruction regularizer $\begin{array} { r } { \mathcal { L } _ { \mathrm { r e c } } = \frac { 1 } { | \mathcal { X } | } \sum _ { j } \| x _ { j } - \hat { x } _ { j } \| _ { 2 } ^ { 2 } } \end{array}$ with $\hat { x } _ { j } = g ( \mathrm { N o r m } ( f ( x _ { j } ) ) )$ , which is also adopted by our ProbPair, Weighted ProbPair, and ECI-PP. Thus, the SpherePair baseline in our experiments is the natural real-valued angular extension of the original hard-constraint method, while its formal geometric guarantee remains tied to the binary setting.

ProbPair and Weighted ProbPair. ProbPair is the direct probabilistic pairwise baseline introduced in Section 4.1. It uses the probabilistic angular readout in Eq. (6) and minimizes the reconstructionregularized objective $\mathcal { L }$ in Eq. (8), without our Estimator–Corrector–Integrator refinement. Weighted ProbPair further applies the information-based constraint weight $\kappa _ { i }$ in Eq. (9) to the pairwise term:

$$
{ \mathcal { L } } _ { \mathrm { W P P } } = - { \frac { 1 } { | { \mathcal { C } } | } } \sum _ { i = 1 } ^ { | { \mathcal { C } } | } \kappa _ { i } \left[ y _ { i } \log \hat { y } _ { a _ { i } b _ { i } } + ( 1 - y _ { i } ) \log ( 1 - \hat { y } _ { a _ { i } b _ { i } } ) \right] + \lambda _ { \mathrm { r e c } } { \mathcal { L } } _ { \mathrm { r e c } } .
$$

Thus, ProbPair isolates the base probabilistic angular formulation, while Weighted ProbPair isolates the effect of decisiveness-based weighting. Comparing Weighted ProbPair with ECI-PP further tests the contribution of the Estimator–Corrector–Integrator pipeline beyond static reweighting.

## C.3 Constraint generation

Here, we detail how probabilistic pairwise constraints are generated in our experiments. We use trained expert classifiers as simulated judgment channels to produce probabilistic pairwise judgments and then apply the corruption channel in Section 3.1, thereby instantiating the UPCC observation process for constraint generation.

Expert classifiers. For each dataset, simulated experts are trained on controlled labeled subsets of the training split and then used to generate predictive class distributions for queried pairs. The limited labeled-subset construction makes these classifiers expert-dependent judgment mechanisms, whose soft predictive distributions reflect both systematic bias and residual uncertainty. The subset selection rules for single- and multi-expert settings are described below, while the expert network architecture and optimization details are given in Appendix C.5.

Table 2: Unfamiliar-class counts per expert under the multi-expert settings. These counts describe the blind-spot distribution used to construct complementary experts.
<table><tr><td>Dataset type</td><td>multi2</td><td>multi3</td><td>multi10</td></tr><tr><td>20-class datasets</td><td>(10,10)</td><td>(7, 7,6)</td><td>(2, 2, 2, 2, 2, 2, 2, 2, 2, 2)</td></tr><tr><td>10-class datasets</td><td>(5,5)</td><td>(4, 3, 3)</td><td> $( 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 )$ </td></tr><tr><td>4-class datasets</td><td>(2, 2)</td><td>(2, 1, 1)</td><td> $( 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 )$ </td></tr></table>

Single-expert settings. In the single-expert setting, one expert classifier is trained per dataset. We use three expert-quality levels, denoted by lv0.1, lv0.01, and lv0.001. For lvr, the expert is trained using an r fraction of the labeled training samples from each class. The sampling is class-stratified: for every class, the number of selected samples is $\lfloor r n _ { c } \rfloor$ , where $n _ { c }$ is the number of available training samples in class c, with at least one sample retained for each class. Thus, smaller values of r produce weaker experts and hence more epistemically distorted probabilistic judgments.

Multi-expert settings. In the multi-expert setting, we construct a group of complementary experts. The three settings multi2, multi3, and multi10 use 2, 3, and 10 experts, respectively. Each expert has a set of familiar classes and a set of unfamiliar classes. For all multi-expert settings, familiar classes use a high labeled-data fraction $h = 0 . 1$ , while unfamiliar classes use a low labeled-data fraction $l = 0 . 0 \bar { 0 } 0 1$ . The class-level training fraction for expert e is therefore

$$
\rho _ { e , c } = \left\{ \begin{array} { l l } { h , } & { c \in \mathcal { F } _ { e } , } \\ { l , } & { c \in \mathcal { U } _ { e } , } \end{array} \right.
$$

where $\mathcal { F } _ { e }$ and $\mathcal { U } _ { e }$ denote the familiar and unfamiliar class sets of expert e.

The unfamiliar classes are assigned to make the experts complementary. Let the dataset contain C classes under the fixed label ordering used in expert generation. When the number of experts E satisfies $E \leq C$ , the class list is split into E consecutive blocks that are as balanced as possible. Expert e takes the e-th block as its unfamiliar classes and treats all remaining classes as familiar. Equivalently, if $C = q E + r$ with $0 \leq r < E$ , then the first r experts have $q + 1$ unfamiliar classes and the remaining experts have q unfamiliar classes. For example, on a 10-class dataset, multi2 uses unfamiliar-class counts (5, 5), multi3 uses (4, 3, 3), and multi10 gives each expert exactly one unfamiliar class. For CIFAR100-20, the corresponding counts are (10, 10), (7, 7, 6), and $( 2 , \ldots , 2 )$ For Reuters subset with four classes, multi2 gives $( 2 , 2 )$ and multi3 gives (2, 1, 1). If $E > C$ the experts are divided into full groups of size C, where each expert in a group is unfamiliar with one distinct class; any remaining experts form a partial group whose unfamiliar classes are sampled without replacement from the class list. This case occurs for Reuters subset under multi10. The resulting unfamiliar-class counts are summarized in Table 2.

Pair sampling and expert assignment. All queried pairs are sampled randomly and uniformly from the training split. In the single-expert setting, every sampled pair is assigned to the single available expert. In the multi-expert setting, constraints are generated separately for each expert, and the total constraint budget is distributed evenly across experts unless otherwise specified. For a queried pair $( x _ { a } , x _ { b } )$ assigned to expert e, the trained expert outputs predictive class distributions $\pmb { p } _ { a } ^ { e }$ and $\pmb { p } _ { b } ^ { e }$ . The clean expert judgment is computed as

$$
y _ { e , a b } ^ { \mathrm { j u d } } = \langle { p _ { a } ^ { e } , p _ { b } ^ { e } } \rangle = \sum _ { c = 1 } ^ { C } p _ { a , c } ^ { e } p _ { b , c } ^ { e } \in [ 0 , 1 ] .
$$

This value is high for similar expert-assigned class distributions, low when the expert separates the samples, and intermediate under uncertainty or overlapping class mass.

Stochastic corruption. After the clean judgment is obtained, the recorded constraint value is produced through the corruption channel in Section 3.1. For each pair, an independent corruption indi cator is drawn as $c _ { i } \sim \mathrm { B e r n o u l l i } ( \pi )$ , where π is the corruption probability used by the corresponding experiment. If $c _ { i } = 0$ , the recorded label is the clean expert judgment:

$$
y _ { i } = y _ { e _ { i } , a _ { i } b _ { i } } ^ { \mathrm { j u d } } .
$$

If ${ { c } _ { i } } = 1$ , the clean judgment is replaced by an unstructured random value:

$$
y _ { i } \sim \mathrm { U n i f o r m } ( 0 , 1 ) .
$$

Thus, the final constraints combine three factors: data-dependent soft pairwise ambiguity expressed through expert predictive distributions, structured expert-dependent epistemic distortion from classspecific familiarity, and unstructured stochastic corruption from the uniform replacement channel.

## C.4 Experimental protocol

Given the dataset details in Appendix C.1 and the constraint-generation procedure in Appendix C.3, we specify the experimental protocol for comparative evaluations, expert-aware ensemble references, held-out diagnostics, and sensitivity/ablation studies.

Comparative evaluations. For clustering-performance comparisons, we report Accuracy (ACC), Normalized Mutual Information (NMI), and Adjusted Rand Index (ARI) over five trials. For embedding-based methods, including SpherePair, ProbPair, Weighted ProbPair, and ECI-PP, the final partition is obtained by applying K-means to the learned representations, while end-to-end DCC baselines use their native clustering outputs. For the full main comparison, we use lv0.01 and multi3 with corruption probability 0.3 and total constraint budgets 3k/6k/9k; under multi3, these correspond to 1k/2k/3k constraints per expert. Robustness comparisons use a total budget of 9k constraints and vary one supervision factor at a time. For single-expert robustness, we use lv0.01 and corruption probability 0.3 as the reference setting: expert quality is varied over lv0.1/lv0.01/lv0.001 while fixing corruption probability at 0.3, and corruption probability is varied over 0.1/0.3/0.5 while fixing the expert level at lv0.01. For multi-expert robustness, we compare multi2, multi3, and multi10 specified in Appendix C.3; under the 9k total budget, these correspond to 4,500, 3,000, and 900 constraints per expert, respectively. Moving from multi2 to multi10 increases both the number of experts and the average expert accuracy (Table 4), because the complementary blind-spot rule assigns fewer unfamiliar classes to each expert (see Table 2).

Expert-aware ensemble references. Under multi-expert supervision, ECI-PP uses expert identities through its expert-specific Correctors, whereas expert-agnostic baselines treat the merged constraints as a single supervision pool. As an additional reference, we also include expert-aware ensemble variants for baselines for multi-expert comparisons. For a baseline trained under an E-expert setting, we instantiate $E$ members, each associated with one expert identity. All members are trained on the full merged constraint set $\mathcal { C } = \{ ( a _ { i } , b _ { i } , e _ { i } , y _ { i } ) \} _ { i = 1 } ^ { | \mathcal { C } | }$ , but member e applies the multiplier

$$
\alpha _ { i } ^ { ( e ) } = \left\{ \begin{array} { l l } { 1 , } & { e _ { i } = e , } \\ { \alpha , } & { e _ { i } \neq e , } \end{array} \right. \quad \alpha \in [ 0 , 1 ] ,
$$

to the pairwise loss term of each constraint. Thus, $\alpha = 0$ makes each member use only constraints from its associated expert, while $\alpha = 1$ recovers the expert-agnostic merged-constraint training for every member. More generally, if the pairwise part of a baseline objective is decomposed into per-constraint terms $\ell _ { i } ^ { \mathrm { p a i r } }$ , member e uses

$$
\mathcal { L } _ { B , \mathrm { p a i r } } ^ { ( e ) } = \frac { 1 } { | \mathcal { C } | } \sum _ { i = 1 } ^ { | \mathcal { C } | } \alpha _ { i } ^ { ( e ) } \ell _ { i } ^ { \mathrm { p a i r } } ,
$$

while all non-pairwise components of the baseline objective are kept unchanged. After training, each member produces a hard partition $\widehat S ^ { ( e ) }$ . We aggregate the members through the co-association matrix

$$
A _ { a b } ^ { \mathrm { e n s } } = \frac { 1 } { E } \sum _ { e = 1 } ^ { E } \mathbb { I } \Big [ \widehat { S } ^ { ( e ) } ( x _ { a } ) = \widehat { S } ^ { ( e ) } ( x _ { b } ) \Big ] ,
$$

then form $D _ { a b } ^ { \mathrm { e n s } } = 1 - A _ { a b } ^ { \mathrm { e n s } }$ and apply hierarchical clustering to obtain the final ensemble partition. The non-target expert weight α is selected separately for each dataset–baseline–multi-expert configuration using the reserved 1,000-sample validation split: we search $\alpha \in \{ 0 , 0 . 0 1 , 0 . 0 5 , 0 . 1 , 0 . 5 , 1 \}$ , run each candidate three times, and choose the value with the highest mean validation ACC before test evaluation. These ensemble variants give expert-agnostic baselines access to expert identities, but they also alter the training and aggregation structure through multiple expert-weighted members; they are therefore interpreted as auxiliary expert-aware references rather than fully symmetric replacements for the native baselines.

Held-out diagnostics. The held-out diagnostics examine whether the internal quantities of ECI-PP show trends consistent with their intended roles: Estimator-side beliefs, corrected relations, and integrated relations are compared with an external pairwise reference, and the post-correction gap is tested as a residual-discrepancy signal. To this end, we construct a diagnostic constraint set $\bar { \mathcal { C } } ^ { \mathrm { d i a g } }$ where $\vert \mathcal { C } ^ { \mathrm { d i a g } } \vert = 1 0 0 0$ from the test split. These constraints are sample-disjoint from the training constraints and are used only for diagnosis, not for training, early stopping, or hyperparameter selection. For a held-out pair $i = ( a _ { i } , b _ { i } )$ with ground-truth class labels $t _ { a _ { 3 } }$ and $t _ { b _ { i } }$ , we define the external hard oracle

$$
o _ { i } ^ { \star } = \mathbb { I } [ t _ { a _ { i } } = t _ { b _ { i } } ] ,
$$

which serves only as a diagnostic reference and is not identified with the canonical aleatoric relation $R _ { a _ { i } b _ { i } } ^ { \star }$ in Section 3. On $\mathcal { C } ^ { \mathrm { d i a g } }$ , which is not assigned to any cross-fitting fold, the Estimator diagnostic belief is computed by averaging the predictions of all $\dot { K }$ fold-specific Estimators; we still denote this held-out Estimator belief by ${ \widetilde { y } } _ { i } ^ { \mathrm { o o f } }$ for notational consistency. We then evaluate $\hat { y } _ { i } ^ { \mathrm { o o f } }$ , the corrected relation $y _ { i } ^ { \mathrm { c o r } }$ , and the Integrator belief $y _ { i } ^ { \mathrm { i n t } }$ against this external oracle using Brier scores. For any held-out score $v _ { i } \in [ 0 , 1 ]$ and index set $\mathcal { T } ,$ we use

$$
\operatorname { B r i e r } ( v ; \mathcal { T } ) = \frac { 1 } { | \mathcal { T } | } \sum _ { i \in \mathcal { T } } ( v _ { i } - o _ { i } ^ { \star } ) ^ { 2 } .
$$

The Estimator and Integrator diagnostics are computed on all diagnostic constraints, whereas Corrector-oracle alignment is evaluated on the clean subset ${ \mathcal { T } } _ { \mathrm { c l e a n } } = { \bar { \{ } i \in }  { \mathcal { C } } ^ { \mathrm { d i a g } } : c _ { i } = 0 \}$ . We also compute the clean post-correction discrepancy

$$
\mathrm { g a p } _ { i , \mathrm { c l e a n } } = \left| \mathrm { s o f t c l i p } _ { \xi } ( \hat { y } _ { i } ^ { \mathrm { o o f } } ) - y _ { i } ^ { \mathrm { c o r } } \right| , \qquad i \in \mathcal { T } _ { \mathrm { c l e a n } } ,
$$

to measure the residual disagreement between the Estimator belief and the corrected relation on uncorrupted held-out judgments. For corruption-screening diagnostics, the corresponding gap score is computed on all diagnostic constraints and used to rank corrupted records $( c _ { i } = 1 )$ against clean records $( c _ { i } = 0 )$ , from which we report $\mathrm { A U C _ { c o r r u p t } }$ and $\mathrm { A P _ { c o r r u p t } }$ . Because this gap captures unexplained discrepancy after correction, it may reflect both stochastic corruption and residual epistemic uncertainty, rather than stochastic corruption alone. We therefore also include an oraclesupervision diagnostic, where the clean relation is given by $o _ { i } ^ { \star }$ before applying the same corruption channel; this provides a reference case in which expert-dependent epistemic distortion is removed.

Sensitivity and ablation studies. Sensitivity and ablation studies are conducted for ECI-PP under the default single-expert setting, namely lv0.01 with corruption probability 0.3. Each study is repeated over five trials and varies one factor at a time while keeping the remaining hyperparameters and the constraint-generation protocol fixed. ProbPair and Weighted ProbPair already provide direct ablation-style references for the base probabilistic pairwise objective and the information-based weighting strategy, as described in Appendix C.2. Here, we further examine ECI-PP-specific design and sensitivity factors across its main components: the number of estimator folds $\dot { K }$ and the use of the information-based warm-up score $\kappa _ { i }$ in Eq. (9); the Corrector regularization weight $\lambda _ { \mathrm { c o r } }$ and its capacity; the reliability-screening parameters $\gamma , n _ { 0 } .$ , and soft-clipping sharpness $\xi ;$ and the reconstruction weight $\lambda _ { \mathrm { { r e c } } }$

## C.5 Implementation

We provide implementation details<sup>11</sup> for the experimental pipeline, including the computing environment, data preprocessing, expert classifiers and generated constraints, model backbones, pretraining, optimization, and hyperparameter settings.

Software and hardware. All model training code is implemented in Python 3.7 and PyTorch $1 . 5 . 1 ^ { 1 2 }$ For methods whose final prediction is obtained by K-means on learned representations, we use the K-means implementation in scikit-learn<sup>13</sup>. For the hierarchical clustering step used by the expertaware ensemble references in Appendix C.4, we use the fastcluster package<sup>14</sup>. Each training run is executed on a single NVIDIA A100 40GB or NVIDIA H200 141GB GPU.

Table 3: Test accuracy (%) of single-expert classifiers used for constraint generation.
<table><tr><td>Dataset</td><td>1v0.1</td><td>1v0.01</td><td>1v0.001</td></tr><tr><td>CIFAR100-20</td><td>73.08</td><td>65.23</td><td>46.58</td></tr><tr><td>CIFAR10</td><td>91.01</td><td>88.49</td><td>73.83</td></tr><tr><td>FMNIST</td><td>84.45</td><td>74.77</td><td>59.83</td></tr><tr><td>ImageNet10</td><td>96.27</td><td>92.62</td><td>87.62</td></tr><tr><td>MNIST</td><td>93.67</td><td>88.76</td><td>65.20</td></tr><tr><td>Reuters</td><td>92.90</td><td>83.60</td><td>49.75</td></tr><tr><td>STL10</td><td>90.04</td><td>84.31</td><td>72.77</td></tr><tr><td>RCV1-10</td><td>94.26</td><td>90.47</td><td>77.62</td></tr></table>

Table 4: Average test accuracy (%) of multi-expert groups used for constraint generation. Each value is the arithmetic mean over experts in the corresponding group.
<table><tr><td>Dataset</td><td>multi2</td><td>multi3</td><td>multi10</td></tr><tr><td>CIFAR100-20</td><td>41.23</td><td>51.90</td><td>67.36</td></tr><tr><td>CIFAR10</td><td>52.72</td><td>65.94</td><td>82.74</td></tr><tr><td>FMNIST</td><td>54.57</td><td>60.68</td><td>77.64</td></tr><tr><td>ImageNet10</td><td>73.87</td><td>76.73</td><td>90.80</td></tr><tr><td>MNIST</td><td>50.45</td><td>65.89</td><td>86.13</td></tr><tr><td>Reuters</td><td>47.58</td><td>63.02</td><td>71.42</td></tr><tr><td>STL10</td><td>53.98</td><td>65.67</td><td>83.22</td></tr><tr><td>RCV1-10</td><td>49.06</td><td>64.57</td><td>85.96</td></tr></table>

Data preprocessing. We follow the same preprocessing pipeline as [3]. For Reuters subset and RCV1-10, we directly use the preprocessed tf–idf representations over the 2,000 most frequent words or word stems, as described in Appendix C.1. For MNIST and FMNIST, the 28 28 grayscale images are flattened into 784-dimensional vectors, following the standard fully connected input format used in prior deep clustering/DCC studies [13, 14, 2, 3]. For CIFAR10, CIFAR100-20, STL10, and ImageNet10, we adopt the unsupervised feature extraction strategy in [53]: a ResNet-34 model [54] is trained for 1,000 epochs, and the resulting 512-dimensional representations are used as input features. This image preprocessing follows the setting used in DCC studies [15, 3] and keeps the inputs compatible with fully connected architectures across datasets. The same preprocessed inputs are used for expert classifier training and for all compared clustering methods.

Expert classifiers and generated constraints. The expert classifiers used in Appendix C.3 are implemented as MLP classifiers with two hidden layers of size 512–512, ReLU activations, dropout rate 0.2, and a final softmax classification layer. They are trained with cross-entropy loss using Adam, with learning rate 10<sup>−3</sup>, batch size 256, maximum 100 epochs, and early stopping patience 10 based on training accuracy. The resulting test accuracies of single experts are reported in Table 3, and the average test accuracies of multi-expert groups are reported in Table 4. These accuracies characterize the effective quality of the simulated experts produced by the subset-selection and blind-spot protocols in Appendix C.3. After the corruption channel in Section 3.1, these expert judgments yield the recorded constraint values used for training. Fig. 4 visualizes their per-dataset distributions for representative lv0.01 single-expert and multi3 multi-expert settings, both with corruption probability 0.3.

Model backbones and pretraining. For encoder-based methods, we follow the fully connected autoencoder backbone used in [13, 14, 3]: the encoder has hidden layers 500–500–2000, paired with a symmetric decoder when reconstruction is used. The embedding dimension is set to D = 10 for all datasets except CIFAR100-20, where D = 20, following [3]. VanillaDCC and VolMaxDCC use their native MLP classifier architectures with two hidden layers of size 512–512, and do not use autoencoder pretraining. For pretrain-capable methods, we use the two-stage stacked denoising autoencoder (SDAE) pretraining approach [55, 56], following the protocol used in [13, 14, 3]: hidden layers are first pretrained layer-wise as denoising autoencoders, followed by end-to-end autoencoder fine-tuning on the training split. Unless otherwise specified, all pretrain-capable methods use this pretrained initialization. For methods using angular reconstruction (SpherePair, ProbPair, Weighted ProbPair, ECI-PP), decoding is performed from normalized embeddings as in [3].

![](images/a86711ed698015b873ea41f51298fe815961c1cb6d6150b0e6bd9c455c01412c.jpg)  
Figure 4: Distributions of recorded constraint values for representative generated-supervision settings. The top row shows lv0.01 single-expert supervision and the bottom row shows multi3 multi-expert supervision, both with corruption probability 0.3.

In ECI-PP, both the K Estimator backbones and the Integrator use the same autoencoder architecture described above. Each Estimator fold has its own encoder–decoder backbone and its own probabilistic readout parameters $( m _ { k } , T _ { k } )$ , while the Integrator has its own readout parameters $( m , \bar { T } )$ in Eq. (6). These scalar readout parameters are parameterized as $m = \operatorname { t a n h } ( \tilde { m } )$ and $T = \mathrm { s o f t p l u s } ( \tilde { T } )$ for numerical stability within the cosine range, with the same parameterization used for each Estimator fold. When pretraining is used, the same pretrained autoencoder weights initialize the Integrator backbone and all Estimator backbones, while the readout parameters and Correctors are initialized separately. Unless otherwise specified, each expert-specific Corrector is an MLP with hidden layers 64–16 and a tanh output, which keeps the predicted probability-space residual within ( 1, 1).

Optimization and hyperparameters. We use the reported or recommended hyperparameters for the baseline methods whenever available. Below, we summarize the optimization settings and implementation configurations for ECI-PP and all compared baselines.

• VanillaDCC. VanillaDCC optimizes the MCL loss $\mathcal { L } _ { \mathrm { M C L } }$ using Adam with learning rate $1 0 ^ { - 3 }$ and batch size 256. Training runs for at most 500 epochs, with early stopping triggered when the relative change in the soft cluster assignments on the training samples falls below $1 0 ^ { - 3 }$ for two consecutive checks. This assignment-change stopping rule follows common practice in deep clustering and deep constrained clustering [49, 52, 13, 14].

• VolMaxDCC. For VolMaxDCC, we follow its original optimization and validation-based hyperparameter-search procedure [15]. The learnable matrix B is parameterized elementwise as $B _ { c c ^ { \prime } } = ( 1 + \exp ( - B _ { c c ^ { \prime } } ^ { \prime } ) ) ^ { - 1 }$ , where $B _ { c c ^ { \prime } } ^ { \prime }$ is initialized to 1 for $c = c ^ { \prime }$ and to 1 otherwise. Optimization uses SGD with learning rate 0.5 for the network parameters and 0.1 for $B ^ { \prime }$ , with batch size 128. The VolMaxDCC geometric regularization weight is selected from $\{ 0 , 1 0 ^ { - 1 } , 1 0 ^ { - 2 } , 1 0 ^ { - 3 } , 1 0 ^ { - 4 } , 1 0 ^ { - 5 } \}$ using the reserved validation split; for each dataset and supervision setting, we choose the value with the highest mean validation ACC over three validation runs. For multi-expert supervision, a shared selected value is used for the multi-expert settings defined in Appendix C.3, where the familiar and unfamiliar labeled-data fractions are fixed as $h = 0 . 1$ and $l = 0 . 0 0 0 1$ 1. The selected values are reported in Table 5.

• CIDEC. For CIDEC, we follow the authors’ recommended settings: the reconstruction/clustering trade-off is set to $\lambda _ { 1 } = 1$ , and the MCL constraint-balancing parameter is set to $\lambda _ { 2 } = 0 . 1$ . The C cluster anchors are initialized by K-means. Optimization uses Adam with learning rate $1 0 ^ { - 3 }$ and batch size 256. Training proceeds for at most 500 epochs, with early stopping applied when the relative change in the soft assignments falls below $1 0 ^ { - 3 }$ over consecutive checks.

• SpherePair. For SpherePair, the reconstruction weight is set to 0.02, matching the setting used in [3]. In the original binary formulation, the negative-zone factor is fixed as $\omega = 2$ which yields the theoretically guaranteed $\pi / 2$ negative-zone boundary. As described in Appendix C.2, we use the real-valued extension of SpherePair by keeping the same $\omega = 2$ angular scale and replacing the original clamped cannot-link score with the continuous counterpart $\begin{array} { r } { s _ { \mathrm { s o f t } , a _ { i } b _ { i } } ^ { - } \stackrel { - } { = } \frac { 1 } { 2 } \big ( \bar { \cos ( 2 \theta _ { z _ { a _ { i } } } } , z _ { b _ { i } } ) + 1 \big ) } \end{array}$ . The observed $y _ { i } \in [ 0 , 1 ]$ is then used directly in the BCE-form angular loss. Under this implementation, $y _ { i } = 1$ favors $\theta _ { z _ { a _ { i } } , z _ { b _ { i } } } = 0$ and $y _ { i } = 0$ favors $\theta _ { z _ { a _ { i } } , z _ { b _ { i } } } = \pi / 2$ , while intermediate labels continuously interpolate between these two angular endpoints through the two BCE terms. Optimization uses Adam with learning rate $\mathrm { \bar { 1 0 } ^ { - 3 } }$ and constraint mini-batch size 256. The instance mini-batch size for reconstruction is set according to the number of constraint mini-batches, following [3]. Training runs for at most 500 epochs, with early stopping after the first 100 epochs if the relative change in the total loss remains below 0.1 for five consecutive checks. For the expert-aware ensemble reference in Appendix C.4, the expert-aware hyperparameter α is selected on the reserved validation split; the selected values are reported in Table 6.

Table 5: Selected VolMaxDCC geometric regularization weights. Values are selected by validation ACC over three runs for each dataset and supervision setting.
<table><tr><td>Dataset</td><td> $\mathtt { 1 0 0 . 1 }$ </td><td> $\mathtt { 1 0 0 . 0 1 }$ </td><td> $_ { 1 \tt V O . 0 0 1 }$ </td><td>multi</td></tr><tr><td>CIFAR100-20</td><td> $1 0 ^ { - 3 }$ </td><td> $1 0 ^ { - 3 }$ </td><td> $1 0 ^ { - 1 }$ </td><td> $1 0 ^ { - 2 }$ </td></tr><tr><td>CIFAR10</td><td> $1 0 ^ { - 3 }$ </td><td> $1 0 ^ { - 3 }$ </td><td> $1 0 ^ { - 1 }$ </td><td> $1 0 ^ { - 2 }$ </td></tr><tr><td>FMNIST</td><td> $1 0 ^ { - 3 }$ </td><td> $1 0 ^ { - 3 }$ </td><td> $1 0 ^ { - 2 }$ </td><td> $1 0 ^ { - 2 }$ </td></tr><tr><td>ImageNet10</td><td>0</td><td> $1 0 ^ { - 4 }$ </td><td> $1 0 ^ { - 1 }$ </td><td> $1 0 ^ { - 2 }$ </td></tr><tr><td>MNIST</td><td> $1 0 ^ { - 2 }$ </td><td> $1 0 ^ { - 3 }$ </td><td> $1 0 ^ { - 2 }$ </td><td> $1 0 ^ { - 2 }$ </td></tr><tr><td>Reuters</td><td> $1 0 ^ { - 2 }$ </td><td>0</td><td> $1 0 ^ { - 1 }$ </td><td> $1 0 ^ { - 1 }$ </td></tr><tr><td>STL10</td><td> $1 0 ^ { - 3 }$ </td><td> $1 0 ^ { - 3 }$ </td><td> $1 0 ^ { - 2 }$ </td><td> $1 0 ^ { - 1 }$ </td></tr><tr><td>RCV1-10</td><td> $1 0 ^ { - 4 }$ </td><td> $1 0 ^ { - 4 }$ </td><td> $1 0 ^ { - 5 }$ </td><td> $1 0 ^ { - 3 }$ </td></tr></table>

Table 6: Selected expert-aware ensemble hyperparameter α. Values are chosen by mean validation ACC over three runs and shared across the multi-expert configurations defined in Appendix C.3.
<table><tr><td>Dataset</td><td>SpherePair</td><td>Weighted ProbPair</td></tr><tr><td>CIFAR100-20</td><td>0.05</td><td>0.05</td></tr><tr><td>CIFAR10</td><td>1.00</td><td>0.05</td></tr><tr><td>FMNIST</td><td>1.00</td><td>0.50</td></tr><tr><td>ImageNet10</td><td>0.00</td><td>0.10</td></tr><tr><td>MNIST</td><td>0.05</td><td>0.50</td></tr><tr><td>Reuters</td><td>0.05</td><td>0.05</td></tr><tr><td>STL10</td><td>0.01</td><td>0.01</td></tr><tr><td>RCV1-10</td><td>0.01</td><td>0.05</td></tr></table>

• ProbPair variants and ECI-PP (Ours). ProbPair, Weighted ProbPair, and ECI-PP use the ProbPair-style readout in Eq. (6) and the reconstruction weight $\lambda _ { \mathrm { r e c } } = 0 . 0 2$ . ProbPair optimizes Eq. (8), while Weighted ProbPair additionally applies the information-based weight $\kappa _ { i }$ in Eq. (9). For the expert-aware ensemble reference based on Weighted ProbPair in Appendix C.4, the expert-aware hyperparameter α is selected on the reserved validation split; the selected values are reported in Table 6. The shared base optimizer is Adam with learning rate $1 0 ^ { - 3 }$ and batch size 256. The probabilistic readout parameters, including $( m , T )$ and the fold-wise $( m _ { k } , T _ { k } )$ when present, are initialized as (0, 0.1) and optimized with learning rate $1 0 ^ { - 2 }$ . We use a fixed outer training budget of 500 epochs; principled stopping criteria for iterative refinement remain an open question for future study. For any weighted ProbPair-style objective, including Weighted ProbPair and the weighted Estimator/Integrator updates in ECI-PP, the weight is applied inside the constraint average:

$$
\mathcal { L } _ { \mathrm { P P } } ^ { w } = - \frac { 1 } { | \mathcal { C } | } \sum _ { i = 1 } ^ { | \mathcal { C } | } w _ { i } \Bigl [ \tilde { y } _ { i } \log \hat { y } _ { a _ { i } b _ { i } } + ( 1 - \tilde { y } _ { i } ) \log ( 1 - \hat { y } _ { a _ { i } b _ { i } } ) \Bigr ] ,
$$

where $\tilde { y } _ { i }$ denotes the corresponding training target, such as $y _ { i } , y _ { i } ^ { \mathrm { i n t } }$ , or $y _ { i } ^ { \mathrm { B C } }$

For ECI-PP, we use one global hyperparameter setting unless explicitly varied in sensitivity or ablation studies: $K = 5$ estimator folds, $\lambda _ { \mathrm { c o r } } = 0 . 5 , \gamma = 1 0 , n _ { 0 } = 1 0$ , soft-clipping sharpness $\xi = 2 0$ , and information-based warm-up enabled. The default warm-up trains the Estimators and Integrator for 50 epochs before iterative refinement. The initial yˆ<sup>oof</sup><sub>i</sub> is obtained by strict fold-heldout prediction, and later refinement rounds use refreshed Estimator beliefs under the Integrator-updated targets while retaining the same fold structure. Each Corrector update uses the shared base optimizer setting, with the Huber quadraticto-linear transition in Eq. (10) fixed at 0.1, holds out 10% of its assigned constraints for internal validation, runs for at most 50 epochs, and stops early if the validation loss does not improve for 10 consecutive epochs. Between refinement iterations, Gaussian noise with standard deviation 0.01 is added to the final linear layers of the Estimators, Correctors, and Integrator to mildly perturb the next refinement step.

Notably, the validation-based selections of the VolMaxDCC geometric regularization weight and the expert-aware ensemble hyperparameter α rely on ground-truth class labels on the validation split, which are typically unavailable in constrained clustering and are not part of the observable supervision under UPCC. In contrast, ECI-PP uses one global hyperparameter setting across datasets and handles expert identities directly through expert-specific Correctors, without datasetspecific tuning or an additional hyperparameterized ensemble wrapper.

## D Additional experimental results

## D.1 Comparison under different constraint budgets

Tables 7 and 8 report the full main-comparison results under the default lv0.01 single-expert and multi3 multi-expert regimes, with corruption probability fixed at 0.3. The total constraint budget is varied over 3k/6k/9k; under multi3, these correspond to 1k/2k/3k constraints per expert. All results in this subsection use pre-trained backbones for methods with a pretraining stage, while VanillaDCC and VolMaxDCC have no pretraining variant.

The main-text conclusion remains stable across budgets. Across the 144 test entries in the two tables, ECI-PP ranks first in 122/144 entries and within the top two in 134/144, while one ProbPair-family method is best in 137/144 entries. The budget-wise pattern is not completely uniform, as expected: with only 3k constraints, all methods are less stable and the separation between methods is smaller, whereas at 9k several strong baselines also improve and narrow some ACC gaps. Even under these effects, ECI-PP remains the leading method at every budget, ranking first in 38/48, 43/48, and 41/48 entries under 3k, 6k, and 9k constraints, respectively.

The advantage is clearest on structure-sensitive metrics. ECI-PP gives the best NMI in 46/48 entries and the best ARI in 43/48, compared with 33/48 for ACC. Since NMI and ARI better reflect grouping consistency than matched-label accuracy alone, this suggests that ECI-PP most reliably improves the learned clustering structure. For example, on single-expert CIFAR100, ECI-PP raises NMI from roughly 44–46% for the strongest non-ECI competitors to 47–50% across budgets; on multi-expert Reuters, it raises NMI from about 53–58% for Weighted ProbPair to 65–69%. Dataset-wise, ECI-PP is best in all entries on CIFAR100, Reuters, and STL10, and in nearly all entries on CIFAR10 and ImageNet10. FMNIST and MNIST show more competition from Weighted ProbPair or SpherePair, especially on ACC and low-budget MNIST, but ECI-PP remains strongest or near-strongest on NMI/ARI in most cases.

The ProbPair variants show a consistent ablation pattern. Weighted ProbPair improves over ProbPair in most entries, and this effect is stronger under multi-expert supervision (67/72 entries) than under single-expert supervision (62/72 entries). This supports the interpretation that information-based weighting is especially useful when low-decisiveness constraints may reflect expert unfamiliarity. ECI-PP further improves over Weighted ProbPair in 129/144 entries, showing that weighting alone does not replace explicit estimation, correction, and integration.

Notably, VolMaxDCC also provides a useful noise-aware reference among the non-ProbPair baselines. It improves over VanillaDCC and is competitive on some datasets, but reaches the top two in only 15/144 entries. This suggests that explicit noise modeling is beneficial, yet its hard-noise-oriented formulation is not sufficient for the expert-conditioned probabilistic supervision in UPCC.

The main caveat remains the severely imbalanced RCV1-10 benchmark. SpherePair remains the main ACC competitor there, consistent with the advantage of its angular geometry for imbalanced pair structures. Nevertheless, ECI-PP gives the best NMI across all RCV1-10 budgets and regimes, and is often strongest in ARI. Thus, ECI-PP corrects much of the weakness of plain ProbPair/Weighted ProbPair on this benchmark, while the ACC results indicate that severe class imbalance remains a limitation for the probabilistic formulation.

Table 7: Comparative performance (%) (ACC, NMI, ARI) across datasets for methods under the lv0.01 setting, with corruption probability 0.3 and 3k/6k/9k constraints. Blue and black represent training and test results, respectively. Best results are in bold, and second-best are underlined.
<table><tr><td rowspan=1 colspan=10></td></tr><tr><td rowspan=1 colspan=10></td></tr><tr><td rowspan=2 colspan=10>16.6                             46.1,    48.3,,48.551.2,,51.16kNMI 12.6,, 13.543.9,44.5 15.0,15.1 43.8,44.9 42.4,43.443.7,44.548.3,,48.8ARI 5.4, 5.6 28.2,, 28.2 3.4,2.9  31.5,32.0 28.7, 29.030.2, 30.534.7, 34.7ACC 16.8, 17.046.1,45.9 14.3,13.6 50.2,,50.5 49.4,49.550.3,50.852.7,52.99kNMIARI</td></tr><tr><td rowspan=1 colspan=8></td></tr><tr><td rowspan=1 colspan=2>3kCIFAR10</td><td rowspan=1 colspan=7>ACC 34.6,34.679.9,79.8 54.1.52.7 84.6,84.9 84.8,85.0NMI27.9,28.169.3,69.440.8,40.7 74.3,74.7 74.9,75.3ARI 20.2,20.364.4,64.423.0,20.5 71.7,72.0 71.9,72.1</td><td rowspan=1 colspan=1>85.8, 85.785.5,85.476.0, 76.077.0,76.973.4,73.273.5,73.3</td></tr><tr><td rowspan=1 colspan=2>6k</td><td rowspan=1 colspan=7>ACC 35.5,35.784.7,84.651.3,51.0 84.8,85.3 86.0,86.0NMI20..28.373.6,73.739.8,41.1 73.1,74.2 75.6,75.9ARI21.271.0,70.922.9,22.871.5,, 72.373.5,73.5</td><td rowspan=1 colspan=1>86.5,86.587.3,87.276.376.578.6,,78.774.374.276.0,76.0</td></tr><tr><td rowspan=1 colspan=2>9k</td><td rowspan=1 colspan=3>ACC 38.3,38.782.5,82.5NMI29.9,31.271.9,72.2ARI 23.2,24.068.3,, 68.3</td><td rowspan=1 colspan=4>44.3,44.0 84.9,85.7 85.9,86.036.1,37.6 72.4,74.1 75.0,75.421.1,21.3 71.2,72.5 73.0,73.2</td><td rowspan=1 colspan=1>86.6,86.787.8, 87.876.2,76.879.0,79.174.374.576.7,, 76.8</td></tr><tr><td rowspan=1 colspan=2>3k</td><td rowspan=1 colspan=3>ACC36.6,36.563.3,62.5NMI     31.155.5,55.2</td><td rowspan=1 colspan=4>35.4,34.4 71.1,70.0 70.5,,69.531.6,31.361.0, 60.761.0, 60.6</td><td rowspan=2 colspan=1>71.0,, 70.070.2,, 69.261.3,61.064.0, 63.453.1,52.153.8, 52.6</td></tr><tr><td rowspan=1 colspan=1>FMNIST</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>ARI 19.7, 19.544.4,, 43.6</td><td rowspan=1 colspan=2>13.5,12.453.252.0</td><td rowspan=1 colspan=2>52.1, 51.0</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>6k</td><td></td><td></td><td></td><td></td><td></td><td rowspan=1 colspan=2>72.7,71.562.4,62.1</td><td rowspan=2 colspan=1>73.6,72.571.3,70.263.1,62.965.3, 64.554.6,53.555.1, 53.8</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>ARI 20.7,20.9</td><td rowspan=1 colspan=1>46.6,45.5</td><td rowspan=1 colspan=2>12.2,11.9 53.2,52.1</td><td rowspan=1 colspan=2>53.9,52.6</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>9k</td><td rowspan=1 colspan=2>ACC 39.8,40.2NMI 30.1,31.3</td><td rowspan=1 colspan=1>66.5,65.557.2,56.8</td><td rowspan=1 colspan=2>36.1.35.7 72.0,71.229.7,30.5 60.4,60.9</td><td rowspan=1 colspan=2>73.6,72.562.9,62.9</td><td rowspan=2 colspan=1>74.6,73.672.4,71.163.9,64.065.7, 65.054.355.5,53.9</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>ARI 21.5,22..2</td><td rowspan=1 colspan=1>46.4,45.2</td><td rowspan=1 colspan=2>15.7.15.6 53.0,52.3</td><td rowspan=1 colspan=2>54.8,53.7</td><td rowspan=1 colspan=1>55.3,54</td></tr><tr><td rowspan=3 colspan=1></td><td rowspan=3 colspan=1>3k</td><td rowspan=3 colspan=2>ACC35.4,37.3NMI27.7,30.8</td><td rowspan=1 colspan=1>86.7,86.9</td><td rowspan=1 colspan=1>69.0,71.2</td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=2 colspan=1>78.8,79.4</td><td rowspan=2 colspan=1>53.9, 58.7</td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=1 colspan=1>88.0,89.278.6, 81.4</td><td rowspan=1 colspan=2>89.6,89.881.8,, 82.3</td><td rowspan=2 colspan=1>90.0,90.1 92.2,,92.582.883.387.7,, 88.180.5,80.585.0, 85.4</td></tr><tr><td rowspan=1 colspan=1>ImageNet10</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>ARI 19.120.874.1,73.9</td><td rowspan=1 colspan=1>43.4,47.1</td><td rowspan=1 colspan=1>76.8,78.9</td><td rowspan=1 colspan=2>79.4,79.6</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>6k</td><td rowspan=1 colspan=2>ACC40.6,42.0</td><td rowspan=1 colspan=1>87.2,87.5</td><td rowspan=1 colspan=1>67.0,71.2</td><td rowspan=1 colspan=1>87.2,89.3</td><td rowspan=1 colspan=2>89.7,90.5</td><td rowspan=2 colspan=1>89.7,90.292.7, 92.981.6,82.888.3, 88.579.5,80.085.9, 86.1</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>ARI 23.2,26.0</td><td rowspan=1 colspan=1>75.1,75.6</td><td rowspan=1 colspan=1>40.2,46.3</td><td rowspan=1 colspan=1>74.7,78.7</td><td rowspan=1 colspan=2>79.3,80.7</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>ACC 39.0,41.5</td><td rowspan=1 colspan=1>87.4,88.1</td><td rowspan=1 colspan=1>63.5,68.3</td><td rowspan=1 colspan=1>85.1,89.1</td><td rowspan=1 colspan=2>89.2,90.5</td><td rowspan=1 colspan=1>89.1,90.392.8, 93.1</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>9k</td><td rowspan=1 colspan=3>NMI 27.6,33.777.8,79.2ARI 21.1,25..275.2,76.1</td><td rowspan=1 colspan=4>47.2,57.5 71.0,79.2 80.183.235.5,42.2 70.7,78.2 78.5,80.7</td><td rowspan=1 colspan=1>80.3,82.788.1, 88.578.4, 80.285.8, 86.2</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>3k</td><td rowspan=1 colspan=3>ACC25.9,26.165.6,66.4NMI 16.0,17.052.7,54.1</td><td rowspan=1 colspan=1>59.1,56.551.9,51.0</td><td rowspan=1 colspan=3>87.3,87.9 86.3,86.678.1,79.375.5,76.5</td><td rowspan=2 colspan=1>89.6,90.284.2, 84.678.4, 79.775.0,76.178.5,79.772.3,73.1</td></tr><tr><td rowspan=1 colspan=1>MNIST</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>ARI 10.4,10.847.8,48.9</td><td rowspan=1 colspan=1>30.3,26.7</td><td rowspan=1 colspan=3>76.8,77.874.3,74.9</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>6k</td><td rowspan=1 colspan=3>ACC33.333.973.4,74.4NMI24.2,, 25.858.4,, 60.5</td><td rowspan=2 colspan=2>89.3,90.178.2,80.229.8,26.4 78.2,79.9</td><td rowspan=1 colspan=2>52.2,57.0</td><td rowspan=2 colspan=1>88.0,89.275.8,78.175.8,78.1</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>ARI 18.4,19.455.2,56.8</td><td></td><td></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>9k</td><td rowspan=2 colspan=3>ACC 38.9,, 40.375.6,76.9NMI29.5,32.259.3,61.6ARI 23..1,24.856.5,58.7</td><td rowspan=2 colspan=2>47.0,46.7 87.8,89.141.4,43.7 75.4,78.024.7,24.2 75.4,77.8</td><td rowspan=2 colspan=2>89.0,,90.076.9,79.077.4,79.4</td><td rowspan=2 colspan=1>89.9,90.890.7, 91.078.3,80.280.7, 81.679.1,80.980.8, 81.5</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>3k</td><td rowspan=1 colspan=1>ACCNMI</td><td rowspan=1 colspan=2>53.2,54.062.8,, 63.215.1,17.925.6,28.0</td><td rowspan=1 colspan=1>74.4,77.042.8,49.8</td><td rowspan=1 colspan=1>81.4,83.454.5,58.9</td><td rowspan=1 colspan=2>79.6,81.355.4,58.5</td><td rowspan=2 colspan=1>82.8,84.486.0,86.8261.663.8, 65.666.169.4, 71.0</td></tr><tr><td rowspan=1 colspan=1>REUTERS</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>ARI</td><td rowspan=1 colspan=1>17.0,18.0</td><td rowspan=1 colspan=1>31.2,32.4</td><td rowspan=1 colspan=1>50.4,56.7</td><td rowspan=1 colspan=1>59.8,64.4</td><td rowspan=1 colspan=2>58.9,62.3</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>6k</td><td rowspan=1 colspan=1>ACCNMI</td><td rowspan=1 colspan=1>66.0,70.627.7,36.3</td><td rowspan=1 colspan=1>72.5,76.836.0,44.9</td><td rowspan=2 colspan=2>74.2,79.3 80.1,83.238.2,48.8 52.0,58.544.6,55.6 56.8, 63.2</td><td rowspan=2 colspan=2>79.6,81.555.0,58.958.1,62.0</td><td rowspan=2 colspan=1>82.283.985.4, 86.456.7,60.462.4, 65.061.5,65.367.8, 70.3</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>ARI</td><td rowspan=1 colspan=1>34.0,42.6</td><td rowspan=1 colspan=1>42.4,50.8</td></tr><tr><td rowspan=2 colspan=1></td><td></td><td></td><td></td><td></td><td rowspan=2 colspan=4>72.9,79.3 77.9,81.878.2,80.835.7,48.1 48.7,56.353.2,58.341.8,54.9 52.9,61.1 55.5,61.0</td><td rowspan=2 colspan=1>80.4,82.785.1, 86.654.5,59.261.4, 64.458.6,63.466.6, 70.0</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>ARI 36.8,50.4</td><td rowspan=1 colspan=1>40.9,50.4</td></tr><tr><td rowspan=1 colspan=2>3k</td><td rowspan=1 colspan=3>ACC 30.9,31.572.8,73.3NMI21.123.464.8,66.5</td><td rowspan=1 colspan=4>66.8,69.4 80.7,81.7 81.9,83.049.9,55.1 66.7,69.1 69.7,72.0</td><td rowspan=2 colspan=1>83.3,83.985.0, 85.371.573.075.3,75.968.8,69.872.0, 72.5</td></tr><tr><td rowspan=1 colspan=1>STL10</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>ARI 14.115.056.3,57.0</td><td rowspan=1 colspan=4>40.5,44.8 64.2,66.1 66.3,68.2</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>6k</td><td rowspan=1 colspan=3>ACC30.4,31.676.8,77.5NMI 19.5,22.265.8,67.7</td><td rowspan=1 colspan=4>62.7,66.7 79.0,81.7 81.6,82.944.4, 51.463.2,68.2 68.7,71.6</td><td rowspan=1 colspan=1>82.8</td></tr><tr><td rowspan=2 colspan=2>9k</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>ARI 14.3,15.760.0,61.3</td><td rowspan=1 colspan=4>34.1, 40.061.2,65.5 65.9, 68.2</td><td></td></tr><tr><td rowspan=1 colspan=3>ACC 35.5,37.879.0,80.1NMI 22.9,27.666.7,69.3ARI 17.119.862.8,64.7</td><td rowspan=1 colspan=4>57.8,63.3 76.1,79.7 81.6,83.439.3,48.5 59.5,66.5 68.2,72.129.8,37.357.2,63.3 65.8,68.9</td><td rowspan=1 colspan=1>81.6,83.185.4,85.867.9,71.675.8, 76.665.8,, 68.472.8, 73.4</td></tr><tr><td rowspan=1 colspan=2>3k</td><td rowspan=2 colspan=3>ACC40.0,40.1 46.346.4NMI 12.3,12.421.5,21.9ARI 16.4,, 16.521.2,21.4</td><td rowspan=2 colspan=4>47.4,48.7 65.1,64.9 54.0,53.833.3,32.5 59.259.6 54.2,54.428.6,29.1 56.1,56.2 43.3,43.1</td><td rowspan=2 colspan=1>52.6,52.461.7.61.755.1,55.360.6, 60.843.5, 43.453.3, 53.5</td></tr><tr><td rowspan=1 colspan=2>RCV1-10</td></tr><tr><td rowspan=1 colspan=2>6k</td><td rowspan=1 colspan=3>ACC 52.4,,52.662.1,62.4NMI21.8,22236.0,36.7</td><td rowspan=2 colspan=4>42.3.,43.5 70.8,70.7 51.5,51.330.5,30.5 60.6,61.1 54.1, 54.224.9,    61.6, 61.842.3,,42.1</td><td rowspan=2 colspan=1>54.0,53.963.7,63.757.5,57.862.9,63.046.2,, 46.155.956.1</td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=3>ARI 28.0,28.242.9.43.3</td><td rowspan=1 colspan=3>24.9, 26.7 61.6, 61.8 42.3, 4</td></tr><tr><td rowspan=1 colspan=2>9k</td><td rowspan=1 colspan=8>ACC46.2,46.461.3,61.8NMI 15.5,15.935.7,36.729.5,30.060.561.3 53.8,54.157.4,57.565.5,65.8ARI 19.8,20.144.7,45.426.2,28.3 61.0,61.4 43.6,43.747.1,47.062.0,62.0</td></tr></table>

Table 8: Comparative performance (%) (ACC, NMI, ARI) across datasets for methods under the multi3 setting, with corruption probability 0.3 and 3k/6k/9k total constraints. Blue and black represent training and test results, respectively. Best results are in bold, and second-best are underlined.
<table><tr><td colspan="2"></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="3"></td><td></td><td></td><td></td><td></td><td>42.4, 42.4</td><td></td><td></td><td>49.1,49.3</td></tr><tr><td>6k NMI ARI</td><td>11.3, 12.2 4.6, 4.7</td><td>44.3, 44.5 27.7 27.5</td><td>16.6 16.2, 16.3 3.2, 2.5</td><td>40.0, 40.9 26.2, 26.5</td><td>42.0, 42.1 39.8, 40.6 25.2, 25.4</td><td>44.1, 44.4 41.1, 41.9 26.4,26.7</td><td>47.1,47.7 32.1, 32.4</td></tr><tr><td>ACC 9k NMI ARI</td><td>15.4, 15.8 12.8, 13.6 5.8, , 5.8</td><td>42.8, 42.5 43.4, 43.8 27.1, 27.1</td><td>12.8, 12.1 10.0, 10.5 2.1, 2.0</td><td>44.3, 44.4 40.5, 41.5 27.1, 27.6</td><td>43.3, 43.4 40.0, 40.9 26.0, 26.2</td><td>46.1, 46.5 41.8, 43.0 28.0, 28.5</td><td>48.6, 48.8 48.1, 48.5 32.9, 32.9</td></tr><tr><td rowspan="3">CIFAR10</td><td>ACC 3k NMI ARI</td><td>28.8, 28.8 25.2, 25.5 16.5, 16.6</td><td>76.0, 75.8 71.0, 70.9 63.3, 63.0</td><td>38.4, 36.8 32.3, 31.7 13.4, 11.9</td><td>74.8, 75.1 69.5, 70.2 63.0, 63.5</td><td>79.8, 79.8 72.0, 72.1 67.0, 66.8</td><td>82.0, 82.1 73.2, 73.3 68.7, 68.7</td><td>82.8, 82.5 76.0, 75.9 70.8, 70.7 ,84.0</td></tr><tr><td>ACC 6k NMI ARI ACC</td><td>33.6, 33.9 30.9, 32.0 22.4, 23.1</td><td>78.0, 78.1 72.6, 72.9 65.7, 65.8</td><td>36.5, 34.9 31.9, 31.9 14.2, 14.0 33.2,</td><td>76.5, 76.8 66.9, , 68.1 61.7, , 62.5</td><td>80.7, 81.0 71.2, 71.6 66.2, 66.6</td><td>80.9, 81.0 71.9, 72.2 67.8, 67.9</td><td>84.0, 77.0, 77.1 72.6, 72.6</td></tr><tr><td>9k NMI ARI ACC 3k NMI</td><td>32.4, 32.8 29.1, 30.9 21.2, 22.4 33.7, 33.7 32.9, 33.0</td><td>76.5, 76.6 73.1, 73.2 65.8, 65.9 58.1, 57.7 56.7,</td><td>31.6 28.7. 29.1 12.3, 12.1 30.1, 29.6 29.8,</td><td>73.8, 74.5 64.7, 66.5 59.0, 60.5 61.3, 61.0</td><td>81.3, 81.7 71.5, 72.3 67.1, 67.8 60.9,60.3</td><td>81.4, 81.7 72.0, 72.6 67.9, 68.4 62.6, 62.5</td><td>81.6, 82.1 77.8,77.8 73.1, 73.0 60.1,59.2</td></tr><tr><td rowspan="3">FMNIST ImageNet10</td><td>ARI ACC 6k NMI</td><td>20.7, 20.6 34.2, 34.3 32.1, 32.8</td><td>56.4 42.7, , 42.1 58.2, 57.9 58.3, 58.1</td><td>30.1 14.1, 14.0 30.4, 30.1 28.1, 29.6</td><td>58.1, 58.2 45.9, 45.6 65.4, 65.3 58.9, 59.7</td><td>59.2, 58.8 46.4, 45.8 64.1, 63.8 59.6, 59.7</td><td>59.1, 59.1 46.4, , 46.2 66.1, 66.0 60.1, 60.3</td><td>62.7, 61.9 49.5, 48.4 62.0, 61.1 64.3, 63.8</td></tr><tr><td>ARI ACC 9k NMI ARI</td><td>21.0, 21.2 35.2, 35.6 31.4, 32.8</td><td>44.4, 43.7 59.4, 59.1 59.2, 59.0</td><td>14.2, , 14.7 28.6, 28.8 28.3, 30.6</td><td>48.7, 48.9 61.0,61.3 56.6,57.8</td><td>47.8, 47.7 66.6, 66.7 60.7, 61.2</td><td>49.6, 49.5 68.0, 67.6 61.7 62.0</td><td>51.0, 51.0 65.3,64.8 66.0, 65.6</td></tr><tr><td>ACC 3k NMI ARI</td><td>20.4, 33.9, 29.0, 19.4,</td><td>21.3 45.0, 44.3 34.7 88.8, 88.9 31.4 85.9, 85.8</td><td>14.5, 15.4 50.7, 49.2 41.2, , 43.3 21.7,</td><td>46.8, 47.4 81.0, 81.5 72.7, 74.2</td><td>49.9, 50.0 87.8, 88.4 78.8, 80.3</td><td>51.9, 51.8 89.5, 89.7 81.6, 82.3</td><td>53.4, 52.8 91.5, 91.8 86.9, 87.3</td></tr><tr><td rowspan="3">MNIST</td><td>ACC 6k NMI</td><td>20.7 32.7, 34.6 25.3, 30.1</td><td>81.9, 81.7 87.6, 88.1 86.5, 87.1</td><td>20.4 43.0, 41.6 35.4, 38.5</td><td>68.2, 69.1 81.8, 83.7 69.2, 73.5</td><td>76.7, 77.7 88.6, 89.4 78.3, 80.4</td><td>79.7,79.9 89.6, ,90.1 80.2, 81.6</td><td>84.1, 84.4 91.9, 92.1 87.2,87.8</td></tr><tr><td>ARI ACC 9k NMI</td><td>17.9, 20.9 31.8, 34.2 22.0, 26.8</td><td>81.9, 82.4 89.9, 90.5 87.8, 88.7</td><td>17.3, 17.9 39.4, 38.8 31.0, 34.8</td><td>66.3, 70.0 80.7, 85.0 65.6, 73.8</td><td>77.2, 78.6 87.4, 89.2 76.1, 80.1</td><td>79.3, 80.1 90.1, 91.3</td><td>84.6, 85.0 92.4, 92.6 87.7,</td></tr><tr><td>ARI ACC 3k NMI</td><td>15.8, 18.7 27.0, 27.3 19.2,</td><td>83.9, , 84.9 54.6, 55.4</td><td>14.3, 15.6 44.9,41.8 45.6, 44.1</td><td>63.4, 71.0 83.6, 84.4</td><td>75.0, 78.3 81.2, 82.2</td><td>80.1, 83.0 79.8, 82.0 86.5, 87.2</td><td>, 88.1 85.4, 85.7 80.9, ,81.5</td></tr><tr><td rowspan="3">REUTERS</td><td>ARI ACC 6k</td><td>20.5 11.5, 12. .2 28.1 29.0</td><td>48.5, 49.8 38.6, 39.5 62.5, 63.3</td><td>20.4, 17.8 39.8, 38.1</td><td>75.2 76.9 71.5, 72.4 78.8, 79.9</td><td>73.0, 74.7 69.4, 70.9 81.0, 82.3</td><td>75.2, 76.5 73.9, ,74.9 86.5, ,87.4</td><td>75.7, , 77.1 71.9, , 73.2 86.1. 86.4</td></tr><tr><td>NMI ARI ACC</td><td>22.0, 23.8 14.7, 15.7 32.2, 33.3</td><td>55.1, 56.6 47.1, 48.4 67.1, 67.8</td><td>39.2, 39.3 16.8, 15.4 31.8, 31.0</td><td>72.8, 75.3 68.2, 70.3 78.6,80.0</td><td>72.2, 74.4 68.7, 70.8 80.5, 81.8</td><td>75.3, 76.9 73.9 75.4 85.1, , 86.1</td><td>79.8, 80.5 77.8, 78.3 84.7, 85.1</td></tr><tr><td>9k NMI ARI ACC 3k NMI</td><td>26.5, 29.3 18.4, 20.1 42.8, , 41.4</td><td>57.5, 58.9 50.9, 52.1 66.9, , 67.3</td><td>28.8, 30.2 13.2, 13.4 61.4,62.1</td><td>70.8, 74.0 66.3, 69.1 72.8, , 75.2</td><td>72.1, 74.7 68.7, 71.2 78.6, 79.9</td><td>75.0, 77.2 73.4, 75.5 75.7, 78.0</td><td>79.9, , 80.7 77.1, 77.5 85.0, 86.3</td></tr><tr><td rowspan="3"></td><td>ARI ACC 6k</td><td>3. .2,3.7 4.7,4.0 51.4, 52.9</td><td>40.8, 42.5 42.8, 43.9 67.0, 68.2</td><td>26.3, 32.2 28.1, 28.9 58.7, 60.4</td><td>40.2, 44.6 45.2, 50.1 66.7, 72.9</td><td>48.0, 50.3 55.0 57.5 78.6, 81.4</td><td>49.8, 53.4 53.8, 58.1 81.0, 83.4</td><td>62.0, 65.0 68.3, 71.1 87.3, 88.5</td></tr><tr><td>NMI ARI ACC</td><td>11.8, 15.9 14.0, 16.4 55.3, 59.2</td><td>44.3, 46.5 46.5, , 48.7 67.5, 68.0</td><td>19.8, 27.0 22.6, 25.1 57.3, 59.1</td><td>32.8, 43.2 37.2, 48.2 62.1, 69.9</td><td>48.3, 54.0 54.5, 60.3 74.5, 79.0</td><td>52.9, 57.7 58.8, 63.6 81.0, 83.9</td><td>63.8, 67.2 71.3, 74.3 88.3, 89.6</td></tr><tr><td>9k NMI ARI ACC</td><td>14.7, 22.8 19.2, 26.2 25.6, 26.4</td><td>45.4, 46.5 47.0, 47.9 74.6, 74.8</td><td>18.3, 25.7 20.4, 23.1 47.4, 47.4</td><td>25.6, 37.2 28.0, 40.7 72.7, 73.4</td><td>44.6, 52.8 50.2, 58.6 79.2, 79.9</td><td>51.5 57.9 57.9 64.2 80.4, 80.8</td><td>65.4, 68.7 72.9, 76.0 85.7, 85.9</td></tr><tr><td rowspan="4">STL10</td><td>3k NMI ARI ACC</td><td>19. .7, 21.7 12.3, 13.1 29.8,</td><td>71.5 72.0 62.3, 62.8</td><td>38.3, 41.5 21.8, 22.4 37.6, 37.2</td><td>61.7, 64.0 55.0, 56.4 67.7,69.8</td><td>66.5, 68.4 61.6, 62.9 81.0, ,82.0</td><td>68.9, 70.0 64.7, 65.3 83.1, 84.2</td><td>75.7, 76.3 72.2, , 72.5 87.4,87.5</td></tr><tr><td>6k NMI ARI</td><td>31.2 22. .3, ,26.4 15.5, 17.7</td><td>72.7, 72.6 71 71.2 60.6</td><td>31.2, 37.2 15.8, 18.2</td><td>54.8, 58.8 48.6, 51.7</td><td>67.4, 69.9 64.0,65.7</td><td>69.5, 72.0 67.4, 69.4</td><td>77.5,78.1 75.1, 75.4 87.9,87.7</td></tr><tr><td>ACC 9k NMI ARI</td><td>29.2, 30.7 21.5, 26.0 15.2, 17.7</td><td>71.0, 71.1 70.2 69.9 59.5, 59.1</td><td>31.4, 31.6 24.9, 31.7 13.6, 16.9</td><td>67.8, 71.4 53.2, 59.7 47.8, 53.3</td><td>81.1,83.0 66.1, 70.2 63.7, 67.1</td><td>83.1, 84.3 69.4, 72.0 67.4, 69.4</td><td>78.3, 78.5 75.9,75.8 55.3, 55.2</td></tr><tr><td>ACC 3k NMI ARI</td><td>29.2 29.2 3.2 2.6 3.1</td><td>31.7, 31.9 23.0, 23.6 10.8, 11.1</td><td>33.5, 34.9 19.1, 18.3 14.3, 14.6</td><td>51 51.0 51.2, 51.7 39.5, 39.6</td><td>47.9, 47.8 50.0, 50.3 37.8, 37.9</td><td>47.0, 46.9 51.4 51.7 37.3, 37.4</td><td>56.8, 56.9 45.6, 45.4</td></tr><tr><td rowspan="4">RCV1-10</td><td>ACC 6k NMI</td><td>43.2,43.4 13.7, , 14.0</td><td>46.3, 46.6 30.2, 30.7</td><td>31.8, 32.0 13.5, 13.2</td><td>51.2, 51.4 51.7</td><td>46.6,46.6 48.1, 48.5</td><td>46.9, 46.8 51 .6, 51 1.8</td><td>55.2, 55.0 58.4, 58.6</td></tr><tr><td>ARI 9k</td><td>16.6, 16.7</td><td>26.3, 26.6</td><td>9.8,9.7</td><td>40.9. 41.1 55.7</td><td>35.7, 35.8</td><td>38.0, 37.9 47.9</td><td>46.8, 46.8 53.3</td></tr><tr><td>ACC NMI ARI</td><td>44.3, 44.6 14.3, 14.7 17.8, 18.0</td><td>48.3, 49.0 31.4, 32.1 30.2, 30.9</td><td>32.3, 32.0 14.1, 13.5 10.8, 10.3</td><td>55.9 51.9, 52.7 43.6, 44.2</td><td>43.4, 43.5 46.6, 47.4 34.4, 34.7</td><td>47.8, 51.6, 52.0 38.2, 38.4</td><td>53.4, 58.9, 59.0 46.2, 46.1</td></tr></table>

## D.2 Comparison without pretraining

Table 9 repeats the single-expert comparison without the autoencoder pretraining stage for pretraincapable methods, denoted $\boldsymbol { \mathrm { b y } } ^ { \dagger }$ in the table: $\mathrm { C I D E C ^ { \dagger } }$ , SpherePair<sup>†</sup>, ProbPair<sup>†</sup>, Weighted ProbPair<sup>†</sup>, and $\mathrm { E C I - P P ^ { \dagger } }$ . These methods are trained from random initialization, while VanillaDCC and VolMaxDCC are unchanged because they have no pretraining variant. All other settings match Table 7: lv0.01 supervision, corruption probability 0.3, and 3k/6k/9k constraints.

The main effect of removing pretraining is not a collapse of $\mathrm { E C I - P P ^ { \dagger } }$ , but a sharper separation from the other pretrain-capable methods. ECI-PP<sup>†</sup> ranks first in $6 8 / 7 2$ test entries and within the top two in $7 1 / 7 2$ , compared with $5 9 / 7 2$ and $6 6 / 7 2$ in the pre-trained comparison. It is also best in every entry on seven of the eight datasets; the only non-best entries occur on RCV1-10 ACC/ARI at 6k/9k. This indicates that $\mathrm { E C I - P P ^ { \dagger } }$ is less dependent on a favorable autoencoder initialization than the compared deep baselines.

The contrast with ${ \mathrm { S p h e r e P a i r } } ^ { \dagger }$ is also clearer without pretraining. $\mathrm { E C I - P P ^ { \dagger } }$ outperforms SpherePair<sup>†</sup> in $6 8 / 7 2$ entries, compared with $6 2 / 7 2$ in the pre-trained setting, and achieves the best NMI in all cases. Its absolute NMI margins over SpherePair<sup>†</sup> range from about $3 \mathrm { - } 1 2 \%$ , with visible gains on FMNIST, MNIST, and $\mathrm { S T L 1 0 }$ . This suggests that the Estimator–Corrector–Integrator structure helps form a more reliable grouping structure when the representation is learned from scratch.

The ProbPair variants show a more mixed response to random initialization. Weighted ProbPair<sup>†</sup> still improves over ProbPair<sup>†</sup> in most entries, so information-based weighting remains useful. However, Weighted ProbPair<sup>†</sup> no longer compares as consistently with SpherePair<sup>†</sup>: it beats SpherePair<sup>†</sup> in only $3 3 / \bar { 7 } 2$ entries, down from $5 3 / 7 2$ with pretraining. By contrast, ECI-PP<sup>†</sup> improves over Weighted ProbPair<sup>†</sup> in all entries, showing that weighting alone is not sufficient when the representation starts from a weaker initialization.

The remaining caveat is again the severely imbalanced RCV1-10 benchmark. ${ \mathrm { S p h e r e P a i r } } ^ { \dagger }$ keeps stronger ACC at 9k and stronger ARI at 6k/9k, consistent with the robustness of its angular geometry under severe imbalance. Nevertheless, ECI-PP<sup>†</sup> gives the best NMI at every budget and substantially improves over ProbPair<sup>†</sup>/Weighted ProbPair<sup>†</sup>, indicating stronger grouping consistency despite the remaining imbalance-related ACC/ARI limitation.

## D.3 Comparison with expert-aware baseline extensions

Table 10 compares ECI-PP with $\mathrm { S P ^ { E A } }$ and $\mathbf { W P P ^ { \mathrm { E A } } }$ under the multi3 setting, where $\mathrm { S P ^ { E A } }$ and $\mathbf { W P P ^ { \mathrm { E A } } }$ denote our expert-aware ensemble extensions of SpherePair and Weighted ProbPair, respectively. As described in Appendix C.4, these extensions instantiate multiple expert-weighted members for a baseline and aggregate their partitions through ensemble clustering. The expert-weighting parameter α is selected on the reserved validation split, with the selected values reported in Table 6. Thus, the comparison should be read as an auxiliary reference rather than a native baseline compari son: $\mathrm { S P ^ { E A } }$ and $\mathrm { W P P ^ { E A } }$ use expert identities, but they also introduce validation tuning and ensemble aggregation, making them inherently uncomparable with the native ECI-PP protocol in a strict sense.

The extensions substantially strengthen the corresponding baselines, confirming that expert identities carry useful information in the multi-expert setting. $\mathrm { \bf W P P ^ { \mathrm { E A } } }$ benefits clearly on datasets such as MNIST, FMNIST, and ${ \mathrm { S T L 1 0 } } ,$ , while $\mathrm { S P ^ { E A } }$ gains competitiveness on Reuters and RCV1-10. However, these gains conflate two effects: using expert identities and ensembling multiple expert-weighted members. For this reason, the results are best interpreted as stronger expert-aware references, not as fully symmetric replacements for the native baselines in Table 1.

Even against these stronger references, ECI-PP remains the most reliable method overall. It obtain the best result in $5 5 / 7 2$ test entries and ranks within the top two in $6 8 / 7 2$ . The advantage is clearest in NMI, where ECI-PP is best in 23/24 cases, indicating that its expert-conditioned design most consistently improves grouping structure. Dataset-wise, ECI-PP is best in all entries on CIFAR100, ImageNet10, Reuters, and STL10, and remains strongest in most CIFAR10 entries. The main losses occur on FMNIST and MNIST, mostly on ACC/ARI, where $\mathbf { W P P ^ { \mathrm { E A } } }$ is particularly strong, and but ECI-PP still provides a more direct and generally stronger way to use expert identities without relying on a validation-tuned ensemble wrapper.

Table 9: Comparative performance (%) (ACC, NMI, ARI) across datasets for methods under the lv0.01 setting with corruption probability 0.3 and 3k/6k/9k constraints. <sup>†</sup> indicates models without pretraining. Blue and black represent training and test results, respectively. Best results are in bold, and second-best are underlined.
<table><tr><td rowspan=1 colspan=12>Vanilla-  VolMax-                                Weighted ECI-PP†CIDEC†SpherePair†ProbPair†DCC    DCC                                ProbPair†  (Ours)</td></tr><tr><td rowspan=2 colspan=12>ACC14.7, 14.742.8,42.514.4, 14.2 44.0, 43.8 43.2,43.042.0,41.946.6, 46.53kNMI 11.2,11.842.5,42.9 11.2,11.5 41.9, 42.4 39.7,, 40.438.9, 39.545.5, 46.2CIFAR100     ARI 4.0, 4.1 26.2,26.0 3.7,3.5  27.5,27.5 25.7, 25.824.9, 25.030.5, 30.8</td></tr><tr><td rowspan=1 colspan=10></td></tr><tr><td rowspan=1 colspan=2>6k</td><td rowspan=1 colspan=10>ACC 16.5,16.645.4,45.3 12.3,, 12.2 47.6,47.8 44.7,, 44.945.2,45.549.3, 49.1NMI 12.6,13.543.9,, 44.5 8.6, 9.4  42.9,43.9 39.9,, 40.840.2,41.047.8, 48.3</td></tr><tr><td rowspan=3 colspan=2>9k</td><td rowspan=1 colspan=10>ARI 5.4, 5.6 28.2,28.2 2.7, 2.7  30.5, 30.9 26.9, 27.327.1, 27.633.9, 33.9</td></tr><tr><td rowspan=2 colspan=10>ACC 16.8,17.046.1,45.9 12.0,11.9  48.9,,49.0 47.4,,47.547.8,48.252.2, 52.5NMI 13.5,14.645.6,46.1 8.1, 9.0  42.5,,43.6 40.942.042.1, 43.149.3, 50.1ARI  6.0, 6.3 30.4,30.5 2.9,, 2.9  31.0,31.4 28.9,, 29.230.0, 30.435.9, 36.3</td></tr><tr><td rowspan=1 colspan=1>ARI</td></tr><tr><td rowspan=1 colspan=2>3k</td><td rowspan=1 colspan=1>ACCNMI</td><td rowspan=2 colspan=9>34.6,34.6 79.9,79.8 39.0,39.0 80.4,80.7  83.0,83.1 82.8,82.984.5,84.427.9,28.169.3,69.429.1, 29.6 70.8,71.1 71.8,72.1 73.3,73.676.0, 75.920.2,20.364.4, 64.417.8,, 17.7 67.2,,67.4 68.6, 68.770.370.472.3, 72.1</td></tr><tr><td rowspan=1 colspan=2>CIFAR10</td><td rowspan=1 colspan=1>ARI</td></tr><tr><td rowspan=1 colspan=2>6k</td><td rowspan=1 colspan=10>ACC35.535.784.7,84.635.7,35.9 84.2, 84.5 84.1,, 84.4385.387.0, 87.0NMI28.373.6,73.727.4, 29.171.6, 72.573.1,, 74.074.778.5, 78.5</td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=9>ARI 20.7,21.271.0,70.9 18.6, 19.0 70.2, 70.8 70.8,71.2</td><td rowspan=1 colspan=1>72.4,72.675.8,75.7</td></tr><tr><td rowspan=1 colspan=2>9k</td><td rowspan=1 colspan=1>ACCNMI</td><td rowspan=1 colspan=8>38.3,38.782.5,82.531.9,, 32.1 83.3,84.0 85.2,85.629.9,31.271.9,72.223.9,25.8 70.1,71.7 73.9,74.8</td><td rowspan=1 colspan=1>86.186.487.7,87.775.0,75.879.1,79.0</td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=9>ARI 23.2,24.068.3,68.316.7, 17.3 68.9, 70.2 71.9,72.5</td><td rowspan=1 colspan=1>73.5,73.976.8, 76.7</td></tr><tr><td rowspan=1 colspan=2>3k</td><td rowspan=1 colspan=9>ACC 36.6,, 36.563.3,62.536.3,35.6 64.6,63.6 58.0,, 57.3NMI31.1,31.155.5,55.229.2,28.9 57.757.2 51.8,51.4</td><td rowspan=2 colspan=1>61.9,61.367.3,, 66.653.2,53.263.1,62.443.0,42.651.8, 50.7</td></tr><tr><td rowspan=1 colspan=2>FMNIST</td><td rowspan=1 colspan=9>ARI 19.7,19.544.4, 43.615.7, 14.9 48.347.3 40.1, 39.3</td><td rowspan=1 colspan=1>43.0, 42.</td></tr><tr><td rowspan=1 colspan=2>6k</td><td rowspan=1 colspan=1>ACC</td><td rowspan=1 colspan=2>38.6,38.566.2,65.1</td><td rowspan=1 colspan=6>34.1,,33.6 64.7, 63.7 65.8,65.027.5,27.8 54.1, 54.1 55.8, 55.7</td><td rowspan=2 colspan=1>66.8,66.069.0, 67.756.1, 56.264.2, 63.447.346.653.1, 51.6</td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>ARI</td><td rowspan=1 colspan=2>20.7,20.946.6,,45.5</td><td rowspan=1 colspan=6>16.4,, 16.1 45.9, 45.4 46.6, 45.8</td></tr><tr><td rowspan=1 colspan=1></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>ARI 21.5,22.246.4, 45.2</td><td rowspan=1 colspan=1>15.9, 15.7</td><td rowspan=1 colspan=5>48.5, 48.0 49.7,, 49.2</td><td rowspan=1 colspan=1>51.0,50.1 55.5, 54.0</td></tr><tr><td rowspan=2 colspan=1></td><td rowspan=2 colspan=1>3k</td><td rowspan=1 colspan=3>ACC35.4,37.386.7,86.9</td><td rowspan=1 colspan=1>50.8,52.0</td><td rowspan=1 colspan=5>88.5, 89.8 87.3,87.6</td><td rowspan=1 colspan=1>88.9,,89.1 92.5, 92.6</td></tr><tr><td rowspan=1 colspan=3>NMI27.7,30.878.8, 79.4</td><td rowspan=1 colspan=1>42.8, 46.5</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=2 colspan=1>ImageNet10</td><td rowspan=2 colspan=1></td><td rowspan=2 colspan=1>ARI</td><td rowspan=2 colspan=1>19.1,20.8</td><td rowspan=2 colspan=1>74.1,73.9</td><td rowspan=2 colspan=1>30.1, 31.5</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=1 colspan=5>78.8, 81.6 78.4,, 79.377.5,79.7 75.7,, 76.0</td><td rowspan=1 colspan=1>80.6,,81.487.9,88.277.9,78.285.5,85.6</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>6k</td><td rowspan=1 colspan=1>NMI</td><td rowspan=1 colspan=1>40.6,42.030.6,35.2</td><td rowspan=1 colspan=1>87.2,,87.578.3,79.3</td><td rowspan=1 colspan=1>32.0,33.623.3, 29.4</td><td rowspan=2 colspan=5>87.0,,89.4 88.8,89.889.6,974.9, 80.2 79.2,, 81.580.974.3,,78.7 77.7,, 79.479.3,</td><td rowspan=1 colspan=1>0.1 92.5, 92.882.387.8, 88.4</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>ARI</td><td rowspan=1 colspan=1>23.2,26.0</td><td rowspan=1 colspan=1>75.1,75.6</td><td rowspan=1 colspan=1>16.1, 19.4</td><td rowspan=1 colspan=1>77.7, 79.4</td><td rowspan=1 colspan=1>85.5, 86.0</td></tr><tr><td rowspan=3 colspan=1></td><td rowspan=3 colspan=1>9k</td><td rowspan=3 colspan=1>NMI</td><td rowspan=3 colspan=1>39.0,,41.527.6,33.7</td><td rowspan=3 colspan=1>87.4,88.177.8,79.2</td><td rowspan=1 colspan=1>31.9.34.5</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=2 colspan=1>23.9,32.4</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=1 colspan=4>85.2, 88.971.1,79.1</td><td rowspan=1 colspan=1>88.4., 90.178.3,82.2</td><td rowspan=2 colspan=1>92.6, 93.187.9, 88.579.885.6, 86.2</td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=2>ARI21.1, 25.2</td><td rowspan=1 colspan=1>75.2,76.1</td><td rowspan=1 colspan=1>17.1, 22.1</td><td rowspan=1 colspan=4>70.9, 77.9</td><td rowspan=1 colspan=1>76.8,, 79.9</td><td rowspan=1 colspan=1>77.8,</td></tr><tr><td rowspan=2 colspan=2>3k</td><td></td><td></td><td></td><td rowspan=1 colspan=1>41.6,41.0</td><td rowspan=1 colspan=4>72.5,, 73.2</td><td rowspan=2 colspan=1>67.6,68.854.0,, 55.9</td><td rowspan=1 colspan=1>70.1,70.7 80.0,80.6</td></tr><tr><td rowspan=1 colspan=1>NMI</td><td rowspan=1 colspan=1>16.0,17.0</td><td rowspan=1 colspan=1>52.7,54.1</td><td rowspan=1 colspan=1>35.1, 35.6</td><td rowspan=1 colspan=4>61.5,63.0</td><td rowspan=1 colspan=1>57.2,58.468.6, 69.6</td></tr><tr><td rowspan=1 colspan=1>MNIST</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>ARI</td><td rowspan=1 colspan=1>10.4,10.8</td><td rowspan=1 colspan=1>47.8,48.9</td><td rowspan=1 colspan=1>19.6, 18.8</td><td rowspan=1 colspan=4>56.4,57.7</td><td rowspan=1 colspan=1>48.7,, 50.4</td><td rowspan=1 colspan=1>52.1,53.1 64.9, 65.9</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>6k</td><td rowspan=1 colspan=1>ACCNMI</td><td rowspan=1 colspan=1>33.3,33.924.2,25.8</td><td rowspan=1 colspan=1>73.4,74.4</td><td rowspan=1 colspan=1>36.4,36.3</td><td rowspan=1 colspan=4>76.8,77.9</td><td rowspan=1 colspan=1>76.6,77.5</td><td rowspan=1 colspan=1>82.4,83.287.4, 88.0</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>ARI</td><td rowspan=1 colspan=1>18.4,19.4</td><td rowspan=1 colspan=1>55.2, 56.8</td><td rowspan=1 colspan=1>18.2, 18.0</td><td rowspan=1 colspan=4>60.1, 61.7</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>59.3,, 60.9</td></tr><tr><td rowspan=4 colspan=2>9k</td><td rowspan=4 colspan=2>ACC38.9,40.3NMI29.5,32.2ARI 23.1,24.8</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td rowspan=3 colspan=1>28.3, 30.919.1, 19.8</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td rowspan=2 colspan=5>80.1, 81.3 82.2,83.667.0, 69.6 67.5,70.265.5, 67.7 66.4,, 68.9</td><td rowspan=2 colspan=1>86.0,87.288.7, 89.272.274.677.4, 78.572.3,74.677.2, 78.2</td></tr><tr><td rowspan=1 colspan=1>56.5,58.7</td></tr><tr><td rowspan=1 colspan=2>3k</td><td rowspan=1 colspan=1>ACCNMI</td><td rowspan=1 colspan=1>53.2,,54.015.1,17.9</td><td rowspan=1 colspan=1>62.8,,63.225.6,28.0</td><td rowspan=1 colspan=1>58.1,56.0</td><td rowspan=1 colspan=4>76.5,, 78.849.554.7</td><td rowspan=1 colspan=1>69.4,71.544.4,, 48.8</td><td rowspan=1 colspan=1>71.6,73.980.0,80.746.7, 51.256.3,57.9</td></tr><tr><td rowspan=1 colspan=1>REUTERS</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>ARI</td><td rowspan=1 colspan=1>17.0,18.0</td><td rowspan=1 colspan=2>31.2,32.429.6, 29.7</td><td rowspan=1 colspan=4>54.0,59.2</td><td rowspan=1 colspan=1>46.7,,51.3</td><td rowspan=1 colspan=1>48.7, 53.361.6, 63.2</td></tr><tr><td rowspan=3 colspan=1></td><td rowspan=3 colspan=1>6k</td><td></td><td rowspan=1 colspan=1>66.0,70.6</td><td rowspan=1 colspan=2>72.5,76.845.1,46.1</td><td rowspan=1 colspan=4>78.4,,81.2</td><td></td><td></td></tr><tr><td rowspan=2 colspan=1>NMI</td><td rowspan=2 colspan=1>27.7,36.3</td><td rowspan=2 colspan=2>36.0, 44.99.9, 14.1</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td rowspan=1 colspan=1>75.6,78.248.1,53.5</td><td rowspan=1 colspan=1>74.5,77.183.7,85.248.1,53.559.5, 62.7</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>ARI</td><td rowspan=1 colspan=1>34.0,42.6</td><td rowspan=1 colspan=2>42.4, 50.811.3, 14.3</td><td rowspan=1 colspan=4>54.5,61.1</td><td rowspan=1 colspan=2>52.6,,57.951.2, 56.365.0, 68.1</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>9k</td><td rowspan=1 colspan=1>NMI</td><td rowspan=1 colspan=1>69.7,77.030.5,43.3</td><td rowspan=1 colspan=2>71.7,76.852.9,54.534.3,43.416.8,25.5</td><td rowspan=1 colspan=6>77.5,, 82.2  76.1,80.1 75.3,79.381.9,83.148.7,57.0 49.5,57.547.754.657.6, 60.2</td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>ARI</td><td rowspan=1 colspan=1>36.8,,50.4</td><td rowspan=1 colspan=2>40.9,50.419.6, 25.5</td><td rowspan=1 colspan=6>52.6, 61.4 52.3,60.051.3,58.662.0, 64.7</td></tr><tr><td rowspan=1 colspan=2>3k</td><td rowspan=1 colspan=1>NMI</td><td rowspan=1 colspan=1>30.9,31.521.1,23.4</td><td rowspan=1 colspan=2>72.8,73.349.2,50.364.8,66.538.2, 41.9</td><td rowspan=1 colspan=6>80.9,82.9 80.3,81.1 80.8,81.584.2, 84.767.2,71.0 66.9,, 69.267.5,69.274.5,75.1</td></tr><tr><td rowspan=1 colspan=2>STL10</td><td rowspan=1 colspan=1>ARI</td><td rowspan=1 colspan=1>14.1,15.0</td><td rowspan=1 colspan=2>56.3, 57.027.0, 28.5</td><td rowspan=1 colspan=6>64.3,67.8 64.0, 65.564.7,,66.071.0,71.6</td></tr><tr><td rowspan=1 colspan=2>6k</td><td rowspan=1 colspan=1>ACCNMI</td><td rowspan=1 colspan=1>30.4,31.619.5,22.2</td><td rowspan=2 colspan=8>76.8,,77.5 35.1,, 36.8 78.7,81.7 80.3,81.981.5,82.785.2,85.765.8, 67.726.8, 31.7 62.6, 68.4 66.3,,69.667.9,70.775.2, 76.260.0, 61.318.6,21.3 60.4, 65.6 63.8,, 66.465.767.872.1, 73.0</td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>ARI</td><td rowspan=1 colspan=1>14.3,15.7</td></tr><tr><td rowspan=2 colspan=2>9k</td><td rowspan=1 colspan=2>ACC35.5,37.8NMI 22.9,27.6</td><td rowspan=1 colspan=8>79.0,80.131.8,33.4 76.9,80.8 80.3,, 82.881.3,,82.785.3,85.766.7, 69.323.3, 28.8 59.7, 66.9 66.0,70.8 67.1,70.675.6,76.3</td></tr><tr><td rowspan=1 colspan=2>ARI 17.1,19.8</td><td rowspan=1 colspan=8>62.8, 64.716.4,19.3 57.6, 64.1 63.6, 67.765.3,, 67.872.4, 73.0</td></tr><tr><td rowspan=1 colspan=2>3k</td><td rowspan=1 colspan=10>ACC 40.0,,40.1 46.3,46.4 39.4,,37.4 52.1,52.2 45.2,45.1 44.6,44.655.5,55.7NMI 12.3,12.421.5,21.921.6,21.4 51.5,51.7 42.3,42.543.9.44.154.755.1</td></tr><tr><td rowspan=1 colspan=2>RCV1-10</td><td rowspan=1 colspan=1>ARI</td><td rowspan=1 colspan=9>16.4, 16.521.2,21.418.3, 19.6 42.6, 42.7  34.5,, 34.535.0,35.047.6,47.9</td></tr><tr><td rowspan=1 colspan=2>6k</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=3 colspan=2></td><td rowspan=3 colspan=1>ARI</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=2 colspan=9>21.8,22.236.0,,36.7 15.1, 14.9 55.8,56.2 44.1,, 44.547.4,,47.458.5,58.828.0,28.2          16.2, 16.1 52.8,52.9 37.5,,37.639.0,38.951.5,,51.7</td></tr><tr><td rowspan=1 colspan=5></td><td rowspan=1 colspan=1>,52</td></tr><tr><td rowspan=1 colspan=2>9k</td><td rowspan=1 colspan=10>ACC                              66.6,NMI 15.5,15.935.7,36.714.4, 14.9 58.0,58.8 43.6, 44.049.7,49.961.4,, 61.6ARI 19.8,20.144.7,45.4 15.8, 16.4 57.8,58.4 35.6,, 35.642.9,43.056.0,56.0</td></tr></table>

on RCV1-10 ACC/ARI, where the angular geometry of SP<sup>EA</sup> remains advantageous under severe imbalance. Overall, expert-aware ensembling makes strong baselines substantially more competitive,

Table 10: Comparative performance (%) (ACC, NMI, ARI) across datasets for ECI-PP, $\mathrm { S P ^ { E A } }$ , and WPP<sup>EA</sup> under the multi3 setting, with corruption probability 0.3 and 3k/6k/9k total constraints. SP<sup>EA</sup> and WPP<sup>EA</sup> denote the expert-aware ensemble extensions of SpherePair and Weighted ProbPair, respectively. Blue and black represent training and test results, respectively. Best results are in bold, and second-best are underlined.
<table><tr><td rowspan="2" colspan="2"></td><td colspan="3">3k</td><td colspan="3">6k</td><td colspan="3">9k</td></tr><tr><td>ACC</td><td>NMI</td><td>ARI</td><td>ACC</td><td>NMI</td><td>ARI</td><td>ACC</td><td>NMI</td><td>ARI</td></tr><tr><td rowspan="3">CIFAR100</td><td>SPEA</td><td>42.2, 43.2</td><td>41.7, 42.5</td><td>26.6, 27.0</td><td>41.0,40.9</td><td>39.7, 40.9</td><td>25.5,25.8</td><td>41.9,42.3</td><td>39.2,40.5</td><td>26.0, 27.1</td></tr><tr><td>WPPEA</td><td>43.1, 43.4</td><td>41.4, 42.2</td><td>26.4, 26.5</td><td>41.5, 41.4</td><td>40.4, 41.3</td><td>25.8, 26.1</td><td>46.4, 46.4</td><td>42.9, 43.8</td><td>29.8,29.9</td></tr><tr><td>ECI-PP</td><td>45.6, 45.3</td><td>45.0, 45.5</td><td>29.7, 29.8</td><td>49.1, 49.3</td><td>47.1, 47.7</td><td>32.1, 32.4</td><td>48.6, 48.8</td><td>48.1, 48.5</td><td>32.9, 32.9</td></tr><tr><td rowspan="3">CIFAR10</td><td>SPEA</td><td>76.6,77.1</td><td>70.7,71.5</td><td>64.0, 64.8</td><td>75.5,75.9</td><td>67.9, 69.4</td><td>62.4, 63.5</td><td>74.1,75.2</td><td>67.0, 69.0</td><td>61.3, 62.9</td></tr><tr><td>WPPEA</td><td>82.6, 82.5</td><td>74.1, 74.2</td><td>69.4, 69.2</td><td>81.2, 81.2</td><td>74.6, 74.9</td><td>69.6, 69.6</td><td>82.5, 82.8</td><td>74.2, 74.5</td><td>70.0, 70.3</td></tr><tr><td>ECI-PP</td><td>82.8, 82.5</td><td>76.0, 75.9</td><td>70.8, 70.7</td><td>84.0, 84.0</td><td>77.0, 77.1</td><td>72.6, 72.6</td><td>81.6,82.1</td><td>77.8,77.8</td><td>73.1, 73.0</td></tr><tr><td rowspan="3">FMNIST</td><td>SPEA</td><td>63.2, 61.6</td><td>60.3, 59.7</td><td>48.4, 46.9</td><td>63.5, 62.5</td><td>59.4,59.9</td><td>48.5,48.3</td><td>62.5, 62.5</td><td>58.6, 60.2</td><td>49.0, 49.4</td></tr><tr><td>WPPEA</td><td>64.5, 64.3</td><td>59.7, 59.8</td><td>47.9, 47.7</td><td>68.0, 68.3</td><td>62.1, 62.4</td><td>52.0, 52.0</td><td>71.8, 71.7</td><td>64.6, 64.8</td><td>55.2, 55.1</td></tr><tr><td>ECI-PP</td><td>60.1, 59.2</td><td>62.7, 61.9</td><td>49.5, 48.4</td><td>62.0, 61.1</td><td>64.3, 63.8</td><td>51.0,51.0</td><td>65.3, 64.8</td><td>66.0, 65.6</td><td>53.4, 52.8</td></tr><tr><td rowspan="3">ImageNet10</td><td>SPEA</td><td>81.7,85.3</td><td>78.0,79.8</td><td>71.8,75.2</td><td>82.1,86.6</td><td>77.8,79.6</td><td>72.8,75.8</td><td>81.2,81.4</td><td>75.3,76.7</td><td>70.6, 71.3</td></tr><tr><td>WPPEA</td><td>90.5, 90.8</td><td>83.8, 84.2</td><td>81.8, 82.0</td><td>90.9, 91.6</td><td>83.4, 84.8</td><td>82.2, 83.4</td><td>91.2, 92.1</td><td>82.4, 84.7</td><td>82.2, 83.8</td></tr><tr><td>ECI-PP</td><td>91.5, 91.8</td><td>86.9, 87.3</td><td>84.1, 84.4</td><td>91.9, 92.1</td><td>87.2, 87.8</td><td>84.6, 85.0</td><td>92.4, 92.6</td><td>87.7, 88.1</td><td>85.4, 85.7</td></tr><tr><td rowspan="3">MNIST</td><td>SPEA</td><td>80.5,81.1</td><td>75.2, 76.7</td><td>70.7,71.7</td><td>78.6, 79.6</td><td>72.7,75.5</td><td>67.9, 70.2</td><td>77.5,78.1</td><td>70.2, 73.7</td><td>65.7,67.8</td></tr><tr><td>WPPEA</td><td>87.9,88.7</td><td>77.2, 78.5</td><td>76.2, 77.4</td><td>85.8, 86.4</td><td>77.4, 79.0</td><td>75.5, 76.8</td><td>89.2, 90.4</td><td>78.1, 80.4</td><td>78.2, 80.3</td></tr><tr><td>ECI-PP</td><td>80.9, 81.5</td><td>75.7, 77.1</td><td>71.9, 73.2</td><td>86.1, 86.4</td><td>79.8,80.5</td><td>77.8,78.3</td><td>84.7, 85.1</td><td>79.9, 80.7</td><td>77.1, 77.5</td></tr><tr><td rowspan="3">REUTERS</td><td>SPEA</td><td>75.8,76.1</td><td>41.8,45.7</td><td>48.0,50.2</td><td>72.0, 77.0</td><td>35.3,43.8</td><td>40.9, 49.8</td><td>68.4,77.0</td><td>30.6,43.1</td><td>34.7, 49.6</td></tr><tr><td>WPPEA</td><td>82.9, 83.3</td><td>55.4, 58.3</td><td>62.4, 64.5</td><td>81.4, 83.7</td><td>53.6, 58.4</td><td>59.5, 64.4</td><td>77.7, 80.9</td><td>50.3, 56.0</td><td>54.8, 61.5</td></tr><tr><td>ECI-PP</td><td>85.0, 86.3</td><td>62.0, 65.0</td><td>68.3, 71.1</td><td>87.3, 88.5</td><td>63.8, 67.2</td><td>71.3,74.3</td><td>88.3, 89.6</td><td>65.4, 68.7</td><td>72.9, 76.0</td></tr><tr><td rowspan="3">STL10</td><td>SPEA</td><td>74.2,76.7</td><td>66.5, 68.1</td><td>59.7, 61.1</td><td>73.0,75.4</td><td>64.0,67.3</td><td>57.4,60.1</td><td>72.7,72.9</td><td>62.5, 66.2</td><td>56.7,58.7</td></tr><tr><td>WPPEA</td><td>84.0, 84.5</td><td>72.4, 73.8</td><td>69.0, 70.0</td><td>85.2, 85.6</td><td>73.7, 75.2</td><td>71.3, 72.0</td><td>85.1, 86.1</td><td>73.3, 75.4</td><td>71.1, 72.7</td></tr><tr><td>ECI-PP</td><td>85.7,85.9</td><td>75.7,76.3</td><td>72.2, 72.5</td><td>87.4, 87.5</td><td>77.5,78.1</td><td>75.1, 75.4</td><td>87.9, 87.7</td><td>78.3, 78.5</td><td>75.9, 75.8</td></tr><tr><td rowspan="3">RCV1-10</td><td>SPEA</td><td>60.2, 60.9</td><td>52.6,53.1</td><td>48.9, 49.5</td><td>68.2, 63.5</td><td>54.8,53.8</td><td>55.7, 51.4</td><td>71.2, 67.7</td><td>54.8,54.4</td><td>58.8, 55.9</td></tr><tr><td>WPPEA</td><td>52.5, 47.6</td><td>52.6,52.5</td><td>41.9, 38.3</td><td>54.6,52.6</td><td>53.6,53.7</td><td>45.0, 42.8</td><td>56.5, 58.4</td><td>53.5,54.2</td><td>47.0, 49.2</td></tr><tr><td>ECI-PP</td><td>55.3, 55.2</td><td>56.8, 56.9</td><td>45.6, 45.4</td><td>55.2, 55.0</td><td>58.4, 58.6</td><td>46.8, 46.8</td><td>53.4,53.3</td><td>58.9, 59.0</td><td>46.2, 46.1</td></tr></table>

## D.4 Robustness across supervision conditions

Figs. 5 to 7 report the full ACC, NMI, and ARI results corresponding to the robustness study in the main text. All three sweeps use a fixed budget of 9k constraints, and the central setting in each sweep, namely lv0.01 expert quality, corruption probability 0.3, and multi3, matches the default setting used in Table 1. Thus, these figures also cover the main-comparison results in Table 1, while additionally reporting the standard deviations omitted from the table for space. For the multi-expert sweep, we further include SP<sup>EA</sup> and WPP<sup>EA</sup>, the expert-aware ensemble extensions introduced in Appendix D.3, as stronger references that also exploit expert identities.

Effects of supervision factors. Across the three sweeps, the results largely follow the expected supervision-quality ordering. In the single-expert sweep, stronger experts generally yield better clustering quality; in the corruption sweep, increasing corruption leads to clear degradation. In the multi-expert sweep, moving from multi2 to multi10 also tends to help the stronger methods, because the protocol progressively reduces expert blind spots and increases average expert accuracy, as quantified in Table 4. Since the total constraint budget is fixed throughout, the performance trends mainly reflect changes in supervision quality and heterogeneity rather than in the amount of supervision.

Baseline behavior. The non-ECI baselines show a clear hierarchy as supervision becomes less reliable. Plain end-to-end methods such as VanillaDCC and CIDEC often nearly collapse under lowquality or heterogeneous supervision, especially in NMI and ARI, suggesting that cluster-assignment training is vulnerable when pairwise supervision is inconsistent. VolMaxDCC is usually more resilient, consistent with the benefit of explicit noise modeling. SpherePair and the ProbPair-family baselines are substantially stronger than the plain end-to-end methods, reflecting the robustness of geometric representation learning under noisy supervision. Nevertheless, corruption still causes visible degradation for the stronger baselines, particularly on FMNIST, ImageNet10, and STL10.

![](images/b9164e59e9bb6c0b6c96016e3cfc9df6b1dc1137a29e17b45f59e8377e73ad25.jpg)  
Figure 5: Full robustness results in ACC (mean std over 5 runs) across datasets under varying (A) expert quality, (B) corruption probability, and (C) multi-expert configuration, with 9k constraints. The multi-expert row additionally includes expert-aware SpherePair and Weighted ProbPair extensions.

![](images/b50d34217c768d380fb64361d3dfa2d29aaf2b69aa6f782baed1c36e3d75fd8b.jpg)  
Figure 6: Full robustness results in NMI (mean std over 5 runs) across datasets under varying (A) expert quality, (B) corruption probability, and (C) multi-expert configuration, with 9k constraints. The multi-expert row additionally includes expert-aware SpherePair and Weighted ProbPair extensions.

![](images/017998d3a9da50d875574649986401dbec042dd1c524bc2ead995a785fa21415.jpg)  
Figure 7: Full robustness results in ARI (mean std over 5 runs) across datasets under varying (A) expert quality, (B) corruption probability, and (C) multi-expert configuration, with 9k constraints. The multi-expert row additionally includes expert-aware SpherePair and Weighted ProbPair extensions.

ECI-PP robustness. Across the supervision sweeps, ECI-PP better preserves clustering quality than the native baselines, especially as expert quality decreases or corruption increases. This advantage is most consistent on NMI and ARI, suggesting that the correction-and-integration pipeline mainly improves the recovered grouping structure. ACC is more competitive and contains more exceptions, especially on FMNIST, MNIST, and RCV1-10, where Weighted ProbPair, $\mathrm { S P ^ { E A } }$ , or $\mathrm { { w p p E A } }$ can sometimes match or exceed ECI-PP. This is consistent with the main-comparison results (Appendix D.1): matched-label accuracy can favor angular or ensemble-based baselines on some datasets, even when ECI-PP remains stronger or near-stronger on grouping-oriented metrics (NMI/ARI). The expert-aware extensions further clarify the role of expert identities: $\mathrm { S P ^ { E A } }$ and $\mathrm { \dot { W P P } ^ { E A } }$ often improve over their native counterparts in the multi-expert sweep, confirming that expert identity information is useful under heterogeneous supervision. However, these extensions rely on validation-selected expert weighting and ensemble aggregation, whereas ECI-PP uses expertconditioned correction within the learning pipeline. Overall, the full results support the main-text conclusion that ECI-PP better preserves clustering quality when supervision is fallible, heterogeneous, or corrupted.

## D.5 Held-out diagnostics

Diagnostic protocol. The held-out diagnostics complement the clustering results by tracking the training dynamics of ECI-PP’s internal quantities. Following Appendix C.4, we construct a sampledisjoint diagnostic constraint set ${ \mathcal { C } } ^ { \mathrm { d i a g } }$ from the test split, used only for diagnosis and never for optimization, early stopping, or hyperparameter selection. Benchmark labels are used only to define an external hard co-membership reference, rather than the canonical aleatoric relation $R ^ { \star }$ . All diagnostics use multi3 supervision with limited 3k training constraints in total (1k per expert) and corruption rate 0.3. To reduce the influence of a strong initial representation or a long warm-up phase, we use random initialization without unsupervised pretraining, give the Estimators and Integrator only one warm-up epoch, and then track the next 500 training iterations, each corresponding to one epoch. We report mean std over 5 runs, recording at each iteration the Estimator out-of-fold relation $\bar { y } _ { i } ^ { \mathrm { o o f } }$ , the corrected relation $y _ { i } ^ { \mathrm { c o r } }$ , and the Integrator relation $y _ { i } ^ { \mathrm { i n t } }$ on $\mathcal { C } ^ { \mathrm { d i a g } }$

Diagnostic metrics. For a held-out pair $i = ( a _ { i } , b _ { i } )$ with benchmark labels $t _ { a _ { i } }$ and $t _ { b _ { i } }$ , define the external hard co-membership reference

$$
o _ { i } ^ { \star } : = \mathbb { I } [ t _ { a _ { i } } = t _ { b _ { i } } ] .
$$

Let ${ \mathcal { T } } _ { \mathrm { c l e a n } } = \{ i \in { \mathcal { C } } ^ { \mathrm { d i a g } } : c _ { i } = 0 \}$ be the uncorrupted held-out subset, and define the residual discrepancy after correction as

$$
\mathrm { g a p } _ { i } = \left| \mathrm { s o f t c l i p } _ { \xi } ( \hat { y } _ { i } ^ { \mathrm { o o f } } ) - y _ { i } ^ { \mathrm { c o r } } \right| .
$$

For any relation score $v _ { i } \in [ 0 , 1 ]$ and index set $\mathcal { T } ,$ we use

$$
\operatorname { B r i e r } ( v ; \mathcal { T } ) = \frac { 1 } { | \mathcal { T } | } \sum _ { i \in \mathcal { T } } ( v _ { i } - o _ { i } ^ { \star } ) ^ { 2 } .
$$

We report $\mathrm { B r i e r } _ { \mathrm { e s t } } ~ = ~ \mathrm { B r i e r } ( \hat { y } ^ { \mathrm { o o f } } ; \mathcal { C } ^ { \mathrm { d i a g } } )$ , $\begin{array} { r l r } { 3 \mathrm { r i e r } _ { \mathrm { c o r , c l e a n } } } & { { } = } & { \mathrm { B r i e r } ( y ^ { \mathrm { c o r } } ; \mathcal { T } _ { \mathrm { c l e a n } } ) , \mathrm { B r i e r } _ { \mathrm { i n t } } = } \end{array}$ Brier $( y ^ { \mathrm { i n t } } ; \mathcal { C } ^ { \mathrm { d i a g } } )$ , and $\begin{array} { r } { \mathbb { E } [ \mathrm { g a p } _ { i , \mathrm { c l e a n } } ] = | \mathcal { T } _ { \mathrm { c l e a n } } | ^ { - 1 } \sum _ { i \in \mathcal { T } _ { \mathrm { c l e a n } } } \mathrm { g a p } _ { i } } \end{array}$ . For reliability-aware screening, gap is used to rank held-out constraints, and we report $\mathrm { \bar { A } \bar { U } \bar { C } _ { c o r r u p t } }$ and $\mathrm { A P _ { c o r r u p t } }$ against the known corruption indicator $c _ { i }$ . These scores measure alignment with corruption rather than pure corruption estimation, since the reliability signal may also reflect difficult clean pairs, residual epistemic variation, and imperfect estimation/correction; nevertheless, corrupted records are expected to rank higher on average. As a reference, we also replace the multi3 expert-generated judgments by $o _ { i } ^ { \star }$ before applying the same 3k budget and corruption rate 0.3, thereby removing expert epistemic uncertainty from the clean supervision.

Evolution of internal estimates. The full diagnostic curves are shown in Figs. 8 to 15. Panels (A–C) track the Estimator, Corrector, and Integrator relation scores through their Brier errors against the external hard co-membership reference $o _ { i } ^ { \star }$ . Across datasets, these errors typically decrease quickly in the early stage and then become relatively stable, with some late-stage fluctuation. This pattern is clearer on CIFAR100-20, CIFAR10, ImageNet10, MNIST, and STL10, while FMNIST and Reuters show more rebound after the initial improvement, and RCV1-10 is the most unstable case. Because these Brier scores use $o _ { i } ^ { \star }$ rather than the latent aleatoric relation $R _ { a _ { i } b _ { i } } ^ { \star }$ , small non-monotone changes should be read as diagnostic behavior rather than as direct evidence about latent-target recovery. Panel (D) gives a complementary correction-side view: the clean residual discrepancy usually drops early, indicating that corrected relations become more aligned with estimator beliefs on uncorrupted held-out constraints. Since the Estimator and Corrector co-evolve, this quantity should be interpreted together with (A–C), as evidence that the correction stage follows the improving estimator signal.

![](images/116c84ad8cb2ba93d215b398ad8948a02a6570c8d3577e48d7ec5ed300a2a4eb.jpg)

![](images/86b7146818232f818efa89a3eb4e66ec2509af0f5785705845ce8624ee80abfd.jpg)

![](images/5c38f2acf3456f823eed248303a54ec8bd62c9ded1e8e08daf0f6b14910b7a9c.jpg)

![](images/fc49ca25fda42032e08ee87824e71129b4e340e38f4830bf5695f18424dc607a.jpg)

![](images/9d9332167329244cc401b2eae5de0cceb2ea5a23a990e508dbe2f5da0dd17223.jpg)

![](images/6568247bbca8f8e331cfb4f63ff1d708ad3120bcf1b0235cd22bf4a7b8216d6a.jpg)

![](images/9c6bf3ff6cf986ad9d6e8f11a6572a96e5375dbd986b94bd187e0414c0f34931.jpg)

![](images/d759b7a64c92dbbf0fed9ee9cc9449b41ec0fd8bdf2efc5477ab40f1a940d87b.jpg)

Figure 8: CIFAR100-20 held-out diagnostic evolution under multi3 with corruption rate 0.3 (mean std over 5 runs). Same panel definitions as in Fig. 3.  
![](images/e32b01ff5f217d3438c3d01f65f50a62959dad0977fd60c08e2991a022e30f7b.jpg)

![](images/67857d4672d56b02a8bdd4c88b758c7010f4acc4bf27a06e3edbdb1b60eb7f09.jpg)

![](images/9229498e59bbccbaeb32b58aa452c83bb08773ac8cd80f942d74e843b8966add.jpg)

![](images/ac38e7deb23b9f317e5723c7e15c5256ab9f9d4eeb3bcbfbb7ff26c850c3cd0d.jpg)

![](images/4242328dbeb344114ca160627e553fb6976a4707618da41a91abe954e61aef77.jpg)

![](images/c3556fa780b28ac297ff6a5c1fb3990c58d7764d6379a5ee2654b28bedd7f8a0.jpg)

![](images/006bb2418fdeff6dc78708be1b3a9996e3a7530eaf6938afc6bca619421c7bce.jpg)

![](images/3d0e80ace76643e0913793ad0ea167c16e1dda26c868b49ec19acd1ea2efe8c8.jpg)  
Figure 9: CIFAR10 held-out diagnostic evolution under multi3 with corruption rate 0.3 (mean std over 5 runs). Same panel definitions as in Fig. 3.

![](images/7010e1cf935ee7edf42cbc09ee10b2b8b5a97106ffe58106086115229cc3f9cb.jpg)

![](images/e5f7e3fc0fa3e50c39f82a259a2721977c4415b438541214f5c59432ec43a92d.jpg)

![](images/afdc658ebeccc70321a9aa8017eeb1858e3dc1e4cf4df66c5bac979e652fd760.jpg)

![](images/7d330d5f43ab5e6e54315f7be3b22a82e89b312626f33051c6f33ea2de3c1030.jpg)

![](images/6392d4a8983e2f92f0250a01f2382861e8842215af2f3b9362dbb73a65fd43c9.jpg)

![](images/fd1c59407c8c8fac37933da97784a978d592d9f75d4ae61a1509d1add85d96ec.jpg)

![](images/54ffd381dd0626665e1cc9fa98a2db1fc4234429c8744dd61ba9c09cf95dd451.jpg)

![](images/e78e3736aef7c16e06b52cffa175211ad1a735ff7824b5336fe5ee07444654d1.jpg)

Figure 10: FMNIST held-out diagnostic evolution under multi3 with corruption rate 0.3 (mean std over 5 runs). Same panel definitions as in Fig. 3.  
![](images/15e15faeec100ccf0f0ada841234559325ca3a35d0c9a5c736b2150848ab5a6f.jpg)

![](images/26c336d12dcf4fce3fbf6985a3e5748f191710d27e254dea30482a785182afd3.jpg)

![](images/1633f01a560ec6e33bd89a1829b77f5a320a0fa7b380eccf77feee4c9b815aa2.jpg)

![](images/57bae2ca04a96804a08f4aed8d6ebd9f301156c58fe30aca29656b88f3ea7ba4.jpg)

![](images/d9600b04285206c2d1a6dc8e796739bb54748468b2534ddecb4ec1fb6f509559.jpg)

![](images/93b8113dc9073b4ef0940589614680637e28282b3093345ab63038ec3ae63e70.jpg)

![](images/a2b59c8b631052b2547ead19bd55ac31d5ac44c0b73ab3f17268b8e761304083.jpg)

![](images/a17b3d69af60de323c5864f04a439560e52b35a476525016995be8ab6151c105.jpg)  
Figure 11: ImageNet10 held-out diagnostic evolution under multi3 with corruption rate 0.3 (mean std over 5 runs). Same panel definitions as in Fig. 3.

![](images/0f35839e1c07767797a5683bf40cc114219aeb6c3067359d37e1e5c891d5b5de.jpg)

![](images/bbae1c15be87b8eb0736d199f0ffb3e69317b4d6dcb6e7b82dee930c0c0c1897.jpg)

![](images/b8ba8017a271558faa4b0c8c2161860b6ad73389eac651e7fe07ee8b21efd33d.jpg)

![](images/6d2cb0429b7b2d1f71640c70a48d8eaaef7e34bb68fbe9fd7b243b312e578604.jpg)

![](images/c27c17c077421db2ad698ff7d9e78b540707b2d91645ec8aa9c34fe2b0b3f6c2.jpg)

![](images/e1d4eda5447c478315659af9bc0cd660ae049277eb685ffef0920270274926bf.jpg)

![](images/a05db4646538ed007c9a769639337b03b4270a7e59e880384a41b3e026afa3bb.jpg)

![](images/659e5f50ba4de6a04cd9e5c7d1b9b2c2c30e7b3b1e0733d13004a4ae029029df.jpg)

Figure 12: MNIST held-out diagnostic evolution under multi3 with corruption rate 0.3 (mean std over 5 runs). Same panel definitions as in Fig. 3.  
![](images/33ce6a189b5fc6af15d536100c6efb815378381bfa2475b9df0f28b17f91c4a9.jpg)

![](images/b5fa5694ce0b773bf257547e0f274e7619edbf3ae3715b44e6201759263d4641.jpg)

![](images/6e352a15c17b8cfeb0fe2dfed9a15098f4779f24f1c270039d1ead10579dd8dd.jpg)

![](images/25c18b7c6d25c49ea266d789805b6dc68aa6a8ab109290a18333f631d0cd3e5a.jpg)

![](images/feb4d58bcaf9c01fad47444b32e909a9d913651fcea7c5427ee5b5a2173037e1.jpg)

![](images/ae8cd2a4cfeeabc389653cca077158d53fd412d2972897919dfd8e56991d2895.jpg)

![](images/88c743bb95f58ca10e4e3e4b8680faf2705153324615612a4145c6db49181709.jpg)

![](images/ac4391c4de0c248be7b7ca7d3875587934a2b7b273e3bc06c6cf4b63de14c374.jpg)

Figure 13: Reuters held-out diagnostic evolution under multi3 with corruption rate 0.3 (mean std over 5 runs). Same panel definitions as in Fig. 3.  
![](images/198aa0089716b0ccf584a42d8798cce279860aa705edaf68ec100f836e066377.jpg)

![](images/7e533ec49a3fb949415b3de9be8707bb3402966000dcfa94cb0b6beff539f6e7.jpg)

![](images/b9b5e6cf5c7b53170d8b64c0dd77bbed47aaeeab841a3c921c5cdc811ce8a110.jpg)

![](images/089f633f8b928767bf0e793bacf0bb4d8766a4e862c687e2938cd6ddeaf1ebd8.jpg)

![](images/0d199a83af7ea7427c18f8b29578d51e018dee4ffd135e2da2b957c59782ae0a.jpg)

![](images/d1d2ac7421021697eceecd88a6e6d2dfbd9c20f60ccd2d795cafbe069c82ac28.jpg)

![](images/0157f314e7a11b8a1948c0c4009946ac45e7dc145f516761c4d43fbd96b10699.jpg)

![](images/14e6b9015961ef0507ea2b8302eea16d542702c8f8443cc95c597262dc9be4fd.jpg)

Figure 14: STL10 held-out diagnostic evolution under multi3 with corruption rate 0.3 (mean std over 5 runs). Same panel definitions as in Fig. 3.  
![](images/35ec52c3e090cebbfac098ac26f9dd2a4e9989729c4752da20ed02a0da476ed9.jpg)

![](images/1d61bbb06941b74518ff261e60c80e1116cd54071b89c72bee558246b55f40b6.jpg)

![](images/ba2911f87419eb4a5e93baa6410318c7b106f69d2d489a062971c7c3827a1e85.jpg)

![](images/ade1c6dd52f9605b2b7601ff268222e19e0dfa0d9dda478b0277833c4c395e13.jpg)

![](images/71cbe4e9ef4212697730d64a8bca49faaf8925b9820d1620011e82f696cef8ee.jpg)

![](images/51208f5ee4fc10160f00b197c1879dec5d22cc1e914cf68f3aaf86a21bcd0799.jpg)

![](images/817f5b0f4101606796cf6d6f072eb3143b081056cb6b8679d41d375e7410de4f.jpg)

![](images/80824a73f3badd206905724f0280e294e16a8cdfbdab2847937bde960146afc3.jpg)  
Figure 15: RCV1-10 held-out diagnostic evolution under multi3 with corruption rate 0.3 (mean std over 5 runs). Same panel definitions as in Fig. 3.

Screening against injected corruption. Panels (E,F) evaluate whether the reliability signal aligns with the injected corruption indicator. Under multi3, $\mathrm { A U C _ { c o r r u p t } }$ and $\mathrm { A P _ { c o r r u p t } }$ are generally above their chance baselines, so corrupted records tend to be ranked higher by the screening signal. The alignment is clearer on most image datasets, especially CIFAR100-20, CIFAR10, ImageNet10, MNIST, and STL10, but weaker or noisier on Reuters and particularly RCV1-10. This limitation reflects the heuristic nature of the screening signal: unreliability may arise not only from stochastic corruption, but also from expert-side epistemic variation, imperfect estimation/correction, and partial self-confirmation during training. The oracle-reference curves in panels (G,H) support this interpretation, as they usually track corruption more closely than the multi3 curves when the clean supervision is generated from the hard oracle relation before corruption.

Summary. Overall, the diagnostics provide consistent qualitative evidence for the intended behavior of ECI-PP, although the curves are not uniformly monotone across datasets. The internal estimates improve mainly in the early stage, the correction discrepancy is reduced, and the screening signal remains meaningfully associated with injected corruption. RCV1-10 is the most challenging case, consistent with the broader experimental picture under severe class imbalance, where improvements in clustering structure are less uniformly reflected by the held-out diagnostic curves. Early stabilization further suggests that the ECI components often reach useful internal agreement before the fixed training horizon; however, we keep this schedule here and leave principled early stopping for future work. These diagnostics therefore provide practical held-out evidence for ECI-PP’s intended behavior, while remaining proxy measurements rather than exact tests of latent aleatoric recovery or pure corruption detection.

## D.6 Sensitivity and ablation studies

We further study the main ECI-PP design choices around the default configuration specified in Section 5.1. All sensitivity runs use the default single-expert supervision condition, i.e., lv0.01 expert quality, corruption rate 0.3, and 9k constraints. To expose the effect of each design choice, we use random initialization without unsupervised pretraining and vary one factor while keeping the others fixed. Figs. 16 to 23 report the corresponding test ACC, NMI, and ARI.

Estimator choices. Figs. 16 and 17 examine the Estimator-side choices. For the number of folds, increasing K from 2 to 20 gives mild gains on several datasets, such as MNIST, FMNIST, CIFAR10, and ImageNet10, but the curves are mostly flat and the improvement is not proportional to the extra cross-fitting cost. This supports using K = 5 as a practical default: it avoids the weakest cross-fitting setting while keeping the main Estimator cost moderate. The information-based warm-up weight κ<sub>i</sub> provides a small but consistent benefit. Enabling it is usually better or comparable across datasets, with visible gains on MNIST, FMNIST, Reuters, and CIFAR10, indicating that informative constraints are useful for stabilizing the early Estimator signal. This agrees with the main comparison in Table 1, where Weighted ProbPair generally improves over ProbPair by using the same information-based weighting principle. At the same time, the gains remain moderate, suggesting that ECI-PP benefits from this design without being overly dependent on it.

Corrector choices. Figs. 18 and 19 study Corrector capacity and regularization. For capacity, we compare a large encoder-sized Corrector (500–500–2000), a medium classifier-sized Corrector (512–512), and the default small Corrector (64–16). The three sizes perform similarly on most datasets, and this supports the default use of a small Corrector: it is cheaper and also limits the risk of fitting arbitrary pair-specific deviations rather than structured expert-conditioned correction. The regularization strength $\lambda _ { \mathrm { c o r } }$ shows the same trade-off more directly. Compared with $\lambda _ { \mathrm { c o r } } = 0 .$ , mild to moderate regularization can help on datasets such as FMNIST, CIFAR10, CIFAR100-20, and RCV1-10, but overly strong regularization is harmful on some datasets, most clearly Reuters when $\lambda _ { \mathrm { c o r } }$ reaches 5 or 10. Thus, the Corrector should not be left completely unconstrained, but it also should not be forced too strongly toward zero correction. The default $\dot { \lambda } _ { \mathrm { c o r } } = 0 . 5$ lies in the stable middle range.

![](images/71f85c3f96cd0ad36291ac9f94f3bd7457a01f266b052126c09a559e171cd79a.jpg)

![](images/72fccfadba2e6c6d1544530e0bd79f512aa123122e49a3ee12d31773214e479d.jpg)  
CIFAR100-20 CIFAR10 FMNIST IMAGENET10 MNIST REUTERS STL10 RCV1-10

![](images/71b464962161574757f75e4e7b7c5abda5f4cde3f18e6d5fe61cede6561e4e06.jpg)  
Figure 16: Sensitivity to the number of estimator folds K. Each panel reports test ACC, NMI, or ARI across datasets (mean std over 5 runs).

![](images/3c29728de6a87ba08df4d1b52d3d0ead8b74decd99001b77b1dd302888b8b8b9.jpg)

![](images/034a282e415a683828a018512cdb9cf77af8b9f092c4aeb98597aad105755a46.jpg)

![](images/781f485d37ad3ebe290ddd029b35d3a2aeba9d4e84b02db7da21e8d09ca4d562.jpg)  
Figure 17: Ablation of the information-based warm-up weight $\kappa _ { i } .$ Each panel reports test ACC, NMI, or ARI across datasets (mean std over 5 runs).

![](images/86d8f8d96a52d3d63e90a43f5dc968014284b9ff6ca028d415e40f0d399ea58f.jpg)

![](images/bd073e59f1ed8394a19f542e592b5e54160ab24c615d34c0ec28d3d4c8405f88.jpg)

![](images/9fd9939fec81ce8bb5fad5ff8b68abf7b21513dca9dd579d928fb1d215ecc2fa.jpg)  
Figure 18: Sensitivity to Corrector capacity. Each panel reports test ACC, NMI, or ARI across datasets (mean std over 5 runs).

![](images/76b876a68aa3d3bfc9613c4d8a5b0b9307432f6a5e81460214def80252dbd68a.jpg)

![](images/d79619a7cb2886d9b8ec13ddfaaae8fa3396b2757cb7c29d36442add77aee1c8.jpg)

![](images/4ec7bff4ba6e7fce3055445cd7126a2cc1996676c00f339d7afeb4fd17662913.jpg)  
CIFAR100-20 CIFAR10 FMNIST IMAGENET10 MNIST REUTERS STL10 RCV1-10  
Figure 19: Sensitivity to the Corrector regularization strength $\lambda _ { \mathrm { c o r } }$ . Each panel reports test ACC, NMI, or ARI across datasets (mean std over 5 runs).

Reliability screening and boundary handling. Figs. 20 to 22 evaluate the parameters used after correction. The screening sharpness $\gamma$ is robust over the tested range, especially for $\gamma \in [ 5 , 2 0 ]$ Larger values are useful on some difficult datasets, notably RCV1-10 and also Reuters or MNIST in some metrics, indicating that stronger suppression of unreliable constraints can help when supervision is more fragile. The Bayesian-confidence scale $n _ { 0 }$ is also stable overall. Larger $n _ { 0 }$ improves RCV1- 10 and slightly helps Reuters, suggesting that these datasets benefit from assigning more evidence to the refined supervision, whereas FMNIST does not benefit from the same increase. Thus, $n _ { 0 }$ controls a real reliability trade-off, but the default value remains within a safe region. Finally, performance changes little as the soft-clipping sharpness ξ varies: most datasets change little from ξ = 5 to 50, with only mild fluctuations on RCV1-10 and a few small metric-specific changes. This supports soft clipping as a robust boundary-handling design for keeping corrected relations in a valid probability range without relying on a finely tuned clipping sharpness.

Reconstruction strength. Fig. 23 shows a noticeable dataset-dependent trade-off for $\lambda _ { \mathrm { { r e c } } } .$ . The default value $\lambda _ { \mathrm { r e c } } = 0 . 0 2$ follows SpherePair [3] and remains a reliable setting, but ECI-PP often benefits from moderately stronger reconstruction. For example, MNIST improves steadily as $\lambda _ { \mathrm { { r e c } } }$ increases, while Reuters and RCV1-10 prefer a moderate range around $0 . 0 5 { - } 0 . 2 $ before performance drops at larger values. In contrast, FMNIST and CIFAR10 do not benefit from overly large reconstruction weights, and ImageNet10 is almost unchanged. This suggests that reconstruction provides useful supervision-independent structure for out-of-fold estimation and final integration, but too much reconstruction can compete with clustering-oriented pairwise learning on some datasets. When validation-based tuning is available, $\lambda _ { \mathrm { { r e c } } }$ is therefore a meaningful parameter to adjust; without such tuning, the shared default remains a fair and stable compromise.

![](images/397c82c26e2606826de05d6524dd49e7180b0f746501d67d6e35553a9760c0db.jpg)

![](images/54a55ce27772404e17ff1359135c8fb236c800f61d829db84afe2b8d881e1e39.jpg)

![](images/eb2281a56f4bf27f6cacb46e58b5b605dda49ecb3fe5a3b922e6704df2c0c65f.jpg)  
CIFAR100-20 CIFAR10 FMNIST IMAGENET10 MNIST REUTERS STL10 RCV1-10  
Figure 20: Sensitivity to the reliability-screening sharpness $\gamma .$ Each panel reports test ACC, NMI, or ARI across datasets (mean std over 5 runs).

![](images/a5e0781a91226d71a00f409857212fad1e54b61834c789c25763990d207de464.jpg)

![](images/09322c807adbe37d60f300404078430050539538c119732b482062edcce17364.jpg)

![](images/a784ce265c0262ceb9e0700677696ce69c656013abd6e61961447c26b5ecbf5c.jpg)  
CIFAR100-20 CIFAR10 FMNIST IMAGENET10 MNIST REUTERS STL10 RCV1-10  
Figure 21: Sensitivity to the Bayesian-confidence scale $n _ { 0 } .$ . Each panel reports test ACC, NMI, or ARI across datasets (mean std over 5 runs).

![](images/6b0745daee37a99e158cf23667ca0b68d59ce2df558b679288a5fcb61183ac31.jpg)

![](images/bc8876b519564d6936b3a6e994139e60600ffb322cbe51ca94f2008d0dbf9201.jpg)  
CIFAR100-20 CIFAR10 FMNIST IMAGENET10 MNIST REUTERS STL10 RCV1-10

![](images/aa38e882e981c9ddef722e8977cbd7c08a80e2d02954359f7de10a74cec45570.jpg)

Figure 22: Sensitivity to the soft-clipping sharpness ξ. Each panel reports test ACC, NMI, or ARI across datasets (mean std over 5 runs).  
![](images/90dff6fe36dd09ffe3cbe4e9b1c1e216a6393990e66a8c682962d56469e6d54f.jpg)

![](images/d6aff75b2eebf3412c38c3cd3fa67c6ed1b9899d85f00b21da4268383008ba19.jpg)

![](images/dddda8100fbd10b609dc34d0e48ff4f3b494fd7d78f05f76fff7af6cf2a361ae.jpg)  
CIFAR100-20 CIFAR10 FMNIST IMAGENET10 MNIST REUTERS STL10 RCV1-10  
Figure 23: Sensitivity to the reconstruction weight $\lambda _ { \mathrm { { r e c } } }$ . Each panel reports test ACC, NMI, or ARI across datasets (mean std over 5 runs).

Overall, the sensitivity and ablation results support the use of the default ECI-PP configuration in the main experiments. Estimator, Corrector, screening, and soft-clipping choices are stable over broad ranges, while the more dataset-dependent reconstruction weight changes performance gradually rather than causing abrupt failure. The results therefore indicate that ECI-PP is not hyperparameter-fragile, even though dataset-specific validation could still improve individual settings when such tuning is allowed.

## E Learning efficiency

Table 11: Overall wall-clock training time for different methods on eight datasets under the multi3 multi-expert supervision setting with 9k constraints and corruption probability 0.3, measured on a single NVIDIA A100 40GB GPU. Methods marked with <sup>∗</sup> require hyperparameter tuning, and their corresponding times are underlined.
<table><tr><td></td><td>CIFAR100-20</td><td>CIFAR10</td><td>FMNIST</td><td>ImageNet10</td><td>MNIST</td><td>Reuters</td><td>STL10</td><td>RCV1-10</td></tr><tr><td>VanillaDCC</td><td>1m12s</td><td>1m10s</td><td>1m20s</td><td>0m45s</td><td>1m27s</td><td>0m37s</td><td>0m41s</td><td>1m51s</td></tr><tr><td>VolMaxDCC*</td><td>10m23s</td><td>45m21s</td><td>1h43m18s</td><td>6m20s</td><td>1h35m30s</td><td>5m00s</td><td>3m06s</td><td>2h39m25s</td></tr><tr><td>CIDEC</td><td>18m27s</td><td>19m26s</td><td>24m49s</td><td>5m03s</td><td>25m06s</td><td>5m53s</td><td>6m03s</td><td>1h07m26s</td></tr><tr><td>SpherePair</td><td>18m27s</td><td>17m58s</td><td>22m25s</td><td>5m09s</td><td>22m54s</td><td>6m05s</td><td>5m18s</td><td>1h01m54s</td></tr><tr><td>SpherePair (EA)*</td><td>3h53m10s</td><td>3h40m47s</td><td>4h30m35s</td><td>1h49m25s</td><td>4h33m48s</td><td>2h12m12s</td><td>1h51m31s</td><td>10h40m49s</td></tr><tr><td>ProbPair</td><td>18m19s</td><td>17m42s</td><td>22m16s</td><td>4m57s</td><td>22m41s</td><td>5m49s</td><td>5m02s</td><td>1h01m36s</td></tr><tr><td>Weighted ProbPair</td><td>18m24s</td><td>17m44s</td><td>22m23s</td><td>5m07s</td><td>22m56s</td><td>5m52s</td><td>5m06s</td><td>1h01m37s</td></tr><tr><td>Weighted ProbPair (EA)*</td><td>3h54m42s</td><td>3h34m20s</td><td>4h24m15s</td><td>1h44m36s</td><td>4h22m09s</td><td>2h01m15s</td><td>1h46m31s</td><td>10h32m39s</td></tr><tr><td>ECI-PP (Ours)</td><td>29m48s</td><td>29m12s</td><td>34m01s</td><td>15m49s</td><td>34m39s</td><td>17m34s</td><td>15m59s</td><td>1h15m41s</td></tr></table>

Overall training time. Appendix B.2 analyzes the computational complexity of ECI-PP; here, based on the implementation in Appendix C.5, we report empirical wall-clock measurements for the complete training pipelines in Table 11. Each entry is measured under the multi3 setting with 9k constraints and corruption probability 0.3, and covers the full practical training procedure: (i) for methods requiring unsupervised pretraining, the reported time includes the pretraining stage; (ii) for methods marked with <sup>∗</sup>, it also includes three scans over the corresponding hyperparameter list and a final training run with the selected hyperparameter. When both pretraining and hyperparameter tuning are required, the timing includes pretraining on the training split excluding validation samples before tuning, followed by pretraining on the full training split before the final run.

Runtime comparison. The runtimes in Table 11 should be interpreted alongside the clustering results in Appendix D.1. VanillaDCC is fastest, but mainly because collapsed assignment learning triggers early stopping; its performance in Table 8 is far below the competitive methods. VolMaxDCC has variable cost, becoming expensive on FMNIST, MNIST, and RCV1-10 due to validation-based hyperparameter search. For methods with autoencoder pretraining, including CIDEC, SpherePair, ProbPair, Weighted ProbPair, and ECI-PP, overall costs remain on the same order, with pretraining as a shared pipeline component. ECI-PP adds moderate overhead from estimator cross-fitting and correction/integration, but this overhead is modest relative to its empirical gains; in more resourcelimited settings, for example when unsupervised pretraining is unavailable, Appendix D.2 shows an even larger performance advantage. The expert-aware SpherePair and Weighted ProbPair extensions are most expensive because ensemble construction and hyperparameter tuning scale with the number of experts. In contrast, Appendix B.2 shows that ECI-PP incorporates expert identities through expert-conditioned correction without replicating the full training pipeline, so its leading cost scales mainly with the constraint budget rather than the expert count.