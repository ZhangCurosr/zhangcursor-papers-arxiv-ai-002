# Same-State Pairs Make One JEPA Proof of the Platonic Representation Hypothesis Vacuous

A Correction and a Gauge Ledger

Xin Li Shenzhen University of Advanced Technology suat25000420@stu.suat-sz.edu.cn

November 2026

## Abstract

In a Joint-Embedding Predictive Architecture (JEPA), the JEPA proof of the Platonic Representation Hypothesis (PRH) assumes that every observation determines the world state and that a spectral gap makes minimizers unique up to rotation. Under the hard-whitening population objective, same-state positive pairs make the induced world kernel the identity: every centered whitened state function is a global minimizer, and, when $d _ { z } - 1 \geq 2 k$ , two minimizers can have best-rotation squared residual 2k. The uniqueness claim is conditional; for $k < d _ { z } - 1$ where $d _ { z }$ is the number of world states, this limit makes its premise unsatisfiable, so the claim has no content there. The experiments that motivate it leave the assumption: a caption sees only part of an image, and the paper states no experiment meets its hypotheses exactly. The theorem does not reach those experiments, and for $k < d _ { z } \mathrm { ~ - ~ } 1$ its idealized same-state limit is vacuous. One of the paper’s four quantitative results survives: a gap-free lower bound. Its bound vanishes only at perfect scene-explained fraction; it bounds alignment error but cannot certify good-view alignment. A canonical frame is determined only up to a rotation and a latent permutation. Paired samples fix the relative frame at efective rank k; a global rotation remains. Component semantics fix the relative permutation only up to signs; value anchors fix it once the mixing frame is pinned. A resource–gauge ledger records which resource fixes which gauge.

## 1 Introduction

Consider two views of one scene in the paper’s idealized model, where every view determines the world state. The induced positive-pair kernel is the identity: the objective can be driven to zero by any whitened function of the state, so two trained encoders can be orthogonal. This is not an optimization failure. It is what the idealized same-state model asks for.

The numbers are not marginal. With |Z| = 60 world states and k = 8 embedding dimensions, the minimizer set is a Stiefel manifold of dimension 436, while a single rotation orbit has dimension 28. When the state space is large enough, two global minimizers can have best-rotation squared residual 2k, the maximum possible. For k = 3, that is 6 out of 6.

This matters because the Platonic Representation Hypothesis (PRH) conjectures that representations converge as models scale [1]; a recent JEPA-based theorem states a precise version, agreement up to one rotation [2].<sup>12</sup> The theorem assumes injective renderings and that minimizers are unique up to rotation; its Proposition 1 gives this under a spectral gap. The experiments that motivate it leave the injective assumption: a caption sees only part of an image. The distinction between the idealized limit and the experiments matters for evaluation: agreement can be high because two encoders share a subspace while the rotation between them stays a free gauge, so an agreement number computed through a fitted rotation certifies an aligning rotation, not the uniqueness assumption on which the theorem rests. The question is whether the theorem applies to the evidence it is meant to explain, and what information would make “identifiable up to a rotation” a scientific statement. This note separates the two questions. It ofers a correction and a resource–gauge ledger, not a new convergence theorem.

The paper’s formal object is more general than its motivating sentence. Appendix A.4 of [2] lets good views be any symmetric positive-pair law $( z , z ^ { \prime } ) \sim \pi$ , while its Introduction and Figure 1 describe them as “two observations of one scene.” In the injective model the latter reading gives $z = z ^ { \prime } { : }$ two observations of one scene are two renderings of one state. We therefore read the theorem under both readings. If π is general, the paper’s Theorem 1 is conditional on a uniqueness premise that the experiments do not establish; if the same-state reading is intended and $1 \leq k < d _ { z } - 1$ Lemma 1 makes that premise unsatisfiable. The two readings support the same conclusion about the evidence: the theorem does not carry over to the experiments.

We establish four claims. The labels are evidence levels: L0 is proved at the population level, L1 is a classical condition with order-level rates, and L2 is a synthetic numerical check.

C1 (L0). At the state level, same-state positive pairs make the induced world kernel the identity, and every   
centered whitened state function is a global minimizer. Under an injective observation, this lifts to encoders:   
the pair operator is a conditional-expectation projection.   
C2 (L0). When there are at least 2k centered state dimensions, two global minimizers can have best-rotation   
squared residual 2k, the maximum possible. The gap-assuming uniqueness theorem is therefore vacuous on the   
same-state regime of the model; the experiments lie outside the theorem’s premises.   
C3 (L0). For the good-view alignment claim, one of the paper’s four quantitative representation results   
survives: a gap-free lower bound. At perfect scene-explained fraction its bound is zero; for imperfect views it is   
positive but cannot certify good-view alignment.   
C4 (L0/L1). Identifiability requires two gauges. Paired samples of efective rank k fix the continuous relative   
O(k) gauge; a global O(k) remains. Component semantics identify the relative permutation only up to signs;   
value anchors fix it once the mixing frame is pinned. Neither is supplied by the JEPA objective.

Recent work on representation identifiability and PRH studies near-identifiability [6], lattice structure [7], displacement geometry [8], structure-over-geometry [9], and multi-way alignment [10]. These papers study diferent notions of representation convergence. Two concurrent notes sharpen the boundary. Xu [4] gives globally certified counterexamples for the Gaussian-regularized extension: pooled Gaussian matching can make every global minimizer retain nuisance, so uniqueness fails for specified soft objectives. That result targets the soft objective; this note targets the hard-whitening population objective in the same-state limit. Syrota et al. [5] characterize when unpaired alignmen is identifiable from the symmetries of the data distribution; our gauge ledger is the paired and anchored counterpart. This note asks whether a specific proof applies to its own evidence.

The rest of the note answers three questions. (Q1) Does the theorem apply? On the experiments it does not: caption views are lossy, so the injective-rendering premise fails. Within the premise, same-state pairs make the kernel diagonal and the uniqueness theorem vacuous (Section 2). (Q2) What survives? One gap-free lower bound survives, but it cannot certify good-view alignment (Section 3). (Q3) What would identifiability take? Two gauges must be fixed, one continuous and one discrete (Section 4). Section 5 checks the claims on small CPU experiments. Section 6 states what changes for PRH and JEPA.

## 2 Question 1: Does the theorem apply?

This section separates two failures. It first records what happens when positive pairs join observations of the same world state: the induced world kernel is the identity and the uniqueness conclusion collapses. It then explains why the paper’s experiments are outside the theorem’s premise.

## 2.1 The pair kernel degenerates

Notation. Z is a finite world with $p ( z ) > 0$ and $d _ { z } = \vert Z \vert ; \pi$ is a symmetric positive-pair law with both marginals $p ; \Pi _ { z z ^ { \prime } } = \pi ( z , z ^ { \prime } ) , D = \mathrm { d i a g } ( p ) , K _ { \pi } = D ^ { - 1 / 2 } \Pi D ^ { - 1 / 2 }$ is the world kernel, and $T _ { \pi } = D ^ { - 1 } \Pi$ is the associated Markov operator on state functions. An encoder is a map $h : Z \to \mathbb { R } ^ { k }$ ; it is centered if $\mathbb { E } h = 0$ and whitened if $\mathbb { E } h h ^ { \top } = I _ { k }$ . The hard-whitening population objective minimizes the prediction term subject to whitening; Ω denotes the anti-collapse term, and $\Omega \in \{ 0 , \infty \}$ in the hard limit. The scene-explained fraction $\rho$ is defined in Section 3. The state-kernel operator $T _ { \pi }$ acts on the centered subspace of $L ^ { 2 } ( p _ { Z } )$ ; the observation-level operator $T _ { s } ,$ , introduced in Section 4, acts on the centered subspace of $L ^ { 2 } ( P _ { X } )$ , where $\lambda _ { 1 } \geq \lambda _ { 2 } \geq \cdots$ are its eigenvalues. Write $p _ { Z } = p$ for the state marginal. A spectral JEPA of dimension k minimizes

$$
\begin{array} { r } { \mathbb { E } _ { ( z , z ^ { \prime } ) \sim \pi } \| h ( z ) - h ( z ^ { \prime } ) \| ^ { 2 } \quad \mathrm { s u b j e c t ~ t o } \quad \Omega ( h ) = 0 } \end{array}
$$

over centered, whitened $h : Z \to \mathbb { R } ^ { k } ;$ ; this is the hard-whitening population objective, with Ω the anti-collapse regularizer and $\Omega \in \{ 0 , \infty \}$ its hard constraint. Write $L _ { \pi } ( h ) = \mathbb { E } _ { ( z , z \prime ) \sim \pi } \| h ( z ) - h ( z \prime ) \| ^ { 2 }$ for this prediction term. The paper also studies a soft SIGReg term [3]; its Proposition 1 does not cover uniqueness under that term, and we do not use soft-SIGReg minimizers here. On the hard objective, for $1 \leq k < d _ { z } - 1$ , minimizers are unique up to rotation only when the k-th eigenvalue of $K _ { \pi }$ exceeds the next, and at $k = d _ { z } - 1$ uniqueness is trivial. This is the paper’s Proposition 1. Let $d _ { z } = | Z |$

Lemma 1 (State level, L0). On the hard-whitening population objective, $i f \pi = \deg ( p )$ , then $K _ { \pi } = I$ and $T _ { \pi } = I$ . Every centered, whitened h satisfies $\Omega ( h ) = 0$ and has zero prediction term. Every eigenvalue of $K _ { \pi }$ is 1. For every $1 \leq k < d _ { z } - 1$ , every k-dimensional subspace of the centered state-function space is the span of some global minimizer; at $k = d _ { z } - 1$ every minimizer spans the whole centered space, so uniqueness holds trivially. For $1 \leq k < d _ { z } - 1$ , the uniqueness premise of the paper’s Theorem 1 fails.

Proof. A same-state pair has $z = z ^ { \prime }$ almost surely, so $L _ { \pi } ( h ) = 0$ for every centered whitened h and $K _ { \pi } = I$ . The eigenvalue-1 eigenspace is the whole state-function space; restricted to centered functions it has dimension $d _ { z } - 1$ □

In words: same-state positives never join two diferent states, so the state graph has no cross-state edges; the prediction term therefore charges nothing for any function of the state, and every centered whitened such function is a global minimizer.

Remark 1 (Observation level, L0). If a view is an injective rendering of $Z ,$ the observation pair operator is the conditional-expectation projection realizing $T _ { \pi }$ , so the state-level conclusion lifts to encoders. Captions are not injective; the paper says so. The state-level argument itself needs no injectivity.

For $| Z | = 6 0$ and $k = 8 .$ , the minimizer set has dimension $k ( | Z | - 1 ) - k ( k + 1 ) / 2 = 4 3 6$ . A single rotation orbit has dimension 28. “Unique up to rotation” fails by a wide margin.

## 2.2 The failure is orbit-level

A natural weakening says that every minimizer is approximately related to every other by a rotation.   
This is also false.

Proposition 1 (Orbit-level failure, L0). Let $\pi = \operatorname { d i a g } ( p )$ and $1 \leq k < d _ { z } - 1 . \ I f d _ { z } - 1 \geq 2 k$ , there exist global minimizers $f , g$ of the hard-whitening objective such that

$$
\mathbb { E } \Vert f - Q g \Vert ^ { 2 } = 2 k \quad f o r \ e v e r y \ Q \in O ( k ) ,\tag{1}
$$

in particular, the best-rotation residual is 2k, the maximum possible. The smallest nontrivial instance is $| Z | = 3 , k = 1$ : any two orthogonal centered unit-variance state functions are global minimizers with best-rotation squared residual 2.

Proof. Lemma 1 makes every centered whitened state function a global minimizer. Choose 2k mutually orthogonal, centered, unit-variance state functions and split them into two blocks f and $g .$ Both are global minimizers and $\mathbb { E } [ f g ^ { \top } ] = 0$ . Hence, for every $Q \in O ( k )$ ，

$$
\begin{array} { r } { \mathbb { E } \| f - Q g \| ^ { 2 } = \| f \| ^ { 2 } + \| g \| ^ { 2 } - 2 \operatorname { t r } ( Q \mathbb { E } [ g f ^ { \top } ] ) = 2 k . } \end{array}
$$

For the best-rotation residual, whitening makes every singular value of $\mathbb { E } [ f g ^ { \top } ]$ at most one, so mi $\boldsymbol { 1 } _ { Q \in O ( k ) } \mathbb { E } \| \boldsymbol { f } - Q g \| ^ { 2 }$ lies in [0, 2k] and equals 2k exactly when the two subspaces are orthogonal. At $d _ { z } - 1 \geq 2 k$ the construction is feasible. For $k = 3$ a numerical instance gives residual 6.0000 out of 6. □

The construction is feasible only when the centered state-function space has dimension at least $2 k ;$ at $d _ { z } = 2 k$ it is not. The smallest case $| Z | = 3 , k = 1$ is minimal because $| Z | = 2 \mathrm { g i v e s } k = d _ { z } - 1$ the boundary at which Lemma 1 says uniqueness is trivial.

## 2.3 Which results survive the degeneracy?

The paper’s Theorem 1 assumes that minimizers are unique up to rotation; its Proposition 1 supplies this under a spectral gap. The paper’s Theorems 2 and 4 and Proposition 3(i) also require a gap. At same-state positives with $1 \leq k < d _ { z } - 1$ , uniqueness fails and the gap is absent. The paper’s Theorem 3 requires no gap. For the good-view alignment claim, Theorem 3 is the only one of the four that survives the degeneracy.

Flickr and Recap compare caption views of one image. At the world-kernel level these pairs are same-state, so the induced world kernel is the identity. But the views are captions, not injective renderings, and the paper notes that this premise fails [2]. For these experiments the paper’s Theorem 1 is inapplicable because the injectivity premise fails; in the suficient-rendering limit and for $1 \leq k < d _ { z } - 1$ it would be vacuous because the gap is absent. MSR-VTT treats frames of a clip as several observations of one scene. On that reading both image and caption world kernels are the identity and agree. The mismatch arises only under the time-index reading, where frames are not same-state. For the synthetic world of Fig. 12 of [2], five classes of six states each are used $( | Z | = 3 0 )$ . For the class-inner complete kernel with self-links, the natural reading of the caption, we compute the uncentered spectrum $1 ^ { \times 5 } , 0 ^ { \times 2 5 }$ ; after centering, the spectrum is $1 ^ { \times 4 } , 0 ^ { \times 2 5 }$ . For this kernel, the only nonzero gap of the centered kernel occurs at $k = C - 1 = 4$ with $C = 5$ , which is exactly the evaluated point. Chain-like kernels can produce additional gaps, so this last claim is kernel-dependent.

Table 1 records the mapping: for $1 \leq k < d _ { z } - 1$ the first, second, fourth, and fifth rows lose their premise in the same-state limit, while Theorem 3 survives as a bound on alignment error, not as a good-view certificate.

<table><tr><td>Original result</td><td>What it asserts</td><td>Premise relevant here</td><td>Status in the same-state limit and in the experiments</td></tr><tr><td>Theorem 1 (with Proposition 1)</td><td>shared representation up to one rotation</td><td>injective renderings; shared  $\pi ;$  minimizers unique up to rotation (spectral gap)</td><td>premise empty in the same-state limit for  $1 \leq k < d _ { z } - 1 ;$  experiments leave the injective premise</td></tr><tr><td>Theorem 2 (kernel mismatch)</td><td>misalignment bounded by kernel mismatch</td><td>spectral solutions; eigengap  $\delta = \mu _ { k } - \mu _ { k + 1 } > 0$ </td><td>no gap for  $1 \leq k < d _ { z } - 1$  in the same-state limit; at  $k = d _ { z } - 1$  the theorem applies trivially (bound</td></tr><tr><td>Theorem 3 (nuisance costs alignment)</td><td>gap-free lower bound from the scene-explained fraction</td><td>centered, whitened encoders; no gap</td><td>zero) survives; positive bound for imperfect views; cannot certify good-view alignment</td></tr><tr><td>Theorem 4 (views and data set the kernel)</td><td>a spectral solution is close to the kernel&#x27;s spectral embedding, and paired modalities align within the theorem&#x27;s two error terms; a gap separates state from</td><td>gap  $\delta _ { m } =$   $\mu _ { k } - \operatorname* { m a x } ( \mu _ { k + 1 } , \eta _ { m } ) > 0$   $( \eta _ { m } \ \mathrm { l a r g e s t \ n u i s a n c e }$  eigenvalue)</td><td>no gap for  $1 \leq k < d _ { z } - 1$  in the same-state limit; at  $k = d _ { z } - 1$  the theorem applies trivially  $( \tau _ { m } = 0 ) ;$  experiments do not verify its hypotheses</td></tr><tr><td>Proposition 3(i)</td><td>a good-view JEPA learns a function of the state  $( \rho = 1 )$ </td><td>gap  $\mu _ { k } > \operatorname* { m a x } ( \mu _ { k + 1 } , 0 )$ </td><td>no gap for  $1 \leq k < d _ { z } - 1 ;$ </td></tr><tr><td>Proposition 2 (k pairs)</td><td>k pairs in general position identify the rotation</td><td>paired rank k</td><td>trivial at  $k = d _ { z } - 1$  refined below by the</td></tr></table>

Table 1: The original results, their premises, and their status here. “Gap” means the spectral gap used for uniqueness in the original; among Theorems $1 { - } 4 .$ , Theorem 3 is the one result that does not use it. The eigenvalues of the original’s world kernel $K _ { \pi }$ of the relevant positive-pair law are $1 = \mu _ { 0 } \geq \mu _ { 1 } \geq \cdot \cdot \cdot$ , where $\mu _ { | Z | } : = - \infty ,$ and $\tau _ { m }$ is the approximation-error bound of its Theorem 4.

## 3 Question 2: What survives?

This section isolates the one result that survives the degenerate regime and states its limit.

The survivor is a lower bound on alignment error. It uses the scene-explained fraction of an encoder, $\rho ( f ) = \mathrm { t r } ( C _ { f } ^ { 1 / 2 } ) / k$ , where $C _ { f } = \operatorname { C o v } ( \mathbb { E } [ f ( X ) \mid Z ] )$ is the covariance of the scene-related part of $f .$ The paper’s Theorem $3 \ [ 2 ]$ states the bound we use: for centered, whitened encoders $f _ { 1 } , f _ { 2 }$ of paired observations of one state with independent nuisances,

$$
\operatorname* { m i n } _ { Q \in O ( k ) } \mathbb { E } \| f _ { 1 } - Q f _ { 2 } \| ^ { 2 } \geq 2 k \left( 1 - \operatorname* { m i n } _ { m \in \{ 1 , 2 \} } \rho ( f _ { m } ) \right) .\tag{2}
$$

The original theorem states this minimum over $Q ;$ the for-every-Q form follows. It is a bound on aligned error, not an overlap between the two views’ signal subspaces. The bound needs no spectral gap, so Lemma 1 does not afect it. In one synthetic construction (L2), $\rho \approx 0 . 7 9$ and 0.497 give errors 2.09 and 4.14 against bounds 1.26 and 3.02, respectively.

At the suficient-rendering limit $( \rho = 1 )$ the bound is zero, but the constructed residuals are not zero: the bound is not attained. Two synthetic constructions (L2) give squared residuals 1.34 and 4.58 for these $k = 3$ constructions, where the maximum possible residual is $2 k = 6$ (the first from the final $\rho = 1$ construction printed by verify review.py (last line), the second from N1-5 of verify N1.py; both single fixed-seed draws). In that limit the good-view encoders are state functions $( \rho = 1 )$ , so the bound is zero. Proposition 2 shows that a zero bound carries no alignment information.

This limits the paper’s claim, not its experiments. A theorem being inapplicable is not the same as the experiment being wrong. The experiments remain evidence for a controlled ordering of view quality. They do not establish convergence up to rotation.

Proposition 2 (Limits of the survivor, L0). At $\rho = 1$ the bound (2) is zero. $I f d _ { z } - 1 \geq 2 k$ , there exist centered, whitened state functions $f , g$ with $\rho ( f ) = \rho ( g ) = 1$ and $\begin{array} { r } { \mathbb { E } \| f - Q g \| ^ { 2 } = 2 k } \end{array}$ for every $Q \in O ( k )$ . The bound therefore cannot certify good-view alignment.

Proof. A centered, whitened state function has $C _ { f } = I _ { k }$ , hence $\rho ( f ) = 1$ ; the lower bound is then $2 k ( 1 - 1 ) = 0$ . The second claim is the construction of Proposition 1: the two blocks are state functions and are whitened, and the residual is 2k for every orthogonal Q. □

## 4 Question 3: What would identifiability take?

This section answers the constructive question. If same-state pairs carry no identifiabili ${ \mathrm { , y , } }$ what does? Two gauges organize the answer.

## 4.1 A subspace, not a frame

The first object to identify is the subspace that is actually identifiable. Let X be an observation with law $P _ { X }$ and channel $P _ { X \mid Z }$ . A same-state pair $( X , X ^ { \prime } ) \sim P _ { s }$ is one in which, given $Z = z$ , the two views are conditionally independent with common conditional law $P _ { X \mid Z = z }$ (hence common marginal $P _ { X } )$ For such pairs, define $T _ { s } = A A ^ { * }$ with $( A \varphi ) ( x ) = \mathbb { E } [ \varphi ( Z ) \mid X = x ]$ and $A ^ { * } f ( z ) = \mathbb { E } [ f ( X ) \mid Z = z ]$ For $0 \leq \varepsilon \leq 1$ , expand the loss for the positive-pair mixture $P _ { + } ^ { \varepsilon } = \varepsilon P _ { s } + ( 1 - \varepsilon ) P _ { X } \otimes P _ { X }$

$$
L _ { \varepsilon } ( f ) = 2 k - 2 \varepsilon \sum _ { j = 1 } ^ { k } \langle f _ { j } , T _ { s } f _ { j } \rangle .\tag{3}
$$

Theorem 1 (L0). For centered and whitened f and $k \geq 1$ , if $\mathrm { r a n k } ( T _ { s } ) \ge k$ , then for every $\varepsilon > 0$ 2 the span of any global minimizer of (3) is a k-dimensional subspace of the top eigenspace of $T _ { s } ,$ it contains every eigenvector with eigenvalue greater than $\lambda _ { k }$ . If $\lambda _ { k } ( T _ { s } ) > \lambda _ { k + 1 } ( T _ { s } )$ , the span is the unique top-k eigenspace and the encoder is unique up to right multiplication by $O ( k )$ . For $\varepsilon > 0$ , if rank $( T _ { s } ) < k$ , the top-k sum is the sum of all positive eigenvalues; every minimizer contains the positive eigenspace and an arbitrary $\left( k - \mathrm { r a n k } T _ { s } \right)$ -dimensional subspace of ker $T _ { s }$ , and every such subspace occurs as a minimizer; no direction in the kernel is identified. $A t \varepsilon = 0 , L _ { \varepsilon } \equiv 2 k$ and no direction is identifiable.

Here the top eigenspace means the eigenspace of $T _ { s }$ for eigenvalues at least $\lambda _ { k } ;$ a top-k eigenspace is a k-dimensional subspace of it containing every eigenvector with eigenvalue $> \lambda _ { k }$ . The eigenvalues of $T _ { s }$ are squared canonical correlations, so the top-k eigenspace is a canonical correlation analysis (CCA) object [18]. It is also the target of spectral contrastive methods [16, 17]. The rotation in PRH is therefore a gauge of the factorization, not a recoverable coordinate of the world.

## 4.2 How many pairs?

The next question is how many synchronized pairs are needed to fix the continuous gauge. Views form vertices of a pairing graph; an edge carries $m _ { e }$ synchronized pairs. In whitened coordinates, let $( x _ { i } , y _ { i } ) _ { i = 1 } ^ { m _ { e } }$ satisfy $y _ { i } = Q x _ { i }$ for an unknown relative map $Q \in O ( k )$ , and set $\mathbf { X } = [ x _ { 1 } , \ldots , x _ { m _ { e } } ] ^ { \top }$ $\mathbf { Y } = [ y _ { 1 } , \ldots , y _ { m _ { e } } ] ^ { \top }$ , and $r = { \mathrm { r a n k } } ( \mathbf { X } )$ . Thus $\mathbf { Y } = \mathbf { X } Q ^ { \top }$

Theorem 2 $( \mathrm { L 0 } / \mathrm { L 1 } )$ . If $r \ = \ k$ , then $Q$ is uniquely determined by orthogonal Procrustes. $I f$ $r < k ,$ , in an orthonormal basis whose first r vectors span the row space of X, the solution set is $Q \left( I _ { r } \oplus O ( k - r ) \right)$ , with $( k - r ) ( k - r - 1 ) / 2$ continuous gauge dimensions. $\textit { A t r } = k - 1$ only a discrete sign (reflection) remains. With iid zero-mean sub-Gaussian noise of covariance $\sigma ^ { 2 } I _ { k }$ , independent of X, iid sub-Gaussian covariates with $\mathbb { E } [ x _ { i } x _ { i } ^ { \top } ] = I _ { k } ,$ and $m _ { e } \ge c _ { 0 } k$ for a suficiently large fixed constant $c _ { 0 } > 1$ , the orthogonal Procrustes estimate has operator-norm error $O _ { P } ( \sigma \sqrt { k / m _ { e } } )$ ; consequently $m _ { e } \gtrsim k \sigma ^ { 2 } / \tau ^ { 2 }$ sufices for target error τ.

The paper’s Proposition 2 already shows that k pairs in general position identify the rotation. The theorem records the partial-rank boundary, the reflection at $r = k - 1$ , and the noise cost; it refines that proposition rather than replacing it. Even with a connected pairing graph and $m _ { e } \to \infty$ on every edge, each connected component retains one common $O ( k )$ . If c is the number of connected components, the residual is $O ( k ) ^ { c }$ . Paired data fix only the relative frames, never the global one.

## 4.3 Which structure helps?

For the gauge discussion set $d = k \colon$ the latent dimension is $k ,$ and the permutation section writes d for this same number of latent coordinates. The next question is which known noise or constraints can break the continuous symmetry. Homoscedastic known noise is neutral. If the joint noise covariance Σ is known, invertible and independent of $z ,$ block whitening is a reversible reparameterization. It leaves the identifiability group unchanged. Rotation-invariant constraints, such as the full space, the orthogonal group, rank or spectral-norm constraints, also leave the $O ( k )$ symmetry intact. The positive semidefinite cone is invariant under conjugation, not under the model’s action $A _ { m } \mapsto A _ { m } Q$ it can break the symmetry.

Heteroscedastic known noise can reduce the symmetry. Let ${ \mathcal { C } } _ { m }$ be the known constraint set of view m. Suppose $N _ { m } \mid Z = z$ has a known conditional law $\eta _ { m } ( \cdot \mid z )$ with the noises $N _ { 1 } , \ldots , N _ { M }$ conditionally independent given $Z ,$ and write $X _ { m } = A _ { m } Z + N _ { m }$ with $A _ { m } \in \mathbb { R } ^ { n _ { m } \times k }$ of full column rank. Assume $p _ { Z }$ is known and fixed. We call $p _ { Z }$ and each $\eta _ { m } ( \cdot \mid z )$ non-degenerate when their supports afinely span $\mathbb { R } ^ { k }$ and the observation space $\mathbb { R } ^ { n _ { m } }$ , respectively. A symmetry acts on latents by $z \mapsto Q ^ { \top } z$ and on mixings by $A _ { m } \mapsto A _ { m } Q$ . Define

$$
{ \mathrm { S t a b } } ( p _ { Z } ) = \{ Q \in O ( k ) : ( Q ^ { \top } ) _ { \# } p _ { Z } = p _ { Z } \} ,
$$

$$
\operatorname { S t a b } ( \eta ) = \{ Q \in O ( k ) : \eta _ { m } ( \cdot \mid Q ^ { \top } z ) = \eta _ { m } ( \cdot \mid z ) \mathrm { ~ f o r ~ a l l ~ } m \mathrm { ~ a n d ~ } p _ { Z ^ { - } } \mathrm { a . e . ~ } z \} ,
$$

$$
G _ { \theta } = \{ Q \in O ( k ) : A _ { m } Q \in \mathcal { C } _ { m } \mathrm { ~ f o r ~ a l l ~ } m \} ,
$$

where θ collects the model parameters.

Proposition 3 (Noise symmetries, L0). Assume the pair $( A _ { m } , \eta _ { m } )$ is identifiable up to $O ( k )$ : two parameter pairs with the same observed law are related by a common orthogonal right factor that preserves $p _ { Z }$ , that is, $A _ { m } ^ { \prime } = A _ { m } R$ and $\eta _ { m } ^ { \prime } ( \cdot \mid z ) = \eta _ { m } ( \cdot \mid R z )$ for some $R \in { \mathrm { S t a b } } ( p _ { Z } )$ and all m, as in the standard factor-analysis/ICA setting (for example, under a non-Gaussian $p _ { Z } )$ . Then, in the full-rank, non-degenerate case $( A _ { m }$ of full column rank; $p _ { Z }$ and each $\eta _ { m } ( \cdot \mid z )$ non-degenerate), a matrix $Q \in O ( k )$ with $A _ { m } Q \in { \mathcal { C } } _ { m }$ for all m is a residual symmetry, that is, it leaves $p _ { Z }$ invariant and the joint law of the $X _ { m }$ unchanged when the mixing is changed to $A _ { m } Q$ with $\eta _ { m }$ held fixed, if and only if

$$
Q \in \mathrm { S t a b } ( p _ { Z } ) \cap \mathrm { S t a b } ( \eta ) \cap G _ { \theta } .\tag{4}
$$

The $ { \mathbf { \hat { \rho } } } _ { 1 }  { \mathbf { f } } ^ { \prime } { }$ direction needs no identifiability: $Q$ leaves $p _ { Z }$ and each $\eta _ { m } ( \cdot { \mathrm { ~ \bf ~ \vert ~ } } z )$ invariant, so $A _ { m } Q ( Q ^ { \top } Z ) + N _ { m }$ has the law of $A _ { m } Z + N _ { m }$ . The converse uses identifiability: an unchanged observed law makes $\left( A _ { m } Q , \eta _ { m } \right)$ an equivalent parameterization of $( A _ { m } , \eta _ { m } )$ ; the two difer by a common orthogonal factor, and full column rank forces that factor to be Q itself, so $Q$ must stabilize η. Without identifiability, (4) is only a suficient family. When $p z$ is rotation-invariant, $\mathrm { S t a b } ( p _ { Z } ) = { \cal O } ( k )$ . For zero-mean Gaussian noise (covariance $\Sigma _ { m } ( z ) )$ , replace $\operatorname { S t a b } ( \eta )$ by $H _ { \Sigma } =$ $\{ Q \in O ( k ) : \Sigma _ { m } ( Q ^ { \top } z ) = \Sigma _ { m } ( z )$ for all m and $p _ { Z ^ { - } } \mathrm { a . e . } ~ z \}$ . If $A _ { m }$ is not of full column rank or the model is not identifiable, the intersection is only a suficient family, not a full characterization. For a non-degenerate $p _ { Z }$ , the example $\Sigma ( z ) = \mathrm { d i a g } ( e ^ { z _ { 1 } } , e ^ { 2 z _ { 2 } } , e ^ { 3 z _ { 3 } } )$ has $H _ { \Sigma } = \{ I \}$ ; under the identifiability assumption its residual set is trivial, so calibrated z-dependent noise can remove the continuous gauge in this example. Whether the heteroscedastic model is identifiable in general is a separate question, which we do not settle. The numerical check is L2; Proposition 3 itself remains L0. Constant isotropic noise leaves $O ( k )$ . If $H _ { \Sigma } = O ( k )$ but $p _ { Z }$ has a finite stabilizer, the intersection is finite rather than continuous.

Constraint sets are analyzed through $G _ { \theta }$ . The setwise stabilizer $H _ { \mathrm { s e t } } = \{ Q : \mathcal { C } _ { m } Q \subseteq \mathcal { C } _ { m } \}$ is only a subgroup. A trivial $H _ { \mathrm { { s e t } } }$ does not imply identifiability. Partial anchor rows spanning $r _ { a }$ dimensions give $O ( k - r _ { a } )$ . Two diferent sparse supports can intersect to a finite sign group. Nonnegativity alone generally leaves a continuous family and becomes a finite permutation group only in special cases such as $A _ { m } = I$

## 4.4 The discrete gauge: permutations

The second gauge is discrete, namely the ordering of latent coordinates. Let d be the number of latent coordinates. Suppose the observations are $X _ { m } = A _ { m } P _ { m } Z + N _ { m }$ , where $P _ { m } \in S _ { d }$ is an unknown permutation, identified with its permutation matrix.

Proposition 4 (Permutation gauge, L0). If $A _ { m }$ is free and of full column rank, the observation depends only on the product $B _ { m } = A _ { m } P _ { m }$ , and for any $P _ { 0 } \in S _ { d }$ the transformation $( A _ { m } , P _ { m } ) \mapsto$ $( A _ { m } P _ { 0 } , P _ { 0 } ^ { - 1 } P _ { m } )$ leaves $B _ { m }$ unchanged. An absolute permutation is therefore not identifiable from the data. With $A _ { m }$ free and of full column rank, the decomposition $B _ { m } = A _ { m } P _ { m }$ has exactly d! solutions, difering by such a $P _ { 0 }$

When each $A _ { m }$ is square and invertible, with the frame fixed, paired data determine only the composite $M _ { m l } = A _ { l } P _ { l } P _ { m } ^ { - 1 } A _ { m } ^ { - 1 }$ . Its decomposition has exactly $S _ { d }$ ambiguity, that ${ \mathrm { i s } } ,$ d! solutions, and $m _ { e }  \infty$ does not remove it. If $p _ { Z }$ is exchangeable, the absolute $S _ { d }$ is an exact gauge even asymptotically. Pairing fixes the relative part of the continuous gauge. It does not fix the permutation gauge.

Relative permutations can be recovered from component semantics. Suppose Z has independent non-Gaussian components with pairwise distinguishable shape fingerprints, such as skewness, kurtosis or quantile shape. Scale alone is normalized away by whitening and is not a fingerprint. With $m _ { e } = 0$ pairs but $n \to \infty$ unpaired samples per view, single-view independent component analysis (ICA) plus fingerprint matching identifies the relative permutation up to signs in the noiseless case (L2, synthetic here) [14, 15, 19, 20]. This is a population statement; finite-sample error is not analyzed here. If components are exchangeable, matching fails. With r correspondence anchors, the residual is a signed-permutation subgroup (a semidirect product in general; at $d - r \leq 1$ it is a direct product); correspondence anchors do not remove signs. A value anchor is the known value of one latent coordinate of an observation; once the frame is pinned, that value, if nonzero, fixes the coordinate’s sign.

<table><tr><td>Resource</td><td>Continuous O(k)?</td><td>Discrete  $S _ { d } ?$ </td></tr><tr><td>Paired samples</td><td>relative frames only (at effective rank k); global O(k) remains</td><td>no</td></tr><tr><td>Homoscedastic known noise Heteroscedastic calibrated</td><td>no (whitening-neutral) exact under identifiability: yes if</td><td>no no (absolute  $S _ { d }$ </td></tr><tr><td>noise</td><td>(4) has no continuous part no (labels only)</td><td>remains a gauge under free factorization) yes, up to a signed-permutation</td></tr><tr><td>Correspondence anchors</td><td>no (the frame must already be</td><td>subgroup yes, for nonzero anchored</td></tr><tr><td>Value anchors (known coordinate values)</td><td>pinned)</td><td>coordinates</td></tr><tr><td>Sparse or nonnegative constraints</td><td>only if rotation symmetry is broken</td><td>sparse: yes; nonnegative: no</td></tr><tr><td>Component shape</td><td>yes, up to signed permutation</td><td>relative permutation, up to signs</td></tr><tr><td>fingerprints Exchangeable pz</td><td>no</td><td>no (absolute  $S _ { d }$  is a gauge)</td></tr></table>

Table 2: The resource–gauge ledger. No resource fixes both gauges exactly in general under free factorization; heteroscedastic calibration is exact under the identifiability assumption of Proposition 3 and only suficient otherwise, and value anchors require a pinned frame. Absolute canonical alignment requires at least one continuous-gauge source and one discrete-gauge source.
<table><tr><td>Test</td><td>Type</td><td>Result</td><td>Statement</td></tr><tr><td>Partial pairing,  $m _ { e } = 0 , 1 , 2 , 3 , 6 4$ </td><td>L2 estimated</td><td> $m _ { e } = 0 ;$  1.14 (no-information gauge),  $m _ { e } = 1 \colon$  one gauge dimension,  $m _ { e } = 2 \colon$  discrete,  $m _ { e } = 3 ; ~ 0 . 0 8 9 , m _ { e } = 6 4 ;$ </td><td>effective-rank threshold</td></tr><tr><td>ICA and component fingerprints,  $m _ { e } = 0$ </td><td>L2 estimated</td><td>0.012 relative-permutation accuracy 1.00  $( n = 2 \times \mathrm { { 1 0 ^ { 3 } - 5 } } \times 1 0 ^ { 4 } ) ;$  same-shape control 0.31 mean  $( 0 . 0 8 – 0 . 5 0 )$  component-level chance 1/3</td><td>discrete gauge needs semantics</td></tr></table>

Table 3: Synthetic checks. The 2k construction is in Section 2.2; the table reports the two estimated rows.

## 4.5 The ledger

Table 2 records which resource fixes which gauge. Component semantics also fixes the continuous frame, up to signed permutations. Value anchors can fix the signs once the frame is pinned; sparse constraints can break both gauges, and nonnegative constraints the continuous one, in special cases. Correspondence anchors fix labels but not the frame. An absolute permutation remains a gauge under free factorization.

## 5 What the numbers say

This section checks the claims on small CPU experiments. None of these checks is evidence for or against the PRH itself. Table 3 contains two L2 estimated rows. A separate joint-covariance invariance check (L2) is reported in Appendix A. The 2k construction is in Section 2.2, not in the table. A reader who accepts the construction can skip this section on a first reading.

The pairing and fingerprint rows locate the two gauges: efective rank fixes the relative part of the continuous gauge (a global O(k) remains), and shape fingerprints fix the relative discrete one

up to signs.

## 6 What changes for PRH and JEPA

The reading of the paper changes in three places; a note on sources follows.

What can be claimed. The controlled ordering, good views better than bad views or nexttoken, remains evidence that positive-pair quality afects alignment. It is not a test of the theorem: the experiments leave the injective-rendering premise, and the paper states that no experiment meets the theorem’s hypotheses exactly, “since our runs use LeJEPA and the image side mostly uses crops as positives,” while its Known limitations section adds that real modalities are lossy. In the same-state limit of the theorem’s hypothesis class, the kernel is the identity and the conclusion is vacuous. The paper’s Theorem 3 remains a valid diagnostic for alignment error, not a good-view certificate.

Two boundaries. The theorem is conditional: we do not claim its implication is false, only that for $1 \leq k < d _ { z } - 1$ the same-state limit makes its antecedent unsatisfiable (Lemma 1). If the general π reading of Appendix A.4 of [2] is intended, the theorem remains conditional and the experiments still fall outside its premises. The soft Gaussian extension is a diferent failure mode, analyzed by Xu [4], and is not covered here.

What would change the picture. A non-vacuous identifiability statement for same-state data would need additional structure, not a spectral gap. It could use paired samples of efective rank k, anchors, calibrated heteroscedastic noise, component fingerprints, or temporal and three-view structure. Even value anchors require a pinned frame; component semantics pins each frame up to signed permutations and the relative permutation up to signs, while the other resources fix at most one gauge. None is supplied by the JEPA objective itself. If future work measures a pair kernel that is not the identity on real views, the vacuous-domain objection would not apply to that setting.

What remains open. The theory here is stated for linear or whitened encoders and for Gaussian or known-noise models. Nonlinear JEPA encoders and real modalities are not covered. The permutation results assume non-Gaussian independent components with distinguishable shapes, which is hard to verify on real embeddings. Two published comparisons remain data-limited rather than erroneous: the unlabeled-data exponent fit in Fig. 11 of [2], because recall-at-1 is bounded and the fit cannot distinguish among exponents consistent with the reported slope, and the singlecheckpoint 42× comparison in Fig. 1 of [2], which confounds repeated data and sample definitions. We mark them as data-limited and do not use them as quantitative support.

Sources. The identifiability tools used here are classical: factor-analysis indeterminacy [11, 12], independent component analysis and label switching [14, 15, 21], mixture identifiability [13], permutation synchronization and graph matching [25, 24], rotation synchronization [26, 27, 28], Procrustes [23], and tensor identifiability [22].

A JEPA-based proof of the PRH can be correct in its algebra and still fail to reach its evidence. Its experiments leave the injective-rendering premise; within that premise, same-state positives make the world kernel the identity, and the minimizer set is a Stiefel manifold, not a rotation orbit. The failure is orbit-level: when $d _ { z } - 1 \geq 2 k$ , two global minimizers can have best-rotation squared residual 2k. One lower bound survives, but it bounds alignment error without certifying good-view alignment. What would make alignment identifiable is additional information: pairs of efective rank k for the continuous gauge, and component semantics or anchors for the permutation gauge. The identifiable formulation of PRH is orbit alignment with an explicit gauge source. An absolute canonical representation is not supplied by the JEPA objective alone.

## A Proofs

This appendix verifies the statements in the main text. The notation is that of Sections 2 and 4; all operators on $L ^ { 2 } ( P _ { X } )$ act on the centered subspace. Lemma 1 and Proposition 1 are proved in the main text; the proofs below cover the remaining claims.

## A.1 Top-k spectral theorem (Theorem 1)

Let $P _ { X }$ be the marginal law of X and let $f = \left( f _ { 1 } , \ldots , f _ { k } \right)$ be centered and whitened, so $\mathbb { E } f = 0$ and $\mathbb { E } f f ^ { \top } = I _ { k }$ . Write $T _ { s } ~ = ~ A A ^ { * }$ , with $A \varphi ( x ) = \mathbb { E } [ \varphi ( Z ) \mid X = x ]$ and $A ^ { * } f ( z ) = \mathbb { E } [ f ( X ) \mid Z = z ]$ . For $( X , X ^ { \prime } ) \sim P _ { s }$ , conditional independence given Z gives

$$
\mathbb { E } \left[ f ( X ) ^ { \top } f ( X ^ { \prime } ) \right] = \sum _ { j = 1 } ^ { k } \mathbb { E } \left[ \mathbb { E } [ f _ { j } ( X ) \mid Z ] \mathbb { E } [ f _ { j } ( X ^ { \prime } ) \mid Z ] \right] = \sum _ { j = 1 } ^ { k } \langle A ^ { * } f _ { j } , A ^ { * } f _ { j } \rangle _ { L ^ { 2 } ( p _ { Z } ) } = \sum _ { j = 1 } ^ { k } \langle f _ { j } , T _ { s } f _ { j } \rangle _ { L ^ { 2 } ( P _ { X } ) } .
$$

For $( X , X ^ { \prime } ) \sim P _ { X }$ ⊗ $P _ { X }$ the same cross term is $\| \mathbb { E } f \| ^ { 2 } = 0$ , so $\mathbb { E } \| f ( X ) - f ( X ^ { \prime } ) \| ^ { 2 } = 2 k$ . Since $P _ { + } ^ { \varepsilon } =$ $\varepsilon P _ { s } + ( 1 - \varepsilon ) P _ { X } \otimes P _ { X }$

$$
L _ { \varepsilon } ( f ) = 2 k - 2 \varepsilon \sum _ { j = 1 } ^ { k } \langle f _ { j } , T _ { s } f _ { j } \rangle .\tag{5}
$$

The operator $T _ { s }$ is self-adjoint and positive semidefinite; let $\lambda _ { 1 } \geq \lambda _ { 2 } \geq \cdots$ · be its eigenvalues on the centered subspace, with orthonormal eigenfunctions $u _ { i }$ . For orthonormal $f _ { 1 } , \ldots , f _ { k }$ , let $P _ { f }$ be the orthogonal projection onto their span. Then

$$
\sum _ { j = 1 } ^ { k } \langle f _ { j } , T _ { s } f _ { j } \rangle = \mathrm { t r } ( P _ { f } T _ { s } ) .
$$

Ky Fan’s maximum principle states that for every orthogonal projection $P$ of rank $k ,$

$$
\mathrm { t r } ( P T _ { s } ) \leq \sum _ { i = 1 } ^ { k } \lambda _ { i } ,
$$

with equality if and only if the range of P is a k-dimensional subspace of the top eigenspace of $T _ { s }$ that contains every eigenvector whose eigenvalue exceeds $\lambda _ { k }$ . Therefore, for $\varepsilon > 0$ , every minimizer of (5) has this property. If $\lambda _ { k } > \lambda _ { k + 1 }$ , the top-k eigenspace is the unique such subspace; the span is unique, and the encoder is unique up to right multiplication by $O ( k )$ . For $\varepsilon > 0$ , if $\mathrm { r a n k } ( T _ { s } ) < k _ {  }$ , the top-k sum is the sum of all positive eigenvalues; every minimizer contains the positive eigenspace and an arbitrary $\left( k - \mathrm { r a n k } T _ { s } \right)$ -dimensional subspace of ker $T _ { s }$ , and every such subspace occurs as a minimizer; no direction in the kernel is identified. At $\varepsilon = 0 , ( 5 )$ is identically 2k and no direction is identifiable.

## A.2 Pairing (Theorem 2)

Write the rows of X as $x _ { i } ^ { \top }$ and those of Y as $y _ { i } ^ { \top }$ ; the relation $y _ { i } = Q x _ { i } { \mathrm { ~ i s ~ } } \mathbf { Y } = \mathbf { X } Q ^ { \top }$

Exact case. If $r = { \mathrm { r a n k } } ( \mathbf { X } ) = k$ , the rows of X span $\mathbb { R } ^ { k }$ . A candidate rotation R satisfies $\mathbf { X } R ^ { \top } = \mathbf { Y } ;$ the exact solution therefore obeys $\mathbf { X } ( R - Q ) ^ { \top } = 0$ . Since the rows of X span $\mathbb { R } ^ { k } , R = Q$ . Thus the synchronized map is unique, and it is the orthogonal Procrustes solution.

If $r < k$ , choose an orthonormal basis of $\mathbb { R } ^ { k }$ whose first r vectors span the row space of X. In this basis, $\mathbf { X } R ^ { \top } = \mathbf { X } Q ^ { \top }$ holds if and only if $\boldsymbol { R } = \boldsymbol { Q } ( \boldsymbol { I _ { r } } \oplus \boldsymbol { S } )$ with $S \in O ( k - r )$ . Hence the solution set is $Q \left( I _ { r } \oplus O ( k - r ) \right)$ , of dimension $( k - r ) ( k - r - 1 ) / 2$ . At $r = k - 1 , O ( 1 ) = \{ \pm 1 \}$ ; the nontrivial element is a reflection on the one-dimensional complement.

Noise. Assume $\mathbf { Y } = \mathbf { X } Q ^ { \top } + \mathbf { E }$ , where the rows of E are independent, zero mean, with covariance $\sigma ^ { 2 } I _ { k }$ independent of X, with iid sub-Gaussian rows satisfying $\mathbb { E } [ x _ { i } x _ { i } ^ { \top } ] = I _ { k }$ and $m _ { e } \geq c _ { 0 } k$ for a suficiently large fixed constant $c _ { 0 } > 1$ . The unconstrained least-squares matrix satisfies

$$
{ \hat { Q } } ^ { \top } - { Q } ^ { \top } = ( \mathbf { X } ^ { \top } \mathbf { X } ) ^ { - 1 } \mathbf { X } ^ { \top } \mathbf { E } .
$$

Matrix concentration [29] gives $\| \mathbf { X } ^ { \top } \mathbf { X } / m _ { e } - I _ { k } \| = O _ { P } ( \sqrt { k / m _ { e } } )$ and $\| \mathbf { X } ^ { \top } \mathbf { E } \| _ { \mathrm { o p } } = O _ { P } ( \sigma { \sqrt { k m _ { e } } } )$ ; hence

$$
\| \hat { \boldsymbol { Q } } - \boldsymbol { Q } \| _ { \mathrm { o p } } = O _ { P } \bigg ( \sigma \sqrt { k / m _ { e } } \bigg ) .
$$

Let $\hat { R }$ be the orthogonal Procrustes estimate; equivalently, $\hat { R } ^ { \top }$ is the orthogonal polar factor of $\mathbf { X } ^ { \top } \mathbf { Y }$ . The unperturbed matrix $\mathbf { X } ^ { \top } \mathbf { X } Q ^ { \top }$ has polar factor $Q ^ { \top }$ . The standard polar-factor perturbation bound gives

$$
\| \hat { R } ^ { \top } - Q ^ { \top } \| _ { \mathrm { o p } } \lesssim \frac { \| \mathbf { X } ^ { \top } \mathbf { E } \| _ { \mathrm { o p } } } { \sigma _ { \mathrm { m i n } } ( \mathbf { X } ^ { \top } \mathbf { X } ) } = O _ { P } ( \sigma \sqrt { k / m _ { e } } ) ,
$$

because $\sigma _ { \mathrm { m i n } } ( \mathbf { X } ^ { \top } \mathbf { X } ) = \Omega ( m _ { e } )$ when $m _ { e } \ge c _ { 0 } k$ and $c _ { 0 }$ is suficiently large; hence $\| \hat { R } - Q \| _ { \mathrm { o p } } = O _ { P } ( \sigma \sqrt { k / m _ { e } } )$ Setting this bound equal to τ gives the suficient sample cost $m _ { e } \gtrsim k \bar { \sigma } ^ { 2 } / \tau ^ { 2 } ;$ no lower bound is needed here.

Pairing graph. Fix a spanning tree of one connected component. Choose a frame at one vertex; each edge determines the relative frame by the exact-case argument; composing along the tree determines a frame at every vertex. Only one common $O ( k )$ remains per component. With c components the residual group is $O ( k ) ^ { c }$ , so paired data fix relative frames but never a global one.

## A.3 The surviving bound and its limits (Proposition 2)

For centered, whitened $f _ { 1 } , f _ { 2 }$ , and any $Q \in O ( k )$ 2

$$
\mathbb { E } \| f _ { 1 } - Q f _ { 2 } \| ^ { 2 } = 2 k - 2 \operatorname { t r } ( Q \Sigma ^ { \top } ) , \qquad \Sigma = \mathbb { E } [ f _ { 1 } f _ { 2 } ^ { \top } ] .
$$

Given $Z ,$ the two views are independent, so $\Sigma = \mathbb { E } [ u ( Z ) w ( Z ) ^ { \top } ]$ ] with $u = \mathbb { E } [ f _ { 1 } ( X _ { 1 } ) \mid Z ]$ and $w = \mathbb { E } [ f _ { 2 } ( X _ { 2 } ) \mid Z ]$ Let $C _ { 1 } , C _ { 2 }$ be the covariances of $u , w ,$ and set $\rho _ { m } = \mathrm { t r } ( C _ { m } ^ { 1 / 2 } ) / k$ . Since $f _ { m }$ is whitened, $C _ { m } \preceq I _ { k }$ . Factor $u = C _ { 1 } ^ { 1 / 2 } \tilde { u }$ and $w = C _ { 2 } ^ { 1 / 2 } \tilde { w }$ on the ranges of $C _ { m }$ (the orthogonal complements are annihilated by $C _ { m } )$ , so $\Sigma = \bar { C _ { 1 } ^ { 1 / 2 } } R C _ { 2 } ^ { 1 / 2 }$ with $\| R \| _ { \mathrm { o p } } \leq 1$ by Cauchy–Schwarz. Schatten H¨older gives

$$
\| \Sigma \| _ { * } \leq \| C _ { 1 } ^ { 1 / 2 } \| _ { * } \| R \| _ { \mathrm { o p } } \| C _ { 2 } ^ { 1 / 2 } \| _ { \mathrm { o p } } \leq \mathrm { t r } ( C _ { 1 } ^ { 1 / 2 } ) = k \rho _ { 1 } ,
$$

and exchanging the roles gives $\| \Sigma \| _ { * } \le k \rho _ { 2 }$ . Von Neumann’s trace inequality gives min<sub>Q</sub> $\mathbb { E } \| f _ { 1 } - Q f _ { 2 } \| ^ { 2 } =$ $2 k - 2 \| \Sigma \| _ { * } \geq 2 k ( 1 - \operatorname* { m i n } \{ \rho _ { 1 } , \rho _ { 2 } \} )$ . This is the bound (2).

At $\rho _ { 1 } = \rho _ { 2 } = 1$ the bound (2) is zero. The construction of Proposition 1 supplies state functions $f , g$ with $\rho ( f ) = \rho ( g ) = 1$ and $\mathbb { E } \| f - Q g \| ^ { 2 } = 2 k$ for every Q; hence the bound cannot certify good-view alignment.

## A.4 Noise symmetries (Proposition 3)

Proof of Proposition 3. If $Q \in \operatorname { S t a b } ( p _ { Z } ) \cap \operatorname { S t a b } ( \eta ) \cap G _ { \theta }$ , then $A _ { m } Q ( Q ^ { \top } Z ) + N _ { m } = A _ { m } Z + N _ { m }$ and the conditional law of $N _ { m }$ given $Q ^ { \top } Z$ agrees with the conditional law given $Z ;$ hence the joint law of the $X _ { m }$ is unchanged. Conversely, let $Q \in O ( k )$ be admissible $( A _ { m } Q \in { \mathcal { C } } _ { m }$ for all m) and leave the joint law of the observations unchanged. Apply the identifiability hypothesis to the parameter pairs $\left( A _ { m } Q , \eta _ { m } \right)$ and $\left( A _ { m } , \eta _ { m } \right)$ : there is a common orthogona $R \in { \mathrm { S t a b } } ( p _ { Z } )$ with $A _ { m } Q = A _ { m } R$ and $\eta _ { m } ( \cdot \mid z ) = \eta _ { m } ( \cdot \mid R z )$ for all m and $p _ { Z } \mathrm { - a . e . ~ } z$ . Full column rank gives $Q = R ,$ so Q stabilizes $p _ { Z }$ and η; $Q \in G _ { \theta }$ by admissibility. Thus the residual set is exactly (4). Without full rank or identifiability, the intersection is only a suficient family.

For homoscedastic known noise, $\Sigma _ { m }$ is constant and invertible. Block-whitening each view is an invertible reparameterization; it left-multiplies the mixing matrices by an invertible matrix and whitens the noise, but leaves the set of admissible Q unchanged. Constant isotropic noise therefore leaves $O ( k )$ , and known constant covariance is neutral.

For the example $\Sigma ( z ) = \mathrm { d i a g } ( e ^ { z _ { 1 } } , e ^ { 2 z _ { 2 } } , e ^ { 3 z _ { 3 } } )$ and a non-degenerate $p _ { Z } .$ , the identity $\Sigma ( Q ^ { \top } z ) = \Sigma ( z )$ for $p _ { Z ^ { - } } \mathrm { a . e . } ~ z$ forces $( Q ^ { \top } z ) _ { 1 } = z _ { 1 } , ( Q ^ { \top } z ) _ { 2 } = z _ { 2 } ,$ , and $( Q ^ { \top } z ) _ { 3 } = z _ { 3 }$ by comparing the three positive diagonal entries (the exponents have distinct weights), so $Q ^ { \top } z = z$ for $p _ { Z } \mathrm { - a . e . } ~ z ;$ when $Q \neq I$ the set $\{ z : Q ^ { \top } z = z \}$ is a proper linear subspace, which cannot carry a non-degenerate $p _ { Z }$ , so $Q = I$ and $H _ { \Sigma } = \{ I \}$ ; under the identifiability hypothesis the residual set is therefore trivial.

The joint-covariance invariance check for view-heterogeneous but state-independent known covariance (L2) gives maximum deviation $1 . 1 \times 1 0 ^ { - 1 5 }$ on the synthetic worlds; it confirms that known constant covariance is neutral. The check is included in verify L4.py (first row of results/l4 exp1.csv); the z-dependent case is handled by the example above.

## A.5 Permutation gauge (Proposition 4)

The first claim is immediate: $B _ { m } = A _ { m } P _ { m } = ( A _ { m } P _ { 0 } ) ( P _ { 0 } ^ { - 1 } P _ { m } )$ for every $P _ { 0 } \in S _ { d }$ , so the observation is unchanged. For the second claim, if $B _ { m } = A _ { m } P _ { m } = A _ { m } ^ { \prime } P _ { m } ^ { \prime } ,$ , put $P _ { 0 } = P _ { m } ( P _ { m } ^ { \prime } ) ^ { - 1 } ;$ then $A _ { m } ^ { \prime } = A _ { m } P _ { 0 }$ and $P _ { m } ^ { \prime } = P _ { 0 } ^ { - 1 } P _ { m }$ . There are d! choices of $P _ { 0 }$ , and diferent choices give diferent pairs $( A _ { m } ^ { \prime } , P _ { m } ^ { \prime } )$ when $A _ { m }$ is full column rank.

When each $A _ { m }$ is square and invertible, with the frame fixed, paired data determine the composite $M _ { m l } = A _ { l } P _ { l } P _ { m } ^ { - 1 } A _ { m } ^ { - 1 }$ ; its decomposition retains the $S _ { d }$ ambiguity above, and $m _ { e } \to \infty$ does not remove it because every pair of decompositions is related by a global $P _ { 0 }$ . If $p _ { Z }$ is exchangeable, absolute labels remain an exact gauge asymptotically.

With r correspondence anchors, the anchored coordinates are fixed up to signs; the remaining $d - r$ coordinates may be permuted and every coordinate may be sign-flipped, so the residual is a signed-permutation subgroup (a semidirect product in general; at $d - r \leq 1$ it is a direct product). Value anchors give known coordinate values and remove the signs. For component semantics, in the noiseless case population ICA identifies independent non-Gaussian components up to permutation and sign [14, 15]; matching scalar shape fingerprints then identifies the relative permutation up to signs. This is a population statement; finite-sample error is not analyzed here.

## B Reproducibility

All numbers come from fixed-seed NumPy scripts, run on CPU. The scripts are verify review.py (review checks), verify K4.py (partial pairing), verify L4.py (noise and anchors), verify M4.py (permutation and fingerprints), and verify N1.py (noise stabilizer and orbit checks). Outputs are in the accompanying results/ directory (the development repository keeps them under theory/results/). Each script fixes its own world size and embedding dimension; see the CSV headers. The protocol uses hard whitening. The scripts report diferent quantities, named in the CSV headers: verify review.py and verify N1.py report Procrustes residuals mi $\boldsymbol { 1 } _ { Q \in O ( k ) } \mathbb { E } \| \boldsymbol { f } - Q g \| ^ { 2 }$ (bounded by 2k) and auxiliary spectra, verify K4.py reports the normalized frame error $\| \hat { R } - R _ { \mathrm { t r u e } } \| _ { F } / \sqrt { 2 d }$ with classification and ambiguity statistics for partial pairing, verify L4.py reports joint-covariance invariance deviations and residual gauge dimensions for known-noise and constrained models, and verify M4.py reports permutation-recovery accuracy, composite-frame error, and sign/permutation ambiguity counts. Seed ranges: K4 exp1 uses 0–7, K4 exp2 and exp4 use 0–4; L4 exp1 and exp2a use 0–5, exp2c uses 0–2; M4 exp1 uses 0–4, exp2 uses 0–3 (a --quick path uses fewer); verify review.py and verify N1.py use default rng(0). verify N1.py prints its checks and writes no CSV. The numbers quoted from verify review.py and verify N1.py are their printed values; the remaining quoted numbers are the corresponding CSV rows.

Code and data. The five scripts, the CSV outputs, and a README with the exact commands and seeds are included as ancillary files with this submission. The three cited files map as follows: k4 exp1.csv (exp,d,m,gauge dim,err,err se,ambiguity,rand base,note); l4 exp1.csv (exp,test,value,se,note; this run uses $d = 3 , k = 3 )$ ; and m4 exp2.csv (exp,mode,n,d,acc,acc se,ambiguity,note).

## References

[1] M. Huh, B. Cheung, T. Wang, and P. Isola. The Platonic Representation Hypothesis. ICML, 2024.

[2] R. Balestriero and Y. LeCun. Proving and Improving the Platonic Representation Hypothesis with JEPAs. arXiv:2610.39302v1, 29 October 2026.

[3] R. Balestriero and Y. LeCun. LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics. arXiv:2511.08544, 2025.

[4] A. Xu. Good Views Are Not Enough: Global Counterexamples for Gaussian-Regularized JEPAs. arXiv:2611.12137, 2026.

[5] S. Syrota, A. Lopez I Serrano, J. Franck, and S. Hauberg. Identifiable alignment of unpaired representations. arXiv:2611.11460, 2026.

[6] W. Nelson, M. Fumero, T. Karaletsos, and F. Locatello. Statistical and structural identifiability in representation learning. ICLR, 2026. arXiv:2603.11970.

[7] D. Shi, A. Han, X. Li, L. Thompson, and J. M. Hern´andez-Lobato. From Plato to Aristotle: Toward a General Theory of Representation Hypotheses. arXiv:2610.20286, 2026.

[8] C. Shang et al. Displacement Geometry Captures Platonic Shared Reality Across Models and Modalities. arXiv:2609.24209, 2026.

[9] You et al. What Converges in the Platonic Representation Hypothesis? Structure over Geometry. arXiv:2609.27252, 2026.

[10] A. Achara et al. Multi-Way Representation Alignment. arXiv:2602.06205, 2026.

[11] T. W. Anderson and H. Rubin. Statistical inference in factor analysis. Proc. Berkeley Symposium, 1956.

[12] K. G. J¨oreskog. A general approach to confirmatory maximum likelihood factor analysis. Psychometrika, 1969.

[13] H. Teicher. Identifiability of finite mixtures. Ann. Math. Statist., 1963.

[14] P. Comon. Independent component analysis, a new concept? Signal Processing, 1994.

[15] A. Hyv¨arinen, J. Karhunen, and E. Oja. Independent Component Analysis. Wiley, 2001.

[16] J. Z. HaoChen, C. Wei, A. Gaidon, and T. Ma. Provable guarantees for self-supervised deep learning with spectral contrastive loss. NeurIPS, 2021.

[17] R. Balestriero and Y. LeCun. Contrastive and non-contrastive self-supervised learning recover global and local spectral embedding methods. NeurIPS, 2022.

[18] J. R. Kettenring. Canonical analysis of several sets of variables. Biometrika, 1971.

[19] J. von K¨ugelgen, Y. Sharma, L. Gresele, W. Brendel, B. Sch¨olkopf, M. Besserve, and F. Locatello. Self-supervised learning with data augmentations provably isolates content from style. NeurIPS, 2021.

[20] I. Daunhawer, A. Bizeul, E. Palumbo, A. Marx, and J. E. Vogt. Identifiability results for multimodal contrastive learning. ICLR, 2023.

[21] R. A. Redner and H. F. Walker. Mixture densities, maximum likelihood and the EM algorithm. SIAM Review, 1984.

[22] J. B. Kruskal. Three-way arrays: rank and uniqueness of trilinear decompositions. Linear Algebra Appl., 1977.

[23] P. H. Sch¨onemann. A generalized solution of the orthogonal Procrustes problem. Psychometrika, 1966.

[24] S. Umeyama. An eigendecomposition approach to weighted graph matching problems. IEEE TPAMI, 1988.

[25] D. Pachauri, R. Kondor, and V. Singh. Solving the multi-way matching problem by permutation synchronization. NIPS, 2013.

[26] A. Singer. Angular synchronization by eigenvectors and semidefinite programming. Appl. Comput. Harmon. Anal., 2011.

[27] M. Cucuringu, A. Lipshitz, and A. Singer. Sensor network localization by eigenvector synchronization over the Euclidean group. ACM TOSN, 2012.

[28] V. M. Govindu. Lie-algebraic averaging for globally consistent motion estimation. CVPR, 2004.

[29] R. Vershynin. High-Dimensional Probability: An Introduction with Applications in Data Science. Cambridge University Press, 2018.