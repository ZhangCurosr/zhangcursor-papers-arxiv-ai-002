# The Geometry of Ignorance: Detecting Suppressed Knowledge in LLMs through Activation Fragmentation

Adar Peleg<sup>\*</sup>, Dvir Alsheich<sup>\*</sup>, Tomer Ashuach, Avi Mendelson

<sup>1</sup>Technion - Israel Institute of Technology

Equal contribution

Correspondence: {a.p, dviralsheich, tomerashuach}@campus.technion.ac.il, mendlson@technion.ac.il

## Abstract

Evaluating whether a large language model (LLM) genuinely knows a concept remains a fundamental problem, since behavioral metrics can be fooled by suppression: a model that refuses to answer may still harbor recoverable knowledge. The framework measures the geometric coherence of anon-vectors, the differences in hidden states between a concept-specific query and its anonymized paraphrase, and shows that genuine knowledge forms a tight, low-entropy cluster while ignorance forces geometric fragmentation. Across ten instruction-tuned LLMs from three model families, all fragmentation metrics reliably separate known from unknown concepts, and postunlearning, fragmentation geometry correlates with adversarial recovery across four attack types and five unlearning algorithms, though the direction of that relationship is shown to depend on attack methodology. Furthermore, a layer-resolved analysis reveals that different adversarial mechanisms exploit this geometry at varying depths, with high attack-specific correlations that are invisible to behavioral evaluation. Together, these results constitute a singlephase, attack-free proxy for unlearning robustness that captures aspects of parametric knowledge inaccessible to output-level metrics.

A reference implementation is available at

## 1 Introduction

The expansive pre-training of Large Language Models (LLMs) inevitably leads to the memorisation of vast amounts of factual associations (Petroni et al., 2019; Meng et al., 2023), raising critical legal, ethical, and safety concerns regarding privacy and the generation of harmful content (Carlini et al., 2021; Seyitoglu et al. ˘ , 2024). Traditionally, the boundaries of a model’s parametric knowledge have been evaluated through behavioral analysis, relying on surface-level metrics such as questionanswering accuracy or token perplexity (Lynch et al., 2024). Behavioral evaluations treat the model as a black box and fail to capture how factual associations are internally stored and retrieved during the forward pass (Geva et al., 2023). This gap is most consequential in the context of machine unlearning, which aims to selectively erase specific concepts without retraining (Li et al., 2024; Gandikota et al., 2025; Ashuach et al., 2026). Recent representationlevel analyses reveal that unlearning interventions often result in the mere suppression or bypassing of knowledge, rather than genuine deletion (Xu et al., 2025; Lee et al., 2024; Schoepf et al., 2026). Because suppression mechanisms leave the knowledge structurally distinguishable in the parameters (Huu-Tien et al., 2025), it can be reliably recovered by malicious actors through activation steering (Seyitoglu et al. ˘ , 2024). Consequently, robust evaluation must shift to fine-grained, intrinsic metrics that can detect suppressed but recoverable knowledge.

Recent mechanistic interpretability work has introduced intrinsic probes that bypass generation and measure knowledge directly within intermediate model layers (Gottesman and Geva, 2024; Hong et al.). However, these metrics remain entangled with prompt syntax, conflating factual knowledge with the linguistic and syntactic structure of the query. Moreover, no single-phase metric reliably predicts whether suppressed knowledge is adversarially recoverable.

To address the limitations of both behavioral evaluations and syntax-entangled intrinsic metrics, we build upon the concept of the anon-vector (Seyitoglu et al. ˘ , 2024), defined as the directional shift between the hidden state of a concept-specific query and its identity-stripped paraphrase. We use this to introduce activation fragmentation, a geometric framework that isolates the precise footprint of parametric knowledge using two forward passes per query (no token generation, no iterative optimization).

![](images/47b75c4f1bda903f96a1c588e37aa5965e7c53a51cb7e8548b7deb8703358700.jpg)  
Figure 1: Activation fragmentation isolates conceptual knowledge. Subtracting the mean hidden state of a cloud of anonymized variants cancels shared linguistic syntax. Plotting the resulting anon-difference vectors $( \Delta v _ { i } )$ from the origin reveals that parametric knowledge manifests as tight, low-entropy covariance clusters of aligned vectors, whereas ignorance forces geometric fragmentation across high-variance directions.

We hypothesize that unknown concepts lack parametric structure: their anon-difference vectors scatter into high-variance, fragmented directions. By contrast, genuine parametric knowledge manifests as a tight, cohesive geometric signature in activation space.

Contributions. Our main contributions are as follows:

• We propose activation fragmentation, a finegrained, representation-level metric that leverages anon-vectors to successfully isolate conceptual identity from syntax in intermediate model layers.

• We demonstrate empirically that fragmentation metrics distinguish known entities (e.g., Harry Potter) from completely unknown fictitious concepts (e.g., TOFU (Maini et al., 2024)) with robust statistical separability across ten instruction-tuned LLMs from three distinct model families.

• Activation Uniformity Scatter Score (AUSS). We establish that post-unlearning fragmentation serves as a highly accurate, attack-free predictor of elements of unlearning robustness. (e.g. AUSS dispersion metrics predict in-context learning (ICL) recoverability).

## 2 Background

Knowledge in LLM activations. LLMs encode factual associations in their parameters, with feedforward layers functioning as key-value memories that promote specific concepts into the residual stream (Geva et al., 2021, 2022). Intermediate layers carry geometrically richer knowledge representations than the final layer (Skean et al., 2025; Razzhigaev et al., 2023). Skean et al. (2025) measure representational compression layer-by-layer via the Rényi-2 Gram-matrix entropy: for any matrix $Z \in \mathbb { R } ^ { \hat { M } \times d }$

$$
\begin{array} { r } { K _ { Z } = \frac { Z Z ^ { \top } } { \operatorname { t r } \left( Z Z ^ { \top } \right) } } \\ { S _ { 2 } ( Z ) = - \log _ { 2 } \operatorname { t r } \left( K _ { Z } ^ { 2 } \right) } \end{array}\tag{1}
$$

Lower $S _ { 2 }$ indicates that the M vectors concentrate in a smaller subspace (higher representational compression), identifying where factual content is most geometrically concentrated in the hidden space. Applied to the raw query hidden states $\textbf { H } = \{ h _ { \ell ^ { * } } ( q _ { i } ) \}$ , this gives the baseline metric Reg $S _ { 2 }$ (Skean et al., 2025): $\mathrm { R e g } _ { S _ { 2 } } = S _ { 2 } ( { \bf H } )$

Anon-vectors and concept-differential representations. A concept-specific signal can be isolated by contrasting the hidden state of a real query with that of a semantically equivalent but identitystripped (anonymised) paraphrase. Seyitoglu et al.˘ (2024) exploit this to mount an activation steering attack: they compute a steering vector as the raw difference between hidden states of a concept query and its anonymised counterpart, then inject it into the residual stream to re-activate suppressed knowledge in an already-unlearned model. The ConceptVectors dataset (Hong et al.) likewise demonstrates that named entities encoded in model parameters leave verifiable parametric traces at intermediate layers, and that jailbreak attacks ride along these very directions to bypass unlearning (Hong et al.). Our anon-difference vector $\Delta v _ { i }$ (Eq. 2) builds directly on the anonymisation technique of Seyitoglu et al.˘ (2024), averaging over K variants to stabilise the concept-differential signal. Neither prior work, however, treats the population geometry of a batch of such vectors as a diagnostic metric; that is a contribution of this paper.

## 3 Method

## 3.1 Metric Definitions

Anon-vectors and Entropy. Let $\{ q _ { i } \} _ { i = 1 } ^ { N }$ be N queries about a target concept, and let $h _ { \ell } ( \cdot ) \in$ $\mathbb { R } ^ { d }$ be the mean-pooled hidden states at layer $\ell .$ For each $q _ { i }$ , we construct $K$ anonymised variants $\{ a _ { i } ^ { ( k ) } \} _ { k = 1 } ^ { K } .$ . The anon-difference vector at layer $\ell$ isolates conceptual identity by subtracting the mean anonymised state:

$$
\Delta v _ { i } ( \ell ) \ = \ h _ { \ell } ( q _ { i } ) \ - \ { \frac { 1 } { K } } { \sum _ { k = 1 } ^ { K } } h _ { \ell } \Big ( a _ { i } ^ { ( k ) } \Big )\tag{2}
$$

This requires two forward passes per query in performed experiments (in general K+1 per query, with $K { = } 1$ in experiment sections) and no token generation. All fragmentation metrics are computed from the matrix of anon-difference vectors at a chosen layer; how that layer is selected is stated in each experimental protocol (Section 4) and expended in (Appendix B.3). To measure representational rank concentration, we apply Grammatrix entropy to these unnormalised vectors, defining our primary dispersion metric: AnonDif $s _ { 2 } =$ $S _ { 2 } ( \Delta \mathbf { V } ( \boldsymbol { \ell } ) )$ . Applying $S _ { 2 }$ to concept-differential vectors rather than raw activations is a novel contribution of this work. We also evaluate two variants: $\mathrm { R e g } _ { S _ { 2 } } = S _ { 2 } ( { \bf H } ^ { \ell } )$ applies the same entropy to the raw hidden-state matrix (no anonymisation subtraction), serving as a baseline; and $\mathrm { N o r m A n o n D i f f } _ { S _ { 2 } } = S _ { 2 } ( \hat { \Delta } \mathbf { V } ( \ell ) )$ applies it to the L -normalised anon-difference vectors.

AUSS Batch-mean metrics (Ours). To assess population-level geometric consistency, we partition the N pairs into $B = 1 0$ disjoint batches $( \mathsf { A p } \cdot$ pendix B.3). For each batch b, we compute the unitnormalised mean anon-difference direction $\hat { d } _ { b }$ . We then measure geometric consistency across these B representative directions using two primary novel matrices that aim to measurefragmentation:

$\mathbf { A U S S _ { L _ { 2 } } }$ : Mean pairwise squared $L _ { 2 }$ distance between batch-mean directions.

$\mathbf { A U S S _ { \mathrm { c o s } ^ { 2 } } } \mathrm { : }$ Complement of mean pairwise squared cosine between batch-mean directions.

$\mathbf { A U S S _ { J a c } } \mathrm { . }$ Jaccard distance between the sets of the $k { = } 3 2$ largest-magnitude coordinates of each pair of batch-mean directions, the fraction of top-activated dimensions that differ between batches.

Note that all three metrics are bounded: $\mathrm { A U S S } _ { L _ { 2 } } \in$ [0, 2] and $\mathrm { A U S S } _ { \mathrm { c o s ^ { 2 } } } , \mathrm { A U S S } _ { J a c } \ \in \ [ 0 , 1 ]$ . We expect that when a concept is known, batch-mean directions cluster tightly, driving all three metrics toward zero; when unknown, they are expected to scatter, driving them toward the upper bound. This bounded range means that Cohen’s |d| can appear large even with modest raw gaps, because the pooled within-group standard deviation is bounded from above. A comprehensive list of additionally evaluated metric variations including base pairwise similarities, Jaccard incoherence, and covariance traces is detailed in Appendix C.

## 3.2 Entropy, Rank, and the Geometry of Parametric Knowledge

The central hypothesis of this work is that parametric knowledge leaves a geometric imprint. If a model knows a concept, diverse queries converge onto a shared subspace, producing a tight, lowentropy cluster of anon-difference vectors. Conversely, ignorance drives these vectors to scatter across the activation space, increasing entropy.

This connection between the Gram-matrix entropy $S _ { 2 }$ and geometric dispersion is formalised by the following result (for proofs see Appendix A):

Theorem 1 (High entropy forces geometric fragmentation). Let $A \ \in \ \mathbb { R } ^ { n \times d }$ be a matrix of $L _ { 2 ^ { - } }$ normalised activation vectors and let $G = A A ^ { \top }$ be the corresponding Gram $m a t r i x ^ { 1 }$ with effective rank EfRank $( G ) = \operatorname { T r } ( G ) / \lambda _ { \operatorname* { m a x } } ( G )$ . Then the mean pairwise squared Euclidean distance satisfies

$$
D _ { L _ { 2 } } ^ { 2 } \geq 2 - \frac { 2 } { \mathrm { E f f R a n k } ( G ) } .
$$

Since $2 ^ { S _ { 2 } / 2 } \leq \mathrm { E f f R a n k } ( G ) \leq 2 ^ { S _ { 2 } }$ (Proposition 1, Appendix $\mathbf { A } )$ , high $S _ { 2 }$ forces the effective rank upward, which structurally guarantees spatial dispersion. The complementary direction holds under a natural assumption: when all pairwise cosine similarities are non-negative (i.e., vectors share a common half-space). For anon-vectors this is geometrically motivated: if a model parametrically encodes a concept, every query about that concept shifts the hidden state toward the same concept direction, so the differences $\Delta v _ { i }$ collectively point away from the anonymised baseline in a consistent orientation. Under this assumption, a matching upper bound applies:

Theorem 2 (Low entropy forces geometric clustering). Under the same setup as Theorem 1, if additionally $G _ { i j } \geq 0 .$ for all $i \neq j$ , then

$$
D _ { L _ { 2 } } ^ { 2 } \ \leq \ \frac { 2 n } { n - 1 } \Big ( 1 - 2 ^ { - S _ { 2 } ( G ) } \Big ) .
$$

In particular, $D _ { L _ { 2 } } ^ { 2 } \to 0 a s S _ { 2 } ( G ) \to 0 .$

Together, these bounds demonstrate that subspace-compression metrics directly reflect the presence or absence of a coherent concept representation, effectively isolating knowledge from mere behavioural suppression.

## 3.3 Semantic Testing

Anon-vectors over raw activations. Raw activations encode both conceptual content and prompt syntax. Subtracting the anonymised baseline (denoted ${ \bf a s } \Delta v _ { i } )$ cancels shared syntactic components, isolating the conceptual signal. This produces a substantially stronger discriminative signal than working with raw activations directly (Table 1).

Semantic vs. generation artifacts. Behavioral evaluations measure accessible outputs, not latent storage. Since representational suppression mechanisms can block factual recall without deleting the underlying knowledge (Lee et al., 2024; Xu et al., 2025), evaluating hidden-state geometry provides a strictly more rigorous proxy for true parametric erasure. A model that refuses to answer, whether through RLHF-driven refusal or explicit unlearning, may still harbor recoverable knowledge in its hidden states; generation-based metrics will report success while the knowledge persists, making them insufficient as a sole evaluation criteria.

The measurement problem: no true oracle. The ideal ground truth for a knowledge-detection metric would be a model trained from scratch on a corpus that provably excludes the target concept, making its ignorance logically certain. In practice, this is prohibitively expensive and inaccessible: controlled pre-training runs require substantial compute, carefully curated multi-billion-token datasets, and institutional resources. There is, therefore, no true oracle for parametric knowledge. We cannot directly observe what a model knows, only what it outputs, and output-level proxies risk circularity when used to calibrate the very metrics designed to go beyond them.

Bogus data as a proxy for not knowing. We sidestep this measurement problem by grounding knowledge in dataset construction rather than model training. As the negative class, we draw from data. For example, the TOFU benchmark (Maini et al., 2024) consists entirely of fictitious authors invented for the benchmark and absent from any pre-training corpus. Because these entities were never written about anywhere, no model can have parametric knowledge of them, and there is no plausible mechanism by which such knowledge could exist. As the positive class, we can draw from Harry Potter (HP) QA pairs; HP knowledge is confirmed to persist in the parameters of most LLMs as a consequence of large-scale pre-training (Carlini et al., 2021; Seyitoglu et al. ˘ , 2024). A metric that separates HP from TOFU under this protocol has therefore demonstrated genuine knowledge detection, not behavioral correlation. The specific datasets and model counts used in our evaluation are described in Section 4.

## 4 Experiments

1. RQ1: How can we measure the binary existence of knowledge in LLMs? Can we suggest a metric that maximizes separation between known and unknown data?

2. RQ2: How can we measure fine-grained knowledge extraction ability from LLMs? Which metric has the highest correlation with Attack success rate (ASR)?

## 4.1 RQ1: Fragmentation as a Proxy for Parametric Knowledge

Setup. We evaluate all nine fragmentation metrics under the protocol presented in (Section 3.3) across ten instruction-tuned LLMs from three families: Qwen (×4), Gemma (×2), and LLaMA (×4), ranging from 7B to 14B parameters (Appendix B), using batch size $B { = } 1 0$ (full sensitivity sweep over $B \in \{ 1 , 2 , 5 , 1 0 , 2 0 , 5 0 \}$ justifying this choice in Appendix B.2). The known-concept class consists of 96 QA pairs drawn from the Harry Potter novels; the unknown-concept class consists of 100 QA pairs from the TOFU benchmark (Maini et al., 2024). All ten models score 0% QA accuracy on TOFU, casually confirming that no knowledge of these fictitious entities exists. For each model, fragmentation is computed at the per-metric layer that maximizes HP–TOFU separation. Appendix B.3 also reports a fixed, deployable heuristic, the layer at $0 . 8 \times$ relative model depth, requiring no permodel search, that matches the oracle sweep’s discriminability.

Results. All nine metrics correctly separate HP from TOFU across all 10 models and survive BH (Benjamini-Hochberg false discovery rate control at $\alpha ~ = ~ 0 . 0 5 )$ correction (Table 1; per-model raw values in Appendix Tables 5-6). $\mathrm { \bf A U S S _ { \mathrm { c o s ^ { 2 } } } }$ achieves the largest separation $( | d | = 2 1 . 5 1 , p _ { \mathrm { a d j } } = 0 . 0 0 2 )$ and $\mathsf { A U S S } _ { \mathrm { L } 2 }$ is second $( | d | { = } 1 8 . 9 1 )$ , both dispersion higher for TOFU than HP in 10/10 models; the scale-invariant classifier check below (AUC= 1.0) confirms this separation holds independent of each metric’s raw scale.

Entropy-based dispersion metrics (NormAnon-Diff $S _ { 2 } ,$ , AnonDiff $S _ { 2 }$ , Reg $S _ { 2 } ;$ Section 3.1) likewise reach 10/10 with large effect sizes. Coherence metrics, i.e. centroid norm $\| \bar { \Delta v } \|$ , Reg cos, Anon-Diff cos, and AnonDiff ${ \bar { L } } _ { 2 }$ , all show strictly higher values for HP (8/10 or 10/10 models), confirming that a known concept induces a tighter, more directionally consistent hidden-state geometry than a purely fictional one. Effect sizes reflect per-metric peak-layer selection; performance under a labelfree layer protocol would be lower (Appendix B.3).

The separation generalizes beyond a single narrow contrast and is not a scale artifact. To rule out that this separation reflects one narrow contrast, we extend the comparison to Harry Potter (retain split) and Star Wars against the same TOFU baseline; all 24 BH-corrected comparisons across the 8 AUSS metrics are significant $( q ~ \in ~ [ 0 . 0 0 6 3 , 0 . 0 2 6 0 ] )$ , with HP-retain effect sizes reaching $d { = } 6 . 2 0$ , larger than the original HP-forget contrast (Appendix Table 7). A scaleinvariant leave-one-model-out classifier trained on the standardized 8-metric peak-layer feature vector, evaluated on the original HP-forget vs. TOFU pair, confirms the known/unknown separation is not an artifact of raw effect-size scale: Logistic Regression reaches 100% accuracy $( \mathrm { { A U C } = 1 . 0 }$ 9/9 models fully correct) on the fresh, directioncorrected extraction (Appendix Table 8).

Table 1: RQ1: Metric discriminability (HP known vs. TOFU unknown) across 10 LLMs, evaluated at the permetric layer maximising HP–TOFU separation, sorted by Cohen’s |d|. Gap: mean |HP−TOFU| with 95% bootstrap CI. $p _ { \mathrm { a d j } } \colon$ Wilcoxon signed-rank, BH-corrected $( n { = } 1 0 ) ; { } ^ { * * } p { < } 0 . { \dot { 0 } } 1 . { } ~ { \stackrel { \star } { \cdot } } ;$ to concept-differential vectors. $^ { \dagger } \colon$ raw-activation baseline, no anon-vector subtraction. Per-model values in Appendix (Tables 5–6).
<table><tr><td rowspan=1 colspan=1>Metric</td><td rowspan=1 colspan=2>|δ|       Gap [95% CI]        $\underline { { p } } \mathrm { a d j }$ Dispersion (↓ HP expected;10/10 models)</td></tr><tr><td rowspan=2 colspan=1> $\mathrm { \overline { { A U S S _ { c o s ^ { 2 } } } } }$  $\mathrm { \ A U S S { } } _ { \mathrm { L } 2 }$ NormAnonDiff S2AnonDiff S $\mathsf { R e g } S _ { 2 } ^ { \dagger }$ </td><td rowspan=2 colspan=1>21.51 0.686 [0.667, 0.70518.91 1.203[1.166, 1.239]4.21  1.206[1.022, 1.332]3.50  0.804 [0.671, 0.941]0.92  0.117 [0.054, 0.200]</td><td rowspan=1 colspan=1> $0 . 0 0 2 ^ { \ast \ast }$  $0 . 0 0 2 ^ { * * }$  $0 . 0 0 2 ^ { * * }$  $0 . 0 0 2 ^ { * * }$ </td></tr><tr><td rowspan=1 colspan=1> $0 . 0 0 2 ^ { * * }$ </td></tr><tr><td rowspan=1 colspan=1>Coherence (↑ HP exp</td><td rowspan=1 colspan=1>ected)</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=2 colspan=1>∥|∆û||AnonDiff cos $\mathrm { R e g } \overline { { \mathrm { c o s } } } ^ { \dagger }$ AnonDiff ${ \bar { L } } _ { 2 }$ </td><td rowspan=2 colspan=1>1.69  0.182[0.118, 0.244]1.29  0.128[0.069, 0.185]1.25  0.044 [0.023, 0.065]1.21  2.634 [1.412, 3.946]</td><td rowspan=1 colspan=1> $\overline { { 0 . 0 0 2 } } ^ { * * }$  $0 . 0 1 0 ^ { * * }$ </td></tr><tr><td rowspan=1 colspan=1> $0 . 0 0 2 ^ { * * }$  $0 . 0 0 2 ^ { * * }$ </td></tr></table>

## 4.2 RQ2: Fragmentation as a Predictor of Unlearning Robustness

Setup. RQ1 established that fragmentation geometry reflects whether knowledge exists. RQ2 asks whether it also reflects how recoverable that knowledge remains after unlearning. Fragmentation metrics for each model–concept–method triple are computed at the layer that minimises $S _ { 2 }$ over the post-unlearning forget-set queries (label-free protocol; Appendix B.3). The layer-resolved sweep evaluates every transformer layer independently (Appendix D.5).

Models and unlearning methods. We apply five unlearning methods spanning output-level suppression to representation-level distortion: GA (Yao et al., 2024) negates the forget-set CE loss; DPO (Rafailov et al., 2024) treats forget pairs as rejected and retain pairs as chosen; NPO (Zhang et al., 2024) applies a negative preference loss against a frozen reference; NPO+KL (Zhang et al.,

2024) augments NPO with KL retain regularisation; and RMU (Huu-Tien et al., 2025) steers forget-set hidden states at a fixed intermediate layer toward a random target direction. GA and DPO act primarily on outputs, whereas RMU explicitly distorts internal representations. This distinction is reliably captured by fragmentation metrics even when the two approaches yield indistinguishable ROUGE-L scores (Appendix B.5). These methods are applied to three instruction-tuned models: LLaMA-2-7Bchat, Meta-LLaMA-3-8B-Instruct, and Qwen2.5- 7B-Instruct, each also evaluated as an unmodified Base control, across three target concepts: Harry Potter (Eldan and Russinovich), Star Wars, and William Shakespeare (Hong et al.) $( 3 \times 6 \times 3 { = } 5 4$ conditions; incomplete runs shown as $\underline { { ^ { 6 6 } } } - \underline { { ^ { , 9 } } } )$ . A run is verified $( \surd )$ when Forget $\mathrm { R O U G E - L < 0 . 1 0 }$ and Retain $\mathrm { R O U G E - L } \ge 0 . 8 0 \times$ baseline; surface suppression alone does not guarantee robustness. Four attacks then probe recoverability: Activation Steering (Seyitoglu et al.˘ , 2024), ICL (Brown et al., 2020), GCG (Zou et al., 2023), and MIA (Fu et al.) at geometric, contextual, gradient, and distributional levels, respectively; lower values indicate stronger unlearning (Appendix B.6). Correlation analysis excluding Base and WHP (Who’s Harry Potter; Eldan and Russinovich) runs $\scriptstyle ( n = 4 4 - 4 5$ non-base runs with complete attack data).

Results. Figure 2 summarizes attack success by method across all three models; full per-run values are in Appendix D.7 (Tables 13–15). Table 2 shows every BH-significant geometry predictor per attack $\scriptstyle ( n = 4 4 - 4 5$ non-Base, non-WHP runs); full Spearman table in Appendix Table 9.

AUSS fragmentation geometry predicts ICL recoverability; entropy-family metrics predict GCG and Steering. ICL is the most significantly predictable attack via the AUSS family: $\mathsf { A U S S } _ { \mathrm { L 2 } }$ $( \rho { = } \mathrm { - } 0 . 5 1 ^ { * * } )$ , centroid norm $( \rho { = } \mathrm { - } 0 . 5 1 ^ { \ast \ast } )$ , and $\mathrm { A U S S } _ { \mathrm { c o s ^ { 2 } } } ( \rho { = } { - } 0 . 5 0 ^ { * * } )$ all survive BH correction; methods that geometrically fragment hidden-state representations more thoroughly are harder to reach via in-context priming. GCG and Steering are predictable in aggregate via the Gram-entropy $\left( \mathrm { R e g } \right)$ and AnonDiff families rather than AUSS: Reg cos $( \rho { = } \mathrm { - } 0 . 5 1 ^ { * * } )$ and Reg S<sub>2</sub> $( \rho { = } \mathrm { - } 1 0 . 4 5 ^ { * * } )$ predict GCG, and AnonDiff $S _ { 2 } ~ ( \rho { = } \mathrm { + } 0 . 4 8 ^ { * * } )$ predicts Steering. MIA is omitted: no metric survives BH correction (Appendix Table 9). Leaveone-run-out validation across verified unlearning runs $\scriptstyle ( n = 2 6 - 2 7 )$ confirms ICL is the only attack that is both leave-one-out stable (8/8 metrics) and BH-significant (7/8); Steering (8/8) and GCG (7/8) are largely stable but not significant, while MIA (2/8) is neither reliably stable nor significant (Appendix D.6).

Table 2: RQ2: BH-significant geometry predictors per attack (Spearman $\rho , n { = } 4 4 { - } 4 5$ non-Base non-WHP runs with complete attack data, BH-corrected within each attack’s 9-metric column). All metrics shown survive BH at $q { < } 0 . 0 5 ;$ full 9-metric table (all attacks, including non-significant cells) in Appendix Table 9.
<table><tr><td rowspan=1 colspan=1>Metric</td><td rowspan=1 colspan=1>ICL</td><td rowspan=1 colspan=1>GCG</td><td rowspan=1 colspan=1>Steering</td></tr><tr><td rowspan=1 colspan=1> $\overline { { \mathrm { A U S S _ { L 2 } } } }$  $\mathrm { A U S S _ { c o s ^ { 2 } } }$  $\| \bar { \Delta v } \|$  $\operatorname { R e g } { \overline { { \cos } } }$  $\mathrm { R e g } \ S _ { 2 }$  $\mathrm { { A n o n D i f f } \ } S _ { 2 }$ </td><td rowspan=1 colspan=1> $- 0 . 5 1 ^ { \ast \ast }$  $- 0 . 5 0 ^ { * * }$  $+ 0 . 5 1 ^ { * * \dagger }$ 一一一</td><td rowspan=1 colspan=1>一一1 $- 0 . 5 1 ^ { * * }$  $+ 0 . 4 5 ^ { * * }$ 一</td><td rowspan=1 colspan=1>一一 $+ 0 . 4 8 ^ { * * }$ </td></tr></table>

$^ { \dag } \| \Delta \bar { v } \|$ is the exact algebraic dual of $\mathsf { A U S S } _ { \mathrm { L } 2 }$ (identical $| \rho |$ , opposite sign by construction, since $\mathrm { A U S S _ { L 2 } } { = } 2 { - } 2 \bar { c }$ for mean pairwise cosine c¯ and $\| \bar { \Delta v } \|$ is a monotonic function of the same c¯), not an independent confirmation.

Layer-resolved geometry concentrates late in the network for the two attacks it reliably predicts. The aggregate correlations in Table 2 reflect peak-layer values averaged over all metriclayer combinations. When we instead sweep Spearman ρ across every transformer layer for both AUSS and entropy metrics (Figure 3; full results in Appendix D.5), only GCG and ICL survive global BH correction, and both peak in the same latenetwork region:

$\mathbf { G C G } \colon \mathrm { A U S S _ { J a c } }$ at L29 reaches $\rho { = } { - } 0 . 8 5 3$ $\textstyle ( p = 0 . 0 0 0 8 , n = 1 1$ runs with complete layerresolved data, smaller than the $n { = } 4 4 { - } 4 5$ aggregate pool), with NormAnonDiff $S _ { 2 }$ close behind at L30 $( \rho { = } { - } 0 . 8 3 2 , p { = } 0 . 0 0 1 5 )$ ; the gradient-based suffix attack’s signal is shaped by whichever layer the model’s final next-token decision is most sensitive to.

ICL: Batch\_Cov\_TopEig at L28 reaches $\rho { = } { - } 0 . 8 8 6 ~ ( p { < } 0 . 0 0 0 1 , ~ n { = } 1 5 )$ , one layer from GCG’s own peak; in-context priming must similarly override the model’s late-stage, generationready representation.

MIA and Steering show only weak, non-BH-significant layer-specific signal in the current dataset (best entropy-family points at L31– L32, $\rho { \approx } 0 . 5 5 { \mathrm { - } } 0 . 6 5$ uncorrected $p { < } 0 . 0 5$ but $q _ { \mathrm { g l o b a l } } { \ge } 0 . 2 2 )$ ; we report these descriptively rather than as reliable predictors (Appendix Table 11).

![](images/e5363877e42af9161b266ca492c46b75802c43e254b13a356db34a11ca59a97f.jpg)

![](images/5eecf6d9671caca97e11c73db2c9dea9ee3a1007b716e13f163ee6e25343b12f.jpg)

![](images/1cae2393e2e55effe988ca774634508afff23b7c6829d1091ba3e909745ed714.jpg)

![](images/b5c51fe030ffb7e09677d09337e26fd42ae92d96cc912e9725c4c3132b47696a.jpg)  
Figure 2: RQ2: Mean attack success per unlearning method across four attack types (2×2 grid), averaged over three concepts (error bars = std). Bars are grouped by unlearning method; colour indicates model (LLaMA-2- 7B: blue, LLaMA-3-8B: green, Qwen2.5-7B: orange-red). Dashed lines mark chance level (0.5 for AUC-based attacks; 0 for ICL ∆). GA is the only method that consistently drives Steer-AUC near chance across all three models. LLaMA-3 (green) reaches GCG Acc = 0 under every method, revealing strong model-family variation in gradient-attack vulnerability. ICL ∆ is near zero across all methods and models; the geometric correlation with ICL recoverability emerges from the layer-resolved analysis (Table 11), not from this aggregate view. Full per-run values in Appendix D (Tables 13–15).

The highest-performing layer is highly predictable across models, consistently residing within the 0.80–0.88 range of total model depth. A single fixed relative depth of 0.8× model depth, with no per-model information at all, already gives 8/8 metrics correct-direction and significant at n=9; leave-one-model-out transfer of this depth achieves 72/72 (model, metric) pairs correct-direction, identical to an oracle per-model search (Appendix B.3).

## 4.3 Benchmarking Against Prior Intrinsic Probes

Trivial raw-similarity baselines. Table 9 already includes head-to-head comparisons against trivial intrinsic baselines: the Reg $S _ { 2 }$ (adapted from Skean et al. 2025) and Reg cos rows measure raw, non-anonymised hidden states with no anon-vector subtraction.

Head-to-head against published method. We benchmark AUSS directly against the representational metrics of Xu et al. (2025) (PCA-similarity, PCA-shift, CKA), which, unlike our method, require both pre- and post-unlearning model checkpoints. Across all 54 RQ2 configurations, BH correction over the full 44-test pool shows a mixed, attack-dependent picture: AUSS’s best metric outperforms Xu et al.’s best metric on ICL (ρ=−0.572, q=0.0001 vs. ρ=+0.381, q=0.0212) and Steering (ρ=−0.452, q=0.0042 vs. ρ=+0.428, q=0.0076), while Xu et al.’s CKA outperforms AUSS’s best metric on GCG (ρ=+0.374, q=0.0212 vs. ρ=−0.315, q=0.068, not BH-significant) and MIA (neither survives correction: ρ=+0.336 vs. $\rho { = } { + } 0 . 0 8 5 )$ . 11 of 44 tests survive BH correction; 8 of those 11 are AUSS metrics (full 44-test table in Appendix Table 10).

## 5 Discussion

The geometry of parametric knowledge. RQ1 establishes that activation fragmentation reliably distinguishes known from unknown concepts, and the geometry explains why. The anon-vector construction subtracts the mean hidden state of identitystripped paraphrases, canceling the shared linguistic and syntactic signal and leaving a residual that reflects only the concept-specific content encoded at that layer. When a model parametrically knows a concept, its internal representations across distinct queries about that concept agree on a direction the residual vectors form a tight, low-entropy cluster. When the concept is absent from the parameters, no such agreement exists: the residual after syntax cancellation is noise, and the vectors scatter into high-variance fragmented directions. This geometric signature is what our metrics quantify, and it is entirely invisible to behavioral evaluation.

![](images/c67d72a490d6bf1ed784fd5b1c73cf40901d29cf7d4a4f9e0610fa82852e1e7f.jpg)  
Figure 3: Spearman $\rho$ by layer, GCG attack. Each line is one AUSS fragmentation metric; the peak at L29 $( \mathrm { A U S S _ { J a c } } , \rho \mathrm { = } \mathrm { - } 0 . 8 5 3 )$ is marked. The strongest single predictor overall is NormAnonDiff $S _ { 2 }$ at L30 $( \rho { = } { - } 0 . 8 3 2$ , entropy family; Appendix D.5). GCG recoverability is governed by late-layer geometry, close to where ICL’s own peak falls (Appendix D.5). (Full four-attack sweep in Appendix D.5.)

Architecture-agnostic universality. The coherence signature described above is not an artifact of any particular architecture. The 10/10 correctdirection result for the strongest metrics $( \mathrm { A U S S _ { c o s ^ { 2 } } }$ $\mathrm { A U S S } _ { L _ { 2 } }$ , AnonDiff $S _ { 2 } )$ in RQ1 spans three model families (Qwen, Gemma, and LLaMA) across different architectures, training recipes, and parameter counts from 2B to 14B. This cross-architecture consistency suggests that the geometric signature of parametric knowledge is not idiosyncratic to any particular model design. Fragmentation metrics qualify as model-family-neutral auditing tools, requiring no architecture-specific calibration.

Suppression versus deletion. A consistent RQ2 pattern is that reducing Forget ROUGE-L does not reliably reduce fragmentation: GA applied to LLaMA-3-8B reaches Forget ROUGE-L = 0.000 yet leaves $\mathrm { A U S S } _ { L _ { 2 } }$ and $\| \bar { \Delta v } \|$ nearly unchanged from the Base model. GA and DPO act primarily on the output distribution, leaving internal concept geometry largely intact, whereas RMU, which steers representations directly, shifts fragmentation more noticeably but does not erase it. This matches the redirection-rather-than-deletion picture reported by Xu et al. (2025) and Schoepf et al. (2026): representation-level interventions come closest to genuine deletion, and fragmentation metrics measure how far even they fall short.

Layer-resolved geometry concentrates late in the residual stream for the attacks it reliably predicts. Of the four attacks, GCG and ICL are the two whose layer-specific signal survives BH correction at the global scope (Appendix D.5), and both concentrate their strongest signal in the same late-network region: GCG peaks at L29 $( \mathrm { \ A U S S _ { J a c } } ,$ $\rho { = } { - } 0 . 8 5 )$ and ICL at L28 (Batch\_Cov\_TopEig, $\rho { = } { - } 0 . 8 9 )$ , one layer apart. This is mechanistically coherent rather than coincidental. GCG’s adversarial suffix is optimised end-to-end by backpropagating through the entire network, so its gradient signal is shaped by whichever layer the model’s final next-token decision is most sensitive to. ICL must similarly override the model’s late-stage, generation-ready representation to redirect its output. Both attacks are therefore best diagnosed by measuring fragmentation deep in the residual stream rather than at an early or intermediate depth. MIA and Steering show only weak, non-BH-significant layer-specific signal in the current dataset (best entropy-family points at L31–L32, $\rho { \approx } 0 . 5 5 { \mathrm { - } } 0 . 6 5$ , uncorrected $p { < } 0 . 0 5$ but not surviving global correction); we treat these descriptively rather than as reliable single-layer predictors. Appendix B.3 reports a practical, labelfree layer-selection rule that performs close to this attack-specific oracle search without per-attack calibration.

Attack-dependent sign phenomena. The correlation’s sign is not uniform across attacks (Table 9): $\mathsf { A U S S } _ { L _ { 2 } }$ predicts ICL negatively $( \rho { = } { - } 0 . 5 1 )$ , while Reg $S _ { 2 }$ and AnonDiff $S _ { 2 }$ predict GCG and Steering positively $( \rho { = } \mathrm { + 0 . 4 5 , \rho { = } \mathrm { + 0 . 4 8 ) } }$ . Based on the different mechanisms by which these attacks recover knowledge, we expect their correlation with fragmentation to differ in signal. ICL recovers "knowledge" through the model’s own generation, which plausibly requires a single coherent direction, empirically it seems that fragmentation tracks the absence of one. GCG and Steering instead work by searching or intervening on the representation directly, where possibly a more dispersed representation offers more directions to exploit, which would explain the opposite sign. We discuss in Limitations how this account could be further validated in future work.

Practical implications for unlearning evaluation. Activation fragmentation can serve as a computationally cheap, pre-attack screening step in unlearning evaluation pipelines. If fragmentation metrics confirm that the concept geometry has been genuinely disrupted, practitioners can have higher confidence that the unlearning is not merely suppression; if fragmentation remains intact, that is a strong signal that adversarial recovery will succeed without running expensive attacks. Its forwardpass-only design makes it applicable even where generation is prohibited, such as API-access audits subject to output restrictions.

## 6 Related Work

Parametric Knowledge and Intrinsic Evaluation. Factual associations are geometrically structured within intermediate transformer layers (Petroni et al., 2019; Geva et al., 2021; Meng et al., 2023). While recent intrinsic probes aim to evaluate this knowledge without token generation (Gottesman and Geva, 2024; Hong et al.), they frequently conflate conceptual content with query syntax or require a reference model. We build upon the Grammatrix entropy framework (Skean et al., 2025) and the geometric grounding of intermediate layers (Razzhigaev et al., 2023), but depart from prior work by measuring the population geometry of anon-vectors. This isolates conceptual identity from structural artifacts without relying on preunlearning baselines.

Machine Unlearning and Adversarial Robustness. Current unlearning interventions— spanning gradient ascent, preference optimisation, and representation steering (Yao et al., 2024; Rafailov et al., 2024; Huu-Tien et al., 2025)— frequently produce suppression rather than genuine deletion (Xu et al., 2025; Lee et al., 2024; Schoepf et al., 2026). Because knowledge remains structurally intact, adversarial techniques such as activation steering (Seyitoglu et al.˘ , 2024), in-context learning (Brown et al., 2020), and GCG (Zou et al., 2023) can bypass behavioral safeguards to recover suppressed facts. Prior representation-level audits either require pre- and post-unlearning model comparisons (Xu et al., 2025) or the active execution of these costly attacks (Lynch et al., 2024). In contrast, our fragmentation framework offers a single-phase, generation-free proxy designed to predict vulnerability to these exact recovery channels using a single geometric forward pass.

## 7 Limitations

Our evaluation has several limitations. First, deterministic string substitution for anonymisation risks tokenizer-driven lexical artifacts. We mitigate this using morphologically similar names and multivariant averaging (K = 5) in RQ2 (supported by consistent ablations in Appendix B.2), though future work should explore tokenization-matched replacements, LLM paraphrasing, or systematically benchmarking alternative anonymisation pipelines beyond the scheme used here. Second, relying on a single anonymised variant (K = 1) per query in RQ1 may retain idiosyncratic phrasing, which we offset by averaging across N = 96–100 queries per concept. Third, our batch-based construction requires $B \geq 2$ forget-set samples to define a valid direction vector, which may not suit single-sample or very small-sample unlearning settings; a promising extension augments a single target sample with neutral or synthetic data to compute a pseudo-batch direction, which we leave to future work. Fourth, the explanation offered in Section 5 for why fragmentation predicts ICL and GCG/Steering with opposite sign is inferred from the correlational pattern alone; we do not validate it against the models actual generations or intermediate reasoning traces, and confirming it would require inspecting chainof-thought or generation-level outputs alongside the geometry, which we leave to future work. Finally, our concept coverage in RQ1, while now validated across four conditions (Harry Potter forget/retain, Star Wars, TOFU; Appendix D.2), remains limited to English-language fictional/factual domains. Expanding to graded knowledge levels and non-fiction technical domains is necessary to fully confirm the generality of our findings.

## References

Tomer Ashuach, Dana Arad, Aaron Mueller, Martin Tutek, and Yonatan Belinkov. 2026. CRISP: Persistent Concept Unlearning via Sparse Autoencoders. arXiv preprint. ArXiv:2508.13650 [cs.CL].

Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, and 12 others. 2020. Language Models are Few-Shot Learners. arXiv preprint. ArXiv:2005.14165 [cs.CL].

Nicholas Carlini, Florian Tramer, Eric Wallace, Matthew Jagielski, Ariel Herbert-Voss, Katherine Lee, Adam Roberts, Tom Brown, Dawn Song, Ulfar Erlingsson, Alina Oprea, and Colin Raffel. 2021. Extracting Training Data from Large Language Models. arXiv preprint. ArXiv:2012.07805 [cs.CR].

Ronen Eldan and Mark Russinovich. Who’s Harry Potter? Approximate Unlearning in LLMs.

Wenjie Fu, Huandong Wang, Chen Gao, Guanghua Liu, Yong Li, and Tao Jiang. Membership Inference Attacks against Fine-tuned Large Language Models via Self-prompt Calibration.

Rohit Gandikota, Sheridan Feucht, Samuel Marks, and David Bau. 2025. Erasing Conceptual Knowledge from Language Models. arXiv preprint. ArXiv:2410.02760 [cs.CL].

Mor Geva, Jasmijn Bastings, Katja Filippova, and Amir Globerson. 2023. Dissecting Recall of Factual Associations in Auto-Regressive Language Models. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 12216–12235, Singapore. Association for Computational Linguistics.

Mor Geva, Avi Caciularu, Kevin Wang, and Yoav Goldberg. 2022. Transformer Feed-Forward Layers Build Predictions by Promoting Concepts in the Vocabulary Space. In Proceedings ofthe 2022 Conference on Empirical Methods in Natural Language Processing, pages 30–45, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

Mor Geva, Roei Schuster, Jonathan Berant, and Omer Levy. 2021. Transformer Feed-Forward Layers Are Key-Value Memories. arXiv preprint. ArXiv:2012.14913 [cs].

Daniela Gottesman and Mor Geva. 2024. Estimating knowledge in large language models without generating a single token. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 3994–4019.

Yihuai Hong, Lei Yu, Haiqin Yang, Shauli Ravfogel, and Mor Geva. Intrinsic Test of Unlearning Using Parametric Knowledge Traces.

Dang Huu-Tien, Trung-Tin Pham, Hoang Thanh-Tung, and Naoya Inoue. 2025. On Effects of Steering Latent Representation for Large Language Model Unlearning. arXiv preprint. ArXiv:2408.06223 [cs].

Andrew Lee, Xiaoyan Bai, Itamar Pres, Martin Wattenberg, Jonathan K. Kummerfeld, and Rada Mihalcea. 2024. A Mechanistic Understanding of Alignment Algorithms: A Case Study on DPO and Toxicity. arXiv preprint. ArXiv:2401.01967 [cs.CL].

Nathaniel Li, Alexander Pan, Anjali Gopal, Summer Yue, Daniel Berrios, Alice Gatti, Justin D. Li, Ann-Kathrin Dombrowski, Shashwat Goel, Long Phan, Gabriel Mukobi, Nathan Helm-Burger, Rassin Lababidi, Lennart Justen, Andrew B. Liu, Michael Chen, Isabelle Barrass, Oliver Zhang, Xiaoyuan Zhu, and 38 others. 2024. The WMDP Benchmark: Measuring and Reducing Malicious Use With Unlearning. arXiv preprint. ArXiv:2403.03218 [cs.LG].

Aengus Lynch, Phillip Guo, Aidan Ewart, Stephen Casper, and Dylan Hadfield-Menell. 2024. Eight Methods to Evaluate Robust Unlearning in LLMs. ArXiv:2402.16835 [cs].

Pratyush Maini, Zhili Feng, Avi Schwarzschild, Zachary C. Lipton, and J. Zico Kolter. 2024. TOFU: A Task of Fictitious Unlearning for LLMs. arXiv preprint. ArXiv:2401.06121 [cs.LG].

Kevin Meng, David Bau, Alex Andonian, and Yonatan Belinkov. 2023. Locating and Editing Factual Associations in GPT. arXiv preprint. ArXiv:2202.05262 [cs.CL].

Fabio Petroni, Tim Rocktäschel, Patrick Lewis, Anton Bakhtin, Yuxiang Wu, Alexander H. Miller, and Sebastian Riedel. 2019. Language Models as Knowledge Bases? arXiv preprint. ArXiv:1909.01066 [cs.CL].

Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, and Chelsea Finn. 2024. Direct Preference Optimization: Your Language Model is Secretly a Reward Model. arXiv preprint. ArXiv:2305.18290 [cs.LG].

Anton Razzhigaev, Matvey Mikhalchuk, Elizaveta Goncharova, Ivan Oseledets, Denis Dimitrov, and Andrey Kuznetsov. 2023. The Shape of Learning: Anisotropy and Intrinsic Dimensions in Transformer-Based Models.

Stefan Schoepf, Michael Curtis Mozer, Nicole Elyse Mitchell, Alexandra Brintrup, Georgios Kaissis, Peter Kairouz, and Eleni Triantafillou. 2026. Redirection for Erasing Memory (REM): Towards a universal unlearning method for corrupted data. arXiv preprint. ArXiv:2505.17730 [cs].

Atakan Seyitoglu, Aleksei Kuvshinov, Leo Schwinn,˘ and Stephan Günnemann. 2024. Extracting Unlearned Information from LLMs with Activation Steering. arXiv preprint. ArXiv:2411.02631 [cs].

Weijia Shi, Jaechan Lee, Yangsibo Huang, Sadhika Malladi, Jieyu Zhao, Ari Holtzman, Daogao Liu, Luke Zettlemoyer, Noah A Smith, and Chiyuan Zhang. MUSE: Machine Unlearning Six-Way Evaluation for Language Models.

Oscar Skean, Md Rifat Arefin, Dan Zhao, Niket Patel, Jalal Naghiyev, Yann LeCun, and Ravid Shwartz-Ziv. 2025. Layer by Layer: Uncovering Hidden Representations in Language Models. arXiv preprint. ArXiv:2502.02013 [cs].

Xiaoyu Xu, Xiang Yue, Yang Liu, Qingqing Ye, Huadi Zheng, Peizhao Hu, Minxin Du, and Haibo Hu. 2025. Unlearning Isn’t Deletion: Investigating Reversibility of Machine Unlearning in LLMs. arXiv preprint. ArXiv:2505.16831 [cs].

Yuanshun Yao, Xiaojun Xu, and Yang Liu. 2024. Large Language Model Unlearning. arXiv preprint. ArXiv:2310.10683 [cs.CL].

Ruiqi Zhang, Licong Lin, Yu Bai, and Song Mei. 2024. Negative Preference Optimization: From Catastrophic Collapse to Effective Unlearning. arXiv preprint. ArXiv:2404.05868 [cs.LG].

Andy Zou, Zifan Wang, Nicholas Carlini, Milad Nasr, J. Zico Kolter, and Matt Fredrikson. 2023. Universal and Transferable Adversarial Attacks on Aligned Language Models. arXiv preprint. ArXiv:2307.15043 [cs.CL].

## A Mathematical Proofs

This section provides complete proofs for the theoretical results stated in Section 3.2.

## A.1 Proof of Theorem 1

Theorem (High entropy forces geometric fragmentation). Let $\begin{array} { c c l } { A } & { { } \in } & { \mathbb { R } ^ { n \times d } } \end{array}$ be a matrix of L<sub>2</sub>-normalised activation vectors and let $G \ =$ $A A ^ { \top }$ be the corresponding Gram matrix with EfRank $\because ( G ) = \operatorname { T r } ( G ) / \lambda _ { \operatorname* { m a x } } ( G )$ . Then the mean pairwise squared Euclidean distance satisfies

$$
D _ { L _ { 2 } } ^ { 2 } \geq 2 - \frac { 2 } { \mathrm { E f f R a n k } ( G ) } .
$$

Proof. Let $v _ { 1 } , \ldots , v _ { n } \in \mathbb { R } ^ { d }$ denote the rows of A, with $\| v _ { i } \| _ { 2 } = 1$ for all i.

Step 1 (Gram matrix structure). Because the rows are unit vectors, $G _ { i j } = v _ { i } ^ { \top } v _ { j }$ and the diagonal satisfies $G _ { i i } ~ = ~ 1$ , so $\operatorname { T r } ( G ) = n$ . Hence EfRank $( G ) = n / \lambda _ { \operatorname* { m a x } } ( G )$

Step 2 (Distance–cosine identity). For any pair $i \neq j$

$$
\begin{array} { r } { \| v _ { i } - v _ { j } \| _ { 2 } ^ { 2 } = \| v _ { i } \| _ { 2 } ^ { 2 } - 2 v _ { i } ^ { \top } v _ { j } + \| v _ { j } \| _ { 2 } ^ { 2 } = 2 - 2 G _ { i j } . } \end{array}
$$

Step 3 (Express $D _ { L _ { 2 } } ^ { 2 }$ via the Gram matrix). Define the mean off-diagonal cosine similarity

$$
\bar { c } = \frac { 1 } { n ( n - 1 ) } \sum _ { i \neq j } G _ { i j } ,
$$

so that $D _ { L _ { 2 } } ^ { 2 } = 2 - 2 \bar { c } .$ Summing all entries of G and separating diagonal from off-diagonal,

$$
\mathbf { 1 } ^ { \top } G \mathbf { 1 } = \operatorname { T r } ( G ) + \sum _ { i \neq j } G _ { i j } = n + n ( n - 1 ) { \bar { c } } .
$$

Solving for c¯ and substituting:

$$
\begin{array} { c } { D _ { L _ { 2 } } ^ { 2 } = 2 - 2 \bar { c } } \\ { = \frac { 2 n - 2 \mathbf { 1 } ^ { \top } G \mathbf { 1 } / n } { n - 1 } . } \end{array}
$$

Step 4 (Rayleigh quotient bound). For any unit vector x, $\lambda _ { \operatorname* { m a x } } ( G ) \geq x ^ { \top } G x$ . Taking $x = \mathbf { 1 } / \sqrt { n }$ (which satisfies $\| { \boldsymbol { x } } \| _ { 2 } = 1 )$

$$
\lambda _ { \operatorname* { m a x } } ( G ) \geq \frac { \mathbf { 1 } ^ { \top } G \mathbf { 1 } } { n } .
$$

Step 5 (Conclusion). From Step 4, $\mathbf { 1 } ^ { \top } G \mathbf { 1 } / n \leq$ $\lambda _ { \operatorname* { m a x } } ( G ) , \operatorname { s o } 2 n - 2 \mathbf { 1 } ^ { \top } G \mathbf { 1 } / n \geq 2 n - 2 \lambda _ { \operatorname* { m a x } } ( G )$ Dividing by $n - 1 > 0 ;$

$$
\begin{array} { l } { \displaystyle { D _ { L _ { 2 } } ^ { 2 } = \frac { 2 n - 2 \mathbf { 1 } ^ { \top } G \mathbf { 1 } / n } { n - 1 } \mathrm { ~ } \geq \frac { 2 ( n - \lambda _ { \operatorname* { m a x } } ( G ) ) } { n - 1 } } } \\ { \displaystyle { \qquad = \frac { 2 n } { n - 1 } \bigg ( 1 - \frac { 1 } { \mathrm { E f f R a n k } ( G ) } \bigg ) . } } \end{array}
$$

Since $n / ( n - 1 ) \geq 1$ for all $n \geq 2$

$$
\begin{array} { c } { { D _ { L _ { 2 } } ^ { 2 } \geq 2 \bigg ( 1 - \frac { 1 } { { \mathrm { E f f R a n k } ( G ) } } \bigg ) } } \\ { { = 2 - \frac { 2 } { { \mathrm { E f f R a n k } ( G ) } } . } } \end{array}
$$

Remark. The proofactually establishes the tighter bound $D _ { L _ { 2 } } ^ { 2 } ~ \stackrel { . } { \geq } ~ \frac { 2 n } { n - 1 } \big ( 1 ~ - ~ 1 / \mathrm { E f f R a n k } ( G ) \big )$ ; the theorem states the slightly weaker form $2 \textrm { -- }$ $2 / \mathrm { E f f R a n k } ( G )$ , which is independent of n and easier to interpret. The bound is tight (up to the $n / ( n { - } 1 )$ factor) when all n vectors are mutually orthogonal: then $G = I _ { n } , \lambda _ { \mathrm { m a x } } ( G ) = 1$ $\operatorname { E f f R a n k } ( G ) = n ,$ and $D _ { L _ { 2 } } ^ { 2 } = 2 = 2 - 2 / n + 2 / n .$

## A.2 Entropy–EffRank Relationship

The following proposition formalises the claim in Section 3.2 that a high Rényi-2 Gram-matrix entropy $S _ { 2 }$ forces a large effective rank, completing the implication chain high $S _ { 2 } \Rightarrow$ high EffRank ⇒ large $D _ { L _ { 2 } } ^ { 2 }$

Proposition 1 (Entropy–EffRank sandwich). Let G be a positive semi-definite matrix with normalised spectral distribution $\begin{array} { r } { p _ { i } = \lambda _ { i } ( G ) / \mathrm { T r } ( G ) \geq 0 , \sum _ { i } p _ { i } = 1 , o r d e r e d p _ { 1 } \geq } \end{array}$ $p _ { 2 } \geq \cdots \geq 0$ . Define ${ \overline { { S _ { 2 } } } } ( G ) = - \log _ { 2 } ( \sum _ { i } p _ { i } ^ { 2 } )$ and $\mathrm { E f f R a n k } ( G ) = 1 / p _ { 1 }$ . Then

$$
2 ^ { S _ { 2 } ( G ) / 2 } \leq \mathrm { E f f R a n k } ( G ) \leq 2 ^ { S _ { 2 } ( G ) } .
$$

Proof. Upper bound. Because $p _ { 1 } \geq p _ { i }$ for every $i ,$

$$
\sum _ { i } p _ { i } ^ { 2 } \ \leq \ p _ { 1 } \sum _ { i } p _ { i } = p _ { 1 } .
$$

Taking reciprocals (both sides positive) reverses the inequality: $\begin{array} { r } { 1 / p _ { 1 } \le 1 / \sum _ { i } p _ { i } ^ { 2 } , } \end{array}$ i.e., EfRank $( G ) \leq$ $2 ^ { S _ { 2 } ( G ) }$

Lower bound. Trivially, $p _ { 1 } ^ { 2 } \leq \sum _ { i } p _ { i } ^ { 2 }$ (one nonnegative term cannot exceed their sum). Taking square roots: $p _ { 1 } \leq \sqrt { \sum _ { i } p _ { i } ^ { 2 } } = 2 ^ { - S _ { 2 } ( G ) / 2 }$ . Hence EfRank $( G ) = 1 / p _ { 1 } \overset { \cdot } { \geq } 2 ^ { S _ { 2 } ( G ) / 2 }$ □

Corollary 1 (High $S _ { 2 }$ forces geometric fragmentation). For any matrix $A \in \mathbb { R } ^ { n \times d } o f L _ { 2 }$ -normalised vectors with Gram matrix $G = A A ^ { \top }$

$$
D _ { L _ { 2 } } ^ { 2 } \ \geq \ 2 - 2 \cdot 2 ^ { - S _ { 2 } ( G ) / 2 } \ = \ 2 \big ( 1 - 2 ^ { - S _ { 2 } ( G ) / 2 } \big ) .
$$

In particular, $D _ { L _ { 2 } } ^ { 2 } \to 2 a s S _ { 2 } ( G ) \to \infty .$

Proof. From Proposition 1, EfRank $\begin{array} { r l } { ( G ) } & { { } \geq } \end{array}$ $2 ^ { S _ { 2 } ( \bar { G } ) / 2 }$ , so 1/EfRank $( G ) \leq 2 ^ { - S _ { 2 } ( G ) / 2 }$ . Substituting into Theorem 1: $D _ { L _ { 2 } } ^ { 2 } \ \geq \ \ 2 \ -$ $2 / \mathrm { E f f R a n k } ( G ) \geq 2 - 2 \cdot 2 ^ { - S _ { 2 } ( G ) / 2 } .$ □

Remark. Proposition 1 shows that $S _ { 2 }$ and EfRank are tightly coupled: both grow together as the spectral mass spreads across more eigenvalue directions. The upper bound EfRank $\leq 2 ^ { S _ { 2 } }$ also shows that the two quantities agree to within a factor of two on the log-scale, justifying the shorthand $^ {  } S _ { 2 }$ is an increasing function of EfRank” used in the main text.

## A.3 Proof of Theorem 2 and the Exact Entropy Identity

We first establish a key lemma that is an exact algebraic identity—no inequalities, no extra assumptions. It shows that $S _ { 2 }$ and the mean squared pairwise cosine are two encodings of the same spectral quantity.

Lemma 1 (Gram entropy equals mean squared cosine). Let $v _ { 1 } , \ldots , v _ { n } \in \mathbb { R } ^ { d }$ be $L _ { 2 }$ -normalised vectors with Gram matrix $G = A A ^ { \top }$ and normalised Gram $K _ { G } ~ = ~ G / \mathrm { T r } ( G ) ~ = ~ G / n$ . Let $\begin{array} { r } { \overline { { \cos ^ { 2 } } } ~ = ~ \frac { 1 } { n ( n - 1 ) } \sum _ { i \neq j } ( v _ { i } ^ { \top } \dot { v } _ { j } ) ^ { 2 } } \end{array}$ be the mean pairwise squared cosine similarity. Then

$$
\mathrm { T r } ( K _ { G } ^ { 2 } ) = \frac { 1 + ( n - 1 ) \overline { { { \cos ^ { 2 } } } } } { n }\tag{3}
$$

equivalently $\begin{array} { r } { S _ { 2 } ( G ) = - \log _ { 2 } \left( \frac { 1 + ( n - 1 ) \overline { { \cos ^ { 2 } } } } { n } \right) } \end{array}$ . In particular, $S _ { 2 }$ is a strictly decreasing bijection of ${ \overline { { \cos ^ { 2 } } } } \colon S _ { 2 } = 0 \Leftrightarrow { \overline { { \cos ^ { 2 } } } } = 1$ (all pairs parallel or anti-parallel), and $S _ { 2 } = \log _ { 2 } n \Leftrightarrow { \overline { { \mathrm { c o s } ^ { 2 } } } } = 0$ (all pairs orthogonal).

Proof. Expanding the Frobenius norm of $G \mathrm { : }$

$$
\begin{array} { l } { \displaystyle \mathrm { T r } ( G ^ { 2 } ) = \sum _ { i , j = 1 } ^ { n } G _ { i j } ^ { 2 } } \\ { = \sum _ { i \displaystyle \mathrm { ( d i a g o n a l ) } \atop i \mathrm { ( d i a g o n a l ) } = 1 } + \sum _ { i \displaystyle \mathrm { \neq } j } G _ { i j } ^ { 2 } } \\ { = n + n ( n - 1 ) \overline { { \cos ^ { 2 } } } . } \end{array}
$$

Dividing by $\operatorname { T r } ( G ) ^ { 2 } = n ^ { 2 }$

$$
\begin{array} { l } { \displaystyle \operatorname { T r } ( K _ { G } ^ { 2 } ) = \frac { \operatorname { T r } ( G ^ { 2 } ) } { n ^ { 2 } } = \frac { n + n ( n - 1 ) \cos ^ { 2 } } { n ^ { 2 } } } \\ { \displaystyle = \frac { 1 + ( n - 1 ) \overline { { \cos ^ { 2 } } } } { n } . } \end{array}
$$

Since $\overline { { \cos ^ { 2 } \in \mathrm { ~ [ 0 , 1 ] ~ } } }$ (each $G _ { i j } ~ \in ~ [ - 1 , 1 ] )$ , the right-hand side is continuous and strictly increasing in $\overline { { \cos ^ { 2 } } }$ , making $S _ { 2 } = - \log _ { 2 } ( \cdot )$ a strictly decreasing bijection. Boundary values: ${ \overline { { \cos ^ { 2 } } } } = 1$ gives $\bar { \mathrm { T r } } ( \bar { K } _ { G } ^ { 2 } ) = 1 , S _ { 2 } = 0 ; \overline { { { \cos ^ { 2 } } } } = 0$ gives $\mathrm { T r } ( K _ { G } ^ { 2 } ) = \bar { 1 / } n , S _ { 2 } = \log _ { 2 } n .$ □

Remark. Lemma 1 gives the converse of Theorem 1 for the squared-cosine metric: $S _ { 2 }  0 ~ i f$ and only $i f \overline { { \cos ^ { 2 } } }  1$ , which means every pair of vectors satisfies $| v _ { i } ^ { \top } v _ { j } |  1$ , i.e. they are either perfectly aligned or perfectly anti-aligned. Crucially, a low $S _ { 2 }$ alone does not guarantee that vectors point in the same direction (positive cosines); two anti-parallel unit vectors achieve $S _ { 2 } = 0$ yet $D _ { L _ { 2 } } ^ { 2 } = 4$ . Theorem 2 closes this gap by adding the non-negativity condition, which exactly rules out anti-alignment.

We are now ready to prove Theorem 2.

ProofofTheorem 2. The key step is the following inequality, which holds under the assumption $G _ { i j } \geq 0$ for $i \neq j { \mathrm { : } }$ : since cos $\theta _ { i j } = G _ { i j } \in [ 0 , 1 ]$ we have $\cos ^ { 2 } \theta _ { i j } \le \cos \theta _ { i j }$ , and therefore

$$
{ \begin{array} { l } { \displaystyle \cos ^ { 2 } \mathbf { \Omega } = \left. { \frac { 1 } { n ( n - 1 ) } } \sum _ { i \neq j } \cos ^ { 2 } \theta _ { i j } \right. } \\ { \displaystyle \leq \left. { \frac { 1 } { n ( n - 1 ) } } \sum _ { i \neq j } \cos \theta _ { i j } \right. = \left. { \bar { c } } \right. } \end{array} }
$$

where c¯ is the mean off-diagonal cosine. Rearranging, $\bar { c } \geq \overline { { \mathrm { c o s } ^ { 2 } } }$

From Lemma 1, $\begin{array} { r l r } { \overline { { \cos ^ { 2 } } } } & { { } = } & { \frac { n \mathrm { T r } ( K _ { G } ^ { 2 } ) - 1 } { n - 1 } } \end{array}$ $\frac { n \cdot 2 ^ { - S _ { 2 } } - 1 } { n - 1 }$ , so

$$
\bar { c } \ge \frac { n \cdot 2 ^ { - S _ { 2 } } - 1 } { n - 1 } .
$$

Using $D _ { L _ { 2 } } ^ { 2 } = 2 - 2 \bar { c }$ (Step 2 and Step 3 of the proof of Theorem 1):

$$
\begin{array} { c l } { { D _ { L _ { 2 } } ^ { 2 } = 2 - 2 \bar { c } \ \leq \ 2 - { \displaystyle \frac { 2 ( n \cdot 2 ^ { - S _ { 2 } } - 1 ) } { n - 1 } } } } \\ { { \ } } \\ { { \qquad = \ { \displaystyle \frac { 2 ( n - 1 ) - 2 n \cdot 2 ^ { - S _ { 2 } } + 2 } { n - 1 } } } } \\ { { \ } } \\ { { \qquad = \ { \displaystyle \frac { 2 n \big ( 1 - 2 ^ { - S _ { 2 } } \big ) } { n - 1 } } } . } \end{array}
$$

As $S _ { 2 } ( G ) ~  ~ 0$ , the right-hand side $\begin{array} { r l } {  } & { { } 0 . } \end{array}$ , so $D _ { L _ { 2 } } ^ { 2 } \to 0 .$ □

Remark. The bound is tight at both extremes. When all n vectors are identical $( v _ { i } = v$ for all i): $G = \mathbf { 1 1 } ^ { \top }$ $\mathrm { T r } ( K _ { G } ^ { 2 } ) = 1 , S _ { 2 } = 0 ,$ and the bound gives $D _ { L { 2 } } ^ { 2 } \leq 0 ,$ matching the exact value $D _ { L _ { 2 } } ^ { 2 } = 0$ . When all pairs are orthogonal $( G _ { i j } = 0$ $f o r i \neq j ,$ so $G = I _ { n } ) \colon S _ { 2 } = \log _ { 2 }$ n and the bound gives $\textstyle { \frac { 2 n } { n - 1 } } ( 1 - 1 / n ) = 2$ , matching the exact value $D _ { L _ { 2 } } ^ { 2 } = 2 .$

## B Implementation Details

This appendix provides the technical implementation details for all experiments. Code and data will be released upon acceptance.

All datasets and models utilized in this study, including MUSE-Books, TOFU, and ConceptVectors, alongside the evaluated language models, were accessed and employed strictly in accordance with their original licenses and intended use cases. These existing artifacts were explicitly designed for advancing machine learning research, which aligns directly with our objective of evaluating knowledge suppression and unlearning mechanisms. We use these datasets and metrics under the MIT license, and therefore publish our code under the same license. Our reference implementation and framework are intended solely for academic and research contexts to audit language model safety; any derivative applications or data distributions must remain compatible with these research-oriented, non-commercial access conditions.

## B.1 Metric Computation

Input format. For every concept query q<sub>i</sub> we construct a regular text and an anonymised text. Regular text concatenates the natural question and its full answer: "{question}\nAnswer: {answer}", and the anonymised text applies a concept-specific substitution map to both fields (see Section B.2 for the three distinct anonymisation procedures). For anon-vector computation, the anon hidden state at layer ℓ is the mean over K anonymised variants:

$$
h _ { \ell } ^ { \mathrm { a n o n } } ( q _ { i } ) = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } h _ { \ell } \Big ( a _ { i } ^ { ( k ) } \Big ) .
$$

$K { = } 1$ for RQ1 hidden-state extraction and RQ2 forget pairs; $K { = } 5$ for the steering-attack SV computation in RQ2 (see Section B.6).

## B.2 Anonymisation Procedures

All anonymised texts are produced by deterministic string substitution — no language model is used at any stage of the anonymisation pipeline. Three distinct procedures are used, depending on the dataset and downstream task.

Procedure 1: HP pairs for RQ1 (single-variant, fictional-name substitution). The MUSE-Books HP pairs are anonymised using the shared entity table from data/generate\_anon\_qa.py (HP\_REPLACEMENTS), which contains 83 namedentity rules applied in order of decreasing string length to prevent partial matches (e.g. “Hermione Granger” is replaced before “Hermione”), and created using Seyitoglu et al. ˘ (2024) and expanded with Claude-Opus-4.6. Replacements are plausible fictional names that carry no HP-specific signal; variant index 0 is selected, yielding a single deterministic anonymised pair per original pair (K=1). Key substitutions: Harry Potter → Alex Ryden, Hermione Granger → Vera Ashwood, Ron Weasley → Sam Coldridge, Lord Voldemort → Dark Emperor Mordax, Albus Dumbledore → Headmaster

Voss, Hogwarts → Crystal Academy, Quidditch → Broomball. The substitution is applied to both the question and the answer using a case-insensitive regex. This is the same table used by Procedure 3 for RQ2; using variant 0 ensures methodological consistency across tasks. 85 of 96 pairs are changed; the remaining 11 contain no HP-specific entity in the table.

Procedure 2: TOFU pairs for RQ1 (singlevariant, author-table substitution). TOFU pairs reference entirely fictitious authors. We prescan all 100 pairs to collect every unique author full name, then build a global fictional-name map: each author is assigned first\_name[i] + last\_name[i] from the name pools of Seyitoglu et˘ al. (tofu\_anonymize.json), indexed deterministically in order of first appearance (with explicit overrides from the dataset’s replacements field taking precedence). The map is applied consistently across all pairs (longest-first, case-insensitive), so cross-pair author identity is preserved throughout the dataset. 83 of 100 pairs are changed; the remaining 17 contain only generic phrasing with no author-name mention that the regex can match.

Procedure 3: ConceptVectors QA pairs for RQ2 (five-variant, entity-table substitution). For the steering-attack SV computation in RQ2, each evaluation question requires multiple anonymised variants so that the steering vector averages over several “identity-free” phrasings of the same question. We pre-generate K=5 such variants per question at data-build time using data/generate\_anon\_qa.py, which maintains a separate replacement table per concept. Each table lists 5 fictional substitutes for every named entity in the concept:

• Harry Potter → Alex Ryden, Marcus Cole, Torin Drake, Daven Ash, Kael Stone (and analogous per-character tables for Hermione Granger, Voldemort, Hogwarts, Quidditch, J.K. Rowling, etc.).

• Star Wars → Kai Starborn, Zane Lightfield, Taran Skyes, Riven Starfall, Dax Brightstar (for Luke Skywalker; equivalent tables for Darth Vader, Han Solo, Tatooine, lightsaber, etc.).

• William Shakespeare → equivalent fivename tables for Shakespeare himself, Hamlet,

Macbeth, the Globe Theatre, Stratford-upon-Avon, etc.

Substitution is again longest-first and literal. Variant k is constructed by replacing every matched entity with its k-th substitute (same index k for all entities within a single variant, so the five variants each represent a self-consistent fictional universe). The five anonymised questions are stored in the anon\_questions list field of each qa\_eval\_pairs entry in the pre-built concept JSON files.

Note on RQ2 forget pairs. The Wikipedia-derived forget pairs used for training unlearning (not for the steering SV) use bracket placeholder tokens instead of fictional names (e.g. Harry Potter → [THE PROTAGONIST], Lord Voldemort → [THE ANTAGONIST], Hogwarts → [THE ACADEMY]). This produces a single anonymised pair per training example; multiple variants are not needed for the CE-loss computation.

## B.3 Extraction, Sampling, and Layer Selection

Hidden-state extraction. All hidden states are extracted in a single forward pass with no gradient computation (torch.no\_grad()). We register output hooks on every transformer block to collect the full stack of L+1 hidden states (embedding layer included) simultaneously for a single tokenised input. The per-token hidden state at position t and layer ℓ is mean-pooled across the sequence dimension using an attention-mask-weighted average, yielding a single vector $h _ { \ell } ( q _ { i } ) \in \mathbb { R } ^ { d }$ per input. All models are loaded in bfloat16 precision with device\_map="auto" to distribute parameters across available GPUs.

Batch sizes and sample counts.

• RQ1 (knowledge detection): N=96 Harry Potter query pairs (MUSE-Books) and N=100 TOFU pairs, each split into B=10 random batches of approximately 9–10 pairs for computing per-batch direction vectors. All ten models are evaluated without repetition (embeddings are deterministic).

• RQ2 (unlearning): Concept-specific forget sets contain approximately 50–80 Wikipediaderived text pairs (see Section B.4). Fragmentation metrics are computed on the full forget set at the ${ \mathrm { \bf A U S S } } _ { \mathrm { \bf L } 2 } .$ -peak layer identified individually for each model–concept pair; B=10 batches.

Rationale for hyperparameter choices. Several choices in our pipeline are not uniquely determined by the experimental design; we document them here for transparency. $B { = } 1 0$ batches: from $B { = } 2$ upward every metric separates HP from TOFU in the hypothesized direction, with effect size growing monotonically with B (Table 3); the paired Wilcoxon test’s resolution is bounded by n=9 models rather than $B ,$ so $B { = } 1 0$ is a practical middle ground, not an arbitrary or fragile choice. B=1 is structurally undefined (no batch-to-batch variance with a single sample); a single-item reformulation that avoids this constraint entirely is discussed in Section 7.

Table 3: Batch-size sensitivity: Cohen’s d (Wilcoxon p) for HP vs. TOFU separation at $B \in \{ 2 , 5 , 1 0 , 2 0 , 5 0 \}$ $n { = } 9$ models, direction-corrected fresh extraction. All cells are in the hypothesized direction.
<table><tr><td>Metric</td><td>B</td><td>d (p)</td></tr><tr><td rowspan="3">∥∆∥|/Global_Coh.</td><td>2</td><td>0.14 (.82)</td></tr><tr><td>5</td><td>0.64 (.13)</td></tr><tr><td>10 20</td><td>0.72 (.13) 0.92 (.04)</td></tr><tr><td colspan="3">50</td></tr><tr><td rowspan="6">Batch_SecondMoment_TopEig</td><td>2</td><td>1.13 (.02) 0.19 (.73)</td></tr><tr><td>5</td><td>0.35 (.36)</td></tr><tr><td>10</td><td>0.42 (.36)</td></tr><tr><td>20</td><td>0.43 (.36)</td></tr><tr><td>50</td><td>0.37 (.57)</td></tr><tr><td>2</td><td>0.17 (.73)</td></tr><tr><td rowspan="6">AUSSL2 / Batch_Cov_Trace</td><td>5</td><td>0.62 (.13)</td></tr><tr><td>10</td><td></td></tr><tr><td></td><td>0.74 (.10)</td></tr><tr><td>20</td><td>0.93 (.04)</td></tr><tr><td>50</td><td>1.10 (.02)</td></tr><tr><td>2</td><td>0.31 (.43)</td></tr><tr><td rowspan="6"> $\mathrm { \bf A U S S _ { c o s ^ { 2 } } }$ </td><td>5</td><td>0.40 (.36)</td></tr><tr><td>10</td><td></td></tr><tr><td></td><td>0.49 (.30)</td></tr><tr><td>20</td><td>0.47 (.30)</td></tr><tr><td>50</td><td>0.43 (.20)</td></tr><tr><td>2 5</td><td>1.25 (.01)</td></tr><tr><td rowspan="4"> $\mathrm { \ A U S S { _ { J a c } } }$ </td><td>10</td><td>1.18 (.02) 1.18 (.01)</td></tr><tr><td></td><td></td></tr><tr><td>20</td><td>0.97 (.02)</td></tr><tr><td>50</td><td>0.54 (.16)</td></tr><tr><td rowspan="5">Batch_Cov_TopEig</td><td>2 5</td><td>0.17 (.73)</td></tr><tr><td></td><td>0.42 (.30)</td></tr><tr><td>10</td><td>0.18 (.65)</td></tr><tr><td></td><td></td></tr><tr><td>20 50</td><td>0.28 (.43) 0.33 (.43)</td></tr></table>

Seed 42: standard reproducibility seed; sensitivity to the partition seed is a stated limitation (Section 7). Peak layer = arg max<sub>ℓ</sub> $\mathbf { A U S S } _ { \mathbf { L } 2 } \mathbf { : }$ $\mathsf { A U S S } _ { \mathrm { L 2 } }$ had the strongest RQ1 signal in preliminary runs; the same layer is used for all metrics within a run to avoid per-metric search costs. Perlayer correlation analysis confirms that the final layers carry the strongest ASR signal for most attacks. K=1 for RQ1 and forget pairs, $K { = } 5$ for steering SV: a single anonymised variant is sufficient for metric computation and minimises hiddenstate extraction time; five variants are used for the steering vector to average over phrasings and reduce direction noise (Seyitoglu et al.˘ , 2024). ICL: $n _ { \mathbf { S h o t s } } { = } 5 ,$ , ROUGE-L threshold 0.30: five-shot is the standard few-shot setting of Brown et al. (2020); the 0.30 threshold is loose enough to accept paraphrases and matches the TOFU benchmark convention (Maini et al., 2024). $\mathbf { G C G } \colon n _ { \mathbf { e v a l } } { = } 5$ (shortest answers), 500 steps, suffix length 20: shortestanswer pairs converge fastest under gradient-based optimisation; 500 steps and a 20-token suffix are the nanogcg defaults from Zou et al. (2023). MIA: 50 forget / 50 retain: equal-class balance for ROC AUC computation; standard in membershipinference literature. RMU: $\ell _ { r } { = } 7 , c { = } 2 0 , \alpha _ { r } { = } 1 2 0 0 { : }$ adopted directly from the defaults of Huu-Tien et al. (2025) for 7–8B models; $\ell _ { r } { = } 7$ is approximately the middle of the network. WHP excluded from correlation analysis: WHP redirects concept queries to anonymised answers rather than erasing the representation; its geometry is qualitatively different from the suppression-based methods (GA/DPO/NPO/NPO+KL/RMU), and including it would confound the correlation analysis.

Peak-layer selection. For RQ1, each metric selects its own peak layer $\ell ^ { * }$ independently as the layer that maximises the HP–TOFU separation for that metric (i.e. arg max<sub>ℓ</sub> $\left[ \bar { m } _ { \ell } ^ { \mathrm { { T O F U } } } - \bar { m } _ { \ell } ^ { \mathrm { { H P } } } \right]$ for dispersion metrics, reversed for coherence metrics where HP should be higher). This per-metric peak is computed across all ten models simultaneously. For RQ2, $\ell ^ { * }$ is selected per run as the layer that maximises $\mathsf { A U S S } _ { \mathrm { L } 2 }$ on the post-unlearning checkpoint; the same layer is used for all other fragmentation metrics in that run.

A practical, label-free layer-selection heuristic. The oracle per-model peak layer is not idiosyncratic: relative depth $( \ell ^ { * } / L )$ clusters tightly across all 9 models (0.79–0.98), with the highestperforming layer consistently falling in the 0.80– 0.88 range. A single fixed relative depth of $0 . 8 \times$ model depth, using no per-model information at all, already gives 8/8 metrics correct-direction and statistically significant at $n { = } 9$ . Leave-one-model-out transfer of the median peak relative depth (learned from the other 8 models, applied blind to the heldout model) achieves 72/72 (model, metric) pairs correct-direction, identical to the oracle per-model search (Table 4). For practical deployment, taking the layer at 0.8× depth is close enough to the oracle-optimal result and requires no per-model calibration.

Table 4: Fixed relative-depth sweep (no per-model search), n=9 models, Cohen’s d (p shown only at $r { = } 0 . 8 ,$ , all $p { \leq } 0 . 0 1 2$ there); sign shown per metric’s own coherence/dispersion convention, all cells correctdirection at r=0.8. LOMO transfer: leave-one-modelout transfer of the median peak depth from the other 8 models, all 8 metrics, all 9 held-out models (72/72 pairs), identical to the 72/72 oracle per-model search.
<table><tr><td rowspan=1 colspan=1>Metric</td><td rowspan=1 colspan=1>r</td><td rowspan=1 colspan=1>d</td></tr><tr><td rowspan=2 colspan=1>||∆σ||</td><td rowspan=1 colspan=1>0.20.3</td><td rowspan=1 colspan=1>0.420.67</td></tr><tr><td rowspan=1 colspan=1>0.40.50.60.70.8</td><td rowspan=1 colspan=1>1.341.511.722.213.30 (.0039)</td></tr><tr><td rowspan=1 colspan=1> $\mathbf { A U S S } _ { \mathrm { L } 2 }$ </td><td rowspan=1 colspan=1>0.20.30.40.50.60.70.8</td><td rowspan=1 colspan=1>0.480.701.301.421.622.043.01 (.0039)</td></tr><tr><td rowspan=1 colspan=1> $\mathrm { \ A U S S _ { c o s ^ { 2 } } }$ </td><td rowspan=1 colspan=1>0.20.30.40.50.60.70.8</td><td rowspan=1 colspan=1>0.040.120.220.310.600.961.65 (.0078)</td></tr><tr><td rowspan=1 colspan=1> $\mathrm { \ A U S S _ { J a c } }$ </td><td rowspan=1 colspan=1>0.20.30.40.50.60.70.8</td><td rowspan=1 colspan=1>0.340.831.211.441.812.152.60 (.0039)</td></tr><tr><td rowspan=1 colspan=1>Batch_Cov_TopEig</td><td rowspan=1 colspan=1>0.20.30.40.50.60.70.8</td><td rowspan=1 colspan=1>0.090.180.841.422.102.606.31 (.0039)</td></tr></table>

Gram-entropy computation. Eigenvalues of $\begin{array} { l l l } { K _ { Z } } & { = } & { Z Z ^ { \top } / \mathrm { T r } ( Z Z ^ { \top } ) } \end{array}$ are computed with torch.linalg.eigvalsh (symmetric eigendecomposition, stable for positive-semidefinite matrices). Negative eigenvalues arising from numerical error are clamped to zero and the spectrum is renormalised to sum to one before computing entropy. For S<sub>1</sub> we mask eigenvalues $\lambda _ { i } < 1 0 ^ { - 1 2 }$ to avoid log(0). Batches with $\mathrm { T r } ( Z Z ^ { \top } ) < 1 0 ^ { - 1 0 }$ or $N < 2$ are skipped and flagged as NaN.

## B.4 Dataset Details

Harry Potter (HP) — RQ1. We use the MUSE-Books benchmark (Shi et al.), accessed via HuggingFace (muse-bench/MUSE-Books, configuration knowmem, split forget\_qa). The split contains 300 open-domain question–answer pairs drawn from the seven Harry Potter novels. We retain N=96 pairs after filtering to those with non-empty answers; these cover characters, spells, locations, and plot events.

Each pair is anonymised using Procedure 1 (Section B.2): the 83-entry fictional-name table from HP\_REPLACEMENTS, variant index 0, yielding a single anonymised pair per original pair $( K { = } 1 )$ . For example, Harry Potter → Alex Ryden and $H o g \mathrm { - }$ warts → Crystal Academy.

TOFU — RQ1. We use the TOFU dataset (Maini et al., 2024) (locuslab/TOFU, HuggingFace), which contains 4,000 question– answer pairs about 200 entirely fictitious authors generated by GPT-4 and never present in any public training corpus. We sample N=100 pairs uniformly at random (seed 42) from the full-retain split. Because the authors are fabricated, all ten models score 0.00 exact-match QA accuracy on TOFU, making it an ideal null baseline for the knowledge-detection experiment. Procedure 2 (Section B.2) is applied: a global fictional-name map over all TOFU fictitious authors, keeping author identity consistent across pairs. 83 of 100 pairs are changed; 17 pairs containing only generic phrasing are unchanged by design—the anon-vector should carry minimal concept-specific signal for unknown concepts.

ConceptVectors — RQ2. For the unlearning experiments we use three target concepts: Harry Potter, Star Wars, and William Shakespeare. Forget pairs and evaluation QA pairs are constructed from the ConceptVectors dataset (Hong et al.) (YihuaiHong/ConceptVectors, Hugging-Face), which provides Wikipedia-sourced text completions and question–answer pairs for 293 named entities across multiple model families.

For each concept we use the test and development splits for both llama2-7b and olmo-7b entries, deduplicating by question text. Wikipedia article text is additionally segmented into slidingwindow (prefix, continuation) pairs of approximately 128 tokens to increase the effective forgetset size. The final forget set contains 50–80 pairs per concept. Retain pairs are drawn from the unrelated\_QA field of other concepts in ConceptVectors, ensuring the retain set covers facts the model should continue to answer correctly throughout unlearning; typically 50 retain pairs are used.

Important implementation note: The ConceptVectors HuggingFace repository contains JSON files with heterogeneous schemas across model families; we bypass the datasets.load\_dataset() API and load the JSON files directly from the local cache to avoid schema-mismatch errors.

## B.5 Unlearning Details

Hardware. All unlearning experiments are conducted on two NVIDIA L40S GPUs (48 GB VRAM each, 96 GB total). Models are loaded with device\_map="auto" in bfloat16 precision, causing the HuggingFace accelerate library to automatically shard larger models (LLaMA-2-7B, LLaMA-3-8B, Qwen2.5-7B) across both GPUs. PYTORCH\_CUDA\_ALLOC\_CONF= expandable\_segments:True is set at process start to eliminate CUDA memory fragmentation across sequential unlearning runs. Total GPUhours were not directly logged, so we estimate a total from process timestamps recorded in our run logs. RQ1 hidden-state extraction averages ≈ 31 minutes per model across the ten evaluated models; the RQ2 unlearning-and-attack pipeline averages ≈10–20 minutes per model–method–concept run across the 63-run grid (Appendix D.7), with additional passes for the layer-resolved sweep (Section D.5), the $\mathrm { C K A } / L _ { 2 }$ baseline comparison (Section 4), and the per-sample analyses in this appendix. Both GPUs are occupied for the duration of every run (device\_map="auto" shards each model across both). Extrapolating these per-run durations across the full set of runs logged for this project, we estimate total compute at approximately 120–150 GPU-hours on the hardware described above.

Common training configuration. Unless noted otherwise, all methods share the following hyperparameters: 10 training epochs, learning rate $\bar { \eta } \ \stackrel { - } { = } \ 1 0 ^ { - 5 }$ , batch size $B \ = \ 4$ , maximum sequence length 256 tokens, gradient clipping to norm 1.0, and the AdamW optimiser (8-bit variant via bitsandbytes when available, otherwise standard PyTorch AdamW). Gradient checkpointing is enabled throughout to reduce peak VRAM usage. A frozen reference-model copy is loaded onto

CPU memory (pinned) and moved to GPU only when needed, which is required by DPO, NPO, $\mathrm { N P O + K L }$ , and RMU.

Gradient Ascent (GA). GA negates the standard language-modelling cross-entropy loss on the forget pairs:

$$
\mathcal { L } _ { \mathrm { G A } } = - \mathcal { L } _ { \mathrm { C E } } ( \theta ; \mathcal { D } _ { \mathrm { f o r g e t } } ) .
$$

No retain regularisation is applied. This follows the original gradient-ascent unlearning baseline (Yao et al., 2024).

DPO. Direct Preference Optimisation (Rafailov et al., 2024) treats forget pairs as rejected continuations and retain pairs as chosen continuations. The DPO objective is:

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { D P O } } = - \mathbb { E } \Bigl [ \log \sigma \Bigl ( \beta \log \frac { \pi _ { \theta } ( y _ { w } | x ) } { \pi _ { \mathrm { r e f } } ( y _ { w } | x ) } } \\ & { ~ - ~ \beta \log \frac { \pi _ { \theta } ( y _ { l } | x ) } { \pi _ { \mathrm { r e f } } ( y _ { l } | x ) } \Bigr ) \Bigr ] , } \end{array}
$$

with $\beta = 0 . 1$ . Our usage of DPO for unlearning follows the open-unlearning framework of Maini et al. (2024).

NPO. Negative Preference Optimisation (Zhang et al., 2024) pushes the policy away from the reference model on the forget set only:

$$
\mathcal { L } _ { \mathrm { N P O } } = \frac { 2 } { \beta } \mathbb { E } _ { \mathcal { D } _ { \mathrm { f o r g e t } } } \left[ \log \left( 1 + \left( \frac { \pi _ { \theta } } { \pi _ { \mathrm { r e f } } } \right) ^ { - \beta } \right) \right] ,
$$

with $\beta = 0 . 1$

NPO+KL. Extends NPO with a KL-divergence regularisation term on the retain set to preserve general model capability (Zhang et al., 2024):

$$
{ \mathcal { L } } _ { \mathrm { N P O + K L } } = { \mathcal { L } } _ { \mathrm { N P O } } + \alpha D _ { \mathrm { K L } } \bigl ( \pi _ { \theta } \bigl ( \cdot | x _ { r } \bigr ) \parallel \pi _ { \mathrm { r e f } } ( \cdot | x _ { r } ) \bigr ) ,
$$

with $\alpha = 1 . 0$ and $\beta = 0 . 1$

RMU. Representation Misdirection for Unlearning (Huu-Tien et al., 2025) steers the hidden states of forget inputs toward a fixed random direction u at a chosen intermediate layer $\ell _ { r }$ , while simultaneously anchoring retain hidden states close to the frozen reference:

$$
\begin{array} { r l } { \mathcal { L } _ { \mathrm { R M U } } = } & { \| h _ { \ell _ { r } } ^ { \theta } ( x _ { f } ) - c \mathbf { u } \| _ { 2 } ^ { 2 } } \\ & { +  \alpha _ { r } \| h _ { \ell _ { r } } ^ { \theta } ( x _ { r } ) - h _ { \ell _ { r } } ^ { \mathrm { r e f } } ( x _ { r } ) \| _ { 2 } ^ { 2 } } \end{array}
$$

where $\mathbf { u } \ \sim \ { \mathcal { N } } ( 0 , I )$ is sampled once and fixed, $c = 2 0 , \ell _ { r } = 7$ (approximately the middle of the network for 7–8B models), and $\alpha _ { r } = 1 2 0 0 .$ . The random vector u is created in bfloat16 to match the model’s parameter dtype.

WHP (WhoIsHarryPotter). Following Eldan and Russinovich, WHP trains the model to redirect concept-specific queries to anonymised answers, rather than maximising the forget loss. Forget pairs use the original question as input and the anonymised answer as the training target; the retain pairs use the standard question–answer format. The combined loss is:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { W H P } } = \mathcal { L } _ { \mathrm { C E } } ( \theta ; q _ { f } , a _ { f } ^ { \mathrm { a n o n } } ) + \alpha _ { r } \mathcal { L } _ { \mathrm { C E } } ( \theta ; \mathcal { D } _ { \mathrm { r e t a i n } } ) , } \end{array}
$$

with $\alpha _ { r } ~ = ~ 1 . 0$ No frozen reference model is required (unlike the KL-based retain regularisers in NPO+KL and the original WHP paper), which avoids the GPU memory fragmentation caused by pinning a second full model copy to a fixed GPU. Because WHP redirects rather than erases the concept representation, it does not satisfy the standard forget ROUGE-L verification criterion (postunlearn forget ROUGE-L rises rather than falls); we annotate WHP runs with a ↔ symbol.

## B.6 Attack Implementation Details

We evaluate four adversarial recovery attacks after unlearning. All attacks use seed 42 and operate on the post-unlearning checkpoint held in GPU memory immediately after training (the model is not reloaded between attacks). Attacks are run in the order Steering → ICL → MIA → GCG; GCG runs last because the nanogcg optimiser can corrupt the CUDA context on some GPU configurations, after which the model is freed.

Activation Steering. Following Seyitoglu et al.˘ (2024), we compute a query-specific steering vector for each evaluation question $Q _ { i } \colon$

$$
S _ { L } ( Q _ { i } ) = h _ { L } ( Q _ { i } ) - \frac { 1 } { 3 } \sum _ { k = 1 } ^ { 3 } h _ { L } \Bigl ( Q _ { i } ^ { * ( k ) } \Bigr ) ,
$$

where $h _ { L } ( \cdot )$ is the hidden state at layer $L \ =$ $n _ { \mathrm { l a y e r s } } \mathrm { ~ - ~ 2 ~ }$ (the second-to-last transformer layer, consistent with the paper’s “just before the final layer” specification), extracted at the first token position. The steering vector is the raw difference (not unit-normalised), so the scale carries magnitude information.

The vector $S _ { L } ( Q _ { i } )$ is injected at the first decode step only (when the key-value cache has been filled by the prompt and the model generates the first new token, i.e. the hidden-state tensor has sequence length 1), scaled by coefficient $c = 2 . 0$ Crucially, the vector is not applied during the prefill phase (sequence length $> 1 )$ , which would modify the KV cache and corrupt attention rather than steering generation. A stateful hook flag ensures the perturbation fires exactly once per generation call.

For each of 25 evaluation pairs, we generate $ { n _ { \mathrm { ~  ~ } } } = \ 5 0 0$ stochastic completions (temperature $= 2 . 0 , \mathsf { t o p } { \cdot } k = 4 0 , \mathsf { m a x \_ n e w \_ t o k e n } s = 1 0 )$ both with and without the steering hook. The score for each pair is the maximum word-frequency of any answer token across the 500 samples (wordfrequency score, following the paper’s scoring protocol). The overall metric is the ROC AUC computed by treating steered scores as the positive class and unsteered scores as the negative class: AUC $> 0 . 5$ indicates that steering recovers more correctanswer tokens than no-steering, i.e. the concept geometry is partially intact.

In-Context Learning (ICL). ICL tests whether presenting the model with $n _ { \mathrm { s h o t s } } ~ = ~ 5$ conceptspecific examples as context is sufficient to recover the forgotten knowledge via in-context prompting.

From the available QA pairs, 25 pairs are held out for evaluation and the remainder form the demonstration pool. For each evaluation question, 5 demonstrations are sampled uniformly (excluding the current question) and prepended in the format "{Q}\nAnswer: $\{ \mathsf { A } \} \backslash \mathsf { n } \backslash \mathsf { n } ^ { \prime \prime }$ . Responses are generated greedily (max\_new\_tokens = 50). A response is marked correct when ROUGE-L with the reference answer is $\geq 0 . 3 0$ (soft-match threshold to account for answer paraphrasing). The attack score is the ICL accuracy; a higher score indicates that the model’s knowledge is still recoverable via context.

GCG Adversarial Suffix. We use the Greedy Coordinate Gradient (GCG) method of Zou et al. (2023) via the nanogcg Python library. To reduce computational cost while retaining sensitivity, we evaluate on the $n _ { \mathrm { e v a l } } = 5$ pairs with the shortest reference answers (easiest targets for gradient-based suffix optimisation).

Each suffix is initialised as 20 repeated "! 11 tokens and optimised for 500 gradient steps against the first 8 tokens of the reference answer (a short target prefix converges substantially faster than the full answer string and corresponds to the affirmative-prefix strategy used in the original paper). The model’s chat template is applied automatically by nanogcg before generation. After optimisation, the adversarial input is constructed as $\{ { \mathsf { q u e s t i o n } } \} ~ + ~ ^ { \prime \prime } ~ ^ { \prime \prime } ~ +$ best\_suffix and decoded greedily (max\_new\_tokens = 256). A pair is marked successful if the generated text contains a keyword from the reference answer or achieves $\mathtt { R O U G E - L } \ge 0 . 2 5$ . The attack score is the attack success rate (ASR) across the 5 pairs.

Membership Inference Attack (MIA). We employ a standard likelihood-ratio MIA: the model’s per-token conditional log-likelihood $s ( q , a ) \ =$ − log $P _ { \theta } ( a \mid q )$ is used as a membership score.

We score $n _ { \mathrm { f o r g e t } } = 5 0$ forget-concept QA pairs and $n _ { \mathrm { r e t a i n } } = 5 0$ retain-set pairs (sampled from pairs about unrelated topics). QA pairs with factual short-answer format are preferred over Wikipediatext pairs for the forget set, as gradient-ascent unlearning on Wikipedia-style text can produce degenerate NLL values exceeding 200 nats for dateprefixed completions. The ROC AUC is computed with forget-set pairs as the positive class (higher −log P = lower likelihood = model has successfully forgotten) and retain-set pairs as the negative class. $\mathrm { A U C } > 0 . 5$ indicates the model still assigns higher probability to the forget set than the retain set, i.e. unlearning is incomplete at the distribution level.

## C Additional Evaluated Metrics

This section details the supplementary geometric and baseline metrics evaluated in our study, expanding upon the primary metrics introduced in Section 3.1.

Baseline pair-wise metrics. These metrics apply standard geometric measures directly to the raw hidden states and serve as comparison baselines for the anon-difference metrics.

Reg $S _ { 2 } -$ Gram-matrix entropy of the regular hidden states:

$$
\mathrm { R e g ~ } S _ { 2 } = S _ { 2 } ( { \bf H } ) .
$$

Reg ${ \bar { L } } _ { 2 }$ — Mean $L _ { 2 }$ norm of regular hidden states:

$$
\operatorname { R e g } \bar { L } _ { 2 } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \lVert h _ { \ell ^ { * } } ( q _ { i } ) \rVert _ { 2 } .
$$

Reg cos — Mean pairwise cosine similarity of regular hidden states:

$$
\mathrm { R e g } \overline { { \cos } } = \frac { 1 } { N ( N - 1 ) } \sum _ { i \neq j } \cos \bigl ( h _ { \ell ^ { * } } ( q _ { i } ) , h _ { \ell ^ { * } } ( q _ { j } ) \bigr ) .
$$

Additional anon-difference pair-wise metrics. AnonDiff $\bar { L } _ { 2 } - \mathbf { M e a n } L _ { 2 }$ norm of anon-difference vectors:

$$
\mathrm { A n o n D i f f } \ \bar { L } _ { 2 } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \| \Delta v _ { i } \| _ { 2 } .
$$

AnonDiff cos — Mean pairwise cosine similarity of anon-difference vectors:

$$
\mathrm { A n o n D i f f } \overline { { \cos } } = \frac { 1 } { N ( N - 1 ) } \sum _ { i \neq j } \cos ( \Delta v _ { i } , \Delta v _ { j } ) .
$$

Additional batch-mean (AUSS) metrics. $\mathbf { A U S S } _ { \mathrm { c o s } } \uparrow -$ Complement of the mean pairwise cosine similarity between batch-mean directions (the linear analogue of $\mathrm { \ A U S S _ { c o s ^ { 2 } } ) }$

$$
\mathrm { A U S S } _ { \mathrm { c o s } } = 1 - \frac { 1 } { B ( B - 1 ) } \sum _ { b \neq b ^ { \prime } } \hat { d } _ { b } \cdot \hat { d } _ { b ^ { \prime } } .
$$

Unlike $\mathrm { \bf A U S S _ { \mathrm { c o s ^ { 2 } } } }$ , this is sensitive to the $s i g n$ of alignment; it equals zero only when all $B$ directions are perfectly parallel (fully consistent concept direction).

$\mathbf { A U S S } _ { \mathrm { J a c } } \uparrow -$ Mean pairwise soft Jaccard incoherence of the positive-component supports of the batch-mean directions. For unit vector $\hat { d } _ { b }$ , define its positive-activation profile $p _ { b } ~ \in ~ [ 0 , 1 ] ^ { d }$ with $p _ { b , j } = \operatorname* { m a x } ( \hat { d } _ { b , j } , 0 )$ . The pairwise soft Jaccard distance is

$$
J ( \hat { d } _ { b } , \hat { d } _ { b ^ { \prime } } ) = 1 - \frac { \sum _ { j } \operatorname* { m i n } ( p _ { b , j } , p _ { b ^ { \prime } , j } ) } { \sum _ { j } \operatorname* { m a x } ( p _ { b , j } , p _ { b ^ { \prime } , j } ) } ,
$$

and

$$
\mathrm { A U S S } _ { \mathrm { J a c } } = \frac { 1 } { B ( B - 1 ) } \sum _ { b \neq b ^ { \prime } } J ( \hat { d } _ { b } , \hat { d } _ { b ^ { \prime } } ) .
$$

This measures whether the sparsity patterns of the batch directions agree, independent of sign or magnitude effects captured by cosine metrics.

Covariance-based metrics. Let $\begin{array} { r l r l } { \overline { { \Delta v } } } & { { } } & { = } \end{array}$ $\begin{array} { r } { \frac { 1 } { N } \sum _ { i } \Delta v _ { i } } \end{array}$ be the mean anon-difference vector and $\begin{array} { r } { \hat { \dot { \Sigma } } = \frac { 1 } { N - 1 } \sum _ { i } ( \Delta v _ { i } - \overline { { \Delta v } } ) ( \Delta v _ { i } - \overline { { \Delta v } } ) ^ { \top } \in \mathbb R ^ { d \times d } } \end{array}$ be the sample covariance matrix. $\mathbf { C o v } _ { \mathrm { t r a c e } } \downarrow$ — Trace of the sample covariance of the anon-difference vectors:

$$
\mathrm { C o v } _ { \mathrm { t r a c e } } = \mathrm { T r } ( \hat { \Sigma } ) .
$$

A large trace indicates concept signal is dispersed across many directions (fragmented); a small value indicates compact, concentrated representations.

$\mathbf { C o v } _ { \mathrm { t o p } } \uparrow$ — Largest eigenvalue of the sample covariance:

$$
\mathrm { C o v } _ { \mathrm { t o p } } = \lambda _ { \mathrm { m a x } } ( \hat { \Sigma } ) .
$$

This captures how much variance concentrates along the dominant concept direction. After unlearning, a large $\operatorname { C o v } _ { \mathrm { t o p } }$ indicates the primary concept direction is intact in the representation space, predicting susceptibility to context-based recovery such as ICL.

SecMomen $\mathbf { \delta t _ { \mathrm { t o p } } } \uparrow - \mathrm { I }$ argest eigenvalue of the (uncentred) second moment matrix:

$$
\mathrm { S e c M o m e n t } _ { \mathrm { t o p } } = \lambda _ { \mathrm { m a x } } \left( \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \Delta v _ { i } \Delta v _ { i } ^ { \top } \right) .
$$

Unlike $\mathrm { C o v } _ { \mathrm { t o p } } ,$ this is sensitive to both the mean direction $\overline { { \Delta v } }$ and the residual spread, making it a joint measure of magnitude and directionality.

## D Full Experimental Results

## D.1 RQ1: Per-Model Fragmentation and Entropy Values

See the full results in Table 5 and Table 6.

## D.2 RQ1: Multi-Domain Validation and Classifier Discriminability

Four-way domain comparison. To test whether the HP-vs-TOFU separation reflects a structural known/unknown difference rather than an artifact of one narrow contrast, we extend the comparison to Harry Potter (retain split) and Star Wars, each tested against the same TOFU baseline across all 9 models (fresh, direction-corrected extraction), with two-sided Wilcoxon signed-rank tests and BH/FDR correction applied jointly across all 24 (metric, pair) tests.

Both additional “known” domains (HP-retain and Star Wars) separate from TOFU on every metric, in the same sign as the HP-forget contrast; HPretain effect sizes are, if anything, larger (up to d=6.20) than the HP-forget contrast itself.

Classifier-based discriminability. The discriminability analysis in Table 1 reports ∆/CI, which depends on the scale of the underlying metric. To confirm the separation is not an artifact of that scale, we build one sample per (model, domain) pair from the 8-dimensional standardized AUSS peak-layer feature vector and train Logistic Regression and a small MLP (8 hidden units) under leave-one-modelout cross-validation, so both domains of a held-out model are always scored by a classifier that never saw that model.

Both classifiers achieve near-perfect discriminability, with zero per-model misclassification for Logistic Regression, confirming the known/unknown separation is a robust structural feature of the representations rather than an artifact of the metrics’ raw scale.

## D.3 RQ2: Full Geometry vs ASR Correlation Table

See the full results at Tables 13–15 and Table 9.

## D.4 Head-to-Head Against Xu et al. (2026)

Full 44-test comparison (8 AUSS metrics + 3 Xu et al. metrics × 4 attacks) underlying the maintext summary in Section 4, BH-corrected globally across all 44 tests, $\scriptstyle n = 5 3 - 5 4$ . AUSS values come from checkpoints retrained specifically for this comparison (the originals were already deleted at the time this analysis was run), paired with the original checkpoints’ attack scores; this table should be read as an independent second data point rather than checkpoint-consistent with Table 9.

## D.5 Layerwise Geometry Analysis

Methodology. Following established practice in mechanistic interpretability (e.g., causal tracing; Meng et al. 2023), we treat this layer-resolved sweep as an exploratory diagnostic map of where different adversarial mechanisms exploit the geometry, rather than a multiplicity-corrected hypothesis test for a new global predictor; the corrected significance counts below quantify how much of the raw signal survives once treated as the latter. We extend the correlation analysis of Section 4 to every transformer layer. For each run we extract hidden states at all $L$ layers in a single forward pass with output\_hidden\_states=True (meanpooled over the response tokens), giving vectors $\mathbf { \bar { h } } _ { \mathrm { r e g } } ^ { ( \ell ) }$ and $\mathbf { h } _ { \mathrm { a n o n } } ^ { ( \ell ) }$ at each layer $\ell \in \{ 0 , \ldots , L - 1 \}$ . We compute Spearman $\rho$ between each (metric, layer) pair and each attack-success score across all runs with complete data. Two metric families are evaluated in parallel:

• AUSS metrics (8 metrics, defined in Section 3): Centroid\_Norm, $\mathsf { A U S S } _ { \mathrm { L 2 } }$ $\mathrm { \ A U S S _ { c o s ^ { 2 } } }$ $\mathrm { \ A U S S _ { J a c } }$ , Global\_Coherence, Batch Covariance, etc.

• Entropy / AnonDiff metrics (14 metrics): Gram-matrix Rényi-α entropy of the regular (Reg), concept-differential (AnonDiff), and row-normalised concept-differential (NormAnonDiff) hidden-state matrices, plus mean cosine and mean $L _ { 2 }$ statistics.

Table 5: Fragmentation metrics (HP known vs. TOFU unknown) for all ten LLMs at peak layer $\ell ^ { \ast } { = } 1 8 . \ \mathrm { Q A _ { H P } } { : }$ QA accuracy on HP; TOFU QA is 0.00 for every model. $\| { \overline { { \Delta { \hat { v } } } } } \| \uparrow \colon$ centroid norm (coherence; ↑ = model knows). $\mathrm { A U S S _ { L 2 } } \downarrow / \mathrm { A U S S _ { c o s ^ { 2 } } } \downarrow$ : dispersion (↓ = model knows). Bold: HP entry in the theoretically expected direction.
<table><tr><td rowspan="2">Model</td><td colspan="4">Harry Potter (known)</td><td colspan="3">TOFU (unknown,  $\overline { { \mathrm { Q A } = 0 . 0 0 ) } }$ </td></tr><tr><td> $\overline { { \mathbf { Q } \mathbf { A } _ { \mathrm { H P } } } }$ </td><td>||∆û||↑</td><td>AUSSL2↓</td><td> $\overline { { \mathrm { A U S S } _ { \mathrm { c o s } ^ { 2 } \downarrow } } }$ </td><td>||∆⁰||</td><td>AUSSL2</td><td> $\mathrm { \bf A U S S _ { c o s ^ { 2 } } }$ </td></tr><tr><td colspan="7">Qwen family</td></tr><tr><td>Qwen-7B-Chat</td><td>0.083</td><td>0.752</td><td>0.952</td><td>0.577</td><td>0.639</td><td>1.309</td><td>0.683</td></tr><tr><td>Qwen2.5-7B</td><td>0.250</td><td>0.684</td><td>1.150</td><td>0.501</td><td>0.601</td><td>1.402</td><td>0.408</td></tr><tr><td>Qwen-14B-Chat</td><td>0.135</td><td>0.804</td><td>0.780</td><td>0.553</td><td>0.688</td><td>1.164</td><td>0.696</td></tr><tr><td>Qwen2.5-14B</td><td>0.323</td><td>0.697</td><td>1.116</td><td>0.522</td><td>0.689</td><td>1.151</td><td>0.440</td></tr><tr><td colspan="8">Gemma family</td></tr><tr><td>Gemma-2B</td><td>0.042</td><td>0.949</td><td>0.223</td><td>0.209</td><td>0.638</td><td>1.316</td><td>0.835</td></tr><tr><td>Gemma-7B</td><td>0.083</td><td>0.882</td><td>0.491</td><td>0.417</td><td>0.773</td><td>0.891</td><td>0.608</td></tr><tr><td colspan="8">LLaMA family</td></tr><tr><td>LLaMA-2-7B</td><td>0.125</td><td>0.800</td><td>0.800</td><td>0.549</td><td>0.553</td><td>1.543</td><td>0.792</td></tr><tr><td>LLaMA-2-13B</td><td>0.219</td><td>0.903</td><td>0.411</td><td>0.363</td><td>0.595</td><td>1.437</td><td>0.891</td></tr><tr><td>LLaMA-3-8B</td><td>0.312</td><td>0.900</td><td>0.423</td><td>0.370</td><td>0.655</td><td>1.268</td><td>0.826</td></tr><tr><td>LLaMA-3.1-8B</td><td>0.219</td><td>0.903</td><td>0.409</td><td>0.360</td><td>0.624</td><td>1.356</td><td>0.850</td></tr></table>

Table 6: Gram-matrix entropy $( S _ { 2 } ,$ Rényi order 2) and companion statistics at per-metric peak layer $\ell ^ { * }$ for ten LLMs. Reg: raw hidden states; AnonDiff: difference vectors $\Delta V = h _ { \mathrm { r e g } } - h _ { \mathrm { a n o n } } ;$ applying Gram entropy to $\Delta V$ is our adaptation of Skean et al. (2025). $S _ { 2 \downarrow }$ : lower = more compressed = model knows concept. $\bar { L } _ { 2 } \uparrow ,$ , cos↑: higher = stronger/more coherent representation. Bold: HP entry consistent across all 10 models. Reg ${ { \bar { L } } _ { 2 } }$ omitted $( p _ { \mathrm { a d j } } { = } 0 . 4 3$ mixed direction).
<table><tr><td rowspan="3">Model</td><td colspan="4">Harry Potter (known)</td><td colspan="4">TOFU (unknown, QA = 0.00)</td></tr><tr><td colspan="2">Reg</td><td colspan="2">AnonDiff</td><td colspan="2"> $\overline { { \mathrm { R e g } } }$ </td><td colspan="2">AnonDiff</td></tr><tr><td> $\overline { { S _ { 2 } } }$  cOS</td><td> $\overline { { S _ { 2 } } }$ </td><td> $\overline { { L _ { 2 } } }$  COS</td><td> $\overline { { S _ { 2 } } }$ </td><td> $\overline { { \mathrm { c o s } } }$ </td><td> $\overline { { S _ { 2 } } }$ </td><td> $\overline { { L _ { 2 } } }$ </td><td>cOs</td></tr><tr><td colspan="9">Qwen family</td></tr><tr><td>Qwen-7B-Chat</td><td>0.007 0.995</td><td></td><td></td><td>3.808 20.4 0.138</td><td>0.043</td><td>0.969</td><td>4.769</td><td>16.9</td><td>0.098</td></tr><tr><td>Qwen2.5-7B</td><td>0.002 0.999</td><td></td><td>3.544 21.5 0.166</td><td>3.715 29.5 0.118</td><td></td><td>0.009 0.996 0.0900.954</td><td>4.614</td><td>24.7</td><td>0.121</td></tr><tr><td>Qwen-14B-Chat Qwen2.5-14B</td><td>0.018 0.990 0.002</td><td>0.999</td><td>3.825 37.2 0.119</td><td></td><td>0.007</td><td>0.997</td><td>4.795 4.743</td><td>16.7 30.8</td><td>0.088 0.132</td></tr><tr><td colspan="8">Gemma family</td><td></td></tr><tr><td>Gemma-2B</td><td>0.327 0.867</td><td></td><td>3.584 15.4 0.328</td><td></td><td></td><td>0.757 0.845</td><td>4.098</td><td>12.7</td><td>0.057</td></tr><tr><td>Gemma-7B</td><td>0.053 0.973</td><td></td><td>3.758 2.43 0.254</td><td></td><td></td><td>0.2000.895</td><td>4.397</td><td>1.79</td><td>0.108</td></tr><tr><td colspan="10">LLaMA family</td></tr><tr><td>LLaMA-2-7B</td><td>0.007 0.996</td><td></td><td>3.813 6.33 0.183</td><td></td><td></td><td>0.032 0.975</td><td></td><td>4.543 5.36 0.048</td><td></td></tr><tr><td>LLaMA-2-13B</td><td>0.055 0.967</td><td></td><td></td><td>1.635 6.90 0.275</td><td></td><td>0.221 0.868</td><td></td><td>2.106 5.16 0.044</td><td></td></tr><tr><td>LLaMA-3-8B</td><td>0.045 0.973</td><td></td><td></td><td>3.755 1.95 0.263</td><td></td><td>0.197 0.893</td><td></td><td>4.580 1.57 0.072</td><td></td></tr><tr><td>LLaMA-3.1-8B</td><td>0.038 0.977</td><td></td><td>3.714 2.41 0.265</td><td></td><td></td><td>0.168 0.907</td><td></td><td>4.543 2.06 0.064</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

1. GCG and ICL are the only two attacks whose layer-specific signal survives correction. Both peaks fall late in the network and one layer apart (GCG at L29, ICL at L28);

Figure 4 shows the full $\rho \mathrm { - v s }$ .-layer sweep for all four attacks using the six leader metrics (three AUSS, three entropy). The GCG panel reproduces and extends Figure 3 from the main text.

Best layer-metric predictors per attack. Refer to Table 11.

Key findings.

GCG survives global correction $\scriptstyle \left( q = 0 . 0 4 6 \right)$ but not the stricter per-attack-column correction $( q { = } 0 . 0 5 7 )$ , while ICL survives both. See the main-text Discussion for the mechanistic reading.

2. MIA and Steering do not survive correction at either scope. Their best (layer, metric) cells are only nominally significant at the raw $p { < } 0 . 0 5$ level $( q _ { \mathrm { g l o b a l } } 2 0 . 2 2 $ for both); we report them descriptively rather than as reliable predictors.

3. The aggregate and layer-resolved analyses find GCG signal via different metric families. The aggregate correlations in Table 9 find GCG significant via the Reg (Gram-entropy) family (Reg cos, $\rho { = } { - } 0 . 5 1 ,$ );

![](images/aa5c6b3d2d29c001d72bbb1788c6af3c60427a981d37aedd88fcf0fcf77f1224.jpg)  
Figure 4: Spearman $\rho$ by layer for all four attacks, leader metrics. Solid lines: AUSS family $( \mathrm { A U S S _ { L 2 } } , \mathrm { A U S S _ { c o s ^ { 2 } } }$ $\mathrm { \ A U S S _ { J a c } ) }$ . Dashed lines: Entropy family (Reg $S _ { 2 } ,$ , AnonDiff cos, NormAnonDiff $S _ { 2 } )$ . Dotted red vertical line: attack-specific peak layer among the six plotted metrics (GCG→L29, ICL→L00 for this restricted set; see Table 11 for the full-metric-universe peaks used in the main text, GCG→L29, ICL→L28). GCG and ICL both concentrate their strongest signal late in the network; MIA and Steering show weaker, non-BH-significant layer-specific signal in this dataset. $n { = } 1 0 { - } 1 5$ runs per attack (complete data).

the layer-resolved sweep instead finds its strongest GCG signal via $\mathrm { \ A U S S _ { J a c } }$ at L29 $\scriptstyle ( \rho = - 0 . 8 5 )$ . Both are real but reflect different aspects of the geometry: the peak-layer AUSS metric (argmax over $\mathsf { A U S S } _ { \mathrm { L } 2 } )$ does not coincide with GCG’s own predictive layer.

4. Positive $\rho$ for entropy metrics on MIA and Steering. Where entropy-family metrics show their strongest (non-significant) signal, higher entropy correlates with higher attack success, directionally opposite to the AUSS findings on GCG/ICL, reflecting that entropy measures spread of the concept-differential subspace while coherence measures its alignment.

## D.6 Ranking Consistency of the Aggregate Correlations

To test whether the aggregate correlations in Table 9 are driven by one influential run, we compute leave-one-run-out (LOO) Spearman $\rho$ for each of the 8 AUSS metrics against each of the 4 attacks, restricted to verified unlearning runs $\scriptstyle ( n = 2 6 - 2 7 )$

Only ICL is both leave-one-out stable and statistically significant after BH correction (8/8 metrics stable, 7/8 significant). Steering and GCG are largely stable (8/8 and 7/8) but not significant; MIA is neither reliably stable nor significant. We therefore present the correlation findings within the context of the ICL attack, rather than as a generalized cross-attack ranking claim.

## D.7 RQ2: Full Per-Run Results

Table 7: Four-way domain comparison: Cohen’s d (p, BH-corrected) for each condition vs. TOFU, n=9 models. All 24 cells are BH-significant (q ∈ [0.0063, 0.0260]).
<table><tr><td rowspan=1 colspan=1>Metric</td><td rowspan=1 colspan=1>vs. TOFU</td><td rowspan=1 colspan=1>d (p)</td></tr><tr><td rowspan=1 colspan=1>∥∆|/Global_Coh.</td><td rowspan=1 colspan=1>HP-forgetHP-retainStar Wars</td><td rowspan=1 colspan=1>+1.58 (.0039)+3.13 (.0039)+1.69 (.0039)</td></tr><tr><td rowspan=1 colspan=1>Batch_SecondMoment_TopEig</td><td rowspan=1 colspan=1>HP-forgetHP-retainStar Wars</td><td rowspan=1 colspan=1>+2.57 (.0039)+6.20 (.0039)+4.62 (.0039)</td></tr><tr><td rowspan=1 colspan=1> $\mathrm { \bf A U S S } _ { \mathrm { L } 2 }$ / Batch_Cov_Trace</td><td rowspan=1 colspan=1>HP-forgetHP-retainStar Wars</td><td rowspan=1 colspan=1>-1.55 (.0039)-3.35 (.0039)−1.43 (.0117)</td></tr><tr><td rowspan=1 colspan=1> $\mathrm { \ A U S S _ { c o s ^ { 2 } } }$ </td><td rowspan=1 colspan=1>HP-forgetHP-retainStar Wars</td><td rowspan=1 colspan=1>-0.92 (.0195)-1.59 (.0039)−2.00 (.0039)</td></tr><tr><td rowspan=1 colspan=1> $\mathrm { \ A U S S _ { J a c } }$ </td><td rowspan=1 colspan=1>HP-forgetHP-retainStar Wars</td><td rowspan=1 colspan=1>-1.87 (.0039)-2.09 (.0039)−2.82 (.0039)</td></tr><tr><td rowspan=1 colspan=1>Batch_Cov_TopEig</td><td rowspan=1 colspan=1>HP-forgetHP-retainStar Wars</td><td rowspan=1 colspan=1>-1.34 (.0078)-2.00 (.0039)-1.36 (.0039)</td></tr></table>

<table><tr><td></td><td>Logistic Regr.</td><td>MLP (8 hidden)</td></tr><tr><td>Accuracy</td><td>100%</td><td>94.4%</td></tr><tr><td>AUC</td><td>1.00</td><td>1.00</td></tr><tr><td>Correct</td><td>18/18</td><td>17/18</td></tr><tr><td>Models fully correct</td><td>9/9</td><td>8/9</td></tr></table>

Table 8: Classification performance on known (HP) vs. unknown (TOFU) domains using standardized 8- dimensional AUSS features, LOMO-CV, n=9 models (18 samples).

Table 9: Spearman $\rho$ between all fragmentation metrics and attack-success rate across n=44–45 post-unlearning runs (excluding Base and WHP). ${ ^ { * * } } / { ^ { * } } / { ^ { \dagger } }$ : BH-adjusted $p < 0 . 0 1 / 0 . 0 5 / 0 . 1 0$ within each attack column (9 metrics). <sup>⋆</sup>: adaptation of Gram-matrix entropy to concept-differential vectors (Skean et al., 2025). Bold: survives BH at $p < 0 . 0 5$ . Only bold, starred cells $( q { < } 0 . 0 5 )$ represent statistically robust findings; unmarked values are shown for completeness so the full 9-metric × 4-attack grid is auditable, and should not be read as evidence of a relationship. None of MIA’s AUSS/AnonDiff correlations survive BH correction; see Section 4.
<table><tr><td rowspan=1 colspan=1>Metric</td><td rowspan=1 colspan=1>Steering</td><td rowspan=1 colspan=1>ICL</td><td rowspan=1 colspan=1>GCG</td><td rowspan=1 colspan=1>MIA</td></tr><tr><td rowspan=1 colspan=5>AUSS dispersion</td></tr><tr><td rowspan=1 colspan=1>AUSSL2 $\mathrm { \ A U S S _ { c o s ^ { 2 } } }$  $\| \overline { { \Delta \hat { v } } } \|$ </td><td rowspan=1 colspan=1>+0.00+0.06-0.00</td><td rowspan=1 colspan=1> $\mathbf { \overline { { { 0 . 5 1 } } } ^ { \ast \ast } }$  $\mathbf { - 0 . 5 0 ^ { * * } }$ +0.51**</td><td rowspan=1 colspan=1>-0.06-0.01+0.06</td><td rowspan=1 colspan=1>+0.04+0.10-0.04</td></tr><tr><td rowspan=1 colspan=3>AnonDiff</td><td rowspan=1 colspan=2></td></tr><tr><td rowspan=1 colspan=1>AnonDiff cosAnonDiff S2AnonDiff ${ \bar { L } } _ { 2 }$ </td><td rowspan=1 colspan=1>-0.24+0.48**+0.15</td><td rowspan=1 colspan=1>+0.27-0.22+0.04</td><td rowspan=1 colspan=1>-0.09+0.22-0.05</td><td rowspan=1 colspan=1>-0.15+0.20-0.07</td></tr><tr><td rowspan=1 colspan=3>Reg (raw-activation baseline)</td><td rowspan=1 colspan=2></td></tr><tr><td rowspan=1 colspan=1>Reg S2Reg cosReg $L _ { 2 }$ </td><td rowspan=1 colspan=1>+0.27-0.31-0.07</td><td rowspan=1 colspan=1>+0.05+0.01-0.09</td><td rowspan=1 colspan=1>+0.45**-0.51**-0.24</td><td rowspan=1 colspan=1>+0.29-0.27-0.16</td></tr></table>

Table 10: Full Xu et al. head-to-head comparison, sorted by ${ q _ { \mathrm { g l o b a l } } . \ ^ { * * } / ^ { * } } ; \ q { < } 0 . 0 1 / 0 . 0 5$
<table><tr><td rowspan=1 colspan=1>Family</td><td rowspan=1 colspan=4>Metric</td><td rowspan=1 colspan=1>Attack</td><td rowspan=1 colspan=1>ρ</td><td rowspan=1 colspan=1>p</td><td rowspan=1 colspan=1>qglobal</td></tr><tr><td rowspan=1 colspan=1>AUSS</td><td rowspan=2 colspan=4>Centroid_Norm $\mathrm { \ A U S S _ { c o s ^ { 2 } } }$ </td><td rowspan=1 colspan=1>ICL</td><td rowspan=1 colspan=1>+0.551</td><td rowspan=1 colspan=1>0.00002</td><td rowspan=1 colspan=1> $\overline { { 0 . 0 0 0 1 } } ^ { * * }$ </td></tr><tr><td rowspan=1 colspan=1>AUSS</td><td rowspan=1 colspan=1>ICL</td><td rowspan=1 colspan=1>-0.572</td><td rowspan=1 colspan=1>0.00001</td><td rowspan=1 colspan=1> $0 . 0 0 0 1 ^ { ^ { * * } }$ </td></tr><tr><td rowspan=1 colspan=1>AUSS</td><td rowspan=1 colspan=4> $\mathrm { A U S S } _ { L 2 }$ </td><td rowspan=1 colspan=1>ICL</td><td rowspan=1 colspan=1>-0.551</td><td rowspan=1 colspan=1>0.00002</td><td rowspan=1 colspan=1> $0 . 0 0 0 1 ^ { ^ { \ast * } }$ </td></tr><tr><td rowspan=1 colspan=1>AUSS</td><td rowspan=1 colspan=4>Global_Coherence</td><td rowspan=1 colspan=1>ICL</td><td rowspan=1 colspan=1>+0.551</td><td rowspan=1 colspan=1>0.00002</td><td rowspan=1 colspan=1> $0 . 0 0 0 1 ^ { ^ { \ast * } }$ </td></tr><tr><td rowspan=1 colspan=1>AUSS</td><td rowspan=1 colspan=4>Batch_Cov_Trace</td><td rowspan=1 colspan=1>ICL</td><td rowspan=1 colspan=1>-0.551</td><td rowspan=1 colspan=1>0.00002</td><td rowspan=1 colspan=1> $0 . 0 0 0 1 ^ { ^ { * * } }$ </td></tr><tr><td rowspan=1 colspan=1>AUSS</td><td rowspan=1 colspan=4>Batch_SecMom_TopEig</td><td rowspan=1 colspan=1>ICL</td><td rowspan=1 colspan=1>+0.557</td><td rowspan=1 colspan=1>0.00001</td><td rowspan=1 colspan=1>*0.0001</td></tr><tr><td rowspan=1 colspan=1>AUSS</td><td rowspan=1 colspan=4>AUSSJac</td><td rowspan=1 colspan=1>Steering</td><td rowspan=1 colspan=1>-0.452</td><td rowspan=1 colspan=1>0.00067</td><td rowspan=1 colspan=1> $0 . 0 0 4 2 ^ { ^ { * * } }$ </td></tr><tr><td rowspan=1 colspan=1>Xu et al.</td><td rowspan=1 colspan=4>cka_mean</td><td rowspan=1 colspan=1>Steering</td><td rowspan=1 colspan=1>+0.428</td><td rowspan=1 colspan=1>0.00137</td><td rowspan=1 colspan=1>0.0076**</td></tr><tr><td rowspan=1 colspan=1>Xu et al.</td><td rowspan=1 colspan=4>pca_sim_mean</td><td rowspan=1 colspan=1>ICL</td><td rowspan=1 colspan=1>+0.381</td><td rowspan=1 colspan=1>0.00485</td><td rowspan=1 colspan=1>0.0212*</td></tr><tr><td rowspan=1 colspan=1>AUSS</td><td rowspan=1 colspan=4>Batch_Cov_TopEig</td><td rowspan=1 colspan=1>ICL</td><td rowspan=1 colspan=1>-0.385</td><td rowspan=1 colspan=1>0.00440</td><td rowspan=1 colspan=1>0.0212*</td></tr><tr><td rowspan=1 colspan=1>Xu et al.</td><td rowspan=1 colspan=4>cka_mean</td><td rowspan=1 colspan=1>GCG</td><td rowspan=1 colspan=1>+0.374</td><td rowspan=1 colspan=1>0.00529</td><td rowspan=1 colspan=1>0.0212*</td></tr><tr><td rowspan=1 colspan=1>Xu et al.</td><td rowspan=1 colspan=4>cka_mean</td><td rowspan=1 colspan=1>MIA</td><td rowspan=1 colspan=1>+0.336</td><td rowspan=1 colspan=1>0.01378</td><td rowspan=1 colspan=1>0.0505</td></tr><tr><td rowspan=1 colspan=1>AUSS</td><td rowspan=1 colspan=4>Batch_Cov_TopEig</td><td rowspan=1 colspan=1>GCG</td><td rowspan=1 colspan=1>-0.315</td><td rowspan=1 colspan=1>0.02020</td><td rowspan=1 colspan=1>0.0684</td></tr><tr><td rowspan=1 colspan=1>Xu et al.</td><td rowspan=1 colspan=4>cka_mean</td><td rowspan=1 colspan=1>ICL</td><td rowspan=1 colspan=1>+0.310</td><td rowspan=1 colspan=1>0.02409</td><td rowspan=1 colspan=1>0.0757</td></tr><tr><td rowspan=1 colspan=1>AUSS</td><td rowspan=1 colspan=4>Batch_Cov_TopEig</td><td rowspan=1 colspan=1>Steering</td><td rowspan=1 colspan=1>-0.287</td><td rowspan=1 colspan=1>0.03693</td><td rowspan=1 colspan=1>0.1083</td></tr><tr><td rowspan=1 colspan=1>Xu et al.</td><td rowspan=1 colspan=4>pca_sim_mean</td><td rowspan=1 colspan=1>MIA</td><td rowspan=1 colspan=1>+0.269</td><td rowspan=1 colspan=1>0.05126</td><td rowspan=1 colspan=1>0.1410</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td rowspan=1 colspan=1>GCG</td><td rowspan=1 colspan=1>+0.263</td><td rowspan=1 colspan=1>0.05497</td><td rowspan=1 colspan=1>0.1423</td></tr><tr><td rowspan=1 colspan=1>AUSS</td><td rowspan=1 colspan=4>Centroid_Norm</td><td rowspan=1 colspan=1>GCG</td><td rowspan=1 colspan=1>+0.218</td><td rowspan=1 colspan=1>0.11245</td><td rowspan=1 colspan=1>0.2262</td></tr><tr><td rowspan=1 colspan=1>AUSS</td><td rowspan=1 colspan=4>Batch_Cov_Trace</td><td rowspan=1 colspan=1>GCG</td><td rowspan=1 colspan=1>-0.218</td><td rowspan=1 colspan=1>0.11245</td><td rowspan=1 colspan=1>0.2262</td></tr><tr><td rowspan=1 colspan=1>AUSS</td><td rowspan=1 colspan=1>AU</td><td rowspan=1 colspan=3>AUSSJac</td><td rowspan=1 colspan=1>GCG</td><td rowspan=1 colspan=1>-0.218</td><td rowspan=1 colspan=1>0.11308</td><td rowspan=1 colspan=1>0.2262</td></tr><tr><td rowspan=1 colspan=1>AUSS</td><td rowspan=1 colspan=2>AUSS</td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>GCG</td><td rowspan=1 colspan=1>-0.218</td><td rowspan=1 colspan=1>0.11245</td><td rowspan=1 colspan=1>0.2262</td></tr><tr><td rowspan=1 colspan=1>Xu et al.</td><td rowspan=1 colspan=3>pca_shif</td><td rowspan=1 colspan=1>t_mean</td><td rowspan=1 colspan=1>MIA</td><td rowspan=1 colspan=1>+0.213</td><td rowspan=1 colspan=1>0.12546</td><td rowspan=1 colspan=1>0.2300</td></tr><tr><td rowspan=1 colspan=1>AUSS</td><td rowspan=1 colspan=4>Batch_SecMom_TopEig</td><td rowspan=1 colspan=1>GCG</td><td rowspan=1 colspan=1>+0.212</td><td rowspan=1 colspan=1>0.12336</td><td rowspan=1 colspan=1>0.2300</td></tr><tr><td rowspan=1 colspan=1>Xu et al.</td><td rowspan=1 colspan=4>pca_shift_mean</td><td rowspan=1 colspan=1>GCG</td><td rowspan=1 colspan=1>-0.205</td><td rowspan=1 colspan=1>0.13778</td><td rowspan=1 colspan=1>0.2425</td></tr><tr><td rowspan=1 colspan=1>AUSS</td><td rowspan=1 colspan=4>AUSScos2</td><td rowspan=1 colspan=1>GCG</td><td rowspan=1 colspan=1>-0.189</td><td rowspan=1 colspan=1>0.17119</td><td rowspan=1 colspan=1>0.2897</td></tr><tr><td rowspan=1 colspan=1>AUSS</td><td rowspan=1 colspan=4>Batch_Cov_Trace</td><td rowspan=1 colspan=1>Steering</td><td rowspan=1 colspan=1>-0.110</td><td rowspan=1 colspan=1>0.43311</td><td rowspan=1 colspan=1>0.5775</td></tr><tr><td rowspan=1 colspan=1>AUSS</td><td rowspan=1 colspan=4>Global_Coherence</td><td rowspan=1 colspan=1>Steering</td><td rowspan=1 colspan=1>+0.110</td><td rowspan=1 colspan=1>0.43311</td><td rowspan=1 colspan=1>0.5775</td></tr><tr><td rowspan=1 colspan=1>Xu et al.</td><td rowspan=1 colspan=4>pca_sim_mean</td><td rowspan=1 colspan=1>Steering</td><td rowspan=1 colspan=1>-0.114</td><td rowspan=1 colspan=1>0.41497</td><td rowspan=1 colspan=1>0.5775</td></tr><tr><td rowspan=1 colspan=1>Xu et al.</td><td rowspan=1 colspan=4>pca_shift_mean</td><td rowspan=1 colspan=1>ICL</td><td rowspan=1 colspan=1>+0.122</td><td rowspan=1 colspan=1>0.38399</td><td rowspan=1 colspan=1>0.5775</td></tr><tr><td rowspan=1 colspan=1>AUSS</td><td rowspan=1 colspan=4>Centroid_Norm</td><td rowspan=1 colspan=1>Steering</td><td rowspan=1 colspan=1>+0.110</td><td rowspan=1 colspan=1>0.43311</td><td rowspan=1 colspan=1>0.5775</td></tr><tr><td rowspan=1 colspan=1>AUSS</td><td rowspan=1 colspan=4>Batch_SecMom_TopEig</td><td rowspan=1 colspan=1>Steering</td><td rowspan=1 colspan=1>+0.111</td><td rowspan=1 colspan=1>0.42803</td><td rowspan=1 colspan=1>0.5775</td></tr><tr><td rowspan=1 colspan=1>AUSS</td><td rowspan=1 colspan=4>Batch_SecMom_TopEig</td><td rowspan=1 colspan=1>MIA</td><td rowspan=1 colspan=1>-0.079</td><td rowspan=1 colspan=1>0.57261</td><td rowspan=1 colspan=1>0.6460</td></tr><tr><td rowspan=1 colspan=1>AUSS</td><td rowspan=1 colspan=4>Batch_Cov_Trace</td><td rowspan=1 colspan=1>MIA</td><td rowspan=1 colspan=1>+0.080</td><td rowspan=1 colspan=1>0.56753</td><td rowspan=1 colspan=1>0.6460</td></tr><tr><td rowspan=1 colspan=1>AUSS</td><td rowspan=1 colspan=4>AUSScos2</td><td rowspan=1 colspan=1>MIA</td><td rowspan=1 colspan=1>+0.085</td><td rowspan=1 colspan=1>0.54492</td><td rowspan=1 colspan=1>0.6460</td></tr><tr><td rowspan=2 colspan=1>AUSSAUSS</td><td rowspan=2 colspan=4>Global_CoherenceAUSSL2</td><td rowspan=1 colspan=1>MIA</td><td rowspan=1 colspan=1>-0.080</td><td rowspan=1 colspan=1>0.56753</td><td rowspan=1 colspan=1>0.6460</td></tr><tr><td rowspan=1 colspan=1>MIA</td><td rowspan=1 colspan=1>+0.080</td><td rowspan=1 colspan=1>0.56753</td><td rowspan=1 colspan=1>0.6460</td></tr><tr><td rowspan=1 colspan=1>AUSS</td><td rowspan=1 colspan=4>Centroid_Norm</td><td rowspan=1 colspan=1>MIA</td><td rowspan=1 colspan=1>-0.080</td><td rowspan=1 colspan=1>0.56753</td><td rowspan=1 colspan=1>0.6460</td></tr><tr><td rowspan=1 colspan=1>AUSS</td><td rowspan=1 colspan=4>Batch_Cov_TopEig</td><td rowspan=1 colspan=1>MIA</td><td rowspan=1 colspan=1>+0.066</td><td rowspan=1 colspan=1>0.63949</td><td rowspan=1 colspan=1>0.7034</td></tr><tr><td rowspan=1 colspan=1>AUSS</td><td rowspan=1 colspan=4>AUSS.2</td><td rowspan=1 colspan=1>Steering</td><td rowspan=1 colspan=1>-0.054</td><td rowspan=1 colspan=1>0.69991</td><td rowspan=1 colspan=1>0.7511</td></tr><tr><td rowspan=1 colspan=1>AUSS</td><td rowspan=1 colspan=4> $\mathrm { \bf A U S S } _ { J a c }$ </td><td rowspan=1 colspan=1>ICL</td><td rowspan=1 colspan=1>-0.051</td><td rowspan=1 colspan=1>0.71931</td><td rowspan=1 colspan=1>0.7536</td></tr><tr><td rowspan=2 colspan=1>AUSSXu et al.</td><td rowspan=2 colspan=4> $\mathrm { \bf A U S S } _ { J a c }$ pca_shift_mean</td><td rowspan=1 colspan=1>MIA</td><td rowspan=1 colspan=1>+0.032</td><td rowspan=1 colspan=1>0.81909</td><td rowspan=1 colspan=1>0.8381</td></tr><tr><td rowspan=1 colspan=1>Steering</td><td rowspan=1 colspan=1>-0.011</td><td rowspan=1 colspan=1>0.93846</td><td rowspan=1 colspan=1>0.9385</td></tr></table>

Table 11: Best (layer, metric) predictor per attack in the layer-resolved analysis. n: number of runs with complete data at that layer. AUSS and entropy families are shown separately; the overall best metric across both is marked ⋆. $q _ { \mathrm { a t t a c k } } \mathrm { : }$ BH correction within the attack’s own column (8 AUSS + entropy tests); $q _ { \mathrm { g l o b a l } } \mathrm { . }$ : BH correction across all 4,224 (layer, metric, attack) tests at once. $^ { * } q { < } 0 . 0 5$
<table><tr><td>Attack</td><td>Family</td><td>Best layer</td><td>Best metric</td><td> $\underline { { \rho } }$ </td><td> $p$ </td><td> $\underline { { q _ { \mathrm { a t t a c k } } \mathrm { ~ / ~ } q _ { \mathrm { g l o b a l } } } }$ </td></tr><tr><td rowspan="2">GCG</td><td rowspan="2">AUSS Entropy</td><td>L29</td><td rowspan="2">AUSSJac NormAnonDiff</td><td>-0.853 -0.832</td><td>0.0008</td><td> $\overline { { 0 . 0 5 7 / 0 . 0 4 6 ^ { * } } }$ </td></tr><tr><td>L30</td><td> $S _ { 2 } \star$ </td><td>0.0015</td><td> $0 . 0 5 7 / 0 . 0 4 6 ^ { \ast }$ </td></tr><tr><td rowspan="2">ICL</td><td rowspan="2">AUSS Entropy</td><td>L28</td><td rowspan="2">Batch_Cov_TopEig * AnonDiff</td><td>-0.886 -0.861</td><td>&lt;0.0001</td><td> $\overline { { { 0 . 0 0 6 } ^ { * } / { 0 . 0 0 9 } ^ { * } } }$ </td></tr><tr><td>L00</td><td> $\bar { S } _ { 2 }$ </td><td>0.0002</td><td> $0 . 0 1 1 ^ { * } / 0 . 0 3 7 ^ { * }$ </td></tr><tr><td rowspan="2">MIA</td><td rowspan="2">AUSS Entropy</td><td>L31</td><td rowspan="2">Centroid_Norm Reg  ${ \bar { L } } _ { 2 } \star$ </td><td>-0.446</td><td>0.170</td><td>0.999 / 0.581</td></tr><tr><td>L32</td><td>+0.627</td><td>0.039</td><td>0.999 / 0.254</td></tr><tr><td rowspan="2">Steering</td><td rowspan="2">AUSS Entropy</td><td>L31</td><td rowspan="2"> $\overline { { | | \Delta \hat { v } | | } }$   $\mathrm { R e g } \overline { { \mathrm { c o s } } } \star$ </td><td>+0.536</td><td>0.089</td><td>0.987/0.401</td></tr><tr><td>L31</td><td>+0.645</td><td>0.032</td><td>0.987/0.224</td></tr></table>

Table 12: Leave-one-out sign-flip stability, 8 AUSS metrics per attack, n=26–27 verified runs.
<table><tr><td>Attack</td><td>Metrics with 0 LOO sign flips</td><td>BH-significant  $( q { < } 0 . 0 5 )$ </td></tr><tr><td rowspan="3">ICL Steering GCG</td><td>8/8</td><td>7/8</td></tr><tr><td>8/8</td><td>0/8</td></tr><tr><td>7/8</td><td>0/8</td></tr><tr><td>MIA</td><td>2/8</td><td>0/8</td></tr></table>

Table 13: RQ2: Full post-unlearning results, LLaMA-2-7B-chat. ROUGE-L↓: forget-set ROUGE-L after unlearning. $\checkmark :$ verified $( \mathrm { R O U G E - L } < 0 . 1 0$ & retain $\ge 0 . 8 0 \times$ base). $\| \overline { { \Delta \hat { v } } } \| \uparrow$ : centroid norm; $\mathrm { A U S S _ { L 2 } } \downarrow , \mathrm { A U S S _ { c o s ^ { 2 } } } \downarrow$ dispersion (per-method peak layer ℓ<sup>∗</sup>). Steer-AUC↓: steering AUC; Steer-WF↓: mean steered word-frequency score. ICL $\Delta \uparrow :$ ICL accuracy minus baseline. MIA↓: membership inference AUC; GCG↓: GCG ASR. Blank cells: checkpoint unavailable.
<table><tr><td colspan="2"></td><td rowspan="2"> $\checkmark$ </td><td>Forget</td><td colspan="3">Geometry  $\overline { { \ell ^ { * } } }$ </td><td colspan="5">Attack Success ↓</td></tr><tr><td>Method</td><td> $\mathrm { C p t }$ </td><td>ROUGE-L↓</td><td> $\| \overline { { \Delta \hat { v } \| \hat { \tau } } }$ </td><td> $\overline { { \mathrm { A U S S } _ { \mathrm { L 2 } \mathrm { \downarrow } } } }$ </td><td> $\overline { { \mathrm { A U S S } _ { \mathrm { c o s } ^ { 2 } \downarrow } } }$ </td><td>Steer-AUC</td><td>Steer-WF</td><td>ICL ∆</td><td>MIA</td><td>GCG</td></tr><tr><td rowspan="3">Base</td><td>HP</td><td></td><td>0.131</td><td>0.878</td><td>0.507</td><td>0.439</td><td>0.428</td><td>0.265</td><td>+0.000</td><td>0.827</td><td>0.000</td></tr><tr><td>SW</td><td></td><td>0.152</td><td>0.876</td><td>0.516</td><td>0.447</td><td>0.449</td><td>0.296</td><td>+0.120</td><td>0.955</td><td>0.000</td></tr><tr><td>WS</td><td></td><td>0.090</td><td>0.911</td><td>0.379</td><td>0.341</td><td>0.503</td><td>0.274</td><td>+0.040</td><td>0.837</td><td>0.000</td></tr><tr><td rowspan="3">GA</td><td>HP</td><td> $\checkmark$ </td><td>0.000</td><td>0.823</td><td>0.717</td><td>0.553</td><td>0.500</td><td>0.000</td><td>+0.000</td><td>0.678</td><td>0.000</td></tr><tr><td>SW</td><td> $\checkmark$ </td><td>0.000</td><td>0.875</td><td>0.521</td><td>0.442</td><td>0.500</td><td>0.000</td><td>+0.000</td><td>0.695</td><td>0.000</td></tr><tr><td>WS</td><td> $\checkmark$ </td><td>0.000</td><td>0.855</td><td>0.598</td><td>0.475</td><td>0.020</td><td>0.000</td><td>+0.000</td><td>0.615</td><td>0.000</td></tr><tr><td rowspan="3">DPO</td><td>HP</td><td> $\overline { { \checkmark } }$ </td><td>0.082</td><td>0.868</td><td>0.549</td><td>0.469</td><td>0.694</td><td>0.283</td><td>-0.040</td><td>0.873</td><td>0.800</td></tr><tr><td>SW</td><td> $\checkmark$ </td><td>0.098</td><td>0.866</td><td>0.554</td><td>0.474</td><td>0.673</td><td>0.281</td><td>+0.040</td><td>0.986</td><td>0.800</td></tr><tr><td>WS</td><td> $\checkmark$ </td><td>0.092</td><td>0.913</td><td>0.371</td><td>0.335</td><td>0.658</td><td>0.287</td><td>+0.080</td><td>0.920</td><td>1.000</td></tr><tr><td rowspan="3">NPO</td><td>HP</td><td></td><td>0.127</td><td>0.860</td><td>0.580</td><td>0.492</td><td>0.756</td><td>0.286</td><td>+0.000</td><td>0.805</td><td>0.600</td></tr><tr><td>SW</td><td></td><td>0.108</td><td>0.877</td><td>0.513</td><td>0.446</td><td>0.668</td><td>0.281</td><td>+0.040</td><td>0.937</td><td>1.000</td></tr><tr><td>WS</td><td> $\checkmark$ </td><td>0.092</td><td>0.922</td><td>0.333</td><td>0.303</td><td>0.795</td><td>0.295</td><td>+0.040</td><td>0.748</td><td>1.000</td></tr><tr><td rowspan="3">NPO+KL</td><td>HP</td><td> $\overline { { \checkmark } }$ </td><td>0.098</td><td>0.854</td><td>0.603</td><td>0.507</td><td>0.769</td><td>0.259</td><td>-0.040</td><td>0.449</td><td>0.600</td></tr><tr><td>SW</td><td></td><td>0.105</td><td>0.867</td><td>0.550</td><td>0.473</td><td>0.746</td><td>0.272</td><td>+0.000</td><td>0.281</td><td>0.600</td></tr><tr><td>WS</td><td> $\checkmark$ </td><td>0.091</td><td>0.917</td><td>0.354</td><td>0.320</td><td>0.823</td><td>0.286</td><td>+0.040</td><td>0.162</td><td>1.000</td></tr><tr><td rowspan="3">RMU</td><td>HP</td><td></td><td>0.131</td><td>0.878</td><td>0.509</td><td>0.441</td><td>0.710</td><td>0.264</td><td>+0.000</td><td>0.806</td><td>1.000</td></tr><tr><td>SW</td><td></td><td>0.143</td><td>0.875</td><td>0.519</td><td>0.449</td><td>0.694</td><td>0.300</td><td>+0.040</td><td>0.951</td><td>1.000</td></tr><tr><td>WS</td><td> $\checkmark$ </td><td>0.100</td><td>0.908</td><td>0.388</td><td>0.348</td><td>0.754</td><td>0.267</td><td>+0.080</td><td>0.787</td><td>1.000</td></tr><tr><td rowspan="3">WHP</td><td>HP</td><td></td><td>0.255</td><td>0.858</td><td>0.585</td><td>0.494</td><td>0.756</td><td>0.248</td><td>-0.040</td><td>0.442</td><td>0.000</td></tr><tr><td>SW</td><td></td><td>0.175</td><td>0.856</td><td>0.595</td><td>0.505</td><td>0.708</td><td>0.269</td><td>+0.000</td><td>0.276</td><td>0.000</td></tr><tr><td>WS</td><td></td><td>0.135</td><td>0.917</td><td>0.353</td><td>0.319</td><td>0.826</td><td>0.281</td><td>+0.040</td><td>0.181</td><td>0.000</td></tr></table>

Table 14: RQ2: Full post-unlearning results, Meta-LLaMA-3-8B-Instruct. ROUGE-L↓: forget-set ROUGE-L after unlearning. $\checkmark :$ verified $( \mathrm { R O U G E - L } < 0 . 1 0$ & retain $\ge 0 . 8 0 \times$ base). $\| { \overline { { \Delta { \hat { v } } } } } \| \uparrow \colon$ centroid norm; $\mathrm { \ A U S S _ { L 2 } } \downarrow$ $\mathrm { \ A U S S _ { c o s ^ { 2 } \downarrow } }$ : dispersion (per-method peak layer ℓ<sup>∗</sup>). Steer-AUC↓: steering AUC; Steer-WF↓: mean steered wordfrequency score. ICL $\Delta \uparrow :$ : ICL accuracy minus baseline. MIA↓: membership inference AUC; GCG↓: GCG ASR. Blank cells: checkpoint unavailable.
<table><tr><td rowspan="2">Method</td><td rowspan="2"> $\mathrm { C p t }$ </td><td rowspan="2"> $\checkmark$ </td><td>Forget</td><td colspan="3">Geometry l*</td><td colspan="5">Attack Success↓</td></tr><tr><td>ROUGE-L↓</td><td> $| \overline { { \Delta \hat { v } } } | | \ d t \vert ^ { }$ </td><td> $\overline { { \mathrm { A U S S } _ { \mathrm { L 2 } \mathrm { \downarrow } } } }$ </td><td> $\overline { { \mathrm { A U S S } _ { \mathrm { c o s } ^ { 2 } \downarrow } } }$ </td><td>Steer-AUC</td><td>Steer-WF</td><td>ICL  $\Delta$ </td><td>MIA</td><td>GCG</td></tr><tr><td rowspan="2">Base</td><td>HP</td><td></td><td>0.062</td><td>0.856</td><td>0.594</td><td>0.493</td><td>0.423</td><td>0.437</td><td>+0.000</td><td>0.915</td><td>0.000</td></tr><tr><td>SW</td><td></td><td>0.113</td><td>0.873</td><td>0.529</td><td>0.455</td><td>0.424</td><td>0.379</td><td>+0.040</td><td>0.987</td><td>0.000</td></tr><tr><td rowspan="3">GA</td><td>WS</td><td></td><td>0.068</td><td>0.901</td><td>0.416</td><td>0.370</td><td>0.425</td><td>0.347</td><td>+0.040</td><td>0.952</td><td>0.000</td></tr><tr><td>HP</td><td> $\overline { { \checkmark } }$ </td><td>0.000</td><td>0.872</td><td>0.534</td><td>0.439</td><td>0.500</td><td>0.000</td><td>+0.000</td><td>0.700</td><td>0.000</td></tr><tr><td>SW</td><td> $\checkmark$ </td><td>0.000</td><td>0.801</td><td>0.797</td><td>0.591</td><td>0.500</td><td>0.000</td><td>+0.000</td><td>0.102</td><td>0.000</td></tr><tr><td rowspan="3">DPO</td><td>WS</td><td> $\checkmark$ </td><td>0.000</td><td>0.814</td><td>0.750</td><td>0.436</td><td>0.500</td><td>0.000</td><td>+0.000</td><td>0.250</td><td>0.000</td></tr><tr><td>HP</td><td> $\overline { { \checkmark } }$ </td><td>0.000</td><td>0.867</td><td>0.551</td><td>0.466</td><td>0.720</td><td>0.076</td><td>+0.040</td><td>1.000</td><td>0.400</td></tr><tr><td>SW</td><td> $\checkmark$ </td><td>0.017</td><td>0.842</td><td>0.646</td><td>0.519</td><td>0.620</td><td>0.017</td><td>+0.000</td><td>0.998</td><td>0.800</td></tr><tr><td rowspan="3">NPO</td><td>WS</td><td> $\checkmark$ </td><td>0.000</td><td>0.923</td><td>0.328</td><td>0.298</td><td>0.700</td><td>0.037</td><td>+0.000</td><td>1.000</td><td>0.000</td></tr><tr><td>HP</td><td></td><td>0.112</td><td>0.863</td><td>0.566</td><td>0.482</td><td>0.930</td><td>0.493</td><td>+0.000</td><td>0.817</td><td>1.000</td></tr><tr><td>SW</td><td></td><td>0.167</td><td>0.871</td><td>0.535</td><td>0.462</td><td>0.838</td><td>0.404</td><td>-0.040</td><td>0.900</td><td>1.000</td></tr><tr><td rowspan="3">NPO+KL</td><td>WS</td><td> $\checkmark$ </td><td>0.096</td><td>0.923</td><td>0.328</td><td>0.298</td><td>0.779</td><td>0.356</td><td>+0.080</td><td>0.834</td><td>1.000</td></tr><tr><td>HP</td><td></td><td>0.178</td><td>0.867</td><td>0.552</td><td>0.472</td><td>0.945</td><td>0.488</td><td>-0.120</td><td>0.389</td><td>1.000</td></tr><tr><td>SW</td><td></td><td>0.208</td><td>0.847</td><td>0.627</td><td>0.525</td><td>0.882</td><td>0.393</td><td>-0.120</td><td>0.249</td><td>1.000</td></tr><tr><td rowspan="3">RMU</td><td>WS</td><td></td><td>0.121</td><td>0.921</td><td>0.335</td><td>0.305</td><td>0.822</td><td>0.313</td><td>+0.080</td><td>0.157</td><td>1.000</td></tr><tr><td>HP</td><td> $\overline { { \checkmark } }$ </td><td>0.063</td><td>0.856</td><td>0.595</td><td>0.493</td><td>0.896</td><td>0.444</td><td>-0.040</td><td>0.862</td><td>1.000</td></tr><tr><td>SW</td><td></td><td>0.123</td><td>0.871</td><td>0.538</td><td>0.460</td><td>0.854</td><td>0.394</td><td>+0.040</td><td>0.925</td><td>1.000</td></tr><tr><td rowspan="3">WHP</td><td>WS</td><td> $\checkmark$ </td><td>0.072</td><td>0.902</td><td>0.416</td><td>0.370</td><td>0.766</td><td>0.350</td><td>+0.040</td><td>0.890</td><td>1.000</td></tr><tr><td>HP</td><td></td><td>0.174</td><td>0.859</td><td>0.583</td><td>0.493</td><td>0.922</td><td>0.483</td><td>-0.080</td><td>0.415</td><td>0.000</td></tr><tr><td>SW</td><td></td><td>0.208</td><td>0.860</td><td>0.579</td><td>0.494</td><td>0.846</td><td>0.382</td><td>-0.120</td><td>0.283</td><td>0.000</td></tr><tr><td rowspan="2"></td><td>WS</td><td></td><td>0.103</td><td>0.926</td><td>0.316</td><td>0.288</td><td>0.850</td><td>0.334</td><td>+0.000</td><td>0.178</td><td>0.000</td></tr></table>

Table 15: RQ2: Full post-unlearning results, Qwen2.5-7B-Instruct. ROUGE-L↓: forget-set ROUGE-L after unlearning. ✓: verified (ROUGE-L < 0.10 & retain $\ge 0 . 8 0 \times$ base). $\| { \overline { { \Delta { \hat { v } } } } } \| \uparrow \colon$ centroid norm; $\mathrm { A U S S _ { L 2 } } \downarrow , \mathrm { A U S S _ { c o s ^ { 2 } } } \downarrow$ dispersion (per-method peak layer ℓ<sup>∗</sup>). Steer-AUC↓: steering AUC; Steer-WF↓: mean steered word-frequency score. ICL $\Delta \uparrow :$ ICL accuracy minus baseline. MIA↓: membership inference AUC; GCG↓: GCG ASR. Blank cells: checkpoint unavailable.
<table><tr><td rowspan="2">Method</td><td rowspan="2"> $\mathrm { C p t }$   $\checkmark$ </td><td></td><td>Forget</td><td colspan="2">Geometry  $\overline { { \ell ^ { * } } }$ </td><td colspan="5">Attack Success↓</td></tr><tr><td></td><td>ROUGE-L↓</td><td>||∆û||↑</td><td>AUSSL2↓  $\overline { { \mathrm { A U S S } _ { \mathrm { c o s } ^ { 2 } } } } .$ </td><td>Steer-AUC</td><td>Steer-WF</td><td>ICL ∆</td><td>MIA</td><td>GCG</td></tr><tr><td rowspan="2">Base</td><td>HP</td><td></td><td>0.106</td><td>0.872</td><td>0.534</td><td>0.439</td><td>0.433 0.373</td><td>+0.040</td><td>0.982</td><td>0.000</td></tr><tr><td>SW</td><td></td><td>0.098</td><td>0.866</td><td>0.557 0.395</td><td>0.473</td><td>0.482 0.261</td><td>+0.080</td><td>0.982</td><td>0.000</td></tr><tr><td rowspan="3">GA</td><td>WS</td><td></td><td>0.080</td><td>0.907</td><td>0.352</td><td>0.495 0.500</td><td>0.373</td><td>+0.000</td><td>0.938</td><td>0.000</td></tr><tr><td>HP</td><td>√</td><td>0.000</td><td>0.936</td><td>0.275</td><td>0.253</td><td>0.000</td><td>+0.000</td><td>0.119</td><td>0.000</td></tr><tr><td>SW</td><td></td><td>0.000</td><td>0.901</td><td>0.417</td><td>0.370 0.500</td><td>0.000</td><td>+0.000</td><td>0.461</td><td>0.000</td></tr><tr><td rowspan="3">DPO</td><td>WS</td><td></td><td>0.000</td><td>0.934</td><td>0.284 0.255 0.422</td><td>0.500</td><td>0.000</td><td>+0.000</td><td>0.185</td><td>0.200</td></tr><tr><td>HP</td><td>√</td><td>0.004</td><td>0.884</td><td>0.484</td><td>0.730</td><td>0.194</td><td>+0.000</td><td>0.861</td><td>0.000</td></tr><tr><td>SW</td><td></td><td>0.117</td><td>0.834</td><td>0.676 0.548</td><td>0.677</td><td>0.343</td><td>+0.000</td><td>0.965</td><td>1.000</td></tr><tr><td rowspan="3">NPO</td><td>WS</td><td> $\checkmark$ </td><td>0.020</td><td>0.895</td><td>0.441 0.389</td><td>0.911</td><td>0.238</td><td>+0.040</td><td>0.871</td><td>0.000</td></tr><tr><td>HP</td><td></td><td>0.198</td><td>0.863</td><td>0.569</td><td>0.478 0.744</td><td>0.374</td><td>+0.000</td><td>0.805</td><td>1.000</td></tr><tr><td>SW</td><td></td><td>0.189</td><td>0.846</td><td>0.632 0.524</td><td>0.782</td><td>0.324</td><td>+0.000</td><td>0.871</td><td>0.000</td></tr><tr><td rowspan="3">NPO+KL</td><td>WS</td><td></td><td>0.105</td><td>0.913</td><td>0.370 0.331</td><td>0.706</td><td>0.433</td><td>+0.040</td><td>0.770</td><td>1.000</td></tr><tr><td>HP</td><td></td><td>0.349</td><td>0.865</td><td>0.559</td><td>0.934</td><td>0.458</td><td>+0.000</td><td>0.482</td><td>0.000</td></tr><tr><td>SW</td><td></td><td>0.167</td><td>0.838 0.662</td><td>0.474 0.542</td><td>0.822</td><td>0.210</td><td>+0.000</td><td>0.246</td><td>0.800</td></tr><tr><td rowspan="2"></td><td>WS</td><td>0.145</td><td>0.908</td><td>0.390</td><td>0.348</td><td>0.874</td><td>0.449</td><td>+0.000</td><td>0.218</td><td>0.000</td></tr><tr><td>HP</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="3">RMU</td><td>SW</td><td></td><td>0.783</td><td>0.860</td><td>0.354</td><td>0.784</td><td>0.263</td><td>+0.000</td><td>0.854</td><td>1.000</td></tr><tr><td>WS</td><td>√</td><td>0.093 0.082</td><td>0.875</td><td>0.522</td><td>0.439</td><td></td><td>+0.040</td><td>0.785</td><td>0.800</td></tr><tr><td>HP</td><td></td><td>0.258</td><td>0.864 0.564</td><td>0.477</td><td>0.634 0.943</td><td>0.364 0.310</td><td>+0.000</td><td>0.495</td><td>0.000</td></tr><tr><td rowspan="2">WHP</td><td>SW</td><td></td><td>0.172</td><td>0.843</td><td>0.644 0.533</td><td>0.845</td><td>0.205</td><td>+0.000</td><td>0.218</td><td>0.000</td></tr><tr><td>WS</td><td></td><td>0.207</td><td>0.900</td><td>0.423</td><td>0.373</td><td>0.881 0.405</td><td>+0.040</td><td>0.245</td><td>0.000</td></tr></table>