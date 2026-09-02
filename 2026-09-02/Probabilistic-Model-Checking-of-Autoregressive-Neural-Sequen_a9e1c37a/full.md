# Probabilistic Model Checking of Autoregressive Neural Sequence Models

Helge Spieker<sup>[0000−0003−2494−4279]</sup>, Dennis Gross<sup>[0009−0001−5734−0538]</sup>, and Arnaud Gotlieb<sup>[0000−0002−8980−7585]</sup>

Simula Research Laboratory, Oslo, Norway Corresponding author: helge@simula.no

Abstract. Test-set accuracy is silent on two issues that matter when deploying autoregressive neural sequence models: how much probability mass the system under test (SUT) places on constraint-violating alternatives that are reachable under sampling and what fraction of the input population satisfies a domain requirement. We answer both with probabilistic model checking. The pipeline extracts a discrete-time Markov chain (DTMC) from the SUT’s token-by-token generation, verifies formal PCTL specifications with the PRISM model checker, and aggregates the per-input verdicts into a coverage curve over the input space. A soundness theorem establishes the DTMC as an under-approximation, so every verdict yields a certified interval on the SUT’s true reachability probability. The coverage built from those verdicts is, therefore, conservative by construction. A counterexample-guided abstraction refinement (CE-GAR) loop adaptively tightens the interval, and a maximum-likelihood algorithm extracts the most probable falsifying trace. Two case studies exercise the pipeline. On a GPT-2 computer-aided process-planning (CAPP) model with 100% test accuracy, the pipeline quantifies the probability mass greedy decoding hides, but that is reachable with sampling; and identifies the smallest training fraction at which an ordering requirement holds population-wide, neither of which test accuracy can report. We then verify the SMILES molecular generator with a 50× larger vocabulary. The only change is an external chemical-validity oracle, and the pipeline identifies the gap between structural completeness and chemical validity.

Keywords: probabilistic model checking · DTMC abstraction · PCTL · autoregressive transformers · neural network verification

## 1 Introduction

Autoregressive transformer models, the technology behind large language models, now appear in engineering pipelines, where generated sequences afect physical artifacts, regulatory outcomes, and downstream automation. However, their reliability is far from guaranteed [28, 14]. Yet their evaluation still rests almost entirely on test-set accuracy: a single number reporting how often the output matches the held-out reference [20]. Three questions matter for deployment that no test set can answer on its own. First, how much probability mass does the model place on low-confidence alternatives that greedy decoding never selects, but a sampled decoder eventually will [13]? Second, do generated sequences obey domain constraints that lie outside the training signal and so never enter an accuracy metric, because it does not report failure causes or because these failures are not part of the ground-truth dataset in the first place? Third, what fraction of the entire input population satisfies a given criterion? Are domain violations corner cases or potentially frequent occurrences?

As an example, consider the problem of computer-aided process planning (CAPP). High-level CAPP is the task of determining appropriate sequences of manufacturing processes (e.g., milling, drilling, grinding) to produce a given part. A part can usually be made in several ways, so the task is naturally a variable-length sequence-prediction problem. Stathatos et al. [27] address this problem with a domain-specific GPT-2 transformer [25] that maps a discrete part description to an ordered sequence of operations. The model reports 100% sequence accuracy on the held-out test set, a number that suggests it is ready to deploy. Our pipeline shows otherwise: even the smallest checkpoint with 100% accuracy (15% of the data) still routes over 9% of its probability mass to completions greedy decoding never selects, and on data-starved checkpoints the most probable hidden trace violates manufacturing phase ordering. Neither failure mode is visible to accuracy, because both reference probability mass or a domain constraint the ground-truth sequence never exercises. This hidden mass is a risk, but it is also a lever.

A deployed model must commit to a decoding rule: greedy decoding always emits the single most likely next operation, whereas a sampled decoder draws from the model’s distribution and so can produce diferent plans on repeated runs. This can potentially improve performance, because the best next token might not always have the highest probability; especially in settings with limited data. Because our pipeline reasons about the whole distribution rather than one trajectory, it predicts (without re-running the model) how often a sampled decoder, paired with the verifier as a filter, recovers a correct plan greedy decoding would miss [5]. On the most data-starved checkpoint a slightly sharpened sampler does exactly that, turning the risk into a usable safety margin.

In our approach, we treat the autoregressive transformer as the system under test (SUT) and apply a three-stage probabilistic-model-checking pipeline. It extracts a discrete-time Markov chain (DTMC) by unrolling the token-by-token generation, verifies Probabilistic Computation Tree Logic (PCTL) specifications against it with the PRISM model checker [18], and aggregates the per-input verdicts into a coverage curve over the input population. Because the DTMC is a sound under-approximation, each input’s set of PRISM verdicts yields a certified interval on the SUT’s reachability probability, not a point estimate, and the population fractions built from these intervals are themselves conservative. A CEGAR-style refinement loop [4] tightens the interval adaptively, and a maximum-likelihood counterexample algorithm extracts the most probable falsifying trace from any violated specification. The result is a population coverage statement: the fraction of inputs for which property ϕ holds with probability at least θ.

Overall, this paper makes the following contributions:

1. A probabilistic model verification methodology for autoregressive transformers built on three artifacts: a DTMC abstraction of the SUT (Sec. 3.3), PCTL specifications including domain-specific properties (Sec. 3.4), and population coverage over the input space (Sec. 3.5).

2. A soundness theorem (Theorem 1) that yields a certified interval on the SUT, tightened adaptively by a CEGAR-style refinement loop (Sec. 3.7).

3. A maximum likelihood counterexample algorithm (Sec. 3.8) extracting the most probable falsifying trace as an inspectable artifact.

4. Experiments on two systems: CAPP (Sec. 4.1) exposing failure modes invisible to accuracy, and SMILES molecule generation (Sec. 4.2) with a substantially larger vocabulary.

## 2 Related Work

Model checking and abstraction of neural systems. A close line of work applies probabilistic model checking to neural controllers: MOSAIC [2] abstracts the closed-loop system as an MDP (or interval MDP) and verifies PCTL properties with PRISM. Similarly, Gross et al. extract DTMCs from RL policies [10, 9] or bounded LLM outputs [11] for safety verification. Our approach difers in its target: the joint distribution of an autoregressive generation process (a DTMC) rather than a policy under environment stochasticity (an MDP), and it adds an aggregation layer over the input population. Weiss et al. [31] extract finite automata from recurrent networks via L\*-style equivalence queries. Here, our extraction targets a probabilistic surrogate by direct forward unrolling under threshold pruning, with the soundness gap explicitly bounded by Theorem 1.

Statistical and deterministic verification. UPPAAL SMC [7] and Plasma [19] estimate probabilities by simulation; we compute exact per-input reachability via PRISM’s backward induction and use the aggregation layer only to combine those exact verdicts into a population. Viewed diferently, DTMC extraction is symbolic execution of a probabilistic program [8], threshold pruning is an abstract interpretation [6], and the refinement loop adapts CEGAR [4] to the DTMC setting. Deterministic verifiers, e.g., Reluplex/Marabou [15, 16], ERAN [26], or α-β-CROWN [29], verify a single forward pass of a feed-forward network; our target is the generative process over multiple decoding steps, which requires a probabilistic formalism.

Population-level guaranties and neural-network testing. Conformal prediction [24, 22, 1] provides population-level claims over black-box output distributions. It is model-agnostic and does not reference internals, whereas our pipeline accesses the SUT’s token probabilities and enables domain-specific PCTL specifications such as phase ordering. Coverage-guided testing (DeepXplore [23], Deep-Gauge [21]) targets activation-space coverage of classifiers; our coverage curve is analogous to a coverage criterion, but with soundness guaranties.

![](images/4cf342160ded74c238cac3531f0027be23db4f2a4bf513bbb0304544f57c1042.jpg)  
Fig. 1. Pipeline overview. A per-input DTMC abstraction is verified by PRISM against PCTL specifications; verdicts aggregate into a population coverage curve. The CEGAR loop (dashed) re-expands the highest-impact pruned transitions; counterexamples are extracted on demand from violated specifications.

## 3 Verification Pipeline

## 3.1 Preliminaries

An autoregressive model over a finite vocabulary V factorizes a sequence distribution left to right, $\begin{array} { r } { P ( x ) \ = \ \prod _ { t = 1 } ^ { n } P ( x _ { t } \ | \ x _ { < t } ) } \end{array}$ , each conditional a softmax over the logits; a temperature T rescales the logits by $1 / T$ before the softmax $( T \ \to \ 0$ recovers greedy decoding). This factorisation maps onto a discretetime Markov chain (DTMC) $D = ( S , s _ { 0 } , \mathbf { P } , L )$ whose states are token prefixes, whose transitions carry the conditionals, and whose labeling L marks states with atomic propositions, because each transition strictly extends the prefix the chain is acyclic, so reachability probabilities follow from a single backward-induction pass with no iterative error. We specify properties in PCTL [12] using the quantitative reachability form $\mathsf { P } _ { = ? } [ \mathsf { F }$ label], the exact probability of eventually reaching a label-state, computed by PRISM [18]. An aggregation layer (Sec. 3.5) summarises these per-input verdicts to a coverage curve over the input population.

## 3.2 Pipeline Overview

The pipeline has three independently operable stages (Figure 1): DTMC abstraction of the SUT, specification verification by PRISM model checking, and coverage aggregation that combines per-input verdicts into a population-level coverage curve. Each stage reads from and writes to files, so any stage can be re-run or replaced independently. Two auxiliary components operate alongside: a CEGAR-style refinement loop (Sec. 3.7) that re-expands the highest-impact pruned transitions, and a counterexample extractor (Sec. 3.8) that produces an inspectable falsifying trace for any violated specification. The following presentation is agnostic of the specific SUT. The two case studies in Sections 4.1 and 4.2 instantiate the same pipeline on very diferent SUTs and specifications.

## 3.3 DTMC Extraction

Tree expansion. Given an input, the extractor builds the DTMC by breadthfirst expansion. At each non-terminal state s, tokens with conditional probability $\geq \tau$ receive explicit successor states; the remaining mass is routed, unrescaled, to an absorbing low\_prob state. Paths whose accumulated probability falls below $\rho$ are also redirected to low\_prob. A hard depth limit $d _ { \mathrm { m a x } }$ guaranties the termination. Paths that reach $\overline { { \mathrm { i t } } }$ without terminating are redirected to an absorbing truncated state.

Table 1. PCTL specifications verified per input (CAPP instantiation).
<table><tr><td>Query</td><td>Meaning</td><td>Category</td></tr><tr><td>P=?[F success]</td><td>valid completion</td><td>Generic</td></tr><tr><td>P=?[F low_prob]</td><td>mass to sink</td><td>Generic</td></tr><tr><td>P=?[F invalid]</td><td>structural violation</td><td>Generic</td></tr><tr><td>P=?[F truncated]</td><td>depth-limit reached</td><td>Generic</td></tr><tr><td>P=?[F correct]</td><td>ground-truth sequence</td><td>Correctness</td></tr><tr><td>P=?[F ordered]</td><td> $P {  } S ^ { * } {  } F ^ { * }$  holds</td><td>Domain (CAPP)</td></tr><tr><td>P=?[F misordered]</td><td> $P {  } S ^ { * } {  } F ^ { * }$  fails</td><td>Domain (CAPP)</td></tr><tr><td>P=?[F critical]</td><td>critical state visited</td><td>Confidence</td></tr></table>

Unless stated otherwise, the extractor operates at $T = 1 . 0$ (the model’s native distribution). Since any fixed T yields a categorical distribution at each step, the extracted object is again a DTMC and Theorem 1 applies unchanged.

Structural validity pruning. At each expansion, the extractor checks structural well-formedness against a SUT-specific grammar of admissible prefixes; for CAPP this forbids padding before <EOS>, restricts <EOS> to terminal position, and requires each <EOC> to be preceded by at least one process token. Violating prefixes go to an absorbing invalid state.

Critical-state annotation. States with top-1/top-2 probability gap $g ( s ) < 1 0 \%$ are flagged as critical; the flag is exported as a Boolean PRISM label and queried via $\scriptstyle \mathtt { P } = ? [ \mathtt { F }$ critical]. Defaults are per-token threshold $\tau = 0 . 5 \%$ , cumulative floor $\rho = 1 0 ^ { - 4 }$ , and depth limit $d _ { \operatorname* { m a x } } = 2 0$

## 3.4 PRISM Export and PCTL Specifications

The pipeline exports the DTMC to the PRISM modelling language, encoding the four absorbing sinks (success, low\_prob, invalid, truncated) and the critical-state flag as Boolean labels. Beyond these generic labels, the extractor accepts domain-specific labels supplied per case study, a phase-ordering predicate on success terminals for CAPP (Sec. 4.1) or a chemical-validity check for SMILES (Sec. 4.2). Two invariants hold by construction for generic labels: $P ( { \mathrm { s u c c e s s } } ) + P ( { \mathrm { L O W } } { \_ }  { \mathrm { P R O B } } ) + P ( { \mathrm { I N V A L I D } } ) + P ( { \mathrm { T R U N C A T E D } } ) = 1$ , and any domain-specification label that partitions success terminals into pass/fail satisfies $P ( { \mathrm { P A S S } } ) + P ( { \mathrm { F A I L } } ) = P ( { \mathrm { S U C C E S S } } )$ . Table 1 lists the PCTL queries used in the CAPP study; the SMILES study substitutes a chemical-validity label for the phase-ordering pair.

## 3.5 Population Coverage

Fix a threshold $\theta \in [ 0 , 1 ]$ and let $\begin{array} { r } { \hat { \boldsymbol \mu } ( \boldsymbol \theta ) = { \frac { 1 } { N } } \sum _ { i = 1 } ^ { N } \mathbf 1 [ P _ { D } ( \phi _ { i } ) \geq \boldsymbol \theta ] } \end{array}$ be the fraction of inputs of the $N$ test whose verified probability of satisfying ϕ satisfies θ. Sweeping θ over [0, 1] trace the coverage curve $\hat { \mu } ( \theta )$ , a population-level summary of how strongly the specification holds in the input space. Sec. 3.6 shows that this curve is conservatively transferred from the abstraction to the SUT itself.

The result is a population coverage statement of the form ${ \hat { \mathbf { \Omega } } } ^ { \left. 4 \right. } { \hat { \boldsymbol { \mu } } } ( \theta )$ of all inputs satisfy ϕ at threshold $\theta . ^ { \mathfrak { n } }$ The population is covered at level c for $( \phi , \theta ) { \mathrm { ~ i f ~ } } { \hat { \mu } } ( \theta ) \geq$ $c .$ The trivial case $c = 0$ asks only that the curve rule out total failure; an operational deployment criterion fixes c to an application-specific floor. Unless stated otherwise we report the coverage value $\hat { \mu } ( \theta )$ and flag the deployment level separately. By Lemma 1 the reported curve already under-estimates the $\mathrm { S U T } { \mathrm { s } }$ true coverage, so we report it directly.

## 3.6 Soundness and Certification Guarantees

The guarantee rests on a single property of the DTMC: it is a sound underapproximation of the model, so every reachability probability the pipeline reports is a certified bound on the true value, conservative by construction. Theorem 1 bounds the abstraction gap; two corollaries discharge a term and record the mass-conservation identity; Lemma 1 carries the per-input bound through the population layer of Sec. 3.5.

Theorem 1 (Sound under-approximation). Let M be the autoregressive model and $D$ the DTMC from Sec. 3.3. For any reachability property $\phi$ that holds only at the success terminals:

$$
\begin{array} { r l } & { P _ { D } ( \phi ) \leq P _ { M } ( \phi ) \leq P _ { D } ( \phi ) + P _ { D } \mathrm { ( L O W \mathrm { \_ P R O B } ) } } \\ & { \qquad + P _ { D } \mathrm { ( I N V A L I D ) } + P _ { D } \mathrm { ( T R U N C A T E D ) } . } \end{array}
$$

Proof (Sketch). By construction, the four sinks are absorbing and, together with the success terminals, partition every path of $D \colon$ each path is classified exactly once, at the step it is either completed or diverted. Every explicit path in $D$ maps to a model path with identical probability (branched tokens carry the original softmax mass; remaining mass is rerouted intact to the appropriate sink), giving the lower bound. For the upper bound, any model path reaching a ϕ-satisfying terminal either survives all checks (counted by $P _ { D } ( \phi ) )$ ) or is diverted at some step into a sink. Because diversion is absorbing, a diverted path contributes no further mass to $\phi$ within $D$ and its fate M is unknown. Attributing all diverted mass adversarially to $\phi$ gives the bound.

The two sides of the bound have a simple reading. The lower bound holds because every explicit path that the extractor keeps is a real model path with its original probability. The upper bound holds because the only mass that the abstraction loses is the mass it diverted into a sink, and the worst case is that all of it would have satisfied $\phi .$ The interval between is exactly the diverted mass.

One of those sink terms can be eliminated outright. The structural validity grammar of Sec. 3.3 is prefix-closed. Once a prefix is rejected, every extension of it is rejected, too. A path diverted to invalid can then never reach a ϕ-satisfying terminal, so its mass does not belong in an upper bound on $P ( \phi )$

Corollary 1 (Tightened upper bound). Under the prefix-closed grammar of $S e c . ~ \mathcal { S } . \mathcal { S } , ~ P _ { M } ( \phi ) \leq P _ { D } ( \phi ) + P _ { D } ( \mathrm { L O W \_ P R O B } ) + P _ { D } ( \mathrm { T R U N C A T E D } )$

Corollary 2 (Mass conservation). For every per-input DTMC, the masses of the four sinks (success, low\_prob, invalid, truncated) sum to 1.

Corollary 2 is what makes the four sink masses a valid decomposition rather than four loosely related quantities. They partition the unit probability, so any one of them is determined by the other three. The evaluation uses this as a consistency check and reports the success lower bound alongside the certified upper bound of Corollary 1 throughout.

It remains to show that the bound survives aggregation. The aggregation layer counts all inputs whose verified probability clears a threshold θ. If each per-input number is a lower bound, the count built from them should itself be a lower bound on the true population fraction.

Lemma 1 (Conservative coverage). Let $\hat { \mu } ( \theta )$ be the coverage computed from the DTMC lower bounds $P _ { D } ( \phi _ { i } )$ , and let ${ \hat { \mu } } _ { M } ( \theta )$ be the same fraction computed from the true model probabilities $P _ { M } ( \phi _ { i } )$ on the same inputs. Then $\hat { \mu } ( \theta ) \leq \hat { \mu } _ { M } ( \theta )$ for every θ.

Proof. By Theorem 1, $P _ { D } ( \phi _ { i } ) \leq P _ { M } ( \phi _ { i } )$ for every input i, so $\mathbf { 1 } [ P _ { D } ( \phi _ { i } ) \geq \theta ] \leq$ $\mathbf { 1 } [ P _ { M } ( \phi _ { i } ) \geq \theta ]$ pointwise; summing over the test set gives $\hat { \mu } ( \theta ) \leq \hat { \mu } _ { M } ( \theta )$

Pessimism in the abstraction propagates as pessimism in the coverage curve: abstraction error can only make the reported coverage more conservative, yet never unsound. A tighter τ or a CEGAR pass can only raise the certified fraction. The one caveat is directional. The guarantee is built for properties that hold at the success terminals, where $P _ { D }$ underestimates. For a violation property such as misordered, this under-estimation runs the other way: a non-zero per-input verdict proves a violation exists, but a zero verdict does not certify its absence.

## 3.7 CEGAR-Style Refinement

To tighten the upper bound of Theorem 1, we adapted the CEGAR principle [4] to the DTMC setting. Each round ranks all pruned $( s , t )$ pairs by impact Pr[reach $s ] \cdot p _ { \mathrm { p r u n e } } ( s , t )$ , re-expands the top-K pairs, splices new subtrees into D, re-runs PRISM, and halts when $P _ { D } ( \mathrm { L O W \_ P R O B } ) \leq \varepsilon _ { \mathrm { t a r g e t } }$ or the round budget is exhausted.

## 3.8 Counterexample Extraction

For any violation, the pipeline extracts the falsifying trace with the highest joint probability by running Dijkstra’s algorithm on the DTMC with − log p edge weights, then decodes the state-ID path back to the SUT token vocabulary.

## 4 Evaluation

The evaluation is organized around four research questions that span both case studies. We address RQ1–3 as part of case study 1. Case study 2 focuses on portability, while also ofering a second perspective on RQ1.

RQ1 (Specification informativeness): Does the pipeline expose failure modes that test-set accuracy cannot report (Secs. 4.1 and 4.2)?

RQ2 (Population coverage): Can per-input PRISM verdicts be aggregated into a population-level deployment guarantee (Sec. 4.1)?

RQ3 (Counterexamples and adaptive refinement): Does the pipeline produce inspectable counterexamples for violated specifications, and does CEGARstyle refinement tighten the soundness interval (Sec. 4.1)?

RQ4 (Portability): Does the methodology generalize to an SUT with a substantially larger vocabulary and a semantic oracle external to the model (Sec. 4.2)?

## 4.1 Case Study 1: CAPP

System Under Test The SUT is the trained GPT-2-based CAPP model of Stathatos et al. [27]. We treat it as a black box and change neither its architecture nor its training. The CAPP task maps a six-feature discrete part description (geometry, holes, threads, surface finish, tolerance, batch size) to an ordered sequence of manufacturing operations. The input space is fully enumerable: 7,840 unique parts arise from the Cartesian product of the feature domains. Outputs are sequences of process tokens delimited by <EOC> and terminated by <EOS>, encoding one or more alternative process chains. The architecture is a 4-layer GPT-2 with 64-dimensional embeddings and a 53-token vocabulary (29 feature tokens, 20 process tokens, 4 special tokens). The authors released eight checkpoints trained on dataset fractions from 1% to 70%; the model reaches 100% sequence accuracy on the held-out test set from 15% training onward.

As an additional domain specification, we consider the phase ordering of manufacturing process steps. The 20 process tokens partition into three manufacturing phases: Primary (7 tokens, shape-giving), Secondary (9 tokens, subtractive refinement), and Finishing (4 tokens, surface quality). By manufacturing convention, operations within each chain must follow the partial order Primary → Secondary → Finishing : exactly one Primary operation, then zero or more Secondary, then zero or more Finishing, without backward phase transition. This constraint is external to the neural model, which is not trained to enforce it explicitly, and is the domain-specific property of interest for the CAPP case study, encoded as the ordered/misordered PCTL labels of Table 1.

We evaluated all eight training fractions (Table 2, omitting the fixed 1,176- part validation split), building a per-input DTMC for every test sample and verifying seven of the eight specifications of Table 1 with PRISM (critical is obtained by forward propagation), yielding 37,554 DTMCs and 262,878 checking runs with zero solver failures. All experiments ran on an Apple M2 Max (32 GB RAM) using PRISM 4.10 with the explicit-state engine and four-way parallelism.

Table 2. Training configurations with accuracy [27] and DTMC structural statistics.
<table><tr><td colspan="4">Training</td><td colspan="4">DTMC structure</td></tr><tr><td>Config Train</td><td></td><td>Test Tkn acc. Seq. acc.|</td><td></td><td>|Avg states Max states Avg paths Multi-branch</td><td></td><td></td><td></td></tr><tr><td>1%</td><td>78 6,586</td><td>93.5%</td><td>47.8%</td><td>933.6</td><td>2,208</td><td>64.7</td><td>100%</td></tr><tr><td>5%</td><td>392 6,272</td><td>99.7%</td><td>96.1%</td><td>55.4</td><td>560</td><td>5.2</td><td>82%</td></tr><tr><td>10%</td><td>784 5,880</td><td>100.0%</td><td>99.7%</td><td>16.3</td><td>54</td><td>1.0</td><td>1.8%</td></tr><tr><td></td><td>15% 1,176 5,488</td><td>100.0%</td><td>100.0%</td><td>16.2</td><td>31</td><td>1.0</td><td>0.6%</td></tr><tr><td></td><td>20% 1,568 5,096</td><td>100.0%</td><td>100.0%</td><td>16.3</td><td>37</td><td>1.0</td><td>1.5%</td></tr><tr><td></td><td>30% 2,352 4,312</td><td>100.0%</td><td>100.0%</td><td>16.2</td><td>23</td><td>1.0</td><td>0.05%</td></tr><tr><td></td><td>50% 3,920 2,744</td><td>100.0%</td><td>100.0%</td><td>16.3</td><td>23</td><td>1.0</td><td>0.04%</td></tr><tr><td></td><td>70% 5,488 1,176</td><td>100.0%</td><td>100.0%</td><td>18.0</td><td>37</td><td>1.0</td><td>0%</td></tr></table>

RQ1: Failure Modes Invisible to Accuracy We answer along two dimensions: probability-mass divergence between accuracy and the specification (Table 3) and structural confidence diagnostics derived from the DTMC (Table 2).

Accuracy and verification verdicts diverge. From 15% training onward, testset sequence accuracy is 100%, yet mean P(success) ranges from 0.907 (15%) to 0.974 (70%). By Theorem 1, the 70% model’s true reachability is certified to lie in [0.974, 1.000], with the gap entirely attributable to low\_prob. The pipeline separates two failure modes that accuracy conflates: exact ground-truth match (what accuracy measures) and concentration of probability mass on any valid completion.

At the same time, structural validity is essentially preserved. P(invalid) = 0 in every configuration except 1% (0.009). Even under severe data starvation, the model has implicitly learned the syntactic well-formedness of the output language; this is itself a verified property, not a stochastic observation.

P(success) flattens from 30% training onward, gaining 0.019 between 30% and 50% and a further 0.006 between 50% and 70% (0.949, 0.968, 0.974). A per-token threshold sweep on the 70% model (200-input subsample) decomposes the remaining gap into two parts with diferent origins. The first is a threshold artifact, and it is small. Across $\tau \in \{ 0 . 5 , 0 . 2 5 , 0 . 1 \}$ % the value barely moves $( P ( { \mathrm { s U C C E S S } } ) = 0 . 9 7 4 6 \to 0 . 9 7 4 7 )$ , and only at $\tau = 0 . 0 1 \%$ does appreciable mass re-enter success (0.981, with low\_prob falling from 2.5% to 1.9%) before saturating again.

The second part is irreducible. A residual ≈ 1.9% survives even at $\tau = 1 0 ^ { - 4 }$ and cumulative floor $\rho = 1 0 ^ { - 6 }$ . This residual is not an artifact of pruning, but a genuine sub-threshold dispersion. The model spreads probability across valid alternative completions, each individually below τ . This dispersion is steep only when training data is scarce. At 1% training, just 36% of the success mass concentrates on the canonical target, rising to 79% at 5% and above 99.5% from 10% onward. For a deployed model the dispersion is therefore negligible. Yet, it is invisible to both greedy decoding and test accuracy, which see only the canonical target whether it holds 36% or 99.5% of the mass.

Table 3. Mean probability mass decomposition (Corollary 2; max per-input deviation $< 1 0 ^ { - 1 0 }$ across all 37,554 DTMCs).
<table><tr><td>Config</td><td>P(suCc.)</td><td> $P ( \mathrm { { L O W } _ { - } \mathrm { { P . } ) } }$ </td><td> $P ( { \mathrm { I N V . } } )$ </td><td>P(TRUNC.)</td><td>Sum</td></tr><tr><td>1%</td><td>0.128</td><td>0.863</td><td>0.009</td><td>0.000</td><td>1.000</td></tr><tr><td>5%</td><td>0.629</td><td>0.371</td><td>0.000</td><td>0.000</td><td>1.000</td></tr><tr><td>10%</td><td>0.845</td><td>0.155</td><td>0.000</td><td>0.000</td><td>1.000</td></tr><tr><td>15%</td><td>0.907</td><td>0.093</td><td>0.000</td><td>0.000</td><td>1.000</td></tr><tr><td>20%</td><td>0.922</td><td>0.079</td><td>0.000</td><td>0.000</td><td>1.000</td></tr><tr><td>30%</td><td>0.949</td><td>0.051</td><td>0.000</td><td>0.000</td><td>1.000</td></tr><tr><td>50%</td><td>0.968</td><td>0.032</td><td>0.000</td><td>0.000</td><td>1.000</td></tr><tr><td>70%</td><td>0.974</td><td>0.026</td><td>0.000</td><td>0.000</td><td>1.000</td></tr></table>

Finally, the DTMC structure is a confidence diagnostic. Table 2 reports structural statistics: the collapse from 934 mean states (1%) to 18 (70%), with the multi-branch fraction falling from 100% to 0%, traces the transition from a diffuse model to a near-deterministic one. The critical-state diagnostic sharpens the picture: at 1% training, 97.9% of inputs have at least one critical state (mean 14.2, max 107); at 5% this drops to 5.5%; from 15% onward, none is observed. The pipeline thus certifies not only that the 70% model is accurate but that it never approaches a decision boundary. RQ1 is addressed again in the SMILES case study (Sec. 4.2) for which the structural specification is not suficient.

## RQ2: Population Coverage and Operating Envelope

Per-input ordering verdicts. From 10% training onwards, $P _ { D } ( \mathrm { { M I S O R D E R E D } } ) = 0$ across the entire test set: every retained success path is then correctly ordered, with only 0.3% of retained probability mass misordered even at 1% training. Because misordered is a violation property, the zero verdict guarantees that no retained path misorders, but does not certify that the pruned sub-threshold mass is free of misordering. This ordering property is invisible to sequence-level accuracy, and the coverage layer turns it into a population claim.

Population coverage. Table 4 reports the population ordering coverage $\hat { \mu } ( \theta )$ at $\theta = 0 . 9 0$ . A sharp transition occurs between 10% and 30% training: coverage climbs from $\hat { \mu } = 0 . 2 5 2$ at 10% to $\hat { \mu } = 0 . 9 7 1$ at 30%, the smallest fraction at which the population is covered at the deployment level $c = 0 . 9 0$ . From 50% onwards, every test input meets the threshold $\begin{array} { r } { ( \hat { \mu } = 1 . 0 0 0 ) } \end{array}$ . Because the curve is built from DTMC lower bounds, Lemma 1 makes it a conservative estimate of the SUT’s true coverage.

Operating envelope over decoder temperature. A single-temperature coverage value answers a single what-if question; a sweep over decoder temperatures characterises a safe operating envelope. We sweep $T \in \{ 0 . 5 , 0 . 7 , 1 . 0 , 1 . 3 , 2 . 0 \}$ for the 1%, 10%, 30%, and 70% configurations and re-run the full PCTL suite at each temperature. For the 30% and 70% models coverage stays at $\hat { \mu } \geq 0 . 9 7$ for $T \leq 1 . 0$ and drops sharply at $T = 1 . 3 .$ . The 1% model never reaches $\hat { \mu } = 0 . 9 0 ;$ even at $T = 0 . 5$ its coverage is 0.789. The certified safe envelope is therefore $T \leq 1 . 0 \colon$ above this the increased entropy places enough mass below τ to break the ordering coverage.

Table 4. Population ordering coverage $\hat { \mu }$ for P(ordered) $\geq 0 . 9 0$ across training fractions (N = test-set size).
<table><tr><td>Config</td><td>1%</td><td>5%</td><td>10%</td><td>15%</td><td>20%</td><td>30%</td><td>50%</td><td>70%</td></tr><tr><td>N</td><td>6,586</td><td>6,272</td><td>5,880</td><td>5,488</td><td>5,096</td><td>4,312</td><td>2,744</td><td>1,176</td></tr><tr><td>û</td><td>0.000</td><td>0.004</td><td>0.252</td><td>0.576</td><td>0.705</td><td>0.971</td><td>1.000</td><td>1.000</td></tr></table>

Table 5. Best-of-N for the 1% model $( N { = } 1 0 )$ ; greedy correctness is 0.47 at every T. pass@10 on correct (oracle) and ordered (deployable), and distinct valid plans.
<table><tr><td>T</td><td>pass@10(CORRECT)</td><td>pass@10(ORDERED)</td><td>coverage@10</td></tr><tr><td>0.5</td><td>0.56</td><td>1.00</td><td>1.87</td></tr><tr><td>0.7</td><td>0.54</td><td>1.00</td><td>2.20</td></tr><tr><td>1.0</td><td>0.20</td><td>0.52</td><td>0.87</td></tr><tr><td>1.3</td><td>0.05</td><td>0.38</td><td>0.49</td></tr><tr><td>2.0</td><td>0.00</td><td>0.34</td><td>0.39</td></tr></table>

Best-of-N as a certifiable decoder. The same propagated terminal probabilities also score, in closed form and with no extra model runs, the success rate of drawing N samples and keeping a verifier-accepted one [5], pass@ $N = \mathbb { E } _ { x } [ 1 -$ $( 1 - P _ { x } ( \phi ) ) ^ { N } ]$ . At $T = 1 . 0$ best-of-N never overtakes greedy decoding in the 1% regime (the sampling mass is too difuse), unless we sharpen the decoder (Table 5). At $T = 0 . 5$ , best-of-10 makes every input yield a verifier-ordered completion within ten draws (pass@10(ordered) = 1.00) and surfaces 1.87 distinct valid plans per input against greedy’s one. Since ordering-validity, unlike correctness, is checkable at run time, the verifier doubles as selector, so this ordered result is the deployable claim; the correct column (0.56 at $T = 0 . 5$ against greedy’s 0.47) is an oracle upper bound on how much an ideal selector could additionally recover. This makes best-of-N at $T < 1 . 0$ the recommended decoder in the data-starved regime, where greedy decoding is weakest, i.e., the 1% checkpoint is a deliberate worst case and not a deployment target. For welltrained $( \ge 1 5 \% )$ models, greedy decoding is already near-deterministic and the decoding gap closes, so there the pipeline’s value shifts from recovering plans to certifying that little hidden mass remains.

RQ3: Counterexamples and Adaptive Refinement The maximum-likelihood counterexample algorithm turns a quantitative verdict into an inspectable falsifying trace. We extract the most likely misordering trace from the DTMC of

![](images/67fb5f1e8bad78bf732559fe0a6e4ff2006b8d8c1da4dfc45877f04fce4129da.jpg)  
Fig. 2. Maximum-likelihood misordering counterexample for the 1% model. Edge labels are (rounded) conditional next-token probabilities. The first operation is 5-axis Grinding (Finishing phase, highlighted), before any Primary operation.

Table 6. CEGAR refinement: distribution of P (low\_prob) across N = 50 sampled inputs at the initial DTMC (round 0) and after 5 refinement rounds. Quartiles are taken over the per-sample verification results.
<table><tr><td rowspan="2">Config</td><td colspan="3">Round 0</td><td colspan="3">Round 5</td></tr><tr><td> $P _ { 2 5 }$ </td><td>Median</td><td> $P _ { 7 5 }$ </td><td> $P _ { 2 5 }$ </td><td>Median</td><td> $P _ { 7 5 }$ </td></tr><tr><td>1%</td><td>0.791</td><td>0.929</td><td>0.975</td><td>0.761</td><td>0.919</td><td>0.961</td></tr><tr><td>5%</td><td>0.312</td><td>0.396</td><td>0.516</td><td>0.235</td><td>0.313</td><td>0.431</td></tr><tr><td>10%</td><td>0.127</td><td>0.163</td><td>0.219</td><td>0.090</td><td>0.120</td><td>0.174</td></tr><tr><td>15%</td><td>0.057</td><td>0.090</td><td>0.121</td><td>0.039</td><td>0.067</td><td>0.093</td></tr><tr><td>20%</td><td>0.054</td><td>0.076</td><td>0.102</td><td>0.037</td><td>0.055</td><td>0.079</td></tr><tr><td>30%</td><td>0.024</td><td>0.048</td><td>0.066</td><td>0.016</td><td>0.035</td><td>0.051</td></tr><tr><td>50%</td><td>0.024</td><td>0.031</td><td>0.039</td><td>0.018</td><td>0.023</td><td>0.030</td></tr><tr><td>70%</td><td>0.019</td><td>0.027</td><td>0.033</td><td>0.014</td><td>0.021</td><td>0.026</td></tr></table>

the 1%-model input with the highest misordering probability. The decoded trace (Figure 2) carries joint probability $3 . 5 \times 1 0 ^ { - 3 }$ . The model emits a Finishing-phase operation before any Primary operation, violating the phase-ordering property. This trace is never selected by greedy decoding and is invisible to test-set evaluation. The counterexample algorithm answers not just whether a violation exists, but which trace is most likely, and reports its formal probability. The result is a per-input artifact that a deployment engineer can inspect.

The CEGAR loop then tightens the soundness interval adaptively. Here, we run the CEGAR loop on $N = 5 0$ inputs per configuration $( \mathrm { t o p } { - } K = 5 $ $\varepsilon _ { \mathrm { t a r g e t } } = 1 0 ^ { - 3 } )$ and report the distribution of $P _ { D } ( \mathrm { L O W \_ P R O B } )$ at rounds 0 and 5 (Table 6). Two patterns are robust. First, per-sample dispersion at round 0 grows as training fraction shrinks, and refinement contracts the inter-quartile range everywhere. Second, configurations whose round-0 median is large (1% and $5 \% )$ show the smallest relative improvement: five rounds cut the 1% median by only a few percent, while a 30%-model sample benefits by 25–35%. Refinement is therefore most efective on the moderate configurations, even though the certified interval is widest on the data-starved ones: there much of the gap is the irreducible sub-threshold dispersion of Sec. 4.1, which bounded top-K re-expansion cannot reclaim. Closing it would need a lower threshold τ (with diminishing returns, Sec. 4.1) rather than more refinement rounds.

## 4.2 Case Study 2: SMILES Molecular Generation

Case study 1 exercises the pipeline on a small, fully enumerable input space with a domain specification (phase ordering) defined over an output grammar the SUT already enforces almost perfectly. The question is whether the extraction of the DTMC or the specification layer depends on these favorable conditions. RQ4 asks whether the methodology generalises to a SUT that stresses both: a much larger vocabulary that produces much larger DTMCs, and a semantic specification (chemical validity) that the output grammar does not enforce by construction. We answer in two parts: an operational portability assessment, and a specification-informativeness assessment showing that the semantic specification surfaces failures invisible both to a structural specification and to accuracy.

System Under Test The SUT is gpt2\_zinc\_87m<sup>1</sup>, an 87 M-parameter GPT-2 pretrained on the ZINC molecular database with a ≈ 2,700-token byte-pair encoding (BPE), which is roughly 50× the CAPP vocabulary. Test inputs are 200 short prefix strings in the SMILES notation (simplified molecular-input lineentry system) [30], drawn from the ZINC alphabet (single atoms and two-atom fragments, e.g., C, N, CC, c1); the pipeline builds one DTMC per prompt. Extraction parameters are unchanged $( \tau = 0 . 5 \% , \rho = 1 0 ^ { - 4 } , d _ { \operatorname* { m a x } } = 2 0 )$

As a domain specification, we consider the chemical validity as assessed by RDKit’s MolFromSmiles. That is, RDKit is our test oracle [3]: a black-box pass/fail verdict procedure, independent of the SUT, that the pipeline consumes as a PCTL label. This oracle enforces semantic constraints (valence, ring-closure consistency, atom-count balance) that are not part of the SMILES grammar, so structural completion and chemical validity are decoupled. This gap is known for neural SMILES generators, which can emit syntactically complete but chemically invalid strings [17]. A P=?[F valid\_smiles] label replaces the CAPP ordered/misordered pair in Table 1. No other pipeline component changes.

Against CAPP’s 53 tokens and 16-state DTMCs, SMILES uses ≈2,700 BPE tokens and produces DTMCs of mean 13,150 states (max 37,397), with mean extraction 485 s and PRISM 43 s per prompt vs. sub-second for CAPP.

RQ4: Operational Portability All 200 PRISM instances completed without solver failure, matching the 0% error rate on CAPP. The extracted DTMCs are substantially larger than CAPP’s: mean 13,150 states $( \mathrm { p 2 5 } = 1 0 \small { , } 4 1 3$ ; median $= 1 2 , 4 0 8 ; \mathrm { ~ p 7 5 } = 1 6 , 2 8 7 ; \mathrm { ~ m a x } = 3 7 , 3 9 7 )$ , roughly 800× the size of well-trained CAPP DTMCs and 14× the size of the worst-case 1%-trained CAPP DTMCs. Build time is dominated by extraction (mean 485 s, max 1,364 s) rather than verification (mean 43 s, max 370 s); both scale super-linearly with vocabulary size but remain tractable on a laptop-class machine. Still, a full sweep over a ZINC-sized prompt set is beyond the scope of this work.

The pipeline is therefore vocabulary-agnostic: no component assumes a fixed token set or a small state space, and the soundness theorem and CEGAR loop carry over unchanged. Scale also makes the depth limit binding for a non-trivial fraction of prompts: mean $P _ { D } ( \mathrm { T R U N C A T E D } ) ~ = ~ 0 . 0 1 8$ across the 200 inputs, whereas on CAPP it was $\leq 1 0 ^ { - 3 }$ throughout (Table 3). By Theorem 1 this widens the certified soundness interval. The gap here is dominated by the truncated term rather than the difuse low\_prob dispersion of case study 1, so it is the component most reducible by CEGAR or a larger $d _ { \mathrm { m a x } }$

RQ4+RQ1: Specification Informativeness The RDKit specification exposes a failure class that does not have an analog in the $\mathrm { C A P P }$ study. Among the 90 prompts for which at least one chemically valid terminal is produced, mean $P ( \mathrm { v A L I D \_ S M I L E S } ) = 0 . 0 8 4$ while mean $P ( \mathrm { s U C C E S S } ) = 0 . 2 4 8$ . Conditional on a structurally complete terminal, roughly 66% $\left( 1 - 0 . 0 8 4 / 0 . 2 4 8 \right)$ of probability mass corresponds to invalid molecules. At the prompt level, 181 of 200 prompts produce at least one chemically invalid but structurally complete terminal; 87 produce both valid and invalid terminals; 94 produce only invalid structurally complete terminals; and a further 16 produce no structurally complete terminal at all. For 110 prompts in total, no chemically valid SMILES is ever produced.

This is the structural-vs-semantic validity gap that the CAPP study cannot illustrate. In CAPP, P(invalid) ≈ 0 from 5% training onwards: the token grammar happens to encode well-formedness, so the structural-validity check of Sec. 3.3 carries most of the load. In SMILES, structural completion is necessary but not suficient for semantic validity, and the model reliably misallocates confidence to chemically meaningless strings. Test-set accuracy, i.e., exact match against a reference, would not distinguish these cases at all. P(success) alone would give a misleadingly optimistic estimate; the gap between P(success) and P(valid\_smiles) is the formal measure of the discrepancy.

From a methodology viewpoint, the RDKit check is integrated as an external test oracle: integrating it required no change to extraction, PRISM export, or coverage aggregation. This shows the oracle-independence of the methodology in one case, supporting the more general claim of Sec. 1: any executable oracle that maps a terminal sequence to a pass/fail verdict can serve as the domain specification, and the same three artifacts (DTMC abstraction, PCTL specification, population coverage) serve diferent domains with no methodology rework.

## 5 Threats to Validity

Internal. The DTMC under-approximates the SUT, and the soundness interval widens monotonically with τ and contracts with $\rho .$ The CEGAR loop mitigates this by tightening the interval to a target ε. The critical-state gap threshold (10%) is used only as a diagnostic flag and does not enter the soundness theorem; we verified that critical-state rankings are qualitatively stable under gap thresholds in {5%, 10%, 20%} on the only configuration with non-trivial criticalstate mass (1% training).

Construct. The CAPP ordering specification encodes $P  S ^ { * }  F ^ { * }$ and does not capture richer manufacturing constraints (resource contention, tool-change costs, batch-dependent process choice). The SMILES specification is RDKit’s MolFromSmiles, which is the de facto chemical-validity check in the community but is not, strictly speaking, a formal specification. The population coverage assumes that the test split is representative of the deployment distribution; for CAPP the input space is finite and fully enumerable, so coverage on the held-out set is an exact enumerable fraction for any random split. For SMILES, the 200 prompts used here are not a random sample of any deployment distribution, and the coverage is read accordingly: it characterises the gap between P(success) and the specification, not a population coverage guarantee, and we therefore report raw P(success) and P(valid\_smiles) without a population layer.

External. The CAPP and SMILES studies span very diferent vocabularies (53 vs. ≈ 2,700 tokens), DTMC sizes (800×), and specification types (syntactic– temporal vs. semantic–external). This supports a portability claim, i.e., the same pipeline runs on both with no source-code change, but not a generality claim across all autoregressive transformers. We have not yet exercised long-horizon generation (d ≫ 20), non-GPT architectures, or diverse sampling schemes.

## 6 Conclusion

We have presented a probabilistic model-checking pipeline for autoregressive transformers that combines a DTMC abstraction with a soundness theorem, exact PCTL verification, and population coverage over the input space. Where test-set accuracy scores a single greedy trajectory against a reference, it is silent on the probability mass a model places elsewhere and on domain constraints the reference never exercises. Our pipeline is both visible and quantifiable. It recasts the deployment question as whether the gap between a model’s confident output and a domain specification is acceptable, and reports a population coverage curve for it. Its natural place is at development and model-selection time: certified coverage and best-of-N analysis inform the decoder and the training-data budget before deployment, rather than monitoring a fixed model at run time. Two properties make our approach practical. The specification is supplied by the application rather than the method: a temporal ordering property and an external executable oracle plug into the same three artifacts with no pipeline change, so the methodology is indiferent to domain. The soundness guarantee is conservative by construction, so the reported coverage can be trusted to understate rather than overstate the truth. We evaluate this on two contrasting systems, difering by an order of magnitude in vocabulary and by specification type. The results give evidence of portability, though not yet generality across all architectures and decoding schemes. Future work includes longer-horizon generation, richer property classes, and sampling strategies beyond temperature scaling.

Acknowledgments. This work is funded by the European Union under grant agreement number 101091783 (MARS Project).

Disclosure of Interests. The authors declare that they have no competing financial interests.

## References

1. Angelopoulos, A.N., Bates, S., Fisch, A., Lei, L., Schuster, T.: Conformal risk control. In: International Conference on Learning Representations (ICLR) (2024)

2. Bacci, E., Parker, D.: Probabilistic guarantees for safe deep reinforcement learning. In: International Conference on Formal Modeling and Analysis of Timed Systems. pp. 231–248. Springer (2020). https://doi.org/10.1007/978-3-030-57628-8\_14

3. Barr, E.T., Harman, M., McMinn, P., Shahbaz, M., Yoo, S.: The oracle problem in software testing: A survey. IEEE Transactions on Software Engineering 41(5), 507–525 (2015). https://doi.org/10.1109/TSE.2014.2372785

4. Clarke, E., Grumberg, O., Jha, S., Lu, Y., Veith, H.: Counterexampleguided abstraction refinement. In: Proc. 12th Int. Conf. Computer Aided Verification (CAV). LNCS, vol. 1855, pp. 154–169. Springer (2000). https://doi.org/10.1007/10722167\_15

5. Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L., Plappert, M., Tworek, J., Hilton, J., Nakano, R., Hesse, C., Schulman, J.: Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168 (2021). https://doi.org/10.48550/arXiv.2110.14168

6. Cousot, P., Cousot, R.: Abstract interpretation: A unified lattice model for static analysis of programs by construction or approximation of fixpoints. In: Proc. 4th ACM Symp. Principles of Programming Languages (POPL). pp. 238–252. ACM (1977). https://doi.org/10.1145/512950.512973

7. David, A., Larsen, K.G., Legay, A., Mikučionis, M., Poulsen, D.B.: Uppaal SMC tutorial. International Journal on Software Tools for Technology Transfer 17, 397– 415 (2015). https://doi.org/10.1007/s10009-014-0361-y

8. Gordon, A.D., Henzinger, T.A., Nori, A.V., Rajamani, S.K.: Probabilistic programming. In: Proc. Future of Software Engineering (FOSE). pp. 167–181. ACM (2014). https://doi.org/10.1145/2593882.2593900

9. Gross, D., Spieker, H.: Safety-oriented pruning and interpretation of reinforcement learning policies. In: 32nd European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning (ESANN) (2024). https://doi.org/10.14428/esann/2024.ES2024-71

10. Gross, D., Spieker, H.: PCTL model checking for temporal RL policy safety explanations. In: Proceedings of the 40th ACM/SIGAPP Symposium on Applied Computing (SAC). pp. 1514–1521. ACM (2025). https://doi.org/10.1145/3672608.3707759

11. Gross, D., Spieker, H., Gotlieb, A.: Bounded PCTL model checking of large language model outputs. In: 37th IEEE International Conference on Tools with Artificial Intelligence (ICTAI). pp. 106–113. IEEE (2025). https://doi.org/10.1109/ICTAI66417.2025.00022

12. Hansson, H., Jonsson, B.: A logic for reasoning about time and reliability. Formal Aspects Comput. 6(5), 512–535 (1994). https://doi.org/10.1007/BF01211866

13. Holtzman, A., Buys, J., Du, L., Forbes, M., Choi, Y.: The curious case of neural text degeneration. In: International Conference on Learning Representations (ICLR) (2020)

14. Ji, Z., Lee, N., Frieske, R., Yu, T., Su, D., Xu, Y., Ishii, E., Bang, Y.J., Madotto, A., Fung, P.: Survey of hallucination in natural language generation. ACM Computing Surveys 55(12), 1–38 (2023). https://doi.org/10.1145/3571730

15. Katz, G., Barrett, C., Dill, D.L., Julian, K., Kochenderfer, M.J.: Reluplex: An eficient SMT solver for verifying deep neural networks. In: Proc. 29th Int. Conf. Computer Aided Verification (CAV). LNCS, vol. 10426 (2017)

16. Katz, G., Huang, D.A., Ibeling, D., Julian, K., Lazarus, C., Lim, R., Shah, P., Thakoor, S., Wu, H., Zeljic, A., Dill, D.L., Kochenderfer, M.J., Barrett, C.: The marabou framework for verification and analysis of deep neural networks. In: Proc. 31st Int. Conf. Computer Aided Verification (CAV). LNCS, vol. 11561 (2019)

17. Krenn, M., Häse, F., Nigam, A., Friederich, P., Aspuru-Guzik, A.: Selfreferencing embedded strings (SELFIES): A 100% robust molecular string representation. Machine Learning: Science and Technology 1(4), 45024 (2020). https://doi.org/10.1088/2632-2153/aba947

18. Kwiatkowska, M., Norman, G., Parker, D.: PRISM 4.0: Verification of probabilistic real-time systems. In: Proc. 23rd Int. Conf. Computer Aided Verification (CAV). LNCS, vol. 6806 (2011). https://doi.org/10.1007/978-3-642-22110-1\_47

19. Legay, A., Delahaye, B., Bensalem, S.: Statistical model checking: An overview. In: Proc. Runtime Verification (RV). LNCS, vol. 6418, pp. 122–135. Springer (2010). https://doi.org/10.1007/978-3-642-16612-9\_11

20. Liang, P., Bommasani, R., Lee, T., et al.: Holistic evaluation of language models. Transactions on Machine Learning Research (TMLR) (2023)

21. Ma, L., Juefei-Xu, F., Zhang, F., Sun, J., Xue, M., Li, B., Chen, C., Su, T., Li, L., Liu, Y., Zhao, J., Wang, Y.: DeepGauge: Multi-granularity testing criteria for deep learning systems. In: Proc. 33rd IEEE/ACM Int. Conf. Automated Software Engineering (ASE). pp. 120–131. ACM (2018). https://doi.org/10.1145/3238147.3238202

22. Mohri, C., Hashimoto, T.: Language models with conformal factuality guarantees. In: Forty-first International Conference on Machine Learning (ICML) (2024)

23. Pei, K., Cao, Y., Yang, J., Jana, S.: DeepXplore: Automated whitebox testing of deep learning systems. In: Proc. 26th ACM Symp. Operating Systems Principles (SOSP). pp. 1–18. ACM (2017). https://doi.org/10.1145/3132747.3132785

24. Quach, V., Fisch, A., Schuster, T., Yala, A., Sohn, J.H., Jaakkola, T.S., Barzilay, R.: Conformal language modeling. In: The Twelfth International Conference on Learning Representations (ICLR) (2024)

25. Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., Sutskever, I., et al.: Language models are unsupervised multitask learners. OpenAI blog 1(8), 9 (2019)

26. Singh, G., Gehr, T., Püschel, M., Vechev, M.T.: An abstract domain for certifying neural networks. In: Proc. 46th ACM SIGPLAN Symp. Principles of Programming Languages (POPL). pp. 41:1–41:30. ACM (2019). https://doi.org/10.1145/3290354

27. Stathatos, E., Benardos, P., Vosniakos, G., Gross, D., Spieker, H., Gotlieb, A.: Large language models for high-level computer-aided process planning in a distributed manufacturing paradigm. Robotics and Computer-Integrated Manufacturing 100, 103233 (2026). https://doi.org/10.1016/j.rcim.2026.103233

28. Valmeekam, K., Marquez, M., Sreedharan, S., Kambhampati, S.: On the planning abilities of large language models - A critical investigation. In: Advances in Neural Information Processing Systems (NeurIPS). vol. 36 (2023)

29. Wang, S., Zhang, H., Xu, K., Lin, X., Jana, S., Hsieh, C., Kolter, J.Z.: Beta-crown: Eficient bound propagation with per-neuron split constraints for neural network robustness verification. In: Advances in Neural Information Processing Systems (NeurIPS) (2021)

30. Weininger, D.: Smiles, a chemical language and information system. 1. introduction to methodology and encoding rules. J. Chem. Inf. Comput. Sci. 28(1) (1988). https://doi.org/10.1021/CI00057A005

31. Weiss, G., Goldberg, Y., Yahav, E.: Extracting automata from recurrent neural networks using queries and counterexamples. In: International Conference on Machine Learning (ICML). PMLR (2018)