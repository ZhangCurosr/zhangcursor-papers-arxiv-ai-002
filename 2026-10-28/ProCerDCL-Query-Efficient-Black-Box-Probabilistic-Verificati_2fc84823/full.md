# ProCerDCL: Query-Efficient Black-Box Probabilistic Verification of Deep Learning Compiler Optimization

Zexin Li<sup>1,\*,†</sup> iD Weibo Zhang<sup>1,\*,†</sup> iD Linyi Li<sup>2</sup> iD Arvind Easwaran<sup>3</sup> iD Simin Chen<sup>1</sup> iD

<sup>1</sup>STAR Lab, George Mason University <sup>2</sup>Simon Fraser University <sup>3</sup>Nanyang Technological University

## ABSTRACT

Deep learning (DL) compiler optimizations can alter model semantics by changing finite-precision execution, potentially causing decision-level inconsistencies between reference and compiled models. Differential testing can reveal such inconsistencies on selected inputs, but observing consistency on tested cases does not, by itself, bound the probability of inconsistency on unseen inputs. Establishing trust under a specified input distribution therefore requires statistical guarantees that account for sampling uncertainty. Obtaining these guarantees, however, can require many expensive model executions. We present ProCerDCL, a query-efficient framework for black-box probabilistic verification. Given a fixed, sampleable input distribution and a binary property oracle, ProCerDCL assesses the property’s violation probability p against a threshold θ. Under independent sampling and stateless oracle behavior, acceptance asserts $p \leq \theta ,$ rejection asserts p > θ + η for a user-specified separation margin, and an inconclusive verdict indicates insufficient evidence for either conclusion. The probability of each type of erroneous verdict is bounded by its specified error budget. To support this three-valued contract efficiently, ProCerDCL uses a single sequential test, eliminating the interval family required by the baseline. The test accounts for repeated sampling checks through either corrected exact binomial confidence intervals or an anytime-valid confidence sequence. We apply ProCerDCL to probabilistic local model semantic consistency, where the property oracle detects decision-level inconsistencies between reference and compiled executions under a specified perturbation distribution. Our evaluation covers 24 tasks across TorchInductor and TensorRT and examines sampling efficiency, conclusiveness, and diagnostic behavior. For sampling efficiency, we use paired one-model clean-label radius searches: the baseline requires 3.6-4.2 times as many unique samples under matched numerical parameters while providing a weaker decision contract. For model semantic consistency, checks at $\theta = \overset { \cdot } { 1 0 } ^ { - 3 }$ yield 817 acceptances, 19 inconclusive verdicts, and 4 rejections across 840 input–backend instances from 21 tasks. Case studies of two documented compiler faults further demonstrate what these local verdicts can diagnose.

GitHub: https://github.com/STAR-Computing/ProCerDCL

Keywords probabilistic verification; sequential testing; deep learning compilers; local robustness

## 1 Introduction

Deep learning (DL) compilers improve inference efficiency by transforming model execution. Compilers such as TVM [1], TensorRT [2], and TorchInductor [3] apply optimizations including operator fusion and computation reordering. Because floating-point arithmetic is not associative, these transformations can change finite-precision results [4], potentially causing reference and compiled executions to produce different predictions on the same input [5]. We study model semantic consistency at this decision level: whether compilation preserves the reference model’s prediction. This property matters because compilation changes the model’s predictive behavior that was already evaluated by developers. Decision-level inconsistencies can undermine the transfer of that evidence to the compiled model. Assessing model semantic consistency therefore helps developers determine whether efficiency improvements preserve the predictive behavior they intend to deploy.

Differential testing provides evidence about this property by comparing reference and compiled executions on selected inputs. It can reveal individual semantic inconsistencies, but observing consistency on tested cases does not, by itself, establish how reliably consistency holds on unseen inputs. For example, a compiled model may preserve the prediction at a regression input while producing different predictions under unnoticeable perturbations. A developer checking that input therefore faces a broader question: how frequently is model semantic consistency violated under a specified perturbation distribution? A probabilistic specification makes this question precise by defining an input distribution and an acceptable violation probability. Such a specification could support local regression checks or comparisons of affected and patched builds against a fixed reference.

Answering this distributional question requires probabilistic verification that accounts for sampling uncertainty. Even when inputs are sampled independently from the specified distribution, an absence of observed violations need not imply a sufficiently low violation probability. If the true violation probability is $1 \% ,$ for example, 100 independent draws contain no violations with probability $( 1 - 0 . 0 1 ) ^ { 1 0 0 } \approx 3 7 \%$ . Thus, a test campaign with no observed failures can still provide insufficient evidence for a below-1% requirement. A probabilistic verifier must connect the observed outcomes to an explicit bound on the risk of an erroneous verdict and allow an inconclusive result when the evidence is insufficient. Here, probability arises from sampling inputs, even when model executions are deterministic. The resulting guarantee complements differential testing by turning sampled observations into a statistically justified assessment of the specified requirement.

Obtaining sufficient evidence can be computationally expensive. Each semantic-consistency observation requires executing both the reference and compiled models, and small violation thresholds can require many samples even when no inconsistencies are observed. Repeating verification across inputs, compilation backends, or model revisions further increases this cost. Query efficiency is therefore essential to making these guarantees useful within a finite verification budget: the verifier should avoid unnecessary executions while preserving its statistical guarantees.

We propose a query-efficient framework for black-box probabilistic verification of properties of DL compiler optimization. Formally, ProCerDCL verifies the following property. Given a fixed, sampleable distribution D and a binary violation oracle $\psi ,$ let $p = \mathrm { P r } _ { x \sim \mathcal { D } } [ \psi ( x ) = 1 ]$ . For a user-specified threshold $\theta ,$ separation margin $\eta > 0$ , and error budget $\delta ,$ acceptance (YES) asserts $p \leq \theta ,$ rejection (NO) asserts $p > \theta + \eta$ , and NONE indicates insufficient evidence. Under independent sampling and stateless oracle behavior, each direction of erroneous decision has probability at most δ. In $\left( \theta , \theta + \eta \right]$ , either decisive output is an error, so abstention is intended.

Inspired by quantitative verification and statistical model checking [6; 7; 8], ProCerDCL introduces a single sequential verification procedure for the three-valued contract above. The baseline constructs an adaptive interval family, applies fixed-sample Chernoff tests, and distributes its error budget across those tests [6]. Its indifference-zone contract permits either decisive answer within the separation band. Our key insight is that a stricter contract, which treats either decisive answer within that band as erroneous, can be supported directly by a single sequential test, without constructing an interval family or estimating p to a prescribed absolute accuracy. Specifically,

ProCerDCL accepts when a valid upper confidence bound establishes $p \leq \theta ,$ rejects when a valid lower bound establishes $p > \theta + \eta ,$ , and returns NONE if neither condition is established before a finite sampling cap. Using either Clopper–Pearson intervals corrected for repeated checks or an anytime-valid confidence sequence, this design eliminates error-budget allocation across constituent interval tests while preserving error control under repeated checks. Our contribution is the resulting single-test procedure and its direction-specific soundness and finite-cost analysis under the stricter contract.

We apply ProCerDCL to probabilistic local model semantic consistency, checking decision-level disagreement between reference and compiled executions under a specified perturbation distribution. The paired oracles measure either pointwise disagreement or the joint event that the reference retains the seed label while the compiled model leaves it. We also evaluate probabilistic local robustness through one-model clean-label radius searches, which lets us compare the two procedures on the property the baseline was designed for. That oracle compares the compiled prediction on a perturbation with the reference prediction at the seed. It measures model sensitivity and does not bound paired disagreement in general. These properties share the verification procedure but answer distinct questions. A verdict speaks only about the chosen distribution and oracle: an acceptance does not show that the two models agree on every input, and a rejection reveals a disagreement rather than proving a compiler bug.

Our evaluation covers 24 tasks across TorchInductor and TensorRT, examining sampling efficiency, conclusiveness, and behavior on documented faults. In paired one-model cleanlabel radius searches on four settings, the baseline uses 3.6–4.2 times as many unique draws at matched numerical parameters under a weaker decision contract; forced-decision comparisons are reported separately. For two-model checks on the studied ResNet settings, the anytime-valid main test uses 0.58–0.90 of the interval-family mode’s draws. Pointwise consistency checks at $\theta = 1 0 ^ { - 3 }$ return 817 acceptances, 19 inconclusive verdicts, and 4 rejections across 840 input– backend instances from 21 tasks. Two documented compiler faults illustrate local verdicts on affected and patched or workaround builds. Reference-model controls further show that the one-model radius is largely model-dominated: it mainly reflects the model’s own sensitivity to noise, so a small radius does not by itself indicate a compilation error, which is why compiler consistency is assessed with the two-model properties.

The paper makes three contributions:

• Probabilistic Verification of Model Semantic Consistency. We formulate the problem of blackbox verification for model semantic consistency between reference and compiled executions under a specified input distribution, with direction-specific error guarantees.

• Query-Efficient Sequential Verification. We develop a single sequential test that eliminates the baseline’s interval family while enforcing a stricter decision contract, and establish its soundness and finite sampling bound under the stated assumptions.

• Empirical Evaluation Across Models and Compilers. We evaluate sampling efficiency, conclusiveness, and local model semantic consistency across DL tasks and compiler backends, distinguish one-model robustness from paired semantic consistency, and demonstrate the diagnostic scope of local verdicts through two documented compiler faults.

## 2 Background

Deep Learning Compiler. DL compilers [9; 10; 11; 12] such as TVM, XLA, Glow, TorchInductor, and TensorRT translate a model written in a framework such as PyTorch into hardware-efficient kernels for a target platform. As illustrated in Fig. 1, the frontend maps the model to a highlevel intermediate representation (IR) and applies platform-independent transformations such as operator fusion; the backend lowers it to a hardware-oriented IR, applies target-specific optimizations such as vectorization and loop unrolling, and emits machine code. Unlike general-purpose frameworks [13], DL compilers exploit both model structure and hardware characteristics, which makes them the standard path for efficient deployment.

Numerical Errors & Inconsistency in DL

Compilers. Numerical deviations are differences in outputs introduced by floatingpoint arithmetic, compiler optimizations or hardware-specific execution [14; 15; 16; 17; 18; 19; 20; 21]. Because floating-point operations are not associative, optimizations that reorder or regroup them, such as fusion, algebraic simplification, vectorization and loop reordering, can change rounding [4], so a compiled model that is mathematically equivalent to its source can still diverge from it at run time, a phenomenon called optimization incon-

![](images/b96293318698bbcb83f34219e857f20d9f45f7d3b72fab37f7e09497486680e6.jpg)  
Figure 1. Overview of the workflow of DL compilers.

sistency [5; 22]. Floating-point semantics are known to depend on compiler choices [23], and DL compiler testing has found transformations that change model semantics [24].

Probability-threshold verification. Statistical verification uses observations under a specified distribution to assess a quantitative property $\left[ 7 ; 8 ; 6 \right]$ . For a fixed distribution D and binary violation oracle $\psi ,$ define $p = \mathrm { P r } _ { x \sim \mathcal { D } } [ \psi ( x ) = 1 ]$ . If draws are independent and the oracle is a fixed function of its input, the observations $Y _ { j } = \psi ( x _ { j } )$ are i.i.d. Bernoulli(p), and $\begin{array} { r } { k _ { n } = \sum _ { j = 1 } ^ { n } Y _ { j } } \end{array}$ is binomial. Threshold verification asks whether p meets a bound; it need not produce an accurate numerical estimate of $p .$ The reference execution used by an agreement oracle specifies which behavior is compared, not ground-truth correctness of either execution.

Fixed-sample and sequential evidence. Chernoff and Hoeffding bounds [25; 26] give sample budgets for probability estimates or tests. At a fixed sample size $n ,$ the Clopper–Pearson interval $[ L _ { n } , U _ { n } ]$ has binomial coverage at least $1 - \alpha \left[ 2 7 ; 2 8 ; 2 9 \right]$ . Repeatedly checking ordinary intervals at the same level does not automatically preserve that coverage at a data-dependent stopping time. One solution is to preallocate checkpoint errors $\alpha _ { t }$ with $\textstyle \sum _ { t } \alpha _ { t } \leq \delta$ and apply a union bound [30]. Another is an anytime-valid confidence sequence, whose coverage holds simultaneously over its checkpoints. Section 3.1 defines our three-valued contract, and Section 3.4.1 explains how these established tools implement it.

## 3 Approach

## 3.1 Problem Formulation

Inputs and scope. Let M denote a reference DL model and ${ \mathcal { C } } = { \mathrm { C o m p i l e } } ( { \mathcal { M } } )$ its compiled execution, with $\mathcal M ( x )$ and $\mathcal C ( x )$ denoting their predicted labels on input x. Given a fixed, sampleable distribution $\mathcal { D }$ over admissible inputs, we define the binary violation oracle for model semantic consistency as

$$
\psi ( x , \mathcal { C } , \mathcal { M } ) = \mathbf { I } [ \mathcal { C } ( x ) \neq \mathcal { M } ( x ) ] .
$$

The oracle evaluates both executions on the same input and reports a violation exactly when their predictions differ. The target quantity is

$$
\begin{array} { r } { p = \underset { x \sim \mathcal { D } } { \operatorname* { P r } } [ \psi ( { x } , \mathcal { C } , \mathcal { M } ) = 1 ] , } \end{array}
$$

the probability that compilation changes the reference model’s prediction under D.

Each guarantee is specific to the chosen distribution and oracle. This enables developers to assess whether the probability of semantic inconsistency meets an explicit tolerance on inputs relevant to their requirements, such as perturbations around regression inputs or samples from a justified application-input distribution. The guarantee does not establish consistency for every input or global compiler correctness.

Local instantiation. In this paper, we use a local perturbation distribution to assess model semantic consistency around a seed input. For a seed $x _ { 0 }$ and radius $\varepsilon ,$ we define

$$
\begin{array} { r } { \mathcal { D } = \mathrm { U n i f } \big ( B _ { q } ( x _ { 0 } , \varepsilon ) \cap \mathcal { X } \big ) , \qquad q \in \{ 2 , \infty \} , } \end{array}
$$

where $B _ { q } ( x _ { 0 } , \varepsilon )$ is the $\ell _ { q }$ ball centered at $x _ { 0 }$ and $\mathcal { X }$ is the admissible input domain. This neighborhood-based formulation lets developers assess how reliably compilation preserves predictions around a regression input. The framework also supports other input distributions by replacing the sampler, provided that it produces independent draws from the specified distribution. The sampling domain, preprocessing, and execution configuration are fixed as part of the specification; the chosen perturbation distribution need not represent deployment traffic.

Each radius defines a separate input distribution. A radius search therefore reports the largest tested radius certified YES.

Assumptions. Throughout each verification run, we fix the input distribution, models, and execution configurations and make the following assumptions:

A1. Independent sampling. The sampler produces independent draws $x _ { j } \sim \mathcal { D }$ from the same specified distribution. This assumption can be satisfied by independently generating perturbations from a fixed distribution or independently sampling with replacement from a fixed dataset.

A2. Black-box oracle access. The supplied binary violation oracle can evaluate each sampled input. For model semantic consistency, this requires executing M and C on the same input and comparing their predicted labels. This requirement is practical when both implementations are executable, as it requires neither compiler internals nor model gradients.

A3. Fixed, stateless oracle. The oracle returns a fixed binary response for each input, independent of previous evaluations and other inputs evaluated alongside it. This assumption is reasonable for DL inference configured with deterministic execution, fixed preprocessing, and no mutable state or dependence on batch composition.

Together, A1 and A3 imply that $Y _ { j } = \psi ( x _ { j } , { \mathcal { C } } , { \mathcal { M } } )$ are i.i.d. Bernoulli(p) observations, where $p = \operatorname* { P r } _ { x \sim \mathcal { D } } [ \psi ( x ) = 1 ] ; \mathrm { A } 2$ ensures that the verifier can obtain these observations. The verifier consumes only this binary stream. We measure its statistical cost by the number of sampled oracle observations, counting repeated input values as separate draws.

Three-valued verification contract. Given a threshold $\theta \in ( 0 , 1 )$ , a separation margin $\eta > 0$ with $\theta + \eta < 1$ , and a direction-specific error budget $\delta \in ( 0 , 1 )$ , the verifier checks two distinct assertions:

$$
\Upsilon \mathrm { E S } : p \leq \theta , \qquad \mathrm { N O } : p > \theta + \eta .
$$

The outcome NONE reports that neither assertion is supported by the available evidence. A verdict concerns the supplied distribution and oracle, rather than unconditional correctness of the compiled model.

Definition 3.1 (Probabilistic Sound Quantitative Certifier). A verifier $\mathcal { A } ( \mathcal { D } , \psi , \theta , \eta , \delta )$ is sound ${ \mathrm { i f } } ,$ for every fixed distribution and oracle satisfying A1–A3 and every admissible parameter setting,

$$
\operatorname* { P r } \left[ \boldsymbol { A } = \mathrm { Y E S } \wedge \boldsymbol { p } > \boldsymbol { \theta } \right] \leq \delta , \qquad \operatorname* { P r } \left[ \boldsymbol { A } = \mathrm { N O } \wedge \boldsymbol { p } \leq \boldsymbol { \theta } + \boldsymbol { \eta } \right] \leq \delta .\tag{1}
$$

The probabilities are over the random sample stream for the fixed $p .$

Inside $( \theta , \theta + \eta ]$ , either decisive output is erroneous, so abstention is intended, subject to these error bounds. NONE can also occur outside this band. The bounds are frequentist guarantees, not posterior probabilities of individual verdicts, and impose no lower bound on conclusiveness. In contrast, the baseline’s indifference-zone contract permits either decisive verdict inside the band [6].

Notation. Table 1 summarizes the notation used by ProCerDCL.

Table 1. Notation of ProCerDCL.
<table><tr><td>Symbol</td><td>Meaning</td></tr><tr><td> ${ \mathcal { M } } , { \mathcal { C } }$ </td><td>Reference and compiled executions; their outputs denote predicted labels.</td></tr><tr><td> $\mathcal { D }$ </td><td>Fixed, sampleable input distribution.</td></tr><tr><td> $\psi ( x )$ </td><td>Shorthand for the supplied binary oracle  $\psi ( x , { \mathcal { C } } , { \mathcal { M } } ) .$ </td></tr><tr><td> $Y _ { j } , p$ </td><td>Observation  $Y _ { j } = \psi ( x _ { j } )$  and unknown probability  $\begin{array} { r } { p = \operatorname* { P r } ( Y _ { j } = 1 ) . } \end{array}$ </td></tr><tr><td> $\theta , \eta$ </td><td>Acceptance threshold and separation margin; rejection asserts  $p > \theta + \eta .$ </td></tr><tr><td> $\delta$ </td><td>Direction-specific error budget for one verification instance.</td></tr><tr><td> $B$ </td><td>Positive number of observations between statistical checkpoints</td></tr><tr><td> $\mathcal { T }$ </td><td>Optional family of entries  $( \theta _ { 1 , i } , \theta _ { 2 , i } , \mathsf { s i d e } _ { i } ) .$ </td></tr><tr><td> $\ell _ { i } , \delta _ { i }$ </td><td>Test width  $\theta _ { 2 , i } - \theta _ { 1 , i }$  and error budget; the main test uses  $\eta , \delta .$ </td></tr><tr><td> $N _ { i } , T _ { i }$ </td><td>Truncation budget in Eq. (3) and checkpoint count  $\left\lceil N _ { i } / B \right\rceil .$ </td></tr><tr><td> $n _ { t } , k _ { t }$ </td><td>Prefix length tB and violation count at checkpoint t.</td></tr><tr><td> $L _ { i , t } , U _ { i , t }$ </td><td>Lower and upper confidence bounds used by tester i.</td></tr><tr><td> $\tau _ { i }$ </td><td>Prefix length consumed before tester i terminates.</td></tr><tr><td> $\mathcal { P }$ </td><td>Pool of observations for the fixed sampler and oracle, updated by reference.</td></tr><tr><td> $N _ { \mathrm { t o t a l } }$ </td><td>Number of distinct oracle observations consumed by the verifier.</td></tr></table>

## 3.2 Challenges and Key Insights

The verification task in § 3.1 presents unique challenges, and we address each through the following insights.

Reduce executions without sacrificing soundness. The first challenge is to obtain sufficient evidence for a sound verdict without spending unnecessary model executions. Our key insight is that verification requires establishing a threshold assertion, not estimating $p$ to a prescribed absolute accuracy. We therefore examine observations sequentially and stop as soon as the evidence establishes either $p \leq \theta$ or $p > \theta + \eta$ . The sampling effort adapts to the observed evidence, with a finite cap bounding the cost when neither assertion can be established.

Preserve error guarantees under early stopping. Checking the evidence repeatedly creates a second challenge: ordinary fixed-sample confidence intervals do not retain their nominal coverage under repeated inspection. Our key insight is to base stopping decisions on bounds whose error control holds across checkpoints. If these bounds remain valid, every decisive verdict follows from a valid inequality. At checkpoint $t ,$ let $\textstyle \left[ L _ { t } , U _ { t } \right]$ denote the bounds for $p .$ For a test on interval $\left( \theta _ { 1 } , \theta _ { 2 } \right)$ , where $\left( \theta _ { 1 } , \theta _ { 2 } \right) = \left( \theta , \theta + \eta \right)$ in the main mode, we use the stopping rules

$$
U _ { t } \leq \theta _ { 1 } \Rightarrow \mathrm { Y E S } , \qquad L _ { t } > \theta _ { 2 } \Rightarrow \mathrm { N O } .\tag{2}
$$

An erroneous acceptance requires an upper-bound failure, while an erroneous rejection requires a lower-bound failure. Controlling these failures across checkpoints therefore provides direction-specific error guarantees. We implement this control using either Clopper–Pearson intervals corrected across predetermined checkpoints or an anytime-valid confidence sequence. Section 3.4.1 specifies the bounds, sampling cap, and stopping procedure.

Handle insufficient evidence within a single test. The third challenge is that neither decisive assertion holds when $p \in ( \theta , \theta + \eta ]$ , and finite observations may also be insufficient outside this band. Our key insight is to treat abstention as an explicit outcome and use the same observation stream to assess both assertions. A single sequential test checks the upper bound for acceptance and the lower bound for rejection, returning NONE if neither condition holds by the cap. This directly supports the three-valued contract without separate tests over an interval family, while retaining the prescribed direction-specific error budget δ across repeated checks.

## 3.3 Verification Procedure and Guarantees

ProCerDCL takes the sampler and binary oracle from Section 3.1, together with user provided $( \theta , \eta , \delta )$ and a positive checkpoint batch size B. A checkpoint follows every B oracle observations; B describes the statistical checking schedule and need not equal the runtime’s GPU batch size. We present the main single-test procedure first, followed by the optional interval-family mode.

## 3.3.1 Main algorithm

The main mode invokes Algorithm 2 once with $\big ( \theta _ { 1 } , \theta _ { 2 } , \delta _ { i } , \mathcal { P } \big ) = \big ( \theta , \theta + \eta , \delta , \big [ \big ] \big )$ , so every observation is fresh and no inter-test reuse occurs. The tester draws observations, updates the violation count, and applies $\operatorname { E q . }$ (2) at each checkpoint. A decisive result establishes the corresponding probability assertion under the error guarantee; exhausting the cap produces NONE. The pool is a shared interface used by both modes. In the main algorithm it supplies fresh observations and performs no inter-test reuse.

Theorem 3.1 (Fixed-distribution Soundness and Query Cap). Under $A 1 { - } A 3$ , for admissible $( \theta , \eta , \delta )$ and integer $B \geq 1$ , this single-test instantiation of Algorithm 2 satisfies:

(a) Soundness. For every fixed $p ,$

$$
\operatorname* { P r } [ o u t p u t = \Upsilon \mathrm { E S } ~ \land ~ p > \theta ] \leq \delta ,
$$

$$
\operatorname* { P r } [ o u t p u t = \mathrm { N O } \land p \leq \theta + \eta ] \leq \delta .
$$

The probability of NONE is not bounded by this contract.

(b) Observation cap. The number of oracle observations is at most $\left\lceil N / B \right\rceil B _ { \iota }$ , where N is Eq. (3) evaluated at $\left( \theta , \theta + \eta , \delta \right)$ . For the consistency oracle, these observations require at most $2 \lceil N / B \rceil B$ model executions, excluding additional runtime buffering.

Proof. Apply Lemma 3.4 with $\left( \theta _ { 1 , 1 } , \theta _ { 2 , 1 } , \delta _ { 1 } \right) = \left( \theta , \theta + \eta , \delta \right)$ . Each observation is fresh because the pool is initially empty and there is only one tester. The cap follows from Eq. (4). The CS variant follows in the same way from its anytime-valid error bound, with the same truncation cap.

The theorem bounds cost and erroneous decisions for one fixed distribution and oracle.

## 3.3.2 Optional interval-family mode

The compatibility mode combines four modules: M1 constructs a deterministic interval family; M2 assigns its error budgets; M3 is the sequential tester shared with the main algorithm; and M4 stores observations for reuse. Only this mode needs M1 and M2 or reuses observations across tests. Algorithm 1 constructs the complete family and allocates its budgets before observing any oracle responses. It then visits the intervals in order, with each tester reading from the beginning of the same stream. A left interval can terminate the procedure only with YES, a right interval only with NO, and the final interval $\left( \theta , \theta + \eta \right)$ returns any of the three outcomes.

Proposition 3.2 (Interval-family mode). Under $A 1 { - } A 3 ,$ Algorithm 1 satisfies Definition 3.1. Its number of distinct oracle observations is

$$
N _ { \mathrm { t o t a l } } = \operatorname* { m a x } _ { i \in \mathcal { V } } \tau _ { i } \le \operatorname* { m a x } _ { i \in \mathcal { V } } \lceil N _ { i } / B \rceil B ,
$$

where $\nu$ is the set of visited intervals, $\tau _ { i }$ is tester i’s consumed prefix length, and $N _ { i }$ is its cap from Eq. (3).

Algorithm 1: AdaptiveBinPCertifyWithReuse   
Input : Sampler for D and fixed binary oracle ψ; $\theta \in ( 0 , 1 ) , \eta > 0$ with $\theta + \eta < 1 , \delta \in ( 0 , 1 )$   
$B \in \mathbb { N } _ { \geq 1 }$   
Output :verdict ∈ {YES, NO, NONE}   
1 $\mathcal { T }  [ ] , ( \theta _ { 1 l } , \theta _ { 2 l } )  ( 0 , 0 ) , ( \theta _ { 1 r } , \theta _ { 2 r } )  ( 0 , 0 )$   
2 while true do   
3 $( \theta _ { 1 l } , \theta _ { 2 l } ) \Leftarrow$ CREATEINTERVAL $( \theta , \theta _ { 1 l } , \theta _ { 2 l } , \eta ,$ , LEFT)   
4 if $\theta _ { 2 l } - \theta _ { 1 l } > \eta$ then   
5 append $\left( \theta _ { 1 l } , \theta _ { 2 l } , \mathrm { L E F T } \right)$ to I   
6 $( \dot { \theta } _ { 1 r } , \theta _ { 2 r } ) \Leftarrow 0$ CREATEINTERVAL $\left( \theta , \theta _ { 1 r } , \theta _ { 2 r } , \eta , \right.$ RIGHT)   
7 if $\theta _ { 2 r } - \theta _ { 1 r } > \eta$ then   
8 append $\left( \theta _ { 1 r } , \theta _ { 2 r } , \right.$ RIGHT) to I   
9 if $\theta _ { 2 l } - \theta _ { 1 l } \leq \eta$ and $\theta _ { 2 r } - \theta _ { 1 r } \leq \eta$ then   
10 append (θ, θ + η, FINAL) to I   
11 break   
12 $\begin{array} { r } { \dot { m }  \vert \mathbb { Z } \vert , \{ \delta _ { i } \} _ { i = 1 } ^ { m }  \mathrm { A L L O C A T E } ( \mathbb { Z } , \delta ) , \mathcal { P }  [ ] } \end{array}$   
13 for i = 1 to m do   
14 $( \theta _ { 1 , i } , \theta _ { 2 , i } , \mathsf { s i d e } _ { i } ) \gets \mathcal { T } [ i ]$   
15 r ⇐ SEQUENTIALTEST $( \mathcal { D } , \psi , \theta _ { 1 , i } , \theta _ { 2 , i } , \delta _ { i } , B , \mathcal { P } )$   
16 if side = LEFT and r = YES then   
17 return YES   
18 if side = RIGHT and r = NO then   
19 return NO   
20 if side = FINAL then   
21 return r   
22 return NONE

Soundness follows from per-tester error control (Lemma 3.4), side-specific propagation and budget allocation (Lemma 3.6), and the fixed-stream reuse argument (Lemma 3.7). The pool grows only when a tester needs a longer prefix, giving the maximum rather than the sum of consumed prefix lengths. Section 3.4 supplies these details. Both modes retain the three-valued contract; the forced-decision variants evaluated separately satisfy only the baseline’s weaker contract.

## 3.4 Sequential Test and Interval-Family Details

We first specify the core sequential tester (M3) and its error and cost guarantees. We then describe the scheduler (M1), allocator (M2), and observation reuse (M4) used by the optional interval-family mode. Throughout, ψ(x) is the fixed binary oracle supplied to the verification instance.

## 3.4.1 Sequential tester (M3)

Cap and checkpoints.. For thresholds $0 \leq \theta _ { 1 , i } < \theta _ { 2 , i } \leq 1 .$ , let $\ell _ { i } = \theta _ { 2 , i } - \theta _ { 1 , i }$ and $\delta _ { i } \in ( 0 , 1 )$ . We retain Chernoff-derived sample budget as a finite truncation cap:

$$
N _ { i } = \left\lceil \frac { \left( \sqrt { 3 \theta _ { 1 , i } } + \sqrt { 2 \theta _ { 2 , i } } \right) ^ { 2 } } { \ell _ { i } ^ { 2 } } \ln \frac { 1 } { \delta _ { i } } \right\rceil .\tag{3}
$$

The main algorithm uses $\ell _ { 1 } = \eta$ and $\delta _ { 1 } = \delta .$ . The interval-family mode uses the allocation in Eq. (5). The budget scales as $O ( \log ( 1 / \delta _ { i } ) / \ell _ { i } ^ { 2 } )$ ; its role here is to bound cost, not to guarantee a decisive verdict from the sequential rule.

For integer $B \geq 1$ , set $T _ { i } = \lceil N _ { i } / B \rceil$ before sampling. Checkpoints occur at prefix lengths $n _ { t } = t B ,$ , for $t = 1 , \ldots , T _ { i }$ . Rounding to complete batches permits at most $B - 1$ observations beyond $N _ { i }$ .

Confidence bounds and stopping.. At checkpoint t, let $\begin{array} { r } { k _ { t } = \sum _ { i = 1 } ^ { n _ { t } } Y _ { j } } \end{array}$ . The CP tester computes the exact two-sided Clopper–Pearson interval $\left[ L _ { i , t } , U _ { i , t } \right]$ from $( \dot { k } _ { t } , n _ { t } )$ [27], using total noncoverage budget $\delta _ { i , t } = \delta _ { i } / T _ { i }$ and tail budgets $\delta _ { i , t } / 2$ . The usual boundary conventions are $L _ { i , t } = 0$ when $k _ { t } = 0$ and $U _ { i , t } = 1$ when $k _ { t } = n _ { t }$ . It applies Eq. (2) and returns NONE if neither condition holds at the last checkpoint.

Corollary 3.3 (Sequential Union Bound). For fixed thresholds and $T _ { i } ,$ suppose the interval at each predetermined checkpoint has noncoverage probability at most $\delta _ { i } / T _ { i }$ . Then

$$
\operatorname* { P r } \left[ \bigcup _ { t = 1 } ^ { T _ { i } } \{ p \notin \left[ L _ { i , t } , U _ { i , t } \right] \} \right] \le \sum _ { t = 1 } ^ { T _ { i } } \frac { \delta _ { i } } { T _ { i } } = \delta _ { i } .
$$

No independence between checkpoint intervals is required.

Procedure.. Algorithm 2 initializes the count, sample size, and pool cursor, then obtains B observations per iteration through FETCHBATCH. In the main algorithm, all observations are fresh. In the interval-family mode, a tester first consumes any cached prefix and extends it only as needed. The pool is associated with one fixed sampler and oracle and is updated by reference. Each tester starts its own cursor at position 1, so reuse never duplicates an observation within that tester’s prefix.

Algorithm 2: SequentialTest   
Input : Sampler for D and fixed oracle ψ; $0 \leq \theta _ { 1 } < \theta _ { 2 } \leq 1 , \delta _ { i } \in ( 0 , 1 ) , B \geq 1 ,$ , pool P   
Output : verdict ∈ {YES, NO, NONE}   
1 N<sub>i</sub> ⇐ Eq. (3) for $\left( \theta _ { 1 } , \theta _ { 2 } , \delta _ { i } \right)$   
2 $T _ { i } \Leftarrow \lceil N _ { i } / B \rceil , \delta _ { i , t } \Leftarrow \delta _ { i } / T _ { i }$   
3 $n \Leftarrow 0 , k \Leftarrow 0 , c \Leftarrow 1$   
4 for t = 1 to T do   
5 (batch, c) ⇐ FETCHBATCH(D, ψ, P, B, c)   
6 $n  n + B , k  k + \sum _ { Y \in \mathrm { b a t c h } } Y$   
7 [L<sub>i,t</sub>, U<sub>i,t</sub>] ⇐ CLOPPERPEARSON $( k , n , \delta _ { i , t } )$   
8 if $U _ { i , t } \leq \theta _ { 1 }$ then   
9 return YES   
10 if $L _ { i , t } > \theta _ { 2 }$ then   
11 return NO   
12 return NONE

Lemma 3.4 (Error Control of the Sequential Tester). Under A1–A3, for fixed thresholds $0 \leq \theta _ { 1 , i } <$ $\theta _ { 2 , i } \leq 1$ , budget $\delta _ { i } \in ( 0 , 1 )$ , and $B \geq 1$ , Algorithm 2 applied to an i.i.d. Bernoulli(p) stream satisfies

$$
\operatorname* { P r } [ o u t p u t = \Upsilon \mathrm { E S } \mathrm { ~ \wedge ~ } p > \theta _ { 1 , i } ] \leq \delta _ { i } ,
$$

$$
\operatorname* { P r } [ o u t p u t = \mathrm { N O } \mathrm { ~ \land ~ } p \le \theta _ { 2 , i } ] \le \delta _ { i } .
$$

It may return NONE at truncation, both inside and outside $\left( \theta _ { 1 , i } , \theta _ { 2 , i } \right]$

Proof. Consider the entire i.i.d. stream, including observations that would remain unread after early stopping. Its checkpoint intervals have simultaneous coverage except on an event of probability at most $\delta _ { i }$ by Corollary 3.3. A wrong YES requires $U _ { i , t } \ \leq \ \theta _ { 1 , i } \ < \ p$ at a visited

checkpoint, which is a coverage failure. A wrong NO requires $p \leq \theta _ { 2 , i } < L _ { i , t }$ , also a coverage failure. Both error events are contained in the union of checkpoint failures. Returning NONE asserts neither inequality and incurs no error under the contract. □

Cost and inconclusiveness.. Let $\tau _ { i }$ be the number of observations consumed by tester i. The loop executes at most $T _ { i }$ iterations, hence

$$
\tau _ { i } \leq T _ { i } B = \lceil N _ { i } / B \rceil B .\tag{4}
$$

If the bounds cross a stopping threshold early, the tester uses fewer observations. Near or inside the separation band, neither condition may hold before the cap. Splitting $\delta _ { i }$ over checkpoints produces wider CP intervals than using $\delta _ { i }$ at a single fixed checkpoint, which can increase abstention. Soundness alone therefore does not establish query savings or conclusiveness; both are measured in the evaluation.

Anytime-valid and forced-decision variants.. The CS variant replaces checkpoint-corrected CP intervals with a Beta $\left( { \frac { 1 } { 2 } } , { \frac { 1 } { 2 } } \right)$ )-mixture confidence sequence. It uses the same $\delta _ { i }$ at every checkpoint and the same cap, with error control supplied by Ville’s inequality. The forced-decision variants (ProCerDCL-F and ProCerDCL-CS-F) instead reserve part of the budget for a fixed-sample decision at truncation. They do not satisfy Definition 3.1: their fallback permits decisive answers inside the separation band. They satisfy only the weaker indifference-zone guarantee.

## 3.4.2 Interval scheduling (M1, optional)

The scheduler constructs the full interval family before sampling. Starting from $( 0 , \theta )$ on the left and $\left( \theta + \eta , 1 \right)$ on the right, CREATEINTERVAL contracts each side toward its fixed endpoint until its width reaches the floor η (Algorithm 3). Algorithm 1 appends only intermediate intervals wider than $\eta ,$ alternating left and right, then appends $\left( \theta , \theta + \eta \right)$ as the final test. Left intervals have upper endpoint $\theta ;$ right intervals have lower endpoint $\theta + \eta$

Algorithm 3: CreateInterval   
Input : θ, (θ , θ ), η, s ∈ {LEFT, RIGHT}   
Output : (θ<sup>′</sup> , θ<sup>′</sup> )   
1 if $\theta _ { 1 } = 0$ and θ = 0 and s = LEFT then   
2 return (0, θ)   
3 if $\theta _ { 1 } = 0$ and θ = 0 and s = RIGHT then   
4 return $\left( \theta + \eta , 1 \right)$   
5 $\dot { \alpha } \Leftarrow \theta _ { 2 } - \theta _ { 1 }$   
6 if $\alpha \leq \eta$ then   
7 return $\left( \theta _ { 1 } , \theta _ { 2 } \right)$   
8 if s = LEFT then   
9 return (θ<sub>2</sub> − max(η, α/2), θ<sub>2</sub>)   
10 else   
11 return $( \theta _ { 1 } , \ \theta _ { 1 } + \mathrm { { m a x } } ( \eta , \alpha / 2 ) )$

Lemma 3.5 (Interval Width Contraction and Termination). For admissible $\theta , \eta ,$ , the two side sequences reach width at most η after finitely many calls. The complete family has size

$$
m \leq 3 + \operatorname* { m a x } \left( 0 , \left\lceil \log _ { 2 } \frac { \theta } { \eta } \right\rceil \right) + \operatorname* { m a x } \left( 0 , \left\lceil \log _ { 2 } \frac { 1 - \theta - \eta } { \eta } \right\rceil \right) .
$$

Proof. After a side is initialized with width $w ,$ each call with $w > \eta$ changes it to max $( \eta , w / 2 ) ,$ ; a width at most η remains unchanged. If $w > \eta ,$ , at most $\lceil \log _ { 2 } ( w / \eta ) \rceil$ contractions reach the floor.

Otherwise, the side already meets the stopping condition. Counting the two initial intervals, these calls, and the final interval gives the stated conservative bound; intermediate intervals at or below the floor are omitted. □

## 3.4.3 Inverse-width allocation (M2, optional)

Narrow intervals have larger caps through the $1 / \ell _ { i } ^ { 2 }$ factor in Eq. (3). The allocator assigns them a larger error budget using deterministic inverse-width weights:

$$
w _ { i } = 1 / \ell _ { i } , \qquad \delta _ { i } = \delta \frac { w _ { i } } { \sum _ { j = 1 } ^ { m } w _ { j } } .\tag{5}
$$

Algorithm 4 normalizes over the complete family, including intervals that may never be visited. Thus every threshold, budget, cap, and checkpoint schedule is fixed before observations are read. Inverse-width allocation is a heuristic for cost; soundness requires only positive budgets whose sum is at most δ.

Algorithm 4: Allocate   
Input $: \mathcal { T } = \{ ( \theta _ { 1 , i } , \theta _ { 2 , i } , \mathsf { s i d e } _ { i } ) \} _ { i = 1 } ^ { m } , \delta$   
Output : $\{ \delta _ { i } \} _ { i = 1 } ^ { m }$   
1 for each $i \in \{ 1 , \ldots , m \}$ do   
2 $\ell _ { i } \Leftarrow \theta _ { 2 , i } - \theta _ { 1 , i } , w _ { i } \Leftarrow 1 / \ell _ { i }$   
3 $\begin{array} { r } { \dot { W } \Leftarrow \sum _ { j = 1 } ^ { m } w _ { j } } \end{array}$   
4 for each $i \in \{ 1 , \ldots , m \}$ do   
5 $\delta _ { i } \Leftarrow \delta \cdot w _ { i } / W$   
6 return $\{ \delta _ { i } \} _ { i = 1 } ^ { m }$

Lemma 3.6 (Soundness of Budget Allocation). If every fixed tester has the direction-specific error bounds of Lemma 3.4 with budget $\delta _ { i } ,$ then Algorithm 1, using Eq. (5), satisfies Definition 3.1.

Proof. A global YES comes from a left or final interval, whose lower threshold satisfies $\theta _ { 1 , i } \leq \theta$ If $p > \theta ,$ that output is also a local YES error. A global NO comes from a right or final interval, whose upper threshold satisfies $\theta _ { 2 , i } \geq \theta + \eta$ . If $p \leq \theta + \eta .$ , that output is also a local NO error. Each global directional error event is therefore contained in the union of the corresponding local error events over the fixed family. Its probability is at most $\Sigma _ { i } \delta _ { i } = \delta ,$ without requiring independence between testers. □

Remark 3.1 (Role of the interval family). The family divides δ among multiple tests, and its nonfinal tests can propagate only one direction of verdict. It is not guaranteed to outperform the single test. For example, a CP test on (0, θ) cannot accept after finitely many observations because its upper confidence bound remains positive. The optional mode enables the evaluation to isolate interval scheduling, allocation, and reuse; it is not needed for the main soundness theorem.

## 3.4.4 Observation reuse (M4, optional)

FETCHBATCH serves observations from one shared stream (Algorithm 5). If the caller’s cursor is within the pool, it returns the cached entry; otherwise it draws $x \sim \mathcal { D } ,$ , evaluates $\psi ( x )$ , and appends the response. It advances the caller’s cursor after each entry. Reuse is permitted only within the same fixed distribution, oracle, model pair, and execution configuration. Changing any of these requires a fresh pool.

Algorithm 5: FetchBatch   
Input : Sampler for D and fixed oracle ψ; shared pool P, $B \geq 1 .$ , caller cursor c   
Output :batch of B observations, updated cursor c (and updated P)   
1 batch ⇐ [ ]   
2 for j = 1 to B do   
3 if $c \leq | \mathcal { P } |$ then   
4 $Y \Leftarrow \mathcal { P } [ c ]$   
5 else   
6 $x \sim \mathcal { D }$   
7 $Y \Leftarrow \psi ( x )$   
8 append Y to $\mathcal { P }$   
9 append Y to batch   
10 $c \Leftarrow c + 1$   
11 return (batch, c)

Lemma 3.7 (Soundness of Sample Reuse). Under A1–A3, consider the shared stream of i.i.d. Bernoulli(p) observations. Fix thefull intervalfamily and all tester parameters before reading it. Each tester, considered as a function of this stream whether or not it is visited, retains the marginal error bounds of Lemma 3.4. Consequently, the interval-family verifier retains direction-specific error bounds of δ under reuse.

Proof. For each fixed tester, its count, checkpoint bounds, and stopping decision depend on a prefix of the same i.i.d. stream that an isolated run would receive. Its marginal error law is therefore unchanged. The decision to visit a tester can depend on earlier outcomes, so its error probability conditional on being visited need not be bounded by $\delta _ { i } .$ . Instead, a wrong verdict propagated by the verifier is contained in the unconditional error event of that fixed tester. The union bound over all such events in the predetermined family applies by Lemma 3.6; dependence among testers does not invalidate it. Since each tester starts at the beginning, the pool ends at the longest prefix consumed:

$$
N _ { \mathrm { t o t a l } } = \operatorname* { m a x } _ { i \in \mathcal { V } } \tau _ { i } \le \operatorname* { m a x } _ { i \in \mathcal { V } } \lceil N _ { i } / B \rceil B .
$$

口

## 4 Evaluation

We conduct empirical evaluation and seek to address the following questions. Throughout, ProCerDCL is distribution-aware: every verdict is a statement about a named input distribution and oracle, and changing the distribution changes only the sampler, not the verifier or its guarantee. We exercise four distribution families, uniform $L _ { 2 }$ and $L _ { \infty }$ balls in normalized input space (RQ1–RQ3), fixed-length token substitutions for text (RQ1) and a fixed-support image patch (RQ5), and report each result together with the distribution it holds for.

RQ 1 Properties and verdicts: What do consistency checks report on the evaluated builds and documented fault cases, and what does the clean-label robustness radius measure?

RQ 2 Sampling cost and conclusiveness: How does the single test compare with PROVERO and interval-family variants, accounting for their decision contracts and inconclusive outcomes?

RQ 3 Robustness across configurations: How does the one-model instantiation behave under quantization, a hardware change, and an $L _ { \infty }$ perturbation distribution?

RQ 4 Parameter sensitivity: How do threshold, margin, and error budget affect the measured robustness radii and sampling cost?

RQ 5 Other distributions: Does ProCerDCL apply to perturbation distributions that are not norm-bounded balls around the input?

## 4.1 Experimental Setup

Subjects. We sweep 24 image (model, dataset) tasks: seven convolutional architectures, LeNet-5 [31], ResNet-18/34/50 [32], VGG-16-BN [33], DenseNet-121 [34] and MobileNetV2 [35], and three vision transformers, ViT-B/16 [36], an in21k-finetuned ViT-B/16 and DeiT-Small [37], on MNIST [31], CIFAR-10/100 [38], SVHN [39] and ImageNet-1k [40], at native resolution (28 × 28 to 224 × 224) with 60K to 80M parameters; three text classifiers are reported separately (Section 4.2). Following PROVERO’s protocol, perturbations are added in normalized input space without clipping, whereas the convolutional attack clamps to the seed’s value range, so the radius is a noise-robustness radius read relative to the input norm, and RQ1 compares how verifier and attack order inputs rather than their magnitudes. Clamping the verifier’s samples as the attack does leaves this picture intact: agreement with the attack moves from 0.71/0.82 to 0.70/0.81 (Pearson/Spearman) on ResNet-18/CIFAR-10 and from 0.93/0.95 to 0.86/0.91 on LeNet-5/MNIST, whose clamped radii are 1.55 times larger because clamping cancels half of the perturbation on its minimum-valued background. Since a high-dimensional uniform ball concentrates near its boundary, the certificate concerns random directions of norm close to $\varepsilon ,$ one reason certified radii exceed attack budgets several-fold.

Robustness-search qualification. For the one-model radius experiment, a task qualifies if, on both backends, at least 90 of its 100 verified inputs return a conclusive certificate, one whose radius is positive and at most twice the input norm. A fixed absolute bound would be dimensionblind: a bound of 100 would discard the 224 × 224 subjects although they receive ordinary finite radii, admitting only 16 of 24 tasks.

Baseline. To the best of our knowledge, ProCerDCL is the first framework to provide probabilistic verification of model semantic consistency under DL compilation. As we are not aware of an off-the-shelf method targeting this setting and contract, we adapt PROVERO [6], an existing probabilistic verification method for DL models, to the same compiler-verification task. PROVERO decides whether a violation probability lies below a threshold by testing an adaptive family of nested intervals around it, each with a fixed-sample Chernoff tester and a share of the error budget, and it always returns YES or NO, which gives the weaker indifference-zone contract. We run it with the same oracle, sampler, threshold, tolerance and confidence as ProCerDCL, and we explicitly account for differences in decision contracts when comparing the methods.

Verification protocol. Unless stated otherwise, θ = 1%, $\eta \ : = \ : 0 . 5 \%$ and $\delta = 1 \%$ . The tester checks its stopping rule after every draw $( B = 1 )$ , while the oracle evaluates perturbations in GPU batches of 256. A search probes $\varepsilon = 0 ,$ , which must return YES, doubles from 0.1 until the first non-YES probe or the cap $4 \| x _ { 0 } \| _ { 2 } ,$ and bisects to a resolution of 0.1. Samplers use seed 123 (except in the seed study of Section 4.3), with one search per input. The certified property is that of the compiled model at batch size 256. To check the oracle assumptions, we evaluated 256 perturbations of each of up to 100 cohort inputs on the four representative settings, about $1 0 ^ { 5 }$ perturbations in total, under four execution contexts. Re-executing the same batch ten times, permuting positions within the batch, and replacing half of the batch with another input’s perturbations never changed a decision. Evaluating each perturbation alone at batch size 1 changed 4 of 25,088 decisions on LeNet-5/MNIST and 1 of 25,600 on ResNet-34/SVHN under TensorRT, whose engines select kernels per batch size, and none under Inductor. These checks support deterministic, batch-composition-independent decisions on the audited inputs; they do not establish A3 for every execution. The certified density belongs to the batch size used, and the observed batch-1 differences are of order $1 0 ^ { - 4 }$

Table 2. Full disclosure of the RQ1 sweep: every swept (model, dataset) task per compiler backend. The three text subjects are certified over a token-substitution set. Mean rows give unweighted / Fisher-z averages of r.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Dataset</td><td colspan="3">inductor</td><td rowspan="2"></td><td colspan="3">tensorrt</td></tr><tr><td> $n _ { \mathrm { c } }$ </td><td>r</td><td> $\rho$ </td><td> $n _ { \mathrm { c } }$ </td><td>r</td><td>ρ</td></tr><tr><td>LeNet-5</td><td>MNIST</td><td>100/100</td><td>0.85</td><td>0.90</td><td></td><td>100/100</td><td>0.93</td><td>0.95</td></tr><tr><td>ResNet-18</td><td>CIFAR-10</td><td>100/100</td><td>0.71</td><td>0.82</td><td>99/100</td><td></td><td>0.58</td><td>0.65</td></tr><tr><td>ResNet-50</td><td>ImageNet-1k</td><td>100/100</td><td>-0.01</td><td>-0.05</td><td>100/100</td><td></td><td>0.05</td><td>0.00</td></tr><tr><td>MobileNetV2</td><td>CIFAR-10</td><td>99/100</td><td>0.05</td><td>0.11</td><td></td><td>99/100</td><td>0.17</td><td>0.24</td></tr><tr><td>ResNet-18</td><td>CIFAR-100</td><td>98/100</td><td>0.67</td><td>0.62</td><td></td><td>99/100</td><td>0.70</td><td>0.68</td></tr><tr><td>ResNet-18</td><td>SVHN</td><td>100/100</td><td>0.88</td><td>0.90</td><td></td><td>100/100</td><td>0.83</td><td>0.85</td></tr><tr><td>ResNet-18</td><td>ImageNet-1k</td><td>100/100</td><td>-0.07</td><td>-0.07</td><td></td><td>100/100</td><td>0.20</td><td>0.17</td></tr><tr><td>ResNet-50</td><td>CIFÅR-100</td><td>99/100</td><td>0.62</td><td>0.67</td><td></td><td>100/100</td><td>0.66</td><td>0.65</td></tr><tr><td>ResNet-50</td><td>SVHN</td><td>100/100</td><td>0.78</td><td>0.83</td><td></td><td>100/100</td><td>0.78</td><td>0.82</td></tr><tr><td>ResNet-34</td><td>CIFAR-10</td><td>100/100</td><td>0.61</td><td>0.71</td><td></td><td>100/100</td><td>0.62</td><td>0.68</td></tr><tr><td>ResNet-34</td><td>CIFAR-100</td><td>98/100</td><td>0.44</td><td>0.48</td><td></td><td>100/100</td><td>0.61</td><td>0.58</td></tr><tr><td>ResNet-34</td><td>SVHN</td><td>100/100</td><td>0.75</td><td>0.86</td><td></td><td>100/100</td><td>0.78</td><td>0.86</td></tr><tr><td>ResNet-50</td><td>CIFAR-10</td><td>100/100</td><td>0.32</td><td>0.69</td><td></td><td>100/100</td><td>0.47</td><td>0.56</td></tr><tr><td>VGG-16-BN</td><td>CIFAR-10</td><td>93/100</td><td>0.37</td><td>0.71</td><td></td><td>94/100</td><td>0.19</td><td>0.64</td></tr><tr><td>VGG-16-BN</td><td>CIFAR-100</td><td>99/100</td><td>0.62</td><td>0.63</td><td></td><td>99/100</td><td>0.36</td><td>0.56</td></tr><tr><td>VGG-16-BN</td><td>SVHN</td><td>100/100</td><td>0.82</td><td>0.85</td><td></td><td>100/100</td><td>0.76</td><td>0.81</td></tr><tr><td>DenseNet-121</td><td>CIFAR-10</td><td>98/100</td><td>0.62</td><td>0.64</td><td></td><td>93/100</td><td>0.67</td><td>0.66</td></tr><tr><td>DenseNet-121</td><td>CIFAR-100</td><td>97/100</td><td>0.48</td><td>0.42</td><td></td><td>99/100</td><td>0.40</td><td>0.48</td></tr><tr><td>DenseNet-121</td><td>SVHN</td><td>100/100</td><td>0.62</td><td>0.84</td><td></td><td>100/100</td><td>0.60</td><td>0.82</td></tr><tr><td>ViT-Base/16</td><td>CIFAR-10</td><td>92/100</td><td>0.63</td><td>0.73</td><td></td><td>97/100</td><td>0.55</td><td>0.66</td></tr><tr><td>DeiT-Small</td><td>ImageNet-1k</td><td>99/100</td><td>0.77</td><td>0.84</td><td></td><td>99/100</td><td>0.77</td><td>0.84</td></tr><tr><td>ViT-B/16 in21k-ft</td><td>CIFAR-10</td><td>94/100</td><td>0.60</td><td>0.68</td><td></td><td>94/100</td><td>0.60</td><td>0.68</td></tr><tr><td>ViT-B/16 in21k-ft</td><td>CIFAR-100</td><td>100/100</td><td>0.56</td><td>0.66</td><td></td><td>100/100</td><td>0.56</td><td>0.66</td></tr><tr><td>ViT-B/16 in21k-ft</td><td>SVHN</td><td>99/100</td><td>0.61</td><td>0.67</td><td></td><td>99/100</td><td>0.61</td><td>0.67</td></tr><tr><td>BERT-base</td><td>SST-2</td><td>82/100</td><td>0.80</td><td>0.75</td><td></td><td>42/81</td><td>0.62</td><td>0.73</td></tr><tr><td>BERT-base</td><td>AG News</td><td>55/64</td><td>0.59</td><td>0.73</td><td></td><td>29/69</td><td>0.33</td><td>0.26</td></tr><tr><td>DistilBERT</td><td>SST-2</td><td>71/94</td><td>0.56</td><td>0.68</td><td></td><td>70/92</td><td>0.52</td><td>0.70</td></tr><tr><td>Mean, qualified</td><td></td><td>0.55 / 0.60</td><td>0.63</td><td></td><td></td><td>0.56 / 0.60</td><td>0.63</td><td></td></tr><tr><td>Mean, all cells</td><td></td><td>0.56 / 0.61</td><td>0.64</td><td></td><td>0.55 /0.60</td><td></td><td>0.62</td><td></td></tr></table>

Compilers and Hardware. We evaluate TorchInductor [3], a JIT compiler built on TorchDynamo and Triton, with max\_autotune and Triton backends, and TensorRT [2], a vendor ahead-of-time optimizer, with FP32 engines. Attack success depends strongly on the Inductor configuration, 71% and 84% under its default cuBLAS/cuDNN kernels against 31% and 21% under Triton autotune, while the certificates agree within one grid step on 20 of 20 inputs, as expected of a model-dominated radius. Experiments run on a 64-core AMD EPYC 7543 with eight Nvidia RTX 6000 Ada GPUs.

## 4.2 Consistency Verdicts and Local Robustness

We first evaluate the paired consistency oracles on sampled input neighborhoods and fault cases, and then the one-model clean-label robustness measure, whose radius $\epsilon _ { \mathrm { H a r d } } ,$ , the largest tested radius certified YES, defines the local distributions. The verdicts below hold for the uniform $L _ { 2 }$ ball of that radius around each seed, or the token-substitution set for text; any other distribution would have to be certified anew.

Pointwise disagreement. We check pointwise reference–compiled disagreement with the main algorithm at $\theta = 1 0 ^ { - 3 }$ on 20 inputs of each of the 21 non-ImageNet tasks under both backends, at each input’s $\epsilon _ { \mathrm { H a r d } }$ . Of the 840 input–backend instances, 817 certify YES, 19 return NONE and 4 are rejected, after a median of 13,312 draws of two model executions each. Per-cell densities range from $4 . 9 \times 1 0 ^ { - 6 }$ to $8 . 3 \times 1 0 ^ { - 4 }$ , and the NONE and NO verdicts concentrate on near-threshold cells such as MobileNetV2/CIFAR-10 under TensorRT and ViT-Base/CIFAR-10 on both backends. These acceptances bound the probability on the sampled instances; they do not certify the builds as semantically correct. The backend matters more than the architecture: Inductor’s density exceeds TensorRT’s on 19 of 21 tasks (medians $4 . 1 \times 1 0 ^ { - 4 }$ and $6 . 6 \times 1 0 ^ { - 5 } )$ whereas vision transformers and convolutional networks differ little. About half of the pointwise disagreements occur where the reference model itself leaves the clean label (median ratio of reference-stable to pointwise density 0.56).

Table 3. Certifying reference-stable and pointwise disagreement on the nominal 50-input cohorts, at the RQ1 radius and at half that radius. Cells give the inputs certified Yes and, in parentheses, mean unique draws in thousands, each draw querying both models. At the tightest threshold and the full radius, the remaining inputs return NO or no verdict.
<table><tr><td rowspan="2">Setting</td><td rowspan="2">Radius</td><td colspan="2">Reference-stable</td><td colspan="2">Pointwise</td></tr><tr><td> $\theta = 1 0 ^ { - 3 }$ </td><td> $\theta = 1 0 ^ { - 4 }$ </td><td> $\theta = 1 0 ^ { - 3 }$ </td><td> $\theta = 1 0 ^ { - 4 }$ </td></tr><tr><td> $\mathrm { L e N e t } { - 5 / \mathrm { M N I } \mathrm { S T } _ { \mathrm { t r t } } }$ </td><td>€Hard</td><td>49/49 (14.0)</td><td>41/49 (682.4)</td><td>49/49 (15.0)</td><td>34/49 (1085.9)</td></tr><tr><td>LeNet  $\cdot 5 / \mathrm { M N I S T _ { t r t } }$ </td><td> $\epsilon _ { \mathrm { H a r d } } / 2$ </td><td>49/49 (12.4)</td><td>49/49 (128.8)</td><td>49/49 (12.4)</td><td>49/49 (128.8)</td></tr><tr><td> $\mathrm { R e s N e t – 1 8 / C I F A R  – 1 0 _ { i n d } }$ </td><td>€Hard</td><td>48/50 (36.2)</td><td>12/50 (1584.8)</td><td>48/50 (37.8)</td><td>1/50 (2014.1)</td></tr><tr><td> $\mathrm { R e s N e t – 1 8 / C I F A R  – 1 0 _ { i n d } ^ { \cdots } }$ </td><td> $\epsilon _ { \mathrm { H a r d } } / 2$ </td><td>50/50 (12.4)</td><td>50/50 (130.0)</td><td>50/50 (12.4)</td><td>50/50 (130.0)</td></tr><tr><td> $\mathrm { V G G { - } 1 6 { - } B N / S V H N _ { i n d } }$ </td><td>€Hard</td><td>50/50 (16.3)</td><td>40/50 (903.1)</td><td>50/50 (18.1)</td><td>30/50 (1576.9)</td></tr><tr><td>VGG-16-BN/SVHNind</td><td> $\epsilon _ { \mathrm { H a r d } } / 2$ </td><td>50/50 (12.4)</td><td>50/50 (128.8)</td><td>50/50 (12.4)</td><td>50/50 (128.8)</td></tr><tr><td> $\mathrm { R e s N e t - } 3 4 / \mathrm { S V H N _ { t r t } }$ </td><td> $\epsilon _ { \mathrm { H a r d } }$ </td><td>50/50 (12.7)</td><td>50/50 (223.8)</td><td>50/50 (13.3)</td><td>48/50 (405.3)</td></tr><tr><td> $\mathrm { R e s N e t - } 3 4 / \mathrm { S V H N _ { t r t } }$ </td><td> $\epsilon _ { \mathrm { H a r d } } / 2$ </td><td>50/50 (12.4)</td><td>50/50 (128.8)</td><td>50/50 (12.4)</td><td>50/50 (129.4)</td></tr></table>

Tight thresholds. Table 3 certifies the reference-stable property, the joint event that the reference keeps the clean label but the compiled model does not, and the pointwise property on 50-input cohorts. $\mathrm { A t } 1 0 ^ { - 3 } ,$ , 197 of 199 inputs certify YES at the full radius and all at half the radius, for 12–38 thousand draws; at $1 0 ^ { - \bar { 4 } }$ and half the radius every input certifies. $\mathrm { A t } 1 0 ^ { - 4 }$ and the full radius the subjects separate: under the reference-stable property ResNet-34/SVHN certifies all 50 inputs, VGG-16-BN/SVHN 40, and ResNet-18/CIFAR-10 only 12 while rejecting 24, and the pointwise property, which also counts disagreements where the reference leaves the clean label, certifies fewer (e.g. 1 rather than 12 on ResNet-18/CIFAR-10). The reference keeps the clean label on 99.0–100% of draws at these radii, but the certificates concern the joint event, not a conditional ratio. Here the main algorithm reaches the same verdicts as the interval family with 0.58–0.90 of its draws and inference-inclusive time, and the saving concentrates on rejections, where it uses 0.25 of the family’s draws.

Detecting an injected miscompilation. We perturb one layer’s weights of the compilation target by a relative amount, modelling a constant folded slightly wrong, and sweep nine magnitudes on ResNet-18/CIFAR-10 under Inductor (Table 5). A clean test set is almost blind: a 2% fault changes none of 1000 predictions, a 20% fault one, and even 120% only 18. So is the certified radius, whose median moves from 6.05 to 5.95 because $\epsilon _ { \mathrm { H a r d } }$ follows the model’s decision geometry. The reference-stable density, measured over $4 \times 1 0 ^ { 6 }$ draws per setting, rises monotonically from $2 . 5 \times 1 0 ^ { - 4 } \mathrm { t o } 3 . 8 \times 1 0 ^ { - 2 }$ , and even the 2% fault raises it by half $( z = 1 0 . 5 )$ Certified at $\dot { \theta } = 1 0 ^ { - 3 }$ , the uninjected compilation is YES on all 30 inputs, the 20% fault is rejected on 16 of 30, and rejection grows to 23 of 30, with intermediate magnitudes passing through NONE. The 2% fault stays below θ (four NONE) and needs $\theta \approx 3 \times 1 0 ^ { - 4 }$ . Detecting miscompilation is thus a question about the compiler-attributable density, with the threshold set near that density.

Real compiler bugs. Our Torch-TensorRT 2.6 case study reproduces an interpolation-mode swap associated with issue #3674: the converter runs a bilinear grid\_sample as nearest-neighbour inside the engine (reported with Torch-TensorRT 2.7.0).<sup>1</sup> We compile a spatial transformer network trained on MNIST (99.0% test accuracy) with the converter as installed and with the upstream fix applied, and certify each against the uncompiled model on 30 test inputs with the reference-stable property at $\theta = 1 0 ^ { - 3 }$ , at the fixed $\mathrm { e n g i n e ^ { \prime } s } \epsilon _ { \mathrm { H a r d } }$ . The fixed engine agrees on all 10,000 test images, has density $5 . 7 \times 1 0 ^ { - 6 }$ over $6 \times 1 0 ^ { 6 }$ draws, and certifies YES on 30 of 30 inputs. The buggy engine changes 52 of 10,000 clean predictions, has density $6 . 5 \times 1 0 ^ { - 2 } ( 3 . 8 \times 1 0 ^ { - 3 }$ to

0.34 per input), and is rejected on 30 of 30 inputs after a median of about 1,000 draws. A clean differential test of the same cost would see one of the 0.52% flipped predictions with probability $1 - ( 1 - 0 . 0 0 5 2 ) ^ { 1 0 0 0 } \approx 9 9 \% ;$ ; the certificate adds a per-input verdict at stated confidence and shows the bug is more than ten times as frequent in perturbation neighbourhoods. Unlike the small injected faults, this bug also shrinks $\epsilon _ { \mathrm { H a r d } } ,$ , by a median of 18%, since it changes the decision geometry. The study covers one model, and the fixed engine is 2.6 with the upstream patch.

A second documented bug miscompiles an nn.Linear applied to a four-dimensional channelslast tensor followed by a reshape, the pattern of ViTDet/SAM-style window attention (Torch-TensorRT issue #4379<sup>2</sup>). In our environment it appears in fp16 engines and is avoided by an equivalent transpose / reshape form of the attention’s output. We deploy a windowed ViT trained on CIFAR-10 (82.5% test accuracy) in fp16 and compare the buggy and workaround engines against the uncompiled fp16 model on 30 test inputs at the workaround engine’s $\epsilon _ { \mathrm { H a r d } }$ The workaround engine agrees on 9,997 of 10,000 test images, has density $3 . 1 \times 1 0 ^ { - 5 }$ and certifies YES on 30 of 30 inputs. The buggy engine changes 1,942 clean predictions and has density 0.17 overall, but from 0 to 0.99 per input: it is rejected on 21 inputs after a median of 768 draws, returns NONE on 6 and is certified on 3 whose neighbourhoods the fault does not reach, while $\epsilon _ { \mathrm { H a r d } }$ moves by only 2%. The two faults thus leave different footprints, the interpolation swap raising the density in every tested neighbourhood and the window-attention fault concentrating on some inputs, so the result depends on which neighbourhoods are checked. Clean testing also exposes both faults, and two cases do not establish comparative detection recall.

The clean-label radius as a robustness measure. We ask whether ProCerDCL’s certified radius agrees with an attack-based notion of robustness, using the white-box compilation-inconsistency search of Chen et al. [5], which looks for a small $L _ { 2 }$ perturbation making the compiled model leave the reference model’s clean label, guided by the reference model’s gradients since TensorRT engines expose none. ProCerDCL searches, from black-box queries, for the largest tested radius certified YES under PROVERO’s protocol [6]. The image sweep attacks 1000 inputs per setting and verifies 100 attack-success inputs, drawn separately per backend; the transformer runs use 100 inputs per setting. Because a searched radius spends the error budget once per probe, it carries at least $1 - K _ { \operatorname* { m a x } } \delta ,$ i.e. 68–78%, confidence over at most $K _ { \operatorname* { m a x } } = 2 2 { - } 3 2$ probes fixed in advance by the cap $4 \| x _ { 0 } \| _ { 2 }$ and the resolution, rather than the 99% that holds per probe; splitting the budget restores 99% at 1.49–1.54 times the samples and moves the radius by under 1%. All tables use the per-probe mode (Tables $^ { 2 , }$ 4 and 6), PROVERO included, while Tables 3 and 5 certify at a single radius. A YES certifies only its own ball, and the bisection assumes p(ε) grows with $\varepsilon ,$ which need not hold: in one dimension with seed 0 and violation set $1 < | x | \leq 2 ,$ $p ( 2 ) = 0 . 5$ but $p ( 2 0 ) = 0 . 0 5$ . We therefore report the largest tested radius certified YES, and a NONE probe only stops the search.

Agreement with attack difficulty. Inputs needing a larger attack budget receive a larger certified radius (Figure 2a; Pearson 0.93, 0.71, 0.82 and 0.78 on four representative settings). Across the 24 tasks, all of which qualify, the measures agree positively on 46 of 48 cells and significantly on 42 at the 5% level (two-sided t-test on Pearson’s r), 41 after Holm correction, averaging 0.55–0.56 by Pearson and 0.63 by Spearman; these averages are descriptive and describe attack-success inputs only. Rank agreement exceeds linear agreement almost everywhere (ResNet-50/CIFAR-10: 0.32 against 0.69), so ProCerDCL orders inputs by robustness even where a few very robust inputs flatten the line. Agreement depends on the subject: the ImageNet-scale ResNets sit between −0.07 and 0.20, and MobileNetV2/CIFAR-10, conclusive on 99 of 100 inputs, carries almost no

![](images/ac955ecf57502be4b3fd1ed3fd113bcce8583c9fa5bbdf54abb63d34db5c2b0f.jpg)  
(a) Certified radius vs. attack budget.

![](images/4fa0dd45c4dbffed6754b07e3fe905aca6c00250a7d900a5eee00ce4670d4f71.jpg)  
(b) Compiled vs. reference certified radius.

Figure 2. Per-input agreement with the flipping budget found by the attack’s heuristic search (a), and the reference-model control (b). Each point is one input; the red dashed line is the least-squares fit. signal (0.05 and 0.17, neither significant). It is nearly the same on the two backends, 0.53 and 0.54 by Pearson and 0.61 by Spearman on the 19 convolutional tasks and $0 . 6 3 / 0 . 6 2$ and 0.72/0.70 on the five transformer tasks under Inductor/TensorRT, as expected of a radius governed by the model.

What the clean-label radius attributes to the compiler. Certifying the uncompiled reference model yields indistinguishable certificates (Figure 2b): the radii correlate at 1.00 with a median per-input difference of 0.00. At the certified radius, 95.1–99.8% of the counted flips also flip the reference model, leaving 0.2–4.9% as reference-stable disagreement at a density of $1 0 ^ { - 5 }$ to $5 \times 1 0 ^ { - 4 }$ , well below the threshold. The certified radius thus measures the model’s own noise robustness, and these correlations validate the certified quantity rather than a compiler-specific signal.

Attack on transformer subjects. The original attack does not carry over to transformers: on our vision transformers it returned nearly the same budget for every input (coefficient of variation 0.17 on ViT-Base/CIFAR-10, against 0.30 on the convolutional subjects). We therefore extended it to transformers; the original authors of [5] reviewed this extension and endorsed it as a reasonable extension of their attack, consistent with its intent. It optimizes the reference model’s top-two logit gap and bisects along the resulting direction, yielding budgets with coefficient of variation 0.84, 92–100 of 100 conclusive certificates per cell, and higher rank agreement on all ten transformer cells, by 0.27 on average. We read the transformer rows, like the text rows, as a diagnostic of whether the certificate orders inputs as an attack does.

Text classifiers. For BERT-base on SST-2 and AG News and DistilBERT on SST-2 under both backends, we certify a token-level perturbation set: length is fixed and each position is independently replaced with probability p by one of its ten nearest embedding neighbours. A greedy forward-query attack searches the same set, so both sides report a substitution rate and $p _ { \mathrm { H a r d } }$ replaces $\epsilon _ { \mathrm { H a r d } }$ . Agreement is positive on all six settings, 0.33–0.80 by Pearson and 0.26–0.75 by Spearman, and the attack budget is essentially uncorrelated with sentence length (0.03); we claim only that this distribution is well defined, not that substitutions preserve meaning. The attack flips 64–100 of 100 inputs, so n differs per cell; conclusive means a positive $p _ { \mathrm { H a r d } } ,$ and the excluded inputs concentrate on TensorRT.

## Answers to RQ1

Under uniform $L _ { 2 }$ neighbourhoods at each seed’s radius, pointwise checks at $\theta = 1 0 ^ { - 3 }$ return 817 YES, 19 NONE, and 4 NO across 840 input–backend instances from 21 tasks. The two documented fault cases illustrate local rejection, acceptance, and abstention patterns rather than general bug-detection recall. The one-model clean-label radius tracks attack difficulty over the 24-task sweep, but the reference-model control shows it is largely model-dominated.

Table 4. Mean unique samples per input $( \times 1 0 ^ { 3 } )$ to reach a verdict, per task and backend, averaged over every search, whatever it concluded. Each backend gives PROVERO first, then ProCerDCL with its speedup over PROVERO in parentheses.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Dataset</td><td colspan="2">inductor</td><td colspan="2">tensorrt</td></tr><tr><td>PROVERO</td><td>ProCerDCL</td><td>PROVERO</td><td>ProCerDCL</td></tr><tr><td>LeNet-5</td><td>MNIST</td><td>408.6</td><td>141.5 (2.89×)</td><td>371.3</td><td>126.3 (2.94×)</td></tr><tr><td rowspan="4">ResNet-18</td><td>CIFAR-10</td><td>336.5</td><td>111.4 (3.02×)</td><td>354.0</td><td>117.2 (3.02×)</td></tr><tr><td>CIFAR-100</td><td>317.8</td><td>105.0 (3.03×)</td><td>324.5</td><td>110.4 (2.94×)</td></tr><tr><td>SVHN</td><td>431.2</td><td>152.3 (2.83×)</td><td>484.4</td><td>167.3 (2.89×)</td></tr><tr><td>ImageNet-1k</td><td>572.2</td><td>205.0 (2.79×)</td><td>563.3</td><td>203.8 (2.76×)</td></tr><tr><td rowspan="3">ResNet-34</td><td>CIFAR-10</td><td>335.3</td><td>114.2 (2.94×)</td><td>369.1</td><td>124.3 (2.97×)</td></tr><tr><td>CIFAR-100</td><td>309.2</td><td>101.4 (3.05×)</td><td>318.9</td><td>105.9 (3.01 ×)</td></tr><tr><td>SVHN</td><td>427.4</td><td>147.0 (2.91×)</td><td>494.2</td><td>173.1 (2.86×)</td></tr><tr><td rowspan="4">ResNet-50</td><td>CIFAR-10</td><td>340.2</td><td>110.5 (3.08×)</td><td>344.9</td><td>114.9 (3.00×)</td></tr><tr><td>CIFAR-100</td><td>314.7</td><td>102.6 (3.07×)</td><td>315.1</td><td>106.0 (2.97×)</td></tr><tr><td>SVHN</td><td>444.9</td><td>154.2 (2.88×)</td><td>489.8</td><td>168.7 (2.90×)</td></tr><tr><td>ImageNet-1k</td><td>562.3</td><td>200.5 (2.80×)</td><td>577.5</td><td>203.6 (2.84×)</td></tr><tr><td rowspan="3">VGG-16-BN</td><td>CIFAR-10</td><td>356.1</td><td>121.3 (2.94×)</td><td>365.8</td><td>125.8 (2.91×)</td></tr><tr><td>CIFAR-100</td><td>297.9</td><td>99.7 (2.99×)</td><td>320.6</td><td>109.4 (2.93×)</td></tr><tr><td>SVHN</td><td>465.4</td><td>163.5 (2.85×)</td><td>501.2</td><td>176.6 (2.84×)</td></tr><tr><td rowspan="3">DenseNet-121</td><td>CIFAR-10</td><td>310.4</td><td>103.8 (2.99×)</td><td>311.5</td><td>105.7 (2.95×)</td></tr><tr><td>CIFAR-100</td><td>282.8</td><td>95.3 (2.97×)</td><td>302.9</td><td>99.3 (3.05×)</td></tr><tr><td>SVHN</td><td>469.9</td><td>159.1 (2.95×)</td><td>489.5</td><td>172.4 (2.84×)</td></tr><tr><td>ViT-Base/16</td><td>CIFAR-10</td><td>570.1</td><td>202.9 (2.81 ×)</td><td>583.7</td><td>213.0 (2.74×)</td></tr><tr><td>DeiT-Small</td><td>ImageNet-1k</td><td>708.3</td><td>257.9 (2.75×)</td><td>709.2</td><td>262.2 (2.71×)</td></tr><tr><td rowspan="3">ViT-B/16 in21k-ft</td><td>CIFAR-10</td><td>615.7</td><td>222.8 (2.76×)</td><td>587.9</td><td>213.3 (2.76×)</td></tr><tr><td>CIFAR-100</td><td>601.2</td><td>212.6 (2.83×)</td><td>580.2</td><td>207.7 (2.79×)</td></tr><tr><td>SVHN</td><td>664.6</td><td>234.1 (2.84×)</td><td>621.7</td><td>225.9 (2.75×)</td></tr></table>

Table 5. Injected constant-folding faults.
<table><tr><td>fault</td><td></td><td>clean top-1 ref.-stable density</td><td>verdicts</td></tr><tr><td>0</td><td>1000/1000</td><td> $2 . 5 \times 1 0 ^ { - 4 }$ </td><td>30 Yes</td></tr><tr><td>0.005</td><td>1000/1000</td><td> $2 . 8 \times 1 0 ^ { - 4 }$ </td><td>29 Yes, 1 None</td></tr><tr><td>0.02</td><td>1000/1000</td><td> $3 . 8 \times 1 0 ^ { - 4 }$ </td><td>26 Yes, 4None</td></tr><tr><td>0.05</td><td>999/1000</td><td> $5 . 9 \times 1 0 ^ { - 4 }$ </td><td>19 Yes, 11 None</td></tr><tr><td>0.1</td><td>999/1000</td><td> $1 . 0 \times 1 0 ^ { - 3 }$ </td><td>12 Yes, 11 None, 7 No</td></tr><tr><td>0.2</td><td>999/1000</td><td> $2 . 1 \times 1 0 ^ { - 3 }$ </td><td>16No, 9Yes, 5None</td></tr><tr><td>0.5</td><td>997/1000</td><td> $7 . 2 \times 1 0 ^ { - 3 }$ </td><td>20 No, 7 Yes, 3 None</td></tr><tr><td>0.8</td><td>990/1000</td><td> $1 . 6 \times 1 0 ^ { - 2 }$ </td><td>22 No, 7 Yes, 1 None</td></tr><tr><td>1.2</td><td>982/1000</td><td> $3 . 8 \times 1 0 ^ { - 2 }$ </td><td>23 No, 7 Yes</td></tr></table>

Table 6. Paired comparison and ablations.
<table><tr><td>Method</td><td> $\mathrm { L e N e t / M _ { \mathrm { t r t } } }$ </td><td> $\mathrm { R N } 1 8 / \mathrm { C } 1 0 _ { \mathrm { i n d } }$ </td><td> $\mathrm { V G G } / \mathrm { S V _ { i n d } }$ </td><td> $\mathrm { R N } 3 4 / \mathrm { S V } _ { \mathrm { t r t } }$ </td></tr><tr><td>ProCerDCL, main (CS)</td><td>90.7 (0.3)</td><td>82.6 (1.6)</td><td>118.9 (1.2)</td><td>130.3 (6.5)</td></tr><tr><td>ProCerDCL, main (CP)</td><td>102.3 (0.3)</td><td>96.0 (1.5)</td><td>131.5 (1.3)</td><td>142.5 (6.8)</td></tr><tr><td>PROVERO</td><td>375.0 (0.8)</td><td>348.7 (7.2)</td><td>473.0 (4.8)</td><td>508.5 (25.9)</td></tr><tr><td>Interval-family mode</td><td></td><td></td><td></td><td></td></tr><tr><td>ProCerDCL-CS (anytime-valid)</td><td>120.6 (0.3)</td><td>106.8 (2.2)</td><td>158.4 (1.6)</td><td>164.3 (8.4)</td></tr><tr><td>ProCerDCL (CP)</td><td>127.1 (0.4)</td><td>120.5 (2.1)</td><td>165.0 (1.6)</td><td>176.6 (8.4)</td></tr><tr><td>ProCerDCL-CS-F (forced)</td><td>155.3 (0.4)</td><td>138.9 (2.8)</td><td>200.5 (2.0)</td><td>210.6 (10.7)</td></tr><tr><td>ProCerDCL-F (CP, forced)</td><td>160.3 (0.5)</td><td>144.5 (2.8)</td><td>205.4 (2.1)</td><td>216.9 (10.7)</td></tr><tr><td>ProCerDCL-UD (no alloc)</td><td>154.6 (0.5)</td><td>141.0 (3.4)</td><td>202.8 (2.9)</td><td>206.4 (10.1)</td></tr><tr><td>ProCerDCL-NR (no reuse)</td><td>266.8 (0.7)</td><td>250.3 (4.4)</td><td>327.3 (3.3)</td><td>341.7 (15.3)</td></tr></table>

## 4.3 Sampling Cost and Conclusiveness

We count the unique draws ProCerDCL needs per radius search against PROVERO’s adaptive certifier [6] at the same threshold, tolerance, and confidence. This comparison concerns the onemodel clean-label deviation property. The two contracts differ: by default ProCerDCL satisfies Definition 3.1, which counts any decisive verdict inside the band as an error, whereas PROVERO, which always answers Yes or No, satisfies only the weaker indifference-zone contract. The full table compares the methods as released; the like-for-like comparison is the forced-decision one on the paired cohort below.

Table 4 covers the 23 tasks on which both methods ran on both backends and reports the interval-family mode; the main algorithm is compared on the paired cohort below. ProCerDCL needs 2.89–2.91 times fewer unique samples than PROVERO on average, about 155 thousand against 445 thousand, and between 2.71 and 3.08 on every setting, the saving being smallest on the costlier transformer and ImageNet searches. The paired forced-decision comparison is reported separately below. Sample reuse acts within a probe, turning the cost of its interval tests from a sum into a maximum; probes at different radii share no draws, so a search costs the sum of its probes, dominated by the last probes near the boundary, which run to PROVERO’s Chernoff-derived truncation budget and return NONE.

Table 7. Stability across ten random seeds on LeNet-5/MNIST under TensorRT.
<table><tr><td>Same verdict</td><td>Mean draws</td><td>Draws over seeds</td><td>Median  $\epsilon _ { \mathrm { H a r d } }$ </td><td>Per-input  $\epsilon _ { \mathrm { H a r d } }$  spread</td><td>Within one step</td><td>Per-input draw CV</td></tr><tr><td>20/20</td><td>141.8k (CV 2.3%)</td><td>136.1k-147.2k</td><td>6.28-6.65</td><td>median 0.30, max 1.40</td><td>8/20</td><td>median 9.2%, max 21.7%</td></tr></table>

Paired comparison. Table 6 reports mean unique draws in thousands per one-model clean-label radius search, with mean wall-clock seconds in parentheses, for a paired cohort of 50 inputs in each of four settings. Draw-count comparisons share one cached stream per input and radius. Timing runs are separate and uncached, including inference and buffered GPU evaluations not consumed by the tester. Thus, draws, model executions, and elapsed time remain distinct measures.

Main algorithm. The main test consumes 102.3, 96.0, 131.5 and 142.5 thousand draws with the Clopper–Pearson (CP) rule and 90.7, 82.6, 118.9 and 130.3 thousand with the confidencesequence (CS) rule, against 375.0, 348.7, 473.0 and 508.5 thousand for PROVERO, i.e. 3.6–4.2 times fewer at matched numerical parameters but under different decision contracts. It uses 0.80–0.81 (CP) and 0.75–0.79 (CS) of the interval-family mode’s draws with radii within 1% at the median, and with CS takes 0.3, 1.6, 1.2 and 6.5 seconds against 0.3, 2.2, 1.6 and 8.4 for the family and 0.8, 7.2, 4.8 and 25.9 for PROVERO on our platform. The family gains nothing at the default parameters: its only left interval, (0, θ), cannot return YES under the CP rule, so acceptance comes from its final test, which receives about a third of δ and truncates at 27,038 draws rather than 22,105, and on synthetic streams with θ up to 50% it was never cheaper, although this is no universal optimality result. Inconclusive probes account for 61% of the main algorithm’s draws, and 43% of its searches end on a NONE probe inside the band; a shorter cap would trade sampling for more abstentions without affecting soundness.

Compatibility-mode ablations. Removing reuse costs 1.96–2.21 times as many draws, and uniform allocation costs 1.17–1.23 times as many, relative to CP interval-family mode. The CS variant reduces that mode’s cost by 3–11%. These components reduce overhead within the family; they are not components of the main single-test algorithm. Forcing a decision costs 1.19– 1.27 times as many draws in the reported variants and produces radii 5–8% larger, with 94–100% of searches ending on a probe reported to be inside the band. Comparing the forced variants under PROVERO’s indifference-zone contract, PROVERO uses 2.30–2.41 times as many draws as CS-F on these four settings. The CP-F row is a separate comparison. Forced decisions inside the band do not satisfy Definition 3.1. Similarly, an SPRT configured for separated hypotheses [7] cannot be treated as satisfying this contract merely from its endpoint error guarantees.

False-certification rate. We ran 1000 verifications at each of 12 known densities below, inside and above the band. No Yes occurred in the 5000 runs above the band and no No in the 5000 below it, a one-sided Clopper–Pearson 95% bound of 0.3% per density and 0.06% pooled, below the 1% allowed. Under the stricter Definition 3.1, the 2000 in-band runs yield one decisive verdict, a NO at exactly $p = \theta + \eta$ . Near the threshold the certifier declines to answer on 88–100% of streams rather than guessing. Away from the band it is decisive: every one of the 1000 runs answers YES at densities up to θ/2 and NO from 2(θ + η) upward, so soundness is not bought by abstaining. Stability across seeds. Re-running LeNet-5/MNIST under TensorRT on 20 inputs with ten seeds (Table 7) leaves every verdict unchanged; mean cost varies by 2.3% and the median radius stays between 6.28 and 6.65. A single input’s radius varies more, with a median spread of 0.30 (about 5%) and at most 1.40, so cohort-level figures are reproducible while per-input radii carry an uncertainty of a few grid steps.

![](images/3a5b98f2c5d8068cf6e67ae35eb3726d4c8396e7d33e4230486f6f747a333894.jpg)

![](images/719db6004ac804a3c376fbf2b8a74cdafde104c7be6b445cc770d0b7963a0310.jpg)

![](images/9289028cb938dcb46abde6e474993b1b1de7ea67e72c4c7536e8ba86aa546d40.jpg)

![](images/e71fc016f4a339b864c7140c5a10d5851071af1c2a0825c41a25950ab4ba666c.jpg)  
(a) Simulated INT8 targets.

![](images/7870f8d554be7d33c0e4071347b609f5b9b989ab96a9e8a8274535ce50389648.jpg)

![](images/9aab52e7ace02f642f5c826fe8e627240ab1448926214d472a80a8d6dbadf1ca.jpg)

![](images/bd7fd9e307434cd4c1cfa36c99cb17058b68265fe13d556802bbc396ae76f31e.jpg)  
(b) Real TensorRT INT8 engines.

Figure 3. Certified hardness of quantized models against the attack’s heuristic flipping budget. LeNet-5/MNIST ResNet-18/C10 VGG-16/SVHN ResNet-34/SVHN LeNet-5/MNIST ResNet-18/C10 VGG-16/SVHN ResNet-34/SVHN  
![](images/b05d0bc060d6ecc9bb0d4e4bde9cf4a33e07737295cc9cd048850c61926d3727.jpg)

![](images/9b70f5e432229bf87e014b736d15c9306b2806fc894de87dedc83fb5bff5339e.jpg)

![](images/cbafd4d69a30137a77211ef2a968014eae899bedaadf39963911f2acb0e88ea2.jpg)

(a) Cross-hardware: RTX 6000 Ada vs. RTX 4500.  
![](images/90cbaf9306d9a885c2f6cbe8ed7148bc26f9e51394bd360e914c4b5b6fdfba30.jpg)  
(b) L∞ threat model.  
Figure 4. One-model local robustness under hardware and perturbation changes.

## Answers to RQ2

PROVERO uses 3.6–4.2 times as many draws as the main test on paired one-model radius searches, with different decision contracts. The corresponding interval-family tests consume more draws than the main test in all four settings. Conclusiveness remains a separate outcome: inconclusive probes consume 61% of the main test’s draws, and the known-density study reports one erroneous decisive verdict in 2,000 in-band runs and none in 10,000 outsideband runs.

## 4.4 RQ 3 Local Robustness across Configurations

We evaluate the one-model clean-label robustness instantiation under quantization, a second GPU generation, and a max-norm perturbation distribution. The same test applies under A1–A3. These experiments measure local robustness under the selected settings. Equivalence between reference and compiled models is a separate property.

Quantized DNNs. We keep the FP32 model as reference and apply simulated static INT8 post-training quantization to its counterpart (per-channel weights, per-tensor activations, 64 calibration inputs), inserted into the compiled model so that it carries faithful INT8 rounding error. Inputs whose clean prediction quantization already flips return no verdict by construction (2 of 400). ProCerDCL is conclusive on 397 of 400 inputs, and the certificate rises with the attack budget on every setting, 0.58–0.83 by Pearson and 0.66–0.88 by Spearman (Figure 3a). As under FP32, the certified radius exceeds the attack’s single-direction budget, with fitted slopes of 3.8–6.5, because it concerns randomly sampled perturbations.

Real INT8 Engines. Actual TensorRT INT8 engines for the two TensorRT settings (entropy calibration) are attackable at 55.3% and 65.3% over 300 inputs, and ProCerDCL certifies them unmodified: 100 and 99 of 100 conclusive, correlating 0.82 and 0.76 by Pearson and 0.87 and 0.82 by Spearman (Figure 3b), with cohort radii close to the simulation (9.5 against 10.6, 26.0 against 24.4).

Across Hardware Platforms. On an Ampere RTX A4500 with identical inputs, seeds and parameters and engines built for that GPU, the certified radii of all four settings agree with the Ada runs at correlation 1.000 and unit slope (Figure 4a), and under TensorRT radii and sample counts are identical on all 200 inputs: the certificate is reproducible across hardware, as a model-dominated radius predicts.

A Different Threat Model. Sampling from a box instead of a ball certifies the max-norm density under the same guarantee; on the same four settings the certified radius again rises with the attack budget, 0.50–0.86 by Pearson and 0.49–0.91 by Spearman (Figure 4b). Only the sampler changes; the verifier, its contract and its error accounting are unchanged, so the guarantee carries over to the new distribution as stated.

## Answers to RQ3

The same one-model verifier runs in all three evaluated configurations. The robustness radii reproduce across the studied GPU generations (correlation 1.000), remain conclusive on 397 of 400 simulated INT8 inputs, and correlate with attack budgets under the max-norm distribution (Pearson 0.50–0.86). These are local robustness results under each configuration’s sampling and oracle assumptions. Moving from the $L _ { 2 }$ ball to the $L _ { \infty }$ box required only a different sampler.

## 4.5 RQ 4 Parameter Study

For the one-model clean-label radius experiment, we vary one parameter at a time around the default (θ = 1%, $\eta = 0 . 5 \% ,$ 99% confidence) on ResNet-18/CIFAR-10 under Inductor, with 100 inputs per configuration (Table 8). All eight configurations are conclusive on all 100 inputs, so within this grid the parameters change cost and the location of the reported boundary, not whether a verdict is reached. Threshold. The median radius rises monotonically with θ, from 5.80 at 0.5% through 6.05 and 6.40 to 7.40 at 5%, because a larger

Table 8. Parameter sensitivity of one-model clean-label radius searches on ResNet-18/CIFAR-10 under Inductor.
<table><tr><td>varied</td><td>value</td><td>concl.</td><td>radius</td><td>draws</td></tr><tr><td>default</td><td></td><td>100%</td><td>6.05</td><td>109,061</td></tr><tr><td>θ</td><td>0.5%</td><td>100%</td><td>5.80</td><td>199,371</td></tr><tr><td>θ</td><td>2%</td><td>100%</td><td>6.40</td><td>57,582</td></tr><tr><td>θ</td><td>5%</td><td>100%</td><td>7.40</td><td>25,978</td></tr><tr><td> $\begin{array} { l } { 1 - \delta } \\ { 1 - \delta } \end{array}$ </td><td>99.9%</td><td>100%</td><td>6.05</td><td>148,694</td></tr><tr><td></td><td>95%</td><td>100%</td><td>6.05</td><td>85,034</td></tr><tr><td>η</td><td>0.25%</td><td>100%</td><td>6.15</td><td>301,219</td></tr><tr><td>η</td><td>1%</td><td>100%</td><td>5.95</td><td>45,287</td></tr></table>

tolerated violation density is first exceeded in a larger ball. Cost falls roughly in inverse proportion to $\theta ,$ at 1.83, 1, 0.53 and 0.24 times the default draws, consistent with the order of ln(1/δ)/θ draws a sequential test needs to establish $p \leq \theta$ when p lies well below θ. Margin. Halving η to 0.25% multiplies the draws by 2.76 and doubling it to 1% by 0.42, while the median radius moves by at most one grid step (5.95–6.15). The growth is smaller than the fourfold growth of the truncation cap because only probes near the boundary run to the cap; probes far from the band stop early. Confidence. Raising the confidence from 99% to 99.9% costs 1.36 times the draws and lowering it to 95% saves a quarter (0.78), close to the logarithmic factors 1.50 and 0.65 of ln $( 1 / \delta )$ , and leaves the median radius at exactly 6.05. Tightening the confidence is therefore cheap, whereas θ and η set both the cost and the meaning of a verdict. This sensitivity result concerns the tested grid on one model-dataset-backend setting.

## Answers to RQ4

All eight configurations are conclusive on every input, and the certified radius moves by about a quarter across the grid while sampling cost spans 11.6 times. The threshold drives both radius and cost, halving the margin multiplies the cost by 2.8, and confidence enters roughly logarithmically without moving the radius.

## 4.6 RQ 5 Beyond Norm-Bounded Perturbations

Since the perturbation distribution is an input to ProCerDCL, we test one that is not a norm ball around the whole input: only the nine pixels of the top-left $3 \times 3$ patch of an MNIST image change, each shifted independently and uniformly within ±ϵ without clipping, and the radius search grows this per-pixel magnitude. The attack is restricted to the patch likewise, measures its budget as the largest per-pixel change, and uses a step size of 0.5 and 1000 steps because nine pixels must move far to change a decision. The protocol otherwise follows Section 4.2 on LeNet-5/MNIST under both backends, with 100 attack-success inputs each, except that a certificate is conclusive when its radius is positive and below the search cap of 400.

Table 9 shows that the certificates stay conclusive on 100 and 99 of 100 inputs and agree with attack difficulty at 0.60 and 0.79 by Pearson and 0.80 and 0.89 by Spearman $( p < 1 0 ^ { - 9 } )$ , at 165 and 171 thousand draws on average. The certified radii spread over more than an order of magnitude, from 2.9 to 52.9 (10th to 90th percentile) under TorchInductor and from 1.9 to 56.2 under TensorRT, so the certificate distinguishes inputs

Table 9. Certificates under the patch distribution on LeNet-5/MNIST.
<table><tr><td></td><td>TorchInductor</td><td>TensorRT</td></tr><tr><td>Conclusive</td><td>100/100</td><td>99/100</td></tr><tr><td>Pearson</td><td>0.60</td><td>0.79</td></tr><tr><td>Spearman</td><td>0.80</td><td>0.89</td></tr><tr><td>Median  $\epsilon _ { \mathrm { H a r d } }$ </td><td>14.3</td><td>16.8</td></tr></table>

whose corner patch barely matters from inputs that react to it. Spearman exceeds Pearson on both backends, which indicates a monotone but nonlinear relation between the two measures, and agreement is higher under TensorRT. The certified radius stays close to the attack budget, at 0.91 and 0.87 of it at the median, and lies below it on 57% and 64% of inputs, as expected of a density-based radius that a single worst-case direction can exceed. Most radii, 87% and 85%, exceed the whole valid pixel range of 3.25 normalized units, so changing LeNet-5’s decision through this patch requires values far outside the image domain; the unclipped distribution makes such a sensitivity measurable rather than truncating it at the range limit. Almost every search, 99 of 100 on each backend, ends on an inconclusive probe near the boundary, and one TensorRT input is rejected already at radius zero, where the compiled engine changes the clean prediction itself. Only the sampler’s support changed, so ProCerDCL applies to distributions defined by where an input may change, not only to $L _ { p }$ -bounded balls.

## Answers to RQ5

On an unclipped $3 \times 3$ patch distribution, ProCerDCL certifies 199 of 200 inputs and its radii track attack difficulty (Spearman 0.80 and 0.89) with the certifier unchanged: its guarantee is tied to the distribution the user specifies beyond one specific perturbation geometry.

## 5 Discussion and Threats to Validity

Meaning of a verdict. A verdict bounds a violation probability for one specified distribution and binary oracle. A low pointwise disagreement probability can support a local regression check or a comparison of a faulty build with its fix, but we have not evaluated such workflows, and the framework neither selects seeds, calibrates thresholds, nor diagnoses faults. A YES permits rare violations and implies neither global equivalence nor worst-case robustness.

Construct validity. The clean-label radius is dominated by model sensitivity (Section 4.2), so its agreement with attack difficulty is not evidence of compiler correctness. The reference-stable property is a joint, not a conditional, probability, and a decision disagreement is a behavioral difference that becomes a semantic error only under an additional correctness specification. Unclipped uniform perturbations define a test distribution, not deployment traffic, and token substitutions need not preserve meaning.

Internal validity. The proof assumes independent draws and a stateless oracle. Our audits found no decision changes under repeated, permuted, or recomposed batches (Section 4.1), which supports A3 for the audited settings only, and TensorRT’s batch-size dependence ties each certificate to its execution configuration. Matched parameters do not make PROVERO’s weaker contract equivalent to ours; the forced-decision variants compare the two under that contract. Conclusion validity. The error bounds hold for one fixed distribution. Radius searches spend δ per probe and carry no search-wide 1 − δ guarantee (Section 4.2), and simultaneous claims over many seeds or configurations would need further allocation. A distribution chosen from earlier observations keeps the guarantee only with fresh draws. Bisection assumes monotonicity, so we report the largest tested radius certified YES, and NONE can occur on either side of the band.

External validity. The image sweep, the pointwise study and the paired comparison cover different populations; attack-success selection, per-backend cohorts and a single sampling seed outside the ten-seed study limit generalization. Two documented faults illustrate verdict patterns but do not measure detection recall, and the deployment-shift experiments concern one-model robustness. Near 10<sup>−4</sup>, two-model checks still need millions of executions; rare-event estimators such as importance sampling or multilevel splitting [41] are a direction for future work.

## 6 Related Work

Testing and verifying DL compilers. Fuzzing finds bugs in DL compilers and frameworks by generating computation graphs or IR programs and differentially executing them: NNSmith [24], MLIRSmith [42], GenCoG [43], coverage-guided fuzzing [44]. Library differential testing includes CRADLE [45], LEMON [46], Audee [47], and EAGLE [48]. Related studies investigate metamorphic testing of compilation [49] and compiler bugs and conversion errors [50; 51; 52; 53]. These approaches investigate implementation bugs using different oracles; numerical differential comparisons can require a tolerance. Chen et al. [5] show decision-level disagreements persist regardless of the threshold. Formal work verifies individual transformations: TASO [54] and PET [55] on graph rewrites, ATL [56] on scheduling rewrites, TensorRight [57] on unbounded tensor sizes, and mlir-tv on MLIR translation validation, following Alive2. Meanwhile, relational verifiers compare network pairs: ReluDiff [58] and NeuroDiff [59] bound output differences, QEBVerif [60] and Giacobbe et al. [61] verify quantized networks, DiffChaser [62] searches black-box for disagreements, and defensive quantization studies robustness effects. Their verification targets and required access differ from our probability-threshold contract. Our procedure observes end-to-end executions through a binary property oracle; decision comparison avoids a tolerance on numerical outputs, while the input distribution, violation threshold, and reference behavior remain explicit parts of the specification.

Reliability of efficient models. Efficiency-oriented model behavior can itself be fragile: adversarial inputs can erase the savings of dynamic transformers [63] or jointly degrade the quality and efficiency of dialogue generation [64], and natural-looking prompts can inflate the inference cost of large language models [65]. Evaluation on fixed test sets can also overstate reliability, for example under benchmark contamination [66], which motivates targeted benchmarks for specific model capabilities [67]. These observations call for guarantees stated over an explicit input distribution rather than a fixed set of tested inputs.

Probabilistic certification and sequential testing. PROVERO [6] is the most closely related work. It develops black-box quantitative verification for a single neural network and instantiates it with probabilistic robustness: sampled perturbations are checked against that model’s prediction at the seed. We use it as the baseline for the one-model clean-label radius-search comparison in Section 4.3. Our framework also instantiates reference–compiled decision consistency, whose oracle compares two executions on the same sampled input. This paired property asks a different question from single-model robustness and requires both models to be executed for each draw. Other related approaches include NPAQ, PROVEN, statistical robustness assessment [41] and statistical model checking [7; 8]. Randomized smoothing [68] also samples with exact binomial confidence, but yields a worst-case radius for a smoothed classifier, whereas ϵ<sub>Hard</sub> bounds a disagreement density of the deployed model; the guarantees are incomparable. Unlike PROVERO’s interval-family procedure, ProCerDCL uses a single sequential test for a stronger three-valued contract, in which a decisive output inside $\left( \theta , \theta + \eta \right]$ counts as an error and NONE is returned when evidence is insufficient. Sequential tests, exact binomial intervals and confidence sequences are established tools; our contribution is the single-test procedure under this contract, its analysis, and its evaluation on compiler-related properties, with the sampling reductions of Section 4.3 reported separately for each contract.

## 7 Conclusion

We presented ProCerDCL, a query-efficient framework for black-box probabilistic verification of consistency properties of DL compiler optimization. Given a fixed, sampleable input distribution and a binary property oracle, its single sequential test returns acceptance, rejection, or an inconclusive verdict with direction-specific error control under independent sampling and stateless oracle behavior. The test supports this three-valued contract without PROVERO’s interval family, and its finite cap bounds cost rather than guaranteeing a decisive answer. In our evaluation, ProCerDCL needs 3.6–4.2 times fewer unique samples than PROVERO at matched parameters, certifies reference–compiled consistency on 817 of 840 inputs at $\theta = 1 0 ^ { - 3 }$ , and separates two documented compiler faults from correct builds; its verdicts remain specific to the chosen distribution, property, and execution configuration.

## References

[1] Tianqi Chen, Thierry Moreau, Ziheng Jiang, Lianmin Zheng, Eddie Yan, Haichen Shen, Meghan Cowan, Leyuan Wang, Yuwei Hu, Luis Ceze, et al. {TVM}: An automated {End-to-End} optimizing compiler for deep learning. In 13th USENIX symposium on operating systems design and implementation (OSDI 18), pages 578–594, 2018.

[2] NVIDIA. Nvidia tensorrt documentation. https://docs.nvidia.com/deeplearning/tensorrt/ latest/index.html, 2024.

[3] Jason Ansel, Edward Yang, Horace He, Natalia Gimelshein, Animesh Jain, Michael Voznesensky, Bin Bao, Peter Bell, David Berard, Evgeni Burovski, et al. Pytorch 2: Faster machine learning through dynamic python bytecode transformation and graph compilation. In Proceedings ofthe 29th ACM international conference on architectural support for programming languages and operating systems, volume 2, pages 929–947, 2024.

[4] David Goldberg. What every computer scientist should know about floating-point arithmetic. ACM computing surveys (CSUR), 23(1):5–48, 1991.

[5] Simin Chen, Jinjun Peng, Yixin He, Junfeng Yang, and Baishakhi Ray. Your compiler is backdooring your model: Understanding and exploiting compilation inconsistency vulnerabilities in deep learning compilers. In 2026 IEEE Symposium on Security and Privacy (SP), pages 233–251. IEEE, 2026.

[6] Teodora Baluta, Zheng Leong Chua, Kuldeep S Meel, and Prateek Saxena. Scalable quantitative verification for deep neural networks. In 2021 IEEE/ACM 43rd International Conference on Software Engineering (ICSE), pages 312–323. IEEE, 2021.

[7] Håkan LS Younes and Reid G Simmons. Probabilistic verification of discrete event systems using acceptance sampling. In International Conference on Computer Aided Verification, pages 223–235. Springer, 2002.

[8] Axel Legay, Benoît Delahaye, and Saddek Bensalem. Statistical model checking: An overview. In International conference on runtime verification, pages 122–135. Springer, 2010.

[9] Zihao Ye, Ruihang Lai, Junru Shao, Tianqi Chen, and Luis Ceze. Sparsetir: Composable abstractions for sparse compilation in deep learning. In Proceedings of the 28th ACM International Conference on

Architectural Support for Programming Languages and Operating Systems, Volume 3, pages 660–678, 2023.

[10] Gus Henry Smith, Andrew Liu, Steven Lyubomirsky, Scott Davidson, Joseph McMahan, Michael Taylor, Luis Ceze, and Zachary Tatlock. Pure tensor program rewriting via access patterns (representation pearl). In Proceedings of the 5th ACM SIGPLAN International Symposium on Machine Programming, pages 21–31, 2021.

[11] Fuxun Yu, Di Wang, Longfei Shangguan, Minjia Zhang, Xulong Tang, Chenchen Liu, and Xiang Chen. A survey of large-scale deep learning serving system optimization: Challenges and opportunities. arXiv preprint arXiv:2111.14247, 2021.

[12] Lianmin Zheng, Chengfan Jia, Minmin Sun, Zhao Wu, Cody Hao Yu, Ameer Haj-Ali, Yida Wang, Jun Yang, Danyang Zhuo, Koushik Sen, et al. Ansor: Generating {High-Performance} tensor programs for deep learning. In 14th USENIX symposium on operating systems design and implementation (OSDI 20), pages 863–879, 2020.

[13] Martín Abadi, Paul Barham, Jianmin Chen, Zhifeng Chen, Andy Davis, Jeffrey Dean, Matthieu Devin, Sanjay Ghemawat, Geoffrey Irving, Michael Isard, et al. {TensorFlow}: a system for {Large-Scale} machine learning. In 12th USENIX symposium on operating systems design and implementation (OSDI 16), pages 265–283, 2016.

[14] Xufan Zhang, Ning Sun, Chunrong Fang, Jiawei Liu, Jia Liu, Dong Chai, Jiang Wang, and Zhenyu Chen. Predoo: precision testing of deep learning operators. In Proceedings ofthe 30th ACM SIGSOFT International Symposium on Software Testing and Analysis, pages 400–412, 2021.

[15] Anthony Di Franco, Hui Guo, and Cindy Rubio-González. A comprehensive study of real-world numerical bug characteristics. In 2017 32nd IEEE/ACM International Conference on Automated Software Engineering (ASE), pages 509–519. IEEE, 2017.

[16] Hao Guan, Ying Xiao, Jiaying Li, Yepang Liu, and Guangdong Bai. A comprehensive study of real-world bugs in machine learning model optimization. In Proceedings of the 45th IEEE/ACM International Conference on Software Engineering (ICSE), pages 147–158, 2023.

[17] Florian Benz, Andreas Hildebrandt, and Sebastian Hack. A dynamic program analysis to find floating-point accuracy problems. ACM SIGPLAN Notices, 47(6):453–462, 2012.

[18] Wei-Fan Chiang, Ganesh Gopalakrishnan, Zvonimir Rakamaric, and Alexey Solovyev. Efficient search for inputs causing high floating-point errors. In Proceedings of the 19th ACM SIGPLAN symposium on Principles and practice of parallel programming, pages 43–52, 2014.

[19] Eliska Kloberdanz, Kyle G Kloberdanz, and Wei Le. Deepstability: A study of unstable numerical methods and their solutions in deep learning. In Proceedings of the 44th international conference on software engineering, pages 586–597, 2022.

[20] Alex Sanchez-Stern, Pavel Panchekha, Sorin Lerner, and Zachary Tatlock. Finding root causes of floating point error. In Proceedings of the 39th ACM SIGPLAN Conference on Programming Language Design and Implementation, pages 256–269, 2018.

[21] Wei Hao, Aahil Awatramani, Jiayang Hu, Chengzhi Mao, Pin-Chun Chen, Eyal Cidon, Asaf Cidon, and Junfeng Yang. A tale of two models: Constructing evasive attacks on edge models. Proceedings of Machine Learning and Systems, 4:414–429, 2022.

[22] Simin Chen, Shiyi Wei, Cong Liu, and Wei Yang. Dycl: Dynamic neural network compilation via program rewriting and graph optimization. In Proceedings ofthe 32nd ACM SIGSOFT International Symposium on Software Testing and Analysis, pages 614–626, 2023.

[23] David Monniaux. The pitfalls of verifying floating-point computations. ACM Transactions on Programming Languages and Systems (TOPLAS), 30(3):1–41, 2008.

[24] Jiawei Liu, Jinkun Lin, Fabian Ruffy, Cheng Tan, Jinyang Li, Aurojit Panda, and Lingming Zhang. Nnsmith: Generating diverse and valid test cases for deep learning compilers. In Proceedings of the 28th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2, pages 530–543, 2023.

[25] Herman Chernoff. A measure of asymptotic efficiency for tests of a hypothesis based on the sum of observations. The Annals of Mathematical Statistics, pages 493–507, 1952.

[26] Wassily Hoeffding. Probability inequalities for sums of bounded random variables. Journal of the American statistical association, 58(301):13–30, 1963.

[27] Charles J Clopper and Egon S Pearson. The use of confidence or fiducial limits illustrated in the case of the binomial. Biometrika, 26(4):404–413, 1934.

[28] Alan Agresti and Brent A Coull. Approximate is better than “exact” for interval estimation of binomial proportions. The American Statistician, 52(2):119–126, 1998.

[29] Lawrence D Brown, T Tony Cai, and Anirban DasGupta. Interval estimation for a binomial proportion. Statistical science, pages 101–117, 2001.

[30] Olive Jean Dunn. Multiple comparisons among means. Journal of the American statistical association, 56(293):52–64, 1961.

[31] Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner. Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11):2278–2324, 1998.

[32] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 770–778, 2016.

[33] Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556, 2014.

[34] Forrest Iandola, Matt Moskewicz, Sergey Karayev, Ross Girshick, Trevor Darrell, and Kurt Keutzer. Densenet: Implementing efficient convnet descriptor pyramids. arXiv preprint arXiv:1404.1869, 2014.

[35] Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, and Liang-Chieh Chen. Mobilenetv2: Inverted residuals and linear bottlenecks. In 2018 IEEE/CVF conference on computer vision and pattern recognition, pages 4510–4520. Ieee, 2018.

[36] Alexey Dosovitskiy. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020.

[37] Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, and Hervé Jégou. Training data-efficient image transformers & distillation through attention. In International conference on machine learning, pages 10347–10357. PMLR, 2021.

[38] Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images. 2009.

[39] Yuval Netzer, Tao Wang, Adam Coates, Alessandro Bissacco, Baolin Wu, Andrew Y Ng, et al. Reading digits in natural images with unsupervised feature learning. In NIPS workshop on deep learning and unsupervisedfeature learning, volume 2011, page 4. Granada, 2011.

[40] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pages 248–255. Ieee, 2009.

[41] Stefan Webb, Tom Rainforth, Yee Whye Teh, and M Pawan Kumar. A statistical approach to assessing neural network robustness. arXiv preprint arXiv:1811.07209, 2018.

[42] Haoyu Wang, Junjie Chen, Chuyue Xie, Shuang Liu, Zan Wang, Qingchao Shen, and Yingquan Zhao. Mlirsmith: Random program generation for fuzzing mlir compiler infrastructure. In 2023 38th IEEE/ACM International Conference on Automated Software Engineering (ASE), pages 1555–1566. IEEE, 2023.

[43] Zihan Wang, Pengbo Nie, Xinyuan Miao, Yuting Chen, Chengcheng Wan, Lei Bu, and Jianjun Zhao. Gencog: A dsl-based approach to generating computation graphs for tvm testing. In Proceedings of the 32nd ACM SIGSOFT International Symposium on Software Testing and Analysis, pages 904–916, 2023.

[44] Jiawei Liu, Yuxiang Wei, Sen Yang, Yinlin Deng, and Lingming Zhang. Coverage-guided tensor compiler fuzzing with joint ir-pass mutation. Proceedings of the ACM on Programming Languages, 6

(OOPSLA1):1–26, 2022.

[45] Hung Viet Pham, Thibaud Lutellier, Weizhen Qi, and Lin Tan. Cradle: cross-backend validation to detect and localize bugs in deep learning libraries. In 2019 IEEE/ACM 41st International Conference on Software Engineering (ICSE), pages 1027–1038. IEEE, 2019.

[46] Zan Wang, Ming Yan, Junjie Chen, Shuang Liu, and Dongdi Zhang. Deep learning library testing via effective model generation. In Proceedings of the 28th ACM joint meeting on european software engineering conference and symposium on the foundations of software engineering, pages 788–799, 2020.

[47] Qianyu Guo, Xiaofei Xie, Yi Li, Xiaoyu Zhang, Yang Liu, Xiaohong Li, and Chao Shen. Audee: Automated testing for deep learning frameworks. In Proceedings of the 35th IEEE/ACM international conference on automated software engineering, pages 486–498, 2020.

[48] Yuhui Li, Fangyun Wei, Chao Zhang, and Hongyang Zhang. Eagle: Speculative sampling requires rethinking feature uncertainty. arXiv preprint arXiv:2401.15077, 2024.

[49] Dongwei Xiao, Zhibo Liu, Yuanyuan Yuan, Qi Pang, and Shuai Wang. Metamorphic testing of deep learning compilers. Proceedings of the ACM on Measurement and Analysis of Computing Systems, 6(1): 1–28, 2022.

[50] Purvish Jajal, Wenxin Jiang, Arav Tewari, Erik Kocinare, Joseph Woo, Anusha Sarraf, Yung-Hsiang Lu, George K Thiruvathukal, and James C Davis. Interoperability in deep learning: A user survey and failure analysis of onnx model converters. In Proceedings of the 33rd ACM SIGSOFT International Symposium on Software Testing and Analysis, pages 1466–1478, 2024.

[51] Qingchao Shen, Yongqiang Tian, Haoyang Ma, Junjie Chen, Lili Huang, Ruifeng Fu, Shing-Chi Cheung, and Zan Wang. A tale of two dl cities: When library tests meet compiler. arXiv preprint arXiv:2407.16626, 2024.

[52] Xiangxiang Chen, Xingwei Lin, Jingyi Wang, Jun Sun, Jiashui Wang, and Wenhai Wang. Scuzer: A scheduling optimization fuzzer for tvm. ACM Transactions on Software Engineering and Methodology, 34(4):1–28, 2025.

[53] Kuiliang Lin, Xiangpu Song, Yingpei Zeng, and Shanqing Guo. Deepdiffer: Find deep learning compiler bugs via priority-guided differential fuzzing. In 2023 IEEE 23rd International Conference on Software Quality, Reliability, and Security (QRS), pages 616–627. IEEE, 2023.

[54] Zhihao Jia, Oded Padon, James Thomas, Todd Warszawski, Matei Zaharia, and Alex Aiken. Taso: optimizing deep learning computation with automatic generation of graph substitutions. In Proceedings of the 27th ACM Symposium on Operating Systems Principles, pages 47–62, 2019.

[55] Haojie Wang, Jidong Zhai, Mingyu Gao, Zixuan Ma, Shizhi Tang, Liyan Zheng, Yuanzhi Li, Kaiyuan Rong, Yuanyong Chen, and Zhihao Jia. {PET}: Optimizing tensor programs with partially equivalent transformations and automated corrections. In 15th USENIX Symposium on Operating Systems Design and Implementation (OSDI 21), pages 37–54, 2021.

[56] Dayou Yu, Weishi Shi, and Qi Yu. Actively testing your model while it learns: realizing label-efficient learning in practice. Advances in Neural Information Processing Systems, 36:31404–31427, 2023.

[57] Jai Arora, Sirui Lu, Devansh Jain, Tianfan Xu, Farzin Houshmand, Phitchaya Mangpo Phothilimthana, Mohsen Lesani, Praveen Narayanan, Karthik Srinivasa Murthy, Rastislav Bodik, et al. Tensorright: Automated verification of tensor graph rewrites. Proceedings of the ACM on Programming Languages, 9(POPL):832–863, 2025.

[58] Brandon Paulsen, Jingbo Wang, and Chao Wang. Reludiff: Differential verification of deep neural networks. In Proceedings of the ACM/IEEE 42nd International Conference on Software Engineering, pages 714–726, 2020.

[59] Brandon Paulsen, Jingbo Wang, Jiawei Wang, and Chao Wang. Neurodiff: scalable differential verification of neural networks using fine-grained approximation. In Proceedings ofthe 35th IEEE/ACM International Conference on Automated Software Engineering, pages 784–796, 2020.

[60] Yedi Zhang, Fu Song, and Jun Sun. Qebverif: Quantization error bound verification of neural networks. In International Conference on Computer Aided Verification, pages 413–437. Springer, 2023.

[61] Mirco Giacobbe, Thomas A Henzinger, and Mathias Lechner. How many bits does it take to quantize your neural network? In International Conference on Tools and Algorithms for the Construction and Analysis ofSystems, pages 79–97. Springer, 2020.

[62] Xiaofei Xie, Lei Ma, Haijun Wang, Yuekang Li, Yang Liu, and Xiaohong Li. Diffchaser: Detecting disagreements for deep neural networks. International Joint Conferences on Artificial Intelligence Organization, 2019.

[63] Yiming Chen, Simin Chen, Zexin Li, Wei Yang, Cong Liu, Robby Tan, and Haizhou Li. Dynamic transformers provide a false sense of efficiency. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (ACL), pages 7164–7180, 2023.

[64] Yufei Li, Zexin Li, Yingfan Gao, and Cong Liu. White-box multi-objective adversarial attack on dialogue generation. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (ACL), pages 1778–1792, 2023.

[65] Yiming Chen, Zexin Li, Xianghu Yue, Robby T. Tan, and Haizhou Li. NaturalSloth: Revisiting denial-of-service attacks on large language models. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (ACL), pages 19685–19702, 2026.

[66] Simin Chen, Yiming Chen, Zexin Li, Yifan Jiang, Zhongwei Wan, Yixin He, Dezhi Ran, Tianle Gu, Haizhou Li, Tao Xie, and Baishakhi Ray. Benchmarking large language models under data contamination: A survey from static to dynamic evaluation. In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 10080–10098, 2025.

[67] Yiming Zeng, Wanhao Yu, Zexin Li, Tao Ren, Yu Ma, Jinghan Cao, Xiyan Chen, and Tingting Yu. Bridging the editing gap in LLMs: FineEdit for precise and targeted text modifications. In Findings of the Association for Computational Linguistics: EMNLP 2025, pages 2193–2206, 2025.

[68] Mathias Lecuyer, Vaggelis Atlidakis, Roxana Geambasu, Daniel Hsu, and Suman Jana. Certified robustness to adversarial examples with differential privacy. arXiv preprint arXiv:1802.03471, 2018.