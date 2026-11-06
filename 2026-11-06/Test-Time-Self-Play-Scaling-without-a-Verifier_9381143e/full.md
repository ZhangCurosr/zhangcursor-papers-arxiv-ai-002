# Test-Time Self-Play: Scaling without a Verifier

Akira Yoshiyama∗ ETH Zurich & Tufa Labs

Akmal Ashirmatov∗ ETH Zurich

Jonas Hübotter ETH Zurich

Ido Hakimi ETH Zurich

Andreas Krause ETH Zurich

## Abstract

Test-time scaling methods spend additional compute at deployment to improve a large language model’s performance on a hard target problem. Most methods rely on a verifier that scores candidate solutions for correctness, using it to select among or refine the model’s outputs. In many deployment regimes, however, a strong verifier on the target problem is unavailable. We introduce Test-Time Self-Play (TTSP), an approach to test-time scaling using asymmetric self-play without reliance on a verifier. On a difficulty-calibrated subset of LiveCodeBench v6, TTSP is the bestperforming method at every verifier strength we evaluate, and the gap to the strongest baseline widens as the verifier weakens; in the no-verifier limit, TTSP yields a 35% absolute improvement in mean@8. The gains hold across a range of verifier strengths, except when the base policy has near-zero coverage of the correct answer.

## 1 Introduction

How can a model improve on a single difficult target problem when it has no related training data and no reliable way to check its own answers? This question arises whenever a large language model’s underlying knowledge is, in principle, sufficient to solve a problem, yet the model rarely, or never, generates the correct solution. Examples of this can be found across domains, including in competitive programming (Zheng et al., 2025), theorem proving (Tsoukalas et al., 2024), and molecule generation (Gao et al., 2022).

Test-time scaling aims to close this gap, spending additional compute at deployment to improve performance on the target problem. Many test-time scaling methods, such as rejection sampling (Snell et al., 2024), rely on a verifier for the target problem that determines whether an attempt is

![](images/f778dc84331c4ff49d73c076ef51b1803949231b21af8f0a12ebf84d80e47795.jpg)  
Figure 1: TTSP outperforms all baseline testtime scaling methods in the weak verifier regime. TTSP and ACE are evaluated at converged checkpoints; FMV uses public and private input test cases.

correct. Similarly, prompt-optimization methods such as ACE (Zhang et al., 2026a) that use execution feedback for evolving prompts. Other methods rely on auxiliary data, typically similar problems retrieved from a large corpus and used to train on a neighborhood of the target problem (Hübotter et al., 2025; Hübotter et al., 2026). Another set of methods requires neither verifier nor auxiliary training data, instead aggregating a signal from the model’s

![](images/97cb08e1773b5fcc37574a87f2887135a9e15a9899e6be85ec06fa4687aab637.jpg)  
Figure 2: Overview of Test-Time Self-Play. Given a target problem $x ^ { \star }$ and previously generated synthetic problems, the proposer policy $\pi _ { \phi }$ generates target-conditioned synthetic problems $\check { \mathcal { D } }$ consisting of a problem statement, tests, and a reference solution. The solver policy is rewarded for correctly solving the synthetic problems, while the proposer is rewarded for generating synthetic problems near the solver’s current learning frontier (both via GRPO; see the red arrows). The synthetic problem history provides context for the proposer and fills a replay buffer . After a sufficient number of iterations, the test-time trained solver can then attempt the target problem.

own rollouts on the target problem through majority voting (Zuo et al., 2025; Du et al., 2026;   
Shafayat et al., 2025; Launer et al., 2026).

We propose Test-Time Self-Play (TTSP), a test-time scaling method that does not rely on auxiliary training data or strong verifiers. TTSP is inspired by training-time asymmetric self-play methods such as Absolute Zero (Zhao et al., 2025) and R-Zero (Huang et al., 2025), which learn from self-generated synthetic problems alone. We adapt this approach to the single-target setting, concentrating the synthetic problems on a target problem rather than across a benchmark distribution. The only external requirement of TTSP is an interactive environment, such as a Python interpreter, to ground the model.

Our contributions are the following. (i) We define the test-time scaling setting with no auxiliary data and absent or weak verifiers, and characterize the behavior of existing testtime scaling methods. (ii) We propose TTSP, a test-time scaling method based on asymmetric self-play that is robust to weak verifiers.

## 2 Related Work

Test-time scaling. Spending additional compute at deployment to improve performance on a target problem has been studied under the umbrellas of test-time scaling (Snell et al., 2024; Brown et al., 2024) and test-time training (TTT) (Sun et al., 2020). Sampling-based testtime scaling repeatedly draws rollouts from the policy and selects among them with a verifier or vote (Brown et al., 2024; Launer et al., 2026). Weight-updating TTT specializes a model’s weights to each target problem, with success across abstract reasoning (Akyürek et al., 2025), language modeling (Hardt & Sun, 2024; Hübotter et al., 2026; Sun et al., 2025), and video generation (Dalal et al., 2025), using supervised losses on target problem neighborhoods drawn from external data (Hardt & Sun, 2024; Hübotter et al., 2026; Bertolissi et al., 2025; Bagatella et al., 2025) or self-supervised losses over the input sequence (Sun et al., 2025; Dalal et al., 2025). Most relevant to our work, Zuo et al. (2025) extend TTT to RL on the test set using majority votes as pseudo-labels, while Simonds & Yoshiyama (2025) generate problem variants at test-time and train via RL with numerical verification in symbolic integration. This connects to broader themes of unsupervised RL (Zhang et al., 2025; Agarwal et al., 2025; Zhang et al., 2026b; Du et al., 2026) and self-improvement in LLMs (Zelikman et al., 2022; Gulcehre et al., 2023; Lee et al., 2025). How best to derive a strong learning signal for test-time scaling from the model alone, without recourse to external data or supervision, remains an open question.

Curriculum learning. Introduced by Bengio et al. (2009), curriculum learning is the principle that ordering training examples from easy to hard accelerates optimization and improves generalization. It has since become a central design pattern in modern LLM training, underpinning pretraining and mid-training (DoReMi, Xie et al. 2023; OLMo 2, Team OLMo et al. 2025; Phi-4, Abdin et al. 2024) and, especially, RL-based reasoning post-training, where DeepSeek-R1 (Guo et al., 2025), Kimi k1.5 (Kimi Team et al., 2025), and adaptive schemes such as Self-Evolving Curriculum (Chen et al., 2025b) and easy-to-hard generalization (Sun et al., 2024) consistently outperform uniform sampling.

Self-play. Self-play underlies many landmark results in reinforcement learning (Schmidhuber, 1991; Tesauro, 1995; Silver et al., 2016; 2017; 2018; Berner et al., 2019). Unlike fixed datasets, the self-play curriculum co-evolves with the learner and is in principle unbounded, a property recently argued to be central to continued progress beyond the limits of humangenerated data (Silver & Sutton, 2025). Recent work imports this property into LLM posttraining by assigning a model dual proposer policy and solver policy roles with learning signal from automatic verifiers such as code execution, majority vote, or proof checkers (Zhao et al., 2025; Huang et al., 2025; Chen et al., 2025a; Kuba et al., 2025; Liu et al., 2026). However, the resulting curricula tend to concentrate on narrow regions of the task space, and gains from fully data-free self-play are often bounded by the base model’s coverage (Chae et al., 2025). Prior self-play methods for LLMs operate at training time and target broad capability gains across a benchmark distribution. We instead apply self-play at test-time, using target problem decomposition to build a specialized curriculum that concentrates adaptation on the target problem.

Context optimization. A complementary line of work adapts LLM behavior by modifying the context rather than the model weights. GEPA (Agrawal et al., 2026) optimizes textual parameters through reflective prompt evolution, using verifier feedback to propose mutations and Pareto-style selection to retain complementary prompt variants. ACE (Zhang et al., 2026a) extends this paradigm with a structured generator–reflector–curator decomposition: a generator attempts the task, a reflector extracts reusable lessons from execution feedback, and a curator incrementally updates a long-form playbook context while preserving useful details. This decomposition lets ACE accumulate more fine-grained domain knowledge than the prompt-level mutations of GEPA, and ACE reports consistent gains over GEPA across reasoning and tool-use benchmarks.

## 3 Preliminaries

We study test-time scaling with individual target problems. Let $x ^ { \star }$ denote a target problem and let y denote a candidate solution. A policy π generates rollouts $y \sim \pi ( { \bf \ddot { \cdot } } \mid { \bf \hat { \sigma } } { \bf \hat { \times } } )$ . The oracle verifier,

$$
v ^ { \star } ( y ; x ^ { \star } ) \in \{ 0 , 1 \} ,
$$

indicates whether y is a correct solution for $x ^ { \star }$ . The oracle is used for evaluation, but is not assumed to be available for training or feedback at test-time. A deployment-time target verifier v may be weak or absent. We further assume that no auxiliary training data (i.e., problems and their ground truth solutions $( x , y )$ related to $x ^ { \star } )$ are available. We ask whether a test-time scaling method can be made robust in this setting.

We propose that this is possible by deriving a learning signal from synthetic problems instead of directly from the target problem. A synthetic problem xˆ is a self-contained problem generated at test-time, conditioned on $x ^ { \star }$ . We evaluate TTSP in the setting of competitive programming, in which xˆ consists of a natural-language statement, an input/output specification, an example input/output pair, held-out input test cases, and a reference Python program. The reference program is not shown to the solver. Executing the reference solution on the held-out input test cases produces the ground truth outputs, which combined with the inputs define a synthetic verifier

$$
v _ { \mathrm { s y n } } ( \hat { y } ; \hat { x } ) \in \{ 0 , 1 \}
$$

The verifier identifies whether the attempted solution $\hat { y }$ passes all held-out test cases. We define synthetic problems to be self-contained training tasks with a deterministic verification mechanism. We do not require a formal decomposition relation between synthetic problems

Algorithm 1 Test-Time Self-Play   
Require: Target problem $x ^ { \star } ,$ proposer policy $\pi _ { \phi } ,$ solver policy $\pi _ { \theta } ,$ , iterations T   
1: Initialize synthetic problem history $\mathbf { \mathcal { H } }  \mathbf { \mathcal { O } }$   
2: for $t = 1$ to T do   
3: $\mathcal { D } _ { t } , \mathcal { B } _ { t } , \mathcal { R } _ { \theta } , \mathcal { R } _ { \phi }  \emptyset$   
4: // Generate synthetic problems   
5: $\mathcal { C } _ { t }  \mathsf { C }$ ONSTRUCTCONTEXTS $( x ^ { \star } , \mathcal { H } )$   
6: for each $c \in { \mathcal { C } } _ { t }$ do   
7: $\{ \hat { x } ^ { ( i ) } \} \sim \pi _ { \phi } ( \cdot | c )$   
8: $\mathcal { A } , \mathcal { T } \gets \mathrm { F I L T E R } \big ( \{ \hat { x } ^ { ( i ) } \} \big )$   
9: $\mathcal { D } _ { t }  \mathcal { D } _ { t } \cup \mathcal { A }$   
10: $\mathcal { R } _ { \phi }  \mathcal { R } _ { \phi } \cup$ INVALIDPENALTY( )   
11: end for   
12: // Samplefrom history   
13: <sub>t</sub> SAMPLEHISTORY( )   
14: // Solver policy attempts synthetic problems   
15: for each $\hat { x } \in \mathcal { D } _ { t } \cup B _ { t } ^ { i }$ do   
16: $\{ \hat { y } ^ { ( j ) } \} \sim \pi _ { \theta } ( \cdot \mid \hat { x } )$   
17: $\{ r _ { \theta } ^ { ( j ) } \}  v _ { \mathrm { s y n } } ( \{ \hat { y } ^ { ( j ) } \} ; \hat { x } )$   
18: $\mathcal { R } _ { \theta }  \mathcal { R } _ { \theta } \cup \{ ( \hat { x } , \hat { y } ^ { ( j ) } , r _ { \theta } ^ { ( j ) } ) \}$   
19: UPDATEHISTORY $( \mathcal { H } , \hat { x } , \{ \hat { y } ^ { ( j ) } \} , \{ r _ { \theta } ^ { ( j ) } \} )$   
20: end for   
21: // Assign proposer policy rewards   
22: $\mathcal { R } _ { \phi }  \mathcal { R } _ { \phi }$ PROPOSERREWARD $( \mathcal { D } _ { t } , \mathcal { R } _ { \theta } )$   
23: // Update proposer policy and solver policy   
24: Update solver policy $\pi _ { \theta }$ with GRPO using $\scriptstyle { \mathcal { R } } _ { \theta }$   
25: Update proposer policy $\pi _ { \phi }$ with GRPO using $\mathcal { R } _ { \phi }$   
26: end for   
27: return adapted solver policy $\pi _ { \theta }$

$\hat { x }$ and $x ^ { \star }$ ; the target problem dependence in TTSP is induced by conditioning the proposer on $x ^ { \star }$ and the synthetic problem history.

## 4 TTSP: Test-Time Self-Play

Test-Time Self-Play (TTSP) adapts a solver policy to a single target problem $x ^ { \star }$ by training on target-conditioned synthetic problems. A proposer policy generates synthetic problems, a solver policy attempts them, execution supplies rewards, and both policies are updated with GRPO. The target problem verifier is not used during TTSP training. Figure 2 illustrates the asymmetric self-play loop.

Self-play setup. TTSP uses two LoRA adapters over a shared frozen Qwen3-8B backbone (Hu et al., 2022; Yang et al., 2025): a proposer policy $\pi _ { \phi }$ and a solver policy $\pi _ { \theta }$ . The proposer policy designs a synthetic curriculum conditioned on $x ^ { \star }$ and the training history, while the solver policy trains on the resulting synthetic problems. Separating the adapters gives each role its own reward signal and avoids the gradient conflicts we observed when using a single adapter for both roles. The memory cost remains close to a single 8B model, since the backbone is shared and only the adapters are updated.

Algorithm overview. A step, t, of the TTSP algorithm begins with the construction of the proposer’s prompt. The prompt (or context) contains a small sample of previously generated synthetic problems and the their solve rates (from the solver policy). The context also includes the target problem and instructions guiding the proposer to generate valid, relevant and appropriately challenging synthetic problems based on previous synthetic problems. The prompt in the first step does not contain previous synthetic problems.

The proposer then generates candidate synthetic problems and reference programs. Invalid candidates are removed by a domain-specific heuristic filter FILTER. The valid synthetic problems $\mathcal { D } _ { t }$ are added to a buffer of old synthetic problems $B _ { t }$ which are sampled from the history of synthetic problems . Problems within the range of solve rates [0.1, 0.8] are uniformly randomly sampled.

The solver policy then generates solutions for the combined set of synthetic problems $\mathcal { D } _ { t } \cup \mathcal { B } _ { t }$ . The synthetic reward is calculated for each solution.

The history is updated with the new synthetic problems, solver rollouts and rewards. The solver is then updated with GRPO using the synthetic rewards. The proposer is updated with GRPO using a transformation of the solver rewards PROPOSER $\operatorname { R E W A R D } ( \mathcal { D } _ { t } , \mathcal { R } _ { \theta } ^ { \bullet } )$ . The transformation is described later in this section.

After a sufficient number of steps, the aim is for the solver policy to have a more accurate distribution of solutions on the target problem $x ^ { \star }$ . The solver can then be used to attempt the target problem.

Synthetic problems provide a dense training signal where direct training on $x ^ { \star }$ would be sparse, misleading (due to weak target problem verification) or unavailable (if no target problem verifier is available). TTSP produces many target-conditioned synthetic problems that are both at the edge of the solver’s abilities (due to resampling of previous problems in the optimal difficulty range, and proposer reward transformation) and anchored to the target problem (due to the context construction). See Algorithm 1 for the TTSP pseudocode.

Synthetic problem generation. The proposer generates each synthetic problem in two stages. First, solution proposal conditions on $x ^ { \star }$ and the recent synthetic problem history $\mathcal { H } ,$ , then produces a draft synthetic problem and a Python reference program. Second, problem refinement reconstructs a clean problem statement, input/output specification, example $\mathrm { I } / \mathrm { O } ,$ and held-out input tests from the reference program. The held-out outputs are obtained by executing the reference program. This ordering makes the tests and statement conform to the program’s actual behavior, rather than requiring a single decoding pass to keep the statement, tests, and solution mutually consistent. The subgoal generation prompt templates are listed in $\mathsf { A p - }$ pendix F.

We then apply FILTER, which rejects candidates with timeouts, runtime errors, duplicate-only test sets, or constant-output behavior. Accepted problems enter the

Table 1: Example synthetic curriculum. The synthetic problems are generated by the proposer, teaching the solver concepts which are useful to recover the final strategy needed to solve the target problem.
<table><tr><td>Target Problem  $x ^ { \star }$ </td><td>Concept</td></tr><tr><td>For each  $A _ { i } ,$  maximize the GCD of a size-K subset containing  $A _ { i } .$ </td><td>Largest feasible divisor of  $A _ { i }$ </td></tr><tr><td>Synthetic Problems</td><td></td></tr><tr><td>Find the largest integer d that divides at least  $\breve { K }$  elements of the array.</td><td>Feasibility predicate:  $\mathrm { c n t } [ d ] \geq \check { K }$ </td></tr><tr><td>For every divisor d, compute  $\mathrm { c n t } [ \dot { d } ] = | \{ j : d \mid A _ { j } \} |$ </td><td>Multiples-sieve divisor counts</td></tr><tr><td>For each  $A _ { i } ,$  output the largest divisor d |  $A _ { i }$  with  $\operatorname { c n t } [ d ] \geq K .$ </td><td>Per-element feasible divisor search</td></tr></table>

fresh-problem set $\mathcal { D } _ { t }$ and are added to the history after the solver attempts them. Invalid candidates do not enter the solver batch and receive a fixed proposer penalty.

Synthetic problem history. TTSP maintains a history  of accepted synthetic problems and the solver’s most recent rollouts on them. At each iteration, the solver batch mixes new synthetic problems $\mathcal { D } _ { t }$ with a buffer of previous problems $B _ { t }$ sampled from . The sampler favors problems whose current solve rate lies in a learnability window: already-solved problems provide little training signal, while near-impossible problems produce nearly all-zero rewards. Since UPDATEHISTORY overwrites each problem’s latest rollouts and solve rate, both proposer contexts and buffer sampling track the current solver rather than a stale snapshot.

Rewards. The solver receives a binary synthetic reward

$$
\begin{array} { r } { r _ { \theta } ( \hat { y } , \hat { x } ) = v _ { \mathrm { s y n } } ( \hat { y } ; \hat { x } ) , } \end{array}
$$

which equals one if and only if the solver rollout yˆ passes all held-out generated tests for synthetic problem xˆ.

The proposer is rewarded for generating problems near the solver’s current frontier. Let $\bar { r } _ { \theta } ( \hat { x } )$ be the mean solver reward over rollouts on xˆ. For accepted synthetic problems, we define

$$
\begin{array} { r } { r _ { \phi } ( \hat { x } ) = 1 - 2 \left| \bar { r } _ { \theta } ( \hat { x } ) - \frac { 1 } { 2 } \right| , } \end{array}\tag{1}
$$

with a fixed penalty for invalid generations. The triangular profile targets the region where sparse Bernoulli rewards induce the largest group-relative signal under GRPO, so the proposer policy is rewarded for keeping the solver policy near its frontier (Hübotter et al., 2025). The buffer eligibility window is chosen to match the high-reward part of this profile.

Optimization. We optimize both adapters with GRPO (Shao et al., 2024), DAPO-style asymmetric clipping (Yu et al., 2025), and zero KL regularization following the test-time adaptation setup of Hübotter et al. (2025). Hyperparameters are listed in Appendix B.1.

## 5 Experiments

## 5.1 Experimental Setup

Task domain. We evaluate TTSP in the domain of competitive programming. We use the benchmark LiveCodeBench v6 (Jain et al., 2024), which contains 131 target problems released between February and May 2025, after the Qwen3-8B knowledge cutoff. The benchmark provides public test cases, private test cases, and a broad range of problem difficulties. This lets us vary verifier strength and target problem difficulty within a single controlled evaluation setting.

Table 2: Required verifier access per method. Each method requires test case inputs, test case outputs, neither, or both, from which a learning signal can be derived.
<table><tr><td>Method</td><td>Verifier Access</td></tr><tr><td>Rejection sampling</td><td>Inputs + Outputs</td></tr><tr><td>Context optimization</td><td>Inputs + Outputs</td></tr><tr><td>Majority voting</td><td>Inputs Only</td></tr><tr><td>TTSP (ours)</td><td>None</td></tr></table>

Target problem set. From LiveCodeBench ${ \mathrm { v } } 6 ,$ we keep problems satisfying $0 \_ <$ mean@512 $< 0 . 7$ for Qwen3-8B and name the resulting target problem set HardLCBv6. Qwen3-8B has a wide range of non-zero accuracies over this set of problems.

We further split HardLCBv6 into MidHardL ${ \it C } \mathrm { B v } 6 ,$ containing target problems with $0 . 1 <$ mean@512 $\overset { \cdot } { < } 0 . 7$ , and VeryHardLCBv6, containing target problems with $0 < \mathrm { m e a n } @ 5 1 2 \le$ 0.1. We evaluate TTSP and baselines on these subsets of LiveCodeBench v6. Qwen3-8B solve rates for each target problem are listed in Appendix A.1.

Verifier construction. We evaluate seven verifier conditions. Four are artificial verifiers, $v _ { 0 } , v _ { 2 5 } , v _ { 5 0 }$ , and $v _ { 7 5 } ,$ calibrated to reject the indicated fraction of incorrect sampled solutions; see $\mathsf { A p } _ { \mathsf { I } }$ pendix $_ \mathrm { A . 4 } ^ { }$ . Three are LiveCodeBench verifiers: public test cases $( \boldsymbol { v } _ { \mathrm { p u b } } ) _ { . }$ , private test cases $( v _ { \mathrm { p r i v } } )$ , and public + private test cases $( v ^ { \star } )$ . The verifier $v _ { 0 }$ accepts every rollout, while $v ^ { \star }$ is the oracle verfier. These enable a closer investigation of a test-time scaling method’s performance across various levels of weak (and absent, in the case of $v _ { 0 } )$ to strong (in the limit, v<sup>⋆</sup>) verifiers.

![](images/49ea105d8eefb94dbd810b0a0fab9f5ae4e9e13cd5d96c29dfd2316429b7a7b8.jpg)  
Figure 3: Baselines performance on HardLCBv6 as compute scales. Each line reports the mean@8 with the indicated evaluation-time verifier applied to the validation problem submissions. Left and middle: Qwen 3 8B and FMV scale on the number of sampled rollouts k. Right: ACE scales on the number of generation–reflection–curation steps.

## 5.2 Evaluation Protocol

Each method output is evaluated as a set of eight target problem submissions. For aggregate comparisons, this set comes from the converged or final method output; for training-curve diagnostics, the same definition is applied at each checkpoint t. Let $\dot { Y } = \dot { \{ { y _ { 1 } , \dots , { y _ { 8 } } \} } }$ denote the eight submitted candidates for $\stackrel { \star } { x ^ { \star } }$ . For a verifier $v \in \left\{ v _ { 0 } , v _ { 2 5 } , v _ { 5 0 } , v _ { 7 5 } , v _ { \mathrm { p u b } } , v _ { \mathrm { p r i v } } , v ^ { \star } \right\}$ , we form the admitted subset

$$
A ( v ) = \{ y \in Y : v ( y ) = 1 \} .
$$

For non-empty admitted sets, we report the filtered mean@8 score

$$
\mathrm { m e a n } @ 8 ( v ) = \frac { 1 } { | A ( v ) | } \sum _ { y \in A ( v ) } v ^ { \star } ( y ) .
$$

Under $v _ { 0 } ,$ no candidate is filtered out, so $A = Y$ and mean@8 is the fraction of all eight candidates that pass $v ^ { \star }$ . Throughout the paper, mean@8 denotes this filtered score; we write unfiltered mean@8 for the fraction of all eight submitted candidates that pass $v ^ { \star }$ with no evaluation-time filtering. Empty admitted sets are handled by the carry-forward rule in Appendix A.5. Since the same filtering pass is applied to all methods, differences in mean@8 reflect differences in candidate quality rather than differences in scoring.

## 5.3 Baselines

We compare TTSP against one representative method from each test-time scaling family in Table $2 { : }$ rejection sampling, majority voting, and context optimization. The table records the verifier access required by each method. The common filtered mean@8 protocol from Section 5.2 is then applied uniformly to all methods.

Rejection sampling. Our direct filtering baseline samples candidates from the unadapted Qwen3-8B policy. The model produces eight submissions for $x ^ { \star }$ , and the verifier is used only in the evaluation-time filtering pass described above. This baseline measures how much performance can be obtained from base-policy sampling without any prompt or weight adaptation.

Majority voting. Our consensus baseline is Functional Majority Voting (FMV) (Launer et al., 2026). The FMV voting rule itself is input-only: it clusters candidate programs by execution behavior on target problem input cases and selects the modal functional cluster. In the verifier-strength sweep, we combine this input-only voting rule with the current verifier condition by first discarding candidates that fail the verifier and then applying FMV to the admitted pool. This baseline measures whether target problem rollouts contain a recoverable correct functional cluster after the available verifier has removed some incorrect candidates. Appendix C.2 gives the exact filtering, scoring, and tie-handling procedure.

![](images/72b449642b2c4194db2979f78cc96fe5b5074901019b59916452de86e9ac00f7.jpg)  
Figure 4: Performance of TTSP on HardL-CBv6 over training steps. Each line reports the mean@8 at different evaluation-time verifier strengths applied to the solver policy’s target problem submissions. Shaded bands are 90% bootstrap confidence intervals over target problems.

![](images/2d74d771dcba67de5a58f74e9cbd9716e46ba93b51bb2ea0ef04f01be6f6ac14.jpg)  
Figure 5: Best performing method per problem and verifier strength. Each point is a (target problem, verifier strength) pair, colored by the method with higher mean@8. The x-axis orders verifier strength and the y-axis reports problem difficulty, measured as the base policy’s mean@512.

TTRL (Zuo et al., 2025), while well-known for its majority voting reward, is designed for benchmark-level adaptation, jointly training on a population of test problems via perprompt majority vote rewards, unlike our per-problem setting. TTRL is also only evaluated on math, whereas FMV was created for competitive programming, in which majority voting can be done in more strategic ways due to the existence of test cases.

Context optimization. Our prompt-optimization baseline is ACE (Zhang et al., 2026a). We run ACE separately at each verifier strength, using the pass/fail outcome under that verifier as the reward signal for its generator–reflector–curator loop. The evaluation-time filtering pass is then applied to ACE’s final submissions. We use ACE rather than GEPA (Agrawal et al., 2026) because ACE was the stronger context-optimization variant in our preliminary runs. Full implementation details are provided in Appendix C.1.

## 5.4 Results

Robustness across verifier strengths. TTSP is the best-performing method at every verifier strength we evaluate (see Figure 1). The TTSP curve is approximately flat across the full sweep, while all three baselines decline as the verifier weakens and the gap to TTSP widens accordingly. TTSP’s training-time reward is intrinsic to the self-play loop, so the verifier strength axis varies only at evaluation time; the baselines all depend on the verifier either for selection (rejection sampling, FMV) or as a reward signal (ACE), and their performance is bounded by the verifier’s discriminative power.

TTSP training dynamics. Figure 4 shows TTSP performance over training steps. Most of the improvement occurs within the first 10 training steps, after which the curves remain stable up to checkpoint-to-checkpoint noise. The spread between verifier-strength conditions also narrows during training. These dynamics suggest that the synthetic curricula quickly shift probability mass toward candidates that remain useful under later evaluation-time filtering.

Scaling baseline methods. Figure 3 reports a scaling study along each baseline’s native compute axis. The Qwen3-8B rejection sampling and FMV panels vary the rollout budget k, while the ACE panel varies the number of context optimization iterations and scores the current ACE submission. Qwen3-8B rejection sampling improves with larger rollout budgets primarily when the verifier is strong enough to reject incorrect candidates. FMV improves with additional rollouts because functional agreement becomes more informative. ACE improves over iterations only when the verifier provides informative pass/fail feedback; under $v _ { 0 } ,$ , it remains at the base-policy level because no optimization steps are taken. These results demonstrate that increasing compute does not remove the baselines’ dependence on target-verifier information.

Verifier-free setting. In the $v _ { 0 }$ setting, the verifier admits every candidate. In this setting, ACE is unable to improve because it is not provided any verification feedback. FMV remains limited to selecting among target problem rollouts sampled from the base policy, while TTSP improves substantially over all baselines.

Majority voting and solution rank. Figure 6 bins target problems by the rank of the correct functional cluster in the base policy’s rollout distribution. FMV is strongest when the correct cluster is modal, slightly exceeding TTSP performance. When the correct cluster is ranked second, FMV drops sharply and TTSP overtakes it. When the correct cluster has rank at least three, FMV is unable to identify the correct solution while TTSP retains non-trivial accuracy. We can observe that, while TTSP is impacted by correct solution output rank, it degrades gracefully as rank increases, whereas FMV is fully reliant on the correct solution being approximately the modal output.

![](images/ee15115b201b8f8dc9b95518630f15bf5460ca2ac94c688e328402b9e303a9f8.jpg)  
Figure 6: Unfiltered mean@8 of TTSP and FMV binned by rank of the correct functional cluster in the base policy. Bars group target problems by the rank of the correct functional cluster among base-policy rollouts.

TTSP is limited by target problem difficulty. TTSP requires the correct rollout to exist in the base policy’s support with nontrivial mass. Splitting HardLCBv6 into Mid-

HardLCBv6 and VeryHardLCBv6 reveals a sharp cutoff (Figure 7). On MidHardLCBv6, the improvement of TTSP over the base policy is small at strong verifiers and rises sharply as the verifier weakens. On VeryHardLCBv6, the improvement is approximately zero across the entire sweep — slightly negative at the strongest verifiers, where rejection sampling on the base policy can still surface the rare correct rollout, and only marginally positive at the weakest. Below mean@512  0.1, the proposer policy cannot bootstrap a curriculum that composes back to $x ^ { \star }$

When is TTSP most effective? Figure 5 summarizes the same pattern at the per-target level. TTSP dominates when base model coverage is non-trivial and the evaluation-time verifier is weak. Base model rejection sampling dominates when the verifier is strong enough to identify rare correct rollouts or when base model coverage is too low for TTSP to build a useful curriculum. The resulting boundary places TTSP in the consistency regime: it is most useful when correct or partially useful target-related behavior is present but not reliable under direct sampling.

## 6 Discussion

Test-Time Self-Play (TTSP) shifts the source of learning signal for test-time scaling. Existing methods derive their selection or learning signal from a verifier on the target problem, and their effectiveness degrades as that verifier weakens. In TTSP, a solver policy is instead trained on execution feedback from synthetic problems generated by a proposer policy (both initialized from the same base policy), with the target problem verifier left out of the training loop. The result is a test-time scaling algorithm that is substantially more robust to weak and absent target verifiers, conditional on the base policy having non-trivial coverage of the target problem $x ^ { \star }$

Limitations. TTSP adapts per target problem and does not amortize compute across targets, suiting deployment contexts where each target is valuable enough to merit a dedicated adaptation budget. TTSP also only addresses the consistency regime: when the base policy has effectively zero coverage of the target answer, no curriculum the proposer policy can synthesize will compose back to it (Figure 7). Furthermore, like other self-play algorithms, TTSP derives its training signal entirely from selfgenerated content rather than from any externally validated target. This makes training-time stability a concern that should be actively monitored. The proposer policy can drift toward synthetic problems that hack its reward function rather than ones that compose back to $x ^ { \star }$ , and the solver policy’s entropy can collapse under sustained training on a narrow distribution (Bailey et al., 2026). See Appendix D for further discussion of training dynamics.

![](images/83a2dd124de95e041c2577cd94ea59b18597cf2bfb904d4af4ec0f37c2290c87.jpg)  
Figure 7: Difference in mean@8 between TTSP and Qwen3-8B. The gap is large and positive on MidHardLCBv6 as the verifier weakens. On VeryHardLCBv6, the gap is near zero: only marginally positive under the weakest verifiers.

Future work. Several natural extensions follow from relaxing TTSP’s constraints. The proposer policy’s synthetic problem set can be augmented with x<sup>⋆</sup> itself whenever a nondegenerate target problem reward exists, combining curriculum-driven and target-driven training in one system. The set can also be augmented with nearest-neighbor synthetic problems retrieved from external corpora when those are available (Hübotter et al., 2025), ingesting auxiliary data without losing the curriculum structure. Furthermore, TTSP could be applied to other domains with rich tools, including domains with execution sandboxes, simulation engines, or database engines. Finally, tackling the discovery regime, where the base model has near-zero coverage of the target problem solution, remains an important open direction; it likely requires either external information or substantially stronger search than the curriculum alone provides.

## Acknowledgements

We thank Leander Diaz-Bone, Antonio Mari, Anton Baumann and Catherine Hou for helpful discussions. AY was supported by Tufa Labs AG. JH was supported by the Swiss National Science Foundation under NCCR Automation, grant agreement 51NF40 180545. IH was supported by an ETH AI Center Postdoctoral Fellowship.

This project was supported through the Swiss AI compute grant a156.

## References

Marah Abdin, Jyoti Aneja, Harkirat Behl, Sébastien Bubeck, Ronen Eldan, Suriya Gunasekar, Michael Harrison, Russell J. Hewett, Mojan Javaheripi, Piero Kauffmann, James R. Lee, Yin Tat Lee, Yuanzhi Li, Weishung Liu, Caio C. T. Mendes, Anh Nguyen, Eric Price, Gustavo de Rosa, Olli Saarikivi, Adil Salim, Shital Shah, Xin Wang, Rachel Ward, Yue Wu, Dingli Yu, Cyril Zhang, and Yi Zhang. Phi-4 technical report. arXiv preprint arXiv:2412.08905, 2024.

Shivam Agarwal, Zimin Zhang, Lifan Yuan, Jiawei Han, and Hao Peng. The unreasonable effectiveness of entropy minimization in LLM reasoning. arXiv preprint arXiv:2505.15134, 2025.

Lakshya A. Agrawal, Shangyin Tan, Dilara Soylu, Noah Ziems, Rishi Khare, Krista Opsahl-Ong, Arnav Singhvi, Herumb Shandilya, Michael J. Ryan, Meng Jiang, Christopher Potts, Koushik Sen, Alexandros G. Dimakis, Ion Stoica, Dan Klein, Matei Zaharia, and Omar Khattab. GEPA: Reflective prompt evolution can outperform reinforcement learning. In International Conference on Learning Representations (ICLR), 2026.

Ekin Akyürek, Mehul Damani, Adam Zweiger, Linlu Qiu, Han Guo, Jyothish Pari, Yoon Kim, and Jacob Andreas. The surprising effectiveness of test-time training for few-shot learning. In International Conference on Machine Learning (ICML), 2025.

Marco Bagatella, Mert Albaba, Jonas Hübotter, Georg Martius, and Andreas Krause. Test-time offline reinforcement learning on goal-related experience. arXiv preprint arXiv:2507.18809, 2025.

Luke Bailey, Kaiyue Wen, Kefan Dong, Tatsunori Hashimoto, and Tengyu Ma. Scaling self-play with self-guidance. arXiv preprint arXiv:2604.20209, 2026.

Yoshua Bengio, Jérôme Louradour, Ronan Collobert, and Jason Weston. Curriculum learning. In International Conference on Machine Learning (ICML), 2009.

Christopher Berner, Greg Brockman, Brooke Chan, Vicki Cheung, Przemysław D˛ebiak, Christy Dennison, David Farhi, Quirin Fischer, Shariq Hashme, Chris Hesse, Rafal Józefowicz, Scott Gray, Catherine Olsson, Jakub Pachocki, Michael Petrov, Henrique P. de Oliveira Pinto, Jonathan Raiman, Tim Salimans, Jeremy Schlatter, Jonas Schneider, Szymon Sidor, Ilya Sutskever, Jie Tang, Filip Wolski, and Susan Zhang. Dota 2 with large scale deep reinforcement learning. arXiv preprint arXiv:1912.06680, 2019.

Ryo Bertolissi, Jonas Hübotter, Ido Hakimi, and Andreas Krause. Local mixtures of experts: Essentially free test-time training via model merging. In Conference on Language Modeling (COLM), 2025.

Bradley Brown, Jordan Juravsky, Ryan Ehrlich, Ronald Clark, Quoc V. Le, Christopher Ré, and Azalia Mirhoseini. Large language monkeys: Scaling inference compute with repeated sampling. arXiv preprint arXiv:2407.21787, 2024.

Justin Yang Chae, Md Tanvirul Alam, and Nidhi Rastogi. Towards understanding self-play for LLM reasoning. arXiv preprint arXiv:2510.27072, 2025.

Lili Chen, Mihir Prabhudesai, Katerina Fragkiadaki, Hao Liu, and Deepak Pathak. Selfquestioning language models. arXiv preprint arXiv:2508.03682, 2025a.

Xiaoyin Chen, Jiarui Lu, Minsu Kim, Dinghuai Zhang, Jian Tang, Alexandre Piché, Nicolas Gontier, Yoshua Bengio, and Ehsan Kamalloo. Self-evolving curriculum for LLM reasoning. arXiv preprint arXiv:2505.14970, 2025b.

Karan Dalal, Daniel Koceja, Jiarui Xu, Yue Zhao, Shihao Han, Ka Chun Cheung, Jan Kautz, Yejin Choi, Yu Sun, and Xiaolong Wang. One-minute video generation with test-time training. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025.

Bodong Du, Xuanqi Huang, and Xiaomeng Li. Distribution-aware reward estimation for test-time reinforcement learning. arXiv preprint arXiv:2601.21804, 2026.

Dan Friedman and Adji Bousso Dieng. The Vendi score: A diversity evaluation metric for machine learning. Transactions on Machine Learning Research, 2023.

Wenhao Gao, Tianfan Fu, Jimeng Sun, and Connor W. Coley. Sample efficiency matters: A benchmark for practical molecular optimization. In Advances in Neural Information Processing Systems (NeurIPS), 2022.

Caglar Gulcehre, Tom Le Paine, Srivatsan Srinivasan, Ksenia Konyushkova, Lotte Weerts, Abhishek Sharma, Aditya Siddhant, Alex Ahern, Miaosen Wang, Chenjie Gu, Wolfgang Macherey, Arnaud Doucet, Orhan Firat, and Nando de Freitas. Reinforced self-training (ReST) for language modeling. arXiv preprint arXiv:2308.08998, 2023.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, Xiaokang Zhang, Xingkai Yu, Yu Wu, Z. F. Wu, Zhibin Gou, Zhihong Shao, Zhuoshu Li, Ziyi Gao, Aixin Liu, Bing Xue, Bingxuan Wang, Bochao Wu, Bei Feng, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chong Ruan, Damai Dai, Deli Chen, Dongjie Ji, Erhang Li, Fangyun Lin, Fucong Dai, Fuli Luo, Guangbo Hao, Guanting Chen, Guowei Li, H. Zhang, Hanwei Xu, Honghui Ding, Huazuo Gao, Hui Qu, Hui Li, Jianzhong Guo, Jiashi Li, Jingchang Chen, Jingyang Yuan, Jinhao Tu, Junjie Qiu, Junlong Li, J. L. Cai, Jiaqi Ni, Jian Liang, Jin Chen, Kai Dong, Kai Hu, Kaichao You, Kaige Gao, Kang Guan, Kexin Huang, Kuai Yu, Lean Wang, Lecong Zhang, Liang Zhao, Litong Wang, Liyue Zhang, Lei Xu, Leyi Xia, Mingchuan Zhang, Minghua Zhang, Minghui Tang, Mingxu Zhou, Meng Li, Miaojun Wang, Mingming Li, Ning Tian, Panpan Huang, Peng Zhang, Qiancheng Wang, Qinyu Chen, Qiushi Du, Ruiqi Ge, Ruisong Zhang, Ruizhe Pan, Runji Wang, R. J. Chen, R. L. Jin, Ruyi Chen, Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shengfeng Ye, Shiyu Wang, Shuiping Yu, Shunfeng Zhou, Shuting Pan, S. S. Li, Shuang Zhou, Shaoqing Wu, Tao Yun, Tian Pei, Tianyu Sun, T. Wang, Wangding Zeng, Wen Liu, Wenfeng Liang, Wenjun Gao, Wenqin Yu, Wentao Zhang, W. L. Xiao, Wei An, Xiaodong Liu, Xiaohan Wang, Xiaokang Chen, Xiaotao Nie, Xin Cheng, Xin Liu, Xin Xie, Xingchao Liu, Xinyu Yang, Xinyuan Li, Xuecheng Su, Xuheng Lin, X. Q. Li, Xiangyue Jin, Xiaojin Shen, Xiaosha Chen, Xiaowen Sun, Xiaoxiang Wang, Xinnan Song, Xinyi Zhou, Xianzu Wang, Xinxia Shan, Y. K. Li, Y. Q. Wang, Y. X. Wei, Yang Zhang, Yanhong Xu, Yao Li, Yao Zhao, Yaofeng Sun, Yaohui Wang, Yi Yu, Yichao Zhang, Yifan Shi, Yiliang Xiong, Ying He, Yishi Piao, Yisong Wang, Yixuan Tan, Yiyang Ma, Yiyuan Liu, Yongqiang Guo, Yuan Ou, Yuduan Wang, Yue Gong, Yuheng Zou, Yujia He, Yunfan Xiong, Yuxiang Luo, Yuxiang You, Yuxuan Liu, Yuyang Zhou, Y. X. Zhu, Yanping Huang, Yaohui Li, Yi Zheng, Yuchen Zhu, Yunxian Ma, Ying Tang, Yukun Zha, Yuting Yan, Z. Z. Ren, Zehui Ren, Zhangli Sha, Zhe Fu, Zhean Xu, Zhenda Xie, Zhengyan Zhang, Zhewen Hao, Zhicheng Ma, Zhigang Yan, Zhiyu Wu, Zihui Gu, Zijia Zhu, Zijun Liu, Zilin Li, Ziwei Xie, Ziyang Song, Zizheng Pan, Zhen Huang, Zhipeng Xu, Zhongyu Zhang, and Zhen Zhang. DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning. Nature, 2025. arXiv preprint arXiv:2501.12948.

Moritz Hardt and Yu Sun. Test-time training on nearest neighbors for large language models. In International Conference on Learning Representations (ICLR), 2024.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations (ICLR), 2022.

Chengsong Huang, Wenhao Yu, Xiaoyang Wang, Hongming Zhang, Zongxia Li, Ruosen Li, Jiaxin Huang, Haitao Mi, and Dong Yu. R-Zero: Self-evolving reasoning LLM from zero data. arXiv preprint arXiv:2508.05004, 2025.

Jonas Hübotter, Leander Diaz-Bone, Ido Hakimi, Andreas Krause, and Moritz Hardt. Learning on the job: Test-time curricula for targeted reinforcement learning. arXiv preprint arXiv:2510.04786, 2025.

Jonas Hübotter, Patrik Wolf, Alexander Shevchenko, Dennis Jüni, Andreas Krause, and Gil Kur. Specialization after generalization: Towards understanding test-time training in foundation models. In International Conference on Learning Representations (ICLR), 2026.

Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, Sida Wang, Armando Solar-Lezama, Koushik Sen, and Ion Stoica. LiveCodeBench: Holistic and contamination free evaluation of large language models for code. arXiv preprint arXiv:2403.07974, 2024.

Kimi Team, Angang Du, Bofei Gao, Bowei Xing, Changjiu Jiang, Cheng Chen, Cheng Li, Chenjun Xiao, Chenzhuang Du, Chonghua Liao, Chuning Tang, Congcong Wang, Dehao

Zhang, Enming Yuan, Enzhe Lu, Fengxiang Tang, Flood Sung, Guangda Wei, Guokun Lai, Haiqing Guo, Han Zhu, Hao Ding, Hao Hu, Hao Yang, Hao Zhang, Haotian Yao, Haotian Zhao, Haoyu Lu, Haoze Li, Haozhen Yu, Hongcheng Gao, Huabin Zheng, Huan Yuan, Jia Chen, Jianhang Guo, Jianlin Su, Jianzhou Wang, Jie Zhao, Jin Zhang, Jingyuan Liu, Junjie Yan, Junyan Wu, Lidong Shi, Ling Ye, Longhui Yu, Mengnan Dong, Neo Zhang, Ningchen Ma, Qiwei Pan, Qucheng Gong, Shaowei Liu, Shengling Ma, Shupeng Wei, Sihan Cao, Siying Huang, Tao Jiang, Weihao Gao, Weimin Xiong, Weiran He, Weixiao Huang, Weixin Xu, Wenhao Wu, Wenyang He, Xianghui Wei, Xianqing Jia, Xingzhe Wu, Xinran Xu, Xinxing Zu, Xinyu Zhou, Xuehai Pan, Y. Charles, Yang Li, Yangyang Hu, Yangyang Liu, Yanru Chen, Yejie Wang, Yibo Liu, Yidao Qin, Yifeng Liu, Ying Yang, Yiping Bao, Yulun Du, Yuxin Wu, Yuzhi Wang, Zaida Zhou, Zhaoji Wang, Zhaowei Li, Zhen Zhu, Zheng Zhang, Zhexu Wang, Zhilin Yang, Zhiqi Huang, Zihao Huang, Ziyao Xu, Zonghan Yang, and Zongyu Lin. Kimi k1.5: Scaling reinforcement learning with LLMs. arXiv preprint arXiv:2501.12599, 2025.

Jakub Grudzien Kuba, Mengting Gu, Qi Ma, Yuandong Tian, Vijai Mohan, and Jason Chen. Language self-play for data-free training. arXiv preprint arXiv:2509.07414, 2025.

Tim Launer, Jonas Hübotter, Marco Bagatella, Ido Hakimi, and Andreas Krause. Majority voting for code generation. arXiv preprint arXiv:2604.15618, 2026. ICLR 2026 Test-Time Updates (TTU) Workshop.

Nayoung Lee, Ziyang Cai, Avi Schwarzschild, Kangwook Lee, and Dimitris Papailiopoulos. Self-improving transformers overcome easy-to-hard and length generalization challenges. arXiv preprint arXiv:2502.01612, 2025.

Bo Liu, Simon Yu, Zichen Liu, Leon Guertler, Penghui Qi, Daniel Balcells, Mickel Liu, Cheston Tan, Weiyan Shi, Min Lin, Wee Sun Lee, and Natasha Jaques. SPIRAL: Selfplay on zero-sum games incentivizes reasoning via multi-agent multi-turn reinforcement learning. In International Conference on Learning Representations (ICLR), 2026.

Jürgen Schmidhuber. A possibility for implementing curiosity and boredom in modelbuilding neural controllers. Technical report, Technische Universität München, 1991.

Sheikh Shafayat, Fahim Tajwar, Ruslan Salakhutdinov, Jeff Schneider, and Andrea Zanette. Can large reasoning models self-train? arXiv preprint arXiv:2505.21444, 2025.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. DeepSeekMath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

David Silver and Richard S. Sutton. Welcome to the era of experience. Designing an Intelligence, 2025. To appear, MIT Press.

David Silver, Aja Huang, Chris J. Maddison, Arthur Guez, Laurent Sifre, George van den Driessche, Julian Schrittwieser, Ioannis Antonoglou, Veda Panneershelvam, Marc Lanctot, Sander Dieleman, Dominik Grewe, John Nham, Nal Kalchbrenner, Ilya Sutskever, Timothy Lillicrap, Madeleine Leach, Koray Kavukcuoglu, Thore Graepel, and Demis Hassabis. Mastering the game of Go with deep neural networks and tree search. Nature, 529:484–489, 2016.

David Silver, Thomas Hubert, Julian Schrittwieser, Ioannis Antonoglou, Matthew Lai, Arthur Guez, Marc Lanctot, Laurent Sifre, Dharshan Kumaran, Thore Graepel, Timothy Lillicrap, Karen Simonyan, and Demis Hassabis. Mastering chess and shogi by self-play with a general reinforcement learning algorithm. arXiv preprint arXiv:1712.01815, 2017.

David Silver, Thomas Hubert, Julian Schrittwieser, Ioannis Antonoglou, Matthew Lai, Arthur Guez, Marc Lanctot, Laurent Sifre, Dharshan Kumaran, Thore Graepel, Timothy Lillicrap, Karen Simonyan, and Demis Hassabis. A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play. Science, 362(6419):1140– 1144, 2018.

Toby Simonds and Akira Yoshiyama. LADDER: Self-improving LLMs through recursive problem decomposition. arXiv preprint arXiv:2503.00735, 2025.

Charlie Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar. Scaling LLM test-time compute optimally can be more effective than scaling model parameters. arXiv preprint arXiv:2408.03314, 2024.

Yu Sun, Xiaolong Wang, Zhuang Liu, John Miller, Alexei A. Efros, and Moritz Hardt. Test-time training with self-supervision for generalization under distribution shifts. In International Conference on Machine Learning (ICML), 2020.

Yu Sun, Xinhao Li, Karan Dalal, Jiarui Xu, Arjun Vikram, Genghan Zhang, Yann Dubois, Xinlei Chen, Xiaolong Wang, Sanmi Koyejo, Tatsunori Hashimoto, and Carlos Guestrin. Learning to (learn at test time): RNNs with expressive hidden states. In International Conference on Machine Learning (ICML), 2025.

Zhiqing Sun, Longhui Yu, Yikang Shen, Weiyang Liu, Yiming Yang, Sean Welleck, and Chuang Gan. Easy-to-hard generalization: Scalable alignment beyond human supervision. In Advances in Neural Information Processing Systems (NeurIPS), 2024.

Team OLMo, Pete Walsh, Luca Soldaini, Dirk Groeneveld, Kyle Lo, Shane Arora, Akshita Bhagia, Yuling Gu, Shengyi Huang, Matt Jordan, Nathan Lambert, Dustin Schwenk, Oyvind Tafjord, Taira Anderson, David Atkinson, Faeze Brahman, Christopher Clark, Pradeep Dasigi, Nouha Dziri, Allyson Ettinger, Michal Guerquin, David Heineman, Hamish Ivison, Pang Wei Koh, Jiacheng Liu, Saumya Malik, William Merrill, Lester James V. Miranda, Jacob Morrison, Tyler Murray, Crystal Nam, Jake Poznanski, Valentina Pyatkin, Aman Rangapur, Michael Schmitz, Sam Skjonsberg, David Wadden, Christopher Wilhelm, Michael Wilson, Luke Zettlemoyer, Ali Farhadi, Noah A. Smith, and Hannaneh Hajishirzi. 2 OLMo 2 Furious. arXiv preprint arXiv:2501.00656, 2025.

Gerald Tesauro. Temporal difference learning and TD-Gammon. Communications of the ACM, 38(3):58–68, 1995.

George Tsoukalas, Jasper Lee, John Jennings, Jimmy Xin, Michelle Ding, Michael Jennings, Amitayush Thakur, and Swarat Chaudhuri. PutnamBench: Evaluating neural theoremprovers on the Putnam mathematical competition. In Advances in Neural Information Processing Systems (NeurIPS) Datasets and Benchmarks Track, 2024.

Sang Michael Xie, Hieu Pham, Xuanyi Dong, Nan Du, Hanxiao Liu, Yifeng Lu, Percy Liang, Quoc V. Le, Tengyu Ma, and Adams Wei Yu. DoReMi: Optimizing data mixtures speeds up language model pretraining. In Advances in Neural Information Processing Systems (NeurIPS), 2023.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jing Zhou, Jingren Zhou, Junyang Lin, Kai Dang, Keqin Bao, Kexin Yang, Le Yu, Lianghao Deng, Mei Li, Mingfeng Xue, Mingze Li, Pei Zhang, Peng Wang, Qin Zhu, Rui Men, Ruize Gao, Shixuan Liu, Shuang Luo, Tianhao Li, Tianyi Tang, Wenbiao Yin, Xingzhang Ren, Xinyu Wang, Xinyu Zhang, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yinger Zhang, Yu Wan, Yuqiong Liu, Zekun Wang, Zeyu Cui, Zhenru Zhang, Zhipeng Zhou, and Zihan Qiu. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, Xin Liu, Haibin Lin, Zhiqi Lin, Bole Ma, Guangming Sheng, Yuxuan Tong, Chi Zhang, Mofan Zhang, Wang Zhang, Hang Zhu, Jinhua Zhu, Jiaze Chen, Jiangjie Chen, Chengyi Wang, Hongli Yu, Yuxuan Song, Xiangpeng Wei, Hao Zhou, Jingjing Liu, Wei-Ying Ma, Ya-Qin Zhang, Lin Yan, Mu Qiao, Yonghui Wu, and Mingxuan Wang. DAPO: An open-source LLM reinforcement learning system at scale. arXiv preprint arXiv:2503.14476, 2025.

Eric Zelikman, Yuhuai Wu, Jesse Mu, and Noah D. Goodman. STaR: Bootstrapping reasoning with reasoning. In Advances in Neural Information Processing Systems (NeurIPS), 2022.

Qingyang Zhang, Haitao Wu, Changqing Zhang, Peilin Zhao, and Yatao Bian. Right question is already half the answer: Fully unsupervised LLM reasoning incentivization. arXiv preprint arXiv:2504.05812, 2025.

Qizheng Zhang, Changran Hu, Shubhangi Upasani, Boyuan Ma, Fenglu Hong, Vamsidhar Kamanuru, Jay Rainton, Chen Wu, Mengmeng Ji, Hanchen Li, Urmish Thakker, James Zou, and Kunle Olukotun. Agentic context engineering: Evolving contexts for selfimproving language models. In International Conference on Learning Representations (ICLR), 2026a.

Zizhuo Zhang, Jianing Zhu, Xinmu Ge, Zihua Zhao, Zhanke Zhou, Xuan Li, Xiao Feng, Jiangchao Yao, and Bo Han. Co-rewarding: Stable self-supervised RL for eliciting reasoning in large language models. In International Conference on Learning Representations (ICLR), 2026b.

Andrew Zhao, Yiran Wu, Yang Yue, Tong Wu, Quentin Xu, Yang Yue, Matthieu Lin, Shenzhi Wang, Qingyun Wu, Zilong Zheng, and Gao Huang. Absolute zero: Reinforced self-play reasoning with zero data. arXiv preprint arXiv:2505.03335, 2025.

Zihan Zheng, Zerui Cheng, Zeyu Shen, Shang Zhou, Kaiyuan Liu, Hansen He, Dongruixuan Li, Stanley Wei, Hangyi Hao, Jianzhu Yao, Peiyao Sheng, Zixuan Wang, Wenhao Chai, Aleksandra Korolova, Peter Henderson, Sanjeev Arora, Pramod Viswanath, Jingbo Shang, and Saining Xie. LiveCodeBench Pro: How do olympiad medalists judge LLMs in competitive programming? arXiv preprint arXiv:2506.11928, 2025.

Yuxin Zuo, Kaiyan Zhang, Li Sheng, Shang Qu, Ganqu Cui, Xuekai Zhu, Haozhan Li, Yuchen Zhang, Xinwei Long, Ermo Hua, Biqing Qi, Youbang Sun, Zhiyuan Ma, Lifan Yuan, Ning Ding, and Bowen Zhou. TTRL: Test-time reinforcement learning. arXiv preprint arXiv:2504.16084, 2025.

## A Target Problems and Evaluation Protocol

This appendix fixes the evaluation substrate used throughout Section 5. We first list the target problem subsets, then define the verifier-strength axis and the common filtered evaluation metric used to compare all methods.

## A.1 Target problem subsets

Section 5.1 defines the evaluation set by filtering a recent slice of LiveCodeBench v6 to problems where Qwen 3 8B has non-zero but imperfect base coverage (Jain et al., 2024; Yang et al., 2025). This appendix lists the exact target problem identifiers and base model mean@512 values. The split is used throughout the paper: MidHardLCBv6 is the consistencyregime subset where TTSP can exploit non-trivial base coverage, and VeryHardLCBv6 is the near-discovery subset where base coverage is too small for the synthetic curriculum to compose reliably back to $x ^ { \star }$

Table 3: MidHardLCBv6 target problems. LiveCodeBench v6 target problem identifiers and base model mean@512 of Qwen 3 8B for each selected target problem in the MidHardLCBv6 subset (0.1 < mean@512 < 0.7).
<table><tr><td>Target Problem</td><td>mean@512</td><td>Correct Solution Rank</td></tr><tr><td>1035</td><td>0.686</td><td>1</td></tr><tr><td>1050</td><td>0.645</td><td>1</td></tr><tr><td>976</td><td>0.633</td><td>1</td></tr><tr><td>920</td><td>0.596</td><td>1</td></tr><tr><td>1047</td><td>0.555</td><td>1</td></tr><tr><td>962</td><td>0.523</td><td>1</td></tr><tr><td>916</td><td>0.500</td><td>1</td></tr><tr><td>958</td><td>0.496</td><td>1</td></tr><tr><td>948</td><td>0.488</td><td>1</td></tr><tr><td>918</td><td>0.426</td><td>1</td></tr><tr><td>926</td><td>0.297</td><td>2</td></tr><tr><td>934</td><td>0.250</td><td>2</td></tr><tr><td>933</td><td>0.219</td><td>2</td></tr><tr><td>921</td><td>0.201</td><td>2</td></tr><tr><td>980</td><td>0.188</td><td>2</td></tr><tr><td>957</td><td>0.186</td><td>2</td></tr><tr><td>987</td><td>0.135</td><td>3</td></tr><tr><td>909</td><td>0.125</td><td>2</td></tr><tr><td>960</td><td>0.111</td><td>4</td></tr></table>

Table 4: VeryHardLCBv6 target problems. LiveCodeBench v6 target problem identifiers and base model mean@512 of Qwen 3 8B for each selected target problem in the VeryHardLCBv6 subset $( 0 < \mathrm { m e a n } @ 5 1 2 \leq \bar { 0 } . 1 )$
<table><tr><td>Target Problem</td><td>mean@512</td><td>Correct Solution Rank</td></tr><tr><td>941</td><td>0.057</td><td>3</td></tr><tr><td>1017</td><td>0.045</td><td>2</td></tr><tr><td>946</td><td>0.043</td><td>3</td></tr><tr><td>919</td><td>0.018</td><td>3</td></tr><tr><td>1027</td><td>0.012</td><td>2</td></tr><tr><td>947</td><td>0.010</td><td>11</td></tr><tr><td>905</td><td>0.010</td><td>4</td></tr><tr><td>979</td><td>0.004</td><td>37</td></tr><tr><td>963</td><td>0.002</td><td>21</td></tr></table>

The remaining subsections make the verifier-strength axis explicit. The distinction is important for interpreting Figure 1: TTSP receives training rewards from execution on proposergenerated synthetic problems, whereas the baselines select, cluster, or optimize directly against information from the target problem verifier.

## A.2 Verifier-strength axis

A verifier is a selection rule applied to candidate solutions for the original target problem $x ^ { \star }$ Strong verifiers reject a large fraction of incorrect candidate programs, while weak verifiers admit many false positives. We evaluate across a sweep of verifier strengths to factor the test-time scaling problem into candidate generation and candidate selection, following the main-text motivation that test-time compute alone is not enough when the selection signal is weak (Snell et al., 2024; Brown et al., 2024).

The weakest condition is $v _ { 0 } .$ , defined by $v _ { 0 } ( y ) = 1$ for every rollout y. It corresponds to no filtering: all generated candidates are treated as admissible, and the method is evaluated solely by how often its unfiltered candidates pass the oracle $v ^ { \star }$ . The calibrated artificial verifiers $v _ { 2 5 } , v _ { 5 0 } ,$ , and $v _ { 7 5 }$ are constructed to reject approximately the indicated fraction of incorrect calibration rollouts. The natural LiveCodeBench verifiers $v _ { \mathrm { p u b } } , v _ { \mathrm { p r i v } } ,$ and $v ^ { \star }$ are built from public tests, private tests, and their union, respectively (Jain et al., 2024). The public+private verifier $\hat { v ^ { \star } }$ is both the strongest evaluation-time filter and the oracle used to measure final correctness.

## A.3 Natural LiveCodeBench verifiers

LiveCodeBench provides public and private test cases for each target problem (Jain et al., 2024). We use these test suites as natural verifier-access levels. The public verifier $v _ { \mathrm { p u b } }$ models the realistic programming-contest or live-coding setting in which only sample or public tests are available at deployment. The private verifier $v _ { \mathrm { p r i v } }$ and the public+private verifier $v ^ { \star }$ provide stronger selection rules and serve as high-verifier-access reference points.

All reported correctness scores are computed against $v ^ { \star }$ . Thus, even when the evaluationtime filter is weaker than $v ^ { \star }$ , the final measurement asks whether an admitted candidate passes the full public+private evaluator. This convention keeps the target quantity fixed across verifier-strength conditions and makes the curves in Figure 1 comparable.

## A.4 Artificial verifier calibration

The calibrated weak verifiers $v _ { 2 5 } , v _ { 5 0 } ,$ , and $v _ { 7 5 }$ are selected from a candidate pool containing retained generated tests as well as public and private LiveCodeBench tests. We call these verifiers artificial because the subset is chosen to match a target empirical rejection rate, not because every selected test is generated. Generated candidate inputs are produced using Qwen 3 8B (Yang et al., 2025) with the same refinement-style prompting pattern used by TTSP to reconstruct synthetic problem tests; the TTSP refinement prompt that this pattern mirrors is reported in Appendix F.4. These tests are not used to train TTSP. They are used only to construct controlled evaluation-time filters whose false-positive rate can be varied more finely than the natural public/private split.

Generated inputs do not come with trusted ground-truth outputs, so we label them by consensus among base model programs that are already correct under $v ^ { \star }$ . Public and private LiveCodeBench tests use their provided labels. For each target problem, we sample 512 Qwen 3 8B baseline rollouts and partition them into correct and incorrect programs using $v ^ { \star }$ . For a generated input z, we execute the correct baseline programs on z. If the correct programs disagree, z is discarded; if they agree, the consensus output is used as the reference label. This step is designed to reduce false negatives: a generated test is kept only when programs known to solve the target problem agree on its output.

We then reduce verifier construction to a coverage problem over the incorrect calibration rollouts. Each retained candidate test case $t _ { j }$ has an associated rejection set

$$
E _ { j } \subseteq \mathcal { Y } _ { \mathrm { i n c o r r e c t } } ,
$$

where $E _ { j }$ contains the incorrect calibration rollouts that either fail to execute on $t _ { j }$ or produce an output different from the consensus output. For a candidate verifier defined by a subset of tests S, its empirical strength is

$$
c ( S ) = { \frac { \left| \bigcup _ { t _ { j } \in S } E _ { j } \right| } { \left| \mathcal { V } _ { \mathrm { i n c o r r e c t } } \right| } } .
$$

This definition accounts for overlap between tests: two tests that reject the same incorrect rollouts do not increase verifier strength additively.

For each target problem, $v _ { 2 5 } , v _ { 5 0 } ,$ and $v _ { 7 5 }$ are selected as subsets of the retained candidate tests whose empirical coverage c(S) is closest to 0.25, 0.50, and 0.75, respectively. The resulting filters intentionally permit false positives, because a weak verifier should accept some incorrect programs. We keep $v _ { \mathsf { p u b } } , v _ { \mathsf { p r i v } } ,$ and $v ^ { \star }$ as separate natural verifier conditions rather than folding them into this calibrated artificial sweep.

## A.5 Filtered evaluation metric

Section 5.2 defines the filtered mean@8 metric. For iterative methods and training-curve diagnostics, the same metric is applied checkpoint-wise to the eight target problem submissions sampled at checkpoint t.

If the evaluation-time verifier admits no candidates for a given target problem, method, checkpoint, and verifier condition, the reported score is carried forward from the most recent earlier checkpoint with a non-empty admitted set for the same target problem, method, and verifier condition. If no earlier checkpoint has a non-empty admitted set, the score is defined to be 0. For non-iterative baselines, this rule reduces to assigning score 0 whenever the admitted set is empty. This convention avoids undefined scores when a strong verifier rejects all sampled candidates.

Table 5 reports how many tests are used by each evaluation-time verifier on HardLCBv6, averaged over the 28 target problems and grouped by test source.

## A.6 Uncertainty bands

For Figures 4 and $^ { 3 , }$ shaded bands denote 90% nonparametric bootstrap confidence intervals over target problems. For each curve point, we resample the HardLCBv6 target problems with replacement 10,000 times, recompute the mean score across the resampled target set, and plot the 5th and 95th percentiles of the resulting bootstrap distribution.

For the diagnostic curves in Figures 10, 12, and the right panel of Figure 13, shaded bands denote 1 standard error of the mean across target problems, computed as the empirical standard deviation of target-level values divided by ${ \sqrt { n } } .$

Table 5: Tests used by each evaluation-time verifier on HardLCBv6. Values are averages over the 28 target problems and are grouped by test source.
<table><tr><td rowspan="3">Verifier</td><td colspan="4">Mean tests used</td></tr><tr><td>Generated</td><td>Public</td><td>Private</td><td>Total</td></tr><tr><td>v0</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td> $v _ { 2 5 }$ </td><td>0.36</td><td>0.07</td><td>0.86</td><td>1.29</td></tr><tr><td>v50</td><td>0.39</td><td>0.07</td><td>1.00</td><td>1.46</td></tr><tr><td>v75</td><td>0.57</td><td>0.18</td><td>0.79</td><td>1.54</td></tr><tr><td> $v _ { \mathrm { p u b } }$ </td><td>0.00</td><td>2.86</td><td>0.00</td><td>2.86</td></tr><tr><td> $v _ { \mathrm { p r i v } }$ </td><td>0.00</td><td>0.00</td><td>37.50</td><td>37.50</td></tr><tr><td> $v ^ { \star }$ </td><td>0.00</td><td>2.86</td><td>37.50</td><td>40.36</td></tr></table>

## B TTSP Training Details and Hyperparameters

Section 4 describes TTSP at the algorithmic level: a proposer policy generates targetconditioned synthetic problems, a solver policy trains on them, and both policies receive execution-based rewards. This appendix records the implementation choices needed to reproduce the experiments. The design follows the main-text constraint that adaptation is performed separately for each target problem and that no target problem verifier is used during TTSP training.

Both roles are implemented as separate LoRA adapters over a frozen Qwen 3 8B backbone (Hu et al., 2022; Yang et al., 2025). The separation mirrors the asymmetric self-play setup in Algorithm 1: the proposer optimizes for a useful curriculum, while the solver optimizes for pass/fail performance on generated synthetic problems. Both adapters are optimized with GRPO (Shao et al., 2024). For each prompt, the active policy samples a group of responses, assigns scalar rewards, computes group-normalized advantages, and performs a clipped on-policy update. We use asymmetric clipping with $( \epsilon _ { \mathrm { l o w } } , \epsilon _ { \mathrm { h i g h } } ) = ( 0 . 2 0 , 0 . 2 8 )$ following DAPO (Yu et al., 2025). We set the KL coefficient to zero following the test-time adaptation setup of Hübotter et al. (2025), so the update is driven by target-specific rewards rather than by explicit regularization toward the initialization.

Each solver step uses a batch of 48 synthetic problem slots: 16 fresh problems generated by the current proposer and 32 replayed problems sampled from the synthetic problem history. For each synthetic problem, the solver samples a group of 8 rollouts and receives binary execution rewards on held-out generated tests. Replay eligibility is determined by the most recent recorded solver solve rate for each historical synthetic problem. When a replayed problem is sampled, stale rewards are not reused: the current solver policy generates fresh rollouts, pass/fail rewards are recomputed, and the stored solve rate used for future proposer context and replay eligibility is overwritten. The synthetic problem history has no fixed horizon; accepted synthetic problems remain eligible for replay whenever their most recent recorded solve rate lies in the learnability window.

## B.1 Hyperparameters

Tables 6 and 7 report the full TTSP training configuration. Unless otherwise stated, values are fixed across target problems. In particular, aggregate method-comparison figures use converged checkpoints rather than per-target hidden-test checkpoint selection, matching the evaluation protocol in Section 5.2.

Table 6: Core TTSP training hyperparameters. All values are fixed across target problems. The model is adapted per target through separate proposer and solver LoRA adapters, while the Qwen 3 8B backbone remains frozen.
<table><tr><td>Setting</td><td>Value</td></tr><tr><td colspan="2">Model</td></tr><tr><td>Backbone</td><td>Qwen3-8B</td></tr><tr><td>Trainable parameters</td><td>Separate proposer and solver LoRA adapters</td></tr><tr><td>LoRA rank / α</td><td>32/32</td></tr><tr><td>LoRA target modules</td><td>All linear layers</td></tr><tr><td colspan="2">Optimization</td></tr><tr><td>RL objective</td><td>GRPO</td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>Learning rate</td><td> $1 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Warmup</td><td>8 steps</td></tr><tr><td>Clipping</td><td> $( \epsilon _ { \mathrm { l o w } } , \epsilon _ { \mathrm { h i g h } } ) = ( 0 . 2 0 , 0 . 2 8 )$  0</td></tr><tr><td>KL coefficient</td><td></td></tr><tr><td>Gradient clip norm</td><td>1.0</td></tr><tr><td colspan="2">Training budget</td></tr><tr><td>Training length</td><td>50 GRPO steps per target problem</td></tr><tr><td>Checkpoint selection</td><td>Converged checkpoint; no best hidden-test check- point selection</td></tr></table>

Table 7: Curriculum, rollout, and replay hyperparameters. Fresh synthetic problems are generated by the proposer; replayed synthetic problems are sampled from history and re-evaluated under the current solver policy when replayed.
<table><tr><td>Setting</td><td>Value</td></tr><tr><td>Generation</td><td></td></tr><tr><td>Inference engine</td><td>vLLM</td></tr><tr><td>Sampling temperature</td><td>1.0</td></tr><tr><td>Max prompt / response length</td><td>8096/16384 tokens</td></tr><tr><td>Proposer contexts per step</td><td>2</td></tr><tr><td>Proposer rollouts per context</td><td>8</td></tr><tr><td>Fresh synthetic problem slots per solver step</td><td>16</td></tr><tr><td>Solver batch</td><td></td></tr><tr><td>Synthetic problem batch size</td><td>48 per solver step</td></tr><tr><td>Batch composition</td><td>16 fresh + 32 replayed synthetic problems</td></tr><tr><td>Solver rollouts per synthetic problem</td><td>8</td></tr><tr><td>Solver reward</td><td>Binary pass/fail on 10 held-out generated tests</td></tr><tr><td>Proposer reward and replay</td><td></td></tr><tr><td>Proposer reward</td><td>Equation 1</td></tr><tr><td>Invalid-generation penalty</td><td>-0.5</td></tr><tr><td>Replay eligibility</td><td>Most recent recorded solver solve rate in [0.1, 0.8]</td></tr><tr><td>History size</td><td>No fixed cap</td></tr><tr><td>Replay refresh</td><td>Fresh solver rollouts, rewards, and recorded solve rate recomputed on replay</td></tr><tr><td>Context and length limits</td><td></td></tr><tr><td>Proposer context window</td><td>Previous 5 steps</td></tr><tr><td>Synthetic problems shown in context</td><td>3</td></tr><tr><td>Synthetic problem max length</td><td>1024 tokens</td></tr><tr><td>Reference solution max length</td><td>4096 tokens</td></tr></table>

## C Baseline Implementation and Diagnostics

This appendix records baseline-specific details that are separate from the shared filtered evaluation protocol in Appendix A.

## C.1 ACE Baseline Implementation

We include ACE (Zhang et al., 2026a) as a test-time context-optimization baseline. In contrast to TTSP, ACE does not update model weights. Instead, it maintains a problem-specific textual playbook, revises that playbook from execution feedback, and conditions later candidate generations on the revised playbook. ACE therefore tests whether gains can be obtained from context-only adaptation, rather than from TTSP’s weight-updating policy adaptation.

For a controlled comparison, all ACE roles use the same base model as our TTSP experiments, Qwen3-8B. Each ACE iteration consists of several generation–reflection rounds. In each round, the Generator proposes a Python solution using the current playbook, the candidate is executed against the feedback tests available under the current verifier condition, and the Reflector turns the resulting execution signal into a concrete diagnosis of the attempt. After the rounds in an iteration are complete, the Curator converts the accumulated reflections into a structured update to the playbook, which is then used in the next iteration.

We evaluate ACE under the same feedback regimes as the other test-time baselines: public tests, private tests, public + private tests, and calibrated artificial verifier tests. The feedback exposed to ACE is restricted to the tests available in the corresponding condition. The full public + private evaluator $v ^ { \star }$ is used only for held-out monitoring after each round and iteration, and is not revealed to the playbook except in the public + private oracle condition.

ACE is evaluated in online mode with one evolving playbook per target problem. We use 15 ACE iterations, at most 5 generation–reflection rounds per iteration, deterministic decoding during playbook evolution, a maximum generation length of 16,384 tokens, and a maximum model context length of 40,960 tokens.

## C.2 Functional Majority Voting Baseline

We implement Functional Majority Voting (FMV) following Launer et al. (2026). The FMV selection rule is input-only: it uses target problem inputs to compare candidate programs by functional behavior, but does not require target problem outputs.

Let $\mathcal { C } = \{ c _ { 1 } , \ldots , c _ { N } \}$ be candidate programs and let $\mathcal { X } = \{ x _ { 1 } , \ldots , x _ { K } \}$ be the target problem input cases used for functional comparison. Executing candidate $c _ { i }$ on input $x _ { k }$ produces output string $o _ { i , k }$ . We discard candidates that fail to execute on the comparison inputs and denote the remaining valid candidates by $\mathcal { C } _ { \mathrm { v a l } }$

For each valid candidate $c _ { i } \in \mathcal { C } _ { \mathrm { v a l } }$ , we compute the FMV score

$$
S ( c _ { i } ) = \sum _ { \stackrel { \scriptstyle { c _ { j } \in { \mathcal { C } } _ { \mathrm { v a l } } } } { j \neq i } } \sum _ { k = 1 } ^ { K } \mathbf { 1 } \{ o _ { i , k } = o _ { j , k } \} .
$$

This score counts pairwise functional agreement between candidate $c _ { i }$ and the other valid candidates over the comparison inputs. The original FMV selection rule returns a candidate in

$$
\arg \operatorname* { m a x } _ { c _ { i } \in \mathcal { C } _ { \mathrm { v a l } } } S ( c _ { i } ) .
$$

In our implementation, when multiple candidates attain the maximum score, we keep all tied maximizers:

$$
{ \mathcal { C } } _ { \mathrm { F M V } } = \left\{ c _ { i } \in { \mathcal { C } } _ { \mathrm { v a l } } : S ( c _ { i } ) = \operatorname* { m a x } _ { c _ { j } \in { \mathcal { C } } _ { \mathrm { v a l } } } S ( c _ { j } ) \right\} .
$$

For the verifier-strength comparison in Figure 1, we wrap this input-only FMV rule with the current verifier condition. Given verifier v, we first form the verifier-admitted candidate pool

$$
{ \mathcal { C } } _ { v } = \{ c \in { \mathcal { C } } : v ( c ) = 1 \} .
$$

We then apply the FMV rule above to ${ \mathcal { C } } _ { v } ,$ using the available target problem input cases for functional comparison. This wrapper uses the same verifier condition as the rest of the evaluation sweep, while Table 2 reports the access required by the FMV voting rule itself. The selected candidates are then passed to the common filtered evaluation protocol in Appendix A.5; when the same verifier has already been used to form $\mathcal { C } _ { v , }$ , this final pass is idempotent except for the shared empty-set handling convention.

## C.3 Functional Majority Voting Under Partial Test-Input Access

Section 5.4 compares TTSP, FMV, ACE, and rejection sampling under the same evaluationtime filtering protocol. We also report a diagnostic variant that removes the final post-hoc filter and varies only the target problem test inputs available to FMV for functional clustering. This separates FMV’s input-access benefit from rejection-sampling effects.

In this diagnostic, every method is scored directly against $v ^ { \star }$ . The access condition determines which test inputs FMV can use to form functional clusters, rather than which candidates are filtered after generation. TTSP does not use target problem test inputs during training, so its unfiltered score is independent of this access condition. FMV can improve when additional inputs separate incorrect functional clusters from the correct one, but it still cannot update the underlying candidate distribution.

![](images/8be8fae20a39b6f19edbb799bdd14897f5921914a5a8c1dcd051f6878e0912f7.jpg)  
Figure 8: FMV under partial test-input access on HardLCBv6. The final evaluation-time filter is removed, and all submitted candidates are scored directly against the public+private evaluator $v ^ { \star }$ . The x-axis denotes the test inputs available to FMV for functional clustering, not a post-hoc verifier. TTSP and Qwen 3 8B are shown as unfiltered baselines independent of FMV’s input-access condition.

## D Synthetic Curriculum Quality and Dynamics

Self-generated training introduces two concrete stability risks. First, the proposer can generate invalid or uninformative synthetic problems. Second, the solver can over-specialize to a narrow generated distribution. This section reports diagnostics for those risks during TTSP adaptation on HardLCBv6. These diagnostics are logged only for analysis; they are not used as rewards, filters, or checkpoint-selection criteria.

Operational criteria. A synthetic problem is used for solver training only after it passes the validity filter in Section 4. Among accepted problems, the proposer reward in Equation 1 favors intermediate solver success, and the replay buffer samples problems whose most recent solver solve rate lies in the eligibility window in Appendix B. Thus the relevant checks are operational: rejected generations should be excluded before solver updates, proposer rewards should retain a non-degenerate high-reward tail, and the generated curriculum should not collapse to a single policy or program family. These checks address the stability concern noted in Section 6; similar concerns arise in recent data-free and self-play LLM training methods (Zhao et al., 2025; Huang et al., 2025; Chae et al., 2025; Bailey et al., 2026).

## D.1 Synthetic problem rejection diagnostics

Before a generated synthetic problem enters solver training, it must pass the validity filter described in Section 4. Figure 9 reports pre-solver-training rejections, averaged over target problems and decomposed by implementation-level failure type. Rejected generations are excluded from the solver batch and produce no solver rollout rewards; instead, they receive the fixed invalid-generation proposer penalty.

The failure types in Figure 9 refine the higher-level filter definition in Section 4. Invalid generated output and missing tags correspond to parsing failures; length-limit, subgoal-too-long, and python-too-long failures correspond to structural constraints; timeouts and runtime errors correspond to execution failures; and insufficient, duplicate, or all-identical test cases correspond to held-out-test degeneracy.

The total number of rejected generations decreases over training. Invalid generated output is the largest rejection category, followed by timeouts and insufficient test cases. This diagnostic shows that TTSP trains on a filtered subset of proposer generations rather than treating all generated problems as valid training data.

![](images/18b431c393d8eeac7e9367e5c945334e4085704f1d0e93f07f65dc7f831d6de8.jpg)  
Figure 9: Synthetic problem rejection diagnostics. Pre-training proposer generations rejected before entering solver training, averaged over target problems and grouped by failure type. Rejection counts decrease over training; invalid generated output is the largest rejection category.

## D.2 Average and top-k proposer reward

The proposer update is based on sampled batches, so the batch mean alone can hide whether a step contains useful high-reward candidates. We therefore log both the mean proposer reward and the mean reward of the top four proposer samples at each step. A persistent gap between these curves indicates within-batch reward concentration: most samples may have low reward while a smaller subset still provides a non-degenerate proposer signal.

Figure 10 shows this pattern. The mean proposer reward remains low, while the top-4 reward is substantially higher. The solver reward also increases over training. Target validation denotes the solver’s unfiltered mean@8 on the target problem under $v _ { 0 } .$ . We use this plot as a non-degeneracy check for the synthetic reward signal; it does not imply that every generated synthetic problem is useful.

![](images/02eedffd45bfa829a4c8f4effb99b0624fff4820dbe747990897958e2d5b3247.jpg)  
Figure 10: Average and top-4 proposer reward during TTSP training. The Proposer curve is the mean proposer reward over generated synthetic problems, while Proposer (top-4) averages the four highest proposer rewards at each step. The gap measures within-batch reward concentration. Solver reward and unfiltered target validation are shown for reference. Shaded bands denote 1 SEM across target problems.

## D.3 Frozen-base and adapted-solver performance

To separate curriculum difficulty from solver adaptation, we evaluate both the frozen initialization and the current TTSP solver on the same synthetic problems generated at each step. The frozen model is Qwen 3 8B with no target-specific updates. Because the synthetic problem distribution changes over training, this is not a fixed-benchmark learning curve. Instead, it compares the adapted solver against the initialization on the current generated curriculum.

Figure 11 shows a representative trajectory. On the same generated synthetic problems, the adapted solver rises above the frozen initialization. This establishes an adaptation gain relative to the initial model on the generated curriculum. Target validation is plotted separately to show when this gain transfers back to the original target problem $x ^ { \star }$

![](images/17be0af8371f47c2a81b02704e95cb761b37e404eef2b642bb5a043aebcea6d0.jpg)  
Figure 11: Frozen-base and adapted-solver performance on generated synthetic problems. Example trajectory for one target problem. Frozen Qwen 3 8B is the initial model evaluated on the synthetic problems generated at each step; TTSP solver is the adapted solver evaluated on the same synthetic problems; Target validation is the adapted solver’s unfiltered mean@8 on the original target problem under $v _ { 0 }$

## D.4 Policy entropy

Figure 12 reports proposer and solver policy entropy across GRPO steps on HardLCBv6. Entropy is a distributional-collapse diagnostic. A sharp decrease in proposer entropy would indicate concentration of the synthetic problem generator, and a sharp decrease in solver entropy would indicate narrowing of the solver response distribution.

In our runs, proposer entropy increases gradually and solver entropy remains stable over the adaptation horizon. Entropy does not measure semantic novelty directly; the embedding diagnostics in Appendix D.5 audit program-level novelty separately.

![](images/86f8fa39a4126f0bb57b4cd871c32e8385fa984a7d919868131850f8709cefe4.jpg)  
Figure 12: Policy entropy during TTSP training. Mean proposer-policy and solver-policy entropy across GRPO steps on HardLCBv6, averaged over target problems. The proposer entropy increases over training, while solver entropy remains stable. Shaded regions show variation across target problems. Shaded bands denote 1 SEM across target problems.

## D.5 Embedding-based novelty and target relatedness

We audit accepted synthetic problems with two embedding-based diagnostics. First, for each valid synthetic problem, we embed its extracted Python reference program using text-embedding-3-large. We compute a Vendi score (Friedman & Dieng, 2023) on the resulting dot-product kernel and log the cumulative normalized Vendi gain when adding each new batch of generated programs to the previous curriculum. This measures programlevel embedding diversity; it is not used as a reward.

Second, we embed the generated synthetic problem statement and compare it to the original target problem statement using cosine similarity. This measures text-level target relatedness and is separate from program-level novelty.

Figure 13 shows two trends. Cumulative Vendi gain increases over training, which is inconsistent with repeatedly adding only near-identical embedded reference programs. Text-level similarity to the target peaks early and then decreases. These diagnostics do not prove semantic equivalence to the target problem; they only audit diversity and textual relatedness of the generated curriculum.

![](images/41ec196956390162b6af0a6a40a88c9f4e1b2f8795447e178d6bf37e16232932.jpg)  
(a) Cumulative normalized Vendi novelty over extracted Python programs.

![](images/5afbce739cc2ccf61b1fa5104ecb3c8973d630cde71e5fa4541e9b100497ab07.jpg)  
(b) Cosine similarity between generated synthetic problem text and the target problem text.

Figure 13: Curriculum novelty and target relatedness. Left: cumulative normalized Vendi gain computed from embeddings of extracted Python reference programs. Right: text-level cosine similarity between generated synthetic problem statements and the original target problem statement. Shaded bands denote 1 SEM across target problems.

## E Per-Target Results and Base-Policy Output Distributions

This appendix expands the aggregate curves in Section 5.4. The per-target TTSP curves show how adaptation varies across problems, while the base-policy output distributions explain when majority-style methods are vulnerable to selecting the wrong functional cluster.

## E.1 TTSP Individual Results

The aggregate curves in Figure 4 show that TTSP improves quickly and remains relatively flat across verifier strengths. The per-target plots below expose the heterogeneity behind that aggregate. They also support the phase-diagram interpretation in Figure 5: TTSP is most useful on targets with enough base model coverage to support curriculum composition, and much less reliable when the correct solution is nearly absent from the base distribution.

## E.2 Functional Output Distributions

Section 5.4 argues that FMV succeeds when the correct functional cluster is modal and fails when it is not. The plots in this appendix visualize the base policy’s functional output distributions directly. Each cluster is defined by execution behavior over target problem test inputs, as in FMV-style consensus selection (Launer et al., 2026). These distributions explain why majority-based selection can be brittle in the consistency regime: the correct solution may be present in the sample set but still be ranked below a more common incorrect functional cluster.

![](images/55b5bc1ddb070db224ab3802a6ea53ab29a023e9e84c7387b850d29bb03dd4d9.jpg)  
Figure 14: Per-target TTSP training curves on MidHardLCBv6. Mean@8 over GRPO steps for each individual target problem, with one panel per target and one line per evaluationtime verifier strength. Many targets show the rapid rise within 10–20 steps and stable plateau seen in the aggregate curve of Figure 4.

![](images/3720e0ebb17f3f3fc3a62002f82c5027930e3ef76edcaf0ec66340c7fdc782d4.jpg)  
Figure 15: Functional output clusters for MidHardLCBv6. Qwen 3 8B functional output cluster distribution for each target problem in MidHardLCBv6. Each panel shows, for one target, the fraction of 512 base rollouts whose execution traces fall into each functional cluster, with the correct cluster highlighted in green.

![](images/dc4b468d1fb8f193ead74e617aec7d08125d5fe950d14b2b7905cf357723dfb5.jpg)

![](images/f471f0b62f09f34aeee161b90780200ce72357b93c08a7c949f929fe4fbfcba3.jpg)  
Figure 16: Per-target TTSP training curves on VeryHardLCBv6. Mean@8 over GRPO steps for each individual target problem, with one panel per target and one line per evaluationtime verifier strength. The flat or noisy curves illustrate the discovery-side failure mode discussed in Section 5.4 when base coverage is below mean@512 0.1.  
Output Index  
Figure 17: Functional output clusters for VeryHardLCBv6. Qwen 3 8B functional output cluster distribution for each target problem in VeryHardLCBv6. Each panel shows, for one target, the fraction of 512 base rollouts whose execution traces fall into each functional cluster, with the correct cluster highlighted in green. The correct cluster typically carries negligible mass, matching the discovery-regime characterization in Figure 7.

## F Prompt Templates

The method description in Section 4 abstracts TTSP into proposer generation, refinement, solver attempts, and execution-based rewards. For reproducibility, we report the exact prompt templates used to instantiate those calls. The prompts are implementation details rather than conceptual contributions, so they appear after the experimental and diagnostic appendices. They are reproduced verbatim from the final system and therefore retain the older internal naming used in the implementation: target\_question denotes the target problem x<sup>⋆</sup>, and subgoal denotes what the main text calls a synthetic problem. Placeholders such as {target\_question}, {context}, {subgoal}, {example\_input}, and {example\_output} are filled at runtime.

## F.1 Initial proposer policy prompt

```markdown
Initial proposer prompt
## Task
You are creating a curriculum of subgoals to help a student learn the necessary skills to solve a target question .
Given the target question , propose a new subgoal and write a Python code that solves the subgoal . You must :
1. ** First , think step -by - step ** about what skills / concepts the student needs to learn to solve the target question .
2. ** Then , propose a subgoal ** that is reasonably challenging for the student and builds toward solving the hard task .
3. ** Finally , write the Python ground truth code ** that solves the subgoal .
** IMPORTANT : You MUST include detailed reasoning before proposing your subgoal . Responses that skip the thinking step
will be rejected .**
---
## Target question
{ target_question }
---
## Step 1: Reasoning ( REQUIRED )
1. ** Identify the core skills **: What algorithmic techniques , data structures , or mathematical concepts are relevant
to the target question ?
2. ** Identify possible subgoals **: Based on the required algorithmic techniques , data structures , or mathematical
concepts , identify and discuss three possible subgoals .
Be as specific as possible in your reasoning .
---
## Step 2: Subgoal Proposal ( REQUIRED )
After your reasoning , propose a subgoal that
- ** Targets relevant concepts **: The subgoal must practice concepts , knowledge and skills needed for the target
question .
- ** Is well - posed and complete **: The subgoal must be well - posed and complete without requiring any external context
to understand .
- ** Is unambiguous **: The subgoal must be as descriptive as possible so that it is completely unambiguous . Define
every term in the subgoal , including all inputs and outputs and what they represent .
You must wrap your subgoal in ` ` ` subgoal ` ` ` tags .
---
## Step 3: Code Requirements (RFOUIRED)
After the subgoal , write the Python code that fully solves the subgoal . Do not just write a placeholder function ,
write the actual code that solves the subgoal .
Your code must meet the following requirements :
Must accept input from stdin
Ensure the code prints a value to stdout
一 Include at least one input parameter
Make the function deterministic
Make the snippet require state tracking across multiple data transformations , ensuring the task requires long multi
step reasoning
- AVOID THE FOLLOWING :
Random functions or variables
* Date / time operations
* I/O operations ( reading files , network requests )
* Printing or logging
* Any external state
一 Ensure execution completes within 10 seconds on a modern CPU
All imports and class definitions should be at the very top of the code snippet
- The snippet should end with a print statement to stdout , only print the final result
- Do not print any example cases you make up yourself
You must wrap your code in ```python ``` tags .
---
## Output Format
You must structure your response in this exact order :
[ Write the detailed step - by - step reasoning . This section must be at least 4 paragraphs long .]
Then :
` ` ` subgoal
```

Write a function that takes a list of numbers and returns the sum of the numbers .   
Then :   
\` \` \` python   
arg = input ()   
# your code here   
print (...)   
---   
## Final reminder   
You must always reason step -by - step first , then propose a subgoal , then write the Python code that solves the subgoal .   
Write a complete python code solution that fully solves the subgoal , do not just write a placeholder function   
or only part of the solution .   
n nn

## F.2 Iterative proposer policy prompt

```markdown
Iterative proposer prompt
nnH
## Task
Given the target question and past attempted subgoals , reason about what the next subgoal should be , and then propose
a new subgoal , and then write a Python code that solves the subgoal .
## Target question
{ target_question }
{ context }
## Task description
1. Using the summary of the previous subgoals , reason about what the next subgoal should be. Carefully devise a
subgoal plan : e.g., understand the previous subgoals , reason about a relevant , appropriately challenging and
creative next subgoal . Isolate specific difficulties the student is having .
2. Write the subgoal . Wrap your subgoal in ` ` ` subgoal ` ` ` tags .
3. Write the solution to the subgoal in ` ` ` python ` ` ` tags . You can use the summary of previous subgoals to help guide
your solution .
## Subgoal Requirements
- The subgoal must target concepts , knowledge and skills in which the student performed poorly in previous subgoals
- The subgoal must be well - posed and complete without requiring any external context to understand
- The subgoal must be challenging for the student . If all previous subgoals are too easy , suggest a subgoal that is
more challenging ( for example , one which gives less hints about the solution , or is more complex , or tackles a
different aspect of the target question ) .
- The subgoal MUST be at least slightly different from previous subgoals
- The subgoal must be relevant to the target question
- Define every term in the subgoal , including all the inputs and outputs and what they represent
- The subgoal must be as descriptive as possible so that it is completely unambiguous
### Subgoal format :
- Wrap your subgoal in ``` subgoal ``` tags .
- For example :
` ` ` subgoal
Write a function that takes a list of numbers and returns the sum of the numbers .
### Code Requirements :
- Must accept input from stdin
- Ensure the code prints a value to stdout
- Include at least one input parameter
- Make the function deterministic
- Make the snippet require state tracking across multiple data transformations , ensuring the task requires long multi
step reasoning
- AVOID THE FOLLOWING :
* Random functions or variables
* Date / time operations
* I/O operations ( reading files , network requests )
* Printing or logging
* Any external state
- Ensure execution completes within 10 seconds on a modern CPU
- All imports and class definitions should be at the very top of the code snippet
- The snippet should end with a print statement to stdout , only print the final result
- Do not print any example cases you make up yourself
### Formatting :
- Format your code with :
``` python
arg = input ()
# your code here
print (...)
## Rules
- Be succinct !
- Don 't overanalyze . Do not analyze every task .
- Do not worry about being too descriptive in the subgoal , as long as it is unambiguous and complete .
- If all previous subgoals are too easy , suggest a subgoal that tackles a different aspect of the target question , or
is more challenging .
YOU MUST ALWAYS WRITE BOTH A SUBGOAL AND A PYTHON CODE SOLUTION .
n n n
```

## F.3 Solver policy prompt

Solver prompt   
## Task   
You are solving a programming subgoal . Given a subgoal description and examples , you must reason through the solution   
step - by - step and then write Python code that solves it .   
\*\* IMPORTANT : You MUST include detailed reasoning before writing your code . Responses that skip the thinking step will   
be rejected .\*\*   
---   
## Subgoal   
{ subgoal }   
### Example input :   
{ example\_input }   
### Example output :   
{ example\_output }   
---   
## Step 1: Reasoning ( REQUIRED )   
Before writing code , think step -by - step :   
1. \*\* Understand the subgoal \*\*: What is the input ? What is the expected output ? What are the constraints ?   
2. \*\* Identify the approach \*\*: What algorithm or technique will you use? Why is it appropriate ?   
3. \*\* Plan the implementation \*\*: What are the key steps in your solution ?   
## Step 2: Solution ( REQUIRED )   
After your reasoning , write clean Python code that solves the subgoal .   
You may use any appropriate techniques including :   
- Data structures : trees , heaps , stacks , queues , graphs , hash maps , etc .   
- Algorithms : dynamic programming , recursion , divide and conquer , greedy , backtracking , BFS/DFS , binary search , etc.   
### Program format :   
- Wrap your Python code in \` \` \` python \` \` \` tags .   
- Must accept input from stdin   
Ensure your code prints a value to stdout   
Make the function deterministic   
AVOID THE FOLLOWING :   
Random functions or variables   
Date / time operations   
\* I/O operations ( reading files , network requests )   
Printing or logging   
Any external state   
Ensure execution completes within 10 seconds on a modern CPU   
All imports and class definitions should be at the very top of the code snippet   
Use imports if it makes it easier to solve the subgoal   
The snippet should end with a print statement to stdout , only print the final result   
Your code will be tested on hidden test cases with varied inputs   
Your code must compute the answer , not assume it   
For example :   
\`\`\` python   
t = input ()   
# code logic here   
print ( result )   
- --   
## Output Format   
You must structure your response in this exact order :   
### Reasoning   
Analyze the subgoal , identify the algorithm / approach , and plan your implementation .   
### Solution   
\` \` \` python   
t = input ()   
# code logic here   
print ( result )   
Do NOT use if \_\_name\_\_ == " \_\_main\_\_ " in your code .   
Include comments explaining your approach . Remember to wrap your final code block in \`\`\` python and \`\`\` tags ."""

## F.4 Refinement / labeling prompt

Refinement / labeling prompt   
n nn   
{ ground\_truth }   
## Task   
Given the above Python code , you must :   
1. Analyze the code thoroughly   
2. Create a competitive programming subgoal where the Python code is the solution   
3. Create { num\_inputs } sets of example inputs for the code

```markdown
** IMPORTANT : You MUST include detailed reasoning before writing your subgoal and test cases . Responses that skip the
thinking steps will be rejected .**
- --
## Step 1: Code Analysis ( REQUIRED )
Before creating anything , analyze the code step -by - step :
1. ** Input parsing **: What inputs does the code expect ? What are their types and formats ( integers , strings , lists ,
etc .)?
** Core logic **: What algorithm or approach does the code implement ? What is its purpose ?
3. ** Output **: What does the code produce ? What is the output format ?
4 ** Constraints **: What are the implicit constraints (e.g., input ranges , sizes )?
5. ** Edge cases **: What boundary conditions does the code handle ?
## Step 2: Subgoal Design ( REQUIRED )
Design each component of the subgoal statement :
1. ** Subgoal narrative **: Create a clear scenario or abstract description that motivates the subgoal . What task is the
solver asked to accomplish ?
2. ** Input specification **: Define the exact input format :
- How many lines of input ?
What does each line contain ?
- What are the data types ( integers , strings , etc .)?
- Are there delimiters (spaces , commas )?
3. ** Output specification **: Define the exact output format :
- What should be printed ?
- What format ( single number , space - separated values , etc .)?
4. ** Constraints **: Specify bounds that match the code 's assumptions :
Value ranges ( e . g . , 1 \ leq n \ leq 10^5)
String lengths
- Array sizes
- Time / memory limits if relevant
## Step 3: Test Case Design ( REQUIRED )
Plan your { num_inputs } test inputs to ensure good coverage :
1. Include at least one simple / basic case
2. Include edge cases (e.g., minimum values , boundary conditions )
3. Include cases that exercise different code paths
4. Ensure all inputs are valid and will run successfully with the code
## Output Format
You must structure your response in this exact order :
### Code Analysis
Describe your understanding of the code 's inputs , logic , and outputs .
### Subgoal Design
Detail your decisions for: narrative , input format , output format , constraints , and examples .
### Test Case Design
Explain your strategy for creating diverse , valid test inputs .
### Subgoal Statement
``` subgoal
[ Your competitive programming subgoal statement here . Must include :
Clear subgoal description / narrative
Input format specification
Output format specification
Constraints
At least one example input / output
Do NOT leak any implementation details or algorithm hints .]
### Test Cases
Provide exactly { num_inputs } sets of inputs :
` ` ` input
arg1 \\ narg2 \\ n ...
` ` ` input
arg1 \\ narg2 \\ n ...
---
## Important Rules
- The subgoal statement must EXACTLY and correctly describe what the Python code does
Do NOT leak any information about the Python solution in the subgoal statement
Do NOT mention specific algorithms , data structures , or implementation hints
Each set of inputs must be enough to run the Python code once
Format multiple arguments delimited by \\n
Do NOT add quotes around arguments . Write values exactly as they would appear in stdin
- Do NOT provide explanations of test cases after the test inputs
n nn
```