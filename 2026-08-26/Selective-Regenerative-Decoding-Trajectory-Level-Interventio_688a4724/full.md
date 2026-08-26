# Selective Regenerative Decoding: Trajectory-Level Intervention for Inference-Time Reasoning

Sophia Xiao Pu<sup>1</sup>\*, Yumo Xu<sup>2†</sup>, Sailik Sengupta<sup>3‡</sup>, Millennium Bismay<sup>3</sup>, Ruixue Lian<sup>4†</sup>, James Gung<sup>3</sup>, Yi-an Lai<sup>3</sup>, Arshit Gupta<sup>3</sup> <sup>1</sup>University of California, Santa Barbara, <sup>2</sup>Netflix, <sup>3</sup>Amazon Science, <sup>4</sup>Meta

## Abstract

Inference-time decoding methods improve LLM reasoning by exploring multiple candidate trajectories, yet treat each trajectory as atomic—either retaining it whole or discard ing it irreversibly. This wastes computation on partially promising candidates whose highquality prefixes are abandoned alongside degraded suffixes. We introduce Selective Regenerative Decoding (SRD), which routes each candidate to discard, keep, or refine only the degraded portion of suffix while preserving all the useful prefix of borderline candidates — without requiring a larger target model. Under mild assumptions, SRD achieves a provable 1.28– 1.36× sample efficiency gain over rejection sampling with strictly higher expected trajectory quality, with the gain growing as the candidate pool grows. Across MATH500, GPQA Diamond, HotpotQA, and AlpacaEval with multiple generation–reward model pairs, SRD matches Best-of-N accuracy with substantially fewer generated tokens and outperforms speculative rejection in low-compute regimes. By enabling segment-level intervention rather than whole-trajectory selection, SRD opens a previously underexplored region of the accuracy– compute tradeoff for inference-time reasoning.

## 1 Introduction

Allocating additional computation at inference time improves the reasoning quality of large language models (Huang et al., 2025; Khanov et al., 2024; Stiennon et al., 2022; Sun et al., 2024; Mudgal et al., 2023; Liao et al., 2025). While ARGS (Khanov et al., 2024), and DeAL (Huang et al., 2025) formulate decoding as a general reward-guided search procedure, a growing body of work seeks to improve the accuracy and efficiency of the search. Best-of-N (Nakano et al., 2021;

Best-of-N  
![](images/5d5d59e4252d9f83c1b7595387ed87e708bbfad438a0c36498006c6f01906347.jpg)

Speculative Rejection  
![](images/5a4402da66852fbf9f995a5efa687170caffb31aa480ede99494f43e8e0c79b0.jpg)

SRD (Ours)  
![](images/5f0db741be29d4818ee1603c69833d98f6cb7125b25419d5ea5849692d35a813.jpg)  
Figure 1: Comparison of decoding strategies on three trajectories (τ<sub>1</sub>: high quality throughout; τ<sub>2</sub>: highquality prefix with degraded suffix; τ<sub>3</sub>: low quality). Best-of-N fully generates all candidates and selects the best, discarding the rest. Speculative Rejection terminates trajectories when quality drops, saving compute but permanently losing useful prefixes. SRD preserves τ<sub>2</sub>’s high-quality prefix and selectively regenerates only its degraded suffix (teal), recovering a trajectory that both baselines waste.

Stiennon et al., 2022) generates multiple complete trajectories and selects the highest-scoring one; speculative rejection (Sun et al., 2024) terminates low-reward prefixes early to reduce cost; Reward-guided Speculative Decoding (Liao et al., 2025) uses a reward model to accept or reject draft steps from a smaller model; and Controlled Decoding (Mudgal et al., 2023) steers token selection via a learned prefix-aware value function.

Despite their differences, these methods treat reasoning trajectories as atomic units. Best-of-N defers all decisions until full trajectories are generated, incurring redundant computation. Speculative rejection makes earlier decisions, but they are irreversible—once a prefix is rejected, any highquality reasoning it contains is permanently lost. This is particularly wasteful for long-form reasoning, where trajectories frequently contain strong prefixes that degrade only in later steps. A more efficient strategy would preserve what is already good and intervene only where quality drops.

We introduce Selective Regenerative Decoding (SRD), a sample-efficient decoding algorithm that enables segment-level intervention within reasoning trajectories (Figure 1). SRD operates in three phases: it generates multiple candidates, routes each to KEEP, REFINE, or DISCARD based on ranknormalized reward scores, and selectively regenerates only the degraded suffixes of borderline candidates using higher-temperature sampling—without requiring a larger target model. By preserving useful prefixes and rewriting only where quality degrades, SRD recovers computation that rejectionbased methods waste.

We prove that, under mild assumptions, SRD achieves a provable 1.28–1.36× sample efficiency gain over rejection sampling with strictly higher expected best-trajectory quality with the gain growing as the candidate pool grows—characterized in general by $( 1 + \rho \cdot p _ { M } / p _ { H } )$ and satisfies formal termination and weak monotonicity guaranties (§3.3). Empirically, across MATH500 (Hendrycks et al., 2021), GPQA Diamond (Rein et al., 2024), HotpotQA (Yang et al., 2018), and AlpacaEval (Li et al., 2023) with multiple generation– reward model combinations, SRD matches Best-of-N accuracy with substantially fewer generated tokens and outperforms speculative rejection in lowcompute regimes (§4.2). Ablation studies reveal that the effectiveness of SRD’s refinement depends on reward model calibration: conservative global rerouting excels under stable rewards, while local self-comparison is more robust under noisy signals (§4.4.2).

Because SRD requires only a generative model and a reward model, it is complementary to speculative decoding (Leviathan et al., 2023) and prefix value functions (Mudgal et al., 2023), which can be composed with it. Our main contributions are:

• A three-phase generation–routing–refinement algorithm that enables segment-level intervention in reasoning trajectories without a target model.

• Formal proofs that SRD achieves strictly better sample efficiency and expected trajectory quality than rejection sampling, with termination and monotonicity guarantees.

• Empirical demonstration that SRD consistently improves the accuracy–compute tradeoff across four benchmarks spanning mathematical reasoning, multi-hop QA, and instruction following, with ablations characterizing when and why refinement helps.

## 2 Related Work

Standard decoding methods, beyond greedy decoding, explore the local neighborhood for every generated token, such as top-k (Fan et al., 2018), nucleus sampling (Holtzman et al., 2019), temperature-based token-sampling (Ackley et al., 1985), or the local neighborhood of a path of decoded tokens, such as beam-search (Freitag and Al-Onaizan, 2017), contrastive search (Su et al., 2022), and Best-of-N (Nakano et al., 2021; Stiennon et al., 2022). While exploring a larger search neighborhood may lead to decoding candidates that are more optimal (w.r.t. a particular objective), such non-adaptive policies represent uninformed search strategies guided by the generation model’s priors and language heuristics.

When viewed as a search process in the space of tokens, we can impose informed heuristics to better guide generated trajectories (Och et al., 2001; Lu et al., 2022; Khanov et al., 2024; Huang et al., 2025). For problems that need reasoning beyond what is well encoded in the language model’s priors, these reward signals can be used during the forward-pass (Willard and Louf, 2023; Wang et al., 2023), with look-ahead (Bertsch et al., 2023; Wan et al., 2023; Huang et al., 2025; Nakshatri et al., 2025), and in exploring search graphs/trees (Yao et al., 2023; Roy et al., 2024).

Along these lines, our work relates more closely to methods that consider pruning search trajectories from a generated set. For example, speculative rejecting (Sun et al., 2024) evaluates top-k beams or candidate paths with a reward model, and discards less promising prefixes early. Unfortunately, its binary rejection strategy can permanently discard promising candidates that have low-rewards at the start. To relax this strict rejection criterion, Reward-Guided Speculative Decoding [RSD; (Liao et al., 2025)] performs a relaxed variant of speculative decoding (Leviathan et al., 2023), where a draft model proposes a candidate reasoning step and a reward model decides to accept/reject the draft step. If rejected, a larger (and more capable) reasoning model is used to generate the next reasoning step. However, RSD operates on a single search candidate and doesn’t explore multiple search candidate.<sup>1</sup> On the other hand, Controlled Decoding (CD), which introduced an inference-time guidance of policy model’s token selection by using a learned prefix value function in tandem, is an implicit way to influence the local search neighborhood for each token rather than having an explicit rejection mechanism (Mudgal et al., 2023).

In our work, we first consider generating set of multiple reasoning paths at each reasoning step. Then, an explicit reward model assesses the promise of each candidate in the set, marking them to be either kept, discarded, or refined. When salvaged, the search candidate undergoes localized surgery to allow exploration of the search neighborhood. Thus, we mitigate the lack of look-ahead in existing approaches and explore the search neighborhood without the need for any target/teacher model (used in speculative decoding and RSD) or a prefix value-function (in CD). Note that both of these elements can thereby be used with SRD.

## 3 Approach

We present SELECTIVE REGENERATIVE DECOD-ING (SRD), a sample-efficient search algorithm that improves upon standard rejection sampling by introducing a refinement mechanism (without access to a larger target model) for borderline candidates. Unlike rejection sampling, which treats trajectory generation as atomic, our approach recognizes that trajectories are compositional and that many generated trajectories can contain highquality prefixes but suffer from degradation in later steps. Thus, discarding such trajectories entirely wastes valuable computation. Instead, we propose to salvage these candidates through targeted regeneration.

## 3.1 Problem Setting

Consider a discrete action space A and let $\boldsymbol { \mathcal { T } } = \boldsymbol { \mathcal { A } } ^ { k }$ denote the space of trajectories, where each trajectory $\tau = ( a _ { 1 } , a _ { 2 } , \ldots , a _ { k } )$ is a sequence of k discrete actions. We assume access to:

$\bullet \mathbf { A }$ generative model $\mathcal { G }$ that produces trajectories $\tau \sim \mathcal { G }$

• A reward model $R : \mathcal T $ R that scores trajectory quality.

Our goal is to identify high-quality trajectories efficiently, minimizing the number of samples required to obtain trajectories exceeding a quality threshold, all without access to a target/teacher model.

## 3.2 Algorithm Overview

At each search iteration, SRD operates in three phases: generation, routing, and refinement. Algorithm 1 provides the complete procedure.

Phase 1: Generation. We sample n candidate trajectories independently from the generative model:

$$
\mathcal { C } = \{ \tau _ { 1 } , \tau _ { 2 } , \dotsc , \tau _ { n } \} , \quad \tau _ { i } \sim \mathcal { G } .\tag{1}
$$

Phase 2: Routing. Each candidate is evaluated using the reward model and assigned a normalized score based on its relative rank. Specifically, we sort candidates by their reward scores in descending order and assign each trajectory a rank $r \in \{ 0 , 1 , \ldots , n - 1 \}$ , where $r = 0$ denotes the highest-scoring candidate. We then compute a normalized score:

$$
u ( \tau ) = 1 - \frac { r ( \tau ) } { n - 1 } ,\tag{2}
$$

yielding $u \in [ 0 , 1 ]$ with larger values indicating better relative quality. Then, routing decisions are determined by comparing $u ( \tau )$ against two thresholds, $\theta _ { \mathrm { l o w } }$ and $\theta _ { \mathrm { h i g h } }$ , where $0 \leq \theta _ { \mathrm { l o w } } < \theta _ { \mathrm { h i g h } } \leq 1 \mathrm { : \Omega }$

$$
\begin{array} { r } { \mathrm { R o U T E } ( \tau ) = \left\{ \begin{array} { l l } { \mathrm { K e e p , ~ } } & { u ( \tau ) \geq \theta _ { \mathrm { h i g h } } , } \\ { \mathrm { R e f i n e , ~ } } & { \theta _ { \mathrm { l o w } } < u ( \tau ) < \theta _ { \mathrm { h i g h } } , } \\ { \mathrm { D i s c a r d , } } & { u ( \tau ) \leq \theta _ { \mathrm { l o w } } . } \end{array} \right. } \end{array}\tag{3}
$$

Phase 3: Regeneration. Trajectories routed to REFINE undergo a targeted regeneration procedure. For each such trajectory τ , we identify a regeneration boundary by evaluating reward scores at fixed intervals of $m \textless$ k steps and selecting the earliest position $j ^ { * }$ at which reward degrades relative to the preceding prefix:

$$
\begin{array} { r } { j ^ { \ast } = \operatorname* { m i n } \bigl \{ j \in \{ m , 2 m , \dots \} : \qquad } \\ { R ( \tau _ { 1 : j + m } ) < R ( \tau _ { 1 : j } ) \bigr \} } \end{array}\tag{4}
$$

We then regenerate the suffix $\tau _ { j ^ { * } + 1 : k }$ using a highertemperature sampling strategy, updating the trajectory:

$$
\begin{array} { c } { { \tau ^ { \prime } = ( \tau _ { 1 : j ^ { * } } , \tilde { \tau } _ { j ^ { * } + 1 : k } ) , } } \\ { { \tilde { \tau } _ { j ^ { * } + 1 : k } \sim \mathcal G _ { \mathrm { h i g h - t e m p } } ( \cdot \mid \tau _ { 1 : j ^ { * } } ) . } } \end{array}\tag{5}
$$

The refined trajectory $\tau ^ { \prime }$ is then re-evaluated and ranked jointly with other active candidates. It is retained only if subsequently routed to KEEP; otherwise, it is discarded.

Termination Guarantees. To ensure termination, we impose two constraints: (i) a maximum regeneration span length $L _ { \mathrm { m a x } }$ , and (ii) a maximum number of refinement attempts per trajectory $N _ { \mathrm { r e f i n e } }$ . These bounds guarantee that the algorithm terminates in at most $n \cdot N _ { \mathrm { r e f i n e } }$ refinement operations.

## 3.3 Theoretical Analysis

We establish two main results: SRD has (i) improved sample efficiency compared to rejection sampling, and (ii) higher expected solution quality. We begin by introducing the necessary notation and assumptions.

Notation and Assumptions Let the generative distribution $\mathcal { G }$ induce the following partition of probability mass:

$$
p _ { H } = \mathbb { P } _ { \tau \sim \mathcal { G } } \left[ u ( \tau ) \geq \theta _ { \mathrm { h i g h } } \right] ,\tag{6}
$$

$$
p _ { M } = \mathbb { P } _ { \tau \sim \mathcal { G } } \left[ \theta _ { \mathrm { l o w } } < u ( \tau ) < \theta _ { \mathrm { h i g h } } \right] ,\tag{7}
$$

$$
p _ { L } = \mathbb { P } _ { \tau \sim \mathcal { G } } \left[ u ( \tau ) \leq \theta _ { \mathrm { l o w } } \right] ,\tag{8}
$$

where $p _ { H } + p _ { M } + p _ { L } = 1$ . These are populationlevel quantities; the rank-based score $u ( \tau )$ is a consistent finite-sample estimator of them Appendix-C.1.

Assumption 3.1 (Refinement Efficacy). There exists a constant $\rho \in ( 0 , 1 ]$ such that for any trajectory τ with $\theta _ { \mathrm { l o w } } < u ( \tau ) < \theta _ { \mathrm { h i g h } }$ , the refinement procedure produces an acceptable trajectory with probability at least $\rho \colon$

$$
\mathbb { P } \left[ \mathrm { R O U T E } \big ( \mathrm { R E F I N E } ( \tau ) \big ) = \mathrm { K E E P } \right] \ge \rho .\tag{9}
$$

This assumption captures the intuition that refinement is beneficial, i.e. trajectories in the middle tier have salvageable prefixes, and regenerating their suffixes with increased diversity yields acceptable results with non-negligible probability. Results in section 4 support this empirically.

Assumption 3.2 (Independence). Refinement outcomes are independent across trajectories, and each refinement attempt is independent of previous attempts on the same trajectory.

Sample Efficiency Our first result establishes SRD requires fewer samples than pure rejection sampling to obtain at least one acceptable trajectory with high probability.

Theorem 3.3 (Sample Efficiency). Let $\delta \in ( 0 , 1 )$ be a failure probability. Under Assumptions 3.1 and 3.2, (i) pure rejection sampling requires at least $N _ { \mathrm { r e j e c t } } \overset { \cdot } { \underset { } { > } } \frac { \ln ( 1 \bar { / } \delta ) } { p _ { H } }$ samples to obtain at least one acceptable trajectory with probability $\geq 1 - \delta .$ In comparison, (ii) SRD requires at least $N _ { \mathrm { S R D } } \geq$ $\frac { \ln ( 1 / \delta ) } { p _ { H } + \rho \cdot p _ { M } }$ samples to achieve the same guarantee. Thus, (iii) the efficiency gain of SRD over rejection sampling is characterized by:

$$
{ \frac { N _ { \mathrm { r e j e c t } } } { N _ { \mathrm { r e f i n e } } } } = 1 + { \frac { \rho \cdot p _ { M } } { p _ { H } } } .\tag{10}
$$

The complete proof is provided in Appendix C.3 and emperical evidence is provided in Appendix F. We note that the efficiency gain is most pronounced when $p _ { H }$ is small (high-quality trajectories are rare) and $p _ { M } \cdot \boldsymbol { \rho }$ is substantial (many refinable trajectories exist and refinement is effective). In the regime where $\rho \cdot p _ { M } \geq p _ { H }$ , SRD achieves at least a factor of 2 improvement in sample efficiency.

Expected Quality Improvement Beyond sample efficiency, SRD also improves the expected quality of the best trajectory found.

Assumption 3.4 (Stochastic Dominance of Refinement). For trajectories in the refinable region, the refinement operation produces outputs whose reward distribution stochastically dominates that of fresh samples from the same region:

$$
\begin{array} { r } { R ( \mathrm { R E F I N E } ( \tau ) ) \succeq \mathrm { s t } R ( \tau ^ { \prime } ) \quad \mathrm { f o r } \tau , \tau ^ { \prime } } \\ { \mathrm { w i t h } \theta _ { \mathrm { l o w } } < u ( \tau ) , u ( \tau ^ { \prime } ) < \theta _ { \mathrm { h i g h } } } \end{array}\tag{11}
$$

where $\succeq \mathrm { s t }$ denotes first-order stochastic dominance. This holds naturally given a larger target model for regeneration, and is motivated here by refinement preserving high-quality prefixes while regenerating problematic suffixes with increased diversity.

Theorem 3.5 (Expected Quality Improvement). Let $R _ { \operatorname* { m a x } } ^ { \mathrm { R S } } ( n ) \ = \ \operatorname* { m a x } _ { i \in [ n ] } R ( \tau _ { i } )$ denote the best reward from n samples under pure rejection sampling, and let $R _ { \operatorname* { m a x } } ^ { \mathrm { S R D } } ( n )$ denote the best rewardfrom SRD. Under Assumptions 3.1–3.4:

$$
\mathbb { E } \left[ R _ { \operatorname* { m a x } } ^ { \mathrm { S R D } } ( n ) \right] \geq \mathbb { E } \left[ R _ { \operatorname* { m a x } } ^ { \mathrm { R S } } ( n ) \right] + \Delta _ { n } ,\tag{12}
$$

where $\Delta _ { n } > 0$ for all $n \geq 1$ .Moreover, under mild regularity conditions on the reward distribution, the improvement satisfies:

$$
\Delta _ { n } = \Omega \left( \frac { p _ { M } \cdot \rho } { n \cdot ( 1 + p _ { M } \cdot \rho ) } \right) .\tag{13}
$$

While the complete proof appears in $\mathsf { A p - }$ pendix C.4, we highlight that the improvement $\Delta _ { n }$ decreases with $n ,$ which may seem counterintuitive. This reflects the diminishing marginal returns of additional candidates, i.e. as n grows, both algorithms are increasingly likely to find near-optimal trajectories, so the gap between them shrinks. Emperical proof has been provided in Appendix G.

Finally, SRD satisfies two correctness properties that follow directly from the algorithm construction: it terminates in at most $n \cdot N _ { \mathrm { r e f i n e } }$ refinement operations and $O ( n \cdot N _ { \mathrm { r e f i n e } } )$ reward evaluations, and it is weakly monotone — the best kept reward $R _ { t } ^ { * }$ is non-decreasing in the number of refinement operations t, so refinement never degrades the incumbent solution. Formal statements and proofs are in Appendix C (Propositions C.7 and C.8).

## 4 Experiments

## 4.1 Experimental Settings

Tasks and Metrics. We evaluate on four datasets spanning reasoning and instruction-following: MATH500 (Hendrycks et al., 2021) (mathematical reasoning, accuracy), GPQA Diamond (Rein et al., 2024) (scientific QA, accuracy following Liao et al. (2025)), HotpotQA (Yang et al., 2018) (multi-hop QA, EM and F1), and AlpacaEval (Li et al., 2023) (instruction following, GPT-4o-mini win rate following Sun et al. (2024)).

Generation and Reward Models. We pair taskappropriate generation and reward models, using the ten combinations listed in Table 1. Reward models are matched to task type: mathand reasoning-specific models for MATH500 and GPQA, a retrieval-augmented QA model for HotpotQA, and general-purpose preference models for AlpacaEval. We intentionally decouple generation and reward models throughout, so the evaluation signal is external rather than biased toward the generator’s own preferences (Panickssery et al., 2024). The same generation model serves as both drafter and editor. Prompt templates are in Appendix D.

<table><tr><td>Generation Model</td><td>Reward Model</td></tr><tr><td>MATH500</td><td></td></tr><tr><td>Llama-3.1-8B-Instruct</td><td>AceMath-7B-RM</td></tr><tr><td>Qwen2.5-Math-1.5B-Instruct</td><td>AceMath-7B-RM</td></tr><tr><td>GPQA DIAMOND</td><td></td></tr><tr><td>Llama-3.1-8B-Instruct</td><td>Skywork-o1-Open-PRM-7B</td></tr><tr><td>Qwen3-4B-Instruct</td><td>Skywork-o1-Open-PRM-7B</td></tr><tr><td>AlpacaEval</td><td></td></tr><tr><td>Llama-3.1-8B-Instruct</td><td>FsfairX-LLaMA3-RM-v0.1</td></tr><tr><td>Llama-3.1-8B-Instruct</td><td>RM-Mistral-7B</td></tr><tr><td>Qwen3-4B-Instruct</td><td>FsfairX-LLaMA3-RM-v0.1</td></tr><tr><td>Qwen3-4B-Instruct</td><td>RM-Mistral-7B</td></tr><tr><td></td><td></td></tr><tr><td>HotpotQA Llama-3.1-8B-Instruct</td><td>Llama3.1-RAG-Reward-v2</td></tr></table>

Table 1: Generation and reward model combinations used for each dataset.

Baselines. We compare against three inferencetime decoding baselines, all sharing SRD’s generation model, prompt template, and sampling hyperparameters and differing only in decoding strategy: Temperature Sampling (N=1, a single trajectory); Best-of-N (BoN), which fully generates N candidates and selects the highest-reward one; and Speculative Rejection (Spec-Rej) (Sun et al., 2024), which scores prefixes during generation and permanently terminates low-reward trajectories. Full baseline descriptions are in Appendix A.1. We exclude Reward-guided Speculative Decoding (Liao et al., 2025) from the main comparison because it assumes access to a larger target model for regeneration, whereas SRD and all baselines above operate with only a single generative model and a reward model. We provide a controlled comparison with a speculative variant of SRD under RSD’s setting in Appendix C.

## 4.2 Main results

We evaluate Selective Regenerative Decoding (SRD) on four benchmarks and compare it with several inference-time decoding baselines under an accuracy–compute tradeoff setting. Throughout this section, we report task-level evaluation metrics (e.g., accuracy, F1, or win rate), rather than reward scores. All decoding hyperparameters and implementation details are provided in Appendix E.

![](images/6bb3bcb009d48c1d5de1f7d1387c225a4bb2d8942174cfc20e5ca1bbead0a4f5.jpg)

Figure 2: Accuracy-compute trade-offs of inference-time decoding methods across four benchmarks and multiple generation-reward model combinations. Compute is measured by the average number of generated tokens from the language model. SRD consistently achieves competitive performance with respect to Best-of-N and Speculative Rejection under lower or comparable generation budgets by selectively reusing and refining partial trajectories.
<table><tr><td>N</td><td>Acc.</td><td>Time (s)</td><td>In Tok.</td><td>Out Tok.</td><td>Drafter (time/calls)</td><td>Scorer (time/calls)</td><td>Editor (time/calls)</td><td>Router (time/calls)</td></tr><tr><td>10</td><td>0.544</td><td>7.21</td><td>6629</td><td>2166</td><td>4.73 / 4.1</td><td>1.29 / 5.6</td><td>1.18 / 1.0</td><td>&lt;0.01 /3.8</td></tr><tr><td>50</td><td>0.611</td><td>18.07</td><td>32895</td><td>10927</td><td>7.43 / 4.2</td><td>6.98 / 6.6</td><td>3.59/ 2.2</td><td>&lt;0.01 / 5.3</td></tr><tr><td>100</td><td>0.640</td><td>29.76</td><td>65737</td><td>21840</td><td>10.96 / 4.2</td><td>13.53 / 6.7</td><td>5.13 / 2.3</td><td>&lt;0.01 / 5.5</td></tr></table>

Table 2: Component-level breakdown of SRD on MATH500 under different values of N. We report overall accuracy, runtime, token statistics, and per-component time and call frequency in the format of time / calls.

Overall Trends. As shown in Figure 2, SRD consistently forms a distinct tradeoff frontier between the early rejection method (Spec-Rej) and full trajectory sampling (Best-of-N). In particular, SRD achieves accuracy levels comparable to or exceeding Spec-Rej when limited additional computation is permitted.

Rather than operating at a single fixed point, SRD explores a different region of the accuracy– compute tradeoff by enabling segment-level intervention within a single trajectory. This allows computation to be allocated conditionally, based on intermediate trajectory quality, rather than uniformly across complete candidates.

This trend is particularly evident on reasoningcentric tasks such as MATH500, GPQA, and HotpotQA, where intermediate reasoning errors are common and suggest opportunities for partial trajectory correction.

Mathematical and Reasoning Tasks. On MATH500 with LLaMA-3.1-8B, SRD achieves 0.544 accuracy at N=10 using only 2,166 output tokens, scaling to 0.640 at N=100 with 21,840 tokens (Table 2). As shown in Figure 2, these accuracy levels are comparable to Best-of-N at substantially lower token budgets for both Qwen2.5-Math-1.5B and LLaMA-3.1-8B. While Best-of-N can further improve accuracy by sampling more complete candidates, this comes at the cost of proportionally increased token generation.

On the more challenging GPQA benchmark, overall accuracy is lower across all methods, reflecting task difficulty. SRD exhibits a more stable accuracy–compute curve in the low- and midbudget regimes. Although Best-of-N can achieve higher accuracy under large budgets, its compute cost grows rapidly, indicating diminishing returns from purely increasing candidate diversity.

Multi-hop Question Answering. On HotpotQA, SRD attains performance comparable to Best-of-N using a moderate number of generated tokens. Given that HotpotQA requires multi-step reasoning over multiple pieces of evidence, this result suggests that selectively refining partial trajectories can effectively reduce redundant generation of entire reasoning chains.

Instruction-following Evaluation. On AlpacaEval, evaluation is based on preference comparisons and therefore exhibits higher variance. Despite this, SRD demonstrates a consistent accuracy–compute trend across all four generation–reward model combinations tested (Table 1). At similar or lower generation budgets, SRD typically achieves win rates comparable to Best-of-N and substantially outperforms single-sample decoding, indicating that SRD remains effective even under noisier reward signals.

## 4.3 Empirical Validation of Theoretical Quantities

To directly validate the quantities in Theorem 3.3, we measure the routing distribution and refinement success rate on MATH500 with Qwen2.5-Math-1.5B-Instruct (N=10). The accuracy–compute tradeoffs in Figure 2 and the scaling behavior in Table 2 (0.544 at N=10 to 0.640 at N=100) provide indirect support for sample efficiency, but do not directly measure ρ, p<sub>H</sub>, or $p _ { M } ;$ we address this gap here. We observe $p _ { H } ~ = ~ 0 . 5 2 5$ $p _ { M } = 0 . 1 4 8 .$ , and $p _ { L } = 0 . 3 2 4$ , with a refinement efficacy of $\hat { \rho } = 1 . 0$ (all refinement attempts yielded trajectories routed to KEEP). Substituting into the efficiency gain formula from Theorem 3.3 gives $1 + \hat { \rho } { \cdot } p _ { M } / p _ { H } = 1 { . } 2 8$ , indicating that SRD requires approximately 22% fewer samples than pure rejection sampling to achieve the same success guarantee. These results are consistent across N: at $N { = } 2 0$ and N=30, we observe $\hat { \rho } = 1 . 0$ with efficiency gains of 1.35× and 1.36× respectively, as larger candidate pools produce more mid-tier trajectories amenable to refinement. Extended results across values of N are provided in Appendix F.

<table><tr><td> $\theta _ { \mathrm { h i g h } }$ </td><td> $\theta _ { \mathrm { l o w } }$ </td><td>Accuracy</td><td>Out Tokens</td></tr><tr><td>0.5</td><td>0.3</td><td>0.5444</td><td>2166</td></tr><tr><td>0.5</td><td>0.5</td><td>0.5400</td><td>1973</td></tr><tr><td>0.5</td><td>0.2</td><td>0.5267</td><td>2325</td></tr><tr><td>0.8</td><td>0.5</td><td>0.4667</td><td>1871</td></tr><tr><td>0.7</td><td>0.3</td><td>0.4467</td><td>2091</td></tr></table>

Table 3: Effect of routing thresholds on MATH500 (Llama-3.1-8B-Instruct, N = 10).

We additionally validate Theorem 3.5 by comparing $R _ { \mathrm { m a x } }$ between SRD and rejection sampling under a coupled setting where both methods start from identical initial drafts (via seeded generation). SRD achieves higher mean $R _ { \mathrm { m a x } }$ than RS at both $N { = } 1 0 \left( \Delta _ { n } = + 0 . 3 6 \right)$ and $N { = } 2 0 \left( \Delta _ { n } = + 0 . 2 8 \right)$ with SRD winning more samples in both cases (23 vs. 9 and 27 vs. 12, respectively). Refinement success rates range from 78–100%, confirming Assumption 3.4. Notably, the SRD advantage grows with N $( \Delta _ { n } = + 1 . 2 9$ at N=30), as larger candidate pools produce more mid-tier trajectories amenable to refinement. Full details are provided in Appendix G.

## 4.4 Ablation Studies

## 4.4.1 Effect of Routing Thresholds

We analyze the effect of routing thresholds $\left( \theta _ { \mathrm { h i g h } } , \theta _ { \mathrm { l o w } } \right)$ on SRD using MATH500 with Llama-3.1-8B-Instruct $( N = 1 0 )$ , while keeping all other settings fixed. Table 3 reports representative threshold configurations. Performance degrades when the thresholds are overly aggressive or overly permissive, while the default setting (0.5, 0.3) achieves the best accuracy-compute tradeoff. We use these defaults across all benchmarks and model combinations in the main results (Figure 2), which provides implicit cross-task validation of this choice.

## 4.4.2 Effect of Refinement Policies

We investigate how different refinement policies— the decision rule applied after regenerating a candidate routed to REFINE—affect SRD’s behavior. We vary only the refinement policy and fix all other settings.

Refinement Policy Variants. After a candidate routed to REFINE undergoes suffix regeneration (Phase 3 in §3.2), a policy determines whether the refined candidate is retained. We consider five policies:

<table><tr><td colspan="2">Refinement Policy Accuracy Out Tokens</td></tr><tr><td colspan="2">Dataset: MATH500</td></tr><tr><td>Reroute (global) 0.5444 Keep on refine 0.5333</td><td>2166 2149 2162</td></tr><tr><td>Force keep Refine-Bo3 0.5222 Refine-Bo10 0.5244</td><td>0.5267 2521 3819</td></tr><tr><td>Self-compare 0.5156 Refine-Bo5 0.4906</td><td>2175 2238</td></tr><tr><td colspan="2">Dataset: GPQA</td></tr><tr><td>Self-compare 0.3243 Refine-Bo10 0.3176</td><td>2201 4030</td></tr><tr><td>Force keep 0.3041</td><td>2212</td></tr><tr><td>Reroute (global) 0.2973</td><td>2214</td></tr><tr><td>Refine-Bo5 0.2635</td><td>2185</td></tr><tr><td></td><td></td></tr><tr><td>Refine-Bo3 0.2635</td><td>2618</td></tr><tr><td></td><td></td></tr><tr><td>Keep on refine 0.2635</td><td>2185</td></tr></table>

Table 4: Effect of different refinement policies on representative reasoning tasks (N = 10).

Reroute (global) (default) ranks the refined candidate jointly with all active candidates and retains it only if it is routed to KEEP. Keep on refine follows the same rerouting procedure but also retains candidates that remain in the REFINE tier. Self-compare compares the refined candidate’s reward against its pre-refinement reward, retaining it if $R ( \tau ^ { \prime } ) \geq R ( \tau ) + \delta$ , where δ is a fixed margin $( \delta = 0 . 0 2$ in this study). Force keep unconditionally retains all refined candidates. Refine-BoN performs N parallel regenerations for each refined candidate and retains the highest-reward result $( N \in \{ 3 , 5 , 1 0 \} )$ ).

Results. Table 4 reports results on MATH500 and GPQA. We use Llama-3.1-8B-Instruct as the generation model; for reward models, we use AceMath-7B-RM on MATH500 and Skyworko1-Open-PRM on GPQA. On MATH500, where the reward model provides relatively stable and well-calibrated feedback, the conservative Reroute (global) strategy achieves the highest accuracy while maintaining low generation cost. More permissive strategies and aggressive refinement variants with internal Best-of-N substantially increase generation without improving accuracy. In contrast, on GPQA, which relies on a noisier process reward model, the locally grounded Self-compare strategy achieves the best performance while using fewer generated tokens. Global rerouting-based strategies perform significantly worse, suggesting that global candidate ranking is less reliable under noisy reward signals. Notably, across both regimes, increasing refinement aggressiveness or internal sampling (Refine-BoN) does not improve accuracy and often increases token cost substantially.

<table><tr><td>Scoring Interval Accuracy</td><td>Out Tokens</td></tr><tr><td>Dataset: MATH500</td><td></td></tr><tr><td>1</td><td>0.5533 2389</td></tr><tr><td>5 0.5422</td><td>2383</td></tr><tr><td>10 0.5667</td><td>2387</td></tr><tr><td>20</td><td>0.5556 2388</td></tr><tr><td>50 0.5689</td><td>2387</td></tr><tr><td>100</td><td>0.5467 2390</td></tr><tr><td>Dataset: GPQA</td><td></td></tr><tr><td>1 0.2365</td><td>2208</td></tr><tr><td>5 0.2973</td><td>2189</td></tr><tr><td>10 0.2973</td><td>2214</td></tr><tr><td>20 0.2500</td><td>2191</td></tr><tr><td>50 0.2432</td><td>2209</td></tr><tr><td>100 0.2365</td><td>2218</td></tr></table>

Table 5: Ablation on scoring interval (localization granularity), which determines where SRD begins suffix regeneration within a trajectory routed to REFINE.

## 4.4.3 Effect of Scoring Interval

Mechanism. The scoring interval controls the granularity at which SRD localizes the regeneration boundary j<sup>∗</sup> within a trajectory routed to REFINE (see Phase 3 in §3.2). Smaller intervals enable finergrained localization but are more sensitive to local reward fluctuations; larger intervals are more stable but may delay detection of the degradation point.

Observation. Across both GPQA and MATH500, as shown in Table 5, extremely small intervals lead to unstable or premature regeneration, while overly large intervals allow errors to propagate before being detected. A moderate interval consistently yields the most reliable refinement, suggesting a favorable balance between localization precision and reward stability.

## 5 Conclusion

We introduced Selective Regenerative Decoding (SRD), which routes candidate trajectories to keep, refine, or discard and selectively regenerates only degraded suffixes while preserving useful prefixes— achieving a provable $( 1 + \rho \cdot p _ { M } / p _ { H } ) \times$ sample efficiency gain over rejection sampling. Across four benchmarks, SRD consistently matches Bestof-N accuracy with fewer tokens and outperforms speculative rejection in low-compute regimes, with ablations showing that refinement effectiveness depends critically on reward model calibration. Key limitations include reliance on fixed routing thresholds and heuristic boundary selection; learning these end-to-end, as well as extending SRD to settings with adaptive or learned reward signals, are promising directions for future work.

## Limitations

SRD requires coordinating multiple components, including a generation model, a reward model, and an editing mechanism, which may increase implementation overhead and inference-time latency in resource-constrained settings.

Moreover, the effectiveness of SRD depends on the quality of the reward model used for routing. Inaccurate or misaligned reward estimates may lead to suboptimal acceptance or salvage decisions.

SRD currently relies on fixed routing thresholds and heuristic boundary selection. Learning these policies end-to-end is an interesting direction for future work.

## Ethical Considerations

Directed regeneration and bias amplification. Unlike Best-of-N, which only selects among samples the model would have produced anyway, SRD rewrites content to raise reward. This gives reward model biases more leverage: systematic preferences over length, style, or assertiveness are actively regenerated toward rather than merely selected for, and repeated refinement can compound the effect. We caution against deploying SRD with a reward model that has not been audited on the target distribution.

Reward models as the optimization target. SRD’s guarantees are stated with respect to the reward model R, not task correctness: weak monotonicity (Proposition C.8) ensures the best scored trajectory never degrades, which does not imply improved factual accuracy or safety. When R is misaligned with the intended objective, SRD faithfully optimizes the proxy, and the mechanism that recovers a correct suffix can equally entrench a confidently wrong one that the reward model happens to favor. Our GPQA results show this concretely: under a noisier process reward model, trusting global reward rankings is measurably worse than trusting only local comparisons (§4.4.2). We therefore report task-level metrics rather than reward scores throughout §4.

## Impact Statement

This paper presents Selective Regenerative Decoding (SRD), an inference-time decoding framework aimed at improving the efficiency and reliability of large language model reasoning. By enabling localized refinement within generated trajectories, SRD may help reduce redundant computation and improve performance in long-form reasoning tasks.

However, SRD requires coordinating multiple components, including a generation model, a reward model, and an editing mechanism, which may increase implementation complexity and inferencetime latency in resource-constrained settings.

Moreover, SRD relies on reward model feedback for routing decisions. Biases, miscalibration, or misalignment in reward estimates could lead to suboptimal refinements, which is particularly important to consider in high-stakes or safety-critical applications.

SRD currently uses fixed routing thresholds and heuristic boundary selection, and future work may explore more adaptive or learned routing policies. Overall, this work is intended as a general decoding-time contribution, and its broader impacts will depend on the downstream deployment context and the reliability of the reward signals used.

## References

David H Ackley, Geoffrey E Hinton, and Terrence J Sejnowski. 1985. A learning algorithm for boltzmann machines. Cognitive science, 9(1):147–169.

Amanda Bertsch, Alex Xie, Graham Neubig, and Matthew R Gormley. 2023. It’s mbr all the way down: Modern generation techniques through the lens of minimum bayes risk. arXiv preprint arXiv:2310.01387.

Thomas M Cover. 1999. Elements of information theory. John Wiley & Sons.

Herbert A David and Haikady N Nagaraja. 2004. Order statistics. John Wiley & Sons.

Angela Fan, Mike Lewis, and Yann Dauphin. 2018. Hierarchical neural story generation. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 889–898, Melbourne, Australia. Association for Computational Linguistics.

Markus Freitag and Yaser Al-Onaizan. 2017. Beam search strategies for neural machine translation. In Proceedings of the First Workshop on Neural Machine Translation, pages 56–60, Vancouver. Association for Computational Linguistics.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. 2021. Measuring mathematical problem solving with the math dataset. NeurIPS.

Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, and Yejin Choi. 2019. The curious case of neural text degeneration. arXiv preprint arXiv:1904.09751.

James Y. Huang, Sailik Sengupta, Daniele Bonadiman, Yi-An Lai, Arshit Gupta, Nikolaos Pappas, Saab Mansour, Katrin Kirchhoff, and Dan Roth. 2025. DeAL: Decoding-time alignment for large language models. In Proceedings of the 63rd Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 26280–26300, Vienna, Austria. Association for Computational Linguistics.

Maxim Khanov, Jirayu Burapacheep, and Yixuan Li. 2024. ARGS: Alignment as reward-guided search. In The Twelfth International Conference on Learning Representations.

Yaniv Leviathan, Matan Kalman, and Yossi Matias. 2023. Fast inference from transformers via speculative decoding. In International Conference on Machine Learning, pages 19274–19286. PMLR.

Xuechen Li, Tianyi Zhang, Yann Dubois, Rohan Taori, Ishaan Gulrajani, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. 2023. Alpacaeval: An automatic evaluator of instruction-following models. https://github.com/tatsu-lab/alpaca\_eval.

Baohao Liao, Yuhui Xu, Hanze Dong, Junnan Li, Christof Monz, Silvio Savarese, Doyen Sahoo, and Caiming Xiong. 2025. Reward-guided speculative decoding for efficient LLM reasoning. In Fortysecond International Conference on Machine Learning.

Ximing Lu, Sean Welleck, Peter West, Liwei Jiang, Jungo Kasai, Daniel Khashabi, Ronan Le Bras, Lianhui Qin, Youngjae Yu, Rowan Zellers, et al. 2022. Neurologic a\* esque decoding: Constrained text generation with lookahead heuristics. In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 780–799.

Sidharth Mudgal, Jong Lee, Harish Ganapathy, YaGuang Li, Tao Wang, Yanping Huang, Zhifeng Chen, Heng-Tze Cheng, Michael Collins, Jilin Chen, Alex Beutel, and Ahmad Beirami. 2023. Controlled decoding from language models. In Socially Responsible Language Modelling Research.

Reiichiro Nakano, Jacob Hilton, Suchir Balaji, Jeff Wu, Ouyang Long, Christina Kim, Christopher Hesse, Shantanu Jain, Vineet Kosaraju, William Saunders,

Xu Jiang, Karl Cobbe, Tyna Eloundou, Gretchen Krueger, Kevin Button, Matthew Knight, Benjamin Chess, and John Schulman. 2021. Webgpt: Browserassisted question-answering with human feedback. ArXiv, abs/2112.09332.

Nishanth Sridhar Nakshatri, Shamik Roy, Rajarshi Das, Suthee Chaidaroon, Leonid Boytsov, and Rashmi Gangadharaiah. 2025. Constrained decoding with speculative lookaheads. In Proceedings ofthe 2025 Conference of the Nations of the Americas Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 4681–4700.

Franz Josef Och, Nicola Ueffing, and Hermann Ney. 2001. An efficient a\* search algorithm for statistical machine translation. In Proceedings of the ACL 2001 Workshop on Data-Driven Methods in Machine Translation.

Arjun Panickssery, Samuel Bowman, and Shi Feng. 2024. Llm evaluators recognize and favor their own generations. Advances in Neural Information Processing Systems, 37:68772–68802.

David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R Bowman. 2024. Gpqa: A graduate-level google-proof q&a benchmark. In First Conference on Language Modeling.

Alfréd Rényi. 1953. On the theory of order statistics. Acta Math. Acad. Sci. Hung.

Shamik Roy, Sailik Sengupta, Daniele Bonadiman, Saab Mansour, and Arshit Gupta. 2024. FLAP: Flow-adhering planning with constrained decoding in LLMs. In Proceedings ofthe 2024 Conference of the North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 517–539, Mexico City, Mexico. Association for Computational Linguistics.

Nisan Stiennon, Long Ouyang, Jeff Wu, Daniel M. Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, and Paul Christiano. 2022. Learning to summarize from human feedback. Preprint, arXiv:2009.01325.

Yixuan Su, Tian Lan, Yan Wang, Dani Yogatama, Lingpeng Kong, and Nigel Collier. 2022. A contrastive framework for neural text generation. Advances in Neural Information Processing Systems, 35:21548– 21561.

Hanshi Sun, Momin Haider, Ruiqi Zhang, Huitao Yang, Jiahao Qiu, Ming Yin, Mengdi Wang, Peter Bartlett, and Andrea Zanette. 2024. Fast best-of-n decoding via speculative rejection. In The Thirty-eighth Annual Conference on Neural Information Processing Systems.

David Wan, Mengwen Liu, Kathleen McKeown, Markus Dreyer, and Mohit Bansal. 2023.

Faithfulness-aware decoding strategies for abstractive summarization. arXiv preprint arXiv:2303.03278.

Shufan Wang, Sebastien Jean, Sailik Sengupta, James Gung, Nikolaos Pappas, and Yi Zhang. 2023. Measuring and mitigating constraint violations of incontext learning for utterance-to-api semantic parsing. arXiv preprint arXiv:2305.15338.

Brandon T Willard and Rémi Louf. 2023. Efficient guided generation for large language models. arXiv preprint arXiv:2307.09702.

Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen, Ruslan Salakhutdinov, and Christopher D. Manning. 2018. HotpotQA: A dataset for diverse, explainable multi-hop question answering. In Conference on Empirical Methods in Natural Language Processing (EMNLP).

Shunyu Yao, Dian ${ \mathrm { Y u } } ,$ Jeffrey Zhao, Izhak Shafran, Tom Griffiths, Yuan Cao, and Karthik Narasimhan. 2023. Tree of thoughts: Deliberate problem solving with large language models. Advances in neural information processing systems, 36:11809–11822.

Algorithm 1 SELECTIVE REGENERATIVE DE-  
CODING   
Require: Generative model ${ \mathcal { G } } ,$ , reward model $R ,$   
sample size $n ,$ thresholds $\theta _ { \mathrm { l o w } } , \theta _ { \mathrm { h i g h } }$ , scoring   
interval $m ,$ max refinements $N _ { \mathrm { r e f i n e } } ,$ max span   
$L _ { \mathrm { m a x } }$   
Ensure: Set of accepted trajectories $\kappa$   
1: $\mathcal { C } \gets \{ \tau _ { i } \sim \mathcal { G } \} _ { i = 1 } ^ { n }$ $\triangleright$ Generate candidates   
2: $\mathcal { K }  \emptyset , \mathcal { R }  \emptyset$ ▷ Initialize kept and refine   
sets   
3: for $\tau \in { \mathcal { C } }$ do   
4: Compute $u ( \tau )$ based on rank in $\mathcal { C }$   
5: if $u ( \tau ) \geq \theta _ { \mathrm { h i g h } }$ then   
6: $\kappa  \kappa \cup \{ \tau \}$   
7: else if $u ( \tau ) > \theta _ { \mathrm { l o w } }$ then   
8: $\mathcal { R }  \mathcal { R } \cup \{ ( \tau , 0 ) \}$ ▷ Add with   
refinement count 0   
9: end if   
10: end for   
11: while $\mathcal { R } \neq \emptyset$ do   
12: $( \tau , c ) \gets \mathrm { P o p } ( \mathcal { R } )$   
13: if $c \geq N _ { \mathrm { r e f i n e } }$ then   
14: continue $\triangleright$ Max refinements reached   
15: end if   
16: $j ^ { * } \gets \mathrm { F I N D B O U N D A R Y } ( \tau , R , m )$   
17: if $k - j ^ { * } > L _ { \mathrm { m a x } }$ then   
18: $j ^ { * } \gets k - L _ { \operatorname* { m a x } }$ ▷ Limit regeneration   
span   
19: end if   
20: $\tau ^ { \prime } \  \ ( \tau _ { 1 : j ^ { * } } , \tilde { \tau } _ { j ^ { * } + 1 : k } )$ where $\tilde { \tau } _ { j ^ { * } + 1 : k } \ \sim$   
G<sub>high−temp</sub> $\left( \cdot \mid \tau _ { 1 : j ^ { * } } \right)$   
21: Compute $u ( \tau ^ { \prime } )$ based on rank in $\kappa \cup \mathcal { R } \cup$   
$\{ \tau ^ { \prime } \}$   
22: if $u ( \tau ^ { \prime } ) \geq \theta _ { \mathrm { h i g h } }$ then   
23: $\kappa  \kappa \cup \{ \tau ^ { \prime } \}$   
24: else if $u ( \tau ^ { \prime } ) > \theta _ { \mathrm { l o w } }$ then   
25: $\mathcal { R }  \mathcal { R } \cup \{ ( \tau ^ { \prime } , c + 1 ) \}$   
26: end if   
27: end while   
28: return $\kappa$

## A Experimental Settings (Extended)

This appendix expands the experimental settings summarized in §4.1.

## A.1 Baseline Descriptions

To ensure a fair comparison, unless otherwise specified, all methods share the same generation model, prompt template, and sampling hyperparameters (e.g., temperature), and differ only in their decoding and selection strategies.

Temperature Sampling (N = 1). This baseline generates a single trajectory under a fixed temperature setting and serves as the simplest sampling-based baseline. When the temperature is set to 0, this method reduces to strictly greedy decoding. In our experiments, we use the same non-zero temperature as other methods to ensure comparability.

Best-of-N (BoN). Best-of-N independently generates N full-length candidate sequences and selects the one with the highest reward score. While this approach increases candidate diversity and often improves final output quality, it incurs substantial generation cost, as all N candidates must be fully generated before selection.

Speculative Rejection (Spec-Rej). Speculative Rejection evaluates candidate prefixes during generation using a reward model and terminates low-reward trajectories early, thereby reducing generation cost compared to Best-of-N. This method adopts a binary accept-or-reject strategy: once a trajectory is rejected, it is permanently discarded and no longer extended.

## B Component Breakdown and Runtime Analysis

To better understand the computational characteristics of SRD, we provide a component-level breakdown on MATH500 under different values of N (Table 2). We report task performance, runtime, and percomponent time and call frequency. The four components correspond to the phases in §3.2: the drafter is the generative model G that produces candidate trajectories (Phase 1); the scorer is the reward model R used for both routing decisions and regeneration boundary detection (Phases 2–3); the router applies the threshold-based routing logic (Phase 2); and the editor performs suffix regeneration for candidates routed to REFINE (Phase 3).

Overall trends. As N increases, accuracy consistently improves, while the overall runtime grows moderately. Importantly, the number of component calls remains largely stable across different N, indicating that SRD does not incur additional structural overhead as candidate exploration increases. Instead, the increased cost mainly comes from processing longer trajectories rather than invoking components more frequently.

Cost distribution across components. The runtime is dominated by the scorer, followed by the editor, while the router contributes negligible overhead. This suggests that SRD’s computational cost is primarily driven by reward evaluation and localized rewriting, whereas routing decisions themselves are lightweight. Such a cost profile is desirable in practice, as it enables selective intervention without introducing expensive control logic.

Selective editing behavior. Even with larger N, the editor is invoked only a small number of times on average, demonstrating that SRD performs localized edits selectively rather than rewriting entire trajectories. This behavior distinguishes SRD from naive Best-of-N approaches and helps explain its favorable accuracy–cost trade-off.

## C Proofs

This appendix provides complete proofs for all theoretical results presented in Section 3.3. We begin by clarifying the population-level interpretation of the routing probabilities, then establish several auxiliary lemmas before proving the main theorems.

## C.1 Population-Level Interpretation of $p _ { H } , p _ { M } , p _ { L }$

Remark C.1. The quantities $p _ { H } , p _ { M }$ , and $p _ { L }$ defined in Equations (6)–(8) should be interpreted as probabilities under the population distribution – i.e., the probability that the reward model’s score for a random trajectory from ${ \mathcal { G } } ,$ when compared against the population of all possible outputs, falls in each tier. The algorithm’s rank-based scoring $u ( \tau ) = 1 - r ( \tau ) / ( n - 1 )$ is a finite-sample approximation to this population quantity. For large n, the empirical rank converges to the population quantile by the Glivenko-Cantelli theorem. The theoretical analysis operates in the population limit; the algorithm provides a consistent estimator.

## C.2 Preliminary Lemmas

Lemma C.2 (Geometric Failure Probability). Let $X _ { 1 } , X _ { 2 } , \ldots , X _ { N }$ be independent Bernoulli random variables with $\operatorname* { P r } [ X _ { i } = 1 ] = p f o r$ some $p \in ( 0 , 1 )$ ). Then:

$$
\operatorname* { P r } \left[ \sum _ { i = 1 } ^ { N } X _ { i } = 0 \right] = ( 1 - p ) ^ { N } .\tag{14}
$$

Furthermore, to ensure $\begin{array} { r } { \operatorname* { P r } [ \sum _ { i = 1 } ^ { N } X _ { i } \geq 1 ] \geq 1 - \delta } \end{array}$ for some $\delta \in ( 0 , 1 )$ , it is necessary and sufficient that:

$$
\begin{array} { c } { { N \geq \displaystyle \frac { \ln ( 1 / \delta ) } { - \ln ( 1 - p ) } } } \\ { { \geq \displaystyle \frac { \ln ( 1 / \delta ) } { p } . } } \end{array}\tag{15}
$$

Proof. The first claim follows directly from independence:

$$
\operatorname* { P r } \left[ \sum _ { i = 1 } ^ { N } X _ { i } = 0 \right] = \prod _ { i = 1 } ^ { N } \operatorname* { P r } [ X _ { i } = 0 ] = \prod _ { i = 1 } ^ { N } ( 1 - p ) = ( 1 - p ) ^ { N } .
$$

For the second claim, we require $\begin{array} { r } { ( 1 - p ) ^ { N } \leq \delta . } \end{array}$ . Taking logarithms:

$$
N \ln ( 1 - p ) \leq \ln ( \delta ) ,
$$

and since ln $( 1 - p ) < 0$ for $p \in ( 0 , 1 )$

$$
N \geq { \frac { \ln ( 1 / \delta ) } { - \ln ( 1 - p ) } }
$$

Inequality $- \ln ( 1 - p ) \leq p$ follows from the concavity of the logarithm: for $p \in [ 0 , 1 )$ , we have $\ln ( 1 - p ) \geq - p / ( 1 - p ) \geq - p$ when $p \leq 1 / 2$ , and direct computation confirms the bound for $p > 1 / 2$ Alternatively, this follows from the Taylor expansion $- \ln ( 1 - p ) = p + p ^ { 2 } / 2 + p ^ { 3 } / 3 + \cdot \cdot \cdot \geq p .$ . Thus:

$$
\begin{array} { c } { { N \geq \displaystyle \frac { \ln ( 1 / \delta ) } { - \ln ( 1 - p ) } } } \\ { { \geq \displaystyle \frac { \ln ( 1 / \delta ) } { p } . } } \end{array}
$$

Lemma C.3 (Effective Acceptance Probability). Under Assumptions 3.1 and 3.2, the probability that a single trajectory sampledfrom G is eventually accepted by SRD (either directly or after Regeneration) is at least:

$$
p _ { \mathrm { e f f } } = p _ { H } + \rho \cdot p _ { M } ,\tag{16}
$$

where $p _ { H } , p _ { M }$ , and ρ are defined in Equations (6)–(7) and Assumption 3.1, respectively.

Proof. A trajectory $\tau \sim \mathcal { G }$ is eventually accepted if:

(a) $u ( \tau ) \geq \theta _ { \mathrm { h i g h } } , \mathrm { i . e . }$ , it is directly routed to Keep. This occurs with probability $p _ { H }$ by $\operatorname { E q . } ( 6 )$

(b) $\theta _ { \mathrm { l o w } } < u ( \tau ) < \theta _ { \mathrm { h i g h } }$ routed to Refine, probability $p _ { M }$ by Eq.(7)), AND the refinement procedure produces an acceptable trajectory (probability $\geq \rho$ by Assumption3.1).

Events (a) and (b) are mutually exclusive since the regions $\{ u \ge \theta _ { \mathrm { h i g h } } \}$ and $\{ \theta _ { \mathrm { l o w } } < u < \theta _ { \mathrm { h i g h } } \}$ are disjoint. By Assumption 3.2 (independence), the refinement outcome in case (b) is independent of the initial routing. Therefore, by the law of total probability:

$$
p _ { \mathrm { a c c e p t e d } } \geq p _ { H } + p _ { M } \cdot \rho = p _ { \mathrm { e f f } } .
$$

Note this is a lower bound: with $N _ { \mathrm { r e f i n e } } > 1$ attempts, the effective probability is $p _ { H } + p _ { M } \cdot ( 1 - ( 1 -$ $\rho ) ^ { N _ { \mathrm { r e f i n e } } } ) \ge p _ { H } + \rho \cdot p _ { M }$

Lemma C.4 (Stochastic Dominance of Maxima). Let X and $Z$ be independent random variables and define $Y = \operatorname* { m a x } \{ X , Z \}$ . Then $Y$ stochastically dominates $X .$

$$
Y \succeq _ { \mathrm { s t } } X ,\tag{17}
$$

with strict dominance $( i . e . , \mathbb { E } [ Y ] > \mathbb { E } [ X ] )$ whenever $\operatorname* { P r } [ Z > X ] > 0 .$

Lemma C.5 (Order Statistic Gap). Let $X _ { 1 } , \ldots , X _ { n }$ be i.i.d. random variables with continuous distribution $F$ having density $f$ bounded away from zero near the right endpoint of its support. Let $X _ { ( n ) }$ and $X _ { ( n - 1 ) }$ denote the largest and second-largest order statistics. Then:

$$
\mathbb { E } [ X _ { ( n ) } - X _ { ( n - 1 ) } ] = \Theta \left( { \frac { 1 } { n } } \right) .\tag{18}
$$

Proof. This is a classical result in order statistics. By the Rényi representation theorem (Rényi, 1953), if $U _ { ( 1 ) } \leq \cdots \leq U _ { ( n ) }$ are order statistics of n uniform random variables on [0, 1], then:

$$
( U _ { ( 1 ) } , \ldots , U _ { ( n ) } ) \stackrel { d } { = } \left( \frac { S _ { 1 } } { S _ { n + 1 } } , \frac { S _ { 2 } } { S _ { n + 1 } } , \ldots , \frac { S _ { n } } { S _ { n + 1 } } \right) ,
$$

where $S _ { k } = E _ { 1 } + \cdot \cdot \cdot + E _ { k }$ and $E _ { 1 } , \ldots , E _ { n + 1 }$ are i.i.d. Exponential(1)/standard-exponential random variables. For the uniform distribution:

$$
\mathbb { E } [ U _ { ( n ) } - U _ { ( n - 1 ) } ] = \mathbb { E } \left[ { \frac { E _ { n } } { S _ { n + 1 } } } \right] = { \frac { 1 } { n + 1 } } ,
$$

using the fact that $\mathbb { E } [ E _ { i } / S _ { n + 1 } ] = 1 / ( n + 1 )$ by symmetry.

For a general distribution F with density f, we use the probability integral transform: $X _ { ( k ) } ~ =$ $F ^ { - 1 } ( U _ { ( k ) } )$ . By a Taylor expansion near the right endpoint:

$$
\mathbb { E } [ X _ { ( n ) } - X _ { ( n - 1 ) } ] \approx \frac { 1 } { f ( F ^ { - 1 } ( 1 - ) ) } \cdot \mathbb { E } [ U _ { ( n ) } - U _ { ( n - 1 ) } ] = \Theta \left( \frac { 1 } { n } \right) ,
$$

provided f is bounded away from zero. Please see David and Nagaraja (2004) for a complete treatment. □

## C.3 Proof of Theorem 3.3 (Sample Efficiency)

Proof. We prove each statement in the theorem separately.

Part (i): Lower bound for rejection sampling. In pure rejection sampling, a trajectory $\tau$ is accepted if and only if $u ( \tau ) \geq \theta _ { \mathrm { h i g h } }$ . By definition, this occurs with probability $p _ { H }$

Let $X _ { i } = \mathbf { 1 } [ u ( \tau _ { i } ) \geq \theta _ { \mathrm { h i g h } } ]$ for $i = 1 , \ldots , N$ , where $\tau _ { i } \sim \mathcal { G }$ are i.i.d. samples. Then $\{ X _ { i } \}$ are i.i.d. Bernoulli $( p _ { H } )$ random variables.

The probability of obtaining at least one accepted trajectory is:

$$
\operatorname* { P r } \left[ \sum _ { i = 1 } ^ { N } X _ { i } \geq 1 \right] = 1 - \operatorname* { P r } \left[ \sum _ { i = 1 } ^ { N } X _ { i } = 0 \right] = 1 - ( 1 - p _ { H } ) ^ { N } .
$$

To ensure this probability is at least $1 - \delta .$ , we require:

$$
1 - ( 1 - p _ { H } ) ^ { N } \geq 1 - \delta \quad \Longleftrightarrow \quad ( 1 - p _ { H } ) ^ { N } \leq \delta .
$$

By Lemma C.2, this requires:

$$
N \geq \frac { \ln ( 1 / \delta ) } { p _ { H } } .
$$

Part (ii): Lower bound for SRD. By Lemma C.3, the effective acceptance probability under SRD is at least $p _ { \mathrm { e f f } } = p _ { H } + \rho \cdot p _ { M }$ . Following the same argument as Part (i), but with $p _ { H }$ replaced by $p _ { \mathrm { e f f } } .$

$$
\mathrm { P r } [ \mathrm { a t } \mathrm { l e a s t } \mathrm { o n e } \mathrm { a c c e p t a n c e } ] = 1 - ( 1 - p _ { \mathrm { e f f } } ) ^ { N } \geq 1 - \delta
$$

requires:

$$
N \ge \frac { \ln ( 1 / \delta ) } { p _ { \mathrm { e f f } } } = \frac { \ln ( 1 / \delta ) } { p _ { H } + \rho \cdot p _ { M } } .
$$

Part (iii): Efficiency ratio. Given the two above statements, we can simply take a ratio of the statements to show the efficiency gain of SRD over Rejection Sampling:

$$
\frac { N _ { \mathrm { r e j e c t } } } { N _ { \mathrm { r e f i n e } } } = \frac { \ln ( 1 / \delta ) / p _ { H } } { \ln ( 1 / \delta ) / ( p _ { H } + \rho \cdot p _ { M } ) } = \frac { p _ { H } + \rho \cdot p _ { M } } { p _ { H } } = 1 + \frac { \rho \cdot p _ { M } } { p _ { H } } .\tag{19}
$$

□

Remark C.6 (Tightness of the Bound). The bounds in Theorem 3.3 are tight up to constant factors. Precisely, we can use Fano’s inequality (Cover, 1999) in information theory to show that any algorithm that achieves a success probability of $1 - \delta$ must collect sufficient information to distinguish between the hypothesis that at least one good trajectory exists and no good trajectory exists, and this requires $\Omega ( \ln ( 1 / \delta ) / p _ { \mathrm { e f f } } )$ samples. Hence, combining the bounds we have:

(i) Rejection sampling is optimal among algorithms that treat each sample as either accept or reject with no intermediate processing.

$$
\frac { c _ { 1 } \ln ( 1 / \delta ) } { p _ { H } } \leq N _ { r e j e c t } \leq \frac { c _ { 2 } \ln ( 1 / \delta ) } { p _ { H } }\tag{20}
$$

(ii) SRD is optimal among algorithms that can additionally salvage borderline samples with success probability $\rho .$

$$
\frac { c _ { 1 } ^ { \prime } \ln ( 1 / \delta ) } { p _ { \mathrm { e f f } } } \leq N _ { r e f i n e } \leq \frac { c _ { 2 } ^ { \prime } \ln ( 1 / \delta ) } { p _ { \mathrm { e f f } } }\tag{21}
$$

(iii) The efficiency gain in Equation 19 is real and guaranteed since any algorithm with effective acceptance probability of $p _ { \mathrm { e f f } }$ will have a sample complexity of $\Theta ( l n ( 1 / \delta ) / p _ { \mathrm { e f f } } )$ , and increasing $p _ { \mathrm { e f f } }$ by salvaging and regeneration will reduce the samples.

Corollary The sample efficiency of Speculative Selective Regenerative Decoding (S-SRD) is higher than that of Reward-guided Speculative Decoding (RSD) (Liao et al., 2025). Assuming a larger target model is more accurate on the reasoning problem at hand, we can guarantee that $p _ { \mathrm { e f f } } \ge p _ { H }$ as any salvaged trajectory (provided the draft model can generate this with non-zero probability) regenerated using a more-accurate target model will be accepted.

## C.4 Proof of Theorem 3.5 (Expected Quality Improvement)

Proof. The proof proceeds in three steps: (1) constructing a coupling, (2) establishing stochastic dominance, and (3) quantifying the improvement.

Step 1: Coupling construction. We construct a coupling between rejection sampling and SRD as follows. Both algorithms receive the same sequence of n initial trajectories $\tau _ { 1 } , . . . , \tau _ { n } \sim \mathcal { G }$

Now, let A denote the set of accepted trajectories. For rejection sampling, we have:

$$
\begin{array} { r } { \mathcal { A } _ { \mathrm { r e j e c t } } = \{ \tau _ { i } : u ( \tau _ { i } ) \geq \theta _ { \mathrm { h i g h } } \} , } \end{array}
$$

For SRD, the accepted set is:

$$
{ \mathcal { A } } _ { \mathrm { r e g e n e r a t e } } = { \mathcal { A } } _ { \mathrm { r e j e c t } } \cup { \mathcal { A } } _ { \mathrm { r e g e n e r a t e / n c c e p t } } ,
$$

where $\scriptstyle { \mathcal { A } } _ { \mathrm { r e g e n e r a t e d } }$ contains trajectories that were regenerated and subsequently accepted. Under this coupling, $\mathcal { A } _ { \mathrm { { r e j e c t } } } \subseteq \mathcal { A } _ { \mathrm { { r e g e n e r a t e } } }$ by construction.

Step 2: Stochastic dominance. Define:

$$
\begin{array} { l } { { X = R _ { \mathrm { m a x } } ^ { \mathrm { R S } } ( n ) = \operatorname* { m a x } _ { \tau \in \mathcal { A } _ { \mathrm { R S } } } R ( \tau ) , } } \\ { { Z = \operatorname* { m a x } _ { \tau \in \mathcal { A } _ { \mathrm { r e g e n e r a t e / n a c c e p t } } } R ( \tau ) . } } \end{array}
$$

By construction:

$$
R _ { \operatorname* { m a x } } ^ { \mathrm { S R D } } ( n ) = \operatorname* { m a x } \{ X , Z \} .
$$

By Lemma C.4, max $\{ X , Z \} \succeq _ { \mathrm { s t } } X$ . Moreover, strict dominance holds because:

$$
\operatorname* { P r } [ Z > X ] \ge \operatorname* { P r } [ \mathcal { A } _ { \mathrm { r e g e n e r a t e / n a c c e p t } } \neq \emptyset ] \cdot \operatorname* { P r } [ Z > X \mid \mathcal { A } _ { \mathrm { r e g e n e r a t e / n a c c e p t } } \neq \emptyset ] > 0 .
$$

The first factor is at least $1 - ( 1 - p _ { M } \cdot \rho ) ^ { n } > 0$ by Assumption 3.1. The second factor is positive under Assumption 3.4, which ensures regenerated trajectories have non-trivial probability of achieving high rewards. Therefore, $\mathbb { E } [ R _ { \operatorname* { m a x } } ^ { \mathrm { S R D } } ( n ) ] > \mathbb { E } [ R _ { \operatorname* { m a x } } ^ { \mathrm { R S } } ( n ) ]$ ], establishing $\Delta _ { n } > 0$

Step 3: Quantifying $\Delta _ { n } .$ . To derive the explicit bound, we analyze the probability that a regenerated trajectory achieves the maximum reward. For this, let $N _ { R S } = | \mathcal { A } _ { \mathrm { R S } } |$ be the number of directly accepted trajectories and $N _ { R + A } = | A _ { \mathrm { r e g e n e r a t e / l a c c e p t } } |$ be the number of successfully regenerated trajectories. By linearity of expectation:

$$
\begin{array} { l } { { \mathbb E } [ N _ { R S } ] = n \cdot p _ { H } , } \\ { { \mathbb E } [ N _ { R + S } ] \geq n \cdot p _ { M } \cdot \rho . } \end{array}
$$

Under Assumption 3.4, the rewards of trajectories in $\mathcal { A } _ { \mathrm { S R D } }$ are exchangeable, the original accepted trajectories in $( \in { \mathcal { A } } _ { R S } )$ and regenerated trajectories that are eventually accepted $( \in \mathcal { A } _ { r e g e n e r a t e \cap a c c e p t } )$ are i.i.d as per the reward function. Now, by symmetry, each trajectory in A<sub>SRD</sub> has probability approximately $\frac { 1 } { | \mathcal { A } _ { \mathrm { S R D } } | }$ of being the maximum.

The probability that the maximum comes from $\mathcal { A } _ { \mathrm { { r e g e n e r a t e } \cap { a c c e p t } } }$ rather than A<sub>RS</sub> is approximately:

$$
\operatorname* { P r } [ \operatorname* { m a x } { \mathrm { ~ f r o m ~ r e g e n e r a t e d } } ] \approx { \frac { \mathbb { E } [ N _ { R + S } ] } { \mathbb { E } [ N _ { H } ] + \mathbb { E } [ N _ { R + S } ] } } = { \frac { n \cdot p _ { M } \cdot \rho } { n \cdot p _ { H } + n \cdot p _ { M } \cdot \rho } } = { \frac { p _ { M } \cdot \rho } { p _ { H } + p _ { M } \cdot \rho } } .\tag{22}
$$

When a regenerated trajectory achieves the maximum, the improvement over the rejection sampling maximum is, by Lemma C.5:

$$
\mathbb { E } [ R _ { \mathrm { m a x } } ^ { \mathrm { S R D } } ( n ) - R _ { \mathrm { m a x } } ^ { \mathrm { R S } } ( n ) \ | \ \mathrm { m a x } \ \mathrm { f r o m } \ \mathrm { r e g e n e r a t e d } ] = \Theta \left( \frac { 1 } { n } \right)\tag{23}
$$

Finally, Combining Equation 22 and Equation 23, we have

$$
\begin{array} { r l } & { \Delta _ { n } = \mathbb { E } [ R _ { \mathrm { m a x } } ^ { \mathrm { S R D } } ( n ) ] - \mathbb { E } [ R _ { \mathrm { m a x } } ^ { \mathrm { R S } } ( n ) ] } \\ & { \quad \ge \operatorname* { P r } [ \mathrm { m a x } \mathrm { ~ f r o m ~ r e g e n e r a t e d } ] \cdot \mathbb { E } [ \mathrm { g a p ~ } | \mathrm { ~ m a x ~ f r o m ~ r e g e n e r a t e d } ] } \\ & { \quad = \frac { p _ { M } \cdot \rho } { p _ { H } + p _ { M } \cdot \rho } \cdot \Theta \left( \frac { 1 } { n } \right) } \\ & { \quad = \Omega \left( \frac { p _ { M } \cdot \rho } { n \cdot ( p _ { H } + p _ { M } \cdot \rho ) } \right) } \\ & { \quad = \Omega \left( \frac { p _ { M } \cdot \rho } { n \cdot ( 1 + p _ { M } \cdot \rho ) } \right) } \end{array}
$$

where the last line uses $p _ { H } \leq 1$ . This completes the proof.

## C.5 Termination

Proposition C.7 (Termination). SRD terminates in at most $n \cdot N _ { \mathrm { r e f i n e } }$ refinement operations and $O ( n$ $N _ { \mathrm { r e f i n e } } )$ reward evaluations.

Proof. We show that SRD terminates in finite time by bounding the number of Regeneration operations.

Each of the n initial trajectories can be regenerated at most $N _ { \mathrm { r e f i n e } }$ times (by the counter c in Algorithm 1). Once a trajectory has been regenerated $N _ { \mathrm { r e f i n e } }$ times, it is never added back to the Regeneration queue R.

Furthermore, each Regeneration operation produces at most one new trajectory, which inherits the Regeneration count of its parent (incremented by one). Thus, the total number of trajectories ever created is at most:

$$
n + n \cdot N _ { \mathrm { r e f i n e } } = n ( 1 + N _ { \mathrm { r e f i n e } } ) .
$$

Since the Regeneration queue R starts with at most n trajectories and each trajectory exits the queue after at most $N _ { \mathrm { r e f i n e } }$ Regeneration attempts, the while loop in Algorithm 1 executes at most $n \cdot N _ { \mathrm { r e f i n e } }$ iterations.

Each iteration involves:

1. Finding the regeneration boundary: $O ( k / m )$ reward evaluations.

2. Generating the regenerated suffix: $O ( 1 )$ call to the generative model.

3. Computing the rank of the regenerated trajectory: $O ( | K | + | \mathcal { R } | )$ comparisons.

The total number of reward evaluations is thus:

$$
O \left( n \cdot N _ { \mathrm { r e f i n e } } \cdot \frac { k } { m } \right) = O ( n \cdot N _ { \mathrm { r e f i n e } } ) ,
$$

since $k / m$ is a constant based on predefined hyperparameter choice.

## C.6 Weak Monotonicity

Proposition C.8 (Weak Monotonicity). Let $\kappa _ { t }$ denote the set of kept trajectories after t refinement operations, and let $\begin{array} { r } { R _ { t } ^ { * } = \operatorname* { m a x } _ { \tau \in \mathcal { K } _ { t } } R ( \tau ) } \end{array}$ (with $R _ { t } ^ { * } = - \infty i f \mathcal { K } _ { t } = \emptyset )$ . Then the sequence $\{ R _ { t } ^ { * } \} _ { t \ge 0 }$ is non-decreasing.

Proof. Let $\textstyle { \boldsymbol { \mathcal { K } } } _ { t }$ denote the set of kept trajectories after t Regeneration operations, and define $R _ { t } ^ { * } ~ =$ $\mathrm { m a x } _ { \tau \in \mathcal { K } _ { t } } R ( \tau )$ (with $R _ { t } ^ { * } = - \infty \operatorname { i f } \mathcal { K } _ { t } = \varnothing )$ . We show that $R _ { t + 1 } ^ { * } \geq R _ { t } ^ { * }$ for all $t \geq 0$

Consider the $( t + 1 )$ -th regeneration operation. Let τ be the trajectory being regenerated, and let $\tau ^ { \prime }$ be the resulting regenerated trajectory. We have 3 cases:

Case $\mathbf { 1 } \colon \ \tau ^ { \prime }$ is routed to KEEP. Then $K _ { t + 1 } ~ = ~ \mathcal { K } _ { t } \cup \{ \tau ^ { \prime } \}$ , so $\begin{array} { r } { R _ { t + 1 } ^ { * } \ = \ \operatorname* { m a x } _ { \tau ^ { \prime \prime } \in \mathcal { K } _ { t + 1 } } \ R ( \tau ^ { \prime \prime } ) \ = \ } \end{array}$ max $\{ R _ { t } ^ { * } , R ( \tau ^ { \prime } ) \} \geq R _ { t } ^ { * }$

Case $2 \colon \tau ^ { \prime }$ is routed to REGENERATE. Then $K _ { t + 1 } = K _ { t } , \mathrm { s o } R _ { t + 1 } ^ { * } = R _ { t } ^ { * }$

Case $3 \colon \tau ^ { \prime }$ is routed to DISCARD. Then ${ \boldsymbol { \mathcal { K } } } _ { t + 1 } = { \boldsymbol { \mathcal { K } } } _ { t }$ , so $R _ { t + 1 } ^ { * } = R _ { t } ^ { * }$

In all cases, $R _ { t + 1 } ^ { * } \geq R _ { t } ^ { * }$ , establishing weak monotonicity.

## D Prompt Templates

This appendix provides the prompt templates used to generate model responses for different benchmarks.   
All prompts are used exclusively for generation and are shared across all decoding methods.

## D.1 HotpotQA

HotpotQA Generation Prompt   
Answer the following question:   
{question}   
Bear in mind that your response should be strictly based on the following passages:   
{passages}   
Your task:   
1. Carefully reason step by step, using only the information from the retrieved documents.   
2. Explicitly cite the supporting evidence by referencing the corresponding document IDs and   
sentence indices.   
3. Ensure that each reasoning step is verifiable from the given documents.   
4. After reasoning, provide the final answer clearly in the format:   
your final answer here

## D.2 MATH500

## MATH500 Generation Prompt

Please reason step by step, and put your final answer within \boxed{}.   
{question}

## D.3 GPQA Diamond

GPQA Diamond Generation Prompt

Please reason step by step, and choose the correct answer from A/B/C/D inside \boxed{}. {question}

AlpacaFarm. For AlpacaFarm, we directly use the original user instruction as the generation input, without any additional task-specific prompt templates or formatting.

## E Implementation Details

This appendix summarizes the hyperparameter settings and implementation details used in our experiments. Unless otherwise specified, all methods share the same generation model, prompt template, and generationside hyperparameters.

Generation Settings. We use temperature-based sampling for all methods, with temperature set to 0.8, top-p set to 0.9, and top-k set to 50. The maximum generation length is set to 500 tokens for MATH500 and HotpotQA, and 1000 tokens for GPQA and AlpacaFarm.

SRD Parameters. Selective Regenerative Decoding uses fixed routing thresholds $\theta _ { \mathrm { l o w } } = 0 . 3$ and $\theta _ { \mathrm { h i g h } } = 0 . 5$ . The drafting interval is set to 100 steps, matching the configuration used for Speculative Rejection. The reward model is invoked every 10 decoding steps. We cap the maximum number of regeneration steps at 30 and enforce a minimum of one active candidate to prevent premature termination.

Speculative Rejection Parameters. For Speculative Rejection, we follow the original paper’s recommended setting, using a rejection threshold $\alpha = 0 . 5$ . All other generation and evaluation settings are identical to those used for SRD and Best-of-N.

Reward Model Usage. Reward models are queried with a fixed input format that concatenates the task instruction, input context, and the partially or fully generated output. For methods that perform step-level scoring, including SRD and Speculative Rejection, the most recent reward score is used for routing decisions. No additional reward normalization is applied. For Best-of-N, reward scores are computed on the final completed sequences and used for candidate selection.

Hardware. Experiments are run on 8×A100 (40GB) GPUs. Generation uses vLLM, and reward models are executed with HuggingFace Transformers.

## F Empirical Validation of Theorem 3.3

To validate the assumptions underlying Theorem 3.3, we directly measure the routing distribution $( p _ { H }$ $p _ { M } , p _ { L } )$ and refinement efficacy $( \hat { \rho } )$ from SRD runs. For each configuration, we record how candidates are routed after initial scoring and track the outcome of each refinement attempt (i.e., whether a trajectory routed to REFINE is subsequently promoted to KEEP).
<table><tr><td>N</td><td>PH</td><td>pM</td><td> $p _ { L }$ </td><td> $\hat { \rho }$ </td><td>Efficiency Gain</td></tr><tr><td>10</td><td>0.522</td><td>0.148</td><td>0.329</td><td>1.00 (100%)</td><td>1.28×</td></tr><tr><td>20</td><td>0.513</td><td>0.180</td><td>0.305</td><td>1.00 (100%)</td><td>1.35×</td></tr><tr><td>30</td><td>0.509</td><td>0.182</td><td>0.308</td><td>1.00 (100%)</td><td>1.36×</td></tr></table>

Table 6: Empirical routing distributions and refinement efficacy (ρˆ) on MATH500 (Qwen2.5-Math-1.5B, AceMath-7B-RM, 50 samples per N). The efficiency gain is computed as $1 + \hat { \rho } \cdot p _ { M } / p _ { H }$ per Theorem 3.3.

The measured $\hat { \rho } = 1 . 0$ across all values of N shows that all salvage attempts were routed to keep, providing strong empirical support for Assumption 3.1. The efficiency gain increases with $N \left( 1 . 2 8 \times \right)$ 1.36×) because $p _ { M }$ grows slightly while $p _ { H }$ remains stable, indicating that larger candidate pools produce more mid-tier trajectories amenable to refinement. The routing distribution is consistent across N, with $p _ { H } \approx 0 . 5 1 \substack { - 0 . 5 2 , p _ { M } \approx 0 . 1 5 - 0 . 1 8 }$ , and $p _ { L } \approx 0 . 3 1 \substack { - 0 . 3 3 }$ . We note that while these measurements directly validate sample efficiency of SRD highlighted in Theorem 3.3, comparing the reward distributions between SRD and rejection sampling to verify Theorem 3.5 is considered in the next section.

## G Empirical Validation of Theorem 3.5

To validate Theorem 3.5, we compare the maximum reward $R _ { \mathrm { m a x } }$ achieved by SRD and rejection sampling (RS) under a coupled setting on MATH500 with Qwen2.5-Math-1.5B-Instruct and AceMath-7B-RM.

Coupling via Seeded Generation. The proof of Theorem 3.5 relies on a coupling argument where both SRD and RS process the same initial trajectories. To realize this coupling empirically, we pass a deterministic per-sample seed to vLLM’s SamplingParams, ensuring that both methods receive identical initial drafts and identical continuations for shared candidates. This guarantees the superset property: every candidate RS keeps, SRD also keeps, plus SRD additionally refines mid-tier candidates.

RS Baseline. The RS baseline uses the same incremental generation loop as SRD—generating N partial drafts, scoring at intermediate checkpoints, and keeping candidates above $\theta _ { \mathrm { h i g h } }$ —but without refinement. This corresponds to the spec\_rej decoder mode (SRDDecoder with editor=None, $\theta _ { \mathrm { l o w } } = \theta _ { \mathrm { h i g h } } = 0 . 5 )$ matching the rejection sampling defined in §3.3.

Results. As shown in Table 7, SRD achieves higher mean $R _ { \mathrm { m a x } }$ than RS across all tested values of N, consistent with Theorem 3.5’s prediction that $\Delta _ { n } > 0$ . SRD wins more samples than RS at every N (46% vs. 18% at N=10; 54% vs. 24% at N=20; 52% vs. 24% at N=30). The ties correspond to cases where both methods’ best candidate is the same high-scoring trajectory that both keep (these are the percentage of high trajectories that needed no refinement).

<table><tr><td>N</td><td>SRD  $\bar { R } _ { \operatorname* { m a x } }$ </td><td> $\mathrm { R S } \ \bar { R } _ { \mathrm { m a x } }$ </td><td> $\Delta _ { n }$ </td><td>SRD wins RS wins</td><td></td><td>Ties</td><td>Refine improved</td><td>1 Pre → Post</td></tr><tr><td>10</td><td>17.70</td><td>17.34</td><td> $+ 0 . 3 6 \pm 1 . 7 0$ </td><td>46%</td><td>18%</td><td>36%</td><td>78%</td><td> $2 . 8 5  5 . 9 6$ </td></tr><tr><td>20</td><td>18.75</td><td>18.47</td><td> $+ 0 . 2 8 \pm 4 . 1 1$ </td><td>54%</td><td>24%</td><td>22%</td><td>98%</td><td> $3 . 4 1  6 . 7 9$ </td></tr><tr><td>30</td><td>19.96</td><td>18.66</td><td> $+ 1 . 2 9 \pm 3 . 4 1$ </td><td>52%</td><td>24%</td><td>24%</td><td>100%</td><td> $3 . 6 9 \to 6 . 8 7$ </td></tr></table>

Table 7: Theorem 3.5 validation on MATH500 (Qwen2.5-Math-1.5B, AceMath-7B-RM, 50 samples per N). Both methods start from identical initial drafts via seeded generation. $\Delta _ { n } = \bar { R } _ { \operatorname* { m a x } } ^ { \mathrm { S R D } } - \bar { R } _ { \operatorname* { m a x } } ^ { \mathrm { R S } }$ (mean ± std). “Pre → Post” reports mean reward before and after refinement.

Notably, the SRD advantage grows with $N \colon \Delta _ { n }$ increases from +0.36 at N=10 to +1.29 at N=30. This is because larger candidate pools produce more mid-tier trajectories amenable to refinement, and the refined candidates increasingly contribute to $R _ { \mathrm { m a x } }$ . Refinement success rates confirm this trend (78% → 98% → 100%), with substantial reward improvements across all settings (e.g., 3.69 → 6.87 at N=30), providing direct empirical support for Assumption 3.4.