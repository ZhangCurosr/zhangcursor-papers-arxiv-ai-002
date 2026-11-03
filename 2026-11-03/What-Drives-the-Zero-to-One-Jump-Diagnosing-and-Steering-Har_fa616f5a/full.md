# What Drives the Zero-to-One Jump? Diagnosing and Steering Hard-Prompt Learning in RLVR

Rui Zhu and Yumou Liu and Zhenzhe Zheng\* and Fan Wu

Shanghai Jiao Tong University

Shanghai, China

{jameszhu, liuyumou, zhengzhenzhe, wu-fan}@sjtu.edu.cn

## Abstract

Reinforcement Learning with Verifiable Rewards (RLVR) has substantially improved mathematical reasoning, yet progress on hard prompts often stalls: a prompt can remain unsolved for many training steps before abruptly transitioning to solved—a discrete “zero-to-one” jump whose underlying dynamics and steerability remain poorly understood. We investigate two questions: what drives this transition, and can it be steered? Through event-based analysis measuring the time-tofirst-positive (TTFP), we find that negative rollouts preceding successful transitions consistently retain higher early-stage generation entropy, distinguishing near-solvable failures from dead ends. This signal motivates Negative-Augmented Policy Optimization (NAPO), a method integrated with Group Relative Policy Optimization (GRPO) that ranks negatives in unsolved groups via an early-branching driver and assigns graded advantages, injecting informative gradients precisely where standard group-relative updates vanish. Under fixed rollout budgets, NAPO accelerates early solution discovery and improves Pass@256 by 10.0 points on AIME24 and 6.6 points on AIME2025, demonstrating that zero-to-one transitions are diagnosable and steerable through driver-guided negative learning. Our code is available at https://github.com/jameszhu-pixel-time /Z2O.

## 1 Introduction

Reinforcement Learning with Verifiable Rewards (RLVR) (Lambert et al., 2025) has rapidly emerged as a dominant paradigm for post-training reasoning models, particularly in mathematical problem solving where correctness can be automatically verified (Yang et al., 2025a; Guo et al., 2025). At each training step, the policy samples multiple rollouts per prompt and receives binary rewards indicating whether each final answer is correct. When some rollouts in a group succeed, the resulting contrast between positive and negative samples provides a clear gradient that drives steady improvement. However, on sufficiently hard prompts the on-policy pass rate is so low that entire rollout groups contain only incorrect samples for many consecutive steps. In group-relative methods such as GRPO (Shao et al., 2024; Guo et al., 2025), this eliminates reward contrast and yields near-zero advantages on precisely the prompts that most need exploration. As a result, hard prompts often remain stagnant for many steps and then transition abruptly from unsolved to solved — a discrete “zero-to-one” jump rather than smooth refinement. The dynamics that drive this transition, and whether they can be steered, remain largely unexplored.

Answering these questions, however, requires confronting two intertwined obstacles that current RLVR pipelines leave unresolved. The first obstacle is diagnosing zero-to-one events under binary supervision. Standard RLVR diagnostics offer little purchase on what happens inside a hard prompt before it is first solved: pass@k (Chen et al., 2021) aggregates end-of-training capability and cannot localize the moment a previously unsolved prompt becomes solvable, while grouplevel rewards remain identically zero across long stretches of training and therefore carry no information about which unsolved prompts are close to a breakthrough. What is missing is (i) an event-based view of training that pinpoints when transitions occur and (ii) an internal, rollout-level signal that distinguishes near-solvable failures from true dead ends using only the information already available at sampling time.

The second obstacle is turning heterogeneous negatives into informative gradient on fullynegative groups. A majority of RLVR methods build on grouped objectives such as GRPO and its variants (Shao et al., 2024; Yu et al., 2025; Zheng et al., 2025a; Wu et al., 2026), where all-negative groups produce vanishing advantages and provide little learning signal (Ma et al., 2026; Yu et al., 2025). Recent variants restore a gradient through weighted negative learning, virtual rewards, or entropy-aware updates (Zhu et al., 2025; Nan et al., 2025; Le et al., 2026), but they apply roughly uniform pressure across negatives. The open problem is therefore how to assign graded credit among them under only binary rewards, in a way that injects gradient into unsolved groups without destabilizing the rest of training.

![](images/785778702e83e91345844e061c0abae6318024cbc5460e2994141ab36973a6a9.jpg)  
Figure 1: Overview of NAPO pipeline

We analyze token-level generation entropy during RLVR training and find that negative rollouts from prompts that later undergo a Negative-to-Positive (N→P) transition exhibit front-loaded uncertainty, with early-response entropy patterns that distinguish them from rollouts on prompts that remain negative (N→N). Based on this signal, we propose NAPO (Negative-Augmented Policy Optimization), a GRPO-integrated method that ranks negative samples by front-loaded entropy and injects informative gradients into unsolved groups.

Our contributions are threefold:

• We study zero-to-one learning dynamics in RLVR through event-based metrics, especially N→P transitions and TTFP, to characterize how hard prompts first become solvable.

• We identify a token-entropy pattern associated with N→P transitions and propose earlybranching drivers that rank negative rollouts by their latent learning potential.

• We introduce NAPO, a GRPO-integrated algorithm that assigns graded advantages to negative samples according to their driverbased ranking while maintaining stable updates via in-group normalization, improving token-efficient exploration on hard prompts as reflected by better TTFP and higher pass@k.

## 2 Preliminaries

Related Work. Reinforcement learning (RL) (Sutton and Barto, 1998) has been shown to improve LLM reasoning via both exploration and exploitation (Auer et al., 2002; Cheng et al., 2026). Approaches such as RLHF (Ouyang et al., 2022), Test-Time RL (Zuo et al., 2025), and RLVR (Guo et al., 2025) train policies using verifiable rewards or reward models, often with policy-gradient methods like PPO, GRPO, and REINFORCE (Schulman et al., 2017; Shao et al., 2024; Williams, 1992). These methods can stall on very hard prompts (Ma et al., 2026). Recent work addresses this by reallocating rollouts toward learnable prompts and by extracting informative signals from negative samples, including token-level entropy to encourage exploration and reduce overfitting (Zhu et al., 2025;

Le et al., 2026; Fang et al., 2026). We further introduce the negative-to-positive transition to predict and leverage latent prompt learnability to overcome this challenge.

Group-Relative Policy Optimization for RLVR . Following recent RLVR practice for reasoning models, we sample a group of G rollouts $\{ y _ { i } \} _ { i = 1 } ^ { \bar { G } }$ for each prompt and optimize the policy with Group Relative Policy Optimization (GRPO) (Guo et al., 2025; Shao et al., 2024). Let $\pi _ { \theta _ { \mathrm { o l d } } }$ be the snapshot policy and define the token-level likelihood ratio $\begin{array} { r } { \rho _ { i , t } ( \theta ) = \frac { \pi _ { \theta } \left( a _ { i , t } | s _ { i , t } \right) } { \pi _ { \theta _ { \mathrm { o l d } } } \left( a _ { i , t } | s _ { i , t } \right) } } \end{array}$ . GRPO then applies the clipped surrogate objective

$$
\begin{array} { l } { \displaystyle \operatorname* { m a x } _ { \theta } \mathbb { E } _ { x } \mathbb { E } _ { y _ { i } \sim \pi _ { \theta _ { \mathrm { o l d } } } } \Bigg [ \sum _ { i = 1 } ^ { G } \sum _ { t } \ell _ { i , t } - \beta \operatorname { K L } ( \pi _ { \theta } \parallel \pi _ { \mathrm { r e f } } ) \Bigg ] } \\ { \ell _ { i , t } = \operatorname* { m i n } \Big ( \rho _ { i , t } ( \theta ) \hat { A } _ { i , t } , } \\ { \qquad \mathrm { c l i p } ( \rho _ { i , t } ( \theta ) , 1 - \epsilon , 1 + \epsilon ) \hat { A } _ { i , t } \Big ) , } \end{array}\tag{1}
$$

where $\hat { A } _ { i , t }$ is the group-relative advantage, $\pi _ { \mathrm { r e f } }$ is a reference policy, and $\beta$ controls KL regularization. For brevity, we use $\pi _ { \theta }$ and $\pi _ { \mathrm { r e f } }$ to denote the distributions conditioned on a given prompt x, i.e., $\pi _ { \boldsymbol { \theta } } ( \cdot \mid x )$ and $\pi _ { \mathrm { r e f } } ( \cdot \mid x )$ , respectively.

PSR/NSR decomposition. For verifiable-reward post-training, each rollout receives a binary outcome. We view the learning signal through the positive-sample reward (PSR) and negative-sample reward (NSR) decomposition (Zhu et al., 2025). Let $r ( x , y ) \in \{ - 1 , + 1 \}$ be the verifiable reward for completion y on prompt x. A simplified RLVR objective is

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { R L V R } } ( \theta ) : = - \mathbb { E } _ { x \sim \mathcal { D } } \mathbb { E } _ { y \sim \pi _ { \theta } ( \cdot \vert x ) } [ r ( x , y ) ] . } \end{array}\tag{2}
$$

Using $r = \mathbb { I } [ r = + 1 ] - \mathbb { I } [ r = - 1 ]$ , we can write

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { R L V R } } ( \theta ) = \underbrace { - \mathbb { E } _ { x \sim \mathcal { D } } \mathbb { E } _ { y \sim \pi _ { \theta } ( \cdot \vert x ) } [ \mathbb { I } [ r ( x , y ) = + 1 ] ] } _ { \mathcal { L } _ { \mathrm { P S R } } ( \theta ) } } \\ { + \underbrace { \mathbb { E } _ { x \sim \mathcal { D } } \mathbb { E } _ { y \sim \pi _ { \theta } ( \cdot \vert x ) } [ \mathbb { I } [ r ( x , y ) = - 1 ] ] } _ { \mathcal { L } _ { \mathrm { N S R } } ( \theta ) } } \end{array}\tag{3}
$$

PSR reinforces verified rollouts, whereas NSR suppresses failed ones. Both sides can provide useful superivision on reasoning tasks (Zhu et al., 2025).

Token-level uncertainty and rollout-level drivers. To analyze how N→P happens, we analyze the patterns of negative rollouts before N→P happens. We measure token-level uncertainty with the entropy of the next-token distribution (Wang et al., 2025; Yang et al., 2025b). For each rollout, let $p _ { t } ^ { ( k ) }$ denote the top-k truncated next-token distribution at position t after renormalization, with invalid and padding positions masked out. We define the token-level entropy sequence

$$
H _ { t } ^ { ( k ) } = - \sum _ { v } p _ { t } ^ { ( k ) } ( v ) \log p _ { t } ^ { ( k ) } ( v ) ,\tag{4}
$$

and its exponential form

$$
\rho _ { t } ^ { ( k ) } = \exp \bigl ( H _ { t } ^ { ( k ) } \bigr ) ,\tag{5}
$$

which we use as a token-level perplexity / effective branching statistic. Our goal is not to measure absolute early-stage uncertainty in isolation, but to summarize the relative prominence of early uncertainty within each rollout. <sup>1</sup>

N→P Transition. To localize progress on hard prompts, let $q _ { t } ( x )$ indicate whether evaluation at training step t contains at least one positive rollout for prompt x. We define the time-to-first-positive (TTFP) as

$$
\mathrm { T T F P } ( x ) = \operatorname* { m i n } \{ t : q _ { t } ( x ) = 1 \} ,
$$

with TTF $\mathrm { P } ( x ) = \infty$ if no positive appears within the training horizon. We define an N→P transition as

$$
q _ { t - 1 } ( x ) = 0 \ \land \ q _ { t } ( x ) = 1 .
$$

This implies that the model solves a problem on which all previous attempts failed.

## 3 Signals of Near-Solvable Failures

This section answers the question of what drives the N→P transition.

We examine two axes of variation in this section: (i) within-batch, comparing rollouts inside all-negative groups, and (ii) between-replay, comparing two training continuations that share a base state but differ in sampling stochasticity.

<table><tr><td></td><td colspan="2">AMC23</td><td colspan="2">AIME2025</td><td colspan="2">Math-Beyond</td><td colspan="3">Training Set</td></tr><tr><td>Replay</td><td>N→P</td><td>P→N</td><td>N→P</td><td>P→N</td><td>N→P</td><td>P→N</td><td>N→P</td><td>P→N</td><td>TTFP</td></tr><tr><td>nl</td><td>2/3</td><td>0.33</td><td>2.67/20</td><td>2</td><td>12/134</td><td>5.33</td><td>120</td><td>116</td><td>1.213</td></tr><tr><td> $n 2$ </td><td>1.67/3</td><td>2.67</td><td>2/20</td><td>2.33</td><td>8.33/134</td><td>5.33</td><td>117.67</td><td>102.67</td><td>1.225</td></tr></table>

Table 1: Transition statistics at $t ^ { * } = 1$ for the n1–n2 replay pair. A prompt is labeled as P if at least one rollout is correct under the fixed rollout budget, and as N otherwise. In the validation columns, $a / b$ reports a N→P transitions among b prompts that are negative at the base checkpoint. Results are averaged over three runs, and TTFP is computed over the first three steps. Best value is shown in bold.

Fixed-batch protocol. RLVR on hard prompts exhibits strong temporal correlations across training steps (Li et al., 2025; Zheng et al., 2025b). To expose this structure directly, we train Qwen2.5- Math-7B on a fixed set of 1024 DAPO-17k prompts for 10 consecutive gradient steps, sampling $G = 8$ rollouts per prompt per step. Holding the prompt batch fixed across steps lets us align rollout-internal uncertainty patterns with later N→P events using the metrics of §2. Unless otherwise noted, we use temperature $T = 1 . 0 $ , maximum context length 4096, learning rate $1 0 ^ { - 6 }$ , and zero entropy/KL penalties.<sup>2</sup>

Critical-step replay. Across the 10-step window, we identify the step $t ^ { * }$ at which N→P events peak and treat it as the critical checkpoint. Starting from this shared state, we replay training with the same base policy, optimizer state, data order, and hyperparameters, varying only rollout stochasticity. This yields two representative continuations, denoted $n _ { 1 }$ and $n _ { 2 }$ . We do not label either replay as “good” or “bad” at this point.

Outcome measures. (i) Prompt-level transition label. Among prompts that are negative at the base checkpoint $t ^ { * }$ , we call a prompt N→P if at least one later checkpoint within the replay window produces a positive rollout for it, and N→N otherwise. Each negative rollout inherits its prompt’s label. At training time the binary reward gives both groups identical labels; the partition into N→P and N→N is only observable after the replay window, which is precisely what makes the within-batch driver signal of §3.2.2 non-trivial.

solution discovery.

(ii) Replay-level transition counts. For each replay we report the number of N→P and P→N transitions under 64 rollouts per prompt, together with the mean TTFP over prompts within the replay window. Smaller first-positive indices indicate earlier

(iii) Validation sets. We evaluate replay checkpoints on AMC23, AIME 2025, and Math-Beyond (Mayilvahanan et al., 2025) under the same 64- rollout budget.

## 3.2 Findings and Implications

We first define a rollout-time driver that compresses the entropy–position pattern into a single scalar (§3.2.1), then validate it at each level in turn (§3.2.2 and §3.2.3).

## 3.2.1 An early-branching driver

Why a ratio, not an absolute level. A naive metric would be the average entropy of a rollout, but the contrast we care about is not “more entropy” in absolute terms; it is a difference in the timing of uncertainty—how much of a rollout’s total uncertainty is allocated to its opening tokens (Fang et al., 2026).

Definition. Let $T _ { \alpha } = \lceil \alpha T \rceil$ denote the number of tokens in the first α fraction of a response of length T. The early mean-perplexity ratio is

$$
d _ { \mathrm { e a r l y - p p l } } = \frac { \frac { 1 } { T _ { \alpha } } \sum _ { t = 1 } ^ { T _ { \alpha } } \rho _ { t } ^ { ( k ) } } { \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \rho _ { t } ^ { ( k ) } } ,\tag{6}
$$

where $\rho _ { t } ^ { ( k ) } = \exp ( H _ { t } ^ { ( k ) } )$ is the exponential of the top-k truncated entropy defined in Eq. (5). Values of $d _ { \mathrm { e a r l y - p p l } } >$ 1 indicate that uncertainty is concentrated early in the response<sup>3</sup>.

Choice of the early fraction. We introduce the early-fraction parameter α and compare 0.1, 0.2, and 0.5 on the $n _ { \mathrm { 1 } } { - } n _ { \mathrm { 2 } }$ replay pair (Fig. $5 ) ^ { 4 }$ . Among these candidates, 0.2 yields the clearest and most stable separation between N→P and N→N rollouts while remaining robust across both replays, and we use it throughout the remainder of the paper.

![](images/ab193d2c001a277d9690227fee9775434e27acc73db07e99fa82adb0c37dee44.jpg)  
(a) Entropy-position curves for the bad replay n2.

![](images/53867bf322df5a5a5c57189b5a21cbc597824ae23d76b8ebed1b5a6735f8edd5.jpg)  
(b) Comparison of the N→P group between n1 and $n 2 .$  
Figure 2: Entropy-position plots. The main separation is localized in the early part of generation. Prompts that later undergo N→P transitions preserve more early-stage uncertainty than prompts that remain N→N, and this pattern is sharper in the good replay n1 than in the bad replay n2.

## 3.2.2 Within-batch evidence: negatives are not homogeneous

Entropy–position pattern. Figure 2a plots mean token entropy as a function of position for negative rollouts in replay $n _ { 2 }$ . The N→P group maintains noticeably higher uncertainty at the beginning of the response, whereas the N→N group collapses earlier to lower-entropy continuations. The two groups converge later in the response, confirming that the discriminative signal is localized in the early stage.

Driver separation. Figure 3a shows the distribution of $d _ { \mathrm { e a r l y - p p l } }$ for the two groups in replay $n _ { 1 }$ Negatives associated with future N→P prompts are systematically shifted toward larger driver values than negatives associated with N→N prompts, mirroring the qualitative pattern of Fig. 2a under the single scalar of Eq. (6).

Claim A (within-batch). The driver provides a rollout-level ordering signal that separates nearsolvable failures from dead ends inside an otherwise indistinguishable negative pool.

## 3.2.3 Between-replay evidence: the driver tracks aggregate transition tendency

Entropy–position pattern. Figure 2b restricts attention to N→P negatives and compares $n _ { 1 }$ against $n _ { 2 }$ . The good replay $n _ { 1 }$ preserves more early-stage uncertainty than the bad replay $n _ { 2 } .$ , sharpening the front-loaded pattern of Claim A but now across replays rather than within one. This is consistent with the relevant contrast being a matter of timing rather than magnitude: the same shape of distribution is shifted earlier in $n _ { 1 }$ than in $n _ { 2 }$

Driver distribution. Figure 3b overlays all four (replay × label) groups. n<sub>1</sub> allocates more probability mass to high-driver rollouts than $n _ { 2 }$ does in both groups. <sup>5</sup>

Realized transitions. Table 1 reports the downstream outcome. Starting from an identical base state, the good replay $n _ { 1 }$ realizes more N→P transitions on the training set and attains a smaller mean first-positive index than n<sub>2</sub>. The ranking of realized transitions matches the ranking of right-tail driver mass, indicating that the batch-level driver shift is not merely a cosmetic difference but is related to which replay will subsequently break more prompts out of the all-negative regime. This is consistent with recent RLVR observations that lowsuccess prompts exhibit structured temporal dynamics rather than noise-like fluctuations throughout training (Zheng et al., 2025b).

Claim B (between-replay). The driver also functions as a batch-level diagnostic: batches whose driver distribution has a heavier right-tail mass go on to realize more N→P events.

![](images/ce31ff65022e1b4764b51e6ff1c1b165da879d2e3d8dfc22ce849f1b83d59837.jpg)  
(a) Driver distribution for the good replay n1. See Appendix B for statistics’ demonstration

![](images/8c7cb0551337a7ea321a9f783c14f5e33555cc37df07b0996aa48c0b5b8ca625.jpg)  
(b) Driver distribution for the bad replay n2 and good replay ${ \romannumeral 1 } .$  
Figure 3: Within-batch separation under the practical driver $d _ { \mathbf { e a r l y - p p l } } .$ . Rollouts associated with prompts that later undergo N→P transitions are systematically right-shifted relative to rollouts from prompts that remain N→N. The separation is present in both replays, but is cleaner and more pronounced in the good replay $n l .$

## Takeaway

Prompts approaching the N→P boundary exhibit a distinctive form of front-loaded uncertainty in their negative rollouts. The early-branching driver $d _ { \mathrm { e a r l y - p p l } }$ captures this pattern at two levels of aggregation: it separates N→P from N→N negatives within a batch (Claim A), and it separates higher-yield from lower-yield batches across replays (Claim B).

## 4 Algorithm

We now turn the empirical findings above into a training design. Our goal is to translate the driver into a practical update rule for RLVR.

## 4.1 Theoretical Interpretation

We provide a view of Fisher information to theoretically interpret the driver on the negative rollouts.

Fisher information as a measure of policy plasticity. Larger local Fisher means that a failed continuation is easier to reshape under a small parameter update, so the policy remains more plastic around that rollout. Let $\mathbf { p } _ { t } = \pi _ { \theta } ( \cdot \mid x , y _ { < t } )$ be the next-token distribution at step t, and let $\mathbf { F } _ { t }$ denote the corresponding token-level Fisher contribution.

Lemma 4.1 (Token-level Fisher information and entropy). For token distribution $\mathbf { p } _ { t } ,$ , the token-level Fisher trace is

$$
\mathrm { T r } ( \mathbf { F } _ { t } ) = 1 - \sum _ { k \in V } p _ { t , k } ^ { 2 } .
$$

Accordingly, Shannon entropy serves as a useful proxy for local Fisher information: higher-entropy token distributions typically correspond to more policy-plastic states, although the two quantities are not identical in general.

Implication for negative learning. Lemma 4.1 suggests that uniform penalties are wasteful: lowentropy failures are already rigid, so pushing them down mostly reinforces dead ends. By contrast, failures generated in higher-entropy states are more plastic and retain nearby alternatives, making negative updates more likely to redirect the policy toward useful exploration. Empirically, the informative plasticity is concentrated early in the rollout, which is why NAPO uses the early-branching driver rather than generic response-level entropy. In short, we want to penalize failures more strongly where the policy is still movable.

## Takeaway

The driver indicates that failures arising in more plastic regions of the policy. The N→P phenomenon suggests that failures with smaller driver values are not random or hopeless: they come from prompts whose trajectories can cross into correctness under nearby policy shifts.

This leads to a design intuition that larger penalties can be assigned to negative trajectories with larger drivers where the policy still retains meaningful uncertainty and plausible alternatives.

## 4.2 Negative-Augmented Policy Optimization

Motivated by the heterogeneity of negative rollouts, we propose NAPO, a GRPO-integrated algorithm that augments standard policy optimization with driver-weighted negative learning on unsolved groups.

![](images/dcf2ba47100791663da75d493aa85005759166b169e9b90d7eb7c60b9153f599.jpg)  
Figure 4: Cumulative solved prompts on validation sets. NAPO achieves faster early progress, solving more prompts within the first 50 steps, while maintaining comparable performance at later stages.

Algorithm 1 NAPO: Negative-Augmented Policy   
Optimization   
1: Sample prompts $x ;$ for each x, sample rollouts   
$\{ y _ { i } \}$ and rewards $r ( x , y _ { i } )$ ;   
2: $\mathbf { i f } \exists i : r ( x , y _ { i } ) = + 1$ then   
3: Compute standard GRPO advantages   
$A _ { \mathrm { G R P O } } ( x , y _ { i } ) ;$   
4: Update the policy with GRPO loss.   
5: else   
6: Compute driver scores for all negative roll  
outs by (6);   
7: Re-normalize scores into [0, 1] by (7);   
8: Assign rollout-level advantages by (8);   
9: Update the policy with GRPO loss.   
10: end if

Driver-weighted negative advantage. NAPO uses the driver to modulate negative learning strength inside unsolved groups. Given driver scores $d ( x , y _ { i } )$ for rollouts in a fully negative group, we first re-normalize them into [0, 1] to avoid exploding gradient while preserving a relevant advantage magnitude:

$$
\tilde { d } ( x , y _ { i } ) = \frac { d ( x , y _ { i } ) - \operatorname* { m i n } _ { j } { d ( x , y _ { j } ) } } { \operatorname* { m a x } _ { j } { d ( x , y _ { j } ) } - \operatorname* { m i n } _ { j } { d ( x , y _ { j } ) } + \epsilon } .
$$

We then assign rollout-level advantages as

(7)

$$
A ( x , y ) = \left\{ \begin{array} { l l } { A _ { \mathrm { G R P O } } ( x , y ) , } & { \mathrm { i f } \ p ( x ) = 1 , } \\ { - \tilde { d } ( x , y ) , } & { \mathrm { i f } \ p ( x ) = 0 , } \end{array} \right.\tag{8}
$$

where $p ( x ) = \mathbb { I } [ \exists i : r ( x , y _ { i } ) = + 1 ]$ indicates whether the current group is solved.

![](images/ab48ca9d7b70c13036c524c5b39316bcf86f228a7d9d628b17c54038daf3b2ab.jpg)  
Figure 5: Number of solved problems across training steps. Setting the early fraction gate to 0.2 leads to the fastest exploration of correct solutions, consistent with our inference-time findings.

Objective: Grouped advantage on solved groups and weighted NSR on unsolved groups. Let $p ( x ) = \mathbb { I } [ \exists i : r ( x , y _ { i } ) = + 1 ]$ denote whether the current group is solved. Equivalently, the rolloutlevel advantage in Eq. (8) can be written as

$$
\begin{array} { c } { { A ( x , y _ { i } ) = p ( x ) A _ { \mathrm { G R P O } } ( x , y _ { i } ) } } \\ { { - \left( 1 - p ( x ) \right) \tilde { d } ( x , y _ { i } ) . } } \end{array}\tag{9}
$$

When p(x) = 1, Eq. (9) reduces to the standard GRPO clipped update. When $p ( x ) = 0$ , we have $A ( x , y _ { i } ) = - \tilde { d } ( x , y _ { i } )$ , so the same clipped surrogate becomes a weighted NSR loss in which largerdriver negatives receive larger-magnitude penalties while clipping preserves the usual PPO/GRPO trust-region effect.

## 5 Evaluation

We empirically evaluate the effectiveness of NAPO on hard prompts.

Setup. We evaluate NAPO under a fixed rollout budget and compare against GRPO and negativesample-based baselines, including NSR’s weighted variant (w-REINFORCE). We also evaluate the impact of early fraction ratio, where NAPO02 refers to taking the first 20% of the response in the driver, while NAPO05 takes 50% and NAPO01 takes 10%. We report (i) Pass@k on validation sets and (ii) TTFP. Experiments are conducted on multiple validation sets (AMC23, AIME2025, AIME24, and Math-Beyond), covering a wide range of difficulty levels. 6

<table><tr><td rowspan="2">Method k</td><td colspan="9">Pass@k</td></tr><tr><td>1</td><td>2</td><td>4</td><td>8</td><td>16</td><td>32</td><td>64</td><td>128</td><td>256</td></tr><tr> colspan="12">Base</td></tr><tr><td>0.2847</td><td>0.4512</td><td>0.6237</td><td>0.7546</td><td>0.8400</td><td>0.9010</td><td>0.9495</td><td>0.9838</td><td>1.0000</td><td>GRPO</td></tr><tr><td>0.7352</td><td>0.7910</td><td>0.8211</td><td>0.8443</td><td>0.8681</td><td>0.8972</td><td>0.9306</td><td>0.9579</td><td>0.9750</td><td>W-REINFORCE</td></tr><tr><td>0.4514</td><td>0.5700</td><td>0.6570</td><td>0.7245</td><td>0.7900</td><td>0.8555</td><td>0.9086</td><td>0.9450</td><td>0.9750</td><td>NAPO02</td></tr><tr><td>0.7141</td><td>0.7718</td><td>0.8051</td><td>0.8328</td><td>0.8628</td><td>0.8940</td><td>0.9222</td><td>0.9467</td><td>0.9750</td><td>NAPO05</td></tr><tr><td>0.7191</td><td>0.7828</td><td>0.8247</td><td>0.8556</td><td>0.8845</td><td>0.9147</td><td>0.9438</td><td>0.9654</td><td>0.9750</td><td>AIME2025</td></tr><tr><td colspan="10">Base</td></tr><tr><td>0.0305</td><td>0.0561</td><td>0.0964</td><td>0.1496</td><td>0.2074</td><td>0.2657</td><td>0.3216</td><td>0.3684</td><td>0.4000</td><td>GRPO</td></tr><tr><td>0.1346</td><td>0.1753</td><td>0.2165</td><td>0.2563</td><td>0.2910</td><td>0.3271</td><td>0.3740</td><td>0.4294</td><td>0.4667</td><td>W-REINFORCE</td></tr><tr><td>0.0254</td><td>0.0439</td><td>0.0689</td><td>0.0970</td><td>0.1300</td><td>0.1767</td><td>0.2425</td><td>0.3278</td><td>0.4333</td><td>NAPO02</td></tr><tr><td>0.1220</td><td>0.1648</td><td>0.2113</td><td>0.2631</td><td>0.3175</td><td>0.3695</td><td>0.4195</td><td>0.4745</td><td>0.5333</td><td>NAPO05</td></tr><tr><td>0.1138</td><td>0.1576</td><td>0.2008</td><td>0.2475</td><td>0.3006</td><td>0.3555</td><td>0.4130</td><td>0.4741</td><td>0.5333</td><td>AIME24</td></tr><tr><td colspan="10">Base</td></tr><tr><td>0.0857</td><td>0.1508</td><td>0.2415</td><td>0.3425</td><td>0.4343</td><td>0.5112</td><td>0.5819</td><td>0.6575</td><td>0.7333</td><td>GRPO</td></tr><tr><td>0.2832</td><td>0.3574</td><td>0.4242</td><td>0.4860</td><td>0.5391</td><td>0.5869</td><td>0.6330</td><td>0.6719</td><td>0.7000</td><td>W-REINFORCE</td></tr><tr><td>0.1516</td><td>0.2322</td><td>0.3105</td><td>0.3706</td><td>0.4152</td><td>0.4500</td><td>0.4821</td><td>0.5209</td><td>0.5667</td><td>NAPO02</td></tr><tr><td>0.2988</td><td>0.3878</td><td>0.4648</td><td>0.5314</td><td>0.5917</td><td>0.6504</td><td>0.7048</td><td>0.7564</td><td>0.8000</td><td>NAPO05</td></tr><tr><td>0.2898</td><td>0.3762</td><td>0.4490</td><td>0.5133</td><td>0.5734</td><td>0.6301</td><td>0.6810</td><td>0.7248</td><td>0.7667</td><td>Math-Beyond</td></tr><tr><td colspan="10">Base</td></tr><tr><td>0.0426</td><td>0.0702</td><td>0.1051</td><td>0.1446</td><td>0.1879</td><td>0.2357</td><td>0.2866</td><td>0.3391</td><td>0.3978</td><td>GRPO</td></tr><tr><td>0.1337</td><td>0.1574</td><td>0.1780</td><td>0.1984</td><td>0.2205</td><td>0.2454</td><td>0.2723</td><td>0.3014</td><td>0.3370</td><td>W-REINFORCE</td></tr><tr><td>0.0587</td><td>0.0866</td><td>0.1147</td><td>0.1400</td><td>0.1639</td><td>0.1899</td><td>0.2197</td><td>0.2564</td><td>0.3094</td><td>NAPO02</td></tr><tr><td>0.1348</td><td>0.1612</td><td>0.1882</td><td>0.2130</td><td>0.2334</td><td>0.2527</td><td>0.2763</td><td>0.3105</td><td>0.3591</td><td>NAPO05</td></tr><tr><td>0.1383</td><td>0.1680</td><td>0.1948</td><td>0.2164</td><td>0.2337</td><td>0.2508</td><td>0.2704</td><td>0.2926</td><td>0.3204</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 2: Pass@k comparison across datasets. The row corresponding to our method is highlighted in gray, the best value within each dataset block and column is shown in bold, and the second-best value is underlined. For W-REINFORCE method, we set the positive advantage weight to 0.5 to handle the much harder prompts.

Training protocol. We use the verl framework (Sheng et al., 2025) and train on DAPO-17k (Yu et al., 2025), a challenging dataset for Qwen2.5- Math-7B (Yang et al., 2024), where the initial pass rate is below 0.2. For each prompt, we sample G = 8 rollouts per step. Unless otherwise specified, we use temperature T = 1.0 during training, a maximum context length of 4096, and train for one epoch.<sup>7</sup> We report both large-scale validation results and controlled experiments under identical training batches (as in § 3.2).

Results. (1) Improved sample efficiency on hard prompts. As shown in Figure 4, NAPO discovers first-positive rollouts earlier across datasets, solving more prompts during early training under the same rollout budget. This indicates improved sample efficiency, particularly in lowsuccess regimes where standard GRPO suffers from weak learning signals. These gains require little additional computation: NAPO uses no extra rollout, verifier call, auxiliary model, or policy forward pass. We compute the uncertainty statistic from logits already produced during the old-policy forward pass and retain one scalar per valid response token rather than a vocabulary-sized tensor; the remaining operations are masking, sequencewise reductions, and within-group min–max normalization. Across common logged steps, the advantage calculation time is 0.935 ± 0.030 s/step for NAPO and $0 . 5 2 5 \pm 0 . 0 1 6 \mathrm { s / s t e p }$ for GRPO (mean ± standard deviation). The resulting increment is only 0.410 s/step, approximately 0.03% of the observed end-to-end step time. Thus, the trace shows no material training-time overhead.

(2) Effect of negative scaling. Comparing NAPO and NAPO05 reveals a clear trade-off rather than a uniform winner. The milder NAPO05 variant performs best on AMC23, on the highest-k region of AIME2025, and on low-k Math-Beyond, whereas full NAPO is clearly strongest on AIME24 and in the mid-k regime of AIME2025. This pattern suggests that negative updates should be tuned to benchmark difficulty: stronger shaping helps when there is substantial room for zero-to-one improvement, while milder scaling is more robust once exploration is less constrained.

<table><tr><td></td><td colspan="3">∆Pass@k (%)</td></tr><tr><td>Dataset</td><td>k = 1</td><td>k = 8</td><td>k = 256</td></tr><tr><td>AMC23</td><td>+14.40</td><td>+7.95</td><td>+1.88</td></tr><tr><td>AIME2025</td><td>+10.48</td><td>+15.31</td><td>+8.33</td></tr><tr><td>AIME24</td><td>+20.15</td><td>+27.28</td><td> $+ 1 7 . 5 0$ </td></tr><tr><td>Math-Beyond</td><td>+7.43</td><td>+8.73</td><td>+0.41</td></tr></table>

Table 3: Pass@k gains of NAPO over Uniform(−0.1) across datasets at $T = 1 . 0 .$ , averaged over four rollout reevaluations with 256 generations per question. Values are percentage points; positive values favor NAPO.

(3) Within-group differentiation versus uniform negative feedback. To isolate the effect of within-group differentiation, we add a Uniform(−0.1) control that assigns the same small negative advantage to every rollout in an allnegative group. All training and evaluation settings are otherwise matched. Table 3 reports the mean paired improvement of NAPO over this control across four independent evaluations.

Among the evaluation results above, the separation is strongest on the AIME benchmarks—for example, +27.28 points at Pass@8 on AIME24— and the mean remains positive at Pass@256 on all four datasets. In a separate singleevaluation control with Uniform(−1), NAPO improves Pass@256 by 17.50, 23.33, 30.00, and 17.13 percentage points on AMC23, AIME2025, AIME24, and Math-Beyond, respectively. Thus, simply activating constant negative feedback does not reproduce NAPO in these runs; these controls isolate the benefit of driver-based withingroup differentiation over a constant update. Because the four reevaluations use fixed checkpoints rather than independent training runs, they establish evaluation-level consistency but not training-seed significance.

## 6 Conclusion

We studied the zero-to-one problem in RLVR, motivated by two obstacles: diagnosing N → P transitions under binary supervision, and extracting informative gradients from heterogeneous negatives in fully-unsolved groups. Using event-based metrics (TTFP and N → P events), we showed that failures preceding successful transitions retain higher early-stage uncertainty, providing a practical signal to separate near-solvable rollouts from dead ends. Building on this, we proposed NAPO, a GRPO-integrated method that assigns graded negative advantages according to driver scores under a Fisher-information view, thereby injecting structured gradients into unsolved groups while preserving stable updates on solved ones. NAPO accelerates early solution discovery and improves Pass@256 by 10.0 points on AIME24 and 6.6 points on AIME2025 compared with GRPO, confirming that driver-guided negative learning is an effective way to broaden reasoning boundaries in low-success RLVR.

## Limitations

Our analysis and method are currently focused on math-oriented RLVR settings with binary verifiers. Although these tasks provide a clean testbed for studying zero-to-one transitions, the same entropybased driver may behave differently in broader domains, longer-horizon reasoning tasks, or environments with partial-credit, noisy, or subjective feedback. Extending NAPO to such settings may require more robust driver definitions and calibration procedures.

NAPO also relies on rollout-level statistics collected during training. Its benefits may therefore depend on the rollout budget, group size, decoding temperature, and the quality of the base policy. When the model rarely approaches a correct solution, early uncertainty patterns may be insufficient to identify useful negative samples; conversely, when success rates are already high, the additional weighting of negatives may provide limited gains. Future work should evaluate the method under more diverse model scales, verifier designs, and compute budgets.

## Acknowledgments

This work was supported in part by China NSF grant No.U2268204, 62322206, 62432007, 62441236, 62132018, 62025204, 62272307, 62372296, U25A6024. The opinions, findings, conclusions, and recommendations expressed in this paper are those of the authors and do not necessarily reflect the views of the funding agencies or the government.

## References

Peter Auer, Nicolò Cesa-Bianchi, and Paul Fischer. 2002. Finite-time analysis of the multiarmed bandit problem. Machine Learning, 47(2–3):235–256.

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Pondé de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger, Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela Mishkin, Brooke Chan, Scott Gray, and 39 others. 2021. Evaluating large language models trained on code. CoRR, abs/2107.03374.

Daixuan Cheng, Shaohan Huang, Xuekai Zhu, Bo Dai, Xin Zhao, Zhenliang Zhang, and Furu Wei. 2026. Reasoning with exploration: An entropy perspective. Proceedings of the AAAI Conference on Artificial Intelligence, 40(36):30377–30385.

Yangyi Fang, Jiaye Lin, Xiaoliang Fu, Cong Qin, Haolin Shi, Chaowen Hu, Lu Pan, Ke Zeng, and Xunliang Cai. 2026. How to allocate, how to learn? dynamic rollout allocation and advantage modulation for policy optimization. Preprint, arXiv:2602.19208.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, Xiaokang Zhang, Xingkai Yu, Yu Wu, Z. F. Wu, Zhibin Gou, Zhihong Shao, Zhuoshu Li, Ziyi Gao, Aixin Liu, and 175 others. 2025. Deepseek-r1 incentivizes reasoning in llms through reinforcement learning. Nature, 645(8081):633–638.

Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, Sida Wang, Armando Solar-Lezama, Koushik Sen, and Ion Stoica. 2024. Livecodebench: Holistic and contamination free evaluation of large language models for code. Preprint, arXiv:2403.07974.

Nathan Lambert, Jacob Morrison, Valentina Pyatkin, Shengyi Huang, Hamish Ivison, Faeze Brahman, Lester James Validad Miranda, Alisa Liu, Nouha Dziri, Xinxi Lyu, Yuling Gu, Saumya Malik, Victoria Graf, Jena D. Hwang, Jiangjiang Yang, Ronan Le Bras, Oyvind Tafjord, Christopher Wilhelm, Luca Soldaini, and 4 others. 2025. Tulu 3: Pushing frontiers in open language model post-training. In Second Conference on Language Modeling.

Thanh-Long V. Le, Myeongho Jeon, Kim Vu, Viet Dac Lai, and Eunho Yang. 2026. No prompt left behind: Exploiting zero-variance prompts in LLM reinforcement learning via entropy-guided advantage shaping. In The Fourteenth International Conference on Learning Representations.

Xuefeng Li, Haoyang Zou, and Pengfei Liu. 2025. Limr: Less is more for rl scaling. Preprint, arXiv:2502.11886.

Shichao Ma, Zhiyuan Ma, Ming Yang, Xiaofan Li, Xing Wu, Jintao Du, Yu Cheng, Weiqiang Wang, Qiliang Liu, Zhengyang Zhou, and Yang Wang. 2026.

Tspo: Breaking the double homogenization dilemma in multi-turn search policy optimization. Preprint, arXiv:2601.22776.

Prasanna Mayilvahanan, Ricardo Dominguez-Olmedo, Thaddäus Wiedemer, and Wieland Brendel. 2025. Math-beyond: A benchmark for rl to expand beyond the base model. Preprint, arXiv:2510.11653.

Gongrui Nan, Siye Chen, Jing Huang, Mengyu Lu, Dexun Wang, Chunmei Xie, Weiqi Xiong, Xianzhou Zeng, Qixuan Zhou, Yadong Li, and Xingzhong Xu. 2025. Ngrpo: Negative-enhanced group relative policy optimization. Preprint, arXiv:2509.18851.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul F Christiano, Jan Leike, and Ryan Lowe. 2022. Training language models to follow instructions with human feedback. In Advances in Neural Information Processing Systems, volume 35, pages 27730–27744. Curran Associates, Inc.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. 2017. Proximal policy optimization algorithms. Preprint, arXiv:1707.06347.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. 2024. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. Preprint, arXiv:2402.03300.

Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. 2025. Hybridflow: A flexible and efficient rlhf framework. In EuroSys, pages 1279– 1297.

Richard S Sutton and Andrew G Barto. 1998. Reinforcement learning: An introduction, volume 1. MIT press.

Shenzhi Wang, Le Yu, Chang Gao, Chujie Zheng, Shixuan Liu, Rui Lu, Kai Dang, Xiong-Hui Chen, Jianxin Yang, Zhenru Zhang, Yuqiong Liu, An Yang, Andrew Zhao, Yang Yue, Shiji Song, Bowen Yu, Gao Huang, and Junyang Lin. 2025. Beyond the 80/20 rule: Highentropy minority tokens drive effective reinforcement learning for llm reasoning. In Advances in Neural Information Processing Systems, volume 38, pages 115452–115486. Curran Associates, Inc.

Ronald J. Williams. 1992. Simple statistical gradientfollowing algorithms for connectionist reinforcement learning. Machine Learning, 8(3–4):229–256.

Junkang Wu, Kexin Huang, Jiancan Wu, An Zhang, Xiang Wang, and Xiangnan He. 2026. Quantile advantage estimation: Stabilizing RLVR for LLM reasoning. In The Fourteenth International Conference on Learning Representations.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, and 41 others. 2025a. Qwen3 technical report. Preprint, arXiv:2505.09388.

An Yang, Beichen Zhang, Binyuan Hui, Bofei Gao, Bowen Yu, Chengpeng Li, Dayiheng Liu, Jianhong Tu, Jingren Zhou, Junyang Lin, Keming Lu, Mingfeng Xue, Runji Lin, Tianyu Liu, Xingzhang Ren, and Zhenru Zhang. 2024. Qwen2.5-math technical report: Toward mathematical expert model via self-improvement. CoRR, abs/2409.12122.

Chenghao Yang, Lin Gui, Chenxiao Yang, Victor Veitch, Lizhu Zhang, and Zhuokai Zhao. 2025b. Let it calm: Exploratory annealed decoding for verifiable reinforcement learning. Preprint, arXiv:2510.05251.

Chenghao Yang and Ari Holtzman. 2025. Llm probability concentration: How alignment shrinks the generative horizon. Preprint, arXiv:2506.17871.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, juncai liu, LingJun Liu, Xin Liu, Haibin Lin, Zhiqi Lin, Bole Ma, Guangming Sheng, Yuxuan Tong, Chi Zhang, Mofan Zhang, and 17 others. 2025. Dapo: An open-source llm reinforcement learning system at scale. In Advances in Neural Information Processing Systems, volume 38, pages 113222– 113244. Curran Associates, Inc.

Chujie Zheng, Shixuan Liu, Mingze Li, Xiong-Hui Chen, Bowen Yu, Chang Gao, Kai Dang, Yuqiong Liu, Rui Men, An Yang, Jingren Zhou, and Junyang Lin. 2025a. Group sequence policy optimization. Preprint, arXiv:2507.18071.

Haizhong Zheng, Yang Zhou, Brian Bartoldson, Bhavya Kailkhura, Fan Lai, Jiawei Zhao, and Beidi Chen. 2025b. Act only when it pays: Efficient reinforcement learning for llm reasoning via selective rollouts. In Advances in Neural Information Processing Systems, volume 38, pages 124321–124346. Curran Associates, Inc.

Xinyu Zhu, Mengzhou Xia, Zhepei Wei, Wei-Lin Chen, Danqi Chen, and Yu Meng. 2025. The surprising effectiveness of negative reinforcement in llm reasoning. In Advances in Neural Information Processing Systems, volume 38, pages 126546–126573. Curran Associates, Inc.

Yuxin Zuo, Kaiyan Zhang, Li Sheng, Shang Qu, Ganqu Cui, Xuekai Zhu, Haozhan Li, yuchen zhang, Xinwei Long, Ermo Hua, Biqing Qi, Youbang Sun, Zhiyuan Ma, Lifan Yuan, Ning Ding, and Bowen Zhou. 2025. Ttrl: Test-time reinforcement learning. In Advances in Neural Information Processing Systems, volume 38, pages 131459–131483. Curran Associates, Inc.

## A Experiment Settings

## A.1 Chat template

## Chat template for Qwen

System: You are a helpful assistant.   
User: {input} Please reason step by step,   
and put your final answer within \boxed{}.   
Assistant: Model generates a step-by-step   
solution; the final answer is enclosed in   
\boxed{}.

## A.2 Hyperparameter Settings

<table><tr><td>Category</td><td>Setting</td></tr><tr><td>Model Model</td><td>Qwen2.5-Math-7B</td></tr><tr><td>Data</td><td></td></tr><tr><td>Train batch size</td><td>1024</td></tr><tr><td>Max prompt length Max response length</td><td>1024 3072</td></tr><tr><td>RL Algorithm</td><td></td></tr><tr><td>KL coefficient Entropy coefficient</td><td>0.0</td></tr><tr><td>Critic warmup</td><td>0.0 0</td></tr><tr><td>Optimization</td><td></td></tr><tr><td>Learning rate</td><td> $1 \times 1 0 ^ { - 6 }$ </td></tr><tr><td>PPO minibatch size</td><td></td></tr><tr><td></td><td>256</td></tr><tr><td>Sampling</td><td></td></tr><tr><td>top_k</td><td>-1</td></tr><tr><td></td><td></td></tr><tr><td>top_p</td><td>1</td></tr><tr><td>temperature</td><td>1.0</td></tr></table>

Table 4: Detailed training hyperparameters.

All experiments are conducted on a single node equipped with eight NVIDIA H100 GPUs. We report final evaluation results using the checkpoint at step 200 for Qwen2.5-Math-7B and the checkpoint at step 100 for Llama-3.1-8B-Instruct.

## B Definitions of Batch-Level Driver Statistics

In this appendix, we define the batch-level summary statistics used in Table 5 and Figure 3 to characterize the distribution of rollout-time drivers within the N→P group. For Figure 3 We use kernel density estimation (KDE) only for visualization, to provide a smoothed view of the empirical distribution of the driver scores. KDE is a standard nonparametric density estimation method that places a smooth kernel around each observed sample and aggregates these kernels to estimate a continuous density. It does not affect training, evaluation, or the computation of the proposed driver; all quantitative analyses are performed on the original empirical scores.

Let $\{ d _ { i } \} _ { i = 1 } ^ { n }$ denote the driver values computed for the rollouts in a given subset (e.g., the $\mathrm { N } {  } \mathrm { P }$ rollout group of a batch), where n is the number of rollouts in that subset.

Mean-shift statistic (msk). The statistic msk summarizes the average magnitude of the driver values within the selected rollout group:

$$
{ \mathfrak { m s k } } = { \frac { 1 } { n } } \sum _ { i = 1 } ^ { n } d _ { i } .\tag{10}
$$

In our analysis, larger msk indicates that the corresponding batch tends to allocate relatively more uncertainty to the early stage of generation.

Right-tail mass statistic (rt). The statistic rt summarizes the heaviness of the right tail of the driver distribution. Let τ be a fixed upper-tail threshold (we set $\tau = 1 . 9 5$ in practice). We define

$$
{ \mathsf { r t } } = { \frac { 1 } { n } } \sum _ { i = 1 } ^ { n } \mathbf { 1 } [ d _ { i } > \tau ] ,\tag{11}
$$

where 1[·] is the indicator function. Larger rt indicates that a larger fraction of rollouts fall into the high-driver region, i.e., the distribution has a heavier right tail.

Interpretation. These statistics are not intended to serve as calibrated predictors of future success or as exact estimates of future reward. In the main text, we use them only to support a weaker but more practically relevant claim: larger driver values, especially a heavier right tail, are enriched in rollout groups associated with stronger N→P tendency. This is the only property required by our method design, where the driver is used as a surrogate ordering signal rather than as a direct estimator of future pass rate.

## C Additional results

## C.1 Generalization beyond Qwen2.5-Math-7B

Transition-level evidence. Starting from a shared Llama-3.1-8B-Instruct checkpoint, we run five stochastic GRPO continuations. Across the 696 prompts that are negative at the shared checkpoint, the mean N→P rate is $( 1 4 . 8 6 \pm 0 . 4 4 ) \%$ (mean ± standard error across continuations). Thus, delayed first-solution discovery is not confined to Qwen2.5-Math-7B.

<table><tr><td>Replay</td><td>Training N→P</td><td>msk</td><td>rt</td></tr><tr><td> $n l$ </td><td>120</td><td>0.924</td><td>0.039</td></tr><tr><td> $n 2$ </td><td>117.67</td><td>0.812</td><td>0.026</td></tr></table>

Table 5: Representative batch-level driver statistics for a good(n1)/bad(n2) replay comparison. Larger msk and rt indicate more mass in the high-driver region.

Training-level evidence. We apply NAPO using the same early-fraction window $\alpha = 0 . 2$ selected on Qwen2.5-Math-7B (Figure 5), without Llamaspecific tuning. Table 6 compares the resulting step-100 checkpoint against GRPO.

The results are mixed, where NAPO performs best on Math-Beyond and high-budget AIME24, while degrading on AIME2025. Thus, the evaluation indicates that the driver window and resulting advantage magnitudes may require calibration across model families, while remaining robust to a certain degree.

## C.2 Beyond mathematics: binary-verifiable code

Setup. We examine whether the transition-level entropy signal extends beyond mathematical reasoning. We train Qwen3-4B-Instruct (Yang et al., 2025a) on 512 distinct coding problems from Live-CodeBench v5 (Jain et al., 2024) using executionbased binary rewards, and evaluate on 100 held-out problems from LiveCodeBench v6. Five consecutive GRPO steps each visit all 512 training prompts and sample $G = 8$ rollouts per prompt, yielding four adjacent transitions for the analysis below. All other settings follow Section 3.1.

Code-domain entropy statistic. On the fixed sensitivity grid $\alpha \in \{ 0 . 1 , 0 . 2 , 0 . 5 \}$ evaluated on the same transitions, $\alpha = 0 . 5$ gives the clearest observed separation, consistent with the different output structure and prompt formatting of coding tasks. For this experiment, msk denotes the prompt-level mean early-to-global plain-entropy ratio across the eight source-step rollouts. Unlike the NAPO training driver $d _ { \mathrm { e a r l y - p p l } }$ , this analysis-only statistic uses entropy directly rather than its exponential.

Pooling the four transitions gives 83 N→P and 550 N→N prompt–transition observations, whose mean msk values are 1.0985 and 1.0709, respectively. The difference is +0.0276, with a 95% prompt-clustered bootstrap confidence interval of [0.0066, 0.0484]. The step-stratified AUC is 0.608, with a 95% confidence interval of [0.532, 0.681]. The difference has the same sign at all four transitions rather than being driven by a single favorable step. These results provide preliminary evidence for an entropy–exploration signal in the coding tasks.

![](images/5025428777516526a9a185b432fbb4ab8cbff94a86f3a6de6c8af42494917dac.jpg)  
Figure 6: TTFP distribution

![](images/88ed01cb14f791c79f7ea2d1275dd8597ead8b9e83f276e53c0aadf23e6fa6ac.jpg)  
Figure 7: Cumulative solved number

<table><tr><td rowspan="2">Method (Llama) k</td><td colspan="4">Pass@k (%)</td></tr><tr><td>1</td><td>2</td><td>8</td><td>256</td></tr><tr><td></td><td>AMC23</td><td></td><td></td><td></td></tr><tr><td>NAPO02</td><td>40.50</td><td>47.68</td><td>57.82</td><td>80.83</td></tr><tr><td>GRPO</td><td>37.55</td><td>47.29</td><td>59.89</td><td>80.00</td></tr><tr><td></td><td>AIME2025</td><td></td><td></td><td></td></tr><tr><td>NAPO02</td><td>0.06</td><td>0.11</td><td>0.45</td><td>13.33</td></tr><tr><td>GRPO</td><td>0.17</td><td>0.33</td><td>1.25</td><td>20.00</td></tr><tr><td></td><td>AIME24</td><td></td><td></td><td></td></tr><tr><td>NAPO02</td><td>4.75</td><td>6.67</td><td>11.96</td><td>37.78</td></tr><tr><td>GRPO</td><td>7.11</td><td>10.41</td><td>16.15</td><td>33.33</td></tr><tr><td></td><td>Math-Beyond</td><td></td><td></td><td></td></tr><tr><td>NAPO02</td><td>5.90</td><td>7.32</td><td>9.68</td><td>17.50</td></tr><tr><td>GRPO</td><td>4.93</td><td>6.39</td><td>8.92</td><td>16.21</td></tr></table>

Table 6: Pass@k comparison across datasets for Llama-3.1-8B-Instruct at temperature $T = 1 . 0 .$ Values are percentages. The row corresponding to NAPO02 is highlighted in gray, and the best value between GRPO and NAPO02 within each dataset block and column is shown in bold. Due to limited computational resources, we evaluate the checkpoint at step 100.
<table><tr><td>Transition</td><td> $\mathrm { N { \to } P / N { \to } N }$ </td><td> ${ \mathfrak { m s k } } \ N { \to } \mathrm { P }$ </td><td> $\mathsf { m s k N { \to } N }$ </td><td>∆msk</td></tr><tr><td> $1  2$ </td><td>20/133</td><td>1.1044</td><td>1.0740</td><td>+0.0304</td></tr><tr><td> $2  3$ </td><td>20/135</td><td>1.0997</td><td>1.0710</td><td>+0.0287</td></tr><tr><td> $3  4$ </td><td>18/144</td><td>1.0968</td><td>1.0679</td><td>+0.0289</td></tr><tr><td> $4  5$ </td><td>25/138</td><td>1.0941</td><td>1.0710</td><td> $+ 0 . 0 2 3 1$ </td></tr></table>

Table 7: Transition-level entropy statistics for Qwen3-4B-Instruct on LiveCodeBench at $\alpha = 0 . 5$ . The second column reports the N→P and N→N prompt counts, and the remaining columns report their mean msk values and difference; positive differences indicate higher early relative entropy for N→P.

## D Theoretical Derivation: Fisher Information and Negative-Augmented Policy Optimization

In this section, we provide a theoretical justification for the proposed driver-based weighting scheme. We demonstrate that penalizing negative samples with uniform advantage (standard GRPO) wastes optimization budget on samples that provide negligible Fisher Information. We then derive Negative-Augmented Policy Optimization (NAPO), where the advantage is proportional to the Fisher Information of the trajectory, effectively maximizing the information gain of the policy update.

## D.1 Fisher Information of Negative Samples

Consider a prompt x with a rollout group $\mathcal { G } ^ { - } =$ $\{ y _ { i } \} _ { i = 1 } ^ { G }$ where all rollouts are negative, i.e., $r ( x , y _ { i } ) = - 1$ . In policy optimization, the Fisher Information Matrix (FIM) F(θ) measures the curvature of the KL divergence and the informativeness of a sample gradient. For a specific trajectory y, the contribution to the FIM is given by the outer product of the score function:

$$
\mathbf { F } _ { y } ( \theta ) = \nabla _ { \theta } \log \pi _ { \theta } ( y | x ) \nabla _ { \theta } \log \pi _ { \theta } ( y | x ) ^ { \top } .\tag{32}
$$

A sample y is considered informative for optimization if its gradient vector has a large magnitude in the parameter space, indicating a significant potential to reshape the policy distribution.

We analyze the Fisher Information at the token level. Let $\mathbf { p } _ { t } ~ = ~ \pi _ { \theta } ( \cdot | x , y _ { < t } )$ be the probability vector over the vocabulary V at step t. Let $\mathbf { F } _ { t }$ denote the Fisher Information contribution from position t.

Lemma D.1 (Fisher Information and Entropy). Let $\begin{array} { r } { H _ { t } = - \sum _ { k } p _ { k } } \end{array}$ log $p _ { k }$ be the entropy of the distribution at step t. The trace of the token-level Fisher Information matrix satisfies:

$$
\mathrm { T r } ( \mathbf { F } _ { t } ) = 1 - \sum _ { k \in V } p _ { k } ^ { 2 } .\tag{33}
$$

Furthermore, for a fixed vocabulary size V, the quantity $1 - \textstyle \sum p _ { k } ^ { 2 }$ and the entropy $H _ { t }$ are positively associated: both are maximized by the uniform distribution (at $\textstyle { \frac { V - 1 } { V } }$ and log V , respectively) and both equal zero onlyfor deterministic distributions. In particular, $H _ { t } \to 0$ implies $\mathrm { T r } ( \mathbf F _ { t } )  0$

Proof. The score function for a sampled token $y _ { t }$ is $\nabla _ { \theta }$ log $\pi _ { \theta } ( y _ { t } | x , y _ { < t } )$ . Under the common softmax parameterization $\begin{array} { r } { p _ { k } = \frac { e ^ { z _ { k } } } { \sum e ^ { z _ { k } } } } \end{array}$ , the gradient of the log-probability with respect to the logits z is:

$$
\nabla _ { \mathbf { z } } \log p ( y _ { t } ) = \mathbf { e } _ { y _ { t } } - \mathbf { p } _ { t } ,\tag{34}
$$

where $\mathbf { e } _ { y _ { t } }$ is the one-hot vector for the sampled token. The Fisher Information matrix is the covariance of this gradient:

$$
\mathbf { F } _ { t } = \mathbb { E } _ { y _ { t } } \big [ ( \mathbf { e } _ { y _ { t } } - \mathbf { p } _ { t } ) ( \mathbf { e } _ { y _ { t } } - \mathbf { p } _ { t } ) ^ { \top } \big ]\tag{35}
$$

$$
\mathbf { \sigma } = \mathbb { E } [ \mathbf { e } _ { y _ { t } } \mathbf { e } _ { y _ { t } } ^ { \top } ] - \mathbf { p } _ { t } \mathbf { p } _ { t } ^ { \top } .\tag{36}
$$

Since $\mathbb { E } [ \mathbf { e } _ { y _ { t } } \mathbf { e } _ { y _ { t } } ^ { \top } ] = \mathrm { d i a g } ( \mathbf { p } _ { t } )$ , the trace of the FIM is:

$$
\mathrm { T r } ( \mathbf { F } _ { t } ) = \mathrm { T r } \big ( \mathrm { d i a g } ( \mathbf { p } _ { t } ) \big ) - \mathrm { T r } ( \mathbf { p } _ { t } \mathbf { p } _ { t } ^ { \top } )\tag{37}
$$

$$
= \sum _ { k } p _ { k } - \sum _ { k } p _ { k } ^ { 2 } = 1 - \sum _ { k } p _ { k } ^ { 2 } .\tag{38}
$$

The quantity $G : = 1 - \textstyle \sum p _ { k } ^ { 2 }$ is the Gini impurity (equivalently, the Tsallis entropy of order 2). Both G and the Shannon entropy $H = - \sum p _ { k }$ log p<sub>k</sub> are Schur-concave functions of the probability vector: they are both uniquely maximized by the uniform distribution and both equal zero if and only if the distribution is deterministic. While G and H are not in general strict monotonic functions of each other for $V \geq 3 .$ , they are tightly coupled via the Rényi entropy ordering $H _ { 1 } \geq H _ { 2 }$ , which gives:

$$
1 - \sum _ { k } p _ { k } ^ { 2 } \ \leq \ 1 - e ^ { - H } .
$$

In particular, $H  0$ forces $\mathrm { T r } ( \mathbf { F } _ { t } )  0$ , and conversely the uniform distribution uniquely maximizes both quantities. Hence, higher-entropy distributions correspond, in a strong qualitative sense, to larger local Fisher information. □

## D.2 Analysis of Standard NSR on Unsolved Groups

Standard NSR assigns uniform negative advantages $A ( y ) \approx - 1$ to all samples in an unsolved group. From the perspective of Fisher Information, this implies that the optimization update treats samples with zero information (confident errors) and high information (uncertain errors) equally.

Proposition D.2 (Gradient Saturation and Fisher Information). Consider a negative sample y generated from a low-entropy (confident) policy state.

1. Zero Information: If $\begin{array} { r l r } { H _ { t } } & { { } \to } & { 0 , } \end{array}$ then $\mathrm { T r } ( \mathbf F _ { t } )  0$ . The gradient $\nabla _ { \theta } \log \pi _ { \theta } ( y _ { t } )$ vanishes.

2. Saddle Points: Negative samples generated with high confidence correspond to “saddle points” or “dead ends” in the likelihood landscape. Uniform penalization $( A = - 1 )$ attempts to push probability mass away from these points, but the vanishing Fisher Information implies that $\Delta \theta \approx 0$ regardless ofthe advantage magnitude.

Thus, uniform advantage assignment wastes gradient budget on samples that lie in low-information regions of the parameter space.

Proof. The gradient vector for the sampled token y<sub>t</sub> is $\mathbf { g } _ { t } = \mathbf { e } _ { y _ { t } } - \mathbf { p } _ { t }$ , where $\mathbf { e } _ { y _ { t } }$ is the one-hot encoding of the sampled token and p<sub>t</sub> is the probability vector $\pi _ { \boldsymbol { \theta } } ( \cdot | \boldsymbol { x } , \boldsymbol { y } _ { < t } )$ . The squared norm of this gradient is:

$$
\| \mathbf { g } _ { t } \| _ { 2 } ^ { 2 } = \| \mathbf { e } _ { y _ { t } } - \mathbf { p } _ { t } \| _ { 2 } ^ { 2 }\tag{39}
$$

$$
= ( 1 - \pi ( y _ { t } ) ) ^ { 2 } + \sum _ { v \ne y _ { t } } \pi ( v ) ^ { 2 } .\tag{40}
$$

$\mathrm { A s } ~ \pi ( y _ { t } ) \to ~ 1$ (confidence approaches certainty, entropy $H _ { t } \to 0 )$ , the first term approaches 0. Simultaneously, the mass of the probability vector concentrates on y<sub>t</sub>, forcing $\pi ( v )  0$ for all other v. Thus, $\| \mathbf { g } _ { t } \| _ { 2 } \to 0$

This implies that confidently generated tokens (which characterize the “dead-end” negative rollouts in Section 3) contribute negligible gradient signal. Uniformly weighting such samples wastes optimization budget on saturated gradients that cannot effectively redirect the policy. □

## D.3 Derivation of NAPO

To maximize the efficiency of the update, we should allocate the “negative learning budget” to the samples whose penalization best reduces the total likelihood of failed trajectories. We formalize this as a constrained optimization over per-rollout weights and show that the solution naturally connects to Fisher Information and, through it, to trajectory entropy.

Exact formulation via the Gram matrix. Let $\mathcal G ^ { - } = \{ y _ { i } \} _ { i = 1 } ^ { G }$ be a fully negative rollout group for prompt x. Write $g _ { y } = \nabla _ { \theta } \log \pi _ { \theta } ( y \mid x )$ for the trajectory-level score vector, and define the Gram matrix $\dot { \mathbf { G } } \in \mathbb { R } ^ { G \times G }$ with entries

$$
G _ { i j } = g _ { y _ { i } } ^ { \top } g _ { y _ { j } } .\tag{43}
$$

Given a weight vector $\boldsymbol { w } = ( w _ { 1 } , \ldots , w _ { G } ) ^ { \top }$ , the parameter update is $\begin{array} { r } { \delta = - \eta \sum _ { i } w _ { i } g _ { y _ { i } } } \end{array}$ . The total first-order reduction in the log-likelihood of every negative rollout in the group is

$$
R ( w ) = - \sum _ { j = 1 } ^ { G } g _ { y _ { j } } ^ { \top } \delta = \eta \mathbf { 1 } ^ { \top } \mathbf { G } w .\tag{44}
$$

We seek the weight vector that maximizes $R ( w )$ subject to a variance budget $\lVert \boldsymbol { w } \rVert ^ { 2 } \leq B$ (which controls the variance of the stochastic gradient and ensures stable updates):

$$
\operatorname* { m a x } _ { w } \mathbf { 1 } ^ { \top } \mathbf { G } w \qquad \mathrm { s . t . } \qquad \| w \| ^ { 2 } \leq B .\tag{45}
$$

By the Cauchy–Schwarz inequality, $\mathbf { 1 } ^ { \top } \mathbf { G } w \leq$ $\| \mathbf { G } ^ { \top } \mathbf { 1 } \| \| w \|$ , with equality when $w \propto \mathbf { G } ^ { \top } \mathbf { 1 }$ . Since G is symmetric, the exact solution is

$$
\begin{array} { c } { { \displaystyle { w ^ { * } \propto \mathbf { G } \mathbf { 1 } } , } } \\ { { \displaystyle { w ^ { * } ( y _ { i } ) \propto \left[ \mathbf { G } \mathbf { 1 } \right] _ { i } = \sum _ { j = 1 } ^ { G } g _ { y _ { i } } ^ { \top } g _ { y _ { j } } } . } } \end{array}\tag{46}
$$

Computing G 1 exactly is impractical (it requires full parameter-space inner products between all pairs of rollout gradients in a multibillionparameter model). We therefore seek a tractable diagonal proxy and state a structural condition under which the ranking induced by G 1 coincides with the ranking induced by the diagonal alone.

Uniform-correlation condition. We decompose each entry of G 1:

$$
\big [ \mathbf { G 1 } \big ] _ { i } \ = \ \underbrace { \| g _ { y _ { i } } \| ^ { 2 } } _ { = \mathrm { T r } ( \mathbf { F } _ { y _ { i } } ) } + \ \sum _ { j \neq i } g _ { y _ { i } } ^ { \top } g _ { y _ { j } } .
$$

The diagonal term is exactly the trajectory-level Fisher Information; the off-diagonal sum captures cross-sample interference. Directly computing the latter is as expensive as the full Gram matrix. We therefore introduce the following structural assumption.

Assumption D.3 (Uniform pairwise correlation). There exists a constant $c \geq 0$ such that, for all distinct rollouts $y _ { i } , y _ { j } \in \mathcal { G } ^ { - }$

$$
g _ { y _ { i } } ^ { \top } g _ { y _ { j } } \approx c \| g _ { y _ { i } } \| \| g _ { y _ { j } } \| .
$$

It states that all same-prompt rollout gradients are roughly equally correlated with each other—a plausible condition because they share the same prompt, model architecture, and sampling distribution, so no single pair of rollouts is systematically more aligned than any other. The discriminating factor across rollouts is then their gradient magnitude, not their mutual orientation.

Theorem D.4 (Optimality of Fisher-Weighted Advantage). Under Assumption D.3, the optimal weight $w ^ { \ast } ( y )$ from Eq. (46) is a monotonically increasing function of $\begin{array} { r c l } { \operatorname { T r } ( \mathbf { F } _ { y } ) } & { = } & { \| g _ { y } \| ^ { 2 } } \end{array}$ . By Lemma D.1 and the token-independence approximation, this quantity is positively associated with the trajectory entropy $\textstyle \sum _ { t } H _ { t } .$ . Therefore, we define the Entropy-Weighted Advantage:

$$
A _ { \mathrm { N A P O } } ( y ) = - \frac { d ( y ) } { \frac { 1 } { | \mathcal { G } | } \sum _ { y ^ { \prime } \in \mathcal { G } } d ( y ^ { \prime } ) } ,\tag{47}
$$

where $d ( y )$ is the driver score (normalized entropy) defined in Eq. (6) of the main text.

Proof. Step 1: Exact solution (Gram matrix). The constrained optimum of Eq. (45) is $w ^ { \ast } \propto \mathbf { G } \mathbf { 1 }$ as derived in Eq. (46).

Step 2: Reduction under uniform correlation. Let $\begin{array} { r } { S = \sum _ { j = 1 } ^ { G } \| g _ { y _ { j } } \| } \end{array}$ denote the total gradient norm across the group. Under Assumption D.3:

$$
\begin{array} { l } { { \displaystyle \left[ { \bf G 1 } \right] _ { i } = \| g _ { y _ { i } } \| ^ { 2 } + c \| g _ { y _ { i } } \| \sum _ { j \ne i } \| g _ { y _ { j } } \| } } \\ { ~ } \\ { { \displaystyle ~ = \| g _ { y _ { i } } \| ^ { 2 } + c \| g _ { y _ { i } } \| \left( S - \| g _ { y _ { i } } \| \right) } } \\ { { \displaystyle ~ = ( 1 - c ) \| g _ { y _ { i } } \| ^ { 2 } + c S \| g _ { y _ { i } } \| } . } \end{array}\tag{48}
$$

Define $f ( u ) = ( 1 - c ) u ^ { 2 } + c S$ u for $u \geq 0$ . Its derivative is $f ^ { \prime } ( u ) = 2 ( 1 - c ) u + c S$ . Since $0 \leq$ $c \leq 1$ (cosine similarity is bounded by one) and $S \geq 0$ , we have $f ^ { \prime } ( u ) \geq 0$ for all $u \geq 0$ . Hence f is monotonically increasing in $u = \| g _ { y _ { i } } \|$ , and therefore also in $\| g _ { y _ { i } } \| ^ { 2 } = \operatorname { T r } ( \mathbf { F } _ { y _ { i } } )$

The ranking of $w ^ { \ast } ( y )$ across the group thus coincides with the ranking of $\mathrm { T r } ( \mathbf { F } _ { y } )$

Step 3: Connection to trajectory entropy. Since $\begin{array} { r } { g _ { \boldsymbol { y } } \ = \ \sum _ { t = 1 } ^ { T } \nabla _ { \boldsymbol { \theta } } } \end{array}$ log $\pi _ { \theta } ( y _ { t } | x , y _ { < t } )$ , under the approximation that token-level score vectors are approximately uncorrelated across positions we have:

$$
\| g _ { y } \| ^ { 2 } \approx \sum _ { t = 1 } ^ { T } \bigl \| \nabla _ { \theta } \log \pi _ { \theta } ( y _ { t } | x , y _ { < t } ) \bigr \| ^ { 2 } .
$$

Taking the expectation over the sampled token at each position and applying Lemma D.1:

$$
\mathbb { E } \big [ \| g _ { y } \| ^ { 2 } \big ] \approx \sum _ { t = 1 } ^ { T } \mathrm { T r } ( \mathbf { F } _ { t } ) = \sum _ { t = 1 } ^ { T } \Big ( 1 - \sum _ { k } p _ { t , k } ^ { 2 } \Big ) ,
$$

which is positively associated with $\textstyle \sum _ { t } H _ { t }$ by Lemma D.1.

The driver $d ( y )$ in Eq. (6) of the main text is a normalized measure of early-stage entropy and thus serves as a tractable proxy for $\mathrm { T r } ( \mathbf { F } _ { y } )$ . The advantage assignment $A _ { \mathrm { { N A P O } } }$ therefore implements an approximation to the optimal Fisher-weighted update, prioritizing samples where the policy is “plastic” (high entropy) over samples where it is rigid (low entropy). □

Remark D.5 (Interpretation as a diagonal preconditioner). The uniform-correlation reduction plays the role of a diagonal preconditioner for the Gram matrix. Just as diagonal preconditioning of a full Newton step retains the correct coordinate-wise scaling while ignoring off-diagonal curvature, ranking by $\mathrm { T r } ( \mathbf { F } _ { y } )$ retains the correct per-rollout scaling while treating cross-rollout interactions as uniform. When the uniform-correlation assumption is violated (e.g., if two rollouts share unusually long common prefixes), the proxy may over- or underweight individual rollouts, but the overall ranking remains a principled first-order approximation.

Remark D.6 (Relation to orthogonality). Setting $c = 0$ recovers the special case of orthogonal score vectors, for which $w ^ { * } ( y ) \propto \| g _ { y } \| ^ { 2 }$ exactly. However, same-prompt rollouts typically share early tokens and pass through shared model parameters, making $c > 0$ the realistic regime. The uniformcorrelation condition accommodates this shared structure while still yielding a tractable ranking criterion.

Remark D.7 (Practical implementation). In practice, we approximate the theoretically optimal weighting via monotonic min–max rescaling, which preserves the sample ordering, and retain the PPO/GRPO clipping mechanism for stable policy updates.