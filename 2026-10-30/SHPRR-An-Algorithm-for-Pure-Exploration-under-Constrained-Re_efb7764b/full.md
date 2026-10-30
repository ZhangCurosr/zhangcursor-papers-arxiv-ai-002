# SHPRR: An Algorithm for Pure Exploration under Constrained Regret

Georgina Nouli<sup>1</sup>, Flore Sentenac<sup>2</sup>, Maximilian Schifer<sup>2</sup>

<sup>1</sup>School of Management, Technical University of Munich, Munich, Germany <sup>2</sup> HEC Paris, Jouy-en-Josas, France

## Abstract

Best Arm Identification (BAI) conventionally measures the cost of experimentation by the number of samples collected. In many operational settings, however, the cost of exploration relative to a trusted baseline already in place is equally important. We study BAI under both a fixed sampling horizon and a regret budget. Unlike generic resources considered in constrained pure exploration, regret has a distinctive structure: sampling better arms is less costly relative to the baseline. We theoretically show that the regret cost of exploration decreases across the phases of Sequential Halving, a widely used pure-exploration algorithm. This observation yields improved guarantees on the probability of identifying the best arm and motivates a more efective allocation of the regret budget. We propose Sequential Halving with Proportional Regret Rationing (SHPRR), which front-loads the regret budget to allow more exploration when many arms remain. Experiments on synthetic and real-world instances show that SHPRR improves best-arm identification performance over uniform regret allocation and existing baselines.

## 1 Introduction

Pure exploration in multi-armed bandits, formalized as Best Arm Identification (BAI), has established foundationa paradigms for sample-eficient decision-making under uncertainty (Kalyanakrishnan et al. 2012; Garivier and Kaufmann 2016; Audibert and Bubeck 2010; Karnin, Koren, and Somekh 2013). In classical settings—such as ofline algorithm selection or clinical pre-screening—the cost of learning is measured naturally in sample complexity: every pull incurs an identical, uniform cost, and the goal is to identify the optimal arm using as few observations as possible.

However, in many live deployment settings, experimentation does not occur in a vacuum. Instead, it runs directly against a trusted, operational baseline, e.g., a production recommender or a standard-of-care treatment. Here, sampling an inferior candidate inflicts an immediate, tangible penalty on the live system. While sample-based bounds remain invaluable for ofline evaluation, live deployments demand an explicit tolerance for cumulative loss. An online A/B test, for instance, cannot simply run for a fixed sample quota if cumulative degradation against the production system exceeds a strict threshold. Accordingly, regret evolves from a post-hoc evaluation metric into a scarce, active budget constraint that directly governs the exploration process in such settings.

While BAI with resource constraints has been studied (Li and Cheung 2024), budgeting realized regret introduces two structural challenges that set it apart from standard exogenous cost models (e.g., financial or computational limits): i) Signed consumption: outperforming the baseline yields a negative cost, efectively replenishing the remaining budget rather than draining it. ii) Consumption–reward coupling: consumption is not an externally sampled attribute but the recentered reward process itself. Hence, better options are less costly, a phenomenon that can be leveraged in the exploration.

Sequential Halving (SH) (Karnin, Koren, and Somekh 2013) provides a natural foundation for phase-based exploration under resource constraints. However, uniform regret allocations across phases fail to exploit a critical structural dynamic of elimination-based algorithms: as suboptimal arms are pruned, the average suboptimality gap of the surviving set contracts. Because early phases must sample from a broader, unrefined pool of arms, the expected regret cost per pull is highest at the start of the algorithm and steadily decreases across phases.

This observation directly motivates our proposed algorithm, Sequential Halving with Proportional Regret Rationing (SHPRR). Rather than distributing the regret budget uniformly, SHPRR strategically front-loads the budget toward earlier phases, where information acquisition incurs the highest regret in proportion to the number of surviving arms. Controlled by a single tuning parameter α ∈ [0, 1], SHPRR smoothly interpolates between classical uniform allocation (α = 0) and aggressive early-phase budget prioritization (α = 1), while guaranteeing budget enforcement. Across extensive synthetic and real-world benchmarks, SHPRR consistently matches or outperforms Sequential Halving with Resource Rationing (SH-RR), AT-LUCB, and Top-Two SPRT (TTSPRT). Crucially, while competing baselines demonstrate brittle performance, SHPRR remains robust across all settings.

Our contributions are summarized as follows: we formalize BAI under a realized regret budget, capturing real-world deployment settings where cumulative underperformance against a baseline—rather than sample count—is the critical resource. We characterize two fundamental structural properties that distinguish regret budgeting from standard resource constraints: signed consumption and consumption– reward coupling. Furthermore, we reveal that eliminationbased algorithms exhibit a contracting average suboptimality gap across phases, establishing that early-phase exploration is inherently the most expensive in terms of regret.

Guided by this phase-contraction insight, we propose SHPRR, which strategically front-loads the regret budget toward early phases in proportion to the active arm count, controlled by a single parameter $\alpha \in [ 0 , 1 ]$

We provide theoretical guarantees showing that SHPRR controls failure probability under bounded regret. Empirically, SHPRR consistently matches or outperforms SH-RR, AT-LUCB, and TTSPRT across synthetic and real-world environments. Crucially, while existing baselines display brittle behavior, performing well in specific settings but collapsing in others, SHPRR maintains strong, robust performance across all domains.

## 2 Related Work

Literature on BAI follows one of two formulations: fixedconfidence (PAC) BAI, where the algorithm must identify the best arm with probability at least 1−δ while minimizing pulls (Kalyanakrishnan et al. 2012; Garivier and Kaufmann 2016; Kaufmann, Cappé, and Garivier 2016), and fixed-budget BAI, where a limited budget T is used to maximize the probability of correct identification (Audibert and Bubeck 2010; Karnin, Koren, and Somekh 2013; Gabillon, Ghavamzadeh, and Lazaric 2012; Carpentier and Locatelli 2016). Identification quality is measured via Success Probability (SP), or via simple regret – the expected gap between the best arm’s mean and that of the recommended arm (Bubeck, Munos, and Stoltz 2009, 2011; Zhao et al. 2023).

Elimination and Confidence-Bound Methods: SH (Karnin, Koren, and Somekh 2013) and Successive Rejects (Audibert and Bubeck 2010) allocate an increasing number of pulls per arm across rounds while discarding the weakest arms. Carpentier and Locatelli (2016) established a matching lower bound for the latter, closing the gap with existing upper bounds, while Barrier, Garivier, and Stoltz (2023) study fixed-budget BAI in non-parametric models, sharpening these bounds via KL-based large deviation arguments. Confidence-bound methods instead use interval estimates to guide sampling and stopping: LUCB (Kalyanakrishnan et al. 2012) pulls the empirical best arm together with its most optimistic competitor, KL-LUCB (Kaufmann and Kalyanakrishnan 2013) and lil’UCB (Jamieson et al. 2014) sharpen the underlying confidence intervals, and Track-and-Stop (Garivier and Kaufmann 2016) instead tracks the sampling proportions prescribed by the instance-dependent lower bound, attaining asymptotically optimal sample complexity. Jun and Nowak (2016) extended LUCB to the anytime setting (AT-LUCB), while UGapE (Gabillon, Ghavamzadeh, and Lazaric 2012) unifies the confidence-bound principle across the fixed-budget and fixed-confidence settings.

Top-Two Methods: Russo (2016) introduced Top-Two Thompson Sampling (TTTS), which maintains a posterior over each arm’s mean and performs randomized sampling between the current best arm and a challenger; Shang et al. (2020) and Jourdan et al. (2022) later established δ-PAC guarantees and sample complexity bounds for TTTS variants.

Mukherjee and Tajer (2023) adopt a related top-two structure but replace posterior sampling with Sequential Probability Ratio Test (SPRT) likelihood-ratio statistics to select arms and decide when to stop, yielding a δ-PAC algorithm that is asymptotically optimal for exponential-family rewards without the computational cost of posterior sampling. None of these classical formulations account for the cost of arm pulls, feasibility constraints, or the regret incurred while exploring, motivating the lines of work surveyed next.

BAI with Costs, Constraints and Cumulative Regret: BAI with safety constraints imposes per-pull feasibility requirements (Wang, Wagenmaker, and Jamieson 2022; Sui et al. 2015), while variance-constrained BAI (Hou, Tan, and Zhong 2022) instead constrains the variance of the identified arm. Poiani, Metelli, and Restelli (2022) and Wang et al. (2023) study multi-fidelity BAI, where biased approximations of each arm’s reward are available at varying known costs, and the goal is to minimize total identification cost while guaranteeing correct identification with high probability. Kanarios, Zhang, and Ying (2024) study Cost-Aware BAI (CABAI), identifying the highest-reward arm at minimum expected cost, whereas Qin et al. (2025) target the highest reward-tocost ratio using the fewest samples. Lardy, Katsimerou, and Koolen (2026) study BAI constrained to the subset of arms with costs under a specific threshold when reward and cost are dependent. Harding and Kandasamy (2026) minimize a risk functionaljointly penalizing a performance criterion and cumulative sampling cost, avoiding the need to pre-specify a budget or confidence level and Yang, Tan, and Jin (2026) couple identification quality with the cumulative regret incurred during exploration. The closest setting to ours is Li and Cheung (2024), who impose cumulative resource-consumption constraints across arm pulls, and develop SH-RR, an extension of SH that eliminates arms in phases while uniformly rationing each phase’s budget given arm costs. Our setting difers from Li and Cheung (2024) in two respects. First, consumption is signed: because it is an arm’s regret relative to a baseline, outperforming the baseline yields a negative cost that replenishes the budget rather than draining it, unlike the strictly non-negative costs assumed there. Second, consumption is coupled to the reward process: it is not an externally sampled attribute but the recentered reward itself, rendering better arms cheaper to explore.

## 3 Problem Setting

We consider a stochastic multi-armed bandit problem augmented with an exogenous baseline and a realized regret budget, in addition to the usual sampling horizon.

## 3.1 Bandit Model

Let $S = \{ 1 , \ldots , K \}$ be a finite set of K arms available to the agent. Pulling arm $i \in S$ at round t yields a stochastic reward $r _ { t } \sim \nu _ { i }$ , bounded almost surely in [0, 1], with mean $\mu _ { i } =$ $\mathbb { E } [ r _ { t } ]$ . We assume a unique arm with highest mean reward, and wlog that $\mu _ { 1 } > \mu _ { 2 } \ge . . . \ge \mu _ { K }$ . We denote by $\mu ^ { \star } = \mu _ { 1 }$ the mean of the optimal arm, and write $\Delta _ { i } = \mu ^ { \star } - \mu _ { i } \geq 0$ for the suboptimality gap of arm i. The distributions $\nu _ { i }$ , and in particular the means $\mu _ { i } .$ , are unknown to the agent for every $i \in S$ at the beginning of the run.

In addition to S, there is a baseline arm $k _ { 0 } \notin S .$ , representing an incumbent policy or status-quo action against which arms in $S$ are evaluated. Unlike the arms in $S ,$ , the baseline arm is not available to the agent: it is never pulled, and its reward distribution $\nu _ { 0 }$ is never observed. Only its mean $\mu _ { 0 } = \mathbb { E } _ { \nu _ { 0 } } [ r ]$ is known, provided as a fixed reference point. We assume $\mu _ { 0 } \leq \mu ^ { \star } , \mathrm { i . e . }$ ., the baseline is weakly dominated by the best available arm, reflecting its role as an incumbent reference rather than an unattainable ideal.

A policy π sequentially selects arms, with $k _ { t }$ being the arm selected at time t, and, at stopping time τ , outputs a recommendation $\phi \in S$ . Formally, let $\mathcal { \bar { F } } _ { t } = \sigma ( k _ { 1 } , \bar { r } _ { 1 } , \dots , k _ { t } , r _ { t } )$ be the filtration generated by the history of pulls and observed rewards, and let $\tau$ be a stopping time with respect to $( \mathcal { F } _ { t } ) _ { t \geq 0 }$ . At each round $t \leq \tau$ , π selects $k _ { t }$ as a function of $\mathcal { F } _ { t - 1 } ;$ ; at τ, π outputs ϕ as a function of $\mathcal { F } _ { \tau }$

Realized baseline regret. When the agent pulls arm $k _ { t }$ at round t and observes reward $r _ { t } ,$ it incurs a realized baseline regret $\delta _ { t } ~ = ~ \mu _ { 0 } - r _ { t }$ , measuring the shortfall of the pulled arm’s reward relative to the baseline mean. Because $\delta _ { t }$ is defined against the realized reward $r _ { t }$ rather than the arm mean $\mu _ { k _ { t } } ,$ it is a noisy, round-level quantity that can be negative whenever $r _ { t }$ exceeds $\mu _ { 0 }$ , at which point the pull replenishes rather than depletes the budget. The cumulative realized regret after n pulls is

$$
\mathcal { R } _ { n } \ = \ \sum _ { t = 1 } ^ { n } \bigl ( \mu _ { 0 } - r _ { t } \bigr ) , \qquad \mathcal { R } _ { 0 } = 0 .\tag{1}
$$

Constraints. The agent operates under two simultaneous constraints. First, a regret budget $B \geq 0$ bounds the cumulative realized regret at every pull along the trajectory:

$$
\mathcal { R } _ { t } \leq B , \forall t \leq \tau .\tag{2}
$$

Second, a pull budget $T \in \mathbb { N }$ caps the total number of arm pulls, so that $\tau \leq T$ . The pull budget is necessary for the problem to be well posed: since $\delta _ { t }$ can be negative, an agent that repeatedly pulls a high-reward arm can drive $\mathcal { R } _ { n }$ to be non-increasing, so the regret constraint alone would permit indefinite sampling. The two constraints together guarantee a genuine trade-of between information gained and risk incurred, even when the regret budget does not bind.

Objective. The goal of π is to maximize the probability of exactly identifying the best arm, subject to both constraints:

$$
\operatorname* { m a x } _ { \pi } ~ \mathbb { P } \big ( \phi = 1 \big ) \mathrm { ~ s . t . ~ } ~ \mathcal { R } _ { t } \leq B , \forall t \leq \tau , ~ \tau \leq T .\tag{3}
$$

## 4 Structural Properties of SH

SH is a widely used elimination algorithm for fixed-budget BAI. For simplicity, suppose throughout this section that K is a power of 2 and that there is a margin of safety in the budget to complete the last round-robin cycle. This standard simplification avoids the clutter associated with floors, ceilings, and incomplete round-robin cycles, without afecting the leading-order terms in the resulting bounds. Starting from the full set of arms, SH proceeds over $\log _ { 2 } ( K )$ phases. In each phase, the active arms are sampled in round-robin fashion, and the half with the largest empirical means is retained. Throughout, denote by $n _ { q }$ the number of arms in each phase $q ,$ , and, when $n _ { q } = 2$ , use the convention $n _ { q } / 4 + 1 = 2$ and $\Delta _ { 2 K } = 1$

This phase structure admits a natural extension to BAI under resource constraints when the consumption associated with each pull is nonnegative. A pull quota and a resource quota are assigned to each phase. The active arms are then sampled until one of these quotas binds, after which the usual SH elimination step is performed. This is the principle underlying SH-RR introduced by Li and Cheung (2024).

A conservative reduction to nonnegative consumption. Our regret-budgeted problem can be embedded into this generic framework. For each arm $i \in [ K ]$ we can define the positive-part regret consumption $D _ { i } ^ { + }$ as the random variable with distribution $( \mu _ { 0 } - r _ { i } ) _ { + }$ , with $r _ { i } \sim \nu _ { i }$ As the positive-part regret consumption dominates the regret consumption, any algorithm that respects a budget B for cumulative positive-part regret also respects the original signed-regret constraint. Applying SH-RR with $D _ { i } ^ { + }$ as the resource consumption therefore provides an immediate feasible benchmark for our problem.

To state the resulting guarantee, let $d _ { i } ^ { + } : = \mathbb { E } [ D _ { i } ^ { + } ]$ , and let $d _ { ( 1 ) } ^ { + } \geq \cdot \cdot \cdot \geq d _ { ( K ) } ^ { + }$ denote these expected consumptions in nonincreasing order. Define

$$
f ( d ) : = \left\{ \begin{array} { l l } { \displaystyle \frac { 2 } { \log ( 1 / d ) } , } & { 0 < d < e ^ { - 2 } , } \\ { \displaystyle e ^ { 2 } d , } & { e ^ { - 2 } \leq d \leq 1 \mathrm { o r } d = 0 , } \end{array} \right.
$$

together with the complexity terms

$$
H _ { 2 , T } : = \operatorname* { m a x } _ { k > 1 } \frac { k } { \Delta _ { k } ^ { 2 } } , H _ { 2 , + } ^ { \mathrm { s t o } } : = \operatorname* { m a x } _ { k > 1 } \frac { \sum _ { j = 1 } ^ { k } f ( d _ { ( j ) } ^ { + } ) } { \Delta _ { k } ^ { 2 } } .
$$

Finally, let $\begin{array} { r } { \gamma _ { + } ~ : = ~ \operatorname* { m i n } \left\{ \frac { T } { H _ { 2 , T } } , \frac { B } { H _ { 2 , + } ^ { \mathrm { s t o } } } \right\} } \end{array}$ . Ignoring integerrounding efects, SH-RR satisfies

$$
\mathbb { P } ( \phi \neq 1 ) \leq 1 4 K \log _ { 2 } ( K ) \exp \left( - \frac { \gamma _ { + } } { 8 \log _ { 2 } ( K ) } \right) .\tag{4}
$$

Remark 4.1 In the Supplementary Material, we also analyze a phase-wise-estimate variant of SH-RR, in which each elimination decision uses only the samples collected during the current phase. This modification leaves the resourcerationing mechanism unchanged and removes the extraneous factor K from the prefactor in (4).

What the positive-part reduction misses. Bound (4) is a useful benchmark, but the reduction on which it relies deliberately discards the distinctive structure of regret.

First, a pull whose reward exceeds the baseline has zero positive-part consumption, whereas its signed consumption is negative. Such a pull replenishes the original regret budget and can finance additional exploration. This efect is entirely absent from the positive-part formulation. For example, suppose that rewards for arm i are Bernoulli and that $0 < \mu _ { 0 } < \mu _ { i } < 1$ . Then $\mathbb { E } _ { X _ { i } \sim \nu _ { i } } \big [ \mu _ { 0 } - X _ { i } \big ] = \mu _ { 0 } - \mu _ { i } < 0 ,$ whereas $\mathbb { E } _ { X _ { i } \sim \nu _ { i } } [ ( \mu _ { 0 } - X _ { i } ) _ { + } ] = \mathbf { \bar { \mu } } \mu _ { 0 } ( 1 - \mathbf { \bar { \mu } } _ { i } ) > 0$ . Thus, an arm that replenishes the signed-regret budget in expectation is nevertheless assigned a strictly positive expected cost by the positive-part reduction.

Second, generic resource-constrained analyses treat each arm’s consumption distribution as a fixed, exogenous attribute. Although the arm-level distributions are also fixed in our setting, the consumption profile of the active set changes endogenously as elimination removes poorly performing arms. In particular, the generic bound controls a phase containing $n _ { q }$ arms using the $n _ { q }$ largest efective consumptions $f ( d _ { i } ^ { + } )$ , independently of which arms have actually survived. It therefore charges each phase according to a worst-case consumption profile of the corresponding cardinality. Signed regret is more structured. Under round-robin sampling, the expected regret consumption per pull in phase $\begin{array} { r } { q \mathrm { i s } \stackrel { - 1 } { n _ { q } } \bar { \sum _ { i \in S _ { q } } } \bar { \Delta _ { i } } - \Delta _ { 0 } } \end{array}$ where $\Delta _ { 0 } = \bar { \mu ^ { \star } } - \bar { \mu _ { 0 } }$ . Consequently, controlling the average suboptimality of the surviving arms simultaneously controls the expected resource consumption of the phase.

A regret-specific structural bound. We now analyze the signed-regret process directly. Consider a phase-rationed version of SH in which phase $q$ is assigned a regret budget $B ^ { ( q ) }$ The allocation $B ^ { ( q ) }$ need not be uniform. The result below applies to any phase-budget allocation; the proportional allocation defining SHPRR is introduced in the next section.

For the theoretical analysis, elimination decisions are based on empirical means computed only from the observations collected during the current phase. Algorithm 1, which is used in our experiments, instead computes empirical means from all observations accumulated up to that phase. We analyze the phase-wise-estimate variant because it follows the standard theoretical formulation of SH and permits sharper guarantees, even though discarding past observations may degrade empirical performance. The two variants otherwise share the same sampling, stopping, and budgetrationing mechanisms.

Define, for each phase q:

$$
\underline { { \tau _ { q } } } = \operatorname* { m i n } \left( \frac { T } { n _ { q } \log _ { 2 } K } ; \frac { B ^ { ( q ) } } { 6 n _ { q } \Delta _ { 2 n _ { q } } } \right) ,
$$

which we show is a conservative lower bound in the number of pulls per arm in phase q, and

$$
\Gamma _ { 2 , R } = \operatorname* { m i n } _ { q \in [ \log _ { 2 } ( K ) ] } \underline { { \tau _ { q } } } \Delta _ { \frac { n _ { q } } { 4 } + 1 } ^ { 2 } .
$$

The central structural property is that the average gap of the surviving set improves at a controlled rate.

## Proposition 4.1 (Phase-wise Arm Improvement)

Suppose $\begin{array} { r l r } { \frac { \log 4 e \log _ { 2 } ( K ) ^ { 2 } } { \Delta _ { \frac { n _ { q } } { 4 } + 1 } ^ { 2 } } } & { \leq } & { \frac { 1 } { 8 } \tau _ { q } . } \end{array}$ Then, with probability at least:

$$
1 - 3 \log _ { 2 } ( K ) ^ { 2 } \exp \left( - \frac { \Gamma _ { 2 , R } } { 8 \log _ { 2 } ( K ) ^ { 2 } } \right) ,
$$

for every phase $q ^ { \prime } ,$ , we have:

$$
\frac { 1 } { n _ { q ^ { \prime } } } \sum _ { i \in S _ { q ^ { \prime } } } \Delta _ { i } \leq 3 \Delta _ { 2 n _ { q ^ { \prime } } } .
$$

This phase-wise improvement also yields a bound on the probability of misidentification.

Theorem 4.1 Suppose $\begin{array} { r } { \frac { \log { 4 e \log _ { 2 } ( K ) ^ { 2 } } } { \Delta _ { \frac { n _ { q } } { 4 } + 1 } ^ { 2 } } \leq \frac { 1 } { 8 } \underline { { \tau _ { q } } } . } \end{array}$ . Then:

$$
\mathbb { P } \left( \phi \neq 1 \right) \leq 7 \log _ { 2 } ( K ) ^ { 2 } \exp \left( - \frac { \Gamma _ { 2 , R } } { 8 \log _ { 2 } ( K ) ^ { 2 } } \right)
$$

Unlike the positive-part benchmark, Theorem 4.1 does not charge each phase according to a fixed worst-case consumption profile. Instead, the efective phase length adapts to the phase-specific quality scale $\Delta _ { 2 n _ { a } }$ , which decreases as suboptimal arms are eliminated. Because regret can be negative and replenish the budget, standard lower-bound techniques do not directly apply; consequently, the theorem does not fully characterize the intrinsic complexity of pure exploration under a regret constraint. It nevertheless captures the declining cost of exploration across phases, while the algorithm further carries forward any budget accumulated through negative realized regret.

This observation motivates allocating a larger share of the regret budget to the early, expensive phases. The next section introduces the proportional rationing schedule used by SHPRR.

## 5 SHPRR Algorithm

In the regret-budgeted setting, an elimination algorithm’s performance is largely determined by how it distributes its budget over time. Spending too cautiously at the start allows low-reward, high-consumption arms to survive into later phases, where each additional pull on a candidate that was never competitive drains budget needed for the discriminations that actually matter. Spending too aggressively might result in an insuficient reserve for separating the final, closely matched candidate arms. Proposition 4.1 provides the structural basis for resolving this tension: since the surviving set improves from phase to phase, the per-pull cost of information declines over the run, so the budget is most valuable, and should be concentrated, where consumption is highest, namely in the early phases. At the same time, the appropriate degree of front-loading admits no universal answer: it depends on how quickly the surviving set improves and on how costly the initial pool is, both properties of the instance rather than the algorithm. The degree of front-loading should therefore itself be a design parameter, exposed to the practitioner rather than fixed in advance.

To this end, we introduce SHPRR. Building on the SH-RR framework, SHPRR utilizes non-uniform rationing of the regret budget, controlled by a tunable parameter $\alpha \in$ [0, 1]. Unlike SH-RR, which would equally divide the regret budget across phases, SHPRR allocates it proportionally to the number of arms remaining in each phase, raised to the power α. With $n _ { q } = \lceil K / 2 ^ { q - 1 } \rceil$ , the regret budget allocated to phase q is defined as:

$$
B _ { q } = \mathcal { R } _ { q } ( K , B , \alpha ) = B \cdot \frac { n _ { q } ^ { \alpha } } { \sum _ { j = 1 } ^ { \lceil \log _ { 2 } K \rceil } n _ { j } ^ { \alpha } } .\tag{5}
$$

When $\alpha \ : = \ : 0$ , this recovers uniform allocation across phases, as in SH-RR. As α approaches 1, more resources are allocated to earlier phases, in proportion to the number of arms considered in each phase. For $0 < \alpha < 1$ , the allocation decays exponentially across phases but at a shallower rate than at $\alpha = 1$ , giving moderate preference to early phases while retaining more resources for later ones than full proportional $( \alpha = 1 )$ allocation would.

Algorithm 1 Sequential Halving with Proportional Regret   
Rationing (SHPRR)   
Require: Regret budget $B \in \mathbb { R } _ { + } .$ , total arm pull budget T,   
arm set $S$ of size K, baseline arm mean $\mu _ { 0 } .$   
1: $S ^ { ( 1 ) } \gets S , t \gets 1$   
2: $B ^ { ( 1 ) }  \mathcal { R } _ { 1 } ( K , B , \alpha ) , T ^ { ( 1 ) }  T / \lceil \log _ { 2 } K \rceil .$   
3: for $q = 1$ to $\lceil \log _ { 2 } K \rceil$ do   
4: Initialize $\tilde { I ^ { ( q ) } } \gets \dot { 0 } , H ^ { ( q ) } \gets \emptyset , t ^ { ( q ) } \gets 1 .$   
5: while $I ^ { ( q ) } \leq B ^ { ( q ) } - 1$ and $t ^ { ( q ) } \leq T ^ { ( q ) } - 1$ do   
6: Pull arm $k _ { t }$ based on round robin selection.   
7: Observe reward $r _ { t }$ and regret consumption $\delta _ { t } .$   
8: Update $I ^ { ( q ) } \gets I ^ { ( q ) } + \delta _ { t }$   
9: Update $H ^ { ( q ) } \gets H ^ { ( q ) } \cup \{ ( k _ { t } , r _ { t } ) \}$   
10: Update $t ^ { ( q ) } \gets t ^ { ( q ) } + 1 , t \gets t + 1 .$   
11: end while   
12: Use $\textstyle \bigcup _ { m = 1 } ^ { q } H ^ { m }$ to compute empirical   
means $\{ \hat { \mu } _ { k } ^ { q } \} _ { k \in S ^ { ( q ) } }$   
13: Let $S ^ { ( q + 1 ) }$ be the top $\lceil | S ^ { ( q ) } | / 2 \rceil$ arms with highest   
empirical means.   
14: Update $B ^ { ( q + 1 ) } \gets \mathcal { R } _ { q + 1 } ( K , B , \alpha ) + ( B ^ { ( q ) } - I ^ { ( q ) } )$   
15: Update $T ^ { ( q + 1 ) } \gets T \dot { / } \lceil \log _ { 2 } K \rceil + ( T ^ { ( q ) } - t ^ { ( q ) } )$   
16: end for   
17: Return the arm with the highest empirical mean in $S ^ { ( q ) }$

Algorithm 1 presents the SHPRR procedure for identifying the optimal arm under regret-based and maximum arm pull resource constraints. The algorithm proceeds over $\lceil \log _ { 2 } K \rceil$ phases, where $K$ denotes the initial number of arms. The active arm set $S ^ { ( 1 ) }$ and the time step counter are first initialized (Line 1). In Line 2, the total regret budget B is proportionally allocated across all phases according to Expression 5, while the total arm pull budget $T$ is split uniformly across phases, giving each phase an initial pull allowance $T / \lceil \log _ { 2 } K \rceil$ . Within each phase q (Lines 4–10), SHPRR samples arms from the active set $S ^ { ( q ) }$ in round-robin order, continuing until either the phase’s regret budget $B ^ { ( q ) }$ or its arm pull budget $T ^ { ( q ) }$ is exhausted, whichever binds first. Each pull’s regret consumption is measured as the deviation of the observed reward from the baseline mean $\mu _ { 0 }$ (Line 7). This quantity is negative whenever an arm outperforms the baseline, in which case the pull replenishes rather than depletes the phase’s regret budget. Once sampling stops, empirical means are computed using the full pull history accumulated across all phases so far (Line 12), and the top half of the active arms by empirical mean advance to the next phase (Line 13), following the standard SH elimination rule.

Before the next phase begins, both budgets are updated (Lines 14–15). The regret budget for phase $q + 1$ combines a fresh allocation from Expression 5, sized to the new number of active arms, with any unspent or replenished regret budget carried over from phase q; the arm pull budget is updated analogously, combining a fresh uniform allocation with any unused pulls from the previous phase. This carry-over mechanism ensures that budget conserved in early phases, whether through early elimination of costly arms or through highreward arms replenishing the budget, remains available for the more dificult discriminations of later phases, rather than being lost. The procedure terminates after $\lceil \log _ { 2 } K \rceil$ phases and returns the arm with the highest empirical mean among the final surviving arms (Line 17).

## 6 Experiments

In this section, we present our experimental design, followed by a performance overview of SHPRR against benchmark algorithms, a sensitivity analysis, and an examination of how varying the rationing parameter afects the arm composition retained at each phase.

## 6.1 Experimental Design

We evaluate our method on synthetic and real-world benchmark datasets to demonstrate its efectiveness across a range of reward structures, gap configurations, and consumption profiles. Following Li and Cheung (2024), our synthetic environments use four mean-reward configurations ofBernoulli arms (OneSub, Trap, Poly, and Geom). Additionally, we consider variations of these environments where we vary the optimal–suboptimal gap via $\mu _ { k } ^ { \prime } = \operatorname* { m a x } ( \mu _ { k } - \delta , 0 )$ for all $\bar { k \neq k ^ { \star } }$ , with $\bar { \boldsymbol \delta } \in \{ 0 , \bar { 0 } . \bar { 1 } , 0 . 2 , 0 . 3 \}$ , where $\delta = 0$ recovers Li and Cheung (2024). We define the regret consumption of each arm relative to the top 3% quantile arm, and we use a regret budget of 384 for the performance overview and 224 for the per-phase analysis, a max pull budget of 50000, and 500 seeds per configuration.

We complement these with two real-world datasets reflecting human preferences. In the New Yorker Caption Contest (Jain et al. 2020), arms are submitted captions with binary rewards reflecting whether voters found them funny, with mean reward given by the empirical positive-vote rate; we select three contests (511, 538, and 780, denoted c511, c538, and c780) with 3,000–6,000 arms each, chosen for their distinct reward distributions to cover a range of real-world cases. We use the top 1% quantile arm as baseline, a regret budget of 1.6 times the number of arms in each contest, a max pull budget of 100,000, and 500 seeds. In the Criteo Attribution Modeling for Bidding dataset (Diemert et al. 2017), arms are ad campaigns; pulling campaign i, with estimated attribution probability $\hat { p } _ { i }$ and mean conversion value $v _ { i }$ , yields $A _ { t } ^ { ( i ) }$ ∼ Bernoulli(ˆp ) and reward $r _ { t } ^ { ( i ) } = v _ { i } A _ { t } ^ { ( i ) }$ . We select this dataset for its very low expected rewards (maximum arm mean 0.015), a challenging low-signal setting. Consumption is again the regret gap to a quantile baseline, tested at the top $\bar { 5 } \%$ , top 1%, and best-arm baselines; and we sweep the regret budget from 0.06 to 1.8 times the number of arms. All Criteo experiments use a max pull budget of 100,000 and 500 random seeds, with further design details in the Supplementary Material.

![](images/91cdab6e45736020d8cae92806ed8dd26082efbfc2edd7b2182b08980cc4f966.jpg)  
Figure 1: SP and NSR across synthetic (top) and real-world (bottom) environments.

We compare SHPRR against three state-of-the-art algorithms: SH-RR, TTSPRT, and AT-LUCB. Since TTSPRT and AT-LUCB were not originally designed to respect a resource budget, we adapt both by adding budget exhaustion as an additional stopping criterion. Note that, even though SH-RR originally assumes strictly positive consumption, we evaluate all three benchmarks under our setting: each operates under the same signed regret consumption $\delta _ { t } = \mu _ { 0 } - r _ { t }$ and thus benefits equally from budget replenishment when arms outperform the baseline. We report the average SP, defined as the fraction of runs identifying the optimal arm, and the average Normalized Simple Regret (NSR), defined as $\frac { \mu ^ { * } - \mu ^ { j } } { \mu ^ { * } - \mu ^ { \mathrm { m i n } } }$ , where $\mu ^ { * } , \mu ^ { j }$ , and $\mu ^ { \mathrm { m i n } }$ denote the optimal, selected, and worst arm means respectively.

## 6.2 Performance Overview

Figure 1 reports SP and NSR for all methods across the four synthetic environments (top) and the real-world datasets (bottom), with the baseline arm set at the top 1% quantile. Overall, SHPRR frequently outperforms the state-of-the-art benchmarks across all four synthetic environments. SHPRR with higher rationing $( \alpha = 0 . 5 , 0 . 9 )$ achieves the best or near-best performance in Geom, Poly, and Trap, yielding both higher SP and lower NSR than all benchmarks. In Trap, SHPRR $( \alpha = 0 . 9 )$ attains an SP of 0.93, nearly double that of SH-RR (0.528) and AT-LUCB (0.462), while TTSPRT collapses to 0.034. A similar trend holds in Geom and Poly, where $\alpha > 0$ improves both SP and NSR. The sole exception is OneSub, a degenerate single-dominant-arm case where AT-LUCB attains near-perfect identification (SP 0.998, NSR 0.002) and all SHPRR variants and TTSPRT perform comparably (SP 0.89–0.91); here the consumption gap is homogeneous across suboptimal arms by construction, ofering no heterogeneity for rationing to exploit. Overall, SHPRR is best or near-best in every environment with non-trivial consumption structure and maintains consistently strong performance across all four environments, whereas the baseline algorithms fluctuate considerably: AT-LUCB and TTSPRT each achieve near-best results in one environment (OneSub) but degrade sharply in others (e.g., AT-LUCB falls to 0.248 in Geom, and TTSPRT falls to 0.034 in Trap), underscoring the robustness of our approach across varying consumption structures. On the real-world datasets SHPRR continues to outperform all benchmarks, though the best-performing rationing level varies more than in the synthetic environments. SHPRR $( \alpha = 0 . 5 )$ achieves the highest SP in three of the four datasets, reaching 0.526 on caption contest 511, 0.72 on 780, and 0.716 on criteo, consistently outperforming all benchmarks and substantially reducing NSR. On caption contest 538, SHPRR $( \alpha = 0 . 9 )$ performs best among all methods; however, front-loading too aggressively can result in decreased performance, as observed in contests 511 and 780.

![](images/cbd9a96aa3e87374089f2c955e561fffd165a7e02744633f7e55467497613201.jpg)  
Figure 2: SP across the four synthetic environments as $\delta$ varies from 0.3 to 0.

Importantly, although the optimal α varies by environment, a single fixed choice of $\alpha \ : = \ : 0 . 5$ already strongly outperforms every benchmark in all but one of the synthetic and real-world environments. The tunability of α is therefore not a burden: moderate rationing $( \alpha = 0 . 5 )$ serves as a strong, robust default when the reward structure is unknown, while the flexibility of α lets practitioners extract further gains when domain knowledge is available.

## 6.3 Sensitivity Analysis

Gap Magnitude. An algorithm intended for real-world deployment cannot rely on favorable conditions and must maintain stable performance across diverse instances. We therefore examine sensitivity to the optimal–suboptimal gap δ, which controls reward similarity as $\delta  0 .$ . Figure 2 shows a consistent pattern: while the baselines are competitive only when the gap is large and the problem is easy, SHPRR pulls decisively ahead as $\delta \  \ 0$ where identification becomes harder, degrading far more gracefully than every benchmark. This is clearest in Trap, where SHPRR $( \alpha = 0 . 9 )$ stays above 0.93 across the entire range, while AT-LUCB falls from 0.988 to 0.462 and TTSPRT collapses from 0.972 to 0.034. The same pattern holds in Geom, where the best SHPRR variant remains at 0.734 at $\delta = 0$ while AT-LUCB falls to 0.248 and TTSPRT to 0.132, and in Poly, where SHPRR $( \alpha = 0 . 9 )$ holds 0.938 against AT-LUCB’s 0.70 and TTSPRT’s 0.60. OneSub is the sole exception: with a single dominant arm, all methods remain near ceiling regardless of δ. Overall, these results indicate that SHPRR is the only dependable choice: the single method that maintains strong performance independent of the optimal–suboptimal gap.

![](images/5110ca8d58543a0ca9f801dc4a80ab658ddec9ce99cd3c12f5af791ea24b36da.jpg)  
Figure 3: Performance analysis across three baseline arm selection criteria (from left to right): best arm, top 1%, and top 5%, evaluated under varying budget levels.

Baseline Arm and Regret Budget. We further examine how performance varies with the choice of baseline arm that defines consumption, comparing best-arm, top 1%, and top 5% baselines on the Criteo dataset across a range of regret budgets (Figure 3). The best-arm case is the most demanding: every suboptimal arm incurs positive consumption and none can replenish the budget, recovering the strictly non-negative, non-replenishing structure of the original BAIunder-constraints formulation. Even in this subcase, SHPRR $( \alpha = 0 . 5 )$ outperforms all benchmarks steadily across budget levels, reaching the highest SP (0.77) and lowest NSR at the largest budget, while AT-LUCB trails throughout. As the baseline is relaxed to the top 1% and top $5 \%$ quantiles, an increasing number of arms outperform the baseline and thus replenish rather than consume the budget. Here SHPRR $( \alpha = 0 . 5 , 0 . 9 )$ saturates at the highest SP (0.77) and uses a smaller fraction of the budget than TTSPRT and AT-LUCB, which plateau early. This shows that SHPRR’s rationing more eficiently exploits budget-replenishing arms, converging to a high-quality solution well before the pull budget is exhausted. Across all three baselines, SHPRR is the only method that remains strong whether the budget is scarce and non-replenishing or lenient and replenishable.

## 6.4 Per Phase Arm Composition

To illustrate the efect of rationing, we track the mean reward of arms still active at each phase under the Poly reward configuration, comparing four SHPRR variants spanning the full rationing spectrum, from $\alpha = 0 \ ( \mathrm { S H \mathrm { - } R R } )$ to $\alpha = 1$ (maximal front-loading). Since arms are progressively eliminated, the mean reward of surviving arms rises over time regardless of α; the question is whether higher rationing accelerates this concentration toward high-reward arms, and does so consistently across phases.

![](images/197b1e2d2e0e546b1258fc3deee4369bd45a5544aab8c614f99b8c8a29056fc9.jpg)  
Figure 4: Phase-wise arm quality under a Poly reward configuration. Boxplots show the mean reward distribution among surviving arms at each phase, for four SHPRR variants with $\alpha \in \lbrace \bar { 0 . 0 } , 0 . 2 , 0 . 8 , 1 . 0 \rbrace$ . Significance brackets indicate ttests paired across seeds against the $\alpha = 0 . 0$ reference (SH-RR); a star (⋆) marks the highest mean at each phase. Phase 9 is the final recommended arm.

Figure 4 shows that this is indeed the case: at every displayed phase, increasing α significantly raises the mean reward of surviving arms relative to the $\alpha = 0 . 0$ reference. The efect is most pronounced in early phases, indicating that rationing helps the algorithm discard low-reward arms earlier in the run, but the efect is maintained throughout. By the end of the run (phase 9), the gap narrows somewhat as all variants converge toward high-reward arms, though $\alpha = 0 . 8$ still attains the highest mean reward among the variants. These results reveal the mechanism behind SHPRR’s empirical gains: rationing actively steers early elimination toward high-reward arms and sustains that advantage throughout the run. Since consumption is the reward recentered against the baseline, concentrating the surviving set on high-reward arms directly lowers per-pull consumption in later phases, so the very mechanism that improves identification also relieves the budget, grounding the higher SP and lower NSR of moderate-to-high α in a concrete, measurable efect.

## 7 Conclusion

We introduced BAI under a realized regret budget, where experimentation cost is measured as regret relative to a trusted baseline, yielding signed, reward-coupled, replenishable consumption that lies outside existing formulations. We proved that the per-pull price of information declines across the phases of SH, and exploited this property to design SHPRR, which front-loads the budget via a tunable parameter α. It is the only method robust across all synthetic and real-world environments, with $\alpha = 0 . 5$ a strong default.

## References

Audibert, J.-Y.; and Bubeck, S. 2010. Best arm identification in multi-armed bandits. In COLT-23th Conference on learning theory-2010, 13–p.

Barrier, A.; Garivier, A.; and Stoltz, G. 2023. On best-arm identification with a fixed budget in non-parametric multiarmed bandits. In International Conference on Algorithmic Learning Theory, 136–181. PMLR.

Bubeck, S.; Munos, R.; and Stoltz, G. 2009. Pure exploration in multi-armed bandits problems. In International conference on Algorithmic learning theory, 23–37. Springer.

Bubeck, S.; Munos, R.; and Stoltz, G. 2011. Pure exploration in finitely-armed and continuous-armed bandits. Theoretical Computer Science, 412(19): 1832–1852.

Carpentier, A.; and Locatelli, A. 2016. Tight (lower) bounds for the fixed budget best arm identification bandit problem. In Conference on Learning Theory, 590–604. PMLR.

Diemert, E.; Meynet, J.; Galland, P.; and Lefortier, D. 2017. Attribution Modeling Increases Eficiency of Bidding in Display Advertising. In Proceedings of the AdKDD and TargetAd Workshop, KDD, Halifax, NS, Canada. ACM.

Gabillon, V.; Ghavamzadeh, M.; and Lazaric, A. 2012. Best arm identification: A unified approach to fixed budget and fixed confidence. Advances in neural information processing systems, 25.

Garivier, A.; and Kaufmann, E. 2016. Optimal best arm identification with fixed confidence. In Conference on Learning Theory, 998–1027. PMLR.

Harding, M.; and Kandasamy, K. 2026. Balancing performance and costs in best arm identification. Advances in Neural Information Processing Systems, 38: 122335–122363.

Hou, Y.; Tan, V. Y.; and Zhong, Z. 2022. Almost optimal variance-constrained best arm identification. IEEE Transactions on Information Theory, 69(4): 2603–2634.

Jain, L.; Jamieson, K.; Mankof, R.; Nowak, R.; and Sievert, S. 2020. The New Yorker Cartoon Caption Contest Dataset. https://nextml.github.io/caption-contest-data/.

Jamieson, K.; Malloy, M.; Nowak, R.; and Bubeck, S. 2014. lil’ucb: An optimal exploration algorithm for multi-armed bandits. In Conference on Learning Theory, 423–439. PMLR.

Jourdan, M.; Degenne, R.; Baudry, D.; de Heide, R.; and Kaufmann, E. 2022. Top Two Algorithms Revisited. In Advances in Neural Information Processing Systems, volume 35, 26791–26803. Curran Associates, Inc.

Jun, K.-S.; and Nowak, R. 2016. Anytime exploration for multi-armed bandits using confidence information. In International Conference on Machine Learning, 974–982. PMLR.

Kalyanakrishnan, S.; Tewari, A.; Auer, P.; and Stone, P. 2012. PAC subset selection in stochastic multi-armed bandits. In ICML, volume 12, 655–662.

Kanarios, K.; Zhang, Q.; and Ying, L. 2024. Cost Aware Best Arm Identification. arXiv preprint arXiv:2402.16710.

Karnin, Z.; Koren, T.; and Somekh, O. 2013. Almost optimal exploration in multi-armed bandits. In International conference on machine learning, 1238–1246. PMLR.

Kaufmann, E.; Cappé, O.; and Garivier, A. 2016. On the complexity of best-arm identification in multi-armed bandit models. The Journal of Machine Learning Research, 17(1): 1–42.

Kaufmann, E.; and Kalyanakrishnan, S. 2013. Information complexity in bandit subset selection. In Conference on Learning Theory, 228–251. PMLR.

Lardy, T.; Katsimerou, C.; and Koolen, W. 2026. Constrained Best Arm Identification. Advances in Neural Information Processing Systems, 38: 100601–100627.

Li, Z.; and Cheung, W. C. 2024. Best arm identification with resource constraints. In International Conference on Artificial Intelligence and Statistics, 253–261. PMLR.

Mukherjee, A.; and Tajer, A. 2023. SPRT-based eficient best arm identification in stochastic bandits. IEEE Journal on Selected Areas in Information Theory, 4: 128–143.

Poiani, R.; Metelli, A. M.; and Restelli, M. 2022. Multi-Fidelity Best-Arm Identification. In Advances in Neural Information Processing Systems, volume 35, 17857–17870. Curran Associates, Inc.

Qin, Z.; Xue, W.; Zheng, L.; Gan, X.; Wu, H.; Jin, H.; and Fu, L. 2025. Cost-aware best arm identification in stochastic bandits. ACM Transactions on Intelligent Systems and Technology, 16(2): 1–28.

Russo, D. 2016. Simple bayesian algorithms for best arm identification. In Conference on learning theory, 1417–1418. PMLR.

Shang, X.; de Heide, R.; Kaufmann, E.; Ménard, P.; and Valko, M. 2020. Fixed-confidence guarantees for bayesian best-arm identification. In International Conference on Artificial Intelligence and Statistics.

Sui, Y.; Gotovos, A.; Burdick, J.; and Krause, A. 2015. Safe exploration for optimization with Gaussian processes. In International conference on machine learning, 997–1005. PMLR.

Wang, X.; Wu, Q.; Chen, W.; and Lui, J. 2023. Multi-fidelity multi-armed bandits revisited. Advances in Neural Information Processing Systems, 36: 31570–31600.

Wang, Z.; Wagenmaker, A. J.; and Jamieson, K. 2022. Best arm identification with safety constraints. In International Conference on Artificial Intelligence and Statistics, 9114– 9146. PMLR.

Yang, J.; Tan, V. Y. F.; and Jin, T. 2026. Best Arm Identification with Minimal Regret. Journal of Machine Learning Research, 27(86): 1–41.

Zhao, Y.; Stephens, C.; Szepesvari, C.; and Jun, K.-S. 2023. Revisiting simple regret: Fast rates for returning a good arm. In International Conference on Machine Learning, 42110– 42158. PMLR.

# SHPRR: An Algorithm for Pure Exploration under Constrained Regret

Supplementary Material

## 1 Sharpening the results of the paper Li and Cheung (2024)

Classical Sequential Halving bases each elimination decision on the observations collected during the current phase. In contrast, SH-RR as defined by Li and Cheung (2024) uses cumulative empirical means. We analyze the phase-wiseestimate variant of SH-RR, obtained all empirical means with empirical means computed from the current-phase history only. This modification leaves the resource-rationing mechanism unchanged and permits the standard expected-count and Markov argument used in the analysis of Sequential Halving. This allows us to sharpen the dependency in the number of arms in the prefactor of each bounds, specifically removing an extraneous factor K. Although this version of the algorithm does not necessarily work better in practice, it eases the comparison of the complexity of the problem with other variants seen in the literature.

In the stochastic-consumption setting, we complement this argument with an aggregate concentration inequality for the total resource consumption in a phase, which avoids the additional armwise union bound in the original proof.

We state the resulting guarantees in the general BAIwRC model with L resource constraints, for completeness.

Setting: There is L ressources, each with a budget $C _ { \ell } . \mathbf { A }$ pull of arm i generates a consumption $D _ { \ell , i }$ of resource $\ell .$ In the deterministic case, $D _ { \ell , i } = d _ { \ell , i }$ deterministically. In the stochastic case, $D _ { \ell , i } \in [ 0 ; 1 ]$ a.s, while E $\mathfrak { L } [ D _ { \ell , i } ] = d _ { \ell , i }$ The rest is the same as in the main text. Note that the original paper assumes that the arm rewards are 1-subgaussian, while we assume that the rewards lie in [0, 1]. The stronger assumption is necessary when one of the resources can be the regret. To avoid confusion, we keep it, though the same bound up to a modified concentration constant could be obtained with 1-subgaussian rewards if none of the ressources are the regret.

Notations: Throughout the Appendix, we denote $t _ { q }$ the (possibly random) number of pulls per arm in phase $q .$ We define $\hat { \mu } _ { i } ^ { q }$ as the empirical mean for arm i at the end of phase q.

## 1.1 The deterministic consumption bound

Define denote $d _ { \ell , i }$ the permutation of the indices so that $d _ { \ell , ( 1 ) } \geq , \ldots , \geq d _ { \ell , ( K ) }$

$$
H _ { 2 , \ell } : = \operatorname* { m a x } _ { k \in [ K ] } \frac { \sum _ { j = 1 } ^ { k } d _ { \ell , ( j ) } } { \Delta _ { k } ^ { 2 } } ,
$$

and

$$
\gamma ( Q ) = \operatorname* { m i n } _ { \ell \in [ L ] } { \frac { C _ { \ell } } { H _ { 2 , \ell } } } .
$$

We prove the following theorem, which sharpens the pre-factors’ dependency in the number of arms in the bound in Li and Cheung (2024) from $K \log _ { 2 } ( K )$ to $\log _ { 2 } ( K )$

Theorem 1.1. Under the deterministic consumption setting,

$$
\mathbb { P } \left( \phi \neq 1 \right) \leq 3 \log ( K ) \exp \left( - \frac { \gamma ( Q ) } { 8 \log K } \right)
$$

Proof. Note that in the deterministic consumption case, conditioned on $\mathcal { F } _ { q } , t _ { q }$ is not a random quantity. Therefore, by applying Hoefding’s inequality, we obtain the following lemma.

Lemma 1.1. Assume that the best arm was not eliminated prior to round q. Then for any arm $i \in S ^ { ( q ) }$

$$
\operatorname* { P r } \left[ \hat { \mu } _ { 1 } ^ { q } < \hat { \mu } _ { i } ^ { q } | \mathcal { F } _ { q } \right] \leq \exp \left( - \frac { 1 } { 2 } t _ { q } \Delta _ { i } ^ { 2 } \right) .
$$

We next bound the probability that the algorithm makes a mistake and excludes the best arm on phase $q .$

Lemma 1.2. The probability that the best arm is eliminated on phase q is at most

$$
3 \exp \left( - \frac { 1 } { 8 \log _ { 2 } K } \cdot \gamma ( Q ) \right) .
$$

Proof. Suppose first that $| S ^ { ( q ) } | \ge 4$ . We treat $\vert S ^ { ( q ) } \vert = 2$ separately. Define $S ^ { \prime ( q ) }$ as the set of arms in $S ^ { ( q ) }$ , excluding the ${ \frac { \mathrm { i } } { 4 } } \ \left| S ^ { ( q ) } \right| = K / 2 ^ { q + 2 }$ arms with the largest mean. Denote:

$$
i _ { q } = \frac { 1 } { 4 } \left| S ^ { ( q ) } \right| + 1 .
$$

If the best arm is eliminated in phase $q ,$ it must be the case that at least half the arms of $S ^ { ( q ) } ( \mathrm { i . e . , } \frac { 1 } { 2 } | S ^ { ( q ) } | = n / 2 ^ { q + 1 }$ arms) have their empirical average larger than its empirical average. In particular, the empirical means of at least ${ \textstyle \frac { 1 } { 3 } } \left| S ^ { \prime ( q ) } \right| = n / 2 ^ { q + 2 }$ of the arms in $S ^ { \prime ( q ) }$ must be larger than that of the best arm at the end of phase $q .$ Letting $N _ { q }$ denote the number of arms in $S ^ { \prime ( q ) }$ whose empirical average is larger than that of the optimal arm, we have by Lemma 1.1:

$$
\begin{array} { l } { { \displaystyle { \bf E } \left[ N _ { q } \right] = \sum _ { i \in S ^ { \prime ( q ) } } \mathrm { P r } \left[ \hat { \mu } _ { 1 } ^ { q } < \hat { \mu } _ { i } ^ { q } \right] \leq \sum _ { i \in S ^ { \prime ( q ) } } \exp \left( - \frac { 1 } { 2 } t _ { q } \Delta _ { i } ^ { 2 } \right) } } \\ { { \displaystyle \quad \leq \left| S ^ { \prime ( q ) } \right| \exp \left( - \frac { 1 } { 2 } t _ { q } \Delta _ { i _ { q } } ^ { 2 } \right) } } \end{array}
$$

We now lower bound $t _ { q } .$ . By design of the algorithm:

$$
\begin{array} { r l } & { t _ { q } \geq \underset { \ell } { \mathrm { m i n } } \frac { C _ { \ell } } { \log _ { 2 } ( K ) } \frac { 1 } { \sum _ { j \in S ^ { ( q ) } } d _ { \ell , j } } } \\ & { \quad \geq \underset { \ell } { \mathrm { m i n } } \frac { C _ { \ell } } { \log _ { 2 } ( K ) } \frac { 1 } { \sum _ { j = 1 } ^ { | S ^ { ( q ) } | } d _ { \ell , ( j ) } } } \\ & { \quad \geq \underset { \ell } { \mathrm { m i n } } \frac { C _ { \ell } } { \log _ { 2 } ( K ) } \frac { 1 } { 4 \sum _ { j = 1 } ^ { i _ { q } - 1 } d _ { \ell , ( j ) } } . } \end{array}
$$

Plugging into the previous equation we obtain:

$$
\begin{array} { r l } & { \mathbf { E } \left[ N _ { q } \right] \le | S ^ { \prime ( q ) } | \exp \left( - \frac { 1 } { 8 \log _ { 2 } ( K ) } \underset { \ell } { \operatorname* { m i n } } \frac { C _ { \ell } } { \frac { 1 } { \Delta _ { i _ { q } } ^ { 2 } } \sum _ { j = 1 } ^ { i _ { q } - 1 } d _ { \ell , ( j ) } } \right) } \\ & { \qquad \le | S ^ { \prime ( q ) } | \exp \left( - \frac { 1 } { 8 \log _ { 2 } ( K ) } \underset { \ell } { \operatorname* { m i n } } \frac { C _ { \ell } } { H _ { 2 , \ell } } \right) } \\ & { \qquad \le | S ^ { \prime ( q ) } | \exp \left( - \frac { 1 } { 8 \log _ { 2 } ( K ) } \gamma ( Q ) \right) , } \end{array}
$$

where the second inequality holds by definition of $H _ { 2 , \ell }$ and the third by definition of $\gamma ( Q )$

We now apply Markov’s inequality to obtain

$$
\begin{array} { l } { { \displaystyle \operatorname* { P r } \left[ N _ { q } > \frac { 1 } { 3 } \left| S ^ { \prime ( q ) } \right| \right] \le 3 { \bf E } \left[ N _ { q } \right] \big / \left| S ^ { \prime ( q ) } \right| } } \\ { { \displaystyle \phantom { \sum } \le 3 \exp \left( - \frac { 1 } { 8 \log _ { 2 } K } \cdot \gamma ( Q ) \right) } } \end{array}
$$

and the lemma is established for all q such that $n _ { q } \geq 4$ . It remains only the case $n _ { q } = 2$ . In that case, provided arm $1 \in S ^ { ( q ) }$ , with i the other arm in $S ^ { ( q ) }$

$$
\mathbb { P } \left( \phi \neq 1 | \mathcal { F } _ { q } \right) = \operatorname* { P r } \left[ \hat { \mu } _ { 1 } ^ { q } < \hat { \mu } _ { i } ^ { q } | \mathcal { F } _ { q } \right] \leq \exp \left( - \frac { 1 } { 2 } t _ { q } \Delta _ { i } ^ { 2 } \right) \leq \exp \left( - \frac { 1 } { 2 } t _ { q } \Delta _ { 2 } ^ { 2 } \right) .
$$

Also:

$$
t _ { q } \geq \operatorname* { m i n } _ { \ell } \frac { C _ { \ell } } { \log _ { 2 } ( K ) } \frac { 1 } { \sum _ { j = 1 } ^ { 2 } d _ { \ell , ( j ) } } .
$$

Hence:

$$
\mathbb { P } \left( \phi \neq 1 | \mathcal { F } _ { q } \right) \leq \exp \left( - \frac { 1 } { 2 \log _ { 2 } ( K ) } \operatorname* { m i n } _ { \ell } \frac { C _ { \ell } } { \frac { 1 } { \Delta _ { 2 } ^ { 2 } } \sum _ { j = 1 } ^ { 2 } d _ { \ell , ( j ) } } \right) \leq 3 \exp \left( - \frac { 1 } { 8 \log _ { 2 } ( K ) } \gamma ( Q ) \right) .
$$

The proof of Theorem 1.1 is now immediate.

Proof. Clearly, the algorithm does not exceed the budget of any resource. Also, if the best arm survives the execution, then the algorithm succeeds as all other arms must be eliminated after $\log _ { 2 }$ n phases. Finally, by Lemma 1.2 and a union bound, the best arm is eliminated in one of the $\log _ { 2 } K$ phases of the algorithm with probability at most 3 log<sub>2</sub> $K \cdot \exp \left( - { \frac { \gamma ( Q ) } { 8 \log _ { 2 } K } } \right)$ , which gives the theorem. □

## 1.2 Stochastic consumption setting

Theorem 1.2. Consider a BAIwRC instance $Q$ in the stochastic consumption setting. The SH-RR algorithm has BAI failure probability $\Pr ( \phi \neq 1 )$ at most

$$
\left( L + 3 \right) \left( \log _ { 2 } K \right) \exp \left( - \frac { 1 } { 8 \left\lceil \log _ { 2 } K \right\rceil } \cdot \gamma ^ { s t o } \left( Q \right) \right) ,
$$

where $\begin{array} { r } { \gamma ^ { s t o } \left( Q \right) = \operatorname* { m i n } _ { \ell \in \left[ L \right] } \left\{ C _ { \ell } / H _ { 2 , \ell } ^ { s t o } \left( Q \right) \right\} } \end{array}$ and

$$
H _ { 2 , \ell } ^ { \mathrm { s t o } } ( Q ) = \operatorname* { m a x } _ { k \in \{ 2 , \ldots , K \} } \left\{ \frac { \sum _ { j = 1 } ^ { k } f \left( d _ { \ell , ( j ) } \right) } { \Delta _ { k } ^ { 2 } } \right\} .
$$

Proof. For the stochastic setting, the main dificulty is that the number of pulls is stochastic. The proof then needs to lower bound the total number of pulls and also take into account that the number of samples in the empirical mean are history dependent.

We have two helper lemmas. The first one will be helpful to lower bound the number of pulls in each phase. It is a modified version of Lemma 7 in Li and Cheung (2024). The lemma in the original paper helps control the consumption per arm, this modified version controls the aggregate consumption. Again, the goal is to avoid resorting to a union bound.

Lemma 1.3. Consider a family $( X _ { i , s } ) _ { i \in [ m ] , s \in [ N ] }$ of mutually independent rewards in $( 0 , 1 ]$ such that for each $i ,$ $\mathbb { E } [ X _ { i , s } ] = d _ { i }$ . Then

$$
\mathbb { P } \left( \sum _ { s = 1 } ^ { N } \sum _ { i = 1 } ^ { m } X _ { i , s } \ge N \sum _ { i = 1 } ^ { m } f ( d _ { i } ) \right) \le e ^ { - N / 2 } .
$$

The function $f$ is chosen so that:

Claim 1.1. For any $d \in ( 0 ; e ^ { - 2 } ] , \operatorname { k l } ( f ( d ) , d ) \geq { \frac { 1 } { 2 } } ,$ , where kl denotes the Bernoulli relative entropy.

This claim, which will be useful for the proof of the lemma, is proved in the paper Li and Cheung (2024), and we omit it here.

We can now show the number of samples in each phase is lower bounded. For each stage $q ,$ define

$$
\underline { { \tau _ { q } } } = \operatorname* { m i n } _ { \ell } \frac { C _ { \ell } } { \log _ { 2 } ( K ) \sum _ { j = 1 } ^ { | S ^ { ( q ) } | } f \left( d _ { \ell , ( j ) } \right) } .
$$

Lemma 1.4. For every phase q, conditionally on the history $\mathcal { F } _ { q }$ at the start of the phase,

$$
\mathbb { P } \left( t _ { q } < \underline { { \tau } } _ { q } \mid \mathcal { F } _ { q } \right) \le L \exp \left( - \frac { \underline { { \tau } } _ { q } } { 2 } \right) .
$$

The second ingredient to the proof is a version of Lemma 1.1 that holds when the number of samples is adaptive.

Lemma 1.5. Let $1 \leq \tau \leq T ,$ , and let $( X _ { i } ) _ { i = 1 } ^ { \infty }$ and $( Y _ { i } ) _ { i = 1 } ^ { \infty }$ be two mutually independent sequences ofindependent and identically distributed random variables taking values in [0, 1]. Suppose that

$$
\Delta = \mathbb { E } [ X _ { 1 } - Y _ { 1 } ] > 0 .
$$

Then

$$
\mathbb { P } \left( \exists t \in \{ \tau , \dots , T \} : \overline { { X } } _ { t } \leq \overline { { Y } } _ { t } \right) \leq \exp \left( - \tau \Delta ^ { 2 } \right) .
$$

With these two lemmas, we are ready to prove the main theorem.

Let $\mathcal { F } _ { q }$ denote the history at the beginning of phase q, and denote on $\mathcal { E } _ { q }$ the event $\{ 1 \in S ^ { ( q ) } \}$ that the best arm has survived so far.

First suppose that $n _ { q } \geq 4 .$ . Put

$$
i _ { q } : = \frac { n _ { q } } { 4 } + 1 ,
$$

and define $S ^ { \prime ( q ) }$ as the set of arms in $S ^ { ( q ) }$ , excluding the $\textstyle { \frac { 1 } { 4 } } n _ { q } \ = \ K / 2 ^ { q + 2 }$ arms with the largest mean. Every arm $i \in S ^ { \prime ( q ) }$ has global reward rank at least $i _ { q } ,$ , and hence

$$
\Delta _ { i } \geq \Delta _ { i _ { q } } .
$$

Let

$$
N _ { q } : = \sum _ { i \in S ^ { \prime } ^ { ( q ) } } { \bf 1 } \{ \widehat { \mu } _ { i } ^ { q } \geq \widehat { \mu } _ { 1 } ^ { q } \} .
$$

If the best arm is eliminated in phase q, it must be the case that at least half the arms of $S ^ { ( q ) }$ have their empirical average larger than its empirical average. In particular, this implies

$$
N _ { q } \geq \frac { n _ { q } } { 4 } = \frac { | S ^ { \prime ( q ) } | } { 3 } .
$$

Therefore,

$$
\mathbb { P } \left( \overline { { \mathscr { E } } } _ { q + 1 } \left| \mathcal { F } _ { q } , \mathscr { E } _ { q } \right. \right) \leq \mathbb { P } ( t _ { q } < \underline { { \tau _ { q } } } \mid \mathscr { F } _ { q } , \mathscr { E } _ { q } ) + \mathbb { P } \left( N _ { q } \geq \frac { | S ^ { \prime ( q ) } | } { 3 } , \ t _ { q } \geq \underline { { \tau _ { q } } } \middle | \mathscr { F } _ { q } , \mathscr { E } _ { q } \right)
$$

By Lemma 1.4,

$$
\mathbb { P } ( t _ { q } < \underline { { \tau _ { q } } } \mid \mathcal { F } _ { q } , \mathcal { E } _ { q } ) \le L \exp \left( - \frac { \tau _ { q } } { 2 } \right) .
$$

By Markov’s inequality:

$$
\mathbb { P } \left( N _ { q } \geq \frac { | S ^ { \prime ( q ) } | } { 3 } , t _ { q } \geq \underline { { \tau _ { q } } } \middle | \mathcal { F } _ { q } , \mathcal { E } _ { q } \right) \leq \frac { 3 } { | S ^ { \prime ( q ) } | } \mathbb { E } \left[ N _ { q } \mathbf { 1 } \{ t _ { q } \geq \underline { { \tau _ { q } } } \} \middle | \mathcal { F } _ { q } , \mathcal { E } _ { q } \right] .
$$

For every $i \in S ^ { \prime ( q ) }$ , the event

$$
\left\{ t _ { q } \geq \underline { { \tau _ { q } } } , \widehat { \mu } _ { i } ^ { q } \geq \widehat { \mu } _ { 1 } ^ { q } \right\}
$$

is contained in the event that the empirical mean of arm i exceeds that of arm 1 at some common sample size $t _ { q } \geq \underline { { \tau _ { q } } }$ Hence Lemma 1.5 gives

$$
\begin{array} { r } { \mathbb { P } \Big ( t _ { q } \geq \underline { { \tau _ { q } } } , \widehat { \mu } _ { i } ^ { q } \geq \widehat { \mu } _ { 1 } ^ { q } \mid \mathcal { F } _ { q } , \mathcal { E } _ { q } \Big ) \leq \exp \Big ( { - \underline { { \tau _ { q } } } \Delta _ { i } ^ { 2 } } \Big ) . } \end{array}
$$

It follows that

$$
\begin{array} { r } { \mathbb { E } \left[ N _ { q } { \bf 1 } \{ t _ { q } \geq \underline { { \tau _ { q } } } \} \Big | \mathcal { F } _ { q } , \mathcal { E } _ { q } \right] \leq | S ^ { \prime ( q ) } | \exp \Big ( { - \underline { { \tau _ { q } } } \Delta _ { i _ { q } } ^ { 2 } } \Big ) , } \end{array}
$$

and therefore

$$
\mathbb { P } \left( \overline { { \mathscr { E } } } _ { q + 1 } \big | \mathcal { F } _ { q } , \mathscr { E } _ { q } \right) \leq L \exp \left( - \frac { \tau _ { q } } { 2 } \right) + 3 \exp \left( - \underline { { \tau _ { q } } } \Delta _ { i _ { q } } ^ { 2 } \right) .\tag{1}
$$

It remains to lower-bound the two exponents. Since f is increasing, $f ( d _ { \ell , ( 1 ) } ) , \ldots , f ( d _ { \ell , ( K ) } )$ is nonincreasing. Thus

$$
\sum _ { i = 1 } ^ { | S ^ { ( q ) } | } f ( d _ { \ell , ( i ) } ) \leq 4 \sum _ { i = 1 } ^ { i _ { q } } f ( d _ { \ell , ( i ) } ) .
$$

Consequently,

$$
\begin{array} { r l } & { \underline { { \tau _ { q } } } \Delta _ { i _ { q } } ^ { 2 } = \displaystyle \operatorname* { m i n } _ { \ell \in [ L ] } \frac { C _ { \ell } \Delta _ { i _ { q } } ^ { 2 } } { \log _ { 2 } ( K ) \sum _ { i = 1 } ^ { | S ( q ) | } f \big ( d _ { \ell , ( i ) } \big ) } } \\ & { \qquad \geq \frac { 1 } { 4 \log _ { 2 } ( K ) } \displaystyle \operatorname* { m i n } _ { \ell \in [ L ] } \frac { C _ { \ell } } { \frac { 1 } { \Delta _ { i _ { q } } ^ { 2 } } \sum _ { i = 1 } ^ { i _ { q } } f \big ( d _ { \ell , ( i ) } \big ) } } \\ & { \qquad \geq \frac { \gamma ^ { \mathrm { s t o } } ( Q ) } { 4 \log _ { 2 } ( K ) } . } \end{array}
$$

Because the rewards lie in [0, 1], $\Delta _ { i _ { q } } \leq 1$ , so also

$$
\underline { { \tau } } _ { q } \geq \underline { { \tau } } _ { q } \Delta _ { i _ { q } } ^ { 2 } \geq \frac { \gamma ^ { \mathrm { s t o } } ( Q ) } { 4 \log _ { 2 } ( K ) } .
$$

Substituting these inequalities into (5) yields

$$
\begin{array} { r l } & { \mathbb { P } \left( \overline { { \mathcal { E } } } _ { q + 1 } \middle \vert \mathcal { F } _ { r } \right) \leq L \exp \left( - \frac { \gamma ^ { \mathrm { s t o } } ( Q ) } { 8 \log _ { 2 } ( K ) } \right) + 3 \exp \left( - \frac { \gamma ^ { \mathrm { s t o } } ( Q ) } { 4 \log _ { 2 } ( K ) } \right) } \\ & { \qquad \leq \left( L + 3 \right) \exp \left( - \frac { \gamma ^ { \mathrm { s t o } } ( Q ) } { 8 \log _ { 2 } ( K ) } \right) . } \end{array}
$$

When $n _ { q } = 2$ , there is only one competing arm. Apply the adaptive Hoefding lemma directly, which also yields:

$$
\mathbb { P } \left( \mathrm { a r m ~ 1 ~ i s ~ e l i m i n a t e d ~ i n ~ p h a s e ~ l o g } _ { 2 } ( K ) \mid \mathcal { F } _ { r } \right) \leq \left( L + 3 \right) \exp \left( - \frac { \gamma ^ { \mathrm { s t o } } ( Q ) } { 8 \log _ { 2 } ( K ) } \right) .
$$

Summing over all phases yields the final result.

ProofofLemma 1.4. We have:

$$
\begin{array} { r l } & { \mathbb { P } \left( t _ { q } \le \underline { { \tau _ { q } } } | \mathcal { F } _ { q } \right) \le \mathbb { P } \left( \exists \ell \mathrm { ~ s . t . ~ t h e ~ t y p e ~ } \ell \mathrm { ~ r e s s o u r c e ~ i s ~ e x h a u s t e d ~ a f t e r ~ } \underline { { \tau _ { t } } } \mathrm { ~ s a m p l e s } | \mathcal { F } _ { q } \right) } \\ & { \qquad \le L \mathbb { P } \left( \mathrm { t y p e ~ } \ell \mathrm { ~ r e s s o u r c e ~ i s ~ e x h a u s t e d ~ a f t e r ~ } \underline { { \tau _ { t } } } \mathrm { ~ s a m p l e s } | \mathcal { F } _ { q } \right) . } \end{array}
$$

Denote $d _ { \ell } ^ { i , t }$ the ressource consumption of ressource type ℓ by arm i on iteration t. We have:

$$
\begin{array} { r l } & { \mathbb { P } \left( \operatorname { t y p e } \ell \mathrm { ~ e x h a u s t e d ~ a t } \underline { { \tau _ { q } } } \operatorname { s a m p l e s } | \mathcal { F } _ { q } \right) = \mathbb { P } \left( \displaystyle \sum _ { t = \tau _ { t - 1 } } ^ { \tau _ { q } + \tau _ { q - 1 } } \displaystyle \sum _ { i \in S ^ { ( a ) } } d _ { \ell } ^ { i , t } \geq \frac { C _ { \ell } } { \log _ { 2 } ( K ) } | \mathcal { F } _ { q } \right) } \\ & { \qquad = \mathbb { P } \left( \displaystyle \sum _ { t = \tau _ { q - 1 } } ^ { \tau _ { q } + \tau _ { q - 1 } } \displaystyle \sum _ { i \in S ^ { ( q ) } } d _ { \ell } ^ { i , t } \geq \displaystyle \sum _ { \tau = 1 } ^ { \frac { \tau _ { q } } { 2 } } \frac { C _ { \ell } } { \log _ { 2 } ( K ) \operatorname* { m i n } _ { \ell } \frac { C _ { \ell } } { \log _ { 2 } ( K ) \sum _ { j = 1 } ^ { | S ^ { ( q ) } | f } \left( d _ { \ell , ( j ) } \right) } } | \mathcal { F } _ { q } \right) } \\ & { \qquad \leq \mathbb { P } \left( \displaystyle \sum _ { t = \tau _ { r - 1 } } ^ { \tau _ { q } + \tau _ { q - 1 } } \displaystyle \sum _ { i \in S ^ { ( q ) } } d _ { \ell } ^ { i , t } \geq \frac { | S ^ { ( q ) } | } { \underline { { \tau _ { q } } } } \displaystyle \sum _ { j = 1 } ^ { | S ^ { ( q ) } | } f \left( d _ { \ell , ( j ) } \right) | \mathcal { F } _ { q } \right) } \end{array}\tag{2}
$$

By Lemma 1.3,

$$
\begin{array} { r } { \mathbb { P } \Big ( \mathrm { \ t y p e \ \ell \ e x h a u s t e d { \ a t } } \underline { { \tau _ { q } } } \mathrm { \ s a m p l e s } | \mathcal { F } _ { q } \Big ) \leq \exp \Big ( { - \underline { { \tau _ { q } } } / 2 } \Big ) . } \end{array}
$$

A union bound over the L possible resources gives the lemma.

Proof of Lemma 1.5. By Hoefding’s lemma, for every $\lambda \in \mathbb { R }$

$$
\mathbb { E } \left[ \exp \left( \lambda ( Y _ { i } - \mathbb { E } [ Y _ { i } ] ) \right) \right] \leq \exp \left( \frac { \lambda ^ { 2 } } { 8 } \right)
$$

and

$$
\mathbb { E } \left[ \exp \left( - \lambda ( X _ { i } - \mathbb { E } [ X _ { i } ] ) \right) \right] \leq \exp \left( \frac { \lambda ^ { 2 } } { 8 } \right) .
$$

Since the two sequences are mutually independent, it follows that

$$
\begin{array} { r l } & { \mathbb { E } \left[ \exp \left( \lambda \big ( ( Y _ { i } - \mathbb { E } [ Y _ { i } ] ) - ( X _ { i } - \mathbb { E } [ X _ { i } ] ) \big ) \right) \right] } \\ & { \quad \quad = \mathbb { E } \left[ \exp \left( \lambda \big ( Y _ { i } - \mathbb { E } [ Y _ { i } ] \big ) \right) \right] \mathbb { E } \left[ \exp \left( - \lambda \big ( X _ { i } - \mathbb { E } [ X _ { i } ] \big ) \right) \right] \leq \exp \left( \displaystyle \frac { \lambda ^ { 2 } } { 4 } \right) . } \end{array}
$$

Therefore, with respect to the natural filtration,

$$
M _ { t } = \exp \left( \lambda \sum _ { i = 1 } ^ { t } \bigl ( ( Y _ { i } - \mathbb { E } [ Y _ { i } ] ) - ( X _ { i } - \mathbb { E } [ X _ { i } ] ) \bigr ) - \frac { t \lambda ^ { 2 } } { 4 } \right) , \qquad M _ { 0 } = 1 ,
$$

is a nonnegative supermartingale.

If $\overline { { X } } _ { t } \leq \overline { { Y } } _ { t }$ , then

$$
\begin{array} { r l } {  { \sum _ { i = 1 } ^ { t } \bigl ( ( Y _ { i } - \mathbb { E } [ Y _ { i } ] ) - ( X _ { i } - \mathbb { E } [ X _ { i } ] ) \bigr ) = \sum _ { i = 1 } ^ { t } ( Y _ { i } - X _ { i } ) + t \bigl ( \mathbb { E } [ X _ { 1 } ] - \mathbb { E } [ Y _ { 1 } ] \bigr ) } } \\ & { \geq t \Delta . } \end{array}
$$

Choosing $\lambda = 2 \Delta$ , we obtain, on this event,

$$
M _ { t } \ge \exp \left( 2 \Delta \cdot t \Delta - \frac { t ( 2 \Delta ) ^ { 2 } } { 4 } \right) = \exp \left( t \Delta ^ { 2 } \right) .
$$

Hence, for every $t \geq \tau$

$$
\left\{ \overline { { X } } _ { t } \leq \overline { { Y } } _ { t } \right\} \subseteq \left\{ M _ { t } \geq \exp \left( \tau \Delta ^ { 2 } \right) \right\} .
$$

Consequently,

$$
\begin{array} { r l } & { \mathbb { P } \left( \exists t \in \{ \tau , \dots , T \} : \overline { { X } } _ { t } \leq \overline { { Y } } _ { t } \right) \leq \mathbb { P } \left( \underset { 0 \leq t \leq T } { \operatorname* { m a x } } M _ { t } \geq \exp \left( \tau \Delta ^ { 2 } \right) \right) } \\ & { \qquad \leq \exp \left( - \tau \Delta ^ { 2 } \right) , } \end{array}
$$

where the last inequality follows from Ville’s inequality.

Proof of Lemma 1.3. The case $\textstyle \sum _ { i = 1 } ^ { m } f ( d _ { i } ) ~ > ~ m$ is trivial as the values lie in [0; 1]. Let us consider the case $\textstyle \sum _ { i = 1 } ^ { m } { \dot { f } } ( d _ { i } ) \leq m$ By Chernof’s inequality,

$$
\mathbb { P } \left( \sum _ { s = 1 } ^ { N } \sum _ { i = 1 } ^ { m } X _ { i , s } \geq N \sum _ { i = 1 } ^ { m } f ( d _ { i } ) \right) \leq \operatorname* { i n f } _ { \lambda \geq 0 } \mathbb { E } \left[ \exp \left( \lambda \sum _ { s = 1 } ^ { N } \sum _ { i = 1 } ^ { m } X _ { i , s } \right) \right] \exp \left( - \lambda N \sum _ { i = 1 } ^ { m } f ( d _ { i } ) \right)\tag{3}
$$

For each $X _ { i , s } ,$ by convexity,

$$
\operatorname { \mathbb { E } } \left[ \exp \left( \lambda X _ { i , s } \right) \right] \leq 1 - d _ { i } + d _ { i } e ^ { \lambda } = \exp \left( \log \left( 1 - d _ { i } + d _ { i } e ^ { \lambda } \right) \right) .
$$

Combining this with the independence of the $X _ { i , s }$ yields:

$$
E [ \exp ( \lambda \sum _ { s = 1 } ^ { N } \sum _ { i = 1 } ^ { m } X _ { i , s } ) ] \leq \exp ( N \sum _ { i = 1 } ^ { m } \log ( 1 - d _ { i } + d _ { i } e ^ { \lambda } ) ) ) .
$$

Injecting in Equation 4:

$$
\mathbb { P } \left( \sum _ { s = 1 } ^ { N } \sum _ { i = 1 } ^ { m } X _ { i , s } \ge N \sum _ { i = 1 } ^ { m } f ( d _ { i } ) \right) \le \exp \left( - N \operatorname* { s u p } _ { \lambda \ge 0 } \sum _ { i = 1 } ^ { m } \underbrace { \lambda f ( d _ { i } ) - \log \bigl ( 1 - d _ { i } + d _ { i } e ^ { \lambda } \bigr ) } _ { g _ { i } ( \lambda ) } \right)\tag{4}
$$

It remains to show $\textstyle \operatorname* { s u p } _ { \lambda \geq 0 } \sum _ { i = 1 } ^ { m } g _ { i } ( \lambda ) \geq 1 / 2$

Consider first an index i such that $f ( d _ { i } ) < 1$ . Since $f ( d _ { i } ) > d _ { i }$ , the function $g _ { i }$ is maximized at

$$
\lambda _ { i } ^ { \star } : = \log \left( \frac { f ( d _ { i } ) ( 1 - d _ { i } ) } { d _ { i } ( 1 - f ( d _ { i } ) ) } \right) > 0 ,
$$

and the value of the maximum is

$$
g _ { i } ( \lambda _ { i } ^ { \star } ) = \mathrm { k l } ( f ( d _ { i } ) , d _ { i } ) ,
$$

where kl denotes the Bernoulli relative entropy. Moreover, $g _ { i }$ is nondecreasing on $[ 0 , \lambda _ { i } ^ { \star } ]$

Suppose that there is at least one index such that $f ( d _ { i } ) < 1$ , and choose

$$
j \in \operatorname * { a r g m i n } _ { \{ i : f ( d _ { i } ) < 1 \} } \lambda _ { i } ^ { \star } .
$$

For every i such that $f ( d _ { i } ) < 1$ , we have ${ \lambda } _ { j } ^ { \star } \le { \lambda } _ { i } ^ { \star }$ , and therefore

$$
g _ { i } ( \lambda _ { j } ^ { \star } ) \geq g _ { i } ( 0 ) = 0 .
$$

For every i such that $f ( d _ { i } ) \geq 1$ , we have

$$
g _ { i } ^ { \prime } ( \lambda ) = f ( d _ { i } ) - \frac { d _ { i } e ^ { \lambda } } { 1 - d _ { i } + d _ { i } e ^ { \lambda } } \geq 0 ,
$$

and hence again

$$
g _ { i } ( \lambda _ { j } ^ { \star } ) \geq g _ { i } ( 0 ) = 0 .
$$

It follows that

$$
\operatorname* { s u p } _ { \lambda \geq 0 } \sum _ { i = 1 } ^ { m } g _ { i } ( \lambda ) \geq \sum _ { i = 1 } ^ { m } g _ { i } ( \lambda _ { j } ^ { \star } ) \geq g _ { j } ( \lambda _ { j } ^ { \star } ) = \mathrm { k l } ( f ( d _ { j } ) , d _ { j } ) .
$$

Since $f ( d _ { j } ) < 1$ , necessarily $d _ { j } < e ^ { - 2 }$ . As shown in the claim,

$$
\begin{array} { r } { \mathrm { k l } ( f ( d _ { j } ) , d _ { j } ) \geq 1 / 2 . } \end{array}
$$

Consequently,

$$
\operatorname* { s u p } _ { \lambda \ge 0 } \sum _ { i = 1 } ^ { m } g _ { i } ( \lambda ) \ge 1 / 2 .
$$

It remains to consider the case in which $f ( d _ { i } ) \geq 1$ for every i. Since $\textstyle \sum _ { i = 1 } ^ { m } f ( d _ { i } ) \leq m$ , this forces $f ( d _ { i } ) = 1$ for every i, and hence $d _ { i } = e ^ { - 2 }$ for every i. In this case,

$$
\begin{array} { l } { { \displaystyle \operatorname* { s u p } _ { \lambda \geq 0 } \sum _ { i = 1 } ^ { m } g _ { i } ( \lambda ) \geq \operatorname* { l i m } _ { \lambda \to \infty } \sum _ { i = 1 } ^ { m } \left[ \lambda - \log \left( 1 - d _ { i } + d _ { i } e ^ { \lambda } \right) \right] } } \\ { { \displaystyle \quad = \sum _ { i = 1 } ^ { m } \log \frac { 1 } { d _ { i } } = 2 m \geq 1 . } } \end{array}
$$

Thus, in all cases,

$$
\operatorname* { s u p } _ { \lambda \ge 0 } \sum _ { i = 1 } ^ { m } g _ { i } ( \lambda ) \ge 1 / 2 .
$$

Substituting this into the Chernof bound gives

$$
\mathbb { P } \left( \sum _ { s = 1 } ^ { N } \sum _ { i = 1 } ^ { m } X _ { i , s } \ge N \sum _ { i = 1 } ^ { m } f ( d _ { i } ) \right) \le e ^ { - N / 2 } .
$$

## 2 Proof of the main Theorem

Here, we derive Theorem 4.1 from the main structural proposition.

For each phase $q ,$ denote $A _ { q }$ the event that the surviving arms are collectively well-concentrated around the top of the reward distribution:

$$
\mathcal { A } _ { q } : = \left\{ \frac { 1 } { n _ { q } } \sum _ { i \in S _ { q } } \Delta _ { i } \leq 3 \Delta _ { 2 n _ { q } } . \right\}
$$

Informally, $A _ { q }$ asserts that the average suboptimality of the surviving set at phase $q$ is within a constant factor of $\Delta _ { 2 n _ { q } }$ the gap of the arm at rank $2 n _ { q }$ in the overall reward ordering.

Under $A _ { q } ,$ each round-robin cycle consumes at most $3 n _ { q } \Delta _ { 2 n _ { q } }$ expected regret. Recall the definition of $\underline { { \tau _ { q } } } ,$ which we will show to be a conservative lower bound on the number of cycles completed in phase $q \colon$

$$
\underline { { \tau _ { q } } } = \operatorname* { m i n } \left( \frac { T } { n _ { q } \log _ { 2 } K } , \ : \frac { B ^ { ( q ) } } { 6 n _ { q } \Delta _ { 2 n _ { q } } } \right) .
$$

Let us reformulate the main structural result (Proposition 4.1) with these notations.

Proposition 2.1. Iffor all $q ^ { \prime } ,$ log 4e $\begin{array} { r } { \le \frac { 1 } { 8 } \underline { { \tau } } _ { q ^ { \prime } } \left( \frac { \Delta \frac { n _ { q ^ { \prime } } } { 4 } + 1 } { \log _ { 2 } ( K ) } \right) ^ { 2 } } \end{array}$ , then

$$
\mathbb { P } \left( \overline { { \bigcap _ { q } { \cal A } _ { q } } } \right) \leq 3 \log _ { 2 } ( K ) ^ { 2 } \exp \left( - \frac { 1 } { 8 \log _ { 2 } ( K ) ^ { 2 } } \operatorname* { m i n } _ { q } \underline { { \tau } } _ { q } \Delta _ { \frac { n _ { q } } { 4 } + 1 } ^ { 2 } \right) .
$$

Note that this is the same proposition as in the main text, since $\begin{array} { r } { \Gamma _ { 2 , R } = \operatorname* { m i n } _ { q \in [ \log _ { 2 } ( K ) ] } \underline { { \tau } } _ { q } \Delta _ { \frac { n _ { q } } { \scriptscriptstyle { n } } + 1 } ^ { 2 } } \end{array}$ . The above proposition is proved in Section 2.1. The main consequence of event $A _ { q } ,$ is that, provided it holds, then the number of pulls per arm in the phase is lower bounded.

Proposition 2.2. For each phase q,

$$
\begin{array} { r } { \mathbb { P } \left( t _ { q } < \underline { { \tau _ { q } } } | \boldsymbol { \mathcal { A } } _ { q } , \mathcal { F } _ { q } \right) \le \exp \left( - 1 8 \underline { { \tau _ { q } } } \Delta _ { 2 n _ { q } } ^ { 2 } \right) } \end{array}
$$

Proposition 2.2 states that, conditional on arm quality event $A _ { q } ,$ the phase completes at least $\underline { { \tau _ { q } } }$ cycles with high probability. The only way fewer cycles are completed is if the regret budget $B ^ { ( q ) }$ is exhausted early due to upward deviations of the realized consumption above its expectation, an event whose probability decays exponentially in $\underline { { \tau _ { q } } } \Delta _ { 2 n _ { q } } ^ { 2 }$

Once this is established, standard SH analysis argument yield the following proposition. Define event:

$$
\mathcal { E } _ { q } : = \Big \{ 1 \in S ^ { ( q ) } \Big \} .
$$

Proposition 2.3. For each phase $q ,$

$$
\begin{array} { r } { \mathbb { P } \left( \overline { { \mathcal { E } } } _ { q + 1 } | \mathcal { E } _ { q } , \mathcal { A } _ { q } , \mathcal { F } _ { q } \right) \leq 4 \exp \left( - \underline { { \tau _ { q } } } \Delta _ { \frac { n _ { q } } { 4 } + 1 } ^ { 2 } \right) . } \end{array}
$$

Suppose that for all $q ^ { \prime } .$ , log 4 $\begin{array} { r } { ? \le \frac { 1 } { 8 } \underline { { \tau } } _ { q ^ { \prime } } \left( \frac { \Delta \frac { n _ { q ^ { \prime } } } { 4 } + 1 } { \log _ { 2 } ( K ) } \right) ^ { 2 } } \end{array}$ . To prove the main theorem, notice first that $\mathcal { E } _ { \log _ { 2 } ( K ) + 1 } = \{ { \phi = }$ 1}. Then:

$$
\begin{array} { r l } & { \mathbb { P } \left( \phi \neq 1 \right) = \mathbb { P } \left( \phi \neq 1 , \bigcap _ { q } { \mathcal { A } } _ { q } \right) + \mathbb { P } \left( \phi \neq 1 , \bigcap _ { q } { \mathcal { A } } _ { q } \right) } \\ & { \qquad \leq \mathbb { P } \left( \phi \neq 1 , \bigcap _ { q } { \mathcal { A } } _ { q } \right) + \mathbb { P } \left( \widehat { \bigcap _ { q } { \mathcal { A } } _ { q } } \right) } \end{array}
$$

We can upper bound the term on the right via Proposition 2.1. For the term on the left:

$$
\begin{array} { r } { \mathbb { P } \left( \phi \neq 1 , \displaystyle \bigcap _ { q } { \mathcal { A } } _ { q } \right) = \mathbb { P } \left( \displaystyle \overline { { \mathcal { E } _ { \log _ { 2 } ( K ) + 1 } } } , \bigcap _ { q } { \mathcal { A } } _ { q } \right) } \\ { \leq \displaystyle \sum _ { q = 1 } ^ { \log _ { 2 } ( K ) } \mathbb { P } \left( \overline { { \mathcal { E } _ { q + 1 } } } , \mathcal { E } _ { q } , { \mathcal { A } } _ { q } \right) } \\ { \leq \displaystyle \sum _ { q = 1 } ^ { \log _ { 2 } ( K ) } \mathbb { P } \left( \overline { { \mathcal { E } _ { q + 1 } } } | \mathcal { E } _ { q } , { \mathcal { A } } _ { q } \right) } \end{array}
$$

Then, Proposition 2.3 yields:

$$
\begin{array} { r l } & { \mathbb { P } \left( \phi \neq 1 , \displaystyle \bigcap _ { q } { \mathcal A } _ { q } \right) \leq \displaystyle \sum _ { q = 1 } ^ { \log _ { 2 } ( K ) } 4 \exp \left( - \frac { \tau _ { q } \Delta _ { \frac { 2 } { 4 } + 1 } ^ { 2 } } { 4 } \right) } \\ & { \qquad \leq 4 \log _ { 2 } ( K ) \exp \left( - \displaystyle \operatorname* { m i n } _ { q } \frac { \tau _ { q } \Delta _ { \frac { 2 } { 4 } + 1 } ^ { 2 } } { 4 } \right) } \\ & { \qquad \leq 4 \log _ { 2 } ( K ) ^ { 2 } \exp \left( - \displaystyle \frac { 1 } { 8 \log _ { 2 } ( K ) ^ { 2 } } \displaystyle \operatorname* { m i n } _ { q } \frac { \tau _ { q } \Delta _ { \frac { 2 } { 4 } + 1 } ^ { 2 } } { 4 } \right) } \end{array}
$$

Combining the two bounds:

$$
\mathbb { P } \left( \phi \neq 1 \right) \leq 7 \log _ { 2 } ( K ) ^ { 2 } \exp \left( - \frac { 1 } { 8 \log _ { 2 } ( K ) ^ { 2 } } \operatorname* { m i n } _ { q } \underline { { \tau _ { q } } } \Delta _ { \frac { n _ { q } } { 4 } + 1 } ^ { 2 } \right) .
$$

## 2.1 Proof of Proposition 2.1

For each phase q, we will see that event $A _ { q ^ { \prime } }$ holds due to a two stage chain of event. First, few arms in the surviving set have gap small than $\Delta _ { 2 n _ { q ^ { \prime } } }$ . In this case, most of them survive. Then, once the average of the gap a significant portion of the arms is order dominated by $\Delta _ { 2 n _ { a ^ { \prime } } }$ , it remains true for the subsequent phases, and that until phase $q ^ { \prime } .$ . Then, when everything goes well in the chain of event, after phase $q ^ { \prime } - 2$ , the event that the average gap of a lot of arms is order dominated by $\Delta _ { 2 n _ { a } }$ happens deterministically.

Before giving the full proof, we introduce each subevent in that chain and study how they imply each other.

For each phase pair $q ^ { \prime } > q + 1$ , define the good-arm conservation event:

$$
\mathcal { C } _ { q ^ { \prime } , q } : = \left\{ C _ { q } ( \Delta _ { 2 n _ { q ^ { \prime } } } ) \geq C _ { 1 } ( \Delta _ { 2 n _ { q ^ { \prime } } } ) \left( 1 - \frac { q } { 2 \log _ { 2 } ( K ) } \right) \right\} ,
$$

where $C _ { q } ( \Delta _ { 2 n _ { q ^ { \prime } } } )$ denotes the number of arms in $S ^ { ( q ) }$ with suboptimality gap at most $\Delta _ { 2 n _ { q ^ { \prime } } } . \ : { \mathcal { C } } _ { q ^ { \prime } , q }$ ensures that a controlled fraction of the near-optimal arms present at phase 1 remain in the surviving set at phase q and, essentially, that elimination has not discarded too many good arms. The event:

$$
\begin{array} { r } { \mathcal T _ { q ^ { \prime } , q } : = \left\{ \Delta _ { \frac { n _ { q } } { 4 } + 1 } ( q ) \leq 2 \Delta _ { 2 n _ { q ^ { \prime } } } \right\} , } \end{array}
$$

where $\Delta _ { \frac { n _ { q } } { A } + 1 } ( q )$ denotes the suboptimality gap of the $\left( \frac { n _ { q } } { 4 } + 1 \right)$ -th best arm within the surviving set $S ^ { ( q ) }$ , ensures the surviving set has concentrated suficiently around the top arms relative to the reference gap $\Delta _ { 2 n _ { a ^ { \prime } } }$ . Together, $\mathcal { C } _ { q ^ { \prime } , q }$ counts whether enough good arms remain, while $\mathcal { T } _ { q ^ { \prime } , q }$ measures whether the survivors are collectively good enough.

Proposition 2.4. For each phase pair $q ^ { \prime } > q + 1$ with $\begin{array} { r } { n _ { q } \geq 8 , i f \log ( 2 e \log K ) \leq \frac { \Delta _ { \frac { n _ { q } } { 4 } + 1 } ^ { 2 } \tau _ { q } } { 1 6 } , } \end{array}$

$$
\mathbb { P } \left( \overline { { \mathscr { C } } } _ { q ^ { \prime } , q + 1 } \mid \mathscr { C } _ { q ^ { \prime } , q } , \mathscr { A } _ { q } , \mathscr { F } _ { q } , \overline { { \mathscr { T } } } _ { q ^ { \prime } , q } \right) \leq 3 \exp \left( - \frac { 1 } { 1 6 } \tau _ { q } \Delta _ { \frac { n _ { q } } { 4 } + 1 } ^ { 2 } \right) .
$$

Proposition 2.4 establishes that good arm conservation propagates from phase q to $q + 1$ in the regime where $\mathcal { C } _ { q ^ { \prime } , q }$ holds but $\mathcal { T } _ { q ^ { \prime } , q }$ does not, provided the phase length $\tau _ { q }$ is suficient relative to $\Delta _ { \frac { n _ { q } } { A } + 1 } ^ { n _ { q } }$ . That is, when the surviving set has not yet concentrated around the top arms, elimination does not discard good arms with high probability.

For each phase pair $q ^ { \prime } > q .$ , define the event:

$$
\mathcal { M } _ { q ^ { \prime } , q } : = \left\{ \frac { 2 } { n _ { q } } \sum _ { i = 1 } ^ { n _ { q } / 2 } \Delta _ { i } ( q ) \leq \left( 2 + \frac { q } { \log _ { 2 } ( K ) } \right) \Delta _ { 2 n _ { q ^ { \prime } } } , \Delta _ { \frac { n _ { q } } { 4 } + 1 } \leq 2 \Delta _ { 2 n _ { q ^ { \prime } } } \right\} .
$$

The first half of $\mathcal { M } _ { q ^ { \prime } , q }$ bounds how good the top half of survivors are on average relative to the reference gap $\Delta _ { 2 n _ { q ^ { \prime } } }$ Notably, since each elimination retains the top half of $S ^ { ( q ) }$ to form $S ^ { ( q + 1 ) } , \mathcal { M } _ { q ^ { \prime } , q }$ directly characterizes the arm quality of the subsequent phase. For the second half of $\mathcal { M } _ { q ^ { \prime } , q }$ , note that it is implied by $\mathcal { T } _ { q ^ { \prime } , q }$ , and then remains true once it holds for a phase, as $n _ { q }$ is decreasing.

Proposition 2.5. For each phase $q ^ { \prime } > q$ with $n _ { q } \geq 8 ,$ , if log 4e $\begin{array} { r } { \le \frac { 1 } { 8 } \underline { { \tau } } _ { q } \left( \frac { \Delta \frac { n _ { q } } { 4 } + 1 } { \log _ { 2 } ( K ) } \right) ^ { 2 } } \end{array}$

$$
\mathbb { P } \left( \overline { { \mathcal { M } } } _ { q ^ { \prime } , q + 1 } | { \mathcal { M } } _ { q ^ { \prime } , q } , { \mathcal { A } } _ { q } , { \mathcal { F } } _ { q } \right) \leq 2 \exp \left( - \frac { 1 } { 8 \log _ { 2 } ( K ) ^ { 2 } } \underline { { \tau } } _ { q } \Delta _ { \frac { n q } { 4 } + 1 } ^ { 2 } \right)
$$

and for each q, $n _ { q } \geq 4 ,$ , if log $\begin{array} { r } { 4 e \le \frac { 1 } { 8 } \underline { { \tau } } _ { q } \left( \frac { \Delta \frac { n _ { q } } { 4 } + 1 } { \log _ { 2 } ( K ) } \right) ^ { 2 } } \end{array}$

$$
\mathbb { P } \left( \overline { { \mathcal { A } } } _ { q + 1 } | \mathcal { M } _ { q + 1 , q } , \mathcal { A } _ { q } , \mathcal { F } _ { q } \right) \leq 2 \exp \left( - \frac { 1 } { 8 \log _ { 2 } ( K ) ^ { 2 } } \underline { { \tau _ { q } } } \Delta _ { \frac { n _ { q } } { 4 } + 1 } ^ { 2 } \right) .
$$

Proposition 2.5 handles the post-transition regime. It establishes two results: first, that $\mathcal { M } _ { q ^ { \prime } , q }$ propagates to $\mathcal { M } _ { q ^ { \prime } , q + 1 }$ with high probability; second, that given $\mathcal { M } _ { q + 1 , q } ,$ , the arm quality event $A _ { q + 1 }$ holds with high probability.

Proposition 2.6. For each phase $q ^ { \prime } > q$ with $n _ { q } \geq 8 ,$ , if log 4e $\begin{array} { r } { \le \frac { 1 } { 8 } \underline { { \tau } } _ { q } \left( \frac { \Delta \frac { n _ { q } } { 4 } + 1 } { \log _ { 2 } ( K ) } \right) ^ { 2 } , } \end{array}$

$$
\mathbb { P } \left( \overline { { \mathcal { M } } } _ { q ^ { \prime } , q + 1 } | \mathcal { T } _ { q ^ { \prime } , q } , \mathcal { A } _ { q } , \mathcal { F } _ { q } \right) \leq 2 \exp \left( - \frac { 1 } { 8 \log _ { 2 } ( K ) ^ { 2 } } \underline { { \tau } } _ { q } \Delta _ { \frac { n _ { q } } { 4 } + 1 } ^ { 2 } \right) .
$$

Finally, we have the following deterministic claim.

## Claim 2.1. For any $q ^ { \prime } \in [ \log _ { 2 } ( K ) ] , \mathcal { C } _ { q ^ { \prime } , q ^ { \prime } - 2 }$ implies $\mathcal { T } _ { q ^ { \prime } , q ^ { \prime } - 2 } .$

Proof of claim 2.1. There are at least $2 n _ { q ^ { \prime } }$ arms whose gaps are no larger than $\Delta _ { 2 n _ { q ^ { \prime } } }$ . Hence

$$
C _ { 1 } ( \Delta _ { 2 n _ { q ^ { \prime } } } ) \geq 2 n _ { q ^ { \prime } } .
$$

On $\mathcal { C } _ { q ^ { \prime } , q } ,$ , it follows that

$$
C _ { q } ( \Delta _ { 2 n _ { q ^ { \prime } } } ) \geq C _ { 1 } ( \Delta _ { 2 n _ { q ^ { \prime } } } ) \left( 1 - \frac { q } { 2 \log _ { 2 } K } \right) \geq 2 n _ { q ^ { \prime } } \left( 1 - \frac { q } { 2 \log _ { 2 } K } \right) .
$$

By integrality, this implies $C _ { q } ( \Delta _ { 2 n _ { a ^ { \prime } } } ) \geq n _ { q ^ { \prime } } + 1$ . On the other hand, on $\overline { { \mathcal { T } } } _ { q ^ { \prime } , q } .$

$$
\Delta _ { \frac { n _ { q } } { 4 } + 1 } ( q ) > 2 \Delta _ { 2 n _ { q ^ { \prime } } } .
$$

Because the arms in $S ^ { ( q ) }$ are also ordered increasingly according to their gaps, this means that at most $n _ { q } / 4$ arms in $S ^ { ( q ) }$ have gap at most $2 \Delta _ { 2 n _ { q ^ { \prime } } }$ . In particular,

$$
C _ { q } ( \Delta _ { 2 n _ { q ^ { \prime } } } ) \leq C _ { q } ( 2 \Delta _ { 2 n _ { q ^ { \prime } } } ) \leq \frac { n _ { q } } { 4 } .
$$

Combining (1) and (2), we obtain the deterministic inclusion

$$
\mathcal { C } _ { q ^ { \prime } , q } \cap \overline { { \mathcal { T } } } _ { q ^ { \prime } , q } \subseteq \left\{ n _ { q ^ { \prime } } + 1 \leq \frac { n _ { q } } { 4 } \right\} .
$$

This implies that if $\mathcal { C } _ { q ^ { \prime } , q ^ { \prime } - 2 }$ holds, then so does $\mathcal { T } _ { q ^ { \prime } , q ^ { \prime } - 2 }$

Proof of Proposition 2.1. First, for every $q ^ { \prime } .$ , the event $\mathcal { C } _ { q ^ { \prime } , 1 }$ holds by definition. Since $S ^ { ( 1 ) }$ contains all the arms, $\mathcal { A } _ { 1 }$ and $\boldsymbol { A } _ { 2 }$ as well as $\mathcal { M } _ { 2 , 1 }$ <sub>1</sub> hold deterministically.

Fix now $q ^ { \prime } \geq 2$ , and suppose that

$$
\bigcap _ { q < q ^ { \prime } } \mathcal { A } _ { q }
$$

holds. We follow the evolution of the arms whose gaps are at most $\Delta _ { 2 n _ { a ^ { \prime } } }$

Starting from $\mathcal { C } _ { q ^ { \prime } , 1 }$ , consider a phase $q \leq q ^ { \prime } - 2$ . As long as $\mathcal { T } _ { q ^ { \prime } , q }$ does not hold, Proposition 2.4 shows that $\mathcal { C } _ { q ^ { \prime } , q }$ is propagated to $\mathcal { C } _ { q ^ { \prime } , q + 1 }$ , except with probability at most

$$
3 \exp \left( - \frac { 1 } { 1 6 } \underline { { { \tau _ { q } } } } \Delta _ { \scriptscriptstyle { \frac { n _ { q } } { 4 } + 1 } } ^ { 2 } \right) \leq 3 \exp \left( - \frac { 1 } { 8 \log _ { 2 } ( K ) ^ { 2 } } \underline { { { \tau _ { q } } } } \Delta _ { \scriptscriptstyle { \frac { n _ { q } } { 4 } + 1 } } ^ { 2 } \right) .
$$

By Claim 2.1, it is impossible to reach phase $q ^ { \prime } - 2$ while both $\mathcal { C } _ { q ^ { \prime } , q ^ { \prime } - 2 }$ and $\overline { { \mathcal { T } } } _ { q ^ { \prime } , q ^ { \prime } - 2 }$ hold. Consequently, provided none of the preceding conservation steps fails, $\mathcal { T } _ { q ^ { \prime } , q }$ must hold for some $q \leq q ^ { \prime } - 2 .$

At the first such phase, Proposition 2.6 gives $\mathcal { M } _ { q ^ { \prime } , q + 1 }$ , except with probability at most

$$
2 \exp \left( - \frac { 1 } { 8 \log _ { 2 } ( K ) ^ { 2 } } \underline { { \tau } } _ { q } \Delta _ { \underline { { n } } _ { q } } ^ { 2 } . . . \right) \leq 3 \exp \left( - \frac { 1 } { 8 \log _ { 2 } ( K ) ^ { 2 } } \underline { { \tau } } _ { q } \Delta _ { \underline { { n } } _ { q } } ^ { 2 } . . . \right) .
$$

Once $\mathcal { M } _ { q ^ { \prime } , q + 1 }$ holds, the first part of Proposition 2.5 propagates it through the subsequent phases:

$$
\mathcal { M } _ { q ^ { \prime } , q + 1 } \Longrightarrow \mathcal { M } _ { q ^ { \prime } , q + 2 } \Longrightarrow \cdots \Longrightarrow \mathcal { M } _ { q ^ { \prime } , q ^ { \prime } - 1 } ,
$$

with the failure probability at each phase bounded by

$$
2 \exp \left( - \frac { 1 } { 8 \log _ { 2 } ( K ) ^ { 2 } } \underline { { \tau } } _ { q } \Delta _ { \underline { { { n _ { q } } } } } ^ { 2 } { } _ { + 1 } \right) \leq 3 \exp \left( - \frac { 1 } { 8 \log _ { 2 } ( K ) ^ { 2 } } \underline { { \tau } } _ { q } \Delta _ { \underline { { { n _ { q } } } } } ^ { 2 } { } _ { + 1 } \right) .
$$

Finally, applying the second part of Proposition 2.5 at phase $q ^ { \prime } - 1$ yields $\boldsymbol { \mathcal { A } } _ { \boldsymbol { q } ^ { \prime } }$ from $\mathcal { M } _ { q ^ { \prime } , q ^ { \prime } - 1 }$ , with the same probability bound.

At every phase, conditional on the past, exactly one of the preceding three steps is relevant: conservation of $\mathcal { C } _ { q ^ { \prime } , q } ,$ transition from $\mathcal { T } _ { q ^ { \prime } , q }$ to $\mathcal { M } _ { q ^ { \prime } , q + 1 }$ , or propagation of $\mathcal { M } _ { q ^ { \prime } , q } .$ Therefore, after conditioning on $\mathcal { F } _ { q }$ and using the tower property, the probability that the construction leading to $\boldsymbol { \mathcal { A } } _ { \boldsymbol { q } ^ { \prime } }$ first fails at phase $q$ is at most

$$
3 \exp \left( - \frac { 1 } { 8 \log _ { 2 } ( K ) ^ { 2 } } \underline { { { \tau _ { q } } } } \Delta _ { \frac { n _ { q } } { 4 } + 1 } ^ { 2 } \right) .
$$

A union bound over the phases preceding $q ^ { \prime }$ consequently gives

$$
\mathbb { P } \left( \overline { { \mathcal { A } } } _ { q ^ { \prime } } \cap \bigcap _ { q < q ^ { \prime } } { A } _ { q } \right) \leq 3 \sum _ { q < q ^ { \prime } } \exp \left( - \frac { 1 } { 8 \log _ { 2 } ( K ) ^ { 2 } } \tau _ { \underline { { q } } } \Delta _ { \frac { n _ { q } } { 4 } + 1 } ^ { 2 } \right) .
$$

Applying the preceding estimate and a union bound over $q ^ { \prime } ,$ , we obtain

$$
\begin{array} { r l } & { \mathbb { P } \left( \overline { { \bigcap _ { q ^ { \prime } } A _ { q ^ { \prime } } } } \right) \leq 3 \sum _ { q ^ { \prime } } \sum _ { q < q ^ { \prime } } \exp \left( - \frac { 1 } { 8 \log _ { 2 } ( K ) ^ { 2 } } \underline { { \tau } } _ { q } \Delta _ { \frac { n _ { q } } { 4 } + 1 } ^ { 2 } \right) } \\ & { \qquad \leq 3 \log _ { 2 } ( K ) ^ { 2 } \exp \left( - \frac { 1 } { 8 \log _ { 2 } ( K ) ^ { 2 } } \operatorname* { m i n } \left\{ \underline { { \tau _ { q } } } \Delta _ { \frac { n _ { q } } { 4 } + 1 } ^ { 2 } \right\} \right) . } \end{array}
$$

Taking q in the statement to be a phase at which $\underline { { \tau _ { q } } } \Delta _ { \frac { n _ { q } } { 4 } + 1 } ^ { 2 }$ is smallest gives the claimed bound.

## 2.2 Proof of the propositions

We start with the following useful claim, which can be derived from Ville’s maximal inequality, and a direct consequence of Corollay 3 in Howard et al. (2020).

Claim 2.2. Let $( X _ { t } ) _ { t \in \mathbb { N } }$ be a sequence of independent random variables in $[ 0 , 1 ] .$ . For any fixed threshold $a \geq 0$ and any m,

$$
\mathbb { P } \left( \exists t \geq m : \sum _ { i \leq t } X _ { i } \geq \sum _ { i \leq t } \mathbb { E } [ X _ { i } ] + t a \right) \leq \exp \left( - 2 m a ^ { 2 } \right) .
$$

Note that a lower tail version of that claim can be obtained by applying the result to $1 - X _ { i }$

Proof of Proposition 2.2. Let $I _ { s } \in S ^ { ( q ) }$ be the arm pulled at the s-th pull of the phase, and let $X _ { s } \in [ 0 , 1 ]$ be the corresponding reward. Define $M _ { s }$ as the deviation from expectation on arm pull s, i.e:

$$
M _ { s } : = \sum _ { u = 1 } ^ { s } \mu _ { I _ { u } } - X _ { u } .
$$

Since the algorithm keeps a margin to complete round robin cycles, if $t _ { q } < \tau _ { q } .$ , this implies that the regret budget (minus the margin) was exhausted at the end of some round-robin cycle. At the end of a complete round-robin cycle, every arm in $S ^ { ( q ) }$ has been pulled equally often, so, under event $\scriptstyle A _ { q } .$ , the total expected regret consumption at a pull s which is the end of a round robin cycle is at most $s ( 3 \Delta _ { 2 n _ { q } } - \Delta _ { 0 } ) \leq 3 s \Delta _ { 2 n _ { q } }$ . Therefore, on the event $\mathcal { A } _ { q } , \mathrm { i f } t _ { q } \le \underline { { \tau _ { q } } } ,$ then for some $s \leq n _ { q } \underline { { \tau _ { q } } } :$

$$
M _ { s } \geq 3 \underline { { \tau _ { q } } } n _ { q } \Delta _ { 2 n _ { q } } .
$$

$M _ { s }$ is a martingale with respect to the filtration generated by $\mathcal { F } _ { q }$ and the rewards observed during phase $q ,$ with bounded increments. By maximal Azuma-Hoefding inequality:

$$
\mathbb { P } \left( \operatorname* { m a x } _ { 1 \leq s \leq n _ { q } \frac { \tau _ { q } } { 2 } } M _ { s } \geq 3 \underline { { \tau _ { q } } } n _ { q } \Delta _ { 2 n _ { q } } \right) \leq \exp \left( - 1 8 \underline { { \tau _ { q } } } n _ { q } \Delta _ { 2 n _ { q } } ^ { 2 } \right) .
$$

Proof of Proposition 2.4. The proof of this proposition relies on the following lemma, whose proof we delay:

Lemma 2.1 (Specialized anytime version of Lemma 9.1 from "Revisiting simple regret"). Suppose that, after t pulls per arm, uniform sampling outputs the $K / 2$ arms with the largest empirical means. Let $C _ { t } ( \epsilon )$ be the set ofϵ-good arms in the output, and let

$$
g ( \epsilon ) : = | \{ i : \mu _ { i } \geq \mu ^ { \star } - \epsilon \} | .
$$

Assume that $\begin{array} { r } { \frac { K } { 2 } > g ( \epsilon ) } \end{array}$ . Let ℓ satisfy

$$
g ( \epsilon ) \leq \ell < { \frac { K } { 2 } } ,
$$

and define

$$
d _ { \ell } : = \mu _ { g ( \epsilon ) } - \mu _ { \ell + 1 } .
$$

Then, for every integer $k \in \{ 1 , \ldots , g ( \epsilon ) \}$ and every $\underline { { \tau } } \geq 1$ -，

$$
\mathbb { P } \left( \exists t \ge \underline { { \tau } } s . t . \ | C _ { t } ( \epsilon ) | \le g ( \epsilon ) - k \right) \le \left( \begin{array} { l } { g ( \epsilon ) } \\ { k } \end{array} \right) \exp \left( - k \frac { d _ { \ell } ^ { 2 } \tau } { 2 } \right) + \left( \begin{array} { l } { K - \ell } \\ { \frac { K } { 2 } - \ell + k } \end{array} \right) \exp \left( - \left( \frac { K } { 2 } - \ell + k \right) \frac { d _ { \ell } ^ { 2 } \tau } { 2 } \right) ,
$$

with the convention that $\binom { a } { b } = 0 w h e n b > a .$

We apply the above lemma with $\begin{array} { r } { \epsilon = \Delta _ { 2 n _ { q ^ { \prime } } } , K = n _ { q } , \ell = \frac { n _ { q } } { 4 } } \end{array}$ . Let us first derive the value of $d _ { \ell }$ with that instantiation. By definition,

$$
d _ { \ell } \ge \mu ^ { * } - \Delta _ { 2 n _ { q ^ { \prime } } } - \mu _ { \frac { n _ { q } } { 4 } + 1 } ( q ) = \Delta _ { \frac { n _ { q } } { 4 } + 1 } ( q ) - \Delta _ { 2 n _ { q ^ { \prime } } } .
$$

Event $\overline { { \mathcal { T } } } _ { q ^ { \prime } , q }$ then implies:

$$
d _ { \ell } \geq \frac { 1 } { 2 } \Delta _ { \frac { n _ { q } } { 4 } + 1 } ( q ) \geq \frac { 1 } { 2 } \Delta _ { \frac { n _ { q } } { 4 } + 1 } .
$$

Then, for any $k \le C _ { q } ( \Delta _ { 2 n _ { q ^ { \prime } } } )$

$$
\begin{array} { r l } & { \mathbb { P } \left( C _ { q + 1 } ( \Delta _ { 2 n _ { q ^ { \prime } } } ) \leq C _ { q } ( \Delta _ { 2 n _ { q ^ { \prime } } } ) - k | \mathcal { C } _ { q ^ { \prime } , q } , A _ { q } , \mathcal { F } _ { q } , \overline { { \mathcal { T } } } _ { q ^ { \prime } , q } \right) \leq \left( { C _ { q } ( \Delta _ { 2 n _ { q ^ { \prime } } } ) } \right) \exp \left( { - k \frac { \Delta _ { \frac { n _ { q } } { 3 4 } + 1 } ^ { 2 } \overline { { C } } _ { q } } { 8 } } \right) } \\ & { \qquad + \left( \frac { 3 } { \frac { n _ { q } } { 4 } } + k \right) \exp \left( { - ( \frac { n _ { q } } { 4 } + k ) \frac { \Delta _ { \frac { n _ { q } } { 3 4 } + 1 } ^ { 2 } \overline { { \mathcal { T } } } _ { q } } { 8 } } \right) , } \\ & { \qquad + \mathbb { P } \left( { t _ { q } \leq \underline { { \tau _ { q } } } | \mathcal { C } _ { q ^ { \prime } , q } } , A _ { q } , \mathcal { F } _ { q } , \overline { { \mathcal { T } } } _ { q ^ { \prime } , q } \right) } \end{array}
$$

Instantiating $\begin{array} { r } { k = \lceil \frac { 1 } { 2 \log _ { 2 } ( K ) } C _ { q } ( \Delta _ { 2 n _ { q ^ { \prime } } } ) \rceil } \end{array}$ and the inequality $\begin{array} { r } { \binom { x } { y } \leq \left( \frac { e x } { y } \right) ^ { y } } \end{array}$ , this yields:

$$
\begin{array} { r l } & { \mathbb { P } \left( C _ { q + 1 } ( \Delta _ { 2 n _ { q ^ { \prime } } } ) \leq C _ { q } ( \Delta _ { 2 n _ { q ^ { \prime } } } ) - k | \mathcal { C } _ { q ^ { \prime } , q } , \mathcal { A } _ { q } , \mathcal { F } _ { q } , \overline { { \mathcal { T } } } _ { q ^ { \prime } , q } \right) } \\ & { \leq \exp \left[ \Gamma \frac { C _ { q } ( \Delta _ { 2 n _ { q ^ { \prime } } } ) } { 2 \log _ { 2 } K } \Big ] \left( \log ( 2 e \log K ) - \frac { \Delta _ { \frac { n _ { q } } { 4 } + 1 } ^ { 2 } \overline { { \mathcal { T } } } _ { q } } { 8 } \right) \right] } \\ & { \quad + \exp \left[ \left( \frac { n _ { q } } { 4 } + \Gamma \frac { C _ { q } ( \Delta _ { 2 n _ { q ^ { \prime } } } ) } { 2 \log K } \big ] \right) \left( \log ( 3 e ) - \frac { \Delta _ { \frac { n _ { q } } { 4 } + 1 } ^ { 2 } \overline { { \mathcal { T } } } _ { q } } { 8 } \right) \right] + \mathbb { P } \left( t _ { q } \leq \underline { { \tau _ { q } } } | \mathcal { C } _ { q ^ { \prime } , q } , \mathcal { A } _ { q } , \mathcal { F } _ { q } , \overline { { \mathcal { T } } } _ { q ^ { \prime } , q } \right) } \end{array}
$$

Finally, the condition log(2e log $K ) \leq \frac { \Delta _ { \frac { n _ { q } } { 4 } + 1 } ^ { 2 } \underline { { \tau } } _ { q } } { 1 6 }$ together with Proposition 2.2 yields:

$$
\mathbb { P } \left( C _ { q + 1 } ( \Delta _ { 2 n _ { q ^ { \prime } } } ) \leq C _ { q } ( \Delta _ { 2 n _ { q ^ { \prime } } } ) - k | \mathcal { C } _ { q ^ { \prime } , q } , A _ { q } , \mathcal { F } _ { q } , \overline { { \mathcal { T } } } _ { q ^ { \prime } , q } \right) \leq 3 \exp \left( - \frac { \Delta _ { \frac { n _ { q } } { 4 } + 1 } ^ { 2 } \frac { T _ { q } } { 4 } } { 1 6 } \right)
$$

ProofofProposition 2.5. The proof relies on a combination of Proposition 2.2 and the following intermediate result whose proof is delayed:

Lemma 2.2 (Anytime bound on top-arms mean decay). Consider a set of K arm, sampled in a round robinfashion. At round t (where each arm has been sampled t times), denote $A ( t )$ the set of $K / 2$ arms with the largest empirical mean, ties broken at random. For $1 \leq s \leq K / 2 ,$ , let

$$
B _ { s } : = \{ 1 , \ldots , s \}
$$

be the set of the true top s arms, and let $B _ { s } ( t )$ be the set of the s arms with largest true means inside $A ( t )$ , ties broken in favor of elements in $B _ { s }$ . Define

$$
\mu _ { B _ { s } } : = \frac { 1 } { s } \sum _ { i \in B _ { s } } \mu _ { i } , \qquad \mu _ { B _ { s } ( t ) } : = \frac { 1 } { s } \sum _ { i \in B _ { s } ( t ) } \mu _ { i } .
$$

Then, for every $\epsilon > 0$ and every $\underline { { \tau } } \geq 1$

$$
\mathbb { P } \left( \exists t \geq \underline { { \tau } } : \mu _ { B _ { s } ( t ) } \leq \mu _ { B _ { s } } - \epsilon \right) \leq \exp \left( s \log \frac { e K } { s } - \underline { { \tau } } s \epsilon ^ { 2 } \right) .
$$

To obtain the first bound, we instantiate the above lemma with $\begin{array} { r } { s = { \frac { n _ { q } } { 4 } } , K = n _ { q } , \epsilon = { \frac { \Delta _ { 2 n _ { q ^ { \prime } } } } { \log _ { 2 } ( K ) } } } \end{array}$ and $\underline { { \tau } } = \tau _ { q }$ . We have:

$$
\begin{array} { l l } { \mathbb { P } \left( \overline { { \mathcal { M } } } _ { q ^ { \prime } , q + 1 } | { \mathcal { M } } _ { q ^ { \prime } , q } , { \mathcal { A } } _ { q } , { \mathcal { F } } _ { q } \right) \leq \mathbb { P } \left( \exists t \geq { \underline { { \tau } } } _ { q } : \mu _ { B _ { { \frac { n _ { q } } { 4 } } } ( t ) } \leq \mu _ { B _ { { \frac { n _ { q } } { 4 } } } } - \frac { \Delta _ { 2 n _ { q ^ { \prime } } } } { \log _ { 2 } ( K ) } | { \mathcal { M } } _ { q ^ { \prime } , q } , { \mathcal { A } } _ { q } , { \mathcal { F } } _ { q } \right) } \\ { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad + \mathbb { P } \left( t _ { q } \leq { \tau } _ { q } | { \mathcal { M } } _ { q ^ { \prime } , q } , { \mathcal { A } } _ { q } , { \mathcal { F } } _ { q } \right) } \end{array}
$$

We upper bound the term in the fist line via Lemma 2.2 and the second term via Proposition 2.2.

$$
\begin{array} { r l } & { \mathbb { P } \left( \overline { { \mathcal { M } } } _ { q ^ { \prime } , q + 1 } \big | \mathcal { M } _ { q ^ { \prime } , q } , \mathcal { F } _ { q } \right) \leq \exp \left( \frac { n _ { q } } { 4 } \log 4 e - \underline { { \tau } } _ { q } \frac { n _ { q } } { 4 } \left( \frac { \Delta _ { 2 n _ { q ^ { \prime } } } } { \log _ { 2 } ( K ) } \right) ^ { 2 } \right) } \\ & { \qquad + \exp \left( - 1 8 \underline { { \tau } } _ { q } \Delta _ { 2 n _ { q } } ^ { 2 } \right) } \end{array}
$$

Using the condition $\Delta _ { \frac { n _ { q } } { 4 } + 1 } \leq 2 \Delta _ { 2 n _ { q ^ { \prime } } }$ and the fact $\Delta _ { \frac { n _ { q } } { 4 } + 1 } \leq \Delta _ { 2 n _ { q } }$ , we obtain

$$
\begin{array} { r l } & { \mathbb { P } \left( \overline { { \mathcal { M } } } _ { q ^ { \prime } , q + 1 } \big | { \mathcal { M } } _ { q ^ { \prime } , q } , \mathcal { F } _ { q } \right) \leq \exp \left( \frac { n _ { q } } { 4 } \log 4 e - \underline { { \tau } } _ { q } \frac { n _ { q } } { 1 6 } \left( \frac { \Delta \frac { n _ { q } } { 4 } + 1 } { \log _ { 2 } ( K ) } \right) ^ { 2 } \right) } \\ & { \qquad + \exp \left( - 1 8 \underline { { \tau } } _ { q } \Delta _ { \frac { n _ { q } } { 4 } + 1 } ^ { 2 } \right) } \end{array}
$$

Finally, from the condition log $\begin{array} { r } { 4 e \le \frac { 1 } { 8 } \underline { { \tau } } _ { q } \left( \frac { \Delta \frac { n _ { q } } { 4 } + 1 } { \log _ { 2 } ( K ) } \right) ^ { 2 } } \end{array}$ , and $n _ { q } \geq 8$ we obtain:

$$
\mathbb { P } \left( \overline { { \mathcal { M } } } _ { q ^ { \prime } , q + 1 } \vert \mathcal { M } _ { q ^ { \prime } , q } , \mathcal { F } _ { q } \right) \leq 2 \exp \left( - \frac { 1 } { 4 \log _ { 2 } ( K ) ^ { 2 } } \underline { { \tau } } _ { q } \Delta _ { \frac { n _ { q } } { 4 } + 1 } ^ { 2 } \right)
$$

For the second bound, we repeat the proof with $\begin{array} { r } { s = \frac { n _ { q } } { 2 } } \end{array}$ and use the fact that $\frac { q + 1 } { \log _ { 2 } ( K ) } \leq 1$

ProofofProposition 2.6. Note that event $\mathcal { T } _ { q ^ { \prime } , q }$ automatically implies that

$$
\frac { 2 } { n _ { q + 1 } } \sum _ { i = 1 } ^ { n _ { q + 1 } / 2 } \Delta _ { i } ( q ) \leq 2 \Delta _ { 2 n _ { q } ^ { \prime } } \mathrm { ~ a n d ~ } \Delta _ { \frac { n _ { q } } { 4 } + 1 } \leq \Delta _ { \frac { n _ { q } } { 4 } + 1 } ( q ) \leq 2 \Delta _ { 2 n _ { q ^ { \prime } } } .
$$

Therefore:

$$
\begin{array} { r l } & { \mathbb { P } \left( \overline { { \mathcal { M } } } _ { q ^ { \prime } , q + 1 } | \mathcal { T } _ { q ^ { \prime } , q } , \mathcal { F } _ { q } , \mathcal { A } _ { q } \right) \leq \mathbb { P } \left( \exists t \geq \underline { { \tau } } _ { q } : \mu _ { B _ { \frac { n _ { q } } { 4 } } ( t ) } \leq \mu _ { B _ { \frac { n _ { q } } { 4 } } } - \frac { \Delta _ { 2 n _ { q ^ { \prime } } } } { \log _ { 2 } ( K ) } | \mathcal { T } _ { q ^ { \prime } , q } , \mathcal { A } _ { q } , \mathcal { F } _ { q } \right) } \\ & { \qquad + \mathbb { P } \left( t _ { q } \leq \tau _ { q } | \mathcal { T } _ { q ^ { \prime } , q } , \mathcal { A } _ { q } , \mathcal { F } _ { q } \right) } \end{array}
$$

We can then proceed repeating the steps from the proof of 2.5.

Proof of Proposition 2.3. First suppose that $n _ { q } \geq 4 .$ Put

$$
i _ { q } : = \frac { n _ { q } } { 4 } + 1 ,
$$

and define $S ^ { \prime ( q ) }$ as the set of arms in $S ^ { ( q ) }$ , excluding the ${ \scriptstyle { \frac { 1 } { 4 } } } n _ { q }$ arms with the largest mean. Every arm $i \in S ^ { \prime ( q ) }$ has global reward rank at least $i _ { q } ,$ and hence

$$
\Delta _ { i } \geq \Delta _ { i _ { q } } .
$$

Let

$$
N _ { q } : = \sum _ { i \in S ^ { \prime } ^ { ( q ) } } { \bf 1 } \{ \widehat { \mu } _ { i } ^ { q } \geq \widehat { \mu } _ { 1 } ^ { q } \} .
$$

If the best arm is eliminated in phase $q ,$ it must be the case that at least half the arms of $S ^ { ( q ) }$ have their empirical average larger than its empirical average. In particular, this implies

$$
N _ { q } \geq \frac { n _ { q } } { 4 } = \frac { | S ^ { \prime ( q ) } | } { 3 } .
$$

Therefore,

$$
\mathbb { P } \left( \overline { { \mathcal { E } } } _ { q + 1 } \left| \mathcal { F } _ { q } , \mathcal { E } _ { q } , \mathcal { A } _ { q } \right. \right) \leq \mathbb { P } ( t _ { q } < \underline { { \tau _ { q } } } \mid F _ { q } , \mathcal { E } _ { q } , \mathcal { A } _ { q } ) + \mathbb { P } \left( N _ { q } \geq \frac { | { S } ^ { \prime ( q ) } | } { 3 } , \ t _ { q } \geq \underline { { \tau _ { q } } } \left| \mathcal { F } _ { q } , \mathcal { E } _ { q } , \mathcal { A } _ { q } \right. \right)
$$

By Proposition 2.2,

$$
\begin{array} { r } { \mathbb { P } \left( t _ { q } < \underline { { \tau _ { q } } } | \boldsymbol { \mathcal { A } } _ { q } , \mathcal { F } _ { q } \right) \le \exp \left( - 1 8 \underline { { \tau _ { q } } } \Delta _ { 2 n _ { q } } ^ { 2 } \right) } \end{array}
$$

By Markov’s inequality:

$$
\mathbb { P } \left( N _ { q } \geq \frac { | S ^ { \prime ( q ) } | } { 3 } , t _ { q } \geq \underline { { \tau _ { q } } } \middle | \mathcal { F } _ { q } , \mathcal { E } _ { q } \right) \leq \frac { 3 } { | S ^ { \prime ( q ) } | } \mathbb { E } \left[ N _ { q } \mathbf { 1 } \{ t _ { q } \geq \underline { { \tau _ { q } } } \} \middle | \mathcal { F } _ { q } , \mathcal { E } _ { q } \right] .
$$

For every $i \in S ^ { \prime ( q ) }$ , the event

$$
\left\{ t _ { q } \geq \underline { { \tau _ { q } } } , \widehat { \mu } _ { i } ^ { q } \geq \widehat { \mu } _ { 1 } ^ { q } \right\}
$$

is contained in the event that the empirical mean of arm i exceeds that of arm 1 at some common sample size $t _ { q } \geq \tau _ { q }$ Hence Lemma 1.5 gives

$$
\begin{array} { r } { \mathbb { P } \Big ( t _ { q } \geq \underline { { \tau _ { q } } } , \widehat { \mu } _ { i } ^ { q } \geq \widehat { \mu } _ { 1 } ^ { q } \mid \mathcal { F } _ { q } , \mathcal { E } _ { q } \Big ) \leq \exp \Big ( { - \underline { { \tau _ { q } } } \Delta _ { i } ^ { 2 } } \Big ) . } \end{array}
$$

It follows that

$$
\begin{array} { r } { \mathbb { E } \left[ N _ { q } { \bf 1 } \{ t _ { q } \geq \underline { { \tau _ { q } } } \} \Big | \mathcal { F } _ { q } , \mathcal { E } _ { q } \right] \leq | S ^ { \prime ( q ) } | \exp \Big ( { - \underline { { \tau _ { q } } } \Delta _ { i _ { q } } ^ { 2 } } \Big ) , } \end{array}
$$

and therefore

$$
\begin{array} { r } { \mathbb { P } \left( \overline { { \mathscr { E } } } _ { q + 1 } \big | \mathcal { F } _ { q } , \mathscr { E } _ { q } , \mathscr { A } _ { q } \right) \leq \exp \left( - 1 8 \underline { { \tau _ { q } } } \Delta _ { 2 n _ { q } } ^ { 2 } \right) + 3 \exp \left( - \underline { { \tau _ { q } } } \Delta _ { i _ { q } } ^ { 2 } \right) \leq 4 \exp \left( - \underline { { \tau _ { q } } } \Delta _ { i _ { q } } ^ { 2 } \right) . } \end{array}\tag{5}
$$

It remains to deal with the case $n _ { q } = 2$ . When $n _ { q } = 2$ , there is only one competing arm. Apply the adaptive Hoefding lemma directly, which also yields:

$$
\begin{array} { r } { \mathbb { P } \left( \overline { { \mathscr { E } } } _ { q + 1 } \big | \mathscr { F } _ { q } , \mathscr { E } _ { q } , \mathscr { A } _ { q } \right) \leq 4 \exp \left( - \underline { { \tau _ { q } } } \Delta _ { i _ { q } } ^ { 2 } \right) . } \end{array}
$$

## 2.3 Proof of the Helper Lemmas

Proof of Lemma 2.1. Fix a time t. If

$$
| C _ { t } ( \epsilon ) | \le g ( \epsilon ) - k ,
$$

then at least k of the $g ( \epsilon )$ ϵ-good arms are not selected. Since these $g ( \epsilon )$ arms are contained in the first ℓ arms, the output contains at most $\ell - k$ arms from [ℓ]. Because the output has size $\begin{array} { l } { { \frac { K } { 2 } } } \end{array}$ , it must contain at least

$$
{ \frac { K } { 2 } } - ( \ell - k ) = { \frac { K } { 2 } } - \ell + k
$$

arms from $\{ \ell + 1 , \ldots , K \}$

Thus, on the event $| C _ { t } ( \epsilon ) | \leq g ( \epsilon ) - k .$ , there exist sets

$$
A \subset [ g ( \epsilon ) ] , \qquad | A | = k ,
$$

and

$$
B \subset \{ \ell + 1 , \ldots , K \} , \qquad | B | = { \frac { K } { 2 } } - \ell + k ,
$$

such that all arms in A are excluded from the output and all arms in $B$ are included in the output. Since the output consists of the $\begin{array} { l } { { \frac { K } { 2 } } } \end{array}$ arms with largest empirical means, for every $i \in A$ and every $j \in B$

$$
\widehat { \mu } _ { i , t } \leq \widehat { \mu } _ { j , t } .
$$

Moreover, for such $i , j ,$

$$
\mu _ { i } - \mu _ { j } \geq \mu _ { g ( \epsilon ) } - \mu _ { \ell + 1 } = d _ { \ell } .
$$

Therefore, for each pair $( i , j ) \in A \times B$ , the inequality $\widehat { \mu } _ { i , t } \leq \widehat { \mu } _ { j , t }$ <sub>t</sub> implies

$$
\widehat { \mu } _ { i } ( t ) \leq \mu _ { i } - \frac { d _ { \ell } } { 2 } \quad \mathrm { o r } \quad \widehat { \mu } _ { j } ( t ) \geq \mu _ { j } + \frac { d _ { \ell } } { 2 } .
$$

Indeed, if both inequalities failed, then

$$
\widehat { \mu } _ { i } ( t ) > \mu _ { i } - \frac { d _ { \ell } } { 2 } \geq \mu _ { j } + \frac { d _ { \ell } } { 2 } > \widehat { \mu } _ { j } ( t ) ,
$$

a contradiction.

By distributivity,

$$
\bigcap _ { i \in A , j \in B } \left\{ { \widehat { \mu } } _ { i , t } \leq \mu _ { i } - { \frac { d _ { \ell } } { 2 } } { \mathrm { ~ o r ~ } } { \widehat { \mu } } _ { j , t } \geq \mu _ { j } + { \frac { d _ { \ell } } { 2 } } \right\}
$$

implies,

$$
\left\{ \forall i \in A , { \widehat { \mu } } _ { i } ( t ) \leq \mu _ { i } - { \frac { d _ { \ell } } { 2 } } \right\} \cup \left\{ \forall j \in B , { \widehat { \mu } } _ { j , t } \geq \mu _ { j } + { \frac { d _ { \ell } } { 2 } } \right\} .
$$

Hence,

$$
\begin{array} { l } { \{ \exists t \geq \underline { { \tau } } : | C _ { t } ( \epsilon ) | \leq g ( \epsilon ) - k \} } \\ { \subseteq \left\{ \exists A \subset [ g ( \epsilon ) ] , | A | = k , \exists t \geq \underline { { \tau } } : \forall i \in A , \widehat { \mu } _ { i } ( t ) \leq \mu _ { i } - \frac { d _ { \ell } } { 2 } \right\} } \\ { \cup \left\{ \exists B \subset \{ \ell + 1 , \dots , K \} , | B | = \displaystyle \frac { K } { 2 } - \ell + k , \exists t \geq \underline { { \tau } } : \forall j \in B , \widehat { \mu } _ { j } ( t ) \geq \mu _ { j } + \frac { d _ { \ell } } { 2 } \right\} . } \end{array}
$$

Fix a set $A \subseteq [ K ]$ . For any $i \in A ,$ , by Claim 2.2,

$$
\mathbb { P } \left( \exists t \geq \underline { { \tau } } \operatorname { s . t . } \hat { \mu } _ { i } ( t ) \leq \mu _ { i } - \frac { d _ { \ell } } { 2 } \right) \leq \exp \left( - \frac { \underline { { \tau } } d _ { \ell } ^ { 2 } } { 2 } \right) .
$$

By independence,

$$
\mathbb { P } \left( \exists t \geq \underline { { \tau } } \mathrm { ~ s . t . ~ } \forall i \in A , \hat { \mu } _ { i } ( t ) \leq \mu _ { i } - \frac { d _ { \ell } } { 2 } \right) \leq \exp \left( - \frac { \underline { { \tau } } | A | d _ { \ell } ^ { 2 } } { 2 } \right) .
$$

Similarly, for a fixed set B,

$$
\mathbb { P } \left( \exists t \geq \underline { { \tau } } \mathrm { ~ s . t . ~ } \forall i \in B , \hat { \mu } _ { i } ( t ) \geq \mu _ { i } + \frac { d _ { \ell } } { 2 } \right) \leq \exp \left( - \frac { \underline { { \tau } } | B | d _ { \ell } ^ { 2 } } { 2 } \right) .
$$

Taking a union bound over all possible sets A and B, and applying the anytime concentration inequalities above, yields

$$
\begin{array} { r l r } {  { \mathbb { P } ( \exists t \geq \underline { { \tau } } : | C _ { t } ( \epsilon ) | \leq g ( \epsilon ) - k ) \leq \binom { g ( \epsilon ) } { k } \exp ( - k \frac { d _ { \ell } ^ { 2 } \underline { { \tau } } } { 2 } ) } } \\ & { } & { + ( \begin{array} { c } { K - \ell } \\ { \frac { K } { 2 } - \ell + k } \end{array} ) \exp ( - ( \frac { K } { 2 } - \ell + k ) \frac { d _ { \ell } ^ { 2 } \underline { { \tau } } } { 2 } ) . } \end{array}
$$

Proof of Lemma 2.2. Define

$$
U ( t ) : = B _ { s } \setminus B _ { s } ( t ) , \qquad V ( t ) : = B _ { s } ( t ) \setminus B _ { s } .
$$

By definition,

$$
\sum _ { i \in B _ { s } } \mu _ { i } - \sum _ { i \in B _ { s } ( t ) } \mu _ { i } = \sum _ { i \in U ( t ) } \mu _ { i } - \sum _ { i \in V ( t ) } \mu _ { i } .
$$

Hence, $\mu _ { B _ { t } ( s ) } \leq \mu _ { B } - \epsilon$ implies

$$
\sum _ { i \in U ( t ) } \mu _ { i } - \sum _ { i \in V ( t ) } \mu _ { i } \geq s \epsilon .
$$

Since $B _ { s } ( t )$ is the set with the largest empirical means and the arms in $V ( t )$ are selected while the arms in $U ( t )$ are not, this implies:

$$
\sum _ { i \in V ( t ) } \hat { \mu } _ { i } ( t ) - \sum _ { i \in U ( t ) } \hat { \mu } _ { i } ( t ) \geq 0 .
$$

Putting these two equations together, we obtain:

$$
\sum _ { i \in U ( t ) } \left( \mu _ { i } - \hat { \mu } _ { i } ( t ) \right) + \sum _ { i \in V ( t ) } \left( \hat { \mu } _ { i } ( t ) - \mu _ { i } \right) \geq s \epsilon ,
$$

or, adding $B _ { s } \setminus U ( t )$ to each set:

$$
\sum _ { i \in B _ { s } } ( \mu _ { i } - \hat { \mu } _ { i } ( t ) ) + \sum _ { i \in V ( t ) \cup ( B _ { s } \setminus U ( t ) ) } \left( \hat { \mu } _ { i } ( t ) - \mu _ { i } \right) \geq s \epsilon .
$$

Therefore:

$$
\left\{ \mu _ { B _ { t } ( s ) } \leq \mu _ { B } - \epsilon \right\} \subseteq \left\{ \exists T , | T | = s \mathrm { ~ a n d ~ } \sum _ { i \in B _ { s } } ( \mu _ { i } - \hat { \mu } _ { i } ( t ) ) + \sum _ { i \in T } ( \hat { \mu } _ { i } ( t ) - \mu _ { i } ) \geq s \epsilon \right\} .
$$

Now, for a fixed set $T ,$ we prove the following claim.

Claim 2.3. For every $\epsilon \geq 0$ and $\underline { { \tau } } > 0$

$$
\mathbb { P } \left( \exists t \geq \underline { { \tau } } : \sum _ { i \in B _ { s } } \left( \mu _ { i } - \widehat { \mu } _ { i } ( t ) \right) + \sum _ { i \in T } \left( \widehat { \mu } _ { i } ( t ) - \mu _ { i } \right) \geq s \epsilon \right) \leq \exp \left( - s \underline { { \tau } } \epsilon ^ { 2 } \right) .
$$

Proof. Let $\ell = s - | B _ { s } \cap T |$ . Now, we can apply Claim 2.2 with $m = 2 \ell \underline { { \tau } }$ and $\begin{array} { r } { a = \frac { s \epsilon } { 2 \ell } } \end{array}$ , and get:

$$
\mathbb { P } \left( \exists t \geq \underline { { \tau } } : \sum _ { i \in B _ { s } } \left( \mu _ { i } - \widehat { \mu } _ { i } ( t ) \right) + \sum _ { i \in T } \left( \widehat { \mu } _ { i } ( t ) - \mu _ { i } \right) \geq s \epsilon \right) \leq \exp \left( - \frac { s ^ { 2 } } { \ell } \underline { { \tau } } \epsilon ^ { 2 } \right) .
$$

The claim then follows from $\ell \leq s .$

Taking a union bound over all $\binom { K } { s }$ deterministic sets $T$ gives

$$
\mathbb { P } \left( \exists t \geq \underline { { \tau } } : \mu _ { B _ { t } ( s ) } \leq \mu _ { B } - \epsilon \right) \leq \binom { K } { s } \exp \left( - \underline { { t } } s \epsilon ^ { 2 } \right) .
$$

Finally, using

$$
{ \binom { K } { s } } \leq \left( { \frac { e K } { s } } \right) ^ { s }
$$

yields

$$
\mathbb { P } \left( \exists t \geq \underline { { \tau } } : \mu _ { B _ { t } ( s ) } \leq \mu _ { B } - \epsilon \right) \leq \exp \left( s \log \frac { e K } { s } - \underline { { \tau } } s \epsilon ^ { 2 } \right) .
$$

![](images/003b66d390a5cc912cf1866b60c62dfd953d5f068631888e892994fba8f12e70.jpg)  
Figure 1: Reward distributions in the four baseline configurations of the synthetic environments.

![](images/4b0e5340e14d505f3780be93a17aa1c6b3ff46bf41c0666241aa98e9977aec44.jpg)  
Figure 2: Arm reward distributions in the four real-world environments: the three New Yorker Caption Contests (c511, c538, c780) and Criteo. Arms are sorted by mean reward in descending order.

## 3 Experimental design details

In the following section, we provide additional details on our experimental design.

Synthetic environments. Figure 1 shows the reward structure of each synthetic environment. The rewards are defined as follows:

OneSub: $\mu _ { 1 } = 0 . 9 , \mu _ { i } = 0 . 8$ for $i = 2 , \ldots , K$ , so all suboptimal arms share a common mean.

Trap: $\mu _ { 1 } = 0 . 9 , \mu _ { i } = 0 . 8$ for $i = 2 , \ldots , 3 2$ , and $\mu _ { i } = 0 . 1$ for $i = 3 3 , \dots , K$ , creating a small cluster of near-optimal competitors alongside a large set of clearly inferior, high-consumption arms.

Geom: $\mu _ { i } = 0 . 9 \bar { ( 1 / 9 ) } ^ { ( i - 1 ) \bar { / } ( K - 1 ) }$ for $i = 1 , \ldots , K$ , yielding means decaying geometrically from 0.9 to 0.1.

Poly: $\mu _ { 1 } = 0 . 9$ and $\mu _ { i } = 0 . 9 \left( 1 - \sqrt { i / K } \right)$ for $i \geq 2 ,$ , producing polynomial decay with a higher density of near-optimal arms.

Real-world environments. Figure 2 shows the arm reward distribution of each real-world environment.

New Yorker Caption Contest. We derive the first dataset from the New Yorker Caption Contest Jain et al. (2020), a widely used benchmark for preference learning and pure-exploration bandits. Each contest corresponds to a fixed cartoon and a finite set of submitted captions, which we treat as arms. Rewards are binary and reflect whether a caption is judged as somewhat funny or funny by human voters. For each arm, we estimate a ground-truth Bernoulli mean reward using the empirical proportion of positive votes aggregated across all voters, yielding an environment with human-generated noisy rewards and heterogeneous arm quality. We select contests 511, 538, and 780 (denoted c511, c538, c780), which induce three distinct empirical reward distributions and each contain a large number of arms (approximately 3,000–6,000), enabling evaluation with many arms. We use the top 1% quantile arm as baseline, a regret budget of 1.6 times the number of arms in each contest, a maximum arm pull budget of 100,000, and 500 random seeds.

Criteo Attribution Modeling for Bidding. The second dataset is based on the Criteo Attribution Modeling for Bidding dataset Diemert et al. (2017), which consists of large-scale logs from a real-world display advertising system. Each record corresponds to an ad impression, i.e., a single instance in which an advertisement is displayed to a user. For each impression, the dataset records whether the displayed ad led to a conversion attributed to Criteo, the monetary value of that conversion, and the cost incurred by the advertiser to show the ad. The dataset is designed to support attribution and bidding decisions by providing both the revenue and cost components of advertising outcomes.

We aggregate impressions at the campaign level and treat each campaign as an arm. Campaigns with insuficient impression volume are filtered out to ensure stable estimation. For each campaign $i ,$ we denote by $\hat { p } _ { i }$ the success (attribution) probability estimated from historical data and by $v _ { i }$ the mean monetary value per successful attribution. When arm i is pulled at round t, an attribution outcome $A _ { t } ^ { ( i ) } \sim$ Bernoulli(ˆp ) is generated, and the realized monetary reward is $r _ { t } ^ { ( i ) } = v _ { i } A _ { t } ^ { ( i ) }$ . The resulting expected rewards are very low, with a maximum arm mean of 0.015, making this a challenging low-signal setting. Consumption is again defined via regret gaps relative to a baseline arm selected by a quantile of the empirical mean-reward distribution, yielding a natural interpretation of consumption as lost profit relative to a reference campaign. Unlike the caption dataset, this setting captures real economic trade-ofs and monetary regret, providing a complementary evaluation setting.

![](images/f628131308ab7ec06140a5edc258acf7b83427280ceecea085b3451e87a2d2d6.jpg)  
(a) β = 0.25

![](images/e07884969a6317ec15f5d11c57785ac68d0d9067186201355537e28da425c21d.jpg)  
(b) $\beta = 0 . 5$

![](images/257b74bc401e2fd53872a5832f94cd85db62679add8fa49ec42e183c6b5b832c.jpg)  
(c) β = 0.75

![](images/84ab41df693bc0b395eabcb6679ed307c03d1acca51d91bb04f91f0e30e823f0.jpg)  
(d) $\beta = 1 . 0$  
Figure 3: Phase-wise arm quality under a polynomial reward family, for $\beta \in \{ 0 . 2 5 , 0 . 5 , 0 . 7 5 , 1 . 0 \}$

This dataset provides a single real-world reward distribution, which we use to study the impact of diferent baseline choices: we select as baseline the top 5% quantile arm, the top 1% quantile arm, or the best arm, and evaluate performance under each. We sweep the regret budget from 0.06 to 1.8 times the number of arms, with a maximum arm pull budget of 100,000 and 500 random seeds.

## 4 Per-Phase Arm Composition across Polynomial Reward Structures

This section extends the per-phase composition analysis of the main paper to a broader family of polynomial reward structures, varying the decay exponent to test whether the efect of rationing on arm quality persists across reward profiles of difering steepness.

Each figure corresponds to a reward family ${ \mathcal { P } } ( \beta )$ with $\beta \in \{ 0 . 2 5 , 0 . 5 , 0 . 7 5 , 1 . 0 \}$ , where arm rewards follow $\mu _ { i } = 0 . 9 \cdot \left( 1 - ( i / n ) ^ { \beta } \right)$ ; larger $\beta$ yields a gentler decay and therefore a higher density of near-optimal arms. In each figure, the left panel shows the true arm reward distribution sorted in descending order. The right panel shows, for each displayed phase, boxplots of the distribution over 500 seeds of the mean reward among arms surviving to that phase, for four Sequential Halving with Proportional Regret Rationing (SHPRR) variants with $\alpha \in \{ 0 . 0 , 0 . 2 , 0 . 8 , 1 . 0 \}$ . Boxes span the interquartile range and the white bar marks the median. Significance brackets indicate t-tests paired across seeds between each variant and the $\alpha = 0 . 0$ reference: $^ { * } p < 0 . 0 5 , ^ { * * } p < 0 . 0 1 , ^ { * * * } p < 0 . 0 0 1$ . Brackets are shown only for statistically significant diferences; their absence indicates the comparison is not significant at the $p < 0 . 0 5$ level. A star (⋆) marks the variant with the highest mean reward at that phase.

## References

Diemert, E., Meynet, J., Galland, P., and Lefortier, D. (2017). Attribution modeling increases eficiency of bidding in display advertising. In Proceedings of the AdKDD and TargetAd Workshop, KDD, Halifax, NS, Canada. ACM.

Howard, S. R., Ramdas, A., McAulife, J., and Sekhon, J. (2020). Time-uniform chernof bounds via nonnegative supermartingales. Probability Surveys, 17(none):257–317.

Jain, L., Jamieson, K., Mankof, R., Nowak, R., and Sievert, S. (2020). The new yorker cartoon caption contest dataset. https://nextml.github.io/caption-contest-data/.

Li, Z. and Cheung, W. C. (2024). Best arm identification with resource constraints. In International Conference on Artificial Intelligence and Statistics, pages 253–261. PMLR.