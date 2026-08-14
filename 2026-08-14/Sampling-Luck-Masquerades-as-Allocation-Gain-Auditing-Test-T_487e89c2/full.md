# Sampling Luck Masquerades as Allocation Gain: Auditing Test-Time Budget Allocation for Neural Combinatorial Optimization

Jinhyung Bae

Abstract—Neural combinatorial optimization (NCO) solvers report the best of many sampled solutions per instance, and the number of samples is, by convention, identical for every instance. Whether a non-uniform allocation of a fixed total sample budget would buy anything has not been measured. We measure it, and we audit the measurement procedure itself.

Two findings. First, on in-distribution workloads the allocation headroom is not detectable: across three pretrained solvers (POMO, AM, SymNCO) on uniform TSP-100, an oracle allocation computed and evaluated on the same stored samples reports a gain of 2.2–2.6% with confidence intervals excluding zero, but the same gain measured out of sample is indistinguishable from zero (0.457, 0.015 and −0.512 percent, all with intervals covering zero). Following the customary in-sample procedure, all three solvers would have supported a published claim of a 2%-level allocation gain that does not exist. We quantify this bias with an instance-wise null in which the true gain is zero by construction, and show that, over the ranges we test, it does not shrink as the number of stored samples per instance or the number of instances grows: within those ranges it cannot be outrun by collecting more data.

Second, the same correction that removes the phantom gains preserves a real one. Under distribution shift — a workload mixing uniform and clustered instances, on which these checkpoints were not trained — a pre-registered confirmatory experiment finds that allocation guided by held-out sample statistics improves best-of-k by 11.5% (AM, primary endpoint; 95% CI [7.4, 19.7]) and 12.0% (SymNCO, replication) at equal evaluation budget, with the cost of acquiring the guiding signal not charged; a preregistered negative control (POMO, whose failure under shift is an order of magnitude smaller) shows −0.3% [−0.7, 0.24]. The gain exceeds a frozen distribution-label baseline by 4.2 percentage points [1.9, 7.7], so it is not merely a proxy for the label. The ordering of the effect across the three solvers matches the ordering of their distribution-shift failure.

Because the registered endpoint does not charge for the guiding signal, we also report an exploratory budget-accounted policy that is executable at deployment time: a 20-sample probe charged against the same total budget, with the allocation driven by the probe’s coefficient of variation and no reference tour. It retains 3.4% (AM) and 4.6% (SymNCO) at the registered 50:50 composition while the negative control stays at −0.4%, and an exploratory sweep over workload composition suggests it retains substantially more — 11.0% at a 10% out-of-distribution share — with the relationship non-monotone in that share.

We give a correction procedure and a reporting checklist, and we release all cost arrays, analysis code, and the pre-registration record including every amendment and its direction.

Index Terms—Neural combinatorial optimization, test-time compute, resource allocation, selection bias, optimizer’s curse, pre-registration, evaluation methodology.

## I. INTRODUCTION

## A. The convention

Constructive neural solvers for routing problems do not emit one solution. They emit many and keep the best. The Attention Model [1] samples 1,280 tours per instance; POMO [2] rolls out one greedy trajectory per starting node and multiplies by eight dihedral augmentations, giving 800 deterministic trajectories for a 100-node instance. In both cases the count is a single global hyperparameter. Every instance in the benchmark receives the same number of attempts.

This is a resource allocation decision made by default rather than by analysis. Instances are not equally responsive to additional samples: for some the best-of-k curve flattens after a handful of rollouts, for others it keeps descending. If the marginal value of a sample differs across instances, then under a fixed total budget the uniform allocation is leaving something on the table. How much is an empirical question that, to our knowledge, has not been answered.

## B. Two questions

Q1 (value). Under a fixed total sample budget, how much does instance-wise allocation buy relative to the uniform convention?

Q2 (measurement). The natural way to answer Q1 is to collect samples, estimate each instance’s best-of-k curve, compute the optimal allocation, and report the improvement. Decision and evaluation then use the same samples. Is the resulting number trustworthy?

Q2 is not a technicality. The allocation step is an explicit maximization over instances, and maximization over noisy estimates is exactly the setting in which selection bias is known to be severe — the optimizer’s curse in decision analysis [3], inference on winners in econometrics [4], datasnooping in forecast evaluation [5], [6]. The contribution of this paper is not to discover that such bias exists. It is to quantify it in the NCO test-time allocation setting, where it has not been measured, to give a domain-specific correction, and to demonstrate that the correction discriminates rather than merely deflates.

## C. Contributions

1. To our knowledge, the first measurement of allocation value for NCO test-time sampling. On indistribution workloads the value is not detectable. Under

J. Bae is with the Department of Industrial and Management Engineering and the AI Data Convergence program, Hankuk University of Foreign Studies, Yongin, Republic of Korea (e-mail: kevinbae2006@hufs.ac.kr).

distribution shift it is 11–12% at equal evaluation budget with the signal-acquisition cost not charged, and 3–5% at that same 50:50 composition when a probe is charged against the total budget (rising to 11.0% at a 10% shifted share, exploratory); its magnitude across three solvers orders with the magnitude of each solver’s failure under shift. The distribution-shift result is a pre-registered confirmatory experiment with a declared primary endpoint, a replication arm, and a negative control.

2. To our knowledge, the first quantification of insample selection bias in this setting. With an instancewise null construction in which the true allocation gain is zero, the customary in-sample procedure manufactures gains of the same order as the effects it is used to detect. The bias is approximately invariant to the number of stored samples per instance and to the number of instances, over the ranges we test.

3. A correction and a two-sided demonstration. We report the gain out of sample and calibrate the insample estimate against the instance-wise null. The same procedure removes the in-distribution phantom gains and leaves the distribution-shift gain intact.

4. A full pre-registration record. Gate criteria, endpoints and amendments were fixed before the corresponding results were seen; every amendment is reported together with the direction in which it moves the verdict, including amendments that work against our own narrative.

## D. What this paper does not claim

We do not claim a new statistical phenomenon; the mechanism is classical selection bias, applied to a setting where it has not been accounted for. We do not claim that uniform allocation is optimal in distribution — only that no gain is detectable at our measurement precision. We do not claim that the test-time compute literature uniformly fails to correct for this: Snell et al. [7] use two-fold cross-validation for strategy selection, and we discuss the landscape in Section VI. We do not claim a dose–response relationship between distributionshift magnitude and allocation gain; we have three solvers and report an ordering.

## II. PROBLEM SETUP

## A. Formulation

Let $\pounds \_ { \mathrm { ~ \ - ~ i ~ } } ( \ k )$ be the expected cost of the best of k solutions drawn for instance i. For a workload of N instances and total budget S:

$$
\begin{array} { r l } { \mathrm { m i n i m i z e } } & { \sum _ { i } f _ { i } ( k _ { i } ) } \\ { \mathrm { s u b j e c t ~ t o } } & { \sum _ { i } k _ { i } = S , k _ { i } \geq 1 , k _ { i } \in \mathbb { Z } } \end{array}
$$

f\_i is non-increasing and convex in k, being the expectation of a minimum order statistic: the hundredth sample cannot help as much as the tenth. The problem is therefore a convex separable resource allocation, for which greedy marginal allocation — repeatedly giving the next sample to the instance with the largest $\mathrm { f \_ i ( k \_ i ) \ ~ - ~ \ f \_ i ( k \_ i + 1 ) }$ — is optimal.

The consequence matters for how this paper is organized: if f were known, the optimization would be solved. The difficulty is estimating f cheaply, and the difficulty of the difficulty is knowing whether the estimate can be trusted.

## B. The unit of allocation

Three quantities can be increased at inference time, and an allocation study is ill-posed until one is designated as the unit.

Axis A — stochastic decoding rollouts. Temperature sampling produces $\mathrm { k \_ i }$ solutions. Multi-start and augmentation are disabled. This is the only axis with no upper bound: multistart is capped by the number of nodes, and augmentation is a fixed set of eight symmetries. It is also the canonical inference mode for the Attention Model.

Axis $\textbf { B } -$ the finite pool of the standard protocol. POMO’s standard inference is n multi-start greedy rollouts times eight augmentations — for TSP-100, a deterministic pool of 800 trajectories. Randomizing the pool order and taking the first $\mathrm { k \_ i }$ without replacement makes the draws exchangeable, so E[min] remains non-increasing and convex and the greedy argument survives with the box constraint $\mathrm { k \_ i }$ $\leq \mathrm { ~ \mathsf ~ { ~ l ~ p ~ o ~ o ~ l ~ } ~ } _ { - ^ { \frac { \mathrm { ~ i ~ } } { 1 } } } |$

Both axes are reported. Axis B answers the objection that a study conducted purely in sampling mode measures a weakened inference procedure; Axis A answers the objection that a bounded pool cannot express arbitrary allocations. Where the two disagree, we say so.

## C. The allocating agent is a single solver

The budget being divided belongs to one fixed solver. If different instances were served by different checkpoints, $\Sigma _ { - } -$ $\mathrm { ~ i ~ } \ \mathrm { f \_ i \ } ( \mathrm { k \_ i } )$ would no longer describe allocation across a workload but selection among solvers, which is a different problem with its own literature [8]. We therefore hold the checkpoint fixed across the workload.

The workload, by contrast, is unconstrained. From the perspective of a deployed solver, an out-of-distribution instance is simply a hard instance, and its difficulty is a legitimate target for allocation regardless of whether that difficulty originates in the instance geometry or in the solver’s training distribution. We therefore describe our findings in terms of workload heterogeneity as seen by a fixed solver, not in terms of intrinsic instance hardness.

## D. Offline replay

For every instance we store the costs of all K sampled solutions. Any allocation policy can then be evaluated by taking, for each instance, the minimum of the first $\mathrm { k \_ i }$ entries of a randomized ordering of its stored array. No further inference is required.

This is not a workaround for limited compute. It forces every policy — uniform, oracle, label-based, out-of-sample — to be compared on literally the same samples, which removes run-to-run noise from all policy comparisons. It is also what makes the audit in Section III affordable: the null distributions are computed by resampling the stored arrays.

Costs are normalized to gap-to-reference, $\begin{array} { r l } { { 9 } \mathrm { a p \mathrm { - } \mathrm { i } \left( k \right) } } & { { } \mathrm { = } } \end{array}$ $\mathrm { 1 0 0 ~ \cdot ~ \ : \ : ( c o s t \_ i ~ ( k ) ~ \ : ~ / ~ \ : ~ r e f \_ i ~ \ : ~ - ~ \ : 1 ~ ) ~ }$ , with $\tt r e f \_ i$ an LKH-3 tour. A deterministic denominator is essential here: a denominator estimated from the same samples (for example the expected cost at $\mathrm { ~ k ~ } = \mathrm { ~ 1 ) }$ would be correlated with the dispersion that drives the allocation effect, inflating the measured gain as a normalization artifact.

## E. Estimands

$\mathrm { d ~ } = \mathrm { ~ \Gamma ~ } ( \mathrm { u n i f { o r m } ~ - ~ \Gamma { o r a c l e } } ) \quad /$ uniform, the relative improvement in mean gap. We report it through two estimators:

• d\_in — allocation decided and evaluated on the same stored array.

• d\_split — the stored array is split in half; the allocation is decided from one half and evaluated on the other. In expectation $\mathtt { d \_ i n }$ is biased upward, because the alloca  
tion exploits the realized noise of the array it is scored on,   
and $\mathsf { d } _ { - } s \mathsf { p } \bot \mathrm { i t }$ is biased downward, because the allocation is   
decided from half as much data as is available. These are bias   
directions in expectation, not per-realization guarantees,   
and we do not present $[ { \mathsf { d } } _ { - } { \mathsf { s p l i t } } , \ { \mathsf { d } } _ { - } { \mathrm { i n } } ]$ as a hard interval.

d\_split is an out-of-sample estimate of allocation value. It is not a budget-accounted policy: the samples used to decide the allocation are not charged against S. See Section VII (Limitation 4).

## III. THE AUDIT

## A. Regularizing the oracle

The greedy rule requires non-increasing marginal gains. Estimated curves violate this — in synthetic checks only 58% of consecutive estimated marginals were non-increasing — and the violation is not innocuous: sorting all marginals and taking the top $\mathrm { ~  ~ { ~ S ~ } ~ } - \mathrm { ~  ~ { ~ N ~ } ~ }$ can then select the j-th marginal of an instance without its (j−1)-th, so the resulting counts no longer correspond to a prefix and the allocation is not a feasible policy. We therefore take the greatest convex minorant of each estimated curve before differencing. The true $\pounds \_ { \mathrm { i } }$ is convex, so this is regularization toward a known property, not a modelling choice.

## B. The instance-wise null

To calibrate d\_in we need data on which the true allocation gain is zero. We construct it per instance: resample with replacement from a single instance’s stored array to create N synthetic instances that share that instance’s marginal cost distribution but are exchangeable, so no allocation can help in expectation. The synthetic set is passed through the identical pipeline, including convex regularization. Repeating this for many source instances yields a distribution of per-source 95th percentiles.

Definition of the floor. We report the median of the persource $\mathsf { p } 9 5$ values as the primary floor and the p90 as a conservative variant. We do not use the maximum. A sweep over the number of source instances shows the maximum diverges: for SymNCO on Axis A it grows from 20.1 (8 sources) to 27.7 (12), 43.9 (25) and 90.4 (50), because the per-source distribution is heavy-tailed. A statistic whose value depends on how many sources one happens to examine cannot serve as a decision threshold. The median and p90 are stable under the same sweep.

![](images/4123f9e34d7eb7fc4d1db9f54ba0ca27abd4f3d44a5738327aa4db76e45ef37d.jpg)  
Fig. 1. In-distribution audit (uniform TSP-100, Axis B, $N { = } 5 0 , S / N { = } 1 0 0 )$ Solid bars: in-sample $d _ { \mathrm { i n } } ,$ all three 95% CIs excluding zero. Hatched bars: the same gain out of sample $( d _ { \mathrm { s p l i t } } ) ;$ every interval covers zero. Grey bands: the instance-wise noise floor, median to p90. Read in-sample, each solver supports a 2%-level published gain; read out of sample, none does.

TABLE I  
IN-DISTRIBUTION AUDIT: UNIFORM TSP-100, AXIS B, N=50, S/N=100.
<table><tr><td>Solver</td><td> $d _ { \mathrm { i n } }$  [95% CI]</td><td> $d _ { \mathrm { s p l i t } }$  [95% CI]</td><td>floor (med/p90)</td></tr><tr><td>POMO</td><td>2.227 [1.63, 2.82]</td><td>0.457 [-0.44, 1.34]</td><td>2.083 / 4.211</td></tr><tr><td>AM</td><td>2.567 [1.96, 3.10]</td><td>0.015 [-1.08, 1.06]</td><td>3.440 / 5.488</td></tr><tr><td>SymNCO</td><td>2.206 [1.61, 2.72]</td><td>-0.512 [-1.79, 0.38]</td><td>3.553 / 5.793</td></tr></table>

An earlier version of this work used a pooled null — pooling all instances’ costs and resampling from the pool — which we discarded: under a size- or distribution-mixed workload the pool has greater dispersion than any real instance, so the resulting floor is inflated. Numbers computed under the pooled null are not reported anywhere in this paper.

## C. In-distribution result

Homogeneous workload (50 uniform TSP-100 instances), Axis B (the standard protocol), $\mathrm { S / N ~ = ~ 1 0 0 , K ~ = ~ 8 0 0 ~ p o o l }$ 95% intervals from instance bootstrap (Table I):

Read the first numeric column alone and every row is a finding: a 2%-level allocation gain with an interval excluding zero, on three independent pretrained solvers. Read the second column and there is nothing: all three out-of-sample estimates cover zero, one of them negative and one indistinguishable from zero at the third decimal, which is what convexity predicts when instances are exchangeable and an allocation is driven by noise.

Against the floor, AM and SymNCO sit below both the median and the p90 variant. POMO’s $\mathsf { d \_ i n }$ of 2.227 sits marginally above its median floor of 2.083 and below its $\mathsf { p } 9 0$ floor of 4.211. We report all three quantities rather than choosing the definition that makes the row cleanest; the discriminating evidence in this paper is the out-of-sample column, which is unaffected by the floor definition.

A synthetic check with true $\mathrm { ~ d ~ } = \mathrm { ~ 0 ~ }$ by construction produces in-sample gains of 1.8–3.6%, the same order as the table above and as the 2% threshold we had pre-registered as a “proceed” criterion.

TABLE II  
NOISE FLOOR (MEDIAN OF PER-SOURCE P95) AGAINST STORED SAMPLES PER INSTANCE K. AXIS B, HOMOGENEOUS WORKLOAD, $S / N { = } 1 0 0 , N { = } 5 0$
<table><tr><td>Solver</td><td> $\mathrm { K } = 4 0 0$ </td><td> $\mathrm { K } = 6 0 0$ </td><td> $\mathbf { K } = 8 0 0$ </td></tr><tr><td>POMO</td><td>1.705</td><td>1.979</td><td>1.840</td></tr><tr><td>AM</td><td>3.157</td><td>3.168</td><td>3.080</td></tr><tr><td>SymNCO</td><td>4.182</td><td>4.173</td><td>3.747</td></tr></table>

TABLE III

NOISE FLOOR (MEDIAN OF PER-SOURCE P95) AGAINST THE NUMBER OF INSTANCES N. AXIS B, HOMOGENEOUS WORKLOAD, K=800, S/N=100.
<table><tr><td>Solver</td><td> $\Nu = 1 0$ </td><td> $\Nu = 2 0$ </td><td> $\mathrm { N } = 3 0$ </td><td> $\mathrm { N } = 5 0$ </td></tr><tr><td>POMO</td><td>2.072</td><td>1.938</td><td>1.971</td><td>1.840</td></tr><tr><td>AM</td><td>3.049</td><td>3.096</td><td>3.436</td><td>3.080</td></tr><tr><td>SymNCO</td><td>3.810</td><td>4.355</td><td>4.307</td><td>3.747</td></tr></table>

## D. Operating characteristics of the floor (Fig. 2)

The natural response to a noise floor is to collect more data. It does not work.

(a) More samples per instance. Axis B, homogeneous, $\mathrm { \Delta S / N ~ } = \mathrm { \Delta } 1 0 0 , \mathrm { \textnormal { N } = \mathrm { \Delta } 5 0 } .$ , floor (median), Table II:

(b) More instances. Axis B, homogeneous, $\begin{array} { r } {  { \mathrm { ~  ~ K ~ } } = \phantom { - } 8 0 0 ,  { \mathrm { { S } / N } } } \end{array}$ = 100, floor (median), Table III:

Both are flat within Monte-Carlo error. The floor does not come down with more samples per instance, and it does not come down with more instances: over the ranges we test it does not come down with scale, and correction is the only exit we have found. That is the caption of Fig. 2 and the operational content of Section V.

Two consequences follow. Reporting a bigger experiment does not make an in-sample allocation gain more trustworthy. And a practitioner cannot infer the floor from ours — it varies by a factor of two across solvers here (1.8 to 3.7 at K = 800) — so it must be computed for the solver and configuration at hand.

Note on an earlier claim. A previous analysis of ours reported that the floor grows with N. That result was produced under the discarded maximum-based definition, in which the statistic grows mechanically with the number of sources examined, and the comparison additionally confounded N with the number of source instances. Under the corrected definition the effect is absent. We record the retraction because it removes a result that would have supported an appealing extreme-value narrative, and because it is the second of two occasions on which our own re-measurement destroyed one of our findings (the first being the pooled null of Section III-B).

## IV. WHEN ALLOCATION IS REAL

## A. The workload

The audit above uses an in-distribution workload, which is the condition least favourable to allocation: instances drawn i.i.d. from the training distribution are close to exchangeable, and under an exchangeable workload convexity favours uniform allocation. The interesting question is what happens when the deployed workload departs from the training distribution — the ordinary condition in deployment.

We construct a mixed workload of 50 instances: 25 uniform TSP-100 and 25 clustered TSP-100 (four Gaussian clusters, $\sigma = 0 . 0 6 )$ , served by the same single checkpoint. Clustered instances are out of distribution for all three checkpoints, and the severity differs sharply between them.

## B. Pre-registered confirmatory experiment

An exploratory pass over three solvers and two axes produced 12 cells, of which two were positive. Rather than report those, we pre-registered a confirmatory experiment: new instance seed and new decoding seed, everything else frozen $( \mathrm { N } = 1 0 0$ instances, $\mathrm {  ~ K ~ } = ~ 1 , 0 0 0 , ~ \mathrm {  ~ S ~ } / \mathrm {  ~ N ~ } = ~ 1 0 0$ set composition, analysis code, the label baseline ratios), with endpoints declared in advance and no tests beyond them.

The primary arm is AM on Axis A because sampling is AM’s canonical inference mode. SymNCO’s canonical mode is multi-start, so its Axis A arm is labelled a mechanism replication rather than a replication of practice.

The 2% threshold is the same one used as the “proceed” criterion throughout the project; we did not lower the bar for the confirmatory run.

Statement of the headline number. Throughout, the 11– 12% figure means: allocation guided by held-out sample statistics improves best-of-k by 11–12% at equal evaluation budget, with the signal-acquisition cost not charged (Section VII, Limitation 4; budget-accounted variant in Section IV-F). We do not state the number without that qualification.

## C. The gain tracks the failure

AM and SymNCO are excellent in distribution and collapse under shift; POMO is mediocre everywhere and degrades mildly. The ordering of allocation gain matches the ordering of shift failure across the three solvers, and the negative control is the informative cell: on the solver that is robust to shift, the gain vanishes (Table V).

We state this as an ordering across three solvers. With three points and a gap between 2.4× and 27.4×, we do not characterize the relationship as a dose–response.

## D. The signal is more than the label

If the gain were simply “spend more on clustered instances”, it would be captured by a rule that reads the distribution label and allocates in a fixed ratio. We freeze such a baseline from the exploratory pass, refit nothing on the confirmatory data, and evaluate it on the same held-out half with the same budget, distributing rounding remainders so that the label policy spends the budget exactly. The pre-registered ratios are AM 20:1 and SymNCO 12:1, the two arms in which the residual is an endpoint; the ratio applied to the negative control (POMO 1.5:1) was taken from the same exploratory pass but was not part of the registered freeze, and no endpoint depends on it. The residual — greedy minus label, both out of sample — is 4.204 points [1.862, 7.702] for AM.

![](images/12bf6b6b73d9781bc9941d83999109159d27d2979b9b5d44c5b4b86c227b22a0.jpg)

![](images/af09925ed5c521b281cbb9794016be3795b6006cf613c069b4c02db2b1965bd2.jpg)  
Fig. 2. Operating characteristics of the floor (median definition). (a) Flat in the number of stored samples per instance K at fixed S/N; (b) flat in the number of instances N. Over the ranges we test, the floor does not decrease with scale; correction is the only exit we have found.

TABLE IV  
PRE-REGISTERED CONFIRMATORY EXPERIMENT: NEW INSTANCE AND DECODING SEEDS, ALL ELSE FROZEN, NO TESTS BEYOND THE REGISTERED ENDPOINTS.
<table><tr><td>Role</td><td>Solver, axis, set</td><td>dsplit [95% CI]</td><td>Endpoint</td><td>Result</td></tr><tr><td>Primary</td><td>AM, Axis A, mixed</td><td>11.549 [7.401, 19.734]</td><td>CI lower ≥ 2%</td><td>pass</td></tr><tr><td>Secondary</td><td>AM, residual over label baseline</td><td>4.204 [1.862, 7.702]</td><td>&gt; 0</td><td>pass</td></tr><tr><td>Replication</td><td>SymNCO, Axis A, mixed</td><td>12.017 [5.183, 19.982]</td><td>CI lower  $\geq 2 \%$ </td><td>pass</td></tr><tr><td>Negative control</td><td>POMO, Axis A, mixed</td><td>-0.290 [−0.720, 0.238]</td><td>interval covers 0</td><td>as predicted</td></tr></table>

Fig 3 — The same correction, applied to both workloads  
![](images/a886d85f7c351e988b09b4c55e6ca1459e5823ab74af426b449814701d44f902.jpg)  
Fig. 3. The same correction, applied to both workloads. Left (in-distribution): in-sample estimates (open markers) sit at the 2% level while out-of-sampl estimates (filled) collapse to zero inside the noise floor. Right (distribution-shifted): the out-of-sample gain survives at 11–12% for the two shift-fragile solvers and vanishes for the shift-robust negative control (POMO, ×). One procedure removes the phantom and keeps the real effect.

So roughly two thirds of the AM gain is available from the label alone, and a third requires per-instance information. Both halves of that sentence matter: the trivial rule is strong, and it is not sufficient.

## E. Allocation does not require detecting the shift

The obvious alternative response to distribution shift is to fix the model — fine-tune on clustered data, or select a matched checkpoint. We do not argue against it, and the two responses are not exclusive. But they differ on two axes.

Cost. Retraining consumes GPU time, engineering time, and labelled or generated data for the new regime. Allocation redistributes an inference budget that is already being spent; its marginal cost is zero.

Prerequisite knowledge. Retraining must be triggered, which requires first recognizing that the workload has shifted and in what way. The allocator does not: it consumes only the solver’s own sample statistics on the instance at hand, never a distribution label. This asymmetry, rather than the cost difference, is the substantive argument — allocation is available in the interval between a shift occurring and its being detected. It is, however, not free: the sample statistics must themselves be bought, and Section IV-F prices them.

TABLE V  
DISTRIBUTION-SHIFT SEVERITY PER SOLVER (MEAN GAP ON UNIFORM VS. CLUSTERED TSP-100) AND THE CORRESPONDING OUT-OF-SAMPLE ALLOCATION GAIN.
<table><tr><td>Solver</td><td>uniform gap</td><td>clustered gap</td><td>ratio</td><td>d_split</td></tr><tr><td>AM</td><td>1.11%</td><td>38.63%</td><td>34.8×</td><td>11.549</td></tr><tr><td>SymNCO</td><td>0.87%</td><td>23.79%</td><td>27.4×</td><td>12.017</td></tr><tr><td>POMO</td><td>7.16%</td><td>17.36%</td><td>2.4×</td><td>-0.290</td></tr></table>

TABLE VI

BUDGET-CHARGED PROBE POLICY (EXPLORATORY; NO TEST REPORTED) AGAINST THE UNCHARGED OUT-OF-SAMPLE ESTIMATOR.
<table><tr><td>Solver</td><td>d_charged (exploratory) uncharged d_split</td><td></td></tr><tr><td>AM</td><td>3.394 [0.609, 7.564]</td><td>11.549</td></tr><tr><td>SymNCO</td><td>4.588 [1.623, 10.541]</td><td>12.017</td></tr><tr><td>POMO (negative control)</td><td>-0.391[-1.002, 0.126]</td><td>-0.290</td></tr></table>

## F. Charging for the signal (exploratory)

The endpoints above do not charge for the samples used to decide the allocation. To price them we run a descriptive variant, outside the pre-registration and with no test reported, in which a fixed total budget S = 100N is spent by a singlepass deployable policy: draw a probe of m = 20 rollouts per instance, charged against the budget and retained as candidate solutions, allocate the remaining S − mN in proportion to each probe’s coefficient of variation, and report the best of each instance’s k\_i samples. The signal uses no reference tour, so the policy is executable at deployment time (Table VI).

Probe size sensitivity for AM: 3.36 (m = 5), 3.76 (m = 10), 3.39 (m = 20), 2.81 (m = 40).

Three observations. The effect survives budget accounting but shrinks by roughly a factor of three. The negative control continues to hold, so the discrimination between shift-fragile and shift-robust solvers is not an artefact of the free signal. And the gap between 3–5% charged and 11–12% uncharged is itself a result: it bounds how much a better signal than a probe’s coefficient of variation could recover, and it is the natural target for the follow-up that Section VII leaves open.

## G. Gain against out-of-distribution share (exploratory)

Subsampling the stored arrays to vary the clustered fraction, holding N = 50 and $\mathrm { ~ S ~ } / \mathrm { ~ N ~ } = \mathrm { ~ 1 0 0 ~ }$ and averaging over 12 resampled workloads. Both columns are AM on Axis A: the pair is the same solver under two policies — the signal-free estimator of Section II-E and the budget-charged policy of Section IV-F — not two solvers. Descriptive only; no test is reported (Table VII).

Fig 4 (exploratory, descriptive — no test)  
![](images/061d10748e7ccc18eb79dae818ebccc9f657a66b4060f0c8662e186c82cee308.jpg)  
Fig. 4. Exploratory composition sweep (descriptive; no test): allocation gain against the out-of-distribution share, signal-free estimator vs. budgetcharged probe policy (AM, Axis A; N=50 subsampled workloads, 12 draws). Dependence is real but not proportional; the registered 50:50 composition is not the most favourable one.

TABLE VII  
ALLOCATION GAIN AGAINST THE OUT-OF-DISTRIBUTION SHARE OF THE WORKLOAD (EXPLORATORY; DESCRIPTIVE, NO TEST).
<table><tr><td>OOD share</td><td>d_split (signal free)</td><td>d_charged (probe m = 20 charged)</td></tr><tr><td>0%</td><td>1.586</td><td>-3.018</td></tr><tr><td>10%</td><td>11.740</td><td>11.001</td></tr><tr><td>25%</td><td>18.161</td><td>11.201</td></tr><tr><td>50%</td><td>12.760</td><td>3.052</td></tr></table>

The 50% row is not identical to the corresponding numbers in Section IV-B and Section IV-F (11.549 and 3.394) because the sweep is built from 12 subsampled workloads of $\mathrm { ~ \tt ~ N ~ } = \mathrm { ~ \tt ~ 5 0 ~ }$ drawn from the confirmatory pool, whereas those sections use the full N = 100 collection once. The two are consistent, not the same estimate.

Dependence on composition is real but it is not proportional. The sweep suggests an interior peak: both curves rise steeply away from a fully in-distribution workload and fall back by 50%. We state this and no more; the sweep is descriptive and carries no test, so we do not estimate the location of the peak or offer a mechanism for it. What does follow is negative: the registered 50:50 composition is not the composition most favourable to allocation, so reading our headline number as an upper bound obtained by stacking the workload with hard instances would be incorrect.

Two further readings. At a fully in-distribution workload the charged policy is negative (−3.0): it spends a probe budget on a discrimination that does not exist, which is the audit result of Section III restated as a deployment cost. And the penalty for charging the probe is small at low OOD share (11.0 versus 11.7 at 10%) and large at 50% (3.1 versus 12.8) — at realistic deployment shares, the executable policy retains most of the headroom.

## V. PRESCRIPTIONS

Stated as a checklist rather than a conclusion.

1. Report allocation gains out of sample. Split the stored samples, decide the allocation on one part, evaluate on the other. This is the single change that separates the phantom gains of Section III-C from the real gain of Section IV-B.

2. If an in-sample number must be reported, calibrate it. Build the instance-wise null at your own (N, K, S/N) and report the in-sample estimate against that floor. Do not borrow ours: it differs by a factor of two across three solvers on the same problem.

3. Do not expect scale to fix it. The floor is flat in both the number of stored samples per instance and the number of instances over the ranges we test. More data does not buy your way out; correction is the only exit. A larger experiment reported in-sample is not a more trustworthy one.

4. Keep the budget shallow relative to the stored depth. Estimating f at budgets approaching the stored array size leaves too few effective independent blocks; in synthetic checks, holding S/N proportional to K left the bias unchanged as K grew, while holding S/N well below K shrank it. We use $\mathrm { ~ S ~ } / \mathrm { ~ N ~ } \le \ \mathrm { ~ K ~ } / 4$ as a hard assertion in code.

5. On the allocation question itself. For in-distribution homogeneous workloads, we find no detectable headroom: uniform allocation is an adequate default within our detection limit. For shifted workloads served by a fixed checkpoint, allocation is worth measuring, it operates without a distribution label, and it retains 3–5% at a 50:50 composition — and, exploratorily, 11.0% at a 10% shifted share — even when the guiding probe is charged against the same total budget.

## VI. RELATED WORK

Conceptual ancestry. Selecting the maximum of noisy estimates and then reporting that maximum’s estimated value is the optimizer’s curse [3] and the winner’s curse in postselection inference [4]. Our d\_in is that construction applied to a budget allocation over instances. We cite this literature as background rather than as a quantitative prediction for our setting: the natural extreme-value scaling of a maximum over N does not describe our floor, which is an average over N instances and is flat in N (Section III-D).

Corrective tooling. White’s Reality Check [5], Hansen’s Superior Predictive Ability test [6], and the Deflated Sharpe Ratio [9] calibrate an apparently strong selected result against the distribution of results obtainable by selection alone; SIREN [10] applies the same logic to LLM benchmark reporting. The instance-wise null of Section III-B is that family’s allocation counterpart, specialized to the fact that the noise here comes from estimating minimum order statistics from a finite stored array.

Allocation of test-time compute. Damani et al. [11] allocate best-of-k per query using a learned reward-distribution predictor; Snell et al. [7] show that difficulty-aware allocation of test-time compute is markedly more efficient than uniform; Brown et al. [12] measure per-problem coverage curves and their heterogeneity. These establish the mechanism in the LLM setting. Our contribution is not the mechanism but the audit, and a test of whether the mechanism survives in a domain whose objective is a continuous minimum order statistic rather than a verifier-checked Bernoulli success.

On the measurement question specifically: many test-time compute studies report adaptive-allocation gains without stating that the samples used to decide the allocation are disjoint from those used to evaluate it. We do not audit any individual study here — verifying such a separation requires the released implementation, which we did not attempt — and we note one published exception: Snell et al. [7] use two-fold crossvalidation for strategy selection, though difficulty binning still uses oracle information. In the NCO allocation setting, no such correction is standard.

Neural combinatorial optimization. POMO [2], the Attention Model [1] and Sym-NCO [13] supply the three checkpoints. Gao et al. [8] select among neural solvers per instance and leave runtime-aware selection as an open problem, adjacent to the question Q1 addresses. Critiques of NCO evaluation practice [14] motivate the reporting checklist. Recht et al. [15] provide a related example of auditing conclusions drawn from heavily reused evaluation benchmarks.

## VII. LIMITATIONS

1. Three points on the shift axis. Shift severity is 2.4×, 27.4× and 34.8×, with nothing between 2.4 and 27.4. We claim an ordering across three solvers, not a functional relationship.

2. Workload composition is a design choice, and not the most favourable one. The registered workload is 50% clustered, so the result should be read as “11– 12% at this composition”. We initially expected the gain to fall roughly in proportion to the shifted share; the exploratory sweep of Section IV-G contradicts that expectation, and we state the correction rather than quietly dropping it. Composition dependence is real but not proportional, and the sweep suggests an interior peak; the budget-charged policy is more favourable at low shares (11.0 at 10% versus 3.1 at 50%). We do not estimate where the peak lies or why. Characterizing the dependence properly requires the confirmatory treatment that Section IV-G does not have, and until then no claim in this paper rests on it.

3. A single problem class. All results are TSP-100. CVRP was gated out before collection: the reference solver’s seed-to-seed spread on CVRP-100 was 0.80% at a 5- second limit and 0.56% at 15 seconds, against a preregistered stability requirement of 0.05% — the denominator would have been as unstable as the effect under study. Three solvers with different decoding distributions and different absolute gap levels (1.0–2.3% in distribution) substitute only partially for problem diversity. A registered fallback exists for a future revision: a fixed-seed reference is deterministic and therefore uncorrelated with the sample arrays, and since all our claims are relative comparisons between policies on a common denominator, its inexactness cancels; it would only forfeit comparability of absolute gaps to other papers.

4. The registered endpoint is not budget-accounted. d\_split decides the allocation using half of each instance’s stored array — 500 samples, five times the evaluation budget of 100 — and evaluates on the other half. Those samples are not charged. d\_split therefore estimates the headroom available to allocation, not the performance of a realizable fixed-budget policy, and we do not describe it as a policy anywhere in the text. The exploratory variant of Section IV-F prices the signal and retains 3–5%; that variant is descriptive, was not pre-registered, and no test is reported on it. Closing this gap properly — designing a signal that recovers more of the 11–12% headroom at a charged probe cost — is left to follow-up work.

5. Transparency of amendments. Every change, with its date and the direction in which it moves the decision criteria, is tabulated in Appendix B; we do not attempt a net tally here, because the changes are not commensurable. Two illustrate the range. Replacing the pooled null with the instance-wise null (Amendment 1) lowered the floor and so favoured detection, and we made it anyway on grounds that hold regardless of outcome — a pooled null is invalid under a mixed workload because it fabricates synthetic instances more dispersed than any real one. Replacing the maximum-based floor with the median, made after the gate, works against this paper’s own audit narrative: it moves POMO’s in-distribution row to the margin of its floor, and — together with separating the number of null source instances from N — it retracted our earlier claim that the floor grows with N (Section III-D). We adopted it because the definition is correct, not because it helped.

The decision rules themselves were never changed retroactively. One known defect is therefore recorded rather than repaired: the termination rule was keyed on d\_in against the floor, so it could not credit the positive out-of-sample evidence that d\_split supplies, and it returned “signal absence” in all four gate cells (Appendix B, v8). We report that verdict as registered and the d\_split evidence alongside it.

## VIII. CONCLUSION

Instance-wise allocation of a test-time sample budget buys nothing measurable for neural combinatorial optimization solvers on the workloads those solvers were trained for. On a workload half of which lies outside that distribution it buys 11–12% at equal evaluation budget with the guiding signal free, and an exploratory policy that charges a 20- sample probe against the same budget retains 3.4% at that composition and 11.0% at a more realistic 10% shifted share — in every case without needing to know that the shift occurred. The procedure conventionally used to measure such gains — deciding and evaluating the allocation on the same stored samples — manufactures gains of 2.2–2.6% on data where the true gain is zero, an amount that, over the ranges we test, diminishes neither with more samples per instance nor with more instances. Reporting the gain out of sample removes the phantom and leaves the real effect standing.

## APPENDIX A REPRODUCIBILITY

Checkpoints. ML4TSPBench/{POMO,AM,SYMNCO} tsp100\_pomo.pt, tsp100\_am.pt, tsp100\_- symnco.pt. These are legacy-format rl4co state dicts and require key remapping to load into current rl4co: feed-forward blocks module.0/2.{weight,bias} → module.lins.0/1.{weight,bias}; decoder logit\_attention.<sub>\*</sub> → pointer.<sub>\*</sub>; the AM archive additionally contains a rollout baseline copy under baseline.<sub>\*</sub>, which must be excluded. Encoder depth is 3 (not the library default) and normalization is batch; both are inferred from the checkpoint. Loading asserts zero missing and zero unexpected keys — a partial load would silently mix randomly initialized layers into the measured curves.

Verification against LKH-3 on 8 held-out instances: POMO multi-start greedy gap 2.27%, SymNCO 1.21%, AM sampling(512) 1.01%.

Collection. N = 100 instances per run (50 uniform, 50 clustered), 100 nodes. Axis A: K = 1,000 stochastic rollouts per instance. Axis B: the full n × 8 = 800 deterministic pool. Evaluation budget S/N = 100 with assert S/N × 4 ≤ K and a per-instance cap k\_i ≤ K/2. References from LKH-3 (elkai). Arrays are stored per solver and per axis with SHA-256 digests; the confirmatory run replicates the collection code with only the instance and decoding seeds changed.

Seeds. Instance generation, decoding, curve ordering and bootstrap are separated and reported. Exploratory run: instance 20260812, decoding 11. Confirmatory run: instance 20260813, decoding 77. Ordering (22) and bootstrap (33) frozen across both.

Reproduction check. Re-collecting POMO with identical seeds reproduced the stored arrays bit-for-bit (fd5e6a6adb646854, 9b6146156f2fa124).

Artifacts. Every number in this paper is traceable to one of six files: expand\_report.txt (three solvers, in-distribution audit), confirm\_report.txt (preregistered confirmatory endpoints), floor\_definition\_sweep.json (per-source null percentiles, the median/p90/max comparison of Section III-B, and the floor values of Section III-C), fig2\_floor\_median.json (floor sweeps over K and N, median and p90), charged\_- variant.json (budget-charged policy and its probesize sensitivity), fig4\_ood\_share.json (composition sweep). Figures: fig1\_audit\_headline.png, fig2\_floor\_median.png, fig3\_two\_sided.png, fig4\_ood\_share.png. Raw cost arrays: {expand, confirm}\_{POMO,AM,SYMNCO}\_axis{A,B}.npz.

Analysis scripts are released with the arrays; the audit, the confirmatory analysis and the two exploratory variants are separate scripts, so the pre-registered path can be re-run without the exploratory additions. All arrays, scripts, and the pre-registration record are available at https://github.com/nepersoned/best-of-k-allocation.

TABLE VIII  
PRE-REGISTRATION TIMELINE. EVERY GATE, ENDPOINT, AND AMENDMENT WAS FIXED BEFORE THE CORRESPONDING RESULT WAS SEEN; EACH AMENDMENT RECORDS THE DIRECTION IN WHICH IT MOVES THE DECISION CRITERIA.
<table><tr><td>Date (2026)</td><td>Ver.</td><td>Change</td><td>Direction / note</td></tr><tr><td>Aug 6</td><td>v1</td><td>Initial preregistration: allocation-gain question; gate (progress  $\geq 2 \%$  , terminate  $< 0 . 5 \% ) ;$  all-branches-reportable design</td><td></td></tr><tr><td>Aug 6</td><td>v2</td><td>Allocation unit fixed (stochastic rollouts); mixed/homogeneous dual sets; positioned against LLM test-time-compute literature</td><td>pre-data</td></tr><tr><td>Aug 6</td><td>v3</td><td>Finite-pool axis (standard N×8 protocol) promoted to co-primary; gap-to-optimal normal- ization (LKH-3); trivial baselines (size-proportional, quantized) added; CI-based gate</td><td>pre-data</td></tr><tr><td>Aug 6</td><td>v4</td><td>After synthetic pipeline validation: bracket  $[ { \mathsf { d } } _ { - } { \mathsf { s p l i t } } , \ { \mathsf { d } } _ { - } { \mathrm { i n } } ]$  , no point estimates; noise- floor requirement; termination split (signal absence vs practical negligibility, the latter qualified by floor  $< 0 . 5 \% ) ;$  one scale-up only;  $\mathrm { ~ S ~ / ~ N ~ } \times \mathrm { ~ 4 ~ } \leq$  K; convex-hull regularization</td><td>tightened against false positives</td></tr><tr><td>Aug 12</td><td>v5</td><td>of marginal gains Amendment 1: pooled → per-instance null</td><td>favors detection (lowers floor); justified inde- pendently — the pooled null is invalid under</td></tr><tr><td>Aug 12</td><td>v5</td><td>Amendment 2: restrict sizes to matched checkpoints;  $\mathrm { n } = 2 0 0 $  labeled OOD arm</td><td>mixed sets regardless of outcome disfavors gate (reduces heterogeneity); oppo- site direction to Am. 1</td></tr><tr><td>Aug 12</td><td>v6</td><td>Amendment 3: core set n = 100, distribution contrast, single checkpoint (supersedes Am. 2&#x27;s remedy)</td><td>direction mixed; purpose is problem identity (single-solver allocation), not gate advantage</td></tr><tr><td>Aug 12</td><td>v7</td><td>Scale-up run registered with three-way outcome rules fixed pre-run Gate judgment: all four cells terminate (signal absence) under the registered rule. Amend-</td><td>pre-result rule left intact after results</td></tr><tr><td>Aug 13</td><td>v8</td><td>ment 4 recorded as a design defect only — no retroactive rule change (termination keyed  $\mathtt { d \_ i n }$  cannot credit positive d_split evidence)</td><td></td></tr><tr><td>Aug 13</td><td></td><td>Floor definition: max → median of per-source p95 (max diverges under heavy tails)</td><td>disfavors audit narrative (one d_in cell be- comes borderline); adopted because the defi- nition is correct</td></tr><tr><td>Aug 13</td><td></td><td>Confirmation preregistered: AM primary  $( \geq 2 \% ) .$  label-residual (&gt; 0, frozen ratios), SymNCO replication, POMO negative control with predicted direction; new seeds; one run only → all four endpoints passed</td><td>pre-result</td></tr><tr><td>Aug 13</td><td></td><td>Pre-drafting corrections: N-scaling claim retracted (artifact of max definition + source-count confound); signal-cost non-charging disclosed; charged variant added as exploratory</td><td>claims reduced after self-remeasurement</td></tr></table>

## APPENDIX B

## PRE-REGISTRATION TIMELINE

All gates, endpoints, and amendments were fixed before the corresponding results were seen. Each amendment records the direction in which it moves the decision criteria; the full record is Table VIII.

## REFERENCES

[1] W. Kool, H. van Hoof, and M. Welling, “Attention, learn to solve routing problems!” in Proc. Int. Conf. Learning Representations (ICLR), 2019, arXiv:1803.08475.

[2] Y.-D. Kwon, J. Choo, B. Kim, I. Yoon, Y. Gwon, and S. Min, “POMO: Policy optimization with multiple optima for reinforcement learning,” in Advances in Neural Information Processing Systems, vol. 33, 2020, pp. 21188–21198, arXiv:2010.16011.

[3] J. E. Smith and R. L. Winkler, “The optimizer’s curse: Skepticism and postdecision surprise in decision analysis,” Management Science, vol. 52, no. 3, pp. 311–322, 2006.

[4] I. Andrews, T. Kitagawa, and A. McCloskey, “Inference on winners,” Quarterly Journal of Economics, vol. 139, no. 1, pp. 305–358, 2024.

[5] H. White, “A reality check for data snooping,” Econometrica, vol. 68, no. 5, pp. 1097–1126, 2000.

[6] P. R. Hansen, “A test for superior predictive ability,” Journal of Business & Economic Statistics, vol. 23, no. 4, pp. 365–380, 2005.

[7] C. Snell, J. Lee, K. Xu, and A. Kumar, “Scaling test-time compute optimally can be more effective than scaling LLM parameters,” in Proc. Int. Conf. Learning Representations (ICLR), 2025, arXiv:2408.03314.

[8] C. Gao, H. Shang, K. Xue, and C. Qian, “Neural solver selection for combinatorial optimization,” in Proc. Int. Conf. Machine Learning (ICML), PMLR 267, 2025, arXiv:2410.09693.

[9] D. H. Bailey and M. López de Prado, “The deflated Sharpe ratio: Correcting for selection bias, backtest overfitting, and non-normality,” Journal of Portfolio Management, vol. 40, no. 5, pp. 94–107, 2014.

[10] Y. Xu, J. Zhang, H. Sun, Z. Zhou, T. Cao, and V. Aggarwal, “Towards reliable LLM evaluation: Correcting the winner’s curse in adaptive benchmarking,” arXiv:2605.05973, 2026 (preprint).

[11] M. Damani, I. Shenfeld, A. Peng, A. Bobu, and J. Andreas, “Learning how hard to think: Input-adaptive allocation of LM computation,” in Proc. Int. Conf. Learning Representations (ICLR), 2025, arXiv:2410.04707.

[12] B. Brown, J. Juravsky, R. Ehrlich, R. Clark, Q. V. Le, C. Ré, and A. Mirhoseini, “Large language monkeys: Scaling inference compute with repeated sampling,” arXiv:2407.21787, 2024.

[13] M. Kim, J. Park, and J. Park, “Sym-NCO: Leveraging symmetricity for neural combinatorial optimization,” in Advances in Neural Information Processing Systems, vol. 35, 2022, pp. 1936–1949, arXiv:2205.13209.

[14] S. Liu, Y. Zhang, K. Tang, and X. Yao, “How good is neural combinatorial optimization? A systematic evaluation on the traveling salesman problem,” IEEE Computational Intelligence Magazine, vol. 18, no. 3, pp. 14–28, 2023.

[15] B. Recht, R. Roelofs, L. Schmidt, and V. Shankar, “Do ImageNet classifiers generalize to ImageNet?” in Proc. Int. Conf. Machine Learning (ICML), PMLR 97, 2019, arXiv:1902.10811.