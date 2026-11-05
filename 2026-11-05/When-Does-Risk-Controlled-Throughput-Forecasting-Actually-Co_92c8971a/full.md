# When Does Risk-Controlled Throughput Forecasting Actually Control Risk? A Reproduction Study on LEO Satellite Links

MAYAN SHARMA, Vellore Institute of Technology, India

KRITI SAINI, Vellore Institute of Technology, India

DEVANSH BEHL, Vellore Institute of Technology, India

SUGANYA R, Vellore Institute of Technology, India

S. VIJAYARAGAVAN, Nelco Network Products Ltd., India

Risk-budgeted throughput forecasting has been proposed as a way to make LEO satellite capacity estimates safe for admission control: a point forecast is converted into a conservative bound whose overestimation rate is held at a user-specified budget. We reproduce the state of the art, BG-CFQS, on the three Starlink traces its authors used and on an independent fourth dataset, and report four findings.

First, its risk guarantee is reproducible but conditional on the calibration split. Under a random split of the same data we recover every published quantity, including the risk-pass column at 3 of 3 datasets. Under a contiguous temporal split, which is what a deployed terminal faces, it holds on 1 of 3.

Second, and independent of any split, the method cannot serve a risk budget below 0.15. Its published candidate quantile set is � = [0.15, 0.40], and when the budget is tighter than the risk at the most conservative candidate the selection interval collapses to a point and the achieved rate pins there. On all four datasets the achieved rate is identical at budgets of 0.05, 0.10 and 0.15, overshooting by 1.4x to 3.7x at the tightest. The paper evaluates only at 0.35, where the floor never binds.

Third, we show that static conformal calibration cannot improve a ranking at all, only shift an operating point along an unchanged curve, and confirm the consequence exactly on latency spike detection. Online calibration is not so constrained and improves detection AUPRC by up to 51%.

Fourth, we evaluate the natural remedy, group-conditional online recalibration, and find that it splits into two mechanisms with very diferent standing. The online part replicates: it tracks the budget within 1% at every point from 0.05 to 0.35, where static calibration is 16 to 21% of, and it dominates static calibration on both tail risk and accuracy at matched achieved risk. The group-conditional part does not: on three of four datasets, conditioning on any covariate we have, including measured serving-satellite geometry, is worse than not conditioning at all. We attempted to predict which datasets benefit, from a statistic computable on the calibration split, and failed. We report that failure in full.

What is new here is empirical, not algorithmic. The mechanisms we evaluate are all published, and Section 6 says which paper each comes from. We claim the findings, not the methods.

CCS Concepts: • Networks → Network measurement; Network performance modeling; • Computing methodologies → Machine learning.

Additional Key Words and Phrases: LEO satellite networks, Starlink, throughput forecasting, conformal prediction, risk control, reproducibility

## 1 Introduction

A LEO access link is a hard thing to allocate bandwidth over. Capacity moves on a 15-second scheduling cadence [4, 15], handovers are frequent, and the achievable rate varies by more than a factor of three within a single hour [12]. An allocator that acts on a point forecast will over-promise whenever the forecast is optimistic, and over-promising drops sessions.

The natural response is to allocate against a conservative bound rather than a point forecast, and to control how often that bound is wrong. BG-CFQS [18] does this by selecting a quantile level whose overestimation rate on a calibration set meets a user-specified budget. It reports a global overestimation rate of 0.349 against a budget of 0.35, and a risk-pass on all three of its datasets.

This paper asks whether that guarantee survives contact with the conditions a deployed terminal actually operates in, and what the remedies cost. We do not propose a method. Both mechanisms we evaluate are published, and Section 6 is explicit about which paper each comes from.

## 1.1 What is new in this paper

Novelty in a measurement paper is novelty of knowledge, not of algorithm. Each of the following is, to our knowledge, unreported, and each is a fact about published methods or about LEO links that was not previously on record. We searched for prior work on each before claiming it; Section 6 gives the full prior-art accounting.

(1) BG-CFQS cannot serve a risk budget below the low end of its candidate quantile set. The achieved overestimation rate is identical at budgets of 0.05, 0.10 and 0.15 on all four datasets, overshooting by up to 3.7x, because the selection interval collapses and the method returns the same quantile. Derived from their Algorithm 1, confirmed against their Table II, and measured. Their own evaluation is conducted at a single budget where the floor never binds, so it cannot surface. This is the strongest result in the paper, and a reader can check it in ten minutes. (Section 4.2)

(2) The method’s risk guarantee depends on an exchangeable calibration split. We recover every published quantity under a random split, including the risk-pass column at 3/3, and 1/3 under a contiguous temporal split. That split conformal degrades under non-exchangeability is textbook [2]; that this method, on these traces, moves from 3/3 to 1/3 is not on record, and it is the diference between the method working and not working in deployment. (Section 4.1)

(3) Measured serving-satellite geometry predicts throughput level but not residual structure. StarNet [12] establishes that elevation, distance and visible-satellite count correlate with throughput, and we reproduce that. We show they carry no information about the forecaster’s uncertainty: conditioning a calibration layer on them is worse than not conditioning at all, on all three of the StarNet traces. The level/residual distinction is not drawn for these features anywhere we could find, and it is what determines whether the covariates are useful for risk control. (Section 5.3)

(4) Static calibration misses a risk budget in both directions on LEO links, with the sign set by the drift. It overshoots on one dataset and undershoots by a fifth on another. The undershoot half is under-reported in a literature that emphasises under-coverage, and it is not a safe failure: it silently withholds capacity the operator said it would risk. (Section 5.2)

(5) A negative result on predicting when group conditioning helps. The intuition that measurable heterogeneity predicts benefit is appealing and cheap to implement. It is wrong on our data, in two independent ways, and we report the failure with the mechanism. (Section 5.4)

(6) Static conformal calibration cannot improve a ranking, only an operating point. A static bound is the point forecast plus a constant; adding a constant does not reorder predictions; AUPRC depends only on the order. We observe the consequence exactly, to three decimal places, on all three locations: point forecast and static bound score identical AUPRC for latency spike detection. An online per-regime bound is not a constant ofset, so it can reorder, and improves AUPRC by 43% and 51% on two of three links. This is a reason to prefer online calibration that is independent of every risk result in this paper. (Section 5.5)

(7) A price-invariant decision rule for whether risk control is worth deploying. Whether a conservative allocator beats an aggressive one depends on prices, and under commodity pricing it does not. The invariant is the break-even ratio: how much more a violated sessionhour must cost than a sold one. It lands at 3.87x, 3.92x and 4.05x on three independent links, within 5% of each other. Consumer broadband does not clear it; enterprise and URLLC services do. (Section 5.6)

What we do not claim. The two mechanisms evaluated here are both published: group-conditional online recalibration is GCACI [13], and deciding whether to condition from data is Clustered Conformal Prediction [7]. Our implementation of the former was written independently and is its naive special case. We include both as baselines rather than presenting either as ours.

## 2 Setup

Datasets. Three Starlink traces released with StarNet [12], covering Chicago (1,123,832 samples, 2024-04-26 to 05-28), Osnabruck (613,295, 2024-07-13 to 07-31) and Victoria (145,053, 2024-07-11 to 07-28), each at 1 Hz with serving-satellite identity, elevation, distance, visible-satellite count and co-located weather. Plus the per-second iperf release of WetLinks [11], 1,019,109 samples of measured capacity at Osnabruck over 180 days, which is an independent measurement campaign with a diferent instrument and no satellite telemetry.

Forecaster. The StarNet backbone: a two-layer GRU sequence-to-sequence model with a periodical embedding over the 15-second cycle and attention. We reproduce its published accuracy within 1.1% RMSE and 2.4% MAE on average across the three locations (Appendix A), which is the gate everything downstream rests on.

Metrics. Following BG-CFQS exactly, so the numbers are comparable: OverRate, the fraction of decisions where the bound exceeded the realised capacity; MPE, the mean positive error; P95+Err. Reported globally and on the two risk slices the paper defines, P30 and P10, the lowest 30% and 10% of realised throughput. These are where over-allocation actually drops sessions.

Splits. Contiguous temporal blocks, 60/20/20, with a purge band so no time step crosses a boundary. Every run carries a leak check.

Use ofgenerative AI.. Generative AI tools were used to assist with implementing the code and with drafting and editing the text. Every number reported here was produced by the authors’ code and read from saved result files, and the authors checked all content and take responsibility for it.

## 3 What a risk budget is supposed to buy

An allocator sets a budget � and expects the bound to over-promise at most that often. The whole value proposition is that � is a dial the operator sets, not a number the method reports afterwards. Two things can go wrong, and both do:

• the achieved rate can exceed the budget, which drops sessions;

• the achieved rate can fall below the budget, which wastes capacity the operator explicitly said it was willing to risk.

The second failure is under-discussed. It is not safe, it is expensive: an allocator that spends 0.29 of a 0.35 budget is leaving utilisation on the table and getting no risk reduction the operator asked for.

Table 1. BG-CFQS under the two splits, averaged over the three StarNet traces. Only the split difers between the first two rows.
<table><tr><td>Split</td><td>OverRate</td><td>P30</td><td>P10</td><td>Risk pass</td></tr><tr><td>Temporal</td><td>0.377</td><td>0.750</td><td>0.909</td><td>1/3</td></tr><tr><td>Random (exchangeable)</td><td>0.340</td><td>0.671</td><td>0.848</td><td>3/3</td></tr><tr><td>Published [18]</td><td>0.349</td><td>0.65 to 0.71</td><td>0.83 to 0.86</td><td>3/3</td></tr></table>

## 4 BG-CFQS

## 4.1 The guarantee is conditional on the split

Reproducing BG-CFQS with its published configuration $( L = 7 5 , H = 1 5 , \varepsilon = 0 . 3 5 , T = [ 0 . 1 5 , 0 . 4 0 ]$ � = 0.05, � = 5, XGBoost with pinball loss) under contiguous temporal splits, the conditional failure it reports comes out clearly, but its risk column does not: average OverRate 0.391 against their 0.349, risk pass 0/3 against 3/3, while accuracy came out better than published. Better accuracy with worse risk is not a signature of a broken reimplementation, so we looked for the confound. Table 1 holds everything constant and varies only the split.

Under the exchangeable split every published quantity lands in range. The reimplementation is correct, and the guarantee holds under the assumption it inherits from split conformal prediction [16].

It does not survive a temporal split. This is not a criticism of the paper, which does not report the temporal case, and it is not a discovery about conformal prediction, where non-exchangeability is textbook [2]. It is a statement about deployment: the assumption is not available to a terminal, and the guarantee is the reason to use the method.

The detail that matters most is in the P10 column of the random split. At 0.848, in the arm where the method passes its budget 3/3, the conditional failure is nearly as bad as under drift. So the tail failure is intrinsic to selecting one global quantile and is not a symptom of non-stationarity. The two problems are independent, and a fix for one is not a fix for the other.

## 4.2 Budgets below 0.15 are unservable

This is a structural limitation and it holds under any split. From Algorithm 1 of the paper, verbatim:

4: if R(tau\_max) <= e then   
5: Set boundary interval [t-, t+] = [tau\_max, tau\_max]   
6: else if R(tau\_min) > e then   
7: Set boundary interval [t-, t+] = [tau\_min, tau\_min]   
21: Build Q\_fine = LinSpace(t-, t+, M)   
23: if exists tau in Q\_fine such that R(tau) <= e then   
24: Select tau\* = argmin A(tau) over the feasible set   
25: else   
26: Select tau\* = argmin A(tau) + lambda max(R(tau) - e, 0)

When the requested budget is tighter than the risk achievable at the most conservative candidate � , line 6 fires. The interval collapses to a single point, line 21 produces � copies of $\tau _ { \mathrm { m i n } } .$ the feasible set at line 23 is empty, and line 26 returns � however small the budget gets.

With their published $T = [ 0 . 1 5 , 0 . 4 0 ]$ (their Table II), the achieved rate therefore pins at �(0.15). The signature is an achieved rate that is identical at every budget at or below 0.15, and it appears on all four datasets (Table 2, Figure 1).

Table 2. BG-CFQS achieved OverRate across the budget sweep. The three tightest budgets return the same number on every dataset.
<table><tr><td>Budget</td><td>0.05</td><td>0.10</td><td>0.15</td><td>0.20</td><td>0.25</td><td>0.30</td><td>0.35</td><td>Overshoot at 0.05</td></tr><tr><td>Chicago</td><td>0.165</td><td>0.165</td><td>0.165</td><td>0.209</td><td>0.267</td><td>0.322</td><td>0.378</td><td>3.3x</td></tr><tr><td>Osnabruck (StarNet)</td><td>0.127</td><td>0.127</td><td>0.127</td><td>0.169</td><td>0.213</td><td>0.251</td><td>0.295</td><td>2.5x</td></tr><tr><td>Victoria</td><td>0.070</td><td>0.070</td><td>0.070</td><td>0.092</td><td>0.120</td><td>0.159</td><td>0.194</td><td>1.4x</td></tr><tr><td>Osnabruck (WetLinks)</td><td>0.185</td><td>0.185</td><td>0.185</td><td>0.244</td><td>0.297</td><td>0.349</td><td>0.405</td><td>3.7x</td></tr></table>

![](images/5dcec336dcdac2262e6ecee038346ced8256f33a7e68c3a9fdab9b220530d8e7.jpg)  
Fig. 1. BG-CFQS achieved OverRate against the requested budget. Every line is flat across the shaded band of budgets at or below 0.15, because the method returns the same quantile there. The dashed line marks achieved equal to requested.

The floor value difers by dataset because it is $R \left( \tau _ { \operatorname* { m i n } } \right)$ , the risk the most conservative available quantile happens to achieve on that link. The flatness does not: on every dataset the three tightest budgets return the same number, because the method is returning the same quantile. For comparison, online recalibration on the same traces tracks every one of those budgets within 1% (Section 5.2).

Their evaluation is conducted at $\varepsilon = 0 . 3 5$ (their Table II, “Default risk budget”), where the floor never binds on any of the four datasets, so their own results cannot expose it.

A budget of 0.35 permits over-promising on more than a third of decisions. We know of no allocator that would accept that, which makes the tighter budgets the interesting ones and the floor a practical obstacle rather than a curiosity.

Widening � fixes it, and would no longer be their method. The general lesson is about the method class: budget-guided selection over a bounded candidate set inherits that set’s bounds as hard limits on the achievable risk.

## 5 The remedy, and how much of it works

## 5.1 Two mechanisms

The natural fix has two parts, and they are separable:

Table 3. Requested budget against achieved OverRate, mean of the three StarNet locations.
<table><tr><td>Budget</td><td>0.05</td><td>0.10</td><td>0.15</td><td>0.20</td><td>0.25</td><td>0.30</td><td>0.35</td></tr><tr><td>Online achieved</td><td>0.050</td><td>0.100</td><td>0.150</td><td>0.200</td><td>0.250</td><td>0.300</td><td>0.350</td></tr><tr><td>Static achieved</td><td>0.040</td><td>0.080</td><td>0.120</td><td>0.160</td><td>0.204</td><td>0.248</td><td>0.293</td></tr><tr><td>Static error</td><td>-21%</td><td>-20%</td><td>-20%</td><td>-20%</td><td>-19%</td><td>-17%</td><td>-16%</td></tr></table>

![](images/55c00aafea215a0909106dfbc9799280bcecec2be2617db55f7e2ebcaa2f1e05.jpg)  
Fig. 2. Budget tracking on the StarNet traces, mean of three locations. Online recalibration sits on the dashed target line at every budget; static split conformal runs 16 to 21% below it.

• Online recalibration. Stop treating the operating point as a constant. Update it from outcomes as they are observed. This is ACI [9].

• Group conditioning. Maintain a separate operating point per regime, where a regime is a bucket over covariates the terminal can compute at prediction time. Combined with the above this is GCACI [13].

## 5.2 The online mechanism replicates

Table 3 and Figure 2 show budget tracking across the sweep, averaged over three locations. The online layer tracks within 0.8% everywhere. Static split conformal is 16 to 21% low, the second failure mode from Section 3.

Note the sign against the WetLinks result, where static calibration overshot (0.423 against 0.35) because the test month was slower than the calibration month. Here the test period is easier and it undershoots. The direction of the miss is a property of the drift, not of the method, which is a stronger claim than either dataset alone supports.

Table 4 makes the comparison a naive table gets wrong: online against static, at matched achieved risk.

Online calibration has better tail risk and better accuracy at every operating point. Methods must be compared at matched achieved risk: ranked by raw P10 alone, static BG-CFQS appears best, but it is spending 0.289 of a 0.35 budget and conservatism buys P10 directly.

Table 4. Online against static per-regime calibration at matched achieved risk, mean of the three StarNet locations. The 0.35 row is omited because static calibration never reaches 0.35.
<table><tr><td>Achieved risk</td><td>Online P10</td><td>Static P10</td><td>Change</td><td>Online MAE</td><td>Static MAE</td></tr><tr><td>0.050</td><td>0.145</td><td>0.170</td><td>-14.6%</td><td>51.54</td><td>52.66</td></tr><tr><td>0.100</td><td>0.256</td><td>0.289</td><td>-11.3%</td><td>39.62</td><td>40.64</td></tr><tr><td>0.150</td><td>0.357</td><td>0.398</td><td>-10.4%</td><td>33.67</td><td>34.47</td></tr><tr><td>0.200</td><td>0.438</td><td>0.499</td><td>-12.1%</td><td>29.76</td><td>30.53</td></tr><tr><td>0.250</td><td>0.521</td><td>0.589</td><td>-11.6%</td><td>27.23</td><td>28.00</td></tr><tr><td>0.300</td><td>0.598</td><td>0.666</td><td>-10.1%</td><td>25.36</td><td>26.39</td></tr></table>

Table 5. Efect of each regime set on P10 OverRate under online calibration at budget 0.35, mean of the three StarNet locations.
<table><tr><td>Regime axes</td><td>P10</td><td>Change against no conditioning</td></tr><tr><td>None</td><td>0.666</td><td>n/a</td></tr><tr><td>Level (observed look-back mean)</td><td>0.673</td><td>+1.1%</td></tr><tr><td>Visible satellite count</td><td>0.676</td><td>+1.4%</td></tr><tr><td>Serving elevation</td><td>0.685</td><td>+2.9%</td></tr><tr><td>All measured geometry</td><td>0.693</td><td>+4.0%</td></tr><tr><td>15 s scheduling phase</td><td>0.694</td><td>+4.2%</td></tr><tr><td>All four axes</td><td>0.709</td><td>+6.3%</td></tr></table>

## 5.3 The group-conditional mechanism mostly does not

Table 5 reports online calibration at budget 0.35, mean P10 over three locations. With � = 3,718 in the Chicago P10 slice the standard error is 0.008, so these diferences are real.

Every axis is worse than not conditioning, including the measured serving-satellite geometry. StarNet reports that elevation, distance and visible count correlate with throughput level, and we reproduce that. They do not carry residual structure, which is what a calibration layer needs, and the distinction matters.

The result is not an artifact of the learning rate. Across seven values spanning two orders of magnitude, conditioning beat no conditioning in 8 of 42 cells, and seven of the eight are Victoria. On Chicago it loses at every setting.

The degradation tracks regime count rather than axis choice (Figure 3): 1 regime 0.666, 4 regimes 0.673, 18 regimes 0.693, 51 regimes 0.709, with a correlation of −0.571 between lo ${ \bf g } _ { 1 0 }$ (outcomes per regime) and P10. Online per-regime calibration must converge one parameter per regime from a stream, and 51 regimes leave about 400 outcomes each. Static per-regime calibration, which fits once, is less afected, and conditioning on level still helps it by about 3%. This is the estimation-noise efect Ding et al. [7] describe for class-conditional conformal prediction, appearing here in an online setting.

## 5.4 A negative result on predicting when conditioning helps

Because conditioning helps on one dataset and hurts on three, we tried to build a test that decides which case a trace is in, using only the calibration split.

![](images/3e1c0040695e2513ce16e63ae8f8abe8d5b55a5f0163f181d66a87fffc920168.jpg)  
Fig. 3. P10 OverRate against the number of regimes, online calibration at budget 0.35, mean of the three StarNet locations. Finer partitions are worse.

Table 6. Ofset spread measured on the calibration split, against the measured efect of conditioning on P10.
<table><tr><td>Location</td><td>Pre-test spread</td><td>Measured effect</td></tr><tr><td>Chicago</td><td>1.99 Mbps</td><td>+6.1% (hurts)</td></tr><tr><td>Osnabruck</td><td>5.83 Mbps</td><td>+0.1% (neutral)</td></tr><tr><td>Victoria</td><td>5.17 Mbps</td><td>-1.7% (helps)</td></tr></table>

The statistic: fit one ofset per regime on the calibration split, weight each by the standard error of an empirical quantile [14], and run Cochran’s � [5] against $\chi ^ { 2 }$ , requiring both significance and a substantial $I ^ { 2 }$ [10]. This is the standard heterogeneity test from meta-analysis, and doing it for conformal calibration is Clustered Conformal Prediction [7] in binary form.

It does not work, for two independent reasons.

It fires everywhere. On Chicago, where conditioning costs 6.1%, the test returns $Q = 1 8 . 3$ on 3 degrees of freedom, $p = 0 . 0 0 0 4 , I ^ { 2 } = 0 . 8 4$ , for an ofset spread of 1.99 Mbps. With about 1,200 calibration points per regime a 2 Mbps diference is comfortably significant, and $I ^ { 2 }$ is scale free so it agrees that the diference is real. Both hurdles ask whether a diference exists; neither asks whether it is worth anything.

And the statistic does not order the outcome (Table 6). Osnabruck has the largest pre-test spread and gets no benefit; Victoria has a smaller one and gets the only benefit.

A correction to our own earlier analysis. We initially recorded this relationship as monotonic across four datasets. Those figures were taken from the ofsets applied during the test replay, a post-hoc quantity not available before a decision is made. Computed correctly on the calibration split, the relationship does not hold. We report this because the failure is the result: the intuition that heterogeneity predicts benefit is appealing, cheap to implement, and wrong on our data.

What would be needed is a stability test rather than a spread test. Both Osnabruck and Victoria have real per-regime diferences on the calibration split; only Victoria’s persist into the test period. That is a diferent question and we did not attempt it.

Table 7. Latency spike detection, AUPRC with the multiple of chance in brackets. The point forecast and the static bound are identical on every link.
<table><tr><td>Location</td><td>Chance</td><td>Point forecast</td><td>Static bound</td><td>Online bound</td></tr><tr><td>Chicago</td><td>0.046</td><td>0.628 (13.6x)</td><td>0.628 (13.6x)</td><td>0.600 (13.0x)</td></tr><tr><td>Osnabruck</td><td>0.065</td><td>0.348 (5.4x)</td><td>0.348 (5.4x)</td><td>0.527 (8.1x)</td></tr><tr><td>Victoria</td><td>0.123</td><td>0.360 (2.9x)</td><td>0.360 (2.9x)</td><td>0.516 (4.2x)</td></tr></table>

Table 8. Service availability by policy on the three StarNet links.
<table><tr><td>Location</td><td>Policy</td><td>Availability</td><td>Nines</td><td>Outages</td><td>Mean time to repair</td></tr><tr><td>Chicago</td><td>Point forecast</td><td>93.50%</td><td>1.19</td><td>8,967</td><td>9.0 s</td></tr><tr><td>Chicago</td><td>Online, budget 0.05</td><td>99.16%</td><td>2.08</td><td>1,494</td><td>5.3 s</td></tr><tr><td>Osnabruck</td><td>Point forecast</td><td>93.37%</td><td>1.18</td><td>4,514</td><td>9.2 s</td></tr><tr><td>Osnabruck</td><td>Static, budget 0.05</td><td>99.18%</td><td>2.09</td><td>651</td><td>5.1 s</td></tr><tr><td>Victoria</td><td>Point forecast</td><td>93.44%</td><td>1.18</td><td>1,001</td><td>8.8 s</td></tr><tr><td>Victoria</td><td>Static, budget 0.05</td><td>99.25%</td><td>2.13</td><td>146</td><td>5.0 s</td></tr></table>

## 5.5 Static calibration cannot improve a ranking

The same layer serves latency spike detection with the bound direction reversed: an upper bound on latency, where the risk event is underestimation. Following Casparsen et al. [4], each 15-second period aligned to the recovered phase is Good if its latency stays within 50 ms and Degraded otherwise. At 1 Hz a period holds fifteen samples, so their 99% criterion is efectively a maximum.

The two-directional failure of static calibration reappears, more severely than for throughput. At a budget of 0.10 the online layer achieves 0.101, 0.100 and 0.088 on Chicago, Osnabruck and Victoria. Static calibration achieves 0.143 (+43%), 0.313 (+213%) and 0.033 (−67%). That the same failure appears with a diferent target and the bound direction reversed is stronger evidence that it is a property of the link than the throughput result alone.

Table 7 shows the ranking result. The point forecast and the static bound have identical AUPRC on all three locations, and this is a necessity rather than a coincidence. A static conformal bound is the point forecast plus a constant. Adding a constant cannot change the order of the predictions, and AUPRC depends only on that order. Static calibration can therefore move the operating point along a precision-recall curve but can never improve the curve. The online per-regime bound applies diferent corrections in diferent regimes and at diferent times, so it can reorder predictions, and on two of the three links it improves AUPRC by 51% and 43%. On Chicago it is slightly worse. F1 is less favourable to the online method: it wins on Osnabruck (0.429 against 0.349) and Victoria (0.348 against 0.229) and loses on Chicago (0.508 against 0.601), where higher recall costs more precision than it gains.

## 5.6 A price-invariant rule for deploying risk control

Admission control admits ⌊�/10⌋ sessions of 10 Mbps against a bound �, following BG-CFQS. Availability is delivered over promised service time. Table 8 shows that calibration moves each link from roughly one nine to two, cuts the number of outages by five to six times, and shortens the mean outage from about nine seconds to about five. No policy reaches three nines; the best is 2.13.

Table 9. Every mechanism evaluated in this paper is prior work.
<table><tr><td>Mechanism</td><td>Prior work</td></tr><tr><td>Per-regime conformal calibration</td><td>Mondrian conformal prediction [17]</td></tr><tr><td>Per-regime online conformal calibration</td><td>GCACI [13]; concurrently [1]</td></tr><tr><td>Parameter-free version</td><td>POGO [3]</td></tr><tr><td>Online control of a user-specified risk</td><td>Rolling RC [8]</td></tr><tr><td></td><td>Deciding whether to condition, from data Clustered Conformal Prediction [7]; AFCP [19]</td></tr><tr><td>Conformal risk control for resource allo- URLLC scheduling [6] cation</td><td></td></tr><tr><td>Admission control from a safe throughput BG-CFQS [18], eqs. (23) to (26) bound</td><td></td></tr></table>

Whether this is worth buying depends on prices. The cost model combines tiered SLA credits for over-allocation with foregone revenue for under-allocation, which is what stops the cheapest policy from being “admit nothing”. Under our default illustrative prices the aggressive policy is cheapest: on Chicago, 0.473 per hour at budget 0.35 against 0.629 at budget 0.05. That is a statement about the prices, not about the method. Foregone revenue accrues on every unsold session-hour, while SLA credits are a fraction of a small fee and only bite below three nines.

The quantity that does not depend on prices is the break-even ratio: how much more a violated session-hour must cost than a sold one before the risk-controlled policy pays for itself. It is 3.87x on Chicago, 3.92x on Osnabruck and 4.05x on Victoria, within 5% across three links on two continents. Consumer broadband, where a degraded minute costs little more than the revenue on it, does not clear a ratio of 4. Enterprise connectivity and URLLC services, where credits run to 10% to 25% of a monthly fee, clear it comfortably. The model prices violations linearly whereas real SLA remedies are stepped and capped, so 4 is an approximation to the crossing point.

## 6 What is and is not new here

Stated plainly, because the answer is mostly “not new”. Table 9 lists every mechanism used and where it comes from.

Our implementation of per-regime adaptive calibration was written independently and is the naive special case of GCACI: for a partition, GCACI’s group membership vector is one-hot and its update reduces to one parameter per regime. We include GCACI and Rolling RC as baselines.

What this paper contributes is the findings in Section 1.1. They are new knowledge, not new machinery: a structural limitation in a published method that its own evaluation cannot surface, the condition its guarantee depends on, a covariate class that predicts level but not uncertainty, a two-directional failure of static calibration, a limit on what static calibration can do for detection, a price-invariant deployment rule, and a negative result.

We were careful about this distinction because we got it wrong first. The project was designed around a per-regime calibration layer believed to be novel, and that belief survived until it was checked against the literature rather than against our own reading of two papers. The check cost a day and removed the methods claim entirely. We record that here because the temptation in a project like this is to check late, or to check in a way that confirms.

## 7 Limitations

• Four datasets, three of them from one measurement campaign. Any cross-dataset claim rests on four observations.

• Cross-location means two European sites 150 km apart on the WetLinks side, and three sites from one collection on the StarNet side.

• The 15-sample constraint on WetLinks caps look-back plus horizon at 15, so neither StarNet’s 30/5 nor BG-CFQS’s 75/15 configuration runs there.

• The online layer’s budget control is empirical, not a finite-sample guarantee, and it requires outcome feedback after each decision.

• No live terminal. Every result is replay over recorded traces.

• We did not search the shrinkage and empirical-Bayes literature, where the poolingversus-conditioning question has its own long history. Our negative result in Section 5.4 should be read against that gap.

## References

[1] Anastasios N. Angelopoulos, Michael I. Jordan, and Ryan J. Tibshirani. 2025. Gradient Equilibrium in Online Learning. Journal of Machine Learning Research 26, 305 (2025).

[2] Rina Foygel Barber, Emmanuel J. Candès, Aaditya Ramdas, and Ryan J. Tibshirani. 2023. Conformal Prediction Beyond Exchangeability. The Annals ofStatistics 51, 2 (2023), 816–845.

[3] Beepul Bharti, Ambar Pal, Jacopo Teneggi, and Jeremias Sulam. 2026. Parameter-Free and Group Conditional Online Conformal Prediction. arXiv:2606.00419.

[4] Anders Casparsen, Jesper Jakobsen, Jimmy J. Nielsen, Petar Popovski, and Israel Leyva Mayorga. 2026. Statistical Characterization and Prediction of E2E Latency over LEO Satellite Networks. arXiv:2601.08439.

[5] William G. Cochran. 1954. The Combination of Estimates from Diferent Experiments. Biometrics 10, 1 (1954), 101–129.

[6] Kfir M. Cohen, Sangwoo Park, Osvaldo Simeone, and Shlomo Shamai Shitz. 2023. Guaranteed Dynamic Scheduling of Ultra-Reliable Low-Latency Trafic via Conformal Prediction. arXiv:2302.07675

[7] Tifany Ding, Anastasios N. Angelopoulos, Stephen Bates, Michael I. Jordan, and Ryan J. Tibshirani. 2023. Class-Conditional Conformal Prediction with Many Classes. In Advances in Neural Information Processing Systems (NeurIPS).

[8] Shai Feldman, Liran Ringel, Stephen Bates, and Yaniv Romano. 2023. Achieving Risk Control in Online Learning Settings. Transactions on Machine Learning Research (2023).

[9] Isaac Gibbs and Emmanuel Candès. 2021. Adaptive Conformal Inference Under Distribution Shift. In Advances in Neural Information Processing Systems (NeurIPS).

[10] Julian P. T. Higgins and Simon G. Thompson. 2002. Quantifying Heterogeneity in a Meta-Analysis. Statistics in Medicine 21, 11 (2002), 1539–1558.

[11] Dominic Laniewski, Eric Lanfer, Bernd Meijerink, Roland van Rijswijk-Deij, and Nils Aschenbruck. 2024. WetLinks: A Large-Scale Longitudinal Starlink Dataset with Contiguous Weather Data. In Proceedings ofthe Network Trafic Measurement and Analysis Conference (TMA).

[12] Zikun Liu, Fan-Xue Gabriella Reidys, Sarah Tanveer, and Deepak Vasisht. 2025. Vivisecting Starlink Throughput: Measurement and Prediction. Proceedings ofthe ACM on Networking 3, CoNEXT4 (2025). doi:10.1145/3768971

[13] Ramya Ramalingam, Shayan Kiyani, and Aaron Roth. 2025. The Relationship between No-Regret Learning and Online Conformal Prediction. arXiv:2502.10947.

[14] M. M. Siddiqui. 1960. Distribution of Quantiles in Samples from a Bivariate Population. J. Res. Nat. Bur. Standards 64B (1960), 145–150.

[15] Hammas Bin Tanveer, Mike Puchol, Rachee Singh, Antonio Bianchi, and Rishab Nithyanand. 2023. Making Sense of Constellations: Methodologies for Understanding Starlink’s Scheduling Algorithms. In Companion ofthe 19th International Conference on Emerging Networking Experiments and Technologies (CoNEXT Companion).

[16] Vladimir Vovk, Alexander Gammerman, and Glenn Shafer. 2005. Algorithmic Learning in a Random World. Springer.

[17] Vladimir Vovk, David Lindsay, Ilia Nouretdinov, and Alex Gammerman. 2003. Mondrian Confidence Machine. Technica Report. Royal Holloway, University of London.

[18] Ming Xie, Wei Zhang, Jun Luo, Hao Zhang, Lei Yang, Qian Zhang, and Boon Hee Soong. 2026. Risk-Aware Safe Throughput Forecasting for Starlink Networks. arXiv:2605.09508.

[19] Yanfei Zhou and Matteo Sesia. 2024. Conformal Classification with Equalized Coverage for Adaptively Selected Groups. In Advances in Neural Information Processing Systems (NeurIPS).

## A Reproduction gate

StarNet with look-back 30, output 5, their published per-location step sizes and a contiguous 8:2 split (Table 10). Gaps scatter in both directions. We do not reproduce their whole-trace scaler, which fits before splitting; ours fits on training windows only.

Table 10. StarNet reproduction against the published results.
<table><tr><td>Location</td><td>RMSE</td><td>Published</td><td>Gap</td><td>MAE</td><td>Published</td><td>Gap</td></tr><tr><td>Chicago</td><td>42.56</td><td>40.33</td><td>+5.5%</td><td>31.94</td><td>29.88</td><td>+6.9%</td></tr><tr><td>Victoria</td><td>38.38</td><td>41.08</td><td>-6.6%</td><td>29.55</td><td>30.84</td><td>-4.2%</td></tr><tr><td>Osnabruck</td><td>38.24</td><td>36.48</td><td>+4.8%</td><td>28.42</td><td>27.11</td><td>+4.8%</td></tr><tr><td>Average</td><td>39.73</td><td>39.30</td><td>+1.1%</td><td>29.97</td><td>29.28</td><td>+2.4%</td></tr></table>

## B Artifacts

All code, run configurations and result files will be released publicly on acceptance. Every table in this paper is generated from a saved result file with its configuration snapshot beside it. The two reproduction gates, the split sensitivity experiment, the learning-rate sweep and the heterogeneity gate are each a single script.