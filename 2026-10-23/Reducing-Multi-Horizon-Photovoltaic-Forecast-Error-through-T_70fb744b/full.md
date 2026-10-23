# Reducing Multi-Horizon Photovoltaic Forecast Error through Tree-Anchored Expert Combination

Sun Ze<sup>a</sup>, Zhou Liguo<sup>a,b,∗</sup>, Xu Yuqing<sup>a</sup>, Yu Lei<sup>a</sup>, Jiang Mingming<sup>a</sup>, Alois Knoll<sup>b</sup>

<sup>a</sup>School of Computer Science and Technology, Huaibei Normal University, China <sup>b</sup>Chair of Robotics, Artificial Intelligence and Real-time Systems, Technical University of Munich, Germany

## Abstract

A strong direct tree model may remain the best standalone photovoltaic forecaster, yet a weighted combination can reduce its residual errors. We examine whether complementary temporal predictors can lower the tree’s mean absolute error (MAE) over 15–240 minutes without weaker ensemble members degrading performance. The candidate bank contains LightGBM and seven neural implementations with diferent temporal representations. A combiner fitted to MAE has nonnegative weights that sum to one. Unhelpful candidates can receive zero weight, while the tree anchors the combination and provides a development-stage fallback. The main design shares weights across horizons; horizon-specific weights and alternative losses test whether further complexity is needed. On 56,078 windows from two co-located public PVDAQ systems, the historical shared-MAE policy reduces scaled MAE by 3.21% relative to the tree.

The latest bounded neural correction does not reduce MAE further than

this static combination. The results support selective combination of complementary predictions in this bank rather than equal averaging or indiscriminate addition of modules. Both analyses reuse a previously inspected evaluation period. They do not establish independent replication, weatherspecific causation or universal gains from eight experts.

Keywords: Photovoltaic power forecasting, Mean absolute error, Tree-anchored ensemble, Complementary temporal representations

## 1. Introduction

We examine whether an MAE-aligned combination can reduce a strong direct tree’s residual error across 16 forecast horizons. We predict photovoltaic AC power from 15 to 240 minutes ahead. LightGBM is the reference because a method that adds several neural networks should first demonstrate value over a competitive tree under the same information and target definitions. The goal is lower forecast error across this interval, not a larger model catalogue or a better result on an unrelated task.

Power variability and the usefulness of recent observations can change with forecast lead. Weather and verification choices also afect reported solarforecast performance (Antonanzas et al., 2016; Yang et al., 2020). These facts motivate testing several temporal representations, but do not prove that the errors in this dataset are caused by cloud motion. The present data do not support assigning a separate cloud regime to each expert. Instead, we test a narrower, measurable hypothesis: some forecasts that are worse than the tree individually can still contribute useful residual information when combined with it.

Related photovoltaic studies examine both temporal representations and how their predictions are combined. Costa (2022) compares convolutional, LSTM, and convolutional–LSTM networks at diferent forecast horizons and studies transfer from geographically aggregated training data to individual household systems. Zheng et al. (2025) combines kernel-clustering-based learner selection, joint feature and hyperparameter optimization, and adaptive weighted stacking for small-sample photovoltaic forecasting. Yang et al. (2025) couples numerical-weather-prediction-based time-series clustering with lag-enhanced gradient-boosting ensembles for day-ahead forecasting. These studies motivate testing temporal representations and fitted aggregation, but their datasets, information sets, and horizons difer from those used here. We do not compare their published error values directly with ours. The present study isolates a narrower question: whether MAE-fitted static weights can extract useful corrections to a strong direct tree, without adding a weather clustering or joint feature-search module.

Simply averaging additional models can increase MAE if their own errors outweigh any cancellation. Research on forecast combination and stacking supports estimating weights from held-out predictions (Bates and Granger, 1969; Clemen, 1989; Wang et al., 2023; Wolpert, 1992; Breiman, 1996). We use a bank of eight candidate models with a constrained linear combiner. The tree captures nonlinear relationships in engineered inputs, while the neural candidates represent temporal dependence in diferent ways. Nonnegative weights that sum to one allow a weak candidate to receive zero weight. Fitting those weights to absolute error aligns the combiner with the paper’s MAE target; squared-error fitting targets a diferent functional (Gneiting,

2011). No candidate is required to receive a positive weight.

The central question is: can an MAE-aligned, tree-anchored combination reduce a strong direct tree’s residual error across 16 forecast horizons? We first compare the combination against the tree, equal averaging and singleexpert selection. Comparisons of fitting loss and weight granularity then examine the individual design choices. Error accounting checks whether the observed benefit is compatible with useful member selection and cancellation (Brown et al., 2005); it is an explanation of the measured errors, not a separate prediction task or a causal weather analysis. The latest public retraining check tests the same accuracy question without future observed weather and with matched training-label access.

This empirical study examines an existing candidate bank and a restricted combiner. We do not claim a new stacking algorithm, an optimal number of experts, or an independently validated improvement over all single-model methods. Both historical and retraining analyses use an evaluation period inspected in earlier research, and their protocols and aggregation rules are reported separately.

## 2. Prediction Task and Expert Roles

## 2.1. Eight Candidates for One Forecasting Task

The primary bank uses PVDAQ systems 34 and 35, two co-located Nevada systems, not two independent climate domains (Deline et al., 2021). Each origin supplies 96 quarter-hour observations and produces 16 direct predictions, covering 15–240 minutes. The tree uses a direct head for each horizon. This is not a recursive experiment that repeatedly feeds the tree’s

own forecasts back as observations.

Figure 1 summarizes the prediction bank. It contains project implementations based on TiDE (Das et al., 2023), PatchTST (Nie et al., 2023), LSTM (Hochreiter and Schmidhuber, 1997), temporal convolution (Bai et al., 2018), N-HiTS (Challu et al., 2023), TimeMixer (Wang et al., 2024a), and TimeXer (Wang et al., 2024b), together with a gradient-boosted tree predictor (Friedman, 2001; Ke et al., 2017). The project names include multidilation TCN, N-HiTS-PV, TimeMixer-PV, TimeXer-PV, and LightGBM-PV. These labels identify local implementations; we have not established reproduction of every published architecture. All members predict the same power targets from the same origins; the bank does not contain eight diferent forecasting tasks. Three historical neural seeds are 20260914, 20260915, and 20260916; the tree predictions are shared across these evaluations. The neural models are trained for up to 20 epochs, batch size 256, learning rate $1 0 ^ { - 4 }$ , weight decay 10<sup>−4</sup>, and MAE training, with checkpoint choice on Epochval. The tree uses an L1 objective, at most 600 boosting rounds, a learning rate of 0.04, 31 leaves, a minimum child count of 64, an L2 coeficient of 1, and early stopping after 40 rounds. Before combination, member predictions are clipped to [0, 1.25]. The tree branch has a defined role: sixteen direct LightGBM heads provide a strong tabular baseline using the available history and contextual features. Tree predictions are never fed back as observations. They also anchor the combination and provide a fallback if a candidate policy fails the subsequent development check. The tree is not included simply to bring the ensemble to eight members.

The neural candidates provide alternative temporal representations.

TiDE uses a dense encoder–decoder, LSTM captures ordered recurrent dependence, and the multidilation TCN provides several local receptive fields. PatchTST groups observations into temporal patches, while the N-HiTS-PV and TimeMixer-PV implementations use diferent multiscale constructions. TimeXer-PV supplies a further contextual attention representation. These are architectural motivations, not demonstrated assignments to particular weather classes. The experiment tests their combined value; it does not identify the causal contribution of each internal component.

An expert here means a candidate predictor, not a weather-labelled specialist or an input-dependent routing branch. Within a fitted policy, weights are fixed across forecast origins. Several candidates may receive zero weight. Thus the hypothesis is selective use of complementary predictions, not that eight simultaneously active branches are necessary.

Power is divided by each system’s observed training maximum. We call the resulting metrics scaled MAE (sMAE) and scaled RMSE (sRMSE), not capacity normalized errors or accuracy percentages. Targets are sampled AC power, not interval-mean energy. We retain the scaling definition because error measures are not interchangeable across normalization conventions (Hyndman and Koehler, 2006). Finite negative targets are retained. Future weather inputs are retrospective ERA5 values without operational forecast issue times (Hersbach et al., 2020). Consequently, this experiment does not measure performance with genuinely issue-time-available numerical weather predictions. The persistence baselines do not have this futureweather information, which limits physical interpretation of that comparison.

For system $s ,$ let $b _ { s }$ be the maximum finite training power in kW (134.0 and 116.6 for systems 34 and 35). Observations and predictions are expressed in units of $b _ { s }$ . With H = 16, N evaluation windows, and $S = 3$ seeds, the reported metrics are

$$
\mathrm { s M A E } = \frac { 1 } { S } \sum _ { r = 1 } ^ { S } \frac { 1 } { N H } \sum _ { i , h } | q _ { i h } ^ { ( r ) } - y _ { i h } | ,\tag{1}
$$

$$
\mathrm { s R M S E } = \frac { 1 } { S } \sum _ { r = 1 } ^ { S } \sqrt { \frac { 1 } { N H } \sum _ { i , h } ( q _ { i h } ^ { ( r ) } - y _ { i h } ) ^ { 2 } } .\tag{2}
$$

These dimensionless metrics pool windows across systems before averaging seed-level scores; sRMSE is not the RMSE of seed-averaged predictions. Relative MAE reduction is $1 0 0 ( L _ { \mathrm { r e f } } - L _ { \mathrm { c a n d i d a t e } } ) / L _ { \mathrm { r e f } }$ , not an accuracypercentage-point increase. Horizon scores use the same order of aggregation with h fixed.

Every retained window has 96 historical and 16 future finite original power samples; missing power is not interpolated. The frozen preparation rules exclude of-grid, ambiguous daylight-saving and duplicate timestamps. Missing standardized weather values are encoded as zero with a false validity mask; physical weather values remain NaN. Clipping member predictions to [0, 1.25] while retaining negative labels can afect errors near zero. All four combination conditions use the same retained windows and clipped prediction bank. These choices were not tuned in this comparison.

## 2.2. Separation of Development Stages

The prepared data include 20,053 training windows, 8,198 Epochval windows, 4,311 Fusionfit windows, 3,411 Selection windows, 5,756 Confirmation windows and 56,078 evaluation windows. Training uses data from 2014–2015;

Direct multi-horizon prediction bank  
![](images/6739a543066962bab7b7c788314184cc1d04459d9993eca1cf11773b3b0dd4d4.jpg)  
Model-specific input preparation | Frozen expert predictions before combination  
Figure 1: Candidate representations for the same power-forecasting task. Model-specific input preparation feeds seven neural implementations and one direct tree predictor. The tree is the baseline and combination anchor; the remaining forecasts are retained only through their fitted weights. Each produces 16 direct forecasts, giving a $1 6 \times 8$ prediction matrix at each origin. The matrix cells show dimensions rather than measured values. Future weather comes from retrospective ERA5. All expert predictions are frozen before fitting the combiner.

all subsequent development stages precede the 2016 evaluation period. Evaluation comprises 897,248 origin–horizon target entries. Overlapping windows, repeated targets, co-location and shared tree predictions prevent treating these entries or random seeds as independent replications.

Stages follow local calendar boundaries: training ends on 1 July 2015, Epochval on 1 September, Fusionfit on 1 October, Selection on 1 November, and Confirmation on 1 January 2016; evaluation ends on 1 January 2017. Intervals are start-inclusive and end-exclusive. All 16 target timestamps must belong to their assigned stage, although historical inputs may extend into the preceding stage. Neither evaluation targets nor the evaluation period’s maximum power define the scaling constants.

Fusionfit estimates combination coeficients, Selection chooses the policy, and Confirmation only accepts that choice or falls back to the tree. Evaluation labels have no role in changing these choices. However, the 2016 period had already been inspected during earlier research. Freezing this execution does not remove the earlier research history. In particular, the integrated baseline and optional-prefix controls were specified after the earlier factorial evaluation. They are exploratory follow-up analyses, not a preregistered confirmation (Nosek et al., 2018).

## 3. Tree-Anchored MAE Reduction

## 3.1. Shared Weights as the Main Design

The shared-MAE policy is the main method evaluated in this paper. A single simplex vector pools evidence across leads and limits the number of fitted coeficients. Horizon-specific weights and MSE fitting are controls for

this design choice, not additional proposed methods. For member predictions $x _ { i h m }$ , target $y _ { i h }$ , and weights $w _ { h } \in \mathcal { S } _ { 8 } = \{ w \ge 0 : \mathbf { 1 } ^ { \top } w = 1 \}$ , define

$$
p _ { i h } = \sum _ { m = 1 } ^ { 8 } w _ { h m } x _ { i h m } .\tag{3}
$$

The shared parameterization uses $w _ { h } ~ = ~ w$ across all 16 horizons, giving seven free coeficients; the horizon-specific parameterization has 112. Each minimizes either pooled squared error or pooled absolute error on Fusionfit, without a ridge term. Separate leadwise fits minimize the corresponding horizon-specific loss. MAE is solved by a linear program with feasibility and objective checks. MSE uses SLSQP with an analytical gradient and an unscaled simplex first-order optimality-gap tolerance of $1 0 ^ { - 8 }$ ; the numerical objective is multiplied by $1 0 ^ { 4 }$ for solver conditioning. Hereafter, horizon-specific and leadwise denote the same parameterization. Explicitly, the shared objectives are

$$
\widehat { w } _ { \ell } = \arg \operatorname* { m i n } _ { w \in { \mathcal { S } _ { \mathrm { s } } } } \frac { 1 } { N _ { F } H } \sum _ { i \in F , h } \ell \left( \sum _ { m } w _ { m } x _ { i h m } - y _ { i h } \right) , \quad \ell ( u ) \in \{ | u | , u ^ { 2 } \} ,\tag{4}
$$

where F is Fusionfit and $N _ { F } = 4 3 1 1$ . Horizon-specific fitting replaces the pooled objective by one objective per h. The MAE solver is HiGHS dual simplex with default internal stopping settings; the post-solve objective agreement tolerance is $1 0 ^ { - 7 }$ and the residual-constraint tolerance is $1 0 ^ { - 6 }$ . SLSQP uses ftol= $1 0 ^ { - 1 2 }$ and at most 2000 iterations, followed by the optimalitygap check above. Solver failure or a failed numerical check aborts the fitting run; it is not a tree fallback. The latter is reserved for a completed policy that fails Confirmation. No failed solve is silently replaced by a local fit. Fit counts and execution provenance are provided in the supplement. Table 1 lists the four fitting conditions. Figure 2 separates coeficient fitting from the subsequent policy decision.

Table 1: Matched coeficient-fitting design. Every condition uses the same eight-expert predictions and Fusionfit windows. These degrees of freedom count only simplex coeficients before boundary solutions. Policy selection is a separate step.
<table><tr><td>Weights</td><td>Fit loss</td><td>Vectors</td><td>Free coeff.</td><td>Solver</td></tr><tr><td>Shared</td><td>MSE</td><td>1</td><td>7</td><td>SLSQP</td></tr><tr><td>Shared</td><td>MAE</td><td>1</td><td>7</td><td>Linear program</td></tr><tr><td>Leadwise</td><td>MSE</td><td>16</td><td>112</td><td>SLSQP</td></tr><tr><td>Leadwise</td><td>MAE</td><td>16</td><td>112</td><td>Linear program</td></tr></table>

Given fitted weights $w _ { h }$ , a policy with prefix k and mixing strength α predicts

$$
q _ { i h } ( k , \alpha ) = \left\{ \begin{array} { l l } { x _ { i h , 8 } , } & { h \leq k , } \\ { ( 1 - \alpha ) x _ { i h , 8 } + \alpha p _ { i h } , } & { h > k . } \end{array} \right.\tag{5}
$$

The original policy grid contains pure tree plus $k \in \{ 1 , \ldots , 1 5 \}$ crossed with $\alpha \in \{ 0 . 1 , 0 . 2 5 , 0 . 5 , 0 . 7 5 , 1 \}$ : 76 candidates per family and seed. Selection minimizes sMAE; ties favor a longer tree prefix and a smaller mixing strength. Confirmation requires strictly lower aggregate sMAE than the tree; it cannot choose a second-best candidate. We distinguish full fitted weights at all horizons, raw weights with only the first horizon forced to the tree, and the accepted policy. These versions must not be conflated when attributing an improvement to the fitting objective.

To test the prefix restriction itself, the integration adds five $k = 0$ candidates to each MAE family. The 81-candidate grid is selected and confirmed remotely using the same stages. Selection also chooses a single expert, either shared across horizons or separately by horizon; Confirmation vetoes any non-improving choice at the same aggregation level. Thus single-expert baselines are not chosen by their evaluation ranking.

The current shared-MAE implementation uses the pure tree for the first forecast and one shared vector for subsequent forecasts. This is an implementation constraint, not evidence that the switching time is universally optimal. The historical optional-prefix control permits $k = 0$ and thus tests this restriction. Neither version estimates an origin-dependent weather router.

## 3.2. Checking Why Combination Helps

Taking the tree as the reference, let $e _ { i h m } = x _ { i h m } - y _ { i h }$ . Averaging over the evaluated origin–horizon entries, define

$$
A ( w ) = \mathbb { E } _ { i , h } \left[ \sum _ { m } w _ { h m } | e _ { i h m } | - | e _ { i h , 8 } | \right] ,\tag{6}
$$

$$
C ( w ) = \mathbb { E } _ { i , h } \left[ \sum _ { m } w _ { h m } | e _ { i h m } | - \left| \sum _ { m } w _ { h m } e _ { i h m } \right| \right] .\tag{7}
$$

The triangle inequality gives $C \geq 0 ;$ direct subtraction then yields

$$
L ( q ) - L ( { \mathrm { t r e e } } ) = A - C ,\tag{8}
$$

where w represents the actual policy weights, including the tree mixture where applicable. This diagnostic describes realized loss, not causal efects. Both terms depend on labels, and cancellation on future targets is unknown at issue time. Development estimates can support a hypothesis but cannot ensure future acceptance. Maximizing cancellation alone may favor a prediction bank with larger member errors.

a Fit convex weights  
b Apply the horizon policy  
![](images/62b9b400f6f55332278429c3c7b940a3531183fb64dbd6287013028798f55e25.jpg)

## c Keep fitting, policy choice and evaluation separate

![](images/8050e04823cda9deb6e69286059e919f31156001a1714019de9033f079ffb2e3.jpg)  
Figure 2: Fusion and policy selection. (a) The factorial design crosses fitting loss with shared or horizon-specific simplex weights. (b) A policy uses the tree through horizon k and blends it with the fitted combination afterwards; $k = 0$ removes the compulsory tree prefix. (c) Distinct development stages fit the experts and weights, select $( k , \alpha )$ , and confirm the selected policy or return to the tree. Confirmation does not search for a replacement candidate. Counts are windows, not independent observations. Evaluation labels have no role in these decisions, although the evaluation period was inspected in earlier research.

## 3.3. Baselines and Uncertainty

Besides the eight members and equal weights, we retain the original ensemble as a historical baseline. It fits shared MSE weights with a penalty toward uniform weights and uses a separate uniform-mixture selection and confirmation rule. It is not the unregularized shared-MSE factorial cell; its comparison does not isolate the fitting-loss efect. The integration also evaluates raw and [0, 1.25]-clipped persistence. A specified smart-persistence proxy uses pvlib 0.13.1 Haurwitz clear-sky global horizontal irradiance (GHI), recorded coordinates, and altitude zero (Huertas Tato and Centeno Brito, 2019; Anderson et al., 2023). It scales the last nonnegative power by target/origin clear-sky GHI when origin GHI exceeds 5 $\mathrm { W / m ^ { 2 } }$ ; otherwise it retains that power. A zero target GHI gives zero output, and the final prediction is clipped to [0, 1.25]. None of the thresholds is tuned using these evaluation results. This proxy lacks panel geometry and cloud advection and is not a definitive physical baseline.

We average losses over seeds, not forecasts. Paired gains are calculated as reference sMAE minus candidate sMAE. Circular blocks of 7, 14 or 28 consecutive UTC calendar days resample the same origins jointly across methods, horizons and systems in 10,000 draws. Day counts weight the sampled losses. This construction retains local temporal dependence instead of resampling overlapping windows independently (Shao and Yu, 1993). Seven and 14 days are the existing analysis settings; 28 days provides a sensitivity check for longer dependence. All exceed the four-hour target span, but that fact does not establish an optimal block length or remove seasonal dependence. We do not select a block length by significance. Reported percentile intervals are conditional on the fitted bank and decisions. They do not account for retraining variability, research-history selection or generalization to independent sites. Main-table and horizon-curve intervals are pointwise and unadjusted. A separate retrospective sensitivity family contains the four raw factorial contrasts (two loss contrasts and two weight-granularity contrasts). Bonferroni percentile endpoints at $0 . 0 5 / ( 2 \times 4 )$ and $1 - 0 . 0 5 / ( 2 \times 4 )$ give nominal 95% family coverage only to the extent that the underlying block intervals are valid. This does not correct for earlier research decisions or simultaneous screening of all horizons. We report efect sizes, do not label this analysis confirmatory, and do not infer equivalence from an interval spanning zero.

## 4. Results

## 4.1. Does Combination Beat the Strong Tree?

Table 2 compares policies on identical evaluation windows. The accepted shared-MAE and horizon-specific-MAE policies yield sMAE values of 0.025420 and 0.025353, compared with 0.026262 for LightGBM. The relative reductions are 3.21% and 3.46%. Both shared and per-horizon singleton selectors retain the tree at every horizon. Equal weights perform worse. The original ensemble has the lowest sRMSE, 0.064109, compared with 0.064348 and 0.064398 for the two accepted MAE policies. Thus the MAE gains do not establish dominance on both metrics. Persistence controls are reported separately in the supplement because they lack the bank’s retrospective future weather.

Table 2: Matched eight-member comparison. Entries average three seed-specific losses, not predictions. Positive gain denotes lower sMAE than the tree. Persistence controls are reported in the supplement.
<table><tr><td>Method</td><td>sMAE</td><td>sRMSE Gain (%)</td><td></td></tr><tr><td>Equal eight-member weights</td><td>0.028874</td><td>0.066017</td><td>-9.94</td></tr><tr><td>Tree / accepted singletons</td><td>0.026262</td><td>0.066494</td><td>+0.00</td></tr><tr><td>Original eight-member ensemble</td><td>0.026095</td><td>0.064109</td><td>+0.64</td></tr><tr><td>Shared MSE, accepted</td><td>0.025945</td><td>0.065726</td><td>+1.21</td></tr><tr><td>Shared MAE, accepted</td><td>0.025420</td><td>0.064348</td><td>+3.21</td></tr><tr><td>Leadwise MSE, accepted</td><td>0.026262</td><td>0.066494</td><td>+0.00</td></tr><tr><td>Leadwise MAE, accepted</td><td>0.025353</td><td>0.064398</td><td>+3.46</td></tr></table>

## 4.2. Does MAE Alignment Help the Accuracy Target?

Under the common raw first-horizon tree constraint, MSE-to-MAE fitting reduces sMAE by 0.000440 for shared weights and 0.000762 for horizonspecific weights (Table 3). Both pointwise 14-day intervals exclude zero. Figure 4a also reports the four-contrast Bonferroni sensitivity intervals: the shared contrast then spans zero, whereas the horizon-specific contrast remains positive. The favorable aggregate point estimates should not be read as uniform evidence across parameterizations and uncertainty conventions.

All horizon-specific-MSE policies are vetoed to the tree. Only one shared-MSE seed is accepted, and that seed slightly outperforms its shared-MAE counterpart. Selection and confirmation therefore do not make MAE fitting superior in every run. Complete full, raw and accepted scores are reported together in Supplementary Section 1.

Table 3: Paired sMAE reductions. Circular UTC-day block resampling uses 14-day blocks and 10,000 draws. Intervals are conditional, exploratory and unadjusted for multiple comparisons or research selection.
<table><tr><td>Contrast</td><td>Reduction</td><td>95% interval</td></tr><tr><td>MAE gain, shared</td><td>0.000440</td><td>[0.000062, 0.000752]</td></tr><tr><td>MAE gain, leadwise</td><td>0.000762</td><td>[0.000459, 0.001030]</td></tr><tr><td>Leadwise gain, MSE</td><td>-0.000242</td><td>[-0.000316, -0.000182]</td></tr><tr><td>Leadwise gain, MAE</td><td>0.000080</td><td>[-0.000012, 0.000182]</td></tr><tr><td>Leadwise gain, MAE (accepted)</td><td>0.000066</td><td>[0.000010, 0.000124]</td></tr></table>

## 4.3. Are More Weight Parameters Needed?

With MSE fitting, horizon-specific weights worsen raw sMAE by 0.000242 relative to shared weights. With MAE fitting, they improve it by only 0.000080, with a 14-day interval spanning zero. The 7-day interval is slightly positive and the 28-day interval spans zero. Evidence for the additional 105 free coeficients is therefore sensitive to temporal resampling, and no practical-equivalence margin was specified in advance.

Figure 3 separates the loss-alignment and weight-granularity contrasts across all 16 horizons. The latter is not uniformly favorable to horizonspecific weights. No horizon is selected from these curves for the main claim. After policy selection and confirmation, the accepted horizon-specific MAE score is 0.000066 lower than the shared score. That comparison includes policy decisions and does not isolate coeficient dimensionality.

a RQ1: MSE minus MAE fitting  
![](images/3c0d573bd7a7bd8b412e8e1df7de471c338c21f3273bda476c69bf97de902f1a.jpg)

b RQ2: shared minus horizon-specific MAE  
![](images/f9b8259075f294425ed19427ab5da4e7af065db0ee158627d32361a35e71903a.jpg)  
Figure 3: Paired sMAE reductions at all 16 horizons. Positive values favor MAE fitting in (a) and horizon-specific MAE weights in (b). Raw policies share the first-horizon tree constraint, so the first zero is imposed, not estimated. Points average seed-specific losses. Bars are conditional pointwise 95% intervals from 14-day UTC blocks and are not adjusted for screening horizons. The curves are exploratory and do not define a switching threshold.

## 4.4. Member Error and Realized Cancellation

For shared weights, moving from MSE to MAE fitting reduces weighted member error A by 0.001200 but also reduces cancellation C by 0.000761, yielding a net sMAE reduction of 0.000440. For horizon-specific weights, the corresponding reductions are 0.001354 and 0.000591, yielding 0.000762 (Table 4; Fig. 4b). The reduction in weighted member error exceeds the loss of cancellation, producing the aggregate gain. These quantities describe realized loss, not changes in the individual frozen predictors or evidence of a causal mechanism.

Table 4: Descriptive A/C accounting relative to the tree for the raw factorial policies. A negative diference A minus C denotes improvement. More cancellation alone is not the objective.
<table><tr><td>Family</td><td>A</td><td>C</td><td>A-C</td></tr><tr><td>Shared MSE</td><td>0.002023</td><td>0.002422</td><td>-0.000399</td></tr><tr><td>Shared MAE</td><td>0.000823</td><td>0.001661</td><td>-0.000839</td></tr><tr><td>Leadwise MSE</td><td>0.002256</td><td>0.002413</td><td>-0.000157</td></tr><tr><td>Leadwise MAE</td><td>0.000903</td><td>0.001822</td><td>-0.000919</td></tr></table>

## 4.5. Robustness and Sensitivity

Within-sample system and seasonal summaries are reported in Supplementary Section 4. The shared-weight MSE-to-MAE gain reverses in summer (sMAE reduction −0.000205), despite a positive annual average. Horizonspecific MAE improves over shared MAE in system 34 (0.000146) but slightly worsens it in system 35 (−0.000010), and also worsens it in winter. These descriptive subgroups are neither independent replications nor evidence for a subgroup-specific policy.

![](images/11ab56dc4f73d17a997d1248ebc8835e58b33779a1fb127224afe63edec9c7d5.jpg)

b RQ3: sources of the MSE-to-MAE gain  
![](images/792474252c4912835b9403a4d79332f5cf2aa567f162c804332290e06e611dc2.jpg)  
Figure 4: Factorial evidence and error accounting for raw policies. (a) Overall paired efects using 14-day blocks. Dark intervals are pointwise 95% intervals; pale intervals show Bonferroni sensitivity for four contrasts. This retrospective adjustment does not account for earlier research decisions. (b) The MSE-to-MAE gain equals $( A _ { \mathrm { M S E } } - A _ { \mathrm { M A E } } ) + ( C _ { \mathrm { M A E } } -$ C<sub>MSE</sub>). Cancellation contributes negatively under both parameterizations. Bar heights are descriptive point estimates with no uncertainty bars.

The optional-prefix control permits zero tree-only horizons. One seed then selects that option; mean horizon-specific-MAE sMAE is 0.025356, close to 0.025353 with a compulsory prefix. This weakens a universal tree-first rule. It does not identify when a recursively applied tree would fail: every member here makes direct multi-horizon predictions. Supplementary Sections 3 and 4 report the prefix choices and sensitivity of the intervals to 7/14/28-day blocks, including unfavorable contrasts.

## 4.6. Matched Public Retraining without Future Weather

The latest retraining check examines the same question with an operationally simpler input set. It retains the PVDAQ systems and frozen chronological splits, making it a protocol check on an exposed period rather than a new external validation. All neural parameters are initialized anew; no private pretrained weights or station embeddings are imported. Inputs comprise the 96 power observations up to the origin, known future calendar/solar geometry, and available target-aligned power lags from one, two and seven days earlier. Weather values and their validity masks are zero. The direct tree receives the same raw power history, lag information and known geometry as the full pipeline. Power at the origin is assumed to be available at issue time; the experiment does not establish actual telemetry latency.

The base experts use 20,053 training windows, fusion uses 4,311 windows, and optional joint adaptation uses 3,411 windows. The standalone comparison tree is refitted on their union of 27,775 windows, so it also receives the adaptation labels. Epochval selects base checkpoints and tree boosting rounds; Confirmation selects the optional correction strength from {0, 0.25, 0.5, 1}. Test targets do not enter any of these decisions. For seeds 201, 202 and 203, each neural candidate is trained for eight epochs of 64 station-balanced batches of 128 examples, with learning rate $3 \times 1 0 ^ { - 4 }$ . The tree uses an L1 objective, 31 leaves, a learning rate of 0.05, at most 500 boosting rounds and 40-round early stopping. The comparison matches information, label access and seed count, but not parameter count or computation. This fixed budget does not establish convergence of every candidate.

This check averages the three seed predictions before scoring. Its primary metric is daylight MAE averaged equally over the two systems:

$$
L _ { \mathrm { d a y } } = \frac { 1 } { 2 } \sum _ { s \in \{ 3 4 , 3 5 \} } \frac { \sum _ { ( i , h ) \in D _ { s } } \left| \overline { { q } } _ { i h } - y _ { i h } \right| } { \left| D _ { s } \right| } , \qquad \overline { { q } } = \frac { 1 } { 3 } \sum _ { r = 1 } ^ { 3 } q ^ { ( r ) } ,\tag{9}
$$

where $D _ { s }$ contains geometrically daylight targets for system s. The numerical station scales remain 134.0 and 116.6 kW. These are fixed scaling constants inherited from the prepared data, not independently verified nameplate capacities. All methods use the same daylight mask, geometry-night zeroing and [0, 1.25] clipping. The aggregation and feature choices difer from those in the historical experiment; its 3.21% reduction and the reduction in this check must therefore remain separate results.

Table 5 supports the static combination as the simpler MAE-focused design. Later-horizon tree weights are 0.5849, 0.5292 and 0.5554 for the three seeds. Only PatchTST, N-HiTS-PV and TimeXer-PV receive positive neural weights in these fits; TiDE, LSTM, the TCN and TimeMixer-PV receive zero static weight. This does not show that the inactive architectures are generally inefective. It shows that this dataset and fitting recipe do not justify an equally weighted or obligatorily eight-active-branch interpretation.

Table 5: Matched public retraining check for the exposed 2016 period. Daylight MAE is computed with equal-system weighting after averaging predictions. The independent tree receives all training labels available to the full pipeline. Percent reductions refer only to this table’s primary metric.
<table><tr><td>Method</td><td> $L _ { \mathrm { d a y } }$ </td><td>Daylight MAE (kW)*</td><td>Reduction (%)</td></tr><tr><td>Matched-label standalone tree</td><td>0.05086079</td><td>6.478810</td><td>0.00</td></tr><tr><td>Static tree-anchored combina- 0.04995439</td><td></td><td>6.372286</td><td>1.78</td></tr><tr><td>tion Combination + bounded joint 0.04995897</td><td></td><td>6.372638</td><td>1.77</td></tr></table>

<sup>∗</sup>The kW column pools daylight targets across systems and is a separate aggregation, not $L _ { \mathrm { d a y } }$ multiplied by a common capacity.

The latest optional extension jointly updates TimeMixer-PV and PatchTST and computes a bounded residual correction. It leaves the first horizon, geometrical night and baseline forecasts below 0.01 unchanged, and caps the correction at 0.005 in scaled units. Its four epochs of 64 batches use a combined absolute/squared-error objective with a risk penalty, not pure MAE. Confirmation selects strengths 0.5, 0 and 0. Aggregate MAE is slightly higher than for the static combination, although the RMSE-derived score improves slightly. This extension is therefore a negative incremental result, not the source of the paper’s MAE gain.

## 5. Discussion and Limitations

## 5.1. What the Accuracy Result Supports

The evidence supports a strong tree combined with selectively weighted temporal predictions. In the historical bank, single-expert selection retains the tree at every horizon, whereas equal averaging performs worse. An MAEaligned combination can nevertheless improve on the tree, indicating that the best standalone forecast does not exhaust the useful information in the bank. Shared weights recover most of the historical accepted gain without the 105 extra free coeficients required by horizon-specific fitting. Matched-label retraining gives the same qualitative result without future weather. These findings justify testing a combiner; they do not establish that eight experts are optimal or that every member is necessary.

The tree anchor, MAE objective and simplex weights that permit zeros address the same accuracy problem. A strong tabular baseline limits weak comparisons, fitting to MAE targets the reported error, and zero weights limit dilution by weak candidates. Protecting the first lead remains a conservative implementation choice, but the optional-prefix results do not support a universal tree-only rule for a fixed initial duration. Additional routing, attention or correction modules must demonstrate their own incremental MAE evidence. The latest joint correction fails that test in the present retraining check.

## 5.2. What Remains Unresolved

We have not established that cloud fluctuations, ramp events or a particular meteorological variable cause the remaining errors. The historical data use retrospective ERA5, and the retraining check removes weather inputs entirely. A claim of weather-specific robustness would require an issue-time weather archive, prespecified regime definitions and error comparisons within those regimes. Adding a weather module solely because it is physically plausible would not answer the present MAE question.

Both systems are co-located and both analyses reuse an exposed evaluation period. The historical tree is shared across neural seeds; the retraining tree also has efectively identical fits under the fixed deterministic recipe. Overlapping targets and seeds are not independent site replications. Historical conditional intervals do not include the sequence of research decisions; subgroup reversals and adjustment sensitivity limit claims of stable improvement. Diferent aggregation, inputs and fitting budgets prevent interpreting the two tables as a numerical replication or a general neural-versus-tree ranking. The matched retraining check supplies no future-weather advantage, but does not validate real telemetry latency or deployment performance.

The next confirmation should freeze the static tree-anchored MAE combination and its matched-label tree comparator before examining a new site or time period. It should test whether the residual MAE reduction exceeds a prespecified practical threshold. Member-removal tests are needed before claiming an indispensable expert, and published-time weather evidence is needed before claiming weather-driven improvement.

## 6. Conclusion

This study addresses one problem: reducing the residual MAE of a strong direct tree over 15–240-minute photovoltaic power forecasts. A treeanchored, shared-MAE combination lowers historical scaled MAE by 3.21%. A separate public retraining check without future weather lowers daylight, equal-system MAE by 1.78% against a tree with the same training-label access. The protocols difer, and both reuse an exposed period. Within this bank, the results support selective combination over indiscriminate averaging. They do not establish a need for eight active experts, horizon-specific weights or the latest joint correction. Independent confirmation is needed before claiming transferable accuracy gains or an operational weather-forecasting benefit.

## Data and Code Availability

PVDAQ is publicly available (Deline et al., 2021). Frozen predictions, configuration records and non-fitting audit scripts are retained locally. The matched retraining records include a frozen feature/split protocol, code and data hashes, three seed predictions, an independent metric audit and hashes for 126 remote model/configuration artifacts. The machine-readable retraining protocol calls the fixed station scales capacities; the manuscript conservatively uses the inherited prepared-data scaling definition stated above. Public archival release and redistribution checks remain pending; no public reproducibility deposit is claimed.

## References

Anderson, K.S., et al., 2023. pvlib python: 2023 project update. Journal of Open Source Software 8 (92), 5994. doi: 10.21105/joss.05994.

Antonanzas, J., et al., 2016. Review of photovoltaic power forecasting. Solar Energy 136, 78–111. doi: 10.1016/j.solener.2016.06.069.

Bai, S., Kolter, J.Z., Koltun, V., 2018. An empirical evaluation of generic convolutional and recurrent networks for sequence modeling. arXiv:1803.01271.

Bates, J.M., Granger, C.W.J., 1969. The combination of forecasts. Operational Research Quarterly 20 (4), 451–468. doi: 10.1057/jors.1969.103.

Breiman, L., 1996. Stacked regressions. Machine Learning 24, 49–64. doi: 10.1007/BF00117832.

Brown, G., Wyatt, J., Harris, R., Yao, X., 2005. Diversity creation methods: A survey and categorisation. Information Fusion 6 (1), 5–20. doi: 10.1016/j.infus.2004.04.004.

Challu, C., Olivares, K.G., Oreshkin, B.N., Garza Ramirez, F., Mergenthaler Canseco, M., Dubrawski, A., 2023. NHITS: Neural hierarchical interpolation for time series forecasting. Proceedings of the AAAI Conference on Artificial Intelligence 37 (6), 6989–6997. doi: 10.1609/aaai.v37i6.25854.

Clemen, R.T., 1989. Combining forecasts: A review and annotated bibliography. International Journal of Forecasting 5 (4), 559–583. doi: 10.1016/0169-2070(89)90012-5.

Costa, R.L. de C., 2022. Convolutional-LSTM networks and generalization in forecasting of household photovoltaic generation. Engineering Applications of Artificial Intelligence 116, 105458. doi: 10.1016/j.engappai.2022.105458.

Das, A., Kong, W., Leach, A., Mathur, S., Sen, R., Yu, R., 2023. Longterm forecasting with TiDE: Time-series dense encoder. Transactions on Machine Learning Research. arXiv:2304.08424.

Deline, C., et al., 2021. Photovoltaic Data Acquisition (PVDAQ) public datasets. DOE OEDI/NREL. doi: 10.25984/1846021.

Friedman, J.H., 2001. Greedy function approximation: A gradient boosting machine. Annals of Statistics 29 (5), 1189–1232. doi: 10.1214/aos/1013203451.

Gneiting, T., 2011. Making and evaluating point forecasts. Journal of the American Statistical Association 106 (494), 746–762. doi: 10.1198/jasa.2011.r10138.

Hersbach, H., et al., 2020. The ERA5 global reanalysis. Quarterly Journal of the Royal Meteorological Society 146, 1999–2049. doi: 10.1002/qj.3803.

Hochreiter, S., Schmidhuber, J., 1997. Long short-term memory. Neural Computation 9 (8), 1735–1780. doi: 10.1162/neco.1997.9.8.1735.

Huertas Tato, J., Centeno Brito, M., 2019. Using smart persistence and random forests to predict photovoltaic energy production. Energies 12 (1), 100. doi: 10.3390/en12010100.

Hyndman, R.J., Koehler, A.B., 2006. Another look at measures of forecast accuracy. International Journal of Forecasting 22 (4), 679–688. doi: 10.1016/j.ijforecast.2006.03.001.

Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., Liu, T.-Y., 2017. LightGBM: A highly eficient gradient boosting decision tree. Advances in Neural Information Processing Systems 30.

Nie, Y., Nguyen, N.H., Sinthong, P., Kalagnanam, J., 2023. A time series is worth 64 words: Long-term forecasting with transformers. In: Proceedings of the International Conference on Learning Representations. arXiv:2211.14730.

Nosek, B.A., Ebersole, C.R., DeHaven, A.C., Mellor, D.T., 2018. The preregistration revolution. Proceedings of the National Academy of Sciences of the USA 115 (11), 2600–2606. doi: 10.1073/pnas.1708274114.

Shao, Q.-M., Yu, H., 1993. Bootstrapping the sample means for stationary mixing sequences. Stochastic Processes and their Applications 48 (1), 175– 190. doi: 10.1016/0304-4149(93)90113-I.

Wang, S., et al., 2024a. TimeMixer: Decomposable multiscale mixing for time series forecasting. In: Proceedings of the International Conference on Learning Representations. arXiv:2405.14616.

Wang, X., Hyndman, R.J., Li, F., Kang, Y., 2023. Forecast combinations: An over 50-year review. International Journal of Forecasting 39 (4), 1518– 1547. doi: 10.1016/j.ijforecast.2022.11.005.

Wang, Y., et al., 2024b. TimeXer: Empowering transformers for time series forecasting with exogenous variables. Advances in Neural Information Processing Systems 37. arXiv:2402.19072.

Wolpert, D.H., 1992. Stacked generalization. Neural Networks 5 (2), 241–259. doi: 10.1016/S0893-6080(05)80023-1.

Yang, D., et al., 2020. Verification of deterministic solar forecasts. Solar Energy 210, 20–37. doi: 10.1016/j.solener.2020.04.019.

Yang, F., Fu, X., Li, Z., Qiu, D., Badihi, H., 2025. A novel clusteringensemble learning model for day-ahead photovoltaic power forecasting. Engineering Applications of Artificial Intelligence 161, 112281. doi: 10.1016/j.engappai.2025.112281.

Zheng, S., Li, D., Li, Y., Zeng, Z., Zhou, Y., Li, C., Song, K., 2025. An adaptive weighted stacking ensemble framework for photovoltaic power generation forecasting with joint optimization of features and hyperparameters. Engineering Applications of Artificial Intelligence 144, 110075. doi: 10.1016/j.engappai.2025.110075.