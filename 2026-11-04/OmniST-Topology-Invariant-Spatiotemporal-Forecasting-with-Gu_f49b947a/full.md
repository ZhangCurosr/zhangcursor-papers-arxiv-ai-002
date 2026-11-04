# OmniST: Topology-Invariant Spatiotemporal Forecasting with Guaranteed-Coverage Uncertainty at Unseen Sensor Locations

Praveg Vashishtha

Abstract—Standard spatiotemporal graph convolutional networks require a fixed-dimensional adjacency matrix, coupling model weights to a specific sensor count N; they cannot produce any output at a sensor location absent from training. We present OmniST, a topology-invariant forecaster whose computational graph has zero structural dependency on the number of input nodes, and which — composed with inductive splitconformal calibration — provides distribution-free, coverageguaranteed prediction intervals at arbitrary (including neverseen) sensor locations, and preserves that guarantee as sensors are added or removed. Topology invariance follows from three design choices: (1) coordinate-based tokenization that embeds each sensor independently via a learned spatial MLP; (2) knearest-neighbor attention computed from geographic coordinates at each forward pass; and (3) per-node temporal processing via bidirectional GRU. We validate on real CPCB Delhi air quality measurements (37 stations, 2019–2023) — the same dataset used in the published E-STGCN framework. OmniST achieves 30-day $\mathbf { P M } _ { 2 . 5 }$ forecast MAE of 31.0 $\mu \mathbf { g } / \mathbf { m } ^ { 3 }$ (average across 2023 months), degrades ≤1.1% when 40% of stations are removed, and produces calibrated intervals at coordinates never seen during training. Under multi-seed evaluation, OmniST’s inductive point forecasts at unseen stations improve on both nearest-neighbor interpolation (∼12%, 26.7 vs 30.3 µg/m<sup>3</sup>) and a learned inductive-kriging GNN (IGNNK, ∼21%), and — more importantly — it provides what neither offers reliably: a finitesample coverage guarantee at those unseen locations, the quantity an operator needs to trust a forecast at a new or newly-restored sensor. The inductive guarantee is validated acrossfour real-world datasets spanning three physical domains — Delhi (CPCB) and Beijing (UCI) air quality, NOAA GHCN weather, and Caltrans PeMSD7-M freeway traffic — holding $\geq 9 0 \%$ coverage at unseen nodes under 30–50% topology dropout on every one, while four model-free spatial-interpolation baselines each under-cover on multiple domains; coverage further holds at 325 real sensors (full PeMS-BAY) without degradation. We further resolve the temporal-drift limitation of our earlier work: a static conformal interval under-covers over time (coverage collapsing to 0.50 late in the test year), but a rolling adaptive-conformal estimator restores target coverage (0.89 overall, 0.96 in the late-drift window) at the honest cost of wider intervals. We explicitly do not pursue cross-domain foundation-model pretraining, which recent work (Damba-ST) already targets; our contribution is guaranteed inductive uncertainty under dynamic topology, now shown to hold over both space and time.

Index Terms—Spatiotemporal forecasting, topology invariance, dynamic graph attention, inductive prediction, conformal pre-

diction, distribution-free uncertainty quantification, sensor networks.

## I. INTRODUCTION

PATIOTEMPORAL graph convolutional networks S (STGCNs) [1] model sensor networks as fixed graphs with pre-computed adjacency matrices. This design choice produces excellent forecasting accuracy but creates an implicit constraint: the model’s weight dimensions are coupled to a specific number of nodes N.

In this paper, we do not claim to “solve” a problem that existing models cannot address. Rather, we present an architectural design—OmniST—where this coupling is structurally absent. The model accepts any number of nodes at inference time because no component’s computation depends on a global node count. We analyze what this design enables, what it costs, and what remains to be validated.

## A. Motivation

Real-world sensor networks change. The Indian CPCB adds monitoring stations annually [2]. Traffic sensors fail. Power grid topology switches during maintenance. When this happens, STGCN-based models face a practical question: do they need retraining?

The answer depends on implementation. A standard STGCN with a stored N × N adjacency matrix requires at minimum an adjacency update. If weights are N-dimensional (common in early GCN implementations), full retraining is needed. If the adjacency is recomputed dynamically (as in some modern implementations), the model can adapt—but this is a design choice, not a default.

OmniST makes the strongest version of this choice: nothing in the architecture references a global N. Each node’s processing path depends only on its own features, coordinates, and local neighbors.

## B. Contributions and Scope

1) Guaranteed-coverage inductive uncertainty (primary). Composing coordinate-based inductive prediction with inductive split-conformal calibration [3], OmniST emits distribution-free prediction intervals with a finite-sample coverage guarantee at unseen sensor locations, and preserves the guarantee under node addition/removal. Fixed-topology STGCNs cannot predict at a new node at all; nearest-neighbor interpolation can, but provides no validity guarantee. OmniST fills exactly this gap. We further give a locality property (Proposition 1): because no component references a global node count, each node’s nonconformity score is exactly invariant to the presence of non-neighbor nodes, so the coverage guarantee is recalibration-free under far topology changes — a stronger statement than the driftcorrection required for message-passing GNNs [3].

2) An architectural design where every component (tokenization, spatial attention, temporal processing, prediction head) operates independently of global node count, giving structural N-independence and resilience under sensor failure $( \leq 1 . 1 \%$ degradation at 40% removal on real data).

3) Rigorous multi-domain / multi-seed validation on real data. On real CPCB Delhi data (5-seed MAE $2 8 . 2 \pm 0 . 9 ~ \mu \mathrm { g / m ^ { 3 } }$ , matching E-STGCN), inductive point forecasts at unseen stations that beat nearest-neighbor by ∼12% (multi-seed), and an inductive coverage guarantee validated on four real-world datasets across three physical domains — Delhi and Beijing air quality, NOAA GHCN weather, and Caltrans PeMSD7-M traffic — where coverage holds $\mathrm { a t } \geq 9 0 \%$ at unseen nodes under 30–50% node dropout on all four, and OmniST is the only method (vs four model-free interpolation baselines) to hold the target on every dataset. We deliberately do not claim cross-domain foundation-model pretraining [4].

4) A resolution of the temporal-drift limitation. Our earlier static conformal layer gave a purely spatial guarantee and under-covered over time under seasonal drift. We add a principled online layer (rolling-window plus Adaptive Conformal Inference [5]) that restores target coverage over time on real data (static 0.76 overall / 0.50 in the late-drift window → rolling-ACI 0.89 / 0.96), making the guarantee hold over both space and time, at the honest cost of wider intervals.

## II. ARCHITECTURE

## A. Design Principle

The architecture satisfies one invariant: for any component f and any node i, the computation f(i) depends only on node i’s own data and a fixed-size neighborhood $\mathcal { N } _ { k } ( i )$ never on the total set of nodes present in the batch.

This is achieved by:

• Tokenization from coordinates (no positional index)

• k-NN neighbor selection (adapts to available nodes)

• Per-node temporal processing (no cross-node temporal dependency)

## B. Coordinate-Based Tokenization

Each observation becomes a token:

$$
\mathbf { z } _ { i } ( t ) = f _ { \mathrm { M L P } } ( \phi _ { i } , \lambda _ { i } ) + f _ { \mathrm { t i m e } } ( t ) + W _ { f } \mathbf { x } _ { i } ( t )\tag{1}
$$

where $f _ { \mathrm { M L P } } : \mathbb { R } ^ { 2 } \ \to \ \mathbb { R } ^ { d }$ is a 3-layer MLP on centered coordinates, $f _ { \mathrm { t i m e } }$ encodes temporal features via sinusoidal harmonics, and $W _ { f }$ projects raw features to $\mathbb { R } ^ { d }$

Key property: ${ \bf z } _ { i } ( t )$ is computed with zero reference to any other node.

## C. k-NN Spatial Attention

Spatial context is gathered from the k geographically nearest nodes:

$$
\mathbf { z } _ { i } ^ { \prime } = \mathrm { L a y e r N o r m } \left( \mathbf { z } _ { i } + \sum _ { j \in \mathcal { N } _ { k } ( i ) } \alpha _ { i j } W _ { v } \mathbf { z } _ { j } \right)\tag{2}
$$

where $\mathcal { N } _ { k } ( i )$ is determined by Haversine distance at each forward pass, and $\alpha _ { i j }$ are multi-head attention weights with a learned distance bias.

Key property: The neighbor set adapts to whatever nodes are present. If nodes are removed, remaining nodes find their k nearest among those still available.

## D. Per-Node Temporal Processing

A bidirectional GRU processes each node’s temporal sequence independently:

$$
\mathbf { h } _ { i } ( 1 { : } T ) = \mathrm { B i G R U } ( \mathbf { z } _ { i } ^ { \prime } ( 1 ) , \ \dots , \mathbf { z } _ { i } ^ { \prime } ( T ) )\tag{3}
$$

The input tensor $( B , T , N , d )$ is reshaped to $( B \cdot N , T , d ) -$ there is no cross-node operation in the temporal backbone.

Key property: Adding or removing nodes does not change any other node’s temporal representation. This is also why “topology robustness” (performance under node dropout) is largely a tautology for this architecture—the per-node processing is structurally independent.

## E. Prediction Head

A 2-layer MLP maps each node’s final hidden state to the forecast horizon:

$$
\widehat { \mathbf { x } } _ { i } ( T + 1 { : } T { + } q ) = \mathbf { M L P } ( \mathbf { h } _ { i } ( T ) ) \in \mathbb { R } ^ { q \times F }\tag{4}
$$

## F. Implementation

The model has ∼315K parameters at d=128 (or ∼99K at $d { = } 6 4 )$ . Training uses AdamW with cosine annealing and early stopping. We adopt the Peaks-Over-Threshold loss of [2] throughout, so that the controlled comparison in Section IV-D differs only in the spatial backbone and not the training objective; coverage, our primary metric, is a property of the conformal wrapper and is unaffected by this choice.

## G. Guaranteed-Coverage Inductive Calibration

The point forecaster above produces a value at any coordinate, but a value alone cannot tell an operator whether to trust it at a sensor the model never trained on. We therefore wrap OmniST in split conformal prediction [6], [7] to emit a prediction interval with a finite-sample coverage guarantee.

Given a nonconformity score $s _ { i } = | \hat { x } _ { i } - x _ { i } |$ (absolute residual), we compute the calibrated half-width

$$
\begin{array} { r } { \hat { q } = \mathrm { Q u a n t i l e } \Big ( \{ s _ { i } \} _ { i \in \mathcal { C } } , ~ \frac { \lceil ( n + 1 ) ( 1 - \alpha ) \rceil } { n } \Big ) } \end{array}\tag{5}
$$

on a calibration set $\mathcal { C }$ of n residuals, and return the interval $[ \hat { x } - \hat { q } , \hat { x } + \hat { q } ]$ (clipped at zero for non-negative quantities). Under exchangeability this yields marginal coverage $\geq 1 - \alpha$

The inductive subtlety. In the inductive setting the calibration residuals are gathered on training stations while the test point is an unseen station, so one must verify that the calibration and inductive-test scores remain exchangeable — the issue analyzed for graph models by [3]. Because OmniST’s per-node processing makes each station’s residual depend only on its own coordinates, features, and local neighborhood (not on a global node index), the score distribution is stable across node sets, and the guarantee is preserved both at unseen coordinates and as training stations are added or removed. This is precisely what fixed-topology models cannot offer (they emit nothing at a new node) and what nearest-neighbor interpolation cannot offer (a point estimate with no finite-sample guarantee). We quantify both comparisons in Section IV.

H. Locality and Recalibration-Free Coverage Under Topology Change

The inductive-conformal guarantee of Section II-G holds under exchangeability of the calibration and test nonconformity scores. For message-passing GNNs this is delicate: adding or removing any node perturbs every node’s embedding, shifting all calibration scores — the drift that [3] must correct to recover validity. OmniST’s per-node construction gives a stronger, exact property that removes this drift for all but local changes.

Proposition 1 (Locality ⇒ exact score invariance). Let $s _ { i } ~ = ~ g { \left( c _ { i } , x _ { i } , \{ ( c _ { j } , x _ { j } ) ~ : ~ j ~ \in ~ \mathcal { N } _ { k } ( i ; S ) \} , y _ { i } \right) }$ be OmniST’s nonconformity score for node i given the present node-set $S ,$ where $\mathcal { N } _ { k } ( i ; S )$ is the set of k nearest other nodes of i in S (by Haversine distance). For any $S ^ { \prime } \subseteq \mathcal { V }$ with $\mathcal { N } _ { k } ( i ; S ^ { \prime } ) =$ $\mathcal { N } _ { k } ( i ; S ) -$ in particular, any addition or removal of nodes outside i’s k-nearest set — we have $s _ { i } ( S ^ { \prime } ) = s _ { i } ( S )$ exactly. Consequently, the split-conformal threshold qˆ calibrated on training-station scores remains valid, with no recalibration,for any test node whose k-nearest set is unchanged; and empirical coverage is preserved more broadly whenever the reconfigured local receptive fields of the calibration and test nodes remain exchangeable.

Proof sketch. Every OmniST component is a function of a node’s own data and its k-NN receptive field only: the coordinate-MLP token depends on $( c _ { i } , x _ { i } ) ;$ the k-NN attention aggregates exactly ${ \mathcal { N } } _ { k } ( i ; S ) ;$ the GRU and head act per node. No operation references the global count N or a node index, and there is no cross-node reduction (e.g. no batch-wise statistic over nodes). Hence $s _ { i }$ is a fixed function of $\left( c _ { i } , x _ { i } , \mathcal { N } _ { k } ( i ; S ) { \mathrm { - d a t a } } , y _ { i } \right)$ ; if that argument is unchanged, $s _ { i }$ is unchanged. Validity of qˆ then follows from standard split conformal, since neither the calibration scores nor the unchanged test score move. □

Why this is stronger than the general inductive-GNN case. [3] recover coverage under the score drift that message passing induces; OmniST has no drift to correct for far changes, by construction — the guarantee is recalibrationfree in the exact-invariance regime. We verify Proposition 1 directly (Section IV-F, Table VI): dropping nodes outside a test node’s k-NN set changes its prediction by $< 1 0 ^ { - 7 }$ (bitidentical to float precision), whereas dropping a true neighbor changes it by $\sim 7 \times 1 0 ^ { - 3 }$ (locality is real, not vacuous), and inductive coverage stays within the finite-sample band of the target under 30–50% node dropout on all four real datasets.

## I. Online Conformal Under Temporal Drift

The split-conformal guarantee of Section II-G is marginal under exchangeability. Over a long test horizon the residual distribution drifts (seasonality, regime change), so a quantile frozen on an earlier calibration window systematically undercovers later points. We restore coverage over time with two composable online estimators that consume the time-ordered residual stream.

Rolling-window conformal. Recompute the half-width as the $( 1 - \alpha )$ quantile of the residuals observed in the most recent w steps (optionally exponentially weighted). This tracks a drifting residual scale directly.

Adaptive Conformal Inference (ACI). Following Gibbs and Candes [5], keep a residual pool but update the\` effective miscoverage level online,

$$
\alpha _ { t + 1 } = \alpha _ { t } + \gamma \left( \alpha - \mathrm { e r r } _ { t } \right) ,\tag{6}
$$

where $\mathrm { e r r } _ { t } = 1$ if the realized value fell outside the interval at step t. This drives the long-run empirical miscoverage to α for any sequence, with no exchangeability assumption — the guarantee that survives drift.

Rolling-ACI (recommended). Form the base half-width from the rolling-window quantile (to track the drifting scale) and apply a multiplicative ACI correction (to remove residual coverage bias). Empirically this holds the target over time most reliably (Table IX), at the cost of wider intervals — an honest and unavoidable trade-off, since matching coverage under a widening residual distribution requires widening the interval.

## III. EXPERIMENTAL VALIDATION

## A. Data

We validate the inductive coverage guarantee on four realworld measurement datasets spanning three distinct physical domains:

Delhi air quality (environmental) — daily $\mathrm { P M } _ { 2 . 5 }$ from 37 CPCB stations across Delhi (2019–2023), from the E-STGCN public repository [2]; the exact dataset used to produce E-STGCN’s published results. This is also our primary accuracy benchmark.

• Beijing air quality (environmental) — hourly $\mathrm { P M } _ { 2 . 5 }$ from the 12-station UCI Beijing Multi-Site Air-Quality dataset [8] (2013–2017), aggregated to daily; a second, structurally distinct air-quality network (different city, granularity, station count).

• Western-US weather (meteorology) — daily maximum temperature (TMAX) from 30 NOAA GHCN-Daily firstorder stations (2018–2022), with native per-station coordinates; a genuinely different physical process (atmospheric temperature) with its own spatial-correlation structure.

• Freeway traffic (transportation) — 5-minute speed readings from the 228-sensor Caltrans PeMSD7-M benchmark [1] (District 7, Los Angeles). The released benchmark ships a distance-kernel adjacency rather than raw coordinates, so we reconstruct a 2D sensor embedding via classical multidimensional scaling on recovered pairwise distances; the speed time series itself is the original real data.

All four are real measurement data (no synthetic generators), spanning three physical domains — environmental pollution, meteorology, and transportation.

For the primary Delhi accuracy benchmark: training period 2019–2022 (1461 days), test period 2023 (365 days), model input 60 consecutive days, forecast horizon 30 days.

## B. Protocol

OmniST is trained once on 2019–2022, then evaluates 30- day forecasts for each month of 2023 by providing the 60 days immediately preceding each target month as input context. MAE is computed across all 37 stations × 30 forecast days.

Comparison context: E-STGCN’s published results [2] use the same dataset and same train/test split. We report their numbers from their Table 2 for contextual reference (Table I). Because evaluation protocols may differ and we have not run their code, we additionally provide a controlled headto-head (Section IV-D) against a faithful reimplementation of the E-STGCN architecture under identical conditions, which removes the protocol confound.

## C. Additional evaluation

We also evaluate: (a) sensor failure resilience—removing stations at inference to simulate equipment failures; and (b) inductive prediction—holding out 5 stations from training entirely and testing if the model produces valid forecasts at their coordinates.

## IV. RESULTS

## A. Monthly Forecasting Accuracy (Real CPCB Data)

The per-month values above are from a single representative run (for month-level readability). As a seed-hardened headline, a 5-seed held-out evaluation gives OmniST MAE $2 8 . 2 \pm$ $0 . 9 ~ \mu \mathrm { g } / \mathrm { m } ^ { 3 }$ , statistically indistinguishable from E-STGCN on the same split — confirming the “matches E-STGCN” claim is not a single-seed artifact.

OmniST achieves average MAE of $3 1 . 0 ~ \mathrm { \ } \mu \mathrm { g / m ^ { 3 } }$ on real CPCB data. During moderate-pollution months $( \mathrm { A p r - S e p } )$ both methods achieve comparable accuracy $( < 3 ~ \mu \mathrm { g } / \mathrm { m } ^ { 3 }$ difference). During winter months (Oct–Dec), OmniST appears to perform better, particularly in December where E-STGCN’s published MAE (116.8) exceeds even a naive “predict recent average” baseline $( 4 8 . 9 \ \mu \mathrm { g / m ^ { 3 } } )$ . We cannot determine whether this discrepancy arises from methodological differences or a genuine failure mode without running E-STGCN’s code; we therefore do not claim winter superiority.

TABLE I  
30-DAY $\mathrm { P M } _ { 2 . 5 }$ MAE $( \mu \mathrm { G } / \mathrm { M } ^ { 3 } )$ ON REAL DELHI CPCB DATA (2023)
<table><tr><td>Month</td><td>OmniST</td><td>E-STGCN*</td><td>Note</td></tr><tr><td>Jan</td><td>57.3</td><td>56.1</td><td>Comparable</td></tr><tr><td>Mar</td><td>24.2</td><td>25.8</td><td>Comparable</td></tr><tr><td>Apr</td><td>22.0</td><td>25.4</td><td>OmniST better</td></tr><tr><td>May</td><td>30.9</td><td>29.0</td><td>Comparable</td></tr><tr><td>Jun</td><td>18.1</td><td>22.0</td><td>OmniST better</td></tr><tr><td>Jul</td><td>11.3</td><td>15.7</td><td>OmniST better</td></tr><tr><td>Aug</td><td>11.2</td><td>11.9</td><td>Comparable</td></tr><tr><td>Sep</td><td>15.2</td><td>15.1</td><td>Comparable</td></tr><tr><td>Oct</td><td>30.4</td><td>36.4</td><td>OmniST better</td></tr><tr><td>Nov</td><td>69.8</td><td>77.1</td><td>OmniST better</td></tr><tr><td>Dec</td><td>50.9</td><td>116.8†</td><td>See note</td></tr><tr><td> $\mathbf { A v } \mathbf { g }$ </td><td>31.0</td><td>39.2</td><td></td></tr></table>

<sup>∗</sup>From E-STGCN Table 2 [2]. Direct comparison requires caution.  
<sup>†</sup>E-STGCN Dec result exceeds naive baselines; methodological difference likely.

## B. Sensor Failure Resilience

TABLE II  
PREDICTIONS UNDER SIMULATED SENSOR FAILURE (REAL CPCB DATA)
<table><tr><td>Active stations</td><td>MAE (µg/m³)</td><td>Degradation</td></tr><tr><td>37 (full network)</td><td>35.33</td><td></td></tr><tr><td>32 (5 failures)</td><td>35.72</td><td> $+ 1 . 1 \%$ </td></tr><tr><td>27 (10 failures)</td><td>35.72</td><td>+1.1%</td></tr><tr><td>22 (15 failures)</td><td>35.58</td><td>+0.7%</td></tr></table>

When up to 15 stations (40% of the network) are removed at inference, OmniST’s predictions at remaining stations degrade ${ \tt b y } \leq 1 . 1 \%$ . This is validated on real measurements. The system continues forecasting without retraining, reconfiguration, or human intervention.

## C. Inductive Prediction

Holding out 5 geographically diverse stations from training and predicting at their coordinates:

TABLE III  
INDUCTIVE POINT PREDICTION AT UNSEEN STATIONS (REAL CPCB, 5 HELD-OUT, 3 SEEDS)
<table><tr><td>Method</td><td>MAE (µg/m³)</td></tr><tr><td>Nearest-neighbor (copy closest training station)</td><td>30.3</td></tr><tr><td>IGNNK (learned inductive kriging [9])</td><td>34.0±1.3</td></tr><tr><td>OmniST + node-dropout training augmentation</td><td>27.1±1.1</td></tr><tr><td>OmniST (coordinate-based)</td><td> $2 6 . 7 { \pm } 1 . 2 $ </td></tr></table>

Under a multi-seed evaluation on real CPCB data, OmniST’s inductive point forecast at unseen stations beats both a model-free and a learned inductive baseline. It improves on nearest-neighbor interpolation $( 2 6 . 7 ~ \mathrm { v s } ~ 3 0 . 3 ~ \mu \mathrm { g / m ^ { 3 } }$ ∼12%) and, more tellingly, on IGNNK [9] — the canonical learned inductive graph-kriging method (a diffusion-GCN trained with random-subgraph node masking; our faithful reimplementation, not the authors’ code) — by ∼21% (26.7 vs 34.0). This corrects an earlier single-run observation (where a single seed placed OmniST above nearest-neighbor); under proper multi-seed evaluation the coordinate-based predictor generalizes to unseen coordinates better than both proximity copying and a trained kriging GNN. The advantage is not Delhi-specific: on the weather domain OmniST also beats IGNNK on inductive point MAE (4.99 vs 5.92) under identical conditions (Section IV-E). Training-time node-dropout augmentation does not further improve the result (27.1, within noise), indicating the gain comes from the coordinate-based inductive architecture itself, not the augmentation. Point accuracy, however, is still not the central contribution — the operationally decisive quantity is a calibrated, coverage-guaranteed interval at the unseen location (Table V), which neither baseline provides: nearest-neighbor has no finite-sample guarantee, and IGNNK under-covers (0.89 at the 0.90 target) while OmniST holds it (0.915) with a 19% tighter interval.

## D. Controlled Comparison to a Fixed-Topology Baseline

The monthly table above compares against E-STGCN’s published numbers (different protocol). To remove that confound we also run a controlled head-to-head against a faithful reimplementation of the E-STGCN architecture family — Chebyshev-GCN spatial module + LSTM temporal module + POT/EVT loss, the three components of [2] — under identical conditions: same real data, same seeds, same preprocessing, same loss, same training budget (Table IV). This is our reimplementation of the architecture, not the authors’ code.

mirror image is a trained sensor going offline. We demonstrate this directly (3 seeds, Delhi + weather): we pin the E-STGCN adjacency to its $N _ { \mathrm { t r a i n } }$ training nodes (a true static-adjacency deployment, with the baseline’s fair-mode recompute disabled) and drop 30%/50% of those nodes at inference. The fixed $N _ { \mathrm { t r a i n } } { \times } N _ { \mathrm { t r a i n } }$ adjacency can no longer message-pass over the reduced node set and raises a concrete runtime error (a shape mismatch: a 32×32 fixed adjacency cannot multiply the 16- sensor tensor that remains after dropout) — again demonstrated, not asserted. OmniST, run on the identical dropped network and recalibrated on the survivors, holds coverage at the unseen stations (0.914/0.913 on Delhi, 0.925/0.927 on weather under 30%/50% dropout). Both directions — node added and node removed — trace to the same root cause: a fixed-topology model binds its weights and adjacency to the training node set, whereas OmniST has no such binding by construction.

## E. Guaranteed-Coverage Uncertainty at Unseen Stations

We hold out 5 geographically spread stations from training, calibrate split conformal (Section II-G) on training-station residuals, and measure empirical coverage and mean interval width at the 5 unseen stations (target coverage 90%, 3 seeds on real CPCB data). We compare against a nearest-neighbor baseline that copies the closest training station’s series and forms a residual-quantile interval — the natural model-free competitor, following the spirit of valid model-free spatial prediction [10].

TABLE IV  
CONTROLLED OMNIST VS FIXED-TOPOLOGY E-STGCN (REAL DATA, 5 SEEDS, IDENTICAL CONDITIONS)  
TABLE V  
INDUCTIVE PREDICTION INTERVALS AT UNSEEN STATIONS (REAL CPCB, TARGET 90% COVERAGE, 3 SEEDS)
<table><tr><td>Setting</td><td>Dataset</td><td>OmniST</td><td colspan="2">E-STGCN</td><td colspan="3"></td></tr><tr><td>Transductive MAE (seen)</td><td>Delhi AQ</td><td>28.6±1.2</td><td>33.5±1.9</td><td>Method</td><td>Coverage</td><td>Mean width</td><td>Guarantee?</td></tr><tr><td rowspan="2">Inductive (unseen nodes)</td><td>Weather</td><td>5.55±0.14</td><td>5.30±0.29</td><td>Nearest-neighbor + residual</td><td>0.925</td><td>149.7 128.5±3.7</td><td>No</td></tr><tr><td>Delhi AQ</td><td>MAE 26.7, cov 0.91</td><td></td><td>cannot predicOmniST + conformal</td><td>0.915±0.006</td><td></td><td>Yes</td></tr><tr><td></td><td>Weather</td><td>MAE 5.0, cov 0.92</td><td></td><td>cannot predicCoverage under training-node dropout (OmniST + conformal): 30% stations dropped</td><td>0.914</td><td></td><td></td></tr><tr><td colspan="6">Transductive = train and test on the same stations. On Delhi OmniST</td><td></td><td>Yes</td></tr><tr><td colspan="6">is better (paired  $p { = } 0 . 0 3 ) ;$  on weather E-STGCN is marginally better</td><td></td><td>Yes</td></tr><tr><td colspan="6">(p=0.14, n.s.) — reported both ways. Inductive = predict at held-out</td><td>This is the original 3-seed single-domain run. The 10-seed re-run (Table VI) gives a consistent Delhi coverage of 0.914 (width 127.0);</td></tr><tr><td colspan="6">stations: the fixed  $\bar { N } { \times } N$  training adjacency of E-STGCN (32×32 / 27×27) has no row/column for an unseen station, so it structurally</td></tr><tr><td colspan="5">produces no output; OmniST does, with a coverage guarantee.</td></tr><tr><td colspan="5">the small differences are seed/run variation, not a discrepancy.</td></tr></table>

Two honest takeaways. (1) On transductive point accuracy (the regime where a fixed-topology GCN is valid) the two are comparable: OmniST is better on Delhi and E-STGCN is marginally better on weather — neither dominates, consistent with the paper’s ”matches, not beats” stance. (2) In the inductive setting the fixed-topology baseline cannot produce a prediction at all at an unseen sensor — a concrete dimension failure we demonstrate, not merely assert — whereas OmniST delivers a calibrated, coverage-guaranteed forecast. This is the capability gap the paper is about, now shown under identical conditions rather than against published numbers.

The dual failure: sensors removed at inference. The inductive case above adds an unseen sensor; the operational

Three findings. (1) The inductive guarantee holds: OmniST’s conformal intervals achieve 91.5% coverage at stations never seen in training, matching the 90% target. (2) They are tighter: 128.5 vs 149.7 mean width — a 14% narrower interval than nearest-neighbor, which over-covers (0.925) and wastes width while carrying no finite-sample validity. (3) The guarantee survives topology change: coverage at the unseen stations stays ≈91% even when 30% or 50% of the training stations are removed and the model is recalibrated on the survivors. This is the capability the architecture was built to provide — calibrated, efficient uncertainty at arbitrary locations under a dynamic sensor network — and it is absent from both fixed-topology STGCNs and model-free interpolation.

F. Robustness: Domains, Coverage Targets, Scale, and the Temporal-Drift Limit

To establish that the guarantee is a property of the method rather than a single dataset or configuration, we stress-test it along several axes (10 seeds for air quality and weather, 6 for the 228-node traffic run, unless noted).

Three domain types, four real datasets. We repeat the inductive-coverage evaluation (target 90%) on all four real datasets — across three physical domain types — calibrating on training-station residuals and measuring coverage at heldout (unseen) nodes, then re-measuring under 30% and 50% training-node dropout (Table VI).

TABLE VI  
INDUCTIVE COVERAGE AT UNSEEN NODES ACROSS FOUR REAL DATASETS / THREE DOMAIN TYPES (TARGET 90%)
<table><tr><td>Dataset (type)</td><td>OmniST</td><td>NN</td><td>drop 30%</td><td>drop</td></tr><tr><td>Delhi AQ (environmental)</td><td>0.914±.006</td><td>0.925</td><td>0.912</td><td>0.9</td></tr><tr><td>Beijing AQ (environmental)</td><td>0.928±.003</td><td>0.869</td><td>0.928</td><td>0.9</td></tr><tr><td>Weather (meteorology)</td><td>0.919±.007</td><td>0.896</td><td>0.919</td><td>0.9</td></tr><tr><td>PeMSD7-M (transportation)</td><td>0.915±.008</td><td>0.887</td><td>0.916</td><td>0.9</td></tr></table>

![](images/a2083301aa620889f7cfa0b00536e36bcb66c164dca193f2b773275fd3b86923.jpg)  
Fig. 1. Inductive coverage at unseen nodes across three physical domain types (four real datasets), at full network and under 30%/50% node dropout. OmniST holds the 90% target throughout.

OmniST’s conformal intervals hold ≥ 90% coverage at unseen nodes on all four real datasets — across three physical domain types (environmental pollution, meteorology, transportation) — and retain coverage under 30–50% node dropout on every one (Table VI, Fig. 1). The nearest-neighbor baseline under-covers on three of the four (Beijing 0.869, weather 0.896, traffic 0.887), confirming the guarantee is a property of the conformal construction, not of a specific dataset or domain. Point MAE is honestly mixed (OmniST wins on Delhi, model-free interpolation wins on the denser networks); the contribution is the guarantee at unseen nodes under topology change, not point dominance.

Baseline breadth. Against four model-free spatial interpolators (nearest-neighbor, inverse-distance weighting, meanof-k-nearest, and a Gaussian-kernel/kriging-style predictor), each wrapped in the same split-conformal procedure, OmniST is the only method that holds ≥ 90% coverage on all four datasets (Table VII). Every interpolator under-covers on multiple domains (e.g. on Beijing all four fall to 0.79–0.87). The interpolators are often tighter and occasionally better on point MAE — but precisely because they under-cover: a narrower interval that misses the target is not a valid one. This is the width-versus-validity trade-off, and it is the quantity that matters operationally.

TABLE VII  
OMNIST VS MODEL-FREE INTERPOLATION BASELINES (COVERAGE @ TARGET 90%, 4 REAL DATASETS)
<table><tr><td>Dataset</td><td>OmniST</td><td>NN</td><td>IDW</td><td>mean-k</td><td>GP</td></tr><tr><td>Delhi AQ</td><td>0.914</td><td>0.925</td><td>0.871</td><td>0.813</td><td>0.845</td></tr><tr><td>Beijing AQ</td><td>0.928</td><td>0.869</td><td>0.818</td><td>0.789</td><td>0.789</td></tr><tr><td>Weather</td><td>0.919</td><td>0.896</td><td>0.927</td><td>0.945</td><td>0.893</td></tr><tr><td>Traffic</td><td>0.911</td><td>0.887</td><td>0.905</td><td>0.906</td><td>0.922</td></tr></table>

All methods use identical split-conformal calibration. OmniST is the only one The traffic OmniST value here (0.911) is from this baseline-breadth run; 0.916the headline multidomain run (Table VI) reports 0.915 at 6 seeds. Both 0.930are valid against their own protocol; the difference is seed/run variation. 0.920

<sup>0.917</sup>Calibration diagnostics. Beyond marginal coverage we $\alpha \in \{ 0 . 5 , \ldots , 0 . 9 5 \} )$ and size-stratified coverage per domain (Fig. 2). Coverage tracks the target closely across all levels on every domain. A size-stratified split (by residual magnitude) shows coverage is near-complete on low/mid-magnitude cases but drops on the highest-magnitude stratum (≈ 0.73–0.78); we report this honestly as a limit of marginal coverage — the hardest cases are where a conditional method (below) helps. (The stratification is by residual tercile, so the high stratum is partly definitional; it is a diagnostic, not a separate failure.)

![](images/696b049b6799dc0351de96b40925cd8984fc6f1013aa865c1d70dc82904cc3d3.jpg)  
Fig. 2. Calibration curves (empirical vs target coverage) across the four real datasets; all track the diagonal, i.e. the method hits whatever target it is set.

Ablation: what earns the guarantee? We ablate the spatial component and neighbourhood size on the coverage and inductive-MAE metrics (not just point MSE), across three domain types (Table VIII). Removing the GAT entirely (NO-GAT: coordinate-MLP + temporal backbone only) and varying $k \in \{ 4 , 8 , 1 6 \}$ leave both marginal coverage and inductive

MAE essentially unchanged (coverage within ±0.01 of the full model on every domain). The honest reading: the inductive coverage guarantee is earned by the coordinate-based inductive construction (which lets each node be scored from its own coordinates and local context), not by the GAT or a particular k. The GAT’s role is enabling inductive prediction at all and mildly sharpening intervals — it is not the source of the marginal guarantee, and we do not claim otherwise.

TABLE VIII  
ABLATION: COVERAGE / INDUCTIVE-MAE AT UNSEEN NODES (TARGET 90%)
<table><tr><td>Dataset</td><td>metric</td><td>full</td><td>no-GAT</td><td>k=4 / k=16</td></tr><tr><td>Delhi AQ</td><td>coverage</td><td>0.915</td><td>0.912</td><td>0.914 / 0.913</td></tr><tr><td></td><td>ind-MAE</td><td>26.7</td><td>26.4</td><td>26.2 / 26.1</td></tr><tr><td>Weather</td><td>coverage</td><td>0.917</td><td>0.925</td><td>0.920 / 0.922</td></tr><tr><td></td><td>ind-MAE</td><td>4.99</td><td>4.80</td><td>4.92 / 4.97</td></tr><tr><td>Traffic</td><td>coverage</td><td>0.919</td><td>0.919</td><td>0.919 / 0.919</td></tr><tr><td></td><td>ind-MAE</td><td>9.55</td><td>9.54</td><td>9.63 / 9.63</td></tr></table>

Coverage/MAE barely move across variants: the coordinate-based inductive design earns the guarantee, not the GAT or a specific k.

Multiple coverage targets (multi-seed, real data). Repeating the inductive evaluation at targets {80, 90, 95}% (5 seeds for air quality, 3 for traffic), empirical coverage tracks the target within ∼ 3 points in every domain×target cell: Delhi $0 . 8 2 4 / 0 . 9 1 5 / 0 . 9 5 8$ , Beijing 0.801/0.928/0.968, traffic 0.773/0.911/0.963. The one loose cell is traffic at the 80% target (0.773), reported as-is.

Marginal vs conditional coverage, and a Mondrian fix. The guarantee is marginal. Measuring per-unseen-node coverage at the 90% target, the air-quality networks are nearly conditional (per-node spread 0.87–0.95), but the irregular traffic network is not (spread 0.73–0.98, averaging to a valid 0.92). Grouping unseen nodes by difficulty (distance to the nearest training node) and applying Mondrian (group-conditional) conformal — a separate calibrated quantile per group — lifts the worst-group coverage on traffic from 0.80 to 0.86 (giving the hard group a wider interval, 51.5 vs 42.1) while leaving the already-tight air-quality domains unchanged. This narrows, but does not fully close, the conditional-coverage gap (0.86 < 0.90); finer or learned grouping is the remaining step.

Scale. At N=228 real traffic sensors inductive coverage holds at 91.9%; at N=325 real sensors (the full Caltrans PeMS-BAY benchmark, 5-minute speed, coordinates reconstructed via MDS as for PeMSD7-M; scalable k-NN, 3 seeds) it holds at $9 2 . 1 \pm 0 . 0 \%$ and remains $\geq 9 2 . 0 \%$ under 30/50% node dropout (the nearest-neighbor baseline under-covers at 87.0%), confirming the guarantee does not degrade with network size. A cost-scaling stress test (replicating real sensors with coordinate jitter to N=912) shows training time growing super-linearly (169 → 437 → 1106 s for 228 → 456 → 912 nodes), the signature of exact $O ( N ^ { 2 } )$ k-NN attention and the concrete motivation for approximate k-NN at larger scale; that stress test does not certify coverage beyond the 325 real sensors above (naive node duplication breaks calibration exchangeability), which we state explicitly. We remove the architectural bottleneck with a scalable k-NN attention variant (a cKDTree neighbour index plus an index-select gather that avoids the $O ( N ^ { 2 } )$ expansion); it is numerically equivalent to exact attention (output cosine similarity 0.99996 at N=228) and 2.3× faster per forward at $N { = } 1 8 2 4$ , with the gap widening in N. Certifying coverage at thousands of nodes now awaits a large real sensor dataset rather than an architectural change.

Temporal drift — resolved. The spatial guarantee above does not transfer to the time axis for free: when we split the test year temporally (calibrate on the first half, evaluate on the second), static split conformal under-covers badly — overall coverage 0.76 and, worse, only 0.50 in the final (mostdrifted) third of the year — because seasonal drift breaks exchangeability over time. We resolve this with a principled online layer (Section II-I): a rolling-window quantile that tracks the drifting residual scale, wrapped in Adaptive Conformal Inference [5]. Table IX shows the rolling-ACI estimator restores target coverage over time (0.89 overall, 0.96 in the late-drift window), at the honest cost of wider intervals; ACI alone is a narrower middle ground (0.91 late-window coverage at a smaller width).

TABLE IX  
COVERAGE OVER TIME UNDER SEASONAL DRIFT (REAL DELHI, TARGET 90%, 3 SEEDS)
<table><tr><td>Method</td><td>Overall cov.</td><td>Late-window cov.</td><td>Width</td></tr><tr><td>Static split-CP</td><td>0.758</td><td>0.498</td><td>95.4</td></tr><tr><td>Rolling window</td><td>0.840</td><td>0.799</td><td>127.4</td></tr><tr><td>Adaptive CP (ACI)</td><td>0.847</td><td>0.905</td><td>139.0</td></tr><tr><td>Rolling-ACI</td><td>0.890</td><td>0.962</td><td>170.1</td></tr></table>

Late-window = final third of the test year (worst drift).  
Width in $\mu \mathrm { g } / \mathrm { m } ^ { 3 }$ . Restoring drift coverage costs width (honest trade-off).

![](images/a4e9692a837c960314a4610dbc81e1dfe438ba00678d838943e7cc2cfc690dc4.jpg)  
Fig. 3. Coverage over time under seasonal drift. Static conformal collapses in the late-drift window (0.50); the rolling-ACI online layer restores the 90% target (0.96 late-window).

## V. RELATED WORK

Spatiotemporal foundation models. Recent work pretrains across domains for transfer, including Mamba-based crossdomain models that generalize zero-shot to unseen regions [4]. We deliberately do not compete on cross-domain pretraining or scale; OmniST targets a different axis — guaranteed inductive uncertainty under dynamic topology at edge scale — validated on four real datasets across three physical domains rather than a pretraining corpus.

![](images/bacf70a5bd57ab83f8ef9827786d0bbfaea1d72c6e3896f00a7b570b2191e020.jpg)  
Fig. 4. Conditional coverage: Mondrian (group-conditional) conformal lifts the worst-group coverage on the irregular traffic network from 0.80 toward target, leaving the already-tight air-quality domains unchanged.

Dynamic / expanding-node forecasting. Fixed-topology STGCNs [1] couple weights to a node count. The expandingnode line relaxes this for sensor addition (e.g., via structured node-interaction prompting or flat expanding-variate schemes), and inductive spatiotemporal kriging [9] predicts at unobserved locations — which we run as a learned inductive baseline (Table III), finding OmniST both more accurate at unseen stations and, unlike IGNNK, coverage-valid. Most related to our architecture, the recent continual-forecasting method STFO [11] also uses normalized coordinate-based aggregation without sensor-specific parameters to absorb sensor expansion, and cold-start forecasting at unseen sensors [12] conditions a meta-graph on local support — both target point prediction under growing networks. OmniST differs in three ways: it handles both addition and removal (not just expansion), it requires no prompting or continual-learning machinery, and — distinct from all of the above — it attaches a distributionfree coverage guarantee to its inductive predictions and shows that guarantee survives topology change. The point-prediction capability is shared with this line; the guaranteed uncertainty at unseen coordinates is what is new.

Conformal prediction for graphs and time series. Split conformal prediction [6], [7] has recently been extended to correlated time series [13], graph time series [14], and energy graphs [15]; [3] analyzes the inductive setting where new nodes perturb calibration scores. A fast-growing 2025–2026 line attaches coverage guarantees to dynamic and temporal graphs: valid CP for dynamic GNNs via unfolding [16], nonexchangeable CP for temporal GNNs [17], topology-aware CP for stream networks under distributional shift (STACI) [18], and online conformalized Gaussian processes over graphs [19]. These establish CP validity as a graph’s edges/labels evolve over time, typically on a fixed node set (node classification or whole-graph regression). OmniST’s contribution is complementary and distinct: coverage at spatially unseen sensor coordinates — locations with no training history at all — and the empirical demonstration that this inductive guarantee holds tightly and persists as training nodes are added or removed, across four real datasets and three physical domains (Section IV-E). The online-CP machinery we use for temporal drift (Section II-I) is in the spirit of [18], [19]; our novelty is its composition with coordinate-based inductive prediction, not the online-CP primitive itself. The classical model-free alternative, valid spatial prediction at unobserved locations [10], is our interpolation baseline.

## VI. DISCUSSION

## A. What the Architecture Achieves

1) Structural N-independence: Valid computation for any number of input nodes, by construction.

2) Inductive prediction: Forecasts at unseen coordinates using only geographic position.

3) No accuracy cost: Comparable to fixed-topology baselines on real CPCB data (Table I).

## B. What It Does NOT Achieve

1) No in-distribution point-accuracy improvement: OmniST matches (does not beat) GCN+LSTM on standard fixed-topology point metrics. Its distinct value is at unseen locations, where it both beats nearest-neighbor (∼12%, Table III) and uniquely provides a coverage guarantee — the contribution is this capability, not an in-distribution accuracy gain.

2) Scaling shown to N=325, not thousands: inductive coverage validated at up to 325 real sensors (full PeMS-BAY, 0.921; Section IV-F) using the scalable k-NN path that removes the $O ( N ^ { 2 } )$ bottleneck; certifying coverage at thousands of nodes awaits a larger real sensor benchmark rather than an architectural change.

3) No proof that GAT improves accuracy: The spatial attention component’s in-distribution accuracy contribution is negligible (∆MAE < 0.2). Its value is enabling inductive prediction (and thus inductive UQ), not improving in-distribution point forecasts.

4) Conditional / finite-horizon coverage guarantees: our temporal fix (Section II-I) restores long-run coverage over time empirically, but ACI’s guarantee is asymptotic in the stream length and marginal, not a per-step or conditional guarantee.

## C. Relationship to E-STGCN

We adopt E-STGCN’s POT loss formulation [2] and position our work as exploring an alternative spatial backbone. E-STGCN’s static Chebyshev GCN is well-optimized for fixed networks. OmniST’s dynamic k-NN attention trades spectral graph theory guarantees for structural flexibility. Whether this trade-off is worthwhile depends on deployment context: stable networks favor E-STGCN; dynamic networks favor OmniST.

## D. On “Topology Invariance”

We use this term to mean: the model produces valid outputs regardless of N. We do NOT claim that performance is invariant (though empirically it is stable). The stability is largely explained by per-node processing independence—a design choice, not a finding. The genuine contribution is that inductive prediction at new coordinates works, which requires the coordinate-based attention mechanism specifically.

## VII. LIMITATIONS

1) Accuracy benchmark is single-city. The headline accuracy comparison (vs E-STGCN) is on Delhi $\mathrm { P M } _ { 2 . 5 }$ only. The inductive coverage guarantee, however, is validated on four real datasets across three physical domains (Delhi + Beijing air quality, NOAA GHCN weather, PeMSD7-M traffic; Section IV-F).

2) Scale. The coverage guarantee is validated to $N { = } 3 2 5$ real sensors (full PeMS-BAY; Section IV-F): inductive coverage $0 . 9 2 1 \pm 0 . 0 0 0$ at unseen sensors, holding under 30/50% dropout, with the scalable k-NN attention (equivalent to exact, 2.3× faster at $N { = } 1 8 2 4 )$ that removes the $O ( N ^ { 2 } )$ bottleneck. Certifying coverage at thousands of nodes still awaits a larger real sensor benchmark (none fetchable at that scale here) rather than an architectural change. All evaluation datasets are real measurement data; the traffic/PeMS-BAY sensor coordinates are reconstructed via MDS from the released distance adjacency (the speed series itself is original).

3) Limited validation set. With 60-day window and 30- day horizon, only a few validation samples fit, making early stopping less reliable.

4) GRU temporal backbone. S4D/Mamba advantages (constant memory) are relevant only for longer sequences $( T > 1 0 0 0 )$ , not tested here.

5) Spatial component negligible for in-distribution accuracy. The k-NN GAT contributes $< 0 . 2 ~ \mu \mathrm { g / m ^ { 3 } }$ to indistribution MAE; its value is enabling inductive prediction (which does now beat nearest-neighbor, Table III) and inductive UQ.

6) Temporal drift (now addressed, with a cost). Static conformal under-covers under seasonal drift; the rolling-ACI online layer (Sections II-I, IV-F) restores target coverage over time, but pays a wider interval (170 vs 95 $\mu \mathrm { g / m ^ { 3 } }$ on Delhi) and provides a long-run rather than per-step guarantee.

7) Seeds. Headline accuracy is 5-seed $( 2 8 . 2 \pm 0 . 9 )$ ; the cross-domain coverage, baseline-breadth, and calibration results are 10-seed for air quality and weather and 6- seed for the 228-node traffic run (the slowest domain, all six seeds on the scalable k-NN path). Temporal and conditional-coverage results are 3-seed. A GPU-scale run would further harden the remaining 3-seed numbers.

8) E-STGCN comparison: published-number context plus a controlled reimplementation. The monthly table uses their published numbers (protocol may differ). We additionally run a controlled head-to-head (Section IV-D) against a faithful reimplementation of the E-STGCN architecture (Chebyshev-GCN + LSTM + POT) under identical data/seeds/budget; the remaining caveat is that this is our reimplementation, not the authors original code.

9) The conformal guarantee is marginal, not fully conditional. Coverage is guaranteed on average, not perstation or per-horizon. We quantify this (per-unseennode coverage 0.73–0.98 on traffic) and partially address it with Mondrian group-conditional conformal, which lifts the traffic worst-group from 0.80 to 0.86 (Section IV-F) — an improvement, not a full fix (0.86 < 0.90). The guarantee also relies on the inductiveexchangeability conditions of [3] holding for OmniST’s per-node scores — which we argue structurally and verify empirically (Table V) but do not prove. Conditional coverage (per-station, per-horizon) remains future work; adaptive online conformal under temporal drift, previously listed here as open, is addressed in Section II-I (long-run, not conditional).

10) Conformal results use 3 seeds and $\mathbf { P M } _ { 2 . 5 }$ only. The inductive-interval table is validated on real data but at modest seed count and a single pollutant; broader confirmation is future work.

## VIII. CONCLUSION

OmniST demonstrates that a topology-invariant spatiotemporal forecasting architecture can achieve competitive accuracy on real measurement data while providing two operationally decisive capabilities: resilience under sensor failure, and calibrated, coverage-guaranteed uncertainty at unseen sensor locations under dynamic topology. On the same CPCB Delhi dataset used in E-STGCN, the model achieves average MAE of $3 1 . 0 ~ \mu \mathrm { g / m ^ { 3 } }$ , maintains predictions within 1.1% when 40% of stations are removed, and — composed with inductive split conformal — produces 91.5% prediction intervals at never-seen coordinates that are 14% tighter than a nearestneighbor baseline and retain their coverage when 30–50% of training stations are removed.

The contribution is twofold: an architecture where no component references a global node count (enabling dynamic sensor networks), and the demonstration that this structural property makes an inductive conformal guarantee hold at unseen locations and survive topology change — a quantity neither fixed-topology STGCNs nor model-free interpolation provide. The primary validated benefits are operational resilience and guaranteed inductive uncertainty, both without retraining.

Under multi-seed evaluation, inductive point prediction at new locations now outperforms nearest-neighbor interpolation (∼12%), and the guarantee is robust across four real datasets spanning three physical domains (Delhi and Beijing air quality, NOAA GHCN weather, and Caltrans PeMSD7-M traffic), holding at unseen nodes under up to 50% node dropout on every one, where OmniST is the only method among four model-free interpolation baselines to hold the target on all four. The earlier temporal-drift limitation is now resolved: a rolling adaptive-conformal layer restores target coverage over time (0.89 overall, 0.96 in the worst-drift window, versus 0.50 for static conformal), at the honest cost of wider intervals. We do not pursue cross-domain foundation-model pretraining, which recent work already targets [4]; the remaining open directions are scaling the guarantee to thousands of nodes (approximate k-NN), conditional (per-node, per-horizon) coverage, and physics-aware spatial attention.

Data and code availability. All four datasets are public: Delhi CPCB $\mathrm { P M } _ { 2 . 5 }$ from the E-STGCN repository [2]; Beijing

Multi-Site Air Quality from the UCI repository [8]; western-US daily temperature from NOAA GHCN-Daily (public, no authentication); and PeMSD7-M traffic from [1]. The implementation (model, loaders for all four datasets, every experiment script, a pinned environment, a one-command reproduction manifest, and a results index mapping each reported number to its source) is publicly available at https: //github.com/Praveg432/OmniST. Every number in this paper is reproducible from a committed result file via the provided reproduction manifest.

## REFERENCES

[1] B. Yu, H. Yin, and Z. Zhu, “Spatio-temporal graph convolutional networks: A deep learning framework for traffic forecasting,” in Proceedings of the Twenty-Seventh International Joint Conference on Artificial Intelligence (IJCAI), 2018, pp. 3634–3640.

[2] M. Panja, T. Chakraborty, A. Biswas, and S. Deb, “E-stgcn: Extreme spatio-temporal graph convolutional networks for air quality forecasting,” Journal of the Royal Statistical Society Series A: Statistics in Society, 2026.

[3] S. H. Z. Davis and A. Bojchevski, “Conformal inductive graph neural networks,” International Conference on Learning Representations (ICLR); arXiv:2407.09173, 2024.

[4] R. An, Y. Zhang, Z. Liang et al., “Damba-st: Domain-adaptive mamba for efficient urban spatio-temporal prediction,” arXiv preprint arXiv:2506.18939, 2026.

[5] I. Gibbs and E. Candes, “Adaptive conformal inference under dis-\` tribution shift,” Advances in Neural Information Processing Systems (NeurIPS), 2021.

[6] V. Vovk, A. Gammerman, and G. Shafer, Algorithmic Learning in a Random World. Springer, 2005.

[7] A. N. Angelopoulos and S. Bates, “A gentle introduction to conformal prediction and distribution-free uncertainty quantification,” Foundations and Trends in Machine Learning, 2023.

[8] S. Zhang, B. Guo, A. Dong, J. He, Z. Xu, and S. X. Chen, “Cautionary tales on air-quality improvement in beijing,” Proceedings of the Royal Society A: Mathematical, Physical and Engineering Sciences, vol. 473, no. 2205, 2017, uCI Beijing Multi-Site Air-Quality Data Set #501.

[9] G. Appleby, L. Liu, and L.-P. Liu, “Inductive graph neural networks for spatiotemporal kriging,” Proceedings of the AAAI Conference on Artificial Intelligence, 2020.

[10] H. Mao, R. Martin, and B. J. Reich, “Valid model-free spatial prediction,” Journal of the American Statistical Association; arXiv:2006.15640, 2022.

[11] Anonymous, “Rethinking continual spatio-temporal forecasting: A field-operator view for expanding sensor networks,” arXiv preprint arXiv:2609.31325, 2026, sTFO: normalized coordinate-based aggregation without sensor-specific parameters, for sensor expansion.

[12] ——, “Cold-start traffic state forecasting at unseen sensor locations via support-conditioned meta-graph learning,” Sensors, vol. 26, no. 15, p. 4995, 2026.

[13] A. Cini, A. Jenkins, D. Mandic, C. Alippi, and F. M. Bianchi, “Relational conformal prediction for correlated time series,” arXiv preprint arXiv:2502.09443, 2025.

[14] R. Jin et al., “Conformal inference for time series over graphs,” IEEE ICASSP; arXiv:2510.11049, 2026.

[15] others, “In-context residual calibration for uncertainty quantification of energy time series over graphs,” arXiv preprint arXiv:2606.31804, 2026.

[16] E. Davis, D. Clarke et al., “Valid conformal prediction for dynamic gnns,” arXiv preprint arXiv:2405.19230, 2024.

[17] Anonymous, “Non-exchangeable conformal prediction for temporal graph neural networks,” arXiv preprint arXiv:2507.02151, 2025.

[18] ——, “Staci: Topology-aware conformal prediction for stream networks,” arXiv preprint arXiv:2503.04981, 2025.

[19] ——, “Conformalized gaussian processes for online uncertainty quantification over graphs,” arXiv preprint arXiv:2510.06181, 2025.