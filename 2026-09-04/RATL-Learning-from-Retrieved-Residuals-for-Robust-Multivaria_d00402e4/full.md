# RATL: Learning from Retrieved Residuals for Robust Multivariate Time-Series Forecasting

Yuchen He<sup>1</sup>, Yueyang Cang<sup>1</sup>, Zhiyuan Ning<sup>1</sup>,

Ningyu Wang<sup>2∗</sup>, Li Shi<sup>1∗</sup>

<sup>1</sup>Department of Automation, Tsinghua University

<sup>2</sup>State Key Laboratory of Hydroscience and Engineering, Tsinghua University heyuchen25@mails.tsinghua.edu.cn, cangyy23@mails.tsinghua.edu.cn, ningzy25@mails.tsinghua.edu.cn wang-ningyu@mail.tsinghua.edu.cn, shilits@tsinghua.edu.cn

## Abstract

Retrieval-augmented generation (RAG) complements parametric models with retrieved external evidence. The same idea is attractive for continuous-output regression, but directly reusing retrieved target values is often not robust when samples difer in output level, numerical scale, or local dynamics. Moreover, conventional forecasting pipelines generally use residuals for model optimization and error diagnosis, but do not retain individual historical residual examples as memory that can be accessed at inference time. For multivariate time-series forecasting, we propose RATL, a plug-in residualretrieval and feedback-correction method. RATL freezes a base forecaster to construct retrieval keys and turns its historical forecast residuals into a train-only memory specific to that base model. At inference time, RATL retrieves residual trajectories from similar historical contexts subject to causal availability constraints, then uses a set-aware router operating over forecast blocks and variables to select and combine these trajectories. Experiments show that historical residuals matched to the current context contain reusable forecasting information and that RATL improves frozen base forecasters in most experimental settings. Ablations further show that learned routing strengthens raw residual feedback, while validation-based correction-strength selection limits residual over-injection. On real-world benchmarks, we use iTransformer as the primary frozen base forecaster, compare against multiple strong forecasting baselines, and test transferability across backbones. The results show that RATL can further improve base-forecaster performance in most settings. Overall, RATL shifts the retrieved object from historical target values to base-model-specific historical forecast errors, providing a plug-in, residual-memory-based paradigm for learned feedback correction in continuous-output forecasting.

## Introduction

Long-horizon multivariate forecasting maps a local context window to a future trajectory. Modern linear, MLP, and Transformer forecasters have improved this mapping substantially (Zeng et al. 2023; Nie et al. 2023; Liu et al. 2024), yet their prediction is still conditioned on a fixed lookback and on historical patterns compressed into learned parameters. When a relevant pattern occurred far outside the lookback, a parametric forecaster cannot explicitly inspect that occurrence at inference time.

Retrieval-augmented forecasting addresses this limitation by searching a historical datastore and incorporating the futures associated with similar contexts (Zhang et al. 2025; Han et al. 2025; Du, Han, and Guo 2026). However, raw future reuse creates two coupled risks. First, similar contexts need not share the same future level or scale. Second, an imperfect retrieved signal may be injected even when the base prediction is already accurate, causing negative transfer. Both issues become pronounced in multivariate, long-horizon settings, where the correction must vary across variables and forecast blocks.

We investigate a diferent retrieval target: the forecast error ofafixed base model. For each training window, we store the base-model residual rather than treating the observed future as a stand-alone prediction, yielding an output-space correction. RATL retrieves only historical residuals whose target windows are temporally available from the training memory. In its main configuration, RATL uses the frozen base forecaster’s per-variable hidden representations as retrieval keys. J5, a set-aware learned residual router, constructs block– variable candidate tokens from the query, candidate residual blocks, a Direct residual reference, and positional features. Set attention assigns weights to retrieved residual candidates and an explicit zero-residual candidate, while a global correction-strength hyperparameter γ ∈ [0, 1] is selected on validation. As a control, the non-parametric Direct baseline uses only retrieval similarity as weights for a raw weighted correction over the same residual candidates, with γ = 1 and no separate correction-strength tuning. Figure 1 presents the overall design: the central pipeline consists of frozen-base forecasting, causal residual retrieval, block–variable routing, and correction-strength selection, while four side panels expand its key operations.

Experiments show that RATL can turn context-matched historical errors into useful corrections across diferent forecasting settings and improves overall forecasting performance over the Direct baseline.

Our contributions are:

• We formulate retrieval-augmented forecasting as basespecific residual retrieval, with a train-only memory and an explicit temporal admissibility rule that prevents target leakage.

• We develop RATL around J5, a set-aware, block-variable residual router with oracle imitation and a no-correction candidate; similarity-weighted Direct serves as the nonparametric retrieval baseline.

<table><tr><td>Method</td><td>Retrieved value</td><td>Integration</td></tr><tr><td>kNN-MTS</td><td>future segment</td><td>similarity aggregation</td></tr><tr><td>RAFT</td><td>future patches</td><td>input/feature augmentation</td></tr><tr><td>PFRP</td><td>global prediction</td><td>confidence/output gates</td></tr><tr><td>RATL</td><td></td><td>frozen-base residual raw Direct / J5 + val. strength</td></tr></table>

Table 1: Retrieval target and integration difer across related forecasters.

• We conduct a 156-run study covering 13 datasets and 52 settings, together with audited transfer studies on frozen DLinear, PatchTST, TimesNet, and TimeMixer checkpoints. RATL improves mean MSE by 9.57% in the main study. The transfer results show that its gains vary substantially with the base-forecaster architecture, dataset, and retrieval-key choice.

## Related Work

Multivariate forecasting. Informer reduces attention cost for long sequences (Zhou et al. 2021); Autoformer and FEDformer introduce decomposition and frequency-aware modeling (Wu et al. 2021; Zhou et al. 2022); TimesNet models temporal variation in two dimensions (Wu et al. 2023); and TimeMixer decomposes multiscale patterns through mixing blocks (Wang et al. 2024). More recent strong baselines include the decomposition-based DLinear (Zeng et al. 2023), channel-independent PatchTST (Nie et al. 2023), and iTransformer, which represents variables as tokens (Liu et al. 2024). RATL does not replace these architectures: it is trained after a base forecaster and operates in prediction space. Our main experiments use iTransformer; separate frozen audits attach the same correction interface to DLinear, PatchTST, TimesNet, and TimeMixer checkpoints.

Retrieval-augmented forecasting. Retrieval augmentation has been efective in language modeling by exposing non-parametric memory at inference time (Lewis et al. 2020; Khandelwal et al. 2020). In time series, kNN-MTS retrieves future segments using learned multivariate representations (Zhang et al. 2025); RAFT retrieves matching historical patches at multiple periods (Han et al. 2025); and PFRP combines global retrieved predictions with a local univariate model using confidence and output gates (Du, Han, and Guo 2026). Foundation-model variants concatenate or prompt with retrieved contexts (Tire et al. 2026). RATL difers in the memory value and in the object being combined: it stores the residual of a specific frozen forecaster, then learns to select and compose an additive correction for multivariate blocks. This makes the retrieved value a model-failure template rather than a stand-alone future.

Table 1 summarizes the diferences between RATL and representative retrieval-augmented forecasting methods along the retrieval-object and fusion dimensions.

Residual modeling and negative transfer. Residual structure has long been exploited by hybrid statistical–neural models and boosting (Zhang 2003; Friedman 2001). ResMem fits a nearest-neighbor regressor to a base model’s training residuals (Yang et al. 2023), while similarity-based macroeconomic forecasting corrects ARIMA predictions with errors from related historical periods (Guerrón-Quintana and Zhong 2023). MSCT-RCM constructs a KNN residualsequence library for ultra-short photovoltaic forecasting (Ye et al. 2026), whereas δ-Adapter learns a bounded postprocessing correction for frozen forecasters without retrieving historical residuals (Liang et al. 2026). RATL extends the residual-reuse principle to multivariate long-horizon trajectories, causal train-only retrieval, and block–variable routing, and uses a zero-residual candidate and a correctionstrength hyperparameter to suppress noise in retrieved historical residuals and reduce forecast error.

## Method

## Problem Setup and Frozen Base

Let $\mathbf { X } _ { t } \in \mathbb { R } ^ { L \times D }$ be a lookback window ending at forecast origin t and $\mathbf { Y } _ { t } \in \mathbb { R } ^ { H \times D }$ its future. A base forecaster $f _ { \theta }$ produces

$$
\widehat { \mathbf { Y } } _ { t } ^ { 0 } = f _ { \theta } ( \mathbf { X } _ { t } ) , \qquad \mathbf { R } _ { t } = \mathbf { Y } _ { t } - \widehat { \mathbf { Y } } _ { t } ^ { 0 } .\tag{1}
$$

We first train $f _ { \theta }$ conventionally and then freeze it. RATL learns only to estimate a correction $\widehat { \mathbf { R } } _ { t } \mathbf { : }$

$$
\widehat { \mathbf { Y } } _ { t } = \widehat { \mathbf { Y } } _ { t } ^ { 0 } + \gamma \widehat { \mathbf { R } } _ { t } , \qquad 0 \leq \gamma \leq 1 .\tag{2}
$$

This separation lets the same residual-retrieval interface wrap diferent forecasters and makes every memory value interpretable as a historical error of the deployed base.

## Train-Only Residual Memory

For each training window $i ,$ we store

$$
\begin{array} { r } { \mathcal { M } _ { i } = ( \mathbf { k } _ { i } , \mathbf { X } _ { i } , \mathbf { R } _ { i } , a _ { i } ) , \qquad \mathbf { k } _ { i } = \phi _ { \theta } ( \mathbf { X } _ { i } ) , } \end{array}\tag{3}
$$

where ϕ<sub>θ</sub> is a frozen base representation and $a _ { i } = t _ { i } + H$ is the time at which the full target used to compute $\mathbf { R } _ { i }$ becomes available. In the main configuration, $\phi _ { \theta }$ is the iTransformer encoder’s variable-token representation. The frozen transfer protocols separately use a backbone-agnostic input-statistics key for DLinear/PatchTST and preregistered native hidden keys for PatchTST, TimesNet, and TimeMixer; no key is selected using test results. Search uses squared Euclidean similarity independently for variable tokens.

Why retrieve residuals? A retrieved future can be decomposed as $\mathbf { Y } _ { i } = f _ { \theta } ( \mathbf { X } _ { i } ) + \mathbf { R } _ { i }$ . Reusing $\mathbf { Y } _ { i }$ asks retrieval to transfer both the neighbor’s forecastable level and its model error. RATL retains only $\mathbf { R } _ { i } { \mathrm { : } }$ the current base remains responsible for level, trend, and cross-variable structure, while memory estimates the systematic component that the same model missed in a related context. The memory is therefore base-specific. Replacing $f _ { \theta }$ requires rebuilding residual values, but not redesigning the retrieval/correction interface.

## Per-Variable Top-K Search with Retrieval Keys

Given a current input window $\mathbf { X } _ { t }$ , we compute the query key $\mathbf { k } _ { t } = \phi _ { \theta } ( \mathbf { X } _ { t } )$ using the same frozen mapping used to build the memory. For a per-variable key, $\mathbf { k } _ { t , d } \in \mathbb { R } ^ { P }$ is the query representation of variable d over the full lookback window. Retrieval is performed independently for each variable, rather than separately for future forecast blocks.

Retrieval first applies a temporal-availability constraint. At forecast origin t, candidate i may participate in the search only if

$$
a _ { i } \leq t - H .\tag{4}
$$

Because $a _ { i } = t _ { i } + H$ is the time at which the candidate target becomes fully available, this constraint inserts an additiona gap oflength H after the candidate target, preventing retrieval of memory entries that are too close to the current forecast or whose future information is not yet available.

In the frozen main protocol, keys receive no additional normalization, and retrieval uses the per-variable mean squared Euclidean distance

$$
d _ { t , i , d } = \frac { 1 } { P } \left\| \mathbf { k } _ { t , d } - \mathbf { k } _ { i , d } \right\| _ { 2 } ^ { 2 } , \qquad s _ { t , i , d } = - d _ { t , i , d } .\tag{5}
$$

A smaller distance gives a larger retrieval score $s _ { t , i , d } .$ For each variable, we select only the K highest-scoring entries among the temporally admissible candidates:

$$
\begin{array} { r } { \mathcal { N } _ { t , d } = \mathrm { T o p K } _ { i : a _ { i } \le t - H } \left( s _ { t , i , d } \right) . } \end{array}\tag{6}
$$

Retrieval depends only on the current query key, historical memory keys, and the temporal-availability constraint; it does not use the current ground-truth future or current true residual. After the search, we take the full residual trajectories $\{ \mathbf { R } _ { i , : , d } \} _ { i \in \mathcal { N } _ { t , d } }$ associated with these K historical windows. Direct combines these trajectories directly according to their retrieval scores, whereas J5 learns candidate weights within the same candidate set for each time block and variable. Thus, Top-K search operates on representations of the full input window, and forecast blocking occurs only after retrieval.

## Direct Similarity-Weighted Baseline

The Direct corrector converts retrieval scores $s _ { t , i , d }$ into weights

$$
\pi _ { t , i , d } = \frac { \exp ( { s _ { t , i , d } / { \tau _ { s } } } ) } { \sum _ { j \in \mathcal { N } _ { t , d } } \exp ( { s _ { t , j , d } / { \tau _ { s } } } ) }\tag{7}
$$

and averages the corresponding residual trajectories:

$$
\widehat { \mathbf { R } } _ { t , : , d } ^ { \mathrm { d i r } } = \sum _ { i \in \mathcal { N } _ { t , d } } \pi _ { t , i , d } \mathbf { R } _ { i , : , d } .\tag{8}
$$

Direct is parameter-free after the base is trained and serves as the non-parametric control for RATL. It tests whether learned routing improves over simply averaging the same retrieved candidates and residual values by context similarity. Direct is evaluated as this raw correction with $\gamma = 1$ and is not separately tuned for correction strength.

## Block-Residual Candidates and the Soft-Oracle Teacher

The retrieval module returns K historical residual trajectories for the current query. Using one candidate weight for an entire trajectory is too coarse because the same historical residual may be useful early in the forecast but fail later; selecting at every time point is more susceptible to local noise and would create $H \times D$ groups of fine-grained decisions. To balance flexibility and stability, we partition the forecast horizon into consecutive temporal blocks of length $B _ { h } = 8$ Let $B _ { b }$ denote the forecast positions in block b, and let the number of blocks be $G = \mathsf { \bar { \Gamma } } [ H / B _ { h } ]$ . J5 assigns candidateresidual weights separately for every temporal block b and variable $d ,$ allowing one candidate to correct only part of the forecast interval or only some variables.

During training, the target $\mathbf { Y } _ { t }$ is known, so the true residual of the frozen base forecaster on the current query can be computed as

$$
\mathbf { R } _ { t } = \mathbf { Y } _ { t } - \widehat { \mathbf { Y } } _ { t } ^ { 0 } .
$$

For each retrieved candidate $i \in \{ 1 , \ldots , K \}$ , we compare its historical residual $\mathbf { R } _ { i }$ with the current true residual $\mathbf { R } _ { t }$ at block–variable position $( b , d )$ . We also define candidate $i \ = \ 0$ as the zero residual, $\mathbf { R } _ { 0 , b , d } = \mathbf { 0 }$ , representing no correction. The candidate error is

$$
e _ { t , i , b , d } = \frac { 1 } { | \mathcal { B } _ { b } | } \sum _ { h \in \mathcal { B } _ { b } } \left( R _ { i , h , d } - R _ { t , h , d } \right) ^ { 2 } .\tag{9}
$$

A smaller error means that candidate i more closely matches the correction that the base forecaster actually needs for that time block and variable. We therefore construct the soft Oracle teacher distribution

$$
q _ { t , i , b , d } ^ { * } = \mathrm { s o f t m a x } _ { i } ( - e _ { t , i , b , d } / \tau _ { o } ) ,\tag{10}
$$

where $\tau _ { o }$ controls the smoothness of the teacher distribution. Compared with a hard label that selects only the minimumerror candidate, the soft distribution can retain probability mass on several similarly efective candidates and reduce supervision instability caused by small fluctuations in candidate error. This Oracle is used only to construct the supervision target during training; the true future is unavailable at validation and test time, so $q ^ { * }$ is neither computed nor used then.

Figure 2 illustrates the “trajectory blocking–candidate comparison–soft teacher distribution” construction for a single variable.

## RATL Set-Aware Residual Router

Direct assumes that context similarity directly represents residual utility. J5 instead predicts the block–variable Oracle distribution above, without observing the current true residual, from information available at inference time. For candidate i, temporal block b, and variable d, the router constructs a candidate token from the current query, candidate residual block, Direct residual reference, and block/variable positional features. The general router also supports query– neighbor window relations and similarity features; the frozen main variant disables similarity input/prior and masks the neighbor-window, window-diference, residual-mean, and Direct-residual-energy channels specified by its featureablation mask. Its active token can be written

![](images/0d7ea0f938f041d74ff4f30c6cf57d7a1afb88563d0d2843d8978be1dcaa79f9.jpg)  
Figure 1: Overview of RATL. The center shows the end-to-end pipeline from an input window to the final corrected forecast; (a) a frozen base forecaster constructs a train-only key–residual memory; (b) Top-K residuals are retrieved independently for each variable under the temporal-availability constraint; (c) candidate trajectories are split into block–variable tokens and augmented with a zero-residual candidate; (d) J5 predicts candidate weights through set attention. Direct similarity weighting is the non-parametric baseline and uses the raw correction with $\gamma = 1$ ; for RATL, the correction-strength hyperparameter γ is selected on validation before one-shot test evaluation.

$$
\mathbf { z } _ { t , i , b , d } = g \Big ( \mathbf { X } _ { t } , \mathbf { R } _ { i , b , d } , \widehat { \mathbf { R } } _ { t , b , d } ^ { \mathrm { d i r } } , \mathbf { e } _ { b } , \mathbf { e } _ { d } , \mathbf { m } _ { t , i , b , d } ^ { \mathrm { a c t i v e } } \Big ) ,\tag{11}
$$

where $g$ contains shared encoders, $\mathbf { e } _ { b } , \mathbf { e } _ { d }$ are learned block and variable embeddings, and $\mathbf { m } ^ { \mathrm { a c t i v e } }$ denotes the remaining scalar residual features. A zero-residual token is appended as candidate $i = 0$ . Set attention exchanges information among the $K + 1$ candidates, followed by a block-variable scoring head:

$$
\begin{array} { r } { \pmb { \alpha } _ { t , b , d } = \mathrm { s o f t m a x } \left( \mathrm { J } 5 ( \{ \mathbf { z } _ { t , i , b , d } \} _ { i = 0 } ^ { K } ) \right) . } \end{array}\tag{12}
$$

The resulting correction is

$$
\widehat { \mathbf { R } } _ { t , b , d } ^ { \mathrm { J 5 } } = s \sum _ { i = 1 } ^ { K } \alpha _ { t , i , b , d } \mathbf { R } _ { i , b , d } ;\tag{13}
$$

weight $\alpha _ { t , 0 , b , d }$ abstains by assigning mass to zero correction and s is a learned global residual scale initialized to one. Candidate order is randomly permuted during training, enforcing set rather than rank semantics.

## Oracle Imitation and Prediction Loss

J5 is trained with two complementary objectives. The first is the MSE of the final prediction, which directly constrains whether the combined residual improves the base forecast. The second is the Oracle-imitation loss, which requires the router’s candidate weights $\alpha _ { t , b , d }$ to approximate the block– variable teacher distribution $\mathbf { q } _ { t , b , d } ^ { * } \colon$

$$
\begin{array} { r } { \mathcal { L } = \mathrm { M S E } ( \widehat { \mathbf { Y } } _ { t } , \mathbf { Y } _ { t } ) + \lambda _ { \mathrm { t e a c h } } \mathrm { C E } ( q _ { t } ^ { * } , \pmb { \alpha } _ { t } ) , } \end{array}\tag{14}
$$

In the frozen main configuration, the prediction-loss weight is 1; the global teacher-loss scale is 2.0 and the local weight of the block–variable Oracle cross-entropy is 0.2, so the effective coeficient in Eq. (14) is $\lambda _ { \mathrm { t e a c h } } = 2 . 0 \times 0 . 2 = 0 . 4$ The prediction loss provides an end-to-end output constraint, while the Oracle cross-entropy provides fine-grained supervision about which historical residual is useful, when, and for which variable. At inference time, J5 uses only the current query, frozen base-model prediction, and train-only memory to produce α; it requires neither the true future nor the Oracle teacher.

## Correction-Strength Selection and Sealed Evaluation

The learned correction can over-inject residuals. For J5/RATL, we therefore treat $\gamma$ as a scalar forecasting hyperparameter and select it by

$$
\gamma ^ { * } = \arg \operatorname* { m i n } _ { \gamma \in \{ 0 , 1 , \dots , 1 \} } \mathrm { M S E } _ { \mathrm { v a l } } \left( \widehat { \mathbf { Y } } ^ { 0 } + \gamma \widehat { \mathbf { R } } , \mathbf { Y } \right) ,\tag{15}
$$

breaking ties toward the smaller γ. The test set is evaluated only at $\gamma ^ { * }$ . Equation (15) applies to J5/RATL, not Direct: Direct is the raw parameter-free similarity-weighted correction fixed at $\gamma = 1$ and is not separately γ-tuned. In the results, RATL denotes J5 with validation-selected $\gamma .$ This is ordinary validation-based hyperparameter selection, not predictive-uncertainty calibration or a safety guarantee.

![](images/5c935ef8e1f7f604589decf7a1e6bb8abcfa9f260e065fd086e47d94660ae089.jpg)

![](images/b53aafe48cf508a01bcc7c08e6df413bbbeceade2613bbdef7ba2bbd984457eb.jpg)

![](images/a63473a5b9141ac107554063c8f536da6d7ba4da8977900751e08d45169d6dc9.jpg)  
J5 learns α<sub>i, b, d</sub> from inference-time features using CE(q <sup>\*</sup> , α).  
Figure 2: Construction of block-residual candidates and the soft Oracle supervision target for a fixed variable $d .$ Full residual trajectories are first divided into consecutive temporal blocks along the forecast horizon. In each block, every historical candidate and the zero-residual candidate $R _ { 0 } = 0$ are compared with the current true residual to obtain the block–variable error $e _ { i , b , d } .$ A temperature-scaled softmax over negative errors then gives the teacher distribution $q _ { i , b , d } ^ { * } .$ Curves and values in the figure illustrate the procedure and are not experimental results.

## Experiments

## Setup

Datasets and metrics. We evaluate 13 public multivariate benchmarks following the provenance convention of iTransformer (Liu et al. 2024): ETTh1, ETTh2, ETTm1, and ETTm2 from the ETT benchmark (Zhou et al. 2021); ECL, Exchange, Trafic, and Weather as used by Autoformer (Wu et al. 2021); Solar-Energy from LSTNet (Lai et al. 2018); and PEMS03/04/07/08 as evaluated by SCINet (Liu et al. 2022). Long-term datasets use horizons {96, 192, 336, 720}; PEMS uses {12, 24, 36, 48}, yielding 52 cells. Lookback is 96 and stride is one. ETT follows its oficial split, the other longterm datasets use chronological 70/10/20 splits, and PEMS uses 60/20/20. Standardization is fit on training data only. We report MSE and MAE, mean and sample standard deviation over seeds {1, 2, 3}.

Models and selection. The main-experiment base mode is iTransformer (Liu et al. 2024). Base-model and J5 checkpoints are selected by minimum validation MSE and always use matched seeds. The frozen RATL configuration uses $K = 6 4$ , forecast-horizon blocks of length eight, no similarity feature or similarity prior in J5, and the pruned feature mask described above. Full per-cell configurations, precision tiers, and commands are included in the supplement.

## Overall Forecasting Accuracy

Across all 52 cells, RATL reduces MSE by 9.57% and MAE by 6.21% on average. Relative to the matched base model, it records 48 wins, 2 ties, and 2 losses; 48 cells improve for every seed, and 47 cells have a positive seed-level 95% confidence interval. Gains are largest on PEMS (dataset averages of 20.56–24.41%), but remain positive on ETT, Weather, ECL, Solar, and Trafic. The 2 losses are Exchange-336 (−2.66%) and Exchange-720 (−8.19%).

Table 2 summarizes the base-model and RATL MSE over every dataset and forecast length in the iTransformer main experiment.

Table 3 directly compares Direct and J5 at a fixed correction scale $( \gamma = 1 )$ . J5 improves the raw aggregate gain by 1.07 percentage points, supporting the role of learned candidate interaction. The dataset-level breakdown in the supplement further shows that raw Direct is strong on ECL, Trafic, Solar, and PEMS, but degrades results on ETTh2, ETTm2, Exchange, and Weather.

Correction strength is a consequential hyperparameter. For RATL, the selected γ is often below one on ETT and Weather, while high-gain Trafic and PEMS cells frequently retain γ = 1. Thus, γ is a dataset–horizon–seed-dependent hyperparameter rather than a universal damping constant. Because $\gamma \in [ 0 , 1 ]$ , RATL can revert to the original base forecaster when validation rejects the correction.

## Comparison with Strong Forecasters

To summarize RATL’s transfer behavior across base forecasters, Table 4 collects the iTransformer main panel and the frozen DLinear, PatchTST, TimesNet, and TimeMixer transfer panels. We freeze the best-performing base model and rebuild a train-only residual memory for that exact model. Each row reports the cell-level macro-average gain of RATL relative to its matched frozen base model, rather than comparing absolute forecasting accuracy across backbones. Because the panels difer in coverage and preregistered retrieval keys, this table is intended to show cross-architecture compatibility and its boundaries, not to provide a strict backbone ranking. Complete per-cell absolute metrics and comparisons with independently trained strong forecasters are provided in the supplement.

Long-term forecasting
<table><tr><td rowspan="2">Dataset</td><td colspan="2">H=96</td><td colspan="2">H=192</td><td colspan="2">H=336</td><td colspan="2">H=720</td></tr><tr><td>Base</td><td>RATL</td><td>Base</td><td>RATL</td><td>Base</td><td>RATL</td><td>Base</td><td>RATL</td></tr><tr><td>ETTh1</td><td> $. 3 8 7 { \pm } . 0 0 0$ </td><td>.377±.001</td><td> $. 4 3 9 { \pm } . 0 0 1$ </td><td>.429±.002</td><td>.480±.001</td><td>.466±.001</td><td> $. 4 9 1 { \pm } . 0 0 1$ </td><td> $\mathbf { \delta } \mathbf { \delta } \mathbf { \mathcal { A } } 7 \mathbf { 0 } \pm . \mathbf { 0 } \mathbf { 0 } 3$ </td></tr><tr><td>ETTh2</td><td>.304±.004 .297±.003</td><td></td><td> $. 3 8 0 { \pm } . 0 0 1$ </td><td>.373±.001</td><td>.421±.005</td><td>.414±.004</td><td>.422±.001</td><td> $\mathbf { \sigma } . 4 2 2 { \pm } . 0 0 1$ </td></tr><tr><td>ETTm1</td><td>.349±.001 .327±.001</td><td></td><td></td><td>.384±.001.362±.001.420±.001</td><td></td><td>1 .399±.002</td><td>.481±.002</td><td> $\mathbf { \delta } \mathbf { . 4 6 3 \pm . 0 0 3 }$ </td></tr><tr><td>ETTm2</td><td> $. 1 8 5 { \pm } . 0 0 0$ </td><td> $\mathbf { . 1 7 8 { \pm } . 0 0 0 }$ </td><td> $. 2 5 2 { \pm } . 0 0 1$ </td><td>.247±.002 .316±.002</td><td></td><td> $\mathbf { 3 0 9 } \pm . \mathbf { 0 0 } 2$ </td><td> $. 4 1 3 { \pm } . 0 0 1$ </td><td> $\mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \mathcal { A } } \mathbf { 0 } \mathbf { 9 } \pm \mathbf { . 0 } \mathbf { 0 } 2$ </td></tr><tr><td>Exchange</td><td></td><td></td><td>.088±.001 .088±.001 .179±.000 .178±.000</td><td></td><td>.329±.002</td><td>.338±.007</td><td>.859±.001</td><td> $. 9 2 9 { \pm } . 0 4 2$ </td></tr><tr><td>Weather</td><td></td><td></td><td></td><td></td><td>.176±.002 .165±.002 .226±.002 .216±.001 .282±.001 .274±.001</td><td></td><td>.358±.001</td><td> $. 3 5 3 { \pm } . 0 0 2$ </td></tr><tr><td>ECL</td><td> $. 1 4 8 { \pm } . 0 0 1$ </td><td>.134±.000.161±.001 .150±.001</td><td></td><td></td><td>.176±.003.163±.001</td><td></td><td>.214±.004</td><td> $\mathbf { \delta } \mathbf { \cdot } \mathbf { 1 9 2 \pm . 0 0 3 }$ </td></tr><tr><td>Solar</td><td> $. 2 0 7 { \pm } . 0 0 2$ </td><td> $\mathbf { \nabla } _ { \mathbf { \cdot } } 2 \mathbf { 0 0 } { \pm } . \mathbf { 0 0 1 }$ </td><td> $. 2 4 2 { \pm } . 0 0 2$ </td><td>.228±.001</td><td>.255±.002 .234±.002</td><td></td><td> $. 2 5 3 { \pm } . 0 0 0$ </td><td> $. 2 3 2 \pm . 0 0 2$ </td></tr><tr><td>Traffic</td><td> $. 4 0 0 { \pm } . 0 0 1 $ </td><td> $. 3 7 7 { \pm } . 0 0 1$ </td><td> $. 4 1 8 { \pm } . 0 0 0$ </td><td>.398±.001.433±.000.412±.001</td><td></td><td></td><td> $. 4 6 6 { \pm } . 0 0 0$ </td><td> $\mathbf { \nabla } . 4 4 4 \pm . 0 0 0$ </td></tr></table>

<table><tr><td colspan="9">PEMS traffic forecasting</td></tr><tr><td></td><td colspan="2">H=12</td><td colspan="2">H=24</td><td colspan="2">H=36</td><td colspan="2">H=48</td></tr><tr><td>Dataset</td><td>Base</td><td>RATL</td><td>Base</td><td>RATL</td><td>Base</td><td>RATL</td><td>Base</td><td>RATL</td></tr><tr><td>PEMS03</td><td> $. 0 6 7 { \scriptstyle \pm . 0 0 1 }$ </td><td>.060±.000 .096±.002 .078±.000 .132±.002 .099±.001 .164±.002 .119±.001</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>PEMS04</td><td> $. 0 8 1 { \pm } . 0 0 0$ </td><td>.068±.000</td><td> $. 1 0 1 { \pm } . 0 0 1$ </td><td>.078±.000</td><td> $. 1 1 8 { \pm } . 0 0 1$ </td><td>.090±.001 .134±.002</td><td></td><td> $\mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \delta \mathbf { \delta } \mathbf { \delta } \delta \mathbf { \delta } \delta \mathbf { \delta } \delta \mathbf { \delta } \delta \mathbf { \delta } \delta \mathbf { \delta } \delta \mathbf { \delta } \delta \mathbf { \delta } \delta \mathbf { \delta \delta } \mathbf { \delta \delta \delta \delta \mathbf } \delta \mathbf { \delta \delta } \delta \mathbf \delta \delta \mathbf  \delta \delta \delta \delta \delta \delta \mathbf \delta \delta \delta \delta \delta \mathbf \delta \delta \delta \delta \delta \delta \delta \mathbf \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta$ </td></tr><tr><td>PEMS07</td><td> $. 0 6 3 { \pm } . 0 0 2$ </td><td> $\mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \delta \mathbf { \delta } \mathbf { \delta } \delta \mathbf { \delta } \delta \mathbf { \delta } \mathbf { \delta \delta } \mathbf { \delta \delta \delta \delta \mathbf } \delta \mathbf { } \delta \delta \mathbf { \delta \delta \delta \delta \mathbf } \delta \delta \mathbf \delta \delta \mathbf  \delta \delta \delta \delta \delta \mathbf \delta \delta \delta \delta \delta \delta \delta \delta \mathbf \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta$ </td><td> $. 0 8 5 { \pm } . 0 0 4$ </td><td>.065±.002</td><td> $. 1 0 4 { \pm } . 0 0 4$ </td><td> ${ \bf . 0 7 4 } \pm . { \bf 0 0 } 2$ </td><td> $1 2 1 { \pm } . 0 0 2$ </td><td> $\mathbf { 0 8 3 \pm . 0 0 1 }$ </td></tr><tr><td>PEMS08</td><td> $. 0 8 4 \pm . 0 0 4$ </td><td> $\mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } ( \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } ( \delta \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta \delta } \mathbf { \delta \delta } \mathbf { \delta \delta } \bf \delta \delta \delta \mathbf { } \delta \delta \delta \mathbf { \delta } \delta \delta \bf \delta \delta \delta \mathbf { } \delta \delta \delta \delta \bf \delta \delta \delta \delta \bf \delta \delta \delta \delta \delta \delta \delta \delta \bf \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta $ </td><td> $. 1 2 3 { \pm } . 0 0 5$ </td><td> ${ \bf . 0 9 6 { \pm } . 0 0 1 }$ </td><td> $. 1 8 8 { \pm } . 0 0 6$ </td><td> ${ \bf . 1 2 9 } \pm . 0 0 3$ </td><td> $. 2 1 2 { \pm } . 0 0 6$ </td><td> ${ \bf 1 4 9 } \pm { \bf . 0 0 5 }$ </td></tr></table>

Table 2: Three-seed MSE (mean±std). RATL denotes the frozen J5 variant with validation-selected γ. Lower is better.

<table><tr><td>Correction</td><td>Mean gain</td><td>95% CI</td></tr><tr><td>Direct</td><td>7.19%</td><td>[3.84, 10.61]</td></tr><tr><td>J5</td><td>8.26%</td><td>[5.18, 11.39]</td></tr><tr><td>J5 – Direct</td><td>+1.07 pp</td><td>[0.57, 1.62]</td></tr></table>

Table 3: Aggregate relative MSE gain over the base at fixed correction scale $( \gamma = 1 )$ across 52 dataset–horizon cells. Confidence intervals are paired cell bootstraps.

multiscale hidden states of the final PDM and then averages them across scales. Every key is predefined before testing and uses the same per-variable L2 Top-K retrieval and trainonly residual memory. Consequently, diferences in the table reflect both the base forecaster and its preregistered key and cannot be attributed to a retrieval representation alone.

The retrieval keys used by the diferent backbones are defined as follows. The iTransformer main experiment uses the per-variable tokens from the final encoder output. The unified DLinear and PatchTST transfer panels in the table use a 12-dimensional input-statistics key consisting of the window’s last value, mean, standard deviation, end-to-start diference, and eight-segment average pooling, which keeps the retrieval rule consistent across architectures. PatchTST is also evaluated separately with a native hidden key obtained by temporal pooling of its final encoder patch tokens. TimesNet uses the forecast-horizon hidden states of the final TimesBlock together with the output-head weights to construct per-variable keys. TimeMixer first temporally pools the

The macro-average gain is positive for every backbone in Table 4, but individual cells can degrade, and the variation in gain shows that transfer depends on the base architecture, dataset, and predefined retrieval key. The TimesNet and TimeMixer hidden-key panels establish compatibility for those evaluated backbone–dataset–key combinations, but they contain no same-backbone input-key arm and therefore do not compare keys or establish hidden-key superiority. PatchTST is additionally evaluated with a native hidden key in a separate matched audit. These results support compatibility under the evaluated settings, not guaranteed improvement. Complete per-cell results, protocol-specific horizons, and integrity audits are provided in the supplement.

The supplement also provides source-labeled, published dataset-average comparisons, which show the same broad pattern: RATL is competitive on ETT and Weather, strong on ECL, Solar, Trafic, and PEMS, and weak on Exchange.

## Robustness, Significance, and Protocol Audit

Paired test-window bootstrap on representative cells yields positive MSE-gain intervals: [3.95, 4.31]% for ETTm1- 336, [4.89, 5.11]% for Trafic-336, and [15.03, 16.02]% for PEMS08-24. In contrast, the intervals for Exchange-336 and Exchange-720 are strictly negative. Analysis of the top K candidates shows that ETTm1 improves as K increases from 16 to 128; Trafic remains stable around K ∈ {16, 32, 64}; and none of the tested K values repairs Exchange-720. The Exchange failures therefore cannot be explained by a single candidate-pool setting.

<table><tr><td>Backbone</td><td>Retrieval key</td><td>Scope</td><td>Cells</td><td>MSE gain</td><td>MAE gain</td></tr><tr><td>iTransformer</td><td>native hidden (encoder)</td><td> $1 3 { \mathrm { ~ d a t a s e t s } } \times 4 H$ </td><td>52</td><td>9.57%</td><td>6.21%</td></tr><tr><td>DLinear</td><td>input statistics</td><td> $9 \mathrm { d a t a s e t s } \times 1 H$ </td><td>9</td><td>5.83%</td><td>5.75%</td></tr><tr><td>PatchTST</td><td>input statistics</td><td> $9 \mathrm { d a t a s e t s } \times 1 H$ </td><td>9</td><td>2.78%</td><td>1.20%</td></tr><tr><td>TimesNet</td><td>native hidden (head)</td><td> $9 \mathrm { d a t a s e t s } \times 1 H$ </td><td>9</td><td>1.42%</td><td>0.48%</td></tr><tr><td>TimeMixer</td><td>native hidden (multiscale)</td><td> $9 \mathrm { d a t a s e t s } \times 1 H$ </td><td>9</td><td>0.58%</td><td>0.40%</td></tr></table>

Table 4: Cross-backbone summary. Each row reports the macro-average cell gain of RATL over its matched frozen base. The panels use preregistered retrieval keys and diferent coverage, so the table summarizes transfer scope rather than ranking backbones. DLinear/PatchTST use $H = \mathrm { 3 3 6 }$ except Exchange at $H = 9 6 ;$ TimesNet/TimeMixer use the same dataset–horizon scope.

Unlike trafic, energy, and weather data with stable physical periodicity, we consider Exchange a relatively small financial series that is strongly afected by exogenous shocks and exhibits clear regime changes. Its exchange-rate levels, volatility, and cross-variable relationships may change over time, while stable daily or weekly periodicity is relatively weak. Thus, even when two input windows are similar in historical shape or backbone representation, their subsequent exchange-rate changes and base-model errors need not be similar. The RATL assumption that similar contexts have transferable residual patterns is more likely to fail on this dataset, causing historical residuals retrieved at test time to be mismatched in direction or magnitude. Long horizons also reduce the number of efective windows available for validation selection, further increasing uncertainty in correctionstrength estimation.

Because retrieved residuals may not provide reliable corrections in settings with weak periodicity or strong distribution shift, RATL includes two predefined fallback-to-base mechanisms. Locally, J5 adds an explicit zero-residual candidate at every block–variable position, allowing the router to reduce or cancel the residual correction at that position. Globally, the validation candidate set includes $\gamma = 0$ , so the final output can revert completely to the frozen base forecaster when validation evidence does not support a nonzero correction. These mechanisms reduce the risk of negative transfer but do not provide a non-degradation or safety guarantee.

## Conclusion and Future Work

We propose RATL, which causally retrieves historical residuals from a frozen base forecaster and uses a block–variable router to select and combine candidate residuals. The iTransformer main experiment shows that model-specific historical errors can serve as reusable inference-time memory; the DLinear, PatchTST, TimesNet, and TimeMixer transfer panels further provide evidence that the interface is compatible with multiple architectures in the evaluated settings.

RATL’s gains vary with the base forecaster, dataset, and retrieval-key definition. For datasets with limited data, weak periodicity, and more frequent changes driven by exogenous factors, historical residuals from similar contexts need not transfer to the test stage. The zero-residual candidate and $\gamma = 0$ fallback can mitigate negative transfer but provide no non-degradation or safety guarantee. At the same time, exact retrieval and long multivariate residual memories incur growing computational and storage costs as the numbers of training windows and variables and the forecast length increase. Semantic representations of exogenous factors could be incorporated into retrieval-key vectors.

Future work will focus on uncertainty-aware abstention based on candidate disagreement and distribution drift, sample-adaptive correction strength, and validationsafe retrieval-key selection. It will also explore approximate nearest-neighbor search, residual quantization, prototype memories, and dynamic memory compression to reduce deployment costs. Further studies should test the crossregime stability of residual patterns across broader backbones and real operating environments, and assess applicability to transportation, energy, and industrial forecasting together with domain constraints, online monitoring, and conservative fallback strategies.

## References

Du, D.; Han, T.; and Guo, S. 2026. Predicting the Future by Retrieving the Past. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 40, 20896–20904.

Friedman, J. H. 2001. Greedy Function Approximation: A Gradient Boosting Machine. The Annals ofStatistics, 29(5): 1189–1232.

Guerrón-Quintana, P.; and Zhong, M. 2023. Macroeconomic Forecasting in Times of Crises. Journal of Applied Econometrics, 38(3): 295–320.

Han, S.; Lee, S.; Cha, M.; Arik, S. O.; and Yoon, J. 2025. Retrieval Augmented Time Series Forecasting. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, 21774–21797. PMLR.

Khandelwal, U.; Levy, O.; Jurafsky, D.; Zettlemoyer, L.; and Lewis, M. 2020. Generalization through Memorization: Nearest Neighbor Language Models. In International Conference on Learning Representations.

Lai, G.; Chang, W.-C.; Yang, Y.; and Liu, H. 2018. Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks. In The 41st International ACM SIGIR Conference on Research and Development in Information Retrieval, 95– 104.

Lewis, P.; Perez, E.; Piktus, A.; Petroni, F.; Karpukhin, V.;Goyal, N.; Küttler, H.; Lewis, M.; Yih, W.-t.; Rocktäschel, T.;

Riedel, S.; and Kiela, D. 2020. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. In Advances in Neural Information Processing Systems, volume 33, 9459– 9474.

Liang, D.; Li, Q.; Wang, Y.; Chen, J.; Zhang, H.; Cui, X.; Wang, Q.; and Li, S. 2026. The Forecast After the Forecast: A Post-Processing Shift in Time Series. In The Fourteenth International Conference on Learning Representations.

Liu, M.; Zeng, A.; Chen, M.; Xu, Z.; Lai, Q.; Ma, L.; and Xu, Q. 2022. SCINet: Time Series Modeling and Forecasting with Sample Convolution and Interaction. In Advances in Neural Information Processing Systems, volume 35, 5816– 5828.

Liu, Y.; Hu, T.; Zhang, H.; Wu, H.; Wang, S.; Ma, L.; and Long, M. 2024. iTransformer: Inverted Transformers Are Efective for Time Series Forecasting. In International Conference on Learning Representations.

Nie, Y.; Nguyen, N. H.; Sinthong, P.; and Kalagnanam, J. 2023. A Time Series Is Worth 64 Words: Long-Term Forecasting with Transformers. In International Conference on Learning Representations.

Tire, K.; Taga, E. O.; Ildiz, M. E.; and Oymak, S. 2026. Retrieval Augmented Time Series Forecasting. In International Conference on Artificial Intelligence and Statistics.

Wang, S.; Wu, H.; Shi, X.; Hu, T.; Luo, H.; Ma, L.; Zhang, J. Y.; and Zhou, J. 2024. TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting. In International Conference on Learning Representations.

Wu, H.; Hu, T.; Liu, Y.; Zhou, H.; Wang, J.; and Long, M. 2023. TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis. In International Conference on Learning Representations.

Wu, H.; Xu, J.; Wang, J.; and Long, M. 2021. Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting. In Advances in Neural Information Processing Systems, volume 34, 22419–22430.

Yang, Z.; Lukasik, M.; Nagarajan, V.; Li, Z.; Rawat, A. S.; Zaheer, M.; Menon, A. K.; and Kumar, S. 2023. ResMem: Learn What You Can and Memorize the Rest. In Advances in Neural Information Processing Systems, volume 36, 60768– 60790.

Ye, X.; Yin, J.; Zhang, J.; Li, A.; Liu, Z.; Chen, B.; Yang, J.; Li, S.; and Li, H. 2026. A Multi-Scale CNN-Transformer Network with Residual Correction for Ultra-Short-Term Photovoltaic Power Forecasting. Processes, 14(5): 759.

Zeng, A.; Chen, M.; Zhang, L.; and Xu, Q. 2023. Are Transformers Efective for Time Series Forecasting? In Proceedings of the AAAI Conference on Artificial Intelligence, volume 37, 11121–11128.

Zhang, G. P. 2003. Time Series Forecasting Using a Hybrid ARIMA and Neural Network Model. Neurocomputing, 50: 159–175.

Zhang, H.; Nie, P.; Sun, L.; and Boulet, B. 2025. Nearest Neighbor Multivariate Time Series Forecasting. IEEE Transactions on Neural Networks and Learning Systems, 36(7): 12606–12618.

Zhou, H.; Zhang, S.; Peng, J.; Zhang, S.; Li, J.; Xiong, H.; and Zhang, W. 2021. Informer: Beyond Eficient Transformer for Long Sequence Time-Series Forecasting. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 35, 11106–11115.

Zhou, T.; Ma, Z.; Wen, Q.; Wang, X.; Sun, L.; and Jin, R. 2022. FEDformer: Frequency Enhanced Decomposed Transformer for Long-Term Series Forecasting. In Proceedings of the 39th International Conference on Machine Learning, volume 162 of Proceedings of Machine Learning Research, 27268–27286. PMLR.