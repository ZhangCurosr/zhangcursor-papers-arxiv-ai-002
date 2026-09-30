# Loss-Guided Pretraining Data Selection for Time-Series Foundation Models

Yike Li Shaoxu Song<sup>\*</sup> Jianmin Wang

Tsinghua University

liyike25@mails.tsinghua.edu.cn sxsong@tsinghua.edu.cn jimwang@tsinghua.edu.cn

## Abstract

Time series foundation models (TSFMs) are pretrained on heterogeneous collections containing billions of observations, yet their training windows are typically sampled without estimating whether they provide useful learning signal. We introduce a static data-selection framework that scores each window with a reference forecaster and retains an intermediate interval within every source dataset. Specifically, we connect forecasting loss to optimization difficulty by showing that normalized squared loss controls the per-sample gradient norm under a local Jacobian condition. We then define a reference loss score and apply dataset-stratified selection to preserve the diversity of samples. Across various TSFM architectures, our method outperforms random selection by an absolute margin and even improves both relative MASE and CRPS over full-data pretraining by retaining fewer candidate pretraining windows. Further analyses show strong cross-scale and cross-architecture score correlations, indicating that a small reference model can often select data for larger targets, provided that the reference and target share compatible difficulty orderings.

## 1 Introduction

Time-series foundation models (TSFMs) aim to transfer temporal structure learned from large, heterogeneous corpora to unseen forecasting tasks. Recent systems such as TimesFM, Chronos, and Moirai demonstrate that a single pretrained model can forecast across domains, sampling frequencies, context lengths, and prediction horizons without task-specific fitting (Das et al., 2024; Ansari et al., 2024; Woo et al., 2024). This universal forecasting paradigm shifts a major part of model development from downstream architecture design to corpus construction. As pretraining collections grow, however, the common assumption that every available window should receive equal attention becomes increasingly costly and scientifically questionable.

Time series samples are constructed by sliding context-future windows over longer sequences as illustrated in Figure 1 and neighboring windows can share part of their observations. Many windows are simple repetitions of seasonal or slowly varying patterns and add little influence once those regularities have been learned as shown in Figure 1(a). At the opposite extreme, a context can be almost constant while its future contains an abrupt, unannounced change. Such a window produces a large loss but supplies no stable relation for a forecaster to learn as in Figure 1(b). Between these extremes are nontrivial yet learnable win dows that contain trends, changing seasonality, and moderate stochastic variation, illustrated by Figure 1(c). Therefore, the central question of this paper is how to identify useful training subset from a multi-source corpus that can match or exceed the performance of training on the full dataset.

![](images/c3b95a0320b128f2b6e879afd870591b7a5996430579e0a594b6b0d9d50c59fe.jpg)

![](images/394a83142298a720d30049663f86895ab01628890b9f4856178c5b009161bc2b.jpg)  
(a) Easy | RFL 0.89

![](images/124d126b26207da38a89cef9f6b260d5808acba522388a67b5bb7c4dcb34b058.jpg)  
(b) Hard | RFL 275.42

![](images/7f5e2869cffd6a3f12d5de3c23df1310930b006d07599089b9c6786298add930.jpg)  
(c) Moderate | RFL 16.59  
Figure 1: Sliding context-future windows over a long sequence and three real candidate windows. RFL is the reference loss of Chronos-Bolt on each original candidate window.

Loss-based data pruning methods have already been adopted in computer vision and language models. In computer vision, importance-sampling methods prioritize samples using loss-derived bounds on per sample gradient magnitude, while EL2N ranks images by their prediction-error norm as a tractable proxy for gradient-based importance (Katharopoulos and Fleuret, 2018; Paul et al., 2021). In LLM pretraining, a frozen reference model assigns each document a token-level cross-entropy or perplexity score and these scores are then used to rank and filter the corpus (Marion et al., 2023). A forecasting time series window has the same basic structure as a language sentence. Nonetheless, text perplexity cannot be directly transferred to time series, which have continuous and scale-dependent targets, may contain missing future observations, and are optimized by TSFMs toward heterogeneous objectives such as squared error or quantile loss. Existing time-series selection methods focus mainly on task-specific training or foundation-model fine-tuning (Taga et al., 2025; Wu et al., 2026) and are not applicable to large-scale data due to high complexity, leaving data selection for TSFM pretraining largely unexplored.

We propose a loss-guided data selection framework for time series foundation model pretraining. A frozen reference forecaster assigns each candidate context–future window a reference loss score as in Figure 2 (2). The sample loss controls gradient norm under a local Jacobian condition, so loss is a principled proxy for how strongly a window can affect an update. Then dataset-stratified selection strategy retains an intermediate RFL interval within every source as in Figure 2 (3). Specifically, we convert RFL values into percentile ranks within each source dataset. For a retention ratio α, keep the central α fraction of every source. It requires one offline reference-model pass, stores only one scalar per window, and does not compute gradients or modify target training online. In conclusion, RFL identifies a useful difficulty region, whereas stratification preserves source coverage. We evaluate our method by training architecturally different TSFMs from scratch and achieving state-of-the-art results in the experiments.

Our contributions are as follows:

• To the best of our knowledge, we present the first systematic study of loss-based data selection at the window level for pretraining time series foundation models from scratch. This setting selects context–future windows once before target model optimization.

• We formalize reference loss (RFL) as a difficulty score computed by a frozen forecaster and clarify that normalized MSE controls the gradient magnitude under an explicit local-Jacobian condition. We also introduce dataset-stratified selection, which retains an intermediate RFL interval independently within each source dataset and therefore preserves source coverage and dataset-level diversity.

• We train architecturally distinct TSFM families from scratch and our method outperforms random selection by an absolute margin, and even surpasses training on the full sample set. Further studies of reference models and selection strategies characterize when reference scores transfer and validate the components of the framework.

## 2 Related Work

## 2.1 Time-series foundation models

Time-series foundation models (TSFMs) replace the conventional one-model-per-dataset workflow with a single forecaster pretrained on heterogeneous time series and transferred to unseen tasks. Representative models differ substantially in how they represent continuous observations and parameterize forecasts. TimesFM applies a decoder-only Transformer to patched continuous inputs (Das et al., 2024). Chronos scales and quantizes observations into a discrete vocabulary and trains a language-model backbone with cross-entropy (Ansari et al., 2024) and Moirai combines masked-encoder pretraining with an any-variate architecture and flexible probabilistic outputs (Woo et al., 2024). Timer-XL (Liu et al., 2025a) formulates univariate, multivariate, and covariate-informed forecasting as multivariate next-token prediction over long contexts. Time-MoE scales decoder-only forecasting through sparse mixture-of-experts layers and large multi-domain corpora (Shi et al., 2025). Sundial uses a flow-matching objective to generate flexible contin uous predictive distributions without a prespecified parametric family (Liu et al., 2025b). The primary focus of these models is architecture and data scaling and we focus on determining which context–future windows within a heterogeneous corpus should be used for pretraining.

## 2.2 Data selection for large language models

LLM pretraining has motivated data selection at both the document and domain levels. At the document level, large-scale pipelines combine heuristic or learned quality filters with exact and semantic deduplication to remove low-quality and redundant text (Albalak et al., 2024; Lee et al., 2022; Abbas et al., 2023). Modelbased methods instead rank documents by reference-model cross-entropy or perplexity (Marion et al., 2023). At the domain level, DoReMi uses proxy-model excess loss to optimize mixture weights (Xie et al., 2023), DOGE estimates how training on one domain affects generalization to others (Fan et al., 2023), and datamixing laws extrapolate the performance of candidate mixtures from smaller training runs (Ye et al., 2025). Online approaches further adapt sampling probabilities during training (Albalak et al., 2023; Jiang et al., 2025). Our study focuses on TSFM pretraining and we score context–future windows using reference-model loss.

## 2.3 Data selection for time series

Existing time series selection methods mostly address data valuation, task-specific model training, or foundationmodel finetuning rather than pretraining a forecasting TSFM from scratch. TimeInf adapts influence functions to temporally dependent blocks and attributes model predictions to individual time points (Zhang et al., 2025). TSRating distills pairwise LLM judgments into a cross-domain quality rater and evaluates selected subsets with conventional models and TSFM fine-tuning (Wu et al., 2026). AdaRho uses the reducible-loss difference between a target and an adaptively updated reference model for online filtering and augmentation on individual forecasting datasets (Taga et al., 2025). These methods demonstrate the value of time-seriesaware selection, but operate in task-specific training or fine-tuning regimes and require repeated adaptation or influence computations. Consequently, they are difficult to apply at TSFM pretraining scale. To the best of our knowledge, this is the first systematic study of loss-based sample-level selection for pretraining of time series foundation models.

![](images/98133ab8d15f872c467aa6e28e3ab1e53492e79b276267f3755a6ac00c5846c7.jpg)  
Figure 2: Overview of the selection framework. (1) Candidate context–future windows are collected by their source datasets. (2) A frozen reference TSFM assigns each window an RFL score. (3) Scores are ranked within each source, and the central α interval is retained. (4) The selected subset is used to pretrain the target TSFM, aiming to improve performance relative to full-data pretraining.

## 3 Method

## 3.1 Problem formulation

Let the pretraining corpus be partitioned into K source datasets, $\textstyle { \mathcal { D } } = \bigcup _ { k = 1 } ^ { K } { \mathcal { D } } _ { k }$ . A sampled forecasting sample is $z _ { i } = ( \pmb { x } _ { i } , \pmb { y } _ { i } , \pmb { m } _ { i } )$ , where $\pmb { x } _ { i } \in \mathbb { R } ^ { C }$ is a context of length $C , y _ { i } \in \mathbb { R } ^ { H }$ is the future window and H is the forecast horizon, $m _ { i } \in \{ 0 , 1 \} ^ { H }$ marks observed future positions, where zero indicates a missing value. Our goal is to choose $s \subset \mathcal { D }$ with $| S | \approx \alpha | \mathcal { D } |$ such that training a target forecaster on $\boldsymbol { \mathcal { S } }$ retains or improves zero-shot performance while reducing the number of distinct windows that must be stored and processed.

Figure 2 illustrates the complete framework. First, the reference loss score assigns every candidate window a difficulty value using a frozen forecaster. Second, dataset-stratified selection ranks these scores within each source dataset and constructs the retained pretraining subset while preserving source-level diversity.

## 3.2 Reference loss

Time-series forecasting models are commonly trained with point or probabilistic objectives. Let $M _ { i } \ =$ $\Sigma _ { h = 1 } ^ { H } m _ { i , h }$ be the number of observed positions in the future window. For a point forecaster $f _ { \pmb { \theta } } ,$ , the masked mean-squared error (MSE) is

$$
\mathcal { L } _ { i } ^ { \mathrm { M S E } } ( \pmb { \theta } ) = \frac { 1 } { M _ { i } } \sum _ { h = 1 } ^ { H } m _ { i , h } \left( f _ { \pmb { \theta } } ( \pmb { x } _ { i } ) _ { h } - y _ { i , h } \right) ^ { 2 } .\tag{1}
$$

MSE measures point accuracy but does not evaluate the full predictive distribution. For a probabilistic forecaster with predictive cumulative distribution $F _ { \pmb { \theta } , i , h }$ , the continuous ranked probability score (CRPS) evaluates both calibration and sharpness. Weighted Quantile Loss (WQL) is a common approximation of CRPS and is commonly adopted in probabilistic forecasting (Gneiting and Raftery, 2007). For simplicity, we mainly use the MSE metric for the analysis.

Reference loss connects forecasting objectives to data selection. Let $f _ { \phi }$ be a frozen reference forecaster and let $\ell _ { i } ( \pmb \theta )$ denote the native masked loss on window $z _ { i }$ . We define the reference loss score (RFL) as

$$
\begin{array} { r } { s _ { i } ^ { \mathrm { R F L } } = \ell _ { i } ( \phi ) . } \end{array}\tag{2}
$$

RFL records the per-window value of an existing forecasting objective under fixed reference model parameters and a larger value indicates that the reference model explains the window less well. Motivated by standard first-order analyses of training-data influence (Pruthi et al., 2020; Xia et al., 2024), we now formalize what RFL reveals about optimization difficulty.

Consider the empirical risk of a model $f _ { \theta }$ over the complete candidate pool $\mathcal { D } = \{ z _ { j } \} _ { j = 1 } ^ { N } , N = | \mathcal { D } |$ is the total number of candidate pretraining windows before selection. We have:

$$
R ( \pmb \theta ) = \frac { 1 } { N } \sum _ { j = 1 } ^ { N } \ell _ { j } ( \pmb \theta ) , \qquad \pmb g _ { j } = \nabla _ { \pmb \theta } \ell _ { j } ( \pmb \theta ) , \qquad \bar { \pmb g } = \nabla _ { \pmb \theta } R ( \pmb \theta ) .\tag{3}
$$

To examine the utility of a single window, consider one SGD update $\pmb { \theta } ^ { + } = \pmb { \theta } - \eta \pmb { g } _ { i }$ . A first-order Taylor expansion of the candidate-pool risk gives $R ( \pmb \theta ^ { + } ) - R ( \pmb \theta ) = - \eta \langle \bar { \pmb g } , \pmb g _ { i } \rangle + o ( \eta )$ Thus, positive alignment with the candidate-pool gradient makes the leading-order risk change negative, whereas zero or negative alignment provides no first-order decrease. This separates two properties that a useful window should possess. Its gradient must be large enough to affect the parameters, but it must also align with the population gradient. A corrupted or intrinsically unpredictable future may have a large gradient yet weak or negative utility.

For the MSE objective in Equation 1, let $\pmb { r } _ { i } = \pmb { \widetilde { y } } _ { i } - f _ { \pmb { \theta } } ( \pmb { \widetilde { x } } _ { i } ) \in \mathbb { R } ^ { M _ { i } }$ collect residuals only at observed future positions. For algebraic convenience, define the half-MSE, which differs from Equation 1 by a positive constant and therefore leaves RFL rankings unchanged:

$$
\ell _ { i } ^ { \mathrm { M S E } } ( \pmb \theta ) = \frac { 1 } { 2 } \mathcal { L } _ { i } ^ { \mathrm { M S E } } ( \pmb \theta ) = \frac { 1 } { 2 M _ { i } } \| \pmb { r } _ { i } \| _ { 2 } ^ { 2 } ,\tag{4}
$$

Then the connection between MSE loss and gradient magnitude is explicit.

Theorem 3.1 (MSE controls potential update magnitude). For MSE in Equation 4, the per-window gradient satisfies

$$
{ \pmb g } _ { i } = - \frac { 1 } { M _ { i } } { \pmb J } _ { i } ^ { \top } { \pmb r } _ { i } , \qquad \| { \pmb g } _ { i } \| _ { 2 } ^ { 2 } = \frac { 1 } { M _ { i } ^ { 2 } } { \pmb r } _ { i } ^ { \top } J _ { i } J _ { i } ^ { \top } { \pmb r } _ { i } .\tag{5}
$$

where $J _ { i } = \partial f _ { \pmb { \theta } } ( \widetilde { \pmb x } _ { i } ) / \partial \pmb \theta$ is the Jacobian ofthe observed predictions. Let $0 \leq \lambda _ { i } ^ { - } \leq \lambda _ { i } ^ { + }$ be the smallest and largest eigenvalues of the positive-semidefinite matrix $J _ { i } J _ { i } ^ { \top }$ . Then

$$
\frac { 2 \lambda _ { i } ^ { - } } { M _ { i } } \ell _ { i } ^ { \mathrm { M S E } } \leq \| { \pmb g } _ { i } \| _ { 2 } ^ { 2 } \leq \frac { 2 \lambda _ { i } ^ { + } } { M _ { i } } \ell _ { i } ^ { \mathrm { M S E } } .\tag{6}
$$

The proof is provided in Appendix A. Theorem 3.1 states the precise sense in which normalized MSE is an optimization-difficulty score. If $M _ { i }$ and local Jacobian conditioning are comparable across windows, ranking their losses approximately ranks the magnitudes of the updates they can induce. For a nonnegative $\beta \mathrm { - s m o o t h }$ objective, the standard self-bounding inequality $\lVert \nabla \ell _ { i } \rVert _ { 2 } ^ { 2 } \leq 2 \beta \ell _ { i }$ provides the one-sided result that very low loss rules out a large gradient (Srebro et al., 2010). Evaluating these relations at the frozen reference parameters ϕ motivates using $s _ { i } ^ { \mathrm { R F L } }$ as a cheap difficulty coordinate.

## 3.3 Dataset-stratified selection

Dataset-stratified selection applies RFL-based difficulty filtering while preserving the diversity of the pretraining corpus. We define the candidate corpus as $\mathcal { D } = \{ z _ { i } \} _ { i = 1 } ^ { \bar { N } } = \bigcup _ { k = 1 } ^ { \bar { K } } \mathcal { D } _ { k }$ , where K is the number of source datasets, $\mathcal { D } _ { k }$ is the set of candidate windows from source $k ,$ and $n _ { k } = | \mathcal { D } _ { k } |$ . Moreover, each window $z _ { i }$ has an RFL score $s _ { i } ^ { \mathrm { R F L } }$ and a source labe $g _ { i } \in \{ 1 , \ldots , K \}$ , such that $z _ { i } \in \mathcal { D } _ { g _ { i } }$

The preceding first-order Taylor expansion analysis motivates retaining an intermediate RFL interval. Low-RFL windows are expected to provide limited update magnitude, whereas the high-RFL windows are more likely to contain a mixture of rare useful patterns, distribution shifts, and unlearnable noise. By contrast, intermediate RFL windows retain nontrivial residuals while remaining close enough to learned temporal structure that their gradients may align with recurring patterns. This interpretation is consistent with empirical findings on moderate-score and intermediate-perplexity data selection in LLM and CV (Xia et al., 2023; Marion et al., 2023).

Therefore, we first consider global selection. It implements this conclusion by pooling all windows, ranking them by absolute RFL, and retaining the central α fraction. Its corpus-level empirical cumulative distribution and selected subset are

$$
\widehat { F } _ { \mathrm { a l l } } ( t ) = \frac { 1 } { N } \sum _ { j = 1 } ^ { N } \mathbb { I } [ s _ { j } ^ { \mathrm { R F L } } \leq t ] , \qquad S _ { \alpha } ^ { \mathrm { g l o b a l } } = \left\{ z _ { i } \in \mathcal { D } : \frac { 1 - \alpha } { 2 } \leq \widehat { F } _ { \mathrm { a l l } } ( s _ { i } ^ { \mathrm { R F L } } ) \leq \frac { 1 + \alpha } { 2 } \right\} .\tag{7}
$$

Here, t is an RFL threshold and $\mathbb { I } [ \cdot ]$ is the indicator function. The desired retention ratio $\alpha \in ( 0 , 1 ]$ specifies the fraction of the complete corpus to retain, and $S _ { \alpha } ^ { \mathrm { g l o b a l } }$ contains the windows between the $( 1 - \alpha ) / 2$ and $( 1 + \alpha ) / 2$ quantiles of the pooled distribution.

Global selection controls $| S _ { \alpha } ^ { \mathrm { g l o b a l } } | \approx \alpha N$ , but it does not control the source-specific retention rate. Absolute RFL distributions can differ across datasets because of their noise levels, temporal patterns, and intrinsic predictability. Consequently, a source whose scores concentrate near either global tail can be severely underrepresented or removed, which may harm dataset diversity.

Dataset-stratified selection instead measures the difficulty of each window relative to other windows from the same source. For source $k ,$ , we define the source-specific empirical cumulative distribution $\widehat { F } _ { k }$ and the within-source percentile rank $r _ { i }$ as

$$
\widehat { F } _ { k } ( t ) = \frac { 1 } { n _ { k } } \sum _ { j : g _ { j } = k } \mathbb { I } [ s _ { j } ^ { \mathrm { R F L } } \leq t ] , \qquad r _ { i } = \widehat { F } _ { g _ { i } } ( s _ { i } ^ { \mathrm { R F L } } ) .\tag{8}
$$

Here, $r _ { i } \in ( 0 , 1 ]$ is the fraction of windows in $\mathcal { D } _ { g _ { i } }$ whose RFL score does not exceed that of window i. We then retain the central α fraction independently within every source:

$$
S _ { \alpha } \equiv S _ { \alpha } ^ { \mathrm { s t r a t } } = \left\{ z _ { i } \in \mathcal { D } : \frac { 1 - \alpha } { 2 } \leq r _ { i } \leq \frac { 1 + \alpha } { 2 } \right\} .\tag{9}
$$

Equation 9 retains $\alpha n _ { k }$ windows from every source k. The proportion of source k in the selected subset is therefore

$$
p _ { k } ^ { S } = \frac { \left| \mathcal { S } _ { \alpha } \cap \mathcal { D } _ { k } \right| } { \left| \mathcal { S } _ { \alpha } \right| } = \frac { \alpha n _ { k } } { \alpha N } = \frac { n _ { k } } { N } = p _ { k } ,\tag{10}
$$

where $p _ { k } = n _ { k } / N$ is the proportion of source k in the original candidate corpus. Dataset-stratified selection thus filters windows by relative difficulty while preserving source coverage and dataset-level diversity. We analyze the benefit of the framework in Section 4.4.2.

## 3.4 Algorithm and computational cost

Algorithm 1 summarizes the two components of the framework. We first assign every candidate window an RFL score with a frozen reference model. We then rank the scores separately within each source dataset and retain the central interval. The reference and target architectures may differ, which permits an inexpensive proxy to amortize selection over several target models.

The offline algorithm requires one reference-model forward pass per candidate window. It additionally requires $\scriptstyle \sum _ { k = 1 } ^ { K } O ( n _ { k } \log n _ { k } )$ time to sort the scores within sources and stores one scalar per window. The generated scores can be reused across target-model sizes and architectures, amortizing the scoring cost.

Algorithm 1 Reference-loss scoring and dataset-stratified selection   
Require: Candidate corpus $\textstyle { \mathcal { D } } = \bigcup _ { k = 1 } ^ { K } { \mathcal { D } } _ { k } ;$ frozen reference model $f _ { \phi } ;$ retention ratio α   
Ensure: Selected subset $S _ { \alpha }$ and trained target parameters $\pmb \theta$   
1: $S _ { \alpha } \gets \emptyset$   
2: for each candidate window $z _ { i } \in \mathcal { D }$ do   
3: Compute its RFL score $s _ { i } ^ { \mathrm { R F L } }  \ell _ { i } ( \phi )$   
4: end for   
5: for each source dataset $k \in \{ 1 , \ldots , K \}$ do   
6: Sort the $n _ { k }$ windows by increasing RFL score   
7: Denote the resulting order by $z _ { \pi _ { k } ( 1 ) } , \ldots , z _ { \pi _ { k } ( n _ { k } ) }$   
8: $m _ { k } \gets \operatorname* { m a x } \{ 1 ;$ round(αn<sub>k</sub>)}   
9: $a _ { k }  \lfloor ( n _ { k } - m _ { k } ) / 2 \rfloor + 1$   
10: $S _ { \alpha }  S _ { \alpha } \cup \{ z _ { \pi _ { k } ( j ) } : a _ { k } \leq j < a _ { k } + m _ { k } \}$   
11: end for   
12: Train the randomly initialized target model $f _ { \theta }$ on $S _ { \alpha }$   
13: return $S _ { \alpha } , \theta$

## 4 Experiments

We organize the experiments around three questions. (Q1) can dataset-stratified RFL selection improve full-data pretraining while retaining fewer training samples? (Q2) can RFL scores be transferred across reference-model scales and architectures? (Q3) how important are the intermediate interval and dataset stratification?

## 4.1 Experimental setup

Data. The pretraining data mainly comes from the Chronos data collection (Ansari et al., 2024) and LOTSA (Woo et al., 2024). It contains more than 100 source datasets and 4.64M univariate series, spanning climate and weather, transportation, health and mobility data, industrial measurements, synthetic series generated via Gaussian processes(Ansari et al., 2024) and so on. We build validation windows from the last complete horizon of each series and keep them separate from the training windows. All architectural variants use the same validation samples to ensure comparable validation losses.

Training. We pretrain three representative TSFM families: Chronos-Bolt Base (Ansari et al., 2024) (encoderdecoder), Moirai-Base (Woo et al., 2024) (encoder), and TimesFM 2.5 (Das et al., 2024) (decoder). All models use AdamW, cosine learning-rate decay, gradient clipping, and an effective batch size of 512 on NVIDIA A100 GPUs. The initial learning rate is $3 \times 1 0 ^ { - 4 }$ for TimesFM and $1 \times 1 0 ^ { - 3 }$ for the other models, with 5,000 and 2,000 warm-up steps, respectively. We set the training budget to 40,000 optimizer steps for TimesFM and 100,000 steps for others. The max context length is set to 2,048 and prediction length is architecture-specific: 64 for Chronos-Bolt, 96 for Moirai, and 128 for TimesFM.

Evaluation. We use GIFT-Eval (Aksu et al., 2024) and remove training-overlapping benchmarks. The current evaluation contains 91 task configurations. Following GIFT-Eval, we normalize MASE and CRPS by a seasonal-naive forecaster and report the geometric mean across tasks. We also report MSE and MAE on five commonly used benchmarks in ablation study: ETTh1, ETTh2, ETTm1, ETTm2, and weather. These datasets have been utilized for benchmarking and publicly available on (Wu et al., 2021). None of these datasets is included in pretraining datasets.

Table 1: GIFT-Eval comparison at different retention ratios. Lower MASE and CRPS are better. Full pretraining uses 100% of the candidate windows. Bold denotes the best selection method.
<table><tr><td rowspan="2">Target model</td><td rowspan="2">Metric ↓</td><td rowspan="2">Full (100%)</td><td colspan="2">40% retention</td><td colspan="2">60% retention</td><td colspan="2">80% retention</td></tr><tr><td>Random</td><td>Ours</td><td>Random</td><td>Ours</td><td>Random</td><td>Ours</td></tr><tr><td rowspan="2">Chronos-Bolt Base</td><td>MASE</td><td>0.796</td><td>0.836</td><td>0.803</td><td>0.806</td><td>0.801</td><td>0.799</td><td>0.777</td></tr><tr><td>CRPS</td><td>0.555</td><td>0.576</td><td>0.569</td><td>0.555</td><td>0.554</td><td>0.549</td><td>0.542</td></tr><tr><td rowspan="2">Moirai-Base</td><td>MASE</td><td>1.017</td><td>1.026</td><td>1.050</td><td>1.061</td><td>1.031</td><td>1.022</td><td>1.005</td></tr><tr><td>CRPS</td><td>0.698</td><td>0.708</td><td>0.749</td><td>0.732</td><td>0.731</td><td>0.701</td><td>0.680</td></tr><tr><td rowspan="2">TimesFM 2.5</td><td>MASE</td><td>0.813</td><td>0.833</td><td>0.821</td><td>0.840</td><td>0.823</td><td>0.835</td><td>0.804</td></tr><tr><td>CRPS</td><td>0.558</td><td>0.573</td><td>0.561</td><td>0.574</td><td>0.560</td><td>0.576</td><td>0.554</td></tr></table>

## 4.2 Overall Comparison

We compare dataset-stratified RFL selection with uniform random selection and full data pretraining under the same computational budget. To test whether the selection rule generalizes beyond model design, we pretrain three distinct target architectures and score their candidate pools using frozen references from the corresponding architecture. Table 1 shows that our method outperforms random selection by an absolute margin, indicating that RFL provides a useful selection signal across architectures. Since the random control retains the same number of windows, this consistency attributes the gain to which windows are retained. At 80% retention, our method improves over both full-data pretraining and a size-matched random subset on all target models. This is because we remove high-loss noise and harmful data and prune unnecessary low-loss data while maintaining diversity.

## 4.3 Reference-model analysis

We investigate how the choice of reference model affects data selection along two dimensions: scale transfer within an architecture and transfer across architectures. We use Chronos-Bolt Base and Moirai-Base as target models and retain 80% of the candidate windows. For within-architecture transfer, we vary the frozen reference among Chronos-Bolt Mini and Small, or between Moirai-Small and Base. For cross-architecture transfer, a frozen TimesFM reference selects data for each target. This design tests whether RFL requires a capacity-matched reference or can instead reuse a smaller or architecturally different pretrained forecaster.

Within-architecture references are largely robust to model scale. As shown in Table 2, all subset configurations improve MASE and CRPS over full-data pretraining, indicating small reference models produce competitive training subsets. The score analysis in Appendix B explains this robustness. In brief, the pairwise Spearman correlations among Chronos-Bolt families range from 0.977 to 0.986, and Moirai families obtain $\rho = 0 . 9 9 1$ . This agreement indicates that models from the same architecture share a stable ordering of window difficulty. Consequently, a smaller pretrained reference can often approximate the ranking of a larger model and reduce the cost of the required forward scoring pass.

Cross-architecture transfer is conditional. A TimesFM reference selects a useful subset for Chronos-Bolt Base target, improving full-data pretraining and remaining competitive with the best reference. In contrast, the TimesFM-selected subset performs poorly in Moirai-Base pretraining, degrading to the level of random selection in Table 1. Indeed, TimesFM has strong RFL score rank agreement with Chronos-Bolt Mini on the candidate pool $( \rho = 0 . 8 2 8 )$ ), but only weak agreement with Moirai-Small $( \rho = 0 . 2 5 4 )$ . For more analysis details, please see Appendix B. A plausible explanation is that TimesFM scores point and quantile errors, whereas Moirai uses a distributional negative log-likelihood, so they may not assign similar difficulty to high-uncertainty windows.

Table 2: Evaluation results of cross-scale and cross-architecture reference models with 80% retention. Bold and underlined values are best and second best for each target model.
<table><tr><td>Target Model</td><td>Reference Model</td><td>Retained</td><td>MASE</td><td>CRPS</td></tr><tr><td rowspan="4">Chronos-Bolt Base</td><td>Chronos-Bolt Mini</td><td>80%</td><td>0.777</td><td>0.542</td></tr><tr><td>Chronos-Bolt Small</td><td>80%</td><td>0.786</td><td>0.545</td></tr><tr><td>TimesFM</td><td>80%</td><td>0.786</td><td>0.541</td></tr><tr><td>— (Full data)</td><td>100%</td><td>0.796</td><td>0.555</td></tr><tr><td rowspan="4">Moirai-Base</td><td>Moirai-Small</td><td>80%</td><td>1.003</td><td>0.695</td></tr><tr><td>Moirai-Base</td><td>80%</td><td>1.005</td><td>0.680</td></tr><tr><td>TimesFM</td><td>80%</td><td>1.022</td><td>0.708</td></tr><tr><td>— (Full data)</td><td>100%</td><td>1.017</td><td>0.698</td></tr></table>

## 4.4 Ablation studies

## 4.4.1 RFL region

Dataset-stratified selection retains the central RFL interval by default. To isolate the effect of this choice, we compare three score-based subsets constructed independently within every source. Bottom retains the lowest-RFL tail, Top retains the highest-RFL tail, and Central retains the central interval. We additionally include a dataset-stratified random baseline that samples windows uniformly at random within each source, retaining the prescribed fraction from every source. All runs in this ablation use TimesFM as both the reference and target model. We evaluate 40% and 80% retention on five forecasting benchmarks in Table 3 and compare matched validation-loss trajectories through 40k optimization steps against source-matched random selection and full-data pretraining in Figure 3.

Table 3 demonstrates the advantages of the Central selection, consistent with the optimization inter pretation in Section 3.2. Bottom retains low-RFL windows that the reference model already predicts well and are therefore relatively easy. Concentrating on this region can overallocate training capacity to redundant patterns with little residual learning signal. One possible explanation for this redundancy is that intermediate-difficulty windows contain the regularities needed for the easier cases. Once the target model learns these richer patterns, it can readily generalize to low-difficulty windows without repeatedly training on them. By contrast, Top retains high-difficulty windows. Although some of these windows may contain rare informative structure, a high loss can also arise from noise, abrupt unpredictable transitions, or distribution shifts, making the entire upper tail less reliably learnable. Central avoids both extremes and retains windows with nontrivial prediction error but sufficient regularity to provide stable learning signal.

The validation trajectories in Figure 3(a,b) reinforce this interpretation. Central remains among the lowest-loss trajectories over much of training and separates most clearly when the retained budget is small. This suggests that the benefit comes from excluding both simple windows and the most unstable tail, not from preferring uniformly easier or harder data.

## 4.4.2 Dataset stratification selection

Dataset stratification is the diversity-preserving component of our framework. We compare the proposed dataset-stratified rule in Equation 9 with global selection in Equation 7. Both rules retain the same central RFL interval and use the same total retention ratio. They differ only in whether percentile ranks are computed within each source or over the pooled corpus. We use the default reference model to score the candidate pool and pretrain a Chronos-Bolt Base target model on the selected subsets. Figure 3(c) shows that the stratified rule improves GIFT-Eval metrics at all retention ratios. This improvement indicates that global

Table 3: RFL-region ablation at 40% and 80% retention. During downstream evaluation, each dataset uses a context window of 512 and a prediction window of 96.
<table><tr><td rowspan="2">Ratio</td><td rowspan="2">Region</td><td colspan="2">ETTh1</td><td colspan="2">ETTh2</td><td colspan="2">ETTm1</td><td colspan="2">ETTm2</td><td colspan="2">Weather</td></tr><tr><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td rowspan="4">40%</td><td>Bottom</td><td>0.404</td><td>0.393</td><td>0.307</td><td>0.342</td><td>0.351</td><td>0.359</td><td>0.192</td><td>0.260</td><td>0.182</td><td>0.214</td></tr><tr><td>Top</td><td>0.394</td><td>0.411</td><td>0.315</td><td>0.358</td><td>0.357</td><td>0.380</td><td>0.197</td><td>0.273</td><td>0.180</td><td>0.221</td></tr><tr><td>Random</td><td>0.388</td><td>0.398</td><td>0.302</td><td>0.348</td><td>0.347</td><td>0.365</td><td>0.182</td><td>0.259</td><td>0.176</td><td>0.212</td></tr><tr><td>Ours</td><td>0.383</td><td>0.390</td><td>0.302</td><td>0.337</td><td>0.341</td><td>0.357</td><td>0.186</td><td>0.258</td><td>0.169</td><td>0.202</td></tr><tr><td rowspan="4">80%</td><td>Bottom</td><td>0.386</td><td>0.391</td><td>0.300</td><td>0.337</td><td>0.348</td><td>0.357</td><td>0.184</td><td>0.255</td><td>0.172</td><td>0.206</td></tr><tr><td>Top</td><td>0.391</td><td>0.400</td><td>0.305</td><td>0.349</td><td>0.346</td><td>0.366</td><td>0.182</td><td>0.259</td><td>0.171</td><td>0.209</td></tr><tr><td>Random</td><td>0.379</td><td>0.390</td><td>0.294</td><td>0.347</td><td>0.345</td><td>0.367</td><td>0.189</td><td>0.262</td><td>0.173</td><td>0.210</td></tr><tr><td>Ours</td><td>0.373</td><td>0.388</td><td>0.287</td><td>0.334</td><td>0.339</td><td>0.359</td><td>0.181</td><td>0.255</td><td>0.170</td><td>0.204</td></tr><tr><td>100%</td><td>Full</td><td>0.397</td><td>0.398</td><td>0.297</td><td>0.342</td><td>0.350</td><td>0.363</td><td>0.187</td><td>0.261</td><td>0.175</td><td>0.210</td></tr></table>

Full data Random Bottom Central (Ours) Top  
Global Dataset-stratified (Ours)

(a) 40% retention

![](images/9b980cae71143d01a260b757c538ccfc3ae7a2f93dba1c6456469011ff1e5c98.jpg)

(b) 80% retention  
![](images/c8b783b58fe1fab5cc80543f2e27aee447c4100e672d2de6d4d5775644eac408.jpg)

(c) Dataset stratification  
![](images/f79f74def5a52a871d0328570acb13183abf6d81e2277660a9cc3080dffc2cb0.jpg)  
Figure 3: Ablations of the two components of our framework. (a,b) Validation-loss trajectories for full-data pretraining, random selection and the Bottom, Central (Ours), and Top RFL regions at 40% and 80% retention. (c) GIFT-Eval results for dataset-stratified selection and global selection at 40% and 80% retention.

percentiles harm the data diversity, whereas within-source percentiles preserve coverage. The persistence of the advantage from 40% to 80% supports dataset stratification as an effective diversity constraint.

## 5 Conclusion

This paper presents a reference-loss selection framework for selecting training windows for time-series foundation-model pretraining. We introduce reference loss as a model-compatible difficulty score computed by a frozen forecaster. Our analysis connects loss to potential update magnitude while clarifying that actual one-step utility also depends on gradient alignment. Based on this, our dataset-stratified selection retains an intermediate interval within every source dataset. The resulting subset removes low-loss windows with limited residual learning signal and avoids overconcentrating on the unstable high-loss tail, while preserving source coverage and dataset-level diversity. Across three distinct TSFM families, our method outperforms size-matched random selection by an absolute margin, and even surpasses the training on the full sample set. Together, these findings establish pretraining-data selection as an important design axis alongside model architecture and corpus scale for time series foundation models.

## References

Amro Abbas, Kushal Tirumala, Daniel Simig, Surya Ganguli, and Ari S. Morcos. SemDeDup: Data-´ efficient learning at web-scale through semantic deduplication. arXiv preprint arXiv:2303.09540, 2023.

Taha Aksu, Gerald Woo, Juncheng Liu, Xu Liu, Chenghao Liu, Silvio Savarese, Caiming Xiong, and Doyen Sahoo. GIFT-Eval: A benchmark for general time series forecasting model evaluation. In arXiv preprint arXiv:2410.10393, 2024.

Alon Albalak, Liangming Pan, Colin Raffel, and William Yang Wang. Efficient online data mixing for language model pre-training. In NeurIPS Workshop on Robustness of Few-shot and Zero-shot Learning in Foundation Models, 2023.

Alon Albalak, Yanai Elazar, Sang Michael Xie, Shayne Longpre, Nathan Lambert, Xinyi Wang, Niklas Muennighoff, Bairu Hou, Liangming Pan, Hae Won Jeong, et al. A survey on data selection for language models. Transactions on Machine Learning Research, 2024.

Abdul Fatir Ansari, Lorenzo Stella, Caner Turkmen, Xiyuan Zhang, Pedro Mercado, Huibin Shen, Oleksandr Shchur, Syama Sundar Rangapuram, Sebastian Pineda Arango, Shubham Kapoor, et al. Chronos: Learning the language of time series. Transactions on Machine Learning Research, 2024.

Abhimanyu Das, Weihao Kong, Rajat Sen, and Yichen Zhou. A decoder-only foundation model for timeseries forecasting. arXiv preprint arXiv:2310.10688, 2024.

Simin Fan, Matteo Pagliardini, and Martin Jaggi. DOGE: Domain reweighting with generalization estimation. In NeurIPS Workshop on Agent Learning in Open-Endedness, 2023.

Tilmann Gneiting and Adrian E. Raftery. Strictly proper scoring rules, prediction, and estimation. Journal ofthe American Statistical Association, 102(477):359–378, 2007.

Yiding Jiang, Allan Zhou, Zhili Feng, Sadhika Malladi, and J. Zico Kolter. Adaptive data optimization: Dynamic sample selection with scaling laws. In International Conference on Learning Representations, 2025.

Angelos Katharopoulos and Franc¸ois Fleuret. Not all samples are created equal: Deep learning with importance sampling. In Proceedings ofthe 35th International Conference on Machine Learning, 2018.

Katherine Lee, Daphne Ippolito, Andrew Nystrom, Chiyuan Zhang, Douglas Eck, Chris Callison-Burch, and Nicholas Carlini. Deduplicating training data makes language models better. In Proceedings of the 60th Annual Meeting ofthe Associationfor Computational Linguistics, 2022.

Yong Liu, Guo Qin, Xiangdong Huang, Jianmin Wang, and Mingsheng Long. Timer-XL: Long-context transformers for unified time series forecasting. In International Conference on Learning Representations, 2025a.

Yong Liu, Guo Qin, Zhiyuan Shi, Zhi Chen, Caiyin Yang, Xiangdong Huang, Jianmin Wang, and Mingsheng Long. Sundial: A family of highly capable time series foundation models. In Proceedings of the 42nd International Conference on Machine Learning, 2025b.

Max Marion, Ahmet Ust<sup>¨</sup> un, Luiza Pozzobon, Alex Wang, Marzieh Fadaee, and Sara Hooker. When less¨ is more: Investigating data pruning for pretraining LLMs at scale. In Advances in Neural Information Processing Systems, volume 36, 2023.

Mansheej Paul, Surya Ganguli, and Gintare Karolina Dziugaite. Deep learning on a data diet: Finding important examples early in training. In Advances in Neural Information Processing Systems, volume 34, 2021.

Garima Pruthi, Frederick Liu, Satyen Kale, and Mukund Sundararajan. Estimating training data influence by tracing gradient descent. In Advances in Neural Information Processing Systems, volume 33, 2020.

Xiaoming Shi, Shiyu Wang, Yuqi Nie, Dianqi Li, Zhou Ye, Qingsong Wen, and Ming Jin. Time-MoE: Billion-scale time series foundation models with mixture of experts. In International Conference on Learning Representations, 2025.

Nathan Srebro, Karthik Sridharan, and Ambuj Tewari. Smoothness, low noise and fast rates. In Advances in Neural Information Processing Systems, volume 23, 2010.

Ege Onur Taga, Halil Alperen Gozeten, Kutay Tire, Rahul Dalvi, Reinhard Heckel, and Samet Oymak. Filter, augment, forecast: Online data selection for robust time series forecasting. In ICML Workshop on Foundation Modelsfor Structured Data, 2025.

Gerald Woo, Chenghao Liu, Akshat Kumar, Caiming Xiong, Silvio Savarese, and Doyen Sahoo. Unified training of universal time series forecasting transformers. In Proceedings of the 41st International Conference on Machine Learning, 2024.

Haixu Wu, Jiehui Xu, Jianmin Wang, and Mingsheng Long. Autoformer: Decomposition transformers with auto-correlation for long-term series forecasting. In Advances in Neural Information Processing Systems, volume 34, pages 22419–22430, 2021.

Shunyu Wu, Dan Li, Wenjie Feng, Haozheng Ye, Jian Lou, and See-Kiong Ng. Rating quality of diverse time series data by meta-learning from LLM judgment. In International Conference on Learning Representations, 2026.

Mengzhou Xia, Sadhika Malladi, Suchin Gururangan, Sanjeev Arora, and Danqi Chen. LESS: Selecting influential data for targeted instruction tuning. In Proceedings of the 41st International Conference on Machine Learning, volume 235, pages 54104–54132, 2024.

Xiaobo Xia, Jiale Liu, Jun Yu, Xu Shen, Bo Han, and Tongliang Liu. Moderate coreset: A universal method of data selection for real-world data-efficient deep learning. In International Conference on Learning Representations, 2023.

Sang Michael Xie, Hieu Pham, Xuanyi Dong, Nan Du, Hanxiao Liu, Yifeng Lu, Percy Liang, Quoc V. Le, Tengyu Ma, and Adams Wei Yu. DoReMi: Optimizing data mixtures speeds up language model pretraining. In Advances in Neural Information Processing Systems, 2023.

Jiasheng Ye, Peiju Liu, Tianxiang Sun, Jun Zhan, Yunhua Zhou, and Xipeng Qiu. Data mixing laws: Optimizing data mixtures by predicting language modeling performance. In International Conference on Learning Representations, 2025.

Yizi Zhang, Jingyan Shen, Xiaoxue Xiong, and Yongchan Kwon. TimeInf: Time series data contribution via influence functions. In International Conference on Learning Representations, 2025.

## A Proof of Theorem 3.1

Proof. Recall that ${ \pmb r } _ { i } = \widetilde { { \pmb y } } _ { i } - f _ { \pmb \theta } ( \widetilde { { \pmb x } } _ { i } )$ contains the $M _ { i }$ residuals at observed future positions and that

$$
\ell _ { i } ^ { \mathrm { M S E } } ( \pmb { \theta } ) = \frac { 1 } { 2 M _ { i } } \pmb { r } _ { i } ^ { \top } \pmb { r } _ { i } .\tag{11}
$$

Because $\partial r _ { i } / \partial \pmb \theta = - \pmb J _ { i }$ , the chain rule gives

$$
{ \pmb g } _ { i } = \nabla _ { \pmb \theta } \ell _ { i } ^ { \mathrm { M S E } } ( { \pmb \theta } ) = \frac { 1 } { M _ { i } } \left( \frac { \partial { \pmb r } _ { i } } { \partial { \pmb \theta } } \right) ^ { \top } { \pmb r } _ { i } = - \frac { 1 } { M _ { i } } { \pmb J } _ { i } ^ { \top } { \pmb r } _ { i } .\tag{12}
$$

Taking its squared Euclidean norm yields

$$
\| \pmb { g } _ { i } \| _ { 2 } ^ { 2 } = \frac { 1 } { M _ { i } ^ { 2 } } ( \pmb { J } _ { i } ^ { \top } \pmb { r } _ { i } ) ^ { \top } ( \pmb { J } _ { i } ^ { \top } \pmb { r } _ { i } ) = \frac { 1 } { M _ { i } ^ { 2 } } \pmb { r } _ { i } ^ { \top } \pmb { J } _ { i } \pmb { J } _ { i } ^ { \top } \pmb { r } _ { i } ,\tag{13}
$$

which proves Equation 5.

Let $\mathbf { } A _ { i } = J _ { i } \mathbf { J } _ { i } ^ { \top }$ . This matrix is positive semidefinite because, for every vector u, $\mathbf { u } ^ { \top } A _ { i } \mathbf { u } = \| J _ { i } ^ { \top } \mathbf { u } \| _ { 2 } ^ { 2 } \geq$ 0. Applying the Rayleigh–Ritz inequality to $\mathbf { A } _ { i }$ gives

$$
\lambda _ { i } ^ { - } \| { \pmb r } _ { i } \| _ { 2 } ^ { 2 } \leq { \pmb r } _ { i } ^ { \top } { \pmb A } _ { i } { \pmb r } _ { i } \leq \lambda _ { i } ^ { + } \| { \pmb r } _ { i } \| _ { 2 } ^ { 2 } .\tag{14}
$$

Finally, Equation 11 implies $\| \boldsymbol { r } _ { i } \| _ { 2 } ^ { 2 } = 2 M _ { i } \ell _ { i } ^ { \mathrm { M S E } }$ . Substituting this identity into Equation 14 and dividing by $M _ { i } ^ { 2 }$ yields

$$
\frac { 2 \lambda _ { i } ^ { - } } { M _ { i } } \ell _ { i } ^ { \mathrm { M S E } } \leq \| \pmb { g } _ { i } \| _ { 2 } ^ { 2 } \leq \frac { 2 \lambda _ { i } ^ { + } } { M _ { i } } \ell _ { i } ^ { \mathrm { M S E } } ,\tag{15}
$$

which is Equation 6.

## B Additional reference-model analysis

Within-architecture RFL rankings are largely insensitive to reference scale. Figure 4(a) compares Chronos-Bolt Mini, Small, and Base on the same candidate pool. Their pairwise Spearman correlations remain high even though the models differ in capacity. Moreover, we also compared the RFL scores computed with Moirai-base and Moirai-small as reference models, respectively, and the Spearman correlation between them is 0.991. This near-invariance indicates that the same architecture shares a stable notion of relative window difficulty. This agreement helps explain why the smaller reference models in Table 2 produce competitive training subsets.

Cross-architecture reuse depends on whether the reference and target families induce compatible difficulty orderings. On the Chronos-Bolt-Base candidate pool, TimesFM and Chronos-Bolt Mini retain strong window-level rank agreement $( \rho = 0 . 8 2 8$ in Figure 4(b)). Because dataset-stratified selection depends on ranks rather than absolute loss magnitudes, this monotonic agreement is sufficient for TimesFM to identify a useful Chronos subset, consistent with the successful Chronos transfer summarized in Section 4.3.

The Moirai pool illustrates a limitation of this transfer. TimesFM and Moirai-Small obtain only weak rank agreement $( \rho ~ = ~ 0 . 2 5 4$ in Figure 4(c)), and the TimesFM-selected subset underperforms both the Moirai-Small subset and full-data pretraining. A plausible explanation is that TimesFM scores point and quantile errors, whereas Moirai uses a distributional negative log-likelihood, so the two references need not assign the same difficulty to high-uncertainty windows.

(a) Chronos scale transfer  
![](images/41e9f4b22434fa98d9c18d49f05c4c35778e9c4a2345dc0ffea7ca727f7e3ec9.jpg)  
Window-level Spearman $\rho$

(b) Chronos cross-model transfer  
![](images/a5e9b5922d7075a4aef1a8d5443e32b8747eea8a727f1f859f3d62a668300a24.jpg)

(c) Moirai cross-model transfer  
![](images/b58a9c54b7cd720e392647a6bcc6bbca37ea7b6380df228a5501e8c90eac8a87.jpg)  
Figure 4: Reference-model agreement on target-specific candidate pools. (a) Window-level Spearman correlations among Chronos-Bolt Mini, Small, and Base over all candidate windows for the Chronos-Bolt-Base target. (b) TimesFM versus Chronos-Bolt Mini on the same Chronos target pool. (c) TimesFM versus Moirai-Small on the Moirai-Base target pool. Spearman coefficients use global ranks over every aligned window. For legibility, each scatter panel displays a deterministic sample of 150k windows, while the reported correlation and dashed ordinary-least-squares fit use all windows.

## C Qualitative analysis

We qualitatively inspect the Chronos-Bolt Mini RFL scores used for dataset-stratified selection. The score file contains all candidate training windows from the original sample cache. For visualization only, we form three global score regions: low RFL (bottom 1%), middle RFL (49.5–50.5%), and high RFL (top 1%). Selection in the main experiments remains dataset-stratified and the global bands here are used only to make the score semantics visually interpretable.

Figure 5 supports the intended interpretation of RFL as a difficulty coordinate. Low-score samples tend to have small residual signal after context normalization. Middle-score samples include visible trend, seasonality, or amplitude changes but remain connected to the context. High-score samples frequently show future behavior that is hard to extrapolate from the context alone, such as sudden jumps or large tail events. This motivate removing both extremes when constructing a compact pretraining subset: low-score windows can be redundant, while the high-score tail mixes rare events with weakly learnable windows.

![](images/ab08da84ee04e799f30dc8b0b84e3644936741f0192307de44cab7d788fd2777.jpg)  
blue=context, red=future; each window is normalized by context mean/std

Figure 5: Representative windows from low, middle, and high Chronos-Bolt Mini RFL regions. Blue denotes the context and red denotes the forecast horizon. Each window is standardized by its valid context mean and standard deviation. The pct value in each panel is the global empirical percentile rank of that win dow’s RFL score among all candidate windows. Low-RFL windows are often highly regular, middle-RFL windows retain learnable nontrivial structure, and high-RFL windows often contain abrupt shifts, extreme future values, or context–future mismatch.