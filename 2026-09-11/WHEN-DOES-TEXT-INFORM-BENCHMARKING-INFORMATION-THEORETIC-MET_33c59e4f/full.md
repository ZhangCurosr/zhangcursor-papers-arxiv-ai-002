# WHEN DOES TEXT INFORM? BENCHMARKING INFORMATION-THEORETIC METRICS FOR MULTI-MODAL TIME-SERIES FORECASTING

Emma Andrews National University of Singapore emma\_andrews@u.nus.edu

Gianmarco Mengaldo National University of Singapore mpegim@nus.edu.sg

## ABSTRACT

Multimodal forecasting models that combine time series with text annotations promise richer prediction through textual context, but how do we know whether a text annotation meaningfully contributes to the forecasters prediction? This is an information-theoretic question, but to evaluate whether information-theoretic metrics can reliably measure the predictive value an annotation provides, a ground truth benchmark is needed, and none currently exist. We create a synthetic time series signal with annotations in three categories: semantically correct, incorrect, and irrelevant. Because the data generation process is fully controlled, groundtruth information content is known exactly, enabling principled evaluation of six complementary mutual information estimators (KSG, MINE, InfoNCE, CCA, PID and V-information). We show that all six estimators identify correct annotations as most informative, and are able to audit the quality of mixed text corpora, choosing the annotations that result in the best downstream forecasting results without the need for model training. Our benchmark identifies limitations of each estimator, and these are validated on seven real-world datasets, which show how estimator performance differs on weak signals. Finally, we establish practical rules for implementing these metrics for annotation auditing and fusion selection.

## 1 INTRODUCTION

Multimodal time series forecasting - combining numerical signals with natural language annotations - has emerged as one of the most promising frontiers in predictive modeling. The integration of language models with time series architectures is increasingly common across domains from healthcare (Chan et al., 2024) to climate forecasting (Li et al., 2025; Kim et al., 2025), yet this progress rests on the key assumption that the text being fused actually helps.

In practice, this is rarely the case. Text may describe an upcoming regime change that the signal alone cannot anticipate, or it may be entirely wrong, or not relevant to the task at hand. A practitioner has no principled way to know, before training, which annotations are driving performance gains and which are adding noise. This is not an uncommon case, but rather a frequent situation in any real-world multimodal dataset, where text relevance to any specific future value is unknown and unverified.

Without a reliable measure of text informativeness with respect to a signal, practitioners cannot audit annotation quality before committing to expensive model training, cannot identify which domains or time periods benefit from text fusion, and cannot distinguish a model that learned from text compared to one that learned despite it. As multimodal forecasting scales to higher-stakes applications, the cost of fusing uninformative or misleading text grows accordingly.

Information-theoretic metrics offer a natural solution. Mutual Information (MI) and related measures quantify statistical dependence between text and future signal values without assuming a specific model architecture, making them ideal pre-training diagnostics: compute once on the training data to estimate if text is worth fusing before model training. The ability to do this reliably would provide a principled annotation auditing step that would transform how multimodal time series systems are built. MI estimation has already shown promise for predicting multimodal model performance (Liang et al., 2023a;b), and as training objectives to balance modality contributions (Kontras et al., 2025). But a fundamental obstacle remains, that is: it is not possible to evaluate these MI estimators in the text-time series setting, because real-world datasets never provide ground-truth information content.

All existing evaluations validate these metrics against human judgment, where the true information content is commonly unknown, creating a circular dependency. Czyz et al.˙ (2023) address this issue for a continuous unimodal numerical distribution, constructing synthetic benchmarks with known ground-truth MI, but does consider multimodal inputs such as natural language, or the extent to which modalities support or contradict each other. Furthermore, the notion of annotation quality – correct, incorrect or irrelevant – has no analogue in existing numerical distributions. Our contribution to fill this gap is MMTT-Bench: a synthetic benchmark with oracle ground truth that enables the first principled evaluation of MI estimators in a text-time series multimodal setting. Specifically:

• We construct a synthetic dataset, MMTT-Bench (Multimodal Text-Time Series Bench): A sine wave with randomly inserted constant segments, each with three labeled annotation categories (correct, incorrect, irrelevant). We validate our findings on real-world datasets, Time-MMD (Liu et al., 2024), and FinTexTS (Lee et al., 2026a). Our code and dataset is released on GitHub<sup>1</sup> and HuggingFace<sup>2</sup>.

• We evaluate six complementary MI estimators, demonstrating that information metrics predict downstream forecasting performance across a variety of model architectures, validating their use as pre-training diagnostic tools.

• We show to what extent information metrics can be used to assess the quality of text annotations across a corpus, and our results are presented as practical rules for implementing these metrics across different signals and use cases.

## 2 MMTT-BENCH DESIGN

We construct a synthetic benchmark, Multimodal Text-Time Series Bench (MMTT-Bench) that pairs a time series signal with controlled natural language annotations. Synthetic data gives us ground truth control over how much information each annotation carries about future signal values, enabling principled evaluation of information-theoretic measures across annotation categories that existing multimodal datasets do not support.

MMTT-Bench is built on a sine wave with randomly placed constant-value (flat line) segments. Specifically, the observable is

$$
y ( t ) = { \left\{ \begin{array} { l l } { \sin ( t ) } & { { \mathrm { i f ~ } } t \not \in S , } \\ { - 0 . 5 } & { { \mathrm { i f ~ } } t \in S , } \end{array} \right. }
$$

![](images/a00fde1acf70603459d366ba7e3f03aced9a1fc68705de875cd53488723f8ad8.jpg)  
Figure 1: Example text annotations in MMTT-Bench

where $s$ is a set of randomly sampled intervals

covering approximately 30% of the time axis, each of length drawn uniformly from $[ \pi / 2 , 2 \pi - \pi / 2 ]$ The flat line value of −0.5 does not coincide with any value sin(t) takes at annotations points: $y \in [ - 1 , 0 , 1 ]$ , ensuring the two regimes are unambiguous. This creates “transition states”, depicted in orange in Figure 1, where the signal drops to −0.5. At these points, the time series signal values y(t) give no deterministic information about the next value of $\bar { Y } _ { \mathrm { f u t u r e } }$ , and correct text annotations become the dominant source of predictive information.

The dataset is split chronologically into non-overlapping temporal segments (training 62.5% , validation 18.75%,testing 18.75%) to prevent data leakage and reflect realistic forecasting settings where models are trained on historical data and evaluated on future observations. Each time point retrieves three annotation variants; correct, incorrect and irrelevant text, based on the known future state of the underlying signal. Full dataset generation details are found in Appendix A.

To ensure the diversity of text is the same for the incorrect text corpus, we randomly choose these transition templates for 30% of incorrect annotation points, equaling the number of true transition points in the signal, so that the correct and incorrect text corpus are indistinguishable without the additional time series information. This is confirmed in Appendix $\mathrm { C } ,$ and prevents metrics from exploiting lexical cues rather than the semantic content and its relation to future predictions. Irrelevant text is generated from separate templates, with text describing generic properties of the signal without any directional or predictive information. A well-calibrated information metric should rank correct highest, irrelevant near zero and incorrect somewhere in-between, since 70% of incorrect annotations provide directly opposing information to correct annotations, they are still correlated with $Y _ { \mathrm { f u t u r e } }$ values. A model could learn to do exactly the opposite of what the incorrect annotations suggest, and be correct up to 70% of the time.

The core question is whether MI and related metrics can distinguish and correctly order these text variants, and if the associated metrics predict the performance gain when each annotation type is given to a forecasting model.

## 3 ESTIMATING MUTUAL INFORMATION

## 3.1 NOTATION AND BACKGROUND

We set up the task of multimodal forecasting as follows: At each annotation point t, we observe three quantities. $X _ { \mathrm { t s } }$ is the time-series input: a vector of the n preceding signal values (optionally tokenized using PatchTST (Nie et al., 2023)). $X _ { \mathrm { t e x t } }$ is the text input: the natural-language annotation at $t ,$ embedded and reduced by PCA to $d _ { \mathrm { t e x t } }$ dimensions. $Y$ is the target: the signal value at t+horizon. Every quantity we report is a mutual information (or mutual-information-like) functional of the joint distribution over $( X _ { \mathrm { t s } } , X _ { \mathrm { t e x t } } , Y )$ , estimated from a finite sample of annotation points.

For each annotation category, we compute four quantities: the marginal MI of each modality individually, $I ( X _ { \mathrm { t s } } ; Y )$ and $I ( X _ { \mathrm { t e x t } } ; Y )$ , which measure how much information each modality shares with $Y _ { \mathrm { f u t u r e } }$ , the joint MI, $I ( X _ { \mathrm { t s } } , X _ { \mathrm { t e x t } } ; Y )$ , which measures the combined information of both modalities, and the conditional MI, $I ( X _ { \mathrm { t e x t } } ; Y | X _ { \mathrm { t s } } ) = I ( X _ { \mathrm { t s } } , X _ { \mathrm { t e x t } } ; Y ) - I ( X _ { \mathrm { t s } } ; Y )$ that isolates the unique contribution of text beyond the time series. This is our primary quantity of interest, because it is the quantity a practitioner actually needs before deciding whether fusing text is worth the cost.

But why must MI be estimated? For continuous, high-dimensional variables with unknown densities, mutual information has no closed form: computing it exactly would require integrating over the true joint density $p ( X _ { \mathrm { t s } } , X _ { \mathrm { t e x t } } , Y )$ , which is never available in practice. This is true even in our synthetic setting, despite the information content of an annotation being known by construction, the mapping from raw text through an embedding model and PCA to a usable representation is not analytically tractable. Every number we report is therefore an estimate with its own bias and variance properties, not a ground truth value. This is precisely why an oracle benchmark that separates estimator error from unknown-true-value uncertainty is necessary. MI estimation is specifically difficult in the text–time-series setting, motivating the need to quantify the performance and limitations of different estimators.

## 3.2 ESTIMATORS

We evaluate six complementary MI estimators spanning the full space of approaches to measuring text-time series information content. K-Nearest-Neighbor Estimator (KSG) (Kraskov et al., 2004), Mutual Information Neural Estimator (MINE) (Ishmael Belghazi et al., 2018), InfoNCE (Rusak et al., 2024) and Canonical Correlation Analysis Estimator (CCA) (Murphy, 2023) all target the same Shannon MI quantity $I ( X ; Y )$ in nats, and are directly comparable in scale but differ in assumptions and practical behavior. Partial Information Decomposition (PID) (Williams & Beer, 2010) decomposes the joint information directly into non-negative redundancy, unique, and synergy atoms, so the conditional quantity of interest is its Unique $( X _ { \mathrm { t e x t } } )$ atom rather than a subtraction. V-Usable Information (V-Information) (Xu et al., 2020) departs from Shannon entropy entirely, measuring the information a Ridge regression can exploit from the given embedding, reported as $\Delta R ^ { 2 }$ , the gain in explained variance from adding $X _ { \mathrm { t e x t } }$ to a fixed predictive family. Full derivations and implementation details are provided in Appendix B.

These estimators all have limitations which we aim to quantify in the text-time series settings. Estimator variance and the required sample size both scale with the magnitude of the quantity being estimated, so weak-signal regimes where text contributes limited additional information are intrinsically harder to estimate. This is especially problematic for KSG, which estimates density locally via nearest-neighbor distances, and therefore becomes unreliable once the joint dimensionality d exceeds exceeds $\sqrt { N / 2 }$ for sample size N (Gao et al., 2017). This motivates dimensionality reduction of $X _ { \mathrm { t e x t } } ,$ (see Section 3.4). Variance for KSG is estimated using the recommended procedure of Holmes & Nemenman (2019) rather than naive bootstrapping, which is used for all other estimators.

MINE and InfoNCE learn a critic network by gradient descent and are insensitive to dimensionality, but require hyperparameter tuning and sufficient signal strength to train reliably. They also produce bounds rather than true estimators. MINE is a lower bound obtained via the Donsker-Varadhan representation (Ishmael Belghazi et al., 2018), and InfoNCE is a lower bound that saturates at logN nats for batch size N (Rusak et al., 2024). Both can therefore systematically underestimate true MI.

Estimating a conditional as a difference of two separately biased estimates can leave a residual larger than the quantity being measured. Because each term carries its own estimator-specific bias, the two biases do not cancel in general. When the marginal $\hat { I } ( X _ { \mathrm { t s } } ; Y )$ is large relative to the true conditional MI, this bias can dominate the difference and produce a negative estimate for a quantity that is non-negative by definition. This can cause negative KSG conditional MI values, and is one advantage of CCA, whose chain rule holds exactly under its Gaussian assumption, so its conditional estimates do not exhibit this inconsistency (Czyz et al.˙ , 2023).

## 3.3 RELATED WORK

The use of mutual information to characterize multimodal data has gained significant traction across two complementary directions; as an analytical tool to predict model behavior before training, and as a training objective to improve model performance.

On the analytical side, Liang et al. (2023a) introduce a PID framework and show it can not only characterize dataset structure but also predict the performance of downstream multimodal models and guide model selection, all without training a single model. A concurrent line of work by Liang et al. (2023b) extends this in a semi-supervised setting. However, both approaches validate estimation quality against human judgment on real-world datasets where the true data generation process is unknown, creating a circular dependency between what PID captures and what humans intuitively expect it to mean. Our benchmark addresses this directly by providing oracle ground truth through control of the data generation process, so the information-theoretic quantities can be validated without recourse to human annotation.

On the training side, Kontras et al. (2025) introduce the Multimodal Competition Regularizer (MCR), which uses a single conditional MI estimate as a training signal to adaptively balance modality contributions, encouraging each modality to maximize its unique predictive role. MCR optimizes this quantity during training rather than evaluating it, and uses a single estimate per training step rather than comparing estimators. Since the quantities MCR relies on - unique and shared modality contributions - are identical to those our benchmark measures, our results directly inform which estimator MCR should use and under what signal conditions its loss can be trusted.

Wang et al. (2026), similarly uses MI as an online training diagnostic during reinforcement learning of LLM agents, finding that mutual information between input prompts and generated reasoning traces correlates with final task performance much more strongly than entropy alone. These applications demonstrate the practical demand for reliable MI estimation, yet both validate their MI-based signals based on model performance alone, when true information content is unknown. Without a benchmark that provides an oracle ground truth for MI estimation, there is no principled way to measure if model performance improvements are driven by accurate MI estimation or occur in spite of it.

A complementary perspective is offered by Lee et al. (2026b) and Zhang et al. (2025), who show empirically that naive multimodal fusion of text and time series frequently under performs unimodal baselines due to uncontrolled integration of irrelevant textual information. This motivates the need for principled measurement of textual relevance before fusion. This also demonstrates a problem with existing multimodal time series benchmarks: they do not provide the ground truth labels or controlled annotation quality necessary to evaluate information-theoretic metrics.

![](images/84b95871fbb31722eaabe28d4018cfb59c9bc7249c01e41967c230b891130640.jpg)  
Figure 2: Results of parameter sweeps on MMTT-Bench. Left plot shows how PCA dimension and lookback steps affect MI ordering. Right plot shows conditional MI estimates for correct text annotations using different text embedding models.

While Time-MMD (Liu et al., 2024) initially demonstrated adding text achieves MSE reduction, more recent work achieves better forecasting results using only the time series, making us question if the text is truly informative (Zhang et al., 2025). FinTexTS (Lee et al., 2026a) claims to improve the text quality of FNSPID (Dong et al., 2024), a stock price dataset that is paired with news articles, by replacing the keyword-based matching strategy with a semantic and multi-level pairing framework. It claims the new pairings result in better forecasting performance than no text and original paired text baselines (Lee et al., 2026a). Context is Key (Williams et al., 2025) provides human-authored context descriptions where the text is strictly necessary to predict the future signal, however it contains a relatively limited number of annotated points, 400 across 14 tasks, making it too small for reliable MI estimation. None of these benchmarks provide the structure of correct, incorrect and irrelevant annotations that are necessary to establish ground truth for information-theoretic measures of tex quality.

The closest precedent is Czyz et al.˙ (2023), who construct a diverse family of synthetic distributions with analytically known ground-truth MI and uses them to systematically benchmark estimator accuracy. We adopt the same principle, with the key distinction is that Czyz et al.˙ (2023) operate on pairs of continuous random numerical variables with no semantic structure. Our benchmark, instead, introduces natural language as one modality. The extension is non-trivial, text must be embedded before MI estimation, introducing a representational bottleneck that does not exist in purely numeric settings, and the notion of annotation quality – correct, incorrect or irrelevant – has no analogue in continuous synthetic distributions.

## 3.4 REPRESENTATION PIPELINE

To ensure the estimator comparisons are robust to different representation pipeline choices, Figure 2 compares how the PCA dimension and number of lookback steps affect the ordering of MI estimates across text categories, where we expect correct > incorrect > irrelevant. As we expect MI to be close to zero, we define an estimate as negative if the mean across seeds if less than zero and the magnitude is greater than the bootstrap standard deviation of that estimator. KSG collapses to produce negative values as it approaches $\sqrt { N / 2 } \approx 2 5$ bound as predicted, hence we choose $d _ { \mathrm { t e x t } } = 1 6 , n _ { \mathrm { l o o k b a c k } } \leq 2$ as our final parameters. Lowering PCA dimension further causes KSG and MINE to estimate MI for irrelevant text above incorrect, which we know not to be true, therefore we assume reducing the text this dramatically loses information that helps differentiate between these two text annotations.

Figure 2 also shows how conditional MI remains relatively stable across all estimators when varying the text encoder. This holds across eight embedding models spanning sentence transformers and LLM encoders. We also include an ’oracle’ embedding which simply includes the next $Y _ { \mathrm { f u t u r e } }$ value, representing an upper bound if the correct text information was embedded perfectly. The ordering of MI estimates across different text categories is preserved in all cases. DistilRoBERTa is selected because it balances high $I ( X _ { \mathrm { t e x t } } ; Y )$ relative to $\tilde { I ( X _ { \mathrm { t s } } ; Y ) }$ for low computational cost, indicating it best preserves semantic content relevant to $Y _ { \mathrm { f u t u r e } }$ within the PCA-reduced representation.

![](images/3aa6b61a237a940a2f362db23bb6b8fb7cdfafed1d210c9b4a170a08f3a8de98.jpg)  
Figure 3: Information Metrics measuring text contribution compared to change in model performance between $X _ { j o i n t }$ and $X _ { t s }$ for MMTT-Bench

## 4 RESULTS

In our results, corpus text quality is in the independent variable, constructed as three categories, correct, incorrect, and irrelevant. The MI metric is the first dependent value, does the estimator’s estimate change with corpus quality? This is our estimator evaluation. Finally downstream model forecasting performance, measured through mean squared error (MSE), is the second dependent variable. This is only presented to establish that the ground-truth quality ordering is consequential; a corpus labeled as better really does train a better model. We train fourteen model architectures and, where appropriate, ten different fusion mechanisms per architecture, so this result does not depend on a single model’s inductive biases.

Figures 3 displays model forecasting performance (MSE) on the y-axis and conditional MI estimates of $X _ { \mathrm { t e x t } }$ for each information metric on the x-axis. Conditional MI of $X _ { \mathrm { t e x t } }$ represents the unique information the text annotations contribute to $Y _ { \mathrm { f u t u r e } }$ predictions. For KSG, MINE, InfoNCE and CCA this is $I ( X _ { \mathrm { t e x t } } \to Y | X _ { \mathrm { t s } } )$ , for PID it is Unique $X _ { \mathrm { t e x t } } )$ , and for V-Information it is the change in ridge regression performance $\Delta R ^ { 2 }$ when trained with $X _ { \mathrm { t e x t } }$ and $X _ { \mathrm { t s } }$ minus the performance when trained only on $X _ { \mathrm { t s } }$

## 4.1 MMTT-BENCH

Metrics were computed on the training split consisting of N = 1279 annotation points, each with correct, incorrect and irrelevant text. Full parameter choices and sweep results are reported in Appendix E, and results for every metric quantity, not just conditional MI, are in Appendix F.

The y-axis in Figure 3 shows the model performance across nine model architectures. For simplicity we only included the best performing models, with the four time series transformers using a basic additive fusion method. Appendix F provides full results, including a further four transformer archi tectures and 10 different fusion strategies. Firstly, focusing on the MI estimates across different text categories, Figure 3 confirms that, on a well-designed benchmark with sufficient data and appropriate dimensionality, correct text is reliably identified as the most informative category. However, KSG and PID assign near-identical values to incorrect and irrelevant, despite it being possible to separate them by design. All other estimators rank incorrect > irrelevant with statistical significance, and we note that larger error bars are expected when bootstrapping neural estimators MINE and InfoNCE because it reflects both sampling uncertainty and training stochasticity.

This rank reflects a genuine property of the data. Incorrect annotations describe the opposite future direction to the true signal for 70% of points, and therefore carries real statistical dependence with $Y _ { \mathrm { f u t u r e } }$ under an inverted mapping, while irrelevant text encodes no dimension systematically related to $Y _ { \mathrm { f u t u r e } }$ in any direction. This is validated in embedding space, where irrelevant annotations form a distinct cluster, and correct and incorrect overlap almost completely (see Figure 6, Appendix C). Neural estimators detect this to a greater extent because their critics learn $p ( Y \mid \bar { X } _ { \mathrm { t e x t } } )$ directly through contrastive scoring, capturing any consistent mapping regardless of sign; non-neural estimators summarize averaged co-variation, making it harder to resolve the distinction.

In Appendix F.3.1 we confirm that the incorrect signal is carried by the text-time series pairing rather than the text alone by shuffling incorrect annotations across time points and recomputing all metrics. MINE and InfoNCE both shows a significant drop in MI estimates of ≈ 80% for correct text and ≈ 30% for incorrect text after shuffling, with no change in irrelevant text, confirming that the inverted signal is a consistent property of the (text, $Y _ { \mathrm { f u t u r e } } )$ pairing and not an artifact of embedding distribution.

Figure 3 shows a consistent positive relationship between metric value and downstream performance across all six estimators, with trend lines well-separated by model baseline but aligned in slope. The time series only baseline (blue markers) anchors the left of each panel at zero MI from the addition of $X _ { \mathrm { t e x t } }$ by default, clearly showing the metric values correctly attribute performance gains to text rather than time series.

Figure 4 extends this to the full quality spectrum by varying the fraction of correct annotations from 0% to 100%, replacing the remainder with either incorrect or irrelevant text. Under irrelevant replacement (right panels), both metric values and MSE change monotonically across all estimators and architectures, confirming that metrics are well-calibrated continuous sensors of corpus quality. Under incorrect replacement (left panels), model performance shows a Ushape: at 0% correct, entirely incorrect annotations still sustain partial model performance via the inverted signal, before degrading as the mix becomes ambiguous and recovering as correct annotations dominate. Metric values mirror this

![](images/2456f312b700baea490b9d3db76ec273dc50380ee696451d22cbb08f456e3f17.jpg)  
Figure 4: Model Performance (top) and Information Metrics (bottom) compared to Percentage of Correct Annotations

pattern, demonstrating that information metrics track the true predictive value of a corpus even in cases, such as the inverted signal, that human annotation quality We use the mixture dataset’s nine corpora of varying qualities to demonstrate that conditional MI estimates can be used as a pre-training audit score to select the corpus which, when used for training, results in the best performance. Across the mixture corpora and five simple downstream models, Spearman $| \rho |$ between conditional MI and MSE is 0.849 on average, 28/36 estimator-model pairs exceed $| \rho | > 0 . 8$ and 32/36 are significant with $p < 0 . 0 5$ . CCA is the highest and best auditor (0.90-1.00) and KSG lowest (0.40-0.73). Every estimator’s top MI is the 100% correct corpus, which is also the best model performance across all models, giving a performance boost of −0.114 MSE compared to selecting text annotations at random from correct, incorrect and irrelevant, and beating the no text baseline. However as selecting a clean corpus is not a hard test, we remove the corpora with all annotations from the same category, and every estimator selects the corpus contaminated with 25% irrelevant text, which is also the best model performance for 5 out of 6 models, and gains −0.07 MSE over a random selection. In Appendix F.3 we further demonstrate these results hold when adding noise to the sine signal and introducing controlled perturbations in the form of temporal jitters, where each annotation is replaced by the correct text of a nearby time point.

## 4.2 REAL-WORLD VALIDATION

To investigate whether the information-theoretic findings from our synthetic benchmark transfer to real-world data, we extend our evaluation to seven domains from Time-MMD (Liu et al., 2024) and ten stocks from FinTexTS (Lee et al., 2026a). Appendix G shows that these span trend, seasonality, non-stationary and multivariate structure, everything the sine signal lacks.

Baseline time-series predictability is substantially higher here for Social Good, Public Health and Energy on time series alone $( R ^ { 2 }$ in [0.65-0.93[)] compared to $0 . 3 7$ on MMTT-Bench. This limits the room available for text to contribute and makes this a meaningfully different, and harder, regime than the synthetic setting. Despite this, every estimator pathology characterized under controlled conditions on MMTT-Bench reappears here: KSG’s conditional MI turns negative where the marginal-to-conditional ratio is large, MINE turns negative and InfoNCE inflates under weak signal, and CCA, V-information and PID remain stable and correctly ordered throughout.

![](images/eb5a6113b2aee86bd9793be9a77347efa0b94d6fcd724eff3a098235e9a5d1b9.jpg)  
Figure 5: MI Estimates on seven real-world datasets

Similar to the audit test carried out on MMTT-Bench, we also seek to evaluate estimators for practical applications. One example is rather than simply identifying whether text is informative in aggregate, a practitioner also needs to know whether a given annotation is trustworthyfor the timestamp it is attached to. We distinguish these with two bootstrapped contrasts (20 resamples; $z = \Delta / \sigma ) \mathrm { : }$ the conditional MI to measure whether text is informative as before, and an alignment contrast between correctly-paired text and same-domain text drawn from the wrong timestamp, which isolates whether the pairing carries information. Across the seven datasets, conditional MI is positive and significant while the alignment contrast is statistically indistinguishable from zero for most estimator–dataset pairs (Appendix G.2, Table 17): text in these corpora carries real information, but that information is largely diffuse and topical rather than tied to the specific date it is attached to - despite every one of these datasets being constructed on the assumption that it is date-specific. Downstream model performance in Appendix G.2, Table 18 corroborates this: when adding text, median MSE gets worse for every dataset except Climate, and on Agriculture and Public Health, irrelevant text degrades performance less than correct text. This is the empirical basis for treating conditional MI and the alignment contrast as answering two different questions.

## 5 RECOMMENDATIONS

1. Default audit pair: CCA and V-information. They are deterministic given a resample, require no training, are scale-invariant, and satisfy the chain rule (exactly for CCA under the Gaussian assumption), never producing a negative conditional MI. On the mixture corpora their conditional MI rank-correlates with downstream MSE at $| \rho | = 0 . 9 0 \substack { - 1 . 0 0 }$ across five model families, the highest of any estimator.

2. Use neural estimators (MINE, InfoNCE) only when N is large and the expected conditional MI is substantial. These estimators reliably separated incorrect from irrelevant text, because their critics learn $p ( Y | X _ { \mathrm { t e x t } } )$ directly and are therefore sensitive to sign-inverted dependence, which averaged co-variation summaries cannot see. But on real world data, where conditional MI is small, MINE collapses to near-zero and InfoNCE shows very high variance, even after hyperparameter tuning. This shows they are unreliable when the signal provided by the text is weak, or N is small.

3. Do not use KSG for conditional MI when the marginal $I ( X _ { \mathbf { t s } } ; Y )$ is large. Because the condi tional is a difference of two separately-biased estimates and KSG’s bias depends on dimensionality, a large marginal makes the residual dominate, producing the negative conditional values visible in our real-world results.

4. Use PID only when Y has interpretable discrete structure. It is the most precise estimator we tested on clean data (essentially zero unique information for irrelevant text) but requires discretizations, and on continuous real targets the binning choice becomes a free parameter that materially changes the answer.

5. Prefer paired contrasts over absolute values. Comparing conditional MI for a corpus against the same estimator’s value for a shuffled or misaligned version of that corpus cancels the estimator’s bias, whereas comparing an absolute conditional MI against zero does not. While conditional MI values alone can help to decide whether to fuse text, this alignment contrast can be used to decide whether the text-time series pairing is trustworthy. We demonstrated this for Time-MMD and FinTexTS, where text carried measurable information that is not tied to its timestamp.

## 6 LIMITATIONS

Synthetic signal design. MMTT-Bench uses a sine wave with externally imposed constant segments, which creates an unusually clean separation between informative and uninformative annotation points. This controlled design is a deliberate choice to enable oracle ground truth that real-world datasets cannot provide: without knowing the true data generation process, there is no principled way to assess whether an estimator is correct. MMTT-Bench fills this gap precisely because it is synthetic. We validate that our estimator evaluation results hold when noise is added to the sine signal, and on seven real-world datasets which, as shown in Appendix G, cover the full spectrum of time series characteristics.

Representation Pipeline As outlined in section 3.4, the MI ordering holds between text annotations when varying both the number of lookback steps and PCA text dimensions, and the exact values are also relatively stable across eight different text embedding models. An embedding model fine-tuned on text and time series pairs may perform better, but would make it impossible to isolate $I ( X _ { \mathrm { t e x t } } ; Y \mid X _ { \mathrm { t s } } )$ the unique contribution of the text without time series, and simple text embedding models represent the realistic deployment setting for annotation auditing in practice, where computational cost constrains embedding choice.

Neural estimator instability on weak signals. MINE and InfoNCE require sufficient signal strength to train a reliable critic. On real-world datasets, where conditional MI is small, MINE collapses to near-zero or negative estimates, whereas InfoNCE inflates, and both show high variance, making them unreliable in low-gain settings. Hyperparameter optimization was performed across different batch sizes and learning rates, with results in Appendix E,G.1, but with only marginal improvements, demonstrating the fundamental sensitivity to signal strength is a structural limitation of critic-based estimation.

PID discretizations. Discrete Y-binning into unique values is semantically motivated for the sine signal in MMTT-Bench, where only four Y values are present, but no natural partition exists for continuous real-world targets such as those in Time-MMD. This limits PID to signals with interpretable discrete structure in the target variable, or requires arbitrary binning choices that introduce sensitivity to hyperparameters. Appendix G.1.1 goes into details of different PID parameters tested.

## 7 CONCLUSION

We compare six mutual information estimators on MMTT-Bench, the first synthetic benchmark for evaluating information-theoretic metrics in text-time series multimodal forecasting. CCA, V-Information and PID are able to identify correct annotations as more informative on both MMTT-Bench and real-world datasets, whereas neural estimators (MINE, InfoNCE) are unreliable in settings with low conditional MI. The pre-training correlation between metric values and downstream MSE across multiple model architectures validates their use as an annotation auditing tools and fusion selection diagnostics.

These findings lead us to present seven recommendations for when and how to utilize MI estimators in multimodal settings. We release MMTT-Bench generation code, estimator implementations, model training pipelines, and the dataset itself to support this research.

## ETHICS STATEMENT

This work introduces tools for evaluating the quality of textual annotations in multimodal time series forecasting before any model is trained. The primary societal benefit is improved reliability and transparency in domains where text-augmented forecasting is consequential: public health, energy demand forecasting, financial risk assessment and environmental monitoring. In adversarial settings, an actor with knowledge of which annotations carry high MI with a target variable could craft annotations that appear informative to MI estimators while encoding misleading signals - the incorrect annotation category studied in this paper demonstrates that anti-correlated text can score above genuinely irrelevant text, a property that could be exploited deliberately. We consider these risks low in the near term, as the methods require access to the time series and future values to compute MI, limiting their applicability to retrospective rather than prospective manipulation. Broader misuse of multimodal forecasting systems in high-stakes automated decision-making remains a concern independent of this work, and we encourage practitioners to treat MI metrics as one component of a broader human-in-the-loop quality assurance process.

All data is synthetically generated and contains no personal, sensitive or proprietary information. The annotation templates and word banks are constructed to described abstract signal properties only, and carry no cultural, political, or demographic content. The Time-MMD and FinTexTS datasets are publically available and used in accordance with their terms.

## REPRODUCIBILITY STATEMENT

Regarding our synthetic dataset MMTT-Bench, we release the full generation code alongside the dataset on both GitHub<sup>3</sup> and HuggingFace<sup>4</sup> so that the construction process is fully auditable and the dataset can be regenerated, modified, or extended by the community. Appendix A goes into details about how the dataset was generated, including the templates for all text annotations.

Our mutual information estimation implementation is also released as part of our code repository, with full mathematical details available in Appendix B and estimator parameters recorded per dataset in Appendix E and G.1. Our training code for all fourteen model architectures is also reproducible from our code base, with training parameters and compute requirements detailed in Appendix D.

To generate all results in this paper, a single script is provided in our code repository. Further scripts enable regeneration of all tables and plots. As MMTT-Bench is available within the repository itself, these experiments can be reproduced by anyone in a single command. For MMTT-Bench signal perturbations and real-world datasets, we provide references to the exact source data where appropriate and data preprocessing scripts, as well as another a ’additional experiments’ script to run all experiments and generate results.

## ACKNOWLEDGMENTS

We thank Keane Ong for his help and advice throughout the process of writing this paper.

## REFERENCES

Nils Bertschinger, Johannes Rauh, Eckehard Olbrich, Jürgen Jost, and Nihat Ay. Quantifying unique information. Entropy, 16(4):2161–2183, 2014. ISSN 1099-4300. doi: 10.3390/e16042161. URL https://www.mdpi.com/1099-4300/16/4/2161.

Nimeesha Chan, Felix Parker, William Bennett, Tianyi Wu, Mung Yao Jia, James Fackler, and Kimia Ghobadi. Medtsllm: Leveraging llms for multimodal medical time series analysis. arXiv preprint arXiv:2408.07773, 2024.

Paweł Czyz, Frederic Grabowski, Julia Vogt, Niko Beerenwinkel, and Alexander Marx. Beyond ˙ normal: On the evaluation of mutual information estimators. In Advances in Neural Information Processing Systems, volume 36, pp. 16957–16990, 2023.

Zihan Dong, Xinyu Fan, and Zhiyuan Peng. Fnspid: A comprehensive financial news dataset in time series. In Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pp. 4918–4927, 2024.

Monroe D Donsker and SR Srinivasa Varadhan. Asymptotic evaluation of certain markov process expectations for large time. iv. Communications on pure and applied mathematics, 36(2):183–212, 1983.

Weihao Gao, Sewoong Oh, and Pramod Viswanath. Demystifying fixed k-nearest neighbor information estimators. In 2017 IEEE International Symposium on Information Theory (ISIT), pp. 1267–1271, 2017. doi: 10.1109/ISIT.2017.8006732.

Caroline M Holmes and Ilya Nemenman. Estimation of mutual information for real-valued data with error bars and controlled bias. Physical Review E, 100(2):022404, 2019.

Mohamed Ishmael Belghazi, Aristide Baratin, Sai Rajeswar, Sherjil Ozair, Yoshua Bengio, Aaron Courville, and R Devon Hjelm. Mine: mutual information neural estimation. arXiv e-prints, pp. arXiv–1801, 2018.

Jongseon Kim, Hyungjoon Kim, HyunGi Kim, Dongjun Lee, and Sungroh Yoon. A comprehensive survey of deep learning for time series forecasting: architectural diversity and open challenges. Artificial Intelligence Review, 58(7):216, 2025.

Konstantinos Kontras, Thomas Strypsteen, Christos Chatzichristos, Paul Pu Liang, Matthew B. Blaschko, and Maarten De Vos. Balancing multimodal training through game-theoretic regularization. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. URL https://openreview.net/forum?id=auiURbhoYx.

Alexander Kraskov, Harald Stögbauer, and Peter Grassberger. Estimating mutual information. Phys. Rev. E, 69:066138, Jun 2004. doi: 10.1103/PhysRevE.69.066138. URL https://link.aps. org/doi/10.1103/PhysRevE.69.066138.

Jaehoon Lee, Suhwan Park, Taeyoon Lim, Seunghan Lee, Jun Seo, Dongwan Kang, Hwanil Choi, Minjae Kim, Sungdong Yoo, SoonYoung Lee, et al. Fintexts: Financial text-paired time-series dataset via semantic-based and multi-level pairing. arXiv preprint arXiv:2603.02702, 2026a.

Seunghan Lee, Jun Seo, Jaehoon Lee, Sungdong Yoo, Minjae Kim, Tae Yoon Lim, Dongwan Kang, Hwanil Choi, SoonYoung Lee, and Wonbin Ahn. Rethinking multimodal fusion for time series: Auxiliary modalities need constrained fusion. arXiv preprint arXiv:2603.22372, 2026b.

Haobo Li, Zhaowei Wang, Jiachen Wang, YueYa Wang, Alexis Kai Hon Lau, and Huamin Qu. Cllmate: A multimodal benchmark for weather and climate events forecasting. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pp. 17547–17573, 2025.

Paul Pu Liang, Yun Cheng, Xiang Fan, Chun Kai Ling, Suzanne Nie, Richard Chen, Zihao Deng, Faisal Mahmood, Ruslan Salakhutdinov, and Louis-Philippe Morency. Quantifying & modeling multimodal interactions: An information decomposition framework. In Advances in Neural Information Processing Systems, 2023a.

Paul Pu Liang, Chun Kai Ling, Yun Cheng, Alexander Obolenskiy, Yudong Liu, Rohan Pandey, Alex Wilf, Louis-Philippe Morency, and Russ Salakhutdinov. Multimodal learning without labeled multimodal data: Guarantees and applications. In The Twelfth International Conference on Learning Representations, 2023b.

Haoxin Liu, Shangqing Xu, Zhiyuan Zhao, Lingkai Kong, Harshavardhan Prabhakar Kamarthi, Aditya Sasanur, Megha Sharma, Jiaming Cui, Qingsong Wen, Chao Zhang, et al. Time-mmd: Multi-domain multimodal dataset for time series analysis. Advances in Neural Information Processing Systems, 37:77888–77933, 2024.

Kevin P Murphy. Probabilistic machine learning: Advanced topics. MIT press, 2023.

Yuqi Nie, Nam H. Nguyen, Phanwadee Sinthong, and Jayant Kalagnanam. A time series is worth 64 words: Long-term forecasting with transformers. In International Conference on Learning Representations, 2023.

Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748, 2018.

Ethan Perez, Florian Strub, Harm de Vries, Vincent Dumoulin, and Aaron C. Courville. Film: Visual reasoning with a general conditioning layer. In AAAI, 2018.

Evgenia Rusak, Patrik Reizinger, Attila Juhos, Oliver Bringmann, Roland S Zimmermann, and Wieland Brendel. Infonce: Identifying the gap between theory and practice. arXiv preprint arXiv:2407.00143, 2024.

Zihan Wang, Chi Gui, Xing Jin, Qineng Wang, Licheng Liu, Kangrui Wang, Shiqi Chen, Linjie Li, Zhengyuan Yang, Pingyue Zhang, et al. Ragen-2: Reasoning collapse in agentic rl. arXiv preprint arXiv:2604.06268, 2026.

Andrew Robert Williams, Arjun Ashok, Étienne Marcotte, Valentina Zantedeschi, Jithendaraa Subramanian, Roland Riachi, James Requeima, Alexandre Lacoste, Irina Rish, Nicolas Chapados, et al. Context is key: A benchmark for forecasting with essential textual information. In International Conference on Machine Learning, pp. 66887–66944. PMLR, 2025.

Paul L. Williams and Randall D. Beer. Nonnegative decomposition of multivariate information, 2010. URL https://arxiv.org/abs/1004.2515.

Yilun Xu, Shengjia Zhao, Jiaming Song, Russell Stewart, and Stefano Ermon. A theory of usable information under computational constraints. arXiv preprint arXiv:2002.10689, 2020.

Xiyuan Zhang, Boran Han, Haoyang Fang, Abdul Fatir Ansari, Shuai Zhang, Danielle C Maddix, Cuixiong Hu, Andrew Gordon Wilson, Michael W Mahoney, Hao Wang, et al. When does multimodality lead to better time series forecasting? arXiv preprint arXiv:2506.21611, 2025.

## A DATASET GENERATION DETAILS

To generate the dataset, or inspect the full code, we make the code available at:https://github. com/MathEXLab/MMTT-Bench

First a sine wave of 512 periods is generated, with annotation points every $\pi / 4 .$ A random mask of periods with length between $[ \pi / 4 , \breve { 7 } \pi / 4 ]$ and total length equal to 30% of the full signal is applied, with all masked values set to -0.5, creating “transition” regions. The final signal is then divided into train (62.5%), validation (18/75%) and test (18/75%) and saved as separate files.

Annotation points are placed at quarter-cycle intervals $\Delta t = \pi / 2$ to align with sine-wave phases, which produces a sparse dataset, realistic with real-world scenarios where human annotation is expensive. Each point in assigned a phase based on it’s position in the sine period, peak, descending\_zero, trough and ascending\_zero, and they are used as the ground truth so text annotations know the exact phase of the next point being predicted. These points are assigned phases drop\_to\_constant for annotation points starting immediately before the signal transitions and continuing up until the annotation point immediately before the signal resumes oscillation, which is assigned the recover\_from\_constant phase.

Correct, incorrect and irrelevant text annotations are generated at each point. For non-transition regions where the signal follows a standard sine wave,the same templates are used for both correct and incorrect text, with all time words being associated to the future and direction and magnitude choices based on the gradient at that annotation point, where steep is defined as gradients exceeding ±0.7. These ‘non-transition‘ templates can be found below.

## Listing 1: MMTT-Bench, Non-transition templates

TEMPLATES = [ # $a d \nu e r b - m e d i a l :$

```csv
" The { s i g n a l } { time_w } { v e r b } { a d v e r b } . " ,
# i mp e r a t i v e − l i k e :
" E x p e c t t h e { s i g n a l } t o { v e r b } { a d v e r b } . " ,
# fr o n t e d t i m e c l a u s e :
" { t i m e _ c l a u s e } , t h e { s i g n a l } { t i m e _ w } { v e r b } { a d v e r b } . " ,
# nominal s u bj e c t :
"A { a dj } { noun } { time_w } c h a r a c t e r i s e t h e s i g n a l . " ,
# p a s s i v e p e r c e p t i o n :
" The { s i g n a l } { t i m e _ w } b e o b s e r v e d t o b e { g e r u n d } { a d v e r b } . " ,
# e x i s t e n t i a l :
" There { time_w } be a { a dj } { noun } i n t h e s i g n a l . " ,
# nominal p r e d i c a t e :
" The { s i g n a l } ’ s b e h a v i o u r { t i m e _ w } t a k e a { a d j } { n o u n } . " ,
# p a r t i c i p i a l a b s o l u t e :
" { g e r u n d } { a d v e r b } , t h e { s i g n a l } { t i m e _ w } c o n t i n u e p a s t t h i s p o i n t . " ,
# fr o n t e d c l a u s e + nominal :
" { t i m e _ c l a u s e } , a { a d j } { n o u n } i s e v i d e n t . " ,
# wh−n o m i n a l s u b j e c t :
" What c h a r a c t e r i s e s t h e { s i g n a l } { time_w } be i t s { a dj } { noun } . " ,
]
SIGNAL_WORDS = [
" s i n e wave " , " s i n u s o d i a l s i g n a l " , " s i n u s o i d " ,
" s i m p l e h a r m o n i c m o t i o n s i g n a l " , " s i n e c u r v e " ,
" s i n u s o i d w a v e fo r m " , " s i n u s wave " , " f l u x s i g n a l " ,
" p u r e t o n e " , " h a r m o n i c wave "
]
TIME_WORDS = {
" f u t u r e " : [ " w i l l " , " i s a b o u t t o " , " i s e x p e c t e d t o " ,
" w i l l so o n " , " i s g o i n g t o " ] ,
}
TIME_CLAUSES = {
" f u t u r e " : [ " Going fo r w a r d " , " From t h i s p o i n t on " ,
" I n t h e n e a r t e r m " , " L o o k i n g a h e a d " ] ,
}
DIRECTION = {
" i n c r e a s i n g " : {
" v e r b " : [ " r i s e " , " i n c r e a s e " , " c l i m b " , " a s c e n d " , " grow " ] ,
" noun " : [ " r i s e " , " i n c r e a s e " , " a s c e n t " , " c l i m b " , " u p w a r d movement " ] ,
" g e r u n d " : [ " r i s i n g " , " i n c r e a s i n g " , " c l i m b i n g " , " a s c e n d i n g " , " g r o w i n g " ] ,
} ,
" d e c r e a s i n g " : {
" v e r b " : [ " f a l l " , " d e c r e a s e " , " d r o p " , " d e s c e n d " , " d e c l i n e " ] ,
" noun " : [ " f a l l " , " d e c r e a s e " , " d e s c e n t " , " d r o p " , " downward movement " ] ,
" g e r u n d " : [ " f a l l i n g " , " d e c r e a s i n g " , " d r o p p i n g " , " d e s c e n d i n g " , " d e c l i n i n g " ]
} ,
}
MAGNITUDE = {
" s t e e p " : {
" a d v e r b " : [ " s t e e p l y " , " s h a r p l y " , " r a p i d l y " ,
" s i g n i f i c a n t l y " , " s u b s t a n t i a l l y " ] ,
" a d j " : [ " s t e e p " , " s h a r p " , " r a p i d " , " s i g n i f i c a n t " , " s u b s t a n t i a l " ] ,
} ,
" s h a l l o w " : {
" a d v e r b " : [ " g e n t l y " , " g r a d u a l l y " , " s l o w l y " , " m o d e s t l y " , " s l i g h t l y " ] ,
```

" a dj " : [ " g e n t l e " , " g r a d u a l " , " s l ow " , " mod est " , " s l i g h t " ]   
} ,   
" n e u t r a l " : {   
" a d v e r b " : [ " " , " n o t i c e a b l y " , " m e a s u r a b l y " ] ,   
" a d j " : [ " " , " n o t i c e a b l e " , " m e a s u r a b l e " ] ,   
} ,   
}   
IRRELEVANT\_TEMPLATES = [   
" The s i g n a l { p h r a s e } . "   
" T h i s f u n c t i o n { p h r a s e } . " ,   
" The waveform { p h r a s e } . "   
" N o t a b l y , t h i s s i g n a l { p h r a s e } . " ,   
" I n t e r m s o f i t s g l o b a l p r o p e r t i e s , t h e s i g n a l { p h r a s e } . " ,   
" What c a n b e o b s e r v e d i s t h a t t h e waveform { p h r a s e } . "   
"A f e a t u r e w o r t h n o t i n g i s t h a t t h e s i g n a l { p h r a s e } . " ,   
" I t i s t h e c a s e t h a t t h i s f u n c t i o n { p h r a s e } . " ,   
" The measured s i g n a l { p h r a s e } . " ,   
" { p h r a s e } − t h i s i s a p r o p e r t y o f t h e s i g n a l . " ,   
]   
IRRELEVANT\_PHRASES = {   
" p e r i o d i c i t y " : [   
" r e p e a t s a f t e r a f i x e d i n t e r v a l " ,   
" c o m p l e t e s a f u l l c y c l e p e r i o d i c a l l y " ,   
" r e t u r n s t o i t s p r e v i o u s v a l u e a f t e r o n e p e r i o d " ,   
" o s c i l l a t e s w i t h a c o n s t a n t f r e q u e n c y " ,   
" e x h i b i t s p e r i o d i c b e h a v i o u r " ,   
] ,   
" b o u n d e d n e s s " : [   
" r e m a i n s bounded b e t w e e n i t s minimum and maximum v a l u e s " ,   
" d o e s n o t e x c e e d i t s a m p l i t u d e " ,   
" i s c o n fi n e d w i t h i n a fi x e d r a n g e " ,   
" s t a y s w i t h i n a f i x e d i n t e r v a l a t a l l t i m e s " ,   
" h a s a f i n i t e a m p l i t u d e " ,   
] ,   
" s m o o t h n e s s " : [   
" i s c o n t i n u o u s l y d i f f e r e n t i a b l e e v e r y w h e r e " ,   
" h a s no d i s c o n t i n u i t i e s "   
" v a r i e s s m o o t h l y a t e v e r y p o i n t " ,   
" h a s a w e l l − d e fi n e d d e r i v a t i v e a t t h i s l o c a t i o n " ,   
" c h a n g e s w i t h o u t any a b r u p t t r a n s i t i o n s " ,   
] ,   
" domain " : [   
" i s d e f i n e d f o r a l l r e a l − v a l u e d i n p u t s " ,   
" h a s a domain s p a n n i n g a l l r e a l n u m b e r s "   
" i s w e l l d e fi n e d a t e v e r y p o i n t a l o n g t h e a x i s " ,   
" t a k e s r e a l v a l u e s a c r o s s i t s e n t i r e domain " ,   
" c a n b e e v a l u a t e d a t a n y p o i n t " ,   
] ,   
}

At transition points, the transition templates below are used for correct annotations. Incorrect annotations are generated for the whole signal at once, but with a 30% chance of using a transition template instead of a non-transition template, so the same number of correct and incorrect points use this template, but the correct ones are only found are drop\_to\_constant and recover\_from\_constant phases, whereas the incorrect transition templates are randomly distributed throughout the signal.

If incorrect templates only used transition templates at transition points, the incorrect annotations would represent a directly conflicting signal which should provide the exact same amount of information as the correct annotations, making the two categories identical. By distributing transition vocabulary randomly throughout the incorrect annotations, but also maintaining a majority that do directly conflict the signal, the incorrect annotations as a whole provide more information for future prediction than the irrelevant templates, with no directionality at all, but still less that the correct annotations, enabling more fine-grained evaluation of mutual information estimators.

## Listing 2: MMTT-Bench, Transition templates

DROP\_WORD\_BANKS = {   
" v e r b " : [ " c u t " , " clamp " , " c o l l a p s e " , " go " , " t r a n s i t i o n " ] ,   
" n o u n " : [ " c u t " , " c l a m p " , " c o l l a p s e " , " t r a n s i t i o n " , " d e s c e n t " ] ,   
" g e r u n d " : [ " c u t t i n g " , " c l a m p i n g " , " c o l l a p s i n g " , " g o i n g " , " t r a n s i t i o n i n g " ] ,   
}   
RECOVERY\_WORD\_BANKS = {   
" v e r b " : [ " r e c o v e r " , " res u me " , " r e t u r n " , " emerge " , " r e v i v e " ] ,   
" noun " : [ " r e c o v e r y " , " r e s u m p t i o n " , " r e t u r n " , " e m e r g e n c e " , " r e v i v a l " ] ,   
" g e r u n d " : [ " r e c o v e r i n g " , " r e s u m i n g " , " r e t u r n i n g " , " e m e r g i n g " , " r e v i v i n g " ] ,   
}   
DROP\_TEMPLATES = [   
# 0 − s i mp l e   
" The s i g n a l w i l l no l o n g e r fo l l o w { s i g n a l } and i n s t e a d soon { v e r b } "   
" t o a n e g a t i v e c o n s t a n t . " ,   
# 1 − imminent   
" The s i g n a l i s n o t a { s i g n a l } and i s a b o u t t o { v e r b } t o − 0 . 5 . " ,   
# 2 − e xp e c t a t i o n   
" E x p e c t t h e s i g n a l t o change from a { s i g n a l } and { v e r b } t o n e g a t i v e "   
" h o r i z o n t a l l i n e s h o r t l y . " ,   
# 3 − nominal   
"A { noun } t o a c o n s t a n t v a l u e i s imminent due t o a change o f s i g n a l "   
" fr o m a { s i g n a l } . " ,   
# 4 − gerund −l e d   
" { g e r u n d } t o minus 0 . 5 , t h e s i g n a l w i l l s h o r t l y become f l a t i n s t e a d "   
" o f f o l l o w i n g a { s i g n a l } . " ,   
# 5 − e x i s t e n t i a l   
"A new s i g n a l , d i f f e r e n t t o { s i g n a l } , w i l l s h o r t l y be { noun } t o "   
" n e g a t i v e h a l f . "   
# 6 − s i g n a l ’ s b e h a v i o u r   
" The s i g n a l ’ s b e h a v i o u r w i l l c h a n g e fr o m { s i g n a l } a n d s h o r t l y "   
" become a { noun } t o minus h a l f . " ,   
# 7 − te mp o ra l c l a u s e   
" I n t h e n e a r term , t h e s i g n a l w i l l { v e r b } t o − 0 . 5 r a t h e r t h a n "   
" f o l l o w i n g a { s i g n a l } . " ,   
# 8 − wh−nominal   
" What w i l l c h a r a c t e r i s e t h e s i g n a l n e x t i s no l o n g e r a { s i g n a l } "   
" b u t a { n o u n } t o a f l a t l i n e . " ,   
# 9 − p a s s i v e −a dj a c e n t   
" The s i g n a l i s n o t a { s i g n a l } a n y m o r e a n d w i l l s h o r t l y b e o b s e r v e d "   
" t o { v e r b } t o a f l a t p e r i o d . " ,   
]   
RECOVERY\_TEMPLATES = [   
# 0   
" The s i g n a l w i l l s o o n { v e r b } fr o m − 0 . 5 b a c k t o a { s i g n a l } . " ,   
# 1   
" The s i g n a l i s a b o u t t o { v e r b } i t s o s c i l l a t i o n a s a { s i g n a l } . " ,

```csv
# 2
" E x p e c t t h e s i g n a l t o { v e r b } fr o m i t s f l a t p e r i o d s h o r t l y . " ,
# 3
"A { noun } from n e g a t i v e h a l f t o fo l l o w a { s i g n a l } i s imminent . " ,
# 4
" { g e r u n d } from a n e g a t i v e h o r i z o n t a l l i n e , t h e { s i g n a l } w i l l s h o r t l y "
" o s c i l l a t e a g a i n . " ,
# 5
" T h e r e w i l l s h o r t l y b e a { n o u n } fr o m m i n u s h a l f t o a { s i g n a l } . " ,
# 6
" The s i g n a l ’ s b e h a v i o u r w i l l s h o r t l y b e c o m e a { s i g n a l } , { n o u n } fr o m n e g a t i v e . " ,
# 7
" I n t h e n e a r t e r m , t h e { s i g n a l } w i l l { v e r b } fr o m i t s f l a t p e r i o d . " ,
# 8
" What w i l l c h a r a c t e r i s e t h e { s i g n a l } n e x t i s a { n o u n } fr o m c o n s t a n t v a l u e . " ,
# 9
" The { s i g n a l } w i l l s h o r t l y b e o b s e r v e d t o { v e r b } fr o m n e g a t i v e 0 . 5 . " , ]
```

## B INFORMATION THEORETIC METRIC DETAILS

Table 1: Comparison of mutual information estimators used in the benchmark. † InfoNCE is a lower bound with ceiling log N nats (4.16 nats at N = 64). § KSG conservative bound $d \leq { \sqrt { N / 2 } } ,$ , liberal bound $d \leq N / 5$
<table><tr><td></td><td>KSG</td><td>MINE</td><td>InfoNCE</td><td>CCA</td><td>V-info</td><td>PID</td></tr><tr><td>Quantity Estimate type</td><td>I(X;Y) Asymp. ex-</td><td>I(X;Y) Lower</td><td>I(X;Y) Lower</td><td>I(X;Y) Exact (Gaus-</td><td> $\Delta R ^ { 2 }$  Exact (lin-</td><td>R,U,S Exact (dis-</td></tr><tr><td>Input type</td><td>act Continuous</td><td>bound Continuous</td><td>bound† Continuous</td><td>sian) Continuous</td><td>ear) Continuous</td><td>crete) Discrete</td></tr><tr><td>Assumption</td><td>None</td><td>None</td><td>None</td><td>Gaussian</td><td>Linear</td><td>None</td></tr><tr><td>Dim. limit</td><td>d  $\sqrt { N / 2 } ^ { \ S }$ </td><td>None</td><td>None</td><td> $N > d _ { X } + d \ll N$ </td><td></td><td>Cdx .CdY</td></tr><tr><td>Decompos.</td><td>No</td><td>No</td><td>No</td><td>dy No</td><td>No</td><td>Yes</td></tr><tr><td>Rep.- dependent</td><td>No</td><td>No</td><td>No</td><td>No</td><td>Yes</td><td>No</td></tr><tr><td>Chain rule</td><td>Approx.</td><td>Approx.</td><td></td><td></td><td></td><td></td></tr><tr><td>Reference</td><td>(Kraskov et al., 2004)</td><td>(Ishmael Bel- ghazi et al.,</td><td>Approx. (Oord et al., 2018)</td><td>Exact (Murphy, 2023)</td><td>Exact (Xu et al.,</td><td>N/A (Liang et al.,</td></tr></table>

We implement and evaluate six estimators, summarized in Table 1.

## B.1 K-NEAREST-NEIGHBOR ESTIMATOR

We estimate mutual information between continuous representations using the k-nearest-neighbor estimator (KSG) of Kraskov et al. (2004). For two random variables with N samples, KSG estimates $I ( X ; Y )$ by exploiting the relationship between entropy and nearest-neighbor distances, avoiding explicit density estimation. For each sample i, let $\epsilon _ { i }$ denote the Chebyshev distance to the k-th nearest neighbor in the joint space (X, Y). The marginal neighbor counts $\bar { n _ { X } ^ { ( i ) } }$ and $n _ { Y } ^ { ( i ) }$ are then obtained by querying the corresponding marginal spaces within the same radius. The estimator is

$$
{ \hat { I } } ( X ; Y ) = \psi ( k ) + \psi ( N ) - \left. \psi ( n _ { X } + 1 ) \right. - \left. \psi ( n _ { Y } + 1 ) \right. ,\tag{1}
$$

where ψ denotes the digamma function and ⟨.⟩ denotes the empirical mean(Kraskov et al., 2004). Rather than using bootstrapping to estimate variance like all other metrics, we follow (Holmes & Nemenman, 2019)’s corrected variance estimator based on $1 / N$ scaling of KSG variance, due to the known overestimation of MI in bootstrapped samples(Holmes & Nemenman, 2019).

The reliability of KSG estimates is characterized by two sample-size bounds, a conservative bound $d \leq \sqrt { N / 2 }$ and a liberal bound $d \leq N / 5$ , beyond which estimates become unreliable (Gao et al., 2017).

## B.2 PARTIAL INFORMATION DECOMPOSITION

Mutual information estimates of $I ( X _ { \mathrm { t e x t } } ; Y )$ conflate the unique contributions of text with information redundantly shared with the time series. To disentangle these interactions we apply Partial Information Decomposition (Williams & Beer, 2010), which decomposes the total joint information $I ( X _ { \mathrm { t s } } , X _ { \mathrm { t e x t } } ; Y )$ into four non-negative atoms following Liang et al. (2023a):

$$
I ( X _ { \mathrm { t s } } , X _ { \mathrm { t e x t } } ; Y ) = R + U _ { \mathrm { t s } } + U _ { \mathrm { t e x t } } + S ,\tag{2}
$$

where R is redundancy, i.e., information both modalities share about $Y , U _ { \mathrm { t s } }$ and $U _ { \mathrm { t e x t } }$ are unique contributions of each modality, and S is synergy, i.e., information available only from their combination. Following Bertschinger et al. (2014), redundancy is defined as the solution to a convex optimization over the set of joint distributions consistent with the observed marginals, which we solve using the CVXPY implementation of Liang et al. (2023a).

Since PID requires discrete inputs, we discretize each modality prior to estimation. Text embeddings are first reduced by PCA to $d _ { \mathrm { t e x t } }$ dimensions and then clustered into $C _ { \mathrm { t e x t } }$ discrete labels via k-means. Time series patch features are similarly clustered into $C _ { \mathrm { t s } }$ labels.

## B.3 V-USABLE INFORMATION

Mutual information measures statistical dependence in the distribution irrespective of whether a specific model can exploit it. Xu et al. (2020) propose predictive V-information as a complementary quantity that incorporates the computational constraints of the observer: given a predictive family $V$ , the V -entropy $H _ { V } ( Y | X )$ is the minimum cross-entropy achievable by any function in family V, where V-information is defined as

$$
I _ { \mathcal V } ( X \to Y ) = H _ { \mathcal V } ( Y ) - H _ { \mathcal V } ( Y \mid X ) .\tag{3}
$$

Without constraints on V, V-information recovers Shannon mutual information. Under a linear Gaussian predictive family with squared loss, V -information specializes to the coefficient of determination (Xu et al., 2020):

$$
I _ { \mathcal { V } } ( X \to Y ) = R ^ { 2 } ( X \to Y ) = 1 - \frac { \mathrm { V a r } ( Y - \hat { Y } ) } { \mathrm { V a r } ( Y ) } ,\tag{4}
$$

where $\hat { Y }$ is the prediction of a Ridge regression fit on X. We estimate the conditional V-information of text beyond the time series as

$$
I _ { \mathcal { V } } ( X _ { \mathrm { t e x t } } \to Y \mid X _ { \mathrm { t s } } ) = R ^ { 2 } ( X _ { \mathrm { t s } } , X _ { \mathrm { t e x t } } ) - R ^ { 2 } ( X _ { \mathrm { t s } } ) ,\tag{5}
$$

which measures the gain in explained variance when text embeddings are added to the time series features. Unlike KSG conditional MI, this quantity is reliable at small sample sizes and high embedding dimensionality up to the number of samples, since Ridge regression with regularization α is well-conditioned even when the number of features approaches the number of samples. V - information is representation-dependent by construction, as it measures information that a linear model can exploit from the given embedding, not the Shannon-theoretic information in the underlying distribution.

## B.4 NEURAL MUTUAL INFORMATION ESTIMATORS

We additionally implement two neural estimators: MINE and InfoNCE. Both estimators compute four quantities per annotation category, namely $I ( X _ { \mathrm { t e x t } } ; Y ) , I ( X _ { \mathrm { t s } } ; Y ) , I ( X _ { \mathrm { t s } } , X _ { \mathrm { t e x t } } ; Y )$ and $X _ { \mathrm { t e x t } } ; Y | X _ { \mathrm { t s } }$ , using the same subtraction $\hat { I } _ { \mathrm { j o i n t } } - \hat { I } _ { \mathrm { t s } }$ for the conditional as KSG, with both terms estimated from the same data resample for consistency. Unlike KSG, both joint and marginal problems are estimated in the same representational space by the same critic architecture, so the bias difference that causes chain-rule violations in KSG does not arise.

## B.4.1 MINE

The Mutual Information Neural Estimator of (Ishmael Belghazi et al., 2018) rewrites MI as a Kullback-Leibler divergence and exploits its Donsker-Varadhan representation (Donsker & Varadhan, 1983) to obtain a lower bound estimable by gradient descent:

$$
I ( X ; Y ) \geq \operatorname* { s u p } _ { T \in \mathcal { F } } \mathbb { E } _ { p ( x , y ) } \left[ T ( x , y ) \right] - \log \mathbb { E } _ { p ( x ) p ( y ) } \left[ e ^ { T ( x , y ) } \right] ,\tag{6}
$$

where $T$ is a statistics network parameterized by a two-layer MLP trained with Adam to maximize the bound. We utilize an open-source MINE implementation<sup>5</sup>: Negative samples are drawn from the product of marginals by shuffling Y within the batch, and an exponential moving average stabilization of the denominator is used to reduce gradient variance (Ishmael Belghazi et al., 2018), training for 500 iterations per MI estimate. MINE is not sensitive to the curse of dimensionality in the way KSG is, since the critic learns a scalar score function rather than estimating a density in d-dimensional space.

## B.4.2 INFONCE

The InfoNCE estimator of Oord et al. (2018) frames MI estimation as a N-way classification problem. For a batch of N pairs $( x _ { i } , y _ { i }$ drawn from the joint distribution $p ( x , y )$ , the critic T scores all $N ^ { 2 }$ combinations $( x _ { i } , y _ { i }$ and the estimator is:

$$
\hat { I } _ { \mathrm { N C E } } ( X ; Y ) = \mathbb { E } \left[ \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left( T ( x _ { i } , y _ { i } ) - \log \sum _ { j = 1 } ^ { N } e ^ { T ( x _ { i } , y _ { j } ) } \right) \right] + \log N ,\tag{7}
$$

where the log N term converts the biased log N normalization to an unbiased lower bound (Oord et al., 2018). The estimator is trained identically to MINE, a two-layer MLP critic optimized with Adam for 500 iterations, but constructs negatives implicitly from the off-diagonal entries of $N \times N$ score matrix rather than explicitly shuffling marginals. InfoNCE provides a tighter bound than MINE in practice but saturates at log $\dot { N }$ nats.

## B.5 CANONICAL CORRELATION ANALYSIS ESTIMATOR

As a model-based complement to the distribution-free estimators above, we implement a mutual information estimator based on Canonical Correlation Analysis (CCA), following Murphy (2023) (Ch.28), and included as a reference estimator in Czyz et al.˙ (2023). Under the assumption that the joint distribution $p ( x , y )$ is multivariate Gaussian, MI admits the closed-form expression:

$$
I ( X ; Y ) = - \frac { 1 } { 2 } \sum _ { k } \log ( 1 - \rho _ { k } ^ { 2 } ) ,\tag{8}
$$

where $\rho _ { 1 } , . . . , \rho \mathrm { m i n } ( d _ { X } , d _ { Y } )$ are the canonical correlations between X and $Y ,$ , obtained as the singular value of $L _ { X X } ^ { - 1 } C _ { X Y } L _ { Y Y } ^ { - \top }$ with $L _ { X X }$ and $L + Y Y$ the Cholesky factors of the regularized marginal covariance matrices $\bar { C } + X X + \varepsilon I$ and $C + Y Y + \varepsilon I$ respectively.

Unlike KSG, MINE and InfoNCE, CCA requires no hyperparamter tuning, no nearest-neighbor searches and no neural network training, it reduces to a single matrix decomposition and is therefore deterministic given a bootstrap resample. Czyz et al.˙ (2023) find that CCA achieves the lowest sample complexity of all estimators they evaluate, and remains competitive even when the Gaussianity assumption is mildly violated, though it fails on heavy-tailed and strongly non-linear distributions such as spiral embeddings. For our benchmark signals, CCA provides a reliable upper bound on the linear-Gaussian MI against which the distribution-free estimators can be calibrated. The chain rule holds exactly under the Gaussian assumption, so CCA conditional MI estimates do not exhibit the chain-rule inconsistency that causes KSG to produce negative conditional MI.

## C EMBEDDING VISUALIZATIONS

Figure 6 shows 2D PCA of the text-only embeddings $X _ { \mathrm { t e x t } }$ on the left, and text + time series joint embeddings $X _ { \mathrm { j o i n t } }$ on the right. Correct and incorrect text use identical template structures, as detailed in Appendix A, so are undifferentiable unless joined with a time series, at which point they either support or contradict the future prediction.

![](images/fb0475f4a8749bea499318b6145a588d58c71de32a0a88780540b77178313e76.jpg)  
Figure 6: Text embeddings for sine wave training dataset

Indeed, the text embedding (left plot) show a correct (green dots)-incorrect (orange dots) overlap, while the irrelevant annotation cluster (gray dots) is distinctly separated from the correct-incorrect cluster. The $X _ { j o i n t } = [ X _ { t s } ; X _ { t e x t } ]$ embeddings in Figure 6 show a dominant structure (PC1, 41.2% of variance) reflecting the four discrete time series states $\{ - 1 , 0 , + 1 , - 0 . 5 \}$ , visible as four vertical stripes. Text categories are almost entirely superimposed within each stripe, confirming the signal from the correct text is a fine-grained conditional effect relative to the dominant time series structure.

## D EXPERIMENT DETAILS

All experiments were conducted on a server equipped with 2xAMD EPYC 7763 64-core processors (128 physical cores, 256 logical CPUs), 2 TiB RAM, and 4xNVIDIA GeForce RTX 4090 GPUs (24 GB VRAM each), running Ubuntu 22.04.5 LTS with CUDA driver 570.133.07.

Experiments were generally lightweight and fast to run, apart from the MINE estimator which on MMTT-Bench training split of N = 1279 points takes ≈ 53 ± 0.64seconds per MI estimate when running only on CPU, where our experiment runs 20 estimates for each result.

## D.1 MODEL TRAINING

We trained two classes of models for both MMTT-Bench and real world datasets: five simple scikit-learn regressors and nine time-series transformer architectures with ten fusion strategies each.

## D.1.1 SCIKIT-LEARN REGRESSORS

Five models, each fit on the concatenation of the time-series lookback window and the PCA-reduced text embedding, with the following defaults: ridge regression (α = 1.0, MLP (hidden-layers 128-64, max 500 iterations), SVR (RBF kernel, $\mathrm { C } { = } 1 0 , \overset { \cdot } { \varepsilon } = \mathrm { \bar { 0 } } . 0 1 \mathrm { \cdot }$ ), gradient boosting (200 estimators, max depth 4, learning rate 0.05), random foresst (200 estimators, max depth 8), and k-nearest-neighbors (k=10, distance-weighted). Seeds are fixed at 42 where the model is stochastic. The joint feature vector is a horizontal concatenation of the two modalities, deliberately the simplest possible fusion, so that any gain is attributable to the information in the text rather than to a learned fusion mechanism.

## D.1.2 TRANSFORMER BACKBONES

We used the Controlled Fusion Adapter (CFA) implementation from (Lee et al., 2026b), which extends time series transformer backbones to accept a text context. We train nine architectures: PatchTST, DLinear, iTransformer, Autoformer, FEDformer, Informer, TiDE, FiLM and Nonstationary Transformer. Shared hyperparameters follow CFA’s Table E.1 (Lee et al., 2026b). Training uses batch size 32, up to 30 epochs with early stopping at patience 5, learning rate $1 e ^ { - 3 }$ for the backbone and $1 e ^ { - 2 }$ for the text projection MLP, with one seed (2021). Text is encoded with BERT (6 layers, average pooling, frozen) and projected by an MLP from $d _ { \mathrm { l l m } }$ to $d _ { \mathrm { l l m } } / 8$ to the prediction length, following CFA’s ‘exp\_long\_term\_forecasting\_text\_integrated‘. Real datasets use Liu et al. (2024)’s MM-TSFlib sliding-window loader with a StandardScaler. The synthetic dataset is one sample per annotation point, with no window, to preserve the sample independence MI estiamates require.

## D.1.3 FUSION STRATEGIES

Each backbone is trained under ten injection odes, all of them from the CFA codebase, spanning three families: Nine backbones times ten text-consuming models give the 90 configurations summarized in

Table 2: Text injection modes. Every backbone is trained under all eleven, giving $9 \times 1 0 = 9 0$ text-consuming configurations per dataset; the unimodal baseline is trained once per backbone and shared across them. All modes are taken from the CFA implementation of Lee et al. (2026b)
<table><tr><td>Mode</td><td>Source</td><td>What it does</td></tr><tr><td colspan="3">Baseline</td></tr><tr><td>unimodal</td><td>CFA</td><td>Text is ignored entirely; the no-text baseline every comparison is made against.</td></tr><tr><td colspan="3">Naive fusion</td></tr><tr><td>first-additive</td><td>CFA</td><td>Projected text embedding is added to the series representation before the backbone encoder.</td></tr><tr><td>middle-additive</td><td>CFA</td><td>Added to the intermediate representation inside the backbone.</td></tr><tr><td>last-additive</td><td>CFA</td><td>Added to the backbone output immediately before the predic- tion head.</td></tr><tr><td>first-concat</td><td>CFA</td><td>Text is concatenated to the series representation at the input, widening the feature dimension.</td></tr><tr><td>middle-concat</td><td>CFA</td><td>Concatenated at the intermediate representation.</td></tr><tr><td>last-concat</td><td>CFA</td><td>Concatenated at the output, before the head.</td></tr><tr><td colspan="3">Constrained fusion</td></tr><tr><td>film</td><td>CFA, after Perez et al. (2018)</td><td>Text produces per-channel scale and shift parameters that modulate the series representation, rather than being summed</td></tr><tr><td>gating</td><td>CFA</td><td>into it. A learned gate decides how much text signal passes through,</td></tr><tr><td>orthogonal</td><td>CFA</td><td>so the model can suppress it. The text contribution is projected onto the subspace orthogo- nal to the series representation, so it can only add information</td></tr><tr><td>cfa</td><td>CFA (proposed method by Lee et al. (2026b))</td><td>the series does not already carry. Constrained fusion adapter: cross-attention from series to text through a bottleneck (reduction factor 8).</td></tr></table>

every distribution we report below in Appendix F and G.2. The unimodal baseline is trained once per backbone and shared across them.

## E MMTT-BENCH PARAMETERS

Table 3 shows the final parameters used for MMTT-Bench experiments. In the following sub-sections we report parameter sweeps for the subset of hyperparameters whose optimal values were not directly inherited from the dataset configuration or established conventions in the MI estimation literature.

## E.1 TEXT TOKENIZATION AND DIMENSIONALITY REDUCTION

Tables 4 and 5 contain the full results that are summarized in Figure 2 in the main manuscript. Table 4 compares sentence-transformer models; DistilRoBERTa achieves the highest $X _ { t e x t }$ information relative to $X _ { t s }$ and is used as the baseline. Table 5 shows as text dimensions increase the information provided by $X _ { t e x t }$ across all estimators also increases. KSG is unreliable above $d \leq \sqrt { N / 2 } \approx 2 5$ so we choose $d _ { t e x t } = 1 6$ as the highest viable value.

Table 3: Final hyperparameters for MMTT-Bench experiments  
(a) Overall hyperparameters
<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Embedding model</td><td>sentence-transformers/all-distilroberta-v1</td></tr><tr><td>Time-series tokeniser</td><td>Identity</td></tr><tr><td>Horizon steps</td><td>1</td></tr><tr><td>Lookback steps</td><td>2</td></tr><tr><td>Patch length</td><td>1</td></tr><tr><td>Stride</td><td>1</td></tr><tr><td>PCA dimension</td><td>16 20</td></tr><tr><td>Bootstrap resamples</td><td></td></tr><tr><td>Embedding strategy</td><td>Discrete</td></tr><tr><td>Shuffle</td><td>True for shuffle experiment, False for everything else</td></tr></table>

(b) Per-estimator hyperparameters
<table><tr><td>Estimator</td><td>Parameter</td><td>Value</td></tr><tr><td>KSG</td><td>Neighbors k</td><td></td></tr><tr><td rowspan="4">MINE</td><td>Training iterations</td><td>500</td></tr><tr><td>Hidden dimension</td><td>128</td></tr><tr><td>Batch size N</td><td>64 (log N ≈ 4.16 nat ceiling)</td></tr><tr><td>Learning rate</td><td> $1 0 ^ { - 4 }$ </td></tr><tr><td rowspan="4">InfoNCE</td><td>Training iterations</td><td>500</td></tr><tr><td>Hidden dimension</td><td>128</td></tr><tr><td>Batch size N</td><td>64 (log N ≈ 4.16 nat ceiling)</td></tr><tr><td>Learning rate</td><td> $1 0 ^ { - 3 }$ </td></tr><tr><td>CCA</td><td>Regularization ε</td><td> $1 0 ^ { - 5 }$ </td></tr><tr><td>V-information</td><td>Ridge CV folds</td><td></td></tr><tr><td rowspan="4">PID</td><td>Time-series clusters  $C _ { \mathrm { t s } }$ </td><td>4</td></tr><tr><td>Text clusters  $C _ { \mathrm { t e x t } }$  Target bins  $n _ { Y }$ </td><td>5 4 (mirrors the four discrete signal values</td></tr><tr><td></td><td> $\{ 1 , 0 , - 1 , - 0 . 5 \} )$ </td></tr><tr><td>Target channel y</td><td>0</td></tr></table>

Table 4: MI estimator results for different text embedding models.
<table><tr><td>Model</td><td>KSG</td><td>MINE</td><td>InfoNCE</td><td>CCA</td><td>V-information</td><td> $\mathrm { P I D } \left( \mathrm { U } _ { \mathrm { t e x t } } \right)$ </td></tr><tr><td>glove.840B.300d</td><td> $0 . 1 3 5 \pm 0 . 0 6 8$ </td><td> $0 . 2 8 2 \pm 0 . 0 5 6$ </td><td> $0 . 5 2 1 \pm 0 . 0 2 8$ </td><td> $0 . 1 4 1 \pm 0 . 0 1 4$ </td><td> $0 . 1 4 0 \pm 0 . 0 1 1$ </td><td> $0 . 1 9 6 \pm 0 . 0 1 4$ </td></tr><tr><td>all-MiniLM-L6-v2</td><td> $0 . 1 6 6 \pm 0 . 0 6 7$ </td><td> $0 . 3 5 7 \pm 0 . 0 5 7$ </td><td> $0 . 5 4 6 \pm 0 . 0 2 9$ </td><td> $0 . 2 3 4 \pm 0 . 0 1 6$ </td><td> $0 . 2 1 3 \pm 0 . 0 1 2$ </td><td>0.179 ± 0.018</td></tr><tr><td>all-distilroberta-v</td><td> $0 . 2 6 1 \pm 0 . 0 6 8$ </td><td> $0 . 4 2 7 \pm 0 . 0 5 7$ </td><td> $0 . 5 5 7 \pm 0 . 0 2 6$ </td><td> $0 . 2 9 7 \pm 0 . 0 1 9$ </td><td> $0 . 2 5 4 \pm 0 . 0 1 4$ </td><td> $0 . 3 0 7 \pm 0 . 0 1 4$ </td></tr><tr><td>bert-base-uncased</td><td> $0 . 1 9 9 \pm 0 . 0 7 2$ </td><td> $0 . 3 2 6 \pm 0 . 0 5 1$ </td><td> $0 . 5 4 1 \pm 0 . 0 2 7$ </td><td> $0 . 1 4 9 \pm 0 . 0 1 6$ </td><td>0.146 ± 0.013</td><td> $0 . 3 4 4 \pm 0 . 0 1 7$ </td></tr><tr><td>gpt2</td><td> $0 . 2 6 8 \pm 0 . 0 6 7$ </td><td> $0 . 3 4 5 \pm 0 . 0 5 3$ </td><td> $0 . 5 4 6 \pm 0 . 0 2 7$ </td><td> $0 . 1 2 8 \pm 0 . 0 2 6$ </td><td> $0 . 1 2 8 \stackrel { - } { \pm } 0 . 0 1 \stackrel { - } { 3 }$ </td><td> $0 . 1 8 6 \pm 0 . 0 1 4$ </td></tr><tr><td>Llama-2-7b-hf</td><td> $0 . 3 1 5 \pm 0 . 0 6 7$ </td><td> $0 . 3 5 3 \pm 0 . 0 5 3$ </td><td> $0 . 5 3 9 \pm 0 . 0 2 7$ </td><td> $0 . 1 2 4 \pm 0 . 0 1 4$ </td><td> $0 . 1 2 4 \pm 0 . 0 1 3$ </td><td> $0 . 1 2 1 \pm 0 . 0 1 5$ </td></tr><tr><td>opt-125m</td><td> $0 . 3 4 4 \pm 0 . 0 6 3$ </td><td> $0 . 3 5 1 \pm 0 . 0 4 9$ </td><td> $0 . 5 3 7 \pm 0 . 0 2 8$ </td><td> $0 . 1 4 9 \pm 0 . 0 1 6$ </td><td> $0 . 1 4 6 \pm 0 . 0 1 3$ </td><td> $0 . 2 5 7 \pm 0 . 0 1 7$ </td></tr><tr><td>phi-2</td><td>0.366 ± 0.066</td><td>0.338 ± 0.054</td><td>0.531 ± 0.018</td><td> $0 . 1 4 0 \pm 0 . 0 1 6$ </td><td>0.139 ± 0.013</td><td> $0 . 2 3 7 \pm 0 . 0 1 6$ </td></tr><tr><td>oracle</td><td>1.114 ± 0.031</td><td> $0 . 6 0 8 \pm 0 . 0 4 3$ </td><td>0.618 ± 0.023</td><td> $4 . 6 8 0 \pm 0 . 0 1 3$ </td><td>0.568 ± 0.020</td><td> $0 . 7 9 6 \pm 0 . 0 1 2$ </td></tr></table>

## E.2 TIME SERIES TOKENIZATION

For time series tokenization, we first test $X _ { t s }$ as $Y$ values of the previous $n _ { l o o k b a c k }$ steps. Table 6 shows an input of $n _ { l o o k b a c k } = 2$ ensures the time series provides some predictive information without making prediction trivial in phases that follow a sine wave oscillation. We did experiment using PatchTST time series tokenization, introduced by (Nie et al., 2023), with results for different patch and stride lengths available in Table 7, however given the simplicity of the signal we opt to keep $X _ { t s }$ as raw Y values.

Table 5: MI estimator results for different PCA dimensions
<table><tr><td>Estimator</td><td>PCA dimension  $d _ { \mathrm { t e x t } }$ </td><td>Correct</td><td>Incorrect</td><td>Irrelevant</td></tr><tr><td>KSG</td><td>4</td><td>0.57</td><td>0.119</td><td>0.183</td></tr><tr><td rowspan="6">MINE</td><td>8</td><td>0.514</td><td>0.102</td><td>0.193</td></tr><tr><td>16</td><td>0.261</td><td>-0.011</td><td>0.118</td></tr><tr><td>32</td><td>0.026</td><td>-0.199</td><td>0.011</td></tr><tr><td>4</td><td>0.318</td><td>0.033</td><td>0.047</td></tr><tr><td>8</td><td>0.4</td><td>0.101</td><td>0.015</td></tr><tr><td>16</td><td>0.413</td><td>0.138</td><td>0.025</td></tr><tr><td rowspan="4">InfoNCE</td><td>32</td><td>0.463</td><td>0.182</td><td>-0.019</td></tr><tr><td>4</td><td>0.429</td><td>0.125</td><td>0.105</td></tr><tr><td>8</td><td>0.515</td><td>0.278</td><td>0.18</td></tr><tr><td>16</td><td>0.557</td><td>0.417</td><td>0.244</td></tr><tr><td rowspan="4">CCA</td><td>32</td><td>0.58</td><td>0.507</td><td>0.3</td></tr><tr><td>4</td><td>0.115</td><td>0.003</td><td>0.002</td></tr><tr><td>8</td><td>0.184</td><td>0.026</td><td>0.004</td></tr><tr><td>16</td><td>0.297</td><td>0.053</td><td>0.01</td></tr><tr><td rowspan="4">V-information</td><td>32</td><td>0.349</td><td>0.075</td><td>0.022</td></tr><tr><td>4</td><td>0.117</td><td>0.003</td><td>0.002</td></tr><tr><td>8</td><td>0.174</td><td>0.028</td><td>0.005</td></tr><tr><td>16</td><td>0.254</td><td>0.057</td><td>0.011</td></tr><tr><td rowspan="5"> $\mathrm { P I D } \left( \mathrm { U } _ { \mathrm { t e x t } } \right)$ </td><td>32</td><td>0.285</td><td>0.078</td><td>0.022</td></tr><tr><td>4</td><td>0.386</td><td>0</td><td>0</td></tr><tr><td>8</td><td>0.301</td><td>0</td><td>0</td></tr><tr><td>16</td><td>0.307</td><td>0.001</td><td>0</td></tr><tr><td>32</td><td>0.318</td><td>0.001</td><td>0</td></tr></table>

## E.3 PID DISCRETIZATION

The target $Y _ { \mathrm { f u t u r e } }$ is binned into four classes, corresponding to the possible signal values at annotation points $\breve { Y } \in [ 1 , 0 , - 1 , - 0 . 5 ]$ . The joint distribution $\dot { ( } P ( X _ { \mathrm { t s } } , X _ { \mathrm { t e x t } } , \dot { Y } )$ is estimated from the resulting discrete triples and passed to the solver.

Table 6: MI estimator results for different numbers of lookback steps. Here there is no time series tokenization, the lookback steps are just passed directly as the time series vector
<table><tr><td>Estimator</td><td>lookback steps  $n _ { \mathrm { l o o k b a c k } }$ </td><td>Correct</td><td>Incorrect</td><td>Irrelevant</td></tr><tr><td>KSG</td><td>1</td><td>0.907</td><td>0.601</td><td>0.555</td></tr><tr><td rowspan="6">MINE</td><td>2</td><td>0.261</td><td>-0.011</td><td>0.118</td></tr><tr><td>4</td><td>-0.857</td><td>-1.061</td><td>-0.948</td></tr><tr><td>8</td><td>-0.781</td><td>-0.894</td><td>-0.86</td></tr><tr><td>1</td><td>0.586</td><td>0.314</td><td>0.187</td></tr><tr><td>2</td><td>0.43</td><td>0.146</td><td>0.05</td></tr><tr><td>4</td><td>0.396</td><td>0.077</td><td>0.014</td></tr><tr><td rowspan="4">InfoNCE</td><td>8</td><td>0.413</td><td>0.093</td><td>0.028</td></tr><tr><td>1</td><td>0.744</td><td>0.581</td><td>0.293</td></tr><tr><td>2</td><td>0.557</td><td>0.417</td><td>0.244</td></tr><tr><td>4</td><td>0.49</td><td>0.342</td><td>0.255</td></tr><tr><td rowspan="4">CCA</td><td>8</td><td>0.414</td><td>0.287</td><td>0.223</td></tr><tr><td>1</td><td>0.401</td><td>0.147</td><td>0.013</td></tr><tr><td>2</td><td>0.297</td><td>0.053</td><td>0.01</td></tr><tr><td>4</td><td>0.278</td><td>0.036</td><td>0.009</td></tr><tr><td rowspan="4">V-information</td><td>8</td><td>0.278</td><td>0.025</td><td>0.009</td></tr><tr><td>1</td><td>0.55</td><td>0.254</td><td>0.022</td></tr><tr><td>2</td><td>0.254</td><td>0.057</td><td>0.011</td></tr><tr><td>4</td><td>0.207</td><td>0.033</td><td>0.007</td></tr><tr><td rowspan="5">PID (Utext)</td><td>8</td><td>0.18</td><td>0.019</td><td>0.006</td></tr><tr><td>1</td><td>0.222</td><td>0.003</td><td>0.002</td></tr><tr><td>2</td><td>0.307</td><td>0.001</td><td>0</td></tr><tr><td>4</td><td>0.321</td><td>0.002</td><td>0.001</td></tr><tr><td>8</td><td>0.324</td><td>0.002</td><td>0.001</td></tr></table>

F MMTT-BENCH FULL RESULTS

## F.1 MI ESTIMATION

Table 7: MI estimator results for different numbers of patch lengths, strides and lookback steps. Here PatchTST time series tokenisation is used, where lookback steps must be greater than or equal to patch length
<table><tr><td>Estimator</td><td>patch length</td><td>stride</td><td>lookback steps Nlookback</td><td>Correct</td><td>Incorrect</td><td>Irrelevant</td></tr><tr><td>KSG</td><td>1</td><td>1</td><td>2</td><td>0.842</td><td>0.578</td><td>0.593</td></tr><tr><td rowspan="9"></td><td>2</td><td>1</td><td>2</td><td>0.854</td><td>0.481</td><td>0.343</td></tr><tr><td>2</td><td>2</td><td>2</td><td>0.854</td><td>0.481</td><td>0.343</td></tr><tr><td>4</td><td>2</td><td>4</td><td>-0.857</td><td>-1.061</td><td>-0.948</td></tr><tr><td>4</td><td>4</td><td>4</td><td>-0.857</td><td>-1.061</td><td>-0.948</td></tr><tr><td>1</td><td>1</td><td>2</td><td>0.404</td><td>0.156</td><td>0.038</td></tr><tr><td>2</td><td>1</td><td>2</td><td>0.762</td><td>0.352</td><td>0.016</td></tr><tr><td>2</td><td>2</td><td>2</td><td>0.762</td><td>0.352</td><td>0.016</td></tr><tr><td>4</td><td>2</td><td>4</td><td>0.397</td><td>0.068</td><td>0.012</td></tr><tr><td>4</td><td>4</td><td>4</td><td>0.393</td><td>0.079</td><td>0.019</td></tr><tr><td rowspan="5">InfoNCE</td><td>1</td><td>1</td><td>2</td><td>0.557</td><td>0.417</td><td>0.244</td></tr><tr><td>2</td><td>1</td><td>2</td><td>0.9</td><td>0.631</td><td>0.131</td></tr><tr><td>2</td><td>2</td><td>2</td><td>0.9</td><td>0.631</td><td>0.131</td></tr><tr><td>4</td><td>2</td><td>4</td><td>0.49</td><td>0.342</td><td>0.255</td></tr><tr><td>4</td><td>4</td><td>4</td><td>0.49</td><td>0.342</td><td>0.255</td></tr><tr><td rowspan="5">CCA</td><td>1</td><td>1</td><td>2</td><td>0.297</td><td>0.053</td><td>0.01</td></tr><tr><td>2</td><td>1</td><td>2</td><td>0.226</td><td>0.037</td><td>0.014</td></tr><tr><td>2</td><td>2</td><td>2</td><td>0.226</td><td>0.037</td><td>0.014</td></tr><tr><td>4</td><td>2</td><td>4</td><td>0.278</td><td>0.036</td><td>0.009</td></tr><tr><td>4</td><td>4</td><td>4</td><td>0.278</td><td>0.036</td><td>0.009</td></tr><tr><td rowspan="5">V-information</td><td>1</td><td>1</td><td>2</td><td>0.254</td><td>0.057</td><td>0.011</td></tr><tr><td>2</td><td>1</td><td>2</td><td>0.306</td><td>0.059</td><td>0.02</td></tr><tr><td>2</td><td>2</td><td>2</td><td>0.306</td><td>0.059</td><td>0.02</td></tr><tr><td>4</td><td>2</td><td>4</td><td>0.207</td><td>0.033</td><td>0.007</td></tr><tr><td>4</td><td>4</td><td>4</td><td>0.207</td><td>0.033</td><td>0.007</td></tr><tr><td rowspan="5"> $\mathrm { P I D } \left( \mathrm { U } _ { \mathrm { t e x t } } \right)$ </td><td>1</td><td>1</td><td>2</td><td>0.307</td><td>0.001</td><td>0</td></tr><tr><td>2</td><td>1</td><td>2</td><td>0.297</td><td>0.002</td><td>0.002</td></tr><tr><td>2</td><td>2</td><td>2</td><td>0.297</td><td>0.002</td><td>0.002</td></tr><tr><td>4</td><td>2</td><td>4</td><td>0.321</td><td>0.002</td><td>0.001</td></tr><tr><td>4</td><td>4</td><td>4</td><td>0.321</td><td>0.002</td><td>0.001</td></tr></table>

Table 8: Conditional mutual information $I ( X _ { \mathrm { t e x t } } ; Y \mid X _ { \mathrm { t s } } )$ estimates on MMTT-Bench, in nats, mean ± standard deviation over 20 bootstrap resamples.
<table><tr><td>Estimator</td><td>Correct</td><td>Incorrect</td><td>Irrelevant</td></tr><tr><td>KSG</td><td> $0 . 4 0 4 \pm 0 . 0 6 6$ </td><td> $0 . 0 0 9 \pm 0 . 0 6 4$ </td><td> $0 . 0 2 1 \pm 0 . 0 6 7$ </td></tr><tr><td>MINE</td><td> $0 . 4 2 6 \pm 0 . 0 5 7$ </td><td> $0 . 1 2 5 \pm 0 . 0 4 9$ </td><td> $0 . 0 3 0 \pm 0 . 0 5 5$ </td></tr><tr><td>InfoNCE</td><td> $0 . 5 0 4 \pm 0 . 0 2 4$ </td><td> $0 . 2 1 9 \pm 0 . 0 2 7$ </td><td> $0 . 1 1 1 \pm 0 . 0 2 4$ </td></tr><tr><td>CCA</td><td> $0 . 2 9 7 \pm 0 . 0 1 9$ </td><td> $0 . 0 5 3 \pm 0 . 0 0 9$ </td><td> $0 . 0 1 0 \pm 0 . 0 0 3$ </td></tr><tr><td>Deep CCA</td><td> $2 . 2 9 1 \pm 0 . 0 7 3$ </td><td> $0 . 5 8 1 \pm 0 . 0 6 0$ </td><td> $0 . 2 2 9 \pm 0 . 0 5 2$ </td></tr><tr><td>V-information</td><td> $0 . 2 5 4 \pm 0 . 0 1 4$ </td><td> $0 . 0 5 7 \pm 0 . 0 0 9$ </td><td> $0 . 0 1 1 \pm 0 . 0 0 4$ </td></tr><tr><td>PID  $\mathrm { ( U _ { t e x t } ) }$ </td><td> $0 . 3 0 7 \pm 0 . 0 1 4$ </td><td> $0 . 0 0 1 \pm 0 . 0 0 1$ </td><td> $0 . 0 0 0 \pm 0 . 0 0 0$ </td></tr></table>

## F.2 MODEL PERFORMANCE

Table 9: MMTT-Bench results for five simple architectures and nine different time-series Transformer architectures, with each transformer architecture trained with 10 different fusion strategies
<table><tr><td>Model</td><td>Fusion</td><td>No text</td><td>Correct</td><td>Incorrect</td><td>Irrelevant</td></tr><tr><td>ridge</td><td>default</td><td>0.2712</td><td>0.1462</td><td>0.2607</td><td>0.2728</td></tr><tr><td>mlp</td><td>default</td><td>0.2222</td><td>0.0967</td><td>0.2828</td><td>0.2965</td></tr><tr><td>svr</td><td>default</td><td>0.2503</td><td>0.1060</td><td>0.2660</td><td>0.2836</td></tr><tr><td>gbr</td><td>default</td><td>0.2161</td><td>0.0901</td><td>0.1995</td><td>0.2478</td></tr><tr><td>rf</td><td>default</td><td>0.2160</td><td>0.0886</td><td>0.1792</td><td>0.2331</td></tr><tr><td rowspan="12">Autoformer</td><td>cfa</td><td>0.2167</td><td>0.2106</td><td>0.2316</td><td>0.2510</td></tr><tr><td>film</td><td>0.2167</td><td>0.0875</td><td>0.2269</td><td>0.2278</td></tr><tr><td>first-additive</td><td>0.2167</td><td>0.1727</td><td>0.2084</td><td>0.2191</td></tr><tr><td>first-concat</td><td>0.2167</td><td>0.2095</td><td>0.2387</td><td>0.2790</td></tr><tr><td>gating</td><td>0.2167</td><td>0.1037</td><td>0.2213</td><td>0.2387</td></tr><tr><td>last-additive</td><td>0.2167</td><td>0.1920</td><td>0.2576</td><td>0.2456</td></tr><tr><td>last-concat</td><td>0.2167</td><td>0.1592</td><td>0.2050</td><td>0.2127</td></tr><tr><td>middle-additive</td><td>0.2167</td><td>0.1695</td><td>0.2224</td><td>0.2346</td></tr><tr><td>middle-concat</td><td>0.2167</td><td>0.2348</td><td>0.2284</td><td>0.2219</td></tr><tr><td>orthogonal</td><td>0.2167 0.2120</td><td>0.1033</td><td>0.2198</td><td>0.2569</td></tr><tr><td>cfa film</td><td>0.2120</td><td>0.2113 0.1064</td><td>0.2113 0.2090</td><td>0.2113</td></tr><tr><td></td><td></td><td></td><td></td><td>0.2118</td></tr><tr><td rowspan="10">DLinear</td><td>first-additive</td><td>0.2120</td><td>0.1357</td><td>0.2056</td><td>0.2134</td></tr><tr><td>first-concat</td><td>0.2120</td><td>0.1358</td><td>0.2130</td><td>0.2133</td></tr><tr><td>gating</td><td>0.2120</td><td>0.1009</td><td>0.2031</td><td>0.2139</td></tr><tr><td>last-additive</td><td>0.2120</td><td>0.1258</td><td>0.2027</td><td>0.2130</td></tr><tr><td>last-concat</td><td>0.2120</td><td>0.1237</td><td>0.2083</td><td>0.2136</td></tr><tr><td>middle-additive</td><td>0.2120</td><td>0.1027</td><td>0.2022</td><td>0.2125</td></tr><tr><td>middle-concat</td><td>0.2120</td><td>0.1283</td><td>0.2149</td><td>0.2168</td></tr><tr><td>orthogonal</td><td>0.2120</td><td>0.2106</td><td>0.2135</td><td>0.2117</td></tr><tr><td>cfa</td><td>0.2096</td><td>0.2210</td><td>0.2147</td><td>0.2086</td></tr><tr><td>film</td><td>0.2096</td><td>0.0560</td><td>0.2450</td><td>0.2069</td></tr><tr><td rowspan="10">FEDformer</td><td>first-additive</td><td>0.2096</td><td>0.0674</td><td>0.2433</td><td>0.2035</td></tr><tr><td>first-concat</td><td>0.2096</td><td>0.0642</td><td>0.2266</td><td>0.2430</td></tr><tr><td>gating</td><td>0.2096</td><td>0.1100</td><td>0.1986</td><td>0.2268</td></tr><tr><td>last-additive</td><td>0.2096</td><td>0.1996</td><td>0.2040</td><td>0.2039</td></tr><tr><td>last-concat</td><td>0.2096</td><td>0.1085</td><td>0.2104</td><td>0.1902</td></tr><tr><td>middle-additive</td><td>0.2096</td><td>0.1821</td><td>0.2228</td><td>0.2105</td></tr><tr><td>middle-concat</td><td>0.2096</td><td>0.1045</td><td>0.2037</td><td>0.2088</td></tr><tr><td>orthogonal</td><td>0.2096</td><td>0.1347</td><td>0.2451</td><td>0.2278</td></tr><tr><td>cfa</td><td>0.4097</td><td>0.4096</td><td>0.4096</td><td>0.4096</td></tr><tr><td>film</td><td>0.4097</td><td>0.4097</td><td>0.4097</td><td>0.4097</td></tr><tr><td rowspan="10">FiLM</td><td>first-additive</td><td>0.4097</td><td>0.4107</td><td>0.4107</td><td>0.4107</td></tr><tr><td>first-concat</td><td>0.4097</td><td></td><td></td><td></td></tr><tr><td></td><td></td><td>0.4894</td><td>0.4894</td><td>0.4894</td></tr><tr><td>gating</td><td>0.4097</td><td>0.4097</td><td>0.4097</td><td>0.4097</td></tr><tr><td>last-additive last-concat</td><td>0.4097</td><td>0.2245</td><td>0.3280</td><td>0.4218</td></tr><tr><td>middle-additive</td><td>0.4097</td><td>0.2163</td><td>0.3277</td><td>0.3963</td></tr><tr><td>middle-concat</td><td>0.4097</td><td>0.4097</td><td>0.4097</td><td>0.4097</td></tr><tr><td></td><td>0.4097</td><td>0.4097</td><td>0.4097</td><td>0.4097</td></tr><tr><td>orthogonal</td><td>0.4097</td><td>0.4096</td><td>0.4096</td><td>0.4096</td></tr><tr><td>cfa</td><td>0.2130</td><td>0.2760</td><td>0.1908</td><td>0.2359</td></tr><tr><td rowspan="6">Informer</td><td>film</td><td>0.2130</td><td>0.0628</td><td>0.2071</td><td>0.2307</td></tr><tr><td>first-additive</td><td>0.2130</td><td>0.0639</td><td>0.1870</td><td>0.1927</td></tr><tr><td>first-concat</td><td>0.2130</td><td>0.2904</td><td>0.3194</td><td>0.3665</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td>gating</td><td>0.2130</td><td>0.0639</td><td>0.1883</td><td>0.1876</td></tr><tr><td>last-additive</td><td>0.2130</td><td>0.1419</td><td>0.2048</td><td>0.2046</td></tr><tr><td rowspan="2"></td><td>last-concat</td><td>0.2130</td><td>0.2203</td><td>0.2419</td><td>0.2452</td></tr><tr><td>middle-additive</td><td>0.2130</td><td>0.0942</td><td>0.2202</td><td>0.1947</td></tr><tr><td rowspan="2"></td><td>middle-concat</td><td>0.2130 0.2130</td><td>0.1148</td><td>0.2031</td><td>0.2038</td></tr><tr><td>orthogonal</td><td></td><td>0.0710</td><td>0.2231</td><td>0.2234</td></tr><tr><td rowspan="9">Nonstationary Transformer</td><td>cfa</td><td>0.2292</td><td>0.1859</td><td>0.2071</td><td>0.2121</td></tr><tr><td>film</td><td>0.2292</td><td>0.0626</td><td>0.1970</td><td>0.1993</td></tr><tr><td>first-additive</td><td>0.2292</td><td>0.0573</td><td>0.2064</td><td>0.2022</td></tr><tr><td>first-concat</td><td>0.2292</td><td>0.4011</td><td>0.4096</td><td>0.1893</td></tr><tr><td>gating</td><td>0.2292</td><td>0.0405</td><td>0.1795</td><td>0.1953</td></tr><tr><td>last-additive</td><td>0.2292</td><td>0.1526</td><td>0.2113</td><td>0.1780</td></tr><tr><td>last-concat</td><td>0.2292</td><td>0.1392</td><td>0.1743</td><td>0.1666</td></tr><tr><td>middle-additive</td><td>0.2292</td><td>0.0632</td><td>0.2131</td><td>0.1699</td></tr><tr><td>middle-concat</td><td>0.2292</td><td>0.1955</td><td>0.1990</td><td>0.1916</td></tr><tr><td rowspan="11">PatchTST</td><td>orthogonal</td><td>0.2292</td><td>0.0566</td><td>0.1928</td><td>0.2187</td></tr><tr><td>cfa</td><td>0.2785</td><td>0.2727</td><td>0.2763</td><td>0.2740</td></tr><tr><td>film</td><td>0.2785</td><td>0.2244</td><td>0.3013</td><td>0.2805</td></tr><tr><td>first-additive</td><td>0.2785</td><td>0.2734</td><td>0.2734</td><td>0.2734</td></tr><tr><td>first-concat</td><td>0.2785</td><td>0.2734</td><td>0.2734</td><td>0.2734</td></tr><tr><td>gating last-additive</td><td>0.2785</td><td>0.3194</td><td>0.3368</td><td>0.4090</td></tr><tr><td>last-concat</td><td>0.2785 0.2785</td><td>0.2150</td><td>0.2700</td><td>0.2620</td></tr><tr><td>middle-additive</td><td>0.2785</td><td>0.1994</td><td>0.2429</td><td>0.2712</td></tr><tr><td>middle-concat</td><td></td><td>0.2789</td><td>0.2500</td><td>0.2985</td></tr><tr><td>orthogonal</td><td>0.2785 0.2785</td><td>0.3009 0.2343</td><td>0.3126</td><td>0.3604</td></tr><tr><td>cfa</td><td>0.2295</td><td>0.2325</td><td>0.2622 0.2325</td><td>0.3478 0.2325</td></tr><tr><td rowspan="10">TiDE</td><td>film</td><td>0.2295</td><td>0.2330</td><td>0.2330</td><td>0.2330</td></tr><tr><td>first-additive</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>0.2295</td><td>1977.9946</td><td>2383.0618</td><td>1956.9061</td></tr><tr><td>first-concat</td><td>0.2295</td><td>0.1448</td><td>0.2246</td><td>0.2288</td></tr><tr><td>gating</td><td>0.2295</td><td>0.2318</td><td>0.2318</td><td>0.2318</td></tr><tr><td>last-additive</td><td>0.2295</td><td>0.1075</td><td>0.2123</td><td>0.2319</td></tr><tr><td>last-concat</td><td>0.2295</td><td>0.1340</td><td>0.2218</td><td>0.2267</td></tr><tr><td>middle-additive</td><td>0.2295</td><td>0.2324</td><td>0.2324</td><td>0.2324</td></tr><tr><td>middle-concat</td><td>0.2295</td><td>0.2329</td><td>0.2329</td><td>0.2329</td></tr><tr><td>orthogonal</td><td>0.2295</td><td>0.2305</td><td>0.2305</td><td>0.2305</td></tr><tr><td rowspan="10">iTransformer</td><td>cfa</td><td>0.1840</td><td>0.2090</td><td>0.1762</td><td>0.2088</td></tr><tr><td>film</td><td>0.1840</td><td>0.0417</td><td>0.1836</td><td>0.1940</td></tr><tr><td>first-additive</td><td>0.1840</td><td>0.0394</td><td>0.1747</td><td>0.1810</td></tr><tr><td>first-concat</td><td>0.1840</td><td>0.0408</td><td>0.1837</td><td>0.1926</td></tr><tr><td>gating</td><td>0.1840</td><td>0.0776</td><td>0.2043</td><td>0.1945</td></tr><tr><td>last-additive</td><td>0.1840</td><td>0.1210</td><td>0.2062</td><td>0.1902</td></tr><tr><td>last-concat</td><td>0.1840</td><td>0.1046</td><td>0.1905</td><td>0.1728</td></tr><tr><td>middle-additive</td><td>0.1840</td><td>0.0445</td><td>0.2040</td><td>0.2018</td></tr><tr><td>middle-concat</td><td>0.1840</td><td>0.0579</td><td>0.1776</td><td>0.2001</td></tr><tr><td>orthogonal</td><td>0.1840</td><td>0.0607</td><td>0.2321</td><td>0.2218</td></tr></table>

## F.3 SIGNAL PERTURBATION EXPERIMENTS

## F.3.1 SHUFFLE RESULTS

To confirm that the incorrect signal is carried by the text-time series pairing rather than the text alone, we shuffle all annotations across time points and recompute all metrics. Table 10 shows full results for MINE and InfoNCE, which both drop their in MI estimates significantly after shuffling, with no change in irrelevant text, confirming that the inverted signal is a consistent property of the (text, Y<sub>future</sub>) pairing and not an artifact of embedding distribution.

## F.3.2 SIGNAL PERTURBATIONS

Our sine signal in MMTT-Bench is not representative of real data by construction, because the information content of every annotation is known, and it is this property that makes estimator

Table 10: Comparison of conditional MI before and after dataset shuffle
<table><tr><td rowspan="2">Category</td><td colspan="2">MINE</td><td colspan="2">InfoNCE</td></tr><tr><td> $\mathrm { P r e } \mathrm { } \mathrm { } \mathrm { } \cdot$ </td><td>Post-</td><td> $\mathrm { P r e } \mathrm { } \mathrm { } \mathrm { } \cdot$ </td><td>Post-</td></tr><tr><td></td><td>Shuffle</td><td>Shuffle</td><td>Shuffle</td><td>Shuffle</td></tr><tr><td>Correct</td><td>0.56±0.06</td><td> $0 . 0 8 { \pm } 0 . 0 5$ </td><td> $0 . 6 4 { \pm } 0 . 0 2$ </td><td> $0 . 1 6 { \pm } 0 . 0 2$ </td></tr><tr><td>Incorrect</td><td> $0 . 1 7 { \pm } 0 . 0 6$ </td><td> $0 . 1 2 { \pm } 0 . 0 5$ </td><td> $0 . 2 4 { \pm } 0 . 0 2$ </td><td> $0 . 1 7 { \pm } 0 . 0 2$ </td></tr><tr><td>Irrelevant</td><td> $0 . 0 4 { \pm } 0 . 0 5$ </td><td> $0 . 0 5 { \pm } 0 . 0 4$ </td><td> $0 . 1 2 { \pm } 0 . 0 2$ </td><td> $0 . 1 2 { \pm } 0 . 0 2$ </td></tr></table>

evaluator possible. That said, MMTT-Bench generation is extensible, and we demonstrate that estimators still rank correct text highest with a noisy sine signal at four different noise levels in Table 11

Table 11: Conditional MI $I ( X _ { \mathrm { t e x t } } ; Y \mid X _ { \mathrm { t s } } )$ (nats, $\mathrm { m e a n } \pm \mathrm { s t d } )$ on the noisy sine benchmark. Observation noise σ is added to the signal, annotations are unchanged.
<table><tr><td>Estimator</td><td>σ</td><td>Correct</td><td>Incorrect</td><td>Irrelevant</td></tr><tr><td rowspan="4">KSG</td><td>0.05</td><td> ${ \bf 0 . 0 6 7 \pm 0 . 0 2 8 }$ </td><td> $- 0 . 2 6 7 \pm 0 . 0 3 2$ </td><td> $- 0 . 2 1 0 \pm 0 . 0 3 8$ </td></tr><tr><td>0.1</td><td> ${ \bf 0 . 0 1 9 \pm 0 . 0 3 5 }$ </td><td> $- 0 . 2 8 8 \pm 0 . 0 4 2$ </td><td> $- 0 . 2 2 3 \pm 0 . 0 3 8$ </td></tr><tr><td>0.2</td><td> ${ \bf 0 . 0 8 7 \pm 0 . 0 3 5 }$ </td><td> $- 0 . 1 5 6 \pm 0 . 0 3 7$ </td><td> $- 0 . 1 4 4 \pm 0 . 0 4 2$ </td></tr><tr><td>0.4</td><td> $\mathbf { 0 . 1 4 3 \pm 0 . 0 3 6 }$ </td><td> $- 0 . 0 4 6 \pm 0 . 0 4 2$ </td><td> $- 0 . 1 1 0 \pm 0 . 0 4 7$ </td></tr><tr><td rowspan="4">MINE</td><td>0.05</td><td> $\mathbf { 0 . 4 3 3 \pm 0 . 0 4 7 }$ </td><td> $0 . 1 3 5 \pm 0 . 0 6 0$ </td><td> $0 . 0 2 0 \pm 0 . 0 5 6$ </td></tr><tr><td>0.1</td><td> $\mathbf { 0 . 4 7 5 \pm 0 . 0 6 3 }$ </td><td> $0 . 1 2 8 \pm 0 . 0 4 8$ </td><td> $0 . 0 4 5 \pm 0 . 0 5 3$ </td></tr><tr><td>0.2</td><td> $\mathbf { 0 . 4 8 2 \pm 0 . 0 6 5 }$ </td><td> $0 . 2 1 7 \pm 0 . 0 5 1$ </td><td> $0 . 0 8 9 \pm 0 . 0 5 7$ </td></tr><tr><td>0.4</td><td> $\mathbf { 0 . 4 5 1 \pm 0 . 0 4 2 }$ </td><td> $0 . 3 4 7 \pm 0 . 0 4 1$ </td><td> $0 . 1 8 5 \pm 0 . 0 4 9$ </td></tr><tr><td rowspan="4">InfoNCE</td><td>0.05</td><td> ${ \bf 0 . 0 6 7 \pm 0 . 0 2 9 }$ </td><td> $0 . 4 5 3 \pm 0 . 0 3 6$ </td><td> $0 . 2 8 6 \pm 0 . 0 2 2$ </td></tr><tr><td>0.1</td><td> $\mathbf { 0 . 7 7 1 \pm 0 . 0 3 6 }$ </td><td> $0 . 5 0 1 \pm 0 . 0 3 8$ </td><td> $0 . 3 1 2 \pm 0 . 0 3 5$ </td></tr><tr><td>0.2</td><td>0.804 ± 0.027</td><td> $0 . 6 2 4 \pm 0 . 0 3 8$ </td><td> $0 . 4 1 1 \pm 0 . 0 4 7$ </td></tr><tr><td>0.4</td><td> ${ \bf 0 . 8 6 0 \pm 0 . 0 4 9 }$ </td><td> $0 . 8 0 3 \pm 0 . 0 3 5$ </td><td> $0 . 5 1 5 \pm 0 . 0 4 6$ </td></tr><tr><td rowspan="4">CCA</td><td>0.05</td><td> ${ \bf 0 . 3 0 9 \pm 0 . 0 1 9 }$ </td><td> $0 . 0 3 8 \pm 0 . 0 0 7$ </td><td> $0 . 0 1 3 \pm 0 . 0 0 3$ </td></tr><tr><td>0.1</td><td> ${ \bf 0 . 2 8 3 \pm 0 . 0 2 2 }$ </td><td> $0 . 0 6 0 \pm 0 . 0 0 7$ </td><td> $0 . 0 0 9 \pm 0 . 0 0 3$ </td></tr><tr><td>0.2</td><td> $\mathbf { 0 . 2 2 3 \pm 0 . 0 1 9 }$ </td><td> $0 . 0 4 9 \pm 0 . 0 1 0$ </td><td> $0 . 0 1 4 \pm 0 . 0 0 4$ </td></tr><tr><td>0.4</td><td> ${ \bf 0 . 1 7 0 \pm 0 . 0 1 4 }$ </td><td> $0 . 0 3 7 \pm 0 . 0 0 6$ </td><td> $0 . 0 1 0 \pm 0 . 0 0 4$ </td></tr><tr><td rowspan="4">V-Information</td><td>0.05</td><td> ${ \bf 0 . 2 6 4 \pm 0 . 0 1 5 }$ </td><td> $0 . 0 4 2 \pm 0 . 0 0 7$ </td><td> $0 . 0 1 5 \pm 0 . 0 0 4$ </td></tr><tr><td>0.1</td><td> $\mathbf { 0 . 2 5 3 \pm 0 . 0 1 5 }$ </td><td> $0 . 0 6 6 \pm 0 . 0 0 8$ </td><td> $0 . 0 0 9 \pm 0 . 0 0 4$ </td></tr><tr><td>0.2</td><td> ${ \bf 0 . 2 2 8 \pm 0 . 0 1 5 }$ </td><td> $0 . 0 5 9 \pm 0 . 0 1 1$ </td><td> $0 . 0 1 7 \pm 0 . 0 0 6$ </td></tr><tr><td>0.4</td><td> ${ \bf 0 . 2 2 0 \pm 0 . 0 1 5 }$ </td><td> $0 . 0 5 4 \pm 0 . 0 0 8$ </td><td> $0 . 0 1 2 \pm 0 . 0 0 7$ </td></tr><tr><td rowspan="4"> $\mathrm { P I D } \left( \mathrm { U } _ { \mathrm { t e x t } } \right)$ </td><td>0.05</td><td> ${ \bf 0 . 3 6 2 \pm 0 . 0 1 7 }$ </td><td> $0 . 0 0 2 \pm 0 . 0 0 2$ </td><td> $0 . 0 0 0 \pm 0 . 0 0 1$ </td></tr><tr><td>0.1</td><td> ${ \bf 0 . 2 9 8 \pm 0 . 0 1 2 }$ </td><td> $0 . 0 0 9 \pm 0 . 0 0 3$ </td><td> $0 . 0 0 2 \pm 0 . 0 0 1$ </td></tr><tr><td>0.2</td><td> $\mathbf { 0 . 1 5 2 \pm 0 . 0 1 1 }$ </td><td> $0 . 0 0 5 \pm 0 . 0 0 2$ </td><td> $0 . 0 0 6 \pm 0 . 0 0 3$ </td></tr><tr><td>0.4</td><td> $\mathbf { 0 . 0 4 2 \pm 0 . 0 0 8 }$ </td><td> $0 . 0 0 7 \pm 0 . 0 0 3$ </td><td> $0 . 0 0 5 \pm 0 . 0 0 2$ </td></tr></table>

## F.3.3 TEMPORAL JITTER

We also introduce controlled perturbations in the form of temporal jitters, where each annotation is replaced by the correct text of a nearby time point, so coupling decays while the text stays lexically identically to the correct corpus. Table 12 shows under temporal jitter, estimated MI decays monotonically to the irrelevant-text level as displacement grows. However downstream performance is not monotone, it is instead worst (maximized) at an intermediate displacement and then partially recovers. The interpretation is that slightly-misaligned text is worse than useless, it is locally plausible and therefore actively misleading, whereas heavily-displaced text is simply de-correlated.

10)

Table 12: MI estimates under temporal jitter, where each annotation is replaced by the correct text or a nearby timepoint with σ as the standard deviation of the displacement, in annotation steps.
<table><tr><td>Jitter σ</td><td>KSG</td><td>MINE</td><td>InfoNCE</td><td>CCA</td><td>V-Info</td><td>PID</td><td>mean MSE</td></tr><tr><td>0 (correct)</td><td>0.199</td><td>0.419</td><td>0.557</td><td>0.308</td><td>0.261</td><td>0.393</td><td>0.100</td></tr><tr><td>0.5</td><td>0.027</td><td>0.260</td><td>0.477</td><td>0.139</td><td>0.138</td><td>0.277</td><td>0.211</td></tr><tr><td>1</td><td>-0.090</td><td>0.154</td><td>0.419</td><td>0.080</td><td>0.084</td><td>0.197</td><td>0.265</td></tr><tr><td>2</td><td>-0.111</td><td>0.113</td><td>0.373</td><td>0.049</td><td>0.053</td><td>0.083</td><td>0.285</td></tr><tr><td>4</td><td>-0.130</td><td>0.050</td><td>0.369</td><td>0.032</td><td>0.035</td><td>0.029</td><td>0.314</td></tr><tr><td>8</td><td>-0.116</td><td>0.053</td><td>0.374</td><td>0.017</td><td>0.019</td><td>0.0004</td><td>0.291</td></tr><tr><td>16</td><td>-0.125</td><td>0.049</td><td>0.371</td><td>0.015</td><td>0.016</td><td>0.002</td><td>0.280</td></tr></table>

Table 13: MMTT-Bench and real world dataset feature comparison. Counts are annotated points per text category. Trend $R ^ { 2 }$ is the coefficient of determination of an ordinary least-squares fit on time; lag-1 ACF is the autocorrelation of the raw series; seasonal ACF is the autocorrelation of the first-differenced series at the natural period (12 monthly, 52 weekly, 365 daily, 252 trading days), differenced so that persistence is not mistaken for seasonality; variance ratio is Var(second half) / Var(first half) and mean shift is the change in mean between halves in units of the series standard deviation, the two together indicating departure from stationarity.
<table><tr><td>Dataset</td><td>Frequency</td><td>Train</td><td>Val</td><td>Test</td><td>Trend  $R ^ { 2 }$ </td><td>Lag-1 ACF</td><td>Seasonal ACF</td><td>Var. ratio</td><td>Mean shift</td></tr><tr><td>MMTT- Bench</td><td>synthetic</td><td>1280</td><td>384</td><td>384</td><td>0.00</td><td>0.05</td><td>0.70</td><td>1.10</td><td>0.05</td></tr><tr><td>Agriculture</td><td>monthly</td><td>347</td><td>50</td><td>99</td><td>0.86</td><td>0.99</td><td>0.00</td><td>0.45</td><td>1.53</td></tr><tr><td>Climate</td><td>monthly</td><td>347</td><td>50</td><td>99</td><td>0.00</td><td>0.23</td><td>0.20</td><td>0.75</td><td>0.11</td></tr><tr><td>Energy</td><td>weekly</td><td>1035</td><td>149</td><td>295</td><td>0.77</td><td>1.00</td><td>0.07</td><td>2.27</td><td>1.71</td></tr><tr><td>Public Health</td><td>weekly</td><td>972</td><td>140</td><td>277</td><td>0.04</td><td>0.95</td><td>0.31</td><td>0.91</td><td>0.44</td></tr><tr><td>Social Good</td><td>monthly</td><td>630</td><td>90</td><td>180</td><td>0.09</td><td>0.95</td><td>0.79</td><td>1.06</td><td>0.84</td></tr><tr><td>Traffic</td><td>monthly</td><td>371</td><td>54</td><td>106</td><td>0.88</td><td>0.96</td><td>0.95</td><td>0.62</td><td>1.67</td></tr><tr><td>FinTexTS (median of</td><td>trading day</td><td>784</td><td>260</td><td>260</td><td>0.83</td><td>1.00</td><td>0.00</td><td>3.20</td><td>1.56</td></tr></table>

## G REAL WORLD DATASETS

Time-MMD Liu et al. (2024) text annotations are sourced from Google web search results, filtered and disentangled from predictions by LLMs and summarized for usability. We construct the TimeMMD datasets using sliding windows based on the approach taken by (Lee et al., 2026b), excluding the Economy and Security domains as these have fewer than 300 training samples per category.

Since Time-MMD contains only real text with no ground truth quality labels, we construct two annotation categories. All processed Time-MMD text is labelled correct, acknowledging that it is not verified ground truth but represents the best available domain-relevant information. Irrelevant text is constructed by randomly sampling processed text from a different domain, following previous work by Lee et al. (2026b).

We add an additional real world dataset FinTexTS Lee et al. (2026a), which combines stock prices with news articles paired based on their semantic embeddings. Due to compute constraints we select a subset of ten stocks: AMD, BA, COST, DIS, GOOGL, INTC, NFLX, NVDA, T and TSLA.

Table 14: Per-estimator hyperparameters
<table><tr><td>Estimator</td><td>Parameter</td><td>Value</td></tr><tr><td>KSG</td><td>Neighbors k</td><td>1</td></tr><tr><td rowspan="4">MINE</td><td>Training iterations</td><td>500</td></tr><tr><td>Hidden dimension</td><td>128</td></tr><tr><td>Batch size N</td><td>256 (log N ≈ 5.55 nat ceiling) 10⁻4</td></tr><tr><td>Learning rate</td><td></td></tr><tr><td rowspan="4">InfoNCE</td><td>Training iterations</td><td>500</td></tr><tr><td>Hidden dimension</td><td>128</td></tr><tr><td>Batch size N</td><td>256 (log N ≈ 5.55 nat ceiling)</td></tr><tr><td>Learning rate</td><td> $1 0 ^ { - 3 }$ </td></tr><tr><td>CCA</td><td>Regularization ε</td><td> $1 0 ^ { - 5 }$ </td></tr><tr><td>V-information</td><td>Ridge CV folds</td><td>5</td></tr><tr><td rowspan="4">PID</td><td>Time-series clusters  $C _ { \mathrm { t s } }$ </td><td>4</td></tr><tr><td>Text clusters  $C _ { \mathrm { t e x t } }$ </td><td>16</td></tr><tr><td>Target bins  $n _ { Y }$ </td><td>4</td></tr><tr><td>Target channel y</td><td>11</td></tr></table>

Table 15: Final hyperparameters for Time-MMD experiment
<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Embedding model</td><td>sentence-transformers/all-distilroberta-v1</td></tr><tr><td>Time-series tokenizer</td><td>Patch mean</td></tr><tr><td>Horizon steps</td><td>12</td></tr><tr><td>Lookback steps</td><td>12</td></tr><tr><td>Patch length</td><td>4</td></tr><tr><td>Stride</td><td>2</td></tr><tr><td>PCA dimension</td><td>16</td></tr><tr><td>Bootstrap resamples</td><td>20</td></tr><tr><td>Embedding strategy</td><td>equal-width bins</td></tr></table>

## G.1 FINAL PARAMETERS

## G.1.1 TIME-MMD PARAMETERS

## G.1.2 FINTEXTS PARAMETERS

Table 16: Final hyperparameters for FinTexTS experiment
<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Embedding model</td><td>answerdotai/ModernBERT-large</td></tr><tr><td>Time-series tokenizer</td><td>Patch mean</td></tr><tr><td>Horizon steps</td><td>3</td></tr><tr><td>Lookback steps</td><td>64</td></tr><tr><td>Patch length</td><td>8</td></tr><tr><td>Stride</td><td>8</td></tr><tr><td>PCA dimension</td><td>5</td></tr><tr><td>Bootstrap resamples</td><td>20</td></tr><tr><td>Embedding strategy</td><td>equal-width bins</td></tr></table>

## G.2 FULL RESULTS

Table 17: Alignment (difference in $I ( X _ { \mathrm { t s } } , X _ { \mathrm { t e x t } } ; Y )$ between correctly-paired text and samedomain text drawn from the wrong timestamp), and conditional $\Delta M \dot { I } \ = \ I ( X _ { \mathrm { t e x t } } ; Y | X _ { \mathrm { t s } } )$ for correctly-paired text, calculated across seven real-world datasets, and reported alongside $\sigma _ { \mathrm { a l i g n } } =$ $\sqrt { ( \sigma _ { \mathrm { c o r r e c t } } ^ { 2 } + \sigma _ { \mathrm { i n c o r r e c t } } ^ { 2 } ) }$ and align $\mathbf { Z } ,$ where a large positive means the \*pairing\* carries information; $z \approx 0$ means the text is informative in aggregate but not tied to the timestamp it is attached to. Conditional σ is the bootstrap standard deviation of that paired difference, and conditional $z \ = \ \Delta M I _ { \prime }$ conditionalσ tests it against zero. FinTexTS z-scores are computed per stock and Stouffer-pooled across the ten analysis stocks.
<table><tr><td>Dataset</td><td>Estimator</td><td>align ∆MI</td><td>align σ</td><td>align z</td><td>conditional ∆MI</td><td>conditional σ</td><td>conditional Z</td></tr><tr><td rowspan="5">Agriculture</td><td>KSG</td><td>0.4361</td><td>0.0767</td><td>5.6857</td><td>-0.192</td><td>0.0743</td><td>-2.5854</td></tr><tr><td>MINE</td><td>0.0055</td><td>0.1119</td><td>0.0487</td><td>0.0098</td><td>0.0976</td><td>0.1</td></tr><tr><td>InfoNCE</td><td>-0.0555</td><td>0.0631</td><td>-0.8803</td><td>2.0315</td><td>0.11</td><td>18.4614</td></tr><tr><td>CCA</td><td>0.0591</td><td>0.0857</td><td>0.6896</td><td>0.3622</td><td>0.0491</td><td>7.3752</td></tr><tr><td>V-info PID</td><td>0.0081 0.0216</td><td>0.0097 0.0287</td><td>0.8334 0.7531</td><td>0.0157 0.1323</td><td>0.0038 0.02</td><td>4.1327 6.616</td></tr><tr><td rowspan="6">Climate</td><td> $\left( U _ { \mathrm { t e x t } } + S \right)$  KSG</td><td>0.018</td><td>0.0873</td><td>0.2056</td><td></td><td></td><td>-1.0496</td></tr><tr><td>MINE</td><td>0.0296 -0.0065</td><td>0.0692</td><td>0.4284</td><td>-0.0605 0.1673</td><td>0.0577 0.058</td><td>2.8864</td></tr><tr><td>InfoNCE</td><td></td><td>0.0298</td><td>-0.2193</td><td>0.0838</td><td>0.0136</td><td>6.1485</td></tr><tr><td>CCA V-info</td><td>0.0865 0.0215</td><td>0.066</td><td>1.3122</td><td>0.3317</td><td>0.0412</td><td>8.046</td></tr><tr><td>PID</td><td>-0.0284</td><td>0.0188 1.141 0.0394 -0.7204</td><td></td><td>0.074 0.1812</td><td>0.0122 0.0223</td><td>6.0662 8.144</td></tr><tr><td>Energy</td><td> $( U _ { \mathrm { t e x t } } + S )$  KSG MINE InfoNCE CCA V-info</td><td>0.749 0.0009 0.2057 0.0277</td><td>0.0804 0.0725 0.0868</td><td>9.3142 0.0118 2.3697</td><td>-2.323 0.0134 0.8293</td><td>0.0648 0.0351 0.0827</td><td>-35.8348 0.3817 10.0276</td></tr><tr><td>Public Health</td><td>PID  $( U _ { \mathrm { t e x t } } + S )$  KSG MINE InfoNCE CCA V-info PID  $( U _ { \mathrm { t e x t } } + S )$ </td><td>0.003 0.0514 0.5618 0.0145 0.1232 0.1191 0.089 0.0814</td><td>0.0964 0.0058 0.0137 0.0545 0.1294 0.0549 0.1111 0.0239</td><td>0.2869 0.5155 3.7486 10.306 0.112 2.2443 1.0717 3.7306</td><td>0.2211 0.0058 0.1143 -0.7536 0.0554 0.9501 0.3308 0.1077</td><td>0.0234 0.0016 0.01 0.0565 0.0973 0.0609 0.0332</td><td>9.4382 3.7011 11.3978 -13.3419 0.5689 15.6087 9.971</td></tr><tr><td>Social Good</td><td>KSG MINE InfoNCE CCA V-info</td><td>-0.3016 -0.1378 -0.0531 -0.0719</td><td>0.0176 0.0771 0.1204 0.1036 0.1562 0.0202</td><td>4.6244 -3.9104 -1.1445 -0.5121 -0.4602 -0.2254</td><td>0.1546 -0.8073 -0.0501 0.4981 0.1053 0.01</td><td>0.0175 0.0135 0.0549 0.0519 0.097 0.0135</td><td>6.1539 11.4341 -14.6978 -0.9655 5.1347 7.7821</td></tr><tr><td>Traffic</td><td>PID  $( U _ { \mathrm { t e x t } } + S )$  KSG MINE InfoNCE CCA V-info</td><td>-0.0046 -0.0132 0.1823 -0.0125 -0.0234</td><td>0.0145 0.0787 0.0758 0.0322 0.5337 0.016</td><td>-0.9098 2.3173 -0.1644 -0.7268 0.1738</td><td>0.0525 -3.0186 -0.1196</td><td>0.0021 0.0077 0.101 0.1054</td><td>4.6964 6.7995 -29.8843 -1.135</td></tr><tr><td></td><td>PID  $\left( U _ { \mathrm { t e x t } } + S \right)$ </td><td>0.0928 0.007 0.0073</td><td>0.0317</td><td>0.4366 0.2294</td><td>0.2064 0.3378 0.0116 0.1107</td><td>0.0335 0.0839 0.0046 0.0235</td><td>6.1656 4.0287 2.5317 4.7117</td></tr><tr><td>FinTexTS</td><td>KSG MINE InfoNCE CCA V-info PID  $\left( U _ { \mathrm { t e x t } } + S \right)$ </td><td>0.0575 0.1013 -0.0052 0.0081 0.0004 0.0132</td><td>0.0303 0.1188 0.0765 0.105 0.0051 0.0163</td><td>6.1426 2.8314 -0.2542 0.2496 0.3364 2.6543</td><td>-1.9379 0.0023 0.2334 0.0268 0.0009 0.0992</td><td>0.0395 0.0916 0.0489 0.009 0.0005 0.0118</td><td>-159.504 0.081 16.3132 10.2013 6.1402 27.1126</td></tr></table>

Table 18: Performance results for real world datasets, constructed with original text (correct), text shuffled to a new timestamp (incorrect) and text from a different domain (irrelevant). Model performance is reported as MSE median and interquartile range across 9 different time series transformer architectures (Autoformer, DLinear, FEDformer, FiLM, Informer, Nonstationary Transformer, PatchTST, TiDE, iTransformer), each with 10 fusion strategies (cfa, film, first-additive, first-concat, gating, last-additive, last-concat, middle-additive, middle-concat, orthogonal). MSE is used as the performance metric to align better with other works using these datasets. We report median reduction in MSE versus no-text (positive = text improves).
<table><tr><td>Dataset</td><td>Correct</td><td>Incorrect</td><td>Irrelevant</td></tr><tr><td>Agriculture</td><td>-2.5% [-12.2, +4.0]</td><td>-1.2% [-10.0, +4.2]</td><td>-1.7% [-9.1, +2.5]</td></tr><tr><td>Climate</td><td>+1.9% [-1.0, +4.7]</td><td>+1.6% [-1.4, +4.9]</td><td>+1.4% [-1.1, +4.2]</td></tr><tr><td>Energy</td><td>-6.7% [-27.8, +2.9]</td><td>-7.8% [-25.5, +3.9]</td><td>-7.0% [-34.7, +3.5]</td></tr><tr><td>PublicHealth</td><td>-5.1% [-11.2, +2.6]</td><td>-5.2% [-13.2, +1.9]</td><td>-4.3% [-12.3, +2.0]</td></tr><tr><td>SocialGood</td><td>-1.3% [-13.1, +9.5]</td><td>-0.7% [-12.8, +9.3]</td><td>-2.6% [-12.1, +5.7]</td></tr><tr><td>Traffic</td><td>-1.8% [-14.0, +9.6]</td><td>-8.1% [-21.8, -0.5]</td><td>-8.2% [-26.9, +2.7]</td></tr><tr><td>FinTexTS</td><td>-0.6% [-25.6, +8.1]</td><td>-5.1% [-22.3, +5.6]</td><td>-4.8% [-24.2, +6.5]</td></tr></table>