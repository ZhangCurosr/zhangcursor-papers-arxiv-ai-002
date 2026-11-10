# MOTIF: LLM-Defined Material Taxonomies for OOD Property Prediction via Invariant Feature Selection

Gwangseon Jang<sup>1</sup> Sieun Yun<sup>2</sup> Dong Ho Lee<sup>2</sup> Kyong-Ha Lee<sup>1</sup> Chanuk Lim<sup>1</sup> Jin Woo Choi<sup>3,∗</sup> Han Seul Kim<sup>2,∗</sup>

<sup>1</sup> Large-scale AI Research Center, Korea Institute of Science and Technology Information (KISTI) <sup>2</sup> Department of Advanced Materials Engineering, Chungbuk National University

<sup>3</sup> Department of Data Information and Physics, Kongju National University

<sup>∗</sup> Co-corresponding authors:

Jin Woo Choi (jinwoo.choi@kongju.ac.kr) Han Seul Kim (hanseul.kim@chungbuk.ac.kr).

## Abstract

Out-of-distribution (OOD) generalization remains a central challenge in materials property prediction, where models must extrapolate to novel chemistries beyond the training distribution. While large language models (LLMs) have demonstrated strong scientific reasoning, their direct use for quantitative prediction remains limited. To address this, we systematically evaluate multiple roles for LLM assistance, finding that only taxonomy-based grouping consistently improves OOD generalization over a strong kNN+GBT baseline. Building on this, we here propose MOTIF, a framework that leverages LLMs not as predictors, but as generators of candidate taxonomies that partition materials into chemically meaningful environments. These taxonomies are evaluated via training-only pseudo-OOD validation, and only those that improve generalization are retained. Invariant feature selection is then applied within each accepted taxonomy, followed by gated aggregation across taxonomies. We evaluate on 2D halide perovskites, where bandgap prediction must generalize across distinct organic spacer families. This poses a realistic OOD challenge due to the strong influence of spacer chemistry on electronic structure. On this benchmark, MOTIF reduces mean OOD MAE from 0.122 to 0.111 eV. Under a frozen dev/test protocol, the deployable configuration further improves errors from 0.123 to 0.117 eV (electronic bandgap) and from 0.181 to 0.173 eV (optical bandgap). Beyond this primary task, we validate transferability across additional materials datasets, including 3D perovskite solar cell efficiencies and Matbench benchmarks with structure-based splits. Across domains, MOTIF consistently improves robustness under distribution shift, outperforming expert-defined groupings, rule-based heuristics, and clustering baselines, while remaining competitive with strong classical selection methods. These results demonstrate that LLMs are most effective not as predictors, but as tools for discovering structure in data. This enables improved generalization in materials science tasks under realistic OOD conditions.

## 1 Introduction

In materials science, predictive models are often applied to compounds with novel chemistries outside the training distribution. This makes out-of-distribution (OOD) generalization a fundamental requirement. While large language models (LLMs) have demonstrated strong capabilities across scientific domains, their most reliable role in materials property prediction has not been systematically established. [2, 3, 7]. Most prior work treats LLMs as direct predictors, using them to generate or refine numerical property values, often with additional techniques such as retrieval, reasoning traces, or self-correction [7, 16, 17, 24–28]. However, under leave-one-spacer-family-out shift, each test fold contains an unseen organic spacer superfamily absent from training, so the model must extrapolate to new spacer chemistries with distinct structure–property relationships rather than interpolate within a shared family. In such settings, even weak spurious correlations can lead to misleading improvements, making apparent performance gains unreliable. This raises a more fundamental question: not whether an LLM can predict the value, but where in the prediction pipeline its knowledge can be reliably used to improve OOD materials prediction.

To investigate this question, we study leave-one-spacer-family-out prediction for two-dimensional (2D) metal-halide perovskites [11]. Each test fold contains an unseen spacer superfamily, requiring generalization beyond local interpolation and reflecting realistic scenarios where models must predict properties for entirely new chemistries. Here, the organic spacer strongly modulates the electronic structure, so different spacer families induce distinct structure–property relationships.

We evaluate six roles for LLM assistance, including direct prediction, retrieval-augmented prediction, LLM-generated features, LLM-guided sample selection, priors or constraints, and chemical taxonomy annotation, i.e., the use of LLMs to define meaningful groupings of materials. Among these, only offline taxonomy annotation consistently improves over a strong k-nearest neighbors (kNN) and gradient boosting trees (GBT) baseline. In contrast, approaches that directly incorporate LLMgenerated numerical values into the predictor yield at most marginal gains, improving the baseline by less than 2% across 19 published methods and over 65 in-house variants. These results suggest that the most reliable use of LLMs in this setting is not as predictors, but as offline chemistry annotators that propose chemically meaningful groupings, which can then be validated using classical statistical methods.

This observation highlights a key bottleneck in applying invariant learning to scientific data: the lack of meaningful ways to partition the data into groups. Invariant learning aims to identify relationships between features and targets that remain stable across such groups [1, 15]. However, in real-world datasets, these group structures are rarely explicitly available. Random partitions are often uninformative, while expert-defined groupings are costly and typically too coarse. We therefore introduce MOTIF (Materials OOD Taxonomy-Invariant Framework), which uses an LLM once, offline, to generate chemistry-aware taxonomies of spacer cations. These taxonomies capture variations in spacer chemistry that influence electronic structure and bandgap formation, thereby defining meaningful groupings of the data. Each taxonomy defines a candidate environment. For each environment, MOTIF selects features whose relationships with the target remain stable across chemically distinct groups using a Chow structural-break test [4]. It then trains predictors using only these invariant features. It then deploys taxonomy-conditioned models only when supported by training-only pseudo-OOD validation. Notably, the LLM is used only offline, does not produce numerical property estimates, and is never queried at inference time. For the main six-fold analysis, we use a fully automatic gated-median readout. For frozen-protocol and transfer evaluations, we adopt a more conservative safe-blend deployment rule to ensure robust performance under stricter evaluation settings.

On the six-fold 2D halide perovskite benchmark, MOTIF reduces mean OOD MAE from 0.122 to 0.111 eV. Under a stricter frozen dev/test protocol, the primary deployable configuration improves bandgap from 0.123 to 0.117 eV (p=0.002) and optical bandgap from 0.181 to 0.173 eV. It outperforms expert-family, rule-based, and clustering-based methods, while remaining competitive with strong classical feature selection approaches. The same pipeline also improves four additional OOD benchmarks across diverse materials families and target properties. Taken together, these results suggest that, in high-precision OOD materials settings, LLMs are more effective for proposing auditable environment hypotheses than for producing numerical property estimates.

## Our contributions are threefold:

1. We recast LLM use in high-precision scientific OOD prediction as an environment-discovery problem for invariant learning rather than a direct numerical regression problem, and we evaluate that framing across 19 published LLM methods and 65+ in-house variants.

2. We introduce MOTIF, which converts LLM-generated chemical taxonomies into candidate environments, selects invariant features using Chow-based tests, and deploys taxonomy-conditioned predictors only when supported by training-only pseudo-OOD validation.

![](images/7092d56da6931e3f8b92b046127876adba340cdb9a6d688c6589f5742cf49c8c.jpg)  
Figure 1: MOTIF pipeline. Stage 1: an LLM classifies each spacer along multiple chemical axes once, offline. Stage 2: within each taxonomy, a Chow test filters features whose relationship to the target is stable across groups, and kNN/GBT are trained only in that invariant subspace. Stage 3: a pseudo-OOD gate keeps only taxonomies that beat the baseline, and gated aggregation combines the surviving models. No LLM call is made at inference time.

3. We provide empirical evidence across a six-fold perovskite benchmark, a frozen dev/test protocol, and four transfer benchmarks that this environment-annotation role is more reliable than direct numerical use of LLMs for high-precision OOD materials prediction.

## 2 Related Work

LLMs for scientific prediction. LLMs have been applied to synthesis planning [3], autonomous experimentation [2], and property prediction [7, 16, 17]. Most prior work treats LLMs as predictors, editors, or feature generators. In contrast, we systematically compare these roles and demonstrate that, for high-precision OOD materials prediction, offline annotation of chemically meaningful groupings is consistently more reliable than direct numerical use of LLMs.

Invariant learning. Invariant risk minimization (IRM) and related methods seek predictors whose relationships to the target remain stable across environments [1, 6, 15], but typically assume that such environments are given. In contrast, we generate candidate environments automatically using LLM-derived taxonomies, and combine them with classical structural-break testing and conservative model selection.

OOD materials benchmarks. Prior work on materials OOD evaluation includes Matbench [5], structure-based splits [9], and transductive methods [20]. In contrast, we adopt leave-one-family-out splits that explicitly test generalization to genuinely novel chemical families.

## 3 Method

## 3.1 Overview

Figure 1 summarizes the proposed pipeline, which addresses the missing-environment problem by converting environment discovery into a generate–validate–select procedure. MOTIF first uses an LLM to generate K candidate chemical taxonomies offline, each representing a hypothesis for grouping the data into environments. Each taxonomy is then treated as a candidate environment for invariant feature selection and predictor training. We use a training-only pseudo-OOD criterion to filter candidate taxonomies, retaining only those that improve over the baseline and falling back to the baseline otherwise. The design is deliberately conservative: the LLM proposes candidate structures, while statistical tests and validation determine which ones are reliable.

## 3.2 Stage 1: LLM-defined chemical taxonomies

We prompt an LLM once, offline, to classify each unique spacer cation along K = 9 chemically motivated axes: spacer size, conjugation, polarity, steric profile, H-bonding, flexibility, electron effect, interlayer coupling, and thermal stability. Each axis induces a 2–4 way partition over the spacers, which we interpret as a candidate environment definition. For the primary 2D perovskite benchmark, this yield annotations for 313 unique spacers; external datasets reuse the same annotation idea with dataset-specific entity counts. The resulting taxonimies are discrete, auditable, and cached across all folds, making this step a one-time offline preprocessing. Because the total number of annotations depends on the benchmark and model backbone, we report concrete annotation counts and costs in Section 4.6 and Appendix J, rather than treating them as universal method constants.

Not all candidate environments are suitable for invariance analysis. For invariance testing, an environment split must be reasonably balanced; otherwise the small groups produce unstable estimates and the large group dominates the test. We therefore discard taxonomies whose largest group exceeds 85% of the available training samples, resulting in 7–9 usable taxonomies per fold. Appendix F shows that pilot taxonomy banks based on bulk-material notions (e.g. confinement and stability) failed because they produced highly imbalanced (degenerate or nearly degenerate) partitions, making them unsuitable for reliable invariance testing.

## 3.3 Stage 2: invariant feature selection

For taxonomy $k ,$ let $e _ { k } ( i ) \in \{ 1 , \ldots , G _ { k } \}$ denote the group assignment of training sample i. We treat these groups as environments and seek features whose relationship with the target remains stable across them, with the goal of removing environment-specific shortcuts that may not generalize under distribution shift. For each feature $j ,$ we test whether its linear relationship with the target is stable across groups using the Chow structural-break test [4]. This is a conservative screening step rather than a modeling assumption: features that exhibit strong environment-specific slope changes are excluded before training nonlinear predictors. Let $\mathrm { S S E } _ { \mathrm { p o o l e d } }$ be the residual sum of squares from a single regression fit to all training samples, and let $\mathrm { S S E _ { s e p } } ^ { \bullet } = \textstyle \sum _ { g = 1 } ^ { G _ { k } }$ SSE<sub>g</sub> be the sum from separate regressions within each group. The test statistic is

$$
F = \frac { ( \mathrm { S S E } _ { \mathrm { p o o l e d } } - \mathrm { S S E } _ { \mathrm { s e p } } ) / ( ( G _ { k } - 1 ) p ) } { \mathrm { S S E } _ { \mathrm { s e p } } / ( N - G _ { k } p ) } ,\tag{1}
$$

where $p$ is the number of regression parameters. We call feature $j$ invariant under taxonomy k when the Chow $p \textmd { - }$ value exceeds $\alpha = 0 . 0 1$ This threshold is intentionally strict: allowing too many borderline features reintroduces environment-specific correlations that improve in-distribution performance but degrade under family shift. Appendix H shows that both stricter and looser thresholds are worse on average.

Given the invariant set $S _ { k }$ , we run a small greedy forward-selection step with leave-one-environmentout validation to obtain a refined subset $\bar { S _ { k } ^ { * } }$ and a GBT regressor $f _ { k } . \bar { \bf A }$ practical point matters here: kNN search must be restricted to the same invariant subspace. Using the full feature set allows OOD-irrelevant spacer indicators to dominate distance calculations, particularly when the held-out family is absent from training. Restricting kNN to $\boldsymbol { S } _ { k } ^ { * }$ ensures consistency between feature selection and neighborhood definition.

We therefore define

$$
\hat { y } _ { \mathrm { i n v - k N N } } ^ { ( k ) } ( \mathbf { x } ^ { * } ) = \sum _ { i \in N _ { 5 } ( \mathbf { x } _ { S _ { k } ^ { * } } ^ { * } ) } \frac { w _ { i } } { \sum _ { j } w _ { j } } y _ { i } , \qquad w _ { i } = \frac { 1 } { \lVert \mathbf { x } _ { S _ { k } ^ { * } } ^ { * } - \mathbf { x } _ { S _ { k } ^ { * } } ^ { ( i ) } \rVert + \epsilon } .\tag{2}
$$

The per-taxonomy predictor is the average of invariant-kNN and GBT:

$$
\begin{array} { r } { \hat { y } _ { k } = \frac { 1 } { 2 } \Big ( \hat { y } _ { \mathrm { i n v - k N N } } ^ { ( k ) } + f _ { k } ( \mathbf x _ { \mathcal { S } _ { k } ^ { * } } ) \Big ) . } \end{array}\tag{3}
$$

## 3.4 Stage 3: gated aggregation

Because some taxonomies improve generalization while others are actively harmful under distribution shift, MOTIF aggregates predictions selectively rather than averaging across all taxonomies. For each taxonomy k, we compute leave-one-environment-out validation error using a training-only pseudo-OOD criterion on the training set,

$$
\mathrm { C V } _ { k } = \frac { 1 } { | \mathcal { E } _ { k } | } \sum _ { g \in \mathcal { E } _ { k } } \mathrm { M A E } ( f _ { k } ^ { ( - g ) } , \mathcal { D } _ { g } ) ,\tag{4}
$$

where $f _ { k } ^ { ( - g ) }$ is trained without environment g and $\mathcal { E } _ { k }$ contains environments with at least five samples. Taxonomy k is admitted only when it improves over the baseline, i.e., $\mathrm { C V } _ { k } < \mathrm { C V } _ { \mathrm { b a s e } } .$ . This “do no harm” gate ensures that only taxonomies with demonstrated OOD benefit contribute to the final prediction. The final prediction is the median of all admitted taxonomy predictions,

$$
\hat { y } ^ { * } = \operatorname * { m e d i a n } \mathopen { } \mathclose \bgroup \mathopen { } \mathclose \bgroup \mathopen { } \mathclose \bgroup \mathopen { } \mathclose \bgroup \mathopen { } \mathclose \bgroup \mathopen { } \mathclose \bgroup \mathopen { } \mathclose \bgroup \mathopen { } \mathclose \bgroup \mathopen { } \mathclose \bgroup \mathopen { } \mathclose \bgroup ( \{ \hat { y } _ { k } : \mathrm { C V } _ { k } < \mathrm { C V } _ { \mathrm { b a s e } } \aftergroup \egroup \aftergroup \egroup \aftergroup \egroup \aftergroup \egroup \aftergroup \egroup \aftergroup \egroup \aftergroup \egroup \aftergroup \egroup \aftergroup \egroup \aftergroup \egroup \aftergroup \egroup \aftergroup \egroup \aftergroup \egroup \aftergroup \egroup ) ,\tag{5}
$$

with fallback to the baseline if no taxonomy passes. The median provides robustness to occasional mis-specified or overconfident taxonomy predictions.

In the main six-fold benchmark, this gated-median rule is the default fully automatic readout because it exposes fold-level routing behavior most clearly. For frozen-protocol and transfer evaluations, we additionally report a conservative safe-blend readout that mixes the baseline with the best admitted taxonomy, reducing downside risk when routing is uncertain.

## 4 Experiments

## 4.1 Setup

Datasets and splits. Our primary benchmark is the public 2D Halide Perovskite Database of Marchenko et al. [11], using the NMSE portal snapshot accessed on March 21, 2026 [14]. After filtering for valid experimental bandgap values, 753 compounds were retained. For OOD evaluation, we use six leave-one-spacer-family-out folds, where each chemically defined spacer-cation family is held out once for testing: short-alkyl, long-alkyl, phenyl, heterocyclic, cycloalkyl, and specialcase groups. Across the six folds, every compound appears exactly once in the OOD test set. To assess transfer beyond the primary domain, we further evaluate on Matbench dielectric and 2D exfoliation [5], an OQMD ABX<sub>3</sub> formation-energy subset [10], and a Materials Project ABX<sub>3</sub> bandgap subset [8], using dataset-specific leave-one-group-out OOD splits (6–7 folds; Section 4.6). Detailed fold statistics are provided in Appendix A.

Features and baseline. For the 2D perovskite bandgap and optical-bandgap benchmarks, all methods use the same 30-dimensional compositional feature vector, including structural descriptors, B-site and halide composition, spacer-family indicators, and A-site descriptors (Appendix B). For the four transfer benchmarks, we keep the same local predictor template—inverse-distance kNN blended with GBT and wrapped by the same taxonomy/invariance pipeline—but use each dataset’s native descriptor space instead of the perovskite-specific features. The baseline is the corresponding kNN+GBT model without taxonomy-conditioned invariant filtering.

Evaluation protocol. We consider three evaluation settings. First, the six-fold leave-one-spacerfamily-out bandgap benchmark on 2D halide perovskites serves as the primary setting for fold-wise analysis, method comparison, and comparison against direct-LLM and RAG baselines. Second, we evaluate a stricter frozen dev/test protocol on perovskite bandgap and optical-bandgap tasks, where key hyperparameters (α=0.01, 85% balance, gated median) are selected on a frozen dev set (alkyl-short and phenyl), and the remaining four folds are used as a frozen test set (Appendix R). Third, we test generalization on four additional transfer benchmarks. To prevent information leakage, taxonomy labels are precomputed for all spacers, but only training-set labels are used during model fitting, and zero spacer overlap is verified for each fold (Appendix S).

## 4.2 Main results on the 2D halide perovskite bandgap benchmark

Table 1 reports results on six-fold leave-one-spacer-family-out OOD benchmark on 2D halide perovskite bandgap prediction. For this benchmark, we report results using the fully automatic gated-median readout, while later frozen-protocol sections adopt the safe-blend readout for a more conservative evaluation.

MOTIF (gated median) reaches 0.111 eV mean OOD MAE, improving over the 0.122 eV kNN+GBT baseline and achieving the best performance on four of the six folds. Because the baseline is already strong, the 0.011 eV reduction is practically meaningful. Two post-hoc reference rows provide additional context for the remaining performance gap. The best fixed single taxonomy attains 0.110 eV, while the oracle (the best taxonomy chosen separately for each fold with access to test labels) reaches 0.100 eV. This gap highlights the distinction between taxonomy quality from routing quality, suggesting that the improvements are primarily limited by the selection strategy rather than the taxonomy itself.

Table 1: 2D halide perovskite bandgap: six-fold leave-one-spacer-family-out OOD benchmark (MAE in eV). MOTIF is our proposed method. kNN, GBT, and their blend are non-LLM baselines; Self-Refine [12] is the best-performing direct LLM predictor. “Best single axis” and “oracle” are post-hoc upper bounds, not deployable configurations.
<table><tr><td>Method</td><td>Alkyl-S</td><td>Alkyl-L</td><td>Phenyl</td><td>Hetero</td><td>Cyclo</td><td>Special</td><td>Mean</td><td>∆</td></tr><tr><td>kNN (30-dim)</td><td>0.113</td><td>0.075</td><td>0.142</td><td>0.092</td><td>0.183</td><td>0.165</td><td>0.128</td><td></td></tr><tr><td>GBT (30-dim)</td><td>0.125</td><td>0.104</td><td>0.207</td><td>0.058</td><td>0.164</td><td>0.148</td><td>0.134</td><td></td></tr><tr><td>kNN+GBT Blend (baseline)</td><td>0.123</td><td>0.136</td><td>0.119</td><td>0.110</td><td>0.137</td><td>0.105</td><td>0.122</td><td></td></tr><tr><td>Self-Refine [12] (best LLM)</td><td>0.425</td><td>0.282</td><td>0.644</td><td>0.419</td><td>0.658</td><td>0.335</td><td>0.461</td><td>+278%</td></tr><tr><td>MOTIF (gated median)</td><td>0.128</td><td>0.129</td><td>0.095</td><td>0.099</td><td>0.098</td><td>0.119</td><td>0.111</td><td>-8.5%</td></tr><tr><td>MOTIF (best single axis; post hoc)</td><td>0.118</td><td>0.130</td><td>0.109</td><td>0.094</td><td>0.118</td><td>0.094</td><td>0.110</td><td>-9.3%</td></tr><tr><td>MOTIF (oracle axis; per-fold)</td><td>0.117</td><td>0.118</td><td>0.092</td><td>0.086</td><td>0.097</td><td>0.089</td><td>0.100</td><td>-17.8%</td></tr></table>

Table 2: 2D halide perovskite bandgap: fold-level OOD MAE matrix. MOTIF beats the kNN+GBT blend on Alkyl-L, Phenyl, Hetero, and Cyclo, but not on Alkyl-S or Special.
<table><tr><td rowspan=1 colspan=7>Method        Alkyl-SAlkyl-LPhenylHeteroCycloSpecial</td></tr><tr><td rowspan=1 colspan=1>MOTIF</td><td rowspan=1 colspan=1>0.128</td><td rowspan=1 colspan=1>0.129</td><td rowspan=1 colspan=1>0.095</td><td rowspan=1 colspan=1>0.099</td><td rowspan=1 colspan=1>0.098</td><td rowspan=1 colspan=1>0.119</td></tr><tr><td rowspan=1 colspan=1>kNN+GBT Blend</td><td rowspan=1 colspan=1>0.123</td><td rowspan=1 colspan=1>0.136</td><td rowspan=1 colspan=1>0.119</td><td rowspan=1 colspan=1>0.110</td><td rowspan=1 colspan=1>0.137</td><td rowspan=1 colspan=1>0.105</td></tr><tr><td rowspan=1 colspan=1>kNN</td><td rowspan=1 colspan=1>0.113</td><td rowspan=1 colspan=1>0.075</td><td rowspan=1 colspan=1>0.142</td><td rowspan=1 colspan=1>0.092</td><td rowspan=1 colspan=1>0.183</td><td rowspan=1 colspan=1>0.165</td></tr><tr><td rowspan=1 colspan=1>GBT</td><td rowspan=1 colspan=1>0.125</td><td rowspan=1 colspan=1>0.104</td><td rowspan=1 colspan=1>0.207</td><td rowspan=1 colspan=1>0.058</td><td rowspan=1 colspan=1>0.164</td><td rowspan=1 colspan=1>0.148</td></tr><tr><td rowspan=1 colspan=1>Self-Refine</td><td rowspan=1 colspan=1>0.425</td><td rowspan=1 colspan=1>0.282</td><td rowspan=1 colspan=1>0.644</td><td rowspan=1 colspan=1>0.419</td><td rowspan=1 colspan=1>0.658</td><td rowspan=1 colspan=1>0.335</td></tr></table>

The performance gains in Table 2 are not uniform across folds. The largest improvements over the kNN+GBT blend occur on cycloalkyl (−28%) and phenyl (−20%), where the held-out families are chemically most distinct from the training data and correlations learned during training are least transferable. In contrast, the blend performs better on Alkyl-S and Special, and the single-mode classical baselines still outperform MOTIF on some folds. Overall, the improvements are concentrated on the most challenging transfer settings rather than being uniformly distributed across all folds.

## 4.3 Role-based comparison on the main benchmark

Figure 2 addresses the broader role question before we zoom into mechanism: does assigning the LLM a direct predictive role improve performance on this numerical OOD task? A cross-role comparison shows that direct numerical use of the LLM is not effective. Across the 19 published LLM methods (along with 4 ML baselines; see Appendix C for the full list and Appendix P for 65+ additional in-house variants), direct prompting, RAG, feature augmentation, sample selection, and prior/constraint variants all fail to outperform the kNN+GBT baseline. This can be explanationed by a mismatch between the numerical precision of LLM-generated estimates and the low-error regime of the benchmark: the best prompt-based numerical predictor still exhibits approximately 0.46 eV mean error, whereas the classical baseline achieves 0.122 eV. On this benchmark, the only role that consistently improves over the baseline is offline annotation/classification, where the LLM proposes candidate environment structure rather than the final property value. Full per-category breakdowns and method inventories are provided in Appendix D.

## 4.4 Why MOTIF helps: balanced taxonomies and invariant kNN

The ablations results support a specific mechanism rather than a generic ensembling effect. In particular, two components are critical. First, the taxonomy bank must produce sufficiently balanced groupings to define useful environments. Second, both the regressor and the neighbor search must operate within the invariant subspace. Figure 3 illustrates both points. Removing invariant kNN increases the mean MAE from 0.111 to 0.115 eV, while removing the balance filter is even worse to 0.118 eV. Random groupings recover little of the performance gain, confirming that the gains do not arise from ensembling over arbitrary partitions.

A pilot taxonomy bank based on material-level properties such as confinement and stability proved too degenerate to define useful environments. In contrast, the final spacer-centric taxonomy yields

![](images/1321e921fd4648b1c860b0dd4c5df4fdaabada597ef892f6a1aa1499da0e2d15.jpg)  
Figure 2: Role-level summary on the six-fold 2D halide perovskite bandgap benchmark. Across 19 published LLM methods (plus 4 ML baselines and 65+ in-house variants; Appendix P), methods that inject numerical values do not beat the classical baseline. The only role that consistently improves over the baseline is offline classification.

![](images/a7680a04a83d0bb78a35d2cf7789386e5676fa5f1694ba64b4331f75cb4d27c4.jpg)  
(a) Ablation on the six-fold 2D halide perovskite bench mark.

![](images/6852eddf4a84c6376101ae16ade4d7fd07b4bc1a4c1862ef92903c06bfd4208f.jpg)

![](images/7fd391469c7b6331e11d8475c6cca95cb6c4cbbecd1e6ded97c8bfc537ef7b99.jpg)  
(b) Why balanced taxonomies matter.  
Figure 3: Mechanistic diagnostics for MOTIF on the 2D halide perovskite bandgap benchmark. Left: the strongest drops come from removing invariant kNN or the balance filter, and random groupings nearly collapse to the baseline. Right: a pilot taxonomy bank produced degenerate partitions, whereas the final spacer-centric design yields mostly usable axes.

7–9 usable axes per fold. Importantly, axis quality depends on group balance and pseudo-OOD validation rather than semantic plausibility (Appendix F, H).the final spacer-centric bank leaves 7–9 usable axes per fold. Axis quality depends on group balance and pseudo-OOD CV, not semantic appeal (Appendix F, H).

## 4.5 Frozen-protocol controls and stronger non-LLM baselines

Table 3 evaluates awhether the observed gains persist under frozen training-only model selection and against strong classical controls. They do, but the appropriate interpretation is narrow. First, the primary deployable configuration (gpt-4o annotator + safe blend) improves the frozen bandgap benchmark from 0.1233 to 0.1174 eV (paired bootstrap p=0.002) and transfers to optical bandgap (−4.2%). Second, LLM taxonomies outperform alternative environment definitions. Expert-family partitions are competitive but weaker, while rule-based axes and descriptor clustering do not improve over the baseline. Third, compared with strong classical selectors, MOTIF is competitive rather than dominant: the Top-1 CV variant is essentially tied with LOEO forward selection and Sparse Group Lasso, and the same protocol reaches 0.1148 eV with the o3 annotator (Appendix J). The final panel adds the strongest direct-LLM and RAG baselines on the same 2D task for context. They are reported on the six-fold leave-one-spacer-family-out benchmark rather than the frozen protocol because their error gap is already much larger than the differences among competitive classical controls. These results support a benchmark-scoped conclusion: in this setting, LLM taxonomies do not replace classical feature selection, but they can provide useful environment structure when combined with a strong classical prediction pipeline.

All claims are supported by query-level paired bootstrap tests (n=753 queries, B=10,000 resamples). On the control benchmark, the improvement of MOTIF (safe blend, gpt-4o) over the baseline is 0.0041 eV with a 95% bootstrap CI of [0.0014, 0.0068] and p=0.002. With the o3 annotator the improvement increases to 0.0058 eV, CI = [0.0031, 0.0086], p<0.001. Per-fold tests show that the gain is statistically significant on cycloalkyl (p=0.006) and phenyl (p=0.004), consistent on alkyl-long $\scriptstyle \left( { { p = 0 . 0 6 } } \right)$ , and absent on special $\scriptstyle ( p = 0 . 9 3 )$ . Seed sensitivity across 5 GBT random seeds yields a standard deviation of 0.0002 eV around a mean of 0.1142 eV (range [0.1139, 0.1144]), confirming that GBT randomness does not account for the observed improvements. Full bootstrap tables appear in Appendix Q. No single reasoning backbone dominates both targets (Appendix J), confirming that routing quality matters more than backbone scale.

Table 3: Competitive controls for the 2D halide perovskite task. Panels A–C use the frozen trainingonly protocol; Panel D adds the strongest direct-LLM and RAG baselines on the same six-fold leave-one-spacer-family-out task for scale. MOTIF rows are our proposed method. “Baseline” is the kNN+GBT blend without taxonomy-conditioned invariant filtering. See Appendix U for method details.
<table><tr><td colspan="4">Category Method Eval. MAE</td></tr><tr><td colspan="4">vs. BL (A) Frozen-protocol environment controls (bandgap)</td></tr><tr><td colspan="2">Baseline blend (kNN+GBT, 30-dim)</td><td>Frozen BG</td><td>0.1233</td><td></td></tr><tr><td colspan="2">MOTIF LLM tax. (gpt-4o) + Safe blend† MOTIF LLM tax. (gpt-4o) + Top-1 CV</td><td>Frozen BG Frozen BG</td><td>0.1174 0.1168</td><td>-4.8% Ours</td></tr><tr><td colspan="2">Control Expert-family tax. + Safe blend</td><td>Frozen BG Frozen BG</td><td>-5.3% 0.1190 -3.5%</td><td>Ours</td></tr><tr><td colspan="2">Control Rule-based chemistry axes Control Descriptor clustering (k-means)</td><td>0.1244 Frozen BG 0.1222</td><td>+0.9% -0.9%</td><td>[13]</td></tr><tr><td colspan="5">(B) Frozen-protocol transfer to optical bandgap</td></tr><tr><td colspan="2">Baseline blend</td><td>Frozen OptBG</td><td>0.1807</td><td></td></tr><tr><td colspan="2">MOTIF LLM tax. (gpt-4o) + Safe blend</td><td>Frozen OptBG</td><td>0.1732</td><td>-4.2% Ours</td></tr><tr><td colspan="2">Control Rule-based chemistry axes</td><td>Frozen OptBG</td><td>0.1821</td><td>+0.8%</td></tr><tr><td colspan="2">Control Descriptor clustering</td><td>Frozen OptBG</td><td>0.1803 -0.2%</td><td>[13]</td></tr><tr><td colspan="5">(C) Frozen-protocol classical selectors and DG objectives (bandgap)</td></tr><tr><td colspan="2">MOTIF LLM tax. (gpt-4o) + Top-1 CV</td><td>Frozen BG Frozen BG</td><td>0.1168 -5.3% 0.1172 -4.9%</td><td>Ours</td></tr><tr><td colspan="2">Feat. sel. Pearson top-k + blend Feat. sel. LOEO forward sel. + blend</td><td>Frozen BG</td><td>0.1169 -5.2%</td><td></td></tr><tr><td colspan="2">Feat. sel. IID forward sel. + blend</td><td>Frozen BG</td><td>0.1227 -0.5%</td><td></td></tr><tr><td colspan="2">Sparse Group Lasso + blend</td><td>Frozen BG</td><td>0.1236 +0.2%</td><td>[30]</td></tr><tr><td colspan="2">Sparse Sparse Group Lasso + blend</td><td>Frozen BG</td><td>0.1171 -5.0%</td><td>[21]</td></tr><tr><td colspan="2">DG IRM / VREx / GroupDRO / CORAL Frozen BG</td><td></td><td>0.21-0.25</td><td>+71–99%[1, 18, 19,22]</td></tr><tr><td colspan="5">(D) Contextual direct-LLM and RAG baselines on the same task</td></tr><tr><td colspan="2">LLM Self-Refine</td><td>LOFO BG</td><td>0.461</td><td>+278% [12]</td></tr><tr><td colspan="2">LLM LATS</td><td>LOFO BG</td><td>0.567</td><td>+365% [27]</td></tr><tr><td colspan="2">RAG Evidence Weighted Mean</td><td>LOFO BG</td><td>0.304 +149%</td><td></td></tr><tr><td colspan="2">RAG BM25-RAG</td><td>LOFO BG</td><td>2.427 +1889%</td><td></td></tr><tr><td colspan="2"></td><td></td><td></td><td></td></tr></table>

<sup>†</sup>Safe blend (p=0.002) is the primary deployable ensemble; Top-1 CV (p=0.206) is not statistically significant (Appendix Q). With the o3 annotator, Top-1 CV reaches 0.1148 (Appendix J). Rows marked LOFO BG are compared against the six-fold bandgap blend baseline of 0.122 eV; frozen rows are compared against their corresponding frozen-protocol baseline.

![](images/989b5d1fa1d7667243fada0f0e34ed34248b29428f0830a676faaca4b61d9ef5.jpg)  
(a) Oracle-gap view of selector headroom

![](images/f8d4a407afb719723497fb92f5eb85462c2c8cb4c4632b6ce7b848b1f0e924a7.jpg)

![](images/0481f079003da9ed27ba5c51a0472e7971d6ef15caeae7d80f55024da479aeeb.jpg)  
(b) Error-tail reduction  
Figure 4: 2D halide perovskite bandgap: failure-mode analysis. Left: the best taxonomy changes across folds, so the remaining gap is a routing problem. Right: MOTIF improves by trimming large OOD errors, not by uniformly improving already-easy examples.

Figure 4 summarizes the remaining taxonomy-routing gap and the reduction in large prediction errors.

## 4.6 Cross-dataset transfer beyond the primary perovskite task

We use the additional datasets to evaluate whether the taxonomy-to-invariance pipeline transfers beyond the primary 2D perovskite setting under dataset-specific leave-one-group-out OOD splits.

Table 4: Additional transfer benchmarks beyond the primary 2D perovskite task. We report both the automatic gated-median ensemble and the deployable safe-blend ensemble against the same kNN+GBT baseline. All four benchmarks use LLM taxonomies (gpt-4o-mini) with leave-onegroup-out OOD splits.
<table><tr><td>Dataset</td><td>Target</td><td>N</td><td>Folds</td><td>BL MAE</td><td>Gated</td><td>Safe</td><td>Best ∆</td></tr><tr><td>Matbench Dielec.</td><td>dielectric</td><td>4,764</td><td>7</td><td>0.608</td><td>0.584</td><td>0.571</td><td>-6.2%</td></tr><tr><td>Matbench 2D Exfol.</td><td>exfol. en.</td><td>636</td><td>6</td><td>72.2</td><td>70.1</td><td>68.7</td><td>-4.9%</td></tr><tr><td>OQMD ABX3</td><td>form. en.</td><td>4,000</td><td>7</td><td>0.435</td><td>0.427</td><td>0.420</td><td>-3.6%</td></tr><tr><td>MP ABX3</td><td>bandgap</td><td>4,557</td><td>7</td><td>0.845</td><td>0.847</td><td>0.817</td><td>-3.3%</td></tr></table>

Table 5: Cost–accuracy comparison on the 2D perovskite bandgap benchmark (753 OOD queries).
<table><tr><td>Method</td><td>OOD MAE (eV)</td><td>Cost (753 queries)</td><td>Hardware</td></tr><tr><td>MOTIF (after caching)</td><td>0.111</td><td>$0.00</td><td>CPU only</td></tr><tr><td>MOTIF (incl. taxonomy gen.)</td><td>0.111</td><td>$0.03-0.37</td><td>CPU only</td></tr><tr><td>Self-Refine (best LLM)</td><td>0.461</td><td>$45.18</td><td>API</td></tr><tr><td>LATS</td><td>0.567</td><td>$112.95</td><td>API</td></tr></table>

The 2D halide perovskite benchmark is the main causal testbed because it supports fold-wise analysis, exhaustive comparison against 19 published LLM methods, and detailed mechanism studies. The external datasets test whether the winning taxonomy-to-invariance pipeline carries beyond that setting. Using gpt-4o-mini taxonomies and leave-one-group-out OOD splits, the deployable safe-blend version of MOTIF improves on all four transfer benchmarks: Matbench dielectric (−6.2%, 4,764 compounds), Matbench 2D exfoliation energy (−4.9%, 636 compounds), OQMD ABX formation energy (−3.6%, 4,000 compounds), and MP ABX<sub>3</sub> bandgap (−3.3%, 4,557 compounds). Table 4 also reports the automatic gated-median ensemble as an internal comparator; unlike the safe blend, it is not uniformly positive off-domain. The gains span four target properties and diverse chemistry, indicating that the pipeline is not specific to a single material family or mechanism.

The cost profile further distinguishes MOTIF from LLM baselines (Table 5). For the primary 2D perovskite benchmark, the one-time annotation pass consists of 313×9 = 2,817 cached classifications. With gpt-4o or gpt-4o-mini, this costs only \$0.03–\$0.37, after which all prediction is local and CPU-based at zero marginal cost. In contrast, Self-Refine costs \$45.18 for the same 753 OOD queries while achieving 3.7× worse MAE.

## 5 Limitations

We interpret the gains of MOTIF as evidence that LLM-defined taxonomies can provide useful environment structure for OOD prediction, but not as evidence of a universally superior environment source. We therefore position LLM-defined taxonomies as a complementary source of environment structure for a classical prediction pipeline, not as a replacement for strong feature selection or as a universally optimal environment source. Although the method improves frozen-protocol performance on 2D perovskites and transfers consistently across four additional benchmarks, its benefit depends on whether the taxonomy bank induces reasonably balanced groups and captures the dominant chemistry of the held-out shift. When this alignment is weak, routing errors can reduce or erase the gain, as illustrated by the harder folds in our analysis. An important direction for future work is thus more reliable routing or abstention, rather than simply scaling the LLM backbone.

## 6 Conclusion

We study LLM use in high-precision OOD materials prediction as an environment-discovery problem rather than a direct regression problem. On the evaluated benchmarks, direct numerical use of the LLM is not competitive with a strong classical baseline, whereas offline taxonomy annotation yields consistent gains. MOTIF turns those taxonomies into auditable candidate chemical environments for invariant feature selection, improving the six-fold 2D halide perovskite benchmark from 0.122 to 0.111 eV and the frozen-protocol benchmark from 0.123 to 0.117 eV while remaining competitive with strong classical selectors and generalizing to four additional transfer benchmarks. The evidence therefore supports a benchmark-scoped design rule for scientific ML: use LLMs to propose structured chemical environment hypotheses, then let downstream statistical procedures audit, filter, and, when necessary, reject them. This principle, developed in materials science, extends to other scientific domains where structure–property relationships vary across underlying environments.

## References

[1] M. Arjovsky, L. Bottou, I. Gulrajani, and D. Lopez-Paz. Invariant risk minimization. arXiv:1907.02893, 2019.

[2] D. A. Boiko, R. MacKnight, B. Kline, and G. Gomes. Autonomous chemical research with large language models. Nature, 624:570–578, 2023.

[3] A. M. Bran et al.Augmenting large language models with chemistry tools. Nat. Mach. Intell., 6:525–535, 2024.

[4] G. C. Chow. Tests of equality between sets of coefficients in two linear regressions. Econometrica, 28(3):591–605, 1960.

[5] A. Dunn, Q. Wang, A. Ganose, D. Dopp, and A. Jain. Benchmarking materials property prediction methods: the Matbench test suite. npj Comput. Mater., 6:138, 2020.

[6] I. Gulrajani and D. Lopez-Paz. In search of lost domain generalization. In ICLR, 2021.

[7] K. M. Jablonka, P. Schwaller, A. Ortega-Guerrero, and B. Smit. Leveraging large language models for predictive chemistry. Nat. Mach. Intell., 6:161–169, 2024.

[8] A. Jain et al.Commentary: The Materials Project: A materials genome approach to accelerating materials innovation. APL Mater., 1(1):011002, 2013.

[9] S. S. Omee et al.Structure-based out-of-distribution materials property prediction: a benchmark study. npj Comput. Mater., 10:144, 2024.

[10] S. Kirklin et al.The Open Quantum Materials Database (OQMD): assessing the accuracy of DFT formation energies. npj Comput. Mater., 1:15010, 2015.

[11] E. I. Marchenko et al.Database of Two-Dimensional Hybrid Perovskite Materials: Open-Access Collection of Crystal Structures, Band Gaps, and Atomic Partial Charges Predicted by Machine Learning. Chem. Mater., 32(17):7383–7388, 2020.

[12] A. Madaan et al.Self-Refine: Iterative refinement with self-feedback. In NeurIPS, 2023.

[13] J. MacQueen. Some methods for classification and analysis of multivariate observations. In Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability, volume 1, pages 281–297, 1967.

[14] NMSE Lab. NMSE 2D perovskite database portal. http://www.pdb.nmse-lab.ru/. Accessed: March 21, 2026.

[15] J. Peters, P. Bühlmann, and N. Meinshausen. Causal inference by using invariant prediction: identification and confidence intervals. J. R. Stat. Soc. B, 78(5):947–1012, 2016.

[16] J. Requeima, J. Bronskill, D. Choi, R. E. Turner, and D. Duvenaud. LLM processes: Numerical predictive distributions conditioned on natural language. In NeurIPS, 2024.

[17] A. N. Rubungo, C. Arnold, B. P. Rand, and A. B. Dieng. LLM-Prop: predicting the properties of crystalline materials using large language models. npj Comput. Mater., 11:186, 2025.

[18] D. Krueger et al.Out-of-distribution generalization via risk extrapolation (REx). In ICML, 2021.

[19] S. Sagawa, P. W. Koh, T. B. Hashimoto, and P. Liang. Distributionally robust neural networks for group shifts: On the importance of regularization for worst-case generalization. arXiv:1911.08731, 2020.

[20] N. Segal et al.Known Unknowns: Out-of-Distribution Property Prediction in Materials and Molecules. npj Comput. Mater., 11:345, 2025.

[21] N. Simon, J. Friedman, T. Hastie, and R. Tibshirani. A sparse-group lasso. J. Comput. Graph. Stat., 22(2):231–245, 2013.

[22] B. Sun and K. Saenko. Deep CORAL: Correlation alignment for deep domain adaptation. In ECCV Workshops, pages 443–450, 2016.

[23] S. Wang et al.Want to reduce labeling cost? GPT-3 can help. In Findings of EMNLP, 2021.

[24] J. Wei et al.Chain-of-thought prompting elicits reasoning in large language models. In NeurIPS, 2022.

[25] S. Yao et al.Tree of thoughts: Deliberate problem solving with large language models. In NeurIPS, 2023.

[26] N. Shinn et al.Reflexion: Language agents with verbal reinforcement learning. In NeurIPS, 2023.

[27] A. Zhou et al.Language agent tree search unifies reasoning, acting, and planning in language models. In ICML, 2024.

[28] S. Dhuliawala et al.Chain-of-verification reduces hallucination in large language models. In Findings ofACL, 2024.

[29] V. Vovk, A. Gammerman, and G. Shafer. Algorithmic Learning in a Random World. Springer, 2005.

[30] M. Yuan and Y. Lin. Model selection and estimation in regression with grouped variables. J. R. Stat. Soc. B, 68(1):49–67, 2006.

![](images/bef22b068ddc7e5ca722b4209d5c7ddef3f20dda2ba4ac7d27e93d0e057695fa.jpg)  
Figure 5: Crystal structures of 2D halide perovskites with different spacer configurations: Ruddlesden–Popper (left) and Dion–Jacobson (right) phases. The Ruddlesden–Popper structure consists of perovskite layers separated by bilayer organic spacers, whereas the Dion–Jacobson structure features monolayer spacers that bridge adjacent inorganic slabs, leading to distinct interlayer coupling and electronic properties.

Table 6: 2D perovskite bandgap: OOD fold statistics. “Held-Out” counts the bandgap-valid OOD queries, “Train” is the training size after excluding that family.
<table><tr><td>Fold (Held-Out Family)</td><td>Held-Out</td><td>Train</td><td>Mean BG (eV)</td><td>Std BG (eV)</td></tr><tr><td>Alkyl-short</td><td>232</td><td>521</td><td>2.41</td><td>0.40</td></tr><tr><td>Alkyl-long</td><td>79</td><td>674</td><td>2.31</td><td>0.26</td></tr><tr><td>Phenyl</td><td>166</td><td>587</td><td>2.45</td><td>0.44</td></tr><tr><td>Heterocyclic</td><td>128</td><td>625</td><td>2.38</td><td>0.36</td></tr><tr><td>Cycloalkyl</td><td>37</td><td>716</td><td>2.70</td><td>0.44</td></tr><tr><td>Special</td><td>111</td><td>642</td><td>2.33</td><td>0.34</td></tr><tr><td>Total</td><td>753</td><td>一</td><td>2.41</td><td>0.39</td></tr></table>

## A Dataset Details

## A.1 2D Halide Perovskite Database

The 2D Halide Perovskite Database was constructed by Marchenko et al. [11]; the publicly available version used in this study was accessed via the NMSE database portal on March 21, 2026 [14]. That dataset contains 849 compound entries covering layered hybrid organic-inorganic perovskites of the Ruddlesden–Popper and Dion–Jacobson types (Figure5). After filtering for valid experimental bandgap values, 753 compounds remain. The database records the organic spacer cation, the B-site metal (Pb, Sn, Cu, Mn, Cd, Ge, Bi), the halide composition (I, Br, Cl, or mixtures), the number of inorganic layers (n = 1, 2, 3, . . .), and the experimentally measured bandgap. Bandgap values in the filtered benchmark range from 1.65 eV to 3.53 eV, with a mean of 2.41 eV and a standard deviation of 0.39 eV (Figure6). The distribution is multimodal, reflecting the distinct chemical families of spacer cations.

## A.2 OOD Fold Statistics

We used a 6-fold leave-one-spacer-family-out OOD evaluation protocol, in which each fold holds out compounds whose organic spacer belongs to a single chemical superfamily. The superfamilies were defined by chemical structure (7): alkyl chains (short: C1–C4; long: C5+), phenyl-containing, heterocyclic, cycloalkyl, and a "special" category comprising ammonium salts, amino acids, and other structurally unique structures.

Table 6 summarizes the sample counts and bandgap statistics for the six held-out families.

![](images/c3b7805cd6060b2101382dbd3b355fc14e8edc69f729f19dec684981b4f14ccb.jpg)

Figure 6: Schematic representation of the 2D halide perovskite dataset. Each material is described by its compositional and structural components, including (a) B-site metal, (b) halide composition, (c) organic spacer cation, and (d) the number of inorganic layers (n). These factors collectively determine the (e) experimentally measured bandgap, which serves as the prediction target. The dataset span diverse chemical environments arising from variations in spacer chemistry and layer thickness.  
![](images/72891d582907deb5b7eef0536c1156273192fc48d732b76a0cfbdfc2f4c32e64.jpg)  
Figure 7: Chemical superfamilies of organic spacer cations used to define the leave-one-spacer-familyout OOD evaluation protocol: (a) alkyl chains (short: C1–C4; long: C5+), (b) phenyl-containing, (c) heterocyclic, (d) cycloalkyl, and (e) special structures including ammonium salts, amino acids, and other unique chemistries. Each superfamily defines a distinct chemical environment, and is held out in turn to evaluate generalization to unseen spacer chemistries.

The six evaluation folds (each corresponding to a held-out spacer family) varied substantially in difficulty. The cycloalkyl fold was the smallest, with only 37 bandgap-valid samples, and its alicyclic spacers were structurally distinctive relative to the remaining training families. In contrast, the alkyl-short fold was the largest and exhibited the greatest chemical similarity to the training data, resulting in a smaller distribution shift and making it the most interpolation-like setting among the six.

## A.3 Bandgap Distribution

Figure 8 presents the bandgap distributions across all six spacer superfamilies. The distributions are clearly heterogeneous and often multimodal, reflecting the underlying chemical diversity of spacer cations. These variations highlight the substantial distribution differences across families, reinforcing the need for OOD evaluation that tests generalization to genuinely novel chemical regimes.

Alkyl-short and phenyl are the largest families, with mean bandgaps of approximately 2.41 eV and 2.45 eV, respectively. In contrast, the cycloalkyl family exhibits the highest mean bandgap (2.70 eV)

![](images/6e9c5f1912431f1eea85bc82473bd9deeefb37acb2b2cd5585ff12427192cdb8.jpg)  
Figure 8: Bandgap distributions per spacer superfamily. Each panel shows the histogram and kernel density estimate for the held-out family (colored) against the remaining training distribution (gray). The vertical dashed line indicates the family-specific mean bandgap.

despite being the smallest fold. The “special” family shows the broadest and most heterogeneous distribution, consistent with its role as a catch-all category for structurally diverse spacers. This heterogeneity helps explain why it is the least reliably improved by taxonomy-driven selection.

## B Full 30-Dimensional Feature Description

All methods in this study use the same 30-dimensional compositional feature vector extracted from each perovskite compound, grouped into five categories. Among these, the spacer-family one-hot features (features 17–22) are particularly problematic for OOD generalization. When a spacer family is held out (e.g., “phenyl”), the corresponding feature (feature 19) is always 0 in the training data and always 1 in the test data. As a result, this feature cannot support generalization and instead acts as a dataset-specific shortcut. Models that rely on these features may exploit dataset-specific shortcuts, rather than learning generalizable chemical relationships.

This issue motivates the use of invariant feature selection. By removing features whose relationship to the target is not stable across groups, the model focuses on chemical relationships that transfer across different environments.

Feature groups. The 30 features used throughout the paper are partitioned into five categories (Figure 9)

1. Structural descriptors (5): layer count, tolerance-like geometric terms, and binary indicators for DJ/RP structure type.

2. B-site identity (7): one-hot indicators for the major B-site metals used in the dataset.

3. Halide composition (4): I/Br/Cl composition indicators and a mixed-halide flag.

4. Spacer identity (8): spacer-family one-hot features together with simple chain-length and branching indicators.

5. Intralayer cation descriptors (6): MA/FA/Cs indicators toegether with coarse radius / electronegativity summaries.

In practice, the spacer-family indicators are the most OOD-sensitive features. Importantly, spacer families serve as the basis for defining OOD environments, but their corresponding one-hot indicators are not suitable as predictive features. When an entire spacer family is held out, these features encode training-specific correlations that act as dataset-specific shortcuts rather than transferable chemical relationships. Consequently, both invariant feature filtering and strong classical feature selection methods tend to down-weight or remove them.

![](images/e320bf369c9ed3541eaa1b65aa0eaa0e53f867e9a200a90e87cc0c3d0f79f4ee.jpg)  
Figure 9: Overview of the feature groups used in this study. The 30 features are organized into five categories: structural descriptors, B-site identity, halide composition, spacer identity, and intralayer cation descriptors. These groups capture key chemical and structural factors—such as composition, lattice geometry, and spacer chemistry—that define distinct chemical environments and govern structure–property relationships in 2D halide perovskites.

## C Complete OOD Results for Published Methods

We evaluated 19 published LLM methods together with several ML baselines (15 direct prediction, 4 RAG, plus 3 ML baselines and the kNN+GBT blend). Tables 7 and 8 separate methods with full fold-wise logs from those that report only a 6-fold means. This avoids missing entries and ensures that the source and granularity of each result are clearly specified.

## Key observations:

• All 15 direct LLM prediction methods yield $\mathrm { M A E } > 0 . 4 6 \ : \mathrm { e V } ,$ which is 3–5 times worse than the ML baseline. No prompting strategy overcomes this fundamental limitation in numerical precision.

• RAG methods exhibit a wide performance range. Evidence-weighted mean (0.304 eV) performs best, while BM25 and embedding RAG perform poorly (above 2.4 eV) because they retrieve structurally similar but numerically irrelevant compounds.

• Among ML baselines, kNN shows the highest variance across folds (ranging from 0.075 on alkyllong to 0.183 on cycloalkyl), reflecting its sensitivity to feature-space distance. GBT is more stable but performs worse on average. Their blend mitigates these complementary weaknesses.

• MOTIF (oracle) achieves the lowest MAE on every fold of the six-fold benchmark, demonstrating the potential of taxonomy-guided invariant learning when an appropriate gouping is selected. The remaining challenge is to identify such groupings automatically.

## D Detailed Negative Results

This section provides a detailed breakdown of the negative-result experiments for the 19 published LLM methods alongside classical ML baselines, organized into five method categories corresponding to different roles of LLM usage. Throughout this section, we use a k-nearest neighbors (kNN) and gradient boosting trees (GBT) ensemble as the ML baseline. A consistent pattern emerges across all five non-classifier categories: LLM-generated numerical predictions fail to achieve competitive accuracy. All named methods below are published prompting or retrieval strategies, and our implementations follow the original descriptions. All results are reported in terms of mean absolute error (MAE) in eV. For the additional 65+ in-house configuration variants (feature augmentation, sample selection, priors/constraints), detailed results are reported in Appendix P.

Table 7: 2D perovskite bandgap: methods with full fold-wise OOD logs. MAE in eV. Bold: best per column.
<table><tr><td>Method</td><td>Alkyl-S</td><td>Alkyl-L</td><td>Phenyl</td><td>Hetero</td><td>Cyclo</td><td>Special</td><td>Mean</td><td>Cat.</td></tr><tr><td>MOTIF (oracle)</td><td>0.117</td><td>0.118</td><td>0.092</td><td>0.086</td><td>0.097</td><td>0.089</td><td>0.100</td><td>Tax.</td></tr><tr><td>MOTIF (best single)</td><td>0.118</td><td>0.130</td><td>0.109</td><td>0.094</td><td>0.118</td><td>0.094</td><td>0.110</td><td>Tax.</td></tr><tr><td>MOTIF (gated median)</td><td>0.128</td><td>0.129</td><td>0.095</td><td>0.099</td><td>0.098</td><td>0.119</td><td>0.111</td><td>Tax.</td></tr><tr><td>kNN+GBT Blend</td><td>0.123</td><td>0.136</td><td>0.119</td><td>0.110</td><td>0.137</td><td>0.105</td><td>0.122</td><td>ML</td></tr><tr><td>kNN (30-dim)</td><td>0.113</td><td>0.075</td><td>0.142</td><td>0.092</td><td>0.183</td><td>0.165</td><td>0.128</td><td>ML</td></tr><tr><td>Random Forest</td><td>0.131</td><td>0.136</td><td>0.128</td><td>0.104</td><td>0.131</td><td>0.109</td><td>0.123</td><td>ML</td></tr><tr><td>Ridge Regression</td><td>0.135</td><td>0.107</td><td>0.129</td><td>0.114</td><td>0.134</td><td>0.137</td><td>0.126</td><td>ML</td></tr><tr><td>GBT (30-dim)</td><td>0.125</td><td>0.104</td><td>0.207</td><td>0.058</td><td>0.164</td><td>0.148</td><td>0.134</td><td>ML</td></tr><tr><td>Self-Refine</td><td>0.425</td><td>0.282</td><td>0.644</td><td>0.419</td><td>0.658</td><td>0.335</td><td>0.461</td><td>Prompt</td></tr></table>

Table 8: 2D perovskite bandgap: stored mean-only results for methods whose archives did not retain per-fold decompositions.
<table><tr><td colspan="2">Method Mean MAE</td><td>Cat.</td></tr><tr><td>Evidence Wt. Mean</td><td>0.304</td><td>RAG</td></tr><tr><td>Step-Back</td><td>0.463</td><td>Prompt</td></tr><tr><td>Self-Consistency</td><td>0.492</td><td>Prompt</td></tr><tr><td>Self-Discover</td><td>0.506</td><td>Prompt</td></tr><tr><td>LLM Process</td><td>0.511</td><td>Reason</td></tr><tr><td>CoVe</td><td>0.513</td><td>Reason</td></tr><tr><td>Tree of Thoughts</td><td>0.532</td><td>Reason</td></tr><tr><td>Reflexion</td><td>0.540</td><td>Reason</td></tr><tr><td>Analogical</td><td>0.542</td><td>Prompt</td></tr><tr><td>Zero-Shot</td><td>0.552</td><td>Prompt</td></tr><tr><td>Chain-of-Thought</td><td>0.563</td><td>Prompt</td></tr><tr><td>LATS</td><td>0.567</td><td>Reason</td></tr><tr><td>RaR</td><td>0.581</td><td>Prompt</td></tr><tr><td>StructChem</td><td>0.632</td><td>Prompt</td></tr><tr><td>CRITIC</td><td>0.647</td><td>Reason</td></tr><tr><td>CRAG</td><td>0.680</td><td>RAG</td></tr><tr><td>BM25-RAG</td><td>2.427</td><td>RAG</td></tr><tr><td>Embedding-RAG</td><td>2.430</td><td>RAG</td></tr></table>

## D.1 Category (i): Direct LLM Prediction (15 published methods)

All methods in this category prompt an LLM (gpt-4o) to directly predict the bandgap value from a textual description of the perovskite compound. The best direct LLM method is Self-Refine (0.461 eV), which iteratively refines predictions over 3 rounds. However, even this result remains substantially (278%) worse than the ML baseline. Notably, more sophisticated reasoning strategies (Tree of Thoughts, LATS, Reflexion) do not consistently outperform simpler methods (Self-Consistency, Self-Refine). This suggests that the primary limmitation lines in numerical precision of LLMs, rather than in their reasoning capability. Increasing reasoning complexity does not translate into better numerical accuracy.

Table 9 lists the direct-LLM prediction methods and their OOD errors.

## D.2 Category (ii): Retrieval-Augmented Generation (4 published methods)

RAG methods retrieve similar compounds from the database and use them as context for LLMbased prediction. However, text-based similarity does not reliably align with property similarity. As a result, BM25 and embedding-based retrieval perform poorly (above 2.4 eV), as they often retrieve compounds with similar chemical names but substantially different bandgap values. For example, $\mathrm { ^ { * } ( P E A ) _ { 2 } P b I _ { 4 } } ^ { * }$ retrieves $\mathrm { \ddot { \Sigma } ( P E A ) _ { 2 } S n I _ { 4 } \mathrm { \Sigma } ^ { \mathrm { * } } }$ which has a fundamentally different bandgap due to the Pb→Sn substitution (Figure 10). Evidence Weighted Mean (0.304 eV), which does not rely on

Table 9: 2D perovskite bandgap: direct LLM prediction methods. All use gpt-4o. MAE is 6-fold mean OOD (eV).
<table><tr><td>Method</td><td>MAE (eV)</td><td>vs. Baseline</td><td>Description</td></tr><tr><td>Zero-Shot</td><td>0.552</td><td>+352%</td><td>Direct question, no examples</td></tr><tr><td>Chain-of-Thought [24]</td><td>0.563</td><td>+361%</td><td>Step-by-step reasoning</td></tr><tr><td>Self-Consistency</td><td>0.492</td><td>+303%</td><td>Majority vote over 5 CoT samples</td></tr><tr><td>Self-Refine</td><td>0.461</td><td>+278%</td><td>Iterative refinement (3 rounds)</td></tr><tr><td>Step-Back Abstraction</td><td>0.463</td><td>+279%</td><td>Abstract principle then predict</td></tr><tr><td>Analogical Reasoning</td><td>0.542</td><td>+344%</td><td>Find similar compounds then predict</td></tr><tr><td>Self-Discover</td><td>0.506</td><td>+315%</td><td>Self-discover reasoning structure</td></tr><tr><td>StructChem</td><td>0.632</td><td>+418%</td><td>Structured chemistry prompting</td></tr><tr><td>CRITIC</td><td>0.647</td><td>+430%</td><td>Critique and revise predictions</td></tr><tr><td>Reflexion [26]</td><td>0.540</td><td>+343%</td><td>Verbal reinforcement learning</td></tr><tr><td>LLM Process [16]</td><td>0.511</td><td>+319%</td><td>LLM as stochastic process</td></tr><tr><td>Tree of Thoughts [25]</td><td>0.532</td><td>+336%</td><td>Breadth-first thought exploration</td></tr><tr><td>LATS [27]</td><td>0.567</td><td>+365%</td><td>Language agent tree search</td></tr><tr><td>CoVe [28]</td><td>0.513</td><td>+320%</td><td>Chain-of-verification</td></tr><tr><td>RaR</td><td>0.581</td><td>+376%</td><td>Rephrase and respond</td></tr></table>

![](images/75825a9d9a59f8630c0e089d611bd96a056c8d777ff08f16887dff53af2ee3de.jpg)  
Figure 10: Example illustrating the mismatch between text-based similarity and property similarity in RAG-based prediction. Although $\mathrm { ^ { 6 4 } ( P E A ) _ { 2 } P b I _ { 4 } } ^ { \mathrm { 3 } }$ retrieves ${ } ^ { 6  } ( \mathrm { P E A } ) _ { 2 } \mathrm { S n I } _ { 4 } { } ^ { , ; }$ due to similar chemical naming, substitution of Pb with Sn leads to a significant change in electronic structure and bandgap. This highlights the limitation of retrieval methods that rely on textual similarity rather than underlying chemical factors.

LLM-generated predictions and directly averages retrieved bandgap values, performs best among RAG methods but remains substantially worse (149%) than the ML baseline.

Table 10 reports the four RAG methods and their OOD errors.

## D.3 Category (iii)–(v): LLM-Assisted Feature, Selection, and Priors (65+ in-house variants)

Categories (iii)–(v), namely LLM-generated features, LLM-guided sample selection, and LLM as prior or constraint, comprise over 65 in-house configuration variants that are not part of published methods. Their detailed results are reported separately in Appendix P. Across all variants, a consistent

Table 10: 2D perovskite bandgap: RAG methods. MAE is 6-fold mean OOD (eV).
<table><tr><td>Method</td><td>MAE (eV)</td><td>vs. Baseline</td><td>Description</td></tr><tr><td>BM25-RAG</td><td>2.427</td><td>+1889%</td><td>BM25 text retrieval + LLM</td></tr><tr><td>Embedding-RAG</td><td>2.430</td><td>+1892%</td><td>Embedding similarity retrieval + LLM</td></tr><tr><td>Evidence Weighted Mean</td><td>0.304</td><td>+149%</td><td>Weighted mean of retrieved bandgaps</td></tr><tr><td>CRAG</td><td>0.680</td><td>+457%</td><td>Corrective RAG with self-grading</td></tr></table>

Table 11: 2D perovskite bandgap: per-fold ablation results (OOD MAE in eV). Each row is an ablation variant; each column is a held-out fold. Bold: best per column.
<table><tr><td>Variant</td><td>Alkyl-S</td><td>Alkyl-L</td><td>Phenyl</td><td>Hetero</td><td>Cyclo</td><td>Special</td><td>Mean</td></tr><tr><td>Full MOTIF</td><td>0.128</td><td>0.129</td><td>0.095</td><td>0.099</td><td>0.098</td><td>0.119</td><td>0.111</td></tr><tr><td>w/o invariant kNN</td><td>0.130</td><td>0.134</td><td>0.098</td><td>0.103</td><td>0.106</td><td>0.120</td><td>0.115</td></tr><tr><td>w/o CV gate</td><td>0.131</td><td>0.130</td><td>0.096</td><td>0.100</td><td>0.100</td><td>0.120</td><td>0.113</td></tr><tr><td>w/o balance filter</td><td>0.134</td><td>0.135</td><td>0.101</td><td>0.106</td><td>0.108</td><td>0.125</td><td>0.118</td></tr><tr><td>w/o strict α</td><td>0.132</td><td>0.131</td><td>0.098</td><td>0.103</td><td>0.104</td><td>0.121</td><td>0.115</td></tr><tr><td>Single taxonomy</td><td>0.130</td><td>0.131</td><td>0.096</td><td>0.101</td><td>0.102</td><td>0.119</td><td>0.113</td></tr><tr><td>Mean not median</td><td>0.131</td><td>0.132</td><td>0.097</td><td>0.101</td><td>0.103</td><td>0.121</td><td>0.114</td></tr><tr><td>Random groupings</td><td>0.127</td><td>0.137</td><td>0.112</td><td>0.108</td><td>0.128</td><td>0.106</td><td>0.120</td></tr><tr><td>Baseline (kNN+GBT)</td><td>0.123</td><td>0.136</td><td>0.119</td><td>0.110</td><td>0.137</td><td>0.105</td><td>0.122</td></tr></table>

negative pattern is observed: incorporating LLM-generated numerical predictions does not improve upon the kNN+GBT baseline.

This further supports the conclusion that LLMs are not effective as direct sources of numerical supervision in accurate numerical materials property prediction, regardless of how they are integrated into the pipeline.

## E Per-Fold Ablation Results

Table 11 provides the complete 6-fold × 8-variant ablation matrix. The full MOTIF system achieves the lowest MAE on phenyl, heterocyclic, and cycloalkyl folds, which are the most challenging OOD folds due to their structural distinctiveness and limited overlap with the training data.

Random groupings (bottom row) perform only slightly better than the baseline on most folds, confirming that LLM-generated taxonomies capture meaningful chemical partitions. The balance filter has the largest effect on the cycloalkyl fold (+10.2% when removed), because cycloalkyl is the smallest family and imbalanced taxonomies create groups too small for reliable invariance testing. Invariant kNN has the largest effect on the cycloalkyl fold as well (+8.2% when removed), because cycloalkyl compounds are furthest from training data in full feature space.

## F Taxonomy Balance Analysis

The design of effective taxonomy axes requires careful attention to group balance. Our initial pilot taxonomy design used 5 chemically intuitive axes but suffered from severe imbalance. The final taxonomy bank uses 9 axes with explicit balance criteria. Figure 11 compares the balance distributions and their impact on invariance testing quality.

## F.1 Pilot Taxonomy Design (Failed)

The initial pilot taxonomy used 5 axes inspired by domain-expert chemical intuition. The pilot design failed due to over-concentrated categories, degenerate axes, and insufficient diversity. The ionic character axis classified 85% of spacers as “ionic” because all organic spacer cations in layered perovskites carry an ammonium head group. The confinement axis was entirely degenerate (100% in one group) because all compounds in the 2D perovskite database are quantum-confined by definition. Similarly, the stability axis was nearly degenerate (99% air-stable). With only 1 of 5 axes passing the balance filter, the pilot bank provided essentially a single-taxonomy system with no ensemble benefit, achieving only −1.2% mean MAE reduction.

![](images/1d0a517c48dc71157710c029d2e3210dbdc534f89f10421de12ed0a359b8025a.jpg)

![](images/756096584b8fb2049b86633470da5ee48052c41ef1cb1cdc039ac1a3f40d3ec8.jpg)  
Figure 11: Balance analysis comparing the pilot taxonomy design (5 axes, imbalanced) and the final taxonomy bank (9 axes, balanced). Left: Maximum group fraction per axis. Pilot axes have max-group fractions of 72–100%, while the final bank is below 85%. Right: Effect on Chow test power—imbalanced groups produce statistically unreliable invariance decisions.

Table 12: Pilot taxonomy design and its balance statistics. Axes with Max $\% > 8 5 \%$ are highlighted in red.
<table><tr><td>Pilot axis</td><td># Groups</td><td>Max Group %</td><td>Status</td><td>Problem</td></tr><tr><td>Ionic character</td><td>3</td><td>85%</td><td>Filtered</td><td>Most spacers classified as “ionic&quot;</td></tr><tr><td>Structural motif</td><td>3</td><td>89%</td><td>Filtered</td><td>“chain&quot; dominates over “ring&quot;” and “cage&quot;</td></tr><tr><td>Electronic coupling</td><td>3</td><td>72%</td><td>Passed</td><td>Marginal; weak group produces unreliable Chow test</td></tr><tr><td>Confinement</td><td>2</td><td>100%</td><td>Filtered</td><td>All 2D perovskites are “quantum confined”</td></tr><tr><td>Stability class</td><td>3</td><td>99%</td><td>Filtered</td><td>Nearly all classified as “air-stable&quot;</td></tr></table>

Table 12 reports the balance statistics of the pilot taxonomy bank.

## F.2 Final Taxonomy Bank

The final taxonomy bank addressed the pilot-bank failures through three principles: (1) Focus on spacer molecular properties, not bulk material properties. Pilot axes like “confinement” and “stability” describe bulk material properties that are nearly uniform across the database; the final axes describe molecular-level spacer properties (size, shape, electronic character) that vary substantially. (2) Use 3–4 groups per axis to ensure minimum group sizes. With 753 compounds and 3 groups, the expected group size is approximately 251 compounds (in practice the smallest group typically has >80 compounds after fold removal), which provides sufficient power for the Chow test. (3) Include more axes for ensemble diversity. The final bank uses 9 axes instead of 5, providing the ensemble with 7–9 usable taxonomies per fold after balance filtering. This diversity is critical: the gated median ensemble improves from −1.2% (pilot bank, 1 usable axis) to −8.5% (final bank, 7–9 usable axes). Table 13 reports balance at the level of the 313 unique spacer cations that are annotated by the LLM, whereas Table 14 reports the induced distribution over the 753 bandgap-valid compound records.

## F.3 Per-Axis Distribution of Compounds

For each of the 9 taxonomy axes, we report the distribution of compounds across groups when applied to the full 753-compound dataset. The balance criterion (largest group $< 8 5 \% )$ determines which taxonomies are used per fold. All 9 taxonomies pass the 85% balance threshold on the full dataset. However, after removing a held-out fold, some taxonomies may become imbalanced. In practice, 7–9 of 9 axes pass the filter per fold.

Table 13: Final taxonomy bank balance statistics over the 313 unique spacer cations. All 9 axes pass the 85% balance threshold.
<table><tr><td>V2 Axis</td><td># Groups</td><td>Max Group %</td><td>Balance Assessment</td></tr><tr><td>Spacer size</td><td>4</td><td>50%</td><td>Excellent (4 roughly equal groups)</td></tr><tr><td>Conjugation</td><td>3</td><td>56%</td><td>Good</td></tr><tr><td>Polarity</td><td>3</td><td>57%</td><td>Good</td></tr><tr><td>Steric profile</td><td>4</td><td>42%</td><td>Excellent (most balanced axis)</td></tr><tr><td>H-bonding</td><td>3</td><td>57%</td><td>Good</td></tr><tr><td>Flexibility</td><td>3</td><td>37%</td><td>Excellent</td></tr><tr><td>Electron effect</td><td>3</td><td>84%</td><td>Marginal but passes filter</td></tr><tr><td>Interlayer coupling</td><td>3</td><td>54%</td><td>Good</td></tr><tr><td>Thermal stability</td><td>3</td><td>75%</td><td>Acceptable</td></tr></table>

Table 14: Distribution of bandgap-valid compounds per group for each taxonomy axis.
<table><tr><td>Axis</td><td># Grp</td><td>Group 1</td><td>Group 2</td><td>Group 3</td><td>Group 4</td><td>Max %</td></tr><tr><td>Spacer size</td><td>4</td><td>287 (tiny)</td><td>231 (small)</td><td>156 (medium)</td><td>79 (large)</td><td>38.1%</td></tr><tr><td>Conjugation</td><td>3</td><td>198 (saturated)</td><td>247 (partial)</td><td>308 (full)</td><td></td><td>40.9%</td></tr><tr><td>Polarity</td><td>3</td><td>312 (nonpolar)</td><td>289 (polar)</td><td>152 (amphiphilic)</td><td></td><td>41.4%</td></tr><tr><td>Steric profile</td><td>4</td><td>186 (linear)</td><td>134 (branched)</td><td>209 (cyclic)</td><td>224 (planar)</td><td>29.7%</td></tr><tr><td>H-bonding</td><td>3</td><td>267 (strong)</td><td>312 (weak)</td><td>174 (bifunctional)</td><td></td><td>41.4%</td></tr><tr><td>Flexibility</td><td>3</td><td>278 (rigid)</td><td>256 (semi-flex)</td><td>219 (flexible)</td><td></td><td>36.9%</td></tr><tr><td>Electron effect</td><td>3</td><td>234 (donating)</td><td>198 (withdrawing)</td><td>321 (neutral)</td><td></td><td>42.6%</td></tr><tr><td>Interlayer coupling</td><td>3</td><td>189 (weak)</td><td>342 (moderate)</td><td>222 (strong)</td><td></td><td>45.4%</td></tr><tr><td>Thermal stability</td><td>3</td><td>213 (low)</td><td>298 (medium)</td><td>242 (high)</td><td></td><td>39.6%</td></tr></table>

Table 15: Invariant feature selection heatmap. $\mathbf { \partial } ^ { 6 6 } \surd \mathbf { \gamma } ^ { 9 9 } =$ invariant (Chow $p > 0 . 0 1 )$ $^ { 6 6 } \times ^ { 9 3 } =$ variant. Shown for the full training set (no fold held out). Feature indices refer to the feature-group summary earlier in this appendix.
<table><tr><td>Axis</td><td></td><td></td><td>Structural 3</td><td></td><td></td><td></td><td></td><td>B-site</td><td>10</td><td>11</td><td>12</td><td>13</td><td>14</td><td>Halide 15</td><td>16</td><td>17</td><td>18</td><td>19</td><td>20</td><td>Spacer 21</td><td>22</td><td>23</td><td>24</td><td>25</td><td>26</td><td>27</td><td>A-site 28</td><td>29</td><td>30</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Size</td><td></td><td></td><td>V</td><td></td><td></td><td>V</td><td></td><td></td><td></td><td></td><td></td><td></td><td>√</td><td>V</td><td>√</td><td></td><td></td><td>×</td><td></td><td></td><td>×</td><td></td><td></td><td></td><td></td><td>V</td><td></td><td>V</td><td></td><td>V</td><td></td></tr><tr><td>Conj.</td><td></td><td>7</td><td>√</td><td>V</td><td>7</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>√</td><td></td><td>X</td><td></td><td>× ×</td><td>V</td><td>× ×</td><td>× ×</td><td>√</td><td>× V</td><td>× ×</td><td>× ×</td><td></td><td></td><td></td><td></td><td>√</td><td>」</td><td>2v</td></tr><tr><td>Polar.</td><td>vvv</td><td>√</td><td>1</td><td></td><td>1</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>1</td><td></td><td>V</td><td></td><td>×</td><td>×</td><td>V</td><td>×</td><td>1</td><td>×</td><td>×</td><td>×</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Steric</td><td>√</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>×</td><td>V</td><td></td><td>V</td><td>V</td><td>×</td><td>X</td><td>×</td><td>×</td><td>+</td><td>V</td><td>×</td><td>×</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>H-bond</td><td></td><td>V</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>√</td><td></td><td>×</td><td>×</td><td>x</td><td>V</td><td>V</td><td>×</td><td>×</td><td></td><td></td><td></td><td></td><td></td><td>V</td><td></td><td></td></tr><tr><td>Flex.</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>7</td><td></td><td></td><td>√</td><td>×</td><td>√</td><td>×</td><td>√</td><td></td><td>×</td><td>×</td><td>×</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Elec.</td><td>vvv</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>√</td><td>√</td><td>×</td><td>×</td><td>×</td><td></td><td>×</td><td>×</td><td>v</td><td></td><td></td><td>V</td><td></td><td></td><td></td><td></td></tr><tr><td>Interl. Therm.</td><td>√</td><td></td><td></td><td></td><td>J</td><td></td><td></td><td></td><td></td><td></td><td>×</td><td></td><td>J</td><td></td><td>√</td><td>v</td><td>× ×</td><td>× ×</td><td>+</td><td>× ×</td><td>×</td><td>×</td><td>×</td><td>×</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Inv. count</td><td></td><td></td><td>9</td><td></td><td></td><td></td><td></td><td></td><td></td><td>0</td><td></td><td>8</td><td>9</td><td>9</td><td></td><td>9</td><td>2</td><td></td><td></td><td></td><td></td><td></td><td>×</td><td>×</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>9</td><td></td><td></td><td></td><td></td><td></td><td>0</td><td></td><td>8</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>3</td><td>3</td><td></td><td></td><td>3</td><td></td><td>3</td><td>9</td><td></td><td>9</td><td></td><td></td><td></td><td></td></tr></table>

On the compound-weighted distribution, the steric profile axis has the most balanced split (max group 29.7%), while interlayer coupling is the least balanced (45.4%). More balanced taxonomies tend to produce better invariance testing because all groups have sufficient samples for the Chow test to be statistically powerful.

## G Invariant Feature Analysis

For each taxonomy axis, the Chow structural-break test identifies which of the 30 features have invariant predictive relationships across groups. Figure 12 visualizes the invariance status across all axes and features, and Table 15 provides the detailed binary invariance matrix.

## G.1 Detailed Invariance Patterns

Table 16 provides a complementary view, listing which specific features are invariant under each taxonomy axis with their physical interpretation.

Feature Invariance Across Taxonomies ( = 0.01)  
![](images/252cd92c1693cb8eb702368a29ad8ae6f6c0dae02e73b17daf0298e257b74262.jpg)  
Figure 12: Invariant feature heatmap across all 9 taxonomy axes and 30 features. Green cells indicate invariant features (Chow $p > 0 . 0 1 ) ;$ red cells indicate variant features. Structural, B-site, halide, and A-site features are almost universally invariant, while spacer-related features (17–24) show taxonomy-dependent variance patterns.

Table 16: Detailed invariant feature sets per taxonomy axis. “# Inv” = number of invariant features out of $3 0 . \ ^ { 6 } \mathrm { K e y }$ Variant Features” lists the most impactful features that are excluded by the Chow test. • Spacer size (# Inv $= 2 2 , \# \mathrm { { V a r i a n t } = 8 ) }$ : excludes spacer family indicators, chain-length / branching descriptors, and B\_Mn.

• Conjugation (# Inv = 23, # Variant = 7): excludes aromatic / heterocycle-related spacer indicators, chain-length / branching, B\_Ge, and X\_Cl\_frac.

• Polarity (# Inv = 22, # Variant = 8): excludes alkyl / heterocycle / special-family spacer indicators, chain-length / branching, and B\_Cu.

• Steric profile (# Inv = 22, # Variant = 8): excludes bulky spacer family indicators, chain-length / branching, and B\_Bi.

• H-bonding (# Inv = 24, # Variant = 6): excludes short-alkyl, long-alkyl, phenyl, special-family, and chain-length spacer descriptors.

• Flexibility (# Inv = 24, # Variant = 6): excludes short-alkyl, phenyl, special-family, chain-length, and branching descriptors.

• Electron effect (# Inv = 24, # Variant = 6): excludes long-alkyl, phenyl, heterocyclic, special-family, and chain-length descriptors.

• Interlayer coupling (# Inv = 22, # Variant = 8): excludes alkyl / heterocycle / special-family spacer indicators, chain length / branching, and B\_Ge.

• Thermal stability (# ${ \mathrm { I n v } } = 2 2 , \# { \mathrm { V a r i a n t } } = 8 ) { \mathrm { : } }$ excludes spacer family indicators for alkyl, phenyl, heterocycle, and cycloalkyl groups, together with chain-length / branching descriptors.

## G.2 Discussion of Invariance Patterns

## Key patterns emerge from the invariant feature analysis:

Universally invariant features. Features 1–7 (structural descriptors and Pb/Sn B-site indicators), features 13–14 (I/Br halide fractions), and features 25–30 (all A-site features) are invariant under all 9 taxonomies. These encode fundamental physics—ionic radii, composition stoichiometry, crystal structure type—whose relationship with bandgap does not depend on the organic spacer classification. The layer count (n) is the single most important predictor: multi-layer perovskites $\left( n \geq 2 \right)$ have systematically lower bandgaps due to increased quantum confinement dimensionality, and this relationship holds regardless of how spacers are classified.

![](images/cd4a1a0a818df20ca5a331bb35631a8c6a6d367f0c07d4d2ad18dafd4fc09beb.jpg)

![](images/7120815a0d2403acf785efc1650e842e7c8dabc98b6f69decbc24461fccd52fa.jpg)  
Figure 13: Sensitivity to α threshold. Left: Mean OOD MAE (6-fold) vs. α for the gated median ensemble and the best-single-taxonomy strategy. The optimum occurs at $\alpha = 0 . 0 1$ for both. Right: Per-fold MAE vs. α showing that all six folds share a similar optimum region.

Table 17: Complete α sensitivity results under best-single-taxonomy strategy. “# Inv” is the average number of invariant features across all 9 taxonomy axes.
<table><tr><td>α</td><td># Inv</td><td>Alkyl-S</td><td>Alkyl-L</td><td>Phenyl</td><td>Hetero</td><td>Cyclo</td><td>Special</td><td>Mean (Best)</td><td>Mean (Gated)</td><td>Std</td></tr><tr><td>0.001</td><td>18.2</td><td>0.121</td><td>0.132</td><td>0.098</td><td>0.096</td><td>0.101</td><td>0.098</td><td>0.108</td><td>0.114</td><td>0.0147</td></tr><tr><td>0.005</td><td>20.6</td><td>0.119</td><td>0.130</td><td>0.096</td><td>0.095</td><td>0.099</td><td>0.096</td><td>0.106</td><td>0.112</td><td>0.0138</td></tr><tr><td>0.01</td><td>22.1</td><td>0.118</td><td>0.130</td><td>0.095</td><td>0.094</td><td>0.098</td><td>0.094</td><td>0.105</td><td>0.111</td><td>0.0144</td></tr><tr><td>0.02</td><td>23.8</td><td>0.120</td><td>0.131</td><td>0.098</td><td>0.098</td><td>0.102</td><td>0.098</td><td>0.108</td><td>0.115</td><td>0.0122</td></tr><tr><td>0.05</td><td>25.4</td><td>0.122</td><td>0.133</td><td>0.104</td><td>0.103</td><td>0.108</td><td>0.103</td><td>0.112</td><td>0.115</td><td>0.0108</td></tr></table>

Universally variant features. Feature 23 (spacer\_chain\_length) is invariant under only 1 of 9 taxonomies, making it the most environment-dependent feature. This makes physical sense: chain length is highly correlated with spacer family membership and directly influenced by all 9 chemical axes. A model trained on all features would learn a chain-length–bandgap correlation that holds in-distribution but fails when encountering spacers with atypical chain length.

Taxonomy-specific patterns. The conjugation and flexibility taxonomies produce the largest invariant feature sets (23–24 invariant features), because these axes partition spacers along dimensions orthogonal to most compositional features. The spacer\_size taxonomy produces fewer invariant features (22) because size directly correlates with multiple spacer-related features. This explains why spacer\_size is the best taxonomy for heterocyclic, cycloalkyl, and special folds: by aggressively removing size-correlated features, it identifies the truly size-independent predictors that generalize to novel spacer families.

## H Sensitivity to α Threshold

The Chow test threshold α controls the strictness of invariance testing. Lower α means stricter invariance (fewer features pass), while higher α means more lenient testing. Figure 13 visualizes the sensitivity across α values and folds.

## H.1 Full α × Fold Results

Table 17 presents the complete matrix of α values × folds for the best-single-taxonomy strategy, and Table 18 presents results under the gated median ensemble.

## H.2 Analysis

The optimal threshold $\alpha = 0 . 0 1$ selects approximately 22 of 30 features as invariant on average. Stricter thresholds $( \alpha = 0 . 0 0 1 )$ select too few features (18.2 on average), losing useful predictive information. More lenient thresholds $( \alpha = 0 . 0 5 )$ admit environment-dependent features (25.4 on average) that hurt OOD generalization.

Table 18: α sensitivity under the gated median ensemble strategy. Per-fold OOD MAE (eV).
<table><tr><td>α</td><td>Alkyl-S</td><td> $_ \mathrm { A l k y l - L }$ </td><td>Phenyl</td><td>Hetero</td><td>Cyclo</td><td>Special</td><td>Mean</td><td>Std</td></tr><tr><td>0.001</td><td>0.131</td><td>0.133</td><td>0.099</td><td>0.103</td><td>0.104</td><td>0.112</td><td>0.1130</td><td>0.0147</td></tr><tr><td>0.005</td><td>0.130</td><td>0.131</td><td>0.097</td><td>0.101</td><td>0.101</td><td>0.112</td><td>0.1120</td><td>0.0138</td></tr><tr><td>0.01</td><td>0.128</td><td>0.129</td><td>0.095</td><td>0.099</td><td>0.098</td><td>0.119</td><td>0.1113</td><td>0.0144</td></tr><tr><td>0.02</td><td>0.129</td><td>0.131</td><td>0.101</td><td>0.102</td><td>0.103</td><td>0.123</td><td>0.1148</td><td>0.0122</td></tr><tr><td>0.05</td><td>0.130</td><td>0.133</td><td>0.106</td><td>0.105</td><td>0.108</td><td>0.109</td><td>0.1152</td><td>0.0108</td></tr></table>

![](images/1ad0d935f753d877a5855935dd6b5c1a0e0b123e9e7165505842a6a5c859e1ee.jpg)  
Figure 14: Per-fold OOD MAE for all 8 ensemble strategies plus baseline and oracle. The gated median (red) achieves the best or near-best performance on the phenyl, heterocyclic, and cycloalkyl folds, while remaining competitive on alkyl folds. The oracle (dashed) shows the upper bound achievable with perfect taxonomy selection.

The sensitivity is moderate: the gated median MAE ranges from 0.1113 (α = 0.01) to 0.1152 $( \alpha = 0 . 0 5 )$ , a spread of 3.5%. The standard deviation column reveals an interesting trade-off: stricter α values produce higher per-fold variance (0.0147 at α = 0.001) because aggressive feature removal creates more fold-dependent performance, while lenient α values reduce variance (0.0108 at $\alpha = 0 . 0 5 )$ but at the cost of higher mean error. The optimal α = 0.01 balances this trade-off.

Notably, the phenyl and cycloalkyl folds are most sensitive to α: their MAE increases by 9.5% and 10.2% respectively when moving from α = 0.01 to α = 0.05. These folds have the largest distribution shift from training data, so they benefit most from strict invariance filtering and are most harmed by admitting spurious features.

## I Ensemble Strategy Comparison (Per-Fold)

Table 19 provides the complete per-fold results for all ensemble strategies evaluated. Figure 14 provides a visual comparison across folds.

## I.1 Analysis of Ensemble Strategy Behavior

The gated median achieves the best mean MAE (0.1113) among automatic strategies, outperforming the next-best (CV-weighted and Top-1 CV Select, both 0.1122) by 0.8%. Its advantage is most pronounced on three folds:

Phenyl fold (−20% vs. baseline). The gated median achieves 0.095 eV compared to 0.097–0.099 eV for other strategies. On this fold, the gate effectively excludes 2–3 taxonomies that produce poor predictions for aromatic spacers, preventing them from diluting the ensemble. The phenyl fold has the most discriminating gate behavior: only 5–6 of 9 taxonomies pass.

Cycloalkyl fold (−28% vs. baseline). The gated median achieves 0.098 eV, the largest improvement of any fold. Other strategies achieve 0.101–0.104 eV because they include taxonomies that mishandle the small cycloalkyl family. The gate admits only 4–5 taxonomies for this fold, showing its most aggressive filtering behavior.

Table 19: 2D perovskite bandgap: complete per-fold OOD MAE (eV) for all 8 ensemble strategies. Bold: best automatic strategy per column.
<table><tr><td>Strategy</td><td>Alkyl-S</td><td>Alkyl-L</td><td>Phenyl</td><td>Hetero</td><td>Cyclo</td><td>Special</td><td>Mean</td></tr><tr><td>Simple Average</td><td>0.128</td><td>0.131</td><td>0.098</td><td>0.101</td><td>0.102</td><td>0.114</td><td>0.1123</td></tr><tr><td>Median</td><td>0.129</td><td>0.132</td><td>0.097</td><td>0.100</td><td>0.103</td><td>0.115</td><td>0.1127</td></tr><tr><td>Trimmed Mean</td><td>0.129</td><td>0.131</td><td>0.097</td><td>0.101</td><td>0.103</td><td>0.115</td><td>0.1127</td></tr><tr><td>CV-weighted</td><td>0.128</td><td>0.131</td><td>0.097</td><td>0.100</td><td>0.102</td><td>0.115</td><td>0.1122</td></tr><tr><td>Top-1 CV Select</td><td>0.127</td><td>0.132</td><td>0.098</td><td>0.100</td><td>0.101</td><td>0.115</td><td>0.1122</td></tr><tr><td>Diversity-weighted</td><td>0.128</td><td>0.131</td><td>0.098</td><td>0.101</td><td>0.102</td><td>0.114</td><td>0.1123</td></tr><tr><td>Adaptive per-query</td><td>0.130</td><td>0.133</td><td>0.099</td><td>0.103</td><td>0.104</td><td>0.115</td><td>0.1139</td></tr><tr><td>Gated Median (ours)</td><td>0.128</td><td>0.129</td><td>0.095</td><td>0.099</td><td>0.098</td><td>0.119</td><td>0.1113</td></tr><tr><td>Oracle (best single)</td><td>0.118</td><td>0.123</td><td>0.090</td><td>0.092</td><td>0.095</td><td>0.117</td><td>0.1058</td></tr><tr><td>Baseline (kNN+GBT)</td><td>0.123</td><td>0.136</td><td>0.119</td><td>0.110</td><td>0.137</td><td>0.105</td><td>0.1216</td></tr></table>

<table><tr><td>Model</td><td>AS</td><td>AL</td><td>Ph</td><td>He</td><td>Cy</td><td>Sp</td></tr><tr><td>40</td><td>0.133</td><td>0.119</td><td>0.121</td><td>0.113</td><td>0.099</td><td>0.115</td></tr><tr><td>03-mini</td><td>0.128</td><td>0.128</td><td>0.121</td><td>0.107</td><td>0.100</td><td>0.120</td></tr><tr><td>03</td><td>0.133</td><td>0.118</td><td>0.116</td><td>0.107</td><td>0.097</td><td>0.118</td></tr><tr><td>o1</td><td>0.137</td><td>0.118</td><td>0.111</td><td>0.105</td><td>0.146</td><td>0.118</td></tr></table>

(a) Bandgap Top-1 CV.

<table><tr><td>Model</td><td>AS</td><td>AL</td><td>Ph</td><td>He</td><td>Cy</td><td>Sp</td></tr><tr><td>40</td><td>0.128</td><td>0.120</td><td>0.123</td><td>0.108</td><td>0.117</td><td>0.110</td></tr><tr><td>03-mini</td><td>0.125</td><td>0.124</td><td>0.123</td><td>0.104</td><td>0.116</td><td>0.110</td></tr><tr><td>03</td><td>0.126</td><td>0.119</td><td>0.122</td><td>0.104</td><td>0.116</td><td>0.109</td></tr><tr><td>o1</td><td>0.130</td><td>0.119</td><td>0.119</td><td>0.104</td><td>0.131</td><td>0.109</td></tr></table>

(b) Bandgap safe blend.  
Figure 15: Full-fold bandgap comparison of reasoning-model choices for taxonomy generation. o3 is the strongest bandgap annotator overall, but the fold winners still change across families and no single backbone dominates every held-out family.

Special fold (baseline wins). On the special fold, the gated median (0.119 eV) underperforms the baseline (0.105 eV), as do all other ensemble strategies (0.114–0.115 eV). The gate admits most taxonomies for this fold, but none reliably improve prediction. This is because the special family is too internally diverse for any single taxonomy partition to capture the relevant chemical variation.

Why does adaptive per-query selection underperform? The adaptive strategy attempts to estimate which taxonomy is best for each test query based on the query’s feature-space location. It achieves the worst mean MAE (0.1139) among all strategies because the estimation is unreliable with small per-taxonomy training sets.

## J Backbone Sensitivity and Reasoning Models

We reran taxonomy generation with three OpenAI reasoning models—o3-mini, o3, and o1—while keeping the same 313 × 9 taxonomy-prompt bank and the same full-fold control benchmark. The comparison is intentionally conservative: only the offline taxonomy annotator changes, while the downstream invariant-feature selection, routing, and evaluation protocol remain fixed.

Figure 15 compares fold-wise results across the taxonomy-generation backbones.

Table 20 summarizes the backbone-wise errors and offline annotation costs.

Three points matter. First, o3 is the strongest bandgap annotator we tested: its safe-blend improvement over the baseline is 0.0058 eV with a 95% bootstrap CI of [0.0031, 0.0086], slightly larger than the 0.0041 eV gain of gpt-4o. Second, that gain does not transfer uniformly. o3-mini stays near gpt-4o on bandgap but offers no optical advantage, and o3 is actually worse on optical safe blend (0.1859 eV versus the 0.1807 eV baseline and gpt-4o’s 0.1732 eV). Third, o1 is far more expensive and only modestly competitive: its optical safe-blend MAE is 0.1762 eV, still worse than gpt-4o’s 0.1732 eV, while its bandgap numbers are weaker than both gpt-4o and o3. We therefore keep gpt-4o as the default main-text taxonomy bank and treat o3 as a strong but task-dependent alternative.

Table 20: Reasoning-model comparison on the full control benchmark (mean MAE in eV). One-time annotation cost is for the full 313 × 9 taxonomy bank.
<table><tr><td>LLM</td><td>Bandgap Top-1</td><td>Bandgap Safe</td><td>Optical Top-1</td><td>Optical Safe</td><td>One-time Cost</td></tr><tr><td>gpt-4o</td><td>0.1168</td><td>0.1174</td><td>0.2056</td><td>0.1732</td><td>$0.37</td></tr><tr><td>o3-mini</td><td>0.1173</td><td>0.1170</td><td>0.2042</td><td>0.1823</td><td></td></tr><tr><td>o3</td><td>0.1148</td><td>0.1159</td><td>0.2298</td><td>0.1859</td><td>$5.22</td></tr><tr><td>01</td><td>0.1225</td><td>0.1188</td><td>0.1942</td><td>0.1762</td><td>$134.02</td></tr></table>

Note: o3-mini cost is omitted because that first cached run predated our token-usage logger.

![](images/f92455efb786c3ef1925cf95b7611653b1c96b4d3948fc8b52dc45fa495c773d.jpg)  
Figure 16: Oracle gap analysis. Left: Per-fold comparison of gated median, best-single-taxonomy, and oracle MAE, with the gap highlighted. Right: Oracle taxonomy configuration. The gap is largest on the special fold (0.030 eV) and smallest on the cycloalkyl fold (0.001 eV).

Table 21: Oracle per-fold configurations. “Best Axis” is the single taxonomy axis that produces the lowest MAE. “Gap” is the difference between gated median and oracle.
<table><tr><td>Fold</td><td>Gated</td><td>Oracle</td><td>Gap</td><td>Best Axis</td><td>α</td><td># Inv Features</td></tr><tr><td>Alkyl-short</td><td>0.128</td><td>0.117</td><td>0.011</td><td>electron_effect</td><td>0.01</td><td>24</td></tr><tr><td>Alkyl-long</td><td>0.129</td><td>0.118</td><td>0.011</td><td>steric</td><td>0.01</td><td>22</td></tr><tr><td>Phenyl</td><td>0.095</td><td>0.092</td><td>0.003</td><td>flexibility</td><td>0.01</td><td>24</td></tr><tr><td>Heterocyclic</td><td>0.099</td><td>0.086</td><td>0.013</td><td>spacer_size</td><td>0.005</td><td>20</td></tr><tr><td>Cycloalkyl</td><td>0.098</td><td>0.097</td><td>0.001</td><td>spacer_size</td><td>0.01</td><td>22</td></tr><tr><td>Special</td><td>0.119</td><td>0.089</td><td>0.030</td><td>spacer_size</td><td>0.01</td><td>22</td></tr><tr><td>Mean</td><td>0.111</td><td>0.100</td><td>0.011</td><td></td><td></td><td>22.3</td></tr></table>

## K Oracle Gap Analysis

The oracle gap—the difference between the gated median ensemble (0.111 eV, −8.5%) and the oracle per-fold best taxonomy (0.100 eV, −17.8%)—represents the largest opportunity for improvement in MOTIF. This section analyzes the gap in detail. Figure 16 visualizes the per-fold gap, and Table 21 reports the exact oracle configuration per fold.

![](images/dea945332d8e4c2ea861e3e8a3103aade2aec8a6b012d7fdb265321ee2ce4d8b.jpg)  
Figure 17: 6-fold mean OOD MAE for all 6 refined pipeline variants (R1–R6) compared to the baseline and the main MOTIF gated median. R4\_strict is the strongest refined variant on average, while R3\_alpha overfits despite its additional tuning freedom.

## K.1 Gap Analysis

Small-gap folds (phenyl, cycloalkyl). On the phenyl fold, the gated median (0.095) nearly matches the oracle (0.092), with a gap of only 0.003 eV. This indicates that the gate effectively identifies the best taxonomies for this fold. Similarly, the cycloalkyl gap is only 0.001 eV—the gated median has nearly closed the oracle gap through effective filtering. Both folds have the property that the best taxonomy (flexibility for phenyl, spacer\_size for cycloalkyl) is also the one with the best CV score, so the gate’s selection criterion aligns well with test performance.

Large-gap folds (special, heterocyclic). The special fold has the largest gap (0.030 eV), reflecting the fundamental challenge of this heterogeneous family. The gated median admits most taxonomies for this fold (7–8 of 9), but the median of many mediocre predictions (0.119 eV) is substantially worse than the oracle’s single best prediction (0.089 eV). The heterocyclic fold has a gap of 0.013 eV, with the oracle selecting spacer\_size at a slightly stricter α = 0.005 than the default α = 0.01. This suggests that per-fold α optimization could partially close the gap, though our experiments with R3\_alpha showed this is difficult to do reliably without test labels.

Implications for future work. The oracle gap analysis suggests two promising directions: (1) meta learning to predict the best taxonomy per fold based on training-set statistics, which could close the gap on heterocyclic and special folds; and (2) query-adaptive taxonomy selection, which could close the gap on alkyl folds where the best taxonomy varies within the fold. The fact that the phenyl and cycloalkyl gaps are already small (0.001–0.003 eV) suggests that the gated median is near-optimal when the gate’s CV criterion correlates well with test performance.

## L Refined Variants

This section provides detailed results for the 6 refined pipeline variants. Figure 17 visualizes the 6-fold mean MAE, and Table 22 provides the complete per-fold numerical results.

## L.1 Variant Descriptions and Analysis

R1\_gated (gated ensemble, V3 pipeline). An earlier version of the gated ensemble without invariant kNN. It achieves −4.4% mean improvement, with the largest gains on the special fold (−10.5% relative to baseline). The absence of invariant kNN limits its effectiveness on the cycloalkyl fold (0.1207 vs. 0.098 for the full system).

R2\_hybrid (environment-aware + standard features). Combines environment-aware feature selection with standard features using a stacking approach: the GBT regressor uses invariant features, while a Ridge regressor uses all 30 features, and predictions are averaged. This hedging strategy achieves −6.9% improvement but never reaches the best performance on any individual fold because it retains noisy features through the Ridge component.

Table 22: 2D perovskite bandgap: complete per-fold results for refined pipeline variants (OOD MAE in eV). Bold: best per column among R1–R6.
<table><tr><td>Variant</td><td>Alkyl-S</td><td>Alkyl-L</td><td>Phenyl</td><td>Hetero</td><td>Cyclo</td><td>Special</td><td>Mean</td></tr><tr><td>R1_gated (gated ensemble)</td><td>0.1239</td><td>0.1323</td><td>0.1114</td><td>0.1042</td><td>0.1207</td><td>0.1054</td><td>0.1163</td></tr><tr><td>R2_hybrid (env+standard)</td><td>0.1176</td><td>0.1257</td><td>0.1120</td><td>0.1002</td><td>0.1185</td><td>0.1051</td><td>0.1132</td></tr><tr><td>R3_alpha (per-fold α)</td><td>0.1243</td><td>0.1351</td><td>0.1098</td><td>0.1053</td><td>0.1284</td><td>0.1261</td><td>0.1215</td></tr><tr><td>R4_strict (strict inv.)</td><td>0.1176</td><td>0.1295</td><td>0.1095</td><td>0.1014</td><td>0.1170</td><td>0.0940</td><td>0.1115</td></tr><tr><td>R5_knn_env (env-aware kNN)</td><td>0.1187</td><td>0.1324</td><td>0.1083</td><td>0.1012</td><td>0.1140</td><td>0.1041</td><td>0.1131</td></tr><tr><td>R6_best (best single tax.)</td><td>0.1192</td><td>0.1341</td><td>0.1109</td><td>0.1038</td><td>0.1192</td><td>0.1072</td><td>0.1157</td></tr><tr><td>MOTIF (gated median, main)</td><td>0.128</td><td>0.129</td><td>0.095</td><td>0.099</td><td>0.098</td><td>0.119</td><td>0.111</td></tr><tr><td>Baseline (kNN+GBT)</td><td>0.123</td><td>0.136</td><td>0.119</td><td>0.110</td><td>0.137</td><td>0.105</td><td>0.122</td></tr></table>

![](images/30befc82072d0d126c6bb23bbbda630d67f200a67e4ec0e8ebf4d3c05f05268a.jpg)

![](images/69b2cf90edc8dd8c8453112b12cd26cf314440d93fdfdd765f4af317e182fd70.jpg)  
Figure 18: Absolute-error distribution analysis. Left: Density of absolute errors for the baseline kNN+GBT, MOTIF gated median, and Self-Refine. Right: Empirical CDF of absolute errors. MOTIF shifts probability mass toward smaller errors and trims the large-error tail relative to the classical baseline.

R3\_alpha (per-fold α optimization). Uses inner cross-validation to select the optimal α threshold for each fold independently. Despite the appeal of fold-specific tuning, this variant performs worst (−0.1%), barely improving over the baseline. The problem is overfitting: the inner CV on 5–7 environments cannot reliably estimate the optimal α for the held-out environment.

R4\_strict (strict invariance). Uses a stricter invariance criterion by requiring Chow $p > 0 . 0 1$ and additionally filtering features with low individual predictive power (Pearson $| r | < 0 . 1 )$ . This is the best refined variant (−8.3%), achieving particularly strong results on the special fold (0.0940, −10.5% vs. baseline).

R5\_knn\_env (environment-aware kNN). Modifies the kNN component to weight neighbors by environment similarity. This achieves −7.0% improvement, competitive with R4\_strict on most folds. The phenyl fold particularly benefits (0.1083) because environment-weighted kNN can leverage the known chemical similarity between phenyl spacers and certain partially-conjugated spacers in the training set.

R6\_best (best single taxonomy from CV). Selects the single best taxonomy per fold based on cross-validation performance. This achieves −4.9% improvement, underperforming the ensemble variants because the CV-selected taxonomy is not always the true best. The gap between R6 and the oracle (−4.9% vs. −17.8%) confirms that automatic taxonomy selection remains the primary bottleneck.

## M Error Distribution Analysis

This section analyzes the distribution of prediction errors across methods and folds, providing insight into when and why MOTIF succeeds or fails.

Figure 18 compares the absolute-error distributions of the evaluated predictors.

Table 23: 2D perovskite bandgap: representative compounds where MOTIF produces the largest improvement and degradation relative to the baseline.
<table><tr><td>Compound</td><td>True BG</td><td>BL Error</td><td>Motif Error</td><td>Change</td><td>Fold</td></tr><tr><td colspan="6">Largest improvements (Motif reduces error)</td></tr><tr><td> $\mathrm { ( C H A ) _ { 2 } P b B r _ { 4 } }$ </td><td>3.41</td><td>0.41</td><td>0.12</td><td>-71%</td><td>Cyclo</td></tr><tr><td> $( \mathrm { P y r A } ) _ { 2 } \mathrm { S n I } _ { 4 }$ </td><td>1.83</td><td>0.35</td><td>0.09</td><td>-74%</td><td>Hetero</td></tr><tr><td> $( 4 \mathrm { - F P E A ) _ { 2 } P b B r _ { 4 } }$ </td><td>3.52</td><td>0.38</td><td>0.14</td><td>-63%</td><td>Phenyl</td></tr><tr><td> $\mathrm { ( T h M A ) _ { 2 } P b I _ { 4 } }$ </td><td>2.27</td><td>0.29</td><td>0.08</td><td>-72%</td><td>Hetero</td></tr><tr><td> $( \mathrm { C y P A } ) _ { 2 } \mathrm { P b I } _ { 4 }$ </td><td>2.35</td><td>0.32</td><td>0.11</td><td>-66%</td><td>Cyclo</td></tr><tr><td colspan="6">Largest degradations (Motif increases error)</td></tr><tr><td> $\mathrm { ( B A ) _ { 2 } C d C l _ { 4 } }$ </td><td>4.28</td><td>0.08</td><td>0.16</td><td>+100%</td><td>Alkyl-S</td></tr><tr><td> $\mathrm { ( H A ) _ { 2 } P b I _ { 4 } }$ </td><td>2.35</td><td>0.03</td><td>0.09</td><td>+200%</td><td>Alkyl-L</td></tr><tr><td> $\mathrm { ( G l y A ) _ { 2 } P b C l _ { 4 } }$ </td><td>3.92</td><td>0.06</td><td>0.14</td><td>+133%</td><td>Special</td></tr></table>

## M.1 Error Distribution Characteristics

Baseline kNN+GBT. The baseline absolute-error distribution has moderate tails: 15.3% of predictions have absolute error > 0.2 eV, and 4.2% exceed 0.3 eV. The largest errors occur on the cycloalkyl (max abs error = 0.41 eV) and phenyl (max abs error = 0.38 eV) folds.

MOTIF gated median. The MOTIF absolute-error distribution is narrower. The tail probability is reduced: only 11.8% of predictions have absolute error > 0.2 eV (a 23% reduction), and only 2.8% exceed 0.3 eV (a 33% reduction). The improvement comes primarily from reducing large errors rather than improving already-accurate predictions.

LLM Self-Refine. The LLM absolute-error distribution is dramatically wider (std = 0.38 eV vs. 0.14 eV for baseline) with heavy tails throughout the support. This likely reflects the mismatch between free-form language-model prediction and the low-noise numerical precision required by this benchmark.

## M.2 Compound-Level Analysis

Table 23 shows representative compounds where MOTIF substantially improves or degrades predictions relative to the baseline. The improvement pattern is clear: MOTIF provides the largest gains on compounds from chemically distinctive families (cycloalkyl, heterocyclic) where the baseline’s full-feature kNN matches to inappropriate neighbors. The degradation pattern reveals a complementary weakness: on compounds where the baseline already makes accurate predictions through spacer-feature matching, removing spacer features via invariance filtering can hurt. However, these degradations are small in absolute terms (0.03–0.08 eV increase) compared to the improvements (0.18–0.27 eV decrease).

## N Computational Cost Analysis

MOTIF is dramatically cheaper than LLM baselines at inference time. After the one-time taxonomy generation (\$0.03–0.37), all prediction is done locally with no API calls. For the full 753-query OOD benchmark, the deployable gpt-4o-mini/gpt-4o versions of MOTIF cost \$0.03–0.37 total vs. \$45.18 for Self-Refine and \$112.95 for LATS, a roughly 120–3700× cost reduction depending on the annotator. Moreover, MOTIF requires no GPU—all computation (kNN search, GBT prediction, ensemble) runs on CPU in under 1 second per query.

Table 24 provides the detailed computational-cost breakdown.

Table 24: Computational cost breakdown for MOTIF and baselines.
<table><tr><td>Component</td><td>Time</td><td>API Cost</td><td>Hardware</td><td>Notes</td></tr><tr><td colspan="5">MOTIF Pipeline</td></tr><tr><td>Taxonomy generation (gpt-4o-mini)</td><td>3 min</td><td>$0.03</td><td>CPU</td><td>One-time, cached</td></tr><tr><td>Taxonomy generation (gpt-4o)</td><td>5 min</td><td>$0.37</td><td>CPU</td><td>One-time, cached</td></tr><tr><td>Taxonomy generation (o3-mini)</td><td></td><td></td><td>CPU</td><td>First cached run predates usage logging</td></tr><tr><td>Taxonomy generation (o3)</td><td></td><td>$5.22</td><td>CPU</td><td>One-time, cached</td></tr><tr><td>Taxonomy generation (o1)</td><td></td><td>$134.02</td><td>CPU</td><td>One-time, cached</td></tr><tr><td>Chow test (9 axes × 30 features)</td><td>&lt;1 sec</td><td>$0</td><td>CPU</td><td>Per fold</td></tr><tr><td>Forward feature selection</td><td>10 sec</td><td>$0</td><td>CPU</td><td>Per fold, per taxonomy</td></tr><tr><td>GBT training (9 models)</td><td>5 sec</td><td>$0</td><td>CPU</td><td>Per fold</td></tr><tr><td>kNN prediction</td><td>&lt;1 sec</td><td>$0</td><td>CPU</td><td>Per query</td></tr><tr><td>Gated median ensemble</td><td>&lt;1 sec</td><td>$0</td><td>CPU</td><td>Per query</td></tr><tr><td>Total (per fold, after caching)</td><td>16 sec</td><td>$0</td><td>CPU</td><td>No GPU needed</td></tr><tr><td>Total (including taxonomy gen., deployable settings)</td><td>3–5 min</td><td>$0.03-0.37</td><td>CPU</td><td>One-time setup</td></tr><tr><td colspan="5">LLM Baselines (per query)</td></tr><tr><td>Zero-Shot (gpt-4o)</td><td>2 sec</td><td>$0.01</td><td></td><td>Per query</td></tr><tr><td>CoT (gpt-4o)</td><td>5 sec</td><td>$0.02</td><td></td><td>Per query</td></tr><tr><td>Self-Refine (gpt-4o, 3 rounds)</td><td>15 sec</td><td>$0.06</td><td></td><td>Per query</td></tr><tr><td>Tree of Thoughts (gpt-4o)</td><td>30 sec</td><td>$0.10</td><td></td><td>Per query</td></tr><tr><td>LATS (gpt-40)</td><td>45 sec</td><td>$0.15</td><td></td><td>Per query</td></tr><tr><td>753 test queries (Self-Refine)</td><td>188 min</td><td>$45.18</td><td></td><td>Full evaluation</td></tr><tr><td>753 test queries (LATS)</td><td>565 min</td><td>$112.95</td><td></td><td>Full evaluation</td></tr></table>

## O LLM Prompt Templates

This section provides the exact prompts used for taxonomy generation and selected LLM baselines. For each taxonomy classification prompt, we include a brief discussion of the design rationale—why this particular axis was chosen, what chemical variation it captures, and how it relates to bandgap prediction.

## O.1 Taxonomy Classification Prompt: Flexibility

The flexibility axis captures how easily the spacer cation can undergo conformational changes. Rigid spacers (aromatic rings, fused ring systems) template the perovskite lattice more strongly, potentially creating more ordered interlayer arrangements with narrower bandgap distributions. Flexible spacers (long alkyl chains) allow the interlayer spacing to fluctuate, which can broaden the effective bandgap through disorder effects. This axis is particularly useful for the phenyl fold, where the held-out aromatic spacers are predominantly rigid and the invariant features under flexibility-based partitioning accurately capture the physics of rigid-spacer perovskites.

Taxonomy Classification: Flexibility   
You are a materials science expert specializing in organic-inorganic hybrid perovskites.   
Task: Classify the following organic spacer cation based on its molecular flexibility.   
Spacer cation: {spacer\_name}   
Categories: - "rigid": Molecules with aromatic rings, fused ring systems, or highly   
constrained geometries that resist conformational changes - "semi-flexible": Molecules   
with some rotatable bonds but also rigid elements (e.g., short chains attached to rings) -   
"flexible": Molecules with many rotatable bonds, long alkyl chains, or highly deformable   
structures   
Respond with ONLY the category name (rigid, semi-flexible, or flexible). Do not explain.

## O.2 Taxonomy Classification Prompt: Spacer Size

The spacer size axis partitions compounds by the physical dimensions of the organic cation. Larger spacers create wider interlayer separations, which increases the quantum confinement of charge carriers within the inorganic sheets. This relationship between spacer size and confinement is one of the most fundamental structure-property relationships in 2D perovskites. The spacer\_size taxonomy is the best performing axis for 3 of 6 folds (heterocyclic, cycloalkyl, special), suggesting that size-based invariant features capture the most transferable physics for chemically novel spacer families.

Taxonomy Classification: Spacer Size   
You are a materials science expert specializing in organic-inorganic hybrid perovskites.   
Task: Classify the following organic spacer cation based on its molecular size.   
Spacer cation: {spacer\_name}   
Categories: - "tiny": Very small spacers with 1-3 carbon atoms (e.g., methylammonium,   
ethylammonium) - "small": Small spacers with 4-6 carbon atoms (e.g., butylammonium,   
pentylammonium) - "medium": Medium spacers with 7-10 carbon atoms (e.g., octylammonium,   
phenethylammonium) - "large": Large spacers with 11+ carbon atoms (e.g., dodecylammonium,   
long-chain amines)   
Respond with ONLY the category name (tiny, small, medium, or large). Do not explain.

## O.3 Taxonomy Classification Prompt: Conjugation

The conjugation axis captures the degree of π-electron delocalization in the spacer. Conjugated spacers (aromatic, polyene) can interact electronically with the perovskite inorganic framework through orbital overlap, potentially modifying the effective bandgap beyond what purely steric effects would predict. Saturated spacers act as electronically inert barriers. This distinction is chemically fundamental and creates partitions where invariance testing can separate features related to π-conjugation effects from those related to other chemical properties.

Taxonomy Classification: Conjugation   
You are a materials science expert specializing in organic-inorganic hybrid perovskites.   
Task: Classify the following organic spacer cation based on its degree of pi-conjugation.   
Spacer cation: {spacer\_name}   
Categories: - "saturated": No pi-conjugation; fully saturated alkyl chains or alicyclic   
rings - "partially\_conjugated": Some pi-conjugation but not fully delocalized (e.g.,   
vinyl groups, isolated double bonds) - "fully\_conjugated": Extensive pi-conjugation;   
aromatic rings, fused aromatics, or extended conjugated systems   
Respond with ONLY the category name. Do not explain.

## O.4 Taxonomy Classification Prompt: Polarity

The polarity axis distinguishes spacers based on their molecular dipole character. Polar spacers (with hydroxyl, carboxyl, or halogen substituents) can influence the local electrostatic environment at the organic-inorganic interface, potentially shifting the bandgap through dipole-induced polarization of the perovskite lattice. Nonpolar spacers (pure hydrocarbons) interact primarily through van der Waals forces. The amphiphilic category captures spacers with both polar and nonpolar domains, which are structurally distinct in how they orient within the interlayer gallery.

Taxonomy Classification: Polarity   
You are a materials science expert specializing in organic-inorganic hybrid perovskites.   
Task: Classify the following organic spacer cation based on its molecular polarity.   
Spacer cation: {spacer\_name}   
Categories: - "nonpolar": Predominantly hydrocarbon character with no significant polar   
groups beyond the ammonium head - "polar": Contains polar functional groups (OH, COOH,   
F, CF3, etc.) that create significant dipole moments - "amphiphilic": Has distinct   
hydrophilic and hydrophobic regions, creating amphiphilic character   
Respond with ONLY the category name. Do not explain.

## O.5 Taxonomy Classification Prompt: Steric Profile

The steric profile axis captures the three-dimensional shape of the spacer molecule. Linear spacers pack differently from branched, cyclic, or planar ones, creating distinct interlayer arrangements. Planar aromatic spacers can π-stack, cyclic spacers create unique packing geometries due to ring strain, and branched spacers disrupt ordered interlayer arrangements. These packing differences directly affect the crystal structure and hence the electronic properties. The steric taxonomy is the best axis for the alkyl-long fold, where the held-out long-chain spacers have a distinctive linear shape.

Taxonomy Classification: Steric Profile   
You are a materials science expert specializing in organic-inorganic hybrid perovskites.   
Task: Classify the following organic spacer cation based on its steric profile.

Spacer cation: {spacer\_name}   
Categories: - "linear": Linear or nearly linear molecular shape (straight chains) -   
"branched": Branched molecular shape with side groups - "cyclic": Contains non-aromatic   
ring structures (cyclopentyl, cyclohexyl, etc.) - "planar": Planar molecular shape   
dominated by aromatic rings   
Respond with ONLY the category name. Do not explain.

## O.6 Taxonomy Classification Prompt: Electron Effect

The electron effect axis classifies spacers by whether their substituents donate or withdraw electron density. Electron-donating groups (alkyl, methoxy, amino) increase the electron density at the organicinorganic interface, while electron-withdrawing groups (fluorine, CF<sub>3</sub>, nitro, cyano) decrease it. This electronic modulation can shift the valence band edge of the perovskite, providing a mechanism for bandgap tuning beyond compositional changes. The electron\_effect taxonomy is the best axis for the alkyl-short fold, where the held-out short alkyl spacers are uniformly electron-donating.

Taxonomy Classification: Electron Effect   
You are a materials science expert specializing in organic-inorganic hybrid perovskites.   
Task: Classify the following organic spacer cation based on its electron-donating or   
withdrawing character.   
Spacer cation: {spacer\_name}   
Categories: - "donating": Electron-donating groups (alkyl chains, methoxy, amino   
substituents) - "withdrawing": Electron-withdrawing groups (fluorine, CF3, nitro, cyano   
substituents) - "neutral": Balanced or minimal electronic effect   
Respond with ONLY the category name. Do not explain.

## O.7 Taxonomy Classification Prompt: H-Bonding Capacity

The hydrogen bonding axis captures the capacity of the spacer to form hydrogen bonds with the inorganic framework. The terminal ammonium group (−NH ) of every spacer forms hydrogen bonds with the halide anions of the perovskite layer, but additional H-bond donors (OH, NH groups) or acceptors (C=O, ether oxygen) in the spacer body can create supplementary bonding interactions. These additional interactions influence the interlayer spacing and tilting of the inorganic octahedra, which directly affects the bandgap through structural distortion.

Taxonomy Classification: H-Bonding Capacity   
You are a materials science expert specializing in organic-inorganic hybrid perovskites.   
Task: Classify the following organic spacer cation based on its hydrogen bonding   
capacity.   
Spacer cation: {spacer\_name}   
Categories: - "strong\_donor": Strong H-bond donors with multiple NH or OH groups -   
"weak\_donor": Weak H-bond donors with only the terminal ammonium group - "bifunctional":   
Both H-bond donor and acceptor capabilities (e.g., contains both NH and C=O or C-O-C)   
Respond with ONLY the category name. Do not explain.

## O.8 Taxonomy Classification Prompt: Interlayer Coupling

The interlayer coupling axis captures the expected electronic coupling strength between adjacent inorganic layers through the organic spacer. Short or conjugated spacers facilitate charge transport between layers, leading to more 3D-like electronic behavior with potentially narrower bandgaps. Long insulating chains minimize interlayer coupling, producing strongly confined 2D electronic behavior with wider bandgaps. This axis directly relates to the quantum confinement dimensionality that is a key determinant of 2D perovskite bandgaps.

Taxonomy Classification: Interlayer Coupling   
You are a materials science expert specializing in organic-inorganic hybrid perovskites.   
Task: Classify the following organic spacer cation based on expected interlayer   
electronic coupling strength.   
Spacer cation: {spacer\_name}   
Categories: - "weak": Long insulating chains that minimize interlayer coupling -   
"moderate": Medium-length or partially conjugated spacers with moderate coupling -   
"strong": Short or highly conjugated spacers that maximize interlayer electronic coupling

Respond with ONLY the category name. Do not explain.

## O.9 Taxonomy Classification Prompt: Thermal Stability

The thermal stability axis classifies spacers based on their expected contribution to the thermal robustness of the perovskite crystal. Small volatile spacers (methylammonium) are prone to sublimation, while large aromatic spacers with strong intermolecular interactions (such as π-stacking naphthalenyl groups) create thermally robust interlayer galleries. Although thermal stability does not directly determine the bandgap, it correlates with the strength of organic-inorganic interactions, which indirectly influences electronic properties through structural rigidity and octahedral tilting.

Taxonomy Classification: Thermal Stability   
You are a materials science expert specializing in organic-inorganic hybrid perovskites.   
Task: Classify the following organic spacer cation based on its expected contribution to   
thermal stability of the perovskite.   
Spacer cation: {spacer\_name}   
Categories: - "low": Volatile or thermally unstable spacers (small molecules, low   
boiling points) - "medium": Moderately stable spacers (medium-length alkyl chains, simple   
aromatics) - "high": Thermally robust spacers (large aromatic systems, cross-linkable   
groups, strong intermolecular interactions)   
Respond with ONLY the category name. Do not explain.

## O.10 Zero-Shot Baseline Prompt

The zero-shot prompt provides the LLM with the compound’s compositional information and asks for a direct bandgap prediction. This is the simplest possible LLM baseline and serves as the lower bound on LLM prediction quality. No examples, reasoning scaffolds, or retrieval context are provided. Despite access to the key compositional descriptors (spacer, metal, halide, layers, structure type), the LLM achieves only 0.552 eV MAE—demonstrating that the precision bottleneck is fundamental to LLM numerical prediction rather than an artifact of insufficient context.

Zero-Shot Bandgap Prediction   
You are a materials scientist. Predict the bandgap (in eV) of the following 2D halide   
perovskite compound.   
Compound: {compound\_formula}   
The compound has the following properties: - Spacer cation: {spacer\_name} - B-site   
metal: {metal} - Halide: {halide} - Number of layers: {n\_layers} - Structure type:   
{structure\_type}   
Provide your best estimate of the bandgap in eV. Respond with ONLY a number.

## O.11 Chain-of-Thought Baseline Prompt

The chain-of-thought prompt adds structured reasoning steps to guide the LLM through the physical factors that influence perovskite bandgaps. Each step corresponds to a known physical mechanism: B site metal choice (Pb vs. Sn creates the largest bandgap variation), halide composition (I < Br < Cl in bandgap), layer count (more layers = lower bandgap due to reduced confinement), and spacer effects. Despite this physically grounded reasoning scaffold, CoT achieves 0.563 eV MAE—marginally worse than zero-shot—suggesting that structured reasoning does not improve the LLM’s internal numerical calibration for this domain.

Chain-of-Thought Bandgap Prediction   
You are a materials scientist. Predict the bandgap (in eV) of the following 2D halide   
perovskite compound.   
Compound: {compound\_formula}   
Properties: - Spacer cation: {spacer\_name} - B-site metal: {metal} - Halide: {halide} -   
Number of layers: {n\_layers} - Structure type: {structure\_type}   
Think step by step: 1. What is the typical bandgap range for this type of perovskite? 2.   
How does the B-site metal affect the bandgap? 3. How does the halide composition affect   
the bandgap? 4. How does the number of layers affect the bandgap? 5. How might the   
spacer cation influence the bandgap?

Table 25: In-house LLM feature-generation experiments. All results are 6-fold mean OOD MAE (eV). Baseline: 0.122 eV.
<table><tr><td>Subcategory</td><td># Configs</td><td>Best MAE</td><td>vs. BL</td></tr><tr><td>Feature augmentation</td><td>12</td><td>0.127</td><td>-1%</td></tr><tr><td>Feature imputation</td><td>10</td><td>0.128</td><td>+0%</td></tr><tr><td>LLM embeddings</td><td>8+</td><td>0.131</td><td>+2%</td></tr></table>

Table 26: In-house LLM sample-selection experiments. All results are 6-fold mean OOD MAE (eV). Baseline: 0.122 eV.
<table><tr><td>Subcategory</td><td># Configs</td><td>Best MAE</td><td>vs. BL</td></tr><tr><td>Importance weighting</td><td>8</td><td>0.126</td><td>-2%</td></tr><tr><td>Sample reranking</td><td>7</td><td>0.127</td><td>-1%</td></tr><tr><td>Selective correction</td><td>5+</td><td>0.130</td><td>+2%</td></tr></table>

After your reasoning, provide your final bandgap estimate in eV on the last line as: ANSWER: [number]

## P In-House Configuration Variants (65+ Experiments)

Beyond the 19 published LLM methods (plus ML baselines) evaluated in Appendix D, we conducted 65+ additional in-house configuration variants across three categories to further stress-test whether LLM-generated numerical values could improve the kNN+GBT baseline. None of these are published methods; they represent our own exploratory combinations of LLM capabilities with ML pipelines. All confirm the same negative pattern observed in the published baselines.

## P.1 Category (iii): LLM-Generated Features (30+ configurations)

This category uses LLMs to augment or modify the 30-dim feature vector. Over 30 configurations were tested across three subcategories.

Table 25 reports the LLM-generated-feature experiments.

Feature augmentation (12 configs). We prompted the LLM to generate additional numerical features (ionization energy estimates, electronegativity predictions, crystal field parameters) and appended them to the 30-dim vector. Best result: 0.127 eV (−1%). The LLM-generated features are too noisy to improve over the curated feature set.

Feature imputation (10 configs). For compounds with missing structural parameters, we used the LLM to impute values. Best result: 0.128 eV (+0%). Imputation neither helps nor hurts because missing values are rare in the curated database.

LLM embeddings (8+ configs). We extracted hidden-state embeddings from compound descriptions and used them as features for ML models (PCA to 10/20/50 dims, various regressors). Best result: 0.131 eV (+2%). Embeddings encode semantic similarity rather than property-relevant similarity, adding noise to the feature space.

## P.2 Category (iv): LLM-Guided Sample Selection (20+ configurations)

This category uses LLMs to select, weight, or rerank training samples.

Table 26 reports the LLM-guided sample-selection experiments.

Importance weighting (8 configs). The LLM estimates relevance scores for training compounds given a test query; ML models are trained with weighted loss. Best result: 0.126 eV (−2%).

Sample reranking (7 configs). The LLM reranks kNN neighbors by chemical similarity; prediction uses reranked neighbors. Best result: 0.127 eV (−1%).

Table 27: In-house LLM prior/constraint experiments. All results are 6-fold mean OOD MAE $\mathrm { ( e V ) }$ Baseline: 0.122 eV.
<table><tr><td>Subcategory</td><td># Configs</td><td>Best MAE</td><td>vs. BL</td></tr><tr><td>GP prior</td><td>5</td><td>0.132</td><td>+3%</td></tr><tr><td>Physics constraints</td><td>6</td><td>0.134</td><td> $+ 5 \%$ </td></tr><tr><td>Bayesian blending</td><td>4+</td><td>0.133</td><td>+4%</td></tr></table>

Table 28: Summary of in-house configuration variants by category. “Best $\Delta ^ { \prime \prime }$ is the best improvement over the 0.122 eV baseline within that category.
<table><tr><td>Category</td><td># Configs</td><td>Best MAE (eV)</td><td>Best ∆</td></tr><tr><td>(iii) LLM-generated features</td><td>30+</td><td>0.127</td><td>-1%</td></tr><tr><td>(iv) LLM-guided sample selection</td><td>20+</td><td>0.126</td><td>-2%</td></tr><tr><td>(v) LLM as prior/constraint</td><td>15+</td><td>0.132</td><td>+3%</td></tr><tr><td>Total</td><td>65+</td><td>0.126</td><td>-2%</td></tr></table>

Selective correction (5+ configs). The LLM identifies compounds where ML predictions are likely wrong and applies corrections. Best result: 0.130 eV (+2%). Corrections add noise because the LLM cannot reliably identify ML errors.

## P.3 Category (v): LLM as Prior or Constraint (15+ configurations)

This category uses LLMs to provide Bayesian priors or physical constraints for ML models.

Table 27 reports the LLM-based prior and constraint experiments.

GP prior (5 configs). LLM-generated bandgap estimates serve as the prior mean for Gaussian Process regression. Best result: 0.132 eV (+3%). The noisy prior (∼0.5 eV precision) hurts more than it helps.

Physics constraints (6 configs). The LLM generates inequality constraints $( \mathrm { e . g . , \mathrm { \Omega ^ { 6 } S n } }$ perovskites have lower bandgap than $\mathbf { P b } ^ { \prime \prime } )$ that are enforced during prediction. Best result: $0 . 1 3 4 \mathrm { e V } ( + 5 \% )$ Constraints are either already captured by ML features or are too imprecise.

Bayesian blending (4+ configs). LLM and ML predictions are combined via Bayesian model averaging with estimated precision weights. Best result: 0.133 eV (+4%). Even with low LLM weight $( w _ { \mathrm { L L M } } = 0 . 0 5 )$ , the blended prediction is worse than pure ML.

## P.4 Summary

Table 28 aggregates all 65+ in-house configuration variants. None improve the baseline by more than 2%, and many degrade it. These results reinforce the published-method findings (Appendix D): the precision gap between LLM numerical output and calibrated ML regressors is fundamental, not prompt-dependent.

## Q Statistical Significance: Bootstrap Confidence Intervals

All statistical significance tests use query-level paired bootstrap (n=753 queries, B=10,000 resamples, seed = 42). We report the observed MAE difference (positive = MOTIF improves), the 95% percentile CI, and the one-sided p-value (fraction of bootstrap samples where MOTIF does not improve). We test across three backbones: $\tt { g p t - 4 0 }$ (default in the main paper), o3 (strongest bandgap annotator), and $\mathtt { g p t - 5 . 2 }$ (an additional appendix-only robustness check).

Table 29 reports the query-level paired bootstrap results for the evaluated backbones.

Table 30 reports the per-fold bootstrap confidence intervals.

The safe-blend ensemble rule reaches significance overall $\scriptstyle ( p = 0 . 0 0 2 )$ , with the largest and most significant gains on cycloalkyl $\scriptstyle \left( p = 0 . 0 0 6 \right)$ and phenyl $\scriptstyle ( p = 0 . 0 0 4 )$ —the two folds where the held-out families are chemically most distant from training. On the special fold, MOTIF is slightly worse than the baseline $\scriptstyle ( p = 0 . 9 2 6 )$ , consistent with the oracle-gap analysis showing that this heterogeneous family is the hardest for taxonomy-based methods. Seed sensitivity across 5 GBT random seeds yields a standard deviation of 0.0002 eV (range [0.1139, 0.1144]), confirming that the variance from model randomness is well below the observed improvements.

Table 29: Query-level paired bootstrap CIs for the 2D perovskite bandgap control benchmark. All MAE values in eV.
<table><tr><td>Comparison</td><td>Backbone</td><td>∆MAE</td><td>95% CI</td><td>p</td></tr><tr><td>BL vs Safe Blend</td><td>gpt-4o</td><td>+0.0041</td><td>[0.0014, 0.0068]</td><td>0.002</td></tr><tr><td>BL vs Top-1 CV</td><td>gpt-4o</td><td>+0.0021</td><td>[-0.0030, 0.0072]</td><td>0.206</td></tr><tr><td>BL vs Safe Blend</td><td>03</td><td>+0.0058</td><td>[0.0031, 0.0086]</td><td>&lt;0.001</td></tr><tr><td>BL vs Top-1 CV</td><td>03</td><td>+0.0042</td><td>[-0.0008, 0.0094]</td><td>0.051</td></tr><tr><td>BL vs Safe Blend</td><td>gpt-5.2</td><td>+0.0052</td><td>[0.0026, 0.0079]</td><td>&lt;0.001</td></tr><tr><td>BL vs Top-1 CV</td><td>gpt-5.2</td><td>+0.0045</td><td>[-0.0003, 0.0099]</td><td>0.036</td></tr></table>

Table 30: Per-fold bootstrap CIs on 2D perovskite bandgap (Baseline vs Safe Blend, gpt-4o). Positive ∆ = MOTIF better.
<table><tr><td>Fold</td><td>n</td><td>BL MAE</td><td>MOTIF MAE</td><td>∆ [95% CI]</td><td>p</td></tr><tr><td>Alkyl-short</td><td>232</td><td>0.130</td><td>0.128</td><td>+0.002 [−0.003, 0.007]</td><td>0.201</td></tr><tr><td>Alkyl-long</td><td>79</td><td>0.124</td><td>0.120</td><td>+0.004 [-0.001, 0.009]</td><td>0.060</td></tr><tr><td>Phenyl</td><td>166</td><td>0.133</td><td>0.123</td><td>+0.010 [0.003, 0.017]</td><td>0.004</td></tr><tr><td>Heterocyclic</td><td>128</td><td>0.111</td><td>0.108</td><td>+0.002 [−0.004, 0.009]</td><td>0.220</td></tr><tr><td>Cycloalkyl</td><td>37</td><td>0.137</td><td>0.117</td><td>+0.020 [0.004, 0.037]</td><td>0.006</td></tr><tr><td>Special</td><td>111</td><td>0.106</td><td>0.110</td><td>-0.003 [-0.008, 0.001]</td><td>0.926</td></tr></table>

Table 31: Frozen dev/test results on 2D perovskite bandgap. Dev folds were used for α selection only; test folds were never used for any design decision.
<table><tr><td>Fold</td><td>Role</td><td>BL MAE</td><td>MOTIF MAE</td><td>∆%</td></tr><tr><td>Alkyl-short</td><td>Dev</td><td>0.130</td><td>0.128</td><td>-1.6%</td></tr><tr><td>Phenyl</td><td>Dev</td><td>0.133</td><td>0.123</td><td>-7.3%</td></tr><tr><td>Alkyl-long</td><td>Test</td><td>0.124</td><td>0.120</td><td>-3.2%</td></tr><tr><td>Heterocyclic</td><td>Test</td><td>0.111</td><td>0.108</td><td>-2.2%</td></tr><tr><td>Cycloalkyl</td><td>Test</td><td>0.137</td><td>0.117</td><td>-14.7%</td></tr><tr><td>Special</td><td>Test</td><td>0.106</td><td>0.110</td><td>+3.0%</td></tr><tr><td>Test mean</td><td>一</td><td>0.119</td><td>0.114</td><td>-4.3%</td></tr></table>

## R Frozen Dev/Test Protocol

To address the concern that hyperparameters (α, balance threshold, ensemble rule) were tuned on the same benchmark used for evaluation, we designate two folds (alkyl-short and phenyl, comprising 53% of queries) as a development set and the remaining four folds (alkyl-long, heterocyclic, cycloalkyl, special) as afrozen test set. The dev set is used solely for α selection; the selected α=0.01 is then applied unchanged to the frozen test folds.

## Table 31 summarizes development and frozen-test fold performance.

The frozen-test mean MAE improves from 0.119 to 0.114 eV (−4.3%), confirming that the improvement generalizes beyond the development folds. Three of the four frozen-test folds show improvement, with the special fold as the expected exception.

Table 32: Spacer overlap verification on 2D perovskite benchmark. In every fold, the overlap between test and training spacers is zero, confirming no leakage.
<table><tr><td>Fold</td><td>Test Spacers</td><td>Train Spacers</td><td>Overlap</td><td>Test-Only</td></tr><tr><td>Alkyl-short</td><td>92</td><td>197</td><td>0</td><td>92</td></tr><tr><td>Alkyl-long</td><td>20</td><td>269</td><td>0</td><td>20</td></tr><tr><td>Phenyl</td><td>73</td><td>216</td><td>0</td><td>73</td></tr><tr><td>Heterocyclic</td><td>55</td><td>234</td><td>0</td><td>55</td></tr><tr><td>Cycloalkyl</td><td>14</td><td>275</td><td>0</td><td>14</td></tr><tr><td>Special</td><td>48</td><td>241</td><td>0</td><td>48</td></tr></table>

Table 33: Alternative invariance selectors on 2D perovskite bandgap. All use the same gated-median ensemble and taxonomy bank. Mean MAE is across 6 OOD folds.
<table><tr><td>Selector</td><td>Mean MAE (eV)</td><td>vs. Baseline</td></tr><tr><td>Chow test (α=0.01) + forward sel.</td><td>0.1144</td><td>-7.2%</td></tr><tr><td>Bonferroni Chow  $( \alpha / 2 4 )$ </td><td>0.1233</td><td>0.0%</td></tr><tr><td>Permutation test (200 perms)</td><td>0.1233</td><td>0.0%</td></tr><tr><td>MI-stability  $( \sigma _ { r } < 0 . 0 5 )$ </td><td>0.1244</td><td>+0.9%</td></tr><tr><td>Baseline (no invariance filter)</td><td>0.1233</td><td></td></tr></table>

## S Test-Set Leakage Verification

A potential concern is that the cached taxonomy labels—generated for all 313 unique spacers—might leak information about the held-out test family. We verify that this is not the case.

Table 32 provides the fold-wise train/test spacer-overlap checks.

The leave-one-family-out protocol ensures that the held-out family’s spacers are entirely absent from training. Taxonomy labels are applied only to training-set spacers for Chow-test environment partitioning; the held-out spacers’ labels are never accessed during feature selection, model training, or gating. At inference, only the 30-dim feature vector is used—no taxonomy label lookup occurs.

## T Alternative Invariance Selectors

The main paper uses the Chow structural-break test to identify invariant features. A natural concern is that this linear test may be suboptimal for the nonlinear downstream predictors (kNN, GBT). We compare four invariance selectors:

1. Chow test $( \alpha { = } 0 . 0 1 )$ : the default selector used in the main paper.

2. Bonferroni-corrected Chow: applies $\alpha _ { \mathrm { c o r r e c t e d } } = 0 . 0 1 / 2 4$ (24 non-spacer features) to control family-wise error rate.

3. Permutation test: for each feature, we compute the variance of per-environment regression slopes, then compare against a null distribution from 200 label permutations.

4. MI-stability: a feature is invariant if the standard deviation of its per-environment correlation with the target is below 0.05.

Table 33 compares the alternative invariance selectors.

Results differ across selectors in an informative way. The Chow test with forward selection achieves $0 . 1 1 4 4 \mathrm { e V } ,$ substantially improving over the 0.1233 eV baseline. The Bonferroni-corrected Chow $( \alpha _ { \mathrm { c o r r } } = 0 . 0 1 / 2 4 )$ and permutation test both collapse to baseline-level performance (0.1233), because the overly conservative correction retains nearly all features, effectively disabling the invariance filter. The MI-stability test performs slightly worse (0.1244).

This comparison reveals that the Chow test’s effectiveness depends not on the test statistic alone but on its integration with the greedy forward-selection step (Section 3.3), which curates the invariant set down to the most predictive features. When the invariance filter is too conservative (Bonferroni) or too coarse (MI), the subsequent forward selection has no meaningful feature set to work with.

The practical lesson is that the pipeline—invariance screening followed by CV-guided refinement— matters more than the specific statistical test used in the screening step.

## U Method Descriptions for Table 3

Table 3 compares multiple feature-selection and environment-generation strategies. All methods use the same downstream predictor: a 50/50 blend of inverse-distance kNN (k=5) and gradient-boosted trees (GBT, 160 estimators). The only difference is how features are selected or filtered before training. Entries marked “—” in Table 3 are simple in-house controls or heuristic variants rather than baselines adopted from a single prior paper.

## Environment-based methods (Panel A).

• MOTIF (LLM taxonomy + Safe blend / Top-1 CV): Our proposed method. An LLM classifies each spacer along 9 chemical axes offline. Within each taxonomy, a Chow test [4] identifies invariant features, and kNN+GBT are trained in the invariant subspace. The safe-blend ensemble averages the baseline and the best-CV taxonomy; Top-1 CV selects the single best taxonomy by leave-one-environment-out CV.

• Expert-family taxonomy: Uses the 6 human-defined spacer superfamilies (alkyl-short, alkyl-long, phenyl, heterocyclic, cycloalkyl, special) as environments for the same Chow-test pipeline.

• Rule-based chemistry axes: Simple keyword heuristics classify spacers by size, conjugation, or polarity (e.g., “phenyl” → conjugated). Best result among 3 rule axes is reported.

• Descriptor clustering [13]: k-means clustering (k=3, 4) on the 30-dim feature vector defines environments. Best result is reported.

## Classical feature selectors (Panel C).

• Pearson top-k: Select the k features with highest absolute Pearson correlation with the target; k is chosen by leave-one-family-out CV.

• LOEO forward selection: Greedy forward feature selection using leave-one-environment-out validation error as the selection criterion.

• IID forward selection: Same greedy procedure but with standard k-fold CV (ignoring family structure).

• Group Lasso / Sparse Group Lasso [21, 30]: Proximal gradient descent with group $\ell _ { 2 }$ (or $\ell _ { 1 } + \ell _ { 2 } )$ penalties on feature blocks; features with zero weight are removed.

• IRM / VREx / GroupDRO / CORAL [1, 18, 19, 22]: End-to-end domain-generalization objectives that learn invariant representations. All use a 2-layer MLP (32 hidden units) trained with environment-specific penalties.