# Prevalence Determines Precision: Silent Contamination in Detector-Defined Datasets

Jia Huang¹, Yankai Wan2,Yangjun Ou³

1Guanghua School of Management, Peking University   
2College of Artificial Intelligence, Jilin University   
3School of Mathematical Sciences, Peking University

## Abstract

A large share of machine learning datasets are not observed but constructed: a detector, heuristic, or model is run over a pool of candidates, and whatever it accepts becomes the dataset. Weak and distant supervision, pseudo-labelling, event extraction, and most anomaly-detection benchmarks all have this form. The precision of the resulting dataset is not a property of the detector. It is governed by the prevalence π of true positives in the pool the detector is deployed on, through elementary Bayes. This is textbook, and it is nonetheless almost never measured end-to-end, for a structural reason: from inside a detector-defined dataset the false positives are unidentifiable, so the quantity that matters cannot be estimated by the people who need it. We report a case in which it can be measured. For the same instrument and period we hold both a detector-defined event dataset and an independent official index that reveals, for every detected item, whether it is real. One detector, three candidate pools, three datasets: phantom rates of 81.7%, 9.0%, and 0.0%. A practitioner transferring precision from the two high-π pools to the low-π pool predicts 0.955 against a measured 0.183, an error of +422%; the Bayes expression predicts within 3.3% across all three pools. Beyond confirming the mechanism we report three findings that we believe are new. First, the detected dataset's response curve is an exact convex combination of a true-event and a phantom component (identity residual $1 . 1 \times 1 0 ^ { - 1 6 } )$ , with phantoms outnumbering true events 473 to 308 — contamination is not noise added to signal but a second signal with its own shape, inherited from the detector's acceptance rule. Second, the direction of contamination is a property of the estimator, not of the data: on identical windows, one statistic shows contamination diluting the effect and another shows it inflating the effect, because the second statistic's denominator is itself contaminated. Third, a normalisation in common use turns the estimator into a mean of ratios whose expectation does not exist; on the same 335 events it returns 0.40 where the well-defined estimator returns 0.10.

## 1 Introduction

Consider how a dataset gets built when nobody is willing to label $1 0 ^ { 6 }$ items by hand. A rule is written — a regular expression, a knowledge-base join, a threshold on a model's confidence, a statistical scan — and applied to a pool of candidates. What the rule accepts becomes the dataset. Downstream, the accepted items are treated as observations. This is the shape of distant supervision (Mintz et al., 2009), of weak supervision (Ratner et al., 2017), of pseudo-labelling, and of the great majority of event-detection and anomaly benchmarks.

The precision of such a dataset obeys

$$
\mathrm { p r e c i s i o n } = { \frac { \pi \mathrm { T P R } } { \pi \mathrm { T P R } + ( 1 - \pi ) \mathrm { F P R } } } ,\tag{1}
$$

where π is the fraction of true positives in the pool the detector was run on. Precision is therefore a property of the deployment pool. What transfers across pools is the pair (TPR, FPR); precision does not. A detector validated where π ≈ 1 — and at π ≈ 1 even a badly broken detector looks flawless — and then redeployed where $\pi \ll 1$ imports false positives at a rate its validation never measured.

None of this is new. Bar-Hillel (1980) named the reasoning error; Hand (2009) and Saito & Rehmsmeier (2015) state the consequence for classifier evaluation directly. What is missing is measurement. The obstacle is structural rather than sociological: inside a detector-defined dataset the false positives are exactly the items one cannot identify If they could be identified, the detector would not have accepted them. Practitioners therefore validate on whatever labelled subset they have — which is, almost by construction, a high-π subset — and deploy elsewhere.

This paper. We report a setting where the phantoms are identifiable. For E-mini S&P 500 futures over 2010–2026 we hold two independent event indices over the same price series: a detector-defined set produced by a magnitude-threshold scan, and an official index (CPI, non-farm payrolls, and FOMC statement releases) published by the institutions that generate the events. Every detected item can be classified. Moreover the detector was run on three structurally different candidate pools — the Federal Reserve's own meeting calendar, the set of first Fridays, and business days 6–16 of each month — giving three deployments of one detector at π ∈ {1.00, 0.82, 0.10} within a single dataset.

We use this to do three things a synthetic study cannot. We measure (1) against reality rather than assuming it. We decompose a real contaminated estimate into its true and phantom components exactly, and characterise what the phantom component looks like. And we show that the contamination's effect on a downstream scientific conclusion is not even sign-stable across reasonable choices of estimator

## Contributions.

1. An end-to-end measurement of prevalence-driven contamination with recoverable ground truth: one detector, three pools, phantom rates 81.7% / 9.0% / 0.0%; naive cross-pool precision transfer errs by +422% while (1) holds to 3.3% (§5.1–§5.2).

2. An exact mixture decomposition showing that a contaminated estimate is a convex combination of a true-event and a phantom estimate, verified to machine precision on real data, with the phantom component reproducible from the detector's acceptance rule applied to pure noise (§5.3).

3. Estimator-dependence of the contamination direction. On identical windows, contamination dilutes under a denominator-free statistic and inflates under a terminally normalised one. We trace this to contamination of the estimator's own scale (§5.4).

4. A non-integrable estimator in common use. Per-item normalisation by a signed reference return yields a mean of ratios with no finite expectation. Empirically it returns 0.40 where the equal-weight estimator returns 0.10 on the same events (§5.5).

5. A seven-rule protocol (§7) and a six-accident appendix (App. A) documenting the mechanism operating on this paper's own pipeline.

## 2 Setup

Observed and detected indices. Let $p _ { t }$ be a series and let $T ^ { \mathrm { o b s } }$ be an observed index published by an external authority. Let A be a detector: any deterministic rule mapping a candidate window to accept/reject. The candidate pool C is the set of windows at which A is evaluated. The detected index is $T ^ { \mathrm { d e t } } = A ( C )$ . Users of $T ^ { \mathrm { d e t } }$ typically see only $\vert T ^ { \mathrm { d e t } } \vert$

Five quantities. We argue five must be reported alongside any detector-defined dataset:

$$
{ \begin{array} { r l r l } { \pi = { \frac { | T ^ { \mathrm { o b s } } \cap C | } { | C | } } } & { \qquad } & & { { \mathrm { p r e v a l e n c e ~ i n ~ t h e ~ d e p l o y m e n t ~ p o o l } } } \\ { \mathrm { T P R } = { \frac { | T ^ { \mathrm { o b s } } \cap T ^ { \mathrm { o t e t } } | } { | T ^ { \mathrm { o b s } } \cap C | } } , } & { \qquad { \mathrm { F P R } } = { \frac { | T ^ { \mathrm { d e t } } \setminus T ^ { \mathrm { o b s } } | } { | C \setminus T ^ { \mathrm { o b s } } | } } } & & { \qquad { \mathrm { d e t e c t o r ~ o p e r a t i n g ~ p o i n t } } } \\ { c = { \frac { n p } { n _ { E } + n p } } } & & { \qquad } & & { \qquad { \mathrm { p h a n t o m ~ m a s s ~ f r a c t i o n } } } \\ { \rho = w _ { P } / w _ { E } } & & { \qquad } & & { \qquad { \mathrm { p h a n t o m . t o - t u e ~ i n t e n s i t y ~ r a t i o n } } } \end{array} }
$$

where $n _ { E } = \vert T ^ { \mathrm { o b s } } \cap T ^ { \mathrm { d e t } } \vert , n _ { P } = \vert T ^ { \mathrm { d e t } } \langle T ^ { \mathrm { o b s } } \vert$ , and $w _ { E } , w _ { P }$ are the mean magnitudes of the two subsets under whatever scale the downstream estimator uses. Precision equals $1 - c ,$ and by Bayes it equals (1).

The last two are the ones normally omitted even by careful authors. c says how much of the dataset is fabricated; $\rho$ says how loud the fabricated part is relative to the real part. §5.3 shows they jointly determine the damage, and that c alone does not.

Response curves. For an index T with anchor sign $s _ { i } = \mathrm { s i g n } \ r _ { i } ( \tau _ { \mathrm { r e f } } )$ , define

$$
S ( \tau ) = \frac { 1 } { n } \sum _ { i } s _ { i } r _ { i } ( \tau ) , \qquad D _ { \mathrm { r e f } } ( \tau ) = \frac { S ( \tau ) } { \frac { 1 } { n } \sum _ { i } | r _ { i } ( \tau _ { \mathrm { r e f } } ) | } , \qquad \widetilde { D } ( \tau ) = \frac { 1 } { n } \sum _ { i } \frac { r _ { i } ( \tau ) } { r _ { i } ( \tau _ { \mathrm { r e f } } ) } ,\tag{2}
$$

where $r _ { i } ( \tau )$ is the log return from $t _ { i }$ to $t _ { i } + \tau , S$ is denominator-free and reported in basis points. D normalises by a pooled scale. $\widetilde { D }$ normalises per item; it looks like a natural way to make heterogeneous events comparable, and §5.5 shows it is not an estimator at all. We use $\tau _ { \mathrm { r e f } } = 1 8 0 0 \mathrm { s }$ throughout the main text and report the legacy pipeline's $\tau _ { \mathrm { r e f } } = 7 2 0 0$ s convention in Appendix D.

## 3 Theory

## 3.1 Precision does not transfer; (TPR, FPR) does

Calibrating a detector means choosing a threshold, which fixes an operating point (TPR, FPR). Those are conditional on the signal and noise distributions, not on π. Precision is then determined by (1) in any pool where π is known. The failure mode is not subtle once stated, but note what makes it invisible in practice: at $\pi \approx 1$ , precision is near 1 for any FPR, so a validation at high π carries essentially no information about FPR — the very quantity that governs behaviour at low π. High-prevalence validation is not weak evidence about low-prevalence deployment; it is close to no evidence.

## 3.2 The mixture identity

Let E, P partition a detected index into true events and phantoms, with intensities $w _ { E } , w _ { P }$ under the estimator's own scale. Because the anchor sign is defined identically on both subsets, for every τ

$$
D ^ { \mathrm { d e t } } ( \tau ) ~ = ~ \frac { ( 1 - c ) w _ { E } D _ { E } ( \tau ) ~ + ~ c w _ { P } D _ { P } ( \tau ) } { ( 1 - c ) w _ { E } + c w _ { P } } ~ = ~ \frac { ( 1 - c ) D _ { E } ( \tau ) + c \rho D _ { P } ( \tau ) } { ( 1 - c ) + c \rho } .\tag{3}
$$

This is an identity, not a model: it holds for any detector, any $\tau ,$ any data. Its content is diagnostic. A dataset owner who knows $( c , \rho , D _ { E } , D _ { P } )$ knows exactly what their published curve measures. A user holding only $D ^ { \mathrm { d e t } }$ cannot invert it which is precisely the difficulty: contamination is not visible from inside the contaminated dataset.

Two consequences. First, contamination is not additive noise. $D _ { P }$ has a shape, and when the detector selects on the same quantity that defines the phenomenon, that shape resembles the phenomenon. Second, the weights involve $\rho ,$ so a dataset with high precision but loud phantoms can be more damaged than one with low precision and quiet phantoms.

## 3.3 The phantom null is a selection-conditioned null

Phantoms are, by construction, windows accepted by the rule. Under a pure-noise null, conditioning on a large move at the acceptance horizon induces structure in any statistic anchored near that horizon — not because the noise contains an event, but because the conditioning event is itself part of the statistic. The correct null for $D _ { P }$ is therefore not"flat" and not the unconditional random walk, but the detector's transfer function: noise passed through the same acceptance rule and aggregated by the same estimator. We construct this null in §5.3.

## 3.4 A normalisation with no finite expectation

Proposition 1. Let $r _ { \mathrm { r e f } }$ have a density f that is continuous and strictly positive in a neighbourhood $o f 0 .$ Then $\mathbb { E } \big [ | r ( \tau ) / r \mathrm { r e f } \big | \big ] = \infty ,$ and the estimator $\begin{array} { r } { \widetilde { D } ( \tau ) = \frac { 1 } { n } \sum _ { i } r _ { i } ( \tau ) / r _ { i } ( \tau _ { \mathrm { r e f } } ) } \end{array}$ of (2) has no inite expectation.

Proof sketch. $\begin{array} { r } { \mathbb { E } | 1 / r _ { \mathrm { r e f } } | = \int f ( x ) / | x | } \end{array}$ dx diverges logarithmically at the origin whenever $f ( 0 ) > 0$ . Conditional on $r ( \tau )$ having non-degenerate conditional mean near $r _ { \mathrm { r e f } } = 0$ , the same divergence carries to the ratio. □

The sample mean of $\widetilde { D }$ is finite for any finite $n ,$ so the pathology is silent: one obtains a number, it looks stable across reruns on the same data, and it does not converge to anything as n grows. It is dominated by the items with the smallest $| r _ { \mathrm { r e f } } | longrightarrow \mathrm { w h i c h }$ , in an event dataset, are the items where nothing happened. §5.5 measures the size of the resulting artefact.

## 4 A dataset with recoverable phantoms

Price series. E-mini S&P 500 futures (front-month continuous, ES . c . 0), best-bid-offer at 1-second resolution from a CME-derived vendor feed, in windows of $t _ { 0 } \pm 2 \mathrm { h }$ . Total spend \$2.04.

Official index. 555 announcement windows over 2010–2026: 212 CPI and 203 non-farm payroll releases (release dates from the Federal Reserve Bank of St. Louis release calendar; release times from the statutory 08:30 ET convention), and 140 FOMC statements (dates and per-statement release times verified against the Federal Reserve's own press-release pages; 14:15 ET before March 2013, 14:00 ET after, with seven unscheduled statements verified individually).

Detected index. A magnitude-threshold scan of the same price series: a window is accepted when the 2-minute move exceeds ten times a per-second standard deviation estimated on the pre-window. Full specification in Appendix E. The scan was run separately on three candidate pools, which is what makes this dataset useful:
<table><tr><td>Pool</td><td>Definition</td><td>|C|</td></tr><tr><td>FOMC</td><td>the Federal Reserve&#x27;s published meeting calendar</td><td>140</td></tr><tr><td>NFP</td><td>the first Friday of each month</td><td>200</td></tr><tr><td>CPI</td><td>business days 6–16 of each month</td><td>1569</td></tr></table>

Screens. Applied identically to every label set before any comparison: a pre-window futures-roll screen (any 1- second log return above 20 bp strictly before to; post-to jumps are deliberately excluded so the screen cannot delete announcements — see App. A, Accident 4) and a degraded-quote screen. 415 official windows survive for the terminalanchor analysis and 417 for the 30-minute-anchor analysis, all from \~2013 onward; the vendor's 1-second book history does not extend reliably before then.

Why this is a good testbed and where it is limited. The two indices are generated by causally independent processes: one by statistical agencies and a central bank, the other by a threshold on prices. Neither was constructed with the other in view. That independence is what makes phantom identification credible. The limitations are equally clear and we state them here rather than only in §8: one instrument, one detector family, one class of official index.

## 5 Results

## 5.1 One detector, three pools, three datasets

Table 1 is the paper's central measurement. Three observations.

The spread is the finding. One detector, one price series, one implementation, three pools — and three qualitatively different datasets. On the Fed's calendar the detector is perfect. On business days 6–16, four out of every five accepted items never happened.

Aggregation conceals it. The pooled phantom rate is 59%, which reads as a mediocre-but-usable detector. Per-pool, one of the three datasets is majority-synthetic. A practitioner who reports a single precision figure for a detector deployed across heterogeneous pools has reported a number that describes none of the deployments.

Table 1: The same detector deployed on three candidate pools. “Phantom $\mathrm { r a t e } ^ { \prime \prime }$ is $n _ { P } / n _ { \mathrm { l e g a c y } }$ , i.e. one minus the precision of the detector in that pool. Counts are raw detections before quality screens; the screened counts used in §5.3 are $n _ { E } = 3 0 8 , n _ { P } = 4 7 3 .$
<table><tr><td>Pool</td><td>|C|</td><td>π</td><td> $n _ { \mathrm { o f f i c i a l } }$ </td><td> $n _ { \mathrm { l e g a c y } }$ </td><td> $n _ { E }$ </td><td> $n _ { P }$ </td><td>phantom rate</td></tr><tr><td>FOMC</td><td>140</td><td>1.000</td><td>140</td><td>126</td><td>126</td><td>0</td><td>0.0%</td></tr><tr><td>NFP</td><td>200</td><td>0.820</td><td>203</td><td>122</td><td>111</td><td>11</td><td>9.0%</td></tr><tr><td>CPI</td><td>1569</td><td>0.105</td><td>212</td><td>596</td><td>109</td><td>487</td><td>81.7%</td></tr><tr><td>All</td><td>1909</td><td>一</td><td>555</td><td>844</td><td>346</td><td>498</td><td>59.0%</td></tr></table>

Phantoms dominate in count. On the screened sample, $n _ { P } = 4 7 3$ against $n _ { E } = 3 0 8 \colon c = 0 . 6 0 6$ . A majority of the published event list consists of events that did not occur

## 5.2 Cross-pool transfer: naive $+ 4 2 2 \%$ , Bayes -3.3%

The three pools give a natural calibration/deployment split. Calibrate on the two high-prevalence pools (FOMC, $\pi = 1 . 0 0 ; \mathrm { N F P } , \pi = 0 . 8 2 )$ , deploy on the low-prevalence one (CPI, $\pi = 0 . 1 0 5 )$ •

A practitioner who treats precision as a detector property transfers the mean of the observed precisions, ${ \frac { 1 } { 2 } } ( 1 . 0 0 0 +$ $0 . 9 1 0 ) = 0 . 9 5 5$ . The measured CPI precision is 0.183: an error of +422%. The Bayes expression (1), using each pool's own π, that pool's TPR, and the detector's analytic FPR = 0.361 (derived in App. A, Accident 1, and confirmed by Monte Carlo to within 2%), predicts 0.177 against 0.183 measured: an error of —3.3%. The corresponding errors are —1.6% for NFP and 0% for FOMC. Figure 1 shows all three.

![](images/fbe3077f36ba22e5657fb18eeca29c85320519b27fcf7f452ea008027e3d976b.jpg)  
Figure 1: Precision as a function of pool prevalence. Solid and dashed black: (1) at the two measured TPR values, with FPR = 0.361 fixed. Dotted grey: the naive belief that precision is a property of the detector. Three measured deployments of one detector fall on the curve. The red marker is the naive transfer from the two high-π pools to the low-π pool; the dashed arrow is the resulting error.

We stress what is and is not being claimed. (1) is elementary and we claim no novelty for it. The claim is that its consequences are large, are realised in a real dataset built by competent people, and are not detectable from inside that dataset. The naive transfer is not a straw man: it is what "we validated the detector and got 95% precision"means when the validation pool and the deployment pool differ in prevalence by an order of magnitude.

## 5.3 Mixture decomposition: the contaminated curve, exactly

Applying (3) to the screened sample:

$$
n _ { E } = 3 0 8 , \quad n _ { P } = 4 7 3 , \quad c = 0 . 6 0 6 , \quad w _ { E } = 6 . 3 8 \mathrm { b p } , \quad w _ { P } = 3 . 7 6 \mathrm { b p } , \quad \rho = 0 . 5 9 0 ,
$$

and the identity check

$$
\operatorname* { m a x } _ { \tau } \left| \cal D _ { \mathrm { p r e d } } ^ { \mathrm { d e t } } ( \tau ) - \cal D _ { \mathrm { o b s } } ^ { \mathrm { d e t } } ( \tau ) \right| = 1 . 1 \times 1 0 ^ { - 1 6 } ,
$$

i.e. exact to machine precision (Figure 2, right). This is first a verification that the bookkeeping is sound, and second a statement with content: the published curve of the detected dataset contains exactly the information in (3) and nothing else. There is no residual to interpret

![](images/8c396f7c856514d4fdf9a00bf1ffe06d5621244a5d08a814efcb55dd4ab005e6.jpg)

![](images/d89c184a4a7463d8dca80e66a269461673571248f8f5723305b0c322780940d5.jpg)  
Figure 2: Left: the detected dataset's curve decomposed into its true-event component $D _ { E } .$ , its phantom component $D _ { P }$ , and the exact mixture, with the unconditional random-walk baseline and the selection-conditioned null for $D _ { P }$ Right: identity (3) verified to machine precision.

What the phantom component is. $D _ { P }$ is not flat and not the unconditional random walk. It rises with τ, qualitatively like $D _ { E }$ . Compared against the selection-conditioned null of $\ S 3$ — pure noise passed through the detector's exact acceptance rule and the same estimator — it is reproduced in shape but not in level (RMSE 0.092 over $\tau \leq 6 0 0 s$ against $D _ { P }$ values in the range 0.02–0.07; the null sits systematically low). We therefore state the weaker claim that the data support: the phantom component's shape is generated by the acceptance rule rather than by event content, while its level is not fully accounted for by our null. We regard the residual as an open question rather than evidence of event content in the phantoms.

Why ρ matters. Here $\rho = 0 . 5 9 \mathrm { : }$ phantoms are quieter than true events, so their mixture weight is below their count share. A dataset with the same c but $\rho > 1$ would be substantially more damaged. Reporting precision alone $( = 1 - c )$ does not distinguish these cases.

## 5.4 The direction of contamination is a property of the estimator

This is the finding we expect to generalise furthest beyond the domain.

Under the terminally normalised statistic used by the original pipeline, contamination inflates: at $\tau = 5 \mathrm { s } .$ legacy 0.082 versus official 0.068; at 300 s, 0.317 versus 0.271. The contaminated dataset looks like it has more structure than the truth. Under the denominator-free statistic $S ( \tau )$ on the same windows, the ordering reverses: official 0.79 bp > legacy 0.48 > phantom-only 0.18 at $\tau = 5 \mathrm { s } .$ and $2 . 5 3 > 1 . 6 5 > 0 . 8 3$ at 300 s. Contamination dilutes. Figure 3 shows both.

![](images/eeb82343a813d5f86b591f382286b7a0d29a18ce91774510c0968cf26d9f4c95.jpg)

![](images/cded128f1c6a80748cabcd3c17e45a0dd5e7dd614137f20e36581314fc33c06d.jpg)  
Figure 3: Same price windows, two label sets, two estimators. Left: denominator-free signed displacement $S ( \tau )$ in bp. Right: normalised $D _ { 3 0 } ( \tau )$ with its random-walk baseline $\tau / 1 8 0 0$ . Official $n { = } 4 1 7 .$ , detected $n { = } 7 7 2$ , phantom-only $n { = } 4 6 3$ . The revision-only control is plotted at $n { = } 3$ and is not interpretable; only 3 of the 17 calendar-identified control windows have price coverage $( { \mathrm { A p p . } } { \mathrm { C } } )$ . Under terminal normalisation $( \mathrm { A p p . D } )$ the official/detected ordering is reversed relative to the left panel.

The mechanism is (3) together with the denominator. S has no denominator, so the mixture simply pulls the true curve toward the phantom curve: dilution, as intuition expects. $D ^ { \mathrm { t e r m } }$ divides by a scale estimated on the same pooled windows. Phantoms enter both numerator and denominator, and because $\rho < 1$ they depress the denominator proportionally more than the numerator at short horizons, inflating the ratio. The estimator's own scale is contaminated.

The consequence for practice is sharper than the mechanism. A user holding only a contaminated dataset cannot determine which regime they are in, because the diagnosis requires $\rho ,$ which requires knowing which items are phantoms. Statements of the form "contamination will attenuate my effect, so my estimate is conservative" — a very common defence in applied work — are not available. Attenuation is one of two possible directions and which one occurs depends on a quantity the analyst cannot measure.

## 5.5 A normalisation that returns 0.40 where the estimator returns 0.10

The pathology of §3.4 is not hypothetical; we found it in our own pipeline, and it accounts for a factor-of-four discrepancy that took two rounds to resolve. On an identical set of 335 events, the two aggregations of (2) give

$$
{ \widetilde { D } } ( 5 { \mathrm { s } } ) = 0 . 3 9 9 7 \qquad { \mathrm { v e r s u s } } \qquad D ( 5 { \mathrm { s } } ) = 0 . 1 0 2 0 .
$$

Splitting by the magnitude of the reference return makes the source visible: on quiet windows $( | r _ { \mathrm { r e f } } |$ below the sample median) $\widetilde { D } ( 5 \mathrm { s } ) = 0 . 4 1 5 ;$ on active windows, 0.097. The quiet windows, where by construction nothing happened and the post-event path is uncorrelated with the anchor sign, dominate the average because they carry the largest $1 / | r _ { \mathrm { r e f } } |$ weights.

Per-item normalisation is an appealing move — it is how one makes heterogeneous events comparable — and its failure is silent, because the estimator returns a plausible number that is stable across reruns. We report it because we suspect it is not rare: any statistic of the form "mean of per-item normalised response" with a signed, mean-zero denominator has this property.

## 5.6 Isolating the statistical channel

Finally we separate the mechanism from everything domain-specific. M = 5000 candidate windows per configuration, 10 seeds; a fraction π carry a delayed smooth jump $( \lambda \in \{ 2 , 1 0 , 6 0 \}$ s, amplitude $U ( 3 , 1 0 ) \times \sigma _ { 2 \mathrm { m } } )$ , the rest are pure random walk. Two detectors at matched nominal strictness: detector A thresholds $| \Delta p _ { 2 \mathrm { m } } | / \sigma _ { 1 \mathrm { s } }$ at 10; detector B thresholds $| \Delta p _ { 2 \mathrm { m } } | / ( \sigma _ { 1 \mathrm { s } } \sqrt { 1 2 0 } )$ at 3. Both read as strict; their actual false-positive rates differ by two orders of magnitude (0.361 versus 0.0027). A self-check at $\pi = 1$ confirms the harness recovers the injected response (relative bias 0.039).

![](images/20e3798b5fb56ff2c88e1a9fc8358471203cafcfe04f238492626ac8935d0b43.jpg)

![](images/f4937bec920f7cae378fe1fec86796096474b2f48f4ef7a8ab2b1f61e4c5897d.jpg)  
Figure 4: Two detectors at matched nominal strictness on identical data. Left: at $\pi = 0 . 0 2 .$ A attains precision 0.053 while B attains 0.882. Right: the response curve measured on each detector's selected subset, as a function of π: A's estimate drifts with the composition of its own selection; B's is stable.

$\mathrm { A t } \pi = 0 . 0 2$ the two equally strict detectors yield precisions of 0.053 and 0.882 on the same data. Nothing about the underlying events changed; only a time-scale normalisation inside the acceptance statistic did. This is the minimal mechanism behind Table 1: a detector-defined dataset is the detector's normalisation choices stamped onto data, and those choices are not recoverable from the output.

## 5.7 Scope, and a pre-registered claim we did not get to make

Before running the analysis we registered a threshold for an absolute claim: if the clean official index gave $D ( 5 \mathrm { s } ) \geq 0 . 3 0$ we would claim a fast price-discovery effect, and $\mathrm { a t } \leq 0 . 1 5$ we would not. The measured value on 415 clean official windows is 0.068. We therefore make no such claim. We report this because the pre-registration is what makes the rest of the paper's numbers credible, and because the temptation to relax the threshold after seeing 0.068 was real and is exactly the behaviour the paper argues against.

## 6 Related work

Base rates and precision under imbalance. That posterior probability depends on prevalence is the base-rate fallacy (Bar-Hillel, 1980); its consequences for classifier evaluation are treated by Hand (2009), and Saito & Rehmsmeier (2015) show specifically that precision-based summaries are prevalence-dependent in a way that ROC summaries are not. Our contribution is not the mechanism but an end-to-end measurement of it in a deployed dataset with recoverable ground truth, plus the mixture and estimator-dependence results that follow.

Weak, distant, and self-supervision. Distant supervision (Mintz et al., 2009) and programmatic weak supervision (Ratner et al., 2017) construct labels by rule application; the label-model literature estimates and corrects labellingfunction accuracies, but typically assumes the accuracies are pool-stable, which is precisely the assumption Table 1 violates. Pseudo-labelling suffers a related confirmation bias (Arazo et al., 2020). Positive-unlabelled learning (Elkan & Noto, 2008) formalises learning when negatives are unobserved and makes prevalence explicit; our setting is its evaluation-side dual.

Label noise. A large literature studies learning under label noise (Natarajan et al., 2013; Frénay & Verleysen, 2014). That work almost always models noise as a stochastic corruption applied to correct labels. The contamination here is not of that form: phantoms are generated by the same rule that generates true positives, so the corruption is systematically correlated with the features and carries its own signal shape (§5.3). Noise-robust methods that assume class-conditional random flips do not address it.

Dataset quality and documentation. Northcutt et al. (2021) document pervasive label errors in benchmark test sets. and Gebru et al. (2021) propose documentation standards. Our protocol (§7) is in this tradition and specialises it to the detector-defined case, where the relevant fields are (π, TPR, FPR, c, ρ) rather than provenance narrative.

Event studies. The applied setting is the event study (MacKinlay, 1997); the specific phenomenon our detector was built to find is announcement response (Andersen et al., 2003). We use this literature only as the source of a realistic detector and a realistic downstream estimand.

## 7 A reporting protocol for detector-defined datasets

1. Observed indices first. Use a published index wherever one exists; detection is a last resort, not a convenience.

2. Validate in the deployment pool. Precision must be measured, or derived via (1), on the pool where the detector is actually run. Cross-prevalence transfer of precision is not permitted. High-prevalence validation is near-zero evidence about FPR.

3. Report five quantities. π, TPR, FPR, $c , \rho .$ Precision alone conceals which one moved, and $\rho$ determines whether a given c is tolerable.

4. Declare the time scale of every z-like statistic. Any ratio of a scale-t quantity to a per-unit scale must be √t-normalised explicitly, and the reference window of every normalisation stated.

5. Ship an analytic null with every estimator. A closed-form null for the acceptance rule is what turns "our detector fires often"into a measured FPR. When the estimator is applied to a selected subset, the null must be the selection-conditioned null, not the unconditional one.

6. Avoid per-item normalisation by a signed, mean-zero denominator. Such estimators have no finite expectation (§3.4); use a pooled denominator or none.

7. Run an oracle power check before comparing methods. Compute the bound achieved by a method given perfect ground truth. If it is within seed noise of the baseline, the configuration cannot distinguish methods, and this is a property of the design rather than of the methods (App. B).

## 8 Limitations

1. One detector family. All measurements concern magnitude-threshold detectors. Equations (1) and (3) hold for any detector; our numbers do not transfer to change-point, likelihood-ratio, or learned detectors.

2. One instrument, one asset class, one index family. Prevalences and phantom rates are pool properties. The mechanism is general; the magnitudes are not.

3. Sample coverage. 415 clean official windows, effectively 2013 onward; 68 pre-2013 windows returned no data at the vendor's 1-second resolution. Cross-decade comparability is limited.

4. The selection-conditioned null is incomplete. It reproduces the shape of $D _ { P }$ but not its level (§5.3).

5. Relative claims only. Having abandoned the terminally normalised statistic for headline use, we support relative conclusions between datasets and estimators, not absolute statements about response speed.

## Reproducibility statement

All event indices, phantom labels, screens, curves, and simulator code are released, together with a per-number traceability map, the full decision log for every pre-registered branch point, and an accident log (App. A). The price data

cannot be redistributed under the vendor's licence; we release the exact request manifests (symbol, schema, window, checksum) so that an identical dataset can be reconstructed

## References

T. Andersen, T. Bollerslev, F. Diebold, and C. Vega. Micro effects of macro announcements: real-time price discovery in foreign exchange. American Economic Review, 93(1):38–62, 2003.

E. Arazo, D. Ortego, P. Albert, N. O'Connor, and K. McGuinness. Pseudo-labeling and confirmation bias in deep semi-supervised learning. IJCNN, 2020.

M. Bar-Hillel. The base-rate fallacy in probability judgments. Acta Psychologica, 44(3):211–233, 1980.

C. Elkan and K. Noto. Learning classifiers from only positive and unlabeled data. KDD, 2008.

B. Frénay and M. Verleysen. Classification in the presence of label noise: a survey. IEEE TNNLS, 25(5):845–869, 2014.

T. Gebru, J. Morgenstern, B. Vecchione, J. Vaughan, H. Wallach, H. Daumé III, and K. Crawford. Datasheets for datasets. Communications of the ACM, 64(12):86–92, 2021.

D. J. Hand. Measuring classifier performance: a coherent alternative to the area under the ROC curve. Machine Learning, 77(1):103– 123, 2009.

A. C. MacKinlay. Event studies in economics and finance. Journal of Economic Literature, 35(1):13–39, 1997

M. Mintz, S. Bills, R. Snow, and D. Jurafsky. Distant supervision for relation extraction without labeled data. ACL-IJCNLP, 2009.

N. Natarajan, I. Dhillon, P. Ravikumar, and A. Tewari. Learning with noisy labels. NeurIPS, 2013.

C. Northcutt, A. Athalye, and J. Mueller. Pervasive label errors in test sets destabilize machine learning benchmarks. NeurIPS Datasets and Benchmarks, 2021.

A. Ratner, S. Bach, H. Ehrenberg, J. Fries, S. Wu, and C. Ré. Snorkel: rapid training data creation with weak supervision. VLDB, 11(3):269–282, 2017.

T. Saito and M. Rehmsmeier. The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. PLoS ONE, 10(3):e0118432, 2015.

## A Six accidents in this paper's own pipeline

Each accident below occurred during construction of this paper's dataset, was resolved by a pre-registered action, and is documented in full in the released log. We present them as worked instances of the paper's mechanism. Each entry gives the error, how it was found, the blast radius, and the fix,

Accident 1: the unscaled z-score. The detector's statistic divides a 2-minute move by a per-second standard deviation, omitting the √120 scaling. Under a random-walk null this gives $\mathrm { F P R } = 2 \big ( 1 - \Phi ( 1 0 / \sqrt { 1 2 0 } ) \big ) = 0 . 3 6 1 $ the threshold fires on roughly one in three ordinary windows. Consequence: 487 phantom CPI detections. Table 1's 81.7% phantom rate is this bug, measured. Found by deriving the closed-form null and checking it against simulation (Rule 5), neither of which the original pipeline had. The bug is retained in §5.6 as detector A because it is realistic.

Accident 2: the hallucinated metadata correction. An automated pipeline step "corrected"the FOMC statement release time from 14:00 ET to 13:30 ET for all statements after a cutoff, citing a secondary report of a central-bank announcement that, on inspection of the institution's own press-release pages, does not exist. Both primary sources checked state release at 14:00 ET. Blast radius: 18 ground-truth timestamps shifted by 30 minutes, and one full response-curve computation silently rerun on the corrupted index before the error was caught. Fix: rollback; a hard rule that no metadata may be modified without a primary-source URL recorded in the manifest; per-statement verification against primary pages. This is the paper's thesis in miniature — a confident pipeline, an unmeasured false-positive rate on the task of "correcting metadata", and a ground truth that then validated itself.

Accident 3: positional indexing on a series with gaps. The first response-curve implementation located $t _ { 0 }$ by row offset, assuming one row per second. The vendor emits a row only on quote update, so the offset drifted by up to 48 minutes in sparse periods and the signal averaged away. Fix: timestamp-based alignment throughout.

Accident 4: a cleaning filter that deleted the signal. A futures-roll filter flagged any 1-second return above 20 bp anywhere in the window. Of 86 official windows so flagged, 68 had their largest jump within ±5 s of $t _ { 0 } { \mathrm { : } }$ the filter was removing announcement reactions as artefacts. Fix: the roll screen is evaluated strictly on the pre-window, which cannot contain the reaction. Lesson, and a general one: any cleaning rule whose support intersects the event window can become selective deletion of the target signal — structurally identical to the contamination this paper studies, with the sign flipped.

Accident 5: the same indexing bug, in a different file. After Accident 3 was fixed in the estimator, the identical row-offset pattern reappeared in the screening code's pre-window slice. In high-activity sessions the "pre-window" slice crossed $t _ { 0 }$ and the largest announcement reactions were deleted as rolls — for instance the November 2022 CPI release. whose true pre-window maximum was 3.3 bp and whose post-event move was 126 bp. The months with the strongest signal were deleted at the highest rate. Fix: timestamp masks, a repository-wide audit for row-versus-time indexing, and a regression test. Clean official windows went from 343 to 415.

Accident 6: a factor-of-four estimator artefact. Two implementations of the response curve disagreed by 4× on identical events. The cause was the per-item normalisation of §5.5. Resolution is reported in the main text rather than only here because we consider the estimator pathology a finding rather than a mishap.

## B Oracle power checks and a retracted experiment

For any timing-sensitive evaluation, the oracle is the same downstream method given the true timestamps. The oracleminus-baseline gap upper-bounds the total effect any alignment or denoising method can produce in that configuration. An earlier version of this project evaluated an alignment module in a configuration whose oracle gap was 0.024 CRPS against a comparable seed-noise floor: no method comparison there could have detected anything, regardless of method quality. The experiment was retracted before producing conclusions. Rule 7 operationalises the check as a pre-registration step. The retracted results are released for completeness and should not be cited.

## C The revision-only negative control

Same-month double releases — where a scheduled release carries only a revision to previously published figures and no new reference-period data — provide events with the exact metadata structure of real releases but, by construction, no new information. A calendar rule identifies 17 such windows (13 CPI, 4 payrolls). Only 3 have usable price coverage in our sample. We report this because the control was pre-registered, and we do not interpret it: $n = 3$ is insufficient, and the curve shown in Figure 3 is included only for completeness. Constructing this control properly requires price coverage we could not obtain within the project's data budget, and we flag it as the single cheapest improvement available to a follow-up.

## D The terminally normalised statistic

For completeness we give the legacy 2-hour-anchor convention on the screened sample. $\mathbf { A t } \tau = 5 \mathrm { : }$ s: official 0.068, detected 0.082; at 300 s: 0.271 and 0.317; random-walk baseline $\tau / 7 2 0 0$ . This is the statistic underlying the original dataset's published curve. Its ordering (detected above official at every horizon) reverses under $S ( \tau )$ , which is the demonstration of §5.4. Per-category values at $\tau \in \{ 5 , 3 0 , 3 0 0 , 3 6 0 0 \}$ s are released in f2\_dtau\_by\_subcat. csv.

## E Detector specification and screens

The detector scans the front-month continuous ES series at 1-second best-bid-offer resolution and accepts a window when $| \Delta p _ { 2 \mathrm { m } } |$ exceeds ten per-second standard deviations estimated on the 5 minutes preceding $t _ { 0 }$ (Accident 1's unscaled form). Windows are $t _ { 0 } \pm 2$ h on bid-ask mid. Screens: (i) pre-window roll screen, any 1-second log return above 20 bp strictly before $t _ { 0 } - 1$ s, evaluated on timestamp masks; (ii) degraded-quote screen, windows with more than 5% missing bid or ask. Screens are applied identically to every label set before any comparison