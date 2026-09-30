# ProCTI: Prototype-Refined Global Conditioning for Diffusion-Based Time Series Imputation

Fariza Rashid1\* Duc Van Le² Rahat Masood²

Gustavo Batista2 Aruna Seneviratne2 Suranga Seneviratne¹

1University of Sydney 2University of New South Wales

## Abstract

Time series imputation has progressed from statistical and deep learning approaches to diffusion-based models, which have shown strong recent performance. Existing diffusion-based methods typically condition the reverse process using local contextual information from the current or neighbouring windows. Meanwhile, global dataset-level structure often remains implicit, limiting performance when local observations are sparse, noisy, or unrepresentative. To address this issue, we propose ProCTI, a diffusion-imputation framework that augments local conditioning with retrieved global dataset-level priors through learned prototypes. A hybrid conditioning mechanism integrates this global context with local signals during reverse diffusion, enabling more accurate reconstruction under varying missingness scenarios. Experiments across multiple benchmark datasets show that ProCTI outperforms strong baselines overall under random missingness, while remaining competitive under attribute-wise missingness. Furthermore, we use a latent-regime data model to characterise the precise conditions under which prototype-derived global conditioning provably improves imputation. We support this with a general theoretical analysis of local-global conditioning.

## ProCTI Code Repository

## 1 Introduction

Time series data appears in a wide range of real-world applications, including weather forecasting, finance, healthcare, cybersecurity, human-activity recognition, and audio signal processing [1–5]. Incomplete or missing data due to sensor malfunctions, human error, or communication failures is therefore a significant concern as it can degrade predictive performance and lead to harmful downstream decisions [6, 7]. Besides statistical, machine learning and deep learning methods [8–13], diffusion probabilistic models have emerged as a powerful paradigm for time series imputation [14, 15]. While existing diffusion-based approaches typically condition the reverse denoising process using signals derived from local context [15–17], broader dataset-level patterns are only implicitly encoded in learned parameters. Consequently, when local observations are sparse or noisy, the available context may be insufficient to accurately reflect the true underlying distribution, increasing uncertainty in the denoising process. Diffusion models conditioned only on local context may therefore struggle to generate accurate reconstructions of missing data.

In light of this limitation, we propose ProCTI, a diffusion-based time series imputation framework that explicitly incorporates dataset-level signals into the denoising process. Specifically, we aim to capture recurring global patterns of multivariate temporal behaviour that may exist across the dataset. We refer to these patterns as regimes. To approximate such regimes, ProCTI introduces a learnable prototype bank within a prototype-conditioned diffusion model. Each partially observed input window queries the prototype bank to retrieve a window-specific global context vector, which refines the conditioning signal throughout the reverse diffusion process. This mechanism enables the model to repeatedly incorporate global structural priors during denoising, producing more accurate and coherent imputations.

![](images/bec938d957f1bd11289ed26ba8e47153e6ca9ca2d23dea7ee29f721a707a3bea.jpg)  
(a)

![](images/08173000808d4b6cca144a2e706d111126df756227c56b0e234ea8594e4030ee.jpg)  
(b)

![](images/8048c318c5a05803db0fe53c891d4ec23cdde55acfcc76c37857d711efe5c28a.jpg)  
(c)  
Figure 1: Regime structure and representative imputation behaviour in the Beijing dataset. (a) UMAP projection of normalised windows after PCA, coloured by K-means derived regimes. (b) Mean feature profiles of the discovered regimes, showing distinct combinations of pollution and meteorological conditions. (c) Example masked-window reconstruction for NO2, where ProCTI better matches the ground truth than competing baselines.

As a pre-hoc motivating analysis, in Figure 1 we present a representative illustration of global regimes using the Beijing Air Quality dataset [18]. The left panel presents a UMAP visualisation of z-score normalised fixed-length windows, where each window is flattened and projected into a lower-dimensional space using PCA. Applying K-means clustering in this PCA-reduced space reveals four coarse groups of windows. The middle panel shows the corresponding average feature profiles, obtained by averaging each feature across all windows assigned to a given cluster. Because these profiles are derived from samples spanning the full dataset, they reveal recurring regimes characterised by different combinations of pollution indicators (PM2.5, PM10, SO2, NO2), temperature (TEMP), and wind speed (WSPM). Such dataset-level structure may be difficult to infer from a partially observed window alone. The right panel presents a masked-window imputation example, where leveraging this additional global context enables ProCTI to achieve lower MAE than competing baselines.

Overall, we make the following contributions:

• We introduce ProCTI, a new conditioning framework for diffusion-based time series imputation that explicitly combines retrieved global dataset-level priors with local context. We implement this framework through a prototype-conditioned diffusion model that improves imputation accuracy

• We present a theoretical perspective that distinguishes local and global conditioning, using a latent-regime data model to characterise the precise conditions under which prototypederived global conditioning provably improves imputation. This builds on standard supporting theoretical results that connect global conditioning to the diffusion training objective.

• We evaluate ProCTI against nine baselines on five real-world datasets: ProCTI achieves the best mean MAE on all five datasets under random missingness, reducing MAE by approximately 3-30% over the strongest diffusion baseline (FGTI), and is best or secondbest in most attribute-wise missingness scenarios.

## 2 Related Work

Time series modelling Time-series modelling has seen rapid progress across generation, forecasting, and imputation tasks, with early approaches relying on statistical techniques and autoregressive methods [19, 20, 8]. More recent machine learning and deep learning methods emerged such as TIDER [10], which uses matrix factorisation to disentangle temporal components, and BRITS [11] and SAITS [21], which employ bidirectional Recurrent Neural Networks (RNNs) and self-attention respectively, to capture long-range dependencies. In forecasting, non-Transformer approaches such as SCINet [22] and FreTS [23] model temporal interactions through recursive convolution and frequency-domain multilayer perceptrons respectively. Transformer-based forecasting methods [24— 29] have also demonstrated strong performance by modeling long-range temporal dependencies through attention-based architectures, while time series generation methods like Diffusion-TS [14] and PaD-TS [30] are diffusion-based models which generate realistic synthetic sequences by learning temporal dynamics and preserving dataset-level statistical properties.

Conditioning in imputation models Existing diffusion-based imputation methods incorporate contextual information in varied ways, mostly relying on local or task-specific priors rather than explicitly learning dataset-level global structure. CSDI [15] conditions the reverse process directly on observed values and attention-based temporal/feature dependencies, without an explicit global prior. PriSTI [31] introduces context for spatiotemporal data by extracting coarse global priors from observed values together with geographic relationships, but this prior is specific to spatial sensor networks rather than general multivariate time series. MTSCI [16] conditions on neighbouringwindow information and consistency constraints, thereby exploiting short-range temporal context instead of dataset-wide structure, while FGTI [17] incorporates frequency-domain priors through dominant- and high-frequency signals. Non-diffusion models such as TG-MSFM [32] use multi-scale flow trajectories to reconcile global trends with local details during deterministic reconstruction.

In contrast, we formalise conditioning for diffusion-based imputation by distinguishing local context, derived from the partially observed input window, and global context, representing recurring datasetlevel structure shared across samples. We define their combination within the reverse diffusion process as complementary sources of guidance for reconstruction. Following this principle, ProCTI leverages a learned prototype bank to retrieve and inject global context alongside local observations into the denoising process for conditional imputation. This differs from PaD-TS [30], which preserves population-level statistics for unconditional time-series generation. We provide an overview of existing prototype-based methods in time-series modelling in Appendix A.1.

## 3 Prototype-Refined Global Conditioning

In this section, we present ProCTI, a diffusion-imputation framework that combines local window evidence with retrieved global dataset-level priors during reverse diffusion. As illustrated in Figure 2 ProCTI learns dataset-level prototypes that approximate recurring global regimes and uses them to produce instance-specific (i.e., window-level) conditioning signals. We first describe the prototype conditioning module, which combines global prototype guidance with local contextual information. We then formally distinguish local and global conditioning and use a latent-regime data model to characterise when combining them provably improves imputation. This builds on standard supporting results connecting global conditioning to the diffusion training objective.

## 3.1 Prototype-conditioning module

The prototype conditioning module contains a learnable prototype bank trained over the training dataset to approximate recurring global regimes. For each input window, cross-attention retrieves an attribute-wise global context vector from the prototype bank. This signal is then combined with local contextual information to form the conditioning input for the reverse diffusion process.

Problem setup. We consider a multivariate time series dataset X consisting of windows $\boldsymbol { x } \in \mathbb { R } ^ { L \times K }$ where L represents the time length of the window and K represents the number of attributes. While the window may have naturally missing values, we apply an artificial keep-mask $m \in \{ 0 , 1 \} ^ { L \times K }$ where $m _ { t , k } = 1$ indicates that the value at time step t and attribute k is observed, and $m _ { t , k } = 0$ otherwise. During training, the model learns to reconstruct values at masked positions through a diffusion denoising process.

Token construction. Given a mini-batch of B input windows $\mathbf { X } \in \mathbb { R } ^ { B \times L \times K }$ and the corresponding keep-masks $\mathbf { M } \in \{ 0 , 1 \} ^ { B \times L \times K }$ , where B denotes the batch size, we construct a 2-dimensional token at each time-attribute position:

$$
\begin{array} { r } { \mathbf { z } _ { b , t , k } = \left[ x _ { b , t , k } , \ m _ { b , t , k } \right] \in \mathbb { R } ^ { 2 } . } \end{array}\tag{1}
$$

![](images/b2f5cdb2a38c96caee0bc55186a3658760aa5160c9496e1296b28511a9962f7b.jpg)  
Figure 2: Prototype-conditioning module

Each token is projected into a D-dimensional embedding space via a learnable linear map:

$$
\mathbf { e } _ { b , t , k } = \mathbf { W } _ { \mathrm { p r o j } } \mathbf { z } _ { b , t , k } \in \mathbb { R } ^ { D } .\tag{2}
$$

The resulting tensor has shape $\mathbb { R } ^ { B \times L \times K \times D }$

Query construction. To obtain a single representation per window, we reshape and average over all time-attribute embeddings to obtain a window-level query:

$$
\mathbf { q } _ { b } = \frac { 1 } { L K } \sum _ { t = 1 } ^ { L } \sum _ { k = 1 } ^ { K } \mathbf { e } _ { b , t , k } ~ \in ~ \mathbb { R } ^ { D } .\tag{3}
$$

This query captures the instance-specific characteristics of the partially observed window, including both observed values and missingness patterns. This is then used to retrieve relevant global patterns from the prototype bank, which encodes dataset-level temporal regimes.

Prototype bank and cross-attention. We maintain a learnable prototype bank $P = \{ p _ { m } \} _ { m = 1 } ^ { M }$ with $\boldsymbol { p _ { m } } \in \mathbb { R } ^ { D }$ , intended to capture characteristic temporal regimes (periodic behaviour, irregular fluctuations, or cross-channel correlations) that may not be fully observable within any single partially observed window. Multi-head cross-attention from the window query $q _ { b }$ to the prototype bank yields a context vector $c _ { b } \in \mathbb { R } ^ { D }$ , which is mapped to the attribute domain by a learnable projection $\mathbf { \bar { \boldsymbol { W } } } _ { r } \in \mathbb { R } ^ { K \times D }$

$$
\begin{array} { r } { c _ { b } = \mathrm { M H C r o s s A t t n } ( q _ { b } ; P ) , \qquad r _ { b } = W _ { r } c _ { b } \in \mathbb { R } ^ { K } . } \end{array}\tag{4}
$$

The instance-dependent regime vector $r _ { b }$ thus translates globally learned temporal regimes into window-level, attribute-specific signals used to condition the diffusion process. Full multi-head cross-attention details are provided in Appendix $_ { \mathrm { A } . 2 } ^ { }$

## 3.2 Global-conditioned diffusion model

We integrate the regime vector $r _ { b }$ from Section 3.1 into the diffusion-based imputation framework, refining the conditioning signal with instance-specific global information rather than relying on local observations or spectral features alone.

## 3.2.1 Conditional diffusion formulation

We follow the standard DDPM formulation [15]: the forward process gradually corrupts a clean input $x _ { 0 }$ into $x _ { T }$ via a Markov chain with a predefined noise schedule, admitting the closed-form $x _ { t } =$ $\sqrt { \bar { \alpha } _ { t } } x _ { 0 } + \sqrt { 1 - \bar { \alpha } _ { t } }$ € with $\epsilon \sim \mathcal { N } ( 0 , \bf { I } )$ . The reverse process $p _ { \theta } ( x _ { t - 1 } \mid x _ { t } ) = \mathcal { N } ( x _ { t - 1 } ; \mu _ { \theta } ( x _ { t } , t ) , \sigma _ { t } ^ { 2 } \mathbf { I } )$ is parameterised by a denoising network $\epsilon _ { \theta } ( x _ { t } , t )$ trained to predict the injected noise €. For imputation, the reverse process is conditioned on a signal h derived from the partially observed window: $\begin{array} { r } { p _ { \theta } ( x _ { 0 : T } ^ { t a } \mid h ) = \hat { p ( x _ { T } ^ { t a } ) \prod _ { t } } p _ { \theta } ( x _ { t - 1 } ^ { t a } \mid x _ { t } ^ { t a } , h ) } \end{array}$ , where $x _ { 0 } ^ { t a }$ denotes the missing (target) entries. Existing approaches typically construct h from local observations or window-level frequency representations; ProCTI refines this signal with prototype-derived global information.

## 3.2.2 Theoretical perspective on local and global conditioning

Let $\boldsymbol { X } \in \mathbb { R } ^ { L \times K }$ denote a multivariate time series window, with observed entries O and missing entries $Y :$ ; the imputation task is to model $p ( Y \mid O )$ . We formalise the distinction between local and global conditioning, then state four results characterising the value of hybrid conditioning during reverse diffusion.

Definition 1 (Local conditioning). Local conditioning refers to information derived directly from the observed values in the input window. Formally, we define a local conditioning representation as

$$
h _ { \mathrm { l o c } } = \phi _ { \mathrm { l o c } } ( O ) ,
$$

where $\phi _ { \mathrm { l o c } }$ is a function of observed values and missingness patterns within the current window.

Definition 2 (Global conditioning). Global conditioning refers to information derived from datasetlevel structure beyond the current input window. We define a latent regime variable $Z$ as an unobserved random variable representing global temporal patterns governing the data distribution, such as periodic structure, long-range trends, or cross-channel dependencies. In ProCTI, this is approximated by a learned prototype bank, yielding a representation

$$
h _ { \mathrm { g l o b } } = \phi _ { \mathrm { g l o b } } ( O ; P ) ,
$$

where P denotes the prototype bank learned over the training dataset.

Definition 3 (Local-global conditioning). We define hybrid conditioning as the combination of local and global components:

$$
h = \psi ( h _ { \mathrm { l o c } } , h _ { \mathrm { g l o b } } ) .
$$

To implement hybrid conditioning, we build upon a frequency-aware diffusion framework [17] that decomposes window-level frequency content into a high-frequency component $h ^ { h f } \in \mathbb { R } ^ { B \times \dot { L } \times \dot { K } }$ and a dominant-frequency component $\bar { h ^ { d o m } } \in \mathbb { R } ^ { B \times L \times K }$ . The dominant-frequency component captures coarse temporal structure such as periodicity and long-range trends — properties associated with dataset-level regimes — while the high-frequency component encodes fine-grained instance-specific variation. We therefore inject the regime vector $r _ { b } ~ \in \mathbb { R } ^ { K }$ from the prototype module into the dominant-frequency branch only:

$$
h _ { b , t , k } ^ { d o m + r } = h _ { b , t , k } ^ { d o m } + \alpha r _ { b , k } , \qquad h = \left[ h _ { b , t } ^ { h f } ; h _ { b , t } ^ { d o m + r } \right] \in \mathbb { R } ^ { 2 K } ,\tag{5}
$$

where α is a learnable scalar controlling the strength of global correction. The resulting prototyperefined conditioning h is fed to the denoising network $\epsilon _ { \theta } ( x _ { t } , t \mid h )$ at each diffusion step.

Theoretical results. We next state a sequence of results: standard uncertainty and score-based arguments, followed by a latent-regime characterisation of precisely when hybrid conditioning improves imputation. Lemma 1 records that conditioning on a global variable $\vec { Z , }$ when informative, reduces residual uncertainty about $Y$ . Proposition 1 decomposes the conditional score that the denoising network approximates into a local term and a regime guidance term, providing a constructive interpretation of how global conditioning intervenes in reverse diffusion. Proposition 2 connects this decomposition to the denoising mean-squared error that the diffusion training objective directly minimises. Proposition 3 specialises to a latent-regime data model and gives equivalent conditions on $( Y , O , Z )$ under which the antecedents of the previous results hold, delineating both when ProCTI's prototype mechanism should be expected to help and when it should not.

Lemma 1 (Conditional uncertainty reduction). If $Z$ carries information about Y beyond that already contained in O, namely $I ( Y ; { \overline { { Z } } } \mid O ) > 0$ , then $H ( Y \mid O , Z ) < H ( Y \mid O )$ 1

Proof. By the definition of conditional mutual information, $I ( Y ; Z \mid O ) = H ( Y \mid O ) { - } H ( Y \mid O , Z )$ so positivity of the left-hand side is equivalent to a strict decrease of conditional entropy. □

The lemma states the information-theoretic premise that global information, when relevant, reduces residual uncertainty in $Y$ , but does not by itself describe how this reduction is realised inside the reverse diffusion process. The next two propositions make that connection explicit, first at the level of the score function approximated by the denoising network, then at the level of the mean-squared error directly minimised by training

Proposition 1 (Score decomposition under hybrid conditioning). Under hybrid conditioning $h = ( O , Z )$ , the conditional score of the noisy state $X _ { t }$ admits the decomposition

$$
\nabla _ { x _ { t } } \log { p _ { t } ( x _ { t } \mid O , Z ) } = \underbrace { \nabla _ { x _ { t } } \log { p _ { t } ( x _ { t } \mid O ) } } _ { \mathrm { l o c a l s c o r e } } + \underbrace { \nabla _ { x _ { t } } \log { p ( Z \mid x _ { t } , O ) } } _ { \mathrm { r e g i m e ~ g u i d a n c e } } .\tag{6}
$$

Proof. By Bayes' rule, $p _ { t } ( x _ { t } \mid O , Z ) = p ( Z \mid x _ { t } , O ) p _ { t } ( x _ { t } \mid O ) / p ( Z \mid O )$ . Taking logarithms and differentiating in $x _ { t } .$ the term $\log p ( \bar { Z } \mid \bar { O } )$ vanishes since it does not depend on $x _ { t }$ □

The decomposition has the same structure as classifier guidance for conditional diffusion [33, 34] and assigns each conditioning source a distinct role: the local term is the score of the noised target given only $O ,$ while the regime guidance term pulls $x _ { t }$ toward states compatible with $Z .$ ProCTI does not have direct access to $\bar { Z }$ at inference time; instead, its prototype-conditioning module learns an estimator $G = \phi _ { \mathrm { g l o b } } ( { \cal O } ; { \cal P } )$ of regime-relevant information and feeds $G$ to the denoising network as a feasible surrogate for the regime guidance term in (6).

Proposition 2 (MMSE reduction under hybrid conditioning). Let $\mu ^ { \star } ( x _ { t } , h ) = \mathbb { E } [ X _ { 0 } \mid X _ { t } = x _ { t } , h ]$ denote the Bayes-optimal denoiser under conditioning $h ,$ and let $h _ { 1 } = O$ and $h _ { 2 } \doteq ( O , Z )$ . Then

$$
\begin{array} { r } { \mathbb { E } \big [ \| X _ { 0 } - \mu ^ { \star } ( X _ { t } , h _ { 1 } ) \| ^ { 2 } \big ] - \mathbb { E } \big [ \| X _ { 0 } - \mu ^ { \star } ( X _ { t } , h _ { 2 } ) \| ^ { 2 } \big ] = \mathbb { E } \Big [ \big \| \mu ^ { \star } ( X _ { t } , h _ { 2 } ) - \mu ^ { \star } ( X _ { t } , h _ { 1 } ) \big \| ^ { 2 } \Big ] \geq 0 , \ ( 7 ) } \end{array}
$$

with strict inequality whenever $Z$ carries information about $X _ { 0 }$ beyond that already in $( X _ { t } , O )$

Proof. By the tower property, $\mu ^ { \star } ( X _ { t } , h _ { 1 } ) \ = \ \mathbb { E } [ \mu ^ { \star } ( X _ { t } , h _ { 2 } ) \ | \ X _ { t } , O ]$ Decomposing $X _ { 0 } \mathrm { ~ - ~ }$ $\mu ^ { \star } ( \dot { X } _ { t } , h _ { 1 } \dot { ) } = ( X _ { 0 } - \bar { \mu ^ { \star } } ( \bar { X } _ { t } , \dot { h } _ { 2 } ) ) + ( \mu ^ { \star } ( \dot { X } _ { t } , h _ { 2 } ) - \mu ^ { \star } ( X _ { t } , \dot { h _ { 1 } } ) )$ , the cross-term has zero expectation by the orthogonality of conditional means. Squaring and taking expectations yields the identity, with strict inequality precisely when the two optimal denoisers differ on a set of positive measure.

Proposition 2 is the operational counterpart of Lemma 1: it records the corresponding reduction in mean-squared error of the optimal denoiser, which is the quantity directly minimised by the diffusion training objective. Together, Propositions 1 and 2 give a unified picture: global conditioning enters the score additively as a regime guidance term, and this addition can only decrease the optimal denoising error, strictly so whenever the guidance term is informative.

Latent-regime data model. The previous three results all depend on the antecedent that $Z$ carries information about $Y \ ( \mathrm { o r } \ X _ { 0 } )$ beyond O. To pin down when this antecedent holds, we adopt a latent-regime model in which each window is associated with $Z \in \{ 1 , \ldots , M ^ { \star } \}$ with $( O , Y ) \mid { \bar { Z } } \sim$ $p ( \cdot , \cdot \mid Z )$ . The model is non-restrictive; any joint distribution over $X$ admits such a representation and serves only to make the conditions for prototype-based improvement explicit.

Proposition 3 (Conditions for prototype informativeness). Under the latent-regime data model, the following are equivalent:

$$
\begin{array} { r l } & { \mathrm { ( i ) ~ } I ( Y ; Z \mid O ) > 0 ; } \\ & { \mathrm { ( i i ) ~ } Y \mathcal { J } \mathcal { L } \mid O ; } \\ & { \mathrm { ( i i i ) ~ } \operatorname* { i n f } _ { f } \mathbb { E } [ \| Y - f ( O ) \| ^ { 2 } ] > \operatorname* { i n f } _ { g } \mathbb { E } [ \| Y - g ( O , Z ) \| ^ { 2 } ] . } \end{array}
$$

When any of these conditions hold, Lemma 1 yields a strict reduction in conditional entropy, and Proposition 2 yields a strict reduction in optimal denoising mean-squared error from hybrid over local conditioning. When they fail, that is when $Y \bot \bot Z \mid O$ , no function of $Z$ adds information about $Y$ beyond O, and prototype-derived global conditioning cannot improve the optimal denoiser.

Proof. $( i ) \Leftrightarrow ( i i )$ is the standard equivalence between vanishing conditional mutual information and conditional independence. For $( i i ) \Leftrightarrow ( i i i )$ , the optimal predictor of $Y$ given $( O , Z )$ in mean-squared error is $\mathbb { E } [ Y \mid { \dot { O } } , Z ]$ , which coincides almost surely with $\mathbb { E } [ Y \mid O ]$ if and only if $\dot { Y } \perp \perp Z \mid O$ □

Table 1: Comparison of imputation effectiveness across baseline methods, aggregated over missingness ratios $\{ 0 . 1 , 0 . 3 , 0 . 5 , 0 . 7 \}$ . Each cell reports the mean with the standard deviation as a subscript. The best mean MAE and RMSE are shown in bold, and the second-best results are underlined. The full table of average results for each missingness ratio is in Appendix A.5
<table><tr><td>Dataset</td><td>Metric |</td><td>BRITS</td><td>CSDI</td><td>SCINet</td><td>TIDER</td><td>MTSCI</td><td>DiffusionTS</td><td>iTransformer</td><td>FGTI</td><td>PaD-TS</td><td>ProCTI</td></tr><tr><td rowspan="2">Gait</td><td>MAE</td><td> $0 . 3 5 2 { \scriptstyle \pm . 0 2 5 }$ </td><td> $0 . 4 4 9 { \scriptstyle \pm . 0 1 7 }$ </td><td> $0 . 4 7 5 { \scriptstyle \pm . 0 4 0 }$ </td><td>0.746±.002</td><td> $0 . 5 0 1 { \scriptstyle \pm . 0 1 9 }$ </td><td> $0 . 3 6 7 { \scriptstyle \pm . 0 0 9 }$ </td><td> $0 . 5 3 7 { \scriptstyle \pm . 0 3 5 }$ </td><td> $\underline { { 0 . 3 3 2 } } . . . 0 2 6$ </td><td> $0 . 9 4 2 { \scriptstyle \pm . 0 3 2 }$ </td><td> $\mathbf { 0 . 3 1 3 _ { \pm . 0 2 6 } }$ </td></tr><tr><td>RMSE</td><td> $\underline { { 0 . 5 3 3 } } _ { \pm . 0 3 7 }$ </td><td> $0 . 7 2 6 { \scriptstyle \pm . 0 1 4 }$ </td><td>0.671±.054</td><td> $0 . 9 8 5 { \scriptstyle \pm . 0 0 3 }$ </td><td>0.750±.019</td><td> $0 . 5 8 5 { \scriptstyle \pm . 0 1 4 }$ </td><td> $0 . 7 6 5 { \scriptstyle \pm . 0 3 9 }$ </td><td>0.563±.035</td><td>1.197±.032</td><td> $\mathbf { 0 . 5 1 8 _ { \pm . 0 4 1 } }$ </td></tr><tr><td rowspan="2">PhysioNet</td><td>MAE</td><td></td><td> $0 . 4 5 6 { \scriptstyle \pm . 0 3 4 }$ </td><td> $0 . 4 0 1 { \scriptstyle \pm . 0 9 1 }$ </td><td> $0 . 7 5 1 { \scriptstyle \pm . 0 3 0 }$ </td><td>0.526±.075</td><td> $0 . 5 4 4 { \scriptstyle \pm . 0 2 6 }$ </td><td> $0 . 4 1 3 { \scriptstyle \pm . 0 9 8 }$ </td><td> $\underline { { 0 . 3 6 2 } } _ { \pm . 0 8 2 }$ </td><td> $0 . 8 6 5 { \scriptstyle \pm . 0 6 2 }$ </td><td> $\mathbf { 0 . 2 5 9 _ { \pm . 0 5 8 } }$ </td></tr><tr><td>RMSE</td><td> $\begin{array} { c } { 0 . 5 0 5 { \scriptstyle \pm . 0 3 2 } } \\ { 3 . 4 0 1 { \scriptstyle \pm . 1 5 8 } } \end{array}$ </td><td> $4 . 9 1 9 { \scriptstyle \pm . 2 0 3 }$ </td><td> $\underline { { 1 . 5 5 7 } } . 4 4 3$ </td><td> $3 . 4 6 5 \pm . 5 5 8$ </td><td>1.609±.210</td><td> $3 . 0 5 4 \pm . 6 9 8$ </td><td> $1 . 9 2 5 { \scriptstyle \pm . 5 0 9 }$ </td><td> $2 . 9 7 0 { \overline { { \pm } } } . 4 3 1$ </td><td> $3 . 1 5 3 { \scriptstyle \pm . 5 5 1 }$ </td><td> $1 . 3 8 3 { \scriptstyle \pm . 2 9 9 }$ </td></tr><tr><td rowspan="2">Weather</td><td>MAE</td><td></td><td> $0 . 1 2 5 { \scriptstyle \pm . 0 1 5 }$ </td><td> $0 . 2 7 6 { \scriptstyle \pm . 0 8 5 }$ </td><td></td><td></td><td> $0 . 2 4 6 { \scriptstyle \pm . 0 3 7 }$ </td><td></td><td></td><td> $0 . 1 7 6 { \scriptstyle \pm . 0 1 3 }$ </td><td> $\mathbf { 0 . 0 9 5 _ { \pm . 0 3 7 } }$ </td></tr><tr><td>RMSE</td><td> $\begin{array} { c } { 0 . 1 4 8 _ { \pm . 0 4 6 } } \\ { 0 . 3 5 5 _ { \pm . 0 4 0 } } \end{array}$ </td><td> $0 . 3 3 4 { \scriptstyle \pm . 0 2 0 }$ </td><td>0.434±.094</td><td> $\begin{array} { c } { 0 . 6 4 9 { \scriptstyle \pm . 0 0 6 } } \\ { 0 . 8 3 3 { \scriptstyle \pm . 0 0 8 } } \end{array}$ </td><td> $\begin{array} { c } { 0 . 2 7 8 _ { \pm . 1 1 5 } } \\ { 0 . 4 7 2 _ { \pm . 1 1 9 } } \end{array}$ </td><td> $0 . 4 9 3 { \scriptstyle \pm . 0 5 6 }$ </td><td> $\begin{array} { c } { 0 . 2 8 3 _ { \pm . 1 1 6 } } \\ { 0 . 4 4 2 _ { \pm . 1 1 1 } } \end{array}$ </td><td> $\frac { 0 . 1 1 3 } { 0 . 2 9 8 } { \pm . 0 4 9 }$ </td><td> $0 . 3 6 2 { \scriptstyle \pm . 0 1 8 }$ </td><td> $\mathbf { 0 . 2 7 4 _ { \pm . 0 4 0 } }$ </td></tr><tr><td rowspan="2">Beijing</td><td>MAE</td><td> $0 . 2 1 7 { \scriptstyle \pm . 0 7 3 }$ </td><td></td><td></td><td></td><td></td><td></td><td> $0 . 4 0 5 { \scriptstyle \pm . 1 2 3 }$ </td><td></td><td> $0 . 6 9 9 { \scriptstyle \pm . 0 7 3 }$ </td><td> $\mathbf { 0 . 1 5 3 _ { \pm . 0 4 8 } }$ </td></tr><tr><td>RMSE</td><td>0.482±.076</td><td> $\begin{array} { c } { 0 . 1 8 6 _ { \pm . 0 9 1 } } \\ { 0 . 4 6 4 _ { \pm . 1 1 9 } } \end{array}$ </td><td> $\begin{array} { c } { 0 . 3 4 1 _ { \pm . 0 7 0 } } \\ { 0 . 5 5 9 _ { \pm . 0 8 5 } } \end{array}$ </td><td> $\begin{array} { c } { 0 . 7 2 2 _ { \pm . 0 0 1 } } \\ { 1 . 0 0 3 _ { \pm . 0 0 3 } } \end{array}$ </td><td> $\begin{array} { c } { 0 . 2 8 2 _ { \pm . 1 2 6 } } \\ { 0 . 5 7 7 _ { \pm . 1 3 7 } } \end{array}$ </td><td> $\begin{array} { c } { 0 . 2 5 6 { \pm } . 0 2 8 } \\ { 0 . 5 8 2 { \pm } . 0 2 2 } \end{array}$ </td><td> $0 . 6 6 0 { \scriptstyle \pm . 1 3 8 }$ </td><td> $\frac { 0 . 1 5 9 } { 0 . 4 3 7 } \pm . 0 5 3$ </td><td> $0 . 9 8 8 { \scriptstyle \pm . 0 7 9 }$ </td><td> $\mathbf { 0 . 4 1 4 _ { \pm . 0 7 0 } }$ </td></tr><tr><td rowspan="2">Stock</td><td>MAE</td><td> $3 . 8 8 0 \pm . 2 4 5$ </td><td></td><td></td><td> $5 . 2 7 8 { \scriptstyle \pm . 0 4 9 }$ </td><td>3.762±.494</td><td> $4 . 3 0 2 { \scriptstyle \pm . 3 6 3 }$ </td><td> $\underline { { 2 . 6 2 2 } } _ { \pm . 9 7 1 }$ </td><td> $3 . 1 8 6 _ { \pm . 4 2 2 }$ </td><td> $4 . 9 8 7 { \scriptstyle \pm . 1 5 0 }$ </td><td> $2 . 2 2 3 _ { \pm . 6 6 0 }$ </td></tr><tr><td>RMSE</td><td> $4 . 2 9 9 { \scriptstyle \pm . 2 4 8 }$ </td><td> $\begin{array} { c } { 2 . 8 4 2 { \pm } . 9 2 6 } \\ { 3 . 2 2 3 { \pm } . 9 7 6 } \end{array}$ </td><td> $\begin{array} { c } { 2 . 6 7 1 _ { \pm . 8 1 2 } } \\ { 3 . 4 8 2 _ { \pm . 9 1 9 } } \end{array}$ </td><td> $5 . 7 1 2 { \scriptstyle \pm . 0 3 9 }$ </td><td>4.120±.488</td><td> $4 . 8 0 1 { \scriptstyle \pm . 2 4 0 }$ </td><td> $\underline { { 3 . 2 2 6 } } _ { \pm . 9 6 3 }$ </td><td> $4 . 0 4 6 { \scriptstyle \pm . 2 3 0 }$ </td><td>5.531±.089</td><td> $2 . 8 0 8 _ { \pm . 6 6 6 }$ </td></tr></table>

Proposition 3 delineates the regime in which ProCTI's prototype-conditioning mechanism is expected to provide gains. When $Y$ is well-approximated by a deterministic function of $O - { \bf a s }$ when an entire missing channel can be recovered by linear regression on the remaining channels — $Y \perp \perp Z \mid O$ holds approximately, and no global conditioning signal can improve on a sufficiently expressive local denoiser. We return to this prediction in Section 4, where it explains the dataset-dependence of ProCTI's gains observed under attribute-wise missingness. In the complementary regime, where Z carries residual information about $Y$ given O, the prototype-conditioning module aims to recover this information from observations alone: under the latent-regime model with bank size $M \geq M ^ { \star }$ and a sufficiently expressive attention map $( \pi _ { m } ( O ) \approx p ( Z = m \mid O ) )$ , the retrieved regime vector $r _ { b }$ functions as a soft sufficient statistic for $Z$ given $O ,$ and Propositions 1 and 2 quantify the corresponding score and MMSE improvements.

## 4 Experiments

This section describes the experiments used to evaluate ProCTI against state-of-the-art baseline methods in terms of imputation effectiveness and robustness under attribute-wise missing scenarios. All experiments were conducted on a machine with an Intel Core i7-13700K CPU, two NVIDIA GeForce RTX 4090 GPUs (24GB each), and 128GB RAM.

## 4.1 Experimental Setup

Datasets We used five real-world time series datasets for our experiments, comparable to other similar work [14, 16, 17, 35]. PhysioNet [3] comprises multivariate clinical time-series data collected from ICU patients, with each record comprising measurements of approximately 40 physiological variables. The Beijing [18] dataset includes air-pollutant data collected from 12 air-quality monitoring sites in Beijing. The Weather [1] dataset is from the weather station of the Max Planck Institute for Biogeochemistry, comprising 14 meteorological attributes recorded between January 1st 2009 and December 31st 2016. The Stock [2] dataset reflects Google stock price data from 2004 to 2019, with each record representing a new day. Finally, the Gait [4] dataset comprises gait data collected using IMU sensors from 744 users during walks on level ground. We provide further details about dataset preparation for our experiments in Appendix A.3

Baselines We evaluate ProCTI against a diverse set of baselines. We consider imputation methods including BRITS [11], CSDI [15], MTSCI [16], FGTI [17] and TIDER [10]. We include two forecasting models, SCINet [22] and iTransformer [29], that can be adapted to imputation via maskedvalue reconstruction [35]. We also evaluate two diffusion-based generation models, Diffusion-TS [14] and PaD-TS [30], that can likewise be adapted for conditional imputation.

Evaluation metrics For all of our experiments, we report the MAE and RMSE values to reflect imputation quality in terms of closeness to the ground truth. Therefore, the lower the value, the better the imputation. While MAE reflects the average error, RMSE reflects extreme deviations from the ground truth.

Table 2: Structured missingness evaluation results. Best results are in bold and second-best are underlined.
<table><tr><td>Dataset</td><td>Metric</td><td>Drop</td><td>BRITS</td><td>CSDI</td><td>SCINet</td><td>TIDER</td><td>MTSCI</td><td>DiffusionTS</td><td>iTransformer</td><td>FGTI</td><td>PaD-TS</td><td>ProCTI</td></tr><tr><td rowspan="4">Gait</td><td>MAE</td><td>1</td><td>0.735</td><td>0.801</td><td>0.745</td><td>0.745</td><td>0.741</td><td>1.053</td><td>0.745</td><td>0.616</td><td>1.037</td><td>0.619</td></tr><tr><td></td><td>2</td><td>0.718</td><td>0.808</td><td>0.748</td><td>0.748</td><td>0.745</td><td>1.059</td><td>0.748</td><td>0.643</td><td>1.044</td><td>0.640</td></tr><tr><td>RMSE</td><td>1</td><td>1.017</td><td>1.039</td><td>0.984</td><td>0.985</td><td>0.961</td><td>1.311</td><td>0.984</td><td>0.839</td><td>1.293</td><td>0.821</td></tr><tr><td></td><td>2</td><td>0.971</td><td>1.047</td><td>0.988</td><td>0.988</td><td>0.967</td><td>1.316</td><td>0.988</td><td>0.877</td><td>1.300</td><td>0.851</td></tr><tr><td rowspan="4">PhysioNet</td><td>MAE</td><td>1</td><td>0.931</td><td>0.841</td><td>0.836</td><td>0.772</td><td>0.946</td><td>0.780</td><td>0.805</td><td>0.856</td><td>0.980</td><td>0.747</td></tr><tr><td></td><td>2</td><td>0.962</td><td>0.694</td><td>0.708</td><td>0.768</td><td>0.740</td><td>0.693</td><td>0.692</td><td>0.645</td><td>0.944</td><td>0.619</td></tr><tr><td>RMSE</td><td>1</td><td>1.246</td><td>1.010</td><td>1.031</td><td>0.938</td><td>1.147</td><td>1.002</td><td>0.998</td><td>1.042</td><td>1.215</td><td>0.903</td></tr><tr><td></td><td>2</td><td>1.281</td><td>0.890</td><td>0.885</td><td>0.969</td><td>0.922</td><td>0.929</td><td>0.874</td><td>0.831</td><td>1.183</td><td>0.781</td></tr><tr><td rowspan="4">Weather</td><td></td><td>1</td><td>0.205</td><td>0.687</td><td>0.669</td><td>0.664</td><td>0.667</td><td>0.671</td><td>0.664</td><td>0.310</td><td>0.724</td><td>0.269</td></tr><tr><td>MAE</td><td>2</td><td>0.212</td><td>0.693</td><td>0.679</td><td>0.674</td><td>0.671</td><td>0.696</td><td>0.673</td><td>0.318</td><td>0.755</td><td>0.288</td></tr><tr><td>RMSE</td><td>1</td><td>0.438</td><td>0.859</td><td>0.840</td><td>0.838</td><td>0.830</td><td>0.874</td><td>0.838</td><td>0.593</td><td>0.932</td><td>0.479</td></tr><tr><td></td><td>2</td><td>0.480</td><td>0.872</td><td>0.855</td><td>0.852</td><td>0.848</td><td>0.902</td><td>0.852</td><td>0.561</td><td>0.969</td><td>0.494</td></tr><tr><td rowspan="4">Beijing</td><td></td><td>1</td><td>1.107</td><td>0.708</td><td>0.737</td><td>0.745</td><td>0.647</td><td>0.875</td><td>0.736</td><td>0.429</td><td>0.863</td><td>0.419</td></tr><tr><td>MAE</td><td>2</td><td>0.927</td><td>0.700</td><td>0.736</td><td>0.734</td><td>0.653</td><td>0.881</td><td>0.735</td><td>0.426</td><td>0.877</td><td>0.436</td></tr><tr><td>RMSE</td><td>1</td><td>1.484</td><td>0.981</td><td>1.053</td><td>1.030</td><td>0.915</td><td>1.180</td><td>1.052</td><td>0.769</td><td>1.173</td><td>0.716</td></tr><tr><td></td><td>2</td><td>1.309</td><td>0.990</td><td>1.053</td><td>1.094</td><td>0.937</td><td>1.191</td><td>1.053</td><td>0.776</td><td>1.189</td><td>0.746</td></tr><tr><td rowspan="4">Stock</td><td></td><td></td><td>2.177</td><td>4.975</td><td>5.350</td><td>5.073</td><td>4.882</td><td>4.655</td><td>5.092</td><td>5.422</td><td>4.776</td><td>4.580</td></tr><tr><td>MAE</td><td>1 2</td><td>2.355</td><td>5.079</td><td>5.367</td><td>5.146</td><td>5.374</td><td>4.827</td><td>5.166</td><td>5.391</td><td>4.800</td><td>4.714</td></tr><tr><td></td><td>1</td><td>2.528</td><td>5.522</td><td>5.797</td><td>5.516</td><td>5.347</td><td>5.152</td><td>5.533</td><td>5.784</td><td>5.243</td><td>4.995</td></tr><tr><td>RMSE</td><td>2</td><td>2.689</td><td>5.581</td><td>5.778</td><td>5.550</td><td>5.793</td><td>5.282</td><td>5.566</td><td>5.787</td><td>5.308</td><td>5.082</td></tr></table>

## 4.2 Imputation effectiveness

We compare imputation effectiveness under stochastic segment masking, artificially masking observed values to simulate missing data. We refer to the resulting proportion of unavailable values as the missing ratio r. Specifically, we implement the masking strategy of Zerveas et al. [36] as a two-state Markov process that generates alternating masked and unmasked segments whose lengths follow geometric distributions. The process is parameterised by a target missing ratio r and an average masked segment length $l _ { m }$ , with the corresponding average unmasked segment length given by $\begin{array} { r } { l _ { u } = \frac { 1 - r } { r } l _ { m } ^ { - } } \end{array}$ . During training, we set r = 0.15, while evaluation is conducted at missing ratios of 0.10, 0.30, 0.50, and 0.70, where larger ratios correspond to increasingly challenging imputation settings. For both training and evaluation, we set $l _ { m } = 6$ , as moderately longer contiguous gaps provide a more challenging missingness pattern than short isolated gaps, while preserving sufficient observed context for reconstruction. Table 1 reports the mean and standard deviation of MAE and RMSE values over five random runs for each missingness ratio. We provide the full table of average results for each missingness ratio in Table 7 in Appendix A.5.

Overall, ProCTI outperforms all baselines in terms of average performance. This demonstrates the benefit of combining global prototype priors with local conditioning. Among the baselines, FGTI attains the second-best performance, yielding the second-lowest mean MAE for all scenarios and the second-lowest mean RMSE in 6 out of 10 scenarios. In comparison to FGTI, ProCTI reduces the average MAE by approximately 3–30%. The largest gains are observed on PhysioNet and Stock, followed by Weather, Beijing and Gait.

## 4.3 Robustness under attribute-wise missingness

We implemented attribute-wise missingness scenarios to evaluate robustness under structured missing patterns. This setting is motivated by real-world situations in which one or more attributes are missing for a period of time, for example, due to sensor malfunctions or system outages. Using the same training configuration as in the previous experiments, we conducted two attribute-wise missing protocols: Drop 1 and Drop 2, where one attribute and two attributes, respectively, were randomly selected and fully masked within the test windows. During evaluation, the reverse diffusion process reconstructed the missing attribute(s) solely from the remaining observed variables. For all datasets, we considered all available attributes except PhysioNet, which exhibited high sparsity. For this dataset, only attributes with more than 80% observed values were included in the experiment. Table 2 reports the MAE and RMSE values on the masked attributes for the different methods.

For the Gait, PhysioNet, and Beijing datasets, ProCTI outperforms all competing methods in terms of overall performance, achieving the lowest MAE and RMSE values in the majority of attribute-drop scenarios. FGTI consistently emerges as the second-best performing method. For the Weather and Stock datasets, however, BRITS displays the lowest MAE and RMSE values. Inspecting the feature-wise correlations of the five datasets, we find that BRITS's superior performance in these two cases can be attributed to its explicit feature-regression mechanism combined with bidirectional temporal recurrence. These are particularly effective when there is strong cross-feature redundancy. In the Stock dataset, for example, price variables such as Open, High, Low, Close, and Adj\_Close are almost perfectly correlated, while many variables in the Weather dataset also exhibit linear dependencies and tightly coupled feature groups. Under attribute-drop evaluation, where an entire variable is missing, these relationships allow BRITS to reconstruct the missing channel directly from the remaining observed variables, consequently reducing the relative advantage of more complex diffusion-based or forecasting-adapted models. This pattern matches Proposition 3 in Section 3.2.2: the strong cross-feature redundancy in Stock and Weather approximately satisfies the conditional independence $Y \perp \perp Z \mid O$ , the regime in which prototype-derived global conditioning provably cannot improve upon a sufficiently expressive local predictor. We provide heatmap visualisations of the datasets’ feature-wise correlations in Figure 3.

![](images/b507c53946552a94a52042697f8a655994f5678fb898eefde2c4b72991f00bf5.jpg)  
(a) PhysioNet

![](images/38dcf002f50d4bea241bd503b0631576e18bdbae31d2a9ac4f0fc31aab886d29.jpg)  
(b) Weather

![](images/9cb17560eb87486a16a0b09bf3aca7881929911b411cec8ab712654cdc7e0749.jpg)  
(c) Stock

![](images/8b5079ceed246faa8e6f5e1456063f3c77949b88b447397afe108d4921fb7f61.jpg)  
(d) Gait

![](images/82297514e2b07b70d8dfcb63506db3310209386ebd793391efb1a03650ee48b6.jpg)  
(e) Beijing  
Figure 3: Feature correlation heatmaps for the five datasets: (a) PhysioNet, (b) Weather, (c) Stock, (d) Gait, and (e) Beijing.

In contrast, the Gait and PhysioNet datasets display substantially weaker feature correlations, indicating that missing-attribute recovery cannot rely primarily on simple cross-variable redundancy. For the Beijing dataset, although some strong pairwise correlations are present, its dependency structure is considerably more heterogeneous, containing mixtures of strongly positive, strongly negative, and weak relationships across pollutant, meteorological, and dispersion-related variables. Furthermore, as illustrated in Figure 1, Beijing windows form multiple recurring operating regimes with distinct feature profiles, suggesting that accurate missing-channel reconstruction depends on broader contextual interactions rather than uniform redundancy alone. Under such conditions, the antecedent of Proposition 3 holds: regimes carry residual information about Y beyond $O ,$ and ProCTI's prototype-derived global conditioning enables it to outperform BRITS and the remaining baselines, consistent with the entropy and mean-squared-error reductions established by Lemma 1 and Proposition 2.

Table 3: Ablation study of ProCTI across all datasets (mean values only). Variant A denotes the base FGTI model without prototype conditioning. Variants B-D use fixed α = 0.1, while Variant E uses a learnable α.
<table><tr><td rowspan="2">Variant</td><td rowspan="2">Proto Bank</td><td rowspan="2">Proto Attn</td><td rowspan="2">Learnable α</td><td colspan="2">Gait</td><td colspan="2">PhysioNet</td><td colspan="2">Weather</td><td colspan="2">Beijing</td><td colspan="2">Stock</td></tr><tr><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td></tr><tr><td>A</td><td>一</td><td>一</td><td></td><td>0.322</td><td>0.531</td><td>0.228</td><td>0.460</td><td>0.149</td><td>0.336</td><td>0.1566</td><td>0.4338</td><td>3.2606</td><td>4.0410</td></tr><tr><td>B</td><td>√</td><td>一</td><td></td><td>0.318</td><td>0.521</td><td>0.219</td><td>0.424</td><td>0.113</td><td>0.296</td><td>0.1518</td><td>0.4062</td><td>2.6317</td><td>3.2982</td></tr><tr><td>C</td><td>√</td><td>√</td><td>一</td><td>0.317</td><td>0.520</td><td>0.217</td><td>0.417</td><td>0.119</td><td>0.304</td><td>0.1519</td><td>0.4044</td><td>2.4705</td><td>3.0971</td></tr><tr><td>D</td><td>frozen</td><td>√</td><td>一</td><td>0.316</td><td>0.521</td><td>0.221</td><td>0.434</td><td>0.139</td><td>0.329</td><td>0.1591</td><td>0.4362</td><td>2.9831</td><td>3.7186</td></tr><tr><td>E (inj-dom)</td><td>√</td><td>√</td><td>√</td><td>0.316</td><td>0.518</td><td>0.219</td><td>0.414</td><td>0.096</td><td>0.280</td><td>0.1515</td><td>0.4043</td><td>2.2686</td><td>2.8323</td></tr><tr><td>F (inj-hf)</td><td>√</td><td>√</td><td>√</td><td>0.320</td><td>0.520</td><td>0.214</td><td>0.412</td><td>0.102</td><td>0.288</td><td>0.1910</td><td>0.4746</td><td>3.1198</td><td>3.7978</td></tr><tr><td>G (inj-both)</td><td>√</td><td>√</td><td>√</td><td>0.316</td><td>0.526</td><td>0.223</td><td>0.422</td><td>0.104</td><td>0.289</td><td>0.1938</td><td>0.4752</td><td>2.8556</td><td>3.4772</td></tr></table>

## 4.4 Ablation analysis

We study the influence of the main components introduced in ProCTI, namely the prototype bank the cross-attention interaction between input windows and prototypes, and the scaling parameter α used to inject regime information. We also study the effect of injecting regime information into both dominant and high-frequency branches together, as well as separately. To isolate the contribution of each component, we consider seven variants: (A) the original FGTI baseline without prototype conditioning, (B) adding only a learnable prototype bank, (C) enabling prototype cross-attention, (D) using a frozen prototype bank with attention, and (E - G) the full ProCTI model with a learnable α injected into the dominant frequency branch (inj-dom), high frequency branch (inj-hf), and both branches (inj-both) respectively. Full numerical results and discussion are reported in Table 3.

Across all five datasets, introducing a learnable prototype bank (Variant B) consistently improves upon the FGTI baseline (Variant A), highlighting the benefit of incorporating global prototype priors in addition to local frequency-based conditioning. Enabling prototype cross-attention (Variant C) yields further gains on most datasets, suggesting that dynamically matching each input window to relevant prototypes is more effective than relying on static prototype representations alone. In contrast, Variant D shows degraded performance relative to the preceding variants, indicating that a learnable prototype bank is more effective than a frozen one. Comparing the results across Variants E, F and G reflects that injecting a learnable α into the dominant frequency branch only yields lower error (with the exception of PhysioNet where Variant F returns a lower MAE and RMSE). Overall, Variant E yields the best results in the majority of cases, demonstrating the value of learning α (allowing the model to regulate how strongly regime information influences the local conditioning signal depending on the dataset) and injecting the regime information into the dominant frequency branch only.

## 5 Conclusion

In this paper, we introduced ProCTI, a diffusion-based imputation framework that combines local observations with reusable global priors through prototype retrieval. The proposed framework incorporates a learnable prototype-conditioning module that provides global context at the windowlevel, augmenting the conditioning signals of the reverse diffusion process. We further present a theoretical framework that distinguishes local and global conditioning in diffusion-based imputation models and demonstrate how ProCTI combines both, thereby improving imputation quality. Extensive experiments across diverse real-world datasets demonstrate that ProCTI consistently outperforms strong baselines under both random missingness and attribute-wise missingness settings. The gains of prototype-derived conditioning are conditional on regime structure being informative beyond observed entries (Proposition 3): when Y  Z |O holds approximately, as under strong cross-feature redundancy, simpler regression baselines remain competitive. Future work includes prototype-bank interpretability and designs for non-stationary regimes.

## Acknowledgments and Disclosure of Funding

This research is funded by the Office of National Intelligence, Australia (Agreement No. GA396359), and the National Intelligence and Security Discovery Research Grant (NI240100159).

## References

[1] Max Planck Institute for Biogeochemistry. Jena climate dataset. https://www . bgc-jena. mpg.de/wetter/, 2016. Accessed: 2026-04-16.

[2] Yahoo Finance. Google (goog) historical stock prices. https://finance.yahoo.com/ quote/G00G/history/?p=G00G, 2026. Accessed: 2026-04-16.

[3] Matthew A Reyna, Christopher S Josef, Russell Jeter, Supreeth P Shashikumar, M Brandon Westover, Shamim Nemati, Gari D Clifford, and Ashish Sharma. Early prediction of sepsis from clinical data: the physionet/computing in cardiology challenge 2019. Critical care medicine, 48 (2):210–217, 2020.

[4] Thanh Trung Ngo, Yasushi Makihara, Hajime Nagahara, Yasuhiro Mukaigawa, and Yasushi Yagi. The largest inertial sensor-based gait database and performance evaluation of gait-based personal authentication. Pattern Recognition, 47(1):228–237, 2014.

[5] Ravin Gunawardena, Sandani Jayawardena, Suranga Seneviratne, Rahat Masood, and Salil S Kanhere. Single-sensor sparse adversarial perturbation attacks against behavioral biometrics. IEEE Internet of Things Journal, 11(16):27303–27321, 2024.

[6] Ahmed A Alwan, Allan J Brimicombe, Mihaela Anca Ciupala, Seyed Ali Ghorashi, Andres Baravalle, and Paolo Falcarin. Time-series clustering for sensor fault detection in large-scale cyber-physical systems. Computer Networks, 218:109384, 2022.

[7] Jinghan Du, Minghua Hu, and Weining Zhang. Missing data problem in the monitoring system: A review. IEEE Sensors Journal, 20(23):13984–13998, 2020.

[8] Mehran Amiri and Richard Jensen. Missing data imputation using fuzzy-rough methods. Neurocomputing, 205:152–164, 2016.

[9] Stef Van Buuren and Karin Groothuis-Oudshoorn. mice: Multivariate imputation by chained equations in r. Journal of statistical software, 45:1–67, 2011.

[10] Shuai Liu, Xiucheng Li, Gao Cong, Yile Chen, and Yue Jiang. Multivariate time-series imputation with disentangled temporal representations. In The Eleventh international conference on learning representations, 2023.

[11] Wei Cao, Dong Wang, Jian Li, Hao Zhou, Lei Li, and Yitan Li. Brits: Bidirectional recurrent imputation for time series. Advances in neural information processing systems, 31, 2018.

[12] Tong Nie, Guoyang Qin, Wei Ma, Yuewen Mei, and Jian Sun. Imputeformer: Low ranknessinduced transformers for generalizable spatiotemporal imputation. In Proceedings of the 30th ACM SIGKDD conference on knowledge discovery and data mining, pages 2260–2271, 2024.

[13] Vincent Fortuin, Dmitry Baranchuk, Gunnar Rätsch, and Stephan Mandt. Gp-vae: Deep probabilistic time series imputation. In International conference on artificial intelligence and statistics, pages 1651–1661. PMLR, 2020.

[14] Xinyu Yuan and Yan Qiao. Diffusion-ts: Interpretable diffusion for general time series generation. arXiv preprint arXiv:2403.01742, 2024.

[15] Yusuke Tashiro, Jiaming Song, Yang Song, and Stefano Ermon. Csdi: Conditional score-based diffusion models for probabilistic time series imputation. Advances in neural information processing systems, 34:24804–24816, 2021.

[16] Jianping Zhou, Junhao Li, Guanjie Zheng, Xinbing Wang, and Chenghu Zhou. Mtsci: A conditional diffusion model for multivariate time series consistent imputation. In Proceedings of the 33rd ACM International Conference on Information and Knowledge Management, pages 3474–3483, 2024.

[17] Xinyu Yang, Yu Sun, Xiaojie Yuan, and Xinyang Chen. Frequency-aware generative models for multivariate time series imputation. Advances in Neural Information Processing Systems, 37: 52595–52623, 2024.

[18] Song Chen. Beijing Multi-Site Air Quality. UCI Machine Learning Repository, 2017. DOI: https://doi.org/10.24432/C5RK5G.

[19] David J Bartholomew. Time series analysis forecasting and control., 1971.

[20] Mark W Watson. Vector autoregressions and cointegration. Handbook of econometrics, 4: 2843–2915, 1994.

[21] Wenjie Du, David Côté, and Yan Liu. Saits: Self-attention-based imputation for time series. Expert Systems with Applications, 219:119619, 2023.

[22] Minhao Liu, Ailing Zeng, Muxi Chen, Zhijian Xu, Qiuxia Lai, Lingna Ma, and Qiang Xu. Scinet: Time series modeling and forecasting with sample convolution and interaction. Advances in neural information processing systems, 35:5816–5828, 2022.

[23] Kun Yi, Qi Zhang, Wei Fan, Shoujin Wang, Pengyang Wang, Hui He, Ning An, Defu Lian, Longbing Cao, and Zhendong Niu. Frequency-domain mlps are more effective learners in time series forecasting. Advances in Neural Information Processing Systems, 36:76656–76679, 2023.

[24] Haoyi Zhou, Shanghang Zhang, Jieqi Peng, Shuai Zhang, Jianxin Li, Hui Xiong, and Wancai Zhang. Informer: Beyond efficient transformer for long sequence time-series forecasting. In Proceedings of the AAAI conference on artificial intelligence, volume 35, pages 11106–11115, 2021.

[25] Haixu Wu, Jiehui Xu, Jianmin Wang, and Mingsheng Long. Autoformer: Decomposition transformers with auto-correlation for long-term series forecasting. Advances in neural information processing systems, 34:22419–22430, 2021.

[26] Shizhan Liu, Hang Yu, Cong Liao, Jianguo Li, Weiyao Lin, Alex X Liu, and Schahram Dustdar. Pyraformer: Low-complexity pyramidal attention for long-range time series modeling and forecasting. In International conference on learning representations, 2021.

[27] Yong Liu, Haixu Wu, Jianmin Wang, and Mingsheng Long. Non-stationary transformers: Exploring the stationarity in time series forecasting. Advances in neural information processing systems, 35:9881–9893, 2022.

[28] Yuqi Nie, Nam H Nguyen, Phanwadee Sinthong, and Jayant Kalagnanam. A time series is worth 64 words: Long-term forecasting with transformers. arxiv 2022. arXiv preprint arXiv:2211.14730, 2022.

[29] Yong Liu, Tengge Hu, Haoran Zhang, Haixu Wu, Shiyu Wang, Lintao Ma, and Mingsheng Long. itransformer: Inverted transformers are effective for time series forecasting. arXiv preprint arXiv:2310.06625, 2023.

[30] Yang Li, Han Meng, Zhenyu Bi, Ingolv T Urnes, and Haipeng Chen. Population aware diffusion for time series generation. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 18520–18529, 2025.

[31] Mingzhe Liu, Han Huang, Hao Feng, Leilei Sun, Bowen Du, and Yanjie Fu. Pristi: A conditional diffusion framework for spatiotemporal imputation. In 2023 IEEE 39th international conference on data engineering (ICDE), pages 1927–1939. IEEE, 2023.

[32] Hangtian Wang and Mahito Sugiyama. Time-gated multi-scale flow matching for time-series imputation. In The Fourteenth International Conference on Learning Representations

[33] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. Advances in neural information processing systems, 34:8780–8794, 2021.

[34] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022.

[35] Wenjie Du, Jun Wang, Linglong Qian, Yiyuan Yang, Zina Ibrahim, Fanxing Liu, Zepu Wang, Haoxin Liu, Zhiyuan Zhao, Yingjie Zhou, et al. Tsi-bench: Benchmarking time series imputation. arXiv preprint arXiv:2406.12747, 2024.

[36] George Zerveas, Srideepika Jayaraman, Dhaval Patel, Anuradha Bhamidipaty, and Carsten Eickhoff. A transformer-based framework for multivariate time series representation learning. In Proceedings of the 27th ACM SIGKDD conference on knowledge discovery & data mining, pages 2114–2124, 2021.

[37] Bin Li, Carsten Jentsch, and Emmanuel Müller. Prototypes as explanation for time series anomaly detection. arXiv preprint arXiv:2307.01601, 2023.

[38] Zhi-Hao Yu, Lian-Tao Ma, Ya-Sha Wang, and Xu Chu. Imputation with inter-series information from prototypes for healthcare time series. Journal of Computer Science and Technology, 40(6): 1499–1511, 2025.

[39] Yu-Hao Huang, Chang Xu, Yueying Wu, Wu-Jun Li, and Jiang Bian. Timedp: Learning to generate multi-domain time series with domain prompts. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 17520–17527, 2025.

[40] Junwei Deng, Chang Xu, Jiaqi W Ma, Ming Jin, Chenghao Liu, and Jiang Bian. Oats: Online data augmentation for time series foundation models. arXiv preprint arXiv:2601.19040, 2026.

[41] Yuhang Duan, Lin Lin, and Xiaoshuai Wu. K-protodiff: Key prototypes-guided diffusion for time series generation. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pages 20959–20967, 2026.

[42] Kamile Stankeviciute, Ahmed M Alaa, and Mihaela Van der Schaar. Conformal time-series forecasting. Advances in neural information processing systems, 34:6216–6228, 2021.

## A Appendix

## A.1 Related work

Prototype-based representations in time series modelling Prototype-based representations have appeared across the broader time-series literature. In anomaly detection, ProtoAD [37] uses prototypes to characterise regular latent patterns and improve interpretability through example-based explanations. In healthcare imputation, PRIME [38] leverages prototype memory to capture interseries information from similar patients within irregular clinical records. In generation, recent works such as TimeDP [39], OATS [40], and K-ProtoDiff [41] employ prototypes as domain prompts, guidance signals, or key-pattern constraints for synthesising realistic sequences. In contrast, ProCTI introduces prototypes for diffusion-based conditional imputation of general multivariate time series, where prototypes are learned as reusable dataset-level regimes and explicitly combined with local observations during reverse diffusion. This differs from prior prototype methods that focus on retrieval-style inter-series memory, interpretability, or unconditional generation rather than principled global conditioning for imputation.

## A.2 ProCTI implementation details

## A.2.1 Multi-head cross-attention details

We provide here the full multi-head cross-attention computation summarised in Section 3.1. Let the number of heads be H, with per-head dimension $d _ { h } \doteq D / H$ . For each head $h \in \{ 1 , \ldots , H \}$ we have learnable projections ${ W } _ { Q } ^ { ( \mathit { \hat { h } } ) } , { W } _ { K } ^ { ( \mathit { h } ) } , { W } _ { V } ^ { ( \mathit { h } ) } \in \mathbb { R } ^ { d _ { \mathit { h } } \times D }$ , and the projected query, key, and value vectors are

$$
\begin{array} { r } { q _ { b } ^ { ( h ) } = W _ { Q } ^ { ( h ) } q _ { b } , \quad k _ { m } ^ { ( h ) } = W _ { K } ^ { ( h ) } p _ { m } , \quad v _ { m } ^ { ( h ) } = W _ { V } ^ { ( h ) } p _ { m } , } \end{array}
$$

with $q _ { b } ^ { ( h ) } , k _ { m } ^ { ( h ) } , v _ { m } ^ { ( h ) } \in \mathbb { R } ^ { d _ { h } }$ . The attention weights are

$$
\pi _ { b , m } ^ { ( h ) } = \frac { \exp \Bigl ( ( q _ { b } ^ { ( h ) } ) ^ { \top } k _ { m } ^ { ( h ) } / \sqrt { d _ { h } } \Bigr ) } { \sum _ { j = 1 } ^ { M } \exp \Bigl ( ( q _ { b } ^ { ( h ) } ) ^ { \top } k _ { j } ^ { ( h ) } / \sqrt { d _ { h } } \Bigr ) } ,
$$

yielding the per-head context vector $\begin{array} { r } { c _ { b } ^ { ( h ) } = \sum _ { m = 1 } ^ { M } \pi _ { b , m } ^ { ( h ) } v _ { m } ^ { ( h ) } \in \mathbb { R } ^ { d _ { h } } } \end{array}$ . The outputs of all heads are concatenated and projected to obtain $c _ { b } = W _ { O }$ Concat $\left( c _ { b } ^ { ( 1 ) } , \ldots , c _ { b } ^ { ( H ) } \right) \in \mathbb { R } ^ { D }$ with $W _ { O } \in \mathbb { R } ^ { D \times D }$ The regime vector is then $r _ { b } = W _ { r } c _ { b } \in \mathbb { R } ^ { K }$ as in Section 3.1.

## A.2.2 Training and imputation algorithms

We provide the training and imputation algorithms for our proposed model ProCTI, in Algorithm 1 and Algorithm 2 respectively.

Algorithm 1 Training process of ProCTI   
Input: Time series X, keep-mask M, prototype bank P, diffusion steps $T$   
Output: Trained $\epsilon _ { \theta } ( \cdot )$ and prototype module   
1: repeat   
2: $\hat { X } _ { 0 } \gets X \odot ( 1 - M ) , \quad X ^ { C } \gets X \odot M$   
3: $h ^ { h f } , h ^ { d o m } \gets \mathrm { F r e q F i l t e r } ( X ^ { C } )$   
4: $q _ { b } \gets \mathrm { Q u e r y } ( X ^ { C } , \mathbf { \bar { \xi } } M )$   
5: $\bar { c } _ { b } \gets \mathrm { C r o s s A t t n } ( q _ { b } , P ) , \quad r _ { b } \gets W _ { r } c _ { b }$   
6: $h _ { b , t , k } ^ { d o m + r } \gets h _ { b , t , k } ^ { d o m } + \alpha r _ { b , k }$   
7: $t \sim \mathrm { U n i f o r m } ( \{ 1 , \dots , T \} ) , \quad \epsilon \sim \mathcal { N } ( 0 , I )$   
8: $\hat { X } _ { t } \gets \sqrt { \bar { \alpha } _ { t } } \hat { X } _ { 0 } + \sqrt { 1 - \bar { \alpha } _ { t } } \epsilon$   
9: $\hat { \epsilon } \gets \epsilon _ { \theta } \big ( \hat { X } _ { t } , t \mid X ^ { C } , h ^ { h f } , h ^ { d o m + r } \big )$   
10: Update parameters using $\mathcal { L } _ { \mathrm { d i f f } } = \stackrel { \prime } { \Vert \epsilon - \hat { \epsilon } \Vert _ { 2 } ^ { 2 } }$   
11: until converged

Algorithm 2 Imputation process of ProCTI   
Input: Incomplete time series $X ,$ keep-mask M, trained prototype bank P, trained denoising   
network $\epsilon _ { \theta } ( \cdot )$ , diffusion steps $T$   
Output: Imputed time series $\tilde { X }$   
1: $\mathbf { \dot { \chi } } ^ { C } \xleftarrow { } \mathbf { \dot { \chi } } \odot M$   
2: $h ^ { h f } , h ^ { d o m } \gets \mathrm { F r e q F i l t e r } ( X ^ { C } )$   
3: $h ^ { d o m + r } \gets \mathrm { P r o t o } \hat { \mathrm { C o n d } } ( \dot { X } ^ { C } , \dot { M } , P , h ^ { d o m } )$   
4: $\hat { X } _ { T } \sim \mathcal { N } ( 0 , I )$   
5: for $t = T , T - 1 , \dots , 1$ do   
6: $\hat { \epsilon } \gets \epsilon _ { \theta } \big ( \hat { X } _ { t } , t \mid X ^ { C } , h ^ { h f } , h ^ { d o m + r } \big )$   
7: Obtain $\hat { X } _ { t - 1 }$ from $p _ { \theta } ( \hat { X } _ { t - 1 } \mid \hat { X } _ { t } , X ^ { C } , h ^ { h f } , h ^ { d o m + r } )$   
8: end for   
9: $\tilde { X }  X \odot M + \hat { X } _ { 0 } \odot ( 1 - M )$   
10: return $\tilde { X }$

## A.3 Dataset preparation

In this section, we provide a detailed description of the dataset preprocessing steps to create training, validation and test splits for our evaluation experiments. Since the Gait, PhysioNet, and Beijing datasets contain identifiable entities that can be used for partitioning (user IDs, patient IDs, and monitoring stations, respectively), we construct training, validation, and test splits such that each entity appears in only one split, preventing information leakage. This strategy differs from conventional approaches that randomly partition data at the window level, and is therefore designed to evaluate the ability of ProCTI and baseline methods to generalise across non-overlapping entities by capturing global patterns.

Specifically, for the Gait and PhysioNet datasets, we adopt user-disjoint and patient-wise splits, respectively, ensuring that all sequences associated with a given subject or patient are confined to a single partition. The training, validation, and test splits follow a 70:15:15 ratio at the entity level. For the Beijing dataset, we use a station-disjoint split, partitioning the 12 monitoring stations into training, validation, and test groups in an 8:2:2 ratio, and constructing windows independently within each station. For the Weather and Stock datasets, we adopt a chronological split along the temporal dimension, using a 70:15:15 ratio for training, validation, and testing. In all cases, nonoverlapping fixed-length windowing is applied after splitting to generate sequences for model training and evaluation.

## A.4 Supplementary experiments

## A.4.1 Probabilistic evaluation

Table 4: Probabilistic evaluation on the test split across missingness ratios on PhysioNet data.
<table><tr><td>Metric</td><td>Missing</td><td>CSDI</td><td>Diffusion-TS</td><td>FGTI</td><td>PaD-TS</td><td>MTSCI</td><td>ProCTI</td></tr><tr><td rowspan="4">CRPS(↓)</td><td>0.10</td><td>0.1798</td><td>0.6579</td><td>0.1187</td><td>0.8727</td><td>0.4076</td><td>0.1286</td></tr><tr><td>0.30</td><td>0.1867</td><td>0.8361</td><td>0.1362</td><td>0.8955</td><td>0.5980</td><td>0.1390</td></tr><tr><td>0.50</td><td>0.2223</td><td>0.8485</td><td>0.1647</td><td>1.0370</td><td>0.6081</td><td>0.1593</td></tr><tr><td>0.70</td><td>0.2284</td><td>0.7599</td><td>0.2266</td><td>0.9718</td><td>0.7307</td><td>0.2040</td></tr><tr><td rowspan="4">PI90 Coverage( 90%)</td><td>0.10</td><td>0.8447</td><td>0.0368</td><td>0.8479</td><td>0.1383</td><td>0.8053</td><td>0.8467</td></tr><tr><td>0.30</td><td>0.8460</td><td>0.0514</td><td>0.9097</td><td>0.2291</td><td>0.7806</td><td>0.8876</td></tr><tr><td>0.50</td><td>0.8566</td><td>0.0734</td><td>0.9307</td><td>0.3375</td><td>0.7612</td><td>0.9017</td></tr><tr><td>0.70</td><td>0.8666</td><td>0.1082</td><td>0.9265</td><td>0.4910</td><td>0.7368</td><td>0.8872</td></tr><tr><td rowspan="4">PI90 Width(↓)</td><td>0.10</td><td>0.5861</td><td>0.0647</td><td>0.5884</td><td>0.2509</td><td>1.2602</td><td>0.5704</td></tr><tr><td>0.30</td><td>0.6368</td><td>0.0806</td><td>0.7693</td><td>0.4349</td><td>1.4165</td><td>0.7012</td></tr><tr><td>0.50</td><td>0.7264</td><td>1.1117</td><td>1.0723</td><td>0.6777</td><td>1.5938</td><td>0.9137</td></tr><tr><td>0.70</td><td>0.8134</td><td>0.1625</td><td>1.5036</td><td>1.0513</td><td>1.7813</td><td>1.1878</td></tr></table>

To complement the imputation evaluation reported in the main experiments (Section 4), we further assess the probabilistic quality of the predictive distributions produced by diffusion-based models. Using the PhysioNet and Beijing datasets under the Markov masking protocol, we adopt the Continuous Ranked Probability Score (CRPS) [15, 17], and 90% prediction interval coverage (PI90 coverage) and width (PI90 width) [42] across different missingness ratios. CRPS evaluates the overall quality of the predictive distribution in comparison to the ground truth, where lower values are preferred. PI90 coverage measures how often the ground-truth values fall within the predicted 90% interval. Values closer to 0.90 are preferred, meaning that well-calibrated predictive intervals are neither too narrow nor excessively wide. Finally, PI90 width reflects interval sharpness, where narrower intervals are preferred, provided calibration remains close to the nominal 90% level. Table 4 and Table 5 show the results for PhysioNet and Beijing, respectively.

Table 5: Probabilistic evaluation on the test split across missingness ratios on Beijing data
<table><tr><td>Metric</td><td>Missing</td><td>CSDI</td><td>Diffusion-TS</td><td>FGTI</td><td>MTSCI</td><td>PaD-TS</td><td>ProCTI</td></tr><tr><td rowspan="4">CRPS(↓)</td><td>0.10</td><td>0.1646</td><td>0.2368</td><td>0.1204</td><td>0.2286</td><td>0.6141</td><td>0.1185</td></tr><tr><td>0.30</td><td>0.1700</td><td>0.2326</td><td>0.1282</td><td>0.2374</td><td>0.5920</td><td>0.1254</td></tr><tr><td>0.50</td><td>0.1894</td><td>0.2393</td><td>0.1472</td><td>0.2639</td><td>0.5784</td><td>0.1424</td></tr><tr><td>0.70</td><td>0.2119</td><td>0.2400</td><td>0.1679</td><td>0.2861</td><td>0.5657</td><td>0.1617</td></tr><tr><td rowspan="4">PI90 Coverage( 90%)</td><td>0.10</td><td>0.8439</td><td>0.3016</td><td>0.8152</td><td>0.7506</td><td>0.1611</td><td>0.8536</td></tr><tr><td>0.30</td><td>0.8479</td><td>0.3478</td><td>0.8279</td><td>0.7657</td><td>0.2946</td><td>0.8748</td></tr><tr><td>0.50</td><td>0.8536</td><td>0.3950</td><td>0.8353</td><td>0.7825</td><td>0.4410</td><td>0.8925</td></tr><tr><td>0.70</td><td>0.8679</td><td>0.4417</td><td>0.8301</td><td>0.8177</td><td>0.5870</td><td>0.9057</td></tr><tr><td rowspan="4">PI90 Width(↓)</td><td>0.10</td><td>0.8081</td><td>0.1771</td><td>0.5441</td><td>0.8911</td><td>0.2970</td><td>0.6532</td></tr><tr><td>0.30</td><td>0.8539</td><td>0.2057</td><td>0.5857</td><td>1.0115</td><td>0.5531</td><td>0.7235</td></tr><tr><td>0.50</td><td>0.9440</td><td>0.2409</td><td>0.6544</td><td>1.1976</td><td>0.8731</td><td>0.8427</td></tr><tr><td>0.70</td><td>1.1174</td><td>0.2789</td><td>0.7461</td><td>1.4426</td><td>1.2630</td><td>1.0122</td></tr></table>

Overall ProCTI consistently achieves strong probabilistic performance across all missingness ratios, obtaining the best or second-best CRPS values while maintaining PI90 Coverage close to 90%. In contrast, some baselines such as Diffusion-TS and PaD-TS, produce narrow intervals but reveal severe under-coverage, indicating overconfident uncertainty estimates. Although FGTI achieves the lowest CRPS in some settings under PhysioNet, it generally requires wider prediction intervals to do so. Overall, ProCTI provides the most balanced trade-off between predictive accuracy, calibration, and sharpness, demonstrating that incorporating global prototype priors improves imputation performance as well as probabilistic reliability.

## A.4.2 Resource consumption

We evaluate the resource consumption of ProCTI against all baselines in Figure 4. ProCTI achieves runtime comparable to FGTI and lower than other diffusion-based methods such as CSDI and MTSCI. Although its GPU memory usage is higher than most baselines, it remains similar to FGTI despite the added model complexity due to the prototype conditioning module. Given its overall superior imputation performance, we argue that this additional resource cost is justified.

![](images/cb93e10a0c1a305004c15f339d8831f5c9d04ef75118078cedcfd585ed404321.jpg)  
Figure 4: Runtime and peak GPU memory usage for different models on the Beijing dataset under the 30% missingness test setting.

## A.4.3 Hyperparameter evaluation

In this section, we present a hyperparameter sensitivity analysis investigating the effect of prototype bank size and the regime-vector scaling parameter α. Table 6 and Figure 5 report ProCTI's imputation performance on held-out validation data, averaged across missingness ratios (10%, 30%, 50%, and 70%), for different values of these hyperparameters. Table 6shows that a prototype bank size of 32 yields lower error in the majority of cases across MAE and RMSE (6/10 cells). Figure 5 indicates that $\alpha = 0 . 1$ wins on Beijing, Gait, and Stock, while $\alpha = 0 . 3$ gives lower MAE on PhysioNet and Weather. We therefore use $\alpha = 0 . 1$ as it gives the best overall performance, and a prototype bank size of 32 in ProCTI.

Table 6: Effect of prototype bank size (number of prototypes) on imputation performance. Best results for each dataset and metric are highlighted in bold.
<table><tr><td>Metric</td><td>Prototype bank size</td><td>Beijing</td><td>Gait</td><td>PhysioNet</td><td>Stock</td><td>Weather</td></tr><tr><td rowspan="3">MAE</td><td>16</td><td>0.200</td><td>0.310</td><td>0.222</td><td>0.894</td><td>0.102</td></tr><tr><td>32</td><td>0.158</td><td>0.310</td><td>0.224</td><td>0.835</td><td>0.118</td></tr><tr><td>64</td><td>0.202</td><td>0.306</td><td>0.216</td><td>1.030</td><td>0.095</td></tr><tr><td rowspan="3">RMSE</td><td>16</td><td>0.481</td><td>0.519</td><td>0.416</td><td>1.187</td><td>0.450</td></tr><tr><td>32</td><td>0.412</td><td>0.510</td><td>0.438</td><td>1.118</td><td>0.387</td></tr><tr><td>64</td><td>0.482</td><td>0.512</td><td>0.413</td><td>1.340</td><td>0.434</td></tr></table>

![](images/5a934f9cdb4359872cacce88d5a0d7834aba5e6f9710dc318e42c8c7bdacda68.jpg)

![](images/1cedc60027f4c456df12269d1721973625a1a9dfbff300aed1cce1f5fc6359c9.jpg)  
Figure 5: Effect of scaling parameter $\alpha \in \{ 0 . 1 , 0 . 3 \}$ on imputation performance. Lower values indicate better imputation performance.

## A.5 Supplementary results

## A.5.1 Imputation effectiveness

In Table 7, we report the detailed imputation performance of all methods under the Markov masking protocol, using MAE and RMSE at each missingness ratio, averaged over five repeated runs. Overall ProCTI achieves the lowest errors in the majority of evaluated scenarios, while FGTI consistently delivers the second-best performance in most cases.

## A.5.2 Statistical significance of imputation effectiveness

As the imputation performance of ProCTI and FGTI were very similar on some datasets such as Gait and Beijing, we statistically analysed the significance of the gains yielded by ProCTI. We conducted paired t-tests and Wilcoxon signed-rank tests across the 5 seeds for each missing ratio, with the one-sided hypothesis that ProCTI's per-seed error is lower than FGTI's (i.e., FGTI - ProCTI > 0). The paired t-test assumes the per-seed differences are approximately normally distributed, while the Wilcoxon signed-rank test is a non-parametric alternative that does not require this assumption and is more robust with small sample sizes. Results are shown in Table 8 (significance at the 0.05 shown in bold).

On Beijing, ProCTI significantly outperforms FGTI $( \mathtt { p } < 0 . 0 5 )$ in 4/8 cells (t-test) and 1/8 (Wilcoxon). On Gait, ProCTI is significantly better in 5/8 cells under both tests, concentrated at $r \in { 0 . 1 , 0 . 3 , 0 . 5 , 0 . 7 } ;$ significance is lost at $\mathrm { r } = 0 . 7 $ , consistent with the smaller gains in Table 1. These tests are based on 5 seeds which is a small sample for reliably estimating normality or the

Table 7: Comparison of imputation effectiveness across baseline methods. Best results are shown in bold and second-best results are underlined.
<table><tr><td>Dataset</td><td>Metric</td><td>Missing</td><td>BRITS</td><td>CSDI</td><td>SCINet</td><td>TIDER</td><td>MTSCI</td><td>DiffusionTS</td><td>iTransformer</td><td>FGTI</td><td>PaD-TS</td><td>ProCTI</td></tr><tr><td rowspan="7">Gait</td><td rowspan="3">MAE</td><td>0.1</td><td>0.324</td><td>0.436</td><td>0.436</td><td>0.749</td><td>0.484</td><td>0.358</td><td>0.501</td><td>0.307</td><td>0.906</td><td>0.288</td></tr><tr><td>0.3</td><td>0.342</td><td>0.438</td><td>0.454</td><td>0.744</td><td>0.488</td><td>0.361</td><td>0.521</td><td>0.317</td><td>0.928</td><td>0.299</td></tr><tr><td>0.5 0.7</td><td>0.360</td><td>0.449</td><td>0.482</td><td>0.745</td><td>0.505</td><td>0.369</td><td>0.544</td><td>0.336</td><td>0.954</td><td>0.317</td></tr><tr><td rowspan="3"></td><td></td><td>0.383</td><td>0.473</td><td>0.528</td><td>0.745</td><td>0.527</td><td>0.380</td><td>0.583</td><td>0.366</td><td>0.981</td><td>0.347</td></tr><tr><td>0.1</td><td>0.491</td><td>0.717</td><td>0.615</td><td>0.989</td><td>0.734</td><td>0.573</td><td>0.722</td><td>0.530</td><td>1.162</td><td>0.477</td></tr><tr><td>0.3</td><td>0.518</td><td>0.715</td><td>0.645</td><td>0.982</td><td>0.737</td><td>0.575</td><td>0.749</td><td>0.543</td><td>1.184</td><td>0.497</td></tr><tr><td rowspan="3">RMSE</td><td>0.5</td><td>0.545</td><td>0.726</td><td>0.682</td><td>0.984</td><td>0.755</td><td>0.588</td><td>0.774</td><td>0.570</td><td>1.208</td><td>0.528</td></tr><tr><td>0.7</td><td>0.577</td><td>0.746</td><td>0.741</td><td>0.984</td><td>0.775</td><td>0.604</td><td>0.814</td><td>0.610</td><td>1.235</td><td>0.572</td></tr><tr><td>0.1</td><td>0.486</td><td>0.426</td><td>0.327</td><td>0.740</td><td>0.456</td><td>0.514</td><td>0.324</td><td>0.273</td><td>0.794</td><td>0.203</td></tr><tr><td rowspan="8">PhysioNet</td><td rowspan="8">MAE</td><td>0.3</td><td>0.481</td><td>0.427</td><td>0.339</td><td>0.796</td><td>0.473</td><td>0.574</td><td>0.353</td><td>0.322</td><td>0.837</td><td>0.223</td></tr><tr><td>0.5</td><td>0.504</td><td>0.480</td><td>0.416</td><td>0.730</td><td>0.556</td><td>0.555</td><td>0.432</td><td>0.397</td><td>0.897</td><td>0.278</td></tr><tr><td>0.7</td><td>0.551</td><td>0.489</td><td>0.524</td><td>0.737</td><td>0.617</td><td>0.534</td><td>0.543</td><td>0.459</td><td>0.934</td><td>0.332</td></tr><tr><td>0.1</td><td>3.245</td><td>4.758</td><td>1.355</td><td>2.996</td><td>1.868</td><td>2.023</td><td>1.484</td><td>2.446</td><td>2.404</td><td>1.182</td></tr><tr><td>0.3</td><td>3.372</td><td>4.789</td><td>1.083</td><td>4.274</td><td>1.355</td><td>3.538</td><td>1.540</td><td>2.841</td><td>3.076</td><td>1.074</td></tr><tr><td>RMSE 0.5</td><td>3.366</td><td>5.203</td><td>1.679</td><td>3.264</td><td>1.623</td><td>3.413</td><td>2.120</td><td>3.459</td><td>3.609</td><td>1.657</td></tr><tr><td>0.7</td><td>3.622</td><td>4.928</td><td>2.111</td><td>3.324</td><td>1.590</td><td>3.242</td><td>2.554</td><td>3.135</td><td>3.524</td><td>1.621</td></tr><tr><td>0.1</td><td>0.105</td><td>0.109</td><td>0.199</td><td>0.641</td><td>0.160</td><td>0.297</td><td>0.173</td><td>0.073</td><td>0.163</td><td>0.062</td></tr><tr><td rowspan="8">Weather</td><td>MAE</td><td></td><td>0.124</td><td>0.120</td><td>0.226</td><td>0.651</td><td>0.221</td><td>0.249</td><td>0.215</td><td>0.085</td><td>0.169</td><td>0.076</td></tr><tr><td></td><td>0.5</td><td>0.155</td><td>0.125</td><td>0.289</td><td>0.653</td><td>0.307</td><td>0.227</td><td>0.309</td><td>0.111</td><td>0.177</td><td>0.097</td></tr><tr><td></td><td>0.7</td><td>0.209</td><td>0.144</td><td>0.389</td><td>0.651</td><td>0.425</td><td>0.213</td><td>0.435</td><td>0.182</td><td>0.193</td><td>0.146</td></tr><tr><td></td><td>0.1</td><td>0.315</td><td>0.309</td><td>0.343</td><td>0.821</td><td>0.338</td><td>0.570</td><td>0.346</td><td>0.295</td><td>0.338</td><td>0.230</td></tr><tr><td>RMSE</td><td>0.3</td><td>0.337</td><td>0.333</td><td>0.382</td><td>0.838</td><td>0.424</td><td>0.498</td><td>0.369</td><td>0.268</td><td>0.362</td><td>0.266</td></tr><tr><td></td><td>0.5</td><td>0.357</td><td>0.335</td><td>0.452</td><td>0.837</td><td>0.508</td><td>0.465</td><td>0.462</td><td>0.281</td><td>0.368</td><td>0.276</td></tr><tr><td></td><td>0.7</td><td>0.410</td><td>0.359</td><td>0.558</td><td>0.836</td><td>0.616</td><td>0.441</td><td>0.591</td><td>0.346</td><td>0.379</td><td>0.326</td></tr><tr><td></td><td>0.1</td><td>0.141</td><td>0.096</td><td>0.332</td><td>0.721</td><td>0.152</td><td>0.225</td><td>0.285</td><td>0.105</td><td>0.619</td><td>0.103</td></tr><tr><td rowspan="8">Beijing</td><td>MAE</td><td></td><td>0.177</td><td>0.130</td><td>0.267</td><td>0.722</td><td>0.212</td><td>0.243</td><td>0.334</td><td>0.130</td><td>0.667</td><td>0.128</td></tr><tr><td></td><td>0.3 0.5</td><td>0.243</td><td>0.217</td><td>0.331</td><td>0.722</td><td>0.331</td><td>0.268</td><td>0.439</td><td>0.176</td><td>0.720</td><td>0.169</td></tr><tr><td></td><td>0.7</td><td>0.306</td><td>0.299</td><td>0.435</td><td>0.722</td><td>0.435</td><td>0.288</td><td>0.562</td><td>0.225</td><td>0.790</td><td>0.214</td></tr><tr><td></td><td>0.1</td><td>0.397</td><td>0.340</td><td>0.512</td><td>0.999</td><td>0.419</td><td>0.554</td><td>0.515</td><td>0.363</td><td>0.899</td><td>0.337</td></tr><tr><td></td><td>0.3</td><td>0.449</td><td>0.400</td><td>0.485</td><td>1.007</td><td>0.519</td><td>0.578</td><td>0.590</td><td>0.406</td><td>0.959</td><td>0.381</td></tr><tr><td>RMSE</td><td>0.5</td><td>0.508</td><td>0.508</td><td>0.562</td><td>1.004</td><td>0.636</td><td>0.589</td><td>0.701</td><td>0.452</td><td>1.011</td><td>0.438</td></tr><tr><td></td><td>0.7</td><td>0.574</td><td>0.609</td><td>0.677</td><td>1.003</td><td>0.733</td><td>0.606</td><td>0.832</td><td>0.526</td><td>1.084</td><td>0.499</td></tr><tr><td></td><td>0.1</td><td>3.582</td><td>1.875</td><td>1.729</td><td>5.278</td><td>3.144</td><td>3.810</td><td>1.520</td><td>2.678</td><td>4.773</td><td>1.348</td></tr><tr><td rowspan="8">Stock</td><td>MAE</td></table>

sign/rank distribution. This likely explains why some cells miss significance despite ProCTI's lower mean error in Table 1. We expect significance to hold more consistently with additional seeds.

Table 8: Paired one-sided significance tests for ProCTI vs. FGTI across 5 seeds $( H _ { 1 } ;$ ProCTI error $<$ FGTI error). p-values below 0.05 are in bold. With $n = 5$ , the minimum attainable Wilcoxon p-value is $1 / 3 2 \approx 0 . 0 3 1$
<table><tr><td>Dataset</td><td>r</td><td>Metric</td><td>t-stat</td><td>t-test p</td><td>Wilcoxon stat</td><td>Wilcoxon p</td></tr><tr><td rowspan="6">Beijing</td><td>0.1</td><td>MAE</td><td>-0.306</td><td>0.388</td><td>5</td><td>0.313</td></tr><tr><td></td><td>RMSE</td><td>-2.590</td><td>0.030</td><td>1</td><td>0.063 0.031</td></tr><tr><td>0.3</td><td>MAE RMSE</td><td>-3.111 -2.156</td><td>0.018 0.049</td><td>0 2</td><td>0.094</td></tr><tr><td>0.5</td><td>MAE RMSE</td><td>-1.950 -2.677</td><td>0.062 0.028</td><td>1</td><td>0.063</td></tr><tr><td></td><td>MAE</td><td>-1.049</td><td></td><td>1</td><td>0.063</td></tr><tr><td>0.7</td><td>RMSE</td><td>-0.422</td><td>0.177 0.347</td><td>6 8</td><td>0.406 0.594</td></tr><tr><td rowspan="6">Gait</td><td>0.1</td><td>MAE</td><td>-3.906</td><td>0.009</td><td>0</td><td>0.031</td></tr><tr><td></td><td>RMSE MAE</td><td>-4.500 -7.160</td><td>0.005</td><td>0</td><td>0.031</td></tr><tr><td>0.3</td><td>RMSE</td><td>-1.814</td><td>0.001 0.072</td><td>0 3</td><td>0.031 0.156</td></tr><tr><td>0.5</td><td>MAE</td><td>-3.780</td><td>0.010</td><td>0</td><td>0.031</td></tr><tr><td></td><td>RMSE</td><td>-3.177</td><td>0.017</td><td>0</td><td>0.031</td></tr><tr><td>0.7</td><td>MAE RMSE</td><td>-0.774 -1.396</td><td>0.241 0.118</td><td>4 2</td><td>0.219 0.094</td></tr></table>

## A.5.3 Prototype utilisation

We analysed how many of the 32 prototypes were actually active at test time by computing the attention weight assigned to each of the 32 prototypes across the entire test sets' windows. We report two summary statistics: (1) the number of prototypes individually receiving more than half of a uniform share of attention (i.e., average weight above 1/(2×32)); and (2) the number of prototypes needed to jointly account for 90% of total attention mass. We present the results for missingness ratio 0.10 (seed 1) across all five datasets in Table 9 below.

We find that prototype utilisation is broad rather than concentrated on a small interpretable subset for each dataset. At least 31 of the 32 prototypes individually receive more than a half-share of attention, and between 27 and 29 of the 32 prototypes are needed to jointly account for 90% of attention mass. This suggests the prototype bank is not over-provisioned at bank size 32. Stock and Gait are marginally more selective under the 90%-cumulative measure (27 and 28 active prototypes respectively) than Weather, PhysioNet, and Beijing (29 each).

Table 9: Number of active prototypes per dataset under the usage threshold and the 90% cumulative usage criterion.
<table><tr><td>Dataset</td><td> $n _ { \mathrm { a c t i v e } }$  (threshold)</td><td> $n _ { \mathrm { a c t i v e } }$  (90% cumulative)</td></tr><tr><td>Stock</td><td>31</td><td>27</td></tr><tr><td>Weather</td><td>32</td><td>29</td></tr><tr><td>PhysioNet</td><td>32</td><td>29</td></tr><tr><td>Beijing</td><td>32</td><td>29</td></tr><tr><td>Gait</td><td>32</td><td>28</td></tr></table>

## A.6 Visualisations

In Figure 6, Figure 7, and Figure 8, we present representative imputation examples for the Beijing, Gait, and PhysioNet datasets, comparing ProCTI, FGTI, and CSDI. Across all three datasets, CSDI imputations are more prone to deviating from the ground-truth trajectories, whereas ProCTI follows the true values within the masked regions more closely. ProCTI and FGTI often produce visually similar reconstructions, reflecting their shared frequency-aware diffusion conditioning framework. Nonetheless, closer inspection of several masked segments, together with the results in Table 1 and Table 7, shows that ProCTI consistently attains lower overall errors than FGTI, demonstrating the benefit of incorporating global conditioning signals.

Finally, Figure 9 illustrates representative examples of prototype usage on window samples from the Stock dataset. Specifically, the figure shows the raw time series, the attention weights assigned by each window to the prototypes, the corresponding regime vector $r _ { b }$ values, and the resulting adjustments to the dominant-frequency component of the three attributes with the largest absolute $r _ { b }$ values.

## A.7 Broader impact statement

The positive societal impacts of this work are directly related to time-series-data applications where missing or incomplete data are common, and improved imputation can positively affect downstream tasks. Example domains include healthcare, transportation, environmental monitoring, and industrial systems, where sensor readings or records may be missing or faulty. In healthcare, for instance, better recovery of missing clinical measurements may assist patient monitoring or diagnosis. On the other hand, potential unintended consequences in downstream applications due to incorrect imputations should also be considered. In such settings, technologies such as ProCTI should be used as decision-support tools rather than as replacements for human judgment, particularly in high-stakes scenarios.

![](images/3d8b4317b00ddd5e4a59e6d489984a223fd6a54c9da74fa0545de437e1d80072.jpg)  
(b) ProCTI vs CSDI  
Figure 6: Representative imputation comparison on the Beijing dataset.

![](images/144053969aec48a91f26ef98572397032e0b7f5439d0df2c7d1dfd72a549436e.jpg)  
(b) ProCTI vs CSDI  
Figure 7: Representative imputation comparison on the Gait dataset.

![](images/446e1ee334c7d683f54a85e77745acbe3927fff1c948839c775a52dae079625a.jpg)  
(b) ProCTI vs CSDI  
Figure 8: Representative imputation comparison on the PhysioNet dataset.

![](images/3372e0606a047fb74e68c535dce5c0c2db279ee39f97edeb92e15039afc7be8a.jpg)

![](images/fefa38cdb0e0612febc885b60d54403e2aa22233f2b990790a0c4456100a4af5.jpg)

![](images/c05270485c2534e803fee73c187258065ce46d3cdc72e4e0e21adf1b88553ef1.jpg)

![](images/21228321989e0e4b477c9ac7b5db5f5e4748abacf789db96d696c146f010b309.jpg)

![](images/d51ac5453f5704deb895e6fcf760e6b49451aa406f96c2cc85bee1ffdaff4170.jpg)

![](images/f27b23b973ba9b1d0dc7b26cddf76bf7a24d2a3ba2f8fbbca6a1e2079ea604ca.jpg)  
(a) Window 0, Top-3 lrbl channels: [2,1,4]

![](images/dfb7edb0872bcbb1d088ac7fae1a6f317486e11f128fb9801cf843e268869eec.jpg)

![](images/6e80acc97ee03ca3b6985de136865b05627205a921965834501f07afa44aeb50.jpg)

![](images/90022d680d5c0e26700d68cfdb06d872c4468d212bd6f66095014d4664fe0259.jpg)

![](images/2771487ad0ac94fdb3ffb8ac8e7f4ae7c6de111e0aa4bcfe9773d7657823129e.jpg)

![](images/64eaf18df3fe62f544f9398c1b5ea2abc18f09a844b1d6ad3e9f63e2ffad923e.jpg)

![](images/9119fc50e41b6e549e7b78f0ef2f605a0aa33becbbc85b9ac73cee557367229f.jpg)

(b) Window 25, Top-3 lrbl channels: [1, 2, 4]  
![](images/eb04dfc125dd64bf8fcaf565be4d839b323797c4a04f135e39529c9916359db4.jpg)

![](images/e33778fbfff094c9c4a3c54598dd968e97a427ed002d7e9e961f9f0168fec079.jpg)

![](images/17c4a5a952172fff43e90febe82bbbbfc31754be5c912cadbb2c576e2166336c.jpg)

![](images/59608d9710b2f51141c6c1825b444731454e1a67656a249b830ecb381e9b9f51.jpg)

![](images/ce56fbabcdfe4d691c9ed1fa1ec6588084173c65b6e553cb786c85c286603cea.jpg)

![](images/309ad43f3db5b90960f6e3f9eb4eae33bcb59b3fb5889e1ed19d0bb3c25df2c7.jpg)  
(c) Window 45, Top-3 lrbl channels: [1, 4, 3]

Figure 9: Prototype usage visualisation examples on the Stock dataset for four representative windows. Each panel shows (a) the raw time-series window, (b) prototype attention weights, (c) the residual regime vector $r _ { b } ,$ and (d)–(f) the dominant-frequency signal before and after prototype-based adjustment for the three channels with the largest $\left| r _ { b } \right|$ . The prototype attention weights and $r _ { b }$ values are obtained after training for 50 epochs, at which point the learned scaling parameter is $\alpha = 0 . 0 9 0 7$