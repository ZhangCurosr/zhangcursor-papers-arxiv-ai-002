# Sparse MLLM Anchors, Dense Adaptation: Breaking the Self-Referential Loop in Wild Test-Time Adaptation

Zhenbin Wang, Lei Zhang<sup>∗</sup>, Lituan Wang, Yan Wang, Zhao Zhang, Wei Huang

Sichuan University wangzhenbin@stu.scu.edu.cn

## Abstract

Wild test-time adaptation (WTTA) updates a source model online under small test batches, concurrent distribution shifts, and time-varying class imbalance. Most WTTA methods derive their adaptation signals, including predictive uncertainty, sample reliability, and local feature geometry, from the model being adapted. When the source model is unreliable under shift, these signals can reinforce its own errors, forming a selfreferential loop. We introduce MASA (Multimodal-LLM-Anchored Semantic Adaptation), which complements modelinternal evidence with structured semantic descriptions from a frozen multimodal large language model (MLLM). To limit inference cost, MASA queries the MLLM only for a small set of diverse, reliability-ranked anchors. The resulting descriptions capture the object family and nuisance factors such as style, viewpoint, and occlusion. MASA encodes these descriptions, propagates them to neighboring test samples, and stores the resulting visual–semantic information in an online prototype memory. Descriptor-aware retrieval from this memory provides an auxiliary target for lightweight adaptation of normalization-afine parameters. We evaluate MASA on the WTTA ImageNet-C benchmark under limited-batch, mixeddomain, and imbalanced-label-shift settings with ResNet and ViT backbones. Code is available at this link.

## Introduction

Deep models deployed in open-world environments must contend with test distributions that depart from the training distribution because of sensor noise, weather, style, viewpoint, and other changes. Test-time adaptation (TTA) updates a pretrained source model online using an unlabeled test stream, without labels or continued access to the source training set. WTTA considers a more demanding setting in which batches can contain a single sample, several shifts can coexist within one stream, and class frequencies can be imbalanced and time-varying. Each update therefore extract a useful signal from little and potentially unreliable evidence while remaining eficient enough for online inference.

Most WTTA methods improve the reliability of signals already produced by the source model. Entropy minimization (Wang et al. 2021) sharpens individual predictions. Sampleselection methods (Niu et al. 2022, 2023; Lee et al. 2024) retain samples according to entropy, sharpness, or perturbation sensitivity, while normalization- and prototype-based methods stabilize online statistics or class centers. ReCAP (Hu et al. 2025) instead regularizes confidence and consistency within local feature neighborhoods. Although these methods act on diferent units, from individual predictions to selected samples and local regions, the evidence used to judge reliability remains largely derived from the model’s own predictions and representations.

![](images/c2115599e8f878738344bc0fb9115388c5ad320cefcb7af7bd01867906df2231.jpg)  
Figure 1: Comparison of adaptation evidence. Left: many WTTA methods estimate reliability and construct adaptation targets from the model being adapted, allowing prediction errors to influence subsequent updates. Right: MASA queries a frozen MLLM on sparse anchors, propagates the encoded descriptors through local feature neighborhoods, and stores them in a visual–semantic prototype memory for descriptoraware retrieval and memory admission.

Reliance on internal evidence becomes risky when the model is itself unreliable under distribution shift. Prediction entropy, sharpness, perturbation sensitivity, and region confidence may then provide unreliable criteria for selecting samples or updating representations. We call this coupling a self-referential loop: an incorrect but confident prediction can be selected for adaptation and subsequently reinforce the same error (Fig. 1). This failure can be particularly consequential in small batches and imbalanced streams, where a few updates have disproportionate influence. Generative priors introduce external information by restoring inputs toward the source manifold (Gao et al. 2022) or distilling a difusion score (Anonymous 2024b), but they act at the pixel or score level and may add substantial per-sample computation. They also do not explicitly describe the nuisance factors afecting an observation or whether recognizable object evidence remains under the degradation.

To address this limitation, we propose MASA, which introduces semantic evidence about image content and nuisance factors into the adaptation process. A frozen MLLM can produce such descriptions without conditioning on the classifier’s logits, complementing signals internal to the source model. Recent work has shown the value of MLLM-based reasoning in out-of-distribution (OOD) detection (Zhu et al. 2026), but WTTA imposes diferent constraints. Querying every test sample is expensive, observations arrive online, and the source backbone need not share an image–text embedding space. The practical problem is therefore to convert a small number of MLLM descriptions into a persistent signal that can guide adaptation throughout the stream.

MASA converts sparse MLLM observations into dense adaptation guidance. It queries the frozen MLLM only for a small set of diverse, reliability-ranked anchors and requests descriptions of the object family and nuisance factors afecting each image. These responses serve neither as pseudolabels nor as direct class predictions. MASA instead encodes them as semantic descriptors, propagates descriptor information across local feature neighborhoods, and integrates it into an online visual–semantic prototype memory. During adaptation, semantic agreement informs prototype retrieval, and the retrieved prototypes provide an auxiliary consistency target. Reusing each description across nearby samples amortizes MLLM queries over the stream. The framework requires no image–text alignment in the source classifier and updates only selected normalization-afine parameters.

Our contributions are threefold. First, we formulate a sparse semantic anchoring strategy that complements sourcemodel reliability signals with structured descriptions from a frozen MLLM, without requiring a text-aligned source classifier. Second, we introduce a visual–semantic prototype memory that propagates anchor descriptors through local feature neighborhoods and uses descriptor-aware retrieval to regularize normalization-afine adaptation. Third, extensive experiments on ImageNet-C demonstrate that MASA achieves state-of-the-art performance across limited-batch, mixed-domain, and time-varying label-shift protocols with both ResNet and ViT backbones.

## Method

## Problem Setup and Overview

Let $f _ { \theta ^ { \mathrm { s r c } } }$ denote a C-class source classifier that outputs logits. We initialize $\theta _ { 0 } = \theta ^ { \mathrm { s r c } }$ and receive an unlabeled batch $X _ { t } = \{ x _ { t , i } \} _ { i = 1 } ^ { B _ { t } } ,$ , with $B _ { t } = | X _ { t } |$ , at each test step t. Labels are unavailable during adaptation and serve only to evaluate stream accuracy. Under MASA’s predict-then-adapt protocol, the pre-update model $\theta _ { t }$ predicts $X _ { t }$ before its update afects subsequent batches.

MASA makes one stream pass and updates only afine scale and shift parameters in selected normalization layers; the classifier head and all other parameters remain fixed. Before $X _ { t } ,$ , the model and persistent states contain only earlier batches. The observation-level FIFO candidate window ${ \mathcal { W } } _ { t }$ ， anchor bank $\boldsymbol { A } _ { t }$ , and bounded prototype memory $\mathcal { M } _ { t }$ are initially empty; bounded transition and coverage histories and a scalar recovery statistic are also maintained. We omit t from sample- and cluster-level quantities when unambiguous.

MASA complements regional evidence with reusable semantic observations (Fig. 2). It ranks bufered observations by regional statistics, queries a frozen MLLM on a small, visually diverse subset, and propagates the descriptors to nearby samples. A bounded visual–semantic memory consolidates them into prototype targets that regularize the regional update.

## Preliminaries and Two-Stage Evidence Selection

MASA combines ReCAP regional confidence (Hu et al. 2025) and DeYO patch sensitivity (Lee et al. 2024) as an ordered filter for regional adaptation, then reuses their detached statistics to rank semantic queries.

ReCAP regional confidence. For sample $i ,$ let $\mathbf { z } _ { i } \in \mathbb { R } ^ { d _ { v } }$ be the pre-update, unnormalized feature immediately before the final linear classifier, where $d _ { v }$ is its dimension. The fixed head has weight $\mathbf { W } \in \mathbb { R } ^ { C \times d _ { v } }$ and bias b $\in \mathbb { R } ^ { C } ; { \mathbf w } _ { c } ^ { \top }$ denotes row c of W. ReCAP’s sampling-free proxy uses a fixed source-side variance vector $\sigma _ { z , \mathrm { s r c } } ^ { 2 } \in \mathbb { R } _ { > 0 } ^ { d _ { v } }$ , whose j-th entry $[ \pmb { \sigma } _ { z , \mathrm { s r c } } ^ { 2 } ] _ { j } = \mathrm { V a r } _ { \mathrm { s r c } } [ z _ { j } ]$ is the variance of coordinate $j$ in the same feature space. This model-specific statistic is provided once and fixed throughout the target stream.

An efective regional coeficient $\rho _ { \mathrm { r e g } } > 0$ scales this statistic. The resulting diagonal scale $\pmb { \Lambda } _ { \mathrm { r e g } }$ induces a class-pair moment matrix $\bar { \mathbf { M } } \in \bar { \mathbb { R } } _ { > 0 } ^ { C \times C }$

$$
\begin{array} { r l } & { \mathbf { A } _ { \mathrm { r e g } } = \rho _ { \mathrm { r e g } } \operatorname { D i a g } \left( \pmb { \sigma } _ { z , \mathrm { s r c } } ^ { 2 } \right) , } \\ & { M _ { c c ^ { \prime } } = \exp \left[ \frac { 1 } { 2 } ( \mathbf { w } _ { c } - \mathbf { w } _ { c ^ { \prime } } ) ^ { \top } \mathbf { A } _ { \mathrm { r e g } } ( \mathbf { w } _ { c } - \mathbf { w } _ { c ^ { \prime } } ) \right] . } \end{array}\tag{1}
$$

The fixed head gives logits $\mathbf { o } _ { i } = \mathbf { W } \mathbf { z } _ { i } + \mathbf { b } \in \mathbb { R } ^ { C }$ and probabilities $\mathbf { p } _ { i } = \mathrm { s o f t m a x } ( \mathbf { o } _ { i } )$ in the C-class simplex; $o _ { i , c }$ and $p _ { i , c }$ denote their c-th entries. The inherited class-wise logit correction is $\delta ^ { \mathrm { r e g } } = ( \delta _ { 1 } ^ { \mathrm { r e g } } , \dots , \delta _ { C } ^ { \mathrm { r e g } } ) ^ { \intercal }$ , with $\delta _ { c } ^ { \mathrm { r e g } } =$ $\lambda _ { \delta } \| \mathbf { w } _ { c } \| _ { 2 } ^ { 2 }$ , where $\lambda _ { \delta } > 0$ is the correction coeficient. Both M and $\bar { \delta } ^ { \mathrm { r e g } }$ are computed once and fixed.

For sample i, the moment matrix transforms $\mathbf { p } _ { i }$ into the unnormalized regional mass $\nu _ { i } \in \mathbb { R } _ { > 0 } ^ { C } ,$ while the corrected logits define a second distribution $\breve { \mathbf { p } } _ { i } \cdot$

$$
\nu _ { i , c } = \sum _ { c ^ { \prime } = 1 } ^ { C } p _ { i , c ^ { \prime } } M _ { c c ^ { \prime } } , \quad \breve { \bf p } _ { i } = \mathrm { s o f t m a x } ( { \bf o } _ { i } + \delta ^ { \mathrm { r e g } } ) .\tag{2}
$$

ReCAP combines these quantities into the closed-form Regional Entropy (RE) and Regional Instability (RI) proxies:

$$
\ell _ { \mathrm { R E } , i } = \sum _ { c = 1 } ^ { C } \breve { p } _ { i , c } \log \frac { \nu _ { i , c } } { p _ { i , c } } , \quad \ell _ { \mathrm { R I } , i } = \sum _ { c = 1 } ^ { C } p _ { i , c } \log \nu _ { i , c } .\tag{3}
$$

RE measures regional predictive uncertainty, whereas RI captures within-region variation; lower values indicate more reliable evidence.

DeYO patch sensitivity. Following DeYO, we test reliance on coherent spatial structure by resizing an image to dimensions divisible by P, partitioning it into a $P \times \bar { P }$ grid, and randomly permuting the cells. Let $x _ { i } ^ { \mathrm { s h u f } }$ be this view and $\mathbf { p } _ { i } ^ { \mathrm { s h u f } } = \operatorname { s o f t m a x } ( f _ { \theta _ { t } } ( x _ { i } ^ { \mathrm { s h u f } } ) )$ its no-gradient prediction. For the original pseudo-label ${ \widehat { y } } _ { i } .$ , the pseudo-label probability difference (PLPD) is

![](images/63deb3de014362c9297e4e706af38595987eb0d14d251257dc06ddcc4cf68aa9.jpg)  
Figure 2: MASA overview. Gray modules form the regional-evidence path. MASA ranks bufered observations, selects diverse anchors, and queries a frozen MLLM for object and nuisance descriptors. It propagates these descriptors to nearby samples and stores them in a bounded visual–semantic memory for descriptor-aware retrieval. Retrieved prototypes provide an auxiliary consistency target for normalization-afine adaptation.

$$
\widehat { y } _ { i } = \arg \operatorname* { m a x } _ { c } p _ { i , c } , \quad \Delta _ { i } ^ { \mathrm { P L P D } } = p _ { i , \widehat { y } _ { i } } - p _ { i , \widehat { y } _ { i } } ^ { \mathrm { s h u f } } .\tag{4}
$$

A large $\Delta _ { i } ^ { \mathrm { P L P D } }$ means spatial disruption weakens the original pseudo-label, indicating reliance on coherent image content.

Two-stage selection and regional objective. To avoid unnecessary shufled passes, MASA applies the cheaper RE test before PLPD, yielding the adaptation set

$$
\begin{array} { r } { S _ { t } = \left\{ i : \ell _ { \mathrm { R E } , i } < \tau _ { \mathrm { R E } } , \Delta _ { i } ^ { \mathrm { P L P D } } > \tau _ { \mathrm { P L P D } } \right\} . } \end{array}\tag{5}
$$

Here, τ<sub>RE</sub> and τ<sub>PLPD</sub> are the respective acceptance thresholds. The set $S _ { t }$ determines which samples contribute to the regional objective. A separate reliability gate provides an additional signal for anchor ranking, prototype adaptation, and memory admission.

On retained samples, MASA optimizes reliabilityweighted regional terms. Let sg[·] denote stop-gradient; the RE threshold is also the reweighting reference, $\omega _ { \mathrm { m a x } } ^ { \mathrm { r e g } }$ caps sample weights, and $\lambda _ { \mathrm { R I } }$ balances the proxies. The weights and batch objective are

$$
\begin{array} { r l } & { \omega _ { i } ^ { \mathrm { r e g } } = \displaystyle \operatorname* { m i n } \left\{ \exp \left( \tau _ { \mathrm { R E } } - \mathrm { s g } [ \ell _ { \mathrm { R E } , i } ] \right) , \omega _ { \mathrm { m a x } } ^ { \mathrm { r e g } } \right\} , } \\ & { \mathcal { L } _ { \mathrm { r e g i o n } } = \displaystyle \frac { 1 } { | S _ { t } | } \sum _ { i \in S _ { t } } \omega _ { i } ^ { \mathrm { r e g } } \big ( \ell _ { \mathrm { R E } , i } + \lambda _ { \mathrm { R I } } \ell _ { \mathrm { R I } , i } \big ) . } \end{array}\tag{6}
$$

The detached weight favors lower-RE samples, while $\omega _ { \mathrm { m a x } } ^ { \mathrm { r e g } }$ limits individual influence. When $S _ { t }$ is empty, ${ \mathcal { L } } _ { \mathrm { r e g i o n } }$ is a diferentiable zero.

## Reliability-Guided Semantic Anchors

Querying every observation is costly and redundant. MASA instead refreshes a small, reliability-ranked, visually diverse anchor set at initialization, periodically, or when drift and retrieval coverage indicate stale semantics.

We normalize the pre-head feature as $\mathbf { v } _ { t , i } = \mathbf { z } _ { t , i } / \lVert \mathbf { z } _ { t , i } \rVert _ { 2 }$ making dot products cosine similarities. Hence, $\mathbf { v } _ { t , i } \in \mathbb { R } ^ { d _ { v } }$ and $\| \mathbf { v } _ { t , i } \| _ { 2 } = 1$ . Let $p _ { t , i , ( 1 ) } \geq p _ { t , i , ( 2 ) }$ be the two largest entries of $\mathbf { p } _ { t , i } ;$ their margin $m _ { t , i } = p _ { t , i , ( 1 ) } - p _ { t , i , ( 2 ) }$ measures top-1 confidence. For short-term stability, $\mathcal { H } _ { t , i } ^ { - } ~ =$ $( y _ { t , i , 1 } ^ { - } , \ldots , y _ { t , i , L _ { t , i } } ^ { - } )$ contains the top-1 predictions immediately preceding $x _ { t , i }$ in loader order, where $L _ { t , i } \leq H _ { \mathrm { h i s t } } .$ Let 1[·] denote the indicator function. The corresponding label-transition rate is

$$
\phi _ { t , i } = \left\{ \begin{array} { l l } { \displaystyle \frac { 1 } { L _ { t , i } - 1 } \sum _ { j = 2 } ^ { L _ { t , i } } \mathbf { 1 } [ y _ { t , i , j } ^ { - } \neq y _ { t , i , j - 1 } ^ { - } ] , } & { L _ { t , i } \geq 2 , } \\ { 0 , } & { L _ { t , i } < 2 . } \end{array} \right.\tag{7}
$$

Thus, $\phi _ { t , i }$ is the recent rate of adjacent prediction changes, not a repeated-image history. Combining it with RE, RI, and the probability margin, we reuse $\tau _ { \mathrm { R E } }$ and introduce thresholds $\tau _ { \mathrm { R I } } ^ { \mathrm { a n c } } , \tau _ { m } .$ , and $\tau _ { \phi }$ for RI, margin, and transition rate, respectively, giving

$$
\begin{array} { r } { q _ { i } = \mathbf { 1 } \Bigg [ \mathrm { \ell } ^ { \ell _ { \mathrm { R E } , i } < \tau _ { \mathrm { R E } } \wedge \ell _ { \mathrm { R I } , i } < \tau _ { \mathrm { R I } } ^ { \mathrm { a n c } } } } \\ { \wedge m _ { i } > \tau _ { m } \wedge \phi _ { i } < \tau _ { \phi } \Bigg ] . } \end{array}\tag{8}
$$

The gate $q _ { i } = 1$ requires low regional uncertainty and instability, a large margin, and stable predictions; it is reused for anchor ranking, prototype adaptation, and memory admission.

Each arriving sample contributes the detached record $\begin{array} { r l r } { \mathsf { r } _ { t , i } } & { : = } & { \left( x _ { t , i } , \mathsf { v } _ { t , i } , \mathsf { p } _ { t , i } , \ell _ { \mathrm { R E } , t , i } , \ell _ { \mathrm { R I } , t , i } , m _ { t , i } , \phi _ { t , i } , q _ { t , i } \right) } \end{array}$ . Besides the RGB input $x _ { t , i } ~ \in ~ \mathbb { R } ^ { 3 \times H _ { x } \times W _ { x } }$ with height $H _ { x }$ and width $W _ { x } ,$ , the record stores a $d _ { v }$ -dimensional visual feature, a C-dimensional prediction, and the five scalars $\ell _ { \mathrm { R E } , t , i } ,$ $\ell _ { \mathrm { R I } , t , i } , m _ { t , i } , \phi _ { t , i }$ , and $q _ { t , i } ,$ , all fixed at arrival. With ∥ denoting sequence concatenation and $\mathrm { t a i l } _ { H }$ retaining the latest H records, the current batch gives

$$
\widetilde { \mathcal { W } } _ { t } = \mathrm { t a i l } _ { H _ { \mathrm { w i n } } } \big ( \mathcal { W } _ { t } \| \big ( \boldsymbol { \mathsf { r } } _ { t , 1 } , \boldsymbol { \mathsf { \ldots } } , \boldsymbol { \mathsf { r } } _ { t , B _ { t } } \big ) \big ) , \quad | \widetilde { \mathcal { W } } _ { t } | { \le } H _ { \mathrm { w i n } } .\tag{9}
$$

Excluding images, the $N _ { t } = | \widetilde { \mathcal { W } } _ { t } | \le H _ { \mathrm { w i n } }$ records form an $N _ { t } \times \mathsf { \bar { ( } } d _ { v } + \mathsf { \bar { C } } + 5 )$ ) metadata array. Since $H _ { \mathrm { w i n } }$ counts observations, not steps, its temporal span varies with $B _ { t }$ . Refresh reads without removing records, and absent recovery, $\mathcal { W } _ { t + 1 } = \widetilde { \mathcal { W } } _ { t }$ . A window candidate is a bufered observation eligible for querying, distinct from a candidate memory cluster.

At a refresh event, $R _ { i } ^ { \mathrm { a n c } }$ ranks bufered candidates by model-derived reliability:

$$
\begin{array} { r } { R _ { i } ^ { \mathrm { a n c } } { = } \biggl ( 1 { - } \operatorname* { m i n } \biggl \{ \frac { \ell _ { \mathrm { R E } , i } } { \log C } , c _ { \mathrm { c l i p } } \biggr \} \biggr ) + \biggl ( 1 { - } \operatorname* { m i n } \biggl \{ \frac { \ell _ { \mathrm { R I } , i } } { \tau _ { \mathrm { R I } } ^ { \mathrm { a n c } } } , c _ { \mathrm { c l i p } } \biggr \} \biggr ) } \\ { + m _ { i } - \phi _ { i } + q _ { i } . \qquad } \end{array}
$$

Clipping prevents either regional proxy from dominating, while higher scores favor low RE and RI, a large margin, and few transitions. To avoid near duplicates, MASA shortlists the top min $( | \widetilde { \mathcal { W } } _ { t } | , r _ { \mathrm { p o o l } } B _ { a } )$ records and runs farthest-point sampling on $\mathbf { v } _ { i } .$ , seeded by the highest-ranked record, to select at most $B _ { a }$ anchors. Here, $B _ { a }$ is the query budget, $r _ { \mathrm { p o o l } }$ the shortlist multiplier, and $c _ { \mathrm { c l i p } }$ the bound on each regional contribution.

Refresh uses recent feature-to-memory distance to detect stale coverage. Let $\{ \mu _ { k } \} _ { k = } ^ { K _ { t } }$ =1 be the unit visual centroids of the $K _ { t } \ ' \le K _ { \operatorname* { m a x } }$ clusters, and let $\mathcal { R } _ { t }$ index the min $( H _ { D } , | \widetilde { \mathcal { W } } _ { t } | )$ ) most recent records of $\widetilde { \mathcal { W } } _ { t }$ , where $H _ { D }$ is the drift horizon. Their mean nearest-centroid cosine distance is

$$
D _ { t } = \frac { 1 } { \vert \mathcal { R } _ { t } \vert } \sum _ { j \in \mathcal { R } _ { t } } \left( 1 - \operatorname* { m a x } _ { 1 \le k \le K _ { t } } \mathbf { v } _ { j } ^ { \top } \pmb { \mu } _ { k } \right) ,\tag{11}
$$

where $D _ { t } ~ = ~ 1$ if either $\mathcal { R } _ { t }$ or $\mathcal { M } _ { t }$ is empty, favoring refresh without coverage. Larger $D _ { t }$ indicates poorer memory coverage.

Feature drift misses failed descriptor matching. After completed step s, let $r _ { s } ^ { \mathrm { c o v } }$ be the fraction of all $B _ { s }$ samples whose descriptor-aware retrieval score exceeds the assignment threshold $\tau _ { \mathrm { a s s i g n } }$ defined below. Coverage retains the latest $H _ { \mathrm { c o v } }$ completed steps. We use the same numerical horizon for $H _ { \mathrm { c o v } }$ and $H _ { \mathrm { w i n } }$ , but the former counts steps and the latter observations. Once the history has $H _ { \mathrm { c o v } }$ entries, the average before step t is

$$
\bar { r } _ { t } ^ { \mathrm { c o v } } = \frac { 1 } { H _ { \mathrm { c o v } } } \sum _ { s = t - H _ { \mathrm { c o v } } } ^ { t - 1 } r _ { s } ^ { \mathrm { c o v } } .\tag{12}
$$

MASA refreshes whenever $\begin{array} { r } { \boldsymbol { \mathcal { A } } _ { t } \ = \ \boldsymbol { \mathcal { O } } , } \end{array}$ , including after initialization or recovery, every $T _ { \mathrm { r e f } }$ adaptation steps, or when $D _ { t } > \tau _ { D }$ or $\bar { r } _ { t } ^ { \mathrm { c o v } } < \tau _ { \mathrm { c o v } }$ . Coverage is tested only after its history is full; $\tau _ { D }$ and $\tau _ { \mathrm { c o v } }$ are the drift and coverage thresholds.

## Semantic Grounding and Propagation

Selected anchors provide visual coverage but not content or nuisance descriptions; MASA obtains these from a frozen MLLM and propagates them through visual neighborhoods.

For a selected record $\mathsf { r } _ { s , j } \in \breve { \mathcal { W } } _ { t }$ , where $s \leq t ,$ assign anchor index a and reuse its detached feature as $\mathbf { u } _ { a } = \mathbf { v } _ { s , j } .$ The query contains only the image and a fixed, task-agnostic prompt, with neither classifier predictions nor source-class names. The frozen MLLM returns open-vocabulary descriptions of object family, scene, style shift, viewpoint, and occlusion, plus response confidence $c _ { a } ^ { \mathrm { r e s p } } \in [ 0 , \dot { 1 } ]$ and object recognizability $c _ { a } ^ { \mathrm { o b j } } \in [ 0 , 1 ]$ , which estimates whether primary-object evidence survives the degradation. Descriptions occupy a semantic space separate from the classifier’s $C .$ -way output, while the scores determine reliability.

We discard empty and unknown values, serialize each retained value as $\mathtt { f i e l d }$ : value, and represent list elements separately. Denote the resulting L phrases by $( \xi _ { 1 } , \dots , \xi _ { L } )$ (Fig. 3). If none remains, the fallback ob ${ \mathrm { j e c t \_ f a m i l y : } }$ unknown object ensures $L \geq 1$ . We use $\scriptstyle { { \mathcal { C } } _ { a } }$ for the retained code set, which excludes this fallback.

A frozen encoder $E _ { \mathrm { t e x t } } : \mathcal { T }  \mathbb { R } ^ { d _ { s } }$ maps phrase space $\tau$ to a $d _ { s }$ -dimensional space distinct from classifier features. Let norm denote $\ell _ { 2 }$ normalization. Uniform aggregation and descriptor reliability are

$$
\begin{array} { l } { \displaystyle { \mathbf e } _ { a } = \mathrm { n o r m } \left( \sum _ { l = 1 } ^ { L } E _ { \mathrm { t e x t } } ( \xi _ { l } ) \right) , } \\ { \displaystyle \kappa _ { a } = \mathrm { m a x } \left( c _ { a } ^ { \mathrm { r e s p } } c _ { a } ^ { \mathrm { o b j } } , \kappa _ { \mathrm { m i n } } \right) . } \end{array}\tag{13}
$$

Uniform aggregation avoids field weights, normalization gives unit descriptors in $\mathbb { R } ^ { d _ { s } }$ , and $\kappa _ { \mathrm { m i n } }$ prevents zero reliability. The bank stores $\left( \mathbf { u } _ { a } , \mathbf { e } _ { a } , \mathcal { C } _ { a } , \kappa _ { a } \right)$ , where $\| \mathbf { u } _ { a } \| _ { 2 } = 1$ and $\scriptstyle { { \mathcal { C } } _ { a } }$ contains codes ⟨field=value⟩. For maximum neighborhood size $K _ { a }$ , its capacity is max $( 4 K _ { \operatorname* { m a x } } , K _ { a } )$ . Refresh appends tuples to $\boldsymbol { A } _ { t }$ in query order and evicts the oldest when full; $\kappa _ { a }$ controls the anchor’s memory update.

Because few observations are queried, MASA estimates each descriptor from visual neighbors. For feature $\mathbf { v } _ { i }$ and a nonempty bank, let $\mathcal { N } _ { A } ( i )$ contain the min $\left( K _ { a } , \left| \mathcal { A } _ { t } \right| \right)$ anchors with largest $\mathbf { v } _ { i } ^ { \top } \mathbf { u } _ { a }$ , where $K _ { a }$ is the maximum neighborhood size. The nearest index is $a _ { i } ^ { \star } \in \arg \operatorname* { m a x } _ { a \in \mathcal { N } _ { A } \left( i \right) } \mathbf { v } _ { i } ^ { \mid } \mathbf { u } _ { a }$ . With temperature $\tau _ { a } .$ propagation is

$$
\begin{array} { r l r } & { } & { \displaystyle \pi _ { i a } = \frac { \exp \left( \mathbf { v } _ { i } ^ { \top } \mathbf { u } _ { a } / \tau _ { a } \right) } { \sum _ { b \in \mathcal { N } _ { A } \left( i \right) } \exp \left( \mathbf { v } _ { i } ^ { \top } \mathbf { u } _ { b } / \tau _ { a } \right) } , \quad \widehat { \kappa } _ { i } = \sum _ { a \in \mathcal { N } _ { A } \left( i \right) } \pi _ { i a } \kappa _ { a } , } \\ & { } & { \displaystyle \widehat { \mathbf { e } } _ { i } = \mathrm { n o r m } \left( \sum _ { a \in \mathcal { N } _ { A } \left( i \right) } \pi _ { i a } \mathbf { e } _ { a } \right) , \quad \widehat { C } _ { i } = \bigcup _ { a \in \mathcal { N } _ { A } \left( i \right) : } \mathcal { C } _ { a } . } \\ & { } & { \displaystyle \pi _ { i a } \kappa \epsilon _ { c } \mathrm { o r } a = a _ { i } ^ { * } } \end{array}\tag{14}
$$

Here, $\epsilon _ { \mathcal { C } }$ is the code-inclusion threshold. The weights $\pi _ { i a }$ interpolate embeddings and reliabilities, producing $\widehat { \mathbf { e } } _ { i }$ and $\widehat { \kappa } _ { i } ; \bar { \widehat { \mathcal { C } } } _ { i }$ collects codes from influential anchors and always considers the nearest, whose code set may be empty. If $\boldsymbol { A } _ { t }$ is empty, $\widehat { \mathbf { e } } _ { i }$ is absent, $\widehat { \kappa } _ { i } = 0$ , and $\widehat { \mathcal { C } } _ { i } = \varnothing$ . All descriptor quantities are detached from the classifier graph.

## Visual–Semantic Prototype Memory

Propagation yields per-observation descriptors but no stream-level consolidation. MASA therefore maintains bounded visual, predictive, and semantic prototypes. Retrieval reads incoming memory; ordinary writes follow adaptation, except that refreshed anchors are upserted before retrieval to afect the current batch.

![](images/06e2867775cadceaac871fecdb8b7588ae094af7f1cd9c287fb26bd6df49811d.jpg)  
Figure 3: From an image-only MLLM query to descriptor phrases. A fixed task-agnostic prompt returns structured fields. Each value is indexed by l and denoted by $\xi _ { l } ;$ list entries remain separate, while gray scores determine $\kappa _ { a }$ and are not embedded.

The memory contains $K _ { t } ~ \le ~ K _ { \operatorname* { m a x } }$ clusters, $\begin{array} { r l } { \mathcal { M } _ { t } } & { { } = } \end{array}$ $\{ \mathfrak { m } _ { k } \} _ { k = 1 } ^ { K _ { t } }$ , each summarizing admitted observations and anchors through four descriptor states. The unit visual centroid is $\boldsymbol { \mu _ { k } } \in \mathbb { R } ^ { d _ { v } }$ , and $\bar { \bf p } _ { k }$ is the predictive prototype in the C-class simplex. The unit semantic centroid $\bar { \mathbf { e } } _ { k } \in \mathbb { R } ^ { \bar { d } _ { s } }$ is absent when no semantic evidence exists. The set $\mathcal { C } _ { k }$ accumulates codes ⟨field=value⟩ from $\widehat { \mathcal { C } } _ { i }$ for observations or $\scriptstyle { { \mathcal { C } } _ { a } }$ for anchors.

Five statistics control trust and retention: reliability $\psi _ { k } \in$ [0, 1] averages write reliabilities, support $n _ { k } \in \mathbb { N } _ { + }$ counts admitted observations and anchors, age $ \ d _ { \textmu } \textmu _ { k } \in \mathbb { N } _ { 0 }$ counts steps since the latest write, and drift $d _ { k } \geq 0$ averages incoming cosine distance. Finally, $\chi _ { k } \in \{ 0 , 1 \}$ distinguishes candidate $( \chi _ { k } = 0 )$ from committed $( \chi _ { k } = 1 )$ clusters. The state is

$$
\mathfrak { m } _ { k } = \left( \mu _ { k } , \bar { \mathbf { p } } _ { k } , \bar { \mathbf { e } } _ { k } , \mathcal { C } _ { k } , \psi _ { k } , n _ { k } , \mathrm { a g e } _ { k } , d _ { k } , \chi _ { k } \right) .\tag{15}
$$

Descriptor-aware retrieval. Retrieval seeks complementary evidence rather than visual proximity alone. It removes clusters below reliability $\tau _ { q } ,$ giving

$$
\begin{array} { r } { \mathcal { K } _ { t } ^ { \mathrm { r e t } } = \big \{ k \in \{ 1 , \dots , K _ { t } \} : \psi _ { k } \geq \tau _ { q } \} , } \end{array}\tag{16}
$$

where $\tau _ { q }$ is the minimum reliability. Define normalized age $\widetilde { \mathrm { a g e } } _ { k } ^ { 2 } ~ = ~ \mathrm { m i n } ( \mathrm { a g e } _ { k } / H _ { \mathrm { w i n } } , 1 )$ and Jaccard similarity $\bar { J ( A , B ) } ^ { \prime \prime } = | A \cap \bar { B | } / | \bar { A } \cup B |$ , with $J ( \emptyset , \emptyset ) \ = \ 0$ . Let $s _ { i k } ^ { \mathrm { s e m } } = \cos ( \widehat { \mathbf { e } } _ { i } , \bar { \mathbf { e } } _ { k } )$ when both vectors exist and $s _ { i k } ^ { \mathrm { s e m } } = 0$ otherwise. Here $H _ { \mathrm { w i n } }$ is reused only as a numerical cap for step-based age, despite denoting observation capacity above. All current-sample arguments are stop-gradient. For $k \in \mathcal { K } _ { t } ^ { \mathrm { r e t } }$

$$
\begin{array} { r l } & { S _ { i k } = \mathbf { v } _ { i } ^ { \top } \pmb { \mu } _ { k } + \mathbf { p } _ { i } ^ { \top } \bar { \mathbf { p } } _ { k } + s _ { i k } ^ { \mathrm { s e m } } } \\ & { ~ + ~ J ( \widehat { \mathcal { C } } _ { i } , \mathcal { C } _ { k } ) - \alpha _ { \mathrm { a g e } } \widehat { \mathrm { a g e } } _ { k } - \alpha _ { \mathrm { c a n d } } ( 1 - \chi _ { k } ) . } \end{array}\tag{17}
$$

The positive terms in $S _ { i k }$ measure visual, predictive, semantic-embedding, and code agreement; penalties weighted by $\alpha _ { \mathrm { a g e } }$ and $\alpha _ { \mathrm { c a n d } }$ discount stale and uncommitted clusters. If $\mathcal { K } _ { t } ^ { \mathrm { r e t } }$ is nonempty, set $k _ { i } ^ { \star } = \arg \operatorname* { m a x } _ { k \in { \mathcal { K } } _ { + } ^ { \mathrm { r e t } } } S _ { i k }$ and accept it when $S _ { i k _ { i } ^ { \star } } > \tau _ { \mathrm { a s s i g n } }$ , where $\tau _ { \mathrm { a s s i g n } }$ is the assignment threshold. Other samples remain unassigned.

Prototype adaptation. An accepted cluster supplies fixed visual and predictive targets. The reliability gate blocks unreliable gradients, while a match weight grows with retrieval confidence at a rate controlled by $\tau _ { \omega }$ . Define

$$
\begin{array} { r l } & { \omega _ { i } ^ { \mathrm { p r o t o } } = \operatorname { s i g m o i d } \left( \frac { S _ { i k _ { i } ^ { \star } } - \tau _ { \mathrm { a s s i g n } } } { \tau _ { \omega } } \right) , } \\ & { \quad \quad \mathcal { T } _ { t } = \left\{ i : k _ { i } ^ { \star } \ : \mathrm { e x i s t s } , \ : S _ { i k _ { i } ^ { \star } } > \tau _ { \mathrm { a s s i g n } } , \ : q _ { i } = 1 \right\} . } \end{array}\tag{18}
$$

Here $q _ { i }$ is the gate in Eq. (8). Let $d _ { i k } ^ { \mathrm { s e m } } = \mathrm { s g } [ 1 - \cos ( \widehat { \bf e } _ { i } , \bar { \bf e } _ { k } ) ]$ when both semantic vectors exist and $d _ { i k } ^ { \mathrm { s e m } } = 0$ otherwise. For $i \in \mathcal { Z } _ { t }$ , the discrepancy and batch objective are

$$
\begin{array} { r l } & { \ell _ { \mathrm { p r o t o } , i } = \left( 1 - \mathbf { v } _ { i } ^ { \top } \pmb { \mu } _ { k _ { i } ^ { \star } } \right) } \\ & { \qquad + D _ { \mathrm { K L } } \left( \bar { \mathbf { p } } _ { k _ { i } ^ { \star } } \parallel \mathbf { p } _ { i } \right) + d _ { i k _ { i } ^ { \star } } ^ { \mathrm { s e m } } , } \\ & { \mathcal { L } _ { \mathrm { p r o t o } } = \displaystyle \frac { 1 } { | \mathcal { T } _ { t } | } \sum _ { i \in \mathcal { T } _ { t } } \omega _ { i } ^ { \mathrm { p r o t o } } \ell _ { \mathrm { p r o t o } , i } . } \end{array}\tag{19}
$$

Here $\begin{array} { r c l } { D _ { \mathrm { K L } } ( \bar { \bf p } _ { k } \| { \bf p } _ { i } ) } & { = } & { \sum _ { c } \bar { p } _ { k , c } \log ( \bar { p } _ { k , c } / p _ { i , c } ) } \end{array}$ treats the stored prediction as target. Visual and predictive terms align the current output; semantics afects matching and $\omega _ { i } ^ { \mathrm { p r o \bar { t } o } }$ through $S _ { i k }$ . Its detached discrepancy changes the recovery objective but not text gradients. All terms enter $\ell _ { \mathrm { p r o t o } , i }$ with unit coeficients, and the loss is diferentiable zero when $\mathcal { T } _ { t }$ is empty.

Memory update. Memory updates are separated from gradient adaptation so that the current optimizer step cannot immediately rewrite its own targets. After adaptation, MASA writes detached features and predictions computed before the update. Let $\eta _ { 0 }$ be the base update rate and let clip(x, 0, 1) truncate x to [0, 1]. The reliability gate and descriptor confidence jointly determine whether an observation may enter memory and, for a matched observation, how strongly it updates the cluster.

$$
\begin{array} { r l } & { \rho _ { i } ^ { \mathrm { m e m } } = q _ { i } \operatorname* { m a x } ( \widehat { \kappa } _ { i } , \kappa _ { \mathrm { m i n } } ) , } \\ & { \eta _ { i } ^ { \mathrm { m e m } } = \mathrm { c l i p } \left( \frac { \eta _ { 0 } \rho _ { i } ^ { \mathrm { m e m } } } { \sqrt { n _ { k _ { i } ^ { \star } } + 1 } } , 0 , 1 \right) , \qquad i \in \mathbb { Z } _ { t } . } \end{array}\tag{20}
$$

The scalar $\rho _ { i } ^ { \mathrm { m e m } }$ is the admission reliability, and $\tau _ { \mathrm { s t o r e } }$ is its minimum storage threshold. Samples with $\rho _ { i } ^ { \mathrm { m e m } } \leq \tau _ { \mathrm { s t o r e } }$ are skipped. For an assignment accepted during batch retrieval, Eq. (20) updates $k \stackrel { = } { = } k _ { i } ^ { \star }$ . The factor $\sqrt { n _ { k _ { i } ^ { \star } } + 1 }$ makes wellsupported clusters change more slowly.

An initially unassigned observation is compared again with the progressively updated memory. If this second retrieval accepts a match, we reuse $k _ { i } ^ { \star }$ for that cluster and evaluate $\eta _ { i } ^ { \mathrm { m e m } }$ by the same rule in Eq. (20), solely for the memory write. The observation is not added retrospectively to $\mathcal { T } _ { t } .$ . If no match is accepted, it initializes a candidate cluster. Let norm<sub>1</sub> denote $\ell _ { 1 }$ normalization. With + denoting the intermediate state after sample i, the three centroids are updated by

$$
\begin{array} { r l } & { { \pmb \mu } _ { k } ^ { + } = \mathrm { n o r m } ( ( 1 - \eta _ { i } ^ { \mathrm { m e m } } ) { \pmb \mu } _ { k } + \eta _ { i } ^ { \mathrm { m e m } } { \pmb v } _ { i } ) , } \\ & { \bar { \bf p } _ { k } ^ { + } = \mathrm { n o r m } _ { 1 } ( ( 1 - \eta _ { i } ^ { \mathrm { m e m } } ) \bar { \bf p } _ { k } + \eta _ { i } ^ { \mathrm { m e m } } { \bf p } _ { i } ) , } \\ & { \bar { \bf e } _ { k } ^ { + } = \mathrm { n o r m } ( ( 1 - \eta _ { i } ^ { \mathrm { m e m } } ) \bar { \bf e } _ { k } + \eta _ { i } ^ { \mathrm { m e m } } \widehat { \bf e } _ { i } ) . } \end{array}\tag{21}
$$

If $\bar { \mathbf { e } } _ { k }$ is absent, it is initialized by $\widehat { \mathbf { e } } _ { i }$ . If $\widehat { \mathbf { e } } _ { i }$ is absent, the semantic update is skipped. Let $\lambda _ { \mathrm { s t a t } }$ be the shared smoothing rate for cluster reliability and visual drift. The remaining matched-cluster statistics follow

$$
\begin{array} { r l } & { \mathcal { C } _ { k } ^ { + } = \mathcal { C } _ { k } \cup \widehat { \mathcal { C } } _ { i } , \quad \psi _ { k } ^ { + } = ( 1 - \lambda _ { \mathrm { s t a t } } ) \psi _ { k } + \lambda _ { \mathrm { s t a t } } \rho _ { i } ^ { \mathrm { m e m } } , } \\ & { d _ { k } ^ { + } = ( 1 - \lambda _ { \mathrm { s t a t } } ) d _ { k } + \lambda _ { \mathrm { s t a t } } \big ( 1 - \pmb { \mu } _ { k } ^ { \top } \mathbf { v } _ { i } \big ) , } \\ & { n _ { k } ^ { + } = n _ { k } + 1 , \quad \quad \mathrm { a g e } _ { k } ^ { + } = 0 . } \end{array}\tag{22}
$$

Eq. (21) and (22) are applied in loader order, so each + state becomes the input to the next admitted write. An admitted observation with no accepted match initializes a candidate $k _ { \mathrm { n e w } }$ by

$$
\begin{array} { r l } & { ( \mu _ { k _ { \mathrm { n e w } } } , \bar { \bf p } _ { k _ { \mathrm { n e w } } } , \bar { \bf e } _ { k _ { \mathrm { n e w } } } , \mathcal { C } _ { k _ { \mathrm { n e w } } } ) = ( { \bf v } _ { i } , { \bf p } _ { i } , \widehat { \bf e } _ { i } , \widehat { \mathcal { C } _ { i } } ) , } \\ & { \quad \quad \quad \quad ( \psi _ { k _ { \mathrm { n e w } } } , n _ { k _ { \mathrm { n e w } } } ) = ( \rho _ { i } ^ { \mathrm { m e m } } , 1 ) , } \\ & { \quad \quad \quad ( \mathrm { a g e } _ { k _ { \mathrm { n e w } } } , d _ { k _ { \mathrm { n e w } } } , \chi _ { k _ { \mathrm { n e w } } } ) = ( 0 , 0 , 0 ) . } \end{array}\tag{23}
$$

If $\widehat { \mathbf { e } } _ { i }$ is unavailable, the new cluster’s semantic centroid is initialized as absent. The mandatory first anchor refresh bootstraps an initially empty memory. For each newly queried anchor, let $\rho _ { a } ^ { \mathrm { a n c } } = q _ { a } \kappa _ { a } .$ where $\mathbf { p } _ { a }$ and $q _ { a }$ are taken from its window record and remain fixed after arrival. Anchor upserts do not use the sample-storage threshold $\tau _ { \mathrm { s t o r e } } .$ . Instead, $\rho _ { a } ^ { \mathrm { a n c } }$ initializes or updates cluster reliability. We obtain $S _ { a k }$ from Eq. (17) by replacing $( \mathbf { v } _ { i } , \mathbf { p } _ { i } , \widehat { \mathbf { e } } _ { i } , \widehat { \mathcal { C } } _ { i } )$ with $\left( \mathbf { u } _ { a } , \mathbf { p } _ { a } , \mathbf { e } _ { a } , \mathcal { C } _ { a } \right)$

If $\boldsymbol { \mathcal { K } } _ { t } ^ { \mathrm { r e t } }$ is nonempty, let $k _ { a } ^ { \star } = \arg$ max<sub>k∈K</sub>ret $S _ { a k }$ . When $S _ { a k _ { a } ^ { \star } } > \tau _ { \mathrm { a s s i g n } } ,$ we set $\eta _ { a } ^ { \mathrm { a n c } } = \mathrm { c l i p } ( \eta _ { 0 } \rho _ { a } ^ { \mathrm { a n c } } / \sqrt { n _ { k _ { a } ^ { \star } } + 1 } , 0 , 1 )$ and apply Eqs. (21)–(22) with $( \mathbf { v } _ { i } , \mathbf { p } _ { i } , \widehat { \mathbf { e } } _ { i } , \widehat { \mathcal { C } } _ { i } , \rho _ { i } ^ { \mathrm { m e m } } , \eta _ { i } ^ { \mathrm { m e m } } , k )$ replaced by $( \mathbf { u } _ { a } , \mathbf { p } _ { a } , \mathbf { e } _ { a } , \mathcal { C } _ { a } , \rho _ { a } ^ { \mathrm { a n c } } , \eta _ { a } ^ { \mathrm { a n c } } , k _ { a } ^ { \star } )$ . If the eligible set is empty or the score test fails, Eq. (23) applies to the anchor quantities. Promotion and capacity maintenance finish before batch retrieval, making retained anchor-seeded clusters immediately available.

A new cluster remains a candidate until repeated support and suficient reliability reduce the risk of storing an isolated error. It becomes committed when $n _ { k } \ge n _ { \operatorname* { m i n } }$ and $\psi _ { k } \geq \tau _ { q } ,$ after which the commitment is irreversible. If $K _ { t } > K _ { \operatorname* { m a x } } ,$ the memory retains the $K _ { \mathrm { m a x } }$ clusters with the largest utility:

$$
U _ { k } = \psi _ { k } + \log ( 1 + n _ { k } ) - \widetilde { \mathrm { a g e } } _ { k } - d _ { k } + \chi _ { k } .\tag{24}
$$

The utility favors reliable, well-supported, committed clusters while discounting stale or drifting ones. Promotion and eviction are deferred until all precomputed assignments have been consumed, which keeps cluster indices stable during sequential writes. Maintenance uses the post-write, preincrement ages. Each surviving age is then increased by one to form $\mathcal { M } _ { t + 1 }$ , so a cluster written at step t has age one when step t + 1 begins.

## Adaptation Objective and Online Procedure

The regional and prototype losses address complementary failure modes: the former selects structurally reliable observations within the current batch, whereas the latter regularizes them against evidence accumulated across the stream.

![](images/3b237fade416ff47544d6c3fcccc52b0655495e9491a8ce2f2a80acb0d1952fb.jpg)  
Figure 4: Feature-space visualization after online adaptation. t-SNE features of 10 classes produced by ReCAP and MASA on ImageNet-C under Elastic Transform at severity level 5. Colors indicate classes, and boxes mark representative regions for comparison.

We next combine the two objectives and specify the causal ordering of prediction, adaptation, and state updates.

Let $\vartheta _ { t }$ collect the trainable normalization-afine parameters of $\theta _ { t }$ . For each test batch, MASA minimizes

$$
\mathcal { L } _ { \mathrm { M A S A } } = \mathcal { L } _ { \mathrm { r e g i o n } } + \lambda _ { \mathrm { p r o t o } } \mathcal { L } _ { \mathrm { p r o t o } } ,\tag{25}
$$

where $\lambda _ { \mathrm { p r o t o } }$ balances the two objectives. One gradient step maps $\vartheta _ { t }$ to $\boldsymbol { \vartheta } _ { t + 1 }$ . MASA returns the logits computed before this update, ensuring that the prediction for $X _ { t }$ does not depend on its own adaptation step.

Following SAR (Niu et al. 2023), MASA monitors an exponential moving average $\bar { \ell } _ { t }$ of the adaptation objective and uses a small value as the recovery criterion.

With momentum $\rho _ { \mathrm { r e c } } ,$ its update is

$$
\begin{array} { r } { \bar { \ell } _ { t } { = } \left\{ \begin{array} { l l } { \mathcal { L } _ { \mathrm { M A S A } } , } & { \mathrm { i f ~ u n i n i t i a l i z e d , } } \\ { \rho _ { \mathrm { r e c } } \bar { \ell } _ { t - 1 } { + } ( 1 { - } \rho _ { \mathrm { r e c } } ) \mathcal { L } _ { \mathrm { M A S A } } , } & { \mathrm { o t h e r w i s e . } } \end{array} \right. } \end{array}\tag{26}
$$

Let $\tau _ { \mathrm { r e c } }$ be the recovery threshold. If $\bar { \ell } _ { t } ~ < ~ \tau _ { \mathrm { r e c } }$ , MASA restores the model and optimizer states saved at initialization. It also clears $\mathcal { W } _ { t } , \mathcal { A } _ { t } , \bar { \mathcal { M } } _ { t }$ and their coverage and transition histories, then marks the recovery average as uninitialized so that the next batch follows the first branch of Eq. (26).

See the appendix for pseudocode of the complete predictthen-adapt procedure. Anchor refresh precedes retrieval so that new descriptions are immediately useful, whereas ordinary cluster writes use detached features and predictions computed before optimization. Unless recovery occurs, the post-append window becomes $\mathcal { W } _ { t + 1 }$ , and the final anchor and memory states are relabeled $( \mathcal { A } _ { t + 1 } , \mathcal { M } _ { t + 1 } )$ .

## Experiments

Experimental setup. We evaluate MASA on ImageNet-C (Hendrycks and Dietterich 2019) under limited-batch (= 1), mixed-domain, and time-varying label-shift protocols using ResNet50-GN (He et al. 2016) and ViT-Base-LN (Dosovit skiy et al. 2021). We compare with MEMO (Zhang, Levine, and Finn 2022), DDA (Gao et al. 2022), Tent (Wang et al. 2021), EATA (Niu et al. 2022), SAR (Niu et al. 2023), DeYO (Lee et al. 2024), and ReCAP (Hu et al. 2025). MASA adapts normalization-afine parameters once per batch and uses frozen Qwen2-VL-2B (Wang et al. 2024) and CLIP ViT-B/16 (Radford et al. 2021) without a classifier hint. Complete settings are provided in the supplementary material.

Main results. Tables 1 and 2 report corruption-wise accuracy under limited-batch and time-varying label-shift evaluation, respectively. Table 3 reports mixed-domain performance at severities 5 and 4 and their average.

Table 1: Limited-batch evaluation on ImageNet-C at severity 5.
<table><tr><td></td><td colspan="3">Noise</td><td colspan="4">Blur</td><td colspan="4">Weather</td><td colspan="4">Digital</td><td rowspan="2">Avg</td></tr><tr><td>Method</td><td>Gau</td><td>Sho</td><td>Imp</td><td>Def</td><td>Gla</td><td>Mot</td><td>Zoo</td><td>Sno</td><td>Fro</td><td>Fog</td><td>Bri</td><td>Con</td><td>Ela</td><td>Pix</td><td>JPG</td></tr><tr><td colspan="10">ResNet50-GN</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Source</td><td>18.0</td><td>19.8</td><td>17.9</td><td>19.8</td><td>11.4</td><td>21.4</td><td>24.9</td><td>40.4</td><td>47.3</td><td>33.6</td><td>69.3</td><td>36.3</td><td>18.6</td><td>28.4</td><td>52.3</td><td>30.6</td></tr><tr><td>MEMO</td><td>18.5</td><td>20.5</td><td>18.4</td><td>17.1</td><td>12.6</td><td>21.8</td><td>26.9</td><td>40.4</td><td>47.0</td><td>34.4</td><td>69.5</td><td>36.5</td><td>19.2</td><td>32.1</td><td>53.3</td><td>31.2</td></tr><tr><td>DDA</td><td>42.4</td><td>43.3</td><td>42.3</td><td>16.6</td><td>19.6</td><td>21.9</td><td>26.0</td><td>35.7</td><td>40.1</td><td>13.7</td><td>61.2</td><td>25.2</td><td>37.5</td><td>46.6</td><td>54.1</td><td>35.1</td></tr><tr><td>Tent</td><td>2.5</td><td>2.9</td><td>2.5</td><td>13.5</td><td>3.6</td><td>18.6</td><td>17.6</td><td>15.3</td><td>23.0</td><td>1.4</td><td>70.4</td><td>42.2</td><td>6.2</td><td>49.2</td><td>53.8</td><td>21.5</td></tr><tr><td>EATA</td><td>24.9</td><td>28.0</td><td>25.8</td><td>18.3</td><td>17.0</td><td>31.2</td><td>29.8</td><td>42.5</td><td>44.1</td><td>41.3</td><td>70.9</td><td>44.2</td><td>27.6</td><td>46.8</td><td>55.4</td><td>36.5</td></tr><tr><td>SAR</td><td>25.5</td><td>28.0</td><td>24.9</td><td>18.7</td><td>16.3</td><td>28.6</td><td>31.4</td><td>46.2</td><td>44.9</td><td>33.4</td><td>72.8</td><td>44.3</td><td>15.3</td><td>47.1</td><td>56.1</td><td>35.6</td></tr><tr><td>DeYO</td><td>41.2</td><td>44.3</td><td>42.5</td><td>22.4</td><td>24.7</td><td>41.8</td><td>21.9</td><td>54.8</td><td>51.6</td><td>21.9</td><td>73.1</td><td>53.2</td><td>48.5</td><td>59.8</td><td>59.6</td><td>44.1</td></tr><tr><td>ReCAP</td><td>42.5</td><td>44.4</td><td>42.9</td><td>19.4</td><td>25.0</td><td>42.2</td><td>44.0</td><td>49.7</td><td>52.4</td><td>57.5</td><td>72.9</td><td>53.6</td><td>29.5</td><td>60.4</td><td>60.0</td><td>46.4</td></tr><tr><td>MASA</td><td>43.2</td><td>44.8</td><td>43.4</td><td>23.3</td><td>25.5</td><td>42.8</td><td>42.9</td><td>55.5</td><td>52.6</td><td>58.3</td><td>73.2</td><td>54.0</td><td>45.7</td><td>60.7</td><td>60.1</td><td>48.4</td></tr><tr><td colspan="10">ViT-Base-LN</td><td colspan="7"></td></tr><tr><td>Source</td><td>9.5</td><td>6.7</td><td>8.2</td><td>29.0</td><td>23.4</td><td>33.9</td><td>27.1</td><td>15.9</td><td>26.5</td><td>47.2</td><td>54.7</td><td>44.1</td><td>30.5</td><td>44.5</td><td>47.8</td><td>29.9</td></tr><tr><td>MEMO</td><td>21.6</td><td>17.3</td><td>20.6</td><td>37.1</td><td>29.6</td><td>40.4</td><td>34.4</td><td>24.9</td><td>34.7</td><td>55.1</td><td>64.8</td><td>54.9</td><td>37.4</td><td>55.4</td><td>57.6</td><td>39.1</td></tr><tr><td>DDA</td><td>41.3</td><td>41.1</td><td>40.7</td><td>24.4</td><td>27.2</td><td>30.6</td><td>26.9</td><td>18.3</td><td>27.5</td><td>34.6</td><td>50.1</td><td>32.4</td><td>42.3</td><td>52.2</td><td>52.6</td><td>36.1</td></tr><tr><td>Tent</td><td>42.2</td><td>1.0</td><td>43.3</td><td>52.4</td><td>48.2</td><td>55.5</td><td>50.5</td><td>16.5</td><td>16.9</td><td>66.4</td><td>74.9</td><td>64.7</td><td>51.6</td><td>67.0</td><td>64.3</td><td>47.7</td></tr><tr><td>EATA</td><td>30.1</td><td>24.6</td><td>34.2</td><td>44.3</td><td>39.6</td><td>48.4</td><td>42.4</td><td>38.1</td><td>46.0</td><td>60.7</td><td>65.8</td><td>61.2</td><td>46.7</td><td>57.8</td><td>59.5</td><td>46.6</td></tr><tr><td>SAR</td><td>42.7</td><td>39.5</td><td>41.9</td><td>54.6</td><td>51.2</td><td>58.3</td><td>54.4</td><td>60.2</td><td>54.7</td><td>70.3</td><td>75.9</td><td>66.8</td><td>58.4</td><td>69.5</td><td>66.3</td><td>57.6</td></tr><tr><td>DeYO</td><td>53.4</td><td>50.4</td><td>55.0</td><td>58.7</td><td>59.5</td><td>64.5</td><td>52.5</td><td>68.1</td><td>66.3</td><td>73.8</td><td>78.3</td><td>67.9</td><td>68.9</td><td>73.8</td><td>70.8</td><td>64.1</td></tr><tr><td>ReCAP</td><td>53.5</td><td>56.7</td><td>56.9</td><td>59.2</td><td>60.5</td><td>65.3</td><td>64.0</td><td>69.6</td><td>67.2</td><td>74.1</td><td>78.4</td><td>64.6</td><td>70.2</td><td>74.4</td><td>71.5</td><td>65.7</td></tr><tr><td>MASA</td><td>54.4</td><td>57.2</td><td>57.7</td><td>60.0</td><td>60.9</td><td>66.0</td><td>63.5</td><td>70.4</td><td>67.5</td><td>74.9</td><td>78.6</td><td>68.9</td><td>71.2</td><td>74.0</td><td>72.3</td><td>66.5</td></tr></table>

Table 3: Mixed-domain evaluation on ImageNet-C. We report accuracy (%) for ResNet50-GN and ViT-Base-LN at corruption severities 5 and 4. Avg averages the results across the two severities.
<table><tr><td colspan="4">ResNet50-GN</td><td colspan="4">ViT-Base-LN</td></tr><tr><td>Method</td><td>Lv.5</td><td>Lv.4</td><td>Avg</td><td>Method</td><td>Lv.5</td><td>Lv.4</td><td>Avg</td></tr><tr><td>Source</td><td>30.6</td><td>42.7</td><td>36.7</td><td>Source</td><td>29.9</td><td>42.9</td><td>36.4</td></tr><tr><td>MEMO</td><td>31.2</td><td>43.0</td><td>37.1</td><td>MEMO</td><td>39.1</td><td>51.3</td><td>45.2</td></tr><tr><td>DDA</td><td>35.1</td><td>43.6</td><td>39.4</td><td>DDA</td><td>36.1</td><td>45.1</td><td>40.6</td></tr><tr><td>Tent</td><td>13.4</td><td>20.6</td><td>17.0</td><td>Tent</td><td>16.5</td><td>64.3</td><td>40.4</td></tr><tr><td>EATA</td><td>38.1</td><td>47.7</td><td>42.9</td><td>EATA</td><td>55.7</td><td>63.7</td><td>59.7</td></tr><tr><td>SAR</td><td>38.3</td><td>48.6</td><td>43.5</td><td>SAR</td><td>57.1</td><td>64.9</td><td>61.0</td></tr><tr><td>DeYO</td><td>38.6</td><td>50.2</td><td>44.4</td><td>DeYO</td><td>59.4</td><td>66.8</td><td>63.1</td></tr><tr><td>ReCAP</td><td>41.5</td><td>51.2</td><td>46.4</td><td>ReCAP</td><td>59.4</td><td>67.1</td><td>63.3</td></tr><tr><td>MASA</td><td>42.3</td><td>52.5</td><td>47.4</td><td>MASA</td><td>59.8</td><td>67.6</td><td>63.7</td></tr></table>

Limited-batch adaptation. With batch size one, each update lacks within-batch support. MASA obtains 48.4%/66.5% accuracy on ResNet/ViT, improving over ReCAP by 2.0/0.8 points. It ranks first on 13 of 15 corruptions for both backbones, showing gains across corruption families. ResNet improves most on elastic transform (+16.2) and snow (+5.8), but drops by 1.1 points on zoom blur. On ViT, contrast gains 4.3 points, with slight losses on zoom blur and pixelation. This suggests that prototype and semantic evidence is useful when the update target is unreliable.

Time-varying label shift. Changing class frequencies can reinforce errors on transiently dominant classes. MASA reaches 48.2%/64.0% on ResNet/ViT, exceeding ReCAP by 2.4/1.0 points. On ResNet, it ranks first on 13 corruptions and ties on brightness. Mean gains are positive across all four corruption families (1.1–4.0 points), with elastic transform and snow contributing the most and zoom blur remaining an exception. On ViT, MASA ranks first on 12 corruptions; shot noise improves by 9.5 points over ReCAP but remains 0.7 points below SAR, while pixelation and JPEG compression decrease slightly. The broad gains suggest that reliability filtering helps retain useful memory evidence as the stream prior changes, while the exceptions reveal continued dependence on the current representation.

Mixed-domain adaptation. When the corruption changes, earlier evidence can become less relevant. MASA nonetheless leads ReCAP by 1.0/0.4 points on ResNet/ViT, although by smaller margins than in the other protocols. The advantages persist at both severity levels (0.8/1.3 points on ResNet and 0.4/0.5 on ViT at levels 5/4), suggesting that memory remains complementary across heterogeneous shifts.

Feature-space organization. Fig. 4 qualitatively compares the adapted representations. Relative to ReCAP, MASA forms more compact local groups with less intermingling in the highlighted regions, consistent with descriptor-aware retrieval aligning compatible observations with shared prototype targets.

Table 2: Label-shift evaluation on ImageNet-C at severity 5.
<table><tr><td></td><td colspan="3">Noise</td><td colspan="4">Blur</td><td colspan="4">Weather</td><td colspan="4">Digital</td><td rowspan="2">Avg</td></tr><tr><td>Method</td><td>Gau</td><td>Sho</td><td>Imp</td><td>Def</td><td>Gla</td><td>Mot</td><td>Zoo</td><td>Sno</td><td>Fro</td><td>Fog</td><td>Bri</td><td>Con</td><td>Ela</td><td>Pix</td><td>JPG</td></tr><tr><td colspan="10">ResNet50-GN</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Source</td><td>17.9</td><td>19.9</td><td>17.9</td><td>19.7</td><td>11.3</td><td>21.3</td><td>24.9</td><td>40.4 47.4</td><td></td><td>33.6</td><td>69.2</td><td>36.3</td><td>18.7</td><td>28.4</td><td>52.2</td><td>30.6</td></tr><tr><td>MEMO</td><td>18.4</td><td>20.6</td><td>18.4</td><td>17.1</td><td>12.7</td><td>21.8</td><td>26.9</td><td>40.7</td><td>46.9</td><td>34.8</td><td>69.6</td><td>36.4</td><td>19.2</td><td>32.2</td><td>53.4</td><td>31.3</td></tr><tr><td>DDA</td><td>42.5</td><td>43.4</td><td>42.3</td><td>16.5</td><td>19.4</td><td>21.9</td><td>26.1</td><td>35.8</td><td>40.2</td><td>13.7</td><td>61.3</td><td>25.2</td><td>37.3</td><td>46.9</td><td>54.3</td><td>35.1</td></tr><tr><td>Tent</td><td>2.6</td><td>3.3</td><td>2.7</td><td>13.9</td><td>7.9</td><td>19.5</td><td>28.7</td><td>16.5</td><td>21.9</td><td>1.8</td><td>70.5</td><td>42.2</td><td>6.6</td><td>49.4</td><td>53.7</td><td>22.8</td></tr><tr><td>EATA</td><td>27.2</td><td>28.5</td><td>28.4</td><td>15.1</td><td>16.7</td><td>24.6</td><td>25.5</td><td>32.5</td><td>32.2</td><td>40.0</td><td>66.5</td><td>33.2</td><td>24.1</td><td>42.2</td><td>38.6</td><td>31.7</td></tr><tr><td>SAR</td><td>34.0</td><td>36.7</td><td>36.2</td><td>21.8</td><td>20.9</td><td>33.2</td><td>32.4</td><td>38.7</td><td>45.6</td><td>50.6</td><td>72.9</td><td>46.8</td><td>14.3</td><td>52.2</td><td>56.8</td><td>39.5</td></tr><tr><td>DeYO</td><td>41.7</td><td>44.0</td><td>42.5</td><td>23.4</td><td>23.9</td><td>41.3</td><td>13.0</td><td>53.9</td><td>52.2</td><td>38.6</td><td>73.1</td><td>52.3</td><td>46.8</td><td>59.3</td><td>59.1</td><td>44.3</td></tr><tr><td>ReCAP</td><td>42.0</td><td>44.1</td><td>42.7</td><td>19.8</td><td>24.3</td><td>39.7</td><td>40.2</td><td>46.0</td><td>52.2</td><td>57.3</td><td>73.1</td><td>52.4</td><td>33.7</td><td>59.4</td><td>59.5</td><td>45.8</td></tr><tr><td>MASA</td><td>43.3</td><td>45.1</td><td>43.7</td><td>24.2</td><td>25.8</td><td>42.9</td><td>37.6</td><td>54.5</td><td>52.4</td><td>59.7</td><td>73.1</td><td>53.2</td><td>47.6</td><td>60.3</td><td>59.7</td><td>48.2</td></tr><tr><td colspan="10">ViT-Base-LN</td><td colspan="7"></td></tr><tr><td>Source</td><td>9.4</td><td>6.7</td><td>8.3</td><td>29.1</td><td>23.4</td><td>34.0</td><td>27.0</td><td>15.8</td><td>26.3</td><td>47.4</td><td>54.7</td><td>43.9</td><td>30.5</td><td>44.5</td><td>47.6</td><td>29.9</td></tr><tr><td>MEMO</td><td>21.6</td><td>17.4</td><td>20.6</td><td>37.1</td><td>29.6</td><td>40.6</td><td>34.4</td><td>25.0</td><td>34.8</td><td>55.2</td><td>65.0</td><td>54.9</td><td>37.4</td><td>55.5</td><td>57.7</td><td>39.1</td></tr><tr><td>DDA</td><td>41.3</td><td>41.3</td><td>40.6</td><td>24.6</td><td>27.4</td><td>30.7</td><td>26.9</td><td>18.2</td><td>27.7</td><td>34.8</td><td>50.0</td><td>32.3</td><td>42.2</td><td>52.5</td><td>52.7</td><td>36.2</td></tr><tr><td>Tent</td><td>32.7</td><td>1.4</td><td>34.6</td><td>54.4</td><td>52.3</td><td>58.2</td><td>52.2</td><td>7.7</td><td>12.0</td><td>69.3</td><td>76.1</td><td>66.1</td><td>56.7</td><td>69.4</td><td>66.4</td><td>47.3</td></tr><tr><td>EATA</td><td>35.8</td><td>34.8</td><td>36.8</td><td>45.1</td><td>47.3</td><td>49.3</td><td>47.8</td><td>56.6</td><td>55.5</td><td>62.1</td><td>72.3</td><td>21.6</td><td>56.0</td><td>64.6</td><td>63.7</td><td>50.0</td></tr><tr><td>SAR</td><td>48.2</td><td>48.7</td><td>49.0</td><td>55.4</td><td>54.5</td><td>59.2</td><td>54.3</td><td>55.8</td><td>54.5</td><td>70.0</td><td>76.9</td><td>66.1</td><td>62.2</td><td>70.2</td><td>66.5</td><td>59.4</td></tr><tr><td>DeYO</td><td>53.0</td><td>34.4</td><td>48.8</td><td>57.6</td><td>58.5</td><td>63.3</td><td>35.4</td><td>67.4</td><td>66.0</td><td>73.0</td><td>77.7</td><td>66.6</td><td>68.1</td><td>73.1</td><td>69.8</td><td>60.8</td></tr><tr><td>ReCAP</td><td>53.1</td><td>38.5</td><td>49.6</td><td>57.3</td><td>59.0</td><td>63.8</td><td>60.7</td><td>67.8</td><td>66.3</td><td>72.9</td><td>77.7</td><td>66.8</td><td>68.2</td><td>73.0</td><td>70.0</td><td>63.0</td></tr><tr><td>MASA</td><td>53.9</td><td>48.0</td><td>50.2</td><td>58.5</td><td>59.3</td><td>64.3</td><td>60.9</td><td>68.5</td><td>66.4</td><td>73.5</td><td>77.8</td><td>67.4</td><td>69.0</td><td>72.7</td><td>69.6</td><td>64.0</td></tr></table>

Table 4: Component ablation and prototype-discrepancy decomposition under label shift (average accuracy, %).
<table><tr><td>Config</td><td>RN50</td><td>ViT</td><td>Loss terms</td><td>RN50</td><td>ViT</td></tr><tr><td>Full</td><td>48.2</td><td>64.0</td><td>Visual only</td><td>46.8</td><td>63.3</td></tr><tr><td>- proto-loss</td><td>46.5</td><td>63.2</td><td>+ predictive</td><td>47.4</td><td>63.6</td></tr><tr><td>- propagation</td><td>47.2</td><td>63.6</td><td>+ semantic</td><td>48.2</td><td>64.0</td></tr><tr><td>— reliability</td><td>46.9</td><td>63.5</td><td></td><td></td><td></td></tr><tr><td>- PLPD</td><td>47.6</td><td>63.8</td><td></td><td></td><td></td></tr></table>

Ablation studies. Table 4 isolates the main components and loss terms. Removing the prototype loss causes the largest decrease among the tested removals (1.7/0.8 points on ResNet/ViT). Reliability filtering, propagation, and PLPD cause smaller drops of 1.3/0.5, 1.0/0.4, and 0.6/0.2 points. From visual matching at 46.8%/63.3%, predictive consistency adds 0.6/0.3 points and semantic agreement adds another 0.8/0.4 points to reach 48.2%/64.0%. These monotonic gains support the complementarity of the three cues; the removals further show the value of filtering and propagation before prototype regularization. Additional ablations are provided in the supplementary material.

## Conclusion

In this paper, we propose MASA, a WTTA framework designed to address the error reinforcement that can arise when adaptation relies only on evidence from the shifted model. MASA selects sparse, reliability-ranked and visually diverse anchors, obtains structured object and nuisance descriptions from a frozen MLLM, and propagates these descriptors through local feature neighborhoods. We further consolidate visual, predictive, and semantic evidence in a bounded prototype memory, where descriptor-aware retrieval supplies auxiliary targets for normalization-afine adaptation without treating MLLM outputs as class labels. Experiments across limited-batch, mixed-domain, and time-varying label-shift protocols demonstrate consistent improvements with both ResNet and ViT, while the ablations support the complementary roles of prototype consistency, propagation, reliability filtering, and semantic agreement. We hope this work encourages WTTA research to move beyond model-internal signals and explore sparse semantic grounding for online adaptation.

## References

Anonymous. 2024a. Empowering Source-Free Domain Adaptation via MLLM-Guided Reliability-Based Curriculum Learning. arXiv:2405.18376.

Anonymous. 2024b. Exploring Structured Semantic Priors Underlying Difusion Score for Test-Time Adaptation. In Advances in Neural Information Processing Systems (NeurIPS).

Bai, Y.; Han, Z.; Cao, B.; Jiang, X.; Hu, Q.; and Zhang, C. 2024. ID-like Prompt Learning for Few-Shot Out-of-Distribution Detection. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).

Boudiaf, M.; Mueller, R.; Ben Ayed, I.; and Bertinetto, L. 2022. Parameter-free Online Test-time Adaptation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).

Cao, C.; Zhong, Z.; Zhou, Z.; Liu, Y.; Liu, T.; and Han, B. 2024. Envisioning Outlier Exposure by Large Language Models for Out-of-Distribution Detection. arXiv:2406.00806.

Chen, D.; Wang, D.; Darrell, T.; and Ebrahimi, S. 2022. Contrastive Test-Time Adaptation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).

Choi, S.; Yang, S.; Choi, S.; and Yun, S. 2022. Improving Test-Time Adaptation via Shift-Agnostic Weight Regularization and Nearest Source Prototypes. In European Conference on Computer Vision (ECCV).

Dosovitskiy, A.; Beyer, L.; Kolesnikov, A.; Weissenborn, D.; Zhai, X.; Unterthiner, T.; Dehghani, M.; Minderer, M.; Heigold, G.; Gelly, S.; Uszkoreit, J.; and Houlsby, N. 2021. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. In International Conference on Learning Representations (ICLR).

Gao, J.; Zhang, J.; Liu, X.; Darrell, T.; Shelhamer, E.; and Wang, D. 2022. Back to the Source: Difusion-Driven Adaptation to Test-Time Corruption. arXiv:2207.03442.

He, K.; Zhang, X.; Ren, S.; and Sun, J. 2016. Deep Residual Learning for Image Recognition. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR).

Hendrycks, D.; and Dietterich, T. 2019. Benchmarking Neural Network Robustness to Common Corruptions and Perturbations. In International Conference on Learning Representations (ICLR).

Hu, Z.; Hu, Y.; Li, X.; Tang, S.; and Duan, L.-Y. 2025. Beyond Entropy: Region Confidence Proxy for Wild Test-Time Adaptation. In International Conference on Machine Learning (ICML).

Iwasawa, Y.; and Matsuo, Y. 2021. Test-Time Classifier Adjustment Module for Model-Agnostic Domain Generalization. In Advances in Neural Information Processing Systems (NeurIPS).

Jiang, X.; Liu, F.; Fang, Z.; Chen, H.; Liu, T.; Zheng, F.; and Han, B. 2024. Negative Label Guided OOD Detection with Pretrained Vision-Language Models. arXiv:2403.20078.

Lee, J.; Jung, D.; Lee, S.; Park, J.; Shin, J.; Hwang, U.; and Yoon, S. 2024. Entropy is not Enough for Test-Time Adaptation: From the Perspective of Disentangled Factors.

In International Conference on Learning Representations (ICLR).

Li, J.; Li, D.; Savarese, S.; and Hoi, S. 2023. BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models. In International Conference on Machine Learning (ICML).

Li, T.; Pang, G.; Bai, X.; Miao, W.; and Zheng, J. 2024. Learning Transferable Negative Prompts for Out-of-Distribution Detection. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).

Liu, H.; Li, C.; Wu, Q.; and Lee, Y. J. 2023. Visual Instruction Tuning. In Advances in Neural Information Processing Systems (NeurIPS).

Ming, Y.; Cai, Z.; Gu, J.; Sun, Y.; Li, W.; and Li, Y. 2022. Delving into Out-of-Distribution Detection with Vision-Language Representations. In Advances in Neural Information Processing Systems (NeurIPS).

Miyai, A.; Yu, Q.; Irie, G.; and Aizawa, K. 2024. LoCoOp: Few-Shot Out-of-Distribution Detection via Prompt Learning. In Advances in Neural Information Processing Systems (NeurIPS).

Niu, S.; Wu, J.; Zhang, Y.; Chen, Y.; Zheng, S.; Zhao, P.; and Tan, M. 2022. Eficient Test-Time Model Adaptation without Forgetting. In International Conference on Machine Learning (ICML).

Niu, S.; Wu, J.; Zhang, Y.; Wen, Z.; Chen, Y.; Zhao, P.; and Tan, M. 2023. Towards Stable Test-Time Adaptation in Dynamic Wild World. In International Conference on Learning Representations (ICLR).

Radford, A.; Kim, J. W.; Hallacy, C.; Ramesh, A.; Goh, G.; Agarwal, S.; Sastry, G.; Askell, A.; Mishkin, P.; Clark, J.; Krueger, G.; and Sutskever, I. 2021. Learning Transferable Visual Models from Natural Language Supervision. In International Conference on Machine Learning (ICML).

Song, J.; Lee, J.; Kweon, I. S.; and Choi, S. 2023. EcoTTA: Memory-Eficient Continual Test-Time Adaptation via Self-Distilled Regularization. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).

Wang, D.; Shelhamer, E.; Liu, S.; Olshausen, B.; and Darrell, T. 2021. Tent: Fully Test-Time Adaptation by Entropy Minimization. In International Conference on Learning Representations (ICLR).

Wang, H.; Li, Y.; Yao, H.; and Li, X. 2023. CLIPN for Zero-Shot OOD Detection: Teaching CLIP to Say No. In IEEE/CVF International Conference on Computer Vision (ICCV).

Wang, P.; Bai, S.; Tan, S.; Wang, S.; Fan, Z.; Bai, J.; Chen, K.; Liu, X.; Wang, J.; Ge, W.; Fan, Y.; Dang, K.; Du, M.; Ren, X.; Men, R.; Liu, D.; Zhou, C.; Zhou, J.; and Lin, J. 2024. Qwen2-VL: Enhancing Vision-Language Model’s Perception of the World at Any Resolution. arXiv:2409.12191.

Wang, Q.; Fink, O.; Van Gool, L.; and Dai, D. 2022. Continual Test-Time Domain Adaptation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).

Yang, Y.; Zhu, L.; Sun, Z.; Liu, H.; Gu, Q.; and Ye, N. 2025. OODD: Test-time Out-of-Distribution Detection with Dynamic Dictionary. arXiv:2503.10468.

Zhang, M.; Levine, S.; and Finn, C. 2022. MEMO: Test Time Robustness via Adaptation and Augmentation. In Advances in Neural Information Processing Systems (NeurIPS).

Zhang, Y.; and Zhang, L. 2024. AdaNeg: Adaptive Negative Proxy Guided OOD Detection with Vision-Language Models. In Advances in Neural Information Processing Systems (NeurIPS).

Zhang, Y.; Zhu, W.; He, C.; and Zhang, L. 2024. LAPT: Label-driven Automated Prompt Tuning for OOD Detection with Vision-Language Models. In European Conference on Computer Vision (ECCV).

Zhu, W.; Zhang, Y.; Jin, X.; Zeng, W.; and Zhang, L. 2026. ANTS: Adaptive Negative Textual Space Shaping for OOD Detection via Test-Time MLLM Understanding and Reasoning. arXiv:2509.03951.

# Sparse MLLM Anchors, Dense Adaptation: Breaking the Self-Referential Loop in Wild Test-Time Adaptation

Appendix

Appendix organization. Appendix A reviews closely related work. Appendix B gives the complete online procedure, Appendix C specifies the implementation, Appendix D presents additional ablations and analyses, and Appendix E discusses limitations.

## A Related Work

## A.1 Wild Test-time Adaptation

TTA updates a pretrained model using unlabeled test data to address distribution shifts encountered during inference (Wang et al. 2021; Boudiaf et al. 2022; Wang et al. 2022; Chen et al. 2022; Song et al. 2023). WTTA focuses on online conditions such as single-sample batches, mixed-domain streams, and time-varying class imbalance (Niu et al. 2023; Hu et al. 2025). Methods designed for these conditions stabilize adaptation in several ways. Sample-selection approaches filter updates using entropy, sharpness, or perturbation sensitivity (Niu et al. 2022, 2023; Lee et al. 2024). Memoryand prototype-based methods accumulate structure from the stream (Iwasawa and Matsuo 2021; Choi et al. 2022; Zhang and Zhang 2024), while ReCAP (Hu et al. 2025) replaces pointwise entropy with confidence and consistency over local feature regions. Despite these diferences, the evidence used for adaptation is derived predominantly from the predictions or representations of the model being adapted. Input restoration (Gao et al. 2022) and difusion-score distillation (Anonymous 2024b) provide additional priors, but their guidance remains implicit at the pixel or score level. MASA instead uses explicit descriptions of image content and nuisance factors as complementary adaptation evidence.

## A.2 MLLMs as External Semantic Knowledge

Vision–language models connect recognition with semantic concepts expressed in text. OOD methods use text to represent negative concepts, outliers, or visually confusing classes, and to refine the scoring of image–text evidence (Cao et al. 2024; Jiang et al. 2024; Ming et al. 2022; Wang et al. 2023; Miyai et al. 2024; Li et al. 2024; Bai et al. 2024; Zhang et al. 2024; Yang et al. 2025). ANTS (Zhu et al. 2026) further uses descriptions from a frozen MLLM to update the negative textual space of a frozen, CLIP-native detector. MLLMs have also served as pseudo-label generators in ofline source-free adaptation (Anonymous 2024a). MASA assigns a diferent role to language outputs. It neither places them in the classifier’s decision space nor treats them as class labels. Instead, sparse MLLM descriptions characterize selected test samples and provide persistent guidance for online adaptation of a closed-set classifier that need not be text aligned.

## B Complete Online Procedure

Algorithm 1 Online MASA at test step t   
Require: Batch $X _ { t } ; \theta _ { t } ;$ incoming states and auxiliary histories   
1: Compute pre-update ${ \bf o } _ { i } , { \bf p } _ { i } , { \bf z } _ { i } , { \bf v } _ { i } , \ell _ { \mathrm { R E } , i } ,$ and $\ell _ { \mathrm { R I } , i }$   
2: Compute $m _ { i } , \phi _ { i } .$ , and q<sub>i</sub>; append $\{ \mathsf { r } _ { t , i } \} _ { i = 1 } ^ { B _ { t } }$ to form $\widetilde { \mathcal { W } } _ { t }$   
3: if semantic refresh is triggered then   
4: Rank by Eq. (10); diversify by farthest-point sampling   
5: Query and encode anchors; append to $\hat { \boldsymbol A } _ { t }$ and immediately   
upsert $\mathcal { M } _ { t }$   
6: end if   
7: Select by Eq. (5); compute ${ \mathcal { L } } _ { \mathrm { r e g i o n } }$   
8: Propagate, retrieve, and compute $\scriptstyle { \mathcal { L } } _ { \mathrm { p r o t o } }$   
9: Take one step on Eq. (25) to obtain $\vartheta _ { t + 1 }$   
10: Sequentially write the batch; then maintain and age memory   
11: Update <sup>¯</sup>ℓ by Eq. (26); recover if $\bar { \ell } _ { t } < \tau _ { \mathrm { r e c } }$   
12: return Pre-update logits o<sub>i</sub>

## C Implementation Details

We use the released WTTA implementation with ResNet50- GN (He et al. 2016) and ViT-Base-LN (Dosovitskiy et al. 2021) from timm. Optimization uses SGD with momentum 0.9, base learning rates 0.00025/0.001 for ResNet/ViT with the released batch-size adjustment, and one normalizationafine update per batch. We optimize BN/LN/GN afine parameters while excluding ResNet layer 4, ViT blocks 9–11, and the final normalization layer. The normalized memory feature is extracted by an additional forward pass before the update.

Following ReCAP (Hu et al. 2025), the regional path loads the fixed, backbone-specific diagonal feature statistic distributed with its implementation. It does not sample or retain a calibration subset. We use $\rho _ { \mathrm { r e g } } = 1 2 , \lambda _ { \delta } = \bar { 5 } \times 1 0 ^ { - 4 }$ $\lambda _ { \mathrm { R I } } ~ = ~ 0 . 5 , ~ P ~ = ~ 4 .$ , and $\tau _ { \mathrm { P L P D } } ~ = ~ 0 . 2$ . For ResNet, $\tau _ { \mathrm { { R E } } } = 0 . 8 \log C$ and $\omega _ { \mathrm { m a x } } ^ { \mathrm { r e g } } = 3$ . For ViT, the corresponding values are log C and 1.5. In the batch-size-one protocol, the cap is set to $\omega _ { \mathrm { m a x } } ^ { \mathrm { r e g } } = 5$ . The reliability gate uses $H _ { \mathrm { h i s t } } = 3 2$ and $( \tau _ { \mathrm { R I } } ^ { \mathrm { a n c } } , \tau _ { m } , \tau _ { \phi } ) = ( 1 0 , 0 . 0 5 , 0 . 5 )$ . Anchor selection uses $( c _ { \mathrm { c l i p } } , \dot { B } _ { a } , r _ { \mathrm { p o o l } } ) = ( 2 , 2 , 8 )$

Unless otherwise stated, MASA uses frozen Qwen2-VL-2B (Wang et al. 2024) and CLIP ViT-B/16 (Radford et al. 2021), with no classifier hint. We set $( \kappa _ { \operatorname* { m i n } } , K _ { a } ) = ( 0 . 1 , 4 )$ and $( \tau _ { a } , \epsilon c ) = ( 0 . 0 7 , 0 . 1 5 )$ for semantic grounding and propagation. Prototype retrieval uses $( \tau _ { q } , \tau _ { \mathrm { a s s i g n } } ) = ( 0 . 1 , 0 . 6 )$ and $( \alpha _ { \mathrm { a g e } } , \alpha _ { \mathrm { c a n d } } ) = ( 0 . 0 2 , 0 . 0 5 )$ , with $\tau _ { \omega } = 0 . 1$ for prototype weighting. We set $\lambda _ { \mathrm { p r o t o } } = 0 . 2$ and $( \rho _ { \mathrm { r e c } } , \tau _ { \mathrm { r e c } } ) =$ (0.9, 0.4) for the objective and recovery. Memory updates use $( \eta _ { 0 } , \tau _ { \mathrm { s t o r e } } , \lambda _ { \mathrm { s t a t } } ) \stackrel { - } { = } ( 0 . 2 5 , 0 . 5 , 0 . 0 5 )$ and $n _ { \mathrm { m i n } } = 2$ . Finally, $K _ { \mathrm { m a x } } ~ = ~ 6 4 , ~ ( H _ { D } , H _ { \mathrm { w i n } } , H _ { \mathrm { c o v } } , T _ { \mathrm { r e f } } ) ~ = ~ ( 8 , 1 2 8 , 1 2 8 , 6 4 )$ and $( \tau _ { D } , \tau _ { \mathrm { c o v } } ) = ( 0 . 4 5 , 0 . 2 5 )$ . Baseline results in Tables 1, 2, and 3 are taken from ReCAP (Hu et al. 2025) under the same benchmark.

![](images/e28234d7c3fe4a247a10ef2f7a6b343419b9d0744ab12709ad805855509f401c.jpg)  
Figure 5: Hyperparameter sensitivity under label shift. Average accuracy (%) with ResNet50-GN when varying (a) the prototype-loss weight, (b) the periodic refresh interval, and (c) the memory capacity. Each sweep changes one parameter while fixing the others. Red squares mark the default settings, and the dashed line shows the ReCAP baseline under the same protocol.

## D Additional Ablations and Analyses

Beyond the component removals in Table 4, we examine where the semantic gains originate and how MASA behaves under its main control parameters.

## D.1 Semantic Source and Class Information

We vary the semantic source while keeping the remaining components and hyperparameters fixed (Table 5). The comparison ranges from no semantic input and a deterministic text-hash control to BLIP-2 (Li et al. 2023), LLaVA-1.5-7B (Liu et al. 2023), and Qwen2-VL-2B (Wang et al. 2024). The hash control maps descriptor tokens to fixed vectors without a learned text encoder. Table 6 separately measures the efect of providing class hints to the MLLM, including a ground-truth oracle that is not used by MASA.

Table 5: Efect of the semantic source on average accuracy (%) under each WTTA protocol. Here, bs1, mix, and lbl denote limited-batch, mixed-domain, and label-shift evaluation.
<table><tr><td>Semantic source</td><td>ResNet50-GN bs1 mix lbl</td><td>bs1</td><td>ViT-Base-LN mix</td><td>lbl</td></tr><tr><td>None</td><td>47.0</td><td>46.7 47.4</td><td>65.9 63.3</td><td>63.6</td></tr><tr><td>Hash (text)</td><td>47.2</td><td>46.8 47.3</td><td>66.0</td><td>63.2 63.5</td></tr><tr><td>BLIP-2</td><td>47.7</td><td>47.0 47.7</td><td>66.2 63.4</td><td>63.7</td></tr><tr><td>LLaVA-1.5-7B</td><td>48.1</td><td>47.2 47.9</td><td>66.3</td><td>63.5 63.8</td></tr><tr><td>Qwen2-VL-2B</td><td>48.4</td><td>47.4 48.2</td><td>66.5 63.7</td><td>64.0</td></tr></table>

Efect of the semantic source. The None and Hash controls remain close under every protocol: hashing changes accuracy by at most 0.2 points and slightly reduces label-shift performance on both backbones. Token identity alone therefore provides little useful structure for prototype retrieval. Learned semantic sources give a diferent pattern. Performance increases consistently from BLIP-2 to LLaVA-1.5-7B and Qwen2-VL-2B across all six backbone–protocol combinations. Relative to using no semantics, Qwen2-VL-2B improves limited-batch, mixed-domain, and label-shift accuracy by 1.4/0.7/0.8 points on ResNet and 0.6/0.4/0.4 points on ViT. The larger ResNet gain, particularly with batch size one, may reflect both its lower starting accuracy and the absence of support from other samples in the current batch. The smaller but consistent ViT gains show that semantic evidence remains complementary when the visual representation is already stronger. Overall, the ordering is consistent with more informative descriptions producing better semantic matches, rather than the benefit arising from an arbitrary extra code.

Table 6: Efect of class information in the MLLM prompt. Results are average accuracy (%) under label shift. MASA uses the no-hint setting.
<table><tr><td>Hint to MLLM</td><td>RN50</td><td>ViT</td></tr><tr><td>None</td><td>48.2</td><td>64.0</td></tr><tr><td>Predicted class</td><td>47.9</td><td>63.8</td></tr><tr><td>Ground-truth class (oracle)</td><td>49.1</td><td>64.6</td></tr></table>

Efect of class information. The no-hint setting reaches 48.2%/64.0% on ResNet/ViT, whereas conditioning the MLLM on the classifier’s predicted class lowers accuracy by 0.3/0.2 points. A predicted hint can steer the description toward the model’s current belief, including an incorrect one, and thereby partially reintroduce the feedback that external semantics is intended to reduce. In contrast, the ground-truth oracle reaches 49.1%/64.6%, showing that correct category information can make the resulting descriptors more discriminative. The gap between predicted and oracle hints emphasizes that the value of class information depends on its correctness. MASA therefore uses no class hint: this retains an image-only semantic query while avoiding unavailable oracle information.

## D.2 Hyperparameter Sensitivity

Figure 5 examines three parameters with distinct roles: the prototype-loss weight $\lambda _ { \mathrm { p r o t o } } ,$ the periodic refresh interval $T _ { \mathrm { r e f } }$ , and the memory budget $K _ { \mathrm { m a x } }$ . These sweeps characterize the balance between regional and prototype adaptation, the accuracy–query-cost trade-of, and sensitivity to memory capacity.

Sensitivity trends. The prototype weight has a broad optimum around the default $\lambda _ { \mathrm { p r o t o } } = 0 . 2$ . Reducing it to 0.1 or increasing it to 0.4 changes accuracy by only 0.3 and 0.2 points, respectively, whereas a larger value of 0.8 reduces accuracy to 47.5%. This decline is consistent with prototype consistency dominating the regional objective and over-constraining updates toward imperfect or stale memory entries. Refreshing every 16 steps gives 48.3%, only 0.1 points above the default interval of 64 but with four times as many periodic refresh opportunities. Longer intervals of 128 and 256 reduce accuracy to 47.8% and 47.6%, suggesting that descriptors become less representative as the stream evolves. Memory capacity shows a similar saturation pattern: increasing $K _ { \mathrm { m a x } }$ from 32 to 64 improves accuracy from 47.7% to 48.2%, while capacities of 128 and 256 yield 48.1% and 47.9%. A small memory cannot cover enough recurring structure, whereas a much larger one may preserve redundant or outdated clusters. The defaults therefore lie near the stable region of each sweep while balancing adaptation strength, refresh frequency, and memory capacity.

## E Limitations

First, our evaluation focuses on ImageNet-C classification. Synthetic corruptions provide controlled WTTA protocols, but they do not cover natural domain drift, open-set arrivals, or dense prediction tasks, where the usefulness and granularity of semantic descriptors may difer. Evaluating MASA on these settings is necessary to establish how broadly sparse semantic grounding transfers beyond image classification. Second, MASA requires a frozen MLLM and a text encoder. Sparse anchor queries amortize their cost across neighboring samples, but do not remove the added latency and memory footprint, which also depend on hardware and model-serving choices. Incorrect or underspecified descriptions may still enter the prototype memory despite reliability filtering. Lightweight semantic encoders, adaptive query budgets, and explicit descriptor uncertainty are promising directions. Third, external descriptions do not make the complete adaptation path independent of the classifier. Anchor ranking, propagation neighborhoods, and transition statistics still use its predictions and feature geometry. If the representation deteriorates severely or class identities alternate faster than the history window can track, these operations may select or propagate unreliable evidence. A more complete treatment should jointly model semantic uncertainty and rapid stream dynamics.