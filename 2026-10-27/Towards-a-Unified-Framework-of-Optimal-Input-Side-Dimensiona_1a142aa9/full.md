# Towards a Unified Framework of Optimal Input-Side Dimensional Configuration for Deep CTR Prediction

Chaohua Yang   
College of Computer Science and   
Software Engineering, Shenzhen   
University   
Shenzhen, China   
chaohua.ych@gmail.com   
Dugang Liu   
College of Computer Science and   
Software Engineering, Shenzhen   
University   
Shenzhen, China   
dugang.ldg@gmail.com   
Fuyuan Lyu   
McGill University   
Montreal, Canada   
fuyuan.lyu@mail.mcgill.ca

Xiuqiang He Shenzhen Technology University Shenzhen, China hexiuqiang@sztu.edu.cn

Xing Tang Shenzhen Technology University Shenzhen, China xing.tang@hotmail.com

Zhong Ming   
College of Computer Science and   
Software Engineering, Shenzhen   
University   
Shenzhen, China   
mingz@szu.edu.cn

## Abstract

Deep click-through rate (CTR) prediction has achieved substantial success through increasingly expressive feature interaction architectures. However, predictive performance also depends on how input-side information is configured before interaction modeling. Existing studies have shown that selecting informative samples, features, or embedding units can improve prediction accuracy. Nevertheless, they primarily optimize these selection mechanisms independently along a single input-side dimension, overlooking their shared basic form and the inherent dependencies across input levels, thereby limiting the efectiveness of input-side dimensional configuration. To address this issue, we propose OptIDC, a unified framework for Optimal Input-side Dimensional Configuration in deep CTR prediction. OptIDC jointly learns sample, feature, and embedding-unit level configurations within a unified framework by integrating their corresponding selection mechanisms. To explicitly coordinate selection decisions across diferent levels, OptIDC further introduces a selection-weighted Optimal Transport mechanism that aligns coarse- and fine-grained selections via transport over finite measures, thereby alleviating cross-level selection conflicts while preserving the semantics of each input level. Moreover, OptIDC is model-agnostic and can be seamlessly integrated into existing CTR models. Extensive experiments on four real-world benchmark datasets demonstrate that OptIDC consistently achieves higher AUC than existing input-side selection methods, validating the efectiveness of unified input-side dimensional configuration.

## CCS Concepts

• Information systems → Personalization.

## Keywords

Deep CTR Prediction, Input-side Dimensional Configuration, Cross-Level Coordination

## 1 Introduction

In web applications such as online advertising and recommendation systems, click-through rate (CTR) prediction estimates the probability that a user will click on a candidate item if displayed and supports content ranking. Much of the progress in deep CTR prediction has come from increasingly expressive architectures for modeling feature interactions [2, 30]. These architectures, however, operate on the information supplied to them. Real-world training data contain samples of uneven quality, feature fields of varying relevance, and embedding units that do not contribute equally to prediction. CTR prediction performance therefore depends on two complementary choices: how information is modeled by the architecture and what information is allowed to shape the model. We use the term input-side dimensional configuration for the latter, spanning the sample, feature, and embedding-unit levels.

![](images/1921fe918be676b741a914b565ff96bef74ca34f9316212ca752cd2e45515ddb.jpg)  
Figure 1: Overview of the input-side selection methods for deep CTR prediction.

Fig. 1 places existing input-side selection methods within a common deep CTR pipeline. At the sample level, instance selection and data denoising reweight or remove training examples [5, 11, 25]. At the feature level, field- and value-level selection methods learn gates that retain informative inputs before interaction modeling [10, 17, 28]. At the embedding-unit level, pruning and dimension search allocate representation capacity at a finer granularity [15, 16, 19, 32]. The gains reported across this substantial body of work confirm that input configuration can materially afect CTR prediction performance and eficiency. In most prior work, however, only one of these three selection paths is optimized at a time, while the other input levels are treated as fixed. Decisions that ultimately shape the same model input are therefore learned without explicit coordination across levels.

Despite their diferent names and granularities, the three selection mechanisms share the same basic form: each assigns a selection weight to an input component and uses that weight to control the component’s contribution to model learning. Sample selection assigns one weight to an entire instance, feature selection assigns weights to its field representations, and embedding-unit selection assigns weights to individual coordinates within those representations. When expressed over the embedding tensor, they can all be viewed as masks applied along diferent input axes. This common form also exposes their dependence. A sample contributes through its selected feature embeddings, and a feature contributes through its selected embedding units. If a feature is masked out, any unitlevel decisions within it become inefective; if nearly all feature representations of a highly weighted sample are suppressed, the sample-level decision has little efect. We therefore view the three mechanisms as coupled decisions about the same efective input, rather than as three unrelated selection problems.

Their common form motivates joint modeling. Blindly attaching all three selection mechanisms to the same backbone, however, leaves the final prediction error as their only cross-level signal. Sample weighting controls how much gradient an instance contributes, feature gating controls which field representations receive that gradient, and embedding-unit masking controls which coordinates carry it. Their updates can therefore interfere: a low sample weight may deprive the finer-grained mechanisms of useful feedback, while aggressive feature or embedding-unit masks can make a highly weighted sample almost inefective. Efective joint configuration must preserve useful specialization at each level and control severe cross-level mismatches, which we call selection conflicts. It must also coordinate outputs with diferent structures: a sample weight is a scalar, a feature gate is defined over fields, and an embedding-unit mask is defined over individual coordinates. These requirements expose the central challenge of unified inputside configuration: how can heterogeneous selection mechanisms be explicitly coordinated while preserving the granularity and semantics ofeach input level?

To tackle this challenge, we propose OptIDC (Optimal Inputside Dimensional Configuration), a unified framework that learns configurations at the sample, feature, and embedding-unit levels. OptIDC introduces a selection-weighted optimal transport (OT) formulation over finite measures for cross-level coordination [4]. For each coarse-to-fine relation, the coarse-level selection weight determines the total transported mass, the finer-grained weights allocate this mass over their semantic support, and representation dissimilarity defines the transport cost. This formulation accom modates diferent selection cardinalities and retains the semantics of each level through the representation-based cost. We construct three OT objectives for the sample–feature, feature–embedding unit, and sample–embedding-unit relations. Together, they provide direct coordination signals between adjacent and non-adjacent levels, encouraging each coarse-grained decision to be supported by compatible fine-grained decisions. During configuration learning, the sample weights, feature gates, embedding-unit masks, and transport costs are optimized together with the prediction objective. The learned configuration is then used to retrain the backbone with the selected input information. Since OptIDC operates on the input side, it leaves the backbone’s interaction architecture unchanged and can be used with diferent CTR models. We evaluate the framework on four public datasets and five representative backbones. Across the evaluated settings, joint configuration improves over the corresponding single-level alternatives, and OT coordination provides further gains over joint configuration alone. Ablation, eficiency, and mask analyses further examine the source of these gains.

Our main contributions are summarized as follows:

• We connect sample, feature, and embedding-unit selection through their common masking form and formulate the input-side dimensional configuration as a coupled problem over the three input levels.

• We develop OptIDC with a selection-weighted coordination objective for the sample–feature, feature–embedding-unit, and sample–embedding-unit relations, combining coarse selection mass, fine-level allocation, and representation compatibility through a finite-measure OT formulation.

• We evaluate OptIDC on four datasets and five CTR prediction backbones. The experiments compare it with single-level alternatives and disentangle the efects of joint configuration and OT coordination through ablation, eficiency, and selection consistency analyses.

## 2 Related Work

In this section, we briefly review some relevant works on two research topics, including deep CTR prediction and automated selection in CTR prediction.

Deep CTR Prediction. Deep learning has become a widely adopted approach to CTR prediction due to its strong capacity for modeling complex interactions among user, item, and contextual features. Early models, such as factorization machines, model pairwise interactions among sparse input features [22], while deep architectures such as Wide&Deep and DeepFM further capture higher-order interactions [2, 7]. Subsequent advances further improve modeling capacity through explicit feature-interaction networks, attention mechanisms, and multi-task learning [20, 23, 24, 27]. Despite their success, these architectural approaches typically treat all inputside information, such as samples, features, and embedding units, equally, without selecting or weighting informative signals during modeling. Thus, redundant or uninformative inputs may still limit eficiency and predictive performance when handling large-scale and high-dimensional data. In contrast, OptIDC complements these architectural advances by jointly configuring information at the sample, feature, and embedding-unit levels for CTR prediction.

Automated Selection in CTR Prediction. Introducing automated selection mechanisms into CTR prediction has attracted increasing attention in recent years as an efective way to improve prediction performance [33]. Existing studies mainly investigate information selection in three input-side dimensions. For feature inputs, prior work focuses on identifying informative feature subsets through field-level or value-level selection to reduce redundancy [10, 17, 28]. For embedding representations, embedding-unit or -dimension selection methods further refine representations by retaining important embedding elements and reducing unnecessary parameters [15, 16, 32]. OptEmbed [19] combines feature selection with embedding-dimension search. For training instances, sample selection methods identify high-value or reliable samples for model training [5, 11, 25]. Beyond input-side information, automated selection has also been explored for feature interaction [12, 18], loss functions [31], and hyperparameters [1]. Unlike these methods, which usually optimize one selection space at a time, our work extends input-side selection to a unified setting that jointly considers the sample, feature, and embedding-unit levels.

## 3 Preliminaries

## 3.1 Motivating Experiments

We first examine whether the three input levels benefit from joint configuration. Using Avazu with the DNN as the backbone, we compare sample selection (DS), feature selection (FS), and embeddingunit selection (ES) with Input-side Dimensional Configuration (IDC), which learns all three selectors under the CTR prediction objective without explicit cross-level coordination. As shown in Fig. 2a, the single-level variants improve AUC by about 0.001 over the backbone, whereas IDC yields a gain of more than 0.002. This result suggests that the three levels provide complementary information when configured together.

Joint learning, however, leaves the prediction loss as the only signal shared by selectors of diferent granularities. We therefore examine five IDC cases with diferent cross-level selection patterns. After mapping their binary decisions to a common space (see the Appendix A for the mapping process), and inspired by the Jaccard similarity, we measure the overlap among the three masks using:

$$
\mathrm { S C R } ( \mathbf { M } ^ { \mathrm { s } } , \mathbf { M } ^ { \mathrm { f } } , \mathbf { M } ^ { \mathrm { e } } ) = \frac { \left\| \operatorname* { m i n } ( \mathbf { M } ^ { \mathrm { s } } , \mathbf { M } ^ { \mathrm { f } } , \mathbf { M } ^ { \mathrm { e } } ) \right\| _ { 1 } } { \left\| \operatorname* { m a x } ( \mathbf { M } ^ { \mathrm { s } } , \mathbf { M } ^ { \mathrm { f } } , \mathbf { M } ^ { \mathrm { e } } ) \right\| _ { 1 } } ,\tag{1}
$$

where $\left. \cdot \right. _ { 1 }$ denotes the $l _ { 1 }$ norm, max(·) and min(·) represent the element-wise maximum and minimum operations, respectively. The SCR metric remains bounded within [0,1], where a larger SCR indicates greater agreement in both support and mapped activation magnitude. Fig. 2b shows that SCR rises by about 0.30 across these cases, accompanied by an AUC gain of nearly 0.008. Although this association is not causal evidence, it motivates explicit coordination rather than relying on a shared prediction loss alone.

![](images/78fdacb5cd087eb846ebcc102fd8c27e6cb784d2c36fb174b3c605a30b27735c.jpg)

![](images/0ea091c0676479776f12bfaa1299fae9c5ae4249d16f0edfcc18ba7c31ef050d.jpg)  
(a) The performance of difer- (b) The correlation between selecent selection methods. tion consistency and performance.  
Figure 2: (a) A comparison of the performance of diferent selection methods on the Avazu dataset. (b) The association between selection consistency and model performance on the Avazu dataset.

## 3.2 Problem Formulation

Let $\mathcal { D } = \{ ( \mathbf { x } _ { i } , y _ { i } ) \} _ { i = 1 } ^ { N }$ be a CTR prediction dataset, where $\mathbf { x } _ { i }$ contains $F$ feature fields and $y _ { i } \in \{ 0 , 1 \}$ is the target, where $y _ { i } = 1$ denotes click. An embedding layer maps $\mathbf { x } _ { i }$ to $\mathbf { H } _ { i } \mathbf { \bar { \Psi } } = [ \mathbf { h } _ { i 1 } , \dots , \mathbf { \bar { h } } _ { i F } ] ^ { \top } \in \mathbb { R } ^ { F \times d }$ where � is the embedding dimension. Then, $\mathbf { H } _ { i }$ is consumed by a backbone model $f _ { \theta } .$ Based on the definition above, we associate each sample with a sample weight $s _ { i } \in [ 0 , 1 ] ,$ , a feature gate $\mathbf { g } _ { i } \in [ 0 , 1 ] ^ { F }$ and an embedding-unit mask $\mathbf { M } _ { i } \in [ 0 , 1 ] ^ { F \times d }$ . Their selection scores are continuous during configuration learning and can be discretized to obtain the final configuration.

Although these have diferent shapes, they perform the same operation at diferent resolutions. The feature and unit decisions determine what enters the backbone model, while the sample decision determines how strongly that input contributes to learning. The two fine-grained masks first produce the configured embedding, represented as follows:

$$
\widetilde { \mathbf { H } } _ { i } = \mathbf { H } _ { i } \odot ( \mathbf { g } _ { i } \mathbf { 1 } _ { d } ^ { \top } ) \odot \mathbf { M } _ { i } ,\tag{2}
$$

Here $\odot$ denotes element-wise multiplication, and $\mathbf { 1 } _ { d } \in \{ 1 \} ^ { d }$ is an all-one vector. The sample weight then weights the loss evaluated on $\widetilde { \mathbf { H } } _ { i }$ from Eq. (2), which can be formalized as:

$$
\mathcal { L } _ { \mathrm { c t r } } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } s _ { i } \cdot \ell \Big ( y _ { i } , f _ { \theta } ( \widetilde { \mathbf { H } } _ { i } ) \Big ) ,\tag{3}
$$

where $\ell ( \cdot )$ is the cross-entropy loss function. Optimizing Eq. (3) with separate selection regularizers yields IDC. The selectors share the prediction loss, but the objective contains no term that directly relates a coarse decision to the finer-grained decisions through which it takes efect.

Thus, we consider the sample–feature, feature–embedding-unit, and sample–embedding-unit relations, abbreviated as sf, fe, and se. For sample �, let $\mathcal { W } _ { \mathrm { s f } } ( i ) , \mathcal { W } _ { \mathrm { f e } } ( i )$ , and $\mathcal { W } _ { \mathrm { s e } } ( i )$ denote diferentiable costs that measure whether a coarse decision is supported by compatible fine-grained components. Their dependence on the selection outputs and learned representations is suppressed here for readability. The coordination target is defined as follows:

$$
\mathcal { L } _ { \mathrm { c o o r d } } = \frac { \lambda _ { \mathrm { c o } } } { N } \sum _ { i = 1 } ^ { N } \left[ \mathcal { W } _ { \mathrm { s f } } ( i ) + \mathcal { W } _ { \mathrm { f e } } ( i ) + \mathcal { W } _ { \mathrm { s e } } ( i ) \right] ,\tag{4}
$$

where the nonnegative $\lambda _ { \mathrm { c o } }$ coeficient controls the relations. Finally, the optimal input-side dimensional configuration problem combines CTR prediction, level-specific selection constraints, and cross-level coordination as follows:

$$
\operatorname* { m i n } _ { \theta , \Theta , \varphi } \mathcal { L } _ { \mathrm { c t r } } + \Omega _ { \mathrm { s e l } } ( s , \mathbf { G } , \mathbf { M } ) + \mathcal { L } _ { \mathrm { c o o r d } } ,\tag{5}
$$

where Θ is the embedding parameters, � is the backbone model parameters, $\begin{array} { r } { \pmb { \mathscr { s } } = ( \mathscr { s } _ { i } ) _ { i = 1 } ^ { N } , \tilde { \pmb { \mathrm { G } } } = ( \mathbf { \mathscr { g } } _ { i } ) _ { i = 1 } ^ { N } , \mathbf { M } = ( \mathbf { M } _ { i } ) _ { i = 1 } ^ { N } } \end{array}$ are the outputs induced by �, and $\Omega _ { \mathrm { s e l } }$ controls the desired amount of selection at each level. Eq. (5) states the desired behavior but leaves two choices open: how to produce the three selectors along one information path, and how to instantiate the three costs without forcing heterogeneous decisions onto an artificial common support. The next section addresses these two choices.

## 4 The OptIDC Framework

## 4.1 Framework Overview

OptIDC is organized around two questions raised by unified input configuration: how to learn sample, feature, and embedding-unit decisions within the same model, and how to keep these decisions from evolving independently. For the first question, we arrange the three selectors along the path by which an input reaches the CTR prediction model. Feature and embedding-unit decisions determine what the backbone observes, and the sample decision evaluates how strongly that configured input should influence learning. For the second, we connect every coarse-to-fine pair through a selection weighted optimal transport (OT) cost that respects the native granularity of both selectors. The framework follows a search-and-retrain workflow. Continuous gates make configuration choices diferentiable during search, allowing the selectors and CTR prediction backbone to be learned together. Once a configuration is identified, the gates are converted into hard decisions, and the backbone is retrained with this configuration fixed. The overall architecture of OptIDC is illustrated in Fig. 3. The following subsections describe the unified input configuration learning, its cross-level coordination, and the two-stage optimization.

## 4.2 Unified Input Configuration Learning

The formulation in Eq.(2) and Eq.(3) tells us how the three decisions act, but not how they are produced. OptIDC learns them in the same order in which they afect the input. Feature and embedding-unit selectors first configure the embedding tensor. The sample selector then scores the resulting tensor, so its decision is based on the information that the backbone actually receives.

For sample �, let $v _ { i j } = x _ { i j }$ be the active feature value in field $j ,$ and write $[ n ] = \{ 1 , . . . , n \}$ . Let V be the disjoint union of the field vocabularies. A lookup table $\mathbf { A } ^ { ( \mathrm { f } ) } \in \mathbb { R } ^ { | \mathcal { V } | }$ provides one feature logit for each value, where $n = \vert \mathcal { V } \vert$ is the total number of feature values. To score individual embedding units, field � owns unit representations $\mathbf { U } _ { j } = \left[ \mathbf { u } _ { j 1 } , \ldots , \mathbf { u } _ { j d } \right] \in \mathbb { R } ^ { d \times d }$ , and $\mathbf { B } ^ { ( \mathrm { e } ) } \in \mathbb { R } ^ { | \mathcal { V } | \times d }$ provides a value-specific bias. Their two logits are expressed as:

$$
\begin{array} { r } { { a } _ { i j } ^ { \left( \mathrm { f } \right) } = { \mathbf { A } } ^ { \left( \mathrm { f } \right) } \left[ v _ { i j } \right] , \qquad { \mathbf { a } } _ { i j } ^ { \left( \mathrm { e } \right) } = { \mathbf { U } } _ { j } ^ { \top } \ : s \mathbf { g } ( \mathbf { h } _ { i j } ) + { \mathbf { B } } ^ { \left( \mathrm { e } \right) } \left[ v _ { i j } \right] , } \end{array}\tag{6}
$$

where the first equality retrieves a scalar score for the active feature value. Because this score controls its entire �-dimensional embedding, it answers a feature-level question: whether $v _ { i j }$ should contribute as a whole. The second equality asks the finer question of which units should contribute. Component-wise, its �-th logit is $a _ { i j k } ^ { \left( \mathrm { e } \right) } = \mathbf { u } _ { j k } ^ { \top } \mathrm { s g } ( \mathbf { h } _ { i j } ) + \mathbf { B } ^ { \left( \mathrm { e } \right) } \left[ v _ { i j } , k \right]$ . The inner product measures the compatibility between the current embedding and unit representation $\mathbf { u } _ { j k }$ . Since $\mathbf { U } _ { j }$ is shared by all values in field �, it captures a reusable field-level scoring pattern, while ${ \bf B } ^ { \left( \mathrm { e } \right) } \left[ v _ { i j } , k \right]$ provides a value-specific correction. The stop-gradient operator sg(·) prevents the unit-scoring branch from reshaping the embedding merely to make its selection easier. Then, following [14, 17], the logits are converted to diferentiable gates with a shared annealing schedule:

$$
\Gamma _ { t } ( a ) = \sigma \big ( \gamma ^ { t / T } a \big ) , \quad g _ { i j } = \Gamma _ { t } ( a _ { i j } ^ { ( \mathrm { f } ) } ) , \quad \mathbf { M } _ { i j : } = \Gamma _ { t } ( \mathbf { a } _ { i j } ^ { ( \mathrm { e } ) } ) ,\tag{7}
$$

where � is the sigmoid function, � is the current epoch, � is the total number of epochs, � controls the final temperature, $\Gamma _ { t } ( \cdot )$ acts

element-wise on vectors, and $\mathbf { M } _ { i j } \mathrm { : }$ is row � of $\mathbf { M } _ { i }$ . Subsequently, the tensor is configured using $\operatorname { E q . } ( 2 )$ and passed to the sample selector:

$$
s _ { i } = \sigma ( \mathcal { \Psi } ( \mathrm { v e c } ( s \mathrm { g } ( \widetilde { \mathbf { H } } _ { i } ) ) ) ) ,\tag{8}
$$

where vec(·) vectorizes its matrix argument. $\Psi ( \cdot ) : \mathbb { R } ^ { F d }  \mathbb { R }$ is a lightweight multilayer perceptron. Eq.(7) turns the two logits into decisions in the order they afect learning. The scalar $g _ { i j }$ gates the whole feature embedding, whereas the �-th entry $M _ { i j k }$ gates one unit within it. Together they form $\widetilde { \mathbf { H } } _ { i }$ . Ψ(·) then evaluates this already-configured tensor to produce $s _ { i } .$ Thus, the feature and embedding-unit selectors decide what the backbone observes, and the sample selector decides how strongly that resulting example contributes to the loss.

The search step $t \in [ 0 , T ]$ progresses over a horizon $T ,$ and $\gamma > 1$ gradually sharpens the feature and embedding-unit gates. During the early search stage, continuous values enable gradient propagation to competing configurations, whereas the logits gradually become hard decisions in the later stage for retraining. Eqs.(6)–(8) therefore instantiate $s _ { i } , \mathbf { g } _ { i } , \mathbf { M } _ { i }$ in the unified formulation. Note that all tables, unit representations, and sample scorers are shared across the dataset rather than learned independently for each sample.

## 4.3 Cross-Level Coordination via OT

The common information path couples the selectors indirectly through prediction, but it still does not say which fine-grained decisions support a coarse one. A sample can receive a large weight while placing most of its feature mass on poorly matched fields; an active feature can likewise retain incompatible units. However, comparing masks entry by entry is not meaningful here because a sample score $s _ { i } ,$ a feature weight vector ${ \bf g } _ { i } ,$ and a unit weight matrix $\mathbf { M } _ { i }$ live on diferent supports. We instead read each relation as a coarse-to-fine transport. A coarse selection score � determines the mass to be explained, and a normalized fine-level allocation � distributes that mass over representations $\{ \mathbf { z } _ { o } \}$ }. For ground cost �, the transport between the corresponding finite measures has the closed form, for $\alpha \ge 0 , \pi _ { \boldsymbol { v } } \ge 0 .$ , and $\begin{array} { r } { \sum _ { v } \pi _ { v } = 1 } \end{array}$ , formalized as:

$$
\mathrm { O T } _ { c } \left( \alpha \delta _ { \mathbf { z } } , \alpha \sum _ { v } \pi _ { v } \delta _ { \mathbf { z } _ { v } } \right) = \alpha \sum _ { v } \pi _ { v } c ( \mathbf { z } , \mathbf { z } _ { v } ) ,\tag{9}
$$

where $\delta _ { \mathbf { z } }$ is a Dirac measure. For given selection outputs, the single coarse support and fixed destination marginal uniquely determine $T _ { v } ^ { \star } = \alpha \pi _ { v }$ . The exact OT cost in Eq. (9) [3, 6] thus acts as a selection-weighted representation-compatibility penalty, with mass, allocation, and representations jointly learned during search.

For example, in the sample–feature relation, $s _ { i }$ supplies the coarse mass, the normalized feature gates distribute it, and the sample and feature representations define the transport cost. The other two relations use the same pattern at their own granularity. Fine-level scores determine a unit-mass destination profile, while the coarse score � alone determines how much mass is transported. For a positive vector q, we use the temperature-normalized selection weights, expressed as follows:

$$
\left[ N _ { \tau } ( \mathbf { q } ) \right] _ { r } = \frac { \exp ( \log ( q _ { r } + \varepsilon ) / \tau ) } { \sum _ { l } \exp ( \log ( q _ { l } + \varepsilon ) / \tau ) } ,\tag{10}
$$

$$
\begin{array} { r } { \pi _ { i } ^ { ( \mathrm { s f } ) } = N _ { \tau } ( \mathbf { g } _ { i } ) , \quad \pi _ { i j } ^ { ( \mathrm { f e } ) } = N _ { \tau } ( \mathbf { M } _ { i j \colon } ) , \quad \pi _ { i } ^ { ( \mathrm { s e } ) } = N _ { \tau } ( \mathrm { v e c } ( \mathbf { M } _ { i } ) ) , } \end{array}\tag{11}
$$

![](images/d28ae78b9860051371db3177a39ee1ec98f4c0508cbd411377fb43b7afc153ef.jpg)  
Figure 3: The architecture of the Optimal Input-side Dimensional Configuration (OptIDC) framework.

where tiny $\varepsilon > 0$ prevents log 0, and $\tau > 0$ controls concentration. We evaluate the normalization with log-sum-exp. In the limit $\varepsilon  0 ,$ a gate that vanishes relative to non-vanishing alternatives receives vanishing mass instead of being treated as an ordinary bounded logit. The indices � and � range over entries of q. We index the last allocation as $\pi _ { i j k } ^ { ( \mathrm { s e } ) }$ after reshaping it to $F \times d .$

Then, semantic compatibility is measured in a shared representation space of dimension �. We use the pooled sample representation and cosine ground cost, formulated as follows:

$$
\overline { { \mathbf { h } } } _ { i } = \frac { 1 } { F } \sum _ { j = 1 } ^ { F } \mathbf { h } _ { i j } , \qquad c ( \mathbf { a } , \mathbf { b } ) = 1 - \frac { \mathbf { a } ^ { \top } \mathbf { b } } { ( \| \mathbf { a } \| _ { 2 } + \varepsilon ) ( \| \mathbf { b } \| _ { 2 } + \varepsilon ) } ,\tag{12}
$$

where tiny $\varepsilon > 0$ also keeps the cosine cost defined for near-zero representations, and $\| \cdot \| _ { 2 }$ is the $l _ { 2 }$ norm. Recall that $\mathbf { h } _ { i j }$ represents feature � and $\mathbf { u } _ { j k } ,$ , the �-th column of $\mathbf { U } _ { j } ,$ represents its �-th embedding unit. We use the unmasked embeddings here as stable semantic anchors while the configuration changes. The stop-gradient in Eq.(6) applies only to the unit-logit scoring path, while the representation costs remain diferentiable. Instantiating $\operatorname { E q . } ( 9 )$ gives the three costs needed as:

$$
\begin{array} { l } { { \displaystyle \mathcal { W } _ { \mathrm { s f } } ( i ) = s _ { i } \sum _ { j = 1 } ^ { F } \pi _ { i j } ^ { ( \mathrm { s f } ) } c ( \overline { { \mathbf { h } } } _ { i } , \mathbf { h } _ { i j } ) } , } \\ { ~ } \\ { { \displaystyle \mathcal { W } _ { \mathrm { f e } } ( i ) = \frac { 1 } { F } \sum _ { j = 1 } ^ { F } g _ { i j } \sum _ { k = 1 } ^ { d } \pi _ { i j k } ^ { ( \mathrm { f e } ) } c ( \mathbf { h } _ { i j } , \mathbf { u } _ { j k } ) } , } \\ { { \displaystyle \qquad \cdot \mathcal { W } _ { \mathrm { s e } } ( i ) = s _ { i } \sum _ { j = 1 } ^ { F } \sum _ { k = 1 } ^ { d } \pi _ { i j k } ^ { ( \mathrm { s e } ) } c ( \overline { { \mathbf { h } } } _ { i } , \mathbf { u } _ { j k } ) } . } \end{array}\tag{13}
$$

The first line transports sample mass to features, the second transports each feature’s mass to its units, and the third provides a direct sample-to-unit path. Substituting these terms into Eq.(4) gives an explicit signal to all three relations. Each term penalizes a selected coarse component when its selector-induced fine allocation has high representation cost. This objective coordinates relative semantic allocation, while the CTR prediction loss and the selection constraints determine the absolute gate magnitudes.

## 4.4 Optimization

Next, we will introduce how OptIDC is optimized, which can be divided into two stages: search and retrain.

Search Stage. Configuration search must avoid two trivial outcomes: retaining every feature and unit, or suppressing nearly all samples. We therefore use sparsity on the fine-grained gates and control sample selection with a ratio constraint [13]. Sparsity is applied directly to selector outputs rather than to their raw lookup parameters. In practice, optimization is performed using mini-batches: sample-indexed sums and averages are evaluated over the current mini-batch, while lookup-table regularizers are computed over the full tables. For clarity, we present the regularization structure using dataset-level notation:

$$
\begin{array} { l } { { \displaystyle \Omega _ { \mathrm { s p } } = \lambda _ { \mathrm { f } } \| \Gamma _ { t } ( { \bf A } ^ { ( f ) } ) \| _ { 1 } + \lambda _ { \mathrm { e } } \big ( \displaystyle \sum _ { i = 1 } ^ { N } \| { \bf M } _ { i } \| _ { 1 } + \| \Gamma _ { t } ( { \bf B } ^ { ( e ) } ) \| _ { 1 } \big ) } , }  \\ { { \displaystyle \Omega _ { s } = \lambda _ { \mathrm { r a t } } \left( \frac { 1 } { N } \displaystyle \sum _ { i = 1 } ^ { N } s _ { i } - \rho \right) ^ { 2 } + \frac { \lambda _ { \mathrm { e n t } } } { N } \displaystyle \sum _ { i = 1 } ^ { N } h _ { \mathrm { b i n } } ( s _ { i } ) } , } \end{array}\tag{14}
$$

where $\lambda _ { \mathrm { f } }$ and $\lambda _ { \mathrm { e } }$ are sparsity regularization weights for feature and embedding-unit masks, $\lambda _ { \mathrm { { r a t } } }$ and $\lambda _ { \mathrm { e n t } }$ are ratio and entropy regularization weights for sample masks, $\rho$ is an expected sampleretention ratio (e.g., 0.98) and $h _ { \mathrm { b i n } } ( z ) = - z \log z - ( 1 - z ) \log ( 1 - z )$ is binary entropy to encourage binarization of the sample masks. Its arguments are numerically clamped in implementation. Finally, combining Eqs. (3), (4), and (14) gives the search stage objective:

$$
\operatorname* { m i n } _ { \theta , \Theta , \varphi } \mathcal { L } _ { \mathrm { s e a r c h } } = \mathcal { L } _ { \mathrm { c t r } } + \mathcal { L } _ { \mathrm { c o o r d } } + \Omega _ { \mathrm { s p } } + \Omega _ { \mathrm { s } } ,\tag{15}
$$

where $\Omega _ { \mathrm { s p } } + \Omega _ { \mathrm { s } }$ instantiates $\Omega _ { \mathrm { s e l } }$ , and $\operatorname { E q . } ( 1 3 )$ instantiates the three components of $\mathcal { L } _ { \mathrm { c o o r d } }$ . Setting the coordination coeficient $\lambda _ { \mathrm { c o } }$ to zero recovers IDC. The selector parameters � collect $\mathbf { A } ^ { ( \mathrm { f } ) } , \mathbf { B } ^ { ( \mathrm { e } ) }$ $\{ \mathbf { U } _ { j } \} _ { j = 1 } ^ { F }$ , and the parameters of $\Psi ( \cdot )$ . Note that during training, parameters are actually updated using mini-batches of samples.

Retrain Stage. Search identifies a configuration, and final fitting estimates the model on that fixed input space. Let $s _ { i } ^ { \star } , \mathbf { g } _ { i } ^ { \star } , \mathbf { M } _ { i } ^ { \star }$ denote the selector outputs obtained by solving Eq.(15). Sample, feature, and unit gates use the natural midpoint of their sigmoid range:

$$
\widehat { s _ { i } } = \mathbb { I } ( s _ { i } ^ { \star } > \frac { 1 } { 2 } ) , \quad \widehat { \mathbf { g } } _ { i } = \mathbb { I } ( \mathbf { g } _ { i } ^ { \star } > \frac { 1 } { 2 } ) , \quad \widehat { \mathbf { M } } _ { i } = \mathbb { I } ( \mathbf { M } _ { i } ^ { \star } > \frac { 1 } { 2 } ) ,\tag{16}
$$

where I(·) is the element-wise indicator function. Because the feature and unit scores are value-indexed, their hard decisions are materialized as lookup tables, while sample decisions define the retained training set $\widehat { \mathcal { D } } = \{ ( \mathbf { x } _ { i } , y _ { i } ) \ | \ \widehat { s _ { i } } = 1 , ( \mathbf { x } _ { i } , y _ { i } ) \in \mathcal { D } \}$ . Then, we remove the auxiliary selectors and coordination loss, initialize the model parameters (Θ, �) from the checkpoint saved at epoch $T _ { c }$ of the search stage, and retrain with all hard masks fixed:

$$
\operatorname* { m i n } _ { \theta , \Theta } \mathcal { L } _ { \mathrm { r e t r a i n } } = \frac { 1 } { | \widehat { \mathcal { D } } | } \sum _ { i \in \widehat { \mathcal { D } } } \ell \Big ( y _ { i } , f _ { \theta } \big ( \widehat { \ H } _ { i } \big ) \Big ) \ , \ \widehat { \ H } _ { i } = \mathbf { H } _ { i } \odot \big ( \widehat { \mathbf { g } } _ { i } \mathbf { 1 } _ { d } ^ { \top } \big ) \odot \widehat { \mathbf { M } } _ { i } .\tag{17}
$$

Therefore, search decides what information to keep, and retraining estimates the embedding and backbone parameters for that choice. At inference time, a new sample retrieves the feature and unit masks of its active values. The sample selector is used only to construct the retraining set and introduces no serving-time overhead.

## 5 Experiments

In this section, we conduct experiments to answer the following five key questions.

• RQ1: How does OptIDC perform compared to the baselines?

• RQ2: What do unified configuration and cross-level coordination each contribute?

• RQ3: How does the eficiency ofOptIDC compare to the baseline?

• RQ4: How do the cost construction and coordination methods afect OptIDC?

• RQ5: How efectively does OptIDC mitigate selection conflicts?

## 5.1 Experiment Setup

Datasets. To comprehensively evaluate the efectiveness ofOptIDC, we follow prior work [9] and conduct experiments on four public datasets: Avazu<sup>1</sup>, Criteo<sup>2</sup>, MovieLens-1M<sup>3</sup>, and AliCCP<sup>4</sup>. Avazu and Criteo are widely used in CTR for studying feature and dimension selection [16, 17]. MovieLens-1M is a representative benchmark dataset commonly adopted to validate diferent selection methods.

In addition, to assess performance in settings closer to real-world prediction systems, we include the AliCCP dataset, which is collected from the Taobao e-commerce platform and is widely used for CTR estimation. Detailed preprocessing procedures and dataset statistics are provided in Appendix B.1.

Metrics. Following prior work [17], we adopt two standard evaluation metrics for CTR prediction: area under the ROC curve (AUC) and cross-entropy loss (LogLoss). Small AUC improvements can be practically meaningful in CTR prediction [7, 17, 29].

Baseline Methods and Backbone Methods. We select representative methods from prior studies on input-side information selection methods in deep CTR prediction. Specifically, we compare OptIDC with feature selection baselines including AutoField [28], LPFS [8], and OptFS [17]; embedding configuration baselines AutoDim [32], OptEmbed [19], and PEP [16]; and the sample selection baseline AutoDenoise [11]. In addition, we compare OptIDC with single-level variants of our configuration path: feature selection (FS), embedding-unit selection (ES), and sample selection (DS). We also report IDC, which activates all three levels but removes crosslevel coordination. Note that the feature selection (FS) in OptIDC is identical to that in OptFS. To evaluate generalization, each method is combined with five main backbone models: DNN, DeepFM [7], DCN [27], AutoInt [23], and FinalMLP [21]. We also include a base model without any selection method, denoted as ‘Backbone’. Implementation details are provided in Appendix B.2.

## 5.2 RQ1: Performance Comparison

This subsection evaluates OptIDC against multiple baseline methods spanning three types of input-side information selection methods: sample selection, feature selection, and embedding configuration. In Table 1, we report the overall performance of OptIDC and other baseline methods across two benchmark datasets, evaluated with five diferent backbone models. The results of the other two datasets are shown in Table 5 in Appendix C.1. Based on the experimental results, we make the following observations: 1) Several strong single-level methods (e.g., OptFS, LPFS, ES, and DS) generally outperform the Backbone, confirming that input-side configuration is useful in its own right. The improvement is not uniform, however: a method that works well for one backbone or dataset may ofer little benefit for another. This variation is consistent with our premise that samples, feature values, and embedding units expose diferent kinds of redundancy. 2) IDC is generally stronger than its single-level counterparts, showing that the three decisions provide complementary information even when they are connected only by the prediction objective. 3) Our proposed OptIDC achieves significant AUC improvements over IDC on all datasets, with a clearer margin on Avazu. This result highlights the importance of introducing the cross-level coordination mechanism, thereby learning a more efective input-side configuration.

## 5.3 RQ2: Ablation Study

Next, we analyze the contributions of joint configuration and OT coordination separately. Specifically, we remove diferent configuration paths or coordination terms from OptIDC, resulting in ten ablation variants. These include joint sample and feature selection (‘DSFS’), joint sample and embedding-unit selection (‘DSES’), and joint feature and embedding-unit selection (‘FSES’), together with their counterparts using the corresponding OT relation (‘DSFS+OT’, ‘DSES+OT’, and ‘FSES+OT’); joint three-level selection without coordination (‘IDC’); and three variants that individually remove W , $\mathcal { W } _ { \mathrm { s e } } , \ o \mathrm { r } \ \mathcal { W } _ { \mathrm { f e } }$ while keeping all three selectors active. The singlelevel results for FS, ES, and DS are reported in Table 1, where FS corresponds to OptFS. We additionally include AOP, which jointly trains AutoDenoise, OptFS, and PEP for three-level selection while preserving their key network modules. AOP’s regularization weights are tuned on the validation set under joint training, while other hyperparameters retain their standalone settings. The corresponding results are summarized in Table 3, and additional results on the Criteo dataset are shown in Table 6 in Appendix C.2. Combining two levels generally improves upon the corresponding single-level variants, while IDC provides further gains in some settings. Adding the corresponding OT relation improves AUC and improves or maintains LogLoss for each two-level model across the five backbones. OptIDC further improves both metrics over IDC, supporting the benefits of explicit coordination within joint threelevel configuration. Removing any individual OT term from OptIDC lowers AUC and increases LogLoss across all five backbones on Avazu. These results support the contribution of each coordination relation. OptIDC also outperforms AOP, suggesting that its benefits extend beyond directly combining existing selection methods.

Table 1: Results on Avazu and Criteo (mean ± standard deviation over five runs with diferent random seeds), where the best and second best results are marked in bold and underlined, respectively. Note that <sup>∗</sup> indicates a significance level of $\textstyle p \leq 0 . 0 5$ based on a two-sample t-test between our method and the best baseline.
<table><tr><td rowspan=2 colspan=1>Dataset</td><td rowspan=2 colspan=1>Model</td><td rowspan=1 colspan=2>DNN</td><td rowspan=1 colspan=2>DeepFM</td><td rowspan=1 colspan=2>DCN</td><td rowspan=1 colspan=2>AutoInt</td><td rowspan=1 colspan=2>FinalMLP</td></tr><tr><td rowspan=1 colspan=4>AUC↑    LogLoss↓    AUC↑   LogLoss↓</td><td rowspan=1 colspan=2>AUC↑   LogLoss↓</td><td rowspan=1 colspan=2>AUC↑    LogLoss↓</td><td rowspan=1 colspan=2>AUC↑    LogLoss↓</td></tr><tr><td rowspan=3 colspan=1></td><td rowspan=1 colspan=1>Backbone</td><td rowspan=1 colspan=2>.7893±.0001 .3757±.0001</td><td rowspan=1 colspan=2>.7896±.0004 .3759±.0004</td><td rowspan=1 colspan=2>.7893±.0004 .3759±.0004</td><td rowspan=1 colspan=2>.7895±.0003.3755±.0003</td><td rowspan=1 colspan=2>.7898±.0004 .3740±.0003</td></tr><tr><td rowspan=1 colspan=1>AutoField</td><td rowspan=1 colspan=2>.7840±.0007 .3812±.0007</td><td rowspan=1 colspan=1>.7842±.0003</td><td rowspan=1 colspan=1>.3809±.0008</td><td rowspan=1 colspan=1>.7838±.0006</td><td rowspan=1 colspan=1>.3806±.0004</td><td rowspan=1 colspan=1>.7838±.0002</td><td rowspan=1 colspan=1>.3815±.0005</td><td rowspan=1 colspan=1>.7835±.0006</td><td rowspan=1 colspan=1>.3812±.0015</td></tr><tr><td rowspan=1 colspan=1>LPFS</td><td rowspan=1 colspan=2>.7888±.0007 .3762±.0009</td><td rowspan=1 colspan=1>.7896±.0003</td><td rowspan=1 colspan=1>.3758±.0003</td><td rowspan=1 colspan=1>.7894±.0003</td><td rowspan=1 colspan=1>.3758±.0003</td><td rowspan=1 colspan=1>.7894±.0004</td><td rowspan=1 colspan=1>.3756±.0004</td><td rowspan=1 colspan=1>.7898±.0002</td><td rowspan=1 colspan=1>.3739±.0001</td></tr><tr><td rowspan=6 colspan=1>Avazu</td><td rowspan=1 colspan=1>OptEmbed</td><td rowspan=1 colspan=2>.7884±.0008 .3766±.0009</td><td rowspan=1 colspan=1>.7895±.0004</td><td rowspan=1 colspan=1>.3756±.0003</td><td rowspan=1 colspan=1>.7885±.0007</td><td rowspan=1 colspan=1>.3765±.0007</td><td rowspan=1 colspan=1>.7884±.0009</td><td rowspan=1 colspan=1>.3768±.0010</td><td rowspan=1 colspan=1>.7885±.0007</td><td rowspan=1 colspan=1>.3757±.0011</td></tr><tr><td rowspan=1 colspan=1>OptFS</td><td rowspan=1 colspan=2>.7907±.0002.3736±.0001</td><td rowspan=1 colspan=1>.7915±.0002</td><td rowspan=1 colspan=1>.3731±.0001</td><td rowspan=1 colspan=1>.7908±.0001</td><td rowspan=1 colspan=1>.3735±.0001</td><td rowspan=1 colspan=1>.7904±.0003</td><td rowspan=1 colspan=1>.3738±.0001</td><td rowspan=1 colspan=1>.7916±.0005</td><td rowspan=1 colspan=1>.3730±.0003</td></tr><tr><td rowspan=1 colspan=1>PEP</td><td rowspan=1 colspan=2>.7892±.0005 .3754±.0008</td><td rowspan=1 colspan=1>.7895±.0005</td><td rowspan=1 colspan=1>.3755±.0006</td><td rowspan=1 colspan=1>.7891±.0001</td><td rowspan=1 colspan=1>.3745±.0001</td><td rowspan=1 colspan=1>.7892±.0006</td><td rowspan=1 colspan=1>.3755±.0006</td><td rowspan=1 colspan=1>.7894±.0002</td><td rowspan=1 colspan=1>.3743±.0003</td></tr><tr><td rowspan=1 colspan=1>AutoDim</td><td rowspan=1 colspan=2>.7898±.0005 .3744±.0005</td><td rowspan=1 colspan=1>.7899±.0006</td><td rowspan=1 colspan=1>.3744±.0005</td><td rowspan=1 colspan=1>.7892±.0006</td><td rowspan=1 colspan=1>.3749±.0003</td><td rowspan=1 colspan=1>.7899±.0005</td><td rowspan=1 colspan=1>.3746±.0006</td><td rowspan=1 colspan=1>.7909±.0005</td><td rowspan=1 colspan=1>.3736±.0004</td></tr><tr><td rowspan=1 colspan=1>ES</td><td rowspan=1 colspan=1>.7908±.0001</td><td rowspan=1 colspan=1>.3735±.0001</td><td rowspan=1 colspan=1>.7915±.0002</td><td rowspan=1 colspan=1>.3731±.0001</td><td rowspan=1 colspan=1>.7909±.0002</td><td rowspan=1 colspan=1>.3734±.0001</td><td rowspan=1 colspan=1>.7907±.0001</td><td rowspan=1 colspan=1>.3736±.0001</td><td rowspan=1 colspan=1>.7915±.0002</td><td rowspan=1 colspan=1>.3730±.0001</td></tr><tr><td rowspan=1 colspan=1>AutoDenoise</td><td rowspan=1 colspan=1>.7834±.0002</td><td rowspan=1 colspan=1>.3820±.0002</td><td rowspan=1 colspan=1>.7835±.0003</td><td rowspan=1 colspan=1>.3819±.0002</td><td rowspan=1 colspan=1>.7832±.0001</td><td rowspan=1 colspan=1>.3819±.0004</td><td rowspan=1 colspan=1>.7831±.0004</td><td rowspan=1 colspan=1>.3820±.0005</td><td rowspan=1 colspan=1>.7836±.0003</td><td rowspan=1 colspan=1>.3820±.0005</td></tr><tr><td rowspan=3 colspan=1></td><td rowspan=1 colspan=1>DS</td><td rowspan=1 colspan=1>.7899±.0004</td><td rowspan=1 colspan=1>.3745±.0006</td><td rowspan=1 colspan=1>.7907±.0003</td><td rowspan=1 colspan=1>.3736±.0006</td><td rowspan=1 colspan=1>.7899±.0006</td><td rowspan=1 colspan=1>.3744±.0005</td><td rowspan=1 colspan=1>.7898±.0002</td><td rowspan=1 colspan=1>.3745±.0003</td><td rowspan=1 colspan=1>.7919±.0002</td><td rowspan=1 colspan=1>.3731±.0008</td></tr><tr><td rowspan=1 colspan=1>IDC</td><td rowspan=1 colspan=1>.7915±.0003</td><td rowspan=1 colspan=1>.3732±.0003</td><td rowspan=1 colspan=1>.7918±.0003</td><td rowspan=1 colspan=1>.3729±.0001</td><td rowspan=1 colspan=1>.7914±.0002</td><td rowspan=1 colspan=1>.3734±.0003</td><td rowspan=1 colspan=1>.7915±.0002</td><td rowspan=1 colspan=1>.3731±.0003</td><td rowspan=1 colspan=1>.7925±.0001</td><td rowspan=1 colspan=1>.3728±.0002</td></tr><tr><td rowspan=1 colspan=1>OptIDC</td><td rowspan=1 colspan=2>.7930*±.0001.3722*±.0001</td><td rowspan=1 colspan=1>.7933*±.0001</td><td rowspan=1 colspan=1>.3721*±.0001</td><td rowspan=1 colspan=1>.7931*±.0002</td><td rowspan=1 colspan=1>.3721*±.0002</td><td rowspan=1 colspan=1>.7934*±.0002</td><td rowspan=1 colspan=1>.3721*±.0001</td><td rowspan=1 colspan=1>.7941*±.0002</td><td rowspan=1 colspan=1>.3719*±.0003</td></tr><tr><td rowspan=12 colspan=1>CTiteo</td><td rowspan=1 colspan=1>Backbone</td><td rowspan=1 colspan=2>.8097±.0001 .4419±.0001</td><td rowspan=1 colspan=1>.8088±.0003</td><td rowspan=1 colspan=1>.4435±.0001</td><td rowspan=1 colspan=1>.8094±.0003</td><td rowspan=1 colspan=1>.4425±.0001</td><td rowspan=1 colspan=1>.8093±.0002</td><td rowspan=1 colspan=1>.4424±.0002</td><td rowspan=1 colspan=1>.8097±.0003</td><td rowspan=1 colspan=1>.4420±.0002</td></tr><tr><td rowspan=1 colspan=1>AutoField</td><td rowspan=1 colspan=1>.8085±.0004</td><td rowspan=1 colspan=1>.4433±.0004</td><td rowspan=1 colspan=1>.8083±.0004</td><td rowspan=1 colspan=1>.4434±.0004</td><td rowspan=1 colspan=1>.8084±.0005</td><td rowspan=1 colspan=1>.4433±.0005</td><td rowspan=1 colspan=1>.8086±.0002</td><td rowspan=1 colspan=1>.4432±.0002</td><td rowspan=1 colspan=1>.8091±.0003</td><td rowspan=1 colspan=1>.4425±.0004</td></tr><tr><td rowspan=1 colspan=1>LPFS</td><td rowspan=1 colspan=1>.8107±.0001</td><td rowspan=1 colspan=1>.4410±.0001</td><td rowspan=1 colspan=1>.8104±.0001</td><td rowspan=1 colspan=1>.4413±.0001</td><td rowspan=1 colspan=1>.8098±.0002</td><td rowspan=1 colspan=1>.4418±.0002</td><td rowspan=1 colspan=1>.8097±.0002</td><td rowspan=1 colspan=1>.4418±.0003</td><td rowspan=1 colspan=1>.8102±.0001</td><td rowspan=1 colspan=1>.4414±.0001</td></tr><tr><td rowspan=1 colspan=1>OptEmbed</td><td rowspan=1 colspan=2>.8105±.0001 .4412±.0002</td><td rowspan=1 colspan=1>.8108±.0001</td><td rowspan=1 colspan=1>.4410±.0001</td><td rowspan=1 colspan=1>.8104±.0001</td><td rowspan=1 colspan=1>.4413±.0001</td><td rowspan=1 colspan=1>.8104±.0002</td><td rowspan=1 colspan=1>.4413±.0002</td><td rowspan=1 colspan=1>.8099±.0001</td><td rowspan=1 colspan=1>.4415±.0002</td></tr><tr><td rowspan=1 colspan=1>OptFS</td><td rowspan=1 colspan=2>.8107±.0001 .4410±.0001</td><td rowspan=1 colspan=1>.8105±.0001</td><td rowspan=1 colspan=1>.4411±.0001</td><td rowspan=1 colspan=1>.8108±.0001</td><td rowspan=1 colspan=1>.4409±.0001</td><td rowspan=1 colspan=1>.8107±.0001</td><td rowspan=1 colspan=1>.4410±.0001</td><td rowspan=1 colspan=1>.8108±.0001</td><td rowspan=1 colspan=1>.4411±.0001</td></tr><tr><td rowspan=1 colspan=1>PEP</td><td rowspan=1 colspan=2>.8105±.0001 .4410±.0001</td><td rowspan=1 colspan=1>.8105±.0001</td><td rowspan=1 colspan=1>.4411±.0001</td><td rowspan=1 colspan=1>.8106±.0001</td><td rowspan=1 colspan=1>.4410±.0001</td><td rowspan=1 colspan=1>.8105±.0000</td><td rowspan=1 colspan=1>.4410±.0001</td><td rowspan=1 colspan=1>.8091±.0005</td><td rowspan=1 colspan=1>.4427±.0007</td></tr><tr><td rowspan=1 colspan=1>AutoDim</td><td rowspan=1 colspan=2>.8085±.0009 .4429±.0009</td><td rowspan=1 colspan=1>.8083±.0008</td><td rowspan=1 colspan=1>.4431±.0008</td><td rowspan=1 colspan=1>.8078±.0012</td><td rowspan=1 colspan=1>.4436±.0012</td><td rowspan=1 colspan=1>.8086±.0010</td><td rowspan=1 colspan=1>.4431±.0011</td><td rowspan=1 colspan=1>.8084±.0011</td><td rowspan=1 colspan=1>.4431±.0009</td></tr><tr><td rowspan=1 colspan=1>ES</td><td rowspan=1 colspan=2>.8107±.0001 .4410±.0001</td><td rowspan=1 colspan=1>.8105±.0002</td><td rowspan=1 colspan=1>.4411±.0002</td><td rowspan=1 colspan=1>.8108±.0001</td><td rowspan=1 colspan=1>.4409±.0001</td><td rowspan=1 colspan=1>.8106±.0001</td><td rowspan=1 colspan=1>.4411±.0002</td><td rowspan=1 colspan=1>.8107±.0001</td><td rowspan=1 colspan=1>.4411±.0001</td></tr><tr><td rowspan=1 colspan=1>AutoDenoise</td><td rowspan=1 colspan=1>.8104±.0002</td><td rowspan=1 colspan=1>.4415±.0002</td><td rowspan=1 colspan=1>.8104±.0002</td><td rowspan=1 colspan=1>.4417±.0001</td><td rowspan=1 colspan=1>.8103±.0002</td><td rowspan=1 colspan=1>.4418±.0003</td><td rowspan=1 colspan=1>.8101±.0001</td><td rowspan=1 colspan=1>.4420±.0001</td><td rowspan=1 colspan=1>.8104±.0002</td><td rowspan=1 colspan=1>.4415±.0001</td></tr><tr><td rowspan=1 colspan=1>DS</td><td rowspan=1 colspan=1>.8107±.0001</td><td rowspan=1 colspan=1>.4410±.0001</td><td rowspan=1 colspan=1>.8104±.0001</td><td rowspan=1 colspan=1>.4413±.0001</td><td rowspan=1 colspan=1>.8106±.0001</td><td rowspan=1 colspan=1>.4412±.0001</td><td rowspan=1 colspan=1>.8104±.0001</td><td rowspan=1 colspan=1>.4412±.0001</td><td rowspan=1 colspan=1>.8106±.0000</td><td rowspan=1 colspan=1>.4411±.0001</td></tr><tr><td rowspan=2 colspan=1>IDCOptIDC</td><td rowspan=1 colspan=2>.8110±.0001 .4408±.0001</td><td rowspan=1 colspan=1>.8110±.0001</td><td rowspan=1 colspan=1>.4407±.0001</td><td rowspan=1 colspan=1>.8110±.0001</td><td rowspan=1 colspan=1>.4408±.0003</td><td rowspan=1 colspan=1>.8110±.0001</td><td rowspan=1 colspan=1>.4407±.0001</td><td rowspan=1 colspan=1>.8110±.0001</td><td rowspan=1 colspan=1>.4407±.0002</td></tr><tr><td rowspan=1 colspan=3>.8113*±.0001.4404*±.0001..8114*±.0001.</td><td rowspan=1 colspan=1>4404*±.0001</td><td rowspan=1 colspan=1>.8114*±.0001.</td><td rowspan=1 colspan=1>4404*±.0001.</td><td rowspan=1 colspan=1>.8113*±.0001</td><td rowspan=1 colspan=1>.4404*±.0001</td><td rowspan=1 colspan=1>.8115*±.0001.</td><td rowspan=1 colspan=1>.4402*±.0001</td></tr></table>

Table 2: Training and inference time on the Avazu dataset.

## 5.4 RQ3: Eficiency Analysis

In this subsection, we evaluate the eficiency of OptIDC from the perspectives of model size and computational cost, using DNN as the backbone model. As shown in Fig. 4, OptIDC has more parameters than Backbone, OptFS, PEP, and OptEmbed, while remaining comparable to AutoField, AutoDim, and AutoDenoise. Notably, unlike methods that focus on a single-level selection, OptIDC jointly models the three-level selection with a well-controlled parameter overhead and achieves improved performance. In terms ofeficiency, Table 2 reports total training time and per-batch training and inference time on the Avazu dataset. Total training time includes search and retraining where applicable. OptIDC requires 2575 seconds in total, less than OptEmbed and AutoDim but more than Backbone, OptFS, and AutoDenoise. Its per-batch training time is 0.1032 seconds, which is 4.3%–14.4% higher than OptEmbed, OptFS, and AutoDenoise, but lower than AutoDim. Its inference time is 0.0041 seconds per batch, comparable to these three selection methods, versus 0.0034 seconds for the Backbone. These results show that OptIDC supports joint three-level selection with moderate additional per-batch training cost relative to these single- and two-level selection methods.

<table><tr><td>Times (sec)</td><td>Backbone</td><td>OptEmbed</td><td>OptFS</td><td>AutoDim</td><td>AutoDenoise</td><td>OptIDC</td></tr><tr><td>Train (Total)</td><td>1069</td><td>3378</td><td>1301</td><td>2613</td><td>2481</td><td>2575</td></tr><tr><td>Train (Batch)</td><td>0.0847</td><td>0.0930</td><td>0.0902</td><td>0.1880</td><td>0.0989</td><td>0.1032</td></tr><tr><td>Inference</td><td>0.0034</td><td>0.0042</td><td>0.0040</td><td>0.0124</td><td>0.0040</td><td>0.0041</td></tr></table>

## 5.5 RQ4: In-depth Analysis of OptIDC

Next, we further analyze the role of OT coordination in OptIDC, focusing on the impact of diferent cost matrix constructions on OT coordination and the efectiveness of diferent coordination strategies. Additionally, selection ratio and hyperparameter sensitivity analyses are provided in Appendix C.4 and Appendix C.5.

Table 3: Ablation analysis of our method on the Avazu dataset, with the best results marked in bold.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Method</td><td colspan="2">DNN</td><td colspan="2">DeepFM</td><td colspan="2">DCN</td><td colspan="2">AutoInt</td><td colspan="2">FinalMLP</td></tr><tr><td>AUC↑</td><td>LogLoss↓</td><td>AUC↑</td><td>LogLoss↓</td><td>AUC↑</td><td>LogLoss↓</td><td>AUC↑</td><td>LogLoss↓</td><td>AUC↑</td><td>LogLoss↓</td></tr><tr><td rowspan="10">Avazu</td><td>AOP</td><td>0.7912</td><td>0.3731</td><td>0.7916</td><td>0.3731</td><td>0.7911</td><td>0.3735</td><td>0.7911</td><td>0.3734</td><td>0.7920</td><td>0.3727</td></tr><tr><td>DSFS</td><td>0.7911</td><td>0.3736</td><td>0.7917</td><td>0.3732</td><td>0.7914</td><td>0.3734</td><td>0.7911</td><td>0.3737</td><td>0.7925</td><td>0.3728</td></tr><tr><td>DSFS+OT</td><td>0.7915</td><td>0.3733</td><td>0.7919</td><td>0.3731</td><td>0.7921</td><td>0.3728</td><td>0.7916</td><td>0.3734</td><td>0.7926</td><td>0.3725</td></tr><tr><td>DSES</td><td>0.7913</td><td>0.3738</td><td>0.7920</td><td>0.3729</td><td>0.7912</td><td>0.3734</td><td>0.7913</td><td>0.3732</td><td>0.7920</td><td>0.3731</td></tr><tr><td>DSES+OT</td><td>0.7915</td><td>0.3738</td><td>0.7921</td><td>0.3729</td><td>0.7917</td><td>0.3732</td><td>0.7923</td><td>0.3727</td><td>0.7921</td><td>0.3729</td></tr><tr><td>FSES</td><td>0.7910</td><td>0.3733</td><td>0.7921</td><td>0.3727</td><td>0.7912</td><td>0.3732</td><td>0.7913</td><td>0.3733</td><td>0.7922</td><td>0.3726</td></tr><tr><td>FSES+OT</td><td>0.7920</td><td>0.3727</td><td>0.7928</td><td>0.3722</td><td>0.7920</td><td>0.3727</td><td>0.7928</td><td>0.3721</td><td>0.7938</td><td>0.3720</td></tr><tr><td>IDC</td><td>0.7915</td><td>0.3732 0.3727</td><td>0.7918</td><td>0.3729</td><td>0.7914</td><td>0.3734</td><td>0.7915</td><td>0.3731</td><td>0.7925</td><td>0.3728</td></tr><tr><td> $\mathrm { w } / \mathrm { o } \mathrm { \mathcal { W } } _ { \mathrm { s f } }$ </td><td>0.7924 0.7929</td><td>0.3723</td><td>0.7928 0.7932</td><td>0.3726 0.3722</td><td>0.7924 0.7929</td><td>0.3729 0.3726</td><td>0.7927 0.7930</td><td>0.3728</td><td>0.7932</td><td>0.3723</td></tr><tr><td> $\mathrm { w } / \mathrm { o } \mathrm { \mathcal { W } } _ { \mathrm { s e } }$ </td><td>0.7926</td><td>0.3727</td><td>0.7930</td><td>0.3724</td><td>0.7925</td><td>0.3730</td><td></td><td>0.3725</td><td>0.7936</td><td>0.3722</td></tr><tr><td> ${ \bf w } / { \bf o } \mathrm { \mathcal { W } _ { f e } }$ </td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.7927</td><td>0.3723</td><td>0.7935</td><td>0.3722</td></tr><tr><td></td><td>OptIDC</td><td>0.7930</td><td>0.3722</td><td>0.7933</td><td>0.3721</td><td>0.7931</td><td>0.3721</td><td>0.7934</td><td>0.3721</td><td>0.7941</td><td>0.3719</td></tr></table>

![](images/155e928c6fbc794de401d201646b67fe0a6b138c2c54a115a2344bdab5e3769f.jpg)

![](images/dbefef93897b7ea63639b22301d61d16e4f008f3a3bcf751e452d5fb86781127.jpg)  
Figure 4: Model size of OptIDC and baselines on four datasets.

Impact of Diferent Cost Matrices. To examine how cost matrix design afects OT coordination, we compare several cost constructions on the Avazu dataset, including a variant without OT coordination (‘IDC’), cost matrices based on weights Euclidean distance (‘w WED’) and embedding Euclidean distance (‘w EED’), and the full OptIDC. Results in Fig. 5a show that both Euclidean distance-based costs yield limited gains, performing similarly to IDC. OptIDC achieves better performance, supporting the efectiveness of its cosine-based ground cost under the evaluated settings.

![](images/70ebbc6e354871f7b3ac9e5079637dbeac0d46c7fcfe27bcd8ec6c765939b8ac.jpg)  
(a) Impact of diferent cost matrix calculation methods.

![](images/e8ecaa9c311dc4f587dbf4abdff2ea6055b29d6d6632af68c3eec7efe68b7746.jpg)  
(b) Efects of diferent coordination methods.

Figure 5: Analysis of cost matrix design in OT coordination and diferent coordination methods on the Avazu dataset.

Efects of diferent coordination methods. To investigate the impact of diferent coordination methods on model performance, we compare several coordination strategies on the Avazu dataset, including a variant without coordination (‘IDC’), a cosine similaritybased coordination method (‘w CS’), a symmetric KL divergencebased coordination method (‘w KL’), and our proposed OptIDC. Definitions, normalization schemes, and tuning details of all alternatives are provided in Appendix C.3. The results are presented in Fig. 5b. As shown, both the cosine similarity-based and KL divergence-based coordination methods yield only marginal improvements, achieving performance comparable to that of IDC without any coordination methods. This suggests that directly matching selection weights provides limited benefits under the evaluated settings. In contrast, OptIDC performs better across all five backbones, supporting the mass-preserving coarse-to-fine formulation used by our method. This comparison does not imply that OT discovers an unconstrained correspondence: the relation supports are fixed by the three configuration levels, and OT provides a common way to weight their compatibility.

## 5.6 RQ5: Selection Consistency Analysis

In this subsection, we evaluate OptIDC’s ability to efectively mitigate selection conflicts and thereby improve selection consistency. Due to the inherent diferences in selection behaviors across diferent input-side levels, joint modeling can lead to selection conflicts, most directly reflected in discrepancies among the selection masks. Here, we use SCR from Eq. (1) as the selection consistency metric. Since the selection masks at diferent levels operate at diferent granularities, they cannot be compared directly. Therefore, they are first mapped into a common space before the consistency is computed. The detailed mapping process is provided in Appendix A. As shown in Fig. 6, after introducing OT coordination, OptIDC has higher SCR than IDC for every backbone on both Avazu and Criteo. The result indicates greater agreement among the learned sample, feature, and embedding-unit configurations, which is consistent with the purpose of explicit cross-level coordination. We interpret it as supporting evidence rather than a causal explanation of the AUC gain, because SCR itself is not optimized by OptIDC.

![](images/2a29c380df066fbbdea4197fc93c471d630fc7abe9ab809beadea8f85322b707.jpg)

![](images/85ae2ba92800b5c680713ca6ec62fb2bd23bc37ba55b88274bfe6d7447126a64.jpg)  
Figure 6: Selection consistency evaluation of OptIDC on the Avazu and Criteo datasets.

## 6 Conclusions

In this paper, we investigate the input-side information configuration problem in deep CTR prediction and propose OptIDC, a unified framework for Optimal Input-side Dimensional Configuration. OptIDC jointly models the selection mechanisms of samples, features, and embedding units, and introduces an Optimal Transport-based coordination strategy to coordinate information configuration across diferent levels. Extensive experiments on four real-world datasets demonstrate that OptIDC can more efectively identify critical information and improve performance.

## References

[1] Yihong Chen, Bei Chen, Xiangnan He, Chen Gao, Yong Li, Jian-Guang Lou, and Yue Wang. 2019. �opt: Learn to regularize recommender models in finer levels. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. 978–986.

[2] Heng-Tze Cheng, Levent Koc, Jeremiah Harmsen, Tal Shaked, Tushar Chandra, Hrishi Aradhye, Glen Anderson, Greg Corrado, Wei Chai, Mustafa Ispir, et al. 2016. Wide & deep learning for recommender systems. In Proceedings of the 1st Workshop on Deep Learning for Recommender Systems. 7–10.

[3] Jaemoo Choi, Jaewoong Choi, and Dohyun Kwon. 2025. Overcoming spurious solutions in semi-dual neural optimal transport: A smoothing approach for learning the optimal transport plan. arXiv preprint arXiv:2502.04583 (2025).

[4] Marco Cuturi. 2013. Sinkhorn distances: Lightspeed computation of optimal transport. In Proceedings ofthe 27th International Conference on Neural Information Processing Systems.

[5] Jingtong Gao, Xiangyu Zhao, Bo Chen, Fan Yan, Huifeng Guo, and Ruiming Tang. 2023. AutoTransfer: Instance transfer for cross-domain recommendations. In Proceedings ofthe 46th International ACM SIGIR Conference on Research and Development in Information Retrieval. 1478–1487.

[6] Zheng Ge, Songtao Liu, Zeming Li, Osamu Yoshie, and Jian Sun. 2021. Ota: Optimal transport assignment for object detection. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition. 303–312

[7] Huifeng Guo, Ruiming Tang, Yunming Ye, Zhenguo Li, and Xiuqiang He. 2017. DeepFM: A factorization-machine based neural network for CTR prediction. In Proceedings ofthe 26th International Joint Conference on Artificial Intelligence. 1725–1731.

[8] Yi Guo, Zhaocheng Liu, Jianchao Tan, Chao Liao, Sen Yang, Lei Yuan, Dongying Kong, Zhi Chen, and Ji Liu. 2022. LPFS: Learnable polarizing feature selection for click-through rate prediction. arXiv preprint arXiv:2206.00267 (2022).

[9] Pengyue Jia, Yejing Wang, Zhaocheng Du, Xiangyu Zhao, Yichao Wang, Bo Chen, Wanyu Wang, Huifeng Guo, and Ruiming Tang. 2024. Erase: Benchmarking feature selection methods for deep recommender systems. In Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 5194–5205.

[10] Weilin Lin, Xiangyu Zhao, Yejing Wang, Tong Xu, and Xian Wu. 2022. AdaFS: Adaptive feature selection in deep recommender system. In Proceedings ofthe 28th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. 3309–3317.

[11] Weilin Lin, Xiangyu Zhao, Yejing Wang, Yuanshao Zhu, and Wanyu Wang. 2023. Autodenoise: Automatic data instance denoising for recommendations. In Proceedings ofthe ACM Web Conference 2023. 1003–1011.

[12] Bin Liu, Chenxu Zhu, Guilin Li, Weinan Zhang, Jincai Lai, Ruiming Tang, Xiuqiang He, Zhenguo Li, and Yong Yu. 2020. Autofis: Automatic feature interaction selection in factorization models for click-through rate prediction. In proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery and

Data Mining. 2636–2645.

[13] Dugang Liu, Chaohua Yang, Yuwen Fu, Xing Tang, Gongfu Li, Fuyuan Lyu, Xiuqiang He, and Zhong Ming. 2025. Scenario shared instance modeling fo click-through rate prediction. In Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 2436–2447.

[14] Dugang Liu, Chaohua Yang, Xing Tang, Yejing Wang, Fuyuan Lyu, Weihong Luo, Xiuqiang He, Zhong Ming, and Xiangyu Zhao. 2024. MultiFS: Automated multi-scenario feature selection in deep recommender systems. In Proceedings of the 17th ACM International Conference on Web Search and Data Mining. 434–442.

[15] Haochen Liu, Xiangyu Zhao, Chong Wang, Xiaobing Liu, and Jiliang Tang. 2020. Automated embedding size search in deep recommender systems. In Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval. 2307–2316.

[16] Siyi Liu, Chen Gao, Yihong Chen, Depeng Jin, and Yong Li. 2021. Learnable embedding sizes for recommender systems. In Proceedings of the 9th International Conference on Learning Representations.

[17] Fuyuan Lyu, Xing Tang, Dugang Liu, Liang Chen, Xiuqiang He, and Xue Liu. 2023. Optimizing feature set for click-through rate prediction. In Proceedings of the ACM Web Conference 2023. 3386–3395.

[18] Fuyuan Lyu, Xing Tang, Dugang Liu, Chen Ma, Weihong Luo, Liang Chen, Xiuqiang He, and Xue Liu. 2023. Towards hybrid-grained feature interaction selection for deep sparse network. In Proceedings ofthe 37th Conference on Neural Information Processing Systems. 49325–49340.

[19] Fuyuan Lyu, Xing Tang, Hong Zhu, Huifeng Guo, Yingxue Zhang, Ruiming Tang, and Xue Liu. 2022. OptEmbed: Learning optimal embedding table for clickthrough rate prediction. In Proceedings ofthe 31st ACM International Conference on Information and Knowledge Management. 1399–1409.

[20] Jiaqi Ma, Zhe Zhao, Xinyang Yi, Jilin Chen, Lichan Hong, and Ed H Chi. 2018. Modeling task relationships in multi-task learning with multi-gate mixture-of experts. In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. 1930–1939.

[21] Kelong Mao, Jieming Zhu, Liangcai Su, Guohao Cai, Yuru Li, and Zhenhua Dong. 2023. FinalMLP: An enhanced two-stream MLP model for CTR prediction. In Proceedings ofthe AAAI Conference on Artificial Intelligence. 4552–4560.

[22] Stefen Rendle. 2010. Factorization machines. In Proceedings of the 2010 IEEE International Conference on Data Mining. 995–1000.

[23] Weiping Song, Chence Shi, Zhiping Xiao, Zhijian Duan, Yewen Xu, Ming Zhang, and Jian Tang. 2019. Autoint: Automatic feature interaction learning via selfattentive neural networks. In Proceedings ofthe 28th ACMInternational Conference on Information and Knowledge Management. 1161–1170.

[24] Hongyan Tang, Junning Liu, Ming Zhao, and Xudong Gong. 2020. Progressive layered extraction (ple): A novel multi-task learning (mtl) model for personalized recommendations. In Proceedings ofthe 14th ACM Conference on Recommender Systems. 269–278.

[25] Thilina Thanthriwatta and David S Rosenblum. 2021. Instance selection for online updating in dynamic recommender environments. In Proceedings ofthe 25th Pacific-Asia Conference on Knowledge Discovery and Data Mining. 612–624.

[26] Kefan Wang, Hao Wang, Kenan Song, Wei Guo, Kai Cheng, Zhi Li, Yong Liu, Defu Lian, and Enhong Chen. 2025. A universal framework for compressing embeddings in ctr prediction. In International Conference on Database Systems for Advanced Applications. 84–100.

[27] Ruoxi Wang, Bin Fu, Gang Fu, and Mingliang Wang. 2017. Deep & cross network for ad click predictions. In Proceedings ofthe ADKDD 2017. 1–7.

[28] Yejing Wang, Xiangyu Zhao, Tong Xu, and Xian Wu. 2022. Autofield: Automating feature selection in deep recommender systems. In Proceedings ofthe ACM Web Conference 2022. 1977–1986.

[29] Mingjia Yin, Junwei Pan, Hao Wang, Ximei Wang, Shangyu Zhang, Jie Jiang, Defu Lian, and Enhong Chen. 2025. From feature interaction to feature generation: A generative paradigm of CTR prediction models. In International Conference on Machine Learning. 72393–72413.

[30] Shuai Zhang, Lina Yao, Aixin Sun, and Yi Tay. 2019. Deep learning based recommender system: A survey and new perspectives. Comput. Surveys 52, 1 (2019), 1–38.

[31] Xiangyu Zhao, Haochen Liu, Wenqi Fan, Hui Liu, Jiliang Tang, and Chong Wang. 2021. AutoLoss: Automated loss function search in recommendations. In Proceedings of the 27th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. 3959–3967.

[32] Xiangyu Zhao, Haochen Liu, Hui Liu, Jiliang Tang, Weiwei Guo, Jun Shi, Sida Wang, Huiji Gao, and Bo Long. 2021. Autodim: Field-aware embedding dimension search in recommender systems. In Proceedings ofthe Web Conference 2021. 3015– 3022.

[33] Ruiqi Zheng, Liang Qu, Bin Cui, Yuhui Shi, and Hongzhi Yin. 2023. Automl for deep recommender systems: A survey. ACM Transactions on Information Systems 41, 4 (2023), 1–38.

## A Mask Mapping Mechanism

In this appendix, ${ \bf G } \in \{ 0 , 1 \} ^ { n }$ and $\mathbf { M } \in \{ 0 , 1 \} ^ { n \times d }$ denote the valueindexed hard feature and embedding-unit mask tables obtained after search, respectively, where $n = \vert \mathcal { V } \vert$

Feature frequency statistics. Given the training set D, for each feature $v _ { j } , j \in \{ 1 , 2 , \ldots , n \}$ , we compute its occurrence frequency in the training set as:

$$
\mathbf { F } _ { j } ^ { \mathrm { o r i } } = \sum _ { i = 1 } ^ { | \mathcal { D } | } \mathbb { I } ( v _ { j } \in \mathbf { x } _ { i } ) ,\tag{18}
$$

where I(·) is the indicator function, $\mathbf { F } ^ { \mathrm { { o r i } } }$ is the feature frequency vector for the original dataset D. Moreover, to compute feature frequencies for selected samples, we first construct the selected sample set $\mathcal { \widehat { D } } = \{ ( \mathbf { x } _ { i } , y _ { i } ) \ | \ \widehat { \ s _ { i } } = 1 , ( \mathbf { x } _ { i } , y _ { i } ) \in \mathcal { D } \}$ , where bs is the sample mask and |D | b is the number of selected samples. Similarly, the frequency of the �-th feature in the selected sample set $\hat { \mathcal D }$ is:

$$
\mathbf { F } _ { j } ^ { \mathrm { s e l } } = \sum _ { i = 1 } ^ { | \widehat { \mathcal { D } } | } \mathbb { I } ( v _ { j } \in \mathbf { x } _ { i } ) .\tag{19}
$$

Sample Mask Mapping. Sample selection itself does not directly operate on embeddings. Instead, it changes the empirical feature distribution of the retained samples. Since embeddings are updated only when their corresponding features appear, the feature occurrence frequency can be regarded as the activation frequency of embedding units. Thus, to quantify the impact of sample selection on embedding learning, we remap the sample selection mask s to an embedding-unit activation matrix M<sup>s</sup>. This representation facilitates a direct comparison of information selection with feature selection and embedding-unit selection. Accordingly, M<sup>s</sup> can be expressed as follows:

$$
\mathbf { M } ^ { s } = \log ( \mathbf { F } ^ { s \mathrm { e l } } + 1 ) \mathbf { 1 } _ { d } ^ { \top } , \quad \mathbf { M } ^ { s } \in \mathbb { R } ^ { n \times d } ,\tag{20}
$$

where the log(� + 1) operation on $\mathbf { F } ^ { \mathrm { s e l } } \in \mathbb { R } ^ { n }$ is applied to reduce the influence of high-frequency features $[ 2 6 ]$ , which is also a commonly used practice in industrial CTR systems. And $1 _ { d } ~ \in ~ \{ 1 \} ^ { d }$ is an all-one vector of size �. This broadcasting operation projects the feature-level statistics into the embedding-unit space for unified comparison.

Feature Mask Mapping. Similarly, to further quantify the impact of feature selection on embedding learning, we remap the feature selection mask G to an embedding-unit activation matrix ${ \bf { M } } ^ { \mathrm { { f } } }$ , which can be denoted as:

$$
\mathbf { w } = \log ( \mathbf { F } ^ { \mathrm { o r i } } + 1 ) ,\tag{21}
$$

$$
\mathbf { M } ^ { \mathrm { f } } = ( \mathbf { w } \odot \mathbf { G } ) \mathbf { 1 } _ { d } ^ { \top } , \quad \mathbf { M } ^ { \mathrm { f } } \in \mathbb { R } ^ { n \times d } ,\tag{22}
$$

where the $\log ( x + 1 )$ operation on $\mathbf { F } ^ { \mathrm { o r i } } \in \mathbb { R } ^ { n }$ is applied to reduce the influence of high-frequency features, ⊙ is element-wise multiplication, and $\mathbf { 1 } _ { d } \in \{ 1 \} ^ { d }$ is an all-one vector of size �.

Embedding-unit Mask Mapping. To quantify the impact of embedding-unit selection on embedding learning, we further consider the activation frequency of each embedding unit. Thus, the embedding-unit selection mask M is remapped to an embedding unit activation matrix ${ \bf { M } } ^ { \mathrm { { e } } }$ , which can be formulated as follows:

$$
\mathbf { M } ^ { \mathrm { e } } = ( \mathbf { w } \mathbf { 1 } _ { d } ^ { \top } ) \odot \mathbf { M } , \quad \mathbf { M } ^ { \mathrm { e } } \in \mathbb { R } ^ { n \times d } .\tag{23}
$$

Table 4: Dataset statistics.
<table><tr><td>Dataset</td><td>Avazu</td><td>Criteo</td><td>ML1M</td><td>AliCCP</td></tr><tr><td>#Train</td><td>32,343,174</td><td>36,672,494</td><td>800,167</td><td>38,070,670</td></tr><tr><td>#Validation</td><td>4,042,897</td><td>4,584,061</td><td>100,021</td><td>4,229,235</td></tr><tr><td>#Test</td><td>4,042,896</td><td>4,584,062</td><td>100,021</td><td>43,016,614</td></tr><tr><td>#Field</td><td>24</td><td>39</td><td>7</td><td>18</td></tr><tr><td>#Feature</td><td>4,428,306</td><td>6,780,406</td><td>13,240</td><td>1,283,107</td></tr></table>

## B Experimental Settings

## B.1 Dataset Preprocessing and Statistics

For Avazu and Criteo, following prior work [19], we filter lowfrequency features using a threshold of 2. In Avazu, the instance ID field is removed, and the timestamp field is transformed into three fields: hour, weekday, and is\_weekend. Both datasets are then randomly split into training, validation, and test sets with an 8:1:1 ratio. For MovieLens-1M, since the original dataset only contains ratings from 1 to 5, we convert it into a binary classification task by setting a rating threshold of 4, where ������ ≥ 4 indicates a click. $_ { \mathrm { A } s }$ the dataset contains fewer non-ID features, no low-frequency filtering is applied, and the data are also randomly divided into training, validation, and test sets in an 8:1:1 ratio. For AliCCP, following previous studies [13, 14], we filter low-frequency features with a threshold of 10. We then randomly sampled 10% of the original training set as the validation set and used the original test set for testing. Table 4 presents the statistics of the processed datasets.

## B.2 Implementation Details

In this subsection, we describe the implementation details of OptIDC and all baseline methods. For shared hyperparameter settings, the embedding size and batch size are fixed to 16 and 4096, respectively. The learning rate is selected from {1e-3, 3e-4, 1e-4, 3e-5, 1e-5} and the �<sub>2</sub> regularization from {3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6}. For the MLP components of the backbone models, we adopt a threelayer fully connected architecture with hidden sizes [1024, 512, 256]. All models are trained using the Adam optimizer with Xavier initialization. For the hyperparameters of OptIDC, we select the optimal regularization penalty $\lambda _ { \mathrm { { r a t } } } , \lambda _ { \mathrm { { e n t } } } , \lambda _ { \mathrm { { f } } } , \lambda _ { \mathrm { { e } } }$ and final value � from {5, 10, 50, 100, 500},{0.01, 0.05, 0.1, 0.5, 1.0}, {1e-6, 1e-7, 1e-8, 1e-9, 1e-10}, {1e-7, 1e-8, 1e-9, 1e-10, 1e-11} and {100, 200, 500, 1000}, respectively, temperature coeficient � is set to 0.1, and select the optimal OT penalty $\lambda _ { \mathrm { c o } }$ to {1e-3, 1e-4, 1e-5, 1e-6, 1e-7}. The expected sample-retention ratio is $\rho = 0 . 9 8$ . During the retraining phase, the optimal learning rate and $l _ { 2 }$ regularization can be re-selected, and the rewinding epoch $T _ { c }$ is chosen from $\{ 1 , 2 , \cdots , T - 1 \}$ . For baseline methods, we directly use the publicly available implementations of AutoField [28], LPFS [8], OptFS [17], PEP [16], OptEmbed [19], and AutoDenoise [11]. Since no oficial implementation ofAutoDim [32] is available, we re-implement it following the specifications in the original paper. For FinalMLP [21], we follow its open-source implementation but disable its built-in feature selection module to isolate the efects of the input-side feature selection methods under comparison, while retaining its two-stream MLP architecture and interaction aggregation module. This backbone variant is used consistently for ‘Backbone’, all baseline models, and OptIDC. Note that all baselines are tuned for optimal parameter settings within the same hyperparameter range. Each experiment is independently repeated with five diferent random seeds. Tables 1 and 5 report the mean ± standard deviation across the five runs, while the remaining experimental results are reported as means.

## C More Experimental Results

## C.1 Performance Comparison

This section evaluates OptIDC against multiple baseline methods spanning three types of input-side information selection methods: sample selection, feature selection, and embedding configuration. In Table 1 and Table 5, we report the overall performance of OptIDC and other baseline methods across four benchmark datasets, evaluated with five diferent backbone models. Based on the experimental results, we make the following observations: 1) Several strong single-level methods (e.g., OptFS, LPFS, ES, and DS) generally outperform the Backbone, confirming that input-side configuration is useful in its own right. The improvement is not uniform, however: a method that works well for one backbone or dataset may ofer little benefit for another. This variation is consistent with our premise that samples, feature values, and embedding units expose diferent kinds of redundancy. 2) IDC is generally stronger than its singlelevel counterparts, showing that the three decisions provide complementary information even when they are connected only by the prediction objective. 3) By contrast, our proposed OptIDC achieves significant AUC improvements over IDC on all datasets, with a clearer margin on Avazu. This result highlights the importance of introducing the cross-level coordination mechanism, thereby learning a more efective input-side configuration. On AliCCP, however, OptIDC achieves the highest AUC across all five backbones but does not achieve the lowest LogLoss, suggesting that its benefits are more consistently reflected in AUC.

## C.2 Ablation Study

In this subsection, we provide additional ablation results on the Criteo dataset in Table 6, using the same variants and the AOP baseline defined in Section 5.3. Joint configuration generally improves upon single-level selection, although IDC does not consistently outperform the two-level variants. Adding OT coordination to the two-level variants improves or maintains both metrics across all five backbones at the reported precision. For the three-level config uration, OptIDC improves both AUC and LogLoss over IDC across all five backbones. Removing any individual OT term lowers AUC across all five backbones, while LogLoss increases or remains unchanged. OptIDC also outperforms AOP in both metrics across all five backbones. These results further support the efectiveness of joint configuration and the contribution of each cross-level coordination relation.

## C.3 Details of Coordination Alternatives

We detail the alternatives in Section 5.5. All alternatives sum the three relation penalties and average over samples as in Eq. (4). For each alternative, $\lambda _ { \mathrm { c o } }$ is independently tuned over the same range as OptIDC (Section B.2), with the remaining objective terms unchanged.

w WED and w EED. These variants replace only the ground costs in Eq. (13), retaining its selection masses, temperature-normalized allocations, and averaging factors. For w WED, the costs are distances between scalar selection weights:

$$
\begin{array} { r l } { c _ { i j } ^ { ( \mathrm { W E D , s f } ) } = | s _ { i } - g _ { i j } | , ~ c _ { i j k } ^ { ( \mathrm { W E D , f e } ) } } & { { } = | g _ { i j } - M _ { i j k } | , } \\ { c _ { i j k } ^ { ( \mathrm { W E D , s e } ) } = | s _ { i } - M _ { i j k } | . } \end{array}\tag{24}
$$

For w EED, the cost uses the same representation pairs as OptIDC, with non-squared Euclidean distance after unit-L2 normalization:

$$
c _ { \mathrm { E E D } } ( \mathbf { a } , \mathbf { b } ) = \left\| { \frac { \mathbf { a } } { \| \mathbf { a } \| _ { 2 } } } - { \frac { \mathbf { b } } { \| \mathbf { b } \| _ { 2 } } } \right\| _ { 2 } ,\tag{25}
$$

where a and b denote the nonzero representations being compared. The representation pairs are $( \overline { { \mathbf { h } } } _ { i } , \mathbf { h } _ { i j } ) , ( \mathbf { h } _ { i j } , \mathbf { u } _ { j k } )$ , and $( \overline { { \mathbf { h } } } _ { i } , \mathbf { u } _ { j k } )$ for the sample–feature, feature–unit, and sample–unit relations, respectively.

w CS and w KL. For $r \in \{ \mathrm { s f } , \mathrm { f e } , \mathrm { s e } \}$ , these variants replace W<sub>�</sub> (�) with a direct comparison of the broadcast selection vectors:

$$
\begin{array} { r l } & { ( \mathbf { a } _ { i } ^ { \mathrm { s f } } , \mathbf { b } _ { i } ^ { \mathrm { s f } } ) = ( s _ { i } \mathbf { 1 } _ { F } , \mathbf { g } _ { i } ) , } \\ & { ( \mathbf { a } _ { i } ^ { \mathrm { f e } } , \mathbf { b } _ { i } ^ { \mathrm { f e } } ) = ( \mathrm { v e c } ( \mathbf { g } _ { i } \mathbf { 1 } _ { d } ^ { \top } ) , \mathrm { v e c } ( \mathbf { M } _ { i } ) ) , } \\ & { ( \mathbf { a } _ { i } ^ { \mathrm { s e } } , \mathbf { b } _ { i } ^ { \mathrm { s e } } ) = ( s _ { i } \mathbf { 1 } _ { F d } , \mathrm { v e c } ( \mathbf { M } _ { i } ) ) . } \end{array}\tag{26}
$$

Then, w CS uses the following calculation:

$$
\mathcal { W } _ { r } ^ { ( \mathrm { C S } ) } ( i ) = \omega _ { i } ^ { r } \left[ 1 - \frac { ( { \mathbf a } _ { i } ^ { r } ) ^ { \top } { \mathbf b } _ { i } ^ { r } } { \operatorname* { m a x } ( \| { \mathbf a } _ { i } ^ { r } \| _ { 2 } , \varepsilon ) \operatorname* { m a x } ( \| { \mathbf b } _ { i } ^ { r } \| _ { 2 } , \varepsilon ) } \right] ,\tag{27}
$$

where $\varepsilon = 1 0 ^ { - 8 } , \omega _ { i } ^ { \mathrm { s f } } = \omega _ { i } ^ { \mathrm { s e } } = s _ { i }$ and $\omega _ { i } ^ { \mathrm { f e } } = 1$ . For w KL, normalize each complete vector z $\in \mathbb { R } _ { + } ^ { m }$ by:

$$
\mathcal { P } _ { \varepsilon } ( \mathbf { z } ) = \frac { \mathbf { z } + \varepsilon \mathbf { 1 } _ { m } } { \mathbf { 1 } _ { m } ^ { \top } \mathbf { z } + m \varepsilon } .\tag{28}
$$

Let $\mathbf { p } _ { i } ^ { r } = \mathcal { P } _ { \varepsilon } ( \mathbf { a } _ { i } ^ { r } )$ and $\mathbf { q } _ { i } ^ { r } = \mathcal { P } _ { \varepsilon } ( \mathbf { b } _ { i } ^ { r } )$ . The symmetric KL penalty is:

$$
\mathcal { W } _ { r } ^ { ( \mathrm { K L } ) } \left( i \right) = \omega _ { i } ^ { r } \sum _ { l } ( \boldsymbol { p } _ { i l } ^ { r } - \boldsymbol { q } _ { i l } ^ { r } ) \log \frac { \boldsymbol { p } _ { i l } ^ { r } } { { q } _ { i l } ^ { r } } .\tag{29}
$$

## C.4 Selection Ratio Analysis

To verify that OptIDC can more efectively identify critical inputside information (e.g., high-value samples, informative features, and key embedding units) for achieving more efective input-side dimensional configuration, we compare the sample retention ratio and embedding parameter retention ratio (jointly determined by feature selection and embedding-unit selection) of OptIDC against baseline methods using three backbone models on the Avazu dataset, as shown in Fig. 7. The results show that, for both sample selection and embedding parameter selection, OptIDC consistently achieves superior predictive performance while generally retaining fewer samples and embedding parameters than most baselines. These findings indicate that OptIDC can more precisely identify predictionrelevant information, thereby enabling more eficient input-side dimensional configuration.

## C.5 Hyperparameter Sensitivity Analysis

Next, we conducted experiments on the Avazu dataset using three backbone models to investigate the sensitivity of five key hyperparameters: $\mathrm { ( i ) } \lambda _ { \mathrm { c o } } , \mathrm { ( i i ) } \lambda _ { \mathrm { r a t } } , \mathrm { ( i i i ) } \lambda _ { \mathrm { e n t } } , \mathrm { ( i v ) } \lambda _ { \mathrm { f } } ,$ , and (v) $\lambda _ { \mathrm { e } } .$ . As described in Section B.2, these hyperparameters are varied within their respective predefined ranges. The results are shown in Fig. 8, where the y-axis denotes the AUC score and the x-axis represents the search range of each hyperparameter. From Fig. 8, we observe that on the Avazu dataset, OptIDC exhibits broadly similar performance trends across the three backbone models. Performance remains relatively stable over parts of the tested ranges, while large feature- and embedding-unit sparsity penalties can substantially degrade AUC. These results suggest that appropriate regularization strengths are important for maintaining predictive performance across diferent backbones.

Table 5: Results on ML1M and AliCCP (mean ± standard deviation over five runs with diferent random seeds), where the best and second best results are marked in bold and underlined, respectively. Note that <sup>∗</sup> indicates a significance level of $\textstyle p \leq 0 . 0 5$ based on a two-sample t-test between our method and the best baseline.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Model</td><td colspan="2">DNN</td><td colspan="2">DeepFM</td><td colspan="2">DCN</td><td colspan="2">AutoInt</td><td colspan="2">FinalMLP</td></tr><tr><td>AUC↑</td><td>LogLoss↓</td><td>AUC↑</td><td>LogLoss↓</td><td>AUC↑</td><td>LogLoss↓</td><td>AUC↑</td><td>LogLoss↓</td><td>AUC↑</td><td>LogLoss↓</td></tr><tr><td rowspan="13">MM</td><td>Backbone AutoField</td><td>.8151±.0001</td><td>.5192±.0007</td><td>.8147±.0006</td><td>.5201±.0015</td><td>.8148±.0004</td><td>.5199±.0013</td><td>.8148±.0004</td><td>.5205±.0020</td><td>.8138±.0003</td><td>.5236±.0017</td></tr><tr><td></td><td>.8137±.0006</td><td>.5211±.0010</td><td>.8137±.0003</td><td>.5212±.0003</td><td>.8141±.0002</td><td>.5206±.0012</td><td>.8137±.0003</td><td>.5209±.0016</td><td>.8137±.0002</td><td>.5225±.0011</td></tr><tr><td>LPFS</td><td>.8154±.0006</td><td>.5181±.0004</td><td>.8158±.0005</td><td>.5194±.0014</td><td>.8157±.0004</td><td>.5177±.0006</td><td>.8153±.0007</td><td>.5184±.0010</td><td>.8136±.0008</td><td>.5227±.0007</td></tr><tr><td>OptEmbed</td><td>.8156±.0004</td><td>.5188±.0007</td><td>.8155±.0006</td><td>.5188±.0009</td><td>.8158±.0004</td><td>.5176±.0007</td><td>.8159±.0004</td><td>.5177±.0011</td><td>.8142±.0010</td><td>.5209±.0012</td></tr><tr><td>OptFS</td><td>.8161±.0002</td><td>.5178±.0004</td><td>.8163±.0003</td><td>.5173±.0004</td><td>.8164±.0001</td><td>.5171±.0003</td><td>.8161±.0004</td><td>.5173±.0007</td><td>.8156±.0003</td><td>.5194±.0013</td></tr><tr><td>PEP</td><td>.8068±.0012</td><td>.5296±.0046</td><td>.8064±.0010</td><td>.5287±.0034</td><td>.8070±.0007</td><td>.5269±.0018</td><td>.8066±.0007</td><td>.5282±.0026</td><td>.8055±.0008</td><td>.5288±.0007</td></tr><tr><td>AutoDim</td><td>.8033±.0011</td><td>.5297±.0021</td><td>.8039±.0010</td><td>.5308±.0011</td><td>.8039±.0015</td><td>.5314±.0029</td><td>.8043±.0011</td><td>.5300±.0014</td><td>.7955±.0038</td><td>.5435±.0019</td></tr><tr><td>ES</td><td>.8159±.0005</td><td>.5200±.0010</td><td>.8164±.0002</td><td>.5178±.0002</td><td>.8156±.0004</td><td>.5180±.0016</td><td>.8162±.0005</td><td>.5191±.0012</td><td>.8152±.0005</td><td>.5215±.0014</td></tr><tr><td>AutoDenoise</td><td>.8145±.0001</td><td>.5193±.0024</td><td>.8150±.0002</td><td>.5200±.0004</td><td>.8148±.0003</td><td>.5199±.0021</td><td>.8145±.0004</td><td>.5193±.0014</td><td>.8135±.0005</td><td>.5228±.0011</td></tr><tr><td>DS</td><td>.8145±.0003</td><td>.5171±.0003</td><td>.8159±.0005</td><td>.5160±.0010</td><td>.8149±.0007</td><td>.5167±.0008</td><td>.8149±.0004</td><td>.5163±.0012</td><td>.8153±.0005</td><td>.5214±.0031</td></tr><tr><td>IDC</td><td>.8174±.0002</td><td>.5143±.0004</td><td>.8177±.0004</td><td>.5141±.0004</td><td>.8176±.0002</td><td>.5148±.0009</td><td>.8174±.0003</td><td>.5148±.0009</td><td>.8164±.0003</td><td>.5192±.0018</td></tr><tr><td>OptIDC</td><td>.8181*±.0002</td><td>.5137*±.0001</td><td>.8185*±.0003</td><td>.5136±.0004</td><td>.8184*±.0004</td><td>.5136*±.0005</td><td>.8182*±.0003</td><td>.5142±.0007</td><td>.8172*±.0003</td><td>.5172±.0009</td></tr><tr><td rowspan="14">AICP</td><td>Backbone</td><td></td><td>.1616±.0000</td><td>.6241±.0003</td><td>.1618±.0001</td><td>.6238±.0005</td><td>.1615±.0000</td><td></td><td></td><td></td><td></td></tr><tr><td>AutoField</td><td>.6243±.0001 .6236±.0006</td><td>.1630±.0005</td><td>.6228±.0006</td><td>.1629±.0004</td><td>.6235±.0007</td><td>.1629±.0005</td><td>.6241±.0002 .6233±.0005</td><td>.1616±.0000 .1629±.0005</td><td>.6247±.0003</td><td>.1616±.0001</td></tr><tr><td>LPFS</td><td>.6244±.0004</td><td>.1623±.0007</td><td>.6247±.0003</td><td>.1623±.0003</td><td>.6249±.0003</td><td>.1619±.0003</td><td>.6237±.0005</td><td>.1628±.0002</td><td>.6236±.0004 .6240±.0008</td><td>.1625±.0004</td></tr><tr><td>OptEmbed</td><td>.6242±.0002</td><td>.1625±.0001</td><td>.6246±.0004</td><td>.1623±.0003</td><td>.6246±.0005</td><td>.1617±.0001</td><td>.6247±.0006</td><td>.1618±.0002</td><td>.6243±.0003</td><td>.1627±.0003</td></tr><tr><td>OptFS</td><td>.6245±.0005</td><td>.1625±.0002</td><td>.6247±.0003</td><td>.1622±.0001</td><td>.6243±.0006</td><td>.1621±.0002</td><td>.6240±.0005</td><td>.1626±.0003</td><td>.6246±.0003</td><td>.1622±.0002</td></tr><tr><td></td><td>.6233±.0004</td><td>.1615±.0002</td><td>.6232±.0007</td><td>.1616±.0001</td><td>.6240±.0004</td><td>.1613±.0001</td><td>.6237±.0008</td><td>.1615±.0001</td><td>.6229±.0002</td><td>.1620±.0001</td></tr><tr><td>PEP</td><td>.6183±.0018</td><td>.1628±.0007</td><td>.6186±.0018</td><td>.1627±.0014</td><td>.6183±.0049</td><td>.1625±.0007</td><td>.6191±.0014</td><td>.1631±.0008</td><td>.6193±.0015</td><td>.1616±.0001</td></tr><tr><td>AutoDim ES</td><td>.6246±.0004</td><td>.1622±.0004</td><td>.6241±.0004</td><td>.1620±.0001</td><td>.6240±.0007</td><td>.1618±.0003</td><td>.6242±.0005</td><td>.1618±.0000</td><td>.6240±.0006</td><td>.1624±.0001</td></tr><tr><td>AutoDenoise</td><td>.6196±.0039</td><td>.1643±.0018</td><td>.6199±.0018</td><td>.1631±.0003</td><td>.6214±.0021</td><td>.1637±.0011</td><td>.6212±.0027</td><td>.1641±.0019</td><td>.6207±.0019</td><td>.1620±.0001 .1635±.0005</td></tr><tr><td>DS</td><td>.6238±.0005</td><td>.1619±.0002</td><td>.6238±.0008</td><td>.1617±.0003</td><td>.6234±.0007</td><td>.1616±.0002</td><td>.6242±.0005</td><td>.1620±.0001</td><td>.6240±.0001</td><td>.1617±.0002</td></tr><tr><td>IDC</td><td>.6256±.0003</td><td>.1621±.0001</td><td>.6254±.0004</td><td>.1619±.0002</td><td>.6255±.0002</td><td>.1618±.0000</td><td>.6253±.0004</td><td>.1623±.0002</td><td>.6256±.0003</td><td>.1620±.0001</td></tr><tr><td>OptIDC</td><td>.6263*±.0002</td><td>.1619±.0001</td><td>.6262*±.0003</td><td>.1618±.0000</td><td>.6261*±.0002</td><td>.1616±.0001</td><td>.6264*±.0003</td><td>.1620±.0002</td><td>.6264*±.0002</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>.1618±.0001</td></tr></table>

![](images/0ab33a2c61a193e33026040627f8bcc3a9e998e259e1a59861a736c532caab69.jpg)  
Figure 7: Comparison of sample retention ratio and embedding retention ratio between OptIDC and baselines on the Avazu dataset.

![](images/22621a314dcfbdc8e09e2962ce86fe5af566702390d4c8bbfdb1a6209b27c5b7.jpg)

Table 6: Ablation analysis of our method on the Criteo dataset, with the best results marked in bold.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Method</td><td colspan="2">DNN</td><td colspan="2">DeepFM</td><td colspan="2">DCN</td><td colspan="2">AutoInt</td><td colspan="2">FinalMLP</td></tr><tr><td>AUC↑</td><td>LogLoss↓</td><td>AUC↑</td><td>LogLoss↓</td><td>AUC↑</td><td>LogLoss↓</td><td>AUC↑</td><td>LogLoss↓</td><td>AUC↑</td><td>LogLoss↓</td></tr><tr><td rowspan="10">Criteo</td><td>AOP</td><td>0.8106</td><td>0.4407</td><td>0.8104</td><td>0.4409</td><td>0.8107</td><td>0.4406</td><td>0.8103</td><td>0.4417</td><td>0.8108</td><td>0.4410</td></tr><tr><td>DSFS</td><td>0.8109</td><td>0.4408</td><td>0.8106</td><td>0.4410</td><td>0.8110</td><td>0.4406</td><td>0.8109</td><td>0.4409</td><td>0.8109</td><td>0.4408</td></tr><tr><td>DSFS+OT</td><td>0.8110</td><td>0.4406</td><td>0.8107</td><td>0.4409</td><td>0.8112</td><td>0.4405</td><td>0.8110</td><td>0.4408</td><td>0.8111</td><td>0.4405</td></tr><tr><td>DSES</td><td>0.8108</td><td>0.4407</td><td>0.8107</td><td>0.4409</td><td>0.8109</td><td>0.4408</td><td>0.8109</td><td>0.4406</td><td>0.8110</td><td>0.4408</td></tr><tr><td>DSES+OT</td><td>0.8109</td><td>0.4407</td><td>0.8108</td><td>0.4408</td><td>0.8110</td><td>0.4407</td><td>0.8110</td><td>0.4405</td><td>0.8112</td><td>0.4404</td></tr><tr><td>FSES</td><td>0.8110</td><td>0.4406</td><td>0.8110</td><td>0.4406</td><td>0.8110</td><td>0.4406</td><td>0.8109</td><td>0.4408</td><td>0.8109</td><td>0.4407</td></tr><tr><td>FSES+OT</td><td>0.8112</td><td>0.4404</td><td>0.8111</td><td>0.4405</td><td>0.8110</td><td>0.4406</td><td>0.8110</td><td>0.4406</td><td>0.8112</td><td>0.4403</td></tr><tr><td>IDC</td><td>0.8110 0.8111</td><td>0.4408 0.4405</td><td>0.8110</td><td>0.4407</td><td>0.8110</td><td>0.4408</td><td>0.8110</td><td>0.4407</td><td>0.8110</td><td>0.4407</td></tr><tr><td> $\mathrm { w } / \mathrm { o } \mathrm { \mathcal { W } } _ { \mathrm { s f } }$ </td><td>0.8111</td><td>0.4404</td><td>0.8112 0.8112</td><td>0.4406 0.4405</td><td>0.8112 0.8112</td><td>0.4404 0.4404</td><td>0.8111</td><td>0.4405</td><td>0.8114</td><td>0.4403</td></tr><tr><td> $\mathrm { w } / \mathrm { o } \mathrm { \mathcal { W } } _ { \mathrm { s e } }$ </td><td>0.8112</td><td>0.4406</td><td>0.8111</td><td>0.4406</td><td>0.8112</td><td>0.4405</td><td>0.8111</td><td>0.4406</td><td>0.8113</td><td>0.4403</td></tr><tr><td> ${ \bf w } / { \bf o } \mathrm { \mathcal { W } _ { f e } }$ </td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.8112</td><td>0.4404</td><td>0.8113</td><td>0.4403</td></tr><tr><td></td><td>OptIDC</td><td>0.8113</td><td>0.4404</td><td>0.8114</td><td>0.4404</td><td>0.8114</td><td>0.4404</td><td>0.8113</td><td>0.4404</td><td>0.8115</td><td>0.4402</td></tr></table>

![](images/d8fc6dba5bd37beebb6a747633a9d38ebbae74213d5702829b8a1bfbad9c6a45.jpg)

![](images/819ec09971172b8efd7a07820da3912739038b71f84325f36af0ebb2023192c4.jpg)

![](images/1c581c8f43ed7fe4f2bd2b19553dd82b72726c5bf04e2fa71eacc21ac557fa5d.jpg)  
Figure 8: Sensitivity analysis of key parameters is performed on the Avazu dataset.

![](images/0137de4e044cd5e485a5b5f8996024982eea6551eea4b484a18e7712f073c252.jpg)

![](images/308cb127366a1de37c5e86a79917ee50d3d8ac37696e140a34ca8259c7a97035.jpg)